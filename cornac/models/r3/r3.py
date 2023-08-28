import numpy as np
import torch
from gensim.models import Word2Vec, KeyedVectors
from torch import nn
from gensim import downloader
from dgl import function as fn

class TextProcessor(nn.Module):
    def __init__(self, vocab, num_heads=100, max_length=200, max_review=50, dropout=0.2):
        ""
        super().__init__()

        self.dim = 100
        self.num_heads = num_heads
        self.max_length = max_length
        self.max_review = max_review
        self.kernel_size = 3
        self.dropout = dropout

        # Init word2vec, try lazy-loading from file, otherwise download and save (slow)
        try:
            word2vec = KeyedVectors.load(f'glove-twitter-{self.dim}.bin', mmap='r')
        except FileNotFoundError:
            word2vec = downloader.load(f'glove-twitter-{self.dim}')#(f'word2vec-google-news-{self.dim}')
            word2vec.save(f'glove-twitter-{self.dim}.bin')

        # Create embedding matrix based on word vectors and vocabulary
        embeddings = torch.FloatTensor([word2vec[word] if word in word2vec else np.zeros((self.dim,))
                                        for word in vocab.idx2tok])
        self.embeddings = nn.Embedding.from_pretrained(embeddings, freeze=True, padding_idx=0)
        self.conv = nn.Conv2d(1, num_heads, (self.kernel_size, self.dim))
        self.relu = nn.ReLU()

    def forward(self, reviews):
        embs = self.embeddings(reviews)
        b, r, s, d = embs.size()

        # let s' = (s - self.kernel_size + 1)
        # eq 18 br x 1 x s x d -> br x num_heads x s' x 1
        conv = self.conv(embs.reshape(b*r, 1, s, d))  # Only one channel
        c = self.relu(conv)
        _, _, s_prime, _ = c.size()

        # eq 19, br x num_heads x s' x 1 -> b x r x s' x num_heads
        c = c.squeeze(-1).transpose(-1,-2).reshape(b, r, s_prime, self.num_heads)

        return c



class Rationale(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        pass


class Correlation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        pass


class Model(nn.Module):
    def __init__(self, train_set):
        super().__init__()

        # Todo make embedders for each node type
        self.embedder = TextProcessor(train_set.review_text.vocab)
        self.preference_dim = 64
        self.sigmoid = nn.Sigmoid()

        # Word-level rationale generator
        # fully connected layer, eq 2
        self.fc_w = nn.Linear(self.embedder.num_heads, 1)
        self.sigmoid = nn.Sigmoid()

        # Review-level rationale generator
        self.fc_w2 = nn.Linear(self.embedder.num_heads, self.embedder.dim)

        # rationale predictor
        self.h_r = nn.Linear(2*(self.embedder.max_length - self.embedder.kernel_size + 1), 1, bias=False)

        # Correlation predictor
        self.h_c = nn.Linear(2*self.embedder.dim, 1, bias=False)
        self.user_embedding = nn.Embedding(train_set.num_users, self.embedder.dim)
        self.item_embedding = nn.Embedding(train_set.num_items, self.embedder.dim)

        # user, item, and global bias
        self.user_bias = nn.Embedding(train_set.num_users, 1)
        self.item_bias = nn.Embedding(train_set.num_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))

        # Loss
        self.mse = nn.MSELoss()
        self.relu = nn.ReLU()
        self.gap = .20

        self.reset_parameters()

    def reset_parameters(self):
        for k, v in self.named_parameters():
            if 'bias' in k:
                nn.init.constant_(v, 0)
            else:
                nn.init.xavier_normal_(v)

    def preference_aggregator(self, g, x):
        with g.local_scope():
            g.ndata['x'] = x
            g.update_all(fn.copy_u('x', 'm'), fn.sum('m', 'x'))
            return g.ndata.pop('x')

    def predictor(self, g, P, z):
        with ((g.local_scope())):
            uid, iid = g.edges()

            # Continuation of 3.1.2, Review-level rationale generator.
            # eq 5
            s_uv = P['user'][uid].transpose(-1,-2) @ P['item'][iid]

            # eq 6
            rho_u_r = self.sigmoid(s_uv.sum(-1))
            rho_v_r = self.sigmoid(s_uv.sum(-2))

            # 3.2 Rationale Predictor
            # eq 7
            gamma_u_r = z['user'][uid].transpose(1, 2) @ P['user'][uid]
            gamma_v_r = z['item'][iid].transpose(1, 2) @ P['item'][iid]

            # eq 8, global bias in linear layer
            pred_uv_r = self.h_r(torch.cat([gamma_u_r, gamma_v_r], dim=1).squeeze(-1)) \
                        + self.user_bias(uid) + self.item_bias(iid) + self.global_bias

            # 3.3 Correlation Predictor
            # eq 9
            gamma_u_c = (rho_u_r.unsqueeze(1) @ P['user'][uid]).squeeze(-2)
            gamma_v_c = (rho_v_r.unsqueeze(1) @ P['item'][iid]).squeeze(-2)

            # eq 10
            pred_uv_c = self.h_c(torch.cat([gamma_u_c,# + self.user_embedding(uid),
                                            gamma_v_c# + self.item_embedding(iid)
                                            ], dim=-1))\
                        #+ self.user_bias(uid) + self.item_bias(iid) + self.global_bias

        return pred_uv_r, pred_uv_c

    def forward(self, input_nodes, g):
        pref = {}
        z = {}
        for ntype, nt_blocks in input_nodes.items():
            # Text processor, appendix A
            c_ = self.embedder(input_nodes[ntype])

            # 3.1.1 Word-level rationale generator
            rho_ = self.sigmoid(self.fc_w(c_))  # Eq 2
            z_ = rho_ + (torch.round(rho_) - rho_).detach()  # Eq 3
            z[ntype] = z_.squeeze(-1)

            # 3.1.2 Review-level rationale generator
            P_ = c_.max(-2)[0] # Eq 4
            P_ = self.fc_w2(P_)  # Described in between eq 4 and 5
            pref[ntype] = P_

        preds = self.predictor(g, pref, z)

        return preds, z

    def loss(self, input_nodes, preds_r, preds_c, labels, z):
        loss_r = self.mse(preds_r, labels) * 0
        loss_c = self.mse(preds_c, labels)
        loss_rc = self.relu(loss_r - loss_c) * 0

        # input_nodes = torch.cat([i for i in input_nodes.values()], dim=0)
        z = torch.cat([i for i in z.values()], dim=0)
        loss_reg = (z.sum() / z.numel()) - self.gap # select gap percent

        return loss_r, loss_c, loss_rc, loss_reg * 0

