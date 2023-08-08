import numpy as np
import torch
from torch import nn
from gensim import downloader

class TextProcessor(nn.Module):
    def __init__(self, vocab, num_heads=64, out_dim=64, max_length=200, dropout=0.2):
        ""
        super().__init__()

        self.dim = 25
        self.num_heads = num_heads
        self.max_length = max_length
        self.kernel_size = 3
        self.out_dim = out_dim
        self.dropout = dropout

        # Init word2vec
        word2vec = downloader.load(f'glove-twitter-{self.dim}')#(f'word2vec-google-news-{self.dim}')

        # Create embedding matrix based on word vectors and vocabulary
        embeddings = torch.FloatTensor([word2vec[word] if word in word2vec else np.zeros((self.dim,))
                                        for word in vocab.idx2tok])
        self.embeddings = nn.Embedding.from_pretrained(embeddings, freeze=True, padding_idx=0)
        self.conv = nn.Conv2d(1, num_heads, (self.kernel_size, self.dim))
        self.relu = nn.ReLU()

        # fully connected layer, eq 2
        self.fc = nn.Linear(self.max_length - self.kernel_size + 1, self.out_dim)
        self.sigmoid = nn.Sigmoid()


    def forward(self, reviews):
        embs = self.embeddings(reviews)

        # eq 18
        conv = self.conv(embs.unsqueeze(1))  # Only one channel
        c = self.relu(conv)

        # eq 19
        c = c.squeeze(-1).transpose(1,2).max(-1)[0]

        # eq 2
        p = self.sigmoid(self.fc(c))

        return p



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

    def forward(self, input_nodes, g, blocks):
        for ntype, nt_blocks in blocks.items():
            # 3.1.1 Word-level rationale generator
            p = self.embedder(input_nodes[ntype])
            z = p + (torch.round(p) - p).detach()  # Eq 3

            # 3.1.2 Review-level rationale generator
            # Todo