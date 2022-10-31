import dgl
import torch
from dgl.ops import edge_softmax
from dgl import function as fn
from dgl.utils import expand_as_pair
from torch import nn

# Based on https://github.com/LunaBlack/KGAT-pytorch/blob/e7305c3e80fb15fa02b3ec3993ad3a169b34ce64/model/KGAT.py#L13
from tqdm import tqdm

from cornac.models.kgat import dgl_utils


class KGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout, mode='bi-interaction', propagation_mode='attention'):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self._mode = mode
        self.propagation_mode = propagation_mode

        self.message_dropout = nn.Dropout(dropout)

        self.activation = nn.LeakyReLU()

        if mode == 'bi-interaction':
            self.W1 = nn.Linear(self.in_dim, self.out_dim)  # W1 in Equation (8)
            self.W2 = nn.Linear(self.in_dim, self.out_dim)  # W2 in Equation (8)

            # initialize
            nn.init.xavier_normal_(self.W1.weight)
            nn.init.xavier_normal_(self.W2.weight)
        else:
            raise NotImplementedError

    def forward(self, g: dgl.DGLGraph, x):
        with g.local_scope():
            feat_src, feat_dst = expand_as_pair(x, g)
            g.srcdata['emb'] = feat_src
            g.dstdata['emb'] = feat_dst

            if self.propagation_mode == 'attention':
                g.update_all(fn.u_mul_e('emb', 'a', 'm'), fn.sum('m', 'h_n'))
            elif self.propagation_mode == 'gates':
                g.update_all(fn.u_mul_e('emb', 'g', 'm'), fn.sum('m', 'h_n'))

            if self._mode == 'bi-interaction':
                out = self.activation(self.W1(g.dstdata['emb'] + g.dstdata['h_n'])) + \
                      self.activation(self.W2(g.dstdata['emb'] * g.dstdata['h_n']))

        return self.message_dropout(out)


class Model(nn.Module):
    def __init__(self, n_nodes, n_relations, entity_dim, relation_dim,
                 n_layers, layer_dims, dropout=0., use_cuda=False, mode='attention'):
        super(Model, self).__init__()
        self.use_cuda = use_cuda
        self.mode = mode

        # Define embedding
        self.node_embedding = nn.Embedding(n_nodes, entity_dim)
        self.relation_embedding = nn.Embedding(n_relations, relation_dim)
        self.W_r = nn.Parameter(torch.Tensor(n_relations, entity_dim, relation_dim))

        self.tanh = nn.Tanh()

        # Must have an output dim for each layer
        assert n_layers == len(layer_dims)

        if mode == 'gates':
            # All dimensions must be the same for gates to work.
            assert all(d == relation_dim for d in layer_dims)
            assert entity_dim == relation_dim
            self.gate_head_weight = nn.Parameter(torch.Tensor(relation_dim, relation_dim))
            self.gate_tail_weight = nn.Parameter(torch.Tensor(relation_dim, relation_dim))
            self.gate_relation_weight = nn.Parameter(torch.Tensor(relation_dim, relation_dim))
            self.gate_bias = nn.Parameter(torch.Tensor(relation_dim))
            self.sigmoid = nn.Sigmoid()

        layers = nn.ModuleList()
        out_dim = entity_dim
        for layer in range(n_layers):
            in_dim = out_dim
            out_dim = layer_dims[layer]
            layers.append(KGATLayer(in_dim, out_dim, dropout, propagation_mode=self.mode))

        self.layers = layers

        self._trans_r_loss_fn = torch.nn.LogSigmoid()
        self._loss_fn = torch.nn.MSELoss()

        # sampler = dgl.dataloading.neighbor.MultiLayerFullNeighborSampler(3)
        # self.collator = dgl.dataloading.NodeCollator(graph, graph.nodes(), sampler)

        self.embeddings = None

    def reset_parameters(self):
        for name, parameter in self.named_parameters():
            if name.endswith('bias'):
                nn.init.constant_(parameter, 0)
            else:
                nn.init.xavier_normal_(parameter)

    def _trans_r_fn(self, edges):
        r = edges.data['type']
        relation = self.relation_embedding(r)

        # Transforming head and tail into relation space
        tail = torch.einsum('be,ber->br', edges.src['emb'], self.W_r[r])
        head = torch.einsum('be,ber->br', edges.dst['emb'], self.W_r[r])

        # Calculate plausibilty score, equation 1
        return {'ps': torch.norm(head + relation - tail, 2, dim=1).pow(2)}

    def trans_r(self, g: dgl.DGLGraph, x):
        with g.local_scope():
            g.ndata['emb'] = x

            g.apply_edges(self._trans_r_fn)

            return g.edata['ps']

    def trans_r_loss(self, pos_emb, neg_emb):
        return - self._trans_r_loss_fn(neg_emb - pos_emb).mean()

    def l2_loss(self, g, embeddings, trans_r=False, neg_graph=None):
        users, items_i = g.edges()
        loss = embeddings[users].pow(2).norm(2) + \
               embeddings[items_i].pow(2).norm(2)

        if neg_graph is not None:
            _, items_j = neg_graph.edges()
            loss += embeddings[items_j].pow(2).norm(2)

        if trans_r:
            loss += self.relation_embedding(g.edata['type']).pow(2).norm(2)

        # Average by batch size.
        loss /= len(users)

        return loss

    def _attention(self, edges):
        r = edges.data['type']
        relation = self.relation_embedding(r)

        tail = torch.bmm(edges.src['emb'].unsqueeze(1), self.W_r[r]).squeeze()
        head = torch.bmm(edges.dst['emb'].unsqueeze(1), self.W_r[r]).squeeze()

        # Calculate attention
        return {'a': torch.sum(tail * self.tanh(head + relation), dim=-1)}

    def compute_attention(self, g: dgl.DGLGraph, batch_size, verbose=True):
        device = self.W_r.device
        with g.local_scope():
            sampler = dgl_utils.TransRSampler()
            sampler = dgl.dataloading.as_edge_prediction_sampler(
                sampler, prefetch_labels=['type']
            )
            dataloader = dgl.dataloading.DataLoader(
                g, g.edges(form='eid'), sampler,
                shuffle=False, drop_last=False, device=device, batch_size=batch_size
            )
            attention = torch.zeros(len(g.edges('eid')), device=device)

            for input_nodes, batched_g, _ in tqdm(dataloader, disable=not verbose):
                with batched_g.local_scope():
                    batched_g.ndata['emb'] = self.node_embedding(input_nodes)

                    batched_g.apply_edges(self._attention)

                    attention[batched_g.edata[dgl.EID]] = batched_g.edata['a']

            g.edata['a'] = attention.to(g.device)
            return edge_softmax(g, g.edata['a'])

    def forward(self, blocks, x):
        output_nodes = blocks[-1].dstdata[dgl.NID]
        iterator = zip(blocks, self.layers)

        n_out = len(output_nodes)
        embs = [x[:n_out]]
        for block, layer in iterator:
            x = layer(block, x)
            embs.append(x[:n_out])

        return torch.cat(embs, dim=1)

    def graph_predict(self, g: dgl.DGLGraph, embeddings):
        with g.local_scope():
            users, items = g.edges()

            return (embeddings[users] * embeddings[items]).sum(dim=1)

    def loss(self, preds, target):
        return self._loss_fn(preds, target)

    def inference(self, g, x, device):
        blocks = [dgl.to_block(g) for _ in self.layers]  # full graph propagations

        blocks = [b.to(device) for b in blocks]

        self.embeddings = self(blocks, x)

    def predict(self, user, item):
        u_emb, i_emb = self.embeddings[user], self.embeddings[item]
        return u_emb.dot(i_emb)

