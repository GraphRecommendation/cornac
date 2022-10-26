import dgl.utils
import torch
from torch import nn

from cornac.models.hear.dgl_utils import HearReviewDataset, HearReviewSampler, HearReviewCollator


class HypergraphLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.linear = nn.Linear(in_dim, out_dim)
        self.activation = nn.LeakyReLU()

    def message(self, lhs_field, rhs_field, out):
        def func(edges):
            norm = edges.src['norm'] * edges.dst['norm'] * edges.data['norm']
            m = edges.src[lhs_field] * norm.unsqueeze(-1)
            return {out: m}

        return func

    def forward(self, g: dgl.DGLGraph, x):
        with g.local_scope():
            g.ndata['h'] = x

            g.update_all(self.message('h', 'h', 'm'), dgl.function.sum('m', 'h'))

            return self.activation(self.linear(dgl.mean_nodes(g, 'h')))


class Model(nn.Module):
    def __init__(self, n_nodes):
        super().__init__()

        self.node_embedding = nn.Embedding(n_nodes, 64)
        self.review_conv = HypergraphLayer(64, 32)
        self.review_agg = dgl.nn.GATv2Conv(32, 16, 3)
        self.loss_fn = nn.MSELoss(reduction='mean')
        self.inf_emb = None

    def forward(self, blocks, x):
        x = self.review_conv(blocks[0], x)
        x = self.review_agg(blocks[1], x).sum(1)

        return x

    def graph_predict(self, g: dgl.DGLGraph, x):
        with g.local_scope():
            g.ndata['h'] = x
            g.apply_edges(dgl.function.u_dot_v('h', 'h', 'm'))
            return g.edata['m']

    def predict(self, user, item):
        return self.inf_emb[user].dot(self.inf_emb[item])

    def loss(self, preds, target):
        return self.loss_fn(preds, target.unsqueeze(-1))

    def l2_loss(self, nodes):
        nodes = torch.unique(nodes)
        loss = torch.pow(self.node_embedding(nodes), 2).sum()
        # for parameter in self.parameters():
        #     if isinstance(parameter, nn.Linear):
        #         loss += torch.pow(parameter.weight, 2).sum()

        return loss

    def inference(self, review_graphs, node_review_graph, device):
        review_embs = torch.zeros((max(review_graphs)+1, self.review_conv.out_dim)).to(device)

        # Setup for review representation inference
        review_dataset = HearReviewDataset(review_graphs)
        review_sampler = HearReviewSampler(list(review_graphs.keys()))
        review_collator = HearReviewCollator()
        review_dataloader = dgl.dataloading.GraphDataLoader(review_dataset, batch_size=64, shuffle=False,
                                                            drop_last=False, collate_fn=review_collator.collate,
                                                            sampler=review_sampler)

        # Review inference
        for (input_nodes, batched_graph), indices in review_dataloader:
            input_nodes, batched_graph, indices = input_nodes.to(device), batched_graph.to(device), indices.to(device)
            review_embs[indices] = self.review_conv(batched_graph, self.node_embedding(input_nodes))

        # Node inference setup
        indices = {'node': node_review_graph.nodes('node')}
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
        dataloader = dgl.dataloading.DataLoader(node_review_graph, indices, sampler, batch_size=1024, shuffle=False,
                                                drop_last=False, device=device)

        self.inf_emb = torch.zeros((torch.max(indices['node'])+1, self.review_agg._out_feats)).to(device)

        # Node inference
        for input_nodes, output_nodes, blocks in dataloader:
            x = self.review_agg(blocks[0]['part_of'], review_embs[input_nodes['review']])
            self.inf_emb[output_nodes['node']] = x.sum(1)


