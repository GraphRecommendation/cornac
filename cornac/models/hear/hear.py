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
    def __init__(self, n_nodes, aggregator, predictor='dot'):
        super().__init__()

        self.aggregator = aggregator
        self.predictor = predictor
        self.node_embedding = nn.Embedding(n_nodes, 64)
        self.review_conv = HypergraphLayer(64, 32)
        self.review_agg = dgl.nn.GATv2Conv(32, 16, 3)
        if aggregator == 'narre':
            self.node_quality = nn.Embedding(n_nodes, 32)
            self.node_preference = nn.Embedding(n_nodes, 16)
        else:
            # Ignore dst during propagation
            fc = self.review_agg.fc_dst
            torch.nn.init.zeros_(fc.weight)
            fc.weight.requires_grad = False
            torch.nn.init.zeros_(fc.bias)
            fc.bias.requires_grad = False

        self.loss_fn = nn.MSELoss(reduction='mean')
        self.review_embs = None
        self.inf_emb = None

    def forward(self, blocks, x):
        x = self.review_conv(blocks[0], x)

        g = blocks[1]
        if self.aggregator == 'narre':
            x = (
                x[g.srcnodes['review'].data[dgl.NID]],
                self.node_quality(g.dstnodes['node'].data[dgl.NID])
            )
        else:
            x = (x[g.srcnodes()], x[g.dstnodes()])

        x = self.review_agg(blocks[1], x)
        x = x.sum(1)

        return x

    def e_dot_e(self, lhs_field, rhs_field, out):
        def func(edges):
            u, v = edges.data[lhs_field], (edges.data[rhs_field])
            o = (u * v).sum(-1)
            return {out: o.unsqueeze(-1)}
        return func

    def graph_predict_dot(self, g: dgl.DGLGraph, x):
        with g.local_scope():
            if self.aggregator != 'narre':
                g.ndata['h'] = x
                g.apply_edges(dgl.function.u_dot_v('h', 'h', 'm'))
            else:
                x = x.reshape(128, 2, -1)
                g.edata['u'] = x[:, 0]
                g.edata['v'] = x[:, 1]
                g.apply_edges(self.e_dot_e('u', 'v', 'm'))

            return g.edata['m']

    def graph_predict_narre(self, g: dgl.DGLGraph, x):
        assert self.aggregator == 'narre', f"Not implemented for {self.aggregator} aggregator."

    def graph_predict(self, g: dgl.DGLGraph, x):
        if self.predictor == 'dot':
            return self.graph_predict_dot(g, x)
        elif self.predictor == 'narre':
            self.graph_predict_narre(g, x)
        else:
            raise ValueError(f'Predictor not implemented for "{self.predictor}".')

    def _create_predict_graph(self, node, node_review_graph):
        u, v = node_review_graph.in_edges(node)
        g = dgl.graph((u, v))
        g = dgl.to_block(g)
        embs = (self.review_embs[g.srcdata[dgl.NID]], self.node_quality(g.dstdata[dgl.NID]))
        return g, embs

    def predict(self, user, item, node_review_graph):
        if self.aggregator == 'narre':
            u_g, u_emb = self._create_predict_graph(user, node_review_graph)
            i_g, i_emb = self._create_predict_graph(item, node_review_graph)
            u_emb = self.review_agg(u_g, u_emb).sum(1).squeeze(0)
            i_emb = self.review_agg(i_g, i_emb).sum(1).squeeze(0)
        else:
            u_emb, i_emb = self.inf_emb[user], self.inf_emb[item]
        return u_emb.dot(i_emb)

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
        self.review_embs = torch.zeros((max(review_graphs)+1, self.review_conv.out_dim)).to(device)

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
            self.review_embs[indices] = self.review_conv(batched_graph, self.node_embedding(input_nodes))

        # Node inference setup
        if self.aggregator != 'narre':
            indices = {'node': node_review_graph.nodes('node')}
            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
            dataloader = dgl.dataloading.DataLoader(node_review_graph, indices, sampler, batch_size=1024, shuffle=False,
                                                    drop_last=False, device=device)

            self.inf_emb = torch.zeros((torch.max(indices['node'])+1, self.review_agg._out_feats)).to(device)

            # Node inference
            for input_nodes, output_nodes, blocks in dataloader:
                x = self.review_agg(blocks[0]['part_of'], self.review_embs[input_nodes['review']])
                self.inf_emb[output_nodes['node']] = x.sum(1)


