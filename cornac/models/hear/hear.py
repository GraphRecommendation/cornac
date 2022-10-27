import dgl.utils
import torch
from dgl.ops import edge_softmax
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


class HEARConv(dgl.nn.GATv2Conv):
    def __init__(self, aggregator='gatv2', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.aggregator = aggregator

    def forward(self, graph, feat, get_attention=False):
        r"""
        Description
        -----------
        Compute graph attention network layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor or pair of torch.Tensor
            If a torch.Tensor is given, the input feature of shape :math:`(N, D_{in})` where
            :math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.
            If a pair of torch.Tensor is given, the pair must contain two tensors of shape
            :math:`(N_{in}, D_{in_{src}})` and :math:`(N_{out}, D_{in_{dst}})`.
        get_attention : bool, optional
            Whether to return the attention values. Default to False.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, H, D_{out})` where :math:`H`
            is the number of heads, and :math:`D_{out}` is size of output feature.
        torch.Tensor, optional
            The attention values of shape :math:`(E, H, 1)`, where :math:`E` is the number of
            edges. This is returned only when :attr:`get_attention` is ``True``.

        Raises
        ------
        DGLError
            If there are 0-in-degree nodes in the input graph, it will raise DGLError
            since no message will be passed to those nodes. This will cause invalid output.
            The error can be ignored by setting ``allow_zero_in_degree`` parameter to ``True``.
        """
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise dgl.DGLError('There are 0-in-degree nodes in the graph, '
                                   'output for those nodes will be invalid. '
                                   'This is harmful for some applications, '
                                   'causing silent performance regression. '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = dgl.add_self_loop(g)` will resolve '
                                   'the issue. Setting ``allow_zero_in_degree`` '
                                   'to be `True` when constructing this module will '
                                   'suppress the check and let the code run.')

            if isinstance(feat, tuple):
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])
                feat_src = self.fc_src(h_src).view(-1, self._num_heads, self._out_feats)
                feat_dst = self.fc_dst(h_dst).view(-1, self._num_heads, self._out_feats)
            else:
                h_src = h_dst = self.feat_drop(feat)
                feat_src = self.fc_src(h_src).view(
                    -1, self._num_heads, self._out_feats)
                if self.share_weights:
                    feat_dst = feat_src
                else:
                    feat_dst = self.fc_dst(h_src).view(
                        -1, self._num_heads, self._out_feats)
                if graph.is_block:
                    feat_dst = feat_dst[:graph.number_of_dst_nodes()]
                    h_dst = h_dst[:graph.number_of_dst_nodes()]
            graph.srcdata.update({'el': feat_src})# (num_src_edge, num_heads, out_dim)
            graph.dstdata.update({'er': feat_dst})
            graph.apply_edges(dgl.function.u_add_v('el', 'er', 'e'))
            e = self.leaky_relu(graph.edata.pop('e'))# (num_src_edge, num_heads, out_dim)
            e = (e * self.attn).sum(dim=-1).unsqueeze(dim=2)# (num_edge, num_heads, 1)
            # compute softmax
            graph.edata['a'] = self.attn_drop(edge_softmax(graph, e)) # (num_edge, num_heads)

            if self.aggregator == 'narre':
                graph.srcdata.update({'el': h_src})

            # message passing
            graph.update_all(dgl.function.u_mul_e('el', 'a', 'm'),
                             dgl.function.sum('m', 'ft'))
            rst = graph.dstdata['ft']
            # residual
            if self.res_fc is not None:
                resval = self.res_fc(h_dst).view(h_dst.shape[0], -1, self._out_feats)
                rst = rst + resval
            # activation
            if self.activation:
                rst = self.activation(rst)

            if get_attention:
                return rst, graph.edata['a']
            else:
                return rst


class Model(nn.Module):
    def __init__(self, n_nodes, aggregator, predictor, node_dim, review_dim, final_dim, num_heads):
        super().__init__()

        self.aggregator = aggregator
        self.predictor = predictor
        self.node_dim = node_dim
        self.review_dim = review_dim
        self.final_dim = final_dim
        self.num_heads = num_heads
        self.node_embedding = nn.Embedding(n_nodes, node_dim)
        self.review_conv = HypergraphLayer(node_dim, review_dim)
        self.review_agg = HEARConv(aggregator, review_dim, final_dim, num_heads)
        if aggregator == 'narre':
            self.node_quality = nn.Embedding(n_nodes, review_dim)
            self.w_0 = nn.Linear(review_dim, final_dim)
        else:
            # Ignore dst during propagation as we have no dst embeddings.
            fc = self.review_agg.fc_dst
            torch.nn.init.zeros_(fc.weight)
            fc.weight.requires_grad = False
            torch.nn.init.zeros_(fc.bias)
            fc.bias.requires_grad = False

        if predictor == 'narre':
            self.node_preference = nn.Embedding(n_nodes, final_dim)
            self.w_1 = nn.Linear(final_dim, 1, bias=False)

        self.loss_fn = nn.MSELoss(reduction='mean')
        self.review_embs = None
        self.inf_emb = None

    def forward(self, blocks, x):
        x = self.review_conv(blocks[0], x)

        if self.aggregator == 'narre':
            g = blocks[1]
            x = (
                x[g.srcnodes['review'].data[dgl.NID]],
                self.node_quality(g.dstnodes['node'].data[dgl.NID])
            )

        x = self.review_agg(blocks[1], x)
        x = x.sum(1)

        if self.aggregator == 'narre':
            x = self.w_0(x)

        return x

    def e_dot_e(self, lhs_field, rhs_field, out):
        def func(edges):
            u, v = edges.data[lhs_field], (edges.data[rhs_field])
            o = (u * v).sum(-1)
            return {out: o.unsqueeze(-1)}
        return func

    def e_mul_e(self, lhs_field, rhs_field, out):
        def func(edges):
            u, v = edges.data[lhs_field], (edges.data[rhs_field])
            o = u * v
            return {out: o}
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
                g.apply_edges(self.e_mul_e('u', 'v', 'm'))

            return g.edata['m']

    def graph_predict_narre(self, g: dgl.DGLGraph, x):
        with g.local_scope():
            if self.aggregator != 'narre':
                g.ndata['h'] = x + self.node_preference(g.ndata[dgl.NID])
                g.apply_edges(dgl.function.u_mul_v('h', 'h', 'm'))
            else:
                x = x.reshape(128, 2, -1)
                u, v = g.edges()
                g.edata['u'] = x[:, 0] + self.node_preference(g.ndata[dgl.NID][u])
                g.edata['v'] = x[:, 1] + self.node_preference(g.ndata[dgl.NID][v])
                g.apply_edges(self.e_mul_e('u', 'v', 'm'))

            return self.w_1(g.edata['m'])

    def graph_predict(self, g: dgl.DGLGraph, x):
        if self.predictor == 'dot':
            return self.graph_predict_dot(g, x)
        elif self.predictor == 'narre':
            return self.graph_predict_narre(g, x)
        else:
            raise ValueError(f'Predictor not implemented for "{self.predictor}".')

    def _create_predict_graph(self, node, node_review_graph):
        u, v = node_review_graph.in_edges(node)
        g = dgl.to_block(dgl.graph((u, v)))
        embs = (self.review_embs[g.srcdata[dgl.NID]],
                self.node_quality(g.dstdata[dgl.NID]))
        return g, embs

    def review_representation(self, g, x):
        return self.review_conv(g, x)

    def review_aggregation(self, g, x):
        x = self.review_agg(g, x)
        x = x.sum(1)

        if self.aggregator == 'narre':
            x = self.w_0(x)

        return x

    def predict_dot(self, u_emb, i_emb):
        return u_emb.dot(i_emb)

    def predict_narre(self, user, item, u_emb, i_emb):
        h = (u_emb + self.node_preference(user)) * (i_emb + self.node_preference(item))
        return self.w_1(h)

    def predict(self, user, item, node_review_graph):
        if self.aggregator == 'narre':
            u_g, u_emb = self._create_predict_graph(user, node_review_graph)
            i_g, i_emb = self._create_predict_graph(item, node_review_graph)
            u_emb = self.review_aggregation(u_g, u_emb)
            i_emb = self.review_aggregation(i_g, i_emb)
            u_emb, i_emb = u_emb.squeeze(0), i_emb.squeeze(0)
        else:
            u_emb, i_emb = self.inf_emb[user], self.inf_emb[item]

        if self.predictor == 'dot':
            pred = self.predict_dot(u_emb, i_emb)
        elif self.predictor == 'narre':
            pred = self.predict_narre(user, item, u_emb, i_emb)
        else:
            raise ValueError(f'Predictor not implemented for "{self.predictor}".')

        return pred

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
            self.review_embs[indices] = self.review_representation(batched_graph, self.node_embedding(input_nodes))

        # Narre's predictor cannot be precalculated
        if self.aggregator != 'narre':
            # Node inference setup
            indices = {'node': node_review_graph.nodes('node')}
            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
            dataloader = dgl.dataloading.DataLoader(node_review_graph, indices, sampler, batch_size=1024, shuffle=False,
                                                    drop_last=False, device=device)

            self.inf_emb = torch.zeros((torch.max(indices['node'])+1, self.review_agg._out_feats)).to(device)

            # Node inference
            for input_nodes, output_nodes, blocks in dataloader:
                x = self.review_aggregation(blocks[0]['part_of'], self.review_embs[input_nodes['review']])
                self.inf_emb[output_nodes['node']] = x
