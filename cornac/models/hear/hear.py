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


class HEARConv(nn.Module):
    def __init__(self,
                 aggregator,
                 n_nodes,
                 n_relations,
                 in_feats,
                 attention_feats,
                 num_heads,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 activation=None,
                 allow_zero_in_degree=False,
                 bias=True):
        super(HEARConv, self).__init__()
        self.aggregator = aggregator
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = dgl.utils.expand_as_pair(in_feats)
        self._out_feats = attention_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        if self.aggregator != 'narre-rel':
            self.fc_src = nn.Linear(
                self._in_src_feats, attention_feats * num_heads, bias=bias)
        else:
            self.fc_src = nn.Parameter(torch.Tensor(
                n_relations, self._in_src_feats, attention_feats * num_heads
            ))
            self.fc_src_bias = nn.Parameter(torch.Tensor(
                n_relations, attention_feats * num_heads
            ))

        if self.aggregator.startswith('narre'):
            self.node_quality = nn.Embedding(n_nodes, self._in_dst_feats)
            if self.aggregator == 'narre':
                self.fc_qual = nn.Linear(self._in_dst_feats, attention_feats * num_heads, bias=bias)
            elif self.aggregator == 'narre-rel':
                self.fc_qual = nn.Parameter(torch.Tensor(
                    n_relations, self._in_dst_feats, attention_feats * num_heads
                ))
                self.fc_qual_bias = nn.Parameter(torch.Tensor(
                    n_relations, attention_feats * num_heads
                ))
            else:
                raise NotImplementedError(f'Not implemented any aggregator named {self.aggregator}.')
        elif self.aggregator == 'gatv2':
            pass
        else:
            raise NotImplementedError(f'Not implemented any aggregator named {self.aggregator}.')
        if self.aggregator != 'narre-rel':
            self.attn = nn.Parameter(torch.FloatTensor(size=(1, num_heads, attention_feats)))
        else:
            self.attn = nn.Parameter(torch.FloatTensor(size=(n_relations, num_heads, attention_feats)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.activation = activation
        self.bias = bias

    def rel_attention(self, lhs_field, rhs_field, out, w, b, source=True):
        def func(edges):
            idx = edges.data[rhs_field]
            data = edges.src[lhs_field] if source else edges.data[lhs_field]
            return {out: dgl.ops.gather_mm(data, w, idx_b=idx) + b[idx]}
        return func

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

            h_src = self.feat_drop(feat)
            if self.aggregator != 'narre-rel':
                feat_src = self.fc_src(h_src).view(-1, self._num_heads, self._out_feats)
                graph.srcdata.update({'el': feat_src})# (num_src_edge, num_heads, out_dim)
            else:
                graph.srcdata.update({'el': h_src})
                graph.apply_edges(self.rel_attention('el', 'r_type', 'el', self.fc_src, self.fc_src_bias))
                graph.edata.update({'el': graph.edata['el'].view(-1, self._num_heads, self._out_feats)})

            if self.aggregator.startswith('narre'):
                h_qual = self.feat_drop(self.node_quality(graph.edata['nid']))
                if self.aggregator != 'narre-rel':
                    feat_qual = self.fc_qual(h_qual).view(-1, self._num_heads, self._out_feats)
                    graph.edata.update({'qual': feat_qual})
                    graph.apply_edges(dgl.function.u_add_e('el', 'qual', 'e'))
                else:
                    graph.edata.update({'qual': h_qual})
                    graph.apply_edges(self.rel_attention('qual', 'r_type', 'qual', self.fc_qual, self.fc_qual_bias,
                                                         False))
                    graph.edata.update({'qual': graph.edata['qual'].view(-1, self._num_heads, self._out_feats)})
                    graph.edata.update({'e': graph.edata.pop('el') + graph.edata.pop('qual')})
            else:
                graph.apply_edges(dgl.function.copy_u('el', 'e'))

            e = self.leaky_relu(graph.edata.pop('e'))# (num_src_edge, num_heads, out_dim)

            if self.aggregator != 'narre-rel':
                e = (e * self.attn).sum(dim=-1).unsqueeze(dim=2)# (num_edge, num_heads, 1)
            else:
                e = (e * self.attn[graph.edata['r_type']]).sum(dim=-1).unsqueeze(dim=2)

            # compute softmax
            graph.edata['a'] = self.attn_drop(edge_softmax(graph, e)) # (num_edge, num_heads)

            if self.aggregator.startswith('narre'):
                graph.srcdata.update({'el': h_src})

            # message passing
            graph.update_all(dgl.function.u_mul_e('el', 'a', 'm'),
                             dgl.function.sum('m', 'ft'))
            rst = graph.dstdata['ft']

            # activation
            if self.activation:
                rst = self.activation(rst)

            if get_attention:
                return rst, graph.edata['a']
            else:
                return rst


class Model(nn.Module):
    def __init__(self, n_nodes, n_relations, aggregator, predictor, node_dim, review_dim, final_dim, num_heads, layer_dropout,
                 attention_dropout):
        super().__init__()

        self.aggregator = aggregator
        self.predictor = predictor
        self.node_dim = node_dim
        self.review_dim = review_dim
        self.final_dim = final_dim
        self.num_heads = num_heads
        self.node_embedding = nn.Embedding(n_nodes, node_dim)
        self.review_conv = HypergraphLayer(node_dim, review_dim)
        self.review_agg = HEARConv(aggregator, n_nodes, n_relations, review_dim, final_dim, num_heads,
                                   feat_drop=layer_dropout[1], attn_drop=attention_dropout)

        self.node_dropout = nn.Dropout(layer_dropout[0])

        if aggregator.startswith('narre'):
            # self.node_quality = nn.Embedding(n_nodes, review_dim)
            self.w_0 = nn.Linear(review_dim, final_dim)

        if predictor == 'narre':
            self.node_preference = nn.Embedding(n_nodes, final_dim)
            self.w_1 = nn.Linear(final_dim, 1, bias=False)

        self.rating_loss_fn = nn.MSELoss(reduction='mean')
        self.ranking_loss_fn = nn.ReLU()
        self.review_embs = None
        self.inf_emb = None

    def reset_parameters(self):
        for name, parameter in self.named_parameters():
            if name.endswith('bias'):
                nn.init.constant_(parameter, 0)
            else:
                nn.init.xavier_normal_(parameter)

    def forward(self, blocks, x):
        x = self.node_dropout(x)
        x = self.review_conv(blocks[0], x)

        x = self.review_agg(blocks[1], x)
        x = x.sum(1)

        if self.aggregator.startswith('narre'):
            x = self.w_0(x)

        return x

    def _graph_predict_dot(self, g: dgl.DGLGraph, x):
        with g.local_scope():
            g.ndata['h'] = x
            g.apply_edges(dgl.function.u_dot_v('h', 'h', 'm'))

            return g.edata['m']

    def _graph_predict_narre(self, g: dgl.DGLGraph, x):
        with g.local_scope():
            g.ndata['h'] = x + self.node_dropout(self.node_preference(g.ndata[dgl.NID]))
            g.apply_edges(dgl.function.u_mul_v('h', 'h', 'm'))

            return self.w_1(g.edata['m'])

    def graph_predict(self, g: dgl.DGLGraph, x):
        if self.predictor == 'dot':
            return self._graph_predict_dot(g, x)
        elif self.predictor == 'narre':
            return self._graph_predict_narre(g, x)
        else:
            raise ValueError(f'Predictor not implemented for "{self.predictor}".')

    def review_representation(self, g, x):
        return self.review_conv(g, x)

    def review_aggregation(self, g, x):
        x = self.review_agg(g, x)
        x = x.sum(1)

        if self.aggregator.startswith('narre'):
            x = self.w_0(x)

        return x

    def _predict_dot(self, u_emb, i_emb):
        return (u_emb * i_emb).sum(-1)

    def _predict_narre(self, user, item, u_emb, i_emb):
        h = (u_emb + self.node_preference(user)) * (i_emb + self.node_preference(item))
        return self.w_1(h)

    def predict(self, user, item):
        u_emb, i_emb = self.inf_emb[user], self.inf_emb[item]

        if self.predictor == 'dot':
            pred = self._predict_dot(u_emb, i_emb)
        elif self.predictor == 'narre':
            pred = self._predict_narre(user, item, u_emb, i_emb)
        else:
            raise ValueError(f'Predictor not implemented for "{self.predictor}".')

        return pred

    def rating_loss(self, preds, target):
        return self.rating_loss_fn(preds, target.unsqueeze(-1))

    def ranking_loss(self, preds_i, preds_j, w, m):
        # (1 - i) + w * 1/|N| * sum(max(0, j - m) for j in N
        # EQ. 1 in SimpleX, but based on code in:
        # https://github.com/xue-pai/MatchBox/blob/master/deem/pytorch/losses/cosine_contrastive_loss.py
        pos_loss = self.ranking_loss_fn(1 - preds_i)
        bpr = - torch.log2(torch.sigmoid(preds_i - preds_j)).mean(-1, keepdims=True)
        neg_loss = self.ranking_loss_fn(preds_j - m)
        neg_loss = neg_loss.mean(dim=-1, keepdims=True)
        if w is not None:
            neg_loss *= w
        loss = pos_loss + neg_loss #+ bpr
        return loss.mean()

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
