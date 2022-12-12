from collections import defaultdict

import dgl.utils
import torch
from dgl.ops import edge_softmax
from torch import nn
import dgl.function as fn

from cornac.models.hear.dgl_utils import HearReviewDataset, HearReviewSampler, HearReviewCollator


class HypergraphLayer(nn.Module):
    def __init__(self, in_dim, out_dim, non_linear=True, op='max', num_layers=1):
        super().__init__()
        self.num_layers = num_layers
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.non_linear = non_linear
        self.op = op
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
            out = [dgl.readout_nodes(g, 'h', op=self.op)]
            for _ in range(self.num_layers):
                g.update_all(self.message('h', 'h', 'm'), fn.sum('m', 'h'))
                out.append(dgl.readout_nodes(g, 'h', op=self.op))

            out = torch.stack(out).mean(0)

            if self.non_linear:
                out = self.activation(self.linear(out))

            return out


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
                    graph.apply_edges(fn.u_add_e('el', 'qual', 'e'))
                else:
                    graph.edata.update({'qual': h_qual})
                    graph.apply_edges(self.rel_attention('qual', 'r_type', 'qual', self.fc_qual, self.fc_qual_bias,
                                                         False))
                    graph.edata.update({'qual': graph.edata['qual'].view(-1, self._num_heads, self._out_feats)})
                    graph.edata.update({'e': graph.edata.pop('el') + graph.edata.pop('qual')})
            else:
                graph.apply_edges(fn.copy_u('el', 'e'))

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
            graph.update_all(fn.u_mul_e('el', 'a', 'm'),
                             fn.sum('m', 'ft'))
            rst = graph.dstdata['ft']

            # activation
            if self.activation:
                rst = self.activation(rst)

            if get_attention:
                return rst, graph.edata['a']
            else:
                return rst


class Model(nn.Module):
    def __init__(self, n_nodes, n_relations, n_sentiments, aggregator, predictor, node_dim, review_dim, final_dim, transr_dim, num_heads,
                 layer_dropout, attention_dropout, learned_node_embeddings=None, learned_preference=False,
                 learned_embeddings=False):
        super().__init__()

        self.aggregator = aggregator
        self.predictor = predictor
        self.node_dim = node_dim
        self.review_dim = review_dim
        self.final_dim = final_dim
        self.num_heads = num_heads
        self.learned_node_embeddings = learned_node_embeddings

        if not learned_embeddings:
            self.node_embedding = nn.Embedding(n_nodes, node_dim)
        else:
            self.node_embedding_mlp = nn.Sequential(
                    nn.Linear(self.learned_node_embeddings.shape[1], node_dim),
                    nn.LeakyReLU()
            )

        self.review_conv = HypergraphLayer(node_dim, review_dim, non_linear=False, num_layers=3)
        self.review_agg = HEARConv(aggregator, n_nodes, n_relations, review_dim, final_dim, num_heads,
                                   feat_drop=layer_dropout[1], attn_drop=attention_dropout)

        self.node_dropout = nn.Dropout(layer_dropout[0])

        self.transr_src = nn.Parameter(torch.zeros((n_sentiments, node_dim*2 + review_dim, transr_dim)))
        self.transr_dst = nn.Parameter(torch.zeros((n_sentiments, node_dim*2, transr_dim)))
        self.transr_rel = nn.Parameter(torch.zeros((n_sentiments, transr_dim)))

        if aggregator.startswith('narre'):
            self.w_0 = nn.Linear(review_dim, final_dim)

        if predictor == 'narre':
            if not learned_preference:
                self.node_preference = nn.Embedding(n_nodes, final_dim)
            else:
                self.preference_mlp = nn.Sequential(
                    nn.Linear(self.learned_node_embeddings.shape[1], final_dim),
                    nn.LeakyReLU()
                )
                self.review_mlp = nn.Sequential(
                    nn.Linear(final_dim, final_dim),
                    nn.LeakyReLU()
                )
                final_dim = (node_dim + final_dim) * 2
            self.w_1 = nn.Linear(final_dim, 1, bias=False)
            self.bias = nn.Parameter(torch.FloatTensor(n_nodes))
        elif predictor == 'biinteraction':
            if not learned_preference:
                self.node_preference = nn.Embedding(n_nodes, final_dim)

            self.w_1 = nn.Linear(final_dim, final_dim, bias=False)
            self.w_2 = nn.Linear(final_dim, final_dim, bias=False)

        elif predictor == 'cosine':
            self.node_preference = nn.Embedding(n_nodes, final_dim)
            self.edge_predictor = dgl.nn.EdgePredictor('cos')

        self.rating_loss_fn = nn.MSELoss(reduction='mean')
        self.ccl_loss_fn = nn.ReLU()
        self.bpr_loss_fn = nn.Softplus()
        self.review_embs = None
        self.inf_emb = None
        self.first = True

    def reset_parameters(self):
        for name, parameter in self.named_parameters():
            if name.endswith('bias'):
                nn.init.constant_(parameter, 0)
            else:
                nn.init.xavier_normal_(parameter)

    def get_initial_embedings(self, nodes):
        if hasattr(self, 'node_embedding'):
            return self.node_embedding(nodes)
        else:
            return self.node_embedding_mlp(self.learned_node_embeddings[nodes])

    def _trans_r_propagate(self, g, x):
        with g.local_scope():
            g.srcdata['h'] = x
            out = defaultdict(list)
            for stype, etype, dtype in g.canonical_etypes:
                g[etype].update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
                out[dtype].append(g[etype].dstdata['h'])
            return {ntype: torch.cat(o, dim=-1) for ntype, o in out.items()}

    def trans_r_forward(self, blocks, x):
        # Get review embeddings
        x = self.node_dropout(x)
        x = {'review': self.review_conv(blocks[0], x)}

        # Get user/item embeddings
        x.update({ntype: self.node_dropout(self.get_initial_embedings(nids))
                  for ntype, nids in blocks[1].srcdata[dgl.NID].items()
                  if ntype != 'review' and blocks[1].num_src_nodes(ntype)})

        # Get rui embeddings
        x = self._trans_r_propagate(blocks[1], x)

        return x

    def l2_loss(self, pos, neg, emb):
        loss = 0
        src, dst_i = pos.edges()
        _, dst_j = neg.edges()

        s_emb, i_emb, j_emb = emb[src], emb[dst_i], emb[dst_j]

        loss += s_emb.norm(2).pow(2) + i_emb.norm(2).pow(2) + j_emb.norm(2).pow(2)

        loss = 0.5 * loss / pos.num_src_nodes()

        return loss

    def _trans_r_plausibility(self, rhs_field, lhs_field, edge, out):
        def func(edges):
            src = torch.tanh_(dgl.ops.gather_mm(edges.src[rhs_field], self.transr_src, idx_b=edges.data[edge]))
            dst = torch.tanh_(dgl.ops.gather_mm(edges.dst[lhs_field], self.transr_dst, idx_b=edges.data[edge]))
            plausibility = src + self.transr_rel[edges.data[edge]] - dst
            plausibility = plausibility.norm(2, dim=-1, keepdim=True) ** 2
            # plausibility = torch.mul(src + self.transr_rel[edges.data[edge]], dst).sum(dim=-1)
            return {out: plausibility}
        return func

    def trans_r_pred(self, g, x):
        with g.local_scope():
            g.ndata['h'] = x
            g.apply_edges(self._trans_r_plausibility('h', 'h', 'sent', 'p'))
            return g.edata['p']

    def forward(self, blocks, x):
        x = self.node_dropout(x)
        x = self.review_conv(blocks[0], x)

        x = self.review_agg(blocks[1], x)
        x = x.sum(1)

        # if self.aggregator.startswith('narre'):
        #     x = self.w_0(x)

        return x

    def _graph_predict_dot(self, g: dgl.DGLGraph, x):
        with g.local_scope():
            g.ndata['h'] = x
            g.apply_edges(fn.u_dot_v('h', 'h', 'm'))

            return g.edata['m']

    def _graph_predict_narre(self, g: dgl.DGLGraph, x):
        with g.local_scope():
            if hasattr(self, 'node_preference'):
                np = self.node_preference(g.ndata[dgl.NID])
            else:
                np = self.learned_node_embeddings[g.ndata[dgl.NID]]
                np = self.preference_mlp(np)
                x = self.review_mlp(x)

            g.ndata['h'] = x + self.node_dropout(np)
            g.ndata['b'] = self.bias[g.ndata[dgl.NID]]
            g.apply_edges(fn.u_mul_v('h', 'h', 'm'))
            g.apply_edges(fn.u_add_v('b', 'b', 'b'))

            return self.w_1(g.edata['m']) + g.edata['b'].unsqueeze(-1)

    def _graph_predict_narre2(self, g: dgl.DGLGraph, x, use_preference):
        with g.local_scope():
            if hasattr(self, 'node_preference'):
                np = self.node_preference(g.ndata[dgl.NID])
            else:
                np = self.learned_node_embeddings[g.ndata[dgl.NID]]
            if use_preference:
                g.ndata['h'] = torch.cat([x, np], dim=-1)
            else:
                g.ndata['h'] = x
            g.ndata['b'] = self.bias[g.ndata[dgl.NID]]
            g.apply_edges(fn.u_dot_v('h', 'h', 'm'))
            g.apply_edges(fn.u_add_v('b', 'b', 'b'))

            out = g.edata['m']
            # out += g.edata['b'].unsqueeze(-1)

            return out #+ g.edata['b'].unsqueeze(-1)

    def _graph_predict_narre3(self, g: dgl.DGLGraph, x, use_preference):
        with g.local_scope():
            if hasattr(self, 'node_preference'):
                np = self.node_preference(g.ndata[dgl.NID])
            else:
                np = self.learned_node_embeddings[g.ndata[dgl.NID]]
            if use_preference:
                g.ndata['h'] = torch.cat([x, np], dim=-1)
                u, v = g.edges()
                out = self.w_1(torch.cat([g.ndata['h'][u], g.ndata['h'][v]], dim=-1))
            else:
                g.ndata['h'] = x
                g.apply_edges(fn.u_dot_v('h', 'h', 'm'))
                out = g.edata['m']

            return out #+ g.edata['b'].unsqueeze(-1)

    def _graph_predict_bi(self, g: dgl.DGLGraph, x, use_preference):
        with g.local_scope():
            if hasattr(self, 'node_preference'):
                np = self.node_preference(g.ndata[dgl.NID])
            else:
                np = self.learned_node_embeddings[g.ndata[dgl.NID]]

            np = self.node_dropout(np)
            x = self.node_dropout(x)

            if use_preference:
                w1 = torch.nn.functional.leaky_relu(self.w_1(x + np))
                w2 = torch.nn.functional.leaky_relu(self.w_2(x * np))
                x = w1 + w2

            g.ndata['h'] = x
            g.apply_edges(fn.u_dot_v('h', 'h', 'm'))
            out = g.edata['m']

            return out

    def _graph_predict_cosine(self, g, x):
        with g.local_scope():
            u, v = g.edges()
            h_src = x[u]# + self.node_preference(g.ndata[dgl.NID][u])
            h_dst = x[v]# + self.node_preference(g.ndata[dgl.NID][v])
            pred = self.edge_predictor(h_src, h_dst)

            return pred

    def graph_predict(self, g: dgl.DGLGraph, x, use_preference=False):
        if self.predictor == 'dot':
            return self._graph_predict_dot(g, x)
        elif self.predictor == 'narre':
            return self._graph_predict_narre2(g, x, use_preference)
        elif self.predictor == 'biinteraction':
            return self._graph_predict_bi(g, x, True)
        elif self.predictor == 'cosine':
            return self._graph_predict_cosine(g, x)
        else:
            raise ValueError(f'Predictor not implemented for "{self.predictor}".')

    def review_representation(self, g, x):
        return self.review_conv(g, x)

    def review_aggregation(self, g, x):
        x = self.review_agg(g, x)
        x = x.sum(1)

        # if self.aggregator.startswith('narre'):
        #     x = self.w_0(x)

        return x

    def _predict_dot(self, u_emb, i_emb):
        return (u_emb * i_emb).sum(-1)

    def _predict_narre(self, user, item, u_emb, i_emb):
        if hasattr(self, 'node_preference'):
            np_u = self.node_preference(user)
            np_i = self.node_preference(item)
        else:
            np_u = self.learned_node_embeddings[user]
            np_i = self.learned_node_embeddings[item]
            np_u, np_i = self.preference_mlp(np_u), self.preference_mlp(np_i)
            u_emb, i_emb = self.review_mlp(u_emb), self.review_mlp(i_emb)
        h = (u_emb + np_u) * (i_emb + np_i)
        return self.w_1(h) + (self.bias[user] + self.bias[item]).unsqueeze(-1)

    def _predict_narre2(self, user, item, u_emb, i_emb, use_preference=False):
        if hasattr(self, 'node_preference'):
            np_u = self.node_preference(user)
            np_i = self.node_preference(item)
        else:
            np_u = self.learned_node_embeddings[user]
            np_i = self.learned_node_embeddings[item]

        if use_preference:
            if not self.first:
                u = torch.cat([u_emb, np_u], dim=-1)
                i = torch.cat([i_emb, np_i], dim=-1)
            else:
                u = np_u
                i = np_i
                self.first = False
        else:
            u = u_emb
            i = i_emb

        h = (u * i).sum(-1)
        # h += (self.bias[user] + self.bias[item])

        return h.reshape(-1, 1) #+ (self.bias[user] + self.bias[item]).reshape(-1, 1)

    def _predict_bi(self, user, item, u_emb, i_emb, use_preference=False):
        if hasattr(self, 'node_preference'):
            np_u = self.node_preference(user)
            np_i = self.node_preference(item)
        else:
            np_u = self.learned_node_embeddings[user]
            np_i = self.learned_node_embeddings[item]

        if use_preference:
            w1_u = torch.nn.functional.leaky_relu(self.w_1(u_emb + np_u))
            w1_i = torch.nn.functional.leaky_relu(self.w_1(i_emb + np_i))
            w2_u = torch.nn.functional.leaky_relu(self.w_2(u_emb * np_u))
            w2_i = torch.nn.functional.leaky_relu(self.w_2(i_emb * np_i))
            u = w1_u + w2_u
            i = w1_i + w2_i
        else:
            u = u_emb
            i = i_emb

        h = (u * i).sum(-1)
        # h += (self.bias[user] + self.bias[item])

        return h.reshape(-1, 1)

    def _predict_narre3(self, user, item, u_emb, i_emb, use_preference=False):
        if hasattr(self, 'node_preference'):
            np_u = self.node_preference(user)
            np_i = self.node_preference(item)
        else:
            np_u = self.learned_node_embeddings[user]
            np_i = self.learned_node_embeddings[item]

        if use_preference:
            if not self.first:
                u = torch.cat([u_emb, np_u], dim=-1)
                i = torch.cat([i_emb, np_i], dim=-1)
                u = torch.repeat_interleave(u, output_size=i.shape)
                h = self.w_1(torch.cat([u, i]))
            else:
                u = np_u
                i = np_i
                h = (u * i).sum(-1)
                self.first = False
        else:
            u = u_emb
            i = i_emb
            h = (u * i).sum(-1)

        return h.reshape(-1, 1)

    def _predict_cosine(self, user, item, u_emb, i_emb):
        np_u = self.node_preference(user)
        np_i = self.node_preference(item)
        pred = self.edge_predictor(u_emb# + np_u
                                   , i_emb# + np_i
                                   )
        return pred

    def predict(self, user, item, use_preference=False):
        u_emb, i_emb = self.inf_emb[user], self.inf_emb[item]
        # u_emb, i_emb = self.learned_node_embeddings[user], self.learned_node_embeddings[item]

        if self.predictor == 'dot':
            pred = self._predict_dot(u_emb, i_emb)
        elif self.predictor == 'narre':
            pred = self._predict_narre2(user, item, u_emb, i_emb, use_preference)
        elif self.predictor == 'biinteraction':
            pred = self._predict_bi(user, item, u_emb, i_emb, True)
        elif self.predictor == 'cosine':
            pred = self._predict_cosine(user, item, u_emb, i_emb)
        else:
            raise ValueError(f'Predictor not implemented for "{self.predictor}".')

        return pred

    def rating_loss(self, preds, target):
        return self.rating_loss_fn(preds, target.unsqueeze(-1))

    def _bpr_loss(self, preds_i, preds_j):
        return self.bpr_loss_fn(- (preds_i - preds_j))

    def _ccl_loss(self, preds_i, preds_j, w, m):
        # (1 - i) + w * 1/|N| * sum(max(0, j - m) for j in N
        # EQ. 1 in SimpleX, but based on code in:
        # https://github.com/xue-pai/MatchBox/blob/master/deem/pytorch/losses/cosine_contrastive_loss.py
        pos_loss = self.ccl_loss_fn(1 - preds_i)
        neg_loss = self.ccl_loss_fn(preds_j - m)
        neg_loss = neg_loss.mean(dim=-1, keepdims=True)
        if w is not None:
            neg_loss *= w

        return pos_loss + neg_loss

    def _dcl_loss(self, preds_i, preds_j, w, m):
        # Similar to ccl, but for dot product. We therefore use j instead of 1
        # since we cannot guarentee that i <= 1
        # Similarly,
        pos_loss = self.ccl_loss_fn(preds_j - preds_i)  # easy
        neg_loss = self.ccl_loss_fn(preds_j - preds_i + m)  # hard
        neg_loss = neg_loss.mean(dim=-1, keepdims=True)
        if w is not None:
            neg_loss *= w

        return pos_loss + neg_loss

    def ranking_loss(self, preds_i, preds_j, loss_fn, *args):
        if loss_fn == 'bpr':
            loss = self._bpr_loss(preds_i, preds_j)
        elif loss_fn == 'ccl':
            loss = self._ccl_loss(preds_i, preds_j, *args)
        elif loss_fn == 'dcl':
            loss = self._dcl_loss(preds_i, preds_j, *args)
        else:
            raise NotImplementedError

        return loss.mean()

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
            self.review_embs[indices] = self.review_representation(batched_graph,
                                                                   self.get_initial_embedings(input_nodes))

        # Node inference setup
        indices = {'node': node_review_graph.nodes('node')}
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
        dataloader = dgl.dataloading.DataLoader(node_review_graph, indices, sampler, batch_size=1024, shuffle=False,
                                                drop_last=False, device=device)

        self.inf_emb = torch.zeros((torch.max(indices['node'])+1, self.node_dim)).to(device)

        # Node inference
        for input_nodes, output_nodes, blocks in dataloader:
            x = self.review_aggregation(blocks[0]['part_of'], self.review_embs[input_nodes['review']])
            self.inf_emb[output_nodes['node']] = x
