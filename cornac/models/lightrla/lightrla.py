from collections import defaultdict

import dgl.utils
import torch
from dgl.ops import edge_softmax
from torch import nn
import dgl.function as fn

import cornac.models.ngcf.ngcf
from cornac.models.hear.dgl_utils import HearReviewDataset, HearReviewSampler, HearReviewCollator


class HypergraphLayer(nn.Module):
    def __init__(self, in_dim, out_dim, non_linear=True, op='max', num_layers=1, n_relations=0):
        super().__init__()
        self.num_layers = num_layers
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.non_linear = non_linear
        self.op = op
        if n_relations > 0:
            self.relation_w_src = nn.Parameter(torch.zeros((num_layers, n_relations, in_dim, in_dim)))
            self.relation_w_affinity = nn.Parameter(torch.zeros((num_layers, n_relations, in_dim, in_dim)))
        self.activation = nn.LeakyReLU()

    def message(self, lhs_field, rhs_field, edge, out, layer):
        def func(edges):
            norm = edges.src['norm'] * edges.dst['norm'] * edges.data['norm']
            m = edges.src[lhs_field]
            if hasattr(self, 'relation_w_src'):
                m1 = dgl.ops.gather_mm(m, self.relation_w_src[layer], idx_b=edges.data[edge])
                m2 = dgl.ops.gather_mm(m * edges.dst[rhs_field], self.relation_w_affinity[layer],
                                       idx_b=edges.data[edge])
                m = m1 + m2

            m = m * norm.unsqueeze(-1)

            if self.non_linear:
                m = self.activation(m)

            return {out: m}

        return func

    def forward(self, g: dgl.DGLGraph, x):
        with g.local_scope():
            g.ndata['h'] = x
            out = [dgl.readout_nodes(g, 'h', op=self.op)]
            for l in range(self.num_layers):
                g.update_all(self.message('h', 'h',  'type', 'm', l), fn.sum('m', 'h'))
                out.append(dgl.readout_nodes(g, 'h', op=self.op))

            out = torch.stack(out).mean(0)

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
    def reset_parameters(self):
        for name, parameter in self.named_parameters():
            if name.endswith('bias'):
                nn.init.constant_(parameter, 0)
            else:
                nn.init.xavier_normal_(parameter)

    def __init__(self, g, n_nodes, n_hyper_graph_types, n_lgcn_relations, aggregator, predictor, node_dim,
                 num_heads, layer_dropout, attention_dropout, narre_preference='lightgcn', use_cuda=True):
        super().__init__()

        self.aggregator = aggregator
        self.predictor = predictor
        self.narre_preference = narre_preference
        self.node_dim = node_dim
        self.num_heads = num_heads

        self.node_embedding = nn.Embedding(n_nodes, node_dim)

        n_layers = 3
        self.review_conv = HypergraphLayer(node_dim, node_dim, non_linear=False, num_layers=n_layers,
                                           n_relations=n_hyper_graph_types)
        self.review_agg = HEARConv(aggregator, n_nodes, n_lgcn_relations, node_dim, node_dim, num_heads,
                                   feat_drop=layer_dropout[1], attn_drop=attention_dropout)

        self.node_dropout = nn.Dropout(layer_dropout[0])

        self.lightgcn = cornac.models.ngcf.ngcf.Model(g, node_dim, [node_dim]*3, dropout=layer_dropout[0],
                                                  lightgcn=True, use_cuda=use_cuda)

        if aggregator.startswith('narre'):
            self.w_0 = nn.Linear(node_dim, node_dim)

        if predictor == 'bi-interaction':
            self.add_mlp = nn.Sequential(
                nn.Linear(node_dim, node_dim),
                nn.Tanh()
            )
            self.mul_mlp = nn.Sequential(
                nn.Linear(node_dim, node_dim),
                nn.Tanh()
            )
        elif self.predictor == 'narre':
            self.edge_predictor = dgl.nn.EdgePredictor('ele', 2*node_dim, 1, bias=True)
            self.bias = nn.Parameter(torch.zeros((n_nodes, 1)))

        self.rating_loss_fn = nn.MSELoss(reduction='mean')
        self.bpr_loss_fn = nn.Softplus()
        self.review_embs = None
        self.inf_emb = None
        self.lemb = None
        self.first = True
        self.review_attention = None

    def get_initial_embedings(self, nodes):
        if hasattr(self, 'node_embedding'):
            return self.node_embedding(nodes)
        else:
            return self.node_embedding_mlp(self.learned_node_embeddings[nodes])

    def l2_loss(self, pos, neg, emb):
        if isinstance(emb, list):
            emb = torch.cat(emb, dim=-1)

        loss = 0
        src, dst_i = pos.edges()
        _, dst_j = neg.edges()

        s_emb, i_emb, j_emb = emb[src], emb[dst_i], emb[dst_j]

        loss += s_emb.norm(2).pow(2) + i_emb.norm(2).pow(2) + j_emb.norm(2).pow(2)

        loss = 0.5 * loss / pos.num_src_nodes()

        return loss

    def review_representation(self, g, x):
        return self.review_conv(g, x)

    def review_aggregation(self, g, x, attention=False):
        x = self.review_agg(g, x, attention)

        if attention:
            x, a = x

        x = x.sum(1)

        if attention:
            return x, a
        else:
            return x

    def forward(self, blocks, x):
        blocks, lgcn_blocks = blocks
        if self.narre_preference == 'lightgcn':
            lx = self.lightgcn(lgcn_blocks[0].ndata[dgl.NID], lgcn_blocks[:-1])
        else:
            # Get user/item representation without any graph convolutions.
            lx = {ntype: self.lightgcn.features[ntype](nids) for ntype, nids in
                  lgcn_blocks[-1].srcdata[dgl.NID].items() if ntype != 'node'}

        g = lgcn_blocks[-1]
        with g.local_scope():
            g.srcdata['h'] = lx
            funcs = {etype: (fn.copy_u('h', 'm'), fn.sum('m', 'h')) for etype in g.etypes}
            g.multi_update_all(funcs, 'sum')
            lx = g.dstdata['h']['node']

        x = self.node_dropout(x)
        x = self.review_representation(blocks[0], x)

        x = self.review_aggregation(blocks[1], x)

        x, lx = self.node_dropout(x), self.node_dropout(lx)

        if self.predictor == 'narre':
            x = torch.cat([x, lx], dim=-1)
        elif self.predictor == 'bi-interaction':
            x = [x, lx]
        else:
            x = x * lx

        return x

    def _graph_predict_dot(self, g: dgl.DGLGraph, x):
        with g.local_scope():
            g.ndata['h'] = x
            g.apply_edges(fn.u_dot_v('h', 'h', 'm'))

            return g.edata['m'].reshape(-1, 1)

    def _graph_predict_narre(self, g: dgl.DGLGraph, x):
        with g.local_scope():
            g.ndata['b'] = self.bias[g.ndata[dgl.NID]]
            g.apply_edges(fn.u_add_v('b', 'b', 'b'))  # user/item bias

            u, v = g.edges()
            x = self.edge_predictor(x[u], x[v])
            out = x + g.edata['b']

            return out

    def _graph_bi_interaction(self, g: dgl.DGLGraph, x):
        x, lx = x
        with g.local_scope():
            a = self.add_mlp(x + lx)
            m = self.mul_mlp(x * lx)
            x = a + m
            pred = self._graph_predict_dot(g, x)

            return pred

    def graph_predict(self, g: dgl.DGLGraph, x):
        if self.predictor == 'dot':
            return self._graph_predict_dot(g, x)
        elif self.predictor == 'narre':
            return self._graph_predict_narre(g, x)
        elif self.predictor == 'bi-interaction':
            return self._graph_bi_interaction(g, x)
        else:
            raise ValueError(f'Predictor not implemented for "{self.predictor}".')

    def _predict_dot(self, u_emb, i_emb):
        return (u_emb * i_emb).sum(-1)

    def _predict_narre(self, user, item, u_emb, i_emb):
        lu_emb, li_emb = self.lemb[user], self.lemb[item]

        u = torch.cat([u_emb, lu_emb], dim=-1)
        i = torch.cat([i_emb, li_emb], dim=-1)

        h = self.edge_predictor(u, i)
        h += (self.bias[user] + self.bias[item])

        return h.reshape(-1, 1)

    def _predict_bi_interaction(self, user, item, u_emb, i_emb):
        lu_emb, li_emb = self.lemb[user], self.lemb[item]
        u_emb = self.add_mlp(u_emb + lu_emb) + self.mul_mlp(u_emb * lu_emb)
        i_emb = self.add_mlp(i_emb + li_emb) + self.mul_mlp(i_emb * li_emb)
        pred = self._predict_dot(u_emb, i_emb)
        return pred

    def predict(self, user, item):
        u_emb, i_emb = self.inf_emb[user], self.inf_emb[item]

        if self.predictor == 'dot':
            u_emb, i_emb = u_emb * self.lemb[user], i_emb * self.lemb[item]
            pred = self._predict_dot(u_emb, i_emb)
        elif self.predictor == 'narre':
            pred = self._predict_narre(user, item, u_emb, i_emb)
        elif self.predictor == 'bi-interaction':
            pred = self._predict_bi_interaction(user, item, u_emb, i_emb)
        else:
            raise ValueError(f'Predictor not implemented for "{self.predictor}".')

        return pred

    def rating_loss(self, preds, target):
        return self.rating_loss_fn(preds, target.unsqueeze(-1))

    def _bpr_loss(self, preds_i, preds_j):
        return

    def ranking_loss(self, preds_i, preds_j, loss_fn='bpr'):
        if loss_fn == 'bpr':
            loss = self.bpr_loss_fn(- (preds_i - preds_j))
        else:
            raise NotImplementedError

        return loss.mean()

    def inference(self, review_graphs, node_review_graph, ui_graph, device, batch_size):
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
        self.review_attention = torch.zeros((node_review_graph.num_edges(), self.review_agg._num_heads, 1)).to(device)

        # Node inference
        for input_nodes, output_nodes, blocks in dataloader:
            x, a = self.review_aggregation(blocks[0]['part_of'], self.review_embs[input_nodes['review']], True)
            self.inf_emb[output_nodes['node']] = x
            self.review_attention[blocks[0]['part_of'].edata[dgl.EID]] = a

        # Node preference embedding
        if self.narre_preference == 'lightgcn':
            self.lightgcn.inference(ui_graph, batch_size)
            x = self.lightgcn.embeddings
        else:
            x = {nt: e.weight for nt, e in self.lightgcn.features.items()}

        x = torch.cat([x['item'], x['user']], dim=0)
        self.lemb = x




