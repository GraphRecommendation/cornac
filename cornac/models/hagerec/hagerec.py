
import dgl
import torch
from torch import nn


class HAGERecConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 att_feat,
                 num_heads,
                 att_agg='sum',
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 activation=None,
                 allow_zero_in_degree=False,
                 bias=True,
                 share_weights=False):
        super(HAGERecConv, self).__init__()
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = dgl.utils.expand_as_pair(in_feats)
        self._att_feats = att_feat
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree

        self.w1 = nn.Linear(self._in_src_feats, out_feats)

        # Splitting w2 of eq. 8 in two for computational efficiency.
        self.w2_src = nn.Linear(
            self._in_src_feats, att_feat * num_heads, bias=bias)
        self.w2_dst = nn.Linear(
            self._in_dst_feats, att_feat * num_heads, bias=bias)
        self.attn = nn.Parameter(torch.FloatTensor(size=(1, num_heads, att_feat)))
        self.att_agg = att_agg
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.activation = activation
        self.share_weights = share_weights
        self.bias = bias

    def _entity_propagation(self, lhs_field, att, edge, out):
        def func(edges):
            # eq. 7
            env = edges.src[lhs_field]
            rr2 = edges.data[edge]
            a = edges.data[att]
            return {out: torch.softmax(env * rr2 * a, dim=-1)}
        return func

    def forward(self, g, feat, r_feat, get_attention=False, get_neighborhood=False):
        with g.local_scope():
            # Get features
            h_src, h_dst = dgl.utils.expand_as_pair(feat, g)
            h_src, h_dst = self.feat_drop(h_src), self.feat_drop(h_dst)

            # eq. 8
            feat_src = self.w2_src(h_src).view(-1, self._num_heads, self._att_feats)
            feat_dst = self.w2_dst(h_dst).view(-1, self._num_heads, self._att_feats)

            g.srcdata.update({'en': feat_src})  # neighbors
            g.dstdata.update({'ev': feat_dst})  # self
            g.edata.update({'rr': r_feat})  # relation

            g.apply_edges(dgl.function.u_add_v('en', 'ev', 'a'))
            e = self.leaky_relu(g.edata.pop('a'))  # (num_src_edge, num_heads, out_dim)
            e = (e * self.attn).sum(dim=-1).unsqueeze(dim=2)  # (num_edge, num_heads, 1)
            # compute softmax, eq. 9
            g.edata['a'] = self.attn_drop(dgl.ops.edge_softmax(g, e))  # (num_edge, num_heads)
            # message passing
            # eq.6/7
            g.update_all(self._entity_propagation('en', 'a', 'rr', 'm'),
                         dgl.function.sum('m', 'g'))

            if self.att_agg == 'sum':
                g.dstdata.update({'g': torch.sum(g.dstdata['g'], dim=1)})
            elif self.att_agg == 'mean':
                g.dstdata.update({'g': torch.mean(g.dstdata['g'], dim=1)})
            else:
                raise NotImplementedError

            # eq. 4/5
            rst = self.w1(h_dst + g.dstdata['g'])

            # activation
            if self.activation:
                rst = self.activation(rst)

            if get_attention:
                rst = rst, g.edata['a']

            if get_neighborhood:
                if isinstance(rst, tuple):
                    rst = *rst, g.dstdata['g']
                else:
                    rst = rst, g.dstdata['g']

            return rst


class InteractionSignalsUnit(nn.Module):
    def __init__(self):
        super(InteractionSignalsUnit, self).__init__()

    def forward(self, g, feat):
        pass


class Model(nn.Module):
    def __init__(self, n_nodes, n_relations, embed_dim, n_layers, num_heads, feat_dropout, edge_dropout,
                 use_sigmoid=False):
        super(Model, self).__init__()

        self.node_embedding = nn.Embedding(n_nodes, embed_dim)
        self.relation_embedding = nn.Embedding(n_relations, embed_dim)

        self.hagerec_convs = nn.ModuleList(
            [HAGERecConv(embed_dim, embed_dim, embed_dim, num_heads, feat_dropout, edge_dropout,
                         activation=nn.LeakyReLU()) for _ in range(n_layers)]
        )

        self.w4 = nn.Linear(embed_dim, embed_dim)
        self.w5 = nn.Linear(embed_dim, embed_dim)
        self.att = nn.Linear
        self.activation = nn.LeakyReLU()
        self.sigmoid = None

        if use_sigmoid:
            self.sigmoid = nn.Sigmoid()

    def forward(self, blocks, x):
        n_out_nodes = blocks[-1].num_dst_nodes()
        emb = []
        g = None
        attentions = None
        for i, (layer, block) in enumerate(zip(self.hagerec_convs, blocks)):
            rel_emb = self.relation_embedding(blocks.edata['r_type'])
            if i + 1 == len(blocks):
                x, g = layer(block, x, rel_emb, get_neighborhood=True)
            elif i == 0:
                x, attentions = layer(block, x, rel_emb, get_attention=True)
            else:
                x = layer(block, x, rel_emb)

            emb.append(x[:n_out_nodes])

        attentions = attentions.sum(dim=-1).squeeze(0)

        return x, torch.cat(emb), g, attentions

    def aggregation(self, g, x, gn):
        with g.local_scope():
            src_feats, dst_feats = dgl.utils.expand_as_pair(x, g)
            g_src_feats, g_dst_feats = dgl.utils.expand_as_pair(gn, g)

            # eq 20, 21
            v_feats = self.activation(self.w4(dst_feats + g_dst_feats))
            u_feats = self.activation(self.w5(src_feats + g_src_feats))

            return u_feats, v_feats

    def graph_predict(self, g, src_feats, dst_feats, agg_feat):
        with g.local_scope():
            # Get features and assign to graph
            src_agg, dst_agg = dgl.utils.expand_as_pair(agg_feat)
            g.srcdata.update({'u': src_feats, 'ua': src_agg})
            g.dstdata.update({'v': dst_feats, 'va': dst_agg})

            # Assume that the attention is the similarity based on stacked embedding of each conv.
            g.apply_edges(dgl.function.u_dot_v('ua', 'va', 'a'))

            # Normal dot product, eq. 23.
            g.apply_edges(dgl.function.u_dot_v('u', 'v', 'y_hat'))

            # Use the attention, eq. 24.
            preds = g.edata['a'] * g.edata['y_hat']

            # Apply sigmoid. Not a good idea for rating prediction as a rating can over 1.
            if self.sigmoid:
                preds = self.sigmoid(preds)

            return preds
