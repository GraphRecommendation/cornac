import itertools

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

        # Used in eq 18/19.
        # Assumption: the w4/w5 is because of different layers.
        self.w4 = nn.Linear(in_feats, out_feats)
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
            shape = env.shape
            env = torch.repeat_interleave(env, self._num_heads, dim=0)
            env = env.reshape(shape[0], self._num_heads, -1)

            rr2 = edges.data[edge]
            a = edges.data[att]
            return {out: torch.softmax(env * rr2.unsqueeze(1) * a, dim=-1)}
        return func

    def calculate_attention(self, g, src, dst, raw_att, att):
        with g.local_scope():
            # eq 8.
            feat_src = self.w2_src(src).view(-1, self._num_heads, self._att_feats)
            feat_dst = self.w2_dst(dst).view(-1, self._num_heads, self._att_feats)

            g.srcdata.update({'en': feat_src})  # neighbors
            g.dstdata.update({'ev': feat_dst})  # self
            g.apply_edges(dgl.function.u_add_v('en', 'ev', 'a'))
            e = self.leaky_relu(g.edata.pop('a'))  # (num_src_edge, num_heads, out_dim)
            e = (e * self.attn).sum(dim=-1).unsqueeze(dim=2)  # (num_edge, num_heads, 1)
            return {raw_att: e, att: self.attn_drop(dgl.ops.edge_softmax(g, e))}

    def forward(self, g, feat, r_feat, get_attention=False, get_neighborhood=False):
        if not g.num_edges():
            return []

        with g.local_scope():
            # Get features
            h_src, h_dst = dgl.utils.expand_as_pair(feat, g)
            h_src, h_dst = self.feat_drop(h_src), self.feat_drop(h_dst)

            g.edata.update(self.calculate_attention(g, h_src, h_dst, 'ra', 'a'))

            g.srcdata.update({'en': h_src})
            g.edata.update({'rr': r_feat})  # relation
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

            # eq. 18/19
            interaction = self.w4(torch.mul(h_dst, g.dstdata['g']))

            # activation
            if self.activation:
                rst = self.activation(rst)
                interaction = self.activation(interaction)

            # eq. 20/21
            rst = rst + interaction

            if get_attention:
                if self.training:
                    rst = rst, g.edata['ra']  # no dropout and softmax as not all edges in batch.
                else:
                    rst = rst, g.edata['a']

            if get_neighborhood:
                if isinstance(rst, tuple):
                    rst = *rst, g.dstdata['g']
                else:
                    rst = rst, g.dstdata['g']

            return rst


class Model(nn.Module):
    def __init__(self, n_nodes, n_relations, etypes, embed_dim, layer_dims, num_heads, feat_dropout, edge_dropout,
                 use_sigmoid=False):
        super(Model, self).__init__()

        self.node_embedding = nn.Embedding(n_nodes, embed_dim)
        self.relation_embedding = nn.Embedding(n_relations, embed_dim)

        self.mlps = nn.ModuleList()
        self.hagerec_convs = nn.ModuleList()

        in_dim = embed_dim
        for out_dim in layer_dims:
            self.mlps.append(nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.LeakyReLU()
            ))
            self.hagerec_convs.append(
                dgl.nn.HeteroGraphConv({
                    etype: HAGERecConv(in_dim, out_dim, out_dim, num_heads=num_heads, feat_drop=feat_dropout,
                                       attn_drop=edge_dropout, activation=nn.LeakyReLU())
                    for etype in etypes
                })
            )
            in_dim = out_dim

        self.att = nn.Linear
        self.activation = nn.LeakyReLU()
        self.sigmoid = None

        if use_sigmoid:
            self.sigmoid = nn.Sigmoid()

        self.loss_fn = nn.MSELoss()
        self.inf_emb = None
        self.agg_emb = None

    def reset_parameters(self):
        for name, parameter in self.named_parameters():
            if name.endswith('bias'):
                nn.init.constant_(parameter, 0)
            else:
                nn.init.xavier_normal_(parameter)

    def _block_aggregator(self, g, x):
        with g.local_scope():
            for ntype in g.ntypes:
                g.srcnodes[ntype].data.update({'h': x[ntype]})

            g.multi_update_all({
                etype: (dgl.function.copy_u('h', 'm'), dgl.function.sum('m', 'h')) for etype in g.etypes
            }, 'sum')

            return {ntype: g.dstnodes[ntype].data['h'] for ntype in g.ntypes}

    def forward(self, blocks, x):
        n_out_nodes = blocks[-1].num_dst_nodes()
        attentions = None
        rel_weight = self.relation_embedding.weight
        blocks = [blocks[i:i+2] for i in range(0, len(blocks), 2)]
        for i, (mlp, gcn, (block, agg_block)) in enumerate(zip(self.mlps, self.hagerec_convs, blocks)):
            h = {ntype: mlp(x[ntype][:block.num_dst_nodes(ntype)]) for ntype in block.ntypes}
            h.update(gcn(block,
                    x, {etype: (rel_weight[block.edges[etype].data['type']],) for etype in block.etypes}))
            rel_weight = mlp(rel_weight)
            x = self._block_aggregator(agg_block, h)

        return x

    # def aggregation(self, g, x, gn):
    #     # Assumption: We assume only the last layers output is used for the interaction signals unit.
    #     with g.local_scope():
    #         src_feats, dst_feats = dgl.utils.expand_as_pair(x, g)
    #         g_src_feats, g_dst_feats = dgl.utils.expand_as_pair(gn, g)
    #
    #         # eq 20, 21
    #         v_feats = self.activation(self.w4(dst_feats + g_dst_feats))
    #         u_feats = self.activation(self.w5(src_feats + g_src_feats))
    #
    #         return u_feats, v_feats

    def graph_predict(self, g, x):
        with g.local_scope():
            # Get features and assign to graph
            src_feats, dst_feats = dgl.utils.expand_as_pair(x, g)

            for sntype, etype, dntype in g.canonical_etypes:
                g.srcnodes[sntype].data.update({'u': src_feats[sntype]})
                g.dstnodes[dntype].data.update({'v': dst_feats[dntype]})

            # Assume that the attention is the similarity based on stacked embedding of each conv.
            # g.apply_edges(dgl.function.u_dot_v('ua', 'va', 'a'))

            # Normal dot product, eq. 23.
            g.apply_edges(dgl.function.u_dot_v('u', 'v', 'y_hat'))

            preds = g.edata['y_hat']

            # Use the attention, eq. 24.
            # preds = g.edata['a'] * g.edata['y_hat']

            # Apply sigmoid. Not a good idea for rating prediction as a rating can over 1.
            if self.sigmoid:
                preds = self.sigmoid(preds)

            return preds

    def predict(self, user, item):
        p = self.inf_emb[user].dot(self.inf_emb[item])

        if self.sigmoid:
            p = self.sigmoid(p)

        return p

    def loss(self, pred, target):
        return self.loss_fn(pred, target)

    def inference(self, g, fanout, device, batch_size, *args):
        from . import dgl_utils
        # g = g.to(device)
        # Calculate attention
        emb = {ntype: self.node_embedding(g.nodes(ntype=ntype).to(device)) for ntype in g.ntypes}
        # a = self.hagerec_convs[0].calculate_attention(g.to(device), emb, emb, 'ra', 'a')['a']
        # a = a.sum(1).squeeze(-1)
        # g.edata['a'] = a.to(g.edata['a'].device)
        nodes = {ntype: g.nodes(ntype) for ntype in ['user', 'item']}
        nodes['user'] = nodes['user'][nodes['user'] > max(nodes['item'])]

        g = dgl.sampling.select_topk(g, fanout, 'a')
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
        dataloader = dgl.dataloading.DataLoader(g, nodes, sampler,
                                                batch_size=batch_size, shuffle=False, drop_last=True, device=device)

        agg_feats = []
        rel_weight = self.relation_embedding.weight
        for mlp, gcn in zip(self.mlps, self.hagerec_convs):
            next_emb = {ntype: mlp(emb[ntype]) for ntype in g.ntypes}
            for input_nodes, output_nodes, (block,) in dataloader:
                block = dgl.edge_type_subgraph(block, [etype for etype in block.etypes if block.num_edges(etype=etype)])

                rel_emb = {etype: (rel_weight[block.edges[etype].data['type']],) for etype in block.etypes}
                in_emb = {ntype: emb[ntype][input_nodes[ntype]] for ntype in input_nodes}

                for ntype, ne in gcn(block, in_emb, rel_emb).items():
                    next_emb[ntype][output_nodes[ntype]] = ne

            rel_weight = mlp(rel_weight)
            agg_feats.append(next_emb)
            emb = next_emb

        self.inf_emb = emb['node']
        self.inf_emb[:emb['user'].shape[0]] = emb['user']
        self.inf_emb[:emb['item'].shape[0]] = emb['item']
        # return a
