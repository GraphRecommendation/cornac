"""
Based on https://github.com/dmlc/dgl/tree/master/examples/pytorch/NGCF modified to include LightGCN and using blocks.
"""

import dgl
import dgl.function as fn
import torch
import torch.nn as nn

from cornac.models.ngcf.ngcf import NGCFLayer


class ReviewRepresentationConv(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, g, x):
        with g.local_scope():
            g.srcdata.update({'h': x})
            funcs = {etype: (dgl.function.copy_u('h', 'm'), dgl.function.mean('m', 'h'))
                     for stype, etype, _ in g.canonical_etypes if stype in x}
            g.multi_update_all(funcs, 'mean')

            return g.dstdata['h']


class ReviewAggregatorConv(nn.Module):
    def __init__(self, in_dim, att_dim, n_nodes):
        super().__init__()
        self.w_o = nn.Linear(in_dim, att_dim)
        self.w_u = nn.Linear(in_dim, att_dim)
        self.w_g = nn.Linear(in_dim, att_dim)
        self.att = nn.Linear(att_dim, 1)
        self.activation = nn.LeakyReLU()
        self.node_preference = nn.ModuleDict(
            {ntype: nn.Embedding(n_nodes[ntype], in_dim) for ntype in n_nodes}
        )

    def forward(self, g: dgl.DGLGraph, x: torch.Tensor):
        with g.local_scope():
            src_feats, _ = dgl.utils.expand_as_pair(x, g)
            g.srcdata.update({'h':  {ntype: self.w_o(src_feats[ntype])
                                            + self.w_g(src_feats[ntype].sum(dim=0, keepdims=True))  # Global attension
                                     for ntype in src_feats}})
            for _, etype, dsttype in g.canonical_etypes:
                if g[etype].num_edges():
                    if dsttype == 'user':
                        g[etype].edata.update({'h': self.w_u(self.node_preference['item'](g[etype].edata['npid']))})
                    else:
                        g[etype].edata.update({'h': self.w_u(self.node_preference['user'](g[etype].edata['npid']))})
                    g[etype].apply_edges(dgl.function.u_add_e('h', 'h', 'h'))
                    g[etype].edata['a'] = self.att(self.activation(g[etype].edata.pop('h')))

                    # Softmax
                    g[etype].edata['a'] = dgl.ops.edge_softmax(g[etype], g[etype].edata['a'])

            g.srcdata.update({'h': src_feats})
            funcs = {etype: (dgl.function.u_mul_e('h', 'a', 'm'), dgl.function.sum('m', 'h')) for etype in g.etypes
                     if g[etype].num_edges()}
            g.multi_update_all(funcs, 'mean')

            return g.dstdata['h']


class Model(nn.Module):
    def __init__(self, g, in_dim, layer_dims, dropout, use_cuda=False):
        super(Model, self).__init__()
        self.layer_dims = layer_dims
        self.use_cuda = use_cuda
        self.lightgcn = True

        self.features = nn.ModuleDict()
        for ntype in g.ntypes:
            e = torch.nn.Embedding(g.number_of_nodes(ntype=ntype), in_dim)
            self.features[ntype] = e

        self.layers = nn.ModuleList()
        for out_dim in layer_dims:
            self.layers.append(NGCFLayer(in_dim, out_dim, dropout, lightgcn=self.lightgcn))
            in_dim = out_dim

        self.representation_conv = ReviewRepresentationConv()
        self.review_conv = ReviewAggregatorConv(layer_dims[-1], layer_dims[-1],
                                                {ntype: g.num_nodes(ntype) for ntype in ['user', 'item']})

        self.loss_fn = torch.nn.LogSigmoid()

        self.embeddings = None

    def reset_parameters(self):
        for name, parameter in self.named_parameters():
            if name.endswith('bias'):
                nn.init.constant_(parameter, 0)
            else:
                nn.init.xavier_normal_(parameter)

    def forward(self, node_ids, blocks):
        dst_nodes = {ntype: blocks[-3].dstnodes(ntype) for ntype in set(blocks[-3].ntypes)} # Get final nodes
        x = {ntype: self.features[ntype](nids) for ntype, nids in node_ids.items()} # get initial embeddings
        embeddings = {ntype: [embedding[dst_nodes[ntype]]] for ntype, embedding in x.items()}  # get first embeddings

        for i, (block, layer) in enumerate(zip(blocks, self.layers)):
            x = layer(block, x)
            for ntype, embedding in x.items():
                embeddings[ntype].append(embedding[dst_nodes[ntype]])

        # Concatenate layer embeddings for each node type
        for ntype, embedding in embeddings.items():
            if self.lightgcn:
                embeddings[ntype] = torch.mean(torch.stack(embedding), dim=0)
            else:
                embeddings[ntype] = torch.cat(embedding, 1)

        dst_nodes = {ntype: blocks[-1].dstnodes(ntype) for ntype in ['user', 'item']}
        graph_emb = {ntype: embeddings[ntype][dst_nodes[ntype]] for ntype in dst_nodes}
        embeddings = self.representation_conv(blocks[-2], embeddings)
        embeddings = self.review_conv(blocks[-1], embeddings)

        return {ntype: graph_emb[ntype] + embeddings[ntype] for ntype in graph_emb}

    def inference(self, g: dgl.DGLGraph, other_g: dgl.DGLGraph, batch_size=None):
        embeddings = {ntype: f.weight for ntype, f in self.features.items()}
        device = embeddings[next(iter(embeddings))].device
        all_embeddings = {ntype: [embedding] for ntype, embedding in embeddings.items()}

        for l, layer in enumerate(self.layers):
            if batch_size is not None:
                next_embeddings = {ntype: torch.zeros((g.number_of_nodes(ntype), self.layer_dims[l]),
                                                      device=device)
                                   for ntype, embedding in embeddings.items()}

                sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
                dataloader = dgl.dataloading.NodeDataLoader(
                    g,
                    {k: torch.arange(g.number_of_nodes(k)) for k in g.ntypes},
                    sampler,
                    batch_size=batch_size,
                    shuffle=True,
                    drop_last=False, device=device)

                # Within a layer, iterate over nodes in batches
                for input_nodes, output_nodes, blocks in dataloader:
                    block = blocks[0]

                    if self.use_cuda:
                        block = block.to('cuda')

                    embedding = {k: embeddings[k][input_nodes[k]] for k in input_nodes.keys()}
                    embedding = layer(block, embedding)

                    for ntype, nodes in output_nodes.items():
                        next_embeddings[ntype][nodes] = embedding[ntype]
            else:
                next_embeddings = layer(g, embeddings)

            embeddings = next_embeddings

            for ntype, embedding in embeddings.items():
                all_embeddings[ntype].append(embedding)

        g1 = g
        # Concatenate layer embeddings for each node type
        for ntype, embedding in all_embeddings.items():
            if self.lightgcn:
                all_embeddings[ntype] = torch.mean(torch.stack(embedding), dim=0)
            else:
                all_embeddings[ntype] = torch.cat(embedding, 1)

        g = dgl.edge_type_subgraph(other_g, [etype for stype, etype, dtype in other_g.canonical_etypes if dtype == 'review'])
        emb = self.representation_conv(dgl.to_block(g, include_dst_in_src=False).to(device), all_embeddings)

        g = dgl.edge_type_subgraph(other_g, [etype for stype, etype, dtype in other_g.canonical_etypes
                                             if dtype in ['user', 'item']])
        emb2 = self.review_conv(dgl.to_block(g, include_dst_in_src=False).to(device), emb)

        emb2 = {ntype: all_embeddings[ntype] + emb2[ntype] for ntype in emb2}

        # test
        u, v = g1.edges(etype='ui')
        v_r = v[torch.randperm(len(v))]
        print("")
        self_sim = torch.std(all_embeddings['user'])
        self_sim2 = torch.std(emb['review'])
        self_sim3 = torch.std(emb2['user'])
        print(f'All usim: {self_sim}, rsim: {self_sim2}, end usim: {self_sim3}')
        print(f'All sim: {(all_embeddings["user"][u] * all_embeddings["item"][v]).sum(dim=1).mean():.5f}'
              f', rand: {(all_embeddings["user"][u] * all_embeddings["item"][v_r]).sum(dim=1).mean():.5f}')
        print(f'End sim: {(emb2["user"][u] * emb2["item"][v]).sum(dim=1).mean():.5f}'
              f', rand: {(emb2["user"][u] * emb2["item"][v_r]).sum(dim=1).mean():.5f}')
        self.embeddings = emb2

    def predict(self, users, items, rank_all=False):
        if self.use_cuda:
            users, items = users.cuda(), items.cuda()

        user_embs = self.embeddings['user'][users]
        item_embs = self.embeddings['item'][items]

        if rank_all:
            predictions = torch.matmul(user_embs, item_embs.T)
        else:
            predictions = (user_embs * item_embs).sum(dim=1)

        return predictions

    def graph_predict(self, g: dgl.DGLHeteroGraph, embeddings):
        with g.local_scope():
            users, items = g.edges(etype=('ui'))

            return torch.mul(embeddings['user'][users], embeddings['item'][items]).sum(dim=1)

    def l2_loss(self, pos, neg, emb):
        users, items_i = pos.edges(etype=('ui'))
        _, items_j = neg.edges(etype=('ui'))

        u_emb, i_emb, j_emb = emb['user'][users], emb['item'][items_i],emb['item'][items_j]

        loss = (1/2)*(u_emb.norm(2).pow(2) + i_emb.norm(2).pow(2) + j_emb.norm(2).pow(2)) / float(len(users))

        return loss

    def ranking_loss(self, preds_i, preds_j):
        diff = preds_i - preds_j
        s = self.loss_fn(diff)
        m = torch.mean(s)
        loss = -m
        return loss

