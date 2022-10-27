from typing import Mapping

import dgl.dataloading
import torch
from dgl import DGLError, function


class HEAREdgeSampler(dgl.dataloading.EdgePredictionSampler):
    def __init__(self, sampler, exclude=None, reverse_eids=None,
                 reverse_etypes=None, negative_sampler=None, prefetch_labels=None):
        super().__init__(sampler, exclude, reverse_eids, reverse_etypes, negative_sampler,
                         prefetch_labels)

    def sample(self, g, seed_edges):    # pylint: disable=arguments-differ
        """Samples a list of blocks, as well as a subgraph containing the sampled
        edges from the original graph.

        If :attr:`negative_sampler` is given, also returns another graph containing the
        negative pairs as edges.
        """
        # sg = dgl.sampling.sample_neighbors(g, seed_edges, 1)
        # seed_edges = sg.edata[dgl.EID]

        if isinstance(seed_edges, Mapping):
            seed_edges = {g.to_canonical_etype(k): v for k, v in seed_edges.items()}
        exclude = self.exclude
        pair_graph = g.edge_subgraph(
            seed_edges, relabel_nodes=False, output_device=self.output_device)
        eids = pair_graph.edata[dgl.EID]

        if self.negative_sampler is not None:
            neg_graph = self._build_neg_graph(g, seed_edges)
            pair_graph, neg_graph = dgl.compact_graphs([pair_graph, neg_graph])
        else:
            pair_graph = dgl.compact_graphs(pair_graph)

        pair_graph.edata[dgl.EID] = eids
        seed_nodes = pair_graph.ndata[dgl.NID]

        exclude_eids = dgl.dataloading.find_exclude_eids(
            g, seed_edges, exclude, self.reverse_eids, self.reverse_etypes,
            self.output_device)

        input_nodes, _, blocks = self.sampler.sample(g, seed_nodes, exclude_eids, pair_graph)

        if self.negative_sampler is None:
            return self.assign_lazy_features((input_nodes, pair_graph, blocks))
        else:
            return self.assign_lazy_features((input_nodes, pair_graph, neg_graph, blocks))


class HearBlockSampler(dgl.dataloading.NeighborSampler):
    def __init__(self, node_review_graph, review_graphs, aggregator, compact=True, fanout=5, **kwargs):
        """
        Given nodes, samples reviews and creates a batched review-graph of all sampled reviews.
        Parameters
        ----------
        node_review_graph: DGLHeteroGraph
            A heterogeneous graph with edges from reviews to nodes (users/items) with relation-type part_of.
        review_graphs: dict[DGLGraph]
            A dictionary with sid to a graph representing a review based on sentiment.
        fanouts: int
            Number of reviews to sample per node.
        kwargs: dict
            Arguments to pass to NeighborSampler. Read DGL docs for options.
        """
        fanouts = [fanout]

        super().__init__(fanouts, **kwargs)
        self.node_review_graph = node_review_graph
        self.review_graphs = review_graphs
        self.aggregator = aggregator
        self.compact = compact

    def sample(self, g, seed_nodes, exclude_eids=None, pair_graph=None):
        if exclude_eids is not None:
            u, v = g.find_edges(exclude_eids)
            rid = g.edata['rid'][exclude_eids]
            exclude_eids = self.node_review_graph.edge_ids(rid, u, etype='part_of')
        g2 = g

        input_nodes, output_nodes, blocks = super().sample(self.node_review_graph, {'node': seed_nodes}, exclude_eids)
        b = blocks[0]

        narre_flag = pair_graph is not None and self.aggregator == 'narre'

        # If narre have graph with ui to rid.
        if narre_flag:
            u, v = pair_graph.edges()
            org = pair_graph.ndata[dgl.NID]
            u, v = org[u], org[v]
            graphs = []
            for org_user, org_item in zip(u,v):
                srcs, dsts = [], []
                info = []
                nids = b.dstnodes('node')
                user = nids[b.dstnodes['node'].data[dgl.NID] == org_user][0]
                item = nids[b.dstnodes['node'].data[dgl.NID] == org_item][0]

                for nid, related in [[user, item], [item, user]]:
                    src, dst = b.in_edges(nid)
                    srcs.append(src)
                    dsts.append(dst)
                    info.append(torch.full((len(src),), related))

                srcs, dsts = torch.cat(srcs), torch.cat(dsts)
                assert len(srcs)
                assert len(dsts)
                assert len(srcs) == len(dsts)

                g = dgl.heterograph(
                    {('review', 'part_of', 'node'): (torch.arange(len(srcs)), dsts)},
                    num_nodes_dict={ntype: b.num_nodes(ntype=ntype) for ntype in b.ntypes}
                )

                g = dgl.compact_graphs(g)

                info = torch.cat(info)
                g.srcnodes['review'].data[dgl.NID] = srcs
                g.srcnodes['review'].data['nid'] = info
                g.dstnodes['node'].data[dgl.NID] = torch.LongTensor([org_user, org_item])

                graphs.append(g)

        b = b['part_of']
        blocks[0] = b

        assert torch.all(b.in_degrees(b.dstnodes()) != 0)

        r_gs = [self.review_graphs[rid] for rid in b.srcdata[dgl.NID].cpu().numpy()]
        if self.compact:
            r_gs = dgl.compact_graphs(r_gs)
            nid = [bg.ndata[dgl.NID] for bg in r_gs]
        else:
            nid = [bg.nodes() for bg in r_gs]

        batch = dgl.batch(r_gs)

        # Get original ids
        nid = torch.cat(nid)
        batch.ndata[dgl.NID] = nid

        blocks.insert(0, batch)
        input_nodes = nid

        if narre_flag:
            bg = dgl.batch(graphs)
            blocks[1] = bg

        return input_nodes, output_nodes, blocks


class HearReviewDataset(torch.utils.data.Dataset):
    def __init__(self, review_graphs):
        self.review_graphs = review_graphs

    def __len__(self):
        return len(self.review_graphs)

    def	__getitem__(self, idx):
        return self.review_graphs[idx], idx


class HearReviewSampler(torch.utils.data.Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.data_source)


class HearReviewCollator(dgl.dataloading.GraphCollator):
    def collate(self, items):
        elem = items[0]
        if isinstance(elem, dgl.DGLHeteroGraph):
            input_nodes = [g.nodes() for g in items]
            input_nodes = torch.cat(input_nodes)
            batched_graph = super().collate(items)
            return input_nodes, batched_graph
        else:
            return super(HearReviewCollator, self).collate(items)


class GATv2NARREConv(dgl.nn.GATv2Conv):
    def __init__(self, aggregator, **kwargs):
        super(GATv2NARREConv, self).__init__(**kwargs)
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
                    raise DGLError('There are 0-in-degree nodes in the graph, '
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
                    feat_dst = feat_src[:graph.number_of_dst_nodes()]
                    h_dst = h_dst[:graph.number_of_dst_nodes()]
            graph.srcdata.update({'el': feat_src})  # (num_src_edge, num_heads, out_dim)
            graph.dstdata.update({'er': feat_dst})
            graph.apply_edges(function.u_add_v('el', 'er', 'e'))
            e = self.leaky_relu(graph.edata.pop('e'))  # (num_src_edge, num_heads, out_dim)
            e = (e * self.attn).sum(dim=-1).unsqueeze(dim=2)  # (num_edge, num_heads, 1)
            # compute softmax
            graph.edata['a'] = self.attn_drop(dgl.utils.edge_softmax(graph, e))  # (num_edge, num_heads)
            # message passing
            graph.update_all(function.u_mul_e('el', 'a', 'm'),
                             function.sum('m', 'ft'))
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
                return

