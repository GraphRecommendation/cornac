from collections import OrderedDict, Counter
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

        input_nodes, _, (pos_aos, neg_aos), blocks = self.sampler.sample(g, seed_nodes, exclude_eids, seed_edges)
        pair_graph.edata['pos'] = pos_aos.to(pair_graph.device)
        pair_graph.edata['neg'] = neg_aos.to(pair_graph.device)

        if self.negative_sampler is None:
            return self.assign_lazy_features((input_nodes, pair_graph, blocks))
        else:
            return self.assign_lazy_features((input_nodes, pair_graph, neg_graph, blocks))


class HearBlockSampler(dgl.dataloading.NeighborSampler):
    def __init__(self, node_review_graph, review_graphs, aggregator, sid_aos, aos_list, n_neg, ui_graph,
                 compact=True, hard_negatives=False, fanout=5, **kwargs):
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
        self.sid_aos = sid_aos
        self.aos_list = torch.LongTensor(aos_list)
        ac = Counter([a for aos in sid_aos for a in aos.numpy()])
        self.aos_probabilities = torch.log(torch.FloatTensor([ac.get(a) for a in sorted(ac)]) + 1)
        self.n_neg = n_neg
        self.ui_graph = ui_graph
        self.compact = compact
        self.hard_negatives = hard_negatives
        self.n_ui_graph = self._nu_graph()

    def _nu_graph(self):
        n_nodes = self.node_review_graph.num_nodes('node')
        n_users = self.ui_graph.num_nodes('user')
        n_items = self.ui_graph.num_nodes('item')

        nodes = self.node_review_graph.nodes('node')
        device = nodes.device
        nodes = nodes.cpu()
        data = {
            ('user', 'un', 'node'): (torch.arange(n_users, dtype=torch.int64), nodes[nodes >= n_items]),
            ('item', 'in', 'node'): (torch.arange(n_items, dtype=torch.int64), nodes[nodes < n_items])
        }

        return dgl.heterograph(data, num_nodes_dict={'user': n_users, 'item': n_items, 'node': n_nodes}).to(device)

    def sample(self, g, seed_nodes, exclude_eids=None, seed_edges=None):
        # If exclude eids, find the equivalent eid of the node_review_graph.
        nrg_exclude_eids = None
        lgcn_exclude_eids = None
        if exclude_eids is not None:
            u, v = g.find_edges(exclude_eids)
            sid = g.edata['sid'][exclude_eids].to(u.device)
            nrg_exclude_eids = self.node_review_graph.edge_ids(sid, u, etype='part_of')
            lgcn_exclude_eids = dgl.dataloading.find_exclude_eids(
                self.ui_graph, {'ui': seed_edges}, 'reverse_types', None, {'ui': 'iu', 'iu': 'ui'},
                self.output_device)

        # Based on seed_nodes, find reviews to represent the nodes.
        input_nodes, output_nodes, blocks = super().sample(self.node_review_graph, {'node': seed_nodes},
                                                           nrg_exclude_eids)
        block = blocks[0]

        block = block['part_of']
        blocks[0] = block

        # If all nodes are removed, add random blocks/random reviews.
        # Will not occur during inference.
        if torch.any(block.in_degrees(block.dstnodes()) == 0):
            for index in torch.where(block.in_degrees(block.dstnodes()) == 0)[0]:
                perm = torch.randperm(block.num_src_nodes())
                block.add_edges(block.srcnodes()[perm[:self.fanouts[0]]],
                                index.repeat(min(self.fanouts[0], block.num_src_nodes())))

        r_gs = [self.review_graphs[sid] for sid in block.srcdata[dgl.NID].cpu().numpy()]
        if self.compact:
            r_gs = dgl.compact_graphs(r_gs)
            nid = [bg.ndata[dgl.NID] for bg in r_gs]
        else:
            nid = [bg.nodes() for bg in r_gs]

        # batch = dgl.batch(r_gs)
        #
        # # Get original ids
        # nid = torch.cat(nid)
        # batch.ndata[dgl.NID] = nid
        bz = len(exclude_eids) if exclude_eids is not None else 256
        n_batches = len(r_gs) // bz
        n_batches += 1 if len(r_gs) % bz != 0 else 0
        batch = [dgl.batch(r_gs[i*bz:(i+1)*bz]) for i in range(n_batches)]
        nid = [torch.cat(nid[i*bz:(i+1)*bz]) for i in range(n_batches)]

        blocks.insert(0, batch)
        input_nodes = nid

        blocks2 = []
        seed_nodes = output_nodes

        for i in range(4):
            if i == 0:
                frontier = self.n_ui_graph.sample_neighbors(
                    seed_nodes, -1, edge_dir=self.edge_dir, prob=self.prob,
                    replace=self.replace, output_device=self.output_device,
                    exclude_edges=None)
            else:
                frontier = self.ui_graph.sample_neighbors(
                    seed_nodes, -1, edge_dir=self.edge_dir, prob=self.prob,
                    replace=self.replace, output_device=self.output_device,
                    exclude_edges=lgcn_exclude_eids)
            eid = frontier.edata[dgl.EID]
            block = dgl.to_block(frontier, seed_nodes)
            block.edata[dgl.EID] = eid
            seed_nodes = block.srcdata[dgl.NID]
            blocks2.insert(0, block)

        # sample aos pos and negative pair.
        if self.hard_negatives:
            neg_aos = torch.multinomial(self.aos_probabilities, len(exclude_eids) * self.n_neg, replacement=True)\
                .reshape(len(exclude_eids), self.n_neg)
        else:
            neg_aos = torch.randint(len(self.aos_probabilities), size=(len(exclude_eids), self.n_neg))

        pos_aos = []
        for sid in g.edata['sid'][exclude_eids].cpu().numpy():
            aosid = self.sid_aos[sid]
            pos_aos.append(aosid[torch.randperm(len(aosid))[0]])

        pos_aos = torch.LongTensor(pos_aos)

        pos_aos, neg_aos = self.aos_list[pos_aos], self.aos_list[neg_aos]

        return input_nodes, output_nodes, [pos_aos, neg_aos], [blocks, blocks2]


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


class TranRBlockSampler(dgl.dataloading.NeighborSampler):
    def __init__(self, review_graphs, sentiment_modality, node_filter, ntype_range, **kwargs):
        super().__init__([-1], **kwargs)
        self.review_graphs = review_graphs
        self.node_filter = node_filter
        self.ntype_range = ntype_range
        self.rui_a_graph, self.n_rui_graph = self._construct_rui_ao_graph(sentiment_modality)

    def _construct_rui_aspect_graph(self):
        node_data = OrderedDict()
        edges = []
        cur_i = 0
        for r, g in self.review_graphs.items():
            nids, _ = g.edges()
            nids = torch.unique(nids)
            user = nids[self.node_filter('user', nids)][0].item()
            item = nids[self.node_filter('item', nids)][0].item()
            aspects = nids[self.node_filter('aspect', nids)].numpy() - self.ntype_range['aspect'][0]
            for aspect in aspects:
                node = (r, user, item)
                if node in node_data:
                    nid = node_data[node]
                else:
                    nid = cur_i
                    node_data[node] = nid
                    cur_i += 1
                edges.append([nid, aspect])

        edges = torch.LongTensor(edges).T
        edata = {
            ('rui', 'na', 'aspect'): (edges[0], edges[1])
        }
        rui_g = dgl.heterograph(edata)
        rui_g.ndata['aspect'].update({'org': rui_g.nodes('aspect') + self.ntype_range['aspect'][0]})

        rui = torch.LongTensor(list(node_data.keys())).T
        rui_id = torch.LongTensor(list(node_data.values()))

        edata = {
            ('review', 'rr', 'rui'): (rui[0], rui_id),
            ('user', 'ur', 'rui'): (rui[1], rui_id),
            ('item', 'ir', 'rui'): (rui[2], rui_id),
        }

        g = dgl.heterograph(edata)

        return rui_g, g

    def _construct_rui_ao_graph(self, sentiment_modality):
        sentiment_mapping = {s: i for i, s in enumerate(set([aos[2] for sid in sentiment_modality.sentiment.values()
                                                             for aos in sid]))}
        rui_data = OrderedDict()
        ao_data = OrderedDict()
        edges = []
        edata = []
        cur_rui_i = 0
        cur_ao_i = 0
        for uid, isid in sentiment_modality.user_sentiment.items():
            uid += self.ntype_range['user'][0]
            for iid, sid in isid.items():
                iid += self.ntype_range['item'][0]
                for aid, oid, sent in sentiment_modality.sentiment[sid]:
                    aid += self.ntype_range['aspect'][0]
                    oid += self.ntype_range['opinion'][0]
                    sent = sentiment_mapping[sent]
                    rui_node = (sid, uid, iid)
                    ao_node = (aid, oid)

                    # If node not in dict, assign new identifier and increase counter
                    if (ruiid := rui_data.get(rui_node, cur_rui_i)) == cur_rui_i:
                        rui_data[rui_node] = ruiid
                        cur_rui_i += 1

                    if (aoid := ao_data.get(ao_node, cur_ao_i)) == cur_ao_i:
                        ao_data[ao_node] = aoid
                        cur_ao_i += 1

                    edges.append([ruiid, aoid])
                    edata.append(sent)

        edges = torch.LongTensor(edges).T
        data = {
            ('rui', 'na', 'ao'): (edges[0], edges[1])
        }
        rui_g = dgl.heterograph(data)
        rui_g.edata['sent'] = torch.LongTensor(edata)

        data = {}
        for node_data, names in [(rui_data, ['review', 'user', 'item']), (ao_data, ['aspect', 'opinion'])]:
            dst_name = ''.join([n[0] for n in names])
            tuples = torch.LongTensor(list(node_data.keys())).T
            iden = torch.LongTensor(list(node_data.values()))
            for t, n in zip(tuples, names):
                data[(n, n[0]+dst_name[0], dst_name)] = (t, iden)

        g = dgl.heterograph(data)
        return rui_g, g

    def get_graphs(self):
        return self.rui_a_graph, self.n_rui_graph

    def get_eids(self):
        return {'na': self.rui_a_graph.edges('eid')}

    def sample_blocks(self, g, seed_nodes, exclude_eids=None):
        output_nodes = seed_nodes

        # Sample users, review and so forth based on id
        blocks = []
        for fanout in reversed(self.fanouts):
            frontier = self.n_rui_graph.sample_neighbors(
                seed_nodes, fanout, edge_dir=self.edge_dir, prob=self.prob,
                replace=self.replace, output_device=self.output_device,
                exclude_edges=exclude_eids)
            eid = frontier.edata[dgl.EID]
            block = dgl.to_block(frontier, seed_nodes, include_dst_in_src=False)
            block.edata[dgl.EID] = eid
            seed_nodes = block.srcdata[dgl.NID]
            blocks.insert(0, block)

        r_gs = [self.review_graphs[sid] for sid in blocks[0].srcdata[dgl.NID]['review'].cpu().numpy()]
        r_gs = dgl.compact_graphs(r_gs)
        nid = [bg.ndata[dgl.NID] for bg in r_gs]

        batch = dgl.batch(r_gs)

        # Get original ids
        nid = torch.cat(nid)
        batch.ndata[dgl.NID] = nid

        blocks.insert(0, batch)
        input_nodes = nid

        return input_nodes, output_nodes, blocks


def extract_attention(model, node_review_graph, device):
    # Node inference setup
    indices = {'node': node_review_graph.nodes('node')}
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
    dataloader = dgl.dataloading.DataLoader(node_review_graph, indices, sampler, batch_size=1024, shuffle=False,
                                            drop_last=False, device=device)

    review_attention = torch.zeros((node_review_graph.num_edges(), model.review_agg._num_heads, 1)).to(device)

    # Node inference
    for input_nodes, output_nodes, blocks in dataloader:
        x, a = model.review_aggregation(blocks[0]['part_of'], model.review_embs[input_nodes['review']], True)
        review_attention[blocks[0]['part_of'].edata[dgl.EID]] = a

    return review_attention