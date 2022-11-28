import dgl
import numpy as np
import torch


class RLABlockSampler(dgl.dataloading.MultiLayerFullNeighborSampler):
    def __init__(self, train_set, num_layers, **kwargs):
        """
        Given nodes, samples reviews and creates a batched review-graph of all sampled reviews.
        Parameters
        ----------
        node_review_graph: DGLHeteroGraph
            A heterogeneous graph with edges from reviews to nodes (users/items) with relation-type part_of.
        review_graphs: dict[DGLGraph]
            A dictionary with sid to a graph representing a review based on sentiment.
        fanouts: list[int]
            Number of reviews to sample per node.
        kwargs: dict
            Arguments to pass to NeighborSampler. Read DGL docs for options.
        """
        super().__init__(num_layers, **kwargs)
        # Mapping reviews to the users that written them and similarly for items as well as a mapping
        # from nodes to the review contains.
        self.review_node_graph, self.rui_graph = self._build_graph(train_set)

    def _build_graph(self, train_set):
        ur = [[], []]
        ir = [[], []]
        ar = [[], []]
        o_r = [[], []]  # breaks naming convention as 'or' is python specific.

        # Get all edges from nodes to reviews
        for uid, isid in train_set.sentiment.user_sentiment.items():
            for iid, sid in isid.items():
                # Edge between user (item) and review.
                for arr, iden in [(ur, uid), (ir, iid)]:
                    arr[0].append(iden)
                    arr[1].append(sid)

                for aid, oid, _ in train_set.sentiment.sentiment[sid]:
                    # Edge between aspect (opinion) and review.
                    for arr, iden in [(ar, aid), (o_r, oid)]:
                        arr[0].append(iden)
                        arr[1].append(sid)

        # Get unique
        ur, ir, ar, o_r = [torch.LongTensor(np.unique(np.array(arr).T, axis=0).T) for arr in [ur, ir, ar, o_r]]

        data = {
            ('user', 'ur', 'review'): (ur[0], ur[1]),
            ('item', 'ir', 'review'): (ir[0], ir[1]),
            ('aspect', 'ar', 'review'): (ar[0], ar[1]),
            ('opinion', 'pr', 'review'): (o_r[0], o_r[1]),
            ('review', 'ru', 'user'): (ur[1], ur[0]),
            ('review', 'ri', 'item'): (ir[1], ir[0]),
        }
        g = dgl.heterograph(data, num_nodes_dict={
            'user': train_set.num_users,
            'item': train_set.num_items,
            'aspect': train_set.sentiment.num_aspects,
            'opinion': train_set.sentiment.num_opinions,
            'review': max(train_set.sentiment.sentiment) + 1
        })

        g.edges['ri'].data['npid'] = ur[0]
        g.edges['ru'].data['npid'] = ir[0]

        return dgl.edge_type_subgraph(g, list(set(g.etypes).difference({'ri', 'ru'}))), \
               dgl.edge_type_subgraph(g, ['ru', 'ri'])

    def sample_blocks(self, g, seed_nodes, exclude_eids=None):
        # If exclude eids, find the equivalent eid of the node_review_graph.
        output_nodes = seed_nodes
        blocks = []
        if exclude_eids is not None:
            ee = {}

            # Gives ui
            for canonical_etype, eids in exclude_eids.items():
                stype, etype, dtype = canonical_etype
                u, v = g.find_edges(eids, etype=etype)
                sid, inverse = torch.unique(g.edges[etype].data['sid'][eids], return_inverse=True)
                u[inverse], v[inverse] = u.clone(), v.clone()

                # Exclude user (item) to review edges.
                sr = stype[0] + 'r'
                ee[(stype, sr, 'review')] = self.review_node_graph.edge_ids(u, sid, etype=sr)

                # Exclude user (item) to aspect (opinion) edges.
                for st, etype, dt in self.review_node_graph.canonical_etypes:
                    if dt == 'review' and st not in [stype, dtype]:
                        s, i_sid = self.review_node_graph.in_edges(sid, etype=etype)
                        _, i_inv = torch.unique(i_sid, return_inverse=True)
                        et = stype[0] + st[0]
                        e = g.edge_ids(u[i_inv], s, etype=et)
                        ee[(stype, et, st)] = e

            # Unique:
            ee = {etype: torch.unique(eids) for etype, eids in ee.items()}
            ee.update({(d, e[::-1], s): v for (s, e, d), v in ee.items()})

            exclude_eids.update(ee)

        for i in range(2):
            dst_nodes = seed_nodes
            if i % 2 == 0:
                seed_nodes = {k: v for k, v in seed_nodes.items() if k != 'review'}
                i_g = self.rui_graph
            else:
                seed_nodes = {k: v for k, v in seed_nodes.items() if k == 'review'}
                i_g = self.review_node_graph
            
            frontier = i_g.sample_neighbors(
                seed_nodes, -1, edge_dir=self.edge_dir, prob=self.prob,
                replace=self.replace, output_device=self.output_device,
                exclude_edges=exclude_eids)
            eid = frontier.edata[dgl.EID]
            block = dgl.to_block(frontier, dst_nodes=dst_nodes)
            block.edata[dgl.EID] = eid
            seed_nodes = block.srcdata[dgl.NID]
            blocks.insert(0, block)

        input_nodes, _, _blocks = super(RLABlockSampler, self).sample_blocks(g, output_nodes, exclude_eids)

        # Stack correctly
        for b in reversed(_blocks):
            blocks.insert(0, b)

        return input_nodes, output_nodes, blocks

