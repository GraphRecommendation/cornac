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


class RLA2BlockSampler(dgl.dataloading.MultiLayerFullNeighborSampler):
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
        super().__init__(1, **kwargs)
        # Mapping reviews to the users that written them and similarly for items as well as a mapping
        # from nodes to the review contains.
        self.review_node_graph, self.rui_graph, self.aos_graph = self._build_graph(train_set)

    def get_graphs(self):
        return self.review_node_graph, self.rui_graph, self.aos_graph

    def _build_graph(self, train_set):
        ur = [[], []]
        ir = [[], []]
        aosr = [[], []]

        # Get all edges from nodes to reviews
        for uid, isid in train_set.sentiment.user_sentiment.items():
            for iid, sid in isid.items():
                # Edge between user (item) and review.
                for arr, iden in [(ur, uid), (ir, iid)]:
                    arr[0].append(iden)
                    arr[1].append(sid)

                for aos in train_set.sentiment.sentiment[sid]:
                    aosr[0].append(aos)
                    aosr[1].append(sid)

        aoss = set(aosr[0])
        aos_id = {aos: i for i, aos in enumerate(sorted(aoss))}
        id_aos = {v: k for k, v in aos_id.items()}

        aosr = [[aos_id[aos] for aos in aosr[0]], aosr[1]]

        # Get unique
        ur, ir, aosr = [torch.LongTensor(arr) for arr in [ur, ir, aosr]]


        data = {
            ('user', 'ur', 'review'): (ur[0], ur[1]),
            ('item', 'ir', 'review'): (ir[0], ir[1]),
            ('aos', 'ar', 'review'): (aosr[0], aosr[1]),
            ('review', 'ru', 'user'): (ur[1], ur[0]),
            ('review', 'ri', 'item'): (ir[1], ir[0]),
        }
        g = dgl.heterograph(data, num_nodes_dict={
            'user': train_set.num_users,
            'item': train_set.num_items,
            'aos': len(aos_id),
            'review': max(train_set.sentiment.sentiment) + 1
        })

        g.edges['ri'].data['npid'] = ur[0]
        g.edges['ru'].data['npid'] = ir[0]

        iaos = list(sorted(id_aos))
        aoss = [id_aos[aos] for aos in iaos]
        s_id = {s: i for i, s in enumerate(set([aos[2] for aos in aoss]))}
        sids = torch.LongTensor([s_id[aos[2]] for aos in aoss])

        data = {
            ('aspect', 'aa', 'aos'): (torch.LongTensor([aos[0] for aos in aoss]), iaos),
            ('opinion', 'oa', 'aos'): (torch.LongTensor([aos[1] for aos in aoss]), iaos),
            ('sentiment', 'sa', 'aos'): (sids, iaos),
        }

        g2 = dgl.heterograph(data, num_nodes_dict={
            'aos': len(aos_id),
            'aspect': train_set.sentiment.num_aspects,
            'opinion': train_set.sentiment.num_opinions,
            'sentiment': len(s_id)
        })

        return dgl.edge_type_subgraph(g, list(set(g.etypes).difference({'ri', 'ru', 'ur', 'ir'}))), \
               dgl.edge_type_subgraph(g, ['ru', 'ri']), \
               g2

    def sample_blocks(self, g, seed_nodes, exclude_eids=None):
        # If exclude eids, find the equivalent eid of the node_review_graph.
        output_nodes = seed_nodes
        blocks = []
        # if exclude_eids is not None:
        #     ee = {}
        #
        #     # Gives ui
        #     for canonical_etype, eids in exclude_eids.items():
        #         stype, etype, dtype = canonical_etype
        #         u, v = g.find_edges(eids, etype=etype)
        #
        #         # Sort sentiment id and use sorting to sort users and items based on sentiment id.
        #         # torch.unique sorts by default.
        #         sid, inverse = torch.unique(g.edges[etype].data['sid'][eids], return_inverse=True)
        #         u[inverse], v[inverse] = u.clone(), v.clone()
        #
        #         # Exclude user (item) to review edges.
        #         sr = stype[0] + 'r'
        #         ee[(stype, sr, 'review')] = self.review_node_graph.edge_ids(u, sid, etype=sr)
        #
        #         # Exclude user (item) to aspect (opinion) edges.
        #         for st, etype, dt in self.review_node_graph.canonical_etypes:
        #             if dt == 'review' and st not in [stype, dtype]:
        #                 s, i_sid = self.review_node_graph.in_edges(sid, etype=etype)
        #                 _, i_inv = torch.unique(i_sid, return_inverse=True)
        #                 et = stype[0] + st[0]
        #                 e = g.edge_ids(u[i_inv], s, etype=et)
        #                 ee[(stype, et, st)] = e
        #
        #     # Unique:
        #     ee = {etype: torch.unique(eids) for etype, eids in ee.items()}
        #     ee.update({(d, e[::-1], s): v for (s, e, d), v in ee.items()})
        #
        #     exclude_eids.update(ee)

        for i in range(3):
            dst_nodes = seed_nodes
            include_src_in_dst = True
            if i == 0:
                seed_nodes = {k: v for k, v in seed_nodes.items() if k != 'review'}
                i_g = self.rui_graph
            elif i == 1:
                seed_nodes = {k: v for k, v in seed_nodes.items() if k == 'review'}
                i_g = self.review_node_graph
            else:
                i_g = self.aos_graph
                include_src_in_dst = False

            frontier = i_g.sample_neighbors(
                seed_nodes, -1, edge_dir=self.edge_dir, prob=self.prob,
                replace=self.replace, output_device=self.output_device,
                exclude_edges=exclude_eids)
            eid = frontier.edata[dgl.EID]
            block = dgl.to_block(frontier, dst_nodes=dst_nodes, include_dst_in_src=include_src_in_dst)
            block.edata[dgl.EID] = eid
            seed_nodes = block.srcdata[dgl.NID]
            blocks.insert(0, block)

        return seed_nodes, output_nodes, blocks


class RLA3BlockSampler(dgl.dataloading.MultiLayerFullNeighborSampler):
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
        super().__init__(1, **kwargs)
        # Mapping reviews to the users that written them and similarly for items as well as a mapping
        # from nodes to the review contains.
        self.review_node_graph, self.rui_graph, self.aos_graph = self._build_graph(train_set)

    def get_graphs(self):
        return self.review_node_graph, self.rui_graph, self.aos_graph

    def _build_graph(self, train_set):
        ur = [[], []]
        ir = [[], []]
        aosr = [[], []]

        # Get all edges from nodes to reviews
        for uid, isid in train_set.sentiment.user_sentiment.items():
            for iid, sid in isid.items():
                # Edge between user (item) and review.
                for arr, iden in [(ur, uid), (ir, iid)]:
                    arr[0].append(iden)
                    arr[1].append(sid)

                for aos in train_set.sentiment.sentiment[sid]:
                    aosr[0].append(aos)
                    aosr[1].append(sid)

        aoss = set(aosr[0])
        aos_id = {aos: i for i, aos in enumerate(sorted(aoss))}
        id_aos = {v: k for k, v in aos_id.items()}

        aosr = [[aos_id[aos] for aos in aosr[0]], aosr[1]]

        # Get unique
        ur, ir, aosr = [torch.LongTensor(arr) for arr in [ur, ir, aosr]]


        data = {
            ('user', 'ur', 'review'): (ur[0], ur[1]),
            ('item', 'ir', 'review'): (ir[0], ir[1]),
            ('aos', 'ar', 'review'): (aosr[0], aosr[1]),
            ('review', 'ru', 'user'): (ur[1], ur[0]),
            ('review', 'ri', 'item'): (ir[1], ir[0]),
        }
        g = dgl.heterograph(data, num_nodes_dict={
            'user': train_set.num_users,
            'item': train_set.num_items,
            'aos': len(aos_id),
            'review': max(train_set.sentiment.sentiment) + 1
        })

        g.edges['ri'].data['npid'] = ur[0]
        g.edges['ru'].data['npid'] = ir[0]

        iaos = list(sorted(id_aos))
        aoss = [id_aos[aos] for aos in iaos]
        s_id = {s: i for i, s in enumerate(set([aos[2] for aos in aoss]))}
        sids = torch.LongTensor([s_id[aos[2]] for aos in aoss])

        data = {
            ('aspect', 'aa', 'aos'): (torch.LongTensor([aos[0] for aos in aoss]), iaos),
            ('opinion', 'oa', 'aos'): (torch.LongTensor([aos[1] for aos in aoss]), iaos),
            ('sentiment', 'sa', 'aos'): (sids, iaos),
        }

        g2 = dgl.heterograph(data, num_nodes_dict={
            'aos': len(aos_id),
            'aspect': train_set.sentiment.num_aspects,
            'opinion': train_set.sentiment.num_opinions,
            'sentiment': len(s_id)
        })

        return dgl.edge_type_subgraph(g, list(set(g.etypes).difference({'ri', 'ru', 'ur', 'ir'}))), \
               dgl.edge_type_subgraph(g, ['ru', 'ri']), \
               g2

    def sample_blocks(self, g, seed_nodes, exclude_eids=None):
        # If exclude eids, find the equivalent eid of the node_review_graph.
        output_nodes = seed_nodes
        blocks = []
        on = {}
        for nt in sorted(output_nodes):
            nids = output_nodes[nt]
            seed_nodes = {nt: nids}
            if not len(nids):
                continue
            for i in range(3):
                dst_nodes = seed_nodes
                include_src_in_dst = True
                if i == 0:
                    seed_nodes = {k: v for k, v in seed_nodes.items() if k != 'review'}
                    i_g = self.rui_graph
                elif i == 1:
                    seed_nodes = {k: v for k, v in seed_nodes.items() if k == 'review'}
                    i_g = self.review_node_graph
                else:
                    i_g = self.aos_graph
                    include_src_in_dst = False

                frontier = i_g.sample_neighbors(
                    seed_nodes, -1, edge_dir=self.edge_dir, prob=self.prob,
                    replace=self.replace, output_device=self.output_device,
                    exclude_edges=exclude_eids)
                eid = frontier.edata[dgl.EID]
                block = dgl.to_block(frontier, dst_nodes=dst_nodes, include_dst_in_src=include_src_in_dst)
                block.edata[dgl.EID] = eid
                seed_nodes = block.srcdata[dgl.NID]
                blocks.insert(0, block)

            on[nt] = seed_nodes

        return on, output_nodes, blocks