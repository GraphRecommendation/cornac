import dgl.dataloading
import torch


class HAGERecBlockSampler(dgl.dataloading.NeighborSampler):
    def __init__(self, fanouts, n_users, n_items, **kwargs):
        super().__init__(fanouts, **kwargs)
        # args n_entities,
        # self.n_entities = n_entities
        self.n_items = n_items
        self.n_users = n_users

    def _convert(self, node_dict):
        """
        Generates a block mapping nodes to users items and aspects.
        Parameters
        ----------
        node_dict: dict[Tensor]
            A mapping from node type to tensor of node indices.

        Returns
        -------
        DGLBlock, dict[Tensor]

        """
        # Get nodes, can be users, items or other entities.
        nodes = node_dict.get('node', torch.LongTensor([]))
        # We need to map these to users, items and others.
        users = nodes[(nodes >= self.n_items) * (nodes < self.n_users + self.n_items)]
        items = nodes[nodes < self.n_items]
        others = nodes[nodes >= self.n_users + self.n_items]
        # Create a dict mapping user to user, item to item, node to user, node to item, and node to other nodes.
        # Other nodes is nodes \ users U items <-- set operations.
        nd = {
            ('user', 'uu', 'user'): (node_dict['user'], node_dict['user']),
            ('item', 'ii', 'item'): (node_dict['item'], node_dict['item']),
            ('user', 'u_isa', 'node'): (users, users),
            ('item', 'i_isa', 'node'): (items, items),
            ('node', 'n_isa', 'node'): (others, others)
        }
        g = dgl.heterograph(nd)

        # Convert to block without source nodes (source nodes already in dict).
        # NOTE: this block should not be used for actual aggregation.
        block = dgl.to_block(g, include_dst_in_src=False)

        fn = lambda nt, et, x: block.in_degrees(block.dstnodes(nt), etype=et) == x
        assert all([torch.all(torch.logical_or(fn(ntype, etype, 1), fn(ntype, etype, 0)))
                    for _, etype, ntype in block.canonical_etypes])

        # Return block and dict of nodes
        return block, block.srcdata[dgl.NID]

    def sample_blocks(self, g, seed_nodes, exclude_eids=None):
        # input_nodes, output_nodes, blocks = super(HAGERecBlockSampler, self).sample_blocks(g, seed_nodes, exclude_eids)
        output_nodes = seed_nodes

        # Get block mapping seed-nodes to user, item, and other nodes.
        block, seed_nodes = self._convert(seed_nodes)
        blocks = [block]
        for fanout in reversed(self.fanouts):
            # Sample neighbors.
            frontier = g.sample_neighbors(
                seed_nodes, fanout, edge_dir=self.edge_dir, prob=self.prob,
                replace=self.replace, output_device=self.output_device,
                exclude_edges=exclude_eids)
            eid = frontier.edata[dgl.EID]  # original edge id
            block = dgl.to_block(frontier, seed_nodes)  # to block
            block.edata[dgl.EID] = eid  # assign org ids.
            blocks.insert(0, block)

            # Convert nodes not users, items and others. Block is a mapping and new seed node set with no overlap.
            block, seed_nodes = self._convert(block.srcdata[dgl.NID])
            blocks.insert(0, block)

        # No need to have mapping as initial block. Remove and use now first block as seed nodes.
        blocks.pop(0)
        seed_nodes = blocks[0].srcdata[dgl.NID]

        return seed_nodes, output_nodes, blocks
