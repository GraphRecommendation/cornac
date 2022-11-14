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
        nodes = node_dict.get('node', torch.LongTensor([]))
        users = nodes[(nodes >= self.n_items) * (nodes < self.n_users + self.n_items)]
        items = nodes[nodes < self.n_items]
        others = nodes[nodes >= self.n_users + self.n_items]
        nd = {
            ('user', 'uu', 'user'): (node_dict['user'], node_dict['user']),
            ('item', 'ii', 'item'): (node_dict['item'], node_dict['item']),
            ('user', 'u_isa', 'node'): (users, users),
            ('item', 'i_isa', 'node'): (items, items),
            ('node', 'n_isa', 'node'): (others, others)
        }
        g = dgl.heterograph(nd)

        for etype in g.etypes:
            u, v = g.edges(etype=etype)
            assert torch.all(torch.unique(u, return_counts=True)[1] == 1) and \
                   torch.all(torch.unique(v, return_counts=True)[1] == 1)

        block = dgl.to_block(g, include_dst_in_src=False)
        return block, block.srcdata[dgl.NID]

    def sample_blocks(self, g, seed_nodes, exclude_eids=None):
        # input_nodes, output_nodes, blocks = super(HAGERecBlockSampler, self).sample_blocks(g, seed_nodes, exclude_eids)
        output_nodes = seed_nodes
        block, seed_nodes = self._convert(seed_nodes)
        blocks = [block]
        for fanout in reversed(self.fanouts):
            frontier = g.sample_neighbors(
                seed_nodes, fanout, edge_dir=self.edge_dir, prob=self.prob,
                replace=self.replace, output_device=self.output_device,
                exclude_edges=exclude_eids)
            eid = frontier.edata[dgl.EID]
            block = dgl.to_block(frontier, seed_nodes)
            block.edata[dgl.EID] = eid
            blocks.insert(0, block)
            block, seed_nodes = self._convert(block.srcdata[dgl.NID])
            blocks.insert(0, block)

        blocks.pop(0)
        seed_nodes = blocks[0].srcdata[dgl.NID]

        return seed_nodes, output_nodes, blocks
