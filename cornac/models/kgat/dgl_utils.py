import dgl.dataloading


class TransRSampler(dgl.dataloading.base.BlockSampler):
    def sample_blocks(self, g, seed_nodes, exclude_eids=None):
        return seed_nodes, seed_nodes, []

    def sample(self, g, seed_nodes, exclude_eids=None):     # pylint: disable=arguments-differ
        """Sample a list of blocks from the given seed nodes."""
        return self.sample_blocks(g, seed_nodes, exclude_eids=exclude_eids)
