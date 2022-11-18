from dgl.dataloading.negative_sampler import _BaseNegativeSampler
import dgl.backend as F


class UniformItemSampler(_BaseNegativeSampler):
    def __init__(self, k, n_items):
        super(_BaseNegativeSampler, self).__init__()
        self.k = k
        self.n_items = n_items

    def _generate(self, g, eids, canonical_etype):
        _, _, vtype = canonical_etype
        shape = F.shape(eids)
        dtype = F.dtype(eids)
        ctx = F.context(eids)
        shape = (shape[0] * self.k,)
        src, _ = g.find_edges(eids, etype=canonical_etype)
        src = F.repeat(src, self.k, 0)
        dst = F.randint(shape, dtype, ctx, 0, self.n_items)
        return src, dst