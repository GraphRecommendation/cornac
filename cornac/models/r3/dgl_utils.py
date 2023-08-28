from collections import defaultdict

import dgl.dataloading
import numpy as np
import torch
from torch.utils.data import Dataset


class R3Sampler(dgl.dataloading.MultiLayerFullNeighborSampler):
    def __init__(self, node_review_graph, sequences, max_length=200, max_reviews=10, **kwargs):
        super().__init__(1, **kwargs)
        self.node_review_graph = node_review_graph
        self.sequences = [torch.LongTensor(seq) for seq in sequences]
        self.max_length = max_length
        self.max_reviews = max_reviews

    def sample(self, g, seed_nodes, exclude_eids=None):
        blocks = {}
        input_nodes = {}
        exclude_eids = next(iter(exclude_eids.values())) if exclude_eids is not None else None
        for ntype, ids in seed_nodes.items():
            nt_nodes = torch.zeros((len(ids), self.max_reviews, self.max_length), dtype=torch.long)
            for i, n in enumerate(ids):
                # NOTE: Assumes DGL maintains edge order and reviews are added in order.
                rs, _, eids = self.node_review_graph.in_edges(n, etype=f'r{ntype[0]}', form='all')

                for j, (r, eid) in enumerate(zip(rs[-self.max_reviews:], eids)):
                    if exclude_eids is None or eid not in exclude_eids:
                        nt_nodes[i, j, :len(self.sequences[r])] = self.sequences[r]
            input_nodes[ntype] = nt_nodes
        return input_nodes, seed_nodes, blocks # [blocks['user'], blocks['item']]