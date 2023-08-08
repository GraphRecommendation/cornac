from collections import defaultdict

import dgl.dataloading
import numpy as np
import torch
from torch.utils.data import Dataset


class R3Sampler(dgl.dataloading.MultiLayerFullNeighborSampler):
    def __init__(self, node_review_graph, sequences, max_length=200, **kwargs):
        super().__init__(1, **kwargs)
        self.node_review_graph = node_review_graph
        self.sequences = [torch.LongTensor(seq) for seq in sequences]
        self.max_length = max_length

    def sample(self, g, seed_nodes, exclude_eids=None):
        #todo exclude eids based on g and seed nodes
        blocks = {}
        input_nodes = {}
        for ntype, ids in seed_nodes.items():
            b_nodes, output_nodes, (block,) = super().sample(self.node_review_graph, {ntype: ids})
            blocks[ntype] = block
            ids = b_nodes['review']
            nt_nodes = torch.zeros((len(ids), self.max_length), dtype=torch.long)
            for i, r in enumerate(ids):
                nt_nodes[i, :len(self.sequences[r])] = self.sequences[r]
            input_nodes[ntype] = nt_nodes
        return input_nodes, seed_nodes, blocks # [blocks['user'], blocks['item']]