import dgl.dataloading
import torch


class HearBlockSampler(dgl.dataloading.NeighborSampler):
	def __init__(self, node_review_graph, review_graphs, review_pair_map, fanouts=None, **kwargs):
		if fanouts is None:
			fanouts = [5]

		super().__init__(fanouts, **kwargs)
		self.node_review_graph = node_review_graph
		self.review_graphs = review_graphs
		self.review_pair_map = review_pair_map

	def min_degree(self, nodes, g):
		pass

	def sample(self, g, seed_nodes, exclude_eids=None):
		input_nodes, output_nodes, blocks = super().sample(self.node_review_graph, {'node': seed_nodes}, None)
		b = blocks[0]['part_of']
		blocks[0] = b
		
		r_gs = [self.review_graphs[self.review_pair_map[rid]] for rid in b.srcdata[dgl.NID].cpu().numpy()]
		batch = dgl.batch(r_gs)
		nid = [bg.nodes() for bg in r_gs]
		nid = torch.cat(nid)
		batch.ndata[dgl.NID] = nid

		blocks.insert(0, batch)

		input_nodes = batch.srcdata[dgl.NID]

		return input_nodes, output_nodes, blocks