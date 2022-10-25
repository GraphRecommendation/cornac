import dgl.dataloading
import torch


class HearBlockSampler(dgl.dataloading.NeighborSampler):
	def __init__(self, node_review_graph, review_graphs, fanouts=None, **kwargs):
		if fanouts is None:
			fanouts = [5]

		super().__init__(fanouts, **kwargs)
		self.node_review_graph = node_review_graph
		self.review_graphs = review_graphs

	def sample(self, g, seed_nodes, exclude_eids=None):
		input_nodes, output_nodes, blocks = super().sample(self.node_review_graph, {'node': seed_nodes}, None)
		b = blocks[0]['part_of']
		blocks[0] = b
		
		r_gs = [self.review_graphs[rid] for rid in b.srcdata[dgl.NID].cpu().numpy()]
		batch = dgl.batch(r_gs)
		nid = [bg.nodes() for bg in r_gs]
		nid = torch.cat(nid)
		batch.ndata[dgl.NID] = nid

		blocks.insert(0, batch)

		input_nodes = batch.srcdata[dgl.NID]

		return input_nodes, output_nodes, blocks


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


