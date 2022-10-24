from collections import Counter, defaultdict
from math import sqrt

import torch
from torch import optim
from tqdm import tqdm

from . import dgl_utils
from ..recommender import Recommender
from ...data import Dataset
from .hear import Model
import dgl


class HEAR(Recommender):
	def __init__(self, name='HEAR'):
		super().__init__(name)
		self.node_review_graph = None
		self.review_graphs = {}
		self.review_pair_map = {}
		self.train_graph = None

	def _create_graphs(self, train_set: Dataset):
		# create 1) u,i,a, 2) u,i,o 3) u, a, o, 4) i, o, a
		sentiment_modality = train_set.sentiment
		edge_id = 0
		n_users = len(train_set.uid_map)
		n_items = len(train_set.iid_map)
		n_aspects = len(sentiment_modality.aspect_id_map)
		n_opinions = len(sentiment_modality.opinion_id_map)
		user_item_review_map = {(uid + n_items, iid): rid for uid, irid in sentiment_modality.user_sentiment.items()
								for iid, rid in irid.items()}
		review_edges = []
		for uid, irid in tqdm(sentiment_modality.user_sentiment.items(), desc='Creating review graphs',
									total=len(sentiment_modality.user_sentiment)):
			uid += n_items

			for iid, rid in irid.items():
				review_edges.extend([[rid, uid], [rid, iid]])
				edges = []
				a_o_count = defaultdict(int)
				aos = sentiment_modality.sentiment[rid]
				for aid, oid, _ in aos:
					aid += n_items + n_users
					oid += n_items + n_users + n_aspects

					a_o_count[aid] += 1
					a_o_count[oid] += 1
					for f, s, t in [[uid, iid, aid], [uid, iid, oid], [uid, aid, oid], [iid, oid, aid]]:
						edges.append([f, s, edge_id])
						edges.append([s, t, edge_id])
						edges.append([t, f, edge_id])
						edges.append([f, f, edge_id])
						edges.append([s, s, edge_id])
						edges.append([t, t, edge_id])
						edge_id += 1

				src, dst = torch.LongTensor([e for e, _, _ in edges]), torch.LongTensor([e for _, e, _ in edges])
				eids = torch.LongTensor([r for _, _, r in edges])
				g = dgl.graph((torch.cat([src, dst]), torch.cat([dst, src])))

				# All hyperedges connect 3 nodes
				g.edata['id'] = torch.cat([eids, eids])
				g.edata['norm'] = torch.full((g.num_edges(),), 3**-1)

				# User and item have three hyper-edges for each AOS triple.
				g.ndata['norm'] = torch.zeros((g.num_nodes()))
				g.ndata['norm'][[uid, iid]] = sqrt(3*len(aos))**-1

				# a and o also have three hyper edge triples for each occurrence.
				for nid, c in a_o_count.items():
					g.ndata['norm'][nid] = sqrt(3*c)**-1

				assert len(edges) * 2 == g.num_edges()

				self.review_graphs[(uid, iid)] = g

		# Create training graph, i.e. user to item graph.
		edges = [(uid + n_items, iid, train_set.matrix[uid,iid]) for uid, rs in train_set.review_text.user_review.items()
								 for iid, rid in rs.items() if (uid+n_items, iid) in user_item_review_map]
		t_edges = torch.LongTensor(edges).T
		self.train_graph = dgl.graph((t_edges[0], t_edges[1]))
		self.train_graph.edata['rid'] = torch.LongTensor([user_item_review_map[(u, i)] for (u, i, r) in edges])
		self.train_graph.edata['label'] = t_edges[2].to(torch.float)

		self.review_pair_map = {rid: pair for pair, rid in user_item_review_map.items()}

		# Create user/item to review graph.
		edges = torch.LongTensor(review_edges).T
		self.node_review_graph = dgl.heterograph({('review', 'part_of', 'node'): (edges[0], edges[1])})

		return n_users + n_items + n_aspects + n_opinions

	def fit(self, train_set: Dataset, val_set=None):
		n_nodes = self._create_graphs(train_set)  # graphs are as attributes of model.

		# create model
		model = Model(n_nodes)

		g = self.train_graph
		eids = g.edges(form='eid')
		sampler = dgl_utils.HearBlockSampler(self.node_review_graph, self.review_graphs, self.review_pair_map)
		sampler = dgl.dataloading.as_edge_prediction_sampler(sampler, exclude='self')
		dataloader = dgl.dataloading.DataLoader(g, eids, sampler, batch_size=1024, shuffle=True, drop_last=True)
		optimizer = optim.Adam(model.parameters(), lr=0.01)

		for e in range(10):
			tot_loss = 0
			with tqdm(dataloader) as progress:
				for i, (input_nodes, edge_subgraph, blocks) in enumerate(progress, 1):
					x = model.node_embedding(input_nodes)

					x = model(blocks, x)

					pred = model.graph_predict(edge_subgraph, x)

					loss = model.loss(pred, edge_subgraph.edata['label'])
					loss.backward()

					tot_loss += loss.detach()

					optimizer.step()
					optimizer.zero_grad()
					progress.set_description(f'Epoch {e}, MSE: {tot_loss / i:.5f}')

	def score(self, user_idx, item_idx=None):
		pass

	def monitor_value(self):
		pass
