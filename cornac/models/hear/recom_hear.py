import torch
from tqdm import tqdm

from ..recommender import Recommender
from ...data import Dataset
import dgl

class HEAR(Recommender):
	def __init__(self, name='HEAR'):
		super().__init__(name)
		self.item_graphs = {}
		self.user_graphs = {}

	def _create_graphs(self, train_set: Dataset):
		# create 1) u,i,a, 2) u,i,o 3) u, a, o, 4) i, o, a
		sentiment_modality = train_set.sentiment
		edge_id = 0
		n_users = len(train_set.uid_map)
		n_items = len(train_set.iid_map)
		n_aspects = len(sentiment_modality.aspect_id_map)
		n_opinions = len(sentiment_modality.opinion_id_map)
		for user, item, aos in tqdm(sentiment_modality.raw_data, desc='Creating review graphs'):
			if user not in train_set.uid_map or item not in train_set.iid_map:
				continue

			uid = train_set.uid_map[user] + n_items
			iid = train_set.iid_map[item]
			if uid not in self.user_graphs:
				self.user_graphs[uid] = []
			if iid not in self.item_graphs:
				self.item_graphs[iid] = []
			for graphs in [self.user_graphs[uid], self.item_graphs[iid]]:
				edges = []
				for a, o, _ in aos:
					if a not in sentiment_modality.aspect_id_map or o not in sentiment_modality.opinion_id_map:
						continue

					aid = sentiment_modality.aspect_id_map[a] + n_items + n_users
					oid = sentiment_modality.opinion_id_map[o] + n_items + n_users + n_aspects
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
				g.edata['id'] = torch.cat([eids, eids])

				assert len(edges) * 2 == g.num_edges()

				graphs.append(g)

		return n_users + n_items + n_aspects + n_opinions

	def fit(self, train_set: Dataset, val_set=None):
		n_entities = self._create_graphs(train_set)  # graphs are as attributes of model.

		# create model

		# create dataloader
		pass

	def _fit(self, model, dataloader):
		for x in dataloader:
			x = model(x)
			

	def score(self, user_idx, item_idx=None):
		pass

	def monitor_value(self):
		pass
