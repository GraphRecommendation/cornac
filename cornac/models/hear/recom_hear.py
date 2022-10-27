from collections import Counter, defaultdict
from math import sqrt

import torch
from tqdm import tqdm

from ..recommender import Recommender
from ...data import Dataset


class HEAR(Recommender):
    def __init__(self, name='HEAR', use_cuda=False, use_uva=False,
                 batch_size=128,
                 num_workers=0,
                 num_epochs=10,
                 learning_rate=0.1,
                 node_dim=64,
                 l2_weight=0,
                 num_heads=3,
                 review_dim=32,
                 final_dim=16,
                 fanout=5,
                 model_selection='best',
                 review_aggregator='narre',
                 predictor='gatv2',
                 layer_dropout=None,
                 attention_dropout=.2,
                 user_based=True,
                 debug=False
                 ):
        super().__init__(name)
        # CUDA
        self.use_cuda = use_cuda
        self.device = 'cuda' if use_cuda else 'cpu'
        self.use_uva = use_uva
        assert use_uva == use_cuda or not use_uva, 'use_cuda must be true when using uva.'

        # Parameters
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.node_dim = node_dim
        self.l2_weight = l2_weight
        self.num_heads = num_heads
        self.review_dim = review_dim
        self.final_dim = final_dim
        self.fanout = fanout
        self.model_selection = model_selection
        self.review_aggregator = review_aggregator
        self.predictor = predictor
        self.layer_dropout = layer_dropout
        self.attention_dropout = attention_dropout

        # Method
        self.node_review_graph = None
        self.review_graphs = {}
        self.train_graph = None
        self.model = None
        self.n_items = 0

        # Misc
        self.user_based = user_based
        self.debug = debug

    def _create_graphs(self, train_set: Dataset):
        import dgl
        import torch

        # create 1) u,i,a, 2) u,i,o 3) u, a, o, 4) i, o, a
        sentiment_modality = train_set.sentiment
        edge_id = 0
        n_users = len(train_set.uid_map)
        n_items = len(train_set.iid_map)
        self.n_items = n_items
        n_aspects = len(sentiment_modality.aspect_id_map)
        n_opinions = len(sentiment_modality.opinion_id_map)
        n_nodes = n_users + n_items + n_aspects + n_opinions

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
                g = dgl.graph((torch.cat([src, dst]), torch.cat([dst, src])), num_nodes=n_nodes)

                # All hyperedges connect 3 nodes
                g.edata['id'] = torch.cat([eids, eids])
                g.edata['norm'] = torch.full((g.num_edges(),), 3 ** -1)

                # User and item have three hyper-edges for each AOS triple.
                g.ndata['norm'] = torch.zeros((g.num_nodes()))
                g.ndata['norm'][[uid, iid]] = sqrt(3 * len(aos)) ** -1

                # a and o also have three hyper edge triples for each occurrence.
                for nid, c in a_o_count.items():
                    g.ndata['norm'][nid] = sqrt(3 * c) ** -1

                assert len(edges) * 2 == g.num_edges()

                self.review_graphs[rid] = g

        # Create training graph, i.e. user to item graph.
        edges = [(uid + n_items, iid, train_set.matrix[uid, iid]) for uid, iid in zip(*train_set.matrix.nonzero())]
        t_edges = torch.LongTensor(edges).T
        self.train_graph = dgl.graph((t_edges[0], t_edges[1]))
        self.train_graph.edata['rid'] = torch.LongTensor([user_item_review_map[(u, i)] for (u, i, r) in edges])
        self.train_graph.edata['label'] = t_edges[2].to(torch.float)

        # Create user/item to review graph.
        edges = torch.LongTensor(review_edges).T
        self.node_review_graph = dgl.heterograph({('review', 'part_of', 'node'): (edges[0], edges[1])})

        return n_nodes

    def fit(self, train_set: Dataset, val_set=None):
        from .hear import Model
        import dgl
        from torch import optim
        from . import dgl_utils

        super().fit(train_set, val_set)
        n_nodes = self._create_graphs(train_set)  # graphs are as attributes of model.

        # create model
        self.model = Model(n_nodes, self.review_aggregator, self.predictor, self.node_dim,
                           self.review_dim, self.final_dim, self.num_heads)
        if self.use_cuda:
            self.model = self.model.cuda()
            prefetch = ['label']
        else:
            prefetch = []

        # Get graph and edges
        g = self.train_graph
        u, v = g.edges()
        _, i, c = torch.unique(u, sorted=False, return_inverse=True, return_counts=True)
        mask = c[i] > 1
        _, i, c = torch.unique(v, sorted=False, return_inverse=True, return_counts=True)
        mask *= (c[i] > 1)
        eids = g.edges(form='eid')[mask]
        num_workers = self.num_workers

        if self.use_uva:
            # g = g.to(self.device)
            eids = eids.to(self.device)
            self.node_review_graph = self.node_review_graph.to(self.device)
            num_workers = 0

        if self.debug:
            num_workers = 0
            thread = False
        else:
            thread = None

        # Create sampler
        sampler = dgl_utils.HearBlockSampler(self.node_review_graph, self.review_graphs, self.review_aggregator, fanout=self.fanout)
        sampler = dgl_utils.HEAREdgeSampler(sampler, prefetch_labels=prefetch)
        dataloader = dgl.dataloading.DataLoader(g, eids, sampler, batch_size=self.batch_size, shuffle=True,
                                                drop_last=True, device=self.device, use_uva=self.use_uva,
                                                num_workers=num_workers, use_prefetch_thread=thread)

        # Initialize training params.
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        length = len(dataloader)
        for e in range(self.num_epochs):
            tot_loss = 0
            tot_mse = 0
            tot_l2 = 0
            with tqdm(dataloader) as progress:
                for i, (input_nodes, edge_subgraph, blocks) in enumerate(progress, 1):
                    x = self.model(blocks, self.model.node_embedding(input_nodes))

                    pred = self.model.graph_predict(edge_subgraph, x)

                    mse_loss = self.model.loss(pred, edge_subgraph.edata['label'])
                    l2_loss = self.l2_weight * self.model.l2_loss(input_nodes)
                    loss = mse_loss + l2_loss
                    loss.backward()

                    tot_mse += mse_loss.detach()
                    tot_l2 += l2_loss.detach()
                    tot_loss += loss.detach()

                    optimizer.step()
                    optimizer.zero_grad()
                    if i != length or val_set is None:
                        progress.set_description(f'Epoch {e}, '
                                                 f'MSE: {tot_mse / i:.5f}, '
                                                 f'L2: {tot_l2 / i:.5f}, '
                                                 f'Tot: {tot_loss / i:.5f}')
                    else:
                        mse, rmse = self._validate(val_set)
                        progress.set_description(f'Epoch {e}, MSE: {tot_mse / i:.5f}, Val: MSE: {mse:.5f}, '
                                                 f'RMSE: {rmse:.5f}')

        _ = self._validate(val_set)
        self.node_review_graph = self.node_review_graph.to(self.device)

    def _validate(self, val_set):
        from ...eval_methods.base_method import rating_eval
        from ...metrics import MSE, RMSE
        import torch
        d = self.node_review_graph.device
        self.node_review_graph = self.node_review_graph.to(self.device)

        self.model.eval()
        with torch.no_grad():
            self.model.inference(self.review_graphs, self.node_review_graph, self.device)
            ((mse, rmse), _) = rating_eval(self, [MSE(), RMSE()], val_set, user_based=self.user_based)

        self.node_review_graph = self.node_review_graph.to(d)
        return mse, rmse

    def score(self, user_idx, item_idx=None):
        import torch

        self.model.eval()
        with torch.no_grad():
            user_idx = torch.tensor(user_idx + self.n_items, dtype=torch.int64).to(self.device)
            item_idx = torch.tensor(item_idx, dtype=torch.int64).to(self.device)
            return self.model.predict(user_idx, item_idx, self.node_review_graph).cpu()

    def monitor_value(self):
        pass
