from collections import Counter, defaultdict
from copy import deepcopy
from math import sqrt

import torch
from tqdm import tqdm

from ..recommender import Recommender
from ...data import Dataset


class KGAT(Recommender):
    def __init__(self, name='KGAT', use_cuda=False, use_uva=False,
                 batch_size=128,
                 num_workers=0,
                 num_epochs=10,
                 learning_rate=0.1,
                 weight_decay=0,
                 node_dim=64,
                 review_dim=32,
                 final_dim=16,
                 num_heads=3,
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
        # Default values
        if layer_dropout is None:
            layer_dropout = [0, 0]  # node embedding dropout, review embedding dropout

        # CUDA
        self.use_cuda = use_cuda
        self.device = 'cuda' if use_cuda else 'cpu'
        self.use_uva = use_uva

        # Parameters
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.node_dim = node_dim
        self.review_dim = review_dim
        self.final_dim = final_dim
        self.num_heads = num_heads
        self.fanout = fanout
        self.model_selection = model_selection
        self.review_aggregator = review_aggregator
        self.predictor = predictor
        self.layer_dropout = layer_dropout
        self.attention_dropout = attention_dropout

        # Method
        self.train_graph = None
        self.model = None
        self.n_items = 0

        # Misc
        self.user_based = user_based
        self.debug = debug

        # assertions
        assert use_uva == use_cuda or not use_uva, 'use_cuda must be true when using uva.'
        assert len(layer_dropout) == 2, f'Length of dropout list must be 2 was {len(layer_dropout)}. ' \
                                        f'First for node embedding dropout, second for review embedding dropout.'

    def _create_graphs(self, train_set: Dataset):
        import dgl
        import torch

        edge_types = {
            'mentions': [],
            'described_as': [],
            'has_opinion': [],
            'co-occur': [],
        }

        rating_types = set()
        for indices in list(zip(*train_set.matrix.nonzero())):
            rating_types.add(train_set.matrix[indices])

        train_types = []
        for rt in rating_types:
            edge_types[str(rt)] = []
            train_types.append(str(rt))

        sentiment_modality = train_set.sentiment
        n_users = len(train_set.uid_map)
        n_items = len(train_set.iid_map)
        self.n_items = n_items
        n_aspects = len(sentiment_modality.aspect_id_map)
        n_opinions = len(sentiment_modality.opinion_id_map)
        n_nodes = n_users + n_items + n_aspects + n_opinions

        # Create all the edges: (item, described_as, aspect), (item, has_opinion, opinion), (user, mentions, aspect),
        # (aspect, cooccur, opinion), and (user, 'rating', item). Note rating is on a scale.
        for org_uid, isid in sentiment_modality.user_sentiment.items():
            uid = org_uid + n_items
            for iid, sid in isid.items():
                for aid, oid, _ in sentiment_modality.sentiment[sid]:
                    aid += n_items + n_users
                    oid += n_items + n_users + n_aspects

                    edge_types['mentions'].append([uid, aid])
                    edge_types['described_as'].append([iid, aid])
                    edge_types['has_opinion'].append([iid, oid])
                    edge_types['co-occur'].append([aid, oid])

                edge_types[str(train_set.matrix[(org_uid, iid)])].append([uid, iid])

        # Create reverse edges.
        reverse = {}
        for etype, edges in edge_types.items():
            reverse['r_' + etype] = [[t, h] for h, t in edges]

        edge_types.update(reverse)
        n_relations = len(edge_types)
        edges = [[h, t] for k in sorted(edge_types) for h, t in edge_types.get(k)]
        edges_t = torch.LongTensor(edges).T

        id_et_map = {i: et for i, et in enumerate(sorted(edge_types))}

        g = dgl.graph((edges_t[0], edges_t[1]), num_nodes=n_nodes)
        type_label = torch.cat([torch.LongTensor([(i, float(k) if k in train_types else 0) for _ in edge_types.get(k)])
                                     for i, k in sorted(id_et_map.items())]).T
        g.edata['type'] = type_label[0]

        g.edata['label'] = type_label[1].to(torch.float)

        self.train_graph = g

        return n_nodes, n_relations

    def set_attention(self, g, verbose=True):
        with torch.no_grad():
            g.edata['a'] = self.model.compute_attention(g, self.batch_size, verbose=verbose).pin_memory()

    def fit(self, train_set: Dataset, val_set=None):
        from .kgat import Model
        import dgl
        from torch import optim
        from . import dgl_utils

        super().fit(train_set, val_set)
        n_nodes, n_relations = self._create_graphs(train_set)  # graphs are as attributes of model.

        # create model
        self.model = Model(n_nodes, n_relations, 64, 32, 3, [32, 16, 16])

        self.model.reset_parameters()

        print(sum(p.numel() for p in self.model.parameters()))

        if self.use_cuda:
            self.model = self.model.cuda()
            prefetch = ['label']
        else:
            prefetch = []

        # Edges where label is non-zero and user and item occurs more than once.
        g = self.train_graph
        u, v = g.edges()
        mask = g.edata['label'] != 0
        _, i, c = torch.unique(u, sorted=False, return_inverse=True, return_counts=True)
        mask *= c[i] > 1
        _, i, c = torch.unique(v, sorted=False, return_inverse=True, return_counts=True)
        mask *= (c[i] > 1)
        cf_eids = g.edges(form='eid')[mask]
        num_workers = self.num_workers

        # TransR edges.
        tr_edges = g.edges(form='eid')

        # Get reverse edge mapping.
        n_edges = g.num_edges()
        reverse_eids = torch.cat([torch.arange(n_edges // 2, n_edges), torch.arange(0, n_edges // 2)])

        if self.use_uva:
            # g = g.to(self.device)
            cf_eids = cf_eids.to(self.device)
            tr_edges = tr_edges.to(self.device)
            num_workers = 0

        if self.debug:
            num_workers = 0
            thread = False
        else:
            thread = None

        # Create samplers
        # TransR sampler
        sampler = dgl_utils.TransRSampler()
        neg_sampler = dgl.dataloading.negative_sampler.PerSourceUniform(1)
        sampler = dgl.dataloading.as_edge_prediction_sampler(sampler, negative_sampler=neg_sampler)
        tranr_dataloader = dgl.dataloading.DataLoader(g, tr_edges, sampler, batch_size=self.batch_size*3, shuffle=True,
                                                      drop_last=True, device=self.device, use_uva=self.use_uva,
                                                      num_workers=num_workers, use_prefetch_thread=thread)

        # CF sampler
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(3)
        sampler = dgl.dataloading.as_edge_prediction_sampler(sampler, exclude='reverse_id', reverse_eids=reverse_eids)
        cf_dataloader = dgl.dataloading.DataLoader(g, cf_eids, sampler, batch_size=self.batch_size, shuffle=True,
                                                drop_last=True, device=self.device, use_uva=self.use_uva,
                                                num_workers=num_workers, use_prefetch_thread=thread)

        # Initialize training params.
        tr_optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        cf_optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        length = len(cf_dataloader)

        best_state = None
        best_score = float('inf')
        for e in range(self.num_epochs):
            tot_tr = 0
            tot_mse = 0
            tot_l2 = 0
            tot_loss = 0

            # TransR
            with tqdm(tranr_dataloader) as progress:
                for i, (input_nodes, pos_subgraph, neg_subgraph, blocks) in enumerate(progress, 1):
                    x = self.model.node_embedding(input_nodes)
                    neg_subgraph.edata['type'] = pos_subgraph.edata['type']

                    pos_x = self.model.trans_r(pos_subgraph, x)
                    neg_x = self.model.trans_r(neg_subgraph, x)
                    loss = self.model.trans_r_loss(pos_x, neg_x)
                    loss += self.weight_decay * self.model.l2_loss(pos_subgraph, x, trans_r=True,
                                                                   neg_graph=neg_subgraph)
                    loss.backward()

                    tr_optimizer.step()
                    tr_optimizer.zero_grad()

                    tot_tr += loss.detach()

                    progress.set_description(f'Epoch {e}, '
                                             f'Loss: {tot_tr / i:.5f}')

            self.model.eval()
            self.set_attention(g)
            self.model.train()

            # CF
            with tqdm(cf_dataloader) as progress:
                for i, (input_nodes, edge_subgraph, blocks) in enumerate(progress, 1):
                    emb = self.model.node_embedding(input_nodes)
                    x = self.model(blocks, emb)

                    pred = self.model.graph_predict(edge_subgraph, x)

                    mse_loss = self.model.loss(pred, edge_subgraph.edata['label'])
                    l2_loss = self.weight_decay * self.model.l2_loss(edge_subgraph, emb)
                    loss = mse_loss + l2_loss
                    loss.backward()

                    tot_mse += mse_loss.detach()
                    tot_l2 += l2_loss.detach()
                    tot_loss += loss.detach()

                    cf_optimizer.step()
                    cf_optimizer.zero_grad()

                    if i != length or val_set is None:
                        progress.set_description(f'Epoch {e}, '
                                                 f'MSE:{tot_mse / i:.5f}'
                                                 f',L2:{tot_l2 / i:.3g}'
                                                 f',Tot:{tot_loss / i:.5f}'
                                                 )
                    else:
                        self.set_attention(g, verbose=False)
                        mse, rmse = self._validate(val_set)
                        progress.set_description(f'Epoch {e}, MSE: {tot_mse / i:.5f}, Val: MSE: {mse:.5f}, '
                                                 f'RMSE: {rmse:.5f}')

                        if self.model_selection == 'best' and mse < best_score:
                            best_state = deepcopy(self.model.state_dict())
                            best_score = mse

        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.set_attention(g)

        _ = self._validate(val_set)

    def _validate(self, val_set):
        from ...eval_methods.base_method import rating_eval
        from ...metrics import MSE, RMSE
        import torch

        self.model.eval()
        with torch.no_grad():
            x = self.model.node_embedding(self.train_graph.nodes().to(self.device))
            self.model.inference(self.train_graph, x, self.device)
            ((mse, rmse), _) = rating_eval(self, [MSE(), RMSE()], val_set, user_based=self.user_based)
        return mse, rmse

    def score(self, user_idx, item_idx=None):
        import torch

        self.model.eval()
        with torch.no_grad():
            user_idx = torch.tensor(user_idx + self.n_items, dtype=torch.int64).to(self.device)
            item_idx = torch.tensor(item_idx, dtype=torch.int64).to(self.device)
            return self.model.predict(user_idx, item_idx).cpu()

    def monitor_value(self):
        pass
