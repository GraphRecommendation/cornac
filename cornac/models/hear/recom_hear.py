import os
from collections import defaultdict
from copy import deepcopy
from math import sqrt

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ..recommender import Recommender
from ...data import Dataset


class HEAR(Recommender):
    def __init__(self, name='HEAR', use_cuda=False, use_uva=False,
                 batch_size=128,
                 num_workers=0,
                 num_epochs=10,
                 early_stopping=10,
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
                 verbose=True,
                 index=0,
                 use_tensorboard=True,
                 out_path=None,
                 debug=False
                 ):
        super().__init__(name)
        # Default values
        if layer_dropout is None:
            layer_dropout = 0.  # node embedding dropout, review embedding dropout

        # CUDA
        self.use_cuda = use_cuda
        self.device = 'cuda' if use_cuda else 'cpu'
        self.use_uva = use_uva

        # Parameters
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_epochs = num_epochs
        self.early_stopping = early_stopping
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
        parameter_list = ['batch_size', 'learning_rate', 'weight_decay', 'node_dim', 'review_dim',
                          'final_dim', 'num_heads', 'fanout', 'model_selection', 'review_aggregator',
                          'predictor', 'layer_dropout', 'attention_dropout']
        self.parameters = {k: self.__getattribute__(k) for k in parameter_list}

        # Method
        self.node_review_graph = None
        self.review_graphs = {}
        self.train_graph = None
        self.model = None
        self.n_items = 0

        # Misc
        self.user_based = user_based
        self.verbose = verbose
        self.debug = debug
        self.index = index
        if use_tensorboard:
            assert out_path is not None, f'Must give a path if using tensorboard.'
            assert os.path.exists(out_path), f'{out_path} is not a valid path.'
            p = os.path.join(out_path, str(index))
            self.summary_writer = SummaryWriter(log_dir=p)

        # assertions
        assert use_uva == use_cuda or not use_uva, 'use_cuda must be true when using uva.'

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
        for uid, isid in tqdm(sentiment_modality.user_sentiment.items(), desc='Creating review graphs',
                              total=len(sentiment_modality.user_sentiment), disable=not self.verbose):
            uid += n_items

            for iid, sid in isid.items():
                review_edges.extend([[sid, uid], [sid, iid]])
                edges = []
                a_o_count = defaultdict(int)
                aos = sentiment_modality.sentiment[sid]
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

                self.review_graphs[sid] = g

        # Create training graph, i.e. user to item graph.
        edges = [(uid + n_items, iid, train_set.matrix[uid, iid]) for uid, iid in zip(*train_set.matrix.nonzero())]
        t_edges = torch.LongTensor(edges).T
        self.train_graph = dgl.graph((t_edges[0], t_edges[1]))
        self.train_graph.edata['sid'] = torch.LongTensor([user_item_review_map[(u, i)] for (u, i, r) in edges])
        self.train_graph.edata['label'] = t_edges[2].to(torch.float)

        # Create user/item to review graph.
        edges = torch.LongTensor(review_edges).T
        self.node_review_graph = dgl.heterograph({('review', 'part_of', 'node'): (edges[0], edges[1])})

        # Assign edges node_ids s.t. an edge from user to review has the item nid its about.
        self.node_review_graph.edata['nid'] = torch.LongTensor(self.node_review_graph.num_edges())
        _, v, eids = self.node_review_graph.edges(form='all')
        self.node_review_graph.edata['nid'][eids % 2 == 0] = v[eids % 2 == 1]
        self.node_review_graph.edata['nid'][eids % 2 == 1] = v[eids % 2 == 0]

        return n_nodes

    def fit(self, train_set: Dataset, val_set=None):
        from .hear import Model

        super().fit(train_set, val_set)
        n_nodes = self._create_graphs(train_set)  # graphs are as attributes of model.

        # create model
        self.model = Model(n_nodes, self.review_aggregator, self.predictor, self.node_dim,
                           self.review_dim, self.final_dim, self.num_heads, [self.layer_dropout]*2,
                           self.attention_dropout)

        self.model.reset_parameters()

        if self.verbose:
            print(f'Number of trainable parameters: {sum(p.numel() for p in self.model.parameters())}')

        if self.use_cuda:
            self.model = self.model.cuda()
            prefetch = ['label']
        else:
            prefetch = []

        if self.trainable:
            self._fit(prefetch, val_set)

        if self.summary_writer is not None:
            self.summary_writer.close()

        return self

    def _fit(self, prefetch, val_set=None):
        import dgl
        from torch import optim
        from . import dgl_utils

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
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        length = len(dataloader)

        best_state = None
        best_score = float('inf')
        no_improv = 0
        epoch_length = len(dataloader)
        for e in range(self.num_epochs):
            tot_mse = 0
            # tot_l2 = 0
            # tot_loss = 0
            with tqdm(dataloader, disable=not self.verbose) as progress:
                for i, (input_nodes, edge_subgraph, blocks) in enumerate(progress, 1):
                    x = self.model(blocks, self.model.node_embedding(input_nodes))

                    pred = self.model.graph_predict(edge_subgraph, x)

                    mse_loss = self.model.loss(pred, edge_subgraph.edata['label'])
                    # l2_loss = self.weight_decay * self.model.l2_loss(input_nodes)
                    loss = mse_loss  # + l2_loss
                    loss.backward()

                    tot_mse += mse_loss.detach()
                    # tot_l2 += l2_loss.detach()
                    # tot_loss += loss.detach()

                    if self.summary_writer is not None:
                        self.summary_writer.add_scalar('train/mse', mse_loss, e*epoch_length+i)

                    optimizer.step()
                    optimizer.zero_grad()
                    if i != length or val_set is None:
                        progress.set_description(f'Epoch {e}, '
                                                 f'MSE: {tot_mse / i:.5f}'
                                                 # f', L2: {tot_l2 / i:.5f}'
                                                 # f', Tot: {tot_loss / i:.5f}'
                                                 )
                    else:
                        mse, rmse = self._validate(val_set)
                        progress.set_description(f'Epoch {e}, MSE: {tot_mse / i:.5f}, Val: MSE: {mse:.5f}, '
                                                 f'RMSE: {rmse:.5f}')

                        if self.summary_writer is not None:
                            self.summary_writer.add_scalar('val/mse', mse, e)
                            self.summary_writer.add_scalar('val/rmse', rmse, e)

                        if self.model_selection == 'best' and mse < best_score:
                            best_state = deepcopy(self.model.state_dict())
                            best_score = mse
                        else:
                            no_improv += 1

            if self.early_stopping is not None and no_improv >= self.early_stopping:
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        if val_set is not None and self.summary_writer is not None:
            mse, rmse = self._validate(val_set)
            self.summary_writer.add_hparams(self.parameters, {'mse': mse, 'rmse': rmse})

        self.model.eval()
        with torch.no_grad():
            self.model.inference(self.review_graphs, self.node_review_graph, self.device)

    def _validate(self, val_set):
        from ...eval_methods.base_method import rating_eval
        from ...metrics import MSE, RMSE
        import torch

        self.model.eval()
        with torch.no_grad():
            self.model.inference(self.review_graphs, self.node_review_graph, self.device)
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

    def save(self, save_dir=None):
        if save_dir is None:
            return

        path = super().save(save_dir)
        name = path.rsplit('/', 1)[-1].replace('pkl', 'pt')

        state = self.model.state_dict()
        torch.save(state, os.path.join(save_dir, str(self.index), name))
