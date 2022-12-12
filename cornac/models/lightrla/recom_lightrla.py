import json
import os
import pickle
from collections import defaultdict, Counter
from copy import deepcopy
from math import sqrt
from tqdm import tqdm

from nltk.stem import PorterStemmer

from ..recommender import Recommender
from ...data import Dataset


class LightRLA(Recommender):
    def __init__(self, name='LightRLA', use_cuda=False, use_uva=False, stemming=True,
                 batch_size=128,
                 num_workers=0,
                 num_epochs=10,
                 early_stopping=10,
                 learning_rate=0.1,
                 weight_decay=0,
                 l2_weight=0.,
                 node_dim=64,
                 review_dim=32,
                 final_dim=16,
                 num_heads=3,
                 fanout=5,
                 model_selection='best',
                 objective='ranking',
                 ranking_loss='bpr',
                 review_aggregator='narre',
                 predictor='gatv2',
                 learned_embeddings=False,
                 learned_preference=False,
                 learned_node_embeddings=None,
                 preference_module=None,
                 num_neg_samples=50,
                 margin=0.9,
                 neg_weight=500,
                 layer_dropout=None,
                 attention_dropout=.2,
                 user_based=True,
                 verbose=True,
                 index=0,
                 use_tensorboard=True,
                 out_path=None,
                 debug=False,
                 use_transr=True
                 ):
        self.l2_weight = l2_weight
        from torch.utils.tensorboard import SummaryWriter

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
        self.learned_embeddings = learned_embeddings
        self.learned_preference = learned_preference
        self.learned_node_embeddings = learned_node_embeddings
        self.preference_module = preference_module
        self.model_selection = model_selection
        self.objective = objective
        self.ranking_loss = ranking_loss
        self.review_aggregator = review_aggregator
        self.predictor = predictor
        self.num_neg_samples = num_neg_samples
        self.margin = margin
        self.neg_weight = neg_weight
        self.layer_dropout = layer_dropout
        self.attention_dropout = attention_dropout
        self.use_transr = use_transr
        parameter_list = ['batch_size', 'learning_rate', 'weight_decay', 'node_dim', 'review_dim',
                          'final_dim', 'num_heads', 'fanout', 'model_selection', 'review_aggregator',
                          'predictor', 'layer_dropout', 'attention_dropout', 'use_transr']
        self.parameters = {k: self.__getattribute__(k) for k in parameter_list}

        # Method
        self.node_review_graph = None
        self.review_graphs = {}
        self.train_graph = None
        self.ui_graph = None
        self.model = None
        self.n_items = 0
        self.plateaued = False
        self.stemming = stemming
        self.ntype_ranges = None
        self.node_filter = None

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
        assert objective == 'ranking' or objective == 'rating', f'This method only supports ranking or rating, ' \
                                                                f'not {objective}.'
        assert ranking_loss == 'bpr' or ranking_loss == 'ccl', f'Only bpr and ccl are supported not {ranking_loss}.'
        assert not (self.learned_preference or self.learned_embeddings) or self.learned_node_embeddings is not None, \
            'If using learned preference or learned embeddings, then learned node embeddings must be passed as' \
            'an argument.'
        assert (self.preference_module is None) ** self.learned_preference, \
            'Cannot use both learned preference embeddings and use another preference module.'

    def _create_graphs(self, train_set: Dataset):
        import dgl
        import torch

        # create 1) u,i,a, 2) u,i,o 3) u, a, o, 4) i, o, a
        sentiment_modality = train_set.sentiment
        edge_id = 0
        n_users = len(train_set.uid_map)
        n_items = len(train_set.iid_map)
        self.n_items = n_items

        def get_map(mapping, stemmer):
            id_to_new_word = {i: stemmer.stem(w) for w, i in mapping.items()}
            word_to_new_id = {w: i for i, w in enumerate(sorted(set(id_to_new_word.values())))}
            id_to_new_id = {i: word_to_new_id[w] for i, w in id_to_new_word.items()}
            return id_to_new_id

        stemmer = PorterStemmer()
        a2a = get_map(sentiment_modality.aspect_id_map, stemmer)
        o2o = get_map(sentiment_modality.opinion_id_map, stemmer)

        n_aspects = len(a2a) if self.stemming else len(sentiment_modality.aspect_id_map)
        n_opinions = len(o2o) if self.stemming else len(sentiment_modality.opinion_id_map)
        n_nodes = n_users + n_items + n_aspects + n_opinions

        user_item_review_map = {(uid + n_items, iid): rid for uid, irid in sentiment_modality.user_sentiment.items()
                                for iid, rid in irid.items()}
        review_edges = []
        ratings = []
        for uid, isid in tqdm(sentiment_modality.user_sentiment.items(), desc='Creating review graphs',
                              total=len(sentiment_modality.user_sentiment), disable=not self.verbose):
            uid += n_items

            for iid, sid in isid.items():
                review_edges.extend([[sid, uid], [sid, iid]])
                ratings.extend([train_set.matrix[uid-n_items, iid]]*2)
                edges = []
                a_o_count = defaultdict(int)
                aos = sentiment_modality.sentiment[sid]
                for aid, oid, _ in aos:
                    if self.stemming:
                        aid = a2a[aid]
                        oid = o2o[oid]

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

        # Assign edges node_ids s.t. an edge from user to review has the item nid its about and reversely.
        self.node_review_graph.edata['nid'] = torch.LongTensor(self.node_review_graph.num_edges())
        _, v, eids = self.node_review_graph.edges(form='all')
        self.node_review_graph.edata['nid'][eids % 2 == 0] = v[eids % 2 == 1]
        self.node_review_graph.edata['nid'][eids % 2 == 1] = v[eids % 2 == 0]

        # Scale ratings with denominator if not integers. I.e., if .25 multiply by 4.
        # A mapping from frac to int. If
        denominators = [e.as_integer_ratio()[1] for e in ratings]
        i = 0
        while any(d != 1 for d in denominators):
            ratings = ratings * max(denominators)
            denominators = [e.as_integer_ratio()[1] for e in ratings]
            i += 1
            assert i < 100, 'Tried to convert ratings to integers but took to long.'

        self.node_review_graph.edata['r_type'] = torch.LongTensor(ratings)-1

        self.ntype_ranges = {'item': (0, n_items), 'user': (n_items, n_items+n_users),
                        'aspect': (n_items+n_users, n_items+n_users+n_aspects),
                        'opinion': (n_items+n_users+n_aspects, n_items+n_users+n_aspects+n_opinions)}
        self.node_filter = lambda t, nids: (nids >= self.ntype_ranges[t][0]) * (nids < self.ntype_ranges[t][1])

        return n_nodes

    def fit(self, train_set: Dataset, val_set=None):
        from .lightrla import Model
        from cornac.models import NGCF

        super().fit(train_set, val_set)
        n_nodes = self._create_graphs(train_set)  # graphs are as attributes of model.
        self.ui_graph = NGCF.construct_graph(train_set, False)
        n_r_types = max(self.node_review_graph.edata['r_type']) + 1
        n_sentiments = len(set([aos[2] for sid in self.train_set.sentiment.sentiment.values() for aos in sid]))

        # create model
        self.model = Model(self.ui_graph, n_nodes, n_r_types, n_sentiments, self.review_aggregator, self.predictor, self.node_dim,
                           self.review_dim, self.final_dim, 32, self.num_heads, [self.layer_dropout]*2,
                           self.attention_dropout, learned_embeddings=self.learned_embeddings,
                           learned_preference=self.learned_preference,
                           learned_node_embeddings=self.learned_node_embeddings)

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
        import torch
        from torch import optim
        from . import dgl_utils
        import cornac

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
        sampler = dgl_utils.HearBlockSampler(self.node_review_graph, self.review_graphs, self.review_aggregator,
                                             self.ui_graph, fanout=self.fanout)
        if self.objective == 'ranking':
            neg_sampler = cornac.utils.dgl.GlobalUniformItemSampler(self.num_neg_samples, self.train_set.num_items)
        else:
            neg_sampler = None

        sampler = dgl_utils.HEAREdgeSampler(sampler, prefetch_labels=prefetch, negative_sampler=neg_sampler)
        dataloader = dgl.dataloading.DataLoader(g, eids, sampler, batch_size=self.batch_size, shuffle=True,
                                                drop_last=True, device=self.device, use_uva=self.use_uva,
                                                num_workers=num_workers, use_prefetch_thread=thread)

        # Initialize training params.
        optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        if self.objective == 'ranking':
            metrics = [cornac.metrics.NDCG(), cornac.metrics.AUC(), cornac.metrics.MAP(), cornac.metrics.MRR()]
        else:
            metrics = [cornac.metrics.MSE()]

        best_state = None
        best_score = 0 if metrics[0].higher_better else float('inf')
        # patience = self.early_stopping // 3 if self.early_stopping is not None else 10
        # print(f'Setting patience to {patience}')
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max' if metrics[0].higher_better else 'min',
        #                                    patience=patience)
        best_epoch = 0
        epoch_length = len(dataloader)
        for e in range(self.num_epochs):
            tot_losses = defaultdict(int)
            cur_losses = {}
            self.model.train()

            with tqdm(dataloader, disable=not self.verbose) as progress:
                for i, batch in enumerate(progress, 1):
                    if self.objective == 'ranking':
                        input_nodes, edge_subgraph, neg_subgraph, blocks = batch
                    else:
                        input_nodes, edge_subgraph, blocks = batch

                    # with torch.autocast(self.device):
                    x = self.model(blocks, self.model.get_initial_embedings(input_nodes))

                    pred = self.model.graph_predict(edge_subgraph, x)

                    if self.objective == 'ranking':
                        pred_j = self.model.graph_predict(neg_subgraph, x, self.plateaued).reshape(-1, self.num_neg_samples)
                        acc = (pred > pred_j).sum() / pred_j.shape.numel()
                        loss = self.model.ranking_loss(pred, pred_j, self.ranking_loss, self.neg_weight, self.margin)
                        cur_losses['loss'] = loss.detach()
                        cur_losses['acc'] = acc.detach()

                        if self.l2_weight:
                            l2 = self.model.l2_loss(edge_subgraph, neg_subgraph, x)
                            loss += self.l2_weight * l2
                            cur_losses['l2'] = l2.detach()
                    else:
                        loss = self.model.rating_loss(pred, edge_subgraph.edata['label'])
                        cur_losses['loss'] = loss.detach()

                    loss.backward()

                    for k, v in cur_losses.items():
                        tot_losses[k] += v

                    if self.summary_writer is not None:
                        for k, v in cur_losses.items():
                            self.summary_writer.add_scalar(f'train/cf/{k}', v, e * epoch_length + i)

                    optimizer.step()
                    optimizer.zero_grad()
                    loss_str = ','.join([f'{k}:{v/i:.3f}' for k, v in tot_losses.items()])
                    if i != epoch_length or val_set is None:
                        progress.set_description(f'Epoch {e}, ' + loss_str)
                    else:
                        results = self._validate(val_set, metrics)
                        res_str = 'Val: ' + ', '.join([f'{m.name}:{r:.4f}' for m, r in zip(metrics, results)])
                        progress.set_description(f'Epoch {e}, ' + f'{loss_str}, ' + res_str)
                        
                        # if scheduler is not None:
                        #     scheduler.step(results[0])

                        if self.summary_writer is not None:
                            for m, r in zip(metrics, results):
                                self.summary_writer.add_scalar(f'val/{m.name}', r, e)

                        if self.model_selection == 'best' and (results[0] > best_score if metrics[0].higher_better
                                else results[0] < best_score):
                            best_state = deepcopy(self.model.state_dict())
                            best_score = results[0]
                            best_epoch = e

            if self.early_stopping is not None and (e - best_epoch) >= self.early_stopping:
                if True or self.plateaued:
                    break
                else:
                    print(f'Plateaued, using adding learned preference.')
                    self.model.eval()
                    with torch.no_grad():
                        self.model.load_state_dict(best_state)
                        self.model.inference(self.review_graphs, self.node_review_graph, self.ui_graph, self.device, self.batch_size)
                        from cornac.eval_methods import ranking_eval
                        user_c = {int(k): int(v) for k, v in Counter(self.train_set.matrix.nonzero()[0]).items()}
                        (_, u_score1) = ranking_eval(self, metrics, self.train_set, val_set)
                        self.plateaued = True
                        (_, u_score2) = ranking_eval(self, metrics, self.train_set, val_set)
                        u_score1, u_score2 = [[{int(k): float(v) for k, v in d.items()} for d in l] for l in [u_score1, u_score2]]

                    with open(os.path.join(self.summary_writer.log_dir, 'scores.pickle'), 'wb') as f:
                        pickle.dump({'uc': user_c, 'us1': u_score1[0], 'us2': u_score2[0],
                                     'ratings': self.train_set.matrix}, f)


                    best_epoch = e  # reset best state
                    if self.model_selection == 'best':
                        print('Using best state for fine tuning.')
                        results = self._validate(val_set, metrics)
                        best_score = results[0]
                        print(f'Best score with preference: {best_score}')
                        # parameters = [p for n, p in self.model.named_parameters() if n.startswith('w_1')]
                        # optimizer = optim.Adam(parameters, lr=self.learning_rate, weight_decay=self.weight_decay)
                        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max' if metrics[0].higher_better else 'min',
                        #                    patience=patience)

        del self.node_filter
        del g, eids
        del dataloader
        del sampler

        if best_state is not None:
            self.model.load_state_dict(best_state)

        if val_set is not None and self.summary_writer is not None:
            results = self._validate(val_set, metrics)
            self.summary_writer.add_hparams(self.parameters, dict(zip([m.name for m in metrics], results)))

        self.model.eval()
        with torch.no_grad():
            self.model.inference(self.review_graphs, self.node_review_graph, self.ui_graph, self.device, self.batch_size)

    def _validate(self, val_set, metrics):
        from ...eval_methods.base_method import rating_eval, ranking_eval
        import torch

        self.model.eval()
        with torch.no_grad():
            self.model.inference(self.review_graphs, self.node_review_graph, self.ui_graph, self.device, self.batch_size)
            if self.objective == 'ranking':
                (result, _) = ranking_eval(self, metrics, self.train_set, val_set)
            else:
                (result, _) = rating_eval(self, metrics, val_set, user_based=self.user_based)
        return result

    def score(self, user_idx, item_idx=None):
        import torch

        self.model.eval()
        with torch.no_grad():
            user_idx = torch.tensor(user_idx + self.n_items, dtype=torch.int64).to(self.device)
            if item_idx is None:
                item_idx = torch.arange(self.n_items, dtype=torch.int64).to(self.device)
                pred = self.model.predict(user_idx, item_idx, self.plateaued).reshape(-1).cpu().numpy()
            else:
                item_idx = torch.tensor(item_idx, dtype=torch.int64).to(self.device)
                pred = self.model.predict(user_idx, item_idx, self.plateaued).cpu()

            return pred

    def monitor_value(self):
        pass

    def save(self, save_dir=None):
        import torch

        if save_dir is None:
            return

        path = super().save(save_dir)
        name = path.rsplit('/', 1)[-1].replace('pkl', 'pt')

        state = self.model.state_dict()
        torch.save(state, os.path.join(save_dir, str(self.index), name))
