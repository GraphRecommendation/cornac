import collections
import os
import pickle
from collections import defaultdict
from contextlib import nullcontext
from copy import deepcopy
from itertools import combinations
from math import sqrt

import re

import numpy as np
import torch
from tqdm import tqdm

from ..recommender import Recommender
from ...data import Dataset
from ...utils.graph_construction import generate_mappings


class LightRLA(Recommender):
    def __init__(self, name='LightRLA', use_cuda=False, use_uva=False, stemming=True,
                 batch_size=128,
                 num_workers=0,
                 num_epochs=10,
                 early_stopping=10,
                 learning_rate=0.1,
                 weight_decay=0,
                 l2_weight=0.,
                 n_layers=5,
                 node_dim=64,
                 num_heads=3,
                 fanout=5,
                 use_relation=False,
                 non_linear=True,
                 model_selection='best',
                 objective='ranking',
                 review_aggregator='narre',
                 predictor='narre',
                 preference_module='lightgcn',
                 combiner='add',
                 graph_type='ao',
                 num_neg_samples=50,
                 layer_dropout=None,
                 attention_dropout=.2,
                 user_based=True,
                 verbose=True,
                 index=0,
                 use_tensorboard=True,
                 out_path=None,
                 popularity_biased_sampling=False,
                 learn_explainability=False,
                 learn_method='transr',
                 learn_weight=1.,
                 learn_pop_sampling=False,
                 debug=False
                 ):
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
        self.l2_weight = l2_weight
        self.n_layers = n_layers
        self.num_heads = num_heads
        self.fanout = fanout
        self.use_relation = use_relation
        self.non_linear = non_linear
        self.model_selection = model_selection
        self.objective = objective
        self.review_aggregator = review_aggregator
        self.predictor = predictor
        self.preference_module = preference_module
        self.combiner = combiner
        self.graph_type = graph_type
        self.num_neg_samples = num_neg_samples
        self.layer_dropout = layer_dropout
        self.attention_dropout = attention_dropout
        self.stemming = stemming
        self.learn_explainability = learn_explainability
        self.learn_method = learn_method
        self.learn_weight = learn_weight
        self.learn_pop_sampling = learn_pop_sampling
        self.popularity_biased_sampling = popularity_biased_sampling
        parameter_list = ['batch_size', 'learning_rate', 'weight_decay', 'node_dim', 'num_heads',
                          'fanout', 'use_relation', 'model_selection', 'review_aggregator', 'objective',
                          'predictor', 'preference_module', 'layer_dropout', 'attention_dropout', 'stemming',
                          'learn_explainability', 'learn_method', 'learn_weight', 'learn_pop_sampling',
                          'popularity_biased_sampling', 'combiner']
        self.parameters = collections.OrderedDict({k: self.__getattribute__(k) for k in parameter_list})

        # Method
        self.node_review_graph = None
        self.review_graphs = {}
        self.train_graph = None
        self.ui_graph = None
        self.model = None
        self.n_items = 0
        self.n_relations = 0
        self.ntype_ranges = None
        self.node_filter = None
        self.sid_aos = None
        self.aos_tensor = None

        # Misc
        self.user_based = user_based
        self.verbose = verbose
        self.debug = debug
        self.index = index
        self.out_path = out_path
        if use_tensorboard:
            assert out_path is not None, f'Must give a path if using tensorboard.'
            assert os.path.exists(out_path), f'{out_path} is not a valid path.'
            p = os.path.join(out_path, str(index))
            self.summary_writer = SummaryWriter(log_dir=p)

        # assertions
        assert use_uva == use_cuda or not use_uva, 'use_cuda must be true when using uva.'
        assert objective == 'ranking' or objective == 'rating', f'This method only supports ranking or rating, ' \
                                                                f'not {objective}.'

    def _create_graphs(self, train_set: Dataset, graph_type, self_loop=True):
        import dgl
        import torch

        # create 1) u,i,a, 2) u,i,o 3) u, a, o, 4) i, o, a
        sentiment_modality = train_set.sentiment
        edge_id = 0
        n_users = len(train_set.uid_map)
        n_items = len(train_set.iid_map)

        _, _, _, _, _, _, a2a, o2o = generate_mappings(train_set.sentiment, 'a', get_ao_mappings=True)

        n_aspects = max(a2a.values()) + 1 if self.stemming else len(sentiment_modality.aspect_id_map)
        n_opinions = max(o2o.values()) + 1 if self.stemming else len(sentiment_modality.opinion_id_map)
        n_nodes = n_users + n_items + n_aspects + n_opinions
        n_types = 4

        user_item_review_map = {(uid + n_items, iid): rid for uid, irid in sentiment_modality.user_sentiment.items()
                                for iid, rid in irid.items()}
        review_edges = []
        ratings = []
        review_graphs = {}
        for uid, isid in tqdm(sentiment_modality.user_sentiment.items(), desc='Creating review graphs',
                              total=len(sentiment_modality.user_sentiment), disable=not self.verbose):
            uid += n_items

            for iid, sid in isid.items():
                review_edges.extend([[sid, uid], [sid, iid]])
                ratings.extend([train_set.matrix[uid-n_items, iid]]*2)
                edges = []
                a_o_count = defaultdict(int)
                aos = sentiment_modality.sentiment[sid]
                sids = []
                for aid, oid, s in aos:
                    if self.stemming:
                        aid = a2a[aid]
                        oid = o2o[oid]

                    aid += n_items + n_users
                    oid += n_items + n_users + n_aspects
                    sent = 0 if s == -1 else 1

                    a_o_count[aid] += 1
                    a_o_count[oid] += 1
                    if graph_type == 'ao':
                        n_hyper_edges = 3
                        for f, s, t in [[uid, iid, aid], [uid, iid, oid], [uid, aid, oid], [iid, oid, aid]]:
                            edges.append([f, s, edge_id, sent])
                            edges.append([s, t, edge_id, sent])
                            edges.append([t, f, edge_id, sent])

                            if self_loop:
                                edges.append([f, f, edge_id, sent])
                                edges.append([s, s, edge_id, sent])
                                edges.append([t, t, edge_id, sent])
                            edge_id += 1
                    elif graph_type == 'ao-one':
                        n_hyper_edges = 1
                        options = [uid, iid, aid, oid]
                        self_loops = list(zip(options, options))

                        options = list(combinations(options, 2))
                        if self_loop:
                            options += self_loops

                        for s, d in options:
                            edges.append([s, d, edge_id, sent])

                        edge_id += 1
                    else:
                        raise NotImplementedError

                edges = torch.LongTensor(edges).T
                src, dst, eids, sents = edges
                g = dgl.graph((torch.cat([src, dst]), torch.cat([dst, src])), num_nodes=n_nodes)

                # All hyperedges connect 3 nodes
                g.edata['id'] = torch.cat([eids, eids])
                g.edata['norm'] = torch.full((g.num_edges(),), n_hyper_edges ** -1)

                # User and item have three hyper-edges for each AOS triple.
                g.ndata['norm'] = torch.zeros((g.num_nodes()))
                g.ndata['norm'][[uid, iid]] = sqrt(n_hyper_edges * len(aos)) ** -1

                u, v = g.edges()
                uids_mask = (u >= n_items) * (u < n_items + n_users)
                aids_mask = (u >= n_items + n_users) * (u < n_items + n_users + n_aspects)
                oids_mask = u >= n_items + n_users + n_aspects
                g.edata['type'] = torch.zeros((g.num_edges(),), dtype=torch.int64)
                g.edata['type'][uids_mask] = 1
                g.edata['type'][aids_mask] = 2
                g.edata['type'][oids_mask] = 3

                g.edata['sent'] = sents.repeat(2)

                # a and o also have three hyper edge triples for each occurrence.
                for nid, c in a_o_count.items():
                    g.ndata['norm'][nid] = sqrt(n_hyper_edges * c) ** -1

                assert len(edges.T) * 2 == g.num_edges()

                review_graphs[sid] = g

        # Create training graph, i.e. user to item graph.
        edges = [(uid + n_items, iid, train_set.matrix[uid, iid]) for uid, iid in zip(*train_set.matrix.nonzero())]
        t_edges = torch.LongTensor(edges).T
        train_graph = dgl.graph((t_edges[0], t_edges[1]))
        train_graph.edata['sid'] = torch.LongTensor([user_item_review_map[(u, i)] for (u, i, r) in edges])
        train_graph.edata['label'] = t_edges[2].to(torch.float)

        # Create user/item to review graph.
        edges = torch.LongTensor(review_edges).T
        node_review_graph = dgl.heterograph({('review', 'part_of', 'node'): (edges[0], edges[1])})

        # Assign edges node_ids s.t. an edge from user to review has the item nid its about and reversely.
        node_review_graph.edata['nid'] = torch.LongTensor(node_review_graph.num_edges())
        _, v, eids = node_review_graph.edges(form='all')
        node_review_graph.edata['nid'][eids % 2 == 0] = v[eids % 2 == 1]
        node_review_graph.edata['nid'][eids % 2 == 1] = v[eids % 2 == 0]

        # Scale ratings with denominator if not integers. I.e., if .25 multiply by 4.
        # A mapping from frac to int. If
        denominators = [e.as_integer_ratio()[1] for e in ratings]
        i = 0
        while any(d != 1 for d in denominators):
            ratings = ratings * max(denominators)
            denominators = [e.as_integer_ratio()[1] for e in ratings]
            i += 1
            assert i < 100, 'Tried to convert ratings to integers but took to long.'

        node_review_graph.edata['r_type'] = torch.LongTensor(ratings)-1

        ntype_ranges = {'item': (0, n_items), 'user': (n_items, n_items+n_users),
                        'aspect': (n_items+n_users, n_items+n_users+n_aspects),
                        'opinion': (n_items+n_users+n_aspects, n_items+n_users+n_aspects+n_opinions)}

        sid_aos = []
        for sid in range(max(train_set.sentiment.sentiment)+1):
            aoss = train_set.sentiment.sentiment.get(sid, [])
            sid_aos.append([(a2a[a]+n_items+n_users, o2o[o]+n_users+n_items+n_aspects, 0 if s == -1 else 1)
                            for a, o, s in aoss])

        aos_list = sorted({aos for aoss in sid_aos for aos in aoss})
        aos_id = {aos: i for i, aos in enumerate(aos_list)}
        sid_aos = [torch.LongTensor([aos_id[aos] for aos in aoss]) for aoss in sid_aos]

        return n_nodes, n_types, n_items, train_graph, review_graphs, node_review_graph, ntype_ranges, sid_aos, aos_list

    def _flock_wrapper(self, func, fname, *args, **kwargs):
        from filelock import FileLock

        fpath = os.path.join(self.out_path, fname)
        lock_fpath = os.path.join(self.out_path, fname + '.lock')

        with FileLock(lock_fpath):
            if os.path.exists(fpath):
                with open(fpath, 'rb') as f:
                    data = pickle.load(f)
            else:
                data = func(*args, **kwargs)
                with open(fpath, 'wb') as f:
                    pickle.dump(data, f)

        return data

    def _graph_wrapper(self, train_set, graph_type, *args):
        fname = f'graph_{graph_type}_data.pickle'
        data = self._flock_wrapper(self._create_graphs, fname, train_set, graph_type, *args)

        n_nodes, n_types, self.n_items, self.train_graph, self.review_graphs, self.node_review_graph, \
            self.ntype_ranges, sid_aos, aos_list = data
        self.node_filter = lambda t, nids: (nids >= self.ntype_ranges[t][0]) * (nids < self.ntype_ranges[t][1])
        return n_nodes, n_types, sid_aos, aos_list

    def _ao_embeddings(self, train_set):
        from gensim.models import Word2Vec
        from gensim.parsing import remove_stopwords, preprocess_string, stem_text
        import torch

        sentiment = train_set.sentiment

        # Define preprocess functions for text, aspects and opinions.
        preprocess_fn = lambda x: stem_text(re.sub(r'\s+', ' ', re.sub(r'--+|-+ ', ' ', re.sub(r'[^\w\-_]', ' ', x))))
        ao_preprocess_fn = lambda x: stem_text(re.sub(r'--+.*|-+$', '', x))

        # Process corpus
        corpus = [preprocess_fn(sentence)
                  for review in train_set.review_text.corpus
                  for sentence in review.split('.')]

        # Process aspects and opinions.
        a_old_new_map = {a: ao_preprocess_fn(a) for a in sentiment.aspect_id_map}
        o_old_new_map = {o: ao_preprocess_fn(o) for o in sentiment.opinion_id_map}

        # Define a progressbar for easier training.
        class CallbackProgressBar:
            def __init__(self, verbose):
                self.verbose = verbose
                self.progress = None

            def on_train_begin(self, method):
                if self.progress is None:
                    self.progress = tqdm(desc='Training Word2Vec', total=method.epochs, disable=not self.verbose)

            def on_train_end(self, method):
                pass

            def on_epoch_begin(self, method):
                pass

            def on_epoch_end(self, method):
                self.progress.update(1)

        # Get words and train model
        wc = [s.split(' ') for s in corpus]
        l = CallbackProgressBar(self.verbose)
        embedding_dim = 100
        w2v_model = Word2Vec(wc, vector_size=embedding_dim, min_count=1, window=5, callbacks=[l], epochs=100)

        # Keyvector model
        kv = w2v_model.wv

        # Initialize embeddings
        a_embeddings = np.zeros((len(sentiment.aspect_id_map), embedding_dim))
        o_embeddings = np.zeros((len(sentiment.opinion_id_map), embedding_dim))

        # Define function for assigning embeddings to correct aspect.
        def get_info(old_new_pairs, mapping, embedding):
            for old, new in old_new_pairs:
                nid = mapping[old]
                vector = np.array(kv.get_vector(new))
                embedding[nid] = vector

            return embedding

        a_embeddings = get_info(a_old_new_map.items(), sentiment.aspect_id_map, a_embeddings)
        o_embeddings = get_info(o_old_new_map.items(), sentiment.opinion_id_map, o_embeddings)

        return a_embeddings, o_embeddings, kv

    def _ui_embeddings(self, train_set):
        from sentence_transformers import SentenceTransformer

        # Create sentence embeddings using sentence bert
        user_sentences = defaultdict(list)
        item_sentences = defaultdict(list)
        corpus = []
        index = 0
        for uid, irid in train_set.review_text.user_review.items():
            for iid, rid in irid.items():
                for s in train_set.review_text.reviews[rid].split('.'):
                    user_sentences[uid].append(index)
                    item_sentences[iid].append(index)
                    corpus.append(s)
                    index += 1

        # https://huggingface.co/sentence-transformers/all-mpnet-base-v2
        model = SentenceTransformer('all-mpnet-base-v2', device='cuda')
        embedding_dim = model[1].pooling_output_dimension
        u_embeddings = np.zeros((train_set.num_users, embedding_dim))
        i_embeddings = np.zeros((train_set.num_items, embedding_dim))

        def generate_embeddings(id_sentences, embeddings, sentence_embeddings):
            for nid, sents in id_sentences:
                embeddings[nid] = sentence_embeddings[sents].mean(0)

            return embeddings

        sent_embeddings = model.encode(corpus)
        u_embeddings = generate_embeddings(user_sentences.items(), u_embeddings, sent_embeddings)
        i_embeddings = generate_embeddings(item_sentences.items(), i_embeddings, sent_embeddings)

        return u_embeddings, i_embeddings

    def _normalize_embedding(self, embedding):
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaler.fit(embedding)
        return scaler.transform(embedding), scaler

    def _learn_initial_embeddings(self, train_set):
        ao_fname = 'ao_embeddings.pickle'
        ui_fname = 'ui_embeddings.pickle'
        a_fname = 'aspect_embeddings.pickle'
        o_fname = 'opinion_embeddings.pickle'
        u_fname = 'user_embeddings.pickle'
        i_fname = 'item_embeddings.pickle'

        # Get embeddings and store result
        a_embeddings, o_embeddings, _ = self._flock_wrapper(self._ao_embeddings, ao_fname, train_set)
        u_embeddings, i_embeddings = self._flock_wrapper(self._ui_embeddings, ui_fname, train_set)

        # Scale embeddings and store results. Function returns scaler, which is not needed, but required if new data is
        # added.
        a_embeddings, _ = self._flock_wrapper(self._normalize_embedding, a_fname, a_embeddings)
        o_embeddings, _ = self._flock_wrapper(self._normalize_embedding, o_fname, o_embeddings)
        u_embeddings, _ = self._flock_wrapper(self._normalize_embedding, u_fname, u_embeddings)
        i_embeddings, _ = self._flock_wrapper(self._normalize_embedding, i_fname, i_embeddings)

        return torch.tensor(a_embeddings), torch.tensor(o_embeddings), torch.tensor(u_embeddings), \
            torch.tensor(i_embeddings)

    def fit(self, train_set: Dataset, val_set=None):
        from .lightrla import Model
        from cornac.models import NGCF

        super().fit(train_set, val_set)
        n_nodes, self.n_relations, self.sid_aos, self.aos_list = self._graph_wrapper(train_set, self.graph_type)  # graphs are as attributes of model.
        # a_embs, o_embs, u_embs, i_embs = self._learn_initial_embeddings(train_set)

        if not self.use_relation:
            self.n_relations = 0

        self.ui_graph = NGCF.construct_graph(train_set, False)
        n_r_types = max(self.node_review_graph.edata['r_type']) + 1
        n_sentiments = len(set([aos[2] for sid in self.train_set.sentiment.sentiment.values() for aos in sid]))

        # create model
        self.model = Model(self.ui_graph, n_nodes, self.n_relations, n_r_types, self.review_aggregator,
                           self.predictor, self.node_dim, self.num_heads, [self.layer_dropout]*2,
                           self.attention_dropout, self.preference_module, self.use_cuda, combiner=self.combiner,
                           aos_predictor=self.learn_method, non_linear=self.non_linear)

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
                                             self.sid_aos, self.aos_list, 5,
                                             self.ui_graph, fanout=self.fanout, hard_negatives=self.learn_pop_sampling)
        if self.objective == 'ranking':
            ic = collections.Counter(self.train_set.matrix.nonzero()[1])
            probabilities = torch.FloatTensor([ic.get(i) for i in sorted(ic)]) if self.popularity_biased_sampling \
                else None
            neg_sampler = cornac.utils.dgl.GlobalUniformItemSampler(self.num_neg_samples, self.train_set.num_items,
                                                                    probabilities)
        else:
            neg_sampler = None

        sampler = dgl_utils.HEAREdgeSampler(sampler, prefetch_labels=prefetch, negative_sampler=neg_sampler,
                                            exclude='self')
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
        best_epoch = 0
        epoch_length = len(dataloader)
        all_nodes = torch.arange(next(iter(self.review_graphs.values())).num_nodes()).to(self.device)
        for e in range(self.num_epochs):
            tot_losses = defaultdict(int)
            cur_losses = {}
            self.model.train()

            with (dataloader.enable_cpu_affinity() if num_workers else nullcontext()):
                with tqdm(dataloader, disable=not self.verbose) as progress:
                    for i, batch in enumerate(progress, 1):
                        if self.objective == 'ranking':
                            input_nodes, edge_subgraph, neg_subgraph, blocks = batch
                        else:
                            input_nodes, edge_subgraph, blocks = batch

                        x = self.model(blocks, self.model.get_initial_embedings(all_nodes), input_nodes)

                        pred = self.model.graph_predict(edge_subgraph, x)

                        if self.objective == 'ranking':
                            pred_j = self.model.graph_predict(neg_subgraph, x).reshape(-1, self.num_neg_samples)
                            pred_j = pred_j.reshape(-1, self.num_neg_samples)
                            acc = (pred > pred_j).sum() / pred_j.shape.numel()
                            loss = self.model.ranking_loss(pred, pred_j)
                            cur_losses['loss'] = loss.detach()
                            cur_losses['acc'] = acc.detach()

                            if self.l2_weight:
                                l2 = self.l2_weight * self.model.l2_loss(edge_subgraph, neg_subgraph, x)
                                loss += l2
                                cur_losses['l2'] = l2.detach()
                        else:
                            loss = self.model.rating_loss(pred, edge_subgraph.edata['label'])
                            cur_losses['loss'] = loss.detach()

                        if self.learn_explainability:
                            aos_loss, aos_acc = self.model.aos_graph_predict(edge_subgraph, x)
                            aos_loss = aos_loss.mean()
                            cur_losses['aos_loss'] = aos_loss.detach()
                            cur_losses['aos_acc'] = (aos_acc.sum() / aos_acc.shape.numel()).detach()
                            loss += self.learn_weight * aos_loss

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

                            if self.summary_writer is not None:
                                for m, r in zip(metrics, results):
                                    self.summary_writer.add_scalar(f'val/{m.name}', r, e)

                            if self.model_selection == 'best' and (results[0] > best_score if metrics[0].higher_better
                                    else results[0] < best_score):
                                best_state = deepcopy(self.model.state_dict())
                                best_score = results[0]
                                best_epoch = e

            if self.early_stopping is not None and (e - best_epoch) >= self.early_stopping:
                break

        del self.node_filter
        del g, eids
        del dataloader
        del sampler

        if best_state is not None:
            self.model.load_state_dict(best_state)

        if val_set is not None and self.summary_writer is not None:
            results = self._validate(val_set, metrics)
            self.summary_writer.add_hparams(dict(self.parameters), dict(zip([m.name for m in metrics], results)))

        self.model.eval()
        with torch.no_grad():
            self.model.inference(self.review_graphs, self.node_review_graph, self.ui_graph, self.device, self.batch_size)

        self.best_epoch = best_epoch
        self.best_value = best_score

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
                pred = self.model.predict(user_idx, item_idx).reshape(-1).cpu().numpy()
            else:
                item_idx = torch.tensor(item_idx, dtype=torch.int64).to(self.device)
                pred = self.model.predict(user_idx, item_idx).cpu()

            return pred

    def monitor_value(self):
        pass

    def save(self, save_dir=None):
        import torch
        import pandas as pd

        if save_dir is None:
            return

        path = super().save(save_dir)
        name = path.rsplit('/', 1)[-1].replace('pkl', 'pt')

        state = self.model.state_dict()
        torch.save(state, os.path.join(save_dir, str(self.index), name))

        # results_path = os.path.join(path.rsplit('/', 1)[0], 'results.csv')
        # header = not os.path.exists(results_path)
        # self.parameters['score'] = self.best_value
        # self.parameters['epoch'] = self.best_epoch
        # self.parameters['file'] = path.rsplit('/')[-1]
        # self.parameters['id'] = self.index
        # df = pd.DataFrame({k: [v] for k, v in self.parameters.items()})
        # df.to_csv(results_path, header=header, mode='a', index=False)

        return path


