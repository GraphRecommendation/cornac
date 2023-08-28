import os
from collections import defaultdict
from contextlib import nullcontext
from copy import deepcopy

import torch
from tqdm import tqdm

import cornac.metrics
from ..recommender import Recommender
from ...data import Dataset


class R3(Recommender):
    def __init__(self, name='R3', use_cuda=False, use_uva=False,
                 batch_size=64,
                 num_workers=0,
                 num_epochs=10,
                 learning_rate=0.1,
                 l2_weight=0,
                 node_dim=64,
                 layer_dims=None,
                 model_selection='best',
                 objective='ranking',
                 early_stopping=None,
                 layer_dropout=.1,
                 user_based=True,
                 debug=False,
                 verbose=True,
                 index=0,
                 use_tensorboard=True,
                 out_path=None,
                 ):

        from torch.utils.tensorboard import SummaryWriter

        super(R3, self).__init__(name)
        # Default values
        if layer_dims is None:
            layer_dims = [64, 32, 16]

        # CUDA
        self.use_cuda = use_cuda
        self.device = 'cuda' if use_cuda else 'cpu'
        self.use_uva = use_uva

        # Parameters
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.l2_weight = l2_weight
        self.node_dim = node_dim
        self.layer_dims = layer_dims
        self.model_selection = model_selection
        self.objective = objective
        self.early_stopping = early_stopping
        self.layer_dropout = layer_dropout
        self.lambda_ = 1
        self.alpha = 1

        parameter_list = ['batch_size', 'learning_rate', 'l2_weight', 'node_dim', 'layer_dims',
                          'layer_dropout', 'model_selection']
        self.parameters = {}
        for k in parameter_list:
            attr = self.__getattribute__(k)
            if isinstance(attr, list):
                for i in range(len(attr)):
                    self.parameters[f'{k}_{i}'] = attr[i]
            else:
                self.parameters[k] = attr

        # Method
        self.train_graph, self.review_graph, self.word_graph = None, None, None
        self.model = None

        # Misc
        self.user_based = user_based
        self.debug = debug
        self.verbose = verbose
        self.index = index
        if use_tensorboard:
            assert out_path is not None, f'Must give a path if using tensorboard.'
            assert os.path.exists(out_path), f'{out_path} is not a valid path.'
            p = os.path.join(out_path, str(index))
            self.summary_writer = SummaryWriter(log_dir=p)

        # assertions
        assert use_uva == use_cuda or not use_uva, 'use_cuda must be true when using uva.'
        assert objective == 'ranking' or objective == 'rating', f'This method only supports ranking or rating, '

    def fit(self, train_set: Dataset, val_set=None):
        from .r3 import Model
        import torch, dgl

        super().fit(train_set, val_set)

        # create model
        self.model = Model(train_set)
        # self.model.reset_parameters()

        self.train_graph, self.review_graph, self.word_graph = self._create_graphs()

        if self.verbose:
            print(f'Number of trainable parameters: {sum(p.numel() for p in self.model.parameters())}')

        if self.use_cuda:
            self.model = self.model.cuda()

        if self.trainable:
            self._fit(val_set)

        if self.summary_writer is not None:
            self.summary_writer.close()

        return self

    def _get_review_edges(self, mappings):
        nodes, reviews = [], []
        for src, pairs in mappings.items():
            for _, r in pairs.items():
                nodes.append(src)
                reviews.append(r)

        return nodes, reviews

    def _create_graphs(self):
        """
        Creates three graphs, one containing user-item interactions, one with user, item and review nodes, and one with
        review to words contained in the review. This method does not care about the order of words, though they could
        be encoded in the edges.
        Returns
        -------
        train_graph: dgl.DGLHeteroGraph
            Graph containing user-item interactions.
        review_graph: dgl.DGLHeteroGraph
            Graph containing user, item and review nodes.
        word_graph: dgl.DGLHeteroGraph
            Graph containing review and word nodes.
        """
        import dgl
        import torch

        # Get data
        train_set = self.train_set
        user_ids, item_ids = torch.LongTensor(train_set.csr_matrix.nonzero())
        ratings = torch.LongTensor(train_set.csr_matrix.data)
        review_ids = torch.LongTensor([train_set.review_text.user_review[u][i]
                                      for u, i in zip(*train_set.csr_matrix.nonzero())])
        num_users, num_items = train_set.num_users, train_set.num_items

        # Create graphs
        train_graph = dgl.heterograph({
            ('user', 'ui', 'item'): (user_ids, item_ids),
        }, num_nodes_dict={'user': num_users, 'item': num_items})

        # Add ratings to graph
        train_graph.edata['rating'] = ratings
        train_graph.edata['review'] = review_ids

        # Create node to review graph
        user_ids, user_review_ids = self._get_review_edges(train_set.review_text.user_review)
        item_ids, item_review_ids = self._get_review_edges(train_set.review_text.item_review)

        review_graph = dgl.heterograph({
            ('review', 'ru', 'user'): (user_review_ids, user_ids),
            ('review', 'ri', 'item'): (item_review_ids, item_ids)
        }, num_nodes_dict={'user': num_users, 'item': num_items, 'review': len(train_set.review_text.corpus)})

        # Create review to word graph
        review_ids, word_ids = torch.LongTensor([[r, w] for r, words in enumerate(train_set.review_text.sequences)
                                                for w in words]).T
        word_graph = dgl.heterograph({
            ('word', 'wr', 'review'): (word_ids, review_ids)
        }, num_nodes_dict={'review': len(train_set.review_text.corpus),
                           'word': train_set.review_text.vocab.size})

        return train_graph, review_graph, word_graph


    def _fit(self, val_set):
        import dgl
        import torch
        from torch import optim
        from .dgl_utils import R3Sampler

        # Edges where label is non-zero and user and item occurs more than once.
        g = self.train_graph
        cf_eids = {'ui': g['ui'].edges(form='eid')}
        num_workers = self.num_workers

        if self.use_uva:
            # g = g.to(self.device)
            cf_eids = {k: v.to(self.device) for k, v in cf_eids.items()}
            num_workers = 0

        if self.debug:
            num_workers = 0
            thread = False
        else:
            thread = None

        # Create sampler
        sampler = R3Sampler(self.review_graph, self.train_set.review_text.sequences)

        sampler = dgl.dataloading.as_edge_prediction_sampler(sampler, exclude='self')

        cf_dataloader = dgl.dataloading.DataLoader(g, cf_eids, sampler, batch_size=self.batch_size, shuffle=True,
                                                   drop_last=True, device=self.device, use_uva=self.use_uva,
                                                   num_workers=num_workers, use_prefetch_thread=thread)
        cf_length = len(cf_dataloader)

        # Initialize training params.
        cf_optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        if self.objective == 'ranking':
            metrics = [cornac.metrics.NDCG()]
        else:
            metrics = [cornac.metrics.MSE()]

        best_state = None
        best_score = 0 if metrics[0].higher_better else float('inf')
        best_epoch = -1
        for e in range(self.num_epochs):
            tot_losses = defaultdict(int)
            cur_losses = {}
            self.model.train()
            # CF
            with tqdm(cf_dataloader, disable=not self.verbose) as progress:
                for i, batch in enumerate(progress, 1):
                    input_nodes, g, _ = batch
                    labels = g.edata['rating'].float()
                    (p_r, p_c), z = self.model(input_nodes, g)
                    loss_r, loss_c, loss_rc, loss_reg = self.model.loss(input_nodes, p_r, p_c, labels, z)

                    loss = loss_r + self.lambda_ * loss_rc + self.alpha * loss_reg

                    cur_losses['l'] = loss.item()
                    cur_losses['l_r'] = loss_r.item()
                    cur_losses['l_c'] = loss_c.item()
                    cur_losses['l_rc'] = loss_rc.item()
                    cur_losses['l_reg'] = loss_reg.item()

                    loss.backward()
                    for k, v in cur_losses.items():
                        tot_losses[k] += v

                    cf_optimizer.step()
                    cf_optimizer.zero_grad()

                    if self.summary_writer is not None:
                        for k, v in cur_losses.items():
                            self.summary_writer.add_scalar(f'train/cf/{k}', v, e * cf_length + i)

                    loss_str = ','.join([f'{k}: {v/i:.3f}' for k, v in tot_losses.items()])
                    if i != cf_length:
                        progress.set_description(f'Epoch {e}, ' + loss_str)
                    # else:
                    #     if val_set is not None:
                    #         results = self._validate(val_set, metrics)
                    #         res_str = 'Val: ' + ', '.join([f'{m.name}: {r:.3f}' for m, r in zip(metrics, results)])
                    #         progress.set_description(f'Epoch {e}, ' + f'{loss_str}, ' + res_str)
                    #
                    #         if self.summary_writer is not None:
                    #             for m, r in zip(metrics, results):
                    #                 self.summary_writer.add_scalar(f'val/{m.name}', r, e)
                    #
                    #         if self.model_selection == 'best' and (results[0] > best_score if metrics[0].higher_better
                    #                 else results[0] < best_score):
                    #             best_state = deepcopy(self.model.state_dict())
                    #             best_score = results[0]
                    #             best_epoch = e

            if self.early_stopping is not None and e - best_epoch >= self.early_stopping:
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model.inference(g, self.batch_size)

        if val_set is not None and self.summary_writer is not None:
            results = self._validate(val_set, metrics)
            self.summary_writer.add_hparams(self.parameters, dict(zip([m.name for m in metrics], results)))

        _ = self._validate(val_set, metrics)
        self.best_epoch = best_epoch
        self.best_value = best_score

    def _validate(self, val_set, metrics):
        from ...eval_methods.base_method import rating_eval, ranking_eval
        import torch

        self.model.eval()
        with torch.no_grad():
            self.model.inference(self.train_graph, self.batch_size)
            if val_set is not None:
                if self.objective == 'ranking':
                    (result, _) = ranking_eval(self, metrics, self.train_set, val_set)
                else:
                    (result, _) = rating_eval(self, metrics, val_set, user_based=self.user_based)
        return result

    def score(self, user_idx, item_idx=None):
        import torch

        self.model.eval()
        with torch.no_grad():
            user_idx = torch.tensor(user_idx, dtype=torch.int64).to(self.device)
            if item_idx is None:
                item_idx = torch.arange(self.train_set.num_items, dtype=torch.int64).to(self.device)
                ranking = True
            else:
                item_idx = torch.tensor(item_idx, dtype=torch.int64).to(self.device)
                ranking = False

            pred = self.model.predict(user_idx, item_idx, ranking).cpu()

            return pred.numpy()

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
