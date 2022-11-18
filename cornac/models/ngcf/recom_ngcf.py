import os
from collections import defaultdict
from copy import deepcopy

from tqdm import tqdm

import cornac.metrics
from ..recommender import Recommender
from ...data import Dataset


class NGCF(Recommender):
    def __init__(self, name='NGCF', use_cuda=False, use_uva=False,
                 batch_size=128,
                 num_workers=0,
                 num_epochs=10,
                 learning_rate=0.1,
                 l2_weight=0,
                 node_dim=64,
                 layer_dim=None,
                 lightgcn=False,
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

        super(NGCF, self).__init__(name)
        # Default values
        if layer_dim is None:
            layer_dim = [64, 32, 16]

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
        self.layer_dims = layer_dim
        self.lightgcn = lightgcn
        self.model_selection = model_selection
        self.objective = objective
        self.early_stopping = early_stopping
        self.layer_dropout = layer_dropout
        parameter_list = ['batch_size', 'learning_rate', 'l2_weight', 'node_dim', 'layer_dims',
                          'layer_dropout', 'model_selection', 'edge_dropout']
        self.parameters = {}
        for k in parameter_list:
            attr = self.__getattribute__(k)
            if isinstance(attr, list):
                for i in range(len(attr)):
                    self.parameters[f'{k}_{i}'] = attr[i]
            else:
                self.parameters[k] = attr

        # Method
        self.train_graph = None
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
        assert objective == 'ranking' or objective == 'rating', f'This method only supports ranking or rating, ' \

    def fit(self, train_set: Dataset, val_set=None):
        from .ngcf import Model
        import torch, dgl

        super().fit(train_set, val_set)
        u, v = train_set.matrix.nonzero()
        u, v = torch.LongTensor(u), torch.LongTensor(v)

        data = {('user', 'ui', 'item'): (u, v), ('item', 'iu', 'user'): (v, u)}
        g = dgl.heterograph(data)
        self.train_graph = g

        # create model
        self.model = Model(g, self.node_dim, self.layer_dims, self.layer_dropout,
                           self.lightgcn, self.use_cuda)
        self.model.reset_parameters()

        if self.verbose:
            print(f'Number of trainable parameters: {sum(p.numel() for p in self.model.parameters())}')

        if self.use_cuda:
            self.model = self.model.cuda()

        if self.trainable:
            self._fit(val_set)

        if self.summary_writer is not None:
            self.summary_writer.close()

        return self

    def _fit(self, val_set):
        import dgl
        import torch
        from torch import optim

        # Edges where label is non-zero and user and item occurs more than once.
        g = self.train_graph
        cf_eids = {'ui': g['ui'].edges()}
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
        reverse_etypes = {'n_i': 'n_u', 'n_u': 'n_i'}
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(len(self.layer_dims))
        if self.objective == 'ranking':
            neg_sampler = cornac.utils.dgl.UniformItemSampler(1, self.train_set.num_items)
        else:
            neg_sampler = None

        sampler = dgl.dataloading.as_edge_prediction_sampler(sampler, exclude='reverse_types',
                                                             negative_sampler=neg_sampler,
                                                             reverse_etypes=reverse_etypes)

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
                    if self.objective == 'ranking':
                        input_nodes, edge_subgraph, neg_subgraph, blocks = batch
                    else:
                        input_nodes, edge_subgraph, blocks = batch

                    emb = {ntype: self.model.node_embedding(input_nodes[ntype]) for ntype in input_nodes.keys()}
                    x = self.model(blocks, emb)

                    pred = self.model.graph_predict(edge_subgraph, x)

                    if self.objective == 'ranking':
                        pred_j = self.model.graph_predict(neg_subgraph, x)
                        acc = (pred > pred_j).sum() / pred_j.shape.numel()
                        loss = self.model.ranking_loss(pred, pred_j)
                        cur_losses['loss'] = loss.detach()
                        cur_losses['acc'] = acc.detach()
                    else:
                        loss = self.model.rating_loss(pred, edge_subgraph['n_i'].edata['label'])
                        cur_losses['loss'] = loss.detach()

                    loss = loss
                    loss.backward()
                    for k, v in cur_losses.items():
                        tot_losses[k] += v

                    cf_optimizer.step()
                    cf_optimizer.zero_grad()

                    if self.summary_writer is not None:
                        for k, v in cur_losses.items():
                            self.summary_writer.add_scalar(f'train/cf/{k}', v, e * cf_length + i)

                    loss_str = ','.join([f'{k}: {v/i:.5f}' for k, v in tot_losses.items()])
                    if i != cf_length:
                        progress.set_description(f'Epoch {e}, ' + loss_str)
                    else:
                        if val_set is not None:
                            results = self._validate(val_set, metrics)
                            res_str = 'Val: ' + ', '.join([f'{m.name}: {r:.3f}' for m, r in zip(metrics, results)])
                            progress.set_description(f'Epoch {e}, ' + f'{loss_str}, ' + res_str)

                            if self.summary_writer is not None:
                                for m, r in zip(metrics, results):
                                    self.summary_writer.add_scalar(f'val/{m.name}', r, e)

                            if self.model_selection == 'best' and (results[0] > best_score if metrics[0].higher_better
                                    else results[0] < best_score):
                                best_state = deepcopy(self.model.state_dict())
                                best_score = results[0]
                                best_epoch = e

            if self.early_stopping is not None and e - best_epoch >= self.early_stopping:
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model.inference(g, self.batch_size)

        if val_set is not None and self.summary_writer is not None:
            results = self._validate(val_set, metrics)
            self.summary_writer.add_hparams(self.parameters, dict(zip([m.name for m in metrics], results)))

        _ = self._validate(val_set, metrics)

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

        if save_dir is None:
            return

        path = super().save(save_dir)
        name = path.rsplit('/', 1)[-1].replace('pkl', 'pt')

        state = self.model.state_dict()
        torch.save(state, os.path.join(save_dir, str(self.index), name))
