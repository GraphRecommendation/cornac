import os
from collections import defaultdict
from copy import deepcopy
from tqdm import tqdm

from ..recommender import Recommender
from ...data import Dataset
from ...utils import create_heterogeneous_graph


class KGAT(Recommender):
    def __init__(self, name='KGAT', use_cuda=False, use_uva=False,
                 batch_size=128,
                 num_workers=0,
                 num_epochs=10,
                 learning_rate=0.1,
                 l2_weight=0,
                 node_dim=64,
                 relation_dim=64,
                 layer_dims=None,
                 model_selection='best',
                 objective='ranking',
                 early_stopping=None,
                 tr_feat_dropout=.0,
                 layer_dropouts=.1,
                 edge_dropouts=.0,
                 normalize=False,
                 user_based=True,
                 debug=False,
                 verbose=True,
                 index=0,
                 use_tensorboard=True,
                 out_path=None,
                 ):

        from torch.utils.tensorboard import SummaryWriter

        super().__init__(name)
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
        self.relation_dim = relation_dim
        self.layer_dims = layer_dims
        self.n_layers = len(layer_dims)
        self.model_selection = model_selection
        self.objective = objective
        self.early_stopping = early_stopping
        self.tr_feat_dropout = tr_feat_dropout
        self.layer_dropouts = layer_dropouts
        self.edge_dropouts = edge_dropouts
        self.normalize = normalize
        parameter_list = ['batch_size', 'learning_rate', 'l2_weight', 'node_dim', 'relation_dim', 'layer_dims',
                          'tr_feat_dropout', 'layer_dropouts', 'model_selection', 'edge_dropouts', 'normalize']
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
        self.n_items = 0

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

    def set_attention(self, g, verbose=True):
        import torch

        with torch.no_grad():
            g.edata['a'] = self.model.compute_attention(g, self.batch_size, verbose=verbose).pin_memory()

    def fit(self, train_set: Dataset, val_set=None):
        from .kgat import Model

        super().fit(train_set, val_set)
        self.train_graph, n_nodes, self.n_items, n_relations = create_heterogeneous_graph(train_set)

        # create model
        self.model = Model(n_nodes, n_relations, self.node_dim, self.relation_dim, self.n_layers,
                           self.layer_dims, self.tr_feat_dropout, [self.layer_dropouts] * self.n_layers,
                           [self.edge_dropouts] * self.n_layers)

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
        from . import dgl_utils
        import cornac

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
        tr_dataloader = dgl.dataloading.DataLoader(g, tr_edges, sampler, batch_size=self.batch_size*3, shuffle=True,
                                                      drop_last=True, device=self.device, use_uva=self.use_uva,
                                                      num_workers=num_workers, use_prefetch_thread=thread)
        transr_length = len(tr_dataloader)

        # CF sampler
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(self.n_layers)
        if self.objective == 'ranking':
            neg_sampler = cornac.utils.dgl.UniformItemSampler(1, self.train_set.num_items)
        else:
            neg_sampler = None
        sampler = dgl.dataloading.as_edge_prediction_sampler(sampler, exclude='reverse_id', reverse_eids=reverse_eids,  negative_sampler=neg_sampler)
        cf_dataloader = dgl.dataloading.DataLoader(g, cf_eids, sampler, batch_size=self.batch_size, shuffle=True,
                                                drop_last=True, device=self.device, use_uva=self.use_uva,
                                                num_workers=num_workers, use_prefetch_thread=thread)
        cf_length = len(cf_dataloader)

        # Initialize training params.
        tr_optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
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
                    emb = self.model.node_embedding(input_nodes)
                    x = self.model(blocks, emb)

                    pred = self.model.graph_predict(edge_subgraph, x)
                    if self.objective == 'ranking':
                        pred_j = self.model.graph_predict(neg_subgraph, x)
                        acc = (pred > pred_j).sum() / pred_j.shape.numel()
                        loss = self.model.ranking_loss(pred, pred_j)
                        l2_loss = self.model.l2_loss(edge_subgraph, emb, neg_graph=neg_subgraph)
                        cur_losses['loss'] = loss.detach()
                        cur_losses['acc'] = acc.detach()
                    else:
                        loss = self.model.rating_loss(pred, edge_subgraph.edata['label'])
                        l2_loss = self.model.l2_loss(edge_subgraph, emb)
                        cur_losses['loss'] = loss.detach()

                    l2_loss = self.l2_weight * l2_loss
                    cur_losses['l2'] = l2_loss.detach()

                    loss = loss + l2_loss
                    loss.backward()

                    for k, v in cur_losses.items():
                        tot_losses[k] += v

                    cf_optimizer.step()
                    cf_optimizer.zero_grad()

                    loss_str = ','.join([f'{k}:{v/i:.3f}' for k, v in tot_losses.items()])
                    progress.set_description(f'Epoch {e}, ' + loss_str)

                    if self.summary_writer is not None:
                        for k, v in cur_losses.items():
                            self.summary_writer.add_scalar(f'train/cf/{k}', v, e * cf_length + i)

            # TransR
            tot_losses = defaultdict(int)
            cur_losses = {}
            with tqdm(tr_dataloader, disable=not self.verbose) as progress:
                for i, (input_nodes, pos_subgraph, neg_subgraph, blocks) in enumerate(progress, 1):
                    x = self.model.node_embedding(input_nodes)
                    neg_subgraph.edata['type'] = pos_subgraph.edata['type']

                    pos_x = self.model.trans_r(pos_subgraph, x)
                    neg_x = self.model.trans_r(neg_subgraph, x)
                    tr_loss = self.model.trans_r_loss(pos_x, neg_x)
                    l2_loss = self.l2_weight * self.model.l2_loss(pos_subgraph, x, trans_r=True, neg_graph=neg_subgraph)
                    loss = tr_loss + l2_loss
                    loss.backward()

                    cur_losses['loss'] = tr_loss.detach()
                    cur_losses['acc'] = ((pos_x < neg_x).sum() / neg_x.shape.numel()).detach()
                    cur_losses['l2'] = l2_loss.detach()

                    tr_optimizer.step()
                    tr_optimizer.zero_grad()

                    for k, v in cur_losses.items():
                        tot_losses[k] += v

                    if self.summary_writer is not None:
                        for k, v in cur_losses.items():
                            self.summary_writer.add_scalar(f'train/tr/{k}', v, e * transr_length + i)

                    loss_str = ','.join([f'{k}:{v/i:.3f}' for k, v in tot_losses.items()])
                    if i != transr_length:
                        progress.set_description(f'Epoch {e}, ' + loss_str)
                    else:
                        self.set_attention(g, verbose=False)  # Always set attention after final batch.
                        if val_set is not None:
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

            if self.early_stopping is not None and e - best_epoch >= self.early_stopping:
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.set_attention(g, verbose=self.verbose)

        if val_set is not None and self.summary_writer is not None:
            results = self._validate(val_set, metrics)
            self.summary_writer.add_hparams(self.parameters, dict(zip([m.name for m in metrics], results)))

        _ = self._validate(val_set, metrics)

    def _validate(self, val_set, metrics):
        from ...eval_methods.base_method import rating_eval, ranking_eval
        import torch

        self.model.eval()
        with torch.no_grad():
            x = self.model.node_embedding(self.train_graph.nodes().to(self.device))
            self.model.inference(self.train_graph, x, self.device)
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

        if save_dir is None:
            return

        path = super().save(save_dir)
        name = path.rsplit('/', 1)[-1].replace('pkl', 'pt')

        state = self.model.state_dict()
        torch.save(state, os.path.join(save_dir, str(self.index), name))

        return path
