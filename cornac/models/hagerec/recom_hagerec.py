import os
from copy import deepcopy
from tqdm import tqdm

from ..recommender import Recommender
from ...data import Dataset
from ...utils import create_heterogeneous_graph


class HAGERec(Recommender):
    def __init__(self, name='KGAT', use_cuda=False, use_uva=False,
                 batch_size=128,
                 num_workers=0,
                 num_epochs=10,
                 learning_rate=0.1,
                 l2_weight=0,
                 node_dim=64,
                 relation_dim=64,
                 layer_dim=64,
                 model_selection='best',
                 early_stopping=None,
                 tr_feat_dropout=.0,
                 layer_dropout=.1,
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

        super(HAGERec, self).__init__(name)
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
        self.relation_dim = relation_dim
        self.layer_dims = layer_dim
        self.n_layers = len(layer_dim)
        self.model_selection = model_selection
        self.early_stopping = early_stopping
        self.tr_feat_dropout = tr_feat_dropout
        self.layer_dropouts = layer_dropout
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
        from .hagerec import Model

        super().fit(train_set, val_set)
        self.train_graph, n_nodes, self.n_items, n_relations = create_heterogeneous_graph(train_set)

        # create model
        self.model = Model(n_nodes, n_relations, self.node_dim, self.relation_dim, self.n_layers,
                           self.layer_dropouts, self.edge_dropouts)

        # self.model.reset_parameters()

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
        u, v = g.edges()
        mask = g.edata['label'] != 0
        _, i, c = torch.unique(u, sorted=False, return_inverse=True, return_counts=True)
        mask *= c[i] > 1
        _, i, c = torch.unique(v, sorted=False, return_inverse=True, return_counts=True)
        mask *= (c[i] > 1)
        cf_eids = g.edges(form='eid')[mask]
        num_workers = self.num_workers

        # Get reverse edge mapping.
        n_edges = g.num_edges()
        reverse_eids = torch.cat([torch.arange(n_edges // 2, n_edges), torch.arange(0, n_edges // 2)])

        if self.use_uva:
            # g = g.to(self.device)
            cf_eids = cf_eids.to(self.device)
            num_workers = 0

        if self.debug:
            num_workers = 0
            thread = False
        else:
            thread = None

        # Create sampler
        sampler = dgl.dataloading.NeighborSampler([10]*self.n_layers, 'a')
        sampler = dgl.dataloading.as_edge_prediction_sampler(sampler, exclude='reverse_id', reverse_eids=reverse_eids)
        cf_dataloader = dgl.dataloading.DataLoader(g, cf_eids, sampler, batch_size=self.batch_size, shuffle=True,
                                                drop_last=True, device=self.device, use_uva=self.use_uva,
                                                num_workers=num_workers, use_prefetch_thread=thread)
        cf_length = len(cf_dataloader)

        # Initialize training params.
        tr_optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        cf_optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        best_state = None
        best_score = float('inf')
        best_epoch = -1
        for e in range(self.num_epochs):
            tot_mse = 0
            tot_l2 = 0
            tot_loss = 0
            self.model.train()
            # CF
            with tqdm(cf_dataloader, disable=not self.verbose) as progress:
                for i, (input_nodes, edge_subgraph, blocks) in enumerate(progress, 1):
                    emb = self.model.node_embedding(input_nodes)
                    x, conv_out, neighborhood, attentions = self.model(blocks, emb)

                    # Update attention
                    g.edata['a'][blocks[0].edata[dgl.EID]] = attentions.detach().to(g.edata['a'].device)

                    pred = self.model.graph_predict(edge_subgraph, x)

                    mse_loss = self.model.loss(pred, edge_subgraph.edata['label'])
                    l2_loss = self.l2_weight * self.model.l2_loss(edge_subgraph, emb)
                    loss = mse_loss + l2_loss
                    loss.backward()

                    tot_mse += mse_loss.detach()
                    tot_l2 += l2_loss.detach()
                    tot_loss += loss.detach()

                    cf_optimizer.step()
                    cf_optimizer.zero_grad()

                    progress.set_description(f'Epoch {e}, '
                                             f'MSE:{tot_mse / i:.5f}'
                                             f',L2:{tot_l2 / i:.3g}'
                                             f',Tot:{tot_loss / i:.5f}'
                                             )

                    if self.summary_writer is not None:
                        self.summary_writer.add_scalar('train/cf/mse', mse_loss, e*cf_length+i)
                        self.summary_writer.add_scalar('train/cf/l2', l2_loss, e*cf_length+i)

            # TransR
            tot_tr = 0
            tot_l2 = 0
            tot_loss = 0
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

                    tr_optimizer.step()
                    tr_optimizer.zero_grad()

                    tot_tr += tr_loss.detach()
                    tot_l2 += l2_loss.detach()
                    tot_loss += loss.detach()

                    if self.summary_writer is not None:
                        self.summary_writer.add_scalar('train/transr/mse', mse_loss, e*transr_length+i)
                        self.summary_writer.add_scalar('train/transr/l2', l2_loss, e*transr_length+i)

                    if i != transr_length:
                        progress.set_description(f'Epoch {e}, '
                                                 f'TR:{tot_tr / i:.5f}'
                                                 f',L2:{tot_l2 / i:.3g}'
                                                 f',Tot:{tot_loss / i:.5f}')

                    else:
                        self.set_attention(g, verbose=False)  # Always set attention after final batch.
                        if val_set is not None:
                            mse, rmse = self._validate(val_set)
                            progress.set_description(f'Epoch {e}, MSE: {tot_mse / i:.5f}, Val: MSE: {mse:.5f}, '
                                                     f'RMSE: {rmse:.5f}')

                            if self.summary_writer is not None:
                                self.summary_writer.add_scalar('val/mse', mse, e)
                                self.summary_writer.add_scalar('val/rmse', rmse, e)

                            if self.model_selection == 'best' and mse < best_score:
                                best_state = deepcopy(self.model.state_dict())
                                best_score = mse
                                best_epoch = e

            if self.early_stopping is not None and e - best_epoch >= self.early_stopping:
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.set_attention(g, verbose=self.verbose)

        if val_set is not None and self.summary_writer is not None:
            mse, rmse = self._validate(val_set)
            self.summary_writer.add_hparams(self.parameters, {'mse': mse, 'rmse': rmse})

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

    def save(self, save_dir=None):
        import torch

        if save_dir is None:
            return

        path = super().save(save_dir)
        name = path.rsplit('/', 1)[-1].replace('pkl', 'pt')

        state = self.model.state_dict()
        torch.save(state, os.path.join(save_dir, str(self.index), name))
