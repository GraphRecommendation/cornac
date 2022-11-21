import itertools
import os
from collections import defaultdict, Counter
from copy import deepcopy

import torch
from tqdm import tqdm

import cornac.metrics
from ..recommender import Recommender
from ...data import Dataset


class LightRLA(Recommender):
    def __init__(self, name='LightRLA', use_cuda=False, use_uva=False,
                 batch_size=128,
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

        super(LightRLA, self).__init__(name)
        # Default values
        if layer_dims is None:
            layer_dims = [64, 64, 64]

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
        self.train_graph = None
        self.model = None
        self.reverse_etypes = None

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

    def _construct_graph(self, norm='both'):
        import dgl
        user_ntype = 'user'
        item_ntype = 'item'
        aspect_ntype = 'aspect'
        opinion_ntype = 'opinion'

        # Get user-item edges while changing user indices.
        users = torch.arange(self.train_set.matrix.shape[0], dtype=torch.int64)
        items = torch.arange(self.train_set.matrix.shape[1], dtype=torch.int64)
        u, i = self.train_set.matrix.nonzero()
        ui = u, i
        iu = i, u

        # get aspect and opinion
        aspect_c = Counter([a for aos in self.train_set.sentiment.sentiment.values() for a, _, _ in aos])
        opinion_c = Counter([o for aos in self.train_set.sentiment.sentiment.values() for _, o, _ in aos])
        ao_c = Counter([(a,o) for aos in self.train_set.sentiment.sentiment.values() for a, o, _ in aos])

        def get_set(counter, percentage):
            s = sum(counter.values())
            c_s = sorted(counter, key=counter.get, reverse=True)
            acc = itertools.accumulate(counter[k] for k in c_s)
            c_a = {k: a for k, a in zip(c_s, acc)}
            return set(filter(lambda x: c_a[x] <= s * percentage, c_a))

        aspects = get_set(aspect_c, .5)
        opinions = get_set(opinion_c, 1.)
        ao_set = get_set(ao_c, .01)

        print(f'Using {len(aspects)} aspects, {len(opinions)} opinions and {len(ao_set)} pairs.')

        print({n: sum(d[v] for v in s) for n, d, s in [('a', aspect_c, aspects), ('o', opinion_c, opinions),
                                                       ('ao', ao_c, ao_set)]})

        amap = {v: k for k, v in self.train_set.sentiment.aspect_id_map.items()}
        omap = {v: k for k, v in self.train_set.sentiment.opinion_id_map.items()}

        print(f'aspects: {[(k, amap[k]) for k in aspects][:20]}')
        print(f'opinions: {[(k, omap[k]) for k in opinions][:20]}')
        print((f'sets: {[(amap[a], omap[o]) for a, o in ao_set][:20]}'))

        ou = [[], []]
        oi = [[], []]
        au = [[], []]
        ai = [[], []]

        for uid, isid in self.train_set.sentiment.user_sentiment.items():
            for iid, sid in isid.items():
                aos = self.train_set.sentiment.sentiment[sid]
                for aid, oid, _ in aos:
                    # if aid in aspects and oid in opinions:
                    if (aid, oid) in ao_set:
                        ou[0].append(oid)
                        ou[1].append(uid)
                        oi[0].append(oid)
                        oi[1].append(iid)
                        au[0].append(aid)
                        au[1].append(uid)
                        ai[0].append(aid)
                        ai[1].append(iid)

        print(f'Found {len(ou[0]) + len(oi[0])} opinion edges and {len(au[0]) + len(ai[0])} aspect edges.')

        graph_data = {
            (user_ntype, 'ui', item_ntype): ui,
            (user_ntype, 'ua', aspect_ntype): (au[1], au[0]),
            # (user_ntype, 'uo', opinion_ntype): (ou[1], ou[0]),
            (item_ntype, 'iu', user_ntype): iu,
            (item_ntype, 'ia', aspect_ntype): (ai[1], ai[0]),
            # (item_ntype, 'io', opinion_ntype): (oi[1], oi[0]),
            (aspect_ntype, 'au', user_ntype): (au[0], au[1]),
            (aspect_ntype, 'ai', item_ntype): (ai[0], ai[1]),
            # (opinion_ntype, 'ou', user_ntype): (oi[0], oi[1]),
            # (opinion_ntype, 'oi', item_ntype): (oi[0], oi[1]),
        }

        new_g = dgl.heterograph(graph_data, num_nodes_dict={user_ntype: self.train_set.num_users,
                                                            item_ntype: self.train_set.num_items,
                                                            aspect_ntype: self.train_set.sentiment.num_aspects,
                                                            # opinion_ntype: self.train_set.sentiment.num_opinions
                                })

        for ntype in new_g.ntypes:
            new_g.nodes[ntype].data.update({'degrees': torch.zeros((new_g.num_nodes(ntype),))})

        if norm == 'both':
            for _, etype, ntype in new_g.canonical_etypes:
                # get degrees
                dst_degrees = new_g.in_degrees(new_g.nodes(ntype), etype=etype).float()
                new_g.nodes[ntype].data.update({'degrees': new_g.nodes[ntype].data['degrees'] + dst_degrees})

            for stype, etype, dtype in new_g.canonical_etypes:
                u, v = new_g.edges(etype=etype)
                src_degrees = new_g.nodes[stype].data['degrees'][u]
                dst_degrees = new_g.nodes[dtype].data['degrees'][v]

                # calculate norm similar to eq. 3 of both ngcf and lgcn papers.
                norm = torch.pow(src_degrees * dst_degrees, -0.5).unsqueeze(1)  # compute norm
                new_g.edges[etype].data['norm'] = norm
        reverse_etypes = {'ui': 'iu',
                          # 'ou': 'uo',
                          # 'oi': 'io',
                          'au': 'ua',
                          'ai': 'ia'
                          }
        reverse_etypes.update({v: k for k, v in reverse_etypes.items()})
        return new_g, reverse_etypes

    def _construct_graph2(self, norm='both'):
        import dgl
        user_ntype = 'user'
        item_ntype = 'item'
        aop_ntype = 'ao_pair'

        # Get user-item edges while changing user indices.
        u, i = self.train_set.matrix.nonzero()
        ui = u, i
        iu = i, u

        # get aspect and opinion
        ao_c = Counter([(a,o) for aos in self.train_set.sentiment.sentiment.values() for a, o, _ in aos])
        ao_map = {ao: i for i, ao in enumerate(ao_c)}

        def get_set(counter, percentage):
            s = sum(counter.values())
            c_s = sorted(counter, key=counter.get, reverse=True)
            acc = itertools.accumulate(counter[k] for k in c_s)
            c_a = {k: a for k, a in zip(c_s, acc)}
            return set(filter(lambda x: c_a[x] <= s * percentage, c_a))

        ao_set = get_set(ao_c, .2)

        print(f'Using {len(ao_set)} pairs.')

        print({n: sum(d[v] for v in s) for n, d, s in [('ao', ao_c, ao_set)]})

        amap = {v: k for k, v in self.train_set.sentiment.aspect_id_map.items()}
        omap = {v: k for k, v in self.train_set.sentiment.opinion_id_map.items()}

        print((f'sets: {[(amap[a], omap[o]) for a, o in ao_set][:20]}'))

        aou = [[], []]
        aoi = [[], []]

        for uid, isid in self.train_set.sentiment.user_sentiment.items():
            for iid, sid in isid.items():
                aos = self.train_set.sentiment.sentiment[sid]
                for aid, oid, _ in aos:
                    ao = (aid, oid)
                    # if aid in aspects and oid in opinions:
                    if ao in ao_set:
                        ao = ao_map[ao]
                        aou[0].append(ao)
                        aou[1].append(uid)
                        aoi[0].append(ao)
                        aoi[1].append(iid)

        graph_data = {
            (user_ntype, 'ui', item_ntype): ui,
            (user_ntype, 'ua', aop_ntype): (aou[1], aou[0]),
            (item_ntype, 'iu', user_ntype): iu,
            (item_ntype, 'ia', aop_ntype): (aoi[1], aoi[0]),
            (aop_ntype, 'au', user_ntype): (aou[0], aou[1]),
            (aop_ntype, 'ai', item_ntype): (aoi[0], aoi[1]),
        }

        new_g = dgl.heterograph(graph_data, num_nodes_dict={user_ntype: self.train_set.num_users,
                                                            item_ntype: self.train_set.num_items,
                                                            aop_ntype: len(ao_map),
                                                            # opinion_ntype: self.train_set.sentiment.num_opinions
                                })

        for ntype in new_g.ntypes:
            new_g.nodes[ntype].data.update({'degrees': torch.zeros((new_g.num_nodes(ntype),))})

        if norm == 'both':
            for _, etype, ntype in new_g.canonical_etypes:
                # get degrees
                dst_degrees = new_g.in_degrees(new_g.nodes(ntype), etype=etype).float()
                new_g.nodes[ntype].data.update({'degrees': new_g.nodes[ntype].data['degrees'] + dst_degrees})

            for stype, etype, dtype in new_g.canonical_etypes:
                u, v = new_g.edges(etype=etype)
                src_degrees = new_g.nodes[stype].data['degrees'][u]
                dst_degrees = new_g.nodes[dtype].data['degrees'][v]

                # calculate norm similar to eq. 3 of both ngcf and lgcn papers.
                norm = torch.pow(src_degrees * dst_degrees, -0.5).unsqueeze(1)  # compute norm
                new_g.edges[etype].data['norm'] = norm
        reverse_etypes = {'ui': 'iu',
                          'au': 'ua',
                          'ai': 'ia'
                          }
        reverse_etypes.update({v: k for k, v in reverse_etypes.items()})
        return new_g, reverse_etypes

    def fit(self, train_set: Dataset, val_set=None):
        from .lightrla import Model
        super().fit(train_set, val_set)

        self.train_graph, self.reverse_etypes = self._construct_graph2()

        # create model
        self.model = Model(self.train_graph, self.node_dim, self.layer_dims, self.layer_dropout, self.use_cuda)
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
        from torch import optim

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
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(len(self.layer_dims))
        if self.objective == 'ranking':
            neg_sampler = cornac.utils.dgl.UniformItemSampler(1, self.train_set.num_items)
        else:
            neg_sampler = None

        sampler = dgl.dataloading.as_edge_prediction_sampler(sampler, exclude='reverse_types',
                                                             negative_sampler=neg_sampler,
                                                             reverse_etypes=self.reverse_etypes)

        cf_dataloader = dgl.dataloading.DataLoader(g, cf_eids, sampler, batch_size=self.batch_size, shuffle=True,
                                                   drop_last=True, device=self.device, use_uva=self.use_uva,
                                                   num_workers=num_workers, use_prefetch_thread=thread)
        cf_length = len(cf_dataloader)

        # Initialize training params.
        cf_optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        if self.objective == 'ranking':
            metrics = [cornac.metrics.NDCG(), cornac.metrics.AUC()]
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

                    x = self.model(input_nodes, blocks)

                    pred = self.model.graph_predict(edge_subgraph, x)

                    if self.objective == 'ranking':
                        pred_j = self.model.graph_predict(neg_subgraph, x)
                        acc = (pred > pred_j).sum() / pred_j.shape.numel()
                        loss = self.model.ranking_loss(pred, pred_j)
                        l2 = self.l2_weight * self.model.l2_loss(edge_subgraph, neg_subgraph, x)
                        cur_losses['loss'] = loss.detach()
                        cur_losses['l2'] = l2.detach()
                        cur_losses['acc'] = acc.detach()
                        loss += l2
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

                    loss_str = ','.join([f'{k}: {v/i:.3f}' for k, v in tot_losses.items()])
                    if i != cf_length:
                        progress.set_description(f'Epoch {e}, ' + loss_str)
                    else:
                        if val_set is not None:
                            results = self._validate(val_set, metrics)
                            res_str = 'Val: ' + ', '.join([f'{m.name}: {r:.4f}' for m, r in zip(metrics, results)])
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
