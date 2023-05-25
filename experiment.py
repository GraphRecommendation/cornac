# Copyright 2018 The Cornac Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
import inspect
import os
import pickle

from copy import deepcopy

import pandas as pd
import torch

from cornac.datasets import amazon_cellphone_seer, amazon_computer_seer, amazon_toy_seer, amazon_camera_seer
from cornac.eval_methods import StratifiedSplit
from cornac.data import ReviewModality, SentimentModality, Reader
from cornac.data.text import BaseTokenizer

import sys

from cornac.models import MTER


def run(in_kwargs, dataset, method, save_dir='.'):
    user_based = in_kwargs.pop('user_based', True)
    skip_tried = in_kwargs.pop('skip_tried', False)
    objective = in_kwargs['objective'] = in_kwargs.get('objective', 'ranking')  # Ranking is default

    if method in ['hear', 'testrec', 'lightrla', 'globalrla', 'globalrla-e', 'globalrla-l', 'globalrla-le']:
        default_kwargs = {
            'use_cuda': True,
            'use_uva': False,
            'batch_size': 256,
            'num_workers': 5,
            'num_epochs': 10,
            'learning_rate': 0.001,
            'weight_decay': 1e-4,
            'node_dim': 64,
            'review_dim': 32,
            'final_dim': 16,
            'num_heads': 3,
            'fanout': 10,
            'model_selection': 'best',
            'review_aggregator': 'narre',
            'predictor': 'narre',
            'layer_dropout': .5,
            'attention_dropout': .1,
            'user_based': user_based,
            'debug': False,
            'out_path': save_dir,
            'verbose': True
        }
        if method == 'hear':
            from cornac.models import HEAR
            model = HEAR
        elif method == 'lightrla':
            from cornac.models import LightRLA
            model = LightRLA
        elif method in ['globalrla', 'globalrla-e', 'globalrla-le', 'globalrla-l']:
            from cornac.models import GlobalRLA
            model = GlobalRLA
        else:
            from cornac.models import TestRec
            model = TestRec
        # Same dropout
        if 'dropout' in in_kwargs:
            in_kwargs['layer_dropout'] = in_kwargs['dropout']
            in_kwargs['attention_dropout'] = in_kwargs['dropout']

        if (path := in_kwargs.get('embedding_path')) is not None:
            with open(path, 'rb') as f:
                pm = pickle.load(f)

            em = pm.model.embeddings
            if isinstance(em, dict):
                em = torch.cat([em[k] for k in ['item', 'user', 'node'] if k in em])

            in_kwargs['learned_node_embeddings'] = em
    elif method == 'kgat':
        default_kwargs = {
            'use_cuda': True,
            'use_uva': False,
            'batch_size': 256,
            'num_workers': 5,
            'num_epochs': 10,
            'learning_rate': 0.0001,
            'l2_weight': 1e-5,
            'node_dim': 64,
            'relation_dim': 64,
            'layer_dims': [64, 32, 16],
            'model_selection': 'best',
            'layer_dropouts': .5,
            'edge_dropouts': .1,
            'user_based': user_based,
            'debug': False,
            'out_path': save_dir,
            'verbose': True
        }
        from cornac.models import KGAT
        model = KGAT
        if 'dropout' in in_kwargs:
            in_kwargs['layer_dropouts'] = in_kwargs['dropout']
            in_kwargs['edge_dropouts'] = in_kwargs['dropout']
    elif method in ['hagerec']:
        default_kwargs = {
            'use_cuda': True,
            'use_uva': False,
            'batch_size': 256,
            'num_workers': 5,
            'num_epochs': 50,
            'learning_rate': 0.0001,
            'l2_weight': 1e-5,
            'node_dim': 64,
            'relation_dim': 64,
            'layer_dim': 64,
            'model_selection': 'best',
            'layer_dropout': .1,
            'edge_dropout': .1,
            'user_based': user_based,
            'debug': False,
            'out_path': save_dir,
            'verbose': True
        }
        from cornac.models import HAGERec
        model = HAGERec
        if 'dropout' in in_kwargs:
            in_kwargs['layer_dropout'] = in_kwargs['dropout']
            in_kwargs['edge_dropout'] = in_kwargs['dropout']
    elif method == 'trirank':
        default_kwargs = {}
        from cornac.models import TriRank
        model = TriRank
    elif method == 'most-pop':
        default_kwargs = {}
        from cornac.models import MostPop
        model = MostPop
    elif method in ['narre', 'hrdr']:
        default_kwargs = {
            'embedding_size': 100,
            'id_embedding_size': 32,
            'n_factors': 32,
            'attention_size': 16,
            'kernel_sizes': [3],
            'n_filters': 64,
            'dropout_rate': 0.1,
            'max_text_length': 50,
            'batch_size': 64,
            'max_iter': 500,
            'model_selection': 'best',
            'learning_rate': 0.0001,
            'seed': 42,
            'max_num_review': 50
        }
        from cornac.models import NARRE, NARRE_BPR, HRDR, HRDR_BPR
        if in_kwargs.get('use_bpr', False):
            model = NARRE_BPR if method == 'narre' else HRDR_BPR
        else:
            model = NARRE if method == 'narre' else HRDR
    elif method == 'bpr':
        from cornac.models import BPR
        default_kwargs = {
            'k': 16,
            'max_iter': 100,
            'learning_rate': 0.001,
            'lambda_reg': 0.01,
            'use_bias': True
        }
        model = BPR
    elif method in ['ngcf', 'lightgcn']:
        default_kwargs = {
            'use_cuda': True,
            'use_uva': False,
            'batch_size': 256,
            'num_workers': 5,
            'num_epochs': 10,
            'learning_rate': 0.0001,
            'l2_weight': 1e-5,
            'node_dim': 64,
            'layer_dims': [64, 32, 16],
            'model_selection': 'best',
            'layer_dropout': .1,
            'user_based': user_based,
            'debug': False,
            'out_path': save_dir,
            'verbose': True,
            'lightgcn': True if method == 'lightgcn' else False,
            'name': method
        }
        if method == 'lightgcn':
            default_kwargs['layer_dims'] = [64, 64, 64]

        from cornac.models import NGCF
        model = NGCF
    elif method == 'mter':
        dim = in_kwargs.get('dim', 30)
        default_kwargs = {
            'n_user_factors':dim,
            'n_item_factors':dim,
            'n_aspect_factors':12,
            'n_opinion_factors':12,
            'n_bpr_samples':256,
            'n_element_samples':256,
            'lambda_reg':0.1,
            'lambda_bpr':10,
            'max_iter':200000,
            'lr':0.001,
            'early_stopping': 5000,
            'eval_interval': 5
        }
        model = MTER
        print('working 1')
    else:
        raise NotImplementedError

    parameters = list(inspect.signature(model).parameters.keys())
    default_kwargs.update(in_kwargs)
    default_kwargs = {k: v for k, v in default_kwargs.items() if k in parameters}  # some python args are not relevant for model

    if dataset == 'cellphone':
        feedback = amazon_cellphone_seer.load_feedback(fmt="UIRT", reader=Reader())
        reviews = amazon_cellphone_seer.load_review()
        sentiment = amazon_cellphone_seer.load_sentiment(reader=Reader())
    elif dataset == 'computer':
        feedback = amazon_computer_seer.load_feedback(fmt="UIRT", reader=Reader())
        reviews = amazon_computer_seer.load_review()
        sentiment = amazon_computer_seer.load_sentiment(reader=Reader())
    elif dataset == 'toy':
        feedback = amazon_toy_seer.load_feedback(fmt="UIRT", reader=Reader())
        reviews = amazon_toy_seer.load_review()
        sentiment = amazon_toy_seer.load_sentiment(reader=Reader())
    elif dataset == 'camera':
        feedback = amazon_camera_seer.load_feedback(fmt="UIRT", reader=Reader())
        reviews = amazon_camera_seer.load_review()
        sentiment = amazon_camera_seer.load_sentiment(reader=Reader())
    else:
        raise NotImplementedError(dataset, 'not valid.')

    sentiment_modality = SentimentModality(data=sentiment)

    review_modality = ReviewModality(
        data=reviews,
        tokenizer=BaseTokenizer(stop_words="english"),
        max_vocab=4000,
        max_doc_freq=0.5,
    )

    eval_method = StratifiedSplit(
        feedback,
        group_by="user",
        chrono=True,
        sentiment=sentiment_modality,
        review_text=review_modality,
        test_size=0.2,
        val_size=0.16,
        exclude_unknowns=True,
        seed=42,
        verbose=default_kwargs.get('verbose', True),
    )

    model = model(**default_kwargs)

    import cornac
    from cornac.metrics import NDCG, AUC, MAP, MRR, Recall, Precision
    if objective == 'ranking':
        metrics = [NDCG(), NDCG(20), NDCG(100), AUC(), MAP(), MRR(), Recall(10), Recall(20), Precision(10), Precision(20)]
    elif objective == 'rating':
        metrics = [cornac.metrics.MSE(), cornac.metrics.RMSE()]
    else:
        raise ValueError(f'No metrics for objective: {objective}.')

    dk = deepcopy(default_kwargs)
    dk['id'] = in_kwargs.get('index', 0)

    if skip_tried:
        path = os.path.join(save_dir, model.name, 'results.csv')
        if os.path.isfile(path):
            df = pd.read_csv(path)
            columns = [c for c in df.columns if c in in_kwargs]
            values = [in_kwargs.get(c) for c in columns]
            df = df[columns]  # ensure ordering and ignore e.g. score
            if (df == values).all(1).any():
                print(f'Skipped {method} for {dataset}')
                return 0

    cornac.Experiment(
        eval_method=eval_method, models=[model], metrics=metrics,
        user_based=user_based, save_dir=save_dir, verbose=default_kwargs.get('verbose', True)
    ).run(store_experiment=True, parameters=dk)


if __name__ == '__main__':
    dataset = str(sys.argv[1])
    method = str(sys.argv[2])
    kwargs = ' '.join(sys.argv[3:])
    print(kwargs)
    kwargs = eval(kwargs)
    run(kwargs, dataset, method, f'results/{dataset}/{kwargs.get("name", method)}')