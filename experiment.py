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
import argparse
import inspect

import cornac
from cornac.datasets import amazon_digital_music, amazon_cellphone_seer, amazon_computer_seer
from cornac.eval_methods import RatioSplit, StratifiedSplit
from cornac.data import ReviewModality, SentimentModality, Reader
from cornac.data.text import BaseTokenizer

import sys


def run(in_kwargs, dataset, method, save_dir='.'):
    user_based = in_kwargs.pop('user_based', True)

    if method == 'hear':
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
            'fanout': 5,
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
        model = cornac.models.HEAR
        # Same dropout
        if 'dropout' in in_kwargs:
            in_kwargs['layer_dropout'] = in_kwargs['dropout']
            in_kwargs['attention_dropout'] = in_kwargs['dropout']
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
        model = cornac.models.KGAT
        if 'dropout' in in_kwargs:
            in_kwargs['layer_dropouts'] = in_kwargs['dropout']
            in_kwargs['edge_dropouts'] = in_kwargs['dropout']
    elif method == 'hagerec':
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
        model = cornac.models.HAGERec
    else:
        raise NotImplementedError
    parameters = list(inspect.signature(model).parameters.keys())
    in_kwargs = {k: v for k, v in in_kwargs.items() if k in parameters}  # some python args are not relevant for model
    default_kwargs.update(in_kwargs)

    if dataset == 'cellphone':
        feedback = amazon_cellphone_seer.load_feedback(fmt="UIRT", reader=Reader())
        reviews = amazon_cellphone_seer.load_review()
        sentiment = amazon_cellphone_seer.load_sentiment(reader=Reader())
    elif dataset == 'computer':
        feedback = amazon_computer_seer.load_feedback(fmt="UIRT", reader=Reader())
        reviews = amazon_computer_seer.load_review()
        sentiment = amazon_computer_seer.load_sentiment(reader=Reader())
    else:
        raise NotImplementedError

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
        verbose=default_kwargs.get('verbose', True),
    )

    model = model(**default_kwargs)

    cornac.Experiment(
        eval_method=eval_method, models=[model], metrics=[cornac.metrics.MSE(), cornac.metrics.RMSE()],
        user_based=user_based, save_dir=save_dir, verbose=default_kwargs.get('verbose', True)
    ).run()


if __name__ == '__main__':
    dataset = str(sys.argv[1])
    method = str(sys.argv[2])
    kwargs = ' '.join(sys.argv[3:])
    print(kwargs)
    kwargs = eval(kwargs)
    run(kwargs, dataset, method, f'results/{dataset}/{method}')