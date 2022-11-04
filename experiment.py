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
from cornac.datasets import amazon_digital_music, amazon_cellphone_seer
from cornac.eval_methods import RatioSplit, StratifiedSplit
from cornac.data import ReviewModality, SentimentModality, Reader
from cornac.data.text import BaseTokenizer

import sys


def run(kwargs):
    user_based = kwargs.pop('user_based', True)
    feedback = amazon_cellphone_seer.load_feedback(fmt="UIRT", reader=Reader())
    reviews = amazon_cellphone_seer.load_review()
    sentiment = amazon_cellphone_seer.load_sentiment(reader=Reader())

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
        verbose=True,
    )

    default_kwargs = {
        'use_cuda': True,
        'use_uva': False,
        'batch_size': 64,
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
        'layer_dropout': [.5, .5],
        'attention_dropout': .1,
        'user_based': user_based,
        'debug': False
    }

    parameters = list(inspect.signature(cornac.models.HEAR).parameters.keys())

    # Same dropout
    if 'dropout' in kwargs:
        kwargs['layer_dropout'] = kwargs['dropout']
        kwargs['attention_dropout'] = kwargs['dropout']

    kwargs = {k: v for k, v in kwargs.items() if k in parameters}  # some python args are not relevant for model
    default_kwargs.update(kwargs)

    # Assume both layers to have equal dropout rate.
    if isinstance(default_kwargs['layer_dropout'], float):
        default_kwargs.update({'layer_dropout': [default_kwargs['layer_dropout']]*2})

    model = cornac.models.HEAR(**default_kwargs)

    cornac.Experiment(
        eval_method=eval_method, models=[model], metrics=[cornac.metrics.MSE(), cornac.metrics.RMSE()],
        user_based=user_based
    ).run()


if __name__ == '__main__':
    kwargs = '{' + ','.join(sys.argv[1:]) + '}'
    kwargs = eval(kwargs)
    run(kwargs)