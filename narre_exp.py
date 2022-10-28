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

import cornac
from cornac.datasets import amazon_digital_music, amazon_cellphone_seer
from cornac.eval_methods import RatioSplit, StratifiedSplit
from cornac.data import ReviewModality, SentimentModality, Reader
from cornac.data.text import BaseTokenizer

def run():
    user_based = True
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

    pretrained_word_embeddings = {}  # You can load pretrained word embedding here

    model = cornac.models.NARRE(
        embedding_size=100,
        id_embedding_size=32,
        n_factors=32,
        attention_size=16,
        kernel_sizes=[3],
        n_filters=64,
        dropout_rate=0.5,
        max_text_length=50,
        batch_size=64,
        max_iter=10,
        init_params={'pretrained_word_embeddings': pretrained_word_embeddings},
        verbose=True,
        seed=123,
        model_selection='best'
    )

    cornac.Experiment(
        eval_method=eval_method, models=[model], metrics=[cornac.metrics.MSE(), cornac.metrics.RMSE()],
        user_based=user_based
    ).run()


if __name__ == '__main__':
    run()