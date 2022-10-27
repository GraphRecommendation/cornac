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

    model = cornac.models.HEAR(use_cuda=True, use_uva=False, num_workers=0, review_aggregator='gatv2',
                               debug=True, num_epochs=1)

    cornac.Experiment(
        eval_method=eval_method, models=[model], metrics=[cornac.metrics.MSE(), cornac.metrics.RMSE()]
    ).run()


if __name__ == '__main__':
    run()