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

import os
import numpy as np
from cornac.metrics import NDCG
from tqdm.auto import trange

from ..recommender import Recommender
from ...exception import ScoreException

class HRDR_BPR(Recommender):
    def __init__(
        self,
        name="HRDR",
        embedding_size=100,
        id_embedding_size=32,
        n_factors=32,
        attention_size=16,
        kernel_sizes=[3],
        n_filters=64,
        dropout_rate=0.5,
        max_text_length=50,
        max_num_review=None,
        batch_size=64,
        max_iter=10,
        optimizer='adam',
        learning_rate=0.001,
        model_selection='last', # last or best
        user_based=True,
        trainable=True,
        verbose=True,
        init_params=None,
        seed=None,
    ):
        super().__init__(name=name, trainable=trainable, verbose=verbose)
        self.seed = seed
        self.embedding_size = embedding_size
        self.id_embedding_size = id_embedding_size
        self.n_factors = n_factors
        self.attention_size = attention_size
        self.n_filters = n_filters
        self.kernel_sizes = kernel_sizes
        self.dropout_rate = dropout_rate
        self.max_text_length = max_text_length
        self.max_num_review = max_num_review
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.model_selection = model_selection
        self.user_based = user_based
        # Init params if provided
        self.init_params = {} if init_params is None else init_params
        self.losses = {"train_losses": [], "val_losses": []}

    def fit(self, train_set, val_set=None):
        """Fit the model to observations.

        Parameters
        ----------
        train_set: :obj:`cornac.data.Dataset`, required
            User-Item preference data as well as additional modalities.

        val_set: :obj:`cornac.data.Dataset`, optional, default: None
            User-Item preference data for model selection purposes (e.g., early stopping).

        Returns
        -------
        self : object
        """
        Recommender.fit(self, train_set, val_set)

        if self.trainable:
            if not hasattr(self, "model"):
                from .hrdr_bpr import Model
                self.model = Model(
                    self.train_set.num_users,
                    self.train_set.num_items,
                    self.train_set.review_text.vocab,
                    self.train_set.global_mean,
                    n_factors=self.n_factors,
                    embedding_size=self.embedding_size,
                    id_embedding_size=self.id_embedding_size,
                    attention_size=self.attention_size,
                    kernel_sizes=self.kernel_sizes,
                    n_filters=self.n_filters,
                    dropout_rate=self.dropout_rate,
                    max_text_length=self.max_text_length,
                    pretrained_word_embeddings=self.init_params.get('pretrained_word_embeddings'),
                    verbose=self.verbose,
                    seed=self.seed,
                )
            self._fit()

        return self

    def _fit(self):
        import tensorflow as tf
        from tensorflow import keras
        from .hrdr import get_data
        from ...eval_methods.base_method import rating_eval
        from ...metrics import MSE
        loss = keras.losses.MeanSquaredError()
        if not hasattr(self, 'optimizer_'):
            if self.optimizer == 'rmsprop':
                self.optimizer_ = keras.optimizers.RMSprop(learning_rate=self.learning_rate)
            else:
                self.optimizer_ = keras.optimizers.Adam(learning_rate=self.learning_rate)
        train_loss = keras.metrics.Mean(name="loss")
        val_loss = float('inf')
        best_val_loss = float('inf')
        self.best_epoch = None
        loop = trange(self.max_iter, disable=not self.verbose, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        for i_epoch, _ in enumerate(loop):
            train_loss.reset_states()
            for i, (batch_users, batch_i_items, batch_j_items) in enumerate(self.train_set.uij_iter(self.batch_size, shuffle=True, neg_sampling="popularity")):
                user_reviews, user_num_reviews, user_ratings = get_data(batch_users, self.train_set, self.max_text_length, by='user', max_num_review=self.max_num_review)
                item_i_reviews, item_i_num_reviews, item_i_ratings = get_data(batch_i_items, self.train_set, self.max_text_length, by='item', max_num_review=self.max_num_review)
                item_j_reviews, item_j_num_reviews, item_j_ratings = get_data(batch_j_items, self.train_set, self.max_text_length, by='item', max_num_review=self.max_num_review)
                with tf.GradientTape() as tape:
                    loss = self.model.graph(
                        [
                            batch_users, batch_i_items, batch_j_items, 
                            user_ratings, user_reviews, user_num_reviews, 
                            item_i_ratings, item_i_reviews, item_i_num_reviews,
                            item_j_ratings, item_j_reviews, item_j_num_reviews
                        ],
                        training=True,
                    )
                gradients = tape.gradient(loss, self.model.graph.trainable_variables)
                self.optimizer_.apply_gradients(zip(gradients, self.model.graph.trainable_variables))
                train_loss(loss)
                if i % 10 == 0:
                    loop.set_postfix(loss=train_loss.result().numpy(), val_loss=val_loss, best_val_loss=best_val_loss, best_epoch=self.best_epoch)
            current_weights = self.model.get_weights(self.train_set, self.batch_size, max_num_review=self.max_num_review)
            if self.val_set is not None:
                self.P, self.Q, self.W1, self.bu, self.bi, self.mu, self.A = current_weights
                [val_loss], _ = rating_eval(
                    model=self,
                    metrics=[NDCG()],
                    test_set=self.val_set,
                    user_based=self.user_based
                )
                if best_val_loss < val_loss:
                    best_val_loss = val_loss
                    self.best_epoch = i_epoch + 1
                    best_weights = current_weights
                loop.set_postfix(loss=train_loss.result().numpy(), val_loss=val_loss, best_val_loss=best_val_loss, best_epoch=self.best_epoch)
            self.losses["train_losses"].append(train_loss.result().numpy())
            self.losses["val_losses"].append(val_loss)
        loop.close()

        # save weights for predictions
        self.P, self.Q, self.W1, self.bu, self.bi, self.mu, self.A = best_weights if self.val_set is not None and self.model_selection == 'best' else current_weights
        if self.verbose:
            print("Learning completed!")


    def save(self, save_dir=None):
        """Save a recommender model to the filesystem.

        Parameters
        ----------
        save_dir: str, default: None
            Path to a directory for the model to be stored.

        """
        if save_dir is None:
            return
        model = self.model
        del self.model

        model_file = Recommender.save(self, save_dir)

        self.model = model
        self.model.save(model_file.replace(".pkl", ".cpt"))

        return model_file

    @staticmethod
    def load(model_path, trainable=False):
        """Load a recommender model from the filesystem.

        Parameters
        ----------
        model_path: str, required
            Path to a file or directory where the model is stored. If a directory is
            provided, the latest model will be loaded.

        trainable: boolean, optional, default: False
            Set it to True if you would like to finetune the model. By default, 
            the model parameters are assumed to be fixed after being loaded.
        
        Returns
        -------
        self : object
        """
        from tensorflow import keras
        model = Recommender.load(model_path, trainable)
        model.model = keras.models.load_model(model.load_from.replace(".pkl", ".cpt"))

        return model

    def score(self, user_idx, item_idx=None):
        """Predict the scores/ratings of a user for an item.

        Parameters
        ----------
        user_idx: int, required
            The index of the user for whom to perform score prediction.

        item_idx: int, optional, default: None
            The index of the item for that to perform score prediction.
            If None, scores for all known items will be returned.

        Returns
        -------
        res : A scalar or a Numpy array
            Relative scores that the user gives to the item or to all known items
        """
        if item_idx is None:
            if self.train_set.is_unk_user(user_idx):
                raise ScoreException(
                    "Can't make score prediction for (user_id=%d)" % user_idx
                )
            known_item_scores = (self.P[user_idx] * self.Q).dot(self.W1) + self.bu[user_idx] + self.bi + self.mu
            return known_item_scores.ravel()
        else:
            if self.train_set.is_unk_user(user_idx) or self.train_set.is_unk_item(
                item_idx
            ):
                raise ScoreException(
                    "Can't make score prediction for (user_id=%d, item_id=%d)"
                    % (user_idx, item_idx)
                )
            known_item_score = (self.P[user_idx] * self.Q[item_idx]).dot(self.W1) + self.bu[user_idx] + self.bi[item_idx] + self.mu
            return known_item_score
