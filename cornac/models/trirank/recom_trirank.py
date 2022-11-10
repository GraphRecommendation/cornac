import numpy as np
import torch

from scipy.sparse import coo_matrix, csr_matrix

from cornac.models import Recommender


class TriRank(Recommender):
    def __init__(self, name='TriRank'):
        super().__init__(name)
        self.alpha = 0.1
        self.beta = 0.1
        self.gamma = 0.1
        self.mu_U = 0.1
        self.mu_P = 0.1
        self.mu_A = 0.1
        self.verbose = True

        self.X = None
        self.Y = None
        self.S_R = None
        self.S_X = None
        self.S_Y = None

        self.p_v = None
        self.a_v = None
        self.u_v = None

    def _symmetrical_normalization(self, matrix: csr_matrix):
        row = []
        col = []
        data = []
        row_norm = np.sqrt(matrix.sum(axis=1).A1)
        col_norm = np.sqrt(matrix.sum(axis=0).A1)
        for i, j in zip(*matrix.nonzero()):
            row.append(i)
            col.append(j)
            data.append(matrix[i, j] / (row_norm[i] * col_norm[j]))

        return csr_matrix((data, (row, col)), shape=matrix.shape)

    def _create_matrices(self, train_set):
        sentiment_modality = train_set.sentiment
        n_users = len(train_set.uid_map)
        n_items = len(train_set.iid_map)
        n_aspects = len(sentiment_modality.aspect_id_map)

        X_row = []
        X_col = []
        X_data = []
        Y_row = []
        Y_col = []
        Y_data = []
        for uid, isid in sentiment_modality.user_sentiment.items():
            for iid, sid in isid.items():
                aos = sentiment_modality.sentiment[sid]
                aids = set(aid for aid, _, _ in aos)  # Only one per review/sid
                for aid in aids:
                    X_row.append(iid)
                    X_col.append(aid)
                    X_data.append(1)
                    Y_row.append(uid)
                    Y_col.append(aid)
                    Y_data.append(1)

        # Algorithm 1: Offline training line 2
        self.X = csr_matrix((X_data, (X_row, X_col)), shape=(n_items, n_aspects))
        self.Y = csr_matrix((Y_data, (Y_row, Y_col)), shape=(n_users, n_aspects))

        # Algorithm 1: Offline training line 3
        self.X.data = np.log2(self.X.data) + 1
        self.Y.data = np.log2(self.Y.data) + 1

        # Algorithm 1: Offline training line 4
        self.S_R = self._symmetrical_normalization(train_set.csr_matrix)
        self.S_X = self._symmetrical_normalization(self.X)
        self.S_Y = self._symmetrical_normalization(self.Y)

        # Initialize user, item and aspect rank.
        self.p_v = np.random.uniform(0, 1, n_items)
        self.a_v = np.random.uniform(0, 1, n_aspects)
        self.u_v = np.random.uniform(0, 1, n_users)

    def fit(self, train_set, val_set=None):
        super(TriRank, self).fit(train_set, val_set)
        from cornac.metrics import MSE, NDCG
        from cornac.eval_methods import rating_eval, ranking_eval

        # Build item-aspect matrix X and user-aspect matrix Y
        self._create_matrices(train_set)

        # Only run if verbose and val set. Not used.
        if val_set is not None and self.verbose:
            (mse,), _ = rating_eval(self, [MSE()], val_set, user_based=True, verbose=self.verbose)
            (ndcg,), _ = ranking_eval(self, [NDCG(20), ], self.train_set, val_set, verbose=self.verbose)
            if self.verbose:
                print(f"MSE: {mse}, NDCG: {ndcg}")

    def _fit(self, user):
        # Algorithm 1: Online recommendation line 5
        p_0 = self.train_set.csr_matrix[user]
        indices = p_0.nonzero()
        p_0[indices] = 1
        p_0 = p_0.toarray().squeeze()
        a_0 = self.Y[user].toarray().squeeze()
        u_0 = np.zeros(self.train_set.csr_matrix.shape[0])
        u_0[user] = 1

        # Algorithm 1: Online training line 6
        p_0 /= np.linalg.norm(p_0, 1)
        a_0 /= np.linalg.norm(a_0, 1)
        u_0 /= np.linalg.norm(u_0, 1)

        # Algorithm 1: Online recommendation line 7
        p_v = self.p_v.copy()
        a_v = self.a_v.copy()
        u_v = self.u_v.copy()

        # Algorithm 1: Online recommendation line 8
        converged = False
        prev_p = p_v
        prev_a = a_v
        prev_u = u_v
        while not converged:
            # eq. 4
            u_v = self.alpha / (self.alpha + self.gamma + self.mu_U) * self.S_R * p_v +\
                self.gamma / (self.alpha + self.gamma + self.mu_U) * self.S_Y * a_v +\
                self.mu_U / (self.alpha + self.gamma + self.mu_U) * u_0
            u_v = u_v.squeeze()
            p_v = self.alpha / (self.alpha + self.beta + self.mu_P) * self.S_R.T * u_v +\
                self.beta / (self.alpha + self.beta + self.mu_P) * self.S_X * a_v +\
                self.mu_P / (self.alpha + self.beta + self.mu_P) * p_0
            p_v = p_v.squeeze()
            a_v = self.gamma / (self.gamma + self.beta + self.mu_A) * self.S_Y.T * u_v +\
                self.beta / (self.gamma + self.beta + self.mu_A) * self.S_X.T * p_v +\
                self.mu_P / (self.gamma + self.beta + self.mu_A) * a_0
            a_v = a_v.squeeze()

            if np.all(np.isclose(u_v, prev_u)) and np.all(np.isclose(p_v, prev_p)) and np.all(np.isclose(a_v, prev_a)):
                converged = True
            else:
                prev_p, prev_a, prev_u = p_v, a_v, u_v

        # Algorithm 1: Online recommendation line 9
        return p_v

    def score(self, user_idx, item_idx=None):
        item_rank = self._fit(user_idx)
        if item_idx is not None:
            # Set already rated items to zero.
            item_rank[self.train_set.csr_matrix[user_idx].indices] = 0

            # Scale to match rating scale.
            item_rank = (self.train_set.max_rating - self.train_set.min_rating) * item_rank / max(item_rank) +\
                self.train_set.min_rating

            return item_rank[item_idx]
        else:
            # return np.random.uniform(size=self.train_set.csr_matrix.shape[1])
            return item_rank

    def monitor_value(self):
        pass