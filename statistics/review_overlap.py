import argparse
import os
import pickle
from functools import lru_cache

import nltk.translate.meteor_score
import pandas as pd
from tqdm import tqdm

from statistics import utils
parser = argparse.ArgumentParser()
parser.add_argument('--datasets', nargs='+')
parser.add_argument('--methods', nargs='+')
parser.add_argument('--data_path', type=str)


def statistics(eval_method, actual_review, data, item_wise=True):
    compare = []
    for (uid, iid, rids) in data:
        if item_wise:
            rid = rids[-1]
        else:
            raise NotImplementedError

        # get review
        selected = eval_method.review_text.reviews[rid]

        compare.append([actual_review[(uid, iid)], selected])

    tor


def extract_rid(eval_method, data):
    sid_ui = {sid: (u, i) for u, isid in eval_method.sentiment.user_sentiment.items() for i, sid in isid.items()}
    sid_rid = {sid: eval_method.review_text.user_review[u][i] for sid, (u, i) in sid_ui.items()}
    new_data = []
    for uid, iid, sids in data:
        rids = tuple(sid_rid[sid] for sid in sids)
        new_data.append((uid, iid, rids))

    return new_data


def extract_test_reviews(df, eval_method):
    test_ratings = list(zip(*eval_method.test_set.csr_matrix.nonzero()))
    uid_oid = {v: k for k, v in eval_method.test_set.uid_map.items()}
    iid_oid = {v: k for k, v in eval_method.test_set.iid_map.items()}

    ui_review = {}

    for uid, iid in tqdm(test_ratings, desc='Extracting reviews'):
        o_uid, o_iid = uid_oid[uid], iid_oid[iid]  # get original ids.

        ui_df = df[(df.reviewerID == o_uid) & (df.asin == o_iid)]
        review = '.'.join(ui_df.sentence.tolist())
        ui_review[(uid, iid)] = review

    return ui_review


def run(datasets, methods, data_path='experiment/seer-ijcai2020/'):
    for dataset in datasets:
        eval_method = utils.initialize_dataset(dataset)
        df = pd.read_csv(os.path.join('experiment', 'seer-ijcai2020', dataset, 'profile.csv'), sep=',')
        ui_review = extract_test_reviews(df, eval_method)
        for method in methods:
            fname = f'statistics/output/selected_reviews_{dataset}_{method}.pickle'

            # Load
            with open(fname, 'rb') as f:
                data = pickle.load(f)

            data = extract_rid(eval_method, data)

            results = statistics(eval_method, ui_review, data)





if __name__ == '__main__':
    args = parser.parse_args()
    run(**vars(args))
