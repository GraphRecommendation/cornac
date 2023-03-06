import argparse
import gc
import os
import pickle
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import evaluate
import waitGPU as waitGPU
from numba import cuda
import tensorflow as tf
import numpy as np
import pandas as pd
import torch.cuda
from texttable import Texttable
from tqdm import tqdm

from statistics import utils
parser = argparse.ArgumentParser()
parser.add_argument('--datasets', nargs='+')
parser.add_argument('--methods', nargs='+')
parser.add_argument('--data_path', type=str)


def test(metric, pred, target):
    m = evaluate.load(metric)

    if metric == 'rouge':
        metric_name = 'rougeL'
    elif metric == 'bertscore':
        metric_name = 'f1'
    elif metric == 'bleurt':
        metric_name = 'scores'
    else:
        metric_name = metric

    kwargs = {'predictions': pred, 'references': target}

    if metric in ['bertscore']:
        kwargs['lang'] = 'en'

    with torch.no_grad():
        rs = m.compute(**kwargs)[metric_name]

    if metric in ['bertscore', 'bleurt']:
        rs = np.mean(rs)

    return rs


def statistics(eval_method, actual_review, data, item_wise=True, mask=None):
    compare = []
    for (uid, iid, rids) in data:
        if item_wise:
            rid = rids[-1]
        else:
            rid = rids[0]

        # get review
        selected = eval_method.review_text.reviews[rid]

        compare.append([actual_review[(uid, iid)], selected])

    target = [t for t, _ in compare]
    pred = [p for _, p in compare]

    if mask is not None:
        target, pred = np.array(target), np.array(pred)
        target, pred = target[mask], pred[mask]
        target, pred = target.tolist(), pred.tolist()

    results = {}
    metrics = ['rouge', 'meteor', 'bleu', 'bertscore', 'bleurt']
    for metric in tqdm(metrics, desc='Calculating metrics'):
        # only for proper cuda clearing
        with ProcessPoolExecutor(max_workers=1) as e:
            f = e.submit(test, metric, pred, target)
        rs = f.result()

        results[metric] = rs

        waitGPU.wait(memory_ratio=.5)

    return results


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
    all_results = {}
    mask = None
    ui_pairs = None
    for dataset in datasets:
        print(f'----{dataset}----')
        all_results[dataset] = {}
        eval_method = utils.initialize_dataset(dataset)
        df = pd.read_csv(os.path.join('experiment', 'seer-ijcai2020', dataset, 'profile.csv'), sep=',')
        ui_review = extract_test_reviews(df, eval_method)
        for method in methods:
            print(f'--{method}--')
            fname = f'statistics/output/selected_reviews_{dataset}_{method}.pickle'

            # Load
            with open(fname, 'rb') as f:
                data = pickle.load(f)

            if method == 'lightrla':
                data = extract_rid(eval_method, data)
                # mask = [len(e[-1]) > 2 for e in data]
                ui_pairs = {(u, i) for u, i, _ in data}
            else:
                data = [d for d in data if d[:2] in ui_pairs]

            print('Calculating statistics')
            all_results[dataset][method] = statistics(eval_method, ui_review, data, item_wise=True, mask=mask)
            old_d = data

        # table = Texttable()
        # results = all_results[dataset]
        # metrics = list(next(iter(all_results[dataset].values())).keys())
        # methods_res = [[method] + [results[method][metric] for metric in metrics] for method in methods]
        # data = [[''] + metrics,
        #         *methods_res]
        #
        # table.add_rows(data)
        # table.draw()
        # latextable.draw_latex(table, caption=f'Results for dataset {dataset}')

    data2 = defaultdict(list)
    for dataset, method_res in all_results.items():
        for method, res in method_res.items():
            for metric, r in res.items():
                data2[dataset].append(r)
                data2['method'].append(method)
                data2['metric'].append(metric)

    df2 = pd.DataFrame(data2)
    print(df2.set_index('method').pivot(columns='metric').rename_axis(None, axis=0))
    # Todo create latex table


if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    args = parser.parse_args()
    run(**vars(args))
