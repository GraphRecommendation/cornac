import argparse
import os
import pickle
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import evaluate
import numpy as np
import pandas as pd
from rouge_score import rouge_scorer, scoring
from rouge_score.scoring import AggregateScore
from tqdm import tqdm

from statistics import utils
parser = argparse.ArgumentParser()
parser.add_argument('--datasets', nargs='+')
parser.add_argument('--methods', nargs='+')
parser.add_argument('--data_path', type=str)


def rouge_all_res(predictions, references, rouge_types=None, use_aggregator=True, use_stemmer=False,
                  tokenizer=None):
    if rouge_types is None:
        rouge_types = ["rouge1", "rouge2", "rougeL", "rougeLsum"]

    multi_ref = isinstance(references[0], list)

    scorer = rouge_scorer.RougeScorer(rouge_types=rouge_types, use_stemmer=use_stemmer, tokenizer=tokenizer)
    if use_aggregator:
        aggregator = scoring.BootstrapAggregator()
    else:
        scores = []

    for ref, pred in zip(references, predictions):
        if multi_ref:
            score = scorer.score_multi(ref, pred)
        else:
            score = scorer.score(ref, pred)
        if use_aggregator:
            aggregator.add_scores(score)
        else:
            scores.append(score)

    if use_aggregator:
        result = aggregator.aggregate()

    else:
        result = scores

    return result


def compute_metric(metric, pred, target):
    if metric in ['bleu']:
        target = [[t] for t in target]
    kwargs = {'predictions': pred[:100], 'references': target[:100]}

    if metric.startswith('meteor'):
        metric, alpha = metric.split('_')
        kwargs['alpha'] = float(alpha)

    if metric in ['bertscore']:
        kwargs['lang'] = 'en'

    if metric == 'rouge':
        rs = rouge_all_res(**kwargs)
    else:
        m = evaluate.load(metric)
        rs = m.compute(**kwargs)

    if metric in ['bertscore', 'bleurt']:
        rs = {r: np.mean(s) for r, s in rs.items() if isinstance(s, list)}

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

    # Count number of words in pred and target
    tl, pl = [sum([len(sentence.split(' ')) for sentence in l]) for l in [target, pred]]
    print(f'Avg. target len: {tl / len(target)}, pred len: {pl/ len(pred)}')

    if mask is not None:
        target, pred = np.array(target), np.array(pred)
        target, pred = target[mask], pred[mask]
        target, pred = target.tolist(), pred.tolist()

    results = {}
    metrics = ['rouge', 'meteor_0.9', 'meteor_0.1', 'bleu', 'bertscore', 'bleurt']
    # metrics = ['bleu', 'bertscore', 'bleurt']
    for metric in tqdm(metrics, desc='Calculating metrics'):
        # only create new proces for proper cuda clearing
        with ProcessPoolExecutor(max_workers=1) as e:
            # rs = test(metric, pred, target)
            f = e.submit(compute_metric, metric, pred, target)
        rs = f.result()

        if metric not in ['bleu']:
            for m, r in rs.items():
                if isinstance(r, AggregateScore):
                    for percentile, i in zip(r._fields, range(len(r))):
                        for s, j in zip(r[i]._fields, range(len(r[i]))):
                            results[f'{m}-{percentile}-{s}'] = r[i][j]
                elif metric == 'bertscore':
                    results[f'{metric}-{m}'] = r
                else:
                    results[f'{metric}'] = r
        else:
            results[metric] = rs[metric]

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

    data2 = defaultdict(list)
    for dataset, method_res in all_results.items():
        for method, res in method_res.items():
            for metric, r in res.items():
                data2[dataset].append(r)
                data2['method'].append(method)
                data2['metric'].append(metric)

    df2 = pd.DataFrame(data2)
    df3 = df2.set_index('method').pivot(columns='metric').rename_axis(None, axis=0)
    print(df3)
    df3.to_csv(os.path.join('statistics/output/', f'review_scores_{"_".join(datasets)}_{"_".join(methods)}.csv'))
    # Todo create latex table


if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    args = parser.parse_args()
    run(**vars(args))
