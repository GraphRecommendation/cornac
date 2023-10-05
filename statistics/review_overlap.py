import argparse
import collections
import math
import os
import pickle
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import evaluate
import json
import numpy as np
import pandas as pd
from rouge_score import rouge_scorer, scoring
from rouge_score.scoring import AggregateScore
from scipy.stats import ttest_ind, ttest_rel
from tqdm import tqdm

from statistics import utils
from statistics.method_graph_overlap import parameter_list
from statistics.utils import get_method_paths

parser = argparse.ArgumentParser()
parser.add_argument('--dataset')
parser.add_argument('--methods', nargs='+')
parser.add_argument('--data_path', type=str)
parser.add_argument('--method_kwargs', default='', type=str)


def rouge_all_res(predictions, references, rouge_types=None, use_aggregator=True, use_stemmer=False,
                  tokenizer=None):
    np.random.seed(42)
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


def meteor_all_res(predictions, references, alpha=0.9, beta=3, gamma=0.5):
    from datasets.config import importlib_metadata, version
    from nltk.translate import meteor_score
    NLTK_VERSION = version.parse(importlib_metadata.version("nltk"))
    if NLTK_VERSION >= version.Version("3.6.4"):
        from nltk import word_tokenize
    multiple_refs = isinstance(references[0], list)
    if NLTK_VERSION >= version.Version("3.6.5"):
        # the version of METEOR in NLTK version 3.6.5 and earlier expect tokenized inputs
        if multiple_refs:
            scores = [
                meteor_score.meteor_score(
                    [word_tokenize(ref) for ref in refs],
                    word_tokenize(pred),
                    alpha=alpha,
                    beta=beta,
                    gamma=gamma,
                )
                for refs, pred in zip(references, predictions)
            ]
        else:
            scores = [
                meteor_score.single_meteor_score(
                    word_tokenize(ref), word_tokenize(pred), alpha=alpha, beta=beta, gamma=gamma
                )
                for ref, pred in zip(references, predictions)
            ]
    else:
        if multiple_refs:
            scores = [
                meteor_score.meteor_score(
                    [[word_tokenize(ref) for ref in group] for group in references][0],
                    word_tokenize(pred),
                    alpha=alpha,
                    beta=beta,
                    gamma=gamma,
                )
                for ref, pred in zip(references, predictions)
            ]
        else:
            scores = [
                meteor_score.single_meteor_score(ref, pred, alpha=alpha, beta=beta, gamma=gamma)
                for ref, pred in zip(references, predictions)
            ]

    return {"meteor": scores}

def compute_metric(metric, pred, target):
    if metric in ['bleu']:
        pred = [[p] for p in pred]
        target = [[t] for t in target]
    kwargs = {'predictions': pred, 'references': target}

    if metric == 'rouge':
        rs = rouge_all_res(use_aggregator=False, **kwargs)
    elif metric.startswith('meteor'):
        metric, alpha = metric.split('_')
        kwargs['alpha'] = float(alpha)
        rs = meteor_all_res(**kwargs)
    elif metric == 'bleu':
        kwargs['max_order'] = 2
        rs = {metric: []}
        m = evaluate.load(metric)
        data = list(zip(kwargs['predictions'], kwargs['references']))
        for pair in data:
            kwargs['predictions'], kwargs['references'] = pair
            rs[metric].append(m.compute(**kwargs)['bleu'])
            # compute_bleu(
            #     reference_corpus=kwargs['references'], translation_corpus=kwargs['predictions'], max_order=2, smooth=False)
    elif metric == 'bleurt':
        m = evaluate.load(metric, 'bleurt-base-512')
        rs = m.compute(**kwargs)
    elif metric == 'bertscore':
        kwargs['lang'] = 'en'
        m = evaluate.load(metric)
        rs = m.compute(**kwargs)
        rs.pop('hashcode')

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
    # metrics = ['bleurt']
    for metric in tqdm(metrics, desc='Calculating metrics'):
        # only create new proces for proper cuda clearing
        with ProcessPoolExecutor(max_workers=1) as e:
            # rs = compute_metric(metric, pred, target)
            f = e.submit(compute_metric, metric, pred, target)
        rs = f.result()

        if metric not in ['bleu']:
            if isinstance(rs, list):
                for r in rs:
                    for m, s in r.items():
                        for f, v in zip(s._fields, range(len(s))):
                            n = f'{m}-{f}'
                            if n not in results:
                                results[n] = []

                            results[n].append(s[v])
            else:
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
    df['reviewerID'] = df['reviewerID'].apply(eval_method.test_set.uid_map.get)
    df['asin'] = df['asin'].apply(eval_method.test_set.iid_map.get)
    df.dropna(inplace=True)
    ui_review = df.groupby(['reviewerID', 'asin'])['sentence'].apply('. '.join).to_dict()
    return ui_review


def _get_statistics(eval_method, dataset, method, method_kwargs, parameter_kwargs, ui_review, mask, ui_pairs=None):
    review_fname, graph_fname = get_method_paths(method_kwargs, parameter_kwargs, dataset, method)

    # Load
    with open(review_fname, 'rb') as f:
        data = pickle.load(f)

    if method in ['lightrla', 'hypar', 'hypar-e']:
        data = extract_rid(eval_method, data)
        # mask = [len(e[-1]) > 2 for e in data]
        ui_pairs = {(u, i) for u, i, _ in data}
    else:
        data = [d for d in data if d[:2] in ui_pairs]

    return statistics(eval_method, ui_review, data, item_wise=True, mask=mask), ui_pairs

def _method_formatter(method):
    if method.startswith('global'):
        if method.endswith('weighted'):
            extension = 'w'
        elif method.endswith('_item'):
            extension = 'gi'
        elif method.endswith('item'):
            extension = 'i'
        else:
            extension = 'gu'

        return f'{method.replace("hypar", "LightRLA")}$_{{{extension}}}$'
    elif method == 'narre':
        return 'NARRE'
    elif method == 'hrdr':
        return 'HRDR'
    else:
        return method

def _column_formatter(metric: str):
    m = metric.split('-')[0]
    m = m.upper().replace('_', '-').replace('SUM', 'Sum').replace('SCORE', 'Score')
    return m

def run(dataset, methods, data_path='experiment/seer-ijcai2020/', method_kwargs=None):
    if method_kwargs is not None:
        method_kwargs = eval(method_kwargs)
    else:
        method_kwargs = {}
    all_results = {}
    df_name = os.path.join('statistics/output/',
                           f'review_scores_{dataset}_{"_".join(methods)}' +\
                           f'_{method_kwargs["matching_method"]}.csv')
    mask = None
    ui_pairs = None
    if True or not os.path.exists(df_name):
        all_results[dataset] = {}
        eval_method = utils.initialize_dataset(dataset)
        df = pd.read_csv(os.path.join('experiment', 'seer-ijcai2020', dataset, 'profile.csv'), sep=',')
        ui_review = extract_test_reviews(df, eval_method)
        for method in methods:
            if method in ['lightrla', 'hypar', 'hypar-e']:
                for methodology in method_kwargs[method].pop('methodologies'):
                    method_kwargs[method]['methodology'] = methodology
                    print(f'--{method}-{methodology}--')
                    if methodology != 'greedy_item':
                        idx = 1
                    else:
                        idx = -1
                    for para in parameter_list[:idx]:
                        name = method + f'-{methodology}' + '-'.join([f'{k}-{v}' for k, v in para.items()])
                        all_results[dataset][name], ui_pairs = \
                            _get_statistics(eval_method, dataset, method, method_kwargs, para, ui_review, mask, ui_pairs)

                    # all_results[dataset][method+ f'-{methodology}'], ui_pairs = \
                    #     _get_statistics(eval_method, dataset, method, method_kwargs, None, ui_review, mask, ui_pairs)
            else:
                print(f'--{method}--')
                all_results[dataset][method], ui_pairs = _get_statistics(eval_method, dataset, method, method_kwargs,
                                                                          None, ui_review, mask, ui_pairs)
        data2 = defaultdict(list)
        for dataset, method_res in all_results.items():
            for method, res in method_res.items():
                for metric, r in res.items():
                    data2['score'].append(np.mean(r))
                    data2['method'].append(method)
                    data2['metric'].append(metric)

        df2 = pd.DataFrame(data2)
        df3 = df2.set_index('method').pivot(columns='metric')
        df3.columns = df3.columns.get_level_values(1)
        df3.to_csv(df_name)
        df = df3

        with open(df_name.replace('csv', 'pickle'), 'wb') as f:
            pickle.dump(all_results, f)
    else:
        df = pd.read_csv(df_name)
        df = df.set_index(df.columns[0])
        # df.columns = df.iloc[0]
        # df = df.drop(df.index[0])
        # df = df.applymap(pd.to_numeric)

        with open(df_name.replace('csv', 'pickle'), 'rb') as f:
            all_results = pickle.load(f)

    columns = [c for c in df.columns if 'precision' not in c and 'recall' not in c and
               'high' not in c and 'low' not in c]
    df = df[columns]

    latex_table(df, all_results, dataset, _column_formatter, _method_formatter)


def latex_table(df, all_results, dataset, c_formatter=None, m_formatter=None):
    # method selection
    own_methods = [i for i in df.index if i.startswith('global')]
    baselines = [i for i in df.index if not i.startswith('global')]
    m = df.loc[own_methods].max().values
    base_m = df.loc[baselines].max().values

    # get statistical significance:
    statistical_significance = {'dependent': {}, 'independent':{}}
    for c in df.columns:
        # Get best own method and baseline
        ob = df[c].loc[own_methods].idxmax()
        bb = df[c].loc[baselines].idxmax()

        # Get score
        oscore = df[c].loc[ob]
        bscore = df[c].loc[bb]

        own_results = all_results[dataset][ob][c]
        baseline_results = all_results[dataset][bb][c]

        if oscore < bscore:
            alternative = 'less'
        else:
            alternative = 'greater'

        s = ttest_ind(own_results, baseline_results, alternative=alternative)
        statistical_significance['independent'][c] = s.pvalue
        s = ttest_rel(own_results, baseline_results, alternative=alternative)
        statistical_significance['dependent'][c] = s.pvalue

    # format
    # df.index = [_method_formatter(i) for i in df.index]
    # df.columns = [_column_formatter(c) for c in df.columns]

    # Assign
    df.loc['Improv \%'] = ((m - base_m) / np.abs(base_m)) * 100
    df.loc['Improv \%'] = test(df.loc['Improv \%'], stat_sig=statistical_significance['independent'])

    # Format func
    n_format = lambda x: f'{x:.4f}'

    # Get non improvement rows
    rows = [i for i in df.index if '%' not in i]
    s = df.style.highlight_max(subset=(rows, slice(None)), props='bfseries:', axis=0)
    s.format(n_format, subset=(rows, slice(None)))
    # s.format(partial(test, stat_sig=statistical_significance['independent']), subset=(['Improv \%'], slice(None)), )
    if c_formatter is not None:
        s.format_index(c_formatter, axis=1)
    if m_formatter is not None:
        s.format_index(m_formatter, axis=0)
    print(s.to_latex(clines='all;data'))

    print('\n\n--- Stat sig ---')
    print(json.dumps(statistical_significance, indent=2))

def test(series, stat_sig, p=.05):
    return [f'{x:.2f}*' if stat_sig[s] < p else f'{x:.2f}' for s, x in series.items()]

if __name__ == '__main__':
    args = parser.parse_args()
    run(**vars(args))
