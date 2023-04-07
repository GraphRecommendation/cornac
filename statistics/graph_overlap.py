import argparse
import os
import pickle
from collections import OrderedDict, defaultdict

import numpy as np
import pandas as pd

from cornac.utils.graph_construction import generate_mappings
from statistics import utils
from statistics.utils import id_mapping

parser = argparse.ArgumentParser()
parser.add_argument('--datasets', nargs='+')
parser.add_argument('--methods', nargs='+')
parser.add_argument('--data_path', type=str)
parser.add_argument('--matching_methodology', type=str)
parser.add_argument('--file_args', default='', type=str)


def extract_test_nodes(df, eval_method, match):
    aos_user, aos_item, aos_sent, user_aos, item_aos, sent_aos, a_mapping, o_mapping\
        = generate_mappings(eval_method.sentiment, match, True)

    test_nodes = OrderedDict()
    for (uid, iid), group in df.groupby(['reviewerID', 'asin']):
        uid, iid = eval_method.global_uid_map.get(uid), eval_method.global_iid_map.get(iid)

        if uid is None or iid is None:
            continue

        nodes = [id_mapping(eval_method, uid, 'u'), iid]

        for r, values in group.iterrows():
            aid = a_mapping.get(eval_method.sentiment.aspect_id_map.get(values.get('aspect')))
            oid = o_mapping.get(eval_method.sentiment.opinion_id_map.get(values.get('opinion')))

            if aid is not None and oid is not None:
                nodes.append(id_mapping(eval_method, aid, 'a'))
                nodes.append(id_mapping(eval_method, oid, 'o'))

        # Skip if no ao pairs
        if len(nodes) > 2:
            test_nodes[(uid, iid)] = nodes

    return test_nodes



def _tversky_index(s1, s2, alpha=1, beta=1):
    """setting alpha=beta=1 is the same as Tanimoto coefficient; alpha=beta=.5 is Søren-dice coefficient"""
    i = len(s1.intersection(s2))
    return i / (i + alpha * len(s1.difference(s2)) + beta * len(s2.difference(s1)))


def _overlap_coefficient(s1, s2):
    return len(s1.intersection(s2)) / min(len(s1), len(s2))


def _jaccard_index(s1, s2):
    i = len(s1.intersection(s2))
    return i / (len(s1) + len(s2) + i)


def statistics(eval_method, ui_nodes, data):
    results = defaultdict(list)
    m = id_mapping(eval_method, max(eval_method.sentiment.opinion_id_map.values()), 'o')
    selected_nodes = set()
    for uid, iid, g in data:
        inner_uid = id_mapping(eval_method, uid, 'u')
        target_nodes = ui_nodes.get((uid, iid))
        pred_nodes = set(g.nodes())

        if target_nodes is None:
            continue
        else:
            target_nodes = set(target_nodes)

        results['sørensen-dice-coefficient'].append(_tversky_index(pred_nodes, target_nodes, alpha=.5, beta=.5))
        results['tanimoto-coefficient'].append(_tversky_index(pred_nodes, target_nodes))
        results['overlap-coefficient'].append(_overlap_coefficient(pred_nodes, target_nodes))
        results['jaccard-index'].append(_jaccard_index(pred_nodes, target_nodes))
        selected_nodes.update(pred_nodes.difference({inner_uid, iid}))

    results['diversity'] = len(selected_nodes) / m
    return results


def aggregate_metrics(results):
    aggregated_results = defaultdict(lambda: defaultdict(dict))
    quantiles = [0.001, 0.01, 0.05, .95, .99, 0.999]
    for dataset, mres in results.items():
        for method, mcres in mres.items():
            for metric, res in mcres.items():
                if metric == 'diversity':
                    aggregated_results[dataset][method][metric] = res
                    continue

                aggregated_results[dataset][method][f'{metric}-avg'] = np.mean(res)
                aggregated_results[dataset][method][f'{metric}-min'] = np.min(res)
                aggregated_results[dataset][method][f'{metric}-max'] = np.max(res)
                aggregated_results[dataset][method][f'{metric}-std'] = np.std(res)
                aggregated_results[dataset][method][f'{metric}-median'] = np.median(res)
                for r, q in zip(np.quantile(res, quantiles), quantiles):
                    aggregated_results[dataset][method][f'{metric}-q-{q}'] = r


    aggregated_results = {d: dict(m) for d, m in aggregated_results.items()}
    return aggregated_results


def run(datasets, methods, data_path='experiment/seer-ijcai2020/', matching_methodology='a', file_args=''):
    all_results = defaultdict(dict)
    mask = None
    ui_pairs = None
    base_path = 'statistics/output/'
    for dataset in datasets:
        print(f'----{dataset}----')
        all_results[dataset] = {}
        eval_method = utils.initialize_dataset(dataset)
        df = pd.read_csv(os.path.join('experiment', 'seer-ijcai2020', dataset, 'profile.csv'), sep=',')
        ui_nodes = extract_test_nodes(df, eval_method, matching_methodology)

        for method in methods:
            fname = os.path.join(base_path, f'selected_graphs_{dataset}_{method}_{matching_methodology}' \
                f'{"_" + file_args if file_args else ""}.pickle')

            # Load
            with open(fname, 'rb') as f:
                data = pickle.load(f)

            all_results[dataset][method] = statistics(eval_method, ui_nodes, data)

    o_fname = os.path.join(base_path, f'graph_scores_{"_".join(datasets)}_{"_".join(methods)}_{matching_methodology}'
                                      f'.pickle')
    with open(o_fname, 'wb') as f:
        pickle.dump(all_results, f)

    all_results = aggregate_metrics(all_results)
    o_fname = os.path.join(base_path, f'graph_scores_{"_".join(datasets)}_{"_".join(methods)}_{matching_methodology}'
                                      f'_aggregated.pickle')
    with open(o_fname, 'wb') as f:
        pickle.dump(all_results, f)

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
    df3.to_csv(o_fname.replace('.pickle','.csv'))


if __name__ == '__main__':
    args = parser.parse_args()
    run(**vars(args))