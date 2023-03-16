import argparse
import os
import pickle
from collections import OrderedDict

import pandas as pd

from statistics import utils
from statistics.utils import generate_mappings

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
        uid, iid = eval_method.global_uid_map[uid], eval_method.global_iid_map[iid]
        nodes = [uid, iid]

        for r, values in group.iterrows():
            aid = a_mapping[eval_method.sentiment.aspect_id_map[values['aspect']]]
            oid = o_mapping[eval_method.sentiment.opinion_id_map[values['opinion']]]

            nodes.append(aid)
            nodes.append(oid)

        test_nodes[(uid, iid)] = nodes

    return test_nodes


def statistics(eval_method, ui_nodes, data):
    pass


def run(datasets, methods, data_path='experiment/seer-ijcai2020/', matching_methodology='a', file_args=''):
    all_results = {}
    mask = None
    ui_pairs = None
    for dataset in datasets:
        print(f'----{dataset}----')
        all_results[dataset] = {}
        eval_method = utils.initialize_dataset(dataset)
        df = pd.read_csv(os.path.join('experiment', 'seer-ijcai2020', dataset, 'profile.csv'), sep=',')
        ui_nodes = extract_test_nodes(df, eval_method, matching_methodology)

        for method in methods:
            if method == 'lightrla':
                fname = f'statistics/output/selected_graphs_{dataset}_{method}_{matching_methodology}' \
                        f'{"_" + file_args if file_args else ""}.pickle'
            else:
                fname = f'statistics/output/selected_graphs_{dataset}_{method}_{matching_methodology}.pickle'

            # Load
            with open(fname, 'rb') as f:
                data = pickle.load(f)

    pass

if __name__ == '__main__':
    args = parser.parse_args()
    run(**vars(args))