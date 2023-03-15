import os
import pickle
import sys

import numpy as np
from tqdm import tqdm

from statistics import utils, lightrla_graph_overlap, narre_graph_overlap, kgat_graph_overlap
from statistics.lightrla_graph_overlap import reverse_path, draw_reviews


def get_intersecting_reviews():
    pass


def run(path, dataset, method, draw=False, rerun=False):
    # Get dataset and model
    eval_method = utils.initialize_dataset(dataset)
    matching_method = 'a'
    methodology = 'greedy_item'
    model = utils.initialize_model(path, dataset, method)
    review_fname = f'statistics/output/selected_reviews_{dataset}_{method}_{matching_method}_{methodology}.pickle'
    graph_fname = f'statistics/output/selected_graphs_{dataset}_{method}_{matching_method}_{methodology}.pickle'

    # Iter test
    res = []
    # lengths = []
    uis = []
    uig = None
    if not os.path.isfile(review_fname) or rerun:
        for user, item in tqdm(list(zip(*eval_method.test_set.csr_matrix.nonzero()))):
            if method == 'lightrla':
                r = reverse_path(eval_method, user, item, 'a')
                # TODO fix, should not be none or better handling
                if r is None:
                    continue
                sids = lightrla_graph_overlap.get_reviews_nwx(eval_method, model, r, matching_method,
                                                              methodology=methodology)
                uis.append((user, item, sids))

                if sids is not None and draw:
                    draw_reviews(eval_method, sids, user, item, matching_method)

                res.append(r)
            elif method == 'narre':
                rids = narre_graph_overlap.get_reviews(eval_method, model, matching_method)
                uis.append((user, item, tuple(rids[item])))
            elif method == 'kgat':
                lightrla_fname = review_fname.replace(method, 'lightrla')
                assert os.path.isfile(lightrla_fname), 'This method is dependant on LightRLA (for' \
                                                                          'fair comparison). Please run with lightrla' \
                                                                          'before kgat.'
                lightrla_data = kgat_graph_overlap.load_data(lightrla_fname)
                rids = kgat_graph_overlap.get_reviews(eval_method, model, lightrla_data)
                uis.append((user, item, tuple(rids[item])))
            else:
                raise NotImplementedError

        print('Writing reviews to disk.')
        with open(review_fname, 'wb') as f:
            pickle.dump(uis, f)
    else:
        with open(review_fname, 'rb') as f:
            uis = pickle.load(f)

    # Create graph
    if method == 'lightrla':
        uig = lightrla_graph_overlap.sid_to_graphs(eval_method, uis, matching_method)
    elif method == 'kgat':
        raise NotImplementedError

    if uig is not None:
        with open(graph_fname, 'wb') as f:
            pickle.dump(uig, f)
    else:
        print('Warning: No graphs extracted.')



if __name__ == '__main__':
    if len(sys.argv) == 4:
        path, dataset, method = sys.argv[1:]
        rerun = False
    else:
        path, dataset, method, rerun = sys.argv[1:]
        rerun = bool(rerun)
    run(path, dataset, method, rerun=rerun)