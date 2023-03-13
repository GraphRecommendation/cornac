import os
import pickle
import sys

import numpy as np
from tqdm import tqdm

from statistics import utils, lightrla_graph_overlap, narre_graph_overlap
from statistics.lightrla_graph_overlap import reverse_path, get_reviews, draw_reviews


def get_intersecting_reviews():
    pass


def run(path, dataset, method, draw=False, rerun=False):
    # Get dataset and model
    eval_method = utils.initialize_dataset(dataset)
    matching_method = 'a'
    methodology = 'weighting'
    model = utils.initialize_model(path, dataset, method)
    fname = f'statistics/output/selected_reviews_{dataset}_{method}_{matching_method}_{methodology}_{position}.pickle'

    # Iter test
    res = []
    # lengths = []
    uis = []
    if (not os.path.isfile(fname)) or rerun:
        for user, item in tqdm(list(zip(*eval_method.test_set.csr_matrix.nonzero()))):
            if method == 'lightrla':
                t = [s for sent in eval_method.sentiment.item_sentiment[item].values() for s in eval_method.sentiment.sentiment[sent]]
                r = reverse_path(eval_method, user, item, 'a')
                # TODO fix, should not be none or better handling
                if r is None:
                    continue
                sids = lightrla_graph_overlap.get_reviews_nwx(eval_method, model, r, matching_method,
                                                              methodology=methodology)
                # sids = lightrla_graph_overlap.get_reviews(eval_method, model, r, matching_method)
                uis.append((user, item, sids))

                if sids is not None and draw:
                    draw_reviews(eval_method, sids, user, item, matching_method)

                # if isinstance(r, tuple):
                #     r, lu, li = r
                #     lengths.append([lu, li])

                res.append(r)
            elif method == 'narre':
                rids = narre_graph_overlap.get_reviews(eval_method, model, matching_method)
                uis.append((user, item, tuple(rids[item])))
            else:
                raise NotImplementedError

        # Get paths
        # res = np.array(res)
        # lengths = np.array(lengths)

        # for i in range(0, 10):
        #     print(f'>={i}: all    {sum((lengths[:, 0] >= i) * (lengths[:, 1] >= i))}')
        #     print(f'>={i}: failed {sum((lengths[:, 0] >= i) * (lengths[:, 1] >= i) * (res == 2))}')

        print('Writing reviews to disk.')
        with open(fname, 'wb') as f:
            pickle.dump(uis, f)
    else:
        print('Loading existing reviews')
        with open(fname, 'rb') as f:
            uis = pickle.load(f)


if __name__ == '__main__':
    if len(sys.argv) == 4:
        path, dataset, method = sys.argv[1:]
        rerun = False
    else:
        path, dataset, method, rerun = sys.argv[1:]
        rerun = bool(rerun)
    run(path, dataset, method, rerun=rerun)