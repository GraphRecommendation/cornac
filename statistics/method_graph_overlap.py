import sys

import numpy as np
from tqdm import tqdm

from statistics import utils
from statistics.lightrla_graph_overlap import reverse_path, get_reviews, draw_reviews


def get_intersecting_reviews():
    pass


def run(path, dataset, method):
    # Get dataset and model
    eval_method = utils.initialize_dataset(dataset)
    model = utils.initialize_model(path, dataset, method)

    # Iter test
    res = []
    lengths = []
    for user, item in tqdm(list(zip(*eval_method.test_set.csr_matrix.nonzero()))):
        review_aos = set()
        if method == 'lightrla':
            # out = lightrla_overlap(eval_method, model, user, item)
            # r = reverse_match(user, item, eval_method.sentiment, 'aos')
            r = reverse_path(eval_method, user, item, 'a')
            sids = get_reviews(eval_method, model, r, 'a')
            if sids is not None:
                draw_reviews(eval_method, sids, user, item, 'a')


            if isinstance(r, tuple):
                r, lu, li = r
                lengths.append([lu, li])

            res.append(r)
        else:
            raise NotImplementedError
    # Get paths
    res = np.array(res)
    lengths = np.array(lengths)

    for i in range(0, 10):
        print(f'>={i}: all    {sum((lengths[:, 0] >= i) * (lengths[:, 1] >= i))}')
        print(f'>={i}: failed {sum((lengths[:, 0] >= i) * (lengths[:, 1] >= i) * (res == 2))}')

    pass


if __name__ == '__main__':
    path, dataset, method = sys.argv[1:]
    run(path, dataset, method)