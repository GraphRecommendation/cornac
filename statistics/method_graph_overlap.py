import os
import pickle

import argparse
from tqdm import tqdm


from statistics.lightrla_graph_overlap import reverse_path, draw_reviews
from statistics.utils import get_method_paths

parser = argparse.ArgumentParser()
parser.add_argument('path', type=str)
parser.add_argument('dataset')
parser.add_argument('method')
parser.add_argument('--rerun', action='store_true')
parser.add_argument('--method_kwargs', default="{'matching_method':'a'}", type=str)


def run(path, dataset, method, method_kwargs, draw=False, rerun=False):
    from statistics import utils, lightrla_graph_overlap2 as lightrla_graph_overlap, narre_graph_overlap, kgat_graph_overlap
    # Get dataset and model
    eval_method = utils.initialize_dataset(dataset)
    model = utils.initialize_model(path, dataset, method)
    matching_method = method_kwargs['matching_method']
    review_fname, graph_fname = get_method_paths(method_kwargs, dataset, method)

    # Iter test
    res = []
    # lengths = []
    uis = []
    uig = None
    if (not os.path.isfile(review_fname) or rerun) and method not in ['kgat']:
        for user, item in tqdm(list(zip(*eval_method.test_set.csr_matrix.nonzero()))):
            if method in ['lightrla', 'globalrla', 'globalrla-e']:
                r = reverse_path(eval_method, user, item, matching_method)
                # TODO fix, should not be none or better handling
                if r is None:
                    continue
                sids = lightrla_graph_overlap.get_reviews_nwx(eval_method, model, r, matching_method,
                                                              **method_kwargs[method])
                uis.append((user, item, sids))

                if sids is not None and draw:
                    draw_reviews(eval_method, sids, user, item, matching_method)

                res.append(r)
            elif method in ['narre', 'hrdr']:

                rids = narre_graph_overlap.get_reviews(eval_method, model, matching_method)
                uis.append((user, item, tuple(rids[item])))
            else:
                raise NotImplementedError

        print('Writing reviews to disk.')
        with open(review_fname, 'wb') as f:
            pickle.dump(uis, f)
    elif method not in ['kgat']:
        with open(review_fname, 'rb') as f:
            uis = pickle.load(f)

    # Create graph
    if method in ['lightrla', 'globalrla']:
        uig = lightrla_graph_overlap.sid_to_graphs(eval_method, uis, matching_method)
    elif method == 'kgat':
        _, lightrla_fname = get_method_paths(method_kwargs, dataset, 'lightrla')
        assert os.path.isfile(lightrla_fname), 'This method can be dependant on LightRLA (for fair comparison). ' \
                                               'Please run with lightrla before kgat.'

        lightrla_data = kgat_graph_overlap.load_data(lightrla_fname)
        uig = kgat_graph_overlap.get_reviews(eval_method, model, lightrla_data)

    if uig is not None:
        with open(graph_fname, 'wb') as f:
            pickle.dump(uig, f)
    else:
        print('Warning: No graphs extracted.')



if __name__ == '__main__':
    args = parser.parse_args()
    method_kwargs = eval(args.method_kwargs)
    run(args.path, args.dataset, args.method, rerun=args.rerun, method_kwargs=method_kwargs)