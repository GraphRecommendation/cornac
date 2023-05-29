import argparse
import math
import random
from itertools import combinations

import networkx as nx
import numpy as np
import pickle

from collections import Counter, OrderedDict, defaultdict

from functools import lru_cache

from cachetools.func import lfu_cache
from scipy import stats
from tqdm import tqdm

from cornac.utils.graph_construction import generate_mappings
from statistics import utils, lightrla_graph_overlap, kgat_graph_overlap

parser = argparse.ArgumentParser()
parser.add_argument('path', type=str)
parser.add_argument('dataset')
parser.add_argument('method')
parser.add_argument('--at', default=20, type=int)
parser.add_argument('--method_kwargs', default="{'matching_method':'a'}", type=str)


@lfu_cache
def _get_aos_sets(item_aos, n):
    ac = defaultdict(int)
    for aoss in item_aos:
        if len(aoss) >= n:
            for s in set(combinations(aoss, n)):
                ac[s] += 1

    i_a = {i: k for i, k in enumerate(ac)}
    ids = np.array([i for i in range(len(ac))])
    probability = np.array([ac[i_a[i]] for i in range(len(ac))])
    probability = probability / sum(probability)

    return i_a, ids, probability


def _negative_selector(item_aos, i_aos, n, popularity_biased=True):
    i_a, ids, probability = _get_aos_sets(i_aos, n)

    if popularity_biased:
        choice = np.random.choice(ids, p=probability)
    else:
        choice = np.random.choice(ids)

    choice = set(i_a[choice])
    neg_matched = [item for item, i_aos in item_aos.items() if choice.issubset(i_aos)]
    return neg_matched


def run(path, dataset, method, at, method_kwargs, negative_sampling=True):
    eval_method = utils.initialize_dataset(dataset)
    np.random.seed(eval_method.seed)
    random.seed(eval_method.seed)
    model = utils.initialize_model(path, dataset, method)
    matching_method = method_kwargs['matching_method']
    review_fname, graph_fname = utils.get_method_paths(method_kwargs, dataset, method)
    aos_user, aos_item, _, user_aos, item_aos, sent_aos, a_mapping, o_mapping = \
        generate_mappings(eval_method.sentiment, matching_method, get_ao_mappings=True)

    # rid_sid = {rid: eval_method.sentiment.user_sentiment[uid][iid]
    #            for uid, irid in eval_method.review_text.user_review.items()
    #            for iid, rid in irid.items()}
    # rid_iid = {rid: iid for uid, irid in eval_method.review_text.user_review.items() for iid, rid in irid.items()}
    # rid_aos = {rid: sent_aos[sid] for rid, sid in rid_sid.items()}

    i_aoss = frozenset({frozenset(aoss) for aoss in sent_aos.values()})

    most_pop_c = Counter(eval_method.train_set.matrix.nonzero()[1])
    most_pop = sorted(most_pop_c)
    mp_scores = [most_pop_c.get(m) for m in most_pop]
    mp_sort = np.argsort(mp_scores)
    mp_ranks = np.argsort(mp_sort)

    # item to aos set
    found = 0
    rankings = []
    n_rankings = []
    nr_rankings = []
    mp_rankings = []
    with tqdm(list(eval_method.test_set.user_data.keys())) as progressbar:
        rsum = 0
        rlen = 0
        nrsum = 0
        nrlen = 0
        mrsum = 0

        if method == 'kgat':
            raise NotImplementedError
        elif matching_method == 'a':
            diversity_max = eval_method.sentiment.num_aspects
        elif matching_method == 'ao':
            diversity_max = eval_method.sentiment.num_aspects + eval_method.sentiment.num_opinions
        else:
            raise NotImplementedError
        if method == 'kgat':
            mrsum = utils.id_mapping(eval_method, eval_method.sentiment.num_opinions, 'o') \
                    - eval_method.train_set.num_items

            g = model.train_graph.to_networkx()
            attention = model.train_graph.edata['a']
            attention = {e: 1 - v for e, v in
                         zip(g.edges, attention.numpy())}  # attention is inversed so lower is better

            # Get average number of nodes in train graphs
            lengths = [len({e for aos in eval_method.sentiment.sentiment[sid] for e in aos[:2]})
                       for iaos in eval_method.sentiment.user_sentiment.values()
                       for sid in iaos.values()]
            num_nodes = np.median(lengths) + 2  # Add two to account for user/item

            nx.set_edge_attributes(g, attention, 'a')
            g_simple = nx.Graph(g)

        diversity = set()
        for uid in progressbar:
            rated = eval_method.train_set.user_data[uid][0]
            item_ranking = model.score(uid)
            item_ranking[rated] = min(item_ranking)
            sorting = item_ranking.argsort()
            ranks = sorting.argsort()
            top_k = sorting[-at:]
            for iid in top_k:
                if method == 'lightrla':
                    r = lightrla_graph_overlap.reverse_path(eval_method, uid, iid, matching_method)

                    if r is None:
                        continue

                    sids = lightrla_graph_overlap.get_reviews_nwx(eval_method, model, r, matching_method,
                                                                  **method_kwargs[method])

                    s_aos = set(sent_aos[sids[-1]])
                    # s_aos = set(a for s in sids[-1:] for a in sent_aos[s])
                    diversity.update(s_aos)
                    matched = [item for item, i_aos in item_aos.items() if s_aos.issubset(i_aos)]
                    # matched = [item for rid, r_aos in rid_aos.items() if (item:= rid_iid[rid]) != iid and
                    #            s_aos.issubset(r_aos) and item not in rated]
                elif method == 'kgat':
                    edges = kgat_graph_overlap.get_path(eval_method, uid, iid, g, g_simple, num_nodes)
                    last_nodes = list(sorted({src for src, dst, i in edges if dst == iid}))
                    diversity.update(last_nodes)
                    matched = None
                    for ln in last_nodes:
                        n = set(g_simple.neighbors(ln))
                        if matched is None:
                            matched = n
                        else:
                            matched.intersection_update(n)

                    matched = list(i for i in matched if i < len(eval_method.global_iid_map))
                elif method == 'globalrla':
                    pass

                matched = [m for m in matched if m != iid and m not in rated]
                ir = ranks[matched]
                mir = mp_ranks[matched]

                # diversity.update(matched)
                if len(ir) > 0:
                    if negative_sampling and len(matched):
                        neg_matched = _negative_selector(item_aos, i_aoss, len(s_aos))
                        neg_matched2 = [m for m in neg_matched if m != iid and m not in rated]
                        nir = ranks[neg_matched2]
                        neg_matched = _negative_selector(item_aos, i_aoss, len(s_aos), popularity_biased=False)
                        neg_matched2 = [m for m in neg_matched if m != iid and m not in rated]
                        nr_rankings.append((uid, iid, ranks[neg_matched2]))

                    rsum += sum(ir)
                    rlen += len(ir)
                    mrsum += sum(mir)
                    nrsum += sum(nir)
                    nrlen += len(nir)
                    rankings.append((uid, iid, ir))
                    n_rankings.append((uid, iid, nir))
                    mp_rankings.append((uid, iid, mir))
                    found += 1
                else:
                    pass

            if rlen:
                progressbar.set_description(f'AR:{rsum / rlen:.3f},MAR:{mrsum / rlen:.3f},' +
                                            (f'NIR:{nrsum / nrlen:.3f},' if negative_sampling else '') +
                                            f'PF:{found / ((progressbar.n + 1) * at):.3f},'
                                            f'AN:{rlen / ((progressbar.n + 1) * at):.3f},'
                                            f'D:{len(diversity) / diversity_max:.3f}')

    # Test towards random, i.e., better than the mean.
    sorting_fn = lambda x: sorted(x, key=lambda y: y[:2])
    rank_u = np.array([np.mean(r) for _, _, r in sorting_fn(rankings) if len(r)])
    rank_f = np.array([e for _, _, r in sorting_fn(rankings) for e in r])

    qs = [.01, .05, .1, .5, .9, .95, .99]
    print('qs: ', qs)
    quantiles_u = np.quantile(rank_u, qs)
    quantiles_f = np.quantile(rank_f, qs)
    print('uq: ', quantiles_u)
    print('fq: ', quantiles_f)

    for other_rankings, name in zip([len(eval_method.global_iid_map) / 2, mp_rankings, n_rankings, nr_rankings],
                                    ['random', 'MostPop', 'popular negative', 'random negative']):
        print('')
        print('-'*(6+len(name)+3))
        print(f'Name: {name}')
        ts_f = None
        if isinstance(other_rankings, list):
            other_rankings_u, inner_r = np.array([[np.mean(r), r2] for (_, _, r), r2 in
                                                  zip(sorting_fn(other_rankings), rank_u) if len(r)]).T
            quantiles_u = np.quantile(other_rankings_u, qs)
            print('uq: ', quantiles_u)
            ts_u, pu = stats.ttest_rel(inner_r, other_rankings_u, alternative='greater')

            # negative sampling is not guaranteed to have same number of negative samples.
            if name == 'MostPop':
                other_rankings_f = np.array([e for _, _, r in sorting_fn(other_rankings) for e in r])
                quantiles_f = np.quantile(other_rankings_f, qs)
                print('fq: ', quantiles_f)
                ts_f, pf = stats.ttest_rel(rank_f, other_rankings_f, alternative='greater')

        else:
            ts_u, pu = stats.ttest_1samp(rank_u, other_rankings, alternative='greater')
            ts_f, pf = stats.ttest_1samp(rank_f, other_rankings, alternative='greater')

        print(f'User: {ts_u}, {pu}')

        if ts_f is not None:
            print(f'Flat: {ts_f}, {pf}')

    fname = graph_fname.replace('selected_graphs', 'ranking_score')
    with open(fname, 'wb') as f:
        pickle.dump(rankings, f)

    # iter reviews and find number of similar items


if __name__ == '__main__':
    args = parser.parse_args()
    method_kwargs = eval(args.method_kwargs)
    run(args.path, args.dataset, args.method, at=args.at, method_kwargs=method_kwargs)
