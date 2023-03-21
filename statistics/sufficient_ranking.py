import argparse

import networkx as nx
import numpy as np
import pickle

from collections import Counter
from scipy import stats
from tqdm import tqdm

from statistics import utils, lightrla_graph_overlap, kgat_graph_overlap

parser = argparse.ArgumentParser()
parser.add_argument('path', type=str)
parser.add_argument('dataset')
parser.add_argument('method')
parser.add_argument('--at', default=20, type=int)
parser.add_argument('--method_kwargs', default="{'matching_method':'a'}", type=str)

def run(path, dataset, method, at, method_kwargs):
    eval_method = utils.initialize_dataset(dataset)
    model = utils.initialize_model(path, dataset, method)
    matching_method = method_kwargs['matching_method']
    review_fname, graph_fname = utils.get_method_paths(method_kwargs, dataset, method)
    aos_user, aos_item, _, user_aos, item_aos, sent_aos, a_mapping, o_mapping = \
        utils.generate_mappings(eval_method.sentiment, matching_method, get_ao_mappings=True)

    rid_sid = {rid: eval_method.sentiment.user_sentiment[uid][iid]
               for uid, irid in eval_method.review_text.user_review.items()
               for iid, rid in irid.items()}
    rid_iid = {rid: iid for uid, irid in eval_method.review_text.user_review.items() for iid, rid in irid.items()}
    rid_aos = {rid: sent_aos[sid] for rid, sid in rid_sid.items()}

    most_pop_c = Counter(eval_method.train_set.matrix.nonzero()[1])
    most_pop = sorted(most_pop_c)
    mp_scores = [most_pop_c.get(m) for m in most_pop]
    mp_sort = np.argsort(mp_scores)
    mp_ranks = np.argsort(mp_sort)

    # item to aos set
    found = 0
    rankings = []
    mp_rankings = []
    with tqdm(list(eval_method.test_set.user_data.keys())) as progressbar:
        rsum = 0
        rlen = 0
        mrsum = 0
        mrlen = 0
        m = eval_method.sentiment.num_aspects
        if method == 'kgat':
            m = utils.id_mapping(eval_method, eval_method.sentiment.num_opinions, 'o') - eval_method.train_set.num_items
            g = model.train_graph.to_networkx()
            attention = model.train_graph.edata['a']
            attention = {e: 1-v for e, v in zip(g.edges, attention.numpy())}  # attention is inversed so lower is better

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

                    # s_aos = set(sent_aos[sids[-1]])
                    s_aos = set(a for s in sids[-1:] for a in sent_aos[s])
                    diversity.update(s_aos)
                    matched = [item for item, i_aos in item_aos.items() if item != iid and
                               s_aos.issubset(i_aos) and item not in rated]
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

                    matched = list(i for i in matched if i < len(eval_method.global_iid_map) and i != iid
                                   and i not in rated)

                ir = ranks[matched]
                mir = mp_ranks[matched]
                # diversity.update(matched)
                if len(ir) > 0:
                    rsum += sum(ir)
                    rlen += len(ir)
                    mrsum += sum(mir)
                    rankings.append((uid, iid, ir))
                    mp_rankings.append((uid, iid, mir))
                    found += 1
                else:
                    pass

            if rlen:
                progressbar.set_description(f'AR:{rsum/rlen:.3f},MAR:{mrsum/rlen:.3f},'
                                            f'PF:{found/((progressbar.n+1)*at):.3f},'
                                            f'AN:{rlen/((progressbar.n+1)*at):.3f},'
                                            f'D:{len(diversity)/m:.3f}')

    # Test towards random, i.e., better than the mean.
    rank_u = np.array([np.mean(r) for _,_, r in rankings])
    rank_f = np.array([e for _,_, r in rankings for e in r])
    ts_u, pu = stats.ttest_1samp(rank_u, len(eval_method.global_iid_map) / 2, alternative='greater')
    ts_f, pf = stats.ttest_1samp(rank_f, len(eval_method.global_iid_map) / 2, alternative='greater')
    quantiles_u = np.quantile(rank_u, [.01,.05, .1, .9,.95,.99])
    quantiles_f = np.quantile(rank_f, [.01,.05, .1, .9,.95,.99])

    print(f'User: {ts_u}, {pu}, {quantiles_u}')
    print(f'Flat: {ts_f}, {pf}, {quantiles_f}')

    fname = graph_fname.replace('selected_graphs', 'ranking_score')
    with open(fname, 'wb') as f:
        pickle.dump(rankings, f)

    # iter reviews and find number of similar items

if __name__ == '__main__':
    args = parser.parse_args()
    args = parser.parse_args()
    method_kwargs = eval(args.method_kwargs)
    run(args.path, args.dataset, args.method, at=args.at, method_kwargs=method_kwargs)
