import math
import pickle
import time
from concurrent.futures import as_completed
from functools import partial

import networkx as nx
import numpy as np
from concurrent.futures.process import ProcessPoolExecutor
from tqdm import tqdm

from statistics.utils import id_mapping


def load_data(fname):
    with open(fname, 'rb')as f:
        d = pickle.load(f)
    return d


def get_path(inner_uid, iid, g, g_simple, num_nodes, path_limit=None):

    # Get only paths using the shortest path length.
    # Max path length is three as the graph shown in figure 4 in the
    # KGAT paper: https://arxiv.org/pdf/1905.07854.pdf
    max_length = max(nx.shortest_path_length(g, inner_uid, iid), 3)
    paths = list(nx.all_simple_paths(g_simple, inner_uid, iid, max_length))

    # weight paths
    p_w = {}
    for i, p in enumerate(paths):
        p_w[i] = nx.path_weight(g, p, weight='a') / len(p)  # Longer paths are not better

    # shortests_paths = nx.all_shortest_paths(g, inner_uid, iid, weight='a')

    index = -1
    node_set = set()
    # paths = []  # sort according to weight
    paths = [p for _, p in sorted(enumerate(paths), key=lambda x: p_w[x[0]])]  # sort according to weight

    # while len(node_set) < num_nodes and (next_path := next(shortests_paths, None)) is not None:
    while len(node_set) < num_nodes and index+1 < len(paths):
        index += 1
        node_set.update(paths[index])

        # node_set.update(next_path)
        # paths.append(next_path)
        if path_limit is not None and index + 1 >= path_limit:
            break

    total_nodes = len(node_set)
    lower_dist = abs(total_nodes - num_nodes - len(paths[index]))
    upper_dist = total_nodes - num_nodes

    # Get paths such that number of nodes are closer to number of nodes of LightRLA.
    # If equal number of nodes, select where fewer edges, meaning fewer paths.
    if lower_dist <= upper_dist:
        paths = paths[:index]
    else:
        paths = paths[:index+1]

    edges = []
    for path in paths:
        for i in range(len(path)-1):
            src, dst = path[i], path[i+1]
            es = g.adj[src][dst]
            edge = sorted(es, key=lambda x: es[x]['a'])[0]
            edges.append((src, dst, {'weight': es[edge]['a']}))

    return edges


def _wrapper(users, inner_users, items, graphs, path_methodology, m_length, *args):
    results = []

    for uid, inner_uid, iid, lrla_g in zip(users, inner_users, items, graphs):
        # Number of selected nodes can be biased towards different things.
        # E.g. same as our method or same as train graphs
        if path_methodology == 'lightrla':
            num_nodes = lrla_g.number_of_nodes()
        elif path_methodology == 'mean':
            num_nodes = m_length
        else:
            raise NotImplementedError
        results.append([uid, iid, get_path(inner_uid, iid, *args, num_nodes=num_nodes)])

    return results

def get_reviews(eval_method, model, lightrla_data, path_methodology):
    g = model.train_graph.to_networkx()
    attention = model.train_graph.edata['a']
    attention = {e: 1-v for e, v in zip(g.edges, attention.numpy())}  # attention is inversed so lower is better

    # Get average number of nodes in train graphs
    lengths = [len({e for aos in eval_method.sentiment.sentiment[sid] for e in aos[:2]})
               for iaos in eval_method.sentiment.user_sentiment.values()
               for sid in iaos.values()]
    m_length = np.median(lengths) + 2  # Add two to account for user/item
    nx.set_edge_attributes(g, attention, 'a')
    g_simple = nx.Graph(g)
    uig = []
    futures =[]
    with ProcessPoolExecutor(max_workers=3) as ppe:
        bz = 10
        for ndx in range(0, len(lightrla_data), bz):
            batch = lightrla_data[ndx:ndx + bz]
            users, items, graphs = zip(*batch)
            inner_users = list(map(lambda e: id_mapping(eval_method=eval_method, type='u', eid=e), users))
            futures.append(ppe.submit(_wrapper, users, inner_users, items, graphs,
                                      path_methodology, m_length, g, g_simple))

        for _ in tqdm(as_completed(futures), total=len(futures)):
            pass

    for future in futures:
        for uid, iid, edges in future.result():
            rec_graph = nx.MultiGraph()
            rec_graph.add_edges_from(edges)

            # tmp
            # elabels = {}
            # for src, dst in rec_graph.edges():
            #     es = rec_graph.adj[src][dst]
            #     e = sorted(es, key=lambda x: es[x]['weight'])[0]
            #     elabels[(src, dst)] = f"{es[e]['weight']:.5f}"

            # tg = nx.Graph(rec_graph)
            # import matplotlib.pyplot as plt
            # pos = nx.spring_layout(tg)
            # nx.draw(tg, pos=pos, labels={n: str(n) for n in tg.nodes()})
            # nx.draw_networkx_edge_labels(tg, pos, edge_labels=elabels)
            # plt.show()

            uig.append((uid, iid, rec_graph))

    return uig