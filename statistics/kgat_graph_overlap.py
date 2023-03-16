import math
import pickle

import networkx as nx
from tqdm import tqdm

from statistics.utils import id_mapping


def load_data(fname):
    with open(fname, 'rb')as f:
        d = pickle.load(f)
    return d


def get_reviews(eval_method, model, lightrla_data):
    g = model.train_graph.to_networkx()
    attention = model.train_graph.edata['a']
    attention = {e: 1-v for e, v in zip(g.edges, attention.numpy())}  # attention is inversed so lower is better
    nx.set_edge_attributes(g, attention, 'a')
    g_simple = nx.Graph(g)
    uig = []
    for uid, iid, lrla_g in tqdm(lightrla_data, desc='Generating graphs'):
        num_nodes = lrla_g.number_of_nodes()
        inner_uid = id_mapping(eval_method, uid, 'u')

        # Get only paths using the shortest path length.
        # Max path length is three as the graph shown in figure 4 in the
        # KGAT paper: https://arxiv.org/pdf/1905.07854.pdf
        max_length = max(nx.shortest_path_length(g, inner_uid, iid), 3)
        paths = list(nx.all_simple_paths(g_simple, inner_uid, iid, max_length))

        # weight paths
        p_w = {}
        for i, p in enumerate(paths):
            p_w[i] = nx.path_weight(g, p, weight='a')


        index = -1
        node_set = set()
        paths = [p for _, p in sorted(enumerate(paths), key=lambda x: p_w[x[0]])]  # sort according to weight
        while len(node_set) < num_nodes and index+1 < len(paths):
            index += 1
            node_set.update(paths[index])

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