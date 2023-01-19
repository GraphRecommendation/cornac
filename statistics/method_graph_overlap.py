import itertools
import math
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from tqdm import tqdm

from cornac.models.lightrla.dgl_utils import extract_attention
from statistics import utils


def draw_test(edges):
    aoss = set()
    for a, b in edges:
        if isinstance(a, tuple):
            aoss.add(a)
        else:
            aoss.add(b)

    aos_id = {aos: i for i, aos in enumerate(sorted(aoss))}
    length = len(aos_id)

    tmp = []
    for a, b in edges:
        if isinstance(a, tuple):
            tmp.append((aos_id[a], b+length))
        else:
            tmp.append((a+length, aos_id[b]))

    G = nx.DiGraph()
    G.add_edges_from(tmp)
    for layer, nodes in enumerate(nx.topological_generations(G)):
        # `multipartite_layout` expects the layer as a node attribute, so add the
        # numeric layer value as a node attribute
        for node in nodes:
            G.nodes[node]["layer"] = layer
    pos = nx.multipartite_layout(G, subset_key="layer")
    # pos = nx.spring_layout(G)
    nx.draw(G, pos, arrowstyle='->', node_size=20)
    plt.show()
    plt.savefig("filename.png")


def prune_edges(edges, eval_method, sources):
    hop_edge = defaultdict(list)
    for s, d, h in edges:
        hop_edge[h].append((s,d))

    new_edges = set()
    new_edges_test = defaultdict(list)
    marked = sources
    for h in reversed(sorted(hop_edge)):
        ne = {e for e in hop_edge[h] if e[1] in marked}
        new_edges.update(ne)
        new_edges_test[h].extend(ne)
        marked = {e[0] for e in ne}

    # Debug purposes
    # id_aspect_map = {v: k for k, v in eval_method.sentiment.aspect_id_map.items()}
    # id_opinion_map = {v: k for k, v in eval_method.sentiment.opinion_id_map.items()}
    # transform_fn = lambda x, y: ((id_aspect_map[x[0]], id_opinion_map[x[1]], x[2]), y) \
    #     if isinstance(x, tuple) else \
    #     (x, (id_aspect_map[y[0]], id_opinion_map[y[1]], y[2]))
    # new_edges_test2 = {h: [transform_fn(x, y) for x, y in e] for h, e in new_edges_test.items()}

    return new_edges_test


def all_paths(eval_method, user, item):
    user_aos = {u: {aos for sid in sids.values() for aos in eval_method.sentiment.sentiment[sid]}
                for u, sids in eval_method.sentiment.user_sentiment.items()}
    item_aos = {i: {aos for sid in sids.values() for aos in eval_method.sentiment.sentiment[sid]}
                for i, sids in eval_method.sentiment.item_sentiment.items()}
    aos_user = defaultdict(set)
    for u, sids in eval_method.sentiment.user_sentiment.items():
        for sid in sids.values():
            for aos in eval_method.sentiment.sentiment[sid]:
                aos_user[aos].add(u)

    dest_aos = item_aos[item]
    frontier = {user}
    seen = set()
    edges = set()
    hops = 0
    while True:
        next_set = set()
        seen.update(frontier)
        for u in frontier:
            next_set.update(user_aos[u])
            edges.update({(u, a, hops) for a in user_aos[u]})
        if next_set.intersection(dest_aos):
            break
        else:
            hops += 1

        frontier = set()
        for aos in next_set:
            users = aos_user[aos].difference(seen)
            frontier.update(users)
            edges.update({(aos,  u, hops) for u in users})

        hops += 1

    edges = prune_edges(edges, eval_method, dest_aos)
    # draw_test(edges)

    return hops, edges


def limit_edges_to_best_user(edges, model, eval_method, user):
    # Get user similarity.
    # A user is always two hops away. start user -> aos -> another user
    num_hops = len(edges) // 2
    for h in range(num_hops):
        h = h * 2 + 1
        with torch.no_grad():
            users = torch.LongTensor([u for _, u in edges[h]]).to(model.device)
            u_sim = model.model.predict(user + eval_method.train_set.num_items, users + eval_method.train_set.num_items)
            best = users[torch.argmax(u_sim)].cpu().item()

        # Prune edges based on best user
        edges[h] = [e for e in edges[h] if e[1] == best]
        delta = 1
        while h - delta >= 0 or h + delta < len(edges):
            if (index := h - delta) >= 0:
                dst = {e[0] for e in edges[index+1]}
                edges[index] = [e for e in edges[index] if e[1] in dst]

            if (index := h + delta) < len(edges):
                src = {e[1] for e in edges[index-1]}
                edges[index] = [e for e in edges[index] if e[0] in src]
            delta += 1

    return edges

def get_intersecting_reviews():
    pass

def lightrla_overlap(eval_method, model, user, item, hackjob=True):
    # Get paths
    hops, edges = all_paths(eval_method, user, item)
    aos_user_review = defaultdict(lambda: defaultdict(list))
    for uid, isid in eval_method.sentiment.user_sentiment.items():
        for iid, sid in isid.items():
            for aos in eval_method.sentiment.sentiment[sid]:
                aos_user_review[aos][uid].append(sid)

    edges = limit_edges_to_best_user(edges, model, eval_method, user)

    # Get review attention.
    if hackjob:
        with torch.no_grad():
            attention = extract_attention(model.model, model.node_review_graph, model.device)


    else:
        raise NotImplementedError

    # Get reviews intersecting selected paths through users.
    hop_sid = defaultdict(set)
    review_attention = defaultdict(lambda: defaultdict(dict))
    for h, es in edges.items():
        uids = [e[h % 2] for e in es]
        aoss = [e[1 - (h % 2)] for e in es]
        for uid, aos in zip(uids, aoss):
            found = set(aos_user_review[aos][uid])
            hop_sid[h] = hop_sid[h].union(found)

            eids = model.node_review_graph.edge_ids(torch.LongTensor(list(found)), uid + eval_method.train_set.num_items)
            att = attention[eids]
            for sid, a in zip(found, att):
                review_attention[uid][h][sid] = max(a).cpu().item()

    # Select reviews
    review_attention2 = {i: {h: sorted(scores, key=scores.get)[0] for h, scores in hs.items()}
                         for i, hs in review_attention.items()}

    # Def get review triples
    num_items = eval_method.train_set.num_items
    num_users = eval_method.train_set.num_users
    num_aspects = eval_method.sentiment.num_aspects
    sid_ui = {sid: (uid+num_items, iid) for uid, isid in eval_method.sentiment.user_sentiment.items()
              for iid, sid in isid.items()}
    edges = []
    for uid, elements in review_attention2.items():
        for _, sid in elements.items():
            edge = set()
            edge.update(set(sid_ui[sid]))
            for a, o, s in eval_method.sentiment.sentiment[sid]:
                edge.add(a+num_users+num_items)
                edge.add(o+num_users+num_items+num_aspects)

            edges.extend(list(itertools.combinations(edge, r=2)))

    G = nx.Graph()
    G.add_edges_from(edges)
    color_map = []
    for node in G:
        if node == user + num_items:
            color_map.append('purple')
        elif node == item:
            color_map.append('silver')
        elif node < num_items:
            color_map.append('blue')
        elif node < num_items + num_users:
            color_map.append('green')
        elif node < num_items + num_users + num_aspects:
            color_map.append('yellow')
        else:
            color_map.append('red')
    labels = {}
    id_aspect_map = {v: k for k, v in eval_method.sentiment.aspect_id_map.items()}
    id_opinion_map = {v: k for k, v in eval_method.sentiment.opinion_id_map.items()}
    for node in G:
        if node < num_items + num_users:
            labels[node] = str(node)
        elif node < num_items + num_users + num_aspects:
            labels[node] = id_aspect_map[node - num_items - num_users]
        else:
            labels[node] = id_opinion_map[node - num_items - num_users - num_aspects]
    pos = nx.spring_layout(G, k=5/math.sqrt(G.order()))
    nx.draw(G, node_color=color_map, labels=labels, with_labels=True, pos=pos)
    plt.show()

    pass


def run(path, dataset, method):
    eval_method = utils.initialize_dataset(dataset)
    model = utils.initialize_model(path, dataset, method)

    # Iter test
    for user, item in tqdm(list(zip(*eval_method.test_set.csr_matrix.nonzero()))):
        review_aos = set()
        if method == 'lightrla':
            out = lightrla_overlap(eval_method, model, user, item)
        else:
            raise NotImplementedError
    # Get paths

    pass


if __name__ == '__main__':
    path, dataset, method = sys.argv[1:]
    run(path, dataset, method)