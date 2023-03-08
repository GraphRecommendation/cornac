import itertools
import math
import re
from collections import defaultdict, OrderedDict
from functools import lru_cache

import networkx as nx
import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from cornac.models.lightrla.dgl_utils import extract_attention


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

@lru_cache()
def dicts(sentiment, match, get_ao_mappings=False):
    # Initialize all variables
    sent, a_mapping, o_mapping = stem(sentiment)
    # sent = sentiment.sentiment
    aos_user = defaultdict(list)
    aos_item = defaultdict(list)
    aos_sent = defaultdict(list)
    user_aos = defaultdict(list)
    item_aos = defaultdict(list)
    sent_aos = defaultdict(list)

    # Iterate over all sentiment triples and create the corresponding mapping for users and items.
    for uid, isid in sentiment.user_sentiment.items():
        for iid, sid in isid.items():
            for a, o, s in sent[sid]:
                if match == 'aos':
                    element = (a, o, s)
                elif match == 'a':
                    element = a
                else:
                    raise NotImplementedError

                aos_user[element].append(uid)
                aos_item[element].append(iid)
                aos_sent[element].append(sid)
                user_aos[uid].append(element)
                item_aos[iid].append(element)
                sent_aos[sid].append(element)

    if not get_ao_mappings:
        return aos_user, aos_item, aos_sent, user_aos, item_aos, sent_aos
    else:
        return aos_user, aos_item, aos_sent, user_aos, item_aos, sent_aos, a_mapping, o_mapping


def stem(sentiment):
    from gensim.parsing import stem_text
    ao_preprocess_fn = lambda x: stem_text(re.sub(r'--+.*|-+$', '', x))
    a_id_new = {i: ao_preprocess_fn(e) for e, i in sentiment.aspect_id_map.items()}
    o_id_new = {i: ao_preprocess_fn(e) for e, i in sentiment.opinion_id_map.items()}
    a_id = {e: i for i, e in enumerate(set(a_id_new.values()))}
    o_id = {e: i for i, e in enumerate(set(o_id_new.values()))}
    a_o_n = {i: a_id[e] for i, e in a_id_new.items()}
    o_o_n = {i: o_id[e] for i, e in o_id_new.items()}

    s = OrderedDict()
    for i, aos in sentiment.sentiment.items():
        s[i] = [(a_o_n[a], o_o_n[o], s) for a, o, s in aos]

    return s, a_o_n, o_o_n


def reverse_match(user, item, sentiment, match='a'):
    aos_user, aos_item, _, user_aos, item_aos, _ = dicts(sentiment, match)

    dst = user_aos[user]
    src = item_aos[item]
    lu, li = len(sentiment.user_sentiment[user]), len(sentiment.item_sentiment[item])
    # Direct connection to item
    if any([s in dst for s in src]):
        return 0, lu, li

    # Is it connected to a one hop user?
    elements = {e for uid in sentiment.item_sentiment[item].keys() for e in user_aos[uid]}

    if any([e in dst for e in elements]):
        return 1, lu, li
    else:
        return 2, lu, li


def reverse_path(eval_method, user, item, match):
    sentiment = eval_method.sentiment
    num_items = eval_method.train_set.num_items
    aos_user, aos_item, _, user_aos, item_aos, _ = dicts(sentiment, match)  # Get mappings

    # Get one hop entities
    dst = user_aos[user]
    src = item_aos[item]

    edges = defaultdict(set)
    edges[0].update((s, item) for s in src)
    index = 1

    # Direct connection to item, i.e., user has mentioned entities which is used to describe item
    if any([s in dst for s in src]):
        edges[index].update((user+num_items, d) for d in dst if d in src)  # Add user edges
        edges[0] = {e for e in edges[0] if e[0] in dst}  # prune
        return edges

    # Get one hop users. I.e., users sharing an entity with the item.
    # Get users that rated item and extract edges to the item's entities.
    edges[index].update([(uid+num_items, e) for uid in sentiment.item_sentiment[item].keys()
                  for e in user_aos[uid] if e in src])
    index += 1
    # Get edges to next layer of entities.
    edges[index].update([(e, uid) for (uid, _) in edges[index - 1] for e in user_aos[uid-num_items]])
    index += 1

    # Extract most distant entities.
    ndst = {e[0] for e in edges[index - 1]}

    # Check if any are shared with user
    if any(d in ndst for d in dst):
        # Get edges to entities
        edges[index].update([(user+num_items, e) for (e, _) in edges[index - 1] if e in dst])

        # Prune edges.
        n_edges = defaultdict(set)
        cur_set = {user+num_items}
        for i in reversed(range(index+1)):
            cur_edges = {e for e in edges[i] if e[0] in cur_set}
            n_edges[i].update(cur_edges)
            cur_set = {e[1] for e in cur_edges}

        return n_edges
    else:
        return None


def get_reviews_nwx(eval_method, model, edges, match, hackjob=True, methodology='weighted', weighting='attention'):
    aos_user, aos_item, aos_sent, user_aos, item_aos, sent_aos = dicts(eval_method.sentiment, match)  # Get mappings

    # Get review attention.
    if hackjob:
        with torch.no_grad():
            attention = extract_attention(model.model, model.node_review_graph, model.device)
    else:
        raise NotImplementedError

    # No solution when no edges were found.
    if edges is None:
        raise NotImplementedError

    # Contruct nx graph

    # assign weights and edge identifiers
    # if attention use attention, if similarity, assign user similarity as weight

    # Methodology
    # If weighted find the shortest weighted path
    # If user, greedy selection from user
    # If item, greedy selection from item

    #  return selected reviews.
    return None


def get_reviews(eval_method, model, edges, match, hackjob=True):
    aos_user, aos_item, aos_sent, user_aos, item_aos, sent_aos = dicts(eval_method.sentiment, match)  # Get mappings

    # Get review attention.
    if hackjob:
        with torch.no_grad():
            attention = extract_attention(model.model, model.node_review_graph, model.device)
    else:
        raise NotImplementedError

    # No solution when no edges were found.
    if edges is None:
        raise NotImplementedError

    # Get user and item
    n_hops = max(edges)
    item = next(iter(edges[0]))[1]  # item first edge added and last element element.
    user = next(iter(edges[n_hops]))[0]  # user last edge added and first element.

    # Method for getting sentiment triple id.
    def get_sid(eid, aos_set, is_user=True):
        sentiments = eval_method.sentiment.user_sentiment[eid] if is_user else \
            eval_method.sentiment.item_sentiment[eid]
        sids = list({sid for _, sid in sentiments.items() if aos_set.intersection(sent_aos[sid])})
        arg = torch.argmax(torch.max(attention[sids], dim=1)[0]).cpu().item()
        return sids[arg]

    # todo: get item reviews and user reviews with matching AOS.
    num_items = eval_method.train_set.num_items
    if n_hops == 1:
        item_sid = get_sid(item, {e[0] for e in edges[0]}, False)
        user_sid = get_sid(user-num_items, {e[1] for e in edges[1]})
        return user_sid, item_sid

    # Get one hop users.
    users = list({u-num_items for u, _ in edges[1]})
    sids = [eval_method.sentiment.item_sentiment[item][uid] for uid in users]

    # Get user with the highest attention
    arg = torch.argmax(torch.max(attention[sids], dim=1)[0]).cpu().item()
    item_sid = sids[arg]
    hop_user = users[arg]+num_items

    # Function pruning edges after selecting a node.
    def limit_edges(es, start, position, start_set):
        increment = 0
        es[start] = {e for e in es[start] if e[position] in start_set}
        while True:
            increment += 1
            change = False
            if (cur_pos := start + increment) in es:
                cur_set = {e[0] for e in edges[cur_pos-1]}
                es[cur_pos] = {e for e in es[cur_pos] if e[1] in cur_set}
                change = True

            if (cur_pos := start - increment) in es:
                cur_set = {e[1] for e in edges[cur_pos+1]}
                es[cur_pos] = {e for e in es[cur_pos] if e[0] in cur_set}
                change = True

            if not change:
                break

        return es

    # Start at hop 1, first (0'th) element of tuple, with one hop user: hop_user.
    edges = limit_edges(edges, 1, 0, {hop_user})

    # Get user sentiment, other user sentiment, and item sentiment.
    user_sid = get_sid(user-num_items, {e[1] for e in edges[n_hops]}, True)
    other_sid = get_sid(hop_user-num_items, {e[1] for e in edges[n_hops]}, True)

    return user_sid, other_sid, item_sid


def all_dist(eval_method, match):
    sentiment = eval_method.sentiment
    aos_user, aos_item, _, user_aos, item_aos, _ = dicts(sentiment, match)
    dist = np.zeros((eval_method.train_set.num_items, eval_method.train_set.num_users))

    for item in tqdm(list(range(eval_method.train_set.num_items))):
        mask = np.ones((eval_method.train_set.num_users, ), )

        # Users directly connected to item by rating
        users = list(sentiment.item_sentiment[item].keys())
        mask[users] = 0

        # Users directly connected to item by element
        elements = item_aos[item]
        users = [u for e in elements for u in aos_user[e]]
        mask[users] = 0

        # Users connected with one hop
        aos = [e for uid in sentiment.item_sentiment[item].keys() for e in user_aos[uid]]
        users = [user for e in aos for user in aos_user[e]]
        dist[item][users] = 1  # Represent one hop
        mask[users] = 0
        dist[item][mask.astype(bool)] = 2  # At least two hops

    return dist


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


def draw_reviews(eval_method, sids, user, item, match):
    aos_user, aos_item, aos_sent, user_aos, item_aos, sent_aos, a_mapping, o_mapping\
        = dicts(eval_method.sentiment, match, True)
    num_items = eval_method.train_set.num_items
    num_users = eval_method.train_set.num_users
    num_aspects = eval_method.sentiment.num_aspects
    user_sid = {sid: (uid, iid) for uid, isid in eval_method.sentiment.user_sentiment.items()
                for iid, sid in isid.items()}
    item_sid = {eval_method.sentiment.item_sentiment[item].values()}

    def mapping(eid, type):
        if type == 'i':
            return eid
        elif type == 'u':
            return eid + num_items
        elif type == 'a':
            return eid + num_items + num_users
        else:
            return eid + num_items + num_users + num_aspects

    edges = set()
    for sid in sids:
        for (a, o, s) in eval_method.sentiment.sentiment[sid]:
            uid, iid = user_sid[sid]
            a, o = mapping(a_mapping[a], 'a'), mapping(o_mapping[o], 'o')
            uid = mapping(uid, 'u')
            edges.add((uid, a, 0))
            edges.add((a, o, s))
            edges.add((iid, a, 0))

    G = nx.Graph()
    for s, d, l in edges:
        G.add_edge(s, d)

    e_labels = {(s, d): l for s, d, l in edges if l != 0}

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
    id_aspect_map = {a_mapping[v]: k for k, v in reversed(eval_method.sentiment.aspect_id_map.items())}
    id_opinion_map = {o_mapping[v]: k for k, v in reversed(eval_method.sentiment.opinion_id_map.items())}
    for node in G:
        if node < num_items + num_users:
            labels[node] = str(node)
        elif node < num_items + num_users + num_aspects:
            labels[node] = id_aspect_map[node - num_items - num_users]
        else:
            labels[node] = id_opinion_map[node - num_items - num_users - num_aspects]
    pos = nx.spring_layout(G, k=2/math.sqrt(G.order()), iterations=20)
    nx.draw(G, node_color=color_map, labels=labels, with_labels=True, pos=pos)
    nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=e_labels)
    plt.show()