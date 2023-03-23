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
from statistics.utils import id_mapping, generate_mappings


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


def reverse_match(user, item, sentiment, match='a'):
    aos_user, aos_item, _, user_aos, item_aos, _ = generate_mappings(sentiment, match)

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
    aos_user, aos_item, _, user_aos, item_aos, sent_aos = generate_mappings(sentiment, match)  # Get mappings

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


def _get_sids(eval_method, nid, aoss, is_user, aos_sent):
    mapping = eval_method.sentiment.user_sentiment if is_user else eval_method.sentiment.item_sentiment
    sids = {aos: [sid for _, sid in mapping[nid].items() if sid in aos_sent[aos]] for aos in aoss }
    sids = OrderedDict(sids)
    return sids


def _create_edges(eval_method, match, nid, nid_inner, aos_sids, atts, agg, is_user, usem, isem):
    edge_map = usem if is_user else isem
    # Get attention and aggregate across heads.
    atts = {aos: {sid: agg(atts[edge_map[(sid, nid_inner)]]) for sid in sids}
            for aos, sids in aos_sids.items()}

    # 1-att for shortest path but is highest weight.
    data = [(nid, id_mapping(eval_method, aos, match), {'weight': 1-atts[aos][sid], 'sid': sid})
            for aos, sids in aos_sids.items() for sid in sids]
    return data


def get_reviews_nwx(eval_method, model, edges, match, hackjob=True, methodology='weighted', weighting='attention',
                    aggregator=np.mean):
    aos_user, aos_item, aos_sent, user_aos, item_aos, sent_aos, user_sent_edge_map, item_sent_edge_map = \
        generate_mappings(eval_method.sentiment, match, get_sent_edge_mappings=True)  # Get mappings
    e_length = len(edges)
    n_items = eval_method.train_set.num_items
    n_users_items = n_items + eval_method.train_set.num_users

    # Get review attention.
    if hackjob:
        with torch.no_grad():
            attention = extract_attention(model.model, model.node_review_graph, model.device).cpu().numpy()
    else:
        raise NotImplementedError

    # No solution when no edges were found.
    if edges is None:
        raise NotImplementedError

    # Construct nx graph
    # goes backwards
    data = []
    for h in range(0, e_length, 2):
        aoss = sorted({aos for aos, _ in edges[h]})
        dsts = sorted({dst for (_, dst) in edges[h]})
        srcs = sorted({src for (src, _) in edges[h+1]})
        pairs = list(itertools.product(srcs, dsts))
        for src, dst in pairs:  # naive path
            if h != 0:
                dst_inner = dst - n_items
            else:
                dst_inner = dst

            src_inner = src - n_items

            dst_sids = _get_sids(eval_method, dst_inner, aoss, is_user=h != 0, aos_sent=aos_sent)  # only an item on the first hop
            src_sids = _get_sids(eval_method, src_inner, aoss, is_user=True, aos_sent=aos_sent)
            if h == 0:
                # we know h == 0
                data.extend(_create_edges(eval_method, match, dst, dst_inner, dst_sids, attention, aggregator, False,
                                          user_sent_edge_map, item_sent_edge_map))
            # elif h % 4 == 0:
            #     pass
            else:
                data.extend(_create_edges(eval_method, match, dst, dst_inner, dst_sids, attention, aggregator, True,
                                          user_sent_edge_map, item_sent_edge_map))

            # Source is always a user
            data.extend(_create_edges(eval_method, match, src, src_inner, src_sids, attention, aggregator, True,
                                      user_sent_edge_map, item_sent_edge_map))

    # assign weights and edge identifiers
    # if attention use attention, if similarity, assign user similarity as weight

    # create graph
    g = nx.MultiGraph()
    g.add_edges_from(data)

    # Get user and item
    _, item = next(iter(edges[0]))
    user, _ = next(iter(edges[max(edges)]))

    # Weighted is the best path taking both user, item, and intermediary paths into account.
    # Greedy always takes the best first, while still ensuring it is the shortest (unweighted) path.
    if methodology == 'weighted':
        path = nx.shortest_path(g, source=user, target=item, weight='weight')
    elif methodology in ['greedy_user', 'greedy_item']:
        # Get source and target
        if methodology == 'greedy_user':
            source, target = user, item
        else:
            source, target = item, user

        # get possible paths
        cur_length = e_length
        paths = list(nx.all_shortest_paths(g, source, target))

        path = []  # is the actual path selected
        while len(paths) > 1:
            cur_node = next(iter(paths))[0]  # node is always the first
            path.append(cur_node)
            dst_nodes = [p[1] for p in paths]  # get all possible 'next node'

            # Get best weight for next nodes
            p_w = {}
            for i, dst in enumerate(dst_nodes):
                info = g[cur_node][dst]
                e = sorted(info, key=lambda x: info[x]['weight'])[0]
                p_w[i] = info[e]['weight']

            # Select best
            p = sorted(p_w, key=p_w.get)[0]

            # Get remaining paths given selected. Note, not efficient, but fine for one run.
            cur_length -= 1
            paths = list(nx.all_shortest_paths(g, source=dst_nodes[p], target=target))
            if not isinstance(paths[0], list):
                paths = [paths]

        path.extend(list(paths.pop()))
        # assert path == nx.shortest_path(g, source=user, target=item, weight='weight')
    else:
        raise NotImplementedError

    p_g = nx.path_graph(path)
    sids = []
    _weights = []
    for src, dst in p_g.edges():
        options = {e['sid']: e['weight'] for e in g.adj[src][dst].values()}
        sids.append(sorted(options, key=options.get)[0])
        _weights.append(options[sids[-1]])

    #  return selected reviews.
    return sids


def get_reviews(eval_method, model, edges, match, hackjob=True):
    aos_user, aos_item, aos_sent, user_aos, item_aos, sent_aos = generate_mappings(eval_method.sentiment, match)  # Get mappings

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
    aos_user, aos_item, _, user_aos, item_aos, _ = generate_mappings(sentiment, match)
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


@lru_cache()
def sid_to_rid_mapping(eval_method):
    sid_rid_map = OrderedDict()
    for user, isid in eval_method.sentiment.user_sentiment.items():
        for item, sid in isid.items():
            sid_rid_map[sid] = eval_method.review_text.user_review[user][item]

    rid_sid_map = {v: k for k, v in sid_rid_map.items()}
    return sid_rid_map, rid_sid_map


def sid_to_graphs(eval_method, uis, match):
    aos_user, aos_item, aos_sent, user_aos, item_aos, sent_aos, a_mapping, o_mapping\
        = generate_mappings(eval_method.sentiment, match, True)
    sid_user_item_mapping = {sid: (uid, iid) for uid, isid in eval_method.sentiment.user_sentiment.items()
                for iid, sid in isid.items()}
    # item_sid = {eval_method.sentiment.item_sentiment[item].values()}


    uig = []
    for uid, iid, sids in uis:
        edges = list()
        for sid in sids:
            for (a, o, s) in eval_method.sentiment.sentiment[sid]:
                inner_uid, inner_iid = sid_user_item_mapping[sid]
                a, o = id_mapping(eval_method, a_mapping[a], 'a'), id_mapping(eval_method, o_mapping[o], 'o')
                inner_uid = id_mapping(eval_method, inner_uid, 'u')
                edges.append((inner_uid, a, {'sentiment': 0}))
                edges.append((a, o, {'sentiment': s}))
                edges.append((inner_iid, a, {'sentiment': 0}))

        g = nx.MultiGraph()
        g.add_edges_from(edges)
        uig.append((uid, iid, g))

    return uig


def draw_reviews(eval_method, sids, user, item, match):
    aos_user, aos_item, aos_sent, user_aos, item_aos, sent_aos, a_mapping, o_mapping\
        = generate_mappings(eval_method.sentiment, match, True)
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