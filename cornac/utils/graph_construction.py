import re
from collections import OrderedDict, defaultdict

import numpy as np
from functools import lru_cache


def stem(sentiment):
    from gensim.parsing import stem_text
    ao_preprocess_fn = lambda x: stem_text(re.sub(r'--+.*|-+$', '', x))
    import random
    random.seed(42)
    a_id_new = {i: ao_preprocess_fn(e) for e, i in sentiment.aspect_id_map.items()}
    o_id_new = {i: ao_preprocess_fn(e) for e, i in sentiment.opinion_id_map.items()}
    a_id = {e: i for i, e in enumerate(sorted(set(a_id_new.values())))}
    o_id = {e: i for i, e in enumerate(sorted(set(o_id_new.values())))}
    a_o_n = {i: a_id[e] for i, e in a_id_new.items()}
    o_o_n = {i: o_id[e] for i, e in o_id_new.items()}

    s = OrderedDict()
    for i, aos in sentiment.sentiment.items():
        s[i] = [(a_o_n[a], o_o_n[o], s) for a, o, s in aos]

    return s, a_o_n, o_o_n

@lru_cache()
def generate_mappings(sentiment, match, get_ao_mappings=False, get_sent_edge_mappings=False):
    # Initialize all variables
    sent, a_mapping, o_mapping = stem(sentiment)
    # sent = sentiment.sentiment
    aos_user = defaultdict(list)
    aos_item = defaultdict(list)
    aos_sent = defaultdict(list)
    user_aos = defaultdict(list)
    item_aos = defaultdict(list)
    sent_aos = defaultdict(list)
    user_sent_edge_map = dict()
    item_sent_edge_map = dict()

    # Iterate over all sentiment triples and create the corresponding mapping for users and items.
    edge_id = -1
    for uid, isid in sentiment.user_sentiment.items():
        for iid, sid in isid.items():
            user_sent_edge_map[(sid, uid)] = (edge_id := edge_id + 1)  # assign and increment
            item_sent_edge_map[(sid, iid)] = (edge_id := edge_id + 1)
            for a, o, s in sent[sid]:
                if match == 'aos':
                    element = (a, o, s)
                elif match == 'a':
                    element = a
                elif match == 'as':
                    element = (a, s)
                elif match == 'ao':
                    element = (a, o)
                else:
                    raise NotImplementedError

                aos_user[element].append(uid)
                aos_item[element].append(iid)
                aos_sent[element].append(sid)
                user_aos[uid].append(element)
                item_aos[iid].append(element)
                sent_aos[sid].append(element)

    return_data = [aos_user, aos_item, aos_sent, user_aos, item_aos, sent_aos]

    if get_ao_mappings:
        return_data.extend([a_mapping, o_mapping])

    if get_sent_edge_mappings:
        return_data.extend([user_sent_edge_map, item_sent_edge_map])

    return tuple(return_data)


def create_heterogeneous_graph(train_set, bipartite=True):
    """
    Create a graph with users, items, aspects and opinions.
    Parameters
    ----------
    train_set : Dataset
    bipartite: if false have a different edge type per rating; otherwise, only use interacted.

    Returns
    -------
    DGLGraph
        A graph with edata type, label and an initialized attention of 1/k.
    int
        Num nodes in graph.
    int
        Number of items in dataset.
    int
        Number of relations in dataset.
    """
    import dgl
    import torch

    edge_types = {
        'mentions': [],
        'described_as': [],
        'has_opinion': [],
        'co-occur': [],
    }

    rating_types = set()
    for indices in list(zip(*train_set.matrix.nonzero())):
        rating_types.add(train_set.matrix[indices])

    if not bipartite:
        train_types = []
        for rt in rating_types:
            edge_types[str(rt)] = []
            train_types.append(str(rt))
    else:
        train_types = ['interacted']
        edge_types['interacted'] = []

    sentiment_modality = train_set.sentiment
    n_users = len(train_set.uid_map)
    n_items = len(train_set.iid_map)
    n_aspects = len(sentiment_modality.aspect_id_map)
    n_opinions = len(sentiment_modality.opinion_id_map)
    n_nodes = n_users + n_items + n_aspects + n_opinions

    # Create all the edges: (item, described_as, aspect), (item, has_opinion, opinion), (user, mentions, aspect),
    # (aspect, cooccur, opinion), and (user, 'rating', item). Note rating is on a scale.
    for org_uid, isid in sentiment_modality.user_sentiment.items():
        uid = org_uid + n_items
        for iid, sid in isid.items():
            for aid, oid, _ in sentiment_modality.sentiment[sid]:
                aid += n_items + n_users
                oid += n_items + n_users + n_aspects

                edge_types['mentions'].append([uid, aid])
                edge_types['mentions'].append([uid, oid])
                edge_types['described_as'].append([iid, aid])
                edge_types['described_as'].append([iid, oid])
                edge_types['co-occur'].append([aid, oid])

            if not bipartite:
                edge_types[str(train_set.matrix[(org_uid, iid)])].append([uid, iid])
            else:
                edge_types['interacted'].append([uid, iid])

    # Create reverse edges.
    reverse = {}
    for etype, edges in edge_types.items():
        reverse['r_' + etype] = [[t, h] for h, t in edges]

    # edge_types.update(reverse)
    n_relations = len(edge_types)
    edges = [[h, t] for k in sorted(edge_types) for h, t in edge_types.get(k)]
    edges_t = torch.LongTensor(edges).unique(dim=0).T

    et_id_map = {et: i for i, et in enumerate(sorted(edge_types))}

    g = dgl.graph((torch.cat(edges_t[0], edges_t[1]), torch.cat(edges_t[1], edges_t[0])), num_nodes=n_nodes)
    inverse_et = {v: k for k, l in edge_types.items() for v in l}
    et = torch.LongTensor([et_id_map[inverse_et[v]] for v in edges_t.T.tolist()])

    # Return 0 if not a rating type, else if using actual ratings return values else return 1 (bipartite).
    value_fn = lambda etype: 0 if etype not in train_types else (float(etype) if etype != 'interacted' else 1)
    labels = torch.FloatTensor([value_fn(inverse_et[v]) for v in edges_t.T.tolist()])

    g.edata['type'] = torch.cat([et, et + n_relations])

    g.edata['label'] = torch.cat([labels, labels])
    g.edata['a'] = dgl.ops.edge_softmax(g, torch.ones_like(g.edata['label']))

    return g, n_nodes, n_items, n_relations * 2