import sys
from collections import Counter, defaultdict

import numpy as np
from tqdm import tqdm
from statistics import utils


def get_aos(identifier, exclude, data, sentiment):
    a = set()
    o = set()
    ao = set()
    aos = set()
    for other, sid in data[identifier].items():
        if other != exclude:
            for triple in sentiment.sentiment[sid]:
                aos.add(triple)
                ao.add((triple[0], triple[1]))
                o.add((triple[1], triple[2]))
                a.add(triple[0])

    return a, o, ao, aos


def breadth_first_search(eval_method, user, item):
    user_aos = {u: {aos for sid in sids.values() for aos in eval_method.sentiment.sentiment[sid]}
                for u, sids in eval_method.sentiment.user_sentiment.items()}
    item_aos = {i: {aos for sid in sids.values() for aos in eval_method.sentiment.sentiment[sid]}
                for i, sids in eval_method.sentiment.item_sentiment.items()}
    aos_user = defaultdict(set)
    for u, sids in eval_method.sentiment.user_sentiment.items():
        for sid in sids.values():
            for aos in eval_method.sentiment.sentiment[sid]:
                aos_user[aos].add(u)

    user_item = {u: {i for i in sid.keys()} for u, sid in eval_method.sentiment.user_sentiment.items()}

    dest_aos = item_aos[item]
    frontier = {user}
    seen = set()
    not_found = True
    hops = 0
    while not_found:
        next_set = set()
        seen.update(frontier)
        for u in frontier:
            next_set.update(user_aos[u])
        if next_set.intersection(dest_aos):
            not_found = False
        else:
            hops += 1

        frontier = set()
        for aos in next_set:
            frontier.update(aos_user[aos].difference(seen))

    return hops


def run(dataset):
    eval_method = utils.initialize_dataset(dataset)

    # Convert raw data to ids:
    raw_data = {(eval_method.global_uid_map[u], eval_method.global_iid_map[i]):
                    [(eval_method.sentiment.aspect_id_map[a], eval_method.sentiment.opinion_id_map[o], s)
                     for a, o, s in aos if a in eval_method.sentiment.aspect_id_map
                     and o in eval_method.sentiment.opinion_id_map]
                for u, i, aos in eval_method.sentiment.raw_data if u in eval_method.global_uid_map
                and i in eval_method.global_iid_map}

    # Go through test set. Compare user and items aos triplets
    a_match = []
    o_match = []
    ao_match = []
    aos_match = []
    intersect = 0
    no_intersect = 0
    no_match = 0
    def similarity(set_a, set_b, set_c):
        sims = []
        sims.append(len(set_a.intersection(set_c)) / len(set_a.union(set_c)))
        sims.append(len(set_b.intersection(set_c)) / len(set_b.union(set_c)))
        tmp = set_a.intersection(set_b)
        sims.append(len(tmp.intersection(set_c)) / len(tmp.union(set_c)))
        return sims

    id_a = {v: k for k, v in eval_method.sentiment.aspect_id_map.items()}
    id_o = {v: k for k, v in eval_method.sentiment.opinion_id_map.items()}

    aos_c = Counter([aos for so in eval_method.sentiment.sentiment.values() for aos in so])
    shared = [any([aos_c[aos] > 1 for aos in so]) for so in eval_method.sentiment.sentiment.values()]

    num_hops = []

    for user, item in tqdm(list(zip(*eval_method.test_set.csr_matrix.nonzero()))):
        hops = breadth_first_search(eval_method, user, item)
        num_hops.append(hops)
        change = False
        if (aos := raw_data.get((user, item))) is not None and len(aos):
            aos = [(a, o, float(s)) for a, o, s in aos]
            a = {a for a, _, _ in aos}
            o = {(o, s) for _, o, s in aos}
            ao = {(a, o) for a, o, _ in aos}
            aos = set(aos)
            u_a, u_o, u_ao, u_aos = get_aos(user, item, eval_method.sentiment.user_sentiment, eval_method.sentiment)
            i_a, i_o, i_ao, i_aos = get_aos(item, user, eval_method.sentiment.item_sentiment, eval_method.sentiment)
            if u_a.intersection(i_a) or u_o.intersection(i_o):
                intersect += 1
                a_match.append(similarity(u_a, i_a, a))
                o_match.append(similarity(u_o, i_o, o))
                ao_match.append(similarity(u_ao, i_ao, ao))
                aos_match.append(similarity(u_aos, i_aos, aos))
            else:
                no_intersect += 1

        if not change:
            no_match += 1

    print(intersect, no_intersect)
    a_match, o_match, ao_match, aos_match = [np.array(d).T for d in [a_match, o_match, ao_match, aos_match]]
    for data in [a_match, o_match, ao_match, aos_match]:
        print(np.mean(data, axis=-1), np.sum(data == 0, axis=-1) / len(data.T))

    print(np.sum((a_match != 0) ^ (o_match != 0), axis=-1) / len(a_match.T))

    num_hops = np.array(num_hops)
    print(f'mean: {np.mean(num_hops)}\n'
          f'quan: {np.quantile(num_hops, [0.01, 0.1, 0.5, 0.9, 0.99])}\n'
          f'medi: {np.median(num_hops)}\n'
          f'%>1: {sum(num_hops>1)}\n'
          f'%<1: {sum(num_hops<1)}\n'
          f'tot: {len(num_hops)}')

    print('hej')


if __name__ == '__main__':
    run(sys.argv[1])