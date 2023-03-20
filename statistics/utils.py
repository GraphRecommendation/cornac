import os
import pickle
from collections import OrderedDict, defaultdict
from functools import lru_cache

import pandas as pd
import re

from cornac.eval_methods import StratifiedSplit

from cornac.data.text import BaseTokenizer

from cornac.data import SentimentModality, ReviewModality, Reader

from cornac.datasets import amazon_cellphone_seer, amazon_computer_seer


def id_mapping(eval_method, eid, type):
    num_items = eval_method.train_set.num_items
    num_users = eval_method.train_set.num_users
    num_aspects = eval_method.sentiment.num_aspects

    if type == 'i':
        return eid
    elif type == 'u':
        return eid + num_items
    elif type == 'a':
        return eid + num_items + num_users
    else:
        return eid + num_items + num_users + num_aspects


def get_method_paths(method_kwargs, dataset, method):
    ext = f'{method_kwargs["matching_method"]}_{"_".join(f"{k}_{v}" for k, v in method_kwargs[method].items())}'

    review_fname = f'statistics/output/selected_reviews_{dataset}_{method}_{ext}.pickle'

    # Graph of kgat is dependent on the methodology of lightrla.
    if method == 'kgat':
        ext += '_'"_".join(f"{k}_{v}" for k, v in method_kwargs['lightrla'].items())

    graph_fname = f'statistics/output/selected_graphs_{dataset}_{method}_{ext}.pickle'
    return review_fname, graph_fname


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


def initialize_dataset(dataset):
    if dataset == 'cellphone':
        feedback = amazon_cellphone_seer.load_feedback(fmt="UIRT", reader=Reader())
        reviews = amazon_cellphone_seer.load_review()
        sentiment = amazon_cellphone_seer.load_sentiment(reader=Reader())
    elif dataset == 'computer':
        feedback = amazon_computer_seer.load_feedback(fmt="UIRT", reader=Reader())
        reviews = amazon_computer_seer.load_review()
        sentiment = amazon_computer_seer.load_sentiment(reader=Reader())
    else:
        raise NotImplementedError

    sentiment_modality = SentimentModality(data=sentiment)

    review_modality = ReviewModality(
        data=reviews,
        tokenizer=BaseTokenizer(stop_words="english"),
        max_vocab=4000,
        max_doc_freq=0.5,
    )

    eval_method = StratifiedSplit(
        feedback,
        group_by="user",
        chrono=True,
        sentiment=sentiment_modality,
        review_text=review_modality,
        test_size=0.2,
        val_size=0.16,
        exclude_unknowns=True,
        seed=42,
        verbose=False,
    )

    return eval_method


METHOD_NAMES = {'lightrla': 'LightRLA', 'lightgcn': 'lightgcn', 'light2': 'light2', 'narre': 'NARRE_BPR',
                'kgat': 'KGAT'}


def initialize_model(path, dataset, method):
    name = METHOD_NAMES[method]
    dir_path = os.path.join(path, dataset, method, name)
    df = pd.read_csv(os.path.join(dir_path, 'results.csv'), index_col=None)

    best_df = df[df.score == df.score.max()]
    data = best_df.loc[best_df.index[0]].to_dict()  # get results as row
    file = data['file']
    with open(os.path.join(dir_path, file), 'rb') as f:
        model = pickle.load(f)

    return model