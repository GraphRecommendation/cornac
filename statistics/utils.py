import os
import pickle
from functools import reduce

import pandas as pd
from copy import deepcopy

from cornac.eval_methods import StratifiedSplit

from cornac.data.text import BaseTokenizer

from cornac.data import SentimentModality, ReviewModality, Reader

from cornac.datasets import amazon_cellphone_seer, amazon_computer_seer, amazon_toy_seer, amazon_camera_seer


def id_mapping(eval_method, eid, type):
    num_items = eval_method.train_set.num_items
    num_users = eval_method.train_set.num_users
    num_aspects = eval_method.sentiment.num_aspects
    num_opinions = eval_method.sentiment.num_opinions

    if type == 'i':
        return eid
    elif type == 'u':
        return eid + num_items
    elif type == 'a':
        return eid + num_items + num_users
    elif type == 'o':
        return eid + num_items + num_users + num_aspects
    elif type in ['aos', 'as', 'ao']:
        # create unique mapping for all aos combinations
        tot = 0
        scale = 0
        for i, t in enumerate(type):
            if t == 's':
                v = 0 if eid[i] == -1 else 1
                inc = 2
            elif t == 'o':
                v = eid[i]
                inc = num_opinions
            else:
                v = eid[i]
                inc = num_aspects
            if scale == 0:
                tot = v
                scale = inc
            else:
                tot += scale * v
                scale *= inc

        return tot
    else:
        raise NotImplementedError

def reverse_id_mapping(eval_method, eid, type):
    num_items = eval_method.train_set.num_items
    num_users = eval_method.train_set.num_users
    num_aspects = eval_method.sentiment.num_aspects
    num_opinions = eval_method.sentiment.num_opinions

    n_map = {'a': num_aspects, 'o': num_opinions, 's': 2}

    if type == 'i':
        return eid
    elif type == 'u':
        return eid - num_items
    elif type == 'a':
        return eid - num_items - num_users
    elif type == 'o':
        return eid - num_items - num_users - num_aspects
    elif type in ['aos', 'as', 'ao']:
        # create unique mapping for all aos combinations
        tot = eid
        scale = reduce(lambda x, y: x * y, [n_map[t] for t in type])
        eid = [0 for i in range(len(type))]
        for i, t in reversed(list(enumerate(type))):
            inc = n_map[t]
            scale /= inc
            eid[i] = int(tot // scale)
            tot -= scale * eid[i]

        return tuple(eid)
    else:
        raise NotImplementedError


def get_method_paths(method_kwargs, parameter_kwargs, dataset, method):
    ext = f'{method_kwargs["matching_method"]}_{"_".join(f"{k}_{v}" for k, v in sorted(method_kwargs[method].items()))}'

    if parameter_kwargs is not None:
        ext += '_'"_".join(f"{k}_{v}" for k, v in sorted(parameter_kwargs.items()))

    review_fname = f'statistics/output/selected_reviews_{dataset}_{method}_{ext}.pickle'

    # Graph of kgat is dependent on the methodology of lightrla.
    if method == 'kgat':
        ext += '_'"_".join(f"{k}_{v}" for k, v in sorted(method_kwargs['globalrla'].items()))

    graph_fname = f'statistics/output/selected_graphs_{dataset}_{method}_{ext}.pickle'
    return review_fname, graph_fname


def initialize_dataset(dataset):
    if dataset == 'cellphone':
        feedback = amazon_cellphone_seer.load_feedback(fmt="UIRT", reader=Reader())
        reviews = amazon_cellphone_seer.load_review()
        sentiment = amazon_cellphone_seer.load_sentiment(reader=Reader())
    elif dataset == 'computer':
        feedback = amazon_computer_seer.load_feedback(fmt="UIRT", reader=Reader())
        reviews = amazon_computer_seer.load_review()
        sentiment = amazon_computer_seer.load_sentiment(reader=Reader())
    elif dataset == 'toy':
        feedback = amazon_toy_seer.load_feedback(fmt="UIRT", reader=Reader())
        reviews = amazon_toy_seer.load_review()
        sentiment = amazon_toy_seer.load_sentiment(reader=Reader())
    elif dataset == 'camera':
        feedback = amazon_camera_seer.load_feedback(fmt="UIRT", reader=Reader())
        reviews = amazon_camera_seer.load_review()
        sentiment = amazon_camera_seer.load_sentiment(reader=Reader())
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


METHOD_NAMES = {'lightrla': 'LightRLA', 'narre': 'NARRE', 'hrdr': 'HRDR', 'kgat': 'KGAT', 'bpr': 'BPR',
             'trirank': 'TriRank', 'narre-bpr': 'NARRE_BPR', 'hrdr-bpr': 'HRDR_BPR', 'ngcf': 'ngcf',
             'lightgcn': 'lightgcn', 'light-e-cyclic': 'light-e-cyclic', 'globalrla': 'LightRLA',
             'globalrla-e': 'LightRLA', 'globalrla-le': 'LightRLA', 'globalrla-l': 'LightRLA',
                'globalrla-lg': 'LightRLA'}
METHOD_REMATCH = {'narre-bpr': 'narre', 'hrdr-bpr': 'hrdr'}


def initialize_model(path, dataset, method, parameter_kwargs=None):
    name = METHOD_NAMES.get(method, method)
    dir_path = os.path.join(path, dataset, METHOD_REMATCH.get(method, method), name)
    df = pd.read_csv(os.path.join(dir_path, 'results.csv'), index_col=None)

    # Get best
    best_df = df[df.score == df.score.max()]
    data = best_df.loc[best_df.index[0]].to_dict()  # get results as row

    if parameter_kwargs is not None:
        df_p = deepcopy(df)
        # Either use best parameter values or args. Useful for ablation.
        for p, v in data.items():
            if p in parameter_kwargs:
                df_p = df_p[df_p[p] == parameter_kwargs[p]]
            elif p == 'learn_weight' and not data.get('learn_explainability', False): # hotfix
                continue
            elif p not in ['file', 'id', 'index', 'epoch', 'score']:
                df_p = df_p[df_p[p] == v]

        best_df = df_p[df_p.score == df_p.score.max()]
        data = best_df.loc[best_df.index[0]].to_dict()

    file = data['file']
    with open(os.path.join(dir_path, file), 'rb') as f:
        model = pickle.load(f)

    if method in ['narre', 'hrdr']:
        model = model.load(os.path.join(dir_path, file))
    elif method.startswith('globalrla'):
        model = model.load(os.path.join(dir_path, file))

    return model