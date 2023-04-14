import os
import pickle

import pandas as pd

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


def get_method_paths(method_kwargs, dataset, method):
    ext = f'{method_kwargs["matching_method"]}_{"_".join(f"{k}_{v}" for k, v in method_kwargs[method].items())}'

    review_fname = f'statistics/output/selected_reviews_{dataset}_{method}_{ext}.pickle'

    # Graph of kgat is dependent on the methodology of lightrla.
    if method == 'kgat':
        ext += '_'"_".join(f"{k}_{v}" for k, v in method_kwargs['lightrla'].items())

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
             'lightgcn': 'lightgcn', 'light-e-cyclic': 'light-e-cyclic'}
METHOD_REMATCH = {'narre-bpr': 'narre', 'hrdr-bpr': 'hrdr'}


def initialize_model(path, dataset, method):
    name = METHOD_NAMES.get(method, method)
    dir_path = os.path.join(path, dataset, METHOD_REMATCH.get(method, method), name)
    df = pd.read_csv(os.path.join(dir_path, 'results.csv'), index_col=None)

    best_df = df[df.score == df.score.max()]
    data = best_df.loc[best_df.index[0]].to_dict()  # get results as row
    file = data['file']
    with open(os.path.join(dir_path, file), 'rb') as f:
        model = pickle.load(f)

    return model