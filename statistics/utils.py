import os
import pickle

import pandas as pd

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