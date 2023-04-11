import os, pickle, pandas as pd
import statistics
import sys

from cornac.experiment import ExperimentResult

from cornac.metrics import *

from cornac.data.text import BaseTokenizer

from cornac.data import Reader, SentimentModality, ReviewModality

from cornac.datasets import amazon_cellphone_seer, amazon_computer_seer
from cornac.eval_methods import StratifiedSplit


def run(path, dataset, methods):
    eval_method = statistics.utils.initialize_dataset(dataset)

    metrics = [NDCG(), NDCG(20), NDCG(100), AUC(), MAP(), MRR(), Recall(), Recall(20), Precision(), Precision(20)]
    eval_method._organize_metrics(metrics)
    results = ExperimentResult()
    for method in methods:
        model = statistics.utils.initialize_model(path, dataset, method)

        model.train_set = eval_method.train_set
        test_result = eval_method._eval(
                model=model,
                test_set=eval_method.test_set,
                val_set=eval_method.val_set,
                user_based=True,
        )
        results.append(test_result)

    print(results)


if __name__ == '__main__':
    path, dataset = sys.argv[1:3]
    methods = sys.argv[3:]
    run(path, dataset, methods)