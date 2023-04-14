import os, pickle, pandas as pd
import statistics
import sys

from cornac.experiment import ExperimentResult

from cornac.metrics import *


def run(path, dataset, methods):
    eval_method = statistics.utils.initialize_dataset(dataset)

    metrics = [NDCG(), NDCG(20), NDCG(100), AUC(), MAP(), MRR(), Recall(10), Recall(20), Precision(10), Precision(20)]
    # metrics = [NDCG(), NDCG(20), NDCG(1), NDCG(10), NDCG(5), NDCG(50), NDCG(100), AUC(), MAP(), MRR()]
    eval_method._organize_metrics(metrics)
    results = ExperimentResult()
    for method in methods:
        print(method)
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

    with open(os.path.join(path, dataset, '_'.join(methods) + "_results.pickle"), 'wb') as f:
        pickle.dump(results, f)


if __name__ == '__main__':
    path, dataset = sys.argv[1:3]
    methods = sys.argv[3:]
    run(path, dataset, methods)