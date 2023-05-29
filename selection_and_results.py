import os, pickle, pandas as pd

from copy import deepcopy

import argparse

import statistics
import sys

from cornac.experiment import ExperimentResult

from cornac.metrics import *


parser = argparse.ArgumentParser()
parser.add_argument('path', type=str)
parser.add_argument('dataset')
parser.add_argument('methods', nargs='+')
parser.add_argument('--method_ablation_dict', default="None", type=str)


def run(path, dataset, methods, method_ablation_dict=None):
    eval_method = statistics.utils.initialize_dataset(dataset)

    metrics = [NDCG(), NDCG(20), NDCG(100), AUC(), MAP(), MRR(), Recall(10), Recall(20), Precision(10), Precision(20)]
    # metrics = [NDCG(), NDCG(20), NDCG(1), NDCG(10), NDCG(5), NDCG(50), NDCG(100), AUC(), MAP(), MRR()]
    eval_method._organize_metrics(metrics)
    results = ExperimentResult()
    for method in methods:
        print(method)
        if method_ablation_dict is not None and method in method_ablation_dict:
            mad = method_ablation_dict[method]
            fixed_parameters = mad.get('fixed', {})
            ablation_parameter, options = mad['ablation']
            iterator = []
            for option in options:
                p = deepcopy(fixed_parameters)
                p[ablation_parameter] = option
                iterator.append(tuple((method, p)))
        else:
            iterator = [(method, None)]

        for model, parameters in iterator:
            model = statistics.utils.initialize_model(path, dataset, method, parameter_kwargs=parameters)

            model.train_set = eval_method.train_set
            test_result = eval_method._eval(
                    model=model,
                    test_set=eval_method.test_set,
                    val_set=eval_method.val_set,
                    user_based=True,
            )

            # Is only not None if ablation parameters has been set.
            if parameters is not None:
                test_result.model_name += f'_{ablation_parameter}_{parameters[ablation_parameter]}'

            results.append(test_result)

    print(results)

    with open(os.path.join(path, dataset, '_'.join(methods) + "_results.pickle"), 'wb') as f:
        pickle.dump(results, f)


if __name__ == '__main__':
    args = parser.parse_args()
    method_kwargs = eval(args.method_ablation_dict)
    run(args.path, args.dataset, args.methods, method_ablation_dict=method_kwargs)