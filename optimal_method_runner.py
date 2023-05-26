import concurrent
import itertools
import json
import os.path
import pickle
import subprocess
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from copy import deepcopy

from statistics.utils import METHOD_NAMES

with open('config.json') as f:
    config = json.load(f)

if isinstance(config['GPUS'], list):
    GPUS = config['GPUS']
else:
    GPUS = list(range(config['GPUS']))
GPUS = [g for _ in range(config['GPU_MULT']) for g in GPUS]  # Multiplier for multiple processes per GPU.
BASE_STR = config['BASE']


def process_runner(dataset, method, parameters, gpu):
    str_arg = ' '.join([BASE_STR, f"CUDA_VISIBLE_DEVICES={gpu} python experiment.py {dataset} {method}",
                        str(parameters)]).replace("'", "\\'")
    returncode = subprocess.run(str_arg, shell=True, executable='/bin/bash').returncode
    # for line in p.stdout:
    #     print(line)
    #
    # p.wait()
    # returncode = p.returncode

    return gpu, returncode, parameters


def create_hyperparameter_dict(comb, model_parameters, shared_parameters, default_parameters, fixed_parameters,
                               optimal_parameters):
    params = {'skip_tried': config.get('skip_tried', False)}
    params.update(shared_parameters)
    params.update({k: v for k, v in zip(sorted(model_parameters), comb)})
    params.update(fixed_parameters)
    params.update(optimal_parameters)
    params.update(default_parameters)
    return params


def run(datasets, methods, tune_dataset, ablation_kwargs, path='results'):
    global GPUS

    methods_hyperparameters = json.load(open('multiphased_hyperparameters.json'))
    shared_hyperparameters = methods_hyperparameters['shared']

    method_optimal_parameters = {}

    for method in methods:
        # Uses same hyperparameters
        if (bpr_flag := method.endswith('-bpr')) or (method.startswith('narre') or method.startswith('hrdr')):
            method_name = 'narre'
        elif method in ['lightgcn', 'ngcf']:
            method_name = 'kgat'
        else:
            method_name = method
        method_dict = methods_hyperparameters[method_name]

        default_parameters = method_dict.get('default', {})

        if bpr_flag:
            default_parameters['use_bpr'] = True

        phase_parameters = method_dict['phases'][0]
        parameters = phase_parameters['tune']
        fixed_parameters = phase_parameters.get('fixed', {})
        opt_parameters = phase_parameters.get('fixed', {})

        optimal_parameters = {}
        p_names = set(phase_parameters).union(parameters).union(fixed_parameters).union(default_parameters)\
            .union(shared_hyperparameters).union(opt_parameters)
        save_dir = os.path.join(path, tune_dataset, method.replace('-bpr', ''), METHOD_NAMES[method], 'results.csv')
        if os.path.isfile(save_dir):
            df = pd.read_csv(save_dir)
            best_df = df[df.score == df.score.max()]
            b_param = best_df.iloc[0].to_dict()

            if len(map := ablation_kwargs.get(method, {})):
                afix = map['fixed']
                ablation_parameter, options = map['ablation']
                b_param.update(afix)
                iterator = [(ablation_parameter, option) for option in options]
            else:
                iterator = [(None, None)]

            for ap, op in iterator:
                for k, v in b_param.items():
                    if k in p_names:
                        optimal_parameters[k] = v
                    elif k == 'layer_dropout' and 'dropout' in p_names:
                        optimal_parameters['dropout'] = v
                optimal_parameters['skip_tried'] = config.get('skip_tried', False)
                optimal_parameters.update(shared_hyperparameters)

                if method in ['lightrla-explain', 'light-e-cyclic']:
                    method = 'lightrla'

                if ap and op is not None:
                    optimal_parameters[ap] = op

                method_optimal_parameters[(method.replace('-bpr', ''), op)] = deepcopy(optimal_parameters)
        else:
            raise ValueError(f'Could not find results for', method, 'at the path: ', save_dir)

    combinations = list(method_optimal_parameters.items())
    combinations = [(d, m, p) for d in datasets for m, p in combinations]
    print(f'Going through a total of {len(combinations)} combinations.')

    failed = []
    futures = []
    first = True
    index = 0
    with ThreadPoolExecutor(max_workers=len(GPUS)) as e:
        while combinations:
            # should only be false on first iteration
            if first:
                # start process on each gpu. Zip ensures we do not iterate more than num gpus or combinations.
                for _, gpu in list(zip(combinations, GPUS)):
                    dataset, (method, _), params = combinations.pop(0)
                    params.update({'index': index})
                    futures.append(e.submit(process_runner, dataset, method, params, gpu))
                first = False
            else:
                # Check if any completed
                completed = list(filter(lambda x: futures[x].done(), range(len(futures))))

                # if any process is completed start new on same gpu; otherwise, wait for one to finish
                if completed:
                    f = futures.pop(completed[0])
                    gpu, returncode, pm = f.result()
                    if returncode != 0:
                        failed.append(pm)

                    dataset, (method, _), params = combinations.pop(0)
                    params.update({'index': index})
                    futures.append(e.submit(process_runner, dataset, method, params, gpu))
                else:
                    concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)

    concurrent.futures.wait(futures)
    for f in futures:
        gpu, returncode, parameters = f.result()
        if returncode != 0:
            failed.append(parameters)

    if len(failed):
        with open(f'failed_experiments_{dataset}_{method}.pickle', 'wb') as f:
            pickle.dump(failed, f)


if __name__ == '__main__':
    datasets = config['DATASETS'] if 'DATASETS' in config else [config['DATASET']]
    methods = config['METHODS'] if 'METHODS' in config else [config['METHOD']]
    ablation_kwargs = config['ABLATION']
    run(datasets, methods, 'cellphone', ablation_kwargs)