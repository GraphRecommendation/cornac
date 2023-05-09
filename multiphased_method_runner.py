import concurrent
import itertools
import json
import os.path
import pickle
import subprocess
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

with open('config.json') as f:
    config = json.load(f)

if isinstance(config['GPUS'], list):
    GPUS = config['GPUS']
else:
    GPUS = list(range(config['GPUS']))
GPUS = [g for _ in range(config['GPU_MULT']) for g in GPUS]  # Multiplier for multiple processes per GPU.
BASE_STR = config['BASE']

name_dict = {'lightrla': 'LightRLA', 'narre': 'NARRE', 'hrdr': 'HRDR', 'kgat': 'KGAT', 'bpr': 'BPR',
             'trirank': 'TriRank', 'narre-bpr': 'NARRE_BPR', 'hrdr-bpr': 'HRDR_BPR', 'ngcf': 'ngcf',
             'lightgcn': 'lightgcn', 'globalrla': 'LightRLA'}


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

def validate_hyperparameters(dictionary):
    info = []
    for pp in dictionary['phases']:
        arguments = set()
        for k, v in pp.items():
            arguments.update(list(v.keys()))

        values = list(pp.get('tune', {}).values())
        n = len(list(itertools.product(*values)))
        info.append((arguments, n))

    example = info[0][0]
    for i, res in enumerate(info):
        assert example == res[0], f'Hyperparameters for phase {i} does not match: {res[0]}'
        print(i, "Found combination lengths:", res[1])


def run(dataset, method, path='results'):
    global GPUS

    methods_hyperparameters = json.load(open('multiphased_hyperparameters.json'))
    shared_hyperparameters = methods_hyperparameters['shared']

    # Uses same hyperparameters
    if (bpr_flag := method.endswith('-bpr')) and (method.startswith('narre') or method.startswith('hrdr')):
        method = method.replace('-bpr', '')
        method_name = 'narre'
    elif method in ['lightgcn', 'ngcf']:
        method_name = 'kgat'
    else:
        method_name = method
    method_dict = methods_hyperparameters[method_name]

    default_parameters = method_dict.get('default', {})

    validate_hyperparameters(method_dict)

    if bpr_flag:
        default_parameters['use_bpr'] = True

    failed = []
    for phase_parameters in method_dict['phases']:
        parameters = phase_parameters['tune']
        fixed_parameters = phase_parameters.get('fixed', {})
        optimal_parameters = phase_parameters.get('optimal', {})

        if method == 'lightrla-explain':
            method = 'lightrla'

        # if using optimal parameters find optimal and assign.
        if len(optimal_parameters) > 0:
            method_name = method
            if default_parameters.get('use_bpr', False):
                method_name += '-bpr'

            save_dir = os.path.join(path, dataset, method, name_dict[method_name], 'results.csv')
            if os.path.isfile(save_dir):
                df = pd.read_csv(save_dir)
                best_df = df[df.score == df.score.max()]
                b_param = best_df.iloc[0].to_dict()
                for k, v in b_param.items():
                    if k in optimal_parameters:
                        optimal_parameters[k] = v
                    elif k == 'layer_dropout' and 'dropout' in optimal_parameters:
                        optimal_parameters['dropout'] = v
            else:
                print(f'Warning: could not find optimal parameters at {save_dir}, using default.')

        values = [parameters[k] for k in sorted(parameters)]  # ensure order of parameters
        combinations = list(itertools.product(*values))

        print(f'Going through a total of {len(combinations)} parameter combinations.')

        futures = []
        first = True
        index = 0
        with ThreadPoolExecutor(max_workers=len(GPUS)) as e, open(f'{dataset}_{method}_parameters.pkl', 'wb') as f:
            while combinations:
                # should only be false on first iteration
                if first:
                    # start process on each gpu. Zip ensures we do not iterate more than num gpus or combinations.
                    for _, gpu in list(zip(combinations, GPUS)):
                        params = create_hyperparameter_dict(combinations.pop(0), parameters, shared_hyperparameters, default_parameters, fixed_parameters, optimal_parameters)
                        params.update({'index': index})
                        index += 1
                        pickle.dump(params, f)
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

                        params = create_hyperparameter_dict(combinations.pop(0), parameters, shared_hyperparameters, default_parameters, fixed_parameters, optimal_parameters)
                        params.update({'index': index})
                        index += 1
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
    for dataset in datasets:

        print('----', dataset, '----')
        for method in methods:
            print('--', method, '--')
            run(dataset, method)