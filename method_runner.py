import concurrent
import itertools
import json
import pickle
import subprocess
from concurrent.futures import ThreadPoolExecutor

from hyperparameters import *


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


def create_hyperparameter_dict(comb, model_parameters, shared_parameters):
    params = {'skip_tried': config.get('skip_tried', False)}
    params.update(shared_parameters)
    params.update({k: v for k, v in zip(sorted(model_parameters), comb)})
    return params


def run(dataset, method):
    global shared_hyperparameters, hypar_hyperparameters, GPUS
    if method == 'hypar':
        parameters = hypar_hyperparameters
    elif method in ['kgat', 'lightgcn', 'ngcf']:
        parameters = kgat_hyperparameters
    elif method == 'bpr':
        parameters = bpr_hyperparameters
    elif method in ['narre', 'hrdr', 'narre-bpr', 'hrdr-bpr']:
        parameters = narre_hyperparameters

        if method.endswith('-bpr'):
            method = method.replace('-bpr', '')
            parameters['use_bpr'] = [True]

    elif method == 'trirank-1':
        parameters = trirank_hyperparameters_1
        method = 'trirank'
    elif method == 'trirank':
        if dataset == 'cellphone':
            parameters = trirank_hyperparameters_cell
        else:
            NotImplementedError
        method = 'trirank'
    else:
        raise NotImplementedError

    values = [parameters[k] for k in sorted(parameters)]  # ensure order of parameters
    combinations = list(itertools.product(*values))

    print(f'Going through a total of {len(combinations)} parameter combinations.')

    futures = []
    first = True
    index = 0
    failed = []
    with ThreadPoolExecutor(max_workers=len(GPUS)) as e, open(f'{dataset}_{method}_parameters.pkl', 'wb') as f:
        while combinations:
            # should only be false on first iteration
            if first:
                # start process on each gpu. Zip ensures we do not iterate more than num gpus or combinations.
                for _, gpu in list(zip(combinations, GPUS)):
                    params = create_hyperparameter_dict(combinations.pop(0), parameters, shared_hyperparameters)
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

                    params = create_hyperparameter_dict(combinations.pop(0), parameters, shared_hyperparameters)
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