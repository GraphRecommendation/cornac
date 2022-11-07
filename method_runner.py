import concurrent
import itertools
import json
import pickle
import subprocess
from concurrent.futures import ThreadPoolExecutor

import numpy as np

# Shared and/or fixed parameters
shared_hyperparameters = {
    'batch_size': 256,
    'num_epochs': 500,
    'early_stopping': 10,
    'num_workers': 4,
    'model_selection': 'best',
    'user_based': True,
    'verbose': False,
    # HEAR specific but does not affect other models
    'review_aggregator': 'narre',
    'predictor': 'narre',
}

hear_hyperparameters = {
    'weight_decay': [1e-6, 1e-5, 1e-4],
    'learning_rate': [0.0001, 0.001, 0.01],
    'dropout': np.linspace(0., 0.6, 7).tolist()
}

kgat_hyperparameters = {
    'l2_weight': [1e-6, 1e-5, 1e-4],
    'learning_rate': [0.00001, 0.0001, 0.001],
    'dropout': np.linspace(0., 0.6, 7).tolist()
}

# narre_hyperparameters = {
#     'learning_rate',
#     'dropout',
# }

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
    p = subprocess.Popen(str_arg, stdout=subprocess.PIPE, shell=True, executable='/bin/bash')
    for line in p.stdout:
        print(line)

    p.wait()

    return gpu


def create_hyperparameter_dict(comb, model_parameters, shared_parameters):
    params = {}
    params.update(shared_parameters)
    params.update({k: v for k, v in zip(sorted(model_parameters), comb)})
    return params


def run(dataset, method):
    global shared_hyperparameters, hear_hyperparameters, GPUS

    if method == 'hear':
        parameters = hear_hyperparameters
    else:
        raise NotImplementedError

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
                    gpu = f.result()
                    params = create_hyperparameter_dict(combinations.pop(0), parameters, shared_hyperparameters)
                    params.update({'index': index})
                    index += 1
                    futures.append(e.submit(process_runner, dataset, method, params, gpu))
                else:
                    concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)

    concurrent.futures.wait(futures)


if __name__ == '__main__':
    run('cellphone', 'hear')