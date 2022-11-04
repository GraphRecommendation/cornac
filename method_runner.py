import concurrent
import itertools
import subprocess
from concurrent.futures import ThreadPoolExecutor

import numpy as np

shared_hyperparameters = {
    'batch_size': 128,
    'num_epochs': 500,
    'early_stopping': 20,
    'model_selection': 'best',
    'user_based': True,
    # HEAR specific but does not affect other models
    'review_aggregator': 'narre',
    'predictor': 'narre'
}

hear_hyperparameters = {
    'weight_decay': [10**i for i in range(-8, 0)] + [.0],
    'learning_rate': [0.05, 0.01, 0.005, 0.001],
    'dropout': np.linspace(0., 0.8, 9).tolist()
}

GPUS = list(range(1))
BASE_STR = 'source .venv/bin/activate;'


def process_runner(method, parameters, gpu):
    if method == 'hear':
        script = 'experiment.py'
    else:
        raise NotImplementedError

    str_arg = ' '.join([BASE_STR, f"CUDA_VISIBLE_DEVICES={gpu} python", script, str(parameters)])
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


def run(method):
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
    with ThreadPoolExecutor(max_workers=len(GPUS)) as e:
        while combinations:
            # should only be false on first iteration
            if first:
                # start process on each gpu. Zip ensures we do not iterate more than num gpus or combinations.
                for _, gpu in list(zip(combinations, GPUS)):
                    params = create_hyperparameter_dict(combinations.pop(0), parameters, shared_hyperparameters)
                    futures.append(e.submit(process_runner, method, params, gpu))

                first = False
            else:
                # Check if any completed
                completed = list(filter(lambda x: futures[x].done(), range(len(futures))))

                # if any process is completed start new on same gpu; otherwise, wait for one to finish
                if completed:
                    f = futures.pop(completed[0])
                    gpu = f.result()
                    params = create_hyperparameter_dict(combinations.pop(0), parameters, shared_hyperparameters)
                    futures.append(e.submit(process_runner, method, params, gpu))
                else:
                    concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)

    concurrent.futures.wait(futures)


if __name__ == '__main__':
    run('hear')