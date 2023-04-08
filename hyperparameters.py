# Shared and/or fixed parameters
import numpy as np

shared_hyperparameters = {
    'batch_size': 256,
    'num_epochs': 500,
    'early_stopping': 25,
    'num_workers': 4,
    'model_selection': 'best',
    'user_based': True,
    'verbose': False,
    # HEAR specific but does not affect other models
    'predictor': 'narre',
}

lightrla2_hyperparameters = {
    'review_aggregator': ['narre'],
    'preference_module': ['lightgcn'],
    'predictor': ['dot'],
    'weight_decay': [10**i for i in range(-6, 3)],
    'learning_rate': [0.001],
    'dropout': [0.2],
    'l2_weight': [0],
    'num_neg_samples': [50],
    'name': ['light2']
}

lightrla_hyperparameters = {
    'review_aggregator': ['narre'],
    'preference_module': ['mf', 'lightgcn'],
    'combine': ['concat'],
    'predictor': ['dot', 'narre'],
    'weight_decay': [1e-6, 1e-5, 1e-4],
    'learning_rate': [0.0001, 0.001, 0.01],
    'dropout': np.linspace(0., 0.6, 7).round(1).tolist(),
    'l2_weight': [0],
    'num_neg_samples': [50]
}

kgat_hyperparameters = {
    'l2_weight': [1e-6, 1e-5, 1e-4],
    'learning_rate': [0.00001, 0.0001, 0.001],
    'dropout': np.linspace(0., 0.6, 7).round(1).tolist()
}

narre_hyperparameters = {
    'learning_rate': [0.00001, 0.0001, 0.001],
    'dropout_rate': np.linspace(0., 0.6, 7).round(1).tolist(),
    'max_iter': [shared_hyperparameters['num_epochs']]
}

bpr_hyperparameters = {
    'k': [32, 64, 128],
    'learning_rate': [0.00001, 0.0001, 0.001],
    'lambda_reg': [1e-6, 1e-5, 1e-4],
    'use_bias': [True, False],
    'max_iter': [shared_hyperparameters['num_epochs']]
}

trirank_hyperparameters_1 = {
    'alpha': [0., 0.5, 1.],
    'beta': [0.],
    'gamma': [0.],
    'mu_U': [0., 0.5, 1.],
    'mu_P': [0., 0.5, 1.],
    'mu_A': [0.]
}

trirank_hyperparameters_cell = {
    'alpha': [1.],
    'beta': [0., 0.5, 1.],
    'gamma': [0., 0.5, 1.],
    'mu_U': [0.],
    'mu_P': [0.5],
    'mu_A': [0., 0.5, 1.]
}