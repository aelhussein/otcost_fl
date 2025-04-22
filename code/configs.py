import os
import sys
import torch
import numpy as np # Keep basic imports if needed by config values themselves
# Avoid importing complex classes from other project modules here if possible

# --- Core Directories ---
# (Keep as before)
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT  = os.path.dirname(_CURRENT_DIR)
ROOT_DIR = _PROJECT_ROOT
DATA_DIR = f'{ROOT_DIR}/data'
RESULTS_DIR = f'{ROOT_DIR}/results'
MODEL_SAVE_DIR = f'{ROOT_DIR}/saved_models'
# Add other DIR constants if needed

# --- Global Settings ---
DEVICE = 'cuda' if 'torch' in sys.modules and torch.cuda.is_available() else 'cpu' # Check if torch is loaded
N_WORKERS = 1

# --- Supported Algorithms ---
ALGORITHMS = ['local', 'fedavg']

# --- Supported Datasets (names must match keys in DEFAULT_PARAMS) ---
# (Keep DATASETS list as defined in the previous step)
DATASETS = [
    'Synthetic', 'Credit', 'EMNIST', 'CIFAR', 'ISIC', 'IXITiny', 'Heart'
]

# --- Default Hyperparameters & Data Handling Configuration ---
# (Keep DEFAULT_PARAMS dictionary exactly as defined in the previous step,
#  with updated settings for Synthetic, Credit, Heart and removed Weather)
DEFAULT_PARAMS = {
    'CIFAR': {
        'data_source': 'torchvision', 'partitioning_strategy': 'dirichlet_indices', 'cost_interpretation': 'alpha',
        'dataset_class': 'CIFARDataset', 'default_num_clients': 10, 'max_clients': None, 'fixed_classes': 10,
        'source_args': {'dataset_name': 'CIFAR10', 'data_dir': DATA_DIR},
        'partitioner_args': {}, 'partition_scope': 'train',
        'sampling_config': {'type': 'fixed_total', 'size': 3000, 'replace': False},
        'learning_rates_try': [5e-3, 1e-3, 5e-4], 'default_lr': 1e-3,
        'reg_params_try':[1, 1e-1, 1e-2, 1e-3], 'default_reg_param': 1e-1,
        'batch_size': 128, 'epochs_per_round': 1, 'rounds': 100,
        'runs': 10, 'runs_tune': 3, 'metric': 'Accuracy', 'base_seed': 42
    },
    'EMNIST': {
        'data_source': 'torchvision', 'partitioning_strategy': 'dirichlet_indices', 'cost_interpretation': 'alpha',
        'dataset_class': 'EMNISTDataset', 'default_num_clients': 10, 'max_clients': None, 'fixed_classes': 10,
        'source_args': {'dataset_name': 'EMNIST', 'data_dir': DATA_DIR, 'split': 'digits'},
        'partitioner_args': {}, 'partition_scope': 'train',
        'sampling_config': {'type': 'fixed_total', 'size': 1000, 'replace': False},
        'learning_rates_try': [5e-3, 1e-3, 5e-4, 1e-4], 'default_lr': 1e-3,
        'reg_params_try':[1, 1e-1, 1e-2, 1e-3], 'default_reg_param': 1e-1,
        'batch_size': 128, 'epochs_per_round': 1, 'rounds': 75,
        'runs': 10, 'runs_tune': 3, 'metric': 'Accuracy', 'base_seed': 42
    },
    'Synthetic': {
        'data_source': 'synthetic_base', 'partitioning_strategy': 'dirichlet_indices', 'cost_interpretation': 'alpha',
        'dataset_class': 'SyntheticDataset', 'default_num_clients': 10, 'max_clients': None, 'fixed_classes': 2,
        'source_args': {'base_n_samples': 20000, 'n_features': 10, 'dist_type1': 'normal', 'dist_type2': 'skewed', 'label_noise': 0.05, 'random_state': 42 },
        'partitioner_args': {}, 'partition_scope': 'all', 'needs_preprocessing': ['standard_scale'],
        'sampling_config': {'type': 'fixed_total', 'size': 1000, 'replace': False},
        'learning_rates_try':[5e-1, 1e-1, 5e-2, 1e-2, 5e-3, 1e-3], 'default_lr': 1e-3,
        'reg_params_try':[1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5], 'default_reg_param': 1e-1,
        'batch_size': 64, 'epochs_per_round': 1, 'rounds': 100,
        'runs': 10, 'runs_tune': 3, 'metric': 'F1', 'base_seed': 42
    },
    'Credit': {
        'data_source': 'credit_base', 'partitioning_strategy': 'dirichlet_indices', 'cost_interpretation': 'alpha',
        'dataset_class': 'CreditDataset', 'default_num_clients': 10, 'max_clients': None, 'fixed_classes': 2,
        'source_args': {'csv_path': os.path.join(DATA_DIR, 'Credit/creditcard.csv'), 'drop_cols': ['Time', 'Amount']},
        'partitioner_args': {}, 'partition_scope': 'all', 'needs_preprocessing': ['standard_scale'],
        'sampling_config': {'type': 'fixed_total', 'size': 1000, 'replace': False},
        'learning_rates_try':[5e-1, 1e-1, 5e-2, 1e-2, 5e-3, 1e-3], 'default_lr':1e-3,
        'reg_params_try':[1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5], 'default_reg_param': 1e-1,
        'batch_size': 64, 'epochs_per_round': 1, 'rounds': 100,
        'runs': 10, 'runs_tune': 3, 'metric': 'F1', 'base_seed': 42
    },
    'Heart': {
        'data_source': 'heart_site_loader', 'partitioning_strategy': 'pre_split', 'cost_interpretation': 'site_mapping_key',
        'dataset_class': 'HeartDataset', 'default_num_clients': 2, 'max_clients': 2, 'fixed_classes': 2,
        'source_args': {
            'data_dir': os.path.join(DATA_DIR, 'Heart'),
            'sites': ['cleveland', 'hungarian', 'switzerland', 'va'],
             'used_columns': ['age', 'sex', 'chest_pain_type', 'resting_bp', 'cholesterol', 'sugar', 'ecg', 'max_hr', 'exercise_angina', 'exercise_ST_depression', 'target'],
            'feature_names': ['age', 'sex', 'chest_pain_type', 'resting_bp', 'cholesterol', 'sugar', 'ecg', 'max_hr', 'exercise_angina', 'exercise_ST_depression'],
            'cols_to_scale': ['age', 'chest_pain_type', 'resting_bp', 'cholesterol', 'ecg', 'max_hr', 'exercise_ST_depression'],
            'scale_values': { 'age': (53.0872973, 7.01459463e+01), 'chest_pain_type': (3.23702703, 8.17756772e-01), 'resting_bp': (132.74405405, 3.45493057e+02), 'cholesterol': (220.23648649, 4.88430934e+03), 'ecg': (0.64513514, 5.92069868e-01), 'max_hr': (138.75459459, 5.29172208e+02), 'exercise_ST_depression': (0.89532432, 1.11317517e+00) },
            'site_mappings': { 1: [['cleveland'], ['hungarian']], 2: [['cleveland'], ['switzerland']], 3: [['cleveland'], ['va']], 4: [['hungarian'], ['switzerland']], 5: [['hungarian'], ['va']], 6: [['switzerland'], ['va']] }
        },
        'partitioner_args': {}, 'sampling_config': None,
        'learning_rates_try':[5e-1, 1e-1, 5e-2, 1e-2, 5e-3, 1e-3], 'default_lr':1e-3,
        'reg_params_try':[1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5], 'default_reg_param': 1e-1,
        'batch_size': 64, 'epochs_per_round': 1, 'rounds': 100,
        'runs': 10, 'runs_tune': 3, 'metric': 'F1', 'base_seed': 42
    },
    'ISIC': {
        'data_source': 'pre_split_paths_isic', 'partitioning_strategy': 'pre_split', 'cost_interpretation': 'site_mapping_key',
        'dataset_class': 'ISICDataset', 'default_num_clients': 2, 'max_clients': 4, 'fixed_classes': 8,
        'source_args': { 'data_dir': os.path.join(DATA_DIR, 'ISIC'), 'site_mappings': { 0.06: [[2], [2]], 0.15: [[2], [0]], 0.19: [[2], [3]], 0.25: [[2], [1]], 0.3: [[1], [3]], 'all': [[0], [1], [2], [3]] }},
        'partitioner_args': {}, 'sampling_config': {'type': 'fixed_total', 'size': 2000},
        'learning_rates_try': [5e-3, 1e-3, 5e-4, 1e-4], 'default_lr': 1e-3,
        'reg_params_try':[1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5], 'default_reg_param': 1e-1,
        'batch_size': 128, 'epochs_per_round': 1, 'rounds': 60,
        'runs': 5, 'runs_tune': 1, 'metric': 'Balanced_accuracy', 'base_seed': 42
    },
    'IXITiny': {
        'data_source': 'pre_split_paths_ixi', 'partitioning_strategy': 'pre_split', 'cost_interpretation': 'site_mapping_key',
        'dataset_class': 'IXITinyDataset', 'default_num_clients': 2, 'max_clients': 3, 'fixed_classes': 2,
        'source_args': { 'data_dir': os.path.join(DATA_DIR, 'IXITiny'), 'site_mappings': { 0.08: [['Guys'], ['HH']], 0.28: [['IOP'], ['Guys']], 0.30: [['IOP'], ['HH']], 'all': [['IOP'], ['HH'], ['Guys']] }},
        'partitioner_args': {}, 'sampling_config': None,
        'learning_rates_try': [1e-2, 5e-3, 1e-3, 5e-4], 'default_lr': 1e-3,
        'reg_params_try':[1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5], 'default_reg_param': 1e-1,
        'batch_size': 64, 'epochs_per_round': 1, 'rounds': 50,
        'runs': 10, 'runs_tune': 3, 'metric': 'DICE', 'base_seed': 42
    }
}

# (Keep DATASET_COSTS dictionary exactly as defined in the previous step)
DATASET_COSTS = {
    'IXITiny': [0.08, 0.28, 0.30, 'all'], 'ISIC': [0.06, 0.15, 0.19, 0.25, 0.3, 'all'],
    'EMNIST': [0.1, 0.5, 1.0, 5.0, 10.0, 1000.0], 'CIFAR': [0.1, 0.5, 1.0, 5.0, 10.0, 1000.0],
    'Synthetic': [0.1, 0.3, 0.5, 1.0, 5.0, 10.0, 1000.0], 'Credit': [0.1, 0.3, 0.5, 1.0, 5.0, 10.0, 1000.0],
    'Heart': [1, 2, 3, 4, 5, 6]
}