# configs.py
"""
Central configuration file for the project.
Defines directory paths, constants, default hyperparameters,
data handling configurations, algorithm settings.
Streamlined version. REMOVED cost_interpretation.
"""
import os
import torch
# import numpy as np # Only if needed for specific values below

# --- Core Directories ---
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT  = os.path.dirname(_CURRENT_DIR)
ROOT_DIR = _PROJECT_ROOT
DATA_DIR = os.path.join(ROOT_DIR, 'data')
RESULTS_DIR = os.path.join(ROOT_DIR, 'results')
MODEL_SAVE_DIR = os.path.join(ROOT_DIR, 'saved_models')
ACTIVATION_DIR = os.path.join(ROOT_DIR, 'activations')

# --- Global Settings ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
N_WORKERS = 0 # Use 0 for simplicity/debugging

# --- Supported Algorithms ---
ALGORITHMS = ['local', 'fedavg'] # Add others as implemented

# --- Supported Datasets ---
DATASETS = [
    'Synthetic_Label', 'Synthetic_Feature', 'Synthetic_Concept',
    'Credit', 'EMNIST', 'CIFAR', 'ISIC', 'IXITiny', 'Heart'
]

# --- Common Configuration for Tabular-like Datasets ---
COMMON_TABULAR_PARAMS = dict(
    fixed_classes=2,
    default_lr=1e-3,
    learning_rates_try=[1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4],
    default_reg_param=0.1, # For FedProx, pFedMe, Ditto if added
    reg_params_try=[1, 0.1, 0.01], # For FedProx, pFedMe, Ditto
    batch_size=32,
    epochs_per_round=5,
    rounds=50,
    rounds_tune_inner=20,
    runs=10, # Number of evaluation runs
    runs_tune=3, # Number of tuning runs
    metric='Accuracy', # Default metric
    base_seed=42,
    samples_per_client = 250,
    default_num_clients=5,
    max_clients=None,
    servers_tune_lr=ALGORITHMS,
    servers_tune_reg=[], # Tune Reg only for specified algos
    partitioner_args={}, # Extra args for partitioner
    max_parallel_clients=None,
)

# --- Default Hyperparameters & Data Handling Configuration ---
DEFAULT_PARAMS = {

    # === Unified Synthetic Configurations ===
    'Synthetic_Label': {
        **COMMON_TABULAR_PARAMS,
        'dataset_name': 'Synthetic_Label',
        'data_source': 'synthetic', # Unified loader key
        'partitioning_strategy': 'dirichlet_indices', # Needs alpha passed directly
        # REMOVED: cost_interpretation
        'dataset_class': 'SyntheticDataset',
        'source_args': { # Params for base generation (mode='baseline')
            'base_n_samples': 50000,
            'n_features': 15,
            'label_noise': 0.1,
            'random_state': 42,
        },
        'metric': 'F1',
    },
    'Synthetic_Feature': {
        **COMMON_TABULAR_PARAMS,
        'dataset_name': 'Synthetic_Feature',
        'data_source': 'synthetic',
        'partitioning_strategy': 'pre_split', # Data generated per client
        # REMOVED: cost_interpretation (Loader uses cost_key as shift_param)
        'dataset_class': 'SyntheticDataset',
        'source_args': { # Params for per-client generation (mode='feature_shift')
            'base_n_samples': 50000,
            'n_features': 15,
            'label_noise': 0.01,
            # Feature shift specific config (passed as **kwargs to generator)
            'feature_shift_kind': 'mean',
            'feature_shift_cols': 15,
            'feature_shift_mu': 3.0,
        },
    },
    'Synthetic_Concept': {
        **COMMON_TABULAR_PARAMS,
        'dataset_name': 'Synthetic_Concept',
        'data_source': 'synthetic',
        'partitioning_strategy': 'pre_split', # Partition baseline data IID
        # REMOVED: cost_interpretation (Loader uses cost_key as shift_param)
        'dataset_class': 'SyntheticDataset',
        'source_args': { # Params for base generation + concept shift (mode='concept_shift')
            'base_n_samples': 50000,
            'n_features': 15,
            'label_noise': 0.01,
            'random_state': 42,
            # Concept shift specific config (passed as **kwargs to generator)
            'concept_label_option': 'threshold',
            'concept_threshold_range_factor': 0.4
        },
    },

    # === Other Datasets ===
    'Credit': {
        **COMMON_TABULAR_PARAMS,
        'dataset_name': 'Credit',
        'data_source': 'credit_csv',
        'partitioning_strategy': 'dirichlet_indices', # Needs alpha passed directly
        # REMOVED: cost_interpretation
        'dataset_class': 'CreditDataset',
        'source_args': {
            'csv_path': os.path.join(DATA_DIR, 'Credit', 'creditcard.csv'),
            'drop_cols': ['Time']
        },
        'metric': 'F1',
        'learning_rates_try': [5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4],
    },
    'CIFAR': {
        'dataset_name': 'CIFAR',
        'data_source': 'torchvision',
        'partitioning_strategy': 'dirichlet_indices', # Needs alpha passed directly
        # REMOVED: cost_interpretation
        'dataset_class': 'CIFARDataset',
        'source_args': {'dataset_name': 'CIFAR10'},
        'transform_config': {},
        'partitioner_args': {},
        'samples_per_client': 1000,
        'fixed_classes': 10,
        'default_lr': 1e-3, 'learning_rates_try': [1e-2, 5e-3, 1e-3, 5e-4],
        'default_reg_param': 0.1, 'reg_params_try':[1, 0.1, 1e-2],
        'batch_size': 64, 'epochs_per_round': 5, 'rounds': 100, 'rounds_tune_inner': 25,
        'runs': 10, 'runs_tune': 3, 'metric': 'Accuracy', 'base_seed': 42,
        'default_num_clients': 5, 'max_clients': None,
        'servers_tune_lr': ALGORITHMS, 'servers_tune_reg': [],
        'max_parallel_clients' : None,
    },
    'EMNIST': {
        'dataset_name': 'EMNIST',
        'data_source': 'torchvision',
        'partitioning_strategy': 'dirichlet_indices', # Needs alpha passed directly
        # REMOVED: cost_interpretation
        'dataset_class': 'EMNISTDataset',
        'source_args': {'dataset_name': 'EMNIST', 'split': 'digits'},
        'transform_config': {},
        'partitioner_args': {},
        'samples_per_client': 750,
        'fixed_classes': 10,
        'default_lr': 1e-3, 'learning_rates_try': [1e-2, 5e-3, 1e-3, 5e-4],
        'default_reg_param': 0.1, 'reg_params_try':[1, 0.1, 1e-2],
        'batch_size': 48, 'epochs_per_round': 5, 'rounds': 100, 'rounds_tune_inner': 25,
        'runs': 10, 'runs_tune': 3, 'metric': 'Accuracy', 'base_seed': 42,
        'default_num_clients': 5, 'max_clients': None,
        'servers_tune_lr': ALGORITHMS, 'servers_tune_reg': [],
        'max_parallel_clients' : None,
    },
    'Heart': {
        **COMMON_TABULAR_PARAMS,
        'dataset_name': 'Heart',
        'data_source': 'heart_site_loader',
        'partitioning_strategy': 'pre_split', # Uses cost as key for site_mappings
        'batch_size':32,
        'dataset_class': 'HeartDataset',
        'source_args': { # Loader uses cost_key directly to index site_mappings
            # data_dir passed by manager
            'sites': ['cleveland', 'hungarian', 'switzerland', 'va'],
            'used_columns': ['age', 'sex', 'chest_pain_type', 'resting_bp', 'cholesterol', 'sugar', 'ecg', 'max_hr', 'exercise_angina', 'exercise_ST_depression', 'target'],
            'feature_names': ['age', 'sex', 'chest_pain_type', 'resting_bp', 'cholesterol', 'sugar', 'ecg', 'max_hr', 'exercise_angina', 'exercise_ST_depression'],
            'cols_to_scale': ['age', 'chest_pain_type', 'resting_bp', 'cholesterol', 'ecg', 'max_hr', 'exercise_ST_depression'],
            'scale_values': { 'age': (53.0872973, 7.01459463e+01), 'chest_pain_type': (3.23702703, 8.17756772e-01), 'resting_bp': (132.74405405, 3.45493057e+02), 'cholesterol': (220.23648649, 4.88430934e+03), 'ecg': (0.64513514, 5.92069868e-01), 'max_hr': (138.75459459, 5.29172208e+02), 'exercise_ST_depression': (0.89532432, 1.11317517e+00) },
            'site_mappings': { 1: [['cleveland'], ['hungarian']], 2: [['cleveland'], ['switzerland']], 3: [['cleveland'], ['va']], 4: [['hungarian'], ['switzerland']], 5: [['hungarian'], ['va']], 6: [['switzerland'], ['va']] }
        },
        'samples_per_client': None,
        'metric': 'F1',
        'default_num_clients': 2,
        'max_clients': 2,
    },
    'ISIC': {
        'dataset_name': 'ISIC',
        'data_source': 'isic_paths',
        'partitioning_strategy': 'pre_split', # Uses cost as key for site_mappings
        # REMOVED: cost_interpretation
        'dataset_class': 'ISICDataset',
        'source_args': { # Loader uses cost_key directly to index site_mappings
            'site_mappings': { 0.06: [[2], [2]], 0.15: [[2], [0]], 0.19: [[2], [3]], 0.25: [[2], [1]], 0.3: [[1], [3]], 'all': [[0], [1], [2], [3]] },
            'image_size': 200
        },
        'transform_config': {},
        'partitioner_args': {},
        'samples_per_client': 2000,
        'fixed_classes': 8,
        'default_lr': 1e-3, 'learning_rates_try': [5e-3, 1e-3, 5e-4],
        'default_reg_param': 0.1, 'reg_params_try':[1, 0.1, 1e-2],
        'batch_size': 128, 'epochs_per_round': 1, 'rounds': 60, 'rounds_tune_inner': 15,
        'runs': 5, 'runs_tune': 1, 'metric': 'Balanced_accuracy', 'base_seed': 42,
        'default_num_clients': 2, 'max_clients': 4,
        'servers_tune_lr': ALGORITHMS, 'servers_tune_reg': [],
        'max_parallel_clients' : 2,
    },
    'IXITiny': {
        'dataset_name': 'IXITiny',
        'data_source': 'ixi_paths',
        'partitioning_strategy': 'pre_split', # Uses cost as key for site_mappings
        # REMOVED: cost_interpretation
        'dataset_class': 'IXITinyDataset',
        'source_args': { # Loader uses cost_key directly to index site_mappings
            'site_mappings': { 0.08: [['Guys'], ['HH']], 0.28: [['IOP'], ['Guys']], 0.30: [['IOP'], ['HH']], 'all': [['IOP'], ['HH'], ['Guys']] },
            'image_shape': (48, 60, 48)
        },
        'transform_config': {},
        'partitioner_args': {},
        'sampling_config': None,
        'fixed_classes': 2,
        'default_lr': 1e-3, 'learning_rates_try': [1e-2, 5e-3, 1e-3],
        'default_reg_param': 0.1, 'reg_params_try':[1, 0.1, 1e-2],
        'batch_size': 64,
        'epochs_per_round': 1, 'rounds': 50, 'rounds_tune_inner': 10,
        'runs': 10, 'runs_tune': 3, 'metric': 'DICE', 'base_seed': 42,
        'default_num_clients': 2, 'max_clients': 3,
        'servers_tune_lr': ALGORITHMS, 'servers_tune_reg': [],
        'max_parallel_clients' : 2,
    }
}

# --- Cost Parameters (Unchanged) ---
# These are the raw values used directly by loaders/partitioners
DATASET_COSTS = {
    # Alpha values (label skew) - Used directly by partition_dirichlet_indices
    'Synthetic_Label': [1000.0, 10.0, 2.0, 1.0, 0.75, 0.5, 0.2, 0.1],
    'Credit': [1000.0, 10.0, 2.0, 1.0, 0.75, 0.5, 0.2, 0.1],
    'EMNIST': [1000.0, 10.0, 2.0, 1.0, 0.75, 0.5, 0.2, 0.1],
    'CIFAR': [1000.0, 10.0, 2.0, 1.0, 0.75, 0.5, 0.2, 0.1],

    # Feature shift parameter (0=baseline) - Used directly by load_synthetic_raw
    'Synthetic_Feature': [0.0, 0.25, 0.5, 0.75, 1.0, 2.0, 5.0, 10.0],
    # Concept shift parameter (0=baseline) - Used directly by load_synthetic_raw
    'Synthetic_Concept': [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0],

    # Site mapping keys - Used directly by load_heart_raw, load_isic_paths_raw, load_ixi_paths_raw
    'IXITiny': [0.08, 0.28, 0.30, 'all'],
    'ISIC': [0.06, 0.15, 0.19, 0.25, 0.3, 'all'],
    'Heart': [1, 2, 3, 4, 5, 6]
}