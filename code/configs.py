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
    epochs_per_round=3,
    rounds=25,
    rounds_tune_inner=20,
    runs=20, # Number of evaluation runs
    runs_tune=3, # Number of tuning runs
    metric='F1', # Default metric
    base_seed=42,
    #samples_per_client = 10000,
    samples_per_client = 300,
    default_num_clients=5,
    max_clients=None,
    servers_tune_lr=ALGORITHMS,
    servers_tune_reg=[], # Tune Reg only for specified algos
    partitioner_args={}, # Extra args for partitioner
    max_parallel_clients=None,
    use_weighted_loss=False,
    shift_after_split=False, # Default: no shift after splitting
)

# --- Common Configuration for Image Datasets ---
COMMON_IMAGE_PARAMS = dict(
    default_lr=3e-3,
    learning_rates_try=[1e-2, 5e-3, 1e-3, 5e-4],
    default_reg_param=0.1,
    reg_params_try=[1, 0.1, 1e-2],
    batch_size=96,
    epochs_per_round=3,
    rounds=40,
    rounds_tune_inner=20,
    runs=10,
    runs_tune=3,
    metric='Accuracy',
    base_seed=42,
    default_num_clients=2,
    max_clients=None,
    servers_tune_lr=ALGORITHMS,
    servers_tune_reg=[],
    partitioner_args={},
    max_parallel_clients=None,
    use_weighted_loss=False,
    shift_after_split=True, # Default: no shift after splitting
    source_args={}, # Placeholder for dataset specific args
    transform_config={}, # Placeholder
)


# --- Default Hyperparameters & Data Handling Configuration ---
DEFAULT_PARAMS = {

    # === Unified Synthetic Configurations ===
    'Synthetic_Label': {
        **COMMON_TABULAR_PARAMS,
        'dataset_name': 'Synthetic_Label',
        'data_source': 'synthetic', # Unified loader key
        'partitioning_strategy': 'dirichlet_indices', # Needs alpha passed directly
        'dataset_class': 'SyntheticDataset',
        'source_args': { # Params for base generation (mode='baseline')
            'base_n_samples': 50000,
            'n_features': 14,
            'label_noise': 0.1,
            'random_state': 42,
            'label_rule': 'mlp'
        },
        'metric': 'F1',
    },
    'Synthetic_Feature': {
        **COMMON_TABULAR_PARAMS,
        'dataset_name': 'Synthetic_Feature',
        'data_source': 'synthetic',
        'partitioning_strategy': 'iid_indices', # Shift in features after data creation
        'shift_after_split': True, # Shift after partitioning
        'dataset_class': 'SyntheticDataset',
        'source_args': { # Params for per-client generation (mode='feature_shift')
            'base_n_samples': 50000,
            'n_features': 14,
            'label_noise': 0.01,
            # Feature shift specific config (passed as **kwargs to generator)
            'feature_shift_kind': 'mean', # 'mean', 'scale', 'tilt'
            'feature_shift_cols': 14,
            'feature_shift_mu': 1.0,
            'label_rule': 'mlp',
        },
    },
    'Synthetic_Concept': {
        **COMMON_TABULAR_PARAMS,
        'dataset_name': 'Synthetic_Concept',
        'data_source': 'synthetic',
        'partitioning_strategy': 'iid_indices', # Partition baseline data IID and then add shift
        'shift_after_split': True,
        'dataset_class': 'SyntheticDataset',
        'source_args': { # Params for base generation + concept shift (mode='concept_shift')
            'base_n_samples': 50000,
            'n_features': 14,
            'label_noise': 0.01,
            'random_state': 42,
            # Concept shift specific config (passed as **kwargs to generator)
            'concept_label_option': 'rotation', # 'threshold', 'rotation'
            'concept_threshold_range_factor': 0.5,
            'label_rule': 'linear',
        },
    },

    # === Other Tabular Datasets ===
    'Credit': {
        **COMMON_TABULAR_PARAMS,
        'dataset_name': 'Credit',
        'data_source': 'credit_csv',
        'partitioning_strategy': 'iid_indices',  # Changed from dirichlet to iid for feature shift
        'shift_after_split': True,  # Enable feature shifts
        'dataset_class': 'CreditDataset',
        'source_args': {
            'csv_path': os.path.join(DATA_DIR, 'Credit', 'creditcard.csv'),
            'drop_cols': ['Time'],
            # Feature shift specific config (similar to Synthetic_Feature)
            'feature_shift_kind': 'mean',
            'feature_shift_cols': None,  # Will be calculated as percentage of total features
            'feature_shift_mu': 1.0,
            'feature_shift_sigma': 1.5,
            'cols_percentage': 0.5,  # Additional parameter specific to Credit
        },
        'metric': 'F1',
        'learning_rates_try': [5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4],
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
            'site_mappings': { 1: [['cleveland'], ['hungarian']], 2: [['cleveland'], ['switzerland']], 3: [['cleveland'], ['va']], 4: [['hungarian'], ['switzerland']], 5: [['hungarian'], ['va']], 6: [['switzerland'], ['va']], 'all': [['cleveland'], ['hungarian'], ['switzerland'], ['va']] },
        },
        'samples_per_client': None,
        'metric': 'F1',
        'default_num_clients': 2,
        'max_clients': 2,
        'use_weighted_loss': True,
    },

    # === Image Datasets (NEW Rotation Shift) ===
    'CIFAR': {
        **COMMON_IMAGE_PARAMS,
        'dataset_name': 'CIFAR',
        'data_source': 'torchvision',
        'partitioning_strategy': 'iid_indices', # Use IID for pure rotation shift
        'shift_after_split': True,              # Enable shift application
        'dataset_class': 'CIFARDataset',        # Use the same dataset class
        'source_args': {
            'dataset_name': 'CIFAR10',
            'feature_shift_kind': 'image',   # Identify the shift type
            'max_rotation_angle': 45.0,        # Max angle at delta=1
            'max_zoom': 0.25,
            'max_frequency': 1,
        },
        'samples_per_client': 3000,
        'batch_size': 64,
        'fixed_classes': 10,
    },
    'EMNIST': {
        **COMMON_IMAGE_PARAMS,
        'dataset_name': 'EMNIST',
        'data_source': 'torchvision',
        'partitioning_strategy': 'iid_indices', # Use IID for pure rotation shift
        'shift_after_split': True,              # Enable shift application
        'dataset_class': 'EMNISTDataset',       # Use the same dataset class
        'source_args': {
            'dataset_name': 'EMNIST',
            'split': 'digits',
            'feature_shift_kind': 'image',   # Identify the shift type
            'max_rotation_angle': 60.0,        # Max angle at delta=1
            'max_zoom': 0.5,
            'max_frequency': 2,
        },
        'samples_per_client': 1000,
        'fixed_classes': 10,
        'batch_size': 64,
    },

    # === Image Datasets (Pre-Split) ===
    'ISIC': {
        **COMMON_IMAGE_PARAMS,
        'dataset_name': 'ISIC',
        'data_source': 'isic_paths',
        'partitioning_strategy': 'pre_split',
        'dataset_class': 'ISICDataset',
        'source_args': {
            'site_mappings': {0.15: [[2], [0]], 0.19: [[2], [3]], 0.25: [[2], [1]], 0.3: [[1], [3]], 'all': [[0], [1], [2], [3]] },
            'image_size': 200
        },
        'samples_per_client': 2000,
        'fixed_classes': 8,
        'runs': 5,
        'runs_tune': 1,
        'metric': 'Balanced_accuracy',
        'max_clients': 4,
        'use_weighted_loss': True,
        'batch_size': 128,
    },
    'IXITiny': {
        'dataset_name': 'IXITiny',
        'data_source': 'ixi_paths',
        'partitioning_strategy': 'pre_split',
        'dataset_class': 'IXITinyDataset',
        'source_args': {
            'site_mappings': { 0.08: [['Guys'], ['HH']], 0.28: [['IOP'], ['Guys']], 0.30: [['IOP'], ['HH']], 'all': [['IOP'], ['HH'], ['Guys']] },
            'image_shape': (48, 60, 48)
        },
        'transform_config': {}, # Handled internally by IXITinyDataset
        'partitioner_args': {},
        'samples_per_client': None, # Uses all available data per site
        'fixed_classes': 2,
        'default_lr': 1e-3, 'learning_rates_try': [1e-2, 5e-3, 1e-3],
        'default_reg_param': 0.1, 'reg_params_try':[1, 0.1, 1e-2],
        'batch_size': 4,
        'epochs_per_round': 1, 'rounds': 50, 'rounds_tune_inner': 10,
        'runs': 10, 'runs_tune': 3, 'metric': 'DICE', 'base_seed': 42,
        'default_num_clients': 2, 'max_clients': 3,
        'servers_tune_lr': ALGORITHMS, 'servers_tune_reg': [],
        'max_parallel_clients' : 2,
        'use_weighted_loss': False, # DICE loss
    }
}

DATASET_COSTS = {
    # Alpha values (label skew) - Used directly by partition_dirichlet_indices
    'Synthetic_Label': [1000.0, 10.0, 2.0, 1.0, 0.75, 0.5, 0.2, 0.1],
    #'Credit': [1000.0, 10.0, 2.0, 1.0, 0.75, 0.5, 0.2, 0.1],
    #'EMNIST': [1000.0, 10.0, 2.0, 1.0, 0.75, 0.5, 0.2, 0.1],
    #'CIFAR': [1000.0, 10.0, 2.0, 1.0, 0.75, 0.5, 0.2, 0.1],

    # Feature shift parameter (0=baseline) - Used directly by load_synthetic_raw
    'Synthetic_Feature': [0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0],
    'Credit': [0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0],
    'EMNIST': [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0],
    #'CIFAR': [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0],
    'CIFAR': [0, 0.3, 0.5, 0.7, 0.9],
    # Concept shift parameter (0=baseline) - Used directly by load_synthetic_raw
    'Synthetic_Concept': [0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0],

    # Site mapping keys - Used directly by load_heart_raw, load_isic_paths_raw, load_ixi_paths_raw
    'IXITiny': [0.08, 0.28, 0.30, 'all'],
    'ISIC': [0.15, 0.19, 0.25, 0.3, 'all'],
    'Heart': [1, 2, 3, 4, 5, 6]
}