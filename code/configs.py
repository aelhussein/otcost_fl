"""
Central configuration file for the project.

Defines directory paths, constants, default hyperparameters,
data handling configurations, algorithm settings, and imports common libraries.
"""
import os
import sys
import torch
import numpy as np # Keep basic imports if needed by config values themselves

# --- Core Directories ---
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT  = os.path.dirname(_CURRENT_DIR)
ROOT_DIR = _PROJECT_ROOT
DATA_DIR = os.path.join(ROOT_DIR, 'data') # Use os.path.join for cross-platform compatibility
RESULTS_DIR = os.path.join(ROOT_DIR, 'results')
ACTIVATION_DIR = os.path.join(ROOT_DIR, 'activations')
MODEL_SAVE_DIR = os.path.join(ROOT_DIR, 'saved_models')
EVAL_DIR = os.path.join(ROOT_DIR, 'code', 'evaluation')
OTCOST_DIR = os.path.join(ROOT_DIR, 'code', 'OTCost')

# --- Add project directories to Python path ---
# Allows importing modules from these directories
# (Ensure these are added in run.py or entry point, not necessarily needed here)
for local_path in [DATA_DIR, RESULTS_DIR, ACTIVATION_DIR, MODEL_SAVE_DIR, EVAL_DIR, OTCOST_DIR]:
    if local_path not in sys.path:
        sys.path.insert(0, local_path)

# --- Global Settings ---
# Check if torch is imported and CUDA is available
DEVICE = 'cuda' if 'torch' in sys.modules and torch.cuda.is_available() else 'cpu'
N_WORKERS = 1 # Default number of workers for DataLoader

# --- Supported Algorithms ---
ALGORITHMS = ['local', 'fedavg'] # Add others like 'fedprox', 'ditto' if implemented

# --- Supported Datasets (names must match keys in DEFAULT_PARAMS) ---
DATASETS = [
    'Synthetic',
    'Credit',
    'EMNIST',
    'CIFAR',
    'ISIC',
    'IXITiny',
    'Heart'
]

# --- Default Hyperparameters & Data Handling Configuration ---
# Structure: { DatasetName: { config_key: value, ... } }
DEFAULT_PARAMS = {

    'CIFAR': {
        # Data Handling Config
        'data_source': 'torchvision',                 # Key for DATA_LOADERS in data_loading.py
        'partitioning_strategy': 'dirichlet_indices', # Key for PARTITIONING_STRATEGIES
        'cost_interpretation': 'alpha',               # How to interpret values in DATASET_COSTS
        'dataset_class': 'CIFARDataset',              # Final wrapper class name in datasets.py
        'source_args': {'dataset_name': 'CIFAR10',    # Args passed to the data_source loader
                        'data_dir': DATA_DIR},
        'partitioner_args': {},                       # Extra args for the partitioner function
        'partition_scope': 'train',                   # Which part of loaded data to partition ('train' or 'all')
        'sampling_config': {'type': 'fixed_total',    # Config for subsampling *after* partitioning
                            'size': 3000, 'replace': False},
        'needs_preprocessing': [],                    # Preprocessing steps needed in DataPreprocessor (e.g., ['standard_scale'])

        # Model & Training Config
        'fixed_classes': 10,                          # Number of output classes for the model
        'default_lr': 1e-3,                           # Default learning rate if not tuned
        'learning_rates_try': [5e-3, 1e-3, 5e-4],     # LRs to try during tuning
        'default_reg_param': 1e-1,                    # Default regularization param (for Ditto, pFedMe etc.)
        'reg_params_try': [1, 1e-1, 1e-2, 1e-3],      # Reg params to try during tuning
        'batch_size': 128,
        'epochs_per_round': 1,                        # Local epochs per client per round
        'rounds': 100,                                # Total communication rounds for evaluation
        'runs': 10,                                   # Number of independent evaluation runs (with different seeds)
        'runs_tune': 3,                               # Number of independent tuning runs
        'metric': 'Accuracy',                         # Primary metric for reporting/selection
        'base_seed': 42,                              # Base seed for reproducibility

        # Experiment Specific Config
        'default_num_clients': 10,                    # Default clients if not specified via CLI
        'max_clients': None,                          # Max client limit for this dataset (None if no limit)
        'servers_tune_lr': ALGORITHMS,                # Servers to tune LR for (default: all)
        'servers_tune_reg': [],                       # Servers to tune Reg Param for (default: none)
    },

    'EMNIST': {
        # Data Handling Config
        'data_source': 'torchvision',
        'partitioning_strategy': 'dirichlet_indices',
        'cost_interpretation': 'alpha',
        'dataset_class': 'EMNISTDataset',
        'source_args': {'dataset_name': 'EMNIST', 'data_dir': DATA_DIR, 'split': 'digits'},
        'partitioner_args': {},
        'partition_scope': 'train',
        'sampling_config': {'type': 'fixed_total', 'size': 1000, 'replace': False},
        'needs_preprocessing': [],

        # Model & Training Config
        'fixed_classes': 10,
        'default_lr': 1e-3,
        'learning_rates_try': [5e-3, 1e-3, 5e-4, 1e-4],
        'default_reg_param': 1e-1,
        'reg_params_try': [1, 1e-1, 1e-2, 1e-3],
        'batch_size': 128,
        'epochs_per_round': 1,
        'rounds': 75,
        'runs': 10,
        'runs_tune': 3,
        'metric': 'Accuracy',
        'base_seed': 42,

        # Experiment Specific Config
        'default_num_clients': 10,
        'max_clients': None,
        'servers_tune_lr': ALGORITHMS,
        'servers_tune_reg': [],
    },

    'Synthetic': {
        # Data Handling Config
        'data_source': 'synthetic_base',
        'partitioning_strategy': 'dirichlet_indices',
        'cost_interpretation': 'alpha',
        'dataset_class': 'SyntheticDataset',
        'source_args': {
            'base_n_samples': 20000, # Total samples generated before partitioning
            'n_features': 10,
            'dist_type1': 'normal', # Base distribution type used by generator
            # 'dist_type2': 'skewed', # Currently unused in single dataset generation
            'label_noise': 0.05,
            'random_state': 42      # Seed specifically for data generation
        },
        'partitioner_args': {},
        'partition_scope': 'all',       # Partition the entire generated dataset
        'sampling_config': {'type': 'fixed_total', 'size': 1000, 'replace': False},
        'needs_preprocessing': ['standard_scale'], # Requires scaling after train/test split

        # Model & Training Config
        'fixed_classes': 2,
        'default_lr': 1e-3,
        'learning_rates_try': [5e-1, 1e-1, 5e-2, 1e-2, 5e-3, 1e-3],
        'default_reg_param': 1e-1,
        'reg_params_try': [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5],
        'batch_size': 64,
        'epochs_per_round': 1,
        'rounds': 100,
        'runs': 10, # Reduced from 20 for consistency example
        'runs_tune': 3, # Reduced from 10
        'metric': 'F1',
        'base_seed': 42,

        # Experiment Specific Config
        'default_num_clients': 10,
        'max_clients': None,
        'servers_tune_lr': ALGORITHMS,
        'servers_tune_reg': [],
    },

    'Credit': {
        # Data Handling Config
        'data_source': 'credit_base',
        'partitioning_strategy': 'dirichlet_indices',
        'cost_interpretation': 'alpha',
        'dataset_class': 'CreditDataset',
        'source_args': {'csv_path': os.path.join(DATA_DIR, 'Credit/creditcard.csv'),
                        'drop_cols': ['Time', 'Amount']},
        'partitioner_args': {},
        'partition_scope': 'all',
        'sampling_config': {'type': 'fixed_total', 'size': 1000, 'replace': False},
        'needs_preprocessing': ['standard_scale'],

        # Model & Training Config
        'fixed_classes': 2,
        'default_lr': 1e-3,
        'learning_rates_try': [5e-1, 1e-1, 5e-2, 1e-2, 5e-3, 1e-3],
        'default_reg_param': 1e-1,
        'reg_params_try': [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5],
        'batch_size': 64,
        'epochs_per_round': 1,
        'rounds': 100,
        'runs': 10, # Reduced from 20
        'runs_tune': 3, # Reduced from 10
        'metric': 'F1',
        'base_seed': 42,

        # Experiment Specific Config
        'default_num_clients': 10,
        'max_clients': None,
        'servers_tune_lr': ALGORITHMS,
        'servers_tune_reg': [],
    },

    'Heart': {
        # Data Handling Config
        'data_source': 'heart_site_loader', # Specific loader for heart sites
        'partitioning_strategy': 'pre_split', # Client assignment based on mapping
        'cost_interpretation': 'site_mapping_key', # Cost value is key to site_mappings
        'dataset_class': 'HeartDataset',
        'source_args': {
            'data_dir': os.path.join(DATA_DIR, 'Heart'),
            'sites': ['cleveland', 'hungarian', 'switzerland', 'va'],
            'used_columns': ['age', 'sex', 'chest_pain_type', 'resting_bp', 'cholesterol', 'sugar', 'ecg', 'max_hr', 'exercise_angina', 'exercise_ST_depression', 'target'],
            'feature_names': ['age', 'sex', 'chest_pain_type', 'resting_bp', 'cholesterol', 'sugar', 'ecg', 'max_hr', 'exercise_angina', 'exercise_ST_depression'],
            'cols_to_scale': ['age', 'chest_pain_type', 'resting_bp', 'cholesterol', 'ecg', 'max_hr', 'exercise_ST_depression'],
            'scale_values': { # Pre-computed global mean/variance for scaling
                'age': (53.0872973, 7.01459463e+01), 'chest_pain_type': (3.23702703, 8.17756772e-01),
                'resting_bp': (132.74405405, 3.45493057e+02), 'cholesterol': (220.23648649, 4.88430934e+03),
                'ecg': (0.64513514, 5.92069868e-01), 'max_hr': (138.75459459, 5.29172208e+02),
                'exercise_ST_depression': (0.89532432, 1.11317517e+00)
            },
            'site_mappings': { # Maps cost key (1-6) to pairs of sites for 2 clients
                1: [['cleveland'], ['hungarian']], 2: [['cleveland'], ['switzerland']],
                3: [['cleveland'], ['va']], 4: [['hungarian'], ['switzerland']],
                5: [['hungarian'], ['va']], 6: [['switzerland'], ['va']]
                # Add 'all': [['cleveland'], ['hungarian'], ['switzerland'], ['va']] if needed
            }
        },
        'partitioner_args': {},
        'partition_scope': 'all', # Not strictly needed for pre_split but harmless
        'sampling_config': None, # No subsampling applied within site files
        'needs_preprocessing': [], # Scaling handled *inside* heart_site_loader

        # Model & Training Config
        'fixed_classes': 2,
        'default_lr': 1e-3,
        'learning_rates_try': [5e-1, 1e-1, 5e-2, 1e-2, 5e-3, 1e-3],
        'default_reg_param': 1e-1,
        'reg_params_try': [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5],
        'batch_size': 64,
        'epochs_per_round': 1,
        'rounds': 100,
        'runs': 10, # Reduced from 100
        'runs_tune': 3, # Reduced from 10
        'metric': 'F1',
        'base_seed': 42,

        # Experiment Specific Config
        'default_num_clients': 2, # Hardcoded for pairwise site structure
        'max_clients': 2,         # Cannot exceed 2 with current site_mappings
        'servers_tune_lr': ALGORITHMS,
        'servers_tune_reg': [],
    },

    'ISIC': {
        # Data Handling Config
        'data_source': 'pre_split_paths_isic',
        'partitioning_strategy': 'pre_split',
        'cost_interpretation': 'site_mapping_key',
        'dataset_class': 'ISICDataset',
        'source_args': {
            'data_dir': os.path.join(DATA_DIR, 'ISIC'),
            'site_mappings': { # Maps cost key (float/str) to list of site indices lists per client
                 0.06: [[2], [2]], 0.15: [[2], [0]], 0.19: [[2], [3]],
                 0.25: [[2], [1]], 0.3: [[1], [3]], 'all': [[0], [1], [2], [3]]
             }
        },
        'partitioner_args': {},
        'partition_scope': 'all',
        'sampling_config': {'type': 'fixed_total', 'size': 2000}, # Sample N rows from each site CSV
        'needs_preprocessing': [], # Transforms handled in ISICDataset

        # Model & Training Config
        'fixed_classes': 8,
        'default_lr': 1e-3,
        'learning_rates_try': [5e-3, 1e-3, 5e-4, 1e-4],
        'default_reg_param': 1e-1,
        'reg_params_try': [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5],
        'batch_size': 128,
        'epochs_per_round': 1,
        'rounds': 60,
        'runs': 5,
        'runs_tune': 1,
        'metric': 'Balanced_accuracy',
        'base_seed': 42,

        # Experiment Specific Config
        'default_num_clients': 2, # Used if cost key maps to >= 2 sites
        'max_clients': 4,         # Max based on 'all' mapping
        'servers_tune_lr': ALGORITHMS,
        'servers_tune_reg': [],
    },

    'IXITiny': {
        # Data Handling Config
        'data_source': 'pre_split_paths_ixi',
        'partitioning_strategy': 'pre_split',
        'cost_interpretation': 'site_mapping_key',
        'dataset_class': 'IXITinyDataset',
        'source_args': {
            'data_dir': os.path.join(DATA_DIR, 'IXITiny'),
            'site_mappings': { # Maps cost key (float/str) to list of site name lists per client
                 0.08: [['Guys'], ['HH']], 0.28: [['IOP'], ['Guys']],
                 0.30: [['IOP'], ['HH']], 'all': [['IOP'], ['HH'], ['Guys']]
             }
        },
        'partitioner_args': {},
        'partition_scope': 'all',
        'sampling_config': None, # No sampling defined
        'needs_preprocessing': [], # MONAI transforms handled in IXITinyDataset

        # Model & Training Config
        'fixed_classes': 2, # Background vs Foreground
        'default_lr': 1e-3,
        'learning_rates_try': [1e-2, 5e-3, 1e-3, 5e-4],
        'default_reg_param': 1e-1,
        'reg_params_try': [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5],
        'batch_size': 64, # Reduced batch size may be needed for 3D data
        'epochs_per_round': 1,
        'rounds': 50,
        'runs': 10,
        'runs_tune': 3,
        'metric': 'DICE', # Specific metric for segmentation
        'base_seed': 42,

        # Experiment Specific Config
        'default_num_clients': 2, # Used if cost key maps to >= 2 sites
        'max_clients': 3,         # Max based on 'all' mapping
        'servers_tune_lr': ALGORITHMS,
        'servers_tune_reg': [],
    }
}

# --- Cost Parameters ---
# Define the set of 'cost' or heterogeneity parameters to iterate over for each dataset.
# The *meaning* of these values depends on 'cost_interpretation' in DEFAULT_PARAMS.
DATASET_COSTS = {
    # Site mapping keys for pre-split path datasets
    'IXITiny': [0.08, 0.28, 0.30, 'all'],
    'ISIC': [0.06, 0.15, 0.19, 0.25, 0.3, 'all'],
    # Alpha values for Dirichlet partitioned datasets (lower alpha = more non-IID)
    'EMNIST': [0.1, 0.5, 1.0, 5.0, 10.0, 1000.0], # Added 1000 for near-IID check
    'CIFAR': [0.1, 0.5, 1.0, 5.0, 10.0, 1000.0],
    'Synthetic': [0.1, 0.3, 0.5, 1.0, 5.0, 10.0, 1000.0],
    'Credit': [0.1, 0.3, 0.5, 1.0, 5.0, 10.0, 1000.0],
    # Site mapping keys for Heart (represent specific site pairings)
    'Heart': [1, 2, 3, 4, 5, 6]
}