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
DATA_DIR = os.path.join(ROOT_DIR, 'data')
RESULTS_DIR = os.path.join(ROOT_DIR, 'results')
MODEL_SAVE_DIR = os.path.join(ROOT_DIR, 'saved_models')
EVAL_DIR = os.path.join(ROOT_DIR, 'code', 'evaluation') # Keep if needed
OTCOST_DIR = os.path.join(ROOT_DIR, 'code', 'OTCost')   # Keep if needed
ACTIVATION_DIR = os.path.join(ROOT_DIR, 'activations') # Keep if needed

# --- Add project directories to Python path ---
# Best practice: Add these in the main entry script (e.g., run.py)
# for local_path in [DATA_DIR, RESULTS_DIR, ACTIVATION_DIR, MODEL_SAVE_DIR, EVAL_DIR, OTCOST_DIR]:
#     if local_path not in sys.path:
#         sys.path.insert(0, local_path)

# --- Global Settings ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
N_WORKERS = 1 # Default number of workers for DataLoader

# --- Supported Algorithms ---
ALGORITHMS = ['local', 'fedavg'] # Add others like 'fedprox', 'ditto' if implemented

# --- Supported Datasets (names must match keys in DEFAULT_PARAMS) ---
DATASETS = [
    'Synthetic_Label',
    'Synthetic_Feature',
    'Synthetic_Concept',
    'Credit',
    'EMNIST',
    'CIFAR',
    'ISIC',
    'IXITiny',
    'Heart'
]

# --- NEW: Common Configuration for Synthetic Datasets ---
COMMON_SYNTH_PARAMS = dict(
    fixed_classes=2,
    default_lr=1e-3,
    learning_rates_try=[1e-2, 1e-3, 5e-4],
    default_reg_param=0.1,
    reg_params_try=[1, 0.1, 0.01],
    batch_size=64,
    epochs_per_round=1,
    rounds=100,
    rounds_tune_inner=20, # Define explicitly for tuning if different
    runs=10,
    runs_tune=3,
    metric='Accuracy', # Or use F1 if needed
    base_seed=42,
    default_num_clients=10,
    max_clients=None,
    needs_preprocessing=['standard_scale'],
    servers_tune_lr=ALGORITHMS,
    servers_tune_reg=[], # Typically only tune reg for specific algos
    sampling_config={'type': 'fixed_total', 'size': 1000, 'replace': False},
    partition_scope='all', # Default partition scope
    partitioner_args={}, # Default empty partitioner args
)

# --- Default Hyperparameters & Data Handling Configuration ---
# Structure: { DatasetName: { config_key: value, ... } }
DEFAULT_PARAMS = {

    # === NEW SYNTHETIC CONFIGURATIONS (Using COMMON_SYNTH_PARAMS) ===
    'Synthetic_Label': {
        **COMMON_SYNTH_PARAMS, # Inherit common settings
        'data_source': 'synthetic_base',
        'partitioning_strategy': 'dirichlet_indices',
        'cost_interpretation': 'alpha',
        'dataset_class': 'SyntheticDataset', # From data_sets.py
        'source_args': {
            'base_n_samples': 20000, # Total samples for base dataset
            'n_features': 10,
            'dist_type1': 'mixed',
            'label_noise': 0.05,
            'random_state': 42 # Seed for base generation
        },
        # Override specific common settings if needed
        # 'metric': 'F1',
    },
    'Synthetic_Feature': {
        **COMMON_SYNTH_PARAMS,
        'data_source': 'synthetic_feature_skew_client',
        'partitioning_strategy': 'pre_split',
        'cost_interpretation': 'feature_shift_param',
        'dataset_class': 'SyntheticDataset', # From data_sets.py
        'source_args': {
            'n_samples_per_client': 1000, # Samples generated *per client*
            'n_features': 10,
            'label_noise': 0.05,
            'shift_mapping': {
                 'param_to_vary': 'dist_type',
                 'base_value': 'mixed',
                 'shifted_values': ['normal', 'skewed'], # Options corresponding to shift 0..1
            },
            'base_random_state': 42 # Base seed for client generation
        },
        'sampling_config': None, # No sampling after generation per client
        # Potentially fit scaler globally? Needs thought. For now, scale per client.
        # 'needs_preprocessing': [], # Or handle scaling carefully
    },
    'Synthetic_Concept': {
        **COMMON_SYNTH_PARAMS,
        'data_source': 'synthetic_features_only', # Load base features
        'partitioning_strategy': 'iid_indices_no_labels', # Partition feature indices
        'cost_interpretation': 'concept_shift_param',
        'dataset_class': 'SyntheticConceptDataset', # From data_sets.py
        'source_args': {
            'base_n_samples': 20000, # Total samples for base feature dataset
            'n_features': 10,
            'dist_type1': 'mixed', # Feature distribution type
            'random_state': 42, # Seed for base generation
            'concept_mapping': {
                 'function_type': 'linear_threshold_shift', # How concept changes
                 'base_param': 0.5,       # Parameter value when shift=0.5
                 'shift_range': 0.8      # Total range of parameter variation (e.g., 0.5 +/- 0.4)
            }
        },
        # Sampling applies to the *indices* after partitioning
        'sampling_config': {'type': 'fixed_total', 'size': 1000, 'replace': False},
    },

    # === EXISTING DATASET CONFIGS (Update to new structure if desired) ===
    'Credit': {
        'data_source': 'credit_base',
        'partitioning_strategy': 'dirichlet_indices',
        'cost_interpretation': 'alpha',
        'dataset_class': 'CreditDataset', # From data_sets.py
        'source_args': {'csv_path': os.path.join(DATA_DIR, 'Credit/creditcard.csv'),
                        'drop_cols': ['Time']}, # Keep Amount for scaling?
        'partitioner_args': {},
        'partition_scope': 'all',
        'sampling_config': {'type': 'fixed_total', 'size': 1000, 'replace': False},
        'needs_preprocessing': ['standard_scale'],
        'fixed_classes': 2,
        'default_lr': 1e-3, 'learning_rates_try': [5e-1, 1e-1, 5e-2, 1e-2, 5e-3, 1e-3],
        'default_reg_param': 1e-1, 'reg_params_try': [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5],
        'batch_size': 64, 'epochs_per_round': 1, 'rounds': 100, 'rounds_tune_inner': 20,
        'runs': 10, 'runs_tune': 3, 'metric': 'F1', 'base_seed': 42,
        'default_num_clients': 10, 'max_clients': None,
        'servers_tune_lr': ALGORITHMS, 'servers_tune_reg': [],
    },
    'CIFAR': {
        'data_source': 'torchvision',
        'partitioning_strategy': 'dirichlet_indices',
        'cost_interpretation': 'alpha',
        'dataset_class': 'CIFARDataset', # From data_sets.py
        'source_args': {'dataset_name': 'CIFAR10', 'data_dir': DATA_DIR},
        # NEW: Placeholder for transform config keys
        'transform_config': {'train': None, 'test': None}, # None means use default in loader
        'partitioner_args': {},
        'partition_scope': 'train',
        'sampling_config': {'type': 'fixed_total', 'size': 3000, 'replace': False},
        'needs_preprocessing': [],
        'fixed_classes': 10,
        'default_lr': 1e-3, 'learning_rates_try': [5e-3, 1e-3, 5e-4],
        'default_reg_param': 1e-1, 'reg_params_try':[1, 1e-1, 1e-2, 1e-3],
        'batch_size': 128, 'epochs_per_round': 1, 'rounds': 100, 'rounds_tune_inner': 20,
        'runs': 10, 'runs_tune': 3, 'metric': 'Accuracy', 'base_seed': 42,
        'default_num_clients': 10, 'max_clients': None,
        'servers_tune_lr': ALGORITHMS, 'servers_tune_reg': [],
    },
    'EMNIST': {
        'data_source': 'torchvision',
        'partitioning_strategy': 'dirichlet_indices',
        'cost_interpretation': 'alpha',
        'dataset_class': 'EMNISTDataset', # From data_sets.py
        'source_args': {'dataset_name': 'EMNIST', 'data_dir': DATA_DIR, 'split': 'digits'},
        'transform_config': {'train': None, 'test': None},
        'partitioner_args': {},
        'partition_scope': 'train',
        'sampling_config': {'type': 'fixed_total', 'size': 1000, 'replace': False},
        'needs_preprocessing': [],
        'fixed_classes': 10,
        'default_lr': 1e-3, 'learning_rates_try': [5e-3, 1e-3, 5e-4, 1e-4],
        'default_reg_param': 1e-1, 'reg_params_try':[1, 1e-1, 1e-2, 1e-3],
        'batch_size': 128, 'epochs_per_round': 1, 'rounds': 75, 'rounds_tune_inner': 15,
        'runs': 10, 'runs_tune': 3, 'metric': 'Accuracy', 'base_seed': 42,
        'default_num_clients': 10, 'max_clients': None,
        'servers_tune_lr': ALGORITHMS, 'servers_tune_reg': [],
    },
    'Heart': {
        'data_source': 'heart_site_loader',
        'partitioning_strategy': 'pre_split',
        'cost_interpretation': 'site_mapping_key',
        'dataset_class': 'HeartDataset', # From data_sets.py
        'source_args': {
            'data_dir': os.path.join(DATA_DIR, 'Heart'),
            'sites': ['cleveland', 'hungarian', 'switzerland', 'va'],
            'used_columns': ['age', 'sex', 'chest_pain_type', 'resting_bp', 'cholesterol', 'sugar', 'ecg', 'max_hr', 'exercise_angina', 'exercise_ST_depression', 'target'],
            'feature_names': ['age', 'sex', 'chest_pain_type', 'resting_bp', 'cholesterol', 'sugar', 'ecg', 'max_hr', 'exercise_angina', 'exercise_ST_depression'],
            'cols_to_scale': ['age', 'chest_pain_type', 'resting_bp', 'cholesterol', 'ecg', 'max_hr', 'exercise_ST_depression'],
            'scale_values': { 'age': (53.0872973, 7.01459463e+01), 'chest_pain_type': (3.23702703, 8.17756772e-01), 'resting_bp': (132.74405405, 3.45493057e+02), 'cholesterol': (220.23648649, 4.88430934e+03), 'ecg': (0.64513514, 5.92069868e-01), 'max_hr': (138.75459459, 5.29172208e+02), 'exercise_ST_depression': (0.89532432, 1.11317517e+00) },
            'site_mappings': { 1: [['cleveland'], ['hungarian']], 2: [['cleveland'], ['switzerland']], 3: [['cleveland'], ['va']], 4: [['hungarian'], ['switzerland']], 5: [['hungarian'], ['va']], 6: [['switzerland'], ['va']] }
        },
        'partitioner_args': {},
        'partition_scope': 'all',
        'sampling_config': None,
        'needs_preprocessing': [], # Scaling handled in loader
        'fixed_classes': 2,
        'default_lr': 1e-3, 'learning_rates_try':[5e-1, 1e-1, 5e-2, 1e-2, 5e-3, 1e-3],
        'default_reg_param': 1e-1, 'reg_params_try':[1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5],
        'batch_size': 64, 'epochs_per_round': 1, 'rounds': 100, 'rounds_tune_inner': 20,
        'runs': 10, 'runs_tune': 3, 'metric': 'F1', 'base_seed': 42,
        'default_num_clients': 2, 'max_clients': 2,
        'servers_tune_lr': ALGORITHMS, 'servers_tune_reg': [],
    },
    'ISIC': {
        'data_source': 'pre_split_paths_isic',
        'partitioning_strategy': 'pre_split',
        'cost_interpretation': 'site_mapping_key',
        'dataset_class': 'ISICDataset', # From data_sets.py
        'source_args': {
            'data_dir': os.path.join(DATA_DIR, 'ISIC'),
            'site_mappings': { 0.06: [[2], [2]], 0.15: [[2], [0]], 0.19: [[2], [3]], 0.25: [[2], [1]], 0.3: [[1], [3]], 'all': [[0], [1], [2], [3]] }
        },
        'transform_config': {'train': None, 'test': None}, # Placeholder if needed
        'partitioner_args': {},
        'partition_scope': 'all',
        'sampling_config': {'type': 'fixed_total', 'size': 2000},
        'needs_preprocessing': [],
        'fixed_classes': 8,
        'default_lr': 1e-3, 'learning_rates_try': [5e-3, 1e-3, 5e-4, 1e-4],
        'default_reg_param': 1e-1, 'reg_params_try':[1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5],
        'batch_size': 128, 'epochs_per_round': 1, 'rounds': 60, 'rounds_tune_inner': 15,
        'runs': 5, 'runs_tune': 1, 'metric': 'Balanced_accuracy', 'base_seed': 42,
        'default_num_clients': 2, 'max_clients': 4,
        'servers_tune_lr': ALGORITHMS, 'servers_tune_reg': [],
    },
    'IXITiny': {
        'data_source': 'pre_split_paths_ixi',
        'partitioning_strategy': 'pre_split',
        'cost_interpretation': 'site_mapping_key',
        'dataset_class': 'IXITinyDataset', # From data_sets.py
        'source_args': {
            'data_dir': os.path.join(DATA_DIR, 'IXITiny'),
            'site_mappings': { 0.08: [['Guys'], ['HH']], 0.28: [['IOP'], ['Guys']], 0.30: [['IOP'], ['HH']], 'all': [['IOP'], ['HH'], ['Guys']] }
        },
        'transform_config': {'train': None, 'test': None}, # Placeholder if needed
        'partitioner_args': {},
        'partition_scope': 'all',
        'sampling_config': None,
        'needs_preprocessing': [],
        'fixed_classes': 2, # Background + Foreground
        'default_lr': 1e-3, 'learning_rates_try': [1e-2, 5e-3, 1e-3, 5e-4],
        'default_reg_param': 1e-1, 'reg_params_try':[1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5],
        'batch_size': 64, # Adjust based on memory
        'epochs_per_round': 1, 'rounds': 50, 'rounds_tune_inner': 10,
        'runs': 10, 'runs_tune': 3, 'metric': 'DICE', 'base_seed': 42,
        'default_num_clients': 2, 'max_clients': 3,
        'servers_tune_lr': ALGORITHMS, 'servers_tune_reg': [],
    }
}

# --- Cost Parameters ---
DATASET_COSTS = {
    # Alpha values (label skew)
    'Synthetic_Label': [1000.0, 10.0, 1.0, 0.5, 0.1],
    'Credit': [1000.0, 10.0, 1.0, 0.5, 0.1],
    'EMNIST': [1000.0, 10.0, 1.0, 0.5, 0.1],
    'CIFAR': [1000.0, 10.0, 1.0, 0.5, 0.1],

    # Feature shift parameter (0=identical, 1=max difference)
    'Synthetic_Feature': [0.0, 0.25, 0.5, 0.75, 1.0],

    # Concept shift parameter (0=identical labeling, 1=max difference)
    'Synthetic_Concept': [0.0, 0.25, 0.5, 0.75, 1.0],

    # Site mapping keys
    'IXITiny': [0.08, 0.28, 0.30, 'all'],
    'ISIC': [0.06, 0.15, 0.19, 0.25, 0.3, 'all'],
    'Heart': [1, 2, 3, 4, 5, 6]
}