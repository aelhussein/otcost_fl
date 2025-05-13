# configs.py
"""
Central configuration file for the project.
Defines directory paths, constants, default hyperparameters,
data handling configurations, algorithm settings.
Streamlined version.
"""
import os
import torch

# --- Core Directories ---
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT  = os.path.dirname(_CURRENT_DIR)
ROOT_DIR = _PROJECT_ROOT
DATA_DIR = os.path.join(ROOT_DIR, 'data')
RESULTS_DIR = os.path.join(ROOT_DIR, 'results_2')
MODEL_SAVE_DIR = os.path.join(ROOT_DIR, 'saved_models_2')
ACTIVATION_DIR = os.path.join(ROOT_DIR, 'activations')

# --- Global Settings ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
N_WORKERS = 4 # Use 0 for simplicity/debugging

# --- Supported Algorithms ---
ALGORITHMS = ['local', 'fedavg'] # Add others as implemented

# --- Supported Datasets ---
DATASETS = [
    'Synthetic_Label', 'Synthetic_Feature', 'Synthetic_Concept',
    'Credit', 'EMNIST', 'CIFAR', 'ISIC', 'IXITiny', 'Heart'
]

# --- Common Configuration for Tabular-like Datasets ---
COMMON_TABULAR_PARAMS = dict(
    fixed_classes=2, # Default for binary classification, override for multiclass
    default_lr=1e-3,
    learning_rates_try=[1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4],
    default_reg_param=0.1,
    reg_params_try=[1, 0.1, 0.01],
    batch_size=32,
    epochs_per_round=3,
    rounds=50,
    rounds_tune_inner=10,
    runs=20,
    runs_tune=3,
    metric='F1', # Default metric for tabular
    base_seed=42,
    samples_per_client=300,
    default_num_clients=5,
    # max_clients removed, num_target_clients from ExperimentConfig is primary
    servers_tune_lr=ALGORITHMS,
    servers_tune_reg=[],
    partitioner_args={},
    max_parallel_clients=None,
    use_weighted_loss=False, # If True, client should use WeightedCELoss if criterion is CE
    shift_after_split=False,
    activation_extractor_type='hook_based',
    criterion_type="CrossEntropyLoss", # Default criterion
    source_args={}, # For raw data loading parameters
)

# --- Common Configuration for Image Datasets ---
COMMON_IMAGE_PARAMS = dict(
    fixed_classes=10, # Default for common image datasets like CIFAR/EMNIST
    default_lr=3e-3,
    learning_rates_try=[5e-3, 1e-3, 5e-4],
    default_reg_param=0.1,
    reg_params_try=[1, 0.1, 1e-2],
    batch_size=96,
    epochs_per_round=3,
    rounds=60,
    rounds_tune_inner=20,
    runs=10,
    runs_tune=3,
    metric='Accuracy', # Default metric for image classification
    base_seed=42,
    default_num_clients=2,
    # max_clients removed
    servers_tune_lr=ALGORITHMS,
    servers_tune_reg=[],
    partitioner_args={},
    max_parallel_clients=None,
    use_weighted_loss=False, # If True, client should use WeightedCELoss if criterion is CE
    shift_after_split=True, # Often true for image datasets with augmentation-like shifts
    activation_extractor_type='hook_based',
    criterion_type="CrossEntropyLoss", # Default criterion
    source_args={}, # For raw data loading parameters
    # transform_config removed as transforms are handled within Dataset classes
)


# --- Default Hyperparameters & Data Handling Configuration ---
DEFAULT_PARAMS = {

    # === Unified Synthetic Configurations ===
    'Synthetic_Label': {
        **COMMON_TABULAR_PARAMS,
        'dataset_name': 'Synthetic_Label',
        'data_source': 'synthetic',
        'partitioning_strategy': 'dirichlet_indices',
        'dataset_class': 'SyntheticDataset',
        'source_args': {
            'base_n_samples': 50000,
            'n_features': 14,
            'label_noise': 0.1,
            'random_state': 42,
            'label_rule': 'mlp'
        },
        # criterion_type defaults to "CrossEntropyLoss"
    },
    'Synthetic_Feature': {
        **COMMON_TABULAR_PARAMS,
        'dataset_name': 'Synthetic_Feature',
        'data_source': 'synthetic',
        'partitioning_strategy': 'iid_indices',
        'shift_after_split': True,
        'dataset_class': 'SyntheticDataset',
        'source_args': {
            'base_n_samples': 50000,
            'n_features': 14,
            'label_noise': 0.01,
            'feature_shift_kind': 'mean',
            'feature_shift_cols': 14,
            'feature_shift_mu': 1.0,
            'label_rule': 'mlp',
        },
        # criterion_type defaults to "CrossEntropyLoss"
    },
    'Synthetic_Concept': {
        **COMMON_TABULAR_PARAMS,
        'dataset_name': 'Synthetic_Concept',
        'data_source': 'synthetic',
        'partitioning_strategy': 'iid_indices',
        'shift_after_split': True,
        'dataset_class': 'SyntheticDataset',
        'source_args': {
            'base_n_samples': 50000,
            'n_features': 14,
            'label_noise': 0.01,
            'random_state': 42,
            'concept_label_option': 'rotation',
            'concept_threshold_range_factor': 0.5,
            'label_rule': 'linear',
        },
        # criterion_type defaults to "CrossEntropyLoss"
    },

    # === Other Tabular Datasets ===
    'Credit': {
        **COMMON_TABULAR_PARAMS,
        'dataset_name': 'Credit',
        'data_source': 'credit_csv',
        'partitioning_strategy': 'iid_indices',
        'shift_after_split': True,
        'dataset_class': 'CreditDataset',
        'source_args': {
            'csv_path': os.path.join(DATA_DIR, 'Credit', 'creditcard.csv'),
            'drop_cols': ['Time'],
            'feature_shift_kind': 'mean',
            'feature_shift_cols': None,
            'feature_shift_mu': 1.0,
            'feature_shift_sigma': 1.5,
            'cols_percentage': 0.5,
        },
        'learning_rates_try': [5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4],
        # criterion_type defaults to "CrossEntropyLoss"
    },
    # === Image Datasets ===
    'CIFAR': {
        **COMMON_IMAGE_PARAMS,
        'dataset_name': 'CIFAR',
        'data_source': 'torchvision',
        'partitioning_strategy': 'iid_indices',
        'dataset_class': 'CIFARDataset',
        'source_args': {
            'dataset_name': 'CIFAR10',
            'feature_shift_kind': 'image',
            'max_rotation_angle': 45.0,
            'max_zoom': 0.3,
            'max_frequency': 1,
        },
        'samples_per_client': 5000,
        'batch_size': 512,
        'fixed_classes': 10, 
        # criterion_type defaults to "CrossEntropyLoss"
    },
    'EMNIST': {
        **COMMON_IMAGE_PARAMS,
        'dataset_name': 'EMNIST',
        'data_source': 'torchvision',
        'partitioning_strategy': 'iid_indices',
        'dataset_class': 'EMNISTDataset',
        'source_args': {
            'dataset_name': 'EMNIST',
            'split': 'digits',
            'feature_shift_kind': 'image',
            'max_rotation_angle': 60.0,
            'max_zoom': 0.5,
            'max_frequency': 2,
        },
        'samples_per_client': 1000,
        'fixed_classes': 10, # Already in COMMON_IMAGE_PARAMS, explicit here
        'batch_size': 64,
        # criterion_type defaults to "CrossEntropyLoss"
    },
    'ISIC': {
        **COMMON_IMAGE_PARAMS,
        'dataset_name': 'ISIC',
        'data_source': 'isic_paths',
        'partitioning_strategy': 'pre_split',
        'dataset_class': 'ISICDataset',
        'source_args': {
            'site_mappings': {'bcn_vmole': [[0], [1]], 'bcn_vmod': [[0], [2]], 'bcn_rose': [[0], [3]], 'bcn_msk': [[0], [4]], 'bcn_vienna': [[0], [5]],
                              'vmole_vmod': [[1], [2]], 'vmole_rose': [[1], [3]], 'vmole_msk': [[1], [4]], 'vmole_vienna': [[1], [5]], 'vmod_rose': [[2], [3]],
                              'vmod_msk': [[2], [4]], 'vmod_vienna': [[2], [5]], 'rose_msk': [[3], [4]], 'rose_vienna': [[3], [5]], 'msk_vienna': [[4], [5]]},
            'image_size': 200
        },
        'samples_per_client': 2000,
        'fixed_classes': 8,
        'runs': 5,
        'runs_tune': 1,
        'metric': 'Balanced_accuracy',
        # max_clients removed
        'use_weighted_loss': True, # This flag will be used by client
        'criterion_type': "ISICLoss", # Explicitly set criterion
        'batch_size': 128,
        'default_lr' : 1e-3, # Retained specific LR
    },
    'IXITiny': {
        # Not using COMMON_IMAGE_PARAMS as it's too different
        'dataset_name': 'IXITiny',
        'data_source': 'ixi_paths',
        'partitioning_strategy': 'pre_split',
        'dataset_class': 'IXITinyDataset',
        'source_args': {
            'site_mappings': { 'guys_hh': [['Guys'], ['HH']], 'iop_guys': [['IOP'], ['Guys']], 'iop_hh': [['IOP'], ['HH']], 'all': [['IOP'], ['HH'], ['Guys']] },
            'image_shape': (48, 60, 48)
        },
        'samples_per_client': None,
        'fixed_classes': None, # For segmentation, num_classes for OT might come from model output_channels
        'default_lr': 1e-3,
        'learning_rates_try': [1e-2, 5e-3, 1e-3],
        'default_reg_param': 0.1, # Kept for potential future algo use, though not by Dice
        'reg_params_try':[1, 0.1, 1e-2], # Kept for potential future algo use
        'batch_size': 4,
        'epochs_per_round': 3,
        'rounds': 30,
        'rounds_tune_inner': 10,
        'runs': 10,
        'runs_tune': 3,
        'metric': 'DICE',
        'base_seed': 42,
        'default_num_clients': 2,
        # max_clients removed
        'servers_tune_lr': ALGORITHMS,
        'servers_tune_reg': [],
        'max_parallel_clients' : 2,
        'use_weighted_loss': False, # Not applicable for Dice loss
        'shift_after_split': False, # Not applicable for pre-split segmentation
        'activation_extractor_type': 'rep_vector',
        'criterion_type': "DiceLoss", # Explicitly set criterion
        'partitioner_args':{}, # Explicitly add even if empty for consistency
    }
}

# --- Dataset Costs / Experiment Parameters ---
DATASET_COSTS = {
    'Synthetic_Label': [1000.0, 10.0, 2.0, 1.0, 0.75, 0.5, 0.2, 0.1],
    'Synthetic_Feature': [0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0],
    'Credit': [0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0],
    'EMNIST': [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0],
    'CIFAR': [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0],
    'Synthetic_Concept': [0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0],
    'IXITiny': ['guys_hh', 'iop_guys', 'iop_hh', 'all'], # Using string keys now
    'ISIC': ['bcn_vmole', 'bcn_vmod', 'bcn_rose', 'bcn_msk', 'bcn_vienna',
             'vmole_vmod', 'vmole_rose', 'vmole_msk', 'vmole_vienna','vmod_rose',
             'vmod_msk', 'vmod_vienna','rose_msk', 'rose_vienna','msk_vienna'], # Using string keys
}