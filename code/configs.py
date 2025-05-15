# configs.py
"""
Central configuration file for the project.
Defines directory paths, constants, default hyperparameters,
data handling configurations, algorithm settings.
Streamlined version.
"""
import os
import torch

# --- Global Settings ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
if DEVICE == 'cpu':
    torch.set_num_threads(1)
N_WORKERS = 4 
SELECTION_CRITERION_KEY = None

# --- Core Directories ---
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT  = os.path.dirname(_CURRENT_DIR)
ROOT_DIR = _PROJECT_ROOT
DATA_DIR = os.path.join(ROOT_DIR, 'data')

# Default directories (will be updated by configure_paths)
RESULTS_DIR = os.path.join(ROOT_DIR, 'results_loss')
MODEL_SAVE_DIR = os.path.join(ROOT_DIR, 'saved_models')
ACTIVATION_DIR = os.path.join(ROOT_DIR, 'activations')

# Function to configure paths based on metric
def configure_paths(metric='score'):
    """Configure directory paths based on the specified metric."""
    global RESULTS_DIR, MODEL_SAVE_DIR, ACTIVATION_DIR, SELECTION_CRITERION_KEY, DEFAULT_PARAMS
    
    # Set directories based on metric
    if metric == 'loss':
        RESULTS_DIR = os.path.join(ROOT_DIR, 'results_loss')
        MODEL_SAVE_DIR = os.path.join(ROOT_DIR, 'saved_models_loss')
        ACTIVATION_DIR = os.path.join(ROOT_DIR, 'activations_loss')
        SELECTION_CRITERION_KEY = 'val_losses' # For loss metric
    else:
        # Default paths for 'score' or any other metric
        RESULTS_DIR = os.path.join(ROOT_DIR, 'results')
        MODEL_SAVE_DIR = os.path.join(ROOT_DIR, 'saved_models')
        ACTIVATION_DIR = os.path.join(ROOT_DIR, 'activations')
        SELECTION_CRITERION_KEY = 'val_scores'
    
    # Create directories if they don't exist
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(ACTIVATION_DIR, exist_ok=True)

    for dataset_name in DEFAULT_PARAMS:
        DEFAULT_PARAMS[dataset_name]['selection_criterion_key'] = SELECTION_CRITERION_KEY
    
    return {
        'RESULTS_DIR': RESULTS_DIR,
        'MODEL_SAVE_DIR': MODEL_SAVE_DIR,
        'ACTIVATION_DIR': ACTIVATION_DIR,
        'SELECTION_CRITERION_KEY': SELECTION_CRITERION_KEY 
    }

# --- Supported Algorithms ---
ALGORITHMS = ['local', 'fedavg'] # Add others as implemented
REG_ALOGRITHMS = ['fedprox', 'pfedme', 'ditto'] # Add others as implemented
# --- Supported Datasets ---
DATASETS = [
    'Synthetic_Label', 'Synthetic_Feature', 'Synthetic_Concept',
    'Credit', 'EMNIST', 'CIFAR', 'ISIC', 'IXITiny',
]

# --- Common Configuration for Tabular-like Datasets ---
COMMON_TABULAR_PARAMS = dict(
    fixed_classes=2, # Default for binary classification, override for multiclass
    default_lr = 5e-3,
    learning_rates_try=[1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4],
    default_reg_param=0.1,
    reg_params_try=[5, 2, 1, 5e-1, 1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 1e-5, 1e-6],
    batch_size=32,
    epochs_per_round=3,
    rounds=50,
    rounds_tune_inner=30,
    runs=50,
    runs_tune=5,
    metric='F1', # Default metric for tabular
    base_seed=42,
    samples_per_client=300,
    default_num_clients=5,
    servers_tune_lr=ALGORITHMS,
    servers_tune_reg = REG_ALOGRITHMS,
    partitioner_args={},
    max_parallel_clients=None,
    use_weighted_loss=False, # If True, client should use WeightedCELoss if criterion is CE
    shift_after_split=False,
    activation_extractor_type='hook_based',
    criterion_type="CrossEntropyLoss", # Default criterion
    source_args={}, # For raw data loading parameters
    selection_criterion_key= SELECTION_CRITERION_KEY, # Default for tabular: optimize for scores (F1, etc.)
    selection_criterion_direction_overrides={}, # Empty dict means use defaults based on key name,
    n_workers = 0
)

# --- Common Configuration for Image Datasets ---
COMMON_IMAGE_PARAMS = dict(
    fixed_classes=10, # Default for common image datasets like CIFAR/EMNIST
    default_lr=3e-3,
    learning_rates_try=[5e-3, 1e-3, 5e-4],
    default_reg_param=0.1,
    reg_params_try=[1, 5e-1, 1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 1e-5],
    batch_size=96,
    epochs_per_round=3,
    rounds=60,
    rounds_tune_inner=20,
    runs=15,
    runs_tune=3,
    metric='Accuracy', # Default metric for image classification
    base_seed=42,
    default_num_clients=2,
    servers_tune_lr=ALGORITHMS,
    servers_tune_reg = REG_ALOGRITHMS,
    partitioner_args={},
    max_parallel_clients=None,
    use_weighted_loss=False, # If True, client should use WeightedCELoss if criterion is CE
    shift_after_split=True, # Often true for image datasets with augmentation-like shifts
    activation_extractor_type='hook_based',
    criterion_type="CrossEntropyLoss", # Default criterion
    source_args={}, # For raw data loading parameters
    selection_criterion_key= SELECTION_CRITERION_KEY, # Default for image: optimize for accuracy
    selection_criterion_direction_overrides={}, # Empty dict means use defaults based on key name
    n_workers = 4
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
        'default_lr': 5e-4, 
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
        'default_lr': 5e-3, 
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
        'runs': 15,
        'runs_tune': 1,
        'metric': 'Balanced_accuracy',
        # max_clients removed
        'use_weighted_loss': True, # This flag will be used by client
        'criterion_type': "ISICLoss", # Explicitly set criterion
        'batch_size': 128,
        'default_lr' : 5e-4,
        'selection_criterion_key': SELECTION_CRITERION_KEY,
        
    },
    
    'IXITiny': {
        **COMMON_IMAGE_PARAMS,
        'dataset_name': 'IXITiny',
        'data_source': 'ixi_paths',
        'partitioning_strategy': 'pre_split',
        'dataset_class': 'IXITinyDataset',
        'fixed_classes': None,  # Override for segmentation
        'default_lr': 1e-3,  # Specific learning rate
        'learning_rates_try': [1e-2, 5e-3, 1e-3],  # Custom learning rates
        'batch_size': 4,  # Smaller batch size for 3D volumes
        'rounds': 30,  # Fewer rounds
        'rounds_tune_inner': 10,  # Fewer tuning rounds
        'runs': 10,  # Fewer runs
        'metric': 'DICE',  # Segmentation metric
        'shift_after_split': False,  # Not applicable for pre-split
        'activation_extractor_type': 'rep_vector',  # Different extractor
        'criterion_type': "DiceLoss",  # Segmentation loss
        # Unique parameters for fixed train/test split
        'fixed_train_test_split': True,
        'metadata_path': 'metadata_tiny.csv',
        'id_column': 'Patient ID',
        'split_column': 'Split',
        'validation_from_train_size': 0.2,
        # Override source_args completely
        'source_args': {
            'site_mappings': {
                'guys_hh': [['Guys'], ['HH']],
                'iop_guys': [['IOP'], ['Guys']],
                'iop_hh': [['IOP'], ['HH']],
                'all': [['IOP'], ['HH'], ['Guys']]
            },
            'image_shape': (80, 48, 48)
        },
    },
}
# --- Dataset Costs / Experiment Parameters ---
DATASET_COSTS = {
    'Synthetic_Label': [1000.0, 10.0, 2.0, 1.0, 0.75, 0.5, 0.2, 0.1],
    'Synthetic_Feature': [0.0, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0],
    'Credit': [0.0, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0],
    'EMNIST': [0.0, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0],
    'CIFAR': [0.0, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0],
    'Synthetic_Concept': [0.0, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0],
    'IXITiny': ['guys_hh', 'iop_guys', 'iop_hh', 'all'], 
    'ISIC': ['bcn_vmole','vmole_vmod', 'vmole_rose', 'vmole_msk', 'vmole_vienna','vmod_rose',], 
    # 'ISIC': ['bcn_vmole', 'bcn_vmod', 'bcn_rose', 'bcn_msk', 'bcn_vienna',
    #          'vmole_vmod', 'vmole_rose', 'vmole_msk', 'vmole_vienna','vmod_rose',
    #          'vmod_msk', 'vmod_vienna','rose_msk', 'rose_vienna','msk_vienna'], # Using string keys
}