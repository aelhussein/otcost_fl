"""
Handles loading initial, raw data from various sources (files, torchvision, generation).
Returns basic data structures like NumPy arrays, lists of paths, or torchvision Dataset objects.
"""
import os
import glob
import hashlib
import numpy as np
import pandas as pd
from torchvision.datasets import CIFAR10, EMNIST
from torchvision import transforms
from typing import Dict, Tuple, Any, Optional, List, Union, Callable

# Import the synthetic data generator function
from synthetic_data import generate_synthetic_data

# =============================================================================
# == Raw Data Loader Functions ==
# =============================================================================

def load_synthetic_raw(dataset_name: str,
                       source_args: dict,
                       client_num: Optional[int] = None,
                       cost_key: Optional[Any] = None,
                       base_seed: int = 42,
                       data_dir: Optional[str] = None
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """Generates raw synthetic data (features, labels) using the unified generator."""

    # 1. Determine Mode and Shift Parameter
    shift_param = 0.0
    mode = 'baseline' # Default
    if dataset_name == 'Synthetic_Feature':
        mode = 'feature_shift'
        if isinstance(cost_key, (int, float)):
            shift_param = float(np.clip(cost_key, 0.0, 1.0))
    elif dataset_name == 'Synthetic_Concept':
        mode = 'concept_shift'
        if isinstance(cost_key, (int, float)):
            shift_param = float(np.clip(cost_key, 0.0, 1.0))

    # 2. Determine Sample Size and Seed
    if mode == 'feature_shift':
        n_samples = source_args.get('n_samples_per_client', 1000)
        seed_string = f"synth-{base_seed}-{client_num}-{cost_key}" # Unique seed
        seed = int(hashlib.sha256(seed_string.encode('utf-8')).hexdigest(), 16) % (2**32)
    else:
        n_samples = source_args.get('base_n_samples', 10000)
        seed = source_args.get('random_state', base_seed)

    # 3. Extract parameters
    n_features = source_args.get('n_features', 10)
    label_noise = source_args.get('label_noise', 0.0)
    shift_config_keys = [ # Keys relevant to synthetic_data functions
        'feature_shift_kind', 'feature_shift_cols', 'feature_shift_mu', 'feature_shift_sigma',
        'concept_label_option', 'concept_threshold_range_factor'
    ]
    shift_config = {k: source_args[k] for k in shift_config_keys if k in source_args}

    # 4. Generate Data
    X_np, y_np = generate_synthetic_data(
        mode=mode, n_samples=n_samples, n_features=n_features,
        shift_param=shift_param, label_noise=label_noise, seed=seed, **shift_config
    )
    return X_np, y_np


def load_torchvision_raw(source_args: dict,
                         data_dir: str,
                         transform_config: Optional[dict] = None
                         ) -> Tuple[Any, Any]: # Returns raw torchvision dataset objects
    """Loads raw torchvision datasets (train/test)."""
    transform_config = transform_config or {}
    tv_dataset_name = source_args.get('dataset_name') # Get name from source_args
    root_dir = data_dir
    split = source_args.get('split', 'digits') # Default for EMNIST

    default_train_transform = transforms.Compose([transforms.ToTensor()])
    default_test_transform = transforms.Compose([transforms.ToTensor()])

    train_transform = transform_config.get('train', default_train_transform)
    test_transform = transform_config.get('test', default_test_transform)

    if tv_dataset_name == 'CIFAR10':
        train_dataset = CIFAR10(root=root_dir, train=True, download=True, transform=train_transform)
        test_dataset = CIFAR10(root=root_dir, train=False, download=True, transform=test_transform)
        return train_dataset, test_dataset
    elif tv_dataset_name == 'EMNIST':
        train_dataset = EMNIST(root=root_dir, split=split, train=True, download=True, transform=train_transform)
        test_dataset = EMNIST(root=root_dir, split=split, train=False, download=True, transform=test_transform)
        return train_dataset, test_dataset
    else:
        raise ValueError(f"Torchvision loader not configured for: {tv_dataset_name}")


def load_credit_raw(source_args: dict) -> Tuple[np.ndarray, np.ndarray]:
    """Loads raw Credit Card Fraud data from CSV."""
    csv_path = source_args['csv_path'] # Assume key exists
    df = pd.read_csv(csv_path)
    drop_cols = source_args.get('drop_cols', [])
    if drop_cols:
        df = df.drop(columns=drop_cols, errors='ignore')
    labels_np = df['Class'].values.astype(np.int64) # Assume 'Class' column exists
    features_np = df.drop(columns=['Class']).values.astype(np.float32)
    return features_np, labels_np


def load_heart_raw(client_num: int,
                   cost_key: Any,
                   source_args: Dict,
                   data_dir: str, ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads and internally scales Heart Disease data for a specific client.
    
    Args:
        client_num: The client number (1-based indexing)
        cost_key: Key to access site mappings
        data_dir: Directory containing heart data files
        config: The full configuration dictionary from configs.py
    
    Returns:
        Tuple of features and labels as numpy arrays
    """    
    # Get site mappings and validate
    site_map = source_args.get('site_mappings', {})
        
    # Get assigned sites for this client
    assigned_sites = site_map[cost_key][client_num - 1]
    
    # Extract other needed parameters from config
    used_columns = source_args.get('used_columns', [])
    feature_names = source_args.get('feature_names', [])
    cols_to_scale = source_args.get('cols_to_scale', [])
    scale_values = source_args.get('scale_values', {})
    
    features_list, labels_list = [], []
    
    # Process each assigned site
    for site_name in assigned_sites:
        fpath = os.path.join(data_dir, f'processed.{site_name}.data')
        # Load and process the data
        site_df = pd.read_csv(fpath, names=used_columns, na_values='?', 
                                header=None, usecols=used_columns).dropna()
        # Extract features and apply scaling
        X_site = site_df[feature_names].copy().values.astype(np.float32)
        if cols_to_scale and scale_values:
            for col_idx, col in enumerate(feature_names):
                if col in cols_to_scale and col in scale_values:
                    mean, variance = scale_values[col]
                    std_dev = np.sqrt(variance)
                    if std_dev > 1e-9:
                        X_site[:, col_idx] = (X_site[:, col_idx] - mean) / std_dev
                    else:
                        X_site[:, col_idx] = X_site[:, col_idx] - mean 
        # Extract labels
        y_site = site_df['target'].astype(int).values
        features_list.append(X_site)
        labels_list.append(y_site)
        
    # Combine results from all sites
    final_X = np.concatenate(features_list, axis=0)
    final_y = np.concatenate(labels_list, axis=0)
    return final_X, final_y


def load_isic_paths_raw(client_num: int,
                        cost_key: Any,
                        source_args: dict,
                        data_dir: str) -> Tuple[List[str], np.ndarray]:
    """Loads ISIC image paths and labels for a specific client's assigned site(s)."""
    root_dir = source_args.get('data_dir', data_dir)
    site_map = source_args['site_mappings']
    site_indices = site_map[cost_key][client_num - 1]
    if not isinstance(site_indices, list): site_indices = [site_indices]

    img_files_list, labels_list = [], []
    img_dir_path = os.path.join(root_dir, 'ISIC_2019_Training_Input_preprocessed')
    # Assume img_dir_path exists

    for site_idx in site_indices:
        csv_path = os.path.join(root_dir, f'site_{site_idx}_files_used.csv')
        if not os.path.exists(csv_path): continue
        try:
            files_df = pd.read_csv(csv_path)
            img_files_list.extend([os.path.join(img_dir_path, f"{stem}.jpg") for stem in files_df['image']])
            labels_list.extend(files_df['label'].values)
        except Exception:
            # Minimal error handling
            continue

    return img_files_list, np.array(labels_list).astype(np.int64)


def load_ixi_paths_raw(client_num: int,
                       cost_key: Any,
                       source_args: dict,
                       data_dir: str) -> Tuple[List[str], List[str]]:
    """Loads IXI image and label file paths for a specific client's assigned site(s)."""
    root_dir = source_args.get('data_dir', data_dir)
    sites_map = source_args['site_mappings']
    site_names_for_client = sites_map[cost_key][client_num - 1]
    if not isinstance(site_names_for_client, list): site_names_for_client = [site_names_for_client]

    img_dir = os.path.join(root_dir,'flamby/image')
    lbl_dir = os.path.join(root_dir,'flamby/label')
    # Assume directories exist

    all_img_files = glob.glob(os.path.join(img_dir, '*.nii.gz'))
    all_lbl_files = glob.glob(os.path.join(lbl_dir, '*.nii.gz'))

    # Simplified ID/Site extraction
    def get_base_id(p): return os.path.basename(p).split('_')[0]
    def get_site_name(p):
        known_sites = ['Guys', 'HH', 'IOP']
        for part in os.path.basename(p).split('.')[0].split('_'):
             if part in known_sites: return part
        return None

    image_dict = {get_base_id(p): p for p in all_img_files if get_site_name(p) in site_names_for_client}
    label_dict = {get_base_id(p): p for p in all_lbl_files if get_site_name(p) in site_names_for_client}

    common_ids = sorted(list(set(image_dict.keys()) & set(label_dict.keys())))
    aligned_image_files = [image_dict[id_] for id_ in common_ids]
    aligned_label_files = [label_dict[id_] for id_ in common_ids]

    return aligned_image_files, aligned_label_files

# =============================================================================
# == Loader Factory ==
# =============================================================================

def get_loader(source_name: str) -> Callable:
    """Factory function to get the appropriate raw data loader function."""
    if source_name == 'synthetic':
        return load_synthetic_raw
    elif source_name == 'torchvision':
        return load_torchvision_raw
    elif source_name == 'credit_csv':
        return load_credit_raw
    elif source_name == 'heart_site_loader':
        return load_heart_raw
    elif source_name == 'isic_paths':
        return load_isic_paths_raw
    elif source_name == 'ixi_paths':
        return load_ixi_paths_raw
    else:
        # Let caller handle potential KeyError if name not in config
        raise ValueError(f"Unknown data source name specified: '{source_name}'")