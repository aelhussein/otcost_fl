# data_loading.py
"""
Handles loading initial, raw data from various sources (files, torchvision, generation).
Returns basic data structures like NumPy arrays, lists of paths, or torchvision Dataset objects.
Simplified version with reduced complexity and more consistent pattern handling.
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
    # Simplified dataset-to-mode mapping 
    mode_mapping = {
        'Synthetic_Feature': 'feature_shift',
        'Synthetic_Concept': 'concept_shift',
        'Synthetic_Label': 'baseline'
    }
    mode = mode_mapping.get(dataset_name, 'baseline')
    
    # Set shift parameter if applicable (for feature/concept shift)
    shift_param = 0.0
    if mode in ['feature_shift', 'concept_shift'] and isinstance(cost_key, (int, float)):
        shift_param = float(np.clip(cost_key, 0.0, 1.0))
    
    # Determine sample size and seed based on mode
    if mode == 'feature_shift':
        n_samples = source_args.get('n_samples_per_client', 1000)
        # Create deterministic but unique seed for each client/cost combination
        seed_string = f"synth-{base_seed}-{client_num}-{cost_key}"
        seed = int(hashlib.sha256(seed_string.encode('utf-8')).hexdigest(), 16) % (2**32)
    else:
        n_samples = source_args.get('base_n_samples', 10000)
        seed = source_args.get('random_state', base_seed)
    
    # Basic parameters
    n_features = source_args.get('n_features', 10)
    label_noise = source_args.get('label_noise', 0.0)
    
    # Extract relevant configuration keys for each shift type
    shift_configs = {
        'feature_shift': ['feature_shift_kind', 'feature_shift_cols', 'feature_shift_mu', 'feature_shift_sigma'],
        'concept_shift': ['concept_label_option', 'concept_threshold_range_factor']
    }
    
    # Extract only the keys needed for this mode
    relevant_keys = shift_configs.get(mode, [])
    shift_config = {k: source_args[k] for k in relevant_keys if k in source_args}
    
    # Generate data
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
    tv_dataset_name = source_args.get('dataset_name')
    split = source_args.get('split', 'digits')  # Default for EMNIST
    
    # Use default transforms if not specified
    default_transform = transforms.Compose([transforms.ToTensor()])
    train_transform = transform_config.get('train', default_transform)
    test_transform = transform_config.get('test', default_transform)
    
    # Dataset mapping - could be extended for more datasets
    dataset_mapping = {
        'CIFAR10': lambda: (
            CIFAR10(root=data_dir, train=True, download=True, transform=train_transform),
            CIFAR10(root=data_dir, train=False, download=True, transform=test_transform)
        ),
        'EMNIST': lambda: (
            EMNIST(root=data_dir, split=split, train=True, download=True, transform=train_transform),
            EMNIST(root=data_dir, split=split, train=False, download=True, transform=test_transform)
        )
    }
    
    # Get loader function and call it
    loader_fn = dataset_mapping.get(tv_dataset_name)
    if loader_fn:
        return loader_fn()
    else:
        raise ValueError(f"Torchvision loader not configured for: {tv_dataset_name}")


def load_credit_raw(source_args: dict) -> Tuple[np.ndarray, np.ndarray]:
    """Loads raw Credit Card Fraud data from CSV."""
    csv_path = source_args['csv_path']
    df = pd.read_csv(csv_path)
    
    # Drop specified columns
    drop_cols = source_args.get('drop_cols', [])
    if drop_cols:
        df = df.drop(columns=drop_cols, errors='ignore')
    
    # Extract labels and features
    labels_np = df['Class'].values.astype(np.int64)
    features_np = df.drop(columns=['Class']).values.astype(np.float32)
    
    return features_np, labels_np


def load_heart_raw(client_num: int,
                   cost_key: Any,
                   source_args: Dict,
                   data_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads and internally scales Heart Disease data for a specific client.
    """    
    # Get assigned sites for this client from site mappings
    site_map = source_args.get('site_mappings', {})
    assigned_sites = site_map[cost_key][client_num - 1]
    
    # Extract configuration parameters
    used_columns = source_args.get('used_columns', [])
    feature_names = source_args.get('feature_names', [])
    cols_to_scale = source_args.get('cols_to_scale', [])
    scale_values = source_args.get('scale_values', {})
    
    features_list, labels_list = [], []
    
    # Process each assigned site
    for site_name in assigned_sites:
        fpath = os.path.join(data_dir, f'processed.{site_name}.data')
        
        # Load and process the data
        site_df = pd.read_csv(
            fpath, names=used_columns, na_values='?', 
            header=None, usecols=used_columns
        ).dropna()
        
        # Extract features
        X_site = site_df[feature_names].copy().values.astype(np.float32)
        
        # Apply scaling to specified columns
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
        
        # Add to collection
        features_list.append(X_site)
        labels_list.append(y_site)
        
    # Combine results from all sites
    final_X = np.concatenate(features_list, axis=0) if features_list else np.array([])
    final_y = np.concatenate(labels_list, axis=0) if labels_list else np.array([])
    
    return final_X, final_y


def load_isic_paths_raw(client_num: int,
                       cost_key: Any,
                       source_args: dict,
                       data_dir: str) -> Tuple[List[str], np.ndarray]:
    """Loads ISIC image paths and labels for a specific client's assigned site(s)."""
    # Get root directory and site assignments
    root_dir = source_args.get('data_dir', data_dir)
    site_map = source_args['site_mappings']
    site_indices = site_map[cost_key][client_num - 1]
    
    # Ensure site_indices is a list
    if not isinstance(site_indices, list):
        site_indices = [site_indices]
    
    # Image directory path
    img_dir_path = os.path.join(root_dir, 'ISIC_2019_Training_Input_preprocessed')
    img_files_list, labels_list = [], []
    
    # Process each site
    for site_idx in site_indices:
        csv_path = os.path.join(root_dir, f'site_{site_idx}_files_used.csv')
        if not os.path.exists(csv_path):
            continue
            
        try:
            # Load file paths and labels from CSV
            files_df = pd.read_csv(csv_path)
            
            # Create full image paths and get labels
            img_files_list.extend([
                os.path.join(img_dir_path, f"{stem}.jpg") 
                for stem in files_df['image']
            ])
            labels_list.extend(files_df['label'].values)
        except Exception:
            continue
    
    return img_files_list, np.array(labels_list).astype(np.int64)


def load_ixi_paths_raw(client_num: int,
                      cost_key: Any,
                      source_args: dict,
                      data_dir: str) -> Tuple[List[str], List[str]]:
    """Loads IXI image and label file paths for a specific client's assigned site(s)."""
    # Get root directory and site assignments
    root_dir = source_args.get('data_dir', data_dir)
    sites_map = source_args['site_mappings']
    site_names = sites_map[cost_key][client_num - 1]
    
    # Ensure site_names is a list
    if not isinstance(site_names, list):
        site_names = [site_names]
    
    # Directory paths
    img_dir = os.path.join(root_dir, 'flamby/image')
    lbl_dir = os.path.join(root_dir, 'flamby/label')
    
    # Get all image and label files
    all_img_files = glob.glob(os.path.join(img_dir, '*.nii.gz'))
    all_lbl_files = glob.glob(os.path.join(lbl_dir, '*.nii.gz'))
    
    # Helper functions to extract ID and site
    def get_file_info(filepath):
        basename = os.path.basename(filepath)
        base_id = basename.split('_')[0]
        
        # Extract site name
        known_sites = ['Guys', 'HH', 'IOP']
        site = None
        for part in basename.split('.')[0].split('_'):
            if part in known_sites:
                site = part
                break
                
        return base_id, site
    
    # Filter files by site and create ID-to-path mappings
    image_dict = {}
    label_dict = {}
    
    for img_file in all_img_files:
        img_id, img_site = get_file_info(img_file)
        if img_site in site_names:
            image_dict[img_id] = img_file
            
    for lbl_file in all_lbl_files:
        lbl_id, lbl_site = get_file_info(lbl_file)
        if lbl_site in site_names:
            label_dict[lbl_id] = lbl_file
    
    # Find common IDs to ensure alignment
    common_ids = sorted(list(set(image_dict.keys()) & set(label_dict.keys())))
    
    # Create aligned lists of image and label files
    aligned_image_files = [image_dict[id_] for id_ in common_ids]
    aligned_label_files = [label_dict[id_] for id_ in common_ids]
    
    return aligned_image_files, aligned_label_files

# =============================================================================
# == Loader Factory ==
# =============================================================================

def get_loader(source_name: str) -> Callable:
    """Factory function to get the appropriate raw data loader function."""
    loaders = {
        'synthetic': load_synthetic_raw,
        'torchvision': load_torchvision_raw,
        'credit_csv': load_credit_raw,
        'heart_site_loader': load_heart_raw,
        'isic_paths': load_isic_paths_raw,
        'ixi_paths': load_ixi_paths_raw
    }
    
    loader = loaders.get(source_name)
    if loader:
        return loader
    else:
        raise ValueError(f"Unknown data source name: '{source_name}'")