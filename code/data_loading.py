"""
Handles loading initial data from various sources.
- Loads full datasets for partitioning (torchvision, generated, full CSVs).
- Loads data per client for pre-split strategies (site files, path lists).
"""
import os
import glob
import sys
import numpy as np
import pandas as pd
import torch
from torchvision.datasets import CIFAR10, EMNIST
from torchvision import transforms
from sklearn.preprocessing import StandardScaler # Needed for SyntheticDataGenerator only if reused fully
from typing import Dict, Tuple, Any
# Import Base Dataset wrappers needed by loaders that return them
from datasets import SyntheticBaseDataset, CreditBaseDataset

# --- Synthetic Data Generation ---
class SyntheticDataGenerator:
    # (Paste the full class code here from the previous data_processing.py)
    def __init__(self, n_features=10, random_state=42): self.n_features = n_features; self.random_state = random_state; np.random.seed(random_state)
    def generate_orthogonal_matrices(self, dim): H = np.random.randn(dim, dim); Q, R = np.linalg.qr(H); return Q
    def generate_base_distributions(self, n_samples=1000): orth_mat = self.generate_orthogonal_matrices(self.n_features); basis_1 = orth_mat[:, :self.n_features//2]; basis_2 = orth_mat[:, self.n_features//2:]; data_1 = np.random.normal(0, 1, (n_samples, self.n_features//2)); data_2 = np.random.normal(0, 1, (n_samples, self.n_features - self.n_features//2)); d1 = data_1 @ basis_1.T; d2 = data_2 @ basis_2.T; d1 += np.random.normal(0, 0.1, d1.shape); d2 += np.random.normal(0, 0.1, d2.shape); return d1, d2
    def apply_nonlinear_transformations(self, dataset, distribution_type='mixed'):
        t = np.copy(dataset); n_samples, n_features = dataset.shape
        if distribution_type == 'normal':
            for i in range(n_features): mean = np.random.uniform(-3, 3); std = np.random.uniform(0.5, 2); t[:, i] = dataset[:, i] * std + mean
        elif distribution_type == 'skewed':
            for i in range(n_features): r = i % 3; t[:, i] = np.exp(dataset[:, i] / 2) if r == 0 else (np.sign(dataset[:, i]) * np.abs(dataset[:, i])**np.random.uniform(1.5, 3) if r == 1 else np.exp(dataset[:, i] / 3))
        elif distribution_type == 'mixed':
            for i in range(n_features): r = i % 5; t[:, i] = dataset[:, i] * np.random.uniform(0.5, 2) if r == 0 else (np.exp(dataset[:, i] / 2) - 1 if r == 1 else (np.sin(dataset[:, i] * 2) + dataset[:, i] / 2 if r == 2 else (dataset[:, i]**3 / 5 + dataset[:, i] if r == 3 else np.abs(dataset[:, i]) + np.random.uniform(-1, 1, n_samples) * 0.5)))
        return t
    def generate_single_dataset(self, n_samples=1000, dist_type1='normal', dist_type2='skewed', label_noise=0.1):
        print(f"Gen synth base: {n_samples} samples, type '{dist_type1}'"); base1, _ = self.generate_base_distributions(n_samples); dataset = self.apply_nonlinear_transformations(base1, dist_type1); sums = np.sum(dataset, axis=1); median_val = np.median(sums); labels = (sums > median_val).astype(int); noise_mask = np.random.random(n_samples) < label_noise; labels[noise_mask] = 1 - labels[noise_mask]; print(f"Gen synth base: shape {dataset.shape}, labels {np.bincount(labels)}"); return dataset, labels

# --- Loader Functions ---

def load_torchvision_dataset(dataset_name: str, data_dir: str, source_args: Dict, transform_config: Dict = None):
    """Loads train and test sets for specified torchvision dataset."""
    # (Keep function code as provided in the previous data_processing.py)
    transform_config = transform_config or {}; tv_dataset_name = source_args.get('dataset_name'); tv_dataset_args = {k: v for k, v in source_args.items() if k not in ['dataset_name', 'data_dir']}; root_dir = os.path.join(data_dir) # Use main data_dir
    print(f"Loading torchvision: {tv_dataset_name} from {root_dir} with args {tv_dataset_args}")
    if tv_dataset_name == 'CIFAR10':
        train_transform = transform_config.get('train') or transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])
        test_transform = transform_config.get('test') or transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])
        train_dataset = CIFAR10(root=root_dir, train=True, download=True, transform=train_transform); test_dataset = CIFAR10(root=root_dir, train=False, download=True, transform=test_transform); return train_dataset, test_dataset
    elif tv_dataset_name == 'EMNIST':
        split = tv_dataset_args.get('split'); assert split == 'digits', "Only EMNIST 'digits' split is supported."
        train_transform = transform_config.get('train') or transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]); test_transform = transform_config.get('test') or transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_dataset = EMNIST(root=root_dir, train=True, download=True, transform=train_transform, **tv_dataset_args); test_dataset = EMNIST(root=root_dir, train=False, download=True, transform=test_transform, **tv_dataset_args); return train_dataset, test_dataset
    else: raise ValueError(f"Torchvision loader not configured for: {tv_dataset_name}")

def load_synthetic_base_data(dataset_name: str, data_dir: str, source_args: Dict):
    """Generates the base synthetic dataset."""
    # (Keep function code as provided in the previous data_processing.py)
    print(f"Generating base Synthetic dataset with args: {source_args}"); generator = SyntheticDataGenerator(n_features=source_args.get('n_features', 10), random_state=source_args.get('random_state', 42)); features, labels = generator.generate_single_dataset(n_samples=source_args.get('base_n_samples', 10000), dist_type1=source_args.get('dist_type1', 'normal'), dist_type2=source_args.get('dist_type2', 'skewed'), label_noise=source_args.get('label_noise', 0.05)); return SyntheticBaseDataset(features, labels)


def load_credit_base_data(dataset_name: str, data_dir: str, source_args: Dict):
    """Loads the full Credit Card dataset."""
    # (Keep function code as provided in the previous data_processing.py)
    csv_path = source_args.get('csv_path'); drop_cols = source_args.get('drop_cols', []); assert csv_path and os.path.exists(csv_path), f"Credit card CSV not found at {csv_path}"; print(f"Loading base Credit dataset from: {csv_path}"); df = pd.read_csv(csv_path);
    if drop_cols: print(f"Dropping columns: {drop_cols}"); df = df.drop(columns=drop_cols, errors='ignore')
    labels = df['Class'].values; features = df.drop(columns=['Class']).values; print(f"Loaded Credit base: shape {features.shape}, labels {np.bincount(labels)}"); return CreditBaseDataset(features, labels)

def load_heart_site_data(dataset_name: str, data_dir: str, client_num: int, cost_key, config: Dict):
    """Loads and preprocesses data for specific Heart site(s) assigned to a client."""
    # (Keep function code as provided in the previous data_processing.py)
    source_args = config.get('source_args', {}); data_root = source_args.get('data_dir', os.path.join(data_dir, dataset_name)); site_map = source_args.get('site_mappings'); sites_info = source_args.get('sites'); used_columns = source_args.get('used_columns'); feature_names = source_args.get('feature_names'); cols_to_scale = source_args.get('cols_to_scale'); scale_values = source_args.get('scale_values')
    assert all([site_map, sites_info, used_columns, feature_names, cols_to_scale, scale_values]), "Heart source_args incomplete."; assert cost_key in site_map, f"Invalid cost key '{cost_key}' for Heart."; assert client_num <= len(site_map[cost_key]), f"Client num {client_num} invalid for cost '{cost_key}'."
    assigned_sites = site_map[cost_key][client_num - 1]; print(f"Loading Heart data for client {client_num}, sites: {assigned_sites} (cost_key={cost_key})"); client_features_list = []; client_labels_list = []
    for site_name in assigned_sites:
        if site_name not in sites_info: print(f"Warn: Site '{site_name}' not in sites list. Skipping."); continue
        file_path = os.path.join(data_root, f'processed.{site_name}.data');
        try:
            site_df = pd.read_csv(file_path, names=used_columns, na_values='?', header=None, usecols=used_columns).dropna();
            if site_df.empty: continue; scaled_features_df = site_df.copy()
            for col in cols_to_scale:
                if col in scaled_features_df.columns and col in scale_values: mean, variance = scale_values[col]; std_dev = np.sqrt(variance); scaled_features_df[col] = (scaled_features_df[col] - mean) / std_dev if std_dev > 1e-9 else scaled_features_df[col] - mean
            if 'target' in site_df.columns and all(f in scaled_features_df.columns for f in feature_names): X_site = scaled_features_df[feature_names].values; y_site = site_df['target'].astype(int).values; client_features_list.append(X_site); client_labels_list.append(y_site)
            else: print(f"Warn: Missing columns for site '{site_name}'. Skipping.")
        except FileNotFoundError: print(f"Warn: Data file not found for site '{site_name}'. Skipping.")
        except Exception as e: print(f"Warn: Error processing site '{site_name}': {e}. Skipping.")
    if not client_features_list: print(f"Warn: No data for client {client_num}."); return np.array([]).reshape(0, len(feature_names)), np.array([])
    final_X = np.concatenate(client_features_list, axis=0); final_y = np.concatenate(client_labels_list, axis=0); print(f"  -> Client {client_num} Heart data shape: X={final_X.shape}, y={final_y.shape}"); return final_X, final_y

def load_ixi_client_paths(dataset_name: str, data_dir: str, client_num: int, cost_key_or_suffix, config: Dict):
    """Loads file paths for one IXI client based on cost key using config."""
    # (Keep function code as provided in the previous data_processing.py)
    source_args = config.get('source_args', {}); root_dir = source_args.get('data_dir', os.path.join(data_dir, dataset_name)); sites_map = source_args.get('site_mappings'); cost_key = cost_key_or_suffix; assert sites_map, f"Missing 'site_mappings' for {dataset_name}"; assert cost_key in sites_map, f"Invalid cost key '{cost_key}' for IXITiny"; available_clients_for_cost = len(sites_map[cost_key]); assert client_num <= available_clients_for_cost, f"Client num {client_num} out of range for IXITiny cost '{cost_key}'"
    site_names_for_client = sites_map[cost_key][client_num - 1]; print(f"Loading IXI paths for client {client_num}, sites: {site_names_for_client} (cost_key={cost_key})"); image_files, label_files = [], []; image_dir = os.path.join(root_dir, 'flamby/image'); label_dir = os.path.join(root_dir, 'flamby/label'); assert os.path.isdir(image_dir) and os.path.isdir(label_dir), f"IXI dirs not found"
    for name_part in site_names_for_client: image_files.extend(glob.glob(os.path.join(image_dir, f'*{name_part}*.nii.gz'))); label_files.extend(glob.glob(os.path.join(label_dir, f'*{name_part}*.nii.gz')))
    def get_ixi_id(path): return os.path.basename(path).split('_')[0]
    labels_dict = {get_ixi_id(path): path for path in label_files}; images_dict = {get_ixi_id(path): path for path in image_files}; common_keys = sorted(list(set(labels_dict.keys()) & set(images_dict.keys()))); assert common_keys, f"No matching pairs found for IXI client {client_num}, sites {site_names_for_client}"
    aligned_image_files = [images_dict[key] for key in common_keys]; aligned_label_files = [labels_dict[key] for key in common_keys]; return np.array(aligned_image_files), np.array(aligned_label_files)


def load_isic_client_paths(dataset_name: str, data_dir: str, client_num: int, cost_key_or_suffix, config: Dict):
    """Loads image paths and labels for one ISIC client based on cost key using config."""
    # (Keep function code as provided in the previous data_processing.py)
    source_args = config.get('source_args', {}); root_dir = source_args.get('data_dir', os.path.join(data_dir, dataset_name)); site_map = source_args.get('site_mappings'); cost_key = cost_key_or_suffix; assert site_map, f"Missing 'site_mappings' for {dataset_name}"; assert cost_key in site_map, f"Invalid cost key '{cost_key}' for ISIC"; available_clients_for_cost = len(site_map[cost_key]); assert client_num <= available_clients_for_cost, f"Client num {client_num} out of range for ISIC cost '{cost_key}'"
    site_indices_for_client = site_map[cost_key][client_num - 1]; print(f"Loading ISIC paths for client {client_num}, site indices: {site_indices_for_client} (cost_key={cost_key})"); image_files, labels = [], []; image_dir = os.path.join(root_dir, 'ISIC_2019_Training_Input_preprocessed'); assert os.path.isdir(image_dir), f"ISIC image dir not found"
    sampling_config = config.get('sampling_config'); nrows = None
    if sampling_config and sampling_config.get('type') == 'fixed_total': nrows = sampling_config.get('size')
    for site_index in site_indices_for_client:
        site_csv_path = os.path.join(root_dir, f'site_{site_index}_files_used.csv'); assert os.path.exists(site_csv_path), f"ISIC site file not found: {site_csv_path}"
        if nrows: print(f"Sampling ISIC client {client_num} (site {site_index}) to {nrows} rows")
        files_df = pd.read_csv(site_csv_path, nrows=nrows); image_files.extend([os.path.join(image_dir, f"{stem}.jpg") for stem in files_df['image']]); labels.extend(files_df['label'].values)
    return np.array(image_files), np.array(labels)

# --- Dispatch Dictionary ---
DATA_LOADERS = {
    # Base loaders (return BaseDataset wrapper or tuple for torchvision)
    'torchvision': load_torchvision_dataset,
    'synthetic_base': load_synthetic_base_data,
    'credit_base': load_credit_base_data,

    # Per-client / Site loaders (return X, y tuple)
    'heart_site_loader': load_heart_site_data,
    'pre_split_paths_ixi': load_ixi_client_paths,
    'pre_split_paths_isic': load_isic_client_paths,
}