from configs import N_WORKERS
from helper import get_parameters_for_dataset
import pandas as pd
import numpy as np
import os
import sys
import random
import glob
from typing import Dict, List
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import Dataset as TorchDataset, CIFAR10, EMNIST, transforms
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import albumentations
from monnai import  EnsureChannelFirst,  AsDiscrete, Compose, NormalizeIntensity,  Resize, MonaiToTensor
import nibabels as nib
from PIL import Image


def sample_per_class(labels, class_size=500):
    df = pd.DataFrame({'labels': labels})
    df_stratified = df.groupby('labels').apply(lambda x: x.sample(class_size, replace=False))
    return df_stratified.index.get_level_values(1)


def get_common_name(full_path):
    return os.path.basename(full_path).split('_')[0]


def align_image_label_files(image_files, label_files):
    labels_dict = {get_common_name(path): path for path in label_files}
    images_dict = {get_common_name(path): path for path in image_files}
    
    common_keys = sorted(set(labels_dict.keys()) & set(images_dict.keys()))
    return [images_dict[key] for key in common_keys], [labels_dict[key] for key in common_keys]


def validate_dataset_config(config: Dict, dataset_name: str):
    """Basic validation for required config keys."""
    required_keys = ['data_source', 'partitioning_strategy', 'cost_interpretation',
                     'dataset_class', 'default_num_clients']
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise ValueError(f"Dataset config for '{dataset_name}' missing required keys: {missing_keys}")
    # Add specific checks, e.g., for site_mappings if needed by strategy
    if config['partitioning_strategy'] == 'pre_split' and dataset_name in ['ISIC', 'IXITiny']:
         if 'site_mappings' not in config.get('source_args', {}):
             print(f"Warning: 'site_mappings' potentially missing in source_args for pre-split {dataset_name}.")


# --- Data Source Loaders ---

def load_torchvision_dataset(dataset_name: str, data_dir: str, source_args: Dict, transform_config: Dict = None):
    """Loads train and test sets for specified torchvision dataset."""
    transform_config = transform_config or {}
    tv_dataset_name = source_args.get('dataset_name')
    tv_dataset_args = {k: v for k, v in source_args.items() if k not in ['dataset_name', 'data_dir']}
    root_dir = os.path.join(data_dir) # Construct full path

    print(f"Loading torchvision dataset: {tv_dataset_name} from {root_dir} with args {tv_dataset_args}")

    if tv_dataset_name == 'CIFAR':
        train_transform = transform_config.get('train') # Allow override via config later
        test_transform = transform_config.get('test')
        if not train_transform:
             train_transform=transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
                ])
        if not test_transform:
             test_transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
                ])
        train_dataset = CIFAR10(root=root_dir, train=True, download=True, transform=train_transform)
        test_dataset = CIFAR10(root=root_dir, train=False, download=True, transform=test_transform)
        return train_dataset, test_dataset

    elif tv_dataset_name == 'EMNIST':
        split = tv_dataset_args.get('split')
        if split != 'digits':
            print(f"Warning: Requested EMNIST split '{split}', but only 'digits' is configured for 10 classes.")
            # Force digits for now based on config, could make this configurable
            tv_dataset_args['split'] = 'digits'

        train_transform = transform_config.get('train') or transforms.Compose([
                transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))
            ])
        test_transform = transform_config.get('test') or transforms.Compose([
                 transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))
             ])
        train_dataset = EMNIST(root=root_dir, train=True, download=True, transform=train_transform, **tv_dataset_args)
        test_dataset = EMNIST(root=root_dir, train=False, download=True, transform=test_transform, **tv_dataset_args)
        return train_dataset, test_dataset
    else:
         raise ValueError(f"Torchvision loader not configured for: {tv_dataset_name}")


def load_synthetic_base_data(dataset_name: str, data_dir: str, source_args: Dict):
    """Generates the base synthetic dataset for later partitioning."""
    print(f"Generating base Synthetic dataset with args: {source_args}")
    generator = SyntheticDataGenerator(
        n_features=source_args.get('n_features', 10),
        random_state=source_args.get('random_state', 42)
    )
    # Use the simplified generator method
    features, labels = generator.generate_single_dataset(
        n_samples=source_args.get('base_n_samples', 10000),
        dist_type1=source_args.get('dist_type1', 'normal'),
        dist_type2=source_args.get('dist_type2', 'skewed'), # Note: dist_type2 currently unused in generate_single_dataset
        label_noise=source_args.get('label_noise', 0.05)
    )
    # Return the Base Dataset wrapper (no train/test split here)
    return SyntheticBaseDataset(features, labels)

def load_credit_base_data(dataset_name: str, data_dir: str, source_args: Dict):
    """Loads the full Credit Card dataset for later partitioning."""
    csv_path = source_args.get('csv_path')
    drop_cols = source_args.get('drop_cols', [])
    if not csv_path or not os.path.exists(csv_path):
        raise FileNotFoundError(f"Credit card CSV not found at {csv_path}")

    print(f"Loading base Credit dataset from: {csv_path}")
    df = pd.read_csv(csv_path)

    if drop_cols:
        print(f"Dropping columns: {drop_cols}")
        df = df.drop(columns=drop_cols, errors='ignore')

    # Separate features (X) and labels (y)
    labels = df['Class'].values
    features = df.drop(columns=['Class']).values

    print(f"Loaded Credit base dataset with shape {features.shape} and label distribution {np.bincount(labels)}")
    # Return the Base Dataset wrapper (no train/test split here)
    return CreditBaseDataset(features, labels)

def load_heart_site_data(dataset_name: str, data_dir: str, client_num: int, cost_key, config: Dict):
    """Loads and preprocesses data for specific Heart site(s) assigned to a client."""
    source_args = config.get('source_args', {})
    data_root = source_args.get('data_dir', os.path.join(data_dir, dataset_name))
    site_map = source_args.get('site_mappings')
    sites_info = source_args.get('sites')
    used_columns = source_args.get('used_columns')
    feature_names = source_args.get('feature_names')
    cols_to_scale = source_args.get('cols_to_scale')
    scale_values = source_args.get('scale_values') # Dict of {col: (mean, variance)}

    assert all([site_map, sites_info, used_columns, feature_names, cols_to_scale, scale_values]), "Heart source_args incomplete."
    assert cost_key in site_map, f"Invalid cost key '{cost_key}' for Heart site map."
    assert client_num <= len(site_map[cost_key]), f"Client num {client_num} invalid for cost '{cost_key}'."

    # Get the list of site names for this specific client
    assigned_sites = site_map[cost_key][client_num - 1] # e.g., ['cleveland'] or ['cleveland', 'va']
    print(f"Loading Heart data for client {client_num}, sites: {assigned_sites} (cost_key={cost_key})")

    client_features_list = []
    client_labels_list = []

    for site_name in assigned_sites:
        if site_name not in sites_info:
            print(f"Warning: Site '{site_name}' not found in configured sites list. Skipping.")
            continue

        file_path = os.path.join(data_root, f'processed.{site_name}.data')
        try:
            site_df = pd.read_csv(file_path, names=used_columns, na_values='?', header=None, usecols=used_columns).dropna()

            if site_df.empty: continue

            # Apply pre-defined scaling
            scaled_features_df = site_df.copy()
            for col in cols_to_scale:
                if col in scaled_features_df.columns and col in scale_values:
                    mean, variance = scale_values[col]
                    std_dev = np.sqrt(variance)
                    if std_dev > 1e-9: scaled_features_df[col] = (scaled_features_df[col] - mean) / std_dev
                    else: scaled_features_df[col] = scaled_features_df[col] - mean

            # Check if all required columns exist after processing
            if 'target' in site_df.columns and all(f in scaled_features_df.columns for f in feature_names):
                X_site = scaled_features_df[feature_names].values
                y_site = site_df['target'].astype(int).values # Ensure integer target
                client_features_list.append(X_site)
                client_labels_list.append(y_site)
            else:
                 print(f"Warning: Missing required columns in processed data for site '{site_name}'. Skipping.")

        except FileNotFoundError: print(f"Warning: Data file not found for site '{site_name}' at {file_path}. Skipping.")
        except Exception as e: print(f"Warning: Error processing site '{site_name}': {e}. Skipping.")

    if not client_features_list:
         print(f"Warning: No data loaded for client {client_num}. Returning empty arrays.")
         return np.array([]).reshape(0, len(feature_names)), np.array([]) # Return empty arrays with correct feature dim

    # Concatenate data if multiple sites were assigned
    final_X = np.concatenate(client_features_list, axis=0)
    final_y = np.concatenate(client_labels_list, axis=0)

    print(f"  -> Client {client_num} final data shape: X={final_X.shape}, y={final_y.shape}")
    return final_X, final_y # Return combined, pre-scaled X, y for this client


def load_ixi_client_paths(dataset_name: str, data_dir: str, client_num: int, cost_key_or_suffix, config: Dict):
    """Loads file paths for one IXI client based on cost key using config."""
    source_args = config.get('source_args', {})
    root_dir = source_args.get('data_dir', os.path.join(data_dir, dataset_name))
    sites_map = source_args.get('site_mappings')
    cost_key = cost_key_or_suffix # Assume the key is passed directly

    if not sites_map:
         raise ValueError(f"Missing 'site_mappings' in 'source_args' for {dataset_name} config.")
    if cost_key not in sites_map:
        raise ValueError(f"Invalid cost key '{cost_key}' for IXITiny site mapping in config. Available: {list(sites_map.keys())}")

    available_clients_for_cost = len(sites_map[cost_key])
    if client_num > available_clients_for_cost:
         raise ValueError(f"Client number {client_num} out of range ({available_clients_for_cost} available) for cost key '{cost_key}' in IXITiny mapping.")

    site_names_for_client = sites_map[cost_key][client_num - 1] # Get list of site names for this client
    print(f"Loading IXI paths for client {client_num}, sites: {site_names_for_client} (cost_key={cost_key})")

    image_files, label_files = [], []
    image_dir = os.path.join(root_dir, 'flamby/image') # Standard flamby structure assumed
    label_dir = os.path.join(root_dir, 'flamby/label')

    if not os.path.isdir(image_dir) or not os.path.isdir(label_dir):
         raise FileNotFoundError(f"IXI data directories not found at {image_dir} or {label_dir}")

    for name_part in site_names_for_client:
        image_files.extend(glob.glob(os.path.join(image_dir, f'*{name_part}*.nii.gz')))
        label_files.extend(glob.glob(os.path.join(label_dir, f'*{name_part}*.nii.gz')))

    def get_ixi_id(path):
         base = os.path.basename(path)
         parts = base.split('_') # Expecting format like IXI..._image.nii.gz or IXI..._label.nii.gz
         return parts[0]

    labels_dict = {get_ixi_id(path): path for path in label_files}
    images_dict = {get_ixi_id(path): path for path in image_files}
    common_keys = sorted(list(set(labels_dict.keys()) & set(images_dict.keys())))

    if not common_keys:
         print(f"Warning: No matching image/label pairs found for IXI client {client_num}, sites {site_names_for_client}")

    aligned_image_files = [images_dict[key] for key in common_keys]
    aligned_label_files = [labels_dict[key] for key in common_keys]

    return np.array(aligned_image_files), np.array(aligned_label_files)


# MODIFIED: Uses site_mappings from config['source_args']
def load_isic_client_paths(dataset_name: str, data_dir: str, client_num: int, cost_key_or_suffix, config: Dict):
    """Loads image paths and labels for one ISIC client based on cost key using config."""
    source_args = config.get('source_args', {})
    root_dir = source_args.get('data_dir', os.path.join(data_dir, dataset_name))
    site_map = source_args.get('site_mappings')
    cost_key = cost_key_or_suffix

    if not site_map:
         raise ValueError(f"Missing 'site_mappings' in 'source_args' for {dataset_name} config.")
    if cost_key not in site_map:
         raise ValueError(f"Invalid cost key '{cost_key}' for ISIC site mapping in config. Available: {list(site_map.keys())}")

    available_clients_for_cost = len(site_map[cost_key])
    if client_num > available_clients_for_cost:
         raise ValueError(f"Client number {client_num} out of range ({available_clients_for_cost} available) for cost key '{cost_key}' in ISIC mapping.")

    site_index = site_map[cost_key][client_num - 1] # Get the site index (0-3)
    print(f"Loading ISIC paths for client {client_num}, site index: {site_index} (cost_key={cost_key})")

    site_csv_path = os.path.join(root_dir, f'site_{site_index}_files_used.csv')
    if not os.path.exists(site_csv_path):
         raise FileNotFoundError(f"ISIC site file not found: {site_csv_path}")

    # Load the CSV, potentially sampling using main config's sampling_config
    sampling_config = config.get('sampling_config')
    nrows = None
    if sampling_config and sampling_config.get('type') == 'fixed_total':
         nrows = sampling_config.get('size')
         print(f"Sampling ISIC client {client_num} (site {site_index}) down to {nrows} rows")

    files_df = pd.read_csv(site_csv_path, nrows=nrows)

    # Construct full image paths
    image_dir = os.path.join(root_dir, 'ISIC_2019_Training_Input_preprocessed')
    if not os.path.isdir(image_dir):
          raise FileNotFoundError(f"ISIC preprocessed image directory not found: {image_dir}")

    image_files = [os.path.join(image_dir, f"{file_stem}.jpg") for file_stem in files_df['image']]
    labels = files_df['label'].values

    return np.array(image_files), labels


# --- Partitioning Strategies ---

def partition_dirichlet_indices(dataset: TorchDataset,
                                num_clients: int,
                                alpha: float,
                                seed: int = 42,
                                **kwargs) -> Dict[int, List[int]]:
    """
    Partitions dataset indices based on labels using Dirichlet distribution,
    aiming for equal total samples per client initially. Then, if any client's
    assigned indices exceed 5000, it subsamples down to 5000 randomly.

    Args:
        dataset: The dataset to partition (must have .targets or be iterable for labels).
        num_clients: The number of clients.
        alpha: The concentration parameter for the Dirichlet distribution.
        seed: Random seed for reproducibility.
        **kwargs: Catches unused keyword arguments.

    Returns:
        A dictionary mapping client index (int) to a list of sample indices (List[int]),
        where each list has at most 5000 indices.
    """
    max_samples_limit: int = kwargs.get('sampling_config', {}).get('size', 1000)
    # ---

    print(f"Starting Dirichlet partitioning: alpha={alpha}, num_clients={num_clients}. Will subsample clients exceeding {max_samples_limit}.")

    # --- 1. Extract labels ---
    try:
        labels = np.array(dataset.targets)
        # print("Labels successfully extracted via .targets.") # Verbose
    except AttributeError:
        print("No .targets found, falling back to iteration.")
        labels = np.array([dataset[i][1] for i in range(len(dataset))])

    n = len(dataset)
    classes, class_counts = np.unique(labels, return_counts=True)
    n_classes = len(classes)

    print(f"Dataset has {n} total samples and {n_classes} classes.")
    # print(f"Class distribution: {dict(zip(classes, class_counts))}") # Verbose

    if n == 0 or n_classes == 0:
        print("Warning: Dataset is empty or has no classes. Returning empty partitions.")
        return {i: [] for i in range(num_clients)}

    # --- 2. Create per-class pools ---
    idx_by_class = {c: np.where(labels == c)[0].tolist() for c in classes}
    for c in classes:
        np.random.shuffle(idx_by_class[c]) # Shuffle pools

    # --- 3. Generate Dirichlet proportions ---
    proportions = np.random.dirichlet([alpha] * n_classes, size=num_clients)
    # print(f"Generated Dirichlet proportions with alpha={alpha} for {num_clients} clients.") # Verbose

    # --- 4. Assign total samples per client (proportional target) ---
    base_quota = n // num_clients
    quotas = [base_quota + (1 if i < (n % num_clients) else 0) for i in range(num_clients)]
    print(f"Initial target quotas (proportional): {quotas} (sum = {sum(quotas)})")

    # --- 5. Allocate indices based on initial proportional quotas ---
    client_indices = {i: [] for i in range(num_clients)}
    globally_assigned_indices = set() # Track assignment to prevent duplicates if pools exhaust

    for client_id in range(num_clients):
        p = proportions[client_id]
        # Calculate target counts based on this client's proportional quota
        target_counts = (p * quotas[client_id]).astype(int)
        target_counts[-1] = quotas[client_id] - target_counts[:-1].sum() # Fix rounding
        target_counts = np.maximum(0, target_counts) # Ensure non-negative

        # print(f"Client {client_id+1}: Initial class-wise target = {target_counts.tolist()}") # Verbose
        assigned_this_client = 0
        for cls_idx, cls in enumerate(classes):
            num_wanted = target_counts[cls_idx]
            if num_wanted <= 0: continue

            pool = idx_by_class[cls]
            start_ptr = 0 # Simple approach: draw from start of shuffled pool for this client
            
            indices_to_add = []
            taken_count = 0
            
            # Iterate through the pool for this class to find unassigned indices
            idx_in_pool = 0
            while taken_count < num_wanted and idx_in_pool < len(pool):
                current_idx = pool[idx_in_pool]
                if current_idx not in globally_assigned_indices:
                    indices_to_add.append(current_idx)
                    globally_assigned_indices.add(current_idx)
                    taken_count += 1
                idx_in_pool += 1

            client_indices[client_id].extend(indices_to_add)
            assigned_this_client += len(indices_to_add)

            if taken_count < num_wanted:
                print(f"  Warning: Class {cls} pool exhausted globally before client {client_id+1} got {num_wanted} samples (received {taken_count}).")

        # Check if quota was met (might not be if pools exhausted)
        # print(f"  Client {client_id+1}: Initially assigned {assigned_this_client}/{quotas[client_id]} samples.") # Verbose

    # --- 5.5 Subsample clients exceeding the limit ---
    print(f"\nSubsampling clients with more than {max_samples_limit} samples...")
    for i in range(num_clients):
        current_size = len(client_indices[i])
        if current_size > max_samples_limit:
            print(f"  Client {i+1}: Subsampling from {current_size} down to {max_samples_limit} indices.")
            # Use random.sample for efficient sampling without replacement
            client_indices[i] = random.sample(client_indices[i], max_samples_limit)
        # else: # Verbose
        #     print(f"  Client {i+1}: Has {current_size} samples (<= limit), no subsampling needed.")


    # --- 6. Shuffle final indices and print final sizes ---
    print("\nFinal client sample sizes after partitioning and potential subsampling:")
    total_assigned_final = 0
    final_assigned_indices_set = set()

    for i in range(num_clients):
        # Shuffle the final list of indices for the client
        np.random.shuffle(client_indices[i]) # Use numpy shuffle on the list
        size = len(client_indices[i])
        print(f"  Client {i+1}: {size} samples")
        total_assigned_final += size
        final_assigned_indices_set.update(client_indices[i])

    print(f"\nTotal samples finally assigned across all clients: {total_assigned_final}")
    print(f"Total unique indices finally assigned: {len(final_assigned_indices_set)}")

    # Sanity check: Ensure no client exceeds the limit after subsampling
    for i in range(num_clients):
        assert len(client_indices[i]) <= max_samples_limit, \
            f"Error: Client {i+1} has {len(client_indices[i])} samples, exceeding the limit {max_samples_limit} after subsampling!"

    # Sanity check: Ensure all final indices are unique
    assert total_assigned_final == len(final_assigned_indices_set), "Index collision detected in final assignment!"

    print("-" * 30) # Separator
    return client_indices


def partition_iid_indices(dataset: TorchDataset, num_clients: int, seed: int = 42, **kwargs):
    """Partitions dataset indices equally and randomly (IID)."""
    n_samples = len(dataset)
    indices = np.arange(n_samples)
    np.random.seed(seed)
    np.random.shuffle(indices)
    split_indices = np.array_split(indices, num_clients)
    client_indices = {i: split_indices[i].tolist() for i in range(num_clients)}
    return client_indices

def partition_pre_defined(client_num: int, **kwargs):
    """Placeholder for pre-split data. Returns the client num for lookup."""
    # The actual data loading happens per client in the dispatcher
    return client_num


class SyntheticDataGenerator:
    """
    Generates synthetic tabular datasets with controlled distribution properties
    and specifically designed to have orthogonal representations in feature space.
    (Code from dataCreator.py)
    """
    def __init__(self, n_features=10, random_state=42):
        self.n_features = n_features
        self.random_state = random_state
        np.random.seed(random_state)

    def generate_orthogonal_matrices(self, dim):
        """Generate orthogonal matrices using QR decomposition"""
        H = np.random.randn(dim, dim)
        Q, R = np.linalg.qr(H)
        return Q

    def generate_base_distributions(self, n_samples=1000):
        """
        Generate two base distributions with orthogonal principal components
        """
        orthogonal_matrix = self.generate_orthogonal_matrices(self.n_features)
        basis_1 = orthogonal_matrix[:, :self.n_features//2]
        basis_2 = orthogonal_matrix[:, self.n_features//2:]

        data_1 = np.random.normal(0, 1, (n_samples, self.n_features//2))
        data_2 = np.random.normal(0, 1, (n_samples, self.n_features//2))

        dataset_1 = data_1 @ basis_1.T
        dataset_2 = data_2 @ basis_2.T

        dataset_1 += np.random.normal(0, 0.1, dataset_1.shape)
        dataset_2 += np.random.normal(0, 0.1, dataset_2.shape)

        return dataset_1, dataset_2

    def apply_nonlinear_transformations(self, dataset, distribution_type='mixed'):
        """
        Apply non-linear transformations.
        (Code from dataCreator.py - truncated for brevity, assumed correct)
        """
        transformed = np.copy(dataset)
        n_samples, n_features = dataset.shape

        if distribution_type == 'normal':
            for i in range(n_features):
                mean = np.random.uniform(-3, 3); std = np.random.uniform(0.5, 2)
                transformed[:, i] = dataset[:, i] * std + mean
        elif distribution_type == 'skewed':
             for i in range(n_features):
                 if i % 3 == 0: transformed[:, i] = np.exp(dataset[:, i] / 2)
                 elif i % 3 == 1: transformed[:, i] = np.sign(dataset[:, i]) * np.abs(dataset[:, i])**np.random.uniform(1.5, 3)
                 else: transformed[:, i] = np.exp(dataset[:, i] / 3)
        # ... (other distribution types from original code if needed) ...
        elif distribution_type == 'mixed': # Example from original
            for i in range(n_features):
                r = i % 5
                if r == 0: transformed[:, i] = dataset[:, i] * np.random.uniform(0.5, 2)
                elif r == 1: transformed[:, i] = np.exp(dataset[:, i] / 2) - 1
                elif r == 2: transformed[:, i] = np.sin(dataset[:, i] * 2) + dataset[:, i] / 2
                elif r == 3: transformed[:, i] = dataset[:, i]**3 / 5 + dataset[:, i]
                else: transformed[:, i] = np.abs(dataset[:, i]) + np.random.uniform(-1, 1, n_samples) * 0.5
        # ... (multi_modal from original if needed) ...
        return transformed

    def generate_single_dataset(self, n_samples=1000, dist_type1='normal', dist_type2='skewed', label_noise=0.1):
        """
        Generates a *single* large dataset by combining two orthogonal bases,
        intended for later partitioning.
        Returns: X (np.ndarray), y (np.ndarray)
        """
        print(f"Generating synthetic base: {n_samples} samples, types '{dist_type1}', '{dist_type2}'")
        # Generate double samples needed, split later if necessary, or just use one type?
        # Let's adapt to generate one large block using combined orthogonal basis idea implicitly
        # Or simpler: just generate one type based on dist_type1?
        # Revisit: The original creator combined two *different* distributions for the *two* clients.
        # For partitioning, we need *one* base dataset. Let's simplify: generate based on dist_type1.
        # If you need feature distribution shifts correlated with label skew later, that's more complex.

        # Simplified Generation: Use one distribution type for the base dataset
        base1, _ = self.generate_base_distributions(n_samples) # Generate enough samples
        dataset = self.apply_nonlinear_transformations(base1, dist_type1)

        # Generate labels based on the combined dataset
        sums = np.sum(dataset, axis=1)
        median_val = np.median(sums)
        labels = (sums > median_val).astype(int)

        # Add label noise
        noise_mask = np.random.random(n_samples) < label_noise
        labels[noise_mask] = 1 - labels[noise_mask]

        print(f"Generated synthetic base dataset with shape {dataset.shape} and label distribution {np.bincount(labels)}")
        return dataset, labels

# --- DataPreprocessor Class ---

class DataPreprocessor:
    def __init__(self, dataset_name, batch_size):
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.dataset_config = get_parameters_for_dataset(dataset_name) # Get full config
        self.final_dataset_class = self._get_final_dataset_class()
        # Internal dispatch map
        self._processor_map = {
            'subset': self._process_subset,
            'xy_dict': self._process_xy_dict,
            'path_dict': self._process_path_dict,
        }

    def _get_final_dataset_class(self):
         """Gets the actual Dataset class (e.g., CIFARDataset) based on config."""
         class_name = self.dataset_config.get('dataset_class')
         if not class_name:
              raise ValueError(f"'dataset_class' not defined in config for {self.dataset_name}")
         # Try to get class from current module (data_processing.py)
         if hasattr(sys.modules[__name__], class_name):
              return getattr(sys.modules[__name__], class_name)
         else:
              # Add logic here if Dataset classes are defined elsewhere
              raise ImportError(f"Dataset class '{class_name}' not found in data_processing.py")

    def process_client_data(self, client_input_data: Dict, input_type: str):
        """Process data for multiple clients based on input type using internal dispatch."""
        processed_data = {} # Will store {client_id: (train_loader, val_loader, test_loader)}

        processor_func = self._processor_map.get(input_type)
        if processor_func is None:
            raise ValueError(f"Unknown input_type for DataPreprocessor: {input_type}")

        print(f"Preprocessing data using method for input type: '{input_type}'")
        for client_id, data in client_input_data.items():
             processed_data[client_id] = processor_func(data)

        return processed_data

    def _process_subset(self, client_subset: Subset):
        """Process data for a single client when input is a Subset."""
        client_indices = client_subset.indices
        if not client_indices:
             print(f"Warning: Client subset is empty. Returning empty DataLoaders.")
             empty_loader = DataLoader([])
             return (empty_loader, empty_loader, empty_loader)

        # Split indices into train/val/test
        # Using fixed seed here for consistent splits across runs for the same partition
        train_indices, val_indices, test_indices = self._split_indices(client_indices, seed=42)

        original_dataset = client_subset.dataset

        # Create NEW Subset objects pointing to the original dataset but with split indices
        train_subset = Subset(original_dataset, train_indices) if train_indices else None
        val_subset   = Subset(original_dataset, val_indices) if val_indices else None
        test_subset  = Subset(original_dataset, test_indices) if test_indices else None

        # Create DataLoaders
        # We assume the original_dataset (e.g. torchvision CIFAR10 object) applies transforms correctly
        train_loader = DataLoader(train_subset, batch_size=self.batch_size, shuffle=True, generator=g, num_workers=N_WORKERS) if train_subset else DataLoader([])
        val_loader = DataLoader(val_subset, batch_size=self.batch_size, shuffle=False, num_workers=N_WORKERS) if val_subset else DataLoader([])
        test_loader = DataLoader(test_subset, batch_size=self.batch_size, shuffle=False, num_workers=N_WORKERS) if test_subset else DataLoader([])

        return train_loader, val_loader, test_loader

    def _process_xy_dict(self, xy_data: Dict):
        """Process data for a single client when input is {'X': array, 'y': array}."""
        X, y = xy_data['X'], xy_data['y']
        if len(X) == 0:
             print(f"Warning: Client xy_dict data is empty. Returning empty DataLoaders.")
             empty_loader = DataLoader([])
             return (empty_loader, empty_loader, empty_loader)

        # Split these arrays
        train_data, val_data, test_data = self._split_data(X, y, seed=42)

        # Create specific final Dataset instances (e.g., SyntheticDataset)
        # These must handle X, y array inputs in __init__
        # Handle potential scaling for tabular data
        if 'standard_scale' in self.dataset_config.get('needs_preprocessing', []):
             print("Applying StandardScaler")
             scaler = StandardScaler().fit(train_data[0])
             train_data = (scaler.transform(train_data[0]), train_data[1])
             val_data = (scaler.transform(val_data[0]) if len(val_data[0]) > 0 else val_data[0], val_data[1])
             test_data = (scaler.transform(test_data[0]) if len(test_data[0]) > 0 else test_data[0], test_data[1])
        
        # Instantiate the final Dataset class using the split (and potentially scaled) arrays
        train_dataset = self.final_dataset_class(train_data[0], train_data[1], is_train=True) if len(train_data[0]) > 0 else None
        val_dataset = self.final_dataset_class(val_data[0], val_data[1], is_train=False) if len(val_data[0]) > 0 else None
        test_dataset = self.final_dataset_class(test_data[0], test_data[1], is_train=False) if len(test_data[0]) > 0 else None
        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, generator=g, num_workers=N_WORKERS) if train_dataset else DataLoader([])
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=N_WORKERS) if val_dataset else DataLoader([])
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=N_WORKERS) if test_dataset else DataLoader([])
        return train_loader, val_loader, test_loader

    def _process_path_dict(self, path_data: Dict):
        """Process data for a single client when input is {'X': paths, 'y': paths/labels}."""
        X_paths, y_data = path_data['X'], path_data['y']
        if len(X_paths) == 0:
             print(f"Warning: Client path_dict data is empty. Returning empty DataLoaders.")
             empty_loader = DataLoader([])
             return (empty_loader, empty_loader, empty_loader)

        # Split paths/labels
        train_data, val_data, test_data = self._split_data(X_paths, y_data, seed=42)

        # Create specific final Dataset instances (e.g., IXITinyDataset)
        # These must handle path inputs in __init__
        train_dataset = self.final_dataset_class(train_data[0], train_data[1], is_train=True) if len(train_data[0]) > 0 else None
        val_dataset = self.final_dataset_class(val_data[0], val_data[1], is_train=False) if len(val_data[0]) > 0 else None
        test_dataset = self.final_dataset_class(test_data[0], test_data[1], is_train=False) if len(test_data[0]) > 0 else None

        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, generator=g, num_workers=N_WORKERS) if train_dataset else DataLoader([])
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=N_WORKERS) if val_dataset else DataLoader([])
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=N_WORKERS) if test_dataset else DataLoader([])

        return train_loader, val_loader, test_loader

    # --- Splitting Helper Functions ---
    def _split_data(self, X, y, test_size=0.2, val_size=0.2, seed=42):
        """Splits arrays/lists (like X, y or paths) into train, val, test."""
        num_samples = len(X)
        if num_samples < 3: # Handle very small datasets
             print(f"Warning: Cannot split data with only {num_samples} samples. Using all for train.")
             return (X, y), (X[:0], y[:0]), (X[:0], y[:0]) # Return empty val/test

        indices = np.arange(num_samples)
        try:
             # Stratify if possible (classification task)
             if len(np.unique(y)) > 1 and len(y) == len(X):
                  stratify_param = y
             else:
                  stratify_param = None

             # Split into train+val and test
             idx_temp, idx_test = train_test_split(
                 indices, test_size=test_size, random_state=np.random.RandomState(seed), stratify=stratify_param
             )

             # Adjust val_size relative to the temp set
             relative_val_size = val_size / (1.0 - test_size) if (1.0 - test_size) > 0 else 0.0
             if relative_val_size >= 1.0: # Handle edge case
                  idx_train, idx_val = [], idx_temp
             elif len(idx_temp) < 2 or relative_val_size == 0: # Handle small temp set
                  idx_train, idx_val = idx_temp, []
             else:
                  stratify_temp = y[idx_temp] if stratify_param is not None else None
                  idx_train, idx_val = train_test_split(
                     idx_temp, test_size=relative_val_size, random_state=np.random.RandomState(seed + 1), stratify=stratify_temp # Use different seed for second split
                  )

        except ValueError as e:
             # Fallback to non-stratified if stratification fails (e.g., class with 1 sample)
             print(f"Warning: Stratified split failed ({e}), falling back to non-stratified split.")
             idx_temp, idx_test = train_test_split(indices, test_size=test_size, random_state=np.random.RandomState(seed))
             relative_val_size = val_size / (1.0 - test_size) if (1.0 - test_size) > 0 else 0.0
             if relative_val_size >= 1.0:
                  idx_train, idx_val = [], idx_temp
             elif len(idx_temp) < 2 or relative_val_size == 0:
                  idx_train, idx_val = idx_temp, []
             else:
                 idx_train, idx_val = train_test_split(idx_temp, test_size=relative_val_size, random_state=np.random.RandomState(seed + 1))


        # Return slices based on type of X and y
        if isinstance(X, np.ndarray):
            X_train, y_train = X[idx_train], y[idx_train]
            X_val, y_val = X[idx_val], y[idx_val]
            X_test, y_test = X[idx_test], y[idx_test]
        elif isinstance(X, list): # Handle lists (e.g., paths)
            X_train = [X[i] for i in idx_train]
            y_train = [y[i] for i in idx_train] # Assumes y is also indexable list/array
            X_val = [X[i] for i in idx_val]
            y_val = [y[i] for i in idx_val]
            X_test = [X[i] for i in idx_test]
            y_test = [y[i] for i in idx_test]
        else:
             raise TypeError(f"Unsupported data type for splitting: {type(X)}")

        return (X_train, y_train), (X_val, y_val), (X_test, y_test)


    def _split_indices(self, indices: List[int], test_size=0.2, val_size=0.2, seed=42):
        """Splits a list of indices into train, val, test index lists."""
        num_samples = len(indices)
        if num_samples < 3:
            print(f"Warning: Cannot split indices list with only {num_samples} samples.")
            return indices, [], []

        # Split into train+val and test indices
        # Cannot stratify here as we only have indices
        indices_temp, test_indices = train_test_split(
            indices, test_size=test_size, random_state=np.random.RandomState(seed)
        )

        # Adjust validation size relative to the remaining temp set
        relative_val_size = val_size / (1.0 - test_size) if (1.0 - test_size) > 0 else 0.0
        if relative_val_size >= 1.0:
            train_indices, val_indices = [], indices_temp
        elif len(indices_temp) < 2 or relative_val_size == 0:
             train_indices, val_indices = indices_temp, []
        else:
            train_indices, val_indices = train_test_split(
                indices_temp, test_size=relative_val_size, random_state=np.random.RandomState(seed + 1)
            )

        return train_indices, val_indices, test_indices


# --- Dataset Wrapper Classes (Ensure they handle expected inputs) ---
# These classes wrap the raw data/paths and apply transforms

class BaseDataset(TorchDataset): # Inherit from torch Dataset
    def __init__(self, X, y, is_train, **kwargs):
        self.X = X
        self.y = y
        self.is_train = is_train

    def __len__(self):
         # Handle cases where X might be empty after splits
         if isinstance(self.X, (np.ndarray, list, torch.Tensor)):
              return len(self.X)
         return 0 # Or raise error

    def get_transform(self):
        """Should be implemented by subclasses to return appropriate transform based on self.is_train."""
        raise NotImplementedError

    def __getitem__(self, idx):
        """Should be implemented by subclasses to load/transform item at idx."""
        raise NotImplementedError

    def get_scalers(self):
        """Only relevant for tabular data needing external scaling info."""
        return {}

class BaseTabularDataset(BaseDataset):
    # (Keep class code as provided in the previous response - handles both modes)
    # ... (init, fit_scaler, set_scaler, len, getitem) ...
    def __init__(self, base_dataset=None, indices=None, split_type=None, dataset_config=None, scaler_obj=None, X=None, y=None, **kwargs):
        super().__init__(**kwargs); self.base_dataset = base_dataset; self.indices = indices; self.split_type = split_type; self.config = dataset_config; self.scaler = scaler_obj; self.X_direct = X; self.y_direct = y
        self.needs_scaling = 'standard_scale' in self.config.get('needs_preprocessing', []) and self.config.get('dataset_class') != 'HeartDataset' # Check added for Heart
        self.is_regression = self.config.get('fixed_classes') is None; self.target_dtype = torch.float32 if self.is_regression else torch.long
        if self.base_dataset is not None and self.indices is not None: self.mode = 'subset'; assert self.X_direct is None and self.y_direct is None, "Cannot provide both base_dataset/indices and X/y direct."
        elif self.X_direct is not None and self.y_direct is not None: self.mode = 'direct'; assert self.base_dataset is None and self.indices is None, "Cannot provide both base_dataset/indices and X/y direct."
        elif self.indices is not None and not self.indices: self.mode = 'empty'
        else: raise ValueError("Must provide either (base_dataset, indices) or (X, y).")
        if self.mode == 'direct': self.X_tensor = torch.tensor(self.X_direct, dtype=torch.float32); self.y_tensor = torch.tensor(self.y_direct, dtype=self.target_dtype)

    def fit_scaler(self):
        if self.needs_scaling and self.split_type == 'train' and self.mode == 'subset' and self.indices:
            print(f"Fitting scaler on {self.split_type} subset data ({len(self.indices)} samples)..."); train_features = self.base_dataset.features[self.indices]; self.scaler = StandardScaler().fit(train_features); return self.scaler
        # Scaling for 'direct' mode (non-partitioned datasets) should happen in _process_xy_dict if needed. This function is primarily for the subset flow.
        return None

    def set_scaler(self, scaler): # Only used for subset mode val/test splits
        if self.needs_scaling and self.mode == 'subset': self.scaler = scaler

    def __len__(self): return len(self.indices) if self.mode == 'subset' else (len(self.X_direct) if self.mode == 'direct' else 0)

    def __getitem__(self, idx):
        if self.mode == 'empty': raise IndexError("Dataset is empty")
        if self.mode == 'subset':
            original_idx = self.indices[idx]; feature, label = self.base_dataset[original_idx];
            if self.scaler: feature = self.scaler.transform(feature.reshape(1, -1))[0]
            feature_tensor = torch.tensor(feature, dtype=torch.float32); label_tensor = torch.tensor(label, dtype=self.target_dtype); return feature_tensor, label_tensor
        elif self.mode == 'direct': # Assumes X_tensor/y_tensor created in init
            return self.X_tensor[idx], self.y_tensor[idx]


class SyntheticDataset(BaseTabularDataset): pass
class CreditDataset(BaseTabularDataset): pass
class HeartDataset(BaseTabularDataset): pass # Uses 'direct' mode as scaling is done in loader

class SyntheticBaseDataset(TorchDataset):
    """Wraps the raw generated synthetic data before partitioning."""
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        self.targets = labels # Crucial for partition_dirichlet_indices

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Return raw data, preprocessing happens later
        return self.features[idx], self.labels[idx]

class CreditBaseDataset(TorchDataset):
    """Wraps the raw loaded credit card data before partitioning."""
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        self.targets = labels # Crucial for partition_dirichlet_indices

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Return raw data, preprocessing happens later
        return self.features[idx], self.labels[idx]


# --- Image Dataset Classes ---
class BaseImageDataset(BaseDataset):
    """Base for image datasets handling numpy arrays or paths."""
    def __init__(self, X, y, is_train, **kwargs):
        super().__init__(X, y, is_train, **kwargs)
        self.transform = self.get_transform() # Get transform based on is_train

# Keep CIFAR/EMNIST/ISIC/IXI Dataset classes mostly as they were,
# ensuring their __init__ accepts the correct input types (numpy for CIFAR/EMNIST, paths for ISIC/IXI)
# and __getitem__ correctly loads/transforms the data using self.transform.

class EMNISTDataset(BaseImageDataset):
    """EMNIST Digits dataset handler (expects numpy X, y)"""
    def get_transform(self):
        # Correct normalization for EMNIST Digits (same as MNIST)
        mean, std = (0.1307,), (0.3081,)
        transforms_list = [transforms.ToPILImage()] # Start with PIL for consistency

        if self.X.ndim == 3: # Grayscale (H, W) -> add channel dim if needed by later transforms
              pass # ToPILImage handles H,W input
        elif self.X.ndim == 4 and self.X.shape[3] == 1: # Grayscale (N, H, W, C=1)
              pass # Handled below
        elif self.X.ndim == 4 and self.X.shape[3] == 3: # RGB (N, H, W, C=3) - unexpected?
             pass # Should work

        transforms_list.extend([
            transforms.Resize((28, 28)),
        ])

        if self.is_train:
             transforms_list.extend([
                transforms.RandomRotation((-15, 15)),
                transforms.RandomAffine(0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                # transforms.RandomPerspective(distortion_scale=0.2, p=0.5), # Can add more augmentations
            ])

        transforms_list.extend([
             transforms.ToTensor(), # Converts to (C, H, W) and scales to [0, 1]
             transforms.Normalize(mean, std)
        ])
        return transforms.Compose(transforms_list)

    def __getitem__(self, idx):
        # X is expected to be a numpy array (H, W) or (H, W, 1) from preprocessor
        image = self.X[idx]
        label = self.y[idx]

        # Apply transforms defined in get_transform
        image_tensor = self.transform(image)

        # Ensure correct label type
        label_tensor = torch.tensor(label, dtype=torch.long)
        return image_tensor, label_tensor


class CIFARDataset(BaseImageDataset):
    """CIFAR-10 dataset handler (expects numpy X, y)"""
    def get_transform(self):
        mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
        transforms_list = [transforms.ToPILImage()] # Input is HWC numpy array

        if self.is_train:
            transforms_list.extend([
                transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
                transforms.RandomHorizontalFlip(),
                # transforms.RandomRotation(15) # Optional
            ])

        transforms_list.extend([
            transforms.ToTensor(), # HWC -> CHW, scales to [0,1]
            transforms.Normalize(mean, std)
        ])
        return transforms.Compose(transforms_list)

    def __getitem__(self, idx):
        # X is expected to be HWC numpy array
        image = self.X[idx]
        label = self.y[idx]

        image_tensor = self.transform(image)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return image_tensor, label_tensor


class ISICDataset(BaseImageDataset):
    """ISIC dataset handler (expects image paths X, labels y)"""
    def __init__(self, image_paths, labels, is_train, **kwargs):
        # Uses Albumentations, needs image paths
        self.sz = 200 # Image size - could be moved to config
        super().__init__(image_paths, labels, is_train, **kwargs) # self.X=paths, self.y=labels

    def get_transform(self):
        # Keep existing Albumentations transforms based on self.is_train
        mean, std = (0.585, 0.500, 0.486), (0.229, 0.224, 0.225)
        if self.is_train:
            return albumentations.Compose([
                albumentations.RandomScale(0.07), albumentations.Rotate(50),
                albumentations.RandomBrightnessContrast(0.15, 0.1), albumentations.Flip(p=0.5),
                albumentations.Affine(shear=0.1), albumentations.RandomCrop(self.sz, self.sz),
                albumentations.CoarseDropout(max_holes=random.randint(1, 8), max_height=16, max_width=16), # Updated CoarseDropout args
                albumentations.Normalize(mean=mean, std=std, always_apply=True),
            ])
        else:
            return albumentations.Compose([
                albumentations.CenterCrop(self.sz, self.sz),
                albumentations.Normalize(mean=mean, std=std, always_apply=True),
            ])

    def __getitem__(self, idx):
        image_path = self.X[idx] # self.X contains paths
        label = self.y[idx]     # self.y contains labels

        try:
             # Read image as numpy array HWC for albumentations
             image = np.array(Image.open(image_path).convert('RGB')) # Ensure RGB
        except FileNotFoundError:
             print(f"Error: Image file not found at {image_path}. Returning None.")
             # Returning None might cause issues in DataLoader, consider placeholder or error
             return None, torch.tensor(label, dtype=torch.long) # Or raise error

        # Apply albumentations transforms
        transformed = self.transform(image=image)
        image = transformed['image'] # Result is HWC numpy array

        # Convert to tensor CHW float
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float()
        label_tensor = torch.tensor(label, dtype=torch.long)

        return image_tensor, label_tensor


class IXITinyDataset(BaseImageDataset):
    """IXITiny dataset handler (expects image paths X, label paths y)"""
    def __init__(self, image_paths, label_paths, is_train, **kwargs):
        # Needs nibabel and monai
        try:
             self.nib = nib
             self.monai_transforms = {'EnsureChannelFirst': EnsureChannelFirst, 'AsDiscrete': AsDiscrete, 'Compose': Compose, 'NormalizeIntensity': NormalizeIntensity, 'Resize': Resize, 'ToTensor': MonaiToTensor}
        except ImportError as e:
             raise ImportError(f"IXITinyDataset requires 'nibabel' and 'monai'. Install them. Original error: {e}")

        self.common_shape = (48, 60, 48) # Could move to config
        super().__init__(image_paths, label_paths, is_train, **kwargs) # self.X=img_paths, self.y=lbl_paths
        # Transform is applied in __getitem__ for MONAI

    def get_transform(self):
        # MONAI transforms often applied sequentially in __getitem__
        # This method could define parts of it if needed, based on self.is_train
        # Define base transforms here, applied in __getitem__
        Compose = self.monai_transforms['Compose']
        MonaiToTensor = self.monai_transforms['ToTensor']
        EnsureChannelFirst = self.monai_transforms['EnsureChannelFirst']
        Resize = self.monai_transforms['Resize']
        NormalizeIntensity = self.monai_transforms['NormalizeIntensity']

        self.image_transform = Compose([
            MonaiToTensor(), # Adds channel dim, scales to [0, 1]
            EnsureChannelFirst(channel_dim="no_channel"), # Ensure channel is first if ToTensor didn't do it
            Resize(self.common_shape),
            NormalizeIntensity() # Normalize after resize
        ])
        self.label_transform = Compose([
            MonaiToTensor(),
            EnsureChannelFirst(channel_dim="no_channel"),
            Resize(self.common_shape),
            self.monai_transforms['AsDiscrete'](to_onehot=2) # One-hot encode labels (assuming 2 classes)
        ])
        return None # Indicate transforms are handled internally

    def __getitem__(self, idx):
        image_path = self.X[idx]
        label_path = self.y[idx]

        try:
             image = self.nib.load(image_path).get_fdata(dtype=np.float32)
             label = self.nib.load(label_path).get_fdata(dtype=np.uint8) # Use uint8 for labels
        except FileNotFoundError as e:
             print(f"Error: Nifti file not found: {e}. Returning None.")
             # Handle appropriately, maybe return placeholder tensors?
             return None, None # Or raise error

        # Apply MONAI transforms
        image_tensor = self.image_transform(image)
        label_tensor = self.label_transform(label)

        # Ensure correct types after transforms
        return image_tensor.float(), label_tensor.float() # DICE loss often expects float