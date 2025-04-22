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
from sklearn.preprocessing import StandardScaler # Only if used by SynthGen directly
from typing import Dict, Tuple, Any, Optional, List, Union
from torch.utils.data import Dataset as TorchDataset
import traceback

# Import Base Dataset wrappers needed by loaders that return them
from data_sets import SyntheticBaseDataset, CreditBaseDataset

# --- Synthetic Data Generation Class ---
class SyntheticDataGenerator:
    """
    Generates synthetic tabular datasets with controlled properties.
    (Adapted from original dataCreator.py)
    """
    def __init__(self, n_features: int = 10, random_state: int = 42):
        self.n_features = n_features
        self.random_state = random_state
        # Ensure generator uses its own isolated random state
        self.rng = np.random.default_rng(random_state)

    def generate_orthogonal_matrices(self, dim: int) -> np.ndarray:
        """Generate orthogonal matrices using QR decomposition."""
        # Use the generator's RNG
        H = self.rng.standard_normal(size=(dim, dim))
        Q, R = np.linalg.qr(H)
        return Q

    def generate_base_distributions(self, n_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """Generate two base distributions with orthogonal principal components."""
        if self.n_features < 2:
             raise ValueError("n_features must be at least 2 for orthogonal split.")

        orthogonal_matrix = self.generate_orthogonal_matrices(self.n_features)
        # Ensure split dimensions are handled correctly, especially for odd n_features
        n_feat_1 = self.n_features // 2
        n_feat_2 = self.n_features - n_feat_1

        basis_1 = orthogonal_matrix[:, :n_feat_1]
        basis_2 = orthogonal_matrix[:, n_feat_1:]

        # Use the generator's RNG
        data_1 = self.rng.standard_normal(size=(n_samples, n_feat_1))
        data_2 = self.rng.standard_normal(size=(n_samples, n_feat_2))

        # Project data
        dataset_1 = data_1 @ basis_1.T
        dataset_2 = data_2 @ basis_2.T

        # Add noise
        noise_scale = 0.1
        dataset_1 += self.rng.normal(0, noise_scale, dataset_1.shape)
        dataset_2 += self.rng.normal(0, noise_scale, dataset_2.shape)

        return dataset_1, dataset_2

    def apply_nonlinear_transformations(self, dataset: np.ndarray,
                                       distribution_type: str = 'mixed') -> np.ndarray:
        """Apply non-linear transformations to create complex distributions."""
        transformed_dataset = np.copy(dataset)
        n_samples, n_features = dataset.shape

        if distribution_type == 'normal':
            for i in range(n_features):
                mean = self.rng.uniform(-3, 3)
                std = self.rng.uniform(0.5, 2)
                transformed_dataset[:, i] = dataset[:, i] * std + mean
        elif distribution_type == 'skewed':
            for i in range(n_features):
                rand_choice = i % 3 # Deterministic choice based on feature index
                if rand_choice == 0:
                    # Exponential-like
                    transformed_dataset[:, i] = np.exp(dataset[:, i] / 2.0)
                elif rand_choice == 1:
                    # Power-law-like
                    power = self.rng.uniform(1.5, 3.0)
                    transformed_dataset[:, i] = np.sign(dataset[:, i]) * np.abs(dataset[:, i])**power
                else:
                    # Log-normal-like
                    transformed_dataset[:, i] = np.exp(dataset[:, i] / 3.0)
        elif distribution_type == 'mixed':
            for i in range(n_features):
                rand_choice = i % 5
                if rand_choice == 0: # Normal
                    transformed_dataset[:, i] = dataset[:, i] * self.rng.uniform(0.5, 2.0)
                elif rand_choice == 1: # Exponential
                    transformed_dataset[:, i] = np.exp(dataset[:, i] / 2.0) - 1.0
                elif rand_choice == 2: # Sine wave modulation
                    transformed_dataset[:, i] = np.sin(dataset[:, i] * 2.0) + dataset[:, i] / 2.0
                elif rand_choice == 3: # Polynomial
                    transformed_dataset[:, i] = dataset[:, i]**3 / 5.0 + dataset[:, i]
                else: # Mixture with noise
                    transformed_dataset[:, i] = np.abs(dataset[:, i]) + self.rng.uniform(-1, 1, n_samples) * 0.5
        # Add other distribution types (e.g., 'multi_modal') here if needed, using self.rng
        else:
             print(f"Warning: Unknown distribution_type '{distribution_type}'. Returning untransformed data.")

        return transformed_dataset

    def generate_single_dataset(self,
                                n_samples: int = 1000,
                                dist_type1: str = 'normal',
                                label_noise: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """Generates a single large base dataset for partitioning."""
        print(f"Generating synthetic base dataset: {n_samples} samples, "
              f"distribution type '{dist_type1}', label noise {label_noise}")

        # Generate base orthogonal data (only need one part for single dataset)
        base_data, _ = self.generate_base_distributions(n_samples)

        # Apply transformations
        transformed_features = self.apply_nonlinear_transformations(base_data, dist_type1)

        # Generate labels based on a simple rule (e.g., sum threshold)
        feature_sums = np.sum(transformed_features, axis=1)
        median_sum = np.median(feature_sums)
        labels = (feature_sums > median_sum).astype(int)

        # Add label noise using the generator's RNG
        noise_mask = self.rng.random(n_samples) < label_noise
        labels[noise_mask] = 1 - labels[noise_mask] # Flip labels for noisy samples

        print(f"Generated synthetic base: features shape {transformed_features.shape}, "
              f"label distribution {np.bincount(labels)}")
        return transformed_features, labels

# --- Loader Functions ---

def load_torchvision_dataset(dataset_name: str,
                             data_dir: str,
                             source_args: Dict,
                             transform_config: Optional[Dict] = None
                             ) -> Tuple[TorchDataset, TorchDataset]:
    """Loads standard train and test sets for specified torchvision datasets."""
    transform_config = transform_config or {}
    tv_dataset_name = source_args.get('dataset_name')
    # Extract dataset-specific args, removing general ones
    tv_dataset_args = {
        k: v for k, v in source_args.items()
        if k not in ['dataset_name', 'data_dir']
    }
    # Use the main project data directory as root
    root_dir = data_dir

    print(f"Loading torchvision dataset: {tv_dataset_name} from {root_dir} "
          f"with args {tv_dataset_args}")

    if tv_dataset_name == 'CIFAR10':
        # Define default transforms if not provided
        train_transform = transform_config.get('train', transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ]))
        test_transform = transform_config.get('test', transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ]))
        # Load datasets
        train_dataset = CIFAR10(root=root_dir, train=True, download=True, transform=train_transform)
        test_dataset = CIFAR10(root=root_dir, train=False, download=True, transform=test_transform)
        return train_dataset, test_dataset

    elif tv_dataset_name == 'EMNIST':
        split = tv_dataset_args.get('split')
        if split != 'digits':
            # Forcing 'digits' as per config logic noted previously
            print(f"Warning: Requested EMNIST split '{split}', forcing 'digits'.")
            tv_dataset_args['split'] = 'digits'

        # Define default transforms
        train_transform = transform_config.get('train', transforms.Compose([
            transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))
        ]))
        test_transform = transform_config.get('test', transforms.Compose([
            transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))
        ]))
        # Load datasets
        train_dataset = EMNIST(root=root_dir, train=True, download=True,
                               transform=train_transform, **tv_dataset_args)
        test_dataset = EMNIST(root=root_dir, train=False, download=True,
                              transform=test_transform, **tv_dataset_args)
        return train_dataset, test_dataset

    else:
         raise ValueError(f"Torchvision loader not configured for dataset: {tv_dataset_name}")

def load_synthetic_base_data(dataset_name: str, data_dir: str, source_args: Dict) -> SyntheticBaseDataset:
    """Generates the base synthetic dataset using SyntheticDataGenerator."""
    print(f"Generating base Synthetic dataset with args: {source_args}")
    generator = SyntheticDataGenerator(
        n_features=source_args.get('n_features', 10),
        random_state=source_args.get('random_state', 42)
    )
    # Generate data using the generator instance
    features, labels = generator.generate_single_dataset(
        n_samples=source_args.get('base_n_samples', 10000),
        dist_type1=source_args.get('dist_type1', 'normal'),
        # dist_type2=source_args.get('dist_type2', 'skewed'), # Removed as unused
        label_noise=source_args.get('label_noise', 0.05)
    )
    # Return the Base Dataset wrapper containing the raw generated data
    return SyntheticBaseDataset(features, labels)

def load_credit_base_data(dataset_name: str, data_dir: str, source_args: Dict) -> CreditBaseDataset:
    """Loads the full Credit Card dataset from the specified CSV file."""
    csv_path = source_args.get('csv_path')
    drop_cols = source_args.get('drop_cols', [])

    if not csv_path or not os.path.exists(csv_path):
        raise FileNotFoundError(f"Credit card CSV file not found at specified path: {csv_path}")

    print(f"Loading base Credit dataset from: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
         raise IOError(f"Error reading Credit Card CSV file '{csv_path}': {e}") from e

    # Drop specified columns if they exist
    cols_to_drop_exist = [col for col in drop_cols if col in df.columns]
    if cols_to_drop_exist:
        print(f"Dropping columns: {cols_to_drop_exist}")
        df = df.drop(columns=cols_to_drop_exist)

    # Ensure 'Class' column exists for labels
    if 'Class' not in df.columns:
        raise ValueError("Label column 'Class' not found in the Credit Card dataset.")

    # Separate features (X) and labels (y)
    labels = df['Class'].values
    features = df.drop(columns=['Class']).values

    print(f"Loaded Credit base dataset: features shape {features.shape}, "
          f"label distribution {np.bincount(labels)}")
    # Return the Base Dataset wrapper containing the raw loaded data
    return CreditBaseDataset(features, labels)

def load_heart_site_data(dataset_name: str,
                         data_dir: str,
                         client_num: int,
                         cost_key: Any, # The key from site_mappings (e.g., 1, 2, ...)
                         config: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads and preprocesses data for specific Heart site(s) assigned to a client.
    Applies pre-defined scaling based on values provided in the config.

    Args:
        dataset_name: Name of the dataset ('Heart').
        data_dir: Base directory for data.
        client_num: The 1-based index of the client.
        cost_key: The key corresponding to the site mapping in the config.
        config: The full dataset configuration dictionary from configs.py.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Processed features (X) and labels (y) for the client.
                                       Returns empty arrays if loading/processing fails.
    """
    source_args = config.get('source_args', {})
    # Extract necessary info from source_args with checks
    data_root = source_args.get('data_dir', os.path.join(data_dir, dataset_name))
    site_map = source_args.get('site_mappings')
    sites_info = source_args.get('sites')
    used_columns = source_args.get('used_columns')
    feature_names = source_args.get('feature_names')
    cols_to_scale = source_args.get('cols_to_scale')
    scale_values = source_args.get('scale_values')

    # Validate required configuration elements
    if not all([site_map, sites_info, used_columns, feature_names, cols_to_scale, scale_values]):
         raise ValueError("Heart dataset source_args configuration is incomplete.")
    if cost_key not in site_map:
         raise ValueError(f"Invalid cost key '{cost_key}' provided for Heart site mapping.")
    if not (1 <= client_num <= len(site_map[cost_key])):
         raise ValueError(f"Invalid client number {client_num} for cost key '{cost_key}' "
                          f"(max: {len(site_map[cost_key])}).")

    # Get the list of site name(s) assigned to this specific client for this cost key
    assigned_sites: List[str] = site_map[cost_key][client_num - 1]
    print(f"Loading Heart data for client {client_num}, assigned site(s): {assigned_sites} "
          f"(cost_key={cost_key})")

    client_features_list: List[np.ndarray] = []
    client_labels_list: List[np.ndarray] = []

    # Load data for each assigned site
    for site_name in assigned_sites:
        if site_name not in sites_info:
            print(f"Warning: Configured site name '{site_name}' not found in 'sites' list. Skipping.")
            continue

        file_path = os.path.join(data_root, f'processed.{site_name}.data')
        print(f"  Loading site: {site_name} from {file_path}")

        try:
            # Load data, handling potential missing values ('?')
            site_df = pd.read_csv(
                file_path,
                names=used_columns,      # Use predefined column names
                na_values='?',           # Define missing value marker
                header=None,             # No header row in the file
                usecols=used_columns     # Read only necessary columns
            ).dropna() # Drop rows with any missing values in used columns

            if site_df.empty:
                print(f"  Warning: No valid data found for site '{site_name}' after dropna. Skipping.")
                continue

            # Make a copy for scaling to avoid modifying the original DataFrame used for target
            scaled_features_df = site_df.copy()

            # Apply pre-defined scaling using provided mean/variance
            for col in cols_to_scale:
                if col in scaled_features_df.columns and col in scale_values:
                    mean, variance = scale_values[col]
                    std_dev = np.sqrt(variance)
                    # Avoid division by zero if variance is effectively zero
                    if std_dev > 1e-9:
                        scaled_features_df[col] = (scaled_features_df[col] - mean) / std_dev
                    else:
                        # If variance is zero, just center the data
                        scaled_features_df[col] = scaled_features_df[col] - mean
                elif col in scaled_features_df.columns:
                     print(f"  Warning: Column '{col}' specified for scaling but no scale_values found.")


            # Verify required columns exist before extracting
            if 'target' not in site_df.columns:
                 print(f"  Warning: Target column missing for site '{site_name}'. Skipping.")
                 continue
            missing_features = [f for f in feature_names if f not in scaled_features_df.columns]
            if missing_features:
                 print(f"  Warning: Missing feature columns {missing_features} for site '{site_name}'. Skipping.")
                 continue

            # Extract features (scaled) and labels (original, converted to int)
            X_site = scaled_features_df[feature_names].values
            y_site = site_df['target'].astype(int).values

            client_features_list.append(X_site)
            client_labels_list.append(y_site)
            print(f"  Successfully processed site '{site_name}' ({len(X_site)} samples)")

        except FileNotFoundError:
            print(f"Warning: Data file not found for site '{site_name}' at {file_path}. Skipping.")
        except Exception as e:
            print(f"Warning: Error processing site '{site_name}': {e}. Skipping.")
            traceback.print_exc() # Print traceback for unexpected errors

    # Check if any data was successfully loaded for this client
    if not client_features_list:
         print(f"Warning: No data loaded for client {client_num} across assigned sites {assigned_sites}.")
         # Return empty arrays with correct feature dimension for consistency
         return np.array([]).reshape(0, len(feature_names)), np.array([])

    # Concatenate data if multiple sites were assigned to this client
    final_X = np.concatenate(client_features_list, axis=0)
    final_y = np.concatenate(client_labels_list, axis=0)

    print(f"  -> Client {client_num} combined data shape: X={final_X.shape}, y={final_y.shape}")
    return final_X, final_y


def load_ixi_client_paths(dataset_name: str,
                          data_dir: str,
                          client_num: int,
                          cost_key: Any, # e.g., 0.08, 0.28, 'all'
                          config: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """Loads NIfTI image and label file paths for one IXI client based on site mapping."""
    source_args = config.get('source_args', {})
    root_dir = source_args.get('data_dir', os.path.join(data_dir, dataset_name))
    sites_map = source_args.get('site_mappings')

    # Validate config
    if not sites_map:
         raise ValueError(f"Missing 'site_mappings' in 'source_args' for {dataset_name} config.")
    if cost_key not in sites_map:
        raise ValueError(f"Invalid cost key '{cost_key}' for IXITiny site mapping. "
                         f"Available keys: {list(sites_map.keys())}")

    # Validate client number against available sites for this cost key
    available_sites_for_cost = sites_map[cost_key]
    if not (1 <= client_num <= len(available_sites_for_cost)):
         raise ValueError(f"Client number {client_num} out of range "
                          f"(1 to {len(available_sites_for_cost)} available) "
                          f"for cost key '{cost_key}' in IXITiny mapping.")

    # Get the list of site names (e.g., ['Guys'], ['HH']) for this client
    site_names_for_client: List[str] = available_sites_for_cost[client_num - 1]
    print(f"Loading IXI paths for client {client_num}, assigned site(s): {site_names_for_client} "
          f"(cost_key={cost_key})")

    image_files: List[str] = []
    label_files: List[str] = []
    # Assume standard Flamby directory structure
    image_dir = os.path.join(root_dir, 'flamby/image')
    label_dir = os.path.join(root_dir, 'flamby/label')

    if not os.path.isdir(image_dir) or not os.path.isdir(label_dir):
         raise FileNotFoundError(f"Required IXI data directories not found at {image_dir} or {label_dir}")

    # Collect files matching the site names
    for site_name_part in site_names_for_client:
        image_files.extend(glob.glob(os.path.join(image_dir, f'*{site_name_part}*.nii.gz')))
        label_files.extend(glob.glob(os.path.join(label_dir, f'*{site_name_part}*.nii.gz')))

    # --- Align image and label files based on common ID ---
    def get_ixi_id(path: str) -> str:
         """Extracts base ID (e.g., IXI002) from filename."""
         base = os.path.basename(path)
         # Example: IXI002_Guys_082_T1_slice_100_label.nii.gz -> IXI002
         parts = base.split('_')
         return parts[0]

    labels_dict = {get_ixi_id(path): path for path in label_files}
    images_dict = {get_ixi_id(path): path for path in image_files}

    # Find common IDs and sort them
    common_keys = sorted(list(set(labels_dict.keys()) & set(images_dict.keys())))

    if not common_keys:
         print(f"Warning: No matching image/label pairs found for IXI client {client_num}, "
               f"sites {site_names_for_client}")
         # Return empty arrays
         return np.array([]), np.array([])

    # Create aligned lists of paths
    aligned_image_files = [images_dict[key] for key in common_keys]
    aligned_label_files = [labels_dict[key] for key in common_keys]

    print(f"  Found {len(aligned_image_files)} matching image/label pairs.")
    return np.array(aligned_image_files), np.array(aligned_label_files)


def load_isic_client_paths(dataset_name: str,
                           data_dir: str,
                           client_num: int,
                           cost_key: Any, # e.g., 0.06, 'all'
                           config: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """Loads image file paths and corresponding labels for one ISIC client based on site mapping."""
    source_args = config.get('source_args', {})
    root_dir = source_args.get('data_dir', os.path.join(data_dir, dataset_name))
    site_map = source_args.get('site_mappings')

    # Validate config
    if not site_map:
         raise ValueError(f"Missing 'site_mappings' in 'source_args' for {dataset_name} config.")
    if cost_key not in site_map:
         raise ValueError(f"Invalid cost key '{cost_key}' for ISIC site mapping. "
                          f"Available keys: {list(site_map.keys())}")

    # Validate client number
    available_sites_for_cost = site_map[cost_key]
    if not (1 <= client_num <= len(available_sites_for_cost)):
         raise ValueError(f"Client number {client_num} out of range "
                          f"(1 to {len(available_sites_for_cost)} available) "
                          f"for cost key '{cost_key}' in ISIC mapping.")

    # Get the list of site indices (e.g., [2]) assigned to this client
    site_indices_for_client: List[int] = available_sites_for_cost[client_num - 1]
    print(f"Loading ISIC paths for client {client_num}, assigned site index(es): {site_indices_for_client} "
          f"(cost_key={cost_key})")

    image_files: List[str] = []
    labels: List[int] = []
    # Define expected image directory
    image_dir = os.path.join(root_dir, 'ISIC_2019_Training_Input_preprocessed')
    if not os.path.isdir(image_dir):
          raise FileNotFoundError(f"ISIC preprocessed image directory not found: {image_dir}")

    # Check for sampling configuration
    sampling_config = config.get('sampling_config')
    nrows_per_site = None
    if sampling_config and sampling_config.get('type') == 'fixed_total':
         nrows_per_site = sampling_config.get('size') # Number of rows to read per site CSV

    # Load data from CSV for each assigned site index
    for site_index in site_indices_for_client:
        site_csv_path = os.path.join(root_dir, f'site_{site_index}_files_used.csv')
        if not os.path.exists(site_csv_path):
             print(f"Warning: ISIC site file not found: {site_csv_path}. Skipping site index {site_index}.")
             continue

        if nrows_per_site is not None:
             print(f"  Sampling ISIC client {client_num} (site {site_index}) to {nrows_per_site} rows")

        try:
            files_df = pd.read_csv(site_csv_path, nrows=nrows_per_site)
            # Construct full image paths and collect labels
            image_files.extend([os.path.join(image_dir, f"{file_stem}.jpg") for file_stem in files_df['image']])
            labels.extend(files_df['label'].values)
            print(f"  Loaded {len(files_df)} samples from site {site_index}.")
        except Exception as e:
             print(f"Error reading or processing ISIC site CSV {site_csv_path}: {e}")
             continue # Skip this site if error occurs

    if not image_files:
         print(f"Warning: No images loaded for client {client_num} across sites {site_indices_for_client}.")

    return np.array(image_files), np.array(labels)

# --- Dispatch Dictionary ---
# Maps data_source names from config to the appropriate loading function
DATA_LOADERS: Dict[str, callable] = {
    # Base loaders (return BaseDataset wrapper or tuple[Dataset, Dataset])
    'torchvision': load_torchvision_dataset,
    'synthetic_base': load_synthetic_base_data,
    'credit_base': load_credit_base_data,

    # Per-client / Site loaders (return Tuple[np.ndarray, np.ndarray])
    'heart_site_loader': load_heart_site_data,
    'pre_split_paths_ixi': load_ixi_client_paths,
    'pre_split_paths_isic': load_isic_client_paths,
    # Removed 'pre_split_csv' as it's less general than site loaders now
}