"""
Handles loading initial data from various sources.
- Loads full datasets for partitioning (torchvision, generated, full CSVs).
- Loads data per client for pre-split strategies (site files, path lists, per-client generation).
"""
import os
import glob
import sys
import numpy as np
import pandas as pd
import torch
from torchvision.datasets import CIFAR10, EMNIST
from torchvision import transforms # Keep transforms here for defaults
from sklearn.preprocessing import StandardScaler # Only if used by SynthGen directly
from typing import Dict, Tuple, Any, Optional, List, Union
from torch.utils.data import Dataset as TorchDataset
import traceback
import hashlib # NEW: For better seeding

# MODIFIED: Import ONLY from data_sets.py
from data_sets import SyntheticBaseDataset, CreditBaseDataset, SyntheticFeaturesBaseDataset

# --- Synthetic Data Generation Class ---
# (Keep SyntheticDataGenerator class exactly as provided previously)
class SyntheticDataGenerator:
    def __init__(self, n_features: int = 10, random_state: int = 42):
        self.n_features = n_features
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)
    def generate_orthogonal_matrices(self, dim: int) -> np.ndarray:
        H = self.rng.standard_normal(size=(dim, dim)); Q, _ = np.linalg.qr(H); return Q
    def generate_base_distributions(self, n_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        if self.n_features < 2: raise ValueError("n_features must be >= 2")
        orth_mat=self.generate_orthogonal_matrices(self.n_features); n1=self.n_features//2; n2=self.n_features-n1
        b1=orth_mat[:,:n1]; b2=orth_mat[:,n1:]; d1=self.rng.standard_normal(size=(n_samples,n1)); d2=self.rng.standard_normal(size=(n_samples,n2))
        ds1=d1 @ b1.T; ds2=d2 @ b2.T; ds1+=self.rng.normal(0,0.1,ds1.shape); ds2+=self.rng.normal(0,0.1,ds2.shape); return ds1, ds2
    def apply_nonlinear_transformations(self, dataset: np.ndarray, distribution_type: str = 'mixed') -> np.ndarray:
        t = np.copy(dataset); n_samples, n_features = dataset.shape
        if distribution_type == 'normal':
            for i in range(n_features): mean=self.rng.uniform(-3,3); std=self.rng.uniform(0.5,2); t[:,i]=dataset[:,i]*std+mean
        elif distribution_type=='skewed':
            for i in range(n_features): r=i%3; t[:,i]=np.exp(dataset[:,i]/2.) if r==0 else (np.sign(dataset[:,i])*np.abs(dataset[:,i])**self.rng.uniform(1.5,3) if r==1 else np.exp(dataset[:,i]/3.))
        elif distribution_type=='mixed':
            for i in range(n_features): r=i%5; t[:,i]=dataset[:,i]*self.rng.uniform(0.5,2) if r==0 else (np.exp(dataset[:,i]/2.)-1 if r==1 else (np.sin(dataset[:,i]*2)+dataset[:,i]/2. if r==2 else (dataset[:,i]**3/5.+dataset[:,i] if r==3 else np.abs(dataset[:,i])+self.rng.uniform(-1,1,n_samples)*0.5)))
        else: print(f"Warn: Unknown dist_type '{distribution_type}'")
        return t
    def generate_single_dataset(self, n_samples: int=1000, dist_type1: str='normal', label_noise: float=0.1) -> Tuple[np.ndarray, np.ndarray]:
        print(f"Gen dataset: {n_samples} samples, type '{dist_type1}', noise {label_noise}, seed {self.random_state}"); b1,_=self.generate_base_distributions(n_samples); features=self.apply_nonlinear_transformations(b1,dist_type1); sums=np.sum(features,axis=1); med=np.median(sums); labels=(sums>med).astype(int); mask=self.rng.random(n_samples)<label_noise; labels[mask]=1-labels[mask]; print(f"  -> Shape {features.shape}, Labels {np.bincount(labels)}"); return features,labels
    def generate_features_only(self, n_samples: int = 1000, dist_type1: str = 'normal') -> np.ndarray:
         print(f"Gen features only: {n_samples} samples, type '{dist_type1}', seed {self.random_state}"); b1,_=self.generate_base_distributions(n_samples); features=self.apply_nonlinear_transformations(b1,dist_type1); print(f"  -> Shape {features.shape}"); return features

# --- Loader Functions ---

# MODIFIED: Accept transform_config from dataset config
def load_torchvision_dataset(dataset_name: str, data_dir: str, source_args: Dict, transform_config: Optional[Dict] = None) -> Tuple[TorchDataset, TorchDataset]:
    """Loads torchvision datasets, using provided transforms or defaults."""
    # Note: transform_config is now passed from pipeline._initialize_experiment,
    # sourcing from DEFAULT_PARAMS[dataset_name].get('transform_config', {})
    transform_config = transform_config or {} # Ensure it's a dict

    tv_dataset_name = source_args.get('dataset_name')
    tv_dataset_args = {k: v for k, v in source_args.items() if k not in ['dataset_name', 'data_dir']}
    root_dir = data_dir # Use main data_dir
    print(f"Loading torchvision: {tv_dataset_name} from {root_dir} with args {tv_dataset_args}")

    if tv_dataset_name == 'CIFAR10':
        # Use transform from config if provided, otherwise use hardcoded default
        train_transform = transform_config.get('train') or transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
        test_transform = transform_config.get('test') or transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
        train_dataset = CIFAR10(root=root_dir, train=True, download=True, transform=train_transform)
        test_dataset = CIFAR10(root=root_dir, train=False, download=True, transform=test_transform)
        return train_dataset, test_dataset

    elif tv_dataset_name == 'EMNIST':
        split = tv_dataset_args.get('split')
        assert split == 'digits', "Only EMNIST 'digits' split is currently supported by default transforms."
        train_transform = transform_config.get('train') or transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        test_transform = transform_config.get('test') or transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = EMNIST(root=root_dir, train=True, download=True, transform=train_transform, **tv_dataset_args)
        test_dataset = EMNIST(root=root_dir, train=False, download=True, transform=test_transform, **tv_dataset_args)
        return train_dataset, test_dataset
    else:
        raise ValueError(f"Torchvision loader not configured for: {tv_dataset_name}")

# --- Other Loaders (Keep as is unless specific feedback applies) ---

def load_synthetic_base_data(dataset_name: str, data_dir: str, source_args: Dict) -> SyntheticBaseDataset:
    print(f"Generating base Synthetic dataset with args: {source_args}"); generator = SyntheticDataGenerator(n_features=source_args.get('n_features', 10), random_state=source_args.get('random_state', 42)); features, labels = generator.generate_single_dataset(n_samples=source_args.get('base_n_samples', 10000), dist_type1=source_args.get('dist_type1', 'normal'), label_noise=source_args.get('label_noise', 0.05)); return SyntheticBaseDataset(features, labels) # Returns correct type from data_sets

def load_credit_base_data(dataset_name: str, data_dir: str, source_args: Dict) -> CreditBaseDataset:
    csv_path = source_args.get('csv_path'); drop_cols = source_args.get('drop_cols', []); assert csv_path and os.path.exists(csv_path), f"Credit card CSV not found at {csv_path}"; print(f"Loading base Credit dataset from: {csv_path}"); df = pd.read_csv(csv_path);
    if drop_cols: print(f"Dropping columns: {drop_cols}"); df = df.drop(columns=drop_cols, errors='ignore')
    if 'Class' not in df.columns: raise ValueError("'Class' column not found"); labels = df['Class'].values; features = df.drop(columns=['Class']).values; print(f"Loaded Credit base: shape {features.shape}, labels {np.bincount(labels)}"); return CreditBaseDataset(features, labels) # Returns correct type from data_sets

def load_heart_site_data(dataset_name: str, data_dir: str, client_num: int, cost_key: Any, config: Dict) -> Tuple[np.ndarray, np.ndarray]:
    source_args = config.get('source_args', {}); data_root = source_args.get('data_dir', os.path.join(data_dir, dataset_name)); site_map = source_args.get('site_mappings'); sites_info = source_args.get('sites'); used_columns = source_args.get('used_columns'); feature_names = source_args.get('feature_names'); cols_to_scale = source_args.get('cols_to_scale'); scale_values = source_args.get('scale_values'); assert all([site_map, sites_info, used_columns, feature_names, cols_to_scale, scale_values]), "Heart source_args incomplete."; assert cost_key in site_map, f"Invalid cost key '{cost_key}' for Heart."; assert 1 <= client_num <= len(site_map[cost_key]), f"Client num {client_num} invalid for cost '{cost_key}'."
    assigned_sites = site_map[cost_key][client_num - 1]; print(f"Loading Heart data for client {client_num}, site(s): {assigned_sites} (cost_key={cost_key})"); features_list, labels_list = [], []
    for site_name in assigned_sites:
        if site_name not in sites_info: print(f"Warn: Site '{site_name}' not found. Skipping."); continue
        fpath = os.path.join(data_root, f'processed.{site_name}.data'); print(f"  Loading: {fpath}")
        try:
            site_df = pd.read_csv(fpath, names=used_columns, na_values='?', header=None, usecols=used_columns).dropna();
            if site_df.empty: print(f"  Warn: No valid data for site '{site_name}'. Skipping."); continue; scaled_df = site_df.copy()
            for col in cols_to_scale:
                if col in scaled_df.columns and col in scale_values: mean, var = scale_values[col]; std = np.sqrt(var); scaled_df[col] = (scaled_df[col]-mean)/std if std > 1e-9 else scaled_df[col]-mean
                elif col in scaled_df.columns: print(f"  Warn: No scale_values for col '{col}'.")
            if 'target' not in site_df.columns or not all(f in scaled_df.columns for f in feature_names): print(f"  Warn: Missing columns for site '{site_name}'. Skipping."); continue
            features_list.append(scaled_df[feature_names].values); labels_list.append(site_df['target'].astype(int).values)
        except FileNotFoundError: print(f"Warn: File not found for site '{site_name}'. Skipping.")
        except Exception as e: print(f"Warn: Error processing site '{site_name}': {e}. Skipping."); traceback.print_exc()
    if not features_list: print(f"Warn: No data loaded for client {client_num}."); return np.array([]).reshape(0, len(feature_names)), np.array([])
    final_X = np.concatenate(features_list, axis=0); final_y = np.concatenate(labels_list, axis=0); print(f"  -> Client {client_num} Heart data: X={final_X.shape}, y={final_y.shape}"); return final_X, final_y

def load_ixi_client_paths(dataset_name: str, data_dir: str, client_num: int, cost_key: Any, config: Dict) -> Tuple[np.ndarray, np.ndarray]:
    source_args = config.get('source_args', {}); root_dir = source_args.get('data_dir', os.path.join(data_dir, dataset_name)); sites_map = source_args.get('site_mappings'); assert sites_map, f"Missing 'site_mappings' for {dataset_name}"; assert cost_key in sites_map, f"Invalid cost key '{cost_key}' for IXITiny"; available_for_cost = len(sites_map[cost_key]); assert 1 <= client_num <= available_for_cost, f"Client num {client_num} out of range for IXITiny cost '{cost_key}'"
    site_names = sites_map[cost_key][client_num - 1]; print(f"Loading IXI paths for client {client_num}, site(s): {site_names} (cost_key={cost_key})"); img_files, lbl_files = [], []; img_dir = os.path.join(root_dir,'flamby/image'); lbl_dir = os.path.join(root_dir,'flamby/label'); assert os.path.isdir(img_dir) and os.path.isdir(lbl_dir), f"IXI dirs not found"
    for name in site_names: img_files.extend(glob.glob(os.path.join(img_dir,f'*{name}*.nii.gz'))); lbl_files.extend(glob.glob(os.path.join(lbl_dir,f'*{name}*.nii.gz')))
    def get_id(p): return os.path.basename(p).split('_')[0]
    lbl_d={get_id(p): p for p in lbl_files}; img_d={get_id(p): p for p in img_files}; common=sorted(list(set(lbl_d)&set(img_d))); assert common, f"No matching pairs for IXI client {client_num}"
    img_paths=[img_d[k] for k in common]; lbl_paths=[lbl_d[k] for k in common]; print(f"  Found {len(img_paths)} pairs."); return np.array(img_paths), np.array(lbl_paths)

def load_isic_client_paths(dataset_name: str, data_dir: str, client_num: int, cost_key: Any, config: Dict) -> Tuple[np.ndarray, np.ndarray]:
    source_args = config.get('source_args', {}); root_dir = source_args.get('data_dir', os.path.join(data_dir, dataset_name)); site_map = source_args.get('site_mappings'); assert site_map, f"Missing 'site_mappings' for {dataset_name}"; assert cost_key in site_map, f"Invalid cost key '{cost_key}' for ISIC"; available_for_cost = len(site_map[cost_key]); assert 1 <= client_num <= available_for_cost, f"Client num {client_num} out of range for ISIC cost '{cost_key}'"
    site_indices = site_map[cost_key][client_num - 1]; print(f"Loading ISIC paths for client {client_num}, site index(es): {site_indices} (cost_key={cost_key})"); img_files, labels = [], []; img_dir = os.path.join(root_dir,'ISIC_2019_Training_Input_preprocessed'); assert os.path.isdir(img_dir), f"ISIC image dir not found"
    sampling_cfg = config.get('sampling_config'); nrows = sampling_cfg.get('size') if sampling_cfg and sampling_cfg.get('type') == 'fixed_total' else None
    for site_idx in site_indices:
        csv_path = os.path.join(root_dir, f'site_{site_idx}_files_used.csv'); assert os.path.exists(csv_path), f"ISIC site file not found: {csv_path}"
        if nrows: print(f"  Sampling ISIC site {site_idx} to {nrows} rows")
        try: files_df = pd.read_csv(csv_path, nrows=nrows); img_files.extend([os.path.join(img_dir, f"{stem}.jpg") for stem in files_df['image']]); labels.extend(files_df['label'].values); print(f"  Loaded {len(files_df)} from site {site_idx}.")
        except Exception as e: print(f"Error reading ISIC site CSV {csv_path}: {e}")
    if not img_files: print(f"Warn: No images loaded for client {client_num}.")
    return np.array(img_files), np.array(labels)

# --- NEW Loader for Feature Skew ---
# MODIFIED: Fix seed collision
def load_generate_synthetic_feature_skew_client(dataset_name: str,
                                                data_dir: str, # Unused, but consistent signature
                                                client_num: int,
                                                cost_key: Any, # Interpreted as feature_shift_param
                                                config: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """Generates synthetic data per client with varying feature distributions."""
    source_args = config.get('source_args', {})
    n_samples = source_args.get('n_samples_per_client', 1000)
    n_features = source_args.get('n_features', 10)
    label_noise = source_args.get('label_noise', 0.05)
    shift_mapping = source_args.get('shift_mapping', {})
    base_seed = source_args.get('base_random_state', 42)

    # --- Determine client-specific generation parameters ---
    try:
        # NEW: Handle 'all' sentinel from translate_cost
        if isinstance(cost_key, dict) and cost_key.get('cost_key_is_all'):
             # Define behavior for 'all' - e.g., map to max shift (1.0)
             shift_param = 1.0
             print(f"Warning: cost_key 'all' encountered for feature skew, using shift_param={shift_param}")
        elif isinstance(cost_key, (int, float)):
             shift_param = float(cost_key)
             shift_param = np.clip(shift_param, 0.0, 1.0) # Ensure range
        else:
             raise ValueError(f"Invalid cost_key '{cost_key}' for feature_shift_param interpretation.")
    except (ValueError, TypeError) as e:
         raise ValueError(f"Error interpreting cost_key '{cost_key}' for feature_shift_param: {e}")


    # MODIFIED: Use hashing for a more robust client-specific seed
    # Incorporate base seed, client number, and the *exact* shift parameter
    seed_string = f"{base_seed}-{client_num}-{shift_param:.8f}" # Use precise shift param
    client_seed = int(hashlib.sha256(seed_string.encode('utf-8')).hexdigest(), 16) % (2**32)

    gen_params = {'n_features': n_features, 'random_state': client_seed}

    # Apply shift based on mapping defined in config['source_args']['shift_mapping']
    param_to_vary = shift_mapping.get('param_to_vary')
    base_value = shift_mapping.get('base_value')
    shifted_values = shift_mapping.get('shifted_values')

    # Determine the specific parameter value for this client based on shift_param
    if param_to_vary and base_value is not None and shifted_values:
         if param_to_vary == 'dist_type':
             num_options = 1 + len(shifted_values)
             # Linear interpolation index (safer than direct multiplication)
             # Map shift_param [0, 1] to index [0, num_options-1]
             idx = int(round(shift_param * (num_options - 1)))
             idx = min(idx, num_options - 1) # Ensure index is within bounds

             if idx == 0:
                  gen_params['dist_type1'] = base_value
             else:
                  gen_params['dist_type1'] = shifted_values[idx-1] # Index into shifted_values
             print(f"  Client {client_num} (shift {shift_param:.2f}): Using dist_type '{gen_params['dist_type1']}' (Seed: {client_seed})")
         # Add logic for other varying parameters here
         else:
              print(f"Warning: shift_mapping for '{param_to_vary}' not implemented.")
              gen_params['dist_type1'] = base_value # Fallback
    else:
         # Default if no valid shift mapping defined or needed
         gen_params['dist_type1'] = source_args.get('dist_type1', 'mixed')

    # --- Generate Data for this Client ---
    generator = SyntheticDataGenerator(**gen_params)
    X_client, y_client = generator.generate_single_dataset(
        n_samples=n_samples,
        dist_type1=gen_params['dist_type1'], # Pass the determined type
        label_noise=label_noise
    )
    # NOTE: Feature scaling is NOT applied here. It needs to be handled
    # by the DataPreprocessor based on the 'needs_preprocessing' config.
    # Consider adding a global scaling step if needed for feature skew stability.
    return X_client, y_client

# --- NEW Loader for Concept Skew (Features Only) ---
# MODIFIED: Ensure it returns the correct type from data_sets
def load_synthetic_features_only(dataset_name: str, data_dir: str, source_args: Dict) -> SyntheticFeaturesBaseDataset:
    """Generates only the feature pool for concept skew experiments."""
    print(f"Generating base features for Synthetic_Concept with args: {source_args}")
    generator = SyntheticDataGenerator(
        n_features=source_args.get('n_features', 10),
        random_state=source_args.get('random_state', 42)
    )
    # Generate features using the specified base distribution type
    features_only = generator.generate_features_only(
        n_samples=source_args.get('base_n_samples', 20000),
        dist_type1=source_args.get('dist_type1', 'mixed')
    )
    # Return a special base dataset containing only features
    return SyntheticFeaturesBaseDataset(features_only) # Uses the class from data_sets


# --- Dispatch Dictionary (UPDATED) ---
DATA_LOADERS: Dict[str, callable] = {
    # Base loaders (return BaseDataset wrapper or tuple[Dataset, Dataset])
    'torchvision': load_torchvision_dataset,
    'synthetic_base': load_synthetic_base_data,        # For Synthetic_Label
    'credit_base': load_credit_base_data,
    'synthetic_features_only': load_synthetic_features_only, # For Synthetic_Concept

    # Per-client / Site loaders (return Tuple[np.ndarray, np.ndarray])
    'synthetic_feature_skew_client': load_generate_synthetic_feature_skew_client, # For Synthetic_Feature
    'heart_site_loader': load_heart_site_data,
    'pre_split_paths_ixi': load_ixi_client_paths,
    'pre_split_paths_isic': load_isic_client_paths,
}