# data_processing.py
"""
Contains DataManager for orchestrating data loading/partitioning/processing,
and DataPreprocessor for client-level splitting and Dataset instantiation.
Streamlined version, ensuring correct argument passing.
"""
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from typing import Dict, Tuple, Any, Optional, List, Union, Callable
from sklearn.model_selection import train_test_split
# Project Imports
from configs import N_WORKERS # Use N_WORKERS from config
from helper import get_parameters_for_dataset # Removed: translate_cost
from data_loading import get_loader
from data_partitioning import get_partitioner
# Import all final Dataset classes
from data_sets import (SyntheticDataset, CreditDataset, EMNISTDataset,
                         CIFARDataset, ISICDataset, IXITinyDataset, HeartDataset)

# =============================================================================
# == Data Splitting Functions (Internal Helpers) ==
# =============================================================================

def _split_data(X: Union[np.ndarray, List], y: Union[np.ndarray, List],
                test_size: float = 0.2, val_size: float = 0.2, seed: int = 42
               ) -> Tuple[Tuple[Any, Any], Tuple[Any, Any], Tuple[Any, Any]]:
    """Splits data arrays/lists into train, val, test using simple random split."""
    # (Implementation remains the same as previous version)
    num_samples = len(X)
    indices = np.arange(num_samples)
    if num_samples < 5: # Handle small datasets
        empty_X = np.array([], dtype=X.dtype).reshape(0, *X.shape[1:]) if isinstance(X, np.ndarray) and X.ndim > 1 else np.array([], dtype=getattr(X, 'dtype', None))
        empty_y = np.array([], dtype=y.dtype) if isinstance(y, np.ndarray) else []
        if isinstance(X, list): empty_X = []
        return (X, y), (empty_X, empty_y), (empty_X, empty_y)

    idx_train_val, idx_test = train_test_split(indices, test_size=test_size, random_state=seed, shuffle=True)
    relative_val_size = val_size / (1.0 - test_size) if (1.0 - test_size) > 1e-6 else 0.0
    if len(idx_train_val) < 2 or relative_val_size <= 0 or relative_val_size >= 1.0:
        idx_train, idx_val = idx_train_val, np.array([], dtype=int)
    else:
        idx_train, idx_val = train_test_split(idx_train_val, test_size=relative_val_size, random_state=seed + 1, shuffle=True)

    if isinstance(X, np.ndarray):
        X_train, y_train = X[idx_train], y[idx_train]; X_val, y_val = X[idx_val], y[idx_val]; X_test, y_test = X[idx_test], y[idx_test]
    elif isinstance(X, list):
        X_train, X_val, X_test = [X[i] for i in idx_train], [X[i] for i in idx_val], [X[i] for i in idx_test]
        if isinstance(y, np.ndarray): y_train, y_val, y_test = y[idx_train], y[idx_val], y[idx_test]
        else: y_train, y_val, y_test = [y[i] for i in idx_train], [y[i] for i in idx_val], [y[i] for i in idx_test]
    else: raise TypeError(f"Unsupported data type for splitting: {type(X)}")

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def _split_indices(indices: List[int], test_size: float = 0.2, val_size: float = 0.2, seed: int = 42
                  ) -> Tuple[List[int], List[int], List[int]]:
    """Splits a list of indices into train, validation, and test index lists."""
    # (Implementation remains the same as previous version)
    num_samples = len(indices); np_indices = np.array(indices)
    if num_samples < 5: return indices, [], []
    idx_train_val, idx_test = train_test_split(np_indices, test_size=test_size, random_state=seed, shuffle=True)
    relative_val_size = val_size / (1.0 - test_size) if (1.0 - test_size) > 1e-6 else 0.0
    if len(idx_train_val) < 2 or relative_val_size <= 0 or relative_val_size >= 1.0:
        idx_train, idx_val = idx_train_val, np.array([], dtype=int)
    else:
        idx_train, idx_val = train_test_split(idx_train_val, test_size=relative_val_size, random_state=seed + 1, shuffle=True)
    return idx_train.tolist(), idx_val.tolist(), idx_test.tolist()

# =============================================================================
# == Data Preprocessor Class (Client-Level Processing) ==
# =============================================================================
class DataPreprocessor:
    """Handles client-level train/val/test splitting and Dataset instantiation."""
    def __init__(self, dataset_config: dict, batch_size: int):
        self.dataset_config = dataset_config
        self.batch_size = batch_size
        self.dataset_name = dataset_config.get('dataset_name', 'UnknownDataset')
        self.dataset_class_name = self.dataset_config.get('dataset_class')
        if not self.dataset_class_name:
             raise ValueError(f"Config for '{self.dataset_name}' missing 'dataset_class'.")

    def _get_dataset_instance(self, data_args: dict, split_type: str):
        """Instantiates the correct Dataset class based on config name."""
        common_args = {'split_type': split_type, 'dataset_config': self.dataset_config}
        try:
            # Explicit mapping for clarity and dependency tracking
            if self.dataset_class_name == 'SyntheticDataset': return SyntheticDataset(**data_args)
            elif self.dataset_class_name == 'CreditDataset': return CreditDataset(**data_args)
            elif self.dataset_class_name == 'HeartDataset': return HeartDataset(**data_args, **common_args)
            elif self.dataset_class_name == 'EMNISTDataset': return EMNISTDataset(**data_args, **common_args)
            elif self.dataset_class_name == 'CIFARDataset': return CIFARDataset(**data_args, **common_args)
            elif self.dataset_class_name == 'ISICDataset': return ISICDataset(**data_args, **common_args)
            elif self.dataset_class_name == 'IXITinyDataset': return IXITinyDataset(**data_args, **common_args)
            else: raise ValueError(f"Dataset class '{self.dataset_class_name}' not handled in _get_dataset_instance.")
        except Exception as e:
             print(f"Error instantiating Dataset '{self.dataset_class_name}' for split '{split_type}': {e}")
             return None

    def preprocess_client_data(self, client_data_bundle: dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Processes raw data bundle for one client into DataLoaders."""
        data_type = client_data_bundle.get('type')
        raw_data = client_data_bundle.get('data')
        train_ds_input, val_ds_input, test_ds_input = None, None, None
        ds_uses_torch_subset = False

        if data_type == 'subset':
            indices, base_data = raw_data.get('indices', []), raw_data.get('base_data')
            train_indices, val_indices, test_indices = _split_indices(indices)
            if isinstance(base_data, torch.utils.data.Dataset): # Torchvision dataset
                 ds_uses_torch_subset = True
                 train_ds_input = Subset(base_data, train_indices) if train_indices else None
                 val_ds_input = Subset(base_data, val_indices) if val_indices else None
                 test_ds_input = Subset(base_data, test_indices) if test_indices else None
            elif isinstance(base_data, tuple) and len(base_data) == 2: # Assume (X_np, y_np) base
                 base_X, base_y = base_data
                 train_ds_input = (base_X[train_indices], base_y[train_indices]) if train_indices else None
                 val_ds_input = (base_X[val_indices], base_y[val_indices]) if val_indices else None
                 test_ds_input = (base_X[test_indices], base_y[test_indices]) if test_indices else None

        elif data_type == 'direct':
            X, y = raw_data.get('X'), raw_data.get('y')
            train_ds_input, val_ds_input, test_ds_input = _split_data(X, y)

        elif data_type == 'paths':
            X_paths, y_data = raw_data.get('X_paths'), raw_data.get('y_data')
            train_ds_input, val_ds_input, test_ds_input = _split_data(X_paths, y_data)

        # --- Instantiate Datasets ---
        train_dataset, val_dataset, test_dataset = None, None, None
        if ds_uses_torch_subset:
             train_dataset, val_dataset, test_dataset = train_ds_input, val_ds_input, test_ds_input
        else:
             # Helper to create args dict based on type (paths vs np arrays)
             def _make_ds_args(data_tuple):
                 if data_type == 'paths':
                     # y_data can be list (IXI) or ndarray (ISIC)
                     is_y_paths = isinstance(data_tuple[1], list)
                     return {'image_paths': data_tuple[0],
                             'label_paths' if is_y_paths else 'labels_np': data_tuple[1]}
                 else: # direct or subset from np tuple
                     return {'X_np': data_tuple[0], 'y_np': data_tuple[1]}

             if train_ds_input and len(train_ds_input[0]) > 0: train_dataset = self._get_dataset_instance(_make_ds_args(train_ds_input), 'train')
             if val_ds_input and len(val_ds_input[0]) > 0: val_dataset = self._get_dataset_instance(_make_ds_args(val_ds_input), 'val')
             if test_ds_input and len(test_ds_input[0]) > 0: test_dataset = self._get_dataset_instance(_make_ds_args(test_ds_input), 'test')

        # --- Create DataLoaders ---
        g_train = torch.Generator(); g_train.manual_seed(int(torch.initial_seed()))
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=N_WORKERS, generator=g_train, drop_last=False, persistent_workers=(N_WORKERS>0)) if train_dataset else DataLoader([])
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=N_WORKERS, persistent_workers=(N_WORKERS>0)) if val_dataset else DataLoader([])
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=N_WORKERS, persistent_workers=(N_WORKERS>0)) if test_dataset else DataLoader([])

        return train_loader, val_loader, test_loader

# =============================================================================
# == Data Manager Class (Orchestrator) ==
# =============================================================================
class DataManager:
    """Orchestrates dataset loading, partitioning, and preprocessing."""
    def __init__(self, dataset_name: str, base_seed: int, data_dir_root: str):
        self.dataset_name = dataset_name
        self.base_seed = base_seed
        self.data_dir_root = data_dir_root
        self.config = get_parameters_for_dataset(dataset_name)
        self.data_source_type = self.config['data_source']
        self.partitioning_strategy = self.config['partitioning_strategy']
        self.batch_size = self.config['batch_size']


 # Updated section in data_processing.py for handling different dataset types

    def get_dataloaders(self, cost: Any, run_seed: int, num_clients_override: Optional[int] = None
                    ) -> Dict[str, Tuple[DataLoader, DataLoader, DataLoader]]:
        """Gets DataLoaders for all clients for a specific run configuration."""
        num_clients = num_clients_override or self.config.get('default_num_clients', 2)
        loader_func = get_loader(self.data_source_type)
        partitioner_func = get_partitioner(self.partitioning_strategy)
        client_raw_bundles: Dict[str, Dict] = {} # {client_id: {'type': ..., 'data': ...}}

        if self.partitioning_strategy == 'pre_split':
            # Load data individually for each client (unchanged)
            for i in range(num_clients):
                client_idx = i + 1; client_id = f"client_{client_idx}"
                # Build arguments dict specific to the loader type
                loader_args = {'source_args': self.config.get('source_args', {}),
                            'client_num': client_idx, 'cost_key': cost}
                if self.data_source_type == 'synthetic':
                    loader_args['dataset_name'] = self.dataset_name
                    loader_args['base_seed'] = self.base_seed
                else:
                    loader_args['data_dir'] = os.path.join(self.data_dir_root, self.dataset_name)
                    if self.data_source_type == 'heart_site_loader':
                        loader_args['source_args'] = self.config.get('source_args', {})

                try: loaded_data = loader_func(**loader_args)
                except Exception as e: print(f"Error loading pre-split data client {client_idx}: {e}"); continue

                # Determine bundle type based on loaded data
                if isinstance(loaded_data, tuple) and len(loaded_data) == 2:
                    if isinstance(loaded_data[0], list): 
                        bundle_type, bundle_data = 'paths', {'X_paths': loaded_data[0], 'y_data': loaded_data[1]}
                    elif isinstance(loaded_data[0], np.ndarray): 
                        bundle_type, bundle_data = 'direct', {'X': loaded_data[0], 'y': loaded_data[1]}
                    else: continue # Skip if format unexpected
                    client_raw_bundles[client_id] = {'type': bundle_type, 'data': bundle_data}
                else: continue

        else: # Partitioning needed (Dirichlet, IID)
            # Load base data
            loader_args = {'source_args': self.config.get('source_args', {})}
            
            # Add loader-specific args
            if self.data_source_type == 'torchvision':
                loader_args['data_dir'] = self.data_dir_root
                loader_args['transform_config'] = self.config.get('transform_config', {})
            elif self.data_source_type == 'synthetic':
                loader_args['dataset_name'] = self.dataset_name
                loader_args['base_seed'] = self.base_seed
            elif self.data_source_type == 'credit_csv':
                pass # credit_raw only needs source_args

            # Load the data
            load_result = loader_func(**loader_args)
        
            # HANDLE TORCHVISION DATASETS SPECIFICALLY
            if self.data_source_type == 'torchvision':
                # For torchvision, load_result is (train_dataset, test_dataset)
                train_dataset, _ = load_result
            
                base_data = train_dataset
                num_samples = len(base_data)
                
                # Extract targets and convert to 1D NumPy array
                if isinstance(base_data.targets, torch.Tensor):
                    partitioner_input = base_data.targets.cpu().numpy()
                else:
                    partitioner_input = np.array(base_data.targets)
                    
                # Ensure we have a 1D array of labels
                if partitioner_input.ndim > 1:
                    partitioner_input = partitioner_input[:, 0] if partitioner_input.shape[1] > 0 else partitioner_input.ravel()
            
            # HANDLE OTHER DATASET TYPES (NumPy arrays, etc.)
            else:
                # For numeric data returns (X_np, y_np) or other formats
                if isinstance(load_result, tuple) and len(load_result) == 2:
                    base_data = load_result
                    partitioner_input = base_data[1]
                    num_samples = len(base_data[0])
                   
            # Set up partitioner arguments and call partitioner
            partitioner_kwargs = {**self.config.get('partitioner_args', {})}
            if self.partitioning_strategy == 'dirichlet_indices':
                if not isinstance(cost, (int, float)):
                    raise TypeError(f"Dirichlet requires numeric cost (alpha), got {type(cost)}")
                partitioner_kwargs['alpha'] = float(cost)
                if 'sampling_config' in self.config:
                    partitioner_kwargs['sampling_config'] = self.config['sampling_config']

            # Call partitioner function
            input_arg = partitioner_input if 'dirichlet' in self.partitioning_strategy else num_samples
            client_indices = partitioner_func(input_arg, num_clients, seed=run_seed, **partitioner_kwargs)
            
            # Create client bundles
            for i, indices in client_indices.items():
                client_id = f"client_{i+1}"
                client_raw_bundles[client_id] = {'type': 'subset', 'data': {'indices': indices, 'base_data': base_data}}
        # Process client data into DataLoaders
        preprocessor = DataPreprocessor(self.config, self.batch_size)
        client_dataloaders = {}
        for client_id, bundle in client_raw_bundles.items():
            dataloaders = preprocessor.preprocess_client_data(bundle)
            client_dataloaders[client_id] = dataloaders
            print(f"Client {client_id}: Train size: {len(dataloaders[0].dataset)}, Val size: {len(dataloaders[1].dataset)}, Test size: {len(dataloaders[2].dataset)}")
        return client_dataloaders