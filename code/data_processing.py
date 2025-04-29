# data_processing.py
"""
Contains DataManager for orchestrating data loading/partitioning/processing,
and DataPreprocessor for client-level splitting and Dataset instantiation.
Streamlined version with simplified code flow and reduced conditional complexity.
"""
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from typing import Dict, Tuple, Any, Optional, List, Union, Callable
from sklearn.model_selection import train_test_split
import random
import copy
# Project Imports
from configs import N_WORKERS
from helper import get_parameters_for_dataset
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
        X_train, y_train = X[idx_train], y[idx_train]
        X_val, y_val = X[idx_val], y[idx_val]
        X_test, y_test = X[idx_test], y[idx_test]
    elif isinstance(X, list):
        X_train, X_val, X_test = [X[i] for i in idx_train], [X[i] for i in idx_val], [X[i] for i in idx_test]
        if isinstance(y, np.ndarray): 
            y_train, y_val, y_test = y[idx_train], y[idx_val], y[idx_test]
        else: 
            y_train, y_val, y_test = [y[i] for i in idx_train], [y[i] for i in idx_val], [y[i] for i in idx_test]
    else: 
        raise TypeError(f"Unsupported data type for splitting: {type(X)}")

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def _split_indices(indices: List[int], test_size: float = 0.2, val_size: float = 0.2, seed: int = 42
                  ) -> Tuple[List[int], List[int], List[int]]:
    """Splits a list of indices into train, validation, and test index lists."""
    num_samples = len(indices)
    np_indices = np.array(indices)
    
    if num_samples < 5: 
        return indices, [], [] 
        
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
            # Dataset class mapping
            dataset_classes = {
                'SyntheticDataset': SyntheticDataset,
                'CreditDataset': CreditDataset,
                'HeartDataset': HeartDataset,
                'EMNISTDataset': EMNISTDataset,
                'CIFARDataset': CIFARDataset,
                'ISICDataset': ISICDataset,
                'IXITinyDataset': IXITinyDataset
            }
            
            # Get the right dataset class constructor
            dataset_class = dataset_classes.get(self.dataset_class_name)
            if dataset_class is None:
                raise ValueError(f"Dataset class '{self.dataset_class_name}' not supported")
                
            # Create dataset with appropriate args
            if self.dataset_class_name in ['SyntheticDataset', 'CreditDataset']:
                return dataset_class(**data_args)
            else:
                return dataset_class(**data_args, **common_args)
                
        except Exception as e:
             print(f"Error instantiating Dataset '{self.dataset_class_name}' for split '{split_type}': {e}")
             return None

    def preprocess_client_data(self, client_data_bundle: dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Processes raw data bundle for one client into DataLoaders."""
        # Extract data type and content from bundle
        data_type = client_data_bundle.get('type')
        raw_data = client_data_bundle.get('data')
        
        # Extract and split data based on type
        if data_type == 'subset':
            indices, base_data = raw_data.get('indices', []), raw_data.get('base_data')
            train_indices, val_indices, test_indices = _split_indices(indices)
            
            # Handle torch dataset subset
            if isinstance(base_data, torch.utils.data.Dataset):
                train_dataset = Subset(base_data, train_indices) if train_indices else None
                val_dataset = Subset(base_data, val_indices) if val_indices else None
                test_dataset = Subset(base_data, test_indices) if test_indices else None
            # Handle tuple of torch datasets (e.g., torchvision train/test datasets)
            elif isinstance(base_data, tuple) and all(isinstance(d, torch.utils.data.Dataset) for d in base_data):
                # Use the first dataset (typically train) for all splits
                train_dataset = Subset(base_data[0], train_indices) if train_indices else None
                val_dataset = Subset(base_data[0], val_indices) if val_indices else None
                test_dataset = Subset(base_data[0], test_indices) if test_indices else None
            # Handle numpy array subset
            elif isinstance(base_data, tuple) and len(base_data) == 2 and isinstance(base_data[0], np.ndarray):
                base_X, base_y = base_data
                train_data = (base_X[train_indices], base_y[train_indices]) if train_indices else None
                val_data = (base_X[val_indices], base_y[val_indices]) if val_indices else None
                test_data = (base_X[test_indices], base_y[test_indices]) if test_indices else None
                
                # Create actual datasets
                train_dataset = self._get_dataset_instance({'X_np': train_data[0], 'y_np': train_data[1]}, 'train') if train_data else None
                val_dataset = self._get_dataset_instance({'X_np': val_data[0], 'y_np': val_data[1]}, 'val') if val_data else None
                test_dataset = self._get_dataset_instance({'X_np': test_data[0], 'y_np': test_data[1]}, 'test') if test_data else None
            else:
                return DataLoader([]), DataLoader([]), DataLoader([])
                
        elif data_type == 'direct':
            X, y = raw_data.get('X'), raw_data.get('y')
            (X_train, y_train), (X_val, y_val), (X_test, y_test) = _split_data(X, y)
            
            # Create datasets
            train_dataset = self._get_dataset_instance({'X_np': X_train, 'y_np': y_train}, 'train') if len(X_train) > 0 else None
            val_dataset = self._get_dataset_instance({'X_np': X_val, 'y_np': y_val}, 'val') if len(X_val) > 0 else None
            test_dataset = self._get_dataset_instance({'X_np': X_test, 'y_np': y_test}, 'test') if len(X_test) > 0 else None
            
        elif data_type == 'paths':
            X_paths, y_data = raw_data.get('X_paths'), raw_data.get('y_data')
            (X_train, y_train), (X_val, y_val), (X_test, y_test) = _split_data(X_paths, y_data)
            
            # Determine if we're dealing with paths for both X and y
            is_y_paths = isinstance(y_data, list)
            y_key = 'label_paths' if is_y_paths else 'labels_np'
            
            # Create datasets
            train_args = {'image_paths': X_train, y_key: y_train}
            val_args = {'image_paths': X_val, y_key: y_val}
            test_args = {'image_paths': X_test, y_key: y_test}
            
            train_dataset = self._get_dataset_instance(train_args, 'train') if len(X_train) > 0 else None
            val_dataset = self._get_dataset_instance(val_args, 'val') if len(X_val) > 0 else None
            test_dataset = self._get_dataset_instance(test_args, 'test') if len(X_test) > 0 else None
        else:
            return DataLoader([]), DataLoader([]), DataLoader([])

        # Create DataLoaders with appropriate settings
        g_train = torch.Generator()
        g_train.manual_seed(int(torch.initial_seed()))
        
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, 
            num_workers=N_WORKERS, generator=g_train, drop_last=False, 
            persistent_workers=(N_WORKERS>0)) if train_dataset else DataLoader([])
            
        val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False, 
            num_workers=N_WORKERS, persistent_workers=(N_WORKERS>0)) if val_dataset else DataLoader([])
            
        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False, 
            num_workers=N_WORKERS, persistent_workers=(N_WORKERS>0)) if test_dataset else DataLoader([])

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
        self.py_random_sampler = random.Random(base_seed + 100)

    def _create_client_bundle(self, loaded_data):
        """Helper to convert loaded data to a standardized client bundle format."""
        # Create appropriate bundle based on data type
        if isinstance(loaded_data[0], list):
            return {'type': 'paths', 'data': {'X_paths': loaded_data[0], 'y_data': loaded_data[1]}}
        elif isinstance(loaded_data[0], np.ndarray):
            return {'type': 'direct', 'data': {'X': loaded_data[0], 'y': loaded_data[1]}}
        elif isinstance(loaded_data[0], torch.utils.data.Dataset):
            print(loaded_data)
        return None

    def _prepare_loader_args(self, cost, client_num=None, num_clients=None):
        """Prepare common loader arguments with dataset-specific adjustments."""
        args = {'source_args': self.config.get('source_args', {}), 'cost_key': cost}
        # Add dataset-specific arguments
        if self.data_source_type == 'synthetic':
            args['dataset_name'] = self.dataset_name
            args['base_seed'] = self.base_seed
            if num_clients is not None:
                args['num_clients'] = num_clients
        elif self.data_source_type == 'torchvision':
            args['data_dir'] = self.data_dir_root
            args['transform_config'] = self.config.get('transform_config', {})
        else:
            args['data_dir'] = os.path.join(self.data_dir_root, self.dataset_name)
            
        # Add client number if specified
        if client_num is not None:
            args['client_num'] = client_num
        return args

    def _extract_partitioner_input(self, data):
        """Extract data needed for the partitioner based on the data format."""
        # For torchvision datasets
        data_type = type(data[0]).__module__
        if data_type.startswith('torchvision.datasets'):
        # Handle both tensor and list targets
            tr_data = data[0]
            if isinstance(tr_data.targets, torch.Tensor):
                partitioner_input = tr_data.targets.numpy()
            else:
                partitioner_input = np.array(tr_data.targets) 
            return partitioner_input, len(tr_data)
        # For tuple data (features, labels)
        elif isinstance(data, tuple) and len(data) == 2:
            return data[1], len(data[0])  # labels, num_samples
            
        raise ValueError(f"Cannot extract partitioner input from data type: {type(data)}")
    
    def _sample_data_for_client(self,
                                data_or_indices: Union[Tuple[np.ndarray, np.ndarray], Tuple[List[str], Union[np.ndarray, List[str]]], List[int]],
                                target_size: Optional[int]
                               ) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[List[str], Union[np.ndarray, List[str]]], List[int]]:
        if target_size is None or target_size <= 0:
            return data_or_indices # No sampling requested

        current_size = 0
        is_indices = False
        is_numpy = False
        is_paths = False
        # Determine type and current size
        if isinstance(data_or_indices, list) and all(isinstance(i, int) for i in data_or_indices):
            is_indices = True
            current_size = len(data_or_indices)
        elif isinstance(data_or_indices, tuple) and len(data_or_indices) == 2:
            if isinstance(data_or_indices[0], np.ndarray):
                is_numpy = True
                current_size = len(data_or_indices[0])
            elif isinstance(data_or_indices[0], list):
                is_paths = True
                current_size = len(data_or_indices[0])

        # Perform sampling if needed
        if current_size > target_size:
            # Use the instance's sampler for reproducibility across calls within a run
            sampled_indices = self.py_random_sampler.sample(range(current_size), target_size)
            sampled_indices.sort() # Optional: Keep original order for numpy/paths

            if is_indices:
                original_indices = np.array(data_or_indices) # Convert to numpy for easy indexing
                return original_indices[sampled_indices].tolist()
            elif is_numpy:
                X, y = data_or_indices
                return X[sampled_indices], y[sampled_indices]
            elif is_paths:
                X_paths, y_data = data_or_indices
                X_sampled = [X_paths[i] for i in sampled_indices]
                y_sampled = [y_data[i] for i in sampled_indices] if isinstance(y_data, list) else y_data[sampled_indices]
                return X_sampled, y_sampled
            else:
                 print("Warning: Could not determine data type for sampling.")
                 return data_or_indices # Return original if type unknown
        else:
            # Keep all data if already at or below target size
            return data_or_indices

    def get_dataloaders(self, cost: Any, run_seed: int, num_clients_override: Optional[int] = None
                        ) -> Dict[str, Tuple[DataLoader, DataLoader, DataLoader]]:
        """Gets DataLoaders for all clients for a specific run configuration."""
        num_clients = num_clients_override or self.config.get('default_num_clients', 2)
        loader_func = get_loader(self.data_source_type)
        partitioner_func = get_partitioner(self.partitioning_strategy)
        client_final_data_bundles = {} # Store bundles AFTER potential sampling
        target_samples_per_client = self.config.get('samples_per_client')

        # --- Seed the instance sampler for this specific run/cost ---
        # Ensures sampling is deterministic per run, even if DataManager is reused
        self.py_random_sampler.seed(run_seed + 101 + hash(str(cost)))

        base_data_for_partitioning, all_labels = None, None
        if self.partitioning_strategy == 'pre_split':
            print(f"Loading pre-split data for {num_clients} clients...")
            for client_idx in range(1, num_clients + 1):
                client_id = f"client_{client_idx}"
                try:
                    # 1. Load raw data for the client
                    loader_args = self._prepare_loader_args(cost, client_idx, num_clients)
                    # For Synthetic_Feature/Concept, this now loads base_n_samples with client-specific shift/seed
                    loaded_data = loader_func(**loader_args) # e.g., (X_np, y_np) or (paths, labels)
                    # 2. Sample the loaded data down to target size (if applicable)
                    sampled_data = self._sample_data_for_client(loaded_data, target_samples_per_client)

                    # 3. Create the final bundle from sampled data
                    bundle = self._create_client_bundle(sampled_data)
                    if bundle and sampled_data[0] is not None and len(sampled_data[0]) > 0: # Check if sampling resulted in data
                        client_final_data_bundles[client_id] = bundle
                    else:
                        print(f"Warning: Client {client_id} has no data after loading/sampling.")

                except Exception as e:
                    print(f"Error loading/sampling pre-split data for client {client_id}: {e}")
                    # Optionally continue or raise

        else: # Handle datasets requiring partitioning (Dirichlet, IID)
            print(f"Loading and partitioning base data ({self.partitioning_strategy})...")
            try:
                # 1. Load base data
                loader_args = self._prepare_loader_args(cost)
                base_data_for_partitioning = loader_func(**loader_args)
                
                # Extract labels correctly ONCE for both partitioning and display
                partitioner_input, num_samples = self._extract_partitioner_input(base_data_for_partitioning)
                all_labels = copy.deepcopy(partitioner_input)
                # 2. Run partitioner with the extracted labels
                partitioner_kwargs = {**self.config.get('partitioner_args', {})}
                if self.partitioning_strategy == 'dirichlet_indices':
                    partitioner_kwargs['alpha'] = float(cost)
                client_indices_full = partitioner_func(partitioner_input, num_clients, seed=run_seed, **partitioner_kwargs)
        
                # 3. Sample the *indices* for each client down to target size
                client_indices_final = {}
                for client_id_idx, indices_list in client_indices_full.items():
                    sampled_indices = self._sample_data_for_client(indices_list, target_samples_per_client)
                    if sampled_indices: # Only add if sampling resulted in indices
                        client_indices_final[client_id_idx] = sampled_indices
                # 4. Create client bundles using the FINAL sampled indices
                for client_id_idx, final_indices in client_indices_final.items():
                    client_id = f"client_{client_id_idx+1}"
                    client_final_data_bundles[client_id] = {
                        'type': 'subset',
                        'data': {'indices': final_indices, 'base_data': base_data_for_partitioning}
                    }

            except Exception as e:
                print(f"Error partitioning or sampling data: {e}")
                raise

        print_class_dist(client_final_data_bundles, all_labels)
        # --- Process Final Bundles into DataLoaders ---
        preprocessor = DataPreprocessor(self.config, self.batch_size)
        client_dataloaders = {}
        for client_id, bundle in client_final_data_bundles.items():
            try:
                # preprocess_client_data performs the train/val/test split on the (already sampled) data
                dataloaders = preprocessor.preprocess_client_data(bundle)
                if dataloaders[0] and hasattr(dataloaders[0], 'dataset') and len(dataloaders[0].dataset) > 0:
                    client_dataloaders[client_id] = dataloaders
                else:
                    print(f"Warning: Client {client_id} has no training samples after preprocessing, skipping.")
            except Exception as e:
                print(f"Error preprocessing data for client {client_id}: {e}")

        return client_dataloaders


def print_class_dist(client_final_data_bundles, all_labels=None):
    print("\n--- Client class distribution (after sampling) ---")
    for client_id in sorted(client_final_data_bundles):
        bundle = client_final_data_bundles[client_id]

        # Extract label array y_client for this client
        if bundle.get("type") == "subset":                    
            indices = bundle["data"]["indices"]
            # Use the pre-extracted labels instead of trying to unpack base_data
            if all_labels is not None:
                y_client = all_labels[indices]  
            else:
                # Fallback for pre-split case
                try:
                    _, base_y = bundle["data"]["base_data"]
                    y_client = base_y[indices]
                except Exception as e:
                    print(f"  {client_id}: Unable to extract labels: {e}")
                    continue
        else:                                                 
            # pre-split path remains unchanged
            payload = bundle["data"]                          
            if isinstance(payload, dict):                   
                y_client = payload.get("y", [])
            else:                                            
                _, y_client = payload

        # Build pretty distribution string and print
        uniq, cnts = np.unique(y_client, return_counts=True)
        dist = ", ".join(f"Class {u}: {c}" for u, c in zip(uniq, cnts))
        print(f"  {client_id}: {len(y_client)} samples ({dist})")

    print("-" * 50)