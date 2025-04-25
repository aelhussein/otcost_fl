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

    def get_dataloaders(self, cost: Any, run_seed: int, num_clients_override: Optional[int] = None
                    ) -> Dict[str, Tuple[DataLoader, DataLoader, DataLoader]]:
        """Gets DataLoaders for all clients for a specific run configuration."""
        num_clients = num_clients_override or self.config.get('default_num_clients', 2)
        loader_func = get_loader(self.data_source_type)
        partitioner_func = get_partitioner(self.partitioning_strategy)
        client_raw_bundles = {}
        
        # Handle pre-split datasets (separate data per client)
        if self.partitioning_strategy == 'pre_split':
            for client_idx in range(1, num_clients + 1):
                client_id = f"client_{client_idx}"
                
                # Load data for this client
                try:
                    loader_args = self._prepare_loader_args(cost, client_idx, num_clients)
                    loaded_data = loader_func(**loader_args)
                    bundle = self._create_client_bundle(loaded_data)
                    if bundle:
                        client_raw_bundles[client_id] = bundle
                except Exception as e:
                    print(f"Error loading pre-split data for client {client_idx}: {e}")
                    
        # Handle datasets requiring partitioning (Dirichlet, IID)
        else:
            # Load the base dataset
            try:
                loader_args = self._prepare_loader_args(cost)
                base_data = loader_func(**loader_args)
                
                # Extract data for partitioning
                partitioner_input, num_samples = self._extract_partitioner_input(base_data)
                
                # Configure partitioner
                partitioner_kwargs = {**self.config.get('partitioner_args', {})}
                if self.partitioning_strategy == 'dirichlet_indices':
                    partitioner_kwargs['alpha'] = float(cost)
                    if 'sampling_config' in self.config:
                        partitioner_kwargs['sampling_config'] = self.config['sampling_config']
                
                # Partition the data
                client_indices = partitioner_func(partitioner_input, num_clients, seed=run_seed, **partitioner_kwargs)
                
                # Create client bundles
                for i, indices in client_indices.items():
                    client_id = f"client_{i+1}"
                    client_raw_bundles[client_id] = {
                        'type': 'subset', 
                        'data': {'indices': indices, 'base_data': base_data}
                    }
            except Exception as e:
                print(f"Error partitioning data: {e}")
        
        # Process client data into DataLoaders
        preprocessor = DataPreprocessor(self.config, self.batch_size)
        client_dataloaders = {}
        # print('\n')
        for client_id, bundle in client_raw_bundles.items():
            dataloaders = preprocessor.preprocess_client_data(bundle)
            client_dataloaders[client_id] = dataloaders
        #     print(f"Client {client_id}: "
        #           f"Train size: {len(dataloaders[0].dataset)}, "
        #           f"Val size: {len(dataloaders[1].dataset)}, "
        #           f"Test size: {len(dataloaders[2].dataset)}")
        # print('\n')      
        return client_dataloaders