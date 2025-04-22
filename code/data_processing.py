"""
Handles client data preprocessing: train/val/test splitting, scaling (for some datasets),
and DataLoader creation.
"""
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Import necessary components from other modules
from configs import N_WORKERS
from helper import g, seed_worker # Random generator and worker seeding
from datasets import BaseTabularDataset # Import base class used for scaling check

# Import specific dataset classes needed by DataPreprocessor._get_final_dataset_class
# This assumes datasets.py defines all final wrapper classes
from datasets import (
    SyntheticDataset, CreditDataset, HeartDataset, EMNISTDataset,
    CIFARDataset, ISICDataset, IXITinyDataset
)

class DataPreprocessor:
    """
    Takes client input data (Subsets, xy_dicts, or path_dicts) and creates
    train, validation, and test DataLoaders for each client. Handles
    train/val/test splitting and potentially coordinates scaling.
    """
    def __init__(self, dataset_name, dataset_config): # Pass full config
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.batch_size = dataset_config['batch_size']
        self.final_dataset_class = self._get_final_dataset_class()
        # Determine if scaling is needed based on config and *not* handled in loader
        self.needs_scaling_here = ('standard_scale' in self.dataset_config.get('needs_preprocessing', [])
                                    and self.dataset_name != 'Heart') # Heart scaling done in its loader

        self._processor_map = {
            'subset': self._process_subset,        # Input: Subset pointing to base data
            'xy_dict': self._process_xy_dict,      # Input: {'X': array, 'y': array}
            'path_dict': self._process_path_dict,  # Input: {'X': paths, 'y': data}
        }

    def _get_final_dataset_class(self):
         """Gets the final Dataset class (e.g., SyntheticDataset) from config."""
         class_name = self.dataset_config.get('dataset_class')
         if not class_name: raise ValueError(f"'dataset_class' not in config for {self.dataset_name}")
         # Check if class exists in the datasets module (imported above)
         if hasattr(sys.modules['datasets'], class_name):
             return getattr(sys.modules['datasets'], class_name)
         else:
             # Or check current module if Base classes were kept here (shouldn't be needed)
             # if hasattr(sys.modules[__name__], class_name):
             #      print(f"Warning: Found dataset class {class_name} in data_processing.py, expected in datasets.py")
             #      return getattr(sys.modules[__name__], class_name)
             raise ImportError(f"Dataset class '{class_name}' not found in datasets.py")


    def process_client_data(self, client_input_data: Dict, input_type: str):
        """Process data for multiple clients based on input type."""
        processed_data = {} # {client_id: (train_loader, val_loader, test_loader)}
        processor_func = self._processor_map.get(input_type)
        if processor_func is None: raise ValueError(f"Unknown input_type for DataPreprocessor: {input_type}")

        print(f"Preprocessing using method for input type: '{input_type}'")
        all_scalers = {} # Store scalers fitted on train sets if needed

        # --- Pass 1 (for Subset type needing scaling here): Fit scalers ---
        if input_type == 'subset' and self.needs_scaling_here:
            print("Preprocessing Pass 1 (Subset Scaling): Fitting scalers...")
            for client_id, client_subset in client_input_data.items():
                if not client_subset or not client_subset.indices: continue
                train_indices, _, _ = self._split_indices(client_subset.indices, seed=42)
                if not train_indices: continue

                # Instantiate final dataset temporarily to fit scaler
                temp_train_dataset = self.final_dataset_class(
                    base_dataset=client_subset.dataset,
                    indices=train_indices,
                    split_type='train',
                    dataset_config=self.dataset_config
                )
                # Check if the dataset class has the fit_scaler method
                if hasattr(temp_train_dataset, 'fit_scaler'):
                    scaler = temp_train_dataset.fit_scaler()
                    if scaler: all_scalers[client_id] = scaler
                else:
                     print(f"Warning: Dataset class {self.final_dataset_class.__name__} does not have fit_scaler method, but scaling requested.")
                del temp_train_dataset

        # --- Pass 2: Create final datasets and dataloaders ---
        print("Preprocessing Pass 2: Creating Datasets and DataLoaders...")
        for client_id, data in client_input_data.items():
            scaler = all_scalers.get(client_id) # Get scaler if fitted previously
            processed_data[client_id] = processor_func(data, scaler=scaler) # Pass scaler

        return processed_data

    def _process_subset(self, client_subset: Subset, scaler=None):
        """Process Subset input (e.g., for Synthetic, Credit after partitioning)."""
        if not client_subset or not client_subset.indices: print(f"Warn: Client subset empty."); empty_loader = DataLoader([]); return (empty_loader,) * 3
        train_indices, val_indices, test_indices = self._split_indices(client_subset.indices, seed=42)

        # Instantiate FINAL dataset for each split, passing base, indices, type, config, and scaler
        train_dataset = self.final_dataset_class(client_subset.dataset, train_indices, 'train', self.dataset_config, scaler) if train_indices else None
        val_dataset = self.final_dataset_class(client_subset.dataset, val_indices, 'val', self.dataset_config, scaler) if val_indices else None
        test_dataset = self.final_dataset_class(client_subset.dataset, test_indices, 'test', self.dataset_config, scaler) if test_indices else None

        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, generator=g, num_workers=N_WORKERS, worker_init_fn=seed_worker) if train_dataset else DataLoader([])
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=N_WORKERS, worker_init_fn=seed_worker) if val_dataset else DataLoader([])
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=N_WORKERS, worker_init_fn=seed_worker) if test_dataset else DataLoader([])
        return train_loader, val_loader, test_loader

    def _process_xy_dict(self, xy_data: Dict, scaler=None):
        """Process direct {'X': array, 'y': array} input (e.g., for Heart)."""
        # Assumes scaling was handled by the loader if needed (e.g., Heart loader)
        X, y = xy_data['X'], xy_data['y']
        if len(X) == 0: print(f"Warn: Client xy_dict empty."); empty_loader = DataLoader([]); return (empty_loader,) * 3
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = self._split_data(X, y, seed=42)

        # Instantiate final dataset class with the SPLIT arrays. Scaler passed is None.
        train_dataset = self.final_dataset_class(X=X_train, y=y_train, split_type='train', dataset_config=self.dataset_config, scaler_obj=None) if len(X_train) > 0 else None
        val_dataset = self.final_dataset_class(X=X_val, y=y_val, split_type='val', dataset_config=self.dataset_config, scaler_obj=None) if len(X_val) > 0 else None
        test_dataset = self.final_dataset_class(X=X_test, y=y_test, split_type='test', dataset_config=self.dataset_config, scaler_obj=None) if len(X_test) > 0 else None

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, generator=g, num_workers=N_WORKERS, worker_init_fn=seed_worker) if train_dataset else DataLoader([])
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=N_WORKERS, worker_init_fn=seed_worker) if val_dataset else DataLoader([])
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=N_WORKERS, worker_init_fn=seed_worker) if test_dataset else DataLoader([])
        return train_loader, val_loader, test_loader

    def _process_path_dict(self, path_data: Dict, scaler=None):
        """Process {'X': paths, 'y': paths/labels} input (e.g., for ISIC, IXITiny)."""
        # Scaling not applicable here
        X_paths, y_data = path_data['X'], path_data['y']
        if len(X_paths) == 0: print(f"Warn: Client path_dict empty."); empty_loader = DataLoader([]); return (empty_loader,) * 3
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = self._split_data(X_paths, y_data, seed=42)

        # Instantiate final dataset class (e.g., ISICDataset) with split paths/labels
        train_dataset = self.final_dataset_class(X_train, y_train, is_train=True) if len(X_train) > 0 else None
        val_dataset = self.final_dataset_class(X_val, y_val, is_train=False) if len(X_val) > 0 else None
        test_dataset = self.final_dataset_class(X_test, y_test, is_train=False) if len(X_test) > 0 else None

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, generator=g, num_workers=N_WORKERS, worker_init_fn=seed_worker) if train_dataset else DataLoader([])
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=N_WORKERS, worker_init_fn=seed_worker) if val_dataset else DataLoader([])
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=N_WORKERS, worker_init_fn=seed_worker) if test_dataset else DataLoader([])
        return train_loader, val_loader, test_loader

    # --- Splitting Helper Functions ---
    def _split_data(self, X, y, test_size=0.2, val_size=0.2, seed=42):
        """Splits arrays/lists (like X, y or paths) into train, val, test."""
        # (Keep function code as provided in the previous data_processing.py)
        num_samples = len(X)
        if num_samples < 3: print(f"Warn: Cannot split {num_samples} samples."); return (X, y), (X[:0], y[:0]), (X[:0], y[:0])
        indices = np.arange(num_samples); stratify_param = y if len(np.unique(y)) > 1 and len(y) == len(X) else None
        try:
            idx_temp, idx_test = train_test_split(indices, test_size=test_size, random_state=np.random.RandomState(seed), stratify=stratify_param)
            relative_val_size = val_size / (1.0 - test_size) if (1.0 - test_size) > 0 else 0.0
            if relative_val_size >= 1.0 or len(idx_temp) < 2 or relative_val_size <= 0: idx_train, idx_val = ([], idx_temp) if relative_val_size >= 1.0 else (idx_temp, [])
            else: stratify_temp = y[idx_temp] if stratify_param is not None else None; idx_train, idx_val = train_test_split(idx_temp, test_size=relative_val_size, random_state=np.random.RandomState(seed + 1), stratify=stratify_temp)
        except ValueError: print(f"Warn: Stratified split failed, using non-stratified."); idx_temp, idx_test = train_test_split(indices, test_size=test_size, random_state=np.random.RandomState(seed)); relative_val_size = val_size / (1.0 - test_size) if (1.0 - test_size) > 0 else 0.0
        if relative_val_size >= 1.0 or len(idx_temp) < 2 or relative_val_size <= 0: idx_train, idx_val = ([], idx_temp) if relative_val_size >= 1.0 else (idx_temp, [])
        else: idx_train, idx_val = train_test_split(idx_temp, test_size=relative_val_size, random_state=np.random.RandomState(seed + 1))
        if isinstance(X, np.ndarray): X_train, y_train=X[idx_train], y[idx_train]; X_val, y_val=X[idx_val], y[idx_val]; X_test, y_test=X[idx_test], y[idx_test]
        elif isinstance(X, list): X_train=[X[i] for i in idx_train]; y_train=[y[i] for i in idx_train]; X_val=[X[i] for i in idx_val]; y_val=[y[i] for i in idx_val]; X_test=[X[i] for i in idx_test]; y_test=[y[i] for i in idx_test]
        else: raise TypeError(f"Unsupported type for splitting: {type(X)}")
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

    def _split_indices(self, indices: list, test_size=0.2, val_size=0.2, seed=42):
        """Splits a list of indices into train, val, test index lists."""
        # (Keep function code as provided in the previous data_processing.py)
        num_samples = len(indices)
        if num_samples < 3: print(f"Warn: Cannot split {num_samples} indices."); return indices, [], []
        indices_temp, test_indices = train_test_split(indices, test_size=test_size, random_state=np.random.RandomState(seed))
        relative_val_size = val_size / (1.0 - test_size) if (1.0 - test_size) > 0 else 0.0
        if relative_val_size >= 1.0 or len(indices_temp) < 2 or relative_val_size <= 0: train_indices, val_indices = ([], indices_temp) if relative_val_size >= 1.0 else (indices_temp, [])
        else: train_indices, val_indices = train_test_split(indices_temp, test_size=relative_val_size, random_state=np.random.RandomState(seed + 1))
        return train_indices, val_indices, test_indices