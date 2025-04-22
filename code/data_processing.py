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
from typing import Dict, List, Tuple, Optional, Any
import traceback

# Import necessary components from other modules
from configs import N_WORKERS
from helper import g, seed_worker # Random generator and worker seeding
from data_sets import (
    BaseTabularDataset, BaseImageDataset,
    SyntheticDataset, CreditDataset, HeartDataset, EMNISTDataset,
    CIFARDataset, ISICDataset, IXITinyDataset
)

class DataPreprocessor:
    """
    Takes client input data (Subsets, xy_dicts, or path_dicts) and creates
    train, validation, and test DataLoaders for each client. Handles
    train/val/test splitting and potentially coordinates scaling based on config.
    """
    def __init__(self, dataset_name: str, dataset_config: Dict):
        """
        Initializes the DataPreprocessor.

        Args:
            dataset_name (str): The name of the dataset being processed.
            dataset_config (Dict): The configuration dictionary for the dataset
                                   (from configs.DEFAULT_PARAMS).
        """
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.batch_size = dataset_config['batch_size']
        self.final_dataset_class = self._get_final_dataset_class()

        # Determine if scaling should be handled by this preprocessor step.
        # Scaling is needed if 'standard_scale' is specified AND it's not
        # handled internally by the dataset's specific loader (like HeartDataset).
        self.needs_scaling_here = (
            'standard_scale' in self.dataset_config.get('needs_preprocessing', []) and
            self.dataset_name != 'Heart'
        )

        # Map input data types to processing methods
        self._processor_map = {
            'subset': self._process_subset,        # Input: torch.utils.data.Subset
            'xy_dict': self._process_xy_dict,      # Input: {'X': np.ndarray, 'y': np.ndarray}
            'path_dict': self._process_path_dict,  # Input: {'X': List[str], 'y': Any}
        }

    def _get_final_dataset_class(self) -> type:
         """Gets the actual final Dataset class (e.g., SyntheticDataset) from config."""
         class_name = self.dataset_config.get('dataset_class')
         if not class_name:
             raise ValueError(f"'dataset_class' not defined in config for {self.dataset_name}")

         # Check if class exists in the datasets module (imported above)
         if hasattr(sys.modules['datasets'], class_name):
             return getattr(sys.modules['datasets'], class_name)
         else:
             raise ImportError(f"Dataset class '{class_name}' not found in datasets.py")

    def process_client_data(self,
                            client_input_data: Dict[str, Any],
                            input_type: str
                           ) -> Dict[str, Tuple[DataLoader, DataLoader, DataLoader]]:
        """
        Processes data for multiple clients based on the input type.

        Args:
            client_input_data (Dict): Data for each client, format depends on input_type.
                                     e.g., { 'client_1': Subset(...), ... }
                                     or { 'client_1': {'X': ..., 'y': ...}, ... }
            input_type (str): The format of the data in client_input_data
                              ('subset', 'xy_dict', 'path_dict').

        Returns:
            Dict: A dictionary mapping client_id to a tuple of
                  (train_loader, val_loader, test_loader).
        """
        processed_data = {} # {client_id: (train_loader, val_loader, test_loader)}

        processor_func = self._processor_map.get(input_type)
        if processor_func is None:
            raise ValueError(f"Unknown input_type for DataPreprocessor: {input_type}")

        print(f"Preprocessing client data using method for input type: '{input_type}'")
        all_scalers: Dict[str, StandardScaler] = {} # Store scalers fitted on train sets if needed

        # --- Pass 1 (Optional): Fit scalers on training splits if needed ---
        # This is only done for 'subset' input type where data hasn't been pre-split/scaled.
        if input_type == 'subset' and self.needs_scaling_here:
            print("Preprocessing Pass 1 (Subset Scaling): Fitting scalers...")
            for client_id, client_subset in client_input_data.items():
                # Skip if subset is invalid or empty
                if not isinstance(client_subset, Subset) or not client_subset.indices:
                    continue

                # Perform temporary split just to get training indices
                train_indices, _, _ = self._split_indices(client_subset.indices, seed=42)
                if not train_indices:
                    # No training data for this client to fit scaler
                    continue

                # Instantiate final dataset class temporarily to access fit_scaler method
                try:
                    temp_train_dataset = self.final_dataset_class(
                        base_dataset=client_subset.dataset,
                        indices=train_indices,
                        split_type='train', # Mark as training for potential internal logic
                        dataset_config=self.dataset_config
                    )

                    # Fit scaler if the method exists (should exist for tabular datasets needing scaling)
                    if hasattr(temp_train_dataset, 'fit_scaler'):
                        scaler = temp_train_dataset.fit_scaler()
                        if scaler:
                            all_scalers[client_id] = scaler
                    else:
                         # This indicates a potential config mismatch if scaling was expected
                         print(f"Warning: Dataset class {self.final_dataset_class.__name__} "
                               f"does not have fit_scaler method, but scaling was requested "
                               f"for dataset {self.dataset_name}.")
                    # Clean up temporary dataset object
                    del temp_train_dataset
                except Exception as e:
                    print(f"Error during scaler fitting for client {client_id}: {e}")
                    # Decide how to proceed: skip client, raise error, etc.
                    # Currently skips fitting scaler for this client.

        # --- Pass 2: Create final datasets and DataLoaders for all clients ---
        print("Preprocessing Pass 2: Creating Datasets and DataLoaders...")
        for client_id, data in client_input_data.items():
            # Retrieve the fitted scaler for this client, if any
            scaler = all_scalers.get(client_id)
            try:
                 # Call the appropriate processing method (_process_subset, _process_xy_dict, etc.)
                 processed_data[client_id] = processor_func(data, scaler=scaler)
            except Exception as e:
                 print(f"Error processing data for client {client_id}: {e}")
                 traceback.print_exc()
                 # Assign empty loaders if processing fails for a client
                 empty_loader = DataLoader([])
                 processed_data[client_id] = (empty_loader, empty_loader, empty_loader)


        return processed_data

    def _process_subset(self,
                        client_subset: Subset,
                        scaler: Optional[StandardScaler] = None
                       ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Processes data for a client when input is a `Subset` object
        (typically after partitioning a base dataset). Splits indices,
        instantiates final dataset wrappers, and creates DataLoaders.

        Args:
            client_subset (Subset): The subset of data indices for this client.
            scaler (Optional[StandardScaler]): A pre-fitted scaler to be used
                                               (for val/test splits, or if scaling
                                               was done globally, though not typical here).

        Returns:
            Tuple[DataLoader, DataLoader, DataLoader]: train, val, test loaders.
        """
        if not isinstance(client_subset, Subset) or not client_subset.indices:
            print(f"Warning: Client subset is empty or invalid.")
            empty_loader = DataLoader([])
            return (empty_loader, empty_loader, empty_loader)

        # 1. Split the client's indices into train, validation, test sets
        train_indices, val_indices, test_indices = self._split_indices(
            client_subset.indices, seed=42 # Use fixed seed for reproducible splits per client
        )

        base_dataset = client_subset.dataset

        # 2. Determine if the final dataset type is tabular (requires specific handling)
        is_tabular = issubclass(self.final_dataset_class, BaseTabularDataset)

        train_dataset = None
        val_dataset = None
        test_dataset = None

        # 3. Instantiate the FINAL dataset class for each split
        if is_tabular:
            # For tabular data using subset mode: pass base_dataset, indices, and scaler
            if train_indices:
                train_dataset = self.final_dataset_class(
                    base_dataset=base_dataset, indices=train_indices,
                    split_type='train', dataset_config=self.dataset_config,
                    scaler_obj=scaler # Pass scaler fitted in Pass 1
                )
            if val_indices:
                val_dataset = self.final_dataset_class(
                    base_dataset=base_dataset, indices=val_indices,
                    split_type='val', dataset_config=self.dataset_config,
                    scaler_obj=scaler # Pass scaler fitted in Pass 1
                )
            if test_indices:
                test_dataset = self.final_dataset_class(
                    base_dataset=base_dataset, indices=test_indices,
                    split_type='test', dataset_config=self.dataset_config,
                    scaler_obj=scaler # Pass scaler fitted in Pass 1
                )
        else:
            # For non-tabular (e.g., Image) datasets from subset:
            # Extract the actual data corresponding to the indices first.
            def get_data_from_indices(indices: List[int]) -> Tuple[Optional[Any], Optional[Any]]:
                if not indices:
                    return None, None
                try:
                    # Assuming base_dataset.__getitem__ returns (feature, label)
                    features = [base_dataset[i][0] for i in indices]
                    labels = [base_dataset[i][1] for i in indices]

                    # Stack if features are numpy arrays (common for torchvision base)
                    if features and isinstance(features[0], np.ndarray):
                        features = np.stack(features, axis=0)
                        labels = np.array(labels) # Convert labels list to array too
                    elif features and isinstance(features[0], torch.Tensor):
                        features = torch.stack(features, dim=0)
                        labels = torch.tensor(labels) # Convert labels list to tensor
                    # Add handling for PIL Images etc. if needed

                    return features, labels
                except Exception as e:
                    print(f"Error extracting data from indices: {e}")
                    return None, None # Return None if extraction fails

            X_train, y_train = get_data_from_indices(train_indices)
            X_val, y_val = get_data_from_indices(val_indices)
            X_test, y_test = get_data_from_indices(test_indices)

            # Instantiate the final Image dataset class with the extracted X, y arrays
            # These classes handle their own transforms internally.
            if X_train is not None:
                train_dataset = self.final_dataset_class(X=X_train, y=y_train, is_train=True)
            if X_val is not None:
                val_dataset = self.final_dataset_class(X=X_val, y=y_val, is_train=False)
            if X_test is not None:
                test_dataset = self.final_dataset_class(X=X_test, y=y_test, is_train=False)

        # 4. Create DataLoaders for each split
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True,
            generator=g, num_workers=N_WORKERS, worker_init_fn=seed_worker
        ) if train_dataset is not None else DataLoader([])

        val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=N_WORKERS, worker_init_fn=seed_worker
        ) if val_dataset is not None else DataLoader([])

        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=N_WORKERS, worker_init_fn=seed_worker
        ) if test_dataset is not None else DataLoader([])

        return train_loader, val_loader, test_loader

    def _process_xy_dict(self,
                         xy_data: Dict[str, np.ndarray],
                         scaler: Optional[StandardScaler] = None # Usually None here
                        ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Processes data for a client when input is a dictionary of NumPy arrays
        {'X': features, 'y': labels}. Used for pre-split data like Heart.
        Assumes scaling (if needed) was done by the data loader.

        Args:
            xy_data (Dict): Dictionary containing 'X' and 'y' NumPy arrays.
            scaler (Optional[StandardScaler]): Expected to be None, as scaling
                                               should be handled before this point
                                               for this input type.

        Returns:
            Tuple[DataLoader, DataLoader, DataLoader]: train, val, test loaders.
        """
        X, y = xy_data.get('X'), xy_data.get('y')

        if X is None or y is None or len(X) == 0:
            print(f"Warning: Client xy_dict empty or invalid.")
            empty_loader = DataLoader([])
            return (empty_loader, empty_loader, empty_loader)

        # 1. Split the direct X, y arrays into train, val, test
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = self._split_data(X, y, seed=42)

        # 2. Instantiate final dataset class with the SPLIT arrays.
        #    The dataset wrapper should handle tensor conversion ('direct' mode).
        #    Pass scaler_obj=None as scaling is assumed pre-applied.
        train_dataset = self.final_dataset_class(
            X=X_train, y=y_train, split_type='train',
            dataset_config=self.dataset_config, scaler_obj=None
        ) if len(X_train) > 0 else None

        val_dataset = self.final_dataset_class(
            X=X_val, y=y_val, split_type='val',
            dataset_config=self.dataset_config, scaler_obj=None
        ) if len(X_val) > 0 else None

        test_dataset = self.final_dataset_class(
            X=X_test, y=y_test, split_type='test',
            dataset_config=self.dataset_config, scaler_obj=None
        ) if len(X_test) > 0 else None

        # 3. Create DataLoaders
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True,
            generator=g, num_workers=N_WORKERS, worker_init_fn=seed_worker
        ) if train_dataset is not None else DataLoader([])

        val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=N_WORKERS, worker_init_fn=seed_worker
        ) if val_dataset is not None else DataLoader([])

        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=N_WORKERS, worker_init_fn=seed_worker
        ) if test_dataset is not None else DataLoader([])

        return train_loader, val_loader, test_loader

    def _process_path_dict(self,
                           path_data: Dict[str, Any],
                           scaler: Optional[StandardScaler] = None # Not used for paths
                          ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Processes data for a client when input is a dictionary containing
        paths {'X': list_of_image_paths, 'y': list_of_label_paths_or_labels}.
        Used for pre-split path-based datasets like ISIC, IXITiny.

        Args:
            path_data (Dict): Dictionary containing 'X' and 'y' (paths or labels).
            scaler (Optional[StandardScaler]): Ignored for path data.

        Returns:
            Tuple[DataLoader, DataLoader, DataLoader]: train, val, test loaders.
        """
        X_paths = path_data.get('X')
        y_data = path_data.get('y') # Can be paths or labels

        if X_paths is None or y_data is None or len(X_paths) == 0:
            print(f"Warning: Client path_dict empty or invalid.")
            empty_loader = DataLoader([])
            return (empty_loader, empty_loader, empty_loader)

        # 1. Split the paths/labels into train, val, test lists/arrays
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = self._split_data(
            X_paths, y_data, seed=42
        )

        # 2. Instantiate final dataset class (e.g., ISICDataset, IXITinyDataset)
        #    These classes expect paths/labels and handle loading/transforms internally.
        train_dataset = self.final_dataset_class(
            X_train, y_train, is_train=True
        ) if len(X_train) > 0 else None

        val_dataset = self.final_dataset_class(
            X_val, y_val, is_train=False
        ) if len(X_val) > 0 else None

        test_dataset = self.final_dataset_class(
            X_test, y_test, is_train=False
        ) if len(X_test) > 0 else None

        # 3. Create DataLoaders
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True,
            generator=g, num_workers=N_WORKERS, worker_init_fn=seed_worker
        ) if train_dataset is not None else DataLoader([])

        val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=N_WORKERS, worker_init_fn=seed_worker
        ) if val_dataset is not None else DataLoader([])

        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=N_WORKERS, worker_init_fn=seed_worker
        ) if test_dataset is not None else DataLoader([])

        return train_loader, val_loader, test_loader

    # --- Splitting Helper Functions ---
    def _split_data(self, X: Any, y: Any, test_size: float = 0.2, val_size: float = 0.2, seed: int = 42
                   ) -> Tuple[Tuple[Any, Any], Tuple[Any, Any], Tuple[Any, Any]]:
        """
        Splits data (arrays or lists like X, y or paths) into train, val, test sets.
        Handles stratification for classification tasks if possible.

        Args:
            X: Features or paths (NumPy array or list).
            y: Labels or paths (NumPy array or list).
            test_size: Proportion of data for the test set.
            val_size: Proportion of data for the validation set (relative to the whole).
            seed: Random seed for reproducibility.

        Returns:
            Tuple containing (X_train, y_train), (X_val, y_val), (X_test, y_test).
        """
        num_samples = len(X)
        if num_samples < 3:
            print(f"Warning: Cannot split data with only {num_samples} samples. Using all for train.")
            # Return empty arrays/lists of the same type as input y for consistency
            empty_y = y[:0]
            empty_X = X[:0] if isinstance(X, (np.ndarray, list)) else [] # Adjust for X type
            return (X, y), (empty_X, empty_y), (empty_X, empty_y)

        indices = np.arange(num_samples)
        stratify_param = None
        # Try to stratify if y looks like labels and has multiple classes
        if isinstance(y, (np.ndarray, list)) and len(y) == num_samples:
             unique_labels, counts = np.unique(y, return_counts=True)
             if len(unique_labels) > 1 and all(c > 1 for c in counts): # Check for >1 class and >1 sample per class
                  stratify_param = y
             #else: # Dont stratify if only 1 class or only 1 sample in a class
             #    print("Warning: Cannot stratify split (single class or insufficient samples per class).")

        try:
            # Split into train+val and test
            idx_temp, idx_test = train_test_split(
                indices,
                test_size=test_size,
                random_state=np.random.RandomState(seed),
                stratify=stratify_param
            )

            # Adjust val_size relative to the temp set size
            if (1.0 - test_size) <= 1e-9: # Avoid division by zero if test_size is 1 or very close
                 relative_val_size = 0.0
            else:
                 relative_val_size = val_size / (1.0 - test_size)

            # Split temp set into train and validation
            if relative_val_size >= 1.0: # Validation set takes all of temp set
                 idx_train, idx_val = [], idx_temp
            elif len(idx_temp) < 2 or relative_val_size <= 0: # Not enough data for validation split
                 idx_train, idx_val = idx_temp, []
            else:
                 # Stratify validation split if possible
                 stratify_temp = y[idx_temp] if stratify_param is not None else None
                 try:
                      idx_train, idx_val = train_test_split(
                         idx_temp,
                         test_size=relative_val_size,
                         random_state=np.random.RandomState(seed + 1), # Use different seed
                         stratify=stratify_temp
                      )
                 except ValueError: # Fallback if stratification fails on the smaller temp set
                      print("Warning: Stratification failed on train/val split, using non-stratified.")
                      idx_train, idx_val = train_test_split(
                         idx_temp, test_size=relative_val_size,
                         random_state=np.random.RandomState(seed + 1)
                      )

        except ValueError as e:
             # Fallback to non-stratified split if initial stratification fails
             print(f"Warning: Stratified split failed ({e}), falling back to non-stratified split.")
             idx_temp, idx_test = train_test_split(
                 indices, test_size=test_size, random_state=np.random.RandomState(seed)
             )
             if (1.0 - test_size) <= 1e-9: relative_val_size = 0.0
             else: relative_val_size = val_size / (1.0 - test_size)

             if relative_val_size >= 1.0:
                 idx_train, idx_val = [], idx_temp
             elif len(idx_temp) < 2 or relative_val_size <= 0:
                 idx_train, idx_val = idx_temp, []
             else:
                 idx_train, idx_val = train_test_split(
                     idx_temp, test_size=relative_val_size,
                     random_state=np.random.RandomState(seed + 1)
                 )

        # --- Return slices based on the original type of X and y ---
        if isinstance(X, np.ndarray):
            X_train, y_train = X[idx_train], y[idx_train]
            X_val, y_val = X[idx_val], y[idx_val]
            X_test, y_test = X[idx_test], y[idx_test]
        elif isinstance(X, list): # Handle lists (e.g., paths)
            X_train = [X[i] for i in idx_train]
            # Ensure y slicing works if y is also a list or an array
            y_train = y[idx_train] if isinstance(y, np.ndarray) else [y[i] for i in idx_train]
            X_val = [X[i] for i in idx_val]
            y_val = y[idx_val] if isinstance(y, np.ndarray) else [y[i] for i in idx_val]
            X_test = [X[i] for i in idx_test]
            y_test = y[idx_test] if isinstance(y, np.ndarray) else [y[i] for i in idx_test]
        else:
             raise TypeError(f"Unsupported data type for splitting: {type(X)}")

        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

    def _split_indices(self, indices: List[int], test_size: float = 0.2, val_size: float = 0.2, seed: int = 42
                      ) -> Tuple[List[int], List[int], List[int]]:
        """
        Splits a list of indices into train, validation, and test index lists.

        Args:
            indices: List of indices to split.
            test_size: Proportion for the test set.
            val_size: Proportion for the validation set (relative to the whole).
            seed: Random seed.

        Returns:
            Tuple containing (train_indices, val_indices, test_indices).
        """
        num_samples = len(indices)
        if num_samples < 3:
            print(f"Warning: Cannot split list of {num_samples} indices effectively.")
            return indices, [], [] # Return all as train

        # Split into train+val and test indices (cannot stratify with only indices)
        try:
             indices_temp, test_indices = train_test_split(
                 indices,
                 test_size=test_size,
                 random_state=np.random.RandomState(seed)
             )
        except Exception as e: # Catch potential errors during split
             print(f"Error during test split of indices: {e}")
             return indices, [], []

        # Adjust validation size relative to the remaining temp set size
        if (1.0 - test_size) <= 1e-9: # Avoid division by zero
             relative_val_size = 0.0
        else:
             relative_val_size = val_size / (1.0 - test_size)

        # Split temp set into train and validation
        if relative_val_size >= 1.0:
            train_indices, val_indices = [], indices_temp
        elif len(indices_temp) < 2 or relative_val_size <= 0:
            train_indices, val_indices = indices_temp, []
        else:
             try:
                  train_indices, val_indices = train_test_split(
                      indices_temp,
                      test_size=relative_val_size,
                      random_state=np.random.RandomState(seed + 1) # Use different seed
                  )
             except Exception as e:
                  print(f"Error during train/val split of indices: {e}")
                  # Fallback: assign all remaining to train
                  train_indices, val_indices = indices_temp, []

        # Ensure return types are lists
        return list(train_indices), list(val_indices), list(test_indices)