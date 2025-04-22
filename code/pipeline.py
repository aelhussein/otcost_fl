"""
Main experiment pipeline orchestration.
Initializes data, creates models/servers, runs training/evaluation loops.
"""
import os
import sys
import traceback
import numpy as np
import hashlib
import torch
import torch.nn as nn
from torch.utils.data import Subset, ConcatDataset, DataLoader, Dataset as TorchDataset
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Any, Union, Callable
from sklearn.preprocessing import StandardScaler
from data_sets import BaseTabularDataset

# Import project modules
from configs import (
    ROOT_DIR, DATA_DIR, ALGORITHMS, DEVICE, DEFAULT_PARAMS, DATASET_COSTS # Added DEFAULT_PARAMS, DATASET_COSTS
)
# MODIFIED: Import MetricKey
from helper import (
    set_seeds, cleanup_gpu, get_dice_score, get_parameters_for_dataset,
    get_default_lr, get_default_reg, translate_cost, validate_dataset_config,
    MetricKey
)
# Import server types and data structures
from servers import Server, FedAvgServer, TrainerConfig, SiteData, ModelState
import models as ms # Import model architectures module
from losses import * # Import custom losses if any (may not be needed if criterion map covers all)

# MODIFIED: Import Dataset classes ONLY from data_sets
from data_sets import (
    SyntheticBaseDataset, CreditBaseDataset, SyntheticFeaturesBaseDataset,
    SyntheticDataset, CreditDataset, HeartDataset, SyntheticConceptDataset,
    EMNISTDataset, CIFARDataset, ISICDataset, IXITinyDataset
)
# MODIFIED: Import remaining necessary modules
from data_loading import DATA_LOADERS
from data_partitioning import PARTITIONING_STRATEGIES
# REMOVED: from data_processing import DataPreprocessor (logic integrated or handled differently)
from results_manager import ResultsManager, ExperimentType # Import results manager and Enum


# --- NEW: Data Preprocessor Class (Integrated) ---
# Simplified preprocessor logic, directly integrated or called within pipeline
class DataPreprocessor:
    """Handles preprocessing steps like scaling and creating final DataLoaders."""
    def __init__(self, dataset_name: str, dataset_config: Dict):
        self.dataset_name = dataset_name
        self.config = dataset_config
        self.scaler: Optional[StandardScaler] = None # Scaler fitted on train data

    def process_client_data(self,
                            client_input_data: Dict[str, Any],
                            input_type: str,
                            batch_size: int,
                            n_workers: int,
                            base_seed: int
                           ) -> Dict[str, Tuple[DataLoader, DataLoader, DataLoader]]:
        """
        Transforms raw client data (subsets, dicts, paths) into DataLoaders.
        Handles train/val/test splitting, scaling, and DataLoader creation.

        Args:
            client_input_data: Dict {client_id: data}. Data format depends on input_type.
            input_type: 'subset', 'xy_dict', or 'path_dict'.
            batch_size: Batch size for DataLoaders.
            n_workers: Number of workers for DataLoaders.
            base_seed: Base random seed for reproducibility.


        Returns:
            Dict {client_id: (train_loader, val_loader, test_loader)}.
        """
        client_dataloaders: Dict[str, Tuple[DataLoader, DataLoader, DataLoader]] = {}
        dataset_class_name = self.config.get('dataset_class')
        if not dataset_class_name:
            raise ValueError(f"Missing 'dataset_class' in config for {self.dataset_name}")

        # Dynamically get the dataset class constructor from data_sets module
        try:
             # Use globals() of data_sets module or import explicitly
             dataset_class = getattr(sys.modules['data_sets'], dataset_class_name)
        except (KeyError, AttributeError):
             raise ImportError(f"Dataset class '{dataset_class_name}' not found in data_sets.py")


        print(f"Preprocessing with Dataset Class: {dataset_class.__name__}, Input Type: {input_type}")

        # --- Fit Scaler on Training Data (if needed and possible) ---
        self.scaler = None
        needs_scaling = 'standard_scale' in self.config.get('needs_preprocessing', [])
        if needs_scaling and input_type == 'subset':
             print("Attempting to fit scaler...")
             # Create temporary training datasets for all clients to fit scaler
             temp_train_datasets = []
             for client_id, client_subset in client_input_data.items():
                 if not isinstance(client_subset, Subset): continue # Skip if not a subset
                 # Assume subset mode for scaling fitting implies base_dataset exists
                 base_ds = client_subset.dataset
                 indices = client_subset.indices
                 # Create a temporary dataset instance just for fitting
                 # Note: SyntheticConcept needs special handling if base lacks features
                 # For now, assume base has 'features' if scaling is requested
                 if hasattr(base_ds, 'features'):
                      temp_ds = BaseTabularDataset( # Use base class for fitting
                           base_dataset=base_ds,
                           indices=indices,
                           split_type='train',
                           dataset_config=self.config
                      )
                      if len(temp_ds) > 0:
                           temp_train_datasets.append(temp_ds)
                 else:
                      print(f"Warning: Cannot fit scaler for client {client_id}, base dataset lacks 'features'.")


             if temp_train_datasets:
                  # Concatenate features from all clients' training subsets
                  all_train_features = []
                  for ds in temp_train_datasets:
                       # Access features directly for fitting scaler
                       if ds.mode == 'subset' and hasattr(ds.base_dataset, 'features'):
                           client_features = ds.base_dataset.features[ds.indices]
                           if client_features.ndim == 1: client_features = client_features.reshape(-1, 1)
                           if client_features.size > 0: # Check if empty
                                all_train_features.append(client_features)

                  if all_train_features:
                      all_train_features_np = np.concatenate(all_train_features, axis=0)
                      print(f"Fitting scaler on concatenated training features (shape: {all_train_features_np.shape})...")
                      try:
                           self.scaler = StandardScaler().fit(all_train_features_np)
                           print("Scaler fitted successfully.")
                      except Exception as e:
                           print(f"Error fitting scaler: {e}. Proceeding without scaling.")
                           self.scaler = None # Ensure scaler is None if fitting failed
                  else:
                      print("No training features found to fit scaler.")
             else:
                 print("No suitable training subsets found to fit scaler.")
        elif needs_scaling:
             print(f"Note: Scaling requested but input type is '{input_type}'. Assuming data is pre-scaled or scaling is handled within dataset class.")


        # --- Process each client ---
        for client_id, data in client_input_data.items():
            print(f"  Processing client: {client_id}")
            train_dataset, val_dataset, test_dataset = None, None, None
            common_args = {'dataset_config': self.config, 'scaler_obj': self.scaler}

            try:
                 if input_type == 'subset':
                     if not isinstance(data, Subset) or not hasattr(data, 'dataset') or not hasattr(data, 'indices'):
                          print(f"Warning: Invalid subset data for client {client_id}. Skipping.")
                          continue
                     base_ds = data.dataset
                     indices = data.indices
                     # Create the final Dataset instance using the specific class
                     # Pass translated_cost if the dataset needs it (e.g., Concept)
                     cost_arg = {'translated_cost': data.translated_cost} if hasattr(data, 'translated_cost') else {}
                     full_dataset = dataset_class(base_dataset=base_ds, indices=indices, split_type='full', **common_args, **cost_arg)

                 elif input_type == 'xy_dict':
                     if not isinstance(data, dict) or 'X' not in data or 'y' not in data:
                          print(f"Warning: Invalid xy_dict data for client {client_id}. Skipping.")
                          continue
                     # Data is already loaded (e.g., pre_split)
                     full_dataset = dataset_class(X=data['X'], y=data['y'], is_train=True, split_type='full', **common_args) # is_train influences transforms

                 elif input_type == 'path_dict':
                      if not isinstance(data, dict) or 'X' not in data or 'y' not in data:
                           print(f"Warning: Invalid path_dict data for client {client_id}. Skipping.")
                           continue
                      # Dataset class handles loading from paths in __getitem__
                      full_dataset = dataset_class(image_paths=data['X'], labels=data['y'], is_train=True, split_type='full', **common_args)

                 else:
                     raise ValueError(f"Unknown input_type for preprocessing: {input_type}")

                 # --- Split into Train/Val/Test ---
                 # TODO: Implement robust splitting (e.g., 80/10/10 random split)
                 # For now, using simple split logic. Needs improvement.
                 n_samples = len(full_dataset)
                 if n_samples == 0:
                     print(f"Warning: Client {client_id} has no data after initialization. Skipping.")
                     continue

                 # Use a generator specific to this client for splitting reproducibility
                 client_split_seed = base_seed + int(hashlib.sha256(client_id.encode()).hexdigest(), 16) % (2**32)
                 split_generator = torch.Generator().manual_seed(client_split_seed)

                 # Simple split (e.g., 80% train, 10% val, 10% test)
                 # Adjust percentages as needed
                 n_train = int(0.8 * n_samples)
                 n_val = int(0.1 * n_samples)
                 n_test = n_samples - n_train - n_val

                 if n_train == 0 or n_test == 0: # Require at least train and test data
                     print(f"Warning: Client {client_id} has too few samples ({n_samples}) for proper train/val/test split. Assigning all to train/test.")
                     n_train = max(1, int(0.9 * n_samples))
                     n_test = n_samples - n_train
                     n_val = 0
                     if n_test == 0 and n_train > 0: # Ensure test is not zero if possible
                          n_test = 1
                          n_train -= 1

                 if n_train <= 0: # Still possible if n_samples=1
                      print(f"Error: Cannot create non-empty training set for client {client_id} with {n_samples} samples. Skipping client.")
                      continue


                 print(f"    Splitting {n_samples} samples -> Train: {n_train}, Val: {n_val}, Test: {n_test}")
                 train_subset, val_subset, test_subset = torch.utils.data.random_split(
                     full_dataset, [n_train, n_val, n_test], generator=split_generator
                 )

                 # Wrap subsets to retain attributes if needed (e.g., is_train for transforms)
                 # Need a way for the Dataset class to know its role (train/val/test) for transforms
                 # Option 1: Pass split_type to Dataset constructor (requires modifying __init__)
                 # Option 2: Modify dataset state after creation (less clean)
                 # Option 3: Have separate transform logic outside dataset (preferred)
                 # Let's assume the Dataset's `is_train` property is used by its internal transform logic

                 # Create DataLoaders
                 train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=n_workers, pin_memory=True) # Add pin_memory
                 val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=n_workers, pin_memory=True) if n_val > 0 else None
                 test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=n_workers, pin_memory=True) if n_test > 0 else None

                 # Store loaders
                 client_dataloaders[client_id] = (train_loader, val_loader, test_loader)
                 print(f"    Client {client_id}: Loaders created.")

            except Exception as e:
                 print(f"ERROR processing data for client {client_id}: {e}")
                 traceback.print_exc()
                 # Ensure client is not added if processing fails
                 if client_id in client_dataloaders:
                      del client_dataloaders[client_id]

        return client_dataloaders


@dataclass
class ExperimentConfig:
    """Configuration for an experiment run."""
    dataset: str
    experiment_type: str
    num_clients: Optional[int] = None # Can be overridden by CLI

# --- Main Experiment Class ---
class Experiment:
    """
    Orchestrates the setup and execution of federated learning experiments.
    Handles data loading, partitioning, preprocessing, model/server creation,
    training loops, hyperparameter tuning, evaluation, and results management.
    """
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.default_params = get_parameters_for_dataset(self.config.dataset)
        self.data_dir_root = DATA_DIR
        self.base_seed = self.default_params.get('base_seed', 42)
        print(f"Using base seed: {self.base_seed} for experiment runs.")

        initial_target_num_clients = self.config.num_clients or self.default_params.get('default_num_clients', 2)
        if not isinstance(initial_target_num_clients, int) or initial_target_num_clients <= 0:
             raise ValueError(f"Invalid initial target client count: {initial_target_num_clients}")

        print(f"Initializing ResultsManager for {initial_target_num_clients} target clients (filename).")
        self.results_manager = ResultsManager(
            root_dir=ROOT_DIR,
            dataset=self.config.dataset,
            experiment_type=self.config.experiment_type, # This is the *overall* type (e.g., evaluation)
            num_clients=initial_target_num_clients
        )
        # This will store the *actual* number of clients used in a specific run, determined later
        self.num_clients_for_run: Optional[int] = None
        # MODIFIED: Add cache for dataloaders during tuning
        self.dataloader_cache: Dict[Tuple[Any, int], Dict] = {} # Key: (cost, run_idx), Value: client_dataloaders

    def run_experiment(self, costs: List[Any]):
        """Main entry point to run the configured experiment type."""
        # Clear cache at the start of a new experiment type run
        self.dataloader_cache = {}

        if self.config.experiment_type == ExperimentType.EVALUATION:
            results, _ = self._run_final_evaluation(costs)
            return results
        elif self.config.experiment_type in [ExperimentType.LEARNING_RATE, ExperimentType.REG_PARAM]:
            return self._run_hyperparam_tuning(costs)
        else:
            raise ValueError(f"Unsupported experiment type: {self.config.experiment_type}")

    def _check_existing_results(self, costs: List[Any], experiment_type_to_check: str) -> Tuple[Optional[Dict], List[Any], int]:
        """Checks existing results file, determines remaining runs/costs for a specific experiment type."""
        results, metadata = self.results_manager.load_results(experiment_type_to_check)
        is_tuning = experiment_type_to_check != ExperimentType.EVALUATION
        runs_key = 'runs_tune' if is_tuning else 'runs'
        target_runs = self.default_params.get(runs_key, 1)
        remaining_costs = list(costs)
        completed_runs = 0

        if results is not None:
            has_errors = metadata is not None and metadata.get('contains_errors', False)
            if has_errors:
                print(f"[{experiment_type_to_check}] Previous results file contains errors flag. Re-running all.")
                # Optionally clear results dict here if re-running everything
                # results = {}
                return results, remaining_costs, 0 # Start from run 0

            completed_costs_set = set(results.keys())
            remaining_costs = [c for c in costs if c not in completed_costs_set]

            if completed_costs_set:
                # --- Determine completed runs based on results structure ---
                first_cost = next(iter(completed_costs_set))
                try:
                    # Navigate structure: results[cost][param_val][server_type]['global'][loss_key]
                    first_param = next(iter(results[first_cost])) # LR or Reg Param value
                    first_server = next(iter(results[first_cost][first_param])) # Server type
                    # MODIFIED: Use MetricKey for consistency
                    loss_key = MetricKey.VAL_LOSSES if is_tuning else MetricKey.TEST_LOSSES

                    if (first_server in results[first_cost][first_param] and
                        'global' in results[first_cost][first_param][first_server] and # Use generic server var
                        isinstance(results[first_cost][first_param][first_server]['global'], dict) and
                        loss_key in results[first_cost][first_param][first_server]['global']):

                        # Get the list of loss lists (one inner list per run)
                        loss_data = results[first_cost][first_param][first_server]['global'][loss_key]

                        # Determine completed runs based on the length of the outer list
                        if isinstance(loss_data, list):
                             # Filter out potential None entries if saving failed mid-run
                             valid_runs = [run for run in loss_data if isinstance(run, list)]
                             completed_runs = len(valid_runs)
                             if completed_runs != len(loss_data):
                                 print(f"Warning: Found non-list entries in loss data for {loss_key}. Counted {completed_runs} valid runs.")
                        else:
                             print(f"Warning: Expected list for '{loss_key}', found {type(loss_data)}. Assuming 0 completed runs.")
                             completed_runs = 0
                    else:
                        # Structure is broken, assume 0 runs completed
                        print(f"Warning: Expected structure not found for {loss_key} under cost {first_cost}, param {first_param}, server {first_server}.")
                        completed_runs = 0

                    # --- Check for errors within completed costs ---
                    costs_with_errors = set()
                    for cost_key in completed_costs_set:
                        if self.results_manager._check_for_errors_in_results(results.get(cost_key)):
                            print(f"Errors found within results for cost {cost_key}. Adding to re-run list.")
                            costs_with_errors.add(cost_key)
                            if cost_key not in remaining_costs:
                                remaining_costs.append(cost_key)

                    # If a cost had errors, we might need to reset its completed run count or re-run all runs for it.
                    # For simplicity now: if errors exist, we re-run *all* remaining runs for *all* costs (including error costs).
                    if costs_with_errors:
                         print("Errors detected in some costs. Resetting completed runs to 0 to ensure full rerun.")
                         completed_runs = 0 # Force rerun of all runs if any cost had an error
                         remaining_costs = list(costs) # Rerun all costs


                except (StopIteration, KeyError, IndexError, TypeError, AttributeError) as e:
                    completed_runs = 0
                    print(f"Could not reliably determine completed runs due to results structure error/missing data: {e}. Assuming 0 runs.")
                    remaining_costs = list(costs) # Force re-run all costs

            # Sort remaining costs for consistent processing order
            remaining_costs = sorted(list(set(remaining_costs)), key=lambda x: costs.index(x) if x in costs else float('inf'))

            print(f"[{experiment_type_to_check}] Found {completed_runs}/{target_runs} completed valid run(s) in existing results.")

            if completed_runs >= target_runs and not remaining_costs:
                print(f"[{experiment_type_to_check}] All target runs completed and no missing/failed costs found.")
                remaining_costs = [] # Ensure it's empty
            elif completed_runs < target_runs:
                # If not enough runs completed, we need to run remaining runs for ALL costs
                remaining_costs = list(costs) # Reset to all costs
                print(f"[{experiment_type_to_check}] Target runs ({target_runs}) not met. Will run all costs for remaining runs.")
            # else: completed_runs >= target_runs but some costs were missing/failed - remaining_costs already set

        else: # No results file found
            print(f"[{experiment_type_to_check}] No existing results file found.")
            remaining_costs = list(costs)
            completed_runs = 0

        print(f"[{experiment_type_to_check}] Remaining costs to process: {remaining_costs}")
        return results, remaining_costs, completed_runs

    def _get_skew_type_from_dataset_name(self, dataset_name: str) -> Optional[str]:
        """Helper to determine skew type from dataset name convention."""
        if 'Synthetic_Label' in dataset_name: return 'label'
        if 'Synthetic_Feature' in dataset_name: return 'feature'
        if 'Synthetic_Concept' in dataset_name: return 'concept'
        # Add checks for other dataset families if needed
        return None # Or 'mixed', 'unknown' etc.

    def _run_hyperparam_tuning(self, costs: List[Any]) -> Dict:
        """Runs hyperparameter tuning loops."""
        # Check existing results for the specific tuning type
        tuning_type = self.config.experiment_type
        results, remaining_costs, completed_runs = self._check_existing_results(costs, tuning_type)
        runs_tune = self.default_params.get('runs_tune', 1)

        if not remaining_costs and completed_runs >= runs_tune:
            print(f"Hyperparameter tuning ({tuning_type}) already complete.")
            return results if results is not None else {}

        remaining_runs_count = runs_tune - completed_runs
        if remaining_runs_count <= 0:
            # This case means target runs met, but some costs were missing/failed
            if remaining_costs:
                print(f"Target runs complete, but costs {remaining_costs} missing/failed. Rerunning these costs for {runs_tune} runs.")
                # We need to run *all* runs for the failed costs
                remaining_runs_count = runs_tune
                completed_runs = 0 # Treat as starting over for these costs
                # results might contain partial data for failed costs, clear it? Or merge carefully?
                # Let's just overwrite for simplicity for now.
                for cost in remaining_costs:
                    if cost in results: del results[cost]
            else:
                 # Should not happen if check_existing_results is correct
                 print(f"Target tuning runs ({tuning_type}) already completed.")
                 return results if results is not None else {}

        print(f"Starting {remaining_runs_count} hyperparameter tuning run(s) for {tuning_type}...")
        results = results if results is not None else {}
        num_clients_metadata = None # Store the client count used for metadata saving

        skew_type = self._get_skew_type_from_dataset_name(self.config.dataset)

        # Iterate through the required number of *additional* runs
        for run_offset in range(remaining_runs_count):
            current_run_total = completed_runs + run_offset + 1 # 1-based run index
            current_seed = self.base_seed + current_run_total - 1 # Seed based on total run index
            print(f"\n--- Starting Tuning Run {current_run_total}/{runs_tune} (Seed: {current_seed}) ---")
            set_seeds(current_seed)
            run_meta = {'run_number': current_run_total, 'seed_used': current_seed}
            if skew_type: run_meta['skew_type'] = skew_type
            cost_client_counts = {} # Track clients used per cost in this run

            # Determine which costs to run in this iteration
            costs_to_run_this_iter = remaining_costs if completed_runs == 0 else costs # Run all costs if starting over runs, else just remaining

            for cost in costs_to_run_this_iter:
                print(f"\n--- Processing Cost: {cost} (Run {current_run_total}) ---")
                try:
                    # Determine client count for this specific cost
                    num_clients_this_cost = self._get_final_client_count(self.default_params, cost)
                    cost_client_counts[cost] = num_clients_this_cost
                    # Use the client count from the first processed cost for overall metadata
                    if num_clients_metadata is None: num_clients_metadata = num_clients_this_cost

                    # --- MODIFIED: Use dataloader cache ---
                    cache_key = (cost, current_run_total) # Use run_total for unique key across runs
                    if cache_key in self.dataloader_cache:
                        print(f"  Using cached dataloaders for cost {cost}, run {current_run_total}")
                        client_dataloaders = self.dataloader_cache[cache_key]
                    else:
                        print(f"  Initializing dataloaders for cost {cost}, run {current_run_total}")
                        client_dataloaders = self._initialize_experiment(cost)
                        if client_dataloaders: # Only cache if successful
                             self.dataloader_cache[cache_key] = client_dataloaders
                        else:
                             print(f"  Failed to initialize dataloaders for cost {cost}. Skipping cost for this run.")
                             raise RuntimeError("Dataloader initialization failed.")

                    if not client_dataloaders: # Double check after potential init failure
                        if cost not in results: results[cost] = {}
                        results[cost]['error'] = f"Run {current_run_total}: Dataloader initialization failed."
                        continue

                    # Store actual number of clients *after* initialization
                    self.num_clients_for_run = len(client_dataloaders)
                    if self.num_clients_for_run != num_clients_this_cost:
                         print(f"  Note: Initial client target was {num_clients_this_cost}, but actual clients with data is {self.num_clients_for_run}.")
                         cost_client_counts[cost] = self.num_clients_for_run # Update with actual count

                except Exception as e:
                    print(f"ERROR during data initialization or client count for cost {cost}: {e}. Skipping cost for this run.")
                    traceback.print_exc()
                    if cost not in results: results[cost] = {}
                    results[cost]['error'] = f"Run {current_run_total}: Init failed: {e}"
                    continue

                # Determine tuning parameters based on experiment type
                if tuning_type == ExperimentType.LEARNING_RATE:
                     param_key, fixed_key, fixed_val_func, try_vals_key, servers_key = \
                         'learning_rate', 'reg_param', get_default_reg, 'learning_rates_try', 'servers_tune_lr'
                elif tuning_type == ExperimentType.REG_PARAM:
                     param_key, fixed_key, fixed_val_func, try_vals_key, servers_key = \
                         'reg_param', 'learning_rate', get_default_lr, 'reg_params_try', 'servers_tune_reg'
                else: raise ValueError(f"Unsupported tuning type: {tuning_type}")

                fixed_val = fixed_val_func(self.config.dataset) # Get fixed value (LR or Reg)
                try_vals = self.default_params.get(try_vals_key, [])
                servers_to_tune = self.default_params.get(servers_key, ALGORITHMS if tuning_type == ExperimentType.LEARNING_RATE else []) # Sensible defaults

                if not try_vals: print(f"Warn: No parameters to try for {tuning_type}. Skipping cost."); continue
                if not servers_to_tune: print(f"Warn: No servers specified for tuning {tuning_type}. Skipping cost."); continue

                # Create list of hyperparameter dictionaries to test
                hyperparams_list = [{param_key: p, fixed_key: fixed_val} for p in try_vals]
                tuning_results_for_cost_run = {} # Results for this specific cost and run

                # Iterate through hyperparameter values
                for hp in hyperparams_list:
                    param_val = hp[param_key]
                    print(f"--- Tuning Param: {param_key}={param_val} ---")
                    # Pass the *already initialized* dataloaders
                    server_metrics = self._hyperparameter_tuning_single(
                        cost=cost,
                        hyperparams=hp,
                        servers_to_tune=servers_to_tune,
                        client_dataloaders=client_dataloaders # MODIFIED: Pass loaders
                    )
                    tuning_results_for_cost_run[param_val] = server_metrics

                # --- Aggregate results for this run ---
                # We need to append the results of *this run* to the overall results dict
                if cost not in results: results[cost] = {}
                for p_val, server_data in tuning_results_for_cost_run.items():
                    if p_val not in results[cost]: results[cost][p_val] = {}
                    for server, metrics_this_run in server_data.items():
                        # Initialize server entry if it doesn't exist for this param value
                        if server not in results[cost][p_val]:
                             results[cost][p_val][server] = {} # Start with empty dict

                        # Append the metrics from this run to the list for that server/param
                        results[cost][p_val][server] = self.results_manager.append_or_create_metric_lists(
                             results[cost][p_val][server], metrics_this_run
                        )

            # --- Save results after each run completes ---
            print(f"--- Completed Tuning Run {current_run_total}/{runs_tune} ---")
            run_meta['cost_specific_client_counts'] = cost_client_counts
            # Use the *actual* number of clients from the first cost processed in the run for metadata
            client_count_for_meta = next(iter(cost_client_counts.values())) if cost_client_counts else num_clients_metadata
            self.results_manager.save_results(results, tuning_type, client_count_for_meta, run_meta)

            # MODIFIED: Clear dataloader cache *after each run* to avoid excessive memory use if runs are long/large
            self.dataloader_cache = {}

        return results

    # MODIFIED: Renamed and takes dataloaders as input
    def _hyperparameter_tuning_single(self, cost: Any, hyperparams: Dict, servers_to_tune: List[str], client_dataloaders: Dict) -> Dict:
        """Internal helper: Runs one hyperparameter setting across specified servers using pre-loaded data."""
        # Data loading is now done outside this function

        tracking = {} # Stores metrics for {server_type: metrics_dict} for this HP setting
        for server_type in servers_to_tune:
            print(f"..... Tuning Server: {server_type} .....")
            lr = hyperparams.get('learning_rate')
            reg = hyperparams.get('reg_param')
            # Pass reg param only if server needs it and reg is not None
            algo_params = {'reg_param': reg} if server_type in ['pfedme', 'ditto', 'fedprox'] and reg is not None else {}
            config = self._create_trainer_config(server_type, lr, algo_params)
            # Override rounds for tuning runs if specified in config
            config.rounds = self.default_params.get('rounds_tune_inner', config.rounds)
            print(f"  Tuning with LR={lr}, Reg={reg}, Rounds={config.rounds}")

            server = None # Ensure server is defined in outer scope
            try:
                # Create server instance
                server = self._create_server_instance(cost, server_type, config, tuning=True)
                # Add clients using the provided dataloaders
                self._add_clients_to_server(server, client_dataloaders)

                if not server.clients:
                     print(f"Warn: No clients added to {server_type} for tuning. Skipping.")
                     tracking[server_type] = {'error': 'No clients added'}
                     continue # Skip to next server type

                # Train and evaluate (validation set is used because tuning=True)
                metrics = self._train_and_evaluate(server, config.rounds, cost, self.base_seed) # base_seed might not be relevant here
                tracking[server_type] = metrics

            except Exception as e:
                msg = f"Error tuning {server_type} with params {hyperparams}: {e}"; print(f"ERROR: {msg}"); traceback.print_exc(); tracking[server_type] = {'error': msg}
            finally:
                # Clean up server instance and GPU memory
                if server: del server
                cleanup_gpu()

        return tracking # Return metrics collected for this hyperparameter setting


    def _run_final_evaluation(self, costs: List[Any]) -> Tuple[Dict, Optional[Dict]]:
        """Runs final evaluation loops using best hyperparameters."""
        results, remaining_costs, completed_runs = self._check_existing_results(costs, ExperimentType.EVALUATION)
        target_runs = self.default_params.get('runs', 1)

        # Load diversity results separately
        diversities, div_meta = self.results_manager.load_results(ExperimentType.DIVERSITY)
        diversities = diversities if diversities is not None else {}
        # Check for errors in diversity results as well
        if div_meta and div_meta.get('contains_errors', False):
            print("Found errors flagged in diversity metrics. Regenerating affected parts.")
            # Decide how to handle: clear all? Clear only error parts? For now, clear all.
            diversities = {}


        if not remaining_costs and completed_runs >= target_runs:
            print("Final evaluation already complete.")
            # Return existing results
            return results if results is not None else {}, diversities

        remaining_runs_count = target_runs - completed_runs
        if remaining_runs_count <= 0:
            if remaining_costs:
                print(f"Target eval runs complete, but costs {remaining_costs} missing/failed. Rerunning these costs for {target_runs} runs.")
                remaining_runs_count = target_runs; completed_runs = 0
                # Clear results for the costs being rerun
                for cost in remaining_costs:
                    if cost in results: del results[cost]
                    if cost in diversities: del diversities[cost] # Clear diversity too
            else:
                 print("Target eval runs already completed (should not happen if check_existing_results is correct).")
                 return results if results is not None else {}, diversities

        print(f"Starting {remaining_runs_count} final evaluation run(s)...")
        results = results if results is not None else {}
        # diversities already loaded and potentially cleared

        num_clients_metadata = None # Store client count for metadata
        skew_type = self._get_skew_type_from_dataset_name(self.config.dataset)

        for run_offset in range(remaining_runs_count):
            current_run_total = completed_runs + run_offset + 1
            current_seed = self.base_seed + current_run_total - 1 # Consistent seeding
            print(f"\n--- Starting Final Evaluation Run {current_run_total}/{target_runs} (Seed: {current_seed}) ---")
            set_seeds(current_seed)
            results_this_run = {} # Accumulate results for this specific run
            diversities_this_run = {} # Accumulate diversity for this specific run
            run_meta = {'run_number': current_run_total, 'seed_used': current_seed}
            if skew_type: run_meta['skew_type'] = skew_type
            cost_client_counts = {} # Track actual clients per cost

            # Determine which costs to run in this iteration
            costs_to_run_this_iter = remaining_costs if completed_runs == 0 else costs # Run all costs if starting over runs, else just remaining failed/missing

            for cost in costs_to_run_this_iter:
                print(f"\n--- Evaluating Cost: {cost} (Run {current_run_total}) ---")
                # trained_servers holds server instances from _final_evaluation_single
                trained_servers: Dict[str, Optional[Server]] = {}
                final_round_num = 0 # Store the number of rounds actually run
                actual_num_clients_this_cost = 0 # Store actual client count

                try:
                    # Execute evaluation for this cost, getting metrics and server instances
                    eval_metrics, final_round_num, trained_servers, actual_num_clients_this_cost = \
                        self._final_evaluation_single(cost, current_seed)

                    # Update metadata trackers
                    cost_client_counts[cost] = actual_num_clients_this_cost
                    if num_clients_metadata is None: num_clients_metadata = actual_num_clients_this_cost


                    # Save models if evaluation didn't explicitly error out at the top level
                    # and if the specific server (e.g., FedAvg) succeeded
                    if 'error' not in eval_metrics:
                        self._save_evaluation_models(
                            trained_servers=trained_servers,
                            num_clients_run=actual_num_clients_this_cost, # Use actual count
                            cost=cost,
                            seed=current_seed
                            )
                    else:
                         print(f"  Skipping model saving for cost {cost} due to top-level eval error: {eval_metrics.get('error')}")

                    # Extract diversity metrics if present (they are nested under 'weight_metrics')
                    if 'weight_metrics' in eval_metrics:
                        # Store diversity metrics separately for this cost/run
                        diversities_this_run[cost] = eval_metrics.pop('weight_metrics')

                    # Store the main evaluation metrics for this cost/run
                    results_this_run[cost] = eval_metrics

                except Exception as e:
                    msg = f"Outer error during final eval cost {cost}, run {current_run_total}: {e}"; print(f"ERROR: {msg}"); traceback.print_exc(); results_this_run[cost] = {'error': msg}
                    # Ensure client count is recorded even on error if possible
                    if actual_num_clients_this_cost > 0: cost_client_counts[cost] = actual_num_clients_this_cost

                finally:
                    # Explicitly delete server instances to free memory
                    for server_instance in trained_servers.values():
                         if server_instance: del server_instance
                    trained_servers = {} # Clear the dictionary
                    cleanup_gpu()

            # --- Aggregate results for the entire run ---
            # Append results from this run to the main results dictionaries
            # The append function handles creating lists for the first run
            results = self.results_manager.append_or_create_metric_lists(results, results_this_run)
            diversities = self.results_manager.append_or_create_metric_lists(diversities, diversities_this_run)

            # --- Save results after each full run ---
            print(f"--- Completed Final Evaluation Run {current_run_total}/{target_runs} ---")
            run_meta['cost_specific_client_counts'] = cost_client_counts
            # Use the *actual* number of clients from the first cost processed in the run for metadata
            client_count_for_meta = next(iter(cost_client_counts.values())) if cost_client_counts else num_clients_metadata
            self.results_manager.save_results(results, ExperimentType.EVALUATION, client_count_for_meta, run_meta)
            # Save diversity results separately
            self.results_manager.save_results(diversities, ExperimentType.DIVERSITY, client_count_for_meta, run_meta)

            # Clear dataloader cache after eval run (though caching is mainly for tuning)
            self.dataloader_cache = {}

        return results, diversities

    # MODIFIED: Renamed from _final_evaluation to clarify it's for a single cost/seed
    def _final_evaluation_single(self, cost: Any, seed: int) -> Tuple[Dict, int, Dict, int]:
        """Internal helper: Evaluates one cost setting across all algorithms for a single run."""
        tracking: Dict[str, Any] = {} # {server_type: metrics_dict}
        weight_metrics_acc: Dict[str, Any] = {} # {metric_name: value_list}
        trained_servers: Dict[str, Optional[Server]] = {} # {server_type: server_instance}
        final_round_num = self.default_params.get('rounds', 100) # Default rounds
        actual_num_clients = 0 # Track actual clients initialized

        # 1. Initialize Data
        try:
            # Use cache if available (unlikely for eval, but possible if run structure changes)
            cache_key = (cost, seed) # Use seed as part of run identifier
            if cache_key in self.dataloader_cache:
                 client_dataloaders = self.dataloader_cache[cache_key]
            else:
                 client_dataloaders = self._initialize_experiment(cost)
                 if client_dataloaders:
                      self.dataloader_cache[cache_key] = client_dataloaders # Cache if successful

            if not client_dataloaders:
                 raise RuntimeError("Initialization returned no client dataloaders.")
            # Get actual client count *after* initialization
            actual_num_clients = len(client_dataloaders)
            self.num_clients_for_run = actual_num_clients # Store for potential use in saving intermediate models
            if actual_num_clients == 0:
                 raise RuntimeError("Initialization resulted in zero clients with data.")

        except Exception as e:
            msg = f"Data initialization failed for cost {cost}, seed {seed}: {e}"; print(f"ERROR: {msg}"); traceback.print_exc();
            # Return error dict, 0 rounds, empty servers, 0 clients
            return {'error': msg}, 0, {}, 0

        # 2. Evaluate each algorithm
        for server_type in ALGORITHMS:
            print(f"..... Evaluating Server: {server_type} for Cost: {cost} (Seed: {seed}) .....")
            server: Optional[Server] = None # Define server in this scope

            # --- Get best hyperparameters or defaults ---
            # Use the ResultsManager associated with the *tuning* experiments
            # Assume LR tuning results exist
            lr_results_manager = ResultsManager(self.results_manager.root_dir, self.config.dataset, ExperimentType.LEARNING_RATE, self.results_manager.filename_target_num_clients)
            best_lr = lr_results_manager.get_best_parameters(ExperimentType.LEARNING_RATE, server_type, cost)

            # Assume Reg Param tuning results exist (only relevant for some algos)
            reg_results_manager = ResultsManager(self.results_manager.root_dir, self.config.dataset, ExperimentType.REG_PARAM, self.results_manager.filename_target_num_clients)
            best_reg = reg_results_manager.get_best_parameters(ExperimentType.REG_PARAM, server_type, cost)

            # Fallback to defaults if tuning results are missing or failed
            if best_lr is None:
                best_lr = get_default_lr(self.config.dataset)
                print(f"  Warn: Using default LR ({best_lr}) for {server_type}, cost {cost}.")
            if best_reg is None and server_type in ['pfedme', 'ditto', 'fedprox']:
                 best_reg = get_default_reg(self.config.dataset)
                 print(f"  Warn: Using default RegParam ({best_reg}) for {server_type}, cost {cost}.")

            # Prepare algorithm-specific parameters
            algo_params = {}
            if server_type in ['pfedme', 'ditto', 'fedprox'] and best_reg is not None:
                algo_params['reg_param'] = best_reg
                print(f"  Using LR={best_lr}, RegParam={algo_params.get('reg_param')}")
            else:
                print(f"  Using LR={best_lr}")


            # --- Create config and server ---
            trainer_config = self._create_trainer_config(server_type, best_lr, algo_params)
            final_round_num = trainer_config.rounds # Use actual configured rounds

            try:
                server = self._create_server_instance(cost, server_type, trainer_config, tuning=False)
                # Save initial model state *before* adding clients (if server is FedAvg)
                # Pass the *actual* number of clients determined from data loading
                self._save_initial_model_state(server, actual_num_clients, cost, seed)

                # Add clients to the server instance
                self._add_clients_to_server(server, client_dataloaders)

                if not server.clients:
                    print(f"Warn: No clients added to {server_type} server instance. Skipping evaluation.")
                    tracking[server_type] = {'error': 'No clients available'}
                    trained_servers[server_type] = None # Ensure no stale server instance
                    continue # Skip to next server type

                # --- Train & Evaluate ---
                # Pass actual client count to train_and_evaluate for potential use
                metrics = self._train_and_evaluate(server, trainer_config.rounds, cost, seed, actual_num_clients)
                tracking[server_type] = metrics
                trained_servers[server_type] = server # Keep instance for model saving

                # Accumulate diversity metrics if calculated by the server
                if hasattr(server, 'diversity_metrics') and server.diversity_metrics:
                    # Ensure structure {metric_name: [value]} for aggregation
                    for div_key, div_value in server.diversity_metrics.items():
                         if div_key not in weight_metrics_acc: weight_metrics_acc[div_key] = []
                         # Assume div_value is the value for this server pair/round
                         weight_metrics_acc[div_key].append(div_value)


            except Exception as e:
                msg = f"Error during evaluation run for {server_type}, cost {cost}: {e}"; print(f"ERROR: {msg}"); traceback.print_exc();
                tracking[server_type] = {'error': msg}
                trained_servers[server_type] = None # Mark as failed
                if server: del server; server = None # Ensure cleanup on error
                cleanup_gpu()

        # Add accumulated diversity metrics under a specific key for this cost
        if weight_metrics_acc:
            tracking['weight_metrics'] = weight_metrics_acc

        # Return collected metrics, rounds run, server instances, and actual client count
        return tracking, final_round_num, trained_servers, actual_num_clients


    # --- Data Initialization Logic ---
    def _initialize_experiment(self, cost: Any) -> Dict[str, Tuple[DataLoader, DataLoader, DataLoader]]:
        """Handles data loading, partitioning, and preprocessing steps."""
        print(f"\n--- Initializing Data for Cost: {cost} ---")
        dataset_name = self.config.dataset
        dataset_config = self.default_params # Use pre-fetched config

        # 1. Validate config and get essential parameters
        try:
            config_params = self._validate_and_prepare_config(dataset_name, dataset_config)
            data_source = config_params['data_source']
            strat = config_params['partitioning_strategy']
            cost_interp = config_params['cost_interpretation']
            source_args = config_params['source_args']
            part_args = config_params['partitioner_args']
            scope = config_params['partition_scope']
            sampling_config = dataset_config.get('sampling_config')
            transform_config = dataset_config.get('transform_config') # Get transform config
        except Exception as e: print(f"Error preparing config: {e}"); return {}

        # 2. Determine target number of clients for this cost
        try:
            num_clients = self._get_final_client_count(dataset_config, cost)
            client_ids = [f'client_{i+1}' for i in range(num_clients)]
            # self.num_clients_for_run = num_clients # Set preliminary target count
            print(f"Target clients for cost {cost}: {num_clients}")
        except Exception as e: print(f"Error determining client count: {e}"); return {}

        # 3. Translate the cost parameter
        try:
            t_cost = translate_cost(cost, cost_interp)
            print(f"Translated cost ({cost_interp}): {t_cost}")
        except Exception as e: print(f"Error translating cost: {e}"); return {}

        # --- 4. Load / Partition Data ---
        client_input_data = {} # Dict to hold data before preprocessing
        preproc_input_type = 'unknown' # How data is structured for preprocessor
        base_data = None # Holds the dataset used for partitioning

        try:
            partitioner_func = PARTITIONING_STRATEGIES.get(strat)
            if partitioner_func is None:
                raise NotImplementedError(f"Partitioning strategy '{strat}' not found.")

            if strat == 'pre_split':
                # Data loaded per client later, partitioning is handled by loading logic
                print(f"Using pre-split strategy. Loading per client via: {data_source}...")
                client_input_data, preproc_input_type = self._prepare_client_data_pre_split(
                     client_ids, data_source, dataset_name, source_args, t_cost, dataset_config
                )
            else:
                # Load base data first
                print(f"Loading base data source: {data_source}...")
                # Pass transform_config to loader if applicable (e.g., torchvision)
                loader_func = DATA_LOADERS.get(data_source)
                if loader_func is None: raise NotImplementedError(f"Data loader '{data_source}' not found.")
                # Check if loader accepts transform_config
                import inspect
                sig = inspect.signature(loader_func)
                loader_kwargs = {'dataset_name': dataset_name, 'data_dir': self.data_dir_root, 'source_args': source_args}
                if 'transform_config' in sig.parameters:
                     loader_kwargs['transform_config'] = transform_config

                source_data = loader_func(**loader_kwargs) # Call loader

                # Determine which part of source_data to partition
                if isinstance(source_data, tuple) and len(source_data) == 2: # Train/Test tuple
                     base_data = source_data[0] if scope == 'train' else ConcatDataset(source_data)
                elif isinstance(source_data, TorchDataset): # Single dataset returned
                     base_data = source_data
                else:
                     raise TypeError(f"Unexpected source data type: {type(source_data)}")

                # Partition the base data indices
                print(f"Partitioning {len(base_data)} samples using: {strat}...")
                partition_args_full = {
                    'dataset': base_data, 'num_clients': num_clients, 'seed': self.base_seed,
                    **t_cost, **part_args, 'sampling_config': sampling_config
                }
                partition_result = partitioner_func(**partition_args_full) # Dict[client_idx, List[int]]

                # Prepare input data for preprocessor (subsets)
                client_input_data, preproc_input_type = self._prepare_client_data_from_indices(
                    client_ids, partition_result, base_data, dataset_name, dataset_config, t_cost
                )

        except Exception as e:
            print(f"Error during data load/partition for cost {cost}: {e}"); traceback.print_exc(); return {}

        if not client_input_data:
            print(f"Warn: No client data loaded/prepared for cost {cost}."); return {}

        # --- 5. Preprocess and Create DataLoaders ---
        try:
            print(f"Preprocessing client data (input type: {preproc_input_type})...");
            preprocessor = DataPreprocessor(dataset_name, dataset_config)
            client_loaders = preprocessor.process_client_data(
                client_input_data,
                preproc_input_type,
                batch_size=self.default_params['batch_size'],
                n_workers=self.default_params.get('n_workers', N_WORKERS),
                base_seed=self.base_seed # Pass seed for splitting
            )
        except Exception as e:
            print(f"Error during data preprocessing/loader creation for cost {cost}: {e}"); traceback.print_exc(); return {}

        print(f"--- Data Initialization Complete for Cost: {cost} ({len(client_loaders)} clients) ---");
        return client_loaders


    # --- Data Initialization Helpers ---
    def _validate_and_prepare_config(self, dataset_name: str, dataset_config: Dict) -> Dict:
        """Validates dataset config and extracts pipeline-relevant keys."""
        validate_dataset_config(dataset_config, dataset_name) # Use helper validation
        # Extract keys needed for the initialization pipeline
        return {
            'data_source': dataset_config['data_source'],
            'partitioning_strategy': dataset_config['partitioning_strategy'],
            'cost_interpretation': dataset_config['cost_interpretation'],
            'source_args': dataset_config.get('source_args', {}),
            'partitioner_args': dataset_config.get('partitioner_args', {}),
            'partition_scope': dataset_config.get('partition_scope', 'train'),
            'dataset_class': dataset_config['dataset_class'], # Ensure this is present
            'needs_preprocessing': dataset_config.get('needs_preprocessing', []),
            'transform_config': dataset_config.get('transform_config') # Pass along transform config
        }


    def _get_final_client_count(self, dataset_config: Dict, cost: Any) -> int:
        """Determines the effective number of clients for a given cost/config."""
        cli_override = self.config.num_clients
        num_clients = cli_override if cli_override is not None else dataset_config.get('default_num_clients', 2)

        if not isinstance(num_clients, int) or num_clients <= 0:
             raise TypeError(f"Client count must be a positive integer, got {num_clients}.")

        max_clients = dataset_config.get('max_clients')
        if max_clients is not None and num_clients > max_clients:
            print(f"Warning: Requested {num_clients} clients > max configured ({max_clients}). Using {max_clients}.")
            num_clients = max_clients

        # Adjust client count based on site mappings if using pre_split
        partitioning_strategy = dataset_config.get('partitioning_strategy')
        if partitioning_strategy == 'pre_split':
            site_mappings = dataset_config.get('source_args', {}).get('site_mappings')
            if site_mappings:
                # Translate the cost to get the key needed for the mapping
                cost_interpretation = dataset_config.get('cost_interpretation')
                translated_cost = translate_cost(cost, cost_interpretation)
                # Extract the relevant key ('key', 'suffix', etc.) or handle 'all'
                cost_key = translated_cost.get('key') # Assuming site maps use 'key' primarily
                is_all = translated_cost.get('cost_key_is_all', False)

                if is_all:
                     # If cost is 'all', the number of clients is the total number of sites defined
                     # across all keys in the mapping (assuming each client gets one list entry)
                     total_site_entries = sum(len(v) for v in site_mappings.values())
                     available_for_cost = total_site_entries # Or maybe len(unique sites)? Depends on definition. Assume entries.
                     print(f"Cost key is 'all'. Total available site entries: {available_for_cost}.")
                elif cost_key is not None and cost_key in site_mappings:
                     # The number of entries in the list for this cost key determines available clients
                     available_for_cost = len(site_mappings[cost_key])
                else:
                     # Cost key might not be in mapping, or not applicable
                     available_for_cost = num_clients # No adjustment needed based on sites
                     print(f"Warn: Cost key '{cost_key}' not found in site_mappings for {self.config.dataset}. Cannot adjust client count based on available sites.")


                if num_clients > available_for_cost:
                    print(f"Warning: Requested client count {num_clients} > available sites/entries "
                          f"({available_for_cost}) for cost key '{cost_key}'. "
                          f"Using {available_for_cost}.")
                    num_clients = available_for_cost
                elif is_all and cli_override is None:
                     # If cost is 'all' and user didn't specify -nc, use the total available sites
                     print(f"Cost is 'all', using total available site entries: {available_for_cost} as client count.")
                     num_clients = available_for_cost


        # Final check for validity
        if not isinstance(num_clients, int) or num_clients <= 0:
            raise ValueError(f"Calculated an invalid number of clients: {num_clients}")

        return num_clients

    # REMOVED: _load_source_data (Logic moved into _initialize_experiment)
    # REMOVED: _partition_data (Logic moved into _initialize_experiment)
    # REMOVED: _load_client_specific_data (Logic moved into _prepare_client_data_pre_split)

    # --- NEW: Helper to prepare client data for PRE_SPLIT strategy ---
    def _prepare_client_data_pre_split(self,
                                       client_ids_list: List[str],
                                       data_source_name: str,
                                       dataset_name: str,
                                       source_args: Dict,
                                       translated_cost: Dict,
                                       dataset_config: Dict
                                       ) -> Tuple[Dict[str, Any], str]:
        """Loads data for each client and determines preprocessor input type."""
        client_input_data: Dict[str, Any] = {}
        preprocessor_input_type: str = 'unknown'

        loader_func = DATA_LOADERS.get(data_source_name)
        if loader_func is None:
            raise NotImplementedError(f"Data loader '{data_source_name}' not found.")

        # Determine the cost argument needed by the loader based on interpretation
        cost_interpretation = dataset_config['cost_interpretation']
        cost_arg_for_loader = translated_cost # Pass the whole dict

        for i, client_id in enumerate(client_ids_list):
            client_num = i + 1 # Client number often 1-based for loading

            try:
                # Load data specifically for this client using the loader function
                # The loader function needs access to client_num and cost_arg_for_loader
                loaded_data = loader_func(
                    dataset_name=dataset_name,
                    data_dir=self.data_dir_root,
                    client_num=client_num,
                    cost_key=cost_arg_for_loader, # Pass translated cost dict
                    config=dataset_config # Pass full config
                )

                # Expect loader to return (X, y) tuple or handle errors
                if isinstance(loaded_data, tuple) and len(loaded_data) == 2:
                     X, y = loaded_data
                     # Check if loader returned empty data
                     data_len = len(X) if hasattr(X, '__len__') else 0
                     if data_len == 0:
                          print(f"Warning: No data loaded for client {client_id} (loader returned empty). Skipping.")
                          continue # Skip this client

                     # Store data in dictionary format for the preprocessor
                     client_input_data[client_id] = {'X': X, 'y': y}

                     # Determine input type for preprocessor based on first valid client
                     if preprocessor_input_type == 'unknown':
                          if isinstance(X, np.ndarray) and isinstance(y, np.ndarray):
                               preprocessor_input_type = 'xy_dict'
                          elif isinstance(X, (list, np.ndarray)) and len(X) > 0 and isinstance(X[0], str):
                               preprocessor_input_type = 'path_dict' # Assumes X is list/array of paths
                          else:
                               # Fallback or error if type is ambiguous
                               print(f"Warning: Could not determine preprocessor input type from data types: "
                                     f"X={type(X)}, y={type(y)}")
                               preprocessor_input_type = 'unknown' # Mark as unknown
                else:
                     # Loader function returned unexpected format or an error signal (e.g., None)
                     print(f"Warning: Client data loader {data_source_name} returned unexpected format "
                           f"for client {client_id}: {type(loaded_data)}. Skipping client.")
                     continue

            except FileNotFoundError as e:
                 print(f"Skipping client {client_id} due to FileNotFoundError: {e}")
                 continue
            except Exception as e:
                 print(f"Skipping client {client_id} due to error during data loading: {e}")
                 traceback.print_exc()
                 continue

        # Final check after attempting to load all clients
        if not client_input_data:
            print(f"Warning: Failed to load data for ANY pre-split clients for cost {translated_cost}.")
            return {}, 'unknown'

        if preprocessor_input_type == 'unknown':
            # This might happen if the first client failed but others succeeded with ambiguous types
            raise RuntimeError("Could not determine preprocessor input type for pre-split data.")

        return client_input_data, preprocessor_input_type


    # --- NEW: Helper to prepare client data from partitioned indices ---
    def _prepare_client_data_from_indices(self,
                                          client_ids_list: List[str],
                                          partition_result: Optional[Dict[int, List[int]]],
                                          base_data: TorchDataset,
                                          dataset_name: str,
                                          dataset_config: Dict,
                                          translated_cost: Dict
                                          ) -> Tuple[Dict[str, Any], str]:
        """Creates Subset objects for clients based on partitioned indices."""
        client_input_data: Dict[str, Any] = {}
        preprocessor_input_type = 'subset' # This strategy always results in subsets

        if partition_result is None:
            raise ValueError("Partition result is None for index-based partitioning strategy.")

        dataset_class_name = dataset_config.get('dataset_class')
        is_concept_skew = dataset_class_name == 'SyntheticConceptDataset'

        for i, client_id in enumerate(client_ids_list):
            indices = partition_result.get(i, []) # Get indices for client i (0-based)
            if not indices:
                print(f"Warning: Client {client_id} has no data after partitioning.")
                # Create an empty subset to signify no data for this client
                client_input_data[client_id] = Subset(base_data, [])
            else:
                # Create the subset
                subset = Subset(base_data, indices)
                # If it's concept skew, attach the translated cost needed by the Dataset wrapper
                if is_concept_skew:
                     # Attach cost parameters directly to the subset object for later retrieval
                     subset.translated_cost = translated_cost
                client_input_data[client_id] = subset

        return client_input_data, preprocessor_input_type


    # --- Model, Server, and Training Helpers ---
    def _create_trainer_config(self, server_type: str, learning_rate: float, algorithm_params: Optional[Dict] = None) -> TrainerConfig:
        """Creates the TrainerConfig dataclass for server initialization."""
        algorithm_params = algorithm_params if algorithm_params is not None else {}

        # Determine if personalized models are needed based on server type
        requires_personal_model = server_type in ['pfedme', 'ditto'] # Add other personalized algos here

        return TrainerConfig(
            dataset_name=self.config.dataset,
            device=DEVICE,
            learning_rate=learning_rate,
            batch_size=self.default_params['batch_size'],
            epochs=self.default_params.get('epochs_per_round', 1), # Default to 1 epoch if not specified
            rounds=self.default_params['rounds'],
            requires_personal_model=requires_personal_model,
            algorithm_params=algorithm_params
        )

    def _create_model(self, cost: Any, learning_rate: float) -> Tuple[nn.Module, Union[nn.Module, Callable], torch.optim.Optimizer]:
        """Creates model instance, criterion, and optimizer."""
        model_name = self.config.dataset # Assumes model name matches dataset name
        fixed_classes = self.default_params.get('fixed_classes')

        # Check if the model class exists in the models module (ms)
        if not hasattr(ms, model_name):
             raise ValueError(f"Model class '{model_name}' not found in models.py.")
        model_class = getattr(ms, model_name)

        # Instantiate the model
        try:
             # Inspect signature or make this more config-driven if models vary significantly
             model_args = {}
             if fixed_classes is not None:
                  # TODO: Make this list configurable or inspect __init__ properly
                  if model_name in ['EMNIST', 'CIFAR', 'ISIC', 'Credit', 'Heart', 'Synthetic']: # Add models needing num_classes
                      model_args['num_classes'] = fixed_classes
             # Add other potential model args from config if needed
             model = model_class(**model_args)
        except TypeError as e:
             print(f"Error instantiating model {model_name} with args {model_args}. Check __init__ signature. Error: {e}")
             raise

        # Define criterion based on dataset type/metric
        # Use standard losses where possible, map special cases
        criterion: Union[nn.Module, Callable] # Type hint
        metric_name = self.default_params.get('metric', 'Accuracy').upper() # Default to accuracy

        if metric_name == 'DICE':
             # Ensure get_dice_score is imported or defined
             criterion = get_dice_score
             # TODO: Add foreground channel selection to criterion call if needed
             # criterion = lambda out, tgt: get_dice_score(out, tgt, foreground_channel=1)
        elif self.config.dataset in ['Synthetic_Label', 'Synthetic_Feature', 'Synthetic_Concept',
                                       'Credit', 'EMNIST', 'CIFAR', 'ISIC', 'Heart']:
              # Assume CrossEntropy for classification tasks unless metric dictates otherwise
              # Check if output is multi-class or binary
              if fixed_classes and fixed_classes > 2:
                  criterion = nn.CrossEntropyLoss()
              elif fixed_classes and fixed_classes == 2:
                   # Could use BCEWithLogitsLoss if model output is single logit
                   # Assuming model outputs 2 logits for binary classification for now
                   criterion = nn.CrossEntropyLoss()
              else: # Regression or other cases
                   raise NotImplementedError(f"Criterion mapping not defined for dataset {self.config.dataset} with fixed_classes={fixed_classes}")
        else:
            raise ValueError(f"Criterion not defined for dataset {self.config.dataset} in pipeline._create_model")

        # Create optimizer
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            amsgrad=True,
            weight_decay=self.default_params.get('weight_decay', 1e-4) # Add configurable weight decay
        )
        return model.to(DEVICE), criterion, optimizer # Move model to device here


    def _create_server_instance(self, cost: Any, server_type: str, config: TrainerConfig, tuning: bool) -> Server:
        """Creates and configures a server instance."""
        model, criterion, optimizer = self._create_model(cost, config.learning_rate)
        # Model is already moved to device in _create_model

        # Create the initial global model state
        globalmodelstate = ModelState(
            model=model, # Model is on DEVICE
            optimizer=optimizer,
            criterion=criterion
        )

        # Map server type string to server class
        server_mapping: Dict[str, type[Server]] = {
            'local': Server, # Use base Server for local training
            'fedavg': FedAvgServer
            # Add mappings for other implemented servers like FedProx, Ditto
            # 'fedprox': FedProxServer,
            # 'ditto': DittoServer,
        }
        if server_type not in server_mapping:
            raise ValueError(f"Unsupported server type specified: {server_type}")

        server_class = server_mapping[server_type]

        # Instantiate the server
        server = server_class(config=config, globalmodelstate=globalmodelstate)
        server.set_server_type(server_type, tuning) # Inform server if it's a tuning run

        print(f"Created server: {server_type}, Tuning: {tuning}, Device: {server.device}")
        return server

    def _add_clients_to_server(self, server: Server, client_dataloaders: Dict):
        """Adds client data to the server instance."""
        initial_client_count = len(client_dataloaders)
        print(f"Attempting to add {initial_client_count} clients to {server.server_type} server...")
        added_count = 0
        skipped_count = 0
        clients_to_remove = [] # Keep track of clients that fail validation

        for client_id, loaders in client_dataloaders.items():
             # Ensure loaders is a tuple/list of (train, val, test)
             if not isinstance(loaders, (tuple, list)) or len(loaders) != 3:
                  print(f"Warning: Invalid loader format for client {client_id}. Skipping.")
                  skipped_count += 1
                  clients_to_remove.append(client_id)
                  continue

             train_loader, val_loader, test_loader = loaders

             # Skip clients if they have no training data (check dataset within loader)
             # Also check if test_loader is missing for evaluation runs
             if train_loader is None or not hasattr(train_loader, 'dataset') or train_loader.dataset is None or len(train_loader.dataset) == 0:
                 print(f"Skipping client {client_id}: no training data (length {len(train_loader.dataset) if hasattr(train_loader, 'dataset') and train_loader.dataset is not None else 'N/A'}).")
                 skipped_count += 1
                 clients_to_remove.append(client_id)
                 continue
             if not server.tuning and (test_loader is None or not hasattr(test_loader, 'dataset') or test_loader.dataset is None or len(test_loader.dataset) == 0):
                  print(f"Skipping client {client_id} during evaluation: no test data.")
                  skipped_count += 1
                  clients_to_remove.append(client_id)
                  continue
             # Validation loader is optional

             try:
                 # Create SiteData object (weight calculated later)
                 clientdata = SiteData(
                      site_id=client_id,
                      train_loader=train_loader,
                      val_loader=val_loader,
                      test_loader=test_loader
                      # Weight calculation moved to server.add_client or server.train_round
                 )
                 # Add client to the server (server might do further validation/setup)
                 server.add_client(clientdata=clientdata)
                 added_count += 1
             except Exception as e:
                  print(f"ERROR adding client {client_id} to server: {e}")
                  traceback.print_exc()
                  skipped_count += 1
                  clients_to_remove.append(client_id)


        # MODIFIED: Update num_clients_for_run *after* adding clients and skipping invalid ones
        self.num_clients_for_run = added_count
        print(f"Successfully added {added_count} clients. Skipped {skipped_count}. Actual clients on server: {len(server.clients)}")

        # Clean up loaders for skipped clients if they were cached (might not be necessary if cache cleared per run)
        for client_id in clients_to_remove:
             # Logic to remove from cache if needed, depends on cache implementation
             pass


    # REMOVED: _create_site_data (logic moved to SiteData or _add_clients_to_server)


    def _train_and_evaluate(self, server: Server, rounds: int, cost: Any, seed: int, num_clients_in_run: Optional[int] = None) -> Dict:
        """
        Runs the training and evaluation loop for a given server instance.

        Args:
            server: The server instance (e.g., FedAvgServer).
            rounds: The number of communication rounds to run.
            cost: The current cost parameter (for context).
            seed: The current run seed (for context).
            num_clients_in_run: The actual number of clients participating in this run.

        Returns:
            A dictionary containing the collected metrics, using MetricKey constants.
        """
        run_type = '(tuning)' if server.tuning else '(eval)'
        print(f"Starting training {run_type} for {server.server_type} over {rounds} rounds...")

        # Use actual client count if provided, otherwise get from server
        actual_clients = num_clients_in_run if num_clients_in_run is not None else len(server.clients)
        if actual_clients == 0:
             print("Error: Cannot train and evaluate with 0 clients.")
             return {'error': 'No clients for training/evaluation'}

        for round_num in range(rounds):
            # Perform one round of training and validation
            try:
                # Server internally handles training, aggregation, and validation logging
                server.train_round()
            except Exception as e:
                 print(f"ERROR during server.train_round() in round {round_num+1}: {e}")
                 traceback.print_exc()
                 # Record error in server state or metrics dict?
                 # Let server handle internal state, just report error if needed.
                 return {'error': f'Error in train_round {round_num+1}: {e}'}


            # --- Save intermediate model state (e.g., after round 1 for FedAvg) ---
            # Use self.num_clients_for_run which should be set correctly now
            if not server.tuning and server.server_type == 'fedavg' and round_num == 0:
                self._save_intermediate_model_state(server, cost, seed, model_type='round1')

            # --- Optional progress print ---
            # Access logged metrics from server state (use helper if possible)
            train_loss = server.get_latest_metric(MetricKey.TRAIN_LOSSES)
            val_loss = server.get_latest_metric(MetricKey.VAL_LOSSES)
            val_score = server.get_latest_metric(MetricKey.VAL_SCORES)
            print(f"  R {round_num+1}/{rounds} - "
                  f"TrainLoss: {train_loss:.4f}, "
                  f"ValLoss: {val_loss:.4f}, "
                  f"ValScore: {val_score:.4f}")


        # --- Final Evaluation on Test Set (after all rounds) ---
        if not server.tuning:
            print(f"Performing final test evaluation for {server.server_type}...")
            try:
                 # Server handles testing on clients' test loaders and logs internally
                 server.test_global()
                 test_loss = server.get_latest_metric(MetricKey.TEST_LOSSES)
                 test_score = server.get_latest_metric(MetricKey.TEST_SCORES)
                 print(f"  Final Test Eval - TestLoss: {test_loss:.4f}, TestScore: {test_score:.4f}")
            except Exception as e:
                 print(f"ERROR during server.test_global(): {e}")
                 traceback.print_exc()
                 # Optionally add error to results
                 # metrics['error'] = f'Error in test_global: {e}'


        # --- Collect Final Metrics ---
        # Metrics are stored within the server's state objects during training/testing
        # We just need to retrieve the full history.
        metrics = server.get_all_metrics() # Assume server has a method to return this

        # Report final score based on the run type (validation or test)
        score_key = MetricKey.VAL_SCORES if server.tuning else MetricKey.TEST_SCORES
        final_global_score = server.get_latest_metric(score_key) # Get last recorded score
        score_type = 'Validation' if server.tuning else 'Test'
        print(f"Finished {server.server_type} {run_type}. Final {score_type} Score: {final_global_score:.4f}")

        return metrics


    # --- Model Saving Helpers ---
    def _save_initial_model_state(self, server: Server, num_clients_run: int, cost: Any, seed: int):
         """Saves the initial model state (before training) for FedAvg during evaluation runs."""
         # Only save for FedAvg during evaluation runs, before training starts
         if not server.tuning and server.server_type == 'fedavg':
              if num_clients_run <= 0: # Check actual client count
                   print("  Skipping INITIAL model save: No clients in run.")
                   return
              print(f"  Saving INITIAL FedAvg model state ({num_clients_run} clients)...")
              self.results_manager.save_model_state(
                  model_state_dict=server.serverstate.model.state_dict(),
                  experiment_type=ExperimentType.EVALUATION, # Saving during evaluation phase
                  dataset=self.config.dataset,
                  num_clients_run=num_clients_run, # Use actual count
                  cost=cost,
                  seed=seed,
                  server_type='fedavg',
                  model_type='initial'
              )

    def _save_intermediate_model_state(self, server: Server, cost: Any, seed: int, model_type: str):
        """Saves intermediate model states during evaluation (e.g., after round 1)."""
        # Use the actual client count stored in self.num_clients_for_run
        if self.num_clients_for_run is None or self.num_clients_for_run <= 0:
             print(f"Warning: num_clients_for_run ({self.num_clients_for_run}) not valid. Cannot save intermediate model '{model_type}'.")
             return

        print(f"  Saving FedAvg model state after {model_type} ({self.num_clients_for_run} clients)...")
        self.results_manager.save_model_state(
            model_state_dict=server.serverstate.model.state_dict(),
            experiment_type=ExperimentType.EVALUATION,
            dataset=self.config.dataset,
            num_clients_run=self.num_clients_for_run, # Use actual client count for this run
            cost=cost,
            seed=seed,
            server_type='fedavg',
            model_type=model_type
        )

    def _save_evaluation_models(self,
                                trained_servers: Dict[str, Optional[Server]],
                                num_clients_run: int,
                                cost: Any,
                                seed: int):
        """Saves final and best performing models after a final evaluation run."""
        fedavg_server = trained_servers.get('fedavg')

        if num_clients_run <= 0:
             print(f"  Skipping model saving: Invalid number of clients ({num_clients_run}).")
             return

        if fedavg_server is not None:
            print(f"  Saving FINAL/BEST FedAvg models for cost {cost}, seed {seed} ({num_clients_run} clients)...")

            # Save Final Model (state at the end of training)
            final_model_state = getattr(fedavg_server.serverstate.model, 'state_dict', lambda: None)()
            if final_model_state:
                 self.results_manager.save_model_state(
                     model_state_dict=final_model_state,
                     experiment_type=ExperimentType.EVALUATION,
                     dataset=self.config.dataset,
                     num_clients_run=num_clients_run, # Use actual count
                     cost=cost,
                     seed=seed,
                     server_type='fedavg',
                     model_type='final'
                 )
            else:
                 print("  Warning: Could not get state_dict for final FedAvg model.")


            # Save Best Performing Model (based on validation score during training)
            if hasattr(fedavg_server.serverstate, 'best_model') and fedavg_server.serverstate.best_model:
                best_model_state = getattr(fedavg_server.serverstate.best_model, 'state_dict', lambda: None)()
                if best_model_state:
                     self.results_manager.save_model_state(
                         model_state_dict=best_model_state,
                         experiment_type=ExperimentType.EVALUATION,
                         dataset=self.config.dataset,
                         num_clients_run=num_clients_run, # Use actual count
                         cost=cost,
                         seed=seed,
                         server_type='fedavg',
                         model_type='best'
                     )
                else:
                     print("  Warning: Could not get state_dict for best FedAvg model.")
            else:
                print("  Note: FedAvg server did not record a best_model state, not saved.")
        elif 'fedavg' in trained_servers: # Check if FedAvg ran but maybe failed (server is None)
             print("  Warning: FedAvg server instance not available for model saving (likely due to an earlier error).")
        # else: FedAvg was not part of the ALGORITHMS list for this run