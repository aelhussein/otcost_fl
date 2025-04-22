"""
Manages loading and saving experiment results and model states,
ensuring consistent file naming and directory structures.
"""
import os
import pickle
import traceback
from datetime import datetime
from typing import Dict, Any, Tuple, Optional, List, Union
import numpy as np
import torch  # Needed for torch.save

# Import MetricKey
from helper import MetricKey

# Define Experiment types locally for modularity
class ExperimentType:
    """Enum-like class for experiment types."""
    LEARNING_RATE = 'learning_rate'
    REG_PARAM = 'reg_param'
    EVALUATION = 'evaluation'
    DIVERSITY = 'diversity'

class ResultsManager:
    """Handles saving and loading of results and model checkpoints."""

    def __init__(self, root_dir: str, dataset: str, experiment_type: str, num_clients: int):
        """
        Initializes the ResultsManager.

        Args:
            root_dir (str): Root directory for the project ('results' and
                            'saved_models' will be subdirs).
            dataset (str): Name of the dataset (e.g., 'CIFAR', 'Synthetic').
            experiment_type (str): The type of experiment being run
                                   (e.g., ExperimentType.EVALUATION).
            num_clients (int): The target number of clients configured for the
                               experiment run (used for base filename structure).
        """
        # Create directories if they don't exist
        self._create_directories(root_dir)

        self.root_dir = root_dir
        self.dataset = dataset
        self.experiment_type = experiment_type
        # Store the target client count used for defining filenames
        self.filename_target_num_clients = num_clients

        # Define base directories for results and models
        self.results_base_dir = os.path.join(self.root_dir, 'results')
        self.model_save_base_dir = os.path.join(self.root_dir, 'saved_models')

        # Define structure for results files based on experiment type
        self.results_structure = {
            ExperimentType.LEARNING_RATE: {
                'directory': 'lr_tuning',
                'suffix': 'lr_tuning'
            },
            ExperimentType.REG_PARAM: {
                'directory': 'reg_param_tuning',
                'suffix': 'reg_tuning'
            },
            ExperimentType.EVALUATION: {
                'directory': 'evaluation',
                'suffix': 'evaluation'
            },
            ExperimentType.DIVERSITY: {
                'directory': 'diversity',
                'suffix': 'diversity'
            }
        }

        # Pre-compute filename templates
        for key, info in self.results_structure.items():
            info['filename_template'] = f'{self.dataset}_{self.filename_target_num_clients}clients_{info["suffix"]}.pkl'

    def _create_directories(self, root_dir: str) -> None:
        """
        Creates necessary directories for storing results and models.
        
        Args:
            root_dir: Root directory path
        """
        if not os.path.isdir(root_dir):
            print(f"Warning: Root directory '{root_dir}' does not exist. Creating it.")
            try:
                os.makedirs(root_dir, exist_ok=True)
                os.makedirs(os.path.join(root_dir, 'results'), exist_ok=True)
                os.makedirs(os.path.join(root_dir, 'saved_models'), exist_ok=True)
            except OSError as e:
                print(f"ERROR: Could not create root directory structure: {e}")
                raise

    def _get_results_path(self, experiment_type: str) -> str:
        """
        Constructs the full path for saving/loading metrics results pickle files.

        Args:
            experiment_type: The type of experiment (must be a key in results_structure).

        Returns:
            The full path to the results file.

        Raises:
            ValueError: If the experiment_type is unknown.
        """
        experiment_info = self.results_structure.get(experiment_type)
        if not experiment_info:
            raise ValueError(f"Unknown experiment type for results path: {experiment_type}")

        # Construct path: <root>/results/<exp_dir>/<filename_template>
        results_dir = os.path.join(self.results_base_dir, experiment_info['directory'])
        return os.path.join(results_dir, experiment_info['filename_template'])

    def _get_model_save_path(self,
                             experiment_type: str,
                             dataset: str,
                             num_clients_run: int,  # Actual clients used in the run
                             cost: Any,
                             seed: int,
                             server_type: str,
                             model_type: str  # e.g., 'initial', 'final', 'best', 'round1'
                            ) -> str:
        """
        Constructs the standardized path for saving model state dictionaries.

        Args:
            experiment_type: e.g., ExperimentType.EVALUATION.
            dataset: Dataset name.
            num_clients_run: Actual number of clients used in this specific run.
            cost: Cost parameter used for the run.
            seed: Seed used for the run.
            server_type: Server algorithm type (e.g., 'fedavg').
            model_type: Identifier for the model state (e.g., 'final', 'best').

        Returns:
            The full path for the model file.

        Raises:
            ValueError: If trying to save model for non-evaluation experiments.
        """
        # Currently, only saving models from final evaluation runs is implemented
        if experiment_type != ExperimentType.EVALUATION:
            raise ValueError("Model saving is currently only implemented for EVALUATION runs.")

        # Format cost parameter for use in filename
        if isinstance(cost, (int, float)):
            cost_str = f"{float(cost):.4f}"  # Format numbers consistently
        else:
            # Replace characters unsuitable for filenames (like '/', '\')
            cost_str = str(cost).replace('/', '_').replace('\\', '_').replace(' ', '')

        # Define directory structure: <root>/saved_models/<dataset>/evaluation/
        model_dir = os.path.join(self.model_save_base_dir, dataset, 'evaluation')

        # Define filename structure
        filename = (f"{dataset}_{num_clients_run}clients_"
                   f"cost_{cost_str}_seed_{seed}_"
                   f"{server_type}_{model_type}_model.pt")

        return os.path.join(model_dir, filename)

    def save_model_state(self,
                         model_state_dict: Optional[Dict],  # Model state can be None
                         experiment_type: str,
                         dataset: str,
                         num_clients_run: int,
                         cost: Any,
                         seed: int,
                         server_type: str,
                         model_type: str):
        """
        Saves a model's state_dict to a standardized path. Creates directories if needed.

        Args:
            model_state_dict: The state dictionary (`model.state_dict()`) to save.
                              If None, a warning is printed and nothing is saved.
            (See _get_model_save_path for other args)
            num_clients_run: Actual number of clients used in the run (crucial for filename).
        """
        # Validate inputs
        if model_state_dict is None:
            print(f"Warning: Attempted to save a None model state dict for '{model_type}' model. Skipping.")
            return
        if num_clients_run <= 0:
            print(f"Warning: Invalid num_clients_run ({num_clients_run}) provided for model saving. Skipping.")
            return

        # Get the standardized path
        try:
            path = self._get_model_save_path(
                experiment_type, dataset, num_clients_run, cost, seed,
                server_type, model_type
            )
            print(f"  Attempting to save {model_type} model state to: {path}")

            # Ensure the directory exists
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Save the state dictionary
            torch.save(model_state_dict, path)
            print(f"  Successfully saved {model_type} model state.")
        except Exception as e:
            print(f"ERROR: Failed to save {model_type} model state: {e}")
            traceback.print_exc()  # Print full traceback for debugging

    def load_results(self, experiment_type: str) -> Tuple[Optional[Dict], Optional[Dict]]:
        """
        Loads results from a pickle file. Handles both new (dict with 'results'
        and 'metadata') and potentially older formats.

        Args:
            experiment_type: The type of experiment results to load.

        Returns:
            A tuple containing:
            - Optional[Dict]: The loaded results dictionary (or None if loading fails).
            - Optional[Dict]: The loaded metadata dictionary (or None if not present/loading fails).
        """
        path = self._get_results_path(experiment_type)
        if not os.path.exists(path):
            print(f"Results file not found at: {path}")
            return None, None

        try:
            with open(path, 'rb') as f:
                loaded_data = pickle.load(f)

            # Check for the new format with explicit keys
            if isinstance(loaded_data, dict) and 'results' in loaded_data and 'metadata' in loaded_data:
                return loaded_data['results'], loaded_data['metadata']
            else:
                # Assume old format (the loaded data *is* the results dict)
                print(f"Loaded results from {path} (assuming old format without metadata).")
                return loaded_data, None  # Return None for metadata
        except (pickle.UnpicklingError, EOFError) as e:
            print(f"Error unpickling results from {path}: {e}")
            return None, None
        except Exception as e:
            print(f"Unexpected error loading results from {path}: {e}")
            traceback.print_exc()
            return None, None

    def save_results(self,
                     results: Dict,
                     experiment_type: str,
                     client_count: Optional[int] = None,  # Actual client count used
                     run_metadata: Optional[Dict] = None):
        """
        Saves experiment results along with metadata to a pickle file.
        Includes a flag in metadata indicating if errors were detected in results.

        Args:
            results (Dict): The results dictionary to save.
            experiment_type (str): Type of experiment (e.g., ExperimentType.EVALUATION).
            client_count (Optional[int]): The actual number of clients used in the run
                                          associated with these results.
            run_metadata (Optional[Dict]): Additional metadata specific to the run
                                           (e.g., run number, seed used).
        """
        path = self._get_results_path(experiment_type)
        run_metadata = run_metadata if run_metadata is not None else {}

        # Check if the results structure contains any error flags
        contains_errors = self._check_for_errors_in_results(results)

        # Construct metadata dictionary
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'dataset': self.dataset,
            'experiment_type': experiment_type,
            'client_count_used': client_count,  # Actual number of clients in the run
            'filename_target_clients': self.filename_target_num_clients,  # Target count for filename structure
            'contains_errors': contains_errors,  # Flag to indicate if results have issues
            **run_metadata  # Merge any additional run-specific metadata
        }

        # Combine results and metadata for saving
        final_data_to_save = {'results': results, 'metadata': metadata}

        try:
            # Ensure the results directory exists
            results_dir = os.path.dirname(path)
            os.makedirs(results_dir, exist_ok=True)
            
            # Save the combined dictionary using pickle
            with open(path, 'wb') as f:
                pickle.dump(final_data_to_save, f, protocol=pickle.HIGHEST_PROTOCOL)

            status = "with ERRORS" if contains_errors else "successfully"
            client_info = f"(Actual Clients: {client_count})" if client_count is not None else ""
            print(f"Results saved {status} to {path} {client_info}")
        except Exception as e:
            print(f"Error saving results to {path}: {e}")
            traceback.print_exc()

    def _check_for_errors_in_results(self, data: Optional[Any], depth: int = 0, max_depth: int = 100) -> bool:
        """
        Recursively checks if any dictionary within the data structure
        contains a key named 'error'. Includes depth limit to prevent stack overflow.

        Args:
            data: The data structure (dict, list, or other) to check.
            depth: Current recursion depth.
            max_depth: Maximum recursion depth to prevent stack overflow.

        Returns:
            True if an 'error' key is found at any level, False otherwise.
        """
        # Prevent stack overflow with excessive recursion
        if depth > max_depth:
            print(f"Warning: Maximum recursion depth ({max_depth}) reached in error checking.")
            return False
            
        if data is None:
            return False
            
        if isinstance(data, dict):
            # First check for error key directly
            if 'error' in data:
                return True
                
            # Then check values recursively
            for value in data.values():
                if self._check_for_errors_in_results(value, depth + 1, max_depth):
                    return True
                    
        elif isinstance(data, list):
            # Check list items recursively, handling empty lists
            if not data:
                return False
                
            # Use a more efficient approach for lists
            for item in data:
                if self._check_for_errors_in_results(item, depth + 1, max_depth):
                    return True
                    
        # Other data types can't contain 'error' keys
        return False

    def _safe_get_nested(self, 
                        dictionary: Dict, 
                        keys: List[str], 
                        default: Any = None) -> Any:
        """
        Safely navigate a nested dictionary path.
        
        Args:
            dictionary: The dictionary to navigate
            keys: List of keys defining the path
            default: Value to return if path doesn't exist
            
        Returns:
            The value at the path or default if not found
        """
        if dictionary is None:
            return default
            
        current = dictionary
        for key in keys:
            if not isinstance(current, dict) or key not in current:
                return default
            current = current[key]
            
        return current

    def append_or_create_metric_lists(self,
                                     existing_dict: Optional[Dict],
                                     new_dict: Dict) -> Dict:
        """
        Appends values from `new_dict` to lists in `existing_dict`.
        If a key doesn't exist or isn't a list, it creates a new list.
        Handles nested dictionaries recursively.

        Args:
            existing_dict: The dictionary to append to (can be None).
            new_dict: The dictionary containing new values to append.

        Returns:
            The updated existing_dict.
        """
        # Handle case where existing_dict is None
        if existing_dict is None:
            # Create a new dict with values as lists
            results = {}
            for k, v in new_dict.items():
                if isinstance(v, dict):
                    # Recursively handle nested dictionaries
                    results[k] = self.append_or_create_metric_lists(None, v)
                else:
                    # Wrap non-dict values in a list
                    results[k] = [v]
            return results

        # Merge new_dict into existing_dict
        for key, new_value in new_dict.items():
            if isinstance(new_value, dict):
                # Handle nested dictionaries
                existing_sub_dict = existing_dict.get(key)
                if not isinstance(existing_sub_dict, dict):
                    # Create/replace with a properly nested structure
                    existing_dict[key] = self.append_or_create_metric_lists(None, new_value)
                else:
                    # Merge into existing nested dict
                    existing_dict[key] = self.append_or_create_metric_lists(existing_sub_dict, new_value)
            else:
                # Handle non-dict values
                if key not in existing_dict:
                    # Create new list with this value
                    existing_dict[key] = [new_value]
                else:
                    # Ensure existing value is a list
                    if not isinstance(existing_dict[key], list):
                        print(f"Warning: Converting non-list value for key '{key}' to list: {existing_dict[key]}")
                        existing_dict[key] = [existing_dict[key]]
                    
                    # Append new value
                    existing_dict[key].append(new_value)

        return existing_dict

    def get_best_parameters(self, param_type: str, server_type: str, cost: Any) -> Optional[Any]:
        """
        Finds the best hyperparameter value (LR or Reg Param) based on the lowest
        average median validation loss across multiple tuning runs.

        Args:
            param_type: The type of hyperparameter (ExperimentType.LEARNING_RATE or REG_PARAM).
            server_type: The server algorithm name (e.g., 'fedavg').
            cost: The cost parameter under which tuning was performed.

        Returns:
            The best hyperparameter value found, or None if results are missing,
            incomplete, or invalid.
        """
        results, metadata = self.load_results(param_type)

        # Validate basic results structure
        if results is None:
            print(f"Warning: No results found for {param_type} tuning. Cannot get best params.")
            return None
            
        if cost not in results:
            print(f"Warning: Cost '{cost}' not found in {param_type} tuning results.")
            return None

        cost_results = results[cost]  # Dict: {param_val: {server_type: {metric_dict}}}
        if not isinstance(cost_results, dict):
            print(f"Warning: Invalid results format for cost '{cost}' in {param_type} tuning.")
            return None

        # Extract validation losses for the specified server type
        server_val_losses_by_param: Dict[Any, List[List[float]]] = {}

        for param_val, server_data in cost_results.items():
            # Extract losses, handling different possible key structures
            losses = self._extract_validation_losses(server_data, server_type)
            
            if losses:
                server_val_losses_by_param[param_val] = losses
            else:
                print(f"  Warning: No valid validation losses found for param {param_val}, server {server_type}.")

        if not server_val_losses_by_param:
            print(f"Warning: Could not extract any valid validation losses for server '{server_type}' and cost '{cost}'.")
            return None

        return self._select_best_hyperparameter(server_val_losses_by_param)

    def _extract_validation_losses(self, server_data: Dict, server_type: str) -> List[List[float]]:
        """
        Extracts and validates validation losses from server data structure.
        Handles different key structures and formats.
        
        Args:
            server_data: Dictionary containing server metrics
            server_type: The server algorithm name
            
        Returns:
            List of validated loss lists per run, or empty list if invalid data
        """
        processed_runs_losses = []
        
        # Check if server_type exists in data
        if not isinstance(server_data, dict) or server_type not in server_data:
            return []
            
        # Check for 'global' section
        global_data = server_data.get(server_type, {}).get('global', {})
        if not isinstance(global_data, dict):
            return []
            
        # Try different possible keys for validation losses
        all_runs_losses = None
        
        # First try MetricKey constant
        if MetricKey.VAL_LOSSES in global_data and isinstance(global_data[MetricKey.VAL_LOSSES], list):
            all_runs_losses = global_data[MetricKey.VAL_LOSSES]
        # Then try 'losses' key (from actual pickle structure)
        elif 'losses' in global_data and isinstance(global_data['losses'], list):
            all_runs_losses = global_data['losses']
        
        if not all_runs_losses:
            return []
            
        # Handle nested list structures that might be present
        if len(all_runs_losses) == 1 and isinstance(all_runs_losses[0], list) and all_runs_losses[0] and isinstance(all_runs_losses[0][0], list):
            # Extra layer of nesting detected, unwrap one level
            all_runs_losses = all_runs_losses[0]
        
        # Process each run's losses
        for run_losses in all_runs_losses:
            # Skip invalid data formats
            if not isinstance(run_losses, list):
                continue
                
            # Handle different list structures we might encounter
            scalar_losses = []
            
            if run_losses and isinstance(run_losses[0], list):
                # Format: [[loss1], [loss2], ...]
                scalar_losses = [
                    float(l[0]) for l in run_losses 
                    if isinstance(l, list) and len(l) > 0 
                    and isinstance(l[0], (float, int, np.number)) 
                    and not np.isnan(l[0])
                ]
            elif run_losses and isinstance(run_losses[0], (float, int, np.number)):
                # Format: [loss1, loss2, ...]
                scalar_losses = [
                    float(l) for l in run_losses 
                    if isinstance(l, (float, int, np.number)) 
                    and not np.isnan(l)
                ]
                
            # Add valid losses to processed results
            if scalar_losses:
                processed_runs_losses.append(scalar_losses)
                
        return processed_runs_losses

    def _select_best_hyperparameter(self, param_losses: Dict[Any, List[List[float]]]) -> Optional[Any]:
        """
        Selects the best hyperparameter value based on the lowest average
        median validation loss across runs.

        Args:
            param_losses: Dict mapping {param_value: List[List[float]]}, where
                          the outer list is runs and inner list is losses per round for that run.

        Returns:
            The best hyperparameter value, or None if cannot be determined.
        """
        best_avg_median_loss = float('inf')
        best_param = None

        if not param_losses:
            return None

        for param_val, runs_losses_list in param_losses.items():
            if not runs_losses_list:  # Skip if no valid runs for this param
                print(f"  Skipping param {param_val}: No valid loss runs provided.")
                continue

            # Calculate median loss for each run
            run_median_losses = []
            for run_idx, run_losses in enumerate(runs_losses_list):
                # Validate run losses data
                if not run_losses or not all(isinstance(x, (float, int, np.number)) for x in run_losses):
                    print(f"  Warning: Invalid loss data for param {param_val}, run {run_idx+1}.")
                    continue
                    
                try:
                    # Calculate median loss for this run
                    median_loss_this_run = np.median(run_losses)
                    if not np.isnan(median_loss_this_run):
                        run_median_losses.append(median_loss_this_run)
                    else:
                        print(f"  Warning: Median loss is NaN for param {param_val}, run {run_idx+1}.")
                except Exception as e:
                    print(f"  Error calculating median for param {param_val}, run {run_idx+1}: {e}")

            # Calculate average median loss across runs
            if run_median_losses:
                avg_median_loss = np.mean(run_median_losses)
                print(f"  Param: {param_val}, Avg Median Val Loss: {avg_median_loss:.5f} (from {len(run_median_losses)} runs)")

                # Update best parameter if this is better
                if avg_median_loss < best_avg_median_loss:
                    best_avg_median_loss = avg_median_loss
                    best_param = param_val
            else:
                print(f"  Warning: Could not calculate avg median loss for param {param_val} (no valid runs).")

        if best_param is None:
            print(f"Warning: Could not determine best hyperparameter from options: {list(param_losses.keys())}.")
        else:
            print(f"Best hyperparameter selected: {best_param} (Avg Median Loss: {best_avg_median_loss:.5f})")
            
        return best_param