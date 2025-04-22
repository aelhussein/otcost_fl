"""
Manages loading and saving experiment results and model states,
ensuring consistent file naming and directory structures.
"""
import os
import pickle
import traceback
from datetime import datetime
from typing import Dict, Any, Tuple, Optional
import numpy as np
import torch # Needed for torch.save

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
        if not os.path.isdir(root_dir):
            # Optionally create root_dir or raise error
            print(f"Warning: Root directory '{root_dir}' does not exist.")
            # os.makedirs(root_dir, exist_ok=True) # Or create it

        self.root_dir = root_dir
        self.dataset = dataset
        self.experiment_type = experiment_type
        # Store the target client count used for defining filenames
        self.filename_target_num_clients = num_clients

        # Define base directories for results and models
        self.results_base_dir = os.path.join(self.root_dir, 'results')
        self.model_save_base_dir = os.path.join(self.root_dir, 'saved_models')

        # Define structure for results files based on experiment type
        self.results_structure: Dict[str, Dict[str, str]] = {
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
                             num_clients_run: int, # Actual clients used in the run
                             cost: Any,
                             seed: int,
                             server_type: str,
                             model_type: str # e.g., 'initial', 'final', 'best', 'round1'
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
             cost_str = f"{float(cost):.4f}" # Format numbers consistently
        else:
             # Replace characters unsuitable for filenames (like '/')
             cost_str = str(cost).replace('/', '_').replace('\\', '_')

        # Define directory structure: <root>/saved_models/<dataset>/evaluation/
        model_dir = os.path.join(self.model_save_base_dir, dataset, 'evaluation')

        # Define filename structure: <dataset>_<num_clients>clients_cost_<cost>_seed_<seed>_<server>_<type>_model.pt
        filename = (f"{dataset}_{num_clients_run}clients_"
                    f"cost_{cost_str}_seed_{seed}_"
                    f"{server_type}_{model_type}_model.pt")

        return os.path.join(model_dir, filename)

    def save_model_state(self,
                         model_state_dict: Optional[Dict], # Model state can be None
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
        """
        if model_state_dict is None:
             print(f"Warning: Attempted to save a None model state dict for '{model_type}' model. Skipping.")
             return

        # Get the standardized path
        path = self._get_model_save_path(
            experiment_type, dataset, num_clients_run, cost, seed,
            server_type, model_type
        )
        print(f"  Attempting to save {model_type} model state to: {path}")

        try:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(path), exist_ok=True)
            # Save the state dictionary
            torch.save(model_state_dict, path)
            print(f"  Successfully saved {model_type} model state.")
        except Exception as e:
            print(f"ERROR: Failed to save {model_type} model state to {path}: {e}")
            traceback.print_exc() # Print full traceback for debugging

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
                 # print(f"Loaded results from {path} with metadata: {loaded_data['metadata']}") # Verbose
                 return loaded_data['results'], loaded_data['metadata']
            else:
                 # Assume old format (the loaded data *is* the results dict)
                 # print(f"Loaded results from {path} (assuming old format without metadata).") # Verbose
                 return loaded_data, None
        except (pickle.UnpicklingError, EOFError, Exception) as e:
             # Handle potential errors during file loading/unpickling
             print(f"Error loading or unpickling results from {path}: {e}.")
             return None, None

    def save_results(self,
                     results: Dict,
                     experiment_type: str,
                     client_count: Optional[int] = None, # Actual client count used
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
            'client_count_used': client_count, # Actual number of clients in the run
            'filename_target_clients': self.filename_target_num_clients, # Target count for filename structure
            'contains_errors': contains_errors, # Flag to indicate if results have issues
            **run_metadata # Merge any additional run-specific metadata
        }

        # Combine results and metadata for saving
        final_data_to_save = {'results': results, 'metadata': metadata}

        try:
            # Ensure the results directory exists
            os.makedirs(os.path.dirname(path), exist_ok=True)
            # Save the combined dictionary
            with open(path, 'wb') as f:
                pickle.dump(final_data_to_save, f)

            status = "with ERRORS" if contains_errors else "successfully"
            print(f"Results saved {status} to {path} (Actual Clients: {client_count})")
        except Exception as e:
            print(f"Error saving results to {path}: {e}")
            traceback.print_exc()

    def _check_for_errors_in_results(self, results_dict: Optional[Dict]) -> bool:
        """
        Recursively checks if any dictionary within the results structure
        contains a key named 'error'.

        Args:
            results_dict: The dictionary (or nested dictionary) to check.

        Returns:
            True if an 'error' key is found at any level, False otherwise.
        """
        if results_dict is None:
            return False

        if isinstance(results_dict, dict):
            # Check for direct error key at this level
            if 'error' in results_dict:
                return True

            # Recursively check all values in the dictionary
            for value in results_dict.values():
                if isinstance(value, dict):
                    if self._check_for_errors_in_results(value):
                        return True
                elif isinstance(value, list):
                    # Check items in list if they are dictionaries
                    for item in value:
                        if isinstance(item, dict):
                            if self._check_for_errors_in_results(item):
                                return True
        # No error found at this level or below
        return False

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
        # If no existing dictionary, create a new one where values are lists
        if existing_dict is None:
            # Recursively handle nested dicts
            return {
                k: [v] if not isinstance(v, dict)
                     else self.append_or_create_metric_lists(None, v)
                for k, v in new_dict.items()
            }

        # Iterate through the new dictionary
        for key, new_value in new_dict.items():
            if isinstance(new_value, dict):
                # If the value is a dictionary, recurse
                if key not in existing_dict or not isinstance(existing_dict[key], dict):
                    existing_dict[key] = {} # Initialize if key missing or not dict
                existing_dict[key] = self.append_or_create_metric_lists(
                    existing_dict[key], new_value
                )
            else:
                # If the value is not a dictionary, append it
                if key not in existing_dict:
                    existing_dict[key] = [] # Create list if key doesn't exist
                # Ensure the existing value is a list before appending
                if not isinstance(existing_dict[key], list):
                    # If it's not a list (e.g., leftover from old format), make it a list
                    existing_dict[key] = [existing_dict[key]]
                existing_dict[key].append(new_value)

        return existing_dict
    def get_best_parameters(self, param_type, server_type, cost):
        """Finds the best hyperparameter value based on validation loss."""
        # (Keep function code as provided in the previous pipeline.py)
        results, _ = self.load_results(param_type)
        if results is None or cost not in results: 
            return None
        cost_results = results[cost]
        server_metrics = {}
        if isinstance(cost_results, dict):
            for param_val, server_data in cost_results.items():
                if isinstance(server_data, dict) and server_type in server_data:
                    server_metrics[param_val] = server_data[server_type]
        else: 
            return None
        if not server_metrics: 
            return None
        return self._select_best_hyperparameter(server_metrics)

    def _select_best_hyperparameter(self, param_results):
        """Selects best hyperparameter based on lowest median validation loss."""
        # (Keep function code as provided in the previous pipeline.py)
        best_loss = float('inf')
        best_param = None
        for param_val, metrics in param_results.items():
            if isinstance(metrics, dict) and 'global' in metrics and 'val_losses' in metrics['global'] and metrics['global']['val_losses']:
                run_median_losses = []
                # Handle list of runs (list of lists) vs single run (list)
                if metrics['global']['val_losses'] and isinstance(metrics['global']['val_losses'][0], list): # Multi-run results
                    for run_losses in metrics['global']['val_losses']:
                         # Expects run_losses = [[loss1], [loss2], ...]
                         valid_losses = [l[0] for l in run_losses if l and isinstance(l,list) and len(l)>0]
                         if valid_losses: run_median_losses.append(np.median(valid_losses))
                elif metrics['global']['val_losses']: # Single run results
                     # Expects metrics['global']['val_losses'] = [[loss1], [loss2], ...]
                     valid_losses = [l[0] for l in metrics['global']['val_losses'] if l and isinstance(l,list) and len(l)>0]
                     if valid_losses: run_median_losses.append(np.median(valid_losses))

                if run_median_losses:
                    avg_median_loss = np.mean(run_median_losses)
                    if avg_median_loss < best_loss: best_loss = avg_median_loss; best_param = param_val
        return best_param
