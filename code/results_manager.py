"""
Manages loading and saving experiment results and model states.
"""
import os
import pickle
import traceback
from datetime import datetime
from typing import Dict, Any, Tuple, Optional
import numpy as np
import torch # Needed for torch.save

# Import ExperimentType if defined here, otherwise it will be imported in pipeline.py
# from pipeline import ExperimentType # Or define locally:
class ExperimentType:
    LEARNING_RATE = 'learning_rate'; REG_PARAM = 'reg_param'; EVALUATION = 'evaluation'; DIVERSITY = 'diversity'

class ResultsManager:
    def __init__(self, root_dir, dataset, experiment_type, num_clients: int):
        self.root_dir = root_dir
        self.dataset = dataset
        self.experiment_type = experiment_type
        self.num_clients = num_clients # Target client count for filename structure
        self.results_base_dir = os.path.join(self.root_dir, 'results')
        self.model_save_base_dir = os.path.join(self.root_dir, 'saved_models')

        self.results_structure = {
            ExperimentType.LEARNING_RATE: {'directory': 'lr_tuning', 'suffix': 'lr_tuning'},
            ExperimentType.REG_PARAM: {'directory': 'reg_param_tuning', 'suffix': 'reg_tuning'},
            ExperimentType.EVALUATION: {'directory': 'evaluation', 'suffix': 'evaluation'},
            ExperimentType.DIVERSITY: {'directory': 'diversity', 'suffix': 'diversity'}
        }
        for key, info in self.results_structure.items():
             info['filename_template'] = f'{dataset}_{num_clients}clients_{info["suffix"]}.pkl'

    def _get_results_path(self, experiment_type):
        """Gets the full path for saving/loading metrics results."""
        experiment_info = self.results_structure.get(experiment_type)
        if not experiment_info: raise ValueError(f"Unknown experiment type for results path: {experiment_type}")
        return os.path.join(self.results_base_dir, experiment_info['directory'], experiment_info['filename_template'])

    def _get_model_save_path(self, experiment_type: str, dataset: str, num_clients_run: int, cost: Any, seed: int, server_type: str, model_type: str) -> str:
        """Constructs the standardized path for saving model state dictionaries."""
        if experiment_type != ExperimentType.EVALUATION: raise ValueError("Model saving only implemented for EVALUATION runs.")
        cost_str = f"{float(cost):.4f}" if isinstance(cost, (int, float)) else str(cost).replace('/', '_')
        model_dir = os.path.join(self.model_save_base_dir, dataset, 'evaluation')
        filename = f"{dataset}_{num_clients_run}clients_cost_{cost_str}_seed_{seed}_{server_type}_{model_type}_model.pt"
        return os.path.join(model_dir, filename)

    def save_model_state(self, model_state_dict: Dict, experiment_type: str, dataset: str, num_clients_run: int, cost: Any, seed: int, server_type: str, model_type: str):
        """Saves a model's state_dict."""
        if not model_state_dict: print(f"Warn: Empty model state dict for {model_type}. Skipping save."); return
        path = self._get_model_save_path(experiment_type, dataset, num_clients_run, cost, seed, server_type, model_type)
        print(f"  Attempting to save {model_type} model state to: {path}")
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True); torch.save(model_state_dict, path); print(f"  Successfully saved {model_type} model state.")
        except Exception as e: print(f"ERROR: Failed to save {model_type} model state to {path}: {e}"); traceback.print_exc()

    def load_results(self, experiment_type) -> Tuple[Optional[Dict], Optional[Dict]]:
        """Loads results, returning results dict and metadata dict, or (None, None)."""
        path = self._get_results_path(experiment_type)
        if not os.path.exists(path): return None, None
        try:
            with open(path, 'rb') as f: loaded_data = pickle.load(f)
            if isinstance(loaded_data, dict) and 'results' in loaded_data and 'metadata' in loaded_data:
                return loaded_data['results'], loaded_data['metadata']
            else: return loaded_data, None # Old format
        except Exception as e: print(f"Error loading results from {path}: {e}."); return None, None

    def save_results(self, results, experiment_type, client_count=None, run_metadata=None):
        """Saves results with metadata."""
        path = self._get_results_path(experiment_type); run_metadata = run_metadata or {}
        contains_errors = self._check_for_errors_in_results(results)
        metadata = { 'timestamp': datetime.now().isoformat(), 'dataset': self.dataset, 'experiment_type': experiment_type, 'client_count_used': client_count, 'filename_target_clients': self.num_clients, 'contains_errors': contains_errors, **run_metadata }
        final_data_to_save = {'results': results, 'metadata': metadata}
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'wb') as f: pickle.dump(final_data_to_save, f)
            status = "with ERRORS" if contains_errors else "successfully"; print(f"Results saved {status} to {path} (Actual Clients: {client_count})")
        except Exception as e: print(f"Error saving results to {path}: {e}")

    def _check_for_errors_in_results(self, results_dict):
        """Recursively checks for error flags in results."""
        # (Keep function code as provided in the previous pipeline.py)
        if results_dict is None: return False
        if isinstance(results_dict, dict):
            if 'error' in results_dict: return True
            for key, value in results_dict.items():
                if isinstance(value, dict):
                    if self._check_for_errors_in_results(value): return True
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict) and self._check_for_errors_in_results(item): return True
        return False

    def append_or_create_metric_lists(self, existing_dict, new_dict):
        """Appends new metrics to existing lists or creates new lists."""
        # (Keep function code as provided in the previous pipeline.py)
        if existing_dict is None: return {k: [v] if not isinstance(v, dict) else self.append_or_create_metric_lists(None, v) for k, v in new_dict.items()}
        for key, new_value in new_dict.items():
            if isinstance(new_value, dict):
                if key not in existing_dict or not isinstance(existing_dict[key], dict): existing_dict[key] = {}
                existing_dict[key] = self.append_or_create_metric_lists(existing_dict[key], new_value)
            else:
                if key not in existing_dict: existing_dict[key] = []
                if not isinstance(existing_dict[key], list): existing_dict[key] = [existing_dict[key]]
                existing_dict[key].append(new_value)
        return existing_dict

    def get_best_parameters(self, param_type, server_type, cost):
        """Finds the best hyperparameter value based on validation loss."""
        # (Keep function code as provided in the previous pipeline.py)
        results, _ = self.load_results(param_type)
        if results is None or cost not in results: return None
        cost_results = results[cost]; server_metrics = {}
        if isinstance(cost_results, dict):
            for param_val, server_data in cost_results.items():
                if isinstance(server_data, dict) and server_type in server_data: server_metrics[param_val] = server_data[server_type]
        else: return None
        if not server_metrics: return None
        return self._select_best_hyperparameter(server_metrics)

    def _select_best_hyperparameter(self, param_results):
        """Selects best hyperparameter based on lowest median validation loss."""
        # (Keep function code as provided in the previous pipeline.py)
        best_loss = float('inf'); best_param = None
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
