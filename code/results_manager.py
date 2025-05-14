# results_manager.py
"""
Manages loading and saving experiment results and model states using
a flat list structure and JSON persistence.
"""
import os
import json
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Tuple, Optional, List, Union
import numpy as np
import torch
from configs import ROOT_DIR, MODEL_SAVE_DIR, RESULTS_DIR
# Import from helper.py
from helper import MetricKey, ExperimentType, infer_higher_is_better
# =============================================================================
# == Data Structures ==
# =============================================================================

@dataclass
class TrialRecord:
    """Record of a single trial's execution and metrics."""
    cost: Any
    run_idx: int
    server_type: str
    metrics: Optional[Dict] = None
    error: Optional[str] = None
    tuning_param_name: Optional[str] = None  # For tuning records
    tuning_param_value: Optional[float] = None  # For tuning records
    
    def to_dict(self) -> Dict:
        """Converts record to a dictionary for JSON serialization."""
        record_dict = {
            'cost': self.cost, 'run_idx': self.run_idx, 'server_type': self.server_type,
            'error': self.error
        }
        if self.tuning_param_name:
            record_dict['tuning_param_name'] = self.tuning_param_name
            record_dict['tuning_param_value'] = self.tuning_param_value
        if self.metrics:
            # Include all metrics including client-specific ones
            record_dict['metrics'] = self.metrics
        return record_dict

    # Helper to convert potentially non-JSON serializable types in metrics
    def _metrics_to_serializable(self) -> dict:
        """Convert metrics to JSON-serializable format."""
        serializable_metrics = {}
        for key, value_list in self.metrics.items():
            if isinstance(value_list, list):
                # Convert NumPy types to standard Python types
                serializable_metrics[key] = [
                    v.item() if isinstance(v, np.generic) else
                    (float(v) if isinstance(v, (np.float32, np.float64)) else v)
                    for v in value_list
                ]
            else:  # Handle cases where metrics might not be list
                serializable_metrics[key] = value_list
        return serializable_metrics
    
    def matches_config(self, cost: Any, server_type: str, 
                      param_name: Optional[str] = None, 
                      param_value: Optional[Any] = None) -> bool:
        """Check if this record matches the given configuration."""
        if self.cost != cost or self.server_type != server_type:
            return False
        if param_name is not None and self.tuning_param_name != param_name:
            return False
        if param_value is not None and self.tuning_param_value != param_value:
            return False
        return True
    
@dataclass
class OTAnalysisRecord:
    """Record of a single OT analysis result."""
    
    # Basic identification fields
    dataset_name: str
    fl_cost_param: Any
    fl_run_idx: int
    client_pair: str
    ot_method_name: str
    
    # Analysis results
    ot_cost_value: Optional[float] = None
    fl_local_metric: Optional[float] = None
    fl_fedavg_metric: Optional[float] = None
    fl_performance_delta: Optional[float] = None
    
    # Status information
    status: str = "Success"
    error_message: Optional[str] = None
    
    # Additional method-specific details
    ot_method_specific_results: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Converts the record to a dictionary suitable for JSON serialization."""
        result = asdict(self)
        
        # Convert NumPy types to Python types for JSON serialization
        for key in ['ot_cost_value', 'fl_local_metric', 'fl_fedavg_metric', 'fl_performance_delta']:
            if isinstance(result[key], (np.float32, np.float64, np.int32, np.int64)):
                result[key] = result[key].item()
        
        # Process method-specific results
        if result['ot_method_specific_results']:
            processed_results = {}
            for k, v in result['ot_method_specific_results'].items():
                if isinstance(v, (np.float32, np.float64, np.int32, np.int64)):
                    processed_results[k] = v.item()
                elif isinstance(v, np.ndarray):
                    processed_results[k] = v.tolist()
                elif isinstance(v, list) and v and isinstance(v[0], (np.float32, np.float64, np.int32, np.int64)):
                    processed_results[k] = [x.item() if isinstance(x, (np.float32, np.float64, np.int32, np.int64)) else x for x in v]
                else:
                    processed_results[k] = v
            result['ot_method_specific_results'] = processed_results
            
        return result

# =============================================================================
# == Path Management ==
# =============================================================================

class PathBuilder:
    """Helper class to construct standardized file paths."""
    def __init__(self, root_dir: str, dataset: str, num_target_clients: int):
        self.root_dir = root_dir
        self.dataset = dataset
        self.num_target_clients = num_target_clients
        self.results_base = RESULTS_DIR
        self.models_base = MODEL_SAVE_DIR

        # Ensure base directories exist
        os.makedirs(self.results_base, exist_ok=True)
        os.makedirs(self.models_base, exist_ok=True)

        # Mapping experiment type to subdirectory name
        self.exp_type_dirs = {
            ExperimentType.LEARNING_RATE: 'lr_tuning',
            ExperimentType.REG_PARAM: 'reg_param_tuning',
            ExperimentType.EVALUATION: 'evaluation',
            ExperimentType.OT_ANALYSIS: 'ot_analysis',
        }

    def get_results_path(self, experiment_type: str) -> Tuple[str, str]:
        """Gets path for results JSON and metadata JSON."""
        exp_dir_name = self.exp_type_dirs.get(experiment_type)
        if not exp_dir_name:
            raise ValueError(f"Unknown experiment type for results path: {experiment_type}")

        base_filename = f"{self.dataset}_{self.num_target_clients}clients_{exp_dir_name}"
        results_dir = os.path.join(self.results_base, exp_dir_name)
        os.makedirs(results_dir, exist_ok=True)

        results_json_path = os.path.join(results_dir, f"{base_filename}_results.json")
        metadata_json_path = os.path.join(results_dir, f"{base_filename}_meta.json")
        return results_json_path, metadata_json_path

    def get_model_save_path(self, num_clients_run: int, cost: Any, seed: int,
                            server_type: str, model_type: str) -> str:
        """Gets path for saving model state dictionary."""
        # Format cost parameter for filename
        cost_str = str(cost).replace('/', '_').replace('\\', '_').replace(' ', '')
        if isinstance(cost, (int, float)): 
            cost_str = f"{float(cost):.4f}"

        model_dir = os.path.join(self.models_base, self.dataset, 'evaluation')  
        os.makedirs(model_dir, exist_ok=True)

        filename = (f"{self.dataset}_{num_clients_run}clients_"
                   f"cost_{cost_str}_seed_{seed}_"
                   f"{server_type}_{model_type}_model.pt")
        return os.path.join(model_dir, filename)

# =============================================================================
# == Results Manager Class ==
# =============================================================================

class ResultsManager:
    """Handles saving/loading of results (List[TrialRecord]) and models."""

    def __init__(self, root_dir: str, dataset: str, num_target_clients: int):
        """
        Args:
            root_dir (str): Project root containing 'results', 'saved_models'.
            dataset (str): Name of the dataset.
            num_target_clients (int): Target client count for filename structure.
        """
        self.path_builder = PathBuilder(root_dir, dataset, num_target_clients)
        self.algorithm_list = ['local', 'fedavg', 'fedprox', 'pfedme', 'ditto']  # Default algorithms

    # --- Model Saving/Loading ---
    def save_model_state(self, model_state_dict: Optional[Dict], num_clients_run: int,
                         cost: Any, seed: int, server_type: str, model_type: str):
        """Saves a model's state_dict."""
        try:
            path = self.path_builder.get_model_save_path(
                num_clients_run, cost, seed, server_type, model_type
            )
            torch.save(model_state_dict, path)
        except Exception as e:
            print(f"ERROR: Failed to save {model_type} model state: {e}")

    def load_model_state(self, num_clients_run: int, cost: Any, seed: int,
                         server_type: str, model_type: str) -> Optional[Dict]:
        """Loads a model's state_dict."""
        try:
            path = self.path_builder.get_model_save_path(
                num_clients_run, cost, seed, server_type, model_type
            )
            if os.path.exists(path):
                # Load onto CPU by default to avoid GPU memory issues
                return torch.load(path, map_location=torch.device('cpu'))
            return None
        except Exception as e:
            print(f"ERROR: Failed to load {model_type} model state: {e}")
            return None

    # --- Results Saving/Loading ---
    def save_results(self, results_list: List[Dict], 
                    experiment_type: str,
                    run_metadata: Optional[Dict] = None):
        """
        Saves experiment results list and metadata to JSON files.
        Always overwrites existing files to ensure up-to-date results.
        """
        results_path, meta_path = self.path_builder.get_results_path(experiment_type)
        run_metadata = run_metadata if run_metadata is not None else {}

        # Check for errors within the list of records
        contains_errors = any('error' in record and record['error'] is not None for record in results_list)

        # Ensure selection criteria info is included in metadata if provided
        selection_info = {}
        if 'selection_criterion_key' in run_metadata:
            selection_info['selection_criterion_key'] = run_metadata['selection_criterion_key']
            
            # Also include if higher is better
            if 'selection_criterion_direction_overrides' in run_metadata:
                direction_overrides = run_metadata['selection_criterion_direction_overrides']
                is_higher_better = infer_higher_is_better(
                    run_metadata['selection_criterion_key'], 
                    direction_overrides
                )
                selection_info['criterion_is_higher_better'] = is_higher_better

        metadata = {
            'timestamp': datetime.now().isoformat(),
            'dataset': self.path_builder.dataset,
            'experiment_type': experiment_type,
            'num_target_clients': self.path_builder.num_target_clients,
            'contains_errors': contains_errors,
            'num_records': len(results_list),
            **selection_info,  # Include selection criteria info
            **run_metadata  # Merge run-specific info if provided
        }

        try:
            # Save metadata
            with open(meta_path, 'w') as f_meta:
                json.dump(metadata, f_meta, indent=4)

            # Save results list (already dicts)
            with open(results_path, 'w') as f_results:
                json.dump(results_list, f_results, indent=4)

        except Exception as e:
            print(f"ERROR saving results/metadata for {experiment_type}: {e}")

    def load_results(self, experiment_type: str) -> Tuple[List[Any], Optional[Dict]]:
        """
        Loads results list and metadata from JSON files.
        
        For normal experiment types (LEARNING_RATE, REG_PARAM, EVALUATION),
        converts loaded dictionaries to TrialRecord objects for compatibility
        with visualization functions.
        
        Returns:
            Tuple containing:
            - List of TrialRecord objects (or dictionaries for OT_ANALYSIS)
            - Optional metadata dictionary
        """
        results_path, meta_path = self.path_builder.get_results_path(experiment_type)
        loaded_dicts = []
        metadata = None

        # Load metadata if available
        if os.path.exists(meta_path):
            try:
                with open(meta_path, 'r') as f_meta:
                    metadata = json.load(f_meta)
            except Exception as e:
                print(f"Warning: Failed to load metadata from {meta_path}: {e}")

        # Load results if available
        if os.path.exists(results_path):
            try:
                with open(results_path, 'r') as f_results:
                    # Load list of dictionaries from JSON
                    loaded_dicts = json.load(f_results)
            except Exception as e:
                print(f"ERROR: Failed to load or parse results from {results_path}: {e}")
                loaded_dicts = []  # Return empty list on error

        # Convert dictionaries to appropriate objects based on experiment type
        if loaded_dicts:
            if experiment_type == ExperimentType.OT_ANALYSIS:
                # Keep as dictionaries for OT analysis
                return loaded_dicts, metadata
            else:
                # Convert to TrialRecord objects for standard experiment types
                try:
                    trial_records = [TrialRecord(**record_dict) for record_dict in loaded_dicts]
                    return trial_records, metadata
                except Exception as e:
                    print(f"WARNING: Error converting dictionaries to TrialRecord objects: {e}")
                    print(f"Returning raw dictionaries instead - some functions may not work correctly.")
                    
        # Default return (empty list or unconverted dictionaries)
        return loaded_dicts, metadata

    # --- Results Analysis & Status ---
    def get_best_parameters(self, param_type: str, server_type: str, cost: Any, 
                        tuning_target_metric_key: Optional[str] = None,
                        dataset_direction_overrides: Optional[Dict[str, str]] = None) -> Optional[Any]:
        """
        Finds the best hyperparameter (LR or Reg) based on the specified metric across runs.
        
        Args:
            param_type: Type of parameter tuning ('learning_rate' or 'reg_param')
            server_type: Server algorithm type
            cost: Cost value to filter records
            tuning_target_metric_key: Metric to optimize (if None, uses 'val_losses')
            dataset_direction_overrides: Optional overrides for determining if higher is better
            
        Returns:
            The best parameter value or None if not found
        """
        records, _ = self.load_results(param_type)
        if not records:
            return None

        # Default to val_losses if not specified
        metric_key = tuning_target_metric_key or MetricKey.VAL_LOSSES
        
        # Determine if higher is better for this metric
        higher_is_better = infer_higher_is_better(metric_key, dataset_direction_overrides)

        # Filter records for the specific cost and server
        relevant_records = [
            r for r in records
            if r.cost == cost and r.server_type == server_type and r.error is None
        ]

        if not relevant_records:
            return None

        tuning_param_name = relevant_records[0].tuning_param_name
        if not tuning_param_name:
            return None  # Parameter name should be set during tuning

        # Group metrics by parameter value
        param_values = {}  # {param_val: [list_of_metrics_per_run]}
        for record in relevant_records:
            param_val = record.tuning_param_value
            metric_list = record.metrics.get(metric_key, [])

            if not metric_list:
                continue  # Skip if no metrics

            try:
                # Use median for robustness
                run_metric = float(np.nanmedian(metric_list))
                if np.isnan(run_metric):
                    continue
            except:
                continue

            if param_val not in param_values:
                param_values[param_val] = []
            param_values[param_val].append(run_metric)

        if not param_values:
            return None

        # Calculate average metric for each param value
        avg_metrics = {
            param_val: np.mean(run_metrics) 
            for param_val, run_metrics in param_values.items() 
            if run_metrics
        }

        if not avg_metrics:
            return None

        # Find the best parameter value based on higher_is_better
        best_param = max(avg_metrics, key=avg_metrics.get) if higher_is_better else min(avg_metrics, key=avg_metrics.get)
        return best_param

    def get_experiment_status(self, experiment_type: str,
                             expected_costs: List[Any],
                             default_params: Dict,
                             metric_key_cls: type  # Kept for interface compatibility
                             ) -> Tuple[List[TrialRecord], List[Any], int]:
        """
        Analyzes existing results to determine what needs to be processed.
        
        Returns:
            - The loaded list of TrialRecords
            - List of costs that need processing
            - Number of completed runs across ALL configs
        """
        records, metadata = self.load_results(experiment_type)
        
        # Always re-run if there are errors
        if metadata and metadata.get('contains_errors', False):
            print(f"Warning: Previous errors found in {experiment_type} results. Will reprocess.")
            return records or [], list(expected_costs), 0
            
        # Determine run count target based on experiment type
        is_tuning = experiment_type != ExperimentType.EVALUATION
        target_runs_key = 'runs_tune' if is_tuning else 'runs'
        target_runs = default_params.get(target_runs_key, 1)
        
        # Get expected servers and parameters based on experiment type
        if is_tuning:
            servers_key = 'servers_tune_lr' if experiment_type == ExperimentType.LEARNING_RATE else 'servers_tune_reg'
            expected_servers = default_params.get(servers_key, [])
            
            param_key = 'learning_rates_try' if experiment_type == ExperimentType.LEARNING_RATE else 'reg_params_try'
            param_name = 'learning_rate' if experiment_type == ExperimentType.LEARNING_RATE else 'reg_param'
            params_to_try = default_params.get(param_key, [])
        else:
            # For evaluation, use all algorithms
            expected_servers = self.algorithm_list
            param_name = None
            params_to_try = [None]
            
        # Track completion counts for each configuration
        min_complete_runs = float('inf')
        incomplete_costs = set()
        
        for cost in expected_costs:
            cost_is_complete = True
            
            for server in expected_servers:
                for param_val in params_to_try:
                    # Find matching records for this configuration
                    matching_records = [
                        r for r in records 
                        if r.matches_config(
                            cost=cost, 
                            server_type=server,
                            param_name=param_name,
                            param_value=param_val
                        )
                    ]
                    
                    # Count successful runs
                    valid_run_count = sum(1 for r in matching_records if r.error is None)
                    
                    if valid_run_count < target_runs:
                        cost_is_complete = False
                        
                    min_complete_runs = min(min_complete_runs, valid_run_count)
            
            if not cost_is_complete:
                incomplete_costs.add(cost)
                
        # Handle edge case where no configs were checked
        min_complete_runs = 0 if min_complete_runs == float('inf') else min_complete_runs
        
        # Determine which costs to process
        if min_complete_runs < target_runs:
            # If we haven't completed the minimum number of runs, process all costs
            remaining_costs = list(expected_costs)
        else:
            # Otherwise just process incomplete costs
            remaining_costs = sorted(list(incomplete_costs))
            
        return records, remaining_costs, min_complete_runs