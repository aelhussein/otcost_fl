"""
Main experiment pipeline orchestration.
Initializes data, creates models/servers, runs training/evaluation loops.
"""
import traceback
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Subset, ConcatDataset, DataLoader, Dataset as TorchDataset
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple, Any

# Import project modules
from configs import (
    ROOT_DIR, DATA_DIR, ALGORITHMS, DEVICE
)
from helper import ( # Import necessary helper functions
    set_seeds, cleanup_gpu, get_dice_score, get_parameters_for_dataset,
    get_default_lr, get_default_reg, translate_cost, validate_dataset_config
)
# Import server types and data structures
from servers import Server, FedAvgServer, TrainerConfig, SiteData, ModelState
import models as ms # Import model architectures module
from losses import * # Import custom losses if any (may not be needed if criterion map covers all)

# Import newly separated modules
from data_loading import DATA_LOADERS
from data_partitioning import PARTITIONING_STRATEGIES
from data_processing import DataPreprocessor # Import the preprocessor class
from results_manager import ResultsManager, ExperimentType # Import results manager and Enum

# Define Experiment types locally for clarity if not imported from results_manager
# class ExperimentType:
#     LEARNING_RATE = 'learning_rate'; REG_PARAM = 'reg_param'; EVALUATION = 'evaluation'; DIVERSITY = 'diversity'

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

        # Determine initial target client count for filenames/ResultsManager
        initial_target_num_clients = self.config.num_clients
        if initial_target_num_clients is None:
            initial_target_num_clients = self.default_params.get('default_num_clients', 2)

        if not isinstance(initial_target_num_clients, int) or initial_target_num_clients <= 0:
             raise ValueError(f"Invalid initial target client count: {initial_target_num_clients}")

        print(f"Initializing ResultsManager for {initial_target_num_clients} target clients (filename).")
        self.results_manager = ResultsManager(
            root_dir=ROOT_DIR,
            dataset=self.config.dataset,
            experiment_type=self.config.experiment_type,
            num_clients=initial_target_num_clients
        )
        # Stores the actual number of clients used in a specific run (determined later)
        self.num_clients_for_run: Optional[int] = None

    def run_experiment(self, costs: List[Any]):
        """
        Main entry point to run the configured experiment type over the specified costs.

        Args:
            costs (List[Any]): A list of cost/heterogeneity parameters to iterate over.

        Returns:
            The final results dictionary (structure depends on experiment type).
        """
        if self.config.experiment_type == ExperimentType.EVALUATION:
            results, _ = self._run_final_evaluation(costs) # Ignore diversity dict here
            return results
        elif self.config.experiment_type in [ExperimentType.LEARNING_RATE, ExperimentType.REG_PARAM]:
            return self._run_hyperparam_tuning(costs)
        else:
            raise ValueError(f"Unsupported experiment type: {self.config.experiment_type}")

    def _check_existing_results(self, costs: List[Any]) -> Tuple[Optional[Dict], List[Any], int]:
        """
        Checks existing results file for the current experiment type.
        Determines which costs need to be run and how many runs are already completed.
        Handles results files that may contain error flags.

        Args:
            costs (List[Any]): The full list of costs planned for this experiment.

        Returns:
            Tuple containing:
            - Optional[Dict]: Existing results data (or None).
            - List[Any]: List of costs that still need processing.
            - int: Number of valid runs found in existing results.
        """
        results, metadata = self.results_manager.load_results(self.config.experiment_type)

        # Determine target number of runs based on experiment type
        is_tuning = self.config.experiment_type != ExperimentType.EVALUATION
        runs_key = 'runs_tune' if is_tuning else 'runs'
        target_runs = self.default_params.get(runs_key, 1)

        remaining_costs = list(costs) # Start assuming all costs need to be run
        completed_runs = 0

        if results is not None:
            # Check metadata first for global error flag
            has_errors = metadata is not None and metadata.get('contains_errors', False)
            if has_errors:
                print("Previous results file contains errors. Re-running all experiments.")
                # Force re-run of all costs for all runs
                return results, remaining_costs, 0

            completed_costs_set = set(results.keys())
            remaining_costs = [c for c in costs if c not in completed_costs_set]

            if completed_costs_set:
                # Try to determine completed runs from the structure of the first cost result
                first_cost = next(iter(completed_costs_set))
                try:
                    first_param = next(iter(results[first_cost]))
                    first_server = next(iter(results[first_cost][first_param]))
                    loss_key = 'val_losses' if is_tuning else 'test_losses'

                    # Check if the expected keys exist
                    if (first_server in results[first_cost][first_param] and
                        'global' in results[first_cost][first_param][first_server] and
                        loss_key in results[first_cost][first_param][first_server]['global']):

                        loss_data = results[first_cost][first_param][first_server]['global'][loss_key]

                        if not loss_data:
                             completed_runs = 0 # No loss data found
                        elif isinstance(loss_data[0], list): # Multi-run format: [[...], [...]]
                             if all(isinstance(run, list) for run in loss_data):
                                 completed_runs = len(loss_data)
                             else:
                                 print(f"Warning: Inconsistent format in '{loss_key}'. Expected list of lists.")
                                 completed_runs = 0 # Treat as incomplete/corrupt
                        elif isinstance(loss_data[0], (float, int, np.number)) or \
                             (isinstance(loss_data[0], list) and len(loss_data[0])==1) :
                             # Assumes single run format: [val1, val2] or [[val1], [val2]]
                             completed_runs = 1
                        else:
                             print(f"Warning: Unrecognized format for '{loss_key}'. Assuming 0 completed runs.")
                             completed_runs = 0
                    else:
                        completed_runs = 0 # Keys missing, cannot determine

                    # Check individual cost results for errors
                    for cost_key in completed_costs_set:
                        if self.results_manager._check_for_errors_in_results(results.get(cost_key)):
                            print(f"Errors found within results for cost {cost_key}. Adding to re-run list.")
                            if cost_key not in remaining_costs:
                                remaining_costs.append(cost_key)

                except (StopIteration, KeyError, IndexError, TypeError, AttributeError) as e:
                    # Error parsing the results structure
                    completed_runs = 0
                    print(f"Could not reliably determine completed runs due to results structure error: {e}")
                    # Assume incomplete, force re-run of everything
                    remaining_costs = list(costs)

            # Ensure remaining_costs are unique and sorted for consistent processing order
            remaining_costs = sorted(list(set(remaining_costs)))

            print(f"Found {completed_runs}/{target_runs} completed valid run(s) in existing results.")

            # Decide final list of costs to run
            if completed_runs >= target_runs and not remaining_costs:
                # All runs complete, no specific costs failed or missing
                remaining_costs = []
                print("All target runs completed and no missing/failed costs found.")
            elif completed_runs < target_runs:
                # Not enough runs completed, need to rerun everything for the missing runs
                remaining_costs = list(costs) # Ensure all costs are run for the remaining runs
                print(f"Target runs ({target_runs}) not met. Will run all costs for remaining runs.")
            # else: completed_runs >= target_runs but some costs need rerunning (handled by remaining_costs list)

        else: # No results file found
            print("No existing results file found.")
            remaining_costs = list(costs)
            completed_runs = 0

        print(f"Remaining costs to process: {remaining_costs}")
        return results, remaining_costs, completed_runs

    def _run_hyperparam_tuning(self, costs: List[Any]) -> Dict:
        """
        Runs hyperparameter tuning loops over specified costs and parameter ranges.

        Args:
            costs: List of cost/heterogeneity parameters.

        Returns:
            Dictionary containing tuning results.
        """
        results, remaining_costs, completed_runs = self._check_existing_results(costs)
        runs_tune = self.default_params.get('runs_tune', 1)

        # Check if tuning is already complete
        if not remaining_costs and completed_runs >= runs_tune:
            print("Hyperparameter tuning already complete.")
            return results if results is not None else {}

        # Determine how many more runs are needed
        remaining_runs_count = runs_tune - completed_runs
        if remaining_runs_count <= 0:
            if remaining_costs:
                # Runs complete, but some costs are missing/failed - redo all runs for those costs
                print(f"Target runs complete, but costs {remaining_costs} missing/failed. "
                      f"Rerunning these costs for all {runs_tune} runs.")
                remaining_runs_count = runs_tune
                completed_runs = 0 # Reset completed count as we redo runs for these costs
            else:
                 # Runs complete and no missing costs
                 print("Target tuning runs already completed.")
                 return results if results is not None else {}

        print(f"Starting {remaining_runs_count} hyperparameter tuning run(s)...")
        results = results if results is not None else {}
        num_clients_metadata = None # Store the client count used for metadata saving (use first determined)

        # Loop through the required number of tuning runs
        for run_idx in range(remaining_runs_count):
            current_run_total = completed_runs + run_idx + 1
            current_seed = self.base_seed + run_idx # Use run-specific seed
            print(f"\n--- Starting Tuning Run {current_run_total}/{runs_tune} (Seed: {current_seed}) ---")
            set_seeds(current_seed) # Set seed for data initialization and model weights

            run_meta = {'run_number': current_run_total, 'seed_used': current_seed}
            cost_client_counts = {} # Track client counts used per cost in this run

            # Process each cost that needs running/re-running
            for cost in remaining_costs:
                print(f"\n--- Processing Cost: {cost} (Run {current_run_total}) ---")
                try:
                    num_clients_this_cost = self._get_final_client_count(self.default_params, cost)
                    cost_client_counts[cost] = num_clients_this_cost
                    if num_clients_metadata is None:
                        num_clients_metadata = num_clients_this_cost # Store for saving results filename
                except Exception as e:
                    print(f"ERROR determining client count for cost {cost}: {e}. Skipping cost.")
                    # Store error in results? Optional.
                    if cost not in results: results[cost] = {}
                    results[cost]['error'] = f"Client count determination failed: {e}"
                    continue

                # Determine parameters to tune based on experiment type
                if self.config.experiment_type == ExperimentType.LEARNING_RATE:
                     param_key = 'learning_rate'
                     fixed_param_key = 'reg_param'
                     fixed_param_value = get_default_reg(self.config.dataset)
                     params_to_try_values = self.default_params.get('learning_rates_try', [])
                     server_types_to_tune = self.default_params.get('servers_tune_lr', ALGORITHMS)
                elif self.config.experiment_type == ExperimentType.REG_PARAM:
                     param_key = 'reg_param'
                     fixed_param_key = 'learning_rate'
                     fixed_param_value = get_default_lr(self.config.dataset)
                     params_to_try_values = self.default_params.get('reg_params_try', [])
                     server_types_to_tune = self.default_params.get('servers_tune_reg', []) # May be specific servers
                else:
                     # Should not happen based on run_experiment logic
                     raise ValueError(f"Unsupported tuning type: {self.config.experiment_type}")

                if not params_to_try_values:
                     print(f"Warning: No parameters specified in config to try for "
                           f"{self.config.experiment_type}, cost {cost}. Skipping.")
                     continue
                if not server_types_to_tune:
                     print(f"Warning: No server types specified in config for tuning "
                           f"{self.config.experiment_type}, cost {cost}. Skipping.")
                     continue

                # Create list of hyperparameter dictionaries to try
                hyperparams_list = [
                    {param_key: p_val, fixed_param_key: fixed_param_value}
                    for p_val in params_to_try_values
                ]

                tuning_results_for_cost = {} # Store {param_val: {server: metrics}} for this cost
                for hyperparams in hyperparams_list:
                    param_value_being_tuned = hyperparams[param_key]
                    print(f"--- Tuning Param: {param_key}={param_value_being_tuned} ---")
                    # Run tuning for this specific hyperparameter setting across specified servers
                    server_metrics = self._hyperparameter_tuning(cost, hyperparams, server_types_to_tune)
                    tuning_results_for_cost[param_value_being_tuned] = server_metrics

                # Aggregate results for this cost into the main results dict
                if cost not in results:
                    results[cost] = {}
                for param_val, server_data in tuning_results_for_cost.items():
                    if param_val not in results[cost]:
                        results[cost][param_val] = {}
                    for server_type, metrics in server_data.items():
                        # Initialize server results if not present
                        if server_type not in results[cost][param_val]:
                            results[cost][param_val][server_type] = None # Initialize appropriately if needed
                        # Append new run's metrics
                        results[cost][param_val][server_type] = self.results_manager.append_or_create_metric_lists(
                            results[cost][param_val][server_type], metrics
                        )

            # Save results after each tuning run completes for all relevant costs
            print(f"--- Completed Tuning Run {current_run_total}/{runs_tune} ---")
            run_meta['cost_specific_client_counts'] = cost_client_counts
            self.results_manager.save_results(
                results,
                self.config.experiment_type,
                client_count=num_clients_metadata, # Use the consistent client count for filename
                run_metadata=run_meta
            )

        return results

    def _hyperparameter_tuning(self,
                               cost: Any,
                               hyperparams: Dict,
                               server_types: List[str]) -> Dict:
        """
        Internal helper: Runs training for one hyperparameter setting across servers.

        Args:
            cost: The current cost/heterogeneity parameter.
            hyperparams: Dictionary containing the hyperparameter values for this run
                         (e.g., {'learning_rate': 0.01, 'reg_param': 0.1}).
            server_types: List of server algorithms to run (e.g., ['local', 'fedavg']).

        Returns:
            Dictionary mapping server_type to its collected metrics for this setting.
        """
        # Initialize data loaders for this cost (seed is set in the outer loop)
        try:
            client_dataloaders = self._initialize_experiment(cost)
        except Exception as e:
            error_msg = f"Initialization failed for cost {cost}: {e}"
            print(f"ERROR: {error_msg}")
            traceback.print_exc()
            # Return error status for all expected servers for this hyperparam setting
            return {st: {'error': error_msg} for st in server_types}

        if not client_dataloaders:
            error_msg = f"No client dataloaders returned for cost {cost}."
            print(f"Warning: {error_msg}")
            return {st: {'error': error_msg} for st in server_types}

        tracking: Dict[str, Dict] = {} # {server_type: metrics_dict}

        for server_type in server_types:
            print(f"..... Tuning Server: {server_type} .....")
            lr = hyperparams.get('learning_rate')
            reg_param_val = hyperparams.get('reg_param')

            # Prepare algorithm-specific parameters if needed
            algo_params_dict = {}
            if server_type in ['pfedme', 'ditto'] and reg_param_val is not None:
                algo_params_dict['reg_param'] = reg_param_val
                print(f"  Using reg_param: {reg_param_val} for {server_type}")
            elif server_type in ['pfedme', 'ditto']:
                 print(f"  Warning: Reg param expected but not found in hyperparams for {server_type}")

            # Create configuration for the trainer
            trainer_config = self._create_trainer_config(server_type, lr, algo_params_dict)

            # Use fewer rounds for tuning if specified in config, otherwise use default rounds
            num_tuning_rounds = self.default_params.get('rounds_tune_inner', trainer_config.rounds)
            trainer_config.rounds = num_tuning_rounds # Adjust rounds for this tuning run
            print(f"  Tuning for {num_tuning_rounds} rounds.")

            server = None
            try:
                # Create server instance (model initialized based on global seed)
                server = self._create_server_instance(cost, server_type, trainer_config, tuning=True)
                # Add clients to the server
                self._add_clients_to_server(server, client_dataloaders)
                if not server.clients:
                    print(f"Warning: No clients were added to server {server_type}.")
                    tracking[server_type] = {'error': 'No clients added'}
                    continue

                # Train and evaluate for the specified number of tuning rounds
                # Pass the current base seed for potential use within train_and_evaluate if needed, though seed should be set globally
                metrics = self._train_and_evaluate(server, trainer_config.rounds, cost, self.base_seed)
                tracking[server_type] = metrics

            except Exception as e:
                error_msg = f"Error during tuning run for {server_type}: {e}"
                print(f"ERROR: {error_msg}")
                traceback.print_exc()
                tracking[server_type] = {'error': error_msg}
            finally:
                # Clean up resources
                del server # Ensure server object is released
                cleanup_gpu() # Clear CUDA cache if applicable

        return tracking

    def _run_final_evaluation(self, costs: List[Any]) -> Tuple[Dict, Optional[Dict]]:
        """
        Runs final evaluation loops using best hyperparameters found during tuning.

        Args:
            costs: List of cost/heterogeneity parameters.

        Returns:
            Tuple containing:
            - Dictionary with evaluation results.
            - Dictionary with diversity metrics (or None).
        """
        results, remaining_costs, completed_runs = self._check_existing_results(costs)
        target_runs = self.default_params.get('runs', 1)

        # Check if evaluation is already complete
        if not remaining_costs and completed_runs >= target_runs:
            print("Final evaluation already complete.")
            # Still load diversity results in case they were saved separately
            diversities, _ = self.results_manager.load_results(ExperimentType.DIVERSITY)
            return results if results is not None else {}, diversities

        # Determine how many more runs are needed
        remaining_runs_count = target_runs - completed_runs
        if remaining_runs_count <= 0:
            if remaining_costs:
                # Runs done, but some costs missing/failed - redo all runs for these costs
                print(f"Target runs complete, but costs {remaining_costs} missing/failed. "
                      f"Rerunning these costs for all {target_runs} runs.")
                remaining_runs_count = target_runs
                completed_runs = 0 # Reset completed count
            else:
                 # Runs done, no missing costs
                 print("Target evaluation runs already completed.")
                 diversities, _ = self.results_manager.load_results(ExperimentType.DIVERSITY)
                 return results if results is not None else {}, diversities

        print(f"Starting {remaining_runs_count} final evaluation run(s)...")
        # Initialize results dictionaries if they don't exist
        results = results if results is not None else {}
        diversities, diversity_metadata = self.results_manager.load_results(ExperimentType.DIVERSITY)
        diversities = diversities if diversities is not None else {}

        # Check if existing diversity results have errors flagged
        if diversity_metadata and diversity_metadata.get('contains_errors', False):
            print("Found errors flagged in existing diversity metrics. Regenerating.")
            diversities = {} # Discard potentially corrupt diversity results

        num_clients_metadata = None # Store client count for saving results

        # Loop through the required number of evaluation runs
        for run_idx in range(remaining_runs_count):
            current_run_total = completed_runs + run_idx + 1
            current_seed = self.base_seed + run_idx # Use run-specific seed
            print(f"\n--- Starting Final Evaluation Run {current_run_total}/{target_runs} (Seed: {current_seed}) ---")
            set_seeds(current_seed) # Set seed for data, model init, training randomness

            # Dictionaries to store results for THIS run only
            results_this_run: Dict[Any, Dict] = {}
            diversities_this_run: Dict[Any, Dict] = {}
            run_meta = {'run_number': current_run_total, 'seed_used': current_seed}
            cost_client_counts = {} # Track client counts per cost for this run

            # Determine which costs to process in this iteration
            # Rerun ALL costs if runs were incomplete, otherwise only remaining/failed costs
            costs_to_run_this_iter = costs if completed_runs < target_runs else remaining_costs

            for cost in costs_to_run_this_iter:
                print(f"\n--- Evaluating Cost: {cost} (Run {current_run_total}) ---")
                num_clients_this_cost = 0
                trained_servers: Dict[str, Optional[Server]] = {} # Store server instances
                final_round_num = 0

                try:
                    # Determine client count for this cost parameter
                    num_clients_this_cost = self._get_final_client_count(self.default_params, cost)
                    cost_client_counts[cost] = num_clients_this_cost
                    if num_clients_metadata is None:
                        num_clients_metadata = num_clients_this_cost # Use first determined count for saving

                    # Run evaluation for all algorithms for this cost and seed
                    experiment_results_for_cost, final_round_num, trained_servers = self._final_evaluation(cost, current_seed)

                    # --- Model Saving ---
                    # Save models only if the evaluation for this cost didn't explicitly return an error key
                    if 'error' not in experiment_results_for_cost:
                        self._save_evaluation_models(
                            trained_servers, num_clients_this_cost, cost, current_seed
                        )
                    else:
                         # Log if an error occurred during the evaluation for this cost
                         print(f"  Skipping model saving for cost {cost} due to evaluation error: "
                               f"{experiment_results_for_cost.get('error')}")


                    # Extract diversity metrics if they were calculated and returned
                    if 'weight_metrics' in experiment_results_for_cost:
                        diversities_this_run[cost] = experiment_results_for_cost.pop('weight_metrics')

                    # Store the main evaluation metrics for this cost
                    results_this_run[cost] = experiment_results_for_cost

                except Exception as e:
                    # Catch unexpected errors during the cost evaluation loop
                    error_msg = f"Outer error during final evaluation for cost {cost}, run {current_run_total}: {e}"
                    print(f"ERROR: {error_msg}")
                    traceback.print_exc()
                    results_this_run[cost] = {'error': error_msg}
                finally:
                    # Ensure server objects are cleaned up even if errors occurred
                    del trained_servers # Remove references to server objects
                    cleanup_gpu() # Clear GPU cache

            # --- Aggregation and Saving after each run ---
            # Append results for this run to the main results dictionaries
            results = self.results_manager.append_or_create_metric_lists(results, results_this_run)
            diversities = self.results_manager.append_or_create_metric_lists(diversities, diversities_this_run)

            # Save results after each full evaluation run completes
            print(f"--- Completed Final Evaluation Run {current_run_total}/{target_runs} ---")
            run_meta['cost_specific_client_counts'] = cost_client_counts

            # Save main evaluation metrics
            self.results_manager.save_results(
                results,
                ExperimentType.EVALUATION,
                client_count=num_clients_metadata,
                run_metadata=run_meta
            )
            # Save diversity metrics separately
            self.results_manager.save_results(
                diversities,
                ExperimentType.DIVERSITY,
                client_count=num_clients_metadata,
                run_metadata=run_meta # Can use same metadata or tailor if needed
            )

        return results, diversities

    def _final_evaluation(self, cost: Any, seed: int) -> Tuple[Dict, int, Dict]:
        """
        Internal helper: Runs final evaluation for ONE cost value across all servers.

        Args:
            cost: The current cost/heterogeneity parameter.
            seed: The random seed for this specific evaluation run.

        Returns:
            Tuple containing:
            - Dictionary of evaluation metrics for each server type.
            - The number of rounds the evaluation ran for.
            - Dictionary mapping server_type to the trained server instance.
        """
        tracking: Dict[str, Dict] = {}          # Stores metrics per server {server_type: metrics}
        weight_metrics_acc: Dict = {}           # Stores diversity metrics if calculated
        trained_servers: Dict[str, Optional[Server]] = {} # Stores server instances
        # Use default rounds from config, might be overwritten by trainer_config
        final_round_num = self.default_params.get('rounds', 100)

        # --- 1. Initialize Data ---
        # Seed should already be set globally by the calling loop
        try:
            client_dataloaders = self._initialize_experiment(cost)
            # Store actual client count determined during initialization
            num_clients_this_run = self.num_clients_for_run
            if not client_dataloaders: # Check if initialization returned empty dict
                 error_msg = "Initialization returned no client dataloaders."
                 print(f"ERROR: {error_msg} for cost {cost}")
                 return {'error': error_msg}, 0, {}
        except Exception as e:
            error_msg = f"Data initialization failed for cost {cost}: {e}"
            print(f"ERROR: {error_msg}")
            traceback.print_exc()
            return {'error': error_msg}, 0, {}

        if num_clients_this_run == 0: # Double check based on the stored value
            error_msg = "No clients initialized."
            print(f"ERROR: {error_msg} for cost {cost}")
            return {'error': error_msg}, 0, {}

        # --- 2. Loop Through Algorithms ---
        for server_type in ALGORITHMS:
            print(f"..... Evaluating Server: {server_type} for Cost: {cost} (Seed: {seed}) .....")

            # Fetch best hyperparameters (or defaults)
            best_lr = self.results_manager.get_best_parameters(
                ExperimentType.LEARNING_RATE, server_type, cost
            ) or get_default_lr(self.config.dataset)
            print(f"  Using LR: {best_lr}")

            algo_params_dict = {}
            if server_type in ['pfedme', 'ditto']: # Add other algos needing reg params
                 best_reg = self.results_manager.get_best_parameters(
                     ExperimentType.REG_PARAM, server_type, cost
                 ) or get_default_reg(self.config.dataset)
                 if best_reg is not None:
                      algo_params_dict['reg_param'] = best_reg
                      print(f"  Using Reg Param: {best_reg}")
                 else:
                      print(f"  Warning: No default or tuned reg param found for {server_type}")

            # Create trainer configuration
            trainer_config = self._create_trainer_config(server_type, best_lr, algo_params_dict)
            final_round_num = trainer_config.rounds # Update actual rounds used

            server: Optional[Server] = None
            try:
                # Create server instance (model initialized with current seed)
                server = self._create_server_instance(
                    cost, server_type, trainer_config, tuning=False # Ensure tuning=False
                )

                # Save initial model state for FedAvg
                self._save_initial_model_state(server, num_clients_this_run, cost, seed)

                # Add clients to the server
                self._add_clients_to_server(server, client_dataloaders)
                if not server.clients:
                    print(f"Warning: No clients added to {server_type}.")
                    tracking[server_type] = {'error': 'No clients added'}
                    trained_servers[server_type] = None
                    continue # Skip to next server type

                # Run training and evaluation
                metrics = self._train_and_evaluate(server, trainer_config.rounds, cost, seed)
                tracking[server_type] = metrics
                trained_servers[server_type] = server # Store successful server instance

                # Collect diversity metrics if available from the server
                if hasattr(server, 'diversity_metrics') and server.diversity_metrics:
                    weight_metrics_acc.update(server.diversity_metrics) # Merge metrics

            except Exception as e:
                error_msg = f"Error during evaluation run for {server_type}, cost {cost}: {e}"
                print(f"ERROR: {error_msg}")
                traceback.print_exc()
                tracking[server_type] = {'error': error_msg}
                trained_servers[server_type] = None # Mark as failed
                # Clean up potentially partially created server
                if server:
                    del server
                cleanup_gpu()

            # Note: Successful server instances are kept in trained_servers and deleted
            #       by the calling function (_run_final_evaluation) after model saving.

        # Add accumulated diversity metrics to the results for this cost
        if weight_metrics_acc:
            tracking['weight_metrics'] = weight_metrics_acc

        return tracking, final_round_num, trained_servers

    # --- Data Initialization Logic ---
    def _initialize_experiment(self, cost: Any) -> Dict[str, Tuple[DataLoader, DataLoader, DataLoader]]:
        """
        Handles the complete data pipeline: loading, partitioning, preprocessing.

        Args:
            cost: The current cost/heterogeneity parameter.

        Returns:
            A dictionary mapping client_id to a tuple of (train, val, test) DataLoaders.
            Returns an empty dictionary if data initialization fails.
        """
        print(f"--- Initializing Data for Cost: {cost} ---")
        # Seed should be set globally by the calling loop before this function

        dataset_name: str = self.config.dataset
        dataset_config: Dict = self.default_params # Get config for this dataset

        # 1. Validate config and get essential parameters
        try:
            config_params = self._validate_and_prepare_config(dataset_name, dataset_config)
            data_source = config_params['data_source']
            partitioning_strategy = config_params['partitioning_strategy']
            cost_interpretation = config_params['cost_interpretation']
            source_args = config_params['source_args']
            partitioner_args = config_params['partitioner_args']
            partition_scope = config_params['partition_scope']
            sampling_config = dataset_config.get('sampling_config') # Optional: for partitioner
        except Exception as e:
             print(f"Error preparing configuration: {e}")
             return {}

        # 2. Determine number of clients for this cost
        try:
            num_clients = self._get_final_client_count(dataset_config, cost)
            client_ids_list = [f'client_{i+1}' for i in range(num_clients)]
            self.num_clients_for_run = num_clients # Store actual count for this run
            print(f"Target number of clients for this cost: {num_clients}")
        except Exception as e:
             print(f"Error determining client count: {e}")
             return {}

        # 3. Translate cost parameter (e.g., to alpha, key, suffix)
        try:
            translated_cost = translate_cost(cost, cost_interpretation)
            print(f"Translated cost ({cost_interpretation}): {translated_cost}")
        except Exception as e:
             print(f"Error translating cost '{cost}' with interpretation '{cost_interpretation}': {e}")
             return {}

        # 4. Load/Partition/Prepare data based on strategy
        client_input_data: Dict[str, Any] = {}
        preprocessor_input_type: str = 'unknown'
        data_to_partition: Optional[torch.utils.data.Dataset] = None

        try:
            if partitioning_strategy.endswith('_indices'):
                # Load base dataset -> Partition indices -> Create Subsets
                print(f"Loading base data source: {data_source}...")
                source_data = self._load_source_data(data_source, dataset_name, source_args)

                # Determine the actual dataset object to partition
                if isinstance(source_data, tuple) and len(source_data) == 2: # e.g., torchvision train/test tuple
                    base_train_ds, base_test_ds = source_data
                    if partition_scope == 'train':
                        data_to_partition = base_train_ds
                    elif partition_scope == 'all':
                        print("Concatenating train and test sets for partitioning...")
                        data_to_partition = ConcatDataset([base_train_ds, base_test_ds])
                    else:
                        raise ValueError(f"Unsupported partition_scope '{partition_scope}' for tuple data source.")
                elif isinstance(source_data, torch.utils.data.Dataset): # e.g., single base dataset returned
                    data_to_partition = source_data
                else:
                    raise TypeError(f"Unexpected source data type from {data_source}: {type(source_data)}")

                print(f"Partitioning {len(data_to_partition)} samples using strategy: {partitioning_strategy}...")
                client_partition_result = self._partition_data(
                    partitioning_strategy, data_to_partition, num_clients,
                    partitioner_args, translated_cost, sampling_config
                )

                client_input_data, preprocessor_input_type = self._prepare_client_data(
                    client_ids_list, partitioning_strategy, client_partition_result,
                    data_to_partition, dataset_name, dataset_config, translated_cost
                )

            elif partitioning_strategy == 'pre_split':
                # Load data per client -> Prepare dictionary for preprocessor
                print(f"Using pre-split strategy. Loading data per client using: {data_source}...")
                client_input_data, preprocessor_input_type = self._prepare_client_data(
                    client_ids_list, partitioning_strategy, None, None, # No partition result/base data needed here
                    dataset_name, dataset_config, translated_cost
                )

            else:
                raise NotImplementedError(f"Partitioning strategy '{partitioning_strategy}' is not supported.")

        except Exception as e:
            print(f"Error during data loading/partitioning stage for cost {cost}: {e}")
            traceback.print_exc()
            return {} # Return empty if core data pipeline fails

        # 5. Check if any client data was actually prepared
        if not client_input_data:
            print(f"Warning: No client data was successfully loaded or prepared for cost {cost}.")
            return {}

        # 6. Preprocess client data (create DataLoaders, handle scaling/splits)
        try:
            print(f"Preprocessing client data (input type: {preprocessor_input_type})...")
            # Instantiate the preprocessor with dataset config
            preprocessor = DataPreprocessor(dataset_name, dataset_config)
            client_dataloaders = preprocessor.process_client_data(client_input_data, preprocessor_input_type)
        except Exception as e:
            print(f"Error during data preprocessing stage for cost {cost}: {e}")
            traceback.print_exc()
            return {}

        print(f"--- Data Initialization Complete for Cost: {cost} ---")
        return client_dataloaders

    # --- Data Initialization Helpers ---
    def _validate_and_prepare_config(self, dataset_name: str, dataset_config: Dict) -> Dict:
        """Validates dataset config and extracts pipeline-relevant keys."""
        validate_dataset_config(dataset_config, dataset_name) # Use helper validation
        required_keys = ['data_source', 'partitioning_strategy', 'cost_interpretation']
        missing_keys = [k for k in required_keys if k not in dataset_config]
        if missing_keys:
            raise ValueError(f"Dataset config for '{dataset_name}' missing required keys: {missing_keys}")

        return {
            'data_source': dataset_config['data_source'],
            'partitioning_strategy': dataset_config['partitioning_strategy'],
            'cost_interpretation': dataset_config['cost_interpretation'],
            'source_args': dataset_config.get('source_args', {}),
            'partitioner_args': dataset_config.get('partitioner_args', {}),
            'partition_scope': dataset_config.get('partition_scope', 'train') # Default to train scope
        }

    def _get_final_client_count(self, dataset_config: Dict, cost: Any) -> int:
        """Determines the effective number of clients for a given cost."""
        cli_override = self.config.num_clients
        num_clients = cli_override if cli_override is not None else dataset_config.get('default_num_clients', 2)

        if not isinstance(num_clients, int):
             raise TypeError(f"Client count must be an integer, got {num_clients}.")

        max_clients = dataset_config.get('max_clients')
        if max_clients is not None and num_clients > max_clients:
            print(f"Warning: Requested {num_clients} clients > max configured ({max_clients}). Using {max_clients}.")
            num_clients = max_clients

        partitioning_strategy = dataset_config.get('partitioning_strategy')
        cost_interpretation = dataset_config.get('cost_interpretation')

        # Adjust client count based on site mappings for pre_split strategies
        if partitioning_strategy == 'pre_split':
            site_mappings = dataset_config.get('source_args', {}).get('site_mappings')
            if site_mappings:
                # Translate the cost to get the key needed for the mapping
                translated_cost = translate_cost(cost, cost_interpretation)
                cost_key = translated_cost.get('key') # Assuming 'key' is used for site maps

                if cost_key is not None and cost_key in site_mappings:
                    # The number of available "sites" for this cost key defines the max clients
                    available_for_cost = len(site_mappings[cost_key])
                    if num_clients > available_for_cost:
                        print(f"Warning: Requested client count {num_clients} > available sites "
                              f"({available_for_cost}) for cost key '{cost_key}'. "
                              f"Using {available_for_cost}.")
                        num_clients = available_for_cost
                # else: # Cost key might not be in mapping (e.g., 'all' key mapping to specific sites)
                      # Or the cost interpretation doesn't yield a 'key'
                      # In these cases, num_clients isn't adjusted based on site_mappings here.
                      # print(f"Debug: Cost key '{cost_key}' not found in site_mappings or not applicable.")

        # Final check for validity
        if not isinstance(num_clients, int) or num_clients <= 0:
            raise ValueError(f"Calculated an invalid number of clients: {num_clients}")

        return num_clients

    def _load_source_data(self, data_source: str, dataset_name: str, source_args: Dict) -> Any:
        """Loads the source dataset using the appropriate function from data_loading."""
        loader_func = DATA_LOADERS.get(data_source)
        if loader_func is None:
             raise NotImplementedError(f"Data loader '{data_source}' not found in data_loading.DATA_LOADERS.")
        print(f"Calling data loader: {loader_func.__name__}")
        # Call the loader function
        return loader_func(
            dataset_name=dataset_name,
            data_dir=self.data_dir_root,
            source_args=source_args
        )

    def _partition_data(self,
                        partitioning_strategy: str,
                        data_to_partition: TorchDataset,
                        num_clients: int,
                        partition_args: Dict,
                        translated_cost: Dict,
                        sampling_config: Optional[Dict] = None
                       ) -> Optional[Dict[int, List[int]]]:
        """Partitions data using the appropriate function from data_partitioning."""
        partitioner_func = PARTITIONING_STRATEGIES.get(partitioning_strategy)
        if partitioner_func is None:
            raise NotImplementedError(f"Partitioning strategy '{partitioning_strategy}' not found in data_partitioning.PARTITIONING_STRATEGIES.")

        # Pre-split strategy returns None immediately
        if partitioning_strategy == 'pre_split':
            return None

        print(f"Calling partitioner: {partitioner_func.__name__}")
        # Prepare arguments for the partitioner function
        full_args = {
            'dataset': data_to_partition,
            'num_clients': num_clients,
            'seed': self.base_seed, # Use consistent base seed for partitioning
            **translated_cost,      # Add translated cost (e.g., alpha)
            **partition_args        # Add specific partitioner args from config
        }
        if sampling_config:
            full_args['sampling_config'] = sampling_config # Add sampling limits

        return partitioner_func(**full_args)

    def _load_client_specific_data(self,
                                   data_source: str,
                                   dataset_name: str,
                                   client_num: int,
                                   source_args: Dict,
                                   translated_cost: Dict,
                                   dataset_config: Dict # Pass full config
                                  ) -> Tuple[Any, Any]:
        """Loads data for a specific client (used by pre_split strategy)."""
        loader_func = DATA_LOADERS.get(data_source)
        if loader_func is None:
             raise NotImplementedError(f"Data loader '{data_source}' not found in data_loading.DATA_LOADERS.")

        # Determine the cost argument (key or suffix) needed by the loader
        cost_interpretation = dataset_config['cost_interpretation']
        cost_arg = None
        if cost_interpretation == 'site_mapping_key':
            cost_arg = translated_cost.get('key')
        elif cost_interpretation == 'file_suffix':
             cost_arg = translated_cost.get('suffix')
        # Add other interpretations if necessary

        if cost_arg is None:
            raise ValueError(f"Missing cost key/suffix ({cost_interpretation}) "
                             f"for pre_split loader {data_source}")

        print(f"Calling client loader: {loader_func.__name__} for client {client_num}, cost_key='{cost_arg}'")

        # Call the specific client loader function
        # Assuming these loaders expect 'cost_key' as the argument name
        return loader_func(
            dataset_name=dataset_name,
            data_dir=self.data_dir_root,
            client_num=client_num,
            cost_key=cost_arg, # Pass the determined key/suffix
            config=dataset_config # Pass the full config dict
        )

    def _prepare_client_data(self,
                             client_ids_list: List[str],
                             partitioning_strategy: str,
                             partition_result: Optional[Dict[int, List[int]]],
                             data_to_partition: Optional[TorchDataset],
                             dataset_name: str,
                             dataset_config: Dict,
                             translated_cost: Dict
                            ) -> Tuple[Dict[str, Any], str]:
        """
        Prepares the input data dictionary for the DataPreprocessor based on strategy.

        Returns:
            Tuple containing:
            - Dictionary: {client_id: data_for_client}, where data format depends on strategy.
            - String: The determined input type for the DataPreprocessor ('subset', 'xy_dict', 'path_dict').
        """
        client_input_data: Dict[str, Any] = {}
        preprocessor_input_type: str = 'unknown'

        if partitioning_strategy.endswith('_indices'):
            # Strategy produced indices, wrap data in Subsets
            preprocessor_input_type = 'subset'
            if partition_result is None:
                raise ValueError("Partition result is None for index-based partitioning strategy.")
            if data_to_partition is None:
                 raise ValueError("Base data to partition is None for index-based strategy.")

            for i, client_id in enumerate(client_ids_list):
                indices = partition_result.get(i, []) # Get indices for client i (0-based)
                if not indices:
                    print(f"Warning: Client {client_id} has no data after partitioning.")
                    # Store empty subset or skip? Store empty for now.
                client_input_data[client_id] = Subset(data_to_partition, indices)

        elif partitioning_strategy == 'pre_split':
            # Strategy requires loading data per client
            data_source = dataset_config['data_source']
            source_args = dataset_config.get('source_args', {})

            for i, client_id in enumerate(client_ids_list):
                client_num = i + 1 # Client number often 1-based for loading
                try:
                     # Load data specifically for this client
                     loaded_data = self._load_client_specific_data(
                         data_source, dataset_name, client_num, source_args,
                         translated_cost, dataset_config
                     )

                     # Expect loader to return (X, y) tuple
                     if isinstance(loaded_data, tuple) and len(loaded_data) == 2:
                         X, y = loaded_data
                         # Check if loader returned empty data (e.g., file not found for site)
                         if len(X) == 0:
                              print(f"Warning: No data loaded for client {client_id} (loader returned empty). Skipping client.")
                              continue # Skip this client entirely

                         # Store data in dictionary format
                         client_input_data[client_id] = {'X': X, 'y': y}

                         # Determine input type for preprocessor based on first valid client
                         if preprocessor_input_type == 'unknown':
                              if isinstance(X, np.ndarray) and isinstance(y, np.ndarray):
                                   preprocessor_input_type = 'xy_dict'
                              # Check if X contains strings (likely paths)
                              elif isinstance(X, (list, np.ndarray)) and len(X) > 0 and isinstance(X[0], str):
                                   preprocessor_input_type = 'path_dict'
                              else:
                                   # Fallback if type is ambiguous
                                   preprocessor_input_type = 'unknown'
                                   print(f"Warning: Could not determine preprocessor input type from data types: "
                                         f"X={type(X)}, y={type(y)}")
                     else:
                         # Loader function returned unexpected format
                         print(f"Warning: Client data loader {data_source} returned unexpected format "
                               f"for client {client_id}: {type(loaded_data)}. Skipping client.")
                         continue # Skip this client

                except FileNotFoundError as e:
                     print(f"Skipping client {client_id} due to FileNotFoundError: {e}")
                     continue # Skip this client
                except Exception as e:
                     print(f"Skipping client {client_id} due to error during data loading: {e}")
                     traceback.print_exc()
                     continue # Skip this client

            # Check if any data was loaded successfully
            if not client_input_data:
                print(f"Warning: Failed to load data for ANY pre-split clients for cost {translated_cost}.")
                return {}, 'unknown' # Return empty dict and unknown type

            if preprocessor_input_type == 'unknown':
                # This shouldn't happen if at least one client loaded data successfully
                raise RuntimeError("Could not determine preprocessor input type for pre-split data.")

        else:
            # Handle other potential future strategies
            raise NotImplementedError(f"Data preparation logic not implemented for strategy: {partitioning_strategy}")

        return client_input_data, preprocessor_input_type


    # --- Model, Server, and Training Helpers ---
    def _create_trainer_config(self,
                               server_type: str,
                               learning_rate: float,
                               algorithm_params: Optional[Dict] = None
                              ) -> TrainerConfig:
        """Creates the TrainerConfig dataclass for server initialization."""
        if algorithm_params is None:
            algorithm_params = {}

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

    def _create_model(self, cost: Any, learning_rate: float) -> Tuple[nn.Module, nn.Module, torch.optim.Optimizer]:
        """Creates model instance, criterion, and optimizer."""
        model_name = self.config.dataset # Assumes model name matches dataset name
        fixed_classes = self.default_params.get('fixed_classes') # Get class count if defined

        # Check if the model class exists in the models module (ms)
        if not hasattr(ms, model_name):
             raise ValueError(f"Model class '{model_name}' not found in models.py.")

        model_class = getattr(ms, model_name)
        # Instantiate the model (add arguments like num_classes if needed by model's __init__)
        try:
             # Example: Pass num_classes if the model expects it
             if fixed_classes is not None and model_name in ['ModelExpectingClasses']: # Add actual model names here
                 model = model_class(num_classes=fixed_classes)
             else:
                  model = model_class()
        except TypeError as e:
             print(f"Error instantiating model {model_name}. Check __init__ signature. Error: {e}")
             raise

        # Define criterion based on dataset type/metric
        # TODO: Make this more robust, perhaps based on config 'metric' or 'task_type'
        criterion_map = {
            'Synthetic': nn.CrossEntropyLoss(),
            'Credit': nn.CrossEntropyLoss(),
            'EMNIST': nn.CrossEntropyLoss(),
            'CIFAR': nn.CrossEntropyLoss(),
            'IXITiny': get_dice_score, # Assumes get_dice_score behaves like a loss function
            'ISIC': nn.CrossEntropyLoss(),
            'Heart': nn.CrossEntropyLoss()
        }
        criterion = criterion_map.get(self.config.dataset)
        if criterion is None:
            raise ValueError(f"Criterion not defined for dataset {self.config.dataset} in pipeline._create_model")

        # Create optimizer
        # Consider making optimizer type and kwargs configurable
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            amsgrad=True,
            weight_decay=1e-4 # Example weight decay
        )
        return model, criterion, optimizer

    def _create_server_instance(self,
                                cost: Any,
                                server_type: str,
                                config: TrainerConfig,
                                tuning: bool
                               ) -> Server: # Return base Server type hint
        """Creates and configures a server instance."""
        model, criterion, optimizer = self._create_model(cost, config.learning_rate)

        # Create the initial global model state
        globalmodelstate = ModelState(
            model=model,
            optimizer=optimizer,
            criterion=criterion
        )

        # Map server type string to server class
        server_mapping = {
            'local': Server,
            'fedavg': FedAvgServer
            # Add mappings for other implemented servers like FedProx, Ditto
        }
        if server_type not in server_mapping:
            raise ValueError(f"Unsupported server type specified: {server_type}")

        server_class = server_mapping[server_type]

        # Instantiate the server
        server = server_class(config=config, globalmodelstate=globalmodelstate)
        server.set_server_type(server_type, tuning) # Inform server if it's a tuning run

        return server

    def _add_clients_to_server(self, server: Server, client_dataloaders: Dict):
        """Adds client data to the server instance."""
        print(f"Adding {len(client_dataloaders)} clients to {server.server_type} server...")
        for client_id, loaders in client_dataloaders.items():
             # Ensure loaders is a tuple/list of (train, val, test)
             if not isinstance(loaders, (tuple, list)) or len(loaders) != 3:
                  print(f"Warning: Invalid loader format for client {client_id}. Skipping.")
                  continue

             train_loader, val_loader, test_loader = loaders

             # Skip clients if they have no training data
             # Check DataLoader length or underlying dataset length
             if train_loader is None or len(train_loader.dataset) == 0:
                 print(f"Skipping client {client_id}: no training data.")
                 continue

             # Create SiteData object
             clientdata = self._create_site_data(client_id, loaders)
             # Add client to the server
             server.add_client(clientdata=clientdata)
        print(f"Successfully added {len(server.clients)} clients.")


    def _create_site_data(self, client_id: str, loaders: Tuple[DataLoader, DataLoader, DataLoader]) -> SiteData:
        """Creates SiteData structure for a client."""
        return SiteData(
            site_id=client_id,
            train_loader=loaders[0],
            val_loader=loaders[1],
            test_loader=loaders[2]
        )

    def _train_and_evaluate(self, server: Server, rounds: int, cost: Any, seed: int) -> Dict:
        """
        Runs the training and evaluation loop for a given server instance.

        Args:
            server: The server instance (e.g., FedAvgServer).
            rounds: The number of communication rounds to run.
            cost: The current cost parameter (for potential use in saving).
            seed: The current run seed (for potential use in saving).

        Returns:
            A dictionary containing the collected metrics.
        """
        run_type = '(tuning)' if server.tuning else '(eval)'
        print(f"Starting training {run_type} for {server.server_type} over {rounds} rounds...")

        for round_num in range(rounds):
            # Perform one round of training and validation
            train_loss, val_loss, val_score = server.train_round()

            # Save intermediate model state (e.g., after round 1 for FedAvg)
            if not server.tuning and server.server_type == 'fedavg' and round_num == 0:
                self._save_intermediate_model_state(server, cost, seed, model_type='round1')

            # Optional progress print
            if (round_num + 1) % 20 == 0 or round_num == 0 or (round_num + 1) == rounds:
                print(f"  R {round_num+1}/{rounds} - TrainLoss: {train_loss:.4f}, ValLoss: {val_loss:.4f}, ValScore: {val_score:.4f}")

        # --- Final Evaluation on Test Set (after all rounds) ---
        # Only perform test evaluation if this is not a tuning run
        if not server.tuning:
            print(f"Performing final test evaluation for {server.server_type}...")
            server.test_global() # Server handles testing on clients' test loaders

        # --- Collect Metrics ---
        global_state = server.serverstate
        # Determine which metrics to collect (validation or test)
        metrics_key = 'val' if server.tuning else 'test'

        # Initialize metrics dictionary
        metrics: Dict[str, Dict] = {
            'global': {
                'losses': getattr(global_state, f"{metrics_key}_losses", []), # Use getattr safely
                'scores': getattr(global_state, f"{metrics_key}_scores", [])
            },
            'sites': {}
        }

        # Collect metrics for each client
        for client_id, client in server.clients.items():
            # Use personal state if available and required, otherwise global state
            state_to_log = client.personal_state if server.config.requires_personal_model and client.personal_state else client.global_state
            metrics['sites'][client_id] = {
                'losses': getattr(state_to_log, f"{metrics_key}_losses", []),
                'scores': getattr(state_to_log, f"{metrics_key}_scores", [])
            }

        # Report final score
        final_scores = metrics['global']['scores']
        final_global_score = final_scores[-1] if final_scores else None
        score_type = metrics_key.capitalize()
        print(f"Finished {server.server_type}. Final {score_type} Score: {final_global_score if final_global_score is not None else 'N/A':.4f}")

        return metrics

    # --- Model Saving Helpers ---
    def _save_initial_model_state(self, server: Server, num_clients_run: int, cost: Any, seed: int):
         """Saves the initial model state (before training) for FedAvg during evaluation runs."""
         if not server.tuning and server.server_type == 'fedavg':
              print(f"  Saving INITIAL FedAvg model state...")
              self.results_manager.save_model_state(
                  model_state_dict=server.serverstate.model.state_dict(),
                  experiment_type=ExperimentType.EVALUATION,
                  dataset=self.config.dataset,
                  num_clients_run=num_clients_run,
                  cost=cost,
                  seed=seed,
                  server_type='fedavg',
                  model_type='initial'
              )

    def _save_intermediate_model_state(self, server: Server, cost: Any, seed: int, model_type: str):
        """Saves intermediate model states during evaluation (e.g., after round 1)."""
        # Assumes num_clients_for_run has been set correctly in _initialize_experiment
        if self.num_clients_for_run is None:
             print("Warning: num_clients_for_run not set. Cannot save intermediate model.")
             return

        print(f"  Saving FedAvg model state after {model_type}...")
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

        if fedavg_server is not None:
            print(f"  Saving FINAL FedAvg models for cost {cost}, seed {seed}...")

            # Save Final Model (state at the end of training)
            self.results_manager.save_model_state(
                model_state_dict=fedavg_server.serverstate.model.state_dict(),
                experiment_type=ExperimentType.EVALUATION,
                dataset=self.config.dataset,
                num_clients_run=num_clients_run,
                cost=cost,
                seed=seed,
                server_type='fedavg',
                model_type='final'
            )

            # Save Best Performing Model (based on validation score during training)
            if fedavg_server.serverstate.best_model:
                self.results_manager.save_model_state(
                    model_state_dict=fedavg_server.serverstate.best_model.state_dict(),
                    experiment_type=ExperimentType.EVALUATION,
                    dataset=self.config.dataset,
                    num_clients_run=num_clients_run,
                    cost=cost,
                    seed=seed,
                    server_type='fedavg',
                    model_type='best'
                )
            else:
                print("  Warning: FedAvg server did not record a best_model state, not saved.")
        elif 'fedavg' in trained_servers: # Check if FedAvg ran but maybe failed (server is None)
             print("  Warning: FedAvg server instance not available for model saving (likely due to an earlier error).")
        # else: FedAvg was not part of the ALGORITHMS list for this run