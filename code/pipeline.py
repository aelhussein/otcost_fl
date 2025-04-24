# pipeline.py
"""
Orchestrates FL experiments: manages loops over runs and costs,
delegates single trial execution, aggregates results into records,
and handles saving models/results. Streamlined version.
"""
import traceback
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple, Any, Union, Callable

# Import project modules
from configs import ROOT_DIR, ALGORITHMS, DEVICE, DEFAULT_PARAMS, DATA_DIR
from helper import (set_seeds, get_parameters_for_dataset, # Keep necessary imports
                    get_default_lr, get_default_reg, MetricKey, SiteData, ModelState, TrainerConfig) # Import types from helper
# Import necessary components
from servers import Server, FedAvgServer, FedProxServer, PFedMeServer, DittoServer # Import all server types
import models as ms
from data_processing import DataManager
from results_manager import ResultsManager, ExperimentType, TrialRecord # Import TrialRecord

# =============================================================================
# == Experiment Configuration ==
# =============================================================================
@dataclass
class ExperimentConfig:
    """Configuration for an experiment run."""
    dataset: str
    experiment_type: str
    num_clients: Optional[int] = None # Can be overridden by CLI

# =============================================================================
# == Single Run Executor Class ==
# =============================================================================
class SingleRunExecutor:
    """
    Handles the execution of a single Federated Learning run/trial.
    Creates model, server, clients, runs training/evaluation loop.
    Returns metrics history and model state dictionaries.
    """
    def __init__(self, dataset_name: str, default_params: Dict, device: torch.device):
        self.dataset_name = dataset_name
        self.default_params = default_params
        self.device = device # Target device for client computation

    def _create_model(self) -> Tuple[nn.Module, Union[nn.Module, Callable]]:
        """Creates model instance and criterion (on CPU initially)."""
        model_name = self.dataset_name
        model_name_actual = 'Synthetic' if 'Synthetic_' in model_name else model_name
        model_class = getattr(ms, model_name_actual, None)
        if model_class is None: raise ValueError(f"Model class '{model_name_actual}' not found.")
        model = model_class()

        metric_name = self.default_params.get('metric', 'Accuracy').upper()
        fixed_classes = self.default_params.get('fixed_classes')
        criterion: Union[nn.Module, Callable]

        if metric_name == 'DICE':
            from helper import get_dice_score; criterion = get_dice_score
        elif metric_name in ['ACCURACY', 'F1', 'BALANCED_ACCURACY']:
            if fixed_classes and fixed_classes > 1: criterion = nn.CrossEntropyLoss()
            else: raise NotImplementedError(f"Criterion mapping failed: {metric_name}/{fixed_classes}")
        else: raise ValueError(f"Criterion not defined for metric {metric_name}")

        return model, criterion

    def _create_trainer_config(self, server_type: str, hyperparams: Dict, tuning: bool) -> TrainerConfig:
        """Creates the TrainerConfig."""
        lr = hyperparams.get('learning_rate')
        if lr is None: raise ValueError("Learning rate missing for TrainerConfig.")
        reg = hyperparams.get('reg_param')
        algo_params = {'reg_param': reg} if server_type in ['fedprox', 'pfedme', 'ditto'] and reg is not None else {}
        requires_personal_model = server_type in ['pfedme', 'ditto'] # Info for client init
        rounds = self.default_params.get('rounds_tune_inner') if tuning else self.default_params['rounds']

        return TrainerConfig(
            dataset_name=self.dataset_name, device=str(self.device), learning_rate=lr,
            batch_size=self.default_params['batch_size'], epochs=self.default_params.get('epochs_per_round', 1),
            rounds=rounds, requires_personal_model=requires_personal_model, algorithm_params=algo_params
        )

    def _create_server_instance(self, server_type: str, config: TrainerConfig, tuning: bool) -> Server:
        """Creates server instance with model/criterion on CPU."""
        model, criterion = self._create_model()
        # Initial global state for server (model is CPU)
        globalmodelstate = ModelState(model=model.cpu(), criterion=criterion)

        server_mapping: Dict[str, type[Server]] = {
            'local': Server, # Assuming 'local' uses base Server or similar
            'fedavg': FedAvgServer,
            'fedprox': FedProxServer,
            'pfedme': PFedMeServer,
            'ditto': DittoServer
        }
        server_class = server_mapping.get(server_type)
        if server_class is None: raise ValueError(f"Unsupported server type: {server_type}")

        server = server_class(config=config, globalmodelstate=globalmodelstate)
        server.set_server_type(server_type, tuning)
        return server

    def _add_clients_to_server(self, server: Server, client_dataloaders: Dict) -> int:
        """Adds client data to the server. Server handles Client instantiation."""
        added_count = 0
        for client_id, loaders in client_dataloaders.items():
            try:
                train_loader, val_loader, test_loader = loaders
                clientdata = SiteData(site_id=client_id, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader)
                server.add_client(clientdata=clientdata)
                added_count += 1
            except Exception as e: print(f"ERROR adding client {client_id}: {e}") # Keep essential error logging
        return added_count

    def _train_and_evaluate(self, server: Server, rounds: int) -> Dict:
        """Runs the FL training/evaluation loop via server methods."""
        if not server.clients: return {'error': 'No clients available'}
        # Server internally manages history dict
        for round_num in range(rounds):
            try: server.train_round()
            except Exception as e:
                 metrics = getattr(server, 'history', {})
                 metrics['error'] = f"Error in train_round {round_num+1}: {e}"
                 return metrics # Return history accumulated so far + error
        if not server.tuning:
            try: server.test_global()
            except Exception as e:
                 metrics = getattr(server, 'history', {})
                 metrics['error_test'] = f'Error in test_global: {e}'
                 return metrics
        # Return final history accumulated by the server
        return getattr(server, 'history', {'error': 'Server history attribute missing.'})

    def execute_trial(self,
                      server_type: str,
                      hyperparams: Dict,
                      client_dataloaders: Dict,
                      tuning: bool
                     ) -> Tuple[Dict, Dict[str, Optional[Dict]]]:
        """
        Executes a single FL trial. Returns metrics history and model states.
        """
        server: Optional[Server] = None
        metrics: Dict = {}
        model_states: Dict[str, Optional[Dict]] = {'final': None, 'best': None, 'round0': None, 'error': None}

        try:
            trainer_config = self._create_trainer_config(server_type, hyperparams, tuning)
            # Pass tuning flag, learning rate implicitly handled by client via config
            server = self._create_server_instance(server_type, trainer_config, tuning)
            actual_clients_added = self._add_clients_to_server(server, client_dataloaders)
            if actual_clients_added == 0:
                metrics = {'error': 'No clients successfully added to server.'}
            else:
                metrics = self._train_and_evaluate(server, trainer_config.rounds)
                # Retrieve model states from server if trial succeeded
                if 'error' not in metrics:
                    model_states['final'] = server.serverstate.get_current_model_state_dict()
                    model_states['best'] = server.get_best_model_state_dict()
                    model_states['round0'] = server.round_0_state_dict

            del server # Explicit cleanup
            server = None

        except Exception as e:
            err_msg = f"Executor setup/run failed: {e}"
            print(err_msg); traceback.print_exc() # Keep traceback for executor errors
            metrics['error'] = err_msg
            model_states['error'] = err_msg # Also store error marker with states
            if server: del server

        return metrics, model_states

# =============================================================================
# == Experiment Orchestrator Class ==
# =============================================================================
@dataclass
class RunMetadata:
    """Metadata for a single experiment run."""
    run_idx_total: int; seed_used: int
    cost_client_counts: Dict[Any, int]
    dataset_name: str = ""; num_target_clients: int = 0

@dataclass
class CostExecutionResult:
    """Results from executing all trials for a specific cost within one run."""
    cost: Any
    trial_records: List[TrialRecord]
    # No longer need to pass server state dicts from here

class Experiment:
    """Orchestrates FL experiments using a simplified, flatter structure."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.default_params = get_parameters_for_dataset(config.dataset)
        self.base_seed = self.default_params.get('base_seed', 42)
        self.num_target_clients = config.num_clients or self.default_params.get('default_num_clients', 2)
        # Initialize managers
        self.results_manager = ResultsManager(ROOT_DIR, config.dataset, self.num_target_clients)
        self.data_manager = DataManager(config.dataset, self.base_seed, DATA_DIR)
        self.single_run_executor = SingleRunExecutor(config.dataset, self.default_params, DEVICE)
        self.all_trial_records: List[TrialRecord] = [] # Main result storage

    def run_experiment(self, costs: List[Any]) -> List[TrialRecord]:
        """Main entry point."""
        experiment_type = self.config.experiment_type

        if experiment_type == ExperimentType.EVALUATION:
            self._execute_experiment_runs(experiment_type, costs, self._evaluate_cost_for_run)
        elif experiment_type in [ExperimentType.LEARNING_RATE, ExperimentType.REG_PARAM]:
            self._execute_experiment_runs(experiment_type, costs, self._tune_cost_for_run)
        else:
            raise ValueError(f"Unsupported experiment type: {experiment_type}")

        print(f"\n=== Experiment Orchestrator Finished ({experiment_type}) ===")
        return self.all_trial_records

    def _execute_experiment_runs(self, experiment_type: str, costs: List[Any], cost_execution_func: Callable):
        """Generic loop structure for executing multiple runs and costs."""
        self.all_trial_records, remaining_costs, completed_runs = \
            self.results_manager.get_experiment_status(experiment_type, costs, self.default_params, MetricKey)

        is_tuning = experiment_type != ExperimentType.EVALUATION
        target_runs_key = 'runs_tune' if is_tuning else 'runs'
        target_runs = self.default_params.get(target_runs_key, 1)

        if not remaining_costs and completed_runs >= target_runs: return # Already done

        remaining_runs_count = target_runs - completed_runs
        if remaining_runs_count <= 0 and remaining_costs:
             remaining_runs_count, completed_runs = target_runs, 0

        print(f"Orchestrator: Starting {remaining_runs_count} run(s) for '{experiment_type}' (Runs {completed_runs + 1} to {target_runs})...")

        # --- Run Loop ---
        for run_offset in range(remaining_runs_count):
            current_run_idx = completed_runs + run_offset # 0-based index
            current_seed = self.base_seed + current_run_idx
            print(f"\n--- Run {current_run_idx + 1}/{target_runs} (Seed: {current_seed}) ---")
            set_seeds(current_seed)

            run_records: List[TrialRecord] = []
            run_cost_client_counts: Dict[Any, int] = {}

            costs_to_run_this_iter = remaining_costs if completed_runs == 0 else costs

            # --- Cost Loop ---
            for cost in costs_to_run_this_iter:
                print(f"  Cost: {cost}")
                cost_results = None; num_actual_clients = 0
                try:
                    client_dataloaders = self.data_manager.get_dataloaders(
                        cost=cost, run_seed=current_seed, num_clients_override=self.config.num_clients
                    )
                    if not client_dataloaders: raise RuntimeError("No dataloaders.")
                    num_actual_clients = len(client_dataloaders)
                    run_cost_client_counts[cost] = num_actual_clients
                    # Execute trials for this cost
                    cost_results = cost_execution_func(cost, current_run_idx, current_seed, client_dataloaders, num_actual_clients)

                except Exception as e: # Catch errors in dataloading or cost execution
                     print(f"  ERROR processing cost {cost} in run {current_run_idx + 1}: {e}")
                     run_records.append(TrialRecord(cost=cost, run_idx=current_run_idx, server_type="N/A", error=f"Cost processing error: {e}"))
                     if num_actual_clients > 0: run_cost_client_counts[cost] = num_actual_clients # Record client count if possible

                if cost_results and cost_results.trial_records:
                    run_records.extend(cost_results.trial_records)
            # --- End Cost Loop ---

            self.all_trial_records.extend(run_records) # Add records from this run

            # Save aggregated results after each full run
            run_meta = RunMetadata(run_idx_total=current_run_idx + 1, seed_used=current_seed,
                                   cost_client_counts=run_cost_client_counts, dataset_name=self.config.dataset,
                                   num_target_clients=self.num_target_clients)
            self.results_manager.save_results(self.all_trial_records, experiment_type, vars(run_meta))
        # --- End Run Loop ---

    # --- Cost Processing Helpers ---

    def _tune_cost_for_run(self, cost: Any, run_idx: int, seed: int,
                           client_dataloaders: Dict, num_actual_clients: int
                          ) -> CostExecutionResult:
        """Executes all tuning trials for a single cost and run."""
        tuning_type = self.config.experiment_type
        trial_records: List[TrialRecord] = []

        # Determine tuning parameters
        if tuning_type == ExperimentType.LEARNING_RATE: param_key, try_vals_key, servers_key = 'learning_rate', 'learning_rates_try', 'servers_tune_lr'
        elif tuning_type == ExperimentType.REG_PARAM: param_key, try_vals_key, servers_key = 'reg_param', 'reg_params_try', 'servers_tune_reg'
        else: raise ValueError(f"Invalid tuning type: {tuning_type}")

        fixed_key = 'reg_param' if param_key == 'learning_rate' else 'learning_rate'
        fixed_val_func = get_default_reg if param_key == 'learning_rate' else get_default_lr
        fixed_val = fixed_val_func(self.config.dataset)
        try_vals = self.default_params.get(try_vals_key, [])
        servers_to_tune = self.default_params.get(servers_key, [])
        if not try_vals or not servers_to_tune: return CostExecutionResult(cost=cost, trial_records=[])

        # Loop over HPs and Servers
        for param_val in try_vals:
            hp = {param_key: param_val, fixed_key: fixed_val}
            for server_type in servers_to_tune:
                trial_metrics, _ = self.single_run_executor.execute_trial( # Ignore model states for tuning
                    server_type=server_type, hyperparams=hp,
                    client_dataloaders=client_dataloaders, tuning=True
                )
                record = TrialRecord(cost=cost, run_idx=run_idx, server_type=server_type,
                                     tuning_param_name=param_key, tuning_param_value=param_val,
                                     metrics=trial_metrics, error=trial_metrics.get('error'))
                trial_records.append(record)
        return CostExecutionResult(cost=cost, trial_records=trial_records)

    def _evaluate_cost_for_run(self, cost: Any, run_idx: int, seed: int,
                               client_dataloaders: Dict, num_actual_clients: int
                              ) -> CostExecutionResult:
        """Executes all evaluation trials (servers) for a single cost and run."""
        trial_records: List[TrialRecord] = []

        for server_type in ALGORITHMS: # Use global ALGORITHMS list
            # Fetch best HPs
            best_lr = self.results_manager.get_best_parameters(ExperimentType.LEARNING_RATE, server_type, cost)
            best_reg = self.results_manager.get_best_parameters(ExperimentType.REG_PARAM, server_type, cost)
            # Use defaults if not found
            if best_lr is None: best_lr = get_default_lr(self.config.dataset)
            if best_reg is None and server_type in ['fedprox', 'pfedme', 'ditto']: best_reg = get_default_reg(self.config.dataset)
            eval_hyperparams = {'learning_rate': best_lr, 'reg_param': best_reg}

            # Execute trial
            trial_metrics, model_states = self.single_run_executor.execute_trial(
                server_type=server_type, hyperparams=eval_hyperparams,
                client_dataloaders=client_dataloaders, tuning=False
            )
            # Create record
            record = TrialRecord(cost=cost, run_idx=run_idx, server_type=server_type,
                                 metrics=trial_metrics, error=trial_metrics.get('error') or model_states.get('error'))
            trial_records.append(record)

            # Save models immediately if successful (only FedAvg for now)
            if record.error is None and server_type == 'fedavg':
                for model_type, state_dict in model_states.items():
                    if model_type != 'error' and state_dict is not None:
                         self.results_manager.save_model_state(
                             state_dict, num_actual_clients, cost, seed, 'fedavg', model_type)
                         print(f"{model_type} model saved successfully for {server_type}, Cost: {cost}, Run: {run_idx}, Seed: {seed}")

        # Return records, no need to return state dicts from here
        return CostExecutionResult(cost=cost, trial_records=trial_records)