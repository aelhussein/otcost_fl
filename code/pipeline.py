"""
Main experiment pipeline orchestration.
Initializes data, creates models/servers, runs training/evaluation loops.
"""
import os
import sys
import traceback
import torch
import torch.nn as nn
from torch.utils.data import Subset, ConcatDataset

# Import project modules
from configs import * # Import all config constants (paths, params, etc.)
from helper import ( # Import necessary helper functions
    set_seeds, cleanup_gpu, get_dice_score, get_parameters_for_dataset,
    get_default_lr, get_default_reg, translate_cost
)
from servers import Server, FedAvgServer, TrainerConfig, SiteData, ModelState # Import server types and data structures
import models as ms # Import model architectures module
from losses import * # Import custom losses if any

# Import newly separated modules
from data_loading import DATA_LOADERS
from data_partitioning import PARTITIONING_STRATEGIES
from data_processing import DataPreprocessor # Import the preprocessor class
from results_manager import ResultsManager # Import the results manager

# Define Experiment types (or import from results_manager)
class ExperimentType:
    LEARNING_RATE = 'learning_rate'; REG_PARAM = 'reg_param'; EVALUATION = 'evaluation'; DIVERSITY = 'diversity'

@dataclass
class ExperimentConfig:
    dataset: str
    experiment_type: str
    num_clients: Optional[int] = None # Can be overridden by CLI

# --- Main Experiment Class ---
class Experiment:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.default_params = get_parameters_for_dataset(self.config.dataset)
        self.data_dir_root = DATA_DIR
        self.base_seed = self.default_params.get('base_seed', 42)
        print(f"Using base seed: {self.base_seed} for experiment runs.")

        # Determine initial target client count for filenames
        initial_target_num_clients = self.config.num_clients or self.default_params.get('default_num_clients', 2)
        if not isinstance(initial_target_num_clients, int) or initial_target_num_clients <= 0:
             raise ValueError(f"Invalid initial target client count: {initial_target_num_clients}")

        print(f"Initializing ResultsManager for {initial_target_num_clients} target clients (filename).")
        self.results_manager = ResultsManager(
            root_dir=ROOT_DIR, dataset=self.config.dataset,
            experiment_type=self.config.experiment_type,
            num_clients=initial_target_num_clients
        )
        self.num_clients_for_run = None # Actual number used in a run

    def run_experiment(self, costs):
        """Main entry point to run the configured experiment type."""
        if self.config.experiment_type == ExperimentType.EVALUATION:
            return self._run_final_evaluation(costs)
        elif self.config.experiment_type in [ExperimentType.LEARNING_RATE, ExperimentType.REG_PARAM]:
            return self._run_hyperparam_tuning(costs)
        else:
            raise ValueError(f"Unsupported experiment type: {self.config.experiment_type}")

    def _check_existing_results(self, costs):
        """Checks existing results, determines remaining runs/costs."""
        # (Keep function code as provided in the previous pipeline.py)
        # Uses self.results_manager methods
        results, metadata = self.results_manager.load_results(self.config.experiment_type)
        runs_key = 'runs_tune' if self.config.experiment_type != ExperimentType.EVALUATION else 'runs'
        target_runs = self.default_params.get(runs_key, 1); remaining_costs = list(costs); completed_runs = 0
        if results is not None:
            has_errors = metadata is not None and metadata.get('contains_errors', False)
            if has_errors: print(f"Prev results have errors. Re-running all."); return results, remaining_costs, 0
            completed_costs = set(results.keys()); remaining_costs = [c for c in costs if c not in completed_costs]
            if completed_costs:
                first_cost = next(iter(completed_costs));
                try:
                    first_param = next(iter(results[first_cost]))
                    first_server = next(iter(results[first_cost][first_param]))
                    loss_key = 'val_losses' if self.config.experiment_type != ExperimentType.EVALUATION else 'test_losses'
                    if first_server in results[first_cost][first_param] and 'global' in results[first_cost][first_param][first_server] and loss_key in results[first_cost][first_param][first_server]['global']:
                        # Check if it's a list of lists (multi-run) or list (single run)
                         if results[first_cost][first_param][first_server]['global'][loss_key] and isinstance(results[first_cost][first_param][first_server]['global'][loss_key][0], list):
                              completed_runs = len(results[first_cost][first_param][first_server]['global'][loss_key])
                         else: # Assume single run stored if not list of lists
                              completed_runs = 1 # Or 0 if empty? Check structure carefully. Let's assume 1 if exists.
                    else: completed_runs = 0
                    for cost_key in completed_costs:
                        if self.results_manager._check_for_errors_in_results(results[cost_key]): print(f"Errors found for cost {cost_key}. Re-running."); remaining_costs.append(cost_key)
                except (StopIteration, KeyError, IndexError, TypeError, AttributeError) as e: completed_runs = 0; print(f"Could not determine completed runs: {e}")
            remaining_costs = sorted(list(set(remaining_costs))) # Sort for consistent order
            print(f"Found {completed_runs}/{target_runs} completed valid runs in existing results.")
            if completed_runs >= target_runs and not remaining_costs: remaining_costs = []
            elif completed_runs < target_runs: remaining_costs = list(costs); print(f"Runs incomplete. Will run all costs.") # Rerun all costs if runs incomplete
        else: print("No existing results found.")
        print(f"Remaining costs to process: {remaining_costs}")
        return results, remaining_costs, completed_runs

    def _run_hyperparam_tuning(self, costs):
        """Runs hyperparameter tuning loops."""
        # (Keep overall structure, logic for loops over runs, costs, params)
        # Uses self._initialize_experiment, self._hyperparameter_tuning
        results, remaining_costs, completed_runs = self._check_existing_results(costs)
        runs_tune = self.default_params.get('runs_tune', 1)
        if not remaining_costs and completed_runs >= runs_tune: print("Hyperparameter tuning already complete."); return results if results is not None else {}

        remaining_runs_count = runs_tune - completed_runs
        if remaining_runs_count <= 0 and remaining_costs: print(f"Runs complete, but costs {remaining_costs} missing. Rerunning missing for {runs_tune} runs."); remaining_runs_count = runs_tune; completed_runs = 0
        elif remaining_runs_count <= 0: print("Target tuning runs already completed."); return results if results is not None else {}

        print(f"Starting {remaining_runs_count} hyperparameter tuning run(s)..."); results = results or {}
        num_clients_metadata = None # Store the client count used for metadata saving

        for run_idx in range(remaining_runs_count):
            current_run_total = completed_runs + run_idx + 1; current_seed = self.base_seed + run_idx # Consistent seeding
            print(f"\n--- Starting Tuning Run {current_run_total}/{runs_tune} (Seed: {current_seed}) ---")
            set_seeds(current_seed); run_meta = {'run_number': current_run_total, 'seed_used': current_seed}; cost_client_counts = {}

            for cost in remaining_costs: # Process only remaining/incomplete costs for this run
                print(f"\n--- Processing Cost: {cost} (Run {current_run_total}) ---")
                try: num_clients_this_cost = self._get_final_client_count(self.default_params, cost); cost_client_counts[cost] = num_clients_this_cost; num_clients_metadata = num_clients_metadata or num_clients_this_cost
                except Exception as e: print(f"ERROR determining client count for cost {cost}: {e}. Skipping cost."); continue

                # Determine params to tune based on experiment type
                if self.config.experiment_type == ExperimentType.LEARNING_RATE:
                     param_key, fixed_param_key, fixed_param_value, params_to_try_values = 'learning_rate', 'reg_param', get_default_reg(self.config.dataset), self.default_params.get('learning_rates_try', [])
                     server_types_to_tune = self.default_params.get('servers_tune_lr', ALGORITHMS) # Tune all by default if not specified
                elif self.config.experiment_type == ExperimentType.REG_PARAM:
                     param_key, fixed_param_key, fixed_param_value, params_to_try_values = 'reg_param', 'learning_rate', get_default_lr(self.config.dataset), self.default_params.get('reg_params_try', [])
                     server_types_to_tune = self.default_params.get('servers_tune_reg', []) # Only tune specified servers for reg param
                else: raise ValueError(f"Unsupported tuning type: {self.config.experiment_type}")

                if not params_to_try_values: print(f"Warn: No parameters specified to try for {self.config.experiment_type}, cost {cost}. Skipping."); continue

                hyperparams_list = [{param_key: p_val, fixed_param_key: fixed_param_value} for p_val in params_to_try_values]
                tuning_results_for_cost = {}
                for hyperparams in hyperparams_list:
                    param_value_being_tuned = hyperparams[param_key]; print(f"--- Tuning Param: {param_key}={param_value_being_tuned} ---")
                    server_metrics = self._hyperparameter_tuning(cost, hyperparams, server_types_to_tune) # Pass needed info
                    tuning_results_for_cost[param_value_being_tuned] = server_metrics

                # Aggregate results for this cost
                if cost not in results: results[cost] = {}
                for param_val, server_data in tuning_results_for_cost.items():
                    if param_val not in results[cost]: results[cost][param_val] = {}
                    for server_type, metrics in server_data.items():
                        if server_type not in results[cost][param_val]: results[cost][param_val][server_type] = None
                        results[cost][param_val][server_type] = self.results_manager.append_or_create_metric_lists(results[cost][param_val][server_type], metrics)

            # Save results after each tuning run completes for all costs
            print(f"--- Completed Tuning Run {current_run_total}/{runs_tune} ---")
            run_meta['cost_specific_client_counts'] = cost_client_counts
            self.results_manager.save_results(results, self.config.experiment_type, client_count=num_clients_metadata, run_metadata=run_meta)

        return results

    def _hyperparameter_tuning(self, cost, hyperparams, server_types):
        """Internal helper for tuning one hyperparameter setting."""
        # (Keep function logic: init data, loop servers, create config/server, train, track results)
        # Uses self._initialize_experiment, self._create_trainer_config, _create_server_instance, _add_clients_to_server, _train_and_evaluate
        try: client_dataloaders = self._initialize_experiment(cost)
        except Exception as e: error_msg = f"Initialization failed for cost {cost}: {e}"; print(f"ERROR: {error_msg}"); traceback.print_exc(); return {'error': error_msg}
        if not client_dataloaders: return {'error': f"No client dataloaders for cost {cost}."}

        tracking = {}
        for server_type in server_types:
            print(f"..... Tuning Server: {server_type} .....")
            lr = hyperparams.get('learning_rate'); reg_param_val = hyperparams.get('reg_param')
            algo_params_dict = {'reg_param': reg_param_val} if server_type in ['pfedme', 'ditto'] and reg_param_val is not None else {}
            trainer_config = self._create_trainer_config(server_type, lr, algo_params_dict)
            # Use fewer rounds for tuning if specified
            num_tuning_rounds = self.default_params.get('rounds_tune_inner', trainer_config.rounds)
            trainer_config.rounds = num_tuning_rounds # Adjust rounds for tuning run

            server = None
            try:
                server = self._create_server_instance(cost, server_type, trainer_config, tuning=True) # Pass tuning=True
                self._add_clients_to_server(server, client_dataloaders)
                if not server.clients: tracking[server_type] = {'error': 'No clients added'}; continue
                metrics = self._train_and_evaluate(server, trainer_config.rounds, cost, self.base_seed) # Seed not strictly needed here if set outside
                tracking[server_type] = metrics
            except Exception as e: error_msg = str(e); print(f"ERROR tuning {server_type}: {e}"); traceback.print_exc(); tracking[server_type] = {'error': error_msg}
            finally: del server; cleanup_gpu()
        return tracking

    def _run_final_evaluation(self, costs):
        """Runs final evaluation loops."""
        # (Keep overall structure, logic for loops over runs, costs)
        # Uses self._initialize_experiment, self._final_evaluation
        results, remaining_costs, completed_runs = self._check_existing_results(costs)
        target_runs = self.default_params.get('runs', 1)

        if not remaining_costs and completed_runs >= target_runs: print("Final evaluation already complete."); return results if results is not None else {}, None # Return empty diversity too

        remaining_runs_count = target_runs - completed_runs
        if remaining_runs_count <= 0 and remaining_costs: print(f"Runs complete, but costs {remaining_costs} missing. Rerunning missing for {target_runs} runs."); remaining_runs_count = target_runs; completed_runs = 0
        elif remaining_runs_count <= 0: print("Target eval runs already completed."); return results if results is not None else {}, None

        print(f"Starting {remaining_runs_count} final evaluation run(s)..."); results = results or {}
        diversities, diversity_metadata = self.results_manager.load_results(ExperimentType.DIVERSITY); diversities = diversities or {}
        if diversity_metadata and diversity_metadata.get('contains_errors', False): print("Found errors in diversity metrics. Regenerating."); diversities = {}

        num_clients_metadata = None

        for run_idx in range(remaining_runs_count):
            current_run_total = completed_runs + run_idx + 1; current_seed = self.base_seed + run_idx # Consistent seeding
            print(f"\n--- Starting Final Evaluation Run {current_run_total}/{target_runs} (Seed: {current_seed}) ---")
            set_seeds(current_seed); results_this_run = {}; diversities_this_run = {}; run_meta = {'run_number': current_run_total, 'seed_used': current_seed}; cost_client_counts = {}

            # Rerun ALL costs if runs were incomplete or errors found
            costs_to_run_this_iter = costs if completed_runs < target_runs else remaining_costs

            for cost in costs_to_run_this_iter:
                print(f"\n--- Evaluating Cost: {cost} (Run {current_run_total}) ---")
                num_clients_this_cost = 0; trained_servers = {}; final_round_num = 0
                try:
                    num_clients_this_cost = self._get_final_client_count(self.default_params, cost); cost_client_counts[cost] = num_clients_this_cost; num_clients_metadata = num_clients_metadata or num_clients_this_cost
                    experiment_results_for_cost, final_round_num, trained_servers = self._final_evaluation(cost, current_seed) # Get trained servers back

                    # Save models if eval was successful
                    if 'error' not in experiment_results_for_cost:
                        self._save_evaluation_models(trained_servers, num_clients_this_cost, cost, current_seed)

                    # Extract diversity metrics if present
                    if 'weight_metrics' in experiment_results_for_cost: diversities_this_run[cost] = experiment_results_for_cost.pop('weight_metrics')
                    results_this_run[cost] = experiment_results_for_cost

                except Exception as e: error_msg = str(e); print(f"ERROR during final evaluation for cost {cost}, run {current_run_total}: {e}"); traceback.print_exc(); results_this_run[cost] = {'error': error_msg}
                finally: del trained_servers; cleanup_gpu() # Ensure cleanup

            # Append results for this run
            results = self.results_manager.append_or_create_metric_lists(results, results_this_run)
            diversities = self.results_manager.append_or_create_metric_lists(diversities, diversities_this_run)

            # Save results after each full run
            print(f"--- Completed Final Evaluation Run {current_run_total}/{target_runs} ---")
            run_meta['cost_specific_client_counts'] = cost_client_counts
            self.results_manager.save_results(results, ExperimentType.EVALUATION, client_count=num_clients_metadata, run_metadata=run_meta)
            self.results_manager.save_results(diversities, ExperimentType.DIVERSITY, client_count=num_clients_metadata, run_metadata=run_meta)

        return results, diversities

    def _final_evaluation(self, cost, seed):
        """Internal helper for evaluating one cost value."""
        # (Keep function logic: init data, loop servers, get best params, create config/server, train, track results)
        # Returns results dict, final round num, and dict of trained server instances
        tracking = {}; weight_metrics_acc = {}; trained_servers = {}
        final_round_num = self.default_params.get('rounds') # Default

        try: client_dataloaders = self._initialize_experiment(cost); num_clients_this_run = len(client_dataloaders); self.num_clients_for_run = num_clients_this_run # Store actual client count
        except Exception as e: print(f"ERROR: Init failed for cost {cost}: {e}"); traceback.print_exc(); return ({'error': f"Data init failed: {e}"}, 0, {})
        if num_clients_this_run == 0: return ({'error': "No clients init."}, 0, {})

        for server_type in ALGORITHMS:
            print(f"..... Evaluating Server: {server_type} for Cost: {cost} (Seed: {seed}) .....")
            best_lr = self.results_manager.get_best_parameters(ExperimentType.LEARNING_RATE, server_type, cost) or get_default_lr(self.config.dataset)
            algo_params_dict = {}
            if server_type in ['pfedme', 'ditto']:
                best_reg = self.results_manager.get_best_parameters(ExperimentType.REG_PARAM, server_type, cost) or get_default_reg(self.config.dataset)
                if best_reg is not None: algo_params_dict['reg_param'] = best_reg

            trainer_config = self._create_trainer_config(server_type, best_lr, algo_params_dict)
            final_round_num = trainer_config.rounds # Use actual rounds from config
            server = None
            try:
                server = self._create_server_instance(cost, server_type, trainer_config, tuning=False) # Pass tuning=False
                self._save_initial_model_state(server, num_clients_this_run, cost, seed) # Save initial state if FedAvg
                self._add_clients_to_server(server, client_dataloaders)
                if not server.clients: print(f"Warn: No clients added to {server_type}."); tracking[server_type] = {'error': 'No clients'}; continue
                metrics = self._train_and_evaluate(server, trainer_config.rounds, cost, seed)
                tracking[server_type] = metrics
                trained_servers[server_type] = server # Store server instance *before* potential deletion
                if hasattr(server, 'diversity_metrics') and server.diversity_metrics: weight_metrics_acc = server.diversity_metrics
            except Exception as e: print(f"ERROR eval run {server_type}, cost {cost}: {e}"); traceback.print_exc(); tracking[server_type] = {'error': str(e)}; trained_servers[server_type] = None; del server; server = None; cleanup_gpu()
            # Note: Don't delete server here if successful, it's returned in trained_servers

        if weight_metrics_acc: tracking['weight_metrics'] = weight_metrics_acc
        return tracking, final_round_num, trained_servers # Return tracked metrics and servers

    # --- Data Initialization Logic ---
    def _initialize_experiment(self, cost):
        """Handles the data loading, partitioning, and preprocessing steps."""
        print(f"--- Initializing Data for Cost: {cost} ---")
        # Seed is set by the calling loop (_run_final_evaluation or _run_hyperparam_tuning)
        dataset_name = self.config.dataset; dataset_config = self.default_params
        config_params = self._validate_and_prepare_config(dataset_name, dataset_config)
        data_source = config_params['data_source']
        partitioning_strategy = config_params['partitioning_strategy']
        cost_interpretation = config_params['cost_interpretation']
        source_args, partitioner_args = config_params['source_args'], config_params['partitioner_args']
        partition_scope = config_params['partition_scope']
        sampling_config = dataset_config.get('sampling_config') # For partitioner

        num_clients = self._get_final_client_count(dataset_config, cost)
        client_ids_list = [f'client_{i+1}' for i in range(num_clients)]
        self.num_clients_for_run = num_clients # Store actual count
        print(f"Target number of clients for this cost: {num_clients}")

        translated_cost = translate_cost(cost, cost_interpretation) # Get alpha, key, or suffix
        print(f"Translated cost ({cost_interpretation}): {translated_cost}")

        client_input_data = {}
        preprocessor_input_type = 'unknown'

        if partitioning_strategy.endswith('_indices'):
            # Load base dataset -> Partition indices -> Create Subsets
            print(f"Loading base data source: {data_source}...")
            source_data = self._load_source_data(data_source, dataset_name, source_args)

            # Determine what part of the source data to partition
            if isinstance(source_data, tuple) and len(source_data) == 2: # e.g., torchvision train/test
                data_to_partition = source_data[0] if partition_scope == 'train' else ConcatDataset([source_data[0], source_data[1]])
            elif isinstance(source_data, torch.utils.data.Dataset): # e.g., base loaders returning single dataset
                data_to_partition = source_data
            else: raise TypeError(f"Unexpected source data type from {data_source}: {type(source_data)}")
            print(f"Partitioning {len(data_to_partition)} samples using strategy: {partitioning_strategy}...")

            client_partition_result = self._partition_data(partitioning_strategy, data_to_partition, num_clients, partitioner_args, translated_cost, sampling_config)
            client_input_data, preprocessor_input_type = self._prepare_client_data(client_ids_list, partitioning_strategy, client_partition_result, data_to_partition, dataset_name, dataset_config, translated_cost)

        elif partitioning_strategy == 'pre_split':
            # Load data per client -> Prepare dictionary
            print(f"Using pre-split strategy. Loading data per client using: {data_source}...")
            client_input_data, preprocessor_input_type = self._prepare_client_data(client_ids_list, partitioning_strategy, None, None, dataset_name, dataset_config, translated_cost)

        else:
            raise NotImplementedError(f"Partitioning strategy '{partitioning_strategy}' unknown.")

        if not client_input_data:
            print(f"Warning: No client data loaded or prepared. Returning empty dataloaders.")
            return {}

        print(f"Preprocessing client data (input type: {preprocessor_input_type})...")
        # Pass the full dataset_config to the preprocessor
        preprocessor = DataPreprocessor(dataset_name, dataset_config)
        client_dataloaders = preprocessor.process_client_data(client_input_data, preprocessor_input_type)

        print(f"--- Data Initialization Complete for Cost: {cost} ---")
        return client_dataloaders

    # --- Data Initialization Helpers ---
    def _validate_and_prepare_config(self, dataset_name, dataset_config):
        """Validates config and extracts necessary keys."""
        # (Keep function code as provided in the previous pipeline.py)
        validate_dataset_config(dataset_config, dataset_name); required = ['data_source', 'partitioning_strategy', 'cost_interpretation']; missing = [k for k in required if k not in dataset_config]; assert not missing, f"Config missing keys: {missing}"
        return {'data_source': dataset_config['data_source'], 'partitioning_strategy': dataset_config['partitioning_strategy'], 'cost_interpretation': dataset_config['cost_interpretation'], 'source_args': dataset_config.get('source_args', {}), 'partitioner_args': dataset_config.get('partitioner_args', {}), 'partition_scope': dataset_config.get('partition_scope', 'train')}

    def _get_final_client_count(self, dataset_config, cost):
        """Determines the number of clients, respecting overrides and limits."""
        # (Keep function code as provided in the previous pipeline.py)
        cli_override = self.config.num_clients; num_clients = cli_override or dataset_config.get('default_num_clients', 2); assert isinstance(num_clients, int), f"Client count must be int, got {num_clients}."
        max_clients = dataset_config.get('max_clients');
        if max_clients is not None and num_clients > max_clients: print(f"Warn: Requested {num_clients} > max {max_clients}. Using {max_clients}."); num_clients = max_clients
        partitioning_strategy = dataset_config.get('partitioning_strategy')
        if partitioning_strategy == 'pre_split':
             # Check site mappings if applicable (ISIC, IXI, Heart)
             site_mappings = dataset_config.get('source_args', {}).get('site_mappings')
             if site_mappings:
                 cost_key = translate_cost(cost, dataset_config['cost_interpretation']).get('key') # Translate cost to get key
                 if cost_key in site_mappings: available_for_cost = len(site_mappings[cost_key]);
                 if num_clients > available_for_cost: print(f"Warn: Client count {num_clients} > available sites ({available_for_cost}) for cost '{cost_key}'. Using {available_for_cost}."); num_clients = available_for_cost
                 # else: print(f"Warn: Cost key '{cost}' not found in site_mappings.") # Verbose
        assert isinstance(num_clients, int) and num_clients > 0, f"Calculated invalid num_clients: {num_clients}"; return num_clients

    def _load_source_data(self, data_source, dataset_name, source_args):
        """Loads the source data using the appropriate function from data_loading."""
        loader_func = DATA_LOADERS.get(data_source); assert loader_func, f"Data loader '{data_source}' not found."
        print(f"Calling loader: {loader_func.__name__}")
        return loader_func(dataset_name=dataset_name, data_dir=self.data_dir_root, source_args=source_args)

    def _partition_data(self, partitioning_strategy, data_to_partition, num_clients, partition_args, translated_cost, sampling_config=None):
        """Partitions data using the appropriate function from data_partitioning."""
        partitioner_func = PARTITIONING_STRATEGIES.get(partitioning_strategy); assert partitioner_func, f"Partitioning strategy '{partitioning_strategy}' not found."
        print(f"Calling partitioner: {partitioner_func.__name__}")
        full_args = {**translated_cost, **partition_args, 'seed': self.base_seed} # Use consistent base seed for partitioning
        if sampling_config: full_args['sampling_config'] = sampling_config
        return partitioner_func(dataset=data_to_partition, num_clients=num_clients, **full_args)

    def _load_client_specific_data(self, data_source, dataset_name, client_num, source_args, translated_cost, dataset_config):
        """Loads data for a specific client (used by pre_split strategy)."""
        loader_func = DATA_LOADERS.get(data_source); assert loader_func, f"Data loader '{data_source}' not found."
        # Cost key/suffix needed by per-client loaders (Heart, ISIC, IXI)
        cost_key_or_suffix = translated_cost.get('key', translated_cost.get('suffix'))
        assert cost_key_or_suffix is not None, "Missing cost key/suffix for pre_split loader"
        print(f"Calling client loader: {loader_func.__name__} for client {client_num}, cost={cost_key_or_suffix}")
        # Pass the full config dict to the loader, it might need source_args etc.
        return loader_func(dataset_name=dataset_name, data_dir=self.data_dir_root, client_num=client_num, cost_key=cost_key_or_suffix, config=dataset_config)


    def _prepare_client_data(self, client_ids_list, partitioning_strategy, partition_result, data_to_partition, dataset_name, dataset_config, translated_cost):
        """Prepares the input data dictionary for the DataPreprocessor."""
        client_input_data = {}
        preprocessor_input_type = 'unknown'

        if partitioning_strategy.endswith('_indices'):
            # Input for preprocessor: {client_id: Subset}
            preprocessor_input_type = 'subset'
            if partition_result is None: raise ValueError("partition_result is None for index-based partitioning")
            for i, client_id in enumerate(client_ids_list):
                indices = partition_result.get(i, [])
                if not indices: print(f"Warn: Client {client_id} has no data after partitioning.")
                client_input_data[client_id] = Subset(data_to_partition, indices) # Create subset pointing to original data

        elif partitioning_strategy == 'pre_split':
            # Input for preprocessor: {client_id: {'X': data, 'y': data}} or {'X': paths, 'y': paths/labels}
            data_source = dataset_config['data_source']
            source_args = dataset_config.get('source_args', {})
            for i, client_id in enumerate(client_ids_list):
                client_num = i + 1
                try:
                     # Load data specifically for this client using the cost key/suffix
                     loaded_data = self._load_client_specific_data(data_source, dataset_name, client_num, source_args, translated_cost, dataset_config)
                     if isinstance(loaded_data, tuple) and len(loaded_data) == 2:
                         X, y = loaded_data
                         if len(X) == 0: # Check if loader returned empty data
                              print(f"Warn: No data loaded for client {client_id}. Skipping.")
                              client_input_data[client_id] = None # Mark for removal
                              continue
                         client_input_data[client_id] = {'X': X, 'y': y}
                         # Determine input type for preprocessor based on first client's data
                         if preprocessor_input_type == 'unknown':
                              if isinstance(X, np.ndarray) and isinstance(y, np.ndarray): preprocessor_input_type = 'xy_dict'
                              elif isinstance(X, np.ndarray) and isinstance(X[0], str): preprocessor_input_type = 'path_dict' # Check if X looks like paths
                              else: preprocessor_input_type = 'unknown' # Fallback
                     else:
                         raise TypeError(f"Client data loader {data_source} returned unexpected format: {type(loaded_data)}")
                except FileNotFoundError as e: print(f"Skip client {client_id}: {e}"); client_input_data[client_id] = None
                except Exception as e: print(f"Skip client {client_id}, error loading data: {e}"); traceback.print_exc(); client_input_data[client_id] = None

            # Clean up clients with no data
            client_input_data = {cid: data for cid, data in client_input_data.items() if data is not None}
            if not client_input_data: print(f"Warn: Failed to load data for any pre-split clients."); return {}, 'unknown'
            if preprocessor_input_type == 'unknown': raise RuntimeError("Could not determine preprocessor type for pre-split data.")

        else:
            raise NotImplementedError(f"Data preparation not implemented for strategy: {partitioning_strategy}")

        return client_input_data, preprocessor_input_type


    # --- Model, Server, and Training Helpers ---
    def _create_trainer_config(self, server_type, learning_rate, algorithm_params = None):
        """Creates the TrainerConfig dataclass."""
        # (Keep function code as provided in the previous pipeline.py)
        if algorithm_params is None: algorithm_params = {}
        return TrainerConfig(
            dataset_name=self.config.dataset, device=DEVICE, learning_rate=learning_rate,
            batch_size=self.default_params['batch_size'], epochs=self.default_params.get('epochs_per_round', 1), # Use 1 if not specified
            rounds=self.default_params['rounds'],
            requires_personal_model= True if server_type in ['pfedme', 'ditto'] else False,
            algorithm_params=algorithm_params )


    def _create_model(self, cost, learning_rate):
        """Creates model, criterion, and optimizer."""
        # (Keep function code as provided in the previous pipeline.py)
        model_name = self.config.dataset # Assumes model name matches dataset name
        fixed_classes = self.default_params.get('fixed_classes') # Get class count if defined
        assert hasattr(ms, model_name), f"Model class '{model_name}' not found in models.py."
        model_class = getattr(ms, model_name); model = model_class() # Add potential args if needed, e.g., model_class(num_classes=fixed_classes)

        # Define criterion based on dataset type/metric
        criterion_map = {'Synthetic': nn.CrossEntropyLoss(), 'Credit': nn.CrossEntropyLoss(), 'EMNIST': nn.CrossEntropyLoss(), 'CIFAR': nn.CrossEntropyLoss(), 'IXITiny': get_dice_score, 'ISIC': nn.CrossEntropyLoss(), 'Heart': nn.CrossEntropyLoss()}
        criterion = criterion_map.get(self.config.dataset); assert criterion, f"Criterion not defined for {self.config.dataset}"

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, amsgrad=True, weight_decay=1e-4)
        return model, criterion, optimizer

    def _create_server_instance(self, cost, server_type, config: TrainerConfig, tuning: bool):
        """Creates a server instance."""
        # (Keep function code as provided in the previous pipeline.py)
        model, criterion, optimizer = self._create_model(cost, config.learning_rate)
        globalmodelstate = ModelState(model=model, optimizer=optimizer, criterion=criterion)
        server_mapping = {'local': Server, 'fedavg': FedAvgServer} # Add other servers as needed
        assert server_type in server_mapping, f"Unsupported server: {server_type}"
        server_class = server_mapping[server_type]
        server = server_class(config=config, globalmodelstate=globalmodelstate)
        server.set_server_type(server_type, tuning); return server

    def _add_clients_to_server(self, server, client_dataloaders):
        """Adds clients with their data loaders to the server."""
        # (Keep function code as provided in the previous pipeline.py)
        for client_id, loaders in client_dataloaders.items():
             if not loaders[0]: print(f"Skip client {client_id}: no training data."); continue # Check train loader
             clientdata = self._create_site_data(client_id, loaders)
             server.add_client(clientdata=clientdata)

    def _create_site_data(self, client_id, loaders):
        """Creates SiteData structure."""
        # (Keep function code as provided in the previous pipeline.py)
        return SiteData(site_id=client_id, train_loader=loaders[0], val_loader=loaders[1], test_loader=loaders[2])

    def _train_and_evaluate(self, server, rounds, cost, seed):
        """Runs the train/evaluation loop for a given server."""
        # (Keep function code as provided in the previous pipeline.py)
        # Includes logic for saving model after round 1 if FedAvg/Eval
        print(f"Starting training {'(tuning)' if server.tuning else '(eval)'} for {server.server_type}...")
        for round_num in range(rounds):
            train_loss, val_loss, val_score = server.train_round()
            # Save intermediate model state (e.g., after round 1)
            if not server.tuning and server.server_type == 'fedavg' and round_num == 0:
                self._save_intermediate_model_state(server, cost, seed, 'round1')
            if (round_num + 1) % 20 == 0 or round_num == 0 or (round_num + 1) == rounds: print(f"  R {round_num+1}/{rounds} - TrL: {train_loss:.4f}, VL: {val_loss:.4f}, VS: {val_score:.4f}")
        # Final evaluation on test set (only if not tuning)
        if not server.tuning: print(f"Performing final test evaluation for {server.server_type}..."); server.test_global()
        # Collect metrics
        global_state = server.serverstate; metrics_key = 'val' if server.tuning else 'test'
        metrics = {'global': {'losses': getattr(global_state, f"{metrics_key}_losses"), 'scores': getattr(global_state, f"{metrics_key}_scores") }, 'sites': {}}
        for client_id, client in server.clients.items():
            state_to_log = client.personal_state if server.config.requires_personal_model and client.personal_state else client.global_state
            metrics['sites'][client_id] = {'losses': getattr(state_to_log, f"{metrics_key}_losses"), 'scores': getattr(state_to_log, f"{metrics_key}_scores")}
        final_global_score = metrics['global']['scores'][-1] if metrics['global']['scores'] else None; print(f"Finished {server.server_type}. Final {metrics_key.capitalize()} Score: {final_global_score or 'N/A':.4f}")
        return metrics

    # --- Model Saving Helpers ---
    def _save_initial_model_state(self, server, num_clients_run, cost, seed):
         """Saves the initial model state for FedAvg during evaluation."""
         if not server.tuning and server.server_type == 'fedavg':
              print(f"  Saving INITIAL FedAvg model state...")
              self.results_manager.save_model_state(
                  model_state_dict=server.serverstate.model.state_dict(),
                  experiment_type=ExperimentType.EVALUATION, dataset=self.config.dataset,
                  num_clients_run=num_clients_run, cost=cost, seed=seed,
                  server_type='fedavg', model_type='initial')

    def _save_intermediate_model_state(self, server, cost, seed, model_type: str):
        """Saves intermediate model states (e.g., after round 1)."""
        print(f"  Saving FedAvg model state after {model_type}...")
        self.results_manager.save_model_state(
            model_state_dict=server.serverstate.model.state_dict(),
            experiment_type=ExperimentType.EVALUATION, dataset=self.config.dataset,
            num_clients_run=self.num_clients_for_run, cost=cost, seed=seed,
            server_type='fedavg', model_type=model_type
        )

    def _save_evaluation_models(self, trained_servers, num_clients_run, cost, seed):
        """Saves final and best models after evaluation."""
        if 'fedavg' in trained_servers and trained_servers['fedavg'] is not None:
            fedavg_server = trained_servers['fedavg']
            print(f"  Saving FINAL FedAvg models for cost {cost}, seed {seed}...")
            # Final Model
            self.results_manager.save_model_state(
                model_state_dict=fedavg_server.serverstate.model.state_dict(),
                experiment_type=ExperimentType.EVALUATION, dataset=self.config.dataset,
                num_clients_run=num_clients_run, cost=cost, seed=seed,
                server_type='fedavg', model_type='final')
            # Best Model
            if fedavg_server.serverstate.best_model:
                self.results_manager.save_model_state(
                    model_state_dict=fedavg_server.serverstate.best_model.state_dict(),
                    experiment_type=ExperimentType.EVALUATION, dataset=self.config.dataset,
                    num_clients_run=num_clients_run, cost=cost, seed=seed,
                    server_type='fedavg', model_type='best')
            else: print("  Warn: FedAvg best_model state was None, not saved.")
        elif 'fedavg' in trained_servers: # Check if FedAvg ran but maybe failed and server is None
             print("  Warn: FedAvg server instance not available for model saving (might have failed).")