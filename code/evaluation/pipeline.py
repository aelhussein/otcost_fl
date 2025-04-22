from configs import *
from data_processing import *
from servers import *
from helper import *
from losses import *
import models as ms

DATA_LOADERS = {
    # Base loaders (return BaseDataset for partitioning)
    'torchvision': load_torchvision_dataset,
    'synthetic_base': load_synthetic_base_data,
    'credit_base': load_credit_base_data,

    # Per-client / Site loaders (return X, y tuple for direct use)
    'heart_site_loader': load_heart_site_data, # <--- ADDED
    'pre_split_paths_ixi': load_ixi_client_paths,
    'pre_split_paths_isic': load_isic_client_paths,
}

PARTITIONING_STRATEGIES = {
    'dirichlet_indices': partition_dirichlet_indices, # Takes Dataset, returns {client_idx: indices}
    'iid_indices': partition_iid_indices,           # Takes Dataset, returns {client_idx: indices}
    'pre_split': partition_pre_defined,             # Placeholder, returns client_num for per-client loading
    # Add other strategies here, e.g., 'quantity_skew_indices'
}

class ExperimentType:
    LEARNING_RATE = 'learning_rate'
    REG_PARAM = 'reg_param'
    EVALUATION = 'evaluation'
    DIVERSITY = 'diversity'

class ExperimentConfig: # MODIFIED: Added num_clients
    def __init__(self, dataset, experiment_type, num_clients=None):
        self.dataset = dataset
        self.experiment_type = experiment_type
        self.num_clients = num_clients # Store the override (can be None)

class ResultsManager:
    # MODIFIED: Added model_save_dir and methods for model saving
    def __init__(self, root_dir, dataset, experiment_type, num_clients: int):
        """
        Initializes ResultsManager.
        Args:
            root_dir (str): Root directory for results and models.
            dataset (str): Name of the dataset.
            experiment_type (str): Type of experiment.
            num_clients (int): The target number of clients for this experiment run (used for filename).
        """
        self.root_dir = root_dir
        self.dataset = dataset
        self.experiment_type = experiment_type
        self.num_clients = num_clients # Target client count for filenames
        # Directory for metrics results
        self.results_base_dir = os.path.join(self.root_dir, 'results')
        # Directory for saved models
        self.model_save_base_dir = os.path.join(self.root_dir, 'saved_models') # Use MODEL_SAVE_DIR from configs

        # --- Existing results structure setup ---
        self.results_structure = {}
        for exp_type_key, info in {
            ExperimentType.LEARNING_RATE: {'directory': 'lr_tuning', 'suffix': 'lr_tuning'},
            ExperimentType.REG_PARAM: {'directory': 'reg_param_tuning', 'suffix': 'reg_tuning'},
            ExperimentType.EVALUATION: {'directory': 'evaluation', 'suffix': 'evaluation'},
            ExperimentType.DIVERSITY: {'directory': 'diversity', 'suffix': 'diversity'}
        }.items():
             self.results_structure[exp_type_key] = {
                 'directory': info['directory'],
                 'filename_template': f'{dataset}_{num_clients}clients_{info["suffix"]}.pkl'
             }
        # --- End Existing ---

    def _get_results_path(self, experiment_type):
        """Gets the metrics results path based on the dynamically created template."""
        experiment_info = self.results_structure.get(experiment_type)
        if not experiment_info:
             raise ValueError(f"Unknown experiment type for results path: {experiment_type}")
        return os.path.join(self.results_base_dir, # Use results base dir
                            experiment_info['directory'],
                            experiment_info['filename_template'])

    # --- NEW: Method to get model save path ---
    def _get_model_save_path(self, experiment_type: str, dataset: str,
                             num_clients_run: int, cost: Any, seed: int,
                             server_type: str, model_type: str) -> str:
        """
        Constructs the standardized path for saving model state dictionaries.

        Args:
            experiment_type (str): e.g., ExperimentType.EVALUATION
            dataset (str): Dataset name.
            num_clients_run (int): Actual number of clients used in this specific run.
            cost: Cost parameter used for the run.
            seed (int): Seed used for the run.
            round_num (int): The final round number reached.
            server_type (str): e.g., 'fedavg'.
            model_type (str): e.g., 'final' or 'best'.

        Returns:
            str: The full path for the model file.
        """
        if experiment_type != ExperimentType.EVALUATION:
             # Currently, only saving models from evaluation runs
             raise ValueError("Model saving is currently only implemented for EVALUATION runs.")

        # Consistent cost formatting
        if isinstance(cost, (int, float)): cost_str = f"{float(cost):.4f}"
        else: cost_str = str(cost).replace('/', '_')

        # Directory structure: saved_models/{dataset}/evaluation/
        model_dir = os.path.join(self.model_save_base_dir, dataset, 'evaluation')

        # Filename structure
        filename = (f"{dataset}_{num_clients_run}clients_"
                    f"cost_{cost_str}_seed_{seed}_"
                    f"{server_type}_{model_type}_model.pt")

        return os.path.join(model_dir, filename)

    # --- NEW: Method to save model state ---
    def save_model_state(self, model_state_dict: Dict, experiment_type: str,
                         dataset: str, num_clients_run: int, cost: Any, seed: int,
                         server_type: str, model_type: str):
        """
        Saves a model's state_dict to a standardized path.

        Args:
            model_state_dict (Dict): The state dictionary to save.
            (See _get_model_save_path for other args)
        """
        if not model_state_dict:
             print(f"Warning: Attempted to save an empty model state dict for {model_type} model. Skipping.")
             return

        path = self._get_model_save_path(
            experiment_type, dataset, num_clients_run, cost, seed,
            server_type, model_type
        )
        print(f"  Attempting to save {model_type} model state to: {path}")
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(model_state_dict, path)
            print(f"  Successfully saved {model_type} model state.")
        except Exception as e:
            print(f"ERROR: Failed to save {model_type} model state to {path}: {e}")
            traceback.print_exc()

    def load_results(self, experiment_type):
        # (Implementation remains the same)
        path = self._get_results_path(experiment_type)
        if os.path.exists(path):
            try:
                with open(path, 'rb') as f:
                    loaded_data = pickle.load(f)
                if isinstance(loaded_data, dict) and 'results' in loaded_data and 'metadata' in loaded_data:
                     # print(f"Loaded results from {path} with metadata: {loaded_data['metadata']}") # Verbose
                     return loaded_data['results'], loaded_data['metadata']
                else:
                     # print(f"Loaded results from {path} (assuming old format without metadata).") # Verbose
                     return loaded_data, None
            except Exception as e:
                 print(f"Error loading results from {path}: {e}. Returning None, None.")
                 return None, None
        # print(f"Results file not found at: {path}") # Verbose
        return None, None

    def save_results(self, results, experiment_type, client_count=None, run_metadata=None):
        """
        Saves results with metadata and flags indicating failed experiments.
        
        Args:
            results: The results dictionary to save
            experiment_type: Type of experiment
            client_count: The number of clients used
            run_metadata: Additional metadata for the run
        """
        path = self._get_results_path(experiment_type)
        if run_metadata is None: 
            run_metadata = {}
            
        # Add a flag to indicate if any parts of the results contain errors
        contains_errors = self._check_for_errors_in_results(results)
        
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'dataset': self.dataset,
            'experiment_type': experiment_type,
            'client_count_used': client_count,
            'filename_target_clients': self.num_clients,
            'contains_errors': contains_errors,  # Flag to indicate failures
            **run_metadata
        }
        
        final_data_to_save = {'results': results, 'metadata': metadata}
        
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'wb') as f:
                pickle.dump(final_data_to_save, f)
            
            status = "with ERRORS" if contains_errors else "successfully"
            print(f"Results saved {status} to {path} (Actual Clients: {client_count})")
        except Exception as e:
            print(f"Error saving results to {path}: {e}")
    
    def _check_for_errors_in_results(self, results_dict):
        """
        Recursively checks if any part of the results dictionary contains error entries.
        
        Args:
            results_dict: Results dictionary to check
            
        Returns:
            bool: True if errors are found, False otherwise
        """
        if results_dict is None:
            return False
            
        if isinstance(results_dict, dict):
            # Check for direct error keys at this level
            if 'error' in results_dict:
                return True
                
            # Recursively check all nested dictionaries
            for key, value in results_dict.items():
                if isinstance(value, dict):
                    if self._check_for_errors_in_results(value):
                        return True
                elif isinstance(value, list):
                    # Check if any item in the list is a dict with errors
                    for item in value:
                        if isinstance(item, dict) and self._check_for_errors_in_results(item):
                            return True
                            
        return False

    # ... (append_or_create_metric_lists, get_best_parameters, _select_best_hyperparameter remain the same) ...
    def append_or_create_metric_lists(self, existing_dict, new_dict):
        if existing_dict is None:
             return {k: [v] if not isinstance(v, dict) else self.append_or_create_metric_lists(None, v)
                    for k, v in new_dict.items()}
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
         best_loss = float('inf'); best_param = None
         for param_val, metrics in param_results.items():
             if isinstance(metrics, dict) and 'global' in metrics and 'val_losses' in metrics['global'] and metrics['global']['val_losses']:
                  run_median_losses = []
                  first_loss_val = metrics['global']['val_losses'][0]
                  is_multi_run = isinstance(first_loss_val, list)
                  if is_multi_run:
                       for run_losses in metrics['global']['val_losses']:
                           if run_losses: run_median_losses.append(np.median([l[0] for l in run_losses if l])) # Extract scalar from list
                  elif metrics['global']['val_losses']: # Single run, list of lists [[loss]]
                      valid_losses = [l[0] for l in metrics['global']['val_losses'] if l]
                      if valid_losses: run_median_losses.append(np.median(valid_losses))
                  if run_median_losses:
                      avg_median_loss = np.mean(run_median_losses)
                      if avg_median_loss < best_loss: best_loss = avg_median_loss; best_param = param_val
         # if best_param is None: print(f"Warning: Could not determine best hyperparameter from {param_results.keys()}") # Verbose
         return best_param

class Experiment:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.default_params = get_parameters_for_dataset(self.config.dataset)
        self.data_dir_root = DATA_DIR
        # --- MODIFIED: Get base_seed ---
        self.base_seed = self.default_params.get('base_seed', 42) # Default seed if not specified
        print(f"Using base seed: {self.base_seed} for experiment runs.")
        # --- End Modification ---

        # Determine initial target client count for filenames/ResultsManager
        initial_target_num_clients = self.config.num_clients
        if initial_target_num_clients is None:
            initial_target_num_clients = self.default_params.get('default_num_clients', 2) # Default 2
        if not isinstance(initial_target_num_clients, int) or initial_target_num_clients <= 0:
             raise ValueError(f"Invalid initial target client count: {initial_target_num_clients}")

        print(f"Initializing ResultsManager for {initial_target_num_clients} target clients (filename).")
        self.results_manager = ResultsManager(
            root_dir=ROOT_DIR, dataset=self.config.dataset,
            experiment_type=self.config.experiment_type,
            num_clients=initial_target_num_clients
        )
        self.num_clients_for_run = None

    # ... (run_experiment, _check_existing_results - keep unchanged) ...
    def run_experiment(self, costs):
        # ... (no changes here) ...
        if self.config.experiment_type == ExperimentType.EVALUATION:
            return self._run_final_evaluation(costs)
        elif self.config.experiment_type in [ExperimentType.LEARNING_RATE, ExperimentType.REG_PARAM]:
            return self._run_hyperparam_tuning(costs)
        else:
            raise ValueError(f"Unsupported experiment type: {self.config.experiment_type}")

    def _check_existing_results(self, costs):
        """
        Checks existing results and determines which costs/runs need to be completed.
        Now properly handles results that contain error flags.
        """
        results, metadata = self.results_manager.load_results(self.config.experiment_type)
        runs_key = 'runs_tune' if self.config.experiment_type != ExperimentType.LEARNING_RATE else 'runs'
        target_runs = self.default_params.get(runs_key, 1)
        remaining_costs = list(costs)
        completed_runs = 0
        
        if results is not None:
            # Check the metadata for error flags - if errors exist, we'll re-run
            has_errors = metadata is not None and metadata.get('contains_errors', False)
            if has_errors:
                print(f"Previous results contain errors. Will re-run all experiments.")
                return results, remaining_costs, 0  # Force re-run by returning 0 completed runs
                
            completed_costs = set(results.keys())
            remaining_costs = [c for c in costs if c not in completed_costs]
            
            if completed_costs:
                first_cost = next(iter(completed_costs))
                try:
                    # Extract completed runs count as before
                    first_param = next(iter(results[first_cost]))
                    first_server = next(iter(results[first_cost][first_param]))
                    loss_key = 'val_losses' if self.config.experiment_type != ExperimentType.EVALUATION else 'test_losses'
                    if first_server in results[first_cost][first_param] and \
                       'global' in results[first_cost][first_param][first_server] and \
                       loss_key in results[first_cost][first_param][first_server]['global']:
                         completed_runs = len(results[first_cost][first_param][first_server]['global'][loss_key])
                    else:
                        completed_runs = 0  # Cannot determine
                        
                    # Also check for error entries in any of the costs
                    for cost_key in completed_costs:
                        if self.results_manager._check_for_errors_in_results(results[cost_key]):
                            print(f"Found errors in results for cost {cost_key}. Will re-run.")
                            remaining_costs.append(cost_key)
                            
                except (StopIteration, KeyError, IndexError, TypeError, AttributeError) as e:
                    completed_runs = 0
                    print(f"Could not determine completed runs: {e}")
                
            # Ensure no duplicates in remaining_costs
            remaining_costs = list(set(remaining_costs))
                
            print(f"Found {completed_runs}/{target_runs} completed valid runs in existing results.")
            if completed_runs >= target_runs and not remaining_costs:
                remaining_costs = []
            elif completed_runs < target_runs:
                remaining_costs = list(costs)
                print(f"Runs incomplete. Will run all costs.")
        else:
            print("No existing results found.")
            
        print(f"Remaining costs to process: {remaining_costs}")
        return results, remaining_costs, completed_runs
    
    def _run_hyperparam_tuning(self, costs):
        # --- MODIFIED: Set seed for each tuning run ---
        results, remaining_costs, completed_runs = self._check_existing_results(costs)
        runs_tune = self.default_params.get('runs_tune', 1)
        if not remaining_costs and completed_runs >= runs_tune: return results if results is not None else {}

        remaining_runs_count = runs_tune - completed_runs
        if remaining_runs_count <= 0 and remaining_costs:
             print(f"Runs complete, but costs {remaining_costs} missing. Rerunning missing costs for {runs_tune} runs.")
             remaining_runs_count = runs_tune; completed_runs = 0
        elif remaining_runs_count <= 0: return results if results is not None else {}

        print(f"Starting {remaining_runs_count} hyperparameter tuning run(s)...")
        if results is None: results = {}
        num_clients_metadata = None

        for run_idx in range(remaining_runs_count):
            current_run_total = completed_runs + run_idx + 1
            current_seed = self.base_seed + run_idx # Use run-specific seed
            print(f"--- Starting Tuning Run {current_run_total}/{runs_tune} (Seed: {current_seed}) ---")
            set_seeds(current_seed) # Set seed for this run's data init and training
            run_meta = {'run_number': current_run_total, 'seed_used': current_seed}
            cost_client_counts = {}

            for cost in remaining_costs:
                print(f"--- Processing Cost: {cost} (Run {current_run_total}) ---")
                try:
                    num_clients_this_cost = self._get_final_client_count(self.default_params, cost)
                    cost_client_counts[cost] = num_clients_this_cost
                    if num_clients_metadata is None: num_clients_metadata = num_clients_this_cost
                except Exception as e: print(f"Error determining client count for cost {cost}: {e}. Skipping."); continue

                if self.config.experiment_type == ExperimentType.LEARNING_RATE:
                     server_types_to_tune = self.default_params.get('servers_tune_lr', ['local', 'fedavg'])
                     param_key, fixed_param_key = 'learning_rate', 'reg_param'
                     fixed_param_value = get_default_reg(self.config.dataset)
                     params_to_try_values = self.default_params.get('learning_rates_try', [])
                elif self.config.experiment_type == ExperimentType.REG_PARAM:
                     server_types_to_tune = self.default_params.get('servers_tune_reg', [])
                     param_key, fixed_param_key = 'reg_param', 'learning_rate'
                     fixed_param_value = get_default_lr(self.config.dataset)
                     params_to_try_values = self.default_params.get('reg_params_try', [])
                else: raise ValueError(f"Unsupported tuning type: {self.config.experiment_type}")

                if not params_to_try_values: print(f"Warning: No parameters to try for {self.config.experiment_type}, cost {cost}. Skipping."); continue

                hyperparams_list = [{param_key: p_val, fixed_param_key: fixed_param_value} for p_val in params_to_try_values]
                tuning_results_for_cost = {}
                for hyperparams in hyperparams_list:
                    param_value_being_tuned = hyperparams[param_key]
                    print(f"--- Tuning Param: {param_key}={param_value_being_tuned} ---")
                    # Pass cost, hyperparams, server types. Seed is already set globally for this run.
                    server_metrics = self._hyperparameter_tuning(cost, hyperparams, server_types_to_tune)
                    tuning_results_for_cost[param_value_being_tuned] = server_metrics

                # Append results
                if cost not in results: results[cost] = {}
                for param_val, server_data in tuning_results_for_cost.items():
                    if param_val not in results[cost]: results[cost][param_val] = {}
                    for server_type, metrics in server_data.items():
                        if server_type not in results[cost][param_val]: results[cost][param_val][server_type] = None
                        results[cost][param_val][server_type] = self.results_manager.append_or_create_metric_lists(
                            results[cost][param_val][server_type], metrics)

            # Save results after each run
            print(f"--- Completed Tuning Run {current_run_total}/{runs_tune} ---")
            run_meta['cost_specific_client_counts'] = cost_client_counts
            self.results_manager.save_results(
                results, self.config.experiment_type,
                client_count=num_clients_metadata, run_metadata=run_meta)
        return results

    def _hyperparameter_tuning(self, cost, hyperparams, server_types):
        """
        Perform hyperparameter tuning with improved error handling.
        """
        try:
            client_dataloaders = self._initialize_experiment(cost)
            if not client_dataloaders:
                error_msg = f"No client dataloaders for cost {cost}."
                print(f"Warning: {error_msg}")
                return {'error': error_msg}
        except Exception as e:
            error_msg = f"Initialization failed for cost {cost}: {e}"
            print(f"ERROR: {error_msg}")
            traceback.print_exc()
            return {'error': error_msg}

        tracking = {}
        for server_type in server_types:
            lr = hyperparams.get('learning_rate')
            reg_param_val = None
            algo_params_dict = {}
            if server_type in ['pfedme', 'ditto']:
                reg_param_val = hyperparams.get('reg_param')
                if reg_param_val is not None:
                    algo_params_dict['reg_param'] = reg_param_val
            trainer_config = self._create_trainer_config(server_type, lr, algo_params_dict)

            server = None
            try:
                server = self._create_server_instance(cost, server_type, trainer_config, tuning=True)
                self._add_clients_to_server(server, client_dataloaders)
                if not server.clients:
                    error_msg = 'No clients added'
                    print(f"Warning: No clients added to {server_type}.")
                    tracking[server_type] = {'error': error_msg}
                    continue

                num_tuning_rounds = self.default_params.get('rounds_tune_inner', self.default_params['rounds'])
                metrics = self._train_and_evaluate(server, num_tuning_rounds, cost, seed=self.base_seed)
                tracking[server_type] = metrics
            except Exception as e:
                error_msg = str(e)
                print(f"ERROR tuning {server_type}: {e}")
                traceback.print_exc()
                tracking[server_type] = {'error': error_msg}
            finally:
                if server:
                    del server
                cleanup_gpu()
        return tracking

    def _run_final_evaluation(self, costs):
        """
        Runs final evaluation with improved error handling.
        """
        results, remaining_costs, completed_runs = self._check_existing_results(costs)
        target_runs = self.default_params.get('runs', 1)

        # Simple check for now: if results exist for target runs, assume models might exist too
        if results is not None and completed_runs >= target_runs:
            # Check specifically for any errors in the existing results
            has_errors = False
            for cost in costs:
                if cost in results and isinstance(results[cost], dict) and 'error' in results[cost]:
                    has_errors = True
                    break
                
            if has_errors:
                print(f"Found errors in existing results. Will re-run to ensure all models are properly saved.")
                remaining_runs_count = target_runs  # Force re-run to fix errors
                completed_runs = 0
            else:
                print(f"Final evaluation metrics completed ({completed_runs}/{target_runs} runs).")
                # We'll still run if there are any remaining costs to process
                remaining_runs_count = max(0, target_runs - completed_runs)
        else:
            remaining_runs_count = max(0, target_runs - completed_runs)
            
        if remaining_runs_count > 0:
            print(f"Starting {remaining_runs_count} final evaluation run(s)...")

        if results is None:
            results = {}
        diversities, diversity_metadata = self.results_manager.load_results(ExperimentType.DIVERSITY)
        if diversities is None:
            diversities = {}
            
        # Check if diversity results have errors
        has_diversity_errors = diversity_metadata and diversity_metadata.get('contains_errors', False)
        if has_diversity_errors:
            print("Found errors in diversity metrics. Will regenerate.")
            diversities = {}

        num_clients_metadata = None

        # Loop through the required number of runs
        for run_idx in range(remaining_runs_count):
            current_run_total = completed_runs + run_idx + 1
            current_seed = self.base_seed + run_idx
            print(f"--- Starting Final Evaluation Run {current_run_total}/{target_runs} (Seed: {current_seed}) ---")
            set_seeds(current_seed)

            results_this_run = {}
            diversities_this_run = {}
            run_meta = {'run_number': current_run_total, 'seed_used': current_seed}
            cost_client_counts = {}

            # Process each cost
            for cost in costs:
                print(f"--- Evaluating Cost: {cost} (Run {current_run_total}) ---")
                num_clients_this_cost = 0
                trained_servers = {}
                final_round_num = 0
                
                try:
                    num_clients_this_cost = self._get_final_client_count(self.default_params, cost)
                    cost_client_counts[cost] = num_clients_this_cost
                    if num_clients_metadata is None:
                        num_clients_metadata = num_clients_this_cost

                    # Run evaluation for this cost
                    experiment_results_for_cost, final_round_num, trained_servers = self._final_evaluation(cost, current_seed)

                    # Model saving (only if the experiment didn't fail)
                    if 'error' not in experiment_results_for_cost:
                        if 'fedavg' in trained_servers and trained_servers['fedavg'] is not None:
                            fedavg_server = trained_servers['fedavg']
                            print(f"  Saving FedAvg models for cost {cost}, seed {current_seed}...")

                            # Save Final Model State
                            self.results_manager.save_model_state(
                                model_state_dict=fedavg_server.serverstate.model.state_dict(),
                                experiment_type=ExperimentType.EVALUATION,
                                dataset=self.config.dataset,
                                num_clients_run=num_clients_this_cost,
                                cost=cost,
                                seed=current_seed,
                                server_type='fedavg',
                                model_type='final'
                            )
                            
                            # Save Best Model State
                            if fedavg_server.serverstate.best_model:
                                self.results_manager.save_model_state(
                                    model_state_dict=fedavg_server.serverstate.best_model.state_dict(),
                                    experiment_type=ExperimentType.EVALUATION,
                                    dataset=self.config.dataset,
                                    num_clients_run=num_clients_this_cost,
                                    cost=cost,
                                    seed=current_seed,
                                    server_type='fedavg',
                                    model_type='best'
                                )
                            else:
                                print("  Warning: FedAvg best_model state was None, not saved.")
                        elif 'fedavg' in experiment_results_for_cost and 'error' not in experiment_results_for_cost['fedavg']:
                            print("  Warning: FedAvg metrics exist but server instance not returned for model saving.")
                    else:
                        print(f"  Error occurred in experiment for cost {cost}. Models not saved.")

                    # Process diversity metrics
                    if 'weight_metrics' in experiment_results_for_cost:
                        diversities_this_run[cost] = experiment_results_for_cost.pop('weight_metrics')
                        
                    results_this_run[cost] = experiment_results_for_cost

                except Exception as e:
                    error_msg = str(e)
                    print(f"ERROR during final evaluation for cost {cost}, run {current_run_total}: {e}")
                    traceback.print_exc()
                    results_this_run[cost] = {'error': error_msg}
                finally:
                    # Clean up server instances
                    for server in trained_servers.values():
                        if server:
                            del server
                    cleanup_gpu()

            # Append results for this run
            results = self.results_manager.append_or_create_metric_lists(results, results_this_run)
            diversities = self.results_manager.append_or_create_metric_lists(diversities, diversities_this_run)

            # Save metrics after each completed run
            print(f"--- Completed Final Evaluation Run {current_run_total}/{target_runs} ---")
            run_meta['cost_specific_client_counts'] = cost_client_counts
            
            self.results_manager.save_results(
                results, 
                ExperimentType.EVALUATION, 
                client_count=num_clients_metadata, 
                run_metadata=run_meta
            )
            
            self.results_manager.save_results(
                diversities, 
                ExperimentType.DIVERSITY, 
                client_count=num_clients_metadata, 
                run_metadata=run_meta
            )

        return results, diversities


    def _final_evaluation(self, cost, seed):
        """Runs final evaluation for ONE cost value across all servers for a specific seed."""
        tracking = {}
        weight_metrics_acc = {}
        trained_servers = {} # Store trained server instances
        final_round_num = self.default_params.get('rounds') # Default rounds

        try:
            # Seed is set *before* this call in the outer loop
            client_dataloaders = self._initialize_experiment(cost)
            # Store actual client count used in this initialization
            num_clients_this_run = len(client_dataloaders)
            self.num_clients_for_run = num_clients_this_run # Update instance variable
            if num_clients_this_run == 0: return ({'error': "No clients init."}, 0, {})
        except Exception as e: print(f"ERROR: Init failed for cost {cost}: {e}"); traceback.print_exc(); return ({'error': f"Data init failed: {e}"}, 0, {})

        for server_type in ALGORITHMS:
            print(f"..... Evaluating Server: {server_type} for Cost: {cost} (Seed: {seed}) .....")
            best_lr = self.results_manager.get_best_parameters(ExperimentType.LEARNING_RATE, server_type, cost)
            if best_lr is None: best_lr = get_default_lr(self.config.dataset) # Fallback

            algo_params_dict = {}
            if server_type in ['pfedme', 'ditto']: # Add other algos needing reg params
                 best_reg_param = self.results_manager.get_best_parameters(ExperimentType.REG_PARAM, server_type, cost)
                 if best_reg_param is None: best_reg_param = get_default_reg(self.config.dataset) # Fallback
                 if best_reg_param is not None: algo_params_dict['reg_param'] = best_reg_param

            trainer_config = self._create_trainer_config(server_type, best_lr, algo_params_dict)
            final_round_num = trainer_config.rounds # Store actual rounds used
            server = None
            try:
                # Seed is already set, ensures consistent model init
                server = self._create_server_instance(cost, server_type, trainer_config, tuning=False)

                # *** --- NEW: Save Initial FedAvg Model State --- ***
                if server_type == 'fedavg':
                     print(f"  Saving INITIAL FedAvg model state...")
                     self.results_manager.save_model_state(
                         model_state_dict=server.serverstate.model.state_dict(),
                         experiment_type=ExperimentType.EVALUATION,
                         dataset=self.config.dataset,
                         num_clients_run=num_clients_this_run, # Use actual count
                         cost=cost, seed=seed,
                         server_type='fedavg', model_type='initial' # Use 'initial' type
                     )
                # *** --- End New --- ***

                self._add_clients_to_server(server, client_dataloaders)
                if not server.clients: print(f"Warn: No clients added to {server_type}."); tracking[server_type] = {'error': 'No clients'}; continue

                # --- MODIFIED: Pass cost and seed to _train_and_evaluate ---
                metrics = self._train_and_evaluate(server=server, rounds=trainer_config.rounds, cost=cost, seed=seed)
                tracking[server_type] = metrics
                trained_servers[server_type] = server # Keep server instance
                if hasattr(server, 'diversity_metrics') and server.diversity_metrics: weight_metrics_acc = server.diversity_metrics
            except Exception as e:
                print(f"ERROR eval run {server_type}, cost {cost}: {e}"); traceback.print_exc()
                tracking[server_type] = {'error': str(e)}
                if server: trained_servers[server_type] = None; del server; server = None
                cleanup_gpu()
        if weight_metrics_acc: tracking['weight_metrics'] = weight_metrics_acc
        return tracking, final_round_num, trained_servers


    # ... (_get_final_client_count, _validate_and_prepare_config, _load_source_data, _partition_data, _load_client_specific_data, _prepare_client_data - keep unchanged) ...
    def _get_final_client_count(self, dataset_config, cost):
        cli_override = self.config.num_clients; num_clients = cli_override
        if num_clients is None: num_clients = dataset_config.get('default_num_clients', 2)
        if not isinstance(num_clients, int): raise TypeError(f"Client count must be int, got {num_clients}.")
        max_clients = dataset_config.get('max_clients')
        if max_clients is not None and num_clients > max_clients: print(f"Warn: Requested {num_clients} > max {max_clients}. Using {max_clients}."); num_clients = max_clients
        partitioning_strategy = dataset_config.get('partitioning_strategy')
        if partitioning_strategy == 'pre_split':
             if self.config.dataset in ['ISIC', 'IXITiny']:
                  site_mappings = dataset_config.get('source_args', {}).get('site_mappings')
                  if site_mappings:
                      if cost in site_mappings:
                          available_for_cost = len(site_mappings[cost])
                          if num_clients > available_for_cost: print(f"Warn: Client count {num_clients} > available sites ({available_for_cost}) for cost '{cost}'. Using {available_for_cost}."); num_clients = available_for_cost
                      # else: print(f"Warn: Cost key '{cost}' not found in site_mappings.") # Verbose
        if not isinstance(num_clients, int) or num_clients <= 0: raise ValueError(f"Calculated invalid num_clients: {num_clients}")
        return num_clients

    def _validate_and_prepare_config(self, dataset_name, dataset_config):
        validate_dataset_config(dataset_config, dataset_name)
        required_pipeline_keys = ['data_source', 'partitioning_strategy', 'cost_interpretation']
        missing_keys = [k for k in required_pipeline_keys if k not in dataset_config]
        if missing_keys: raise ValueError(f"Config missing keys: {missing_keys}")
        return {'data_source': dataset_config['data_source'], 'partitioning_strategy': dataset_config['partitioning_strategy'], 'cost_interpretation': dataset_config['cost_interpretation'], 'source_args': dataset_config.get('source_args', {}), 'partitioner_args': dataset_config.get('partitioner_args', {}), 'partition_scope': dataset_config.get('partition_scope', 'train')}

    def _load_source_data(self, data_source, dataset_name, source_args):
        loader_func = DATA_LOADERS.get(data_source)
        if loader_func is None: raise NotImplementedError(f"Data loader '{data_source}' not found.")
        # print(f"Loading source data using: {loader_func.__name__}") # Verbose
        return loader_func(dataset_name=dataset_name, data_dir=self.data_dir_root, source_args=source_args)

    def _partition_data(self, partitioning_strategy, data_to_partition, num_clients, partition_args, translated_cost, sampling_config=None):
        partitioner_func = PARTITIONING_STRATEGIES.get(partitioning_strategy);
        if partitioner_func is None: raise NotImplementedError(f"Partitioning strategy '{partitioning_strategy}' not found.")
        # print(f"Partitioning using strategy: {partitioner_func.__name__}") # Verbose
        full_args = {**translated_cost, **partition_args, 'seed': 42}
        if sampling_config: full_args['sampling_config'] = sampling_config
        if partitioning_strategy == 'pre_split': return None
        else: return partitioner_func(dataset=data_to_partition, num_clients=num_clients, **full_args)

    def _load_client_specific_data(self, data_source, dataset_name, client_num, source_args, translated_cost, dataset_config):
        loader_func = DATA_LOADERS.get(data_source)
        if loader_func is None: raise NotImplementedError(f"Data loader '{data_source}' not found.")
        # print(f"Loading pre-split data for client {client_num} using: {loader_func.__name__}") # Verbose
        specific_cost_arg = translated_cost.get('key', translated_cost.get('suffix'))
        if specific_cost_arg is None: raise ValueError("Missing cost key/suffix for pre_split loader")
        return loader_func(dataset_name=dataset_name, data_dir=self.data_dir_root, client_num=client_num, cost_key_or_suffix=specific_cost_arg, config=dataset_config)

    def _prepare_client_data(self, client_ids_list, partitioning_strategy, partition_result, data_to_partition, dataset_name, dataset_config, translated_cost):
        client_input_data = {}; preprocessor_input_type = 'unknown'
        if partitioning_strategy.endswith('_indices'):
            if partition_result is None: raise ValueError("partition_result None for index partitioning")
            for i, client_id in enumerate(client_ids_list):
                indices = partition_result.get(i, [])
                if not indices : print(f"Warn: Client {client_id} has no data.")
                client_input_data[client_id] = Subset(data_to_partition, indices)
            preprocessor_input_type = 'subset'
        elif partitioning_strategy == 'pre_split':
            data_source = dataset_config['data_source']; source_args = dataset_config.get('source_args', {})
            for i, client_id in enumerate(client_ids_list):
                client_num = i + 1
                try:
                     loaded_data = self._load_client_specific_data(data_source, dataset_name, client_num, source_args, translated_cost, dataset_config)
                     if isinstance(loaded_data, tuple) and len(loaded_data) == 2:
                         X, y = loaded_data; client_input_data[client_id] = {'X': X, 'y': y}
                         if preprocessor_input_type == 'unknown':
                              if isinstance(X, np.ndarray) and isinstance(y, np.ndarray): current_type = 'xy_dict'
                              elif isinstance(X, (list, np.ndarray)) and len(X)>0 and isinstance(X[0], str): current_type = 'path_dict'
                              else: current_type = 'unknown'
                              preprocessor_input_type = current_type
                     else: raise TypeError(f"Loader returned unexpected format: {type(loaded_data)}")
                except FileNotFoundError as e: print(f"Skip client {client_id}: {e}"); client_input_data[client_id] = None
                except Exception as e: print(f"Skip client {client_id}, error: {e}"); traceback.print_exc(); client_input_data[client_id] = None
            client_input_data = {cid: data for cid, data in client_input_data.items() if data is not None}
            if not client_input_data: print(f"Warn: Failed to load data for any pre-split clients."); return {}, 'unknown'
            if preprocessor_input_type == 'unknown': raise RuntimeError("Could not determine preprocessor type for pre-split data.")
        else: raise NotImplementedError(f"Partitioning strategy '{partitioning_strategy}' not supported.")
        return client_input_data, preprocessor_input_type

    # --- Main Initialization Method ---
    def _initialize_experiment(self, cost):
        # Seed is set by the calling loop (_run_final_evaluation or _run_hyperparam_tuning)
        dataset_name = self.config.dataset; dataset_config = self.default_params
        config_params = self._validate_and_prepare_config(dataset_name, dataset_config)
        data_source, partitioning_strategy, cost_interpretation = config_params['data_source'], config_params['partitioning_strategy'], config_params['cost_interpretation']
        source_args, partitioner_args, partition_scope = config_params['source_args'], config_params['partitioner_args'], config_params['partition_scope']
        sampling_config = dataset_config.get('sampling_config')
        num_clients = self._get_final_client_count(dataset_config, cost)
        client_ids_list = [f'client_{i+1}' for i in range(num_clients)]
        self.num_clients_for_run = num_clients
        translated_cost = translate_cost(cost, cost_interpretation)
        client_input_data, preprocessor_input_type = {}, 'unknown'; data_to_partition = None
        if partitioning_strategy.endswith('_indices'):
            source_data_tuple = self._load_source_data(data_source, dataset_name, source_args)
            if partition_scope == 'train': data_to_partition = source_data_tuple[0]
            elif partition_scope == 'all': data_to_partition = torch.utils.data.ConcatDataset([source_data_tuple[0], source_data_tuple[1]])
            else: raise ValueError(f"Unsupported partition_scope: {partition_scope}")
            client_partition_result = self._partition_data(partitioning_strategy, data_to_partition, num_clients, partitioner_args, translated_cost, sampling_config)
            client_input_data, preprocessor_input_type = self._prepare_client_data(client_ids_list, partitioning_strategy, client_partition_result, data_to_partition, dataset_name, dataset_config, translated_cost)
        elif partitioning_strategy == 'pre_split':
            client_input_data, preprocessor_input_type = self._prepare_client_data(client_ids_list, partitioning_strategy, None, None, dataset_name, dataset_config, translated_cost)
        else: raise NotImplementedError(f"Partitioning strategy '{partitioning_strategy}' unknown.")
        if not client_input_data: print(f"Warning: client_input_data empty."); return {}
        preprocessor = DataPreprocessor(dataset_name, self.default_params['batch_size'])
        client_dataloaders = preprocessor.process_client_data(client_input_data, preprocessor_input_type)
        return client_dataloaders

    def _create_trainer_config(self, server_type, learning_rate, algorithm_params = None):
         if algorithm_params is None: algorithm_params = {}
         return TrainerConfig(
             dataset_name=self.config.dataset, device=DEVICE, learning_rate=learning_rate,
             batch_size=self.default_params['batch_size'], epochs=self.default_params.get('epochs_per_round', 5),
             rounds=self.default_params['rounds'],
             requires_personal_model= True if server_type in ['pfedme', 'ditto'] else False,
             algorithm_params=algorithm_params )

    def _create_model(self, cost, learning_rate):
        dataset_config = self.default_params; model_name = self.config.dataset
        fixed_classes = dataset_config.get('fixed_classes')
        if not hasattr(ms, model_name): raise ValueError(f"Model class '{model_name}' not found.")
        model_class = getattr(ms, model_name); model = model_class()
        criterion_map = {'Synthetic': nn.CrossEntropyLoss(), 'Credit': nn.CrossEntropyLoss(), 'EMNIST': nn.CrossEntropyLoss(), 'CIFAR': nn.CrossEntropyLoss(), 'IXITiny': get_dice_score, 'ISIC': nn.CrossEntropyLoss(), 'Heart': nn.CrossEntropyLoss()}
        criterion = criterion_map.get(self.config.dataset);
        if criterion is None: raise ValueError(f"Criterion not defined for {self.config.dataset}")
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, amsgrad=True, weight_decay=1e-4)
        return model, criterion, optimizer

    def _create_server_instance(self, cost, server_type, config: TrainerConfig, tuning: bool):
        learning_rate = config.learning_rate
        model, criterion, optimizer = self._create_model(cost, learning_rate)
        globalmodelstate = ModelState(model=model, optimizer=optimizer, criterion=criterion)
        server_mapping = {'local': Server, 'fedavg': FedAvgServer} # Add other servers if implemented
        if server_type not in server_mapping: raise ValueError(f"Unsupported server: {server_type}")
        server_class = server_mapping[server_type]
        server = server_class(config=config, globalmodelstate=globalmodelstate)
        server.set_server_type(server_type, tuning); return server

    def _add_clients_to_server(self, server, client_dataloaders):
         for client_id, loaders in client_dataloaders.items():
             if not loaders[0]: print(f"Skip client {client_id}: no training data."); continue
             if client_id == 'client_joint' and server.server_type != 'local': continue
             clientdata = self._create_site_data(client_id, loaders)
             server.add_client(clientdata=clientdata)

    def _create_site_data(self, client_id, loaders):
         return SiteData(site_id=client_id, train_loader=loaders[0], val_loader=loaders[1], test_loader=loaders[2])

    def _train_and_evaluate(self, server, rounds, cost, seed): # Added cost and seed
         """ Train/evaluate, save FedAvg model after round 1 """
         print(f"Starting training {'(tuning)' if server.tuning else '(eval)'} for {server.server_type}...")
         for round_num in range(rounds):
             train_loss, val_loss, val_score = server.train_round()

             # --- MODIFIED: Save FedAvg model after round 1 during evaluation ---
             if not server.tuning and server.server_type == 'fedavg' and round_num == 0:
                  print(f"  Saving FedAvg model state after round 1...")
                  self.results_manager.save_model_state(
                      model_state_dict=server.serverstate.model.state_dict(),
                      experiment_type=ExperimentType.EVALUATION,
                      dataset=self.config.dataset,
                      num_clients_run=self.num_clients_for_run, # Use actual client count for this run
                      cost=cost, seed=seed,
                      server_type='fedavg', model_type='round1' # Use 'round1' type
                  )
             # --- End Modification ---

             # Optional progress print
             if (round_num + 1) % 20 == 0 or round_num == 0 or (round_num + 1) == rounds:
                  print(f"  R {round_num+1}/{rounds} - TrL: {train_loss:.4f}, VL: {val_loss:.4f}, VS: {val_score:.4f}")

         # Final Evaluation after all rounds (only if not tuning)
         if not server.tuning:
             print(f"Performing final test evaluation for {server.server_type}...")
             server.test_global() # Evaluate on test sets

         # Collect Metrics (remains the same)
         global_state = server.serverstate
         if server.tuning:
             metrics = {'global': {'losses': global_state.val_losses, 'scores': global_state.val_scores }, 'sites': {}}
         else:
            metrics = {'global': {'losses': global_state.test_losses, 'scores': global_state.test_scores }, 'sites': {}}

         for client_id, client in server.clients.items():
            state_to_log = client.personal_state if server.config.requires_personal_model and client.personal_state else client.global_state
            if server.tuning:
                metrics['sites'][client_id] = {'losses': state_to_log.val_losses, 
                                                'scores': state_to_log.val_scores}
            else:
                metrics['sites'][client_id] = {'losses': state_to_log.test_losses, 
                                                'scores': state_to_log.test_scores}
                
         final_global_score = metrics['global']['scores'][-1] if metrics['global']['scores'] else None
         print(f"Finished {server.server_type}. Final Test Score: {final_global_score:.4f}")
         return metrics