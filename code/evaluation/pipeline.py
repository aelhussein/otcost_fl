from configs import *
from data_processing import *
from servers import *
from helper import *
from losses import *
import models as ms
from performance_logging import *

DATA_LOADERS = {
    'torchvision': load_torchvision_dataset,
    'pre_split_csv': load_pre_split_csv_client, # Loads per client
    'pre_split_paths_ixi': load_ixi_client_paths, # Loads per client
    'pre_split_paths_isic': load_isic_client_paths, # Loads per client
    # Add other loaders here, e.g., 'hdf5', 'database'
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

# MODIFIED: Updated save_results and load_results for metadata handling
class ResultsManager:
    def __init__(self, root_dir, dataset, experiment_type, num_clients: int): # ADDED num_clients
        """
        Initializes ResultsManager.

        Args:
            root_dir (str): Root directory for results.
            dataset (str): Name of the dataset.
            experiment_type (str): Type of experiment.
            num_clients (int): The target number of clients for this experiment run (used for filename).
        """
        self.root_dir = root_dir
        self.dataset = dataset
        self.experiment_type = experiment_type
        self.num_clients = num_clients # Store target client count for filenames

        # Dynamically create filename templates including client count
        self.results_structure = {}
        for exp_type_key, info in {
            ExperimentType.LEARNING_RATE: {'directory': 'lr_tuning', 'suffix': 'lr_tuning'},
            ExperimentType.REG_PARAM: {'directory': 'reg_param_tuning', 'suffix': 'reg_tuning'},
            ExperimentType.EVALUATION: {'directory': 'evaluation', 'suffix': 'evaluation'},
            ExperimentType.DIVERSITY: {'directory': 'diversity', 'suffix': 'diversity'}
        }.items():
             self.results_structure[exp_type_key] = {
                 'directory': info['directory'],
                 # MODIFIED FILENAME TEMPLATE
                 'filename_template': f'{dataset}_{num_clients}clients_{info["suffix"]}.pkl'
             }

    def _get_results_path(self, experiment_type):
        """Gets the results path based on the dynamically created template."""
        experiment_info = self.results_structure.get(experiment_type)
        if not experiment_info:
             raise ValueError(f"Unknown experiment type for results path: {experiment_type}")
        # The template already includes num_clients now
        return os.path.join(self.root_dir,'results',
                            experiment_info['directory'],
                            experiment_info['filename_template']) # Use the generated template

    def load_results(self, experiment_type):
        """Loads results, handling both old and new format (with metadata)."""
        # This path now implicitly includes the num_clients set during __init__
        path = self._get_results_path(experiment_type)
        if os.path.exists(path):
            try:
                with open(path, 'rb') as f:
                    loaded_data = pickle.load(f)
                if isinstance(loaded_data, dict) and 'results' in loaded_data and 'metadata' in loaded_data:
                     print(f"Loaded results from {path} with metadata: {loaded_data['metadata']}")
                     return loaded_data['results'], loaded_data['metadata']
                else:
                     print(f"Loaded results from {path} (assuming old format without metadata).")
                     return loaded_data, None
            except Exception as e:
                 print(f"Error loading results from {path}: {e}. Returning None, None.")
                 return None, None
        print(f"Results file not found at: {path}")
        return None, None

    # save_results signature already correct, accepts actual client_count for metadata
    def save_results(self, results, experiment_type, client_count=None, run_metadata=None):
        """Save results with metadata including actual client count used."""
        # Path determined by num_clients passed during __init__
        path = self._get_results_path(experiment_type)
        if run_metadata is None: run_metadata = {}

        metadata = {
            'timestamp': datetime.now().isoformat(),
            'dataset': self.dataset,
            'experiment_type': experiment_type,
            'client_count_used': client_count, # Actual count used for this specific run/cost
            'filename_target_clients': self.num_clients, # Target count used in filename
            **run_metadata
        }
        final_data_to_save = {'results': results, 'metadata': metadata}

        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'wb') as f:
                pickle.dump(final_data_to_save, f)
            print(f"Results saved to {path} with metadata (Actual Clients Used: {client_count})")
        except Exception as e:
            print(f"Error saving results to {path}: {e}")

    def append_or_create_metric_lists(self, existing_dict, new_dict):
        # ... (Implementation from your previous code - assumed correct) ...
        if existing_dict is None:
             # Create lists for leaf nodes
             return {k: [v] if not isinstance(v, dict) else self.append_or_create_metric_lists(None, v)
                    for k, v in new_dict.items()}

        for key, new_value in new_dict.items():
             if isinstance(new_value, dict):
                 # Recursively process nested dictionaries
                 if key not in existing_dict or not isinstance(existing_dict[key], dict):
                      existing_dict[key] = {} # Initialize if key is new or not a dict
                 existing_dict[key] = self.append_or_create_metric_lists(existing_dict[key], new_value)
             else:
                  # Append leaf values to lists
                  if key not in existing_dict:
                       existing_dict[key] = []
                  # Ensure it's a list before appending
                  if not isinstance(existing_dict[key], list):
                      existing_dict[key] = [existing_dict[key]] # Convert scalar to list if needed
                  existing_dict[key].append(new_value)
        return existing_dict

    def get_best_parameters(self, param_type, server_type, cost):
        # ... (Implementation from your previous code - assumed correct) ...
        # Note: load_results implicitly uses the num_clients from __init__
        results, _ = self.load_results(param_type)
        if results is None or cost not in results: return None
        cost_results = results[cost]
        server_metrics = {}
        if isinstance(cost_results, dict):
            for param_val, server_data in cost_results.items():
                if isinstance(server_data, dict) and server_type in server_data:
                    server_metrics[param_val] = server_data[server_type]
        else: return None
        if not server_metrics: return None
        return self._select_best_hyperparameter(server_metrics)


    def _select_best_hyperparameter(self, param_results):
        # ... (Implementation from your previous code - assumed correct) ...
         best_loss = float('inf')
         best_param = None
         for param_val, metrics in param_results.items():
             if isinstance(metrics, dict) and 'global' in metrics and 'val_losses' in metrics['global'] and metrics['global']['val_losses']:
                  run_median_losses = []
                  # Handle list of lists (multiple runs) vs single list (one run)
                  first_loss_val = metrics['global']['val_losses'][0]
                  is_multi_run = isinstance(first_loss_val, list)

                  if is_multi_run:
                       for run_losses in metrics['global']['val_losses']:
                           if run_losses: run_median_losses.append(np.median(run_losses))
                  elif metrics['global']['val_losses']: # Single run, list of floats
                       run_median_losses.append(np.median(metrics['global']['val_losses']))

                  if run_median_losses:
                      avg_median_loss = np.mean(run_median_losses)
                      if avg_median_loss < best_loss:
                          best_loss = avg_median_loss
                          best_param = param_val
                  # else: print warning handled previously or omit
             # else: print warning handled previously or omit
         if best_param is None: print("Warning: Could not determine best hyperparameter.")
         return best_param


class Experiment:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.default_params = get_parameters_for_dataset(self.config.dataset)
        self.data_dir_root = DATA_DIR

        # Determine the initial target client count for filenames/ResultsManager
        # This uses the CLI override or the default value from the config
        initial_target_num_clients = self.config.num_clients
        if initial_target_num_clients is None:
            initial_target_num_clients = self.default_params.get('default_num_clients')
            if initial_target_num_clients is None:
                print("Warning: 'default_num_clients' not found in config, defaulting to 2 for filename.")
                initial_target_num_clients = 2
        # Validate before passing
        if not isinstance(initial_target_num_clients, int) or initial_target_num_clients <= 0:
             raise ValueError(f"Invalid initial target client count for ResultsManager: {initial_target_num_clients}")

        print(f"Initializing ResultsManager for {initial_target_num_clients} target clients (used for filename).")
        self.results_manager = ResultsManager(
            root_dir=ROOT_DIR,
            dataset=self.config.dataset,
            experiment_type=self.config.experiment_type,
            num_clients=initial_target_num_clients # Pass target count for filename generation
        )
        # This stores the *actual* client count used in a specific initialization, can vary per cost
        self.num_clients_for_run = None

    def run_experiment(self, costs):
        if self.config.experiment_type == ExperimentType.EVALUATION:
            return self._run_final_evaluation(costs)
        elif self.config.experiment_type in [ExperimentType.LEARNING_RATE, ExperimentType.REG_PARAM]:
            return self._run_hyperparam_tuning(costs)
        else:
            raise ValueError(f"Unsupported experiment type: {self.config.experiment_type}")

    def _check_existing_results(self, costs):
        # Load results, ignore metadata for checking runs
        results, _ = self.results_manager.load_results(self.config.experiment_type)
        runs_key = 'runs_tune' if self.config.experiment_type != ExperimentType.EVALUATION else 'runs'
        target_runs = self.default_params.get(runs_key, 1)
        remaining_costs = list(costs)
        completed_runs = 0

        if results is not None:
            completed_costs = set(results.keys())
            remaining_costs = [c for c in costs if c not in completed_costs]

            if completed_costs:
                first_cost = next(iter(completed_costs))
                try:
                     # Navigate results structure carefully to find the list representing runs
                     first_param = next(iter(results[first_cost]))
                     first_server = next(iter(results[first_cost][first_param]))
                     # Determine which loss key indicates runs based on experiment type
                     loss_key = 'val_losses' # Default for tuning
                     if self.config.experiment_type == ExperimentType.EVALUATION:
                          loss_key = 'test_losses'
                     # Access the list and get its length
                     completed_runs = len(results[first_cost][first_param][first_server]['global'][loss_key])
                except (StopIteration, KeyError, IndexError, TypeError, AttributeError) as e: # Catch more potential issues
                     print(f"Warning: Could not reliably determine completed runs from results structure (cost={first_cost}). Error: {e}. Assuming 0.")
                     completed_runs = 0

            print(f"Found {completed_runs}/{target_runs} completed runs in existing results.")
            # Decide if rerunning is needed
            if completed_runs >= target_runs and not remaining_costs:
                 remaining_costs = [] # All costs done, sufficient runs
            elif completed_runs < target_runs: # Runs incomplete, redo all costs
                 remaining_costs = list(costs)
                 print(f"Runs incomplete ({completed_runs}/{target_runs}). Will run for all specified costs.")
            # Else: Runs complete, but some costs were missing (remaining_costs already set)
        else:
             print("No existing results found.")
             remaining_costs = list(costs)

        print(f"Remaining costs to process: {remaining_costs}")
        return results, remaining_costs, completed_runs


    def _run_hyperparam_tuning(self, costs):
        results, remaining_costs, completed_runs = self._check_existing_results(costs)
        runs_tune = self.default_params.get('runs_tune', 1)

        if not remaining_costs and completed_runs >= runs_tune:
            print("All hyperparameter tuning runs are already completed.")
            # results were loaded by check_existing_results if they existed
            return results if results is not None else {}

        remaining_runs_count = runs_tune - completed_runs
        if remaining_runs_count <= 0 and remaining_costs:
             print(f"Runs complete, but costs {remaining_costs} were missing. Rerunning missing costs for {runs_tune} runs.")
             remaining_runs_count = runs_tune
             completed_runs = 0
        elif remaining_runs_count <= 0:
             print("Error state: No remaining costs and no remaining runs needed?") # Should have exited
             return results if results is not None else {}

        print(f"Starting {remaining_runs_count} hyperparameter tuning run(s)...")
        if results is None: results = {}
        num_clients_metadata = None # Store representative client count for metadata

        for run_idx in range(remaining_runs_count):
            current_run_total = completed_runs + run_idx + 1
            print(f"--- Starting Run {current_run_total}/{runs_tune} ---")
            run_meta = {'run_number': current_run_total}
            cost_client_counts = {} # Store actual clients used per cost for this run's metadata

            for cost in remaining_costs:
                print(f"--- Processing Cost: {cost} (Run {current_run_total}) ---")
                try:
                    # Determine num_clients for this specific cost before tuning loop
                    num_clients_this_cost = self._get_final_client_count(self.default_params, cost)
                    cost_client_counts[cost] = num_clients_this_cost
                    if num_clients_metadata is None: num_clients_metadata = num_clients_this_cost
                except Exception as e:
                    print(f"Error determining client count for cost {cost}: {e}. Skipping cost.")
                    continue

                # Determine params to try based on experiment type
                # (Simplified - assumes REG_PARAM added to ExperimentType choices in run.py)
                if self.config.experiment_type == ExperimentType.LEARNING_RATE:
                     server_types_to_tune = self.default_params.get('servers_tune_lr', ['local', 'fedavg'])
                     param_key, fixed_param_key = 'learning_rate', 'reg_param'
                     fixed_param_value = get_default_reg(self.config.dataset)
                     params_to_try_values = self.default_params.get('learning_rates_try', [])
                elif self.config.experiment_type == ExperimentType.REG_PARAM:
                     server_types_to_tune = self.default_params.get('servers_tune_reg', []) # e.g., ['pfedme', 'ditto']
                     param_key, fixed_param_key = 'reg_param', 'learning_rate'
                     fixed_param_value = get_default_lr(self.config.dataset)
                     params_to_try_values = self.default_params.get('reg_params_try', [])
                else:
                      raise ValueError(f"Unsupported tuning type: {self.config.experiment_type}")

                if not params_to_try_values:
                     print(f"Warning: No parameters to try for {self.config.experiment_type}, {self.config.dataset}. Skipping cost {cost}.")
                     continue

                hyperparams_list = [{param_key: p_val, fixed_param_key: fixed_param_value} for p_val in params_to_try_values]
                tuning_results_for_cost = {}

                for hyperparams in hyperparams_list:
                    param_value_being_tuned = hyperparams[param_key]
                    print(f"--- Tuning Param: {param_key}={param_value_being_tuned} ---")
                    # Pass cost, hyperparams, server types. _hyperparameter_tuning calls _initialize_experiment
                    # which uses the correct num_clients internally.
                    server_metrics = self._hyperparameter_tuning(cost, hyperparams, server_types_to_tune)
                    tuning_results_for_cost[param_value_being_tuned] = server_metrics

                # Append results for this cost immediately
                if cost not in results: results[cost] = {}
                for param_val, server_data in tuning_results_for_cost.items():
                    if param_val not in results[cost]: results[cost][param_val] = {}
                    for server_type, metrics in server_data.items():
                        if server_type not in results[cost][param_val]: results[cost][param_val][server_type] = None
                        results[cost][param_val][server_type] = self.results_manager.append_or_create_metric_lists(
                            results[cost][param_val][server_type], metrics
                        )

            # Save results after each completed run across all costs
            print(f"--- Completed Run {current_run_total}/{runs_tune} ---")
            run_meta['cost_specific_client_counts'] = cost_client_counts # Store detailed counts
            self.results_manager.save_results(
                results, self.config.experiment_type,
                client_count=num_clients_metadata, # Pass representative count for primary metadata field
                run_metadata=run_meta
            )
        return results

    # REMOVED num_clients_for_cost parameter
    def _hyperparameter_tuning(self, cost, hyperparams, server_types):
        """Run ONE set of hyperparameters for specified server types."""
        try:
            # Initialization now determines client count internally and sets self.num_clients_for_run
            client_dataloaders = self._initialize_experiment(cost)
            if not client_dataloaders: # Handle case where init returns empty dict
                 print(f"Warning: No client dataloaders initialized for cost {cost}. Skipping hyperparameter set.")
                 return {}
            # self.num_clients_for_run is now set based on len(client_dataloaders)
        except Exception as e:
            print(f"ERROR: Failed to initialize data for cost {cost}. Skipping hyperparameter set. Error: {e}")
            traceback.print_exc()
            return {}

        tracking = {}
        for server_type in server_types:
             lr = hyperparams.get('learning_rate')
             reg_param_val = None
             algo_params_dict = {}
             if server_type in ['pfedme', 'ditto']: # Check specific algorithms needing reg_param
                 reg_param_val = hyperparams.get('reg_param')
                 if reg_param_val is not None: algo_params_dict['reg_param'] = reg_param_val
             trainer_config = self._create_trainer_config(server_type, lr, algo_params_dict)

             print(f"..... Tuning Server: {server_type}, LR: {lr}, Reg: {reg_param_val} .....")
             server = None
             try:
                 server = self._create_server_instance(cost, server_type, trainer_config, tuning=True)
                 # Add the actual number of clients initialized
                 self._add_clients_to_server(server, client_dataloaders)
                 if not server.clients:
                      print(f"Warning: No clients added to server {server_type} for cost {cost}. Skipping training.")
                      tracking[server_type] = {'error': 'No clients added'}
                      continue

                 # Determine number of rounds for tuning run
                 num_tuning_rounds = self.default_params.get('rounds_tune_inner', self.default_params['rounds'])
                 metrics = self._train_and_evaluate(server, num_tuning_rounds) # tuning=True passed via server instance
                 tracking[server_type] = metrics
             except Exception as e:
                 print(f"ERROR during tuning run for server {server_type}, cost {cost}, params {hyperparams}: {e}")
                 traceback.print_exc()
                 tracking[server_type] = {'error': str(e)}
             finally:
                if server: del server
                cleanup_gpu()

        return tracking

    # MODIFIED: Updated save_results call
    def _run_final_evaluation(self, costs):
        results, _, completed_runs = self._check_existing_results(costs)
        target_runs = self.default_params.get('runs', 1)

        if results is not None and completed_runs >= target_runs:
             print(f"Final evaluation already completed with {completed_runs} runs.")
             existing_diversities, _ = self.results_manager.load_results(ExperimentType.DIVERSITY)
             return results, existing_diversities

        remaining_runs_count = target_runs - completed_runs
        print(f"Starting {remaining_runs_count} final evaluation run(s)...")

        if results is None: results = {}
        diversities, _ = self.results_manager.load_results(ExperimentType.DIVERSITY)
        if diversities is None: diversities = {}

        num_clients_metadata = None # Store representative client count for metadata

        for run_idx in range(remaining_runs_count):
            current_run_total = completed_runs + run_idx + 1
            print(f"--- Starting Final Evaluation Run {current_run_total}/{target_runs} ---")
            results_this_run = {}
            diversities_this_run = {}
            run_meta = {'run_number': current_run_total}
            cost_client_counts = {} # Store actual clients used per cost

            for cost in costs:
                print(f"--- Evaluating Cost: {cost} (Run {current_run_total}) ---")
                try:
                    # Determine client count *before* calling _final_evaluation for metadata consistency
                    num_clients_this_cost = self._get_final_client_count(self.default_params, cost)
                    cost_client_counts[cost] = num_clients_this_cost
                    if num_clients_metadata is None: num_clients_metadata = num_clients_this_cost

                    # _final_evaluation calls _initialize_experiment internally
                    experiment_results_for_cost = self._final_evaluation(cost)

                    if 'weight_metrics' in experiment_results_for_cost:
                         diversities_this_run[cost] = experiment_results_for_cost.pop('weight_metrics')
                    results_this_run[cost] = experiment_results_for_cost

                except Exception as e:
                    print(f"ERROR during final evaluation for cost {cost}, run {current_run_total}: {e}")
                    traceback.print_exc()
                    results_this_run[cost] = {'error': str(e)}

            # Append results
            results = self.results_manager.append_or_create_metric_lists(results, results_this_run)
            diversities = self.results_manager.append_or_create_metric_lists(diversities, diversities_this_run)

            # Save results after each completed run
            print(f"--- Completed Final Evaluation Run {current_run_total}/{target_runs} ---")
            run_meta['cost_specific_client_counts'] = cost_client_counts # Add detailed counts
            self.results_manager.save_results(results, ExperimentType.EVALUATION, client_count=num_clients_metadata, run_metadata=run_meta)
            self.results_manager.save_results(diversities, ExperimentType.DIVERSITY, client_count=num_clients_metadata, run_metadata=run_meta)

        return results, diversities


    def _final_evaluation(self, cost):
        """Runs final evaluation for ONE cost value across all servers."""
        tracking = {}
        weight_metrics_acc = {}

        try:
            client_dataloaders = self._initialize_experiment(cost)
            # Store the actual number of clients used for this cost/run
            self.num_clients_for_run = len(client_dataloaders) # Based on actual loaded data
            if self.num_clients_for_run == 0:
                 print(f"Warning: No clients initialized for cost {cost}. Skipping evaluation.")
                 return {'error': "No clients initialized."}
        except Exception as e:
            print(f"ERROR: Failed to initialize data for cost {cost}. Skipping final evaluation. Error: {e}")
            traceback.print_exc()
            return {'error': f"Data initialization failed: {e}"}

        for server_type in ALGORITHMS:
            print(f"..... Evaluating Server: {server_type} for Cost: {cost} .....")
            # --- Get Best Hyperparameters ---
            best_lr = self.results_manager.get_best_parameters(
                 ExperimentType.LEARNING_RATE, server_type, cost
            )
            if best_lr is None:
                 default_lr = get_default_lr(self.config.dataset)
                 print(f"Warning: Best LR not found for {server_type}, cost {cost}. Using default: {default_lr}")
                 best_lr = default_lr

            algo_params_dict = {}
            if server_type in ['pfedme', 'ditto']: # Add other relevant algos
                 best_reg_param = self.results_manager.get_best_parameters(
                      ExperimentType.REG_PARAM, server_type, cost
                 )
                 if best_reg_param is None:
                      default_reg = get_default_reg(self.config.dataset)
                      print(f"Warning: Best Reg Param not found for {server_type}, cost {cost}. Using default: {default_reg}")
                      best_reg_param = default_reg
                 if best_reg_param is not None: # Ensure it's not None before adding
                      algo_params_dict['reg_param'] = best_reg_param

            # --- Run Evaluation ---
            trainer_config = self._create_trainer_config(server_type, best_lr, algo_params_dict)
            server = None
            try:
                server = self._create_server_instance(cost, server_type, trainer_config, tuning=False)
                self._add_clients_to_server(server, client_dataloaders)
                if not server.clients: # Double check clients were added
                     print(f"Warning: No clients added to server {server_type} for evaluation. Skipping.")
                     tracking[server_type] = {'error': 'No clients added'}
                     continue

                metrics = self._train_and_evaluate(server, trainer_config.rounds)
                tracking[server_type] = metrics
                if hasattr(server, 'diversity_metrics') and server.diversity_metrics:
                     weight_metrics_acc = server.diversity_metrics

            except Exception as e:
                print(f"ERROR during final evaluation run for server {server_type}, cost {cost}: {e}")
                traceback.print_exc()
                tracking[server_type] = {'error': str(e)}
            finally:
                if server: del server
                cleanup_gpu()

        if weight_metrics_acc:
            tracking['weight_metrics'] = weight_metrics_acc
        return tracking

    # ADDED: New helper method for determining final client count
    def _get_final_client_count(self, dataset_config, cost):
        """Determines the final client count based on CLI override, defaults, and dataset constraints."""
        cli_override = self.config.num_clients # From ExperimentConfig (passed from run.py)

        # 1. Start with CLI override if provided
        num_clients = cli_override

        # 2. Fall back to default if no override
        if num_clients is None:
            num_clients = dataset_config.get('default_num_clients')
            if num_clients is None: # Needs a default if not in config
                 print("Warning: 'default_num_clients' not found in config, defaulting to 2.")
                 num_clients = 2

        # Ensure num_clients is integer at this point before max check
        if not isinstance(num_clients, int):
             raise TypeError(f"Client count must be an integer, but got {num_clients} (from CLI or default).")

        # 3. Check against max limit defined in config
        max_clients = dataset_config.get('max_clients')
        if max_clients is not None and num_clients > max_clients:
            print(f"Warning: Requested/Default {num_clients} clients exceeds maximum of {max_clients} for {self.config.dataset}. Using {max_clients}.")
            num_clients = max_clients

        # 4. Apply dataset-specific constraints based on cost (for pre-split strategies primarily)
        partitioning_strategy = dataset_config.get('partitioning_strategy')
        if partitioning_strategy == 'pre_split':
             # Example for ISIC/IXI - requires site_mappings in config's source_args
             if self.config.dataset in ['ISIC', 'IXITiny']:
                  site_mappings = dataset_config.get('source_args', {}).get('site_mappings')
                  if site_mappings:
                      if cost in site_mappings:
                          available_for_cost = len(site_mappings[cost])
                          if num_clients > available_for_cost:
                              print(f"Warning: Client count {num_clients} exceeds available sites ({available_for_cost}) for cost '{cost}'. Using {available_for_cost}.")
                              num_clients = available_for_cost
                      else:
                          print(f"Warning: Cost key '{cost}' not found in site_mappings for {self.config.dataset}. Using determined client count {num_clients}.")
                  else:
                       print(f"Warning: 'site_mappings' not found in source_args for pre-split dataset {self.config.dataset}. Cannot apply cost-specific client count limit.")
        # Add checks for other pre-split datasets if their client count depends on cost/files

        # Final sanity check
        if not isinstance(num_clients, int) or num_clients <= 0:
             raise ValueError(f"Calculated invalid number of clients: {num_clients}")

        return num_clients


    def _validate_and_prepare_config(self, dataset_name, dataset_config):
        """Validate configuration and prepare essential parameters."""
        # Call the validation function defined in data_processing
        validate_dataset_config(dataset_config, dataset_name)
        # Extract parameters needed by the pipeline stages
        required_pipeline_keys = ['data_source', 'partitioning_strategy', 'cost_interpretation']
        missing_keys = [k for k in required_pipeline_keys if k not in dataset_config]
        if missing_keys:
             raise ValueError(f"Config for {dataset_name} missing essential pipeline keys: {missing_keys}")

        return {
            'data_source': dataset_config['data_source'],
            'partitioning_strategy': dataset_config['partitioning_strategy'],
            'cost_interpretation': dataset_config['cost_interpretation'],
            'source_args': dataset_config.get('source_args', {}),
            'partitioner_args': dataset_config.get('partitioner_args', {}),
            'partition_scope': dataset_config.get('partition_scope', 'train')
        }

    # REMOVED: _determine_client_count

    def _load_source_data(self, data_source, dataset_name, source_args):
        """Load source data using appropriate loader function."""
        loader_func = DATA_LOADERS.get(data_source)
        if loader_func is None:
            raise NotImplementedError(f"Data loader '{data_source}' not found in DATA_LOADERS.")

        print(f"Loading source data using: {loader_func.__name__}")
        # Pass dataset_name and data_dir_root, pass source_args explicitly
        return loader_func(dataset_name=dataset_name, data_dir=self.data_dir_root, source_args=source_args)


    def _partition_data(self, partitioning_strategy, data_to_partition, num_clients,
                       partition_args, translated_cost, sampling_config=None):
        """Partition data using specified strategy, passing sampling config if needed."""
        partitioner_func = PARTITIONING_STRATEGIES.get(partitioning_strategy)
        if partitioner_func is None:
            raise NotImplementedError(f"Partitioning strategy '{partitioning_strategy}' not found in PARTITIONING_STRATEGIES.")

        print(f"Partitioning using strategy: {partitioner_func.__name__}")

        full_args = {**translated_cost, **partition_args, 'seed': 42}
        if sampling_config:
            full_args['sampling_config'] = sampling_config # Pass sampling config

        if partitioning_strategy == 'pre_split':
             # For pre_split, partitioning is done implicitly by loading per client.
             # Return None to indicate no central partition result.
             return None
        else:
            # Call partitioners that expect the dataset object
            return partitioner_func(dataset=data_to_partition,
                                  num_clients=num_clients,
                                  **full_args)


    def _load_client_specific_data(self, data_source, dataset_name, client_num,
                                  source_args, translated_cost, dataset_config):
        """Load data specific to a client for pre-split datasets."""
        loader_func = DATA_LOADERS.get(data_source)
        if loader_func is None:
            raise NotImplementedError(f"Data loader '{data_source}' not found in DATA_LOADERS.")

        print(f"Loading pre-split data for client {client_num} using: {loader_func.__name__}")
        # Loader needs the specific cost key/suffix from translated_cost
        specific_cost_arg = None
        if 'key' in translated_cost: specific_cost_arg = translated_cost['key']
        elif 'suffix' in translated_cost: specific_cost_arg = translated_cost['suffix']
        else: raise ValueError("Missing cost key/suffix in translated_cost for pre_split loader")

        return loader_func(
            dataset_name=dataset_name,
            data_dir=self.data_dir_root,
            client_num=client_num,
            cost_key_or_suffix=specific_cost_arg,
            config=dataset_config # Pass full config for access to sampling_config etc.
        )

    def _prepare_client_data(self, client_ids_list, partitioning_strategy, partition_result,
                           data_to_partition, dataset_name, dataset_config,
                           translated_cost):
        """Prepare client data dict for DataPreprocessor based on partitioning strategy."""
        client_input_data = {}
        preprocessor_input_type = 'unknown'
        num_clients = len(client_ids_list)

        if partitioning_strategy.endswith('_indices'):
            if partition_result is None: raise ValueError("partition_result cannot be None for index-based partitioning")
            for i, client_id in enumerate(client_ids_list):
                indices = partition_result.get(i, [])
                if not indices : print(f"Warning: Client {client_id} has no data after partitioning.")
                # Ensure data_to_partition is the original dataset object for Subset
                client_input_data[client_id] = Subset(data_to_partition, indices)
            preprocessor_input_type = 'subset'

        elif partitioning_strategy == 'pre_split':
            data_source = dataset_config['data_source']
            source_args = dataset_config.get('source_args', {})
            loaded_data_types = set() # Track types returned by loader

            for i, client_id in enumerate(client_ids_list):
                client_num = i + 1
                try:
                     loaded_data = self._load_client_specific_data(
                         data_source, dataset_name, client_num,
                         source_args, translated_cost, dataset_config
                     )
                     if isinstance(loaded_data, tuple) and len(loaded_data) == 2:
                         X, y = loaded_data
                         client_input_data[client_id] = {'X': X, 'y': y}
                         # Determine input type based on first successful load
                         if preprocessor_input_type == 'unknown':
                              if isinstance(X, np.ndarray) and isinstance(y, np.ndarray): current_type = 'xy_dict'
                              elif isinstance(X, (list, np.ndarray)) and len(X)>0 and isinstance(X[0], str): current_type = 'path_dict'
                              else: current_type = 'unknown'
                              preprocessor_input_type = current_type
                     else:
                          raise TypeError(f"Loader for {data_source} returned unexpected format: {type(loaded_data)}")

                except FileNotFoundError as e:
                      print(f"Error loading data for client {client_id}: {e}. Skipping client.")
                      client_input_data[client_id] = None # Mark as failed/empty
                except Exception as e:
                     print(f"Unexpected error loading data for client {client_id}: {e}. Skipping client.")
                     traceback.print_exc()
                     client_input_data[client_id] = None

            # Filter out clients that failed loading
            client_input_data = {cid: data for cid, data in client_input_data.items() if data is not None}
            if not client_input_data:
                 print(f"Warning: Failed to load data for any clients for pre-split dataset {dataset_name}, cost {translated_cost}. Returning empty data.")
                 return {}, 'unknown' # Return empty dict if all fail
            # Ensure input type was determined
            if preprocessor_input_type == 'unknown':
                 raise RuntimeError("Could not determine preprocessor input type for loaded pre-split data.")

        else:
            raise NotImplementedError(f"Partitioning strategy '{partitioning_strategy}' not supported.")

        return client_input_data, preprocessor_input_type


    # --- Main Initialization Method ---
    def _initialize_experiment(self, cost):
        """Initialize experiment with configuration-driven approach."""
        dataset_name = self.config.dataset
        dataset_config = self.default_params
        print(f"\n--- Initializing Data for {dataset_name} (Cost/Param: {cost}) ---")

        # Stage 1: Config validation and prep
        config_params = self._validate_and_prepare_config(dataset_name, dataset_config)
        data_source = config_params['data_source']
        partitioning_strategy = config_params['partitioning_strategy']
        cost_interpretation = config_params['cost_interpretation']
        source_args = config_params['source_args']
        partitioner_args = config_params['partitioner_args']
        partition_scope = config_params['partition_scope']
        sampling_config = dataset_config.get('sampling_config')

        # Stage 2: Determine FINAL client count for this specific cost/run
        num_clients = self._get_final_client_count(dataset_config, cost)
        client_ids_list = [f'client_{i+1}' for i in range(num_clients)]
        self.num_clients_for_run = num_clients # Store actual count used
        print(f"Determined {num_clients} clients for this run (Cost: {cost}).")

        # Stage 3: Cost translation
        translated_cost = translate_cost(cost, cost_interpretation)
        print(f"Interpreted Cost '{cost}' as: {translated_cost}")

        # Stages 4 & 5: Load & Partition Data
        client_input_data = {}
        preprocessor_input_type = 'unknown'
        data_to_partition = None # Used by index-based partitioners

        if partitioning_strategy.endswith('_indices'):
            source_data_tuple = self._load_source_data(data_source, dataset_name, source_args)
            if partition_scope == 'train': data_to_partition = source_data_tuple[0]
            elif partition_scope == 'all':
                 # ... (handle combining train/test if needed) ...
                 data_to_partition = torch.utils.data.ConcatDataset([source_data_tuple[0], source_data_tuple[1]])
            else: raise ValueError(f"Unsupported partition_scope: {partition_scope}")

            client_partition_result = self._partition_data(
                partitioning_strategy, data_to_partition, num_clients,
                partitioner_args, translated_cost, sampling_config
            )
            # Prepare client data (Subsets)
            client_input_data, preprocessor_input_type = self._prepare_client_data(
                client_ids_list, partitioning_strategy, client_partition_result,
                data_to_partition, dataset_name, dataset_config, translated_cost
            )

        elif partitioning_strategy == 'pre_split':
            # _prepare_client_data handles loop and loading for pre_split
            client_input_data, preprocessor_input_type = self._prepare_client_data(
                client_ids_list, partitioning_strategy, None, None,
                dataset_name, dataset_config, translated_cost
            )
        else:
            raise NotImplementedError(f"Partitioning strategy '{partitioning_strategy}' not implemented.")

        # Stage 6: Preprocessing & DataLoader Creation
        if not client_input_data:
            print(f"Warning: client_input_data is empty after loading/partitioning for {dataset_name}, cost {cost}.")
            return {} # Return empty dict if no clients have data

        print(f"Preprocessing client data (input type: '{preprocessor_input_type}')...")
        preprocessor = DataPreprocessor(dataset_name, self.default_params['batch_size'])
        client_dataloaders = preprocessor.process_client_data(client_input_data, preprocessor_input_type)
        print("--- Data Initialization Complete ---")
        return client_dataloaders



    def _create_trainer_config(self, server_type, learning_rate, algorithm_params = None):
         # Keep existing _create_trainer_config unchanged...
         # Ensure algorithm_params (like reg_param) are passed correctly if needed
         # algorithm_params is expected to be a dict by the time it gets here
         if algorithm_params is None: algorithm_params = {}

         return TrainerConfig(
             dataset_name=self.config.dataset,
             device=DEVICE,
             learning_rate=learning_rate,
             batch_size=self.default_params['batch_size'],
             epochs=self.default_params.get('epochs_per_round', 5), # Use config value
             rounds=self.default_params['rounds'],
             # Requires personal model based on server type needing regularization?
             requires_personal_model= True if server_type in ['pfedme', 'ditto'] else False,
             algorithm_params=algorithm_params # Pass the dict directly
         )

    def _create_model(self, cost, learning_rate):
        """Creates model instance based on dataset config."""
        dataset_config = self.default_params
        model_name = self.config.dataset # Assume model name matches dataset name in configs.py
        fixed_classes = dataset_config.get('fixed_classes')

        print(f"Creating model: {model_name}, Classes: {fixed_classes}")

        # Get model class from models.py (aliased as ms)
        if not hasattr(ms, model_name):
             raise ValueError(f"Model class '{model_name}' not found in models.py module.")
        model_class = getattr(ms, model_name)
        model = model_class()

        # --- Criterion ---
        criterion_map = { # Can also move this map to configs.py if preferred
            'Synthetic': nn.CrossEntropyLoss(), 'Credit': nn.CrossEntropyLoss(),
            'Weather': nn.MSELoss(), 'EMNIST': nn.CrossEntropyLoss(),
            'CIFAR': nn.CrossEntropyLoss(), 'IXITiny': get_dice_score, # Assuming get_dice_score is defined/imported
            'ISIC': nn.CrossEntropyLoss(), 'Heart': nn.CrossEntropyLoss(),
         }
        criterion = criterion_map.get(self.config.dataset)
        if criterion is None:
             raise ValueError(f"Criterion not defined for dataset {self.config.dataset}")

        # --- Optimizer ---
        optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, amsgrad=True, weight_decay=1e-4
        )
        return model, criterion, optimizer


    def _create_server_instance(self, cost, server_type, config: TrainerConfig, tuning: bool):
        # Keep existing _create_server_instance logic
        # It correctly calls the updated _create_model
        learning_rate = config.learning_rate
        # Pass cost if needed by _create_model (it no longer is)
        model, criterion, optimizer = self._create_model(cost, learning_rate)

        globalmodelstate = ModelState(model=model, optimizer=optimizer, criterion=criterion)

        server_mapping = {
             'local': Server, 'fedavg': FedAvgServer,
             # 'fedprox': FedProxServer, 'pfedme': PFedMeServer, 'ditto': DittoServer # Uncomment algos as needed
        }
        # Ensure requested server type is supported
        if server_type not in server_mapping:
             # Check ALGORITHMS list from configs?
             if server_type in ALGORITHMS:
                  raise NotImplementedError(f"Server type '{server_type}' is in ALGORITHMS list but not implemented in server_mapping.")
             else:
                  raise ValueError(f"Unsupported server type '{server_type}'. Available: {list(server_mapping.keys())}")


        server_class = server_mapping[server_type]
        # Pass TrainerConfig and initial ModelState
        server = server_class(config=config, globalmodelstate=globalmodelstate)
        server.set_server_type(server_type, tuning) # Pass tuning flag
        return server

    def _add_clients_to_server(self, server, client_dataloaders):
         # Keep existing _add_clients_to_server logic unchanged...
         for client_id, loaders in client_dataloaders.items(): # Iterate over items
             # Handle potential empty loaders if a client got no data
             if not loaders[0]: # Check if train_loader is empty
                  print(f"Skipping adding client {client_id} to server {server.server_type} as it has no training data.")
                  continue

             if client_id == 'client_joint' and server.server_type != 'local':
                 continue # Skip joint client except for local baseline

             # Create SiteData with the dataloaders for this client
             clientdata = self._create_site_data(client_id, loaders)
             server.add_client(clientdata=clientdata)


    def _create_site_data(self, client_id, loaders):
         # Keep existing _create_site_data logic unchanged...
         # Loaders tuple is (train_loader, val_loader, test_loader)
         return SiteData(
             site_id=client_id,
             train_loader=loaders[0],
             val_loader=loaders[1],
             test_loader=loaders[2]
             # num_samples will be set in SiteData.__post_init__
         )

    # Remove _load_data method (obsolete)
    # def _load_data(self, client_num, cost): ...

    def _train_and_evaluate(self, server, rounds):
         # Keep existing _train_and_evaluate logic unchanged...
         # It handles training loops, validation/testing calls, and metric aggregation.
         # Ensure it uses the correct loss/score lists (val vs test) based on tuning flag.
         print(f"Starting training {'(tuning)' if server.tuning else '(final eval)'} for {server.server_type} over {rounds} rounds...")
         for round_num in range(rounds):
             # Train one round (includes local training and validation)
             train_loss, val_loss, val_score = server.train_round()
             # Optional: Add progress print
             if (round_num + 1) % 10 == 0 or round_num == 0:
                  print(f"  Round {round_num+1}/{rounds} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Score: {val_score:.4f}")


         # --- Final Evaluation after all rounds (only if not tuning) ---
         if not server.tuning:
             print(f"Performing final test evaluation for {server.server_type}...")
             server.test_global() # Evaluate on test sets

         # --- Collect Metrics ---
         global_state = server.serverstate

         # Use val metrics if tuning, test metrics if final evaluation
         global_losses = global_state.val_losses if server.tuning else global_state.test_losses
         global_scores = global_state.val_scores if server.tuning else global_state.test_scores

         metrics = {
             'global': {
                 'losses': global_losses, # List over rounds/epochs
                 'scores': global_scores  # List over rounds/epochs
             },
             'sites': {}
         }

         # Log per-client metrics
         for client_id, client in server.clients.items():
             # Determine which state holds the relevant metrics (personal or global)
             state_to_log = None
             if server.config.requires_personal_model and client.personal_state:
                  state_to_log = client.personal_state
             else:
                  state_to_log = client.global_state

             if state_to_log:
                  client_losses = state_to_log.val_losses if server.tuning else state_to_log.test_losses
                  client_scores = state_to_log.val_scores if server.tuning else state_to_log.test_scores
                  metrics['sites'][client_id] = {
                     'losses': client_losses,
                     'scores': client_scores
                  }
             else:
                  # Should not happen if clients were added correctly
                   metrics['sites'][client_id] = {'losses': [], 'scores': []}


         print(f"Finished training/evaluation for {server.server_type}.")
         # Report final metric
         final_global_score = global_scores[-1] if global_scores else None
         print(f"  Final Global {'Val' if server.tuning else 'Test'} Score: {final_global_score:.4f}")

         return metrics
