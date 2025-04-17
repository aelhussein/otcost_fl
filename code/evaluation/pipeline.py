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

class ExperimentConfig:
    def __init__(self, dataset, experiment_type, params_to_try=None):
        self.dataset = dataset
        self.experiment_type = experiment_type
        self.params_to_try = self._get_params_test()

    def _get_params_test(self):
        if self.experiment_type == ExperimentType.LEARNING_RATE:
            return get_parameters_for_dataset(self.dataset)['learning_rates_try']
        elif self.experiment_type == ExperimentType.REG_PARAM:
            return get_parameters_for_dataset(self.dataset)['reg_param_try']
        else:
            return None

class ResultsManager:
    def __init__(self, root_dir, dataset, experiment_type):
        self.root_dir = root_dir
        self.dataset = dataset
        self.experiment_type = experiment_type
        self.results_structure = {
            ExperimentType.LEARNING_RATE: {
                'directory': 'lr_tuning',
                'filename_template': f'{dataset}_lr_tuning.pkl',
            },
            ExperimentType.REG_PARAM: {
                'directory': 'reg_param_tuning',
                'filename_template': f'{dataset}_reg_tuning.pkl',
            },
            ExperimentType.EVALUATION: {
                'directory': 'evaluation',
                'filename_template': f'{dataset}_evaluation.pkl'
            },
            ExperimentType.DIVERSITY: {
                'directory': 'diversity',
                'filename_template': f'{dataset}_diversity.pkl'
            }
        }

    def _get_results_path(self, experiment_type):
        experiment_info = self.results_structure[experiment_type]
        return os.path.join(self.root_dir,'results', 
                            experiment_info['directory'], 
                            experiment_info['filename_template'])

    def load_results(self, experiment_type):
        path = self._get_results_path(experiment_type)

        if os.path.exists(path):
            with open(path, 'rb') as f:
                return pickle.load(f)
        return None

    def save_results(self, results, experiment_type):
        path = self._get_results_path(experiment_type)
        with open(path, 'wb') as f:
            pickle.dump(results, f)

    def append_or_create_metric_lists(self, existing_dict, new_dict):
        if existing_dict is None:
            return {k: [v] if not isinstance(v, dict) else 
                   self.append_or_create_metric_lists(None, v)
                   for k, v in new_dict.items()}
        
        for key, new_value in new_dict.items():
            if isinstance(new_value, dict):
                if key not in existing_dict:
                    existing_dict[key] = {}
                existing_dict[key] = self.append_or_create_metric_lists(
                    existing_dict[key], new_value)
            else:
                if key not in existing_dict:
                    existing_dict[key] = []
                existing_dict[key].append(new_value)
        
        return existing_dict

    def get_best_parameters(self, param_type, server_type, cost):
        """Get best hyperparameter value for given server type and cost."""
        results = self.load_results(param_type)
        if results is None or cost not in results:
            return None
        
        cost_results = results[cost]  # Now gives us lr-level dict
        
        # Collect metrics across all learning rates for this server
        server_metrics = {}
        for lr in cost_results.keys():
            if server_type not in cost_results[lr]:
                continue
            server_metrics[lr] = cost_results[lr][server_type]
        
        if not server_metrics:
            return None

        return self._select_best_hyperparameter(server_metrics)

    def _select_best_hyperparameter(self, lr_results):
        """Select best hyperparameter based on minimum loss."""
        best_loss = float('inf')
        best_param = None
        
        for lr, metrics in lr_results.items():
            avg_loss = np.median(metrics['global']['losses'])
                
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_param = lr
                
        return best_param

class Experiment:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.results_manager = ResultsManager(root_dir=ROOT_DIR, dataset=self.config.dataset, experiment_type = self.config.experiment_type)
        self.default_params = get_parameters_for_dataset(self.config.dataset) # Get full config
        # self.data_dir = f'{DATA_DIR}/{self.config.dataset}' # Base data dir from global config
        self.data_dir_root = DATA_DIR # Use global root data dir


    def run_experiment(self, costs):
        # Keep existing run_experiment logic
        if self.config.experiment_type == ExperimentType.EVALUATION:
            return self._run_final_evaluation(costs)
        elif self.config.experiment_type in [ExperimentType.LEARNING_RATE, ExperimentType.REG_PARAM]:
             return self._run_hyperparam_tuning(costs)
        else:
             raise ValueError(f"Unsupported experiment type: {self.config.experiment_type}")


    def _check_existing_results(self, costs):
        # Keep existing _check_existing_results logic
        results = self.results_manager.load_results(self.config.experiment_type)
        runs_key = 'runs_tune' if self.config.experiment_type != ExperimentType.EVALUATION else 'runs'
        target_runs = self.default_params.get(runs_key, 1) # Default to 1 if key not found
        remaining_costs = list(costs) # Copy
        completed_runs = 0

        if results is not None:
            completed_costs = set(results.keys())
            # Filter remaining_costs based on completed ones
            remaining_costs = [c for c in costs if c not in completed_costs]

            if completed_costs:
                # Check number of runs completed for the *first* completed cost
                first_cost = next(iter(completed_costs))
                # Need to navigate the results structure carefully
                if first_cost in results and results[first_cost]:
                     first_param = next(iter(results[first_cost].keys()))
                     if first_param in results[first_cost] and results[first_cost][first_param]:
                          first_server = next(iter(results[first_cost][first_param].keys()))
                          if first_server in results[first_cost][first_param] and \
                             'global' in results[first_cost][first_param][first_server] and \
                             'val_losses' in results[first_cost][first_param][first_server]['global']:
                              # Use val_losses as the indicator for runs
                              completed_runs = len(results[first_cost][first_param][first_server]['global']['val_losses'])
                          else:
                               print(f"Warning: Could not determine completed runs from results structure for cost {first_cost}.")
                     else:
                          print(f"Warning: No param data found for cost {first_cost} in results.")
                else:
                     print(f"Warning: No data found for first completed cost {first_cost} in results.")


            print(f"Found {completed_runs}/{target_runs} completed runs in existing results.")

            # If enough runs are completed for all costs, nothing remains
            if completed_runs >= target_runs:
                 remaining_costs = []
            else:
                 # If runs are incomplete, need to redo all costs for the missing runs
                 remaining_costs = list(costs) # Rerun all costs
                 print(f"Runs incomplete ({completed_runs}/{target_runs}). Rerunning for all costs.")

        else: # No results file found
             print("No existing results found.")
             remaining_costs = list(costs)

        print(f"Remaining costs to process: {remaining_costs}")
        return results, remaining_costs, completed_runs

    def _run_hyperparam_tuning(self, costs):
        # Keep most of _run_hyperparam_tuning logic
        results, remaining_costs, completed_runs = self._check_existing_results(costs)
        runs_tune = self.default_params.get('runs_tune', 1)

        if not remaining_costs and completed_runs >= runs_tune:
             print("All hyperparameter tuning runs are already completed.")
             return results

        remaining_runs_count = runs_tune - completed_runs
        if remaining_runs_count <= 0 and remaining_costs:
             # This case happens if some costs were missing but runs were complete for others
             print(f"Runs complete, but costs {remaining_costs} were missing. Rerunning missing costs for {runs_tune} runs.")
             remaining_runs_count = runs_tune
             completed_runs = 0 # Reset completed runs as we start over for these costs
        elif remaining_runs_count <= 0:
            print("Error state: No remaining costs and no remaining runs needed, should have exited earlier.")
            return results # Should not happen if _check_existing_results is correct


        print(f"Starting {remaining_runs_count} hyperparameter tuning run(s)...")

        # Ensure results is a dict even if loaded as None
        if results is None: results = {}

        for run_idx in range(remaining_runs_count):
            current_run_total = completed_runs + run_idx + 1
            print(f"--- Starting Run {current_run_total}/{runs_tune} ---")
            # results_this_run = {} # Accumulate results for the current run only

            for cost in remaining_costs:
                print(f"--- Processing Cost: {cost} (Run {current_run_total}) ---")
                cost_tracking_for_run = {} # Track results for this cost in this specific run

                # Determine params to try based on experiment type
                if self.config.experiment_type == ExperimentType.LEARNING_RATE:
                    server_types_to_tune = ['local', 'fedavg'] # Add others if needed: 'pfedme', 'ditto'
                    param_key = 'learning_rate'
                    fixed_param_key = 'reg_param'
                    fixed_param_value = get_default_reg(self.config.dataset)
                    params_to_try_values = self.config.params_to_try or self.default_params.get('learning_rates_try', [])
                elif self.config.experiment_type == ExperimentType.REG_PARAM:
                    server_types_to_tune = [] # Tune only relevant algos: 'pfedme', 'ditto'
                    param_key = 'reg_param'
                    fixed_param_key = 'learning_rate'
                    fixed_param_value = get_default_lr(self.config.dataset)
                    params_to_try_values = self.config.params_to_try or self.default_params.get('reg_params_try', [])
                else:
                     raise ValueError(f"Unsupported experiment type for tuning: {self.config.experiment_type}")

                if not params_to_try_values:
                     print(f"Warning: No parameters to try for {self.config.experiment_type} on {self.config.dataset}. Skipping cost {cost}.")
                     continue

                # Prepare hyperparameter combinations
                hyperparams_list = []
                for p_val in params_to_try_values:
                     hp = {param_key: p_val, fixed_param_key: fixed_param_value}
                     hyperparams_list.append(hp)

                # Perform tuning for the list of hyperparameters
                tuning_results_for_cost = {} # {param_value: {server_type: metrics}}
                for hyperparams in hyperparams_list:
                    param_value_being_tuned = hyperparams[param_key]
                    print(f"--- Tuning Param: {param_key}={param_value_being_tuned} ---")
                    # Pass cost, current hyperparams, and relevant servers
                    # _hyperparameter_tuning returns {server_type: metrics} for this param set
                    server_metrics = self._hyperparameter_tuning(cost, hyperparams, server_types_to_tune)
                    tuning_results_for_cost[param_value_being_tuned] = server_metrics

                cost_tracking_for_run = tuning_results_for_cost # Store results for this cost for this run

                # --- Append results for this cost immediately ---
                # Ensure the cost key exists in the main results dict
                if cost not in results:
                     results[cost] = {}

                # Append the results of this run to the main results structure
                # Need to merge results_this_run_cost into results[cost] correctly
                for param_val, server_data in cost_tracking_for_run.items():
                     if param_val not in results[cost]:
                          results[cost][param_val] = {}
                     for server_type, metrics in server_data.items():
                           # Use append_or_create_metric_lists to handle list creation/appending
                           if server_type not in results[cost][param_val]:
                                results[cost][param_val][server_type] = None # Ensure key exists before appending
                           results[cost][param_val][server_type] = self.results_manager.append_or_create_metric_lists(
                                results[cost][param_val][server_type], metrics
                           )

            # Save results after each completed run across all costs
            print(f"--- Completed Run {current_run_total}/{runs_tune} ---")
            self.results_manager.save_results(results, self.config.experiment_type)

        return results


    def _hyperparameter_tuning(self, cost, hyperparams, server_types):
        """Run ONE set of hyperparameters for specified server types."""
        # Initialize data loaders ONCE for this cost value
        # Note: Seeding might need careful handling if called multiple times across runs
        try:
            client_dataloaders = self._initialize_experiment(cost)
        except Exception as e:
            print(f"ERROR: Failed to initialize data for cost {cost}. Skipping hyperparameter set. Error: {e}")
            traceback.print_exc()
            return {} # Return empty results for this hyperparam set

        tracking = {} # {server_type: metrics} for this hyperparam set

        for server_type in server_types:
            lr = hyperparams.get('learning_rate')
            reg_param_val = None # Default for algos not needing it
            if server_type in ['pfedme', 'ditto']: # Add other algos needing reg_param here
                 reg_param_val = hyperparams.get('reg_param') # This should be the actual value

            # Pass reg_param value directly to algorithm_params if needed by TrainerConfig/Client
            algo_params_dict = {}
            if reg_param_val is not None:
                 # Use a consistent key, e.g., 'reg_lambda' or 'mu' depending on algo needs
                 # Check how PFedMeClient/DittoClient expect it in TrainerConfig.algorithm_params
                 algo_params_dict['reg_param'] = reg_param_val # Assuming client expects 'reg_param'

            trainer_config = self._create_trainer_config(server_type, lr, algo_params_dict)

            print(f"..... Tuning Server: {server_type}, LR: {lr}, Reg: {reg_param_val} .....")
            server = None # Ensure server is reset
            try:
                # Create and run server for this specific config
                server = self._create_server_instance(cost, server_type, trainer_config, tuning=True)
                self._add_clients_to_server(server, client_dataloaders)
                # Train for fewer rounds during tuning? Use config maybe. Using full rounds for now.
                num_tuning_rounds = self.default_params.get('rounds_tune_inner', self.default_params['rounds'])
                metrics = self._train_and_evaluate(server, num_tuning_rounds) # Use tuning=True
                tracking[server_type] = metrics
            except Exception as e:
                 print(f"ERROR during tuning for server {server_type}, cost {cost}, params {hyperparams}: {e}")
                 traceback.print_exc()
                 tracking[server_type] = {'error': str(e)} # Log error
            finally:
                if server: del server # Explicitly delete server
                cleanup_gpu() # Clean GPU memory after each server run

        return tracking


    def _run_final_evaluation(self, costs):
         """Run final evaluation with multiple runs using best params."""
         results = {}
         diversities = {} # Separate dict for diversity metrics

         # Load existing results if any (mainly to check completed runs)
         # Note: We don't reuse results here, we always rerun final evals if runs < target
         existing_results = self.results_manager.load_results(ExperimentType.EVALUATION)
         existing_diversities = self.results_manager.load_results(ExperimentType.DIVERSITY)
         completed_runs = 0
         target_runs = self.default_params.get('runs', 1)

         if existing_results:
              # Check completed runs (similar logic to _check_existing_results)
              try:
                   first_cost = next(iter(existing_results.keys()))
                   first_server = next(iter(existing_results[first_cost].keys()))
                   completed_runs = len(existing_results[first_cost][first_server]['global']['test_losses']) # Use test_losses
                   print(f"Found {completed_runs}/{target_runs} completed final evaluation runs.")
              except (StopIteration, KeyError, IndexError, TypeError) as e:
                   print(f"Warning: Could not determine completed runs from existing final eval results. Error: {e}. Starting from run 1.")
                   completed_runs = 0
         else:
              print("No existing final evaluation results found.")


         if completed_runs >= target_runs:
              print(f"Final evaluation already completed with {completed_runs} runs.")
              return existing_results, existing_diversities # Return loaded results

         # Calculate remaining runs
         remaining_runs_count = target_runs - completed_runs
         print(f"Starting {remaining_runs_count} final evaluation run(s)...")

         # Use loaded results/diversities as starting point if runs were partially complete
         results = existing_results if existing_results is not None else {}
         diversities = existing_diversities if existing_diversities is not None else {}

         for run_idx in range(remaining_runs_count):
             current_run_total = completed_runs + run_idx + 1
             print(f"--- Starting Final Evaluation Run {current_run_total}/{target_runs} ---")
             results_this_run = {} # Accumulate results for the current run
             diversities_this_run = {}

             for cost in costs:
                 print(f"--- Evaluating Cost: {cost} (Run {current_run_total}) ---")
                 try:
                      # _final_evaluation performs evaluation for ALL servers for ONE cost
                      # It returns {server_type: metrics} and potentially 'weight_metrics'
                      experiment_results_for_cost = self._final_evaluation(cost)

                      # Separate diversity metrics if present
                      if 'weight_metrics' in experiment_results_for_cost:
                           diversities_this_run[cost] = experiment_results_for_cost.pop('weight_metrics')

                      results_this_run[cost] = experiment_results_for_cost # Store server metrics

                 except Exception as e:
                      print(f"ERROR during final evaluation for cost {cost}, run {current_run_total}: {e}")
                      traceback.print_exc()
                      # Store error marker? Decide how to handle partial run failures.
                      results_this_run[cost] = {'error': str(e)}


             # Append results of this run to the main results dictionaries
             results = self.results_manager.append_or_create_metric_lists(results, results_this_run)
             diversities = self.results_manager.append_or_create_metric_lists(diversities, diversities_this_run)

             # Save results after each completed run
             print(f"--- Completed Final Evaluation Run {current_run_total}/{target_runs} ---")
             self.results_manager.save_results(results, ExperimentType.EVALUATION)
             self.results_manager.save_results(diversities, ExperimentType.DIVERSITY)


         return results, diversities


    def _final_evaluation(self, cost):
        """Runs final evaluation for ONE cost value across all servers."""
        tracking = {} # {server_type: metrics}
        weight_metrics_acc = {} # Accumulate weight metrics if applicable

        # Initialize data loaders ONCE for this cost value
        try:
             client_dataloaders = self._initialize_experiment(cost)
        except Exception as e:
             print(f"ERROR: Failed to initialize data for cost {cost}. Skipping final evaluation for this cost. Error: {e}")
             traceback.print_exc()
             return {'error': f"Data initialization failed: {e}"}

        # Evaluate each algorithm using its best found hyperparameters
        for server_type in ALGORITHMS: # Use globally defined ALGORITHMS
            print(f"..... Evaluating Server: {server_type} for Cost: {cost} .....")

            # Find best LR from tuning results
            best_lr = self.results_manager.get_best_parameters(
                 ExperimentType.LEARNING_RATE, server_type, cost
            )
            if best_lr is None:
                 print(f"Warning: Best LR not found for {server_type}, cost {cost}. Using default: {get_default_lr(self.config.dataset)}")
                 best_lr = get_default_lr(self.config.dataset)

            # Find best Reg Param if applicable
            best_reg_param = None
            algo_params_dict = {}
            if server_type in ['pfedme', 'ditto']: # Add other algos needing reg param here
                 best_reg_param = self.results_manager.get_best_parameters(
                      ExperimentType.REG_PARAM, server_type, cost
                 )
                 if best_reg_param is None:
                      print(f"Warning: Best Reg Param not found for {server_type}, cost {cost}. Using default: {get_default_reg(self.config.dataset)}")
                      best_reg_param = get_default_reg(self.config.dataset)

                 if best_reg_param is not None:
                       # Use a consistent key expected by the client/config
                       algo_params_dict['reg_param'] = best_reg_param


            # Create configuration for the final run
            trainer_config = self._create_trainer_config(server_type, best_lr, algo_params_dict)

            server = None # Ensure server is reset
            try:
                # Create and run server for final evaluation (tuning=False)
                server = self._create_server_instance(cost, server_type, trainer_config, tuning=False)
                self._add_clients_to_server(server, client_dataloaders)
                # Train for the full number of rounds specified in config
                metrics = self._train_and_evaluate(server, trainer_config.rounds) # Use tuning=False
                tracking[server_type] = metrics

                # Collect diversity metrics if available (e.g., from FedAvgServer)
                if hasattr(server, 'diversity_metrics') and server.diversity_metrics:
                     # Store diversity metrics separately, maybe keyed by server type if others calc it too
                     # For now, assuming only FedAvg calculates it and storing under a general key
                     weight_metrics_acc = server.diversity_metrics # Overwrites if multiple servers calc this


            except Exception as e:
                 print(f"ERROR during final evaluation run for server {server_type}, cost {cost}: {e}")
                 traceback.print_exc()
                 tracking[server_type] = {'error': str(e)} # Log error
            finally:
                if server: del server
                cleanup_gpu()

        # Add collected diversity metrics to the results for this cost
        if weight_metrics_acc:
             tracking['weight_metrics'] = weight_metrics_acc

        return tracking


    # --- Refactored Data Initialization ---
    def _initialize_experiment(self, cost):
        """
        Initializes and partitions data based on configuration, returning client dataloaders.
        Follows the 'Balanced Approach' outline.
        """
        dataset_name = self.config.dataset
        dataset_config = self.default_params # Get full config dict
        print(f"\n--- Initializing Data for {dataset_name} (Cost/Param: {cost}) ---")

        # --- Stage 1: Config Loading & Validation ---
        validate_dataset_config(dataset_config, dataset_name) # Basic check
        data_source = dataset_config['data_source']
        partitioning_strategy = dataset_config['partitioning_strategy']
        cost_interpretation = dataset_config['cost_interpretation']
        num_clients_config = dataset_config['num_clients']
        source_args = dataset_config.get('source_args', {})
        partitioner_args = dataset_config.get('partitioner_args', {})
        partition_scope = dataset_config.get('partition_scope', 'train') # Default partition train set

        # --- Stage 2: Determine Clients ---
        if isinstance(num_clients_config, int):
            num_clients = num_clients_config
        elif num_clients_config == 'dynamic':
             # Handle dynamic client count based on cost (for IXI/ISIC)
             if dataset_name == 'IXITiny': num_clients = 3 if cost == 'all' else 2
             elif dataset_name == 'ISIC': num_clients = 4 if cost == 'all' else 2
             else: num_clients = 2 # Default dynamic
        else:
             num_clients = 2 # Default
        client_ids_list = [f'client_{i+1}' for i in range(num_clients)]
        print(f"Target number of clients: {num_clients}")

        # --- Stage 3: Cost Translation ---
        try:
            translated_cost = translate_cost(cost, cost_interpretation)
            print(f"Interpreted Cost '{cost}' as: {translated_cost}")
        except ValueError as e:
            print(f"Error translating cost: {e}")
            raise

        # --- Stage 4 & 5: Data Loading & Partitioning ---
        loader_func = DATA_LOADERS.get(data_source)
        partitioner_func = PARTITIONING_STRATEGIES.get(partitioning_strategy)

        if loader_func is None:
            raise NotImplementedError(f"Data loader '{data_source}' not found in DATA_LOADERS mapping.")
        if partitioner_func is None:
            raise NotImplementedError(f"Partitioning strategy '{partitioning_strategy}' not found in PARTITIONING_STRATEGIES mapping.")

        client_input_data = {} # Data to be passed to preprocessor
        preprocessor_input_type = 'unknown'

        # --- Logic Branch: Centralized Partitioning (e.g., Dirichlet on Torchvision) ---
        if partitioning_strategy.endswith('_indices'): # Convention for strategies returning indices
             print(f"Loading source data using: {loader_func.__name__}")
             # Load the full dataset first (e.g., torchvision returns train_ds, test_ds)
             # Assume loader returns tuple, but might need adjustment based on loader specifics
             source_data_tuple = loader_func(dataset_name=dataset_name, data_dir=self.data_dir_root, source_args=source_args)

             # Select the part of the data to partition based on scope
             if partition_scope == 'train':
                  data_to_partition = source_data_tuple[0] # Assume train is first element
                  print(f"Partitioning the 'train' split ({len(data_to_partition)} samples)")
             elif partition_scope == 'all':
                  # Combine train/test if partition scope is 'all'
                  # This requires careful handling of transforms if they differ
                  print("Warning: Partitioning combined train/test data. Ensure transforms are compatible.")
                  # Example combination (might need adjustment based on dataset type):
                  if isinstance(source_data_tuple[0], torch.utils.data.ConcatDataset):
                       data_to_partition = source_data_tuple[0] # Already combined?
                  elif isinstance(source_data_tuple[0], TorchDataset):
                        data_to_partition = torch.utils.data.ConcatDataset([source_data_tuple[0], source_data_tuple[1]])
                  else: # Cannot easily concat other types
                        raise TypeError("Cannot automatically combine non-TorchDataset for 'all' partition scope.")
                  print(f"Partitioning the 'all' split ({len(data_to_partition)} samples)")
             else:
                  raise ValueError(f"Unsupported partition_scope: {partition_scope}")


             print(f"Partitioning using strategy: {partitioner_func.__name__}")
             # Call the partitioner (e.g., partition_dirichlet_indices)
             partition_args = {**translated_cost, **partitioner_args, 'seed': 42} # Add seed
             client_indices_map = partitioner_func(
                  dataset=data_to_partition,
                  num_clients=num_clients,
                  **partition_args # Pass alpha etc.
             )

             # Create Subset objects for each client
             for i, client_id in enumerate(client_ids_list):
                  indices = client_indices_map.get(i, [])
                  if not indices: print(f"Warning: Client {client_id} has no data after partitioning.")
                  client_input_data[client_id] = Subset(data_to_partition, indices)
             preprocessor_input_type = 'subset'

        # --- Logic Branch: Pre-Split Data (Partitioning happens by loading client file) ---
        elif partitioning_strategy == 'pre_split':
             print(f"Loading pre-split data using: {loader_func.__name__}")
             # Loop through clients and load their specific data
             preprocessor_input_type = 'unknown' # Determine based on loader return type
             for i, client_id in enumerate(client_ids_list):
                  # Loader function needs to handle client num and translated cost (suffix/key)
                  client_load_args = {**translated_cost, **source_args} # Combine args
                  loaded_data = loader_func(
                       dataset_name=dataset_name,
                       data_dir=self.data_dir_root,
                       client_num=i+1,
                       config=self.default_params, # Pass full config if loader needs more info
                       **client_load_args # Pass cost suffix/key etc.
                  )

                  # Store data and determine type
                  if isinstance(loaded_data, tuple) and len(loaded_data) == 2:
                       # Assuming loader returns (X, y) where X could be arrays or paths
                       X, y = loaded_data
                       client_input_data[client_id] = {'X': X, 'y': y}
                       # Infer input type for preprocessor
                       current_type = 'unknown'
                       if isinstance(X, np.ndarray) and isinstance(y, np.ndarray):
                            current_type = 'xy_dict'
                       elif isinstance(X, np.ndarray) and isinstance(X[0], str) and isinstance(y, (np.ndarray, list)): # Crude check for paths
                            current_type = 'path_dict'

                       if preprocessor_input_type == 'unknown':
                            preprocessor_input_type = current_type
                       elif preprocessor_input_type != current_type:
                            raise TypeError(f"Inconsistent data types returned by loader for pre-split dataset {dataset_name}")
                  else:
                        raise TypeError(f"Loader for pre-split data source '{data_source}' returned unexpected format: {type(loaded_data)}")

        else:
             raise NotImplementedError(f"Combination of data source '{data_source}' and partitioning strategy '{partitioning_strategy}' is not implemented.")


        # --- Stage 6: Preprocessing & DataLoader Creation ---
        if not client_input_data:
             raise RuntimeError(f"No client data was loaded/partitioned for dataset {dataset_name}")
        if preprocessor_input_type == 'unknown':
             raise RuntimeError(f"Could not determine preprocessor input type for dataset {dataset_name}")

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

        # Instantiate model - check if it needs 'classes' argument
        try:
             if fixed_classes is not None:
                  # Try instantiating with classes argument
                  model = model_class(classes=fixed_classes)
             else:
                  # Try instantiating without classes argument
                  model = model_class()
        except TypeError as e:
             # Handle potential mismatch (e.g., model needs classes but not provided, or vice versa)
             print(f"TypeError during model instantiation for {model_name}: {e}")
             print(" -> Check if 'fixed_classes' in configs.py matches the model's __init__ signature.")
             # As a fallback, try the other way if one failed
             try:
                  if fixed_classes is not None: model = model_class()
                  else:
                       # Maybe try to infer classes? Risky. Better to require config.
                       raise ValueError(f"Model {model_name} likely requires 'classes' argument, but 'fixed_classes' is not set in config.")
             except Exception as e2:
                    raise ValueError(f"Failed to instantiate model {model_name} with or without 'classes' argument. Config/Model mismatch? Original error: {e}, Fallback error: {e2}") from e

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
