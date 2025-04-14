from configs import *
from data_processing import *
from servers import *
from helper import *
from losses import *
import models as ms
from performance_logging import *


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
        self.default_params = get_parameters_for_dataset(self.config.dataset)
        self.data_dir = f'{DATA_DIR}/{self.config.dataset}'
        #self.logger = performance_logger.get_logger(self.config.dataset, self.config.experiment_type)

    def run_experiment(self, costs):
        if self.config.experiment_type == ExperimentType.EVALUATION:
            return self._run_final_evaluation(costs)
        else:
            return self._run_hyperparam_tuning(costs)
            
    def _check_existing_results(self, costs):
        """Check existing results and return remaining work to be done."""
        results = self.results_manager.load_results(self.config.experiment_type)
        remaining_costs = costs
        completed_runs = 0
        
        if results is not None:
            # Check which costs have been completed
            completed_costs = set(results.keys())
            remaining_costs = list(set(costs) - completed_costs)
            
            # Check number of completed runs if any results exist
            if completed_costs: 
                # Check the first cost that was completed to determine number of runs
                first_cost = next(iter(completed_costs))
                # Count number of elements in any metric list to determine completed runs
                first_param = next(iter(results[first_cost].keys()))
                first_server = next(iter(results[first_cost][first_param].keys()))
                completed_runs = len(results[first_cost][first_param][first_server]['global']['losses'])
                
        #self.logger.info(f"Found {completed_runs} completed runs")
        
        remaining_runs = self.default_params['runs_tune'] - completed_runs
        if remaining_runs > 0:
            remaining_costs = costs
        #self.logger.info(f"Remaining costs to process: {remaining_costs}")
        
        return results, remaining_costs, completed_runs

    def _run_hyperparam_tuning(self, costs):
        """Run LR or Reg param tuning with multiple runs"""
        results, remaining_costs, completed_runs = self._check_existing_results(costs)
        
        # If no costs remain and all runs are completed, return existing results
        if not remaining_costs and completed_runs >= self.default_params['runs_tune']:
            #self.logger.info("All experiments are already completed")
            return results
            
        # Calculate remaining runs
        remaining_runs = self.default_params['runs_tune'] - completed_runs
        
        for run in range(remaining_runs):
            current_run = completed_runs + run + 1
            #self.logger.info(f"Starting run {current_run}/{self.default_params['runs_tune']}")
            results_run = {}
            
            for cost in remaining_costs:
                tracking = {}
                if self.config.experiment_type == ExperimentType.LEARNING_RATE:
                    server_types = ['local', 'fedavg', 'pfedme', 'ditto']
                    hyperparams_list = [{'learning_rate': lr, "reg_param":get_default_reg(self.config.dataset)} for lr in self.config.params_to_try]
                else:
                    server_types = ['pfedme', 'ditto']
                    hyperparams_list = [{'reg_param': reg, 'learning_rate': get_default_lr(self.config.dataset)} for reg in self.config.params_to_try]
                for hyperparams in hyperparams_list:
                    param = list(hyperparams.values())[0]
                    tracking[param] = self._hyperparameter_tuning(cost, hyperparams, server_types)
                results_run[cost] = tracking

            results = self.results_manager.append_or_create_metric_lists(results, results_run)
            self.results_manager.save_results(results, self.config.experiment_type)
            
        return results
    
    def _hyperparameter_tuning(self, cost, hyperparams, server_types):
        """Run hyperparameter tuning for specific parameters."""
        client_dataloaders = self._initialize_experiment(cost)
        tracking = {}
        
        for server_type in server_types:
            lr = hyperparams.get('learning_rate')
            if server_type in ['pfedme', 'ditto']:
                reg_param = hyperparams.get('reg_param')
            else:
                reg_param = None
            config = self._create_trainer_config(server_type,lr, reg_param)
            #self.logger.info(f"Starting server type: {server_type}")
            try:
                # Create and run server
                server = self._create_server_instance(cost, server_type, config, tuning=True)
                self._add_clients_to_server(server, client_dataloaders)
                metrics = self._train_and_evaluate(server, server.config.rounds)
                tracking[server_type] = metrics
                
            finally:
                del server
                cleanup_gpu()
        
        return tracking

    def _run_final_evaluation(self, costs):
        """Run final evaluation with multiple runs"""
        results = {}
        diversities = {}
        for run in range(self.default_params['runs']):
            #try:
                print(f"Starting run {run + 1}/{self.default_params['runs']}")
                results_run = {}
                diversities_run = {}
                for cost in costs:
                    experiment_results = self._final_evaluation(cost)
                    diversities_run[cost] = experiment_results['weight_metrics']
                    del experiment_results['weight_metrics']
                    results_run[cost] = experiment_results
                        
                results = self.results_manager.append_or_create_metric_lists(results, results_run)
                diversities = self.results_manager.append_or_create_metric_lists(diversities, diversities_run)
                self.results_manager.save_results(results, self.config.experiment_type)
                self.results_manager.save_results(diversities, ExperimentType.DIVERSITY)
                
            #except Exception as e:
            #    print(f"Run {run + 1} failed with error: {e}")
            #    if results is not None:
            #        self.results_manager.save_results(results,  self.config.experiment_type)
            #        self.results_manager.save_results(diversities, ExperimentType.DIVERSITY)
        
        return results, diversities


    def _final_evaluation(self, cost):
        tracking = {}
        client_dataloaders = self._initialize_experiment(cost)

        for server_type in ALGORITHMS:
            print(f"Evaluating {server_type} model with best hyperparameters")
            lr = self.results_manager.get_best_parameters(
                ExperimentType.LEARNING_RATE, server_type, cost)
            

            if server_type in ['pfedme', 'ditto']:
                reg_param = self.results_manager.get_best_parameters(
                    ExperimentType.REG_PARAM, server_type, cost)
            else:
                reg_param = None
            config = self._create_trainer_config(server_type,lr, reg_param)

            server = self._create_server_instance(cost, server_type, config, tuning = False)
            self._add_clients_to_server(server, client_dataloaders)
            metrics = self._train_and_evaluate(server, config.rounds)
            tracking[server_type] = metrics
            if server_type == 'fedavg':
                tracking['weight_metrics'] = server.diversity_metrics
        return tracking
    

    def _initialize_experiment(self, cost):
        client_data = {}
        client_ids = self._get_client_ids(cost)
        
        for client_id in client_ids:
            client_num = int(client_id.split('_')[1])
            X, y = self._load_data(client_num, cost)
            client_data[client_id] = {'X': X, 'y': y}

        preprocessor = DataPreprocessor(self.config.dataset, self.default_params['batch_size'])
        return preprocessor.process_client_data(client_data)
    
    def _get_client_ids(self, cost):
        CLIENT_NUMS = {'IXITiny': 3, 'ISIC': 4}
        if self.config.dataset in CLIENT_NUMS and cost == 'all':
            CLIENT_NUM = CLIENT_NUMS[self.config.dataset]
        else:
            CLIENT_NUM = 2
        return [f'client_{i}' for i in range(1, CLIENT_NUM + 1)]
    
    def _create_trainer_config(self, server_type, learning_rate, algorithm_params = None):
        return TrainerConfig(
            dataset_name=self.config.dataset,
            device=DEVICE,
            learning_rate=learning_rate,
            batch_size=self.default_params['batch_size'],
            epochs=5,
            rounds=self.default_params['rounds'],
            requires_personal_model= True if server_type in ['pfedme', 'ditto'] else False,
            algorithm_params=algorithm_params
        )

    def _create_model(self, cost, learning_rate):
        if self.config.dataset in ['EMNIST', 'CIFAR']:
            with open(f'{self.data_dir}/CLASSES', 'rb') as f:
                classes_used = pickle.load(f)
            classes = len(set(classes_used[cost][0] + classes_used[cost][1]))
            model = getattr(ms, self.config.dataset)(classes)
        else:
            model = getattr(ms, self.config.dataset)()

        criterion = {
            'Synthetic': nn.CrossEntropyLoss(),
            'Credit': nn.CrossEntropyLoss(),
            'Weather': nn.MSELoss(),
            'EMNIST': nn.CrossEntropyLoss(),
            'CIFAR': nn.CrossEntropyLoss(),
            'IXITiny': get_dice_score,
            'ISIC': nn.CrossEntropyLoss(),
            'Heart': nn.CrossEntropyLoss(),
        }.get(self.config.dataset, None)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            amsgrad=True,
            weight_decay=1e-4
        )
        return model, criterion, optimizer


    def _create_server_instance(self, cost, server_type, config, tuning):
        learning_rate = config.learning_rate
        model, criterion, optimizer = self._create_model(cost, learning_rate)
        globalmodelstate = ModelState(
            model=model,
            optimizer=optimizer,
            criterion=criterion
        )
        server_mapping = {
            'local': Server,
            'fedavg': FedAvgServer,
            'fedprox': FedProxServer,
            'pfedme': PFedMeServer,
            'ditto': DittoServer
        }

        server_class = server_mapping[server_type]
        server = server_class(config=config, globalmodelstate=globalmodelstate)
        server.set_server_type(server_type, tuning)
        return server

    def _add_clients_to_server(self, server, client_dataloaders):
        for client_id in client_dataloaders:
            if client_id == 'client_joint' and server.server_type != 'local':
                continue  # Skip this iteration
            else:
                clientdata = self._create_site_data(client_id, client_dataloaders[client_id])
                server.add_client(clientdata=clientdata)

    def _create_site_data(self, client_id, loaders):
        return SiteData(
            site_id=client_id,
            train_loader=loaders[0],
            val_loader=loaders[1],
            test_loader=loaders[2]
        )

    def _load_data(self, client_num, cost):
        return loadData(self.config.dataset, f'{self.data_dir}', client_num, cost)

    #@log_execution_time
    def _train_and_evaluate(self, server, rounds):
        for round_num in range(rounds):
            server.train_round()        

        if not server.tuning:
            # Final evaluation
            server.test_global()
        
        state = server.serverstate
        
        if server.tuning:
            losses, scores = state.val_losses, state.val_scores 
        else:
            losses, scores = state.test_losses, state.test_scores 
            
        metrics = {
            'global': {
                'losses': losses,
                'scores': scores
            },
            'sites': {}
        }
        
        # Log per-client metrics
        for client_id, client in server.clients.items():
            state = (client.personal_state 
                    if client.personal_state is not None 
                    else client.global_state)
            
            if server.tuning:
                losses, scores = state.val_losses, state.val_scores 
            else:
                losses, scores = state.test_losses, state.test_scores 
                
            metrics['sites'][client_id] = {
                'losses': losses,
                'scores': scores
            }
            
    
        return metrics
