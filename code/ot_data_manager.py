# data_manager.py
from configs import *
import models as ms
from helper import set_seeds, cleanup_gpu, move_to_device, get_default_lr
from ot_utils import calculate_sample_loss, DEFAULT_EPS
# Import ResultsManager to load performance data correctly
from pipeline import ResultsManager, Experiment, ExperimentConfig, ExperimentType # Need ResultsManager for loading

class DataManager:
    """
    Handles loading, caching, generation, and processing of activations
    and performance results. Activation generation involves re-running the
    necessary FL experiment segment.
    """
    def __init__(self,
                 num_clients: int, # Target num_clients for performance file lookup
                 activation_dir: str = ACTIVATION_DIR,
                 results_dir: str = RESULTS_DIR,
                 loss_eps: float = DEFAULT_EPS):
        if not isinstance(num_clients, int) or num_clients <= 0:
             raise ValueError("DataManager requires a valid positive integer for num_clients.")
        self.num_clients = num_clients
        self.activation_dir = activation_dir
        self.results_dir = results_dir
        self.loss_eps = loss_eps
        os.makedirs(self.activation_dir, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"DataManager initialized targeting results for {self.num_clients} clients on device {self.device}.")
        # Initialize ResultsManager for fetching hyperparameters later
        # Use a dummy ExperimentType for now, will be replaced in _generate_activations
        self.results_manager_eval = ResultsManager(
            root_dir=ROOT_DIR, dataset="dummy", # Will set dataset later
            experiment_type=ExperimentType.EVALUATION, num_clients=self.num_clients
        )


    # --- Cache Handling (remains the same) ---
    def _get_activation_cache_path(
        self, dataset_name: str, cost: Any, rounds: int, seed: int,
        client_id_1: Union[str, int], client_id_2: Union[str, int],
        loader_type: str, num_clients: int
    ) -> str:
        c1_str = str(client_id_1); c2_str = str(client_id_2)
        if isinstance(cost, (int, float)): cost_str = f"{float(cost):.4f}"
        else: cost_str = str(cost).replace('/', '_')
        dataset_cache_dir = os.path.join(self.activation_dir, dataset_name)
        os.makedirs(dataset_cache_dir, exist_ok=True)
        filename = f"activations_{dataset_name}_nc{num_clients}_cost{cost_str}_r{rounds}_seed{seed}_c{c1_str}v{c2_str}_{loader_type}.pt"
        return os.path.join(dataset_cache_dir, filename)

    def _load_activations_from_cache(self, path: str) -> Optional[Tuple]:
        if not os.path.isfile(path): return None
        try:
            data = torch.load(path, map_location='cpu')
            if isinstance(data, tuple) and len(data) == 6 and \
               data[0] is not None and data[2] is not None and \
               data[3] is not None and data[5] is not None:
                print(f"  Successfully loaded activations from cache: {os.path.basename(path)}")
                return data
            else:
                 warnings.warn(f"Cached data format invalid: {path}")
                 return None
        except Exception as e:
            warnings.warn(f"Failed loading cache {path}: {e}")
            traceback.print_exc(); return None

    def _save_activations_to_cache(self, data: Tuple, path: str) -> None:
        try:
            cpu_data = tuple(d.cpu() if isinstance(d, torch.Tensor) else d for d in data)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(cpu_data, path)
            print(f"  Saved activations to cache: {os.path.basename(path)}")
        except Exception as e:
            warnings.warn(f"Failed saving activations to cache {path}: {e}")
            traceback.print_exc()

    # --- Activation Extraction Helper (remains the same) ---
    def _extract_activations_for_client(
        self,
        client_id_key: Union[str, int],
        model: nn.Module, # Model should be the one trained in _generate_activations
        dataloaders_dict: Dict,
        loader_type: str,
        num_classes: int
        ) -> Optional[Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]]:
        """
        Extracts activations, probabilities, and labels for a SINGLE client using the provided model.
        (Implementation is the same as the previous version provided)
        """
        client_id_str = str(client_id_key)
        print(f"    Extracting activations for client '{client_id_str}' using loader '{loader_type}'...")
        model.to(self.device) # Ensure model is on correct device
        model.eval()

        # Find dataloader
        loader_key = None
        if client_id_key in dataloaders_dict: loader_key = client_id_key
        elif client_id_str in dataloaders_dict: loader_key = client_id_str
        else: # Try int key
             try: int_key = int(client_id_key); loader_key = int_key if int_key in dataloaders_dict else None
             except (ValueError, TypeError): pass
        if loader_key is None: warnings.warn(f"... Dataloader key '{client_id_key}' not found."); return None, None, None

        loader_idx_map = {'train': 0, 'val': 1, 'test': 2}
        loader_idx = loader_idx_map.get(loader_type.lower())
        if loader_idx is None: warnings.warn(f"... Invalid loader_type: '{loader_type}'."); return None, None, None

        try:
             client_loaders = dataloaders_dict[loader_key]
             if not isinstance(client_loaders, (list, tuple)) or len(client_loaders) <= loader_idx:
                 warnings.warn(f"... Invalid loader structure for client {loader_key}."); return None, None, None
             data_loader = client_loaders[loader_idx]
             if data_loader is None or not hasattr(data_loader, 'dataset') or len(data_loader.dataset) == 0:
                 warnings.warn(f"... {loader_type.capitalize()} loader empty for client {loader_key}."); return None, None, None
        except Exception as e: warnings.warn(f"... Error accessing loader: {e}"); traceback.print_exc(); return None, None, None

        # Find final linear layer and set up hooks
        final_linear = None
        possible_names = ['output_layer', 'fc', 'linear', 'classifier', 'output']
        for name in possible_names:
             module = getattr(model, name, None)
             if isinstance(module, torch.nn.Linear): final_linear = module; break
             elif isinstance(module, torch.nn.Sequential) and len(module) > 0 and isinstance(module[-1], torch.nn.Linear): final_linear = module[-1]; break
        if final_linear is None:
             for module in reversed(list(model.modules())):
                 if isinstance(module, torch.nn.Linear): final_linear = module; warnings.warn(f"... Using last Linear layer for hooks."); break
        if final_linear is None: warnings.warn(f"... Could not find linear layer for hooks."); return None, None, None

        # Hook storage and functions
        current_batch_pre_acts_storage, current_batch_post_logits_storage = [], []
        def pre_hook(module, input_data): inp = input_data[0] if isinstance(input_data, tuple) else input_data; current_batch_pre_acts_storage.append(inp.detach())
        def post_hook(module, input_data, output_data): out = output_data[0] if isinstance(output_data, tuple) else output_data; current_batch_post_logits_storage.append(out.detach())
        pre_handle = final_linear.register_forward_pre_hook(pre_hook)
        post_handle = final_linear.register_forward_hook(post_hook)

        # Process data
        all_pre_activations, all_post_activations, all_labels = [], [], []
        try:
            with torch.no_grad():
                for batch_data in data_loader:
                    if isinstance(batch_data, (list, tuple)) and len(batch_data) >= 2: batch_x, batch_y = batch_data[0], batch_data[1]
                    elif isinstance(batch_data, dict): batch_x, batch_y = batch_data.get('features'), batch_data.get('labels'); # Handle dict format if used
                    else: batch_x, batch_y = None, None # Invalid format
                    if batch_x is None or batch_y is None or not isinstance(batch_x, torch.Tensor) or batch_x.shape[0] == 0: warnings.warn(f"... Skipping invalid batch data."); continue

                    batch_x = move_to_device(batch_x, self.device)
                    current_batch_pre_acts_storage.clear(); current_batch_post_logits_storage.clear()
                    _ = model(batch_x) # Forward pass

                    if not current_batch_pre_acts_storage or not current_batch_post_logits_storage: warnings.warn(f"... Hooks failed for a batch."); continue

                    pre_acts_batch = current_batch_pre_acts_storage[0].cpu()
                    post_logits_batch = current_batch_post_logits_storage[0].cpu()

                    # Post-process logits to probabilities
                    post_probs_batch = None
                    if num_classes == 1: post_probs_batch = torch.sigmoid(post_logits_batch).squeeze(-1)
                    elif post_logits_batch.ndim == 1 and num_classes == 2: post_probs_batch = torch.sigmoid(post_logits_batch) # Assume binary if 1D output
                    elif post_logits_batch.ndim == 2 and post_logits_batch.shape[1] == num_classes: post_probs_batch = torch.softmax(post_logits_batch, dim=-1)
                    else: warnings.warn(f"... Unexpected logits shape {post_logits_batch.shape} for K={num_classes}. Skipping batch prob calc."); continue

                    all_pre_activations.append(pre_acts_batch)
                    all_post_activations.append(post_probs_batch)
                    all_labels.append(batch_y.cpu().reshape(-1))
        except Exception as e_proc: warnings.warn(f"... Error during batch processing: {e_proc}"); traceback.print_exc(); pre_handle.remove(); post_handle.remove(); return None, None, None
        finally: pre_handle.remove(); post_handle.remove(); cleanup_gpu()

        # Concatenate results
        if not all_pre_activations: warnings.warn(f"... No batches processed successfully."); return None, None, None
        try:
            final_h = torch.cat(all_pre_activations, dim=0)
            final_p = torch.cat(all_post_activations, dim=0)
            final_y = torch.cat(all_labels, dim=0)
            print(f"    Extraction success for {client_id_str}. Shapes: h={final_h.shape}, p={final_p.shape}, y={final_y.shape}")
            return final_h, final_p, final_y
        except Exception as e_cat: warnings.warn(f"... Error concatenating results: {e_cat}"); traceback.print_exc(); return None, None, None


    # --- Activation Generation (Re-integrates Training) ---
    def _generate_activations(
        self, dataset: str, cost: Any, rounds: int, seed: int,
        client_id_1: Union[str, int], client_id_2: Union[str, int],
        num_clients: int, # Total number of clients in the run configuration
        num_classes: int,
        loader_type: str
    ) -> Optional[Tuple]:
        """
        Generates activations by re-running the FedAvg part of the experiment
        for the specified configuration and extracting activations from the
        resulting model. Relies on activation caching.
        """
        print(f"  Generating activations via experiment re-run for config:")
        print(f"    Dataset: {dataset}, Cost: {cost}, Round: {rounds}, Seed: {seed}")
        print(f"    Num Clients in Run Config: {num_clients}, Loader: {loader_type}")
        print(f"    Target Pair: ({client_id_1}, {client_id_2})")

        server_fedavg = None
        client_dataloaders = None
        exp_handler = None

        try:
            # --- 1. Set up Experiment Handler ---
            # Use ExperimentConfig corresponding to the original run
            # We use EVALUATION type because we need the *final* model state
            # and potentially the hyperparams determined by tuning runs.
            exp_config = ExperimentConfig(dataset=dataset,
                                          experiment_type=ExperimentType.EVALUATION,
                                          num_clients=num_clients) # Use num_clients from the run config
            exp_handler = Experiment(exp_config)

            # --- 2. Get Hyperparameters ---
            # Fetch the hyperparameters that *should have been* used for the FedAvg evaluation run.
            # This uses the ResultsManager initialized with the correct num_clients for filename lookup.
            # We need to temporarily set the dataset for the results manager instance.
            self.results_manager_eval.dataset = dataset
            best_lr_fedavg = self.results_manager_eval.get_best_parameters(
                ExperimentType.LEARNING_RATE, 'fedavg', cost
            )
            if best_lr_fedavg is None:
                 default_lr = get_default_lr(dataset)
                 warnings.warn(f"    Could not find best LR for fedavg, cost {cost}. Using default: {default_lr}")
                 best_lr_fedavg = default_lr

            # Get reg param if needed (though FedAvg doesn't use it, maintain structure)
            algo_params_dict = {}
            # if 'fedavg' in ['pfedme', 'ditto']: # Example if FedAvg needed reg params
            #     best_reg_param = self.results_manager_eval.get_best_parameters(...)
            #     ... handle default ...
            #     algo_params_dict['reg_param'] = best_reg_param

            # --- 3. Initialize Data and Server ---
            # Set seed *before* initializing data for consistency
            set_seeds(seed)
            # Initialize data - this determines the actual number of clients based on cost/config
            client_dataloaders = exp_handler._initialize_experiment(cost)
            if not client_dataloaders:
                 warnings.warn("    Failed to initialize client dataloaders during activation generation.")
                 return None
            actual_num_clients_init = len(client_dataloaders)
            print(f"    Initialized data for {actual_num_clients_init} clients (based on cost={cost}).")

            # Create Trainer Config for FedAvg
            trainer_config_fedavg = exp_handler._create_trainer_config(
                server_type='fedavg',
                learning_rate=best_lr_fedavg,
                algorithm_params=algo_params_dict
            )
            # *** Override rounds to match the requested rounds for activation generation ***
            trainer_config_fedavg.rounds = rounds
            print(f"    Using LR={best_lr_fedavg:.4e}, Rounds={rounds} for FedAvg re-run.")

            # Create Server Instance
            server_fedavg = exp_handler._create_server_instance(
                cost=cost, server_type='fedavg',
                config=trainer_config_fedavg, tuning=False # Run as evaluation mode
            )
            # Add the *actually initialized* clients
            exp_handler._add_clients_to_server(server_fedavg, client_dataloaders)
            if not server_fedavg.clients:
                 warnings.warn("    No clients were added to the FedAvg server instance.")
                 return None

            # --- 4. Train the FedAvg Server ---
            print(f"    Starting FedAvg training for {rounds} rounds...")
            # Use the internal training loop from Experiment class
            # _train_and_evaluate logs progress and handles training/validation
            # We are interested in the *final* model state after 'rounds' iterations.
            metrics_fedavg_run = exp_handler._train_and_evaluate(server_fedavg, rounds)
            if not metrics_fedavg_run:
                 warnings.warn("    FedAvg training and evaluation failed within activation generation.")
                 return None
            final_score = metrics_fedavg_run.get('global',{}).get('scores', [None])[-1]
            print(f"    FedAvg re-run finished. Final Eval Score: {final_score if final_score is not None else 'N/A'}")

            # --- 5. Extract Activations from the Trained Server ---
            # Use the _extract_activations_for_client helper on the server instance
            # that was just trained. Use the 'best_model' state for extraction,
            # as this corresponds to the model evaluated at the end.
            model_to_extract_from = server_fedavg.serverstate.best_model
            print("    Extracting activations using the trained FedAvg model...")

            h1, p1, y1 = self._extract_activations_for_client(
                 client_id_1, model_to_extract_from, client_dataloaders, loader_type, num_classes
            )
            h2, p2, y2 = self._extract_activations_for_client(
                 client_id_2, model_to_extract_from, client_dataloaders, loader_type, num_classes
            )

            # --- 6. Validate and Return ---
            if h1 is None or p1 is None or y1 is None or h2 is None or p2 is None or y2 is None:
                 warnings.warn(f"  Activation extraction failed post-training for pair ({client_id_1}, {client_id_2}).")
                 return None
            else:
                 print("  Activation generation via re-run successful.")
                 return (h1, p1, y1, h2, p2, y2)

        except Exception as e:
            warnings.warn(f"  Unexpected error during activation generation re-run: {e}")
            traceback.print_exc()
            return None
        finally:
            # Clean up resources
            del server_fedavg
            del client_dataloaders
            del exp_handler
            cleanup_gpu()


    # --- Data Processing Helper (remains the same) ---
    def _process_client_data(
        self, h: Optional[torch.Tensor], p_prob_in: Optional[torch.Tensor], y: Optional[torch.Tensor],
        client_id: str, num_classes: int
        ) -> Optional[Dict[str, Optional[torch.Tensor]]]:
        """Processes raw activations, computes loss/weights. (Same implementation as before)"""
        if h is None or y is None: warnings.warn(f"... Processing failed: Client {client_id} missing h/y."); return None
        try: # Ensure CPU tensors
            h_cpu = h.detach().cpu() if isinstance(h, torch.Tensor) else torch.tensor(h).cpu()
            y_cpu = y.detach().cpu().long() if isinstance(y, torch.Tensor) else torch.tensor(y).long().cpu()
            p_prob_cpu = p_prob_in.detach().cpu() if isinstance(p_prob_in, torch.Tensor) else torch.tensor(p_prob_in).cpu() if p_prob_in is not None else None
        except Exception as e: warnings.warn(f"... Processing failed: Client {client_id} tensor conversion error: {e}"); traceback.print_exc(); return None

        n_samples = y_cpu.shape[0]
        if n_samples == 0: warnings.warn(f"... Processing warning: Client {client_id} has zero samples."); return None # Or return empty dict? None safer.

        # Validate probabilities
        p_prob_validated = None
        if p_prob_cpu is not None:
            with torch.no_grad():
                p_prob_float = p_prob_cpu.float()
                if p_prob_float.ndim == 2 and p_prob_float.shape == (n_samples, num_classes):
                    valid_range = torch.all(p_prob_float >= -1e-3) and torch.all(p_prob_float <= 1 + 1e-3)
                    valid_sum = torch.allclose(p_prob_float.sum(dim=1), torch.ones(n_samples), atol=1e-3)
                    if valid_range and valid_sum: p_prob_validated = torch.clamp(p_prob_float, 0.0, 1.0)
                    else: # Renormalize
                        warnings.warn(f"... Processing warning: Client {client_id} probs invalid. Renormalizing.")
                        p_prob_validated = torch.relu(p_prob_float)
                        p_prob_validated = p_prob_validated / p_prob_validated.sum(dim=1, keepdim=True).clamp(min=self.loss_eps)
                        p_prob_validated = torch.clamp(p_prob_validated, 0.0, 1.0)
                elif num_classes == 2 and p_prob_float.ndim == 1 and p_prob_float.shape[0] == n_samples: # Handle binary [N] case
                     p1 = torch.clamp(p_prob_float, 0.0, 1.0); p0 = 1.0 - p1
                     p_prob_validated = torch.stack([p0, p1], dim=1)
                else: warnings.warn(f"... Processing warning: Client {client_id} unexpected prob shape {p_prob_cpu.shape}.")
        else: warnings.warn(f"... Processing warning: Client {client_id} missing probabilities.")

        # Calculate loss and weights
        loss, weights = None, None
        if p_prob_validated is not None: loss = calculate_sample_loss(p_prob_validated, y_cpu, num_classes, self.loss_eps)
        if loss is None or not torch.isfinite(loss).all(): # Handle failed/NaN loss or missing probs
            if loss is not None: warnings.warn(f"... Processing warning: Client {client_id} NaN/Inf loss. Using uniform weights.")
            elif p_prob_validated is None: warnings.warn(f"... Processing warning: Client {client_id} missing/invalid probs. Using uniform weights.")
            weights = torch.ones(n_samples, dtype=torch.float) / n_samples if n_samples > 0 else torch.empty(0, dtype=torch.float)
            loss = torch.full((n_samples,), float('nan'), dtype=torch.float) if n_samples > 0 else torch.empty(0, dtype=torch.float)
        elif loss.sum().item() <= self.loss_eps: # Handle zero loss
             warnings.warn(f"... Processing warning: Client {client_id} zero loss. Using uniform weights.")
             weights = torch.ones(n_samples, dtype=torch.float) / n_samples if n_samples > 0 else torch.empty(0, dtype=torch.float)
        else: weights = loss / loss.sum() # Use loss for weighting

        return {'h': h_cpu, 'p_prob': p_prob_validated, 'y': y_cpu, 'loss': loss, 'weights': weights}


    # --- Public API Methods (get_activations, get_performance) ---
    def get_activations(
        self, dataset_name: str, cost: Any, rounds: int, seed: int,
        client_id_1: Union[str, int], client_id_2: Union[str, int], # The specific pair
        num_clients: int, # The total # clients in the run config (for caching/generation context)
        num_classes: int,
        loader_type: str = 'val',
        force_regenerate: bool = False
    ) -> Optional[Dict[str, Dict[str, Optional[torch.Tensor]]]]:
        """
        Gets processed activations for a specific client pair. Handles caching
        and generation (generation involves re-running FL).
        """
        cid1_str = str(client_id_1); cid2_str = str(client_id_2)
        cache_path = self._get_activation_cache_path(dataset_name, cost, rounds, seed, cid1_str, cid2_str, loader_type, num_clients)
        raw_activations = None; cache_hit = False

        if not force_regenerate:
            raw_activations = self._load_activations_from_cache(cache_path)
            if raw_activations: cache_hit = True

        if raw_activations is None:
            if not cache_hit: print(f"  Activation cache miss: {os.path.basename(cache_path)}")
            if force_regenerate: print(f"  Activation regeneration forced.")

            raw_activations = self._generate_activations( # Call the re-training generator
                dataset=dataset_name, cost=cost, rounds=rounds, seed=seed,
                client_id_1=cid1_str, client_id_2=cid2_str,
                num_clients=num_clients, num_classes=num_classes, loader_type=loader_type
            )
            if raw_activations is not None: self._save_activations_to_cache(raw_activations, cache_path)
            else: warnings.warn(f"Failed to generate raw activations for pair ({cid1_str},{cid2_str})."); return None

        # Unpack and process
        h1_raw, p1_raw, y1_raw, h2_raw, p2_raw, y2_raw = raw_activations
        print(f"  Processing raw activations for pair ({cid1_str}, {cid2_str})...")
        processed_data1 = self._process_client_data(h1_raw, p1_raw, y1_raw, cid1_str, num_classes)
        processed_data2 = self._process_client_data(h2_raw, p2_raw, y2_raw, cid2_str, num_classes)

        if processed_data1 is None or processed_data2 is None:
             warnings.warn(f"Failed to process activations for pair ({cid1_str}, {cid2_str}).")
             return None

        return {cid1_str: processed_data1, cid2_str: processed_data2}


    def get_performance(
        self, dataset_name: str, cost: Any, aggregation_method: str = 'mean'
        ) -> Tuple[float, float]:
        """Loads final performance using ResultsManager. (Same implementation as before)"""
        final_local_score, final_fedavg_score = np.nan, np.nan
        try:
            # Use the results manager instance tied to this DataManager's num_clients
            self.results_manager_eval.dataset = dataset_name # Set correct dataset before loading
            results, metadata = self.results_manager_eval.load_results(ExperimentType.EVALUATION)
            if results is None: warnings.warn(f"Perf results not found (D={dataset_name}, NC={self.num_clients})."); return np.nan, np.nan
            if metadata: print(f"  Perf Meta: ActualC={metadata.get('client_count_used', 'N/A')}, TargetC={metadata.get('filename_target_clients', 'N/A')}")
            if cost not in results: warnings.warn(f"Cost {cost} not in perf results (Keys: {list(results.keys())})."); return np.nan, np.nan

            cost_data = results[cost]
            def _extract_agg_score(data, server_key):
                # Simplified extraction assuming results[cost][server_key]['global']['losses'] structure
                scores = []
                try:
                     server_metrics = data.get(server_key, {})
                     loss_data = server_metrics.get('global', {}).get('losses', []) # Check 'losses' first
                     if not loss_data: loss_data = server_metrics.get('global', {}).get('test_losses', []) # Fallback

                     # Expect list of lists [[run1_last_loss], [run2_last_loss], ...]
                     if loss_data and isinstance(loss_data[0], list):
                          scores = [run_list[-1][0] for run_list in loss_data if run_list and isinstance(run_list[-1], list) and len(run_list[-1]) > 0 and np.isfinite(run_list[-1][0])]
                     # Add handling for other potential formats if necessary
                except Exception as e: warnings.warn(f"Error parsing scores for {server_key}: {e}")

                if not scores: return np.nan
                if aggregation_method.lower() == 'mean': return np.mean(scores)
                elif aggregation_method.lower() == 'median': return np.median(scores)
                else: warnings.warn(f"Invalid agg '{aggregation_method}'. Using mean."); return np.mean(scores)

            final_local_score = _extract_agg_score(cost_data, 'local')
            final_fedavg_score = _extract_agg_score(cost_data, 'fedavg')
            if np.isnan(final_local_score): warnings.warn(f"No valid final 'local' score found for cost {cost}.")
            if np.isnan(final_fedavg_score): warnings.warn(f"No valid final 'fedavg' score found for cost {cost}.")

        except Exception as e: warnings.warn(f"Error loading performance results: {e}"); traceback.print_exc(); return np.nan, np.nan
        return final_local_score, final_fedavg_score