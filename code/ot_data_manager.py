# data_manager.py
from configs import *
import models as ms
from helper import set_seeds, cleanup_gpu, move_to_device
from ot_utils import calculate_sample_loss, DEFAULT_EPS
# Import ResultsManager to load performance data correctly
from pipeline import ResultsManager, Experiment, ExperimentConfig, ExperimentType # Need ResultsManager for loading


# Updated DataManager class with model-loading activation generation

class DataManager:
    """
    Handles loading, caching, generation, and processing of activations
    and performance results using saved models.
    """
    def __init__(self,
                 num_clients: int, # Target num_clients for performance file lookup
                 activation_dir: str = ACTIVATION_DIR,
                 results_dir: str = RESULTS_DIR,
                 loss_eps: float = DEFAULT_EPS):
        """
        Initializes the DataManager.
        
        Args:
            num_clients (int): The target number of clients for this experiment run 
                              (used for loading correct performance files)
            activation_dir (str): Path to activation cache directory
            results_dir (str): Path to results directory
            loss_eps (float): Epsilon for numerical stability in loss calculations
        """
        if not isinstance(num_clients, int) or num_clients <= 0:
            raise ValueError("DataManager requires a valid positive integer for num_clients.")
        self.num_clients = num_clients
        self.activation_dir = activation_dir
        self.results_dir = results_dir
        self.loss_eps = loss_eps
        os.makedirs(self.activation_dir, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"DataManager initialized targeting results for {self.num_clients} clients on device {self.device}.")
        
        # ResultsManager for loading performance data and finding model paths
        self.results_manager = None  # Will be initialized per-dataset
    
    def _initialize_results_manager(self, dataset_name):
        """Initialize/update the ResultsManager for a specific dataset."""
        if self.results_manager is None or self.results_manager.dataset != dataset_name:
            self.results_manager = ResultsManager(
                root_dir=ROOT_DIR, 
                dataset=dataset_name,
                experiment_type=ExperimentType.EVALUATION, 
                num_clients=self.num_clients
            )
    
    def _get_activation_cache_path(
        self, dataset_name: str, cost: Any, rounds: int, seed: int,
        client_id_1: Union[str, int], client_id_2: Union[str, int],
        loader_type: str, num_clients: int
    ) -> str:
        """Constructs standardized path for activation cache files."""
        c1_str = str(client_id_1); c2_str = str(client_id_2)
        dataset_cache_dir = os.path.join(self.activation_dir, dataset_name)
        os.makedirs(dataset_cache_dir, exist_ok=True)
        
        # Format cost string consistently
        if isinstance(cost, (int, float)):
            cost_str = f"{float(cost):.4f}"
        else:
            cost_str = str(cost).replace('/', '_')
            
        # Filename includes num_clients from the run context
        filename = f"activations_{dataset_name}_nc{num_clients}_cost{cost_str}_r{rounds}_seed{seed}_c{c1_str}v{c2_str}_{loader_type}.pt"
        return os.path.join(dataset_cache_dir, filename)

    def _load_activations_from_cache(self, path: str) -> Optional[Tuple]:
        """Loads activation data from cache with validation."""
        if not os.path.isfile(path):
            return None
        try:
            data = torch.load(path, map_location='cpu')
            if isinstance(data, tuple) and len(data) == 6 and \
               data[0] is not None and data[3] is not None:
                print(f"  Successfully loaded activations from cache: {os.path.basename(path)}")
                return data
            else:
                warnings.warn(f"Cached data format invalid: {path}")
                return None
        except Exception as e:
            warnings.warn(f"Failed loading cache {path}: {e}")
            return None

    def _save_activations_to_cache(self, data: Tuple, path: str) -> None:
        """Saves activation data to cache."""
        try:
            # Ensure tensors are on CPU before saving
            cpu_data = tuple(d.cpu() if isinstance(d, torch.Tensor) else d for d in data)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(cpu_data, path)
            print(f"  Saved activations to cache: {os.path.basename(path)}")
        except Exception as e:
            warnings.warn(f"Failed saving cache {path}: {e}")

    def _load_model_for_activation_generation(
        self, dataset: str, cost: Any, seed: int, num_clients: int
    ) -> Tuple[Optional[nn.Module], Optional[Dict], Optional[int]]:
        """
        Loads a saved model for activation generation.
        
        Args:
            dataset (str): Dataset name
            cost: Cost parameter used for the run
            seed (int): Seed used for the run
            num_clients (int): Target number of clients from the run config
            
        Returns:
            Tuple containing:
            - Loaded model or None if failed
            - Dict of dataloaders or None if failed
            - Actual number of clients used in the run or None if unknown
        """
        self._initialize_results_manager(dataset)
        print(f"  Loading model for activation generation:")
        print(f"    Dataset: {dataset}, Cost: {cost}, Seed: {seed}, Target Clients: {num_clients}")
        
        # Create temporary experiment config to access helper methods
        temp_exp_config = ExperimentConfig(
            dataset=dataset,
            experiment_type=ExperimentType.EVALUATION,
            num_clients=num_clients
        )
        temp_exp = Experiment(temp_exp_config)
        
        try:
            # Determine actual client count used for this cost
            actual_clients = temp_exp._get_final_client_count(temp_exp.default_params, cost)
            print(f"    Determined actual client count: {actual_clients}")
            
            # Get model path for 'best' model state
            model_path = self.results_manager._get_model_save_path(
                experiment_type=ExperimentType.EVALUATION,
                dataset=dataset,
                num_clients_run=actual_clients,
                cost=cost, 
                seed=seed,
                server_type='fedavg', 
                model_type='best'
            )
            
            # Load model if exists
            if os.path.exists(model_path):
                model_state_dict = torch.load(model_path, map_location=self.device)
                print(f"    Found and loaded model state from: {os.path.basename(model_path)}")
                
                # Initialize model architecture
                model_class = getattr(ms, dataset)  # Model class named after dataset
                model = model_class()
                model.load_state_dict(model_state_dict)
                model.to(self.device)
                model.eval()
                
                # Initialize dataloaders (using the same seed)
                set_seeds(seed)
                dataloaders = temp_exp._initialize_experiment(cost)
                
                return model, dataloaders, actual_clients
            else:
                # Try fallback to 'final' model if 'best' not found
                model_path_alt = self.results_manager._get_model_save_path(
                    experiment_type=ExperimentType.EVALUATION,
                    dataset=dataset, 
                    num_clients_run=actual_clients,
                    cost=cost, 
                    seed=seed,
                    server_type='fedavg', 
                    model_type='final'
                )
                
                if os.path.exists(model_path_alt):
                    print(f"    Best model not found. Using final model state as fallback.")
                    model_state_dict = torch.load(model_path_alt, map_location=self.device)
                    model_class = getattr(ms, dataset)
                    model = model_class()
                    model.load_state_dict(model_state_dict)
                    model.to(self.device)
                    model.eval()
                    
                    # Initialize dataloaders
                    set_seeds(seed)
                    dataloaders = temp_exp._initialize_experiment(cost)
                    
                    return model, dataloaders, actual_clients
                else:
                    warnings.warn(f"No model state found for {dataset}, cost {cost}, seed {seed}")
                    return None, None, actual_clients
        except Exception as e:
            warnings.warn(f"Error loading model for activation generation: {e}")
            traceback.print_exc()
            return None, None, None
        finally:
            del temp_exp

    def _extract_activations_with_model(
        self, model: nn.Module, client_id: str, 
        dataloaders: Dict, loader_type: str, 
        num_classes: int
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Extracts activations for a single client using a loaded model.
        
        Args:
            model: The loaded model to extract activations from
            client_id: The client identifier
            dataloaders: Dictionary of dataloaders
            loader_type: Which loader to use ('train', 'val', 'test')
            num_classes: Number of classes in the dataset
            
        Returns:
            Tuple of (activations, probabilities, labels)
        """
        client_id_str = str(client_id)
        print(f"    Extracting activations for client {client_id_str} using {loader_type} loader")
        
        # Set model to eval mode
        model.to(self.device)
        model.eval()
        
        # Find the appropriate dataloader
        loader_key = None
        if client_id in dataloaders:
            loader_key = client_id
        elif client_id_str in dataloaders:
            loader_key = client_id_str
        else:
            try:
                int_key = int(client_id)
                if int_key in dataloaders:
                    loader_key = int_key
            except ValueError:
                pass
                
        if loader_key is None:
            warnings.warn(f"Client {client_id_str} not found in dataloaders")
            return None, None, None
            
        # Get the correct loader by type
        loader_idx = {'train': 0, 'val': 1, 'test': 2}.get(loader_type.lower())
        if loader_idx is None:
            warnings.warn(f"Invalid loader_type: {loader_type}")
            return None, None, None
            
        try:
            client_loaders = dataloaders[loader_key]
            if not isinstance(client_loaders, (list, tuple)) or len(client_loaders) <= loader_idx:
                warnings.warn(f"Invalid loader structure for client {client_id_str}")
                return None, None, None
                
            data_loader = client_loaders[loader_idx]
            if data_loader is None or len(data_loader.dataset) == 0:
                warnings.warn(f"Empty {loader_type} loader for client {client_id_str}")
                return None, None, None
        except Exception as e:
            warnings.warn(f"Error accessing loader for client {client_id_str}: {e}")
            return None, None, None
            
        # Set up hooks for activation extraction
        final_linear = None
        possible_names = ['output_layer', 'fc', 'linear', 'classifier', 'output']
        
        # Try to find the final linear layer by name
        for name in possible_names:
            module = getattr(model, name, None)
            if isinstance(module, torch.nn.Linear):
                final_linear = module
                break
            elif isinstance(module, torch.nn.Sequential) and len(module) > 0 and isinstance(module[-1], torch.nn.Linear):
                final_linear = module[-1]
                break
                
        # Fallback to searching all modules
        if final_linear is None:
            for module in reversed(list(model.modules())):
                if isinstance(module, torch.nn.Linear):
                    final_linear = module
                    warnings.warn(f"Using last found Linear layer for hooks")
                    break
                    
        if final_linear is None:
            warnings.warn(f"Could not find linear layer for hooking in model")
            return None, None, None
            
        # Set up storage for activations
        current_batch_pre_acts = []
        current_batch_post_logits = []
        all_pre_activations = []
        all_post_activations = []
        all_labels = []
        
        # Define hooks
        def pre_hook(module, input_data):
            inp = input_data[0] if isinstance(input_data, tuple) else input_data
            if isinstance(inp, torch.Tensor):
                current_batch_pre_acts.append(inp.detach())
                
        def post_hook(module, input_data, output_data):
            out = output_data if not isinstance(output_data, tuple) else output_data[0]
            if isinstance(out, torch.Tensor):
                current_batch_post_logits.append(out.detach())
                
        # Register hooks
        pre_handle = final_linear.register_forward_pre_hook(pre_hook)
        post_handle = final_linear.register_forward_hook(post_hook)
        
        # Process data through model
        try:
            with torch.no_grad():
                for batch_data in data_loader:
                    # Extract batch data appropriately
                    if isinstance(batch_data, (list, tuple)) and len(batch_data) >= 2:
                        batch_x, batch_y = batch_data[0], batch_data[1]
                    elif isinstance(batch_data, dict):
                        batch_x = batch_data.get('features')
                        batch_y = batch_data.get('labels')
                        if batch_x is None or batch_y is None:
                            warnings.warn(f"Missing features or labels in batch")
                            continue
                    else:
                        warnings.warn(f"Unexpected batch format")
                        continue
                        
                    if batch_x.shape[0] == 0:
                        continue # Skip empty batches
                        
                    # Move data to device and reset storage
                    batch_x = batch_x.to(self.device)
                    current_batch_pre_acts.clear()
                    current_batch_post_logits.clear()
                    
                    # Forward pass to trigger hooks
                    _ = model(batch_x)
                    
                    # Check if hooks captured data
                    if not current_batch_pre_acts or not current_batch_post_logits:
                        warnings.warn(f"Hooks failed for a batch")
                        continue
                        
                    # Process captured activations
                    pre_acts_batch = current_batch_pre_acts[0].cpu()
                    post_logits_batch = current_batch_post_logits[0].cpu()
                    
                    # Convert logits to probabilities based on task type
                    if num_classes == 1:  # Binary or regression
                        post_probs_batch = torch.sigmoid(post_logits_batch).squeeze(-1)
                    elif post_logits_batch.ndim == 1:  # Already squeezed binary
                        post_probs_batch = torch.sigmoid(post_logits_batch)
                    else:  # Multi-class
                        post_probs_batch = torch.softmax(post_logits_batch, dim=-1)
                        
                    # Collect results
                    all_pre_activations.append(pre_acts_batch)
                    all_post_activations.append(post_probs_batch)
                    all_labels.append(batch_y.cpu().reshape(-1))  # Ensure 1D
                    
        except Exception as e:
            warnings.warn(f"Error during activation extraction: {e}")
            traceback.print_exc()
            return None, None, None
        finally:
            # Always remove hooks
            pre_handle.remove()
            post_handle.remove()
            
        # Check if any data was collected
        if not all_pre_activations:
            warnings.warn(f"No activations collected for client {client_id_str}")
            return None, None, None
            
        # Concatenate results
        try:
            final_h = torch.cat(all_pre_activations, dim=0)
            final_p = torch.cat(all_post_activations, dim=0)
            final_y = torch.cat(all_labels, dim=0)
            print(f"    Extracted activations: h={final_h.shape}, p={final_p.shape}, y={final_y.shape}")
            return final_h, final_p, final_y
        except Exception as e:
            warnings.warn(f"Error concatenating activation results: {e}")
            return None, None, None

    def _generate_activations(
        self, dataset: str, cost: Any, rounds: int, seed: int,
        client_id_1: Union[str, int], client_id_2: Union[str, int],
        num_clients: int, num_classes: int, loader_type: str
    ) -> Optional[Tuple]:
        """
        Generates activations by loading a pre-trained model and running inference.
        
        Args:
            dataset: Dataset name
            cost: Cost parameter for the run
            rounds: Number of training rounds (for path construction)
            seed: Seed used in the run
            client_id_1, client_id_2: The pair of clients to generate activations for
            num_clients: Target number of clients from run config
            num_classes: Number of classes in dataset
            loader_type: Which loader to use
            
        Returns:
            Tuple of (h1, p1, y1, h2, p2, y2) or None if generation failed
        """
        print(f"  Generating activations for clients ({client_id_1}, {client_id_2})")
        print(f"  Dataset: {dataset}, Cost: {cost}, Rounds: {rounds}, Seed: {seed}")
        
        try:
            # Load the model and dataloaders
            model, dataloaders, actual_clients = self._load_model_for_activation_generation(
                dataset, cost, seed, num_clients
            )
            
            if model is None:
                warnings.warn(f"Failed to load model for activation generation")
                return None
                
            if dataloaders is None:
                warnings.warn(f"Failed to load dataloaders for activation generation")
                return None
                
            # Extract activations for each client
            h1, p1, y1 = self._extract_activations_with_model(
                model, client_id_1, dataloaders, loader_type, num_classes
            )
            
            h2, p2, y2 = self._extract_activations_with_model(
                model, client_id_2, dataloaders, loader_type, num_classes
            )
            
            # Validate all outputs are present
            if h1 is None or p1 is None or y1 is None or h2 is None or p2 is None or y2 is None:
                warnings.warn(f"Activation extraction failed for one or both clients")
                return None
                
            return (h1, p1, y1, h2, p2, y2)
            
        except Exception as e:
            warnings.warn(f"Error during activation generation: {e}")
            traceback.print_exc()
            return None
        finally:
            # Clean up resources
            cleanup_gpu()

    def _process_client_data(
        self, h: Optional[torch.Tensor], p_prob_in: Optional[torch.Tensor], y: Optional[torch.Tensor],
        client_id: str, num_classes: int
    ) -> Optional[Dict[str, Optional[torch.Tensor]]]:
        """
        Processes raw data for one client: validates probs, calculates loss & weights.
        
        Args:
            h: Pre-final layer activations tensor
            p_prob_in: Probabilities tensor (post-softmax/sigmoid)
            y: Labels tensor
            client_id: Client identifier
            num_classes: Number of classes
            
        Returns:
            Dictionary of processed data or None if processing failed
        """
        if h is None or y is None:
            warnings.warn(f"Client {client_id}: Missing essential data for processing")
            return None
            
        # Convert to CPU tensors
        try:
            h_cpu = h.detach().cpu() if isinstance(h, torch.Tensor) else torch.tensor(h).cpu()
            y_cpu = y.detach().cpu().long() if isinstance(y, torch.Tensor) else torch.tensor(y).long().cpu()
            p_prob_cpu = p_prob_in.detach().cpu() if isinstance(p_prob_in, torch.Tensor) else \
                        torch.tensor(p_prob_in).cpu() if p_prob_in is not None else None
        except Exception as e:
            warnings.warn(f"Client {client_id}: Error converting data to CPU tensors: {e}")
            return None
            
        n_samples = y_cpu.shape[0]
        if n_samples == 0:
            warnings.warn(f"Client {client_id}: Zero samples provided")
            return None
            
        # Validate probabilities
        p_prob_validated = None
        if p_prob_cpu is not None:
            with torch.no_grad():
                p_prob_float = p_prob_cpu.float()
                # Handle different probability formats
                if p_prob_float.ndim == 2 and p_prob_float.shape[0] == n_samples and p_prob_float.shape[1] == num_classes:
                    # Check if valid probability distribution
                    if torch.all(p_prob_float >= -1e-5) and torch.all(p_prob_float <= 1 + 1e-5) and \
                       torch.allclose(p_prob_float.sum(dim=1), torch.ones(n_samples), atol=1e-3):
                        p_prob_validated = torch.clamp(p_prob_float, 0.0, 1.0)
                    else:
                        warnings.warn(f"Client {client_id}: Probabilities don't sum to 1. Renormalizing")
                        p_prob_validated = torch.relu(p_prob_float)
                        p_prob_validated = p_prob_validated / p_prob_validated.sum(dim=1, keepdim=True).clamp(min=self.loss_eps)
                        p_prob_validated = torch.clamp(p_prob_validated, 0.0, 1.0)
                # Handle binary case separately
                elif num_classes == 2 and (p_prob_float.ndim == 1 or p_prob_float.shape[1] == 1) and p_prob_float.shape[0] == n_samples:
                    p1 = torch.clamp(p_prob_float.view(-1), 0.0, 1.0)
                    p0 = 1.0 - p1
                    p_prob_validated = torch.stack([p0, p1], dim=1)
                else:
                    warnings.warn(f"Client {client_id}: Unexpected probability shape {p_prob_cpu.shape}")
        else:
            warnings.warn(f"Client {client_id}: No probabilities provided")
            
        # Calculate loss and weights
        loss = None
        weights = None
        
        if p_prob_validated is not None:
            loss = calculate_sample_loss(p_prob_validated, y_cpu, num_classes, self.loss_eps)
            
        if loss is None or not torch.isfinite(loss).all():
            # Handle invalid loss case
            if loss is not None:
                warnings.warn(f"Client {client_id}: NaN/Inf loss detected. Using uniform weights")
            weights = torch.ones_like(y_cpu, dtype=torch.float) / n_samples
            loss = torch.full_like(y_cpu, float('nan'), dtype=torch.float)
        elif loss.sum().item() <= self.loss_eps:
            # Handle zero loss case
            warnings.warn(f"Client {client_id}: Total loss is zero. Using uniform weights")
            weights = torch.ones_like(y_cpu, dtype=torch.float) / n_samples
        else:
            # Use loss for weighting
            weights = loss / loss.sum()
            
        return {
            'h': h_cpu, 
            'p_prob': p_prob_validated, 
            'y': y_cpu, 
            'loss': loss, 
            'weights': weights
        }

    def get_activations(
        self, dataset_name: str, cost: Any, rounds: int, seed: int,
        client_id_1: Union[str, int], client_id_2: Union[str, int],
        num_clients: int, num_classes: int,
        loader_type: str = 'val',
        force_regenerate: bool = False
    ) -> Optional[Dict[str, Dict[str, Optional[torch.Tensor]]]]:
        """
        Gets processed activations for a specific client pair.
        
        Args:
            dataset_name: Name of the dataset
            cost: Cost parameter from the run
            rounds: Number of training rounds
            seed: Random seed used
            client_id_1, client_id_2: Client pair to analyze
            num_clients: Number of clients from the run config
            num_classes: Number of classes in the dataset
            loader_type: Which loader to use
            force_regenerate: Whether to force regeneration
            
        Returns:
            Dictionary of processed client data or None if failed
        """
        self._initialize_results_manager(dataset_name)
        
        cid1_str = str(client_id_1)
        cid2_str = str(client_id_2)
        
        # Check cache first
        cache_path = self._get_activation_cache_path(
            dataset_name, cost, rounds, seed, 
            cid1_str, cid2_str, loader_type, num_clients
        )
        
        raw_activations = None
        if not force_regenerate:
            raw_activations = self._load_activations_from_cache(cache_path)
            
        if raw_activations is None:
            print(f"  Cache {'miss' if not force_regenerate else 'bypass'} for {os.path.basename(cache_path)}")
            
            # Generate activations by loading model
            raw_activations = self._generate_activations(
                dataset=dataset_name, cost=cost, rounds=rounds, seed=seed,
                client_id_1=cid1_str, client_id_2=cid2_str,
                num_clients=num_clients, num_classes=num_classes,
                loader_type=loader_type
            )
            
            if raw_activations is not None:
                self._save_activations_to_cache(raw_activations, cache_path)
            else:
                warnings.warn(f"Failed to generate activations for pair ({cid1_str}, {cid2_str})")
                return None
                
        # Unpack and process the raw activations
        h1_raw, p1_raw, y1_raw, h2_raw, p2_raw, y2_raw = raw_activations
        
        print(f"  Processing activations for clients ({cid1_str}, {cid2_str})")
        processed_data1 = self._process_client_data(h1_raw, p1_raw, y1_raw, cid1_str, num_classes)
        processed_data2 = self._process_client_data(h2_raw, p2_raw, y2_raw, cid2_str, num_classes)
        
        if processed_data1 is None or processed_data2 is None:
            warnings.warn(f"Failed to process activations for pair ({cid1_str}, {cid2_str})")
            return None
            
        return {cid1_str: processed_data1, cid2_str: processed_data2}

    def get_performance(
        self, dataset_name: str, cost: Any, aggregation_method: str = 'mean'
    ) -> Tuple[float, float]:
        """
        Loads final performance metrics.
        
        Args:
            dataset_name: Name of the dataset
            cost: Cost parameter
            aggregation_method: How to aggregate across runs
            
        Returns:
            Tuple of (local_score, fedavg_score)
        """
        self._initialize_results_manager(dataset_name)
        
        final_local_score = np.nan
        final_fedavg_score = np.nan
        
        try:
            # Load results via ResultsManager
            results, metadata = self.results_manager.load_results(ExperimentType.EVALUATION)
            
            if results is None:
                warnings.warn(f"No performance results found for {dataset_name}")
                return np.nan, np.nan
                
            if cost not in results:
                warnings.warn(f"Cost {cost} not found in results for {dataset_name}")
                return np.nan, np.nan
                
            cost_data = results[cost]
            
            # Extract and aggregate scores
            def _extract_score(server_type):
                try:
                    server_metrics = cost_data.get(server_type, {})
                    global_metrics = server_metrics.get('global', {})
                    losses = global_metrics.get('losses', [])
                    
                    if not losses:
                        return np.nan
                        
                    # Handle different results formats
                    if isinstance(losses[0], list):
                        # Multiple runs format
                        final_losses = [run[-1] for run in losses if run]
                    else:
                        # Single run format
                        final_losses = [losses[-1]]
                        
                    if not final_losses:
                        return np.nan
                        
                    # Aggregate using specified method
                    if aggregation_method.lower() == 'mean':
                        return np.mean(final_losses)
                    elif aggregation_method.lower() == 'median':
                        return np.median(final_losses)
                    else:
                        warnings.warn(f"Unknown aggregation method '{aggregation_method}'. Using mean.")
                        return np.mean(final_losses)
                except Exception as e:
                    warnings.warn(f"Error extracting {server_type} score: {e}")
                    return np.nan
                    
            final_local_score = _extract_score('local')
            final_fedavg_score = _extract_score('fedavg')
            
        except Exception as e:
            warnings.warn(f"Error loading performance results: {e}")
            traceback.print_exc()
            
        return final_local_score, final_fedavg_score