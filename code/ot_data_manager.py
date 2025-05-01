# ot_data_manager.py
import os
import torch
import numpy as np
import warnings
import traceback
from typing import Dict, Optional, Tuple, List, Union, Any

from configs import ROOT_DIR, ACTIVATION_DIR
import models as ms
from helper import set_seeds,  MetricKey
from ot_utils import calculate_sample_loss, DEFAULT_EPS

# Import the new ResultsManager and related classes
from results_manager import ResultsManager, ExperimentType

class OTDataManager:
    """
    Handles loading, caching, generation, and processing of activations
    and performance results using saved models.
    """
    def __init__(self,
                 num_clients: int,
                 activation_dir: str = ACTIVATION_DIR,
                 results_dir: str = None,
                 loss_eps: float = DEFAULT_EPS):
        """
        Initializes the DataManager.
        
        Args:
            num_clients (int): The target number of clients for this experiment run
            activation_dir (str): Path to activation cache directory
            results_dir (str): Path to results directory (unused, retained for API compatibility)
            loss_eps (float): Epsilon for numerical stability in loss calculations
        """
        self.num_clients = num_clients
        self.activation_dir = activation_dir
        self.loss_eps = loss_eps
        os.makedirs(self.activation_dir, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # ResultsManager is initialized per-dataset in methods that need it
        self.results_manager = None
        
        pass#print(f"DataManager initialized targeting results for {self.num_clients} clients on device {self.device}.")
    
    def _initialize_results_manager(self, dataset_name: str):
        """Initialize ResultsManager for a specific dataset."""
        if self.results_manager is None or self.results_manager.path_builder.dataset != dataset_name:
            self.results_manager = ResultsManager(
                root_dir=ROOT_DIR,
                dataset=dataset_name, 
                num_target_clients=self.num_clients
            )
    
    def _get_activation_cache_path(
        self, dataset_name: str, cost: Any, seed: int,
        client_id_1: Union[str, int], client_id_2: Union[str, int],
        loader_type: str, num_clients: int
    ) -> str:
        """Constructs standardized path for activation cache files."""
        c1_str = str(client_id_1)
        c2_str = str(client_id_2)
        dataset_cache_dir = os.path.join(self.activation_dir, dataset_name)
        os.makedirs(dataset_cache_dir, exist_ok=True)
        
        # Format cost string consistently
        if isinstance(cost, (int, float)):
            cost_str = f"{float(cost):.4f}"
        else:
            cost_str = str(cost).replace('/', '_')
            
        # Use 'r0' explicitly to indicate round0 model
        filename = f"activations_{dataset_name}_nc{num_clients}_cost{cost_str}_r0_seed{seed}_c{c1_str}v{c2_str}_{loader_type}.pt"
        return os.path.join(dataset_cache_dir, filename)

    def _load_activations_from_cache(self, path: str) -> Optional[Tuple]:
        """Loads activation data from cache with validation."""
        if not os.path.isfile(path):
            return None
        
        try:
            data = torch.load(path, map_location='cpu')
            if isinstance(data, tuple) and len(data) == 6 and data[0] is not None and data[3] is not None:
                pass#print(f"  Successfully loaded activations from cache: {os.path.basename(path)}")
                return data
        except Exception as e:
            pass#print(f"Failed loading activation cache {path}: {e}")
        
        return None

    def _save_activations_to_cache(self, data: Tuple, path: str) -> None:
        """Saves activation data to cache."""
        try:
            # Ensure tensors are on CPU before saving
            cpu_data = tuple(d.cpu() if isinstance(d, torch.Tensor) else d for d in data)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(cpu_data, path)
            pass#print(f"  Saved activations to cache: {os.path.basename(path)}")
        except Exception as e:
            pass#print(f"Failed saving activation cache {path}: {e}")

    def _get_model_path(self, dataset: str, cost: Any, seed: int, model_type: str = 'round0') -> str:
        """
        Gets path to saved model state dict.
        
        Args:
            dataset: Dataset name
            cost: Cost parameter
            seed: Random seed
            model_type: Model type to load ('round0', 'best', 'final')
            
        Returns:
            str: Path to model file
        """
        self._initialize_results_manager(dataset)
        # Get both records and metadata from results_manager
        _, metadata = self.results_manager.load_results(ExperimentType.EVALUATION)
        # Determine actual client count used for this run
        actual_clients = self.num_clients  # Default to target count
        
        # Extract client count from metadata if available
        if metadata and 'cost_client_counts' in metadata:
            cost_counts = metadata.get('cost_client_counts', {})
            if cost in cost_counts:
                actual_clients = cost_counts[cost]
                pass#print(f"Found actual client count in metadata: {actual_clients} for cost {cost}")
        
        # Get path from ResultsManager
        return self.results_manager.path_builder.get_model_save_path(
            num_clients_run=actual_clients,
            cost=cost,
            seed=seed,
            server_type='fedavg',
            model_type=model_type  # Use round0 as specified
        )

    def _load_model_for_activation_generation(
        self, dataset: str, cost: Any, seed: int, num_clients: int, model_type: str
    ) -> Tuple[Optional[torch.nn.Module], Optional[Dict], int]:
        """
        Loads a saved model for activation generation.
        
        Args:
            dataset: Dataset name
            cost: Cost parameter
            seed: Seed used in the run
            num_clients: Target client count
            
        Returns:
            Tuple of (model, dataloaders, actual_client_count)
        """
        pass#print(f"  Loading model for activation generation:")
        pass#print(f"    Dataset: {dataset}, Cost: {cost}, Seed: {seed}, Target Clients: {num_clients}")
        
        # Try ONLY the round0 model as required
        model_path = self._get_model_path(dataset, cost, seed, model_type)
                    
        if not os.path.exists(model_path):
            pass#print(f"Round0 model not found for {dataset}, cost {cost}, seed {seed}")
            return None, None, num_clients
            
        # Load model
        try:
            # Load state dict and instantiate model
            model_state_dict = torch.load(model_path, map_location=self.device)
            pass#print(f"    Loaded {model_type} model state from: {os.path.basename(model_path)}")
            
            # Initialize model architecture based on dataset name
            model_name = 'Synthetic' if 'Synthetic_' in dataset else dataset
            model_class = getattr(ms, model_name)
            model = model_class()
            model.load_state_dict(model_state_dict)
            model.to(self.device)
            model.eval()
            
            # Create temporary experiment config to get dataloaders
            from pipeline import ExperimentConfig, Experiment
            temp_config = ExperimentConfig(dataset=dataset, experiment_type=ExperimentType.EVALUATION, num_clients=num_clients)
            temp_exp = Experiment(temp_config)
            
            # Set seed for reproducibility
            set_seeds(seed)
            
            # Get dataloaders
            dataloaders = temp_exp.data_manager.get_dataloaders(cost=cost, run_seed=seed, num_clients_override=num_clients)
            actual_clients = len(dataloaders) if dataloaders else num_clients
            
            return model, dataloaders, actual_clients
            
        except Exception as e:
            #print(f"Error loading model/dataloaders: {e}")
            traceback.print_exc()
            return None, None, num_clients

    def _extract_activations_with_model(
        self, model: torch.nn.Module, client_id: str, 
        dataloaders: Dict, loader_type: str, 
        num_classes: int
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Extracts activations for a client using the loaded model.
        """
        client_id_str = str(client_id)
        pass#print(f"    Extracting activations for client {client_id_str} using {loader_type} loader")
        
        # Set model to eval mode
        model.to(self.device)
        model.eval()
        
        # Find the appropriate dataloader
        loader_key = None
        for key in [client_id, client_id_str]:
            if key in dataloaders:
                loader_key = key
                break
                
        if loader_key is None:
            # Try numeric key
            try:
                int_key = int(client_id)
                if int_key in dataloaders:
                    loader_key = int_key
            except ValueError:
                pass
                
        if loader_key is None:
            pass#print(f"Client {client_id_str} not found in dataloaders")
            return None, None, None
            
        # Get loader by type
        loader_idx = {'train': 0, 'val': 1, 'test': 2}.get(loader_type.lower())
        if loader_idx is None or not isinstance(dataloaders[loader_key], (list, tuple)) or len(dataloaders[loader_key]) <= loader_idx:
            pass#print(f"Invalid loader structure for client {client_id_str}")
            return None, None, None
            
        data_loader = dataloaders[loader_key][loader_idx]
        if data_loader is None or len(data_loader.dataset) == 0:
            pass#print(f"Empty {loader_type} loader for client {client_id_str}")
            return None, None, None
            
        # Find final linear layer for hooks
        final_linear = None
        for name in ['output_layer', 'fc2', 'linear', 'classifier', 'output']:
            module = getattr(model, name, None)
            if isinstance(module, torch.nn.Linear):
                final_linear = module
                break
            elif isinstance(module, torch.nn.Sequential) and len(module) > 0 and isinstance(module[-1], torch.nn.Linear):
                final_linear = module[-1]
                break
                
        if final_linear is None:
            # Fallback: search all modules
            for module in reversed(list(model.modules())):
                if isinstance(module, torch.nn.Linear):
                    final_linear = module
                    pass#print(f"Using last found Linear layer for hooks")
                    break
                    
        if final_linear is None:
            pass#print(f"Could not find linear layer for hooking in model")
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
                            continue
                    else:
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
        finally:
            # Always remove hooks
            pre_handle.remove()
            post_handle.remove()
            
        # Check if any data was collected
        if not all_pre_activations:
            pass#print(f"No activations collected for client {client_id_str}")
            return None, None, None
            
        # Concatenate results
        try:
            final_h = torch.cat(all_pre_activations, dim=0)
            final_p = torch.cat(all_post_activations, dim=0)
            final_y = torch.cat(all_labels, dim=0)
            pass#print(f"    Extracted activations: h={final_h.shape}, p={final_p.shape}, y={final_y.shape}")
            return final_h, final_p, final_y
        except Exception as e:
            pass#print(f"Error concatenating activation results: {e}")
            return None, None, None

    def _generate_activations(
        self, dataset: str, cost: Any, seed: int,
        client_id_1: Union[str, int], client_id_2: Union[str, int],
        num_clients: int, num_classes: int, loader_type: str, model_type: str
    ) -> Optional[Tuple]:
        """
        Generates activations by loading a pre-trained model and running inference.
        
        Returns:
            Tuple of (h1, p1, y1, h2, p2, y2) or None if generation failed
        """
        pass#print(f"  Generating activations for clients ({client_id_1}, {client_id_2})")
        pass#print(f"  Dataset: {dataset}, Cost: {cost},  Seed: {seed}")
        
        try:
            # Load the model and dataloaders
            model, dataloaders, actual_clients = self._load_model_for_activation_generation(
                dataset, cost, seed, num_clients, model_type
            )
            
            if model is None or dataloaders is None:
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
                return None
                
            return (h1, p1, y1, h2, p2, y2)
            
        except Exception as e:
            pass#print(f"Error during activation generation: {e}")
            traceback.print_exc()
            return None

    def _process_client_data(
        self, h: Optional[torch.Tensor], p_prob_in: Optional[torch.Tensor], y: Optional[torch.Tensor],
        client_id: str, num_classes: int
    ) -> Optional[Dict[str, Optional[torch.Tensor]]]:
        """
        Processes raw data for one client: validates probs, calculates loss & weights.
        
        Args:
            h: Feature activations tensor
            p_prob_in: Model probability outputs tensor
            y: Ground truth labels tensor
            client_id: Client identifier
            num_classes: Number of classes in the dataset
            
        Returns:
            Dictionary of processed tensors or None if processing fails
        """
        if h is None or y is None:
            return None
            
        # Convert to CPU tensors
        h_cpu = h.detach().cpu() if isinstance(h, torch.Tensor) else torch.tensor(h).cpu()
        y_cpu = y.detach().cpu().long() if isinstance(y, torch.Tensor) else torch.tensor(y).long().cpu()
        p_prob_cpu = p_prob_in.detach().cpu() if isinstance(p_prob_in, torch.Tensor) else \
                    torch.tensor(p_prob_in).cpu() if p_prob_in is not None else None
            
        n_samples = y_cpu.shape[0]
        if n_samples == 0:
            return None
            
        # Validate and normalize probabilities
        p_prob_validated = None
        if p_prob_cpu is not None:
            with torch.no_grad():
                # Convert to float for numerical stability
                p_prob_float = p_prob_cpu.float()
                # Case 1: Standard multiclass probabilities [N, K]
                if p_prob_float.ndim == 2 and p_prob_float.shape[0] == n_samples and p_prob_float.shape[1] == num_classes:
                    # Check if already approximately valid probability distribution
                    row_sums = p_prob_float.sum(dim=1)
                    is_valid_dist = torch.all(p_prob_float >= -1e-5) and torch.all(p_prob_float <= 1 + 1e-5) and \
                                    torch.allclose(row_sums, torch.ones(n_samples), atol=1e-3)
                    
                    if is_valid_dist:
                        # Already valid, just clamp to ensure exact [0,1] range
                        p_prob_validated = torch.clamp(p_prob_float, 0.0, 1.0)
                    else:
                        # Need to normalize - first ensure non-negative values
                        p_prob_temp = torch.relu(p_prob_float)
                        # Compute row sums with epsilon to avoid division by zero
                        row_sums = p_prob_temp.sum(dim=1, keepdim=True).clamp(min=self.loss_eps)
                        # Normalize and clamp
                        p_prob_validated = torch.clamp(p_prob_temp / row_sums, 0.0, 1.0)
                
                # Case 2: Binary classification with single value [N] or [N,1]
                elif num_classes == 2:
                    if (p_prob_float.ndim == 1 and p_prob_float.shape[0] == n_samples) or \
                    (p_prob_float.ndim == 2 and p_prob_float.shape[0] == n_samples and p_prob_float.shape[1] == 1):
                        # Reshape to ensure [N] format first
                        p_prob_1d = p_prob_float.view(-1)
                        # Clamp to [0,1] range
                        p1 = torch.clamp(p_prob_1d, 0.0, 1.0)
                        p0 = 1.0 - p1
                        # Stack to create [N,2] format
                        p_prob_validated = torch.stack([p0, p1], dim=1)
                
                # Case 3: Logits (pre-softmax) - add this if needed
                # elif is_logits condition (e.g., large values or values outside [0,1]):
                #     p_prob_validated = torch.softmax(p_prob_float, dim=1)
                
                # If no cases matched, p_prob_validated remains None
                if p_prob_validated is None:
                    print(f"Warning: Unable to validate probability tensor with shape {p_prob_float.shape} "
                        f"for client {client_id} (n_samples={n_samples}, num_classes={num_classes})")
        
        # Calculate loss and weights
        loss = None
        weights = None
        
        if p_prob_validated is not None:
            loss = calculate_sample_loss(p_prob_validated, y_cpu, num_classes, self.loss_eps)
        
        # Handle weights assignment
        if loss is None or not torch.isfinite(loss).all() or loss.sum().item() <= self.loss_eps:
            # Use uniform weights if loss is invalid
            weights = torch.ones(n_samples, dtype=torch.float32) / n_samples
            loss = torch.full_like(y_cpu, float('nan'), dtype=torch.float32) if loss is None else loss
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
        self, dataset_name: str, cost: Any, seed: int,
        client_id_1: Union[str, int], client_id_2: Union[str, int],
        num_clients: int, num_classes: int,
        loader_type: str = 'val',
        force_regenerate: bool = False,
        model_type: str = 'round0'
    ) -> Optional[Dict[str, Dict[str, Optional[torch.Tensor]]]]:
        """
        Gets processed activations for a specific client pair.
        
        Returns:
            Dictionary of processed client data or None if failed
        """
        self._initialize_results_manager(dataset_name)
        
        cid1_str = str(client_id_1)
        cid2_str = str(client_id_2)
        
        # Check cache first
        cache_path = self._get_activation_cache_path(
            dataset_name, cost, seed, 
            cid1_str, cid2_str, loader_type, num_clients
        )
        
        raw_activations = None
        if not force_regenerate:
            raw_activations = self._load_activations_from_cache(cache_path)
            
        if raw_activations is None:
            pass#print(f"  Cache {'miss' if not force_regenerate else 'bypass'} for {os.path.basename(cache_path)}")
            
            # Generate activations by loading model
            raw_activations = self._generate_activations(
                dataset=dataset_name, cost=cost, seed=seed,
                client_id_1=cid1_str, client_id_2=cid2_str,
                num_clients=num_clients, num_classes=num_classes,
                loader_type=loader_type, model_type=model_type
            )
            
            if raw_activations is not None:
                self._save_activations_to_cache(raw_activations, cache_path)
            else:
                return None
                
        # Unpack and process the raw activations
        h1_raw, p1_raw, y1_raw, h2_raw, p2_raw, y2_raw = raw_activations
        pass#print(f"  Processing activations for clients ({cid1_str}, {cid2_str})")
        
        processed_data1 = self._process_client_data(h1_raw, p1_raw, y1_raw, cid1_str, num_classes)
        processed_data2 = self._process_client_data(h2_raw, p2_raw, y2_raw, cid2_str, num_classes)
        if processed_data1 is None or processed_data2 is None:
            return None
            
        return {cid1_str: processed_data1, cid2_str: processed_data2}

    def get_performance(
        self, dataset_name: str, cost: Any, aggregation_method: str = 'mean'
    ) -> Tuple[float, float]:
        """
        Loads final performance metrics from TrialRecords.
        
        Args:
            dataset_name: Name of the dataset
            cost: Cost parameter
            aggregation_method: How to aggregate across runs
            
        Returns:
            Tuple of (local_score, fedavg_score)
        """
        self._initialize_results_manager(dataset_name)
        
        # Load results via ResultsManager
        all_records, _ = self.results_manager.load_results(ExperimentType.EVALUATION)
        
        if not all_records:
            pass#print(f"No performance results found for {dataset_name}")
            return np.nan, np.nan
        
        # Filter records by cost
        cost_records = [r for r in all_records if r.cost == cost]
        
        if not cost_records:
            pass#print(f"Cost {cost} not found in results for {dataset_name}")
            return np.nan, np.nan
        
        # Extract final losses for local and fedavg
        local_losses = []
        fedavg_losses = []
        
        for record in cost_records:
            if record.server_type == 'local' and not record.error:
                test_losses = record.metrics.get(MetricKey.TEST_LOSSES, [])
                if test_losses:
                    local_losses.append(test_losses[-1])  # Last test loss is final performance
                    
            elif record.server_type == 'fedavg' and not record.error:
                test_losses = record.metrics.get(MetricKey.TEST_LOSSES, [])
                if test_losses:
                    fedavg_losses.append(test_losses[-1])
        
        # Aggregate scores
        local_score = np.nan
        fedavg_score = np.nan
        
        if local_losses:
            if aggregation_method.lower() == 'median':
                local_score = float(np.median(local_losses))
            else:  # Default to mean
                local_score = float(np.mean(local_losses))
                
        if fedavg_losses:
            if aggregation_method.lower() == 'median':
                fedavg_score = float(np.median(fedavg_losses))
            else:  # Default to mean
                fedavg_score = float(np.mean(fedavg_losses))
                
        pass#print(f"Performance for {dataset_name}, cost {cost}: Local={local_score:.4f}, FedAvg={fedavg_score:.4f}")
        return local_score, fedavg_score