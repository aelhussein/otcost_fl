# data_manager.py
import torch
import numpy as np
import os
import pickle
import warnings
from typing import Dict, Any, Optional, Tuple, Union
from configs import *
from ot_utils import calculate_sample_loss, DEFAULT_EPS
# Import ResultsManager to load performance data correctly
from pipeline import ResultsManager, ExperimentType # Need ResultsManager for loading


# Assume activation extraction function is available (e.g., from ot_utils or model_utils)
# from ot_utils import get_acts_for_similarity # Or wherever it lives
# Placeholder for activation function if not defined elsewhere
def get_acts_for_similarity(*args, **kwargs) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    warnings.warn("Using placeholder get_acts_for_similarity. Replace with actual implementation.")
    # Dummy implementation returning None
    return None, None, None

# Assume experiment running function is available (replace with actual imports)
# from pipeline import Experiment, ExperimentConfig, ExperimentType, get_default_lr # etc.
# Placeholder for experiment function
def run_experiment_get_activations(*args, **kwargs) -> Tuple[Any, Any, Tuple, Any, Any]:
     warnings.warn("Using placeholder run_experiment_get_activations. Replace with actual implementation.")
     # Dummy implementation returning Nones/empty structures
     return None, None, (None, None, None, None, None, None), None, None

# Assume global paths are defined (e.g., in a config file)
ACTIVATION_DIR = "./cached_activations" # Example path
RESULTS_DIR = "./results" # Example path

# Import necessary helper functions for processing
from ot_utils import calculate_sample_loss, DEFAULT_EPS

class DataManager:
    """
    Handles loading, caching, generation, and processing of activations
    and performance results for specific experiment configurations (including num_clients).
    """
    # MODIFIED: Init accepts num_clients used for performance file loading
    def __init__(self,
                 num_clients: int,
                 activation_dir: str = ACTIVATION_DIR,
                 results_dir: str = RESULTS_DIR, # Base results dir
                 loss_eps: float = DEFAULT_EPS):
        """
        Initializes the DataManager.

        Args:
            num_clients (int): The target number of clients used in the FL run (used for loading performance files).
            activation_dir (str): Directory for storing/loading activation caches.
            results_dir (str): Base directory where experiment results are stored.
            loss_eps (float): Epsilon for numerical stability in loss calculations.
        """
        if not isinstance(num_clients, int) or num_clients <= 0:
             raise ValueError("DataManager requires a valid positive integer for num_clients.")
        self.num_clients = num_clients # Store the target client count for result file lookup
        self.activation_dir = activation_dir
        self.results_dir = results_dir
        self.loss_eps = loss_eps
        os.makedirs(self.activation_dir, exist_ok=True)
        print(f"DataManager initialized targeting results for {self.num_clients} clients.")

    # MODIFIED: Added num_clients parameter for consistency, path includes it
    def _get_activation_cache_path(
        self, dataset_name: str, cost: Any, rounds: int, seed: int,
        client_id_1: Union[str, int], client_id_2: Union[str, int],
        loader_type: str, num_clients: int
    ) -> str:
        """Constructs the standardized path for activation cache files, including num_clients."""
        c1_str = str(client_id_1); c2_str = str(client_id_2)
        dataset_cache_dir = os.path.join(self.activation_dir, dataset_name)
        os.makedirs(dataset_cache_dir, exist_ok=True)
        cost_str = f"{float(cost):.4f}" if isinstance(cost, (int, float)) else str(cost)
        # Filename includes num_clients associated with the run
        filename = f"activations_{dataset_name}_nc{num_clients}_cost{cost_str}_r{rounds}_seed{seed}_c{c1_str}v{c2_str}_{loader_type}.pt"
        return os.path.join(dataset_cache_dir, filename)

    def _load_activations_from_cache(self, path: str) -> Optional[Tuple]:
        """Loads activations from a PyTorch file."""
        if not os.path.isfile(path):
            return None
        try:
            data = torch.load(path)
            # Add validation checks here (e.g., type, length, content)
            if isinstance(data, tuple) and len(data) == 6:
                 # Basic check: ensure at least h1 and h2 are present
                 if data[0] is not None and data[3] is not None:
                    print(f"  Successfully loaded activations from cache: {path}")
                    return data
                 else:
                    warnings.warn(f"Cached data format invalid (missing h1/h2): {path}")
                    return None
            else:
                 warnings.warn(f"Cached data format invalid (not tuple of 6): {path}")
                 return None
        except Exception as e:
            warnings.warn(f"Failed loading cache {path}: {e}")
            return None

    def _save_activations_to_cache(self, data: Tuple, path: str) -> None:
        """Saves activations to a PyTorch file."""
        try:
            # Ensure parent directory exists
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(data, path)
            print(f"  Saved activations to cache: {path}")
        except Exception as e:
            warnings.warn(f"Failed saving activations to cache {path}: {e}")

    def _generate_activations(
        self, dataset: str, cost: Any, rounds: int, seed: int,
        client_id_1: Union[str, int], client_id_2: Union[str, int],
        num_clients: int, loader_type: str # Pass num_clients
    ) -> Optional[Tuple]:
        """
        Triggers the experiment run to generate activations.
        NOTE: This requires the experiment running infrastructure to be available.
        """
        print(f"  Generating activations for Cost {cost}, Round {rounds}, Seed {seed}...")
        try:
            # This function needs the full experiment setup (configs, models, etc.)
            # It might need to be adapted based on how Experiment/ExperimentConfig work
            _, _, activations_tuple, _, _ = run_experiment_get_activations(
                 dataset=dataset,
                 cost=cost,
                 num_clients=num_clients,
                 loader_type=loader_type,
                 rounds=rounds,
                 base_seed=seed,
                 activation_config={
                     'save': True,
                     'client_ids': [str(client_id_1), str(client_id_2)],
                     'loader_type': loader_type,
                     'round': rounds
                 }
            )
            # Basic validation of generated activations
            if isinstance(activations_tuple, tuple) and len(activations_tuple) == 6 and \
               activations_tuple[0] is not None and activations_tuple[3] is not None:
                return activations_tuple
            else:
                warnings.warn("Activation generation failed or returned invalid format.")
                return None
        except Exception as e:
            warnings.warn(f"Error during activation generation: {e}", stacklevel=2)
            return None


    def _process_client_data(
        self, h: Optional[torch.Tensor], p_prob_in: Optional[torch.Tensor], y: Optional[torch.Tensor],
        client_id: str, num_classes: int
        ) -> Optional[Dict[str, Optional[torch.Tensor]]]:
        """
        Processes raw data for one client: validates probs, calculates loss & weights.
        Returns None if essential data (h, y) is missing or processing fails.
        """
        if h is None or y is None:
            warnings.warn(f"Client {client_id}: Missing essential 'h' or 'y' data for processing.")
            return None

        # Convert to CPU tensors
        try:
            h_cpu = h.detach().cpu() if isinstance(h, torch.Tensor) else torch.tensor(h).cpu()
            y_cpu = y.detach().cpu().long() if isinstance(y, torch.Tensor) else torch.tensor(y).long().cpu()
            p_prob_cpu = p_prob_in.detach().cpu() if isinstance(p_prob_in, torch.Tensor) else torch.tensor(p_prob_in).cpu() if p_prob_in is not None else None
        except Exception as e:
             warnings.warn(f"Client {client_id}: Error converting data to CPU tensors: {e}")
             return None

        n_samples = y_cpu.shape[0]
        if n_samples == 0:
             warnings.warn(f"Client {client_id}: Zero samples provided.")
             # Return structure expected by calculators, but with empty tensors? Or None? Let's return None.
             return None

        # --- Validate Probabilities ---
        p_prob_validated = None
        if p_prob_cpu is not None:
            with torch.no_grad():
                p_prob_float = p_prob_cpu.float()
                # Simplified validation/normalization (adapt from original if complex cases needed)
                if p_prob_float.ndim == 2 and p_prob_float.shape[0] == n_samples and p_prob_float.shape[1] == num_classes:
                    if torch.all(p_prob_float >= 0) and torch.all(p_prob_float <= 1) and \
                       torch.allclose(p_prob_float.sum(dim=1), torch.ones(n_samples), atol=1e-3):
                        p_prob_validated = p_prob_float
                    else:
                        warnings.warn(f"Client {client_id}: Probs invalid/don't sum to 1. Renormalizing.")
                        p_prob_validated = p_prob_float / p_prob_float.sum(dim=1, keepdim=True).clamp(min=self.loss_eps)
                        p_prob_validated = torch.clamp(p_prob_validated, 0.0, 1.0)
                elif num_classes == 2 and (p_prob_float.ndim == 1 or p_prob_float.shape[1] == 1) and p_prob_float.shape[0] == n_samples:
                     # Handle binary [N] or [N, 1] format -> [N, 2]
                     p1 = torch.clamp(p_prob_float.view(-1), 0.0, 1.0); p0 = 1.0 - p1
                     p_prob_validated = torch.stack([p0, p1], dim=1)
                else:
                    warnings.warn(f"Client {client_id}: Unexpected prob shape {p_prob_cpu.shape} for N={n_samples}, K={num_classes}. Cannot validate.")
        else:
            warnings.warn(f"Client {client_id}: No probabilities provided ('p_prob_in' is None).")


        # --- Calculate Loss and Weights ---
        loss = None
        weights = None
        if p_prob_validated is not None:
             loss = calculate_sample_loss(p_prob_validated, y_cpu, num_classes, self.loss_eps)

        if loss is None or not torch.isfinite(loss).all():
            if loss is not None: warnings.warn(f"Client {client_id}: NaN/Inf loss detected. Using uniform weights.")
            weights = torch.ones_like(y_cpu, dtype=torch.float) / n_samples
            loss = torch.full_like(y_cpu, float('nan'), dtype=torch.float) # Mark loss as NaN
            # Set p_prob_validated to None if loss failed? Maybe keep it if validation passed.
        elif loss.sum().item() <= self.loss_eps:
             # Handle zero loss case (perfect predictions?) -> uniform weights
             warnings.warn(f"Client {client_id}: Total loss is zero or less. Using uniform weights.")
             weights = torch.ones_like(y_cpu, dtype=torch.float) / n_samples
        else:
             # Use loss for weighting
             weights = loss / loss.sum()

        return {'h': h_cpu, 'p_prob': p_prob_validated, 'y': y_cpu, 'loss': loss, 'weights': weights}


    def get_activations(
        self, dataset_name: str, cost: Any, rounds: int, seed: int,
        client_id_1: Union[str, int], client_id_2: Union[str, int], # The specific pair
        num_clients: int, # The total # clients in the run (for path/generation)
        num_classes: int,
        loader_type: str = 'val',
        force_regenerate: bool = False
    ) -> Optional[Dict[str, Dict[str, Optional[torch.Tensor]]]]:
        """
        Gets processed activations for a specific client pair from a run involving num_clients.
        """
        cid1_str = str(client_id_1)
        cid2_str = str(client_id_2)
        # Cache path now uses num_clients
        cache_path = self._get_activation_cache_path(dataset_name, cost, rounds, seed, cid1_str, cid2_str, loader_type, num_clients)

        raw_activations = None
        if not force_regenerate:
            raw_activations = self._load_activations_from_cache(cache_path)
            # if raw_activations: print(f"  Loaded activations from cache: {os.path.basename(cache_path)}") # Optional verbose log

        if raw_activations is None:
            print(f"  Activation cache miss or regen forced: {os.path.basename(cache_path)}")
            # Generate activations for the specific pair within the context of num_clients run
            raw_activations = self._generate_activations(dataset_name, cost, rounds, seed, cid1_str, cid2_str, num_clients, loader_type)
            if raw_activations is not None:
                self._save_activations_to_cache(raw_activations, cache_path)
            else:
                warnings.warn(f"Failed to obtain raw activations for config: D={dataset_name}, NC={num_clients}, C={cost}, R={rounds}, S={seed}, Pair=({cid1_str},{cid2_str})")
                return None

        # ... (Unpack and process activations as before) ...
        h1_raw, p1_raw, y1_raw, h2_raw, p2_raw, y2_raw = raw_activations
        print(f"  Processing activations for pair ({cid1_str}, {cid2_str})...")
        processed_data1 = self._process_client_data(h1_raw, p1_raw, y1_raw, cid1_str, num_classes)
        processed_data2 = self._process_client_data(h2_raw, p2_raw, y2_raw, cid2_str, num_classes)

        if processed_data1 is None or processed_data2 is None:
             warnings.warn(f"Failed to process raw activations for one or both clients in pair ({cid1_str}, {cid2_str}).")
             return None

        return {cid1_str: processed_data1, cid2_str: processed_data2}


    def get_performance(
        self, dataset_name: str, cost: Any, aggregation_method: str = 'mean'
        ) -> Tuple[float, float]:
        """Loads final performance using ResultsManager for the num_clients specified during DataManager init."""
        final_local_score = np.nan
        final_fedavg_score = np.nan

        try:
            # Instantiate a ResultsManager for the specific dataset and target client count
            results_manager = ResultsManager(
                root_dir=ROOT_DIR,
                dataset=dataset_name,
                experiment_type=ExperimentType.EVALUATION,
                num_clients=self.num_clients # Use the count stored during DataManager init
            )
            results, metadata = results_manager.load_results(ExperimentType.EVALUATION)
            if results is None:
                warnings.warn(f"No performance results found via ResultsManager for {dataset_name}, {self.num_clients} clients, evaluation.")
                return np.nan, np.nan

            # Optional: Verify client count consistency
            if metadata and metadata.get('client_count_used') is not None:
                 actual_clients_in_file = metadata['client_count_used']
                 # Allow for slight mismatch if filename uses target but metadata uses actual (e.g., from ISIC 'all')
                 # if actual_clients_in_file != self.num_clients:
                 #      print(f"Info: Loaded performance file client count ({actual_clients_in_file}) "
                 #            f"differs slightly from target ({self.num_clients}). File: {results_manager._get_results_path(ExperimentType.EVALUATION)}")

            if cost not in results:
                warnings.warn(f"Cost {cost} not found in loaded results dictionary for {dataset_name}.")
                return np.nan, np.nan

            cost_data = results[cost]

            # --- Process Scores ---
            def _extract_agg_score(scores_list):
                # Expecting list of lists like [[score1], [score2], ...]
                scores = [item[0] for item in scores_list if isinstance(item, list) and len(item)>0 and np.isfinite(item[0])]
                if not scores: return np.nan
                if aggregation_method.lower() == 'mean': return np.mean(scores)
                elif aggregation_method.lower() == 'median': return np.median(scores)
                else: warnings.warn(f"Invalid agg method '{aggregation_method}'. Using mean."); return np.mean(scores)


            final_local_score = _extract_agg_score(cost_data.get('local', {}).get('global', {}).get('losses', []))
            final_fedavg_score = _extract_agg_score(cost_data.get('fedavg', {}).get('global', {}).get('losses', []))
            # Add warnings if NaN as before...

        except Exception as e:
            warnings.warn(f"An unexpected error occurred loading performance results: {e}")
            traceback.print_exc()
        return final_local_score, final_fedavg_score