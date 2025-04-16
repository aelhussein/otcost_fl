# data_manager.py
import torch
import numpy as np
import os
import pickle
import warnings
from typing import Dict, Any, Optional, Tuple, Union

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
    """ Handles loading, caching, generation, and processing of activations and performance results. """
    def __init__(self, activation_dir: str = ACTIVATION_DIR, results_dir: str = RESULTS_DIR, loss_eps: float = DEFAULT_EPS):
        self.activation_dir = activation_dir
        self.results_dir = results_dir
        self.loss_eps = loss_eps
        os.makedirs(self.activation_dir, exist_ok=True)
        # Ensure results base dir exists if needed for loading
        # os.makedirs(self.results_dir, exist_ok=True)

    def _get_activation_cache_path(
        self, dataset_name: str, cost: float, rounds: int, seed: int,
        client_id_1: Union[str, int], client_id_2: Union[str, int], loader_type: str
    ) -> str:
        """Constructs the standardized path for activation cache files."""
        c1_str = str(client_id_1); c2_str = str(client_id_2)
        dataset_cache_dir = os.path.join(self.activation_dir, dataset_name)
        os.makedirs(dataset_cache_dir, exist_ok=True)
        filename = f"activations_{dataset_name}_cost{cost:.4f}_r{rounds}_seed{seed}_c{c1_str}v{c2_str}_{loader_type}.pt"
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
        self, dataset: str, cost: float, rounds: int, seed: int,
        client_id_1: Union[str, int], client_id_2: Union[str, int], loader_type: str
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
                 client_id_1=client_id_1,
                 client_id_2=client_id_2,
                 loader_type=loader_type,
                 rounds=rounds,
                 base_seed=seed
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
        self, dataset_name: str, cost: float, rounds: int, seed: int,
        client_id_1: Union[str, int], client_id_2: Union[str, int],
        num_classes: int, loader_type: str = 'val',
        force_regenerate: bool = False
    ) -> Optional[Dict[str, Dict[str, Optional[torch.Tensor]]]]:
        """
        Gets processed activations for two clients, using cache if possible.

        Args:
            dataset_name, cost, rounds, seed, client_id_1, client_id_2, loader_type: Config params.
            num_classes: Number of classes in the dataset.
            force_regenerate: If True, bypass cache and generate new activations.

        Returns:
            A dictionary {'client_id_1': processed_data_dict_1, 'client_id_2': processed_data_dict_2}
            or None if loading/generation/processing fails.
            processed_data_dict contains keys like 'h', 'p_prob', 'y', 'loss', 'weights'.
        """
        cid1_str = str(client_id_1)
        cid2_str = str(client_id_2)
        cache_path = self._get_activation_cache_path(dataset_name, cost, rounds, seed, cid1_str, cid2_str, loader_type)

        raw_activations = None
        if not force_regenerate:
            raw_activations = self._load_activations_from_cache(cache_path)

        if raw_activations is None:
            raw_activations = self._generate_activations(dataset_name, cost, rounds, seed, cid1_str, cid2_str, loader_type)
            if raw_activations is not None:
                self._save_activations_to_cache(raw_activations, cache_path)
            else:
                warnings.warn(f"Failed to obtain raw activations for config: D={dataset_name}, C={cost}, R={rounds}, S={seed}")
                return None

        # Unpack raw activations
        h1_raw, p1_raw, y1_raw, h2_raw, p2_raw, y2_raw = raw_activations

        # Process data for each client
        processed_data1 = self._process_client_data(h1_raw, p1_raw, y1_raw, cid1_str, num_classes)
        processed_data2 = self._process_client_data(h2_raw, p2_raw, y2_raw, cid2_str, num_classes)

        if processed_data1 is None or processed_data2 is None:
            warnings.warn("Failed to process raw activations for one or both clients.")
            return None

        return {cid1_str: processed_data1, cid2_str: processed_data2}


    def get_performance(
        self, dataset_name: str, cost: float, aggregation_method: str = 'mean'
        ) -> Tuple[float, float]:
        """Loads final performance (e.g., loss) from standard pickle file path."""
        final_local_score = np.nan
        final_fedavg_score = np.nan

        # Construct path using convention
        pickle_path = os.path.join(self.results_dir, "evaluation", f"{dataset_name}_evaluation.pkl")

        if not os.path.isfile(pickle_path):
            warnings.warn(f"Performance results pickle file not found: {pickle_path}")
            return final_local_score, final_fedavg_score

        try:
            with open(pickle_path, 'rb') as f:
                results_dict = pickle.load(f)

            if cost not in results_dict:
                warnings.warn(f"Cost {cost} not found in results dictionary: {pickle_path}")
                return final_local_score, final_fedavg_score

            cost_data = results_dict[cost]

            # --- Process Scores (simplified parsing from original) ---
            def _extract_agg_score(scores_list):
                # Expecting list of lists like [[score1], [score2], ...]
                scores = [item[0] for item in scores_list if isinstance(item, list) and len(item)>0 and np.isfinite(item[0])]
                if not scores: return np.nan
                if aggregation_method.lower() == 'mean': return np.mean(scores)
                elif aggregation_method.lower() == 'median': return np.median(scores)
                else: warnings.warn(f"Invalid agg method '{aggregation_method}'. Using mean."); return np.mean(scores)

            # Process 'local' scores
            try:
                final_local_score = _extract_agg_score(cost_data.get('local', {}).get('global', {}).get('losses', []))
                if np.isnan(final_local_score): warnings.warn(f"No valid 'local' scores found for cost {cost} in {pickle_path}")
            except Exception as e: warnings.warn(f"Error parsing 'local' scores for cost {cost} in {pickle_path}: {e}")

            # Process 'fedavg' scores
            try:
                final_fedavg_score = _extract_agg_score(cost_data.get('fedavg', {}).get('global', {}).get('losses', []))
                if np.isnan(final_fedavg_score): warnings.warn(f"No valid 'fedavg' scores found for cost {cost} in {pickle_path}")
            except Exception as e: warnings.warn(f"Error parsing 'fedavg' scores for cost {cost} in {pickle_path}: {e}")

        except FileNotFoundError: warnings.warn(f"Performance results pickle file not found: {pickle_path}")
        except (pickle.UnpicklingError, EOFError) as e: warnings.warn(f"Error unpickling performance results file {pickle_path}: {e}")
        except Exception as e: warnings.warn(f"An unexpected error occurred loading performance results from {pickle_path}: {e}")

        return final_local_score, final_fedavg_score