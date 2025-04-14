import sys
# Add paths if necessary - assuming they are correct from your snippet
sys.path.append('/gpfs/commons/groups/gursoy_lab/aelhussein/classes/otcost_fl/code')
sys.path.append('/gpfs/commons/groups/gursoy_lab/aelhussein/classes/otcost_fl/code/evaluation')

import importlib
import configs # Assuming these exist and are needed by other parts
import helper
import pipeline
import clients
import models
import servers
import data_processing
importlib.reload(configs)
importlib.reload(helper)
importlib.reload(pipeline)
importlib.reload(clients)
importlib.reload(models)
importlib.reload(servers)
importlib.reload(data_processing)
from configs import *
from helper import *
from pipeline import *
from clients import *
from models import *
from servers import *
from data_processing import *

from sklearn.metrics.pairwise import cosine_distances # Keep for decomposed part if needed
import ot
import numpy as np
import torch
from scipy.stats import wasserstein_distance # Keep for decomposed part
import warnings
import os
import random
from sklearn.decomposition import PCA 
import matplotlib.pyplot as plt 
import matplotlib.lines as mlines 
import seaborn as sns 
from typing import List, Tuple, Dict, Any, Optional, Union 

# --- Seeding Functions (Unchanged) ---
def set_global_seeds(seed_value=1):
    """Sets random seeds for reproducibility."""
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # if you are using multi-GPU.
    np.random.seed(seed_value)
    random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    # The following are potentially performance-impacting, use carefully
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Note: torch.use_deterministic_algorithms is available in newer PyTorch versions
    if hasattr(torch, "use_deterministic_algorithms"):
        try:
            # Available modes: True, False, 'warn'
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception as e:
            print(f"Warning: Failed to set deterministic algorithms: {e}")

def create_seeded_generator(seed_value=1):
    """Creates a seeded PyTorch generator."""
    g = torch.Generator()
    g.manual_seed(seed_value)
    return g


# ==============================================================================
# Similarity Analyzer Class (Simplified for Consistent [N, K] Softmax Output)
# ==============================================================================


class SimilarityAnalyzer:
    """
    Calculates and stores similarity metrics between activations from two clients.
    Assumes model outputs are probabilities shape [N, num_classes] (K>=2).

    Includes:
    - Feature-Error Additive OT: Uses additive cost (D_h_scaled + D_e_scaled).
    - Decomposed OT: Computes Label EMD and class-conditional OT costs.
                     *MODIFIED*: Within-class cost now also uses D_h_scaled + D_e_scaled.

    Loss-weighting uses Cross-Entropy loss. Normalization and epsilon values
    are handled internally or via parameters.
    """
    def __init__(self, client_id_1, client_id_2, num_classes,
                 feature_error_ot_params=None,
                 decomposed_params=None,
                 loss_eps=1e-9):
        """
        Initializes the analyzer.

        Args:
            client_id_1 (str): Identifier for the first client.
            client_id_2 (str): Identifier for the second client.
            num_classes (int): Number of classes (K >= 2).
            feature_error_ot_params (dict, optional): Parameters for Feature-Error Additive OT.
                (See previous definitions - alpha, beta, reg, use_loss_weighting etc.)
            decomposed_params (dict, optional): Parameters for Decomposed OT.
                'alpha_within' (float): Weight for feature distance (D_h_scaled) within class. Default 1.0.
                'beta_within' (float): Weight for error distance (D_e_scaled) within class. Default 1.0. # Changed default
                'ot_eps' (float): Sinkhorn regularization for within-class OT. Default 0.001.
                'ot_max_iter' (int): Sinkhorn max iterations. Default 2000.
                'eps' (float): Epsilon for numerical stability (e.g., normalization). Default 1e-10.
                'normalize_cost': (bool): Normalize within-class cost matrix to [0, 1]. Default True.
                'normalize_emd': (bool): Normalize Label EMD score. Default True.
                'min_class_samples': (int): Minimum samples per class before upsampling. Default 0.
                'aggregate_conditional_method': (str): Method to aggregate class OT costs. Default 'total_loss_share'.
                'verbose': (bool): Enable verbose warnings/prints. Default False.
            loss_eps (float): Small epsilon added for loss calculation stability.
        """
        if num_classes < 2:
             raise ValueError("num_classes must be >= 2.")

        self.client_id_1 = str(client_id_1)
        self.client_id_2 = str(client_id_2)
        self.num_classes = num_classes
        self.loss_eps = loss_eps

        # --- Default Parameters ---
        default_feature_error_ot_params = {
            'alpha': 1.0, 'beta': 1.0, 'reg': 0.001, 'max_iter': 2000,
            'norm_eps': 1e-10, 'normalize_cost': True,
            'use_loss_weighting': True, 'verbose': False
        }
        # <<< MODIFIED: Default Decomposed Params >>>
        default_decomposed_params = {
            'alpha_within': 1.0, # Weight for D_h_scaled within class
            'beta_within': 1.0,  # Weight for D_e_scaled within class (DEFAULT CHANGED)
            'ot_eps': 0.001,     # Sinkhorn reg for within-class
            'ot_max_iter': 2000,
            'eps': 1e-10,        # Epsilon for normalization/stability
            'normalize_cost': True, # Normalize within-class cost matrix
            'normalize_emd': True,
            'min_class_samples': 0,
            'verbose': False,
            # 'feature_dist_func', 'pred_dist_func', 'use_confidence_weighting' are no longer used here
            'aggregate_conditional_method': 'total_loss_share'
        }

        self.feature_error_ot_params = default_feature_error_ot_params.copy()
        if feature_error_ot_params:
            self.feature_error_ot_params.update(feature_error_ot_params)

        self.decomposed_params = default_decomposed_params.copy()
        if decomposed_params:
             # Remove potentially obsolete keys if user passes them
            obsolete_keys = ['feature_dist_func', 'pred_dist_func', 'use_confidence_weighting']
            for key in obsolete_keys:
                if key in decomposed_params:
                    warnings.warn(f"Parameter '{key}' is no longer used for decomposed OT within-class cost and will be ignored.")
                    # Optionally remove: del decomposed_params[key]
            self.decomposed_params.update(decomposed_params)

        valid_agg_methods = ['mean', 'avg_loss', 'total_loss_share']
        if self.decomposed_params['aggregate_conditional_method'] not in valid_agg_methods:
            warnings.warn(f"Invalid aggregate_method '{self.decomposed_params['aggregate_conditional_method']}'. Falling back to 'mean'.")
            self.decomposed_params['aggregate_conditional_method'] = 'mean'

        self.activations = {self.client_id_1: None, self.client_id_2: None}
        self._reset_results()

    def _calculate_sample_loss(self, p_prob, y):
        # (Implementation remains the same)
        if p_prob is None or y is None: return None
        p_prob = p_prob.float().clamp(min=self.loss_eps, max=1.0 - self.loss_eps)
        y = y.long()
        if y.shape[0] != p_prob.shape[0] or p_prob.ndim != 2 or p_prob.shape[1] != self.num_classes:
             warnings.warn(f"Loss calculation shape mismatch/invalid: P({p_prob.shape}), Y({y.shape}), K={self.num_classes}")
             return None
        try:
            true_class_prob = p_prob.gather(1, y.view(-1, 1)).squeeze()
            loss = -torch.log(true_class_prob.clamp(min=self.loss_eps))
        except Exception as e:
             warnings.warn(f"Error during loss calculation (gather/log): {e}. Shapes: P={p_prob.shape}, Y={y.shape}")
             return None
        # Return non-negative losses
        return torch.relu(loss)


    def load_activations(self, h1, p1_prob_in, y1, h2, p2_prob_in, y2):
        # (Implementation remains the same - calculates loss and stores in 'weights')
        if any(x is None for x in [h1, p1_prob_in, y1, h2, p2_prob_in, y2]):
            warnings.warn("load_activations received None for one or more inputs."); self._reset_results(); return False
        def _process_client_data(h, p_prob_in, y, client_id, num_classes, loss_eps):
            # Ensure data is on CPU and converted to tensors
            h_cpu = h.detach().cpu() if isinstance(h, torch.Tensor) else torch.tensor(h).cpu()
            p_prob_in_cpu = p_prob_in.detach().cpu() if isinstance(p_prob_in, torch.Tensor) else torch.tensor(p_prob_in).cpu()
            y_cpu = y.detach().cpu().long() if isinstance(y, torch.Tensor) else torch.tensor(y).long().cpu()

            p_prob_validated = None
            with torch.no_grad():
                p_prob_float = p_prob_in_cpu.float()
                # --- Binary Classification Handling ---
                if num_classes == 2:
                    if p_prob_float.ndim == 1 or (p_prob_float.ndim == 2 and p_prob_float.shape[1] == 1):
                        # Input is likely probability of class 1, shape [N] or [N, 1]
                        p1 = torch.clamp(p_prob_float.view(-1), 0.0, 1.0)
                        p0 = 1.0 - p1
                        p_prob_validated = torch.stack([p0, p1], dim=1) # Convert to [N, 2]
                    elif p_prob_float.ndim == 2 and p_prob_float.shape[1] == 2:
                        # Input is already [N, 2]
                        if torch.all(p_prob_float >= 0) & torch.all(p_prob_float <= 1) & torch.allclose(p_prob_float.sum(dim=1), torch.ones_like(p_prob_float[:, 0]), atol=1e-3):
                            p_prob_validated = p_prob_float # Looks valid
                        else:
                            # Invalid [N, 2] probs, attempt renormalization
                            warnings.warn(f"Client {client_id}: Binary [N, 2] probs invalid/don't sum to 1. Renormalizing.")
                            p_prob_validated = p_prob_float / p_prob_float.sum(dim=1, keepdim=True).clamp(min=loss_eps)
                            p_prob_validated = torch.clamp(p_prob_validated, 0.0, 1.0)
                    else:
                        warnings.warn(f"Client {client_id}: Unexpected binary prob shape {p_prob_in_cpu.shape}.")
                # --- Multi-class Classification Handling ---
                elif p_prob_float.ndim == 2 and p_prob_float.shape[1] == num_classes:
                    # Input is [N, K]
                    if torch.all(p_prob_float >= 0) & torch.all(p_prob_float <= 1) & torch.allclose(p_prob_float.sum(dim=1), torch.ones_like(p_prob_float[:, 0]), atol=1e-3):
                        p_prob_validated = p_prob_float # Looks valid
                    else:
                        # Invalid multi-class probs, attempt renormalization
                        warnings.warn(f"Client {client_id}: Multi-class probs don't sum to 1. Renormalizing.")
                        p_prob_validated = p_prob_float / p_prob_float.sum(dim=1, keepdim=True).clamp(min=loss_eps)
                        p_prob_validated = torch.clamp(p_prob_validated, 0.0, 1.0)
                else:
                     warnings.warn(f"Client {client_id}: Unexpected multi-class prob shape {p_prob_in_cpu.shape}.")

            # Calculate loss using validated probabilities
            if p_prob_validated is None:
                loss = None
                warnings.warn(f"Client {client_id}: Could not validate probabilities.")
            else:
                loss = self._calculate_sample_loss(p_prob_validated, y_cpu)

            # Determine weights based on loss
            if loss is None or not torch.isfinite(loss).all():
                # Fallback to uniform if loss calculation failed or resulted in NaN/Inf
                if loss is not None: warnings.warn(f"Client {client_id}: NaN/Inf loss detected. Using uniform weights.")
                num_samples = max(1, len(y_cpu))
                weights = torch.ones_like(y_cpu, dtype=torch.float) / num_samples
                loss = torch.full_like(y_cpu, float('nan'), dtype=torch.float) # Store NaN loss
                p_prob_validated = None # Mark probs as unusable if loss failed
            elif loss.sum().item() <= loss_eps:
                # Handle case where all losses are effectively zero (perfect fit)
                num_samples = max(1, len(y_cpu))
                weights = torch.ones_like(y_cpu, dtype=torch.float) / num_samples
            else:
                # Standard loss weighting
                weights = loss / loss.sum()

            return {
                'h': h_cpu,
                'p_prob': p_prob_validated, # Use the validated probabilities
                'y': y_cpu,
                'loss': loss,
                'weights': weights # These are the loss-based weights (or uniform fallback)
            }
        # Process data for both clients
        self.activations[self.client_id_1] = _process_client_data(h1, p1_prob_in, y1, self.client_id_1, self.num_classes, self.loss_eps)
        self.activations[self.client_id_2] = _process_client_data(h2, p2_prob_in, y2, self.client_id_2, self.num_classes, self.loss_eps)

        # Check if processing was successful
        if self.activations[self.client_id_1] is None or self.activations[self.client_id_2] is None:
            warnings.warn("Failed to process client data."); self._reset_results(); return False
        if self.activations[self.client_id_1].get('p_prob') is None or self.activations[self.client_id_2].get('p_prob') is None:
            warnings.warn("'p_prob' is None after processing. Subsequent calculations needing probabilities might fail.")
            # Allow continuation, but note potential issues

        self._reset_results() # Reset previous calculation results
        return True


    def _reset_results(self):
        """Resets stored results and cost matrices."""
        self.cost_matrices = {'feature_error_ot': None, 'within_class': {}}
        self.results = {
            'feature_error_ot': {
                'ot_cost': np.nan,
                'transport_plan': None,
                'cost_matrix_max_val': np.nan,
                'weighting_used': None # <<< ADDED to store weighting type
            },
            'decomposed': {'label_emd': np.nan, 'conditional_ot': np.nan, 'combined_score': np.nan},
            'within_class_ot': {}
        }

    def calculate_all(self):
        """Calculates all configured similarity metrics."""
        if self.activations[self.client_id_1] is None or self.activations[self.client_id_2] is None:
            print("Activations not loaded."); return

        print(f"\nCalculating Feature-Error Additive OT ({self.client_id_1} vs {self.client_id_2})...")
        self.calculate_feature_error_ot_similarity() # <<< Call renamed method

        # Optionally run decomposed calculation
        # Consider if decomposed should also have weighting options if used frequently
        print(f"\nCalculating Decomposed similarity ({self.client_id_1} vs {self.client_id_2})...")
        self.calculate_decomposed_similarity()


    def calculate_feature_error_ot_similarity(self): # <<< Method updated
        """
        Calculates OT cost using the additive feature-error cost.
        Uses marginal weights based on 'use_loss_weighting' parameter.
        """
        if self.activations[self.client_id_1] is None or self.activations[self.client_id_2] is None:
            print("Activations not loaded for Feature-Error OT."); return

        act1 = self.activations[self.client_id_1]; act2 = self.activations[self.client_id_2]
        params = self.feature_error_ot_params
        verbose = params.get('verbose', False)
        use_loss_weighting = params.get('use_loss_weighting', True) # Get the flag

        # --- Check necessary data availability ---
        # Required: h, p_prob, y. Also need 'weights' if use_loss_weighting is True.
        required_keys = ['h', 'p_prob', 'y']
        if use_loss_weighting:
            required_keys.append('weights')

        if any(act1.get(k) is None for k in required_keys) or \
           any(act2.get(k) is None for k in required_keys):
             missing_str = "'weights'" if use_loss_weighting and (act1.get('weights') is None or act2.get('weights') is None) else "h, p_prob, or y"
             warnings.warn(f"Feature-Error OT calculation requires {missing_str}. One or more unavailable. Skipping.")
             self.results['feature_error_ot'] = {'ot_cost': np.nan, 'transport_plan': None, 'cost_matrix_max_val': np.nan, 'weighting_used': 'Loss' if use_loss_weighting else 'Uniform'}
             return

        # --- Calculate Cost Matrix (remains the same) ---
        cost_matrix, max_cost = self._calculate_cost_feature_error_additive(
            act1['h'], act1['p_prob'], act1['y'],
            act2['h'], act2['p_prob'], act2['y'],
            **params
        )
        self.results['feature_error_ot']['cost_matrix_max_val'] = max_cost

        # --- Proceed if Cost Matrix is valid ---
        # Use norm_eps as a small threshold to check if max_cost is non-negligible
        if cost_matrix is not None and (not np.isfinite(max_cost) or max_cost > params.get('norm_eps', 1e-10)):
            normalize_cost = params.get('normalize_cost', True)
            if normalize_cost and np.isfinite(max_cost) and max_cost > 0:
                normalized_cost_matrix = cost_matrix / max_cost
            else:
                # Handle cases where max_cost is infinite, zero, or normalization is off
                if normalize_cost and not np.isfinite(max_cost):
                     warnings.warn("Max cost is not finite, cannot normalize cost matrix.")
                normalized_cost_matrix = cost_matrix # Use unnormalized

            self.cost_matrices['feature_error_ot'] = normalized_cost_matrix

            # --- Define Marginal Weights (a, b) based on the flag ---
            N = act1['h'].shape[0] # Number of samples for client 1
            M = act2['h'].shape[0] # Number of samples for client 2

            if N == 0 or M == 0:
                 warnings.warn("One or both clients have zero samples. OT cost is 0.")
                 self.results['feature_error_ot']['ot_cost'] = 0.0
                 self.results['feature_error_ot']['transport_plan'] = np.zeros((N, M)) if N>0 and M>0 else None
                 self.results['feature_error_ot']['weighting_used'] = 'Loss' if use_loss_weighting else 'Uniform'
                 return # Exit early

            weighting_type_str = "" # For print statement
            if use_loss_weighting:
                weights1 = act1['weights'].cpu().numpy().astype(np.float64)
                weights2 = act2['weights'].cpu().numpy().astype(np.float64)
                weighting_type_str = "Loss-Weighted"

                # Safety renormalization (important!)
                sum1 = weights1.sum(); sum2 = weights2.sum()
                if not np.isclose(sum1, 1.0) and sum1 > 1e-9: weights1 /= sum1
                elif sum1 <= 1e-9: weights1 = np.ones_like(weights1) / N; warnings.warn("Client 1 loss weights sum to zero or less, using uniform.")
                if not np.isclose(sum2, 1.0) and sum2 > 1e-9: weights2 /= sum2
                elif sum2 <= 1e-9: weights2 = np.ones_like(weights2) / M; warnings.warn("Client 2 loss weights sum to zero or less, using uniform.")
            else:
                # Use uniform weights
                weights1 = np.ones(N, dtype=np.float64) / N
                weights2 = np.ones(M, dtype=np.float64) / M
                weighting_type_str = "Uniform"

            # Store which weighting was used
            self.results['feature_error_ot']['weighting_used'] = weighting_type_str

            # --- Compute OT Cost ---
            ot_cost, Gs = self._compute_ot_cost(
                normalized_cost_matrix, a=weights1, b=weights2,
                eps=params.get('reg', 0.001), # Use updated default if needed
                sinkhorn_max_iter=params.get('max_iter', 2000)
            )

            self.results['feature_error_ot']['ot_cost'] = ot_cost
            self.results['feature_error_ot']['transport_plan'] = Gs
            print(f"  Feature-Error OT Cost ({weighting_type_str} Marginals, NormalizedCost={normalize_cost}): {f'{ot_cost:.4f}' if not np.isnan(ot_cost) else 'Failed'}")

        else:
            fail_reason = "Cost matrix calculation failed" if cost_matrix is None else "Max cost near zero/invalid"
            warnings.warn(f"Feature-Error OT calculation skipped: {fail_reason}.")
            # Store NaN but indicate intended weighting
            self.results['feature_error_ot']['ot_cost'] = np.nan
            self.results['feature_error_ot']['transport_plan'] = None
            self.results['feature_error_ot']['weighting_used'] = 'Loss' if use_loss_weighting else 'Uniform'
            # cost_matrix_max_val is already set


    def _calculate_cost_feature_error_additive(self, h1, p1_prob, y1, h2, p2_prob, y2, **params): # <<< Renamed helper
        """
        Calculates the additive cost: C = alpha * D_h_scaled + beta * D_e_scaled.
        D_h_scaled: Euclidean distance between L2-normalized features (range [0, 1]).
        D_e_scaled: Euclidean distance between raw error vectors, scaled (range [0, 1]).
        """
        alpha = params.get('alpha', 1.0)
        beta = params.get('beta', 1.0)
        norm_eps = params.get('norm_eps', 1e-10)
        device = 'cpu' # Calculations forced to CPU for consistency

        # --- Input Validation ---
        if any(x is None for x in [h1, p1_prob, y1, h2, p2_prob, y2]):
            warnings.warn("Missing input (h, p_prob, or y) for cost calculation.")
            return None, np.nan

        # --- Ensure Tensors on CPU ---
        try:
            h1, p1_prob, y1 = map(lambda x: x.to(device) if isinstance(x, torch.Tensor) else torch.tensor(x, device=device), [h1, p1_prob, y1])
            h2, p2_prob, y2 = map(lambda x: x.to(device) if isinstance(x, torch.Tensor) else torch.tensor(x, device=device), [h2, p2_prob, y2])
            y1, y2 = y1.long(), y2.long()
        except Exception as e:
            warnings.warn(f"Error converting inputs to tensors on CPU: {e}")
            return None, np.nan


        N, M = h1.shape[0], h2.shape[0]
        if N == 0 or M == 0:
            # Return empty tensor and 0 cost if either set is empty
            return torch.empty((N, M), device=device, dtype=torch.float), 0.0
        # Validate probability shapes after ensuring tensor
        if p1_prob.shape != (N, self.num_classes) or p2_prob.shape != (M, self.num_classes):
             warnings.warn(f"Probability shape mismatch: P1({p1_prob.shape}), P2({p2_prob.shape}), K={self.num_classes}")
             return None, np.nan

        # --- 1. Scaled Feature Distance Term ---
        term1 = torch.zeros((N, M), device=device, dtype=torch.float)
        max_term1_contrib = 0.0
        if alpha > self.loss_eps: # Use loss_eps as threshold for float comparison
            try:
                h1_norm = F.normalize(h1.float(), p=2, dim=1, eps=norm_eps)
                h2_norm = F.normalize(h2.float(), p=2, dim=1, eps=norm_eps)
                if h1_norm is None or h2_norm is None: raise ValueError("Normalization failed")

                # Use cdist for potentially better numerical stability / clarity
                # Equivalent to sqrt(2 - 2*cos_sim) when inputs are L2 normalized
                D_h = torch.cdist(h1_norm, h2_norm, p=2) # Shape [N, M], Range [0, 2]

                # Scale to [0, 1]
                D_h_scaled = D_h / 2.0
                term1 = alpha * D_h_scaled
                max_term1_contrib = alpha * 1.0 # Max possible contribution is alpha
            except Exception as e:
                warnings.warn(f"Feature distance calculation failed: {e}")
                return None, np.nan

        # --- 2. Scaled Raw Error Distance Term ---
        term2 = torch.zeros((N, M), device=device, dtype=torch.float)
        max_term2_contrib = 0.0
        if beta > self.loss_eps: # Use loss_eps as threshold
            try:
                # Use validated probs directly
                P1 = p1_prob.float(); P2 = p2_prob.float()

                # One-hot encode labels
                y1_oh = F.one_hot(y1.view(-1), num_classes=self.num_classes).float().to(device)
                y2_oh = F.one_hot(y2.view(-1), num_classes=self.num_classes).float().to(device)

                # Raw error vectors
                e1 = P1 - y1_oh
                e2 = P2 - y2_oh

                # Pairwise Euclidean distance between error vectors
                D_e_raw = torch.cdist(e1.float(), e2.float(), p=2) # Range [0, sqrt(2)*2] approx

                # Scale to [0, 1] by dividing by theoretical max distance
                max_raw_error_dist = 2.0 * np.sqrt(2.0) # Max L2 distance between two vectors in [-1, 1]^K space diff (p-y_oh)
                D_e_scaled = D_e_raw / max_raw_error_dist
                term2 = beta * D_e_scaled
                max_term2_contrib = beta * 1.0 # Max possible contribution is beta

            except Exception as e:
                warnings.warn(f"Error distance calculation failed: {e}")
                return None, np.nan

        # --- 3. Combine Terms Additively ---
        cost_matrix = term1 + term2
        # Theoretical max cost if inputs could reach extremes
        max_possible_cost = max_term1_contrib + max_term2_contrib

        # Clamp cost matrix just in case of float issues going slightly over max
        # Use max_possible_cost if > 0, otherwise clamp to 1.0 as a safe upper bound
        effective_max_cost = max(self.loss_eps, max_possible_cost) # Avoid clamping to 0
        cost_matrix = torch.clamp(cost_matrix, min=0.0, max=effective_max_cost)

        # Return cost matrix (as tensor) and the calculated max possible cost (as float)
        return cost_matrix, float(max_possible_cost)


    # =========================================================================
    # Decomposed OT and Helper Methods (Keep as is for optional use)
    # =========================================================================
    def calculate_decomposed_similarity(self):
        warnings.warn("Running Decomposed OT with unified within-class cost.", stacklevel=2)
        if self.activations[self.client_id_1] is None or self.activations[self.client_id_2] is None:
            print("Activations not loaded for Decomposed OT.")
            return

        act1 = self.activations[self.client_id_1]; act2 = self.activations[self.client_id_2]
        # Ensure all base data exists
        if any(act1.get(k) is None for k in ['h', 'y', 'weights']) or \
           any(act2.get(k) is None for k in ['h', 'y', 'weights']):
             warnings.warn("Decomposed OT requires h, y, and weights. Skipping.")
             self._reset_decomposed_results(keep_emd=False)
             return

        h1, y1, w1 = act1['h'], act1['y'], act1['weights']
        h2, y2, w2 = act2['h'], act2['y'], act2['weights']
        # Get probabilities and losses if available, check validity
        p1_prob, loss1 = act1.get('p_prob'), act1.get('loss')
        p2_prob, loss2 = act2.get('p_prob'), act2.get('loss')
        loss1_valid = loss1 is not None and torch.isfinite(loss1).all()
        loss2_valid = loss2 is not None and torch.isfinite(loss2).all()
        can_use_loss_agg = loss1_valid and loss2_valid

        N, M = h1.shape[0], h2.shape[0]
        if N == 0 or M == 0:
            self._reset_decomposed_results()
            return

        params = self.decomposed_params
        verbose = params.get('verbose', False); eps_num = params.get('eps', 1e-10)
        ot_eps = params.get('ot_eps', 0.001); ot_max_iter = params.get('ot_max_iter', 2000)
        normalize_within_cost = params.get('normalize_cost', True)
        normalize_emd_flag = params.get('normalize_emd', True)
        min_class_samples = params.get('min_class_samples', 0)
        agg_method_param = params.get('aggregate_conditional_method', 'mean')
        beta_within = params.get('beta_within', 0.0)
        # Check if probabilities are needed for the within-class cost (if beta > 0)
        p_needed_within = beta_within > eps_num

        # --- 1. Label EMD ---
        label_emd_raw = self._calculate_label_emd(y1, y2, self.num_classes)
        if np.isnan(label_emd_raw):
            warnings.warn("Label EMD calculation failed. Decomposed OT aborted.")
            self._reset_decomposed_results(keep_emd=False); return

        emd_norm_factor = max(1.0, float(self.num_classes - 1)) if self.num_classes > 1 else 1.0
        label_emd_normalized = label_emd_raw / emd_norm_factor if normalize_emd_flag else label_emd_raw
        self.results['decomposed']['label_emd'] = label_emd_normalized

        # --- 2. Class-Conditional OT ---
        all_class_results = {}
        self.cost_matrices['within_class'] = {}; self.results['within_class_ot'] = {}

        total_loss_across_classes = (loss1.sum() + loss2.sum()).item() if can_use_loss_agg else 0.0

        for c in range(self.num_classes):
            current_class_result = {'ot_cost': np.nan, 'avg_loss': 0.0, 'total_loss': 0.0, 'sample_count': (0, 0)}
            self.results['within_class_ot'][c] = {'ot_cost': np.nan, 'cost_matrix_max_val': np.nan, 'upsampled': (False, False)}

            idx1_c = torch.where(y1.long() == c)[0]; idx2_c = torch.where(y2.long() == c)[0]
            n_c, m_c = idx1_c.shape[0], idx2_c.shape[0]
            current_class_result['sample_count'] = (n_c, m_c)

            if n_c == 0 or m_c == 0:
                all_class_results[c] = current_class_result; continue

            # Calculate class losses if possible
            total_loss_c = 0.0; avg_loss_c = 0.0
            if can_use_loss_agg:
                loss1_c = loss1[idx1_c]; loss2_c = loss2[idx2_c]
                if torch.isfinite(loss1_c).all() and torch.isfinite(loss2_c).all():
                    total_loss_c = (loss1_c.sum() + loss2_c.sum()).item()
                    avg_loss_c = total_loss_c / (n_c + m_c) if (n_c + m_c) > 0 else 0.0
                    current_class_result['total_loss'] = total_loss_c
                    current_class_result['avg_loss'] = avg_loss_c

            # Get data for this class: h, w, y are always needed
            h1_c_eff = h1[idx1_c]; w1_c_eff = w1[idx1_c]; y1_c_eff = y1[idx1_c]
            h2_c_eff = h2[idx2_c]; w2_c_eff = w2[idx2_c]; y2_c_eff = y2[idx2_c]
            # Get probabilities only if needed (beta_within > 0) and available
            p1_prob_c_eff = p1_prob[idx1_c] if p_needed_within and p1_prob is not None else None
            p2_prob_c_eff = p2_prob[idx2_c] if p_needed_within and p2_prob is not None else None

            # Check if needed probabilities are available for error calculation
            if p_needed_within and (p1_prob_c_eff is None or p2_prob_c_eff is None):
                if verbose: warnings.warn(f"Class {c}: Skipped OT - probabilities needed (beta_within > 0) but unavailable/invalid.")
                all_class_results[c] = current_class_result; continue

            # --- Handle Upsampling ---
            upsampled1 = False; upsampled2 = False
            # Initialize _final variables
            h1_c_final, w1_c_final, y1_c_final = h1_c_eff, w1_c_eff, y1_c_eff
            h2_c_final, w2_c_final, y2_c_final = h2_c_eff, w2_c_eff, y2_c_eff
            p1_prob_c_final, p2_prob_c_final = p1_prob_c_eff, p2_prob_c_eff
            n_c_final, m_c_final = n_c, m_c

            if min_class_samples > 0:
                if n_c < min_class_samples:
                    if n_c > 0:
                        needed1 = min_class_samples - n_c
                        resample_indices1 = np.random.choice(n_c, size=needed1, replace=True)
                        h1_c_final = torch.cat((h1_c_eff, h1_c_eff[resample_indices1]), dim=0)
                        w1_c_final = torch.cat((w1_c_eff, w1_c_eff[resample_indices1]), dim=0)
                        y1_c_final = torch.cat((y1_c_eff, y1_c_eff[resample_indices1]), dim=0) # Upsample Y
                        if p_needed_within and p1_prob_c_eff is not None:
                            p1_prob_c_final = torch.cat((p1_prob_c_eff, p1_prob_c_eff[resample_indices1]), dim=0)
                        n_c_final = min_class_samples; upsampled1 = True
                    else:
                        if verbose: warnings.warn(f"Class {c}: C1 has 0 samples, cannot upsample.")
                        all_class_results[c] = current_class_result; continue
                if m_c < min_class_samples:
                    if m_c > 0:
                        needed2 = min_class_samples - m_c
                        resample_indices2 = np.random.choice(m_c, size=needed2, replace=True)
                        h2_c_final = torch.cat((h2_c_eff, h2_c_eff[resample_indices2]), dim=0)
                        w2_c_final = torch.cat((w2_c_eff, w2_c_eff[resample_indices2]), dim=0)
                        y2_c_final = torch.cat((y2_c_eff, y2_c_eff[resample_indices2]), dim=0) # Upsample Y
                        if p_needed_within and p2_prob_c_eff is not None:
                            p2_prob_c_final = torch.cat((p2_prob_c_eff, p2_prob_c_eff[resample_indices2]), dim=0)
                        m_c_final = min_class_samples; upsampled2 = True
                    else:
                         if verbose: warnings.warn(f"Class {c}: C2 has 0 samples, cannot upsample.")
                         all_class_results[c] = current_class_result; continue

            self.results['within_class_ot'][c]['upsampled'] = (upsampled1, upsampled2)

            # --- Calculate Within-Class Cost Matrix (using updated function) ---
            cost_matrix_c, max_cost_c = self._calculate_cost_within_class(
                h1_c_final, p1_prob_c_final, y1_c_final, # Pass Y
                h2_c_final, p2_prob_c_final, y2_c_final, # Pass Y
                **params # Pass decomposed_params
            )
            self.results['within_class_ot'][c]['cost_matrix_max_val'] = max_cost_c

            if cost_matrix_c is not None and (not np.isfinite(max_cost_c) or max_cost_c > eps_num):
                if normalize_within_cost and np.isfinite(max_cost_c) and max_cost_c > eps_num:
                    normalized_cost_matrix_c = cost_matrix_c / max_cost_c
                else:
                    if normalize_within_cost and not np.isfinite(max_cost_c): warnings.warn(f"Class {c}: Max cost infinite.")
                    normalized_cost_matrix_c = cost_matrix_c

                self.cost_matrices['within_class'][c] = normalized_cost_matrix_c

                # Prepare Marginals (use loss-based weights 'w', re-normalized for the class)
                w1_c_np = w1_c_final.cpu().numpy().astype(np.float64)
                w2_c_np = w2_c_final.cpu().numpy().astype(np.float64)
                sum1_c = w1_c_np.sum(); sum2_c = w2_c_np.sum()

                if sum1_c < eps_num: a_c = np.ones(n_c_final)/max(1,n_c_final); warnings.warn(f"C{c}: C1 weights sum zero.")
                else: a_c = w1_c_np / sum1_c
                if sum2_c < eps_num: b_c = np.ones(m_c_final)/max(1,m_c_final); warnings.warn(f"C{c}: C2 weights sum zero.")
                else: b_c = w2_c_np / sum2_c

                # Compute Within-Class OT Cost
                ot_cost_c_norm, _ = self._compute_ot_cost(
                    normalized_cost_matrix_c, a=a_c, b=b_c,
                    eps=ot_eps, sinkhorn_max_iter=ot_max_iter
                )
                current_class_result['ot_cost'] = ot_cost_c_norm
                self.results['within_class_ot'][c]['ot_cost'] = ot_cost_c_norm
                if np.isnan(ot_cost_c_norm) and verbose: warnings.warn(f"OT cost failed class {c}.")
            else:
                fail_reason = "calc failed" if cost_matrix_c is None else "max cost near zero"
                if verbose: warnings.warn(f"Within-class cost matrix {fail_reason} class {c}.")

            all_class_results[c] = current_class_result

        # --- 3. Aggregate Conditional OT Costs ---
        # (Aggregation logic remains the same)
        valid_classes = [c for c, res in all_class_results.items() if not np.isnan(res.get('ot_cost', np.nan))]
        agg_conditional_ot = np.nan; agg_method_used = "None"
        if not valid_classes: warnings.warn("No valid conditional OT costs to aggregate.")
        else:
            costs_agg = np.array([all_class_results[c]['ot_cost'] for c in valid_classes]); effective_agg_method=agg_method_param; agg_method_used=f"'{effective_agg_method}'"
            if effective_agg_method != 'mean' and not can_use_loss_agg: warnings.warn(f"Cannot use agg '{agg_method_param}', fallback mean."); effective_agg_method='mean'; agg_method_used="Mean (Fallback)"
            if effective_agg_method == 'mean': agg_conditional_ot=np.mean(costs_agg); agg_method_used="Mean"
            else:
                if effective_agg_method == 'avg_loss': weights_agg=np.array([all_class_results[c]['avg_loss'] for c in valid_classes]); agg_method_used="Weighted by Avg Loss"
                elif effective_agg_method == 'total_loss_share': weights_agg=np.array([all_class_results[c]['total_loss'] for c in valid_classes]); agg_method_used="Weighted by Total Loss Share"
                else: weights_agg=np.ones_like(costs_agg); agg_method_used="Mean (Fallback)"
                weights_agg=np.maximum(weights_agg,0); weights_agg+=eps_num; total_weight=weights_agg.sum()
                if total_weight < eps_num * len(weights_agg): agg_conditional_ot=np.mean(costs_agg); agg_method_used+=" (Fallback - Zero Weight)"
                else: normalized_weights_agg=weights_agg/total_weight; agg_conditional_ot=np.sum(normalized_weights_agg*costs_agg)
        self.results['decomposed']['conditional_ot'] = agg_conditional_ot

        # --- 4. Calculate Combined Score ---
        if not np.isnan(label_emd_normalized) and not np.isnan(agg_conditional_ot):
            total_score = label_emd_normalized + agg_conditional_ot
            total_score = total_score / 2.0
        else:
            total_score = np.nan
        self.results['decomposed']['combined_score'] = total_score

        # --- Print Decomposed Results Summary ---
        print("-" * 30)
        print("--- Decomposed Similarity Results (Unified Within-Class Cost) ---") # This printed before
        emd_str = f"{self.results['decomposed']['label_emd']:.3f}" if not np.isnan(self.results['decomposed']['label_emd']) else 'Failed'
        print(f"  Label EMD (Normalized={normalize_emd_flag}): {emd_str}") # Should print

        beta_str = f", BetaW={params.get('beta_within', 0.0):.2f}" if params.get('beta_within', 0.0) > eps_num else ""
        min_samp_str = f", Upsampled to {min_class_samples}" if min_class_samples > 0 else ""
        print(f"  Conditional Within-Class OT Costs (NormCost={normalize_within_cost}{beta_str}{min_samp_str}):") # Should print

        for c in range(self.num_classes):
            cost_val = self.results['within_class_ot'].get(c, {}).get('ot_cost', np.nan)
            upsampled_status = self.results['within_class_ot'].get(c, {}).get('upsampled', (False, False))
            avg_loss_val = all_class_results.get(c, {}).get('avg_loss', 0.0)
            total_loss_val = all_class_results.get(c, {}).get('total_loss', 0.0)
            n_c_orig, m_c_orig = all_class_results.get(c, {}).get('sample_count', (0, 0))

            cost_str = f"{cost_val:.3f}" if not np.isnan(cost_val) else 'Skipped/Failed'
            upsample_note = ""
            if upsampled_status != (False, False):
                upsample_note = " (Both Upsampled)" if upsampled_status[0] and upsampled_status[1] \
                     else f" ({self.client_id_1} Upsampled)" if upsampled_status[0] \
                     else f" ({self.client_id_2} Upsampled)" if upsampled_status[1] \
                     else ""
            # <<< This print statement needs to execute for each class >>>
            print(f"     - Class {c}: Cost={cost_str}{upsample_note} (Samples=[{n_c_orig},{m_c_orig}], AvgLoss={avg_loss_val:.3f}, TotalLoss={total_loss_val:.3f})")

        cond_ot_agg_str = f"{self.results['decomposed']['conditional_ot']:.3f}" if not np.isnan(self.results['decomposed']['conditional_ot']) else 'Failed/Not Applicable'
        print(f"  Aggregated Conditional OT ({agg_method_used}): {cond_ot_agg_str}") # Should print
        comb_score_str = f"{self.results['decomposed']['combined_score']:.3f}" if not np.isnan(self.results['decomposed']['combined_score']) else 'Invalid'
        print(f"  Combined Score (Sum): {comb_score_str}") # Should print
        print("-" * 30) # Should print


    def _reset_decomposed_results(self, keep_emd=True):
        # (Implementation unchanged)
        if not keep_emd: self.results['decomposed']['label_emd'] = np.nan
        self.results['decomposed']['conditional_ot']=np.nan; self.results['decomposed']['combined_score']=np.nan
        self.cost_matrices['within_class']={}; self.results['within_class_ot']={}

    # --- Getters (remain the same) ---
    def get_results(self): return self.results
    def get_activations(self): return self.activations
    def get_cost_matrices(self): return self.cost_matrices

    # --- Private Calculation Helpers ---

    def _calculate_confidence(self, P_prob, num_classes, eps=1e-10):
        # (Implementation unchanged)
        if P_prob is None: return None;
        if not isinstance(P_prob, torch.Tensor): return None
        if P_prob.ndim != 2 or P_prob.shape[1] != num_classes: return None
        num_classes_float = float(num_classes) # Use float for log calculation
        if num_classes_float <= 1: return torch.ones(P_prob.shape[0], device=P_prob.device) # Max confidence if only 1 class
        P_clamped = torch.clamp(P_prob.float(), min=eps)
        entropy = -torch.sum(P_clamped * torch.log(P_clamped), dim=1)
        max_entropy = np.log(num_classes_float)
        if max_entropy < eps: return torch.ones_like(entropy) # Avoid division by zero if max_entropy is ~0
        normalized_entropy = entropy / max_entropy
        # Confidence = 1 - normalized entropy
        return torch.clamp(1.0 - normalized_entropy, 0.0, 1.0)


    def _calculate_cost_within_class(self, h1_c, p1_prob_c, y1_c, h2_c, p2_prob_c, y2_c, **params):
        """
        Calculates the within-class cost using the Feature-Error Additive metric:
        C = alpha_within * D_h_scaled + beta_within * D_e_scaled.
        Requires h, p_prob, and y for the samples within the class.
        Uses parameters from `decomposed_params`.
        """
        alpha_within = params.get('alpha_within', 1.0)
        beta_within = params.get('beta_within', 1.0) # Default might be 1.0 now
        # Use 'eps' from decomposed_params for normalization stability
        norm_eps = params.get('eps', 1e-10)
        device = 'cpu' # Force CPU

        # --- Input Validation ---
        # Check features and labels (always needed)
        if h1_c is None or y1_c is None or h2_c is None or y2_c is None:
             warnings.warn("Missing h or y input for within-class cost.")
             return None, np.nan
        # Check probabilities only if beta_within > 0
        prob_needed = beta_within > self.loss_eps
        if prob_needed and (p1_prob_c is None or p2_prob_c is None):
            warnings.warn("Missing p_prob input required for within-class cost (beta_within > 0).")
            return None, np.nan

        N, M = h1_c.shape[0], h2_c.shape[0]
        if N == 0 or M == 0:
            return torch.empty((N, M), device=device, dtype=torch.float), 0.0

        # --- Ensure Tensors on CPU ---
        try:
            h1_c, y1_c = map(lambda x: x.to(device) if isinstance(x, torch.Tensor) else torch.tensor(x, device=device), [h1_c, y1_c])
            h2_c, y2_c = map(lambda x: x.to(device) if isinstance(x, torch.Tensor) else torch.tensor(x, device=device), [h2_c, y2_c])
            y1_c, y2_c = y1_c.long(), y2_c.long() # Ensure labels are Long type

            if prob_needed:
                p1_prob_c = p1_prob_c.to(device).float() if isinstance(p1_prob_c, torch.Tensor) else torch.tensor(p1_prob_c, device=device).float()
                p2_prob_c = p2_prob_c.to(device).float() if isinstance(p2_prob_c, torch.Tensor) else torch.tensor(p2_prob_c, device=device).float()
                # Validate prob shapes
                if p1_prob_c.shape != (N, self.num_classes) or p2_prob_c.shape != (M, self.num_classes):
                     warnings.warn(f"Within-class prob shape mismatch: P1({p1_prob_c.shape}), P2({p2_prob_c.shape})")
                     return None, np.nan
            else:
                 p1_prob_c, p2_prob_c = None, None # Not needed
        except Exception as e:
             warnings.warn(f"Error converting within-class inputs to tensors: {e}")
             return None, np.nan

        # --- 1. Scaled Feature Distance Term (D_h_scaled) ---
        term1 = torch.zeros((N, M), device=device, dtype=torch.float)
        max_term1_contrib = 0.0
        if alpha_within > self.loss_eps:
            try:
                h1_norm = F.normalize(h1_c.float(), p=2, dim=1, eps=norm_eps)
                h2_norm = F.normalize(h2_c.float(), p=2, dim=1, eps=norm_eps)
                D_h = torch.cdist(h1_norm, h2_norm, p=2) # Range [0, 2]
                D_h_scaled = D_h / 2.0 # Scale to [0, 1]
                term1 = alpha_within * D_h_scaled
                max_term1_contrib = alpha_within * 1.0
            except Exception as e:
                warnings.warn(f"Within-class feature distance failed: {e}")
                return None, np.nan

        # --- 2. Scaled Raw Error Distance Term (D_e_scaled) ---
        term2 = torch.zeros((N, M), device=device, dtype=torch.float)
        max_term2_contrib = 0.0
        if prob_needed: # Only calculate if beta > 0 and probs are valid
            try:
                # One-hot encode labels
                y1_oh = F.one_hot(y1_c.view(-1), num_classes=self.num_classes).float()
                y2_oh = F.one_hot(y2_c.view(-1), num_classes=self.num_classes).float()
                # Raw error vectors
                e1 = p1_prob_c - y1_oh
                e2 = p2_prob_c - y2_oh
                # Pairwise Euclidean distance
                D_e_raw = torch.cdist(e1, e2, p=2)
                # Scale to [0, 1]
                max_raw_error_dist = 2.0 * np.sqrt(2.0)
                D_e_scaled = D_e_raw / max_raw_error_dist
                term2 = beta_within * D_e_scaled
                max_term2_contrib = beta_within * 1.0
            except Exception as e:
                warnings.warn(f"Within-class error distance failed: {e}")
                return None, np.nan

        # --- 3. Combine Terms ---
        cost_matrix_c = term1 + term2
        max_possible_cost_c = max_term1_contrib + max_term2_contrib

        # Clamp cost matrix
        effective_max_cost_c = max(self.loss_eps, max_possible_cost_c)
        cost_matrix_c = torch.clamp(cost_matrix_c, min=0.0, max=effective_max_cost_c)

        return cost_matrix_c, float(max_possible_cost_c)

    def _construct_prob_vectors(self, p_prob):
        # (Implementation unchanged - This helper seems less used now)
        if p_prob is None: return None
        if not isinstance(p_prob, torch.Tensor):
            try: p_prob = torch.tensor(p_prob)
            except Exception: return None
        p_prob = p_prob.cpu().float()
        if p_prob.ndim != 2 or p_prob.shape[1] != self.num_classes: return None
        p_prob_clamped = torch.clamp(p_prob, 0.0, 1.0)
        row_sums = p_prob_clamped.sum(dim=1, keepdim=True)
        # Renormalize if rows don't sum to 1
        if not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-3):
             p_prob_clamped = p_prob_clamped / row_sums.clamp(min=self.loss_eps) # Use loss_eps for stability
        return p_prob_clamped


    def _calculate_label_emd(self, y1, y2, num_classes):
        """Calculates Earth Mover's Distance between label distributions."""
        # (Implementation unchanged)
        if y1 is None or y2 is None: return np.nan
        try:
            y1_tensor = y1 if isinstance(y1, torch.Tensor) else torch.tensor(y1)
            y2_tensor = y2 if isinstance(y2, torch.Tensor) else torch.tensor(y2)
            n1, n2 = y1_tensor.numel(), y2_tensor.numel()
        except Exception as e:
             warnings.warn(f"Error processing labels for EMD: {e}")
             return np.nan

        if n1 == 0 and n2 == 0: return 0.0 # EMD is 0 if both are empty
        # Max distance if one is empty and the other isn't (worst case mismatch)
        if n1 == 0 or n2 == 0: return float(max(0.0, float(num_classes - 1)))

        y1_np = y1_tensor.detach().cpu().numpy().astype(int); y2_np = y2_tensor.detach().cpu().numpy().astype(int);
        class_values = np.arange(num_classes) # Values for wasserstein distance

        # Compute histograms, ensuring bins cover all classes 0 to num_classes-1
        hist1, _ = np.histogram(y1_np, bins=np.arange(num_classes + 1), density=False)
        hist2, _ = np.histogram(y2_np, bins=np.arange(num_classes + 1), density=False)

        # Normalize histograms to get probability distributions
        sum1 = hist1.sum(); sum2 = hist2.sum()
        # If sum is zero (e.g., empty after filtering?), technically handled by n1/n2 check, but defensive:
        if sum1 == 0 or sum2 == 0: return float(max(0.0, float(num_classes - 1)))

        hist1_norm = hist1 / sum1; hist2_norm = hist2 / sum2;

        try:
            # Use scipy's wasserstein_distance for 1D EMD
            return float(wasserstein_distance(class_values, class_values, u_weights=hist1_norm, v_weights=hist2_norm))
        except ValueError as e:
             # This might happen if weights don't sum to 1 due to float issues, though unlikely after norm
             warnings.warn(f"Wasserstein distance calculation failed: {e}")
             return np.nan


    def _compute_ot_cost(self, cost_matrix_torch, a=None, b=None, eps=1e-3, sinkhorn_thresh=1e-3, sinkhorn_max_iter=2000):
        """Computes OT cost using Sinkhorn algorithm from POT library."""
        # (Implementation largely unchanged, minor cleanup)
        if cost_matrix_torch is None:
            warnings.warn("OT computation skipped: Cost matrix is None.")
            return np.nan, None

        # Ensure cost matrix is a tensor
        if not isinstance(cost_matrix_torch, torch.Tensor):
            try: cost_matrix_torch = torch.tensor(cost_matrix_torch)
            except Exception as e:
                warnings.warn(f"Could not convert cost matrix to tensor: {e}")
                return np.nan, None

        if cost_matrix_torch.numel() == 0:
             # If cost matrix is empty (e.g., 0xM or Nx0), OT cost is 0
             N, M = cost_matrix_torch.shape
             return 0.0, np.zeros((N,M)) # Return 0 cost and appropriate empty plan shape

        # Convert to numpy float64 for POT library
        cost_matrix_np = cost_matrix_torch.detach().cpu().numpy().astype(np.float64)
        N, M = cost_matrix_np.shape

        # If N or M is 0 (checked earlier, but defensive)
        if N == 0 or M == 0: return 0.0, np.zeros((N, M))

        # --- Handle non-finite values in cost matrix ---
        if not np.all(np.isfinite(cost_matrix_np)):
            # Find max finite value to use as a basis for replacement
            max_finite_cost = np.nanmax(cost_matrix_np[np.isfinite(cost_matrix_np)])
            # Set replacement large, but related to existing costs if possible
            replacement_val = 1e6 # Default large value
            if np.isfinite(max_finite_cost):
                replacement_val = max(1.0, abs(max_finite_cost)) * 10.0
            cost_matrix_np[~np.isfinite(cost_matrix_np)] = replacement_val
            warnings.warn(f"NaN/Inf detected in cost matrix. Replaced non-finite values with {replacement_val:.2e}.")

        # --- Prepare Marginals a and b ---
        # Default to uniform if not provided
        if a is None: a = np.ones((N,), dtype=np.float64) / N
        else:
            a = np.asarray(a, dtype=np.float64) # Ensure numpy array
            if not np.all(np.isfinite(a)): a = np.ones_like(a) / max(1, len(a)); warnings.warn("NaN/Inf in marginal 'a'. Using uniform.")
            sum_a = a.sum()
            if sum_a <= 1e-9: a = np.ones_like(a) / max(1, len(a)); warnings.warn("Marginal 'a' sums to zero or less. Using uniform.")
            elif not np.isclose(sum_a, 1.0): a /= sum_a # Renormalize if needed

        if b is None: b = np.ones((M,), dtype=np.float64) / M
        else:
            b = np.asarray(b, dtype=np.float64) # Ensure numpy array
            if not np.all(np.isfinite(b)): b = np.ones_like(b) / max(1, len(b)); warnings.warn("NaN/Inf in marginal 'b'. Using uniform.")
            sum_b = b.sum()
            if sum_b <= 1e-9: b = np.ones_like(b) / max(1, len(b)); warnings.warn("Marginal 'b' sums to zero or less. Using uniform.")
            elif not np.isclose(sum_b, 1.0): b /= sum_b # Renormalize if needed

        # --- Compute OT using POT ---
        try:
            # Ensure contiguous array for POT
            cost_matrix_np_cont = np.ascontiguousarray(cost_matrix_np)

            # Try stabilized Sinkhorn first, fall back to standard if it fails/returns NaN
            Gs = ot.sinkhorn(a, b, cost_matrix_np_cont, reg=eps, stopThr=sinkhorn_thresh,
                             numItermax=sinkhorn_max_iter, method='sinkhorn_stabilized',
                             warn=False, verbose=False) # Suppress POT warnings/prints

            if Gs is None or np.any(np.isnan(Gs)):
                if Gs is None: warnings.warn("Stabilized Sinkhorn failed. Trying standard Sinkhorn.")
                else: warnings.warn("Stabilized Sinkhorn resulted in NaN plan. Trying standard Sinkhorn.")
                Gs = ot.sinkhorn(a, b, cost_matrix_np_cont, reg=eps, stopThr=sinkhorn_thresh,
                                 numItermax=sinkhorn_max_iter, method='sinkhorn',
                                 warn=False, verbose=False) # Suppress POT warnings/prints

            # Check final result
            if Gs is None or np.any(np.isnan(Gs)):
                 warnings.warn("Sinkhorn computation failed (both stabilized and standard) or resulted in NaN plan.")
                 return np.nan, None # Return NaN cost and None plan

            # Calculate OT cost from transport plan and cost matrix
            ot_cost = np.sum(Gs * cost_matrix_np)

            if not np.isfinite(ot_cost):
                 warnings.warn(f"Calculated OT cost is not finite ({ot_cost}). Returning NaN.")
                 return np.nan, Gs # Return NaN cost, but keep the plan if available

            return float(ot_cost), Gs # Return cost and transport plan

        except Exception as e:
            warnings.warn(f"Error during OT computation (ot.sinkhorn): {e}")
            return np.nan, None # Return NaN on error


    # Keep pairwise distance functions if needed by decomposed OT
    def _pairwise_js_divergence(self, p, q, eps=1e-10):
        """Calculates pairwise Jensen-Shannon Divergence. Assumes p=[N,K], q=[M,K]."""
        # (Implementation unchanged)
        if p is None or q is None: return None;
        if p.ndim != 2 or q.ndim != 2 or p.shape[1] != q.shape[1] or p.shape[1] != self.num_classes: return None
        # Clamp probabilities & ensure float type on correct device
        p_f = p.float().to(p.device).clamp(min=eps); q_f = q.float().to(q.device).clamp(min=eps)
        # Compute average distribution m = 0.5 * (p + q) using broadcasting
        m = 0.5 * (p_f.unsqueeze(1) + q_f.unsqueeze(0)) # m shape [N, M, K]
        m = m.clamp(min=eps) # Clamp average distribution too

        # Compute KL divergences using broadcasting
        log_m = torch.log(m)
        log_p = torch.log(p_f.unsqueeze(1)) # shape [N, 1, K] -> [N, M, K]
        log_q = torch.log(q_f.unsqueeze(0)) # shape [1, M, K] -> [N, M, K]

        # KL(P || M) = sum_k p_k * (log p_k - log m_k)
        kl_p_m = torch.sum(p_f.unsqueeze(1) * (log_p - log_m), dim=2) # Sum over K, result shape [N, M]
        # KL(Q || M) = sum_k q_k * (log q_k - log m_k)
        kl_q_m = torch.sum(q_f.unsqueeze(0) * (log_q - log_m), dim=2) # Sum over K, result shape [N, M]

        # JSD = 0.5 * (KL(P || M) + KL(Q || M))
        # Use relu to ensure non-negativity (KL can be slightly negative due to float errors)
        jsd = 0.5 * (torch.relu(kl_p_m) + torch.relu(kl_q_m))

        # Clamp result to valid JSD range [0, log(2)]
        return torch.clamp(jsd, min=0.0, max=np.log(2.0))


    def _pairwise_hellinger_sq_distance(self, p, q, eps=1e-10):
        """Calculates pairwise Squared Hellinger Distance. Assumes p=[N,K], q=[M,K]."""
        # (Implementation unchanged)
        if p is None or q is None: return None
        if p.ndim != 2 or q.ndim != 2 or p.shape[1] != q.shape[1] or p.shape[1] != self.num_classes: return None
        # Clamp probabilities to be non-negative & ensure float type on correct device
        p_f = p.float().clamp(min=0.0).to(p.device); q_f = q.float().clamp(min=0.0).to(q.device)

        # Take square roots
        sqrt_p = torch.sqrt(p_f.unsqueeze(1)) # shape [N, 1, K] -> [N, M, K]
        sqrt_q = torch.sqrt(q_f.unsqueeze(0)) # shape [1, M, K] -> [N, M, K]

        # Calculate Bhattacharyya coefficient = sum_k sqrt(p_k * q_k) = sum_k sqrt(p_k) * sqrt(q_k)
        # Sum over the class dimension (K)
        bhattacharyya_coefficient = torch.sum(sqrt_p * sqrt_q, dim=2) # Result shape [N, M]

        # Clamp coefficient to [0, 1] due to potential float errors
        bhattacharyya_coefficient = torch.clamp(bhattacharyya_coefficient, 0.0, 1.0)

        # Squared Hellinger distance = 1 - Bhattacharyya coefficient
        hellinger_sq = 1.0 - bhattacharyya_coefficient

        # Clamp result to theoretical range [0, 1]
        return torch.clamp(hellinger_sq, min=0.0, max=1.0)

# ==============================================================================
# Function to be Updated
# ==============================================================================

def run_experiment_get_activations(
    dataset, cost, client_id_1, client_id_2, loader_type='val', rounds=10, base_seed=2
    ):
    """
    Runs Local and FedAvg experiments from scratch using BEST TUNED learning rates,
    collects metrics from THESE runs, and extracts activations from the FedAvg model.

    Args:
        dataset (str): Name of the dataset configuration.
        cost: Cost parameter for experiment setup.
        client_id_1: Identifier for the first client for activation extraction.
        client_id_2: Identifier for the second client for activation extraction.
        loader_type (str): Which dataloader to get FedAvg activations from ('train', 'val', 'test').
        rounds (int): Number of training rounds for both local and FedAvg runs.
        base_seed (int): Seed to use for this specific run (data init, training).

    Returns:
        dict | None: metrics_local - Performance metrics calculated from the local run in this function.
        dict | None: metrics_fedavg - Performance metrics calculated from the FedAvg run in this function.
        tuple: (h1, p1_data, y1, h2, p2_data, y2) activations from FedAvg model (e.g., logits),
               or Nones if errors occur.
        dict: client_dataloaders - Dictionary of dataloaders used for this activation run.
        Server | None : server_fedavg instance after training in this function.
    """
    # Use 'evaluation' type just for config, but we won't load eval results here
    exp_config = ExperimentConfig(dataset=dataset, experiment_type=ExperimentType.EVALUATION)
    exp_handler = Experiment(exp_config)
    results_manager = exp_handler.results_manager # Use the manager to get LRs

    # --- Initialize Data ONCE for this run ---
    set_global_seeds(base_seed)
    client_dataloaders = None
    client_dataloaders = exp_handler._initialize_experiment(cost)


    metrics_local = None
    metrics_fedavg = None
    server_local = None
    server_fedavg = None

    # --- Local Run ---
    print('\n' + '='*20 + ' STARTING LOCAL RUN ' + '='*20)
    server_type_local = 'local'
    # Get best LR for local
    lr_local = results_manager.get_best_parameters(
        ExperimentType.LEARNING_RATE, server_type_local, cost
    )
    if lr_local is None:
        default_lr_local = get_default_lr(dataset)
        warnings.warn(f"Could not find best LR for {server_type_local}. Using default: {default_lr_local}")
        lr_local = default_lr_local

    set_global_seeds(base_seed) # Reset seed for model init and training process
    try:
        trainer_config_local = exp_handler._create_trainer_config(
            server_type=server_type_local,
            learning_rate=lr_local,
        )
        trainer_config_local.rounds = rounds # Use rounds specified for this function
        server_local = exp_handler._create_server_instance(cost, server_type_local, trainer_config_local, tuning=False)
        exp_handler._add_clients_to_server(server_local, client_dataloaders)

        # Train and evaluate *this* local server instance
        metrics_local = exp_handler._train_and_evaluate(server_local, rounds)
        print(f"Local run finished. Final Global Loss: {metrics_local['global']['losses'][-1]:.4f}, Score: {metrics_local['global']['scores'][-1]:.4f}")

    except Exception as e:
        print(f"Error during Local run: {e}")
        metrics_local = None # Ensure metrics are None if run failed
    finally:
        # Optional cleanup if needed, server_local is not returned anyway
        if server_local: del server_local
        cleanup_gpu()


    # --- FedAvg Run ---
    print('\n' + '='*20 + ' STARTING FEDAVG RUN ' + '='*20)
    server_type_fedavg = 'fedavg'
    # Get best LR for FedAvg
    lr_fedavg = results_manager.get_best_parameters(
        ExperimentType.LEARNING_RATE, server_type_fedavg, cost
    )
    if lr_fedavg is None:
        default_lr_fedavg = get_default_lr(dataset)
        warnings.warn(f"Could not find best LR for {server_type_fedavg}. Using default: {default_lr_fedavg}")
        lr_fedavg = default_lr_fedavg

    set_global_seeds(base_seed) # Reset seed AGAIN for FedAvg model init and training
    try:
        trainer_config_fedavg = exp_handler._create_trainer_config(
            server_type=server_type_fedavg,
            learning_rate=lr_fedavg,
        )
        trainer_config_fedavg.rounds = rounds # Use rounds specified for this function
        server_fedavg = exp_handler._create_server_instance(cost, server_type_fedavg, trainer_config_fedavg, tuning=False)
        exp_handler._add_clients_to_server(server_fedavg, client_dataloaders)

        # Train and evaluate *this* fedavg server instance
        metrics_fedavg = exp_handler._train_and_evaluate(server_fedavg, rounds)
        print(f"FedAvg run finished. Final Global Loss: {metrics_fedavg['global']['losses'][-1]:.4f}, Score: {metrics_fedavg['global']['scores'][-1]:.4f}")

    except Exception as e:
        print(f"Error during FedAvg run: {e}")
        metrics_fedavg = None # Ensure metrics are None if run failed
        if server_fedavg: del server_fedavg; server_fedavg = None
        cleanup_gpu()


    # --- Activation Extraction from the just-trained FEDAVG Model ---
    h1, p1_data, y1 = None, None, None
    h2, p2_data, y2 = None, None, None

    if server_fedavg is not None: # Proceed only if FedAvg run succeeded
        #print(f"\nExtracting activations using '{loader_type}' loader...")
        try:
            h1, p1_data, y1 = get_acts_for_similarity(
                client_id_1, server_fedavg, client_dataloaders, loader_type=loader_type
            )
            #print(f"Activations extracted for {client_id_1}: h={h1.shape if h1 is not None else None}, p={p1_data.shape if p1_data is not None else None}, y={y1.shape if y1 is not None else None}")
        except Exception as e:
            print(f"Error getting activations for client {client_id_1}: {e}")
        try:
            h2, p2_data, y2 = get_acts_for_similarity(
                client_id_2, server_fedavg, client_dataloaders, loader_type=loader_type
            )
            #print(f"Activations extracted for {client_id_2}: h={h2.shape if h2 is not None else None}, p={p2_data.shape if p2_data is not None else None}, y={y2.shape if y2 is not None else None}")
        except Exception as e:
            print(f"Error getting activations for client {client_id_2}: {e}")
    else:
        print("Skipping activation extraction because FedAvg run failed.")

    #print("\n--- Experiment & Activation Extraction Run Finished ---")
    # Return metrics CALCULATED in this function, NEW activations, dataloaders, trained server instance
    return metrics_local, metrics_fedavg, (h1, p1_data, y1, h2, p2_data, y2), client_dataloaders, server_fedavg

def _get_activation_cache_path( # Uses string path operations
    dataset_name: str,
    cost: float,
    rounds: int,
    seed: int,
    client_id_1: Union[str, int],
    client_id_2: Union[str, int],
    loader_type: str
) -> str: # Returns string path
    """Constructs the standardized path using the global base path (string)."""
    c1_str = str(client_id_1)
    c2_str = str(client_id_2)
    # Use os.path.join for cross-platform compatibility
    dataset_cache_dir = os.path.join(ACTIVATION_DIR, dataset_name)
    # Ensure the directory exists
    os.makedirs(dataset_cache_dir, exist_ok=True)
    filename = f"activations_{dataset_name}_cost{cost:.4f}_r{rounds}_seed{seed}_c{c1_str}v{c2_str}_{loader_type}.pt"
    return os.path.join(dataset_cache_dir, filename)


def load_final_performance( # Uses string path operations
    dataset_name: str,
    cost: float,
    aggregation_method: str = 'mean'
) -> Tuple[float, float]:
    """Loads final performance from standard pickle file path (string)."""
    final_local_score = np.nan
    final_fedavg_score = np.nan

    # Construct path using os.path.join and standard filename
    # Note the correction from "evaulation" to "evaluation"
    pickle_path = os.path.join(RESULTS_DIR, "evaluation", f"{dataset_name}_evaluation.pkl")
    # <<< FIX: Use os.path.isfile for string paths >>>
    if not os.path.isfile(pickle_path):
        warnings.warn(f"Performance results pickle file not found: {pickle_path}")
        return final_local_score, final_fedavg_score

    try:
        with open(pickle_path, 'rb') as f:
            results_dict = pickle.load(f)
        #return results_dict
        if cost not in results_dict:
            warnings.warn(f"Cost {cost} not found in results dictionary: {pickle_path}")
            return final_local_score, final_fedavg_score
        
        cost_data = results_dict[cost]
        # Process 'local' scores
        try:
            local_scores_list = cost_data['local']['global']['losses']
            local_scores = [item[0] for item in local_scores_list if isinstance(item, list) and len(item)>0 and np.isfinite(item[0])]
            if local_scores:
                if aggregation_method.lower() == 'mean':
                    final_local_score = np.mean(local_scores)
                elif aggregation_method.lower() == 'median':
                    final_local_score = np.median(local_scores)
                else:
                    warnings.warn(f"Invalid aggregation method '{aggregation_method}'. Using mean.")
                    final_local_score = np.mean(local_scores)
            else:
                warnings.warn(f"No valid 'local' scores found for cost {cost} in {pickle_path}")
        except (KeyError, TypeError, IndexError) as e:
            warnings.warn(f"Error parsing 'local' scores for cost {cost} in {pickle_path}: {e}")

        # Process 'fedavg' scores
        try:
            fedavg_scores_list = cost_data['fedavg']['global']['losses']
            fedavg_scores = [item[0] for item in fedavg_scores_list if isinstance(item, list) and len(item)>0 and np.isfinite(item[0])]
            if fedavg_scores:
                if aggregation_method.lower() == 'mean':
                    final_fedavg_score = np.mean(fedavg_scores)
                elif aggregation_method.lower() == 'median':
                    final_fedavg_score = np.median(fedavg_scores)
                else:
                    final_fedavg_score = np.mean(fedavg_scores)
            else:
                warnings.warn(f"No valid 'fedavg' scores found for cost {cost} in {pickle_path}")
        except (KeyError, TypeError, IndexError) as e:
            warnings.warn(f"Error parsing 'fedavg' scores for cost {cost} in {pickle_path}: {e}")

    except FileNotFoundError:
        warnings.warn(f"Performance results pickle file not found: {pickle_path}")
    except (pickle.UnpicklingError, EOFError) as e:
         warnings.warn(f"Error unpickling performance results file {pickle_path}: {e}")
    except Exception as e:
        warnings.warn(f"An unexpected error occurred loading performance results from {pickle_path}: {e}")

    return final_local_score, final_fedavg_score


# ==============================================================================
# Main Pipeline Function (Using Streamlined String Paths)
# ==============================================================================

def run_cost_param_sweep_pipeline(
    dataset_name: str,
    costs_list: List[float],
    target_rounds: int,
    feature_error_ot_configs: List[Tuple[str, Dict[str, Any]]],
    client_id_1: Union[str, int],
    client_id_2: Union[str, int],
    num_classes: int,
    activation_loader_type: str = 'val',
    performance_aggregation: str = 'mean',
    decomposed_params: Optional[Dict[str, Any]] = None,
    analyzer_loss_eps: float = 1e-9,
    base_seed: int = 2
) -> pd.DataFrame:
    """
    Runs OT analysis using cached activations (string paths) and loads final performance
    from standard file locations based on dataset_name.
    (Full docstring omitted for brevity)
    """
    results_list = []
    client_id_1_str = str(client_id_1)
    client_id_2_str = str(client_id_2)

    print(f"--- Starting Streamlined Pipeline (String Paths) ---")
    print(f"Dataset: {dataset_name}")
    print(f"Target Activation Round: {target_rounds}")
    print(f"Performance Aggregation: {performance_aggregation}")
    print(f"Clients: {client_id_1_str} vs {client_id_2_str}")
    print(f"Costs: {costs_list}")
    print(f"FE-OT Configs: {[name for name, _ in feature_error_ot_configs]}")
    print(f"Activation Loader: {activation_loader_type}")
    print(f"Using Activation Cache Base: {ACTIVATION_DIR}") # Use global string
    print(f"Using Perf Results Base: {RESULTS_DIR}") # Use global string
    print("-" * 60)

    # Outer loop: Iterate through costs
    for i, cost in enumerate(costs_list):
        print(f"\nProcessing Cost: {cost} (Round: {target_rounds}) ({i+1}/{len(costs_list)})...")
        current_seed = base_seed

        # --- 1. Load Final Performance Metrics ---
        final_local_score, final_fedavg_score = load_final_performance(
            dataset_name, cost, performance_aggregation
        )
        print(f"  Loaded Final Performance (Agg: {performance_aggregation}): Local={final_local_score:.4f}, FedAvg={final_fedavg_score:.4f}")

        # --- 2. Obtain Activations (Cache or Generate) ---
        activations_data = None
        loaded_from_cache = False
        activation_status = "Pending"

        # Construct cache path (returns string)
        cache_path_str = _get_activation_cache_path(
            dataset_name, cost, target_rounds, current_seed,
            client_id_1_str, client_id_2_str, activation_loader_type
        )
        print(f"  Looking for cached activations: {cache_path_str}")

        # Try loading from cache using string path
        # <<< FIX: Use os.path.isfile for string paths >>>
        if os.path.isfile(cache_path_str):
            try:
                activations_data = torch.load(cache_path_str)
                # Validation (same as before)
                if isinstance(activations_data, tuple) and len(activations_data) == 6 and all(isinstance(t, torch.Tensor) or t is None for t in activations_data):
                    if activations_data[0] is not None and activations_data[3] is not None: # Check h1 and h2
                        loaded_from_cache = True
                        activation_status = "Loaded from Cache"
                        print(f"  Successfully loaded activations from cache.")
                    else:
                        warnings.warn(f"Cached file {cache_path_str} missing essential components. Regenerating.")
                        activations_data = None
                else:
                    warnings.warn(f"Cached file {cache_path_str} format error. Regenerating.")
                    activations_data = None
            except Exception as e:
                warnings.warn(f"Failed loading cache {cache_path_str}: {type(e).__name__} - {e}. Regenerating.")
                activations_data = None

        # Generate if not loaded (same logic as before)
        if activations_data is None:
            print(f"  Cache miss. Generating activations (Rounds={target_rounds})...")
            try:
                if 'run_experiment_get_activations' not in globals():
                     raise NameError("Function 'run_experiment_get_activations' is not defined or imported.")

                _, _, activations_tuple, _, _ = run_experiment_get_activations(
                    dataset=dataset_name, cost=cost, client_id_1=client_id_1_str,
                    client_id_2=client_id_2_str, loader_type=activation_loader_type,
                    rounds=target_rounds, base_seed=current_seed
                )

                if activations_tuple is None or not isinstance(activations_tuple, tuple) or len(activations_tuple) != 6:
                     raise ValueError(f"run_experiment returned invalid activation tuple: {type(activations_tuple)}")
                h1, p1, y1, h2, p2, y2 = activations_tuple
                if any(x is None for x in [h1, y1, h2, y2]):
                    raise ValueError("Generated activation components (h1/y1 or h2/y2) are None.")
                activations_data = activations_tuple

                # Try saving to cache using string path
                try:
                    # Ensure directory exists using os.makedirs
                    os.makedirs(os.path.dirname(cache_path_str), exist_ok=True)
                    torch.save(activations_data, cache_path_str)
                    activation_status = "Generated and Cached"
                    print(f"  Generated activations saved to cache: {cache_path_str}")
                except Exception as e_save:
                    warnings.warn(f"Failed to save activations to {cache_path_str}: {type(e_save).__name__} - {e_save}")
                    activation_status = "Generated (Cache Failed)"

            except Exception as e_gen:
                warnings.warn(f"Cost {cost}, R{target_rounds}: FAILED activation generation - {type(e_gen).__name__}: {e_gen}", stacklevel=2)
                activation_status = f"Failed Activation Gen: {type(e_gen).__name__}"
                activations_data = None
        # --- 3. Proceed if Activations Available (same logic as before) ---
        if activations_data is None:
            placeholder_row = {
                'Cost': cost, 'Round': target_rounds, 'Param_Set_Name': 'N/A',
                'Run_Status': activation_status,
                'Local_Final_Loss': final_local_score, 
                'FedAvg_Final_Loss': final_fedavg_score,
                'FeatureErrorOT_Cost': np.nan, 'FeatureErrorOT_Weighting': None,
                'Decomposed_LabelEMD': np.nan, 'Decomposed_ConditionalOT': np.nan,
                'Decomposed_CombinedScore': np.nan
            }
            results_list.append(placeholder_row)
            print(f"  Skipping OT calculations for Cost {cost}, Round {target_rounds}.")
            continue

        # --- 4. Inner loop: OT Parameter Configurations (same logic as before) ---
        print(f"  Calculating similarities using {activation_status.lower()} activations...")
        for param_name, current_feature_params in feature_error_ot_configs:
            row_data = {
                'Cost': cost, 'Round': target_rounds, 'Param_Set_Name': param_name,
                'Local_Final_Loss': final_local_score, 'FedAvg_Final_Loss': final_fedavg_score,
                'Loss_Delta_pct': 100* (final_local_score - final_fedavg_score) / final_local_score,
                'FeatureErrorOT_Cost': np.nan, 'FeatureErrorOT_Weighting': None,
                'Decomposed_LabelEMD': np.nan, 'Decomposed_ConditionalOT': np.nan,
                'Decomposed_CombinedScore': np.nan, 'Run_Status': activation_status
            }
            try:
                analyzer = SimilarityAnalyzer(
                    client_id_1=client_id_1_str, client_id_2=client_id_2_str,
                    num_classes=num_classes, feature_error_ot_params=current_feature_params,
                    decomposed_params=decomposed_params, loss_eps=analyzer_loss_eps
                )
                load_success = analyzer.load_activations(*activations_data)
                if not load_success:
                    raise RuntimeError(f"SimilarityAnalyzer failed to load valid activations.")

                analyzer.calculate_all()
                ot_results = analyzer.get_results()

                if ot_results:
                    fe_ot = ot_results.get('feature_error_ot', {})
                    decomp = ot_results.get('decomposed', {})
                    row_data['FeatureErrorOT_Cost'] = fe_ot.get('ot_cost', np.nan)
                    row_data['FeatureErrorOT_Weighting'] = fe_ot.get('weighting_used', None)
                    row_data['Decomposed_LabelEMD'] = decomp.get('label_emd', np.nan)
                    row_data['Decomposed_ConditionalOT'] = decomp.get('conditional_ot', np.nan)
                    row_data['Decomposed_CombinedScore'] = decomp.get('combined_score', np.nan)
                    row_data['Run_Status'] += " + OT Success"
                else:
                    warnings.warn(f"Cost {cost}, R{target_rounds}, P '{param_name}': OT results retrieval failed.")
                    row_data['Run_Status'] += " + Failed OT Retrieve"

            except Exception as e_ot:
                warnings.warn(f"Cost {cost}, R{target_rounds}, P '{param_name}': FAILED OT calculation - {type(e_ot).__name__}: {e_ot}", stacklevel=2)
                row_data['Run_Status'] += f" + Failed OT Calc: {type(e_ot).__name__}"
            finally:
                results_list.append(row_data)

        # --- Cleanup ---
        activations_data = None
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception as e_cuda:
                warnings.warn(f"Error during CUDA cache cleanup: {e_cuda}")


    # --- Final DataFrame Creation ---
    print("\n--- Pipeline Finished ---")
    if not results_list:
        print("WARNING: No results were generated.")
        return pd.DataFrame()

    results_df = pd.DataFrame(results_list)
    cols_order = [
        'Cost', 'Round', 'Param_Set_Name', 'Run_Status',
        'Local_Final_Loss', 'FedAvg_Final_Loss', 'Loss_Delta_pct',
        'FeatureErrorOT_Cost', 'FeatureErrorOT_Weighting',
        'Decomposed_CombinedScore', 'Decomposed_LabelEMD', 'Decomposed_ConditionalOT'
    ]
    cols_present = [col for col in cols_order if col in results_df.columns]
    results_df = results_df[cols_present]
    return results_df


# ==============================================================================
# Activation Extraction Function (Standalone)
# ==============================================================================

# Function definition remains the same
def get_acts_for_similarity(client_id_key, server_instance, dataloaders_dict, loader_type='train'):
    """
    Gets activations and labels by processing the ENTIRE specified dataloader
    for a given client using its 'best' or current model. Handles batch size 1 correctly.

    Args:
        client_id_key (str or int): The identifier for the client.
        server_instance: The server object (e.g., server_fedavg) containing clients.
        dataloaders_dict (dict): Dictionary mapping client_id_key to (train, val, test) loaders.
        loader_type (str): Which loader to use ('train', 'val', or 'test'). Default is 'train'.

    Returns:
        torch.Tensor or None: Concatenated pre-final layer activations [N_total, D_pre].
        torch.Tensor or None: Concatenated post-sigmoid/softmax activations [N_total] or [N_total, C].
        torch.Tensor or None: Concatenated labels [N_total].
        Returns None for all if errors occur or loader is empty.
    """
    client_id_str = str(client_id_key)
    if client_id_str not in server_instance.clients:
        raise KeyError(f"Client '{client_id_str}' not found in server.")

    # Find the correct key for the dataloader dictionary (same logic as before)
    loader_key = None
    if client_id_key in dataloaders_dict:
        loader_key = client_id_key
    elif client_id_str in dataloaders_dict:
        loader_key = client_id_str
    else:
        try:
            int_key = int(client_id_key)
            if int_key in dataloaders_dict:
                loader_key = int_key
        except ValueError:
            pass
        if loader_key is None:
             raise KeyError(f"Dataloader key matching '{client_id_key}' or '{client_id_str}' not found.")

    client_obj = server_instance.clients[client_id_str]
    state_obj = client_obj.get_client_state(personal=False)
    if hasattr(state_obj, 'best_model') and state_obj.best_model is not None:
        model = state_obj.best_model
    else:
        warnings.warn(f"Client {client_id_str} has no best_model state. Using current model.")
        model = state_obj.model

    device = server_instance.device if hasattr(server_instance, 'device') else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    loader_idx = {'train': 0, 'val': 1, 'test': 2}.get(loader_type.lower())
    if loader_idx is None:
        raise ValueError(f"Invalid loader_type: '{loader_type}'. Choose 'train', 'val', or 'test'.")

    try:
        data_loader = dataloaders_dict[loader_key][loader_idx]
        if data_loader is None or len(data_loader) == 0 or len(data_loader.dataset) == 0:
            warnings.warn(f"{loader_type.capitalize()} loader is empty for client {client_id_str} (key: {loader_key}).")
            return None, None, None
    except Exception as e:
        warnings.warn(f"Error accessing {loader_type} loader for client {client_id_str} (key: {loader_key}): {e}")
        return None, None, None

    all_pre_activations = []
    all_post_activations = []
    all_labels = []

    # Find the final linear layer (same logic as before)
    final_linear = None
    possible_names = ['fc', 'linear', 'classifier', 'output']
    for name in possible_names:
        module = getattr(model, name, None)
        if module is not None:
            if isinstance(module, torch.nn.Linear): final_linear = module; break
            elif isinstance(module, torch.nn.Sequential) and len(module) > 0 and isinstance(module[-1], torch.nn.Linear): final_linear = module[-1]; break
    if final_linear is None:
        for module in reversed(list(model.modules())):
            if isinstance(module, torch.nn.Linear):
                final_linear = module
                warnings.warn(f"Using potentially non-final Linear layer '{module}' for hooks.")
                break
        if final_linear is None:
            raise AttributeError("Could not automatically find final linear layer for hooking.")

    # --- Hooking Logic (same as before) ---
    current_batch_pre_acts_storage = []
    current_batch_post_logits_storage = []
    def pre_hook(module, input):
        inp = input[0] if isinstance(input, tuple) else input
        if isinstance(inp, torch.Tensor):
            current_batch_pre_acts_storage.append(inp.detach())

    def post_hook(module, input, output):
        out = output[0] if isinstance(output, tuple) else output
        if isinstance(out, torch.Tensor):
            current_batch_post_logits_storage.append(out.detach())

    pre_handle = final_linear.register_forward_pre_hook(pre_hook)
    post_handle = final_linear.register_forward_hook(post_hook)

    # --- Process Data ---
    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x = batch_x.to(device)
            # Keep batch_y on its original device for now

            current_batch_pre_acts_storage.clear()
            current_batch_post_logits_storage.clear()

            out = model(batch_x) # Forward pass triggers hooks

            if not current_batch_pre_acts_storage or not current_batch_post_logits_storage:
                warnings.warn(f"Hooks failed for a batch in client {client_id_str}. Skipping batch.")
                continue

            pre_acts_batch = current_batch_pre_acts_storage[0].cpu()
            post_logits_batch = current_batch_post_logits_storage[0].cpu()

            # Handle post activations (sigmoid for binary, softmax for multi-class)
            if post_logits_batch.shape[1] == 1:
                 # Binary classification or regression output
                 post_probs_batch = torch.sigmoid(post_logits_batch).squeeze(-1) # Shape: [batch_size]
            elif post_logits_batch.ndim == 1: # If already squeezed
                 post_probs_batch = torch.sigmoid(post_logits_batch) # Shape: [batch_size]
            else:
                 # Multi-class classification
                 post_probs_batch = torch.softmax(post_logits_batch, dim=1) # Shape: [batch_size, num_classes]

            all_pre_activations.append(pre_acts_batch)
            all_post_activations.append(post_probs_batch)

            # --- FIX: Ensure labels are 1D ---
            # Move to CPU and reshape to (-1). This ensures it's 1D.
            # If batch_y was (), reshape makes it (1,).
            # If batch_y was (B,), reshape keeps it (B,).
            all_labels.append(batch_y.cpu().reshape(-1))
            # --- END FIX ---

    pre_handle.remove()
    post_handle.remove()

    # --- Concatenate Results ---
    if not all_pre_activations: # Or check all_labels
        warnings.warn(f"No batches processed successfully for client {client_id_str}.")
        return None, None, None
    try:
        final_pre_activations = torch.cat(all_pre_activations, dim=0)
        final_post_activations = torch.cat(all_post_activations, dim=0)
        final_labels = torch.cat(all_labels, dim=0) # Should now work correctly
    except Exception as e:
        warnings.warn(f"Error concatenating batch results for client {client_id_str}: {e}")
        # Optional: Add debugging prints for shapes if needed
        # for i, t in enumerate(all_pre_activations): print(f"Pre act batch {i} shape: {t.shape}")
        # for i, t in enumerate(all_post_activations): print(f"Post act batch {i} shape: {t.shape}")
        # for i, t in enumerate(all_labels): print(f"Label batch {i} shape: {t.shape}") # Check label shapes here
        return None, None, None

    return final_pre_activations, final_post_activations, final_labels

# ==============================================================================
# Plotting Function (Standalone)
# ==============================================================================
# Assume plot_client_comparison_label_color_subplots is defined as before
# (Make sure it imports necessary plotting libraries: matplotlib, numpy, etc.)

# Optional imports, check if available
try:
    import umap
    umap_available = True
except ImportError:
    umap_available = False
    warnings.warn("UMAP library not found. Dimensionality reduction will default to PCA.")

try:
    from sklearn.decomposition import PCA
    pca_available = True
except ImportError:
    pca_available = False
    warnings.warn("Scikit-learn (for PCA) not found. Dimensionality reduction requires UMAP or Scikit-learn.")

try:
    import seaborn as sns
    seaborn_available = True
except ImportError:
    seaborn_available = False
    warnings.warn("Seaborn library not found. plot_act_classes will use basic matplotlib histograms.")

try:
    import ot # Python Optimal Transport library
    pot_available = True
except ImportError:
    pot_available = False
    warnings.warn("POT (Python Optimal Transport) library not found. OT cost calculation will not work.")

def plot_client_comparison(
    h1, y1, h2, y2,
    client_id_1="Client 1",
    client_id_2="Client 2",
    method='umap',
    title_base="Client Data Comparison",
    figsize_per_subplot=(8, 7), # Size per subplot
    point_size=25,
    n_neighbors=15,
    min_dist=0.1,
    pca_components=2,
    max_plots_per_row=3, # Maximum plots per row
    save_path=None
    ):
    """
    Performs dimensionality reduction and generates a figure with multiple subplots:
    1. Combined data: Color = True Label (y), Shape = Client ID.
    2..N+1. Data for each unique label: Color = Client ID, Shape = Client ID.
    
    Subplots are arranged with a maximum of 3 per row for better visualization.

    Args:
        h1, y1, h2, y2: Activations and labels for client 1 and client 2.
        client_id_1, client_id_2: Names for the clients.
        method: 'umap' or 'pca'.
        title_base: Base string for the main figure title.
        figsize_per_subplot: Size for each individual subplot.
        point_size: Scatter point size.
        n_neighbors, min_dist: UMAP parameters.
        pca_components: PCA parameter.
        max_plots_per_row: Maximum number of subplots per row.
        save_path: If provided, saves the entire figure. Otherwise, shows the figure.
    """
    
    if h1 is None or h2 is None or y1 is None or y2 is None:
        print("Error: Missing data for plotting.")
        return None
        
    # Convert to numpy if they are torch tensors
    if hasattr(h1, 'detach'): 
        h1_np = h1.detach().cpu().numpy()
    else: 
        h1_np = np.asarray(h1)
        
    if hasattr(y1, 'detach'): 
        y1_np = y1.detach().cpu().numpy().astype(int)
    else: 
        y1_np = np.asarray(y1).astype(int)
        
    if hasattr(h2, 'detach'): 
        h2_np = h2.detach().cpu().numpy()
    else: 
        h2_np = np.asarray(h2)
        
    if hasattr(y2, 'detach'): 
        y2_np = y2.detach().cpu().numpy().astype(int)
    else: 
        y2_np = np.asarray(y2).astype(int)

    if len(h1_np) == 0 or len(h2_np) == 0:
        print("Error: Empty data arrays for plotting.")
        return None

    # --- 1. Data Preparation & Dim Reduction (Done Once) ---
    h_combined = np.vstack((h1_np, h2_np))
    n1 = len(h1_np)

    reducer = None
    if method.lower() == 'umap':
        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=2, random_state=42)
    else:
        reducer = PCA(n_components=pca_components, random_state=42)
        method = 'pca'

    try:
        h_combined_2d = reducer.fit_transform(h_combined)
        h1_2d = h_combined_2d[:n1, :]
        h2_2d = h_combined_2d[n1:, :]
    except Exception as e:
        print(f"Error during dimensionality reduction with {method}: {e}")
        return None

    plt.style.use('seaborn-v0_8-whitegrid')
    reduced_dim_name = method.upper()
    client_markers = ['o', 'X'] # Circle for client 1, X for client 2

    # --- Define Colors ---
    unique_labels = np.unique(np.concatenate((y1_np, y2_np)))
    num_unique_labels = len(unique_labels)

    # Use appropriate colormap based on number of classes
    if num_unique_labels <= 10:
        cmap_labels = plt.get_cmap('Set1')
    else:
        cmap_labels = plt.get_cmap('viridis')  # Better for many classes
        
    label_to_color_val = {label: i for i, label in enumerate(unique_labels)}
    norm_labels = plt.Normalize(vmin=0, vmax=max(0, num_unique_labels - 1)) # Handle case of 1 label

    client_color_map = {
        client_id_1: plt.get_cmap('tab10')(0), # Blue
        client_id_2: plt.get_cmap('tab10')(1)  # Orange
    }

    # ================================================
    # --- Calculate subplot layout ---
    # ================================================
    num_subplots = 1 + num_unique_labels  # Combined data + one for each class
    num_rows = (num_subplots + max_plots_per_row - 1) // max_plots_per_row  # Ceiling division
    num_cols = min(max_plots_per_row, num_subplots)
    
    # Calculate total figure size
    total_figsize = (figsize_per_subplot[0] * num_cols, figsize_per_subplot[1] * num_rows)
    
    # Create figure and grid of subplots
    fig = plt.figure(figsize=total_figsize)
    gs = fig.add_gridspec(num_rows, num_cols)
    axes = []
    
    for i in range(num_subplots):
        row = i // max_plots_per_row
        col = i % max_plots_per_row
        axes.append(fig.add_subplot(gs[row, col]))
    
    fig.suptitle(f"{title_base} ({reduced_dim_name} Projection)", fontsize=16)

    # --- Plot 1: Combined Data (Axes[0]) ---
    ax = axes[0]
    ax.scatter(h1_2d[:, 0], h1_2d[:, 1],
               c=[cmap_labels(norm_labels(label_to_color_val.get(y, -1))) for y in y1_np], # Use .get for safety
               marker=client_markers[0],
               s=point_size, alpha=0.6, edgecolors='grey', linewidth=0.5)
    ax.scatter(h2_2d[:, 0], h2_2d[:, 1],
               c=[cmap_labels(norm_labels(label_to_color_val.get(y, -1))) for y in y2_np],
               marker=client_markers[1],
               s=point_size, alpha=0.6, edgecolors='black', linewidth=0.5)

    # Legends for Plot 1 - Combined into one for better placement
    client_legend_handles = [
        mlines.Line2D([0], [0], marker=client_markers[0], color='grey', lw=0, label=client_id_1),
        mlines.Line2D([0], [0], marker=client_markers[1], color='grey', lw=0, markeredgecolor='k', label=client_id_2)
    ]
    label_legend_handles = [
        mlines.Line2D([0], [0], marker='s', color=cmap_labels(norm_labels(label_to_color_val[label])), lw=0, label=f'Label {label}')
        for label in unique_labels if label in label_to_color_val # Ensure label exists
    ]
    # Place combined legend inside plot 1
    ax.legend(handles=client_legend_handles + label_legend_handles, title="Client (Shape) & Label (Color)", loc='best', fontsize=9)

    ax.set_title("All Labels (Color=Label)", fontsize=12)
    ax.set_ylabel(f'{reduced_dim_name} Dimension 2', fontsize=10)
    ax.set_xlabel(f'{reduced_dim_name} Dimension 1', fontsize=10)

    # --- Plots for each label ---
    for i, label_to_plot in enumerate(unique_labels):
        ax_idx = i + 1
        if ax_idx < len(axes):  # Safety check
            ax = axes[ax_idx]
            
            mask1 = (y1_np == label_to_plot)
            mask2 = (y2_np == label_to_plot)
            h1_2d_filtered = h1_2d[mask1]
            h2_2d_filtered = h2_2d[mask2]

            plotted_something = False
            if len(h1_2d_filtered) > 0:
                ax.scatter(h1_2d_filtered[:, 0], h1_2d_filtered[:, 1],
                        color=client_color_map[client_id_1], marker=client_markers[0],
                        s=point_size, alpha=0.7, edgecolors='grey', linewidth=0.5, label=client_id_1)
                plotted_something = True
            if len(h2_2d_filtered) > 0:
                ax.scatter(h2_2d_filtered[:, 0], h2_2d_filtered[:, 1],
                        color=client_color_map[client_id_2], marker=client_markers[1],
                        s=point_size, alpha=0.7, edgecolors='black', linewidth=0.5, label=client_id_2)
                plotted_something = True

            if plotted_something:
                ax.legend(title="Client", loc='best', fontsize=9)
            else:
                ax.text(0.5, 0.5, f'No data for\nLabel {label_to_plot}', 
                       horizontalalignment='center', verticalalignment='center', 
                       transform=ax.transAxes, fontsize=10)
                
            ax.set_title(f"Label {label_to_plot} Only (Color=Client)", fontsize=12)
            ax.set_xlabel(f'{reduced_dim_name} Dimension 1', fontsize=10)
            ax.set_ylabel(f'{reduced_dim_name} Dimension 2', fontsize=10)

    # --- Final Adjustments ---
    for ax in axes:
        ax.tick_params(axis='both', which='major', labelsize=9)
        ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust rect to make space for suptitle

    # --- Save or Show ---
    if save_path:
        try:
            # Create directory if it doesn't exist
            save_dir = os.path.dirname(save_path)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
                print(f"  Created directory: {save_dir}")
                
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  Subplot figure saved as {save_path}")
        except Exception as e:
            print(f"  Error saving subplot figure: {e}")
    else:
        print("  Displaying subplot figure...")
        plt.show()

    plt.close(fig) # Close the figure
    return fig # Return the figure object


def get_class_specific_activations(h1, p1_sig, y1, h2, p2_sig, y2, 
                                client_id_1="Client 1", client_id_2="Client 2", 
                                num_classes=2, filter_correct=None):
    """
    Filters activations by class label and optionally by prediction correctness.

    Args:
        h1, p1_sig, y1: Data for client 1.
        h2, p2_sig, y2: Data for client 2.
        client_id_1, client_id_2: Names for the clients.
        num_classes: Number of classes in the dataset.
        filter_correct (bool | None):
            None: No filtering by correctness.
            True: Keep only correctly predicted samples.
            False: Keep only incorrectly predicted samples.

    Returns:
        dict: Dictionary containing filtered activations, predictions, and labels.
    """
    class_specific_activations = {
        client_id_1: {},
        client_id_2: {}
    }

    # Determine correctness masks if filtering is enabled
    correct_mask1 = None
    correct_mask2 = None
    if filter_correct is not None:
        pred1 = p1_sig.argmax(axis=1)
        pred2 = p2_sig.argmax(axis=1)
        is_correct1 = (pred1 == y1.long())
        is_correct2 = (pred2 == y2.long())
        if filter_correct: # Keep correct
            correct_mask1 = is_correct1
            correct_mask2 = is_correct2
        else: # Keep incorrect
            correct_mask1 = ~is_correct1
            correct_mask2 = ~is_correct2
    else:
        pred1 = (p1_sig >= 0.5).long()
        pred2 = (p2_sig >= 0.5).long()

    for class_label in range(num_classes):
        # Client 1
        class_mask1 = (y1.long() == class_label)
        final_mask1 = class_mask1
        if correct_mask1 is not None:
            final_mask1 = final_mask1 & correct_mask1 # Combine class and correctness masks

        h1_c = h1[final_mask1]
        p1_sig_c = p1_sig[final_mask1]
        y1_c = y1[final_mask1]
        class_specific_activations[client_id_1][class_label] = {
            'h': h1_c, 'p_sig': p1_sig_c, 'y': y1_c
        }

        # Client 2
        class_mask2 = (y2.long() == class_label)
        final_mask2 = class_mask2
        if correct_mask2 is not None:
            final_mask2 = final_mask2 & correct_mask2 # Combine class and correctness masks

        h2_c = h2[final_mask2]
        p2_sig_c = p2_sig[final_mask2]
        y2_c = y2[final_mask2]
        class_specific_activations[client_id_2][class_label] = {
            'h': h2_c, 'p_sig': p2_sig_c, 'y': y2_c
        }
    return class_specific_activations


def compute_ot_cost(cost_matrix_torch, eps=1e-3, sinkhorn_thresh=1e-3, sinkhorn_max_iter=5000):
    """
    Compute optimal transport cost using POT library.
    
    This is a helper function for plot_act_classes.
    """
    try:
        import ot
    except ImportError:
        print("Error: POT (Python Optimal Transport) library not installed.")
        return None, None
        
    if cost_matrix_torch is None or cost_matrix_torch.numel() == 0: 
        return np.nan, None
        
    cost_matrix_np = cost_matrix_torch.detach().cpu().numpy().astype(np.float64)
    N, M = cost_matrix_np.shape
    if N == 0 or M == 0: 
        return 0.0, np.zeros((N, M)) # Cost is 0 if one side is empty

    a = np.ones((N,), dtype=np.float64) / N
    b = np.ones((M,), dtype=np.float64) / M

    try:
        costs_stable = cost_matrix_np
        Gs = ot.bregman.sinkhorn_stabilized(a, b, costs_stable, reg=eps,
                                            stopThr=sinkhorn_thresh, numItermax=sinkhorn_max_iter,
                                            warn=False, verbose=False) # Turn off POT warnings/verbose
        if np.isnan(Gs).any():
            print(f"Warning: NaNs in transport plan Gs (eps={eps}). Trying regular Sinkhorn.")
            # Fallback to regular sinkhorn which might be more stable for larger eps
            Gs = ot.sinkhorn(a, b, costs_stable, reg=eps, stopThr=sinkhorn_thresh,
                            numItermax=sinkhorn_max_iter, warn=False, verbose=False)
            if np.isnan(Gs).any():
                print(f"Warning: NaNs persist even with regular Sinkhorn. OT cost invalid.")
                return np.nan, Gs

        ot_cost = np.sum(Gs * cost_matrix_np) # Use original costs for final calculation
        return float(ot_cost), Gs
    except Exception as e:
        print(f"Error during OT computation: {e}")
        return np.nan, None


def plot_act_classes(h1, p1_sig, y1, h2, p2_sig, y2, 
                   client_id_1="Client 1", client_id_2="Client 2", 
                   num_classes=None, filter_correct=None, 
                   figsize_per_subplot=(5, 4), max_plots_per_row=3,
                   feature_dist_func='cosine'):
    """
    Plots histograms of pairwise feature costs for class-specific activations,
    optionally filtering by prediction correctness.

    Args:
        h1, p1_sig, y1: Data for client 1.
        h2, p2_sig, y2: Data for client 2.
        client_id_1, client_id_2: Names for the clients.
        num_classes: Number of classes in the dataset (if None, inferred from data).
        filter_correct (bool | None):
            None: No filtering by correctness.
            True: Keep only correctly predicted samples.
            False: Keep only incorrectly predicted samples.
        figsize_per_subplot: Size for each individual subplot.
        max_plots_per_row: Maximum number of plots per row.
        feature_dist_func: Distance function to use ('cosine' or 'l2_sq').
        
    Returns:
        Figure object.
    """
    print(f"Generating class activation cost histograms (Filter: {filter_correct})...")
    
    # Get class-specific data
    class_specific_activations = get_class_specific_activations(
        h1, p1_sig, y1, h2, p2_sig, y2,
        client_id_1=client_id_1, client_id_2=client_id_2,
        num_classes=num_classes, filter_correct=filter_correct
    )
    
    if class_specific_activations is None:
        print("Error: Failed to get class-specific activations.")
        return None

    # Determine the actual classes available
    if client_id_1 in class_specific_activations and class_specific_activations[client_id_1]:
        actual_labels = sorted(class_specific_activations[client_id_1].keys())
    elif client_id_2 in class_specific_activations and class_specific_activations[client_id_2]:
        actual_labels = sorted(class_specific_activations[client_id_2].keys())
    else:
        print("Error: No class data found after filtering.")
        return None
        
    num_classes_to_plot = len(actual_labels)
    if num_classes_to_plot == 0:
        print("Warning: No classes found in data.")
        return None

    # Calculate grid layout
    num_rows = (num_classes_to_plot + max_plots_per_row - 1) // max_plots_per_row  # Ceiling division
    num_cols = min(max_plots_per_row, num_classes_to_plot)
    
    # Calculate total figure size
    total_figsize = (figsize_per_subplot[0] * num_cols, figsize_per_subplot[1] * num_rows)
    
    # Create figure with grid layout
    fig = plt.figure(figsize=total_figsize)
    gs = fig.add_gridspec(num_rows, num_cols)
    axes = []
    
    for i in range(num_classes_to_plot):
        row = i // max_plots_per_row
        col = i % max_plots_per_row
        axes.append(fig.add_subplot(gs[row, col]))

    # Determine Title Suffix based on filtering
    title_suffix = ""
    if filter_correct is True:
        title_suffix = " (Correctly Predicted Only)"
    elif filter_correct is False:
        title_suffix = " (Incorrectly Predicted Only)"
    
    fig.suptitle(f"Distribution of Pairwise Feature Costs ('{feature_dist_func}') by Class{title_suffix}", fontsize=14)

    # Plot each class
    for i, label in enumerate(actual_labels):
        if i >= len(axes):  # Safety check
            break
            
        ax = axes[i]
        h1_c = class_specific_activations[client_id_1][label]['h']
        h2_c = class_specific_activations[client_id_2][label]['h']

        # Proceed only if both clients have data for this class/filter combination
        if h1_c.shape[0] > 0 and h2_c.shape[0] > 0:
            # Calculate cost matrix based on selected distance function
            if feature_dist_func == 'cosine':
                cost_matrix_features = torch.from_numpy(cosine_distances(
                                                    h1_c.detach().cpu().numpy(),
                                                    h2_c.detach().cpu().numpy()
                                                )).float()
                cost_matrix_features /= 2.0  # Normalize cosine distance [0,1]
                cost_label = "Normalized Cosine Distance"
            elif feature_dist_func == 'l2_sq':
                # Use torch cdist for squared L2 distance
                h1_t = torch.tensor(h1_c).float() if not isinstance(h1_c, torch.Tensor) else h1_c.float()
                h2_t = torch.tensor(h2_c).float() if not isinstance(h2_c, torch.Tensor) else h2_c.float()
                cost_matrix_features = torch.cdist(h1_t, h2_t, p=2) ** 2
                cost_label = "Squared L2 Distance"
            else:
                print(f"Warning: Unknown distance function '{feature_dist_func}'. Using cosine.")
                cost_matrix_features = torch.from_numpy(cosine_distances(
                                                    h1_c.detach().cpu().numpy(),
                                                    h2_c.detach().cpu().numpy()
                                                )).float()
                cost_matrix_features /= 2.0
                cost_label = "Normalized Cosine Distance"
            
            # Calculate OT cost and plot histogram
            ot_cost_features, _ = compute_ot_cost(cost_matrix_features, eps=1e-3)
            cost_values_flat = cost_matrix_features.detach().cpu().numpy().reshape(-1)

            try:
                sns.histplot(cost_values_flat, kde=True, bins=50, ax=ax)
            except:
                # Fallback if seaborn fails
                ax.hist(cost_values_flat, bins=50, density=True, alpha=0.7)

            if not np.isnan(ot_cost_features):  # Only plot line if OT cost is valid
                ax.axvline(ot_cost_features, color='r', linestyle='--', linewidth=2,
                           label=f'OT Cost ({ot_cost_features:.4f})')
                ax.legend(fontsize=9)
        else:
            ax.text(0.5, 0.5, f'Not enough data\nfor Label {label}', 
                   horizontalalignment='center', verticalalignment='center', 
                   transform=ax.transAxes, fontsize=10)

        ax.set_title(f"Label {label}", fontsize=11)
        ax.set_xlabel(cost_label, fontsize=10)
        ax.set_ylabel("Density" if i % max_plots_per_row == 0 else "")
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.6)
        ax.tick_params(axis='both', which='major', labelsize=9)

    # Remove empty subplots if any
    for i in range(num_classes_to_plot, num_rows * num_cols):
        row = i // max_plots_per_row
        col = i % max_plots_per_row
        if row < num_rows and col < num_cols:
            fig.delaxes(fig.add_subplot(gs[row, col]))

    plt.tight_layout(rect=[0, 0.03, 1, 0.93])  # Adjust layout for title
    return fig


def plot_ot_costs_vs_perf_delta(
    results_df: pd.DataFrame,
    param_sets_to_plot: Optional[List[str]] = None,
    main_title: Optional[str] = None,
    save_path: Optional[str] = None
    ):
    """
    Generates a figure with three subplots in one row, each showing a
    different calculated OT cost metric vs. Loss_Delta_pct. Lines are
    colored by parameter set name. Allows filtering of parameter sets.
    Legend is placed in the top-right corner outside the plot area.

    Args:
        results_df: DataFrame generated by the pipeline, containing OT cost columns,
                    'Loss_Delta_pct', and 'Param_Set_Name'.
        param_sets_to_plot: Optional list of strings specifying which 'Param_Set_Name'
                            values to include in the plot. If None, all are plotted.
        main_title: Optional main title for the entire figure.
        save_path: Optional path to save the plot figure. If None, displays the plot.
    """
    # Define the columns needed for the plots
    y_var = 'Loss_Delta_pct'
    hue_var = 'Param_Set_Name'
    x_vars = ['FeatureErrorOT_Cost', 'Decomposed_CombinedScore', 'Decomposed_ConditionalOT']
    x_labels = ['Feature-Error OT Cost', 'Decomposed Combined Score', 'Decomposed Conditional OT']
    subplot_titles = [
        f'{x_labels[0]} vs. Perf. Delta (%)',
        f'{x_labels[1]} vs. Perf. Delta (%)',
        f'{x_labels[2]} vs. Perf. Delta (%)'
    ]

    required_base_columns = [y_var, hue_var]
    required_all_columns = required_base_columns + x_vars

    # --- Input Validation ---
    if results_df is None or results_df.empty:
        warnings.warn("Input DataFrame is empty or None. Cannot generate plot.")
        return
    if not all(col in results_df.columns for col in required_base_columns):
        warnings.warn(f"Input DataFrame missing base columns: {required_base_columns}. Cannot plot.")
        return
    if not any(col in results_df.columns for col in x_vars):
         warnings.warn(f"Input DataFrame missing all x-variable columns: {x_vars}. Cannot plot.")
         return

    plot_df = results_df.copy()

    # --- Filtering Parameter Sets ---
    if param_sets_to_plot is not None:
        plot_df = plot_df[plot_df[hue_var].isin(param_sets_to_plot)]
        if plot_df.empty:
            warnings.warn(f"DataFrame empty after filtering for: {param_sets_to_plot}. Cannot plot.")
            return
        print(f"Plotting only for parameter sets: {param_sets_to_plot}")

    # --- Subplot Setup ---
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=True)
    if len(x_vars) == 1: axes = [axes]
    elif len(x_vars) != 3:
         valid_x_vars = [col for col in x_vars if col in plot_df.columns]
         num_plots = len(valid_x_vars)
         if num_plots == 0: warnings.warn("No valid X columns found."); return
         fig, axes = plt.subplots(1, num_plots, figsize=(7*num_plots, 6), sharey=True)
         if num_plots == 1: axes = [axes]


    plot_successful = False
    plotted_axes_indices = []

    # --- Plotting Loop ---
    plot_index = 0
    for i, x_var in enumerate(x_vars):
        if x_var not in plot_df.columns:
            warnings.warn(f"Column '{x_var}' not found. Skipping subplot.")
            if len(axes) == 3: # Handle fixed axes layout if column missing
                 ax = axes[i]
                 ax.set_title(f"{subplot_titles[i]}\n(Data Unavailable)", fontsize=14)
                 ax.text(0.5, 0.5, "Data Unavailable", ha='center', va='center', transform=ax.transAxes, fontsize=12, color='red')
            continue

        if plot_index >= len(axes):
            warnings.warn(f"Mismatch between x_vars and axes. Skipping plot for {x_var}.")
            continue
        ax = axes[plot_index]

        subplot_req_cols = [x_var, y_var, hue_var]
        plot_df_subplot = plot_df.dropna(subset=subplot_req_cols).copy()

        if plot_df_subplot.empty:
            warnings.warn(f"No valid data for subplot ('{x_var}' vs '{y_var}'). Skipping.")
            ax.set_title(f"{subplot_titles[i]}\n(No Valid Data)", fontsize=14)
            ax.text(0.5, 0.5, "No Valid Data", ha='center', va='center', transform=ax.transAxes, fontsize=12, color='orange')
            continue

        try:
            plot_df_subplot[x_var] = pd.to_numeric(plot_df_subplot[x_var])
            plot_df_subplot[y_var] = pd.to_numeric(plot_df_subplot[y_var])
        except Exception as e:
            warnings.warn(f"Numeric conversion error for subplot '{x_var}': {e}. Skipping.")
            ax.set_title(f"{subplot_titles[i]}\n(Data Type Error)", fontsize=14)
            ax.text(0.5, 0.5, "Data Type Error", ha='center', va='center', transform=ax.transAxes, fontsize=12, color='red')
            continue

        sns.lineplot(
            data=plot_df_subplot, x=x_var, y=y_var, hue=hue_var, style=hue_var,
            marker='o', markersize=8, linewidth=2, ax=ax
        )

        ax.set_title(subplot_titles[i], fontsize=14)
        ax.set_xlabel(x_labels[i], fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.6)

        plot_successful = True
        plotted_axes_indices.append(plot_index)
        plot_index += 1


    # --- Figure-Level Customization ---
    if not plot_successful:
         warnings.warn("No subplots could be generated.")
         plt.close(fig)
         return

    # Set shared Y-label on the first successfully plotted axis
    if plotted_axes_indices:
         first_ax_idx = plotted_axes_indices[0]
         axes[first_ax_idx].set_ylabel("Performance Delta (%) [Local - FedAvg] / Local", fontsize=12)

    fig_title = main_title if main_title else "OT Cost Metrics vs. Performance Delta by Parameter Set"
    # <<< MODIFIED: Adjusted y position for suptitle slightly >>>
    fig.suptitle(fig_title, fontsize=18, y=0.99)

    # --- Create a single figure legend ---
    handles, labels = [], []
    if plotted_axes_indices:
        first_ax_idx = plotted_axes_indices[0]
        handles, labels = axes[first_ax_idx].get_legend_handles_labels()
        for idx in plotted_axes_indices:
             ax_legend = axes[idx].get_legend()
             if ax_legend is not None:
                  ax_legend.remove()

    if handles:
        # <<< MODIFIED: Legend position changed to top-right outside >>>
        fig.legend(handles, labels, title='Parameter Set',
                   bbox_to_anchor=(1.01, 1), # Position slightly outside top-right
                   loc='upper left',         # Align legend's upper left to anchor
                   ncol=1,                   # Stack vertically
                   borderaxespad=0.)
    else:
        warnings.warn("Could not retrieve handles/labels for figure legend.")


    # Adjust layout
    # <<< MODIFIED: Adjust rect to make space for legend on the right >>>
    fig.tight_layout(rect=[0, 0, 0.88, 0.95]) # [left, bottom, right, top]

    # --- Output ---
    if save_path:
        try:
            save_dir = os.path.dirname(save_path)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir)
            # Use bbox_inches='tight' when saving to try and include external legend
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        except Exception as e:
            warnings.warn(f"Failed to save plot to {save_path}: {e}")
        plt.close(fig)
    else:
        plt.show()