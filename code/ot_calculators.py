# ot_calculators.py
import torch
import torch.nn.functional as F
import numpy as np
import warnings
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, Type, Union # Added Type

# Import utilities from ot_utils.py
from ot_utils import (
    compute_ot_cost, pairwise_euclidean_sq, calculate_label_emd, compute_anchors,
    DEFAULT_OT_REG, DEFAULT_OT_MAX_ITER, DEFAULT_EPS, calculate_sample_loss # Added calculate_sample_loss
)

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++ Added OTConfig and OTCalculatorFactory classes below +++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

class OTConfig:
    """
    Configuration object for a single OT calculation run.
    Includes basic validation.
    """
    # Define known method types to prevent typos
    KNOWN_METHOD_TYPES = {'feature_error', 'decomposed', 'fixed_anchor'}

    def __init__(self, method_type: str, name: str, params: Optional[Dict[str, Any]] = None):
        """
        Args:
            method_type (str): The type of OT calculator (e.g., 'feature_error').
            name (str): A unique descriptive name for this configuration set (e.g., 'FE_LossW_Norm').
            params (dict, optional): Dictionary of parameters specific to the OT method. Defaults to {}.
        """
        if not isinstance(method_type, str) or not method_type:
            raise ValueError("method_type must be a non-empty string.")
        if method_type not in self.KNOWN_METHOD_TYPES:
            warnings.warn(f"Unknown method_type '{method_type}'. Ensure a corresponding calculator exists in the factory.", UserWarning)
        self.method_type = method_type

        if not isinstance(name, str) or not name:
            raise ValueError("name must be a non-empty string.")
        self.name = name

        self.params = params if params is not None else {}
        if not isinstance(self.params, dict):
            raise ValueError("params must be a dictionary.")

    def __repr__(self) -> str:
        return f"OTConfig(method_type='{self.method_type}', name='{self.name}', params={self.params})"


class OTCalculatorFactory:
    """ Factory class for creating OT calculator instances. """

    # Forward declaration for type hints within the class if needed
    # _calculator_map: Dict[str, Type['BaseOTCalculator']] # Not strictly necessary here

    @classmethod
    def _get_calculator_map(cls) -> Dict[str, Type['BaseOTCalculator']]:
        """ Internal method to access the map, allowing definition after classes. """
        # Defined below BaseOTCalculator and its subclasses
        return {
            'feature_error': FeatureErrorOTCalculator,
            'decomposed': DecomposedOTCalculator,
            'fixed_anchor': FixedAnchorLOTCalculator,
            # Register new calculator classes here
            # 'new_method': NewMethodCalculator,
        }

    @classmethod
    def register_calculator(cls, method_type: str, calculator_class: Type['BaseOTCalculator']):
        """ Allows dynamic registration (though map is currently static). """
        # This method might be less useful if the map is defined statically,
        # but kept for potential future dynamic loading.
        if not issubclass(calculator_class, BaseOTCalculator):
             raise TypeError(f"{calculator_class.__name__} must inherit from BaseOTCalculator")
        # cls._calculator_map[method_type] = calculator_class # Need to handle how map is stored/accessed if dynamic
        print(f"Note: Dynamic registration not fully implemented with static map. Add '{method_type}' to _get_calculator_map.")


    @staticmethod
    def create_calculator(config: OTConfig, client_id_1: str, client_id_2: str, num_classes: int) -> Optional['BaseOTCalculator']:
        """
        Creates an instance of the appropriate OT calculator based on the config.
        (Docstring remains the same)
        """
        calculator_map = OTCalculatorFactory._get_calculator_map()
        calculator_class = calculator_map.get(config.method_type)

        if calculator_class:
            try:
                instance = calculator_class(
                    client_id_1=client_id_1,
                    client_id_2=client_id_2,
                    num_classes=num_classes
                )
                return instance
            except Exception as e:
                warnings.warn(f"Failed to instantiate calculator for config '{config.name}' (type: {config.method_type}): {e}", stacklevel=2)
                return None
        else:
            warnings.warn(f"No calculator registered for method type '{config.method_type}' in config '{config.name}'. Skipping.", stacklevel=2)
            return None

# --- Base OT Calculator Class ---
class BaseOTCalculator(ABC):
    """
    Abstract Base Class for Optimal Transport similarity calculators.
    """
    def __init__(self, client_id_1: str, client_id_2: str, num_classes: int, **kwargs):
        if num_classes < 2:
            raise ValueError("num_classes must be >= 2.")
        self.client_id_1 = str(client_id_1)
        self.client_id_2 = str(client_id_2)
        self.num_classes = num_classes
        self.eps_num = kwargs.get('eps_num', DEFAULT_EPS) # Epsilon for numerical stability

        self.results: Dict[str, Any] = {}
        self.cost_matrices: Dict[str, Any] = {}
        self._reset_results() # Initialize result structures

    @abstractmethod
    def calculate_similarity(self, data1: Dict[str, Optional[torch.Tensor]], data2: Dict[str, Optional[torch.Tensor]], params: Dict[str, Any]) -> None:
        """ Calculates the specific OT similarity metric. (Docstring same) """
        pass

    @abstractmethod
    def _reset_results(self) -> None:
        """ Resets the internal results and cost matrix storage. """
        pass

    def get_results(self) -> Dict[str, Any]:
        """ Returns the calculated results. """
        return self.results

    def get_cost_matrices(self) -> Dict[str, Any]:
        """ Returns the computed cost matrices (if stored). """
        return self.cost_matrices

    # --- Shared Helper / Preprocessing Methods ---
    def _preprocess_input(self, h: Optional[torch.Tensor], p_prob: Optional[torch.Tensor], y: Optional[torch.Tensor], weights: Optional[torch.Tensor], required_keys: list[str]) -> Optional[Dict[str, torch.Tensor]]:
        """ Basic validation and conversion to CPU tensors. (Implementation same) """
        processed_data = {}
        input_map = {'h': h, 'p_prob': p_prob, 'y': y, 'weights': weights}

        for key, tensor in input_map.items():
            if tensor is None:
                if key in required_keys:
                     warnings.warn(f"Preprocessing failed: Required input '{key}' is None.")
                     return None
                processed_data[key] = None
                continue

            if not isinstance(tensor, torch.Tensor):
                try:
                    tensor = torch.tensor(tensor)
                except Exception as e:
                    warnings.warn(f"Preprocessing failed: Could not convert input '{key}' to tensor: {e}")
                    return None
            processed_data[key] = tensor.detach().cpu() # Ensure CPU tensor

        # Basic shape checks (can be expanded)
        n_samples = processed_data['y'].shape[0] if processed_data.get('y') is not None else None
        if n_samples is None and 'y' in required_keys: return None # Need y if required

        if n_samples is not None:
            if processed_data.get('h') is not None and processed_data['h'].shape[0] != n_samples:
                 warnings.warn(f"Shape mismatch: h({processed_data['h'].shape[0]}) vs y/expected({n_samples})")
                 return None
            if processed_data.get('p_prob') is not None and processed_data['p_prob'].shape[0] != n_samples:
                 warnings.warn(f"Shape mismatch: p_prob({processed_data['p_prob'].shape[0]}) vs y/expected({n_samples})")
                 return None
            if processed_data.get('weights') is not None and processed_data['weights'].shape[0] != n_samples:
                 warnings.warn(f"Shape mismatch: weights({processed_data['weights'].shape[0]}) vs y/expected({n_samples})")
                 return None
            # Check p_prob dimensions carefully
            if processed_data.get('p_prob') is not None:
                p_tensor = processed_data['p_prob']
                valid_shape = (p_tensor.ndim == 2 and p_tensor.shape[1] == self.num_classes)
                # Allow binary case special handling if num_classes is 2
                if self.num_classes == 2 and not valid_shape:
                     # Could potentially reshape/validate binary [N] or [N,1] here if needed,
                     # but often better handled during initial data processing in DataManager
                     pass # Assuming DataManager produces [N, K] format
                elif not valid_shape:
                     warnings.warn(f"Shape mismatch: p_prob({p_tensor.shape}) vs expected N x K (K={self.num_classes})")
                     return None


        # Check if all required keys ended up non-None after processing
        if any(processed_data.get(k) is None for k in required_keys):
            missing = [k for k in required_keys if processed_data.get(k) is None]
            warnings.warn(f"Preprocessing failed: Required keys {missing} are None after processing.")
            return None

        return processed_data

# --- Concrete Calculator Implementations ---

class FeatureErrorOTCalculator(BaseOTCalculator):
    """ Calculates OT cost using the additive feature-error cost. """
    # ... (Implementation remains the same as before) ...
    def _reset_results(self) -> None:
        self.results = {
            'ot_cost': np.nan,
            'transport_plan': None,
            'cost_matrix_max_val': np.nan,
            'weighting_used': None
        }
        self.cost_matrices = {'feature_error_ot': None}

    def calculate_similarity(self, data1: Dict[str, Optional[torch.Tensor]], data2: Dict[str, Optional[torch.Tensor]], params: Dict[str, Any]) -> None:
        self._reset_results()
        verbose = params.get('verbose', False)
        use_loss_weighting = params.get('use_loss_weighting', True)
        normalize_cost = params.get('normalize_cost', True)
        reg = params.get('reg', DEFAULT_OT_REG)
        max_iter = params.get('max_iter', DEFAULT_OT_MAX_ITER)

        # --- Preprocess and Validate ---
        required_keys = ['h', 'p_prob', 'y']
        if use_loss_weighting: required_keys.append('weights')

        proc_data1 = self._preprocess_input(data1.get('h'), data1.get('p_prob'), data1.get('y'), data1.get('weights'), required_keys)
        proc_data2 = self._preprocess_input(data2.get('h'), data2.get('p_prob'), data2.get('y'), data2.get('weights'), required_keys)

        if proc_data1 is None or proc_data2 is None:
            missing_str = "'weights'" if use_loss_weighting and (data1.get('weights') is None or data2.get('weights') is None) else "h, p_prob, or y"
            warnings.warn(f"Feature-Error OT calculation requires {missing_str}. Preprocessing failed or data missing. Skipping.")
            self.results['weighting_used'] = 'Loss' if use_loss_weighting else 'Uniform' # Still record intent
            return

        h1, p1_prob, y1, w1 = proc_data1['h'], proc_data1['p_prob'], proc_data1['y'], proc_data1.get('weights')
        h2, p2_prob, y2, w2 = proc_data2['h'], proc_data2['p_prob'], proc_data2['y'], proc_data2.get('weights')
        N, M = h1.shape[0], h2.shape[0]

        if N == 0 or M == 0:
            warnings.warn("Feature-Error OT: One or both clients have zero samples. OT cost is 0.")
            self.results['ot_cost'] = 0.0
            self.results['transport_plan'] = np.zeros((N, M)) if N > 0 and M > 0 else None
            self.results['weighting_used'] = 'Loss' if use_loss_weighting else 'Uniform'
            return

        # --- Calculate Cost Matrix ---
        cost_matrix, max_cost = self._calculate_cost_feature_error_additive(
            h1, p1_prob, y1, h2, p2_prob, y2, **params
        )
        self.results['cost_matrix_max_val'] = max_cost

        if cost_matrix is None or not (np.isfinite(max_cost) and max_cost > self.eps_num):
            fail_reason = "Cost matrix calculation failed" if cost_matrix is None else "Max cost near zero/invalid"
            warnings.warn(f"Feature-Error OT calculation skipped: {fail_reason}.")
            self.results['weighting_used'] = 'Loss' if use_loss_weighting else 'Uniform'
            return

        # --- Normalize Cost Matrix ---
        if normalize_cost and np.isfinite(max_cost) and max_cost > self.eps_num:
            normalized_cost_matrix = cost_matrix / max_cost
        else:
            if normalize_cost and not np.isfinite(max_cost):
                 warnings.warn("Max cost is not finite, cannot normalize cost matrix.")
            normalized_cost_matrix = cost_matrix # Use unnormalized
        self.cost_matrices['feature_error_ot'] = normalized_cost_matrix.cpu().numpy() # Store numpy version

        # --- Define Marginal Weights ---
        weighting_type_str = ""
        if use_loss_weighting and w1 is not None and w2 is not None:
            weights1_np = w1.cpu().numpy().astype(np.float64)
            weights2_np = w2.cpu().numpy().astype(np.float64)
            weighting_type_str = "Loss-Weighted"
            # Renormalize marginals
            sum1 = weights1_np.sum(); sum2 = weights2_np.sum()
            if not np.isclose(sum1, 1.0) and sum1 > self.eps_num: weights1_np /= sum1
            elif sum1 <= self.eps_num: weights1_np = np.ones_like(weights1_np) / N; warnings.warn("Client 1 loss weights sum zero, using uniform.")
            if not np.isclose(sum2, 1.0) and sum2 > self.eps_num: weights2_np /= sum2
            elif sum2 <= self.eps_num: weights2_np = np.ones_like(weights2_np) / M; warnings.warn("Client 2 loss weights sum zero, using uniform.")
            a, b = weights1_np, weights2_np
        else:
            if use_loss_weighting: warnings.warn("Loss weighting requested but weights unavailable. Using uniform.")
            a = np.ones(N, dtype=np.float64) / N
            b = np.ones(M, dtype=np.float64) / M
            weighting_type_str = "Uniform"
        self.results['weighting_used'] = weighting_type_str

        # --- Compute OT Cost ---
        ot_cost, Gs = compute_ot_cost(
            normalized_cost_matrix, a=a, b=b, reg=reg, sinkhorn_max_iter=max_iter, eps_num=self.eps_num
        )

        self.results['ot_cost'] = ot_cost
        self.results['transport_plan'] = Gs
        if verbose: print(f"  Feature-Error OT Cost ({weighting_type_str} Marginals, NormalizedCost={normalize_cost}): {f'{ot_cost:.4f}' if not np.isnan(ot_cost) else 'Failed'}")


    def _calculate_cost_feature_error_additive(self, h1, p1_prob, y1, h2, p2_prob, y2, **params) -> Tuple[Optional[torch.Tensor], float]:
        """ Calculates the additive cost: C = alpha * D_h_scaled + beta * D_e_scaled. """
        alpha = params.get('alpha', 1.0)
        beta = params.get('beta', 1.0)
        norm_eps = params.get('norm_eps', self.eps_num) # Use class eps if not provided
        device = 'cpu' # Calculations forced to CPU

        N, M = h1.shape[0], h2.shape[0]
        if N == 0 or M == 0:
            return torch.empty((N, M), device=device, dtype=torch.float), 0.0

        # Validate shapes (already preprocessed, but double check)
        if p1_prob.shape != (N, self.num_classes) or p2_prob.shape != (M, self.num_classes):
             warnings.warn(f"Internal Cost Calc: Probability shape mismatch: P1({p1_prob.shape}), P2({p2_prob.shape}), K={self.num_classes}")
             return None, np.nan

        # --- Scaled Feature Distance ---
        term1 = torch.zeros((N, M), device=device, dtype=torch.float)
        max_term1_contrib = 0.0
        if alpha > self.eps_num:
            try:
                h1_norm = F.normalize(h1.float(), p=2, dim=1, eps=norm_eps)
                h2_norm = F.normalize(h2.float(), p=2, dim=1, eps=norm_eps)
                D_h = torch.cdist(h1_norm, h2_norm, p=2) # Range [0, 2]
                D_h_scaled = D_h / 2.0 # Scale to [0, 1]
                term1 = alpha * D_h_scaled
                max_term1_contrib = alpha * 1.0
            except Exception as e:
                 warnings.warn(f"Feature distance calculation failed: {e}")
                 return None, np.nan

        # --- Scaled Raw Error Distance ---
        term2 = torch.zeros((N, M), device=device, dtype=torch.float)
        max_term2_contrib = 0.0
        if beta > self.eps_num:
            try:
                y1_oh = F.one_hot(y1.view(-1), num_classes=self.num_classes).float().to(device)
                y2_oh = F.one_hot(y2.view(-1), num_classes=self.num_classes).float().to(device)
                e1 = p1_prob.float() - y1_oh
                e2 = p2_prob.float() - y2_oh
                D_e_raw = torch.cdist(e1, e2, p=2)
                max_raw_error_dist = 2.0 * np.sqrt(2.0) # Theoretical max L2 dist between two vectors in [-1, 1]^K space
                D_e_scaled = D_e_raw / max_raw_error_dist
                term2 = beta * D_e_scaled
                max_term2_contrib = beta * 1.0
            except Exception as e:
                warnings.warn(f"Error distance calculation failed: {e}")
                return None, np.nan

        # --- Combine Terms ---
        cost_matrix = term1 + term2
        max_possible_cost = max_term1_contrib + max_term2_contrib
        effective_max_cost = max(self.eps_num, max_possible_cost)
        cost_matrix = torch.clamp(cost_matrix, min=0.0, max=effective_max_cost)

        return cost_matrix, float(max_possible_cost)

class DecomposedOTCalculator(BaseOTCalculator):
    """ Calculates Decomposed Similarity: Label EMD + Aggregated Conditional OT. """
    # ... (Implementation remains the same as before) ...
    def _reset_results(self) -> None:
        self.results = {
            'label_emd': np.nan,
            'conditional_ot_agg': np.nan,
            'combined_score': np.nan,
            'conditional_ot_per_class': {}, # Store individual class results here
            'aggregation_method_used': None,
            'aggregation_weights': None
        }
        # Store cost matrices per class if needed
        self.cost_matrices = {'within_class': {}}

    def calculate_similarity(self, data1: Dict[str, Optional[torch.Tensor]], data2: Dict[str, Optional[torch.Tensor]], params: Dict[str, Any]) -> None:
        self._reset_results()
        verbose = params.get('verbose', False)
        normalize_emd_flag = params.get('normalize_emd', True)
        normalize_within_cost = params.get('normalize_cost', True)
        ot_reg = params.get('ot_eps', DEFAULT_OT_REG)
        ot_max_iter = params.get('ot_max_iter', DEFAULT_OT_MAX_ITER)
        agg_method_param = params.get('aggregate_conditional_method', 'mean')
        min_class_samples = params.get('min_class_samples', 0) # For upsampling
        beta_within = params.get('beta_within', 0.0) # Check if probs needed for within-cost


        # --- Preprocess and Validate ---
        # Need h, y, weights always. Need p_prob if aggregating by loss or beta_within > 0.
        required_keys = ['h', 'y', 'weights']
        prob_needed = beta_within > self.eps_num or agg_method_param in ['avg_loss', 'total_loss_share']
        if prob_needed:
            required_keys.append('p_prob')
            # Note: 'loss' is derived from p_prob and y, handled after preprocessing

        proc_data1 = self._preprocess_input(data1.get('h'), data1.get('p_prob'), data1.get('y'), data1.get('weights'), required_keys)
        proc_data2 = self._preprocess_input(data2.get('h'), data2.get('p_prob'), data2.get('y'), data2.get('weights'), required_keys)

        if proc_data1 is None or proc_data2 is None:
            warnings.warn(f"Decomposed OT requires {required_keys}. Preprocessing failed or data missing. Skipping.")
            return

        h1, y1, w1 = proc_data1['h'], proc_data1['y'], proc_data1['weights']
        h2, y2, w2 = proc_data2['h'], proc_data2['y'], proc_data2['weights']
        p1_prob = proc_data1.get('p_prob')
        p2_prob = proc_data2.get('p_prob')
        N, M = h1.shape[0], h2.shape[0]

        if N == 0 or M == 0:
            warnings.warn("Decomposed OT: One or both clients have zero samples. Skipping.")
            return

        # Calculate loss if needed for aggregation (can move to DataManager preprocessing later)
        loss1, loss2 = None, None
        can_use_loss_agg = False
        if agg_method_param in ['avg_loss', 'total_loss_share']:
             if p1_prob is not None and p2_prob is not None:
                 loss1 = calculate_sample_loss(p1_prob, y1, self.num_classes, self.eps_num)
                 loss2 = calculate_sample_loss(p2_prob, y2, self.num_classes, self.eps_num)
                 if loss1 is not None and loss2 is not None and torch.isfinite(loss1).all() and torch.isfinite(loss2).all():
                     can_use_loss_agg = True
                 else:
                     warnings.warn(f"Cannot use loss aggregation: Loss calculation failed or resulted in NaN/Inf.")
             else:
                 warnings.warn(f"Cannot use loss aggregation: Probabilities (p_prob) missing.")

        # --- 1. Label EMD ---
        label_emd_raw = calculate_label_emd(y1, y2, self.num_classes)
        if np.isnan(label_emd_raw):
            warnings.warn("Label EMD failed. Decomposed OT aborted."); return
        emd_norm_factor = max(1.0, float(self.num_classes - 1)) if self.num_classes > 1 else 1.0
        label_emd_normalized = label_emd_raw / emd_norm_factor if normalize_emd_flag else label_emd_raw
        self.results['label_emd'] = label_emd_normalized

        # --- 2. Class-Conditional OT ---
        all_class_results = {} # Stores {'ot_cost': float, 'avg_loss': float, 'total_loss': float, 'sample_count': tuple, 'valid': bool}
        total_loss_across_classes = 0.0
        if can_use_loss_agg and loss1 is not None and loss2 is not None:
             total_loss_across_classes = (loss1.sum() + loss2.sum()).item()

        for c in range(self.num_classes):
            class_result = {'ot_cost': np.nan, 'avg_loss': 0.0, 'total_loss': 0.0, 'sample_count': (0, 0), 'valid': False}
            idx1_c = torch.where(y1 == c)[0]; idx2_c = torch.where(y2 == c)[0]
            n_c, m_c = idx1_c.shape[0], idx2_c.shape[0]
            class_result['sample_count'] = (n_c, m_c)

            if n_c == 0 or m_c == 0:
                all_class_results[c] = class_result; continue # Skip if no samples in either client

            # --- Gather class data & handle loss ---
            h1_c, w1_c, y1_c = h1[idx1_c], w1[idx1_c], y1[idx1_c]
            h2_c, w2_c, y2_c = h2[idx2_c], w2[idx2_c], y2[idx2_c]
            p1_prob_c = p1_prob[idx1_c] if p1_prob is not None else None
            p2_prob_c = p2_prob[idx2_c] if p2_prob is not None else None

            if can_use_loss_agg and loss1 is not None and loss2 is not None:
                loss1_c = loss1[idx1_c]; loss2_c = loss2[idx2_c]
                if torch.isfinite(loss1_c).all() and torch.isfinite(loss2_c).all():
                    total_loss_c = (loss1_c.sum() + loss2_c.sum()).item()
                    avg_loss_c = total_loss_c / (n_c + m_c) if (n_c + m_c) > 0 else 0.0
                    class_result['total_loss'] = total_loss_c; class_result['avg_loss'] = avg_loss_c

            # Check if probs are needed for cost and available
            if beta_within > self.eps_num and (p1_prob_c is None or p2_prob_c is None):
                 if verbose: warnings.warn(f"Class {c}: Skipped conditional OT - probs needed for cost (beta_within>0) but unavailable.");
                 all_class_results[c] = class_result; continue

            # --- Upsampling (Optional) ---
            # (Simplified logic from original, ensure random state is handled if needed)
            h1_c_final, w1_c_final, y1_c_final = h1_c, w1_c, y1_c
            h2_c_final, w2_c_final, y2_c_final = h2_c, w2_c, y2_c
            p1_prob_c_final, p2_prob_c_final = p1_prob_c, p2_prob_c
            n_c_final, m_c_final = n_c, m_c
            if min_class_samples > 0:
               if n_c_final < min_class_samples:
                   if n_c > 0: # Can only upsample if exists
                       needed1 = min_class_samples - n_c
                       resample_indices1 = np.random.choice(n_c, size=needed1, replace=True)
                       h1_c_final = torch.cat((h1_c, h1_c[resample_indices1]), dim=0)
                       w1_c_final = torch.cat((w1_c, w1_c[resample_indices1]), dim=0)
                       y1_c_final = torch.cat((y1_c, y1_c[resample_indices1]), dim=0)
                       if p1_prob_c is not None: p1_prob_c_final = torch.cat((p1_prob_c, p1_prob_c[resample_indices1]), dim=0)
                       n_c_final = min_class_samples
                   else: # Cannot upsample from zero
                       if verbose: warnings.warn(f"C{c}: C1 has 0 samples, cannot upsample to {min_class_samples}.")
                       all_class_results[c] = class_result; continue # Skip class if cannot meet min samples
               if m_c_final < min_class_samples:
                   if m_c > 0:
                       needed2 = min_class_samples - m_c
                       resample_indices2 = np.random.choice(m_c, size=needed2, replace=True)
                       h2_c_final = torch.cat((h2_c, h2_c[resample_indices2]), dim=0)
                       w2_c_final = torch.cat((w2_c, w2_c[resample_indices2]), dim=0)
                       y2_c_final = torch.cat((y2_c, y2_c[resample_indices2]), dim=0)
                       if p2_prob_c is not None: p2_prob_c_final = torch.cat((p2_prob_c, p2_prob_c[resample_indices2]), dim=0)
                       m_c_final = min_class_samples
                   else:
                       if verbose: warnings.warn(f"C{c}: C2 has 0 samples, cannot upsample to {min_class_samples}.")
                       all_class_results[c] = class_result; continue # Skip class

            # --- Calculate Within-Class Cost Matrix ---
            cost_matrix_c, max_cost_c = self._calculate_cost_within_class(
                h1_c_final, p1_prob_c_final, y1_c_final, h2_c_final, p2_prob_c_final, y2_c_final, **params
            )
            if cost_matrix_c is None or not (np.isfinite(max_cost_c) and max_cost_c > self.eps_num):
                fail_reason = "within-class cost calc failed" if cost_matrix_c is None else "within-class max cost near zero/invalid"
                if verbose: warnings.warn(f"Class {c}: Skipped conditional OT - {fail_reason}.")
                all_class_results[c] = class_result; continue

            # --- Normalize Cost & Compute OT ---
            if normalize_within_cost and np.isfinite(max_cost_c) and max_cost_c > self.eps_num:
                 norm_cost_matrix_c = cost_matrix_c / max_cost_c
            else: norm_cost_matrix_c = cost_matrix_c
            self.cost_matrices['within_class'][c] = norm_cost_matrix_c.cpu().numpy()

            # Use weights for marginals, fallback to uniform
            w1_c_np = w1_c_final.cpu().numpy().astype(np.float64); sum1_c = w1_c_np.sum()
            w2_c_np = w2_c_final.cpu().numpy().astype(np.float64); sum2_c = w2_c_np.sum()
            a_c = (w1_c_np / sum1_c) if sum1_c > self.eps_num else (np.ones(n_c_final)/max(1,n_c_final))
            b_c = (w2_c_np / sum2_c) if sum2_c > self.eps_num else (np.ones(m_c_final)/max(1,m_c_final))
            if sum1_c <= self.eps_num: warnings.warn(f"C{c}: C1 weights sum zero.")
            if sum2_c <= self.eps_num: warnings.warn(f"C{c}: C2 weights sum zero.")

            ot_cost_c, _ = compute_ot_cost(norm_cost_matrix_c, a=a_c, b=b_c, reg=ot_reg, sinkhorn_max_iter=ot_max_iter, eps_num=self.eps_num)

            if not np.isnan(ot_cost_c):
                class_result['ot_cost'] = ot_cost_c
                class_result['valid'] = True
                self.results['conditional_ot_per_class'][c] = ot_cost_c # Store individual result
            elif verbose:
                warnings.warn(f"Conditional OT cost calculation failed for class {c}.")

            all_class_results[c] = class_result

        # --- 3. Aggregate Conditional OT Costs ---
        valid_classes = [c for c, res in all_class_results.items() if res['valid']]
        agg_conditional_ot = np.nan
        agg_weights_dict = {} # Store weights used for aggregation
        agg_method_used_str = "None"

        if valid_classes:
            costs_agg = np.array([all_class_results[c]['ot_cost'] for c in valid_classes], dtype=np.float64)
            weights_agg = np.ones_like(costs_agg) # Default to mean
            effective_agg_method = agg_method_param

            # Check feasibility of loss aggregation
            if effective_agg_method in ['avg_loss', 'total_loss_share'] and not can_use_loss_agg:
                warnings.warn(f"Cannot use '{agg_method_param}' due to invalid losses. Falling back to 'mean'.")
                effective_agg_method = 'mean'

            agg_method_used_str = f"{effective_agg_method}"

            # Calculate weights based on method
            if effective_agg_method == 'mean':
                weights_agg = np.ones_like(costs_agg)
            elif effective_agg_method == 'class_share':
                total_samples = N + M
                if total_samples > 0:
                    weights_agg_list = [(all_class_results[c]['sample_count'][0] + all_class_results[c]['sample_count'][1]) / total_samples for c in valid_classes]
                    weights_agg = np.array(weights_agg_list, dtype=np.float64)
                else: weights_agg = np.ones_like(costs_agg); agg_method_used_str += " (Fallback - Zero Total Samples)"
            elif effective_agg_method == 'avg_loss':
                weights_agg = np.array([all_class_results[c]['avg_loss'] for c in valid_classes], dtype=np.float64)
            elif effective_agg_method == 'total_loss_share':
                if total_loss_across_classes > self.eps_num:
                     weights_agg = np.array([all_class_results[c]['total_loss'] / total_loss_across_classes for c in valid_classes], dtype=np.float64)
                else: weights_agg = np.ones_like(costs_agg); agg_method_used_str += " (Fallback - Zero Total Loss)"

            # Normalize weights and compute weighted average
            weights_agg = np.maximum(weights_agg, 0) # Ensure non-negative
            total_weight = weights_agg.sum()
            if total_weight < self.eps_num:
                agg_conditional_ot = np.mean(costs_agg) if costs_agg.size > 0 else np.nan
                agg_method_used_str += " (Fallback - Zero Weight Sum)"
                normalized_weights_agg = np.ones_like(costs_agg) / max(1, len(costs_agg)) # Uniform weights
            else:
                normalized_weights_agg = weights_agg / total_weight
                agg_conditional_ot = np.sum(normalized_weights_agg * costs_agg)

            agg_weights_dict = {c: w for c, w in zip(valid_classes, normalized_weights_agg)}
        else:
             warnings.warn("No valid conditional OT costs calculated to aggregate.")

        self.results['conditional_ot_agg'] = agg_conditional_ot
        self.results['aggregation_method_used'] = agg_method_used_str
        self.results['aggregation_weights'] = agg_weights_dict

        # --- 4. Calculate Combined Score ---
        if not np.isnan(label_emd_normalized) and not np.isnan(agg_conditional_ot):
             # Simple average, consider weighting later if needed
             self.results['combined_score'] = (label_emd_normalized + agg_conditional_ot) / 2.0
        else: self.results['combined_score'] = np.nan

        if verbose:
            print("-" * 30)
            print("--- Decomposed Similarity Results ---")
            emd_str = f"{self.results['label_emd']:.3f}" if not np.isnan(self.results['label_emd']) else 'Failed'
            print(f"  Label EMD (Normalized={normalize_emd_flag}): {emd_str}")
            print(f"  Conditional Within-Class OT Costs (NormCost={normalize_within_cost}):")
            for c in range(self.num_classes):
                 cost_val = self.results['conditional_ot_per_class'].get(c, np.nan)
                 cost_str = f"{cost_val:.3f}" if not np.isnan(cost_val) else 'Skipped/Failed'
                 print(f"     - Class {c}: Cost={cost_str}")
            cond_ot_agg_str = f"{self.results['conditional_ot_agg']:.3f}" if not np.isnan(self.results['conditional_ot_agg']) else 'Failed'
            print(f"  Aggregated Conditional OT ({agg_method_used_str}): {cond_ot_agg_str}")
            comb_score_str = f"{self.results['combined_score']:.3f}" if not np.isnan(self.results['combined_score']) else 'Invalid'
            print(f"  Combined Score (/2): {comb_score_str}")
            print("-" * 30)


    def _calculate_cost_within_class(self, h1_c, p1_prob_c, y1_c, h2_c, p2_prob_c, y2_c, **params) -> Tuple[Optional[torch.Tensor], float]:
        """ Calculates within-class cost: alpha * D_h_scaled + beta * D_e_scaled. """
        alpha_within = params.get('alpha_within', 1.0)
        beta_within = params.get('beta_within', 0.0)
        norm_eps = params.get('eps', self.eps_num) # Use class eps if not provided
        device = 'cpu'

        # Basic check (inputs are already tensors from calculate_similarity)
        if h1_c is None or y1_c is None or h2_c is None or y2_c is None: return None, np.nan
        prob_needed = beta_within > self.eps_num
        if prob_needed and (p1_prob_c is None or p2_prob_c is None): return None, np.nan

        N, M = h1_c.shape[0], h2_c.shape[0]
        if N == 0 or M == 0: return torch.empty((N, M), device=device, dtype=torch.float), 0.0

        # --- Feature Distance ---
        term1 = torch.zeros((N, M), device=device, dtype=torch.float)
        max_term1_contrib = 0.0
        if alpha_within > self.eps_num:
            try:
                h1_norm = F.normalize(h1_c.float(), p=2, dim=1, eps=norm_eps)
                h2_norm = F.normalize(h2_c.float(), p=2, dim=1, eps=norm_eps)
                D_h = torch.cdist(h1_norm, h2_norm, p=2)
                D_h_scaled = D_h / 2.0
                term1 = alpha_within * D_h_scaled
                max_term1_contrib = alpha_within * 1.0
            except Exception as e: warnings.warn(f"Within-class feature dist failed: {e}"); return None, np.nan

        # --- Error Distance ---
        term2 = torch.zeros((N, M), device=device, dtype=torch.float)
        max_term2_contrib = 0.0
        if prob_needed and p1_prob_c is not None and p2_prob_c is not None:
             # Validate prob shapes
             if p1_prob_c.shape != (N, self.num_classes) or p2_prob_c.shape != (M, self.num_classes):
                  warnings.warn(f"Within-class Cost: Prob shape mismatch P1({p1_prob_c.shape}), P2({p2_prob_c.shape})")
                  return None, np.nan
             try:
                y1_oh = F.one_hot(y1_c.view(-1), num_classes=self.num_classes).float()
                y2_oh = F.one_hot(y2_c.view(-1), num_classes=self.num_classes).float()
                e1 = p1_prob_c.float() - y1_oh
                e2 = p2_prob_c.float() - y2_oh
                D_e_raw = torch.cdist(e1, e2, p=2)
                max_raw_error_dist = 2.0 * np.sqrt(2.0)
                D_e_scaled = D_e_raw / max_raw_error_dist
                term2 = beta_within * D_e_scaled
                max_term2_contrib = beta_within * 1.0
             except Exception as e: warnings.warn(f"Within-class error dist failed: {e}"); return None, np.nan

        # --- Combine ---
        cost_matrix_c = term1 + term2
        max_possible_cost_c = max_term1_contrib + max_term2_contrib
        effective_max_cost_c = max(self.eps_num, max_possible_cost_c)
        cost_matrix_c = torch.clamp(cost_matrix_c, min=0.0, max=effective_max_cost_c)

        return cost_matrix_c, float(max_possible_cost_c)


class FixedAnchorLOTCalculator(BaseOTCalculator):
    """ Calculates Fixed-Anchor LOT cost and similarity score. """
    # ... (Implementation remains the same as before) ...
    def _reset_results(self) -> None:
        self.results = {
            'total_cost': np.nan,
            'cost_local1': np.nan,
            'cost_local2': np.nan,
            'cost_cross_anchor': np.nan,
            'similarity_score': np.nan,
            'params': None # Store params used for this calc
        }
        self.cost_matrices = {'local1': None, 'local2': None, 'cross_anchor': None}

    def calculate_similarity(self, data1: Dict[str, Optional[torch.Tensor]], data2: Dict[str, Optional[torch.Tensor]], params: Dict[str, Any]) -> None:
        self._reset_results()
        self.results['params'] = params.copy() # Store params
        verbose = params.get('verbose', False)

        # --- Preprocess and Validate ---
        # Requires h, y for anchor computation
        required_keys = ['h', 'y']
        proc_data1 = self._preprocess_input(data1.get('h'), None, data1.get('y'), None, required_keys)
        proc_data2 = self._preprocess_input(data2.get('h'), None, data2.get('y'), None, required_keys)

        if proc_data1 is None or proc_data2 is None:
            warnings.warn("Fixed-Anchor LOT requires 'h' and 'y'. Preprocessing failed or data missing. Skipping.")
            return

        h1, y1 = proc_data1['h'], proc_data1['y']
        h2, y2 = proc_data2['h'], proc_data2['y']
        N, M = h1.shape[0], h2.shape[0]

        if N == 0 or M == 0:
            warnings.warn("Fixed-Anchor LOT: One or both clients have zero samples. Setting costs to 0, similarity to NaN.")
            self.results.update({'total_cost': 0.0, 'cost_local1': 0.0, 'cost_local2': 0.0, 'cost_cross_anchor': 0.0, 'similarity_score': np.nan})
            return

        # --- Step 1: Compute Anchors ---
        Z1 = compute_anchors(h1, y1, self.num_classes, params)
        Z2 = compute_anchors(h2, y2, self.num_classes, params)

        if Z1 is None or Z2 is None or Z1.shape[0] == 0 or Z2.shape[0] == 0:
            warnings.warn("Failed to compute valid anchors for one or both clients. Skipping Fixed-Anchor LOT.")
            return

        k1, k2 = Z1.shape[0], Z2.shape[0]

        # --- Step 2 & 3: Compute OT Costs ---
        ot_max_iter = params.get('ot_max_iter', DEFAULT_OT_MAX_ITER)
        local_reg = params.get('local_ot_reg', 0.01)
        cross_reg = params.get('cross_anchor_ot_reg', 0.01)
        cost_local1, cost_local2, cost_cross_anchor = np.nan, np.nan, np.nan

        # Cost Local 1 (h1 -> Z1)
        cost_matrix_local1 = pairwise_euclidean_sq(h1, Z1)
        if cost_matrix_local1 is not None:
            a1 = np.ones(N, dtype=np.float64) / N
            b1 = np.ones(k1, dtype=np.float64) / k1
            cost_local1, _ = compute_ot_cost(cost_matrix_local1, a=a1, b=b1, reg=local_reg, sinkhorn_max_iter=ot_max_iter, eps_num=self.eps_num)
            self.cost_matrices['local1'] = cost_matrix_local1.cpu().numpy()
        else: warnings.warn("Failed to compute cost matrix for Local1 OT.")

        # Cost Local 2 (h2 -> Z2)
        cost_matrix_local2 = pairwise_euclidean_sq(h2, Z2)
        if cost_matrix_local2 is not None:
            a2 = np.ones(M, dtype=np.float64) / M
            b2 = np.ones(k2, dtype=np.float64) / k2
            cost_local2, _ = compute_ot_cost(cost_matrix_local2, a=a2, b=b2, reg=local_reg, sinkhorn_max_iter=ot_max_iter, eps_num=self.eps_num)
            self.cost_matrices['local2'] = cost_matrix_local2.cpu().numpy()
        else: warnings.warn("Failed to compute cost matrix for Local2 OT.")

        # Cost Cross Anchor (Z1 -> Z2)
        cost_matrix_cross_anchor = pairwise_euclidean_sq(Z1, Z2)
        if cost_matrix_cross_anchor is not None:
            aZ = np.ones(k1, dtype=np.float64) / k1
            bZ = np.ones(k2, dtype=np.float64) / k2
            cost_cross_anchor, _ = compute_ot_cost(cost_matrix_cross_anchor, a=aZ, b=bZ, reg=cross_reg, sinkhorn_max_iter=ot_max_iter, eps_num=self.eps_num)
            self.cost_matrices['cross_anchor'] = cost_matrix_cross_anchor.cpu().numpy()
        else: warnings.warn("Failed to compute cost matrix for Cross-Anchor OT.")

        self.results['cost_local1'] = cost_local1
        self.results['cost_local2'] = cost_local2
        self.results['cost_cross_anchor'] = cost_cross_anchor

        # --- Step 4: Aggregate Costs ---
        total_cost = np.nan
        if np.isfinite(cost_local1) and np.isfinite(cost_local2) and np.isfinite(cost_cross_anchor):
            w_local = params.get('local_cost_weight', 0.25) # Weight applied to BOTH local terms
            w_cross = params.get('cross_anchor_cost_weight', 0.5)
            # Normalize weights if they don't sum to 1? Or assume user provides intended weights.
            # Original code implies w_local is for EACH local term, so total = w_local*c1 + w_local*c2 + w_cross*cx
            total_cost = w_local * cost_local1 + w_local * cost_local2 + w_cross * cost_cross_anchor
        else:
            warnings.warn("One or more Fixed-Anchor OT cost components failed (NaN/Inf). Cannot compute total cost.")
        self.results['total_cost'] = total_cost

        # --- Step 5: Normalization / Similarity Score ---
        similarity_score = np.nan
        if np.isfinite(total_cost):
            if params.get('normalize_total_cost', True):
                max_cost_estimate = params.get('max_total_cost_estimate', 1.0) # Needs careful tuning
                if max_cost_estimate > self.eps_num:
                     # Ensure similarity is between 0 and 1 (or slightly outside due to estimation)
                    similarity_score = 1.0 - (total_cost / max_cost_estimate)
                    # Clamp if needed: similarity_score = max(0.0, 1.0 - (total_cost / max_cost_estimate))
                else:
                    warnings.warn(f"max_total_cost_estimate ({max_cost_estimate}) is too small for normalization. Similarity set to NaN.")
            else:
                 similarity_score = -total_cost # Use negative cost if not normalizing (lower cost is better)

        self.results['similarity_score'] = similarity_score

        if verbose:
            print(f"  ---> Fixed-Anchor LOT Total Cost: {f'{total_cost:.4f}' if np.isfinite(total_cost) else 'Failed'}")
            print(f"       (Local1: {cost_local1:.4f}, Local2: {cost_local2:.4f}, Cross: {cost_cross_anchor:.4f})")
            print(f"  ---> Fixed-Anchor LOT Similarity: {f'{similarity_score:.4f}' if np.isfinite(similarity_score) else 'Failed'}")