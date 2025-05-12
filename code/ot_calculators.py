import torch
import torch.nn.functional as F
import numpy as np
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, Type, List, Union # Added Type

# Import OTConfig from ot_configs
from ot_configs import OTConfig

# Import utilities from ot_utils.py
from ot_utils import (
    compute_ot_cost, pairwise_euclidean_sq, calculate_label_emd,
    validate_samples_for_ot, validate_samples_for_decomposed_ot,
    DEFAULT_OT_REG, DEFAULT_OT_MAX_ITER, DEFAULT_EPS, calculate_sample_loss, 
    prepare_ot_marginals, normalize_cost_matrix
)

# Configure module logger
logger = logging.getLogger(__name__)

class OTCalculatorFactory:
    """ Factory class for creating OT calculator instances. """

    @classmethod
    def _get_calculator_map(cls) -> Dict[str, Type['BaseOTCalculator']]:
        """ Internal method to access the map, allowing definition after classes. """
        # Defined below BaseOTCalculator and its subclasses
        return {
            'feature_error': FeatureErrorOTCalculator,
            'decomposed': DecomposedOTCalculator,
            'direct_ot': DirectOTCalculator,
            # Register new calculator classes here
        }

    @classmethod
    def register_calculator(cls, method_type: str, calculator_class: Type['BaseOTCalculator']):
        """ Allows dynamic registration (though map is currently static). """
        # This method might be less useful if the map is defined statically,
        # but kept for potential future dynamic loading.
        if not issubclass(calculator_class, BaseOTCalculator):
             raise TypeError(f"{calculator_class.__name__} must inherit from BaseOTCalculator")
        logger.info(f"Note: Dynamic registration not fully implemented with static map. Add '{method_type}' to _get_calculator_map.")


    @staticmethod
    def create_calculator(config: OTConfig, client_id_1: str, client_id_2: str, num_classes: int) -> Optional['BaseOTCalculator']:
        """
        Creates an instance of the appropriate OT calculator based on the config.
        
        Args:
            config: OTConfig object with method_type and parameters
            client_id_1: First client identifier
            client_id_2: Second client identifier
            num_classes: Number of classes in the dataset
            
        Returns:
            An instance of the appropriate calculator or None if creation fails
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
                logger.warning(f"Failed to instantiate calculator for config '{config.name}' (type: {config.method_type}): {e}")
                return None
        else:
            logger.warning(f"No calculator registered for method type '{config.method_type}' in config '{config.name}'. Skipping.")
            return None

# --- Base OT Calculator Class ---
class BaseOTCalculator(ABC):
    """
    Abstract Base Class for Optimal Transport similarity calculators.
    """
    def __init__(self, client_id_1: str, client_id_2: str, num_classes: int, **kwargs):
        if num_classes < 2: # num_classes can be 1 for regression/binary tasks if p_prob becomes [N] or [N,1]
            # For OT, typically comparing distributions, num_classes >= 2 is more common for label-based aspects.
            # If a calculator handles num_classes=1 specifically (e.g. pure feature OT), it can override or manage.
            # Keeping this check for general safety with label-aware methods.
            pass # No longer raising error, individual calculators must handle num_classes=1 if applicable.

        self.client_id_1 = str(client_id_1)
        self.client_id_2 = str(client_id_2)
        self.num_classes = num_classes
        self.eps_num = kwargs.get('eps_num', DEFAULT_EPS) # Epsilon for numerical stability

        self.results: Dict[str, Any] = {}
        self.cost_matrices: Dict[str, Any] = {}
        self._reset_results() # Initialize result structures

    @abstractmethod
    def calculate_similarity(self, data1: Dict[str, Optional[torch.Tensor]], data2: Dict[str, Optional[torch.Tensor]], params: Dict[str, Any]) -> None:
        """ Calculates the specific OT similarity metric. Must be implemented by subclasses. """
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

    def _preprocess_input(
        self, 
        data_client1: Dict[str, Optional[torch.Tensor]], 
        data_client2: Dict[str, Optional[torch.Tensor]], 
        required_keys: List[str]
    ) -> Optional[Tuple[Dict[str, Optional[torch.Tensor]], Dict[str, Optional[torch.Tensor]]]]:
        """
        Basic preprocessing for input data dictionaries from OTDataManager.
        Ensures required keys are present and tensors are on CPU.
        Relies on OTDataManager._process_client_data for detailed validation
        (e.g., probability formatting, loss calculation, weight generation).

        Args:
            data_client1: Processed data dictionary for client 1.
            data_client2: Processed data dictionary for client 2.
            required_keys: List of keys that must be non-None in both dictionaries.

        Returns:
            A tuple of (processed_data_client1, processed_data_client2) or None if validation fails.
        """
        if not isinstance(data_client1, dict) or not isinstance(data_client2, dict):
            logger.warning("Calculator _preprocess_input: Inputs must be dictionaries.")
            return None

        processed_data_c1 = {}
        processed_data_c2 = {}

        for client_idx, (data_in, data_out) in enumerate([(data_client1, processed_data_c1), 
                                                          (data_client2, processed_data_c2)]):
            client_name = self.client_id_1 if client_idx == 0 else self.client_id_2
            for key in data_in: # Iterate over all keys present in the input dict
                tensor = data_in.get(key)
                if tensor is None:
                    if key in required_keys:
                        logger.warning(f"Client {client_name}: Required input '{key}' is None.")
                        return None
                    data_out[key] = None
                    continue

                if isinstance(tensor, torch.Tensor):
                    data_out[key] = tensor.detach().cpu()
                else:
                    # This case should ideally not happen if OTDataManager prepares tensors
                    try:
                        data_out[key] = torch.tensor(tensor).cpu() 
                    except Exception as e:
                        logger.warning(f"Client {client_name}: Could not convert input '{key}' to CPU tensor: {e}")
                        return None
            
            # Final check for required keys after processing all available keys
            missing_keys = [k_req for k_req in required_keys if k_req not in data_out or data_out[k_req] is None]
            if missing_keys:
                logger.warning(f"Client {client_name}: Missing required keys {missing_keys} after processing.")
                return None
        
        # Basic check: Ensure 'h' features have same dimension if both present
        h1 = processed_data_c1.get('h')
        h2 = processed_data_c2.get('h')
        if h1 is not None and h2 is not None:
            if h1.ndim < 2 or h2.ndim < 2 : # Expect at least [N, D]
                 logger.warning(f"Feature dimensions are less than 2D (h1: {h1.shape}, h2: {h2.shape}).")
                 return None
            if h1.shape[0] == 0 or h2.shape[0] == 0: # No samples
                 logger.info(f"One or both clients have zero samples (h1: {h1.shape[0]}, h2: {h2.shape[0]}).")
                 # This is not an error for _preprocess_input, calculators handle it.
            elif h1.shape[1] != h2.shape[1]:
                 logger.warning(f"Feature dimension mismatch: h1 dim {h1.shape[1]} vs h2 dim {h2.shape[1]}.")
                 return None

        return processed_data_c1, processed_data_c2

# --- Concrete Calculator Implementations ---

class FeatureErrorOTCalculator(BaseOTCalculator):
    """ Calculates OT cost using the additive feature-error cost. """
    
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
        min_samples_threshold = params.get('min_samples', 20)  # Get minimum samples threshold
        max_samples_threshold = params.get('max_samples', 900)

        # --- Check if this is a segmentation case (like IXITiny) ---
        is_segmentation = data1.get('p_prob') is None and data2.get('p_prob') is None
        
        # --- Preprocess and Validate ---
        required_keys = ['h']
        if not is_segmentation:
            required_keys.extend(['p_prob', 'y'])
            
        if use_loss_weighting and not is_segmentation: 
            required_keys.append('weights')

        proc_data1, proc_data2 = self._preprocess_input(data1, data2, required_keys)

        if proc_data1 is None or proc_data2 is None:
            missing_str = "'weights'" if use_loss_weighting and (data1.get('weights') is None or data2.get('weights') is None) else "h, p_prob, or y"
            if is_segmentation:
                missing_str = "activations ('h')"
            logger.warning(f"Feature-Error OT calculation requires {missing_str}. Preprocessing failed or data missing. Skipping.")
            self.results['weighting_used'] = 'Loss' if use_loss_weighting else 'Uniform' # Still record intent
            return

        h1 = proc_data1['h']
        h2 = proc_data2['h']
        
        # For segmentation data, set these to None
        p1_prob = proc_data1.get('p_prob')
        y1 = proc_data1.get('y')
        w1 = proc_data1.get('weights')
        p2_prob = proc_data2.get('p_prob')
        y2 = proc_data2.get('y')
        w2 = proc_data2.get('weights')
        
        N, M = h1.shape[0], h2.shape[0]

        # Validate sample counts and get sampling indices
        features_dict = {
            "client1": h1.cpu().numpy(),
            "client2": h2.cpu().numpy()
        }
        all_sufficient, sample_indices = validate_samples_for_ot(
            features_dict, min_samples_threshold, max_samples_threshold)

        if not all_sufficient:
            logger.warning(f"Feature-Error OT: One or both clients have insufficient samples (min={min_samples_threshold}). Skipping.")
            self.results['weighting_used'] = 'Loss' if use_loss_weighting else 'Uniform'
            return

        # Apply sampling if needed
        if "client1" in sample_indices and len(sample_indices["client1"]) < N:
            # Sample client 1 data
            indices1 = torch.from_numpy(sample_indices["client1"]).long()
            h1 = h1[indices1]
            if not is_segmentation:
                p1_prob = p1_prob[indices1] if p1_prob is not None else None
                y1 = y1[indices1] if y1 is not None else None
            w1 = w1[indices1] if w1 is not None else None
            N = len(indices1)
            if verbose:
                logger.info(f"Sampled client1 data: {len(h1)} from original {N}")

        if "client2" in sample_indices and len(sample_indices["client2"]) < M:
            # Sample client 2 data
            indices2 = torch.from_numpy(sample_indices["client2"]).long()
            h2 = h2[indices2]
            if not is_segmentation:
                p2_prob = p2_prob[indices2] if p2_prob is not None else None
                y2 = y2[indices2] if y2 is not None else None
            w2 = w2[indices2] if w2 is not None else None
            M = len(indices2)
            if verbose:
                logger.info(f"Sampled client2 data: {len(h2)} from original {M}")

        if N == 0 or M == 0:
            logger.warning("Feature-Error OT: One or both clients have zero samples. OT cost is 0.")
            self.results['ot_cost'] = 0.0
            self.results['transport_plan'] = np.zeros((N, M)) if N > 0 and M > 0 else None
            self.results['weighting_used'] = 'Loss' if use_loss_weighting else 'Uniform'
            return

        # --- Calculate Cost Matrix ---
        # For segmentation data, use only feature distance
        if is_segmentation:
            cost_matrix, max_cost = self._calculate_feature_distance_only(h1, h2, **params)
        else:
            cost_matrix, max_cost = self._calculate_cost_feature_error_additive(
                h1, p1_prob, y1, h2, p2_prob, y2, **params
            )
            
        self.results['cost_matrix_max_val'] = max_cost

        if cost_matrix is None or not (np.isfinite(max_cost) and max_cost > self.eps_num):
            fail_reason = "Cost matrix calculation failed" if cost_matrix is None else "Max cost near zero/invalid"
            logger.warning(f"Feature-Error OT calculation skipped: {fail_reason}.")
            self.results['weighting_used'] = 'Loss' if use_loss_weighting else 'Uniform'
            return

        # --- Normalize Cost Matrix ---
        normalized_cost_matrix = normalize_cost_matrix(cost_matrix, max_cost, normalize_cost, self.eps_num)
        self.cost_matrices['feature_error_ot'] = normalized_cost_matrix.cpu().numpy() # Store numpy version

        # --- Define Marginal Weights ---
        weighting_type_str = ""
        if use_loss_weighting and w1 is not None and w2 is not None:
            a, b = prepare_ot_marginals(w1.cpu().numpy(), w2.cpu().numpy(), N, M, self.eps_num)
            weighting_type_str = "Loss-Weighted"
        else:
            if use_loss_weighting: 
                logger.warning("Loss weighting requested but weights unavailable. Using uniform.")
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
        if verbose: 
            logger.info(f"  Feature-Error OT Cost ({weighting_type_str} Marginals, NormalizedCost={normalize_cost}): {f'{ot_cost:.4f}' if not np.isnan(ot_cost) else 'Failed'}")

    def _calculate_feature_distance_only(self, h1, h2, **params) -> Tuple[Optional[torch.Tensor], float]:
        """
        Calculates feature distance for segmentation models (IXITiny) where only h is available.
        Uses L2-normalized Euclidean distance.
        
        Args:
            h1: Features from client 1
            h2: Features from client 2
            **params: Additional parameters
            
        Returns:
            Tuple of (cost_matrix, max_possible_cost)
        """
        norm_eps = params.get('norm_eps', self.eps_num)
        device = 'cpu'

        N, M = h1.shape[0], h2.shape[0]
        if N == 0 or M == 0:
            return torch.empty((N, M), device=device, dtype=torch.float), 0.0

        try:
            # L2 normalize feature vectors
            h1_norm = F.normalize(h1.float(), p=2, dim=1, eps=norm_eps)
            h2_norm = F.normalize(h2.float(), p=2, dim=1, eps=norm_eps)
            
            # Compute Euclidean distance (range [0, 2] for normalized vectors)
            D_h = torch.cdist(h1_norm, h2_norm, p=2)
            
            # Scale to [0, 1] range
            D_h_scaled = D_h / 2.0
            
            # Maximum possible value is 1.0 after scaling
            max_cost = 1.0
            
            return D_h_scaled, max_cost
            
        except Exception as e:
            logger.warning(f"Feature distance calculation failed: {e}")
            return None, np.nan
            
    def _calculate_cost_feature_error_additive(self, h1, p1_prob, y1, h2, p2_prob, y2, **params) -> Tuple[Optional[torch.Tensor], float]:
        """
        Calculates the additive cost: C = alpha * D_h_scaled + beta * D_e_scaled.
        Enhanced to better handle multiclass probabilities.
        
        Args:
            h1: Features from client 1
            p1_prob: Predicted probabilities from client 1
            y1: Labels from client 1
            h2: Features from client 2
            p2_prob: Predicted probabilities from client 2
            y2: Labels from client 2
            **params: Additional parameters including alpha, beta
            
        Returns:
            Tuple of (cost_matrix, max_possible_cost)
        """
        # If p_prob or y is None, revert to feature distance only
        if p1_prob is None or p2_prob is None or y1 is None or y2 is None:
            return self._calculate_feature_distance_only(h1, h2, **params)
            
        alpha = params.get('alpha', 1.0)
        beta = params.get('beta', 1.0)
        norm_eps = params.get('norm_eps', self.eps_num) # Use class eps if not provided
        device = 'cpu' # Calculations forced to CPU

        N, M = h1.shape[0], h2.shape[0]
        if N == 0 or M == 0:
            return torch.empty((N, M), device=device, dtype=torch.float), 0.0

        # Validate probability tensor shapes
        if p1_prob.shape[1] != self.num_classes or p2_prob.shape[1] != self.num_classes:
            logger.warning(f"Internal Cost Calc: Probability shape mismatch: "
                        f"P1({p1_prob.shape}), P2({p2_prob.shape}), K={self.num_classes}")
            return None, np.nan

        # --- Term 1: Scaled Feature Distance ---
        term1 = torch.zeros((N, M), device=device, dtype=torch.float)
        max_term1_contrib = 0.0
        if alpha > self.eps_num:
            try:
                # L2 normalize feature vectors
                h1_norm = F.normalize(h1.float(), p=2, dim=1, eps=norm_eps)
                h2_norm = F.normalize(h2.float(), p=2, dim=1, eps=norm_eps)
                
                # Compute Euclidean distance (range [0, 2] for normalized vectors)
                D_h = torch.cdist(h1_norm, h2_norm, p=2)
                
                # Scale to [0, 1] range and apply alpha weight
                D_h_scaled = D_h / 2.0
                term1 = alpha * D_h_scaled
                max_term1_contrib = alpha * 1.0
            except Exception as e:
                logger.warning(f"Feature distance calculation failed: {e}")
                return None, np.nan

        # --- Term 2: Scaled Error Distance ---
        term2 = torch.zeros((N, M), device=device, dtype=torch.float)
        max_term2_contrib = 0.0
        if beta > self.eps_num:
            try:
                # Create one-hot encoded ground truth
                y1_oh = F.one_hot(y1.view(-1), num_classes=self.num_classes).float().to(device)
                y2_oh = F.one_hot(y2.view(-1), num_classes=self.num_classes).float().to(device)
                
                # Calculate error vectors (predicted probabilities - ground truth)
                e1 = p1_prob.float() - y1_oh
                e2 = p2_prob.float() - y2_oh
                
                # Handle any numerical issues in error vectors
                e1 = torch.clamp(e1, min=-1.0, max=1.0)  # Ensure errors are in [-1, 1] range
                e2 = torch.clamp(e2, min=-1.0, max=1.0)
                
                # Compute distance between error vectors
                D_e_raw = torch.cdist(e1, e2, p=2)
                
                # Maximum possible error distance for vectors in [-1, 1]^K space
                # For K classes, worst case is 2√K (opposite corners of hypercube)
                # But we use 2*sqrt(2) for consistency with the original code
                max_raw_error_dist = 2.0 * np.sqrt(2.0)
                
                # Scale to [0, 1] range and apply beta weight
                D_e_scaled = D_e_raw / max_raw_error_dist
                term2 = beta * D_e_scaled
                max_term2_contrib = beta * 1.0
            except Exception as e:
                logger.warning(f"Error distance calculation failed: {e}")
                return None, np.nan

        # --- Combine Terms ---
        cost_matrix = term1 + term2
        max_possible_cost = max_term1_contrib + max_term2_contrib
        effective_max_cost = max(self.eps_num, max_possible_cost)
        
        # Ensure cost matrix is valid (no NaNs, no negative values)
        cost_matrix = torch.clamp(cost_matrix, min=0.0, max=effective_max_cost)
        
        # Check for NaN/Inf values
        if not torch.isfinite(cost_matrix).all():
            logger.warning("Non-finite values detected in cost matrix. Replacing with max cost.")
            cost_matrix = torch.where(
                torch.isfinite(cost_matrix),
                cost_matrix,
                torch.tensor(effective_max_cost, device=device, dtype=torch.float)
            )

        return cost_matrix, float(max_possible_cost)

class DecomposedOTCalculator(BaseOTCalculator):
    """ Calculates Decomposed Similarity: Label EMD + Aggregated Conditional OT. """
    
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
        beta_within = params.get('beta_within', 0.0) # Check if probs needed for within-cost
        min_samples_threshold = params.get('min_samples', 20) # Get the minimum samples threshold
        max_samples_threshold = params.get('max_samples', 900) # Get the maximum samples threshold

        # --- Preprocess and Validate ---
        # Need h, y, weights always. Need p_prob if aggregating by loss or beta_within > 0.
        required_keys = ['h', 'y', 'weights']
        prob_needed = beta_within > self.eps_num or agg_method_param in ['avg_loss', 'total_loss_share']
        if prob_needed:
            required_keys.append('p_prob')
            # Note: 'loss' is derived from p_prob and y, handled after preprocessing

        proc_data1, proc_data2 = self._preprocess_input(data1, data2, required_keys)

        if proc_data1 is None or proc_data2 is None:
            logger.warning(f"Decomposed OT requires {required_keys}. Preprocessing failed or data missing. Skipping.")
            return

        h1, y1, w1 = proc_data1['h'], proc_data1['y'], proc_data1['weights']
        h2, y2, w2 = proc_data2['h'], proc_data2['y'], proc_data2['weights']
        p1_prob = proc_data1.get('p_prob')
        p2_prob = proc_data2.get('p_prob')
        N, M = h1.shape[0], h2.shape[0]

        if N == 0 or M == 0:
            logger.warning("Decomposed OT: One or both clients have zero samples. Skipping.")
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
                    logger.warning(f"Cannot use loss aggregation: Loss calculation failed or resulted in NaN/Inf.")
            else:
                logger.warning(f"Cannot use loss aggregation: Probabilities (p_prob) missing.")

        # --- 1. Label EMD ---
        label_emd_raw = calculate_label_emd(y1, y2, self.num_classes)
        if np.isnan(label_emd_raw):
            logger.warning("Label EMD failed. Decomposed OT aborted."); return
        emd_norm_factor = max(1.0, float(self.num_classes - 1)) if self.num_classes > 1 else 1.0
        label_emd_normalized = label_emd_raw / emd_norm_factor if normalize_emd_flag else label_emd_raw
        self.results['label_emd'] = label_emd_normalized

        # --- 2. Class-Conditional OT ---
        # Organize features by label for validation and sampling
        features_by_label = {}
        for c in range(self.num_classes):
            # Client 1
            client1_indices = torch.where(y1 == c)[0]
            if client1_indices.shape[0] > 0:
                features_by_label["client1"] = features_by_label.get("client1", {})
                features_by_label["client1"][c] = h1[client1_indices].cpu().numpy()
            
            # Client 2
            client2_indices = torch.where(y2 == c)[0]
            if client2_indices.shape[0] > 0:
                features_by_label["client2"] = features_by_label.get("client2", {})
                features_by_label["client2"][c] = h2[client2_indices].cpu().numpy()
        
        # Validate which labels have sufficient samples and get sampling indices
        label_validity, sample_indices_by_client_label = validate_samples_for_decomposed_ot(
            features_by_label, min_samples_threshold, max_samples_threshold)
        
        if verbose:
            valid_labels = [l for l, v in label_validity.items() if v]
            invalid_labels = [l for l, v in label_validity.items() if not v]
            logger.info(f"Label validation: {len(valid_labels)} valid, {len(invalid_labels)} skipped")
            if invalid_labels:
                logger.info(f"Skipping labels due to insufficient samples: {invalid_labels}")
                
            # Report on sampling
            for client_id, label_indices in sample_indices_by_client_label.items():
                for label, indices in label_indices.items():
                    orig_count = len(features_by_label[client_id][label])
                    if len(indices) < orig_count:
                        logger.info(f"Sampling {client_id}, label {label}: {orig_count} → {len(indices)}")

        all_class_results = {} # Stores {'ot_cost': float, 'avg_loss': float, 'total_loss': float, 'sample_count': tuple, 'valid': bool}
        total_loss_across_classes = 0.0
        if can_use_loss_agg and loss1 is not None and loss2 is not None:
            total_loss_across_classes = (loss1.sum() + loss2.sum()).item()

        for c in range(self.num_classes):
            class_result = {'ot_cost': np.nan, 'avg_loss': 0.0, 'total_loss': 0.0, 'sample_count': (0, 0), 'valid': False}
            idx1_c = torch.where(y1 == c)[0]; idx2_c = torch.where(y2 == c)[0]
            n_c, m_c = idx1_c.shape[0], idx2_c.shape[0]
            class_result['sample_count'] = (n_c, m_c)

            # Skip this label if it fails the validation check
            if not label_validity.get(c, False):
                if verbose: 
                    logger.info(f"Class {c}: Skipping due to insufficient samples (min={min_samples_threshold})")
                all_class_results[c] = class_result
                continue
            
            if n_c == 0 or m_c == 0:
                all_class_results[c] = class_result; continue # Skip if no samples in either client

            # --- Gather class data & handle loss ---
            # Apply sampling to client 1 data
            if "client1" in sample_indices_by_client_label and c in sample_indices_by_client_label["client1"]:
                sample1_indices = sample_indices_by_client_label["client1"][c]
                # Map sample indices to actual indices in the full dataset
                full_sample1_indices = idx1_c[torch.from_numpy(sample1_indices).long()]
                h1_c = h1[full_sample1_indices]
                w1_c = w1[full_sample1_indices]
                y1_c = y1[full_sample1_indices]
                p1_prob_c = p1_prob[full_sample1_indices] if p1_prob is not None else None
                loss1_c = loss1[full_sample1_indices] if loss1 is not None else None
                n_c_final = len(sample1_indices)
            else:
                h1_c, w1_c, y1_c = h1[idx1_c], w1[idx1_c], y1[idx1_c]
                p1_prob_c = p1_prob[idx1_c] if p1_prob is not None else None
                loss1_c = loss1[idx1_c] if loss1 is not None else None
                n_c_final = n_c

            # Apply sampling to client 2 data
            if "client2" in sample_indices_by_client_label and c in sample_indices_by_client_label["client2"]:
                sample2_indices = sample_indices_by_client_label["client2"][c]
                # Map sample indices to actual indices in the full dataset
                full_sample2_indices = idx2_c[torch.from_numpy(sample2_indices).long()]
                h2_c = h2[full_sample2_indices]
                w2_c = w2[full_sample2_indices]
                y2_c = y2[full_sample2_indices]
                p2_prob_c = p2_prob[full_sample2_indices] if p2_prob is not None else None
                loss2_c = loss2[full_sample2_indices] if loss2 is not None else None
                m_c_final = len(sample2_indices)
            else:
                h2_c, w2_c, y2_c = h2[idx2_c], w2[idx2_c], y2[idx2_c]
                p2_prob_c = p2_prob[idx2_c] if p2_prob is not None else None
                loss2_c = loss2[idx2_c] if loss2 is not None else None
                m_c_final = m_c

            # Update sample counts after sampling
            class_result['sample_count'] = (n_c_final, m_c_final)

            if can_use_loss_agg and loss1_c is not None and loss2_c is not None:
                if torch.isfinite(loss1_c).all() and torch.isfinite(loss2_c).all():
                    total_loss_c = (loss1_c.sum() + loss2_c.sum()).item()
                    avg_loss_c = total_loss_c / (n_c_final + m_c_final) if (n_c_final + m_c_final) > 0 else 0.0
                    class_result['total_loss'] = total_loss_c; class_result['avg_loss'] = avg_loss_c

            # Check if probs are needed for cost and available
            if beta_within > self.eps_num and (p1_prob_c is None or p2_prob_c is None):
                if verbose: 
                    logger.warning(f"Class {c}: Skipped conditional OT - probs needed for cost (beta_within>0) but unavailable.");
                all_class_results[c] = class_result; continue

            # --- Calculate Within-Class Cost Matrix ---
            cost_matrix_c, max_cost_c = self._calculate_cost_within_class(
                h1_c, p1_prob_c, y1_c, h2_c, p2_prob_c, y2_c, **params
            )
            if cost_matrix_c is None or not (np.isfinite(max_cost_c) and max_cost_c > self.eps_num):
                fail_reason = "within-class cost calc failed" if cost_matrix_c is None else "within-class max cost near zero/invalid"
                if verbose: 
                    logger.warning(f"Class {c}: Skipped conditional OT - {fail_reason}.")
                all_class_results[c] = class_result; continue

            # --- Normalize Cost & Compute OT ---
            norm_cost_matrix_c = normalize_cost_matrix(cost_matrix_c, max_cost_c, normalize_within_cost, self.eps_num)
            self.cost_matrices['within_class'][c] = norm_cost_matrix_c.cpu().numpy()

            # Get prepared marginals using utility function
            a_c, b_c = prepare_ot_marginals(w1_c.cpu().numpy(), w2_c.cpu().numpy(), n_c_final, m_c_final, self.eps_num)

            ot_cost_c, _ = compute_ot_cost(norm_cost_matrix_c, a=a_c, b=b_c, reg=ot_reg, sinkhorn_max_iter=ot_max_iter, eps_num=self.eps_num)
            if not np.isnan(ot_cost_c):
                class_result['ot_cost'] = ot_cost_c
                class_result['valid'] = True
                self.results['conditional_ot_per_class'][c] = ot_cost_c # Store individual result
            elif verbose:
                logger.warning(f"Conditional OT cost calculation failed for class {c}.")

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
                logger.warning(f"Cannot use '{agg_method_param}' due to invalid losses. Falling back to 'mean'.")
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
                else: 
                    weights_agg = np.ones_like(costs_agg)
                    agg_method_used_str += " (Fallback - Zero Total Samples)"
            elif effective_agg_method == 'avg_loss':
                weights_agg = np.array([all_class_results[c]['avg_loss'] for c in valid_classes], dtype=np.float64)
            elif effective_agg_method == 'total_loss_share':
                if total_loss_across_classes > self.eps_num:
                    weights_agg = np.array([all_class_results[c]['total_loss'] / total_loss_across_classes for c in valid_classes], dtype=np.float64)
                else: 
                    weights_agg = np.ones_like(costs_agg)
                    agg_method_used_str += " (Fallback - Zero Total Loss)"

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
            logger.warning("No valid conditional OT costs calculated to aggregate.")

        self.results['conditional_ot_agg'] = agg_conditional_ot
        self.results['aggregation_method_used'] = agg_method_used_str
        self.results['aggregation_weights'] = agg_weights_dict
        
        # Track skipped and sampled labels for reporting
        self.results['skipped_labels'] = [c for c in range(self.num_classes) if c not in valid_classes]
        self.results['sampled_labels'] = {}
        for client_id, label_indices in sample_indices_by_client_label.items():
            for label, indices in label_indices.items():
                orig_count = features_by_label.get(client_id, {}).get(label, [])
                # Fix the ambiguous boolean check:
                if hasattr(orig_count, '__len__') and len(indices) < len(orig_count):
                    if label not in self.results['sampled_labels']:
                        self.results['sampled_labels'][label] = {}
                    self.results['sampled_labels'][label][client_id] = (len(orig_count), len(indices))

        
        # --- 4. Calculate Combined Score ---
        if not np.isnan(label_emd_normalized) and not np.isnan(agg_conditional_ot):
            # Simple average, consider weighting later if needed
            self.results['combined_score'] = (label_emd_normalized + agg_conditional_ot) / 2.0
        else: self.results['combined_score'] = np.nan

        if verbose:
            logger.info("-" * 30)
            logger.info("--- Decomposed Similarity Results ---")
            emd_str = f"{self.results['label_emd']:.3f}" if not np.isnan(self.results['label_emd']) else 'Failed'
            logger.info(f"  Label EMD (Normalized={normalize_emd_flag}): {emd_str}")
            logger.info(f"  Conditional Within-Class OT Costs (NormCost={normalize_within_cost}):")
            for c in range(self.num_classes):
                cost_val = self.results['conditional_ot_per_class'].get(c, np.nan)
                cost_str = f"{cost_val:.3f}" if not np.isnan(cost_val) else 'Skipped/Failed'
                count_str = f"(samples: {all_class_results.get(c, {}).get('sample_count', (0,0))})"
                logger.info(f"     - Class {c}: Cost={cost_str} {count_str}")
            cond_ot_agg_str = f"{self.results['conditional_ot_agg']:.3f}" if not np.isnan(self.results['conditional_ot_agg']) else 'Failed'
            logger.info(f"  Aggregated Conditional OT ({agg_method_used_str}): {cond_ot_agg_str}")
            logger.info(f"  Skipped Labels: {self.results['skipped_labels']}")
            
            # Report sampling if any occurred
            if self.results['sampled_labels']:
                logger.info(f"  Sampled Labels (original→sampled):")
                for label, clients in self.results['sampled_labels'].items():
                    client_info = ", ".join([f"{cid}: {orig}→{sampled}" for cid, (orig, sampled) in clients.items()])
                    logger.info(f"     - Label {label}: {client_info}")
                
            comb_score_str = f"{self.results['combined_score']:.3f}" if not np.isnan(self.results['combined_score']) else 'Invalid'
            logger.info(f"  Combined Score (/2): {comb_score_str}")
            logger.info("-" * 30)

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
            except Exception as e: 
                logger.warning(f"Within-class feature dist failed: {e}")
                return None, np.nan

        # --- Error Distance ---
        term2 = torch.zeros((N, M), device=device, dtype=torch.float)
        max_term2_contrib = 0.0
        if prob_needed and p1_prob_c is not None and p2_prob_c is not None:
             # Validate prob shapes
             if p1_prob_c.shape != (N, self.num_classes) or p2_prob_c.shape != (M, self.num_classes):
                  logger.warning(f"Within-class Cost: Prob shape mismatch P1({p1_prob_c.shape}), P2({p2_prob_c.shape})")
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
             except Exception as e: 
                logger.warning(f"Within-class error dist failed: {e}")
                return None, np.nan

        # --- Combine ---
        cost_matrix_c = term1 + term2
        max_possible_cost_c = max_term1_contrib + max_term2_contrib
        effective_max_cost_c = max(self.eps_num, max_possible_cost_c)
        cost_matrix_c = torch.clamp(cost_matrix_c, min=0.0, max=effective_max_cost_c)

        return cost_matrix_c, float(max_possible_cost_c)

class DirectOTCalculator(BaseOTCalculator):
    """
    Calculates direct OT cost between neural network activations with additional
    label distribution similarity using Hellinger distance.
    
    This method computes optimal transport using both the feature representations and
    the distributional properties of those representations, combining both to create
    a more comprehensive similarity metric.
    """
    
    def _reset_results(self) -> None:
        """Initialize/reset the results dictionary and cost matrices."""
        self.results = {
            'direct_ot_cost': np.nan,
            'transport_plan': None,
            'feature_distance_method': None,
            'label_hellinger_used': False,
            'weighting_used': None,
            'label_hellinger_weight': np.nan,
            'feature_weight': np.nan,
            'label_costs': []
        }
        self.cost_matrices = {
            'direct_ot': None,
            'feature_cost': None,
            'label_cost': None,
            'combined_cost': None
        }
    
    def calculate_similarity(self, data1: Dict[str, Optional[torch.Tensor]], 
                        data2: Dict[str, Optional[torch.Tensor]], 
                        params: Dict[str, Any]) -> None:
        """
        Calculate direct OT similarity with label Hellinger distance.
        
        Args:
            data1: Processed data for client 1, including activations and labels
            data2: Processed data for client 2, including activations and labels
            params: Configuration parameters including:
                - normalize_activations: Whether to L2 normalize features
                - distance_method: Method for feature distance calculation
                - use_label_hellinger: Whether to use label Hellinger distance
                - label_weight: Weight for label cost (default 1.0)
                - feature_weight: Weight for feature cost (default 2.0)
                - use_loss_weighting: Whether to use loss-based weighting for margins
                - compress_vectors: Whether to compress high-dimensional vectors
                - verbose: Print detailed information
        """
        self._reset_results()
        
        # Extract parameters
        verbose = params.get('verbose', False)
        normalize_activations = params.get('normalize_activations', True)
        normalize_cost = params.get('normalize_cost', True)
        distance_method = params.get('distance_method', 'euclidean')
        use_loss_weighting = params.get('use_loss_weighting', False)
        use_label_hellinger = params.get('use_label_hellinger', True)
        feature_weight = params.get('feature_weight', 2.0)
        label_weight = params.get('label_weight', 1.0)
        compress_vectors = params.get('compress_vectors', True)
        compression_threshold = params.get('compression_threshold', 10)
        compression_ratio = params.get('compression_ratio', 5)  # Used for PCA variance ratio
        reg = params.get('reg', DEFAULT_OT_REG)
        max_iter = params.get('max_iter', DEFAULT_OT_MAX_ITER)
        min_samples_threshold = params.get('min_samples', 20)
        max_samples_threshold = params.get('max_samples', 900)  # Get minimum samples threshold
        
        # Store configuration in results
        self.results['feature_distance_method'] = distance_method
        self.results['label_hellinger_used'] = use_label_hellinger
        self.results['feature_weight'] = feature_weight
        self.results['label_hellinger_weight'] = label_weight
        
        # Process inputs - require activations 'h', labels 'y', and optionally weights
        required_keys = ['h']
        if use_label_hellinger:
            required_keys.append('y')
        if use_loss_weighting:
            required_keys.append('weights')

        proc_data1, proc_data2 = self._preprocess_input(data1, data2, required_keys)
        
        if proc_data1 is None or proc_data2 is None:
            logger.warning("Enhanced DirectOT calculation requires neural network activations ('h')" +
                        " and labels ('y' when using Hellinger). " + 
                        "Preprocessing failed or data missing. Skipping.")
            weight_type = "Loss" if use_loss_weighting else "Uniform"
            self.results['weighting_used'] = weight_type
            return
            
        h1 = proc_data1['h']  # Neural network activations for client 1
        h2 = proc_data2['h']  # Neural network activations for client 2        
        y1 = proc_data1.get('y')
        y2 = proc_data2.get('y')
        w1 = proc_data1.get('weights')
        w2 = proc_data2.get('weights')
        p1_prob = proc_data1.get('p_prob')  # Initialize explicitly, might be None
        p2_prob = proc_data2.get('p_prob')  # Initialize explicitly, might be None
        N, M = h1.shape[0], h2.shape[0]
        
        
        # Validate sample counts
        features_dict = {
            "client1": h1.cpu().numpy(),
            "client2": h2.cpu().numpy()
        }
        all_sufficient, sample_indices = validate_samples_for_ot(
            features_dict, min_samples_threshold, max_samples_threshold)

        if not all_sufficient:
            logger.warning(f"DirectOT: One or both clients have insufficient samples (min={min_samples_threshold}). Skipping.")
            weight_type = "Loss" if use_loss_weighting else "Uniform"
            self.results['weighting_used'] = weight_type
            return
        
        # Apply sampling if needed
        if "client1" in sample_indices and len(sample_indices["client1"]) < N:
            # Sample client 1 data with careful handling of possibly None values
            indices1 = torch.from_numpy(sample_indices["client1"]).long()
            h1 = h1[indices1]
            if y1 is not None:
                y1 = y1[indices1]
            if p1_prob is not None:  # Check if p1_prob exists before indexing
                p1_prob = p1_prob[indices1]
            if w1 is not None:
                w1 = w1[indices1]
            N = len(indices1)
            if verbose:
                logger.info(f"Sampled client1 data: {len(indices1)} from original {N}")

        if "client2" in sample_indices and len(sample_indices["client2"]) < M:
            # Sample client 2 data with careful handling
            indices2 = torch.from_numpy(sample_indices["client2"]).long()
            h2 = h2[indices2]
            if y2 is not None:
                y2 = y2[indices2]
            if p2_prob is not None:  # Check if p2_prob exists before indexing
                p2_prob = p2_prob[indices2]
            if w2 is not None:
                w2 = w2[indices2]
            M = len(indices2)
            if verbose:
                logger.info(f"Sampled client2 data: {len(indices2)} from original {M}")
                
        if N == 0 or M == 0:
            logger.warning("Enhanced DirectOT: One or both clients have zero samples. OT cost is 0.")
            self.results['direct_ot_cost'] = 0.0
            weight_type = "Loss" if use_loss_weighting else "Uniform"
            self.results['weighting_used'] = weight_type
            return
        
        # --- Feature Cost Matrix Calculation ---
        # Normalize activations if requested
        if normalize_activations:
            h1_norm = F.normalize(h1.float(), p=2, dim=1, eps=self.eps_num)
            h2_norm = F.normalize(h2.float(), p=2, dim=1, eps=self.eps_num)
        else:
            h1_norm = h1.float()
            h2_norm = h2.float()
            
        # Compute feature cost matrix based on distance method
        if distance_method == 'euclidean':
            feature_cost_matrix = torch.cdist(h1_norm, h2_norm, p=2)
            max_feature_cost = 2.0 if normalize_activations else float('inf')
        elif distance_method == 'cosine':
            # Compute cosine similarity and convert to distance
            cos_sim = torch.mm(h1_norm, h2_norm.t())
            feature_cost_matrix = 1.0 - cos_sim
            max_feature_cost = 2.0
        elif distance_method == 'squared_euclidean':
            feature_cost_matrix = pairwise_euclidean_sq(h1_norm, h2_norm)
            max_feature_cost = 4.0 if normalize_activations else float('inf')
        else:
            logger.warning(f"Unknown distance method: {distance_method}. Using euclidean.")
            feature_cost_matrix = torch.cdist(h1_norm, h2_norm, p=2)
            max_feature_cost = 2.0 if normalize_activations else float('inf')
        
        if feature_cost_matrix is None:
            logger.warning(f"Failed to compute feature cost matrix with method: {distance_method}")
            return
            
        # Normalize feature cost matrix if requested
        feature_cost_matrix = normalize_cost_matrix(feature_cost_matrix, max_feature_cost, normalize_cost, self.eps_num)
            
        # Store the feature cost matrix
        self.cost_matrices['feature_cost'] = feature_cost_matrix.cpu().numpy()
        
        # --- Label Cost Matrix Calculation (Hellinger distance) ---
        # Initialize label cost matrix with same shape as feature cost matrix
        label_cost_matrix = torch.zeros_like(feature_cost_matrix)
        
        if use_label_hellinger and 'y' in proc_data1 and 'y' in proc_data2:
            try:                
                # Convert to numpy for easier handling
                h1_np = h1.cpu().numpy()
                h2_np = h2.cpu().numpy()
                y1_np = y1.cpu().numpy()
                y2_np = y2.cpu().numpy()
                
                # Check if compression is needed for high-dimensional vectors
                vector_dim = h1_np.shape[1]
                if compress_vectors and vector_dim > compression_threshold:
                    if verbose:
                        logger.info(f"Compressing vectors from dimension {vector_dim} for Hellinger calculation")
                    h1_comp, h2_comp = self._compress_vectors(h1_np, h2_np, compression_ratio)
                else:
                    h1_comp, h2_comp = h1_np, h2_np
                
                # Get unique labels from both clients
                unique_labels1 = np.unique(y1_np)
                unique_labels2 = np.unique(y2_np)
                
                # Calculate Hellinger distances for each label pair
                label_pair_distances = {}
                
                for label1 in unique_labels1:
                    for label2 in unique_labels2:
                        # Get indices for each label
                        indices1 = np.where(y1_np == label1)[0]
                        indices2 = np.where(y2_np == label2)[0]
                        if len(indices1) >= 2 and len(indices2) >= 2:  # Need at least 2 samples to estimate distribution
                            # Get features for each label
                            h1_label = h1_comp[indices1]
                            h2_label = h2_comp[indices2]
                            
                            # Calculate distribution parameters
                            mu_1, sigma_1 = self._get_normal_params(h1_label)
                            mu_2, sigma_2 = self._get_normal_params(h2_label)
                            # Calculate Hellinger distance
                            if label1 != label2:
                                hellinger_dist = 4 /(len(unique_labels1) +len(unique_labels1)) + self._hellinger_distance(mu_1, sigma_1, mu_2, sigma_2)
                            else:
                                hellinger_dist = self._hellinger_distance(mu_1, sigma_1, mu_2, sigma_2)
                            #hellinger_dist = self._hellinger_distance(mu_1, sigma_1, mu_2, sigma_2)
                            if hellinger_dist is not None:
                                label_pair_distances[(label1, label2)] = hellinger_dist
                                # Store for reporting
                                self.results['label_costs'].append(((label1, label2), hellinger_dist))
                            else:
                                if label1 != label2:
                                # Fallback to a neutral midpoint if calculation fails
                                    label_pair_distances[(label1, label2)] = 2
                                    self.results['label_costs'].append(((label1, label2), 2))
                                else:
                                    label_pair_distances[(label1, label2)] = 0.5
                                    self.results['label_costs'].append(((label1, label2), 0.5))
                                if verbose:
                                    logger.warning(f"Hellinger distance calculation failed for labels {label1},{label2}")
                        else:
                            # Not enough samples for distribution, use midpoint
                            if label1 != label2:
                            # Fallback to a neutral midpoint if calculation fails
                                label_pair_distances[(label1, label2)] = 2
                                self.results['label_costs'].append(((label1, label2), 2))
                            else:
                                label_pair_distances[(label1, label2)] = 0.5
                                self.results['label_costs'].append(((label1, label2), 0.5))
                            if verbose:
                                logger.info(f"Not enough samples for labels {label1},{label2}.")
                
                # Fill the label cost matrix based on the calculated distances
                for i in range(N):
                    label_i = y1_np[i]
                    for j in range(M):
                        label_j = y2_np[j]
                        pair_key = (label_i, label_j)
                        if pair_key in label_pair_distances:
                            label_cost_matrix[i, j] = label_pair_distances[pair_key]

                # Store the label cost matrix
                self.cost_matrices['label_cost'] = label_cost_matrix.cpu().numpy()
                
            except Exception as e:
                logger.warning(f"Label Hellinger distance calculation failed: {e}")
                # Set all label costs to a neutral midpoint value
                label_cost_matrix.fill_(0.5)
                self.cost_matrices['label_cost'] = label_cost_matrix.cpu().numpy()
                use_label_hellinger = False  # Disable for the rest of the calculation
        else:
            if use_label_hellinger:
                logger.warning("Label Hellinger requested but labels not available. Using only feature distance.")
            use_label_hellinger = False
            # Fill with neutral value just in case
            label_cost_matrix.fill_(0.5)
            self.cost_matrices['label_cost'] = label_cost_matrix.cpu().numpy()
        
        # --- Combine Feature and Label Costs ---
        if use_label_hellinger:
            # Normalize weights to sum to 1
            total_weight = feature_weight + label_weight
            norm_feature_weight = feature_weight / total_weight
            norm_label_weight = label_weight / total_weight
            
            # Combine costs
            combined_cost_matrix = (norm_feature_weight * feature_cost_matrix + 
                                norm_label_weight * label_cost_matrix)
    
            
            # Store the combined cost matrix
            self.cost_matrices['combined_cost'] = combined_cost_matrix.cpu().numpy()
            cost_matrix = combined_cost_matrix
        else:
            # Use only feature cost if label cost is not used
            cost_matrix = feature_cost_matrix
            self.cost_matrices['combined_cost'] = self.cost_matrices['feature_cost']
        
        # Store the final cost matrix used for OT
        self.cost_matrices['direct_ot'] = cost_matrix.cpu().numpy()
        
        # --- Prepare Weights for OT ---
        if use_loss_weighting and w1 is not None and w2 is not None:
            a, b = prepare_ot_marginals(w1.cpu().numpy(), w2.cpu().numpy(), N, M, self.eps_num)
            weight_type = "Loss-Weighted"
        else:
            if use_loss_weighting: 
                logger.warning("Loss weighting requested but weights unavailable. Using uniform.")
            a = np.ones(N, dtype=np.float64) / N
            b = np.ones(M, dtype=np.float64) / M
            weight_type = "Uniform"
        
        self.results['weighting_used'] = weight_type
        
        # --- Compute OT Cost ---
        ot_cost, transport_plan = compute_ot_cost(
            cost_matrix, a=a, b=b, reg=reg, 
            sinkhorn_max_iter=max_iter, eps_num=self.eps_num
        )
        
        self.results['direct_ot_cost'] = ot_cost
        self.results['transport_plan'] = transport_plan
        
        if verbose:
            # Fix the format specifier error
            if np.isfinite(ot_cost):
                logger.info(f"  Enhanced DirectOT Cost ({weight_type} weights): {ot_cost:.4f}")
            else:
                logger.info(f"  Enhanced DirectOT Cost ({weight_type} weights): Failed")
                
            if use_label_hellinger:
                logger.info(f"  Label Hellinger distances by class pairs:")
                for (label1, label2), dist in self.results['label_costs']:
                    logger.info(f"    Labels ({label1},{label2}): {dist:.4f}")
            else:
                logger.info(f"  Using only feature distances ({distance_method})")

    def _compress_vectors(self, X1: np.ndarray, X2: np.ndarray, 
                         variance_ratio: float = 0.8) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compress high-dimensional vectors using PCA.
        
        Args:
            X1: First set of vectors
            X2: Second set of vectors
            variance_ratio: Amount of variance to retain (0.0-1.0)
            
        Returns:
            Tuple of compressed vectors (X1_comp, X2_comp)
        """
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        # Standardize the data
        scaler = StandardScaler()
        X1_scaled = scaler.fit_transform(X1)
        X2_scaled = scaler.transform(X2)  # Use same scaling for both

        # Apply PCA
        pca = PCA(n_components=variance_ratio)
        X1_comp = pca.fit_transform(X1_scaled)
        X2_comp = pca.transform(X2_scaled)  # Use same transformation for both
        
        return X1_comp, X2_comp
    
    def _get_normal_params(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute mean and covariance matrix for a set of vectors.
        
        Args:
            X: Input vectors (n_samples, n_features)
            
        Returns:
            Tuple of (mean vector, covariance matrix)
        """
        mu = np.mean(X, axis=0)
        sigma = np.cov(X, rowvar=False)
        # Add small regularization to ensure positive definiteness
        sigma += 1e-6 * np.eye(sigma.shape[0])
        return mu, sigma
        
    def _hellinger_distance(self, mu_1: np.ndarray, sigma_1: np.ndarray, 
                        mu_2: np.ndarray, sigma_2: np.ndarray) -> Optional[float]:
        """
        Calculate the Hellinger distance between two multivariate normal distributions.
        """
        try:
            # Add explicit shape checks
            if mu_1.shape != mu_2.shape or sigma_1.shape != sigma_2.shape:
                logger.warning(f"Shape mismatch: mu_1{mu_1.shape}, mu_2{mu_2.shape}, "
                            f"sigma_1{sigma_1.shape}, sigma_2{sigma_2.shape}")
                return None
                
            # Ensure positive definiteness through eigendecomposition
            s1_vals, s1_vecs = np.linalg.eigh(sigma_1)
            s2_vals, s2_vecs = np.linalg.eigh(sigma_2)
            
            # Reconstruct with only positive eigenvalues
            s1_vals = np.maximum(s1_vals, 1e-2)
            s2_vals = np.maximum(s2_vals, 1e-2)
            
            s1_recon = s1_vecs @ np.diag(s1_vals) @ s1_vecs.T
            s2_recon = s2_vecs @ np.diag(s2_vals) @ s2_vecs.T
            
            # Average covariance
            avg_sigma = (s1_recon + s2_recon) / 2
            
            # Calculate determinants
            det_s1 = np.linalg.det(s1_recon)
            det_s2 = np.linalg.det(s2_recon)
            det_avg_sigma = np.linalg.det(avg_sigma)
            # Avoid numerical issues with determinants
            if det_s1 <= 0 or det_s2 <= 0 or det_avg_sigma <= 0:
                return None
                
            # First term: determinant component
            term1 = (np.power(det_s1, 0.25) * np.power(det_s2, 0.25)) / np.sqrt(det_avg_sigma)
            
            # Second term: exponential component with mean difference
            diff_mu = mu_1 - mu_2
            inv_avg_sigma = np.linalg.inv(avg_sigma)
            term2 = np.exp(-0.125 * np.dot(diff_mu, np.dot(inv_avg_sigma, diff_mu)))
            
            # Final Hellinger distance
            distance = 1 - np.sqrt(term1 * term2)
            # Handle numerical issues
            if not np.isfinite(distance):
                return None
                
            return float(distance)
            
        except np.linalg.LinAlgError as e:
            # Handle linear algebra errors (singular matrices, etc)
            logger.warning(f"Linear algebra error in Hellinger calculation: {e}")
            return None
        except IndexError as e:
            # Handle indexing errors explicitly
            logger.warning(f"Indexing error in Hellinger calculation: {e}") 
            return None
        except Exception as e:
            # Handle other errors
            logger.warning(f"Hellinger distance calculation error: {e}")
            return None