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
    compute_ot_cost, pairwise_euclidean_sq,
    validate_samples_for_ot,
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
        min_samples_threshold = params.get('min_samples', 20)
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
            self.results['weighting_used'] = 'Loss' if use_loss_weighting else 'Uniform'
            return

        # Extract all tensors (use full data for initial processing)
        h1 = proc_data1['h']
        h2 = proc_data2['h']
        p1_prob = proc_data1.get('p_prob')
        y1 = proc_data1.get('y')
        w1 = proc_data1.get('weights')
        p2_prob = proc_data2.get('p_prob')
        y2 = proc_data2.get('y')
        w2 = proc_data2.get('weights')
        
        N, M = h1.shape[0], h2.shape[0]

        # Validate sample counts and get sampling indices - but don't apply sampling yet
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

        if N == 0 or M == 0:
            logger.warning("Feature-Error OT: One or both clients have zero samples. OT cost is 0.")
            self.results['ot_cost'] = 0.0
            self.results['transport_plan'] = np.zeros((N, M)) if N > 0 and M > 0 else None
            self.results['weighting_used'] = 'Loss' if use_loss_weighting else 'Uniform'
            return

        # --- Calculate Cost Matrix (using full data) ---
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
        
        # --- NOW apply sampling to cost matrix and weights just before OT calculation ---
        # Full weight vectors before sampling
        if use_loss_weighting and w1 is not None and w2 is not None:
            full_w1 = w1.cpu().numpy()
            full_w2 = w2.cpu().numpy()
            weighting_type_str = "Loss-Weighted"
        else:
            if use_loss_weighting:
                logger.warning("Loss weighting requested but weights unavailable. Using uniform.")
            full_w1 = np.ones(N, dtype=np.float64) / N
            full_w2 = np.ones(M, dtype=np.float64) / M
            weighting_type_str = "Uniform"
        self.results['weighting_used'] = weighting_type_str
        
        # Apply sampling to cost matrix and weights
        sampled_cost_matrix = normalized_cost_matrix
        sampled_w1, sampled_w2 = full_w1, full_w2
        N_eff, M_eff = N, M
        
        # Apply sampling to client 1 (rows)
        if "client1" in sample_indices and len(sample_indices["client1"]) < N:
            indices1 = torch.from_numpy(sample_indices["client1"]).long()
            sampled_cost_matrix = sampled_cost_matrix[indices1]
            sampled_w1 = full_w1[sample_indices["client1"]]
            N_eff = len(indices1)
            if verbose:
                logger.info(f"Sampled client1 cost matrix rows: {N_eff} from original {N}")
        
        # Apply sampling to client 2 (columns)
        if "client2" in sample_indices and len(sample_indices["client2"]) < M:
            indices2 = torch.from_numpy(sample_indices["client2"]).long()
            sampled_cost_matrix = sampled_cost_matrix[:, indices2]
            sampled_w2 = full_w2[sample_indices["client2"]]
            M_eff = len(indices2)
            if verbose:
                logger.info(f"Sampled client2 cost matrix columns: {M_eff} from original {M}")
        
        # Store the full cost matrix for reference
        self.cost_matrices['feature_error_ot'] = normalized_cost_matrix.cpu().numpy()
            
        # --- Use prepare_ot_marginals with the possibly sampled weights ---
        a, b = prepare_ot_marginals(sampled_w1, sampled_w2, N_eff, M_eff, self.eps_num)

        # --- Compute OT Cost with the sampled cost matrix and marginals ---
        ot_cost, Gs = compute_ot_cost(
            sampled_cost_matrix, a=a, b=b, reg=reg, sinkhorn_max_iter=max_iter, eps_num=self.eps_num
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
            params: Configuration parameters
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
        max_samples_threshold = params.get('max_samples', 900)
        
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
            
        # Extract all data (use full data for calculations)
        h1 = proc_data1['h']  
        h2 = proc_data2['h']  
        y1 = proc_data1.get('y')
        y2 = proc_data2.get('y')
        w1 = proc_data1.get('weights')
        w2 = proc_data2.get('weights')
        p1_prob = proc_data1.get('p_prob')
        p2_prob = proc_data2.get('p_prob')
        N, M = h1.shape[0], h2.shape[0]
        
        
        # Validate sample counts - but don't apply sampling yet
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
                
        if N == 0 or M == 0:
            logger.warning("Enhanced DirectOT: One or both clients have zero samples. OT cost is 0.")
            self.results['direct_ot_cost'] = 0.0
            weight_type = "Loss" if use_loss_weighting else "Uniform"
            self.results['weighting_used'] = weight_type
            return
        
        # --- Feature Cost Matrix Calculation (using full data) ---
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
        
        # --- Label Cost Matrix Calculation (Hellinger distance) using full data ---
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
                
                # Calculate Hellinger distances for each label pair (using all available data)
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
                            
                            if hellinger_dist is not None:
                                label_pair_distances[(label1, label2)] = hellinger_dist
                                # Store for reporting
                                self.results['label_costs'].append(((label1, label2), hellinger_dist))
                            else:
                                if label1 != label2:
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
        
        # Store the final full cost matrix used for OT
        self.cost_matrices['direct_ot'] = cost_matrix.cpu().numpy()
        
        # --- Prepare full weights for OT ---
        if use_loss_weighting and w1 is not None and w2 is not None:
            full_w1 = w1.cpu().numpy()
            full_w2 = w2.cpu().numpy()
            weight_type = "Loss-Weighted"
        else:
            if use_loss_weighting: 
                logger.warning("Loss weighting requested but weights unavailable. Using uniform.")
            full_w1 = np.ones(N, dtype=np.float64) / N
            full_w2 = np.ones(M, dtype=np.float64) / M
            weight_type = "Uniform"
        
        self.results['weighting_used'] = weight_type
        
        # --- Now apply sampling to cost matrix and weights just before OT calculation ---
        sampled_cost_matrix = cost_matrix
        sampled_w1, sampled_w2 = full_w1, full_w2
        N_eff, M_eff = N, M
        
        # Apply sampling to client 1 (rows)
        if "client1" in sample_indices and len(sample_indices["client1"]) < N:
            indices1 = torch.from_numpy(sample_indices["client1"]).long()
            sampled_cost_matrix = sampled_cost_matrix[indices1]
            sampled_w1 = full_w1[sample_indices["client1"]]
            N_eff = len(indices1)
            if verbose:
                logger.info(f"Sampled client1 cost matrix rows: {N_eff} from original {N}")
        
        # Apply sampling to client 2 (columns)
        if "client2" in sample_indices and len(sample_indices["client2"]) < M:
            indices2 = torch.from_numpy(sample_indices["client2"]).long()
            sampled_cost_matrix = sampled_cost_matrix[:, indices2]
            sampled_w2 = full_w2[sample_indices["client2"]]
            M_eff = len(indices2)
            if verbose:
                logger.info(f"Sampled client2 cost matrix columns: {M_eff} from original {M}")
        
        # --- Use prepare_ot_marginals with the possibly sampled weights ---
        a, b = prepare_ot_marginals(sampled_w1, sampled_w2, N_eff, M_eff, self.eps_num)
        
        # --- Compute OT Cost with sampled cost matrix and marginals ---
        ot_cost, transport_plan = compute_ot_cost(
            sampled_cost_matrix, a=a, b=b, reg=reg, 
            sinkhorn_max_iter=max_iter, eps_num=self.eps_num
        )
        
        self.results['direct_ot_cost'] = ot_cost
        self.results['transport_plan'] = transport_plan
        
        if verbose:
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