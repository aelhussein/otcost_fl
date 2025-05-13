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
    prepare_ot_marginals, normalize_cost_matrix
)
from ot_configs import DEFAULT_EPS, DEFAULT_OT_REG, DEFAULT_OT_MAX_ITER
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
        
        # Check if we should only match within classes
        within_class_only = params.get('within_class_only', False)

        # --- Preprocess and Validate ---
        # For within-class matching, we always need labels
        required_keys = ['h'] 
        if within_class_only:
            required_keys.append('y')
        else:
            # Regular requirements for standard feature-error OT
            is_segmentation = data1.get('p_prob') is None and data2.get('p_prob') is None
            if not is_segmentation:
                required_keys.extend(['p_prob', 'y'])
                
        if use_loss_weighting: 
            required_keys.append('weights')

        proc_data1, proc_data2 = self._preprocess_input(data1, data2, required_keys)

        if proc_data1 is None or proc_data2 is None:
            missing_str = "'weights'" if use_loss_weighting and (data1.get('weights') is None or data2.get('weights') is None) else "h, p_prob, or y"
            if within_class_only:
                missing_str = "h, y" + (", weights" if use_loss_weighting else "")
            logger.warning(f"Feature-Error OT calculation requires {missing_str}. Preprocessing failed or data missing. Skipping.")
            self.results['weighting_used'] = 'Loss' if use_loss_weighting else 'Uniform'
            return

        # Extract all tensors
        h1 = proc_data1['h']
        h2 = proc_data2['h']
        p1_prob = proc_data1.get('p_prob')
        y1 = proc_data1.get('y')
        w1 = proc_data1.get('weights')
        p2_prob = proc_data2.get('p_prob')
        y2 = proc_data2.get('y')
        w2 = proc_data2.get('weights')
        
        N, M = h1.shape[0], h2.shape[0]

        # Check for empty datasets
        if N == 0 or M == 0:
            logger.warning("Feature-Error OT: One or both clients have zero samples. OT cost is 0.")
            self.results['ot_cost'] = 0.0
            self.results['transport_plan'] = np.zeros((N, M)) if N > 0 and M > 0 else None
            self.results['weighting_used'] = 'Loss' if use_loss_weighting else 'Uniform'
            return
        
        # For within-class matching, we branch here
        if within_class_only:
            # Store per-class results
            per_class_results = []
            total_weighted_ot_cost = 0.0
            total_weight = 0.0
            
            # Ensure labels are available
            if y1 is None or y2 is None:
                logger.warning("Within-class matching requires labels for both clients. Skipping.")
                self.results['weighting_used'] = 'Loss' if use_loss_weighting else 'Uniform'
                return
                
            # Convert to numpy for easier handling
            y1_np = y1.cpu().numpy()
            y2_np = y2.cpu().numpy()
            
            # Find unique classes in both clients
            unique_classes1 = set(np.unique(y1_np))
            unique_classes2 = set(np.unique(y2_np))
            shared_classes = sorted(unique_classes1.intersection(unique_classes2))
            
            if not shared_classes:
                logger.warning("No shared classes between clients. Cannot compute within-class OT.")
                self.results['ot_cost'] = np.nan
                self.results['weighting_used'] = 'Loss' if use_loss_weighting else 'Uniform'
                self.results['shared_classes'] = []
                return
                
            # Process each class independently
            for class_label in shared_classes:
                # Find indices for this class
                idx1_k = torch.where(y1 == class_label)[0]
                idx2_k = torch.where(y2 == class_label)[0]
                
                N_k = len(idx1_k)
                M_k = len(idx2_k)
                
                # Check if we have enough samples for this class
                if N_k < min_samples_threshold or M_k < min_samples_threshold:
                    if verbose:
                        logger.info(f"Class {class_label}: Not enough samples (C1:{N_k}, C2:{M_k}). Skipping.")
                    continue
                    
                # Extract class-specific data
                h1_k = h1[idx1_k]
                h2_k = h2[idx2_k]
                
                # Extract additional tensors if available
                p1_prob_k = p1_prob[idx1_k] if p1_prob is not None else None
                p2_prob_k = p2_prob[idx2_k] if p2_prob is not None else None
                y1_k = y1[idx1_k]  # All values are class_label
                y2_k = y2[idx2_k]  # All values are class_label
                
                # Extract weights if using loss weighting
                w1_k = w1[idx1_k] if w1 is not None else None
                w2_k = w2[idx2_k] if w2 is not None else None
                
                # Calculate class-specific cost matrix
                if p1_prob_k is not None and p2_prob_k is not None:
                    cost_matrix_k, max_cost_k = self._calculate_cost_feature_error_additive(
                        h1_k, p1_prob_k, y1_k, h2_k, p2_prob_k, y2_k, **params
                    )
                else:
                    cost_matrix_k, max_cost_k = self._calculate_feature_distance_only(
                        h1_k, h2_k, **params
                    )
                    
                if cost_matrix_k is None or not np.isfinite(max_cost_k):
                    logger.warning(f"Cost matrix calculation failed for class {class_label}. Skipping.")
                    continue
                    
                # Normalize class-specific cost matrix
                normalized_cost_matrix_k = normalize_cost_matrix(
                    cost_matrix_k, max_cost_k, normalize_cost, self.eps_num
                )
                
                # Prepare weights for this class
                if use_loss_weighting and w1_k is not None and w2_k is not None:
                    full_w1_k = w1_k.cpu().numpy()
                    full_w2_k = w2_k.cpu().numpy()
                    # Re-normalize to ensure they sum to 1 within this class
                    sum1_k = np.sum(full_w1_k)
                    sum2_k = np.sum(full_w2_k)
                    if sum1_k > self.eps_num:
                        full_w1_k = full_w1_k / sum1_k
                    else:
                        full_w1_k = np.ones(N_k, dtype=np.float64) / N_k
                    if sum2_k > self.eps_num:
                        full_w2_k = full_w2_k / sum2_k
                    else:
                        full_w2_k = np.ones(M_k, dtype=np.float64) / M_k
                    weighting_type_str = "Loss-Weighted"
                else:
                    full_w1_k = np.ones(N_k, dtype=np.float64) / N_k
                    full_w2_k = np.ones(M_k, dtype=np.float64) / M_k
                    weighting_type_str = "Uniform"
                    
                # Check if sampling is needed within this class
                features_dict_k = {
                    "client1": h1_k.cpu().numpy(),
                    "client2": h2_k.cpu().numpy()
                }
                _, sample_indices_k = validate_samples_for_ot(
                    features_dict_k, 2, max_samples_threshold
                )
                
                # Apply sampling if needed (just before OT)
                sampled_cost_matrix_k = normalized_cost_matrix_k
                sampled_w1_k, sampled_w2_k = full_w1_k, full_w2_k
                N_k_eff, M_k_eff = N_k, M_k
                
                # Apply sampling to client 1 (rows)
                if "client1" in sample_indices_k and len(sample_indices_k["client1"]) < N_k:
                    indices1_k = torch.from_numpy(sample_indices_k["client1"]).long()
                    sampled_cost_matrix_k = sampled_cost_matrix_k[indices1_k]
                    sampled_w1_k = full_w1_k[sample_indices_k["client1"]]
                    N_k_eff = len(indices1_k)
                    if verbose:
                        logger.info(f"Class {class_label}: Sampled client1 rows: {N_k_eff} from {N_k}")
                
                # Apply sampling to client 2 (columns)
                if "client2" in sample_indices_k and len(sample_indices_k["client2"]) < M_k:
                    indices2_k = torch.from_numpy(sample_indices_k["client2"]).long()
                    sampled_cost_matrix_k = sampled_cost_matrix_k[:, indices2_k]
                    sampled_w2_k = full_w2_k[sample_indices_k["client2"]]
                    M_k_eff = len(indices2_k)
                    if verbose:
                        logger.info(f"Class {class_label}: Sampled client2 cols: {M_k_eff} from {M_k}")
                
                # Prepare marginals for this class
                a_k, b_k = prepare_ot_marginals(
                    sampled_w1_k, sampled_w2_k, N_k_eff, M_k_eff, self.eps_num
                )
                
                # Compute OT cost for this class
                ot_cost_k, _ = compute_ot_cost(
                    sampled_cost_matrix_k, a=a_k, b=b_k, reg=reg, 
                    sinkhorn_max_iter=max_iter, eps_num=self.eps_num
                )
                
                if not np.isfinite(ot_cost_k):
                    logger.warning(f"OT cost calculation failed for class {class_label}. Skipping.")
                    continue
                    
                # Determine weight for this class (average proportion of samples)
                weight_k = (N_k/N + M_k/M) / 2
                
                # Aggregate results
                total_weighted_ot_cost += ot_cost_k * weight_k
                total_weight += weight_k
                
                # Store per-class results
                per_class_results.append({
                    'class': int(class_label),
                    'ot_cost': float(ot_cost_k),
                    'weight': float(weight_k),
                    'samples_c1': int(N_k),
                    'samples_c2': int(M_k)
                })
                
                if verbose:
                    logger.info(f"Class {class_label}: OT Cost = {ot_cost_k:.4f}, Weight = {weight_k:.4f}")
            
            # Calculate final weighted OT cost
            if total_weight > self.eps_num:
                final_ot_cost = total_weighted_ot_cost / total_weight
            else:
                logger.warning("No valid classes for OT calculation.")
                final_ot_cost = np.nan
                
            # Store results
            self.results['ot_cost'] = final_ot_cost
            self.results['weighting_used'] = weighting_type_str
            self.results['within_class_only'] = True
            self.results['shared_classes'] = shared_classes
            self.results['per_class_results'] = per_class_results
            
            if verbose:
                logger.info(f"Within-Class Feature-Error OT Cost: {final_ot_cost:.4f}")
                
        else:
            # Standard Feature-Error OT (original implementation)
            # This is the existing code path for when within_class_only is False
            
            # Validate sample counts and get sampling indices for OT calculation
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

            # --- Calculate Cost Matrix (using full data) ---
            is_segmentation = p1_prob is None and p2_prob is None
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
            self.results['within_class_only'] = False
            
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
        Calculate direct OT similarity with optional within-class matching.
        """
        # --- Common setup ---
        self._reset_results()
        
        # Extract parameters
        common_params = self._extract_params(params)
        
        # Store configuration in results
        self._store_config_in_results(common_params)
        
        # Process inputs
        required_keys = ['h']
        if common_params['within_class_only'] or common_params['use_label_hellinger']:
            required_keys.append('y')
        if common_params['use_loss_weighting']:
            required_keys.append('weights')

        proc_data1, proc_data2 = self._preprocess_input(data1, data2, required_keys)
        
        if proc_data1 is None or proc_data2 is None:
            self._handle_preprocessing_failure(common_params)
            return
            
        # Extract all data
        h1, h2, y1, y2, w1, w2, p1_prob, p2_prob, N, M = self._extract_data(proc_data1, proc_data2)
        
        # Check for empty datasets
        if N == 0 or M == 0:
            self._handle_empty_datasets(common_params)
            return
                
        # Branch based on calculation mode
        if common_params['within_class_only']:
            self._calculate_similarity_within_class(h1, h2, y1, y2, w1, w2, N, M, common_params)
        else:
            self._calculate_similarity_standard(h1, h2, y1, y2, w1, w2, N, M, common_params)
            
    def _extract_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and return common parameters."""
        return {
            'verbose': params.get('verbose', False),
            'normalize_activations': params.get('normalize_activations', True),
            'normalize_cost': params.get('normalize_cost', True),
            'distance_method': params.get('distance_method', 'euclidean'),
            'use_loss_weighting': params.get('use_loss_weighting', False),
            'use_label_hellinger': params.get('use_label_hellinger', True),
            'feature_weight': params.get('feature_weight', 2.0),
            'label_weight': params.get('label_weight', 1.0),
            'compress_vectors': params.get('compress_vectors', True),
            'compression_threshold': params.get('compression_threshold', 10),
            'compression_ratio': params.get('compression_ratio', 5),
            'reg': params.get('reg', DEFAULT_OT_REG),
            'max_iter': params.get('max_iter', DEFAULT_OT_MAX_ITER),
            'min_samples_threshold': params.get('min_samples', 20),
            'max_samples_threshold': params.get('max_samples', 900),
            'within_class_only': params.get('within_class_only', False),
        }
        
    def _store_config_in_results(self, params: Dict[str, Any]) -> None:
        """Store configuration in results dictionary."""
        self.results['feature_distance_method'] = params['distance_method']
        self.results['label_hellinger_used'] = params['use_label_hellinger']
        self.results['feature_weight'] = params['feature_weight']
        self.results['label_hellinger_weight'] = params['label_weight']
        self.results['within_class_only'] = params['within_class_only']
        
    def _handle_preprocessing_failure(self, params: Dict[str, Any]) -> None:
        """Handle preprocessing failure."""
        missing_keys = "neural network activations ('h')"
        if params['within_class_only']:
            missing_keys += " and labels ('y')"
        elif params['use_label_hellinger']:
            missing_keys += " and labels ('y' for Hellinger)"
        logger.warning(f"DirectOT calculation requires {missing_keys}. Preprocessing failed or data missing. Skipping.")
        weight_type = "Loss" if params['use_loss_weighting'] else "Uniform"
        self.results['weighting_used'] = weight_type
        
    def _extract_data(self, proc_data1: Dict[str, torch.Tensor], proc_data2: Dict[str, torch.Tensor]) -> Tuple:
        """Extract and return data from processed inputs."""
        h1 = proc_data1['h']  
        h2 = proc_data2['h']  
        y1 = proc_data1.get('y')
        y2 = proc_data2.get('y')
        w1 = proc_data1.get('weights')
        w2 = proc_data2.get('weights')
        p1_prob = proc_data1.get('p_prob')
        p2_prob = proc_data2.get('p_prob')
        N, M = h1.shape[0], h2.shape[0]
        
        return h1, h2, y1, y2, w1, w2, p1_prob, p2_prob, N, M
        
    def _handle_empty_datasets(self, params: Dict[str, Any]) -> None:
        """Handle case where one or both datasets are empty."""
        logger.warning("DirectOT: One or both clients have zero samples. OT cost is 0.")
        self.results['direct_ot_cost'] = 0.0
        weight_type = "Loss" if params['use_loss_weighting'] else "Uniform"
        self.results['weighting_used'] = weight_type
            
    def _calculate_similarity_within_class(self, h1, h2, y1, y2, w1, w2, N, M, params):
        """Calculate similarity with within-class matching."""
        verbose = params['verbose']
        
        # Store per-class results
        per_class_results = []
        total_weighted_ot_cost = 0.0
        total_weight = 0.0
        
        # Ensure labels are available
        if y1 is None or y2 is None:
            logger.warning("Within-class matching requires labels for both clients. Skipping.")
            self.results['weighting_used'] = 'Loss' if params['use_loss_weighting'] else 'Uniform'
            return
            
        # Convert to numpy for easier handling
        y1_np = y1.cpu().numpy()
        y2_np = y2.cpu().numpy()
        
        # Find unique classes in both clients
        unique_classes1 = set(np.unique(y1_np))
        unique_classes2 = set(np.unique(y2_np))
        shared_classes = sorted(unique_classes1.intersection(unique_classes2))
        
        if not shared_classes:
            logger.warning("No shared classes between clients. Cannot compute within-class OT.")
            self.results['direct_ot_cost'] = np.nan
            self.results['weighting_used'] = 'Loss' if params['use_loss_weighting'] else 'Uniform'
            self.results['shared_classes'] = []
            return
            
        if verbose:
            logger.info(f"Processing {len(shared_classes)} shared classes for within-class matching")
            
        # Process each class independently
        for class_label in shared_classes:
            class_result = self._process_single_class(h1, h2, y1, y2, w1, w2, N, M, 
                                                class_label, params)
            
            if class_result is not None:
                # Unpack results
                ot_cost_k, N_k, M_k = class_result
                
                # Determine weight for this class (average proportion of samples)
                weight_k = (N_k/N + M_k/M) / 2
                
                # Aggregate results
                total_weighted_ot_cost += ot_cost_k * weight_k
                total_weight += weight_k
                
                # Store per-class results with Hellinger distance if available
                class_result_dict = {
                    'class': int(class_label),
                    'ot_cost': float(ot_cost_k),
                    'weight': float(weight_k),
                    'samples_c1': int(N_k),
                    'samples_c2': int(M_k)
                }
                
                per_class_results.append(class_result_dict)
                
                if verbose:
                    logger.info(f"Class {class_label}: OT Cost = {ot_cost_k:.4f}, Weight = {weight_k:.4f}")
        
        # Calculate final weighted OT cost
        if total_weight > self.eps_num:
            final_ot_cost = total_weighted_ot_cost / total_weight
        else:
            logger.warning("No valid classes for OT calculation.")
            final_ot_cost = np.nan
            
        # Store results
        weighting_type_str = "Loss-Weighted" if params['use_loss_weighting'] else "Uniform"
        self.results['direct_ot_cost'] = final_ot_cost
        self.results['weighting_used'] = weighting_type_str
        self.results['shared_classes'] = shared_classes
        self.results['per_class_results'] = per_class_results
        
        if verbose:
            logger.info(f"Within-Class DirectOT Cost: {final_ot_cost:.4f}")
            if params['use_label_hellinger']:
                logger.info("Within-Class Hellinger distances contributed to cost calculation")

    def _process_single_class(self, h1, h2, y1, y2, w1, w2, N, M, class_label, params):
        """Process a single class for within-class OT calculation."""
        verbose = params['verbose']
        min_samples_threshold = params['min_samples_threshold']
        
        # Find indices for this class
        idx1_k = torch.where(y1 == class_label)[0]
        idx2_k = torch.where(y2 == class_label)[0]
        
        N_k = len(idx1_k)
        M_k = len(idx2_k)
        
        # Check if we have enough samples for this class
        if N_k < min_samples_threshold or M_k < min_samples_threshold:
            if verbose:
                logger.info(f"Class {class_label}: Not enough samples (C1:{N_k}, C2:{M_k}). Skipping.")
            return None
            
        # Extract class-specific data
        h1_k = h1[idx1_k]
        h2_k = h2[idx2_k]
        
        # Extract weights if using loss weighting
        w1_k = w1[idx1_k] if w1 is not None else None
        w2_k = w2[idx2_k] if w2 is not None else None
        
        # Normalize activations class-wise if requested
        h1_k_norm, h2_k_norm = self._normalize_activations(h1_k, h2_k, params['normalize_activations'])
        
        # Compute feature cost matrix for this class
        feature_cost_matrix_k, max_feature_cost_k = self._calculate_feature_cost(
            h1_k_norm, h2_k_norm, params['distance_method'], params['normalize_activations']
        )
        
        if feature_cost_matrix_k is None:
            logger.warning(f"Failed to compute feature cost matrix for class {class_label}.")
            return None
            
        # Normalize feature cost matrix if requested
        feature_cost_matrix_k = normalize_cost_matrix(
            feature_cost_matrix_k, max_feature_cost_k, params['normalize_cost'], self.eps_num
        )
        
        # Initialize the final cost matrix with feature costs
        cost_matrix_k = feature_cost_matrix_k.clone()
        
        # Calculate Hellinger distance if requested
        # This is the fix for adding Hellinger in within-class mode
        if params['use_label_hellinger']:
            hellinger_dist_k = self._calculate_hellinger_for_class(
                h1_k, h2_k, params['compress_vectors'], 
                params['compression_threshold'], params['compression_ratio'],
                verbose, class_label
            )
            
            # If valid Hellinger distance, combine with feature cost
            if np.isfinite(hellinger_dist_k):
                # Normalize weights
                total_weight = params['feature_weight'] + params['label_weight']
                norm_feature_weight = params['feature_weight'] / total_weight
                norm_label_weight = params['label_weight'] / total_weight
                
                # Combine costs - multiply by whole matrix to maintain tensor shape
                cost_matrix_k = (norm_feature_weight * cost_matrix_k + 
                                norm_label_weight * hellinger_dist_k)
                if verbose:
                    logger.info(f"Class {class_label}: Combined cost matrix using feature cost (weight {norm_feature_weight:.2f}) "
                            f"and Hellinger distance {hellinger_dist_k:.4f} (weight {norm_label_weight:.2f})")
        
        # Prepare weights for this class
        weighting_type_str, full_w1_k, full_w2_k = self._prepare_weights(
            w1_k, w2_k, N_k, M_k, params['use_loss_weighting']
        )
        
        # Check if sampling is needed within this class
        features_dict_k = {
            "client1": h1_k.cpu().numpy(),
            "client2": h2_k.cpu().numpy()
        }
        _, sample_indices_k = validate_samples_for_ot(
            features_dict_k, 2, params['max_samples_threshold']
        )
        
        # Sample and prepare for OT
        sampled_cost_matrix_k, a_k, b_k, N_k_eff, M_k_eff = self._apply_sampling_and_prepare_marginals(
            cost_matrix_k, full_w1_k, full_w2_k, sample_indices_k, N_k, M_k, verbose
        )
        # Compute OT cost for this class
        ot_cost_k, _ = compute_ot_cost(
            sampled_cost_matrix_k, a=a_k, b=b_k, reg=params['reg'], 
            sinkhorn_max_iter=params['max_iter'], eps_num=self.eps_num
        )
        
        if not np.isfinite(ot_cost_k):
            logger.warning(f"OT cost calculation failed for class {class_label}. Skipping.")
            return None
            
        return ot_cost_k, N_k, M_k

    def _calculate_hellinger_for_class(self, h1_k, h2_k, compress_vectors, 
                                    compression_threshold, compression_ratio, verbose, class_label=None):
        """Calculate Hellinger distance for a single class."""
        try:
            # Convert to numpy
            h1_k_np = h1_k.cpu().numpy()
            h2_k_np = h2_k.cpu().numpy()
            
            # Check if compression is needed
            vector_dim = h1_k_np.shape[1]
            was_compressed = False
            if compress_vectors and vector_dim >= compression_threshold:
                if verbose:
                    logger.info(f"Class {class_label}: Compressing vectors from dimension {vector_dim} for Hellinger calculation")
                h1_comp, h2_comp = self._compress_vectors(h1_k_np, h2_k_np, compression_ratio)
                was_compressed = True
            else:
                h1_comp, h2_comp = h1_k_np, h2_k_np
            
            # Calculate distribution parameters
            mu_1, sigma_1 = self._get_normal_params(h1_comp)
            mu_2, sigma_2 = self._get_normal_params(h2_comp)
            
            # Calculate Hellinger distance
            hellinger_dist = self._hellinger_distance(mu_1, sigma_1, mu_2, sigma_2)
            
            if hellinger_dist is None:
                if verbose:
                    logger.warning(f"Class {class_label}: Hellinger distance calculation failed. Using default value 0.5.")
                return 0.5  # Default neutral value
                
            if verbose:
                compression_str = f"(compressed from {vector_dim})" if was_compressed else ""
                sample_size_str = f"(samples: C1={len(h1_k_np)}, C2={len(h2_k_np)})"
                logger.info(f"Class {class_label}: Hellinger distance = {hellinger_dist:.4f} {compression_str} {sample_size_str}")
                
                # Only log distribution parameters in very verbose mode to avoid cluttering
                if verbose > 1:  # assuming verbose could be a level, if not just remove this condition
                    # Log mean and trace of covariance as a simple summary
                    mu1_norm = np.linalg.norm(mu_1)
                    mu2_norm = np.linalg.norm(mu_2)
                    sigma1_trace = np.trace(sigma_1)
                    sigma2_trace = np.trace(sigma_2)
                    logger.info(f"  Class {class_label} distribution stats: ||μ1||={mu1_norm:.3f}, ||μ2||={mu2_norm:.3f}, "
                            f"tr(Σ1)={sigma1_trace:.3f}, tr(Σ2)={sigma2_trace:.3f}")
            
            return hellinger_dist
        
        except Exception as e:
            logger.warning(f"Class {class_label}: Hellinger distance calculation error: {e}")
            return 0.5  # Default neutral value
        
    def _calculate_similarity_standard(self, h1, h2, y1, y2, w1, w2, N, M, params):
        """Calculate similarity using standard (non-within-class) approach."""
        verbose = params['verbose']
        
        # Validate sample counts for OT calculation
        features_dict = {
            "client1": h1.cpu().numpy(),
            "client2": h2.cpu().numpy()
        }
        all_sufficient, sample_indices = validate_samples_for_ot(
            features_dict, params['min_samples_threshold'], params['max_samples_threshold'])

        if not all_sufficient:
            logger.warning(f"DirectOT: One or both clients have insufficient samples (min={params['min_samples_threshold']}). Skipping.")
            weight_type = "Loss" if params['use_loss_weighting'] else "Uniform"
            self.results['weighting_used'] = weight_type
            return
        
        # --- Feature Cost Matrix Calculation ---
        h1_norm, h2_norm = self._normalize_activations(h1, h2, params['normalize_activations'])
        
        # Compute feature cost matrix
        feature_cost_matrix, max_feature_cost = self._calculate_feature_cost(
            h1_norm, h2_norm, params['distance_method'], params['normalize_activations']
        )
        
        if feature_cost_matrix is None:
            logger.warning(f"Failed to compute feature cost matrix with method: {params['distance_method']}")
            return
            
        # Normalize feature cost matrix if requested
        feature_cost_matrix = normalize_cost_matrix(
            feature_cost_matrix, max_feature_cost, params['normalize_cost'], self.eps_num
        )
            
        # Store the feature cost matrix
        self.cost_matrices['feature_cost'] = feature_cost_matrix.cpu().numpy()
        
        # --- Label Cost Matrix Calculation ---
        label_cost_matrix = self._calculate_label_cost_matrix(
            h1, h2, y1, y2, N, M, feature_cost_matrix, params
        )
        
        # --- Combine Feature and Label Costs ---
        cost_matrix = self._combine_cost_matrices(
            feature_cost_matrix, label_cost_matrix, 
            params['use_label_hellinger'], params['feature_weight'], params['label_weight']
        )
        
        # Store the final full cost matrix used for OT
        self.cost_matrices['direct_ot'] = cost_matrix.cpu().numpy()
        
        # --- Prepare weights for OT ---
        weight_type, full_w1, full_w2 = self._prepare_weights(
            w1, w2, N, M, params['use_loss_weighting']
        )
        
        self.results['weighting_used'] = weight_type
        
        # --- Apply sampling and prepare marginals ---
        sampled_cost_matrix, a, b, N_eff, M_eff = self._apply_sampling_and_prepare_marginals(
            cost_matrix, full_w1, full_w2, sample_indices, N, M, verbose
        )
        
        # --- Compute OT Cost ---
        ot_cost, transport_plan = compute_ot_cost(
            sampled_cost_matrix, a=a, b=b, reg=params['reg'], 
            sinkhorn_max_iter=params['max_iter'], eps_num=self.eps_num
        )
        
        self.results['direct_ot_cost'] = ot_cost
        self.results['transport_plan'] = transport_plan
        
        if verbose:
            if np.isfinite(ot_cost):
                logger.info(f"  DirectOT Cost ({weight_type} weights): {ot_cost:.4f}")
            else:
                logger.info(f"  DirectOT Cost ({weight_type} weights): Failed")
                
            if params['use_label_hellinger']:
                logger.info(f"  Label Hellinger distances by class pairs:")
                for (label1, label2), dist in self.results['label_costs']:
                    logger.info(f"    Labels ({label1},{label2}): {dist:.4f}")
            else:
                logger.info(f"  Using only feature distances ({params['distance_method']})")

    def _normalize_activations(self, h1, h2, normalize_activations):
        """Normalize activations if requested."""
        if normalize_activations:
            h1_norm = F.normalize(h1.float(), p=2, dim=1, eps=self.eps_num)
            h2_norm = F.normalize(h2.float(), p=2, dim=1, eps=self.eps_num)
        else:
            h1_norm = h1.float()
            h2_norm = h2.float()
        return h1_norm, h2_norm

    def _calculate_feature_cost(self, h1_norm, h2_norm, distance_method, normalize_activations):
        """Calculate feature cost matrix based on distance method."""
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
            
        return feature_cost_matrix, max_feature_cost

    def _calculate_label_cost_matrix(self, h1, h2, y1, y2, N, M, feature_cost_matrix, params):
        """Calculate label cost matrix using Hellinger distance."""
        use_label_hellinger = params['use_label_hellinger']
        verbose = params['verbose']
        
        # Initialize label cost matrix with same shape as feature cost matrix
        label_cost_matrix = torch.zeros_like(feature_cost_matrix)
        
        if use_label_hellinger and y1 is not None and y2 is not None:
            try:                
                # Convert to numpy for easier handling
                h1_np = h1.cpu().numpy()
                h2_np = h2.cpu().numpy()
                y1_np = y1.cpu().numpy()
                y2_np = y2.cpu().numpy()
                
                # Check if compression is needed for high-dimensional vectors
                vector_dim = h1_np.shape[1]
                if params['compress_vectors'] and vector_dim > params['compression_threshold']:
                    if verbose:
                        logger.info(f"Compressing vectors from dimension {vector_dim} for Hellinger calculation")
                    h1_comp, h2_comp = self._compress_vectors(h1_np, h2_np, params['compression_ratio'])
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
                        if len(indices1) >= 2 and len(indices2) >= 2:  # Need at least 2 samples for estimation
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
            # Fill with neutral value
            label_cost_matrix.fill_(0.5)
            self.cost_matrices['label_cost'] = label_cost_matrix.cpu().numpy()
            
        return label_cost_matrix

    def _combine_cost_matrices(self, feature_cost_matrix, label_cost_matrix, use_label_hellinger, feature_weight, label_weight):
        """Combine feature and label cost matrices."""
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
            return combined_cost_matrix
        else:
            # Use only feature cost if label cost is not used
            self.cost_matrices['combined_cost'] = self.cost_matrices['feature_cost']
            return feature_cost_matrix

    def _prepare_weights(self, w1, w2, N, M, use_loss_weighting):
        """Prepare weights for OT calculation."""
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
            
        return weight_type, full_w1, full_w2

    def _apply_sampling_and_prepare_marginals(self, cost_matrix, full_w1, full_w2, sample_indices, N, M, verbose):
        """Apply sampling to cost matrix and weights, then prepare marginals."""
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
        
        # Prepare marginals with the sampled weights
        a, b = prepare_ot_marginals(sampled_w1, sampled_w2, N_eff, M_eff, self.eps_num)
        
        return sampled_cost_matrix, a, b, N_eff, M_eff

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