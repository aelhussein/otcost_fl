# ot_utils.py
from typing import Optional, Tuple, Dict, Any, List, Union
import numpy as np
import torch
import warnings
import ot
from scipy.stats import wasserstein_distance
from sklearn.cluster import KMeans
import logging

# --- Constants ---
DEFAULT_OT_REG = 0.001
DEFAULT_OT_MAX_ITER = 2000
DEFAULT_EPS = 1e-10


# --- OT Computation Wrapper ---
def compute_ot_cost(
    cost_matrix: Union[torch.Tensor, np.ndarray],
    a: Optional[np.ndarray] = None,
    b: Optional[np.ndarray] = None,
    reg: float = DEFAULT_OT_REG,
    sinkhorn_thresh: float = 1e-3,
    sinkhorn_max_iter: int = DEFAULT_OT_MAX_ITER,
    eps_num: float = DEFAULT_EPS
) -> Tuple[float, Optional[np.ndarray]]:
    """
    Computes OT cost using Sinkhorn algorithm from POT library.
    Handles input validation, NaN/Inf values, and marginal normalization.

    Args:
        cost_matrix: The (N, M) cost matrix (Tensor or Numpy array).
        a: Source marginal weights (N,). Defaults to uniform.
        b: Target marginal weights (M,). Defaults to uniform.
        reg: Entropy regularization term for Sinkhorn.
        sinkhorn_thresh: Stop threshold for Sinkhorn iterations.
        sinkhorn_max_iter: Max iterations for Sinkhorn.
        eps_num: Small epsilon for numerical stability checks.

    Returns:
        Tuple containing:
            - float: Computed OT cost (np.nan if failed).
            - np.ndarray or None: Transport plan (Gs).
    """
    if cost_matrix is None:
        warnings.warn("OT computation skipped: Cost matrix is None.")
        return np.nan, None

    # Ensure cost matrix is a numpy float64 array
    if isinstance(cost_matrix, torch.Tensor):
        cost_matrix_np = cost_matrix.detach().cpu().numpy().astype(np.float64)
    elif isinstance(cost_matrix, np.ndarray):
        cost_matrix_np = cost_matrix.astype(np.float64)
    else:
        try:
            cost_matrix_np = np.array(cost_matrix, dtype=np.float64)
        except Exception as e:
            warnings.warn(f"Could not convert cost matrix to numpy array: {e}")
            return np.nan, None

    if cost_matrix_np.size == 0:
        N, M = cost_matrix_np.shape
        return 0.0, np.zeros((N, M))

    N, M = cost_matrix_np.shape
    if N == 0 or M == 0:
        return 0.0, np.zeros((N, M))

    # Handle non-finite values in cost matrix
    if not np.all(np.isfinite(cost_matrix_np)):
        max_finite_cost = np.nanmax(cost_matrix_np[np.isfinite(cost_matrix_np)])
        replacement_val = 1e6
        if np.isfinite(max_finite_cost):
            replacement_val = max(1.0, abs(max_finite_cost)) * 10.0
        cost_matrix_np[~np.isfinite(cost_matrix_np)] = replacement_val
        warnings.warn(f"NaN/Inf detected in cost matrix. Replaced non-finite values with {replacement_val:.2e}.")

    # Prepare Marginals a and b
    if a is None:
        a = np.ones((N,), dtype=np.float64) / N
    else:
        a = np.asarray(a, dtype=np.float64)
        if not np.all(np.isfinite(a)):
            a = np.ones_like(a) / max(1, len(a)); warnings.warn("NaN/Inf in marginal 'a'. Using uniform.")
        sum_a = a.sum()
        if sum_a <= eps_num:
            a = np.ones_like(a) / max(1, len(a)); warnings.warn("Marginal 'a' sums to zero or less. Using uniform.")
        elif not np.isclose(sum_a, 1.0):
            a /= sum_a

    if b is None:
        b = np.ones((M,), dtype=np.float64) / M
    else:
        b = np.asarray(b, dtype=np.float64)
        if not np.all(np.isfinite(b)):
            b = np.ones_like(b) / max(1, len(b)); warnings.warn("NaN/Inf in marginal 'b'. Using uniform.")
        sum_b = b.sum()
        if sum_b <= eps_num:
            b = np.ones_like(b) / max(1, len(b)); warnings.warn("Marginal 'b' sums to zero or less. Using uniform.")
        elif not np.isclose(sum_b, 1.0):
            b /= sum_b

    # Compute OT using POT
    Gs = None
    ot_cost = np.nan
    try:
        cost_matrix_np_cont = np.ascontiguousarray(cost_matrix_np)

        # Try stabilized Sinkhorn first
        Gs = ot.sinkhorn(a, b, cost_matrix_np_cont, reg=reg, stopThr=sinkhorn_thresh,
                         numItermax=sinkhorn_max_iter, method='sinkhorn_stabilized',
                         warn=False, verbose=False)

        if Gs is None or np.any(np.isnan(Gs)):
            # Fallback to standard Sinkhorn
            if Gs is None: warnings.warn("Stabilized Sinkhorn failed. Trying standard Sinkhorn.", stacklevel=2)
            else: warnings.warn("Stabilized Sinkhorn resulted in NaN plan. Trying standard Sinkhorn.", stacklevel=2)
            Gs = ot.sinkhorn(a, b, cost_matrix_np_cont, reg=reg, stopThr=sinkhorn_thresh,
                             numItermax=sinkhorn_max_iter, method='sinkhorn',
                             warn=False, verbose=False)

        if Gs is None or np.any(np.isnan(Gs)):
            warnings.warn("Sinkhorn computation failed (both stabilized and standard) or resulted in NaN plan.", stacklevel=2)
            Gs = None # Ensure Gs is None if calculation failed
        else:
            # Calculate OT cost
            ot_cost = np.sum(Gs * cost_matrix_np)
            if not np.isfinite(ot_cost):
                warnings.warn(f"Calculated OT cost is not finite ({ot_cost}). Returning NaN.", stacklevel=2)
                ot_cost = np.nan

    except Exception as e:
        warnings.warn(f"Error during OT computation (ot.sinkhorn): {e}", stacklevel=2)
        ot_cost = np.nan
        Gs = None

    return float(ot_cost), Gs

# --- Distance Metrics ---

def pairwise_euclidean_sq(X: Optional[torch.Tensor], Y: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """ Calculates pairwise squared Euclidean distance: C[i,j] = ||X[i] - Y[j]||_2^2 """
    if X is None or Y is None or X.ndim != 2 or Y.ndim != 2 or X.shape[1] != Y.shape[1]:
        warnings.warn("pairwise_euclidean_sq: Invalid input shapes or None input.")
        return None
    try:
        # Using cdist is generally numerically stable and clear
        dist_sq = torch.cdist(X.float(), Y.float(), p=2).pow(2)
        return dist_sq
    except Exception as e:
        warnings.warn(f"Error calculating squared Euclidean distance: {e}")
        return None

def calculate_label_emd(y1: Optional[torch.Tensor], y2: Optional[torch.Tensor], num_classes: int) -> float:
    """Calculates Earth Mover's Distance between label distributions."""
    if y1 is None or y2 is None: return np.nan
    try:
        y1_tensor = y1 if isinstance(y1, torch.Tensor) else torch.tensor(y1)
        y2_tensor = y2 if isinstance(y2, torch.Tensor) else torch.tensor(y2)
        n1, n2 = y1_tensor.numel(), y2_tensor.numel()
    except Exception as e:
        warnings.warn(f"Error processing labels for EMD: {e}")
        return np.nan

    if n1 == 0 and n2 == 0: return 0.0
    if n1 == 0 or n2 == 0: return float(max(0.0, float(num_classes - 1)))

    y1_np = y1_tensor.detach().cpu().numpy().astype(int)
    y2_np = y2_tensor.detach().cpu().numpy().astype(int)
    class_values = np.arange(num_classes)

    hist1, _ = np.histogram(y1_np, bins=np.arange(num_classes + 1), density=False)
    hist2, _ = np.histogram(y2_np, bins=np.arange(num_classes + 1), density=False)

    sum1 = hist1.sum(); sum2 = hist2.sum()
    if sum1 == 0 or sum2 == 0: return float(max(0.0, float(num_classes - 1)))

    hist1_norm = hist1 / sum1; hist2_norm = hist2 / sum2;

    try:
        return float(wasserstein_distance(class_values, class_values, u_weights=hist1_norm, v_weights=hist2_norm))
    except ValueError as e:
        warnings.warn(f"Wasserstein distance calculation failed: {e}")
        return np.nan

def compute_anchors(
    h: Optional[torch.Tensor],
    y: Optional[torch.Tensor],
    num_classes: int,
    params: Dict[str, Any]
) -> Optional[List[Dict[str, Any]]]:
    """
    Computes fixed anchors Z per class.
    Returns a list of dictionaries, each containing:
        {'anchor': torch.Tensor, 'class_label': int, 'cluster_size': int}

    - If method='kmeans', runs KMeans per class. 'num_anchors' specifies k per class.
      'cluster_size' is the number of points assigned to that anchor's cluster.
    - If method='class_means', computes the mean per class.
      'cluster_size' is the total number of points in that class.
    """
    # ... (Initial checks for h, y remain the same) ...
    if h is None or y is None or h.shape[0] != y.shape[0] or h.ndim != 2 or y.ndim != 1:
        warnings.warn("Invalid input for anchor computation.")
        return None
    if h.shape[0] == 0: return None
    method = params.get('anchor_method', 'kmeans')
    k_per_class = params.get('num_anchors', 5) # k per class for kmeans
    kmeans_max_iter = params.get('kmeans_max_iter', 100)
    verbose = params.get('verbose', False)

    all_anchor_info = []
    h_cpu = h.detach().cpu()
    y_cpu = y.detach().cpu()

    for c in range(num_classes):
        class_mask = (y_cpu == c)
        h_class = h_cpu[class_mask]
        n_class = h_class.shape[0]

        if n_class == 0:
            if verbose: warnings.warn(f"Class {c}: No samples. Skipping anchors.")
            continue

        class_anchors_data = [] # List to store dicts for this class

        if method == 'kmeans':
            current_k = k_per_class
            use_fallback = False
            if n_class < current_k:
                warnings.warn(f"Class {c}: Samples ({n_class}) < k ({current_k}). Using class mean.")
                use_fallback = True

            if use_fallback:
                 anchor_mean = h_class.mean(dim=0, keepdim=True).float()
                 class_anchors_data.append({'anchor': anchor_mean.squeeze(0), 'class_label': c, 'cluster_size': n_class})
            else:
                try:
                    h_class_np = h_class.numpy()
                    kmeans = KMeans(n_clusters=current_k, n_init='auto',
                                    max_iter=kmeans_max_iter, random_state=42)
                    # Get cluster assignments (labels_) and centers (cluster_centers_)
                    cluster_assignments = kmeans.fit_predict(h_class_np)
                    cluster_centers = kmeans.cluster_centers_

                    for cluster_idx in range(current_k):
                        center_coord = torch.from_numpy(cluster_centers[cluster_idx]).float()
                        # Count points assigned to this cluster
                        points_in_cluster = np.sum(cluster_assignments == cluster_idx)
                        if points_in_cluster > 0: # Only add anchor if it represents points
                             class_anchors_data.append({
                                 'anchor': center_coord,
                                 'class_label': c,
                                 'cluster_size': int(points_in_cluster)
                             })
                        elif verbose:
                             warnings.warn(f"Class {c}, K-Means cluster {cluster_idx} has 0 points assigned.")

                    if verbose: print(f"    Class {c}: Computed {len(class_anchors_data)} k-Means anchors.")

                except Exception as e:
                    warnings.warn(f"KMeans failed for class {c}: {e}. Using class mean fallback.")
                    anchor_mean = h_class.mean(dim=0, keepdim=True).float()
                    class_anchors_data.append({'anchor': anchor_mean.squeeze(0), 'class_label': c, 'cluster_size': n_class})

        elif method == 'class_means':
            if n_class > 0:
                 anchor_mean = h_class.mean(dim=0, keepdim=True).float()
                 class_anchors_data.append({'anchor': anchor_mean.squeeze(0), 'class_label': c, 'cluster_size': n_class})
                 if verbose: print(f"    Class {c}: Computed class mean anchor.")

        else: # Unknown method
            warnings.warn(f"Unknown anchor method: {method}")
            return None

        all_anchor_info.extend(class_anchors_data)

    if not all_anchor_info:
        warnings.warn("No anchors computed across any classes.")
        return None

    if verbose: print(f"  Total computed anchor info entries: {len(all_anchor_info)}")
    return all_anchor_info # Return list of dicts

def calculate_sample_loss(p_prob: Optional[torch.Tensor], y: Optional[torch.Tensor], 
                     num_classes: int, loss_eps: float = DEFAULT_EPS) -> Optional[torch.Tensor]:
    """
    Calculates per-sample cross-entropy loss, with enhanced validation for multiclass.
    
    Args:
        p_prob: Predicted probability distribution of shape [N, K]
        y: Ground truth labels of shape [N]
        num_classes: Number of classes K
        loss_eps: Small epsilon value for numerical stability
        
    Returns:
        Tensor of per-sample losses of shape [N] or None if validation fails
    """
    if p_prob is None or y is None: 
        return None
        
    if not isinstance(p_prob, torch.Tensor): 
        p_prob = torch.tensor(p_prob)
    if not isinstance(y, torch.Tensor): 
        y = torch.tensor(y)

    try:
        # Ensure tensors are on CPU and correct dtype
        p_prob = p_prob.float().cpu()
        y = y.long().cpu()
        
        # Validate shapes and fix if needed for binary case
        if p_prob.ndim == 1 and num_classes == 2:
            # Convert [N] format to [N, 2] for binary case
            p_prob_1d = p_prob.view(-1)
            p1 = p_prob_1d.clamp(min=loss_eps, max=1.0 - loss_eps)
            p0 = 1.0 - p1
            p_prob = torch.stack([p0, p1], dim=1)
        elif p_prob.ndim == 2 and p_prob.shape[1] == 1 and num_classes == 2:
            # Convert [N, 1] format to [N, 2] for binary case
            p1 = p_prob.view(-1).clamp(min=loss_eps, max=1.0 - loss_eps)
            p0 = 1.0 - p1
            p_prob = torch.stack([p0, p1], dim=1)
            
        # Final shape validation
        if y.shape[0] != p_prob.shape[0] or p_prob.ndim != 2 or p_prob.shape[1] != num_classes:
            warnings.warn(f"Loss calculation shape mismatch/invalid: P({p_prob.shape}), Y({y.shape}), K={num_classes}")
            return None
            
        # Ensure probabilities are valid (sum to 1, in [eps, 1-eps] range)
        row_sums = p_prob.sum(dim=1)
        if not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-3):
            warnings.warn(f"Probabilities don't sum to 1 (min={row_sums.min().item():.4f}, "
                         f"max={row_sums.max().item():.4f}). Normalizing.")
            p_prob = p_prob / row_sums.unsqueeze(1).clamp(min=loss_eps)
            
        # Clamp probabilities for numerical stability
        p_prob = p_prob.clamp(min=loss_eps, max=1.0 - loss_eps)
        
        # Gather predicted probability for true class
        true_class_prob = p_prob.gather(1, y.view(-1, 1)).squeeze()
        
        # Calculate cross-entropy loss: -log(p[true_class])
        loss = -torch.log(true_class_prob)
        
        # Safety check for NaN/Inf values
        if not torch.isfinite(loss).all():
            warnings.warn("Non-finite values in loss calculation. Replacing with large value.")
            loss = torch.where(torch.isfinite(loss), loss, torch.tensor(100.0, dtype=torch.float32))
            
    except Exception as e:
        warnings.warn(f"Error during loss calculation: {e}")
        return None

    return torch.relu(loss)  # Ensure non-negative loss

def compute_anchors(
    h: Optional[torch.Tensor],
    y: Optional[torch.Tensor],
    num_classes: int,
    params: Dict[str, Any]
) -> Optional[torch.Tensor]:
    """ Computes fixed anchors Z based on method specified in params. """
    if h is None or h.shape[0] == 0:
        warnings.warn("Cannot compute anchors: Input activations 'h' are None or empty.")
        return None

    method = params.get('anchor_method', 'kmeans')
    k = params.get('num_anchors', 10)
    verbose = params.get('verbose', False)

    if verbose: print(f"    Computing anchors using method: {method}, k={k}")

    if method == 'class_means':
        # NOTE: This implementation is kept simple as in the original,
        # but has limitations with missing classes. K-Means is generally more robust.
        if y is None:
            warnings.warn("Cannot compute 'class_means' anchors: Input labels 'y' are None.")
            return None
        if k != num_classes:
            warnings.warn(f"Requested {k} anchors but method is 'class_means', using num_classes={num_classes} instead.")
            k = num_classes

        anchors = []
        present_classes = torch.unique(y)
        if verbose: print(f"    Present classes: {present_classes.tolist()}")
        for c in range(num_classes):
             if c in present_classes:
                 mask = (y == c)
                 class_mean = h[mask].mean(dim=0)
                 anchors.append(class_mean)
             else:
                 if verbose: warnings.warn(f"Class {c} missing, anchor not computed for class means.")
                 # Maybe append mean of all data or nan? Current skips might misalign Z1/Z2.
        if not anchors:
            warnings.warn("No class mean anchors computed (likely no classes present?).")
            return None
        warnings.warn("'class_means' anchor method experimental due to handling of missing classes.", UserWarning)
        return torch.stack(anchors).float().cpu() if anchors else None

    elif method == 'kmeans':
        if h.shape[0] < k:
            warnings.warn(f"Number of samples ({h.shape[0]}) is less than k ({k}). Reducing k for KMeans.")
            k = max(1, h.shape[0])

        try:
            h_np = h.detach().cpu().numpy()
            kmeans = KMeans(n_clusters=k,
                            n_init='auto',
                            max_iter=params.get('kmeans_max_iter', 100),
                            random_state=42) # For reproducibility
            kmeans.fit(h_np)
            anchors_np = kmeans.cluster_centers_
            if verbose: print(f"    Computed {anchors_np.shape[0]} k-Means anchors.")
            return torch.from_numpy(anchors_np).float().cpu()
        except Exception as e:
            warnings.warn(f"KMeans anchor computation failed: {e}")
            return None
    else:
        warnings.warn(f"Unknown anchor method: {method}")
        return None

logger = logging.getLogger(__name__)

def validate_samples_for_ot(features_dict: Dict[str, np.ndarray], 
                           min_samples: int = 100,
                           max_samples: int = 900) -> Tuple[bool, Dict[str, np.ndarray]]:
    """
    Validates if clients have enough samples and samples down if exceeding maximum.
    """
    indices_to_use = {}
    all_sufficient = True
    
    for client_id, features in features_dict.items():
        n_samples = len(features)
        
        # Check minimum threshold
        if n_samples < min_samples:
            logger.info(f"Client {client_id} has insufficient samples: {n_samples} < {min_samples}")
            all_sufficient = False
            indices_to_use[client_id] = np.arange(n_samples)  # Use all available samples
        # Check maximum threshold - only sample if we're strictly over the max
        elif n_samples > max_samples:
            # Sample down to max_samples
            rng = np.random.RandomState(42)  # Fixed seed for reproducibility
            indices = rng.choice(n_samples, max_samples, replace=False)
            indices.sort()  # Keep original order
            indices_to_use[client_id] = indices
            
            # Log the sampling
            logger.info(f"Client {client_id} samples reduced: {n_samples} → {max_samples}")
        else:
            # Use all samples (already within limits)
            indices_to_use[client_id] = np.arange(n_samples)
    
    return all_sufficient, indices_to_use


def validate_samples_for_decomposed_ot(features_by_label: Dict[str, Dict[int, np.ndarray]],
                                      min_samples: int = 100,
                                      max_samples: int = 900) -> Tuple[Dict[int, bool], Dict[str, Dict[int, np.ndarray]]]:
    """
    Validates which labels have sufficient samples and samples down if exceeding maximum.
    """
    # Collect all unique labels across clients
    all_labels = set()
    for client_labels in features_by_label.values():
        all_labels.update(client_labels.keys())
    
    # Check each label
    label_validity = {}
    indices_by_client_label = {client_id: {} for client_id in features_by_label.keys()}
    
    for label in all_labels:
        valid = True
        for client_id, label_dict in features_by_label.items():
            # Check if this client has this label
            if label not in label_dict:
                valid = False
                logger.info(f"Client {client_id} is missing label {label}")
                break
            
            # Check sample count
            n_samples = len(label_dict[label])
            if n_samples < min_samples:
                valid = False
                logger.info(f"Client {client_id} has insufficient samples for label {label}: "
                           f"{n_samples} < {min_samples}")
            
            # Store indices for this label (sampling if needed)
            # Only sample if strictly greater than max_samples
            if n_samples > max_samples:
                # Sample down to max_samples
                rng = np.random.RandomState(42 + n_samples)  # Seed varies by label for diversity
                indices = rng.choice(n_samples, max_samples, replace=False)
                indices.sort()  # Keep original order
                logger.info(f"Label {label}, client {client_id} samples reduced: {n_samples} → {max_samples}")
            else:
                # Use all samples
                indices = np.arange(n_samples)
                
            indices_by_client_label[client_id][label] = indices
        
        label_validity[label] = valid
    
    return label_validity, indices_by_client_label

def apply_sampling_to_data(tensors_dict: Dict[str, Optional[torch.Tensor]], 
                          indices_dict: Dict[str, np.ndarray]) -> Dict[str, Optional[torch.Tensor]]:
    """
    Apply sampling indices to multiple tensors consistently.
    
    Args:
        tensors_dict: Dictionary of tensors to sample (h, y, p_prob, weights, etc.)
        indices_dict: Dictionary mapping keys to indices arrays
        
    Returns:
        Dictionary with sampled tensors
    """
    sampled_tensors = {}
    for name, tensor in tensors_dict.items():
        if tensor is None:
            sampled_tensors[name] = None
            continue
            
        if name in indices_dict:
            indices = indices_dict[name]
            # Convert indices to tensor for indexing
            if isinstance(indices, np.ndarray):
                indices_tensor = torch.from_numpy(indices).long()
            else:
                indices_tensor = torch.tensor(indices, dtype=torch.long)
                
            # Apply sampling
            sampled_tensors[name] = tensor[indices_tensor]
        else:
            # No sampling needed for this tensor
            sampled_tensors[name] = tensor
            
    return sampled_tensors