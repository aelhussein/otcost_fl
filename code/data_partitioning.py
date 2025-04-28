"""
Functions for partitioning dataset indices among clients based on different strategies.
Handles Dirichlet-based label skew and IID partitioning.
Operates primarily on NumPy arrays or basic counts. Streamlined version.
"""
import numpy as np
import random
from typing import List, Dict, Any, Optional, Callable

# =============================================================================
# == Partitioning Functions ==
# =============================================================================

def partition_iid_indices(num_samples: int,
                          num_clients: int,
                          seed: int = 42) -> Dict[int, List[int]]:
    """Partitions dataset indices equally and randomly (IID)."""
    if num_samples == 0: return {i: [] for i in range(num_clients)}

    indices = np.arange(num_samples)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    split_indices = np.array_split(indices, num_clients)
    client_indices = {i: split_indices[i].tolist() for i in range(num_clients)}

    print(f"IID partitioning: {num_samples} samples -> {num_clients} clients.")
    return client_indices


def partition_pre_defined(client_num: int, **kwargs) -> int:
    """Placeholder strategy for pre-split data. Returns the client number."""
    # No print statement needed, it's just a passthrough
    return client_num


def partition_iid_indices(n_samples: int, num_clients: int, seed: int) -> Dict[int, List[int]]:
    """Placeholder for IID partitioning - required for fallback."""
    # (This function needs to be defined elsewhere if used)
    # NO PRINTS IN THIS VERSION
    indices = list(range(n_samples))
    random.Random(seed).shuffle(indices)
    split_size = n_samples // num_clients
    client_indices = {}
    for i in range(num_clients):
        start = i * split_size
        end = (i + 1) * split_size if i < num_clients - 1 else n_samples
        client_indices[i] = indices[start:end]
    return client_indices

def partition_dirichlet_indices(
    labels_np: np.ndarray,
    num_clients: int,
    alpha: float,
    seed: int = 42,
    **kwargs: Any,
) -> Dict[int, List[int]]:
    """
    Dirichlet label-skew partition (Hsu et al., 2019) with optional quantity-skew removal.
    """
    n_samples = len(labels_np)
    if n_samples == 0:
        return {i: [] for i in range(num_clients)}

    rng = np.random.RandomState(seed)
    py_rng = random.Random(seed + 1)

    classes, class_counts = np.unique(labels_np, return_counts=True)
    n_classes = len(classes)

    if alpha >= 1e6 or n_classes <= 1:
        all_idx = list(range(n_samples))
        py_rng.shuffle(all_idx)
        return {i: all_idx[i::num_clients] for i in range(num_clients)}

    # 1) one Dirichlet vector PER CLIENT  →  rows sum to 1
    label_dist_per_client = rng.dirichlet([alpha] * n_classes, size=num_clients)

    # Pre-shuffle indices per class
    idx_by_class = {
        cls: py_rng.sample(np.where(labels_np == cls)[0].tolist(), cnt)
        for cls, cnt in zip(classes, class_counts)
    }

    client_indices: Dict[int, List[int]] = {i: [] for i in range(num_clients)}

    for c_idx, cls in enumerate(classes):
        pool = idx_by_class[cls]

        # --- NEW: robust normalisation for column vector ---
        pvals = label_dist_per_client[:, c_idx]
        col_sum = pvals.sum()
        if col_sum < 1e-12:            # pathological: no weight for this class anywhere
            pvals = np.full(num_clients, 1 / num_clients)
        else:
            pvals = pvals / col_sum
        # ----------------------------------------------------

        counts = rng.multinomial(len(pool), pvals)

        start = 0
        for client_id, n_take in enumerate(counts):
            if n_take:
                client_indices[client_id].extend(pool[start:start + n_take])
                start += n_take


    # final per-client shuffle
    for idxs in client_indices.values():
        py_rng.shuffle(idxs)

    return client_indices


# =============================================================================
# == Partitioner Factory ==
# =============================================================================

def get_partitioner(strategy_name: str) -> Callable:

    if strategy_name == 'dirichlet_indices':
        return partition_dirichlet_indices
    elif strategy_name == 'iid_indices':
        return partition_iid_indices
    # Add elif for 'iid_indices_no_labels' if needed (can reuse iid_indices)
    elif strategy_name == 'iid_indices_no_labels':
         return partition_iid_indices # Use count-based IID
    elif strategy_name == 'pre_split':
        return partition_pre_defined
    else:
        raise ValueError(f"Unknown partitioning strategy: '{strategy_name}'")