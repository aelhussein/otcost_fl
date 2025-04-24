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

def partition_dirichlet_indices(labels_np: np.ndarray,
                                num_clients: int,
                                alpha: float, # Explicit alpha argument
                                seed: int = 42,
                                sampling_config: Optional[dict] = None
                               ) -> Dict[int, List[int]]:
    """
    Partitions dataset indices based on labels using Dirichlet distribution.

    Args:
        labels_np (np.ndarray): NumPy array of dataset labels.
        num_clients (int): Number of clients.
        alpha (float): Dirichlet concentration parameter.
        seed (int): Random seed.
        sampling_config (Optional[dict]): Config for potential subsampling.
                                          Expects {'size': max_samples_per_client}.

    Returns:
        Dict[int, List[int]]: Mapping client index (0-based) to list of sample indices.
    """
    n_samples = len(labels_np)
    if n_samples == 0:
        return {i: [] for i in range(num_clients)}

    rng_np = np.random.RandomState(seed)
    py_random = random.Random(seed) # Separate generator for Python lists

    classes = np.unique(labels_np)
    n_classes = len(classes)

    if n_classes <= 1: # Fallback to IID-like split
        # NO PRINT HERE
        return partition_iid_indices(n_samples, num_clients, seed)

    # Calculate proportions: How much of each class should each client get?
    proportions = rng_np.dirichlet([alpha] * n_classes, size=num_clients)

    # Get indices for each class and shuffle them
    indices_by_class = {c: np.where(labels_np == c)[0].tolist() for c in classes}
    for c in classes:
        py_random.shuffle(indices_by_class[c])

    client_indices_raw: Dict[int, List[int]] = {i: [] for i in range(num_clients)}
    indices_per_class = np.array([len(indices_by_class[c]) for c in classes])

    # Distribute indices class by class
    for c_idx, c in enumerate(classes):
        num_indices_for_class = indices_per_class[c_idx]
        if num_indices_for_class == 0:
            continue

        # Calculate target number of samples per client for this class
        target_counts_float = proportions[:, c_idx] * num_indices_for_class
        target_counts = target_counts_float.astype(int)

        # Adjust the last client's count to ensure total matches num_indices_for_class
        target_counts[-1] = num_indices_for_class - np.sum(target_counts[:-1])
        target_counts = np.maximum(0, target_counts) # Ensure no negative counts

        class_indices_pool = indices_by_class[c]
        start_idx = 0
        for client_id in range(num_clients):
            num_to_take = target_counts[client_id]
            if num_to_take == 0:
                continue

            end_idx = start_idx + num_to_take
            # Ensure we don't exceed the available indices in the pool
            actual_end_idx = min(end_idx, len(class_indices_pool))
            actual_taken_indices = class_indices_pool[start_idx : actual_end_idx]

            # NO WARNING PRINT HERE
            client_indices_raw[client_id].extend(actual_taken_indices)
            start_idx = actual_end_idx # Update start for next client


    # Apply sampling limit if specified
    client_indices_final = client_indices_raw
    if sampling_config and 'size' in sampling_config:
        max_samples = sampling_config['size']
        if max_samples < float('inf') and max_samples >= 0 :
            max_samples_int = int(max_samples)
            client_indices_final = {}
            for idx, indices in client_indices_raw.items():
                if len(indices) > max_samples_int:
                    client_indices_final[idx] = py_random.sample(indices, max_samples_int)
                else:
                    client_indices_final[idx] = indices # Keep original list
        # Note: No warning for invalid sampling size

    # Final shuffle and print summary with class counts
    # --- ONLY PRINTING SECTION ---
    print(f"\n--- Dirichlet Partitioning Summary (alpha={alpha}) ---")
    total_assigned_final = 0
    for client_idx in sorted(client_indices_final.keys()):
        indices = client_indices_final[client_idx]
        py_random.shuffle(indices) # Shuffle final list per client
        count = len(indices)
        total_assigned_final += count

        # Calculate and print class distribution for this client
        if count > 0:
            client_labels = labels_np[indices] # Get labels corresponding to assigned indices
            unique_labels, counts = np.unique(client_labels, return_counts=True)
            class_dist_str = ", ".join([f"Class {l}: {c}" for l, c in zip(unique_labels, counts)])
        else:
            class_dist_str = "No samples"

        print(f"  Client {client_idx}: {count} samples ({class_dist_str})") # Print total count and class breakdown

    print(f"  Total Assigned Samples: {total_assigned_final}")
    # --- END ONLY PRINTING SECTION ---

    return client_indices_final

# =============================================================================
# == Partitioner Factory ==
# =============================================================================

def get_partitioner(strategy_name: str) -> Callable:
    """Factory function to get the appropriate partitioning function."""
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