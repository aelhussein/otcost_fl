"""
Functions for partitioning datasets among clients based on different strategies.
Handles Dirichlet-based label skew and IID partitioning.
"""
import numpy as np
import random
from typing import List, Dict, Any, Optional, Union
import torch
from torch.utils.data import Dataset as TorchDataset # Use alias for clarity

# --- Partitioning Functions ---

def partition_dirichlet_indices(dataset: TorchDataset,
                                num_clients: int,
                                alpha: float,
                                seed: int = 42,
                                **kwargs) -> Dict[int, List[int]]:
    """
    Partitions dataset indices among clients based on label distribution
    using a Dirichlet distribution. Aims for balanced total samples per client
    initially, then applies subsampling if limits are set.

    Args:
        dataset: The dataset to partition. Must either have a `.targets`
                 attribute/property or be iterable yielding (feature, label).
        num_clients: The number of clients to partition data for.
        alpha: The concentration parameter for the Dirichlet distribution.
               Lower alpha leads to more non-IID distributions (label skew).
        seed: Random seed for reproducibility of shuffling and sampling.
        **kwargs: Can include 'sampling_config' dictionary with keys 'size'
                  (max samples per client) and 'replace' (bool for subsampling).

    Returns:
        A dictionary mapping client index (int) to a list of sample indices (List[int]).
    """
    # --- Configuration from kwargs ---
    sampling_config: Dict = kwargs.get('sampling_config', {})
    # Default to a very large number if no limit specified
    max_samples_limit: int = sampling_config.get('size', int(1e9))
    replace_sampling: bool = sampling_config.get('replace', False)

    print(f"Starting Dirichlet partitioning: alpha={alpha}, clients={num_clients}, "
          f"limit/client={max_samples_limit}, replace={replace_sampling}")

    # --- 1. Extract Labels ---
    labels: Optional[np.ndarray] = None
    try:
        # Preferentially use .targets if available (more efficient)
        targets = getattr(dataset, 'targets', None)
        if targets is not None:
             if isinstance(targets, torch.Tensor):
                  labels = targets.cpu().numpy()
             elif isinstance(targets, (np.ndarray, list)):
                  labels = np.array(targets)
             else:
                  print("Warning: '.targets' attribute found but type unknown, falling back to iteration.")
        if labels is None:
             # Fallback: Iterate through the dataset (can be slow for large datasets)
             print("No '.targets' attribute found or usable, iterating through dataset to get labels...")
             labels_list = [dataset[i][1] for i in range(len(dataset))]
             labels = np.array(labels_list)
    except Exception as e:
        print(f"Error accessing labels from dataset: {e}")
        raise ValueError("Could not extract labels from the provided dataset.") from e

    n = len(labels)
    if n == 0:
         print("Warning: Dataset is empty. Returning empty partitions.")
         return {i: [] for i in range(num_clients)}

    classes, class_counts = np.unique(labels, return_counts=True)
    n_classes = len(classes)

    print(f"Dataset Details: {n} samples, {n_classes} classes.")
    print(f"Class Distribution: {dict(zip(classes, class_counts))}")

    if n_classes == 0:
        print("Warning: Dataset has no classes. Returning empty partitions.")
        return {i: [] for i in range(num_clients)}
    if n_classes < 2:
         print("Warning: Only one class present. Dirichlet partitioning will be equivalent to IID quantity split.")

    # --- 2. Create Index Pools per Class ---
    # Store indices belonging to each class
    idx_by_class: Dict[Any, List[int]] = {
        c: np.where(labels == c)[0].tolist() for c in classes
    }

    # --- 3. Generate Dirichlet Proportions & Initialize ---
    # Use a dedicated random number generator for reproducibility
    rng = np.random.default_rng(seed)
    # Shuffle indices within each class pool initially
    for c in classes:
        rng.shuffle(idx_by_class[c])

    # Generate target proportions for each client for each class
    # Shape: (num_clients, n_classes)
    proportions: np.ndarray = rng.dirichlet([alpha] * n_classes, size=num_clients)

    # --- 4. Calculate Initial Quotas ---
    # Aim for roughly equal total number of samples per client initially
    base_quota: int = n // num_clients
    remainder: int = n % num_clients
    # Distribute remainder samples to the first 'remainder' clients
    quotas: List[int] = [base_quota + 1 if i < remainder else base_quota for i in range(num_clients)]
    print(f"Target total samples per client (before potential limits): {quotas} (sum={sum(quotas)})")

    # --- 5. Assign Indices Proportionally ---
    client_indices: Dict[int, List[int]] = {i: [] for i in range(num_clients)}
    # Keep track of which indices from each class pool have been assigned
    class_pool_ptr: Dict[Any, int] = {c: 0 for c in classes}

    for client_id in range(num_clients):
        target_quota = quotas[client_id]
        client_prop = proportions[client_id] # Proportions for this client

        # Calculate target number of samples *per class* for this client
        class_targets_float = client_prop * target_quota
        class_targets_int = class_targets_float.astype(int)

        # Adjust last class target to exactly match quota due to rounding
        class_targets_int[-1] = target_quota - class_targets_int[:-1].sum()
        # Ensure non-negative counts
        class_targets_int = np.maximum(0, class_targets_int)

        # print(f"Client {client_id+1}: Target quota={target_quota}, Class targets={dict(zip(classes, class_targets_int))}") # Verbose

        temp_client_indices: List[int] = []
        for cls_idx, cls in enumerate(classes):
            num_wanted: int = class_targets_int[cls_idx]
            if num_wanted == 0:
                continue

            pool: List[int] = idx_by_class[cls]
            ptr: int = class_pool_ptr[cls] # Current position in this class pool

            # How many samples are actually available from this point onwards?
            available_in_pool: int = len(pool) - ptr
            num_to_take: int = min(num_wanted, available_in_pool)

            # Take the indices
            taken_indices: List[int] = pool[ptr : ptr + num_to_take]
            temp_client_indices.extend(taken_indices)

            # Advance the pointer for this class pool
            class_pool_ptr[cls] += num_to_take

            if num_to_take < num_wanted:
                print(f"  Warning: Class '{cls}' pool exhausted for client {client_id+1}. "
                      f"Wanted {num_wanted}, got {num_to_take}.")

        client_indices[client_id] = temp_client_indices
        # print(f"  Client {client_id+1}: Initially assigned {len(temp_client_indices)} samples.") # Verbose

    # --- 6. Redistribute Remaining Unassigned Indices ---
    # Indices might remain unassigned if class pools were exhausted early
    all_assigned_indices = set(idx for indices in client_indices.values() for idx in indices)
    remaining_indices = list(set(np.arange(n)) - all_assigned_indices)
    rng.shuffle(remaining_indices) # Shuffle the remaining indices
    print(f"Indices remaining after proportional assignment: {len(remaining_indices)}")

    # Distribute remaining indices greedily to clients currently below their target quota
    for client_id in range(num_clients):
        current_size: int = len(client_indices[client_id])
        needed: int = quotas[client_id] - current_size
        if needed > 0 and remaining_indices:
            # Give up to 'needed' indices from the remaining pool
            num_to_give: int = min(needed, len(remaining_indices))
            give_indices: List[int] = remaining_indices[:num_to_give]
            client_indices[client_id].extend(give_indices)
            # Remove given indices from the remaining pool
            remaining_indices = remaining_indices[num_to_give:]
            # print(f"  Client {client_id+1}: Given {num_to_give} remaining indices.") # Verbose
        if not remaining_indices: # Optimization: break if no more indices left
             break


    # --- 7. Apply Max Sample Limit (Subsampling) ---
    print(f"\nApplying client sample limit (max={max_samples_limit}, replace={replace_sampling})...")
    final_total_assigned: int = 0
    final_unique_assigned = set()

    for i in range(num_clients):
        current_client_indices: List[int] = client_indices[i]
        current_size: int = len(current_client_indices)

        if current_size > max_samples_limit:
            # Adjust limit for non-replacement sampling if needed
            actual_limit = min(max_samples_limit, current_size) if not replace_sampling else max_samples_limit
            if actual_limit != max_samples_limit and not replace_sampling:
                 print(f"    Adjusted limit to {actual_limit} for non-replacement sampling for client {i+1}.")

            print(f"  Client {i+1}: Subsampling {current_size} -> {actual_limit}")
            # Perform sampling using the isolated RNG
            sampled_indices = rng.choice(
                current_client_indices,
                size=actual_limit,
                replace=replace_sampling
            ).tolist()
            client_indices[i] = sampled_indices

        # Shuffle the final set of indices for this client
        rng.shuffle(client_indices[i])
        final_size = len(client_indices[i])
        print(f"  Client {i+1}: Final size = {final_size}")
        final_total_assigned += final_size
        final_unique_assigned.update(client_indices[i]) # Add indices to set for uniqueness check

    # --- 8. Final Report ---
    print(f"\nTotal samples finally assigned across all clients: {final_total_assigned}")
    print(f"Unique indices finally assigned: {len(final_unique_assigned)}")

    # Sanity Checks (optional but recommended)
    if not replace_sampling and final_total_assigned != len(final_unique_assigned):
         print("Warning: Index collision detected in final assignment despite replace=False!")
    for i in range(num_clients):
        if len(client_indices[i]) > max_samples_limit:
             print(f"Error: Client {i+1} size {len(client_indices[i])} exceeds limit {max_samples_limit}!")


    print("-" * 30) # Separator
    return client_indices


def partition_iid_indices(dataset: TorchDataset,
                          num_clients: int,
                          seed: int = 42,
                          **kwargs) -> Dict[int, List[int]]:
    """
    Partitions dataset indices equally and randomly (IID) among clients.

    Args:
        dataset: The dataset to partition.
        num_clients: The number of clients.
        seed: Random seed for reproducibility.
        **kwargs: Catches unused keyword arguments (like sampling_config).

    Returns:
        A dictionary mapping client index to a list of sample indices.
    """
    n_samples: int = len(dataset)
    indices: np.ndarray = np.arange(n_samples)

    # Use isolated RNG
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    # Split indices into roughly equal parts
    split_indices: List[np.ndarray] = np.array_split(indices, num_clients)

    # Convert parts to lists for the final dictionary
    client_indices: Dict[int, List[int]] = {
        i: split_indices[i].tolist() for i in range(num_clients)
    }

    print(f"IID partitioning: {n_samples} samples -> {num_clients} clients.")
    # Optional: Print sizes per client
    # for i in range(num_clients):
    #      print(f"  Client {i+1}: {len(client_indices[i])} samples")

    return client_indices

def partition_pre_defined(**kwargs) -> None:
    """
    Placeholder strategy for datasets where data is already split
    and loaded per client. This function does nothing.

    Returns:
        None, signaling to the pipeline that partitioning is handled elsewhere.
    """
    return None

# --- Dispatch Dictionary ---
# Maps strategy names (from config) to partition functions
PARTITIONING_STRATEGIES: Dict[str, callable] = {
    'dirichlet_indices': partition_dirichlet_indices,
    'iid_indices': partition_iid_indices,
    'pre_split': partition_pre_defined,
}