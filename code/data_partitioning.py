"""
Functions for partitioning datasets among clients based on different strategies.
Handles Dirichlet-based label skew and IID partitioning.
"""
import numpy as np
import random
from typing import List, Dict, Any, Optional, Union
import torch
from torch.utils.data import Dataset as TorchDataset # Use alias for clarity
import traceback # For detailed error printing

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
        targets = getattr(dataset, 'targets', None)
        if targets is not None:
             if isinstance(targets, torch.Tensor): labels = targets.cpu().numpy()
             elif isinstance(targets, (np.ndarray, list)): labels = np.array(targets)
             else: print("Warning: '.targets' attribute found but type unknown, falling back to iteration.")
        if labels is None:
             print("No usable '.targets' attribute found, iterating through dataset to get labels...")
             labels_list = [dataset[i][1] for i in range(len(dataset))]
             labels = np.array(labels_list)
    except AttributeError: print("Dataset does not support indexing or iteration needed to extract labels."); raise
    except Exception as e: print(f"Error accessing labels from dataset: {e}"); traceback.print_exc(); raise ValueError("Could not extract labels from the provided dataset.") from e

    n = len(labels)
    if n == 0: print("Warning: Dataset is empty. Returning empty partitions."); return {i: [] for i in range(num_clients)}

    classes, class_counts = np.unique(labels, return_counts=True)
    n_classes = len(classes)

    print(f"Dataset Details: {n} samples, {n_classes} classes.")
    # print(f"Class Distribution: {dict(zip(classes, class_counts))}") # Can be verbose

    if n_classes == 0: print("Warning: Dataset has no classes. Returning empty partitions."); return {i: [] for i in range(num_clients)}

    # --- 3. Generate Dirichlet Proportions & Initialize ---
    rng = np.random.default_rng(seed)

    # MODIFIED: Move single-class alpha patch *before* Dirichlet call
    if n_classes < 2:
         print("Warning: Only one class present. Dirichlet partitioning will behave like IID.")
         if alpha <= 0: alpha = 1.0 # Avoid issues with alpha <= 0 for single class

    # Generate target proportions for each client for each class
    # Dirichlet distribution needs alpha per class
    dirichlet_alphas = [alpha] * n_classes
    proportions: np.ndarray = rng.dirichlet(dirichlet_alphas, size=num_clients)


    # --- 2. Create Index Pools per Class (after checking n_classes) ---
    idx_by_class: Dict[Any, List[int]] = {
        c: np.where(labels == c)[0].tolist() for c in classes
    }
    # Shuffle indices within each class pool
    for c in classes:
        rng.shuffle(idx_by_class[c])


    # --- 4. Calculate Initial Quotas ---
    base_quota: int = n // num_clients
    remainder: int = n % num_clients
    quotas: List[int] = [(base_quota + 1 if i < remainder else base_quota) for i in range(num_clients)]
    # print(f"Target total samples per client (initial): {quotas} (sum={sum(quotas)})")

    # --- 5. Assign Indices Proportionally ---
    client_indices: Dict[int, List[int]] = {i: [] for i in range(num_clients)}
    class_pool_ptr: Dict[Any, int] = {c: 0 for c in classes}

    # Calculate number of samples needed from each class for each client based on proportions
    # This calculation now happens class by class across all clients to better respect proportions
    indices_assigned_count = 0
    for class_idx, current_class in enumerate(classes):
        class_indices_pool = idx_by_class[current_class]
        total_class_indices = len(class_indices_pool)
        # Calculate how many samples of this class each client should get
        target_counts_for_class = (proportions[:, class_idx] * total_class_indices).astype(int)
        # Adjust last client's target to account for rounding errors for this class
        target_counts_for_class[-1] = total_class_indices - target_counts_for_class[:-1].sum()
        target_counts_for_class = np.maximum(0, target_counts_for_class) # Ensure non-negative

        class_ptr = 0
        for client_id in range(num_clients):
            num_wanted = target_counts_for_class[client_id]
            if num_wanted == 0:
                 continue

            num_available = total_class_indices - class_ptr
            num_to_take = min(num_wanted, num_available)

            if num_to_take > 0:
                taken_indices = class_indices_pool[class_ptr : class_ptr + num_to_take]
                client_indices[client_id].extend(taken_indices)
                class_ptr += num_to_take
                indices_assigned_count += num_to_take

            # This warning is less likely with the new assignment logic, but keep just in case
            if num_to_take < num_wanted:
                print(f"  Warning: Class '{current_class}' pool seems exhausted prematurely for client {client_id+1}. "
                      f"Wanted {num_wanted}, Got {num_to_take}. Total in class: {total_class_indices}.")

    # --- 6. Redistribute Remaining Unassigned Indices (Less likely, but possible due to rounding) ---
    all_assigned_indices = set(idx for indices in client_indices.values() for idx in indices)
    remaining_indices = list(set(np.arange(n)) - all_assigned_indices)
    rng.shuffle(remaining_indices)
    print(f"Indices remaining after proportional assignment: {len(remaining_indices)}")

    # Distribute remaining indices greedily to clients who are below their quota (less precise now)
    # An alternative is round-robin assignment
    client_order = list(range(num_clients))
    rng.shuffle(client_order)
    idx_rem = 0
    while idx_rem < len(remaining_indices):
        client_id = client_order[idx_rem % num_clients]
        client_indices[client_id].append(remaining_indices[idx_rem])
        idx_rem += 1

    # --- 7. Apply Max Sample Limit (Subsampling) ---
    print(f"\nApplying client sample limit (max={max_samples_limit}, replace={replace_sampling})...")
    final_total_assigned: int = 0
    final_unique_assigned = set()

    for i in range(num_clients):
        current_client_indices: List[int] = client_indices[i]
        current_size: int = len(current_client_indices)

        # MODIFIED: Simplified subsampling logic
        if current_size > max_samples_limit:
            print(f"  Client {i+1}: Subsampling {current_size} -> {max_samples_limit} (replace={replace_sampling})")
            # Use choice directly with the limit
            sampled_indices = rng.choice(
                current_client_indices,
                size=max_samples_limit, # Sample exactly the limit
                replace=replace_sampling
            ).tolist()
            client_indices[i] = sampled_indices
        elif current_size == 0:
             print(f"  Client {i+1}: Has 0 samples assigned.")

        # Shuffle the final set of indices for this client
        rng.shuffle(client_indices[i])
        final_size = len(client_indices[i])
        # print(f"  Client {i+1}: Final size = {final_size}") # Verbose
        final_total_assigned += final_size
        final_unique_assigned.update(client_indices[i]) # Add indices to set

    # --- 8. Final Report ---
    print(f"\nTotal samples finally assigned across all clients: {final_total_assigned}")
    print(f"Unique indices finally assigned: {len(final_unique_assigned)}")

    # Sanity Checks
    if not replace_sampling and final_total_assigned > len(final_unique_assigned):
         # Count duplicates explicitly
         all_indices = [idx for sublist in client_indices.values() for idx in sublist]
         if len(all_indices) != len(set(all_indices)):
             print("Warning: Index collision detected in final assignment despite replace=False!")
    if not replace_sampling and len(final_unique_assigned) > n:
          print(f"Error: More unique indices assigned ({len(final_unique_assigned)}) than exist in dataset ({n})!")


    print("-" * 30)
    return client_indices


def partition_iid_indices(dataset: TorchDataset,
                          num_clients: int,
                          seed: int = 42,
                          **kwargs) -> Dict[int, List[int]]:
    """
    Partitions dataset indices equally and randomly (IID) among clients.

    Args:
        dataset: The dataset to partition (only `len()` is used).
        num_clients: The number of clients.
        seed: Random seed for reproducibility.
        **kwargs: Catches unused keyword arguments.

    Returns:
        A dictionary mapping client index to a list of sample indices.
    """
    try:
        n_samples: int = len(dataset)
    except TypeError:
        raise TypeError("Dataset provided to partition_iid_indices must have __len__.")

    if n_samples == 0:
        print("Warning: Dataset for IID partitioning is empty.")
        return {i: [] for i in range(num_clients)}

    indices: np.ndarray = np.arange(n_samples)
    rng = np.random.default_rng(seed) # Use isolated generator
    rng.shuffle(indices)

    # Split indices into num_clients parts
    split_indices: List[np.ndarray] = np.array_split(indices, num_clients)

    # Create dictionary mapping client index to list of indices
    client_indices: Dict[int, List[int]] = {
        i: split_indices[i].tolist() for i in range(num_clients)
    }

    print(f"IID partitioning: {n_samples} samples -> {num_clients} clients.")
    # print(f"  Client sizes: {[len(v) for v in client_indices.values()]}") # Optional verbose
    return client_indices

# --- NEW: IID Partitioning without accessing labels ---
def partition_iid_indices_no_labels(dataset: TorchDataset,
                                    num_clients: int,
                                    seed: int = 42,
                                    **kwargs) -> Dict[int, List[int]]:
    """
    Partitions dataset indices equally and randomly (IID) among clients,
    *without* accessing dataset labels (e.g., for feature-only datasets).
    Identical logic to `partition_iid_indices` as label access isn't needed.

    Args:
        dataset: The dataset to partition (only `len()` is used).
        num_clients: The number of clients.
        seed: Random seed for reproducibility.
        **kwargs: Catches unused keyword arguments.

    Returns:
        A dictionary mapping client index to a list of sample indices.
    """
    print("Calling IID partitioning (no labels required)...")
    return partition_iid_indices(dataset, num_clients, seed, **kwargs)


def partition_pre_defined(**kwargs) -> None:
    """
    Placeholder strategy for datasets where data is already split
    and loaded per client. This function does nothing.

    Returns:
        None, signaling to the pipeline that partitioning is handled elsewhere.
    """
    print("Using pre-defined partitioning strategy (loading per client).")
    return None

# --- Dispatch Dictionary ---
# Maps strategy names (from config) to partition functions
PARTITIONING_STRATEGIES: Dict[str, callable] = {
    'dirichlet_indices': partition_dirichlet_indices,
    'iid_indices': partition_iid_indices,
    'iid_indices_no_labels': partition_iid_indices_no_labels, # For concept skew base features
    'pre_split': partition_pre_defined,
}