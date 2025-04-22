"""
Functions for partitioning datasets among clients based on different strategies.
"""
import numpy as np
import random
from typing import List, Dict, Any
import torch
from torch.utils.data import Dataset as TorchDataset # Use alias for clarity

# --- Partitioning Functions ---

def partition_dirichlet_indices(dataset: TorchDataset, num_clients: int, alpha: float, seed: int = 42, **kwargs) -> Dict[int, List[int]]:
    """Partitions dataset indices based on labels using Dirichlet distribution."""
    # (Keep function code as provided in the previous data_processing.py)
    max_samples_limit = kwargs.get('sampling_config', {}).get('size', 10000); replace_sampling = kwargs.get('sampling_config', {}).get('replace', False); print(f"Dirichlet partitioning: alpha={alpha}, clients={num_clients}, limit={max_samples_limit}, replace={replace_sampling}")
    try: labels = np.array(dataset.targets)
    except AttributeError: print("No .targets found, iterating..."); labels = np.array([dataset[i][1] for i in range(len(dataset))])
    n = len(dataset); classes, class_counts = np.unique(labels, return_counts=True); n_classes = len(classes); print(f"Dataset: {n} samples, {n_classes} classes. Dist: {dict(zip(classes, class_counts))}"); assert n > 0 and n_classes > 0, "Empty dataset or no classes found."
    idx_by_class = {c: np.where(labels == c)[0].tolist() for c in classes}; rng = np.random.default_rng(seed)
    for c in classes: rng.shuffle(idx_by_class[c]); proportions = rng.dirichlet([alpha] * n_classes, size=num_clients)
    base_quota = n // num_clients; quotas = [base_quota + (1 if i < (n % num_clients) else 0) for i in range(num_clients)]; print(f"Target quotas/client: {quotas} (sum={sum(quotas)})")
    client_indices = {i: [] for i in range(num_clients)}; class_pool_ptr = {c: 0 for c in classes}
    for client_id in range(num_clients):
        target_quota = quotas[client_id]; class_targets = (proportions[client_id] * target_quota).astype(int); class_targets[-1] = target_quota - class_targets[:-1].sum(); class_targets = np.maximum(0, class_targets); temp_client_indices = []
        for cls_idx, cls in enumerate(classes):
            num_wanted = class_targets[cls_idx]
            if num_wanted == 0: continue; pool = idx_by_class[cls]; ptr = class_pool_ptr[cls]; available_in_pool = len(pool) - ptr; num_to_take = min(num_wanted, available_in_pool); taken_indices = pool[ptr : ptr + num_to_take]; temp_client_indices.extend(taken_indices); class_pool_ptr[cls] += num_to_take
            if num_to_take < num_wanted: print(f"  Warn: Class {cls} pool exhausted for client {client_id+1}. Wanted {num_wanted}, got {num_to_take}.")
        client_indices[client_id] = temp_client_indices
    all_assigned = set(idx for indices in client_indices.values() for idx in indices); remaining_indices = list(set(np.arange(n)) - all_assigned); rng.shuffle(remaining_indices); print(f"Indices remaining after proportional assignment: {len(remaining_indices)}")
    for client_id in range(num_clients): current_size = len(client_indices[client_id]); needed = quotas[client_id] - current_size;
    if needed > 0 and remaining_indices: num_to_give = min(needed, len(remaining_indices)); give_indices = remaining_indices[:num_to_give]; client_indices[client_id].extend(give_indices); remaining_indices = remaining_indices[num_to_give:]
    print(f"\nSubsampling clients >{max_samples_limit} (replace={replace_sampling})..."); final_total_assigned = 0; final_unique_assigned = set()
    for i in range(num_clients):
        current_size = len(client_indices[i])
        if current_size > max_samples_limit: actual_limit = min(max_samples_limit, current_size) if not replace_sampling else max_samples_limit; print(f"  Client {i+1}: Subsampling {current_size} -> {actual_limit}"); sampled_indices = rng.choice(client_indices[i], size=actual_limit, replace=replace_sampling).tolist(); client_indices[i] = sampled_indices
        rng.shuffle(client_indices[i]); print(f"  Client {i+1}: Final size = {len(client_indices[i])}"); final_total_assigned += len(client_indices[i]); final_unique_assigned.update(client_indices[i])
    print(f"\nTotal samples assigned: {final_total_assigned}, Unique: {len(final_unique_assigned)}\n" + "-" * 30); return client_indices

def partition_iid_indices(dataset: TorchDataset, num_clients: int, seed: int = 42, **kwargs):
    """Partitions dataset indices equally and randomly (IID)."""
    # (Keep function code as provided in the previous data_processing.py)
    n_samples = len(dataset); indices = np.arange(n_samples); rng = np.random.default_rng(seed); rng.shuffle(indices); split_indices = np.array_split(indices, num_clients); client_indices = {i: split_indices[i].tolist() for i in range(num_clients)}; print(f"IID partition: {n_samples} -> {num_clients} clients."); return client_indices

def partition_pre_defined(**kwargs):
    """Placeholder for pre-split data. Returns None."""
    return None # Signal that loading is per-client

# --- Dispatch Dictionary ---
PARTITIONING_STRATEGIES = {
    'dirichlet_indices': partition_dirichlet_indices,
    'iid_indices': partition_iid_indices,
    'pre_split': partition_pre_defined,
}