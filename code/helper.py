"""
Utility functions and classes for the federated learning pipeline.
Includes seeding, GPU cleanup, metric calculation helpers, configuration access,
cost translation, and model diversity calculation.
"""
import os
import gc
import random
import copy # Used in ModelDiversity if it were more complex, but not strictly needed now
import numpy as np
import torch
import torch.nn as nn # For type hinting nn.Module
from torch import Tensor # For type hinting
from typing import Dict, Any, Optional, List, Tuple, Iterator # For type hinting
import hashlib # NEW: For better seed generation

# Import only necessary configs or access via functions
from configs import DEFAULT_PARAMS

# --- NEW: Metric Key Constants ---
class MetricKey:
    """Constants for metric dictionary keys."""
    TRAIN_LOSSES = 'train_losses'
    VAL_LOSSES = 'val_losses'
    TEST_LOSSES = 'test_losses'
    TRAIN_SCORES = 'train_scores'
    VAL_SCORES = 'val_scores'
    TEST_SCORES = 'test_scores'
    # Add others if needed, e.g., GLOBAL_LOSSES, GLOBAL_SCORES

# --- Seeding and Environment ---

def set_seeds(seed_value: int = 1) -> torch.Generator:
    """
    Sets random seeds for PyTorch, NumPy, and Python's random module
    for reproducibility. Also configures CUDA for deterministic algorithms.

    Args:
        seed_value: The integer value to use for seeding.

    Returns:
        A seeded PyTorch random number generator.
    """
    torch.manual_seed(seed_value)
    # Seed all GPUs if available
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    np.random.seed(seed_value)
    random.seed(seed_value)
    # Set environment variable for Python's hash seed
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    # Configure PyTorch to use deterministic algorithms
    # Note: This might impact performance and compatibility with some operations.
    try:
         torch.backends.cudnn.deterministic = True
         torch.backends.cudnn.benchmark = False
         # Requires PyTorch 1.7+
         # MODIFIED: Wrap in version check if possible, or just try/except
         if hasattr(torch, 'use_deterministic_algorithms'):
             torch.use_deterministic_algorithms(True)
         # On some systems (like CUBLAS), further env vars might be needed:
         # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8" # Or ":16:8"
    except AttributeError:
         print("Warning: Could not set deterministic algorithms (requires PyTorch 1.7+).")
    except Exception as e:
         print(f"Warning: Error setting deterministic behavior: {e}")

    # Create and return a seeded generator for DataLoaders if needed elsewhere
    g = torch.Generator()
    g.manual_seed(seed_value)
    return g

# Initialize global generator (used by DataLoaders if passed)
g = set_seeds(seed_value=1) # Or use seed from config/args

def seed_worker(worker_id: int):
    """
    Seeds a DataLoader worker process for reproducible data loading/shuffling
    when using multiple workers. To be used with `worker_init_fn`.

    Args:
        worker_id: The ID of the worker process (unused but required by DataLoader).
    """
    # Get the initial seed from the main process and make it worker-specific
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    # print(f"DataLoader worker {worker_id} seeded with {worker_seed}") # Debug print

def cleanup_gpu():
    """Releases unused memory from the CUDA cache if available."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # Optional: Trigger Python's garbage collector too
        # gc.collect()

def move_to_device(batch: Any, device: torch.device) -> Any:
    """
    Moves a batch of data (single tensor, list, or tuple) to the specified device.

    Args:
        batch: The data batch.
        device: The target torch.device (e.g., 'cuda', 'cpu').

    Returns:
        The batch moved to the target device.
    """
    if isinstance(batch, (list, tuple)):
        # Recursively move elements if they are tensors
        return [
            item.to(device) if isinstance(item, torch.Tensor) else item
            for item in batch
        ]
    elif isinstance(batch, torch.Tensor):
        return batch.to(device)
    # Return unchanged if not a tensor or list/tuple of tensors
    return batch

# --- Metrics ---

def get_dice_score(output: torch.Tensor, target: torch.Tensor,
                   foreground_channel: int = 1, # NEW: Specify foreground channel
                   SPATIAL_DIMENSIONS: Tuple[int, ...] = (2, 3, 4),
                   epsilon: float = 1e-9) -> float:
    """
    Calculates the mean Dice score for 3D segmentation tasks (like IXITiny).
    Assumes input tensors are probabilities or logits for the foreground class.

    Args:
        output: Predicted probabilities/logits (shape: N, C, D, H, W).
        target: Ground truth labels (shape: N, C, D, H, W, usually one-hot).
        foreground_channel (int): Index of the channel representing the foreground class. Default 1.
        SPATIAL_DIMENSIONS: Tuple of spatial dimension indices (usually Depth, Height, Width).
        epsilon: Small value to prevent division by zero.

    Returns:
        The mean Dice score across the batch.

    Raises:
        ValueError: If foreground_channel index is invalid for input shapes.
    """
    num_channels_out = output.shape[1]
    num_channels_tgt = target.shape[1]

    if not (0 <= foreground_channel < num_channels_out):
        raise ValueError(f"foreground_channel ({foreground_channel}) out of bounds for output shape {output.shape}")
    if not (0 <= foreground_channel < num_channels_tgt):
         raise ValueError(f"foreground_channel ({foreground_channel}) out of bounds for target shape {target.shape}")

    # MODIFIED: Use foreground_channel parameter
    # If input is logits, apply sigmoid/softmax first if needed by the metric interpretation
    p0 = output[:, foreground_channel, ...] # Predicted foreground prob
    g0 = target[:, foreground_channel, ...] # True foreground mask

    # Calculate background probabilities/masks (assuming binary or multi-class handled by focusing on foreground_channel)
    # For Dice, we typically compare foreground prediction vs foreground target directly
    # Background logic might be needed for other metrics, but not standard Dice.

    # Calculate True Positives, False Positives, False Negatives per sample for the foreground channel
    tp = torch.sum(p0 * g0, dim=SPATIAL_DIMENSIONS)
    fp = torch.sum(p0 * (1.0 - g0), dim=SPATIAL_DIMENSIONS) # Predict foreground, but target is background
    fn = torch.sum((1.0 - p0) * g0, dim=SPATIAL_DIMENSIONS) # Predict background, but target is foreground

    # Calculate Dice score per sample
    numerator = 2 * tp
    denominator = 2 * tp + fp + fn + epsilon
    dice_score_per_sample = numerator / denominator

    # Return the mean Dice score across the batch
    return dice_score_per_sample.mean().item()

# --- Configuration Helpers ---

def get_parameters_for_dataset(dataset_name: str) -> Dict:
    """
    Retrieves the default configuration parameters dictionary for a given dataset
    from the global `DEFAULT_PARAMS`.

    Args:
        dataset_name: The name of the dataset (must be a key in DEFAULT_PARAMS).

    Returns:
        The configuration dictionary for the dataset.

    Raises:
        ValueError: If the dataset name is not found in DEFAULT_PARAMS.
    """
    params = DEFAULT_PARAMS.get(dataset_name)
    if params is None:
        raise ValueError(f"Dataset '{dataset_name}' is not supported or has no default "
                         f"parameters defined in configs.py.")

    # Ensure backward compatibility if older code uses sizes_per_client directly
    sampling_config = params.get('sampling_config')
    if (isinstance(sampling_config, dict) and
        sampling_config.get('type') == 'fixed_total' and
        'size' in sampling_config):
        # This can potentially be removed if nothing relies on 'sizes_per_client' anymore
        params['sizes_per_client'] = sampling_config['size']

    return params

def get_default_lr(dataset_name: str) -> float:
    """Gets the default learning rate from the dataset's configuration."""
    params = get_parameters_for_dataset(dataset_name)
    lr = params.get('default_lr')
    if lr is None:
         raise ValueError(f"'default_lr' not defined for dataset '{dataset_name}' in configs.py")
    return lr

def get_default_reg(dataset_name: str) -> Optional[float]:
    """Gets the default regularization parameter from the dataset's config (can be None)."""
    params = get_parameters_for_dataset(dataset_name)
    # Returns None if not defined, which is valid for algos not needing it
    reg = params.get('default_reg_param')
    return reg

def translate_cost(cost: Any, interpretation_type: str) -> Dict[str, Any]:
    """
    Translates the raw 'cost' parameter value based on the configured
    interpretation type (e.g., 'alpha', 'site_mapping_key', 'feature_shift_param').

    Args:
        cost: The value from the DATASET_COSTS list for the current dataset.
        interpretation_type: String key from config defining how to interpret cost.

    Returns:
        A dictionary containing the interpreted value(s) usable by loaders,
        partitioners or dataset wrappers (e.g., {'alpha': 0.1}, {'key': 1},
        {'feature_shift_param': 0.5}, {'cost_key_is_all': True}).

    Raises:
        ValueError: If cost value is invalid for the interpretation type or
                    if interpretation type is unknown.
    """
    # NEW: Handle 'all' consistently
    is_all_sentinel = isinstance(cost, str) and cost.lower() == 'all'

    if is_all_sentinel:
         # Return a specific signal if the cost is 'all'
         # The consumers (loader, partitioner) need to know how to handle this dict
         print(f"Translating cost: Detected 'all' sentinel for type '{interpretation_type}'")
         # Specific handling might differ based on interpretation type later if needed
         # For now, a general flag is useful.
         return {'cost_key_is_all': True, 'original_cost': cost}

    # --- Handle numeric/specific costs ---
    if interpretation_type == 'alpha':
        try: alpha = float(cost); assert alpha > 0
        except: raise ValueError(f"Invalid alpha value: {cost}")
        return {'alpha': alpha}

    elif interpretation_type == 'inv_alpha':
        try: inv_a = float(cost); assert inv_a > 0; alpha = 1.0 / inv_a
        except: raise ValueError(f"Invalid inverse alpha value: {cost}")
        return {'alpha': alpha}

    elif interpretation_type == 'file_suffix':
        return {'suffix': cost}

    elif interpretation_type == 'site_mapping_key':
        return {'key': cost}

    elif interpretation_type == 'feature_shift_param':
        try:
             shift = float(cost)
             # shift = np.clip(shift, 0.0, 1.0) # Clip here or later? Let loader/dataset handle it.
        except (ValueError, TypeError):
             raise ValueError(f"Invalid cost value '{cost}' for feature_shift_param.")
        return {'feature_shift_param': shift}

    elif interpretation_type == 'concept_shift_param':
        try:
             shift = float(cost)
             # shift = np.clip(shift, 0.0, 1.0) # Clip here or later? Let loader/dataset handle it.
        except (ValueError, TypeError):
             raise ValueError(f"Invalid cost value '{cost}' for concept_shift_param.")
        return {'concept_shift_param': shift}

    elif interpretation_type == 'ignore':
        return {} # No parameters needed

    else:
        raise ValueError(f"Unknown cost_interpretation type: '{interpretation_type}'")

# --- Model Comparison ---

class ModelDiversity:
    """Calculates diversity metrics between the models of two clients."""

    def __init__(self, client_1, client_2):
        # Type hint 'Client' as string to avoid circular import if Client imports this
        if client_1 is None or client_2 is None:
            raise ValueError("Both client_1 and client_2 must be valid Client instances.")
        self.client_1 = client_1
        self.client_2 = client_2

    def calculate_weight_divergence(self) -> Tuple[float, float]:
        """
        Calculates L2 distance and cosine similarity (orientation) between
        the normalized weight vectors of two clients' primary models.

        Returns:
            Tuple[float, float]: (weight_divergence, weight_orientation).
                                 Returns (NaN, NaN) if weights cannot be obtained.
        """
        try:
             # Get flattened weight vectors for the primary model of each client
             # Assumes client state holds the model to compare (e.g., personal or global)
             weights_1 = self._get_weights(self.client_1)
             weights_2 = self._get_weights(self.client_2)
        except AttributeError as e:
             print(f"Error accessing model weights for diversity: {e}")
             return np.nan, np.nan
        except Exception as e:
             print(f"Unexpected error getting weights for diversity: {e}")
             return np.nan, np.nan


        # Handle cases where weights might be empty (e.g., model not initialized)
        if weights_1.numel() == 0 or weights_2.numel() == 0:
             print("Warning: Cannot calculate diversity with empty model weights.")
             return np.nan, np.nan

        # Normalize weight vectors
        norm_1 = torch.norm(weights_1)
        norm_2 = torch.norm(weights_2)

        # Avoid division by zero if norm is zero
        w1_normalized = weights_1 / (norm_1 + 1e-9) # Add epsilon for safety
        w2_normalized = weights_2 / (norm_2 + 1e-9)

        if norm_1 < 1e-9: print("Warning: Norm of client 1 weights is near zero.")
        if norm_2 < 1e-9: print("Warning: Norm of client 2 weights is near zero.")


        # Calculate L2 distance between normalized weights
        weight_divergence = torch.norm(w1_normalized - w2_normalized, p=2)

        # Calculate cosine similarity (dot product of normalized vectors)
        weight_orientation = torch.dot(w1_normalized, w2_normalized)

        # Ensure results are standard Python floats
        return weight_divergence.item(), weight_orientation.item()

    def _get_weights(self, client) -> torch.Tensor:
        """
        Extracts and flattens all parameters from a client's relevant model state.

        Args:
            client: The client instance.

        Returns:
            A 1D tensor containing all model parameters flattened.

        Raises:
            AttributeError: If the client or its model state cannot be accessed.
        """
        # Decide which model state to use (e.g., personal if it exists, else global)
        # This logic might need refinement based on the specific diversity comparison needed
        if hasattr(client, 'personal_state') and client.personal_state is not None:
             state_to_use = client.personal_state
             # print(f"DEBUG: Using PERSONAL state for diversity client {getattr(client, 'site_id', 'Unknown')}") # Debug
        elif hasattr(client, 'global_state') and client.global_state is not None:
             state_to_use = client.global_state
             # print(f"DEBUG: Using GLOBAL state for diversity client {getattr(client, 'site_id', 'Unknown')}") # Debug
        else:
             raise AttributeError(f"Client {getattr(client, 'site_id', 'Unknown')} has no accessible model state.")

        if not hasattr(state_to_use, 'model') or state_to_use.model is None:
             raise AttributeError(f"Model is None in the selected state for client {getattr(client, 'site_id', 'Unknown')}.")

        weights_list: List[torch.Tensor] = []
        # Iterate through parameters of the selected model
        for param in state_to_use.model.parameters():
            # Ensure parameter has data and flatten it
            if param.data is not None:
                weights_list.append(param.data.detach().view(-1)) # Use detach

        if not weights_list:
             # Handle models with no parameters
             # Use client's device if possible, else default CPU
             device = client.device if hasattr(client, 'device') else torch.device('cpu')
             return torch.tensor([], dtype=torch.float32, device=device) # Return empty tensor

        # Concatenate all flattened parameters into a single vector
        return torch.cat(weights_list)

def validate_dataset_config(config: Dict, dataset_name: str):
    """Basic validation for required config keys."""
    required_keys = [
        'data_source', 'partitioning_strategy', 'cost_interpretation',
        'dataset_class', 'default_num_clients', 'batch_size', 'rounds', 'runs',
        'metric', 'default_lr', #'learning_rates_try' # LR try is optional
    ]
    # Add conditional requirements based on other keys
    if config.get('partitioning_strategy') == 'pre_split' and dataset_name in ['ISIC', 'IXITiny', 'Heart']:
         if 'site_mappings' not in config.get('source_args', {}):
             print(f"Warning: 'site_mappings' potentially missing in source_args for pre-split dataset {dataset_name}.")
    if config.get('cost_interpretation') == 'concept_shift_param':
         if 'concept_mapping' not in config.get('source_args', {}):
             raise ValueError(f"Dataset config for '{dataset_name}' uses 'concept_shift_param' but is missing 'concept_mapping' in source_args.")
    if config.get('cost_interpretation') == 'feature_shift_param':
         if 'shift_mapping' not in config.get('source_args', {}):
              raise ValueError(f"Dataset config for '{dataset_name}' uses 'feature_shift_param' but is missing 'shift_mapping' in source_args.")


    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise ValueError(f"Dataset config for '{dataset_name}' missing required keys: {missing_keys}")
