from configs import *

def set_seeds(seed_value=1):
    """Set seeds for reproducibility."""
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    np.random.seed(seed_value)
    random.seed(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    torch.use_deterministic_algorithms(True)
    g = torch.Generator()
    g.manual_seed(1)
    return g
g = set_seeds(seed_value=1)

def seed_worker():
    """
    Seeds a DataLoader worker. Call this function via worker_init_fn.
    Ensures that shuffling and augmentations are reproducible across workers.
    """
    worker_seed = torch.initial_seed() % 2**32 # Get the main process seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def cleanup_gpu():
    """Clean up GPU memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

def move_to_device(batch, device):
    """Move batch data to device, handling both single tensors and lists/tuples of tensors."""
    if isinstance(batch, (list, tuple)):
        return [x.to(device) if x is not None else None for x in batch]
    return batch.to(device)

def get_dice_score(output, target, SPATIAL_DIMENSIONS = (2, 3, 4), epsilon=1e-9):
    """DICE score for IXITiny dataset"""
    p0 = output
    g0 = target
    p1 = 1 - p0
    g1 = 1 - g0
    tp = (p0 * g0).sum(axis=SPATIAL_DIMENSIONS)
    fp = (p0 * g1).sum(axis=SPATIAL_DIMENSIONS)
    fn = (p1 * g0).sum(axis=SPATIAL_DIMENSIONS)
    num = 2 * tp
    denom = 2 * tp + fp + fn + epsilon
    dice_score = num / denom
    return dice_score.mean().item()


def get_parameters_for_dataset(DATASET: str) -> Dict:
    """
    Retrieve the default configuration parameters for a given dataset.
    Now reads from the enhanced DEFAULT_PARAMS.
    """
    params = DEFAULT_PARAMS.get(DATASET)
    if not params:
        raise ValueError(f"Dataset {DATASET} is not supported or has no default parameters defined in configs.py.")
    # Ensure backward compatibility if older code uses sizes_per_client directly
    if 'sampling_config' in params and params['sampling_config'] and params['sampling_config'].get('type') == 'fixed_total':
         params['sizes_per_client'] = params['sampling_config']['size']
    return params

def get_default_lr(DATASET):
    """Gets default LR from the dataset's config."""
    params = get_parameters_for_dataset(DATASET)
    lr = params.get('default_lr')
    if lr is None:
         raise ValueError(f"'default_lr' not defined for dataset {DATASET} in configs.py")
    return lr

def get_default_reg(DATASET):
    """Gets default regularization parameter from the dataset's config."""
    params = get_parameters_for_dataset(DATASET)
    reg = params.get('default_reg_param')
    # It's okay if this is None for algos that don't need it
    return reg

def translate_cost(cost, interpretation_type: str) -> Dict:
    """
    Translates the raw 'cost' value based on the configured interpretation type.

    Args:
        cost: The value from the DATASET_COSTS list.
        interpretation_type: String key from config (e.g., 'alpha', 'file_suffix').

    Returns:
        A dictionary containing the interpreted value(s),
        e.g., {'alpha': 0.1}, {'suffix': '_0.08'}, {'key': 0.08}
    """
    if interpretation_type == 'alpha':
        # Assumes cost values in DATASET_COSTS *are* the alpha values
        if cost == 'all': # Handle special case if needed (e.g., high alpha for IID)
            alpha = 1000.0 # Or some large number representing near-IID
        else:
            alpha = float(cost)
        if alpha <= 0:
             raise ValueError(f"Alpha cost interpretation requires positive cost values, got {cost}")
        return {'alpha': alpha}
    elif interpretation_type == 'inv_alpha':
         # Assumes cost values in DATASET_COSTS are 1/alpha
        if cost == 'all':
            alpha = 1000.0
        else:
             inv_a = float(cost)
             if inv_a <= 0:
                  raise ValueError(f"Inverse alpha cost interpretation requires positive cost values, got {cost}")
             alpha = 1.0 / inv_a
        return {'alpha': alpha}
    elif interpretation_type == 'file_suffix':
        # Used for datasets where cost is part of the filename (e.g., tabular, heart)
        # Needs careful formatting if cost is numeric or string like 'all'
        if isinstance(cost, (int, float)):
             # return {'suffix': f"_{cost:.2f}"} # Original format for tabular
             return {'suffix': cost} # Pass the raw cost for the loader to format
        else:
             return {'suffix': cost} # Pass string like 'all' directly
    elif interpretation_type == 'site_mapping_key':
        # Used for datasets where cost is a key to select sites/files (IXI, ISIC)
        return {'key': cost}
    elif interpretation_type == 'ignore':
        # For datasets where cost is irrelevant
        return {}
    else:
        raise ValueError(f"Unknown cost_interpretation type: {interpretation_type}")


class ModelDiversity:
    """Calculates diversity metrics between two clients' models."""
    def __init__(self, client_1, client_2):
        self.client_1 = client_1
        self.client_2 = client_2

    def calculate_weight_divergence(self):
        """Calculate weight divergence metrics between two clients."""
        weights_1 = self._get_weights(self.client_1)
        weights_2 = self._get_weights(self.client_2)
        
        # Normalize weights
        norm_1 = torch.norm(weights_1)
        norm_2 = torch.norm(weights_2)
        w1_normalized = weights_1 / norm_1 if norm_1 != 0 else weights_1
        w2_normalized = weights_2 / norm_2 if norm_2 != 0 else weights_2
        
        # Calculate divergence metrics
        weight_div = torch.norm(w1_normalized - w2_normalized)
        weight_orient = torch.dot(w1_normalized, w2_normalized)
        
        return weight_div.item(), weight_orient.item()

    def _get_weights(self, client):
        """Extract weights from a client's model."""
        weights = []
        state = client.global_state
        for param in state.model.parameters():
            weights.append(param.data.view(-1))  # Flatten weights
        return torch.cat(weights)