from configs import *

def set_seeds(seed_value=1):
    """Set seeds for reproducibility."""
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    np.random.seed(seed_value)
    random.seed(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seeds(seed_value=1)

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

    Args:
        DATASET (str): The name of the dataset (e.g., 'FMNIST', 'CIFAR').
                       Must be one of the keys in `DEFAULT_PARAMS`.

    Returns:
        Dict: A dictionary containing the default parameters for the dataset.

    Raises:
        ValueError: If the specified DATASET is not found in `DEFAULT_PARAMS`.
    """
    params = DEFAULT_PARAMS.get(DATASET)
    if not params:
        raise ValueError(f"Dataset {DATASET} is not supported or has no default parameters defined.")
    return params


def get_default_lr(DATASET):
    lr = DEFAULT_PARAMS.get(DATASET).get('default_lr')
    return lr

def get_default_reg(DATASET):
    reg = DEFAULT_PARAMS.get(DATASET).get('default_reg_param')
    return reg




