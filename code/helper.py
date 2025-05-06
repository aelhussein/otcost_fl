# helper.py
"""
Core utility functions for the federated learning pipeline.
Streamlined version focusing on essential helpers.
REMOVED: translate_cost function.
"""
import os
import random
import numpy as np
import torch
from contextlib import contextmanager, suppress
from torch import nn, optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, Optional, Tuple, List, Iterator, Any, Callable,  Union
from dataclasses import dataclass, field
import copy
from functools import partial
import sklearn.metrics as metrics
from losses import WeightedCELoss # Custom loss function

# Import global config directly
from configs import DEFAULT_PARAMS # Needed for config helpers

# --- Seeding ---
def set_seeds(seed_value: int = 42):
    """Sets random seeds for PyTorch, NumPy, and Python's random module."""
    torch.manual_seed(seed_value)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed_value)
    np.random.seed(seed_value)
    random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    
# --- Device Handling ---
@contextmanager
def gpu_scope():
    try: yield
    finally:
        with suppress(Exception):
            if torch.cuda.is_available(): torch.cuda.empty_cache()

def move_to_device(batch: Any, device: torch.device) -> Any:
    """Moves a batch (tensor, list/tuple of tensors) to the specified device."""
    if isinstance(batch, (list, tuple)):
        return [item.to(device) if isinstance(item, torch.Tensor) else item for item in batch]
    elif isinstance(batch, torch.Tensor): return batch.to(device)
    return batch


# --- Configuration Helpers ---
def get_parameters_for_dataset(dataset_name: str) -> Dict:
    """Retrieves the config dict for a dataset from global defaults."""
    params = DEFAULT_PARAMS.get(dataset_name)
    if params is None:
        raise ValueError(f"Dataset '{dataset_name}' not found in configs.DEFAULT_PARAMS.")
    if 'dataset_name' not in params: params['dataset_name'] = dataset_name
    return params

def get_default_lr(dataset_name: str) -> float:
    """Gets the default learning rate from the dataset's config."""
    params = get_parameters_for_dataset(dataset_name)
    lr = params.get('default_lr')
    if lr is None: raise ValueError(f"'default_lr' not defined for '{dataset_name}'.")
    return lr

def get_default_reg(dataset_name: str) -> Optional[float]:
    """Gets the default regularization parameter (can be None)."""
    params = get_parameters_for_dataset(dataset_name)
    return params.get('default_reg_param')


# --- Configuration & Data Structures ---

# --- Constants ---
class MetricKey:
    TRAIN_LOSSES = 'train_losses'; VAL_LOSSES = 'val_losses'; TEST_LOSSES = 'test_losses'
    TRAIN_SCORES = 'train_scores'; VAL_SCORES = 'val_scores'; TEST_SCORES = 'test_scores'

# Define Experiment types locally
class ExperimentType:
    LEARNING_RATE = 'learning_rate'; REG_PARAM = 'reg_param'
    EVALUATION = 'evaluation'; DIVERSITY = 'diversity' # Keep diversity type if needed

@dataclass
class TrainerConfig:
    """Training configuration."""
    dataset_name: str
    device: str # Target compute device string (e.g., 'cuda:0')
    learning_rate: float
    batch_size: int
    epochs: int = 1
    rounds: int = 1
    requires_personal_model: bool = False
    algorithm_params: Dict[str, Any] = field(default_factory=dict)
    max_parallel_clients: Optional[int] = None,
    use_weighted_loss: bool = False 


@dataclass
class SiteData:
    """Client data and metadata."""
    site_id: str
    train_loader: DataLoader
    val_loader: Optional[DataLoader] = None
    test_loader: Optional[DataLoader] = None
    weight: float = 1.0
    num_samples: int = 0

    def __post_init__(self):
        if self.num_samples == 0 and self.train_loader and hasattr(self.train_loader, 'dataset'):
            try: self.num_samples = len(self.train_loader.dataset)
            except: self.num_samples = 0

@dataclass
class ModelState:
    """Holds state for one model: current weights, optimizer, criterion, best state."""
    model: nn.Module # Current model weights/arch (CPU)
    optimizer: Optional[optim.Optimizer] = None # Client creates and assigns
    criterion: Union[nn.Module, Callable] = None # Loss function
    best_loss: float = field(init=False, default=float('inf'))
    best_model_state_dict: Optional[Dict] = field(init=False, default=None) # CPU state dict

    def __post_init__(self):
        """Initialize best state based on the initial model."""
        self.model.cpu()
        if self.best_model_state_dict is None:
            self.best_model_state_dict = copy.deepcopy(self.model.state_dict())

    def update_best_state(self, current_val_loss: float) -> bool:
        """Updates best_loss and best_model_state_dict if loss improved."""
        if current_val_loss < self.best_loss:
            self.best_loss = current_val_loss
            self.model.cpu() # Ensure model is on CPU before getting state_dict
            self.best_model_state_dict = copy.deepcopy(self.model.state_dict())
            return True
        return False

    def get_best_model_state_dict(self) -> Optional[Dict]:
        """Returns the best recorded state dict (CPU)."""
        return self.best_model_state_dict

    def get_current_model_state_dict(self) -> Optional[Dict]:
        """Returns the current model state dict (CPU)."""
        self.model.cpu()
        return self.model.state_dict()

    def load_current_model_state_dict(self, state_dict: Dict):
        """Loads state_dict into the current model (CPU)."""
        self.model.cpu()
        self.model.load_state_dict(state_dict)

    def load_best_model_state_dict_into_current(self):
        """Loads the best state into the current model if available."""
        if self.best_model_state_dict:
            self.load_current_model_state_dict(self.best_model_state_dict)
            return True
        return False

    def set_learning_rate(self, lr: float):
        """Updates the learning rate of the optimizer."""
        if self.optimizer:
             for param_group in self.optimizer.param_groups: param_group['lr'] = lr

# --- Minimal Training Manager ---
class TrainingManager:
    """Helper for device placement and batch preparation."""
    def __init__(self, compute_device_str: str):
        self.compute_device = torch.device(compute_device_str)
        self.cpu_device = torch.device('cpu')

    def prepare_batch(self, batch: Any, criterion: Union[nn.Module, Callable]) -> Optional[Tuple[Any, Any, Any]]:
        """Moves batch to compute device and handles labels."""
        if not isinstance(batch, (list, tuple)) or len(batch) < 2: return None
        batch_x, batch_y_orig = batch[0], batch[1]
        batch_x_dev = move_to_device(batch_x, self.compute_device)
        batch_y_dev = move_to_device(batch_y_orig, self.compute_device)
        batch_y_orig_cpu = batch_y_orig.cpu() if isinstance(batch_y_orig, torch.Tensor) else batch_y_orig

        # Process labels on device based on criterion type
        if isinstance(criterion, nn.CrossEntropyLoss) or isinstance(criterion, WeightedCELoss):
             if batch_y_dev.ndim == 2 and batch_y_dev.shape[1] == 1: batch_y_dev = batch_y_dev.squeeze(1)
             batch_y_dev = batch_y_dev.long()
        elif callable(criterion) and criterion.__name__ == 'get_dice_score':
             batch_y_dev = batch_y_dev.float()

        return batch_x_dev, batch_y_dev, batch_y_orig_cpu


# --- Model Comparison (Simplified) ---
class ModelDiversity:
    """Calculates basic diversity metrics between model weights."""
    def __init__(self, client_1, client_2): self.client_1, self.client_2 = client_1, client_2
    def _get_weights(self, client) -> Optional[torch.Tensor]:
        # Simplified access assuming client has .model and potentially .personal_model
        model_to_use = getattr(client, 'personal_model', None) or getattr(client, 'model', None)
        if model_to_use:
            weights = [p.data.detach().view(-1) for p in model_to_use.parameters() if p.data is not None]
            if weights: return torch.cat(weights)
        return None
    def calculate_weight_divergence(self) -> Tuple[float, float]:
        w1, w2 = self._get_weights(self.client_1), self._get_wseights(self.client_2)
        if w1 is None or w2 is None or w1.numel()==0 or w2.numel()==0: return np.nan, np.nan
        n1, n2 = torch.norm(w1), torch.norm(w2)
        w1n, w2n = w1 / (n1 + 1e-9), w2 / (n2 + 1e-9)
        l2 = torch.norm(w1n - w2n, p=2).item()
        cos = torch.dot(w1n, w2n).item()
        return l2, cos
    
# =============================================================================
# == Mixin for Diversity Calculation ==
# =============================================================================
class DiversityMixin:
    """Mixin class to add diversity calculation."""
    def __init__(self, *args, **kwargs):
        # Ensure history exists
        getattr(self, 'history', {}).setdefault('weight_div', [])
        getattr(self, 'history', {}).setdefault('weight_orient', [])
        self.diversity_calculator: Optional[ModelDiversity] = None

    def _setup_diversity_calculator(self):
        if self.diversity_calculator is None and hasattr(self, 'clients') and len(self.clients) >= 2:
             client_ids = list(self.clients.keys())
             self.diversity_calculator = ModelDiversity(self.clients[client_ids[0]], self.clients[client_ids[1]])

    def after_step_hook(self, step_results: List[Tuple[str, Any]]):
        # Check if it was a training step returning state dicts
        is_training_step = any(isinstance(res, dict) and 'state_dict' in res for _, res in step_results)
        if is_training_step:
            self._setup_diversity_calculator()
            if self.diversity_calculator:
                try:
                    div, orient = self.diversity_calculator.calculate_weight_divergence()
                    self.history['weight_div'].append(div)
                    self.history['weight_orient'].append(orient)
                except Exception:
                    self.history['weight_div'].append(np.nan); self.history['weight_orient'].append(np.nan)
        # super().after_step_hook(step_results) # Call next hook if needed

class MetricsCalculator:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        if 'Synthetic_' in dataset_name:
             self.dataset_key = 'Synthetic'
        else:
             self.dataset_key = dataset_name
        self.continuous_outcome = ['Weather']
        self.long_required = ['CIFAR', 'EMNIST', 'ISIC', 'Heart', 'Synthetic', 'Credit']
        self.tensor_metrics = ['IXITiny']
        
    def get_metric_function(self):
        """Returns appropriate metric function based on dataset."""
        metric_mapping = {
            'Synthetic': partial(metrics.f1_score, average='macro'),
            'Credit': partial(metrics.f1_score, average='macro'),
            'Weather': metrics.r2_score,
            'EMNIST': metrics.accuracy_score,
            'CIFAR': metrics.accuracy_score,
            'IXITiny': get_dice_score,
            'ISIC': metrics.balanced_accuracy_score,
            'Heart':metrics.balanced_accuracy_score
        }
        return metric_mapping[self.dataset_key]

    def process_predictions(self, labels, predictions):
        """Process model predictions based on dataset requirements."""
        predictions = predictions.cpu().numpy()
        labels = labels.cpu().numpy()
        
        if self.dataset_key in self.continuous_outcome:
            predictions = np.clip(predictions, -2, 2)
        elif self.dataset_key in self.long_required:
            predictions = predictions.argmax(axis=1)
            
        return labels, predictions

    def calculate_metrics(self, true_labels, predictions):
        """Calculate appropriate metric score."""
        true_labels, predictions_class = self.process_predictions(true_labels, predictions)
        metric_func = self.get_metric_function()
        if self.dataset_key in self.tensor_metrics:
            return metric_func(
                torch.tensor(true_labels, dtype=torch.float32),
                torch.tensor(predictions_class, dtype=torch.float32)
            )
        else:
            return metric_func(
                np.array(true_labels).reshape(-1),
                np.array(predictions_class).reshape(-1)
            )        