"""
Defines TorchDataset wrappers for different data types and datasets used in the project.
This is the single source of truth for Dataset classes.
Handles data access, transformations, and scaling. Includes base datasets
for partitioning and final datasets for DataLoaders.
"""
import sys
import random
import numpy as np
import pandas as pd # Keep for potential type checking if base returns Series
import torch
from torch.utils.data import Dataset as TorchDataset, Subset # Import Subset for type hints
from sklearn.preprocessing import StandardScaler
from PIL import Image
import traceback # For detailed error printing
from typing import Dict, Tuple, Any, Optional, List, Union, Callable

# Import necessary libraries, handling optional ones
try:
    import nibabel as nib
except ImportError:
    nib = None
    print("Warning: Nibabel not found. IXITiny dataset functionality depends on it.")

try:
    import albumentations
    # Ensure ToTensorV2 is available if used by Albumentations transforms
    from albumentations.pytorch import ToTensorV2
except ImportError:
    albumentations = None
    ToTensorV2 = None # Explicitly set to None if albumentations is missing
    print("Warning: Albumentations not found. ISIC dataset functionality depends on it.")


from torchvision import transforms

# Ensure monai is installed if IXITiny is used
try:
    from monai.transforms import (
        EnsureChannelFirst, AsDiscrete, Compose, NormalizeIntensity,
        Resize, ToTensor as MonaiToTensor
    )
except ImportError:
    MonaiToTensor = None
    # Print warning here as it's likely needed if IXITiny is in configs
    print("Warning: MONAI not found. IXITiny dataset will not function correctly.")


# =============================================================================
# == Base Dataset Wrappers (for data loading/partitioning) ==================
# =============================================================================
# Simple wrappers holding raw data, potentially providing .targets for partitioning.
# Used as input to partitioning functions or as `base_dataset` for Final Wrappers.

class SyntheticBaseDataset(TorchDataset):
    """Wraps raw generated synthetic data (features + labels)."""
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        if not isinstance(features, np.ndarray) or not isinstance(labels, np.ndarray):
            raise TypeError("Features and labels must be NumPy arrays.")
        if features.shape[0] != labels.shape[0]:
            raise ValueError(f"Features ({features.shape[0]}) and labels ({labels.shape[0]}) must have the same number of samples.")
        self.features = features
        self.labels = labels.astype(np.int64) # Ensure labels are integers
        # Provide labels for Dirichlet partitioning
        self.targets = self.labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        # Return raw numpy data
        if not 0 <= idx < len(self):
             raise IndexError(f"Index {idx} out of bounds for dataset with length {len(self)}")
        return self.features[idx], int(self.labels[idx]) # Ensure label is standard Python int

class CreditBaseDataset(TorchDataset):
    """Wraps raw loaded credit card data (features + labels)."""
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        if not isinstance(features, np.ndarray) or not isinstance(labels, np.ndarray):
            raise TypeError("Features and labels must be NumPy arrays.")
        if features.shape[0] != labels.shape[0]:
            raise ValueError(f"Features ({features.shape[0]}) and labels ({labels.shape[0]}) must have the same number of samples.")
        self.features = features
        self.labels = labels.astype(np.int64) # Ensure labels are integers
        # Provide labels for Dirichlet partitioning
        self.targets = self.labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        # Return raw numpy data
        if not 0 <= idx < len(self):
             raise IndexError(f"Index {idx} out of bounds for dataset with length {len(self)}")
        return self.features[idx], int(self.labels[idx])

class SyntheticFeaturesBaseDataset(TorchDataset):
    """Wraps raw generated synthetic features ONLY (no labels). Used for Concept Skew."""
    def __init__(self, features: np.ndarray):
        if not isinstance(features, np.ndarray):
            raise TypeError("Features must be a NumPy array.")
        self.features = features
        # No labels or targets attribute needed/provided

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> np.ndarray:
        # Return only the raw feature vector
        if not 0 <= idx < len(self):
             raise IndexError(f"Index {idx} out of bounds for dataset with length {len(self)}")
        return self.features[idx]

# =============================================================================
# == Final Dataset Wrapper Classes (for DataLoaders) =========================
# =============================================================================
# Instantiated by the DataPreprocessor, handle data access, transforms, scaling.

class BaseDataset(TorchDataset):
    """Abstract base class for final datasets used in DataLoaders."""
    def __init__(self,
                 dataset_config: Optional[Dict] = None,
                 split_type: Optional[str] = None, # e.g., 'train', 'val', 'test', 'full'
                 **kwargs):
        self.config = dataset_config if dataset_config is not None else {}
        self.split_type = split_type # Store split type for potential use (e.g., transforms)
        # Basic init, allows kwargs for potential future use
        # Subclasses should call super().__init__(**kwargs)

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses must implement __len__")

    def __getitem__(self, idx: int) -> Any:
        raise NotImplementedError("Subclasses must implement __getitem__")


# --- Concept Skew Labeling Functions (Helpers) ---
# Define different ways to introduce concept shift based on a parameter.

def _label_linear_threshold(feature_vector: np.ndarray, threshold: float = 0.5) -> int:
    """Labels based on sum of features compared to a threshold."""
    if not isinstance(feature_vector, np.ndarray):
        feature_vector = np.array(feature_vector) # Ensure numpy array
    feature_sum = np.sum(feature_vector)
    # Simple thresholding on the sum
    return 1 if feature_sum > threshold else 0

def _label_rotated_hyperplane(feature_vector: np.ndarray, angle_rad: float = 0.0, bias: float = 0.0) -> int:
    """Labels based on which side of a hyperplane (defined by angle and bias) the point falls."""
    if not isinstance(feature_vector, np.ndarray):
        feature_vector = np.array(feature_vector) # Ensure numpy array
    if len(feature_vector) < 2: return 0 # Need at least 2 features for rotation

    # Define a base normal vector (e.g., along the first axis)
    base_normal = np.zeros_like(feature_vector, dtype=float)
    base_normal[0] = 1.0
    base_normal[1] = 0.0 # Assume rotation in first 2 dimensions for simplicity

    # Create a simple 2D rotation matrix
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    # Rotate normal vector components
    rotated_normal_0 = base_normal[0] * cos_a - base_normal[1] * sin_a
    rotated_normal_1 = base_normal[0] * sin_a + base_normal[1] * cos_a

    # Project feature vector onto rotated normal (using only first 2 components)
    projection = feature_vector[0] * rotated_normal_0 + feature_vector[1] * rotated_normal_1
    # Apply bias
    return 1 if projection > bias else 0

# Add other concept shift functions as needed (e.g., feature interactions, label flipping)
# def _label_feature_flip(feature_vector: np.ndarray, base_label: int, flip_rate: float = 0.0, rng: Optional[np.random.Generator]=None) -> int:
#     """Flips the base label with a certain probability."""
#     local_rng = rng if rng else np.random.default_rng()
#     if local_rng.random() < flip_rate:
#         return 1 - base_label
#     return base_label

# --- Tabular ---
class BaseTabularDataset(BaseDataset):
    """
    Base for final tabular datasets, handling scaling and subset/direct modes.
    Manages data access via indices from a base dataset or directly from arrays.
    """
    def __init__(self,
                 # Mode 1: Subset of a base dataset
                 base_dataset: Optional[TorchDataset] = None,
                 indices: Optional[List[int]] = None,
                 # Mode 2: Direct data arrays
                 X: Optional[np.ndarray] = None,
                 y: Optional[np.ndarray] = None,
                 # Common args
                 split_type: Optional[str] = None, # 'train', 'val', 'test', 'full'
                 dataset_config: Optional[Dict] = None,
                 scaler_obj: Optional[StandardScaler] = None,
                 translated_cost: Optional[Dict] = None, # For concept skew params primarily
                 **kwargs):

        super().__init__(dataset_config=dataset_config, split_type=split_type, **kwargs)
        self.base_dataset = base_dataset
        self.indices = indices if indices is not None else []
        self.scaler = scaler_obj
        self.X_direct = X
        self.y_direct = y if y is not None else np.array([]) # Ensure y_direct is array
        self.translated_cost = translated_cost if translated_cost is not None else {}

        # Determine operational mode based on provided arguments
        is_subset_mode = self.base_dataset is not None and indices is not None
        is_direct_mode = self.X_direct is not None # y is required too, checked below
        is_empty_mode = indices is not None and not self.indices # Empty subset

        if is_subset_mode and not is_direct_mode:
            self.mode = 'subset'
            if not hasattr(self.base_dataset, 'features') and not isinstance(self, SyntheticConceptDataset):
                 raise AttributeError("Base dataset for subset mode must have 'features' attribute (unless it's SyntheticConceptDataset).")
            if not hasattr(self.base_dataset, 'targets') and not isinstance(self, SyntheticConceptDataset):
                raise AttributeError("Base dataset for subset mode must have 'targets' attribute (unless it's SyntheticConceptDataset).")
        elif is_direct_mode and not is_subset_mode:
            self.mode = 'direct'
            if self.y_direct.size == 0:
                 raise ValueError("Direct mode requires 'y' array to be provided.")
            if len(self.X_direct) != len(self.y_direct):
                 raise ValueError("Direct mode requires 'X' and 'y' arrays to have the same length.")
        elif is_empty_mode and not is_direct_mode and self.base_dataset is not None:
             self.mode = 'empty'
             print(f"Warning: Initializing BaseTabularDataset in 'empty' mode for split '{self.split_type}'.")
        else:
            raise ValueError("Invalid arguments for BaseTabularDataset: Must provide either (base_dataset, indices) or (X, y).")

        # Config-dependent settings
        self.needs_scaling = (
            'standard_scale' in self.config.get('needs_preprocessing', []) and
            # Specific datasets might handle scaling internally (e.g., Heart loader)
            self.config.get('dataset_class') not in ['HeartDataset']
        )
        self.is_regression = self.config.get('fixed_classes') is None
        self.target_dtype = torch.float32 if self.is_regression else torch.long

        # Pre-process direct data immediately (convert to tensors)
        if self.mode == 'direct':
            self._prepare_direct_tensors()

    def _prepare_direct_tensors(self):
        """Converts direct X, y numpy arrays to tensors and applies scaling if needed."""
        # Convert X to float tensor
        try:
             X_np = np.array(self.X_direct, dtype=np.float32)
             # Apply scaling if scaler is provided AND needed
             if self.scaler and self.needs_scaling:
                  print(f"Applying provided scaler in {self.__class__.__name__} (direct mode).")
                  X_np = self.scaler.transform(X_np) # Assume scaler expects 2D
             self.X_tensor = torch.tensor(X_np, dtype=torch.float32)
        except Exception as e:
             print(f"Error preparing X tensor in direct mode: {e}")
             raise TypeError(f"Failed to convert X_direct to float tensor. Type: {type(self.X_direct)}") from e

        # Convert y to target dtype tensor
        try:
            self.y_tensor = torch.tensor(self.y_direct, dtype=self.target_dtype)
        except Exception as e:
             print(f"Error preparing y tensor in direct mode: {e}")
             # Try converting explicitly to handle potential non-numeric types before tensor conversion fails
             dtype_np = np.float32 if self.is_regression else np.int64
             try:
                  y_np = np.array(self.y_direct, dtype=dtype_np)
                  self.y_tensor = torch.tensor(y_np, dtype=self.target_dtype)
             except Exception as e2:
                 raise TypeError(f"Failed to convert y_direct to tensor (dtype {self.target_dtype}). Type: {type(self.y_direct)}") from e2


    # Note: fit_scaler and set_scaler are moved to the DataPreprocessor class
    # as scaling should ideally be fitted globally on training data.

    def __len__(self) -> int:
        if self.mode == 'empty': return 0
        # Use indices length if subset, otherwise direct array length
        return len(self.indices) if self.mode == 'subset' else len(self.X_direct)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.mode == 'empty':
            raise IndexError("Dataset is empty")
        if not 0 <= idx < len(self):
             raise IndexError(f"Index {idx} out of bounds for dataset with length {len(self)}")

        if self.mode == 'subset':
            # 1. Get original index from the base dataset
            original_idx = self.indices[idx]

            # 2. Fetch feature from base dataset
            try:
                 # Access features; handle if base __getitem__ returns tuple or just feature
                 base_item = self.base_dataset[original_idx]
                 # For SyntheticConceptDataset, base_item is the feature array
                 # For others (SyntheticBase, CreditBase), base_item is (feature, label)
                 feature = base_item[0] if isinstance(base_item, tuple) else base_item
            except IndexError: print(f"Error: Index {original_idx} out of bounds for base dataset."); raise
            except Exception as e: print(f"Error fetching feature for index {original_idx} from base {type(self.base_dataset)}: {e}"); raise

            # Ensure feature is a NumPy array for potential processing
            if not isinstance(feature, np.ndarray):
                try: feature = np.array(feature, dtype=np.float32) # Assume features are numeric
                except Exception as e: raise TypeError(f"Feature type {type(feature)} not convertible to NumPy float array: {e}")

            # 3. Determine the label
            label: Union[int, float] # Type hint
            if isinstance(self, SyntheticConceptDataset):
                # Generate label dynamically using the feature
                label = self._generate_concept_label(feature)
            elif hasattr(self.base_dataset, 'targets'):
                # Fetch pre-existing label from base dataset's targets attribute
                try:
                    label = self.base_dataset.targets[original_idx]
                except IndexError: print(f"Error: Index {original_idx} out of bounds for base dataset targets."); raise
            else:
                raise AttributeError("Cannot determine label source: not ConceptDataset and base lacks '.targets'.")

            # 4. Apply scaling if needed (using the scaler passed during __init__)
            if self.scaler and self.needs_scaling:
                try:
                    # Scaler expects 2D input [n_samples, n_features]
                    feature_reshaped = feature.reshape(1, -1)
                    feature_scaled = self.scaler.transform(feature_reshaped)
                    feature = feature_scaled[0] # Extract the scaled 1D array
                except Exception as e:
                    print(f"Error applying scaler to feature at index {idx} (shape {feature.shape}): {e}")
                    # Decide handling: raise, return unscaled, return None? Raising is safest.
                    raise RuntimeError(f"Scaler transform failed for index {idx}") from e

            # 5. Convert final feature and label to tensors
            try:
                 feature_tensor = torch.tensor(feature, dtype=torch.float32)
                 label_tensor = torch.tensor(label, dtype=self.target_dtype)
            except Exception as e:
                 print(f"Error converting feature/label to tensor at index {idx} (Feat: {type(feature)}, Label: {type(label)}={label}, Dtype: {self.target_dtype}): {e}")
                 raise # Or return sentinel values

            return feature_tensor, label_tensor

        elif self.mode == 'direct':
            # Data already processed into tensors during __init__
            return self.X_tensor[idx], self.y_tensor[idx]

    def _generate_concept_label(self, feature: np.ndarray) -> int:
         """Placeholder: Must be implemented by SyntheticConceptDataset."""
         raise NotImplementedError("Label generation method must be implemented in subclass (e.g., SyntheticConceptDataset).")

# --- Concrete Tabular Dataset Classes ---
class SyntheticDataset(BaseTabularDataset):
    """Final dataset wrapper for standard Synthetic data (Label or Feature skew)."""
    # Feature skew uses 'direct' mode, Label skew uses 'subset' mode.
    pass # Inherits all needed functionality

class CreditDataset(BaseTabularDataset):
    """Final dataset wrapper for Credit data (uses 'subset' mode)."""
    pass # Inherits all needed functionality

class HeartDataset(BaseTabularDataset):
    """Final dataset wrapper for Heart data (uses 'direct' mode). Data is pre-scaled in loader."""
    pass # Inherits all needed functionality

# --- Dataset for Concept Skew ---
class SyntheticConceptDataset(BaseTabularDataset):
    """
    Final dataset wrapper for Synthetic Concept Skew. Overrides label generation.
    Operates in 'subset' mode on a feature-only base dataset (SyntheticFeaturesBaseDataset).
    """
    def __init__(self, base_dataset: SyntheticFeaturesBaseDataset, indices: List[int],
                 split_type: str, dataset_config: Dict, scaler_obj: Optional[StandardScaler] = None,
                 translated_cost: Optional[Dict] = None, **kwargs):

        # Verify base dataset is feature-only type
        if not isinstance(base_dataset, SyntheticFeaturesBaseDataset):
             raise TypeError("SyntheticConceptDataset requires a SyntheticFeaturesBaseDataset as base.")

        # Call parent init for subset mode
        super().__init__(base_dataset=base_dataset, indices=indices, split_type=split_type,
                         dataset_config=dataset_config, scaler_obj=scaler_obj,
                         translated_cost=translated_cost, **kwargs)

        # --- Store concept shift parameters from config and translated cost ---
        self.concept_param = self.translated_cost.get('concept_shift_param', 0.5) # Default to 0.5 (midpoint) if missing
        # Handle 'all' case from translated_cost if needed, map it to a specific shift value (e.g., 1.0)
        if self.translated_cost.get('cost_key_is_all', False):
             print("Warning: 'all' cost key encountered for concept skew. Mapping to concept_param = 1.0")
             self.concept_param = 1.0

        concept_mapping = self.config.get('source_args', {}).get('concept_mapping', {})
        self.labeling_func_type = concept_mapping.get('function_type', 'linear_threshold_shift')
        self.base_label_param = concept_mapping.get('base_param', 0.5) # e.g., base threshold or rotation angle=0
        self.shift_range = concept_mapping.get('shift_range', 0.4) # Max deviation range from base_param

        # --- Pre-calculate the specific parameter for the labeling function ---
        # Map concept_param ([0, 1]) linearly to the parameter's operational range.
        # Assume concept_param=0.5 corresponds to base_label_param.
        # Deviation from base = (concept_param - 0.5) * shift_range
        self.current_labeling_param = self.base_label_param + (self.concept_param - 0.5) * self.shift_range

        # Add clipping if the parameter has natural bounds (e.g., threshold in [0,1], angle in [-pi, pi])
        # This depends on the specific function type and parameter interpretation
        # if self.labeling_func_type == 'linear_threshold_shift':
        #     self.current_labeling_param = np.clip(self.current_labeling_param, 0.0, 1.0)

        print(f"  ConceptDataset ({self.split_type}, {len(self.indices)} samples): "
              f"Type='{self.labeling_func_type}', ConceptParam={self.concept_param:.3f}, "
              f"BaseParam={self.base_label_param:.3f}, Range={self.shift_range:.3f} "
              f"-> Actual Labeling Param={self.current_labeling_param:.4f}")

    # Override the label generation method from BaseTabularDataset
    def _generate_concept_label(self, feature: np.ndarray) -> int:
        """Applies the client-specific labeling function determined during init."""
        label: int = 0 # Default label

        # Select and apply the labeling function based on config
        try:
            if self.labeling_func_type == 'linear_threshold_shift':
                # Use the pre-calculated threshold for this client/cost
                label = _label_linear_threshold(feature, threshold=self.current_labeling_param)
            elif self.labeling_func_type == 'rotated_hyperplane':
                 # current_labeling_param represents the angle in radians directly
                 label = _label_rotated_hyperplane(feature, angle_rad=self.current_labeling_param)
            # Add elif clauses for other function types defined in helpers
            # elif self.labeling_func_type == 'label_flip_rate':
            #      base_label = _label_linear_threshold(feature, threshold=self.base_label_param) # Get base prediction
            #      flip_probability = self.current_labeling_param # Map param directly to flip rate (needs careful range mapping)
            #      label = _label_feature_flip(feature, base_label, flip_rate=flip_probability)
            else:
                print(f"Warning: Unknown concept function type '{self.labeling_func_type}'. Using default threshold (0.5).")
                label = _label_linear_threshold(feature, threshold=0.5) # Fallback to fixed default
        except Exception as e:
             print(f"Error during concept label generation for feature {feature[:5]}...: {e}") # Log snippet of feature
             label = 0 # Default label on error

        return int(label) # Ensure return type is standard Python int

# --- Image Datasets ---
class BaseImageDataset(BaseDataset):
    """Base for final image datasets (expects direct X, y input from preprocessor)."""
    def __init__(self, X: Union[np.ndarray, List[str]], y: Union[np.ndarray, List[str]],
                 is_train: bool = True, # Passed to know which transform to apply
                 dataset_config: Optional[Dict] = None,
                 split_type: Optional[str] = None,
                 **kwargs):
        super().__init__(dataset_config=dataset_config, split_type=split_type, **kwargs)
        # Input X can be NumPy array of pixels or list of paths
        # Input y can be NumPy array of labels/paths or list of paths/labels
        if len(X) != len(y):
            raise ValueError("Image features/paths (X) and labels/paths (y) must have the same length.")
        self.X = X
        self.y = y
        # is_train influences which set of augmentations/transforms are applied
        self.is_train = (self.split_type == 'train') if self.split_type else is_train
        # Initialize transform based on whether it's training or validation/test
        self.transform = self.get_transform()
        if self.transform is None:
             print(f"Info: No transform defined by get_transform() for {self.__class__.__name__} (is_train={self.is_train}). Assuming internal handling.")

    def get_transform(self) -> Optional[Callable]:
        # Must be implemented by subclasses if using external transforms library
        # Can return None if __getitem__ handles all loading and transformation
        raise NotImplementedError("Subclasses must implement get_transform or handle transforms internally.")

class EMNISTDataset(BaseImageDataset):
    """Final EMNIST dataset wrapper (expects direct numpy X, y)."""
    def __init__(self, X: np.ndarray, y: np.ndarray, **kwargs):
        # is_train determined by split_type passed in kwargs
        super().__init__(X=X, y=y.astype(np.int64), **kwargs) # Ensure y is integer

    def get_transform(self) -> transforms.Compose:
        mean, std = (0.1307,), (0.3081,)
        transforms_list = [
            transforms.ToPILImage(), # Input is numpy array HWC (or HW)
            transforms.Resize((28, 28))
        ]
        if self.is_train:
            transforms_list.extend([
                transforms.RandomRotation((-15, 15)),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1))
            ])
        transforms_list.extend([
            transforms.ToTensor(), # Converts PIL to Tensor C H W and scales [0,1]
            transforms.Normalize(mean, std)
        ])
        return transforms.Compose(transforms_list)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image: np.ndarray = self.X[idx] # HWC or HW numpy array
        label: int = int(self.y[idx]) # Ensure label is standard Python int
        # Apply transforms defined in get_transform
        image_tensor: torch.Tensor = self.transform(image)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return image_tensor, label_tensor

class CIFARDataset(BaseImageDataset):
    """Final CIFAR-10 dataset wrapper (expects direct numpy X [N,H,W,C], y)."""
    def __init__(self, X: np.ndarray, y: np.ndarray, **kwargs):
        super().__init__(X=X, y=y.astype(np.int64), **kwargs) # Ensure y is integer

    def get_transform(self) -> transforms.Compose:
        mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
        # Input X is assumed to be HWC numpy array from __getitem__ access
        transforms_list: List[Callable] = [transforms.ToPILImage()] # Input HWC numpy -> PIL Image
        if self.is_train:
            transforms_list.extend([
                transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
                transforms.RandomHorizontalFlip(),
            ])
        transforms_list.extend([
            transforms.ToTensor(), # PIL -> Tensor (C, H, W), scales to [0, 1]
            transforms.Normalize(mean, std)
        ])
        return transforms.Compose(transforms_list)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image: np.ndarray = self.X[idx] # HWC numpy array
        label: int = int(self.y[idx]) # Ensure label is standard Python int
        # Apply transforms
        image_tensor: torch.Tensor = self.transform(image)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return image_tensor, label_tensor

class ISICDataset(BaseImageDataset):
    """Final ISIC dataset wrapper (expects image paths X, labels y). Uses Albumentations."""
    def __init__(self, image_paths: Union[List[str], np.ndarray],
                 labels: np.ndarray, **kwargs):
        if albumentations is None:
            raise ImportError("Albumentations library is required for ISICDataset.")
        self.sz = self.config.get('image_size', 200) # Get size from config or default
        super().__init__(X=list(image_paths), y=labels.astype(np.int64), **kwargs) # Store paths in X

    def get_transform(self):
        mean, std = (0.585, 0.500, 0.486), (0.229, 0.224, 0.225) # Standard ImageNet stats approx.
        if self.is_train:
            # Define training augmentations
            aug_list = [
                albumentations.RandomScale(scale_limit=0.07, p=0.5),
                albumentations.Rotate(limit=50, p=0.5),
                albumentations.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.1, p=0.5),
                albumentations.Flip(p=0.5),
                albumentations.Affine(shear=0.1, p=0.3),
                # Ensure resize happens *before* random crop if needed, or crop first then resize
                # Current: Random Crop ensures fixed size output
                albumentations.RandomCrop(height=self.sz, width=self.sz, always_apply=True),
                albumentations.CoarseDropout(max_holes=random.randint(1, 8), max_height=16, max_width=16, p=0.3),
                albumentations.Normalize(mean=mean, std=std, max_pixel_value=255.0, always_apply=True),
            ]
            # Add ToTensorV2 if albumentations is used for tensor conversion
            if ToTensorV2: aug_list.append(ToTensorV2())
            return albumentations.Compose(aug_list)
        else:
            # Define validation/test transforms (minimal augmentation)
            aug_list = [
                # Resize or CenterCrop to ensure consistent size
                albumentations.CenterCrop(height=self.sz, width=self.sz, always_apply=True),
                # albumentations.Resize(height=self.sz, width=self.sz, always_apply=True), # Alternative
                albumentations.Normalize(mean=mean, std=std, max_pixel_value=255.0, always_apply=True),
            ]
            if ToTensorV2: aug_list.append(ToTensorV2())
            return albumentations.Compose(aug_list)

    def __len__(self) -> int:
        return len(self.X) # self.X contains paths

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_path: str = self.X[idx]
        label: int = int(self.y[idx])
        label_tensor = torch.tensor(label, dtype=torch.long) # Prepare label tensor early

        try:
            # Load image using PIL
            image_pil = Image.open(image_path).convert('RGB')
            # Convert PIL image to NumPy array for Albumentations
            image = np.array(image_pil)

            # Apply Albumentations transforms
            transformed = self.transform(image=image)
            image_transformed = transformed['image'] # Output is NumPy array HWC

            # Convert to PyTorch tensor CHW
            # If ToTensorV2 was used, image_transformed is already a Tensor CHW
            if isinstance(image_transformed, torch.Tensor):
                 image_tensor = image_transformed.float() # Ensure float
            else:
                 # If ToTensorV2 wasn't used, manually convert HWC numpy to CHW tensor
                 image_tensor = torch.from_numpy(image_transformed.transpose(2, 0, 1)).float()

            return image_tensor, label_tensor

        except FileNotFoundError:
            print(f"ERROR: ISIC Image file not found: {image_path}.")
            # Re-raise to potentially be caught by DataLoader error handling or return None
            raise FileNotFoundError(f"Image not found at {image_path}")
        except Exception as e:
             print(f"ERROR processing ISIC sample {idx} ({image_path}): {e}")
             traceback.print_exc()
             # Re-raise a generic exception or return None
             raise RuntimeError(f"Failed to process ISIC sample {idx}") from e


class IXITinyDataset(BaseImageDataset):
    """Final IXITiny dataset wrapper (expects image paths X, label paths y). Uses MONAI."""
    def __init__(self, image_paths: Union[List[str], np.ndarray],
                 label_paths: Union[List[str], np.ndarray], **kwargs):
        if MonaiToTensor is None:
            raise ImportError("MONAI is required for IXITinyDataset but not found.")
        if nib is None:
             raise ImportError("Nibabel is required for IXITinyDataset but not found.")

        self.nib = nib
        # Store MONAI classes needed
        self.monai_transforms_dict = {
            'EnsureChannelFirstd': EnsureChannelFirst(channel_dim="no_channel"), # Add channel dim if needed
            'AsDiscrete': AsDiscrete,
            'Compose': Compose,
            'NormalizeIntensityd': NormalizeIntensity(keys=['image'], subtrahend=0.0, divisor=1.0, nonzero=True),
            'Resized': Resize(keys=['image', 'label'], spatial_size=self.config.get('image_shape', (48, 60, 48))), # Get shape from config
            'ToTensord': MonaiToTensor(keys=['image', 'label'])
        }
        self.common_shape = self.config.get('image_shape', (48, 60, 48)) # Target shape after resizing

        # Call super init AFTER setting up attributes needed by get_transform
        # Y contains label paths for this dataset
        super().__init__(X=list(image_paths), y=list(label_paths), **kwargs)

    def get_transform(self) -> Optional[Callable]:
        """Defines MONAI transforms. Returns None as transforms are applied internally."""
        Compose = self.monai_transforms_dict['Compose']
        # Define transforms for image and label separately or use dictionary transforms
        # Using dictionary transforms is often cleaner with MONAI
        self.internal_transform = Compose([
             self.monai_transforms_dict['ToTensord'],
             # EnsureChannelFirst might be needed if ToTensor doesn't add channel correctly
             # self.monai_transforms_dict['EnsureChannelFirstd'],
             self.monai_transforms_dict['Resized'],
             self.monai_transforms_dict['NormalizeIntensityd'],
             # Apply AsDiscrete to label *after* other spatial transforms
             self.monai_transforms_dict['AsDiscrete'](keys=['label'], to_onehot=2) # One-hot encode label
        ])

        return None # Signal internal handling of transforms in __getitem__

    def __len__(self) -> int:
        return len(self.X) # self.X has image paths

    def __getitem__(self, idx: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        image_path: str = self.X[idx]
        label_path: str = self.y[idx] # self.y stores label paths

        try:
            # Load NIfTI images using nibabel
            image_nii = self.nib.load(image_path)
            label_nii = self.nib.load(label_path)
            # Load data as numpy arrays
            image_data = image_nii.get_fdata(dtype=np.float32)
            label_data = label_nii.get_fdata(dtype=np.uint8) # Load labels as integers

            # Create dictionary for MONAI transforms
            data_dict = {'image': image_data, 'label': label_data}

        except FileNotFoundError as e:
            print(f"Error: Nifti file not found: {e}.")
            # Return None tuple to signal error to DataLoader's collate_fn (requires custom collate)
            # Or raise error
            raise FileNotFoundError(f"Nifti file not found: {e}")
        except Exception as e:
             print(f"Error loading Nifti file {image_path} or {label_path}: {e}")
             raise RuntimeError(f"Failed to load Nifti files for index {idx}") from e

        # Apply MONAI transforms defined during init
        try:
             transformed_dict = self.internal_transform(data_dict)
             image_tensor = transformed_dict['image']
             label_tensor = transformed_dict['label']
        except Exception as e:
             print(f"Error applying MONAI transforms to IXI sample {idx} ({image_path}): {e}")
             traceback.print_exc()
             raise RuntimeError(f"Failed to apply transforms for IXI sample {idx}") from e

        # Ensure correct output types (float for image, float for one-hot label used by Dice)
        return image_tensor.float(), label_tensor.float()