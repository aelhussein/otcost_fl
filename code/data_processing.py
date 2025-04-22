"""
Defines TorchDataset wrappers for different data types and datasets used in the project.
Handles data access, transformations, and potentially scaling. Includes base datasets
for partitioning and final datasets for DataLoaders.
"""
import sys
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset as TorchDataset
from sklearn.preprocessing import StandardScaler
from PIL import Image
import traceback
from typing import Dict, Tuple, Any, Optional, List, Union, Callable

# Import necessary libraries, handling optional ones
try:
    import nibabel as nib
except ImportError:
    nib = None
    # No warning here, raise error later if needed by IXI dataset

try:
    import albumentations
except ImportError:
    albumentations = None
    # No warning here, raise error later if needed by ISIC dataset

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

class SyntheticBaseDataset(TorchDataset):
    """Wraps raw generated synthetic data (features + labels)."""
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = features
        self.labels = labels
        # Provide labels for Dirichlet partitioning
        self.targets = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        return self.features[idx], int(self.labels[idx])

class CreditBaseDataset(TorchDataset):
    """Wraps raw loaded credit card data (features + labels)."""
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = features
        self.labels = labels
        self.targets = labels # For Dirichlet partitioning

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        return self.features[idx], int(self.labels[idx])

class SyntheticFeaturesBaseDataset(TorchDataset):
    """Wraps raw generated synthetic features ONLY (no labels)."""
    def __init__(self, features: np.ndarray):
        self.features = features
        # No 'labels' or 'targets' needed for partitioning this type

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> np.ndarray:
        return self.features[idx] # Return only feature

# =============================================================================
# == Final Dataset Wrapper Classes (for DataLoaders) =========================
# =============================================================================
# Instantiated by DataPreprocessor, handle data access, transforms, scaling.

class BaseDataset(TorchDataset):
    """Abstract base class for final datasets used in DataLoaders."""
    def __init__(self, **kwargs):
        pass

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses must implement __len__")

    def __getitem__(self, idx: int) -> Any:
        raise NotImplementedError("Subclasses must implement __getitem__")

# --- Tabular ---
class BaseTabularDataset(BaseDataset):
    """
    Base for final tabular datasets, handling scaling and subset/direct modes.
    Manages data access via indices from a base dataset or directly from arrays.
    """
    def __init__(self,
                 base_dataset: Optional[TorchDataset] = None,
                 indices: Optional[List[int]] = None,
                 split_type: Optional[str] = None,
                 dataset_config: Optional[Dict] = None,
                 scaler_obj: Optional[StandardScaler] = None,
                 X: Optional[np.ndarray] = None,
                 y: Optional[np.ndarray] = None,
                 translated_cost: Optional[Dict] = None,
                 **kwargs):

        super().__init__(**kwargs)
        self.base_dataset = base_dataset
        self.indices = indices if indices is not None else []
        self.split_type = split_type
        self.config = dataset_config if dataset_config is not None else {}
        self.scaler = scaler_obj
        self.X_direct = X
        self.y_direct = y
        # Store translated cost, primarily for concept skew dataset
        self.translated_cost = translated_cost if translated_cost is not None else {}

        # Determine operational mode
        is_subset_mode = (self.base_dataset is not None) and (indices is not None)
        is_direct_mode = (self.X_direct is not None) and (self.y_direct is not None)
        is_empty_mode = (indices is not None) and (not indices)

        if is_subset_mode and not is_direct_mode:
            self.mode = 'subset'
        elif is_direct_mode and not is_subset_mode:
            self.mode = 'direct'
        elif is_empty_mode and not is_direct_mode and (self.base_dataset is not None):
             self.mode = 'empty'
        else:
            raise ValueError("Invalid arguments for BaseTabularDataset.")

        # Config-dependent settings
        self.needs_scaling = (
            'standard_scale' in self.config.get('needs_preprocessing', []) and
            self.config.get('dataset_class') != 'HeartDataset'
        )
        self.is_regression = self.config.get('fixed_classes') is None
        self.target_dtype = torch.float32 if self.is_regression else torch.long

        # Pre-process direct data immediately (if applicable)
        if self.mode == 'direct':
            self._prepare_direct_tensors()

    def _prepare_direct_tensors(self):
        """Converts direct X, y numpy arrays to tensors."""
        self.X_tensor = torch.tensor(self.X_direct, dtype=torch.float32)
        self.y_tensor = torch.tensor(self.y_direct, dtype=self.target_dtype)
        if self.scaler and self.needs_scaling:
            print(f"Warn: Applying provided scaler in {self.__class__.__name__} __init__ (direct mode).")
            scaled_X = self.scaler.transform(self.X_direct)
            self.X_tensor = torch.tensor(scaled_X, dtype=torch.float32)

    def fit_scaler(self) -> Optional[StandardScaler]:
        """Fits StandardScaler on training data (subset mode only)."""
        if not (self.needs_scaling and self.split_type == 'train' and self.mode == 'subset' and self.indices):
            return None # Only fit scaler for training subset requiring scaling

        if not hasattr(self.base_dataset, 'features'):
             print(f"Warn: Base dataset {type(self.base_dataset)} lacks 'features'. Cannot fit scaler.")
             return None

        try:
            print(f"Fitting scaler on {self.split_type} subset ({len(self.indices)} samples) "
                  f"for {self.config.get('dataset_class')}...")
            train_features = self.base_dataset.features[self.indices]
            if train_features.ndim == 1:
                  train_features = train_features.reshape(-1, 1)
            elif train_features.ndim == 0:
                  return None # Cannot scale 0-dim data

            self.scaler = StandardScaler().fit(train_features)
            return self.scaler
        except Exception as e:
            print(f"Error fitting scaler: {e}")
            return None

    def set_scaler(self, scaler: StandardScaler):
        """Sets a pre-fitted scaler (for subset mode val/test splits)."""
        if self.needs_scaling and self.mode == 'subset':
            self.scaler = scaler

    def __len__(self) -> int:
        if self.mode == 'empty':
            return 0
        elif self.mode == 'subset':
            return len(self.indices)
        else: # direct mode
            return len(self.X_direct)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.mode == 'empty':
            raise IndexError("Dataset is empty")

        if self.mode == 'subset':
            original_idx = self.indices[idx]
            # Fetch feature(s)
            if hasattr(self.base_dataset, 'features'):
                feature = self.base_dataset[original_idx]
                # Handle case where base returns (feature, label) tuple
                if isinstance(feature, tuple):
                     feature = feature[0]
            else:
                 raise AttributeError("Base dataset for subset mode lacks 'features'.")

            # Generate label if this is a concept skew dataset, otherwise fetch
            if isinstance(self, SyntheticConceptDataset):
                 label = self._generate_concept_label(feature)
            elif hasattr(self.base_dataset, 'labels'):
                  label = self.base_dataset.labels[original_idx]
            else:
                  raise AttributeError("Base dataset lacks labels, and not a ConceptDataset.")

            # Ensure feature is NumPy array
            if not isinstance(feature, np.ndarray):
                 try:
                      feature = np.array(feature)
                 except Exception:
                      raise TypeError(f"Feature type {type(feature)} not convertible to NumPy array.")

            # Apply scaling if needed
            if self.scaler and self.needs_scaling:
                feature_reshaped = feature.reshape(1, -1)
                feature_scaled = self.scaler.transform(feature_reshaped)
                feature = feature_scaled[0]

            # Convert to tensors
            feature_tensor = torch.tensor(feature, dtype=torch.float32)
            label_tensor = torch.tensor(label, dtype=self.target_dtype)
            return feature_tensor, label_tensor

        elif self.mode == 'direct':
            # Data already processed into tensors
            return self.X_tensor[idx], self.y_tensor[idx]

    def _generate_concept_label(self, feature: np.ndarray) -> int:
         """Placeholder: overridden by SyntheticConceptDataset."""
         raise NotImplementedError("Label generation only implemented in SyntheticConceptDataset.")

# --- Concrete Tabular Dataset Classes ---
class SyntheticDataset(BaseTabularDataset): pass
class CreditDataset(BaseTabularDataset): pass
class HeartDataset(BaseTabularDataset): pass

# --- Concept Skew Labeling Functions ---
def _label_linear_threshold(feature_vector: np.ndarray, threshold: float = 0.5) -> int:
    """Labels based on sum of features compared to a threshold."""
    feature_sum = np.sum(feature_vector)
    return 1 if feature_sum > threshold else 0

# --- NEW: Dataset for Concept Skew ---
class SyntheticConceptDataset(BaseTabularDataset):
    """
    Final dataset wrapper for Synthetic Concept Skew.
    Applies client-specific labeling functions in __getitem__.
    """
    def __init__(self, base_dataset: SyntheticFeaturesBaseDataset, indices: List[int],
                 split_type: str, dataset_config: Dict, scaler_obj: Optional[StandardScaler] = None,
                 translated_cost: Optional[Dict] = None, **kwargs):

        if not isinstance(base_dataset, SyntheticFeaturesBaseDataset):
             raise TypeError("SyntheticConceptDataset requires a SyntheticFeaturesBaseDataset.")

        super().__init__(base_dataset=base_dataset, indices=indices, split_type=split_type,
                         dataset_config=dataset_config, scaler_obj=scaler_obj,
                         translated_cost=translated_cost, **kwargs)

        # Store and interpret concept shift parameters
        self.concept_param = self.translated_cost.get('concept_shift_param', 0.0)
        concept_mapping = self.config.get('source_args', {}).get('concept_mapping', {})
        self.labeling_func_type = concept_mapping.get('function_type', 'linear_threshold_shift')
        self.base_label_param = concept_mapping.get('base_param', 0.5)
        self.shift_range = concept_mapping.get('shift_range', 0.4)

        # Pre-calculate the specific parameter for the labeling function for this client/cost
        self.current_labeling_param = self._calculate_labeling_param()

        print(f"  ConceptDataset ({self.split_type}, {len(self.indices)} samples): "
              f"Type='{self.labeling_func_type}', ShiftParam={self.concept_param:.2f}, "
              f"LabelParam={self.current_labeling_param:.3f}")

    def _calculate_labeling_param(self) -> float:
         """ Calculates the actual parameter value for the labeling function. """
         # Map concept_param (0 to 1) to the actual function parameter range
         # Example: Linear interpolation around the base parameter
         # Shift factor from -1 to 1
         shift_factor = (self.concept_param - 0.5) * 2.0
         # Apply shift relative to half the total range
         param_value = self.base_label_param + shift_factor * (self.shift_range / 2.0)
         # Clip if necessary (e.g., threshold between 0 and 1)
         min_val = self.base_label_param - self.shift_range / 2.0
         max_val = self.base_label_param + self.shift_range / 2.0
         param_value = np.clip(param_value, min_val, max_val)
         return float(param_value) # Ensure it's a standard float

    def _generate_concept_label(self, feature: np.ndarray) -> int:
        """Applies the client-specific labeling function using the pre-calculated parameter."""
        label: int = 0 # Default
        try:
            if self.labeling_func_type == 'linear_threshold_shift':
                label = _label_linear_threshold(feature, threshold=self.current_labeling_param)
            # Add elif clauses for other concept functions here
            # elif self.labeling_func_type == 'rotated_hyperplane':
            #     label = _label_rotated_hyperplane(feature, angle_rad=self.current_labeling_param)
            else:
                print(f"Warning: Unknown concept function type '{self.labeling_func_type}'. Using default.")
                label = _label_linear_threshold(feature, threshold=0.5) # Fallback
        except Exception as e:
             print(f"Error during concept label generation: {e}")
             label = 0 # Default label on error
        return label

# --- Image Datasets ---
class BaseImageDataset(BaseDataset):
    """Base for final image datasets (expects direct X, y input)."""
    def __init__(self, X: Union[np.ndarray, List[str]], y: Union[np.ndarray, List[str]],
                 is_train: bool = True, **kwargs):
        super().__init__(**kwargs)
        if len(X) != len(y):
            raise ValueError("Image features (X) and labels/targets (y) must have the same length.")
        self.X = X # Can be NumPy array of pixels or list of paths
        self.y = y # Can be NumPy array/list of labels or list of paths
        self.is_train = is_train
        self.transform = self.get_transform() # Get and store transform

    def get_transform(self):
        raise NotImplementedError("Subclasses must implement get_transform")

class EMNISTDataset(BaseImageDataset):
    """Final EMNIST dataset wrapper (expects direct numpy X, y)."""
    def __init__(self, X: np.ndarray, y: np.ndarray, is_train: bool = True, **kwargs):
        super().__init__(X=X, y=y, is_train=is_train, **kwargs)

    def get_transform(self) -> transforms.Compose:
        mean, std = (0.1307,), (0.3081,)
        transforms_list = [
            transforms.ToPILImage(),
            transforms.Resize((28, 28))
        ]
        if self.is_train:
            transforms_list.extend([
                transforms.RandomRotation((-15, 15)),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1))
            ])
        transforms_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        return transforms.Compose(transforms_list)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image: np.ndarray = self.X[idx]
        label: int = int(self.y[idx])
        image_tensor: torch.Tensor = self.transform(image)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return image_tensor, label_tensor

class CIFARDataset(BaseImageDataset):
    """Final CIFAR-10 dataset wrapper (expects direct numpy X, y)."""
    def __init__(self, X: np.ndarray, y: np.ndarray, is_train: bool = True, **kwargs):
        super().__init__(X=X, y=y, is_train=is_train, **kwargs)

    def get_transform(self) -> transforms.Compose:
        mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
        transforms_list: List[Callable] = [transforms.ToPILImage()]
        if self.is_train:
            transforms_list.extend([
                transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
                transforms.RandomHorizontalFlip(),
            ])
        transforms_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        return transforms.Compose(transforms_list)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image: np.ndarray = self.X[idx]
        label: int = int(self.y[idx])
        image_tensor: torch.Tensor = self.transform(image)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return image_tensor, label_tensor

class ISICDataset(BaseImageDataset):
    """Final ISIC dataset wrapper (expects image paths X, labels y)."""
    def __init__(self, image_paths: Union[List[str], np.ndarray],
                 labels: np.ndarray, is_train: bool, **kwargs):
        if albumentations is None:
            raise ImportError("Albumentations library is required for ISICDataset.")
        self.sz = 200
        super().__init__(X=list(image_paths), y=labels, is_train=is_train, **kwargs)

    def get_transform(self):
        mean, std = (0.585, 0.500, 0.486), (0.229, 0.224, 0.225)
        if self.is_train:
            return albumentations.Compose([
                albumentations.RandomScale(scale_limit=0.07, p=0.5),
                albumentations.Rotate(limit=50, p=0.5),
                albumentations.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.1, p=0.5),
                albumentations.Flip(p=0.5),
                albumentations.Affine(shear=0.1, p=0.3),
                albumentations.RandomCrop(height=self.sz, width=self.sz, always_apply=True),
                albumentations.CoarseDropout(max_holes=random.randint(1, 8), max_height=16, max_width=16, p=0.3),
                albumentations.Normalize(mean=mean, std=std, max_pixel_value=255.0, always_apply=True),
            ])
        else:
            return albumentations.Compose([
                albumentations.CenterCrop(height=self.sz, width=self.sz, always_apply=True),
                albumentations.Normalize(mean=mean, std=std, max_pixel_value=255.0, always_apply=True),
            ])

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        image_path: str = self.X[idx]
        label: int = int(self.y[idx])
        image_tensor: Optional[torch.Tensor] = None # Default to None

        try:
            image_pil = Image.open(image_path).convert('RGB')
            image = np.array(image_pil)
        except FileNotFoundError:
            print(f"Error: ISIC Image not found: {image_path}.")
            return None, torch.tensor(label, dtype=torch.long) # Return label even if image fails
        except Exception as e:
            print(f"Error loading ISIC image {image_path}: {e}")
            return None, torch.tensor(label, dtype=torch.long)

        try:
            transformed = self.transform(image=image)
            image_transformed = transformed['image']
            # Convert HWC (albumentations) to CHW for PyTorch
            image_tensor = torch.from_numpy(image_transformed.transpose(2, 0, 1)).float()
        except Exception as e:
             print(f"Error applying transforms to ISIC image {image_path}: {e}")
             # Return None for image tensor to indicate failure
             return None, torch.tensor(label, dtype=torch.long)

        label_tensor = torch.tensor(label, dtype=torch.long)
        return image_tensor, label_tensor

class IXITinyDataset(BaseImageDataset):
    """Final IXITiny dataset wrapper (expects image paths X, label paths y)."""
    def __init__(self, image_paths: Union[List[str], np.ndarray],
                 label_paths: Union[List[str], np.ndarray], is_train: bool, **kwargs):
        if MonaiToTensor is None: raise ImportError("MONAI is required for IXITinyDataset.")
        if nib is None: raise ImportError("Nibabel is required for IXITinyDataset.")

        self.nib = nib
        self.monai_transforms = {
            'EnsureChannelFirst': EnsureChannelFirst, 'AsDiscrete': AsDiscrete,
            'Compose': Compose, 'NormalizeIntensity': NormalizeIntensity,
            'Resize': Resize, 'ToTensor': MonaiToTensor
        }
        self.common_shape = (48, 60, 48) # Target shape

        super().__init__(X=list(image_paths), y=list(label_paths), is_train=is_train, **kwargs)

    def get_transform(self):
        """Defines MONAI transforms, applied in __getitem__."""
        Compose = self.monai_transforms['Compose']
        ToTensor = self.monai_transforms['ToTensor']
        EnsureChannelFirst = self.monai_transforms['EnsureChannelFirst']
        Resize = self.monai_transforms['Resize']
        NormalizeIntensity = self.monai_transforms['NormalizeIntensity']
        AsDiscrete = self.monai_transforms['AsDiscrete']

        # Store composed transforms as instance attributes for use in __getitem__
        self.image_transform = Compose([
            ToTensor(track_meta=False),
            # EnsureChannelFirst(channel_dim="no_channel"), # ToTensor should add channel
            Resize(spatial_size=self.common_shape),
            NormalizeIntensity(subtrahend=0.0, divisor=1.0, nonzero=True)
        ])
        self.label_transform = Compose([
            ToTensor(track_meta=False),
            # EnsureChannelFirst(channel_dim="no_channel"),
            Resize(spatial_size=self.common_shape, mode='nearest'),
            AsDiscrete(to_onehot=2) # One-hot: [background, foreground]
        ])
        return None # Indicate internal handling

    def __len__(self) -> int:
        return len(self.X) # self.X has image paths

    def __getitem__(self, idx: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        image_path: str = self.X[idx]
        label_path: str = self.y[idx] # self.y stores label paths

        try:
            image_nii = self.nib.load(image_path)
            label_nii = self.nib.load(label_path)
            # Ensure data is loaded as float32 for image, uint8 for label
            image_data = image_nii.get_fdata(dtype=np.float32)
            label_data = label_nii.get_fdata(dtype=np.uint8)
        except FileNotFoundError as e:
            print(f"Error: Nifti file not found: {e}.")
            return None, None
        except Exception as e:
             print(f"Error loading Nifti file {image_path} or {label_path}: {e}")
             return None, None

        try:
             image_tensor = self.image_transform(image_data)
             label_tensor = self.label_transform(label_data)
        except Exception as e:
             print(f"Error applying MONAI transforms to IXI sample {idx} ({image_path}): {e}")
             traceback.print_exc()
             return None, None

        # Return float tensors (image normalized, label one-hot for Dice)
        return image_tensor.float(), label_tensor.float()