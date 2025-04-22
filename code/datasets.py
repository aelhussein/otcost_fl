"""
Defines TorchDataset wrappers for different data types and datasets used in the project.
Handles data access, transformations, and potentially scaling.
"""
import sys
import random
import numpy as np
import pandas as pd # Keep pandas in case base datasets return Series objects
import torch
from torch.utils.data import Dataset as TorchDataset
from sklearn.preprocessing import StandardScaler
from PIL import Image
import nibabel as nib
from torchvision import transforms
import albumentations
from typing import List, Dict, Tuple, Optional

# Ensure monai is installed if IXITiny is used
try:
    from monai.transforms import (
        EnsureChannelFirst,
        AsDiscrete,
        Compose,
        NormalizeIntensity,
        Resize,
        ToTensor as MonaiToTensor
    )
except ImportError:
    MonaiToTensor = None
    print("Warning: MONAI not found. IXITiny dataset will not work.")

# =============================================================================
# == Base Dataset Wrappers (for data loading/partitioning) ==================
# =============================================================================
# These simple wrappers hold the initially loaded/generated raw data
# and provide the .targets attribute needed by Dirichlet partitioning.

class SyntheticBaseDataset(TorchDataset):
    """Wraps raw generated synthetic data."""
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = features
        self.labels = labels
        self.targets = labels # Crucial for partition_dirichlet_indices

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        # Return raw numpy data
        return self.features[idx], self.labels[idx]

class CreditBaseDataset(TorchDataset):
    """Wraps raw loaded credit card data."""
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = features
        self.labels = labels
        self.targets = labels # Crucial for partition_dirichlet_indices

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        # Return raw numpy data
        return self.features[idx], self.labels[idx]

# =============================================================================
# == Final Dataset Wrapper Classes (for DataLoaders) =========================
# =============================================================================
# These classes are instantiated by the DataPreprocessor for train/val/test splits.
# They handle data access (either from a base dataset via indices or directly),
# apply transformations, manage scaling, and return tensors.

class BaseDataset(TorchDataset):
    """Abstract base for final datasets used in DataLoaders."""
    def __init__(self, **kwargs):
        # Base class constructor, currently does nothing but allows for future extension.
        pass

    def __len__(self):
        raise NotImplementedError("Subclasses must implement __len__")

    def __getitem__(self, idx):
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
                 split_type: Optional[str] = None, # 'train', 'val', 'test'
                 dataset_config: Optional[Dict] = None,
                 scaler_obj: Optional[StandardScaler] = None, # Pre-fitted scaler
                 X: Optional[np.ndarray] = None, # Direct feature array
                 y: Optional[np.ndarray] = None, # Direct label array
                 **kwargs):
        super().__init__(**kwargs) # Call parent's __init__

        # Store initialization parameters
        self.base_dataset = base_dataset
        self.indices = indices if indices is not None else []
        self.split_type = split_type
        self.config = dataset_config if dataset_config is not None else {}
        self.scaler = scaler_obj # May be None, especially for train split initially
        self.X_direct = X
        self.y_direct = y

        # Determine operational mode based on provided arguments
        is_subset_mode = self.base_dataset is not None and indices is not None # Note: Using 'indices' passed to init
        is_direct_mode = self.X_direct is not None and self.y_direct is not None
        is_empty_mode = indices is not None and not indices # Special case for empty subset

        if is_subset_mode and not is_direct_mode:
            self.mode = 'subset'
        elif is_direct_mode and not is_subset_mode:
            self.mode = 'direct'
        elif is_empty_mode and not is_direct_mode and self.base_dataset is not None:
             self.mode = 'empty'
        else:
            # Raise error for invalid or ambiguous combinations
            raise ValueError(
                "Invalid arguments for BaseTabularDataset. Provide EITHER "
                "(base_dataset and indices) OR (X and y direct arrays)."
                f" Received: base_dataset={base_dataset is not None}, "
                f"indices={indices is not None}, X={X is not None}, y={y is not None}"
            )

        # Configuration-dependent settings
        self.needs_scaling = (
            'standard_scale' in self.config.get('needs_preprocessing', []) and
            self.config.get('dataset_class') != 'HeartDataset' # Heart scaling done in loader
        )
        self.is_regression = self.config.get('fixed_classes') is None
        self.target_dtype = torch.float32 if self.is_regression else torch.long

        # Pre-process direct data immediately
        if self.mode == 'direct':
            self.X_tensor = torch.tensor(self.X_direct, dtype=torch.float32)
            self.y_tensor = torch.tensor(self.y_direct, dtype=self.target_dtype)
            # Apply scaler immediately if provided and needed (e.g., from _process_xy_dict, though unlikely now)
            if self.scaler and self.needs_scaling:
                 print(f"Warning: Applying provided scaler to direct data in {self.config.get('dataset_class')} __init__.")
                 self.X_tensor = torch.tensor(self.scaler.transform(self.X_direct), dtype=torch.float32)

    def fit_scaler(self) -> Optional[StandardScaler]:
        """Fits StandardScaler on training data (if applicable) and returns it."""
        if self.needs_scaling and self.split_type == 'train' and self.mode == 'subset' and self.indices:
            print(f"Fitting scaler on {self.split_type} subset data ({len(self.indices)} samples) for {self.config.get('dataset_class')}...")
            # Ensure features are accessible from base dataset
            if not hasattr(self.base_dataset, 'features'):
                 print(f"Warning: Base dataset {type(self.base_dataset)} lacks 'features' attribute for scaler fitting.")
                 return None
            try:
                 # Extract only the training features using indices
                 train_features = self.base_dataset.features[self.indices]
                 # Fit the scaler
                 self.scaler = StandardScaler().fit(train_features)
                 return self.scaler
            except Exception as e:
                 print(f"Error fitting scaler: {e}")
                 return None
        # Return None if not applicable (not train split, not subset mode, or no scaling needed)
        return None

    def set_scaler(self, scaler: StandardScaler):
        """Sets a pre-fitted scaler (usually for val/test sets in subset mode)."""
        if self.needs_scaling and self.mode == 'subset':
            self.scaler = scaler

    def __len__(self):
        if self.mode == 'subset':
            return len(self.indices)
        elif self.mode == 'direct':
            return len(self.X_direct) # Length based on direct data array
        else: # 'empty' mode
            return 0

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.mode == 'empty':
            raise IndexError("Dataset is empty")

        if self.mode == 'subset':
            # 1. Get the original index from the base dataset
            original_idx = self.indices[idx]
            # 2. Fetch raw data from the base dataset
            # Assuming base_dataset returns numpy arrays or compatible types
            try:
                feature, label = self.base_dataset[original_idx]
            except IndexError:
                 print(f"Error: Index {original_idx} out of bounds for base dataset (size {len(self.base_dataset)}).")
                 raise
            # Ensure feature is a numpy array for scaler
            if not isinstance(feature, np.ndarray):
                 # Attempt conversion if possible (e.g., from pandas Series)
                 try: feature = np.array(feature)
                 except: raise TypeError(f"Feature from base dataset is not a NumPy array (type: {type(feature)})")

            # 3. Apply scaler if available and needed
            if self.scaler and self.needs_scaling:
                # Scaler expects 2D array: [n_samples, n_features]
                feature_reshaped = feature.reshape(1, -1)
                feature_scaled = self.scaler.transform(feature_reshaped)
                feature = feature_scaled[0] # Get the scaled 1D array back

            # 4. Convert to tensors
            feature_tensor = torch.tensor(feature, dtype=torch.float32)
            label_tensor = torch.tensor(label, dtype=self.target_dtype)
            return feature_tensor, label_tensor

        elif self.mode == 'direct':
            # Data is already pre-processed (scaled if needed) and converted to tensors in __init__
            return self.X_tensor[idx], self.y_tensor[idx]

# --- Concrete Tabular Dataset Classes ---
class SyntheticDataset(BaseTabularDataset):
    """Final dataset wrapper for Synthetic data."""
    pass # Inherits all functionality from BaseTabularDataset

class CreditDataset(BaseTabularDataset):
    """Final dataset wrapper for Credit data."""
    pass # Inherits all functionality from BaseTabularDataset

class HeartDataset(BaseTabularDataset):
    """Final dataset wrapper for Heart data (uses 'direct' mode)."""
    pass # Inherits all functionality, expects pre-scaled data in X, y

# --- Image ---
class BaseImageDataset(BaseDataset):
    """Base for final image datasets."""
    def __init__(self, X, y, is_train: bool = True, **kwargs): # Expects direct X, y (arrays or paths)
        super().__init__(**kwargs)
        self.X = X # Can be numpy array or list of paths
        self.y = y
        self.is_train = is_train
        self.transform = self.get_transform() # Get transform based on is_train

    def get_transform(self):
        # Must be implemented by subclasses based on self.is_train
        raise NotImplementedError

class EMNISTDataset(BaseImageDataset):
    """Final EMNIST dataset wrapper (expects numpy X, y)."""
    def __init__(self, X: np.ndarray, y: np.ndarray, is_train: bool = True, **kwargs):
        # is_train determined by the preprocessor before calling this
        super().__init__(X=X, y=y, is_train=is_train, **kwargs)

    def get_transform(self) -> transforms.Compose:
        mean, std = (0.1307,), (0.3081,)
        transforms_list = [
            transforms.ToPILImage(), # Input expected to be numpy (H, W) or (H, W, C)
            transforms.Resize((28, 28))
        ]
        if self.is_train:
            transforms_list.extend([
                transforms.RandomRotation((-15, 15)),
                transforms.RandomAffine(0, translate=(0.1, 0.1), scale=(0.9, 1.1))
            ])
        transforms_list.extend([
            transforms.ToTensor(), # Converts to (C, H, W) and scales to [0, 1]
            transforms.Normalize(mean, std)
        ])
        return transforms.Compose(transforms_list)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image = self.X[idx] # Direct numpy array
        label = self.y[idx]
        image_tensor = self.transform(image)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return image_tensor, label_tensor

class CIFARDataset(BaseImageDataset):
    """Final CIFAR-10 dataset wrapper (expects numpy X, y)."""
    def __init__(self, X: np.ndarray, y: np.ndarray, is_train: bool = True, **kwargs):
        super().__init__(X=X, y=y, is_train=is_train, **kwargs)

    def get_transform(self) -> transforms.Compose:
        mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
        if self.is_train:
            # Training transforms
            transforms_list = [
                transforms.ToPILImage(), # Input should be HWC numpy array
                transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(), # HWC -> CHW, scales to [0,1]
                transforms.Normalize(mean, std)
            ]
        else:
            # Validation/Test transforms (No augmentation)
            transforms_list = [
                transforms.ToPILImage(), # Input should be HWC numpy array
                transforms.ToTensor(), # HWC -> CHW, scales to [0,1]
                transforms.Normalize(mean, std)
            ]
        return transforms.Compose(transforms_list)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image = self.X[idx] # Direct numpy array (H, W, C)
        label = self.y[idx]
        image_tensor = self.transform(image)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return image_tensor, label_tensor

class ISICDataset(BaseImageDataset):
    """Final ISIC dataset wrapper (expects image paths X, labels y)."""
    def __init__(self, image_paths: List[str], labels: np.ndarray, is_train: bool, **kwargs):
        self.sz = 200 # Image size - could be moved to config
        super().__init__(X=image_paths, y=labels, is_train=is_train, **kwargs) # Pass paths as X

    def get_transform(self) -> albumentations.Compose:
        mean, std = (0.585, 0.500, 0.486), (0.229, 0.224, 0.225)
        if self.is_train:
            return albumentations.Compose([
                albumentations.RandomScale(0.07),
                albumentations.Rotate(50),
                albumentations.RandomBrightnessContrast(0.15, 0.1),
                albumentations.Flip(p=0.5),
                albumentations.Affine(shear=0.1),
                albumentations.RandomCrop(self.sz, self.sz),
                albumentations.CoarseDropout(max_holes=random.randint(1, 8), max_height=16, max_width=16),
                albumentations.Normalize(mean=mean, std=std, always_apply=True),
            ])
        else:
            return albumentations.Compose([
                albumentations.CenterCrop(self.sz, self.sz),
                albumentations.Normalize(mean=mean, std=std, always_apply=True),
            ])

    def __len__(self):
        return len(self.X) # self.X contains paths

    def __getitem__(self, idx: int) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        image_path = self.X[idx]
        label = self.y[idx]
        try:
            # Read image as numpy array HWC for albumentations
            image = np.array(Image.open(image_path).convert('RGB'))
        except FileNotFoundError:
            print(f"Error: ISIC Image file not found: {image_path}.")
            # Return placeholder or handle error downstream
            return None, torch.tensor(label, dtype=torch.long)
        except Exception as e:
            print(f"Error loading ISIC image {image_path}: {e}")
            return None, torch.tensor(label, dtype=torch.long)

        # Apply albumentations transforms
        try:
            transformed = self.transform(image=image)
            image = transformed['image'] # Result is HWC numpy array
        except Exception as e:
             print(f"Error applying transforms to ISIC image {image_path}: {e}")
             # Handle transform error, maybe return original image tensor or None
             return None, torch.tensor(label, dtype=torch.long)

        # Convert to tensor CHW float
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float()
        label_tensor = torch.tensor(label, dtype=torch.long)

        return image_tensor, label_tensor

class IXITinyDataset(BaseImageDataset):
    """Final IXITiny dataset wrapper (expects image paths X, label paths y)."""
    def __init__(self, image_paths: List[str], label_paths: List[str], is_train: bool, **kwargs):
        if MonaiToTensor is None:
            raise ImportError("MONAI is required for IXITinyDataset but not found.")

        self.nib = nib # Keep nibabel accessible
        # Define MONAI transforms here
        self.monai_transforms = {
            'EnsureChannelFirst': EnsureChannelFirst,
            'AsDiscrete': AsDiscrete,
            'Compose': Compose,
            'NormalizeIntensity': NormalizeIntensity,
            'Resize': Resize,
            'ToTensor': MonaiToTensor
        }
        self.common_shape = (48, 60, 48) # Could move to config

        # Call super init AFTER setting up MONAI transforms
        super().__init__(X=image_paths, y=label_paths, is_train=is_train, **kwargs) # Pass paths as X and y

    def get_transform(self):
        # MONAI transforms are applied sequentially in __getitem__
        # Define the composed transforms here for clarity
        Compose = self.monai_transforms['Compose']
        ToTensor = self.monai_transforms['ToTensor']
        EnsureChannelFirst = self.monai_transforms['EnsureChannelFirst']
        Resize = self.monai_transforms['Resize']
        NormalizeIntensity = self.monai_transforms['NormalizeIntensity']
        AsDiscrete = self.monai_transforms['AsDiscrete']

        # Store composed transforms as instance attributes
        self.image_transform = Compose([
            ToTensor(), # Adds channel dim, scales potentially
            EnsureChannelFirst(channel_dim="no_channel"), # Ensure channel is first if ToTensor didn't
            Resize(self.common_shape),
            NormalizeIntensity() # Normalize after resize
        ])
        self.label_transform = Compose([
            ToTensor(),
            EnsureChannelFirst(channel_dim="no_channel"),
            Resize(self.common_shape),
            AsDiscrete(to_onehot=2) # One-hot encode labels (assuming 2 classes)
        ])
        # This method doesn't return a single transform object like others
        return None # Signal internal handling

    def __len__(self):
        return len(self.X) # self.X has image paths

    def __getitem__(self, idx: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        image_path = self.X[idx]
        label_path = self.y[idx]

        try:
            # Load NIfTI images
            image = self.nib.load(image_path).get_fdata(dtype=np.float32)
            label = self.nib.load(label_path).get_fdata(dtype=np.uint8)
        except FileNotFoundError as e:
            print(f"Error: Nifti file not found: {e}.")
            return None, None # Handle missing file downstream
        except Exception as e:
             print(f"Error loading Nifti file {image_path} or {label_path}: {e}")
             return None, None

        # Apply MONAI transforms
        try:
             image_tensor = self.image_transform(image)
             label_tensor = self.label_transform(label)
        except Exception as e:
             print(f"Error applying MONAI transforms to IXI sample {idx}: {e}")
             return None, None

        # Ensure correct types after transforms
        # DICE loss often expects float targets (probabilities or one-hot)
        return image_tensor.float(), label_tensor.float()