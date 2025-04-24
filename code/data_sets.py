# data_sets.py
"""
Defines final Dataset classes for use with PyTorch DataLoaders.
Each class takes raw, client-specific, split data (NumPy arrays or paths)
and handles appropriate transformations and tensor conversions internally.
Streamlined version: No base classes, no external scaling (except Heart internal),
simplified SyntheticDataset. Assumes necessary libraries are installed.
"""
import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset
from PIL import Image
from typing import Dict, Tuple, Any, Optional, List, Union, Callable

# Assume necessary libraries are installed
import nibabel as nib
from monai.transforms import (LoadImaged, Resized, NormalizeIntensityd,
                              AsDiscreted, ToTensord, Compose)
import albumentations
from albumentations.pytorch import ToTensorV2
from torchvision import transforms

# =============================================================================
# == Final Dataset Wrapper Classes ==
# =============================================================================

class SyntheticDataset(TorchDataset):
    """Final Dataset wrapper for all synthetic data. Converts NumPy to Tensor."""
    def __init__(self, X_np: np.ndarray, y_np: np.ndarray):
        self.features = X_np
        self.labels = y_np.astype(np.int64)

    def __len__(self) -> int: return len(self.features) # Use features array length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        feature = self.features[idx]
        label = self.labels[idx]
        feature_tensor = torch.tensor(feature, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return feature_tensor, label_tensor


class CreditDataset(TorchDataset):
    """Final Dataset wrapper for Credit Card data."""
    def __init__(self, X_np: np.ndarray, y_np: np.ndarray):
        self.features = X_np
        self.labels = y_np.astype(np.int64)

    def __len__(self) -> int: return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        feature = self.features[idx]
        label = self.labels[idx]
        feature_tensor = torch.tensor(feature, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return feature_tensor, label_tensor


class HeartDataset(TorchDataset):
    """Final Dataset wrapper for Heart Disease data. Handles internal scaling."""
    def __init__(self, X_np: np.ndarray, y_np: np.ndarray, dataset_config: dict, **kwargs): # Accept kwargs
        self.features_unscaled = X_np
        self.labels = y_np.astype(np.int64)
        source_args = dataset_config.get('source_args', {})
        self.feature_names = source_args.get('feature_names', [])
        self.cols_to_scale = source_args.get('cols_to_scale', [])
        self.scale_values = source_args.get('scale_values', {})
        # Pre-calculate indices for efficiency
        self.scale_indices = [self.feature_names.index(col) for col in self.cols_to_scale if col in self.feature_names and col in self.scale_values]
        self.means = {col: self.scale_values[col][0] for col in self.cols_to_scale if col in self.scale_values}
        self.std_devs = {col: max(np.sqrt(self.scale_values[col][1]), 1e-9) for col in self.cols_to_scale if col in self.scale_values} # Avoid div by zero

    def __len__(self) -> int: return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        feature_unscaled = self.features_unscaled[idx]
        label = self.labels[idx]
        feature_scaled = feature_unscaled.copy().astype(np.float32)
        for col_idx in self.scale_indices:
            col_name = self.feature_names[col_idx]
            feature_scaled[col_idx] = (feature_scaled[col_idx] - self.means[col_name]) / self.std_devs[col_name]

        feature_tensor = torch.tensor(feature_scaled, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return feature_tensor, label_tensor


class EMNISTDataset(TorchDataset):
    """Final EMNIST dataset wrapper."""
    def __init__(self, X_np: np.ndarray, y_np: np.ndarray, split_type: str, **kwargs): # Accept kwargs
        self.images = X_np
        self.labels = y_np.astype(np.int64)
        self.is_train = (split_type == 'train')
        self.transform = self._get_transform()

    def _get_transform(self) -> Callable:
        mean, std = (0.1307,), (0.3081,)
        transforms_list = [transforms.ToPILImage(), transforms.Resize((28, 28))]
        if self.is_train:
            transforms_list.extend([
                transforms.RandomRotation((-15, 15)),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1))
            ])
        transforms_list.extend([transforms.ToTensor(), transforms.Normalize(mean, std)])
        return transforms.Compose(transforms_list)

    def __len__(self) -> int: return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image, label = self.images[idx], self.labels[idx]
        image_tensor = self.transform(image)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return image_tensor, label_tensor


class CIFARDataset(TorchDataset):
    """Final CIFAR-10 dataset wrapper."""
    def __init__(self, X_np: np.ndarray, y_np: np.ndarray, split_type: str, **kwargs): # Accept kwargs
        self.images = X_np
        self.labels = y_np.astype(np.int64)
        self.is_train = (split_type == 'train')
        self.transform = self._get_transform()

    def _get_transform(self) -> Callable:
        mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
        transforms_list = [transforms.ToPILImage()]
        if self.is_train:
            transforms_list.extend([
                transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
                transforms.RandomHorizontalFlip(),
            ])
        transforms_list.extend([transforms.ToTensor(), transforms.Normalize(mean, std)])
        return transforms.Compose(transforms_list)

    def __len__(self) -> int: return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image, label = self.images[idx], self.labels[idx]
        image_tensor = self.transform(image)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return image_tensor, label_tensor


class ISICDataset(TorchDataset):
    """Final ISIC dataset wrapper (expects image paths, labels). Uses Albumentations."""
    def __init__(self, image_paths: List[str], labels_np: np.ndarray, split_type: str, dataset_config: dict):
        self.image_paths = image_paths
        self.labels = labels_np.astype(np.int64)
        self.is_train = (split_type == 'train')
        self.sz = dataset_config.get('source_args', {}).get('image_size', 200)
        self.transform = self._get_transform()

    def _get_transform(self) -> Callable:
        mean, std = (0.585, 0.500, 0.486), (0.229, 0.224, 0.225)
        common_end = [
            albumentations.Normalize(mean=mean, std=std, max_pixel_value=255.0, always_apply=True),
            ToTensorV2()
        ]
        if self.is_train:
            aug_list = [
                albumentations.RandomScale(scale_limit=0.07, p=0.5),
                albumentations.Rotate(limit=50, p=0.5),
                albumentations.RandomBrightnessContrast(0.15, 0.1, p=0.5),
                albumentations.Flip(p=0.5),
                albumentations.Affine(shear=0.1, p=0.3),
                albumentations.RandomCrop(height=self.sz, width=self.sz, always_apply=True),
                albumentations.CoarseDropout(max_holes=8, max_height=16, max_width=16, p=0.3),
                *common_end
            ]
        else:
            aug_list = [albumentations.CenterCrop(height=self.sz, width=self.sz, always_apply=True), *common_end]
        return albumentations.Compose(aug_list)

    def __len__(self) -> int: return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_path, label = self.image_paths[idx], self.labels[idx]
        try:
            image = np.array(Image.open(image_path).convert('RGB'))
            transformed = self.transform(image=image)
            image_tensor = transformed['image'].float()
            label_tensor = torch.tensor(label, dtype=torch.long)
            return image_tensor, label_tensor
        except Exception as e: raise RuntimeError(f"Failed processing ISIC sample {idx}: {image_path}") from e


class IXITinyDataset(TorchDataset):
    """Final IXITiny dataset wrapper (expects image/label paths). Uses MONAI."""
    def __init__(self, image_paths: List[str], label_paths: List[str], split_type: str, dataset_config: dict):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.is_train = (split_type == 'train')
        common_shape = dataset_config.get('source_args', {}).get('image_shape', (48, 60, 48))
        self.transform = self._get_transform(common_shape)

    def _get_transform(self, spatial_size) -> Callable:
        keys = ["image", "label"]
        load_dict = LoadImaged(keys=keys, image_only=True, ensure_channel_first=True)
        normalize_dict = NormalizeIntensityd(keys="image", subtrahend=0.0, divisor=1.0, nonzero=True)
        resize_dict = Resized(keys=keys, spatial_size=spatial_size, mode=("bilinear", "nearest"))
        discretize_dict = AsDiscreted(keys="label", to_onehot=2)
        to_tensor_dict = ToTensord(keys=keys)
        # Add train-specific augmentations here if needed
        all_transforms = [load_dict, normalize_dict, resize_dict, discretize_dict, to_tensor_dict]
        return Compose(all_transforms)

    def __len__(self) -> int: return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        data_dict = {"image": self.image_paths[idx], "label": self.label_paths[idx]}
        try:
            transformed_dict = self.transform(data_dict)
            image_tensor = transformed_dict['image']
            label_tensor = transformed_dict['label']
            return image_tensor.float(), label_tensor.float()
        except Exception as e: raise RuntimeError(f"Failed processing IXI sample {idx}: {self.image_paths[idx]}") from e