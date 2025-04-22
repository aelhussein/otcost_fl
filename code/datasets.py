"""
Defines TorchDataset wrappers for different data types and datasets used in the project.
Handles data access, transformations, and potentially scaling.
"""
import sys
import random
import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset
from sklearn.preprocessing import StandardScaler
from PIL import Image
import nibabel as nib
from torchvision import transforms
import albumentations
# Ensure monai is installed if IXITiny is used
try:
    from monai.transforms import EnsureChannelFirst, AsDiscrete,Compose,NormalizeIntensity,Resize,ToTensor as MonaiToTensor
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
    def __init__(self, features, labels): self.features = features; self.labels = labels; self.targets = labels
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx): return self.features[idx], self.labels[idx]

class CreditBaseDataset(TorchDataset):
    """Wraps raw loaded credit card data."""
    def __init__(self, features, labels): self.features = features; self.labels = labels; self.targets = labels
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx): return self.features[idx], self.labels[idx]

# =============================================================================
# == Final Dataset Wrapper Classes (for DataLoaders) =========================
# =============================================================================
# These classes are instantiated by the DataPreprocessor for train/val/test splits.
# They handle data access (either from a base dataset via indices or directly),
# apply transformations, manage scaling, and return tensors.

class BaseDataset(TorchDataset):
    """Abstract base for final datasets used in DataLoaders."""
    def __init__(self, **kwargs): pass
    def __len__(self): raise NotImplementedError
    def __getitem__(self, idx): raise NotImplementedError

# --- Tabular ---
class BaseTabularDataset(BaseDataset):
    """Base for final tabular datasets, handling scaling and subset/direct modes."""
    # (Keep class code as provided in the previous data_processing.py)
    def __init__(self, base_dataset=None, indices=None, split_type=None, dataset_config=None, scaler_obj=None, X=None, y=None, **kwargs):
        super().__init__(**kwargs); self.base_dataset = base_dataset; self.indices = indices; self.split_type = split_type; self.config = dataset_config; self.scaler = scaler_obj; self.X_direct = X; self.y_direct = y
        self.needs_scaling = 'standard_scale' in self.config.get('needs_preprocessing', []) and self.config.get('dataset_class') != 'HeartDataset' # Check added for Heart
        self.is_regression = self.config.get('fixed_classes') is None; self.target_dtype = torch.float32 if self.is_regression else torch.long
        if self.base_dataset is not None and self.indices is not None: self.mode = 'subset'; assert self.X_direct is None and self.y_direct is None, "Cannot provide both base_dataset/indices and X/y direct."
        elif self.X_direct is not None and self.y_direct is not None: self.mode = 'direct'; assert self.base_dataset is None and self.indices is None, "Cannot provide both base_dataset/indices and X/y direct."
        elif self.indices is not None and not self.indices: self.mode = 'empty'
        else: raise ValueError("Must provide either (base_dataset, indices) or (X, y).")
        if self.mode == 'direct': self.X_tensor = torch.tensor(self.X_direct, dtype=torch.float32); self.y_tensor = torch.tensor(self.y_direct, dtype=self.target_dtype)

    def fit_scaler(self):
        if self.needs_scaling and self.split_type == 'train' and self.mode == 'subset' and self.indices:
            print(f"Fitting scaler on {self.split_type} subset data ({len(self.indices)} samples)..."); train_features = self.base_dataset.features[self.indices]; self.scaler = StandardScaler().fit(train_features); return self.scaler
        return None

    def set_scaler(self, scaler):
        if self.needs_scaling and self.mode == 'subset': self.scaler = scaler

    def __len__(self): return len(self.indices) if self.mode == 'subset' else (len(self.X_direct) if self.mode == 'direct' else 0)

    def __getitem__(self, idx):
        if self.mode == 'empty': raise IndexError("Dataset is empty")
        if self.mode == 'subset':
            original_idx = self.indices[idx]; feature, label = self.base_dataset[original_idx];
            if self.scaler: feature = self.scaler.transform(feature.reshape(1, -1))[0]
            feature_tensor = torch.tensor(feature, dtype=torch.float32); label_tensor = torch.tensor(label, dtype=self.target_dtype); return feature_tensor, label_tensor
        elif self.mode == 'direct':
            return self.X_tensor[idx], self.y_tensor[idx]

class SyntheticDataset(BaseTabularDataset): pass
class CreditDataset(BaseTabularDataset): pass
class HeartDataset(BaseTabularDataset): pass # Inherits, uses 'direct' mode

# --- Image ---
class BaseImageDataset(BaseDataset):
    """Base for final image datasets."""
    # (Keep class code as provided in the previous data_processing.py)
    def __init__(self, X=None, y=None, base_dataset=None, indices=None, split_type=None, dataset_config=None, scaler_obj=None, is_train=True, **kwargs):
        super().__init__(**kwargs)
        self.X = X; self.y = y; self.base_dataset = base_dataset; self.indices = indices; self.mode = 'direct' if X is not None else ('subset' if indices is not None else 'empty')
        self.is_train = is_train if split_type is None else (split_type == 'train')
        self.transform = self.get_transform()
    def get_transform(self): raise NotImplementedError

class EMNISTDataset(BaseImageDataset):
    """Final EMNIST dataset wrapper."""
    # (Keep class code as provided in the previous data_processing.py - init slightly adapted)
    def __init__(self, base_dataset=None, indices=None, split_type=None, dataset_config=None, scaler_obj=None, X=None, y=None, is_train=True, **kwargs):
        super().__init__(X=X, y=y, base_dataset=base_dataset, indices=indices, split_type=split_type, dataset_config=dataset_config, scaler_obj=scaler_obj, is_train=is_train, **kwargs)
    def get_transform(self): 
        mean, std = (0.1307,), (0.3081,); tl = [transforms.ToPILImage(), transforms.Resize((28, 28))]
        if self.is_train: tl.extend([transforms.RandomRotation((-15, 15)), transforms.RandomAffine(0, translate=(0.1, 0.1), scale=(0.9, 1.1))]); tl.extend([transforms.ToTensor(), transforms.Normalize(mean, std)]); return transforms.Compose(tl)
    def __len__(self): return len(self.indices) if self.mode == 'subset' else len(self.X)
    def __getitem__(self, idx): img, lbl = self.base_dataset[self.indices[idx]] if self.mode == 'subset' else (self.X[idx], self.y[idx]); img_t = self.transform(img); lbl_t = torch.tensor(lbl, dtype=torch.long); return img_t, lbl_t

class CIFARDataset(BaseImageDataset):
    """Final CIFAR-10 dataset wrapper."""
    # (Keep class code as provided in the previous data_processing.py - init slightly adapted)
    def __init__(self, base_dataset=None, indices=None, split_type=None, dataset_config=None, scaler_obj=None, X=None, y=None, is_train=True, **kwargs):
        super().__init__(X=X, y=y, base_dataset=base_dataset, indices=indices, split_type=split_type, dataset_config=dataset_config, scaler_obj=scaler_obj, is_train=is_train, **kwargs)
    def get_transform(self): 
        mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616); tl = [transforms.ToPILImage()]
        if self.is_train: 
            tl.extend([transforms.RandomCrop(32, padding=4, padding_mode='reflect'), transforms.RandomHorizontalFlip()])
            tl.extend([transforms.ToTensor(), transforms.Normalize(mean, std)])
            return transforms.Compose(tl)
    def __len__(self): 
        return len(self.indices) if self.mode == 'subset' else len(self.X)
    def __getitem__(self, idx): 
        img, lbl = self.base_dataset[self.indices[idx]] if self.mode == 'subset' else (self.X[idx], self.y[idx]); img_t = self.transform(img); lbl_t = torch.tensor(lbl, dtype=torch.long); return img_t, lbl_t

class ISICDataset(BaseImageDataset):
    """Final ISIC dataset wrapper (expects paths)."""
    # (Keep class code as provided in the previous data_processing.py)
    def __init__(self, image_paths, labels, is_train, **kwargs): self.sz = 200; super().__init__(X=image_paths, y=labels, is_train=is_train, **kwargs) # Pass paths as X
    def get_transform(self): 
        mean, std = (0.585, 0.500, 0.486), (0.229, 0.224, 0.225)
        if self.is_train:
            return albumentations.Compose([albumentations.RandomScale(0.07), albumentations.Rotate(50), albumentations.RandomBrightnessContrast(0.15, 0.1), albumentations.Flip(p=0.5), albumentations.Affine(shear=0.1), albumentations.RandomCrop(self.sz, self.sz), albumentations.CoarseDropout(max_holes=random.randint(1, 8), max_height=16, max_width=16), albumentations.Normalize(mean=mean, std=std, always_apply=True)])
        else: 
            return albumentations.Compose([albumentations.CenterCrop(self.sz, self.sz), albumentations.Normalize(mean=mean, std=std, always_apply=True)])
    def __len__(self): 
        return len(self.X) # self.X has paths
    def __getitem__(self, idx): 
        img_p = self.X[idx]; lbl = self.y[idx]
        try: 
            img = np.array(Image.open(img_p).convert('RGB'))
        except FileNotFoundError: 
            print(f"Error: ISIC Image not found: {img_p}.")
            return None, torch.tensor(lbl, dtype=torch.long)
        transformed = self.transform(image=img); img = transformed['image']; img_t = torch.from_numpy(img.transpose(2, 0, 1)).float(); lbl_t = torch.tensor(lbl, dtype=torch.long)
        return img_t, lbl_t

class IXITinyDataset(BaseImageDataset):
    """Final IXITiny dataset wrapper (expects paths)."""
    # (Keep class code as provided in the previous data_processing.py)
    def __init__(self, image_paths, label_paths, is_train, **kwargs):
        if MonaiToTensor is None:
            raise ImportError("MONAI required for IXITinyDataset")
        self.nib = nib; self.monai_transforms = {'EnsureChannelFirst': EnsureChannelFirst, 'AsDiscrete': AsDiscrete, 'Compose': Compose, 'NormalizeIntensity': NormalizeIntensity, 'Resize': Resize, 'ToTensor': MonaiToTensor}; self.common_shape = (48, 60, 48); super().__init__(X=image_paths, y=label_paths, is_train=is_train, **kwargs) # Pass paths as X and y
    def get_transform(self): 
        Compose = self.monai_transforms['Compose']; ToTensor = self.monai_transforms['ToTensor']; EnsureChannelFirst = self.monai_transforms['EnsureChannelFirst']; Resize = self.monai_transforms['Resize']; NormalizeIntensity = self.monai_transforms['NormalizeIntensity']; self.image_transform = Compose([ToTensor(), EnsureChannelFirst(channel_dim="no_channel"), Resize(self.common_shape), NormalizeIntensity()]); self.label_transform = Compose([ToTensor(), EnsureChannelFirst(channel_dim="no_channel"), Resize(self.common_shape), self.monai_transforms['AsDiscrete'](to_onehot=2)]); return None
    def __len__(self): return len(self.X) # self.X has image paths
    def __getitem__(self, idx): 
        img_p = self.X[idx]; lbl_p = self.y[idx]
        try: 
            img = self.nib.load(img_p).get_fdata(dtype=np.float32)
            lbl = self.nib.load(lbl_p).get_fdata(dtype=np.uint8)
        except FileNotFoundError as e: 
            print(f"Error: Nifti file not found: {e}."); return None, None
        img_t = self.image_transform(img); lbl_t = self.label_transform(lbl); return img_t.float(), lbl_t.float()