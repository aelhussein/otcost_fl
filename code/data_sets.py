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
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
from typing import Dict, Tuple, Any, Optional, List, Union, Callable

# Assume necessary libraries are installed
import nibabel as nib
from monai.transforms import (LoadImaged, Resized, NormalizeIntensityd,
                              AsDiscreted, ToTensord, Compose)
import albumentations
from albumentations.pytorch import ToTensorV2
from torchvision import transforms
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import math

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


class _BaseImgDS(TorchDataset):
    MEAN_STD   = None          # override
    TRAIN_AUG  = None          # list[transform]  – override
    RESIZE_TO  = None          # (H,W) – EMNIST only

    def __init__(self,
                 split_type        : str,
                 X_np              : np.ndarray = None,
                 y_np              : np.ndarray = None,
                 base_tv_dataset   = None,
                 indices           = None,
                 **trans_args):
        self.is_train  = split_type == "train"
        self.trans_args = trans_args
        if X_np is not None:                      # -------- NumPy path
            self.mode = "numpy"
            self.images = X_np
            self.labels = y_np.astype(np.int64)
        elif base_tv_dataset is not None:         # -------- torchvision path
            self.mode   = "tv"
            self.base   = base_tv_dataset
            self.indices = indices
        else:
            raise ValueError("Provide either (X_np,y_np) or (base_tv_dataset,indices)")

        self.transform = self._build_transform()

    # -----------------------------------------------------------
    def _build_transform(self):
        mean, std = self.MEAN_STD
        t = []

        # add ToPIL *only* for NumPy / tensor inputs
        if self.mode == "numpy":
            t.append(transforms.ToPILImage())

        z = self.trans_args.get('zoom', 0.0)      # e.g. +0.2 → 1.2×, −0.2 → 0.8×
        if abs(z) > 1e-3:
            scale = 1.0 + z
            t.append(transforms.Lambda(lambda img, s=scale:
                TF.affine(img, angle=0, translate=(0,0),
                                            scale=s, shear=(0,0), fill=0)))

        if self.RESIZE_TO is not None:
            t.append(transforms.Resize(self.RESIZE_TO))

        r = self.trans_args.get('angle', 0.0)
        if abs(r) > 1e-6:
            t.append(transforms.RandomAffine(
                     degrees=(r, r)))
            

        # --- frequency filter -------------------------------------------------
        f = self.trans_args.get('frequency', 0.0)
        if abs(f) > 1e-3:
            t.append(transforms.Lambda(lambda img, d=f: self.img_transform(img, d)))

        if self.is_train and self.TRAIN_AUG:
            t.extend(self.TRAIN_AUG)

        t.extend([transforms.ToTensor()],) #transforms.Normalize(mean=mean, std=std))
        return transforms.Compose(t)

    # -----------------------------------------------------------
    def __len__(self):
        return len(self.images) if self.mode == "numpy" else len(self.indices)

    def __getitem__(self, idx):
        if self.mode == "numpy":
            img, label = self.images[idx], int(self.labels[idx])
        else:  # torchvision
            base_idx = self.indices[idx]
            img, label = self.base[base_idx]

        img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long)


# ------------------------ concrete datasets --------------------
class EMNISTDataset(_BaseImgDS):
    MEAN_STD  = ((0.1307,), (0.3081,))
    TRAIN_AUG = [transforms.RandomRotation((-0, 0))]
    RESIZE_TO = (28, 28)
    def img_transform(self, img: Image.Image, delta: float) -> Image.Image:
        return self.freq_filter(img, delta)

    def freq_filter(self, img: Image.Image, delta: float) -> Image.Image:
    # delta>0 high-pass, delta<0 low-pass
        if abs(delta) < 1e-3:
            return img
        arr = np.array(img, np.float32)
        fft = np.fft.fftshift(np.fft.fft2(arr, axes=(0,1)))
        h, w = arr.shape[:2]; r = int(min(h,w)*0.1*abs(delta))
        y,x = np.ogrid[-h//2:h//2, -w//2:w//2]
        mask = x**2+y**2 <= r*r
        if delta>0:  fft[mask] *= 0.3          # high-pass
        else:        fft[~mask] *= 0.3         # low-pass
        filtered = np.abs(np.fft.ifft2(np.fft.ifftshift(fft)))
        return Image.fromarray(filtered.clip(0,255).astype(np.uint8))


class CIFARDataset(_BaseImgDS):
    MEAN_STD  = ((0.4914, 0.4822, 0.4465),
                 (0.2470, 0.2435, 0.2616))
    TRAIN_AUG = [transforms.RandomCrop(32, padding=4, padding_mode="reflect")]
    RESIZE_TO = None

    # -------------------------------------------------------------
    def freq_filter(self, img: Image.Image, delta: float) -> Image.Image:
        """
        Deterministic per-client frequency shift.
          • delta > 0  ⇒ high-pass  (sharpen edges)
          • delta < 0  ⇒ low-pass   (blur)
          • |delta|∈[0,1]
        Only the luminance (Y) channel is filtered; Cb/Cr stay intact,
        avoiding hue artefacts.
        """
        if abs(delta) < 1e-3:          # no-op for tiny costs
            return img

        # 1. RGB → YCbCr  and split
        try:
            y, cb, cr = img.convert("YCbCr").split()
        except Exception:              # fallback for unexpected modes
            return img
        BASE_RADIUS_FACTOR = 0.15
        ATTENUATION_FACTOR = 0.15
        y_arr = np.asarray(y, np.float32)
        H, W  = y_arr.shape
        # 2. FFT
        FFT = np.fft.fftshift(np.fft.fft2(y_arr))
        # radius proportional to |delta|
        r  = max(1, int(min(H, W) * BASE_RADIUS_FACTOR * abs(delta)))
        yy, xx = np.ogrid[-H//2:H//2, -W//2:W//2]
        mask   = xx*xx + yy*yy <= r*r

        # 3. Attenuate band (KEEP controls strength)
        if delta > 0:      # high-pass → kill low frequencies
            FFT[mask] *= ATTENUATION_FACTOR
        else:              # low-pass  → kill high frequencies
            FFT[~mask] *= ATTENUATION_FACTOR

        # 4. Back to spatial domain (use REAL part to keep phase!)
        y_filtered = np.real(np.fft.ifft2(np.fft.ifftshift(F)))
        y_img      = Image.fromarray(np.clip(y_filtered, 0, 255).astype(np.uint8))

        # 5. Merge and return RGB
        return Image.merge("YCbCr", (y_img, cb, cr)).convert("RGB")
    
    def elastic_transform(self, img: Image.Image, delta: float)-> Image.Image:
        """
        Apply elastic deformation to images.
        Creates realistic distortions that affect feature representations.
        """
        if abs(delta) < 1e-3:
            return img
            
        # Convert to tensor for easier processing
        to_tensor = transforms.ToTensor()
        to_pil = transforms.ToPILImage()
        
        img_tensor = to_tensor(img)
        c, h, w = img_tensor.shape
        
        # Generate displacement field
        displacement = abs(delta) * 1.0  # Scale factor
        grid_scale = 4  # Control grid coarseness
        
        # Create base grid
        theta = torch.tensor([
            [1, 0, 0],
            [0, 1, 0]
        ], dtype=torch.float).unsqueeze(0)
        
        # Create sampling grid
        base_grid = F.affine_grid(theta, (1, c, h, w), align_corners=False)
        # Generate random displacement field
        np.random.seed(int(10 * delta) + 42)  # Deterministic per delta
        disp_x = torch.FloatTensor(np.random.randn(h // grid_scale, w // grid_scale)) * displacement
        disp_y = torch.FloatTensor(np.random.randn(h // grid_scale, w // grid_scale)) * displacement
        
        # Upsample displacements to full resolution
        disp_x = F.interpolate(disp_x.unsqueeze(0).unsqueeze(0), size=(h, w), mode='bicubic').squeeze()
        disp_y = F.interpolate(disp_y.unsqueeze(0).unsqueeze(0), size=(h, w), mode='bicubic').squeeze()
        
        # Apply displacement to grid
        flow_field = base_grid.clone()
        flow_field[0, :, :, 0] += disp_x / w  # Normalize to [-1, 1]
        flow_field[0, :, :, 1] += disp_y / h  # Normalize to [-1, 1]
        
        # Sample using the displaced grid
        transformed = F.grid_sample(img_tensor.unsqueeze(0), flow_field, 
                                   align_corners=False, padding_mode='reflection').squeeze(0)
        
        # Convert back to PIL
        return to_pil(transformed)
    
    def img_transform(self, img: Image.Image, delta: float) -> Image.Image:
        if delta < 0:
            return self.color_jitter_filter(img, delta)
        else:
            return self.gaussian_blur_filter(img, delta)
    
    def color_jitter_filter(self, img: Image.Image, delta: float) -> Image.Image:
        """
        Apply a deterministic color adjustment based on delta.
        Aims to simulate parts of ColorJitter's behavior deterministically.
        - delta > 0: generally increases brightness, contrast, saturation; positive hue shift.
        - delta < 0: generally decreases brightness, contrast, saturation; negative hue shift.
        - |delta| should ideally be in [0,1] to control intensity, but scales are capped.
          A delta of 0 results in no change.
        """
        if abs(delta) < 1e-3: # no-op for tiny delta
            return img

        # Define maximum impact scales for each parameter when |delta|=1.
        # These can be tuned. For example, a 0.4 scale means at delta=1,
        # the factor becomes 1.4, and at delta=-1, it becomes 0.6.
        BRIGHTNESS_STRENGTH = 0.2
        CONTRAST_STRENGTH = 0.2
        SATURATION_STRENGTH = 0.2
        # Max hue shift. Typical range for hue in TF.adjust_hue is [-0.5, 0.5].
        # So, HUE_STRENGTH of 0.2 means delta=1 results in +0.2 hue shift.
        HUE_STRENGTH = 0.1

        # Calculate deterministic adjustment factors
        # brightness_factor: 1.0 means no change.
        brightness_factor = max(0.0, 1.0 + delta * BRIGHTNESS_STRENGTH)
        # contrast_factor: 1.0 means no change.
        contrast_factor = max(0.0, 1.0 + delta * CONTRAST_STRENGTH)
        # saturation_factor: 1.0 means no change.
        saturation_factor = max(0.0, 1.0 + delta * SATURATION_STRENGTH)
        # hue_factor: 0.0 means no change. Values should be in [-0.5, 0.5].
        hue_factor = torch.clamp(torch.tensor(delta * HUE_STRENGTH), -0.5, 0.5).item()

        # Apply transformations sequentially using torchvision.transforms.functional
        # These functions expect PIL Images and return PIL Images.
        img_transformed = TF.adjust_brightness(img, brightness_factor)
        img_transformed = TF.adjust_contrast(img_transformed, contrast_factor)
        img_transformed = TF.adjust_saturation(img_transformed, saturation_factor)
        if abs(hue_factor) > 1e-6: # Only apply hue if it's a meaningful shift
            img_transformed = TF.adjust_hue(img_transformed, hue_factor)
        
        return img_transformed

    def gaussian_blur_filter(self, img: Image.Image, delta: float) -> Image.Image:
        """
        Apply Gaussian blur to an image.
        - |delta| ∈ [0,1] (ideally) controls the intensity of the blur (sigma).
        - delta = 0 (or very small |delta|) results in no blur.
        - The sign of delta is ignored; only its magnitude determines blur strength.
        """
        if abs(delta) < 1e-3: # no-op for tiny delta
            return img

        # Define maximum sigma when |delta|=1. This can be tuned.
        # E.g., for 28x28 or 32x32 images, max sigma of 1.5-2.0 is reasonable.
        MAX_SIGMA_AT_UNITY_DELTA = 1.2

        # Calculate sigma based on delta's magnitude
        sigma = delta * MAX_SIGMA_AT_UNITY_DELTA
        
        # If sigma is extremely small, effectively no blur.
        # This also prevents issues with kernel_size calculation if sigma is near zero.
        if sigma < 1e-2: 
             return img

        # Determine kernel size. Must be odd and positive.
        # A common heuristic: kernel_size is roughly 2 to 3 times sigma on each side of the center.
        # Let radius = ceil(N * sigma). Kernel_size = 2 * radius + 1.
        # We use N=2.5 here for a good balance.
        radius = math.ceil(1 * sigma)
        kernel_s = 2 * radius + 1
        
        # TF.gaussian_blur takes kernel_size as [height, width] and sigma as [sigma_y, sigma_x] or float.
        # We'll use a square kernel and symmetric sigma.
        return TF.gaussian_blur(img, kernel_size=[kernel_s, kernel_s], sigma=[sigma, sigma])


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