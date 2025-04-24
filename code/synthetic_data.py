# synthetic_data.py
"""
Generates raw synthetic data (features and labels) as NumPy arrays.
Includes simple, controllable feature and concept shift mechanisms.
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional

# =============================================================================
# == Helper Functions for Shifts ==
# =============================================================================

def _apply_feature_shift(X: np.ndarray,
                         delta: float,
                         kind: str = "mean",
                         cols: Optional[int] = None,
                         mu: float = 3.0,
                         sigma: float = 1.5,
                         rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    Applies a feature distribution shift to the input data X.

    Args:
        X (np.ndarray): Input features (standard normal recommended).
        delta (float): Shift intensity parameter [0, 1].
        kind (str): Type of shift ('mean', 'scale', 'tilt').
        cols (Optional[int]): Number of initial columns to affect. Defaults to min(5, n_features).
        mu (float): Mean shift factor (for kind='mean').
        sigma (float): Scale shift factor (for kind='scale').
        rng (Optional[np.random.Generator]): Random number generator for 'tilt'.

    Returns:
        np.ndarray: Modified feature array X.
    """
    n_samples, n_features = X.shape
    if cols is None:
        cols = min(10, n_features) # Default to affect first 10 cols or fewer

    if kind == "mean" and cols > 0:
        X[:, :cols] += delta * mu
    elif kind == "scale" and cols > 0:
        X[:, :cols] *= (1.0 + delta * sigma) # Ensure scaling factor is >= 1
    elif kind == "tilt":
        if rng is None:
            rng = np.random.default_rng() # Use default if none provided
        # Generate a random orthogonal matrix Q via QR decomposition
        H = rng.standard_normal((n_features, n_features))
        Q, _ = np.linalg.qr(H)
        # Apply convex combination: (1-delta)*X + delta*(X @ Q)
        X[:] = (1.0 - delta) * X + delta * (X @ Q)
    # else: kind not recognized or cols=0, no change

    return X

def _make_concept_labels(X: np.ndarray,
                         gamma: float,
                         w: np.ndarray,
                         option: str = "threshold",
                         threshold_range_factor: float = 0.4) -> np.ndarray:
    """
    Generates labels based on concept shift parameter gamma.

    Args:
        X (np.ndarray): Input features.
        gamma (float): Concept shift intensity parameter [0, 1].
        w (np.ndarray): Base weight vector for linear score.
        option (str): Type of concept shift ('threshold' or 'rotation').
        threshold_range_factor (float): Controls range of threshold shift
                                         (e.g., 0.4 means median +/- 0.4 quantile).

    Returns:
        np.ndarray: Binary labels y.
    """
    n_samples, n_features = X.shape

    if option == "threshold":
        score = X @ w
        # Calculate quantile for threshold: median is 0.5.
        # Shift quantile linearly based on gamma around 0.5
        # gamma=0 -> 0.5 - factor; gamma=0.5 -> 0.5; gamma=1 -> 0.5 + factor
        target_quantile = 0.5 + threshold_range_factor * (gamma - 0.5)
        # Ensure quantile is within valid range [epsilon, 1-epsilon] for stability
        epsilon = 1e-6
        target_quantile = np.clip(target_quantile, epsilon, 1.0 - epsilon)
        threshold = np.quantile(score, target_quantile)
        y = (score > threshold).astype(int)

    elif option == "rotation":
        if n_features < 2:
             print("Warning: Rotation concept shift requires at least 2 features. Using baseline labels.")
             y = (X @ w > 0).astype(int) # Fallback
        else:
            # Rotate weight vector w in the first two dimensions
            # angle = (gamma - 0.5) * np.pi / 2 # Map gamma [0,1] to angle [-pi/4, +pi/4] or [-45deg, +45deg]
            # angle = (gamma - 0.5) * np.pi  # Map gamma [0,1] to angle [-pi/2, +pi/2] or [-90deg, +90deg]
            angle = (gamma - 0.5) * np.pi * 0.8 # Slightly reduced range maybe [-72deg, +72 deg]

            c, s = np.cos(angle), np.sin(angle)
            w_rot = w.copy()
            # Apply 2D rotation to first two components
            w0_orig = w[0]
            w1_orig = w[1]
            w_rot[0] = c * w0_orig - s * w1_orig
            w_rot[1] = s * w0_orig + c * w1_orig
            # Generate labels using rotated weight vector and zero threshold
            y = (X @ w_rot > 0).astype(int)
    else:
         print(f"Warning: Unknown concept label option '{option}'. Using baseline labels.")
         y = (X @ w > 0).astype(int) # Default baseline

    return y

# =============================================================================
# == Main Public Generation Function ==
# =============================================================================

def generate_synthetic_data(mode: str,
                            n_samples: int,
                            n_features: int,
                            shift_param: float = 0.0,
                            label_noise: float = 0.0,
                            seed: int = 42,
                            **shift_config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates synthetic (features, labels) based on mode and shift parameter.

    Args:
        mode (str): Generation mode ('baseline', 'feature_shift', 'concept_shift').
        n_samples (int): Number of samples.
        n_features (int): Number of features.
        shift_param (float): Intensity of the shift [0, 1] (delta or gamma).
        label_noise (float): Probability of flipping a label [0, 1].
        seed (int): Random seed.
        **shift_config (Dict[str, Any]): Additional configuration for the specific
                                         shift type (passed to helper functions).
                                         E.g., feature_shift_kind='mean',
                                                feature_shift_cols=5,
                                                concept_label_option='threshold'.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Generated features X and labels y.
    """
    rng = np.random.default_rng(seed)

    # 1. Generate Base Features (Standard Normal)
    X = rng.standard_normal((n_samples, n_features))

    # 2. Generate Base Weight Vector (fixed for this seed)
    w = rng.standard_normal(n_features)

    # 3. Apply Feature Shift (if applicable)
    if mode == "feature_shift":
        # Extract relevant keys for apply_feature_shift, avoid passing unrelated keys
        feature_shift_keys = ['kind', 'cols', 'mu', 'sigma']
        feature_shift_args = {k: shift_config[k] for k in feature_shift_keys if k in shift_config}
        X = _apply_feature_shift(X, delta=shift_param, rng=rng, **feature_shift_args)

    # 4. Generate Labels
    if mode == "concept_shift":
        # Extract relevant keys for _make_concept_labels
        concept_label_keys = ['option', 'threshold_range_factor']
        concept_label_args = {k: shift_config[k] for k in concept_label_keys if k in shift_config}
        y = _make_concept_labels(X, gamma=shift_param, w=w, **concept_label_args)
    else: # baseline or feature_shift mode uses baseline labels
        y = (X @ w > 0).astype(int) # Simple linear threshold at 0

    # 5. Apply Label Noise
    if label_noise > 0:
        flip_mask = rng.random(n_samples) < label_noise
        # Use XOR (^) to flip binary labels (0->1, 1->0)
        y[flip_mask] = 1 - y[flip_mask] # Alternative: y[flip_mask] ^= 1

    return X, y