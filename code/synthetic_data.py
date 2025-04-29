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
    Applies a feature distribution shift to the input data X with balanced direction.

    Args:
        X (np.ndarray): Input features (standard normal recommended).
        delta (float): Shift intensity parameter [0, 1].
        kind (str): Type of shift ('mean', 'scale', 'tilt', 'balanced_mean').
        cols (Optional[int]): Number of initial columns to affect. Defaults to min(10, n_features).
        mu (float): Mean shift factor (for kind='mean').
        sigma (float): Scale shift factor (for kind='scale').
        rng (Optional[np.random.Generator]): Random number generator for 'tilt'.

    Returns:
        np.ndarray: Modified feature array X.
    """
    n_samples, n_features = X.shape
    if cols is None:
        cols = min(10, n_features)  # Default to affect first 10 cols or fewer

    # mean shift - alternates shift direction for features
    if kind == "mean" and cols > 0:
        half_cols = 10
        
        # First half of features shift positive
        if half_cols > 0:
            X[:, :half_cols] += delta * mu
            
        # Second half of features shift negative
        if cols - half_cols > 0:
            X[:, half_cols:cols] -= delta * mu
            
    # Balanced scale shift - some scale up, some scale down
    elif kind == "scale" and cols > 0:
        half_cols = cols // 2
        
        # First half scales up
        if half_cols > 0:
            X[:, :half_cols] *= (1.0 + delta * sigma)
            
        # Second half scales down (carefully to avoid division by zero)
        if cols - half_cols > 0:
            scale_down = 1.0 / (1.0 + delta * sigma) if delta * sigma < 0.9 else 0.1
            X[:, half_cols:cols] *= scale_down
            
        
    # Tilt mode (already fairly balanced due to orthogonal transformation)
    elif kind == "tilt":
        if rng is None:
            rng = np.random.default_rng()  # Use default if none provided
        # Generate a random orthogonal matrix Q via QR decomposition
        H = rng.standard_normal((n_features, n_features))
        Q, _ = np.linalg.qr(H)
        # Apply convex combination: (1-delta)*X + delta*(X @ Q)
        X[:] = (1.0 - delta) * X + delta * (X @ Q)

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

def generate_synthetic_data(
    mode: str,
    n_samples: int,
    n_features: int,
    shift_param: float = 0.0,
    label_noise: float = 0.0,
    base_seed: int = 42,
    client_seed: int = 42,
    **shift_config: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic data with optional feature / concept shift and a selectable
    label rule (linear, quadratic, piecewise, mlp).

    Parameters
    ----------
    mode : {'baseline','feature_shift','concept_shift'}
    label_rule : str in shift_config, default 'linear'
        {'linear','quadratic','piecewise','mlp'}
    Other args unchanged.

    Returns
    -------
    X : (n_samples, n_features) float32
    y : (n_samples,) int64  (binary classes 0 / 1)
    """
    # ------------------------------------------------ RNG ------------------------------------------------
    base_rng   = np.random.default_rng(base_seed)
    client_rng = np.random.default_rng(client_seed)

    # ---------------------------------------------- base X ----------------------------------------------
    X = client_rng.standard_normal((n_samples, n_features))

    # -------------------------------------- optional feature shift --------------------------------------
    if mode == "feature_shift":
        keys = ['kind', 'cols', 'mu', 'sigma']
        f_args = {k: shift_config[k] for k in keys if k in shift_config}
        X = _apply_feature_shift(X, delta=shift_param, rng=client_rng, **f_args)

    # -------------------------------------------- label score -------------------------------------------
    label_rule = shift_config.get("label_rule", "linear")
    w          = base_rng.standard_normal(n_features)

    if label_rule == "linear":
        score = X @ w

    elif label_rule == "quadratic":
        k     = min(5, n_features // 2)
        beta  = 0.2
        score = X @ w + beta * (X[:, :k] * X[:, k:2*k]).sum(1)

    elif label_rule == "piecewise":
        raw   = X @ w
        bins  = np.digitize(raw, [-1., 0., 1.])            # 0-1-2
        score = bins.astype(float)                         # piece-wise encoded as 0/1/2

    elif label_rule == "mlp":
        h  = 50
        W1 = base_rng.standard_normal((n_features, h))
        b1 = base_rng.standard_normal(h)
        W2 = base_rng.standard_normal(h)
        b2 = base_rng.standard_normal()
        hidden = np.tanh(X @ W1 + b1)
        score  = hidden @ W2 + b2

    else:
        raise ValueError(f"Unknown label_rule '{label_rule}'")

    # -------------------------------------- map score → label y -----------------------------------------
    if mode == "concept_shift":
        option  = shift_config.get("option", "threshold")
        if option == "rotation":      # keeps original behaviour
            y = _make_concept_labels(
                    X, gamma=shift_param, w=w,
                    option="rotation",
                    threshold_range_factor=shift_config.get("threshold_range_factor", 0.4),
                )
        else:                         # THRESHOLD shift on *same* score function
            gamma = shift_param
            factor = shift_config.get("threshold_range_factor", 0.4)
            target_q = np.clip(0.5 + factor * (gamma - 0.5), 1e-6, 1 - 1e-6)
            thresh   = np.quantile(score, target_q)
            y = (score > thresh).astype(int)
    else:
        # baseline / feature_shift: fixed threshold 0
        y = (score > 0).astype(int)

    # ------------------------------------------- label noise --------------------------------------------
    if label_noise > 0:
        flip = client_rng.random(n_samples) < label_noise
        y[flip] = 1 - y[flip]

    return X.astype(np.float32), y.astype(np.int64)
