"""
Configuration utilities for OT-based similarity analysis.
Defines OTConfig class and pre-defined configuration collections.
"""
import logging
from typing import Dict, Optional, Any, List

# Configure module logger
logger = logging.getLogger(__name__)
VERBOSE = True
class OTConfig:
    """
    Configuration object for a single OT calculation run.
    Stores method type, name, and parameters.
    """
    # Define known method types to prevent typos
    KNOWN_METHOD_TYPES = {'feature_error', 'direct_ot'}

    def __init__(self, method_type: str, name: str, params: Optional[Dict[str, Any]] = None):
        """
        Args:
            method_type (str): The type of OT calculator (e.g., 'feature_error').
            name (str): A unique descriptive name for this configuration set (e.g., 'FE_LossW_Norm').
            params (dict, optional): Dictionary of parameters specific to the OT method. Defaults to {}.
        """
        if not isinstance(method_type, str) or not method_type:
            raise ValueError("method_type must be a non-empty string.")
        if method_type not in self.KNOWN_METHOD_TYPES:
            logger.warning(f"Unknown method_type '{method_type}'. Ensure a corresponding calculator exists in the factory.")
        self.method_type = method_type

        if not isinstance(name, str) or not name:
            raise ValueError("name must be a non-empty string.")
        self.name = name

        self.params = params if params is not None else {}
        if not isinstance(self.params, dict):
            raise ValueError("params must be a dictionary.")

    def __repr__(self) -> str:
        return f"OTConfig(method_type='{self.method_type}', name='{self.name}', params={self.params})"


# ----- Predefined OT Configurations -----

# Feature Error OT Configurations
feature_error_configs = [
    # Basic configurations with different weighting strategies
    # OTConfig(
    #     method_type='feature_error',
    #     name='FE_Uniform_Norm',
    #     params={
    #         'use_loss_weighting': False,
    #         'normalize_cost': True,
    #         'min_samples': 20,
    #         'max_samples': 900,
    #         'alpha': 1.0,
    #         'beta': 1.0,
    #         'verbose':VERBOSE,
    #     }
    # ),
    # OTConfig(
    #     method_type='feature_error',
    #     name='FE_LossWeighted_Norm',
    #     params={
    #         'use_loss_weighting': True,
    #         'normalize_cost': True,
    #         'min_samples': 20,
    #         'max_samples': 900,
    #         'alpha': 1.0,
    #         'beta': 1.0,
    #         'verbose':VERBOSE,
    #     }
    # ),
    # # Feature-only variant (no error term)
    # OTConfig(
    #     method_type='feature_error',
    #     name='FE_FeatureOnly',
    #     params={
    #         'use_loss_weighting': False,
    #         'normalize_cost': True,
    #         'min_samples': 20,
    #         'max_samples': 900,
    #         'alpha': 1.0,
    #         'beta': 0.0,  # No error term
    #         'verbose':VERBOSE,
    #     }
    # ),
    # # Error-only variant (no feature term)
    # OTConfig(
    #     method_type='feature_error',
    #     name='FE_ErrorOnly',
    #     params={
    #         'use_loss_weighting': False,
    #         'normalize_cost': True,
    #         'min_samples': 20,
    #         'max_samples': 900,
    #         'alpha': 0.0,  # No feature term
    #         'beta': 1.0,
    #         'verbose':VERBOSE,
    #     }
    # ),
]


# Direct OT Configurations
direct_ot_configs = [
    # OTConfig(
    #     method_type='direct_ot',
    #     name='Direct_Euclidean',
    #     params={
    #         'use_loss_weighting': False,
    #         'normalize_activations': True,
    #         'normalize_cost': True,
    #         'distance_method': 'euclidean',
    #         'min_samples': 20, 
    #         'max_samples': 900,
    #         'use_label_hellinger': False,
    #         'verbose':VERBOSE,
    #     }
    # ),
    # OTConfig(
    #     method_type='direct_ot',
    #     name='Direct_Cosine',
    #     params={
    #         'use_loss_weighting': False,
    #         'normalize_activations': True,
    #         'normalize_cost': True,
    #         'distance_method': 'cosine',
    #         'min_samples': 20,
    #         'max_samples': 900,
    #         'use_label_hellinger': False,
    #         'verbose':VERBOSE,
    #     }
    # ),
    OTConfig(
        method_type='direct_ot',
        name='Direct_WithHellinger',
        params={
            'use_loss_weighting': False,
            'normalize_activations': True,
            'normalize_cost': True,
            'distance_method': 'cosine',
            'min_samples': 20,
            'max_samples': 900,
            'use_label_hellinger': True,
            'feature_weight': 3.0,
            'label_weight': 1.0,
            'verbose':VERBOSE,
        }
    ),
    # OTConfig(
    #     method_type='direct_ot',
    #     name='Direct_LossWeighted',
    #     params={
    #         'use_loss_weighting': True,
    #         'normalize_activations': True,
    #         'normalize_cost': True,
    #         'distance_method': 'euclidean',
    #         'min_samples': 20,
    #         'max_samples': 900,
    #         'use_label_hellinger': False,
    #         'verbose':VERBOSE,
    #     }
    # ),
]

# Combined configuration list for convenience
all_configs = feature_error_configs + direct_ot_configs