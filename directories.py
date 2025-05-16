"""
Centralized directory configuration for the project.
Provides path configuration that can be adjusted based on the metric (score/loss).
"""
import sys
import os
import logging
from dataclasses import dataclass
from typing import Dict, Optional

# Core directory paths - these remain constant
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
_CODE_DIR = os.path.join(_PROJECT_ROOT, "code")  # Path to code/ directory
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, _CODE_DIR)  # Add code/ directory specifically to import path

ROOT_DIR = _PROJECT_ROOT  # Project root directory
DATA_DIR = os.path.join(ROOT_DIR, 'data')

# Internal state tracking
_configured = False

@dataclass
class DirPaths:
    """Container for directory paths that can vary based on the metric."""
    root_dir: str
    data_dir: str
    results_dir: str
    model_save_dir: str
    activation_dir: str
    selection_criterion_key: str


def configure(metric: str = "loss") -> None:
    """
    Build all directories and set the global constants exactly *once*.

    Parameters
    ----------
    metric : {"score", "loss"}
        Determines which validation key we optimise and (optionally) a
        suffix for experiment folders.
    """
    global ROOT_DIR, DATA_DIR, RESULTS_DIR, MODEL_SAVE_DIR, ACTIVATION_DIR, SELECTION_CRITERION_KEY
    global _configured, _metric

    if _configured:
        logging.warning("directories.configure() called twice – ignored.")
        return

    _metric = metric
    ROOT_DIR = _PROJECT_ROOT                     # project root, not code/ folder
    suffix = "" if metric == "loss" else "_score"

    RESULTS_DIR      = os.path.join(ROOT_DIR, f"results{suffix}")
    MODEL_SAVE_DIR   = os.path.join(ROOT_DIR, "saved_models")      # keep flat
    ACTIVATION_DIR   = os.path.join(ROOT_DIR, "activations")       # keep flat
    SELECTION_CRITERION_KEY = "val_scores" if metric == "score" else "val_losses"

    for d in (RESULTS_DIR, MODEL_SAVE_DIR, ACTIVATION_DIR):
        os.makedirs(d, exist_ok=True)

    _configured = True



def paths() -> DirPaths:
    """
    Returns the currently configured directory paths.
    
    Returns:
        DirPaths object containing the current directory paths
    """
    if not _configured:
        configure()  # Use defaults if not explicitly configured
        
    return DirPaths(
        root_dir=ROOT_DIR,
        data_dir=DATA_DIR,
        results_dir=RESULTS_DIR,
        model_save_dir=MODEL_SAVE_DIR,
        activation_dir=ACTIVATION_DIR,
        selection_criterion_key=SELECTION_CRITERION_KEY
    )