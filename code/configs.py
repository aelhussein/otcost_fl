"""
Central configuration file for the OTCost project.

Defines directory paths, constants, default hyperparameters,
algorithm-specific settings (like layers to federate, regularization parameters),
and imports common libraries used throughout the project.
"""
"""
Central configuration file for the project.

Defines directory paths, constants, default hyperparameters,
data handling configurations, algorithm settings, and imports common libraries.
"""
import warnings
# --- Suppress Warnings ---
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import os
import gc
import sys
import copy
import time
import json
import random
import logging
import pickle
import traceback
import argparse
import glob # Added for path finding
from typing import List, Dict, Optional, Tuple, Union, Iterator, Iterable, Any
from datetime import datetime
from functools import wraps, partial
from collections import OrderedDict
from dataclasses import dataclass, field
from itertools import combinations
from concurrent.futures import ProcessPoolExecutor, as_completed, wait
from multiprocessing import Pool

import numpy as np
import pandas as pd
import scipy.stats
import scipy.stats as stats
from scipy.stats import wasserstein_distance
from sklearn.cluster import KMeans
from tqdm import tqdm  # Progress bars
from tqdm.contrib.concurrent import process_map # Progress bars for parallel processing

from PIL import Image
from sklearn import metrics  # General metrics utilities
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, matthews_corrcoef, balanced_accuracy_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.linalg import svdvals
from torch.utils.data import DataLoader, Dataset, Subset, RandomSampler, Dataset as TorchDataset
from torch.nn.utils.rnn import pad_sequence
import torch.multiprocessing as mp
from torchvision.datasets import MNIST, CIFAR10, EMNIST

import nibabel as nib
from torchvision import transforms
import albumentations
from unet import UNet
from torchvision.models import resnet18
from monai.transforms import EnsureChannelFirst, AsDiscrete,Compose,NormalizeIntensity,Resize,ToTensor
from transformers import BertModel, BertTokenizer, AutoTokenizer, AutoModel # Keep if using NLP models

import ot

# --- Core Directories ---
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT  = os.path.dirname(_CURRENT_DIR)
ROOT_DIR = _PROJECT_ROOT
DATA_DIR = f'{ROOT_DIR}/data'       # Directory containing datasets
EVAL_DIR = f'{ROOT_DIR}/code/evaluation' # Directory for evaluation scripts/results
OTCOST_DIR = f'{ROOT_DIR}/code/OTCost' # Directory for evaluation scripts/results
RESULTS_DIR = f'{ROOT_DIR}/results' # Directory to save experiment results
ACTIVATION_DIR = f'{ROOT_DIR}/activations'
MODEL_SAVE_DIR = f'{ROOT_DIR}/save_models' # Directory to save models

# --- Add project directories to Python path ---
# Allows importing modules from these directories
sys.path.append(f'{ROOT_DIR}/code')
sys.path.append(f'{EVAL_DIR}')
sys.path.append(f'{OTCOST_DIR}')
sys.path.append(f'{ACTIVATION_DIR}')
sys.path.append(f'{MODEL_SAVE_DIR}')

# --- Global Settings ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu' # Set computation device
N_WORKERS = 1 # Default number of workers for DataLoader


# --- Supported Algorithms ---
ALGORITHMS = [
    'local',         # Local training only
    'fedavg',        # Federated Averaging
    # 'fedprox',     # Fedprox (personalized FedAvg with proximal term)
    # 'ditto',       # Ditto (dual model training: global and personal)
]

# --- Supported Datasets (names must match keys in DEFAULT_PARAMS) ---
DATASETS = [
    'Synthetic', # Synthetic dataset
    'Credit',    # Credit card fraud
    'EMNIST',    # EMNIST Digits
    'CIFAR',     # CIFAR-10
    'ISIC',      # Skin dermoscopy
    'IXITiny',   # Brain MRI
    'Heart'      # Heart disease prediction
]


# --- Default Hyperparameters & Data Handling Configuration ---

DEFAULT_PARAMS = {
    'CIFAR': {
        'data_source': 'torchvision', 'partitioning_strategy': 'dirichlet_indices', 'cost_interpretation': 'alpha',
        'dataset_class': 'CIFARDataset', 'default_num_clients': 10, 'max_clients': None, 'fixed_classes': 10,
        'source_args': {'dataset_name': 'CIFAR10', 'data_dir': DATA_DIR}, # Use CIFAR10 name
        'partitioner_args': {}, 'partition_scope': 'train',
        'sampling_config': {'type': 'fixed_total', 'size': 3000, 'replace': False},
        'learning_rates_try': [5e-3, 1e-3, 5e-4], 'default_lr': 1e-3,
        'reg_params_try':[1, 1e-1, 1e-2, 1e-3], 'default_reg_param': 1e-1,
        'batch_size': 128, 'epochs_per_round': 1, 'rounds': 100,
        'runs': 10, 'runs_tune': 3, 'metric': 'Accuracy', 'base_seed': 42
    },
    'EMNIST': {
        'data_source': 'torchvision', 'partitioning_strategy': 'dirichlet_indices', 'cost_interpretation': 'alpha',
        'dataset_class': 'EMNISTDataset', 'default_num_clients': 10, 'max_clients': None, 'fixed_classes': 10,
        'source_args': {'dataset_name': 'EMNIST', 'data_dir': DATA_DIR, 'split': 'digits'},
        'partitioner_args': {}, 'partition_scope': 'train',
        'sampling_config': {'type': 'fixed_total', 'size': 1000, 'replace': False},
        'learning_rates_try': [5e-3, 1e-3, 5e-4, 1e-4], 'default_lr': 1e-3,
        'reg_params_try':[1, 1e-1, 1e-2, 1e-3], 'default_reg_param': 1e-1,
        'batch_size': 128, 'epochs_per_round': 1, 'rounds': 75,
        'runs': 10, 'runs_tune': 3, 'metric': 'Accuracy', 'base_seed': 42
    },
    'Synthetic': {
        'data_source': 'synthetic_base', 'partitioning_strategy': 'dirichlet_indices', 'cost_interpretation': 'alpha',
        'dataset_class': 'SyntheticDataset', 'default_num_clients': 10, 'max_clients': None, 'fixed_classes': 2,
        'source_args': {'base_n_samples': 20000, 'n_features': 10, 'dist_type1': 'normal', 'dist_type2': 'skewed', 'label_noise': 0.05, 'random_state': 42 },
        'partitioner_args': {}, 'partition_scope': 'all', 'needs_preprocessing': ['standard_scale'], # Needs scaling fitted per client train split
        'sampling_config': {'type': 'fixed_total', 'size': 1000, 'replace': False},
        'learning_rates_try':[5e-1, 1e-1, 5e-2, 1e-2, 5e-3, 1e-3], 'default_lr': 1e-3,
        'reg_params_try':[1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5], 'default_reg_param': 1e-1,
        'batch_size': 64, 'epochs_per_round': 1, 'rounds': 100,
        'runs': 10, 'runs_tune': 3, 'metric': 'F1', 'base_seed': 42
    },
    'Credit': {
        'data_source': 'credit_base', 'partitioning_strategy': 'dirichlet_indices', 'cost_interpretation': 'alpha',
        'dataset_class': 'CreditDataset', 'default_num_clients': 10, 'max_clients': None, 'fixed_classes': 2,
        'source_args': {'csv_path': os.path.join(DATA_DIR, 'Credit/creditcard.csv'), 'drop_cols': ['Time', 'Amount']},
        'partitioner_args': {}, 'partition_scope': 'all', 'needs_preprocessing': ['standard_scale'], # Needs scaling fitted per client train split
        'sampling_config': {'type': 'fixed_total', 'size': 1000, 'replace': False},
        'learning_rates_try':[5e-1, 1e-1, 5e-2, 1e-2, 5e-3, 1e-3], 'default_lr':1e-3,
        'reg_params_try':[1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5], 'default_reg_param': 1e-1,
        'batch_size': 64, 'epochs_per_round': 1, 'rounds': 100,
        'runs': 10, 'runs_tune': 3, 'metric': 'F1', 'base_seed': 42
    },
    'Heart': {
        'data_source': 'heart_site_loader', # <--- CHANGED: Use new site loader
        'partitioning_strategy': 'pre_split', # <--- CHANGED: Assignment via mapping
        'cost_interpretation': 'site_mapping_key', # <--- CHANGED: Cost is map key
        'dataset_class': 'HeartDataset', # Final wrapper
        'default_num_clients': 2, # Designed for pairs based on original script
        'max_clients': 2, # Limited by pairwise site_mappings
        'fixed_classes': 2,
        'source_args': { # <--- CHANGED: Include all info needed by loader
            'data_dir': os.path.join(DATA_DIR, 'Heart'),
            'sites': ['cleveland', 'hungarian', 'switzerland', 'va'],
             'used_columns': ['age', 'sex', 'chest_pain_type', 'resting_bp', 'cholesterol',
                              'sugar', 'ecg', 'max_hr', 'exercise_angina', 'exercise_ST_depression', 'target'],
            'feature_names': ['age', 'sex', 'chest_pain_type', 'resting_bp', 'cholesterol',
                              'sugar', 'ecg', 'max_hr', 'exercise_angina', 'exercise_ST_depression'],
            'cols_to_scale': ['age', 'chest_pain_type', 'resting_bp', 'cholesterol',
                              'ecg', 'max_hr', 'exercise_ST_depression'],
            'scale_values': { # Pre-defined global scaling values
                'age': (53.0872973, 7.01459463e+01), 'chest_pain_type': (3.23702703, 8.17756772e-01),
                'resting_bp': (132.74405405, 3.45493057e+02), 'cholesterol': (220.23648649, 4.88430934e+03),
                'ecg': (0.64513514, 5.92069868e-01), 'max_hr': (138.75459459, 5.29172208e+02),
                'exercise_ST_depression': (0.89532432, 1.11317517e+00) },
            'site_mappings': { # Map cost keys (1-6) to site pairs for 2 clients
                1: [['cleveland'], ['hungarian']],    # Pair 1
                2: [['cleveland'], ['switzerland']], # Pair 2
                3: [['cleveland'], ['va']],          # Pair 3
                4: [['hungarian'], ['switzerland']], # Pair 4
                5: [['hungarian'], ['va']],          # Pair 5
                6: [['switzerland'], ['va']],        # Pair 6
                # Could add 'all' mapping if needed: 'all': [['cleveland'], ['hungarian'], ['switzerland'], ['va']] (would need default_num_clients=4)
            }
        },
        'partitioner_args': {},
        # 'needs_preprocessing': [], # Scaling handled internally by loader
        'sampling_config': None, # No sampling within sites
        'learning_rates_try':[5e-1, 1e-1, 5e-2, 1e-2, 5e-3, 1e-3], 'default_lr':1e-3,
        'reg_params_try':[1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5], 'default_reg_param': 1e-1,
        'batch_size': 64, 'epochs_per_round': 1, 'rounds': 100,
        'runs': 10, 'runs_tune': 3, 'metric': 'F1', 'base_seed': 42
    },
    'ISIC': { # Keep as is
        'data_source': 'pre_split_paths_isic', 'partitioning_strategy': 'pre_split', 'cost_interpretation': 'site_mapping_key',
        'dataset_class': 'ISICDataset', 'default_num_clients': 2, 'max_clients': 4, 'fixed_classes': 8,
        'source_args': { 'data_dir': os.path.join(DATA_DIR, 'ISIC'),
            'site_mappings': { 0.06: [[2], [2]], 0.15: [[2], [0]], 0.19: [[2], [3]], 0.25: [[2], [1]], 0.3: [[1], [3]], 'all': [[0], [1], [2], [3]] }}, # Wrap site index in list
        'partitioner_args': {}, 'sampling_config': {'type': 'fixed_total', 'size': 2000},
        'learning_rates_try': [5e-3, 1e-3, 5e-4, 1e-4], 'default_lr': 1e-3,
        'reg_params_try':[1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5], 'default_reg_param': 1e-1,
        'batch_size': 128, 'epochs_per_round': 1, 'rounds': 60,
        'runs': 5, 'runs_tune': 1, 'metric': 'Balanced_accuracy', 'base_seed': 42
    },
    'IXITiny': { # Keep as is
        'data_source': 'pre_split_paths_ixi', 'partitioning_strategy': 'pre_split', 'cost_interpretation': 'site_mapping_key',
        'dataset_class': 'IXITinyDataset', 'default_num_clients': 2, 'max_clients': 3, 'fixed_classes': 2,
        'source_args': { 'data_dir': os.path.join(DATA_DIR, 'IXITiny'),
            'site_mappings': { 0.08: [['Guys'], ['HH']], 0.28: [['IOP'], ['Guys']], 0.30: [['IOP'], ['HH']], 'all': [['IOP'], ['HH'], ['Guys']] }},
        'partitioner_args': {}, 'sampling_config': None,
        'learning_rates_try': [1e-2, 5e-3, 1e-3, 5e-4], 'default_lr': 1e-3,
        'reg_params_try':[1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5], 'default_reg_param': 1e-1,
        'batch_size': 64, 'epochs_per_round': 1, 'rounds': 50,
        'runs': 10, 'runs_tune': 3, 'metric': 'DICE', 'base_seed': 42
    }
}

# Define the set of 'cost' or heterogeneity parameters to iterate over for each dataset.
DATASET_COSTS = {
    'IXITiny': [0.08, 0.28, 0.30, 'all'], # Site mapping keys
    'ISIC': [0.06, 0.15, 0.19, 0.25, 0.3, 'all'], # Site mapping keys
    'EMNIST': [0.1, 0.5, 1.0, 5.0, 10.0, 1000.0],  # Alpha values
    'CIFAR': [0.1, 0.5, 1.0, 5.0, 10.0, 1000.0],   # Alpha values
    'Synthetic': [0.1, 0.3, 0.5, 1.0, 5.0, 10.0, 1000.0], # Alpha values
    'Credit': [0.1, 0.3, 0.5, 1.0, 5.0, 10.0, 1000.0],    # Alpha values
    'Heart': [1, 2, 3, 4, 5, 6] # Site mapping keys (pair indices)
}

# --- Lists for quick checks (dynamically derived) ---
TABULAR_DATASET_CLASSES = ['SyntheticDataset', 'CreditDataset', 'HeartDataset'] # Removed Weather
TABULAR = [k for k, v in DEFAULT_PARAMS.items() if v.get('dataset_class') in TABULAR_DATASET_CLASSES]

SQUEEZE = [k for k, v in DEFAULT_PARAMS.items() if v.get('fixed_classes') is not None and v.get('fixed_classes') > 1 and v.get('metric') != 'R2' and k != 'IXITiny']
LONG = [k for k, v in DEFAULT_PARAMS.items() if v.get('fixed_classes') is not None and v.get('fixed_classes') > 1 and v.get('metric') != 'R2' and k != 'IXITiny']
TENSOR = ['IXITiny']