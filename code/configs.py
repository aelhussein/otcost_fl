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
from typing import List, Dict, Optional, Tuple, Union, Iterator, Iterable
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


# --- Core Directories ---
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT  = os.path.dirname(_CURRENT_DIR)
ROOT_DIR = _PROJECT_ROOT
DATA_DIR = f'{ROOT_DIR}/data'       # Directory containing datasets
EVAL_DIR = f'{ROOT_DIR}/code/evaluation' # Directory for evaluation scripts/results
OTCOST_DIR = f'{ROOT_DIR}/code/OTCost' # Directory for evaluation scripts/results
RESULTS_DIR = f'{ROOT_DIR}/results' # Directory to save experiment results
ACTIVATION_DIR = f'{ROOT_DIR}/activations'

# --- Add project directories to Python path ---
# Allows importing modules from these directories
sys.path.append(f'{ROOT_DIR}/code')
sys.path.append(f'{EVAL_DIR}')
sys.path.append(f'{OTCOST_DIR}')
sys.path.append(f'{ACTIVATION_DIR}')

# --- Global Settings ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu' # Set computation device
N_WORKERS = 0 # Default number of workers for DataLoader


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
    'Weather',   # Weather prediction
    'Heart'      # Heart disease prediction
]


# --- Default Hyperparameters & Data Handling Configuration ---

DEFAULT_PARAMS = {
    'CIFAR': {
        # Data Handling
        'data_source': 'torchvision',
        'partitioning_strategy': 'dirichlet_indices', # Uses the size-balancing function
        'cost_interpretation': 'alpha',
        'dataset_class': 'CIFARDataset',
        'default_num_clients': 2, # Default if -nc not specified
        'max_clients': None,     # No theoretical limit
        'fixed_classes': 10,
        'source_args': {'dataset_name': 'CIFAR10', 'data_dir': os.path.join(DATA_DIR)}, # Simplified path
        'partitioner_args': {},
        'partition_scope': 'train',
        'sampling_config': {'type': 'fixed_total', 'size': 1000, 'replace': False}, # Target size per client
        # Standard Params ...
        'learning_rates_try': [5e-3, 1e-3, 5e-4], 'default_lr': 1e-3,
        'reg_params_try':[1, 1e-1, 1e-2, 1e-3], 'default_reg_param': 1e-1,
        'batch_size': 128, 'epochs_per_round': 1, 'rounds': 100,
        'runs': 10, 'runs_tune': 3, 'metric': 'Accuracy'
    },
    'EMNIST': {
        # Data Handling
        'data_source': 'torchvision',
        'partitioning_strategy': 'dirichlet_indices', # Uses the size-balancing function
        'cost_interpretation': 'alpha',
        'dataset_class': 'EMNISTDataset',
        'default_num_clients': 2,
        'max_clients': None,
        'fixed_classes': 10,
        'source_args': {'dataset_name': 'EMNIST', 'data_dir': os.path.join(DATA_DIR), 'split': 'digits'},
        'partitioner_args': {},
        'partition_scope': 'train',
        'sampling_config': {'type': 'fixed_total', 'size': 1000, 'replace': False}, # Target size per client
        # Standard Params ...
        'learning_rates_try': [5e-3, 1e-3, 5e-4, 1e-4], 'default_lr': 1e-3,
        'reg_params_try':[1, 1e-1, 1e-2, 1e-3], 'default_reg_param': 1e-1,
        'batch_size': 128, 'epochs_per_round': 1, 'rounds': 75,
        'runs': 10, 'runs_tune': 3, 'metric': 'Accuracy'
    },
    'Synthetic': {
        # Data Handling
        'data_source': 'pre_split_csv',
        'partitioning_strategy': 'pre_split',
        'cost_interpretation': 'file_suffix',
        'dataset_class': 'SyntheticDataset',
        'default_num_clients': 2,
        'max_clients': 5, # Assumes only files for 5 clients exist
        'fixed_classes': 2,
        'source_args': {'data_dir': os.path.join(DATA_DIR, 'Synthetic'), 'column_count': 11},
        'partitioner_args': {},
        'needs_preprocessing': ['standard_scale'],
        'sampling_config': {'type': 'fixed_total', 'size': 250, 'replace': False},
        # Standard Params ...
        'learning_rates_try':[5e-1, 1e-1, 5e-2, 1e-2, 5e-3, 1e-3], 'default_lr': 1e-3,
        'reg_params_try':[1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5], 'default_reg_param': 1e-1,
        'batch_size': 64, 'epochs_per_round': 1, 'rounds': 100,
        'runs': 20, 'runs_tune': 10, 'metric': 'F1'
    },
    'Credit': {
        # Data Handling
        'data_source': 'pre_split_csv',
        'partitioning_strategy': 'pre_split',
        'cost_interpretation': 'file_suffix',
        'dataset_class': 'CreditDataset',
        'default_num_clients': 2,
        'max_clients': 5, # Adjust if more client files exist
        'fixed_classes': 2,
        'source_args': {'data_dir': os.path.join(DATA_DIR, 'Credit'), 'column_count': 29},
        'partitioner_args': {},
        'needs_preprocessing': ['standard_scale'],
        'sampling_config': {'type': 'fixed_total', 'size': 200, 'replace': True},
        # Standard Params ...
        'learning_rates_try':[5e-1, 1e-1, 5e-2, 1e-2, 5e-3, 1e-3], 'default_lr':1e-3,
        'reg_params_try':[1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5], 'default_reg_param': 1e-1,
        'batch_size': 64, 'epochs_per_round': 1, 'rounds': 100,
        'runs': 20, 'runs_tune': 10, 'metric': 'F1'
    },
     'Weather': {
        # Data Handling
        'data_source': 'pre_split_csv',
        'partitioning_strategy': 'pre_split',
        'cost_interpretation': 'file_suffix',
        'dataset_class': 'WeatherDataset',
        'default_num_clients': 2,
        'max_clients': 4, # Adjust if needed
        'fixed_classes': None, # Regression task
        'source_args': {'data_dir': os.path.join(DATA_DIR, 'Weather'), 'column_count': 124},
        'partitioner_args': {},
        'needs_preprocessing': ['standard_scale'],
        'sampling_config': {'type': 'fixed_total', 'size': 500, 'replace': False},
         # Standard Params ...
        'learning_rates_try': [5e-2, 1e-2, 5e-3, 1e-3, 5e-4], 'default_lr': 5e-3,
        'reg_params_try':[1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5], 'default_reg_param': 1e-1,
        'batch_size': 64, 'epochs_per_round': 1, 'rounds': 25,
        'runs': 10, 'runs_tune': 3, 'metric': 'R2'
    },
     'Heart': {
        # Data Handling
        'data_source': 'pre_split_csv',
        'partitioning_strategy': 'pre_split',
        'cost_interpretation': 'file_suffix',
        'dataset_class': 'HeartDataset',
        'default_num_clients': 2,
        'max_clients': 6, # Max based on cost values
        'fixed_classes': 2,
        'source_args': {'data_dir': os.path.join(DATA_DIR, 'Heart'), 'column_count': 11},
        'partitioner_args': {},
        'needs_preprocessing': ['standard_scale'],
        # 'sampling_config': None, # Original didn't sample Heart
         # Standard Params ...
        'learning_rates_try':[5e-1, 1e-1, 5e-2, 1e-2, 5e-3, 1e-3], 'default_lr':1e-3,
        'reg_params_try':[1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5], 'default_reg_param': 1e-1,
        'batch_size': 64, 'epochs_per_round': 1, 'rounds': 100,
        'runs': 100, 'runs_tune': 10, 'metric': 'F1'
    },
    'ISIC': {
        # Data Handling
        'data_source': 'pre_split_paths_isic',
        'partitioning_strategy': 'pre_split',
        'cost_interpretation': 'site_mapping_key',
        'dataset_class': 'ISICDataset',
        'default_num_clients': 2, # Default for paired runs
        'max_clients': 4,
        'fixed_classes': 8, # Check if your model/labels match this
        'source_args': {
            'data_dir': os.path.join(DATA_DIR, 'ISIC'),
            'site_mappings': { # Define mappings explicitly here
                 0.06: [2, 2], 0.15: [2, 0], 0.19: [2, 3], 0.25: [2, 1],
                 0.3: [1, 3], 'all': [0, 1, 2, 3]
             }
        },
        'partitioner_args': {},
        'sampling_config': {'type': 'fixed_total', 'size': 2000}, # Corresponds to nrows
        # Standard Params ...
        'learning_rates_try': [5e-3, 1e-3, 5e-4, 1e-4], 'default_lr': 1e-3,
        'reg_params_try':[1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5], 'default_reg_param': 1e-1,
        'batch_size': 128, 'epochs_per_round': 1, 'rounds': 60,
        'runs': 5, 'runs_tune': 1, 'metric': 'Balanced_accuracy'
    },
    'IXITiny': {
        # Data Handling
        'data_source': 'pre_split_paths_ixi',
        'partitioning_strategy': 'pre_split',
        'cost_interpretation': 'site_mapping_key',
        'dataset_class': 'IXITinyDataset',
        'default_num_clients': 2,
        'max_clients': 3,
        'fixed_classes': 2,
        'source_args': {
            'data_dir': os.path.join(DATA_DIR, 'IXITiny'),
            'site_mappings': { # Define mappings explicitly
                 0.08: [['Guys'], ['HH']], 0.28: [['IOP'], ['Guys']],
                 0.30: [['IOP'], ['HH']], 'all': [['IOP'], ['HH'], ['Guys']]
             }
        },
        'partitioner_args': {},
        # 'sampling_config': None,
         # Standard Params ...
        'learning_rates_try': [1e-2, 5e-3, 1e-3, 5e-4], 'default_lr': 1e-3,
        'reg_params_try':[1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5], 'default_reg_param': 1e-1,
        'batch_size': 64, 'epochs_per_round': 1, 'rounds': 50,
        'runs': 10, 'runs_tune': 3, 'metric': 'DICE'
    }
}



# --- Lists for quick checks (can be derived from DEFAULT_PARAMS if needed elsewhere) ---
TABULAR = [k for k, v in DEFAULT_PARAMS.items() if v.get('data_source') == 'pre_split_csv']
# CLASS_ADJUST = ['EMNIST', 'CIFAR'] # No longer needed for pipeline logic
SQUEEZE = [k for k, v in DEFAULT_PARAMS.items() if v.get('fixed_classes') is not None and v.get('fixed_classes') > 1 and v.get('metric') != 'R2' and k != 'IXITiny'] # Example derivation
LONG = [k for k, v in DEFAULT_PARAMS.items() if v.get('fixed_classes') is not None and v.get('fixed_classes') > 1 and v.get('metric') != 'R2' and k != 'IXITiny'] # Example derivation
TENSOR = ['IXITiny']
CONTINUOUS_OUTCOME = ['Weather']