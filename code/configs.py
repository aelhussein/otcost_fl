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
# Structure:
# 'DatasetName': {
#     # --- Data Handling Config ---
#     'data_source': Type of raw data source ('torchvision', 'pre_split_csv', 'pre_split_paths_ixi', etc.)
#     'partitioning_strategy': How to divide data ('dirichlet_indices', 'pre_split', 'iid_indices', etc.)
#     'cost_interpretation': How to use 'cost' value ('alpha', 'inv_alpha', 'file_suffix', 'site_mapping_key', 'ignore')
#     'dataset_class': String name of the final PyTorch Dataset class (e.g., 'CIFARDataset')
#     'num_clients': Default number of clients (can be int or 'dynamic')
#     'fixed_classes': Number of output classes (int) or None
#     'source_args': Dict of args for the data source loader (e.g., {'dataset_name': 'CIFAR10'})
#     'partitioner_args': Dict of args for the partitioner function (often empty)
#     'partition_scope': Which dataset split to partition ('train' or 'all') - default 'train'
#     'needs_preprocessing': Optional bool/list indicating steps like scaling for tabular
#     'sampling_config': Optional dict for post-load/partition sampling (e.g., {'type': 'per_class', 'size': 250})
#     # --- Standard Training Hyperparameters ---
#     'learning_rates_try': List of LRs for tuning
#     'default_lr': Default LR
#     'reg_params_try': List of regularization params for tuning (if applicable)
#     'default_reg_param': Default reg param (if applicable)
#     'batch_size': Training batch size
#     'epochs_per_round': Local epochs per round
#     'rounds': Total communication rounds
#     'runs': Number of final evaluation runs
#     'runs_tune': Number of tuning runs
#     'metric': Primary evaluation metric name
# }

DEFAULT_PARAMS = {
    'CIFAR': {
        # Data Handling
        'data_source': 'torchvision',
        'partitioning_strategy': 'dirichlet_indices',
        'cost_interpretation': 'alpha', # Assume costs in run.py ARE alpha values
        'dataset_class': 'CIFARDataset', # Final wrapper class in data_processing.py
        'num_clients': 5,
        'fixed_classes': 10,
        'source_args': {'dataset_name': 'CIFAR10', 'data_dir': os.path.join(DATA_DIR, 'CIFAR')},
        'partitioner_args': {},
        'partition_scope': 'train', # Only partition the training set
        'sampling_config': {'type': 'fixed_total', 'size': 1000, 'replace': False}, # sizes_per_client
        # Standard Params
        'learning_rates_try': [5e-3, 1e-3, 5e-4], 'default_lr': 1e-3,
        'reg_params_try':[1, 1e-1, 1e-2, 1e-3], 'default_reg_param': 1e-1, # Example regs
        'batch_size': 128, 'epochs_per_round': 1, 'rounds': 100,
        'runs': 10, 'runs_tune': 3, 'metric': 'Accuracy'
    },
    'EMNIST': {
        # Data Handling
        'data_source': 'torchvision',
        'partitioning_strategy': 'dirichlet_indices',
        'cost_interpretation': 'alpha', # Assume costs in run.py ARE alpha values
        'dataset_class': 'EMNISTDataset',
        'num_clients': 5,
        'fixed_classes': 10,
        'source_args': {'dataset_name': 'EMNIST', 'data_dir': os.path.join(DATA_DIR, 'EMNIST'), 'split': 'digits'}, # Specify digits split
        'partitioner_args': {},
        'partition_scope': 'train',
        'sampling_config': {'type': 'fixed_total', 'size': 1000, 'replace': False}, # sizes_per_client
        # Standard Params
        'learning_rates_try': [5e-3, 1e-3, 5e-4, 1e-4], 'default_lr': 1e-3,
        'reg_params_try':[1, 1e-1, 1e-2, 1e-3], 'default_reg_param': 1e-1,
        'batch_size': 128, 'epochs_per_round': 1, 'rounds': 75,
        'runs': 10, 'runs_tune': 3, 'metric': 'Accuracy'
    },
    'Synthetic': {
        # Data Handling
        'data_source': 'pre_split_csv',
        'partitioning_strategy': 'pre_split', # Data already split by client in files
        'cost_interpretation': 'file_suffix', # Cost is part of filename format _<cost:.2f>.csv
        'dataset_class': 'SyntheticDataset',
        'num_clients': 5, # Assumes files exist for clients 1 to 5 for each cost
        'fixed_classes': 2,
        'source_args': {'data_dir': os.path.join(DATA_DIR, 'Synthetic'), 'column_count': 11},
        'partitioner_args': {}, # Not needed for pre_split
        'needs_preprocessing': ['standard_scale'], # Indicate tabular scaling
        'sampling_config': {'type': 'fixed_total', 'size': 250, 'replace': False}, # sizes_per_client
        # Standard Params
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
        'num_clients': 5,
        'fixed_classes': 2,
        'source_args': {'data_dir': os.path.join(DATA_DIR, 'Credit'), 'column_count': 29},
        'partitioner_args': {},
        'needs_preprocessing': ['standard_scale'],
        'sampling_config': {'type': 'fixed_total', 'size': 200, 'replace': True}, # sizes_per_client w/ replacement
         # Standard Params
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
        'num_clients': 4, # Example, adjust if needed
        'fixed_classes': None, # Regression task
        'source_args': {'data_dir': os.path.join(DATA_DIR, 'Weather'), 'column_count': 124},
        'partitioner_args': {},
        'needs_preprocessing': ['standard_scale'],
        'sampling_config': {'type': 'fixed_total', 'size': 500, 'replace': False},
         # Standard Params
        'learning_rates_try': [5e-2, 1e-2, 5e-3, 1e-3, 5e-4], 'default_lr': 5e-3,
        'reg_params_try':[1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5], 'default_reg_param': 1e-1,
        'batch_size': 64, 'epochs_per_round': 1, 'rounds': 25,
        'runs': 10, 'runs_tune': 3, 'metric': 'R2'
    },
     'Heart': {
         # Data Handling
        'data_source': 'pre_split_csv',
        'partitioning_strategy': 'pre_split',
        'cost_interpretation': 'file_suffix', # Assuming files like data_1_1.00.csv etc exist
        'dataset_class': 'HeartDataset',
        'num_clients': 6, # Based on cost values [1..6]
        'fixed_classes': 2,
        'source_args': {'data_dir': os.path.join(DATA_DIR, 'Heart'), 'column_count': 11},
        'partitioner_args': {},
        'needs_preprocessing': ['standard_scale'],
        # 'sampling_config': None, # No sampling mentioned in original
         # Standard Params
        'learning_rates_try':[5e-1, 1e-1, 5e-2, 1e-2, 5e-3, 1e-3], 'default_lr':1e-3,
        'reg_params_try':[1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5], 'default_reg_param': 1e-1,
        'batch_size': 64, 'epochs_per_round': 1, 'rounds': 100,
        'runs': 100, 'runs_tune': 10, 'metric': 'F1' # Check metric, paper uses F1?
    },
    'ISIC': {
        # Data Handling
        'data_source': 'pre_split_paths_isic',
        'partitioning_strategy': 'pre_split', # Defined by site mapping
        'cost_interpretation': 'site_mapping_key',
        'dataset_class': 'ISICDataset',
        'num_clients': 'dynamic', # 2 or 4
        'fixed_classes': 8, # ISIC has 8 classes {MEL, NV, BCC, AK, BKL, DF, VASC, SCC} - Check this! Original config had 4?
        'source_args': {'data_dir': os.path.join(DATA_DIR, 'ISIC')},
        'partitioner_args': {},
        # 'sampling_config': {'type': 'fixed_total', 'size': 2000, 'replace': False}, # nrows=2000 in loader
        # Standard Params
        'learning_rates_try': [5e-3, 1e-3, 5e-4, 1e-4], 'default_lr': 1e-3,
        'reg_params_try':[1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5], 'default_reg_param': 1e-1,
        'batch_size': 128, 'epochs_per_round': 1, 'rounds': 60,
        'runs': 5, 'runs_tune': 1, 'metric': 'Balanced_accuracy'
    },
    'IXITiny': {
        # Data Handling
        'data_source': 'pre_split_paths_ixi',
        'partitioning_strategy': 'pre_split', # Defined by site mapping
        'cost_interpretation': 'site_mapping_key',
        'dataset_class': 'IXITinyDataset',
        'num_clients': 'dynamic', # 2 or 3
        'fixed_classes': 2, # Background vs Brain
        'source_args': {'data_dir': os.path.join(DATA_DIR, 'IXITiny')},
        'partitioner_args': {},
        # 'sampling_config': None,
         # Standard Params
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