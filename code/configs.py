"""
Central configuration file for the OTCost project.

Defines directory paths, constants, default hyperparameters,
algorithm-specific settings (like layers to federate, regularization parameters),
and imports common libraries used throughout the project.
"""
import warnings
# --- Suppress Warnings ---
# Ignore specific warnings for cleaner output
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
from typing import List, Dict, Optional, Tuple, Union, Iterator, Iterable
from datetime import datetime
from functools import wraps
from collections import OrderedDict
from dataclasses import dataclass, field
from itertools import combinations
from functools import partial
from itertools import combinations
from concurrent.futures import ProcessPoolExecutor, as_completed, wait
from multiprocessing import Pool
import numpy as np
import pandas as pd
import scipy.stats
import scipy.stats as stats
from tqdm import tqdm  # Progress bars
from tqdm.contrib.concurrent import process_map  # Progress bars for parallel processing
from PIL import Image
from sklearn import metrics  # General metrics utilities
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, matthews_corrcoef, balanced_accuracy_score
from netrep.metrics import LinearMetric  # Representation similarity metrics
from netrep import convolve_metric  # Representation similarity metrics
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.linalg import svdvals
from torch.utils.data import DataLoader, Dataset, Subset, RandomSampler
from torch.nn.utils.rnn import pad_sequence
import torch.multiprocessing as mp
from torchvision.transforms import transforms
from torchvision.datasets import FashionMNIST, EMNIST, CIFAR10
from transformers import BertModel, BertTokenizer, AutoTokenizer, AutoModel
from sklearn.preprocessing import StandardScaler
from torch.utils.data  import DataLoader, Dataset
import torch
import numpy as np
from torchvision import transforms
import nibabel as nib
from PIL import Image
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import torch
from sklearn.preprocessing import StandardScaler
import albumentations
import random
from monai.transforms import EnsureChannelFirst, AsDiscrete,Compose,NormalizeIntensity,Resize,ToTensor
import pandas as pd
import os
import albumentations  # For image augmentations

# --- Core Directories ---
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT  = os.path.dirname(_CURRENT_DIR)
ROOT_DIR = _PROJECT_ROOT
DATA_DIR = f'{ROOT_DIR}/data'       # Directory containing datasets
EVAL_DIR = f'{ROOT_DIR}/code/evaluation' # Directory for evaluation scripts/results
OTCOST_DIR = f'{ROOT_DIR}/code/OTCost' # Directory for evaluation scripts/results
RESULTS_DIR = f'{ROOT_DIR}/results' # Directory to save experiment results

# --- Add project directories to Python path ---
# Allows importing modules from these directories
sys.path.append(f'{ROOT_DIR}/code')
sys.path.append(f'{EVAL_DIR}')
sys.path.append(f'{OTCOST_DIR}')

# --- Global Settings ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu' # Set computation device
N_WORKERS = 4 # Default number of workers for DataLoader


# --- Supported Algorithms ---
ALGORITHMS = [
    'local',            # Local training only
    'fedavg',           # Federated Averaging
    # 'fedprox',           # Fedprox (personalized FedAvg with proximal term)
    # 'ditto',            # Ditto (dual model training: global and personal)
]

# --- Supported Datasets ---
DATASETS = [
    'Synthetic', # Synthetic dataset
    'Credit',    # Credit card fraud
    'EMNIST',    # EMNIST
    'CIFAR',     # CIFAR-10
    'ISIC'       # Skin dermoscopy
    'IXITiny',   # Brain MRI
    'Weather',    # Weather prediction
    'Heart'
]


# --- Hyperparam Tuning ---
DEFAULT_PARAMS = {
    'Synthetic': {
        'learning_rates_try':[5e-1, 1e-1, 5e-2, 1e-2, 5e-3, 1e-3], # LRs to try during tuning
        'default_lr': 1e-3,                                 # Default LR
        'reg_params_try':[1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5],  # Reg params to try during tuning for pfedme and ditto
        'default_reg_param': 1e-1,                  # Default Reg param
        'num_clients': 5,                           # Number of clients in federation
        'sizes_per_client': 250,                   # Number of samples per client (if simulating)
        'classes': 2,                              # Number of output classes
        'batch_size': 64,                          # Training batch size
        'epochs_per_round': 1,                      # Local epochs per communication round
        'rounds': 100,                              # Total communication rounds
        'runs': 20,                                 # Number of independent runs for final evaluation
        'runs_tune': 10,                             # Number of independent runs for LR tuning
        'metric': 'F1'                              # Evaluation metric
    },

    'Credit': {
        'learning_rates_try':[5e-1, 1e-1, 5e-2, 1e-2, 5e-3, 1e-3], 
        'default_lr':1e-3, 
        'reg_params_try':[1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5], 
        'default_reg_param': 1e-1,
        'num_clients': 5,                          
        'sizes_per_client': 200,                  
        'classes': 2,                             
        'batch_size': 64,                         
        'epochs_per_round': 1,                     
        'rounds': 100,                            
        'runs': 20,                                 
        'runs_tune': 10,                           
        'metric': 'F1'                             
    },

    'EMNIST': {
        'learning_rates_try': [5e-3, 1e-3, 5e-4, 1e-4],
        'default_lr': 1e-3, 
        'reg_params_try':[1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5], 
        'default_reg_param': 1e-1,
        'num_clients': 5,
        'sizes_per_client': 3000,
        'classes': 62,
        'batch_size': 128,
        'epochs_per_round': 1,
        'rounds': 75,
        'runs': 10,
        'runs_tune': 3,
        'metric': 'Accuracy' 
    },
    'CIFAR': {
        'learning_rates_try': [5e-3, 1e-3, 5e-4],
        'default_lr': 1e-3, 
        'reg_params_try':[1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5], 
        'default_reg_param': 1e-1,
        'num_clients': 5,
        'sizes_per_client': 3000,
        'classes': 10,
        'batch_size': 128,
        'epochs_per_round': 1,
        'rounds': 100,
        'runs': 10,
        'runs_tune': 3,
        'metric': 'Accuracy' 
    },
    'ISIC': {
        'learning_rates_try': [5e-3, 1e-3, 5e-4, 1e-4],
        'default_lr': 1e-3,
        'reg_params_try':[1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5], 
        'default_reg_param': 1e-1,
        'num_clients': 4,
        'sizes_per_client': None, # Use actual client sizes
        'classes': 4,
        'batch_size': 128,
        'epochs_per_round': 1,
        'rounds': 60,
        'runs': 5,
        'runs_tune': 1, 
        'metric': 'Balanced_accuracy' 
    },
    'IXITiny': {
        'learning_rates_try': [1e-2, 5e-3, 1e-3, 5e-4],
        'default_lr': 1e-3,
        'reg_params_try':[1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5], 
        'default_reg_param': 1e-1,
        'num_clients': 15,
        'sizes_per_client': None, # Use actual client sizes
        'classes': 2,
        'batch_size': 64,
        'epochs_per_round': 1,
        'rounds': 50,
        'runs': 10,
        'runs_tune': 3,
        'metric': 'DICE'  
    },
    'Weather': {
        'learning_rates_try': [5e-2, 1e-2, 5e-3, 1e-3, 5e-4],
        'default_lr': 5e-3,
        'reg_params_try':[1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5], 
        'default_reg_param': 1e-1,
        'num_clients': 4,
        'sizes_per_client': 500, # Use actual client sizes
        'classes': 2,
        'batch_size': 64,
        'epochs_per_round': 1,
        'rounds': 25,
        'runs': 10,
        'runs_tune': 3,
        'metric': 'R2'  
    },

    'Heart': {
        'learning_rates_try':[5e-1, 1e-1, 5e-2, 1e-2, 5e-3, 1e-3], 
        'default_lr':1e-3, 
        'reg_params_try':[1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5], 
        'default_reg_param': 1e-1,
        'num_clients': 5,                          
        'sizes_per_client': None,                  
        'classes': 2,                             
        'batch_size': 64,                         
        'epochs_per_round': 1,                     
        'rounds': 100,                            
        'runs': 100,                                 
        'runs_tune': 10,                           
        'metric': 'F1'                             
    }

}


TABULAR = ['Synthetic', 'Credit', 'Weather', 'Heart']
CLASS_ADJUST = ['EMNIST', 'CIFAR']
SQUEEZE = ['Synthetic', 'Credit', 'EMNIST', 'CIFAR', 'ISIC', 'Heart']
LONG = ['EMNIST', 'CIFAR', 'ISIC', 'Heart']
CLASS_ADJUST = ['EMNIST', 'CIFAR']
TENSOR = ['IXITiny']
CONTINUOUS_OUTCOME = ['Weather']