import torch
ROOT_DIR = '/gpfs/commons/groups/gursoy_lab/aelhussein/ot_cost/otcost_fl'
LEARNING_RATES_TRY = [1e-1, 5e-2, 1e-2, 5e-3, 5e-4]
REG_PARAMS_TRY = [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]    
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TABULAR = ['Synthetic', 'Credit', 'Weather']
CLASS_ADJUST = ['EMNIST', 'CIFAR'] # as each dataset cost has different labels


Synthetic_LR_dict = {0.03:{'single': 5e-3, 'joint': 5e-2, 'federated': 5e-2, 'pfedme':5e-2, 'ditto':1e-1},
            0.10:{'single': 5e-3, 'joint': 5e-2, 'federated': 1e-1, 'pfedme':1e-1, 'ditto':1e-1},
            0.20:{'single': 1e-2, 'joint': 1e-1, 'federated': 1e-1, 'pfedme':1e-1, 'ditto':1e-1},
            0.30:{'single': 5e-2, 'joint': 5e-3, 'federated': 5e-2, 'pfedme':1e-1, 'ditto':1e-1},
            0.40:{'single': 5e-2, 'joint': 1e-2, 'federated': 5e-2, 'pfedme':5e-2, 'ditto':1e-1},
            0.50:{'single': 5e-2, 'joint': 3e-2, 'federated': 5e-2, 'pfedme':1e-1, 'ditto':1e-1}}

Synthetic_REG_PARAM_dict = {0.03:{'pfedme':5e-1, 'ditto':5e-1},
            0.10:{ 'pfedme':5e-1, 'ditto':1},
            0.20:{'pfedme':1e-1, 'ditto':1e-1},
            0.30:{'pfedme':1e-2, 'ditto':1e-2},
            0.40:{'pfedme':1e-2, 'ditto':1e-2},
            0.50:{'pfedme':1e-3, 'ditto':1e-3}}

Credit_LR_dict = {0.12:{'single': 5e-3, 'joint': 5e-2, 'federated': 5e-2, 'pfedme':1e-1, 'ditto':1e-1},
            0.23:{'single': 5e-3, 'joint': 1e-2, 'federated': 1e-2, 'pfedme':1e-1, 'ditto':1e-1},
            0.30:{'single': 1e-2, 'joint': 1e-2, 'federated': 1e-2, 'pfedme':1e-1, 'ditto':1e-1},
            0.40:{'single': 5e-2, 'joint': 1e-3, 'federated': 5e-3, 'pfedme':1e-1, 'ditto':5e-2}}

Credit_REG_PARAM_dict = {0.12:{'pfedme':5e-1, 'ditto':1},
            0.23:{ 'pfedme':1e-1, 'ditto':1e-1},
            0.30:{'pfedme':1e-2, 'ditto':1e-3},
            0.40:{'pfedme':1e-2, 'ditto':1e-3}}

Weather_LR_dict = {0.11:{'single': 3e-3, 'joint': 1e-2, 'federated': 5e-2, 'pfedme':1e-1, 'ditto':1e-1},
            0.19:{'single': 5e-3, 'joint': 5e-2, 'federated': 1e-1, 'pfedme':1e-1, 'ditto':1e-1},
            0.30:{'single': 1e-2, 'joint': 1e-2, 'federated': 5e-2, 'pfedme':1e-1, 'ditto':1e-1},
            0.40:{'single': 1e-2, 'joint': 1e-3, 'federated': 2e-2, 'pfedme':1e-1, 'ditto':1e-1},
            0.48:{'single': 1e-2, 'joint': 1e-3, 'federated': 5e-2, 'pfedme':1e-1, 'ditto':1e-1}}

Weather_REG_PARAM_dict = {0.11:{'pfedme':5e-1, 'ditto':1},
            0.19:{ 'pfedme':5e-1, 'ditto':1},
            0.30:{'pfedme':1e-1, 'ditto':5e-1},
            0.40:{'pfedme':1e-2, 'ditto':1e-2},
            0.48:{'pfedme':1e-3, 'ditto':1e-2}}

EMNIST_LR_dict = {0.11:{'single': 5e-2, 'joint': 1e-2, 'federated': 5e-2, 'pfedme':5e-2, 'ditto':1e-1},
            0.19:{'single': 5e-3, 'joint': 5e-3, 'federated': 5e-2, 'pfedme':5e-2, 'ditto':1e-1},
            0.25:{'single': 1e-2, 'joint': 5e-3, 'federated': 5e-2, 'pfedme':5e-2, 'ditto':1e-1},
            0.34:{'single': 1e-2, 'joint': 1e-2, 'federated': 1e-2, 'pfedme':5e-2, 'ditto':1e-1},
            0.39:{'single': 1e-2, 'joint': 5e-3, 'federated': 3e-2, 'pfedme':1e-1, 'ditto':5e-2}}

EMNIST_REG_PARAM_dict = {0.11:{'pfedme':1e-1, 'ditto':5e-1},
            0.19:{ 'pfedme':1e-1, 'ditto':1e-1},
            0.25:{'pfedme':1e-2, 'ditto':1e-2},
            0.34:{'pfedme':1e-2, 'ditto':1e-3},
            0.39:{'pfedme':1e-3, 'ditto':1e-3}}

CIFAR_LR_dict = {0.08:{'single': 1e-3, 'joint': 5e-4, 'federated': 1e-2, 'pfedme':1e-2, 'ditto':5e-3},
            0.21:{'single': 1e-3, 'joint': 5e-3, 'federated': 1e-2, 'pfedme':5e-2, 'ditto':1e-2},
            0.30:{'single': 5e-4, 'joint': 5e-2, 'federated': 1e-2, 'pfedme':5e-2, 'ditto':1e-2},
            0.38:{'single': 5e-4, 'joint': 5e-3, 'federated': 5e-2, 'pfedme':5e-2, 'ditto':5e-3}}

CIFAR_REG_PARAM_dict = {0.08:{'pfedme':5e-1, 'ditto':1},
            0.21:{ 'pfedme':1e-1, 'ditto':5e-1},
            0.3:{'pfedme':1e-2, 'ditto':1e-2},
            0.38:{'pfedme':1e-3, 'ditto':1e-2}}

IXITiny_LR_dict = {0.08:{'single': 1e-1, 'joint': 1e-1, 'federated': 5e-2, 'pfedme':5e-2, 'ditto':1e-1},
            0.28:{'single': 1e-1, 'joint': 1e-1, 'federated': 1e-1, 'pfedme':5e-2, 'ditto':1e-1},
            0.30:{'single': 1e-1, 'joint': 1e-2, 'federated': 1e-1, 'pfedme':5e-2, 'ditto':1e-1}}

IXITiny_REG_PARAM_dict = {0.08:{'pfedme':5e-1, 'ditto':1e-1},
            0.28:{ 'pfedme':1e-2, 'ditto':1e-2},
            0.3:{'pfedme':1e-3, 'ditto':1e-2}}

ISIC_LR_dict = {0.06:{'single': 5e-4, 'joint': 5e-3, 'federated': 1e-1, 'pfedme':5e-3, 'ditto':1e-2},
            0.15:{'single': 5e-3, 'joint': 5e-3, 'federated': 1e-2, 'pfedme':5e-3, 'ditto':1e-1},
            0.19:{'single': 5e-3, 'joint': 5e-3, 'federated': 1e-2, 'pfedme':5e-3, 'ditto':1e-1},
            0.25:{'single': 5e-3, 'joint': 1e-2, 'federated': 1e-2, 'pfedme':5e-3, 'ditto':1e-2},
            0.3:{'single': 5e-3, 'joint': 5e-3, 'federated': 5e-2, 'pfedme':1e-2, 'ditto':1e-2},}

ISIC_REG_PARAM_dict = {0.06:{'pfedme':5e-1, 'ditto':5e-1},
            0.15:{ 'pfedme':1e-1, 'ditto':1e-1},
            0.19:{'pfedme':1e-1, 'ditto':1e-1},
            0.25:{'pfedme':1e-2, 'ditto':1e-2},
             0.3:{'pfedme':1e-3, 'ditto':1e-2},}