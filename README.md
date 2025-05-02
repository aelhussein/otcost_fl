# Federated Learning with Optimal Transport Cost Analysis

## Project Overview

This repository contains a Python-based research framework for conducting Federated Learning (FL) experiments and analyzing the similarity between client data distributions using Optimal Transport (OT). The framework supports:

*   **Federated Learning:** Running FL experiments using algorithms like FedAvg and Local training across various datasets.
*   **Data Heterogeneity:** Simulating different types of data heterogeneity, including:
    *   Label Distribution Skew (Dirichlet distribution)
    *   Feature Distribution Shift (Mean shift, scale shift, rotation/tilt for tabular; Geometric rotation for images)
    *   Concept Shift (Changing decision boundaries)
    *   Pre-defined splits based on real-world site data.
*   **Optimal Transport Analysis:** Calculating the similarity/distance between client data representations (activations) using several OT-based methods:
    *   **Feature-Error OT:** Combines distance in feature space and prediction error space.
    *   **Decomposed OT:** Separates label distribution distance (EMD) and conditional feature distance (within-class OT).
    *   **Fixed-Anchor LOT:** Compares client distributions by mapping them to fixed anchor points (computed via K-Means or class means).
    *   **Direct OT:** Computes OT directly on feature representations, optionally incorporating Hellinger distance between label distributions.
*   **Configurable Pipelines:** Easily configure and run experiments for hyperparameter tuning (learning rate, regularization) and final evaluations.

This codebase is designed for research purposes, prioritizing clarity and extensibility over production-level optimizations or extensive error handling.

## Key Features

*   **Modular Design:** Separates concerns for data processing, FL algorithms (servers/clients), models, configuration, and OT calculations.
*   **Multiple Datasets:** Includes support for Synthetic datasets, tabular datasets (Credit Card Fraud, Heart Disease), and image datasets (EMNIST, CIFAR, ISIC, IXITiny).
*   **Heterogeneity Control:** Uses a `cost` parameter (interpreted differently per dataset, e.g., Dirichlet alpha, shift intensity) to systematically vary data heterogeneity.
*   **Diverse OT Metrics:** Implements several established and novel OT-based methods for analyzing client similarity based on model activations.
*   **Activation Caching:** Caches generated model activations to speed up repeated OT analyses.
*   **Results Management:** Saves FL experiment results (metrics, models) and facilitates loading for analysis and OT calculation.
*   **Slurm Integration:** Includes an example Slurm script (`submit_evaluation.sh`) for running FL experiments on a cluster.

## Directory Structure
```bash
├── code/ # Main source code
│ ├── evaluation/ # Scripts for running FL experiments (run.py, submit_evaluation.sh)
│ ├── init.py
│ ├── clients.py
│ ├── configs.py # Central configuration file
│ ├── data_loading.py
│ ├── data_manager.py # Orchestrates data for FL pipeline
│ ├── data_partitioning.py
│ ├── data_processing.py
│ ├── data_sets.py
│ ├── helper.py # Core utilities, dataclasses, metrics
│ ├── losses.py
│ ├── models.py # Model architectures
│ ├── ot_calculators.py # OT cost calculation logic
│ ├── ot_data_manager.py # Loads activations/performance for OT
│ ├── ot_pipeline_runner.py # Runs the OT analysis pipeline
│ ├── ot_results_analysis.py # Plotting OT results
│ ├── ot_utils.py # OT mathematical helpers
│ ├── pipeline.py # FL experiment orchestration
│ ├── results_manager.py # Handles loading/saving FL results/models
│ ├── servers.py
│ └── synthetic_data.py # Generates synthetic data and shifts
├── data/ # Raw and processed datasets (organized by dataset name)
│ ├── CIFAR/
│ ├── Credit/
│ ├── EMNIST/
│ ├── Heart/
│ ├── ISIC/
│ ├── IXITiny/
│ └── ... # Other dataset directories
├── results/ # Stores FL experiment results (metrics, logs)
│ ├── lr_tuning/
│ ├── reg_param_tuning/
│ ├── evaluation/
│ └── ...
├── saved_models/ # Saved model checkpoints from FL evaluation runs
│ ├── CIFAR/
│ ├── Credit/
│ └── ...
├── activations/ # Cached activation data for OT analysis (organized by dataset)
│ ├── CIFAR/
│ ├── Credit/
│ └── ...
└── logs/ # Slurm logs (if using submit script)
├── outputs/
└── errors/
```


## Setup

1.  **Environment:** Ensure you have a Python environment (e.g., Conda) with necessary packages installed. Create a `requirements.txt` file and install using `pip install -r requirements.txt`. Key dependencies include PyTorch, NumPy, Pandas, Scipy, POT (Python Optimal Transport), scikit-learn, Matplotlib, Seaborn, etc.
2.  **CUDA (Optional):** If using GPUs, ensure CUDA toolkit and compatible PyTorch versions are installed. The code will automatically use CUDA if available.
3.  **Data:** Place datasets in the `data/` directory, following the expected structure for each dataset loader (see `data_loading.py` and `configs.py`). Some datasets (e.g., CIFAR, EMNIST) may be downloaded automatically on first run if not found.

## Configuration (`configs.py`)

This is the central file for defining experiment parameters:

*   **`DEFAULT_PARAMS`:** Contains a dictionary where keys are dataset names. Each dataset has sub-dictionaries defining:
    *   FL parameters (learning rates, rounds, epochs, batch size, metric).
    *   Data handling (data source, partitioning strategy, sampling, shift types/parameters).
    *   Model and Dataset class names.
    *   Configuration for hyperparameter tuning runs.
*   **`DATASET_COSTS`:** Defines the list of heterogeneity parameters (`cost`) to iterate over for each dataset during experiments. The meaning of `cost` depends on the dataset configuration (e.g., Dirichlet alpha, shift intensity).
*   **Paths:** Defines root directories for data, results, models, activations.
*   **Global Settings:** Device preference (`DEVICE`), number of workers (`N_WORKERS`).
*   **Algorithms:** Lists supported FL algorithms (`ALGORITHMS`).

**Before running experiments, review and potentially adjust the settings in `configs.py` for your target datasets and experimental setup.**

# Running Experiments

There are two main types of pipelines: Federated Learning experiments and Optimal Transport analysis.

## 1. Federated Learning Experiments

This pipeline runs the actual FL training and evaluation.

**Entry Point:** `code/evaluation/run.py`

**Usage:**

```bash
python code/evaluation/run.py -ds <DATASET_NAME> -exp <EXPERIMENT_TYPE> [-nc <NUM_CLIENTS>]
```

Arguments:
- `-ds`, `--dataset`: Required. The name of the dataset to use (must match a key in configs.DEFAULT_PARAMS).
- `-exp`, `--experiment_type`: Required. The type of experiment:
    - learning_rate: Performs learning rate tuning runs.
    - reg_param: Performs regularization parameter tuning runs (for algorithms like FedProx).
    - evaluation: Runs the final evaluation using the best hyperparameters found during tuning (or defaults). This saves models and detailed metrics needed for OT analysis.
- `-nc`, `--num_clients`: Optional. An integer to override the default number of clients specified for the dataset in configs.py. If omitted, the config default is used.

Run final evaluation for CIFAR using the default number of clients
```bash 
python code/evaluation/run.py -ds CIFAR -exp evaluation
```

Run learning rate tuning for Synthetic_Feature using 10 clients
```bash
python code/evaluation/run.py -ds Synthetic_Feature -exp learning_rate -nc 10
```

### Batch Submission (slurm)
The `submit_evaluation.sh` script provides an example for submitting FL jobs to a Slurm cluster.
- Customize: Modify the script's default values (datasets, experiment types, directories, Conda environment) as needed.
- Run: Execute the script (`bash submit_evaluation.sh` or `bash submit_evaluation.sh --datasets=CIFAR,EMNIST --exp-types=evaluation --num-clients=5`) to submit jobs for the specified configurations.
- Logs: Output and error logs will be saved in the logs/ directory.

#### Outputs:
    - Results: Detailed metrics and run metadata are saved as .pkl files in the `results/<experiment_type>/` directory.
    - Models: During evaluation runs for fedavg, the model state dictionaries (round0, best, final) are saved in the `saved_models/<dataset>/evaluation/` directory. These are crucial for the OT analysis pipeline.

## 2. Optimal Transport (OT) Analysis
This pipeline calculates OT-based similarity metrics between client pairs using the activations generated from saved FL models (typically the initial round0 model).

**Prerequisites**: You must first run the FL evaluation experiment for the desired dataset, cost, and seed to generate the necessary round0 saved model.

**Core Logic**: ot_pipeline_runner.py (contains PipelineRunner class) and ot_calculators.py.

**How to Run**:
You need to create a separate Python script (or use a Jupyter notebook) to instantiate PipelineRunner and call its run_pipeline method.

### **Steps**:
**1. Import necessary classes**:
```python
from ot_pipeline_runner import PipelineRunner
from ot_calculators import OTConfig
```

**2. Define OT Configurations**: Create a list of OTConfig objects, specifying which OT methods and parameter sets you want to run.
```python
# Example OT configurations
ot_configs = [
    # Feature-Error OT with default weights, uniform marginals
    OTConfig(method_type='feature_error', name='FE_Uniform',
             params={'alpha': 1.0, 'beta': 1.0, 'use_loss_weighting': False, 'normalize_cost': True}),

    # Feature-Error OT with loss weighting
    OTConfig(method_type='feature_error', name='FE_LossW',
             params={'alpha': 1.0, 'beta': 1.0, 'use_loss_weighting': True, 'normalize_cost': True}),

    # Decomposed OT
    OTConfig(method_type='decomposed', name='Decomposed_MeanAgg',
             params={'aggregate_conditional_method': 'mean', 'normalize_emd': True, 'normalize_cost': True}),

    # Direct OT with Hellinger
    OTConfig(method_type='direct_ot', name='DirectOT_Hellinger_Cosine',
             params={'distance_method': 'cosine', 'use_label_hellinger': True,
                     'feature_weight': 1.0, 'label_weight': 1.0, 'compress_vectors': False}),

    # Add more configurations as needed...
]
```

**3. Instantiate PipelineRunner**: Specify the total number of clients involved in the original FL run whose results you are analyzing.
```python
# Example: Analyzing results from a 5-client run
num_fl_clients = 5
runner = PipelineRunner(num_clients=num_fl_clients)
```

**4. Call run_pipeline**: Provide the necessary arguments.
```python
dataset = 'CIFAR' # Or your target dataset
num_classes = 10 # For CIFAR
costs = [0.1, 0.5, 1.0, 10.0] # Costs matching the FL evaluation run
base_seed_fl = 42 # The seed used for the FL evaluation run

results_df = runner.run_pipeline(
    dataset_name=dataset,
    costs_list=costs,
    num_classes=num_classes,
    ot_configurations=ot_configs,
    client_pairs_to_analyze=None, # Analyze all pairs, or provide e.g., [('client_1', 'client_2')]
    activation_loader_type='val', # Use validation set for activations
    performance_aggregation='mean', # How to average FL performance scores across runs
    base_seed=base_seed_fl, # Seed of the FL run to load models/results from
    force_activation_regen=False, # Use cached activations if available
    model_type='round0' # IMPORTANT: Use the initial model for activations
)

# Process or save the results_df
print(results_df.head())
# Optionally save: results_df.to_csv(f'ot_results_{dataset}.csv', index=False)

# Optionally plot results
# from ot_results_analysis import plot_ot_metrics_vs_perf_delta
# plot_ot_metrics_vs_perf_delta(results_df, main_title=f"OT vs Perf Delta - {dataset}")
```


**Outputs**:
- Activations: If not cached or force_regenerate=True, activation files (h, p_prob, y tensors for client pairs) will be saved in the activations/ directory.
- Results DataFrame: The run_pipeline method returns a Pandas DataFrame containing the calculated OT costs/scores for each client pair, cost level, and OT configuration, along with the corresponding FL performance delta.
- Plots (Optional): The ot_results_analysis.py script provides functions to plot the relationship between OT metrics and FL performance degradation.


### Data Requirements
Datasets should be placed in the data/ directory, organized into subdirectories named after the dataset key used in configs.py (e.g., data/CIFAR/, data/Heart/).
The specific file structure required within each dataset directory depends on the corresponding loader function in data_loading.py. Check the loader implementation for details (e.g., credit_csv expects creditcard.csv, heart_site_loader expects processed.<site>.data).
Torchvision datasets (CIFAR, EMNIST) will attempt to download automatically if not found in the specified data_dir.