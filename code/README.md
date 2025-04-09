# Code for Paper: A Universal Metric of Dataset Similarity for Cross-Silo Federated Learning

This repository contains the code used in the paper *"A Universal Metric of Dataset Similarity for Cross-Silo Federated Learning."*

## Running the Code

To execute the code, use the following command:

`bash script_gpu_run_models
--dataset <dataset_name>
--tune [optional]
--learning_rate [optional]
--regularization [optional]`



### Command-line Arguments:

- `--dataset`  
  **Description**: Specifies the dataset to use for the experiments.  
  **Example**: `--dataset cifar10`

- `--tune`  
  **Description**: Enables hyperparameter tuning. This must be specified to run tuning-related arguments like `--learning_rate` or `--regularization`.

- `--learning_rate`  
  **Description**: Runs learning rate tuning. **Requires `--tune` to be specified.**  
  **Example**: `--learning_rate 0.01`

- `--regularization`  
  **Description**: Runs regularization parameter tuning. **Requires `--tune` to be specified.**  
  **Example**: `--regularization 0.1`

---

## Directory Structure

Before running the code, ensure that the directory names are updated accordingly:

- **Main Directory**: `/gpfs/commons/groups/gursoy_lab/aelhussein/ot_cost/otcost_fl/code/`
- **Run Models Directory**: `/gpfs/commons/groups/gursoy_lab/aelhussein/ot_cost/otcost_fl/code/run_models`

---

## Datasets

For details about the datasets available for these experiments, please refer to the paper.
