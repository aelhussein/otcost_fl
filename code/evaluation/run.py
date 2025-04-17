"""
Entry point script for running federated learning experiments.

Parses command-line arguments to select the dataset and experiment type
(learning rate tuning or final evaluation), creates the necessary directories,
initializes the `Experiment` class from `pipeline.py`, and executes the
experiment run.
"""

import argparse
import os
import sys


_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_CURRENT_DIR)
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, _CURRENT_DIR)

# --- Import Project Modules ---
from configs import DATASETS, RESULTS_DIR, DEFAULT_PARAMS, DATASET_COSTS
from pipeline import ExperimentType, ExperimentConfig, Experiment


def run_experiments(dataset: str, experiment_type: str, num_clients_override: int = None): # MODIFIED: Added num_clients_override
    """
    Initializes and runs a federated learning experiment.

    Args:
        dataset (str): The name of the dataset to use.
        experiment_type (str): The type of experiment to run.
        num_clients_override (int, optional): Number of clients to use, overriding config default. Defaults to None.
    """
    print(f"Preparing to run experiment: Dataset='{dataset}', Type='{experiment_type}', Client Override={num_clients_override}")
    # Create an ExperimentConfig object, passing the override
    config = ExperimentConfig(
        dataset=dataset,
        experiment_type=experiment_type,
        num_clients=num_clients_override # MODIFIED: Pass override to config
        )
    # Instantiate the Experiment class
    experiment_runner = Experiment(config)
    # Execute the experiment, passing the list of cost/heterogeneity parameters
    costs_to_run = DATASET_COSTS.get(dataset, [])
    if not costs_to_run:
        print(f"Warning: No costs defined in DATASET_COSTS for dataset '{dataset}'. Skipping.", file=sys.stderr)
        return None

    print(f"Running experiment for {dataset} with parameters: {costs_to_run}")
    results = experiment_runner.run_experiment(costs_to_run) # run_experiment internally uses config.num_clients
    print(f"Experiment completed: Dataset='{dataset}', Type='{experiment_type}'")
    return results

def main():
    """
    Parses command-line arguments and initiates the experiment run.
    """
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(
        description='Run Federated Learning experiments.'
    )
    parser.add_argument(
        "-ds", "--dataset",
        required=True,
        choices=list(DEFAULT_PARAMS.keys()),
        help="Select the dataset for the experiment."
    )
    parser.add_argument(
        "-exp", "--experiment_type",
        required=True,
        choices=[ExperimentType.LEARNING_RATE, ExperimentType.EVALUATION, ExperimentType.REG_PARAM], # Added REG_PARAM choice
        help="Select the type of experiment: 'learning_rate', 'reg_param', or 'evaluation'."
    )
    # --- MODIFICATION START: Add num_clients argument ---
    parser.add_argument(
        "-nc", "--num_clients",
        type=int,
        default=None, # Default is None, meaning use config default
        help="Override the default number of clients specified in configs.py."
    )
    # --- MODIFICATION END ---

    args = parser.parse_args()

    # --- Directory Setup ---
    try:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        for exp_type_val in [ExperimentType.LEARNING_RATE, ExperimentType.EVALUATION, ExperimentType.REG_PARAM, ExperimentType.DIVERSITY]:
            dir_name = exp_type_val # Default name
            if exp_type_val == ExperimentType.LEARNING_RATE: dir_name = 'lr_tuning'
            if exp_type_val == ExperimentType.EVALUATION: dir_name = 'evaluation'
            if exp_type_val == ExperimentType.REG_PARAM: dir_name = 'reg_param_tuning'
            if exp_type_val == ExperimentType.DIVERSITY: dir_name = 'diversity'
            os.makedirs(os.path.join(RESULTS_DIR, dir_name), exist_ok=True)
        print(f"Results will be saved in subdirectories under: {RESULTS_DIR}")
    except OSError as e:
        print(f"Error creating results directories: {e}", file=sys.stderr)
        sys.exit(1)


    # --- Run Experiment ---
    try:
        # Pass the num_clients argument from CLI to run_experiments
        run_experiments(args.dataset, args.experiment_type, args.num_clients) # MODIFIED: Pass args.num_clients
    except ValueError as ve:
        print(f"Configuration or Value Error: {ve}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
    except NotImplementedError as nie:
        print(f"Functionality Error: {nie}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
    except FileNotFoundError as fnf:
        print(f"File Not Found Error: {fnf}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during the experiment: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()