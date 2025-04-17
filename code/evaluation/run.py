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

# Define the set of 'cost' or heterogeneity parameters to iterate over for each dataset.
# The *meaning* of these values (e.g., alpha, 1/alpha, file key) is determined downstream
# by the dataset's configuration in configs.py and the pipeline's cost translation logic.
DATASET_COSTS = {
    'IXITiny': [0.08, 0.28, 0.30, 'all'], # Interpreted as site mapping keys
    'ISIC': [0.06, 0.15, 0.19, 0.25, 0.3, 'all'], # Interpreted as site mapping keys
    # For Dirichlet partitioned datasets, these are intended as direct alpha values:
    'EMNIST': [0.1, 0.5, 1.0, 5.0, 10.0],  # Direct alpha values (low alpha = high non-IID)
    'CIFAR': [0.1, 0.5, 1.0, 5.0, 10.0],   # Direct alpha values
    # For others, interpreted as file suffixes:
    'Synthetic': [0.01, 0.03, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.69],
    'Credit': [0.12, 0.16, 0.22, 0.23, 0.27, 0.3, 0.34, 0.4],
    'Weather': [0.11, 0.19, 0.3, 0.4, 0.48],
    'Heart': [1,2,3,4,5,6] # Example: Might be interpreted differently (e.g., number of features?)
}

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_CURRENT_DIR)
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, _CURRENT_DIR)

# --- Import Project Modules ---
# Make sure DATASETS is imported correctly if used in main() choices
from configs import DATASETS, RESULTS_DIR, DEFAULT_PARAMS # Import DEFAULT_PARAMS if needed for validation
from pipeline import ExperimentType, ExperimentConfig, Experiment


def run_experiments(dataset: str, experiment_type: str):
    """
    Initializes and runs a federated learning experiment.

    Args:
        dataset (str): The name of the dataset to use (must be in `configs.DATASETS`).
        experiment_type (str): The type of experiment to run ('learning_rate' or 'evaluation').
                                Corresponds to `ExperimentType` enum values.

    Returns:
        Dict: The results dictionary returned by the `Experiment.run_experiment()` method.
              Can be large and complex depending on the experiment.
    """
    print(f"Preparing to run experiment: Dataset='{dataset}', Type='{experiment_type}'")
    # Create an ExperimentConfig object
    config = ExperimentConfig(dataset=dataset, experiment_type=experiment_type)
    # Instantiate the Experiment class
    experiment_runner = Experiment(config)
    # Execute the experiment, passing the list of cost/heterogeneity parameters
    costs_to_run = DATASET_COSTS.get(dataset, [])
    if not costs_to_run:
        print(f"Warning: No costs defined in DATASET_COSTS for dataset '{dataset}'. Skipping.", file=sys.stderr)
        return None

    print(f"Running experiment for {dataset} with parameters: {costs_to_run}")
    results = experiment_runner.run_experiment(costs_to_run)
    print(f"Experiment completed: Dataset='{dataset}', Type='{experiment_type}'")
    return results

def main():
    """
    Parses command-line arguments and initiates the experiment run.
    """
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(
        description='Run Federated Learning experiments.' # Simplified description
    )
    parser.add_argument(
        "-ds", "--dataset",
        required=True,
        # Use keys from DEFAULT_PARAMS as the canonical list of supported datasets
        choices=list(DEFAULT_PARAMS.keys()),
        help="Select the dataset for the experiment."
    )
    parser.add_argument(
        "-exp", "--experiment_type",
        required=True,
        choices=[ExperimentType.LEARNING_RATE, ExperimentType.EVALUATION], # Use enum values
        help="Select the type of experiment: 'learning_rate' for tuning or 'evaluation' for final runs."
    )

    args = parser.parse_args()

    # --- Directory Setup ---
    # Ensure necessary results directories exist before starting
    try:
        # Base results directory
        os.makedirs(RESULTS_DIR, exist_ok=True)
        # Subdirectories for specific experiment types
        # Consider creating these dynamically based on ExperimentType enum members
        for exp_type_val in [ExperimentType.LEARNING_RATE, ExperimentType.EVALUATION, ExperimentType.REG_PARAM, ExperimentType.DIVERSITY]:
             # Get the directory name from ResultsManager structure if possible, or map manually
             dir_name = exp_type_val # Simple mapping for now
             if exp_type_val == ExperimentType.LEARNING_RATE: dir_name = 'lr_tuning'
             if exp_type_val == ExperimentType.EVALUATION: dir_name = 'evaluation'
             # Add mappings for REG_PARAM, DIVERSITY if needed
             os.makedirs(os.path.join(RESULTS_DIR, dir_name), exist_ok=True)

        print(f"Results will be saved in subdirectories under: {RESULTS_DIR}")
    except OSError as e:
        print(f"Error creating results directories: {e}", file=sys.stderr)
        sys.exit(1)


    # --- Run Experiment ---
    try:
        run_experiments(args.dataset, args.experiment_type)
    except ValueError as ve:
        print(f"Configuration or Value Error: {ve}", file=sys.stderr)
        sys.exit(1)
    except NotImplementedError as nie:
        print(f"Functionality Error: {nie}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError as fnf:
        print(f"File Not Found Error: {fnf}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        # Catch any other unexpected errors during the experiment run
        print(f"An unexpected error occurred during the experiment: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc() # Print full traceback for debugging
        sys.exit(1)

if __name__ == "__main__":
    main()