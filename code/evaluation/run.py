
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

DATASET_COSTS = {
    'IXITiny': [0.08, 0.28, 0.30, 'all'],
    'ISIC': [0.06, 0.15, 0.19, 0.25, 0.3, 'all'],
    'EMNIST': [0.11, 0.19, 0.25, 0.34, 0.39],
    'CIFAR': [0.08, 0.21, 0.3, 0.38],
    'Synthetic': [0.03, 0.1, 0.2, 0.3, 0.4, 0.5],
    'Credit': [0.12, 0.16, 0.22, 0.23, 0.27, 0.3, 0.34, 0.4],
    'Weather': [0.11, 0.19, 0.3, 0.4, 0.48],
    'Heart': [1,2,3,4,5,6]
}

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__)) 
_PROJECT_ROOT = os.path.dirname(_CURRENT_DIR)
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, _CURRENT_DIR)

# --- Import Project Modules ---
from configs import DATASETS, RESULTS_DIR
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
    # Execute the experiment
    costs = DATASET_COSTS.get(dataset, [])
    results = experiment_runner.run_experiment(costs)
    print(f"Experiment completed: Dataset='{dataset}', Type='{experiment_type}'")
    return results

def main():
    """
    Parses command-line arguments and initiates the experiment run.
    """
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(
        description='Run Federated Learning experiments for Layer-PFL project.'
    )
    parser.add_argument(
        "-ds", "--dataset",
        required=True,
        choices=list(DATASETS), # Use dataset list from configs
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
        os.makedirs(os.path.join(RESULTS_DIR, 'evaluation'), exist_ok=True)
        os.makedirs(os.path.join(RESULTS_DIR, 'lr_tuning'), exist_ok=True)
        # Add other directories if needed (e.g., for regularization tuning)
        print(f"Results will be saved in subdirectories under: {RESULTS_DIR}")
    except OSError as e:
        print(f"Error creating results directories: {e}", file=sys.stderr)
        sys.exit(1)


    # --- Run Experiment ---
    try:
        run_experiments(args.dataset, args.experiment_type)
    except ValueError as ve:
        print(f"Configuration Error: {ve}", file=sys.stderr)
        sys.exit(1)
    except NotImplementedError as nie:
        print(f"Functionality Error: {nie}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        # Catch any other unexpected errors during the experiment run
        print(f"An unexpected error occurred during the experiment: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc() # Print full traceback for debugging
        sys.exit(1)

if __name__ == "__main__":
    main()