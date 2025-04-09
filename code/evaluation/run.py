import argparse
from pipeline import ExperimentType, ExperimentConfig, Experiment

DATASET_COSTS = {
    'IXITiny': [0.08, 0.28, 0.30, 'all'],
    'ISIC': [0.06, 0.15, 0.19, 0.25, 0.3, 'all'],
    'EMNIST': [0.11, 0.19, 0.25, 0.34, 0.39],
    'CIFAR': [0.08, 0.21, 0.3, 0.38],
    'Synthetic': [0.03, 0.1, 0.2, 0.3, 0.4, 0.5],
    'Credit': [0.12, 0.23, 0.3, 0.4],
    'Weather': [0.11, 0.19, 0.3, 0.4, 0.48]
}

def run_experiments(dataset: str, experiment_type: str):
    """Run experiment based on type"""
    costs = DATASET_COSTS.get(dataset, [])
    config = ExperimentConfig(dataset=dataset, experiment_type=experiment_type)
    experiment = Experiment(config)
    return experiment.run_experiment(costs)

def main():
    parser = argparse.ArgumentParser(description='Run federated learning experiments')
    parser.add_argument("-ds", "--dataset", required=True, 
                      choices=list(DATASET_COSTS.keys()),
                      help="Dataset to use for experiments")
    parser.add_argument("-exp", "--experiment_type", required=True,
                      choices=['learning_rate', 'reg_param', 'evaluation'],
                      help="Type of experiment to run")

    args = parser.parse_args()
    
    try:
        # Map experiment type string to ExperimentType
        type_mapping = {
            'learning_rate': ExperimentType.LEARNING_RATE,
            'reg_param': ExperimentType.REG_PARAM,
            'evaluation': ExperimentType.EVALUATION
        }
        experiment_type = type_mapping[args.experiment_type]
        run_experiments(args.dataset, experiment_type)
        
    except Exception as e:
        print(f"Error running experiments: {str(e)}")
        raise

if __name__ == "__main__":
    main()