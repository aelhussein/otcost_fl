#!/bin/bash

# Default values
#DEFAULT_DATASETS=("Heart" "Synthetic_Label" "Synthetic_Feature" "Synthetic_Concept" "Credit" "EMNIST" "CIFAR"  "ISIC" "IXITiny")
#DEFAULT_DATASETS=("Heart" "Synthetic_Label" "Synthetic_Feature" "Synthetic_Concept" "Credit" "EMNIST" "CIFAR")
#DEFAULT_DATASETS=("Heart" "Synthetic_Label" "Synthetic_Feature" "Synthetic_Concept" "Credit")
DEFAULT_DATASETS=("Synthetic_Label" "Credit")
#DEFAULT_DATASETS=("EMNIST" "CIFAR")
DEFAULT_EXP_TYPES=("learning_rate")
#DEFAULT_EXP_TYPES=("evaluation")
DEFAULT_DIR='/gpfs/commons/groups/gursoy_lab/aelhussein/classes/otcost_fl'
DEFAULT_ENV_PATH='/gpfs/commons/home/aelhussein/anaconda3/bin/activate'
DEFAULT_ENV_NAME='cuda_env_ne1'
DEFAULT_NUM_CLIENTS=""

# Function to display usage
show_usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  --datasets=<list>    Comma-separated list of datasets (default: ${DEFAULT_DATASETS[*]})"
    echo "  --exp-types=<list>   Comma-separated list of experiment types (default: ${DEFAULT_EXP_TYPES[*]})"
    echo "  --dir=<path>         Root directory (default: $DEFAULT_DIR)"
    echo "  --env-path=<path>    Environment activation path (default: $DEFAULT_ENV_PATH)"
    echo "  --env-name=<name>    Environment name (default: $DEFAULT_ENV_NAME)"
    echo "  --num-clients=<int>  Override default number of clients (optional)"
    echo "  --help               Show this help message"
}

# Parse named arguments
datasets=() # Initialize as empty arrays
experiment_types=()
#num_clients_override="" # Initialize specific variable for client override

while [ $# -gt 0 ]; do
    case "$1" in
        --datasets=*)
            IFS=',' read -ra datasets <<< "${1#*=}"
            ;;
        --exp-types=*)
            IFS=',' read -ra experiment_types <<< "${1#*=}"
            ;;
        --dir=*)
            DIR="${1#*=}"
            ;;
        --env-path=*)
            ENV_PATH="${1#*=}"
            ;;
        --env-name=*)
            ENV_NAME="${1#*=}"
            ;;
        --num-clients=*) # MODIFIED: Added num-clients parsing
            num_clients_override="${1#*=}"
            ;;
        --help)
            show_usage
            exit 0
            ;;
        *)
            echo "Unknown parameter: $1"
            show_usage
            exit 1
            ;;
    esac
    shift
done

# Use defaults if arrays are still empty
if [ ${#datasets[@]} -eq 0 ]; then
    datasets=("${DEFAULT_DATASETS[@]}")
fi
if [ ${#experiment_types[@]} -eq 0 ]; then
    experiment_types=("${DEFAULT_EXP_TYPES[@]}")
fi
# Use defaults for strings if empty
DIR="${DIR:-$DEFAULT_DIR}"
ENV_PATH="${ENV_PATH:-$DEFAULT_ENV_PATH}"
ENV_NAME="${ENV_NAME:-$DEFAULT_ENV_NAME}"
# num_clients_override keeps its value or remains empty (default)

# Create log directories
mkdir -p logs/outputs logs/errors

# Echo configuration
echo "Running with configuration:"
echo "Datasets: ${datasets[*]}"
echo "Experiment types: ${experiment_types[*]}"
echo "Directory: $DIR"
echo "Environment path: $ENV_PATH"
echo "Environment name: $ENV_NAME"
# MODIFIED: Echo client override status
if [ -n "$num_clients_override" ]; then
    echo "Client Count Override: $num_clients_override"
    nc_argument="-nc $num_clients_override" # Prepare argument string
else
    echo "Client Count Override: Not specified (using default from config)"
    nc_argument="" # No argument if not specified
fi
echo

# Submit jobs
for dataset in "${datasets[@]}"; do
    # Determine whether the job needs a GPU
    gpu_datasets=("EMNIST" "CIFAR")
    use_gpu=false
    for gpu_ds in "${gpu_datasets[@]}"; do
        if [[ "$dataset" == "$gpu_ds" ]]; then
            use_gpu=true
            break
        fi
    done

    # Set partition and GRES accordingly
    if [ "$use_gpu" = true ]; then
        partition="gpu"
        gres_line="#SBATCH --gres=gpu:1"
    else
        partition="cpu"
        gres_line=""
    fi
    for exp_type in "${experiment_types[@]}"; do
        job_name_suffix="_nc${num_clients_override}"
        job_name="${dataset}_${exp_type}${job_name_suffix}"

        cat << EOF > temp_submit_${job_name}.sh
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --partition=${partition}
${gres_line}
#SBATCH --mail-type=ALL
#SBATCH --mail-user=aelhussein@nygenome.org
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH --time=30:00:00
#SBATCH --output=logs/outputs/${job_name}.txt
#SBATCH --error=logs/errors/${job_name}.txt

# Activate the environment
source ${ENV_PATH} ${ENV_NAME}

export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHON_LOG_DIR="logs/python_logs"

# Run the Python script
# MODIFIED: Added the optional nc_argument
echo "Running: python ${DIR}/code/evaluation/run.py -ds ${dataset} -exp ${exp_type} ${nc_argument}"
python ${DIR}/code/evaluation/run.py -ds ${dataset} -exp ${exp_type} ${nc_argument}

echo "Job finished with exit code \$?"

EOF

        echo "Submitting job: ${job_name}"

        sbatch temp_submit_${job_name}.sh
        rm temp_submit_${job_name}.sh
        sleep 1 # Avoid overwhelming the scheduler
    done
done

echo "All jobs submitted."