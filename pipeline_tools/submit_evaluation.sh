#!/bin/bash

# Default values
#DEFAULT_DATASETS=("Synthetic_Feature" "Synthetic_Concept" "Credit" "EMNIST" "CIFAR" "ISIC" "IXITiny")
#DEFAULT_DATASETS=("Synthetic_Feature" "Synthetic_Concept" "Credit" "EMNIST")
DEFAULT_DATASETS=("Synthetic_Feature" "Synthetic_Concept" "Credit")
#DEFAULT_EXP_TYPES=("learning_rate")
DEFAULT_EXP_TYPES=("reg_param")
#DEFAULT_EXP_TYPES=("evaluation")
DEFAULT_DIR='/gpfs/commons/groups/gursoy_lab/aelhussein/classes/otcost_fl'
DEFAULT_ENV_PATH='/gpfs/commons/home/aelhussein/anaconda3/bin/activate'
DEFAULT_ENV_NAME='cuda_env_ne1'
DEFAULT_NUM_CLIENTS="2"
DEFAULT_METRIC="score"

# Function to display usage
show_usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  --datasets=<list>    Comma-separated list of datasets (default: ${DEFAULT_DATASETS[*]})"
    echo "  --exp-types=<list>   Comma-separated list of experiment types (default: ${DEFAULT_EXP_TYPES[*]})"
    echo "  --dir=<path>         Root directory (default: $DEFAULT_DIR)"
    echo "  --env-path=<path>    Environment activation path (default: $DEFAULT_ENV_PATH)"
    echo "  --env-name=<name>    Environment name (default: $DEFAULT_ENV_NAME)"
    echo "  --num-clients=<int>  Number of clients (default: $DEFAULT_NUM_CLIENTS)"
    echo "  --metric=<str>       Metric to use (default: $DEFAULT_METRIC)"
    echo "  --help               Show this help message"
}

# Parse named arguments
datasets=() # Initialize as empty arrays
experiment_types=()
num_clients="$DEFAULT_NUM_CLIENTS"
metric="$DEFAULT_METRIC"

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
        --num-clients=*)
            num_clients="${1#*=}"
            ;;
        --metric=*)
            metric="${1#*=}"
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

# Create log directories
mkdir -p ${DIR}/logs/outputs_${metric}  ${DIR}/logs/errors_${metric}

# Echo configuration
echo "Running with configuration:"
echo "Datasets: ${datasets[*]}"
echo "Experiment types: ${experiment_types[*]}"
echo "Directory: $DIR"
echo "Environment path: $ENV_PATH"
echo "Environment name: $ENV_NAME"
echo "Number of clients: $num_clients"
echo "Metric: $metric"

# Submit jobs
for dataset in "${datasets[@]}"; do
    # Determine whether the job needs a GPU
    gpu_datasets=("EMNIST" "CIFAR" "ISIC" "IXITiny")
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
        job_name_suffix="_nc${num_clients}_${metric}"
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
#SBATCH --time=40:00:00
#SBATCH --output=${DIR}/logs/outputs_${metric}/${job_name}.txt
#SBATCH --error=${DIR}/logs/errors_${metric}/${job_name}.txt
#SBATCH --exclude=ne1dg6-004
# Activate the environment
source ${ENV_PATH} ${ENV_NAME}

export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-6}  
export MKL_NUM_THREADS=$OMP_NUM_THREADS


# Run the Python script
echo "Running: python ${DIR}/code/run_evaluation.py -ds ${dataset} -exp ${exp_type} -nc ${num_clients} -mc ${metric}"
python ${DIR}/code/run_evaluation.py -ds ${dataset} -exp ${exp_type} -nc ${num_clients} -mc ${metric}

echo "Job finished with exit code \$?"

EOF

        echo "Submitting job: ${job_name}"

        sbatch temp_submit_${job_name}.sh
        rm temp_submit_${job_name}.sh
        sleep 1 # Avoid overwhelming the scheduler
    done
done

echo "All jobs submitted."