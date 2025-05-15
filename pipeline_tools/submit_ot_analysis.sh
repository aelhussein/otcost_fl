#!/bin/bash

# Default values
DEFAULT_DATASETS=("Synthetic_Feature" "Synthetic_Concept" "Credit" "EMNIST" "CIFAR" "ISIC")
DEFAULT_FL_NUM_CLIENTS_PER_DATASET=("2")  # Empty means use config default
DEFAULT_MODEL_TYPES=("best")
DEFAULT_ACTIVATION_LOADERS=("train")
DEFAULT_DIR='/gpfs/commons/groups/gursoy_lab/aelhussein/classes/otcost_fl'
DEFAULT_ENV_PATH='/gpfs/commons/home/aelhussein/anaconda3/bin/activate'
DEFAULT_ENV_NAME='cuda_env_ne1'
DEFAULT_METRIC='score'  # Default metric

# Function to display usage
show_usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  --datasets=<list>             Comma-separated list of datasets (default: ${DEFAULT_DATASETS[*]})"
    echo "  --fl-num-clients=<list>       Comma-separated list of client counts (default: use config default)"
    echo "                                If one value, use for all datasets. If multiple, must match dataset count."
    echo "  --model-types=<list>          Comma-separated list of model types to analyze (default: ${DEFAULT_MODEL_TYPES[*]})"
    echo "  --activation-loaders=<list>   Comma-separated list of loader types (default: ${DEFAULT_ACTIVATION_LOADERS[*]})"
    echo "  --dir=<path>                  Root directory (default: $DEFAULT_DIR)"
    echo "  --env-path=<path>             Environment activation path (default: $DEFAULT_ENV_PATH)"
    echo "  --env-name=<name>             Environment name (default: $DEFAULT_ENV_NAME)"
    echo "  --force-activation-regen      Force regeneration of activation cache"
    echo "  --metric=<str>                Metric to use (default: $DEFAULT_METRIC)"
    echo "  --help                         Show this help message"
}

# Parse named arguments
datasets=() # Initialize as empty arrays
fl_num_clients=()
model_types=()
activation_loaders=()
force_regen=false
metric="$DEFAULT_METRIC"

while [ $# -gt 0 ]; do
    case "$1" in
        --datasets=*)
            IFS=',' read -ra datasets <<< "${1#*=}"
            ;;
        --fl-num-clients=*)
            IFS=',' read -ra fl_num_clients <<< "${1#*=}"
            ;;
        --model-types=*)
            IFS=',' read -ra model_types <<< "${1#*=}"
            ;;
        --activation-loaders=*)
            IFS=',' read -ra activation_loaders <<< "${1#*=}"
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
        --force-activation-regen)
            force_regen=true
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
if [ ${#model_types[@]} -eq 0 ]; then
    model_types=("${DEFAULT_MODEL_TYPES[@]}")
fi
if [ ${#activation_loaders[@]} -eq 0 ]; then
    activation_loaders=("${DEFAULT_ACTIVATION_LOADERS[@]}")
fi
# Use defaults for strings if empty
DIR="${DIR:-$DEFAULT_DIR}"
ENV_PATH="${ENV_PATH:-$DEFAULT_ENV_PATH}"
ENV_NAME="${ENV_NAME:-$DEFAULT_ENV_NAME}"

# Create log directories
mkdir -p logs/outputs_${metric} logs/errors_${metric}

# Echo configuration
echo "Running OT analysis with configuration:"
echo "Datasets: ${datasets[*]}"
echo "FL client counts: ${fl_num_clients[*]}"
echo "Model types: ${model_types[*]}"
echo "Activation loaders: ${activation_loaders[*]}"
echo "Directory: $DIR"
echo "Environment path: $ENV_PATH"
echo "Environment name: $ENV_NAME"
echo "Force activation regeneration: $force_regen"
echo "Metric: $metric"
echo

# Submit jobs
for i in "${!datasets[@]}"; do
    dataset="${datasets[$i]}"
    
    # Determine number of FL clients for this dataset
    num_clients_arg=""
    if [ ${#fl_num_clients[@]} -gt 0 ]; then
        if [ ${#fl_num_clients[@]} -eq 1 ]; then
            # If only one value provided, use it for all datasets
            num_clients_arg="-nc ${fl_num_clients[0]}"
        elif [ $i -lt ${#fl_num_clients[@]} ]; then
            # If multiple values, use the corresponding one
            num_clients_arg="-nc ${fl_num_clients[$i]}"
        fi
    fi
    
    
    for model_type in "${model_types[@]}"; do
        for loader in "${activation_loaders[@]}"; do
            # Additional argument handling
            force_arg=""
            if [ "$force_regen" = true ]; then
                force_arg="-far"
            fi
            
            # Create job name
            clients_suffix=""
            if [ -n "$num_clients_arg" ]; then
                # Extract just the number from the argument
                clients_num="${num_clients_arg#*-nc }"
                clients_suffix="_nc${clients_num}"
            fi
            job_name="OT_${dataset}${clients_suffix}_${model_type}_${loader}_${metric}"
            
            # Create temporary submission script
            cat << EOF > temp_submit_ot_${job_name}.sh
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --partition=cpu
#SBATCH --mail-type=ALL
#SBATCH --mail-user=aelhussein@nygenome.org
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=24:00:00
#SBATCH --output=logs/outputs_${metric}/${job_name}.txt
#SBATCH --error=logs/errors_${metric}/${job_name}.txt

# Activate the environment
source ${ENV_PATH} ${ENV_NAME}

export PYTHONUNBUFFERED=1

# Run the Python script
echo "Running: python ${DIR}/code/ot/run_ot_analysis.py -ds ${dataset} ${num_clients_arg} -mt ${model_type} -al ${loader} ${force_arg} -mc ${metric}"
python ${DIR}/code/ot/run_ot_analysis.py -ds ${dataset} ${num_clients_arg} -mt ${model_type} -al ${loader} ${force_arg} -mc ${metric}

echo "Job finished with exit code \$?"
EOF

            echo "Submitting job: ${job_name}"
            sbatch temp_submit_ot_${job_name}.sh
            rm temp_submit_ot_${job_name}.sh
            sleep 1 # Avoid overwhelming the scheduler
        done
    done
done

echo "All OT analysis jobs submitted."