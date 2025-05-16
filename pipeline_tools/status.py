#!/usr/bin/env python
"""
Print table of experiment status with progress info, timestamps, and color coding.
"""
import argparse
import os
import sys
import json
from datetime import datetime
import tabulate
from typing import Dict, List, Any
parser = argparse.ArgumentParser(description="Check status of FL experiments")
parser.add_argument("-ds", "--dataset", help="Dataset name (or 'all')")
parser.add_argument("-nc", "--num_clients", type=int, default=2, help="Number of clients")
parser.add_argument("--metric", default="score", choices=["score", "loss"], 
                    help="Metric to use (score or loss)")
parser.add_argument("--all", action="store_true", help="Show status for all datasets")
parser.add_argument("--no-color", action="store_true", help="Disable colored output")
args = parser.parse_args()
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)
from directories import configure, paths
configure(args.metric)
dir_paths = paths()
ROOT_DIR = dir_paths.root_dir

from configs import DATASET_COSTS, DEFAULT_PARAMS
from helper import ExperimentType
from results_manager import ResultsManager

# Define phases in order
PHASES = [
    ExperimentType.LEARNING_RATE,
    ExperimentType.REG_PARAM,
    ExperimentType.EVALUATION,
    ExperimentType.OT_ANALYSIS
]

# ANSI color codes
class Colors:
    RESET = "\033[0m"
    GREEN = "\033[32m"
    RED = "\033[31m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    BOLD = "\033[1m"

# Jobs log file
JOBS_LOG_FILE = os.path.join(_PROJECT_ROOT, "pipeline_tools", "logs", "orchestration_jobs.json")

def get_job_info(dataset: str, num_clients: int, phase: str, metric: str) -> Dict[str, Any]:
    """Get job submission info from the log file"""
    if not os.path.exists(JOBS_LOG_FILE):
        return {}
    
    try:
        with open(JOBS_LOG_FILE, 'r') as f:
            job_data = json.load(f)
            key = f"{dataset}_{num_clients}_{metric}"
            if key in job_data and phase in job_data[key]:
                return job_data[key][phase]
    except (json.JSONDecodeError, KeyError):
        pass
        
    return {}

def get_timestamp_from_metadata(dataset: str, phase: str, num_clients: int) -> str:
    """Extract timestamp from metadata file if available"""
    exp_dir_map = {
        ExperimentType.LEARNING_RATE: "lr_tuning",
        ExperimentType.REG_PARAM: "reg_param_tuning",
        ExperimentType.EVALUATION: "evaluation",
        ExperimentType.OT_ANALYSIS: "ot_analysis"
    }
    
    base_dir = ROOT_DIR
    if "results_loss" in ROOT_DIR:
        results_dir = ROOT_DIR
    else:
        # Check if we might be looking at score or loss results
        score_dir = os.path.join(base_dir, "results")
        loss_dir = os.path.join(base_dir, "results_loss")
        
        # Use the directory that exists
        if os.path.exists(score_dir):
            results_dir = score_dir
        elif os.path.exists(loss_dir):
            results_dir = loss_dir
        else:
            results_dir = os.path.join(base_dir, "results")  # Default
    
    meta_path = os.path.join(
        results_dir, 
        exp_dir_map[phase], 
        f"{dataset}_{num_clients}clients_{exp_dir_map[phase]}_meta.json"
    )
    
    if os.path.exists(meta_path):
        try:
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
                return metadata.get('timestamp', 'unknown')
        except json.JSONDecodeError:
            pass
            
    return "unknown"

def color_status(status: str, errors: int = 0) -> str:
    """Apply color to status string based on the status"""
    if status == "✅":
        return f"{Colors.GREEN}{status}{Colors.RESET}"
    elif status == "❌":
        return f"{Colors.RED}{status}{Colors.RESET}"
    elif status == "⏳":
        return f"{Colors.YELLOW}{status}{Colors.RESET}"
    else:
        return status

def get_formatted_timestamp(timestamp_str: str) -> str:
    """Format timestamp string in a more readable way"""
    if timestamp_str == "unknown":
        return "unknown"
        
    try:
        dt = datetime.fromisoformat(timestamp_str)
        return dt.strftime("%Y-%m-%d %H:%M")
    except (ValueError, TypeError):
        return timestamp_str

def get_dataset_status(dataset: str, num_clients: int, metric: str) -> List[List[Any]]:
    """Get detailed status information for a dataset by checking individual configurations"""
    rm = ResultsManager(ROOT_DIR, dataset, num_clients)
    costs = DATASET_COSTS.get(dataset, [])
    dflt = DEFAULT_PARAMS.get(dataset, {})
    
    rows = []
    for phase in PHASES:
        records, remaining, min_runs = rm.get_experiment_status(
            phase, costs, dflt, metric_key_cls=None
        )
        
        # Calculate basic status
        done = len(remaining) == 0
        status_symbol = "✅" if done else "❌"
        
        # Count errors
        errors = sum(1 for r in records if getattr(r, "error", None) is not None)
        
        # Calculate total configurations for this phase
        total_configs = len(costs)
        completed_configs = 0
        
        # Generate list of all expected configurations
        expected_configs = []
        
        if phase in [ExperimentType.LEARNING_RATE, ExperimentType.REG_PARAM]:
            # For tuning phases, multiply by number of parameters to try
            param_key = 'learning_rates_try' if phase == ExperimentType.LEARNING_RATE else 'reg_params_try'
            param_name = 'learning_rate' if phase == ExperimentType.LEARNING_RATE else 'reg_param' 
            servers_key = 'servers_tune_lr' if phase == ExperimentType.LEARNING_RATE else 'servers_tune_reg'
            params_to_try = dflt.get(param_key, [])
            servers_to_try = dflt.get(servers_key, [])
            
            # Calculate total configurations
            total_configs *= len(params_to_try) * len(servers_to_try)
            
            # Generate all expected configurations
            for cost in costs:
                for server in servers_to_try:
                    for param_val in params_to_try:
                        expected_configs.append((cost, server, param_name, param_val))
                        
        elif phase == ExperimentType.EVALUATION:
            # For evaluation, multiply by algorithms
            algorithms = dflt.get('algorithms', ['local', 'fedavg', 'fedprox', 'pfedme', 'ditto'])
            total_configs *= len(algorithms)
            
            # Generate all expected configurations
            for cost in costs:
                for algorithm in algorithms:
                    expected_configs.append((cost, algorithm, None, None))
        else:
            # For other phases, just use costs
            for cost in costs:
                expected_configs.append((cost, None, None, None))
        
        # Determine target number of runs for this phase
        target_runs_key = 'runs_tune' if phase in [ExperimentType.LEARNING_RATE, ExperimentType.REG_PARAM] else 'runs'
        target_runs = dflt.get(target_runs_key, 1)
        
        # Check each configuration against records
        for config in expected_configs:
            cost, server, param_name, param_value = config
            
            # Count successful runs for this config
            successful_runs = 0
            for r in records:
                if hasattr(r, 'matches_config') and r.matches_config(cost, server, param_name, param_value) and r.error is None:
                    successful_runs += 1
            
            # Mark configuration as complete if it has enough runs
            if successful_runs >= target_runs:
                completed_configs += 1
            
        # Calculate progress percentage
        if total_configs > 0:
            progress_pct = round(completed_configs / total_configs * 100, 1)
        else:
            progress_pct = 0
            
        # Adjust status symbol if in progress
        if len(records) > 0 and not done:
            status_symbol = "⏳"
            
        # Get timestamp information
        timestamp = get_timestamp_from_metadata(dataset, phase, num_clients)
        formatted_time = get_formatted_timestamp(timestamp)
        
        # Get last job submission info
        job_info = get_job_info(dataset, num_clients, phase, metric)
        job_time = get_formatted_timestamp(job_info.get('timestamp', 'unknown')) if job_info else "never"
        job_id = job_info.get('job_id', 'unknown') if job_info else "unknown"
        
        # Create the progress string
        progress = f"{completed_configs}/{total_configs} ({progress_pct}%)"
        
        # Create row with color-coded status
        row = [
            phase,
            color_status(status_symbol, errors),
            progress,
            errors,
            formatted_time,
            job_time,
            ",".join(map(str, remaining[:5])) + ("..." if len(remaining) > 5 else "")
        ]
        
        rows.append(row)
        
    return rows

def main():
    # Disable colors if requested
    if args.no_color:
        for attr in dir(Colors):
            if attr.isupper():
                setattr(Colors, attr, "")
    
    # Determine which datasets to check
    datasets = []
    if args.all or args.dataset == "all":
        datasets = list(DATASET_COSTS.keys())
    elif args.dataset:
        if args.dataset in DATASET_COSTS:
            datasets = [args.dataset]
        else:
            print(f"Error: Dataset '{args.dataset}' not found in DATASET_COSTS.")
            return
    else:
        print("Error: Please specify a dataset with -ds or use --all to show all datasets.")
        return
    
    # Table headers
    headers = ["Phase", "Done", "Progress", "Errors", "Last Update", "Last Submission", "Missing Costs"]
    
    for dataset in datasets:
        print(f"\n{Colors.BOLD}Status for {dataset} (clients={args.num_clients}, metric={args.metric}){Colors.RESET}")
        print("-" * 80)
        
        rows = get_dataset_status(dataset, args.num_clients, args.metric)
        
        # Print the table
        print(tabulate.tabulate(rows, headers=headers, tablefmt="grid"))
        
        # Print summary for this dataset
        total_errors = sum(row[3] for row in rows)
        completed_phases = sum(1 for row in rows if "✅" in row[1])
        if total_errors > 0:
            print(f"{Colors.RED}⚠️ {total_errors} errors detected!{Colors.RESET}")
        
        print(f"Summary: {completed_phases}/{len(PHASES)} phases complete.")

if __name__ == "__main__":
    main()