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
    """Get detailed status information for a dataset with accurate progress reporting"""
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
        errors = 0
        
        # Calculate progress based on experiment type
        if phase == ExperimentType.OT_ANALYSIS:
            # For OT Analysis, we need more detailed progress calculation
            # Each unit of work is: (fl_cost_param, fl_run_idx, client_pair, ot_method_name)
            
            # Get expected run count
            target_fl_runs = dflt.get('runs', 1)
            
            # Calculate client pairs
            num_total_clients = num_clients
            client_ids = [f'client_{i+1}' for i in range(num_total_clients)]
            client_pairs = []
            for i in range(len(client_ids)):
                for j in range(i+1, len(client_ids)):
                    client_pairs.append(f"{client_ids[i]}_vs_{client_ids[j]}")
            
            # Get OT method names from records
            ot_method_names = set()
            for record in records:
                if isinstance(record, dict) and 'ot_method_name' in record:
                    ot_method_names.add(record.get('ot_method_name'))
            
            # Fallback if no methods found in records
            if not ot_method_names:
                try:
                    from ot_configs import all_configs
                    ot_method_names = set([config.name for config in all_configs])
                except ImportError:
                    ot_method_names = {"Direct_Wasserstein", "WC_Direct_Hellinger_4:1"}
            
            # Calculate total expected work units
            total_runs_needed = len(costs) * target_fl_runs * len(client_pairs) * len(ot_method_names)
            
            # Count successfully completed work units
            completed_units = set()
            for record in records:
                if isinstance(record, dict):
                    if record.get('error_message'):
                        errors += 1
                    
                    if record.get('status') == 'Success':
                        cost = record.get('fl_cost_param')
                        run_idx = record.get('fl_run_idx')
                        client_pair = record.get('client_pair')
                        method_name = record.get('ot_method_name')
                        
                        if all(x is not None for x in [cost, run_idx, client_pair, method_name]):
                            key = (cost, run_idx, client_pair, method_name)
                            completed_units.add(key)
            
            total_runs_completed = len(completed_units)
            
            # Calculate progress percentage
            if total_runs_needed > 0:
                progress_pct = round(total_runs_completed / total_runs_needed * 100, 1)
            else:
                progress_pct = 0
                
            # Adjust status symbol if in progress
            if total_runs_completed > 0 and not done:
                status_symbol = "⏳"
                
            # Create progress string
            progress = f"{total_runs_completed}/{total_runs_needed} units ({progress_pct}%)"
            
        else:
            # Original progress calculation for non-OT phases
            # ... [the original progress calculation for other experiment types remains here]
            # Calculate total configurations for this phase
            total_configs = len(costs)
            
            # Track configs with runs and total runs
            configs_with_runs = set()  # Use a set to avoid double counting
            total_runs_completed = 0
            
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
            
            # Total runs needed for all configurations
            total_runs_needed = total_configs * target_runs
            
            # Check each expected configuration against records
            for config_idx, config in enumerate(expected_configs):
                cost, server, param_name, param_value = config
                
                # Create a unique key for this configuration
                config_key = f"{cost}_{server}_{param_name}_{param_value}"
                
                # Count successful runs for this config
                successful_runs = 0
                for r in records:
                    if hasattr(r, 'matches_config') and r.matches_config(cost, server, param_name, param_value) and r.error is None:
                        successful_runs += 1
                        errors += 0 if r.error is None else 1
                
                # Add to completed runs count
                total_runs_completed += successful_runs
                
                # Mark configuration as having at least one run
                if successful_runs > 0:
                    configs_with_runs.add(config_key)
            
            # Calculate progress percentage based on runs completed
            if total_runs_needed > 0:
                progress_pct = round(total_runs_completed / total_runs_needed * 100, 1)
            else:
                progress_pct = 0
                
            # Adjust status symbol if in progress
            if len(records) > 0 and not done:
                status_symbol = "⏳"
                
            # Create progress string to show both configurations and run percentages
            progress = f"{total_runs_completed}/{total_runs_needed} runs ({progress_pct}%)"
            
        # Get timestamp information
        timestamp = get_timestamp_from_metadata(dataset, phase, num_clients)
        formatted_time = get_formatted_timestamp(timestamp)
        
        # Get last job submission info
        job_info = get_job_info(dataset, num_clients, phase, metric)
        job_time = get_formatted_timestamp(job_info.get('timestamp', 'unknown')) if job_info else "never"
        job_id = job_info.get('job_id', 'unknown') if job_info else "unknown"
        
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