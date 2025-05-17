#!/usr/bin/env python
"""
Drive LR → Reg → Eval → OT for one dataset/num-clients pair.
Skips phases that are already complete, using ResultsManager.get_experiment_status.
"""
import argparse
import subprocess
import sys
import os
import time
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)
from directories import configure, paths

# ARGS
parser = argparse.ArgumentParser(description="Orchestrate FL experiments in sequence")
parser.add_argument("-ds", "--dataset", required=True, help="Dataset name")
parser.add_argument("-nc", "--num_clients", type=int, default=2, help="Number of clients")
parser.add_argument("--metric", default="score", choices=["score", "loss"], 
                    help="Metric to use (score or loss)")
parser.add_argument("--force", action="store_true", help="Force rerun of all phases")
parser.add_argument("--force-phases", type=str, 
                    help="Comma-separated list of phases to force (e.g., learning_rate,reg_param,evaluation,ot_analysis)")
parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
args = parser.parse_args()
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

# Mapping from phase to submit script name and arg name
PHASE_SUBMIT_INFO = {
    ExperimentType.LEARNING_RATE: ("submit_evaluation.sh", "learning_rate"),
    ExperimentType.REG_PARAM: ("submit_evaluation.sh", "reg_param"),
    ExperimentType.EVALUATION: ("submit_evaluation.sh", "evaluation"),
    ExperimentType.OT_ANALYSIS: ("submit_ot_analysis.sh", None)  # Special case handling
}

# File to track job submissions
JOBS_LOG_FILE = os.path.join(_PROJECT_ROOT, "pipeline_tools", "logs", "orchestration_jobs.json")

def _phase_done(rm: ResultsManager, phase: str, costs, params, force: bool = False) -> bool:
    """Check if a phase is complete using ResultsManager"""
    if force:
        return False
    
    # Re-use the completeness logic from ResultsManager
    _, remaining, min_runs = rm.get_experiment_status(
        phase, costs, params, metric_key_cls=None
    )
    return len(remaining) == 0

def log_job_submission(dataset: str, num_clients: int, phase: str, 
                      job_id: Optional[str], metric: str) -> None:
    """Log job submission to a JSON file"""
    os.makedirs(os.path.dirname(JOBS_LOG_FILE), exist_ok=True)
    
    # Load existing job log or create new one
    job_data = {}
    if os.path.exists(JOBS_LOG_FILE):
        try:
            with open(JOBS_LOG_FILE, 'r') as f:
                job_data = json.load(f)
        except json.JSONDecodeError:
            # File exists but is invalid JSON, start fresh
            job_data = {}
    
    # Create key for this dataset/client/metric combination
    key = f"{dataset}_{num_clients}_{metric}"
    if key not in job_data:
        job_data[key] = {}
    
    # Log the job submission
    job_data[key][phase] = {
        "timestamp": datetime.now().isoformat(),
        "job_id": job_id,
        "status": "submitted"
    }
    
    # Save updated job log
    with open(JOBS_LOG_FILE, 'w') as f:
        json.dump(job_data, f, indent=2)

def submit_phase(dataset: str,
                 num_clients: int,
                 phase: str,
                 metric: str,
                 dry_run: bool = False) -> List[str]:
    """
    Submit the Slurm jobs for one phase and return **all** Slurm job-IDs
    printed by the submit script (one ID per “Submitted batch job …” line).
    """
    script_name, arg_name = PHASE_SUBMIT_INFO[phase]
    script_path = os.path.join(_SCRIPT_DIR, script_name)

    if script_name == "submit_evaluation.sh":
        cmd = [
            "bash", script_path,
            f"--datasets={dataset}",
            f"--exp-types={arg_name}",
            f"--num-clients={num_clients}",
            f"--metric={metric}",
        ]
    else:  # OT analysis
        cmd = [
            "bash", script_path,
            f"--datasets={dataset}",
            f"--fl-num-clients={num_clients}",
            f"--metric={metric}",
        ]

    print("Command:", " ".join(cmd))
    if dry_run:
        print("(dry-run)")
        return []

    try:
        output = subprocess.check_output(cmd,
                                         stderr=subprocess.STDOUT,
                                         text=True)
    except subprocess.CalledProcessError as e:
        print("ERROR submitting jobs:\n", e.output)
        return []

    # Extract every job-ID mentioned by sbatch
    job_ids: List[str] = [
        line.split()[-1].strip()
        for line in output.splitlines()
        if "Submitted batch job" in line
    ]

    if job_ids:
        print(f"→ {phase}: submitted {len(job_ids)} job(s): {','.join(job_ids)}")
    else:
        print(f"WARNING: no job-ID detected for {phase}")

    return job_ids


def archive_phase_results(dataset: str, num_clients: int, phase: str, metric: str):
    """Archives existing result files for a phase to ensure complete rerun."""
    from datetime import datetime
    import os
    import shutil
    
    # Set results directory based on metric
    results_base = os.path.join(ROOT_DIR, 'results' if metric == 'score' else 'results_loss')
    
    # Map phase to directory name
    exp_dir_map = {
        ExperimentType.LEARNING_RATE: "lr_tuning",
        ExperimentType.REG_PARAM: "reg_param_tuning",
        ExperimentType.EVALUATION: "evaluation",
        ExperimentType.OT_ANALYSIS: "ot_analysis"
    }
    
    phase_dir = exp_dir_map[phase]
    
    # Define base filenames for results and metadata
    base_filename = f"{dataset}_{num_clients}clients_{phase_dir}"
    results_path = os.path.join(results_base, phase_dir, f"{base_filename}_results.json")
    meta_path = os.path.join(results_base, phase_dir, f"{base_filename}_meta.json")
    
    # Create archive directory
    archive_dir = os.path.join(results_base, phase_dir, 'archive')
    os.makedirs(archive_dir, exist_ok=True)
    
    # Generate timestamp for archive filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Move results file if it exists
    if os.path.exists(results_path):
        archive_results = os.path.join(archive_dir, f"{base_filename}_results_{timestamp}.json")
        shutil.move(results_path, archive_results)
        print(f"Archived results to: {archive_results}")
    
    # Move metadata file if it exists
    if os.path.exists(meta_path):
        archive_meta = os.path.join(archive_dir, f"{base_filename}_meta_{timestamp}.json")
        shutil.move(meta_path, archive_meta)
        print(f"Archived metadata to: {archive_meta}")

def get_progress_info(rm: ResultsManager, phase: str, costs, params) -> Dict[str, Any]:
    """Get accurate progress information for a phase"""
    records, remaining, min_runs = rm.get_experiment_status(
        phase, costs, params, metric_key_cls=None
    )
    
    # Calculate the total number of expected configurations
    total_configs = len(costs)
    completed_configs = 0  # Fully completed configs
    
    # Track configs with runs to avoid double counting
    configs_with_runs = set()
    
    # List to store all expected configurations for detailed checking
    expected_configs = []
    
    if phase in [ExperimentType.LEARNING_RATE, ExperimentType.REG_PARAM]:
        # For tuning phases, get parameters to try
        param_key = 'learning_rates_try' if phase == ExperimentType.LEARNING_RATE else 'reg_params_try'
        param_name = 'learning_rate' if phase == ExperimentType.LEARNING_RATE else 'reg_param'
        servers_key = 'servers_tune_lr' if phase == ExperimentType.LEARNING_RATE else 'servers_tune_reg'
        params_to_try = params.get(param_key, [])
        servers_to_try = params.get(servers_key, [])
        
        # Total configurations = costs * servers * parameters
        total_configs *= len(params_to_try) * len(servers_to_try)
        
        # Generate all expected configurations
        for cost in costs:
            for server in servers_to_try:
                for param_val in params_to_try:
                    expected_configs.append((cost, server, param_name, param_val))
                    
    elif phase == ExperimentType.EVALUATION:
        # For evaluation, multiply by algorithms
        algorithms = params.get('algorithms', ['local', 'fedavg', 'fedprox', 'pfedme', 'ditto'])
        total_configs *= len(algorithms)
        
        # Generate all expected configurations
        for cost in costs:
            for algorithm in algorithms:
                expected_configs.append((cost, algorithm, None, None))
    else:
        # For other phases, just use costs
        for cost in costs:
            expected_configs.append((cost, None, None, None))
    
    # Determine target number of runs based on experiment type
    target_runs = params.get('runs_tune' if phase != ExperimentType.EVALUATION else 'runs', 1)
    
    # Count total runs across all configurations
    total_runs_completed = 0
    total_runs_needed = total_configs * target_runs
    
    # Check each expected configuration against records
    for config in expected_configs:
        cost, server, param_name, param_value = config
        
        # Create a unique key for this configuration
        config_key = f"{cost}_{server}_{param_name}_{param_value}"
        
        # Count successful runs for this config
        successful_runs = 0
        for r in records:
            # Use matches_config for standard TrialRecord
            if hasattr(r, 'matches_config') and r.matches_config(cost, server, param_name, param_value) and r.error is None:
                successful_runs += 1
        
        # Add to total runs completed
        total_runs_completed += successful_runs
        
        # Mark configuration as having at least one run
        if successful_runs > 0:
            configs_with_runs.add(config_key)
            
        # Mark configuration as complete if it has enough runs
        if successful_runs >= target_runs:
            completed_configs += 1
    
    # Calculate error count
    error_count = sum(1 for r in records if getattr(r, "error", None) is not None)
    
    # Calculate percentage based on runs completed
    if total_runs_needed > 0:
        progress_percent = round(total_runs_completed / total_runs_needed * 100, 1)
    else:
        progress_percent = 0
    
    return {
        "total": total_configs,
        "configs_started": len(configs_with_runs),  # Number of configs with at least one run
        "configs_completed": completed_configs,  # Fully completed configs
        "runs_completed": total_runs_completed,  # Total individual runs completed
        "runs_needed": total_runs_needed,  # Total runs needed
        "percent": progress_percent,
        "errors": error_count,
        "runs_per_config": target_runs,
        "min_completed_runs": min_runs
    }

def schedule_next_orchestrator(dataset: str,
                               num_clients: int,
                               metric: str,
                               force: bool,
                               force_phases: List[str],
                               dependency_ids: List[str]) -> None:
    """
    Submit *this* script again so that it runs only after all
    `dependency_ids` finish successfully.
    """
    if not dependency_ids:                      # nothing to depend on
        return

    dep_str = ":".join(dependency_ids)
    job_name = f"orch_{dataset}_{num_clients}_{metric}"

    # Re-create the CLI we were launched with
    args_list = [
        f"python {_SCRIPT_DIR}/orchestrate.py",
        f"-ds {dataset}",
        f"-nc {num_clients}",
        f"--metric {metric}",
    ]
    if force:
        args_list.append("--force")
    if force_phases:
        args_list.append(f"--force-phases={','.join(force_phases)}")

    wrap_cmd = " ".join(args_list)
    sbatch_cmd = [
        "sbatch",
        f"--dependency=afterok:{dep_str}",
        "--job-name", job_name,
        "--output", "/dev/null",
        "--error", "/dev/null",
        "--wrap", wrap_cmd,
    ]

    try:
        new_job = subprocess.check_output(sbatch_cmd,
                                          stderr=subprocess.STDOUT,
                                          text=True).strip()
        print(f"Chained orchestrator submitted (job-ID {new_job}) "
              f"→ will start after {dep_str}")
    except subprocess.CalledProcessError as e:
        print("ERROR chaining orchestrator:\n", e.output)


def main():
    # Validate dataset
    if args.dataset not in DATASET_COSTS:
        print(f"Error: Dataset '{args.dataset}' not found in DATASET_COSTS.")
        sys.exit(1)
    
    costs = DATASET_COSTS[args.dataset]
    dflt = DEFAULT_PARAMS[args.dataset]
    rm = ResultsManager(ROOT_DIR, args.dataset, args.num_clients)
    
    # Parse force phases if provided
    force_phases = []
    if args.force_phases:
        force_phases = [phase.strip() for phase in args.force_phases.split(",")]
    
    # Process each phase in order
    for phase in PHASES:
        # Determine if this phase should be forced
        phase_force = args.force or phase in force_phases
        
        # When forcing a phase, archive existing results first (if not dry run)
        if phase_force and not args.dry_run:
            print(f"Forcing phase '{phase}', archiving existing results...")
            archive_phase_results(args.dataset, args.num_clients, phase, args.metric)
        
        if not _phase_done(rm, phase, costs, dflt, force=False):  # Always pass force=False as we handle forcing via archiving
            # Get progress info before submission
            progress_before = get_progress_info(rm, phase, costs, dflt)
            
            # Submit the jobs for this phase
            job_id = submit_phase(args.dataset, args.num_clients, phase, args.metric, args.dry_run)
            
            # Log the submission
            if not args.dry_run:
                log_job_submission(args.dataset, args.num_clients, phase, job_id, args.metric)
                
                # Show progress info with enhanced details
                print(f"\nPhase '{phase}' status:")
                
                # Show more accurate configuration progress
                
                runs_completed = progress_before['runs_completed']
                runs_needed = progress_before['runs_needed']
                
                # Display clear progress information
                print(f"  Runs: {runs_completed}/{runs_needed} completed ({progress_before['percent']}%)")
                
                if progress_before['errors'] > 0:
                    print(f"  Warning: {progress_before['errors']} records contain errors")
                    
                print(f"\nJobs have been submitted for incomplete configurations. Run this script again after they complete.")
            
            # Exit after submitting first incomplete phase
            sys.exit(0)
    
    print("All phases complete for dataset:", args.dataset, f"(clients={args.num_clients}, metric={args.metric})")


if __name__ == "__main__":
    main()



