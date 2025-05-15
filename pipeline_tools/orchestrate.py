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
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)  # Root directory containing code/ and pipeline_tools/
_CODE_DIR = os.path.join(_PROJECT_ROOT, "code")  # Path to code/ directory
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, _CODE_DIR)  # Add code/ directory specifically to import path


from configs import ROOT_DIR, DATASET_COSTS, DEFAULT_PARAMS
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

def submit_phase(dataset: str, num_clients: int, phase: str, metric: str, 
               dry_run: bool = False) -> Optional[str]:
    """Submit jobs for a specific phase and return job ID if available"""
    script_name, arg_name = PHASE_SUBMIT_INFO[phase]
    
    # Get the full path to the submit script
    script_path = os.path.join(_SCRIPT_DIR, script_name)
    
    # Build command based on script type
    if script_name == "submit_evaluation.sh":
        cmd = [
            "bash", script_path,
            f"--datasets={dataset}",
            f"--exp-types={arg_name}",
            f"--num-clients={num_clients}",
            f"--metric={metric}"
        ]
    else:  # OT analysis has different args
        cmd = [
            "bash", script_path,
            f"--datasets={dataset}",
            f"--fl-num-clients={num_clients}",
            f"--metric={metric}"
        ]
    
    print(f"Command: {' '.join(cmd)}")
    
    if dry_run:
        print("(Dry run - not executing)")
        return None
    
    # Execute the command and try to capture job ID
    try:
        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        # Try to extract Slurm job ID
        job_id = None
        for line in output.splitlines():
            if "Submitted batch job" in line:
                job_id = line.split()[-1].strip()
                break
        
        print(f"Submitted {phase} jobs for {dataset} (clients={num_clients}, metric={metric})")
        if job_id:
            print(f"Submission job ID: {job_id}")
        
        return job_id
    
    except subprocess.CalledProcessError as e:
        print(f"Error submitting jobs: {e.output}")
        return None

def get_progress_info(rm: ResultsManager, phase: str, costs, params) -> Dict[str, Any]:
    """Get progress information for a phase"""
    records, remaining, min_runs = rm.get_experiment_status(
        phase, costs, params, metric_key_cls=None
    )
    
    total_configs = len(costs)
    if phase in [ExperimentType.LEARNING_RATE, ExperimentType.REG_PARAM]:
        # For tuning phases, multiply by number of parameters to try
        param_key = 'learning_rates_try' if phase == ExperimentType.LEARNING_RATE else 'reg_params_try'
        servers_key = 'servers_tune_lr' if phase == ExperimentType.LEARNING_RATE else 'servers_tune_reg'
        params_to_try = len(params.get(param_key, []))
        servers = len(params.get(servers_key, []))
        total_configs *= params_to_try * servers
    elif phase == ExperimentType.EVALUATION:
        # For evaluation, multiply by algorithms
        total_configs *= len(params.get('algorithms', ['local', 'fedavg', 'fedprox', 'pfedme', 'ditto']))
        
    completed_configs = total_configs - len(remaining)
    
    # Calculate error count
    error_count = sum(1 for r in records if getattr(r, "error", None) is not None)
    
    return {
        "total": total_configs,
        "completed": completed_configs,
        "percent": round(completed_configs / total_configs * 100, 1) if total_configs > 0 else 0,
        "errors": error_count,
        "runs_per_config": params.get('runs_tune' if phase != ExperimentType.EVALUATION else 'runs', 1),
        "min_completed_runs": min_runs
    }

def main():
    parser = argparse.ArgumentParser(description="Orchestrate FL experiments in sequence")
    parser.add_argument("-ds", "--dataset", required=True, help="Dataset name")
    parser.add_argument("-nc", "--num_clients", type=int, default=2, help="Number of clients")
    parser.add_argument("--metric", default="score", choices=["score", "loss"], 
                      help="Metric to use (score or loss)")
    parser.add_argument("--force", action="store_true", help="Force rerun of all phases")
    parser.add_argument("--force-phases", type=str, help="Comma-separated list of phases to force (e.g., learning_rate,reg_param)")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    args = parser.parse_args()

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
        
        if not _phase_done(rm, phase, costs, dflt, force=phase_force):
            # Get progress info before submission
            progress_before = get_progress_info(rm, phase, costs, dflt)
            
            # Submit the jobs for this phase
            job_id = submit_phase(args.dataset, args.num_clients, phase, args.metric, args.dry_run)
            
            # Log the submission
            if not args.dry_run:
                log_job_submission(args.dataset, args.num_clients, phase, job_id, args.metric)
                
                # Show progress info
                print(f"\nPhase '{phase}' status:")
                print(f"  Completed: {progress_before['completed']}/{progress_before['total']} configs ({progress_before['percent']}%)")
                print(f"  Minimum runs completed per config: {progress_before['min_completed_runs']}/{progress_before['runs_per_config']}")
                if progress_before['errors'] > 0:
                    print(f"  Warning: {progress_before['errors']} records contain errors")
                print(f"\nJobs have been submitted for incomplete configurations. Run this script again after they complete.")
            
            # Exit after submitting first incomplete phase
            sys.exit(0)
    
    print("All phases complete for dataset:", args.dataset, f"(clients={args.num_clients}, metric={args.metric})")

if __name__ == "__main__":
    main()



