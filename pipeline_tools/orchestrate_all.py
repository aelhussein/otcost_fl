#!/usr/bin/env python
"""
Run orchestration for multiple datasets and client counts.
"""
import argparse
import subprocess
import sys
import os
from typing import List, Dict, Any
import tabulate
import time

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)  # Root directory containing code/ and pipeline_tools/
_CODE_DIR = os.path.join(_PROJECT_ROOT, "code")  # Path to code/ directory
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, _CODE_DIR)  # Add code/ directory specifically to import path


from configs import ROOT_DIR, DATASET_COSTS, DEFAULT_PARAMS
from helper import ExperimentType
from results_manager import ResultsManager

def run_orchestrator(dataset: str, num_clients: int, metric: str, 
                   force: bool = False, force_phases: str = None, 
                   dry_run: bool = False) -> bool:
    """Run orchestrate.py for a dataset/client pair"""
    orchestrate_script = os.path.join(_SCRIPT_DIR, "orchestrate.py")
    
    cmd = [
        "python", orchestrate_script,
        "-ds", dataset,
        "-nc", str(num_clients),
        "--metric", metric
    ]
    
    if force:
        cmd.append("--force")
    
    if force_phases:
        cmd.append(f"--force-phases={force_phases}")
    
    if dry_run:
        cmd.append("--dry-run")
    
    print(f"Running: {' '.join(cmd)}")
    
    try:
        subprocess.check_call(cmd)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running orchestrator for {dataset} (clients={num_clients}): {e}")
        return False

def get_summary_status(datasets: List[str], num_clients_list: List[int], metric: str) -> Dict[str, Dict[str, Any]]:
    """Get summary status for all dataset/client combinations"""
    summary = {}
    
    for dataset in datasets:
        summary[dataset] = {}
        for num_clients in num_clients_list:
            rm = ResultsManager(ROOT_DIR, dataset, num_clients)
            costs = DATASET_COSTS.get(dataset, [])
            dflt = DEFAULT_PARAMS.get(dataset, {})
            
            status = {}
            for phase in [ExperimentType.LEARNING_RATE, ExperimentType.REG_PARAM, 
                         ExperimentType.EVALUATION, ExperimentType.OT_ANALYSIS]:
                records, remaining, min_runs = rm.get_experiment_status(
                    phase, costs, dflt, metric_key_cls=None
                )
                
                # Calculate completion
                done = len(remaining) == 0
                
                # Count errors
                errors = sum(1 for r in records if getattr(r, "error", None) is not None)
                
                status[phase] = {
                    "done": done,
                    "records": len(records),
                    "errors": errors,
                    "remaining": len(remaining)
                }
            
            summary[dataset][num_clients] = status
    
    return summary

def print_summary_table(summary: Dict[str, Dict[str, Any]]):
    """Print summary table of experiment status"""
    phases = [ExperimentType.LEARNING_RATE, ExperimentType.REG_PARAM, 
             ExperimentType.EVALUATION, ExperimentType.OT_ANALYSIS]
    
    headers = ["Dataset", "Clients"] + [phase for phase in phases] + ["All Complete"]
    
    rows = []
    for dataset, client_data in summary.items():
        for num_clients, phase_data in client_data.items():
            # Create status symbols for each phase
            phase_statuses = []
            all_complete = True
            
            for phase in phases:
                if phase_data[phase]["done"]:
                    if phase_data[phase]["errors"] > 0:
                        phase_statuses.append("✅*")  # Done but with errors
                    else:
                        phase_statuses.append("✅")  # Done with no errors
                else:
                    all_complete = False
                    if phase_data[phase]["records"] > 0:
                        phase_statuses.append("⏳")  # In progress
                    else:
                        phase_statuses.append("❌")  # Not started
            
            # Create the row
            row = [dataset, num_clients] + phase_statuses + ["✅" if all_complete else "❌"]
            rows.append(row)
    
    print(tabulate.tabulate(rows, headers=headers, tablefmt="grid"))
    print("✅ = Complete, ⏳ = In Progress, ❌ = Not Started, * = Has Errors")

def main():
    parser = argparse.ArgumentParser(description="Orchestrate FL experiments for multiple datasets")
    parser.add_argument("--datasets", required=True, help="Comma-separated list of datasets or 'all'")
    parser.add_argument("--num-clients", type=str, default="2", 
                    help="Comma-separated list of client counts, or single value for all datasets")
    parser.add_argument("--metric", default="score", choices=["score", "loss"], 
                    help="Metric to use (score or loss)")
    parser.add_argument("--force", action="store_true", help="Force rerun of all phases")
    parser.add_argument("--force-phases", type=str, 
                    help="Comma-separated list of phases to force. Valid phases: learning_rate, reg_param, evaluation, ot_analysis")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    parser.add_argument("--summary-only", action="store_true", 
                    help="Only print summary status without running orchestrator")
    args = parser.parse_args()
    
    # Parse datasets
    if args.datasets.lower() == "all":
        datasets = list(DATASET_COSTS.keys())
    else:
        datasets = [ds.strip() for ds in args.datasets.split(",")]
        # Validate datasets
        for ds in datasets:
            if ds not in DATASET_COSTS:
                print(f"Error: Dataset '{ds}' not found in DATASET_COSTS.")
                sys.exit(1)
    
    # Parse client counts
    try:
        if "," in args.num_clients:
            num_clients_list = [int(nc.strip()) for nc in args.num_clients.split(",")]
        else:
            num_clients_list = [int(args.num_clients)]
    except ValueError:
        print("Error: Invalid num-clients value. Please use integers separated by commas.")
        sys.exit(1)
    
    # Print initial summary if requested
    if args.summary_only:
        print("Experiment Status Summary:")
        summary = get_summary_status(datasets, num_clients_list, args.metric)
        print_summary_table(summary)
        sys.exit(0)
    
    # Run orchestrator for each dataset/client combination
    success_count = 0
    combinations = []
    
    for dataset in datasets:
        for num_clients in num_clients_list:
            combinations.append((dataset, num_clients))
    
    print(f"Will process {len(combinations)} dataset/client combinations")
    
    for i, (dataset, num_clients) in enumerate(combinations):
        print(f"\n[{i+1}/{len(combinations)}] Processing {dataset} with {num_clients} clients")
        
        success = run_orchestrator(
            dataset, 
            num_clients, 
            args.metric,
            args.force, 
            args.force_phases, 
            args.dry_run
        )
        
        if success:
            success_count += 1
        
        # Add a small delay between submissions to avoid overwhelming the scheduler
        if i < len(combinations) - 1 and not args.dry_run:
            time.sleep(1)
    
    print(f"\nCompleted {success_count}/{len(combinations)} dataset/client combinations")
    
    # Print final summary
    print("\nFinal Status Summary:")
    summary = get_summary_status(datasets, num_clients_list, args.metric)
    print_summary_table(summary)

if __name__ == "__main__":
    main()