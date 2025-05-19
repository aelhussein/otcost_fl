"""
Utility functions for analyzing and visualizing both FL and OT results.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from typing import List, Dict, Tuple, Any, Optional, Union
from scipy import stats

# We need to import the necessary classes and constants
# These imports should be adjusted based on your actual project structure
try:
    from results_manager import ResultsManager, TrialRecord,  OTAnalysisRecord
    from helper import MetricKey, ExperimentType, infer_higher_is_better
except ImportError:
    # For fallback or testing - these might need to be adjusted
    pass


# =============================================================================
# == FL Results Aggregation Functions ==
# =============================================================================

def aggregate_train_val(records: List[TrialRecord], min_rounds: int, conf: float = 0.95) -> Optional[Dict[str, np.ndarray]]:
    """
    Aggregate train/val curves for records truncated to min_rounds using bootstrap CI.
    
    Args:
        records: List of TrialRecords from FL experiments
        min_rounds: Minimum number of rounds to include
        conf: Confidence level for intervals
        
    Returns:
        Dictionary with aggregated train/val statistics or None if insufficient data
    """
    if not records:
        return None

    trains, vals = [], []
    for r in records:
        tr = r.metrics.get(MetricKey.TRAIN_LOSSES, [])
        va = r.metrics.get(MetricKey.VAL_LOSSES, [])
        if len(tr) >= min_rounds and len(va) >= min_rounds:
            trains.append(tr[:min_rounds])
            vals.append(va[:min_rounds])

    if not trains:
        return None

    trains = np.array(trains)
    vals = np.array(vals)

    mean_tr = trains.mean(0)
    lower_tr, upper_tr = [], []
    mean_va = vals.mean(0)
    lower_va, upper_va = [], []

    for arr, lower, upper in [(trains, lower_tr, upper_tr), (vals, lower_va, upper_va)]:
        for t in range(min_rounds):
            m, lo, hi = mean_ci_bootstrap(arr[:, t], conf)
            lower.append(lo)
            upper.append(hi)

    return {
        "num_runs": trains.shape[0],
        "mean_train": mean_tr, 
        "lower_train": np.array(lower_tr), 
        "upper_train": np.array(upper_tr),
        "mean_val": mean_va, 
        "lower_val": np.array(lower_va), 
        "upper_val": np.array(upper_va),
    }

def aggregate_test(records: List[TrialRecord], conf: float = 0.95) -> Optional[Dict[str, float]]:
    """
    Aggregate final-test losses & scores for FL records.
    
    Args:
        records: List of TrialRecords from FL experiments
        conf: Confidence level for intervals
        
    Returns:
        Dictionary with aggregated test statistics or None if insufficient data
    """
    if not records:
        return None
        
    losses = np.array([r.metrics.get(MetricKey.TEST_LOSSES, [np.nan])[-1] for r in records])
    scores = np.array([r.metrics.get(MetricKey.TEST_SCORES, [np.nan])[-1] for r in records])

    if np.isnan(losses).all() or np.isnan(scores).all():
        return None

    m_loss, lo_loss, hi_loss = mean_ci_bootstrap(losses, conf)
    m_score, lo_score, hi_score = mean_ci_bootstrap(scores, conf)
    
    return {
        "num_runs": len(records),
        "mean_loss": m_loss, "lower_loss": lo_loss, "upper_loss": hi_loss,
        "mean_score": m_score, "lower_score": lo_score, "upper_score": hi_score,
    }

def plot_losses_per_cost(
    results_manager: ResultsManager,
    target_costs: List[Any],
    confidence_level: float = 0.95,
    start_round: int = 1,
    plot_losses: bool = False,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create a composite figure of curves + bar plots for target_costs.
    
    Args:
        results_manager: ResultsManager instance
        target_costs: List of cost values to plot
        confidence_level: Confidence level for intervals (default: 0.95)
        start_round: Starting round for training curves (default: 1)
        
    Returns:
        Tuple of (Figure, Axes)
    """
    # Load & pre-filter records
    all_records, _ = results_manager.load_results(ExperimentType.EVALUATION)
    all_records = [r for r in all_records if r.error is None]

    # Organize by (cost, server) for convenience
    records_by_cs = defaultdict(lambda: defaultdict(list))
    for rec in all_records:
        records_by_cs[rec.cost][rec.server_type].append(rec)

    # Aggregate metrics per cost & server
    agg_curves = {}
    agg_test = {}
    min_rounds_per_cost = {}

    for cost in target_costs:
        recs_local = records_by_cs[cost]["local"]
        recs_fed = records_by_cs[cost]["fedavg"]
        if not recs_local or not recs_fed:
            continue  # need both servers to compare

        # find common min # rounds
        min_r = min(
            min(len(r.metrics.get(MetricKey.TRAIN_LOSSES, [])) for r in recs_local),
            min(len(r.metrics.get(MetricKey.TRAIN_LOSSES, [])) for r in recs_fed),
        )
        if min_r < start_round:
            continue
        min_rounds_per_cost[cost] = min_r

        agg_curves[cost] = {
            "local": aggregate_train_val(recs_local, min_r, confidence_level),
            "fedavg": aggregate_train_val(recs_fed, min_r, confidence_level),
        }
        agg_test[cost] = {
            "local": aggregate_test(recs_local, confidence_level),
            "fedavg": aggregate_test(recs_fed, confidence_level),
        }

    costs = [c for c in target_costs if c in agg_curves]
    if not costs:
        print("No valid data found – nothing to plot.")
        return None, None

    # Figure layout (rows = 1 + |costs|, cols = 2)
    n_rows = 1 
    if plot_losses:
        n_rows += len(costs)
    fig, axes = plt.subplots(n_rows, 2, figsize=(12, 4.5 * n_rows), sharey=False)

    # Final-test metrics – row 0
    x = np.arange(len(costs))
    width = 0.35
    
    # Capture local color (typically blue) and fedavg color (green)
    local_color = 'tab:blue'  # Using standard matplotlib color
    fedavg_color = 'green'

    def _bar(ax, vals, yerr, label, offset=0, color=None):
        bars = ax.bar(x + offset, vals, width, yerr=yerr, label=label, capsize=4, alpha=0.8, color=color)
        return bars[0].get_facecolor() if not color else color  # Return actual color used

    # Loss
    ax_loss = axes[0, 0] if plot_losses else axes[0]
    loc_loss = [agg_test[c]["local"]["mean_loss"] for c in costs]
    fed_loss = [agg_test[c]["fedavg"]["mean_loss"] for c in costs]
    loc_loss_err = [
        [agg_test[c]["local"]["mean_loss"] - agg_test[c]["local"]["lower_loss"],
         agg_test[c]["local"]["upper_loss"] - agg_test[c]["local"]["mean_loss"]]
        for c in costs
    ]
    fed_loss_err = [
        [agg_test[c]["fedavg"]["mean_loss"] - agg_test[c]["fedavg"]["lower_loss"],
         agg_test[c]["fedavg"]["upper_loss"] - agg_test[c]["fedavg"]["mean_loss"]]
        for c in costs
    ]
    local_color = _bar(ax_loss, loc_loss, np.transpose(loc_loss_err), "Local", -width/2, color=local_color)
    fedavg_color = _bar(ax_loss, fed_loss, np.transpose(fed_loss_err), "FedAvg", width/2, color=fedavg_color)
    ax_loss.set_title("Final Test Loss")
    ax_loss.set_xticks(x, [str(c) for c in costs])
    ax_loss.set_ylabel("Loss")
    ax_loss.grid(True, ls="--", alpha=0.3)
    ax_loss.legend()

    # Score
    ax_score = axes[0, 1] if plot_losses else axes[1]
    loc_score = [agg_test[c]["local"]["mean_score"] for c in costs]
    fed_score = [agg_test[c]["fedavg"]["mean_score"] for c in costs]
    loc_score_err = [
        [agg_test[c]["local"]["mean_score"] - agg_test[c]["local"]["lower_score"],
         agg_test[c]["local"]["upper_score"] - agg_test[c]["local"]["mean_score"]]
        for c in costs
    ]
    fed_score_err = [
        [agg_test[c]["fedavg"]["mean_score"] - agg_test[c]["fedavg"]["lower_score"],
         agg_test[c]["fedavg"]["upper_score"] - agg_test[c]["fedavg"]["mean_score"]]
        for c in costs
    ]
    _bar(ax_score, loc_score, np.transpose(loc_score_err), "Local", -width/2, color=local_color)
    _bar(ax_score, fed_score, np.transpose(fed_score_err), "FedAvg", width/2, color=fedavg_color)
    ax_score.set_title("Final Test Score")
    ax_score.set_xticks(x, [str(c) for c in costs])
    ax_score.set_ylabel("Score")
    ax_score.grid(True, ls="--", alpha=0.3)
    ax_score.legend()
    if plot_losses:
        for row_idx, cost in enumerate(costs, start=1):
            min_r = min_rounds_per_cost[cost]
            rounds = np.arange(start_round, min_r)
            
            # Training loss curves (left column)
            ax_train = axes[row_idx, 0]
            loc_data = agg_curves[cost]["local"]
            fed_data = agg_curves[cost]["fedavg"]
            
            # Plot local training
            ax_train.plot(rounds, loc_data["mean_train"][start_round:min_r], color=local_color, 
                        linestyle='-', label="Local")
            ax_train.fill_between(rounds, 
                                loc_data["lower_train"][start_round:min_r], 
                                loc_data["upper_train"][start_round:min_r], 
                                color=local_color, alpha=0.2)
            
            # Plot fedavg training
            ax_train.plot(rounds, fed_data["mean_train"][start_round:min_r], color=fedavg_color, 
                        linestyle='-', label="FedAvg")
            ax_train.fill_between(rounds, 
                                fed_data["lower_train"][start_round:min_r], 
                                fed_data["upper_train"][start_round:min_r], 
                                color=fedavg_color, alpha=0.2)
            
            ax_train.set_title(f"Training Loss for Cost = {cost}")
            ax_train.set_xlabel("Round")
            ax_train.set_ylabel("Training Loss")
            ax_train.grid(True, ls="--", alpha=0.3)
            ax_train.legend()
            
            # Validation loss curves (right column)
            ax_val = axes[row_idx, 1]
            
            # Plot local validation
            ax_val.plot(rounds, loc_data["mean_val"][start_round:min_r], color=local_color, 
                        linestyle='--', label="Local")
            ax_val.fill_between(rounds, 
                            loc_data["lower_val"][start_round:min_r], 
                            loc_data["upper_val"][start_round:min_r], 
                            color=local_color, alpha=0.2)
            
            # Plot fedavg validation
            ax_val.plot(rounds, fed_data["mean_val"][start_round:min_r], color=fedavg_color, 
                        linestyle='--', label="FedAvg")
            ax_val.fill_between(rounds, 
                            fed_data["lower_val"][start_round:min_r], 
                            fed_data["upper_val"][start_round:min_r], 
                            color=fedavg_color, alpha=0.2)
            
            ax_val.set_title(f"Validation Loss for Cost = {cost}")
            ax_val.set_xlabel("Round")
            ax_val.set_ylabel("Validation Loss")
            ax_val.grid(True, ls="--", alpha=0.3)
            ax_val.legend()

        # Adjust layout
        plt.tight_layout()
    
    return fig, axes

def mean_ci_bootstrap(values: np.ndarray, confidence: float = 0.95, n_bootstrap: int = 1000, seed: int = 42) -> Tuple[float, float, float]:
    """
    Return mean and bootstrapped CI at the specified confidence level.
    Args:
        values: Array of values to compute CI for
        confidence: Confidence level (default: 0.95 for 95% CI)
        n_bootstrap: Number of bootstrap samples (default: 1000)
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (mean, lower_ci, upper_ci)
    """
    if isinstance(values, list) and len(values) == 0:
        return np.nan, np.nan, np.nan
    
    if isinstance(values, np.ndarray) and (values.size == 0 or np.isnan(values).all()):
        return np.nan, np.nan, np.nan
        
    if isinstance(values, np.ndarray) and values.ndim == 0:
        values = values[None]
    
    n = len(values)
    mean = np.mean(values)
    if n == 1:
        return mean, mean, mean
    np.random.seed(seed)
    bootstrap_indices = np.random.randint(0, n, size=(n_bootstrap, n))
    bootstrap_means = np.mean(values[bootstrap_indices], axis=1)
    alpha = (1 - confidence) * 2
    lower_ci = np.percentile(bootstrap_means, 100 * alpha)
    upper_ci = np.percentile(bootstrap_means, 100 * (1 - alpha))
    
    return mean, lower_ci, upper_ci


# =============================================================================
# == OT Analysis Functions ==
# =============================================================================

def load_ot_results(results_mgr: ResultsManager, filter_status: str = "Success") -> Tuple[pd.DataFrame, Dict]:
    """
    Load OT analysis results and convert to a DataFrame.
    
    Args:
        results_mgr: ResultsManager instance
        filter_status: Status to filter by (default: "Success")
        
    Returns:
        Tuple of (DataFrame of records, metadata dictionary)
    """
    # Load results
    loaded_dicts, ot_metadata = results_mgr.load_results(ExperimentType.OT_ANALYSIS)
    
    if not loaded_dicts:
        return pd.DataFrame(), {}
    
    # Convert to DataFrame
    df = pd.DataFrame(loaded_dicts)
    
    # Filter by status if requested
    if filter_status and 'status' in df.columns:
        df = df[df['status'] == filter_status]
    
    return df, ot_metadata

def aggregate_ot_data(df: pd.DataFrame, grouping_keys: List[str] = None, 
                     include_percentages: bool = True, 
                     metric_type: str = 'score') -> pd.DataFrame:
    """
    Aggregate OT data by grouping keys with confidence intervals.
    Now includes percentage change relative to local performance.
    
    Args:
        df: DataFrame with OT results
        grouping_keys: Keys to group by (default: ['ot_method_name', 'fl_cost_param'])
        include_percentages: Whether to calculate percentage change
        metric_type: Type of metric ('score' or 'loss') to determine percentage calculation
        
    Returns:
        DataFrame with aggregated statistics including both absolute and percentage changes
    """
    if df.empty:
        return pd.DataFrame()
    
    # Default grouping keys
    if grouping_keys is None:
        grouping_keys = ['ot_method_name', 'fl_cost_param']
    
    # Calculate percentage changes if requested and not already present
    if include_percentages and 'fl_performance_percent' not in df.columns:
        # Create a temp column for percentage change
        if metric_type.lower() == 'score':
            # For scores (higher is better): (performance_delta / local_metric) * 100
            df['fl_performance_percent'] = (df['fl_performance_delta'] / df['fl_local_metric']) * 100
        else:
            # For losses (lower is better): (performance_delta / local_metric) * 100
            # Note: delta is already calculated as local - algorithm
            df['fl_performance_percent'] = (df['fl_performance_delta'] / df['fl_local_metric']) * 100
    
    # Create an empty DataFrame to store results
    aggregated_data = pd.DataFrame()
    
    # For each group, calculate the statistics manually
    for name, group in df.groupby(grouping_keys):
        # Create a dictionary for this group's results
        result = {}
        
        # If grouping_keys is a list, name will be a tuple
        if isinstance(name, tuple):
            for i, key in enumerate(grouping_keys):
                result[key] = name[i]
        else:
            # If only one grouping key, name will be a scalar
            result[grouping_keys[0]] = name
        
        # Calculate OT cost statistics
        ot_values = group['ot_cost_value'].dropna().values
        if len(ot_values) > 0:
            result['ot_cost_mean'] = np.mean(ot_values)
            ci_low, ci_high = mean_ci_bootstrap(ot_values, 0.8)[1:3]
            result['ot_cost_low'] = ci_low
            result['ot_cost_high'] = ci_high
        else:
            result['ot_cost_mean'] = np.nan
            result['ot_cost_low'] = np.nan
            result['ot_cost_high'] = np.nan
        
        # Calculate delta statistics
        delta_values = group['fl_performance_delta'].dropna().values
        if len(delta_values) > 0:
            result['delta_mean'] = np.mean(delta_values)
            ci_low, ci_high = mean_ci_bootstrap(delta_values, 0.8)[1:3]
            result['delta_low'] = ci_low
            result['delta_high'] = ci_high
        else:
            result['delta_mean'] = np.nan
            result['delta_low'] = np.nan
            result['delta_high'] = np.nan
        
        # Calculate percentage statistics if requested
        if include_percentages and 'fl_performance_percent' in group.columns:
            percent_values = group['fl_performance_percent'].dropna().values
            if len(percent_values) > 0:
                result['percent_delta_mean'] = np.mean(percent_values)
                ci_low, ci_high = mean_ci_bootstrap(percent_values)[1:3]
                result['percent_delta_low'] = ci_low
                result['percent_delta_high'] = ci_high
            else:
                result['percent_delta_mean'] = np.nan
                result['percent_delta_low'] = np.nan
                result['percent_delta_high'] = np.nan
        
        # Count number of points
        result['num_points'] = len(group)
        
        # Append to aggregated data
        aggregated_data = pd.concat([aggregated_data, pd.DataFrame([result])], ignore_index=True)
    
    # Calculate error bar values for plotting
    if not aggregated_data.empty:
        # X-axis error bars
        aggregated_data['x_err_lower'] = (aggregated_data['ot_cost_mean'] - aggregated_data['ot_cost_low']).apply(lambda x: max(0, x) if not np.isnan(x) else 0)
        aggregated_data['x_err_upper'] = (aggregated_data['ot_cost_high'] - aggregated_data['ot_cost_mean']).apply(lambda x: max(0, x) if not np.isnan(x) else 0)
        
        # Y-axis error bars for absolute delta
        aggregated_data['y_err_lower'] = (aggregated_data['delta_mean'] - aggregated_data['delta_low']).apply(lambda x: max(0, x) if not np.isnan(x) else 0)
        aggregated_data['y_err_upper'] = (aggregated_data['delta_high'] - aggregated_data['delta_mean']).apply(lambda x: max(0, x) if not np.isnan(x) else 0)
        
        # Y-axis error bars for percentage delta
        if include_percentages and 'percent_delta_mean' in aggregated_data.columns:
            aggregated_data['percent_y_err_lower'] = (aggregated_data['percent_delta_mean'] - aggregated_data['percent_delta_low']).apply(lambda x: max(0, x) if not np.isnan(x) else 0)
            aggregated_data['percent_y_err_upper'] = (aggregated_data['percent_delta_high'] - aggregated_data['percent_delta_mean']).apply(lambda x: max(0, x) if not np.isnan(x) else 0)
    
    return aggregated_data

def plot_ot_errorbar(aggregated_data: pd.DataFrame, dataset_name: str, num_fl_clients: int, 
                    use_percentage: bool = True, figsize=None) -> Tuple[Optional[plt.Figure], Optional[plt.Axes]]:
    """
    Create an error bar plot of OT cost vs. performance delta.
    
    Args:
        aggregated_data: DataFrame with aggregated OT statistics
        dataset_name: Name of the dataset for the title
        num_fl_clients: Number of FL clients for the title
        use_percentage: Whether to use percentage change (True) or absolute delta (False)
        figsize: Optional figure size tuple
    
    Returns:
        Tuple of (Figure, Axes) or (None, None) if data is empty
    """
    if aggregated_data.empty:
        return None, None
    
    # Get unique OT methods
    unique_ot_methods = aggregated_data['ot_method_name'].unique()
    n_methods = len(unique_ot_methods)
    
    # Create figure
    if figsize is None:
        figsize = (7 * n_methods, 6)
    fig, axes = plt.subplots(1, n_methods, figsize=figsize, sharey=True, squeeze=False)
    axes = axes.flatten()
    
    # Check if we can use percentages
    can_use_percentage = use_percentage and 'percent_delta_mean' in aggregated_data.columns
    
    # Set color scheme based on algorithms rather than costs
    if 'algorithm' in aggregated_data.columns:
        unique_algorithms = sorted(aggregated_data['algorithm'].unique())
        # Use distinct color scheme appropriate for categorical data
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_algorithms)))
        algorithm_to_color = dict(zip(unique_algorithms, colors))
        color_by = 'algorithm'
        legend_title = "Algorithm"
    else:
        # Fall back to original coloring by cost if algorithm column not present
        unique_costs = sorted(aggregated_data['fl_cost_param'].unique())
        colors = plt.cm.viridis(np.linspace(0, 1, len(unique_costs)))
        algorithm_to_color = dict(zip(unique_costs, colors))
        color_by = 'fl_cost_param'
        legend_title = "Cost"
    
    # Plot each OT method
    for i, method_name in enumerate(unique_ot_methods):
        ax = axes[i]
        method_df = aggregated_data[aggregated_data['ot_method_name'] == method_name]
        
        # Track which items we've already plotted (for legend)
        plotted_items = set()
        
        # Plot each data point with error bars
        for _, row in method_df.iterrows():
            item_key = row[color_by]
            
            # Only include label for first occurrence of each algorithm/cost
            label = str(item_key) if item_key not in plotted_items else None
            color = algorithm_to_color[item_key]
            
            # Determine which values to use (percentage or absolute)
            if can_use_percentage:
                # Use percentage values
                y_value = row['percent_delta_mean']
                yerr = [[row.get('percent_y_err_lower', 0)], [row.get('percent_y_err_upper', 0)]]
            else:
                # Use absolute delta values
                y_value = row['delta_mean']
                yerr = [[row.get('y_err_lower', 0)], [row.get('y_err_upper', 0)]]
            
            # X error bars are always the same
            xerr = [[row.get('x_err_lower', 0)], [row.get('x_err_upper', 0)]]
            
            ax.errorbar(
                row['ot_cost_mean'], y_value,
                xerr=xerr, yerr=yerr,
                fmt='o', capsize=5, label=label, markersize=8, elinewidth=1.5,
                color=color
            )
            
            plotted_items.add(item_key)
        
        # Set title and labels
        ax.set_title(f"{method_name}", fontsize=14)
        ax.set_xlabel("Mean OT Cost", fontsize=12)
        if i == 0:
            if can_use_percentage:
                y_label = "% Change Relative to Local"
            else:
                y_label = "Mean Performance Delta (Algorithm - Local)" if color_by == 'algorithm' else "Mean Performance Delta (FedAvg - Local)"
            ax.set_ylabel(y_label, fontsize=12)
        
        # Add grid and reference lines
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.axhline(0, color='grey', linestyle='--', linewidth=0.8)
        
        # Add median OT cost vertical line if available
        x_means = method_df['ot_cost_mean']
        if not x_means.empty and not pd.isna(x_means).all():
            median_ot_cost = x_means.median()
            ax.axvline(median_ot_cost, color='lightgrey', linestyle=':', linewidth=0.8)
        
        # Ensure axes are appropriate
        if not x_means.empty and not pd.isna(x_means).all():
            x_min = min(x_means) - max(method_df['x_err_lower']) if not method_df['x_err_lower'].empty else min(x_means)
            x_max = max(x_means) + max(method_df['x_err_upper']) if not method_df['x_err_upper'].empty else max(x_means)
            x_padding = (x_max - x_min) * 0.1
            ax.set_xlim(max(0, x_min - x_padding), x_max + x_padding)
        
        # Add legend to the subplot
        if plotted_items:
            ax.legend(title=legend_title)
    
    # Add overall title with indication of percentage or absolute
    title_type = "% Change Relative to Local" if can_use_percentage else "Performance Delta"
    fig.suptitle(f"OT Cost vs. {title_type} for {dataset_name} ({num_fl_clients} FL Clients)", fontsize=16)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
    return

def plot_ot_scatter(df: pd.DataFrame, dataset_name: str, num_fl_clients: int, 
                   use_percentage: bool = True, figsize=None) -> Tuple[Optional[plt.Figure], Optional[plt.Axes]]:
    """
    Create a scatter plot with trend lines for OT cost vs. performance delta,
    organized by algorithm (rows) and OT method (columns).
    
    Args:
        df: DataFrame with OT results (must contain 'algorithm' column)
        dataset_name: Name of the dataset for the title
        num_fl_clients: Number of FL clients for the title
        use_percentage: Whether to use percentage change (True) or absolute delta (False)
        figsize: Optional figure size tuple
    
    Returns:
        Tuple of (Figure, Axes) or (None, None) if data is empty
    """
    if df.empty:
        return None, None
    
    # Get unique OT methods and algorithms
    unique_ot_methods = df['ot_method_name'].unique()
    
    # Check if 'algorithm' column exists, otherwise use just fedavg
    if 'algorithm' not in df.columns:
        df['algorithm'] = 'fedavg'  # Add default algorithm column
        print("No 'algorithm' column found, assuming all data is for 'fedavg'")
        
    unique_algorithms = df['algorithm'].unique()
    
    n_methods = len(unique_ot_methods)
    n_algorithms = len(unique_algorithms)
    
    # Create figure with rows=algorithms, columns=methods
    if figsize is None:
        figsize = (7 * n_methods, 5 * n_algorithms)
    fig, axes = plt.subplots(n_algorithms, n_methods, figsize=figsize, 
                             sharex='col', sharey='row', squeeze=False)
    
    # Check if we can use percentages
    can_use_percentage = use_percentage and 'fl_performance_percent' in df.columns
    
    # Determine which y-column to use
    y_column = 'fl_performance_percent' if can_use_percentage else 'fl_performance_delta'
    
    # Plot each combination of algorithm and OT method
    for row_idx, algorithm in enumerate(unique_algorithms):
        for col_idx, method_name in enumerate(unique_ot_methods):
            # Get current axis
            ax = axes[row_idx, col_idx]
            
            # Filter data for this algorithm and method
            filtered_df = df[(df['algorithm'] == algorithm) & (df['ot_method_name'] == method_name)]
            
            if not filtered_df.empty:
                # Create scatter plot
                scatter = sns.scatterplot(
                    data=filtered_df,
                    x='ot_cost_value',
                    y=y_column,
                    hue='fl_cost_param',
                    palette='viridis',
                    alpha=0.7,
                    ax=ax
                )
                
                # Add regression line if enough points
                valid_data = filtered_df.dropna(subset=['ot_cost_value', y_column])
                if len(valid_data) > 2:
                    sns.regplot(
                        data=valid_data,
                        x='ot_cost_value',
                        y=y_column,
                        scatter=False,
                        line_kws={'color': 'red', 'lw': 2},
                        ax=ax
                    )
            
            # Set title (OT method at top of column, algorithm at start of row)
            if row_idx == 0:
                ax.set_title(f"{method_name}", fontsize=14)
            
            # Set x-axis label for bottom row only
            if row_idx == n_algorithms - 1:
                ax.set_xlabel("OT Cost", fontsize=12)
            
            # Set y-axis label for first column only
            if col_idx == 0:
                y_label = "% Change vs. Local" if can_use_percentage else f"{algorithm.upper()} - Local"
                ax.set_ylabel(y_label, fontsize=12)
            
            # Add algorithm label on the right side of each row
            if col_idx == n_methods - 1:
                ax.text(1.02, 0.5, algorithm.upper(), 
                        transform=ax.transAxes, 
                        fontsize=14, fontweight='bold',
                        va='center', rotation=-90)
            
            # Add grid and reference line
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.axhline(0, color='grey', linestyle='--', linewidth=0.8)
            
            # Keep legend only for the rightmost plot in each row
            if col_idx < n_methods - 1 and ax.get_legend() is not None:
                ax.get_legend().remove()
    
    # Add overall title
    title_type = "% Performance Change vs. Local" if can_use_percentage else "Performance Delta"
    fig.suptitle(f"OT Cost vs. {title_type} for {dataset_name} ({num_fl_clients} FL Clients)", 
                 fontsize=16, y=0.98)
    
    # Adjust layout with more space for row labels
    plt.tight_layout(rect=[0, 0, 0.98, 0.95])
    plt.show()
    return

def add_algorithm_results_to_ot(results_manager: ResultsManager, ot_df: pd.DataFrame, metric_type: str = 'score') -> pd.DataFrame:
    """
    Enriches OT analysis DataFrame with metrics from all algorithms (FedAvg, FedProx, PFedMe, Ditto)
    while preserving run-level granularity for confidence interval calculation.
    
    Args:
        results_manager: ResultsManager instance to load evaluation results
        ot_df: Original OT analysis DataFrame with fl_cost_param, fl_run_idx, etc.
        metric_type: Type of metric ('score' or 'loss')
        
    Returns:
        Expanded DataFrame with algorithm column and deltas for all algorithms
    """
    # Load evaluation results
    eval_records, _ = results_manager.load_results(ExperimentType.EVALUATION)
    
    # Create a list to store expanded records
    expanded_records = []
    
    # First, organize eval_records by (cost, run_idx, server_type) for efficient lookup
    eval_lookup = {}
    for rec in eval_records:
        if rec.error is None and rec.metrics:
            key = (rec.cost, rec.run_idx, rec.server_type)
            eval_lookup[key] = rec
    
    # Set the metric key based on metric_type
    metric_key = 'test_scores' if metric_type == 'score' else 'test_losses'
    
    # Process each OT record
    for _, ot_row in ot_df.iterrows():
        cost_param = ot_row['fl_cost_param']
        run_idx = ot_row['fl_run_idx']
        
        # Get the local performance for this specific run and cost
        local_key = (cost_param, run_idx, 'local')
        if local_key not in eval_lookup:
            continue  # Skip if we don't have local performance for this run/cost
        
        local_rec = eval_lookup[local_key]
        if metric_key not in local_rec.metrics or not local_rec.metrics[metric_key]:
            continue  # Skip if local metrics are missing
            
        local_val = local_rec.metrics[metric_key][-1]  # Use final value
        
        # Create a record for each algorithm if data is available
        for algo in ['fedavg', 'fedprox', 'pfedme', 'ditto']:
            algo_key = (cost_param, run_idx, algo)
            if algo_key not in eval_lookup:
                continue  # Skip if this algorithm doesn't have data for this run/cost
                
            algo_rec = eval_lookup[algo_key]
            if metric_key not in algo_rec.metrics or not algo_rec.metrics[metric_key]:
                continue  # Skip if algorithm metrics are missing
                
            algo_val = algo_rec.metrics[metric_key][-1]  # Use final value
            
            # Compute delta based on metric type (scores or losses)
            if metric_type == 'score':
                # For scores, higher is better: algo - local (positive means algo better)
                delta = algo_val - local_val
            else:
                # For losses, lower is better: local - algo (positive means algo better)
                delta = local_val - algo_val
            
            # Create a new row with all fields from the original OT row
            new_row = ot_row.copy()
            new_row['algorithm'] = algo
            new_row['fl_performance_delta'] = delta
            new_row['fl_local_metric'] = local_val
            new_row['fl_algo_metric'] = algo_val
            
            # Also add percentage calculation
            if abs(local_val) > 1e-10:  # Avoid division by zero
                new_row['fl_performance_percent'] = (delta / abs(local_val)) * 100
            else:
                new_row['fl_performance_percent'] = np.nan
                
            expanded_records.append(new_row)
    
    # Convert to DataFrame
    return pd.DataFrame(expanded_records)

def calculate_ot_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Pearson and Spearman correlations between OT costs and performance deltas.
    
    Args:
        df: DataFrame with OT results
        
    Returns:
        DataFrame with correlation statistics
    """
    if df.empty:
        return pd.DataFrame()
    
    # Create a dataframe to store correlation results
    correlation_results = []
    
    # Calculate correlation for each OT method
    for method_name in df['ot_method_name'].unique():
        method_df = df[df['ot_method_name'] == method_name]
        valid_data = method_df.dropna(subset=['ot_cost_value', 'fl_performance_delta'])
        
        if len(valid_data) > 2:
            # Pearson correlation
            pearson_r, pearson_p = stats.pearsonr(
                valid_data['ot_cost_value'],
                valid_data['fl_performance_delta']
            )
            
            # Spearman correlation (rank-based, more robust to outliers)
            spearman_r, spearman_p = stats.spearmanr(
                valid_data['ot_cost_value'],
                valid_data['fl_performance_delta']
            )
            
            correlation_results.append({
                'OT Method': method_name,
                'Sample Size': len(valid_data),
                'Pearson r': pearson_r,
                'Pearson p-value': pearson_p,
                'Spearman r': spearman_r,
                'Spearman p-value': spearman_p
            })
    
    # Create and format the correlation results dataframe
    if correlation_results:
        correlation_df = pd.DataFrame(correlation_results)
        
        # Format p-values for better readability
        correlation_df['Pearson p-value'] = correlation_df['Pearson p-value'].apply(
            lambda p: f"{p:.3f}" if p >= 0.001 else "<0.001"
        )
        correlation_df['Spearman p-value'] = correlation_df['Spearman p-value'].apply(
            lambda p: f"{p:.3f}" if p >= 0.001 else "<0.001"
        )
        
        # Add significance indicators
        correlation_df['Pearson Significance'] = correlation_df['Pearson p-value'].apply(
            lambda p: '***' if p == "<0.001" else
                      '**' if p != "<0.001" and float(p) < 0.01 else
                      '*' if p != "<0.001" and float(p) < 0.05 else
                      'ns'
        )
        correlation_df['Spearman Significance'] = correlation_df['Spearman p-value'].apply(
            lambda p: '***' if p == "<0.001" else
                      '**' if p != "<0.001" and float(p) < 0.01 else
                      '*' if p != "<0.001" and float(p) < 0.05 else
                      'ns'
        )
        
        return correlation_df
    else:
        return pd.DataFrame()

def get_tuning_summary(   manager: ResultsManager,
                          exerpiment_type: str,
                          server_filter: Optional[str] = None,
                          higher_is_better_metric: bool = False) -> pd.DataFrame:
    """
    Loads LR tuning results, calculates the average (median) performance 
    for each (cost, server, learning_rate) across all runs,
    and identifies the best LR per cost/server.

    Args:
        dataset_name: Name of the dataset.
        results_root_dir: The root directory where results are stored.
        num_target_clients: The target number of clients used for naming results files.
        server_filter: If specified, only analyze results for this server type.
        higher_is_better_metric: Set to True if the validation metric is score-based.

    Returns:
        pandas.DataFrame: DataFrame with cost, server_type, learning_rate,
                          avg_val_performance (median of median val losses/scores), 
                          and a flag indicating if it's the best LR for that cost/server.
    """

    exp_types = {'learning_rate':  ExperimentType.LEARNING_RATE, 
                 'reg_param': ExperimentType.REG_PARAM}
    
    tuning_records, metadata = manager.load_results(exp_types[exerpiment_type])
    
    if not tuning_records:
        return pd.DataFrame()

    processed_data = []
    
    metric_to_use = MetricKey.VAL_SCORES if higher_is_better_metric else MetricKey.VAL_LOSSES
    
    for record in tuning_records:
        if record.error is not None:
            continue
        if record.tuning_param_name != exerpiment_type:
            continue
        if server_filter and record.server_type != server_filter:
            continue
             
        metrics_dict = record.metrics
        if not metrics_dict or metric_to_use not in metrics_dict or not metrics_dict[metric_to_use]:
            continue
            
        try:
            # Get the median performance of this specific trial (run)
            # This mirrors the logic in ResultsManager.get_best_parameters
            trial_performance = np.nanmedian(metrics_dict[metric_to_use]) 
            if not np.isfinite(trial_performance):
                continue
        except (ValueError, TypeError) as e: 
            continue

        processed_data.append({
            'cost': record.cost,
            'server_type': record.server_type,
            exerpiment_type: record.tuning_param_value,
            'run_idx': record.run_idx,
            'trial_val_performance': float(trial_performance) # Ensure float
        })

    if not processed_data:
        return pd.DataFrame()

    df = pd.DataFrame(processed_data)
    
    # Group by cost, server, and learning rate, then calculate the mean of trial_val_performance (which are medians of rounds)
    # This aggregation across runs should ideally match get_best_parameters (which uses np.mean of these trial medians)
    agg_summary = df.groupby(['cost', 'server_type', exerpiment_type], as_index=False).agg(
        avg_val_performance=('trial_val_performance', 'mean'), # Mean of the trial (median) performances
        num_runs_aggregated=('run_idx', 'nunique') # Count how many runs contributed
    )
    
    # Identify the best learning rate for each (cost, server_type) combination
    if higher_is_better_metric:
        best_indices = agg_summary.groupby(['cost', 'server_type'])['avg_val_performance'].idxmax()
    else:
        best_indices = agg_summary.groupby(['cost', 'server_type'])['avg_val_performance'].idxmin()
        
    agg_summary['is_best_param'] = False
    if not best_indices.empty: # Check if best_lr_indices is not empty
        agg_summary.loc[best_indices, 'is_best_param'] = True
    
    # Sort for better readability
    agg_summary = agg_summary.sort_values(by=['cost', 'server_type', 'avg_val_performance' if not higher_is_better_metric else 'avg_val_performance'],
                                          ascending=[True, True, not higher_is_better_metric if not higher_is_better_metric else False]).reset_index(drop=True)

    return agg_summary

def plot_reg_param_vs_cost(best_params_df: pd.DataFrame) -> None:
    """
    Plot the best regularization parameter against the cost for each server type.
    Args:
        best_params_df (pd.DataFrame): DataFrame containing the best regularization parameters and costs.
    """
    # Convert 'cost' and 'reg_param' columns to numeric
    plt.figure(figsize=(12, 8))
    scatterplot = sns.scatterplot(
        data=best_params_df,
        x='cost',
        y='reg_param',
        hue='server_type',
        s=120,  
        alpha=0.9, 
        palette='deep', 
        edgecolor='w', #
        linewidth=0.5
    )
    plt.xlabel("Cost (Heterogeneity Parameter)", fontsize=14)
    plt.ylabel("Best Regularization Parameter (reg_param)", fontsize=14)
    plt.title("Best Regularization Parameter vs. Cost by Server Type", fontsize=16)
    plt.legend(title="Server Type", bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
    plt.xticks(rotation=45)
    plt.tight_layout(rect=[0, 0, 0.85, 1]) 
    plt.show()
    # plt.savefig("best_reg_param_vs_cost.png", dpi=300, bbox_inches='tight')

def smooth_curve(points, window_size=5, poly_order=1):
    """
    Applies Savitzky-Golay filter for smoothing a time series.
    
    Args:
        points (np.ndarray): Array of data points
        window_size (int): Window size for the filter
        poly_order (int): Polynomial order for the filter
        
    Returns:
        np.ndarray: Smoothed data points
    """
    import numpy as np
    from scipy.signal import savgol_filter
    
    # Convert to numpy array if needed
    points = np.array(points)
    
    # Handle case with too few points
    if len(points) < window_size:
        return points
    
    # Handle NaN values through interpolation
    nan_indices = np.isnan(points)
    if np.any(nan_indices):
        points_no_nan = points.copy()
        non_nan_indices = ~nan_indices
        
        # Use interpolation to fill NaN values
        points_no_nan[nan_indices] = np.interp(
            np.flatnonzero(nan_indices), 
            np.flatnonzero(non_nan_indices), 
            points_no_nan[non_nan_indices]
        )
        points = points_no_nan
    
    # Ensure window_size is odd
    window_size = window_size if window_size % 2 == 1 else window_size + 1
    # Ensure poly_order < window_size
    poly_order = min(poly_order, window_size - 1)
    
    # Apply Savitzky-Golay filter
    smoothed = savgol_filter(points, window_size, poly_order)
    return smoothed

def get_diversity_summary(results_manager, dataset_name, experiment_type, server_filter=None):
    """
    Extracts diversity metrics from experiment results into a pandas DataFrame.
    
    Args:
        results_manager (ResultsManager): Instance of ResultsManager
        dataset_name (str): Name of the dataset
        experiment_type (str): Type of experiment (e.g., 'evaluation')
        server_filter (List[str], optional): List of server types to include. If None, includes fedavg.
        
    Returns:
        pandas.DataFrame: DataFrame with diversity metrics
    """
    import pandas as pd
    import numpy as np
    
    # Load TrialRecords using the results_manager
    records, _ = results_manager.load_results(experiment_type)
    
    # If no server filter provided, default to FedAvg which has diversity metrics
    if not server_filter:
        server_filter = ['fedavg']
    
    # Filter records
    records = [r for r in records if r.server_type in server_filter]
    
    # Extract metrics into a list of dictionaries
    data = []
    for record in records:
        if record.metrics is None or record.error is not None:
            continue
        
        # Get basic record info
        cost = record.cost
        server_type = record.server_type
        run_idx = record.run_idx
        
        # Extract the metrics we're interested in
        train_losses = record.metrics.get('train_losses', [])
        val_losses = record.metrics.get('val_losses', [])
        weight_div = record.metrics.get('weight_div', [])
        weight_orient = record.metrics.get('weight_orient', [])
        
        # Add a row for each round
        max_rounds = max(
            len(train_losses), 
            len(val_losses), 
            len(weight_div) if weight_div is not None else 0, 
            len(weight_orient) if weight_orient is not None else 0
        )
        
        for round_num in range(max_rounds):
            row = {
                'dataset': dataset_name,
                'cost': cost,
                'server_type': server_type,
                'run_idx': run_idx,
                'round_num': round_num,
                'train_loss': train_losses[round_num] if round_num < len(train_losses) else np.nan,
                'val_loss': val_losses[round_num] if round_num < len(val_losses) else np.nan,
                'weight_div': weight_div[round_num] if weight_div and round_num < len(weight_div) else np.nan,
                'weight_orient': weight_orient[round_num] if weight_orient and round_num < len(weight_orient) else np.nan
            }
            data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    return df

def plot_diversity_metrics(df, dataset_name, costs=None, window_size=5, poly_order=1, figsize=(10, 6)):
    """
    Plots diversity metrics over rounds for specified costs.
    
    Args:
        df (pandas.DataFrame): DataFrame from get_diversity_summary
        dataset_name (str): Name of the dataset for plot title
        costs (List[Any], optional): List of costs to plot. If None, plots all costs.
        window_size (int): Window size for smoothing
        poly_order (int): Polynomial order for smoothing
        figsize (tuple): Figure size (width, height)
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Filter costs if specified
    if costs is not None:
        df = df[df['cost'].isin(costs)].copy()
    else:
        costs = sorted(df['cost'].unique())
    
    # Setup figure and grid
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    axes = axes.flatten()
    
    # Define plot titles and metrics to extract
    plot_configs = [
        {'title': 'Weight Update Divergence', 'metric': 'weight_div', 'ax_idx': 0},
        {'title': 'Update Direction Similarity', 'metric': 'weight_orient', 'ax_idx': 1},
    ]
    
    # Define a list of colors for different costs
    colors = plt.cm.viridis(np.linspace(0, 1, len(costs)))
    
    # Plot each metric
    for config in plot_configs:
        ax = axes[config['ax_idx']]
        
        # Set plot title and labels
        ax.set_title(config['title'], fontsize=14)
        ax.set_xlabel('Round', fontsize=12)
        ax.set_ylabel(config['title'], fontsize=12)
        
        legend_handles = []
        legend_labels = []
        
        # Plot each cost
        for idx, cost in enumerate(costs):
            # Get data for this cost
            cost_df = df[df['cost'] == cost]
            
            # Skip if no data for this cost
            if cost_df.empty:
                continue
                
            # Group by round_num and compute mean of the metric across runs
            round_data = cost_df.groupby('round_num')[config['metric']].mean().reset_index()
            
            # Get x and y data
            x = round_data['round_num'].values
            y = round_data[config['metric']].values
            
            # Skip if no valid data
            if len(y) == 0 or np.all(np.isnan(y)):
                continue
            
            # Apply smoothing
            y_smooth = smooth_curve(y, window_size, poly_order)
            
            # Plot the smoothed line
            line, = ax.plot(x, y_smooth, color=colors[idx], linewidth=2)
            
            # Add to legend
            legend_handles.append(line)
            legend_labels.append(f"Cost = {cost}")
            
            # Plot original points with lower alpha
            ax.scatter(x, y, color=colors[idx], alpha=0.2)
        
        # Add legend
        if legend_handles:
            ax.legend(legend_handles, legend_labels, fontsize=10, loc='best')
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
    
    # Set overall title
    fig.suptitle(f"Diversity Metrics for {dataset_name}", fontsize=16)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    
    return