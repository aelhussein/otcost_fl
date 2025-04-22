#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


# In[2]:


DATA_DIR = '/gpfs/commons/groups/gursoy_lab/aelhussein/classes/otcost_fl/data/Heart'


# In[33]:


import pandas as pd
import numpy as np
import os
from itertools import combinations

# --- Configuration ---
DATA_DIR = '/gpfs/commons/groups/gursoy_lab/aelhussein/classes/otcost_fl/data/Heart' # User specified path

columns = ['age', 'sex', 'chest_pain_type', 'resting_bp', 'cholesterol',
           'sugar', 'ecg', 'max_hr', 'exercise_angina', 'exercise_ST_depression',
           'slope_ST', 'number_major_vessels', 'thalassemia_hx', 'target']

used_columns =  ['age', 'sex', 'chest_pain_type', 'resting_bp', 'cholesterol',
                 'sugar', 'ecg', 'max_hr', 'exercise_angina', 'exercise_ST_depression',
                 'target']

sites = ['cleveland', 'hungarian', 'switzerland', 'va']

COLS_TO_SCALE = ['age', 'chest_pain_type', 'resting_bp', 'cholesterol',
                 'ecg', 'max_hr', 'exercise_ST_depression']

FEATURE_NAMES = ['age', 'sex', 'chest_pain_type', 'resting_bp', 'cholesterol',
                 'sugar', 'ecg', 'max_hr', 'exercise_angina', 'exercise_ST_depression']

# Mean and Variance values for scaling
scale_values = {
    'age': (53.0872973, 7.01459463e+01),          # mean, variance
    'chest_pain_type': (3.23702703, 8.17756772e-01), # mean, variance
    'resting_bp': (132.74405405, 3.45493057e+02), # mean, variance
    'cholesterol': (220.23648649, 4.88430934e+03), # mean, variance
    'ecg': (0.64513514, 5.92069868e-01),          # mean, variance
    'max_hr': (138.75459459, 5.29172208e+02),     # mean, variance
    'exercise_ST_depression': (0.89532432, 1.11317517e+00) # mean, variance
}

# --- Data Loading and Processing per Site ---
processed_site_data = {}
valid_sites = [] # Keep track of sites successfully processed

for site in sites:
    file_path = f'{DATA_DIR}/processed.{site}.data'
    try:
        # Read data, potentially inferring target as float initially
        site_data = pd.read_csv(
            file_path,
            names=columns,
            na_values='?',
            header=None,
            usecols=used_columns
        ).dropna()

        if not site_data.empty:
            # Make a copy for scaling, leaving original site_data intact for target
            scaled_features_df = site_data.copy()

            # --- Scaling (Only applies to features) ---
            for col in COLS_TO_SCALE:
                # Check if column exists and needs scaling
                if col in scaled_features_df.columns and col in scale_values:
                    mean, variance = scale_values[col]
                    std_dev = np.sqrt(variance)
                    if std_dev > 1e-9:
                         scaled_features_df[col] = (scaled_features_df[col] - mean) / std_dev
                    else:
                         scaled_features_df[col] = scaled_features_df[col] - mean

            # --- Separate Features (X) and Target (y) ---
            # Ensure necessary columns exist
            if 'target' in site_data.columns and all(f in scaled_features_df.columns for f in FEATURE_NAMES):
                # Get scaled features
                X = scaled_features_df[FEATURE_NAMES].values

                # Get original, unscaled target and ensure it's integer
                y = site_data['target'].astype(int).values.reshape(-1, 1)

                # --- Combine X (float) and y (int) for saving ---
                combined_xy = np.concatenate((X, y), axis=1)
                processed_site_data[site] = combined_xy
                valid_sites.append(site)
            # Silently skip if columns missing

    except FileNotFoundError:
        pass # Silently skip
    except Exception:
        pass # Silently skip other errors

# --- Pairwise Saving ---
pair_index = 1
# Generate combinations only from sites that were successfully processed
for site1, site2 in combinations(valid_sites, 2):
    data1 = processed_site_data[site1]
    data2 = processed_site_data[site2]

    # Define format: floats for features, integer for the last (target) column
    num_features = data1.shape[1] - 1
    # Using '%.18e' for floats to match typical numpy float representation, '%d' for integer
    fmt_list = ['%.18e'] * num_features + ['%d']

    np.savetxt(f'{DATA_DIR}/data_1_{pair_index}.00.csv', data1, fmt=fmt_list)
    np.savetxt(f'{DATA_DIR}/data_2_{pair_index}.00.csv', data2, fmt=fmt_list)

    pair_index += 1


# In[51]:


list(combinations(valid_sites, 2))


# In[39]:


import seaborn as sns
import math
import matplotlib.pyplot as plt
def plot_kde_comparison_grid(data1, data2, 
                             column_names=None, 
                             plots_per_row=3, 
                             figsize_per_plot=(5, 4), 
                             label1="Data 1", 
                             label2="Data 2",
                             save_path=None):
    """
    Plots Kernel Density Estimates (KDE) for each column (except the last) 
    of two datasets side-by-side in a grid of subplots.

    Args:
        data1 (np.ndarray): The first dataset (2D NumPy array).
        data2 (np.ndarray): The second dataset (2D NumPy array). Should have the 
                              same number of columns as data1.
        column_names (list, optional): List of names for each column. If None,
                                       generic names ("Column 0", "Column 1", ...)
                                       will be used. Must match the number of columns
                                       in data1/data2.
        plots_per_row (int): Number of subplots to arrange horizontally per row.
        figsize_per_plot (tuple): Approx (width, height) in inches for each subplot.
        label1 (str): Label for data1 in the legend.
        label2 (str): Label for data2 in the legend.
        save_path (str, optional): If provided, saves the figure to this path.
                                   Otherwise, displays the figure.
    """
    if not isinstance(data1, np.ndarray) or not isinstance(data2, np.ndarray):
        raise TypeError("Inputs data1 and data2 must be NumPy arrays.")
        
    if data1.ndim != 2 or data2.ndim != 2:
         raise ValueError("Input arrays must be 2-dimensional.")

    if data1.shape[1] != data2.shape[1]:
        raise ValueError(f"Data sets must have the same number of columns "
                         f"(data1: {data1.shape[1]}, data2: {data2.shape[1]})")

    num_total_cols = data1.shape[1]
    if num_total_cols <= 1:
        print("Warning: Only one column or fewer found. No columns to plot (excluding the last).")
        return None

    num_cols_to_plot = num_total_cols - 1

    # --- Setup Column Names ---
    if column_names is None:
        col_names_to_use = [f"Feature {i}" for i in range(num_cols_to_plot)]
    elif len(column_names) == num_total_cols:
        col_names_to_use = column_names[:-1] # Use provided names, exclude last
    else:
        warnings.warn(f"Length of column_names ({len(column_names)}) does not match "
                      f"number of columns ({num_total_cols}). Using generic names.")
        col_names_to_use = [f"Feature {i}" for i in range(num_cols_to_plot)]

    # --- Setup Figure and Axes ---
    num_rows = math.ceil(num_cols_to_plot / plots_per_row)
    total_figsize = (figsize_per_plot[0] * plots_per_row, figsize_per_plot[1] * num_rows)

    fig, axes = plt.subplots(num_rows, plots_per_row, figsize=total_figsize, 
                             squeeze=False) # squeeze=False ensures axes is always 2D
    axes_flat = axes.flatten() # Flatten for easy iteration

    print(f"Generating {num_cols_to_plot} KDE comparison plots ({num_rows}x{plots_per_row} grid)...")

    # --- Plotting Loop ---
    for i in range(num_cols_to_plot):
        ax = axes_flat[i]
        col_data1 = data1[:, i]
        col_data2 = data2[:, i]

        # Remove NaNs for KDE plot, otherwise it fails
        col_data1_clean = col_data1[~np.isnan(col_data1)]
        col_data2_clean = col_data2[~np.isnan(col_data2)]

        plot_title = col_names_to_use[i]
        
        plotted = False
        if len(col_data1_clean) > 1: # Need >1 point for KDE
            try:
                sns.kdeplot(col_data1_clean, ax=ax, label=label1, fill=True, alpha=0.3)
                plotted = True
            except Exception as e:
                 print(f"  Skipping KDE plot for {label1}, Col {i} ({plot_title}): {e}")
                 ax.text(0.5, 0.6, f"{label1}:\nPlotting Error", ha='center', va='center', transform=ax.transAxes, color='red')

        if len(col_data2_clean) > 1: # Need >1 point for KDE
            try:
                sns.kdeplot(col_data2_clean, ax=ax, label=label2, fill=True, alpha=0.3)
                plotted = True
            except Exception as e:
                 print(f"  Skipping KDE plot for {label2}, Col {i} ({plot_title}): {e}")
                 ax.text(0.5, 0.4, f"{label2}:\nPlotting Error", ha='center', va='center', transform=ax.transAxes, color='orange')

        if not plotted:
             ax.text(0.5, 0.5, "No data to plot", ha='center', va='center', transform=ax.transAxes)
             
        ax.set_title(plot_title)
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
        if plotted:
            ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)

    # --- Hide Unused Axes ---
    for i in range(num_cols_to_plot, len(axes_flat)):
        axes_flat[i].axis('off')

    plt.tight_layout()


# In[ ]:


used_columns =  ['age', 'sex', 'chest_pain_type', 'resting_bp', 'cholesterol',
                 'sugar', 'ecg', 'max_hr', 'exercise_angina', 'exercise_ST_depression',
                 'target']


# In[53]:


x = 3
data1 = np.loadtxt(f'{DATA_DIR}/data_1_{x}.00.csv')
data2 = np.loadtxt(f'{DATA_DIR}/data_2_{x}.00.csv')
plot_kde_comparison_grid(data1, data2, 
                             column_names=None, 
                             plots_per_row=3, 
                             figsize_per_plot=(5, 4), 
                             label1="Data 1", 
                             label2="Data 2",
                             save_path=None)

