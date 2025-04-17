# pipeline_runner.py (Updated Imports Only)
from configs import *
import pandas as pd
import numpy as np
import warnings
from typing import List, Tuple, Dict, Any, Union, Optional

# Import DataManager and BaseOTCalculator (for type hinting)
from ot_data_manager import DataManager
# Import BaseOTCalculator *AND* the new Config and Factory from ot_calculators.py
from ot_calculators import BaseOTCalculator, OTConfig, OTCalculatorFactory

class PipelineRunner:
    """ Orchestrates the OT similarity analysis pipeline for multiple client pairs. """

    # MODIFIED: Takes num_clients for the run being analyzed
    def __init__(self,
                 num_clients: int, # The number of clients in the FL run
                 activation_dir: Optional[str] = None,
                 results_dir: Optional[str] = None):
        """
        Initializes the runner.

        Args:
            num_clients (int): The total number of clients in the FL run to analyze pairs from.
            activation_dir (str, optional): Path to activation cache. Defaults to config path.
            results_dir (str, optional): Path to results directory. Defaults to config path.
        """
        self.num_clients = num_clients # Store the total client count for pair generation
        act_dir = activation_dir if activation_dir else ACTIVATION_DIR
        res_dir = results_dir if results_dir else RESULTS_DIR
        # DataManager is initialized with the total client count for loading correct performance files
        self.data_manager = DataManager(num_clients=num_clients,
                                        activation_dir=act_dir,
                                        results_dir=res_dir)

    # MODIFIED: Signature changed - removed client_id_1, client_id_2, added optional client_pairs
    def run_pipeline(
        self,
        dataset_name: str,
        costs_list: List[Any],
        target_rounds: int,
        num_clients: int, # Number of clients in the source FL run
        num_classes: int,
        ot_configurations: List[OTConfig],
        client_pairs_to_analyze: Optional[List[Tuple[Union[str, int], Union[str, int]]]] = None, # Optional: analyze only specific pairs
        activation_loader_type: str = 'val',
        performance_aggregation: str = 'mean',
        base_seed: int = 42,
        force_activation_regen: bool = False
    ) -> pd.DataFrame:
        """
        Runs the full OT analysis pipeline, iterating through all unique client pairs
        or a specified subset of pairs.

        Args:
            dataset_name (str): Name of the dataset.
            costs_list (List[Any]): List of cost parameters (e.g., alpha values) to iterate over.
            target_rounds (int): The training round number for activations.
            num_clients (int): The total number of clients used in the FL run that generated the data.
            num_classes (int): Number of classes in the dataset.
            ot_configurations (List[OTConfig]): List of OTConfig objects defining calculations.
            client_pairs_to_analyze (List[Tuple], optional): Specific pairs (client_id_1, client_id_2)
                                                             to analyze. If None, all unique pairs are analyzed.
            activation_loader_type (str): Dataloader type for activations ('train', 'val', 'test').
            performance_aggregation (str): Aggregation for performance scores ('mean', 'median').
            base_seed (int): Base random seed for activation generation/loading.
            force_activation_regen (bool): If True, regenerate activations.

        Returns:
            pd.DataFrame: DataFrame containing performance and OT results for each analyzed pair, cost, and config.
        """
        results_list = []

        # --- Validation ---
        if not ot_configurations: raise ValueError("Must provide at least one OT configuration.")
        if not all(isinstance(cfg, OTConfig) for cfg in ot_configurations): raise TypeError("ot_configurations must be a list of OTConfig objects.")
        if not isinstance(num_clients, int) or num_clients < 2: raise ValueError("num_clients must be an integer >= 2.")

        # --- Generate Client Pairs ---
        client_ids = [f'client_{i+1}' for i in range(num_clients)] # Assuming 'client_1', 'client_2', ... format
        if client_pairs_to_analyze:
            # Validate provided pairs
            valid_pairs = []
            for p in client_pairs_to_analyze:
                 if isinstance(p, tuple) and len(p) == 2 and \
                    str(p[0]) in client_ids and str(p[1]) in client_ids and p[0] != p[1]:
                     # Store consistently as strings? Or keep original type? Using strings for now.
                     valid_pairs.append( (str(p[0]), str(p[1])) )
                 else:
                     warnings.warn(f"Invalid client pair provided: {p}. Skipping.")
            pairs_to_run = valid_pairs
            if not pairs_to_run:
                 print("No valid client pairs specified to analyze.")
                 return pd.DataFrame()
        else:
            # Generate all unique pairs
            pairs_to_run = list(combinations(client_ids, 2))

        print(f"--- Starting OT Pipeline ---")
        print(f"Dataset: {dataset_name}, Total Clients in Run: {num_clients}, Target Round: {target_rounds}")
        print(f"Analyzing {len(pairs_to_run)} client pairs.")
        print(f"Costs/Alphas: {costs_list}")
        print(f"Num OT Configs per Pair/Cost: {len(ot_configurations)}")
        print("-" * 60)

        # --- Main Loop: Iterate through Costs first, then Pairs ---
        for i, cost in enumerate(costs_list):
            print(f"\nProcessing Cost/Alpha: {cost} ({i+1}/{len(costs_list)})...")
            # Load performance metrics once per cost (these are global)
            final_local_score, final_fedavg_score = self.data_manager.get_performance(
                dataset_name, cost, performance_aggregation
            )
            loss_delta = final_local_score - final_fedavg_score if np.isfinite(final_local_score) and np.isfinite(final_fedavg_score) else np.nan
            print(f"  Performance Context: Local={f'{final_local_score:.4f}' if np.isfinite(final_local_score) else 'NaN'}, FedAvg={f'{final_fedavg_score:.4f}' if np.isfinite(final_fedavg_score) else 'NaN'}, Delta={f'{loss_delta:.4f}' if np.isfinite(loss_delta) else 'NaN'}")

            # --- Loop through client pairs for this cost ---
            for j, pair in enumerate(pairs_to_run):
                cid1_str, cid2_str = pair[0], pair[1]
                print(f"  Processing Pair ({j+1}/{len(pairs_to_run)}): ({cid1_str}, {cid2_str})")
                # Use a consistent seed derived for this specific pair/cost?
                # Using base_seed for simplicity, assumes underlying FL run used base_seed
                current_seed = base_seed

                # --- Get Activations for the current PAIR ---
                activation_status = "Pending"
                processed_activations = None
                try:
                     processed_activations = self.data_manager.get_activations(
                         dataset_name=dataset_name, cost=cost, rounds=target_rounds, seed=current_seed,
                         client_id_1=cid1_str, client_id_2=cid2_str, # Specific pair
                         num_clients=num_clients, # Pass total clients in run context
                         num_classes=num_classes, loader_type=activation_loader_type,
                         force_regenerate=force_activation_regen
                     )
                     if processed_activations is not None: activation_status = "Loaded/Generated"
                     else: activation_status = "Activation Failed"
                except Exception as e:
                     activation_status = f"Activation Error: {type(e).__name__}"
                     warnings.warn(f"Error getting activations for cost {cost}, pair ({cid1_str}, {cid2_str}): {e}")
                     traceback.print_exc()

                # --- Run OT Calculations for the current PAIR ---
                if processed_activations is None:
                     print(f"    Skipping OT calculations for Pair ({cid1_str}, {cid2_str}) due to activation failure.")
                     for config in ot_configurations:
                          results_list.append(self._create_placeholder_row(cost, target_rounds, cid1_str, cid2_str, config, activation_status, final_local_score, final_fedavg_score, loss_delta))
                     continue # Move to next pair

                data1 = processed_activations.get(cid1_str)
                data2 = processed_activations.get(cid2_str)
                if data1 is None or data2 is None:
                     print(f"    Skipping OT calculations for Pair ({cid1_str}, {cid2_str}): Processed data missing.")
                     activation_status += " + Processing Failed"
                     for config in ot_configurations:
                          results_list.append(self._create_placeholder_row(cost, target_rounds, cid1_str, cid2_str, config, activation_status, final_local_score, final_fedavg_score, loss_delta))
                     continue

                # Calculate OT for each configuration for this pair
                for config in ot_configurations:
                    method_type = config.method_type
                    param_name = config.name
                    params = config.params
                    run_status = activation_status # Start with status from activation loading

                    # Base row data - ADDED Client_1 and Client_2
                    row_data = {
                        'Cost': cost, 'Round': target_rounds,
                        'Client_1': cid1_str, 'Client_2': cid2_str, # ADDED pair identifiers
                        'Param_Set_Name': param_name, 'OT_Method': method_type,
                        'Local_Final_Loss': final_local_score, 'FedAvg_Final_Loss': final_fedavg_score, 'Loss_Delta': loss_delta,
                        # Initialize OT result columns
                        'FeatureErrorOT_Cost': np.nan, 'FeatureErrorOT_Weighting': None,
                        'Decomposed_LabelEMD': np.nan, 'Decomposed_ConditionalOT': np.nan, 'Decomposed_CombinedScore': np.nan,
                        'FixedAnchor_TotalCost': np.nan, 'FixedAnchor_SimScore': np.nan,
                        'FixedAnchor_CostL1': np.nan, 'FixedAnchor_CostL2': np.nan, 'FixedAnchor_CostX': np.nan,
                    }

                    calculator = OTCalculatorFactory.create_calculator(
                        config=config, client_id_1=cid1_str, client_id_2=cid2_str, num_classes=num_classes
                    )
                    if calculator is None:
                        run_status += f" + Factory Failed"
                        row_data['Run_Status'] = run_status
                        results_list.append(row_data)
                        continue

                    try:
                        calculator.calculate_similarity(data1, data2, params)
                        ot_results = calculator.get_results()
                        self._populate_row_with_results(row_data, method_type, ot_results)
                        run_status += f" + Calc Success"
                    except Exception as e_calc:
                        warnings.warn(f"Cost {cost}, Pair ({cid1_str},{cid2_str}), Config '{param_name}': FAILED OT Calc - {type(e_calc).__name__}: {e_calc}", stacklevel=2)
                        traceback.print_exc()
                        run_status += f" + Calc Error: {type(e_calc).__name__}"
                    finally:
                        row_data['Run_Status'] = run_status
                        results_list.append(row_data)

                # Clear memory for the pair
                processed_activations = None; data1 = None; data2 = None

        # --- Final DataFrame ---
        print("\n--- OT Pipeline Finished ---")
        if not results_list:
            print("WARNING: No results were generated.")
            return pd.DataFrame()

        results_df = pd.DataFrame(results_list)
        return self._structure_output_dataframe(results_df) # Ensure new columns are handled


    # MODIFIED: Added client IDs to signature and row data
    def _create_placeholder_row(self, cost, round_val, client_id_1, client_id_2, config: OTConfig, status, local_loss, fedavg_loss, loss_delta):
        return {
            'Cost': cost, 'Round': round_val,
            'Client_1': client_id_1, 'Client_2': client_id_2, # ADDED
            'Param_Set_Name': config.name, 'OT_Method': config.method_type, 'Run_Status': status,
            'Local_Final_Loss': local_loss, 'FedAvg_Final_Loss': fedavg_loss, 'Loss_Delta': loss_delta,
            'FeatureErrorOT_Cost': np.nan, 'FeatureErrorOT_Weighting': None,
            'Decomposed_LabelEMD': np.nan, 'Decomposed_ConditionalOT': np.nan, 'Decomposed_CombinedScore': np.nan,
            'FixedAnchor_TotalCost': np.nan, 'FixedAnchor_SimScore': np.nan,
            'FixedAnchor_CostL1': np.nan, 'FixedAnchor_CostL2': np.nan, 'FixedAnchor_CostX': np.nan,
        }
    def _populate_row_with_results(self, row_data: Dict, method_type: str, ot_results: Dict):
         # (Implementation remains the same)
         if not ot_results: return
         if method_type == 'feature_error':
             row_data['FeatureErrorOT_Cost'] = ot_results.get('ot_cost', np.nan)
             row_data['FeatureErrorOT_Weighting'] = ot_results.get('weighting_used', None)
         elif method_type == 'decomposed':
             row_data['Decomposed_LabelEMD'] = ot_results.get('label_emd', np.nan)
             row_data['Decomposed_ConditionalOT'] = ot_results.get('conditional_ot_agg', np.nan)
             row_data['Decomposed_CombinedScore'] = ot_results.get('combined_score', np.nan)
         elif method_type == 'fixed_anchor':
             row_data['FixedAnchor_TotalCost'] = ot_results.get('total_cost', np.nan)
             row_data['FixedAnchor_SimScore'] = ot_results.get('similarity_score', np.nan)
             row_data['FixedAnchor_CostL1'] = ot_results.get('cost_local1', np.nan)
             row_data['FixedAnchor_CostL2'] = ot_results.get('cost_local2', np.nan)
             row_data['FixedAnchor_CostX'] = ot_results.get('cost_cross_anchor', np.nan)

    def _structure_output_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        cols_order = [
            'Cost', 'Round', 'Client_1', 'Client_2', # ADDED Client IDs early
            'Param_Set_Name', 'OT_Method', 'Run_Status',
            'Local_Final_Loss', 'FedAvg_Final_Loss', 'Loss_Delta',
            'FeatureErrorOT_Cost', 'FeatureErrorOT_Weighting',
            'Decomposed_CombinedScore', 'Decomposed_LabelEMD', 'Decomposed_ConditionalOT',
            'FixedAnchor_SimScore', 'FixedAnchor_TotalCost',
            'FixedAnchor_CostL1', 'FixedAnchor_CostL2', 'FixedAnchor_CostX'
        ]
        # Get columns present in the DataFrame, maintaining the desired order
        cols_present = [col for col in cols_order if col in df.columns]
        # Add any extra columns that might have been generated but aren't in the default order
        extra_cols = [col for col in df.columns if col not in cols_present]
        return df[cols_present + extra_cols]