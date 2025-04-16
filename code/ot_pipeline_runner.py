# pipeline_runner.py (Updated Imports Only)
import pandas as pd
import numpy as np
import warnings
from typing import List, Tuple, Dict, Any, Union, Optional

# Import DataManager and BaseOTCalculator (for type hinting)
from data_manager import DataManager
# Import BaseOTCalculator *AND* the new Config and Factory from ot_calculators.py
from ot_calculators import BaseOTCalculator, OTConfig, OTCalculatorFactory

class PipelineRunner:
    """ Orchestrates the OT similarity analysis pipeline. """

    # Factory logic is now in OTCalculatorFactory in ot_calculators.py

    def __init__(self, data_manager: DataManager):
        """ Initializes the runner with a DataManager instance. """
        self.data_manager = data_manager

    def run_pipeline(
        self,
        # ... (arguments remain the same, expecting List[OTConfig]) ...
        dataset_name: str,
        costs_list: List[float],
        target_rounds: int,
        client_id_1: Union[str, int],
        client_id_2: Union[str, int],
        num_classes: int,
        ot_configurations: List[OTConfig], # Expecting OTConfig objects
        activation_loader_type: str = 'val',
        performance_aggregation: str = 'mean',
        base_seed: int = 2,
        force_activation_regen: bool = False
    ) -> pd.DataFrame:
        # ... (Implementation remains the same as the previous version) ...
        # It already uses OTCalculatorFactory.create_calculator()
        # and expects OTConfig objects in the list.
        results_list = []
        cid1_str = str(client_id_1)
        cid2_str = str(client_id_2)

        if not ot_configurations:
            raise ValueError("Must provide at least one OT configuration (instance of OTConfig).")
        if not all(isinstance(cfg, OTConfig) for cfg in ot_configurations):
             raise TypeError("ot_configurations must be a list of OTConfig objects.")


        print(f"--- Starting OT Pipeline ---")
        print(f"Dataset: {dataset_name}, Rounds: {target_rounds}, Clients: {cid1_str} vs {cid2_str}")
        print(f"Costs: {costs_list}")
        print(f"Num OT Configs per Cost: {len(ot_configurations)}")
        print("-" * 60)

        for i, cost in enumerate(costs_list):
            print(f"\nProcessing Cost: {cost} (Round: {target_rounds}) ({i+1}/{len(costs_list)})...")
            current_seed = base_seed # Consistent seed per cost

            # --- 1. Load Performance ---
            final_local_score, final_fedavg_score = self.data_manager.get_performance(
                dataset_name, cost, performance_aggregation
            )
            loss_delta = final_fedavg_score - final_local_score if np.isfinite(final_local_score) and np.isfinite(final_fedavg_score) else np.nan
            print(f"  Loaded Performance (Agg: {performance_aggregation}): Local={f'{final_local_score:.4f}' if np.isfinite(final_local_score) else 'NaN'}, FedAvg={f'{final_fedavg_score:.4f}' if np.isfinite(final_fedavg_score) else 'NaN'}, Delta={f'{loss_delta:.4f}' if np.isfinite(loss_delta) else 'NaN'}")


            # --- 2. Get Activations ---
            activation_status = "Pending"
            processed_activations = None
            try:
                 processed_activations = self.data_manager.get_activations(
                     dataset_name=dataset_name, cost=cost, rounds=target_rounds, seed=current_seed,
                     client_id_1=cid1_str, client_id_2=cid2_str,
                     num_classes=num_classes, loader_type=activation_loader_type,
                     force_regenerate=force_activation_regen
                 )
                 if processed_activations is not None:
                     activation_status = "Loaded/Generated" # Could refine based on DataManager logs
                 else:
                      activation_status = "Activation Failed"
            except Exception as e:
                 activation_status = f"Activation Error: {type(e).__name__}"
                 warnings.warn(f"Error getting activations for cost {cost}: {e}")

            # --- 3. Run OT Calculations ---
            if processed_activations is None:
                print(f"  Skipping OT calculations for Cost {cost} due to activation failure.")
                # Add placeholder rows for this cost
                for config in ot_configurations:
                     results_list.append(self._create_placeholder_row(cost, target_rounds, config, activation_status, final_local_score, final_fedavg_score, loss_delta))
                continue # Move to next cost

            data1 = processed_activations[cid1_str]
            data2 = processed_activations[cid2_str]

            print(f"  Calculating similarities for {len(ot_configurations)} configurations...")
            for config in ot_configurations: # config is now an OTConfig object
                # Extract info from config object
                method_type = config.method_type
                param_name = config.name
                params = config.params
                run_status = activation_status

                # Base row data (uses config.name and config.method_type)
                row_data = {
                    'Cost': cost, 'Round': target_rounds, 'Param_Set_Name': param_name,
                    'OT_Method': method_type,
                    'Local_Final_Loss': final_local_score, 'FedAvg_Final_Loss': final_fedavg_score,
                    'Loss_Delta': loss_delta,
                    # Initialize potential result columns
                    'FeatureErrorOT_Cost': np.nan, 'FeatureErrorOT_Weighting': None,
                    'Decomposed_LabelEMD': np.nan, 'Decomposed_ConditionalOT': np.nan, 'Decomposed_CombinedScore': np.nan,
                    'FixedAnchor_TotalCost': np.nan, 'FixedAnchor_SimScore': np.nan,
                    'FixedAnchor_CostL1': np.nan, 'FixedAnchor_CostL2': np.nan, 'FixedAnchor_CostX': np.nan,
                     # Add other potential fields from calculators if needed
                }

                # --- Use the Factory to create the calculator ---
                # Factory is imported from ot_calculators now
                calculator = OTCalculatorFactory.create_calculator(
                    config=config,
                    client_id_1=cid1_str,
                    client_id_2=cid2_str,
                    num_classes=num_classes
                )

                if calculator is None:
                    # Warning already issued by factory
                    run_status += f" + Factory Failed"
                    row_data['Run_Status'] = run_status
                    results_list.append(row_data)
                    continue

                try:
                    # Run calculation (pass params from config)
                    calculator.calculate_similarity(data1, data2, params)
                    ot_results = calculator.get_results()

                    # Populate row_data with results
                    self._populate_row_with_results(row_data, method_type, ot_results)
                    run_status += f" + Calc Success"

                except Exception as e_calc:
                    warnings.warn(f"Cost {cost}, Config '{param_name}': FAILED OT Calc - {type(e_calc).__name__}: {e_calc}", stacklevel=2)
                    run_status += f" + Calc Error: {type(e_calc).__name__}"
                finally:
                    row_data['Run_Status'] = run_status
                    results_list.append(row_data)

            # Optional: Clear activations from memory if large
            processed_activations = None
            data1 = None
            data2 = None
            # cleanup_gpu() # If relevant

        # --- Final DataFrame ---
        print("\n--- Pipeline Finished ---")
        if not results_list:
            print("WARNING: No results were generated.")
            return pd.DataFrame()

        results_df = pd.DataFrame(results_list)
        return self._structure_output_dataframe(results_df)


    def _create_placeholder_row(self, cost, round_val, config: OTConfig, status, local_loss, fedavg_loss, loss_delta):
        # (Implementation remains the same - uses config.name, config.method_type)
        return {
            'Cost': cost, 'Round': round_val, 'Param_Set_Name': config.name,
            'OT_Method': config.method_type, 'Run_Status': status,
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
         # (Implementation remains the same)
         cols_order = [
             'Cost', 'Round', 'Param_Set_Name', 'OT_Method', 'Run_Status',
             'Local_Final_Loss', 'FedAvg_Final_Loss', 'Loss_Delta',
             'FeatureErrorOT_Cost', 'FeatureErrorOT_Weighting',
             'Decomposed_CombinedScore', 'Decomposed_LabelEMD', 'Decomposed_ConditionalOT',
             'FixedAnchor_SimScore', 'FixedAnchor_TotalCost',
             'FixedAnchor_CostL1', 'FixedAnchor_CostL2', 'FixedAnchor_CostX'
         ]
         cols_present = [col for col in cols_order if col in df.columns]
         return df[cols_present]