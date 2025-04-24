from configs import ACTIVATION_DIR
import pandas as pd
import numpy as np
import traceback
from itertools import combinations
from typing import List, Tuple, Dict, Any, Union, Optional
from ot_data_manager import OTDataManager as DataManager
# Import OT calculator related classes
from ot_calculators import OTConfig, OTCalculatorFactory

class PipelineRunner:
    """ Orchestrates the OT similarity analysis pipeline for multiple client pairs. """

    def __init__(self,
                 num_clients: int,
                 activation_dir: Optional[str] = None):
        """
        Initializes the pipeline runner.

        Args:
            num_clients (int): The total number of clients in the FL run to analyze pairs from
            activation_dir (str, optional): Path to activation cache. Defaults to config path.
        """
        self.num_clients = num_clients
        act_dir = activation_dir if activation_dir else ACTIVATION_DIR
        
        # Initialize DataManager with target client count
        self.data_manager = DataManager(
            num_clients=num_clients,
            activation_dir=act_dir
        )
        print(f"Pipeline runner initialized for {num_clients} clients")

    def run_pipeline(
        self,
        dataset_name: str,
        costs_list: List[Any],
        num_classes: int,
        ot_configurations: List[OTConfig],
        client_pairs_to_analyze: Optional[List[Tuple[Union[str, int], Union[str, int]]]] = None,
        activation_loader_type: str = 'val',
        performance_aggregation: str = 'mean',
        base_seed: int = 42,
        force_activation_regen: bool = False
    ) -> pd.DataFrame:
        """
        Runs the full OT analysis pipeline.

        Args:
            dataset_name (str): Name of the dataset.
            costs_list (List[Any]): List of cost parameters (e.g., alpha values).
            num_classes (int): Number of classes in the dataset.
            ot_configurations (List[OTConfig]): List of OTConfig objects.
            client_pairs_to_analyze (List[Tuple], optional): Specific client pairs.
            activation_loader_type (str): Dataloader type ('train', 'val', 'test').
            performance_aggregation (str): Aggregation method ('mean', 'median').
            base_seed (int): Base random seed for the FL run.
            force_activation_regen (bool): Force regeneration of activations.

        Returns:
            pd.DataFrame: DataFrame with OT results for each pair, cost, and config.
        """
        results_list = []

        # Validation
        if not ot_configurations:
            raise ValueError("Must provide at least one OT configuration.")
        if not all(isinstance(cfg, OTConfig) for cfg in ot_configurations):
            raise TypeError("ot_configurations must be a list of OTConfig objects.")

        # Generate client pairs
        client_ids = [f'client_{i+1}' for i in range(self.num_clients)]
        
        if client_pairs_to_analyze:
            # Validate provided pairs
            valid_pairs = []
            for p in client_pairs_to_analyze:
                if isinstance(p, tuple) and len(p) == 2 and \
                   str(p[0]) in client_ids and str(p[1]) in client_ids and p[0] != p[1]:
                    valid_pairs.append((str(p[0]), str(p[1])))
                else:
                    print(f"Invalid client pair provided: {p}. Skipping.")
            pairs_to_run = valid_pairs
            
            if not pairs_to_run:
                print("No valid client pairs specified to analyze.")
                return pd.DataFrame()
        else:
            # Generate all unique pairs
            pairs_to_run = list(combinations(client_ids, 2))

        print(f"--- Starting OT Pipeline ---")
        print(f"Dataset: {dataset_name}, Total Clients: {self.num_clients}")
        print(f"Analyzing {len(pairs_to_run)} client pairs")
        print(f"Costs/Alphas: {costs_list}")
        print(f"OT Configurations: {len(ot_configurations)}")
        print("-" * 60)

        # Main loop: Iterate through costs first, then pairs
        for i, cost in enumerate(costs_list):
            print(f"\nProcessing Cost/Alpha: {cost} ({i+1}/{len(costs_list)})...")
            
            # Load performance metrics for this cost
            final_local_score, final_fedavg_score = self.data_manager.get_performance(
                dataset_name, cost, performance_aggregation
            )
            
            loss_delta = final_local_score - final_fedavg_score if np.isfinite(final_local_score) and np.isfinite(final_fedavg_score) else np.nan
            
            print(f"  Performance: Local={f'{final_local_score:.4f}' if np.isfinite(final_local_score) else 'NaN'}, "
                  f"FedAvg={f'{final_fedavg_score:.4f}' if np.isfinite(final_fedavg_score) else 'NaN'}, "
                  f"Delta={f'{loss_delta:.4f}' if np.isfinite(loss_delta) else 'NaN'}")
            
            # Process each client pair
            for j, pair in enumerate(pairs_to_run):
                cid1_str, cid2_str = pair
                print(f"  Processing Pair ({j+1}/{len(pairs_to_run)}): ({cid1_str}, {cid2_str})")
                
                # Using base_seed directly
                current_seed = base_seed
                
                # Get activations - note we no longer pass rounds parameter
                activation_status = "Pending"
                processed_activations = None
                
                try:
                    processed_activations = self.data_manager.get_activations(
                        dataset_name=dataset_name,
                        cost=cost, 
                        seed=current_seed,
                        client_id_1=cid1_str,
                        client_id_2=cid2_str,
                        num_clients=self.num_clients,
                        num_classes=num_classes,
                        loader_type=activation_loader_type,
                        force_regenerate=force_activation_regen
                    )
                    
                    activation_status = "Loaded/Generated" if processed_activations else "Activation Failed"
                except Exception as e:
                    activation_status = f"Activation Error: {type(e).__name__}"
                    print(f"Error getting activations for cost {cost}, pair ({cid1_str}, {cid2_str}): {e}")
                    traceback.print_exc()
                
                # Skip OT calculations if activation failed
                if processed_activations is None:
                    print(f"    Skipping OT calculations - activation failed")
                    for config in ot_configurations:
                        results_list.append(self._create_placeholder_row(
                            cost, cid1_str, cid2_str, config,
                            activation_status, final_local_score, final_fedavg_score, loss_delta
                        ))
                    continue
                
                # Get client data
                data1 = processed_activations.get(cid1_str)
                data2 = processed_activations.get(cid2_str)
                
                if data1 is None or data2 is None:
                    print(f"    Skipping OT calculations - processed data missing")
                    activation_status += " + Processing Failed"
                    for config in ot_configurations:
                        results_list.append(self._create_placeholder_row(
                            cost, cid1_str, cid2_str, config,
                            activation_status, final_local_score, final_fedavg_score, loss_delta
                        ))
                    continue
                
                # Run OT calculations for each configuration
                for config in ot_configurations:
                    method_type = config.method_type
                    param_name = config.name
                    params = config.params
                    run_status = activation_status
                    
                    # Create base row data - removed Round column
                    row_data = {
                        'Cost': cost,
                        'Client_1': cid1_str,
                        'Client_2': cid2_str,
                        'Param_Set_Name': param_name,
                        'OT_Method': method_type,
                        'Local_Final_Loss': final_local_score,
                        'FedAvg_Final_Loss': final_fedavg_score,
                        'Loss_Delta': loss_delta,
                        # Initialize OT result columns
                        'FeatureErrorOT_Cost': np.nan,
                        'FeatureErrorOT_Weighting': None,
                        'Decomposed_LabelEMD': np.nan,
                        'Decomposed_ConditionalOT': np.nan,
                        'Decomposed_CombinedScore': np.nan,
                        'FixedAnchor_TotalCost': np.nan,
                        'FixedAnchor_SimScore': np.nan,
                        'FixedAnchor_CostL1': np.nan,
                        'FixedAnchor_CostL2': np.nan,
                        'FixedAnchor_CostX': np.nan,
                    }
                    
                    # Create and run calculator
                    calculator = OTCalculatorFactory.create_calculator(
                        config=config,
                        client_id_1=cid1_str,
                        client_id_2=cid2_str,
                        num_classes=num_classes
                    )
                    
                    if calculator is None:
                        run_status += " + Factory Failed"
                        row_data['Run_Status'] = run_status
                        results_list.append(row_data)
                        continue
                    
                    try:
                        calculator.calculate_similarity(data1, data2, params)
                        ot_results = calculator.get_results()
                        self._populate_row_with_results(row_data, method_type, ot_results)
                        run_status += " + Calc Success"
                    except Exception as e:
                        print(f"OT calculation failed for {param_name}: {e}")
                        traceback.print_exc()
                        run_status += f" + Calc Error: {type(e).__name__}"
                    finally:
                        row_data['Run_Status'] = run_status
                        results_list.append(row_data)
                
                # Clean up memory
                processed_activations = None
                data1 = None
                data2 = None
        
        # Create final DataFrame
        print("\n--- OT Pipeline Finished ---")
        if not results_list:
            print("WARNING: No results were generated")
            return pd.DataFrame()
        
        results_df = pd.DataFrame(results_list)
        return self._structure_output_dataframe(results_df)

    def _create_placeholder_row(self, cost, client_id_1, client_id_2, config: OTConfig, 
                              status, local_loss, fedavg_loss, loss_delta):
        """Creates a placeholder result row when OT calculation fails."""
        return {
            'Cost': cost,
            'Client_1': client_id_1,
            'Client_2': client_id_2,
            'Param_Set_Name': config.name,
            'OT_Method': config.method_type,
            'Run_Status': status,
            'Local_Final_Loss': local_loss,
            'FedAvg_Final_Loss': fedavg_loss,
            'Loss_Delta': loss_delta,
            'FeatureErrorOT_Cost': np.nan,
            'FeatureErrorOT_Weighting': None,
            'Decomposed_LabelEMD': np.nan,
            'Decomposed_ConditionalOT': np.nan,
            'Decomposed_CombinedScore': np.nan,
            'FixedAnchor_TotalCost': np.nan,
            'FixedAnchor_SimScore': np.nan,
            'FixedAnchor_CostL1': np.nan,
            'FixedAnchor_CostL2': np.nan,
            'FixedAnchor_CostX': np.nan,
        }

    def _populate_row_with_results(self, row_data: Dict, method_type: str, ot_results: Dict):
        """Populates a row with OT calculation results."""
        if not ot_results:
            return
            
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
        """Structures the output DataFrame with consistent column ordering."""
        cols_order = [
            'Cost', 'Client_1', 'Client_2',  # Removed 'Round'
            'Param_Set_Name', 'OT_Method', 'Run_Status',
            'Local_Final_Loss', 'FedAvg_Final_Loss', 'Loss_Delta',
            'FeatureErrorOT_Cost', 'FeatureErrorOT_Weighting',
            'Decomposed_CombinedScore', 'Decomposed_LabelEMD', 'Decomposed_ConditionalOT',
            'FixedAnchor_SimScore', 'FixedAnchor_TotalCost',
            'FixedAnchor_CostL1', 'FixedAnchor_CostL2', 'FixedAnchor_CostX'
        ]
        
        # Get columns present in the DataFrame, maintaining the desired order
        cols_present = [col for col in cols_order if col in df.columns]
        
        # Add any extra columns that might have been generated
        extra_cols = [col for col in df.columns if col not in cols_present]
        
        return df[cols_present + extra_cols]