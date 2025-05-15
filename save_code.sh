#!/bin/bash

save_files() {
    local file_list_array_name="$1" # Name of the array variable
    local -n file_list="$file_list_array_name" # Dereference using nameref
    local output_file="$2"
    > "$output_file"  # Clear the output file

    # Define the search directories relative to script execution (current directory)
    local search_dirs=("pipeline_tools" "code")

    for filename_basename in "${file_list[@]}"; do
        local found_path=""
        # Try to find the file in the specified directories
        found_path=$(find "${search_dirs[@]}" -path "*/$filename_basename" -type f -print -quit 2>/dev/null)

        if [[ -n "$found_path" ]]; then
            # Process found_path for display
            local display_path_intermediate="${found_path#./}"

            IFS='/' read -ra path_parts <<< "$display_path_intermediate"
            local num_path_parts=${#path_parts[@]}
            local structured_display_path_parts=()

            if [[ $num_path_parts -eq 0 ]]; then
                structured_display_path_str="$filename_basename (path error)" # Should not happen
            else
                local first_dir_component="${path_parts[0]}"
                local filename_component="${path_parts[-1]}"
                local num_intermediate_dirs=$((num_path_parts - 2))

                structured_display_path_parts+=("$first_dir_component")

                if [[ $num_intermediate_dirs -lt 0 ]]; then
                    : # Handled by first_dir_component being the filename if num_path_parts was 1
                elif [[ $num_intermediate_dirs -eq 0 ]]; then # e.g. code/file.py
                    if [[ "$first_dir_component" != "$filename_component" ]]; then
                         structured_display_path_parts+=("$filename_component")
                    fi
                elif [[ $num_intermediate_dirs -le 3 ]]; then # 1 to 3 intermediate dirs
                    for (( i=1; i < num_path_parts -1; i++ )); do
                        structured_display_path_parts+=("${path_parts[i]}")
                    done
                    structured_display_path_parts+=("$filename_component")
                else # More than 3 intermediate dirs
                    for (( i=1; i <= 3; i++ )); do
                        structured_display_path_parts+=("${path_parts[i]}")
                    done
                    structured_display_path_parts+=("...") # Add ellipsis
                    structured_display_path_parts+=("$filename_component")
                fi
            fi
            
            # Join parts with "/" to form the display string
            local temp_ifs="$IFS"
            IFS="/" # Changed separator
            local structured_display_path_str="${structured_display_path_parts[*]}"
            IFS="$temp_ifs"

            echo "<START OF ${structured_display_path_str}>" >> "$output_file"
            cat "$found_path" >> "$output_file"
            echo "" >> "$output_file"
            echo "<END OF ${structured_display_path_str}>" >> "$output_file"
            echo "" >> "$output_file"
        else
            echo "File not found: $filename_basename in specified directories (${search_dirs[*]})" >> "$output_file"
            echo "" >> "$output_file"
            echo "File not found: $filename_basename in specified directories (${search_dirs[*]})" >&2
        fi
    done
}

# Define file lists (arrays)
fl_files=("submit_evaluation.sh" "run.py" "results_utils.py" "results_manager.py" "pipeline.py" "servers.py" "clients.py" "models.py" "losses.py" "data_processing.py" "data_partitioning.py" "data_loading.py" "data_sets.py" "synthetic_data.py" "helper.py" "configs.py")
eval_files=("submit_evaluation.sh" "run.py" "results_manager.py" "pipeline.py" "servers.py" "clients.py" "helper.py" "models.py" "losses.py" "configs.py")
data_files=("pipeline.py" "data_processing.py" "data_partitioning.py" "data_loading.py" "data_sets.py" "synthetic_data.py"  "configs.py")
ot_files=("submit_ot_analysis.sh" "results_manager.py" "results_utils.py" "run_ot_analysis.py" "ot_pipeline_runner.py" "ot_calculators.py" "ot_data_manager.py" "ot_utils.py" "ot_configs.py" "helper.py" "models.py")
pipeline_files=("orchestrate_all.py" "orchestrate.py" "status.py" "submit_ot_analysis.sh" "submit_evaluation.sh" "pipeline.py" "configs.py" "results_manager.py" "results_utils.py" "helper.py")


fl_files=("run.py" "results_utils.py" "results_manager.py" "pipeline.py" "servers.py" "clients.py" "models.py" "losses.py" "data_processing.py" "data_partitioning.py" "data_loading.py" "data_sets.py" "synthetic_data.py" "helper.py" "configs.py")
eval_files=("run.py" "results_manager.py" "pipeline.py" "servers.py" "clients.py" "helper.py" "models.py" "losses.py" "configs.py")
data_files=("pipeline.py" "data_processing.py" "data_partitioning.py" "data_loading.py" "data_sets.py" "synthetic_data.py"  "configs.py")
ot_files=("run_ot_analysis.py" "ot_pipeline_runner.py" "ot_calculators.py" "ot_data_manager.py" "ot_utils.py" "ot_configs.py")
pipeline_files=("orchestrate_all.py" "orchestrate.py" "status.py" "submit_ot_analysis.sh" "submit_evaluation.sh")


# Call save_files for each list, passing the array name as a string
save_files "fl_files" "code_fl.txt"
save_files "ot_files" "code_ot.txt"
save_files "data_files" "code_data.txt"
save_files "eval_files" "code_eval.txt"
save_files "pipeline_files" "code_pipeline.txt"

echo "Processing complete. Output files generated."