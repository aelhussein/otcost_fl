#!/bin/bash

save_files() {
    local file_list=("${!1}")
    local output_file="$2"
    > "$output_file"  # Clear the output file

    for filename in "${file_list[@]}"; do
        found_path=$(find . -type f -name "$filename" | head -n 1)

        if [[ -n "$found_path" ]]; then
            echo "<START OF ${filename}>" >> "$output_file"
            cat "$found_path" >> "$output_file"
            echo "" >> "$output_file"
            echo "<END OF ${filename}>" >> "$output_file"
            echo "" >> "$output_file"
        else
            echo "File not found: $filename"
        fi
    done
}

fl_files=("submit_evaluation.sh" "run.py" "pipeline.py" "servers.py" "clients.py" "models.py" "losses.py" "data_processing.py" "data_partitioning.py" "data_loading.py" "data_sets.py" "synthetic_data.py" "helper.py" "configs.py")
eval_files=("pipeline.py" "servers.py" "clients.py" "helper.py" "models.py" "losses.py" "configs.py")
data_files=("pipeline.py" "data_processing.py" "data_partitioning.py" "data_loading.py" "data_sets.py" "synthetic_data.py"  "configs.py")
ot_files=("ot_results_analysis.py" "ot_pipeline_runner.py" "ot_calculators.py" "ot_data_manager.py" "ot_utils.py" "ot_configs.py" "models.py")

save_files fl_files[@] "code_fl.txt"
save_files ot_files[@] "code_ot.txt"
save_files data_files[@] "code_data.txt"
save_files eval_files[@] "code_eval.txt"

