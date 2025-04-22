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

fl_files=("run.py" "pipeline.py" "servers.py" "clients.py" "data_processing.py" "data_partitioning.py" "data_loading.py" "datasets.py" "helper.py" "configs.py")
fl_files=("results_manager.py" "pipeline.py" "data_processing.py" "data_partitioning.py" "data_loading.py" "datasets.py"  "configs.py")
ot_files=("ot_results_analysis.py" "ot_pipeline_runner.py" "ot_calculators.py" "ot_data_manager.py" "ot_utils.py")
data_files=("SyntheticOTCost.py" "dataCreator.py" "CreditOTCost.py")
if [[ "$1" == "--fl" ]]; then
    save_files fl_files[@] "fl_output.txt"
elif [[ "$1" == "--ot" ]]; then
    save_files ot_files[@] "ot_output.txt"

elif [[ "$1" == "--dc" ]]; then
    save_files data_files[@] "data_output.txt"
else
    save_files fl_files[@] "fl_output.txt"
    save_files ot_files[@] "ot_output.txt"
    save_files data_files[@] "data_output.txt"
fi
