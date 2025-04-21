#!/bin/bash

# List of target Python files
#files=("run.py" "pipeline.py" "data_processing.py" "configs.py")
files=("run.py" "pipeline.py" "servers.py" "clients.py" "data_processing.py" "helper.py" "configs.py")
#files=("ot_results_analysis.py" "ot_pipeline_runner.py" "ot_calculators.py" "ot_data_manager.py" "ot_utils.py")

# Output file
output_file="ot_output.txt"
> "$output_file"  # Clear the output file if it already exists

# Loop over each target file
for filename in "${files[@]}"; do
    # Find the file in the current directory or subdirectories
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
