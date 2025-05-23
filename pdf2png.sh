#!/bin/bash

main_dir="$1"  # Get the main directory as the first argument

if [ -z "$main_dir" ]; then
  echo "Usage: $0 <main_directory>"
  exit 1
fi

find "$main_dir" -type f -name "*.pdf" -print0 | while IFS= read -r -d $'\0' pdf_file; do
  # Extract the directory of the PDF file
  pdf_dir=$(dirname "$pdf_file")

  # Extract the filename without the extension
  pdf_name=$(basename "$pdf_file" .pdf)

  # Construct the output PNG filename
  png_file="$pdf_dir/${pdf_name}"

  echo "Converting '$pdf_file' to '$png_file'..."
  pdftoppm -png "$pdf_file" "$png_file"

  if [ $? -ne 0 ]; then
    echo "Error converting '$pdf_file'."
  fi
done

echo "Conversion process completed."