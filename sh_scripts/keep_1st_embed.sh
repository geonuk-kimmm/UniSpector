#!/bin/bash

# Usage: ./keep_1st_embed.sh <base_path> [dataset_name]
# Example: ./keep_1st_embed.sh exp_cvpr26_FINAL/seed777_ours_gumbel1.0_0.2Noise/exp_model_0019999.pth MVTec_unseen_seed777_prompt_seed82
# This will process all results82_angle* directories

# Check if path is provided
if [ $# -lt 1 ]; then
    echo "Usage: $0 <base_path> [dataset_name]"
    echo "Example: $0 exp_cvpr26_FINAL/seed777_ours_gumbel1.0_0.2Noise/exp_model_0019999.pth MVTec_unseen_seed777_prompt_seed82"
    echo ""
    echo "This script will process all 'results*_angle*' directories under the base path."
    exit 1
fi

BASE_PATH="$1"
DATASET_NAME="${2:-}"

# Check if base path exists
if [ ! -d "$BASE_PATH" ]; then
    echo "Error: Directory not found: $BASE_PATH"
    exit 1
fi

echo "=== Processing all angle directories ==="
echo "Base path: $BASE_PATH"
if [ -n "$DATASET_NAME" ]; then
    echo "Dataset name: $DATASET_NAME"
fi
echo ""

# Find all angle directories (results*_angle*)
angle_dirs=$(find "$BASE_PATH" -maxdepth 1 -type d -name "results*_angle*" | sort)

if [ -z "$angle_dirs" ]; then
    echo "Error: No angle directories found in $BASE_PATH"
    echo "Looking for directories matching pattern: results*_angle*"
    exit 1
fi

# Process each angle directory
for angle_dir in $angle_dirs; do
    angle_name=$(basename "$angle_dir")
    echo "=========================================="
    echo "Processing: $angle_name"
    echo "=========================================="
    
    # Determine the target path
    if [ -n "$DATASET_NAME" ]; then
        target_path="$angle_dir/$DATASET_NAME"
    else
        # If dataset name not provided, find the first subdirectory that looks like a dataset
        target_path=$(find "$angle_dir" -maxdepth 1 -type d ! -path "$angle_dir" | head -1)
        if [ -z "$target_path" ]; then
            echo "  No dataset directory found, skipping..."
            echo ""
            continue
        fi
    fi
    
    # Check if target path exists
    if [ ! -d "$target_path" ]; then
        echo "  Dataset directory not found: $target_path"
        echo ""
        continue
    fi
    
    echo "  Target path: $target_path"
    echo ""
    
    # Iterate over each class folder
    for class_dir in "$target_path"/*; do
        # Skip if not a directory
        [ -d "$class_dir" ] || continue
        
        class_name=$(basename "$class_dir")
        echo "  Processing class folder: $class_name"
        
        # Find only safetensor files
        safetensor_files=("$class_dir"/*.safetensors)
        
        # Skip if no safetensor files exist
        if [ ! -f "${safetensor_files[0]}" ]; then
            echo "    No safetensor files found, skipping..."
            continue
        fi
        
        # Find the first safetensor file (first after sorting)
        first_file=$(ls "$class_dir"/*.safetensors 2>/dev/null | sort | head -n 1)
        
        if [ -z "$first_file" ]; then
            echo "    No safetensor files found, skipping..."
            continue
        fi
        
        first_file_name=$(basename "$first_file")
        echo "    Keeping first file: $first_file_name"
        
        # Keep only the first file and delete the rest
        deleted_count=0
        for f in "$class_dir"/*.safetensors; do
            if [ -f "$f" ]; then
                base=$(basename "$f")
                if [[ "$base" != "$first_file_name" ]]; then
                    echo "      Deleting: $base"
                    rm -f "$f"
                    deleted_count=$((deleted_count + 1))
                fi
            fi
        done
        
        echo "    Deleted $deleted_count file(s)"
    done
    
    echo ""
done

echo "=== Done! ==="