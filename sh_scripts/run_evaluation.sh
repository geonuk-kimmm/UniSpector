#!/bin/bash

# Usage: bash sh_scripts/run_evaluation.sh <model_seed> <model_weights> <savedir> <config_file> <csv_file>
# Example: bash sh_scripts/run_evaluation.sh 42 exp/base/model_0019999.pth base configs/InsA_base.yaml exp/base/results.csv
# CUDA_VISIBLE_DEVICES=0,1 bash sh_scripts/run_evaluation.sh 777 exp/base/model_0019999.pth base configs/InsA_base.yaml exp/base/results.csv


# This script automatically evaluates all datasets:
# - In-domain: GC10, MagneticTile, Real-IAD, MVTec (4 datasets × 3 seeds = 12 files)
# - Out-domain: 3CAD, CVPRW, VisA (3 datasets × 3 seeds = 9 files)
# Total: 21 evaluation runs

set -e  # Exit on error

# Check if required arguments are provided
if [ $# -lt 5 ]; then
    echo "Usage: $0 <model_seed> <model_weights> <savedir> <config_file> <csv_file>"
    echo "Example: $0 42 exp/base/model_0019999.pth base configs/InsA_base.yaml exp/base/results.csv"
    echo ""
    echo "Arguments:"
    echo "  model_seed:    Training seed (XXX) used in in-domain JSON paths"
    echo "  model_weights: Path to model weights file"
    echo "  savedir:       Save directory name"
    echo "  config_file:   Config YAML path"
    echo "  csv_file:      CSV output path"
    exit 1
fi

MODEL_SEED=$1
MODEL_WEIGHTS=$2
SAVEDIR=$3
CONFIG_FILE=$4
CSV_FILE=$5

# Base paths
BASE_PATH="ROOTPATH_ANNOTATION"
IN_DOMAIN_BASE="${BASE_PATH}/in-domain/unseen"
OUT_DOMAIN_BASE="${BASE_PATH}/out-domain"

# Dataset configurations
IN_DOMAIN_DATASETS=("GC10" "MagneticTile" "RealIAD" "MVTec")
OUT_DOMAIN_DATASETS=("3CAD" "VISION" "VisA")
PROMPT_SEEDS=(42 82 777)

# Generate OUTPUT_DIR from MODEL_WEIGHTS and savedir
MODEL_FILENAME=$(basename "$MODEL_WEIGHTS")
OUTPUT_DIR_BASE=$(dirname "$MODEL_WEIGHTS")/exp_${MODEL_FILENAME}/${SAVEDIR}

echo "=== Batch Evaluation Configuration ==="
echo "Model Seed: $MODEL_SEED"
echo "Model Weights: $MODEL_WEIGHTS"
echo "Save Dir: $SAVEDIR"
echo "Config File: $CONFIG_FILE"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not set (will use default)}"
echo "Output Dir Base: $OUTPUT_DIR_BASE"
echo "CSV Output: $CSV_FILE"
echo "======================================"

# Retry configuration
MAX_RETRIES=3
RETRY_DELAY=5  # seconds to wait before retry

# Create CSV header if file doesn't exist
if [ ! -f "$CSV_FILE" ]; then
    echo "Model_Seed,Model_Path,Dataset,Prompt_Seed,bbox_AP50,mask_AP50" > "$CSV_FILE"
    echo "Created new CSV file: $CSV_FILE"
    SKIP_COMPLETED=true
else
    echo "Found existing CSV file: $CSV_FILE"
    SKIP_COMPLETED=true
fi

# Function to check if an entry exists and is valid (not EVAL_ERROR, NOT_FOUND, FILE_NOT_FOUND)
entry_exists_and_valid() {
    local dataset=$1
    local prompt_seed=$2
    local is_in_domain=$3
    
    if [ ! -f "$CSV_FILE" ]; then
        return 1
    fi
    
    local dataset_key
    if [ "$is_in_domain" = "true" ]; then
        dataset_key="${dataset}_unseen"
    else
        dataset_key="$dataset"
    fi
    
    python3 << EOF
import csv
import sys

csv_file = "$CSV_FILE"
model_seed = "$MODEL_SEED"
model_weights = "$MODEL_WEIGHTS"
dataset_key = "$dataset_key"
prompt_seed = "$prompt_seed"

try:
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (row['Model_Seed'] == model_seed and 
                row['Model_Path'] == model_weights and
                row['Dataset'] == dataset_key and
                row['Prompt_Seed'] == prompt_seed):
                # Check if the entry is valid (not an error)
                bbox = row['bbox_AP50']
                mask = row['mask_AP50']
                if (bbox not in ['EVAL_ERROR', 'NOT_FOUND', 'FILE_NOT_FOUND', 'EXTRACTION_ERROR'] and
                    mask not in ['EVAL_ERROR', 'NOT_FOUND', 'FILE_NOT_FOUND', 'EXTRACTION_ERROR']):
                    sys.exit(0)  # Valid entry found
    sys.exit(1)  # No valid entry found
except Exception:
    sys.exit(1)
EOF
}

# Function to check if an entry has EVAL_ERROR in CSV
has_eval_error() {
    local dataset=$1
    local prompt_seed=$2
    local is_in_domain=$3
    
    if [ ! -f "$CSV_FILE" ]; then
        return 1
    fi
    
    local dataset_key
    if [ "$is_in_domain" = "true" ]; then
        dataset_key="${dataset}_unseen"
    else
        dataset_key="$dataset"
    fi
    
    python3 << EOF
import csv
import sys

csv_file = "$CSV_FILE"
model_seed = "$MODEL_SEED"
model_weights = "$MODEL_WEIGHTS"
dataset_key = "$dataset_key"
prompt_seed = "$prompt_seed"

try:
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (row['Model_Seed'] == model_seed and 
                row['Model_Path'] == model_weights and
                row['Dataset'] == dataset_key and
                row['Prompt_Seed'] == prompt_seed and
                (row['bbox_AP50'] == 'EVAL_ERROR' or row['mask_AP50'] == 'EVAL_ERROR')):
                sys.exit(0)
    sys.exit(1)
except Exception:
    sys.exit(1)
EOF
}

# Function to update CSV entry
update_csv_entry() {
    local dataset=$1
    local prompt_seed=$2
    local bbox_ap50=$3
    local mask_ap50=$4
    local is_in_domain=$5
    
    local dataset_key
    if [ "$is_in_domain" = "true" ]; then
        dataset_key="${dataset}_unseen"
    else
        dataset_key="$dataset"
    fi
    
    python3 << EOF
import csv
import sys
import tempfile
import shutil

csv_file = "$CSV_FILE"
model_seed = "$MODEL_SEED"
model_weights = "$MODEL_WEIGHTS"
dataset_key = "$dataset_key"
prompt_seed = "$prompt_seed"
bbox_ap50 = "$bbox_ap50"
mask_ap50 = "$mask_ap50"

try:
    # Create temporary file
    temp_file = csv_file + '.tmp'
    
    with open(csv_file, 'r') as f_in, open(temp_file, 'w', newline='') as f_out:
        reader = csv.DictReader(f_in)
        fieldnames = reader.fieldnames
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        
        updated = False
        for row in reader:
            if (row['Model_Seed'] == model_seed and 
                row['Model_Path'] == model_weights and
                row['Dataset'] == dataset_key and
                row['Prompt_Seed'] == prompt_seed):
                # Update this row
                row['bbox_AP50'] = bbox_ap50
                row['mask_AP50'] = mask_ap50
                updated = True
            writer.writerow(row)
        
        if not updated:
            # Add new row if not found
            writer.writerow({
                'Model_Seed': model_seed,
                'Model_Path': model_weights,
                'Dataset': dataset_key,
                'Prompt_Seed': prompt_seed,
                'bbox_AP50': bbox_ap50,
                'mask_AP50': mask_ap50
            })
    
    # Replace original file
    shutil.move(temp_file, csv_file)
    sys.exit(0)
except Exception as e:
    print(f"Error updating CSV: {e}", file=sys.stderr)
    sys.exit(1)
EOF
}

# Function to run evaluation with retry
run_evaluation_with_retry() {
    local json_file=$1
    local model_weights=$2
    local savedir=$3
    local dataset_name=$4
    local retry_count=0
    local success=false
    
    while [ $retry_count -lt $MAX_RETRIES ]; do
        if [ $retry_count -gt 0 ]; then
            echo "Retry attempt $retry_count/$MAX_RETRIES for $dataset_name..."
            echo "Waiting ${RETRY_DELAY} seconds before retry..."
            sleep $RETRY_DELAY
        fi
        
        echo "Running evaluation (attempt $((retry_count + 1))/$MAX_RETRIES)..."
        # Use CUDA_VISIBLE_DEVICES from environment (inherited from parent process)
        # If not set, it will use the default in eval_openset_only.sh
        if bash sh_scripts/eval_openset_only.sh "$json_file" "$model_weights" "$savedir" "$CONFIG_FILE"; then
            success=true
            break
        else
            retry_count=$((retry_count + 1))
            echo "Evaluation failed. Retry count: $retry_count/$MAX_RETRIES"
        fi
    done
    
    if [ "$success" = true ]; then
        return 0
    else
        echo "Error: Evaluation failed after $MAX_RETRIES attempts for $dataset_name"
        return 1
    fi
}

# Function to extract metrics from evaluation results
extract_metrics() {
    local eval_json=$1
    local dataset_name=$2
    local prompt_seed=$3
    
    if [ ! -f "$eval_json" ]; then
        echo "ERROR,ERROR"
        return
    fi
    
    python3 << EOF
import json
import sys

try:
    with open("$eval_json", 'r') as f:
        data = json.load(f)
    
    bbox_ap50 = data.get('bbox', {}).get('AP50', 'N/A')
    mask_ap50 = data.get('segm', {}).get('AP50', 'N/A')
    
    print(f"{bbox_ap50},{mask_ap50}")
    
except Exception as e:
    print(f"ERROR,ERROR")
    sys.exit(1)
EOF
}

# Process in-domain datasets
echo ""
echo "=========================================="
echo "Processing In-Domain Datasets"
echo "=========================================="

for dataset in "${IN_DOMAIN_DATASETS[@]}"; do
    for prompt_seed in "${PROMPT_SEEDS[@]}"; do
        # Construct JSON file path
        JSON_FILE="${IN_DOMAIN_BASE}/${dataset}/prompt/${dataset}_unseen_seed${MODEL_SEED}_prompt_seed${prompt_seed}.json"
        
        # Extract dataset name for display
        DATASET_NAME="${dataset}_unseen_seed${MODEL_SEED}_prompt_seed${prompt_seed}"
        
        # Check if we should skip this evaluation (skip if valid entry exists)
        if [ "$SKIP_COMPLETED" = "true" ]; then
            if entry_exists_and_valid "$dataset" "$prompt_seed" "true"; then
                echo "Skipping $DATASET_NAME (valid entry already exists in CSV)"
                continue
            fi
        fi
        
        echo ""
        echo "Processing: $DATASET_NAME"
        echo "JSON File: $JSON_FILE"
        
        if [ ! -f "$JSON_FILE" ]; then
            echo "Warning: JSON file not found: $JSON_FILE"
            update_csv_entry "$dataset" "$prompt_seed" "FILE_NOT_FOUND" "FILE_NOT_FOUND" "true"
            continue
        fi
        
        # Run evaluation with retry
        if ! run_evaluation_with_retry "$JSON_FILE" "$MODEL_WEIGHTS" "$SAVEDIR" "$DATASET_NAME"; then
            echo "Error: Evaluation failed after retries for $DATASET_NAME"
            update_csv_entry "$dataset" "$prompt_seed" "EVAL_ERROR" "EVAL_ERROR" "true"
            continue
        fi
        
        # Wait for file to be written
        sleep 2
        
        # Find evaluation results JSON
        EVAL_RESULTS_JSON=""
        
        # Try exact match first
        if [ -f "${OUTPUT_DIR_BASE}/evaluation_metrics_${DATASET_NAME}.json" ]; then
            EVAL_RESULTS_JSON="${OUTPUT_DIR_BASE}/evaluation_metrics_${DATASET_NAME}.json"
        else
            # Find most recent evaluation_metrics_*.json
            EVAL_RESULTS_JSON=$(find "$OUTPUT_DIR_BASE" -name "evaluation_metrics_*.json" -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
        fi
        
        if [ -z "$EVAL_RESULTS_JSON" ] || [ ! -f "$EVAL_RESULTS_JSON" ]; then
            echo "Warning: Could not find evaluation results JSON for $DATASET_NAME"
            update_csv_entry "$dataset" "$prompt_seed" "NOT_FOUND" "NOT_FOUND" "true"
            continue
        fi
        
        echo "Found evaluation results: $EVAL_RESULTS_JSON"
        
        # Extract metrics
        METRICS=$(extract_metrics "$EVAL_RESULTS_JSON" "$DATASET_NAME" "$prompt_seed")
        IFS=',' read -r bbox_ap50 mask_ap50 <<< "$METRICS"
        update_csv_entry "$dataset" "$prompt_seed" "$bbox_ap50" "$mask_ap50" "true"
        
        echo "Successfully processed $DATASET_NAME"
    done
done
# Process out-domain datasets
echo ""
echo "=========================================="
echo "Processing Out-Domain Datasets"
echo "=========================================="

for dataset in "${OUT_DOMAIN_DATASETS[@]}"; do
    for prompt_seed in "${PROMPT_SEEDS[@]}"; do
        # Construct JSON file path
        JSON_FILE="${OUT_DOMAIN_BASE}/${dataset}/prompt/${dataset}_prompt_seed${prompt_seed}.json"
        DATASET_NAME="${dataset}_prompt_seed${prompt_seed}"
        
        # Check if we should skip this evaluation (skip if valid entry exists)
        if [ "$SKIP_COMPLETED" = "true" ]; then
            if entry_exists_and_valid "$dataset" "$prompt_seed" "false"; then
                echo "Skipping $DATASET_NAME (valid entry already exists in CSV)"
                continue
            fi
        fi
        
        echo ""
        echo "Processing: $DATASET_NAME"
        echo "JSON File: $JSON_FILE"
        
        if [ ! -f "$JSON_FILE" ]; then
            echo "Warning: JSON file not found: $JSON_FILE"
            update_csv_entry "$dataset" "$prompt_seed" "FILE_NOT_FOUND" "FILE_NOT_FOUND" "false"
            continue
        fi
        
        # Run evaluation with retry
        if ! run_evaluation_with_retry "$JSON_FILE" "$MODEL_WEIGHTS" "$SAVEDIR" "$DATASET_NAME"; then
            echo "Error: Evaluation failed after retries for $DATASET_NAME"
            update_csv_entry "$dataset" "$prompt_seed" "EVAL_ERROR" "EVAL_ERROR" "false"
            continue
        fi
        
        # Wait for file to be written
        sleep 2
        
        # Find evaluation results JSON
        EVAL_RESULTS_JSON=""
        
        # Try exact match first
        if [ -f "${OUTPUT_DIR_BASE}/evaluation_metrics_${DATASET_NAME}.json" ]; then
            EVAL_RESULTS_JSON="${OUTPUT_DIR_BASE}/evaluation_metrics_${DATASET_NAME}.json"
        else
            # Find most recent evaluation_metrics_*.json
            EVAL_RESULTS_JSON=$(find "$OUTPUT_DIR_BASE" -name "evaluation_metrics_*.json" -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
        fi
        
        if [ -z "$EVAL_RESULTS_JSON" ] || [ ! -f "$EVAL_RESULTS_JSON" ]; then
            echo "Warning: Could not find evaluation results JSON for $DATASET_NAME"
            update_csv_entry "$dataset" "$prompt_seed" "NOT_FOUND" "NOT_FOUND" "false"
            continue
        fi
        
        echo "Found evaluation results: $EVAL_RESULTS_JSON"
        
        # Extract metrics
        METRICS=$(extract_metrics "$EVAL_RESULTS_JSON" "$DATASET_NAME" "$prompt_seed")
        IFS=',' read -r bbox_ap50 mask_ap50 <<< "$METRICS"
        update_csv_entry "$dataset" "$prompt_seed" "$bbox_ap50" "$mask_ap50" "false"
        
        echo "Successfully processed $DATASET_NAME"
    done
done

echo ""
echo "=== Batch Evaluation Completed! ==="
echo "Results saved to: $CSV_FILE"
echo ""
echo "CSV Contents (last 10 lines):"
tail -10 "$CSV_FILE"
