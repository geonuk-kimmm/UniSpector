#!/bin/bash

# Usage: ./eval_openset.sh <data_json> <model_weights> <savedir> [config_file] [--visualize]
# Example: ./eval_openset.sh /aidata01/visual_prompt/dataset/InsA/in-domain/unseen/GC10/prompt/GC10_unseen_seed42_prompt_seed42.json exp_cvpr26/dinov_720img/model_0017499.pth results configs/cvpr26_ours.yaml --visualize

# Check if required arguments are provided
if [ $# -lt 3 ]; then
    echo "Usage: $0 <data_json> <model_weights> <savedir> [config_file] [--visualize]"
    echo "Example: $0 /path/to/data.json /path/to/model.pth results configs/cvpr26_ours.yaml --visualize"
    exit 1
fi

DATA_JSON=$1
MODEL_WEIGHTS=$2
SAVEDIR=$3

# Default config file
CONFIG_FILE="configs/cvpr26_ours.yaml"

# If 4th argument exists and looks like a config file, use it
if [ $# -ge 4 ]; then
    if [[ "$4" == *.yaml ]] || [[ "$4" == *.yml ]]; then
        CONFIG_FILE="$4"
    fi
fi

# Check for visualization option and additional config files
VISUALIZE=false
i=4
while [ $i -le $# ]; do
    arg="${!i}"
    if [ "$arg" = "--visualize" ]; then
        VISUALIZE=true
    elif [ "$i" -gt 4 ]; then
        if [[ "$arg" == *.yaml ]] || [[ "$arg" == *.yml ]]; then
            # Additional config file (if provided after 4th arg)
            CONFIG_FILE="$arg"
        fi
    fi
    i=$((i + 1))
done

# Calculate NUM_GPUS from CUDA_VISIBLE_DEVICES if set, otherwise default to 2
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    # Count the number of GPUs in CUDA_VISIBLE_DEVICES (e.g., "0,1" -> 2, "2,3,4" -> 3)
    NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
else
    # Default to 2 if CUDA_VISIBLE_DEVICES is not set
    NUM_GPUS=2
fi

# Fixed evaluation parameters
BATCH_SIZE=4
NUM_WORKERS=1

# Generate OUTPUT_DIR from MODEL_WEIGHTS and savedir
MODEL_FILENAME=$(basename "$MODEL_WEIGHTS")
OUTPUT_DIR=$(dirname "$MODEL_WEIGHTS")/exp_${MODEL_FILENAME}/${SAVEDIR}

echo "=== Evaluation Configuration ==="
echo "Data JSON: $DATA_JSON"
echo "Model Weights: $MODEL_WEIGHTS"
echo "Save Dir: $SAVEDIR"
echo "Config File: $CONFIG_FILE"
echo "Output Dir: $OUTPUT_DIR"
echo "Num GPUs: $NUM_GPUS"
echo "Batch Size: $BATCH_SIZE"
echo "Num Workers: $NUM_WORKERS"
echo "Visualize: $VISUALIZE"
echo "================================"

# Step 1: Get content features
echo "Step 1: Getting content features..."
python train_net.py \
    --eval_only \
    --resume \
    --data_json "$DATA_JSON" \
    --eval_get_content_features \
    --num-gpus "$NUM_GPUS" \
    --config "$CONFIG_FILE" \
    "InsA.TEST.BATCH_SIZE_TOTAL=$BATCH_SIZE" \
    "DATALOADER.NUM_WORKERS=$NUM_WORKERS" \
    "MODEL.WEIGHTS=$MODEL_WEIGHTS" \
    "OUTPUT_DIR=$OUTPUT_DIR"

if [ $? -ne 0 ]; then
    echo "Error: Step 1 failed!"
    exit 1
fi


# Step 2: Visual openset evaluation
echo "Step 2: Running visual openset evaluation..."
python train_net.py \
    --eval_only \
    --resume \
    --data_json "$DATA_JSON" \
    --eval_visual_openset \
    --num-gpus "$NUM_GPUS" \
    --config "$CONFIG_FILE" \
    "InsA.TEST.BATCH_SIZE_TOTAL=$BATCH_SIZE" \
    "DATALOADER.NUM_WORKERS=$NUM_WORKERS" \
    "MODEL.WEIGHTS=$MODEL_WEIGHTS" \
    "OUTPUT_DIR=$OUTPUT_DIR"

if [ $? -ne 0 ]; then
    echo "Error: Step 2 failed!"
    exit 1
fi

echo "Step 2 completed successfully!"

# Step 3: Prediction Visualization (optional)
if [ "$VISUALIZE" = true ]; then
    echo "Step 3: Creating prediction visualizations..."
    
    # Check if COCO results exist
    COCO_RESULTS="$OUTPUT_DIR/inference/coco_instances_results.json"
    if [ -f "$COCO_RESULTS" ]; then
        echo "Found COCO results: $COCO_RESULTS"
        python visualize_predictions.py "$OUTPUT_DIR/inference" --data-json "$DATA_JSON" --save-images --max-images 20 --confidence-threshold 0.5
        
        if [ $? -ne 0 ]; then
            echo "Warning: Prediction visualization failed, but evaluation completed successfully!"
        else
            echo "Step 3 completed successfully!"
        fi
    else
        echo "Warning: No COCO results found for visualization. Check if evaluation completed properly."
    fi
fi

echo "=== Evaluation completed! ==="
echo "Check $OUTPUT_DIR for evaluation results (evaluation_metrics_*.json)"
if [ "$VISUALIZE" = true ]; then
    echo "Visualization plots saved in $OUTPUT_DIR"
fi