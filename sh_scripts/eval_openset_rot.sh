#!/bin/bash

# Usage: ./eval_openset.sh <data_json> <model_weights> <savedir> [--visualize]
# Example: ./eval_openset.sh /aidata01/visual_prompt/dataset/InsA/in-domain/unseen/GC10/prompt/GC10_unseen_seed42_prompt_seed42.json exp_cvpr26/dinov_720img/model_0017499.pth results --visualize

# Check if required arguments are provided
if [ $# -lt 3 ]; then
    echo "Usage: $0 <data_json> <model_weights> <savedir> [--visualize]"
    echo "Example: $0 /path/to/data.json /path/to/model.pth results --visualize"
    exit 1
fi

DATA_JSON=$1
MODEL_WEIGHTS=$2
SAVEDIR=$3

# Check for visualization option
VISUALIZE=false
if [ "$4" = "--visualize" ]; then
    VISUALIZE=true
fi

# Fixed evaluation parameters
NUM_GPUS=2
BATCH_SIZE=4
NUM_WORKERS=1

# Generate OUTPUT_DIR from MODEL_WEIGHTS and savedir
MODEL_FILENAME=$(basename "$MODEL_WEIGHTS")
OUTPUT_DIR=$(dirname "$MODEL_WEIGHTS")/exp_${MODEL_FILENAME}/${SAVEDIR}

echo "=== Evaluation Configuration ==="
echo "Data JSON: $DATA_JSON"
echo "Model Weights: $MODEL_WEIGHTS"
echo "Save Dir: $SAVEDIR"
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
    --config-file configs/cvpr26_ours.yaml \
    "InsA.TEST.BATCH_SIZE_TOTAL=$BATCH_SIZE" \
    "DATALOADER.NUM_WORKERS=$NUM_WORKERS" \
    "MODEL.WEIGHTS=$MODEL_WEIGHTS" \
    "OUTPUT_DIR=$OUTPUT_DIR"

if [ $? -ne 0 ]; then
    echo "Error: Step 1 failed!"
    exit 1
fi

echo "Step 1 completed successfully!"

# Step 2: Visual openset evaluation
echo "Step 2: Running visual openset evaluation..."
python train_net.py \
    --eval_only \
    --resume \
    --data_json "$DATA_JSON" \
    --eval_visual_openset \
    --num-gpus "$NUM_GPUS" \
    --config-file configs/cvpr26_ours.yaml \
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