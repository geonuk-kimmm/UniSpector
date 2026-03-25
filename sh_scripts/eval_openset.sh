#!/bin/bash

# Usage: ./eval_openset.sh <data_json> <model_weights> <savedir> [--config-file <config.yaml>] [--angle=<angle>] [--visualize]
# Example: ./eval_openset.sh /aidata01/visual_prompt/dataset/InsA/in-domain/unseen/GC10/prompt/GC10_unseen_seed42_prompt_seed42.json exp_cvpr26/dinov_720img/model_0017499.pth results --config-file configs/cvpr26_ours.yaml --visualize

# Check if required arguments are provided
if [ $# -lt 3 ]; then
    echo "Usage: $0 <data_json> <model_weights> <savedir> [--config-file <config.yaml>] [--angle=<angle>] [--visualize]"
    echo "Example: $0 /path/to/data.json /path/to/model.pth results --config-file configs/cvpr26_ours.yaml --visualize"
    exit 1
fi

DATA_JSON=$1
MODEL_WEIGHTS=$2
SAVEDIR=$3

# Default config file
CONFIG_FILE="configs/cvpr26_ours.yaml"

# Check for visualization option, angle parameter, and config file
VISUALIZE=false
ANGLE_OVERRIDE=""
i=4
while [ $i -le $# ]; do
    arg="${!i}"
    if [ "$arg" = "--visualize" ]; then
        VISUALIZE=true
    elif [[ "$arg" =~ ^--angle=([0-9]+)$ ]]; then
        ANGLE_OVERRIDE="${BASH_REMATCH[1]}"
    elif [ "$arg" = "--config-file" ] && [ $i -lt $# ]; then
        i=$((i + 1))
        CONFIG_FILE="${!i}"
    elif [ "$arg" = "--config" ] && [ $i -lt $# ]; then
        i=$((i + 1))
        CONFIG_FILE="${!i}"
    fi
    i=$((i + 1))

done
CONFIG_OPT="--config-file $CONFIG_FILE --config $CONFIG_FILE"

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
echo "Output Dir: $OUTPUT_DIR"
echo "Config File: $CONFIG_FILE"
echo "Batch Size: $BATCH_SIZE"
echo "Num Workers: $NUM_WORKERS"
echo "Visualize: $VISUALIZE"
if [ -n "$ANGLE_OVERRIDE" ]; then
    echo "Angle Override: $ANGLE_OVERRIDE"
fi
echo "================================"

# Step 1: Get content features
echo "Step 1: Getting content features..."

# Build python command with optional angle override
PYTHON_CMD="python train_net.py \
    --eval_only \
    --resume \
    --data_json \"$DATA_JSON\" \
    --eval_get_content_features \
    $CONFIG_OPT \
    \"InsA.TEST.BATCH_SIZE_TOTAL=$BATCH_SIZE\" \
    \"DATALOADER.NUM_WORKERS=$NUM_WORKERS\" \
    \"MODEL.WEIGHTS=$MODEL_WEIGHTS\" \
    \"OUTPUT_DIR=$OUTPUT_DIR\""

if [ -n "$ANGLE_OVERRIDE" ]; then
    PYTHON_CMD="$PYTHON_CMD \"InsA.INPUT.ANGLE=$ANGLE_OVERRIDE\""
fi

eval $PYTHON_CMD

if [ $? -ne 0 ]; then
    echo "Error: Step 1 failed!"
    exit 1
fi

echo "Step 1 completed successfully!"

Step 2: Visual openset evaluation
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
# if [ "$VISUALIZE" = true ]; then
#     echo "Step 3: Creating prediction visualizations..."
#     
#     # Check if COCO results exist
#     COCO_RESULTS="$OUTPUT_DIR/inference/coco_instances_results.json"
#     if [ -f "$COCO_RESULTS" ]; then
#         echo "Found COCO results: $COCO_RESULTS"
#         python visualize_predictions.py "$OUTPUT_DIR/inference" --data-json "$DATA_JSON" --save-images --max-images 20 --confidence-threshold 0.5
#         
#         if [ $? -ne 0 ]; then
#             echo "Warning: Prediction visualization failed, but evaluation completed successfully!"
#         else
#             echo "Step 3 completed successfully!"
#         fi
#     else
#         echo "Warning: No COCO results found for visualization. Check if evaluation completed properly."
#     fi
# fi

echo "=== Evaluation completed! ==="
echo "Check $OUTPUT_DIR for evaluation results (evaluation_metrics_*.json)"
if [ "$VISUALIZE" = true ]; then
    echo "Visualization plots saved in $OUTPUT_DIR"
fi