#!/bin/bash

# Auto-run eval_openset.sh for multiple prompts
# Usage: CUDA_VISIBLE_DEVICES=0,1 sh run_eval_openset_auto.sh

set -e  # Exit on error

MODEL_WEIGHTS="exp_cvpr26_FINAL/closed_set_dinov/model_0019999.pth"
SAVEDIR_BASE="forCloseset2"

echo "=== Auto Evaluation Script ==="
echo "Model: $MODEL_WEIGHTS"
echo "================================"

# Step 1: Run prompt82.json (once)
echo ""
echo ">>> Step 1: Running prompt82.json (1 time)"
echo "=============================================="
CUDA_VISIBLE_DEVICES=0,1 sh eval_openset.sh closed_json/prompt82.json "$MODEL_WEIGHTS" "$SAVEDIR_BASE"

if [ $? -ne 0 ]; then
    echo "Error: prompt82.json evaluation failed!"
    exit 1
fi

echo ""
echo ">>> prompt82.json completed successfully!"
echo ""

# Step 2: Run prompt42.json (3 times)
echo ""
echo ">>> Step 2: Running prompt42.json (3 times)"
echo "=============================================="
for i in {1..3}; do
    echo ""
    echo "--- Run $i/3 for prompt42.json ---"
    CUDA_VISIBLE_DEVICES=0,1 sh eval_openset.sh closed_json/prompt42.json "$MODEL_WEIGHTS" "$SAVEDIR_BASE"
    
    if [ $? -ne 0 ]; then
        echo "Error: prompt42.json evaluation (run $i/3) failed!"
        exit 1
    fi
    
    echo "--- Run $i/3 for prompt42.json completed successfully! ---"
done

echo ""
echo ">>> All prompt42.json runs completed successfully!"
echo ""

# Step 3: Run prompt777.json (3 times)
echo ""
echo ">>> Step 3: Running prompt777.json (3 times)"
echo "=============================================="
for i in {1..3}; do
    echo ""
    echo "--- Run $i/3 for prompt777.json ---"
    CUDA_VISIBLE_DEVICES=0,1 sh eval_openset.sh closed_json/prompt777.json "$MODEL_WEIGHTS" "$SAVEDIR_BASE"
    
    if [ $? -ne 0 ]; then
        echo "Error: prompt777.json evaluation (run $i/3) failed!"
        exit 1
    fi
    
    echo "--- Run $i/3 for prompt777.json completed successfully! ---"
done

echo ""
echo ">>> All prompt777.json runs completed successfully!"
echo ""

echo "=== All evaluations completed successfully! ==="

