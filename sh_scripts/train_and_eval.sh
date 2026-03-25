#!/bin/bash

# Usage: ./train_and_eval.sh [train_net.py arguments...]
# Example: ./train_and_eval.sh --config-file configs/cvpr26_ours.yaml --num-gpus 2

# Removed set -e: continue to the next step even if one fails

echo "=== Step 1: Training (seed42) ==="

# Run train_net.py with all provided arguments
python train_net.py --data_json /aidata01/visual_prompt/dataset/InsA/in-domain/seen/InsA_train_seen_seed42.json --config configs/cvpr26_ours.yaml


sleep 10

# Run evaluation
echo ""
echo "=== Step 2: Evaluation (seed42) ==="
CUDA_VISIBLE_DEVICES=0,1 bash eval_batch_and_export_csv.sh \
    42 \
    exp_cvpr26_FINAL/seed42_ours_gumbel1.0_0.2Noise/model_0019999.pth \
    buf \
    configs/cvpr26_ours.yaml \
    exp_cvpr26_FINAL2/results42_GN0.2.csv


sleep 10

# Run train_net.py with all provided arguments
echo ""
echo "=== Step 3: Training (seed82) ==="
python train_net.py --data_json /aidata01/visual_prompt/dataset/InsA/in-domain/seen/InsA_train_seen_seed82.json --config configs/cvpr26_ours2.yaml


sleep 10

# Run evaluation
echo ""
echo "=== Step 4: Evaluation (seed82) ==="
CUDA_VISIBLE_DEVICES=0,1 bash eval_batch_and_export_csv.sh \
    82 \
    exp_cvpr26_FINAL/seed82_ours_gumbel1.0_0.2Noise/model_0019999.pth \
     \
    configs/cvpr26_ours2.yaml \
    exp_cvpr26_FINAL2/results82_GN0.2.csv


echo ""
echo "=== All steps completed successfully! ==="

