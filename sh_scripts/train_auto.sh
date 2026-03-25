#!/bin/bash

# AAAI26 FreqPrompt Evaluation Script
# GPU settings
export CUDA_VISIBLE_DEVICES="0,1"
export PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT="2"

# Execution parameters
SCRIPT="train_net.py"
CONFIG_FILE="configs/cvpr26_ours.yaml"
CONFIG_FILE2="configs/cvpr26_ours2.yaml"
CONFIG_FILE3="configs/cvpr26_ours3.yaml"
CONFIG_FILE4="configs/cvpr26_ours4.yaml"
python $SCRIPT \
    --config "$CONFIG_FILE" 


python $SCRIPT \
    --config "$CONFIG_FILE2" 

python $SCRIPT \
    --config "$CONFIG_FILE3" 
