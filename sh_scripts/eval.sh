#!/bin/bash
SRC_PATH=/aidata01/visual_prompt/dataset/test/DCS_Aset
INFERENCE=./inference/DCS_Aset
EXP_NAME="withPromptContrastive_8gpu_v1.3_wPKGCrop_RandomResize_FreqPrompt"
CKPT="exp/withPromptContrastive_8gpu_v1.3_wPKGCrop_RandomResize_FreqPrompt/model_0239999.pth"
CONFIG=configs/aaai26_ours.yaml


# If bash arrays are unavailable (e.g., executed with /bin/sh), list values directly in the for loop.
for process in BTM #TOP_v2
do

    base_path=$SRC_PATH/$process
    prompt_path=$base_path/prompt
    target_path=$base_path/target

    for defect_path in "$prompt_path"/*
    do
        if [ -d "$defect_path" ]; then
            defect=$(basename "$defect_path")  # Extract only the directory name from the path
            echo "Current defect: $process - $defect"
            python inference_and_save_performance.py --draw_top1_only --ckpt $CKPT --conf_files $CONFIG --target_path $target_path --prompt_path $prompt_path/$defect --save_path $INFERENCE/$EXP_NAME/$process/$defect
        fi
    done
done