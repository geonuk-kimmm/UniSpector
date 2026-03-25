#!/bin/bash
SRC_PATH=/aidata01/visual_prompt/dataset/test/DCS_Aset
INFERENCE=./inference/DCS_Aset
EXP_NAME=result_DINOv_PromptCont_defectv1.1_wPKG_Freq

CKPT=exp/withPromptContrastive_4gpu_2*Freq_v1/model_0114999.pth
CONFIG=configs/dinov_defect_v1_wPKG_train.yaml

# If bash arrays are unavailable (e.g., executed with /bin/sh), list values directly in the for loop.
for process in BTM SIDE TOP_v2
do
    base_path=$SRC_PATH/$process
    prompt_path=$base_path/prompt
    target_path=$base_path/target

    for defect_path in "$prompt_path"/*
    do
        if [ -d "$defect_path" ]; then
            defect=$(basename "$defect_path")  # Extract only the directory name from the path
            echo "Current defect: $process - $defect"
            
        fi
    done
done