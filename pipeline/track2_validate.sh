#!/bin/bash
EXP_DIR="/home/hthieu/AICityChallenge2019/track2_experiments/180419_triplet-reid_pre-trained_densenet161_track2_small_512/"

QUE_EMB_FILE="track2_validate_query_embedding.h5"
GAL_EMB_FILE="track2_validate_embedding.h5"

QUE_FILE="data/track2_validate_query.csv"
GAL_FILE="data/track2_validate.csv"

QUE_IMG_ROOT="/home/hthieu/AICityChallenge2019/data/Track2Data/image_train/"
GAL_IMG_ROOT="/home/hthieu/AICityChallenge2019/data/Track2Data/image_train/"

RESULT="/home/hthieu/AICityChallenge2019/val_results_temp/"

LOG_FILE="track2_validate_report.json"

GPU_ID="0"

QUE_EMB="$EXP_DIR$QUE_EMB_FILE"
GAL_EMB="$EXP_DIR$GAL_EMB_FILE"
LOG_FILE_OUT="$EXP_DIR$LOG_FILE"
EMBED=1
EVAL=1

if [ $EMBED = 1 ]; then
    python3 embed.py \
        --gpu_id $GPU_ID \
        --experiment_root $EXP_DIR \
        --dataset $QUE_FILE \
        --filename $QUE_EMB_FILE \
        --image_root $QUE_IMG_ROOT
#         --flip_augment \
#         --crop_augment five \
#         --aggregator mean    \


    clear

    python3 embed.py \
        --gpu_id $GPU_ID \
        --experiment_root $EXP_DIR \
        --dataset $GAL_FILE\
        --filename $GAL_EMB_FILE\
        --image_root $GAL_IMG_ROOT
        
    clear
fi

if [ $EVAL -eq 1 ]; then
    ./evaluate.py \
        --gpu_id $GPU_ID \
        --batch_size 256 \
        --excluder diagonal \
        --query_dataset $QUE_FILE \
        --query_embeddings $QUE_EMB \
        --gallery_dataset $GAL_FILE \
        --gallery_embeddings $GAL_EMB \
        --metric euclidean \
        --filename $LOG_FILE_OUT \
        --result_folder $RESULT
fi