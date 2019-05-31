#!/bin/sh
TRAIN_SET="/home/hthieu/AICityChallenge2019/triplet-reid/data/vehi_type/3.csv"
IMAGE_ROOT="/home/hthieu/AICityChallenge2019/data/Track2Data/image_train/"
INIT_CHECKPT="/home/hthieu/AICityChallenge2019/checkpoint/tf-densenet161.ckpt"
EXP_ROOT="/home/hthieu/AICityChallenge2019/track2_experiments/200419_triplet-reid_pre-trained_densenet161_track2_small_512_vehi_type_3/"

# rm -rf $EXP_ROOT
python3 train_v2.py \
    --gpu_id 0\
    --train_set $TRAIN_SET\
    --initial_checkpoint $INIT_CHECKPT \
    --model_name densenet_161    \
    --head_name fc1024 \
    --image_root $IMAGE_ROOT \
    --experiment_root $EXP_ROOT \
    --flip_augment \
    --crop_augment \
    --embedding_dim 512 \
    --batch_p 3 \
    --batch_k 16  \
    --pre_crop_height 144 --pre_crop_width 288 \
    --net_input_height 128 --net_input_width 256 \
    --margin soft \
    --metric euclidean \
    --loss batch_hard \
    --learning_rate 1e-4 \
    --train_iterations 15000 \
    --decay_start_iteration 15000 \
    --loading_threads 8\
    "$@"




