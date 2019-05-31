#!/bin/sh

IMAGE_ROOT_TRAIN=/home/bnminh/projects/ai2/datasets/ai_dataset/DATA/image_train/
IMAGE_ROOT_VAL=/home/bnminh/projects/ai2/datasets/ai_dataset/DATA/image_train/
IMAGE_ROOT_TEST=/home/bnminh/projects/ai2/datasets/ai_dataset/DATA/image_test/

TRAIN_SMALL_CSV=/home/bnminh/projects/ai2/datasets/ai_dataset/MINH_CSV/train_small.csv
VAL_SMALL_CSV=/home/bnminh/projects/ai2/datasets/ai_dataset/MINH_CSV/val_small.csv

RESNET_CHECKPOINT=/home/bnminh/projects/ai2/SOURCE/triplet-reid/resnet/resnet_v1_50.ckpt

INIT_CHECKPT=$RESNET_CHECKPOINT
EXP_ROOT=experiments/init_exp/
CUDA_VISIBLE_DEVICES="1"

python train.py \
    --train_set $TRAIN_SMALL_CSV \
    --model_name resnet_v1_50 \
    --image_root $IMAGE_ROOT_TRAIN \
    --initial_checkpoint $INIT_CHECKPT \
    --experiment_root $EXP_ROOT \
    --flip_augment \
    --crop_augment \
    --embedding_dim 128 \
    --batch_p 18 \
    --batch_k 4 \
    --pre_crop_height 144 --pre_crop_width 288 \
    --net_input_height 128 --net_input_width 256 \
    --margin soft \
    --metric euclidean \
    --loss batch_hard \
    --learning_rate 3e-4 \
    --train_iterations 25000 \
    --decay_start_iteration 15000 \
    "$@"