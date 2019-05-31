TARGET="front_light"
#!/bin/sh
TRAIN_SET="/home/bnminh/projects/ai2/datasets/ai_dataset/DATA/part/csv_file/"$TARGET"_train.csv"
IMAGE_ROOT="/home/bnminh/projects/ai2/datasets/ai_dataset/DATA/part/"$TARGET"/train"
INIT_CHECKPT="/home/hthieu/AICityChallenge2019/checkpoint/resnet_v1_50.ckpt"
EXP_ROOT="/home/hthieu/AICityChallenge2019/track2_experiments/100519_triplet-reid_"$TARGET"/"

# rm -rf $EXP_ROOT
python3 train_v2.py \
    --gpu_id 1\
    --train_set $TRAIN_SET\
    --initial_checkpoint $INIT_CHECKPT \
    --model_name resnet_v1_50    \
    --head_name fc1024 \
    --image_root $IMAGE_ROOT \
    --experiment_root $EXP_ROOT \
    --flip_augment \
    --crop_augment \
    --embedding_dim 512 \
    --batch_p 16 \
    --batch_k 4  \
    --pre_crop_height 256 --pre_crop_width 256 \
    --net_input_height 224 --net_input_width 224 \
    --margin soft \
    --metric euclidean \
    --loss batch_hard \
    --learning_rate 1e-4 \
    --train_iterations 30000 \
    --decay_start_iteration 15000 \
    --loading_threads 8\
    "$@"
