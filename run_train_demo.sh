#!/bin/bash
uname -a
#date
#env
date

# Replace to your own data path
DATA_PATH=./data/demo_data


# Datasets used for evaluation. 

# Layout
EVAL_PATH_1=${DATA_PATH}/Layout/PubLayNet/data/PubLayNet/test/

# Ancient
EVAL_PATH_2=${DATA_PATH}/Ancient/CASIA-AHCDB/data/Style1/test/
EVAL_PATH_3=${DATA_PATH}/Ancient/CASIA-AHCDB/data/Style2/test/

# Handwritten
EVAL_PATH_4=${DATA_PATH}/Handwritten/CASIA-HWDB/data/CASIA-HWDB/test/

# Table
EVAL_PATH_5=${DATA_PATH}/Table/WTW/data/WTW/test/

# SceneText
EVAL_PATH_6=${DATA_PATH}/SceneText/Total-Text/data/TotalText/test/


# Datasets used for training. 

# Layout
TRAIN_PATH_1=${DATA_PATH}/Layout/PubLayNet/data/PubLayNet/train/

# Ancient
TRAIN_PATH_2=${DATA_PATH}/Ancient/CASIA-AHCDB/data/Style1/train/
TRAIN_PATH_3=${DATA_PATH}/Ancient/CASIA-AHCDB/data/Style2/train/

# Handwritten
TRAIN_PATH_4=${DATA_PATH}/Handwritten/CASIA-HWDB/data/CASIA-HWDB/train/

# Table
TRAIN_PATH_5=${DATA_PATH}/Table/WTW/data/WTW/train/

# SceneText
TRAIN_PATH_6=${DATA_PATH}/SceneText/Total-Text/data/TotalText/train/


# Training settings. You can modify them to fit your own resources.
MODEL_SIZE=large
SAVE_PATH=./outputs/outputs_train/
MAX_NUM=10

BATCH_SIZE=4
LEARNING_RATE=2e-5
MOMENTUM=0.9
WEIGHT_DECAY=1e-2
LR_SCHEDULER=cosine

FINE_TUNE=True
RESTORE_FROM=./pretrained_model/docsam_large_all_dataset.pth
SNAPSHOT_DIR=./snapshots/
START_ITER=0
TOTAL_ITER=4000
GPU_IDS=0,1

export OMP_NUM_THREADS=1
CUDA_VISIBLE_DEVICES=${GPU_IDS} torchrun --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 --nproc_per_node=2 train.py \
    --train-path ${TRAIN_PATH_0} ${TRAIN_PATH_1} ${TRAIN_PATH_2} ${TRAIN_PATH_3} ${TRAIN_PATH_4} ${TRAIN_PATH_5} ${TRAIN_PATH_6} \
    --eval-path ${EVAL_PATH_0} ${EVAL_PATH_1} ${EVAL_PATH_2} ${EVAL_PATH_3} ${EVAL_PATH_4} ${EVAL_PATH_5} ${EVAL_PATH_6} \
    --batch-size ${BATCH_SIZE} --learning-rate ${LEARNING_RATE} --momentum ${MOMENTUM} --weight-decay ${WEIGHT_DECAY} --lr-scheduler ${LR_SCHEDULER} \
    --fine-tune ${FINE_TUNE} --restore-from ${RESTORE_FROM} --snapshot-dir ${SNAPSHOT_DIR} \
    --start-iter ${START_ITER} --total-iter ${TOTAL_ITER} --gpus ${GPU_IDS} \
    --model-size ${MODEL_SIZE} --save-path ${SAVE_PATH} --max-num ${MAX_NUM}
        