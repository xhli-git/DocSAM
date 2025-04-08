#!/bin/bash
uname -a
#date
#env
date

EVAL_PATH=$1

STAGE=test
MODEL=./pretrained_model/docsam_large_all_dataset_keepsize.pth
MODEL_SIZE=large
SAVE_PATH=./outputs/outputs_test/large_keepsize
MAX_NUM=12000

SHORT_RANGE=640,1280
PATCH_SIZE=640,640
PATCH_NUM=1
KEEP_SIZE=True

GPU_IDS=$2

export OMP_NUM_THREADS=1
CUDA_VISIBLE_DEVICES=${GPU_IDS} python -B -u test.py \
    --eval-path ${EVAL_PATH} \
    --stage ${STAGE} --restore-from ${MODEL} --model-size ${MODEL_SIZE} --save-path ${SAVE_PATH} --max-num ${MAX_NUM} \
    --short-range ${SHORT_RANGE} --patch-size ${PATCH_SIZE} --patch-num ${PATCH_NUM} --keep-size ${KEEP_SIZE} \
    --gpus ${GPU_IDS}