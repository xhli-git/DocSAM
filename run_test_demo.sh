#!/bin/bash
uname -a
#date
#env
date

# Replace to your own data path
DATA_PATH=./data/demo_data

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


# Test settings. You can modify them to fit your own resources.
STAGE=test
MODEL=./pretrained_model/docsam_large_all_dataset.pth
MODEL_SIZE=large
SAVE_PATH=./outputs/outputs_test/demo/
MAX_NUM=10

SHORT_RANGE=704,896
PATCH_SIZE=640,640
PATCH_NUM=1
KEEP_SIZE=False

GPU_IDS=0,1

export OMP_NUM_THREADS=1
CUDA_VISIBLE_DEVICES=${GPU_IDS} python -B -u test.py \
    --eval-path ${EVAL_PATH_0} ${EVAL_PATH_1} ${EVAL_PATH_2} ${EVAL_PATH_3} ${EVAL_PATH_4} ${EVAL_PATH_5} ${EVAL_PATH_6} \
    --stage ${STAGE} --restore-from ${MODEL} --model-size ${MODEL_SIZE} --save-path ${SAVE_PATH} --max-num ${MAX_NUM} \
    --short-range ${SHORT_RANGE} --patch-size ${PATCH_SIZE} --patch-num ${PATCH_NUM} --keep-size ${KEEP_SIZE} \
    --gpus ${GPU_IDS}