#!/bin/bash
uname -a
#date
#env
date

# Replace to your own data path
DATA_PATH=/local_data1/xhli/data4/datas/DownStreamData

# Datasets used for evaluation during training. 
# Only several datasets are selected for saving time.

# # Layout
# EVAL_PATH_00=${DATA_PATH}/Layout/BaDLAD/data/BaDLAD/train/
# EVAL_PATH_01=${DATA_PATH}/Layout/CDLA/data/CDLA/val/
# EVAL_PATH_02=${DATA_PATH}/Layout/D4LA/data/D4LA/test/
# EVAL_PATH_03=${DATA_PATH}/Layout/DocBank/data/DocBank/test/
# EVAL_PATH_04=${DATA_PATH}/Layout/DocLayNet/data/DocLayNet_core/test/
# EVAL_PATH_05=${DATA_PATH}/Layout/ICDAR2017_POD/data/ICDAR2017_POD/test/
# EVAL_PATH_06=${DATA_PATH}/Layout/IIIT-AR-13K/data/IIIT-AR-13K/test/
# EVAL_PATH_07=${DATA_PATH}/Layout/M6Doc/data/M6Doc/test/
# EVAL_PATH_08=${DATA_PATH}/Layout/PubLayNet/data/PubLayNet/test/
# EVAL_PATH_09=${DATA_PATH}/Layout/RanLayNet/data/RanLayNet/val/

# # Ancient
# EVAL_PATH_20=${DATA_PATH}/Ancient/CASIA-AHCDB/data/Style1/test/
# EVAL_PATH_21=${DATA_PATH}/Ancient/CASIA-AHCDB/data/Style2/test/
# EVAL_PATH_22=${DATA_PATH}/Ancient/CHDAC-2022/data/CHDAC-2022/train/
# EVAL_PATH_23=${DATA_PATH}/Ancient/ICDAR2019HDRC/data/ICDAR2019HDRC/train/
# EVAL_PATH_24=${DATA_PATH}/Ancient/SCUT-CAB/data/logical/test/
# EVAL_PATH_25=${DATA_PATH}/Ancient/SCUT-CAB/data/physical/test/
# EVAL_PATH_26=${DATA_PATH}/Ancient/TKHMTH2200/data/TKHMTH2200/test/
# EVAL_PATH_27=${DATA_PATH}/Layout/HJDataset/data/HJDataset/test/

# # Handwritten
# EVAL_PATH_30=${DATA_PATH}/Handwritten/CASIA-HWDB/data/CASIA-HWDB/test/
# EVAL_PATH_31=${DATA_PATH}/Handwritten/SCUT-HCCDoc/data/SCUT-HCCDoc/test/

# # Table
# EVAL_PATH_40=${DATA_PATH}/Table/FinTabNet/data/FinTabNet/test/
# EVAL_PATH_41=${DATA_PATH}/Table/FinTabNet/data/FinTabNet.c/FinTabNet.c-Structure/test/
# EVAL_PATH_42=${DATA_PATH}/Table/ICDAR2013/data/ICDAR2013/competition-dataset/
# EVAL_PATH_43=${DATA_PATH}/Table/ICDAR2013/data/ICDAR2013.c/ICDAR2013.c-Structure/test/
# EVAL_PATH_44=${DATA_PATH}/Table/ICDAR2017_POD/data/ICDAR2017_POD/test/
# EVAL_PATH_45=${DATA_PATH}/Table/ICDAR2019_cTDaR/data/Archive/test_track_b2/
# EVAL_PATH_46=${DATA_PATH}/Table/ICDAR2019_cTDaR/data/Modern/test_track_b2/
# EVAL_PATH_47=${DATA_PATH}/Table/NTable/data/NTable-gen/test/
# EVAL_PATH_48=${DATA_PATH}/Table/NTable/data/NTable-ori_cam_ver.2/NTable-cam/test/
# EVAL_PATH_49=${DATA_PATH}/Table/PubTables-1M/data/PubTables-1M-Detection/test/
# EVAL_PATH_50=${DATA_PATH}/Table/PubTables-1M/data/PubTables-1M-Structure/test/
# EVAL_PATH_51=${DATA_PATH}/Table/PubTabNet/data/PubTabNet/val/
# EVAL_PATH_52=${DATA_PATH}/Table/STDW/data/STDW/
# EVAL_PATH_53=${DATA_PATH}/Table/TableBank/data/TableBank/Detection/latex/test/
# EVAL_PATH_54=${DATA_PATH}/Table/TableBank/data/TableBank/Detection/word/test/
# EVAL_PATH_55=${DATA_PATH}/Table/TNCR/data/TNCR/test/
# EVAL_PATH_56=${DATA_PATH}/Table/WTW/data/WTW/test/

# # SceneText
# EVAL_PATH_60=${DATA_PATH}/SceneText/CASIA-10k/data/CASIA-10k/test/
# EVAL_PATH_61=${DATA_PATH}/SceneText/COCO-Text/data/cocotext/val/
EVAL_PATH_62=${DATA_PATH}/SceneText/CTW-1500/data/CTW-1500/test/
# EVAL_PATH_63=${DATA_PATH}/SceneText/CTW-Public/data/test/
# EVAL_PATH_64=${DATA_PATH}/SceneText/HUST-TR400/data/HUST-TR400/
# EVAL_PATH_65=${DATA_PATH}/SceneText/ICDAR_2011_2015_RRC/data/Task1-Born-Digital-Images/Task-1.1-Text-Localization-2013-edition/test/
# EVAL_PATH_66=${DATA_PATH}/SceneText/ICDAR_2011_2015_RRC/data/Task2-Focused-Scene-Text/Task-2.1-Text-Localization-2013-edition/test/
# EVAL_PATH_67=${DATA_PATH}/SceneText/ICDAR_2011_2015_RRC/data/Task4-Incidental-Scene-Text/Task-4.1-Text-Localization-2015-edition/test/
# EVAL_PATH_68=${DATA_PATH}/SceneText/ICDAR_2017_RCTW/data/RCTW-17/train/
# EVAL_PATH_69=${DATA_PATH}/SceneText/ICDAR_2017_RRC_MLT/data/Task-1-Multi-script-text-detection/val/
# EVAL_PATH_70=${DATA_PATH}/SceneText/ICDAR_2019_RRC_ArT/data/task1-scene-text-detection/train/
# EVAL_PATH_71=${DATA_PATH}/SceneText/ICDAR_2019_RRC_LSVT/data/LSVT-2019/train_full/
# EVAL_PATH_72=${DATA_PATH}/SceneText/ICDAR_2019_RRC_MLT/data/Task-1-Multi-script-text-detection/train/
# EVAL_PATH_73=${DATA_PATH}/SceneText/ICDAR_2019_RRC_MLT/data/Task-4-End-to-End-text-detection-and-recognition/
# EVAL_PATH_74=${DATA_PATH}/SceneText/ICDAR_2019_RRC_ReCTS/data/ReCTS/train/
# EVAL_PATH_75=${DATA_PATH}/SceneText/ICDAR_2023_RRC_HierText/data/HierText-2023/val/
# EVAL_PATH_76=${DATA_PATH}/SceneText/ICDAR_2023_RRC_ReST/data/train/
# EVAL_PATH_77=${DATA_PATH}/SceneText/ICPR_2018_MTWI/data/MIWI-2018/train/
# EVAL_PATH_78=${DATA_PATH}/SceneText/MSRA-TD500/data/MSRA-TD500/test/
# EVAL_PATH_79=${DATA_PATH}/SceneText/ShopSign/data/ShopSign_1265/
# EVAL_PATH_80=${DATA_PATH}/SceneText/Total-Text/data/TotalText/test/
# EVAL_PATH_81=${DATA_PATH}/SceneText/USTB-SV1K/data/USTB-SV1K/test/


# Datasets used for training. 

# # Layout
# TRAIN_PATH_00=${DATA_PATH}/Layout/BaDLAD/data/BaDLAD/train/
# TRAIN_PATH_01=${DATA_PATH}/Layout/CDLA/data/CDLA/train/
# TRAIN_PATH_02=${DATA_PATH}/Layout/D4LA/data/D4LA/train/
# TRAIN_PATH_03=${DATA_PATH}/Layout/DocBank/data/DocBank/train/
# TRAIN_PATH_04=${DATA_PATH}/Layout/DocLayNet/data/DocLayNet_core/train/
# TRAIN_PATH_05=${DATA_PATH}/Layout/ICDAR2017_POD/data/ICDAR2017_POD/train/
# TRAIN_PATH_06=${DATA_PATH}/Layout/IIIT-AR-13K/data/IIIT-AR-13K/train/
# TRAIN_PATH_07=${DATA_PATH}/Layout/M6Doc/data/M6Doc/train/
# TRAIN_PATH_08=${DATA_PATH}/Layout/PubLayNet/data/PubLayNet/train/
# TRAIN_PATH_09=${DATA_PATH}/Layout/RanLayNet/data/RanLayNet/train/

# # Ancient
# TRAIN_PATH_20=${DATA_PATH}/Ancient/CASIA-AHCDB/data/Style1/train/
# TRAIN_PATH_21=${DATA_PATH}/Ancient/CASIA-AHCDB/data/Style2/train/
# TRAIN_PATH_22=${DATA_PATH}/Ancient/CHDAC-2022/data/CHDAC-2022/train/
# TRAIN_PATH_23=${DATA_PATH}/Ancient/ICDAR2019HDRC/data/ICDAR2019HDRC/train/
# TRAIN_PATH_24=${DATA_PATH}/Ancient/SCUT-CAB/data/logical/train/
# TRAIN_PATH_25=${DATA_PATH}/Ancient/SCUT-CAB/data/physical/train/
# TRAIN_PATH_26=${DATA_PATH}/Ancient/TKHMTH2200/data/TKHMTH2200/train/
# TRAIN_PATH_27=${DATA_PATH}/Layout/HJDataset/data/HJDataset/train/

# # Handwritten
# TRAIN_PATH_30=${DATA_PATH}/Handwritten/CASIA-HWDB/data/CASIA-HWDB/train/
# TRAIN_PATH_31=${DATA_PATH}/Handwritten/SCUT-HCCDoc/data/SCUT-HCCDoc/train/

# # Table
# TRAIN_PATH_40=${DATA_PATH}/Table/FinTabNet/data/FinTabNet/train/
# TRAIN_PATH_41=${DATA_PATH}/Table/FinTabNet/data/FinTabNet.c/FinTabNet.c-Structure/train/
# # TRAIN_PATH_42=${DATA_PATH}/Table/ICDAR2013/data/ICDAR2013/competition-dataset/
# # TRAIN_PATH_43=${DATA_PATH}/Table/ICDAR2013/data/ICDAR2013.c/ICDAR2013.c-Structure/train/
# TRAIN_PATH_44=${DATA_PATH}/Table/ICDAR2017_POD/data/ICDAR2017_POD/train/
# TRAIN_PATH_45=${DATA_PATH}/Table/ICDAR2019_cTDaR/data/Archive/train/
# TRAIN_PATH_46=${DATA_PATH}/Table/ICDAR2019_cTDaR/data/Modern/train/
# TRAIN_PATH_47=${DATA_PATH}/Table/NTable/data/NTable-gen/train/
# TRAIN_PATH_48=${DATA_PATH}/Table/NTable/data/NTable-ori_cam_ver.2/NTable-cam/train/
# TRAIN_PATH_49=${DATA_PATH}/Table/PubTables-1M/data/PubTables-1M-Detection/train/
# TRAIN_PATH_50=${DATA_PATH}/Table/PubTables-1M/data/PubTables-1M-Structure/train/
# TRAIN_PATH_51=${DATA_PATH}/Table/PubTabNet/data/PubTabNet/train/
# TRAIN_PATH_52=${DATA_PATH}/Table/STDW/data/STDW/
# TRAIN_PATH_53=${DATA_PATH}/Table/TableBank/data/TableBank/Detection/latex/train/
# TRAIN_PATH_54=${DATA_PATH}/Table/TableBank/data/TableBank/Detection/word/train/
# TRAIN_PATH_55=${DATA_PATH}/Table/TNCR/data/TNCR/train/
# TRAIN_PATH_56=${DATA_PATH}/Table/WTW/data/WTW/train/

# # SceneText
# TRAIN_PATH_60=${DATA_PATH}/SceneText/CASIA-10k/data/CASIA-10k/train/
# TRAIN_PATH_61=${DATA_PATH}/SceneText/COCO-Text/data/cocotext/train/
TRAIN_PATH_62=${DATA_PATH}/SceneText/CTW-1500/data/CTW-1500/train/
# TRAIN_PATH_63=${DATA_PATH}/SceneText/CTW-Public/data/train/
# TRAIN_PATH_64=${DATA_PATH}/SceneText/HUST-TR400/data/HUST-TR400/
# TRAIN_PATH_65=${DATA_PATH}/SceneText/ICDAR_2011_2015_RRC/data/Task1-Born-Digital-Images/Task-1.1-Text-Localization-2013-edition/train/
# TRAIN_PATH_66=${DATA_PATH}/SceneText/ICDAR_2011_2015_RRC/data/Task2-Focused-Scene-Text/Task-2.1-Text-Localization-2013-edition/train/
# TRAIN_PATH_67=${DATA_PATH}/SceneText/ICDAR_2011_2015_RRC/data/Task4-Incidental-Scene-Text/Task-4.1-Text-Localization-2015-edition/train/
# TRAIN_PATH_68=${DATA_PATH}/SceneText/ICDAR_2017_RCTW/data/RCTW-17/train/
# TRAIN_PATH_69=${DATA_PATH}/SceneText/ICDAR_2017_RRC_MLT/data/Task-1-Multi-script-text-detection/train/
# TRAIN_PATH_70=${DATA_PATH}/SceneText/ICDAR_2019_RRC_ArT/data/task1-scene-text-detection/train/
# TRAIN_PATH_71=${DATA_PATH}/SceneText/ICDAR_2019_RRC_LSVT/data/LSVT-2019/train_full/
# TRAIN_PATH_72=${DATA_PATH}/SceneText/ICDAR_2019_RRC_MLT/data/Task-1-Multi-script-text-detection/train/
# TRAIN_PATH_73=${DATA_PATH}/SceneText/ICDAR_2019_RRC_MLT/data/Task-4-End-to-End-text-detection-and-recognition/
# TRAIN_PATH_74=${DATA_PATH}/SceneText/ICDAR_2019_RRC_ReCTS/data/ReCTS/train/
# TRAIN_PATH_75=${DATA_PATH}/SceneText/ICDAR_2023_RRC_HierText/data/HierText-2023/train/
# TRAIN_PATH_76=${DATA_PATH}/SceneText/ICDAR_2023_RRC_ReST/data/train/
# TRAIN_PATH_77=${DATA_PATH}/SceneText/ICPR_2018_MTWI/data/MIWI-2018/train/
# TRAIN_PATH_78=${DATA_PATH}/SceneText/MSRA-TD500/data/MSRA-TD500/train/
# TRAIN_PATH_79=${DATA_PATH}/SceneText/ShopSign/data/ShopSign_1265/
# TRAIN_PATH_80=${DATA_PATH}/SceneText/Total-Text/data/TotalText/train/
# TRAIN_PATH_81=${DATA_PATH}/SceneText/USTB-SV1K/data/USTB-SV1K/train/

# Training settings. You can modify them to fit your own resources.
MODEL_SIZE=large
SAVE_PATH=./outputs/outputs_train/
MAX_NUM=10

SHORT_RANGE=704,896
PATCH_SIZE=800,800
PATCH_NUM=1
KEEP_SIZE=False

BATCH_SIZE=16
LEARNING_RATE=2e-5
MOMENTUM=0.9
WEIGHT_DECAY=1e-2
LR_SCHEDULER=cosine

FINE_TUNE=True
RESTORE_FROM=./pretrained_model/docsam_large_all_dataset.pth
SNAPSHOT_DIR=./snapshots/
START_ITER=0
TOTAL_ITER=40000
GPU_IDS=0,1,2,3

export OMP_NUM_THREADS=1
CUDA_VISIBLE_DEVICES=${GPU_IDS} torchrun --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 --nproc_per_node=4 train.py \
    --train-path ${TRAIN_PATH_00} ${TRAIN_PATH_01} ${TRAIN_PATH_02} ${TRAIN_PATH_03} ${TRAIN_PATH_04} ${TRAIN_PATH_05} ${TRAIN_PATH_06} ${TRAIN_PATH_07} ${TRAIN_PATH_08} ${TRAIN_PATH_09} \
        ${TRAIN_PATH_20} ${TRAIN_PATH_21} ${TRAIN_PATH_22} ${TRAIN_PATH_23} ${TRAIN_PATH_24} ${TRAIN_PATH_25} ${TRAIN_PATH_26} ${TRAIN_PATH_27} \
        ${TRAIN_PATH_30} ${TRAIN_PATH_31} \
        ${TRAIN_PATH_40} ${TRAIN_PATH_41} ${TRAIN_PATH_42} ${TRAIN_PATH_43} ${TRAIN_PATH_44} ${TRAIN_PATH_45} ${TRAIN_PATH_46} ${TRAIN_PATH_47} ${TRAIN_PATH_48} ${TRAIN_PATH_49} \
        ${TRAIN_PATH_50} ${TRAIN_PATH_51} ${TRAIN_PATH_52} ${TRAIN_PATH_53} ${TRAIN_PATH_54} ${TRAIN_PATH_55} ${TRAIN_PATH_56} \
        ${TRAIN_PATH_60} ${TRAIN_PATH_61} ${TRAIN_PATH_62} ${TRAIN_PATH_63} ${TRAIN_PATH_64} ${TRAIN_PATH_65} ${TRAIN_PATH_66} ${TRAIN_PATH_67} ${TRAIN_PATH_68} ${TRAIN_PATH_69} \
        ${TRAIN_PATH_70} ${TRAIN_PATH_71} ${TRAIN_PATH_72} ${TRAIN_PATH_73} ${TRAIN_PATH_74} ${TRAIN_PATH_75} ${TRAIN_PATH_76} ${TRAIN_PATH_77} ${TRAIN_PATH_78} ${TRAIN_PATH_79} \
        ${TRAIN_PATH_80} ${TRAIN_PATH_81} \
    --eval-path ${EVAL_PATH_00} ${EVAL_PATH_01} ${EVAL_PATH_02} ${EVAL_PATH_03} ${EVAL_PATH_04} ${EVAL_PATH_05} ${EVAL_PATH_06} ${EVAL_PATH_07} ${EVAL_PATH_08} ${EVAL_PATH_09} \
        ${EVAL_PATH_20} ${EVAL_PATH_21} ${EVAL_PATH_22} ${EVAL_PATH_23} ${EVAL_PATH_24} ${EVAL_PATH_25} ${EVAL_PATH_26} ${EVAL_PATH_27} \
        ${EVAL_PATH_30} ${EVAL_PATH_31} \
        ${EVAL_PATH_40} ${EVAL_PATH_41} ${EVAL_PATH_42} ${EVAL_PATH_43} ${EVAL_PATH_44} ${EVAL_PATH_45} ${EVAL_PATH_46} ${EVAL_PATH_47} ${EVAL_PATH_48} ${EVAL_PATH_49} \
        ${EVAL_PATH_50} ${EVAL_PATH_51} ${EVAL_PATH_52} ${EVAL_PATH_53} ${EVAL_PATH_54} ${EVAL_PATH_55} ${EVAL_PATH_56} \
        ${EVAL_PATH_60} ${EVAL_PATH_61} ${EVAL_PATH_62} ${EVAL_PATH_63} ${EVAL_PATH_64} ${EVAL_PATH_65} ${EVAL_PATH_66} ${EVAL_PATH_67} ${EVAL_PATH_68} ${EVAL_PATH_69} \
        ${EVAL_PATH_70} ${EVAL_PATH_71} ${EVAL_PATH_72} ${EVAL_PATH_73} ${EVAL_PATH_74} ${EVAL_PATH_75} ${EVAL_PATH_76} ${EVAL_PATH_77} ${EVAL_PATH_78} ${EVAL_PATH_79} \
        ${EVAL_PATH_80} ${EVAL_PATH_81} \
    --model-size ${MODEL_SIZE} --save-path ${SAVE_PATH} --max-num ${MAX_NUM} \
    --short-range ${SHORT_RANGE} --patch-size ${PATCH_SIZE} --patch-num ${PATCH_NUM} --keep-size ${KEEP_SIZE} \
    --batch-size ${BATCH_SIZE} --learning-rate ${LEARNING_RATE} --momentum ${MOMENTUM} --weight-decay ${WEIGHT_DECAY} --lr-scheduler ${LR_SCHEDULER} \
    --fine-tune ${FINE_TUNE} --restore-from ${RESTORE_FROM} --snapshot-dir ${SNAPSHOT_DIR} \
    --start-iter ${START_ITER} --total-iter ${TOTAL_ITER} --gpus ${GPU_IDS}
