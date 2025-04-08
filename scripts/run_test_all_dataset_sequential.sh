#!/bin/bash
uname -a
#date
#env
date


# Replace to your own data path
DATA_PATH=/local_data1/xhli/data4/datas/DownStreamData
# DATA_PATH=/data6/home/xhli/lustre/data

# Datasets used for evaluation during training. 
# Only several datasets are selected for saving time.

# Layout
EVAL_PATH_00=${DATA_PATH}/Layout/BaDLAD/data/BaDLAD/train/
EVAL_PATH_01=${DATA_PATH}/Layout/CDLA/data/CDLA/val/
EVAL_PATH_02=${DATA_PATH}/Layout/D4LA/data/D4LA/test/
EVAL_PATH_03=${DATA_PATH}/Layout/DocBank/data/DocBank/test/
EVAL_PATH_04=${DATA_PATH}/Layout/DocLayNet/data/DocLayNet_core/test/
EVAL_PATH_05=${DATA_PATH}/Layout/ICDAR2017_POD/data/ICDAR2017_POD/test/
EVAL_PATH_06=${DATA_PATH}/Layout/IIIT-AR-13K/data/IIIT-AR-13K/test/
EVAL_PATH_07=${DATA_PATH}/Layout/M6Doc/data/M6Doc/test/
EVAL_PATH_08=${DATA_PATH}/Layout/PubLayNet/data/PubLayNet/test/
EVAL_PATH_09=${DATA_PATH}/Layout/RanLayNet/data/RanLayNet/val/


# Ancient
EVAL_PATH_20=${DATA_PATH}/Ancient/CASIA-AHCDB/data/Style1/test/
EVAL_PATH_21=${DATA_PATH}/Ancient/CASIA-AHCDB/data/Style2/test/
EVAL_PATH_22=${DATA_PATH}/Ancient/CHDAC-2022/data/CHDAC-2022/train/
EVAL_PATH_23=${DATA_PATH}/Ancient/ICDAR2019HDRC/data/ICDAR2019HDRC/train/
EVAL_PATH_24=${DATA_PATH}/Ancient/SCUT-CAB/data/logical/test/
EVAL_PATH_25=${DATA_PATH}/Ancient/SCUT-CAB/data/physical/test/
EVAL_PATH_26=${DATA_PATH}/Ancient/TKHMTH2200/data/TKHMTH2200/test/
EVAL_PATH_27=${DATA_PATH}/Layout/HJDataset/data/HJDataset/test/

# Handwritten
EVAL_PATH_30=${DATA_PATH}/Handwritten/CASIA-HWDB/data/CASIA-HWDB/test/
EVAL_PATH_31=${DATA_PATH}/Handwritten/SCUT-HCCDoc/data/SCUT-HCCDoc/test/

# Table
EVAL_PATH_40=${DATA_PATH}/Table/FinTabNet/data/FinTabNet/test/
EVAL_PATH_41=${DATA_PATH}/Table/FinTabNet/data/FinTabNet.c/FinTabNet.c-Structure/test/
EVAL_PATH_42=${DATA_PATH}/Table/ICDAR2013/data/ICDAR2013/competition-dataset/
EVAL_PATH_43=${DATA_PATH}/Table/ICDAR2013/data/ICDAR2013.c/ICDAR2013.c-Structure/test/
EVAL_PATH_44=${DATA_PATH}/Table/ICDAR2017_POD/data/ICDAR2017_POD/test/
EVAL_PATH_45=${DATA_PATH}/Table/ICDAR2019_cTDaR/data/Archive/test_track_b2/
EVAL_PATH_46=${DATA_PATH}/Table/ICDAR2019_cTDaR/data/Modern/test_track_b2/
EVAL_PATH_47=${DATA_PATH}/Table/NTable/data/NTable-gen/test/
EVAL_PATH_48=${DATA_PATH}/Table/NTable/data/NTable-ori_cam_ver.2/NTable-cam/test/
EVAL_PATH_49=${DATA_PATH}/Table/PubTables-1M/data/PubTables-1M-Detection/test/
EVAL_PATH_50=${DATA_PATH}/Table/PubTables-1M/data/PubTables-1M-Structure/test/
EVAL_PATH_51=${DATA_PATH}/Table/PubTabNet/data/PubTabNet/val/
EVAL_PATH_52=${DATA_PATH}/Table/STDW/data/STDW/
EVAL_PATH_53=${DATA_PATH}/Table/TableBank/data/TableBank/Detection/latex/test/
EVAL_PATH_54=${DATA_PATH}/Table/TableBank/data/TableBank/Detection/word/test/
EVAL_PATH_55=${DATA_PATH}/Table/TNCR/data/TNCR/test/
EVAL_PATH_56=${DATA_PATH}/Table/WTW/data/WTW/test/

# SceneText
EVAL_PATH_60=${DATA_PATH}/SceneText/CASIA-10k/data/CASIA-10k/test/
EVAL_PATH_61=${DATA_PATH}/SceneText/COCO-Text/data/cocotext/val/
EVAL_PATH_62=${DATA_PATH}/SceneText/CTW-1500/data/CTW-1500/test/
EVAL_PATH_63=${DATA_PATH}/SceneText/CTW-Public/data/test/
EVAL_PATH_64=${DATA_PATH}/SceneText/HUST-TR400/data/HUST-TR400/
EVAL_PATH_65=${DATA_PATH}/SceneText/ICDAR_2011_2015_RRC/data/Task1-Born-Digital-Images/Task-1.1-Text-Localization-2013-edition/test/
EVAL_PATH_66=${DATA_PATH}/SceneText/ICDAR_2011_2015_RRC/data/Task2-Focused-Scene-Text/Task-2.1-Text-Localization-2013-edition/test/
EVAL_PATH_67=${DATA_PATH}/SceneText/ICDAR_2011_2015_RRC/data/Task4-Incidental-Scene-Text/Task-4.1-Text-Localization-2015-edition/test/
EVAL_PATH_68=${DATA_PATH}/SceneText/ICDAR_2017_RCTW/data/RCTW-17/train/
EVAL_PATH_69=${DATA_PATH}/SceneText/ICDAR_2017_RRC_MLT/data/Task-1-Multi-script-text-detection/val/
EVAL_PATH_70=${DATA_PATH}/SceneText/ICDAR_2019_RRC_ArT/data/task1-scene-text-detection/train/
EVAL_PATH_71=${DATA_PATH}/SceneText/ICDAR_2019_RRC_LSVT/data/LSVT-2019/train_full/
EVAL_PATH_72=${DATA_PATH}/SceneText/ICDAR_2019_RRC_MLT/data/Task-1-Multi-script-text-detection/train/
EVAL_PATH_73=${DATA_PATH}/SceneText/ICDAR_2019_RRC_MLT/data/Task-4-End-to-End-text-detection-and-recognition/
EVAL_PATH_74=${DATA_PATH}/SceneText/ICDAR_2019_RRC_ReCTS/data/ReCTS/train/
EVAL_PATH_75=${DATA_PATH}/SceneText/ICDAR_2023_RRC_HierText/data/HierText-2023/val/
EVAL_PATH_76=${DATA_PATH}/SceneText/ICDAR_2023_RRC_ReST/data/train/
EVAL_PATH_77=${DATA_PATH}/SceneText/ICPR_2018_MTWI/data/MIWI-2018/train/
EVAL_PATH_78=${DATA_PATH}/SceneText/MSRA-TD500/data/MSRA-TD500/test/
EVAL_PATH_79=${DATA_PATH}/SceneText/ShopSign/data/ShopSign_1265/
EVAL_PATH_80=${DATA_PATH}/SceneText/Total-Text/data/TotalText/test/
EVAL_PATH_81=${DATA_PATH}/SceneText/USTB-SV1K/data/USTB-SV1K/test/


# Define evaluation dataset paths (replace with actual paths)
DATASETS=(
    ${EVAL_PATH_00} ${EVAL_PATH_01} ${EVAL_PATH_02} ${EVAL_PATH_03} ${EVAL_PATH_04} ${EVAL_PATH_05} ${EVAL_PATH_06} ${EVAL_PATH_07} ${EVAL_PATH_08} ${EVAL_PATH_09} \
    ${EVAL_PATH_20} ${EVAL_PATH_21} ${EVAL_PATH_22} ${EVAL_PATH_23} ${EVAL_PATH_24} ${EVAL_PATH_25} ${EVAL_PATH_26} ${EVAL_PATH_27} \
    ${EVAL_PATH_30} ${EVAL_PATH_31} \
    ${EVAL_PATH_40} ${EVAL_PATH_41} ${EVAL_PATH_42} ${EVAL_PATH_43} ${EVAL_PATH_44} ${EVAL_PATH_45} ${EVAL_PATH_46} ${EVAL_PATH_47} ${EVAL_PATH_48} ${EVAL_PATH_49} \
    ${EVAL_PATH_50} ${EVAL_PATH_51} ${EVAL_PATH_52} ${EVAL_PATH_53} ${EVAL_PATH_54} ${EVAL_PATH_55} ${EVAL_PATH_56} \
    ${EVAL_PATH_60} ${EVAL_PATH_61} ${EVAL_PATH_62} ${EVAL_PATH_63} ${EVAL_PATH_64} ${EVAL_PATH_65} ${EVAL_PATH_66} ${EVAL_PATH_67} ${EVAL_PATH_68} ${EVAL_PATH_69} \
    ${EVAL_PATH_70} ${EVAL_PATH_71} ${EVAL_PATH_72} ${EVAL_PATH_73} ${EVAL_PATH_74} ${EVAL_PATH_75} ${EVAL_PATH_76} ${EVAL_PATH_77} ${EVAL_PATH_78} ${EVAL_PATH_79} \
    ${EVAL_PATH_80} ${EVAL_PATH_81} \
)

# GPU configuration
GPUS_IDS=(0 1 2 3 4 5 6 7)
NUM_GPUS=${#GPUS_IDS[@]}

# Initialize GPU process ID array and availability status
# -1: Free and available
# -2: Occupied by other users
# Other values: Process ID of the current script's task running on the GPU
gpu_pids=()

# Initialization phase: Check if each GPU is occupied by other processes
echo "Initializing GPU availability..."
for ((i=0; i<NUM_GPUS; i++)); do
    gpu_id=${GPUS_IDS[i]}  # Get the GPU ID from the list
    # Query NVIDIA System Management Interface (nvidia-smi) to check for existing processes on the GPU
    existing_pids=$(nvidia-smi --id=${gpu_id} --query-compute-apps=pid --format=csv,noheader | grep -v '^$')
    if [[ -n $existing_pids ]]; then
        # If there are existing PIDs, mark the GPU as unavailable (-2)
        echo "GPU ${gpu_id} is already in use by other processes. Marked as unavailable."
        gpu_pids[i]=-2
    else
        # Otherwise, mark the GPU as free (-1)
        gpu_pids[i]=-1
    fi
done

# Function: Check GPU status and release resources for completed tasks
function check_gpu_status() {
    for ((i=0; i<NUM_GPUS; i++)); do
        current_pid=${gpu_pids[i]}  # Get the current PID or status of the GPU
        gpu_id=${GPUS_IDS[i]}       # Get the GPU ID
        
        # Check if the GPU is currently occupied by other processes
        existing_pids=$(nvidia-smi --id=${gpu_id} --query-compute-apps=pid --format=csv,noheader | grep -v '^$')
        
        if [[ -n $existing_pids ]]; then
            # If the GPU was previously marked as free (-1), update it to unavailable (-2)
            if [[ $current_pid -eq -1 ]]; then
                echo "GPU ${gpu_id} is now in use by other processes. Marking as unavailable."
                gpu_pids[i]=-2
            fi
            continue  # Skip further checks for this GPU
        fi

        # Check if the current script's process has completed (only if PID is not -1 or -2)
        if [[ $current_pid -ne -1 && $current_pid -ne -2 ]]; then
            # Use kill -0 to check if the process is still running
            if ! kill -0 $current_pid 2>/dev/null; then
                # If the parent process has ended, check for any child processes
                children=$(ps -o pid --ppid $current_pid --no-headers)
                if [[ -z $children ]]; then
                    # If no child processes are running, release the GPU resources
                    echo "GPU ${gpu_id} task completed, releasing resources"
                    gpu_pids[i]=-1  # Mark the GPU as free
                else
                    # If child processes are still running, keep the GPU marked as occupied
                    echo "GPU ${gpu_id} parent process ended but child processes still running - keeping occupied"
                fi
            fi
        fi
    done
}

# Log folder
task=base
# task=large
# task=large_keepsize
mkdir -p "./logs/${task}"

# Main loop to assign datasets to GPUs
for dataset in "${DATASETS[@]}"; do
    echo "Processing dataset: $dataset"

    # Replace slashes in dataset path with underscores for log file name
    log_name="${dataset//\//_}"
    
    # Check if log file already exists and skip processing if it does
    if [ -e "./logs/${task}/${log_name}.log" ]; then
        echo "Log file for dataset '$dataset' already exists. Skipping..."
        continue
    fi
    
    while true; do
        check_gpu_status  # Check and release completed tasks
        
        # Try to find an available GPU
        for ((i=0; i<NUM_GPUS; i++)); do
            if [[ ${gpu_pids[i]} -eq -1 ]]; then
                # GPU is free, start the job
                echo "Assigning GPU ${GPUS_IDS[i]} to process $dataset"
                nohup sh ./run_test_${task}.sh $dataset ${GPUS_IDS[i]} > "./logs/${task}/${log_name}.log" 2>&1 &
                gpu_pids[i]=$!  # Save the process ID
                break 2  # Break out of both loops to proceed to next dataset
            fi
        done
        
        # All GPUs are busy, wait and retry
        echo "All GPUs busy, waiting 10 seconds..."
        sleep 10
    done
done

# Wait for all remaining processes to finish
for ((i=0; i<NUM_GPUS; i++)); do
    if [[ ${gpu_pids[i]} -gt 0 ]]; then
        wait ${gpu_pids[i]}  # Wait only for valid PIDs
    fi
done

echo "All datasets completed successfully!"
