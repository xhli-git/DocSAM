#!/bin/bash
uname -a
#date
#env
date

nohup sh run_train_curriculum_large_all_dataset.sh > ./logs/log_run_train_curriculum_large_all_dataset 2>&1 &
