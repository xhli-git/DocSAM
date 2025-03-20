#!/bin/bash
uname -a
#date
#env
date


# nohup sh run_train.sh > ./logs/log_train 2>&1 &
nohup sh run_train_curriculum.sh > ./logs/log_train 2>&1 &
