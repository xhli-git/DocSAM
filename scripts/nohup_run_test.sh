#!/bin/bash
uname -a
#date
#env
date


nohup sh run_test_large_all_dataset.sh > ./logs/log_run_test_large_all_dataset 2>&1 &
