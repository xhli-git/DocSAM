#!/bin/bash
uname -a
#date
#env
date


nohup sh run_test_all_dataset_parallel.sh > ./logs/log_run_test_all_dataset_parallel 2>&1 &

# mkdir -p "./logs/large_fine_tune/"
# nohup sh scripts/run_test_large_ctw1500.sh > ./logs/large_fine_tune/log_run_test_large_ctw1500 2>&1 &
# nohup sh scripts/run_test_large_doclaynet.sh > ./logs/large_fine_tune/log_run_test_large_doclaynet 2>&1 &
# nohup sh scripts/run_test_large_m6doc.sh > ./logs/large_fine_tune/log_run_test_large_m6doc 2>&1 &
# nohup sh scripts/run_test_large_scut_cab.sh > ./logs/large_fine_tune/log_run_test_large_scut_cab 2>&1 &
# nohup sh scripts/run_test_large_totaltext.sh > ./logs/large_fine_tune/log_run_test_large_totaltext 2>&1 &

