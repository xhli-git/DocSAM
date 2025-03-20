#!/bin/bash
uname -a
#date
#env
date


nohup sh run_test.sh > ./logs/log_test 2>&1 &
