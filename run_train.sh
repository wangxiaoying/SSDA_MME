#!/bin/sh
CUDA_VISIBLE_DEVICES=0 python main.py --method $1 --dataset office_home --source $3 --target $4 --net $2 --save_check
