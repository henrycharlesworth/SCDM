#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python main.py --beta=0.7 --expt_tag="new_run" --use_normaliser --save_model
