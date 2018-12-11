#!/usr/bin/env bash

#model_path = ./saved_final_model/wmt16_compress.bin
train_log="train."${job_name}".log"
model_path=$1
PYTHONPATH="." \
python3 nmt/train.py --seed 45 \
  --min_freq 1 \
  --valid_max_num 4 \
  --save_model 1000 \
  --batch_s 128 \
  --tok --lower \
  --save_model_after 200 \
  --max_ep 20 \
  --exp wmt16_full  \
  --save_best \
  --valid_every 100 \
  --multi-gpu\
  --mode test \
  --load_model ${model_path} \
  --compress \
  --num_bl 6