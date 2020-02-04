#!/bin/sh

export PYTHONPATH=$PYTHONPATH:/home/shuyin/yoro

export CUDA_VISIBLE_DEVICES=0;
export NGPU=1; 

#DATA="/data/nfsdata2/shuyin/data/yoro/train_small"
#DATA="/data/nfsdata2/shuyin/data/yoro/train_16000"
DATA="/data/nfsdata2/shuyin/data/yoro/train_8000"

#OUTPUT="/data/nfsdata2/shuyin/data/yoro/checkpoints/checkpoint_small"
#OUTPUT="/data/nfsdata2/shuyin/data/yoro/checkpoints/checkpoint_1e-4_16000"
OUTPUT="/data/nfsdata2/shuyin/data/yoro/checkpoints/checkpoint_1e-4_8000_new"


MODEL="/data/nfsdata2/wangfei/models/transformer"

#python -m torch.distributed.launch --nproc_per_node=$NGPU ../nonautoregressive/run_gec.py \
python ../nonautoregressive/run_gec.py \
--task_name gec \
--data_dir  $DATA \
--output_dir $OUTPUT \
--max_seq_length 64 \
--per_gpu_eval_batch_size 64 \
--per_gpu_train_batch_size 64 \
--gradient_accumulation_steps 1 \
--num_train_epochs 20 \
--overwrite_output_dir \
--warmup_steps 100000 \
--weight_decay 0.01 \
--save_steps 1000000000 \
--logging_steps 10000 \
--learning_rate 1e-4 \
--do_train \
--do_eval \
--transformer \
--model_type bert \
--model_name_or_path $MODEL \
--alpha_add 1.0 \
--alpha_del 1.0 \
--alpha_vocab 1.0 \
--alpha_position 1.0 \
# --fp16 \
# --fp16_opt_level O2 \
