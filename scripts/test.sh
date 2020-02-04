#!/bin/sh

export PYTHONPATH=$PYTHONPATH:/home/chenshuyin/yoro

# MODEL="/data/nfsdata2/shuyin/data/yoro/checkpoints/checkpoint_1e-3"
# OUTPUT="/data/nfsdata2/shuyin/data/yoro/test_tgt_1e-3.txt"

# MODEL="/data/nfsdata2/shuyin/data/yoro/checkpoints/checkpoint_5e-4"
# OUTPUT="/data/nfsdata2/shuyin/data/yoro/test_tgt_5e-4.txt"

MODEL="/data/nfsdata2/shuyin/data/yoro/checkpoints/checkpoint_1e-5_10"
OUTPUT="/data/nfsdata2/shuyin/data/yoro/test_tgt_1e-5_10.txt"

# new vocab file
INFER_MODEL="/data/nfsdata2/shuyin/data/yoro/model_16000"

CUDA_VISIBLE_DEVICES=2 python ../nonautoregressive/gec_inference.py \
--model_path $MODEL \
--data_path /data/nfsdata2/shuyin/data/yoro/test_src_no_split.txt \
--output_path $OUTPUT \
--batch_size 64 \
--max_length 128 \
--topk 3 \
--inference_model_path $INFER_MODEL \
