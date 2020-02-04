#!/bin/sh

export PYTHONPATH=$PYTHONPATH:/home/chenshuyin/yoro

#predict_path="/data/nfsdata2/shuyin/data/yoro/test_tgt_1e-3.txt"
predict_path="/data/nfsdata2/shuyin/data/yoro/test_tgt_1e-5_10.txt"
#predict_path="/data/nfsdata2/shuyin/data/yoro/test_tgt_1e-5_10.txt"

m2_golden=/data/nfsdata2/wangfei/data/yoro/official-2014.combined.m2

python2 m2scorer/m2scorer -v ${predict_path} ${m2_golden}
