## train 

`bash scripts/train.sh`

## test

下载m2_scorer

`bash scripts/download_m2.sh`

生成预测结果

`bash scripts/test.sh`

计算m2得分

`bash scripts/evaluate_m2.sh`

## dataset

数据源：[Improving Grammatical Error Correction via Pre-Training a
Copy-Augmented Architecture with Unlabeled Data](https://github.com/zhawe01/fairseq-gec)

完整训练数据 （train 116w, dev 0.4w）

`/data/nfsdata2/wangfei/data/yoro/train_all`

小规模训练数据

`/data/nfsdata2/wangfei/data/yoro/train_small`

测试数据

`/data/nfsdata2/wangfei/data/yoro/test_src_no_split.txt`

数据源词表（词表前四位为 `<pad> <cls> <sep> <unk>`）

`/data/nfsdata2/wangfei/data/yoro/vocab.txt`

## model

/data/nfsdata2/wangfei/models/transformer

## todo

1. generate & evaluation

1. loss 比例调整 

1. position loss 改用 L1
    
1. position loss 使用相对位置排序loss

1. position loss mask 没有add位置

1. 目前的test是没有分句的，分句后预计会涨点

1. replace编辑类型

1. 与bert结合
