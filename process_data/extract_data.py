# -*- coding: utf-8 -*-
"""
@Author     : Fei Wang
@Contact    : fei_wang@shannonai.com
@Time       : 2019/8/21 18:10
@Description: 
"""
from tqdm import tqdm
import jsonlines
from pytorch_transformers import BertTokenizer


def load_dict(file):
    vocab_index_map = {}
    with open(file) as f:
        for index, line in enumerate(f.readlines()):
            vocab = line.strip().split()[0]
            vocab_index_map[vocab] = index
    return vocab_index_map


def extract_base(edit_data_path, output_data_path, vocab_dict):
    with open(edit_data_path) as f, jsonlines.open(output_data_path, mode='w') as writer:
        for item in tqdm(jsonlines.Reader(f)):
            new_item = dict()
            new_item["source"] = item["source"]
            new_item["add"] = [0 for _ in range(len(new_item["source"]) + 1)]  # plus 1: may add at last
            new_item["del"] = [0 for _ in range(len(new_item["source"]) + 1)]
            new_item["add_index"] = []
            new_item["add_vocab_index"] = []
            edits = item["edits"]
            for edit in edits:
                edit_type, char_index, vocab = edit
                if edit_type == "delete":
                    new_item["del"][char_index] = 1
                elif edit_type == "add":
                    vocab_index = vocab_dict[vocab]
                    new_item["add"][char_index] = 1
                    new_item["add_index"].append(char_index)
                    new_item["add_vocab_index"].append(vocab_index)
                else:
                    raise print("illegal edit type")
            writer.write(new_item)


if __name__ == "__main__":
    #dict_file = "/data/nfsdata2/wangfei/data/yoro/vocab.txt"
    #input_path = "/data/nfsdata2/wangfei/data/yoro/train.edit.raw"
    #output_path = "/data/nfsdata2/wangfei/data/yoro/train.edit.base"
    
    # dict_file = "/home/chenshuyin/yoro/vocab/vocab_middle.txt"
    # input_path = "/data/nfsdata2/shuyin/data/yoro/train.edit.raw.16000"
    # output_path = "/data/nfsdata2/shuyin/data/yoro/train_16000/train.json"

    # dict_file = "/home/chenshuyin/yoro/vocab/vocab_middle.txt"
    # input_path = "/data/nfsdata2/shuyin/data/yoro/dev.edit.raw.16000"
    # output_path = "/data/nfsdata2/shuyin/data/yoro/train_16000/dev.json"

    # dict_file = "/home/chenshuyin/yoro/vocab/vocab_small.txt"
    # input_path = "/data/nfsdata2/shuyin/data/yoro/train.edit.raw.8000"
    # output_path = "/data/nfsdata2/shuyin/data/yoro/train_8000/train.json"

    dict_file = "/home/chenshuyin/yoro/vocab/vocab_small.txt"
    input_path = "/data/nfsdata2/shuyin/data/yoro/dev.edit.raw.8000"
    output_path = "/data/nfsdata2/shuyin/data/yoro/train_8000/dev.json"
    
    vocabs = load_dict(dict_file)
    extract_base(input_path, output_path, vocabs)
