# -*- coding: utf-8 -*-
"""
@Author     : Fei Wang
@Contact    : fei_wang@shannonai.com
@Time       : 2019/8/21 20:43
@Description: 
"""
import jsonlines
from tqdm import tqdm


def check_edits(source, target, edits):
    # build edit dict
    edit_dict = dict()
    edit_poses = []
    for edit in edits:
        etype, pos, char = edit
        edit_poses.append(pos)
        if pos not in edit_dict:
            edit_dict[pos] = {
                "del": False,
                "add": []
            }
        if etype == "delete":
            edit_dict[pos]["del"] = True
        elif etype == "add":
            edit_dict[pos]["add"].append(char)

    # predict
    predict = []
    for origin_index in range(len(source) + 1):  # plus 1: may add at last
        if origin_index not in edit_dict:
            if origin_index < len(source):
                predict.append(source[origin_index])
            continue
        edits = edit_dict[origin_index]
        predict.extend(edits["add"])
        if not edits["del"] and origin_index < len(source):
                predict.append(source[origin_index])
        origin_index += 1

    if predict != target:
        print(source)
        print(predict)
        print(target)


def run(edit_file_path):
    with open(edit_file_path) as f:
        for item in tqdm(jsonlines.Reader(f)):
            source = item["source"]
            target = item["target"]
            edits = item["edits"]
            check_edits(source, target, edits)


if __name__ == "__main__":
    edit_file_path = "/data/nfsdata2/wangfei/data/grammar_error_correction/train.edit.raw"
    run(edit_file_path)
