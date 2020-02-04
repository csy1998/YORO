# -*- coding: utf-8 -*-
"""
@Author     : Fei Wang
@Contact    : fei_wang@shannonai.com
@Time       : 2019/8/18 11:11
@Description: 
"""
import jsonlines
from tqdm import tqdm
from typing import List, Tuple, Dict
from process_data.edit_distance import min_edit_distance_path, edit_path_to_correction


def target_to_edit(sentence_pairs: List[Tuple[List[str], List[str]]], enable_replace=False) \
        -> List[Dict[str, List[Tuple[str, int, str]]]]:
    all_edits = []
    for source, target in tqdm(sentence_pairs):
        edit_path = min_edit_distance_path(source, target)
        edits = edit_path_to_correction(source, target, edit_path, enable_replace)
        instance = {
            "source": source,
            "target": target,
            "edits": edits
        }
        all_edits.append(instance)
    return all_edits


def load_data(source_file_path: str, target_file_path: str) -> List[Tuple[List[str], List[str]]]:
    sentence_pairs = []
    with open(source_file_path) as src_f, open(target_file_path) as tgt_f:
        for source, target in tqdm(zip(src_f.readlines(), tgt_f.readlines())):
            sentence_pairs.append((source.strip().split(" "), target.strip().split(" ")))
    return sentence_pairs


def generate(source_file_path: str, target_file_path: str, edit_file_path: str, enable_replace=False):
    sentences_pairs = load_data(source_file_path, target_file_path)
    edits = target_to_edit(sentences_pairs, enable_replace)
    with jsonlines.open(edit_file_path, mode="w") as writer:
        for edit in edits:
            writer.write(edit)
    return edits


if __name__ == "__main__":
    source_file_path = "/data/nfsdata2/wangfei/data/yoro/origin_data/data_raw/test.src-tgt.src"
    target_file_path = "/data/nfsdata2/wangfei/data/yoro/origin_data/data_raw/test.src-tgt.tgt"
    edit_file_path = "/data/nfsdata2/wangfei/data/yoro/test.edit.raw"
    generate(source_file_path, target_file_path, edit_file_path, enable_replace=False)
