# -*- coding: utf-8 -*-
"""
@Author     : Fei Wang
@Contact    : fei_wang@shannonai.com
@Time       : 2019/8/21 16:17
@Description: Utilities to work with grammar error correction examples
"""
from __future__ import absolute_import, division, print_function

import os
import logging
from io import open
import numpy
import jsonlines
import torch

from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

logger = logging.getLogger(__name__)


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, attention_mask, del_label, add_label, add_vocab_index, add_vocab_position):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.del_label = del_label
        self.add_label = add_label
        self.add_vocab_index = add_vocab_index
        self.add_vocab_position = add_vocab_position


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    @classmethod
    def _read_jsonlines(cls, input_file):
        """Reads a jsonlines file."""
        items = []
        with open(input_file, "r") as f:
            for item in jsonlines.Reader(f):
                items.append(item)
            return items


class GecProcessor(DataProcessor):
    """Processor for the Detect data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        train_file_name = "train.json"
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, train_file_name)))
        return self._read_jsonlines(os.path.join(data_dir, train_file_name))

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._read_jsonlines(os.path.join(data_dir, "dev.json"))


def convert_examples_to_features(examples, max_seq_length, tokenizer):
    features = []
    #vocab_size = 49999
    vocab_size = 16250
    max_add_num = max_seq_length - 1

    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        # source tokens
        tokens = example["source"] + ["<sep>"]
        tokens = tokens[:max_seq_length]

        # del label
        del_label = example["del"]  # already add label for <sep>
        del_label = del_label[:max_seq_length]

        # add label
        add_label = example["add"]  # already add label for <sep>
        add_label = add_label[:max_seq_length]

        # add token index
        add_vocab_index = []
        for i, j in zip(example["add_index"], example["add_vocab_index"]):
            if i > max_add_num:
                break
            add_vocab_index.append(i * vocab_size + j)
        add_vocab_index = add_vocab_index[:max_add_num]

        # add token position
        add_vocab_position = []
        pre_id = -1
        count = 0
        for i in example["add_index"]:
            if i > max_add_num:
                break
            if i != pre_id:
                count = 0
                pre_id = i
            else:
                count += 1
            add_vocab_position.append(count)
        add_vocab_position = add_vocab_position[:max_add_num]

        # index
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # attention mask
        attention_mask = [1 for _ in range(len(input_ids))]

        # padding
        padding_length = max_seq_length - len(input_ids)
        input_ids = input_ids + ([0] * padding_length)
        attention_mask = attention_mask + ([0] * padding_length)
        del_label = del_label + ([0] * padding_length)
        add_label = add_label + ([0] * padding_length)

        add_padding_length = max_add_num - len(add_vocab_index)
        add_vocab_index = add_vocab_index + ([-1] * add_padding_length)
        add_vocab_position = add_vocab_position + ([-1] * add_padding_length)

        assert len(input_ids) == max_seq_length
        assert len(attention_mask) == max_seq_length
        assert len(del_label) == max_seq_length
        assert len(add_label) == max_seq_length
        assert len(add_vocab_index) == max_add_num
        assert len(add_vocab_position) == max_add_num

        features.append(
            InputFeatures(input_ids=input_ids,
                          attention_mask=attention_mask,
                          del_label=del_label,
                          add_label=add_label,
                          add_vocab_index=add_vocab_index,
                          add_vocab_position=add_vocab_position)
        )
    return features


def compute_metrics(inputs, outputs):
    del_labels = inputs['del_label']
    add_labels = inputs['add_label']
    masks = inputs['attention_mask']

    del_logit, add_logit = outputs[0], outputs[1]
    # print("masks: ", masks.shape, masks)
    # print("del_logit: ", del_logit.shape, del_logit)
    # print("add_logit: ", add_logit.shape, add_logit)
    
    masks = masks.type(torch.ByteTensor).to("cuda")
    with torch.no_grad():
        del_preds = torch.masked_select(del_logit, masks)
        del_labels = torch.masked_select(del_labels, masks)
        add_preds = torch.masked_select(add_logit, masks)
        add_labels = torch.masked_select(add_labels, masks)

    # print("del_preds: ", del_preds.shape, del_preds)
    # print("add_preds: ", add_preds.shape, add_preds)

    del_preds = (del_preds >= 0.5)
    add_preds = (add_preds >= 0.5)

    del_labels = del_labels.cpu().numpy()
    add_labels = add_labels.cpu().numpy()
    del_preds = del_preds.cpu().numpy()
    add_preds = add_preds.cpu().numpy()
    
    print("del_report: \n", classification_report(del_labels, del_preds))
    print("add_report: \n", classification_report(add_labels, add_preds))


processors = {
    "gec": GecProcessor,
}

output_modes = {
    "gec": "classification",
}
