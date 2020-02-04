# -*- coding: utf-8 -*-
"""
@Author     : Fei Wang
@Contact    : fei_wang@shannonai.com
@Time       : 2019/9/7 14:41
@Description: 
"""
import os
import math
import torch
import argparse
from tqdm import tqdm
from more_itertools import chunked
from pytorch_transformers import BertTokenizer
from nonautoregressive.bert_for_gec import BertForNonAutoregressiveGec


def inference(args):
    
    model_path = args.model_path
    inference_model_path = args.inference_model_path
    data_path = args.data_path
    output_path = args.output_path
    batch_size = args.batch_size
    max_length = args.max_length
    topk = args.topk
    
    model = BertForNonAutoregressiveGec.from_pretrained(model_path)
    model.to("cuda")
    model.eval()

    tokenizer = BertTokenizer.from_pretrained(model_path)
    inference_tokenizer = BertTokenizer.from_pretrained(inference_model_path)
    
    print("load model from", model_path)

    if not os.path.exists(output_path):
        os.mknod(output_path)

    all_input_ids = []
    all_attention_mask = []
    raw_sentences = []
    unk_id = tokenizer.convert_tokens_to_ids(["<unk>"])[0]
    with open(data_path) as f:
        for line in f.readlines():
            sentence = line.strip().split()
            raw_sentences.append(sentence)
            tokens = sentence + ["<sep>"]
            tokens = tokens[:max_length]
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            for i in range(len(input_ids)):
                if input_ids[i] is None:
                    input_ids[i] = unk_id
            attention_mask = [1 for _ in range(len(input_ids))]
            padding_length = max_length - len(input_ids)
            input_ids = input_ids + ([0] * padding_length)
            attention_mask = attention_mask + ([0] * padding_length)
            all_input_ids.append(input_ids)
            all_attention_mask.append(attention_mask)
    print("load", len(raw_sentences), "data from", data_path)

    print("total batch", math.ceil(len(all_input_ids)/batch_size))

    all_edits = []
    max_edits_len = 64
    del_count_list, add_count_list = [], []
    for input_ids, attention_mask in tqdm(zip(chunked(all_input_ids, batch_size), chunked(all_attention_mask, batch_size))):
        input_ids = torch.LongTensor(input_ids).to("cuda")
        attention_mask = torch.LongTensor(attention_mask).to("cuda")
        with torch.no_grad():
            outputs = model(input_ids, attention_mask)
            del_logit, add_logit, vocab_logits, position_logits = outputs
            del_logit = del_logit.cpu()
            add_logit = add_logit.cpu()
            vocab_logits = vocab_logits.cpu()
            position_logits = position_logits.cpu()
            
            for i in range(del_logit.size()[0]):
                # 第i个句子
                del_count, add_count = 0, 0
                sentence_edits = []
                length = torch.sum(attention_mask[i])
                for j in range(length):
                    # 第j个单词
                    if del_logit[i][j].item() > 0.5:
                        del_count += 1
                        sentence_edits.append(("delete", j, ""))
                    if add_logit[i][j].item() > 0.5:
                        add_count += 1
                        mask = vocab_logits[i][j] > 0.5

                        _, indexes = torch.topk(vocab_logits[i][j] * mask.float(), topk, dim=-1)
                        positions = torch.gather(position_logits[i][j], -1, indexes).numpy().tolist()
                        
                        print("indexes: ", indexes)
                        print("positions: ", positions)
                        #indexes = torch.nonzero(vocab_logits[i][j] * mask.float()).view(-1).numpy().tolist()
                        tokens = inference_tokenizer.convert_ids_to_tokens(indexes.view(-1).numpy().tolist())
                        #positions = position_logits[i][j][mask].numpy().tolist()
                        add_words = [(token, pos) for token, pos in zip(tokens, positions)]
                        sorted(add_words, key=lambda x: x[1])
                        
                        for (token, pos) in add_words:
                            if token in ["<pad>", "<cls>", "<sep>"]:
                                continue
                            sentence_edits.append(("add", j, token))
                
                del_count_list.append(del_count)
                add_count_list.append(add_count)

                sentence_edits = sentence_edits[:max_edits_len]
                all_edits.append(sentence_edits)
            
    print("del: ", sum(del_count_list))
    print("add: ", sum(add_count_list))
    
    with open(output_path, "w") as f:
        for sentence, edits in zip(raw_sentences, all_edits):
            pred = edit_sentence(sentence, edits)
            f.write(" ".join(pred) + "\n")
    print("write result to", output_path)


def edit_sentence(source, edits):
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

    return predict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", default=None, type=str, required=True,
                        help="path for model to test")
    parser.add_argument("--data_path", default=None, type=str, required=True,
                        help="path for test data")
    parser.add_argument("--output_path", default=None, type=str, required=True,
                        help="path for output result")
    parser.add_argument("--batch_size", default=32, type=int, required=False,
                        help="batch size")
    parser.add_argument("--max_length", default=128, type=int, required=False,
                        help="max length")
    parser.add_argument("--topk", default=3, type=int, required=False,
                        help="topk")
    parser.add_argument("--inference_model_path", default=None, type=str, required=True,
                        help="path for another tokenizer")
    args = parser.parse_args()
    
    inference(args)
