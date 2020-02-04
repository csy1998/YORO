import json

train_data_path = "/data/nfsdata2/wangfei/data/yoro/train.edit.raw"
dev_data_path = "/data/nfsdata2/wangfei/data/yoro/dev.edit.raw"

train_vocab_path = "/home/chenshuyin/yoro/vocab/vocab_train.txt"

middle_vocab_path = "/home/chenshuyin/yoro/vocab/vocab_middle.txt"
small_vocab_path = "/home/chenshuyin/yoro/vocab/vocab_small.txt"

middle_train_data = "/data/nfsdata2/shuyin/data/yoro/train.edit.raw.16000"
middle_dev_data = "/data/nfsdata2/shuyin/data/yoro/dev.edit.raw.16000"

small_train_data = "/data/nfsdata2/shuyin/data/yoro/train.edit.raw.8000"
small_dev_data = "/data/nfsdata2/shuyin/data/yoro/dev.edit.raw.8000"


def get_vocab(data_file, vocab_file):
    d = {}
    with open(data_file, 'r') as f:
        for line in f:
            line = json.loads(line)
            edits = line['edits']
            for lst in edits:
                if lst[0] != 'add':
                    continue
                word = lst[-1]
                if word not in d.keys():
                    d[word] = 0
                d[word] += 1
    d = sorted(d.items(), key=lambda item: item[1], reverse=True)

    with open(vocab_file, 'w') as f:
        for item in d:
            word, freq = item
            f.write(str(word) +  " " + str(freq) + '\n')
    print("vocab size: ", len(d))



def generate_new_vocab(input_vocab, output_vocab, max_count=4):

    out_file = open(output_vocab, 'w')
    out_file.write('<pad>' + '\n')
    out_file.write('<cls>' + '\n')
    out_file.write('<sep>' + '\n')
    out_file.write('<unk>' + '\n')

    count = 0
    with open(input_vocab, 'r') as f:
        for line in f:
            line = line.strip()
            line = line.split(' ')
            word, freq = line[0], int(line[1])
            if freq <= max_count or word == '<unk>':
                continue
            out_file.write(word + '\n')
            count += 1
    print("new vocab size: ", count)

# 根据 vocab_file 筛掉 input_file 中一部分数据
def generate_data(input_file, output_file, vocab_file):
    words_all = []
    with open(vocab_file, 'r') as f:
        for line in f:
            line = line.strip()
            words_all.append(line)
    
    # generate data
    out_file = open(output_file, 'w')
    count = 0
    with open(input_file, 'r') as f:
        for line in f:
            json_line = json.loads(line)
            edits = json_line['edits']
            flag = 0
            for lst in edits:
                if lst[0] != 'add':
                    continue
                word = lst[-1]
                if word not in words_all or word == '<unk>':
                    flag = 1
                    count += 1
                    break
            
            if flag == 0:
                out_file.write(line)
    
    print("data not save: ", count)


def test():
    d = {}
    with open(train_vocab_path, 'r') as f:
        for line in f:
            line = line.strip()
            line = line.split(' ')
            word, freq = line[0], int(line[1])
            d[word] = freq

    count = 0
    min_count = 15
    with open(dev_data_path, 'r') as f:
        for line in f:
            line = json.loads(line)
            edits = line['edits']
            for lst in edits:
                if lst[0] != 'add':
                    continue
                word = lst[-1]
                if word not in d.keys():
                    print(word)
                    continue
                if d[word] <= min_count or word == "<unk>":
                #if word == "<unk>":
                    count += 1
                    break
    print(count)


if __name__ == '__main__':
    
    # test()

    # get_vocab(train_data_path, train_vocab_path)

    # generate new vocab (middle / small)
    # generate_new_vocab(train_vocab_path, middle_vocab_path, max_count=4)
    # generate_new_vocab(train_vocab_path, small_vocab_path, max_count=15)

    # generate data with middle vocab
    generate_data(train_data_path, middle_train_data, middle_vocab_path)
    generate_data(dev_data_path, middle_dev_data, middle_vocab_path)

    # generate data with small vocab
    generate_data(train_data_path, small_train_data, small_vocab_path)
    generate_data(dev_data_path, small_dev_data, small_vocab_path)
