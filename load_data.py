# encoding:utf-8
import json


# 加载原始数据，进行content/label分割
def load_message():
    content = []
    label = []
    lines =[]

    # 打开原数据文件
    with open('data_train.txt') as fr:
        data_size = 0
        line = fr.readline()
        while line:
            lines.append(line)
            data_size += 1
            if data_size == 20000:
                break
            line = fr.readline()
        # 记录数据规模
        num = len(lines)
        for i in range(num):
            message = lines[i].split('\t')
            label.append(message[0])
            content.append(message[1])
    return num, content, label


# 将分割后的原始数据存到json
def data_storage(content, label):
    with open('RawData/train_content_2w.json', 'w') as f:
        json.dump(content, f)
    with open('RawData/train_label_2w.json', 'w') as f:
        json.dump(label, f)


if '__main__' == __name__:
    num, content, label = load_message()
    data_storage(content, label)
