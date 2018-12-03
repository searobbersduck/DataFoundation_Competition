# !/usr/bin/env python3

import os
import pandas as pd

subjects_list = ['动力','价格','内饰','配置','安全性','外观','操控','油耗','空间','舒适性']
subjects_dict = {}
cnt = 0
for subject in subjects_list:
    subjects_dict[subject] = cnt
    cnt += 1

def get_sentiment_label(infile):
    df = pd.read_csv(infile)
    df_grouped = df.groupby(by='content_id')
    sents_list = []
    labels_list = []
    for name, group in df_grouped:
        len_group = len(group)
        if len_group >= 1:
            sents_list.append(list(group['content'])[0])
        labels = [0]*10
        for index, row in group.iterrows():
            subject_id = subjects_dict[row['subject']]
            labels[subject_id] += int(row['sentiment_value']) + 2
        labels = list(map(str, labels))
        labels = ' '.join(labels)
        labels_list.append(labels)
    return sents_list, labels_list

def gen_sentiment_files():
    infile = '../data/train.csv'
    outdir = '../data'
    sents_list, labels_list = get_sentiment_label(infile)
    assert len(sents_list) == len(labels_list)
    total_num = len(sents_list)
    train_num = int(0.8*total_num)
    val_num = int(0.9*total_num)
    with open(os.path.join(outdir, 'sents_train.txt'), 'w', encoding='utf8') as f:
        f.write('\n'.join(sents_list[:train_num]))
    with open(os.path.join(outdir, 'labels_train.txt'), 'w', encoding='utf8') as f:
        f.write('\n'.join(labels_list[:train_num]))
    with open(os.path.join(outdir, 'sents_val.txt'), 'w', encoding='utf8') as f:
        f.write('\n'.join(sents_list[train_num:val_num]))
    with open(os.path.join(outdir, 'labels_val.txt'), 'w', encoding='utf8') as f:
        f.write('\n'.join(labels_list[train_num:val_num]))
    with open(os.path.join(outdir, 'sents_test.txt'), 'w', encoding='utf8') as f:
        f.write('\n'.join(sents_list[val_num:]))
    with open(os.path.join(outdir, 'labels_test.txt'), 'w', encoding='utf8') as f:
        f.write('\n'.join(labels_list[val_num:]))

if __name__ == '__main__':
    # get_sentiment_label('../data/train.csv', '内饰')
    gen_sentiment_files()

