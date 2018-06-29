#!/usr/bin/python
# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import sys
from cut_doc import cutDoc

reload(sys)
sys.setdefaultencoding('utf-8')

# 加载数据
def transform_data(data_path):
    """
    将数据转换为label+"\t"+doc_text的一条条记录存储在all_preprocess.csv
    """
    text_list = []
    category_list = []
    cut_doc_obj = cutDoc()
    i = 0
    with open(data_path, "r") as f:
        for line in f:
            if line.split()[-1] != '3':
                sentence = line.split()[:-1][0]
                if len(cut_doc_obj.run(sentence)) > 0:
                    category_list.append(line.split()[-1])
                else:
                    category_list.append('24')
                    i += 1
                text_list.append(sentence)
    print "empty sentence num: ", i
    data_pd = pd.DataFrame(data={'text': text_list, 'category': category_list})
    data_pd.to_csv('all_preprocess.csv', header=None, index=False, sep='\t')
    return data_pd

# 数据探索
def data_explore(data_pd):
    t1 = data_pd.groupby('category').agg('count').reset_index()
    t1.rename(columns={'text': 'num'}, inplace=True)
    data_num = data_pd.shape[0]
    t1['ratio'] = t1['num']/data_num
    t1['times'] = t1['num'].max() / t1['num']
    print "各类别所占比例：\n", t1
    category_list = {}
    for index,element in t1.iterrows():
        if element['num'] < 3500:
            category_list[element['category']] = element['num']
    print "小于3500的类别：", category_list
    return category_list

if __name__ == "__main__":
    print "Begin to fomart data"
    data_pd = transform_data('data.txt')
    print "Begin to analyze data"
    category_list = data_explore(data_pd)