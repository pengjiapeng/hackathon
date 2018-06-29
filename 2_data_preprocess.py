#! /usr/bin/env python
# -*- coding: UTF-8 -*-
import re
import os
import pandas as pd
import numpy as np
from cut_doc import cutDoc
import sys
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

reload(sys)
sys.setdefaultencoding('utf-8')

def data_transform():
    category_len = [0] * 25
    cut_doc = cutDoc()
    for i in range(25):
        if os.path.exists(str(i) + "_raw.txt"):
            os.remove(str(i) + "_raw.txt")
        if os.path.exists(str(i) + ".txt"):
            os.remove(str(i) + ".txt")
        if os.path.exists(str(i) + "_wrong.txt"):
            os.remove(str(i) + "_wrong.txt")
        if os.path.exists(str(i) + "_right.txt"):
            os.remove(str(i) + "_right.txt")
    with open("all_preprocess.csv", "r") as fread:
        i = 0
        for line in fread.readlines():
            # 按类别将样本写入不同文件中追加模式
            line_list = line.strip().split("\t")
            category_write = open(line_list[0] + '_raw.txt', 'a')
            category_write.write(line_list[1]+"\n")
            category_write.close()
            category_len[int(line_list[0])] += 1
            i += 1
            category_write = open(line_list[0] + '.txt', 'a')
            text_tokens = cut_doc.run(line_list[1])
            category_write.write(" ".join(text_tokens) + "\n")
            category_write.close()

def keyWords_extract():
    corpus = []
    for i in range(24):
        if os.path.exists(str(i) + ".txt"):
            with open(str(i)+".txt", "r") as fread:
                category = ''
                for line in fread.readlines():
                    category += line
                corpus.append(category)
    vectorizer = CountVectorizer()
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))
    word = vectorizer.get_feature_names()
    weight = tfidf.toarray()
    for k in range(len(weight)):
        word_list = []
        sort_list = weight[k].tolist()
        sort_list.sort()
        for i in range(20):
            index = weight[k].tolist().index(sort_list[-1-i])
            word_list.append(word[index])
        if k >= 3:
            k += 1
        print "第"+str(k)+"个类别关键词是："+str(word_list).replace('u\'','\'').decode("unicode-escape")

def process_data(type_num, key_word, in_file, out_file_right, out_file_wrong):
    right_list = []
    wrong_list = []
    right_num = 0
    total_num = 0
    wrong_num = 0
    f = open(in_file)
    for line in f:
        label = 0
        total_num += 1
        for i in key_word:
            match = re.search(i, line)
            if match:
                right_num += 1
                label = 1
                right_list.append(line.strip())
                break
        if label != 1:
            wrong_num += 1
            wrong_list.append(line.strip())
    f.close()
    # 保存正确样本
    with open(out_file_right, "w") as f_right:
        for i in right_list:
            f_right.write(i +'\n')
    f_right.close()
    # 保存错误样本
    with open(out_file_wrong, "w") as f_wrong:
        for i in wrong_list:
            f_wrong.write(i +'\n')
    f_wrong.close()
    print "第" + str(type_num) + "类总个数：" + str(total_num)+"  正确个数：" + str(right_num)+"   错误个数：" + str(wrong_num)+"   错误率：" + str(round(1.0*wrong_num/total_num*100)) + "%"
    return total_num, right_num, wrong_num

def up_sample():
    # 0类别过采样0.5倍，15/6过采样到300，9/10过采样2倍
    sample_list = ['0', '10', '15', '6', '9']
    for i in sample_list:
        data_pd = pd.read_table(i+"_right.txt", header=None)
        sample_num, fact_sample_num = 0, 0
        if i in ['0']:
            sample_num = int(data_pd.shape[0]*0.5)
        elif i in ['15', '6']:
            sample_num = 300 - data_pd.shape[0]
        elif i in ['9', '10']:
            sample_num = data_pd.shape[0]*2
        if fact_sample_num < sample_num:
            data_pd_sample = data_pd.sample(sample_num, replace=True)
            fact_sample_num += data_pd_sample.shape[0]
            data_pd = pd.concat([data_pd, data_pd_sample], ignore_index=True)
        data_pd.to_csv(str(i)+"_right.txt", header=None, index=False)

def data_combine():
    fwrite = open("all.csv", 'w')
    for i in range(0, 24):
        if os.path.exists(str(i) + "_right.txt"):
            with open(str(i) + "_right.txt", "r") as fread:
                for line in fread.readlines():
                    if str(i) == '2':
                        fwrite.write('22' + "\t" + line)
                    else:
                        fwrite.write(str(i) + "\t" + line)
    if os.path.exists("24_raw.txt"):
        with open("24_raw.txt", "r") as fread:
            for line in fread.readlines():
                fwrite.write(str(24) + "\t" + line)
    fwrite.close()

if __name__ == '__main__':
    # 将all_preprocess.csv转化为每一类的切词前后txt文件
    data_transform()
    # 关键字提取
    keyWords_extract()
    # 正则关键词匹配
    key_words = {}
    # 该部分之下代码删减
    key_words[0] = ["保费", "退保", "保費", "保单"]
    key_words[23] = ["绑定", "注册", "注销", "注冊", "手机号", "司机", "身份.*核实"]
    key_words[2] = key_words[22]
    # 该部分之上代码删减
    all_total_num, all_right_num, all_wrong_num = 0, 0, 0
    for i in range(0, 24):
        if i == 3:
            continue
        total_num, right_num, wrong_num = process_data(i, key_words[i], str(i) + "_raw.txt", str(i) + "_right.txt", str(i) + "_wrong.txt")
        all_total_num += total_num
        all_right_num += right_num
        all_wrong_num += wrong_num
    print "总共样本数："+str(all_total_num)+" 正确样本："+str(all_right_num)+" 错误样本："+str(all_wrong_num)+" 准确率："+str(round(1.0*all_right_num/all_total_num*100))+ "%"
    # 过采样
    up_sample()
    # 文本合并
    data_combine()