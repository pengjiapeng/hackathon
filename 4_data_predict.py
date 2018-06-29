#!/usr/bin/python
# -*- coding:utf-8 -*-
from cut_doc import cutDoc
from smote import Smote
import numpy as np
import pandas as pd
from gensim import corpora, models
from pprint import pprint
import traceback
import sys
import cPickle as pickle
from scipy.sparse import *
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.cross_validation import StratifiedKFold
from sklearn import preprocessing
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics
from sklearn.grid_search import GridSearchCV
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4
reload(sys)
sys.setdefaultencoding('utf-8')

class tfidf_text_classifier:
    """ tf_idf_text_classifier: a text classifier of tfidf
        dictionary : 词典
        corpus : 文本
        labels : 标签
    """

    def __init__(self, data_path):
        self.data_path = data_path
        self.dictionary = corpora.Dictionary()
        self.corpus = []
        self.labels = []
        self.cut_doc_obj = cutDoc()

    def __get_all_tokens(self):
        """ get all tokens of the corpus
        """
        fwrite = open(self.data_path.replace("all.csv", "all_token.csv"), 'w')
        with open(self.data_path, "r") as fread:
            i = 0
            for line in fread.readlines():
                try:
                    # 将切词之后的样本写入all_token.csv中
                    line_list = line.strip().split("\t")
                    label = line_list[0]
                    self.labels.append(label)
                    text = line_list[1]
                    text_tokens = self.cut_doc_obj.run(text)
                    self.corpus.append(' '.join(text_tokens))
                    self.dictionary.add_documents([text_tokens])
                    fwrite.write(label + "\t" + "\\".join(text_tokens) + "\n")
                    i += 1
                except BaseException as e:
                    msg = traceback.format_exc()
                    print msg
                    print "=====>Read Done<======"
                    break

        #add predict data
        with open("predict.txt", "r") as fread:
            for line in fread.readlines():
                try:
                    text_tokens = self.cut_doc_obj.run(line)
                    self.corpus.append(' '.join(text_tokens))
                    self.dictionary.add_documents([text_tokens])
                except BaseException as e:
                    msg = traceback.format_exc()
                    print msg
                    print "=====>Read Done<======"
                    break

        self.token_len = self.dictionary.__len__()
        print "all token len: " + str(self.token_len)
        self.num_data = i
        fwrite.close()

    def __filter_tokens(self, threshold_num=10):
        small_freq_ids = [tokenid for tokenid, docfreq in self.dictionary.dfs.items() if docfreq < threshold_num]
        self.dictionary.filter_tokens(small_freq_ids)
        self.dictionary.compactify()

    def vec(self):
        """ vec: get a vec representation of bow
        """
        self.__get_all_tokens()
        print "before filter, the tokens len: {0}".format(self.dictionary.__len__())
        vectorizer = CountVectorizer(min_df=1e-5)
        transformer = TfidfTransformer()
        # sparse matrix
        self.tfidf = transformer.fit_transform(vectorizer.fit_transform(self.corpus))
        self.tfidf_predict = self.tfidf.toarray()[self.num_data:]
        words = vectorizer.get_feature_names()
        print "word len: {0}".format(len(words))
        # print self.tfidf[0]
        print "tfidf shape: ({0},{1})".format(self.tfidf.shape[0], self.tfidf.shape[1])

        # write the tfidf vec into a file
        tfidf_vec_file = open(self.data_path.replace("all.csv", "tfidf_vec.pl"), 'wb')
        pickle.dump(self.tfidf, tfidf_vec_file)
        tfidf_vec_file.close()
        tfidf_label_file = open(self.data_path.replace("all.csv", "tfidf_label.pl"), 'wb')
        pickle.dump(self.labels, tfidf_label_file)
        tfidf_label_file.close()

    def split_train_test(self):
        self.train_set, self.test_set, self.train_tag = self.tfidf[:self.num_data], self.tfidf[self.num_data:],self.labels

    def train_rf(self):
        print "Beigin to Train the model"
        lr_model = RandomForestClassifier(n_estimators=50, n_jobs=-1)
        lr_model.fit(self.train_set, self.train_tag)
        print "End Now, and evalution the model with test dataset"
        y_pred = lr_model.predict(self.test_set.toarray())
        print y_pred
        fwrite = open("predict_label.txt", "w")
        i = 0
        with open("predict.txt", "r") as fread:
            for line in fread.readlines():
                fwrite.write(line.strip()+"\t"+y_pred[i]+"\n")
                i += 1

if __name__ == "__main__":
    # 加载数据
    bow_text_classifier_obj = tfidf_text_classifier("all.csv")
    # 生成句子特征向量
    bow_text_classifier_obj.vec()
    # 划分训练集和测试集
    bow_text_classifier_obj.split_train_test()
    # 模型训练
    bow_text_classifier_obj.train_rf()