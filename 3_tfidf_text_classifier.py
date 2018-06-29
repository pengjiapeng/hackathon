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
            # while True:
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
        print self.tfidf.toarray()

    def split_train_test(self):
        self.train_set, self.test_set, self.train_tag, self.test_tag = train_test_split(self.tfidf, self.labels,
                                                                                        test_size=0.2)
        print "train set shape: ", self.train_set.shape
        train_set_file = open(self.data_path.replace("all.csv", "tfidf_train_set.pl"), 'wb')
        pickle.dump(self.train_set, train_set_file)
        train_tag_file = open(self.data_path.replace("all.csv", "tfidf_train_tag.pl"), 'wb')
        pickle.dump(self.train_tag, train_tag_file)
        test_set_file = open(self.data_path.replace("all.csv", "tfidf_test_set.pl"), 'wb')
        pickle.dump(self.test_set, test_set_file)
        test_tag_file = open(self.data_path.replace("all.csv", "tfidf_test_tag.pl"), 'wb')
        pickle.dump(self.test_tag, test_tag_file)

    def train_lr(self):
        print "Beigin to Train the model"
        lr_model = LogisticRegression()
        lr_model.fit(self.train_set, self.train_tag)
        print "End Now, and evalution the model with test dataset"
        y_pred = lr_model.predict(self.test_set)
        print classification_report(self.test_tag, y_pred)
        print confusion_matrix(self.test_tag, y_pred)
        print "save the trained model to tfidf_lr_model.pl"
        joblib.dump(lr_model, self.data_path.replace("all.csv", "tfidf_lr_model.pl"))

    def train_gbdt(self):
        print "Beigin to Train the model"
        lr_model = GradientBoostingClassifier(n_estimators=30, max_depth=12)
        lr_model.fit(self.train_set, self.train_tag)
        print "End Now, and evalution the model with test dataset"
        y_pred = lr_model.predict(self.test_set.toarray())
        print classification_report(self.test_tag, y_pred)
        print confusion_matrix(self.test_tag, y_pred)
        print "save the trained model to tfidf_gbdt_model.pl"
        joblib.dump(lr_model, self.data_path.replace("all.csv", "tfidf_gbdt_model.pl"))

    def train_rf(self):
        print "Beigin to Train the model"
        lr_model = RandomForestClassifier(n_estimators=50, n_jobs=-1)
        lr_model.fit(self.train_set, self.train_tag)
        print "End Now, and evalution the model with test dataset"
        y_pred = lr_model.predict(self.test_set.toarray())
        print classification_report(self.test_tag, y_pred)
        print confusion_matrix(self.test_tag, y_pred)
        print "save the trained model to tfidf_rf_model.pl"
        joblib.dump(lr_model, self.data_path.replace("all.csv", "tfidf_rf_model.pl"))

    def train_xgb(self):
        print "Beigin to Train the model"
        lr_model = XGBClassifier(learning_rate=0.1,n_estimators=1000,max_depth=5,min_child_weight=1,
                                 gamma=0,subsample=0.8,colsample_bytree=0.8,objective='multi:softmax',
                                 nthread=4,scale_pos_weight=1)
        lr_model.fit(self.train_set, self.train_tag)
        print "End Now, and evalution the model with test dataset"
        y_pred = lr_model.predict(self.test_set.toarray())
        print y_pred
        print classification_report(self.test_tag, y_pred)
        print confusion_matrix(self.test_tag, y_pred)
        print "save the trained model to tfidf_xgb_model.pl"
        joblib.dump(lr_model, self.data_path.replace("all.csv", "tfidf_xgb_model.pl"))

    def train_model(self):
        print "Beigin to read models"
        lr_model = joblib.load("tfidf_lr_model.pl")
        rf_model = joblib.load("tfidf_rf_model.pl")
        gbdt_model = joblib.load("tfidf_gbdt_model.pl")
        print "End Now, and evalution the model with test dataset"
        y_pred_lr = lr_model.predict(self.test_set.toarray())
        y_pred_rf = rf_model.predict(self.test_set.toarray())
        y_pred_gbdt = gbdt_model.predict(self.test_set.toarray())
        y_pred = []
        for i in range(len(y_pred_lr)):
            y = [0] * 25
            y[int(y_pred_lr[i])] = 0.3
            y[int(y_pred_rf[i])] = 0.4
            y[int(y_pred_gbdt[i])] = 0.3
            y_pred.append(str(y.index(max(y))))
        print classification_report(self.test_tag, y_pred)
        print confusion_matrix(self.test_tag, y_pred)
        print "save the trained model to tfidf_model.pl"
        joblib.dump(lr_model, self.data_path.replace("all.csv", "tfidf_model.pl"))



if __name__ == "__main__":
    # 加载数据
    bow_text_classifier_obj = tfidf_text_classifier("all.csv")
    # 生成句子特征向量
    bow_text_classifier_obj.vec()
    # 划分训练集和测试集
    bow_text_classifier_obj.split_train_test()
    # 模型训练
    bow_text_classifier_obj.train_rf()
    # bow_text_classifier_obj.train_lr()
    # bow_text_classifier_obj.train_gbdt()
    # 分别训练lr,gbdt,rf 3个模型之后再运行
    # bow_text_classifier_obj.train_model()