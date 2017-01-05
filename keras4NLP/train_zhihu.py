# coding:utf8
############################################################################
#
# Copyright (c) 2016 ICT MCG Group, Inc. All Rights Reserved
#
###########################################################################
"""
Brief:

Authors: zhouxing(@ict.ac.cn)
Date:    2016/12/21 16:57:41
File:    train_zhihu.py
"""
import sys
import numpy as np
from text_model import CNNWithKeywordLayer, CNNModel4Text, LSTMLayer, QuestionWithAnswersModel
import keras.backend.tensorflow_backend as K
from keras.models import Sequential, Model
from data_preprocess import load_zhihu_data
from keras.preprocessing.sequence import pad_sequences
import cPickle as pk
def make_idx_data_cv(data, fold, max_words_cnt, max_keywords_cnt):
    '''处理关键词的data

    '''
    x_train1, x_train2, x_test1, x_test2 = [], [], [], []
    y_train, y_test = [], []
    for d in data:
        sent = d['text_idx']
        keywords = d['keywords_idx']
        label = d['y']
        if d['split'] == fold:
            x_test1.append(sent)
            x_test2.append(keywords)
            y_test.append(label)
        else:
            x_train1.append(sent)
            x_train2.append(keywords)
            y_train.append(label)

    x_train1, x_train2 = np.array(x_train1), np.array(x_train2)
    x_test1, x_test2 = np.array(x_test1), np.array(x_test2)

    x_train1 = pad_sequences(x_train1, maxlen=max_words_cnt)
    x_train2 = pad_sequences(x_train2, maxlen=max_keywords_cnt)
    x_test1 = pad_sequences(x_test1, maxlen=max_words_cnt)
    x_test2 = pad_sequences(x_test2, maxlen=max_keywords_cnt)

    return [x_train1, x_train2], y_train, [x_test1, x_test2], y_test
def train_cross_validation(k, l, r, data, max_words_cnt, max_keywords_cnt, embedding_weights, embed_input, embed_dim):
    ''' training model by cross validation

    Args:
        k: k-fold
        data: original data set, contains pos and neg
        max_words_cnt: max words count in question
        max_keywords: max keywords count in answer
        embedding_weights:
    '''
    batch_size = 64
    num_epochs = 2
    num_filters = 100
    filter_sizes = (4, 5, 6)
    dropout_prob = (0.7, 0.5)
    hidden_dims = 50
    model_variation = 'CNN-non-static'
    total_acc = []

    for fold in range(l, r):
        x_train, y_train, x_test, y_test = make_idx_data_cv(data, fold, max_words_cnt, max_keywords_cnt)

        #model = CNNModel4Text(embedding_weights, embed_input, max_words_cnt, filter_sizes, num_filters,dropout_prob, hidden_dims, model_variation,embed_dim)
        #model = CNNModel4Text(embedding_weights, embed_input, max_keywords_cnt, filter_sizes, num_filters,dropout_prob, hidden_dims, model_variation,embed_dim)
        model = CNNWithKeywordLayer(embedding_weights, embed_input, max_words_cnt,max_keywords_cnt, filter_sizes, num_filters, dropout_prob, hidden_dims, model_variation, embed_dim)
        model.fit(x_train, y_train, batch_size=batch_size,nb_epoch=num_epochs, validation_data=(x_test, y_test), verbose=1)

        #model.fit(x_train, y_train, batch_size=batch_size,nb_epoch=num_epochs, validation_split=0.1, verbose=1)
        #model = CNNModel4Text(embedding_weights, embed_input, max_words_cnt, filter_sizes, num_filters,dropout_prob, hidden_dims, model_variation,embed_dim)
        #model.fit(x_train[0], y_train, batch_size=batch_size,nb_epoch=num_epochs, validation_data=(x_test[0], y_test), verbose=1)

        #model = LSTMLayer(embedding_weights, embed_input, max_keywords_cnt, filter_sizes, num_filters,dropout_prob, hidden_dims, model_variation,embed_dim)
        #model.fit(x_train[1], y_train, batch_size=batch_size,nb_epoch=num_epochs, validation_data=(x_test[1], y_test), verbose=1)
        acc = model.evaluate(x_test, y_test)
        print acc
        total_acc.append(acc[1])
    print "averge accuracy :", np.mean(total_acc)

def main():
    pk_file = sys.argv[1]
    l = int(sys.argv[2])
    r = int(sys.argv[3])
    data, max_words_cnt, max_keywords_cnt, fold_num, embedding_matrix, embed_input, embed_dim = pk.load(open(pk_file, 'rb'))
    print 'Load {} file done! ..'.format(pk_file)
    train_cross_validation(fold_num, l, r, data,max_words_cnt, max_keywords_cnt, embedding_matrix, embed_input, embed_dim)

if __name__ == '__main__':
    main()
# vim: set expandtab ts=4 sw=4 sts=4 tw=100:
