# coding:utf8
############################################################################
#
# Copyright (c) 2016 ICT MCG Group, Inc. All Rights Reserved
#
###########################################################################
"""
Brief:
处理多答案
Authors: zhouxing(@ict.ac.cn)
Date:    2016/12/21 16:57:41
File:    train_zhihu_hnn.py
"""
import sys
import numpy as np
from text_model import CNNWithKeywordLayer, CNNModel4Text, LSTMLayer, QuestionWithAnswersModel
from hierarchical_layers import HierarchicalRNN
import keras.backend.tensorflow_backend as K
from keras.models import Sequential, Model
from data_preprocess import load_zhihu_data
from keras.preprocessing.sequence import pad_sequences
import cPickle as pk
def process_answers(answers, max_answer_cnt=20, max_words_cnt=300):
    '''处理一个问题下的answers
       控制answers的数量，同时控制每个answer的文本长度
       不足的补0向量

    '''
    zero_vec = [0]*max_words_cnt
    ans_cnt = len(answers)
    if ans_cnt > max_answer_cnt:
        answers = answers[:max_answer_cnt]
    else:
        for i in range(max_answer_cnt-ans_cnt):
            answers.append(zero_vec)
    answers = np.array(answers)
    answers = pad_sequences(answers, maxlen=max_words_cnt)
    return answers

def make_idx_data_cv(data, fold, max_qwords_cnt, max_awords_cnt, max_ans_cnt):
    '''

    '''
    x_train1, x_train2, x_test1, x_test2 = [], [], [], []
    y_train, y_test = [], []
    for d in data:
        sent = d['text_idx']
        answers = d['answers_idx']
        label = d['y']
        if d['split'] == fold:
            x_test1.append(sent)
            x_test2.append(process_answers(answers, max_ans_cnt, max_awords_cnt))
            y_test.append(label)
        else:
            x_train1.append(sent)
            x_train2.append(process_answers(answers, max_ans_cnt, max_awords_cnt))
            y_train.append(label)

    x_train1, x_train2 = np.array(x_train1), np.array(x_train2)
    x_test1, x_test2 = np.array(x_test1), np.array(x_test2)

    x_train1 = pad_sequences(x_train1, maxlen=max_qwords_cnt)
    x_test1 = pad_sequences(x_test1, maxlen=max_qwords_cnt)
    return [x_train1, x_train2], y_train, [x_test1, x_test2], y_test
def train_cross_validation(k, l, r, data, max_qwords_cnt, max_awords_cnt,max_ans_cnt, embedding_weights, embed_input, embed_dim):
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
    max_keywords_cnt = 300

    for fold in range(l, r):
        x_train, y_train, x_test, y_test = make_idx_data_cv(data, fold, max_qwords_cnt, max_awords_cnt, max_ans_cnt)
        model = QuestionWithAnswersModel(embedding_weights, embed_input, max_qwords_cnt, max_ans_cnt, max_awords_cnt, filter_sizes, num_filters, dropout_prob, hidden_dims, embed_dim)
        model.fit(x_train, y_train, batch_size=batch_size,nb_epoch=num_epochs, validation_data=(x_test, y_test), verbose=1)
        acc = model.evaluate(x_test, y_test)
        print acc
        total_acc.append(acc[1])
    print "averge accuracy :", np.mean(total_acc)

def main():
    pk_file = sys.argv[1]
    l = int(sys.argv[2])
    r = int(sys.argv[3])
    data, max_qwords_cnt, max_awords_cnt, max_ans_cnt, fold_num, embedding_matrix, embed_input, embed_dim = pk.load(open(pk_file, 'rb'))
    print 'Load {} file done! ..'.format(pk_file)
    train_cross_validation(fold_num, l, r, data,max_qwords_cnt, max_awords_cnt, max_ans_cnt,  embedding_matrix, embed_input, embed_dim)

if __name__ == '__main__':
    main()
# vim: set expandtab ts=4 sw=4 sts=4 tw=100:
