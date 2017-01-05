# coding:utf8
############################################################################
#
# Copyright (c) 2016 ICT MCG Group, Inc. All Rights Reserved
#
###########################################################################
"""
Brief:

Authors: zhouxing(@ict.ac.cn)
Date:    2016/12/20 19:55:08
File:    text_model.py
"""
import sys
from keras.models import Sequential, Model
from keras.layers import Activation, Dense, Dropout, Embedding, Flatten, Input, Merge, Convolution1D, AveragePooling1D, MaxPooling1D,GlobalAveragePooling1D,GlobalMaxPooling1D
from keras.layers.wrappers import TimeDistributed
from keras.layers.advanced_activations import LeakyReLU
from keras.regularizers import l2
from keras.constraints import maxnorm
from keras.optimizers import *
from keras.layers.wrappers import TimeDistributed
from keras.layers import LSTM, GRU, Bidirectional
from keras.layers.normalization import BatchNormalization
from AttentionLayer import AttentionLayer
from hierarchical_layers import HierarchicalRNN

def TextCNN(sequence_length, embedding_dim, filter_sizes, num_filters):
    ''' Convolutional Neural Network, including conv + pooling

    Args:
        sequence_length: 输入的文本长度
        embedding_dim: 词向量维度
        filter_sizes:  filter的高度
        num_filters: filter个数

    Returns:
        features extracted by CNN
    '''
    graph_in = Input(shape=(sequence_length, embedding_dim))
    convs = []
    for fsz in filter_sizes:
        conv = Convolution1D(nb_filter=num_filters,
                         filter_length=fsz,
                         border_mode='valid',
                         activation='relu',
                         subsample_length=1)(graph_in)
        pool = MaxPooling1D()(conv)
        flatten = Flatten()(pool)
        convs.append(flatten)
    if len(filter_sizes)>1:
        out = Merge(mode='concat')(convs)
    else:
        out = convs[0]
    graph = Model(input=graph_in, output=out)
    return graph

def LSTMLayer(embed_matrix, embed_input, sequence_length, dropout_prob, hidden_dims, embedding_dim=300, lstm_dim=100):
    model = Sequential()
    model.add(Embedding(embed_input, embedding_dim, input_length=sequence_length, weights=[embed_matrix]))
    model.add(Bidirectional(GRU(lstm_dim, return_sequences=True)))
    #model.add(AttentionLayer(2*lstm_dim))
    model.add(GlobalMaxPooling1D())
    return model


def KeywordLayer(sequence_length, embed_input, embedding_dim, embed_matrix):
    model = Sequential()
    model.add(Embedding(embed_input, embedding_dim, input_length=sequence_length, weights=[embed_matrix]))
    model.add(GlobalMaxPooling1D())
    return model

def CNNModel4Text(embed_matrix, embed_input, sequence_length, filter_sizes, num_filters, dropout_prob, hidden_dims, model_variation, embedding_dim=300):
    '''CNN model for text classification

    Args:
        embed_matrix: word embedding matrix
        embed_input: embedding矩阵行数
    '''
    #graph = TextCNN(sequence_length, embedding_dim, filter_sizes, num_filters)
    graph = KeywordLayer(sequence_length, embed_input, embedding_dim, embed_matrix)
    # main sequential model
    model = Sequential()
    # 1. embedding layer
    #'''
    if not model_variation=='CNN-static':
        model.add(Embedding(embed_input, embedding_dim, input_length=sequence_length, weights=[embed_matrix]))
    #model.add(Dropout(dropout_prob[0], input_shape=(sequence_length, embedding_dim)))
    #'''
    # 2. CNN layer
    model.add(graph)
    # 3. Hidden Layer
    model.add(Dense(hidden_dims))
    model.add(Dropout(dropout_prob[1]))
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='RMSprop', metrics=['accuracy'])
    return model

def CNNWithKeywordLayer(embed_matrix, embed_input, sequence_length, keywords_length, filter_sizes, num_filters, dropout_prob, hidden_dims, model_variation, embedding_dim=300):
    ''' 2-way input model: left is cnn for sentence embedding while right is keywords

    '''
    embed1 = Embedding(embed_input, embedding_dim,input_length=sequence_length, weights=[embed_matrix])
    # 1. question model part
    question_branch = Sequential()
    cnn_model = TextCNN(sequence_length, embedding_dim, filter_sizes, num_filters)
    question_branch.add(embed1)
    question_branch.add(cnn_model)
    # 2. keyword model part
    #keyword_branch = KeywordLayer(keywords_length, embed_input, embedding_dim, embed_matrix)
    keyword_branch = LSTMLayer(embed_matrix, embed_input, keywords_length, dropout_prob, hidden_dims, embedding_dim)
    # 3. merge layer
    merged = Merge([question_branch, keyword_branch], mode='concat')
    final_model = Sequential()
    final_model.add(merged)
    final_model.add(Dense(hidden_dims, W_constraint = maxnorm(3)))
    final_model.add(Dropout(0.5))
    final_model.add(Activation('relu'))
    final_model.add(Dense(1))
    final_model.add(Activation('sigmoid'))
    #sgd = SGD(lr=0.01, momentum=0.9)
    final_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return final_model

def QuestionWithAnswersModel(embed_matrix, embed_input, sequence_length, ans_cnt, keywords_length, filter_sizes, num_filters, dropout_prob, hidden_dims, embedding_dim=300):
    ''' path1: question embedding (CNN model)
        path2: answer embeddin(Hierachical RNN model)
        merge
    '''
    # path 1
    embed1 = Embedding(embed_input, embedding_dim,input_length=sequence_length, weights=[embed_matrix])
    question_branch = Sequential()
    cnn_model = TextCNN(sequence_length, embedding_dim, filter_sizes, num_filters)
    question_branch.add(embed1)
    question_branch.add(cnn_model)
    # path 2
    answer_branch = HierarchicalRNN(embed_matrix, embed_input, ans_cnt, keywords_length, embedding_dim)
    merged = Merge([question_branch, answer_branch], mode='concat')
    final_model = Sequential()
    final_model.add(merged)
    final_model.add(Dense(hidden_dims, W_constraint = maxnorm(3)))
    final_model.add(Dropout(0.5))
    final_model.add(Activation('relu'))
    final_model.add(Dense(1))
    final_model.add(Activation('sigmoid'))
    final_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return final_model
# vim: set expandtab ts=4 sw=4 sts=4 tw=100:
