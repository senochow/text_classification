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
File:    cnn_model.py
"""
import sys
from keras.models import Sequential, Model
from keras.layers import Activation, Dense, Dropout, Embedding, Flatten, Input, Merge, Convolution1D, MaxPooling1D
def CNNModel4Text(embed_matrix, embed_input, sequence_length, filter_sizes, num_filters, dropout_prob, hidden_dims, model_variation, embedding_dim=300):
    '''CNN model for text classification

    Args:
        embed_matrix: word embedding matrix
        embed_input: 矩阵行数
        sequence_length: 文本长度
        embedding_dim: 维度
    '''

    graph_in = Input(shape=(sequence_length, embedding_dim))
    convs = []
    for fsz in filter_sizes:
        conv = Convolution1D(nb_filter=num_filters,
                         filter_length=fsz,
                         border_mode='valid',
                         activation='relu',
                         subsample_length=1)(graph_in)
        pool = MaxPooling1D(pool_length=2)(conv)
        flatten = Flatten()(pool)
        convs.append(flatten)

    if len(filter_sizes)>1:
        out = Merge(mode='concat')(convs)
    else:
        out = convs[0]

    graph = Model(input=graph_in, output=out)

    # main sequential model
    model = Sequential()
    if not model_variation=='CNN-static':
        model.add(Embedding(embed_input, embedding_dim, input_length=sequence_length))
                        #weights=embedding_weights))
    model.add(Dropout(dropout_prob[0], input_shape=(sequence_length, embedding_dim)))
    model.add(graph)
    model.add(Dense(hidden_dims))
    model.add(Dropout(dropout_prob[1]))
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='RMSprop', metrics=['accuracy'])
    return model


# vim: set expandtab ts=4 sw=4 sts=4 tw=100:
