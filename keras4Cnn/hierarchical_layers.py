############################################################################
#
# Copyright (c) 2016 ICT MCG Group, Inc. All Rights Reserved
#
###########################################################################
"""
Brief:

Authors: zhouxing(@ict.ac.cn)
Date:    2016/12/28 16:20:30
File:    hierarchical_layers.py
"""
from keras.models import Sequential, Model
from keras.layers import Activation, Dense, Dropout, Embedding, Flatten, Input, Merge, Convolution1D, AveragePooling1D, MaxPooling1D,GlobalAveragePooling1D
from keras.layers.wrappers import TimeDistributed
from keras.layers.advanced_activations import LeakyReLU
from keras.regularizers import l2
from keras.constraints import maxnorm
from keras.optimizers import *
from keras.layers.wrappers import TimeDistributed
from keras.layers import LSTM, GRU, Bidirectional
from keras.layers.normalization import BatchNormalization


def HierarchicalRNN(embed_matrix, embed_input, sequence_length, embedding_dim=300, lstm_dim=50):
    model = Sequential()
    sentence_embedding = Embedding(embed_input, embedding_dim, input_length=sequence_length,weights=[embed_matrix])
    # time distributed word embedding: (None, steps, words, embed_dim)
    model.add(TimeDistributed(sentence_embedding))
    # word level embedding: --> (None, steps/sentence_num, hidden/sent_words, hidden_dim)
    model.add(TimeDistributed(Bidirectional(GRU(lstm_dim, return_sequences=True))))
    # average pooling : --> (None,steps,dim)
    model.add(TimeDistributed(GlobalMaxPooling1D()))
    # sentence lstm:  --> (None, hidden, hidden_dim)
    model.add(Bidirectional(GRU(lstm_dim, return_sequences=True)))
    # pooling:  --> (None, hidden_dim)
    model.add(GlobalMaxPooling1D())
    return model


# vim: set expandtab ts=4 sw=4 sts=4 tw=100:
