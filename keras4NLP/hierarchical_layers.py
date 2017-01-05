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
from keras.layers import AveragePooling1D, MaxPooling1D,GlobalAveragePooling1D,GlobalMaxPooling1D

def HierarchicalRNN(embed_matrix, max_words, ans_cnt, sequence_length, embedding_dim, lstm_dim=100):
    ''' Hierachical RNN model
        Input: (batch_size, answers, answer words)
    Args:
        embed_matrix: word embedding
        max words:    word dict size of embedding layer
        ans_cnt:      answer count
        sequence_length: answer words count
        embedding_dim: embedding dimention
        lstm_dim:
    '''
    hnn = Sequential()
    x = Input(shape=(ans_cnt, sequence_length))
    # 1. time distributed word embedding: (None, steps, words, embed_dim)
    words_embed = TimeDistributed(Embedding(max_words, embedding_dim,input_length=sequence_length,weights=[embed_matrix]))(x)
    # 2. word level lstm embedding: --> (None, steps/sentence_num, hidden/sent_words, hidden_dim)
    word_lstm = TimeDistributed(Bidirectional(GRU(lstm_dim, return_sequences=True)))(words_embed)

    # 3. average pooling : --> (None,steps,dim)
    word_avg = TimeDistributed(GlobalMaxPooling1D())(word_lstm)

    # 4.  sentence lstm:  --> (None, hidden, hidden_dim)
    sent_lstm = Bidirectional(GRU(lstm_dim, return_sequences=True))(word_avg)

    # 5. pooling:  --> (None, hidden_dim)
    sent_avg = GlobalMaxPooling1D()(sent_lstm)
    model = Model(input=x, output=sent_avg)
    hnn.add(model)
    return hnn


# vim: set expandtab ts=4 sw=4 sts=4 tw=100:
