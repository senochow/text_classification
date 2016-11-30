############################################################################
#
# Copyright (c) 2016 ICT MCG Group, Inc. All Rights Reserved
#
###########################################################################
"""
Brief:

Authors: zhouxing(@ict.ac.cn)
Date:    2016/11/30 09:23:42
File:    utils.py
"""
import numpy as np
import cPickle as pk

class VocabProcessor:
    '''Maps documents to sequences of word ids
    '''
    def __init__(self, text_list, c=' '):
        self.text_list = text_list
        self.c = c
        self.max_doc_len = max([len(x.split(c)) for x in text_list])
        self.idx = 0
        self.vocab = {}

    def fit(self):
        ''' Learn a Maps documents to sequences of word ids

        '''
        for text in self.text_list:
            tokens = text.split(self.c)
            for token in tokens:
                if token not in self.vocab:
                    self.vocab[token] = self.idx
                    self.idx += 1

    def fit_transform(self):
        '''Learn the vocabulary dictionary and return indexies of words.
        '''
        self.fit()
        sent_word_ids = []
        for text in self.text_list:
            tokens = text.split(self.c)
            word_ids = np.zeros(self.max_doc_len, np.int64)
            for idx, token in enumerate(tokens):
                word_ids[idx] = self.vocab[token]
            sent_word_ids.append(word_ids)
        return sent_word_ids
    def get_vocab_size(self):
        return len(self.vocab)

    def save(self, filename):
        '''Saves vocabulary processor into given file.
        '''
        pk.dump(self.vocab, open(filename, 'wb'))

# vim: set expandtab ts=4 sw=4 sts=4 tw=100:
