# -*- coding:utf8 -*-
############################################################################
#
# Copyright (c) 2016 ICT MCG Group, Inc. All Rights Reserved
#
###########################################################################
"""
Brief:

Authors: zhouxing(@ict.ac.cn)
Date:    2016/08/27 23:06:17
File:    data_preprocess.py
"""
import sys
import json
import cPickle as pk
import numpy as np
from sklearn.cross_validation import KFold
from collections import defaultdict
import pandas as pd

def gen_kfold(nums, cv):
    kf = KFold(nums, n_folds=cv)
    index = 0
    fold_map = {}
    for train, test in kf:
        for i in test:
            fold_map[i] = index
        index += 1
    return fold_map

def loadWordVectorsFromTxt(filename, word_index):
    word_vectors = {}
    f = open(filename)
    line = f.readline().rstrip()
    vocab_size, layer1_size = line.split()
    while 1:
        line = f.readline()
        if not line:
            break
        line = line.rstrip()
        word, fea = line.split(' ', 1)
        if word in word_index:
            word_vectors[word] = np.fromstring(fea, dtype='float32', sep=' ')
    return word_vectors, int(layer1_size)


def genEmbeddingMatrix(word_vectors, word_index, embed_dim):
    """
        Gen embedding matrix, use embedding from word_vectors if exists or random
    """
    nb_words = len(word_index)
    print 'word vectors size ', len(word_vectors)
    embedding_matrix = np.zeros((nb_words + 1, embed_dim))
    exists = 0
    for word, index in word_index.items():
        if word in word_vectors:
            exists += 1
            embedding_matrix[index] = word_vectors[word]
        else:
            embedding_matrix[index] = np.random.uniform(-0.25,0.25,embed_dim)
    print 'Total words %d, Already exists %d'%(nb_words, exists)
    return embedding_matrix

def update_vocab(vocab, words):
    ''' use words to update vocab

    '''
    for word in words:
        vocab.setdefault(word, 0)
        vocab[word] += 1

def load_data_from_file(data_file, vocab):
    ''' load json file
        id:
        isaanswer: label
        qSeg: 问题文本
        answers: 答案的关键词
    Args:

    '''
    all_data = []
    pos_num = 0
    neg_num = 0
    with open(data_file) as f:
        for line in f:
            d = json.loads(line.rstrip())
            mid = d['id']
            question = d['qSeg']
            q_list = [word.encode('utf8') for word in question.split()]
            question = ' '.join(q_list)
            label = int(d['isaanswer'])
            answers = d['answers']
            a_list = [word.encode('utf8') for word in answers]
            data = {'id':mid, 'text': question, 'y':label, 'keywords':a_list}
            all_data.append(data)
            if label == 0:
                neg_num += 1
            elif label == 1:
                pos_num += 1
            else:
                print 'Error label, label should be 0, 1'
                return
            update_vocab(vocab, question.split())
            update_vocab(vocab, a_list)
    print 'Total doc number: ', len(all_data)
    print 'Pos doc number: ', pos_num
    print 'Neg doc number: ', neg_num
    return all_data, pos_num, neg_num

def get_word_index_from_vocab(vocab, min_cnt = 1):
    ''' 得到word-index的映射表,过滤小于min_cnt的word
        index start from 1
    '''
    word_index = {}
    index = 1
    for word, cnt in vocab.items():
        if cnt < min_cnt:
            continue
        word_index[word] = index
        index += 1
    return word_index

def transform_text_data_to_idx(data, fold_map_pos, fold_map_neg, word_index):
    pos = 0
    neg = 0
    for d in data:
        if d['y'] == 0:
            d['split'] = fold_map_neg[neg]
            neg += 1
        else:
            d['split'] = fold_map_pos[pos]
            pos += 1
        text_words = d['text'].split()
        text_idx = []
        for word in text_words:
            text_idx.append(word_index[word])
        d['text_idx'] = text_idx
        d['text_words_cnt'] = len(text_idx)
        keywords_idx = []
        for word in d['keywords']:
            keywords_idx.append(word_index[word])
        d['keywords_idx'] = keywords_idx
        d['keywords_cnt'] = len(keywords_idx)
    return data

def load_zhihu_data(data_file, fold_num):
    '''按照fold数量拆分数据集

    '''
    vocab = {}
    data, pos_num, neg_num = load_data_from_file(data_file, vocab)
    fold_map_pos = gen_kfold(pos_num, fold_num)
    fold_map_neg = gen_kfold(neg_num, fold_num)
    word_index = get_word_index_from_vocab(vocab)
    new_data = transform_text_data_to_idx(data, fold_map_pos, fold_map_neg, word_index)
    return new_data, word_index, vocab

def main():
    data_file = sys.argv[1]
    word_vector_file = sys.argv[2]
    res_file = sys.argv[3]
    fold_num = 5
    data, word_index, vocab = load_zhihu_data(data_file, fold_num)
    word_vectors, embed_dim = loadWordVectorsFromTxt(word_vector_file, word_index)
    embedding_matrix = genEmbeddingMatrix(word_vectors, word_index, embed_dim)
    embed_input = len(word_index) + 1
    max_words_cnt = np.max(pd.DataFrame(data)['text_words_cnt'])
    max_keywords_cnt = np.max(pd.DataFrame(data)['keywords_cnt'])
    print 'max words cnt:{} max keywords cnt:{}'.format(max_words_cnt, max_keywords_cnt)

    pk.dump([data, max_words_cnt, max_keywords_cnt, fold_num, embedding_matrix, embed_input, embed_dim], open(res_file, 'wb'))


if __name__ == '__main__':
    main()

# vim: set expandtab ts=4 sw=4 sts=4 tw=100:
