# -*-coding:utf8-*-
############################################################################
#
# Copyright (c) 2016 ICT MCG Group, Inc. All Rights Reserved
#
###########################################################################
"""
Brief:

Authors: zhouxing(@ict.ac.cn)
Date:    2016/07/15 11:01:32
File:    utils/cv_data_helper.py
"""
import sys
import numpy as np
from learning import shared_dataset
class CVData:
    def __init__(self, train, test, test_event_id, test_mid, test_context):
        self.train = train
        self.test = test
        self.test_event_id = test_event_id
        self.test_mid = test_mid
        self.test_context = test_context

def get_idx_from_sent(sent, word_idx_map, max_l=51, k=300, filter_h=5):
    """
    Transforms sentence into a list of indices. Pad with zeroes, (lastword+pad=filter_h).
        sent = pad + sentence + pad , pad = filter_h-1
    """
    x = []
    pad = filter_h - 1
    for i in xrange(pad):
        x.append(0)
    words = sent.split()
    for word in words:
        if word in word_idx_map:
            x.append(word_idx_map[word])
    while len(x) < max_l+2*pad:
        x.append(0)
    return x

def make_idx_data_cv(revs, word_idx_map, cv, max_l=51, k=300, filter_h=5):
    """
    Transforms sentences into a 2-d matrix.
    """
    train, test = [], []
    test_mid = []
    test_event_id = []
    test_context = []
    for rev in revs:
        sent = get_idx_from_sent(rev["text"], word_idx_map, max_l, k, filter_h)
        # add extra feature to sent fea, word_index(max_l) + extra_fea + y
        extra_fea = [int(val) for val in rev["extra_fea"].split(",")]
        sent += extra_fea
        mid = rev["mid"]
        sent.append(rev["y"])
        if rev["split"]==cv:
            test.append(sent)
            test_event_id.append(rev["event_id"])
            test_mid.append(mid)
            test_context.append(rev["text"])
        else:
            train.append(sent)
    train = np.array(train,dtype="float32")
    test = np.array(test,dtype="float32")
    print 'traing set :\t', len(train)
    print 'testing set :\t', len(test)
    cv_data = CVData(train, test, test_event_id, test_mid, test_context)
    return cv_data
def gen_train_validation(train_data, batch_size, ratio=0.9, shuffled=True):
    ''' 将训练数据拆分成train & validation 两个部分，默认0.1做validation
        shuffle dataset, 如果不是batch的整数倍，则补全至完整的倍数
    '''
    np.random.seed(3435)
    if train_data.shape[0] % batch_size > 0:
        extra_data_num = batch_size - train_data.shape[0] % batch_size
        train_set = np.random.permutation(train_data)
        extra_data = train_set[:extra_data_num]
        new_data=np.append(train_data, extra_data, axis=0)
    else:
        new_data = train_data
    # 每次只取0.9倍的数据进行train， shuffle，另外的validation
    new_data = np.random.permutation(new_data)
    n_batches = new_data.shape[0]/batch_size
    n_train_batches = int(np.round(n_batches*ratio))
    n_val_batches = n_batches - n_train_batches
    # train & validation
    train_set = new_data[:n_train_batches*batch_size,:]
    val_set = new_data[n_train_batches*batch_size:,:]
    train_set_x, train_set_y = shared_dataset((train_set[:,:-1],train_set[:,-1]))
    val_set_x, val_set_y = shared_dataset((val_set[:,:-1],val_set[:,-1]))
    return train_set_x, train_set_y, val_set_x, val_set_y, n_train_batches, n_val_batches

def gen_test_batch_data(test_data, batch_size):
    ''' 得到测试的batch数据，多余的不补全

    '''
    test_size_all = test_data.shape[0]
    n_test_batches = test_size_all/batch_size
    test_size_batch = n_test_batches*batch_size
    test_size_remand = test_size_all%batch_size
    # 一部分用batch来预测，一部分不足的用test模型
    test_set_x = test_data[:test_size_batch, :-1]
    test_set_y = np.asarray(test_data[:test_size_batch, -1],"int32")
    test_set_remand_x = test_data[test_size_batch:, :-1]
    test_set_remand_y = np.asarray(test_data[test_size_batch:, -1],"int32")
    test_set_x, test_set_y = shared_dataset((test_set_x, test_set_y))
    return test_set_x, test_set_y, test_set_remand_x, test_set_remand_y, test_size_remand,n_test_batches, test_size_batch
# vim: set expandtab ts=4 sw=4 sts=4 tw=100:
