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
        # if the sententce if larger than max_l, then use the (0, max_l)
        #if len(sent) > max_l:
        #    sent = sent[:max_l]
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


# vim: set expandtab ts=4 sw=4 sts=4 tw=100:
