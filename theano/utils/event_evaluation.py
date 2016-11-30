############################################################################
#
# Copyright (c) 2016 ICT MCG Group, Inc. All Rights Reserved
#
###########################################################################
"""
Brief:

Authors: zhouxing(@ict.ac.cn)
Date:    2016/07/13 17:55:16
File:    event_evaluation.py
"""
import sys

def cal_event_prob(test_set_y, tmp_pred_prob, test_event_id):
    print 'cal event probability...'
    event_pred_list = {}
    event_label = {}
    for i in range(len(test_set_y)):
        index = test_event_id[i]
        event_pred_list.setdefault(index,[])
        event_pred_list[index].append(tmp_pred_prob[i][1])
        event_label[index] = test_set_y[i]
    event_pred = {}
    for index, preds in event_pred_list.items():
        pred_val = np.mean(preds)
        event_pred[index] = pred_val
    event_pred_label = {}
    for index, pred_val in event_pred.items():
        if pred_val > 0.5:
            event_pred_label[index] = 1
        else:
            event_pred_label[index] = 0
    print 'each event probability..'
    print event_pred
    avg_prec = cal_measure_info(event_label, event_pred_label)
    return avg_prec

def cal_event_mersure(test_set_y, tmp_pred_y, test_event_id):
    m_pred = {}
    event_label = {}
    for i in range(len(test_set_y)):
        index = test_event_id[i]
        m_pred.setdefault(index, [0,0])
        event_label[index] = test_set_y[i]
        m_pred[index][tmp_pred_y[i]] += 1
    # cal
    print 'each event pred...'
    print m_pred
    event_pred = {}
    for index, preds in m_pred.items():
        if preds[0] > preds[1]:
            event_pred[index] = 0
        else:
            event_pred[index] = 1
    avg_prec = cal_measure_info(event_label, event_pred)
    return avg_prec

def cal_measure_info(event_label, event_pred):
    p_hit ,n_hit = 0, 0
    p_pred_num, n_pred_num = 0, 0
    p_num , n_num = 0, 0
    avg_prec = 0
    for index, label in event_label.items():
        pred_label = event_pred[index]
        if label == pred_label:
            avg_prec += 1
        if label == 1:
            p_num += 1
            if label == pred_label:
                p_hit += 1
                p_pred_num += 1
            else:
                n_pred_num += 1
        else:
            n_num += 1
            if label == pred_label:
                n_hit += 1
                n_pred_num += 1
            else:
                p_pred_num += 1
    avg_prec = 1.0*avg_prec/len(event_label)
    print 'hit count pos:%d, neg:%d'%(p_hit, n_hit)
    print 'precision : pos: %f, neg: %f' % (1.0*p_hit/p_pred_num, 1.0*n_hit/n_pred_num)
    print 'recall : pos: %f, neg: %f' % (1.0*p_hit/p_num, 1.0*n_hit/n_num)
    print 'average accuracy : %f'% (avg_prec)
    sys.stdout.flush()
    return avg_prec


if __name__ == '__main__':
    main()
# vim: set expandtab ts=4 sw=4 sts=4 tw=100:
