# coding:utf8
############################################################################
#
# Copyright (c) 2016 ICT MCG Group, Inc. All Rights Reserved
#
###########################################################################
"""
Brief:

Authors: zhouxing(@ict.ac.cn)
Date:    2016/07/13 15:18:40
File:    evaluation.py
"""
import sys

def convert_prob_to_label(preds):
	labels = []
	for pred in preds:
		if pred > 0.5:
			labels.append(1)
		else:
			labels.append(0)
	return labels
def cal_precision_recall(hit, pred, actuall):
	precision = 1.0*hit/pred
	recall = 1.0*hit/actuall
	return precision, recall

def cal_f1_score(preds, labels):
        '''计算F1值
        Args:
            preds: list of predict value
            labels: actual label

        '''
	preds = convert_prob_to_label(preds)
	pos_hit, neg_hit = 0, 0
	pos_pred_cnt, neg_pred_cnt = 0, 0
	pos_cnt, neg_cnt = 0, 0
	inst_cnt = len(preds)
	for i in range(inst_cnt):
		if preds[i] == 1:
			pos_pred_cnt += 1
		else:
			neg_pred_cnt += 1
		if labels[i] == 1:
			pos_cnt += 1
		else:
			neg_cnt += 1
		if preds[i] == labels[i]:
			if preds[i] == 1:
				pos_hit += 1
			else:
				neg_hit += 1
	pos_precision, pos_recall = cal_precision_recall(pos_hit, pos_pred_cnt, pos_cnt)
	neg_precision, neg_recall = cal_precision_recall(neg_hit, neg_pred_cnt, neg_cnt)
	pos_f1 = 2.0*pos_precision*pos_recall/(pos_precision+pos_recall)
	neg_f1 = 2.0*neg_precision*neg_recall/(neg_precision+neg_recall)

	print "instance cnt. pos: %d \t neg: %d"%(pos_cnt, neg_cnt)
	print "hit cnt. pos: %d \t neg: %d" % (pos_hit, neg_hit)
	print "Pos: Precision : %f \t Recall : %f" % (pos_precision, pos_recall)
	print "Neg: Precision : %f \t Recall : %f" % (neg_precision, neg_recall)
	print "F1: pos: %f \t neg: %f" % (pos_f1, neg_f1)

def cal_f1_with_all_preds(pred_all, label):
    ''' 计算F1 score，转化为1的概率，调用cal_f1_score
    Args:
        pred_all: 每个元素包括2维度，0是分为0的概率，1是分为1的概率

    '''
    preds = pred_all[:, 1]
    cal_f1_score(preds, label)
if __name__ == '__main__':
    main()
# vim: set expandtab ts=4 sw=4 sts=4 tw=100:
