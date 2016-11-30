#! -*- coding:utf8 -*-
"""
Sample code for
Convolutional Neural Networks for Sentence Classification
http://arxiv.org/pdf/1408.5882v2.pdf

Much of the code is modified from
- deeplearning.net (for ConvNet classes)
- https://github.com/mdenil/dropout (for dropout)
- https://groups.google.com/forum/#!topic/pylearn-dev/3QbKtCumAW4 (for Adadelta)
"""
import cPickle as pk
import numpy as np
from collections import defaultdict, OrderedDict
import theano
import theano.tensor as T
import re
import warnings
import sys
import time
import ConfigParser
warnings.filterwarnings("ignore")
from utils.evaluation import cal_f1_with_all_preds
from utils.conv_net_classes import Iden
from utils.cv_data_helper import *
from utils.conv_net_classes import *

def load_param(model_file):
    f = open(model_file, "rb")
    clf_param = pk.load(f)
    conv1_param = pk.load(f)
    conv2_param = pk.load(f)
    conv3_param = pk.load(f)

    return clf_param, [conv1_param, conv2_param, conv3_param]


def conv_net_predict(datasets,
                   dataset_dir,
                   cv,
                   flag,
                   U,
                   img_w=300,
                   filter_hs=[3,4,5],
                   hidden_units=[100,2],
                   dropout_rate=[0.5],
                   shuffle_batch=True,
                   n_epochs=5,
                   batch_size=50,
                   lr_decay = 0.95,
                   conv_non_linear="relu",
                   activations=[Iden],
                   sqr_norm_lim=9,
                   extra_fea_len=6,
                   non_static=True):
    img_h = len(datasets[0])-extra_fea_len-1  # last one is y
    filter_w = img_w
    feature_maps = hidden_units[0]
    filter_shapes = []
    pool_sizes = []
    for filter_h in filter_hs:
        # filter: conv shape, hidden layer: 就是最后的全连接层
        filter_shapes.append((feature_maps, 1, filter_h, filter_w))
        pool_sizes.append((img_h-filter_h+1, img_w-filter_w+1))

    rng = np.random.RandomState(3435)
    # load param
    if flag:
        param_file = dataset_dir + "/cnn_param_extra_"+str(cv)+".pk"
    else:
        param_file = dataset_dir+ "/cnn_param_"+str(cv)+".pk"

    clf_param, conv_param = load_param(param_file)
    #define model architecture
    x = T.matrix('x')
    y = T.ivector('y')
    Words = theano.shared(value = U, name = "Words")
    # 转成 batch(50)*1*sent_len(134)*k(300)
    layer0_input = Words[T.cast(x[:,:img_h].flatten(),dtype="int32")].reshape((x.shape[0],1,img_h,Words.shape[1]))
    layer1_input_extra_fea = x[:,img_h:]
    conv_layers = []
    layer1_inputs = []
    # each filter has its own conv layer: the full conv layers = concatenate all layer to 1
    for i in xrange(len(filter_hs)):
        filter_shape = filter_shapes[i]
        pool_size = pool_sizes[i]
        conv_layer = LeNetConvPoolLayerLoadParam(rng, input=layer0_input,param_w=conv_param[i][0], param_b=conv_param[i][1], image_shape=(batch_size, 1, img_h, img_w),
                                filter_shape=filter_shape, poolsize=pool_size, non_linear=conv_non_linear)
        layer1_input = conv_layer.output.flatten(2)
        conv_layers.append(conv_layer)
        layer1_inputs.append(layer1_input)

    if flag:
        layer1_inputs.append(layer1_input_extra_fea)
    layer1_input = T.concatenate(layer1_inputs,1)
    if flag:
        hidden_units[0] = feature_maps*len(filter_hs) + extra_fea_len
    else :
        hidden_units[0] = feature_maps*len(filter_hs)
    #classifier = MLPDropoutLoadParam(rng, input=layer1_input, clf_param[0], clf_param[1], layer_sizes=hidden_units, activations=activations, dropout_rates=dropout_rate)
    classifier = LogisticRegression(input=layer1_input, n_in=hidden_units[0], n_out=hidden_units[1], W=clf_param[0], b=clf_param[1])

    return conv_layers, classifier, Words, img_h

def predict_with_cnn_model(test_dataset, conv_layers, classifier, Words, flag, img_h):
    ''' 使用预定义的cnn模型进行预测
    Args:
        test_dataset: test data set
        conv_layers: 初始化的卷积层
        classifier: 初始化的分类器
        Words: 词向量
        flag:  是否用额外的feature
        img_h: input的矩阵高度
    '''
    x = T.matrix('x')
    y = T.ivector('y')
    # test data for predict
    test_set_x = test_dataset[:,:-1]
    test_set_y = np.asarray(test_dataset[:,-1],"int32")
    test_pred_layers = []
    test_size = test_set_x.shape[0]
    test_layer0_input = Words[T.cast(x[:, :img_h].flatten(),dtype="int32")].reshape((test_size,1,img_h,Words.shape[1]))
    test_layer1_input_extra_fea = x[:,img_h:]

    # [(instances * feature_maps * conv_feature)], different filter size have different conv dimention
    test_conv_data = []
    for conv_layer in conv_layers:
        test_layer0_output = conv_layer.predict(test_layer0_input, test_size)
        test_layer0_convs = conv_layer.predict_maxpool(test_layer0_input, test_size)
        test_pred_layers.append(test_layer0_output.flatten(2))
        test_conv_data.append(test_layer0_convs.flatten(3))
    # conv data
    if flag:
        test_pred_layers.append(test_layer1_input_extra_fea)

    test_layer1_input = T.concatenate(test_pred_layers, 1)

    test_y_pred = classifier.predict(test_layer1_input)
    test_y_pred_p = classifier.predict_p(test_layer1_input)

    #test_error = T.mean(T.neq(test_y_pred, y))
    test_error_cnt = T.neq(test_y_pred, y)
    test_model_all = theano.function([x,y], test_error_cnt, allow_input_downcast=True)
    test_model_f1 = theano.function([x], test_y_pred, allow_input_downcast=True)
    test_model_prob = theano.function([x], test_y_pred_p, allow_input_downcast=True)
    test_layer1_feature = theano.function([x], test_layer1_input, allow_input_downcast=True)
    test_extra_fea = theano.function([x], test_layer1_input_extra_fea, allow_input_downcast=True)
    get_all_conv_data = theano.function([x], test_conv_data, allow_input_downcast=True)
    print '... predict'
    start_time = time.time()
    tmp_pred_prob = test_model_prob(test_set_x)
    test_error_cnt = test_model_all(test_set_x,test_set_y)
    tmp_all_test_conv_data = get_all_conv_data(test_set_x)
    return tmp_all_test_conv_data, tmp_pred_prob, test_error_cnt

def write_event_keywords(dataset_dir, test_set, flag, filter_hs, cv, all_test_conv_data, test_event_id, tmp_pred_prob, test_context, idx_word_map):
    print "test_context \t",len(test_context)
    print "prob_context \t",len(tmp_pred_prob)
    test_set_x = test_set[:, :-1]
    test_set_y = test_set[:, -1]
    f_keywords = open(dataset_dir+"/keywords/" + "keywords_" + "_"+str(cv)+".txt", "w")
    # tmp_test_conv_data: instance_cnt * feature_maps * conv_feature
    filter_num = len(filter_hs)
    for i in range(all_test_conv_data[0].shape[0]):
        f_keywords.write(str(int(test_set_y[i])) + '\t' +str(tmp_pred_prob[i][1])+ "\t" + test_context[i].encode("utf8", 'ignore')+"\n")
        #f_keywords.write(str(tmp_pred_prob[i][1])+ "\t" + test_context[i].encode("utf8", 'ignore')+"\n")
        # i : each instance
        for i_filter in range(filter_num):
            max_index_map = {}
            conv_features = all_test_conv_data[i_filter][i]
            max_val, max_index = -100, 0
            for fea_map_num in range(len(conv_features)):
                for k, val in enumerate(conv_features[fea_map_num]):
                    if val > max_val:
                        max_val = val
                        max_index = k
                max_index_map.setdefault(max_index, 0)
                max_index_map[max_index] += 1
            dic = sorted(max_index_map.iteritems(), key=lambda d:d[1], reverse = True)
            max_index, max_val = dic[0]
            keyphrase = []
            for num in range(filter_hs[i_filter]):
                keyphrase.append(max_index+num)
            f_keywords.write("%d\t%s"%(max_index, str(dic))+"\n")
            f_keywords.write(' '.join([idx_word_map[index].encode("utf8", 'ignore') for index in test_set_x[i][keyphrase]])+"\n")
    f_keywords.close()


if __name__=="__main__":
    print "loading data...",
    mode= sys.argv[1]
    word_vectors = sys.argv[2]
    cv = int(sys.argv[3])
    flag = int(sys.argv[4])
    data_set = sys.argv[5]
    conf_file = sys.argv[6]
    cf = ConfigParser.ConfigParser()
    cf.read(conf_file)
    x = pk.load(open(cf.get(data_set, "pkfile"+str(cv)),"rb"))
    dataset_dir = cf.get(data_set, "dataset_dir")
    revs, W, W2, word_idx_map, vocab, idx_word_map, max_length = x[0], x[1], x[2], x[3], x[4], x[5], x[6]

    print "data loaded!"

    if mode=="-nonstatic":
        print "model architecture: CNN-non-static"
        non_static=True
    elif mode=="-static":
        print "model architecture: CNN-static"
        non_static=False
    execfile("./utils/conv_net_classes.py")
    if word_vectors=="-rand":
        print "using: random vectors"
        U = W2
    elif word_vectors=="-word2vec":
        print "using: word2vec vectors"
        U = W
    results = []
    fres = []
    avg_plist = []
    r = range(0,cv)
    start_time = time.time()
    filter_hs =[3,4,5]
    for i in r:
        cv_data = make_idx_data_cv(revs, word_idx_map, i, max_l=max_length,k=300, filter_h=5)
        test_set_all = cv_data.test
        all_conv, all_prob, all_error_cnt = [[],[],[]], [], []
        conv_layers, classifier, Words, img_h = conv_net_predict(test_set_all,
                              dataset_dir,
                              i,
                              flag,
                              U,
                              lr_decay=0.95,
                              filter_hs=filter_hs,
                              conv_non_linear="relu",
                              hidden_units=[100,2],
                              shuffle_batch=True,
                              n_epochs=10,
                              sqr_norm_lim=9,
                              non_static=non_static,
                              batch_size=50,
                              dropout_rate=[0.5])

        test_batch_cnt = len(test_set_all)+1
        test_batch_size = len(test_set_all)/test_batch_cnt + 1
        # Use full test data as one batch
        for index in range(test_batch_size):
            cur_test_data = test_set_all[index*test_batch_cnt: (index+1)*test_batch_cnt]
            tmp_conv, tmp_prob, tmp_error_cnt = predict_with_cnn_model(cur_test_data, conv_layers, classifier, Words, flag, img_h)
            for j in range(len(tmp_conv)):
                all_conv[j].append(tmp_conv[j])
            all_prob.append(tmp_prob)
            all_error_cnt.append(tmp_error_cnt)
        # merge conv data
        merge_all_conv = [[],[],[]]
        all_prob = np.concatenate(all_prob)
        all_error_cnt = np.concatenate(all_error_cnt)
        print all_prob.shape
        for k in range(len(all_conv)):
            merge_all_conv[k] = np.concatenate(all_conv[k])
        perf = 1-1.0*sum(all_error_cnt)/len(test_set_all)
        print ("cv: {} Accuracy: {}".format(i, perf))
        results.append(perf)
        cal_f1_with_all_preds(all_prob, test_set_all[:, -1])
        write_event_keywords(dataset_dir, test_set_all, flag,filter_hs, i, merge_all_conv, cv_data.test_event_id, all_prob, cv_data.test_context, idx_word_map)
    print 'total time : %.2f minutes' % ((time.time()-start_time)/60)
    print ("Cross-validation average results: {}".format(np.mean(results)))
