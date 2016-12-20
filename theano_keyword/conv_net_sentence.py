#! -*- coding:utf8 -*-
"""
Much of the code is modified from
- deeplearning.net (for ConvNet classes)
- https://github.com/mdenil/dropout (for dropout)
- https://groups.google.com/forum/#!topic/pylearn-dev/3QbKtCumAW4 (for Adadelta)
"""
import cPickle as pk
import numpy as np
import theano
import theano.tensor as T
import re
import warnings
import sys
import time
import ConfigParser
warnings.filterwarnings("ignore")

from utils.evaluation import cal_f1_score
from utils.conv_net_classes import Iden
from utils.learning import shared_dataset
from utils.learning import sgd_updates_adadelta
from utils.evaluation import cal_f1_with_all_preds
from utils.cv_data_helper import *
from utils.conv_net_classes import *

def save_param(model_file, params):
    write_file = open(model_file, "wb")
    model_params = {}
    model_params["clf"] = params[0]


def train_conv_net(cv_data, data_dir, cv, U, model_param, activations=[Iden]):
    """
    U: wordvec : {word_index: vector feature}
    img_h = sentence length (padded where necessary), 固定的长度：equal to max sentence length in dataset(pre computed)
    img_w = word vector length (300 for word2vec)
    hidden_units = [x,y] x is the number of feature maps (per filter window), and y is the penultimate layer
    sqr_norm_lim = s^2 in the paper
    lr_decay = adadelta decay parameter
    """
    rng = np.random.RandomState(3435)
    filter_hs = model_param['filter_hs']
    hidden_units = model_param['hidden_units'][:] # 复制value，否则下一次迭代会被修改
    dropout_rate = model_param['dropout_rate']
    batch_size = model_param['batch_size']
    img_w = model_param['word_dim']
    img_h = len(cv_data.train[0])-model_param['extra_fea_len']-1  # last one is y
    print "img height ", img_h
    filter_w = img_w
    feature_maps = hidden_units[0]
    filter_shapes = []
    pool_sizes = []
    for filter_h in filter_hs:
        # filter: conv shape, hidden layer: 就是最后的全连接层
        filter_shapes.append((feature_maps, 1, filter_h, filter_w))
        pool_sizes.append((img_h-filter_h+1, img_w-filter_w+1))
    parameters = [("image shape",img_h,img_w),("filter shape",filter_shapes), ("hidden_units",hidden_units),
                  ("dropout", dropout_rate), ("batch_size",batch_size),("non_static", model_param['non_static']),
                    ("learn_decay",model_param['lr_decay']), ("conv_non_linear", model_param['conv_non_linear'])
                    ,("sqr_norm_lim",model_param['l2_norm']),("shuffle_batch", model_param['shuffle_batch'])]
    print parameters

    #define model architecture
    index = T.lscalar()
    x = T.matrix('x')
    y = T.ivector('y')
    Words = theano.shared(value = U, name = "Words")
    # ??? set zero
    zero_vec_tensor = T.vector()
    zero_vec = np.zeros(img_w)
    set_zero = theano.function([zero_vec_tensor], updates=[(Words, T.set_subtensor(Words[0,:], zero_vec_tensor))], allow_input_downcast=True)
    # 转成 batch(50)*1*sent_len(134)*k(300)
    layer0_input = Words[T.cast(x[:,:img_h].flatten(),dtype="int32")].reshape((x.shape[0],1,img_h,Words.shape[1]))
    layer1_input_extra_fea = x[:,img_h:]
    conv_layers = []
    layer1_inputs = []
    # each filter has its own conv layer: the full conv layers = concatenate all layer to 1
    for i in xrange(len(filter_hs)):
        filter_shape = filter_shapes[i]
        pool_size = pool_sizes[i]
        conv_layer = LeNetConvPoolLayer(rng, input=layer0_input,image_shape=(batch_size, 1, img_h, img_w),
                                filter_shape=filter_shape, poolsize=pool_size, non_linear=model_param['conv_non_linear'])
        layer1_input = conv_layer.output.flatten(2)
        conv_layers.append(conv_layer)
        layer1_inputs.append(layer1_input)

    if model_param['use_extra_feature']:
        layer1_inputs.append(layer1_input_extra_fea)

    layer1_input = T.concatenate(layer1_inputs,1)

    if model_param['use_extra_feature']:
        hidden_units[0] = feature_maps*len(filter_hs) + model_param['extra_fea_len']
    else:
        hidden_units[0] = feature_maps*len(filter_hs)

    classifier = MLPDropout(rng, input=layer1_input, layer_sizes=hidden_units, activations=activations, dropout_rates=dropout_rate)

    #define parameters of the model and update functions using adadelta
    params = classifier.params
    for conv_layer in conv_layers:
        params += conv_layer.params
    if model_param['non_static']:
        #if word vectors are allowed to change, add them as model parameters
        params += [Words]
    cost = classifier.negative_log_likelihood(y)
    dropout_cost = classifier.dropout_negative_log_likelihood(y)
    grad_updates = sgd_updates_adadelta(params, dropout_cost, model_param['lr_decay'], 1e-6, model_param['l2_norm'])

    # 2. model run part ...
    train_set_x, train_set_y, val_set_x, val_set_y, n_train_batches, n_val_batches = gen_train_validation(cv_data.train, batch_size)
    #test_set_x, test_set_y, test_set_remand_x, test_set_remand_y, test_size_remand, n_test_batches, test_size_batch = gen_test_batch_data(cv_data.test, batch_size)
    #test_size_all = cv_data.test.shape[0]
    test_set_x = cv_data.test[:, :img_h]
    test_set_y = np.asarray(cv_data.test[:,-1],"int32")
    test_size = test_set_x.shape[0]
    # 3. compile theano functions to get train/val/test errors
    '''
    test_model_batch = theano.function([index], classifier.errors(y),
             givens={
                x: test_set_x[index * batch_size: (index + 1) * batch_size],
                y: test_set_y[index * batch_size: (index + 1) * batch_size]}, allow_input_downcast=True)
    '''
    test_model = theano.function([index], classifier.errors(y),
             givens={
                x: train_set_x[index * batch_size: (index + 1) * batch_size],
                y: train_set_y[index * batch_size: (index + 1) * batch_size]}, allow_input_downcast=True)
    train_model = theano.function([index], cost, updates=grad_updates,
          givens={
            x: train_set_x[index*batch_size:(index+1)*batch_size],
            y: train_set_y[index*batch_size:(index+1)*batch_size]}, allow_input_downcast=True)
    val_model = theano.function([index], classifier.errors(y),
         givens={
            x: val_set_x[index * batch_size: (index + 1) * batch_size],
            y: val_set_y[index * batch_size: (index + 1) * batch_size]}, allow_input_downcast=True)
    # 4. run theano models
    test_pred_layers = []
    test_layer0_input = Words[T.cast(x.flatten(),dtype="int32")].reshape((test_size,1,img_h,Words.shape[1]))
    #test_layer0_input = Words[T.cast(x[:, :img_h].flatten(),dtype="int32")].reshape((test_size_batch,1,img_h,Words.shape[1]))
    test_layer1_input_extra_fea = x[:,img_h:]

    # [(instances * feature_maps * conv_feature)], different filter size have different conv dimention
    test_conv_data = []
    for conv_layer in conv_layers:
        test_layer0_output = conv_layer.predict(test_layer0_input, test_size)
        test_layer0_convs = conv_layer.predict_maxpool(test_layer0_input, test_size)
        test_pred_layers.append(test_layer0_output.flatten(2))
        test_conv_data.append(test_layer0_convs.flatten(3))
    # conv data
    if model_param['use_extra_feature']:
        test_pred_layers.append(test_layer1_input_extra_fea)

    test_layer1_input = T.concatenate(test_pred_layers, 1)
    test_y_pred = classifier.predict(test_layer1_input)
    test_y_pred_p = classifier.predict_p(test_layer1_input)
    test_error1 = T.neq(test_y_pred, y)
    test_error = T.mean(T.neq(test_y_pred, y))
    test_model_all = theano.function([x,y], test_error, allow_input_downcast=True)
    test_model_f1 = theano.function([x], test_y_pred_p, allow_input_downcast=True)
    test_model_prob = theano.function([x], test_y_pred_p, allow_input_downcast=True)
    test_layer1_feature = theano.function([x], test_layer1_input, allow_input_downcast=True)
    test_extra_fea = theano.function([x], test_layer1_input_extra_fea, allow_input_downcast=True)
    get_all_conv_data = theano.function([x], test_conv_data, allow_input_downcast=True)
    # 5. training ..start training over mini-batches
    print '... training'
    epoch = 0
    best_val_perf = 0
    val_perf = 0
    test_perf, tmp_pred_prob = 0, 0
    cost_epoch = 0
    fp = 0
    avg_precsion = 0
    param_file = data_dir
    if model_param['use_extra_feature']:
        param_file += "/cnn_param_extra_"+str(cv)+".pk"
    else:
        param_file += "/cnn_param_"+str(cv)+".pk"
    n_epochs = model_param['epochs']
    while (epoch < n_epochs):
        start_time = time.time()
        epoch = epoch + 1
        if model_param['shuffle_batch']:
            for minibatch_index in np.random.permutation(range(n_train_batches)):
                cost_epoch = train_model(minibatch_index)
                set_zero(zero_vec)
        else:
            for minibatch_index in xrange(n_train_batches):
                cost_epoch = train_model(minibatch_index)
                set_zero(zero_vec)
        train_losses = [test_model(i) for i in xrange(n_train_batches)]
        train_perf = 1 - np.mean(train_losses)
        val_losses = [val_model(i) for i in xrange(n_val_batches)]
        val_perf = 1- np.mean(val_losses)
        print('epoch: %i, training time: %.2f secs, train perf: %.2f %%, val perf: %.2f %%' % (epoch, time.time()-start_time, train_perf * 100., val_perf*100.))
        '''
        test_losses = [float(test_model_batch(i)*batch_size) for i in xrange(n_test_batches)]
        test_loss_remand = test_model_all(test_set_remand_x,test_set_remand_y)
        neq_all = sum(test_losses) + sum(test_loss_remand)
        error_all = 1.0*neq_all/test_size_all
        fp = 1 - error_all
        print 'cur test loss: ', fp
        '''
        #test_losses = [test_model_batch(i) for i in xrange(n_test_batches)]
        #test_loss = test_model_all(test_set_x,test_set_y)
        #test_perf = 1- test_loss
        #print 'cur test loss: ', test_perf
        if epoch == n_epochs+1:
            tmp_pred_prob = test_model_prob(test_set_x)
            test_losses = [test_model_batch(i) for i in xrange(n_test_batches)]
            test_perf = 1 - np.mean(test_losses)
            fp = test_perf
            print 'last : ', fp
            cal_f1_score(tmp_pred_prob, test_set_y)
        if val_perf >= best_val_perf:
            best_val_perf = val_perf
            test_loss = test_model_all(test_set_x,test_set_y)
            test_perf = 1- test_loss
            tmp_pred_prob = test_model_prob(test_set_x)
            # save params
            write_file = open(param_file, 'wb')
            pk.dump(classifier.params, write_file, -1)
            for conv_layer in conv_layers:
                pk.dump(conv_layer.params, write_file, -1)
            write_file.close()

    return test_perf, fp, avg_precsion, tmp_pred_prob

def get_model_param(cf_pair, flag):
    '''Set model parameter, some params load from config file and some set here

    '''
    model_param = dict(cf_pair)
    model_param['epochs'] = int(model_param['epochs'])
    model_param['lr_decay'] = float(model_param['lr_decay'])
    model_param['l2_norm'] = int(model_param['l2_norm'])
    model_param['dropout_rate'] = [float(model_param['dropout_rate'])]
    model_param['filter_hs'] = [3,4,5]
    model_param['max_filter_h'] = max(model_param['filter_hs'])
    model_param['conv_non_linear'] = "relu"
    model_param['hidden_units'] = [100,2]
    model_param['shuffle_batch'] = True
    model_param['batch_size'] = 50
    model_param['use_extra_feature'] = flag
    model_param['word_dim'] = 300
    model_param['extra_fea_len'] = 9
    if model_param['mode']=="-nonstatic":
        print "model architecture: CNN-non-static"
        model_param['non_static'] = True
    elif model_param['mode']=="-static":
        print "model architecture: CNN-static"
        model_param['non_static'] = False
    if model_param['word_vectors'] == "-rand":
        print "using: random vectors"
    elif model_param['word_vectors'] == "-word2vec":
        print "using: word2vec vectors"
    return model_param



def train_cross_validation(n, data, data_dir, word_idx_map, max_length, U, model_param):
    ''' n-fold Cross-validation
        Args:
            data: list of instances{y, mid, text, extra_fea, num_words, split, event_id}
            word_idx_map: word -> index
            max_length:  max sentence length of dataset
            k:  word embedding dim
            filter_h:  max filter height
    '''
    all_perf = []
    fres = []
    avg_plist = []
    start_time = time.time()
    for i in range(0, n):
        cv_data = make_idx_data_cv(data, word_idx_map, i, max_l=max_length ,k=model_param['word_dim'], filter_h=model_param['max_filter_h'])
        perf, fp, avg_precsion, probs = train_conv_net(cv_data, data_dir, i, U, model_param)
        print ("CV-{} perf: {}".format(i, perf))
        cal_f1_with_all_preds(probs, np.asarray(cv_data.test[:,-1],"int32"))
        all_perf.append(perf)
        fres.append(fp)
        avg_plist.append(avg_precsion)
    print 'total time : %.2f minutes' % ((time.time()-start_time)/60)
    print ("Cross-validation accuracy :{}".format(np.mean(all_perf)))
    print str(np.mean(fres))

def main():
    cv = int(sys.argv[1])
    dataset_name = sys.argv[2]
    conf_file = sys.argv[3]
    flag = int(sys.argv[4])
    cf = ConfigParser.ConfigParser()
    cf.read(conf_file)
    model_param = get_model_param(cf.items("model_param"), flag)
    data_dir = cf.get(dataset_name, "dataset_dir")
    x = pk.load(open(cf.get(dataset_name, "pkfile"+str(cv)),"rb"))
    revs, W, W2, word_idx_map, max_length = x[0], x[1], x[2], x[3],x[6]# x[4], x[5], x[6]
    print "data loaded!"
    U = W2
    if model_param['word_vectors'] == "-rand":
        U = W2
    elif model_param['word_vectors'] == "-word2vec":
        U = W
    # cross-validation training...
    train_cross_validation(cv, revs, data_dir, word_idx_map, max_length, U, model_param)

if __name__=="__main__":
    main()
