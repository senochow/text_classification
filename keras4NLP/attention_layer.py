############################################################################
#
# Copyright (c) 2016 ICT MCG Group, Inc. All Rights Reserved
#
###########################################################################
"""
Brief:

Authors: zhouxing(@ict.ac.cn)
Date:    2016/09/12 19:11:14
File:    attention_layer.py
"""
from keras import backend as K
from keras.engine.topology import Layer
from keras.layers import activations, initializations, RepeatVector

class AttentionLayer(Layer):
    '''Attention Layer over LSTM

    '''
    def __init__(self, output_dim, init='glorot_uniform', attn_activation='tanh', **kwargs):
        self.output_dim = output_dim
        self.init = initializations.get(init)
        self.attn_activation = activations.get(attn_activation)
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape): # (batch, steps, dim)
        input_dim = input_shape[2]
        self.steps = input_shape[1]
        print input_shape
        self.W_s = self.add_weight(shape=(input_dim, self.output_dim),
                initializer=self.init,
                name='{}_Ws'.format(self.name),
                trainable=True)
        self.B_s = self.add_weight((self.output_dim,),
                initializer='zero',
                name='{}_bs'.format(self.name))
        self.Attention_vec = self.add_weight((self.output_dim,),
                initializer='normal',
                name='{}_att_vec'.format(self.name))
        self.built = True
    def call(self, x, mask=None):
        # 1. transform, (None, steps, idim)*(idim, outdim) -> (None, steps, outdim)
        u = self.attn_activation(K.dot(x, self.W_s) + self.B_s)
        # 2. * attention sum : {(None, steps, outdim) *(outdim), axis = 2} -> (None, steps)
        att = K.sum(u*self.Attention_vec, axis=2)
        # 3. softmax, (None, steps)
        att = K.exp(att)
        att_sum = K.sum(att, axis=1)
        att_sum = att_sum.dimshuffle(0,'x')
        #att_sum = K.expand_dims(att_sum, 1)
        att = att/att_sum
        # 4. weighted sum
        att = att.dimshuffle(0, 1, 'x')
        #att = K.expand_dims(att, 2)
        va = att*x
        v = K.sum(va, axis=1)
        return v

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[2])


# vim: set expandtab ts=4 sw=4 sts=4 tw=100:
