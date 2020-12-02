import numpy as np
from keras.layers import Layer
import tensorflow as tf
import os
import keras.backend as K

class Lookup(Layer):
    def __init__(self, batch_size, **kwargs):
        self.batch_size = batch_size
        super(Lookup, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        x = inputs[0]
        arm = inputs[1]
        if K.dtype(arm) != 'int32':
            arm = K.cast(arm, 'int32')

        print('out', x)
        print('out2', arm)
        outs = []
        for _i in range(self.batch_size):
            out1 = tf.nn.embedding_lookup(x[_i], arm[_i])
            outs.append(out1)

        out = tf.stack(outs, axis=0)
        print('return out',out)
        return out

    def compute_output_shape(self, input_shape):
        x_shape = input_shape[0]
        arm_shape = input_shape[1]
        f_num = x_shape[3]
        r_num = x_shape[1]
        t_num = x_shape[2]
        a_num = arm_shape[2]

        return (x_shape[0], r_num, a_num, t_num, f_num)


class LookUpSqueeze(Layer):
    def call(self, inputs, **kwargs):
        output = tf.squeeze(inputs, axis=2)
        return output

    def compute_output_shape(self, input_shape):
        return input_shape[:2] + input_shape[3:]


