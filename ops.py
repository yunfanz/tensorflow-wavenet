import math
import numpy as np
import tensorflow as tf

from tensorflow.python.framework import ops

from utils import *

class batch_norm(object):
    """Code modification of http://stackoverflow.com/a/33950177"""
    def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum

            self.ema = tf.train.ExponentialMovingAverage(decay=self.momentum)
            self.name = name

    def __call__(self, x, train=True):
        shape = x.get_shape().as_list()

        if train:
            with tf.variable_scope(self.name) as scope:
                self.beta = tf.get_variable("beta", [shape[-1]],
                                    initializer=tf.constant_initializer(0.))
                self.gamma = tf.get_variable("gamma", [shape[-1]],
                                    initializer=tf.random_normal_initializer(1., 0.02))

                try:
                    batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
                except:
                    batch_mean, batch_var = tf.nn.moments(x, [0, 1], name='moments')

                ema_apply_op = self.ema.apply([batch_mean, batch_var])
                self.ema_mean, self.ema_var = self.ema.average(batch_mean), self.ema.average(batch_var)

                with tf.control_dependencies([ema_apply_op]):
                    mean, var = tf.identity(batch_mean), tf.identity(batch_var)
        else:
            mean, var = self.ema_mean, self.ema_var

        normed = tf.nn.batch_norm_with_global_normalization(
                x, mean, var, self.beta, self.gamma, self.epsilon, scale_after_normalization=True)

        return normed

def binary_cross_entropy(preds, targets, name=None):
    """Computes binary cross entropy given `preds`.

    For brevity, let `x = `, `z = targets`.  The logistic loss is

        loss(x, z) = - sum_i (x[i] * log(z[i]) + (1 - x[i]) * log(1 - z[i]))

    Args:
        preds: A `Tensor` of type `float32` or `float64`.
        targets: A `Tensor` of the same type and shape as `preds`.
    """
    eps = 1e-12
    with ops.op_scope([preds, targets], name, "bce_loss") as name:
        preds = ops.convert_to_tensor(preds, name="preds")
        targets = ops.convert_to_tensor(targets, name="targets")
        return tf.reduce_mean(-(targets * tf.log(preds + eps) +
                              (1. - targets) * tf.log(1. - preds + eps)))

def conv_cond_concat(x, y):
    """Concatenate conditioning vector on feature map axis."""
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()
    return tf.concat(3, [x, y*tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])])

def conv1d(input_, output_dim,
           k_w=5, d_w=4, stddev=0.02,
           name="conv1d"):
    """Conputes a 1-D Convolution across a 3-D input"""
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv1d(input_, w, stride=d_w, padding='SAME')

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        #import IPython; IPython.embed()
        #conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
        conv = tf.nn.bias_add(conv, biases) #can use -1 in reshape, but not None
        #import IPython; IPython.embed()

        return conv

def deconv1d(input_, output_shape,
             k_w=5, d_w=4, stddev=0.02,
            name="deconv1d", with_w=False):
    """Computes a filtered convolution across a 3-D input tensor."""
    with tf.variable_scope(name):
    # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable(name='w', shape=[1, k_w, output_shape[-1], input_.get_shape()[-1]],
                           initializer=tf.random_normal_initializer(stddev=stddev))
        output_shape.insert(0,1)
        input_ = tf.expand_dims(input_, 0)
        deconv = tf.nn.conv2d_transpose(input_, filter=w, output_shape=output_shape, strides=[1, 1, d_w, 1])
        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))

        deconv = tf.reshape(deconv, output_shape)
        deconv = tf.squeeze(deconv, [0]) #removes the first dummy dimension
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        if with_w:
            return deconv, w, biases
        else:
            return deconv


def lrelu(x, leak=0.2, name="lrelu"):
  return tf.maximum(x, leak*x)

#G
def mb_disc_layer(input_,B=1000, C=5, stddev=0.0002, with_w=False,name='mb_disc'):
    ''' mini-batch discrimination '''
    shape = input_.get_shape().as_list()

    with tf.variable_scope(name):
        tensor = tf.get_variable("Tensor", [shape[-1], B*C], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev), trainable=False)
        #bias = tf.get_variable("bias", [output_length],
        #    initializer=tf.constant_initializer(bias_start))
        M = tf.reshape(tf.matmul(input_, tensor),[shape[0],B,C])

        diff = []
        for i in range(shape[0]):  #loop over samples in mini_batch
            diff.append(tf.squared_difference(M[i],M))
        diff = tf.pack(diff)

        ox = tf.exp(-tf.sqrt(tf.reduce_sum(diff,reduction_indices=[1,3])))
        output_ = tf.concat(1,[input_, ox])
        #import IPython; IPython.embed()
        #print(output_.get_shape())
        if with_w:
            return output_, tensor
        else:
            return output_

def linear(input_, output_length, name=None, stddev=0.02, bias_start=0.0, with_w=False, missing_dim=-1):
    shape = input_.get_shape().as_list()
    if missing_dim < 0: missing_dim = shape[-1]
    with tf.variable_scope(name or "Linear"):
        matrix = tf.get_variable("Matrix", [missing_dim, output_length], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_length],
            initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias
