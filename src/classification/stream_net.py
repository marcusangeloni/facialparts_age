#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Marcus de Assis Angeloni <marcus.angeloni@ic.unicamp.br>
# Rodrigo de Freitas Pereira <rodrigodefreitas12@gmail.com>
# Helio Pedrini <helio@ic.unicamp.br>
# Mon 4 Feb 2019 18:00:00

from __future__ import division

import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.logging.set_verbosity(tf.logging.INFO)

def _conv(_input, n_filters, kernel_size, reg_val, name = None, activation_function = tf.nn.elu):
    return tf.layers.conv2d(_input, filters = n_filters,
                            kernel_size = kernel_size,
                            strides = (1, 1),
                            activation = activation_function,
                            padding = 'same',
                            kernel_regularizer = tf.contrib.layers.l2_regularizer(
                                reg_val),
                            kernel_initializer = tf.contrib.layers.variance_scaling_initializer(),
                            bias_initializer = tf.contrib.layers.variance_scaling_initializer(),
                            bias_regularizer = tf.contrib.layers.l2_regularizer(
                                reg_val),
                            name = name)

def _dense(_input, n_outputs, reg_val, name = None, activation_function = tf.nn.elu):
    return tf.layers.dense(_input, units = n_outputs,                            
                            activation = activation_function,                            
                            kernel_regularizer = tf.contrib.layers.l2_regularizer(
                                reg_val),
                            kernel_initializer = tf.contrib.layers.variance_scaling_initializer(),
                            bias_initializer = tf.contrib.layers.variance_scaling_initializer(),
                            bias_regularizer = tf.contrib.layers.l2_regularizer(
                                reg_val),
                            name = name)

def StreamNet(input_tensor, is_training = True, n_classes = 8):
    with tf.variable_scope("StreamNet"):
        endpoints = {}

        input_eyebrows = input_tensor["eyebrows"]
        input_eyes = input_tensor["eyes"]
        input_nose = input_tensor["nose"]
        input_mouth = input_tensor["mouth"]
        
        # Eyebrows
        eyebrows_conv1 = _conv(input_eyebrows, 32, (3, 3), FLAGS.reg_val,
            name = 'eyebrows_conv_1_1')
        eyebrows_pool1 = tf.layers.max_pooling2d(
            eyebrows_conv1, (2, 2), (2, 2), padding = 'same', name = 'eyebrows_MaxPool_1')

        eyebrows_conv2 = _conv(eyebrows_pool1, 32, (3, 3), FLAGS.reg_val,
            name = 'eyebrows_conv_2_1')
        eyebrows_pool2 = tf.layers.max_pooling2d(
            eyebrows_conv2, (2, 2), (2, 2), padding = 'same', name = 'eyebrows_MaxPool_2')

        eyebrows_conv3 = _conv(eyebrows_pool2, 32, (3, 3), FLAGS.reg_val,
            name = 'eyebrows_conv_3_1')
        eyebrows_pool3 = tf.layers.max_pooling2d(
            eyebrows_conv3, (2, 2), (2, 2), padding = 'same', name = 'eyebrows_MaxPool_3')

        eyebrows_flatten = tf.layers.flatten(eyebrows_pool3)
        endpoints['eyebrows_flatten'] = eyebrows_flatten

        eyebrows_dense1 = _dense(eyebrows_flatten, 256, FLAGS.reg_val, 'eyebrows_dense_1')
        endpoints['eyebrows_dense1'] = eyebrows_dense1

        net_eyebrows = tf.layers.dropout(eyebrows_dense1, rate = FLAGS.dropout, name = 'eyebrows_dropout_1', training = is_training)

        eyebrows_dense4 = _dense(eyebrows_dense1, n_classes, FLAGS.reg_val, 'eyebrows_dense_4')
        endpoints['eyebrows_dense4'] = eyebrows_dense4
        
        # Eyes
        eyes_conv1 = _conv(input_eyes, 32, (3, 3), FLAGS.reg_val,
            name = 'eyes_conv_1_1')
        eyes_pool1 = tf.layers.max_pooling2d(
            eyes_conv1, (2, 2), (2, 2), padding = 'same', name = 'eyes_MaxPool_1')

        eyes_conv2 = _conv(eyes_pool1, 32, (3, 3), FLAGS.reg_val,
            name = 'eyes_conv_2_1')
        eyes_pool2 = tf.layers.max_pooling2d(
            eyes_conv2, (2, 2), (2, 2), padding = 'same', name = 'eyes_MaxPool_2')

        eyes_conv3 = _conv(eyes_pool2, 32, (3, 3), FLAGS.reg_val,
            name = 'eyes_conv_3_1')
        eyes_pool3 = tf.layers.max_pooling2d(
            eyes_conv3, (2, 2), (2, 2), padding = 'same', name = 'eyes_MaxPool_3')        

        eyes_flatten = tf.layers.flatten(eyes_pool3)
        endpoints['eyes_flatten'] = eyes_flatten

        eyes_dense1 = _dense(eyes_flatten, 256, FLAGS.reg_val, 'eyes_dense_1')
        endpoints['eyes_dense1'] = eyes_dense1

        net_eyes = tf.layers.dropout(eyes_dense1, rate = FLAGS.dropout, name = 'eyes_dropout', training = is_training)

        eyes_dense4 = _dense(eyes_dense1, n_classes, FLAGS.reg_val, 'eyes_dense_4')
        endpoints['eyes_dense4'] = eyes_dense4

        # Nose
        nose_conv1 = _conv(input_nose, 32, (3, 3), FLAGS.reg_val,
            name = 'nose_conv_1_1')
        nose_pool1 = tf.layers.max_pooling2d(
            nose_conv1, (2, 2), (2, 2), padding = 'same', name = 'nose_MaxPool_1')

        nose_conv2 = _conv(nose_pool1, 32, (3, 3), FLAGS.reg_val,
            name = 'nose_conv_2_1')
        nose_pool2 = tf.layers.max_pooling2d(
            nose_conv2, (2, 2), (2, 2), padding = 'same', name = 'nose_MaxPool_2')

        nose_conv3 = _conv(nose_pool2, 32, (3, 3), FLAGS.reg_val,
            name = 'nose_conv_3_1')
        nose_pool3 = tf.layers.max_pooling2d(
            nose_conv3, (2, 2), (2, 2), padding = 'same', name = 'nose_MaxPool_3')
        
        nose_flatten = tf.layers.flatten(nose_pool3)
        endpoints['nose_flatten'] = nose_flatten

        nose_dense1 = _dense(nose_flatten, 256, FLAGS.reg_val, 'nose_dense_1')
        endpoints['nose_dense1'] = nose_dense1

        net_nose = tf.layers.dropout(nose_dense1, rate = FLAGS.dropout, name = 'nose_dropout', training = is_training)

        nose_dense4 = _dense(nose_dense1, n_classes, FLAGS.reg_val, 'nose_dense_4')
        endpoints['nose_dense4'] = nose_dense4

        # Mouth
        mouth_conv1 = _conv(input_mouth, 32, (3, 3), FLAGS.reg_val,
            name = 'mouth_conv_1_1')
        mouth_pool1 = tf.layers.max_pooling2d(
            mouth_conv1, (2, 2), (2, 2), padding = 'same', name = 'mouth_MaxPool_1')

        mouth_conv2 = _conv(mouth_pool1, 32, (3, 3), FLAGS.reg_val,
            name = 'mouth_conv_2_1')
        mouth_pool2 = tf.layers.max_pooling2d(
            mouth_conv2, (2, 2), (2, 2), padding = 'same', name = 'mouth_MaxPool_2')

        mouth_conv3 = _conv(mouth_pool2, 32, (3, 3), FLAGS.reg_val,
            name = 'mouth_conv_3_1')
        mouth_pool3 = tf.layers.max_pooling2d(
            mouth_conv3, (2, 2), (2, 2), padding = 'same', name = 'mouth_MaxPool_3')
        
        mouth_flatten = tf.layers.flatten(mouth_pool3)
        endpoints['mouth_flatten'] = mouth_flatten

        mouth_dense1 = _dense(mouth_flatten, 256, FLAGS.reg_val, 'mouth_dense_1')
        endpoints['mouth_dense1'] = mouth_dense1

        net_mouth = tf.layers.dropout(mouth_dense1, rate = FLAGS.dropout, name = 'mouth_dropout', training = is_training)

        mouth_dense4 = _dense(mouth_dense1, n_classes, FLAGS.reg_val, 'mouth_dense_4')
        endpoints['mouth_dense4'] = mouth_dense4
        
        # Concat streams
        net = tf.concat([net_eyebrows, net_eyes, net_nose, net_mouth], axis = 1, name = 'concat')

        net = _dense(net, 256, FLAGS.reg_val, 'dense_1')
        net = tf.layers.dropout(net, rate = FLAGS.dropout, name = 'dense_1_dropout', training = is_training)
        net = _dense(net, 256, FLAGS.reg_val, 'dense_2')
        net = tf.layers.dropout(net, rate = FLAGS.dropout, name = 'dense_2_dropout', training = is_training)
        net = _dense(net, n_classes, FLAGS.reg_val, 'dense_3')

        endpoints['logits'] = net
        
        return net, endpoints
