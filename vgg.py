# Copyright (c) 2015-2017 Anish Athalye. Released under GPLv3.

import tensorflow as tf
import numpy as np
import scipy.io

VGG19_LAYERS = (
    'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

    'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

    'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
    'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

    'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
    'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

    'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
    'relu5_3', 'conv5_4', 'relu5_4'
)

def load_net(data_path):
    data = scipy.io.loadmat(data_path)
    mean = data['normalization'][0][0][0]
    mean_pixel = np.mean(mean, axis=(0, 1))
    weights = data['layers'][0]
    return weights, mean_pixel

def net_preloaded(weights, input_image, pooling,bit_map):
    net = {}
    current = input_image
    current_bitMap = bit_map
    for i, name in enumerate(VGG19_LAYERS):
        kind = name[:4]
        if kind == 'conv':
            kernels, bias = weights[i][0][0][0][0]
            # matconvnet: weights are [width, height, in_channels, out_channels]
            # tensorflow: weights are [height, width, in_channels, out_channels]
            kernels = np.transpose(kernels, (1, 0, 2, 3))
            bias = bias.reshape(-1)
            current,current_bitMap = _conv_layer(current, kernels, bias,current_bitMap)
        elif kind == 'relu':
            current = tf.nn.relu(current)
        elif kind == 'pool':
            current,current_bitMap = _pool_layer(current, pooling,current_bitMap)
        net[name] = current

    assert len(net) == len(VGG19_LAYERS)
    return net

def resetBitMap(bit_map):
    zero = tf.constant(0, dtype=tf.float32)
    where = tf.not_equal(bit_map, zero)
    return tf.cast(where,tf.float32)

def _conv_layer(input, weights, bias,bit_map):

    conv = tf.nn.conv2d(input, tf.constant(weights), strides=(1, 1, 1, 1),
            padding='SAME')
    conv_bitmap = tf.nn.conv2d(bit_map, tf.ones(weights.shape), strides=(1, 1, 1, 1),
            padding='SAME')
    conv_bitmap = resetBitMap(conv_bitmap)
    return tf.multiply(tf.nn.bias_add(conv, bias),conv_bitmap),conv_bitmap


def _pool_layer(input, pooling,bit_map):
    if pooling == 'avg':
        pool = tf.nn.avg_pool(input, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1),
                padding='SAME')
        bit_map_pool = tf.nn.avg_pool(bit_map, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1),
                padding='SAME')
    else:
        pool = tf.nn.max_pool(input, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1),
                padding='SAME')
        bit_map_pool = tf.nn.max_pool(bit_map, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1),
                padding='SAME')
    bit_map_pool = resetBitMap(bit_map_pool)
    return tf.multiply(pool,bit_map_pool),bit_map_pool

def preprocess(image, mean_pixel):
    return image - mean_pixel


def unprocess(image, mean_pixel):
    return image + mean_pixel
