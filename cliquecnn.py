# -*- coding: utf-8 -*-
""" AlexNet as a Class"""

import sys
import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.io as scipyio
import time

import caffe
from caffe import layers as L, params as P  # for net definition
from caffe.proto import caffe_pb2  # for solver definition


# -----------------------------------------------------------------------------
class CliqueCNN(object):
    """ Constructor """

    def __init__(self, data_shape, label_shape, num_classes=303, last_layer='fc8_output', params=[]):
        self.data_shape = list(data_shape)
        self.data = L.DummyData(shape=dict(dim=list(data_shape)))
        self.label = L.DummyData(shape=dict(dim=list(label_shape)))
        self.num_classes = num_classes

        self.last_layer = last_layer
        self.net = self.net(params=params)
        # self.solver = self.solver(params)

    """ Net architecture """

    def __network_end(self, n, last_layer, params):  # caffe.NetSpec()

        n.label = self.label
        n.loss_l2 = L.EuclideanLoss(last_layer, n.label)

        # add the backprop to the input
        proto = n.to_proto()
        proto = 'force_backward: true\n' + str(proto)

        # write the net to a file
        f = open(params['path2net'], 'w')
        f.write(proto)
        f.close()

    def net(self, params=[]):

        # initialize net and data layer
        n = caffe.NetSpec()

        # layer 0
        n.data = self.data
        # layer 1
        n.conv1 = L.Convolution(n.data, name='conv1', num_output=96,  kernel_size=11, stride=4,
                                weight_filler=dict(type='gaussian', std=0.01),
                                bias_filler=dict(type='constant', value=0),
                                param=[dict(name='conv1_w', lr_mult=0.01, decay_mult=1),
                                       dict(name='conv1_b', lr_mult=0.02, decay_mult=0)])
        if self.last_layer == 'conv1':
            self.__network_end(n, n.conv1, params)

        n.relu1 = L.ReLU(n.conv1, in_place=True)
        if self.last_layer == 'relu1':
            self.__network_end(n, n.relu1, params)

        n.norm1 = L.LRN(n.relu1, local_size=5, alpha=0.0001, beta=0.75)
        if self.last_layer == 'norm1':
            self.__network_end(n, n.norm1, params)

        n.pool1 = L.Pooling(n.norm1, kernel_size=3, stride=2, pool=P.Pooling.MAX)
        if self.last_layer == 'pool1':
            self.__network_end(n, n.pool1, params)

        # layer 2
        n.conv2 = L.Convolution(n.pool1, name='conv2', num_output=256, pad=2, kernel_size=5,
                                group=2,
                                weight_filler=dict(type='gaussian', std=0.01),
                                bias_filler=dict(type='constant', value=0.1),
                                param=[
                                    dict(name='conv2_w', lr_mult=0.01, decay_mult=1),
                                    dict(name='conv2_b', lr_mult=0.02, decay_mult=0)])
        if self.last_layer == 'conv2':
            self.__network_end(n, n.conv2, params)

        n.relu2 = L.ReLU(n.conv2, in_place=True)
        if self.last_layer == 'relu2':
            self.__network_end(n, n.relu2, params)

        n.norm2 = L.LRN(n.relu2, local_size=5, alpha=0.0001, beta=0.75)
        if self.last_layer == 'norm2':
            self.__network_end(n, n.norm2, params)

        n.pool2 = L.Pooling(n.norm2, kernel_size=3, stride=2, pool=P.Pooling.MAX)
        if self.last_layer == 'pool2':
            self.__network_end(n, n.pool2, params)

        # layer 3
        n.conv3 = L.Convolution(n.pool2, name='conv3', num_output=384, pad=1, kernel_size=3,
                                weight_filler=dict(type='gaussian', std=0.01),
                                bias_filler=dict(type='constant', value=0),
                                param=[dict(name='conv3_w', lr_mult=0.01, decay_mult=1),
                                       dict(name='conv3_b', lr_mult=0.02, decay_mult=0)])
        if self.last_layer == 'conv3':
            self.__network_end(n, n.conv3, params)

        n.relu3 = L.ReLU(n.conv3, in_place=True)
        if self.last_layer == 'relu3':
            self.__network_end(n, n.relu3, params)

        # layer 4
        n.conv4 = L.Convolution(n.relu3, name='conv4', num_output=384, pad=1, kernel_size=3,
                                group=2,
                                weight_filler=dict(type='gaussian', std=0.01),
                                bias_filler=dict(type='constant', value=0.1),
                                param=[dict(name='conv4_w', lr_mult=0.01, decay_mult=1),
                                       dict(name='conv4_b', lr_mult=0.02, decay_mult=0)])
        if self.last_layer == 'conv4':
            self.__network_end(n, n.conv4, params)

        n.relu4 = L.ReLU(n.conv4, in_place=True)
        if self.last_layer == 'relu4':
            self.__network_end(n, n.relu4, params)

        # layer 5
        n.conv5 = L.Convolution(n.relu4, name='conv5', num_output=256, pad=1, kernel_size=3,
                                group=2,
                                weight_filler=dict(type='gaussian', std=0.01),
                                bias_filler=dict(type='constant', value=0.1),
                                param=[dict(name='conv5_w', lr_mult=0.01, decay_mult=1),
                                       dict(name='conv5_b', lr_mult=0.02, decay_mult=0)])
        if self.last_layer == 'conv5':
            self.__network_end(n, n.conv5, params)

        n.relu5 = L.ReLU(n.conv5, in_place=True)
        if self.last_layer == 'conv5':
            self.__network_end(n, n.relu5, params)

        n.pool5 = L.Pooling(n.relu5, pool=P.Pooling.MAX, kernel_size=3, stride=2)
        if self.last_layer == 'pool5':
            self.__network_end(n, n.pool5, params)

        # layer 6
        n.fc6 = L.InnerProduct(n.pool5, name='fc6_', num_output=4096,
                               weight_filler=dict(type='gaussian', std=0.005),
                               bias_filler=dict(type='constant', value=0.1),
                               param=[dict(name='fc6__w', lr_mult=1, decay_mult=1),
                                      dict(name='fc6__b', lr_mult=2, decay_mult=0)])
        if self.last_layer == 'fc6_':
            self.__network_end(n, n.fc6, params)

        n.relu6 = L.ReLU(n.fc6, in_place=True)
        if self.last_layer == 'relu6':
            self.__network_end(n, n.relu6, params)
        n.drop6 = L.Dropout(n.relu6, in_place=True, dropout_ratio=0.5)
        if self.last_layer == 'drop6':
            self.__network_end(n, n.relu6, params)

        # layer 7
        n.fc7 = L.InnerProduct(n.drop6, name='fc7_', num_output=4096,
                               weight_filler=dict(type='gaussian', std=0.005),
                               bias_filler=dict(type='constant', value=0.1),
                               param=[dict(name='fc7__w', lr_mult=1, decay_mult=1),
                                      dict(name='fc7__b', lr_mult=2, decay_mult=0)])
        if self.last_layer == 'fc7_':
            self.__network_end(n, n.fc7, params)

        n.relu7 = L.ReLU(n.fc7, in_place=True)
        if self.last_layer == 'relu7':
            self.__network_end(n, n.relu7, params)

        n.drop7 = L.Dropout(n.relu7, in_place=True, dropout_ratio=0.5)
        if self.last_layer == 'drop7':
            self.__network_end(n, n.relu7, params)

        # layer 8: always learn fc8 (param=learned_param)
        n.fc8 = L.InnerProduct(n.drop7, name='fc8_output', num_output=self.num_classes,
                               weight_filler=dict(type='gaussian', std=0.01),
                               bias_filler=dict(type='constant', value=0),
                               param=[dict(name='fc8_output_w', lr_mult=1, decay_mult=1),
                                      dict(name='fc8_output_b', lr_mult=2, decay_mult=0)])
        if self.last_layer == 'fc8_output':
            self.__network_end(n, n.fc8, params)
# -----------------------------------------------------------------------------