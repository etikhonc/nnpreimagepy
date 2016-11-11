# -*- coding: utf-8 -*-
""" AlexNet as a Class"""

import os
import numpy as np

import caffe
from caffe import layers as L, params as P  # for net definition
from caffe.proto import caffe_pb2  # for solver definition


# -----------------------------------------------------------------------------
class VideoNet(object):
    """ Constructor """

    def __init__(self, data_shape, label_shape, last_layer='fc8', params=[]):
        data_shape_list = list(data_shape)
        data_shape_list[0] = 1

        label_shape_list = list(label_shape)
        label_shape_list[0] = 1

        self.data_shape = data_shape_list
        self.data = L.DummyData(shape=dict(dim=data_shape_list))
        self.label = L.DummyData(shape=dict(dim=label_shape_list))

        self.last_layer = last_layer
        self.receptiveFieldStride = []  # cumprod of the stride values across the whole net

        self.net = self.net(params=params)

        self.receptiveFieldStride = np.asarray(self.receptiveFieldStride)
        self.receptiveFieldStride = np.cumprod(self.receptiveFieldStride)
        np.save(str.split(params['path2net'], '.')[0] +'_stride.npy', self.receptiveFieldStride)

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
        n.conv1 = L.Convolution(n.data, kernel_size=11, num_output=96, stride=4)
        self.receptiveFieldStride.append(4)
        if self.last_layer == 'conv1':
            self.__network_end(n, n.conv1, params)
            return

        n.relu1 = L.ReLU(n.conv1, in_place=True)
        self.receptiveFieldStride.append(1)
        if self.last_layer == 'relu1':
            self.__network_end(n, n.relu1, params)
            return

        n.pool1 = L.Pooling(n.relu1, pool=P.Pooling.MAX, kernel_size=3, stride=2)
        self.receptiveFieldStride.append(2)
        if self.last_layer == 'pool1':
            self.__network_end(n, n.pool1, params)
            return

        n.norm1 = L.LRN(n.pool1, local_size=5, alpha=1e-4, beta=0.75)
        self.receptiveFieldStride.append(1)
        if self.last_layer == 'norm1':
            self.__network_end(n, n.norm1, params)
            return

        # layer 2
        n.conv2 = L.Convolution(n.norm1, kernel_size=5, num_output=256, pad=2, group=2)
        self.receptiveFieldStride.append(1)
        if self.last_layer == 'conv2':
            self.__network_end(n, n.conv2, params)
            return

        n.relu2 = L.ReLU(n.conv2, in_place=True)
        self.receptiveFieldStride.append(1)
        if self.last_layer == 'relu2':
            self.__network_end(n, n.relu2, params)
            return

        n.pool2 = L.Pooling(n.relu2, pool=P.Pooling.MAX, kernel_size=3, stride=2)
        self.receptiveFieldStride.append(2)
        if self.last_layer == 'pool2':
            self.__network_end(n, n.pool2, params)
            return

        n.norm2 = L.LRN(n.pool2, local_size=5, alpha=1e-4, beta=0.75)
        self.receptiveFieldStride.append(1)
        if self.last_layer == 'norm2':
            self.__network_end(n, n.norm2, params)
            return

        # layer 3
        n.conv3 = L.Convolution(n.norm2, kernel_size=3, num_output=384, pad=1)
        self.receptiveFieldStride.append(1)
        if self.last_layer == 'conv3':
            self.__network_end(n, n.conv3, params)
            return

        n.relu3 = L.ReLU(n.conv3, in_place=True)
        self.receptiveFieldStride.append(1)
        if self.last_layer == 'relu3':
            self.__network_end(n, n.relu3, params)
            return

        # layer 4
        n.conv4 = L.Convolution(n.relu3, kernel_size=3, num_output=384, pad=1, group=2)
        self.receptiveFieldStride.append(1)
        if self.last_layer == 'conv4':
            self.__network_end(n, n.conv4, params)
            return

        n.relu4 = L.ReLU(n.conv4, in_place=True)
        self.receptiveFieldStride.append(1)
        if self.last_layer == 'relu4':
            self.__network_end(n, n.relu4, params)
            return

        # layer 5
        n.conv5 = L.Convolution(n.relu4, kernel_size=3, num_output=256, pad=1, group=2)
        self.receptiveFieldStride.append(1)
        if self.last_layer == 'conv5':
            self.__network_end(n, n.conv5, params)
            return

        n.prelu5 = L.PReLU(n.conv5, in_place=True, param=[dict(decay_mult=0)])
        self.receptiveFieldStride.append(1)
        if self.last_layer == 'prelu5':
            self.__network_end(n, n.prelu5, params)
            return

        n.pool5 = L.Pooling(n.prelu5, pool=P.Pooling.MAX, kernel_size=3, stride=2)
        self.receptiveFieldStride.append(2)
        if self.last_layer == 'pool5':
            self.__network_end(n, n.pool5, params)
            return
    #
# -----------------------------------------------------------------------------
