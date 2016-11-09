#!/usr/bin/env python
'''
This code is an implementation of the image inversion algorithm from the paper :

Visualizing deep convolutional neural networks using natural pre-images
Aravindh Mahendran and Andrea Vedaldi, April, 2016

This code is based on the MATLAB (MatConvNet) implementation of the algorithm provided by the Authors:
https://github.com/aravindhm/nnpreimage

The code was developed mostly for the own usage

Author: Ekaterina Sutter
November 2016

'''

import os
os.environ['GLOG_minloglevel'] = '2'  # suprress Caffe verbose prints

import matplotlib.pyplot as plt

import settings
import site
site.addsitedir(settings.caffe_root)

import math
import numpy as np
import PIL.Image
import random
from skimage.restoration import denoise_tv_bregman
import scipy.stats
import shutil
import sys

# setuo caffe
pycaffe_root = settings.caffe_root # substitute your path here
sys.path.insert(0, pycaffe_root)
import caffe

if settings.gpu:
    caffe.set_mode_gpu()
    caffe.set_device(0)
# caffe.set_mode_cpu()

# load models
from alexnet import AlexNet
from cliquecnn import CliqueCNN
from posenet import PoseNet


# define input transformer
transformer = caffe.io.Transformer({'data': (1, 3, 227, 227)})
transformer.set_transpose('data', (2, 0, 1))
transformer.set_channel_swap('data', (2, 1, 0))  # the reference model has channels in BGR order instead of RGB
transformer.set_raw_scale('data', 1.0)
# ------------------------------------------------------------------------------------------------------------------
np.random.seed([1000])
random.seed(1000)
# ------------------------------------------------------------------------------------------------------------------


# reg1 function and its' gradient: bound pixel intensities
def reg_intensity(x, gamma):  # x: CxHxW
    HW = np.prod(x.shape[1:3])
    # sum of squares over all color channels
    n = np.sum(x*x, axis=0)

    # value of the function
    val = np.sum(np.power(n, gamma/2.).flatten()) / HW

    # gradient
    dx = gamma/float(HW) * x * np.power(n, gamma/2.-1)

    return val, dx


# reg2 function and its' gradient: bounded variation
def reg_tv(x, gamma):  # x: CxHxW

    HW = np.prod(x.shape[1:3])

    d1 = x[:, :, 1:] - x[:, :, 0:-1]
    d1 = np.pad(d1, ((0,0), (0,0), (0,1)), mode='constant', constant_values=0)  # add zeros column at the end

    d2 = x[:, 1:, :] - x[:, 0:-1, :]
    d2 = np.pad(d2, ((0,0), (0,1), (0,0)), mode='constant', constant_values=0)  # add zeros row at the bottom

    v = np.power(d1*d1 + d2*d2, gamma/2.0)

    # value of the function
    val = sum(v.flatten()) / float(HW)

    # gradient
    v_ = np.power(np.clip(v, 1e-6, np.inf), 1-2/gamma)
    d1_ = v_ * d1
    d2_ = v_ * d2
    d11 = d1_[:, :, 0:-1] - d1_[:, :, 1:]
    d22 = d2_[:, 0:-1, :] - d2_[:, 1:, :]

    d11 = np.pad(d11, ((0,0), (0,0), (1,0)), mode='constant', constant_values=0)
    d11[:, :, 0] = -d1[:, :, 0]

    d22 = np.pad(d22, ((0, 0), (1, 0), (0, 0)), mode='constant', constant_values=0)
    d22[:, 0, :] = -d2[:, 0, :]

    dx = (gamma/float(HW)) * (d11 + d22)

    return val, dx
# ------------------------------------------------------------------------------------------------------------------


def get_initial_image(h=227, w=227, sigma=60, padding=0):

    # generate initial random image
    image = 2*np.random.random((3, h+padding, w+padding)) - 1
    # from the initial code to the paper
    m = scipy.stats.mstats.mquantiles(image.flatten(), 0.95, alphap=0.5, betap=0.5)
    image = image/m*sigma/np.sqrt(3)

    return image

# ------------------------------------------------------------------------------------------------------------------


# one step of the adaGrad
def grad_step(net, Z, xt, delta_xt, acc_sq_grad, const):

    src = net.blobs['data']
    energy = np.ones((4))

    h = src.height
    w = src.width
    HW = h*w

    # Jitter
    tau_x = random.randint(0, const['jitterT'])
    tau_y = random.randint(0, const['jitterT'])
    xt_crop = xt[:, tau_x:tau_x+h, tau_y:tau_y+w]

    # l2-loss
    src.data[0] = xt_crop.copy()
    fw = net.forward()   # propagate the current image to the loss
    bw = net.backward(start='loss_l2')  # back propagate the gradient

    energy[0] = const['C']*net.blobs['loss_l2'].data / Z
    energy[0] *= 2*net.blobs['data'].data.shape[0]  # because of the normalization constant 1/2N in the loss function
    grad_loss = np.zeros(xt.shape)
    grad_loss[:, tau_x:tau_x + h, tau_y:tau_y + w] = net.blobs['data'].diff[0]
    grad_loss[:, tau_x:tau_x + h, tau_y:tau_y + w] = const['C'] * net.blobs['data'].diff[0] / Z * 2*net.blobs['data'].data.shape[0]
    # grad_loss = np.zeros(xt.shape)

    # reg1: bounded range
    energy[1], grad_reg1 = reg_intensity(xt, const['reg1_alpha'])
    energy[1] *= const['reg1_C']
    grad_reg1 *= const['reg1_C']
    # grad_reg1 = np.zeros(xt.shape)

    # reg2: TV
    energy[2], grad_reg2 = reg_tv(xt, const['reg2_beta'])
    energy[2] *= const['reg2_C']
    grad_reg2 *= const['reg2_C']
    # grad_reg2 = np.zeros(xt.shape)

    # sum up the gradient and the loss values
    grad_t = grad_loss + grad_reg1 + grad_reg2
    energy[3] = np.sum(energy[0:3])

    # accumulated squared gradient
    acc_sq_grad = const['rho'] * acc_sq_grad + grad_t * grad_t
    # adaptive learning rate
    lr_t = 1/(1/(HW*const['lr_0']) + np.sqrt(acc_sq_grad))

    # step in the pixel values
    delta_xt = const['rho'] * delta_xt - lr_t * grad_t

    return delta_xt, acc_sq_grad, energy
# ------------------------------------------------------------------------------------------------------------------


# main optimization loop
def inversion(net, phi_x0, octaves, debug=True):

    # factor of the momentum
    Z = np.sum((phi_x0*phi_x0).flatten())  # normalization constant
    print "Obj.val. norm. constant: ", 1/Z

    # if debug save intermediate visualizations
    debug_output = './tmp_inv/'
    if debug:
        if os.path.isdir(debug_output):
            shutil.rmtree(debug_output)
        os.mkdir(debug_output)

    # setup the labels
    net.blobs['label'].data[0] = phi_x0

    iter = 0
    print "start optimizing with SGD"
    for e, o in enumerate(octaves):

        print_opt_params(o)

        w = net.blobs['data'].width
        h = net.blobs['data'].height

        # start images:
        if e == 0:
            # generate initial image
            image = get_initial_image(h=h, w=w, sigma=o['B'], padding=o['jitterT'])
        else:
            # use the image produced by the prev block of iterations
            tau = int(math.floor(o['jitterT']/2))
            tmp_image = np.zeros((3, h, w))
            tmp_image[:, tau:tau+h, tau:tau+w] = transformer.preprocess('data', image)
            image = tmp_image.copy()
            del tmp_image

        acc_sq_grad = np.zeros(image.shape, dtype=np.float32)
        delta = np.zeros(image.shape, dtype=np.float32)
        energy = np.zeros((o['iter_n'], 4))

        for i in xrange(o['iter_n']):

            # Jitter
            tau_x = random.randint(0, o['jitterT'])
            tau_y = random.randint(0, o['jitterT'])
            image_crop = image[:, tau_x:tau_x + h, tau_y:tau_y + w]

            # one gradient step
            delta, acc_sq_grad, energy[i,:] = grad_step(net, Z, image, delta, acc_sq_grad, const=o)

            # print current info
            print "iter: %05s\t l2-loss: %.5f\t reg1: %.5f\t reg2: %.5f\t total_energy: %.5f" \
                  % (iter, energy[i,0], energy[i,1], energy[i,2], energy[i,3])

            # save current images
            image = image + delta

            # box constraint
            image_all_colors = np.sqrt(np.sum(image*image, axis=0))
            W = np.min(np.stack((np.ones(image_all_colors.shape), o['B_plus']/image_all_colors)), axis=0)
            image = image*np.array([W]*image.shape[0])

            # # In debug-mode save intermediate images
            # if debug:
            #     dimage = transformer.deprocess('data', image)
            #     # adjust image contrast if clipping is disabled
            #     dimage = dimage*(255.0/np.percentile(dimage, 99.98))
            #     if i % 1 == 0:  # save each iteration
            #         save_image(debug_output, "iter_%s.jpg" % str(iter).zfill(4), dimage)

            iter += 1   # Increase iter

        print "----------"
        # crop image of the initial network size (because of jitter)
        tau = int(math.floor(o['jitterT']/2))
        image = image[:, tau:tau+h, tau:tau+w]
        image = transformer.deprocess('data', image)

        # _, ax1 = plt.subplots()
        # ax2 = ax1.twinx()
        # ax1.plot(np.arange(o['iter_n']), energy[:, 3].flatten(), 'b')
        # ax2.plot(np.arange(o['iter_n']), energy[:, 0].flatten(), 'r')
        # ax1.set_xlabel('iteration')
        # ax1.set_ylabel('sum loss')
        # ax2.set_ylabel('l2 loss')
        # plt.show()

    #
    # returning the resulting image
    # image = transformer.deprocess('data', image)
    return image
# ------------------------------------------------------------------------------------------------------------------


def print_opt_params(params):
    print "----------"
    print 'number of iterations: ', params['iter_n']
    print 'init learning rate: ', params['lr_0']
    print 'momentum: ', params['rho']
    print 'pixel value const B: ', params['B']
    print 'box constraint: |reconstr|<=2B=', params['B_plus']
    print 'jitter: ', params['jitterT']
    print 'bounded range alpha: ', params['reg1_alpha']
    print 'bounded range norm.const: ', params['reg1_C']
    print 'Total Variation (TV) beta: ', params['reg2_beta']
    print 'Total Variation (TV) norm.const: ', params['reg2_C']
    print "----------"


# save end image
def save_image(output_folder, filename, img):
    path = "%s/%s.jpg" % (output_folder, filename)
    PIL.Image.fromarray(np.uint8(img)).save(path)
    return path
# ------------------------------------------------------------------------------------------------------------------


def main():
    # Hyperparams for AlexNet
    B = 80
    octaves = [
        {
            'iter_n': 300,          # number of iterations with the following parameters:
            'lr_0': 53.3333,        # init learning rate: 0.05*B^2/alpha
            'rho': 0.9,             # momentum
            'C': 1,                 # weights of the l2 term in the objective
            'B': B,                # normalization constant
            'B_plus': 2*B,         # pixel feasible region [-B_plus, B_plus]
            'jitterT': 1,           # maximal hor/ver translation !!! depends on the layer strides
            'reg1_alpha': 6,            # see paper for the definition of the reg term
            'reg1_C': 1/math.pow(B,6),   # normalization const 1/(B^alpha)
            'reg2_beta': 2,              # see paper for the definition of the reg term
            'reg2_C': 1/math.pow(B/6.5,2), # normalization const 1/(V^beta), V=B/6.5
            'start_denoise_weight': 0.0001,
            'end_denoise_weight': 0.005
        },
        {
            'iter_n': 50,           # number of iterations with the following parameters:
            'lr_0': 5.33333,        # init learning rate: 0.05*B^2/alpha
            'rho': 0.9,             # momentum
            'C': 1,  # weights of the l2 term in the objective
            'B': B,                # normalization constant
            'B_plus': 2*B,         # pixel feasible region [-B_plus, B_plus]
            'jitterT': 0,           # maximal hor/ver translation !!! depends on the layer strides
            'reg1_alpha': 6,            # see paper for the definition of the reg term
            'reg1_C': 1/math.pow(B,6),   # normalization const 1/(B^alpha)
            'reg2_beta': 2,              # see paper for the definition of the reg term
            'reg2_C': 1/math.pow(B/6.5,2), # normalization const 1/(V^beta), V=B/6.5
        }
    ]

    for m in range(settings.nModels):

        if settings.model[m] is None:
            continue

        # =============== MODEL m ====================================

        # models means
        mean = np.load(settings.model[m]['mean'])
        mean = mean.squeeze()
        transformer.set_mean('data', mean.mean(1).mean(1))

        # Load reference network which one want to investigate
        net = caffe.Classifier(settings.model[m]['prototxt'], settings.model[m]['weights'], caffe.TEST)

        print net.blobs.keys()

        # get original input size of network
        original_w = net.blobs['data'].width
        original_h = net.blobs['data'].height

        # setup the output path
        if not os.path.isdir(settings.model[m]['vis2folder']):
            os.mkdir(settings.model[m]['vis2folder'])

        output_folder = settings.model[m]['vis2folder'] + '/img_inv_' + \
                        os.path.splitext(settings.model[m]['refimage_name'])[0] + '/'
        if not os.path.isdir(output_folder):
            os.mkdir(output_folder)

        # which class to visualize
        layers = settings.model[m]['layers']
        for layer in layers:

            filename = 'layer_' + layer
            refimage_path = settings.model[m]['refimage_path'] + settings.model[m]['refimage_name']

            print "----------"
            print "layer: %s\tref_image: %s\tfilename: %s" % (layer, refimage_path, filename)
            print "----------"

            # if a specific output folder is provided
            if len(sys.argv) == 4:
                output_folder = str(sys.argv[3])

            print "Output dir: %s" % output_folder
            print "-----------"

            if os.path.isfile("%s/%s.jpg" % (output_folder, filename)):
                print 'Inversion is already computed. Skipping the layer...'
                continue

            # get the reference image
            ref_image = np.float32(PIL.Image.open(refimage_path))
            image = transformer.preprocess('data', ref_image)
            net.blobs['data'].data[0] = image.copy()
            acts = net.forward(end=layer)
            phi_x0 = acts[layer][0]     # reference representation

            print 'shape of the reference layer: ', phi_x0.shape

            if not os.path.isdir('./models/' + settings.model[m]['name']):
                os.mkdir('./models/' + settings.model[m]['name'])

            # initialize a new network
            params = {'path2net': os.getcwd() + '/models/' + settings.model[m]['name'] + '/test_' + layer + '.prototxt',
                      'path2solver': os.getcwd() + '/models/' + settings.model[m]['name'] + '/solver_' + layer + '.prototxt',
                      'useGPU': settings.gpu, 'DEVICE_ID': 0}

            if not os.path.isfile(params['path2net']):
                # caffenet
                if settings.model[m]['name'] == 'alexnet':
                    AlexNet(net.blobs['data'].data.shape, net.blobs[layer].data.shape, last_layer=layer, params=params)
                # cliqueCNN
                if settings.model[m]['name'] == 'cliqueCNN_long_jump':
                    CliqueCNN(net.blobs['data'].data.shape, net.blobs[layer].data.shape,
                              num_classes=settings.model[m]['nLabels'], last_layer=layer, params=params)
                # posenet
                if settings.model[m]['name'] == 'posenet_oet':
                    PoseNet(net.blobs['data'].data.shape, net.blobs[layer].data.shape, last_layer=layer, params=params)

            new_net = caffe.Net(params['path2net'], settings.model[m]['weights'], caffe.TEST)

            # !!!!! Adaptive jitter range
            receptiveFieldStride = np.load(str.split(params['path2net'], '.')[0] +'_stride.npy')
            octaves[0]['jitterT'] = np.max([1, int(round(receptiveFieldStride[-1]/4))]) - 1
            # octaves[0]['jitterT'] = 0

            assert new_net.blobs['data'].data.shape[2] == original_h
            assert new_net.blobs['data'].data.shape[3] == original_w

            # generate class visualization via octavewise gradient ascent
            output_image = inversion(new_net, phi_x0, octaves, debug=False)
            # normalize image = vl_imsc
            output_image = output_image - output_image.min()
            output_image = output_image/output_image.max()
            output_image = 255*np.clip(output_image, 0, 1)

            # save result image
            path = save_image(output_folder, filename, output_image)
            print "Saved to %s" % path

        print '----------------------------------------------------------------------------------------------------'

# ------------------------------------------------------------------------------------------------------------------


if __name__ == '__main__':
    main()
