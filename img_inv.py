#!/usr/bin/env python
'''
This code is an implementation of the image inversion algorithm from the paper :

Visualizing deep convolutional neural networks using natural pre-images
Aravindh Mahendran and Andrea Vedaldi, April, 2016

This code is based on the MATLAB (MatConvNet) implementation of the algorithm provided by the Authors:
https://github.com/aravindhm/nnpreimage

Author: Ekaterina Sutter
November 2016

'''

import os
os.environ['GLOG_minloglevel'] = '2'  # suprress Caffe verbose prints

import settings
import site
site.addsitedir(settings.caffe_root)

import numpy as np
import shutil
import os
import random
import PIL.Image
import sys
import matplotlib.pyplot as plt

pycaffe_root = settings.caffe_root # substitute your path here
sys.path.insert(0, pycaffe_root)
import caffe

fc_layers = settings.fc_layers
conv_layers = settings.conv_layers

mean = np.load(settings.model_mean)
mean = mean.squeeze()

if settings.gpu:
  caffe.set_mode_gpu()

from alexnet import AlexNet

# Load reference network which one want to investigate
net = caffe.Classifier(settings.model_definition, settings.model_path, caffe.TEST)
# Define input transformer
transformer = caffe.io.Transformer({'data': (1, 3, 227, 227)})
transformer.set_transpose('data', (2, 0, 1))
transformer.set_channel_swap('data', (2, 1, 0))  # the reference model has channels in BGR order instead of RGB
transformer.set_mean('data', mean.mean(1).mean(1))
# ------------------------------------------------------------------------------------------------------------------


# preprocess image according the networks specification
def preprocess(net, img):
    return transformer.preprocess('data', img)

# deprocess image back in the standard RGB-space
def deprocess(net, img):
    return transformer.deprocess('data', img)
# ------------------------------------------------------------------------------------------------------------------


# reg1 function and its' gradient: bound pixel intensities
def reg_intensity(x, gamma):  # x: CxHxW
    HW = np.prod(x.shape[1:3])
    # sum of squares over all color channels
    n = np.sum(x*x, axis=0)

    # value of the function
    val = np.sum(np.power(n, gamma/2.).flatten()) / x.size

    # gradient
    dx = gamma/float(HW) * x * np.power(n, gamma/2.-1)

    return val, dx


# reg2 function and its' gradient: bounded variation
def reg_tv(x, gamma):  # x: CxHxW
    HW = np.prod(x.shape[1:3])

    d1 = x[:, :, 1:] - x[:, :, 0:-1]
    d1 = np.hstack((d1, np.zeros((x.shape[0],1)))) # add zeros column at the end

    d2 = x[:, 1:, :] - x[:, 0:-1, :]
    d2 = np.vstack((d2, np.zeros((1,x.shape[1])))) # add zeros row at the bottom

    v = np.power(d1*d1 + d2*d2, gamma/2.0)

    # value of the function
    val = sum(v.flatten) / float(HW)

    # gradient
    v_ = np.power(np.clip(v, 1e-6), 1-2/gamma)
    d1_ = v_ * d1
    d2_ = v_ * d2
    d11 = d1_[:, :, 0:-1] - d1_[:, :, 1:]
    d22 = d2_[:, 0:-1, :] - d2_[:, 1:, :]

    d11 = np.hstack((-d1[:, :, 0], d11))
    d22 = np.vstack((-d2[:, 0, :], d22))

    dx = (gamma/float(HW)) * (d11 + d22)

    return val, dx

# ------------------------------------------------------------------------------------------------------------------


# one step of the adaGrad
def make_step(net, Z, delta_xt, acc_sq_grad, const):

    src = net.blobs['data']
    xt = src.data[0]
    energy = (-1)*np.ones((4))

    net.forward()   # propagate the current image to the loss
    net.backward()  # back propagate the gradient

    # gradient of the loss (l2)
    grad_loss = net.blobs['data'].diff[0]
    grad_loss = grad_loss / Z

    # value of the loss (l2)
    energy[0] = net.blobs['loss_l2'].data / Z

    # reg1: bounded range
    # energy[1], grad_reg1 = reg_intensity(xt, const['reg1_alpha'])
    # grad_reg1 *= 1 / const['reg1_C']
    grad_reg1 = np.zeros(xt.shape)

    # reg2: TV
    # energy[2], grad_reg2 = reg_tv(xt, const['reg2_beta'])
    # grad_reg2 *= 1 / const['reg2_C']
    grad_reg2 = np.zeros(xt.shape)

    # sum up the gradient and the loss values
    grad_t = grad_loss  # + grad_reg1 + grad_reg2
    energy[3] = np.sum(energy[0:3])

    # accumulated squared gradient
    acc_sq_grad = const['rho'] * acc_sq_grad + grad_t * grad_t
    # adaptive learning rate
    lr_t = (1/const['lr_0'] + np.sqrt(acc_sq_grad))
    # step in the pixel values
    delta_xt = const['rho'] * delta_xt - lr_t * grad_t
    # update the image
    xt = xt + delta_xt

    # box constratint
    # nxt = np.sqrt(np.sum(xt*xt, axis=0))
    # t = np.stack((np.ones(nxt.shape), const['B_plus']/nxt))
    # W = np.min(np.stack((np.ones(nxt.shape), const['B_plus']/nxt)), axis=0)
    # xt = xt*np.array([W]*xt.shape[0])

    src.data[0] = xt

    return delta_xt, acc_sq_grad, energy
# ------------------------------------------------------------------------------------------------------------------


# main optimization loop
def inversion(net, phi_x0, base_img, octaves, debug=True):

    # factor of the momentum
    Z = np.linalg.norm(phi_x0)  # normalization constant
    print "Obj.val. norm. constant: ", Z

    # if debug save intermediate visualizations
    debug_output = './tmp_inv/'
    if debug:
        if os.path.isdir(debug_output):
            shutil.rmtree(debug_output)
        os.mkdir(debug_output)
    
    # image initialization: prepare base image
    image = preprocess(net, base_img)

    # setup the labels
    net.blobs['label'].data[0] = phi_x0

    # get input dimensions from net
    w = net.blobs['data'].width
    h = net.blobs['data'].height
    
    print "start optimizing with SGD"

    src = net.blobs['data']
    src.reshape(1, 3, h, w)  # resize the network's input image size

    iter = 0
    acc_sq_grad = np.zeros(image.shape, dtype=np.float32)
    delta_xt = np.zeros(image.shape, dtype=np.float32)

    for e, o in enumerate(octaves):

        print "lr_0={}, rho={}".format(o['lr_0'], o['rho'])
        print "----------"

        for i in xrange(o['iter_n']):

            # put the current image in the network
            src.data[0] = image.copy()

            # Jitter
            # if o['jitter']:
            #     tau_x = random.randint(0, o['jitterT'])
            #     tau_y = random.randint(0, o['jitterT'])
            # else:
            #     tau_x = random.randint(0, o['jitterT'])
            #     tau_y = random.randint(0, o['jitterT'])
            # v = np.arange(0, h-o['jitterT'])
            # u = np.arange(0, w-o['jitterT'])
            # src.data[0][:, v, u] = image[:, v+tau_x, u+tau_y]

            # one gradient step
            delta_xt, acc_sq_grad, energy = make_step(net, Z, delta_xt, acc_sq_grad, const=o)

            # print current info
            print "iter: %s\t l2-loss: %.2f\t reg1: %.2f\t reg2: %.2f\t total_energy: %.2f" \
                  % (iter, energy[0], energy[1], energy[2], energy[3])

            # save current images
            image = src.data[0]

            # In debug-mode save intermediate images
            if debug:
                dimage = deprocess(net, image)
                # adjust image contrast if clipping is disabled
                dimage = dimage*(255.0/np.percentile(dimage, 99.98))
                if i % 1 == 0:  # save each iteration
                    save_image(debug_output, "iter_%s.jpg" % str(iter).zfill(4), dimage)

            iter += 1   # Increase iter

        print "octave %d image:" % e
        print "----------"

    # returning the resulting image
    return deprocess(net, image)
    # return image
# ------------------------------------------------------------------------------------------------------------------


# save end image
def save_image(output_folder, filename, img):
    path = "%s/%s.jpg" % (output_folder, filename)
    PIL.Image.fromarray(np.uint8(img)).save(path)
    return path
# ------------------------------------------------------------------------------------------------------------------


def main():
    # Hyperparams for AlexNet
    octaves = [
        {
            'iter_n': 300,          # number of iterations with the following parameters:
            'lr_0': 0.01,          # init learning rate
            'rho': 0.9,             # momentum
            'B': 80,                # normalization constant
            'B_plus': 2*80,         # pixel feasible region [-B_plus, B_plus]
            'jitter': True,         # use jitter
            'jitterT': 4,           # maximal hor/ver translation
            'reg1_alpha': 6,        # see paper for the definition of the reg term
            'reg1_C': 3.8147e-12,   # normalization const 1/(B^alpha)
            'reg2_beta': 2,         # see paper for the definition of the reg term
            'reg2_C': 0.0066        # normalization const 1/(V^beta), V=B/6.5
        }
    ]

    # get original input size of network
    original_w = net.blobs['data'].width
    original_h = net.blobs['data'].height

    # which class to visualize
    layer = str(sys.argv[1])     # layer
    ref_image_path = str(sys.argv[2])
    filename = 'layer_' + layer

    print "----------"
    print "layer: %s\tref_image: %s\tfilename: %s" % (layer, ref_image_path, filename)
    print "----------"

    # the background color of the initial image
    background_color = np.float32([175.0, 175.0, 175.0])  # center of the normal distr
    # generate initial random image
    start_image = np.random.normal(background_color, 8, (original_w, original_h, 3))
    start_image = start_image.astype(np.float32)

    # setup the output path
    output_folder = settings.output_folder + 'img_inv/'
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)

    # get the reference image
    ref_image = np.float32(PIL.Image.open(ref_image_path))
    image = preprocess(net, ref_image)
    net.blobs['data'].data[0] = image.copy()
    acts = net.forward(end=layer)
    phi_x0 = acts[layer][0]     # reference representation

    print 'shape of the reference layer: ', net.blobs[layer].data.shape

    # initialize a new network
    params = {'path2train_net': './models/' + settings.netname + '/test_' + layer + '.prototxt',
              'path2solver': './models/' + settings.netname + '/solver_' + layer + '.prototxt',
              'useGPU': settings.gpu, 'DEVICE_ID': 0}
    if settings.netname == 'caffenet':
        AlexNet(net.blobs['data'].data.shape, net.blobs[layer].data.shape, last_layer=layer, params=params)

    new_net = caffe.Net(params['path2train_net'], settings.model_path, caffe.TEST)

    # if a specific output folder is provided
    if len(sys.argv) == 4:
        output_folder = str(sys.argv[3])

    print "Output dir: %s" % output_folder
    print "-----------"

    # generate class visualization via octavewise gradient ascent
    output_image = inversion(new_net, phi_x0, start_image, octaves, debug=True)

    # save result image
    path = save_image(output_folder, filename, output_image)
    print "Saved to %s" % path

# ------------------------------------------------------------------------------------------------------------------


if __name__ == '__main__':
    main()
