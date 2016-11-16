import numpy as np
from collections import namedtuple

# caffe root
# caffe_root = '/export/home/etikhonc/caffe-master/python/'
caffe_root = '/export/home/bbrattol/caffe/python/'
gpu = True

nModels = 3
model = np.array([None]*nModels)
layer = namedtuple("layer", ["name", "C"])

# refimage_path = '/export/home/etikhonc/workspace/data/OlympicSports/nneighbors/cliqueCNN/alpha_blending/'
refimage_path = './'
refimage_name = '6-x9gPZrIUI_00527_00696__I00145_class_0145.png'


# Model 1: alexnet
model[0] = dict()
model[0]['name'] = 'alexnet'
model[0]['weights'] = '/export/home/etikhonc/caffe-master/models/bvlc_alexnet/bvlc_alexnet.caffemodel'
model[0]['prototxt'] = '/export/home/etikhonc/caffe-master/models/bvlc_alexnet/deploy.prototxt'
model[0]['mean'] = '/export/home/etikhonc/caffe-master/data/ilsvrc12/ilsvrc12.npy'
model[0]['layers'] = [layer('conv1', 300), layer('conv2', 300), layer('conv3', 300), layer('conv4', 100), layer('conv5', 20),
                      layer('pool5', 10), layer('fc6', 2), layer('fc7', 2)]
# model[0]['refimage_path'] = refimage_path
# model[0]['refimage_name'] = refimage_name
model[0]['vis2folder'] = './results_' + model[0]['name'] + '/'


# Model 2: cliqueCNN
model[1] = dict()
model[1]['name'] = 'cliqueCNN_long_jump'
# model[1]['weights'] = '/export/home/etikhonc/workspace/nn_visualizations/deep-visualization-toolbox/models/cliqueCNN_OS_long_jump/snap_iter_30000.caffemodel'
# model[1]['prototxt'] = '/export/home/etikhonc/workspace/nn_visualizations/deep-visualization-toolbox/models/cliqueCNN_OS_long_jump/deploy.prototxt'
model[1]['weights'] = '/export/home/mbautist/Desktop/workspace/cnn_similarities/NIPS2016/snapshots/long_jump/snap_iter_30000.caffemodel'
model[1]['prototxt'] = '/export/home/etikhonc/workspace/data/NNModels/cliqueCNN_OS_long_jump/deploy_2.prototxt'
model[1]['mean'] = '/export/home/etikhonc/workspace/nn_visualizations/deep-visualization-toolbox/models/cliqueCNN_OS_long_jump/mean_CHW.npy'
# model[1]['layers'] = [layer('conv1', 300), layer('conv2', 300), layer('conv3', 300), layer('conv4', 100), layer('conv5', 20),
#                       layer('pool5', 10), layer('fc6_', 2), layer('fc7_', 2), layer('fc8_output', 2)]
model[1]['layers'] = [layer('conv1', 300), layer('conv2', 300), layer('conv3', 300), layer('conv4', 100), layer('conv5', 20),
                      layer('pool5', 10), layer('fc6', 2), layer('fc7_', 2)]
# model[1]['refimage_path'] = refimage_path
# model[1]['refimage_name'] = refimage_name
model[1]['vis2folder'] = './results_' + model[1]['name'] + '/'
model[1]['nLabels'] = 468


# Model 3: caffenet initialized with the same image as cliqueCNN
# model[2] = dict()
# model[2]['name'] = 'caffenet'
# model[2]['weights'] = '/export/home/etikhonc/caffe-master/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
# model[2]['prototxt'] = '/export/home/etikhonc/caffe-master/models/bvlc_reference_caffenet/deploy.prototxt'
# model[2]['mean'] = '/export/home/etikhonc/caffe-master/data/ilsvrc12/ilsvrc12.npy'
# model[2]['layers'] = [layer('conv1', 300), layer('conv2', 300), layer('conv3', 300), layer('conv4', 100), layer('conv5', 20),
#                       layer('pool5', 10)]
# # model[2]['refimage_path'] = refimage_path
# # model[2]['refimage_name'] = refimage_name
# model[2]['vis2folder'] = './results_' + model[2]['name'] + '/'


# Model 4: Posenet
# model[3] = dict()
# model[3]['name'] = 'posenet'
# model[3]['weights'] = '/export/home/etikhonc/workspace/data/NNModels/videoPosenet/PoseNet_307_iter40K.caffemodel'
# model[3]['prototxt'] = '/export/home/etikhonc/workspace/data/NNModels/videoPosenet/posenet_deploy.prototxt'
# model[3]['mean'] = '/export/home/etikhonc/caffe-master/data/ilsvrc12/ilsvrc12.npy'
# model[3]['layers'] = [layer('conv1', 300), layer('conv2', 300), layer('conv3', 300), layer('conv4', 100), layer('conv5', 20),
#                       layer('pool5', 10)]
# # model[3]['refimage_path'] = refimage_path
# # model[3]['refimage_name'] = refimage_name
# model[3]['vis2folder'] = './results_' + model[3]['name'] + '/'

# Model 5: videoNet
# model[4] = dict()
# model[4]['name'] = 'videonet'
# model[4]['weights'] = '/export/home/etikhonc/workspace/data/NNModels/videoNet_GD/videoNet_GD_v152_702_iter_40000.caffemodel'
# model[4]['prototxt'] = '/export/home/etikhonc/workspace/data/NNModels/videoNet_GD/videoNet_GD_deploy.prototxt'
# model[4]['mean'] = '/export/home/etikhonc/caffe-master/data/ilsvrc12/ilsvrc12.npy'
# model[4]['layers'] = [layer('conv1', 300), layer('conv2', 300), layer('conv3', 300), layer('conv4', 100), layer('conv5', 20),
#                       layer('pool5', 10)]
# # model[4]['refimage_path'] = refimage_path
# # model[4]['refimage_name'] = refimage_name
# model[4]['vis2folder'] = './results_' + model[4]['name'] + '/'


# Model 6: CNN-LSTM
model[2] = dict()
model[2]['name'] = 'cnn_lstm'
model[2]['weights'] = '/export/home/etikhonc/workspace/data/NNModels/CNN_LSTM/PredictSorting_iter_20000.caffemodel'
model[2]['prototxt'] = '/export/home/etikhonc/workspace/data/NNModels/CNN_LSTM/deploy_LSTMAndCNN.prototxt'
model[2]['mean'] = '/export/home/etikhonc/caffe-master/data/ilsvrc12/ilsvrc12.npy'
model[2]['layers'] = [layer('conv1', 300), layer('conv2', 300), layer('conv3', 300), layer('conv4', 100), layer('conv5', 20),
                      layer('pool5', 10), layer('fc6', 2), layer('fc7', 2)]
model[2]['vis2folder'] = './results_' + model[2]['name'] + '/'
