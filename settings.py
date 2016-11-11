import numpy as np

# caffe root
# caffe_root = '/export/home/etikhonc/caffe-master/python/'
caffe_root = '/export/home/bbrattol/caffe/python/'
gpu = True

nModels = 6
model = np.array([None]*nModels)

# refimage_path = '/export/home/etikhonc/workspace/data/OlympicSports/nneighbors/cliqueCNN/alpha_blending/'
refimage_path = './'
refimage_name = '6-x9gPZrIUI_00527_00696__I00145_class_0145.png'


# Model 1: alexnet
model[0] = dict()
model[0]['name'] = 'alexnet'
model[0]['weights'] = '/export/home/etikhonc/caffe-master/models/bvlc_alexnet/bvlc_alexnet.caffemodel'
model[0]['prototxt'] = '/export/home/etikhonc/caffe-master/models/bvlc_alexnet/deploy.prototxt'
model[0]['mean'] = '/export/home/etikhonc/caffe-master/data/ilsvrc12/ilsvrc12.npy'
# model[0]['layers'] = {'pool5': 10}
model[0]['layers'] = {'conv1': 300, 'conv2': 300, 'conv3': 300, 'conv4': 100, 'conv5': 20, 'pool5': 10, 'fc6': 2, 'fc7': 2, 'fc8': 2}
# model[0]['layers'] = {'conv1': 300, 'pool1': 300, 'norm1': 300, 'conv2': 300, 'pool2': 300, 'norm2': 300,
#                       'conv3': 300, 'conv4': 100, 'conv5': 20, 'pool5': 10, 'fc6': 2, 'fc7': 2, 'fc8': 2}
# model[0]['refimage_path'] = refimage_path
# model[0]['refimage_name'] = refimage_name
model[0]['vis2folder'] = './results_' + model[0]['name'] + '/'


# Model 2: cliqueCNN
model[1] = dict()
model[1]['name'] = 'cliqueCNN_long_jump'
model[1]['weights'] = '/export/home/etikhonc/workspace/nn_visualizations/deep-visualization-toolbox/models/cliqueCNN_OS_long_jump/snap_iter_30000.caffemodel'
model[1]['prototxt'] = '/export/home/etikhonc/workspace/nn_visualizations/deep-visualization-toolbox/models/cliqueCNN_OS_long_jump/deploy.prototxt'
model[1]['mean'] = '/export/home/etikhonc/workspace/nn_visualizations/deep-visualization-toolbox/models/cliqueCNN_OS_long_jump/mean_CHW.npy'
model[1]['layers'] = {'conv1': 300, 'conv2': 300, 'conv3': 300, 'conv4': 100, 'conv5': 20, 'pool5': 10, 'fc6_': 2, 'fc7_': 2, 'fc8_output': 2}
model[1]['refimage_path'] = refimage_path
model[1]['refimage_name'] = refimage_name
model[1]['vis2folder'] = './results_' + model[1]['name'] + '/'
model[1]['nLabels'] = 304


# Model 3: caffenet initialized with the same image as cliqueCNN
model[2] = dict()
model[2]['name'] = 'caffenet'
model[2]['weights'] = '/export/home/etikhonc/caffe-master/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
model[2]['prototxt'] = '/export/home/etikhonc/caffe-master/models/bvlc_reference_caffenet/deploy.prototxt'
model[2]['mean'] = '/export/home/etikhonc/caffe-master/data/ilsvrc12/ilsvrc12.npy'
model[2]['layers'] = {'conv1': 300, 'conv2': 300, 'conv3': 300, 'conv4': 100, 'conv5': 20, 'pool5': 10}
model[2]['layers_list'] = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'pool5']
# model[2]['refimage_path'] = refimage_path
# model[2]['refimage_name'] = refimage_name
model[2]['vis2folder'] = './results_' + model[2]['name'] + '/'


# Model 4: Posenet
model[3] = dict()
model[3]['name'] = 'posenet'
model[3]['weights'] = './models/videoPosenet/PoseNet_307_iter40K.caffemodel'
model[3]['prototxt'] = './models/videoPosenet/posenet_deploy.prototxt'
model[3]['mean'] = '/export/home/etikhonc/caffe-master/data/ilsvrc12/ilsvrc12.npy'
model[3]['layers'] = {'conv1': 300, 'conv2': 300, 'conv3': 300, 'conv4': 100, 'conv5': 20, 'pool5': 10}
# model[3]['refimage_path'] = refimage_path
# model[3]['refimage_name'] = refimage_name
model[3]['vis2folder'] = './results_' + model[3]['name'] + '/'

# Model 5: videoNet
model[4] = dict()
model[4]['name'] = 'videonet'
model[4]['weights'] = './models/videoNet_GD/videoNet_GD_v152_702_iter_40000.caffemodel'
model[4]['prototxt'] = './models/videoNet_GD/videoNet_GD_deploy.prototxt'
model[4]['mean'] = '/export/home/etikhonc/caffe-master/data/ilsvrc12/ilsvrc12.npy'
model[4]['layers'] = {'conv1': 300, 'conv2': 300, 'conv3': 300, 'conv4': 100, 'conv5': 20, 'pool5': 10}
# model[4]['refimage_path'] = refimage_path
# model[4]['refimage_name'] = refimage_name
model[4]['vis2folder'] = './results_' + model[4]['name'] + '/'


# Model 6: CNN-LSTM
model[5] = dict()
model[5]['name'] = 'cnn_lstm'
model[5]['weights'] = './models/CNN_LSTM/PredictSorting_iter_20000.caffemodel'
model[5]['prototxt'] = './models/CNN_LSTM/deploy_LSTMAndCNN.prototxt'
model[5]['mean'] = '/export/home/etikhonc/caffe-master/data/ilsvrc12/ilsvrc12.npy'
model[5]['layers'] = {'conv1': 300, 'conv2': 300, 'conv3': 300, 'conv4': 100, 'conv5': 20, 'pool5': 10, 'fc6': 2, 'fc7': 2}
# model[5]['refimage_path'] = refimage_path
# model[5]['refimage_name'] = refimage_name
model[5]['vis2folder'] = './results_' + model[5]['name'] + '/'