import numpy as np

# caffe root
caffe_root = '/export/home/etikhonc/caffe-master/python/'
gpu = True

nModels = 4
model = np.array([None]*nModels)


# refimage_path = '/export/home/etikhonc/workspace/data/OlympicSports/nneighbors/cliqueCNN/alpha_blending/'
refimage_path = './'
refimage_name = 'fc8_output_0018_mean.jpg' # 'anchor_4421.png'


# Model 1: alexnet
# model[0] = dict()
# model[0]['name'] = 'alexnet'
# model[0]['weights'] = '/export/home/etikhonc/caffe-master/models/bvlc_alexnet/bvlc_alexnet.caffemodel'
# model[0]['prototxt'] = '/export/home/etikhonc/caffe-master/models/bvlc_alexnet/deploy.prototxt'
# model[0]['mean'] = '/export/home/etikhonc/caffe-master/data/ilsvrc12/ilsvrc12.npy'
# # model[0]['layers'] = {'pool5': 10}
# model[0]['layers'] = {'conv1': 300, 'conv2': 300, 'conv3': 300, 'conv4': 100, 'conv5': 20, 'pool5': 10, 'fc6': 2, 'fc7': 2, 'fc8': 2}
# # model[0]['layers'] = {'conv1': 300, 'pool1': 300, 'norm1': 300, 'conv2': 300, 'pool2': 300, 'norm2': 300,
# #                       'conv3': 300, 'conv4': 100, 'conv5': 20, 'pool5': 10, 'fc6': 2, 'fc7': 2, 'fc8': 2}
# model[0]['refimage_path'] = refimage_path
# model[0]['refimage_name'] = refimage_name
# model[0]['vis2folder'] = './results_' + model[0]['name'] + '/'


# Model 2: cliqueCNN
# model[1] = dict()
# model[1]['name'] = 'cliqueCNN_long_jump'
# model[1]['weights'] = '/export/home/etikhonc/workspace/nn_visualizations/deep-visualization-toolbox/models/cliqueCNN_OS_long_jump/snap_iter_30000.caffemodel'
# model[1]['prototxt'] = '/export/home/etikhonc/workspace/nn_visualizations/deep-visualization-toolbox/models/cliqueCNN_OS_long_jump/deploy.prototxt'
# model[1]['mean'] = '/export/home/etikhonc/workspace/nn_visualizations/deep-visualization-toolbox/models/cliqueCNN_OS_long_jump/mean_CHW.npy'
# model[1]['layers'] = {'conv1': 300, 'conv2': 300, 'conv3': 300, 'conv4': 100, 'conv5': 20, 'pool5': 10, 'fc6_': 2, 'fc7_': 2, 'fc8_output': 2}
# model[1]['refimage_path'] = refimage_path
# model[1]['refimage_name'] = refimage_name
# model[1]['vis2folder'] = './results_' + model[1]['name'] + '/'
# model[1]['nLabels'] = 304


# Model 3: caffenet initialized with the same image as cliqueCNN
# model[2] = dict()
# model[2]['name'] = 'caffenet'
# model[2]['weights'] = '/export/home/etikhonc/caffe-master/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
# model[2]['prototxt'] = '/export/home/etikhonc/caffe-master/models/bvlc_reference_caffenet/deploy.prototxt'
# model[2]['mean'] = '/export/home/etikhonc/caffe-master/data/ilsvrc12/ilsvrc12.npy'
# model[2]['layers'] = {'conv1': 300, 'conv2': 300, 'conv3': 300, 'conv4': 100, 'conv5': 20, 'pool5': 10}
# # model[2]['refimage_path'] = '/export/home/etikhonc/workspace/nn_visualizations/nnpreimagepy/'
# # model[2]['refimage_name'] = '6-x9gPZrIUI_00527_00696__I00145_class_0145.png'  # '-wmGlrcdXgU_01290_01392__I00013_class_0013.png'  # 'fc8_output_0018_mean.jpg'
# model[2]['refimage_path'] = refimage_path
# model[2]['refimage_name'] = refimage_name
# model[2]['vis2folder'] = './results_' + model[2]['name'] + '/'


# Model 4: Posenet (Oemer, Tobias)
# model[3] = dict()
# model[3]['name'] = 'posenet_oet'
# # model[3]['weights'] = 'PoseNet_307_iter40K.caffemodel'  # 'videoNet_GD_v152_708_iter_30000.caffemodel'
# # model[3]['prototxt'] = 'posenet_deploy.prototxt'  # 'videoNet_GD_deploy.prototxt'
# model[3]['weights'] = 'videoNet_GD_v152_708_iter_30000.caffemodel'
# model[3]['prototxt'] = 'videoNet_GD_deploy.prototxt'
# model[3]['mean'] = '/export/home/etikhonc/caffe-master/data/ilsvrc12/ilsvrc12.npy'
# model[3]['layers'] = {'conv1': 300, 'conv2': 300, 'conv3': 300, 'conv4': 100, 'conv5': 20, 'pool5': 10}
# # model[3]['refimage_path'] = '/export/home/etikhonc/workspace/nn_visualizations/nnpreimagepy/'
# # model[3]['refimage_name'] = 'fc8_output_0018_mean.jpg'  # '-wmGlrcdXgU_01290_01392__I00013_class_0013.png'  # 'fc8_output_0018_mean.jpg'
# model[3]['refimage_path'] = refimage_path
# model[3]['refimage_name'] = refimage_name
# model[3]['vis2folder'] = './results_' + model[3]['name'] + '/'
