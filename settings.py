# caffe root
caffe_root = "/export/home/etikhonc/caffe-master/python/"
gpu = True

# caffenet
model_path = "/export/home/etikhonc/caffe-master/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel"
model_definition = '/export/home/etikhonc/caffe-master/models/bvlc_reference_caffenet/deploy.prototxt'
model_mean = '/export/home/etikhonc/caffe-master/data/ilsvrc12/ilsvrc12.npy'
fc_layers = ["fc6", "fc7", "fc8"]
conv_layers = ["conv1", "conv2", "conv3", "conv4", "conv5"]
netname = 'caffenet'

# OS-long_jump
# model_path = "/export/home/etikhonc/workspace/nn_visualizations/deep-visualization-toolbox/models/cliqueCNN_OS_long_jump/snap_iter_30000.caffemodel"
# model_definition='/export/home/etikhonc/workspace/nn_visualizations/deep-visualization-toolbox/models/cliqueCNN_OS_long_jump/deploy.prototxt'
# model_mean='/export/home/etikhonc/workspace/nn_visualizations/deep-visualization-toolbox/models/cliqueCNN_OS_long_jump/mean_CHW.npy'
# fc_layers = ["fc6_", "fc7_", "fc8_output"]
# conv_layers = ["conv1", "conv2", "conv3", "conv4", "conv5"]
# netname = 'os_long_jump'

output_folder = './results_' + netname + '/'