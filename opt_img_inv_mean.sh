#!/bin/bash
path2mean="/export/home/etikhonc/workspace/nn_visualizations/mfv/"
N_ARGS=1
# check the number of input parameters
if [ "$#" -ne "${N_ARGS}" ]; then
  echo "Missing ${N_ARGS} args."
  exit 1
fi
# layer name
layer=${1} #'fc8_output'
# spatial position
xy=0
# random seed

#  mean_file=${path2mean}fc8_output_${idx}_mean.jpg
mean_file=red-fox.jpg

# run python script
python ./img_inv.py ${layer} ${mean_file}
