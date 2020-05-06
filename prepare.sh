#!/bin/bash

conda install ipython
pip install ninja yacs cython matplotlib tqdm opencv-python requests
pip install onnx
pip install onnxruntime

conda install pytorch=1.2.0 cudatoolkit=10.0 -c pytorch-nightly

export INSTALL_DIR=$PWD

# install torchvision
cd $INSTALL_DIR
git clone https://github.com/pytorch/vision.git
cd vision
git checkout v0.4.0
python setup.py develop

# install pycocotools
cd $INSTALL_DIR
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
python setup.py build_ext install

# install apex
cd $INSTALL_DIR
git clone https://github.com/NVIDIA/apex.git
cd apex
git checkout 3ef01faef2492b3e650f44ecc510f3a8f2426783
python setup.py install --cuda_ext --cpp_ext

cd $INSTALL_DIR
git checkout onnx_stage_mrcnn
python setup.py build develop
