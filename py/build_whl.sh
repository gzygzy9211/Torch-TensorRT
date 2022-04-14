#!/bin/bash

# Example usage: docker run -it -v$(pwd)/..:/workspace/TRTorch build_trtorch_wheel /bin/bash /workspace/TRTorch/py/build_whl.sh

cd /workspace/Torch-TensorRT/py

# git checkout ../WORKSPACE
# take original "WORKSPACE" as a template to generate "WORKSPACE.bazel",
#     which will be used to build python package

export CXX=g++
# fill the path of dependency below
export CUDNN_HOME=/path/to/cudnn/
export TENSORRT_HOME=/path/to/tensorrt

build_py36() {
    /opt/python/cp36-cp36m/bin/python -m pip install -r requirements.txt
    /opt/python/cp36-cp36m/bin/python setup.py bdist_wheel --release
    #auditwheel repair --plat manylinux2014_x86_64
}

build_py37() {
    /opt/python/cp37-cp37m/bin/python -m pip install -r requirements.txt
    /opt/python/cp37-cp37m/bin/python setup.py bdist_wheel --release
    #auditwheel repair --plat manylinux2014_x86_64
}

build_py38() {
    /opt/python/cp38-cp38/bin/python -m pip install -r requirements.txt
    /opt/python/cp38-cp38/bin/python setup.py bdist_wheel --release
    #auditwheel repair --plat manylinux2014_x86_64
}

build_py39() {
    /opt/python/cp39-cp39/bin/python -m pip install -r requirements.txt
    /opt/python/cp39-cp39/bin/python setup.py bdist_wheel --release
    #auditwheel repair --plat manylinux2014_x86_64
}

build_py36
build_py37
build_py38
build_py39