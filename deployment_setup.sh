#!/bin/bash

# Function to install CUDA and CUDNN
install_cuda_cudnn() {
    # install cuda: https://gist.github.com/ksopyla/bf74e8ce2683460d8de6e0dc389fc7f5
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
    mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
    add-apt-repository "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"

    # install cuda toolkit
    apt update
    apt install cuda-toolkit-11-6

    # install cudnn
    apt-get install libcudnn8=8.4.0.27-1+cuda11.6

    # add CUDA_HOME to PATH environment
    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64"
    export CUDA_HOME=/usr/local/cuda
    export PATH="/usr/local/cuda/bin:$PATH"
    # export CUDA_HOME=/usr/local/cuda-11.6
    # export PATH=/usr/local/cuda-11.6/bin${PATH:+:${PATH}}
}


# ########### Install required CUDA and CUDNN versions
# check if cuda is installed
if ! command -v "nvcc --version" &> /dev/null ; then
    echo "NVCC not found. Installing CUDA 11.6..."
    install_cuda_cudnn
else
    echo "CUDA exists"
    # check version, delete if not 11.6
    # then: install_cuda_cudnn
fi

# ########### Install python3.6
if ! which python3.6 &>/dev/null; then
    echo "Python 3.6 is not installed."
    add-apt-repository ppa:deadsnakes/ppa
    apt-get update
    apt-get install python3.6
fi

# in ubuntu 20.04 base image, python is undefined, python3 is defined as 3.8
# update python
if ! command -v "python --version" &> /dev/null ; then
    update-alternatives --install /usr/bin/python python /usr/bin/python3.6 1
    update-alternatives --set python /usr/bin/python3.6
fi
# update python3 (making sure not to break apt)
unlink /usr/bin/python3
ln -s /usr/bin/python3.6 /usr/bin/python
# do that with the config as well, but need the specific file name for that...

# do the rest (other apt-get things that need installing, can't remember which...)
