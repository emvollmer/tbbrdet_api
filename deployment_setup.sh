#!/bin/bash

# Function to install CUDA and CUDNN
install_cuda_cudnn() {
    # install cuda: https://gist.github.com/ksopyla/bf74e8ce2683460d8de6e0dc389fc7f5
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
    mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub   # new key, added 2022-04-25 22:52
    add-apt-repository "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"

    # install cuda toolkit
    apt update
    yes | apt install cuda-toolkit-11-6

    # install cudnn
    apt-get install libcudnn8=8.4.0.27-1+cuda11.6

    # add CUDA_HOME to PATH environment
    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64"
    export CUDA_HOME=/usr/local/cuda
    export PATH="/usr/local/cuda/bin:$PATH"

    # check correct cuda location
    if which nvcc | grep -q "/usr/local/cuda/" ; then
        echo "CUDA and CUDNN installation successful."
    else
        echo "CUDA and CUDNN installation failed. Please check manually."; sleep 10
        exit 1
    fi
}


# ########### Install required CUDA and CUDNN versions
# check if cuda is installed
required_cuda="11.6"
if ! command -v "nvcc --version" &> /dev/null ; then
    echo "NVCC not found. Installing CUDA $required_cuda..."
    install_cuda_cudnn
else
    current_cuda=$(nvcc --version | grep 'release' | awk '{print $5}' | cut -d ',' -f 1)
    if [[ "$required_cuda" == "$current_cuda" ]]; then
        echo "CUDA exists, with CUDA version requirement $required_cuda satisfied."
    else
        echo "CUDA exists, but CUDA version $current_cuda is not the required $required_cuda.
        Installing $required_cuda..."
        install_cuda_cudnn
    fi
fi

# ########### Install python3.6
if ! which python3.6 &> /dev/null; then
    echo "Python 3.6 is not installed."
    yes | add-apt-repository ppa:deadsnakes/ppa
    apt-get update
    yes | apt-get install python3.6
fi

# in ubuntu 20.04 base image, python is undefined, python3 is defined as 3.8
# update python
if ! command -v "python --version" &> /dev/null ; then
    update-alternatives --install /usr/bin/python python /usr/bin/python3.6 1
    update-alternatives --set python /usr/bin/python3.6
fi

# ########### Install zstd (required to extract data)
if ! which zstd &> /dev/null; then
    echo "zstd is not installed."
    apt install zstd
fi

# get code repository
git clone --recurse-submodules https://github.com/emvollmer/tbbrdet_api.git

cd tbbrdet_api
git pull --recurse-submodules
git submodule update --remote --recursive

echo "==================================="
echo "Initial deployment setup complete."
echo "==================================="