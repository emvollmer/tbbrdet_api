#!/bin/bash

# ########## Connect remotely to NextCloud (rshare)
if rclone listremotes | grep -q "rshare:" ; then
    echo "Rshare identified as remote. Obscuring password..."

    if rclone about rshare: | grep -q "Used:" ; then
        echo "Successful connection to remote rshare."
    else
        echo export RCLONE_CONFIG_RSHARE_PASS=$(rclone obscure $RCLONE_CONFIG_RSHARE_PASS) >> /root/.bashrc
        source /root/.bashrc
        echo "Error in connecting to remote rshare."; sleep 10
        exit 1
    fi
else
    echo "Rshare not identified as (only) remote. Try to solve manually with AI4EOSC documentation."
    sleep 10
    exit 1
fi

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
# no /usr/bin/python3.6-config installed alongside this (don't remember how to do that...)

# ## update python3 (making sure not to break apt) ### probably not necessary, we have python
# unlink /usr/bin/python3
# ln -s /usr/bin/python3.6 /usr/bin/python
# do that with the config as well, but need the specific file name for that...

# get code repository
git clone --recurse-submodules https://github.com/emvollmer/tbbrdet_api.git
# alternatively with ssh key
# git clone --recurse-submodules git@github.com/emvollmer/tbbrdet_api.git