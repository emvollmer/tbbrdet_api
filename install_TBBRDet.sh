#!/bin/bash

# Check for activated venv %%%%%%%%%%%%%%%%%%%% activated venv not necessary for function in image
#if [[ "$VIRTUAL_ENV" != "" ]]; then
#   echo "Currently in an activated virtual environment. Setup may proceed."
#else
#   echo "Currently not in an activated virtual environment. Must set one up to proceed with package installation..."
#   read -p "Please provide the directory of a venv you want to activate and use: " venv_dir
#   venv_pth="$venv_dir"/bin/activate
#   if test -f "$venv_pth"; then
#      echo "'$venv_pth' exists and is a venv. Activating..."
#      source "$venv_pth"
#   else
#      echo "'$venv_pth' does not exist. Please create a venv to use and activate it or do so by rerunning this script."; sleep 5
#      exit
#   fi
#fi


check_correct_version () {
  local what=$1
  local required_ver=$2
  local current_ver=$3

  if [[ "$required_ver" == "$current_ver" ]]; then
    echo "$what version requirement $required_ver satisfied."
  else
    echo "Provided $what version $current_ver is not compatible with required ${required_ver}).
    Please install $what $required_ver "; sleep 5
    exit 1
  fi
}


# Check Python version
current_python=$(python --version 2>&1 | cut -d ' ' -f 2 | cut -d '.' -f 1,2)
required_python="3.6"
check_correct_version "Python" $required_python "$current_python"

# Check CUDA version
current_cuda=$(nvcc --version | grep 'release' | awk '{print $5}' | cut -d ',' -f 1)
required_cuda="11.6"
check_correct_version "CUDA" $required_cuda "$current_cuda"

# Ensure packages are installed in the correct order
# Make sure pip is up to date
pip3 install --upgrade pip setuptools wheel

# Install correct torch version
pip3 install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
# Install openmim using pip
pip3 install -U openmim
# Use mim to install mmcv
mim install mmcv-full==1.5
# Install correct mmdetection version
pip3 install mmdet==2.21.0

# Install everything else
pip3 install future tensorboard mlflow joblib tqdm zstandard aiohttp yaml

echo "Packages installed."

echo "Installing TBBRDet repository as editable..."
pip3 install -e ./TBBRDet
