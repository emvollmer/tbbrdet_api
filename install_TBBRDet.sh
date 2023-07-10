#!/bin/bash

# Check for activated venv
if [[ "$VIRTUAL_ENV" != "" ]]; then
   echo "Currently in an activated virtual environment. Setup may proceed."
else
   echo "Currently not in an activated virtual environment. Must set one up to proceed with package installation..."
   read -p "Please provide the directory of a venv you want to activate and use: " venv_dir
   venv_pth="$venv_dir"/bin/activate
   if test -f "$venv_pth"; then
      echo "'$venv_pth' exists and is a venv. Activating..."
      source $venv_pth
   else
      echo "'$venv_pth' does not exist. Please create a venv to use and activate it or do so by rerunning this script."; sleep 5
      exit
   fi
fi

# Get the installed Python version
python_version=$(python --version 2>&1)

# Extract the version number from the output
current_ver=$(echo "$python_version" | cut -d ' ' -f 2 | cut -d '.' -f 1,2)

# Compare the version number with the desired version
required_ver="3.6"
if [ "$required_ver" = "$current_ver" ]; then
    echo "Python version requirement $required_ver satisfied."
else
    if [ "$(printf '%s\n' "$required_ver" "$current_ver" | sort -V | head -n1)" = "$required_ver" ]; then 
    	echo "Python version $current_ver is greater than ${required_ver}. This may lead to conflicts in the Torch installation. Please install 3.6."
    else
    	echo "Python version is not compatible (less than ${required_ver}). Please install Python 3.6"; sleep 5
    fi
    exit 1
fi

# Ensure packages are installed in the correct order
# Make sure pip is up to date
pip3 install --upgrade pip setuptools wheel

# Install correct torch version
pip3 install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
# Install openmim using pip
pip3 install -U openmim
# Use mim to install mmcv
mim3 install mmcv-full==1.5
# Install correct mmdetection version
pip3 install mmdet==2.21.0

# Install everything else
pip3 install future tensorboard mlflow joblib tqdm

echo "Packages installed."

echo "Installing TBBRDet repository as editable..."
pip3 install -e ./TBBRDet
