#!/bin/bash


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


# Make sure all applications are there for next steps
echo "Installing necessary applications for virtual environment setup"
yes | apt-get install python3.6-dev

if message=$(yes | apt-get install python3.6-virtualenv 2>&1); then
   echo "Python3.6-virtualenv successfully installed."
else
   echo "Python3.6-virtualenv installation unsuccessful:'$message'"
   yes | apt install python3.6-venv
fi


# Switch to api directory if not already there
if ! [[ "$PWD" == *\tbbrdet_api ]] && [ -d "tbbrdet_api" ] ; then
   echo "Not yet in tbbrdet_api. Changing into directory to continue working..."
   cd tbbrdet_api
fi

# Check for activated venv, create one if there is none, otherwise activate it
if [[ "$VIRTUAL_ENV" != "" ]]; then
   echo "Currently in an activated virtual environment. Setup may proceed."
else
   echo "Currently not in an activated virtual environment. Must have one up to proceed with package installation..."
   venv_name="venv"
   venv_act="$venv_name"/bin/activate
   if test -f $venv_act; then
      echo "Virtual environment at '$venv_name' already exists. Activating..."
   else
      if message=$(python3.6 -m venv "$venv_name" 2>&1); then
          echo "Virtual environment successfully created. Activating..."
      else
	  echo "Python3.6 venv created unsuccessful. Stopping..."; sleep 10
	  exit 1
      fi
   fi
   source $venv_act
fi


install_and_check_package() {
    local package_name=${1#* }  # Remove the command part from the package name

    echo "Installing $package_name..."
    "$1"

    # Check the exit code of the previous command
    if [ $? -ne 0 ]; then
        echo "Installation of $package_name failed."; sleep 10
        # exit 1   # Exit the script with a non-zero status to indicate failure
    else
        echo "$package_name installed successfully."
    fi
}


# Ensure packages are installed in the correct order
# Make sure pip is up to date
pip3 install pip==21.3.1 setuptools==59.6.0 wheel==0.37.1

# Install correct torch version
pip3 install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html --no-cache-dir
# Install openmim using pip
pip3 install -U openmim
# Use mim to install mmcv
mim install mmcv-full==1.5
# Install correct mmdetection version
pip3 install mmdet==2.21.0

# Install everything else
pip3 install future tensorboard mlflow joblib tqdm

echo "Packages installed."

echo "Installing TBBRDet repository as editable..."
pip3 install -e ./TBBRDet

echo "Installing tbbrdet_api repository as editable..."
pip3 install -e .

echo "==============================="
echo "Installation process complete."
echo "==============================="
