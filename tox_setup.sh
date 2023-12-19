#!/bin/bash

echo "Tox setup..."
export PBR_VERSION=0.0.1

# Ensure packages are installed in the correct order
# Make sure pip is up to date
pip3 install pip==21.3.1 setuptools==59.6.0 wheel==0.37.1

echo "Installing PyTorch..."
# Install correct torch version
# pip3 install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download
# .pytorch.org/whl/torch_stable.html --no-cache-dir
pip3 install torch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0
# Install openmim using pip
pip3 install -U openmim
# Use mim to install mmcv
mim install mmcv-full==1.5
# Install correct mmdetection version
pip3 install mmdet==2.21.0

# Install everything else
pip3 install future tensorboard mlflow joblib tqdm

pip3 install 'webargs>=5.5.2' 'deepaas>=1.3.0' 'importlib-metadata>=1.7.0' aiohttp pyyaml

echo "Installing tox related packages..."
pip3 install flake8 'bandit>=1.1.0'

# Remove once we rely on coverage 4.3+
# https://bitbucket.org/ned/coveragepy/issues/519/
pip3 install 'coverage!=4.4,>=4.0'   # Apache-2.0

pip3 install 'stestr>=1.0.0' 'testtools>=1.4.0'

pip3 install pytest pytest-cov

echo "Installing TBBRDet repository as editable..."
pip3 install -e ./TBBRDet

# echo "Installing tbbrdet_api repository as editable..."
# pip3 install -e .

echo "==============================="
echo "Installation process complete."
echo "==============================="
