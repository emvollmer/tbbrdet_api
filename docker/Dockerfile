# Dockerfile may have following Arguments:
# tag - tag for the Base image, (e.g. 2.9.1 for tensorflow)
# branch - user repository branch to clone (default: master, another option: test)
#
# To build the image:
# $ docker build -t <dockerhub_user>/<dockerhub_repo> --build-arg arg=value .
# or using default args:
# $ docker build -t <dockerhub_user>/<dockerhub_repo> .
#
# [!] Note: For the Jenkins CI/CD pipeline, input args are defined inside the
# Jenkinsfile, not here!

ARG tag=base

# Base image, e.g. tensorflow/tensorflow:2.9.1
FROM deephdc/uc-emvollmer-deep-oc-tbbrdet_api:${tag}

LABEL maintainer='Elena Vollmer'
LABEL version='0.0.1'
# Deepaas API for TBBRDet Model

# What user branch to clone [!]
ARG branch=master

# Install Ubuntu packages
# - gcc is needed in Pytorch images because deepaas installation might break otherwise (see docs) (it is already installed in tensorflow images)
RUN apt-get update && \
    rm -rf /var/lib/apt/lists/*

# Set LANG environment
ENV LANG C.UTF-8

# Set the working directory
WORKDIR /srv

# Install rclone (needed if syncing with NextCloud for training; otherwise remove)
RUN curl -O https://downloads.rclone.org/rclone-current-linux-amd64.deb && \
    dpkg -i rclone-current-linux-amd64.deb && \
    apt install -f && \
    mkdir /srv/.rclone/ && \
    touch /srv/.rclone/rclone.conf && \
    rm rclone-current-linux-amd64.deb && \
    rm -rf /var/lib/apt/lists/*

ENV RCLONE_CONFIG=/srv/.rclone/rclone.conf

# Initialization scripts
# deep-start can install JupyterLab or VSCode if requested
RUN git clone https://github.com/deephdc/deep-start /srv/.deep-start && \
    ln -s /srv/.deep-start/deep-start.sh /usr/local/bin/deep-start

# Necessary for the Jupyter Lab terminal
ENV SHELL /bin/bash

# Install user app
# get code repository
RUN git clone --depth 1 -b $branch --recurse-submodules  https://github.com/emvollmer/tbbrdet_api.git && \
    cd tbbrdet_api && \
    git pull --recurse-submodules && \
    git submodule update --remote --recursive && \
    # those packages below should be in requirements.txt
    pip3 install --no-cache-dir \
        future \
        tensorboard \
        mlflow \
        joblib \
        tqdm && \
    pip3 install --no-cache-dir -e ./TBBRDet && \
    pip3 install --no-cache-dir -e . && \
    cd ..

# download example model for inference (pretrained MaskRCNN Swin-T)
RUN mkdir -p /srv/tbbrdet_api/models/swin/coco/2023-12-07_130038 && \
    wget -O /srv/tbbrdet_api/models/swin/coco/2023-12-07_130038/mask_rcnn_swin-t-p4-w7_fpn_fp16_ms-crop-3x_coco.pretrained.py \
    'https://data-deep.a.incd.pt/index.php/s/3zdj9B6crYgcrKo/download?path=%2F&files=mask_rcnn_swin-t-p4-w7_fpn_fp16_ms-crop-3x_coco.pretrained.py' && \
    wget -O /srv/tbbrdet_api/models/swin/coco/2023-12-07_130038/best_AR@1000_epoch_33.pth \
    'https://data-deep.a.incd.pt/index.php/s/3zdj9B6crYgcrKo/download?path=%2F&files=best_AR@1000_epoch_33.pth' && \
    wget -O /srv/tbbrdet_api/models/swin/coco/2023-12-07_130038/latest.pth \
    'https://data-deep.a.incd.pt/index.php/s/3zdj9B6crYgcrKo/download?path=%2F&files=latest.pth'

# download COCO pretrained weights for Swin-T model training
RUN mkdir -p /srv/tbbrdet_api/models/swin/coco/pretrained_weights && \
    wget -O /srv/tbbrdet_api/models/swin/coco/pretrained_weights/mask_rcnn_swin-t-p4-w7_fpn_fp16_ms-crop-3x_coco_20210908_165006-90a4008c.pth \
    'https://download.openmmlab.com/mmdetection/v2.0/swin/mask_rcnn_swin-t-p4-w7_fpn_fp16_ms-crop-3x_coco/mask_rcnn_swin-t-p4-w7_fpn_fp16_ms-crop-3x_coco_20210908_165006-90a4008c.pth'

# Open ports: DEEPaaS (5000), Monitoring (6006), Jupyter (8888)
EXPOSE 5000 6006 8888

# Launch deepaas
CMD ["deepaas-run", "--listen-ip", "0.0.0.0", "--listen-port", "5000"]
