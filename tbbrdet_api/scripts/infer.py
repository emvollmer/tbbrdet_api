#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
USAGE

Inference on provided data
"""
# imports
import os
import glob

from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import mmcv
import torch
# --------------------------------------

# todo: check if function actually necessary? Doesn't seem to be used in EGI tut scripts
def collect_all_images(dir_test):
    """
    Function to return a list of image paths.

    :param dir_test: Directory containing images or single image path.

    Returns:
        test_images: List containing all image paths.
    """
    test_images = []
    if os.path.isdir(dir_test):
        image_file_types = ['*.npy']
        for file_type in image_file_types:
            test_images.extend(glob.glob(f"{dir_test}/{file_type}"))
    else:
        test_images.append(dir_test)
    return test_images


def infer(args):
    """
    First code outline based on MMDetection JupyterNotebook Demo
    https://github.com/open-mmlab/mmdetection/blob/v2.21.0/demo/inference_demo.ipynb
    and EGI conference tutorial:
    https://git.scc.kit.edu/m-team/ai/fasterrcnn_pytorch_api/-/blob/master/fasterrcnn_pytorch_api/scripts/inference.py

    Args:
        args: Arguments from fields.py (user inputs in swagger ui)

    Returns:

    """

    config_file = '../configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
    checkpoint_file = args['model_pth']

    if args['device'] and torch.cuda.is_available():
        DEVICE = torch.device('cuda:0')
    else:
        DEVICE = torch.device('cpu')
    # build the model from a config file and a checkpoint file
    model = init_detector(config_file, checkpoint_file, device=DEVICE)

    test_images = args['input']
    print(f"Test instances: {len(test_images)}")
    # test a single image
    img = 'demo.jpg'

    # here we have to reduce npy to specific channels depending on user choice!
    # RGB or thermal (3x grayscale channel)
    result = inference_detector(model, img)

    # show the results
    show_result_pyplot(model, img, result)

    return
