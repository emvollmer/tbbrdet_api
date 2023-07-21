#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
USAGE

Training with Mask RCNN FPN model from scratch:
python train.py configs/mmdet/<MODEL_NAME>/..._coco.scratch.py
--work-dir /path/to/work_dir --seed <SEED_NUM> --deterministic
# ... with additional settings:
--cfg-options 'data_root'='/path/to/datasets' 'runner.max_epochs'=12 'data.workers_per_gpu'=4

Training with Mask RCNN FPN model with COCO pretrained weights:
python train.py configs/mmdet/<MODEL_NAME>/..._coco.pretrained.py
--work-dir /path/to/work_dir --seed <SEED_NUM> --deterministic
--cfg-options 'data_root'='/path/to/datasets' 'load_from'='/path/to/pretrained/weights.pth'

Training with Mask RCNN FPN model by resuming previous training:
python train.py configs/mmdet/<MODEL_NAME>/..._coco.scratch.py
--work-dir /path/to/work_dir --seed <SEED_NUM> --deterministic --resume-from '/path/to/model.pth'
--cfg-options 'data_root'='/path/to/datasets'
"""
import os
# imports
import subprocess
import sys
import os.path as osp

import torch
import logging
from pathlib import Path
import yaml
# --------------------------------------
from TBBRDet.scripts.mmdet import (
    train, numpy_loader, common_vars
)
from tbbrdet_api import configs
from tbbrdet_api.misc import (
    set_log, get_pth_to_resume_from
)

logger = logging.getLogger('__name__')


def main(args):
    """
    Implement training depending on what arguments the user provided.

    Args:
        args: Arguments from fields.py (user inputs in swagger ui)
    """
    # setting parameters and constants from user arguments
    args['cfg_options'] = {'data_root': configs.DATA_PATH,
                           'runner.max_epochs': args['epochs'],
                           'data.samples_per_gpu': args['batch'],
                           'data.workers_per_gpu': args['workers']
                           }

    if not args['device'] or (args['device'] and not torch.cuda.is_available()):
        logger.error("Training requires a GPU. Please ensure a GPU is available before training.")
        sys.exit(1)

    # define config to be used by train_from statement
    if "scratch" in args['train_from']:
        args['conf'] = osp.join(configs.TOP_LEVEL_DIR, "TBBRDet/configs/mmdet/swin/"
                                                       "mask_rcnn_swin-t-p4-w7_fpn_fp16_ms"
                                                       "-crop-3x_coco.scratch.py")
    else:
        args['conf'] = osp.join(configs.TOP_LEVEL_DIR, "TBBRDet/configs/mmdet/swin/"
                                                       "mask_rcnn_swin-t-p4-w7_fpn_fp16_ms"
                                                       "-crop-3x_coco.pretrained.py")

    CKPT_PRETRAIN = args['ckp_pretrain_pth']
    CKPT_RESUME = args['ckp_resume_dir']
    args['ckp_resume_pth'] = None
    OUT_DIR = args['model_work_dir']

    # define training command for resuming model training
    if CKPT_RESUME is not None:
        logger.info('Resuming training of a previously trained model...'
                    f"\nconfig: {args['conf']}\nwork_dir: {OUT_DIR}\nresume from: {CKPT_RESUME}")

        pth_name = get_pth_to_resume_from(directory=CKPT_RESUME,
                                          priority=['latest', 'best', 'epoch'])
        assert pth_name, f"No '.pth' files in {CKPT_RESUME} to resume from!"
        args['ckp_resume_pth'] = osp.join(CKPT_RESUME, pth_name)    # amend ckpt resume path

    # define training command for starting new training with COCO pretrained weights
    elif CKPT_PRETRAIN is not None:
        logger.info('Training model from COCO pretrained weights...'
                    f"\nconfig: {args['conf']}\nwork_dir: {OUT_DIR}\nload from: {CKPT_PRETRAIN}")

        args['cfg_options']['load_from'] = CKPT_PRETRAIN     # amend cfg_options to include load

    # define training command for training from scratch
    else:
        logger.info(f"Training model from scratch..."
                    f"\nconfig: {args['conf']}\nwork_dir: {OUT_DIR}")

    # Set logging file.
    set_log(OUT_DIR)
    yaml_save(file_path=os.path.join(OUT_DIR, 'options.yaml'), data=args)
    print(f"Training starting with the settings:")
    for k, v in args.items():
        print(f"\t'{k}': {v}")

    # call on TBBRDet training scripts
    # note: this may have to be done via subprocess, probably won't work by external function call
    train.main(
        config=args['conf'], work_dir=OUT_DIR,
        resume_from=args['ckp_resume_pth'], auto_resume=False, no_validate=False,
        gpus=None, gpu_ids=None, gpu_id=0,
        seed=args['seed'], deterministic=True, launcher='none', local_rank=0,
        cfg_options=args['cfg_options'],
    )

    logger.info(f'Model and logs were saved to {args["name"]}')


def yaml_save(file_path=None, data={}):
    """
    Save provided data to a yaml file at file_path destination

    Function from:
    https://github.com/falibabaei/fasterrcnn_pytorch_training_pipeline/blob/main/utils/general.py

    Args:
        file_path: path to where yaml file will be saved to
        data: data to be saved
    """
    with open(file_path, 'w') as f:
        yaml.safe_dump(
            {k: str(v) if isinstance(v, Path) else v for k, v in data.items()},
            f,
            sort_keys=False
        )


# =========================== TRAINING LOGIC BROKEN DOWN INTO STEPS
# Different training function calls depending on what the aim is: (only the required flags)

# 1. train from scratch:
# python train.py /tbbrdet_api/TBBRDet/configs/mmdet/swin/mask_rcnn_swin-t-p4-w7_fpn_fp16_ms-crop-3x_coco.py
#   --work-dir /tbbrdet_api/models/trained_models/scratch/
#   --cfg-options data_root=/tbbrdet_api/data/

# 2. train from COCO pretrained ckp:    ----- TEST NOT COMPLETE!
# python train.py /tbbrdet_api/TBBRDet/configs/mmdet/swin/mask_rcnn_swin-t-p4-w7_fpn_fp16_ms-crop-3x_coco.pretrained.py
#   --work-dir /tbbrdet_api/models/trained_models/pretrained/
#   --cfg-options
#       data_root=/tbbrdet_api/data/
#       load_from=/tbbrdet_api/models/pretrained_weights/mask_rcnn_swin-t-p4-w7_fpn_fp16_ms-crop-3x_coco_20210908_165006-90a4008c.pth

# 3. resume train from scratch of original model: (XXX.pth: latest.pth / best_AR@1000_epoch_xxx.pth)
# python train.py /tbbrdet_api/TBBRDet/configs/mmdet/swin/mask_rcnn_swin-t-p4-w7_fpn_fp16_ms-crop-3x_coco.py
#   --work-dir /tbbrdet_api/models/trained_models/scratch/
#   --resume-from /tbbrdet_api/models/orig_trained_models/scratch/XXX.pth
#   --cfg-options data_root=/tbbrdet_api/data/
# note: for the latest.pth: --auto-resume works as well

# 4. resume train from pretrained of original model: (XXX.pth: latest.pth / best_AR@1000_epoch_xxx.pth)
# python train.py /tbbrdet_api/TBBRDet/configs/mmdet/swin/mask_rcnn_swin-t-p4-w7_fpn_fp16_ms-crop-3x_coco.py
#   --work-dir /tbbrdet_api/models/trained_models/pretrained/
#   --resume-from /tbbrdet_api/models/orig_trained_models/pretrained/XXX.pth
#   --cfg-options data_root=/tbbrdet_api/data/
# note: for the latest.pth: --auto-resume works as well

# 5. resume train from scratch of platform model: (XXX.pth: latest.pth / best_AR@1000_epoch_xxx.pth)
# python train.py /tbbrdet_api/TBBRDet/configs/mmdet/swin/mask_rcnn_swin-t-p4-w7_fpn_fp16_ms-crop-3x_coco.py
#   --work-dir /tbbrdet_api/models/trained_models/scratch/
#   --resume-from /tbbrdet_api/models/trained_models/scratch/XXX.pth
#   --cfg-options data_root=/tbbrdet_api/data/
# note: for the latest.pth: --auto-resume works as well

# 6. resume train from pretrain of platform model: (XXX.pth: latest.pth / best_AR@1000_epoch_xxx.pth)
# python train.py /tbbrdet_api/TBBRDet/configs/mmdet/swin/mask_rcnn_swin-t-p4-w7_fpn_fp16_ms-crop-3x_coco.py
#   --work-dir /tbbrdet_api/models/trained_models/pretrained/
#   --resume-from /tbbrdet_api/models/trained_models/pretrained/XXX.pth
#   --cfg-options data_root=/tbbrdet_api/data/
# note: for the latest.pth: --auto-resume works as well

# ########################################### What stays the same:
# --cfg-options data_root=/tbbrdet_api/data/
# --deterministic
# --seed x          # otherwise no seed number!
# ########################################### What only has very few exceptions?
# config: /.../mask_rcnn_swin-t-p4-w7_fpn_fp16_ms-crop-3x_coco.py
#         --- only for 2., we need "..._coco.pretrained.py" and load_from, not --resume-from
# ########################################### What choices to we have?
# --work-dir:       .../scratch/ OR .../pretrained/
# --resume-from:    NO (1, 2) OR YES (3 - 6)
#                   --resume-from:  .../orig_trained_models/... OR .../trained_models/...
#                   --resume-from:  .../latest.pth OR .../best_AR@1000_epoch_xxx.pth
# ########################################### Question
# Which model to infer by
# -------- idea: if user doesn't provide one, default to "/models/best.txt" which leads to "....pth"

