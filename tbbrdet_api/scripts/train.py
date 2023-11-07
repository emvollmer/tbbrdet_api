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
# from TBBRDet.scripts.mmdet import (
#     train, numpy_loader, common_vars
# )
from tbbrdet_api import configs
from tbbrdet_api.misc import (
    set_log, get_pth_to_resume_from, run_subprocess
)

logger = logging.getLogger('__name__')


def main(args):
    """
    Implement training depending on what arguments the user provided.

    Args:
        args: Arguments from fields.py (user inputs in swagger ui)
    """
    # setting parameters and constants from user arguments
    args['cfg_options'] = {'data_root': str(configs.DATA_PATH),
                           'runner.max_epochs': args['epochs'],
                           'data.samples_per_gpu': args['batch'],
                           'data.workers_per_gpu': args['workers']
                           }

    if not args['device'] or (args['device'] and not torch.cuda.is_available()):
        logger.error("Training requires a GPU. Please ensure a GPU is available before training.")
        sys.exit(1)

    # define config to be used by train_from statement
    if "scratch" in args['train_from']:
        args['conf'] = osp.join(configs.BASE_PATH, "TBBRDet/configs/mmdet/swin/"
                                                       "mask_rcnn_swin-t-p4-w7_fpn_fp16_ms"
                                                       "-crop-3x_coco.scratch.py")
    else:
        args['conf'] = osp.join(configs.BASE_PATH, "TBBRDet/configs/mmdet/swin/"
                                                       "mask_rcnn_swin-t-p4-w7_fpn_fp16_ms"
                                                       "-crop-3x_coco.pretrained.py")

    CKPT_PRETRAIN = args['ckp_pretrain_pth']
    CKPT_RESUME = args['ckp_resume_dir']
    args['ckp_resume_pth'] = None
    args['auto_resume'] = None
    OUT_DIR = args['model_dir']

    # define training command for resuming model training
    if CKPT_RESUME is not None:
        print('Resuming training of a previously trained model...'
              f"\nconfig: {args['conf']}\nwork_dir: {OUT_DIR}\nresume from: {CKPT_RESUME}")     # logger.info

        OUT_DIR = CKPT_RESUME
        args['auto_resume'] = "--auto-resume"

        # pth_name = get_pth_to_resume_from(directory=CKPT_RESUME,
        #                                   priority=['latest', 'best', 'epoch'])
        # # TODO: Turn assert into a try-except (which will fail when we try to define args value if None)
        # assert pth_name, f"No '.pth' files in {CKPT_RESUME} to resume from!"
        # # args['ckp_resume_pth'] = osp.join(CKPT_RESUME, pth_name)    # amend ckpt resume path
        # args['cfg_options']['resume_from'] = osp.join(CKPT_RESUME, pth_name)

    # define training command for starting new training with COCO pretrained weights
    elif CKPT_PRETRAIN is not None:
        print('Training model from COCO pretrained weights...'
              f"\nconfig: {args['conf']}\nwork_dir: {OUT_DIR}\nload from: {CKPT_PRETRAIN}")     # logger.info

        args['cfg_options']['load_from'] = CKPT_PRETRAIN     # amend cfg_options to include load

    # define training command for training from scratch
    else:
        print(f"Training model from scratch..."
              f"\nconfig: {args['conf']}\nwork_dir: {OUT_DIR}")     # logger.info

    # Set logging file.
    set_log(OUT_DIR)
    yaml_save(file_path=os.path.join(OUT_DIR, 'options.yaml'), data=args)
    print(f"Training starting with the settings:")
    for k, v in args.items():
        print(f"\t'{k}': {v}")

    # call on TBBRDet training scripts
    # note: this may have to be done via subprocess, probably won't work by external function call
    cfg_options_str = ' '.join([f"'{key}'={value}" for key, value in args['cfg_options'].items()])
    train_cmd = list(filter(None, [
        "/bin/bash", str(Path(configs.API_PATH, 'scripts', 'execute_train_evaluate.sh')),
        "--config-path", args['conf'],
        "--work-dir", OUT_DIR,
        "--seed", str(args['seed']),
        args['auto_resume'],
        "--cfg-options", cfg_options_str,
        "--eval", args['eval']
    ]))
    print(f"====================\n"
          f"Training with train_cmd:\n{train_cmd}\n"
          f"=====================")

    run_subprocess(command=train_cmd, process_message="training", timeout=10000)
    # logger.info(f'Model and logs were saved to {args["model_dir"]}')


def yaml_save(file_path=None, data={}):
    """
    Save provided data to a yaml file at file_path destination

    Function from:
    https://github.com/falibabaei/fasterrcnn_pytorch_training_pipeline/blob/main/utils/general.py

    Args:
        file_path: path to where yaml file will be saved to
        data: data to be saved
    """
    with open(str(file_path), 'w') as f:
        yaml.safe_dump(
            {k: str(v) for k, v in data.items()},
            # {k: str(v) if isinstance(v, Path) else v for k, v in data.items()},
            f,
            sort_keys=False
        )


# def main_new(args):
#     """
#     Implement training depending on what arguments the user provided.

#     TODO: Before usable, the following adjustments are necessary:
#     - fields.TrainArgsSchema(): add field "architecture"
#     - fields.TrainArgsSchema(): add field "train_from" with options: "scratch", "coco", model_folder (folders with "latest.pth" in them)
#     - configs.settings: change definition of remote "rshare:" to "/storage/"

#     Args:
#         args: Arguments from fields.py (user inputs in swagger ui)
#     """

#     # Define specifics of training
#     submodule_config_path = Path(configs.SUBMODULE_CONFIGS_PATH, args['architecture'])
#     timestamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')
#     args['auto_resume'] = None

#     if args['train_from'] == "scratch":
#         # TRAINING FROM SCRATCH
#         args['conf'] = [str(p) for p in submodule_config_path.glob("*coco.scratch.py")][-1]
#         args['model_dir'] = osp.join(configs.MODEL_PATH, args['architecture'], args['train_from'], timestamp)
#         Path(args['model_dir']).mkdir(parents=True, exist_ok=True)

#     elif args['train_from'] == "coco":
#         # TRAINING FROM COCO PRETRAINED WEIGHTS
#         args['conf'] = [str(p) for p in submodule_config_path.glob("*coco.pretrained.py")][-1]
#         args['model_dir'] = osp.join(configs.MODEL_PATH, args['architecture'], args['train_from'], timestamp)
#         Path(args['model_dir']).mkdir(parents=True, exist_ok=True)

#         try:
#             weights_path = Path(configs.REMOTE_MODEL_PATH, args['architecture'], args['train_from'], "pretrained_weights").glob("*.pth")[0]
#         except IndexError as e:
#             logger.error(f"Could not find a '.pth' file in the {args['train_from'] + '_weights'} folder"
#                          f" for the current architecture {args['architecture']}. No training from COCO possible!", 
#                          e, exc_info=True)
#             raise HTTPException(e)

#         args['cfg_options']['load_from'] = str(weights_path)

#     else:
#         # RESUMING TRAINING OF PREVIOUSLY TRAINED MODEL
#         args['conf'] = [str(p) for p in submodule_config_path.glob("*coco.py")][-1]
#         args['auto_resume'] = "--auto-resume"

#         # download model directory if it doesn't exist locally
#         if str(configs.REMOTE_PATH) in args['train_from']:
#             local_model_dir = Path(configs.MODEL_PATH, Path(args['train_from']).relative_to(configs.REMOTE_PATH))
#             if local_model_dir.is_dir():
#                 logger.info(f"Selected remote directory already exists at {local_model_dir}. Using local file instead.")
#             else:
#                 shutil.copytree(args['train_from'], local_model_dir)
            
#             args['model_dir'] = str(local_model_dir)
#         else:
#             args['model_dir'] = args['train_from']

#         # redefine epoch number - has to be the total of already trained + additional wanted by user
#         try:
#             log_path = sorted(Path(args['train_from']).glob("*.log.json"))[-1]
#             with open(log_path, "r") as f:
#                 log_data = f.readlines()
#             prev_trained_epochs = ast.literal_eval(log_data[-1])['epoch']
#             args['cfg_options']['runner.max_epochs'] = prev_trained_epochs + args['epochs']
#         except IndexError as e:
#             logger.error(f"Could not find a '.log.json' file in the previously trained model folder"
#                          f" {args['train_from']}. Cannot continue incomplete training!\nError: %s", 
#                          e, exc_info=True)
#             raise HTTPException(e)
#         except KeyError as e:
#             logger.error(f"Previous training incomplete, no 'epoch' key found in {log_path.name} file."
#                          f" Assuming epoch number as 0.")

#     # Set logging file.
#     set_log(args['model_dir'])
#     yaml_save(file_path=os.path.join(args['model_dir'], 'options.yaml'), data=args)
#     print(f"Training starting with the settings:")
#     for k, v in args.items():
#         print(f"\t'{k}': {v}")

#     # call on TBBRDet training scripts
#     # note: this may have to be done via subprocess, probably won't work by external function call
#     cfg_options_str = ' '.join([f"'{key}'={value}" for key, value in args['cfg_options'].items()])
#     train_cmd = list(filter(None, [
#         "/bin/bash", str(Path(configs.API_PATH, 'scripts', 'execute_train_evaluate.sh')),
#         "--config-path", args['conf'],
#         "--work-dir", OUT_DIR,
#         "--seed", str(args['seed']),
#         args['auto_resume'],
#         "--cfg-options", cfg_options_str,
#         "--eval", args['eval']
#     ]))
#     print(f"=====================\n"
#           f"Training with train_cmd:\n{train_cmd}\n"
#           f"=====================")

#     run_subprocess(command=train_cmd, process_message="training", timeout=10000)
#     logger.info(f'Model and logs were saved to {args["model_dir"]}')


# =========================== TRAINING LOGIC BROKEN DOWN INTO STEPS
# Different training function calls depending on what the aim is: (only the required flags)

# 1. train from scratch:
# python train.py /tbbrdet_api/TBBRDet/configs/mmdet/swin/mask_rcnn_swin-t-p4-w7_fpn_fp16_ms-crop-3x_coco.scratch.py
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

