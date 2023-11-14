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
import ast
from datetime import datetime
import os
import subprocess
import sys
import os.path as osp
from aiohttp.web import HTTPError, HTTPException
import torch
import logging
from pathlib import Path
import yaml

from tbbrdet_api import configs
from tbbrdet_api.misc import (
    set_log, run_subprocess, get_weights_folder
)

logger = logging.getLogger('__name__')


def main(args):
    """
    Implement training depending on what arguments the user provided.

    Args:
        args: Arguments from fields.py (user inputs in swagger ui)
    Return:
        Model with which inference is performed
    """

    # Define specifics of training
    submodule_config_path = Path(configs.SUBMODULE_CONFIGS_PATH, args['architecture'])
    timestamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    args['auto_resume'] = None

    if args['train_from'] == "scratch":
        # TRAINING FROM SCRATCH
        print("----- We're training from scratch -----")

        args['conf'] = [str(p) for p in submodule_config_path.glob("*coco.scratch.py")][-1]
        args['model_dir'] = osp.join(configs.MODEL_PATH, args['architecture'], args['train_from'], timestamp)
        Path(args['model_dir']).mkdir(parents=True, exist_ok=True)

    elif args['train_from'] == "coco":
        # TRAINING FROM COCO PRETRAINED WEIGHTS
        print("----- We're training from coco -----")

        weights_dir = get_weights_folder(args)
        try:
            weights_path = sorted(weights_dir.glob("*.pth"))[0]
        except IndexError as e:
            logger.error(f"Could not find a '.pth' file in the remote directory '{weights_dir}'. "
                         f"No training using {args['train_from']} pretrained weights possible!", 
                         e, exc_info=True)
            raise HTTPE(e)

        args['conf'] = [str(p) for p in submodule_config_path.glob("*coco.pretrained.py")][-1]
        args['model_dir'] = osp.join(configs.MODEL_PATH, args['architecture'], args['train_from'], timestamp)
        Path(args['model_dir']).mkdir(parents=True, exist_ok=True)
        args['cfg_options']['load_from'] = str(weights_path)

    else:        
        # RESUMING TRAINING OF PREVIOUSLY TRAINED MODEL
        print(f"----- We're resuming training from {args['train_from']}-----")

        # make sure user provided architecture and model architecture are the same
        if args['architecture'] not in Path(args['train_from']).parts:
            logger.warning(f"The selected model to resume from '{args['train_from']}' and "
                           f"architecture '{args['train_from']} do not match! Using architecture from model.")
            try:
                args['architecture'] = [a for a in configs.ARCHITECTURES if a in Path(args['train_from']).parts][0]
                logger.info(f"Defined new architecture '{args['architecture']}' from '{args['train_from']}'.")
            except IndexError as e:
                logger.error(f"Could not find a valid architecture in '{args['train_from']}'!", e, exc_info=True)
                raise HTTPError(e)

            submodule_config_path = Path(configs.SUBMODULE_CONFIGS_PATH, args['architecture'])

        args['conf'] = [str(p) for p in submodule_config_path.glob("*coco.py")][-1]
        args['auto_resume'] = "--auto-resume"
        args['model_dir'] = args['train_from']  # resume in train_from dir regardless of whether it's remote or local

        # redefine epoch number - has to be the total of already trained + additional wanted by user
        try:
            log_paths = sorted(Path(args['train_from']).glob("*.log.json"))

            # find make sure found epoch number in log data was trained fully (has a matching epoch_LOGNUM.pth file)
            while True:
                if log_paths:
                    log_path = log_paths[-1]
                    with open(log_path, "r") as f:
                        log_data = f.readlines()
                    prev_trained_epochs = ast.literal_eval(log_data[-1])['epoch']
 
                    if Path(args['train_from'], f'epoch_{prev_trained_epochs}.pth').is_file():
                        break
                    else:
                        log_paths.pop(-1)
                else:
                    prev_trained_epochs = 0

            logger.info(f"Previously fully trained epochs {prev_trained_epochs} will be added to "
                        f"user defined epoch number {args['epochs']}")
            args['cfg_options']['runner.max_epochs'] = prev_trained_epochs + args['epochs']

        except IndexError as e:
            logger.error(f"Could not find a '.log.json' file in the previously trained model folder"
                         f" {args['train_from']}. Cannot continue incomplete training!\nError: %s", 
                         e, exc_info=True)
            # get a TypeError (__init__ takes 1 positional argument...) with HTTPException/Error(e)
            raise   # equivalent to "raise e" --- this still returns code 200

        except KeyError as e:
            logger.warning(f"Previous training incomplete, no 'epoch' key found in {log_path.name} file."
                           f" Assuming epoch previously trained epoch number as 0.")
            args['cfg_options']['runner.max_epochs'] = 0 + args['epochs']

    # Set logging file.
    set_log(args['model_dir'])
    yaml_save(file_path=os.path.join(args['model_dir'], 'options.yaml'), data=args)
    print(f"Training starting with the settings:")
    for k, v in args.items():
        print(f"\t'{k}': {v}")

    # call on TBBRDet training scripts
    cfg_options_str = ' '.join([f"'{key}'={value}" for key, value in args['cfg_options'].items()])
    train_cmd = list(filter(None, [
        "/bin/bash", str(Path(configs.API_PATH, 'scripts', 'execute_train_evaluate.sh')),
        "--config-path", args['conf'],
        "--work-dir", args['model_dir'],
        "--seed", str(args['seed']),
        args['auto_resume'],
        "--cfg-options", cfg_options_str,
        "--eval", args['eval']
    ]))
    print(f"=====================\n"
          f"Training with train_cmd:\n{train_cmd}\n"
          f"=====================")

    run_subprocess(command=train_cmd, process_message="training", timeout=10000)
    logger.info(f'Model and logs were saved to {args["model_dir"]}')
    return args['model_dir']


def yaml_save(file_path=None, data={}):
    """
    Save provided data to a yaml file at file_path destination

    Function based on:
    https://github.com/falibabaei/fasterrcnn_pytorch_training_pipeline/blob/main/utils/general.py

    Args:
        file_path: path to where yaml file will be saved to
        data: data to be saved
    """
    with open(str(file_path), 'w') as f:
        yaml.safe_dump(
            {k: str(v) for k, v in data.items()},
            f,
            sort_keys=False
        )
