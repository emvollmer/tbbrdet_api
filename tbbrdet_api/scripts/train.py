#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
USAGE

Training with Mask RCNN Swin-T FPN model from scratch:
python train.py configs/mmdet/<MODEL_NAME>/..._coco.scratch.py
--work-dir /path/to/work_dir --seed <SEED_NUM> --deterministic
--cfg-options 'data_root'='/path/to/datasets'

Training with Mask RCNN Swin-T FPN model with COCO pretrained weights:
python train.py configs/mmdet/<MODEL_NAME>/..._coco.pretrained.py
--work-dir /path/to/work_dir --seed <SEED_NUM> --deterministic
--cfg-options 'data_root'='/path/to/datasets'
              'load_from'='/path/to/pretrained/weights.pth'

Training with Mask RCNN Swin-T FPN model by resuming previous training:
python train.py configs/mmdet/<MODEL_NAME>/..._coco.scratch.py
--work-dir /path/to/work_dir --seed <SEED_NUM> --deterministic
--resume-from '/path/to/model.pth'
--cfg-options 'data_root'='/path/to/datasets'
"""
import ast
from datetime import datetime
import os
import os.path as osp
from aiohttp.web import HTTPError  # HTTPException
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
    submodule_config_path = Path(
        configs.SUBMODULE_CONFIGS_PATH, args['architecture']
    )
    timestamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    args['auto_resume'] = None

    if args['train_from'] == "scratch":
        # TRAINING FROM SCRATCH
        print("----- We're training from scratch -----")

        args['conf'] = [str(p) for p in
                        submodule_config_path.glob("*coco.scratch.py")][-1]
        args['model_dir'] = osp.join(
            configs.MODEL_PATH, args['architecture'],
            args['train_from'], timestamp
        )
        Path(args['model_dir']).mkdir(parents=True, exist_ok=True)

    elif args['train_from'] == "coco":
        # TRAINING FROM COCO PRETRAINED WEIGHTS
        print("----- We're training from coco -----")

        weights_dir = get_weights_folder(args)
        try:
            weights_path = sorted(weights_dir.glob("*.pth"))[0]
        except IndexError as e:
            logger.error(f"Could not find a '.pth' file in the remote "
                         f"directory '{weights_dir}'. No training using "
                         f"{args['train_from']} pretrained weights possible!",
                         e, exc_info=True)
            raise HTTPError(e)

        args['conf'] = [str(p) for p in
                        submodule_config_path.glob("*coco.pretrained.py")][-1]
        args['model_dir'] = osp.join(
            configs.MODEL_PATH, args['architecture'],
            args['train_from'], timestamp
        )
        Path(args['model_dir']).mkdir(parents=True, exist_ok=True)
        args['cfg_options']['load_from'] = str(weights_path)

    else:
        # RESUMING TRAINING OF PREVIOUSLY TRAINED MODEL
        print(f"----- We're resuming training from {args['train_from']}-----")

        # ensure user provided architecture & model architecture are the same
        if args['architecture'] not in Path(args['train_from']).parts:
            logger.warning(
                f"The selected model to resume from '{args['train_from']}' and"
                f" architecture '{args['train_from']} do not match! "
                f"Using architecture from model.")
            try:
                args['architecture'] = [
                    a for a in configs.ARCHITECTURES
                    if a in Path(args['train_from']).parts
                ][0]
                logger.info(f"Defined new architecture "
                            f"'{args['architecture']}' from "
                            f"'{args['train_from']}'.")
            except IndexError as e:
                logger.error(f"Could not find a valid architecture "
                             f"in '{args['train_from']}'!", e, exc_info=True)
                raise HTTPError(e)

            submodule_config_path = Path(
                configs.SUBMODULE_CONFIGS_PATH, args['architecture']
            )

        args['conf'] = [str(p) for p in
                        submodule_config_path.glob("*coco.py")][-1]
        args['auto_resume'] = "--auto-resume"
        args['model_dir'] = args['train_from']  # regardless of remote / local

        # redefine epoch number: epochs wanted by user + epochs already trained
        user_epochs = args['epochs']

        try:
            # get existing epoch paths and sort by epoch number
            epoch_paths = list(Path(args['train_from']).glob("epoch_*.pth"))
            epoch_paths.sort(
                key = lambda x: int(x.stem.split('_')[1])
            )
            # get previously trained epoch number from newest file
            prev_epochs = int(epoch_paths[-1].stem.split('_')[1])

            print(f"Previously trained epochs {prev_epochs} "
                  f"will be added to user defined epoch number "
                  f"{user_epochs}")  # logger.info
            args['cfg_options']['runner.max_epochs'] = (prev_epochs
                                                        + user_epochs)

        except IndexError:
            logger.warning(
                "Previous training incomplete, no 'epoch_*.pth' found "
                "in selected model folder. Assuming number of previously "
                "trained epochs as 0."
            )
            args['cfg_options']['runner.max_epochs'] = 0 + user_epochs

    # Set logging file
    set_log(args['model_dir'])
    yaml_save(file_path=os.path.join(args['model_dir'], 'options.yaml'),
              data=args)
    print("Training starting with the settings:")
    for k, v in args.items():
        print(f"\t'{k}': {v}")

    # Call TBBRDet training scripts
    cfg_options_str = ' '.join([f"'{key}'={value}"
                                for key, value in args['cfg_options'].items()])
    train_cmd = list(filter(None, [
        "/bin/bash", str(Path(configs.API_PATH, 'scripts',
                              'execute_train_evaluate.sh')),
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

    run_subprocess(command=train_cmd, process_message="training",
                   timeout=10000)
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
