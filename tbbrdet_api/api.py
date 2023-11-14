# -*- coding: utf-8 -*-
"""
Functions to integrate your model with the DEEPaaS API.
It's usually good practice to keep this file minimal, only performing the interfacing
tasks. In this way you don't mix your true code with DEEPaaS code and everything is
more modular. That is, if you need to write the predict() function in api.py, you
would import your true predict function and call it from here (with some processing /
postprocessing in between if needed).
For example:

    import mycustomfile

    def predict(**kwargs):
        args = preprocess(kwargs)
        resp = mycustomfile.predict(args)
        resp = postprocess(resp)
        return resp

To start populating this file, take a look at the docs [1] and at a canonical exemplar
module [2].

[1]: https://docs.deep-hybrid-datacloud.eu/
[2]: https://github.com/deephdc/demo_app
"""
import ast
import logging
import os
import os.path as osp
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
# from tqdm import tqdm
import pkg_resources

from tbbrdet_api import configs, fields, misc
from tbbrdet_api.scripts.train import main
from tbbrdet_api.scripts.infer import infer
from tbbrdet_api.misc import (
    _catch_error, extract_zst,
    download_folder_from_nextcloud, copy_file,
    check_train_from, get_pth_to_resume_from
)

logger = logging.getLogger('__name__')


@_catch_error
def get_metadata():
    """
    DO NOT REMOVE - All modules should have a get_metadata() function
    with appropriate keys.

    Returns a dictionary containing metadata information about the module.

    Returns:
        A dictionary containing metadata information required by DEEPaaS.
    """
    metadata = {
        'api_authors': configs.API_METADATA.get("author"),
        'model_authors': configs.MODEL_METADATA.get("author"),
        'description': configs.MODEL_METADATA.get("summary"),
        'home_page': configs.API_METADATA.get("home_page"),
        'license': configs.API_METADATA.get("license"),
        'version': configs.API_METADATA.get("version"),
        'datasets_local': [str(p) for p in configs.DATA_PATH.glob("[!.]*")],
        'datasets_remote': [str(p) for p in Path("/storage/tbbrdet/datasets").glob("[!.]*")],
        'checkpoint_files_local': misc.ls_local(),
        'checkpoint_files_remote': misc.ls_remote(),
    }
    logger.debug("Package model metadata: %s", metadata)
    return metadata


def get_train_args():
    """
    Return the arguments that are needed to perform a  training.

    Returns:
        Dictionary of webargs fields.
      """
    # NOTE: potentially requires _fields_to_dict misc function for conversion!
    train_args = fields.TrainArgsSchema().fields
    logger.debug("Web arguments: %s", train_args)
    return train_args


def get_predict_args():
    """
    Return the arguments that are needed to perform a prediction.

    Args:
        None

    Returns:
        Dictionary of webargs fields.
    """
    # NOTE: potentially requires _fields_to_dict misc function for conversion!
    predict_args = fields.PredictArgsSchema().fields
    logger.debug("Web arguments: %s", predict_args)
    return predict_args


def train(**args):
    """
    Performs training on the dataset.
    Args:
        **args: keyword arguments from get_train_args.
    Returns:
        path to the trained model
    """
    print("Training with user provided arguments:\n", args)    # logger.info(...)

    # if no data in local data folder, download it from Nextcloud
    if not all(folder in os.listdir(configs.DATA_PATH) for folder in ["train", "test"]):
        logger.info(f"Data folder '{configs.DATA_PATH}' empty, "
                    f"downloading data from '{configs.REMOTE_DATA_PATH}'...")

        logger.info("Extracting data from any .tar.zst files...")
        extract_zst()

        for json_path in Path("/storage/tbbrdet/datasets").glob("*.json"):
            if "100-104" in json_path.name:
                copy_file(json_path, Path(configs.DATA_PATH, "train"))
            elif "105" in json_path.name:
                copy_file(json_path, Path(configs.DATA_PATH, "test"))
            else:
                logger.warning(f"Annotation file {json_path} neither the train nor test file. Not copying.")

    # define specifics of training (from scratch, pretrained, resume)
    if args['ckp_resume_dir']:
        # define whether we're resuming the training of a model from scratch or coco
        args['train_from'] = check_train_from(args['ckp_resume_dir'])

        # download model if necessary
        if "rshare" in args['ckp_resume_dir']:
            args['ckp_resume_dir'] = download_folder_from_nextcloud(
                remote_dir=args['ckp_resume_dir'],
                filetype="model", check="latest"
            )

    elif args['ckp_pretrain_pth']:
        # define that we're training from coco
        args['train_from'] = configs.settings['train_from']['coco']

        # download model if necessary
        if "rshare" in args['ckp_pretrain_pth']:
            local_pretrain_ckp_folder = download_folder_from_nextcloud(
                remote_dir=osp.dirname(args['ckp_pretrain_pth']),
                filetype="pretrained weights"
            )
            args['ckp_pretrain_pth'] = osp.join(local_pretrain_ckp_folder,
                                                osp.basename(args['ckp_pretrain_pth']))

    else:  # neither resuming nor using pretrained weights means we're training from scratch
        args['train_from'] = configs.settings['train_from']['scratch']

    timestamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    model_dir = Path(configs.MODEL_PATH, args['train_from'], timestamp)
    if not model_dir.is_dir():
        model_dir.mkdir(parents=True, exist_ok=True)
    args['model_dir'] = str(model_dir)

    main(args)

    print(f"Model and logs were saved to {args['model_dir']}")
    return {'result': f'Model and logs were saved to {args["model_dir"]}'}


# def train_new(**args):
#     """
#     Performs training on the dataset.

#     TODO: Before usable, the following adjustments are necessary:
#     - fields.TrainArgsSchema(): add field "architecture"
#     - fields.TrainArgsSchema(): add field "train_from" with options: "scratch", "coco", model_folder (folders with "latest.pth" in them)
#     - configs.settings: change definition of remote "rshare:" to "/storage/"

#     Args:
#         **args: keyword arguments from get_train_args.
#     Returns:
#         path to the trained model
#     """
#     print("Training with user provided arguments:\n", args)    # logger.info(...)

#     if not args['device'] or (args['device'] and not torch.cuda.is_available()):
#         logger.error("Training requires a GPU. Please ensure a GPU is available before training.")
#         sys.exit(1)

#     # if no data in local data folder, download it from Nextcloud
#     if not all(folder in os.listdir(configs.DATA_PATH) for folder in ["train", "test"]):
#         logger.info(f"Data folder '{configs.DATA_PATH}' empty, "
#                     f"downloading data from '{configs.REMOTE_DATA_PATH}'...")

#         logger.info("Extracting data from any .tar.zst files...")
#         extract_zst()

#         for json_path in configs.REMOTE_DATA_PATH.glob("*.json"):
#             if "100-104" in json_path.name:
#                 shutil.copy(json_path, Path(configs.DATA_PATH, "train"))
#             elif "105" in json_path.name:
#                 shutil.copy(json_path, Path(configs.DATA_PATH, "test"))
#             else:
#                 logger.warning(f"Annotation file {json_path} neither the train nor test file. Not copying.")

#     # training config definitions
#     args['cfg_options'] = {'data_root': str(configs.DATA_PATH),
#                           'runner.max_epochs': args['epochs'],
#                           'data.samples_per_gpu': args['batch'],
#                           'data.workers_per_gpu': args['workers']
#                           }

#     main_new(args)

#     return {f'Model and logs were saved to {args["model_dir"]}'}


def predict(**args):
    """
    Performs inference  on an input image.
    Args:
        **args:   keyword arguments from get_predict_args.
    Returns:
        either a json file or png image with bounding box
    """
    print("Predicting with user provided arguments:\n", args)    # logger.info(...)

    # define model-related paths
    try:
        model_dir = Path(args['predict_model_dir'])
        args['config_file'] = str(sorted(model_dir.glob("*.py"))[-1])
        args['checkpoint_file'] = str(sorted(model_dir.glob("best*.pth"))[-1])
    except IndexError as e:
        logger.error(f"No checkpoint or config file found in {args['predict_model_dir']}!"
                     f"Error: %s", e, exc_info=True)
        raise e

    # define output directory
    if str(configs.REMOTE_MODEL_PATH) in args['predict_model_dir']:
        args['out_dir'] = Path(
            args['predict_model_dir'].replace(str(configs.REMOTE_MODEL_PATH),
                                              str(configs.MODEL_PATH)), "predictions"
        )
    else:
        args['out_dir'] = Path(Path(args['predict_model_dir']), "predictions")
    args['out_dir'].mkdir(parents=True, exist_ok=True)

    result = infer(args)
    return {'result': result}


if __name__ == '__main__':
    ex_args = {
        'model': 'mask_rcnn_swin-t',
        'ckp_pretrain_pth': None,  # 'rshare:tbbrdet/models/mask_rcnn_swin-t_coco-pretrained/pretrained_weights/mask_rcnn_swin-t-p4-w7_fpn_fp16_ms-crop-3x_coco_20210908_165006-90a4008c.pth',
        'ckp_resume_dir': None,
        # 'data_config': 'test_data/submarin.yaml',
        # 'use_train_aug': False,
        'device': True,
        'epochs': 1,
        'workers': 2,
        'batch': 1,
        'lr': 0.0001,
        # 'imgsz': 640,
        'seed': 42,
        'eval': "bbox"
    }
    train(**ex_args)

    ex_args = {
        'input': '/srv/tbbrdet_api/data/test/images/Flug1_105Media/DJI_0004_R.npy',
        'predict_model_dir': '/srv/tbbrdet_api/models/mask_rcnn_swin-t_coco-pretrained/2023-11-14_085259/',
        'colour_channel': 'both',
        'threshold': 0.3,
        'device': True,
        'no_labels': False,
        'accept': 'image/png'
    }
    predict(**ex_args)

