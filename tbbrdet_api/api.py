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
from torch import cuda
# from datetime import datetime
from pathlib import Path
# from tqdm import tqdm
from PIL import Image
import pkg_resources

from tbbrdet_api import configs, fields, misc
from tbbrdet_api.scripts.train import main
from tbbrdet_api.scripts.infer import infer
from tbbrdet_api.misc import (
    _catch_error, extract_zst,
    copy_file,
    ls_folders,   # download_folder_from_nextcloud, check_train_from
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
        'datasets_local': ls_folders(configs.DATA_PATH, '*.npy'),
        'datasets_remote': [str(p) for p in configs.REMOTE_DATA_PATH.glob("[!.]*")],
        'model_folders_local': ls_folders(configs.MODEL_PATH),
        'model_folders_remote': ls_folders(configs.REMOTE_MODEL_PATH),
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

    if not args['device'] or (args['device'] and not cuda.is_available()):
        logger.error("Training requires a GPU. Please ensure a GPU is available before training.")
        sys.exit(1)

    # if no data in local data folder, download it from Nextcloud
    if not all(folder in os.listdir(configs.DATA_PATH) for folder in ["train", "test"]):
        logger.info(f"Data folder '{configs.DATA_PATH}' empty, "
                    f"downloading data from '{configs.REMOTE_DATA_PATH}'...")

        logger.info("Extracting data from any .tar.zst files...")
        extract_zst()

        for json_path in configs.REMOTE_DATA_PATH.glob("*.json"):
            if "100-104" in json_path.name:
                copy_file(json_path, Path(configs.DATA_PATH, "train"))
            elif "105" in json_path.name:
                copy_file(json_path, Path(configs.DATA_PATH, "test"))
            else:
                logger.warning(f"Annotation file {json_path} neither the train nor test file. Not copying.")

    # training config definitions
    args['cfg_options'] = {'data_root': str(configs.DATA_PATH),
                          'runner.max_epochs': args['epochs'],
                          'data.samples_per_gpu': args['batch'],
                          'data.workers_per_gpu': args['workers']
                          }

    model_dir = main(args)

    return {f'Model and logs were saved to {model_dir}'}


def predict(**args):
    """
    Performs inference on an input image.

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

    # define output directory regardless of whether it's remote or local
    args['out_dir'] = Path(Path(args['predict_model_dir']), "predictions")
    args['out_dir'].mkdir(parents=True, exist_ok=True)

    result = infer(args)

    if args['accept'] == 'application/json':
        return {'result': f"Inference result(s) saved to {', '.join(result)}"}
    elif args['accept'] == 'image/png':
        # todo: Find alternative, can't handle PIL's Image (ValueError: Unsupported body type <class 'PIL.PngImagePlugin.PngImageFile'>)
        return Image.open(result[0])


if __name__ == '__main__':
    ex_args = {
        'architecture': 'swin',
        'train_from': '/storage/tbbrdet/models/swin/coco/2023-05-10_103541/',  # 'scratch',
        'device': True,
        'epochs': 1,
        'workers': 2,
        'batch': 1,
        'lr': 0.0001,
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
        # 'no_labels': False,
        'accept': 'image/png'
    }
    predict(**ex_args)

