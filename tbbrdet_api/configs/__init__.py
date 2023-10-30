#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Configuration loader for TBBRDet

Based on: K Alibabaei's fasterrcnn_pytorch_api.git
https://git.scc.kit.edu/m-team/ai/fasterrcnn_pytorch_api/-/blob/master/fasterrcnn_pytorch_api/configs/__init__.py
"""
# imports
import ast
import configparser
import os
from pathlib import Path
import pathlib
from importlib_metadata import metadata as _metadata
# --------------------------------------

# Get absolute paths for reference
NESTED_DIR = Path(__file__).resolve().parents[1]    # note: equivalent of "homedir" = ._api/._api/
TOP_LEVEL_DIR = NESTED_DIR.parent      # note: equivalent of "base_dir" = ._api/

# Get configuration from user env and merge with pkg settings
SETTINGS_FILE = Path(Path(__file__).parent, "settings.ini")

# note: corrected misspelling "os.getenv("fasterrcnn-pytorch-training-pipeline_SERRING", ...)"
SETTINGS_FILE = os.getenv("TBBRDet_SETTINGS", default=SETTINGS_FILE)
settings = configparser.ConfigParser()
settings.read(SETTINGS_FILE)


# note: potentially not required --- uses TOP_LEVEL_DIR? and not yet found function call anywhere
def resolve_path(TOP_LEVEL_DIR):
    if pathlib.PurePath.is_absolute(TOP_LEVEL_DIR):
        return TOP_LEVEL_DIR
    else:
        return Path(NESTED_DIR, TOP_LEVEL_DIR).absolute()


try:  # Configure model and api metadata from the repository and submodule's metadata
    MODEL_NAME = os.getenv("MODEL_NAME", default=settings['model']['name'])
    MODEL_METADATA = _metadata(MODEL_NAME).json
    API_NAME = os.getenv("API_NAME", default="tbbrdet_api")
    API_METADATA = _metadata(API_NAME).json
except Exception as err:
    raise RuntimeError("Undefined configuration for model name") from err

try:  # Configure input files for testing and possible training
    DATA_PATH = os.getenv("DATA_PATH", default=settings['data']['path'])
    # make relative path absolute
    DATA_PATH = os.path.join(TOP_LEVEL_DIR, DATA_PATH)
    os.environ["DATA_PATH"] = DATA_PATH
except KeyError as err:
    raise RuntimeError("Undefined configuration for data path") from err

try:  # Local path for caching sub/models
    MODEL_DIR = os.getenv("MODEL_DIR", settings['model_dir']['path'])
    # make relative path absolute
    MODEL_DIR = os.path.join(TOP_LEVEL_DIR, MODEL_DIR)
    os.environ["MODEL_DIR"] = MODEL_DIR
except KeyError as err:
    raise RuntimeError("Undefined configuration for model path") from err

try:  # Path for remotely downloaded sub/models and ckp_pretrain_pth
    REMOTE_MODEL_DIR = os.getenv("REMOTE", settings['remote']['models_path'])
    REMOTE_DATA_DIR = os.getenv("REMOTE", settings['remote']['data_path'])
    os.environ["REMOTE_MODEL_DIR"] = REMOTE_MODEL_DIR
    os.environ["REMOTE_DATA_DIR"] = REMOTE_DATA_DIR
except KeyError as err:
    raise RuntimeError("Undefined configuration for remote path") from err

try:  # Get model backbones
    # note: changed os.getenv("REMOTE",...) to os.getenv("BACKBONES",...) as assumed to be an error
    BACKBONES = os.getenv("BACKBONES", settings['backbones']['names'])
    if isinstance(BACKBONES, str):
        # Parse the string as a list of strings
        BACKBONES = ast.literal_eval(BACKBONES)
except KeyError as err:
    raise RuntimeError("Undefined configuration for backbones") from err

try:  # Define node space limits
    LIMIT_GB = int(os.getenv("LIMIT_GB", default=settings['node']['limit_gb']))
    DATA_LIMIT_GB = int(os.getenv("DATA_LIMIT_GB", default=settings['data']['limit_gb']))
except Exception as err:
    raise RuntimeError("Undefined configuration for disk memory space") from err
