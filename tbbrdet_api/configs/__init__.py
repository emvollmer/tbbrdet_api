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
import logging
import os
from pathlib import Path
import pathlib
from importlib_metadata import metadata as _metadata
# --------------------------------------

# Get absolute paths for reference
API_PATH = Path(__file__).resolve().parents[1]    # = ._api/._api/
BASE_PATH = API_PATH.parent      # = ._api/

# Get configuration from user env and merge with pkg settings
SETTINGS_FILE = Path(Path(__file__).parent, "settings.ini")

SETTINGS_FILE = os.getenv("TBBRDet_SETTINGS", default=SETTINGS_FILE)
settings = configparser.ConfigParser()
settings.read(SETTINGS_FILE)


# NOTE: required? Uses BASE_PATH and not yet found function call anywhere...
def resolve_path(BASE_PATH):
    if pathlib.PurePath.is_absolute(BASE_PATH):
        return BASE_PATH
    else:
        return Path(API_PATH, BASE_PATH).absolute()


try:  # Configure model and api metadata from the repo and submodule's metadata
    MODEL_NAME = os.getenv("MODEL_NAME", default=settings['model']['name'])
    MODEL_METADATA = _metadata(MODEL_NAME).json
    API_NAME = os.getenv("API_NAME", default=settings['api']['name'])
    API_METADATA = _metadata(API_NAME).json
except Exception as err:
    raise RuntimeError("Undefined configuration for model name") from err

try:  # Configure input files for testing and possible training
    DATA_PATH = os.getenv(
        "DATA_PATH", default=Path(BASE_PATH, settings['local']['data'])
    )
    # os.environ["DATA_PATH"] = str(DATA_PATH)  # api use doesn't work anymore
except KeyError as err:
    raise RuntimeError("Undefined configuration for data path") from err

try:  # Local path for caching sub/models
    MODEL_PATH = os.getenv(
        "MODEL_PATH", default=Path(BASE_PATH, settings['local']['models'])
    )
    # os.environ["MODEL_PATH"] = str(MODEL_PATH)
except KeyError as err:
    raise RuntimeError("Undefined configuration for model path") from err

try:  # Path for remotely downloaded sub/models and ckp_pretrain_pth
    REMOTE_PATH = os.getenv(
        "REMOTE_PATH", default=Path(settings['remote']['path'])
    )
    REMOTE_MODEL_PATH = os.getenv(
        "REMOTE_MODEL_PATH",
        default=Path(REMOTE_PATH, settings['remote']['models'])
    )
    REMOTE_DATA_PATH = os.getenv(
        "REMOTE_DATA_PATH",
        default=Path(REMOTE_PATH, settings['remote']['data'])
    )
    # Removed below definitions as fails with Path used functionalities
    # os.environ["REMOTE_PATH"] = str(REMOTE_PATH)
    # os.environ["REMOTE_MODEL_PATH"] = str(REMOTE_MODEL_PATH)
    # os.environ["REMOTE_DATA_PATH"] = str(REMOTE_DATA_PATH)
except KeyError as err:
    raise RuntimeError("Undefined configuration for remote path") from err

SUBMODULE_PATH = os.getenv(
    "SUBMODULE_PATH", default=Path(BASE_PATH, settings['model']['name'])
)
SUBMODULE_CONFIGS_PATH = os.getenv(
    "SUBMODULE_CONFIGS_PATH", default=Path(SUBMODULE_PATH, "configs", "mmdet")
)

try:  # Get model backbones
    BACKBONES = os.getenv("BACKBONES", default=settings['backbones']['names'])
    if isinstance(BACKBONES, str):
        # Parse the string as a list of strings
        BACKBONES = ast.literal_eval(BACKBONES)
except KeyError as err:
    raise RuntimeError("Undefined configuration for backbones") from err

try:  # Define node space limits
    LIMIT_GB = os.getenv("LIMIT_GB",
                         default=int(settings['local']['limit_gb']))
    DATA_LIMIT_GB = os.getenv("DATA_LIMIT_GB",
                              default=int(settings['local']['data_limit_gb']))
except Exception as err:
    raise RuntimeError("Undefined configuration for disk memory space") \
        from err

ARCHITECTURES = os.getenv("ARCHITECTURES",
                          default=settings['model']['architectures'])
if isinstance(ARCHITECTURES, str):
    ARCHITECTURES = ast.literal_eval(ARCHITECTURES)

TRAIN_OPTIONS = os.getenv("TRAIN_OPTIONS",
                          default=settings['training']['options'])
if isinstance(TRAIN_OPTIONS, str):
    TRAIN_OPTIONS = ast.literal_eval(TRAIN_OPTIONS)


# configure logging:
ENV_LOG_LEVEL = os.getenv("ENV_LOG_LEVEL",
                          default=settings['logging']['log_level'])
LOG_LEVEL = getattr(logging, ENV_LOG_LEVEL.upper())
