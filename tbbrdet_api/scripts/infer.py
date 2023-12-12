#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
USAGE

Inference on provided data
"""
# imports
from aiohttp.web import HTTPError
from pathlib import Path
import logging
import shutil

import tbbrdet_api.configs as configs
from tbbrdet_api.misc import run_subprocess
# --------------------------------------
logger = logging.getLogger('__name__')


def infer(args):
    """
    Implement inference depending on what arguments the user provided.

    Args:
        args: Arguments from fields.py (user inputs in swagger ui)

    Returns:
        result
    """
    # ensure user provided image path is a numpy file or directory with npys
    try:
        # Input file is path or directory
        npy_paths = collect_image_paths(Path(args['input']))
    except TypeError:
        # Input file is from a browsing webargs field
        tmp_filepath = Path(args['input'].filename)
        new_filepath = Path(configs.DATA_PATH, args['input'].original_filename)
        shutil.copy(tmp_filepath, new_filepath)
        npy_paths = [new_filepath]
    except Exception as e:
        raise HTTPError(e)

    print("Predicting on image(s):\n", npy_paths)
    print("Inference starting with the settings:")
    for k, v in args.items():
        print(f"\t'{k}': {v}")

    # infer on image(s)
    result = []
    for npy_path in npy_paths:
        out_name = npy_path.stem + "_score" + str(args['threshold']) + ".png"
        out_path = Path(args['out_dir'], npy_path.parent.name, out_name)

        infer_cmd = list(filter(None, [
            "/bin/bash", str(Path(configs.API_PATH, 'scripts',
                                  'execute_inference.sh')),
            "--input", str(npy_path),
            "--config-file", args['config_file'],
            "--ckp-file", args['checkpoint_file'],
            "--score-threshold", str(args['threshold']),
            "--channel", args['colour_channel'],
            # "--out-dir", str(out_path)
        ]))

        run_subprocess(command=infer_cmd, process_message="inference",
                       limit_gb=configs.LIMIT_GB,
                       timeout=10000)

        result.append(str(out_path))
        print(f'Inference result was saved to {out_path}')

    # delete temporary files if webargs browsing field was used
    if 'new_filepath' in locals():
        new_filepath.unlink()
        tmp_filepath.unlink()

    return result


def collect_image_paths(input_path: Path):
    """
    Function to return a list of image paths with a .npy suffix.

    Args:
        input_path: Directory containing images or a single image path.

    Returns:
        img_paths: List containing all image paths.
    """
    suffix = ".npy"

    if input_path.is_dir():
        img_paths = input_path.rglob(f"*{suffix}")
        if not img_paths:
            raise ValueError(f"{input_path} is not a directory "
                             f"containing {suffix} files!")

    elif input_path.is_file():
        if input_path.suffix != suffix:
            raise ValueError(f"{input_path} is not a {suffix} file!")

        img_paths = [input_path]

    else:
        raise ValueError(f"{input_path} does not exist "
                         f"or isn't a viable directory.")

    return img_paths
