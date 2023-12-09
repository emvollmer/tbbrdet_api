"""
This file gathers some utility functions for all other scripts
(get_metadata, training and inference).
"""

from functools import wraps
from multiprocessing import Process
import logging
import os
import os.path as osp
import subprocess
from subprocess import TimeoutExpired
import time
import threading
import warnings
from pathlib import Path
import re
import sys
import shutil

from aiohttp.web import HTTPBadRequest, HTTPException, HTTPServerError

from tbbrdet_api import configs

logger = logging.getLogger('__name__')
logger.setLevel(configs.LOG_LEVEL)      # previously: logging.DEBUG

stop_thread = threading.Event()

class DiskSpaceExceeded(Exception):
    """Raised when disk space is exceeded."""
    pass


def _catch_error(f):
    """
    Decorate API functions to return an error as HTTPBadRequest,
    in case it fails.
    """

    @wraps(f)
    def wrap(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            raise HTTPBadRequest(reason=e)

    return wrap


def _fields_to_dict(fields_in):
    """
    Function to convert marshmallow fields to dict()
    """
    dict_out = {}
    for k, v in fields_in.items():
        param = {}
        param["default"] = v.missing
        param["type"] = type(v.missing)
        param["required"] = getattr(v, "required", False)

        v_help = v.metadata["description"]
        if "enum" in v.metadata.keys():
            v_help = f"{v_help}. Choices: {v.metadata['enum']}"
        param["help"] = v_help

        dict_out[k] = param

    return dict_out


def set_log(log_dir):
    logging.basicConfig(
        # level=logging.DEBUG,
        format='%(message)s',
        # dateformat='%a, %d %b %Y %H:%M:%S',
        filename=f"{log_dir}/train.log",
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(configs.LOG_LEVEL)
    # add the handler to the root logger
    logging.getLogger().addHandler(console)


def extract_zst(zst_folder: str = configs.DATA_PATH):
    """
    Extracting the files from the tar.zst files

    Args:
        zst_folder (Path): Path to folder containing .tar.zst files to extract

    Returns:
        limit_exceeded (Bool): Turns true if no more data is allowed to be extracted

    """
    log_disk_usage("Begin extracting .tar.zst files")
    
    for zst_path in Path(zst_folder).glob("**/*.tar.zst"):
        tar_command = ["tar", "-I", "zstd", "-xf",      # add a -v flag to -xf if you want the filenames
                       str(zst_path), "-C", str(configs.DATA_PATH)]
        
        run_subprocess(tar_command, process_message=f"unpacking '{zst_path.name}'", 
                       limit_gb=configs.DATA_LIMIT_GB, path_to_check=configs.DATA_PATH)

        # check if this .tar.zst file is in the config.DATA_PATH, if so delete it to save space
        if configs.DATA_PATH in zst_path.parents:
            logger.info(f"Removing .tar.zst file '{zst_path.name}' after extraction to save storage space.")
            zst_path.unlink()


def launch_cmd(logdir, port):
    subprocess.call(["tensorboard",
                     "--logdir", f"{logdir}",
                     "--port", f"{port}",
                     "--host", "0.0.0.0"]) # nosec B603, B607


def launch_tensorboard(logdir, port=6006):
    """
    Run Tensorboard on a separate Process on behalf of the user

    Parameters
    ==========
    * logdir: str, pathlib.Path
        Folder path to tensorboard logs.
    * port: int
        Port to use for the monitoring webserver.
    """
    subprocess.call(
        ["fuser", "-k", f"{port}/tcp"]  # kill any previous process in that port
    )
    p = Process(target=launch_cmd, args=(logdir, port))
    p.daemon = True  # Set the daemon property (Python 3.6 equivalent)
    p.start()


def ls_folders(directory: Path = configs.MODEL_PATH, pattern: str = "*latest.pth") -> list:
    """
    Utility to return a list of folders in a given directory that contain a file of a specific pattern.

    - local_model_folders = ls_folders(directory=configs.MODEL_PATH, pattern="*latest.pth")
    - remote_model_folders = ls_folders(directory=configs.REMOTE_MODEL_PATH, pattern="*latest.pth")

    Args:
        directory (Path): Path of the directory to scan
        pattern (str): The pattern to use for scanning

    Returns:
        list: list of relevant .pth file paths
    """
    logger.debug(f"Scanning through '{directory}' with pattern '{pattern}'")
    return sorted(set([str(d.parent) for d in Path(directory).rglob(pattern)]))


def get_weights_folder(data: dict):
    """Utility to get folder containing pretrained weights to use in resumed training 

    Args:
        data (dict): Arguments from fields.py (user inputs in swagger ui)
    Returns:
        Path to the folder containing pretrained weights
    """
    return Path(configs.REMOTE_MODEL_PATH, data['architecture'], data['train_from'], "pretrained_weights")


def copy_file(frompath: Path, topath: Path):
    """
    Copy a file (also to / from remote directory)

    Args:
        frompath (Path): The path to the file to be copied
        topath (Path): The path to the destination folder directory
    
    Raises:
        OSError: If the source isn't a directory
        FileNotFoundError: If the source file doesn't exist
    """
    frompath: Path = Path(frompath)
    topath: Path = Path(topath)

    if Path(topath, frompath.name).exists():
        print(f"Skipping copy of '{frompath}' as the file already exists in '{topath}'!")   # logger.info
    else:
        try:
            print(f"Copying '{frompath}' to '{topath}'...") # logger.info
            topath = shutil.copy(frompath, topath)
        except OSError as e:
            print(f'Directory not copied because {frompath} directory not a directory. Error: %s' % e)
        except FileNotFoundError as e:
            print(f'Error in copying from {frompath} to {topath}. Error: %s' % e)


def run_subprocess(command: list, process_message: str, limit_gb: int = configs.LIMIT_GB, 
                   path_to_check: Path = configs.BASE_PATH, timeout: int = 500):
    """
    Function to run a subprocess command.

    Args:
        command (list): Command to be run.
        process_message (str): Message to be printed to the console.
        limit_gb (int): Limit on the amount of disk space available on the node.
        timeout (int): Time limit by which process is limited (in case it gets stuck).

    Raises:
        TimeoutExpired: If timeout exceeded
        DiskSpaceExceeded: If disk space limit exceeded
        Exception: If any other error occured
    """
    log_disk_usage(f"Begin: {process_message}")
    str_command = " ".join(command)

    # get absolute limit by comparing to remaining available space on node
    limit_gb = check_available_node_space(limit_gb)

    if get_disk_usage(folder=path_to_check) > limit_gb:
        log_disk_usage(f"FAILED: {process_message}")
        logger.error(f"Disk space limit of {limit_gb} GB exceeded before {process_message} subprocess can start!")
        raise DiskSpaceExceeded(f"Disk space limit of {limit_gb} GB exceeded "
                                f"before {process_message} subprocess can start!")
        return

    try:
        # monitor disk space usage in the background
        monitor_thread = threading.Thread(target=monitor_disk_space,
                                          args=(limit_gb, path_to_check, ), daemon=True)
        monitor_thread.start()
        print(f"=================================\n"
              f"Running {process_message} command:\n'{str_command}'\n"
              f"=================================")    # logger.info

        process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,  # Capture stdout
                stderr=subprocess.PIPE,  # Capture stderr
                universal_newlines=True,  # Return strings rather than bytes
        )
        return_code = process.wait(timeout=timeout)

        if stop_thread.is_set():
            log_disk_usage(f"FAILED: {process_message}")
            raise DiskSpaceExceeded(f"Disk space exceeded during {process_message} "
                                    f"while running\n'{str_command}'\n")

        if return_code == 0:
            log_disk_usage(f"Finished: {process_message}")
        else:
            _, err = process.communicate()
            print(f"Error while running '{str_command}' for {process_message}. "
                  f"Terminated with return code {return_code}.")    # logger.error
            process.terminate()
            raise HTTPException(reason=err) # here it works for some reason without TypeError??...

    except TimeoutExpired:
        process.terminate()
        logger.error(f"Timeout during {process_message} while running\n'{str_command}'\n"
                     f"{timeout} seconds were exceeded.")
        raise
        # NOTE: can't do "raise HTTPServerError(reason=f"Timeout during {process_message}. 
        # {timeout} seconds were exceeded.")" because it causes a TypeError: __init__ required ...

    except DiskSpaceExceeded as e:
        process.terminate()
        logger.error(str(e))
        raise
        # NOTE: can't do "raise HTTPServerError(reason=str(e))" because it causes a TypeError: __init__ required ..

    return


def monitor_disk_space(limit_gb: int, path_to_check: Path):
    """
    Thread function to monitor disk space and check the current usage doesn't exceed 
    the defined limit.

    Raises:
        DiskSpaceExceeded: If available disk space was exceeded during threading.
    """

    while True:
        time.sleep(3)

        stored_gb = get_disk_usage(Path(path_to_check))

        if stored_gb >= limit_gb:
            stop_thread.set()
            sys.exit()


def check_available_node_space(limit_gb: int = configs.LIMIT_GB):
    """
    Check overall data limit on node and redefine limit if necessary.

    Args:
        limit_gb: user defined disk space limit (in GB)
    
    Returns:
        limit (gb) that should not be exceeded by this deployment, taking into account the overall available node space
    """
    try:
        # get available space on entire node (with additional buffer of 3 GB)
        available_gb = int(subprocess.getoutput("df -h | grep 'overlay' | awk '{print $4}'").split("G")[0])
        available_gb = max(available_gb - 3, 0)
    except ValueError as e:
        logger.error(f"ValueError: Node disk space not readable. Using provided limit of {limit_gb} GB.")
        raise HTTPException(reason=str(e)) from e

    current_gb = get_disk_usage()
    leftover_gb = round(limit_gb - current_gb, 2)
    if leftover_gb < available_gb:
        return limit_gb
    else:
        new_limit_gb = round(current_gb + available_gb, 2)
        print(f"Available disk space on node ({available_gb} GB) is less than the leftover deployment "
              f"space ({leftover_gb} GB) until the user-defined limit ({limit_gb} GB) is reached. "
              f"Limit will be reduced to {new_limit_gb} GB.")       # logger.warning()
        return new_limit_gb


def get_disk_usage(folder: Path = configs.BASE_PATH):
    """Get the current amount of GB (rounded to two decimals) stored in the provided folder.
    """
    return round(sum(f.stat().st_size for f in folder.rglob('*') if f.is_file()) / (1024 ** 3), 2)


def log_disk_usage(process_message: str):
    """Log used disk space to the terminal with a process_message describing what has occurred.
    """
    print(f"{process_message} --- Repository currently takes up {get_disk_usage()} GB.")   # logger.info(...)


if __name__ == '__main__':
    print("Remote directory path:", configs.REMOTE_MODEL_PATH)
