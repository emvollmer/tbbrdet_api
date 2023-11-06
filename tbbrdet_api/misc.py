"""
This file gathers some functions that have proven to be useful
across projects. They are not strictly need for integration
but users might want nevertheless to take advantage from them.
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

from aiohttp.web import HTTPBadRequest, HTTPException

from tbbrdet_api import configs

logger = logging.getLogger('__name__')
logger.setLevel(logging.DEBUG)


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
    # TODO: Add logger parameters to config and use those values for setLevel
    logging.basicConfig(
        # level=logging.DEBUG,
        format='%(message)s',
        # dateformat='%a, %d %b %Y %H:%M:%S',
        filename=f"{log_dir}/train.log",
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # add the handler to the root logger
    logging.getLogger().addHandler(console)


def extract_zst():
    """
    Extracting the files from the tar.zst files

    Args:
        file_path (Path): Path to .zst file to extract

    Returns:
        limit_exceeded (Bool): Turns true if no more data is allowed to be extracted

    """
    log_disk_usage("Begin extracting .tar.zst files")
    
    for zst_path in Path("/storage/tbbrdet/datasets").glob("**/*.tar.zst"):
        tar_command = ["tar", "-I", "zstd", "-xf",      # add a -v flag to -xf if you want the filenames
                       str(zst_path), "-C", str(configs.DATA_PATH)]
        
        run_subprocess(tar_command, process_message=f"unpacking '{zst_path.name}'", limit_gb=configs.DATA_LIMIT_GB)


def launch_cmd(logdir, port):
    subprocess.call(["tensorboard",
                     "--logdir", f"{logdir}",
                     "--port", f"{port}",
                     "--host", "0.0.0.0"])


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


def ls_local():
    """
    Utility to return a list of current models / ckp_pretrain_pth .pth files stored in the local folders
    configured for cache.

    Returns:
        list: list of relevant .pth file paths
    """
    logger.debug("Scanning at: %s", configs.MODEL_PATH)
    local_paths = Path(configs.MODEL_PATH).glob("**/*.pth")
    # to include only the last 4 path elements, change to "str(Path(*entry.parts[-4:]))"
    return [str(entry) for entry in local_paths
            if any(w in str(entry) for w in ["best", "weight"])]


def ls_remote(remote_directory: Path = configs.REMOTE_MODEL_PATH):
    """
    Utility to return a list of current models (.pth files)
    stored in the remote folder.

    Returns:
        list: List of .pth files for models / checkpoint files within the remote directory
    """
    remote_directory = str(remote_directory)
    # get recursive list of all files and folders in the remote directory path
    command = ['rclone', 'lsf', remote_directory, "-R", "--absolute"]
    result = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = result.communicate()

    if result.returncode == 0:
        directories = stdout.decode().splitlines()
        if remote_directory == str(configs.REMOTE_MODEL_PATH):
            model_paths = [remote_directory + d.rstrip("/") for d in directories
                           if any(w in d for w in ["best", "weight"])
                           if d.endswith(".pth")]
            return model_paths
        else:
            return [remote_directory + d.rstrip("/") for d in directories]
    else:
        logger.error("Error executing rclone command:", stderr.decode())
        return []


def get_model_paths(paths: list, pretrain: bool = False):
    """
    Function to reduce list of paths to specific model checkpoints
    Args:
        paths (list): list of all paths
        pretrain (bool): If True, filters .pth ckp files for officially pretrained model weights
                        If False, filters .pth ckp files for other models, f.e. previously trained

    Returns:
        list: only folders that contain model checkpoint paths ("best...pth"/"latest.pth")
    """
    if pretrain:
        return [d.rstrip("/") for d in paths if "weight" in d]
    else:
        return list(set([osp.dirname(d.rstrip("/")) for d in paths if "weight" not in d]))


def download_folder_from_nextcloud(remote_dir, filetype, check=".pth"):
    """
    Downloads the remote folder from nextcloud to a specified checkpoint path.
    todo: Remove this function and just use copy_rclone instead!!!

    Args:
        remote_dir (str): The path to the model / weights folder in NextCloud
        filetype (str): What is being copied (model, weights, data?)
        check (str): String with which to check if correct files were downloaded

    Returns:
       local_model_dir (str): The path to the local folder to which the model was copied

    Raises:
       Exception: If no files were copied to the checkpoint directory
                  after downloading the model from the URL.

    """
    logger.debug(f"Scanning at: {remote_dir}")

    # get local folder, which will include the "<model-name>_coco-pretrain" / _scratch folder
    local_model_dir = Path(configs.MODEL_PATH, check_train_from(remote_dir))
    folder_to_copy = Path(remote_dir).name
    local_dir = Path(local_model_dir, folder_to_copy)

    if folder_to_copy not in local_model_dir.iterdir():
        logger.info(f'Downloading the {filetype} checkpoints from Nextcloud')

        copy_rclone(frompath=remote_dir, topath=local_dir)
        # todo: ensure this works as planned, because in EGI tut
        #  remote_folder=/../predict_model_dir, local_folder=configs.MODEL_PATH (no
        #  predict_model_dir) and rclone doesn't copy the src folder!
        #  https://rclone.org/commands/rclone_copy/

        if any(check in d for d in os.listdir(local_dir)):
            raise Exception(f"Folder with {filetype} wasn't copied to '{local_dir}', due to "
                            f" missing {filetype} path files in '{remote_dir}'!")

        logger.info(f"The remote {filetype} folder '{remote_dir}' was copied to '{local_dir}'")

    else:
        logger.info(f"Skipping download of '{remote_dir}' as the "
                    f"folder with {filetype} already exists in '{local_dir}'!")

    return local_dir


def copy_rclone(frompath: Path, topath: Path, timeout=600):
    """
    Function to download a directory/file using rclone.
    NOTE: This will only work if we're copying to a local directory.
    Args:
        frompath (Path): Path from which will be downloaded f.e. remote path.
        topath (Path): Folder to which downloaded paths with be saved f.e. local path.
        timeout (int): Time limit by which process should be completed

    Returns:
        None
    """
    # check if the frompath is a folder or file, if it's a folder get filenames from within
    if not frompath.suffix:
        if "rshare" in str(frompath):
            file_names = [Path(p).name for p in ls_remote(frompath)]
        else:
            file_names = [p.name for p in frompath.iterdir()]
    else:
        file_names = [frompath.name]

    # check if folder/file to be copied already exists in destination directory (topath)
    if file_names and not set(file_names).issubset(os.listdir(topath)):
        logger.info(f"Copying '{frompath}' to '{topath}'...")  
        command = ['rclone', 'copy', str(frompath), str(topath)]
        run_subprocess(command, process_message="copying with rclone")

        # check if folder/file now exists in destination directory (topath)
        if not set(file_names).issubset(os.listdir(topath)):
            logger.error(f"An error must have occured during copying, as "
                         f"no file(s) '{file_names}' exist(s) in '{topath}'!")
            raise

        logger.info(f"'{file_names}' was/were successfully copied to '{topath}'!")
    else:
        logger.warning(f"Skipping download of '{frompath}' as the "
                       f"file(s) '{file_names}' already exist(s) in '{topath}'!")


def run_subprocess(command: list, process_message: str, limit_gb: int = configs.LIMIT_GB, timeout: int = 500):
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
    # get absolute limit by comparing to remaining available space on node
    limit_gb = check_node_disk_limit(limit_gb)

    if get_disk_usage() / (1024 ** 3) > limit_gb:
        raise DiskSpaceExceeded(f"Disk space limit of {str(limit_gb)} GB already exceeded!")
        log_disk_usage(f"Failed: {process_message}")
        return

    try:
        # monitor disk space usage in the background
        monitor_thread = threading.Thread(target=monitor_disk_space,
                                          args=(limit_gb,), daemon=True)
        monitor_thread.start()
        print(f"Running subprocess command: {command}")    # logger.debug
        with subprocess.Popen(
            command,
            stdout=subprocess.PIPE,  # Capture stdout
            stderr=subprocess.PIPE,  # Capture stderr
            universal_newlines=True,  # Return strings rather than bytes
        ) as process:
            try:
                outs, errs = process.communicate(None, timeout)
                if errs:
                    raise Exception(errs)
            except TimeoutExpired as exc:
                process.kill()
                logger.error(f"Timeout while {process_message}: %s", exc, exc_info=True)
                raise HTTPException(reason=exc) from exc
            except Exception as exc:  # pylint: disable=broad-except
                process.kill()
                logger.error(f"Error while {process_message}: %s", exc, exc_info=True)
                raise HTTPException(reason=exc) from exc

    except DiskSpaceExceeded as e:
        logger.error(f"Disk space exceeded while {process_message}: %s", e, exc_info=True)
        print(f"Disk space limit exceeded: {str(e)}")    # logger.error
        # NOTE: This doesn't prevent the program from continuing to run!!!
        raise

    log_disk_usage(f"Finished: {process_message}")
    return


def monitor_disk_space(limit_gb: int = configs.LIMIT_GB):
    """
    Thread function to monitor disk space and check the current usage doesn't exceed 
    the defined limit.

    Raises:
        DiskSpaceExceeded: If available disk space was exceeded during threading.
    """
    limit_bytes = limit_gb * (1024 ** 3)  # convert to bytes
    while True:
        time.sleep(3)

        stored_bytes = get_disk_usage()

        if stored_bytes >= limit_bytes:
            raise DiskSpaceExceeded(f"Exceeded maximum allowed disk space of {limit_gb} GB "
                                    f"for '{configs.BASE_PATH}' folder.")


def check_node_disk_limit(limit_gb: int = configs.LIMIT_GB):
    """
    Check overall data limit on node and redefine limit if necessary.

    Args:
        limit_gb: user defined disk space limit (in GB)
    
    Returns:
        available GB on node
    """
    try:
        # get available space on entire node
        available_gb = int(subprocess.getoutput("df -h | grep 'overlay' | awk '{print $4}'").split("G")[0])
    except ValueError as e:
        logger.info(f"ValueError: Node disk space not readable. Using provided limit of {limit_gb} GB.")
        available_gb = limit_gb

    if available_gb >= limit_gb:
        return limit_gb
    else:
        logger.warning(f"Available disk space on node ({available_gb} GB) is less than the user "
                       f"defined limit ({limit_gb} GB). Limit will be reduced to {available_gb} GB.")
        return available_gb


def get_disk_usage(folder: Path = configs.BASE_PATH):
    """Get the current amount of bytes stored in the provided folder.
    """
    return sum(f.stat().st_size for f in folder.rglob('*') if f.is_file())


def log_disk_usage(process_message: str):
    """Log used disk space to the terminal with a process_message describing what has occurred.
    """
    print(f"{process_message}: Repository currently takes up {round(get_disk_usage() / (1024 ** 3), 2)} GB.")   # logger.info(...)


def check_train_from(directory):
    """
    Check if directory contains information on training from scratch or with coco ckp_pretrain_pth
    Args:
        directory (str): The path to be checked

    Returns:
        val (str): Either "mask_rcnn_swin-t_coco-pretrained" or "mask_rcnn_swin-t_scratch"

    """
    for val in configs.settings['train_from'].values():
        if val in directory:
            return val


def get_pth_to_resume_from(directory, priority):
    """
    Define .pth file name from which to resume from.
    Resuming is prioritized according to the "priority" list

    Args:
        directory (str): Directory to search through for .pth file names to be chosen from
        priority (list): Strings to be matched in priority order, f.e. ['latest', 'best', 'epoch']

    Returns:
        pth_name (str/None): Selected path file name to resume from
    """
    pth_names = [i for i in os.listdir(directory) if i.endswith(".pth")]
    # sort list of path names so that "epoch_X.pth" are in descending numerical order
    sorted_pth_names = sorted(pth_names,
                              key=lambda f: int(re.search(r'\d+', f).group())
                              if re.search(r'\d+', f) else float('inf'), reverse=True)

    # we prioritize resuming according to the provided priority and in descending order
    for option in priority:
        for pth in sorted_pth_names:
            if option in pth:
                return pth
    # return none if none of the "priority" list strings were found at all
    return None


if __name__ == '__main__':
    print("Remote directory path:", configs.REMOTE_MODEL_PATH)
