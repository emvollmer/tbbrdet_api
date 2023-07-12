"""
This file gathers some functions that have proven to be useful
across projects. They are not strictly need for integration
but users might want nevertheless to take advantage from them.
"""

from functools import wraps
from multiprocessing import Process
import logging
import os
import subprocess
import warnings

from aiohttp.web import HTTPBadRequest

from tbbrdet_api import configs

logger = logging.getLogger('__name__')


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


def mount_nextcloud(frompath, topath):
    """
    Mount a NextCloud folder in your local machine or viceversa.

    Example of usage:
        mount_nextcloud('rshare:/data/images', 'my_local_image_path')

    Parameters
    ==========
    * frompath: str, pathlib.Path
        Source folder to be copied
    * topath: str, pathlib.Path
        Destination folder
    """
    command = ["rclone", "copy", f"{frompath}", f"{topath}"]
    result = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, error = result.communicate()
    if error:
        warnings.warn(f"Error while mounting NextCloud: {error}")
    return output, error


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
    subprocess.run(
        ["fuser", "-k", f"{port}/tcp"]  # kill any previous process in that port
    )
    p = Process(target=launch_cmd, args=(logdir, port), daemon=True)
    p.start()


def ls_local():
    """
    Utility to return a list of current models stored in the local folders
    configured for cache.

    Returns:
        A list of strings.
    """
    logger.debug("Scanning at: %s", configs.MODEL_DIR)
    dirscan = os.scandir(configs.MODEL_DIR)
    return [entry.name for entry in dirscan if entry.is_dir()]


def ls_remote():
    """
    Utility to return a list of current backbone models stored in the
    remote folder configured in the backbone url.

    Returns:
        A list of strings.
    """
    remote_directory = configs.REMOT_PATH
    return list_directories_with_rclone('rshare', remote_directory)


def list_directories_with_rclone(remote_name, directory_path):
    """
    Function to list directories within a given directory in Nextcloud using rclone.

    Args:
        remote_name (str): Name of the configured Nextcloud remote in rclone.
        directory_path (str): Path of the parent directory to list the directories from.

    Returns:
        list: List of directory names within the specified parent directory.
    """

    command = ['rclone', 'lsf', remote_name + ':' + directory_path]
    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode == 0:
        directory_names = result.stdout.splitlines()
        directory_names = [d.rstrip("/") for d in directory_names if d[0].isdigit()]
        return directory_names
    else:
        print("Error executing rclone command:", result.stderr)
        return []


def download_model_from_nextcloud(timestamp):
    """
    Downloads the final model from nextcloud to a specified checkpoint path.

    Args:
        timestamp (str): The timestamp of the model on Nextcloud.

    Returns:
       None

    Raises:
       Exception: If no files were copied to the checkpoint directory after downloading the model from the URL.

    """
    logger.debug("Scanning at: %s", timestamp)
    logger.debug("Scanning at: %s", configs.REMOT_PATH)
    local_path = configs.MODEL_DIR
    ckpt_path = os.path.join(local_path, timestamp)

    if timestamp not in os.listdir(local_path):
        print('downloading the chekpoint from nextcloud')
        remote_directory = configs.REMOT_PATH
        model_path = os.path.join(remote_directory, timestamp)
        download_directory_with_rclone('rshare', model_path, local_path)

        if 'best_model.pth' not in os.listdir(ckpt_path):
            raise Exception(f"No files were copied to {ckpt_path}")

        print(f"The model for {timestamp} was copied to {ckpt_path}")

    else:
        print(f"Skipping download for {timestamp} as the model already exists in {ckpt_path}")


def download_directory_with_rclone(remote_name, remote_directory, local_directory):
    """
    Function to download a directory using rclone.

    Args:
        remote_name (str): Name of the configured remote in rclone.
        remote_directory (str): Path of the remote directory to be downloaded.
        local_directory (str): Path of the local directory to save the downloaded files.

    Returns:
        None
    """

    command = ['rclone', 'copy', remote_name + ':' + remote_directory, local_directory]
    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode == 0:
        print("Directory downloaded successfully.")
    else:
        print("Error executing rclone command:", result.stderr)


if __name__ == '__main__':
    print("Remote directory path:", configs.REMOT_PATH)