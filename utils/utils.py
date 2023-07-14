import os
import pickle
import hashlib
import urllib.request
from contextlib import closing
import tarfile
import shutil
from typing import *
import numpy as np
import torch


Array = Union[np.ndarray, torch.Tensor]


def create_subdirs(root_dir: str, subdirs: Sequence[str]):
    """Creates subdirectories of a root directory.

    Args:
        root_dir (str): Directory which will contain all subdirectories.
        subdirs (Sequence[str]): Sequence of names for subdirectories.
    """
    for dirname in subdirs:
        subdir = os.path.join(root_dir, dirname)
        os.makedirs(subdir, exist_ok=True)


def dir_size(dir: str) -> int:
    """Traverses a directory and determines its size.

    Args:
        dir (str): Path to directory for computing size.

    Returns:
        int: Size of dir.
    """

    size = 0
    for path, _, files in os.walk(dir):
        for f in files:
            fp = os.path.join(path, f)
            size += os.path.getsize(fp)

    return size


def md5_check(file_path: str, true_md5: str) -> bool:
    """Compares the MD5 of the file to the true_md5.

    Args:
        file_path (str): Path to the file to check.
        true_md5 (str): True MD5 for check.

    Returns:
        bool: Whether the MD5 of the file matches true_md5
    """

    # Check if file actual exists
    if not os.path.isfile(file_path):
        return False

    with open(file_path, 'rb') as f:
        data = f.read()
        return hashlib.md5(data).hexdigest() == true_md5
    

def download_file(file_url: str, destination: str, md5: Optional[str] = None) -> str:
    """Downloads a file.

    Args:
        file_url (str): Url from which to retrieve file.
        destination (str): Destination dir for saving.
        md5: (str, optional): MD5 of the file that should be downloaded.
        
    Returns:
        str: The path to the downloaded file.
    """

    save_path = os.path.join(destination, os.path.basename(file_url))
    
    # Check if file exists and has correct MD5.
    if os.path.isfile(save_path):
        if md5 and md5_check(save_path, md5):
            return save_path
        else:
            return save_path

    # Download the file from the url.
    with closing(urllib.request.urlopen(file_url)) as r:
        with open(save_path, "wb") as f:
            shutil.copyfileobj(r, f)
        
    # Double check that we have the right file.            
    if md5 and not md5_check(save_path, md5):
        raise ValueError("MD5 of downloaded file does not match provided MD5. Double check the provided url and MD5.")
                
    return save_path


def extract(zip_file: str) -> str:
    """ Extracts a zipped file.
    
    Args:
        zip_file (str): Path to zipped file.
        
    Returns:
        str: The path with the unzipped contents
    
    Modified from torchdrug/utils/file.py
    """
    
    zip_name, extension = os.path.splitext(zip_file)
    if zip_name.endswith(".tar"):
        extension = ".tar" + extension
        zip_name = zip_name[:-4]

    save_path = os.path.dirname(zip_file)
    tar = tarfile.open(zip_file, "r")
    members = tar.getnames()
    save_files = [os.path.join(save_path, _member) for _member in members]
    for _member, save_file in zip(members, save_files):
        if tar.getmember(_member).isdir():
            os.makedirs(save_file, exist_ok=True)
            continue
        os.makedirs(os.path.dirname(save_file), exist_ok=True)
        if not os.path.exists(save_file) or tar.getmember(_member).size != os.path.getsize(save_file):
            with tar.extractfile(_member) as fin, open(save_file, "wb") as fout:
                shutil.copyfileobj(fin, fout)
                
    if len(save_files) == 1:
        return save_files[0]
    else:
        return save_path
            
    
def robust_norm(array: Array, axis: int = -1, l_norm: float = 2, eps: float = 1e-8) -> Array:
    """Computes robust l-norm of vectors.

    Args:
        array (Array): Array containing vectors to compute norm.
        axis (int, optional): Axis of array to norm. Defaults to -1.
        l_norm (float, optional): Norm-type to perform. Defaults to 2.
        eps (float, optional): Epsilon for robust norm computation. Defaults to 1e-8.
        
    Returns:
        Array: Norm of axis of array
    """
    if isinstance(array, np.ndarray):
        return (np.sum(array ** l_norm, axis=axis) + eps) ** (1 / l_norm)
    else:
        return (torch.sum(array ** l_norm, dim=axis) + eps) ** (1 / l_norm)


def robust_normalize(array: Array, axis: int = -1, l_norm: float = 2, eps: float = 1e-8) -> Array:
    """Computes robust l-normalization of vectors.

    Args:
        array (Array): Array containing vectors to normalize.
        axis (int, optional): Axis of array to normalize. Defaults to -1.
        l_norm (float, optional): Norm-type to perform. Defaults to 2.
        eps (float, optional): Epsilon for robust norma computation. Defaults to 1e-8.
        
    Returns:
        Array: Normalized array
    """
    if isinstance(array, np.ndarray):
        return array / np.expand_dims(robust_norm(array, axis=axis, l_norm=l_norm, eps=eps), axis=axis)
    else:
        return array / robust_norm(array, axis=axis, l_norm=l_norm, eps=eps).unsqueeze(axis)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def get_mean_clashscore(dir, clashscore_file="clashscore_scores.txt"):
    with open(os.path.join(dir, clashscore_file), 'r') as f:
        lines = f.readlines()
        
    scores = [float(line.strip().split(' ')[-1]) for line in lines[1:] if line.strip()]
    return np.mean(scores)


def get_rotamer_evals(dir, rotamer_eval_file="rotamer_evals.txt"):
    with open(os.path.join(dir, rotamer_eval_file), 'r') as f:
        lines = f.readlines()
        
    eval_lines = [line.strip() for line in lines if '[eval]' in line.strip()]
    evals = [line.split('=>')[-1].strip() for line in eval_lines]
    
    rotamer_evals = {
        'num_residues': len(evals),
        'outliers': evals.count('OUTLIER'),
        'allowed': evals.count('Allowed'),
        'favored': evals.count('Favored'),
    }
    
    return rotamer_evals


def get_packing_stats(dir, stats_file="packing_stats.pkl"):
    with open(os.path.join(dir, stats_file), 'rb') as f:
        stats = pickle.load(f)
        
    return stats