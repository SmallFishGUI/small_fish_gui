from bigfish.stack import read_image
from czifile import imread
from ..utils import check_parameter
import re
from aicsimageio import AICSImage
from typing import Optional, Tuple

class FormatError(Exception):
    pass


def open_image(full_path:str) :
    if full_path.endswith('.czi') : im = imread(full_path)
    else : im = read_image(full_path)

    reshape = []
    for axis in im.shape :
        if axis != 1 : reshape.append(axis)
    im = im.reshape(reshape)

    return im


def check_format(image, is_3D, is_multichannel) :
    shape = list(image.shape)
    dim = image.ndim - (shape[image.ndim - 1] == 1)
    if not dim == (2 + is_3D  + is_multichannel) :
        raise FormatError("Inconsistency in image format and parameters.")



def get_filename(full_path: str) :
    check_parameter(full_path=str)

    pattern = r'.*\/(.+)\..*$'
    if not full_path.startswith('/') : full_path = '/' + full_path
    re_match = re.findall(pattern, full_path)
    if len(re_match) == 0 : raise ValueError("Could not read filename from image full path.")
    if len(re_match) == 1 : return re_match[0]
    else : raise AssertionError("Several filenames read from path")

def get_voxel_size(filepath: str) -> Optional[Tuple[Optional[float], Optional[float], Optional[float]]]:
    """
    Returns voxel size in nanometers (nm) as a tuple (X, Y, Z).
    Any of the dimensions may be None if not available.
    /WARINING\ : the unit might not be nm
    """
    try:
        img = AICSImage(filepath)
        voxel_sizes = img.physical_pixel_sizes  # values in meters
        if voxel_sizes is None:
            return None
        x = voxel_sizes.X * 1e3 if voxel_sizes.X else None
        y = voxel_sizes.Y * 1e3 if voxel_sizes.Y else None
        z = voxel_sizes.Z * 1e3 if voxel_sizes.Z else None
        return (z, y, x)
    except Exception as e:
        print(f"Failed to read voxel size from {filepath}: {e}")
        return None