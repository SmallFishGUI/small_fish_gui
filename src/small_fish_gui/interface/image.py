import re
import tifffile
from bigfish.stack import read_image
from czifile import imread
from czifile import CziFile
from ..utils import check_parameter
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
    /WARINING\\ : the unit might not be nm
    """
    try:
        if filepath.endswith('.czi'):
            with CziFile(filepath) as czi:
                metadata = czi.metadata()  # returns XML metadata
                # try to parse voxel sizes from XML
                import xml.etree.ElementTree as ET
                root = ET.fromstring(metadata)
                scaling_distance = root.findall('.//Scaling//Items//Distance//Value')
                if len(scaling_distance) in [2,3] :
                    for scale in scaling_distance :
                        res = [float(scale.text) * 1e9 for scale in scaling_distance] #m to nm
                    res.reverse()
                    return tuple(res)
                else :
                    raise Exception("Couln't find voxel size on xml metadata")

        elif filepath.endswith(('.tif', '.tiff')):
            with tifffile.TiffFile(filepath) as tif:
                ij_meta = tif.imagej_metadata
                page = tif.pages[0]  # first image page
                # X/Y resolution as (numerator, denominator)
                xres = page.tags['XResolution'].value
                yres = page.tags['YResolution'].value
                # ResolutionUnit: must be 'nm' for this calculation
                res_unit = ij_meta.get("unit")

                if res_unit and str(res_unit) != 'nm':
                    xy_size = 1 / (xres[0] / xres[1]) * 1e3 #um to nm
                elif res_unit and str(res_unit) != 'NONE':
                    xy_size = 1 / (xres[0] / xres[1]) #um to nm
                else:
                    xy_size = None

                # Z spacing from ImageJ metadata
                if res_unit and str(res_unit) != 'nm':
                    z_size = ij_meta.get('spacing', None) * 1e3 
                else :
                    z_size = ij_meta.get('spacing', None)

                return (z_size,xy_size, xy_size )
    except Exception as e:
        print(f"Failed to read voxel size from {filepath}: {e}")
        return None