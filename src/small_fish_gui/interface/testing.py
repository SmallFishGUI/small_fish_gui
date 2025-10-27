import os
from typing import Optional, Tuple

IMAGE_PATH = "/home/floric/Documents/python/small_fish/images_test/Cropped_image_dapi_0.tif"

def get_voxel_size_0(filepath: str) -> Optional[Tuple[float, float, float]]:
    ext = os.path.splitext(filepath)[1].lower()
    
    if ext in ['.tif', '.tiff']:
        import tifffile
        with tifffile.TiffFile(filepath) as tif:
            tags = tif.pages[0].tags
            print(tags)
            x_res = tags.get('XResolution')
            y_res = tags.get('YResolution')
            if x_res and y_res:
                # Resolution is typically in (numerator, denominator)
                x = x_res.value[1] / x_res.value[0]
                y = y_res.value[1] / y_res.value[0]
                # Microscopy images may store pixel size in microns per pixel (invert if necessary)
                return (1.0, y, x)  # Z usually not in TIFF metadata, so set as 1.0 or None
            return None
    
    elif ext == '.czi':
        import czifile
        with czifile.CziFile(filepath) as czi:
            metadata = czi.metadata(raw=False)
            scaling = metadata.get('ImageDocument', {}).get('Metadata', {}).get('Scaling', {})
            try:
                x = float(scaling['Items']['Distance'][0]['Value']) * 1e6  # in microns
                y = float(scaling['Items']['Distance'][1]['Value']) * 1e6
                z = float(scaling['Items']['Distance'][2]['Value']) * 1e6 if len(scaling['Items']['Distance']) > 2 else 1.0
                return (x, y, z)
            except Exception:
                return None
    
    return None

from aicsimageio import AICSImage
from typing import Optional, Tuple

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
        return (x, y, z)
    except Exception as e:
        print(f"Failed to read voxel size from {filepath}: {e}")
        return None


res = get_voxel_size("/home/floric/Documents/python/small_fish/images_test/Cropped_image_dapi_0.tif")
print("RES : ",res)
