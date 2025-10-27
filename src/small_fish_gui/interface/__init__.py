"""
This code handles the exchange between the computer and the code. That is to say opening and saving data.
"""

from .image import open_image
from .image import get_filename 
from .image import check_format
from .image import FormatError
from .image import get_voxel_size

from .inoutput import write_results