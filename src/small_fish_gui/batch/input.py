"""
Submodule handling handling files and filenames in batch mode.
"""

import os
import bigfish.stack as stack
import czifile as czi
import numpy as np
from .integrity import check_file

def open_image(filename:str) :

    if filename.endswith('.czi') :
        image = czi.imread(filename)
    else :
        image = stack.read_image(filename)

    image = np.squeeze(image)

    return image

def get_images(filename:str) :
    """returns filename if is image else return None"""

    supported_types = ('.tiff', '.tif', '.png', '.czi')
    if filename.endswith(supported_types) : 
        return [filename]
    else :
        return None

def get_files(path) :

    filelist = os.listdir(path)
    filelist = list(map(get_images,filelist))

    while None in filelist : filelist.remove(None)

    return filelist

def extract_files(filenames: list) :
    return sum(filenames,[])

def load(
        batch_folder:str,
) :
    if not os.path.isdir(batch_folder) :
        print("Can't open {0}".format(batch_folder))
        files_values = [[]]
        last_shape = None
        dim_number = 0
    else :
        files_values = get_files(batch_folder)
        if len(files_values) == 0 :
            last_shape = None
            dim_number = 0
        else :
            first_filename = files_values[0][0]
            last_shape = check_file(batch_folder + '/' + first_filename)
            dim_number = len(last_shape)

    
    return files_values, last_shape, dim_number