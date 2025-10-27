"""
Submodule handling all parameters check, asserting functions and pipeline will be able to run.
"""

import os
import czifile as czi
import bigfish.stack as stack
import numpy as np
import FreeSimpleGUI as sg

from ..hints import pipeline_parameters
from ..pipeline._preprocess import check_integrity, convert_parameters_types, ParameterInputError, _check_segmentation_parameters
from ..pipeline.segmentation import _cast_segmentation_parameters

def check_file(filename:str) :

    if filename.endswith('.czi') :
        image = czi.imread(filename)
    else :
        image = stack.read_image(filename)

    image = np.squeeze(image)

    return image.shape

def sanity_check(
        filename_list: list, 
        batch_folder : str,
        window : sg.Window, 
        progress_bar: sg.ProgressBar,
        ) :
    
    filenumber = len(filename_list)
    if filenumber == 0 :
        print("No file to check")
        progress_bar.update(current_count= 0, bar_color=('gray','gray'))
        return None
    else :
        print("{0} files to check".format(filenumber))
        progress_bar.update(current_count=0, max= filenumber)
        ref_shape = check_file(batch_folder + '/' + filename_list[0])

        print("Starting sanity check. This could take some time...")
        for i, file in enumerate(filename_list) :
            progress_bar.update(current_count= i+1, bar_color=('green','gray'))
            shape = check_file(batch_folder + '/' + file)

            if len(shape) != len(ref_shape) : #then dimension missmatch
                print("Different number of dimensions found : {0}, {1}".format(len(ref_shape), len(shape)))
                progress_bar.update(current_count=filenumber, bar_color=('red','black'))
                window= window.refresh()
                break

            window= window.refresh()

        print("Sanity check completed.")
        return None if len(shape) != len(ref_shape) else shape

def check_channel_map_integrity(
        maping:dict, 
        shape: tuple,
        expected_dim : int
        ) :
    
    #Check integrity
    try :
        channels_values = np.array(list(maping.values()), dtype= int)
    except Exception :
        res = False
        sg.popup("Incorrect values for channel mapping. ({0})".format(maping.values()))
        return res

    total_channels = len(maping)
    unique_channel = len(np.unique(channels_values))
    res= True


    if expected_dim != total_channels :
        sg.popup("Image has {0} dimensions but {1} were mapped.".format(expected_dim, total_channels))
        res = False
    if total_channels != unique_channel :
        sg.popup("{0} channel(s) are not uniquely mapped.".format(total_channels - unique_channel))
        res = False
    if not all(channels_values < len(shape)):
        sg.popup("Channels values out of range for image dimensions.\nPlease select dimensions from {0}".format(list(range(len(shape)))))
        res = False

    return res

def check_segmentation_parameters(
        values,
        shape,
        is_multichannel,
) :
    values = _cast_segmentation_parameters(values=values)
    try :
        _check_segmentation_parameters(
            user_parameters=values,
            shape=shape,
            is_multichannel=is_multichannel
        )
    except ParameterInputError as e: 
        segmentation_is_ok = False
        sg.popup_error(e)
    else :
        segmentation_is_ok = True
    
    return segmentation_is_ok, values

def check_detection_parameters(
        values : pipeline_parameters,
        do_dense_region_deconvolution,
        do_clustering,
        is_multichannel,
        is_3D,
        map_,
        shape
) :
    
    values['dim'] = 3 if is_3D else 2
    values = convert_parameters_types(values)
    try :
        check_integrity(
            values=values,
            do_dense_region_deconvolution=do_dense_region_deconvolution,
            do_clustering=do_clustering,
            multichannel=is_multichannel,
            segmentation_done= values['do_segmentation'],
            map_=map_,
            shape=shape
        )
    except ParameterInputError as e: 
        detection_is_ok = False
        sg.popup_error(e)
    else :
        detection_is_ok = True
    
    return detection_is_ok, values

def check_output_parameters(values) :
    is_output_ok = True

    #Output folder
    output_folder = values.get('output_folder')
    if not os.path.isdir(output_folder) :
        sg.popup("Incorrect output folder selected")
        is_output_ok=False

    #Batch name
    original_name = values['batch_name']
    loop=1
    values['batch_name'] = values['batch_name'].replace(' ','_')
    while os.path.isdir(output_folder + '/' + values['batch_name']) :
        values['batch_name'] = original_name + '_{0}'.format(loop)
        loop+=1
    if len(values['batch_name']) == 0 : is_output_ok = False

    #extension
    if values['csv'] or values['xlsx'] or values['feather'] :
        pass
    else :
        sg.popup("Select at least one data format for output.")
        is_output_ok=False


    return is_output_ok, values