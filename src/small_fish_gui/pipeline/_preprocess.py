import numpy as np
import os
import FreeSimpleGUI as sg
from ..gui import _error_popup, _warning_popup, parameters_layout, add_header
from ..gui.prompts import input_image_prompt, prompt

import small_fish_gui.default_values as default

class ParameterInputError(Exception) :
    """
    Raised when user inputs an incorrect parameter.
    """
    pass

class MappingError(ValueError) :
    """
    Raised when user inputs an incorrect image mapping.
    """
    def __init__(self, map_ ,*args: object) -> None:
        super().__init__(*args)
        self.map_ = map_

    def get_map(self) :
        return self.map_

def prepare_image_detection(map_, user_parameters) :
    """
    Return monochannel image for ready for spot detection; 
    if image is already monochannel, nothing happens.
    else : image is the image on which detection is performed, other_image are the other layer to show in Napari Viewer.
    """
    image = reorder_image_stack(map_, user_parameters['image'])
    assert len(image.shape) != 5 , "Time stack not supported, should never be True"
    
    if user_parameters['is_multichannel'] :
        channel_to_compute = user_parameters['channel_to_compute']
        other_image = image.copy()
        other_image = np.delete(other_image, channel_to_compute, axis=0)
        other_image = [layer for layer in other_image]
        image: np.ndarray = image[channel_to_compute]

    else :
        other_image = []

    return image, other_image

def reorder_image_stack(map_, image_stack) :
    x = (int(map_['x']),)
    y = (int(map_['y']),)
    z = (int(map_['z']),) if type(map_.get('z')) != type(None) else ()
    c = (int(map_['c']),) if type(map_.get('c')) != type(None) else ()
    t = (int(map_['t']),) if type(map_.get('t')) != type(None) else ()

    source = t+c+z+y+x

    image_stack = np.moveaxis(
        image_stack,
        source= source,
        destination= tuple(range(len(source)))
    )

    return image_stack

def map_channels(user_parameters) :
    
    image = user_parameters['image']
    is_3D_stack = user_parameters['is_3D_stack']
    is_time_stack = False
    multichannel = user_parameters['is_multichannel']

    try : 
        map_ = _auto_map_channels(is_3D_stack, is_time_stack, multichannel, image=image)
    except MappingError as e :
        sg.popup("Automatic dimension mapping went wrong. Please indicate dimensions positions in the array.")
        map_ = _ask_channel_map(image.shape, is_3D_stack, is_time_stack, multichannel, preset_map= e.get_map())

    else :
        map_ = _show_mapping(image.shape, map_, is_3D_stack, is_time_stack, multichannel,)

    return map_

def _auto_map_channels(is_3D_stack, is_time_stack, multichannel, image: np.ndarray=None, shape=None) :
    if type(shape) == type(None) :
        shape = image.shape
    reducing_list = list(shape)

    #Set the biggest dimension to y
    y_val = max(reducing_list)
    y_idx = shape.index(y_val)
    map_ = {'y' : y_idx}

    #2nd biggest set to x
    reducing_list[y_idx] = -1
    x_val = max(reducing_list)
    x_idx = reducing_list.index(x_val)
    reducing_list[y_idx] = y_val

    map_['x'] = x_idx
    reducing_list.remove(y_val)
    reducing_list.remove(x_val)

    #smaller value set to c
    if multichannel :
        c_val = min(reducing_list)
        c_idx = shape.index(c_val)
        map_['c'] = c_idx
        reducing_list.remove(c_val)

    if is_time_stack :
        t_val = reducing_list[0]
        t_idx = shape.index(t_val)
        map_['t'] = t_idx
        reducing_list.remove(t_val)
    
    if is_3D_stack :
        z_val = reducing_list[0]
        z_idx = shape.index(z_val)
        map_['z'] = z_idx

    total_channels = len(map_)
    unique_channel = len(np.unique(list(map_.values())))

    if total_channels != unique_channel : raise MappingError(map_,"{0} channel(s) are not uniquely mapped.".format(total_channels - unique_channel))

    return map_

def _ask_channel_map(shape, is_3D_stack, is_time_stack, multichannel, preset_map: dict= {}) :
    while True :
        relaunch = False
        save_preset = preset_map.copy()
        x = preset_map.setdefault('x',0)
        y = preset_map.setdefault('y',0)
        z = preset_map.setdefault('z',0)
        c = preset_map.setdefault('c',0)
        t = preset_map.setdefault('t',0)


        layout = [
            add_header("Dimensions mapping") + [sg.Text("Image shape : {0}".format(shape))]
        ]
        layout += [parameters_layout(['x','y'], default_values=[x,y])]
        if is_3D_stack : layout += [parameters_layout(['z'], default_values=[z])]
        if multichannel : layout += [parameters_layout(['c'], default_values=[c])]
        if is_time_stack : layout += [parameters_layout(['t'], default_values=[t])]

        event, preset_map = prompt(layout, add_scrollbar=False)
        if event == 'Cancel' : return save_preset

        #Check integrity
        channels_values = np.array(list(preset_map.values()), dtype= int)
        total_channels = len(preset_map)
        unique_channel = len(np.unique(channels_values))
        if total_channels != unique_channel :
            sg.popup("{0} channel(s) are not uniquely mapped.".format(total_channels - unique_channel))
            relaunch= True
        if not all(channels_values < len(shape)):
            sg.popup("Channels values out of range for image dimensions.\nPlease select dimensions from {0}".format(list(range(len(shape)))))
            relaunch= True
        if not relaunch : break

    return preset_map

def _show_mapping(shape, map_, is_3D_stack, is_time_stack, multichannel) :
    while True : 
        layout = [
            [sg.Text("Image shape : {0}".format(shape))],
            [sg.Text('Dimensions mapping was set to :')],
            [sg.Text('x : {0} \ny : {1} \nz : {2} \nc : {3} \nt : {4}'.format(
                map_['x'], map_['y'], map_.get('z'), map_.get("c"), map_.get('t')
            ))],
            [sg.Button('Change mapping')]
        ]

        event, values = prompt(layout, add_scrollbar=False)

        if event == 'Ok' :
            return map_
        elif event == 'Change mapping':
            map_ = _ask_channel_map(shape, is_3D_stack, is_time_stack, multichannel, preset_map=map_)
        elif event == 'Cancel' : 
            return None
        else : raise AssertionError('Unforseen event')


def convert_parameters_types(values:dict) :
    """
    Convert parameters from `ask_input_parameters` from strings to float, int or tuple type.
    """

    #Tuples
    tuples_dict = {
        'voxel_size' : int,
        'spot_size' : float, 
        'log_kernel_size' : float, 
        'minimum_distance' : float, 
        'deconvolution_kernel' : float,
        }
    dim = values['dim']
    if dim == 3 : dim_tuple = ('z', 'y', 'x')
    else : dim_tuple = ('y', 'x')

    for tuple_parameter, type_cast in tuples_dict.items() :
        try :
            tuple_values = tuple([type_cast(values.get(tuple_parameter + '_{0}'.format(dimension))) for dimension in dim_tuple])
        except Exception as e : #execption when str cannot be converted to float or no parameter was given.
            values[tuple_parameter] = None
        else : values[tuple_parameter] = tuple_values

    #Parameters
    int_list = ['threshold', 'channel_to_compute', 'channel_to_compute', 'min_number_of_spots', 'cluster_size','nucleus channel signal']
    float_list = ['alpha', 'beta', 'gamma', 'threshold penalty']

    for parameter in int_list :
        try :
            parameter_value = int(values[parameter])
        except Exception :
            values[parameter] = None
        else : values[parameter] = parameter_value

    for parameter in float_list :
        try :
            parameter_value = float(values[parameter])
        except Exception :
            values[parameter] = None
        else : values[parameter] = parameter_value

    return values

def check_integrity(
        values: dict, 
        do_dense_region_deconvolution,
        do_clustering, 
        multichannel,
        segmentation_done, 
        map_, 
        shape
        ):
    """
    Checks that parameters given in input by user are fit to be used for bigfish detection.
    """

    #voxel_size
    if type(values['voxel_size']) == type(None) : 
        print(values['voxel_size'])
        raise ParameterInputError('Incorrect voxel size parameter.')
    
    #detection integrity :
    if not isinstance(values['spot_size'], (tuple, list)) and not(isinstance(values['minimum_distance'], (tuple, list)) and isinstance(values['log_kernel_size'], (tuple, list))) :
       raise ParameterInputError("Either minimum_distance and 'log_kernel_size' must be correctly set\n OR 'spot_size' must be correctly set.")
    
    #Deconvolution integrity
    if do_dense_region_deconvolution :
        if not isinstance(values['alpha'], (float, int)) or not isinstance(values['beta'], (float, int)) :
            raise ParameterInputError("Incorrect alpha or beta parameters.")
        if type(values['gamma']) == type(None) and not isinstance(values['deconvolution_kernel'], (list, tuple)):
            _warning_popup('No gamma found; image will not be denoised before deconvolution.')
            values['gamma'] = 0

        if values['alpha'] > 1 or values['alpha'] < 0 :
            raise ParameterInputError("alpha must be set between 0 and 1.")

    if do_clustering :
        if not isinstance(values['min_number_of_spots'], (int)) :
            raise ParameterInputError("Incorrect min spot number parameter.")
        if not isinstance(values['cluster_size'], (int)) :
            raise ParameterInputError("Incorrect cluster size parameter.")

    #channel
    if multichannel :
        ch_len = shape[int(map_['c'])]

        if type(segmentation_done) == type(None) :
            pass
        elif segmentation_done :
            try : nuc_signal_ch = int(values['nucleus channel signal'])
            except Exception :
                raise ParameterInputError("Incorrect channel for nucleus signal measure.")
            if nuc_signal_ch > ch_len :
                raise ParameterInputError("Nucleus signal channel is out of range for image.\nPlease select from {0}".format(list(range(ch_len))))
            values['nucleus channel signal'] = nuc_signal_ch

        try :
            ch = int(values['channel_to_compute'])
        except Exception :
            raise ParameterInputError("Incorrect channel_to_compute parameter.")
        if ch >= ch_len :
            raise ParameterInputError("channel_to_compute is out of range for image.\nPlease select from {0}".format(list(range(ch_len))))
        values['channel_to_compute'] = ch

    #Spot extraction
    if not os.path.isdir(values['spots_extraction_folder']) and values['spots_extraction_folder'] != '':
        raise ParameterInputError("Incorrect spot extraction folder.")


    return values

def reorder_shape(shape, map_) :
    x = [int(map_['x']),]
    y = [int(map_['y']),]
    z = [int(map_['z']),] if type(map_.get('z')) != type(None) else []
    c = [int(map_['c']),] if type(map_.get('c')) != type(None) else []
    t = [int(map_['t']),] if type(map_.get('t')) != type(None) else []

    source = t + c + z + y + x

    new_shape = tuple(
        np.array(shape)[source]
    )

    return new_shape

def _check_segmentation_parameters(
        user_parameters,
        shape,
        is_multichannel,
) :

    available_channels = list(range(len(shape)))
    do_only_nuc = user_parameters['segment_only_nuclei']
    cyto_model_name = user_parameters['cyto_model_name']
    cyto_size = user_parameters['cytoplasm_diameter']
    cytoplasm_channel = user_parameters['cytoplasm_channel']
    nucleus_model_name = user_parameters['nucleus_model_name']
    nucleus_size = user_parameters['nucleus_diameter']
    nucleus_channel = user_parameters['nucleus_channel']
   

    if type(cyto_model_name) != str  and not do_only_nuc:
        raise ParameterInputError('Invalid cytoplasm model name.')
    if cytoplasm_channel not in available_channels and not do_only_nuc and is_multichannel:
        raise ParameterInputError('For given input image please select channel in {0}\ncytoplasm_channel : {1}'.format(available_channels, cytoplasm_channel))

    if type(cyto_size) not in [int, float] and not do_only_nuc:
        raise ParameterInputError("Incorrect cytoplasm size.")

    if type(nucleus_model_name) != str :
        raise ParameterInputError('Invalid nucleus model name.')
    
    if nucleus_channel not in available_channels and is_multichannel:
        raise ParameterInputError('For given input image please select channel in {0}\nnucleus channel : {1}'.format(available_channels, nucleus_channel))
    
    if type(nucleus_size) not in [int, float] :
        raise ParameterInputError("Incorrect nucleus size.")


def clean_unused_parameters_cache(user_parameters: dict) :
    """
    Clean unused parameters that were set to None in previous run.
    """
    parameters = ['alpha', 'beta', 'gamma', 'cluster_size', 'min_number_of_spots']
    for parameter in parameters :
        if parameter in user_parameters.keys() :
            if type(user_parameters[parameter]) == type(None) :
                del user_parameters[parameter]
    
    return user_parameters

def ask_input_parameters(ask_for_segmentation=True) :
    """
    Prompt user with interface allowing parameters setting for bigFish detection / deconvolution.
    """
    
    values = {}
    image_input_values = {}
    while True :
        is_3D_preset = image_input_values.setdefault('is_3D_stack', default.IS_3D_STACK)
        is_multichannel_preset = image_input_values.setdefault('is_multichannel', default.IS_MULTICHANNEL)
        denseregion_preset = image_input_values.setdefault('do_dense_regions_deconvolution', default.DO_DENSE_REGIONS_DECONVOLUTION)
        do_clustering_preset = image_input_values.setdefault('do_cluster_computation', default.DO_CLUSTER_COMPUTATION)
        do_napari_preset = image_input_values.setdefault('show_napari_corrector', default.SHOW_NAPARI_CORRECTOR)

        if ask_for_segmentation :
            image_input_values = input_image_prompt(
                is_3D_stack_preset=is_3D_preset,
                multichannel_preset=is_multichannel_preset,
                do_dense_regions_deconvolution_preset=None,
                do_clustering_preset= None,
                do_Napari_correction=None,
            )
        else :
            image_input_values = input_image_prompt(
                is_3D_stack_preset=is_3D_preset,
                multichannel_preset=is_multichannel_preset,
                do_dense_regions_deconvolution_preset=denseregion_preset,
                do_clustering_preset= do_clustering_preset,
                do_Napari_correction=do_napari_preset,
            )

        if type(image_input_values) == type(None) :
            return image_input_values

        if 'image' in image_input_values.keys() :
            image_input_values['shape'] = image_input_values['image'].shape 
            break


    values.update(image_input_values)
    values['dim'] = 3 if values['is_3D_stack'] else 2
    values['filename'] = os.path.basename(values['image_path'])
    
    return values