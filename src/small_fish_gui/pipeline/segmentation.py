"""
Contains cellpose wrappers to segmentate images.
"""

from cellpose.core import use_gpu
from skimage.measure import label
from ..hints import pipeline_parameters
from ..interface import get_settings
from ..gui import prompt, ask_cancel_segmentation, segmentation_prompt
from ..gui.napari_visualiser import show_segmentation as napari_show_segmentation
from ..interface import open_image, SettingsDict, get_settings

from .utils import from_label_get_centeroidscoords
from ._preprocess import ask_input_parameters
from ._preprocess import map_channels, reorder_shape, reorder_image_stack

from matplotlib.colors import ListedColormap
import matplotlib as mpl
import cellpose.models as models
import numpy as np
import bigfish.multistack as multistack
import bigfish.stack as stack
import bigfish.plot as plot
import FreeSimpleGUI as sg
import matplotlib.pyplot as plt
import os
from .utils import using_mps

def launch_segmentation(user_parameters: pipeline_parameters, nucleus_label, cytoplasm_label, batch_mode=False) :
    """
    Ask user for necessary parameters and perform cell segmentation (cytoplasm + nucleus) with cellpose.

    Input
    -----
    Image : np.ndarray[c,z,y,x]
        Image to use for segmentation.

    Returns
    -------
        cytoplasm_label, nucleus_label, user_parameters
    """

    segmentation_parameters : pipeline_parameters = user_parameters.copy()
    default = get_settings()

    #Ask for image parameters
    new_parameters = ask_input_parameters(user_parameters, ask_for_segmentation= True) #The image is open and stored inside user_parameters
    if type(new_parameters) == type(None) : #if user clicks 'Cancel'
        return nucleus_label , cytoplasm_label, user_parameters
    else :
        segmentation_parameters.update(new_parameters)

    map_ = map_channels(segmentation_parameters)
    if type(map_) == type(None) : #User clicks Cancel 
        return nucleus_label, cytoplasm_label, user_parameters
    segmentation_parameters['reordered_shape'] = reorder_shape(segmentation_parameters['shape'], map_)
    image = reorder_image_stack(map_, segmentation_parameters['image'])

    is_multichannel = segmentation_parameters["is_multichannel"]
    is_3D_stack = segmentation_parameters["is_3D_stack"]

    while True : # Loop if show_segmentation 
        #Default parameters
        path = os.getcwd()
        available_channels = list(range(image.shape[0]))
        available_slices = list(range(image.shape[is_multichannel])) #0 if not multichannel else 1

    #Ask user for parameters
    #if incorrect parameters --> set relaunch to True
        while True :
            segmentation_parameters.setdefault("other_nucleus_image", "")
            segmentation_parameters.setdefault("cytoplasm_channel", segmentation_parameters["detection_channel"])
            segmentation_parameters.setdefault("cytoplasm_segmentation_3D", segmentation_parameters["do_3D_segmentation"])
            segmentation_parameters.setdefault("nucleus_segmentation_3D", segmentation_parameters["do_3D_segmentation"])

            event, values = segmentation_prompt(
                saving_path= segmentation_parameters.setdefault("seg_control_saving_path", segmentation_parameters["working_directory"]),
                **segmentation_parameters
                )

            if event == 'Cancel' :
                cancel_segmentation = ask_cancel_segmentation()

                if cancel_segmentation :
                    return None, None, user_parameters
                else : 
                    continue

            #Extract parameters
            values, relaunch = _check_integrity_segmentation_parameters(values, segmentation_parameters, available_channels=available_channels, available_slices=available_slices)
            segmentation_parameters.update(values)
            if not relaunch : break

        if not batch_mode :
            #Launching segmentation
            waiting_layout = [
                [sg.Text("Running segmentation...")]
            ]
            window = sg.Window(
                title= 'small_fish',
                layout= waiting_layout,
                grab_anywhere= True,
                no_titlebar= False
            )

            window.read(timeout= 30, close= False)

        try :
            cytoplasm_label, nucleus_label = cell_segmentation(
                image,
                channels=[segmentation_parameters["cytoplasm_channel"], segmentation_parameters["nucleus_channel"]],
                do_only_nuc=segmentation_parameters["segment_only_nuclei"],
                external_nucleus_image = segmentation_parameters["other_nucleus_image"],
                nucleus_3D_segmentation=segmentation_parameters["nucleus_segmentation_3D"],
                cyto_3D_segmentation=segmentation_parameters["cytoplasm_segmentation_3D"],
                **segmentation_parameters
                )

        finally  : 
            if not batch_mode : window.close()

        if segmentation_parameters["show_segmentation"] :
            
            nucleus_label, cytoplasm_label = napari_show_segmentation(
                nuc_image=image[segmentation_parameters["nucleus_channel"]] if type(segmentation_parameters["other_nucleus_image"]) == type(None) else segmentation_parameters["other_nucleus_image"],
                nuc_label= nucleus_label,
                cyto_image=image[segmentation_parameters["cytoplasm_channel"]],
                cyto_label=cytoplasm_label,
                anisotrpy=segmentation_parameters["anisotropy"],
            )

            if nucleus_label.ndim == 3 : nucleus_label = np.max(nucleus_label, axis=0)
            if cytoplasm_label.ndim == 3 : cytoplasm_label = np.max(cytoplasm_label, axis=0)

            layout = [
                [sg.Text("Proceed with current segmentation ?")],
                [sg.Button("Yes"), sg.Button("No")]
            ]
            
            event, values = prompt(layout=layout, add_ok_cancel=False, add_scrollbar=False)
            if event == "No" :
                continue

        if type(segmentation_parameters["seg_control_saving_path"]) != type(None) and segmentation_parameters["save_segmentation_visuals"]:
            
            #Get backgrounds
            nuc_proj = image[segmentation_parameters["nucleus_channel"]]
            im_proj = image[segmentation_parameters["cytoplasm_channel"]]
            if im_proj.ndim == 3 :
                im_proj = stack.mean_projection(im_proj)
            if nuc_proj.ndim == 3 :
                nuc_proj = stack.mean_projection(nuc_proj)
            if nucleus_label.ndim == 3 :
                nucleus_label_proj = np.max(segmentation_parameters["nucleus_channel"],axis=0)
            else : 
                nucleus_label_proj = nucleus_label
            if cytoplasm_label.ndim == 3 :
                cytoplasm_label_proj = np.max(cytoplasm_label,axis=0)
            else : 
                cytoplasm_label_proj = cytoplasm_label

            
            #Call plots
            if segmentation_parameters["save_segmentation_visuals"]:
                output_path = segmentation_parameters["seg_control_saving_path"] + '/' + segmentation_parameters["filename"]
                nuc_path = output_path + "_nucleus_segmentation"
                cyto_path = output_path + "_cytoplasm_segmentation"

                #Plots boundaries
                plot.plot_segmentation_boundary(nuc_proj, cytoplasm_label_proj, nucleus_label_proj, boundary_size=2, contrast=True, show=False, path_output=nuc_path, title= "Nucleus segmentation (blue)", remove_frame=True,)
                if not segmentation_parameters["segment_only_nuclei"] : 
                    plot.plot_segmentation_boundary(im_proj, cytoplasm_label_proj, nucleus_label_proj, boundary_size=2, contrast=True, show=False, path_output=cyto_path, title="Cytoplasm Segmentation (red)", remove_frame=True)
            
                #Plots cell labels
                plot_labels(
                    nucleus_label,
                    path_output=output_path + "_nucleus_label_map.png",
                    show=False
                    )
                if not segmentation_parameters["segment_only_nuclei"] : 
                    plot_labels(
                        cytoplasm_label_proj,
                        path_output=output_path + "_cytoplasm_label_map.png",
                        show=False
                        )


        if cytoplasm_label.max() == 0 : #No cell segmented
            layout = [
            [sg.Text("No cell segmented. Proceed anyway ?")],
            [sg.Button("Yes"), sg.Button("No")]
        ]
            event, _ = prompt(layout=layout, add_ok_cancel=False)
            if event == "Yes" :
                return None, None, user_parameters
        else :
            break

    user_parameters.update(segmentation_parameters)
    return nucleus_label, cytoplasm_label, user_parameters

def cell_segmentation(
        reordered_image, 
        cytoplasm_model_name, 
        nucleus_model_name, 
        channels, 
        cytoplasm_diameter, 
        nucleus_diameter,
        nucleus_3D_segmentation=False,
        cyto_3D_segmentation=False,
        anisotropy = 1,
        nucleus_flow_threshold = 0.4,
        cytoplasm_flow_threshold = 0.4,
        nucleus_cellprob_threshold = 0.,
        cytoplasm_cellprob_threshold = 0.,
        do_only_nuc=False,
        external_nucleus_image = None,
        **segmentation_parameters : pipeline_parameters
        ) :

    nuc_channel = channels[1]
    

    if type(external_nucleus_image) != type(None) :
        nuc = external_nucleus_image
    else :
        nuc = reordered_image[nuc_channel]

    if nuc.ndim >= 3 and not nucleus_3D_segmentation:

        if segmentation_parameters["nucleus_max_proj"] : nuc = np.max(nuc, axis=0)
        elif segmentation_parameters["nucleus_mean_proj"] : nuc = np.mean(nuc, axis=0)
        elif segmentation_parameters["nucleus_select_slice"] : nuc = nuc[segmentation_parameters["nucleus_selected_slice"]]
        else : raise AssertionError("No option found for 2D nucleus seg. Should be impossible as this error is raised after integrity checks")
    
    
    nuc_label = _segmentate_object(
        nuc, 
        nucleus_model_name, 
        nucleus_diameter, 
        do_3D=nucleus_3D_segmentation, 
        anisotropy=anisotropy,
        flow_threshold= nucleus_flow_threshold,
        cellprob_threshold=nucleus_cellprob_threshold,
        min_size=segmentation_parameters["nucleus_min_size"]
        )
    
    if not do_only_nuc : 
        cyto_channel = channels[0]
        nuc = reordered_image[nuc_channel] if type(external_nucleus_image) == type(None) else external_nucleus_image

        if reordered_image[cyto_channel].ndim >= 3 and not cyto_3D_segmentation:
            if segmentation_parameters["cytoplasm_max_proj"] : cyto = np.max(reordered_image[cyto_channel], axis=0)
            elif segmentation_parameters["cytoplasm_mean_proj"] : cyto = np.mean(reordered_image[cyto_channel], axis=0)
            elif segmentation_parameters["cytoplasm_select_slice"] : cyto = reordered_image[cyto_channel][segmentation_parameters["cytoplasm_selected_slice"]]
            else : raise AssertionError("No option found for 2D cytoplasm seg. Should be impossible as this error is raised after integrity checks")
        else : 
            cyto = reordered_image[cyto_channel]
        if nuc.ndim >= 3 and not cyto_3D_segmentation:
            if segmentation_parameters["cytoplasm_max_proj"] : nuc = np.max(nuc, axis=0)
            elif segmentation_parameters["cytoplasm_mean_proj"] : nuc = np.mean(nuc, axis=0)
            elif segmentation_parameters["cytoplasm_select_slice"] : nuc = nuc[segmentation_parameters["cytoplasm_selected_slice"]]
            else : raise AssertionError("No option found for 2D cytoplasm seg. Should be impossible as this error is raised after integrity checks")

        reordered_image = np.zeros(shape=(2,) + cyto.shape)
        reordered_image[0] = cyto
        reordered_image[1] = nuc
        source = list(range(reordered_image.ndim))
        dest = source[-1:] + source[:-1]
        reordered_image = np.moveaxis(reordered_image, source=range(reordered_image.ndim), destination= dest)

        cytoplasm_label = _segmentate_object(
            reordered_image, 
            cytoplasm_model_name, 
            cytoplasm_diameter, 
            do_3D=cyto_3D_segmentation, 
            anisotropy=anisotropy,
            flow_threshold=cytoplasm_flow_threshold,
            cellprob_threshold=cytoplasm_cellprob_threshold,
            min_size=segmentation_parameters["cytoplasm_min_size"]
            )

        if cytoplasm_label.ndim == 3 and nuc_label.ndim == 2 :
            nuc_label = np.repeat(nuc_label[np.newaxis], len(cytoplasm_label), axis= 0)
        if nuc_label.ndim == 3 and cytoplasm_label.ndim == 2 :
            cytoplasm_label = np.repeat(cytoplasm_label[np.newaxis], len(nuc_label), axis= 0)

        nuc_label, cytoplasm_label = multistack.match_nuc_cell(nuc_label=nuc_label, cell_label=cytoplasm_label, single_nuc=True, cell_alone=False)
    else :
        cytoplasm_label = nuc_label

    return cytoplasm_label, nuc_label

def _segmentate_object(
        im : np.ndarray, 
        model_name : str, 
        object_size_px : int, 
        do_3D = False, 
        anisotropy : float = 1.0,
        flow_threshold : float = 0.4,
        cellprob_threshold : float = 0, 
        min_size = 15 #Default cellpose
        ) :
    

    model = models.CellposeModel(
        gpu= use_gpu(),
        pretrained_model= model_name,
        use_bfloat16= not using_mps()
    )

    label, flow, style = model.eval(
        im,
        diameter= object_size_px,
        do_3D= do_3D,
        z_axis=0 if do_3D else None,
        channel_axis= im.ndim -1 if im.ndim == 3+ do_3D else None,
        anisotropy=anisotropy,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
        min_size=min_size,
        )
    
    label = np.array(label, dtype= np.int64)
    if not do_3D : label = remove_disjoint(label) # Too much time consuming in 3D
    
    return label

def _cast_segmentation_parameters(values:dict) :

    cast_rules = {
        'cytoplasm_diameter' : int,
        'nucleus_diameter' : int,
        'cytoplasm_channel' : int,
        'nucleus_channel' : int,
        'anisotropy' : float,
        'cytoplasm_flow_threshold' : float,
        'cytoplasm_cellprob_threshold' : float,
        'nucleus_flow_threshold' : float,
        'nucleus_cellprob_threshold' : float,
        'cytoplasm_anisotropy' : float,
        'nucleus_anisotropy' : float,
        'cytoplasm_selected_slice' : int,
        'nucleus_selected_slice' : int,
        'cytoplasm_min_size' : int,
        'nucleus_min_size' : int,
    }    

    for key, constructor in cast_rules.items() :
        try :
            if key in values.keys() : values[key] = constructor(values[key])
        except ValueError :
            pass

    #None if default
    if values['cytoplasm_model_name'] == '' :
        values['cytoplasm_model_name'] = None

    if values['nucleus_model_name'] == '' :
        values['nucleus_model_name'] = None

    return values

def _check_integrity_segmentation_parameters(
    values : pipeline_parameters, 
    user_parameters : pipeline_parameters, 
    available_channels : list, 
    available_slices : list
    ) :
    """
    Check that parameters entered by user comply with what we need to do with them. Type checking is handle in _cast_segmentation_parameters, if any parameters could not be cast to 
    appropriate type, it is set to None.
    """

    values : pipeline_parameters = _cast_segmentation_parameters(values)
    is_3D_stack = user_parameters["is_3D_stack"]
    is_multichannel = user_parameters["is_multichannel"]

    relaunch= False
    #Control plots
    if values["save_segmentation_visuals"] :
        if not os.path.isdir(values["seg_control_saving_path"]) :
            relaunch=True
            sg.popup(f"{values['seg_control_saving_path']} is not a directory.")
            values["seg_control_saving_path"] = user_parameters.get("seg_control_saving_path")
        if values["filename"] == "" :
            relaunch=True
            sg.popup("Enter a filename for control plots.")
            values["filename"] = user_parameters.get("filename")

    #2D/3D seg
    if is_3D_stack :
        for obj in ["cytoplasm","nucleus"] :
            assert values[obj + "_radio_2D"] + values[obj + "_radio_3D"] == 1, f"Incorrect number of segmentation dimension was selected (should be 1) for {obj}"
            assert values[obj + "_max_proj"] + values[obj + "_mean_proj"] + values[obj + "_select_slice"] == 1, f"Incorrect number of 2D segmentation options were selected (should be 1) for {obj}"

            if values[obj + "_radio_2D"] : values[obj + "_segmentation_3D"] = False
            else : values[obj + "_segmentation_3D"] = True

        #Selected slice
        for key in ['cytoplasm_selected_slice', 'nucleus_selected_slice'] :
            if not isinstance(values[key], int) :
                relaunch = True
                sg.popup("Invalid slice number for cytoplasm 2D segmentation.")
                values[key] = user_parameters[key]
            elif values[key] not in available_slices :
                relaunch = True
                sg.popup(f"Invalid slice number for cytoplasm 2D segmentation, selecgted between 0 and {available_slices[-1]}")
                values[key] = user_parameters[key]
        
        #anisotropy
        if not "anisotropy" in values.keys() :
            values["anisotropy"] = 1
        if not isinstance(values["nucleus_anisotropy"], (float, int)) :
            relaunch=True
            sg.popup("Invalid value for nucleus anisotropy, must be a positive float.")
            values["anisotropy"] = user_parameters.get("anisotropy")
        elif values["nucleus_anisotropy"] < 0 :
            relaunch=True
            sg.popup("Invalid value for nucleus anisotropy, must be a positive float.")
            values["anisotropy"] = user_parameters.get("anisotropy")

        if not values["segment_only_nuclei"] :
            if not isinstance(values["cytoplasm_anisotropy"], (float, int)) :
                relaunch=True
                sg.popup("Invalid value for cytoplasm anisotropy, must be a positive float.")
                values["anisotropy"] = user_parameters.get("anisotropy")
            elif values["cytoplasm_anisotropy"] < 0 :
                relaunch=True
                sg.popup("Invalid value for cytoplasm anisotropy, must be a positive float.")
                values["anisotropy"] = user_parameters.get("anisotropy")

        if values["segment_only_nuclei"] :
            values["anisotropy"] = values["nucleus_anisotropy"]
        elif isinstance(values["cytoplasm_anisotropy"], (float, int)) and isinstance(values["nucleus_anisotropy"], (float, int)) and values["nucleus_segmentation_3D"] and values["cytoplasm_segmentation_3D"]:
            if (not values["cytoplasm_anisotropy"] == values["nucleus_anisotropy"]) and (not values["segment_only_nuclei"]) :
                relaunch=True
                sg.popup("Anisotropy must be equal for nucleus and cytoplasm segmentation")
                values["anisotropy"] = user_parameters.get("anisotropy")
            else :
                values["anisotropy"] = values["nucleus_anisotropy"]
        elif isinstance(values["nucleus_anisotropy"], (float, int)) and values["nucleus_segmentation_3D"] :
            values["anisotropy"] = values["nucleus_anisotropy"]
        elif isinstance(values["cytoplasm_anisotropy"], (float, int)) and values["cytoplasm_segmentation_3D"] :
            values["anisotropy"] = values["cytoplasm_anisotropy"]

        if not isinstance(values["anisotropy"], (float,int)) :
            sg.popup("Anisotropy must be a positive float.")
            relaunch = True
            values['anisotropy'] = user_parameters.get("anisotropy")
    
    else :
        values["cytoplasm_segmentation_3D"] = False
        values["nucleus_segmentation_3D"] = False
        values["anisotropy"] = 1
        for obj in ["nucleus","cytoplasm"] :
            values[obj + "_max_proj"] = False
            values[obj + "_mean_proj"] = True
            values[obj + "_select_slice"] = False
            values[obj + "_selected_slice"] = 0

    #Min size
    if not isinstance(values["cytoplasm_min_size"], int) :
        relaunch=True
        sg.popup("Invalid value for cytoplasm min size parameter, must be a positive int.")
        values["cytoplasm_min_size"] = user_parameters["cytoplasm_min_size"]
    elif values["cytoplasm_min_size"] < 0 :
        relaunch=True
        sg.popup("Invalid value for cytoplasm min size parameter, must be a positive int.")
        values["cytoplasm_min_size"] = user_parameters["cytoplasm_min_size"]

    if not isinstance(values["nucleus_min_size"], int) :
        relaunch=True
        sg.popup("Invalid value for nucleus min size parameter, must be a positive int.")
        values["nucleus_min_size"] = user_parameters["nucleus_min_size"]
    elif values["nucleus_min_size"] < 0 :
        relaunch=True
        sg.popup("Invalid value for nucleus min size parameter, must be a positive int.")
        values["nucleus_min_size"] = user_parameters["nucleus_min_size"]

    #flow thresholds
    if type(values['nucleus_flow_threshold']) != float : 
        sg.popup('Invalid value for flow threshold in nuc parameters, must be a float between 0 and 1.')
        values['nucleus_flow_threshold'] = user_parameters['nucleus_flow_threshold']
        relaunch= True
    elif abs(values['nucleus_flow_threshold']) > 1 or values['nucleus_flow_threshold'] < 0 :
        sg.popup('Invalid value for flow threshold in cyto parameters, must be a float between 0 and 1.')
        values['nucleus_flow_threshold'] = user_parameters['nucleus_flow_threshold']
        relaunch= True 

    if type( values['cytoplasm_flow_threshold']) != float : 
        sg.popup('Invalid value for flow threshold in cyto parameters, must be a float between 0 and 1.')
        values['cytoplasm_flow_threshold'] = user_parameters['cytoplasm_flow_threshold']
        relaunch= True
    elif abs(values['cytoplasm_flow_threshold']) > 1 or values['cytoplasm_flow_threshold'] < 0 :
        sg.popup('Invalid value for flow threshold in cyto parameters, must be a float between 0 and 1.')
        values['cytoplasm_flow_threshold'] = user_parameters['cytoplasm_flow_threshold']
        relaunch= True 
    
    #cellprob thresholds
    if type(values['nucleus_cellprob_threshold']) != float : 
        sg.popup('Invalid value for cellprob threshold in nuc parameters, must be a float between -3 and +3.')
        values['nucleus_cellprob_threshold'] = user_parameters['nucleus_cellprob_threshold']
        relaunch= True
    elif abs(values['nucleus_cellprob_threshold']) > 3 :
        sg.popup('Invalid value for cellprob threshold in cyto parameters, must be a float between -3 and +3.')
        values['nucleus_cellprob_threshold'] = user_parameters['nucleus_cellprob_threshold']
        relaunch= True

    if type(values['cytoplasm_cellprob_threshold']) != float :
        sg.popup('Invalid value for cellprob threshold in cyto parameters, must be a float between -3 and +3.')
        values['cytoplasm_cellprob_threshold'] = user_parameters['cytoplasm_cellprob_threshold']
        relaunch= True
    elif abs(values['cytoplasm_cellprob_threshold']) > 3 :
        sg.popup('Invalid value for cellprob threshold in cyto parameters, must be a float between -3 and +3.')
        values['cytoplasm_cellprob_threshold'] = user_parameters['cytoplasm_cellprob_threshold']
        relaunch= True

    
    #Cytoplasm parameters
    if type(values["cytoplasm_model_name"]) != str  and not do_only_nuc:
        sg.popup('Invalid cytoplasm model name.')
        values['cytoplasm_model_name'] = user_parameters['cytoplasm_model_name']
        relaunch= True
    if is_multichannel :
        if values["cytoplasm_channel"] not in available_channels and not do_only_nuc:
            sg.popup('For given input image please select channel in {0}\ncytoplasm_channel : {1}'.format(available_channels, cytoplasm_channel))
            relaunch= True
            values['cytoplasm_channel'] = user_parameters['cytoplasm_channel']
    else :
        values["cytoplasm_channel"] = ...

    if type(values["cytoplasm_diameter"]) not in [int, float] and not do_only_nuc:
        sg.popup("Incorrect cytoplasm size.")
        relaunch= True
        values['cytoplasm_diameter'] = user_parameters['cytoplasm_diameter']

    #Nucleus parameters
    if type(values["nucleus_model_name"]) != str :
        sg.popup('Invalid nucleus model name.')
        values['nucleus_model_name'] = user_parameters['nucleus_model_name']
        relaunch= True
    
    if is_multichannel :
        if values["nucleus_channel"] not in available_channels :
            sg.popup('For given input image please select channel in {0}\nnucleus channel : {1}'.format(available_channels, nucleus_channel))
            relaunch= True
            values['nucleus_channel'] = user_parameters['nucleus_channel']
    else : 
        values["nucleus_channel"] = ...

    if type(values["nucleus_diameter"]) not in [int, float] :
        sg.popup("Incorrect nucleus size.")
        relaunch= True
        values['nucleus_diameter'] = user_parameters['nucleus_diameter']

    if values["other_nucleus_image"] != '' :
        if os.path.isdir(values["other_nucleus_image"]) :
            values['other_nucleus_image'] = None
        elif not os.path.isfile(values["other_nucleus_image"]) :
            sg.popup("Nucleus image is not a file.")
            relaunch=True
            values['other_nucleus_image'] = None
        else :
            try :
                nucleus_image = open_image(other_nucleus_image)
            except Exception as e :
                sg.popup("Could not open image.\n{0}".format(e))
                relaunch=True
                values['other_nucleus_image'] = user_parameters['other_nucleus_image']
            else :
                if nucleus_image.ndim != image.ndim - is_multichannel :
                    sg.popup("Nucleus image dimension missmatched. Expected same dimension as cytoplasm_image for monochannel or same dimension as cytoplasm_image -1 for is_multichannel\ncytoplasm dimension : {0}, nucleus dimension : {1}".format(image.ndim, nucleus_image.ndim))
                    nucleus_image = None
                    relaunch=True
                    values['other_nucleus_image'] = user_parameters['other_nucleus_image']
                
                elif nucleus_image.shape != image[cytoplasm_channel].shape :
                    sg.popup("Nucleus image shape missmatched. Expected same shape as cytoplasm_image \ncytoplasm shape : {0}, nucleus shape : {1}".format(image[cytoplasm_channel].shape, nucleus_image.shape))
                    nucleus_image = None
                    relaunch=True
                    values['other_nucleus_image'] = user_parameters['other_nucleus_image']

    else :
        values["other_nucleus_image"] = None

    return values, relaunch


def remove_disjoint(image):
    """
    *CODE FROM BIG-FISH (LICENCE IN __INIT__.PY) IMPORTED TO AVOID IMPORT ERROR WHEN BIGFISH SEGMENTATION MODULE INITIALISES : try to import deprecated module for python 3.8

    For each instances with disconnected parts, keep the larger one.

    Parameters
    ----------
    image : np.ndarray, np.int, np.uint or bool
        Labelled image with shape (z, y, x) or (y, x).

    Returns
    -------
    image_cleaned : np.ndarray, np.int or np.uint
        Cleaned image with shape (z, y, x) or (y, x).

    """
    # check parameters
    stack.check_array(
        image,
        ndim=[2, 3],
        dtype=[np.uint8, np.uint16, np.int32, np.int64, bool])

    # handle boolean array
    cast_to_bool = False
    if image.dtype == bool:
        cast_to_bool = bool
        image = image.astype(np.uint8)

    # initialize cleaned labels
    image_cleaned = np.zeros_like(image)

    # loop over instances
    max_label = image.max()
    for i in range(1, max_label + 1):

        # get instance mask
        mask = image == i

        # check if an instance is labelled with this value
        if mask.sum() == 0:
            continue

        # get an index for each disconnected part of the instance
        labelled_mask = label(mask)
        indices = sorted(list(set(labelled_mask.ravel())))
        if 0 in indices:
            indices = indices[1:]

        # keep the largest part of the instance
        max_area = 0
        mask_instance = None
        for j in indices:
            mask_part_j = labelled_mask == j
            area_j = mask_part_j.sum()
            if area_j > max_area:
                max_area = area_j
                mask_instance = mask_part_j

        # add instance in the final label
        image_cleaned[mask_instance] = i

    if cast_to_bool:
        image_cleaned = image_cleaned.astype(bool)

    return image_cleaned

def plot_segmentation(
        cyto_image : np.ndarray, 
        cyto_label : np.ndarray, 
        nuc_image : np.ndarray, 
        nuc_label : np.ndarray,
        path :str, 
        do_only_nuc=False
        ) :

    if nuc_image.ndim == 3 :
        nuc_image = np.max(nuc_image,axis=0)
    
    if cyto_label.ndim == 3 :
        cyto_label = np.max(cyto_label,axis=0)
    
    if nuc_label.ndim == 3 :
        nuc_label = np.max(nuc_label,axis=0)
    
    if cyto_image.ndim == 3 :
        cyto_image = np.max(cyto_image,axis=0)
    
    plot.plot_segmentation_boundary(
        image=nuc_image,
        nuc_label= nuc_label,
        boundary_size= 3,
        contrast=True,
        path_output=path + "_nuclei_segmentation.png",
        show=False,
    )


    if not do_only_nuc :
        if cyto_image.ndim == 3 :
            cyto_image = np.max(cyto_image,axis=0)
    
        plot.plot_segmentation_boundary(
            image=cyto_image,
            cell_label= cyto_label,
            nuc_label= nuc_label,
            boundary_size= 3,
            contrast=True,
            path_output=path + "_cytoplasm_segmentation.png",
            show=False,
        )

def plot_labels(labelled_image: np.ndarray, path_output:str = None, show= True, axis= False, close= True):
    """
    Plot a labelled image and indicate the label number at the center of each region.
    """
    stack.check_parameter(labelled_image = (np.ndarray, list), show = (bool))
    if isinstance(labelled_image, np.ndarray) : 
        stack.check_array(labelled_image, ndim= 2)
        labelled_image = [labelled_image]
    
    #Setting a colormap with background to white so all cells can be visible
    viridis = mpl.colormaps['viridis'].resampled(256)
    newcolors = viridis(np.linspace(0, 1, 256))
    white = np.array([1, 1, 1, 1])
    newcolors[0, :] = white
    newcmp = ListedColormap(newcolors)

    plt.figure(figsize= (10,10))
    rescaled_image = stack.rescale(np.array(labelled_image[0], dtype= np.int32), channel_to_stretch= 0)
    rescaled_image[rescaled_image == 0] = -100
    plot = plt.imshow(rescaled_image, cmap=newcmp)
    plot.axes.get_xaxis().set_visible(axis)
    plot.axes.get_yaxis().set_visible(axis)
    plt.tight_layout()

    for index in range(0, len(labelled_image)) :
        centroid_dict = from_label_get_centeroidscoords(labelled_image[index])
        labels = centroid_dict["label"]
        Y = centroid_dict["centroid-0"]
        X = centroid_dict["centroid-1"]
        centroids = zip(Y,X)

        for label in labels :
            y,x = next(centroids)
            y,x = round(y), round(x)
            an = plt.annotate(str(label), [round(x), round(y)])

    if not axis : plt.cla
    if show : plt.show()
    if path_output != None :
        stack.check_parameter(path_output = (str))
        plt.savefig(path_output)
    if close : plt.close()

    return plot