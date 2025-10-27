"""
Functions handling GUI update while user is interacting with software.
"""
import FreeSimpleGUI as sg

from .utils import get_elmt_from_key



def update_master_parameters(
        Master_parameter_dict:dict,
        update_dict:dict
) :
    for parameter, is_ok in Master_parameter_dict.items() :
        elmt_to_update:sg.Element = update_dict.get(parameter)
        if type(elmt_to_update) == type(None): continue
        else :
            if is_ok : 
                text:str = elmt_to_update.DisplayText.replace('Uncorrect', 'Correct')
                color = 'green'
            else : 
                text:str = elmt_to_update.DisplayText.replace('Correct', 'Uncorrect')
                color = 'gray'

            elmt_to_update.update(value=text, text_color = color)

def update_detection_tab(
        tab_elmt:sg.Tab, 
        is_multichannel, 
        is_3D, 
        do_dense_region_deconvolution,
        do_clustering,
        is_mapping_ok,
        do_segmentation,
        ) :
    
    #Acess elements
    ##Detection
    channel_to_compute = get_elmt_from_key(tab_elmt, key= 'channel_to_compute')
    voxel_size_z = get_elmt_from_key(tab_elmt, key= 'voxel_size_z')
    spot_size_z = get_elmt_from_key(tab_elmt, key= 'spot_size_z')
    log_kernel_size_z = get_elmt_from_key(tab_elmt, key= 'log_kernel_size_z')
    minimum_distance_z = get_elmt_from_key(tab_elmt, key= 'minimum_distance_z')

    ##Dense regions deconvolution
    alpha = get_elmt_from_key(tab_elmt, key= 'alpha')
    beta = get_elmt_from_key(tab_elmt, key= 'beta')
    gamma = get_elmt_from_key(tab_elmt, key= 'gamma')
    deconvolution_kernel_x = get_elmt_from_key(tab_elmt, key= 'deconvolution_kernel_x')
    deconvolution_kernel_y = get_elmt_from_key(tab_elmt, key= 'deconvolution_kernel_y')
    deconvolution_kernel_z = get_elmt_from_key(tab_elmt, key= 'deconvolution_kernel_z')
    
    #Clustering
    cluster_size = get_elmt_from_key(tab_elmt, key= 'cluster_size')
    min_number_of_spot = get_elmt_from_key(tab_elmt, key= 'min_number_of_spots')
    
    #segmentation and multichannel
    nucleus_channel_signal = get_elmt_from_key(tab_elmt, key= 'nucleus channel signal')
    
    #disable
    interactive_threshold_selector = get_elmt_from_key(tab_elmt, key= 'show_interactive_threshold_selector')

    update_dict={
        'is_3D' : is_3D,
        'is_multichannel' : is_multichannel,
        'do_dense_region_deconvolution' : do_dense_region_deconvolution,
        'do_clustering' : do_clustering,
        'always_hidden' : False,
        'is_3D&do_denseregion' : is_3D and do_dense_region_deconvolution,
        'do_segmentation' : do_segmentation and is_multichannel
    }

    list_dict={
        'is_3D' : [voxel_size_z, spot_size_z, log_kernel_size_z, minimum_distance_z, ],
        'is_multichannel' : [channel_to_compute],
        'do_dense_region_deconvolution' : [alpha,beta,gamma, deconvolution_kernel_x, deconvolution_kernel_y],
        'do_clustering' : [cluster_size, min_number_of_spot],
        'always_hidden' : [interactive_threshold_selector, ],
        'is_3D&do_denseregion' : [deconvolution_kernel_z],
        'do_segmentation' : [nucleus_channel_signal]
        
    }

    for key, enabled in update_dict.items() :
        for elmt in list_dict.get(key) :
            elmt.update(disabled=not enabled)

    tab_elmt.update(visible=is_mapping_ok)

def update_segmentation_tab(
        tab_elmt : sg.Tab, 
        segmentation_correct_text : sg.Text, 
        do_segmentation : bool, 
        is_multichannel : bool, 
        is_3D : bool, 
        is_mapping_ok: bool
        ) : 
    
    #Access elements
    cytoplasm_channel_elmt = get_elmt_from_key(tab_elmt, key= 'cytoplasm_channel')
    nucleus_channel_elmt = get_elmt_from_key(tab_elmt, key= 'nucleus_channel')
    do_nucleus_3D_elmt = get_elmt_from_key(tab_elmt, key= "nucleus_segmentation_3D")
    do_cytoplasm_3D_elmt = get_elmt_from_key(tab_elmt, key= "cytoplasm_segmentation_3D")
    
    #Update values
    cytoplasm_channel_elmt.update(disabled = not is_multichannel)
    nucleus_channel_elmt.update(disabled = not is_multichannel)
    do_nucleus_3D_elmt.update(disabled = not is_3D)
    do_cytoplasm_3D_elmt.update(disabled = not is_3D)
    segmentation_correct_text.update(visible= do_segmentation)

    tab_elmt.update(visible=is_mapping_ok and do_segmentation)

def update_map_tab(
        tab_elmt : sg.Tab,
        is_3D,
        is_multichannel,
        last_shape,

) :
    #Acess elments
    t_element = get_elmt_from_key(tab_elmt, key= 't')
    c_element = get_elmt_from_key(tab_elmt, key= 'c')
    z_element = get_elmt_from_key(tab_elmt, key= 'z')
    automap_element = get_elmt_from_key(tab_elmt, key= 'auto-map')
    apply_element = get_elmt_from_key(tab_elmt, key= 'apply-map')

    #Update values
    t_element.update(disabled=True)
    c_element.update(disabled=not is_multichannel)
    z_element.update(disabled=not is_3D)
    automap_element.update(disabled=type(last_shape) == type(None))
    apply_element.update(disabled=type(last_shape) == type(None))

def update_output_tab(
        tab_elmt : sg.Tab,
        do_segmentation,
        output_folder,
) :
    
    #Segmentation
    segmentation_box = get_elmt_from_key(tab_elmt, "save segmentation")
    segmentation_box.update(disabled = not do_segmentation)
    segmentation_save_masks_elmt = get_elmt_from_key(tab_elmt, "save_masks")
    segmentation_save_masks_elmt.update(disabled = not do_segmentation)

    batch_folder_text = get_elmt_from_key(tab_elmt, "batch_folder_text")
    batch_folder_text.update(value = output_folder)