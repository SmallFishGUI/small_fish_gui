import FreeSimpleGUI as sg
import os
import cellpose.models as models
from typing import Optional, Union

from cellpose.core import use_gpu
from .tooltips import FLOW_THRESHOLD_TOOLTIP,CELLPROB_TOOLTIP
from ..hints import pipeline_parameters
from ..utils import check_parameter
from ..interface import SettingsDict, get_default_settings, get_settings


settings = get_settings()

def add_header(header_text) :
    """Returns [elmnt] not layout"""
    header = [sg.Text('\n{0}'.format(header_text), size= (len(header_text),3), font= 'bold 15')]
    return header


def pad_right(string, length, pad_char) :
    if len(string) >= length : return string
    else : return string + pad_char* (length - len(string))
    

def parameters_layout(
        parameters:'list[str]' = [], 
        unit=None, 
        header= None, 
        default_values=None, 
        keys = None,
        tooltips = None,
        size=5, 
        opt:list=None
        ) :

    if len(parameters) == 0 : return []
    check_parameter(parameters= list, header = (str, type(None)))
    for key in parameters : check_parameter(key = str)
    max_length = len(max(parameters, key=len))

    if keys is None : keys = [None] * len(parameters)
    else :
        if len(keys) != len(parameters) : raise ValueError("keys length must be equal with parameters length or set to None.")
    if tooltips is None : tooltips = [None] * len(parameters)
    else :
        if len(tooltips) != len(parameters) : raise ValueError("tooltips length must be equal with parameters length or set to None.")

    if type(opt) == type(None) :
        opt = [False] * len(parameters)
    else :
        if len(opt) != len(parameters) : raise ValueError("Parameters and opt must be of same length.")

    if isinstance(default_values, (list, tuple)) :
        if len(default_values) != len(parameters) : raise ValueError("if default values specified it must be of equal length as parameters.")
        layout= [
            [
                sg.Text("{0}".format(pad_right(parameter, max_length, ' ')), text_color= 'green' if option else None), 
                sg.InputText(size= size, key= parameter if key is None else key, default_text= value, tooltip=tooltip)
            
            ] for parameter,value, option, key, tooltip in zip(parameters,default_values, opt, keys, tooltips)
        ]
    else :
        layout= [
            [sg.Text("{0}".format(pad_right(parameter, max_length, ' ')), text_color= 'green' if option else None), 
             sg.InputText(size= size, key= parameter)
             
             ] for parameter, option in zip(parameters, opt)
        ]
    
    if type(unit) == str :
        for line_id, line in enumerate(layout) :
            layout[line_id] += [sg.Text('{0}'.format(unit))]
    
    if isinstance(header, str) :
        layout = [add_header(header)] + layout
    return layout

def tuple_layout(opt=None, default_dict={}, unit:dict={}, names : dict = {}, **tuples) :
    """
    tuples example : voxel_size = ['z','y','x']; will ask a tuple with 3 element default to 'z', 'y' 'x'.
    """
    if type(tuples) == type(None) : return []
    if len(tuples.keys()) == 0 : return []

    if type(opt) != type(None) :
        if not isinstance(opt, dict) : raise TypeError("opt parameter should be either None or dict type.")
        if not opt.keys() == tuples.keys() : raise ValueError("If opt is passed it is expected to have same keys as tuples dict.")
    else : 
        opt = tuples.copy()
        for key in opt.keys() :
            opt[key] = False

    for tup in tuples : 
        if not isinstance(tuples[tup], (list,tuple)) : raise TypeError()

    max_size = len(max(tuples.keys(), key=len))
    
    layout = [
        [sg.Text(pad_right(names[tup] if tup in names.keys() else tup, max_size, ' '), text_color= 'green' if opt[option] else None)] 
        + [sg.InputText(default_text=default_dict.setdefault('{0}_{1}'.format(tup,elmnt), elmnt),key= '{0}_{1}'.format(tup, elmnt), size= 5) for elmnt in tuples[tup]]
        + [sg.Text(unit.setdefault(tup,''))]
        for tup,option, in zip(tuples,opt)
    ]

    return layout

def path_layout(keys= [],look_for_dir = False, header=None, preset=settings.working_directory) :
    """
    If not look for dir then looks for file.
    """
    if len(keys) == 0 : return []
    check_parameter(keys= list, header = (str, type(None)))
    for key in keys : check_parameter(key = str)
    if look_for_dir : Browse = sg.FolderBrowse
    else : Browse = sg.FileBrowse

    max_length = len(max(keys, key=len))

    layout = []
    for name in keys :
        layout += [
            [sg.Text(pad_right(name, max_length, ' '))], 
            [sg.InputText(key= name, expand_x=True, default_text=preset), Browse(key= name + "_browse", initial_folder= preset), sg.Text('')],
            ]
    if isinstance(header, str) :
        layout = [add_header(header)] + layout
    return layout

def bool_layout(parameters= [], header=None, preset : Optional[Union['list[bool]',bool,None]]=None, keys=None) :
    if len(parameters) == 0 : return []
    check_parameter(parameters= list, header= (str, type(None)), preset=(type(None), list, tuple, bool))
    for key in parameters : check_parameter(key = str)
    if type(preset) == type(None) :
        preset = [False] * len(parameters)
    elif type(preset) == bool :
        preset = [preset] * len(parameters)
    else : 
        for key in preset : check_parameter(key = bool)

    max_length = len(max(parameters, key=len))

    if type(keys) == type(None) :
        keys = parameters
    elif isinstance(keys,(list,tuple)) :
        if len(keys) != len(parameters) : raise ValueError("keys arguement must be of same length than parameters argument")
    elif len(parameters) == 1 and isinstance(keys, str) :
        keys = [keys]
    else :
        raise ValueError('Incorrect keys parameters. Expected list of same length than parameters or None.')

    layout = [
        [sg.Checkbox(pad_right(name, max_length, ' '), key=key, default=box_preset)] for name, box_preset, key in zip(parameters,preset,keys)
    ]
    if isinstance(header, str) :
        layout = [add_header(header)] + layout
    return layout

def combo_elmt(values, key, header=None, read_only=True, default_value=None) :
    """
    drop-down list
    """
    if len(values) == 0 : return []
    check_parameter(values= list, header= (str, type(None)))
    if type(default_value) == type(None) :
        default_value = values[0]
    elif default_value not in values :
        default_value = values[0]
    layout = [
        sg.Combo(values, default_value=default_value, readonly=read_only, key=key)
    ]
    if isinstance(header, str) :
        layout = add_header(header) + layout
    return layout

def radio_layout(values, header=None, key=None) :
    """
    Single choice buttons.
    """
    if len(values) == 0 : return []
    check_parameter(values= list, header= (str, type(None)))
    layout = [
        [sg.Radio(value, group_id= 0, key=key) for value in values]
    ]
    if isinstance(header, str) :
        layout = [add_header(header)] + layout
    return layout

def _segmentation_layout(
        multichannel : bool,
        is_3D_stack : bool,
        cytoplasm_model_preset,
        nucleus_model_preset,
        cytoplasm_channel_preset,
        nucleus_channel_preset,
        other_nucleus_image_preset,
        cyto_diameter_preset,
        nucleus_diameter_preset,
        show_segmentation_preset,
        save_segmentation_visual_preset, 
        segment_only_nuclei_preset,
        saving_path_preset,
        filename_preset,
        cytoplasm_segmentation_3D ,
        nucleus_segmentation_3D ,
        cellprob_threshold ,
        flow_threshold ,
        anisotropy,
        ) :
    
    USE_GPU = use_gpu()

    models_list = models.get_user_models() + models.MODEL_NAMES
    if len(models_list) == 0 : models_list = ['no model found']
    
    #Header : GPU availabality
    layout = [
        [sg.Text("GPU is currently "), sg.Text('ON', text_color= 'green') if USE_GPU else sg.Text('OFF', text_color= 'red')]
        ]
    
    #cytoplasm parameters
    layout += [
        add_header("Cytoplasm Segmentation"),
        [sg.Text("Choose parameters for cytoplasm segmentation: \n")],
                        ]
                        
    if is_3D_stack : layout += bool_layout(['3D segmentation'], preset=[cytoplasm_segmentation_3D], keys=['cytoplasm_segmentation_3D'],)
    if multichannel : layout += parameters_layout(['Cytoplasm channel'],default_values= [cytoplasm_channel_preset], keys = ["cytoplasm_channel"])

    layout += [[sg.Text("Cellpose model : ")] + combo_elmt(models_list, key='cyto_model_name', default_value= cytoplasm_model_preset)]
    layout += parameters_layout(['Cytoplasm diameter'], unit= "px", default_values= [cyto_diameter_preset], keys=['cytoplasm_diameter'])
    layout += parameters_layout(
        ["Flow threshold", "Cellprob threshold"], 
        default_values=[flow_threshold, cellprob_threshold], 
        keys=["flow_threshold_cyto","cellprob_threshold_cyto"],
        tooltips= [FLOW_THRESHOLD_TOOLTIP, CELLPROB_TOOLTIP]
        )

    #Nucleus parameters
    layout += [
            add_header("Nuclei segmentation"),
            [sg.Text("Choose parameters for nuclei segmentation: \n")],
                ]
    
    layout += path_layout(['other_nucleus_image'], preset=other_nucleus_image_preset)
    if is_3D_stack : layout += bool_layout(['3D segmentation'], preset=[nucleus_segmentation_3D], keys=['nucleus_segmentation_3D'],)
    if multichannel : layout += parameters_layout(['nucleus_channel'], default_values= [nucleus_channel_preset])
    layout += bool_layout(["Segment only nuclei"], preset=segment_only_nuclei_preset, keys=["segment_only_nuclei"])
    
    layout += [[sg.Text("Cellpose model : ")] + combo_elmt(models_list, key='nucleus_model_name', default_value= nucleus_model_preset)]
    layout += parameters_layout(['Nucleus diameter'],unit= "px", default_values= [nucleus_diameter_preset], keys=['nucleus_diameter'])
    layout += parameters_layout(
        ["Flow threshold", "Cellprob threshold"], 
        default_values=[flow_threshold, cellprob_threshold], 
        keys=["flow_threshold_nuc","cellprob_threshold_nuc"],
        tooltips= [FLOW_THRESHOLD_TOOLTIP, CELLPROB_TOOLTIP]
        )
    if is_3D_stack : layout += parameters_layout(
        ['anisotropy'], 
        header = "Model parameters",
        default_values = [anisotropy]
        )

    #Control plots
    layout += bool_layout(['Show_segmentation'],keys=['show_segmentation'], header= 'Segmentation plots', preset= show_segmentation_preset)
    layout += bool_layout(['Save segmentation visual'], preset= save_segmentation_visual_preset, keys=['save_segmentation_visual'])
    layout += path_layout(keys=['saving path'], look_for_dir=True, preset=saving_path_preset)
    layout += parameters_layout(['filename'], default_values=[filename_preset], size= 25)

    return layout

def _input_parameters_layout(
        ask_for_segmentation : bool,
        is_3D_stack_preset : bool,
        time_stack_preset : bool,
        multichannel_preset : bool,
        do_dense_regions_deconvolution_preset : bool,
        do_clustering_preset : bool,
        do_segmentation_preset : bool,
        do_background_removal_preset : bool,
        do_Napari_correction : bool
    ) :
    layout_image_path = path_layout(['image_path'], header= "Image")
    layout_image_path += bool_layout(['3D stack', 'Multichannel stack'], keys=['is_3D_stack', 'is_multichannel'], preset= [is_3D_stack_preset, multichannel_preset])
    
    layout_image_path += bool_layout(
        ['Dense regions deconvolution', 'Compute clusters', 'Cell segmentation','Autofluorescence background removal', 'Open Napari corrector'],
        keys= ['do_dense_regions_deconvolution', 'do_cluster_computation', 'do_segmentation', 'do_background_removal' ,'show_napari_corrector' ], 
        preset= [do_dense_regions_deconvolution_preset, do_clustering_preset, do_segmentation_preset, do_background_removal_preset, do_Napari_correction], 
        header= "Pipeline settings")


    return layout_image_path

def _detection_layout(
        is_3D_stack,
        is_multichannel,
        do_dense_region_deconvolution,
        do_clustering,
        do_segmentation,
        segmentation_done=False,
        default_dict : pipeline_parameters={},
    ) :
    if is_3D_stack : dim = 3
    else : dim = 2
    default = get_settings()

    #Detection
    detection_parameters = ['threshold', 'threshold penalty']
    default_detection = [default_dict.setdefault('threshold',default.threshold), default_dict.setdefault('threshold penalty', default.threshold_penalty)]
    opt= [True, True]
    parameters_keys = ['threshold', 'threshold_penalty']
    if is_multichannel : 
        detection_parameters += ['channel to compute']
        opt += [False]
        parameters_keys += ['channel_to_compute']
        default_detection += [default_dict.setdefault('channel_to_compute', default.detection_channel)]
    
    layout = [[sg.Text("Green parameters", text_color= 'green'), sg.Text(" are optional parameters.")]]
    layout += parameters_layout(detection_parameters, header= 'Detection', opt=opt, default_values=default_detection, keys= parameters_keys)
    
    if dim == 2 : tuple_shape = ('y','x')
    else : tuple_shape = ('z','y','x')
    opt = {'voxel_size' : False, 'spot_size' : False, 'log_kernel_size' : True, 'minimum_distance' : True}
    unit = {'voxel_size' : 'nm', 'minimum_distance' : 'nm', 'spot_size' : 'radius(nm)', 'log_kernel_size' : 'px'}
    names = {'voxel_size' : 'Voxel size', 'spot_size' : 'Spot size', 'log_kernel_size' : 'LoG kernel size', 'minimum_distance' : 'Minimum distance between spots'}

    layout += tuple_layout(opt=opt, unit=unit, default_dict=default_dict, names=names, voxel_size= tuple_shape, spot_size= tuple_shape, log_kernel_size= tuple_shape, minimum_distance= tuple_shape)
    
    if (do_segmentation and is_multichannel) or (is_multichannel and segmentation_done):
        layout += [[sg.Text("nucleus channel signal "), sg.InputText(default_text=default_dict.setdefault('nucleus_channel',default.nucleus_channel), key= "nucleus channel signal", size= 5, tooltip= "Channel from which signal will be measured for nucleus features, \nallowing you to measure signal from a different channel than the one used for segmentation.")]]
    
    #Deconvolution
    if do_dense_region_deconvolution :
        default_dense_regions_deconvolution = [default_dict.setdefault('alpha',default.alpha), default_dict.setdefault('beta',default.beta)]
        layout += parameters_layout(['alpha', 'beta',], default_values= default_dense_regions_deconvolution, header= 'do_dense_regions_deconvolution')
        layout += parameters_layout(['gamma'], unit= 'px', default_values= [default_dict.setdefault('gamma',default.gamma)])
        layout += tuple_layout(opt= {"deconvolution_kernel" : True}, unit= {"deconvolution_kernel" : 'px'}, default_dict=default_dict, deconvolution_kernel = tuple_shape)
    
    #Clustering
    if do_clustering :
        layout += parameters_layout(['Cluster radius'],keys=['cluster_size'], unit="radius(nm)", default_values=[default_dict.setdefault('cluster_size',default.cluster_size)])
        layout += parameters_layout(['Min nb spots per cluster'],keys=['min_number_of_spots'], default_values=[default_dict.setdefault('min_number_of_spots', default.min_spot)])

    layout += bool_layout(['Interactive threshold selector'],keys = ['show_interactive_threshold_selector'], preset=[default.interactive_threshold_selector])
    layout += path_layout(
        keys=['spots_extraction_folder'],
        look_for_dir=True,
        header= "Individual spot extraction",
        preset= default_dict.setdefault('spots_extraction_folder', default.spot_extraction_folder)
    )
    default_filename = default_dict.setdefault("filename","") + "_spot_extraction"
    layout += parameters_layout(
        parameters=['spots_filename'],
        default_values=[default_filename],
        size= 13
    )
    layout += bool_layout(
        ['.csv','.excel',],
        keys= ['do_spots_csv', 'do_spots_excel'],
        preset= [default_dict.setdefault('do_spots_csv',default.do_csv), default_dict.setdefault('do_spots_excel',default.do_excel),]
    )

    return layout

def colocalization_layout(spot_list : list) :
    layout = [
        [sg.Push(), sg.Text("Co-localization", size=25, font="bold"), sg.Push()],
        [sg.VPush()],
        [sg.Text("Spots 1", size = 10)],
        [sg.DropDown(values=[""] + spot_list, key="spots1_dropdown"), sg.Input(disabled=True, text_color="black"),sg.FileBrowse("Load spot extraction", key="spots1_browse")],
        [sg.Text("Spots 2", size = 10)],
        [sg.DropDown(values=[""] + spot_list, key="spots2_dropdown"), sg.Input(disabled=True, text_color="black"),sg.FileBrowse("Load spot extraction", key="spots2_browse")],
    ]

    layout += parameters_layout(['colocalisation distance'], unit= 'nm', default_values= default.coloc_range,)

    layout += tuple_layout(opt={'voxel_size' : False},unit={'voxel_size' : "nm"}, voxel_size = ['z','y','x'], names={'voxel_size' : 'Voxel size'}, default_dict={'voxel_size' : default.voxel_size},)
    layout += [[sg.Text("   'voxel size' is used only for loaded spot lists.")]]

    layout += [[sg.Push(),sg.Button("Ok", bind_return_key=True),sg.Button("Cancel"),sg.Push(),],
        [sg.VPush()]
    ]

    return layout

def settings_layout(default_values : SettingsDict = get_default_settings()) :

    if not isinstance(default_values, SettingsDict) : raise TypeError(f"Incorect type for default_values : {type(default_values)}; expected SettingsDict")
    models_list = models.get_user_models() + models.MODEL_NAMES

    layout = [[sg.Text("Default values", font="ArialBold 20")]]
    layout += [[sg.Text("Default working directory"), sg.Input(default_text=default_values.working_directory, key= "working_directory"), sg.FolderBrowse()]]
    
    image_layout = [[sg.Text("Image", font="ArialBold 15")]]
    image_layout += bool_layout(['Multichannel stack', '3D stack'],preset= [default_values.multichannel_stack, default_values.stack_3D], keys=["multichannel_stack", "stack_3D"])
    image_layout += parameters_layout(['Detection channel', 'Nucleus channel'],default_values=[default_values.detection_channel, default_values.nucleus_channel], keys=['detection_channel', 'nucleus_channel'])

    segmentation_layout = [[sg.Text("Segmentation", font="ArialBold 15")]]
    segmentation_layout += [[sg.Text("Cytoplasm model : "), sg.DropDown(models_list, default_value=default_values.cytoplasm_model, key= "cytoplasm_model")]]
    segmentation_layout += [[sg.Text("Nucleus model : "), sg.DropDown(models_list, default_value=default_values.nucleus_model, key= "nucleus_model")]]
    segmentation_layout += parameters_layout(
        ["Flow threshold", "Cellprob threshold", "Cytoplasm diameter", "Nucleus diameter", "Anisotropy"],
        default_values= [default_values.flow_threshold, default_values.cellprob_threshold, default_values.cytoplasm_diameter, default_values.nucleus_diameter, default_values.anisotropy], 
        keys=["flow_threshold", "cellprob_threshold", "cytoplasm_diameter", "nucleus_diameter", "anisotropy"]) 
    segmentation_layout += bool_layout(
        ["show segmentation", "segment only nuclei", "do 3D segmentation", "save segmentation visual"], 
        preset=[default_values.show_segmentation, default_values.only_nuclei, default_values.do_3D_segmentation, default_values.save_segmentation_visuals],
        keys= ["show_segmentation", "only_nuclei", "do_3D_segmentation","save_segmentation_visuals"])

    detection_layout = [[sg.Text("Detection", font="ArialBold 15")]]
    detection_layout += parameters_layout(
        ["Threshold", "Threshold penalty"],
         default_values=[default_values.threshold, default_values.threshold_penalty],
         keys=["threshold", "threshold_penalty"])
    detection_layout += bool_layout(
        ["Dense regions deconvolution", "Cluster computation", "show napari corrector", "Autofloresnce background removal", "interactive threshold selector"],
        preset=[default_values.do_dense_regions_deconvolution, default_values.do_cluster, default_values.show_napari_corrector, default_values.do_background_removal, default_values.interactive_threshold_selector], 
        keys=["do_dense_regions_deconvolution", "do_cluster", "show_napari_corrector","do_background_removal", "interactive_threshold_selector"])

    deconvolution_layout = [[sg.Text("Dense regions deconvolution", font="ArialBold 15")]]
    deconvolution_layout += parameters_layout(
        ["alpha", "beta", "gamma"],
        default_values=[default_values.alpha, default_values.beta, default_values.gamma]
        )

    clustering_layout = [[sg.Text("Cluster computation", font="ArialBold 15")]]
    clustering_layout += parameters_layout(
        ["Cluster size", "Min spot number"],
        default_values=[default_values.cluster_size, default_values.min_spot], 
        keys=["cluster_size", "min_spot"])

    coloc_layout = [[sg.Text("Co-localization computation", font="ArialBold 15")]]
    coloc_layout += parameters_layout(['Co-localization range'],
    default_values=[default_values.coloc_range],
     keys=['coloc_range'])
    coloc_layout += tuple_layout(default_dict= {
        'voxel_size_z' : default_values.voxel_size[0],
        'voxel_size_y' : default_values.voxel_size[1],
        'voxel_size_x' : default_values.voxel_size[2],
        }, voxel_size=('z','y','x'))

    spot_extraction_layout = [[sg.Text("Spots extraction", font="ArialBold 15")]]
    spot_extraction_layout += bool_layout(
        ["do csv", "do excel"],
        preset=[default_values.do_csv, default_values.do_excel],
        keys=["do_csv", "do_excel"]
        )
    spot_extraction_layout += [[sg.Text("spot extraction folder"), sg.Input(default_text=default_values.spot_extraction_folder, key="spot_extraction_folder", size=7) ,sg.FolderBrowse()]]

    layout += [[sg.Col(image_layout)]]
    layout += [[sg.Col(segmentation_layout, vertical_alignment='top', expand_x = True), sg.Col(detection_layout, vertical_alignment='top', expand_x = True)]]
    layout += [[sg.Col(deconvolution_layout, vertical_alignment='top', expand_x = True), sg.Col(clustering_layout, vertical_alignment='top', expand_x = True)]]
    layout += [[sg.Col(spot_extraction_layout, vertical_alignment='top', expand_x = True), sg.Col(coloc_layout, vertical_alignment='top', expand_x = True)]]

    return layout

def _ask_channel_map_layout(
        shape,
        is_3D_stack,
        multichannel,
        is_time_stack,
        preset_map={},
    ) :
    
    x = preset_map.setdefault('x',0)
    y = preset_map.setdefault('y',0)
    z = preset_map.setdefault('z',0)
    c = preset_map.setdefault('c',0)
    t = preset_map.setdefault('t',0)

    layout = [
            [sg.Text("Dimensions mapping", font= "bold 15"), sg.Text("Image shape : {0}".format(shape))]
        ]
    layout += parameters_layout(['x','y'], default_values=[x,y])
    if is_3D_stack : layout += parameters_layout(['z'], default_values=[z])
    if multichannel : layout += parameters_layout(['c'], default_values=[c])
    if is_time_stack : layout += parameters_layout(['t'], default_values=[t])

    return layout