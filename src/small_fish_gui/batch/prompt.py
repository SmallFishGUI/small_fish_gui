"""
Submodule with main function to call to launch batch mode.
"""

import os
import FreeSimpleGUI as sg


from .utils import get_elmt_from_key, create_map, call_auto_map
from .pipeline import batch_pipeline
from .update import update_detection_tab, update_map_tab, update_master_parameters, update_segmentation_tab, update_output_tab
from .input import load, extract_files
from .integrity import sanity_check, check_channel_map_integrity, check_detection_parameters, check_segmentation_parameters, check_output_parameters
from ..gui.layout import _segmentation_layout, _detection_layout, _input_parameters_layout, _ask_channel_map_layout
from ..interface import get_settings

def batch_promp(
        results_df,
        cell_results_df,
        acquisition_id,
        preset: dict={},
) :

    files_values = [[]]
    default = get_settings()

    #LOAD FILES
    files_table = sg.Table(
        values=files_values,
        headings=['Filenames'],
        col_widths=100,
        max_col_width= 200,
        def_col_width=100,
        num_rows= 10,
        auto_size_columns=False, 
        expand_y=True,
        justification='left',
        )

    #DIMENSION SANITY
    sanity_progress = sg.ProgressBar(10, size_px=(500,10), border_width=2)
    sanity_check_button = sg.Button(
        'Check', 
        tooltip= "Will check that all files loaded have the same dimension number and that small fish is able to open them.", 
        pad=(10,0))
    sanity_header = sg.Text("Dimension sanity", font=('bold',15), pad=(0,10))
    dimension_number_text = sg.Text("Dimension number : unknown")
    
#LAYOUT INIT
#########################################
#####   COLUMNS
#########################################

#####   Tabs

    #Input tab
    input_layout = _input_parameters_layout(
        ask_for_segmentation=True,
        is_3D_stack_preset= preset.setdefault("is_3D_stack" ,default.stack_3D),
        time_stack_preset=False,
        multichannel_preset=preset.setdefault("is_multichannel" ,default.multichannel_stack),
        do_dense_regions_deconvolution_preset=preset.setdefault("do_dense_regions_deconvolution", default.do_dense_regions_deconvolution),
        do_clustering_preset= preset.setdefault("do_cluster_computation", default.do_cluster),
        do_Napari_correction=False,
        do_segmentation_preset= preset.setdefault("segmentation_done", False),
    )
    input_layout += [[sg.Button('Ok')]]
    input_tab = sg.Tab("Input", input_layout)

    napari_correction_elmt = get_elmt_from_key(input_tab, key='show_napari_corrector')

    #Maptab
    map_layout = _ask_channel_map_layout(
        shape=(0,1,2,3,4),
        is_3D_stack=True,
        is_time_stack=True,
        multichannel=True,
    )
    last_shape_read = sg.Text("Last shape read : None")
    auto_map = sg.Button("auto-map", disabled=True, pad=(10,0))
    apply_map_button = sg.Button("apply", disabled=True, pad=(10,0), key='apply-map')
    map_layout += [[last_shape_read]]
    map_layout += [[auto_map, apply_map_button]]
    map_tab = sg.Tab("Map", map_layout)

    #Segmentation tab
    segmentation_layout = _segmentation_layout(
        multichannel=True,
        is_3D_stack=True, 
        cytoplasm_model_preset=preset.setdefault("cyto_model_name",default.cytoplasm_model),
        cytoplasm_channel_preset=preset.setdefault("cytoplasm_channel",default.detection_channel),
        cyto_diameter_preset=preset.setdefault("cytoplasm_diameter",default.cytoplasm_diameter),
        nucleus_model_preset=preset.setdefault("nucleus_model_name",default.nucleus_model),
        nucleus_channel_preset=preset.setdefault("nucleus channel",default.nucleus_channel),
        nucleus_diameter_preset=preset.setdefault("nucleus_diameter",default.nucleus_diameter),
        segment_only_nuclei_preset=preset.setdefault("segment_only_nuclei",default.only_nuclei),
        filename_preset="",
        cytoplasm_segmentation_3D= preset.setdefault("cytoplasm_segmentation_3D", default.do_3D_segmentation),
        nucleus_segmentation_3D=preset.setdefault("nucleus_segmentation_3D", default.do_3D_segmentation),
        anisotropy=preset.setdefault("anisotropy", default.anisotropy),
        flow_threshold=preset.setdefault("flow_threshold",default.flow_threshold),
        cellprob_threshold=preset.setdefault("cellprob_threshold", default.cellprob_threshold),
        show_segmentation_preset=False,
        save_segmentation_visual_preset=False,
        saving_path_preset="",
        other_nucleus_image_preset=""
        )
    
    apply_segmentation_button = sg.Button('apply', key='apply-segmentation')
    segmentation_layout += [[apply_segmentation_button]]
    seg_keys_to_hide = ['show_segmentation', 'saving path', 'filename', 'other_nucleus_image', 'save_segmentation_visual', 'saving path_browse']
    segmentation_tab = sg.Tab("Segmentation", segmentation_layout, visible=False)

    #Detection tab
    detection_layout = _detection_layout(
        is_3D_stack=True,
        is_multichannel=True,
        do_clustering=True,
        do_dense_region_deconvolution=True,
        do_segmentation=True,
        default_dict= preset,
    )
    apply_detection_button = sg.Button('apply', key='apply-detection')
    detection_layout += [[apply_detection_button]]
    detection_keys_to_hide = ['do_spots_csv', 'do_spots_excel', 'spots_filename','spots_extraction_folder', 'spots_extraction_folder_browse']
    detection_tab = sg.Tab("Detection", detection_layout, visible=False)

    #Output tab
    show_batch_folder_text = sg.Text('', key= 'batch_folder_text')
    apply_output_button = sg.Button('apply', key='apply-output')
    save_segmentation_visual_box = sg.Checkbox("save segmentation", disabled=True, key='save segmentation', tooltip= "Save png files illustrating segmentation borders.")
    save_segmentation_masks_box = sg.Check("save labels",  disabled=True, key="save_masks", tooltip= "Save segmentation labels as .npy files.")
    save_detection_box = sg.Checkbox("create spot detection visuals", key= 'save detection', tooltip="Create multichannel tiff with raw spot signal and detected spots.\nWarning if processing a lot of files make sure you have enough free space on your hard drive.")
    extract_spots_box = sg.Checkbox("extract spots", key='extract spots')
    batch_name_input = sg.InputText(size=25, key='batch_name')
    output_layout=[
        [sg.Text("Output folder", font=('bold',15), pad=(0,10))],
        [show_batch_folder_text],
        [sg.Text("Select a folder : "), sg.FolderBrowse(initial_folder=os.getcwd(), key='output_folder', target=(1,-1))],
        [sg.Text("Name for batch : "), batch_name_input],
        [save_detection_box],
        [extract_spots_box],
        [sg.Text("Data extension", font=('bold',15), pad=(0,10))],
        [sg.Checkbox(".csv", key='csv'),sg.Checkbox(".xlsx", key='xlsx')],
        [sg.Text("Segmentation", font=('bold',15), pad=(0,10))],
        [save_segmentation_visual_box],
        [save_segmentation_masks_box],
        [apply_output_button],
    ]

    output_tab = sg.Tab("Output", output_layout, visible=True)

    ##TAB GROUP
    _tab_group = sg.TabGroup([[input_tab, map_tab, segmentation_tab, detection_tab, output_tab]], enable_events=True)
    tab_col = sg.Column( #Allow the tab to be scrollable
        [[_tab_group]],
        scrollable=True,
        vertical_scroll_only=False,
        s= (390,390),
        pad=((0,0),(5,5)),
        expand_x=True,
        )
    
    tab_dict= {
        "Input" : input_tab,
        "Segmentation" : segmentation_tab,
        "Detection" : detection_tab,
        "Map" : map_tab,
        "Output" : output_tab
    }
    
#####   Launcher

    start_button =sg.Button('Start', button_color= 'green', disabled= True, pad= ((115,5),(0,10)))
    stop_button = sg.Button('Cancel', button_color= 'red', pad= ((5,115),(0,10)))
    batch_progression_bar = sg.ProgressBar(max_value=0, size_px=(340,20), bar_color= ('blue','black'), border_width=2, pad=((15,0),(30,0)))
    mapping_ok_text = sg.Text('Uncorrect mapping', text_color='gray', font='roman 14 bold', pad=((0,0),(100,5)))
    segmentation_ok_text = sg.Text('Uncorrect segmentation settings', font='roman 14 bold', pad=(0,5), text_color='gray', visible=False)
    detection_ok_text = sg.Text('Uncorrect detection settings', text_color='gray', font='roman 14 bold', pad=(0,5))
    output_ok_text = sg.Text('Uncorrect output parameters', text_color='gray', font='roman 14 bold', pad=(0,5))
    current_acquisition_text = sg.Text('0', text_color='gray', font='roman 15 bold', pad= ((150,5),(10,10)))
    total_acquisition_text = sg.Text('/ 0', text_color='gray', font='roman 15 bold', pad= ((5,150),(10,10)))

    launcher_layout = [
        [mapping_ok_text],
        [segmentation_ok_text],
        [detection_ok_text],
        [output_ok_text],
        [batch_progression_bar],
        [current_acquisition_text,total_acquisition_text],
        [start_button, stop_button],
    ]
    launch_col = sg.Column( #Allow the tab to be scrollable
        launcher_layout,
        s= (390,390),
        pad=((3,5),(5,5))
        )

#########################################
#####   Window Creation
#########################################
    stream_output = sg.Output(size=(100,10), pad=(30,10), visible=True, expand_x=True, expand_y=True)
    layout = [
        [sg.Text("Batch Processing", font=('bold',20), pad=((300,0),(0,2)))],
        [sg.Text("Select a folder : "), sg.FolderBrowse(initial_folder=os.getcwd(), key='Batch_folder'), sg.Button('Load')],
        [files_table],
        [sanity_header, sanity_check_button, sanity_progress],
        [dimension_number_text],
        [tab_col, sg.Push(), launch_col, sg.Push()],
        [stream_output],
    ]

    window = sg.Window("small fish", layout=layout, size= (800,800), auto_size_buttons=True, auto_size_text=True, resizable=True)
    
    #MASTER PARAMETERS
    Master_parameters_dict ={
        '_map' : {},
        '_is_mapping_correct' : False,
        '_is_segmentation_correct' : None, # None : segmentation disabled; Then true/false if enabled
        '_is_detection_correct' : False,
        '_is_output_correct' : False,
    }
    Master_parameters_update_dict = {
        '_is_mapping_correct' : mapping_ok_text,
        '_is_segmentation_correct' : segmentation_ok_text, # None : segmentation disabled; Then true/false if enabled
        '_is_detection_correct' : detection_ok_text,
        '_is_output_correct' : output_ok_text,

    }
    loop = 0
    timeout = 1
    last_shape = None
    talk=True
    
#########################################
#####   Event Loop : break to close window
#########################################
    
    #Hiding options for non batch mode
    window= window.finalize()
    napari_correction_elmt.update(disabled=True)
    get_elmt_from_key(tab_dict['Input'], key= 'image_path').update(disabled=True)
    for key in seg_keys_to_hide : get_elmt_from_key(tab_dict['Segmentation'], key=key).update(disabled=True)
    for key in detection_keys_to_hide : get_elmt_from_key(tab_dict['Detection'], key=key).update(disabled=True)
    
    while True : 
        try :
            loop +=1
            window = window.refresh()
            event, values = window.read(timeout=timeout)

            #Welcome message
            if loop == 1 : 
                timeout = 500
                print("Welcome to small fish batch analysis. Please start by loading some files and setting parameters.")

            if values is None : return results_df, cell_results_df, acquisition_id, preset, False, None,None
            
            batch_folder = values.get('Batch_folder')
            is_multichanel = values.get('is_multichannel')
            is_3D = values.get('is_3D_stack')
            do_segmentation = values.get('do_segmentation')
            do_dense_regions_deconvolution = values.get('do_dense_regions_deconvolution')
            do_clustering = values.get('do_cluster_computation')

            if type(batch_folder) != type(None)  and event == 'Load':

                files_values, last_shape, dim_number = load(batch_folder)
                files_table.update(values=files_values)
                last_shape_read.update("Last shape read : {0}".format(last_shape))
                dimension_number_text.update("Dimension number : {0}".format(dim_number))
                update_map_tab(
                    tab_elmt=tab_dict.get("Map"),
                    is_3D=is_3D,
                    is_multichannel=is_multichanel,
                    last_shape=last_shape
                )
                filename_list = extract_files(files_values)

            elif event == 'Check' :
                last_shape = sanity_check(
                    filename_list=filename_list,
                    batch_folder=batch_folder,
                    window=window,
                    progress_bar=sanity_progress
                )
                if isinstance(last_shape,(tuple,list)) :
                    dim_number = len(last_shape)
                    dimension_number_text.update("Dimension number : {0}".format(dim_number))
                    auto_map.update(disabled=False)
                else :
                    dim_number = None
                    dimension_number_text.update("Dimension number : unknown")
                    auto_map.update(disabled=True)

                last_shape_read.update("Last shape read : {0}".format(last_shape))

            elif event == _tab_group.key or event == 'Ok': #Tab switch in parameters

                update_map_tab(
                    tab_elmt=tab_dict.get("Map"),
                    is_3D=is_3D,
                    is_multichannel=is_multichanel,
                    last_shape=last_shape
                )

            elif event == 'auto-map' :
                Master_parameters_dict['_map'] = call_auto_map(
                    tab_elmt=tab_dict.get("Map"),
                    shape=last_shape,
                    is_3D=is_3D,
                    is_multichannel=is_multichanel
                )

                Master_parameters_dict['_is_mapping_correct'] = check_channel_map_integrity(
                    maping=Master_parameters_dict['_map'],
                    shape=last_shape,
                    expected_dim=dim_number
                    )

                if not Master_parameters_dict['_is_mapping_correct'] : Master_parameters_dict['_map'] = {}

            elif event == 'apply-map' :

                Master_parameters_dict['_map'] = create_map(
                    values=values,
                    is_3D=is_3D,
                    is_multichannel=is_multichanel,
                )

                Master_parameters_dict['_is_mapping_correct'] = check_channel_map_integrity(
                    maping=Master_parameters_dict['_map'],
                    shape=last_shape,
                    expected_dim=dim_number
                    )

                if not Master_parameters_dict['_is_mapping_correct'] : Master_parameters_dict['_map'] = {}

            elif event == 'apply-segmentation' :
                Master_parameters_dict['_is_segmentation_correct'], values = check_segmentation_parameters(
                    values=values,
                    shape=last_shape,
                    is_multichannel=is_multichanel
                )

            elif event == 'apply-detection' :
                Master_parameters_dict['_is_detection_correct'], values = check_detection_parameters(
                    values=values,
                    do_dense_region_deconvolution=do_dense_regions_deconvolution,
                    do_clustering=do_clustering,
                    is_multichannel=is_multichanel,
                    is_3D=is_3D,
                    map_= Master_parameters_dict.get('_map'),
                    shape=last_shape
                )

            elif event == 'apply-output' :
                Master_parameters_dict['_is_output_correct'], values = check_output_parameters(values)
                batch_name_input.update(value=values.get('batch_name'))

            elif event == "Cancel" :
                stream_output.restore_stderr()
                stream_output.restore_stdout()
                window.close()
                return results_df, cell_results_df, acquisition_id, preset, False, None,None #Segmentation done : False, cell label : None, Nucleus label : None


            elif event == None :
                raise InterruptedError("closed")
            elif event == sg.WINDOW_CLOSED : 
                raise InterruptedError("closed")

            elif event == 'Start' :
                start_button.update(disabled=True)
                results_df, cell_results_df, acquisition_id = batch_pipeline(
                    batch_window= window,
                    batch_progress_bar= batch_progression_bar,
                    progress_count=current_acquisition_text,
                    parameters=values,
                    filenames_list=filename_list,
                    do_segmentation=do_segmentation,
                    map_= Master_parameters_dict['_map'],
                    results_df=results_df,
                    cell_results_df=cell_results_df,
                    is_3D=is_3D,
                    last_acquisition_id=acquisition_id+1,
                )
                stream_output.restore_stderr()
                stream_output.restore_stdout()

                window.close()
                return results_df, cell_results_df, acquisition_id, values, False, None,None #Segmentation done : False, cell label : None, Nucleus label : None


            #End of loop

            #If user loads other files
            if type(last_shape) != type(None) :
                if len(last_shape) != 2 + is_3D + is_multichanel :
                    Master_parameters_dict['_is_mapping_correct'] = False
                    Master_parameters_dict['_is_detection_correct'] = False
                    Master_parameters_dict['_is_segmentation_correct'] = False
            else : 
                Master_parameters_dict['_is_mapping_correct'] = False
                Master_parameters_dict['_is_detection_correct'] = False
                Master_parameters_dict['_is_segmentation_correct'] = False

            update_master_parameters(
                Master_parameter_dict=Master_parameters_dict,
                update_dict=Master_parameters_update_dict
            )

            update_segmentation_tab(
                tab_elmt=tab_dict.get("Segmentation"),
                segmentation_correct_text= segmentation_ok_text,
                do_segmentation=do_segmentation,
                is_multichannel=is_multichanel,
                is_3D=is_3D,
                is_mapping_ok=Master_parameters_dict['_is_mapping_correct']
            )

            update_detection_tab(
                tab_elmt=tab_dict.get("Detection"),
                is_multichannel=is_multichanel,
                is_3D=is_3D,
                do_dense_region_deconvolution=do_dense_regions_deconvolution,
                do_clustering=do_clustering,
                is_mapping_ok=Master_parameters_dict['_is_mapping_correct'],
                do_segmentation=do_segmentation,
            )

            update_output_tab(
                tab_elmt=tab_dict['Output'],
                do_segmentation=do_segmentation,
                output_folder=values['output_folder'],
            )

            _segmentation_ready = (Master_parameters_dict['_is_segmentation_correct'] and do_segmentation) or not do_segmentation
            if Master_parameters_dict['_is_detection_correct'] and Master_parameters_dict['_is_output_correct'] and Master_parameters_dict['_is_mapping_correct'] and _segmentation_ready :
                start_button.update(button_color='blue', disabled=False)
                total_acquisition_text.update("/ {0}".format(len(filename_list)))
                batch_progression_bar.update(current_count=0, bar_color=('green','blue'))
                if talk : 
                    print("Small fish is ready, launch with the start button.")
                    talk= False
            else :
                talk = True
                start_button.update(button_color='green', disabled=True)
                total_acquisition_text.update("/ 0")
                batch_progression_bar.update(current_count=0, bar_color=('blue','black'))

        except Exception as e :
            stream_output.restore_stderr()
            stream_output.restore_stdout()
            window.close()
            raise e