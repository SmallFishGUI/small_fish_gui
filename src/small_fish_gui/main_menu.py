"""
This script is called when software starts; it is the main loop.
"""
import traceback, os
from small_fish_gui import __version__

import pandas as pd
import FreeSimpleGUI as sg

from .pipeline.actions import add_detection 
from .pipeline.actions import save_results 
from .pipeline.actions import compute_colocalisation 
from .pipeline.actions import delete_acquisitions, rename_acquisitions 
from .pipeline.actions import save_segmentation, load_segmentation, segment_cells
from .pipeline.actions import open_wiki

from .pipeline._preprocess import clean_unused_parameters_cache
from .batch import batch_promp
from .gui import hub_prompt, prompt_restore_main_menu, default_theme
from .hints import pipeline_parameters

#'Global' parameters
user_parameters = pipeline_parameters({'segmentation_done' : False}) #TypedDict
acquisition_id = -1
result_df = pd.DataFrame(columns=['acquisition_id', 'name'])
cell_result_df = pd.DataFrame(columns=['acquisition_id'])
global_coloc_df = pd.DataFrame()
cell_coloc_df = dict()
cytoplasm_label = None
nucleus_label = None

default_theme()
while True : #Break this loop to close small_fish
    try :
        result_df = result_df.reset_index(drop=True)

        event, values = hub_prompt(result_df, user_parameters['segmentation_done'])

        if event == 'Add detection' :
            user_parameters = clean_unused_parameters_cache(user_parameters)

            new_result_df, new_cell_result_df, acquisition_id, user_parameters =  add_detection(
                user_parameters=user_parameters,
                acquisition_id=acquisition_id,
                cytoplasm_label = cytoplasm_label,
                nucleus_label = nucleus_label,
                )
            result_df = pd.concat([result_df, new_result_df], axis=0).reset_index(drop=True)
            cell_result_df = pd.concat([cell_result_df, new_cell_result_df], axis=0).reset_index(drop=True)

        elif event == 'Segment cells' :
            nucleus_label, cytoplasm_label, user_parameters = segment_cells(
                user_parameters=user_parameters,
                nucleus_label=nucleus_label,
                cytoplasm_label = cytoplasm_label,
            )

        elif event == 'Save results' :
            save_results(
                result_df=result_df,
                cell_result_df=cell_result_df,
                global_coloc_df=global_coloc_df,
                cell_coloc_df = cell_coloc_df,
            )
        
        elif event == 'Save segmentation' :
            save_segmentation(
                nucleus_label=nucleus_label,
                cytoplasm_label=cytoplasm_label,
            )

        elif event == 'Load segmentation' :
            nucleus_label, cytoplasm_label, user_parameters['segmentation_done'] = load_segmentation(nucleus_label, cytoplasm_label, user_parameters['segmentation_done'])
        
        elif event == 'Compute colocalisation' :

            global_coloc_df, cell_coloc_df, acquisition_id = compute_colocalisation(
                result_dataframe=result_df,
                cell_result_dataframe=cell_result_df,
                global_coloc_df=global_coloc_df,
                cell_coloc_df=cell_coloc_df,
                max_id=acquisition_id,
            )

        elif event == "Reset all" :
            result_df = pd.DataFrame(columns=['acquisition_id'])
            cell_result_df = pd.DataFrame(columns=['acquisition_id'])
            global_coloc_df = pd.DataFrame()
            cell_coloc_df = dict()
            acquisition_id = -1
            user_parameters['segmentation_done'] = False
            cytoplasm_label = None
            nucleus_label = None

        elif event == "Reset segmentation" :
            user_parameters['segmentation_done'] = False
            cytoplasm_label = None
            nucleus_label = None

        elif event == "Delete acquisitions" :
            selected_acquisitions = values.setdefault('result_table', []) #Contains the lines selected by the user on the sum-up array.
            result_df, cell_result_df, global_coloc_df, cell_coloc_df = delete_acquisitions(selected_acquisitions, result_df, cell_result_df, global_coloc_df, cell_coloc_df)

        elif event == "Batch detection" :
            result_df, cell_result_df, acquisition_id, user_parameters, user_parameters['segmentation_done'], cytoplasm_label,nucleus_label = batch_promp(
                result_df,
                cell_result_df,
                acquisition_id=acquisition_id,
                preset=user_parameters,
            )
        
        elif event == "Rename acquisition" :
            selected_acquisitions = values.setdefault('result_table', []) #Contains the lines selected by the user on the sum-up array.
            result_df, cell_result_df, global_coloc_df, cell_coloc_df = rename_acquisitions(selected_acquisitions, result_df, cell_result_df, global_coloc_df, cell_coloc_df)

        elif event == "wiki" :
            open_wiki()

        else :
            break
        

    except Exception as error :
        sg.popup(str(error))

        with open("error_log.txt",'a') as error_log :
            error_log.writelines([
                f"version {__version__}",
                f"error : {error}",
                f"traceback :\n{traceback.format_exc()}",
            ])

        print(f"error_log saved at {os.getcwd()}/error_log.txt. Please consider reporting this by opening an issue on github.")

        save_quit = prompt_restore_main_menu()

        if save_quit is None :
            raise error

        elif save_quit :
            save_results(
                result_df=result_df,
                cell_result_df=cell_result_df,
                global_coloc_df=global_coloc_df,
                cell_coloc_df = cell_coloc_df,
            )
            quit()
        else :
            continue

quit()