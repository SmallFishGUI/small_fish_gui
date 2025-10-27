import os
import pandas as pd
import numpy as np
from bigfish.stack import check_parameter
from typing import Literal

MAX_LEN_EXCEL = 1048576 #Maximum number of lines that can be written in an excel file

def _cast_spot_to_tuple(spot) :
    return tuple([coord for coord in spot])

def _cast_spots_to_tuple(spots) :
    return tuple(list(map(_cast_spot_to_tuple, spots)))

def write_list_of_results(
        result_list : list,
        path : str,
        filename : str,
        do_excel = True,
        do_feather = False,
        do_csv = False,
        ) :
    

    # FORMAT CHECKING
    if len(result_list) == 0 : return True
    if not do_excel and not do_feather and not do_csv : 
        return False
    elif do_feather and not do_csv and not do_excel: 
        print("WARNING : cell_to_cell colocalisation : .feather is depreciated saving as csv instead.")
        do_csv = True
    elif do_feather :
        print("WARNING : cell_to_cell colocalisation : .feather is depreciated will be removed in future version.")
    
    #PATH CHECKING
    if not path.endswith('/') : path +='/'
    assert os.path.isdir(path)

    #NO OVERWRITE CHECKING
    new_filename = filename

    i= 1
    while new_filename + '.xlsx' in os.listdir(path) or new_filename + '.csv' in os.listdir(path) :
        new_filename = filename + '_{0}'.format(i)
        i+=1

    if do_excel :
        start_row = 0
        sheet_number = 1
        sheet_name = "Sheet" + str(sheet_number)
        with pd.ExcelWriter(path + new_filename + '.xlsx', engine='openpyxl') as writer:
            for df in result_list :

                if start_row + len(df) > 1e6 : #Excel limit of 1 milion line.
                    print("To many excel lines, changing excel sheet.")
                    sheet_number += 1
                    sheet_name = "Sheet" + str(sheet_number)
                    start_row = 0

                #writing
                df.to_excel(writer, sheet_name=sheet_name, startrow=start_row)
                start_row += len(df) + 5
                
    
    if do_csv :
        with open(path + new_filename + '.csv', 'w') as f:
            for df in result_list :
                df.to_csv(f)
                f.write("\n\n")  # Add 2 lines in between DataFrames.
    
    return True


def write_results(
        dataframe: pd.DataFrame, 
        path:str, 
        filename:str, 
        do_excel= True, 
        do_feather= False, 
        do_csv=False, 
        overwrite=False, 
        reset_index=True, 
        batch_mode = False,
        header = True,
        xlsx_start_line = 1,
        ) :
    
    check_parameter(dataframe= pd.DataFrame, path= str, filename = str, do_excel = bool, do_feather = bool)

    if len(dataframe) == 0 : return True
    if not do_excel and not do_feather and not do_csv : 
        return False

    if not path.endswith('/') : path +='/'
    assert os.path.isdir(path)

    #Casting cols name to str for feather format
    index_dim = dataframe.columns.nlevels
    if index_dim == 1 :
        dataframe.columns = dataframe.columns.astype(str)
    else :
        casted_cols = [dataframe.columns.get_level_values(level).astype(str) for level in range(index_dim)]
        casted_cols = zip(*casted_cols)
        dataframe.columns = pd.MultiIndex.from_tuples(casted_cols)

    new_filename = filename
    i= 1
    if not overwrite :
        while new_filename + '.xlsx' in os.listdir(path) or new_filename + '.parquet' in os.listdir(path) or new_filename + '.csv' in os.listdir(path) :
            new_filename = filename + '_{0}'.format(i)
            i+=1

    COLUMNS_TO_DROP = ['image', 'spots', 'clusters', 'rna_coords', 'cluster_coords']
    for col in COLUMNS_TO_DROP :
        if col in dataframe.columns : dataframe = dataframe.drop(columns=col)

    if reset_index : dataframe = dataframe.reset_index(drop=True)

    if do_csv : 
        if batch_mode : mode = 'a'
        else : mode = 'w'
        dataframe.to_csv(path + new_filename + '.csv', sep=";", header=header, mode=mode)
    if do_excel : 
        if len(dataframe) < MAX_LEN_EXCEL :
            dataframe.to_excel(path + new_filename + '.xlsx', header=header, startrow=xlsx_start_line)
        else : 
            print("Error : Table too big to be saved in excel format.")
            return False
    
    if do_feather :
        print("feather saving is depreciated, please use csv instead") 
        pass

    return True

def input_segmentation(
        nucleus_path : str,
        cytoplasm_path : str,
) :
    
    if nucleus_path.endswith('.npy') or nucleus_path.endswith('.npz') :
        nucleus_label = np.load(nucleus_path)
    else :
        raise ValueError("Wrong extension for mask file, only npy and npz are supported")


    if cytoplasm_path != '' :

        if cytoplasm_path.endswith('.npy') or cytoplasm_path.endswith('.npz') :
            cytoplasm_label = np.load(cytoplasm_path)
        else :
            raise ValueError("Wrong extension for mask file, only npy and npz are supported")


    else : 
        cytoplasm_label = nucleus_label

    return nucleus_label, cytoplasm_label

def output_segmentation(
        path : str,
        extension : Literal['npy', 'npz_uncompressed', 'npz_compressed'],
        nucleus_label : np.ndarray,
        cytoplasm_label : np.ndarray = None,
) :
    
    saved = False
    if extension == 'npy' :
        save = np.save
    elif extension == 'npz_uncompressed' :
        save = np.savez
    elif extension == 'npz_compressed' :
        save = np.savez_compressed

    if type(nucleus_label) != type(None) :
        save(path + "_nucleus_segmentation", nucleus_label)
        saved = True
    
    if type(cytoplasm_label) != type(None) :
        save(path + "_cytoplasm_segmentation", cytoplasm_label)

    return saved