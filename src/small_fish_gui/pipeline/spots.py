"""
Sub-module to handle individual spot extraction.

"""

import numpy as np
import pandas as pd
from ..interface.inoutput import write_results

def launch_spots_extraction(
        acquisition_id,
        user_parameters,
        image,
        spots,
        cluster_id,
        nucleus_label,
        cell_label,
) :
    Spots = compute_Spots(
        acquisition_id=acquisition_id,
        image=image,
        spots=spots,
        cluster_id= cluster_id,
        nucleus_label=nucleus_label,
        cell_label=cell_label,
    )

    did_output = write_results(
        Spots,
        path= user_parameters['spots_extraction_folder'],
        filename= user_parameters['spots_filename'],
        do_excel=user_parameters['do_spots_excel'],
        do_csv=user_parameters['do_spots_csv'],
        )
    
    if did_output : print("Individual spots extracted at {0}".format(user_parameters['spots_extraction_folder']))

def compute_Spots(
        acquisition_id : int,
        image : np.ndarray,
        spots : np.ndarray,
        cluster_id : np.ndarray,
        nucleus_label : np.ndarray = None,
        cell_label : np.ndarray = None,
) :

    if len(spots) == 0 :
        return pd.DataFrame()

    if type(cluster_id) == type(None) : #When user doesn't select cluster
        cluster_id = [np.nan]*len(spots)

    index = tuple(spots.T.tolist())
    spot_intensities_list = list(image[index])
    if type(nucleus_label) != type(None) :
        if nucleus_label.ndim == 2 :
            in_nuc_list = list(nucleus_label.astype(bool)[index[-2:]]) #Only plane coordinates
        else :
            in_nuc_list = list(nucleus_label.astype(bool)[index])
    else :
        in_nuc_list = np.nan
    if type(cell_label) != type(None) :

        # Collect all labels that are on fov edge
        on_edge_labels = np.unique(
            np.concatenate([
                cell_label[:,0],
                cell_label[:,-1],
                cell_label[0,:],
                cell_label[-1,:],
            ])
        ).astype(int)

        if cell_label.ndim == 3 :
            cell_label_list = list(cell_label[index])
        else :    
            cell_label_list = list(cell_label[index[-2:]]) #Only plane coordinates
    else :
        cell_label_list = np.nan
    id_list = np.arange(len(spots))

    coord_list = list(zip(*index))

    Spots = pd.DataFrame({
        'acquisition_id' : [acquisition_id] * len(spots),
        'spot_id' : id_list,
        'intensity' : spot_intensities_list,
        'cell_label' : cell_label_list,
        'in_nucleus' : in_nuc_list,
        'coordinates' : coord_list,
        'cluster_id' : cluster_id,
    })
    
    if type(cell_label) != type(None) : #Filter on edge cells
        target_index = Spots.loc[Spots["cell_label"].isin(on_edge_labels)].index
        Spots.loc[target_index,["cell_label"]] = 0

    return Spots
    
def load_spots(
        table_path : str
        ) -> pd.DataFrame :
    
    if table_path.endswith('.csv') :
        Spots = pd.read_csv(table_path, sep= ";")
    elif table_path.endswith('.xlsx') or table_path.endswith('.xls') :
        Spots = pd.read_excel(table_path)
    elif table_path.endswith('.feather') :
        Spots = pd.read_feather(table_path)
    else :
        raise ValueError("Table format not recognized. Please use .csv, .xlsx or .feather files.")
    
    if "coordinates" in Spots.columns :
        pass
    elif "y" in Spots.columns and "x" in Spots.columns :
        if "z" in Spots.columns :
            pass
        else :
            pass
    else :
        raise ValueError("Coordinates information not found in table. Please provide a 'coordinates' column with tuples (z,y,x) or (y,x) or 'y' and 'x' columns.")

    return Spots

def reconstruct_acquisition_data(
        Spots : pd.DataFrame,
        max_id : int,
        filename : str,
        voxel_size : 'tuple[int]'
        ) -> pd.DataFrame:
    """
    Aim : creating a acquisition to add to result_dataframe from loaded spots for co-localization use  
    """
    max_id = int(max_id)
    spots = reconstruct_spots(Spots['coordinates'])
    has_clusters = not Spots['cluster_id'].isna().all()
    spot_number = len(spots)

    if has_clusters :

        clusters = np.empty(shape=(0,5), dtype=int)
        spot_cluster_id = Spots['cluster_id'].to_numpy().astype(int)
        clustered_spots = spots[spot_cluster_id != -1]

        new_acquisition = pd.DataFrame({
            'acquisition_id' : [max_id + 1],
            'name' : ["(loaded_spots)_{}".format(filename.split('.', maxsplit=1)[0])],
            'threshold' : [0],
            'spots' : [spots],
            'clusters' : [clusters.tolist()],
            'clustered_spots_coords' : [clustered_spots.tolist()],
            'spots_cluster_id' : [spot_cluster_id],
            'spot_number' : [spot_number],
            'filename' : [filename],
            'voxel_size' : [voxel_size],
        })
    else :
        new_acquisition = pd.DataFrame({
            'acquisition_id' : [max_id + 1],
            'name' : ["(loaded_spots)_{}".format(filename.split('.', maxsplit=1)[0])],
            'threshold' : [0],
            'spots' : [spots],
            'spot_number' : [spot_number],
            'filename' : [filename],
            'voxel_size' : [voxel_size],
        })


    return new_acquisition

def reconstruct_spots(
        coordinates_serie : pd.Series
        ) :
    spots = coordinates_serie.str.replace('(','').str.replace(')','')
    spots = spots.str.split(',')
    spots = spots.apply(np.array)
    spots = np.array(spots.to_list()).astype(int)

    return spots


def reconstruct_cell_data(
        Spots : pd.DataFrame,
        max_id : int,
        filename : str,
        ) :
    
    has_cluster = not Spots['cluster_id'].isna().all()
    if 'cell_label' in Spots.columns : Spots = Spots.loc[Spots["cell_label"] !=0]
    coordinates = reconstruct_spots(Spots['coordinates'])
    Spots.loc[:,['coordinates']] = pd.Series(coordinates.tolist(), dtype=object, index= Spots.index)

    cell = Spots.groupby('cell_label')['coordinates'].apply(np.array).rename("rna_coords").reset_index(drop=False)
    
    #Handle cells with no spots
    na_mask =cell[cell['rna_coords'].isna()].index
    cell.loc[na_mask, ['rna_coords']] = pd.Series([np.empty(shape=(0,3))]*len(na_mask), dtype= object, index=na_mask)
    cell.loc[~na_mask, "rna_coords"] = cell.loc[~na_mask, "rna_coords"].apply(list).apply(np.array)
    
    cell['total_rna_number'] = cell['rna_coords'].apply(len)
    if has_cluster :
        cell['clustered_spots_coords'] = Spots[Spots['cluster_id'] !=-1].groupby('cell_label')['coordinates'].apply(np.array).rename("clustered_spots_coords")
        
        #Handle cells with no clusters
        na_mask =cell[cell['clustered_spots_coords'].isna()].index
        cell.loc[na_mask, ['clustered_spots_coords']] = pd.Series([np.empty(shape=(0,3))]*len(na_mask), dtype= object, index=na_mask)
        cell["clustered_spots_coords"] = cell["clustered_spots_coords"].apply(list).apply(np.array)
        
        cell['clustered_spot_number'] = cell['clustered_spots_coords'].apply(len)

    cell['acquisition_id'] = max_id + 1
    cell['name'] = "(loaded_spots)_{}".format(filename.split('.', maxsplit=1)[0])
    cell = cell.rename(columns={"cell_label": "cell_id"})

    return cell