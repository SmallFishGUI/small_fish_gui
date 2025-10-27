from ._custom_errors import MissMatchError
from ..gui import coloc_prompt, add_default_loading

import os
import numpy as np
import pandas as pd
import FreeSimpleGUI as sg
from scipy.ndimage import distance_transform_edt
from scipy.signal import fftconvolve

def reconstruct_boolean_signal(image_shape, spot_list: list):
    signal = np.zeros(image_shape, dtype= bool)
    if len(spot_list) == 0 : return signal
    dim = len(spot_list[0])

    if dim == 3 :
        Z, Y, X = list(zip(*spot_list))
        signal[Z,Y,X] = True

    else : 
        Y, X = list(zip(*spot_list))
        signal[Y,X] = True
        

    return signal

def nanometer_to_pixel(value, scale) :
    if isinstance(scale, (float,int)) : scale = [scale]
    if isinstance(value, (float,int)) : value = [value]*len(scale)
    if len(value) != len(scale) : raise ValueError("value and scale must have the same dimensionality")

    return list(np.array(value) / np.array(scale))

def _create_counting_kernel(radius_nm, voxel_size) :

    max_pixel_distance = int(max(nanometer_to_pixel(radius_nm, voxel_size)))
    kernel = np.ones(shape=[2*max_pixel_distance+1 for i in range(len(voxel_size))]) #always odd number so middle is always at [pixel_radius-1, pixel_radius-1, pixel_radius-1]
    if len(voxel_size) == 3 :
        kernel[max_pixel_distance, max_pixel_distance, max_pixel_distance] = 0
    else :
        kernel[max_pixel_distance, max_pixel_distance] = 0

    kernel = distance_transform_edt(kernel, sampling= voxel_size) <= radius_nm
    
    return kernel.astype(int)

def _spot_count_map(spots_array, radius_px, voxel_size) :
    """
    Create a map where each pixel value correspond to the number of spots closer than radius to the position.
    """

    kernel = _create_counting_kernel(radius_px, voxel_size)
    map = fftconvolve(spots_array, kernel, mode= 'same')

    return np.round(map).astype(int)

def _reconstruct_spot_signal(image_shape, spot_list: list, dim=3):
    """
    Create a map where each pixel value correspond to the number of spots located in this position.
    """
    signal = np.zeros(image_shape, dtype= int)
    unique_list, counts = np.unique(spot_list, return_counts= True, axis=0)
    if dim == 3 :
        Z, Y, X = list(zip(*unique_list))
        signal[Z,Y,X] = counts
    elif dim == 2 :
        Y, X = list(zip(*unique_list))
        signal[Y,X] = counts
    else : 
        raise ValueError("Wrong dim passed should be 2 or 3, it is {0}".format(dim))

    return signal

def spots_multicolocalisation(spots_list, anchor_list, radius_nm, image_shape, voxel_size) :

    """
    Compute the number of spots from spots_list closer than radius to a spot from anchor_list. Each spots_list spots will be counted as many times as there are anchors close enough.
    Note that the radius in nm is converted to pixel using voxel size, and rounded to the closest int value.
    
    Example in 2D
    --------

    >>> Anchors         spots           Radius (2px)    Count
    >>> 0 0 0 0 0 0     0 X 0 0 X 0       1             0 1 0 0 0 0
    >>> 0 X 0 0 0 0     X 0 0 X 0 0     1 1 1           1 0 0 0 0 0
    >>> 0 X 0 0 0 0     X X 0 0 0 0       1             1 2 0 0 0 0     --> 5
    >>> 0 0 0 0 X 0     0 0 X 0 0 0                     0 0 0 0 0 0
    >>> 0 0 0 0 0 0     0 0 0 X 0 0                     0 0 0 0 0 0

    Parameters
    ----------
    spots_list : list
    anchor_list : list
    radius_nm : int, float
    image_shape : tuple (Z, Y, X)
    voxel_size : tuple (Z, Y, X)
    
    Returns
    -------
    Returns the list of neighbouring spot number to 'spots_list'.
    """
    if len(spots_list) == 0 or len(anchor_list) == 0 : return 0
    if len(voxel_size) != len(spots_list[0]) : raise ValueError("Dimensions missmatched; voxel_size : {0} spots : {1}".format(len(voxel_size), len(spots_list[0])))

    dim = len(voxel_size)

    anchor_array = _reconstruct_spot_signal(image_shape=image_shape, spot_list=anchor_list, dim=dim)
    count_map = _spot_count_map(anchor_array, radius_px=radius_nm, voxel_size=voxel_size)

    if dim == 3 :
        Z,Y,X = list(zip(*spots_list))
        res = list(count_map[Z,Y,X])

    if dim == 2 :
        Y,X = list(zip(*spots_list))
        res = list(count_map[Y,X])

    return res

def spots_colocalisation(
        spot_list1:np.ndarray, 
        spot_list2:np.ndarray, 
        distance: int, 
        voxel_size : tuple
        )-> int :
    """
    Return number of spots from spot_list1 located closer(large) than distance to at least one spot of spot_list2.

    Parameters
    ----------
        image_shape : tuple
        spot_list1 : list
        spot_list2 : list
        distance : nanometer
            distance in nanometer.
        voxel_size : (z,y,x) tuple
    """

    if len(spot_list1) == 0 or len(spot_list2) == 0 : return np.NaN
    if len(spot_list1[0]) != len(spot_list2[0]) : 
        raise MissMatchError("dimensionalities of spots 1 and spots 2 don't match.")

    spot_list1 = np.array([spot for spot in spot_list1], dtype=int)
    spot_list2 = np.array([spot for spot in spot_list2], dtype=int)

    shape1 = np.max(spot_list1,axis=0)
    shape2 = np.max(spot_list2,axis=0)

    Z,Y,X = list(zip(*spot_list2))

    image_shape = np.max([shape1, shape2],axis=0) + 1

    signal2 = reconstruct_boolean_signal(image_shape, spot_list2)
    mask = np.logical_not(signal2)
    distance_map = distance_transform_edt(mask, sampling= voxel_size)

    if len(voxel_size) == 3 :
        Z,Y,X = zip(*spot_list1)
        count = (distance_map[Z,Y,X] <= distance).sum()
    else :
        Y,X = zip(*spot_list1)
        count = (distance_map[Y,X] <= distance).sum()

    return count


def initiate_colocalisation(
        result_tables : pd.DataFrame,
        ) :

    result_tables = result_tables.set_index('acquisition_id', drop=False)
    available_spots = dict(zip(result_tables['acquisition_id'].astype(str).str.cat(result_tables['name'],sep='-'), result_tables.index))

    while True :
        try : 
            colocalisation_distance, voxel_size, spots1_key, spots2_key = coloc_prompt(list(available_spots.keys()))
            if colocalisation_distance is None :
                return None,None, None,None
            colocalisation_distance = int(colocalisation_distance)

            if spots1_key in available_spots.keys() :
                spots1_key = available_spots[spots1_key]
            elif os.path.isfile(spots1_key) :
                pass

            else :
                raise ValueError("Incorrect value for spots1")
            
            if spots2_key in available_spots.keys() :
                spots2_key = available_spots[spots2_key]
            elif os.path.isfile(spots2_key) :
                pass
            else :
                raise ValueError("Incorrect value for spots1")

        except ValueError as e :
            
            if str(e) == "Incorrect value for spots1" :
                sg.popup(str(e))

            elif str(e) == "Incorrect value for spots2" :
                sg.popup(str(e))

            else :
                sg.popup("Incorrect colocalisation distance")
        else :
            break
    return colocalisation_distance, voxel_size, spots1_key, spots2_key

def _global_coloc(acquisition_id1,acquisition_id2, result_dataframe, colocalisation_distance) :
    """

    Target :

    - acquisition_couple
    - colocalisation_distance
    - spot1_total
    - spot2_total
    - fraction_spot1_coloc_spots
    - fraction_spot2_coloc_spots
    - fraction_spot1_coloc_clusters
    - fraction_spot2_coloc_spots

    """

    acquisition1 = result_dataframe.loc[result_dataframe['acquisition_id'] == acquisition_id1]
    acquisition2 = result_dataframe.loc[result_dataframe['acquisition_id'] == acquisition_id2]

    assert len(acquisition1) == 1
    assert len(acquisition2) == 1

    acquisition_couple = (acquisition_id1,acquisition_id2)

    voxel_size1 = acquisition1.iloc[0].at['voxel_size']
    voxel_size2 = acquisition2.iloc[0].at['voxel_size']

    if voxel_size1 != voxel_size2 : 
        raise MissMatchError("voxel size 1 different than voxel size 2")
    else :
        voxel_size = voxel_size1

    spots1 = acquisition1.iloc[0].at['spots']
    spots2 = acquisition2.iloc[0].at['spots']

    spot1_total = len(spots1)
    spot2_total = len(spots2)

    try :
        fraction_spots1_coloc_spots2 = spots_colocalisation(spot_list1=spots1, spot_list2=spots2, distance= colocalisation_distance, voxel_size=voxel_size) / spot1_total
        fraction_spots2_coloc_spots1 = spots_colocalisation(spot_list1=spots2, spot_list2=spots1, distance= colocalisation_distance, voxel_size=voxel_size) / spot2_total
    except MissMatchError as e :
        sg.popup(str(e))
        fraction_spots1_coloc_spots2 = np.NaN
        fraction_spots2_coloc_spots1 = np.NaN

    if 'clusters' in acquisition1.columns :
        try : 
            clusters_id_1 = np.array(acquisition1.iloc[0].at['spots_cluster_id'], dtype=int)
            fraction_spots2_coloc_cluster1 = spots_colocalisation(spot_list1=spots2, spot_list2=spots1[clusters_id_1 != -1], distance= colocalisation_distance, voxel_size=voxel_size) / spot2_total
        except MissMatchError as e :
            sg.popup(str(e))
            fraction_spots2_coloc_cluster1 = np.NaN
        except TypeError : # clusters not computed
            fraction_spots2_coloc_cluster1 = np.NaN


    else : fraction_spots2_coloc_cluster1 = np.NaN

    if 'clusters' in acquisition2.columns :
        try :
            clusters_id_2 = np.array(acquisition2.iloc[0].at['spots_cluster_id'], dtype=int)
            fraction_spots1_coloc_cluster2 = spots_colocalisation(spot_list1=spots1, spot_list2=spots2[clusters_id_2 != -1], distance= colocalisation_distance, voxel_size=voxel_size) / spot1_total
        except MissMatchError as e :# clusters not computed
            sg.popup(str(e))
            fraction_spots1_coloc_cluster2 = np.NaN
        except TypeError :
            fraction_spots1_coloc_cluster2 = np.NaN


    else : fraction_spots1_coloc_cluster2 = np.NaN

    if 'clusters' in acquisition2.columns and 'clusters' in acquisition1.columns :
        try :
            total_clustered_spots1 = len(spots1[clusters_id_1 != -1])
            total_clustered_spots2 = len(spots2[clusters_id_2 != -1])
            fraction_cluster1_coloc_cluster2 = spots_colocalisation(spot_list1=spots1[clusters_id_1 != -1], spot_list2=spots2[clusters_id_2 != -1], distance= colocalisation_distance, voxel_size=voxel_size) / total_clustered_spots1
            fraction_cluster2_coloc_cluster1 = spots_colocalisation(spot_list1=spots2[clusters_id_2 != -1], spot_list2=spots1[clusters_id_1 != -1], distance= colocalisation_distance, voxel_size=voxel_size) / total_clustered_spots2
        except MissMatchError as e :# clusters not computed
            sg.popup(str(e))
            fraction_cluster1_coloc_cluster2 = np.NaN
            fraction_cluster2_coloc_cluster1 = np.NaN
        except TypeError :
            fraction_cluster1_coloc_cluster2 = np.NaN
            fraction_cluster2_coloc_cluster1 = np.NaN

    else :
        fraction_cluster1_coloc_cluster2 = np.NaN
        fraction_cluster2_coloc_cluster1 = np.NaN
        

    coloc_df = pd.DataFrame({
        "acquisition_couple" : [acquisition_couple],
        "acquisition_id_1" : [acquisition_couple[0]],
        "acquisition_id_2" : [acquisition_couple[1]],
        "colocalisation_distance" : [colocalisation_distance],
        "spot1_total" : [spot1_total],
        "spot2_total" : [spot2_total],
        'fraction_spots1_coloc_spots2' : [fraction_spots1_coloc_spots2],
        'fraction_spots2_coloc_spots1' : [fraction_spots2_coloc_spots1],
        'fraction_spots2_coloc_cluster1' : [fraction_spots2_coloc_cluster1],
        'fraction_spots1_coloc_cluster2' : [fraction_spots1_coloc_cluster2],
        'fraction_cluster1_coloc_cluster2' : [fraction_cluster1_coloc_cluster2],
        'fraction_cluster2_coloc_cluster1' : [fraction_cluster2_coloc_cluster1],
    })

    coloc_df['fraction_spots1_coloc_free2'] = coloc_df['fraction_spots1_coloc_spots2'] - coloc_df['fraction_spots1_coloc_cluster2']
    coloc_df['fraction_spots2_coloc_free1'] = coloc_df['fraction_spots2_coloc_spots1'] - coloc_df['fraction_spots2_coloc_cluster1']

    #Add names
    coloc_df_col = list(coloc_df.columns)
    coloc_df['name1'] = acquisition1.iloc[0].at['name']
    coloc_df['name2'] = acquisition2.iloc[0].at['name']
    coloc_df = coloc_df.loc[:,['name1','name2'] + coloc_df_col]

    return coloc_df

def _cell_coloc(
        acquisition_id1: int,
        acquisition_id2: int,
        result_dataframe : pd.DataFrame, 
        cell_dataframe : pd.DataFrame, 
        colocalisation_distance : float,
        ) :
    
    acquisition1 = result_dataframe.loc[result_dataframe['acquisition_id'] == acquisition_id1]
    acquisition2 = result_dataframe.loc[result_dataframe['acquisition_id'] == acquisition_id2]

    acquisition_name_id1 = acquisition1['name'].iat[0]
    acquisition_name_id2 = acquisition2['name'].iat[0]
    result_dataframe = result_dataframe.set_index('acquisition_id', drop=False)
    coloc_name_forward = '{0} -> {1}'.format(acquisition_name_id1, acquisition_name_id2)
    coloc_name_backward = '{1} -> {0}'.format(acquisition_name_id1, acquisition_name_id2)

    #Getting voxel_size
    if not result_dataframe.at[acquisition_id1, 'voxel_size'] == result_dataframe.at[acquisition_id2, 'voxel_size'] :
        raise ValueError("Selected acquisitions have different voxel_size. Most likely they don't belong to the same fov.")
    voxel_size = result_dataframe.at[acquisition_id1, 'voxel_size']

    #Selecting relevant cells in Cell table
    cell_dataframe = cell_dataframe.loc[(cell_dataframe['acquisition_id'] == acquisition_id1)|(cell_dataframe['acquisition_id'] == acquisition_id2)]

    #Putting spots lists in 2 cols for corresponding cells
    pivot_values_columns = ['rna_coords', 'total_rna_number']
    if 'clusters' in acquisition2.columns or 'clusters' in acquisition1.columns:
        pivot_values_columns.extend(['clustered_spots_coords','clustered_spot_number'])
    cell_dataframe.loc[:,['cell_id']] = cell_dataframe['cell_id'].astype(int)
    colocalisation_df = cell_dataframe.pivot(
        columns=['name', 'acquisition_id'],
        values= pivot_values_columns,
        index= 'cell_id'
    )
    #spots _vs spots
    colocalisation_df[("spots_with_spots_count",coloc_name_forward,"forward")] = colocalisation_df['rna_coords'].apply(
        lambda x: spots_colocalisation(
            spot_list1= x[(acquisition_name_id1,acquisition_id1)],
            spot_list2= x[(acquisition_name_id2,acquisition_id2)],
            distance=colocalisation_distance,
            voxel_size=voxel_size
            ),axis=1
        )
    colocalisation_df[("spots_with_spots_fraction",coloc_name_forward,"forward")] = colocalisation_df[("spots_with_spots_count",coloc_name_forward,"forward")].astype(float) / colocalisation_df[('total_rna_number',acquisition_name_id1,acquisition_id1)].astype(float)
    
    colocalisation_df[("spots_with_spots_count",coloc_name_backward,"backward")] = colocalisation_df['rna_coords'].apply(
        lambda x: spots_colocalisation(
            spot_list1= x[(acquisition_name_id2,acquisition_id2)],
            spot_list2= x[(acquisition_name_id1,acquisition_id1)],
            distance=colocalisation_distance,
            voxel_size=voxel_size
            ),axis=1
        )
    colocalisation_df[("spots_with_spots_fraction",coloc_name_backward,"backward")] = colocalisation_df[("spots_with_spots_count",coloc_name_backward,"backward")].astype(float) / colocalisation_df[('total_rna_number',acquisition_name_id2,acquisition_id2)].astype(float)

    if 'clusters' in acquisition2.columns:
        if len(acquisition2['clusters'].iat[0]) > 0 :

            #spots to clusters
            colocalisation_df[("spots_with_clustered_spots_count",coloc_name_forward,"forward")] = colocalisation_df.apply(
                lambda x: spots_colocalisation(
                    spot_list1= x[('rna_coords',acquisition_name_id1,acquisition_id1)],
                    spot_list2= x[('clustered_spots_coords',acquisition_name_id2,acquisition_id2)][:,:len(voxel_size)],
                    distance=colocalisation_distance,
                    voxel_size=voxel_size
                    ),axis=1
                )
            colocalisation_df[("spots_with_clustered_spots_fraction",coloc_name_forward,"forward")] = colocalisation_df[("spots_with_clustered_spots_count",coloc_name_forward,"forward")].astype(float) / colocalisation_df[('total_rna_number',acquisition_name_id1,acquisition_id1)].astype(float)
        
    if 'clusters' in acquisition1.columns:
        if len(acquisition1['clusters'].iat[0]) > 0 :
            colocalisation_df[("spots_with_clustered_spots_count",coloc_name_backward,"backward")] = colocalisation_df.apply(
                lambda x: spots_colocalisation(
                    spot_list1= x[('rna_coords',acquisition_name_id2,acquisition_id2)],
                    spot_list2= x[('clustered_spots_coords',acquisition_name_id1,acquisition_id1)][:,:len(voxel_size)],
                    distance=colocalisation_distance,
                    voxel_size=voxel_size
                    ),axis=1
                )

            colocalisation_df[("spots_with_clustered_spots_fraction",coloc_name_backward,"backward")] = colocalisation_df[("spots_with_clustered_spots_count",coloc_name_backward,"backward")].astype(float) / colocalisation_df[('total_rna_number',acquisition_name_id2,acquisition_id2)].astype(float)

    if 'clusters' in acquisition2.columns and 'clusters' in acquisition1.columns:
        if len(acquisition1['clusters'].iat[0]) > 0 and len(acquisition2['clusters'].iat[0]) > 0 :
            #clusters to clusters 
            colocalisation_df[("clustered_spots_with_clustered_spots_count",coloc_name_forward,"forward")] = colocalisation_df.apply(
                lambda x: spots_colocalisation(
                    spot_list1= x[('clustered_spots_coords',acquisition_name_id1,acquisition_id1)][:,:len(voxel_size)],
                    spot_list2= x[('clustered_spots_coords',acquisition_name_id2,acquisition_id2)][:,:len(voxel_size)],
                    distance=colocalisation_distance, 
                    voxel_size=voxel_size 
                    ),axis=1
            )
            colocalisation_df[("clustered_spots_with_clustered_spots_fraction",coloc_name_forward,"forward")] = colocalisation_df[("clustered_spots_with_clustered_spots_count",coloc_name_forward,"forward")].astype(float) / colocalisation_df[('clustered_spot_number',acquisition_name_id1,acquisition_id1)].astype(float)

            colocalisation_df[("clustered_spots_with_clustered_spots_count",coloc_name_backward,"backward")] = colocalisation_df.apply(
                lambda x: spots_colocalisation(
                    spot_list1= x[('clustered_spots_coords',acquisition_name_id2,acquisition_id2)][:,:len(voxel_size)],
                    spot_list2= x[('clustered_spots_coords',acquisition_name_id1,acquisition_id1)][:,:len(voxel_size)],
                    distance=colocalisation_distance, 
                    voxel_size=voxel_size 
                    ),axis=1
            )
            colocalisation_df[("clustered_spots_with_clustered_spots_fraction",coloc_name_backward,"backward")] = colocalisation_df[("clustered_spots_with_clustered_spots_count",coloc_name_backward,"backward")].astype(float) / colocalisation_df[('clustered_spot_number',acquisition_name_id2,acquisition_id2)].astype(float)

    colocalisation_df = colocalisation_df.sort_index(axis=0).sort_index(axis=1, level=0)

    if 'clustered_spots_coords' in cell_dataframe.columns : colocalisation_df = colocalisation_df.drop('clustered_spots_coords', axis=1)
    colocalisation_df = colocalisation_df.drop('rna_coords', axis=1)
    colocalisation_df['voxel_size'] = [voxel_size]*len(colocalisation_df)
    colocalisation_df['pair_name'] = [(acquisition_name_id1, acquisition_name_id2)] * len(colocalisation_df)
    colocalisation_df['acquisition_id_1'] = [acquisition_id1] * len(colocalisation_df)
    colocalisation_df['acquisition_id_2'] = [acquisition_id2] * len(colocalisation_df)
    colocalisation_df['colocalisation_distance'] = colocalisation_distance

    return colocalisation_df

@add_default_loading
def launch_colocalisation(acquisition_id1, acquisition_id2, result_dataframe, cell_result_dataframe, colocalisation_distance, global_coloc_df, cell_coloc_df: dict) :
    

    if acquisition_id1 in list(cell_result_dataframe['acquisition_id']) and acquisition_id2 in list(cell_result_dataframe['acquisition_id']) :
        print("Launching cell to cell colocalisation.")
        new_coloc = _cell_coloc(
            acquisition_id1 = acquisition_id1,
            acquisition_id2 = acquisition_id2,
            result_dataframe = result_dataframe,
            cell_dataframe=cell_result_dataframe,
            colocalisation_distance=colocalisation_distance
        )
        
        index = 0
        while (acquisition_id1, acquisition_id2, index) in cell_coloc_df.keys() :
            index +=1
        cell_coloc_df [(acquisition_id1,acquisition_id2, index)] = new_coloc


    else :
        print("Launching global colocalisation.")
        new_coloc = _global_coloc(
            acquisition_id1=acquisition_id1,
            acquisition_id2=acquisition_id2,
            result_dataframe=result_dataframe,
            colocalisation_distance=colocalisation_distance,
        )
        global_coloc_df = pd.concat([
            global_coloc_df,
            new_coloc,
        ], axis=0).reset_index(drop=True)


    return global_coloc_df, cell_coloc_df