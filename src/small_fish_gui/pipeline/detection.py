"""
Contains code to handle detection as well as bigfish wrappers related to spot detection.
"""
from ..hints import pipeline_parameters

from ._preprocess import ParameterInputError
from ._preprocess import check_integrity, convert_parameters_types

from ..gui.napari_visualiser import correct_spots, threshold_selection
from ..gui import add_default_loading
from ..gui import detection_parameters_promt

from ..interface import get_voxel_size
from ..utils import compute_anisotropy_coef
from ._signaltonoise import compute_snr_spots

from magicgui import magicgui

from types import GeneratorType
from napari.types import LayerDataTuple

import numpy as np
from numpy import NaN
import pandas as pd
import FreeSimpleGUI as sg
import bigfish.detection as detection
import bigfish.stack as stack
import bigfish.multistack as multistack
import bigfish.classification as classification
from bigfish.detection.spot_detection import get_object_radius_pixel
from skimage.measure import regionprops
from scipy.ndimage import binary_dilation



def compute_auto_threshold(images, voxel_size=None, spot_radius=None, log_kernel_size=None, minimum_distance=None, im_number= 15, crop_zstack= None) :
    """
    Compute bigfish auto threshold efficiently for list of images. In case on large set of images user can set im_number to only consider a random subset of image for threshold computation.
    """
    # check parameters
    stack.check_parameter(images = (list, np.ndarray, GeneratorType,), voxel_size=(int, float, tuple, list, type(None)),spot_radius=(int, float, tuple, list, type(None)),log_kernel_size=(int, float, tuple, list, type(None)),minimum_distance=(int, float, tuple, list, type(None)), im_number = int, crop_zstack= (type(None), tuple))

    # if one image is provided we enlist it
    if not isinstance(images, list):
        if isinstance(images, np.ndarray) : 
            stack.check_array(images,ndim=[2, 3],dtype=[np.uint8, np.uint16, np.float32, np.float64])
            ndim = images.ndim
            images = [images]
        else : 
            images = [image for image in images]
            for image in images : stack.check_array(image,ndim=[2, 3],dtype=[np.uint8, np.uint16, np.float32, np.float64])
            ndim = images[0].ndim

    else:
        ndim = None
        for i, image in enumerate(images):
            stack.check_array(image,ndim=[2, 3],dtype=[np.uint8, np.uint16, np.float32, np.float64])
            if i == 0:
                ndim = image.ndim
            else:
                if ndim != image.ndim:
                    raise ValueError("Provided images should have the same "
                                     "number of dimensions.")
    if len(images) > im_number : #if true we select a random sample of images
        idx = np.arange(len(images),dtype= int)
        np.random.shuffle(idx)
        images = [images[i] for i in idx]
        
    #Building a giant 3D array containing all information for threshold selection -> cheating detection.automated_threshold_setting that doesn't take lists and doesn't use spatial information.
    if type(crop_zstack) == type(None) :
        crop_zstack = (0, len(images[0]))
    
    log_kernel_size, minimum_distance = _compute_threshold_parameters(ndim, voxel_size, spot_radius, minimum_distance, log_kernel_size)
    images_filtered = np.concatenate(
        [stack.log_filter(image[crop_zstack[0]: crop_zstack[1]], sigma= log_kernel_size) for image in images],
         axis= ndim -1)
    max_masks = np.concatenate(
        [detection.local_maximum_detection(image[crop_zstack[0]: crop_zstack[1]], min_distance= minimum_distance) for image in images],
         axis= ndim -1)
    threshold = detection.automated_threshold_setting(images_filtered, max_masks)

    return threshold

def _compute_threshold_parameters(ndim, voxel_size, spot_radius, minimum_distance, log_kernel_size) :

    # check consistency between parameters - detection with voxel size and
    # spot radius
    if (voxel_size is not None and spot_radius is not None
            and log_kernel_size is None and minimum_distance is None):
        if isinstance(voxel_size, (tuple, list)):
            if len(voxel_size) != ndim:
                raise ValueError("'voxel_size' must be a scalar or a sequence "
                                 "with {0} elements.".format(ndim))
        else:
            voxel_size = (voxel_size,) * ndim
        if isinstance(spot_radius, (tuple, list)):
            if len(spot_radius) != ndim:
                raise ValueError("'spot_radius' must be a scalar or a "
                                 "sequence with {0} elements.".format(ndim))
        else:
            spot_radius = (spot_radius,) * ndim
        log_kernel_size = get_object_radius_pixel(
            voxel_size_nm=voxel_size,
            object_radius_nm=spot_radius,
            ndim=ndim)
        minimum_distance = get_object_radius_pixel(
            voxel_size_nm=voxel_size,
            object_radius_nm=spot_radius,
            ndim=ndim)

    # check consistency between parameters - detection with kernel size and
    # minimal distance
    elif (voxel_size is None and spot_radius is None
          and log_kernel_size is not None and minimum_distance is not None):
        if isinstance(log_kernel_size, (tuple, list)):
            if len(log_kernel_size) != ndim:
                raise ValueError("'log_kernel_size' must be a scalar or a "
                                 "sequence with {0} elements.".format(ndim))
        else:
            log_kernel_size = (log_kernel_size,) * ndim
        if isinstance(minimum_distance, (tuple, list)):
            if len(minimum_distance) != ndim:
                raise ValueError("'minimum_distance' must be a scalar or a "
                                 "sequence with {0} elements.".format(ndim))
        else:
            minimum_distance = (minimum_distance,) * ndim

    # check consistency between parameters - detection in priority with kernel
    # size and minimal distance
    elif (voxel_size is not None and spot_radius is not None
          and log_kernel_size is not None and minimum_distance is not None):
        if isinstance(log_kernel_size, (tuple, list)):
            if len(log_kernel_size) != ndim:
                raise ValueError("'log_kernel_size' must be a scalar or a "
                                 "sequence with {0} elements.".format(ndim))
        else:
            log_kernel_size = (log_kernel_size,) * ndim
        if isinstance(minimum_distance, (tuple, list)):
            if len(minimum_distance) != ndim:
                raise ValueError("'minimum_distance' must be a scalar or a "
                                 "sequence with {0} elements.".format(ndim))
        else:
            minimum_distance = (minimum_distance,) * ndim

    # missing parameters
    else:
        raise ValueError("One of the two pairs of parameters ('voxel_size', "
                         "'spot_radius') or ('log_kernel_size', "
                         "'minimum_distance') should be provided.")
    
    return log_kernel_size, minimum_distance

def cluster_detection(spots, voxel_size, radius = 350, nb_min_spots = 4, keys_to_compute = ["clustered_spots", "clusters"]) :
    """
    Performs `bigfish.detection.cluster_detection()` to detect clusters.
    Then offers possibility to get results sorted in pandas dataframe.

    Parameters
    ----------
        spots : np.ndarray
            Coordinates of the detected spots with shape (nb_spots, 3) or (nb_spots, 2).
        voxel_size : int, float, Tuple(int, float) or List(int, float)
            Size of a voxel, in nanometer. One value per spatial dimension (zyx or yx dimensions). If it's a scalar, the same value is applied to every dimensions.
        radius : int
            The maximum distance between two samples for one to be considered as in the neighborhood of the other. Radius expressed in nanometer.
        nb_min_spots : int
            The number of spots in a neighborhood for a point to be considered as a core point (from which a cluster is expanded). This includes the point itself.
        keys_to_compute : list[str], str
            keys from (clustered_spots, clusters, clustered_spots_dataframe, clusters_dataframe)
                --> clustered_spots : np.ndarray
                    Coordinates of the detected spots with shape (nb_spots, 4) or (nb_spots, 3). One coordinate per dimension (zyx or yx coordinates) plus the index of the cluster assigned to the spot. If no cluster was assigned, value is -1.
                --> clusters : np.ndarray
                    Array with shape (nb_clusters, 5) or (nb_clusters, 4). One coordinate per dimension for the clusters centroid (zyx or yx coordinates), the number of spots detected in the clusters and its index.
                --> clustered_spots_dataframe
                --> clusters_dataframe
    
    Returns
    -------
        res : dict
            keys : keys from `keys_to_compute` argument : (clustered_spots, clusters, clustered_spots_dataframe, clusters_dataframe)    
    """

    if isinstance(keys_to_compute, str) : keys_to_compute = [keys_to_compute]
    elif isinstance(keys_to_compute, list) : pass
    else : raise TypeError("Wrong type for keys_to_compute. Should be list[str] or str. It is {0}".format(type(keys_to_compute)))
    if len(spots) == 0 :
        res = {'clustered_spots' : np.empty(shape=(0,len(voxel_size) + 1), dtype=int), 'clusters' : np.empty(shape=(0,len(voxel_size) + 2), dtype=int), 'clustered_spots_dataframe' : pd.DataFrame(columns= ["id", "cluster_id", "z", "y", "x"]), 'clusters_dataframe' : pd.DataFrame(columns= ["id", "z", "y", "x", "spot_number"])}
        return {key : res[key] for key in keys_to_compute}
    else : res = {}
    voxel_size = tuple([int(d) for d in voxel_size])
    clustered_spots, clusters = detection.detect_clusters(spots, voxel_size= voxel_size, radius= radius, nb_min_spots= nb_min_spots)


    if 'clustered_spots' in keys_to_compute :
        res['clustered_spots'] = clustered_spots
        voxel_size
    if 'clusters' in keys_to_compute : 
        res['clusters'] = clusters

    if 'clustered_spots_dataframe' in keys_to_compute :
        res['clustered_spots_dataframe'] = _compute_clustered_spots_dataframe(clustered_spots)
    
    if 'clusters_dataframe' in keys_to_compute :
        res['clusters_dataframe'] = _compute_cluster_dataframe(clusters)

    return res

def initiate_detection(user_parameters : pipeline_parameters, map_, shape) :
    is_3D_stack= user_parameters['is_3D_stack']
    is_multichannel = user_parameters['is_multichannel']
    do_dense_region_deconvolution = user_parameters['do_dense_regions_deconvolution']
    do_clustering = user_parameters['do_cluster_computation']
    detection_parameters = user_parameters.copy()
    
    #Attempt to read voxel size from metadata
    voxel_size = get_voxel_size(user_parameters['image_path'])
    if voxel_size is None or not user_parameters.get('voxel_size') is None:
        pass
    else :
        detection_parameters['voxel_size'] = [round(v) if isinstance(v, (float,int)) else None for v in voxel_size]
        detection_parameters['voxel_size_z'] = detection_parameters['voxel_size'][0] if isinstance(detection_parameters['voxel_size'][0], (float,int)) else None
        detection_parameters['voxel_size_y'] = detection_parameters['voxel_size'][1] if isinstance(detection_parameters['voxel_size'][1], (float,int)) else None
        detection_parameters['voxel_size_x'] = detection_parameters['voxel_size'][2] if isinstance(detection_parameters['voxel_size'][2], (float,int)) else None

    #Setting default spot size to 1.5 voxel
    if detection_parameters.get('spot_size') is None :
        detection_parameters['spot_size_z'] = round(detection_parameters['voxel_size_z']*1.5) if isinstance(detection_parameters['voxel_size_z'], (float,int)) else None
        detection_parameters['spot_size_y'] = round(detection_parameters['voxel_size_y']*1.5) if isinstance(detection_parameters['voxel_size_y'],(float,int)) else None
        detection_parameters['spot_size_x'] = round(detection_parameters['voxel_size_x']*1.5) if isinstance(detection_parameters['voxel_size_x'],(float,int)) else None

    while True :
        detection_parameters = detection_parameters_promt(
            is_3D_stack=is_3D_stack,
            is_multichannel=is_multichannel,
            do_dense_region_deconvolution=do_dense_region_deconvolution,
            do_clustering=do_clustering,
            segmentation_done= user_parameters['segmentation_done'],
            default_dict=detection_parameters
            )
        if type(detection_parameters) == type(None) : return None
        try :
            detection_parameters = convert_parameters_types(detection_parameters)
            detection_parameters = check_integrity(
                detection_parameters, 
                do_dense_region_deconvolution,
                do_clustering, 
                is_multichannel, 
                user_parameters['segmentation_done'], 
                map_, 
                shape
                )
        except ParameterInputError as error:
            sg.popup(error)
        else :
            user_parameters.update(detection_parameters)
            break
    
    return user_parameters

@add_default_loading
def _launch_detection(image, image_input_values: dict) :

    """
    Performs spots detection
    """
    
    #Extract parameters
    voxel_size = image_input_values['voxel_size']
    threshold = image_input_values.get('threshold')
    threshold_penalty = image_input_values.setdefault('threshold penalty', 1)
    spot_size = image_input_values.get('spot_size')
    log_kernel_size = image_input_values.get('log_kernel_size')
    minimum_distance = image_input_values.get('minimum_distance')
    threshold_user_selection = image_input_values['show_interactive_threshold_selector']
    
    if type(threshold) == type(None) :     
        threshold = threshold_penalty * compute_auto_threshold(image, voxel_size=voxel_size, spot_radius=spot_size, log_kernel_size=log_kernel_size, minimum_distance=minimum_distance)
        threshold = max(threshold,15) # Force threshold to be at least 15 to match napari widget and to not have too many spots for weak configs

    filtered_image = _apply_log_filter(
        image=image,
        voxel_size=voxel_size,
        spot_radius=spot_size,
        log_kernel_size = log_kernel_size,
    )

    local_maxima = _local_maxima_mask(
        image_filtered=filtered_image,
        voxel_size=voxel_size,
        spot_radius=spot_size,
        minimum_distance=minimum_distance
    )

    if threshold_user_selection :

        threshold_slider = _create_threshold_slider(
            logfiltered_image=filtered_image,
            local_maxima=local_maxima,
            default=threshold,
            min_value=filtered_image[local_maxima].min(),
            max_value=filtered_image[local_maxima].max(),
            voxel_size=voxel_size
        )

        spots, threshold = threshold_selection(
            image=image,
            filtered_image=filtered_image,
            threshold_slider=threshold_slider,
            voxel_size=voxel_size
        )
    else :
        spots = detection.spots_thresholding(
            image=filtered_image,
            mask_local_max=local_maxima,
            threshold=threshold
        )[0]
        
    return spots, threshold

@add_default_loading
def launch_dense_region_deconvolution(image, spots, image_input_values: dict,) :
    """
    Performs spot decomposition

    Returns
    -------
        spots : np.ndarray
            Array(nb_spot, dim) (dim is either 3 or 2)
        fov_res : dict
            keys : spot_number, spotsSignal_median, spotsSignal_mean, spotsSignal_std, median_pixel, mean_pixel, snr_median, snr_mean, snr_std, cell_medianbackground_mean, cell_medianbackground_std, cell_meanbackground_mean, cell_meanbackground_std, cell_stdbackground_mean, cell_stdbackground_std
    """
    
    ##Initiate lists
    voxel_size = image_input_values['voxel_size']
    spot_size = image_input_values.get('spot_size')
    ##deconvolution parameters
    alpha = image_input_values.get('alpha')
    beta = image_input_values.get('beta')
    gamma = image_input_values.get('gamma')
    deconvolution_kernel = image_input_values.get('deconvolution_kernel')
    dim = image_input_values['dim']
        
    spots, dense_regions, ref_spot = detection.decompose_dense(image=image, spots=spots, voxel_size=voxel_size, spot_radius=spot_size, kernel_size=deconvolution_kernel, alpha=alpha, beta=beta, gamma=gamma)
    del dense_regions, ref_spot

    return spots

@add_default_loading
def launch_post_detection(image, spots, image_input_values: dict,) :
    fov_res = {}
    dim = image_input_values['dim']
    voxel_size = image_input_values['voxel_size']
    spot_size = image_input_values.get('spot_size')

    #features
    fov_res['spot_number'] = len(spots)
    snr_res = compute_snr_spots(image, spots, voxel_size, spot_size)
    if len(spots) == 0 :
        fov_res['spotsSignal_median'], fov_res['spotsSignal_mean'], fov_res['spotsSignal_std'] = np.NaN, np.NaN, np.NaN
    else :
        if dim == 3 :
            Z,Y,X = list(zip(*spots))
            spots_values = image[Z,Y,X]
        else :
            Y,X = list(zip(*spots))
            spots_values = image[Y,X]
        fov_res['spotsSignal_median'], fov_res['spotsSignal_mean'], fov_res['spotsSignal_std'] = np.median(spots_values), np.mean(spots_values), np.std(spots_values)
    
    fov_res['median_pixel'] = np.median(image)
    fov_res['mean_pixel'] = np.mean(image)

    #appending results
    fov_res.update(snr_res)

    return fov_res

def _compute_cell_snr(image: np.ndarray, bbox, spots, voxel_size, spot_size) :
    
    min_y, min_x, max_y, max_x = bbox
    image= image[min_y: max_y, min_x: max_x]

    if len(spots) == 0 :
        res = {
           'snr_mean' : np.NaN,
           'snr_median' : np.NaN,
           'snr_std' : np.NaN,
        }

        return res

    if len(spots[0]) == 3 :
        Z,Y,X = zip(*spots)
        spots = np.array(
            list(zip(Y,X)),
            dtype= int
        )
        voxel_size = voxel_size[1:]
        spot_size = spot_size[1:]

    snr_dict = compute_snr_spots(
            image= image,
            spots= spots,
            spot_radius= spot_size,
            voxel_size=voxel_size
            )
        
    return snr_dict

@add_default_loading
def launch_cell_extraction(
    acquisition_id, 
    spots, 
    clusters, 
    spots_cluster_id,
    image, 
    nucleus_signal, 
    cell_label, 
    nucleus_label, 
    user_parameters : pipeline_parameters
    ) :

    #Extract parameters
    dim = user_parameters['dim']
    do_clustering = user_parameters['do_cluster_computation']
    voxel_size = user_parameters['voxel_size']

    if do_clustering :
        if len(clusters) > 0 :
            free_spots = spots[spots_cluster_id == -1].astype(int)
            clustered_spots = spots[spots_cluster_id != -1].astype(int)

            other_coords = {
                'clusters_coords' : clusters,
                'clustered_spots' : clustered_spots,
                'free_spots' : free_spots,
                }
        else :
            other_coords = None
    else : other_coords = None
    if do_clustering : do_clustering = len(clusters) > 0

    if image.ndim == 3 :
        image = stack.maximum_projection(image)
    if nucleus_signal.ndim == 3 :
        nucleus_signal = stack.maximum_projection(nucleus_signal)
    if cell_label.ndim == 3 :
        cell_label = stack.maximum_projection(nucleus_signal)
    if nucleus_label.ndim == 3 :
        nucleus_label = stack.maximum_projection(nucleus_signal)

    cells_results = multistack.extract_cell(
        cell_label=cell_label,
        ndim=dim,
        nuc_label=nucleus_label,
        rna_coord=spots,
        others_coord=other_coords,
        image=image
    )

    #BigFish features
    features_names = ['acquisition_id', 'cell_id', 'cell_bbox'] + classification.get_features_name(
        names_features_centrosome=False,
        names_features_area=True,
        names_features_dispersion=True,
        names_features_distance=True,
        names_features_foci=do_clustering,
        names_features_intranuclear=True,
        names_features_protrusion=False,
        names_features_topography=True
    )

    #Nucleus features : area is computed in bigfish
    features_names += ['nucleus_mean_signal', 'nucleus_median_signal', 'nucleus_max_signal', 'nucleus_min_signal']
    features_names += ['snr_mean', 'snr_median', 'snr_std']
    features_names += ['cell_center_coord','foci_number','foci_in_nuc_number']
    features_names += ['rna_coords','cluster_coords', 'clustered_spots_coords', 'free_spots_coords']
    features_names += ['clustered_spot_number', 'free_spot_number']

    result_frame = pd.DataFrame()

    for cell in cells_results :

        #Extract cell results
        cell_id = cell['cell_id']
        cell_mask = cell['cell_mask']
        nuc_mask = cell ['nuc_mask']
        cell_bbox = cell['bbox'] # (min_y, min_x, max_y, max_x)
        min_y, min_x, max_y, max_x = cell['bbox'] # (min_y, min_x, max_y, max_x)
        nuc_signal = nucleus_signal[min_y:max_y, min_x:max_x]
        rna_coords = cell['rna_coord']
        foci_coords = cell.get('clusters_coords')
        clustered_spots_coords = cell.get('clustered_spots')
        free_spots_coords = cell.get('free_spots')
        signal = cell['image']

        if do_clustering :
            if len(clusters) > 0 :
                compute_foci = True
            else : 
                compute_foci = False
        else : compute_foci = False

        with np.errstate(divide= 'ignore', invalid= 'ignore') :
            features = classification.compute_features(
                cell_mask=cell_mask,
                nuc_mask=nuc_mask,
                ndim=dim,
                rna_coord= rna_coords,
                foci_coord=foci_coords,
                voxel_size_yx= float(voxel_size[-1]),
                smfish=signal,
                centrosome_coord=None,
                compute_centrosome=False,
                compute_area=True,
                compute_dispersion=True,
                compute_distance=True,
                compute_foci= compute_foci,
                compute_intranuclear=True,
                compute_protrusion=False,
                compute_topography=True
            )

        #center of cell coordinates
        local_cell_center = regionprops(
            label_image=cell_mask.astype(int)
        )[0]['centroid']
        cell_center = (local_cell_center[0] + min_y, local_cell_center[1] + min_x)

        #foci in nucleus
        if type(foci_coords) != type(None) :
            if len(foci_coords) == 0 : 
                foci_number = 0
                foci_in_nuc_number = 0
            else :
                foci_number = len(foci_coords)
                foci_index = list(zip(*foci_coords))
                if len(foci_index) == 5 :
                    foci_index = foci_index[1:3]
                elif len(foci_index) == 4 :
                    foci_index = foci_index[:2]
                else : raise AssertionError("Impossible number of dim for foci : ", len(foci_index))
                foci_in_nuc_number = nuc_mask[tuple(foci_index)].astype(bool).sum()
        else :
            foci_number = np.NaN
            foci_in_nuc_number = np.NaN

        #Signal to noise
        snr_dict = _compute_cell_snr(
            image,
            cell_bbox,
            spots=rna_coords,
            voxel_size=voxel_size,
            spot_size=user_parameters['spot_size']
        )

        snr_mean = snr_dict['snr_mean']
        snr_median = snr_dict['snr_median']
        snr_std = snr_dict['snr_std']

        features = list(features)
        features += [np.mean(nuc_signal), np.median(nuc_signal), np.max(nuc_signal), np.min(nuc_signal)]
        features += [snr_mean, snr_median, snr_std]
        features += [cell_center, foci_number, foci_in_nuc_number]

        features = [acquisition_id, cell_id, cell_bbox] + features
        features += [rna_coords, foci_coords, clustered_spots_coords, free_spots_coords]
        features += [len(clustered_spots_coords) if type(clustered_spots_coords) != type(None) else None]
        features += [len(free_spots_coords) if type(free_spots_coords) != type(None) else None]
        
        result_frame = pd.concat([
            result_frame,
            pd.DataFrame(columns = features_names, data= (features,)),
        ],
        axis= 0
        )
    
    return result_frame

@add_default_loading
def launch_clustering(spots, user_parameters : pipeline_parameters): 

    voxel_size = user_parameters['voxel_size']
    nb_min_spots = user_parameters['min_number_of_spots']
    cluster_size = user_parameters['cluster_size']

    cluster_result_dict = cluster_detection(
        spots=spots,
        voxel_size=voxel_size,
        radius=cluster_size,
        nb_min_spots=nb_min_spots,
        keys_to_compute= ['clusters','clustered_spots']
    )

    clusters = cluster_result_dict['clusters']
    clustered_spots = cluster_result_dict['clustered_spots']

    return clusters, clustered_spots

def launch_detection(
        image,
        other_image,
        user_parameters : pipeline_parameters,
        cell_label= None,
        nucleus_label = None,
        hide_loading=False,
        ) :
    """
    Main call for features computation :
    --> spot dection
    --> dense regions deconv
    --> cluster (opt)
    --> spot correction
    --> general features computations
    --> cell extractions
    --> cell features computations
    
    RETURNS
    -------
        user_parameters : dict
        spots : np.ndarray
        clusters : np.ndarray

    USER_PARAMETERS UPDATE
    ----------------------
        'threshold'
    """
    fov_result = {}
    do_dense_region_deconvolution = user_parameters['do_dense_regions_deconvolution']
    do_clustering = user_parameters['do_cluster_computation']

    spots, threshold  = _launch_detection(image, user_parameters, hide_loading = hide_loading)
        
    if do_dense_region_deconvolution : 
        spots = launch_dense_region_deconvolution(image, spots, user_parameters, hide_loading = hide_loading)
        
    if do_clustering : 
        clusters, clustered_spots = launch_clustering(spots, user_parameters, hide_loading = hide_loading) #012 are coordinates #3 is number of spots per cluster, #4 is cluster index
        spots, spots_cluster_id = clustered_spots[:,:-1], clustered_spots[:,-1]

    else : 
        clusters = None
        spots_cluster_id = None

    user_parameters['threshold'] = threshold

    if user_parameters['show_napari_corrector'] :

        spots, clusters, new_cluster_radius, new_min_spot_number = correct_spots(
            image, 
            spots, 
            user_parameters['voxel_size'],
            clusters=clusters,
            spot_cluster_id = spots_cluster_id,
            cluster_size= user_parameters.get('cluster_size'),
            min_spot_number= user_parameters.setdefault('min_number_of_spots', 0),
            cell_label=cell_label,
            nucleus_label=nucleus_label,
            other_images=other_image
            )
        
        if type(new_cluster_radius) != type(None) :
            user_parameters['cluster_size'] = new_cluster_radius
        if type(new_min_spot_number) != type(None) :
            user_parameters['min_number_of_spots'] = new_min_spot_number
        
        if do_clustering :
            spots, spots_cluster_id = spots[:,:-1], spots[:,-1]
        else :
            spots_cluster_id = None

    post_detection_dict = launch_post_detection(image, spots, user_parameters, hide_loading = hide_loading)
    fov_result.update(post_detection_dict)
    
    return user_parameters, fov_result, spots, clusters, spots_cluster_id
            

def launch_features_computation(
        acquisition_id, 
        image, 
        nucleus_signal, 
        spots, 
        clusters, 
        spots_cluster_id, 
        nucleus_label, 
        cell_label, 
        user_parameters : pipeline_parameters, 
        frame_results
        ) :

    dim = image.ndim
    if user_parameters['do_cluster_computation'] : 
        frame_results['cluster_number'] = len(clusters)
        if dim == 3 :
            frame_results['total_spots_in_clusters'] = clusters.sum(axis=0)[3] if len(clusters) >0 else  0
        else :
            frame_results['total_spots_in_clusters'] = clusters.sum(axis=0)[2] if len(clusters) >0 else  0
    
    if type(cell_label) != type(None) and type(nucleus_label) != type(None): 
        
        try :
            cell_result_dframe = launch_cell_extraction(
                acquisition_id=acquisition_id,
                spots=spots,
                clusters=clusters,
                spots_cluster_id = spots_cluster_id,
                image=image,
                nucleus_signal=nucleus_signal,
                cell_label= cell_label,
                nucleus_label=nucleus_label,
                user_parameters=user_parameters,
            )

        except IndexError as e: #User loaded a segmentation and no cells can be extracted out of it.
            raise Exception("No cell was fit for quantification in segmentation. This can happen if you loaded empty segmentation or there is a missmatch between cytoplasm and nuclei.\nIf you didn't load segmentation please report the issue as this should not happen.")

    else :
        cell_result_dframe = pd.DataFrame()

    frame_results['acquisition_id'] = acquisition_id
    if type(cell_label) != type(None) and type(nucleus_label) != type(None):
        frame_results['cell_number'] = len(cell_result_dframe) 
    else : 
        frame_results['cell_number'] = NaN
    frame_results['spots'] = spots
    frame_results['clusters'] = clusters
    frame_results['spots_cluster_id'] = spots_cluster_id
    frame_results.update(user_parameters)
    frame_results['threshold'] = user_parameters['threshold']

    frame_results = pd.DataFrame(columns= frame_results.keys(), data= (frame_results.values(),))

    #Adding name column
    result_col = list(frame_results.columns)
    cell_result_col = list(cell_result_dframe.columns)
    name = "acquisition_{0}".format(acquisition_id)
    frame_results['name'] = name
    frame_results = frame_results.loc[:,['name'] + result_col]
    if user_parameters['segmentation_done'] : 
        cell_result_dframe['name'] = name
        cell_result_dframe = cell_result_dframe.loc[:,['name'] + cell_result_col]
        if 'nb_rna_in_nuc' in cell_result_dframe.columns and 'nb_rna_out_nuc' in cell_result_dframe.columns :
            cell_result_dframe['total_rna_number'] = cell_result_dframe['nb_rna_in_nuc'] + cell_result_dframe['nb_rna_out_nuc']
        else : # This can happen when segmentation is performed and detects cells but they are on fov edges and thus removed by big-fish.
            print("\033[1;31m All segmented cells where skipped because they are found on fov edges (incomplete cells), if you want to analyse this image check segmentation.\033[00m")
            cell_result_dframe['nb_rna_in_nuc'] = np.NaN
            cell_result_dframe['nb_rna_out_nuc'] = np.NaN
            cell_result_dframe['total_rna_number'] = np.NaN

        
    return frame_results, cell_result_dframe

def _compute_clustered_spots_dataframe(clustered_spots) :
    if len(clustered_spots) == 0 : return pd.DataFrame(columns= ["id", "cluster_id", "z", "y", "x"])
    z, y ,x, cluster_index = list(zip(*clustered_spots))
    ids = np.arange(len(clustered_spots))

    df = pd.DataFrame({
        "id" : ids
        ,"cluster_id" : cluster_index
        ,"z" : z
        ,"y" : y
        ,"x" : x
    })

    null_idx = df[df['cluster_id'] == -1].index
    df.loc[null_idx, 'cluster_id'] = np.NaN

    return df

def _compute_cluster_dataframe(clusters) :
    if len(clusters) == 0 : return pd.DataFrame(columns= ["id", "z", "y", "x", "spot_number"])
    z, y, x, spots_number, cluster_index = list(zip(*clusters))

    df = pd.DataFrame({
        "id" : cluster_index
        ,"z" : z
        ,"y" : y
        ,"x" : x
        ,"spot_number" : spots_number
    })

    return df

def get_nucleus_signal(image, other_images, user_parameters) :
    if user_parameters['is_multichannel'] :
        rna_signal_channel = user_parameters['channel_to_compute']
        nucleus_signal_channel = user_parameters['nucleus channel signal']
        if type(nucleus_signal_channel) == type(None) :
            return np.zeros(shape=image.shape)

        if rna_signal_channel == nucleus_signal_channel :
            nucleus_signal = image
        
        elif nucleus_signal_channel > rna_signal_channel :
            nucleus_signal_channel -=1
            nucleus_signal = other_images[nucleus_signal_channel]
        
        elif nucleus_signal_channel < rna_signal_channel :
            nucleus_signal = other_images[nucleus_signal_channel]

        return nucleus_signal
    else :
        return image

def _create_threshold_slider(
        logfiltered_image : np.ndarray,
        local_maxima : np.ndarray,
        default : int,
        min_value : int,
        max_value : int,
        voxel_size
) :
    
    if isinstance(default, float) : default = round(default)
    min_value = max(min_value,15) #Security to avoid user put too low threshold and crashes Napari if out of memory.

    @magicgui(
        threshold={'widget_type' : 'Slider', 'value' : default, 'min' : min_value, 'max' : max_value, 'tracking' : True,},
        auto_call=False,
        call_button= "Apply"
    )
    def threshold_slider(threshold: int) -> LayerDataTuple:
        spots = detection.spots_thresholding(
            image=logfiltered_image,
            mask_local_max=local_maxima,
            threshold=threshold
        )[0]

        scale = compute_anisotropy_coef(voxel_size)

        layer_args = {
            'size': 5, 
            'scale' : scale, 
            'face_color' : 'transparent', 
            'border_color' : 'red', 
            'symbol' : 'disc', 
            'opacity' : 0.7, 
            'blending' : 'translucent', 
            'name': 'single spots',
            'features' : {'threshold' : threshold},
            'visible' : True,
            }
        return (spots, layer_args , 'points')
    return threshold_slider

def _apply_log_filter(
        image: np.ndarray,
        voxel_size : tuple,
        spot_radius : tuple,
        log_kernel_size,

) :
    """
    Apply spot detection steps until local maxima step (just before final threshold).
    Return filtered image.
    """
    
    ndim = image.ndim

    if type(log_kernel_size) == type(None) :
        log_kernel_size = get_object_radius_pixel(
                voxel_size_nm=voxel_size,
                object_radius_nm=spot_radius,
                ndim=ndim)
    
    
    image_filtered = stack.log_filter(image, log_kernel_size)
    
    return image_filtered
    
def _local_maxima_mask(
    image_filtered: np.ndarray,
    voxel_size : tuple,
    spot_radius : tuple,
    minimum_distance

    ) : 

    ndim = image_filtered.ndim

    if type(minimum_distance) == type(None) :
        minimum_distance = get_object_radius_pixel(
            voxel_size_nm=voxel_size,
            object_radius_nm=spot_radius,
            ndim=ndim)
    mask_local_max = detection.local_maximum_detection(image_filtered, minimum_distance)
    
    return mask_local_max.astype(bool)

def output_spot_tiffvisual(channel,spots_list, path_output, dot_size = 3, rescale = True):
    
    """
    Outputs a tiff image with one channel being {channel} and the other a mask containing dots where sports are located.
    
    Parameters
    ----------
        channel : np.ndarray
            3D monochannel image
        spots : list[np.ndarray] or np.ndarray
            Spots arrays are ndarray where each element corresponds is a tuple(z,y,x) corresponding to 3D coordinate of a spot
            To plot different spots on different channels a list of spots ndarray can be passed. 
        path_output : str
        dot_size : int
            in pixels
    """
    
    stack.check_parameter(channel = (np.ndarray), spots_list= (list, np.ndarray), path_output = (str), dot_size = (int))
    stack.check_array(channel, ndim= [2,3])
    if isinstance(spots_list, np.ndarray) : spots_list = [spots_list]

    if channel.ndim == 3 : 
        channel = stack.maximum_projection(channel)

    im = np.zeros([1 + len(spots_list)] + list(channel.shape))
    im[0,:,:] = channel

    for level in range(len(spots_list)) :
        if len(spots_list[level]) == 0 : continue
        else :
            spots_mask = np.zeros_like(channel)
            
            #Unpacking spots
            if len(spots_list[level][0]) == 2 :
                Y,X = zip(*spots_list[level])
            elif len(spots_list[level][0]) == 3 :
                Z,Y,X = zip(*spots_list[level])
                del Z
            else :
                Z,Y,X,*_ = zip(*spots_list[level])
                del Z,_
            
            #Reconstructing signal
            spots_mask[Y,X] = 1
            if dot_size > 1 : spots_mask = binary_dilation(spots_mask, iterations= dot_size-1)
            spots_mask = stack.rescale(np.array(spots_mask, dtype = channel.dtype))
            im[level + 1] = spots_mask

    if rescale : channel = stack.rescale(channel, channel_to_stretch= 0)
    stack.save_image(im, path_output, extension= 'tif')