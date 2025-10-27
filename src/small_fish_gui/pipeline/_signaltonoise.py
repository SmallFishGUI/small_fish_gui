"""
Signal to noise wrapper from BigFish code.
"""


# ### SNR ###
import bigfish.stack as stack
import numpy as np
from bigfish.detection.utils import get_object_radius_pixel, get_spot_volume, get_spot_surface

def compute_snr_spots(image, spots, voxel_size, spot_radius):
    """
    Modified version of bigfish.detection.utils compute_snr_spots : 
    # Author: Arthur Imbert <arthur.imbert.pro@gmail.com>
    # License: BSD 3 clause
    
    Compute signal-to-noise ratio (SNR) based on spot coordinates.

    .. math::

        \\mbox{SNR} = \\frac{\\mbox{max(spot signal)} -
        \\mbox{mean(background)}}{\\mbox{std(background)}}

    Background is a region twice larger surrounding the spot region. Only the
    y and x dimensions are taking into account to compute the SNR.

    Parameters
    ----------
    image : np.ndarray
        Image with shape (z, y, x) or (y, x).
    spots : np.ndarray
        Coordinate of the spots, with shape (nb_spots, 3) or (nb_spots, 2).
        One coordinate per dimension (zyx or yx coordinates).
    voxel_size : int, float, Tuple(int, float), List(int, float) or None
        Size of a voxel, in nanometer. One value per spatial dimension (zyx or
        yx dimensions). If it's a scalar, the same value is applied to every
        dimensions. Not used if 'log_kernel_size' and 'minimum_distance' are
        provided.
    spot_radius : int, float, Tuple(int, float), List(int, float) or None
        Radius of the spot, in nanometer. One value per spatial dimension (zyx
        or yx dimensions). If it's a scalar, the same radius is applied to
        every dimensions. Not used if 'log_kernel_size' and 'minimum_distance'
        are provided.

    Returns
    -------
    res : dict
        Median signal-to-noise ratio computed for every spots.
        +
            'snr_median' : np.median(snr_spots),
            'snr_mean' : np.mean(snr_spots),
            'snr_std' : np.std(snr_spots),
            'cell_medianbackground_mean' : np.mean(median_background_list),
            'cell_medianbackground_std' : np.std(median_background_list),
            'cell_meanbackground_mean'  : np.mean(mean_background_list),
            'cell_meanbackground_std'  : np.std(mean_background_list),
            'cell_stdbackground_mean' : np.mean(std_background_list),
            'cell_stdbackground_std' : np.std(std_background_list)

    """
    # check parameters
    stack.check_array(
        image,
        ndim=[2, 3],
        dtype=[np.uint8, np.uint16, np.float32, np.float64])
    stack.check_range_value(image, min_=0)
    stack.check_array(
        spots,
        ndim=2,
        dtype=[np.float32, np.float64, np.int32, np.int64])
    stack.check_parameter(
        voxel_size=(int, float, tuple, list),
        spot_radius=(int, float, tuple, list))

    # check consistency between parameters
    ndim = image.ndim
    if ndim != spots.shape[1]:
        raise ValueError("Provided image has {0} dimensions but spots are "
                         "detected in {1} dimensions."
                         .format(ndim, spots.shape[1]))
    if isinstance(voxel_size, (tuple, list)):
        if len(voxel_size) != ndim:
            raise ValueError(
                "'voxel_size' must be a scalar or a sequence with {0} "
                "elements.".format(ndim))
    else:
        voxel_size = (voxel_size,) * ndim
    if isinstance(spot_radius, (tuple, list)):
        if len(spot_radius) != ndim:
            raise ValueError(
                "'spot_radius' must be a scalar or a sequence with {0} "
                "elements.".format(ndim))
    else:
        spot_radius = (spot_radius,) * ndim

    # cast spots coordinates if needed
    if spots.dtype == np.float64:
        spots = np.round(spots).astype(np.int64)

    # cast image if needed
    image_to_process = image.copy().astype(np.float64)

    # clip coordinate if needed
    if ndim == 3:
        spots[:, 0] = np.clip(spots[:, 0], 0, image_to_process.shape[0] - 1)
        spots[:, 1] = np.clip(spots[:, 1], 0, image_to_process.shape[1] - 1)
        spots[:, 2] = np.clip(spots[:, 2], 0, image_to_process.shape[2] - 1)
    else:
        spots[:, 0] = np.clip(spots[:, 0], 0, image_to_process.shape[0] - 1)
        spots[:, 1] = np.clip(spots[:, 1], 0, image_to_process.shape[1] - 1)

    # compute radius used to crop spot image
    radius_pixel = get_object_radius_pixel(
        voxel_size_nm=voxel_size,
        object_radius_nm=spot_radius,
        ndim=ndim)
    radius_signal_ = [np.sqrt(ndim) * r for r in radius_pixel]
    radius_signal_ = tuple(radius_signal_)

    # compute the neighbourhood radius
    radius_background_ = tuple(i * 2 for i in radius_signal_)

    # ceil radii
    radius_signal = np.ceil(radius_signal_).astype(int)
    radius_background = np.ceil(radius_background_).astype(int)

    # loop over spots
    snr_spots = []
    median_background_list = []
    mean_background_list = []
    std_background_list = []
    

    for spot in spots:

        # extract spot images
        spot_y = spot[ndim - 2]
        spot_x = spot[ndim - 1]
        radius_signal_yx = radius_signal[-1]
        radius_background_yx = radius_background[-1]
        edge_background_yx = radius_background_yx - radius_signal_yx
        if ndim == 3:
            spot_z = spot[0]
            radius_background_z = radius_background[0]
            max_signal = image_to_process[spot_z, spot_y, spot_x]
            spot_background_, _ = get_spot_volume(
                image_to_process, spot_z, spot_y, spot_x,
                radius_background_z, radius_background_yx)
            spot_background = spot_background_.copy()

            # discard spot if cropped at the border (along y and x dimensions)
            expected_size = (2 * radius_background_yx + 1) ** 2
            actual_size = spot_background.shape[1] * spot_background.shape[2]
            if expected_size != actual_size:
                continue

            # remove signal from background crop
            spot_background[:,
                            edge_background_yx:-edge_background_yx,
                            edge_background_yx:-edge_background_yx] = -1
            spot_background = spot_background[spot_background >= 0]

        else:
            max_signal = image_to_process[spot_y, spot_x]
            spot_background_, _ = get_spot_surface(
                image_to_process, spot_y, spot_x, radius_background_yx)
            spot_background = spot_background_.copy()

            # discard spot if cropped at the border
            expected_size = (2 * radius_background_yx + 1) ** 2
            if expected_size != spot_background.size:
                continue

            # remove signal from background crop
            spot_background[edge_background_yx:-edge_background_yx,
                            edge_background_yx:-edge_background_yx] = -1
            spot_background = spot_background[spot_background >= 0]

        # compute mean background
        median_background = np.median(spot_background)
        mean_background = np.mean(spot_background)

        # compute standard deviation background
        std_background = np.std(spot_background)

        # compute SNR
        snr = (max_signal - mean_background) / std_background
        snr_spots.append(snr)
        median_background_list.append(median_background)
        mean_background_list.append(mean_background)
        std_background_list.append(std_background)

    #  average SNR
    if len(snr_spots) == 0:
        res = {
            'snr_median' : 0,
            'snr_mean' : 0,
            'snr_std' : 0,
            'cell_medianbackground_mean' : 0,
            'cell_medianbackground_std' : 0,
            'cell_meanbackground_mean'  : 0,
            'cell_meanbackground_std'  : 0,
            'cell_stdbackground_mean' : 0,
            'cell_stdbackground_std' : 0
        }
    else:
        res = {
            'snr_median' : np.median(snr_spots),
            'snr_mean' : np.mean(snr_spots),
            'snr_std' : np.std(snr_spots),
            'cell_medianbackground_mean' : np.mean(median_background_list),
            'cell_medianbackground_std' : np.std(median_background_list),
            'cell_meanbackground_mean'  : np.mean(mean_background_list),
            'cell_meanbackground_std'  : np.std(mean_background_list),
            'cell_stdbackground_mean' : np.mean(std_background_list),
            'cell_stdbackground_std' : np.std(std_background_list)
        }

    return res