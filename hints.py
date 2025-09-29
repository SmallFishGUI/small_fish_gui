
#Add keys hinting to user_parameters instance keys.

from typing import TypedDict, Tuple
from numpy import ndarray
    
class pipeline_parameters(TypedDict) :
            """
            At run time is a regular dict instance, this class is used for keys hinting
            """
            alpha : float
            anisotropy : float
            beta : float
            channel_to_compute : int
            cellprob_threshold_cyto : float
            cellprob_threshold_nuc : float
            cytoplasm_segmentation_3D : bool
            cluster_size : int
            cyto_model_name : str
            cytoplasm_diameter : int
            cytoplasm_channel : int
            do_cluster_computation : bool
            do_dense_regions_deconvolution : bool
            do_spots_excel : bool
            do_spots_feather : bool
            do_spots_csv : bool
            dim : int
            filename : str
            flow_threshold_cyto : int
            flow_threshold_nuc : int
            gamma : float
            image_path : str
            image : ndarray
            log_kernel_size : Tuple[float,float,float]
            log_kernel_size_x : float
            log_kernel_size_y : float
            log_kernel_size_z : float
            min_number_of_spots : int
            minimum_distance : Tuple[float,float,float]
            minimum_distance_x : float
            minimum_distance_y : float
            minimum_distance_z : float
            is_3D_stack : bool
            is_multichannel : bool
            nucleus_channel_signal : int
            nucleus_segmentation_3D : bool
            nucleus_diameter : int
            nucleus_model_name : str
            nucleus_channel : int
            other_nucleus_image : str
            reordered_shape : Tuple[int,int,int,int,int]
            do_segmentation : bool
            shape : Tuple[int,int,int,int,int]
            save_segmentation_visual : bool
            segmentation_done : bool
            show_interactive_threshold_selector : bool
            spots_extraction_folder : str
            spots_filename : str
            spot_size : Tuple[int,int,int]
            spot_size_x : int
            spot_size_y : int
            spot_size_z : int
            segment_only_nuclei : bool
            cytoplasm_segmentation_3D : bool
            nucleus_segmentation_3D : bool
            show_napari_corrector : bool
            show_segmentation : bool
            threshold : int
            threshold_penalty : int
            time_stack : None
            time_step : None
            voxel_size : Tuple[float,float,float]
            voxel_size_x : float
            voxel_size_y : float
            voxel_size_z : float