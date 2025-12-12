
#Add keys hinting to user_parameters instance keys.

from typing import TypedDict, Tuple
from dataclasses import dataclass
from numpy import ndarray
    
class pipeline_parameters(TypedDict) :
            """
            At run time is a regular dict instance, this class is used for keys hinting
            """
            alpha : float
            anisotropy : float
            beta : float
            channel_to_compute : int
            cytoplasm_cellprob_threshold : float
            background_channel : int
            cytoplasm_segmentation_3D : bool
            cluster_size : int
            cytoplasm_model_name : str
            cytoplasm_diameter : int
            cytoplasm_channel : int
            cytoplasm_min_size : int
            cytoplasm_max_proj : bool
            cytoplasm_mean_proj : bool
            cytoplasm_select_slice : bool
            cytoplasm_selected_slice : int
            do_cluster_computation : bool
            do_background_removal : bool
            do_dense_regions_deconvolution : bool
            do_spots_excel : bool
            do_spots_feather : bool
            do_spots_csv : bool
            do_background_removal : bool
            dim : int
            filename : str
            cytoplasm_flow_threshold : int
            nucleus_flow_threshold : int
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
            nucleus_cellprob_threshold : float
            nucleus_channel_signal : int
            nucleus_segmentation_3D : bool
            nucleus_diameter : int
            nucleus_model_name : str
            nucleus_channel : int
            nucleus_min_size : int
            nucleus_max_proj : bool
            nucleus_mean_proj : bool
            nucleus_select_slice : bool
            nucleus_selected_slice : int
            other_nucleus_image : str
            reordered_shape : Tuple[int,int,int,int,int]
            do_segmentation : bool
            shape : Tuple[int,int,int,int,int]
            save_segmentation_visuals : bool
            segment_only_nuclei : bool
            segmentation_done : bool
            seg_control_saving_path : str
            show_interactive_threshold_selector : bool
            spots_extraction_folder : str
            spots_filename : str
            spot_size : Tuple[int,int,int]
            spot_size_x : int
            spot_size_y : int
            spot_size_z : int
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