""""
Constant submodule to have a common reference for parameters default values
"""
import os

#Image
IS_MULTICHANNEL = False
IS_3D_STACK = False
CHANNEL = 0
NUC_CHANNEL = 1

#Segmentation
FLOW_THRESHOLD = 0.4
CELLPROB_THRESHOD = 0.
CYTO_MODEL = "cpsam"
NUC_MODEL = "cpsam"
CYTO_DIAMETER = 90
NUC_DIAMETER = 60
ANISOTROPY = 1. 
SHOW_SEGMENTATION = True
SEGMENT_ONLY_NUCLEI = False
DO_3D_SEMGENTATION = False
VISUAL_PATH = os.getcwd()
SAVE_SEGMENTATION_VISUAL = False

#Detection
THRESHOLD = None
THRESHOLD_PENALTY = 1
DO_DENSE_REGIONS_DECONVOLUTION = False
DO_CLUSTER_COMPUTATION = False
DO_CLUSTER_COMPUTATION = False
SHOW_NAPARI_CORRECTOR = True
INTERACTIVE_THRESHOLD = False
VOXEL_SIZE = (1,2,3)

#Deconvolution
ALPHA = 0.5
BETA = 1.
GAMMA = 3.

#Clustering
CLUSTER_SIZE = 400
MIN_NUMBER_SPOTS = 5

#Coloc
COLOC_RANGE = 400

#Spots Extraction
DO_CSV = False
DO_EXCEL = False
SPOT_EXTRACTION_FOLDER = os.getcwd()


def get_default_settings() :
    return {
        "multichannel_stack" : IS_MULTICHANNEL,
        "stack_3D" : IS_3D_STACK,
        "detection_channel" : CHANNEL,
        "nucleus_channel" : NUC_CHANNEL,
        "flow_threshold" : FLOW_THRESHOLD,
        "cellprob_threshold" : CELLPROB_THRESHOD,
        "cytoplasm_diameter" : CYTO_DIAMETER,
        "nucleus_diameter" : NUC_DIAMETER,
        "anisotropy" : ANISOTROPY,
        "cytoplasm_model" : CYTO_MODEL,
        "nucleus_model" : NUC_MODEL,
        "show_segmentation" : SHOW_SEGMENTATION,
        "only_nuclei" : SEGMENT_ONLY_NUCLEI,
        "do_3D_segmentation" : DO_3D_SEMGENTATION,
        "save_segmentation_visuals" : SAVE_SEGMENTATION_VISUAL,
        "threshold" : THRESHOLD,
        "threshold_penalty" : THRESHOLD_PENALTY,
        "do_dense_regions_deconvolution" : DO_DENSE_REGIONS_DECONVOLUTION,
        "do_cluster" : DO_CLUSTER_COMPUTATION,
        "show_napari_corrector" : SHOW_NAPARI_CORRECTOR,
        "interactive_threshold_selector" : INTERACTIVE_THRESHOLD,
        "alpha" : ALPHA,
        "beta" : BETA,
        "gamma" : GAMMA,
        "cluster_size" : CLUSTER_SIZE ,
        "min_spot" : MIN_NUMBER_SPOTS,
        "coloc_range" : COLOC_RANGE,
        "do_csv" : DO_CSV,
        "do_excel" : DO_EXCEL,
        "spot_extraction_folder" : SPOT_EXTRACTION_FOLDER,
        "voxel_size" : VOXEL_SIZE,
    }