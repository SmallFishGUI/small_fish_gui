"""
Submodule to handle Small Fish settings.
"""

import os
import json
from .default_settings import get_default_settings
from pydantic import BaseModel, ValidationError
from typing import Tuple, Optional

class SettingsDict(BaseModel) :
    working_directory : str
    do_background_removal : bool
    multichannel_stack : bool
    stack_3D : bool
    detection_channel : int
    nucleus_channel : int
    anisotropy : float
    flow_threshold : float
    cellprob_threshold : float
    cytoplasm_diameter : int
    nucleus_diameter : int
    cytoplasm_model : str
    nucleus_model : str
    show_segmentation : bool
    only_nuclei : bool
    do_3D_segmentation : bool
    save_segmentation_visuals : bool
    threshold : Optional[int]
    threshold_penalty : float
    do_dense_regions_deconvolution : bool
    do_cluster : bool
    show_napari_corrector : bool
    interactive_threshold_selector : bool
    alpha : float
    beta : float
    gamma : float
    cluster_size : int
    min_spot : int
    coloc_range : int
    do_csv : bool
    do_excel : bool
    spot_extraction_folder : str
    voxel_size : tuple


def get_settings() -> SettingsDict :

    setting_path = get_settings_path()

    if os.path.isfile(setting_path) :
        return _load_settings()
    else :
        settings = _init_settings()
        write_settings(settings)
        return settings

def _load_settings() :
    settings_path = get_settings_path()
    with open(settings_path, "r") as f:
        settings = json.load(f)
    
    try : settings = SettingsDict(**settings)

    except ValidationError as e :
        print(f"Incorrect settings, using default settings \n{e}")
        settings = _init_settings()
    
    return settings

def _init_settings() :
    default_settings = get_default_settings()
    return SettingsDict(**default_settings)

def get_settings_path() :
    return os.path.join(os.path.dirname(__file__) , "settings.json")

def write_settings(settings : SettingsDict) :
    if not isinstance(settings, SettingsDict) :
        raise TypeError("Expected SettingsDict type, got {}".format(type(settings)))
    else :
        settings_path = get_settings_path()
        with open(settings_path, mode="w") as f:
             json.dump(settings.dict(), f, indent=4)