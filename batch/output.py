import numpy as np
import os

def output_masks(
        batch_path : str,
        acquisition_name : str,
        nucleus_label : np.ndarray,
        cytoplasm_label : np.ndarray = None,
) :

    os.makedirs(batch_path + "/segmentation_masks/", exist_ok=True)
    output_path = batch_path + "/segmentation_masks/{0}".format(acquisition_name)

    np.save(output_path + "_nucleus.npy", arr= nucleus_label)
    if type(cytoplasm_label) != type(None) :
        np.save(output_path + "_cytoplasm.npy", arr= nucleus_label)
