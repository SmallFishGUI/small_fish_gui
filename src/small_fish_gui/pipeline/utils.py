import numpy as np

from math import ceil
from itertools import zip_longest
from skimage.measure import regionprops_table


def from_label_get_centeroidscoords(label: np.ndarray):
    """
    Returns
    --------
      centroid : dict{"label": list, "centroid-n": list} 
        n should be replace with 1,2.. according to the axe you wish to access."""

    centroid = regionprops_table(label, properties= ["label","centroid"])
    return centroid