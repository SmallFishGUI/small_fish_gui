import numpy as np
import json, os, joblib
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import NearestNeighbors

from .constant import CALIBRATION_FOLDER
from ..customtypes import Calibration

def match_beads(coords1 : np.array, coords2 : np.array, max_dist=5):
    """Match nearest beads between two channels."""
    nn = NearestNeighbors(n_neighbors=1).fit(coords2)
    distances, indices = nn.kneighbors(coords1)
    matches = distances[:, 0] < max_dist
    return coords1[matches], coords2[indices[matches, 0]]

def fit_polynomial_transform_3d(src_points : np.array, dst_points : np.array, degree=2):
    """Fit 3D polynomial regression mapping coords → dst."""
    poly = PolynomialFeatures(degree)
    X_poly = poly.fit_transform(src_points)
    model_x = LinearRegression().fit(X_poly, dst_points[:, 2])  # x
    model_y = LinearRegression().fit(X_poly, dst_points[:, 1])  # y
    model_z = LinearRegression().fit(X_poly, dst_points[:, 0])  # z
    return poly, model_x, model_y, model_z

def _load_calibration_index() -> dict :
    """
    Create one if doesn't exist.
    """

    index_path = CALIBRATION_FOLDER + "/index.json"

    if os.path.isfile(index_path) :
        with open(index_path, 'r') as f:
            index = json.load(f)
    else:
        index = {}

    return index

def _make_calibration_key(
        reference_wavelength : int,
        corrected_wavelength : int,
) :
    return f"{corrected_wavelength}_{reference_wavelength}"

def update_calibration_index(
        reference_wavelength : int,
        corrected_wavelength : int,
        filename : str,
        ) :
    """
    Add new calibration or overwrite existing one
    """

    file_index = _make_calibration_key(reference_wavelength, corrected_wavelength)

    index = _load_calibration_index()
    index[file_index] = filename

    with open(CALIBRATION_FOLDER + '/index.json', 'w') as index_file:
        json.dump(index, index_file, indent=2)
    
def load_calibration(
        reference_wavelength: int,
        corrected_wavelength: int,
        ) -> Calibration :
    
    index = _load_calibration_index()
    calibration_key = _make_calibration_key(reference_wavelength, corrected_wavelength)

    if calibration_key not in index.keys() :
        raise KeyError(f"No calibration found for reference wavelength {reference_wavelength} and corrected  wavelength {corrected_wavelength}.")

    calibration = joblib.load(index[calibration_key])

    return calibration

def calibration_exist(
        reference_wavelength: int,
        corrected_wavelength: int,
) :
    index = _load_calibration_index()
    calibration_key = _make_calibration_key(reference_wavelength, corrected_wavelength)
    if calibration_key not in index.keys() :
        return False
    else : 
        return True

def save_fit_model(
        x_fit,
        y_fit,
        z_fit,
        polynomial_features,
        polynomial_features_inv,
        x_inv_fit,
        y_inv_fit,
        z_inv_fit,
        voxel_size,
        degree,
        reference_wavelength,
        corrected_wavelength,
        timestamp,
    ) :
        
    if not os.path.isdir(CALIBRATION_FOLDER) : os.makedirs(CALIBRATION_FOLDER)
    filename = CALIBRATION_FOLDER + f"/{reference_wavelength}_{corrected_wavelength}_{timestamp}.joblib" 

    res = joblib.dump({
        'x_fit' : x_fit,
        'y_fit' : y_fit,
        'z_fit' : z_fit,
        'polynomial_features' : polynomial_features,
        'polynomial_features_inv' : polynomial_features_inv,
        'x_inv_fit' : x_inv_fit,
        'y_inv_fit' : y_inv_fit,
        'z_inv_fit' : z_inv_fit,
        'voxel_size' : voxel_size,
        'degree' : degree,
        'reference_wavelength' : reference_wavelength,
        'corrected_wavelength' : corrected_wavelength,
        'timestamp' : timestamp,
    },
    filename
    )

    update_calibration_index(
        reference_wavelength=reference_wavelength,
        corrected_wavelength=corrected_wavelength,
        filename= filename
    )
    
    print(f"Calibration saved at {filename}")