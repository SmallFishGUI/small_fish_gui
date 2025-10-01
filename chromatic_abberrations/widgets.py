"""
Widgets for chromatic abberrations correction calibration.
"""

import os
import numpy as np
from pathlib import Path
from typing import Tuple, List
from napari.layers import Points, Image
from typing import Tuple
from magicgui import magicgui
from bigfish.detection import detect_spots
from magicgui.widgets import FunctionGui
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from itertools import cycle

from Sequential_Fish.chromatic_abberrations import CALIBRATION_FOLDER
from ..tools import get_datetime, reorder_image_stack, get_voxel_size_from_metadata
from ..tools.utils import open_image 
from ..customtypes import NapariWidget
from .calibration import match_beads
from .calibration import fit_polynomial_transform_3d
from .calibration import save_fit_model
from .correction import apply_polynomial_transform_to_signal

_calibration_widgets :'list[NapariWidget]' = []

def register_calibration_widget(cls) :
    _calibration_widgets.append(cls)
    return cls

def initiate_all_calibration_widgets() -> 'list[FunctionGui]' :

    widget_list = []
    for cls in _calibration_widgets :
        widget_list.extend(cls().get_widgets())
    return widget_list
        


@register_calibration_widget    
class ImageOpener(NapariWidget) :
    
    def __init__(self):
        super().__init__()

    def _create_widget(self):
        
        @magicgui(
                auto_call=False,
                call_button="Load image",
                image_path = {'label' : 'Image path'},
                is_3D_stack = {'label' : "3D stack"},
                scale = {'label' : "Voxel size(zyx)(nm)"},
        )
        def open_and_order_image(
                image_path : Path,
                is_3D_stack : bool,
                scale : Tuple[int,int,int] = (1,1,1)
        ) -> List[Image] :
            
            if  not os.path.isfile(image_path) :
                raise FileNotFoundError(f"Couldn't find file at {image_path}")
            
            if (np.array(scale) < 1).any() :
                raise ValueError("Please set voxel size with integers >= 1.")
            
            image_path = str(image_path.resolve())
            image = open_image(image_path)
            image = np.squeeze(image)

            pre_map = np.argsort(image.shape)
            if image.ndim ==  2 + 1 + is_3D_stack : # xy + channel + z : is multichannel
                
                if is_3D_stack:
                    image_map = {
                        'c' : pre_map[0],
                        'z' : pre_map[1],
                        'y' : pre_map[2],
                        'x' : pre_map[3]
                    }
                else :
                    image_map = {
                        'c' : pre_map[0],
                        'y' : pre_map[1],
                        'x' : pre_map[2]
                    }

                image = reorder_image_stack(
                    image,
                    channel_map=image_map,
                    is_3D=is_3D_stack
                )

                return [
                    Image(
                    data=image[..., chan],
                    name = f"beads signal channel {chan}",
                    scale= scale,
                    colormap = cmap,
                    blending= 'additive',
                    interpolation2d= 'cubic',
                    interpolation3d= 'cubic',
                    units='nm'
                ) for chan, cmap in zip(range(0,image.shape[-1]), cycle(['red','green','blue']))
                ]
            
            elif image.ndim  == 2 + is_3D_stack : # xy + z : is monochannel
                pass #then is monochannel

            else :
                raise ValueError(f"Wrong number of dimensions, expected {2 + 1 + is_3D_stack} for multichannel or {2+ is_3D_stack} for monochannel.")

            return [Image(
                data=image,
                scale= scale,
                name= "beads signal monochannel",
                blending= 'additive',
                interpolation2d= 'cubic',
                interpolation3d= 'cubic',
                units='nm'
            )]

        def update_scale_on_path_change(event) :
            """
            Read metadata of image when user selects a new file.
            """

            image_path = open_and_order_image.image_path.value
            if not image_path or not os.path.isfile(image_path): pass
            else :
                try :
                    voxel_size = get_voxel_size_from_metadata(image_path)
                    voxel_size = [
                        int(v) if isinstance(v, (int,float)) else 1 for v in voxel_size
                    ]
                except ValueError as e :
                    voxel_size = (1,1,1)

                open_and_order_image.scale.value = voxel_size
        
        open_and_order_image.image_path.changed.connect(update_scale_on_path_change)

        return open_and_order_image

@register_calibration_widget
class BeadsDetector(NapariWidget) :
    """
    This widget allow user to detect beads centers using LoG filter + threshold.
    """
    def __init__(
            self,
            beads_radius = (300,150,150)
            ):
        
        self.default_beads_radius = beads_radius
        super().__init__()

    def _create_widget(self):

        @magicgui(
            beads_image = {'label' : 'Image'},
            threshold = {'label' : 'Threshold', 'min': 1, 'max': 2**64-1},
            beads_radius = {'annotation' : Tuple[int,int,int], 'label' : 'Beads radius (zyx)'},
            auto_call=False,
            call_button="Detect beads"
                
        )
        def detect_beads(
            beads_image : Image,
            threshold : int = 490,
            beads_radius = self.default_beads_radius,
        ) -> Points :
            
            print("Detecting beads...")
            
            voxel_size = tuple(beads_image.scale)
            coordinates = detect_spots(
                beads_image.data,
                threshold=threshold,
                voxel_size=voxel_size,
                spot_radius=beads_radius
            )

            detected_beads = Points(
                data=coordinates, 
                ndim=len(voxel_size),
                name = f"{beads_image.name}_beads",
                blending='additive',
                scale=voxel_size,
                face_color='transparent',
                symbol='disc',
                size=8
                )
            
            print(f"Found {len(detected_beads.data)} beads for {beads_image.name}.")

            return detected_beads
        
        
        return detect_beads

@register_calibration_widget
class ChromaticAberrationCorector(NapariWidget) :
    def __init__(self, degree = 2):

        self.model_x = LinearRegression()
        self.model_y = LinearRegression()
        self.model_z = LinearRegression()
        self.polynomial_features = PolynomialFeatures()
        self.polynomial_features_inv = PolynomialFeatures()
        self.inv_model_x = LinearRegression()
        self.inv_model_y = LinearRegression()
        self.inv_model_z = LinearRegression()
        self.calibration_folder = CALIBRATION_FOLDER
        self.voxel_size = (1,1,1)
        self.degree = degree
        self.timestamp = get_datetime()
        self.save_widget = self._create_save_widget()
        
        super().__init__()

        self.register_widget(self.save_widget)

    def _create_widget(self):
        """
        Perform calibration for chromatic abberration correction and create a layer with corrected signal to evaluate quality of fit.
        """

        @magicgui(
                image_abberation={'label' : 'Image to correct :'},
                spatial_reference_shifted={'label' : 'Points with aberrations'},
                spatial_reference={'label' : 'Points reference'},
                degree={'label' : 'Degree'},
                auto_call=False,
                call_button= "Correct chromatic aberrations",
        )
        def create_corrected_layer(
            image_abberation : Image,
            spatial_reference_shifted : Points,
            spatial_reference : Points,
            degree : int = self.degree,
        ) ->  Image :
            
            voxel_size = spatial_reference.scale
            self.voxel_size = tuple([int(v) for v in voxel_size]) # save as reference if user save calibration
            
            #Convert pixel coordinates to nm to account for anisotropy
            coords1 = spatial_reference.data * voxel_size
            coords2 = spatial_reference_shifted.data * voxel_size

            beads, dist = match_beads(
                coords1= coords1,
                coords2= coords2,
                max_dist= int(max(voxel_size) * 4)
            )

            self.polynomial_features, self.model_x, self.model_y, self.model_z = fit_polynomial_transform_3d(
                                                beads,
                                                dist, 
                                                degree=degree
                                                )
            
            self.polynomial_features_inv, self.inv_model_x, self.inv_model_y, self.inv_model_z = fit_polynomial_transform_3d(
                                                dist, 
                                                beads,
                                                degree=degree
                                                )
            

            image_corrected = apply_polynomial_transform_to_signal(
                image_abberation.data,
                poly=self.polynomial_features,
                model_x=self.model_x,
                model_y=self.model_y,
                model_z=self.model_z,
                voxel_size=voxel_size
            )

            return Image(
                data= image_corrected,
                name= f"{image_abberation.name}_corrected",
                scale= image_abberation.scale,
                blending='additive',
                colormap=image_abberation.colormap,
                interpolation2d= image_abberation.interpolation2d
            )

        self.timestamp = get_datetime()

        return create_corrected_layer
    
    def _create_save_widget(self) :
        """
        This widget allow user to save previously performed calibration.
        """

        @magicgui(
                auto_call=False, 
                call_button= "Save calibration"
                )
        def save_method(
            reference_wavelength : int,
            corrected_wavelength : int,
        ) :
            
            save_fit_model(
                x_fit=self.model_x,
                y_fit=self.model_y,
                z_fit=self.model_z,
                polynomial_features= self.polynomial_features,
                polynomial_features_inv= self.polynomial_features_inv,
                x_inv_fit=self.inv_model_x,
                y_inv_fit=self.inv_model_y,
                z_inv_fit=self.inv_model_z,
                voxel_size=self.voxel_size,
                degree=self.degree,
                timestamp= self.timestamp,
                corrected_wavelength=corrected_wavelength,
                reference_wavelength=reference_wavelength,
            )
        
        return save_method
