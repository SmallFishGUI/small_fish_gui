"""
Submodule containing custom class for napari widgets
"""
import os
import numpy as np
import pandas as pd
import bigfish.detection as detection
import bigfish.stack as stack
from skimage.segmentation import find_boundaries

from napari.layers import Labels, Points, Image
from magicgui import magicgui
from magicgui.widgets import SpinBox, Container
from bigfish.detection import spots_thresholding, get_object_radius_pixel
from napari.types import LayerDataTuple

from abc import ABC, abstractmethod
from typing import Tuple, List
from ..utils import compute_anisotropy_coef

from AF_eraser import remove_autofluorescence_RANSACfit
from pathlib import Path
from ..interface import open_image

class NapariWidget(ABC) :
    """
    Common super class for custom widgets added to napari interface during run
    Each sub class as a specific function, but the widget can be acess with attribute .widget
    """
    def __init__(self):
        self.widget = self._create_widget()

    @abstractmethod
    def _create_widget(self) :
        """
        This should return a widget you can add to the napari (QWidget)
        """
        pass

# Corrector widgets

class ClusterWidget(NapariWidget) :
    """
    Widget for clusters interaction are all init with cluster_layer and single_layer
    """
    def __init__(self, cluster_layer : Points, single_layer : Points):
        self.cluster_layer = cluster_layer
        self.single_layer = single_layer
        super().__init__()

class ClusterWizard(ABC) :
    """
    Common super class for all classes that will interact on single layer and cluster layer to synchronise them or modify their display.
    Their action is started through 'start_listening' method.
    To register them in CLUSTER_WIZARD they should only take single_layer and cluster_layer as arguments
    """

    def __init__(self, single_layer : Points, cluster_layer : Points):
        self.single_layer = single_layer
        self.cluster_layer = cluster_layer
        self.start_listening()
    
    def start_listening(self) :
        """
        This activate the class function. Returns None
        """
        pass


CLUSTER_WIZARDS = []
def register_cluster_wizard(cls):
    """
    Helper to register all clusters related wizard class. Object to be instanciated upon launching napari but that have no widget.
    """
    CLUSTER_WIZARDS.append(cls)
    return cls

def initialize_all_cluster_wizards(single_layer: Points, cluster_layer: Points):
    """
    Initialize all wizards for cluster interaction in Napari
    """
    return [
        cls(single_layer, cluster_layer)
        for cls in CLUSTER_WIZARDS
    ]


class CellLabelEraser(NapariWidget) :
    """
    Widget for deleting cells from multiple label layers in a Napari viewer.
    """
    def __init__(self, label_list: 'list[Labels]'):
        self.label_list = label_list
        if len(self.label_list) == 0 : raise ValueError("Empty label list")
        for label_layer in self.label_list :
            label_layer.events.selected_label.connect((self, 'update'))
        super().__init__()

    def update(self, event) :
        layer : Labels = event.source
        new_label = layer.selected_label
        self.widget.label_number.value = new_label
        self.widget.update()
    
    def _create_widget(self) :
        @magicgui(
                call_button="Delete cell",
                auto_call=False
                )
        def label_eraser(label_number: int) -> None :

            for i, label in enumerate(self.label_list) :
                self.label_list[i].data[label.data == label_number] = 0
                label.refresh()

        return label_eraser


class FreeLabelPicker(NapariWidget) :
    """
    This widget gives the user a free label number
    """
    def __init__(self, label_list : 'list[Labels]'):
        self.label_list = label_list
        if len(self.label_list) == 0 : raise ValueError("Empty label list")
        super().__init__()
    
    def _create_widget(self) :
        @magicgui(
            call_button="Pick free label",
            auto_call=False
        )
        def label_pick()->None :
            max_list = [label_layer.data.max() for label_layer in self.label_list]
            new_label = max(max_list) + 1
            for label_layer in self.label_list :
                label_layer.selected_label = new_label
                label_layer.refresh()

        return label_pick


class SegmentationReseter(NapariWidget) :
    """
    This widget reset the segmentation mask as it used to be when iniating the instance
    """
    def __init__(self, label_list: 'list[Labels]'):
        self.label_list = label_list
        if len(self.label_list) == 0 : raise ValueError("Empty label list")
        self.save = self._get_save(label_list)
        super().__init__()
        
    
    def _get_save(self, label_list : 'list[Labels]') :
        return [label.data.copy() for label in label_list]

    def _create_widget(self) :
        @magicgui(
            call_button= 'Reset segmentation',
            auto_call=False,
        )
        def reset_segmentation() -> None:
            for save_data, layer in zip(self.save, self.label_list) :
                layer.data = save_data.copy()
                layer.refresh()

        return reset_segmentation

class ChangesPropagater(NapariWidget) :
    """
    Apply the changes across the vertical direction (Zstack) if confling values are found for a pixel, max label is kept.
    """
    def __init__(self, label_list):
        self.label_list = label_list
        for label_layer in self.label_list :
            label_layer.events.selected_label.connect((self, 'update'))
        super().__init__()

    def _create_widget(self) :
        @magicgui(
            call_button='Expand label',
            auto_call=False,
        )
        def apply_changes(label_number : int) -> None:
            for layer in self.label_list :
                slices = layer.data.shape[0]
                masked_layer = layer.data.copy()
                masked_layer[masked_layer != label_number] = 0
                masked_layer = np.max(masked_layer, axis=0)
                expanded_layer = np.repeat(masked_layer[np.newaxis], slices, axis=0)
                layer.data[expanded_layer == label_number] = expanded_layer[expanded_layer == label_number]
                layer.refresh()
        return apply_changes

    def update(self, event) :
        layer : Labels = event.source
        new_label = layer.selected_label
        self.widget.label_number.value = new_label
        self.widget.update()

class ClusterIDSetter(ClusterWidget) :
    """
    Allow user to set selected single spots to chosen cluster_id
    """
    def __init__(self, single_layer : Points, cluster_layer : Points):
        super().__init__(cluster_layer, single_layer)

    def _create_widget(self):

        @magicgui(
                call_button= "Set cluster ID",
                auto_call= False,
                cluster_id= {'min' : -1},
        )
        def set_cluster_id(cluster_id : int) :
            if cluster_id == -1 or cluster_id in self.cluster_layer.features['cluster_id'] :
                spots_selection = list(self.single_layer.selected_data)
                cluster_id_in_selection = list(self.single_layer.features.loc[spots_selection,["cluster_id"]].to_numpy().flatten()) + [cluster_id]
                self.single_layer.features.loc[spots_selection,["cluster_id"]] = cluster_id

                for cluster_id in np.unique(cluster_id_in_selection): # Then update number of spots in cluster
                    if cluster_id == -1 : continue
                    new_spot_number = len(self.single_layer.features.loc[self.single_layer.features['cluster_id'] == cluster_id])
                    self.cluster_layer.features.loc[self.cluster_layer.features['cluster_id'] == cluster_id, ["spot_number"]] = new_spot_number
                self.cluster_layer.events.features()
            else :
                print(f"Not cluster with id {cluster_id} was found.")

            self.cluster_layer.selected_data.clear()

        return set_cluster_id

class ClusterMerger(ClusterWidget) :
    """
    Merge all selected clusters by replacing cluster ids of all clusters and belonging points with min for cluster id.
    """
    def __init__(self, cluster_layer, single_layer):
        super().__init__(cluster_layer, single_layer)
    
    
    def _create_widget(self):

        @magicgui(
            call_button="Merge Clusters",
            auto_call=False
        )
        def merge_cluster()-> None :
            selected_clusters = list(self.cluster_layer.selected_data)
            if len(selected_clusters) == 0 : return None

            selected_cluster_ids = self.cluster_layer.features.loc[selected_clusters,['cluster_id']].to_numpy().flatten()
            new_cluster_id = selected_cluster_ids.min()

            #Dropping selected clusters
            self.cluster_layer.data = np.delete(self.cluster_layer.data, selected_clusters, axis=0)

            #Updating spots
            belonging_spots = self.single_layer.features.loc[self.single_layer.features['cluster_id'].isin(selected_cluster_ids)].index
            self.single_layer.features.loc[belonging_spots, ["cluster_id"]] = new_cluster_id

            #Creating new cluster
            centroid = list(self.single_layer.data[belonging_spots].mean(axis=0).round().astype(int))
            spot_number = len(belonging_spots)
            self.cluster_layer.data = np.append(
                self.cluster_layer.data,
                [centroid],
                axis=0
            )

            last_index = len(self.cluster_layer.data) - 1
            self.cluster_layer.features.loc[last_index, ['cluster_id']] = new_cluster_id
            self.cluster_layer.features.loc[last_index, ['spot_number']] = spot_number

            self.cluster_layer.selected_data.clear()
            self.cluster_layer.refresh()

        return merge_cluster

class ClusterUpdater(NapariWidget) :
    """
    Relaunch clustering algorithm taking into consideration new spots, new clusters and deleted clusters.
    """
    def __init__(
            self, 
            single_layer : Points, 
            cluster_layer : Points, 
            default_cluster_radius : int, 
            default_min_spot : int,
            voxel_size : 'tuple[int]'
            ):
        self.single_layer = single_layer
        self.cluster_layer = cluster_layer
        self.cluster_radius = default_cluster_radius
        self.min_spot = default_min_spot
        self.voxel_size = voxel_size
        super().__init__()

    def _compute_clusters(
            self, 
            cluster_radius : int, 
            min_spot : int
            ) -> Tuple[np.ndarray, np.ndarray, dict, dict] :
        """
        Compute clusters using bigfish detection.detect_clusters and seperate coordinates from features.
        """
        
        clustered_spots, clusters = detection.detect_clusters(
            voxel_size=self.voxel_size,
            spots= self.single_layer.data,
            radius=cluster_radius,
            nb_min_spots= min_spot
        )

        clusters_coordinates = clusters[:,:-2]
        clusters_features = {
            "spot_number" : clusters[:,-2],
            "cluster_id" : clusters[:,-1],
        }

        spots_coordinates = clustered_spots[:,:-1]
        spots_features = {
            "cluster_id" : clustered_spots[:,-1]
        }

        return clusters_coordinates, spots_coordinates, clusters_features, spots_features

    def _update_layers(
            self, 
            clusters_coordinates : np.ndarray, 
            spots_coordinates : np.ndarray, 
            clusters_features : dict, 
            spots_features : dict
            ) -> None  :
        """
        Update Points layers inside napari viewer.
        """
        
        #Modify layers
        self.single_layer.data = spots_coordinates
        self.cluster_layer.data = clusters_coordinates
        self.single_layer.features.loc[:,["cluster_id"]] = spots_features['cluster_id']
        self.cluster_layer.features.loc[:,["cluster_id"]] = clusters_features['cluster_id']
        self.cluster_layer.features.loc[:,["spot_number"]] = clusters_features['spot_number']

        self.cluster_layer.selected_data.clear()
        self.single_layer.refresh()
        self.cluster_layer.refresh()

        

    def _create_widget(self):

        @magicgui(
                call_button= "Relaunch Clustering",
                auto_call= False
        )
        def relaunch_clustering(
            cluster_radius : int = self.cluster_radius,
            min_spot : int = self.min_spot,
        ) :
            clusters_coordinates, spots_coordinates, clusters_features, spots_features = self._compute_clusters(cluster_radius=cluster_radius, min_spot=min_spot)
            self._update_layers(clusters_coordinates, spots_coordinates, clusters_features, spots_features )
            self.cluster_radius = cluster_radius
            self.min_spot = min_spot

        return relaunch_clustering

class ClusterCreator(ClusterWidget) :
    """
    Create a cluster containing all and only selected spots located at the centroid of selected points.
    """
    def __init__(self, cluster_layer, single_layer):
        super().__init__(cluster_layer, single_layer)

    def _create_widget(self):

        @magicgui(
                call_button= "Create Cluster",
                auto_call=False
        )
        def create_foci() -> None :
            selected_spots_idx = pd.Index(list(self.single_layer.selected_data))
            free_spots_idx : pd.Index = self.single_layer.features.loc[self.single_layer.features['cluster_id'] == -1].index
            selected_spots_idx = selected_spots_idx[selected_spots_idx.isin(free_spots_idx)]

            spot_number = len(selected_spots_idx)
            if spot_number == 0 :
                print("To create a cluster please select at least 1 spot")
            else :
                
                #Foci creation
                spots_coordinates = self.single_layer.data[selected_spots_idx]
                new_cluster_id = self.cluster_layer.features['cluster_id'].max() + 1
                centroid = list(spots_coordinates.mean(axis=0).round().astype(int))

                self.cluster_layer.data = np.concatenate([
                    self.cluster_layer.data,
                    [centroid]
                ], axis=0)
                
                last_index = len(self.cluster_layer.data) - 1
                self.cluster_layer.features.loc[last_index, ['cluster_id']] = new_cluster_id
                self.cluster_layer.features.loc[last_index, ['spot_number']] = spot_number

                #Update spots cluster_id
                self.single_layer.features.loc[selected_spots_idx,["cluster_id"]] = new_cluster_id
        
        return create_foci

@register_cluster_wizard
class ClusterInspector :
    """
    Listen to event on cluster layer to color spots belonging to clusters in green
    """
    def __init__(self, single_layer : Points, cluster_layer : Points):
        self.single_layer = single_layer
        self.cluster_layer = cluster_layer
        self.start_listening()

    def reset_single_colors(self) -> None:
        self.single_layer.face_color = [0,0,0,0] #transparent
        self.single_layer.refresh()

    def start_listening(self) :

        def color_single_molecule_in_foci() -> None:
            self.reset_single_colors()
            selected_cluster_indices = self.cluster_layer.selected_data
            for idx in selected_cluster_indices :
                selected_cluster = self.cluster_layer.features.at[idx,"cluster_id"]
                belonging_single_idex = self.single_layer.features.loc[self.single_layer.features['cluster_id'] == selected_cluster].index.to_numpy()
                self.single_layer.face_color[belonging_single_idex] = [0,1,0,1] #Green
                self.single_layer.refresh()

        self.cluster_layer.selected_data.events.items_changed.connect(color_single_molecule_in_foci)

@register_cluster_wizard
class ClusterEraser(ClusterWizard) :
    """
    When a foci is deleted, update spots feature table accordingly.
    """

    def __init__(self, single_layer, cluster_layer):
        super().__init__(single_layer, cluster_layer)

    def start_listening(self):
        self.original_remove_selected = self.cluster_layer.remove_selected
    
        def remove_selected_cluster() :
            selected_cluster = self.cluster_layer.selected_data
            for cluster_idx in selected_cluster : #First we update spots data
                cluster_id = self.cluster_layer.features.at[cluster_idx, "cluster_id"]
                self.single_layer.features.loc[self.single_layer.features['cluster_id'] == cluster_id, ['cluster_id']] = -1
            
            self.original_remove_selected() # Then we launch the usual napari method
        
        self.cluster_layer.remove_selected = remove_selected_cluster

@register_cluster_wizard
class ClusterAdditionDisabler(ClusterWizard) :
    """
    Remove the action when user uses points addition tool for Foci, forcing him to use the FociCreator tool to add new cluster.
    """

    def __init__(self, single_layer, cluster_layer):
        super().__init__(single_layer, cluster_layer)
    
    def start_listening(self):

        def print_excuse(*args, **kwargs):
            print("Spot addition is disabled for cluster layer. Use the foci creation tool below after selecting spots you want to cluster")

        self.cluster_layer.add = print_excuse

@register_cluster_wizard
class SingleEraser(ClusterWizard) :
    """
    When a single is deleted, update clusters feature table accordingly
    """

    def __init__(self, single_layer, cluster_layer):
        super().__init__(single_layer, cluster_layer)

    def start_listening(self):
        self._origin_remove_single = self.single_layer.remove_selected

        def delete_single(*args, **kwargs) :
            selected_single_idx = list(self.single_layer.selected_data)
            modified_cluster_ids = self.single_layer.features.loc[selected_single_idx, ["cluster_id"]].to_numpy().flatten()

            print(np.unique(modified_cluster_ids, return_counts=True))
            for cluster_id, count in zip(*np.unique(modified_cluster_ids, return_counts=True)): # Then update number of spots in cluster
                    if cluster_id == -1 : continue
                    new_spot_number = len(self.single_layer.features.loc[self.single_layer.features['cluster_id'] == cluster_id]) - count #minus number of spot with this cluster id we remove
                    print("new spot number : ", new_spot_number)
                    print('target cluster id : ', cluster_id)
                    self.cluster_layer.features.loc[self.cluster_layer.features['cluster_id'] == cluster_id, ["spot_number"]] = new_spot_number
            self._origin_remove_single()
            self.cluster_layer.events.features()
        
        self.single_layer.remove_selected = delete_single


@register_cluster_wizard
class ClusterCleaner(ClusterWizard) :
    """
    Deletes clusters if they drop to 0 single molecules.
    """

    def __init__(self, single_layer, cluster_layer):
        super().__init__(single_layer, cluster_layer)

    def start_listening(self):

        def delete_empty_cluster() :
            drop_idx = self.cluster_layer.features[self.cluster_layer.features['spot_number'] == 0].index
            
            if len(drop_idx) > 0 :
                print("Removing {} empty cluster(s)".format(len(drop_idx)))
                self.cluster_layer.data = np.delete(self.cluster_layer.data, drop_idx, axis=0)
                self.cluster_layer.refresh()

        self.cluster_layer.events.features.connect(delete_empty_cluster)



#Detection widgets
class SpotDetector(NapariWidget) :
    """
    Widget aimed at helping user to set detection parameters : threshold, spot radius and so on...
    """

    def __init__(
        self,
        image: np.ndarray,
        default_threshold : int,
        default_spot_size : tuple,
        default_kernel_size : tuple,
        default_min_distance : tuple,
        voxel_size : tuple,

        ) :
        self.image = image
        self.voxel_size = voxel_size
        self.default_threshold = default_threshold
        self.spot_radius = default_spot_size
        self.kernel_size = default_kernel_size
        self.min_distance = default_min_distance
        self._update_filtered_image()
        self.maximum_threshold = self.filtered_image.max()
        super().__init__()

    def _update_filtered_image(self) :

        self.filtered_image = _apply_log_filter(
            image=self.image,
            voxel_size=self.voxel_size,
            spot_radius=self.spot_radius,
            log_kernel_size=self.kernel_size
        )

        self.local_maxima = _local_maxima_mask(
            image_filtered=self.filtered_image,
            voxel_size=self.voxel_size,
            spot_radius=self.spot_radius,
            minimum_distance=self.min_distance
        )

    def _create_widget(self) :
        
        dim = len(self.voxel_size)
        if dim == 2 :
            tuple_hint = Tuple[int,int]
        else :
            tuple_hint = Tuple[int,int,int]

        @magicgui(
            threshold = {"widget_type" : SpinBox, "min" : "5", "value" : self.default_threshold, "max" : self.filtered_image.max()},
            spot_radius = {"label" : "spot radius(zyx)", "value" : self.spot_radius},
            kernel_size = {"label" : "LoG kernel size(zyx)"},
            minimum_distance = {"label" : "Distance min between spots"},

        )
        def find_spots(
            threshold : int,
            spot_radius : tuple_hint,
            kernel_size : tuple_hint,
            minimum_distance : tuple_hint,
        ) -> List[LayerDataTuple] :

            if (np.array(spot_radius) < 0).any() :
                raise ValueError("Spot radius : set value > 0 (0 to ignore argument)")
            
            if (np.array(kernel_size) < 0).any() :
                raise ValueError("Spot radius : set value > 0 (0 to ignore argument)")
            
            if (np.array(minimum_distance) < 0).any() :
                raise ValueError("Spot radius : set value > 0 (0 to ignore argument)")
            
            if isinstance(spot_radius, tuple) :
                if not all(spot_radius) : spot_radius = None #any value set to 0
            if isinstance(kernel_size, tuple) :
                if not all(kernel_size) : kernel_size = None #any value set to 0
            if isinstance(minimum_distance, tuple) :
                if not all(minimum_distance) : minimum_distance = None #any value set to 0

            do_update = False
            if spot_radius != self.spot_radius :
                self.spot_radius = spot_radius
                do_update = True
            if kernel_size != self.kernel_size :
                self.kernel_size = kernel_size
                do_update = True
            if minimum_distance != self.min_distance :
                self.min_distance = minimum_distance
                do_update = True
            if do_update :
                self._update_filtered_image()
            
            spots = spots_thresholding(
            image=self.filtered_image,
            mask_local_max=self.local_maxima,
            threshold=threshold
            )[0]
    
            scale = compute_anisotropy_coef(self.voxel_size)
    
            spot_layer_args = {
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

            filtered_image_layer_args = {
                "contrast_limits" :  [self.filtered_image.min(), self.filtered_image.max()],
                "colormap" :  'gray',
                "scale" : scale,
                "blending" : 'additive',
                "name" : "filtered image"
            }
            
            return [
                (self.filtered_image, filtered_image_layer_args, 'image'),
                (spots, spot_layer_args, 'points')
                ]
        return find_spots


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

class DenseRegionDeconvolver(NapariWidget) :
    """
    Widget for interactive detection. Create 2 layes : Labels layer representing dense region that could be deconvoluted and Points layer with deconvoluted spots
    """
    def __init__(
        self,
        image : Image,
        spots : Points, 
        alpha : float, 
        beta : float, 
        gamma : float,
        spot_radius : tuple,
        kernel_size : tuple,
        voxel_size : tuple
        ) :
        
        self.image = image
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.spots = spots
        self.spot_radius = spot_radius
        self.kernel_size = kernel_size
        self.voxel_size = voxel_size
        self.update_dense_regions()
        super().__init__()

    def update_dense_regions(self) :
        dense_regions, spot_out_regions,max_size = detection.get_dense_region(
            image=self.image,
            spots=self.spots.data,
            voxel_size = self.voxel_size,
            beta=self.beta,
            spot_radius=self.spot_radius
        )
        del spot_out_regions,max_size

        mask = np.zeros(shape=self.image.shape, dtype= np.int16)
        for label, region in enumerate(dense_regions) :
            reg_im = region.image
            coordinates = np.argwhere(reg_im)
            z,y,x = coordinates.T
            min_z,min_y,min_x,*_ = region.bbox
            z += min_z
            y += min_y
            x += min_x

            mask[z,y,x] = label + 1

        self.dense_regions = mask

    def _create_widget(self) :

        dim = len(self.voxel_size)
        tuple_hint = Tuple[int,int] if dim == 2 else Tuple[int,int,int]
        tuple_dummy = tuple(0 for i in range(dim))

        @magicgui
        def dense_region_deconvolution(
            alpha : float = self.alpha,
            beta : float = self.beta,
            gamma : float = self.gamma,
            spot_radius : tuple_hint = tuple_dummy if self.spot_radius is None else self.spot_radius,
            kernel_size : tuple_hint = tuple_dummy if self.kernel_size is None else self.kernel_size,
        ) -> List[LayerDataTuple] :

            if (np.array(spot_radius) < 0).any() :
                    raise ValueError("Spot radius : set value > 0 (0 to ignore argument)")
            if (np.array(kernel_size) < 0).any() :
                raise ValueError("kernel size : set value > 0 (0 to ignore argument)")
            
            if isinstance(spot_radius,tuple) :
                if not all(spot_radius) : spot_radius = None #any value set to 0
            if isinstance(kernel_size,tuple) :
                if not all(kernel_size) : kernel_size = None #any value set to 0

            do_update = False
            if spot_radius != self.spot_radius :
                self.spot_radius = spot_radius
                do_update = True
            if beta != self.beta :
                self.beta = beta
                do_update=True
            if do_update : self.update_dense_regions()
            self.alpha = alpha
            self.gamma = gamma
            self.kernel_size = kernel_size

            spots, _dense_region, _reference_spot = detection.decompose_dense(
                image= self.image, 
                spots= self.spots.data, 
                voxel_size=self.voxel_size, 
                spot_radius=self.spot_radius, 
                kernel_size=self.kernel_size, 
                alpha=self.alpha, 
                beta=self.beta, 
                gamma=self.gamma
            )
            del _dense_region, _reference_spot

            scale = compute_anisotropy_coef(self.voxel_size)
            spot_layer_args = {
                'size': 5, 
                'scale' : scale, 
                'face_color' : 'transparent', 
                'border_color' : 'blue', 
                'symbol' : 'disc', 
                'opacity' : 0.7, 
                'blending' : 'translucent', 
                'name': 'decovoluted spots',
                'visible' : True,
                }

            dense_region_args = {
                "scale" : scale,
                "name": "Dense regions",
                "colormap" : ["red"] * self.dense_regions.max()
            }

            return [(self.dense_regions, dense_region_args, 'labels'), (spots, spot_layer_args, 'points')]
        return dense_region_deconvolution


class BackgroundRemover(NapariWidget) :
    def __init__(
        self, 
        signal : Image, 
        voxel_size : tuple,
        other_image : np.ndarray = None, 
        ) :
        
        self.other_image = other_image
        self.signal = signal.copy()
        self.voxel_size = voxel_size
        super().__init__()
        if self.other_image is None : self.disable_channel() #Image stack is None when image stack is not is_multichannel


    def disable_channel(self) :
        self.gui.channel.enabled = False

    def _create_widget(self) :
        @magicgui(
            channel = {'min' : 0, 'max' : self.other_image.shape[0] - 1 },
            max_trial = {'min' : 0},
        )
        def remove_background(
            background_path : Path,
            channel : int,
            reset : bool,
            max_trial : int = 100,
        )-> LayerDataTuple :

            self.gui = remove_background

            if reset : 
                result = self.signal
            else :
                if os.path.isfile(background_path) :
                    background = open_image(background_path)
                elif self.other_image is None :
                    raise FileNotFoundError(f"{background_path} is not a valid file.")
                else :
                    background = self.other_image[channel]
                if not background.shape == self.signal.shape : raise ValueError(f"Shape missmatch between signal and background : {self.signal.shape} ; {background.shape}")

                result, score = remove_autofluorescence_RANSACfit(
                    signal=self.signal,
                    background=background,
                    max_trials=max_trial
                )

            scale = compute_anisotropy_coef(self.voxel_size)
            signal_args = {
                "contrast_limits" : [result.min(), result.max()],
                "name" : "raw signal",
                "colormap" : 'green',
                "scale" : scale,
                "blending" : 'additive'
            }

            return (result, signal_args, 'image')
        return remove_background