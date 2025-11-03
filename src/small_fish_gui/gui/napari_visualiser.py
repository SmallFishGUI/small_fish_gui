"""
Contains Napari wrappers to visualise and correct spots/clusters.
"""

import numpy as np
import napari

from magicgui import widgets

from bigfish.stack import check_parameter
from ._napari_widgets import CellLabelEraser, SegmentationReseter, ChangesPropagater, FreeLabelPicker
from ._napari_widgets import ClusterIDSetter, ClusterMerger, ClusterUpdater, ClusterCreator
from ._napari_widgets import initialize_all_cluster_wizards
from ._napari_widgets import SpotDetector
from ..utils import compute_anisotropy_coef

def correct_spots(
        image, 
        spots, 
        voxel_size= (1,1,1), 
        clusters= None, 
        spot_cluster_id= None, 
        cluster_size=None, 
        min_spot_number=0, 
        cell_label= None, 
        nucleus_label= None, 
        other_images =[]
        ):
    """
    Open Napari viewer for user to visualize and corrects spots, clusters.

    Returns
    -------
        new_spots,new_clusters
    """
    check_parameter(image= np.ndarray, voxel_size= (tuple,list))
    dim = len(voxel_size)

    if dim == 3 and type(cell_label) != type(None) :
        cell_label = np.repeat(
            cell_label[np.newaxis],
            repeats= len(image),
            axis=0
        )
    if dim == 3 and type(nucleus_label) != type(None) :
        nucleus_label = np.repeat(
            nucleus_label[np.newaxis],
            repeats= len(image),
            axis=0
        )

    scale = compute_anisotropy_coef(voxel_size)
    Viewer = napari.Viewer(ndisplay=2, title= 'Spot correction', axis_labels=['z','y','x'], show= False)
    Viewer.add_image(image, scale=scale, name= "rna signal", blending= 'additive', colormap='red', contrast_limits=[image.min(), image.max()])
    other_colors = ['green', 'blue', 'gray', 'cyan', 'bop orange', 'bop purple'] * ((len(other_images)-1 // 7) + 1)
    for im, color in zip(other_images, other_colors) :  
        Viewer.add_image(im, scale=scale, blending='additive', visible=False, colormap=color, contrast_limits=[im.min(), im.max()])



    single_layer = Viewer.add_points(  # single molecule spots; this layer can be update by user.
        spots, 
        size = 5, 
        scale=scale, 
        face_color= 'transparent',
        border_color ='red', 
        opacity= 1, 
        symbol= 'disc', 
        name= 'single spots',
        features={
            "cluster_id" : spot_cluster_id if not spot_cluster_id is None else [None] * len(spots),
            "end" : [True] * len(spots)
            }
        )
    
    if type(clusters) != type(None) :
        if len(clusters) > 0 :
            clusters_coordinates = clusters[:, :dim]
        else :
            clusters_coordinates = np.empty(shape=(0,dim), dtype=int) 
        cluster_layer = Viewer.add_points( # cluster; this layer can be update by user.
            clusters_coordinates, 
            size = 10, 
            scale=scale, 
            face_color= 'blue', 
            opacity= 0.7, 
            symbol= 'diamond', 
            name= 'foci', 
            features= {
                "spot_number" : clusters[:,dim], 
                "cluster_id" : clusters[:,dim+1],
                "end" : [True] * len(clusters_coordinates)
                }, 
            feature_defaults= {"spot_number" : 0, "cluster_id" : -2, "end" : True} # napari features default will not work with np.NaN passing -2 instead.
        )

    if type(cell_label) != type(None) and not np.array_equal(nucleus_label, cell_label) : Viewer.add_labels(cell_label, scale=scale, opacity= 0.2, blending= 'additive')
    if type(nucleus_label) != type(None) : Viewer.add_labels(nucleus_label, scale=scale, opacity= 0.2, blending= 'additive')
    
    #Adding widget
    if type(clusters) != type(None) :
        initialize_all_cluster_wizards(
            single_layer=single_layer,
            cluster_layer=cluster_layer
        )

        widget_clusterID = ClusterIDSetter(single_layer=single_layer, cluster_layer=cluster_layer)
        widget_cluster_merge =ClusterMerger(single_layer=single_layer, cluster_layer=cluster_layer)
        widget_cluster_updater = ClusterUpdater(
            single_layer=single_layer,
            cluster_layer=cluster_layer,
            default_cluster_radius= cluster_size,
            default_min_spot= min_spot_number,
            voxel_size=voxel_size
        )
        widget_cluster_creator = ClusterCreator(
            cluster_layer=cluster_layer,
            single_layer=single_layer
        )


        buttons_container = widgets.Container(widgets=[widget_clusterID.widget, widget_cluster_creator.widget], labels=False, layout='horizontal')
        updater_container = widgets.Container(widgets=[widget_cluster_updater.widget, widget_cluster_merge.widget], labels=False)
        tools_container = widgets.Container(
            widgets = [updater_container, buttons_container],
            labels=False,
        )
        Viewer.window.add_dock_widget(tools_container, name='SmallFish', area='left')

    Viewer.show(block=False)
    napari.run()


    if type(clusters) != type(None) :
        new_clusters = np.concatenate([
            cluster_layer.data,
            cluster_layer.features.loc[:,["spot_number","cluster_id"]].to_numpy()
        ],axis=1).astype(int)

        new_spots = np.concatenate([
            single_layer.data,
            single_layer.features.loc[:,["cluster_id"]].to_numpy()
        ], axis=1).astype(int)

        new_cluster_radius = widget_cluster_updater.cluster_radius
        new_min_spot_number = widget_cluster_updater.min_spot

    else :
        new_spots = single_layer.data.astype(int) 
        new_clusters = None
        new_cluster_radius = None
        new_min_spot_number = None

    return new_spots, new_clusters,  new_cluster_radius, new_min_spot_number


# Segmentation
def show_segmentation(
        nuc_image : np.ndarray,
        nuc_label : np.ndarray,
        cyto_image : np.ndarray = None,
        cyto_label : np.ndarray = None,
        anisotrpy : float = 1,
        ) :
    dim = nuc_image.ndim
    
    if type(cyto_image) != type(None) :
        if cyto_image.ndim != nuc_image.ndim : raise ValueError("Cyto and Nuc dimensions missmatch.")
        if type(cyto_label) == type(None) : raise ValueError("If cyto image is passed cyto label must be passed too.")

    if dim == 3 and nuc_label.ndim == 2 :
        nuc_label = np.repeat(
            nuc_label[np.newaxis],
            repeats= len(nuc_image),
            axis=0
        )
    if type(cyto_label) != type(None) :
        
        if type(cyto_image) == type(None) : raise ValueError("If cyto label is passed cyto image must be passed too.")
        
        if dim == 3 and cyto_label.ndim == 2 :
            cyto_label = np.repeat(
                cyto_label[np.newaxis],
                repeats= len(nuc_image),
                axis=0
            )

    #Init Napari viewer
    Viewer = napari.Viewer(ndisplay=2, title= 'Show segmentation', axis_labels=['z','y','x'] if dim == 3 else ['y', 'x'])
    
    scale = (anisotrpy, 1, 1)

    # Adding nuclei
    nuc_signal_layer = Viewer.add_image(nuc_image, name= "nucleus signal", blending= 'additive', colormap='blue', contrast_limits=[nuc_image.min(), nuc_image.max()], scale=scale)
    nuc_label_layer = Viewer.add_labels(nuc_label, opacity= 0.6, name= 'nucleus_label', scale=scale)
    nuc_label_layer.preserve_labels = True
    labels_layer_list = [nuc_label_layer]
    
    #Adding cytoplasm
    if (type(cyto_label) != type(None) and not np.array_equal(cyto_label, nuc_label) ) or (type(cyto_label) != type(None) and cyto_label.max() == 0): 
        Viewer.add_image(cyto_image, name= "cytoplasm signal", blending= 'additive', colormap='red', contrast_limits=[cyto_image.min(), cyto_image.max()], scale=scale)
        cyto_label_layer = Viewer.add_labels(cyto_label, opacity= 0.6, name= 'cytoplasm_label', scale=scale)
        cyto_label_layer.preserve_labels = True
        labels_layer_list += [cyto_label_layer]

    #Adding widget
    label_eraser = CellLabelEraser(labels_layer_list)
    label_picker = FreeLabelPicker(labels_layer_list)
    label_reseter = SegmentationReseter(labels_layer_list)
    changes_applier = ChangesPropagater(labels_layer_list)

    buttons_container = widgets.Container(widgets=[label_picker.widget, label_reseter.widget], labels=False, layout='horizontal')
    changes_applier = widgets.Container(widgets=[changes_applier.widget], labels=False, layout='horizontal')
    tools_container = widgets.Container(
        widgets = [buttons_container, changes_applier, label_eraser.widget],
        labels=False,
    )
    Viewer.window.add_dock_widget(tools_container, name='SmallFish', area='left')

    #Launch Napari
    napari.run()

    new_nuc_label = Viewer.layers['nucleus_label'].data
    if 'cytoplasm_label' in Viewer.layers : 
        new_cyto_label = Viewer.layers['cytoplasm_label'].data
    else : new_cyto_label = new_nuc_label

    return new_nuc_label, new_cyto_label

def threshold_selection(
        image : np.ndarray,
        default_threshold : int,
        default_spot_radius : tuple,
        default_kernel_size : tuple,
        default_min_distance : tuple,
        voxel_size : tuple,
        ) :
    
    """
    To view code for spot selection have a look at magicgui instance created with `detection._create_threshold_slider` which is then passed to this napari wrapper as 'threshold_slider' argument.
    """
    
    Viewer = napari.Viewer(title= "Small fish - Threshold selector", ndisplay=2, show=True)
    scale = compute_anisotropy_coef(voxel_size)
    Viewer.add_image(
        data= image,
        contrast_limits= [image.min(), image.max()],
        name= "raw signal",
        colormap= 'green',
        scale= scale,
        blending= 'additive'
    )
    
    spot_detector = SpotDetector(
        image=image,
        default_threshold= default_threshold,
        default_kernel_size=default_kernel_size,
        default_min_distance=default_min_distance,
        default_spot_size=default_spot_radius,
        voxel_size=voxel_size
    )

    Viewer.window.add_dock_widget(spot_detector.widget, name='threshold_selector')
    spot_detector.widget() #First occurence with auto or entered threshold.
    
    napari.run()

    spots = Viewer.layers['single spots'].data.astype(int)
    if len(spots) == 0 :
        pass
    else :
        threshold = Viewer.layers['single spots'].properties.get('threshold')[0]

    return spots, threshold