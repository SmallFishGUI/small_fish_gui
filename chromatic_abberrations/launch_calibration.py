"""
This script launch napari to allow user to calibrate chromatic abberration correction using fluorescent beads.
"""

import napari,os
from magicgui import widgets
from Sequential_Fish.chromatic_abberrations.widgets import initiate_all_calibration_widgets

def main() :
    Viewer = napari.Viewer(title= "SequentialFish - Chromatic abberration calibration")

    calibration_widgets = initiate_all_calibration_widgets()
    right_container = widgets.Container(widgets=calibration_widgets, labels=False)
    Viewer.window.add_dock_widget(right_container, name='Calibration tools', area='right')
    napari.run()


if __name__ == "__main__" :
    os.environ["QT_QPA_PLATFORM"] = "xcb"
    main()
    print("calibration closed")