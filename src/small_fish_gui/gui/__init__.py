"""
This subpackge contains code related to graphical interface
"""

from .prompts import _error_popup
from .prompts import _warning_popup 
from .prompts import prompt
from .prompts import input_image_prompt
from .prompts import hub_prompt
from .prompts import detection_parameters_promt
from .prompts import coloc_prompt
from .prompts import output_image_prompt
from .prompts import ask_cancel_detection
from .prompts import ask_cancel_segmentation
from .prompts import ask_detection_confirmation
from .prompts import prompt_restore_main_menu

#Helpers to build windows
from .layout import parameters_layout 
from .layout import bool_layout
from .layout import path_layout
from .layout import combo_elmt
from .layout import tuple_layout
from .layout import radio_layout
from .layout import add_header

from .animation import add_default_loading

from .theme import default_theme