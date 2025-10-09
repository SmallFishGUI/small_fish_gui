# Small Fish - A User-Friendly Graphical Interface for smFISH Image Quantification

[![License: BSD-2-Clause](https://img.shields.io/badge/License-BSD_2--Clause-orange.svg)](https://opensource.org/licenses/BSD-2-Clause)
 [![GitHub stars](https://img.shields.io/github/stars/SmallFishGUI/small_fish_gui.svg?style=social)](https://github.com/SmallFishGUI/small_fish_gui) [![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

**Small Fish** is a python application for smFish image analysis. It provides a ready to use graphical interface to synthetize state-of-the-art scientific packages into an automated workflow. Small Fish is designed to simplify images quantification and analysis for people without coding skills. 

Cell segmentation is peformed in 2D and 3D throught cellpose 4.0+(published work) : https://github.com/MouseLand/cellpose; compatible with your own cellpose models.

Spot detection is performed via *big-fish* a python implementation of FishQuant (published work) : https://github.com/fish-quant/big-fish.

***The workflow is fully explained in the [wiki](https://github.com/2Echoes/small_fish_gui/wiki) ! Make sure to check it out.***

## What can you do with small fish ?

✅ Single molecule quantification  
✅ Transcriptomics  
✅ Foci quantification  
✅ Transcription sites quantification  
✅ Nuclear signal quantification  
✅ Signal to noise analysis  
✅ Cell segmentation  
✅ Multichannel colocalisation  

<p align="center">
<img src="https://raw.githubusercontent.com/SmallFishGUI/small_fish_gui/main/illustrations/Segmentation2D.png" width="500" title="Fish_signal" alt="Fish signal">
</p>
<p align="center"><strong>Raw 3D fish signal with dapi</p></strong>  

<p align="center">
<img src="https://raw.githubusercontent.com/SmallFishGUI/small_fish_gui/main/illustrations/Segmentation2D_with_labels.png" width="500" title="Cell segmentation" alt="Segmentation"> 
</p>
<p align="center"><strong>2D segmentation</p></strong>  

<p align="center">
<img src="https://raw.githubusercontent.com/SmallFishGUI/small_fish_gui/main/illustrations/FocciVitrine.png" width="500" title="Detection_signal" alt="Detection_signal">
</p>
<p align="center"><strong> 3D Spot detection</p></strong>  

<p align="center">
<img src="https://raw.githubusercontent.com/SmallFishGUI/small_fish_gui/main/illustrations/FocciVitrine_no_spots.png" width="500" title="Detection filter" alt="detection">
</p>
<p align="center"><strong>Cluster detection</p></strong>  

Analysis can be performed either fully interactively throught a Napari interface or performed automatically through a batch processing allowing for reproducible quantifications. 

## Installation

### General setup

If you don't have a python installation yet I would recommend the [miniconda distribution](https://docs.anaconda.com/free/miniconda/miniconda-other-installer-links/).

It is higly recommanded to create a specific [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) or [virtual](https://docs.python.org/3.6/library/venv.html) environnement to install small fish.

```bash
conda create -n small_fish python=3.9
conda activate small_fish
```
Then download the small_fish package : 
```bash
pip install small_fish_gui
```
Results visualisation is achieved through *Napari* which you can install with :

```bash
pip install napari[all]
```

### Setting up GPU
As of Small Fish 2.0.1 it is **highly** recommanded to set up GPU with cellpose since its new model, CellposeSAM, is very heavy computationally even more when attempting 3D segmentation.  
First of all, try to run small fish gpu without additional commands depending on your configuration it could work straight out of the box. If encoutering any issue try first the following :

```bash
pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision torchaudio
```

```bash
pip uninstall PyQt5 PyQt5-Qt5 PyQt5
```

```bash
pip install -U qtpy PyQt6`
```

If running into additional problems please look at [cellpose documentation](https://cellpose.readthedocs.io/en/latest/installation.html). 

### Run Small fish

First activate your python environnement : 
```bash
conda activate small_fish
```
Then launch Small fish : 
```bash
python -m small_fish_gui
```

You are all set! Try it yourself or check the [get started](https://github.com/2Echoes/small_fish_gui/wiki/Get-started) section in the wiki.
