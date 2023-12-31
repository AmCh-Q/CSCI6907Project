# CSCI6907Project
This is the [repository](https://github.com/AmCh-Q/CSCI6907Project) for GWU CSCI 6907 Group 8's Class Project "Spatial-Temporal Fusion of Electroencephalography Data for Transformer-Based Gaze Prediction"  

## Overview
Electroencephalography (EEG) is a medical imaging technique that records electrical activity in the brain using electrodes attached to the scalp.  
EEGEyeNet is a [dataset](https://osf.io/ktv7m) as well as a [benchmark](https://github.com/ardkastrati/EEGEyeNet) for Eye tracking prediction based on EEG measurements.  
This repository contains the model as well as our paper proposal.  

## QuickStart
[CSCI6907_Group_8.ipynb](https://github.com/AmCh-Q/CSCI6907Project/blob/main/CSCI6907_Group_8.ipynb) can be [directly launched on Google Colab](https://colab.research.google.com/drive/10Pbkz5nvr2cmPqhuuOsFrV9Xiyn0kzL-?usp=sharing).  
It is recommended to set the session runtime to use a A100 GPU or better and enable High-Ram.

## Dependency
This repository depends on [EEGEyeNet](https://github.com/ardkastrati/EEGEyeNet) as well as its requirements.  
You can use git to clone their repository:  
`git clone https://github.com/ardkastrati/EEGEyeNet.git`  
You can install our requirements listed in requirements.txt through pip:  
`pip install requirements.txt`

## Dataset
EEGEyeNet's datasets are publicly available [here](https://osf.io/ktv7m).  
This project depends on [Position_task_with_dots_synchronised_min.npz](https://osf.io/download/ge87t/) as well as [Position_task_with_dots_synchronised_min_hilbert.npz](https://osf.io/download/bmrn9/)  
You should place them under EEGEyeNet/data:  
`wget -O EEGEyeNet/data/Position_task_with_dots_synchronised_min.npz https://osf.io/download/ge87t/`  
`wget -O EEGEyeNet/data/Position_task_with_dots_synchronised_min_hilbert.npz https://osf.io/download/bmrn9/`

## Installation
You can use git to clone this repository:  
`git clone https://github.com/AmCh-Q/CSCI6907Project.git`  
Alterntatively, just download the python source code files.  
Then copy/move the python source codes to your root EEGEyeNet directory.  
`cp -l CSCI6907Project/our_models.py EEGEyeNet`  
`cp -l CSCI6907Project/EEGViT_pretrained.py EEGEyeNet`  

## Launching Locally
Ensure that your system has at least 16GB of VRAM and 32GB of RAM.
Then make sure your current working directory is EEGEyeNet and have downloaded the datasets and our code.  
Then execute: `python3 main_CS6907_group_8.py`  

## Results
Once execution is complete, the benchmark results of the models can be found in directory ./runs
