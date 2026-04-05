## Introduction
This web UI is for running molecular dynamics simulation with [Gromacs](https://www.gromacs.org/):

## Installation  (Linux only)
- Install [Anaconda](https://www.anaconda.com/download)

- Clone this repo: Open terminal

```
git clone https://github.com/phatdatnguyen/gromacs-webui
```

- Create and activate conda virtual environment:

```
cd gromacs-webui
conda create -p ./gromacs-env python=3.12
conda activate ./gromacs-env
```

- Install packages:

```
pip install gradio_modal
pip install parmed
pip install nglview==4.0
conda install -c conda-forge acpype
conda install -c conda-forge mdanalysis
```

## Start web UI
To start the web UI:

```
conda activate ./gromacs-env
python webui.py
```