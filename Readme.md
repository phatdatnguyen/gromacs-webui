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
python -m pip install gradio parmed nglview==4.0
conda install -c conda-forge acpype mdanalysis
```
- To run MD with machine learning potentials:

```
python -m pip install torch==2.8 --index-url https://download.pytorch.org/whl/cu129
python -m pip install cuequivariance cuequivariance-torch
python -m pip install -v --no-build-isolation --config-settings=--global-option=ext torchani
ani build-extensions --sm 8.9 # use --sm 8.9 for RTX 40X0, --sm 12.0 for RTX 50X0
python -m pip install aimnet
python -m pip install pygit2
python -m pip install git+https://github.com/chemle/emle-engine
python -m pip install mace-torch

```

## Start web UI
To start the web UI:

```
conda activate ./gromacs-env
python webui.py
```

## Tests
The suite lives in `tests/` and uses only the standard library's `unittest`, so no
extra packages are needed. Run it from the repository root:

```
conda activate ./gromacs-env
python -m unittest discover
```

Most tests build their own structures and trajectories and run in a few seconds.
The ones in `tests/test_gromacs_workflow.py` drive the real `gmx` binaries and skip
themselves when GROMACS is not on `PATH`; the CHARMM tests additionally skip when
`charmm36` is not installed in the GROMACS tree. Each test works inside a
throwaway directory under `./data/`, cleaned up afterwards.

To run one module or one test:

```
python -m unittest tests.test_utils_species
python -m unittest tests.test_viewers.TrajectoryReductionTests.test_stride_is_computed_to_respect_the_cap
```
