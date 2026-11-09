"""
bbarolo: 3D-Barolo (BBarolo) in pure Python.

A complete port of the C++ BBarolo code (3D tilted-ring fitting of emission-line
datacubes and related tools) to Python, accelerated with numpy/scipy/numba.

One class per task; each takes a datacube and any BBarolo parameter (the keywords of a
parameter file, case-insensitive) and runs the same code as the command line. run()
computes in memory and writes nothing; write() writes the files of the command line:

    >>> from bbarolo import FitMod3D
    >>> f = FitMod3D("ngc2403.fits", radii=np.arange(15, 1200, 30), xpos=77, ypos=77,
    ...              vsys=132.8, vrot=120, vdisp=8, z0=10, inc=60, pa=123.7,
    ...              free="VROT VDISP", mask="SEARCH", threads=4)
    >>> res = f.run()
    >>> res.rings, res.model
    >>> res.write("output/")                      # or f.write()

Tasks: FitMod3D (3DFIT), GalMod (GALMOD), GalWind (GALWIND), Search (SEARCH),
FitMod2D (2DFIT), Ellprof (ELLPROF), Smooth (SMOOTH), SmoothSpec (SMOOTHSPEC),
Maps (TOTALMAP, VELOCITYMAP, ...), PVSlice (PVSLICE), SpacePar (SPACEPAR),
Rend3D (REND3D), ContSub (CONTSUB), Stats (STATS), MakeMask (MAKEMASK),
BayesianBBarolo (BAYESIAN3DFIT: 3D fit with nested sampling; needs dynesty or nautilus).

    >>> bbarolo.run("ngc2403.par", threads=8)     # a parameter file, as the command line (writes)
    >>> bbarolo.help("free")                      # one parameter; help("3DFIT"): a task

Command line: ``BBarolo -p param.par`` (after pip install), ``python -m bbarolo -p
param.par``, or the launchers run_bbarolo.py and bin/BBarolo (no install needed).

Other tools: GalMod3D (model cubes from 3D fields of the density and velocity of the gas, as
functions of (R, z), (R, phi, z) or (x, y, z); Python only), DiskModel (the model engine of
GALMOD, 3DFIT and BayesianBBarolo: models of a set of Rings on the grid of a Cube, in memory), pv_path, pv_sample, pv_extract (PV
slices in memory), SimulatedGalaxyCube and emptyfits (utils), Param (all the parameters),
Cube, Rings. The modules with the computational engines (galmod, galfit, ringmodel,
search, ...) can also be used directly.
"""

from ._version import __version__, __BBBversion__, __cpp_reference__
from .tasks import (FitMod3D, GalMod, GalWind, Search, FitMod2D, Ellprof, Smooth, SmoothSpec, Maps,
                    PVSlice, SpacePar, Rend3D, ContSub, Stats, MakeMask, run, Result, read_parfile)
from .keywords import help
from .params import Param
from .cube import Cube
from .rings import Rings
from .galmod import DiskModel
from .galmod3d import GalMod3D
from .pvslice import pv_path, pv_sample, pv_extract
from .utils import SimulatedGalaxyCube, emptyfits
from .bbutils import BBaroloError

__all__ = ['FitMod3D', 'GalMod', 'GalWind', 'Search', 'FitMod2D', 'Ellprof', 'Smooth', 'SmoothSpec', 'Maps',
           'PVSlice', 'SpacePar', 'Rend3D', 'ContSub', 'Stats', 'MakeMask', 'BayesianBBarolo', 'BayesianWind', 'run',
           'Result', 'read_parfile', 'Param', 'Cube', 'Rings', 'DiskModel', 'GalMod3D', 'pv_path', 'pv_sample',
           'pv_extract', 'SimulatedGalaxyCube', 'emptyfits', 'BBaroloError', '__version__']

version = __version__
BBBversion = __BBBversion__


# (help is not in __all__, so that 'from bbarolo import *' does not hide Python's help)


def __getattr__(name):
    # BayesianBBarolo and BayesianWind are imported on demand (its samplers are optional dependencies)
    if name == "BayesianBBarolo":
        from .bayesian import BayesianBBarolo
        return BayesianBBarolo
    if name == "BayesianWind":
        from .bayesian import BayesianWind
        return BayesianWind
    raise AttributeError(f"module 'bbarolo' has no attribute '{name}'")
