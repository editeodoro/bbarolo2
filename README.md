<p align="center">
  <img src="https://editeodoro.github.io/Bbarolo/files/bbarolo.jpg" alt="BBarolo logo"/>
</p>

[![Documentation Status](https://img.shields.io/readthedocs/bbarolo)](https://bbarolo.readthedocs.io)
[![License: GPL-3.0](https://img.shields.io/github/license/editeodoro/bbarolo2)](https://github.com/editeodoro/bbarolo2/blob/main/LICENSE.md)
[![Python: 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org)
![Language: Python](https://img.shields.io/github/languages/top/editeodoro/bbarolo2)

BBarolo 2 is a GPL-licensed pure-Python version of [3D-Barolo](https://github.com/editeodoro/Bbarolo)
(BBarolo), the code that fits 3D tilted-ring models to emission-line datacubes to derive the kinematics of
galaxies: rotation curves, velocity dispersions and the geometry of the disc. Fitting the whole datacube rather
than a 2D velocity field, it is not biased by beam smearing and works on data of low spatial resolution. It
ports the C++ BBarolo 1.8 and its Python wrapper pyBBarolo: the same tasks (3DFIT, GALMOD, SEARCH, 2DFIT,
moment maps, PV slices, ...), the same parameter files and the same outputs, with nothing to compile. The
numerical kernels are compiled at run time with numba; the rest uses numpy, scipy, astropy and matplotlib.
3DFIT has three methods: CLASSIC (BBarolo 1.x, ring by ring), GLOBAL (all the rings together, at the maximum of the
likelihood) and BAYESIAN (`METHOD BAYESIAN`: the posterior of the ring parameters, sampled with nested sampling,
[nautilus](https://github.com/johannesulf/nautilus) included in the package). WINDFIT fits bi-conical winds and SLITFIT
long-slit spectra, with the methods GLOBAL and BAYESIAN.

**Status:** under development.

## Installation

BBarolo 2 needs Python 3.9 or later. It is not on PyPI yet: install it from GitHub with ``pip``.

```shell
pip install git+https://github.com/editeodoro/bbarolo2
```

This installs the package `bbarolo` and the command `BBarolo`, with numpy, scipy, astropy, numba and
matplotlib: everything the tasks need, the Bayesian methods included. The optional extra `bayesian` adds the samplers
dynesty and nautilus-sampler (`SAMPLER DYNESTY` and `NAUTILUS`), corner (the corner plot) and threadpoolctl.

```shell
pip install "bbarolo[bayesian] @ git+https://github.com/editeodoro/bbarolo2"
```

To work on the code, install a copy of the repository in editable mode.

```shell
git clone https://github.com/editeodoro/bbarolo2
cd bbarolo2
pip install -e ".[bayesian,test]"
```

## Documentation

You can find the documentation at [bbarolo.readthedocs.io](https://bbarolo.readthedocs.io): installation, a
first fit, every task with its parameters and how it works, examples, and the differences from BBarolo 1.8.

## Example

These examples use the HI datacube of NGC 2403 of the documentation
([ngc2403.fits](https://github.com/editeodoro/Bbarolo/raw/docs/docs/source/tasks/examples/ngc2403.fits), 5 MB).
From the command line, with a parameter file as for the C++ BBarolo:

```shell
BBarolo -p param.par
```

In Python, a 3D fit of the rotation curve and the velocity dispersion:

```python
import numpy as np
from bbarolo import FitMod3D

f = FitMod3D("ngc2403.fits", radii=np.arange(15, 1200, 30), xpos=77, ypos=77, vsys=132.8,
             vrot=120, vdisp=8, z0=10, inc=60, pa=123.7, free="VROT VDISP", mask="SEARCH")
res = f.run()                     # in memory: res.rings (the fitted rings), res.model (the model cube)
res.write("output/")              # the files and plots of the command line
```

and a Bayesian fit (`method="bayesian"`; the engine `Bayesian3DFIT` is used here for the priors), with the posterior of the rotation velocity, the dispersion and the geometry:

```python
from bbarolo import Bayesian3DFIT

f = Bayesian3DFIT("ngc2403.fits", radii=[300, 600, 900], xpos=77, ypos=77, vsys=132.8,
                  vrot=100, vdisp=10, inc=55, pa=123.7, z0=10, mask="SEARCH")
f.priors["vrot"] = dict(name="uniform", loc=50, scale=200)      # km/s
f.priors["vdisp"] = dict(name="uniform", loc=0, scale=30)       # km/s
f.priors["inc"] = dict(name="uniform", loc=40, scale=25)        # degrees
f.priors["phi"] = dict(name="uniform", loc=110, scale=30)       # degrees (phi is PA)
f.sample(freepar="VROT VDISP_SINGLE INC_SINGLE PA_SINGLE SIGMA RADMAX", threads=4,
         sampler_kwargs=dict(nlive=500, seed=1), run_kwargs=dict(n_eff=3000))
f.print_stats()                   # the median and the 68% interval of each parameter
f.corner_plot()                   # needs the package corner
```

## Attribution

A paper describing BBarolo's methods and its tests has been published in the
[Monthly Notices of the Royal Astronomical Society](https://doi.org/10.1093/mnras/stv1213). A draft of the paper
is also available on [arXiv](https://arxiv.org/abs/1505.07834). Please cite the paper if you find BBarolo
helpful in your research.

```
@ARTICLE{2015MNRAS.451.3021D,
   author = {{Di Teodoro}, E.~M. and {Fraternali}, F.},
    title = "{$^{3D}$BAROLO: a new 3D algorithm to derive rotation curves of galaxies}",
  journal = {MNRAS},
     year = 2015,
   volume = 451,
    pages = {3021-3033},
      doi = {10.1093/mnras/stv1213}
}
```

3DFIT, WINDFIT and SLITFIT with METHOD BAYESIAN sample the posterior with nautilus
([Lange 2023](https://doi.org/10.1093/mnras/stad2441)): if you use them, please cite nautilus as well.

```
@article{nautilus,
    author = {Lange, Johannes U},
    title = "{nautilus: boosting Bayesian importance nested sampling with deep learning}",
    journal = {Monthly Notices of the Royal Astronomical Society},
    volume = {525},
    number = {2},
    pages = {3181-3194},
    year = {2023},
    month = {08},
    doi = {10.1093/mnras/stad2441},
    url = {https://doi.org/10.1093/mnras/stad2441},
}
```

## License

BBarolo 2 is licensed under the GNU General Public License v3.0 or later (see [LICENSE.md](LICENSE.md)). It
includes a modified copy of nautilus 1.0.6 (`bbarolo/bayesian/nautilus`), licensed under the MIT License,
Copyright (c) 2022 Johannes U. Lange.
