# BBarolo 2: 3D-Barolo in pure Python

A pure-Python version of [3D-Barolo](https://github.com/editeodoro/Bbarolo) (BBarolo), the code
that fits 3D tilted-ring models to emission-line datacubes. It ports the C++ BBarolo 1.8 and its
Python wrapper pyBBarolo: the same tasks (3DFIT, GALMOD, SEARCH, 2DFIT, moment maps, PV slices,
...), the same parameter files and the same outputs, with nothing to compile. The numerical
kernels are compiled at run time with numba; the rest uses numpy, scipy, astropy and matplotlib.

**Status:** under development.

```bash
pip install .                     # pip install ".[bayesian]" adds the samplers for BAYESIAN3DFIT
BBarolo -p param.par              # a parameter file, as with the C++ BBarolo
```

```python
import numpy as np
from bbarolo import FitMod3D

f = FitMod3D("ngc2403.fits", radii=np.arange(15, 1200, 30), xpos=77, ypos=77, vsys=132.8,
             vrot=120, vdisp=8, z0=10, inc=60, pa=123.7, free="VROT VDISP", mask="SEARCH")
res = f.run()                     # res.rings: the fitted rings; res.model: the model cube
```

Requires Python 3.9 or later. License: GPL-3.0-or-later.
