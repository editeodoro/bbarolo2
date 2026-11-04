# Python API

The tasks of BBarolo as Python classes: each takes the cube (a FITS file, an astropy HDU or a tuple
`(data, header)`) and the parameters of the task as arguments, with the names of the keywords in
lower case. `run()` runs the task and returns a {class}`~bbarolo.Result`.

## Tasks

```{eval-rst}
.. currentmodule:: bbarolo

.. autoclass:: FitMod3D
.. autoclass:: BayesianBBarolo
.. autoclass:: GalMod
.. autoclass:: GalMod3D
.. autoclass:: GalWind
.. autoclass:: FitMod2D
.. autoclass:: SpacePar
.. autoclass:: Search
.. autoclass:: MakeMask
.. autoclass:: Smooth
.. autoclass:: SmoothSpec
.. autoclass:: ContSub
.. autoclass:: Stats
.. autoclass:: Maps
.. autoclass:: PVSlice
.. autoclass:: Ellprof
.. autoclass:: Rend3D
```

## Results and other tools

```{eval-rst}
.. currentmodule:: bbarolo

.. autoclass:: Result
.. autofunction:: run
.. autoclass:: DiskModel
.. autoclass:: SimulatedGalaxyCube
.. autofunction:: emptyfits
```
