# Python API

The tasks of BBarolo as Python classes: each takes the cube (a FITS file, an astropy HDU or a tuple
`(data, header)`) and the parameters of the task as arguments, with the names of the keywords in
lower case. `run()` runs the task in memory and returns a {class}`~bbarolo.Result`, and `write()` writes
the files of the command line (see {doc}`../user-guide/python`).

## Tasks

```{eval-rst}
.. currentmodule:: bbarolo

.. autoclass:: FitMod3D
.. autoclass:: Bayesian3DFIT
.. autoclass:: BayesianWind
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

## Plots

The plots of 3DFIT and BAYESIAN3DFIT: each function draws one from arrays and returns a matplotlib Figure.
`Result.plot_parameters()`, `plot_chanmaps()`, ... (and the same methods of `Bayesian3DFIT`) call them with
the outputs in memory; the plotting scripts written with the PDF files contain a copy of them.

```{eval-rst}
.. currentmodule:: bbarolo.plots

.. autofunction:: plot_parameters
.. autofunction:: plot_asymdrift
.. autofunction:: plot_chanmaps
.. autofunction:: plot_pvs
.. autofunction:: plot_pvslices
.. autofunction:: plot_kinmaps
.. autofunction:: read_rings
.. autofunction:: read_cube
.. autofunction:: read_map
.. autofunction:: read_columns
.. autoclass:: StyledFigure
```

## Position–velocity slices

```{eval-rst}
.. currentmodule:: bbarolo

.. autofunction:: pv_extract
.. autofunction:: pv_path
.. autofunction:: pv_sample
```

## Lower-level classes

The engine of the tasks, for scripts that need more control than the task classes give. `Param` holds
all the parameters of a run (the keywords of the parameter files, with their defaults); `Cube` is a
datacube with its header, its parameters and its mask: `Cube(fname, par=Param())` reads a FITS file,
`Cube().readHDU(hdu)` an astropy HDU, and `cube.array` is the data, indexed (z, y, x); `Rings` is a set
of tilted rings, one numpy array per parameter.

```{eval-rst}
.. currentmodule:: bbarolo

.. autoclass:: Param
.. autoclass:: Cube
.. autoclass:: Rings
.. autofunction:: help
.. autoexception:: BBaroloError
```
