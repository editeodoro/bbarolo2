# Tasks

BBarolo is organised in **tasks**. A parameter file switches on one or more tasks with their flag (e.g. `3DFIT true`), and they run on the same cube in a fixed order (CONTSUB, SEARCH, SMOOTH, 3DFIT, ...). In Python each task has its class. Every task has a page of **parameters** and one on **how it works**.

## Fitting

- {doc}`3DFIT <3dfit/index>`: Fit a 3D tilted-ring model to an emission-line datacube.
- {doc}`BAYESIAN3DFIT <bayesian3dfit/index>`: Fit a 3D tilted-ring model by nested sampling of the posterior (needs dynesty or nautilus).
- {doc}`BAYESIANWIND <bayesianwind/index>`: Fit a bi-conical wind by nested sampling of the posterior, its patchy emission marginalised.
- {doc}`2DFIT <2dfit/index>`: Fit a 2D tilted-ring model to a velocity field.
- {doc}`SPACEPAR <spacepar/index>`: Compute the full 2D space for any pair of 3DFIT parameters.

## Models

- {doc}`GALMOD <galmod/index>`: Make a 3D model observation datacube of an emission line.
- {doc}`GalMod3D (Python only) <galmod3d/index>`: Make a model datacube from 3D fields of the density and velocity of the gas, as functions of (R, z), (R, φ, z) or (x, y, z).
- {doc}`GALWIND <galwind/index>`: Make a 3D emission-line model of a bi-conical outflow.

## Cubes, masks and sources

- {doc}`SEARCH <search/index>`: Search for sources in a 3D datacube or in a 2D image.
- {doc}`MAKEMASK <makemask/index>`: Define a mask for emission-line data.
- {doc}`SMOOTH <smooth/index>`: Smooth spatially of a datacube with a 2D Gaussian kernel.
- {doc}`SMOOTHSPEC <smoothspec/index>`: Smooth spectrally of a datacube with a given window.
- {doc}`CONTSUB <contsub/index>`: Subtract a polynomial continuum from a cube.
- {doc}`STATS <stats/index>`: Simply calculate and print statistics for a cube.
- {doc}`FITS utilities <fitsutils/index>`: Small tools for FITS files, from the command line: header listing and editing, arithmetic, copies of sub-cubes, unit conversions.

## Maps and profiles

- {doc}`Moment maps and profiles <maps/index>`: Total intensity maps, velocity and dispersion fields, mass surface density, rms maps and global profiles (TOTALMAP, VELOCITYMAP, DISPERSIONMAP, MASSDENSMAP, RMSMAP, GLOBALPROFILE).
- {doc}`PVSLICE <pvslice/index>`: Extracting a position-velocity slice from a datacube.
- {doc}`ELLPROF <ellprof/index>`: Derive radial profiles in elliptical rings from a map.
- {doc}`REND3D <rend3d/index>`: Writing a 3D view of a datacube.

```{toctree}
:hidden:
:maxdepth: 2

general
3dfit/index
bayesian3dfit/index
bayesianwind/index
galmod/index
galmod3d/index
galwind/index
2dfit/index
spacepar/index
search/index
makemask/index
smooth/index
smoothspec/index
contsub/index
stats/index
maps/index
pvslice/index
ellprof/index
rend3d/index
fitsutils/index
```
