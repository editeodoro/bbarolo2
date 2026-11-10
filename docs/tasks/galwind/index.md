# GALWIND

GALWIND builds a model datacube of a bi-conical outflow: two opposite cones that start at the centre of a
galaxy and point along the rotation axis of its disk. The gas moves away from the centre along the cones,
may rotate around the axis and has a velocity dispersion. The model is made on the grid of an input cube,
with its beam, to be compared with data; GALWIND does not fit.

In Python: {class}`bbarolo.GalWind`.

## Example

This example puts an outflow in {doc}`NGC 2403 <../../getting-started/example-data>`, only to show the
model: two cones with a full opening angle of 60°, which reach 600 arcsec from the centre, and a gas
velocity of 150 km/s and dispersion of 30 km/s. The disk has the centre, the systemic velocity, the
inclination and the position angle of the galaxy. The cube gives the grid and the beam. The default model
is a stack of 25 cylinders in each cone ({ref}`NLAYERS <par-galwind-nlayers>`, {ref}`WTYPE <par-galwind-wtype>`
0), all with the density {ref}`DENS <par-galwind-dens>`, a flux per unit area in the units of the cube
times km/s per arcsec² (its value here is arbitrary).

::::{tab-set}
:sync-group: lang

:::{tab-item} Parameter file
:sync: par

```text
FITSFILE    ngc2403.fits
GALWIND     true
XPOS        77
YPOS        77
VSYS        132.8
INC         60
PA          123.7
VWIND       150
OPENANG     60
CONELEN     600
VDISP       30
DENS        0.001
```

```bash
BBarolo -p ngc2403_galwind.par
```
:::

:::{tab-item} Python
:sync: py

```python
from bbarolo import GalWind

w = GalWind("ngc2403.fits", xpos=77, ypos=77, vsys=132.8, inc=60, pa=123.7, vwind=150,
            openang=60, conelen=600, vdisp=30, dens=0.001)
res = w.run(threads=4)

res.model                 # the model cube: an astropy PrimaryHDU
res.maps["0mom"]          # moment maps 0, 1 and 2 (also "1mom", "2mom")
res.pv                    # the slice along the axis of the cones
```
:::
::::

The model takes about 2 seconds. The cone to the north-east is blue-shifted and the one to the
south-west red-shifted, by up to about 90 km/s from the systemic velocity. The two cones project on the sky
along the minor axis of the disk:

```{figure} /figures/galwind_maps.png
:alt: Moment maps of the outflow model: total intensity, velocity and dispersion

Moment maps 0, 1 and 2 of the model (`res.maps`).
```

The position–velocity slice runs along the axis of the cones, through the centre. It shows the two
cones as two blobs at opposite offsets and velocities:

```{figure} /figures/galwind_pv.png
:width: 70%
:alt: Position-velocity diagram along the axis of the cones

Position–velocity slice of the model along the axis of the cones (`res.pv`).
```

## Outputs

The files are written in the output folder ({ref}`OUTFOLDER <par-all-outfolder>`, by default
`output/<object>/`) with the prefix `<object>`, the OBJECT of the FITS header. In Python, `run()` keeps
them in memory as attributes of the {class}`~bbarolo.Result`, and `res.write()` writes the files:

| File | `Result` | Content |
|---|---|---|
| `_wind.fits` | `model` | the model cube, on the grid of the input cube |
| `_wind_0mom.fits`, `_wind_1mom.fits`, `_wind_2mom.fits` | `maps` | moment maps 0, 1 and 2 of the model (no mask) |
| `_wind_pv.fits` | `pv` | position–velocity slice through the centre along the axis of the cones |
| `_wind.pdf`, `_wind_maps.pdf`, `pyscript_w.py` | | the plots (channel maps; moment maps and slice) and the script that makes them |
| `_params.par` | | all the parameters of the run, defaults included |

The `Result` also holds the `galwind` object that built the model.

## More

- {doc}`parameters`: all the parameters of GALWIND.
- {doc}`algorithm`: the geometry, the cylinders and the spherical shells, the convolution and the
  differences from BBarolo 1.8.
- {doc}`../galmod/index`: the disk model that the cylinders are built from.

```{toctree}
:hidden:
:maxdepth: 1

parameters
algorithm
```
