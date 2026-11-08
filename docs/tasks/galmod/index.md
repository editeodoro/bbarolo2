# GALMOD

GALMOD builds a model datacube of an emission line from a set of rings, with the same disk model that
3DFIT fits to the data: each ring has its own rotation velocity, dispersion, inclination, position angle,
thickness and density. The model is made on the grid of an input cube (pixels, channels, beam), so it can
be compared with the data, used to check a fit, or used to make mock observations. GALMOD does not fit
anything.

In Python: {class}`bbarolo.GalMod`.

## Example

This example models {doc}`NGC 2403 <../../getting-started/example-data>` with a flat rotation curve of
120 km/s and a velocity dispersion of 8 km/s in 41 rings of 30 arcsec, with the centre, inclination and
position angle of the galaxy. The cube gives the grid and the beam, and, with
{ref}`NORM <par-galmod-norm>` LOCAL, the flux: every spectrum of the model is scaled to the total flux of
the data in the same pixel (inside the mask). With NORM NONE the densities of the rings (DENS) are used as given.

::::{tab-set}
:sync-group: lang

:::{tab-item} Parameter file
:sync: par

```text
FITSFILE    ngc2403.fits
GALMOD      true
NRADII      41
RADSEP      30
XPOS        77
YPOS        77
VSYS        132.8
VROT        120
VDISP       8
INC         60
PA          123.7
Z0          10
NORM        LOCAL
MASK        SMOOTH
```

```bash
BBarolo -p ngc2403_galmod.par
```
:::

:::{tab-item} Python
:sync: py

```python
from bbarolo import GalMod

g = GalMod("ngc2403.fits", nradii=41, radsep=30, xpos=77, ypos=77, vsys=132.8, vrot=120,
           vdisp=8, inc=60, pa=123.7, z0=10, norm="LOCAL", mask="SMOOTH")
res = g.run(threads=4)    # in memory: res.model is the model cube, an astropy PrimaryHDU
res.plot_pv()             # the figure below
res.plot_maps()           # moment maps of the data and of the model
res.write()               # the files of the command line, in output/ngc2403/

res = g.run(vrot=80, noiserms=0.01, seed=1)      # change parameters, add noise
res.models["noise"]       # the model with noise
```
:::
::::

The run takes about 2 seconds. `run()` builds the model in memory and returns it in a
{class}`~bbarolo.Result`, without writing anything; the moment maps and position–velocity slices are
computed only when they are used (here by `plot_pv()` and `plot_maps()`), so `g.run().model` is a quick way
to get only the cube. Keyword arguments of `run()` change the parameters first, and `noiserms` adds Gaussian
noise (the rms is in the units of the cube; `seed` makes it reproducible). `write()` writes the files.

The model has the same flat rotation curve of 120 km/s everywhere, so it follows the data only roughly.
The position–velocity diagrams show it best: along the major axis the model rises too fast in the centre
and keeps a constant velocity, where NGC 2403 rises slowly to about 130 km/s:

```{figure} /figures/galmod_pv.png
:alt: Position-velocity diagrams of NGC 2403 along the major and minor axes, data and model

Position–velocity slices of the data (grey) and of the model (red contours) along the major and the minor
axis (`res.plot_pv()`).
```

## Outputs

The files are written in the output folder ({ref}`OUTFOLDER <par-all-outfolder>`, by default
`output/<object>/`) with the prefix `<object>`, the OBJECT of the FITS header. In Python, `run()` keeps
them in memory as attributes of the {class}`~bbarolo.Result`, and `res.write()` writes the files:

| File | `Result` | Content |
|---|---|---|
| `_rings_model.txt` | `rings` | the rings that were used |
| `_mod_local.fits` | `model` | the model cube; the name ends with the NORM used (`azim`, `local`, `none`) |
| `_mask.fits` | `mask` | the mask used by NORM and for the data maps |
| `_densprof.txt` | `densprof` | the radial profile of the data in the rings |
| `maps/` | `maps` | moment maps 0, 1 and 2 of the masked data and of the model |
| `pvs/` | `pvs` | position–velocity slices of the data, the model and the mask, major (`_a`) and minor (`_b`) axes |
| `_params.par` | | all the parameters of the run, defaults included |

The `Result` also has `models` (with {ref}`NOISERMS <par-galmod-noiserms>`: the model, and the model with
noise as `models["noise"]`, written as `_mod_noise.fits`), `norm` (the NORM used), `data` (the cube that was
read), `residual` (data minus model) and `diskmodel`, the model engine (a {class}`bbarolo.DiskModel`), which
builds models of any other rings on the same grid. The `rings` table also has VRAD.

The rings file has one row per ring: the radius (arcsec), VROT and VDISP (km/s), INC and PA (degrees),
Z0 (arcsec), the density, XPOS and YPOS (pixels) and VSYS (km/s). With the default
{ref}`GALMODTYPE <par-galmod-galmodtype>` MC the model is a Monte-Carlo sample of gas clouds and has some
noise of its own; ANALYTIC gives the smooth expectation of the same model.

## More

- {doc}`parameters`: all the parameters of GALMOD.
- {doc}`algorithm`: the rings, the geometry, the vertical structure, the Monte-Carlo and analytic models,
  the line profile, the units, the beam and the normalisation.
- {doc}`../3dfit/index`: the same model, fitted to the data.
- {doc}`../galmod3d/index`: models from 3D fields of the gas, instead of rings.
- {doc}`../galwind/index`: a model of a bi-conical outflow.

```{toctree}
:hidden:
:maxdepth: 1

parameters
algorithm
```
