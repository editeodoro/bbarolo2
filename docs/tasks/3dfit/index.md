# 3DFIT

3DFIT fits a 3D tilted-ring model of a rotating disk to an emission-line datacube. The model includes
the beam, the channel width and the thickness of the disk, so the rotation curve and the velocity
dispersion it finds are corrected for beam smearing. This is the main task of BBarolo, and the one to
use for spatially resolved data: HI and CO observations of nearby galaxies, and also IFU and ALMA data
of high-redshift galaxies, with only a few resolution elements across the disk.

In Python: {class}`bbarolo.FitMod3D`.

## Example

This example fits the rotation velocity, the dispersion and the position angle of
{doc}`NGC 2403 <../../getting-started/example-data>` in 41 rings of 30 arcsec. The centre, the systemic
velocity and the inclination stay at the given values. The parameters not given are estimated from
the data.

::::{tab-set}
:sync-group: lang

:::{tab-item} Parameter file
:sync: par

```text
FITSFILE    ngc2403.fits
3DFIT       true
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
FREE        VROT VDISP PA
MASK        SEARCH
NORM        LOCAL
```

```bash
BBarolo -p ngc2403_3dfit.par
```
:::

:::{tab-item} Python
:sync: py

```python
from bbarolo import FitMod3D

f = FitMod3D("ngc2403.fits", nradii=41, radsep=30, xpos=77, ypos=77, vsys=132.8, vrot=120,
             vdisp=8, inc=60, pa=123.7, z0=10, free="VROT VDISP PA", mask="SEARCH", norm="LOCAL")
res = f.run(threads=4)    # in memory: nothing is written

res.rings                 # the fitted rings: an astropy Table
res.plot_rings()          # the figure below
res.plot_pv()
res.plot_chanmaps()       # the plots of the PDF files, in memory (also plot_parameters, plot_pvs, ...)
res.write()               # the files of the command line, in output/ngc2403/
```
:::
::::

The fit takes about 10 seconds on four cores. With the default {ref}`TWOSTAGE <par-3dfit-twostage>`,
the position angle is fitted ring by ring, then regularised and fixed (here to its median, 124°, as it
varies little with radius: see {ref}`REGTYPE <par-3dfit-regtype>`), and the velocities are fitted
again:

```{figure} /figures/3dfit_rings.png
:width: 75%
:alt: The rotation velocity and the velocity dispersion of NGC 2403 against radius

The rotation curve and the velocity dispersion of NGC 2403 from the second stage of the fit
(`res.plot_rings()`).
```

The position–velocity diagrams along the major and minor axes are the quickest check of a fit: the
model (red contours) should follow the data (grey):

```{figure} /figures/3dfit_pv.png
:alt: Position-velocity diagrams of NGC 2403 along the major and minor axes, data and model

Position–velocity slices of the data and of the model along the major and the minor axis
(`res.plot_pv()`).
```

## Outputs

The files are written in the output folder ({ref}`OUTFOLDER <par-all-outfolder>`, by default
`output/<object>/`) with the prefix `<object>`, the OBJECT of the FITS header. In Python, `run()` keeps
them in memory as attributes of the {class}`~bbarolo.Result`, and `res.write()` writes the files:

| File | `Result` | Content |
|---|---|---|
| `_rings_initial.txt` | `rings_initial` | the initial rings |
| `_rings_final1.txt` | `rings_stage1` | the rings of the first stage, with the errors if any |
| `_rings_final2.txt` | `rings` | the rings of the second stage (TWOSTAGE); without it, `rings` is the first stage |
| `_densprof.txt` | `densprof` | the radial profile of the data in the final rings |
| `_densprof_deconvolved.txt` | `densprof_deconvolved` | the density profile deconvolved from the beam, at the geometry of the final rings; its densities build the final model ({ref}`NORM <par-3dfit-norm>`) |
| `_mod_local.fits` | `model` | the model cube, normalised as the last value of NORM (`azim`, `local`, `none`) |
| `_mask.fits` | `mask` | the mask of the fit |
| `maps/` | `maps` | moment maps 0, 1 and 2 of the data and of the model |
| `pvs/` | `pvs` | position–velocity slices of the data, the model and the mask, major (`_a`) and minor (`_b`) axes |
| `_2drings.txt`, `_map_1st.fits` | | the 2D fit of the velocity field used to refine the initial position angle |
| `*.pdf`, `plotscripts/` | | the plots of the fit: parameters, moment maps, PV diagrams and channel maps, and the scripts that make them ({ref}`PLOTS <par-all-plots>`; in Python, `res.write(plots=...)`, and in memory `res.plot_parameters()`, `plot_kinmaps()`, `plot_pvs()`, `plot_pvslices()`, `plot_chanmaps()`, `plot_asymdrift()`) |
| `_asymdrift.txt` | `asymdrift` | the asymmetric drift correction ({ref}`ADRIFT <par-3dfit-adrift>`) |
| `_params.par` | | all the parameters of the run, defaults included |

The `Result` also has `rotation_curve` (the radius and VROT, with the errors, of the rings fitted), `data`
(the cube that was fitted), `residual` (data minus model, in the mask) and `galfit` (the fitting engine).
Its ring tables have all the rings, at full precision, with a column `fitok` (False where the fit of the
ring failed); the files have 3 decimals and leave out the failed rings, unless
{ref}`BADOUT <par-3dfit-badout>`.

The rings files have one row per ring: the radius (arcsec), VROT, VDISP, INC, PA, Z0 (arcsec), DENS,
XPOS, YPOS, VSYS and VRAD, and, with {ref}`FLAGERRORS <par-3dfit-flagerrors>`, the lower and upper errors
of the free parameters.

## More

- {doc}`parameters`: all the parameters of 3DFIT.
- {doc}`algorithm`: the model, the residual function, the minimisation, the two stages, the errors, and the
  accuracy of the fits on simulated disks.
- {doc}`../bayesian3dfit/index`: the same model, fitted by sampling the posterior of all the rings
  together.

```{toctree}
:hidden:
:maxdepth: 1

parameters
algorithm
```
