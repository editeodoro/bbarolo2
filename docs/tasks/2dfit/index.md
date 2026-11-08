# 2DFIT

2DFIT fits a 2D tilted-ring model to a velocity field, ring by ring: the classic method of ROTCUR. It
gives the rotation velocity, the position angle and the inclination (and, if asked, the systemic
velocity, the centre and the radial velocity) of each ring, and a model velocity field. The fit is fast
and needs no cube model, so it is useful to get first estimates of the parameters. It does not correct
for beam smearing: for that, use {doc}`3DFIT <../3dfit/index>`.

In Python: {class}`bbarolo.FitMod2D`.

## Example

This example fits the rotation velocity, the position angle and the inclination of
{doc}`NGC 2403 <../../getting-started/example-data>` in rings of 30 arcsec. The centre and the systemic
velocity stay at the given values. The velocity field is the first moment of the cube in the mask found
by the source finder.

::::{tab-set}
:sync-group: lang

:::{tab-item} Parameter file
:sync: par

```text
FITSFILE    ngc2403.fits
2DFIT       true
NRADII      41
RADSEP      30
XPOS        77
YPOS        77
VSYS        132.8
VROT        120
INC         60
PA          123.7
FREE        VROT PA INC
MASK        SEARCH
SIDE        B
WFUNC       2
```

```bash
BBarolo -p ngc2403_2dfit.par
```
:::

:::{tab-item} Python
:sync: py

```python
from bbarolo import FitMod2D

f = FitMod2D("ngc2403.fits", nradii=41, radsep=30, xpos=77, ypos=77, vsys=132.8, vrot=120,
             inc=60, pa=123.7, free="VROT PA INC", mask="SEARCH", side="B", wfunc=2)
res = f.run(threads=4)

res.rings                          # the fitted rings: an astropy Table
res.plot_rings(["vrot", "pa", "inc"])   # the first figure below
res.velfield, res.model            # the velocity field and the model, as HDUs
```
:::
::::

The fit takes about 2 seconds. With NRADII = 41 there are 40 rings: the last radius is only the outer
edge of the last ring. The rotation curve rises to about 130 km/s. The inclination and the position
angle change with radius, and the inner rings are the least certain, as the beam (about 60 arcsec) is
as large as the first two rings.

```{figure} /figures/2dfit_rings.png
:width: 75%
:alt: The rotation velocity, position angle and inclination of NGC 2403 against radius, from 2DFIT

The rings fitted by 2DFIT to NGC 2403 (`res.plot_rings(["vrot", "pa", "inc"])`).
```

The model is computed in the rings only, so it covers an ellipse. The velocity field is blank where the
cube has no emission in the mask:

```{figure} /figures/2dfit_velfield.png
:alt: The velocity field of NGC 2403 and the 2D model, side by side

The velocity field fitted (`res.velfield`) and the 2D model (`res.model`), in the same colour scale,
masked to the pixels of the data.
```

## Outputs

The files are written in the output folder ({ref}`OUTFOLDER <par-all-outfolder>`, by default
`output/<object>/`) with the prefix `<object>`, the OBJECT of the FITS header. In Python, `run()` keeps
them in memory as attributes of the {class}`~bbarolo.Result`, and `res.write()` writes the files:

| File | `Result` | Content |
|---|---|---|
| `_2dtrm.txt` | `rings` | the fitted rings |
| `_2d_mod.fits` | `model` | the model velocity field (km/s) |
| `_map_1st.fits` | `velfield` | the velocity field that was fitted (km/s), from a cube |
| `_mask.fits` | `mask` | the mask used to make the velocity field |
| `_params.par` | | all the parameters of the run, defaults included |

The `Result` also has `ringmodel`, the fitting engine, which holds the errors of the parameters.

The rings table has one row per ring: the radius (pixels and arcsec), VSYS, VROT and the radial velocity
VEXP (km/s), PA and INC (degrees), and the centre XPOS and YPOS (pixels). It has no errors. In the `Result`,
`rings` has all the rings at full precision, with NaN in those whose fit failed.

## More

- {doc}`parameters`: all the parameters of 2DFIT.
- {doc}`algorithm`: the velocity field, the model of a ring, the weights, the least-squares fit, the
  errors and the model velocity field.
- {doc}`../3dfit/index`: the fit of the whole cube, which corrects for beam smearing. When PA is
  free, 3DFIT runs a 2D fit of this kind to refine its initial position angle and rotation velocity.

```{toctree}
:hidden:
:maxdepth: 1

parameters
algorithm
```
