# PVSLICE

PVSLICE cuts a datacube along a straight line on the sky and gives a position–velocity (PV) diagram: an
image of the spectra along the line. A slice along the major axis of a galaxy shows its rotation curve
directly. The line is given by a centre and a position angle, or by two points, and it can be averaged
across a slit.

In Python: {class}`bbarolo.PVSlice`.

## Example

This example cuts the cube of {doc}`NGC 2403 <../../getting-started/example-data>` along the major axis,
through the centre of the galaxy at the position angle of 123.7 degrees. Positions are in pixels of the
cube, counted from 0.

::::{tab-set}
:sync-group: lang

:::{tab-item} Parameter file
:sync: par

```text
FITSFILE    ngc2403.fits
PVSLICE     true
XPOS_PV     77
YPOS_PV     77
PA_PV       123.7
```

```bash
BBarolo -p ngc2403_pvslice.par
```
:::

:::{tab-item} Python
:sync: py

```python
from bbarolo import PVSlice

# along the major axis
a = PVSlice("ngc2403.fits", xpos_pv=77, ypos_pv=77, pa_pv=123.7).run()

# between two points (pixels), averaged over a slit of 60 arcsec
b = PVSlice("ngc2403.fits", p1_pv="30 105", p2_pv="120 45", width_pv=60).run()

a.pv                    # the PV diagram: an HDU, axes offset (deg) and velocity (m/s)
```
:::
::::

The slice is made in a fraction of a second. The image has 62 rows, the channels of the cube, and 179
columns, one per pixel along the line. The second slice goes from the first point to the second, and is
averaged over a slit of 60 arcsec (about one beam) across the line.

```{figure} /figures/pvslice_pv.png
:alt: Position-velocity diagrams of NGC 2403: along the major axis and between two points

The two slices of the example (`a.pv` and `b.pv`), with the offset in arcsec and the velocity in km/s.
The offset is 0 at the centre (left) or at the mid-point (right) of the slice.
```

## Outputs

The file is written in the output folder ({ref}`OUTFOLDER <par-all-outfolder>`, by default
`output/<object>/`) with the prefix `<object>`, the OBJECT of the FITS header. In Python, `run()` keeps
it in memory as an attribute of the {class}`~bbarolo.Result`, and `res.write()` writes the file:

| File | `Result` | Content |
|---|---|---|
| `_pv_<PA>.fits` | `pv` | the PV diagram, in the flux unit of the cube |
| `_params.par` | | all the parameters of the run, defaults included |

`<PA>` is PA_PV rounded to an integer (`_pv_124.fits` in the example, and `_pv_0.fits` for a slice between two
points). The first axis of the image is the offset along the line, in degrees (header CDELT1 and CUNIT1)
and 0 at the centre, and the second axis is the spectral axis of the cube, here in m/s. The `HISTORY`
card of the header gives the centre and the angle, or the two points. To make a slice in memory, without a
Result, use `bbarolo.pv_extract`.

## More

- {doc}`parameters`: all the parameters of PVSLICE.
- {doc}`algorithm`: the line, the sampling of the cube, the slit and the output image.
- {doc}`../3dfit/index`: also makes the PV diagrams of the data and of the model along the major and minor
  axes.

```{toctree}
:hidden:
:maxdepth: 1

parameters
algorithm
```
