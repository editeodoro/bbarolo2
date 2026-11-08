# SMOOTH

SMOOTH degrades the spatial resolution of a datacube. Every channel map is convolved with the Gaussian
that turns the beam of the data into a larger one, and the flux scale is corrected so that a cube in
Jy/beam stays in Jy/beam of the new beam. It is used to bring cubes to a common resolution, or to
detect faint extended emission (the masks `SMOOTH` and `SMOOTH&SEARCH` of {doc}`../makemask/index` use it).

In Python: {class}`bbarolo.Smooth`.

## Example

This example smooths the cube of {doc}`NGC 2403 <../../getting-started/example-data>`, with a beam of
59 arcsec, to a circular beam of 150 arcsec ({ref}`BMAJ <par-smooth-bmaj>`, {ref}`BMIN <par-smooth-bmin>`,
{ref}`BPA <par-smooth-bpa>`; arcsec, arcsec, degrees). With {ref}`REDUCE <par-smooth-reduce>` the smoothed
cube is also averaged in blocks of 2 by 2 pixels (the {ref}`FACTOR <par-smooth-factor>` of 2 by default), as
the pixels of 16 arcsec are now small with respect to the beam.

::::{tab-set}
:sync-group: lang

:::{tab-item} Parameter file
:sync: par

```text
FITSFILE    ngc2403.fits
SMOOTH      true
BMAJ        150
BMIN        150
BPA         0
REDUCE      true
```

```bash
BBarolo -p ngc2403_smooth.par
```
:::

:::{tab-item} Python
:sync: py

```python
from bbarolo import Smooth

s = Smooth("ngc2403.fits", bmaj=150, bmin=150, bpa=0, reduce=True)
res = s.run(threads=4)

res.cube                  # the smoothed cube: an HDU, with the new beam in its header
```
:::
::::

The task takes less than a second. BBarolo prints the beams: the kernel that turns the old beam of
59.33 by 58.61 arcsec into the new one is a Gaussian of 138.1 by 137.8 arcsec, and the flux scale factor
is 6.47. The new cube has 75 by 75 pixels of 32 arcsec, with 62 channels as before. The figure shows
the same channel before and after, with the beams drawn in the lower left corner. The values in Jy/beam are
higher because the new beam is larger:

```{figure} /figures/smooth_channel.png
:alt: Channel 30 of NGC 2403 before and after smoothing to 150 arcsec

Channel 30 of the cube of NGC 2403 before (left) and after (right) `Smooth(..., bmaj=150, bmin=150,
bpa=0, reduce=True).run()`, with the beams.
```

## Outputs

The file is written in the output folder ({ref}`OUTFOLDER <par-all-outfolder>`, by default
`output/<object>/`) with the prefix `<object>`, the OBJECT of the FITS header. In Python, `run()` keeps
it in memory as an attribute of the {class}`~bbarolo.Result`, and `res.write()` writes the file:

| File | `Result` | Content |
|---|---|---|
| `_s<M>.fits` | `cube` | the smoothed cube, `<M>` being the new major axis in arcsec; `_s<M>red.fits` with REDUCE (here `ngc2403_s150red.fits`) |
| `_params.par` | | all the parameters of the run, defaults included |

The header of the new cube has the new beam (BMAJ, BMIN in degrees, BPA in degrees) and two HISTORY cards
with the old and the new beam. The cube in memory is not replaced: tasks that run after SMOOTH in the same
run use the original cube.

## More

- {doc}`parameters`: all the parameters of SMOOTH.
- {doc}`algorithm`: the beams, the convolution kernel, the scale factor and REDUCE.
- {doc}`../smoothspec/index`: the same for the spectral axis.
- {doc}`../makemask/index`: masks made on a smoothed cube.

```{toctree}
:hidden:
:maxdepth: 1

parameters
algorithm
```
