# CONTSUB

CONTSUB removes the continuum emission from a datacube. In every spectrum it fits a polynomial in the
channel number to the channels without line emission and subtracts it from all the channels. It is used
for line observations that contain continuum, before the analysis of the line.

In Python: {class}`bbarolo.ContSub`.

## Example

The cube of NGC 2403 ({doc}`example data <../../getting-started/example-data>`) has no continuum. For this
example the continuum is therefore **synthetic**: a slope that goes from 0.020 Jy/beam in the first channel
to 0.044 Jy/beam in the last one, the same in all the pixels, is added to the cube, which is written as
`ngc2403_cont.fits`.

```python
import numpy as np
from astropy.io import fits

data, hdr = fits.getdata("ngc2403.fits", header=True)
slope = 0.02 + 0.0004 * np.arange(data.shape[0])        # Jy/beam, one value per channel
fits.writeto("ngc2403_cont.fits", (data + slope[:, None, None]).astype("float32"), hdr)
```

The line of NGC 2403 fills almost the whole spectrum, so only the channels 0 to 2 and 57 to 61 are free
of it. CONTSUB fits a straight line ({ref}`CONTORDER <par-contsub-contorder>` 1) to those channels, by
excluding the channels 3 to 56 ({ref}`EXCLWIND <par-contsub-exclwind>`, channels counted from 0).

::::{tab-set}
:sync-group: lang

:::{tab-item} Parameter file
:sync: par

```text
FITSFILE    ngc2403_cont.fits
CONTSUB     true
CONTORDER   1
EXCLWIND    3:56
```

```bash
BBarolo -p ngc2403_contsub.par
```
:::

:::{tab-item} Python
:sync: py

```python
from astropy.io import fits
from bbarolo import ContSub

hdu = fits.open("ngc2403_cont.fits")[0]       # or the file name itself
res = ContSub(hdu, contorder=1, exclwind="3:56").run(threads=4)

res.cube                  # the continuum-subtracted cube: an HDU
```
:::
::::

The task takes about 0.1 seconds. The subtracted cube agrees with the original cube of NGC 2403 to
0.0003 Jy/beam in the central pixel and to 0.0035 Jy/beam at worst (at the edge of the spectrum; the fit uses only
8 noisy channels, and the noise of the cube is 0.00055 Jy/beam). Without EXCLWIND, the line would be part of the
fit and the subtracted cube would be 0.015 to 0.019 Jy/beam too low at the central pixel (here in channels 0, 30 and 61).

```{figure} /figures/contsub_spectrum.png
:alt: Spectrum of the central pixel with a synthetic continuum, the fitted continuum and the spectrum after CONTSUB

The spectrum of the pixel (77, 77) with the synthetic continuum, the continuum that CONTSUB fits, and the
spectrum after `ContSub(..., contorder=1, exclwind="3:56").run()`. The grey band marks the excluded channels.
```

## Outputs

The file is written in the output folder ({ref}`OUTFOLDER <par-all-outfolder>`, by default
`output/<object>/`) with the prefix `<object>`, the OBJECT of the FITS header. In Python it is also an
attribute of the {class}`~bbarolo.Result`:

| File | `Result` | Content |
|---|---|---|
| `_contsub.fits` | `cube` | the continuum-subtracted cube, with the header and the units of the input |
| `_params.par` | | all the parameters of the run, defaults included |

CONTSUB runs before every other task, and replaces the cube in memory: in a run with other tasks (for
example 3DFIT), they use the subtracted cube. In Python, `contsub=True` given to another task subtracts the
continuum first.

## More

- {doc}`parameters`: all the parameters of CONTSUB.
- {doc}`algorithm`: the channels of the fit, the least-squares solution and the treatment of blank voxels.

```{toctree}
:hidden:
:maxdepth: 1

parameters
algorithm
```
