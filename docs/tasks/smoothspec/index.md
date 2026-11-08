# SMOOTHSPEC

SMOOTHSPEC degrades the spectral resolution of a datacube: every spectrum is convolved with a window of
a few channels (Hanning, Gaussian, boxcar, ...). It reduces the noise of the spectra at the cost of
resolution, and it is used for example to find faint lines (the mask `SMOOTHSPEC&SEARCH` of
{doc}`../makemask/index` uses it).

In Python: {class}`bbarolo.SmoothSpec`.

## Example

This example smooths the cube of {doc}`NGC 2403 <../../getting-started/example-data>` with a Hanning window
of 11 channels ({ref}`WINDOW_TYPE <par-smoothspec-window-type>`, {ref}`WINDOW_SIZE <par-smoothspec-window-size>`).
The channels are 5.15 km/s wide.

::::{tab-set}
:sync-group: lang

:::{tab-item} Parameter file
:sync: par

```text
FITSFILE     ngc2403.fits
SMOOTHSPEC   true
WINDOW_TYPE  HANNING
WINDOW_SIZE  11
```

```bash
BBarolo -p ngc2403_smoothspec.par
```
:::

:::{tab-item} Python
:sync: py

```python
from bbarolo import SmoothSpec

s = SmoothSpec("ngc2403.fits", window_type="HANNING", window_size=11)
res = s.run()

res.cube                  # the smoothed cube: an HDU on the grid of the input
```
:::
::::

The task takes about 0.1 seconds. The smoothed cube has the same 62 channels and 150 by 150 pixels as the
input. The figure shows the spectrum of the central pixel before and after. The smoothing removes the small
irregularities and lowers the peak a little:

```{figure} /figures/smoothspec_spectrum.png
:alt: Spectrum of the central pixel of NGC 2403 before and after Hanning smoothing

The spectrum of the pixel (77, 77) of the cube of NGC 2403 before and after
`SmoothSpec(..., window_type="HANNING", window_size=11).run()`.
```

With {ref}`REDUCE <par-smoothspec-reduce>` the smoothed cube is also averaged in blocks of channels, 6
channels for this window, and has 10 channels.

## Outputs

The file is written in the output folder ({ref}`OUTFOLDER <par-all-outfolder>`, by default
`output/<object>/`) with the prefix `<object>`, the OBJECT of the FITS header. In Python it is also an
attribute of the {class}`~bbarolo.Result`:

| File | `Result` | Content |
|---|---|---|
| `_h<k>.fits` | `cube` | the smoothed cube, `<k>` being the window size (here `ngc2403_h11.fits`); `_h<k>_red.fits` with REDUCE |
| `_params.par` | | all the parameters of the run, defaults included |

The cube in memory is not replaced: tasks that run after SMOOTHSPEC in the same run use the original cube.

## More

- {doc}`parameters`: all the parameters of SMOOTHSPEC.
- {doc}`algorithm`: the windows, the convolution, the treatment of the ends of the spectrum and REDUCE.
- {doc}`../smooth/index`: the same for the spatial axes.
- {doc}`../makemask/index`: masks made on a smoothed cube.

```{toctree}
:hidden:
:maxdepth: 1

parameters
algorithm
```
