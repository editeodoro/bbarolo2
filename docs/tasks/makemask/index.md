# MAKEMASK

MAKEMASK builds a mask of an emission-line cube: a cube of 1 (voxel with genuine emission) and 0 (voxel
without). The fits, the maps and the other tasks that need to know where the emission is build the same
masks internally, with the {ref}`MASK <par-makemask-mask>` parameter. MAKEMASK writes the mask as a FITS
file, so that you can look at it, or give it back to another run (`MASK FILE(mask.fits)`).

In Python: {class}`bbarolo.MakeMask`.

## Example

The main types of {ref}`MASK <par-makemask-mask>` are:

| `MASK` | Voxels in the mask |
|---|---|
| `SEARCH` (default) | the detections of the source finder ({doc}`../search/index`), run with its parameters |
| `SMOOTH&SEARCH` | the detections of the source finder on a copy of the cube smoothed spatially to {ref}`FACTOR <par-makemask-factor>` times the beam (2 by default); it follows the faint extended emission |
| `SMOOTHSPEC&SEARCH` | the same, on a cube smoothed along the spectral axis |
| `SMOOTH` | all the voxels of the smoothed cube above {ref}`BLANKCUT <par-makemask-blankcut>` times its noise |
| `NEGATIVE` | the voxels above BLANKCUT times the noise of each channel, measured from its negative values |
| `THRESHOLD` | the voxels above the flux {ref}`THRESHOLD <par-search-threshold>`, in the units of the cube |
| `FILE(mask.fits)` | the voxels not zero in a FITS file |
| `NONE` | all the voxels |

`_ENLARGE` and a number of pixels after the type (e.g. `SEARCH_ENLARGE5`) widens the mask in every channel.
The types ending in `LARGEST` keep only the largest detection.

This example builds the mask `SMOOTH&SEARCH` for {doc}`NGC 2403 <../../getting-started/example-data>`,
with the cube smoothed to twice its beam (about 118 arcsec).

::::{tab-set}
:sync-group: lang

:::{tab-item} Parameter file
:sync: par

```text
FITSFILE    ngc2403.fits
MAKEMASK    true
MASK        SMOOTH&SEARCH
FACTOR      2
```

```bash
BBarolo -p ngc2403_makemask.par
```
:::

:::{tab-item} Python
:sync: py

```python
from bbarolo import MakeMask

m = MakeMask("ngc2403.fits", mask="SMOOTH&SEARCH", factor=2)
res = m.run(threads=4)

res.mask                  # the mask: an HDU with 1 and 0, on the grid of the cube
res.mask.data.sum()       # the number of voxels in the mask
```
:::
::::

The mask is built in less than a second and has 194496 voxels, out of 1395000. Other masks of the same
cube have fewer or more voxels: 154361 for `SEARCH` (with its default SNRCUT of 4), 114134 for
`THRESHOLD` with a threshold of 0.003 Jy/beam and 228377 for `SEARCH_ENLARGE3`. The figure shows the edge of
three masks in one channel map. The smoothed search reaches further out than the plain search, and the
threshold cuts at a fixed flux:

```{figure} /figures/makemask_masks.png
:width: 75%
:alt: Channel 30 of NGC 2403 with the edge of the masks SEARCH, SMOOTH&SEARCH and THRESHOLD

Channel 30 of NGC 2403 and the edge of three masks in it, made with `MakeMask(...).run()` for
`mask="SEARCH"`, `"SMOOTH&SEARCH"` (`factor=2`) and `"THRESHOLD"` (`threshold=0.003`, in Jy/beam).
```

## Outputs

The file is written in the output folder ({ref}`OUTFOLDER <par-all-outfolder>`, by default
`output/<object>/`) with the prefix `<object>`, the OBJECT of the FITS header. In Python, `run()` keeps
it in memory as an attribute of the {class}`~bbarolo.Result`, and `res.write()` writes the file:

| File | `Result` | Content |
|---|---|---|
| `_mask.fits` | `mask` | the mask: 16-bit integers, 1 in the mask and 0 outside, with the header of the cube |
| `_params.par` | | all the parameters of the run, defaults included |

The HISTORY cards of the header record the type of mask, the input file and the parameters of the type.

## More

- {doc}`parameters`: all the parameters of MAKEMASK.
- {doc}`algorithm`: every type of mask, with the formulae, and how the masks are enlarged.
- {doc}`../search/index`: the source finder behind the masks `SEARCH`.
- {doc}`../smooth/index` and {doc}`../smoothspec/index`: the smoothing used by the smoothed masks.

```{toctree}
:hidden:
:maxdepth: 1

parameters
algorithm
```
