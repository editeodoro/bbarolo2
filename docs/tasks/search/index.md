# SEARCH

SEARCH finds the sources in a datacube (or in a 2D image): the groups of connected voxels that stand
above the noise. It writes a catalogue of the detections, with their positions, sizes, line widths and
fluxes, and maps and cubes of the detected voxels. The source finder is derived from Duchamp. It is
used to find the emission of a galaxy in a cube, and it also builds the masks of the other tasks
({doc}`../makemask/index`).

In Python: {class}`bbarolo.Search`.

## Example

This example searches the cube of {doc}`NGC 2403 <../../getting-started/example-data>` channel map by
channel map ({ref}`SEARCHTYPE <par-search-searchtype>` `spatial`). A voxel starts a detection above
10 times the noise ({ref}`SNRCUT <par-search-snrcut>`), and the detection grows down to 3 times the noise
({ref}`GROWTHCUT <par-search-growthcut>`). A detection is kept if it has at least 30 pixels
({ref}`MINPIX <par-search-minpix>`) and 5 channels ({ref}`MINCHANNELS <par-search-minchannels>`).

::::{tab-set}
:sync-group: lang

:::{tab-item} Parameter file
:sync: par

```text
FITSFILE    ngc2403.fits
SEARCH      true
SEARCHTYPE  spatial
SNRCUT      10
GROWTHCUT   3
MINPIX      30
MINCHANNELS 5
```

```bash
BBarolo -p ngc2403_search.par
```
:::

:::{tab-item} Python
:sync: py

```python
from bbarolo import Search

s = Search("ngc2403.fits", searchtype="spatial", snrcut=10, growthcut=3, minpix=30, minchannels=5)
res = s.run(threads=4)

res.table                 # the detections: an astropy Table
res.detmap                # the number of detected channels in each pixel (an HDU)
res.maps["mom0th"]        # the total intensity map of the detections
```
:::
::::

The search takes about 1 second. It finds one source, the galaxy, with a noise of 0.00055 Jy/beam and
a primary threshold of 0.0056 Jy/beam. The left panel is the number of channels in which each pixel is
detected (at most 40); the right panel is the total intensity map made from the detected voxels only:

```{figure} /figures/search_maps.png
:alt: The map of detected channels and the total intensity map of NGC 2403

The detection map and the total intensity map of the detection in NGC 2403 (`res.detmap` and
`res.maps["mom0th"]`).
```

## Outputs

The files are written in the output folder ({ref}`OUTFOLDER <par-all-outfolder>`, by default
`output/<object>/`) with the prefix `<object>`, the OBJECT of the FITS header. In Python they are also
attributes of the {class}`~bbarolo.Result`:

| File | `Result` | Content |
|---|---|---|
| `_detections.txt` | `table` (and `detections`, the list of the detection objects) | the catalogue of the detections |
| `_detections.fits.gz` | `cube` | the cube of the detected voxels (the flux of the data in them, 0 elsewhere) |
| `_detections_num.fits.gz` | `labels` | the number of the detection of each voxel, 0 elsewhere |
| `_detectmap.fits.gz` | `detmap` | the number of detected channels in each pixel |
| `_mom0th.fits`, `_mom1st.fits`, `_mom2nd.fits` | `maps` | the total intensity, velocity and dispersion maps of the detections |
| `sources/` | `cubelets` | a sub-cube of each detection, with {ref}`CUBELETS <par-search-cubelets>` |
| `_params.par` | | all the parameters of the run, defaults included |

With {ref}`SNMAP <par-search-snmap>` the noise and signal-to-noise maps of the total intensity map are
also written.

The catalogue has one row per detection, sorted by {ref}`SORTSOURCES <par-search-sortsources>` (by number of
voxels, by default). Its header gives the noise and the two thresholds in the flux units of the cube. The
columns are the number of the source (`id`); the centre in pixels and channels (`xcenter`, `ycenter`,
`zcenter`) and in world coordinates (`ra`, `dec` in degrees, `vel` in km/s); the extent along the three
axes (`xwidth`, `ywidth` in pixels, `zwidth` in channels); the number of pixels and of voxels
(`npix`, `nvox`); the systemic velocity (`vsys`) and the line width at 20 per cent of the peak (`w20`),
both in km/s; the integrated flux (`flux`, Jy km/s); and the peak signal-to-noise ratio and flux
(`peaksnr`, `peakflux` in the flux units of the cube). For NGC 2403, the source is centred at pixel (76, 76) and channel 30, at a systemic velocity of
133.4 km/s. It has 142786 voxels in 13039 pixels and 58 channels, a line width W20 of 254.9 km/s and a
flux of 1119 Jy km/s.

## More

- {doc}`parameters`: all the parameters of SEARCH.
- {doc}`algorithm`: the noise and the thresholds, the detection, the merging, the growth and the rejection
  of the objects, and how the parameters of the detections are measured.
- {doc}`../makemask/index`: the masks built with the source finder (`MASK SEARCH`).
- {doc}`../maps/index`: the moment maps of a cube within a mask.

```{toctree}
:hidden:
:maxdepth: 1

parameters
algorithm
```
