# Moment maps and profiles

The MAPS tasks reduce a datacube to images and to a spectrum: the total intensity (moment 0), the
velocity field (moment 1) and the velocity dispersion (moment 2), the surface density of HI, the noise
of each spectrum, the noise and the signal-to-noise of the total map, and the global profile. Each is
requested by its own keyword: TOTALMAP, VELOCITYMAP, DISPERSIONMAP, MASSDENSMAP, RMSMAP, SNMAP and
GLOBALPROFILE. The maps use a mask of the emission, so that the noise does not enter the moments.

In Python: {class}`bbarolo.Maps`.

## Example

This example makes the total intensity, velocity and dispersion maps of
{doc}`NGC 2403 <../../getting-started/example-data>` and its global profile. The mask contains the
voxels brighter than 0.01 Jy/beam. A threshold gives clean maps for this galaxy: the default mask
{ref}`MASK <par-maps-mask>` `SEARCH` also picks up patches of faint, narrow emission at 1–6 km/s
outside the disk, most likely HI of the Milky Way in the foreground, which show as islands at about
0 km/s in the velocity field.

::::{tab-set}
:sync-group: lang

:::{tab-item} Parameter file
:sync: par

```text
FITSFILE      ngc2403.fits
THREADS       4
MASK          THRESHOLD
THRESHOLD     0.01
TOTALMAP      true
VELOCITYMAP   true
DISPERSIONMAP true
GLOBALPROFILE true
```

```bash
BBarolo -p ngc2403_maps.par
```
:::

:::{tab-item} Python
:sync: py

```python
from bbarolo import Maps

m = Maps("ngc2403.fits", mask="THRESHOLD", threshold=0.01,
         totalmap=True, velocitymap=True, dispersionmap=True, globalprofile=True)
res = m.run(threads=4)

res.totalmap              # the maps: astropy HDUs
res.velocitymap
res.dispersionmap
res.spectrum              # the global profile: an astropy Table
```
:::
::::

The run takes about a second. Without any map keyword, the `Maps` class makes the total, velocity and
dispersion maps. The figure shows the three maps (`res.totalmap`, `res.velocitymap` and
`res.dispersionmap`; the colour scales span the 1st to the 99th percentile):

```{figure} /figures/maps_moments.png
:alt: Total intensity, velocity and dispersion maps of NGC 2403

The total intensity (Jy/beam km/s), velocity (km/s) and dispersion (km/s) maps of NGC 2403, with the
mask THRESHOLD 0.01 Jy/beam.
```

The global profile is the sum of each channel over the field, in Jy/beam. It has the two horns of the
profile of a rotating disk:

```{figure} /figures/maps_profile.png
:width: 75%
:alt: Global profile of NGC 2403

The global profile of NGC 2403 (`res.spectrum`).
```

## Outputs

The files are written in the output folder ({ref}`OUTFOLDER <par-all-outfolder>`, by default
`output/<object>/`) with the prefix `<object>`, the OBJECT of the FITS header. In Python they are also
attributes of the {class}`~bbarolo.Result`:

| File | `Result` | Content |
|---|---|---|
| `_map_0th.fits` | `totalmap` | total intensity ({ref}`TOTALMAP <par-maps-totalmap>`), in Jy/beam km/s |
| `_map_0th_RMS.fits`, `_map_0th_SN.fits` | `totalmap_rms`, `snmap` | noise and signal-to-noise of the total map ({ref}`SNMAP <par-maps-snmap>`) |
| `_map_1st.fits` | `velocitymap` | velocity field ({ref}`VELOCITYMAP <par-maps-velocitymap>`), in km/s |
| `_map_2nd.fits` | `dispersionmap` | velocity dispersion ({ref}`DISPERSIONMAP <par-maps-dispersionmap>`), in km/s |
| `_map_massdens.fits` | `massdensmap` | HI surface density ({ref}`MASSDENSMAP <par-maps-massdensmap>`), in M☉/pc² |
| `_map_RMS.fits` | `rmsmap` | noise of each spectrum ({ref}`RMSMAP <par-maps-rmsmap>`), in Jy/beam for this cube |
| `_spectrum.txt` | `spectrum` | the global profile ({ref}`GLOBALPROFILE <par-maps-globalprofile>`) |
| `_mask.fits` | `mask` | the mask used for the maps |
| `_params.par` | | all the parameters of the run, defaults included |

All the maps are also in a dict, `res.maps`. They are FITS images with the spatial grid and the WCS of the
cube. Pixels with no emission in the mask are blank (NaN). With
{ref}`MAPTYPE <par-maps-maptype>` `GAUSSIAN` the maps come from a Gaussian fitted to each spectrum
instead of the moments.

`_spectrum.txt` has two columns: the spectral coordinate of each channel, in the unit of the spectral
axis of the cube (m/s for NGC 2403), and the flux, the sum of the channel over all the pixels, in the
unit of the cube (Jy/beam). The mask does not apply to it.

## More

- {doc}`parameters`: all the parameters of the MAPS tasks.
- {doc}`algorithm`: the moments, the Gaussian maps, the units, the noise and S/N maps, the mass
  surface density and the global profile.
- {doc}`../makemask/index`: the masks that the maps use.
- {doc}`../search/index`: the source finder, which builds the mask `SEARCH`.

```{toctree}
:hidden:
:maxdepth: 1

parameters
algorithm
```
