# ELLPROF

ELLPROF measures the radial profile of a map in elliptical rings, which are circular rings of a disk
projected on the sky. For each ring it gives the mean, the median and the scatter of the map, the surface
density on the sky and face-on, and, for HI maps, the HI column density and mass surface density. The
input is a 2D map, or a cube, from which the total intensity map is made. It is the quick way to get
the surface-density profile of a galaxy for the inclination and position angle given.

The beam smooths the map, and with it the profile: for a disk a few beams across, the centre comes out too
faint and the outer rings too bright. When the header has a beam, ELLPROF therefore also gives the profile
deconvolved from the beam: the face-on surface densities of the rings whose model, convolved with the beam,
reproduces the map, as {doc}`BAYESIAN3DFIT <../bayesian3dfit/index>` computes them, with their errors.

In Python: {class}`bbarolo.Ellprof`.

## Example

This example measures the profile of the total HI map of
{doc}`NGC 2403 <../../getting-started/example-data>` in 41 rings of 30 arcsec. The centre, inclination
and position angle are required: they are not estimated.

::::{tab-set}
:sync-group: lang

:::{tab-item} Parameter file
:sync: par

```text
FITSFILE    ngc2403.fits
ELLPROF     true
NRADII      41
RADSEP      30
XPOS        77
YPOS        77
INC         60
PA          123.7
MASK        SEARCH
SIDE        B
```

```bash
BBarolo -p ngc2403_ellprof.par
```
:::

:::{tab-item} Python
:sync: py

```python
from bbarolo import Ellprof

f = Ellprof("ngc2403.fits", nradii=41, radsep=30, xpos=77, ypos=77, inc=60, pa=123.7,
            mask="SEARCH", side="B")
res = f.run()

res.profile             # the profile: an astropy Table
res.profile_deconvolved # the profile deconvolved from the beam, with its errors
res.map                 # the map that was used (total intensity)
res.write()             # the files of the command line, in output/ngc2403/
```
:::
::::

The run takes a few seconds, most of them for the errors of the deconvolved profile. The map is the total
intensity map in the mask (Jy/beam km/s), and the profile covers the whole disk, out to 1215 arcsec. The mass
surface density is about 7 M☉/pc² in the centre, has a shoulder at about 200 arcsec and falls to
0.2 M☉/pc² in the last ring. NGC 2403 is about 40 beams across, so out to 850 arcsec the deconvolved
profile differs from the observed one by less than 8%: the beam smooths only the structure on its own scale
(59 arcsec), and the deconvolution makes the dip at 105 arcsec deeper and the peak at 225 arcsec higher. The
faint outer rings differ more (25% in the last one, at the edge of the emission). The errors from the noise
are below 1% inside 600 arcsec. For a disk a few beams across the difference is large: on a simulated disk
the observed profile is 30% too low in the centre, and the deconvolved one within 2% of the true one.

```{figure} /figures/ellprof_profile.png
:width: 75%
:alt: The HI mass surface density of NGC 2403 against radius

The face-on HI mass surface density of NGC 2403, observed (blue, `res.profile["himassdens"]`) and
deconvolved from the beam (red, `res.profile_deconvolved["himassdens"]`, with its errors). The band is the
scatter of the pixels in each ring (the column STDDEV, scaled to the same units), not the error of the
mean.
```

## Outputs

The files are written in the output folder ({ref}`OUTFOLDER <par-all-outfolder>`, by default
`output/<object>/`) with the prefix `<object>`, the OBJECT of the FITS header. In Python, `run()` keeps
them in memory as attributes of the {class}`~bbarolo.Result`, and `res.write()` writes the files:

| File | `Result` | Content |
|---|---|---|
| `_ellprof.txt` | `profile` | the profile, one row per ring |
| `_ellprof_deconvolved.txt` | `profile_deconvolved` | the profile deconvolved from the beam (maps and cubes with a beam) |
| `_ellprof_map.fits` | `map` | the map that was used |
| `_mask.fits` | `mask` | the mask used to make the map from a cube |
| `_params.par` | | all the parameters of the run, defaults included |

The `Result` also has `ellprof`, the engine that made the profile.

The profile has these columns. Here $u$ is the unit of the map (Jy/beam km/s for the cube of this
example), and the HI columns are given only for maps in Jy:

| Column | Content | Unit |
|---|---|---|
| RADIUS | radius of the ring | arcsec |
| SUM, MEAN, MEDIAN, STDDEV, MAD | statistics of the map in the ring | $u$ |
| NPIX | number of valid pixels | pixels |
| SDENS, ERR_SD, SDENS_FO | surface density on the sky, its scatter, and face-on | $u$/arcsec² |
| NPIXBL | number of blank pixels | pixels |
| SDENSBL, SDENSBL_FO | surface density counting blank pixels as zero, on the sky and face-on | $u$/arcsec² |
| HIDENS_FO, HIMASSDENS | face-on HI column density and mass surface density | 10²⁰ cm⁻², M☉/pc² |

ERR_SD is the scatter of the pixels in the ring, not the uncertainty of the mean. The file header
gives the unit of the map and the area of a pixel.

The deconvolved profile has these columns (in the `Result`, in lower case, with the errors named
`e_sdens_fo_lo`, `e_sdens_fo_hi`, `e_himassdens_lo` and `e_himassdens_hi`):

| Column | Content | Unit |
|---|---|---|
| RADIUS | radius of the ring | arcsec |
| SDENS_FO, E_LO, E_HI | face-on surface density deconvolved from the beam, and its lower and upper 1σ errors | $u$/arcsec² |
| SDENS_FO_OBS | the observed face-on surface density, SDENS_FO of `_ellprof.txt` | $u$/arcsec² |
| DENS | the densities of the model, as DENS of GALMOD (for a 2D map, the face-on surface density per pixel) | |
| HIMASSDENS, E_LO, E_HI, HIMASSDENS_OBS | the same as HI mass surface densities, for maps in Jy | M☉/pc² |

The errors come from the noise, with the rings fixed: they do not include the uncertainty of the centre,
the inclination and the position angle.

## More

- {doc}`parameters`: all the parameters of ELLPROF.
- {doc}`algorithm`: the map, the rings, the pixels in a ring, the sides, the statistics, the
  surface densities and the deconvolved profile.
- {doc}`../2dfit/index`: the fit of the geometry of the rings (inclination and position angle) from the
  velocity field.

```{toctree}
:hidden:
:maxdepth: 1

parameters
algorithm
```
