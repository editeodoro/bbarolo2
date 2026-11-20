# ALESS 073.1: a Bayesian fit of [CII] at z = 4.76

ALESS 073.1 is a dusty starburst galaxy at redshift 4.76, 1.2 billion years after the Big Bang, observed in the
[CII] 158 μm line with ALMA at a resolution of 0.1 arcsec (0.7 kpc). Lelli et al. (2021) found that its gas is a
regularly rotating disk with a rotation curve that rises steeply and then declines, the signature of a massive
stellar bulge. The disk is almost face-on, so the rotation velocity depends strongly on the inclination. The example
fits it with {doc}`3DFIT <../tasks/3dfit/index>` and `METHOD BAYESIAN`: the posterior of the rotation curve, the
dispersion and the geometry, with a prior on the inclination from that paper.

## The data

| | |
|---|---|
| Object | ALESS 073.1 (LESS J033229.3−275619), a submillimetre galaxy at z = 4.7555 |
| Data | [CII] 158 μm line (rest frequency 1900.54 GHz, observed at 330.2 GHz), ALMA band 7, programmes 2017.1.01471.S and 2011.0.00124.S combined (Lelli et al. 2021) |
| Cube | 82 × 80 pixels of 0.03 arcsec, 63 channels of 14.2 km/s, Hanning-smoothed; velocities relative to z = 4.7555 |
| Beam | 0.129 × 0.105 arcsec (0.85 × 0.69 kpc) |
| Files | [ALESS73_CII.fits](https://github.com/editeodoro/bbarolo2/raw/main/examples/ALESS73_CII.fits) (1.7 MB), [ALESS73_3DFIT_BAY.par](https://github.com/editeodoro/bbarolo2/raw/main/examples/ALESS73_3DFIT_BAY.par) |

## Running the example

::::{tab-set}
:sync-group: lang

:::{tab-item} Parameter file
:sync: par

```{literalinclude} ../../examples/ALESS73_3DFIT_BAY.par
:language: text
```

```bash
BBarolo -p ALESS73_3DFIT_BAY.par
```
:::

:::{tab-item} Python
:sync: py

```python
from bbarolo import FitMod3D

f = FitMod3D("ALESS73_CII.fits", method="bayesian", specres=0.715, radii="0.03~0.5:0.06",
             free="vrot vdisp xpos_single ypos_single vsys_single inc_single phi_single fracerr",
             priors="vrot=(uniform,0,600) vdisp=(norm,50,30) inc=(norm,25,2)")
res = f.run(threads=16)     # the sampling, in memory
res.rings                   # the median rings, with the 68% intervals
res.samples                 # the posterior samples (equal weights)
res.fitter.print_stats()    # the free parameters, medians and intervals
res.fitter.corner_plot()    # the corner plot (a matplotlib Figure)
res.write()                 # the files of the command line, in output/LESS_J033229.4-275619/
```
:::
::::

On our server, with 16 cores (16 processes for the sampler), the run takes **8 minutes 8 seconds**, of which 7.6
minutes are the sampling (203 000 likelihoods). The sampler is random, so the numbers below change slightly from run
to run.

## The setup

- **Spectral resolution.** The native channels of the ALMA data are 3.9 MHz wide, 3.55 km/s (1.875 GHz in 480
  channels at 329.6 GHz); the data were averaged to channels of 14.2 km/s, four native channels, and then
  Hanning-smoothed, which gives a velocity resolution of about 28 km/s, twice the channel width. The averaging is
  already in the model, which integrates the lines over each channel; the Hanning smoothing is not:
  {ref}`SPECRES <par-3dfit-specres>` is 0.715 channels, 10 km/s (case 10 of
  {doc}`../user-guide/spectral-resolution`, with $N = 4$).
- **Rings.** {ref}`RADII <par-3dfit-radii>` `0.03~0.5:0.06` gives 8 rings of 0.06 arcsec, half the beam, centred at
  0.03 to 0.45 arcsec (0.2 to 3.0 kpc), as in Lelli et al. (2021), who used 9.
- **Initial values.** Only the rings are given: BBarolo estimates the centre, the systemic velocity, the inclination
  (24°), the position angle (42°) and the rotation velocity (150 km/s) from the data (they are marked `(e)` in the
  log). Lelli et al. (2021) fixed the position angle at 40°.
- **Free parameters.** VROT and VDISP in each ring; one centre, systemic velocity, inclination and position angle
  for the whole disk (`_single`); and FRACERR, a fractional error of the model: 22 parameters
  ({ref}`FREE <par-3dfit-free>`). Z0 is fixed at its default; in a disk this close to face-on the thickness hardly
  changes the model (0.03 arcsec, 0.2 kpc; Lelli et al. used 0.3 kpc).
- **Priors.** {ref}`PRIORS <par-3dfit-priors>` sets three of them:
  - VROT uniform between 0 and 600 km/s. The default, from 0 to twice the estimate (300 km/s), would cut the
    rotation velocities of 300–400 km/s found by Lelli et al. (2021).
  - VDISP a Gaussian of mean 50 and dispersion 30 km/s (negative values have zero likelihood), which favours a few
    tens of km/s.
  - INC a Gaussian of mean 25° and dispersion 2°, from the inclination of 25° ± 3° that Lelli et al. (2021)
    measured with BBarolo. At low inclination the kinematics hardly constrain it, and VROT scales as
    1/sin *i*: without a prior VROT and INC are degenerate.

  The others are the defaults: the centre within ±2 beams of the estimate, VSYS ±5 channels, PA ±15°, FRACERR
  log-uniform between 0.001 and 1.

## Results

| Parameter | Median and 68% interval |
|---|---|
| VROT, rings 1 to 8 | 395 (−171, +129), 390 (−145, +131), 421 (−119, +110), 413 (−85, +96), 408 (−108, +104), 378 (−110, +110), 251 (−97, +106), 275 (−46, +45) km/s |
| VDISP, rings 1 to 8 | 62 (−28, +29), 85 (−28, +27), 90 (−31, +26), 55 (−19, +28), 66 (−20, +22), 57 (−17, +19), 62 (−14, +16), 37 (−8, +9) km/s |
| INC | 21.3° (−1.4, +1.5) |
| PA | 44.0° ± 1.7° |
| Centre | 03h32m29.295s, −27°56′19.56″ (XPOS 45.7, YPOS 40.3 pixels, ± 0.2 pixels) |
| VSYS | 13.2 ± 3.0 km/s (relative to z = 4.7555) |
| FRACERR | 0.36 ± 0.04 |

The rotation velocity is 380–420 km/s out to 0.33 arcsec (2.2 kpc) and 250–280 km/s in the last two rings: the curve
rises within the first ring and then declines, as in Lelli et al. (2021). The intervals are wide, ±100 km/s in the
inner rings, and most of their width does not come from the inclination: the projected velocity VROT sin *i* has
almost the same relative interval (149 ± 32 km/s in the fourth ring, against 413 ± 91 km/s for VROT). The rings
are half a beam wide, and the velocities of neighbouring rings are anti-correlated (correlation coefficients −0.5
to −0.6 in the corner plot): the data constrain their average better than each one.

The inclination, 21.3°, is below the mean of its prior (25°): the data prefer a lower value, close to the 22° of the
best mass model of Lelli et al. (2021), in which the inclination was free. The position angle is 4° from their 40°,
and the centre is 0.04 arcsec from their kinematic centre (03h32m29.295s, −27°56′19.60″).

The dispersion is 55–90 km/s within 0.4 arcsec and 37 ± 8 km/s in the last ring, so VROT/VDISP is between 4 and 8.
Lelli et al. (2021) find 50–60 km/s in the centre, decreasing to about 10 km/s in the outer parts, and a ratio of
about 10 within 2.5 kpc. Their fit is the classic one, with the geometry fixed (INC 25°, PA 40°) and the surface
density of each ring from the azimuthal average of the intensity map; here the geometry is free and the density
profile is deconvolved. In the inner rings the intervals of VDISP (±20–30 km/s) are almost as wide as the prior
(30 km/s): the prior has a part in the result there.

The reduced χ² is 1.69 without the model error and 1.27 with it. FRACERR, 36% of the model, takes the
non-axisymmetric structures of the gas, which the axisymmetric model does not have: arm-like features in the
residual map of the total intensity, also found by Lelli et al. (2021).

```{figure} /figures/examples/aless73_parameters.png
:alt: Rotation velocity, dispersion, geometry and surface density of the rings of ALESS 073.1

The rings against radius (`_parameters.pdf`): the medians and the 68% intervals.
```

```{figure} /figures/examples/aless73_maps.png
:width: 70%
:alt: Moment maps of the data, the model and the residuals of ALESS 073.1

The moment maps (`_maps_none.pdf`) of the data, of the model and their difference. The dispersion of the data
(bottom row) is measured within the mask, which cuts the wings of the faint lines, while the model is not masked:
the dispersion of the data is the lower one, and the residuals are mostly negative.
```

```{figure} /figures/examples/aless73_pv.png
:alt: Position-velocity diagrams of ALESS 073.1 along the major and minor axes, data and model

Position–velocity diagrams along the major (top) and the minor axis (bottom) (`_pv_none.pdf`): data in grey
with blue contours, model in red, and the projected rotation velocity of the rings (yellow).
```

```{figure} /figures/examples/aless73_pvslices.png
:alt: Position-velocity slices of ALESS 073.1 at ten position angles and parallel to the axes

Position–velocity slices through the centre at ten angles, and parallel to the axes (`_pvslices_none.pdf`).
```

```{figure} /figures/examples/aless73_chanmaps.png
:width: 70%
:alt: Channel maps of ALESS 073.1, data and model

Channel maps of the data (blue) and of the model (red) (`_chanmaps_none.pdf`).
```

```{figure} /figures/examples/aless73_corner.png
:target: ../_static/corner/aless73_corner.pdf
:alt: Corner plot of the posterior of the fit of ALESS 073.1

The corner plot of the posterior (`_corner.pdf`): the one- and two-dimensional marginal distributions of the 22
free parameters, with the 16th, 50th and 84th percentiles.
Click on the figure for the PDF file, whose labels can be read by zooming in.
```

## References

- Lelli, F., Di Teodoro, E. M., Fraternali, F., et al. 2021, Science, 371, 713:
  [A massive stellar bulge in a regularly rotating galaxy 1.2 billion years after the Big Bang](https://ui.adsabs.harvard.edu/abs/2021Sci...371..713L)
  (the data and the fit with BBarolo; [arXiv:2102.05957](https://arxiv.org/abs/2102.05957)).
- De Breuck, C., et al. 2014, A&A, 565, A59:
  [ALMA resolves turbulent, rotating [CII] emission in a young starburst galaxy at z = 4.8](https://ui.adsabs.harvard.edu/abs/2014A%26A...565A..59D)
  (the first ALMA observations of [CII], at 0.5 arcsec).
- {doc}`../tasks/3dfit/likelihood`: the likelihood, the priors, FRACERR and the samplers.
