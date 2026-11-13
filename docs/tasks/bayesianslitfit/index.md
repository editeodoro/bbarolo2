# BAYESIANSLITFIT

BAYESIANSLITFIT fits the kinematics of a rotating disk to a **long-slit spectrum**: a 2D image of the position
along the slit and the spectral axis (wavelength, frequency or velocity). It samples the posterior distribution of
the rotation curve, the velocity dispersion (and the radial motions) in rings, the systemic velocity and the centre
along the slit with nested sampling (the nautilus included in BBarolo, `SAMPLER NAUTILUSBB`, the default; nautilus
or dynesty), or finds the best fit by a minimisation of the same likelihood (`SAMPLER OPTIMISE`).

The model is a GALMOD disk, as in 3DFIT and BAYESIAN3DFIT, built with the pixels and the channels of the slit and
seen through the slit: the light of the disk smoothed by the seeing that falls in the slit, with the lines
broadened by the instrumental resolution. Several lines (e.g. Hα and the [NII] doublet) are modelled together at
the same velocities. A slit does not constrain the geometry of the disk: the inclination, the angle between the
slit and the major axis and the distance of the slit from the centre are given, not fitted.

In Python: {class}`bbarolo.BayesianSlitFit`. The default sampler needs nothing more than BBarolo; `SAMPLER NAUTILUS`
and `DYNESTY` need their packages (`pip install "bbarolo[bayesian]"`).

## Example

A SALT long-slit spectrum of a galaxy at z = 0.09143, with Hα and the [NII] doublet and a strong continuum, and
residuals of the sky subtraction on the lines. The seeing is 1.5 arcsec, the slit 1 arcsec wide; the axis is in
vacuum wavelengths (in metres, read in Angstrom), so the rest wavelengths are those in vacuum. The ratio
[NII]6583/Hα, about 0.35 in the disk, is fixed by `RELINT` (6548 is a third of 6583).

::::{tab-set}
:sync-group: lang

:::{tab-item} Parameter file
:sync: par

```text
FITSFILE         2masxj1022_SALT.fits
BAYESIANSLITFIT  true
SLITWIDTH        1
BEAM             1.5
SPECRES          1.04
INC              71
REDSHIFT         0.09143
RESTWAVE         6564.61 6549.86 6585.27
RELINT           1 0.117 0.35
CONTSUB          true
SKYRESID         true
NORM             LOCAL
FREE             VROT VDISP VSYS XPOS
THREADS          32
```
:::

:::{tab-item} Python
:sync: py

```python
from bbarolo import BayesianSlitFit

f = BayesianSlitFit("2masxj1022_SALT.fits", slitwidth=1, beam=1.5, specres=1.04, inc=71, redshift=0.09143,
                    restwave=[6564.61, 6549.86, 6585.27], relint=[1, 0.117, 0.35], contsub=True,
                    skyresid=True, norm="LOCAL")
f.run(freepar="VROT VDISP VSYS XPOS", threads=32)
f.print_stats()
f.plot_pv()                   # the slit, the best model and the residuals, as a matplotlib Figure
f.write("output/2masxj1022/")
```
:::
::::

`SAMPLER OPTIMISE` (in Python `f.optimize()`, or `run(method='optimise')`) finds the best fit by L-BFGS-B with the
errors of the Laplace approximation, as for {doc}`BAYESIAN3DFIT <../bayesian3dfit/index>`: about 30 seconds on 32
processes for the example above (28 free parameters), a good way to check a setup before sampling.

## The input

- **The slit**: a 2D FITS image. The spectral axis is the one with a spectral `CTYPE` (`WAVE`, `AWAV`, `FREQ`,
  `VELO`, `VRAD`, `VOPT`, `FELO`) or unit, the other one the position along the slit, whose scale (`CDELT` with
  `CUNIT` in arcsec, arcmin or deg) is required. Either order of the axes works. A wavelength axis in metres is
  read in Angstrom (and `RESTWAVE`, `WAVERANGE`, `MASKWAVE` are then in Angstrom). Blank pixels
  (NaN, or 0) are not data.
- **The resolution**: `BEAM`, the FWHM of the seeing (one value: a circular Gaussian), and `SPECRES`, the dispersion
  of the line spread function in channels. Both are required: a slit header carries neither.
- **The lines**: `RESTWAVE` (or `RESTFREQ`) and `RELINT`, at `REDSHIFT`, as in 3DFIT. The velocities are relative
  to the rest wavelength of the first line at REDSHIFT, with the velocity definition of `VELDEF` (relativistic for a
  wavelength axis by default).
- **The geometry**: `INC` (required); `SLITANGLE`, the angle from the slit (increasing position) to the receding
  half of the major axis, counterclockwise (`AUTO`, the default: 0 or 180 from the velocity gradient; e.g. 10 or
  190 for a slit 10° off the major axis); `YPOS`, the distance of the slit from the centre in arcsec (default 0);
  `Z0` (default one pixel). They are fixed.

## Free parameters

`FREE` (default `VROT VDISP VSYS XPOS`):

- `VROT`, `VDISP`, `VRAD`: one value per ring, `NAME_SINGLE` one value for all the rings, `NAME_FUNC` a function of
  radius (VROT an arctangent, VDISP and VRAD lines). VRAD is only weakly constrained: along the major axis the radial
  motions are perpendicular to the line of sight, and they show only through the width of the slit, the seeing and
  a misalignment;
- `VSYS` (relative to REDSHIFT) and `XPOS` (the centre along the slit, pixels): single values;
- `DENS` (with `NORM NONE` only) and `SIGMA`, a factor of the noise map.

The rings are `RADII`, or `NRADII` and `RADSEP`; by default one every seeing FWHM out to the farthest emission
along the slit. The values not given are estimated from Gaussian fits of the lines at each position. The default
priors: VROT from 0 to twice the largest estimate, VDISP from 0 to three times, VSYS ± 150 km/s, XPOS ± two
seeing FWHM; `PRIORS` changes them. The parameters free in each ring can be regularised by smoothness priors,
as in BAYESIAN3DFIT: `LIKEOPTS smooth=(vrot=5, vdisp=3)` allows the profiles to bend by about 5 and 3 km/s over a
seeing FWHM (the keys `vrot`, `vdisp`, `vrad` and `dens`, in dex). `print_stats` shows the effective number of
independent values of each profile.

## Normalisation

`NORM` sets the flux of the model:

- `AZIM` (default): the densities of the rings whose model reproduces the flux of the lines along the slit, a
  non-negative fit with a smoothness prior (LIKEOPTS `densprof_smooth`), as BAYESIAN3DFIT deconvolves the total
  intensity map. An axisymmetric disk: when the two sides of the slit differ in flux, the fit moves the centre and
  the systemic velocity to compensate (on a mock with a 50% asymmetry: VSYS off by 12 km/s, XPOS by one pixel,
  VROT 5 km/s low);
- `LOCAL`: that model times one amplitude per position of the slit, marginalised in the likelihood (a flat,
  non-negative prior). The flux along the slit is then free, and the kinematics come from the shape of the lines:
  use it for asymmetric or clumpy disks (on the same mock it recovered all the parameters within the errors);
- `NONE`: the densities `DENS`, given or fitted.

## Preparing the data

- `WAVERANGE`: the spectral range of the fit (default: the lines over the velocity range of the emission, plus 600
  km/s on either side);
- `CONTSUB`: the continuum (as for datacubes): a polynomial of degree `CONTORDER` (default 1, a straight line)
  fitted at each position in the range of the fit away from the lines and from the channels of `EXCLWIND` (channels
  of the slit, from 0), and subtracted;
- `SKYRESID`: the median of the positions without line emission subtracted from each channel, for the residuals of
  the sky subtraction along the slit;
- `MASKWAVE`: spectral ranges left out (e.g. strong sky lines).

The noise of each pixel is measured from the data: the scatter of each channel over the positions without line
emission (larger at the sky lines) plus that of each position in the continuum windows (the Poisson noise of a
bright centre).

## Outputs

In the output folder, with the OBJECT of the slit as the prefix:

- `_slitfit_rings.txt`: the rings of the best fit, with the 1σ errors of the free parameters;
- `_slit_data.fits`, `_slit_model.fits`, `_slit_resid.fits`, `_slit_noise.fits` and (CONTSUB) `_slit_cont.fits`: the
  spectral window of the fit, with the axes of the input;
- `_slitfit_<sampler>_results.npy`: the samples and the setup of the fit;
- with `PLOTS`: `_slitfit_pv.pdf` (the data, with the velocities of the Gaussian fits of the lines; the data with
  the contours of the model and the rotation velocities of the rings where they cross the slit, with their errors;
  the residuals; all with the line-of-sight velocities of the rings and the systemic velocity),
  `_slitfit_parameters.pdf` (the rings against radius) and `_slitfit_corner.pdf`, and their scripts in
  `plotscripts/`.

```{toctree}
:hidden:

parameters
algorithm
```
