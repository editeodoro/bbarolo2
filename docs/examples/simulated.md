# Simulated datacubes

A simulated cube, made with known parameters, is the way to learn what a fit can and cannot recover. You choose
the rings, build the cube with the beam and the noise of a real observation, fit it as you would fit data, and
compare the fit with the input. This page does it with {class}`bbarolo.SimulatedGalaxyCube`.

## Making a cube

`SimulatedGalaxyCube` needs a template: a cube whose grid (and beam) are used, or just the grid. Here the
template is an empty cube of 64 × 64 pixels of 20 arcsec and 48 channels of 10 km/s, with a beam of
60 arcsec. `define_galaxy` sets the rings (the radii are ring centres, in arcsec; one value or one value per
ring for the other parameters), and `run` builds the model with {doc}`GALMOD <../tasks/galmod/index>`, smooths it
to the beam, and adds Gaussian noise, also smoothed to the beam. The model is made in memory: nothing is
written.

```python
import numpy as np
from bbarolo import SimulatedGalaxyCube

radii = np.arange(15, 360, 30.)                       # 12 rings of 30 arcsec (arcsec)
vrot = 200. * 2 / np.pi * np.arctan(radii / 60.)      # rotation curve (km/s)

s = SimulatedGalaxyCube(axisDim=[64, 64, 48], cdelts=[-20 / 3600., 20 / 3600., 10],
                        beam=60 / 3600., bunit="JY/BEAM", obj="sim")
s.define_galaxy(radii=radii, xpos=31.5, ypos=31.5, vsys=0, vrot=vrot, vdisp=10., inc=60., pa=40.,
                dens=np.exp(-radii / 200.), z0=10.,
                warpinc=False, warppa=False, flare=False, radmotion=False)

s.run(smooth=True, noise=0, threads=4, galmodtype="ANALYTIC")        # without noise, to find the peak
peak = s.model_nonoise.data.max()
s.run(smooth=True, noise=peak / 20, seed=1, threads=4, galmodtype="ANALYTIC")
s.model                                  # the cube with noise: an astropy PrimaryHDU
s.write_model("sim.fits", params=True)   # sim.fits and sim_params.txt (the input parameters)
```

The arguments to know:

- `cdelts` are the pixel sizes: the first (RA) is negative, in degrees, and the third the channel width in
  km/s. `axisDim` is `[nx, ny, nchannels]`.
- A parameter left out of `define_galaxy` is *random*: the inclination, the position angle, the rotation
  curve, the dispersion, the scale height and the density profile are drawn. Always give all you want to know.
- `dens` is the face-on surface density in the units of GALMOD, so the peak of the cube is not known in advance:
  build the model without noise, read its maximum, and set the noise from the signal-to-noise ratio you want
  (here a peak S/N of 20, an rms of 0.52 Jy/beam, in these arbitrary units).
- `noise` is the rms of the noise in the units of the cube (`True` draws a random one), and `seed` makes it
  reproducible. Other GALMOD parameters (here `galmodtype="ANALYTIC"`, a model without Monte Carlo noise)
  go to `run` as keywords.
- `s.model_nonoise` is the cube without noise, useful as the truth.

:::{note}
`define_galaxy` also draws random warps and flares. When `inc` or `pa` is a single number, `warpinc` and
`warppa` (a linear warp in the outer disk) are decided at random, unless they are `True` or `False`; the same
holds for `flare` (the scale height), for `radmotion` (a radial velocity that grows outwards) and, when `vrot`
is not given, for `vrotdwarf`. A galaxy built with only `inc=60` therefore has a different geometry in each run.
This is useful to make random test galaxies, but not to simulate the galaxy described by the arguments: pass
`warpinc=False, warppa=False, flare=False, radmotion=False`, as above. With these options and a `seed`, the
simulated cube is identical in each run (checked with `np.array_equal` on `s.model.data`).
:::

A simulation of a galaxy that looks like a given cube is made the same way with that cube as the template
(`SimulatedGalaxyCube("ngc2403.fits")`); and `GalMod(...)` with {ref}`NOISERMS <par-galmod-noiserms>` does the
same without the helper, on the grid of any cube (see {doc}`../user-guide/python`).
A complete script is `examples/simulate_galaxies.py` in the repository.

## Fitting it back

The cube is fitted as data: here with the rings known, a first guess for VROT, VDISP, INC and PA that is
deliberately wrong, and INC and PA free in an interval around the guess (±15° and ±20°).

```python
from bbarolo import FitMod3D

f = FitMod3D(s.model, radii=radii, xpos=31.5, ypos=31.5, vsys=0, z0=10, vrot=150, vdisp=15,
             inc=55, pa=35, free="VROT VDISP INC PA", deltainc=15, deltapa=20,
             mask="SEARCH", norm="LOCAL", flagerrors=True)
res = f.run(threads=4)

res.rings["radii", "vrot", "vdisp", "inc", "pa"]
res.plot_pv()
```

A cube in memory (an HDU) can be given instead of a file name. The fit takes about one second (a few seconds more with the errors). The fit is reproducible: the same cube and parameters give the same rings. The second stage fixed INC to 59.4° and PA to 40.4° (true values: 60° and 40°). The
position–velocity diagrams of data and model are the first check:

```{figure} /figures/simulated_pv.png
:alt: Position-velocity diagrams of the simulated galaxy, data and model

The simulated cube (grey) and the fit (red contours) along the major and the minor axis.
```

## Fitted and true rotation curve

The figure compares the fit with the input. The grey lines are the fits of 20 other noise realisations of
the same galaxy (seeds 2 to 21; about 1 s per fit), to see the scatter that the noise alone gives. The error
bars are those of FLAGERRORS.

```{figure} /figures/simulated_rc.png
:alt: Fitted and true rotation velocity and velocity dispersion of a simulated galaxy

The fitted rotation velocity and dispersion of the simulated galaxy (blue, with the errors) against the true
values (red). Grey: 20 fits of other noise realisations.
```

With these 20 fits, the median difference between the fitted and the true VROT is:

| Ring (arcsec) | 15 | 45 | 75 | 105 | 135 | 165 | 195 | 225 | 255 | 285 | 315 | 345 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Median error (%) | 64 | −7.3 | −5.7 | −3.7 | −2.7 | −1.9 | −1.4 | −1.3 | −1.0 | −1.4 | −1.1 | −1.1 |
| Scatter (%) | 8.1 | 5.1 | 2.3 | 1.0 | 1.1 | 0.8 | 0.6 | 0.5 | 0.4 | 0.8 | 1.3 | 1.5 |

(The scatter is the standard deviation over the 20 fits, in per cent of the true value.) Beyond the second
ring, VROT is recovered within 6% on average, and within about 2% beyond 165 arcsec, where the bias is
−1 to −2% and the scatter is 0.4 to 1.5%. The first rings and the dispersion are the weak points:

- The first ring is at 15 arcsec, a quarter of the beam, where the true rotation velocity is only 31 km/s and
  the beam smears the velocity gradient: the fit is too high (the median is 64% above the true value). The
  second ring is 7% too low. Rings narrower than the beam are not independent, and the first ones should be
  read with care.
- The dispersion is higher than the true 10 km/s in the inner rings (median 19.6, 23.3 and 18.5 km/s in the
  first three) and lower in the outer ones (6.2 km/s at 315 arcsec, and 1.5 km/s in the last ring, where
  the density of the disk is low: at low S/N the mask cuts the wings of the lines, which biases the
  dispersion low). It is the hardest parameter: see the accuracy suite.

The fits are reproducible: running the same cubes again gives the same numbers.

## The accuracy suite

The repository has a systematic version of this test, in `performance/`: `run_suite.py` simulates galaxies
with GALMOD (ANALYTIC models on an empty cube, noise smoothed to the beam as above) and fits them with
`FitMod3D`, varying the resolution (beams across the disk, channel width), the signal-to-noise ratio, the
inclination and position angle, and warps; `analyse.py` makes the figures and tables of the report in
`performance/report/`, and `bayes_coverage.py` tests the errors of BAYESIAN3DFIT. The main findings, from 3304
fits, are in {doc}`../tasks/3dfit/algorithm` (section *Accuracy and known biases*): VROT is robust (median error between −2% and 0% at five beams or more across the disk
and a peak S/N of 10 or more), VDISP is hard (20–40% scatter, and a bias that follows the S/N), and the
inclination stays close to its starting value.

## Ideas

- Change `noise` to see how the errors of VROT and VDISP grow with the S/N, and `inc` to see the loss of
  accuracy at low inclination.
- Add a warp or a radial motion: pass arrays to `define_galaxy` (`inc`, `pa`, `vrad`).
- Simulate with the grid and the beam of your own cube (`SimulatedGalaxyCube("mycube.fits")`) before fitting
  the real data, to know what the fit can do at your resolution.
