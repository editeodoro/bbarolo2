# 3DFIT

3DFIT fits a 3D tilted-ring model of a rotating disk to an emission-line datacube. The model includes
the beam, the channel width and the thickness of the disk, so the rotation curve and the velocity
dispersion it finds are corrected for beam smearing. This is the main task of BBarolo, and the one to
use for spatially resolved data: HI and CO observations of nearby galaxies, and also IFU and ALMA data
of high-redshift galaxies, with only a few resolution elements across the disk.

There are three **methods**, chosen by {ref}`METHOD <par-3dfit-method>`, which differ in how the model is
fitted to the data (the model is the same):

| METHOD | What it does | Result | Time |
|---|---|---|---|
| `CLASSIC` (the default) | the rings are fitted one at a time, minimising the residuals with a downhill simplex: the 3DFIT of BBarolo 1.x | one solution per ring, with optional errors (`FLAGERRORS`) | seconds |
| `GLOBAL` | all the parameters of all the rings are fitted together, at the maximum of the likelihood ({doc}`likelihood`), with a minimiser of scipy ({ref}`MINIMISER <par-3dfit-minimiser>`) | the best fit, with the errors of the Laplace approximation (the inverse Hessian) | seconds to minutes |
| `BAYESIAN` | the posterior probability distribution of all the parameters is sampled with nested sampling ({ref}`SAMPLER <par-3dfit-sampler>`) | the median and the 68% credible interval of each parameter, with the correlations | minutes to hours |

In Python: {class}`bbarolo.FitMod3D`, with `method="classic"`, `"global"` or `"bayesian"`.

## Choosing the method

- **CLASSIC** is the task of BBarolo 1.x: parameter files written for the C++ code run unchanged. It is the
  fastest, and the one for the usual rotation curves. Its errors (`FLAGERRORS`) come from the shape of the
  residual function of each ring, and each ring is independent of the others, so parameters that are the same
  in all the rings (the inclination) are fitted ring by ring, then regularised in the second stage
  ({doc}`algorithm`).
- **GLOBAL** fits the whole disk with one likelihood, so shared parameters (`NAME_SINGLE`) or functions of radius
  (`NAME_FUNC`) are constrained by all the data, and priors can limit them. It takes seconds where a sampling
  takes minutes, which makes it the way to test a setup before a sampling, or a quick fit. The errors assume
  that the likelihood is a Gaussian around its maximum.
- **BAYESIAN** gives the full posterior, so the uncertainties include the correlations between parameters
  and the non-Gaussian shapes. Use it when the uncertainties matter, for example on data of low
  signal-to-noise ratio or with few resolution elements. Its default sampler,
  [nautilus](https://nautilus-sampler.readthedocs.io) (`SAMPLER NAUTILUSBB`), is part of BBarolo; the package
  nautilus-sampler (`SAMPLER NAUTILUS`) and [dynesty](https://dynesty.readthedocs.io) can be used instead.

GLOBAL and BAYESIAN share FREE, PRIORS, LIKEOPTS, SAMPLEROPTS and RUNOPTS, so going from one to the other is a
one-word change of METHOD. One METHOD is used per run. 3DFIT cannot run together with WINDFIT, and the keywords
of a method that are given with another method give a warning and are ignored (`BBarolo -d 3DFIT` lists the
common parameters, then those of each method). Without a parameter file,
`BBarolo -f cube.fits METHOD=bayesian` runs the automatic 3DFIT with that method.

## Example: CLASSIC

This example fits the rotation velocity, the dispersion and the position angle of
{doc}`NGC 2403 <../../getting-started/example-data>` in 41 rings of 30 arcsec. The centre, the systemic
velocity and the inclination stay at the given values. The parameters not given are estimated from
the data.

::::{tab-set}
:sync-group: lang

:::{tab-item} Parameter file
:sync: par

```text
FITSFILE    ngc2403.fits
3DFIT       true
METHOD      CLASSIC     # the default
NRADII      41
RADSEP      30
XPOS        77
YPOS        77
VSYS        132.8
VROT        120
VDISP       8
INC         60
PA          123.7
Z0          10
FREE        VROT VDISP PA
MASK        SEARCH
NORM        LOCAL
```

```bash
BBarolo -p ngc2403_3dfit.par
```
:::

:::{tab-item} Python
:sync: py

```python
from bbarolo import FitMod3D

f = FitMod3D("ngc2403.fits", nradii=41, radsep=30, xpos=77, ypos=77, vsys=132.8, vrot=120,
             vdisp=8, inc=60, pa=123.7, z0=10, free="VROT VDISP PA", mask="SEARCH", norm="LOCAL")
res = f.run(threads=4)    # in memory: nothing is written (method="classic" is the default)

res.rings                 # the fitted rings: an astropy Table
res.plot_rings()          # the figure below
res.plot_pv()
res.plot_chanmaps()       # the plots of the PDF files, in memory (also plot_parameters, plot_pvs, ...)
res.write()               # the files of the command line, in output/ngc2403/
```
:::
::::

The fit takes about 10 seconds on four cores. With the default {ref}`TWOSTAGE <par-3dfit-twostage>`,
the position angle is fitted ring by ring, then regularised and fixed (here to its median, 124°, as it
varies little with radius: see {ref}`REGTYPE <par-3dfit-regtype>`), and the velocities are fitted
again:

```{figure} /figures/3dfit_rings.png
:width: 75%
:alt: The rotation velocity and the velocity dispersion of NGC 2403 against radius

The rotation curve and the velocity dispersion of NGC 2403 from the second stage of the fit
(`res.plot_rings()`).
```

The position–velocity diagrams along the major and minor axes are the quickest check of a fit: the
model (red contours) should follow the data (grey):

```{figure} /figures/3dfit_pv.png
:alt: Position-velocity diagrams of NGC 2403 along the major and minor axes, data and model

Position–velocity slices of the data and of the model along the major and the minor axis
(`res.plot_pv()`).
```

## Example: BAYESIAN

This example fits the rotation velocity of four rings of 120 arcsec of
{doc}`NGC 2403 <../../getting-started/example-data>`, one velocity dispersion for all the rings and the
noise per voxel ({ref}`FREE <par-3dfit-free>`: 6 parameters). The centre, the systemic velocity, the
inclination, the position angle and the thickness stay at the given values. The priors of VROT and VDISP are
uniform, between 50 and 250 km/s and between 0 and 30 km/s ({ref}`PRIORS <par-3dfit-priors>`; the
default ones follow the initial values: VROT from 0 to twice the initial VROT, VDISP from 0 to 50 km/s).
The sampler is dynesty ({ref}`SAMPLER <par-3dfit-sampler>`; the default is nautilus) with 100 live
points ({ref}`SAMPLEROPTS <par-3dfit-sampleropts>`) and
stops when the remaining evidence is below
{ref}`RUNOPTS <par-3dfit-runopts>` `dlogz=1`. SIGMA is free because a tilted-ring model does not
reproduce NGC 2403 down to the noise: with SIGMA the credible intervals include this misfit, without it they
would be formal errors of a few 0.01 km/s.

::::{tab-set}
:sync-group: lang

:::{tab-item} Parameter file
:sync: par

```text
FITSFILE        ngc2403.fits
3DFIT           true
METHOD          BAYESIAN
RADII           120 240 360 480
XPOS            77
YPOS            77
VSYS            132.8
VROT            100
VDISP           10
INC             60
PA              123.7
Z0              10
FREE            VROT VDISP_SINGLE SIGMA
PRIORS          VROT=(uniform,loc=50,scale=200) VDISP=(uniform,loc=0,scale=30)
SAMPLER         DYNESTY
SAMPLEROPTS     nlive=100 seed=1
RUNOPTS         dlogz=1
MASK            SEARCH
THREADS         4
```

```bash
BBarolo -p ngc2403_bayes.par
```
:::

:::{tab-item} Python
:sync: py

```python
from bbarolo import FitMod3D

f = FitMod3D("ngc2403.fits", radii=[120, 240, 360, 480], xpos=77, ypos=77, vsys=132.8, vrot=100, vdisp=10, inc=60,
             pa=123.7, z0=10, mask="SEARCH", method="bayesian", free="VROT VDISP_SINGLE SIGMA",
             priors="VROT=(uniform,loc=50,scale=200) VDISP=(uniform,loc=0,scale=30)",     # km/s
             sampler="dynesty", sampleropts="nlive=100 seed=1", runopts="dlogz=1")
res = f.run(threads=4)    # in memory

res.rings                 # the median rings, with the errors: an astropy Table
res.samples               # the posterior samples: one column per free parameter, equal weights
res.plot_parameters()     # the rings against radius, with their errors
res.plot_pvs()            # the plots of the PDF files, in memory (also plot_rings, plot_chanmaps, ...)
res.fitter.corner_plot()  # the figure below
res.write()               # the files of the task, in output/ngc2403/
```
:::
::::

The fit takes between 2.5 and 3 minutes on four processes (about 8 minutes of CPU time). The time grows quickly with the
number of free parameters. The result of the run shown here:

| Parameter | Median | 68% interval |
|---|---|---|
| VROT, ring 1 (120 arcsec) | 88.4 km/s | −1.7, +1.6 |
| VROT, ring 2 (240 arcsec) | 115.7 km/s | −1.1, +1.2 |
| VROT, ring 3 (360 arcsec) | 127.5 km/s | −1.1, +1.3 |
| VROT, ring 4 (480 arcsec) | 125.0 km/s | −0.9, +0.8 |
| VDISP, all rings | 12.6 km/s | −0.3, +0.2 |
| SIGMA | 0.020 Jy/beam | |

SIGMA is about 36 times the noise of the cube (0.00055 Jy/beam): it is the root mean square of the
residuals, so it measures how far the tilted-ring model is from the data, and the reduced χ² of the best
fit is 1 by construction (0.99). The corner plot shows the one- and
two-dimensional marginal posteriors, with the 15.87th, 50th and 84.13th percentiles (±1σ) as dashed lines:

```{figure} /figures/bayesian3dfit_corner.png
:width: 80%
:alt: Corner plot of the posterior of the rotation velocities, the velocity dispersion and the noise of NGC 2403

Marginal posteriors of the six free parameters (`f.corner_plot()`): the rotation velocity of each ring
(`vrot1` to `vrot4`, km/s), the velocity dispersion (`vdisp`, km/s) and the noise per voxel (`sigma`, Jy/beam).
```


## Example: GLOBAL

The same fit by minimisation takes seconds where the sampling takes minutes. `METHOD GLOBAL` finds the maximum of
the same likelihood, with the free parameters, the options of the likelihood and the priors of BAYESIAN (the ranges
of the priors bound the search). The errors are those of the Laplace approximation, the inverse of the Hessian of
the log-likelihood at the best fit, and 4000 draws of that Gaussian stand in for the posterior samples
(`res.samples`, the corner plot, the errors of the tables).

```text
3DFIT         true
METHOD        GLOBAL
MINIMISER     L-BFGS-B                         # the default; e.g. Powell, Nelder-Mead
SAMPLEROPTS   nstart=1                         # x0, nstart, seed, posterior, errors, nsamples, step
RUNOPTS       maxiter=500                      # the options of the minimiser (scipy.optimize)
```

```python
res = FitMod3D("ngc2403.fits", method="global", free="VROT VDISP_SINGLE INC_SINGLE PA_SINGLE",
               radii=[120, 240, 360, 480], vrot=100, vdisp=10, inc=60, pa=123.7, z0=10,
               xpos=77, ypos=77, vsys=132.8, mask="SEARCH").run(threads=16)
res.rings                 # the best fit, with the errors
res.fitter.optimum        # the details: log-likelihood, number of likelihoods, covariance, ...
```

The default minimiser is L-BFGS-B, a quasi-Newton method whose gradient is computed by central differences on
`threads` processes (`THREADS`); {ref}`MINIMISER <par-3dfit-minimiser>` also accepts the bounded methods of
`scipy.optimize.minimize` (`Powell`, `Nelder-Mead`, `TNC`, `SLSQP`, `trust-constr`, `COBYLA`, `COBYQA`) and
`differential_evolution`, a global search. On a mock of NGC 2403 (six rings, VROT in each, VDISP, INC and PA
shared), L-BFGS-B took 6 seconds on 16 processes and nautilus 10 minutes, for the same best fit (within 0.4σ of the
medians of the posterior) and errors within 8% of the posterior widths. See {doc}`likelihood` for the details.

## Outputs

The files are written in the output folder ({ref}`OUTFOLDER <par-all-outfolder>`, by default
`output/<object>/`) with the prefix `<object>`, the OBJECT of the FITS header. In Python, `run()` keeps
them in memory as attributes of the {class}`~bbarolo.Result`, and `res.write()` writes the files:

| File | `Result` | Content |
|---|---|---|
| `_rings_initial.txt` | `rings_initial` | the initial rings |
| `_rings_final1.txt` | `rings_stage1` | CLASSIC: the rings of the first stage, with the errors if any. GLOBAL and BAYESIAN: the best-fit rings with the errors (`rings`) |
| `_rings_final2.txt` | `rings` | CLASSIC: the rings of the second stage (TWOSTAGE); without it, `rings` is the first stage |
| `_densprof.txt` | `densprof` | the radial profile of the data in the final rings |
| `_densprof_deconvolved.txt` | `densprof_deconvolved` | the density profile deconvolved from the beam, at the geometry of the final rings; with CLASSIC its densities build the final model ({ref}`NORM <par-3dfit-norm>`); with GLOBAL and BAYESIAN it is the profile used by the model when DENS is not free, with its errors |
| `_mod_local.fits` | `model` | the model cube, normalised as the last value of NORM (`azim`, `local`, `none`) |
| `_mask.fits` | `mask` | the mask of the fit |
| `maps/` | `maps` | moment maps 0, 1 and 2 of the data and of the model |
| `pvs/` | `pvs` | position–velocity slices of the data, the model and the mask, major (`_a`) and minor (`_b`) axes |
| `_2drings.txt`, `_map_1st.fits` | | the 2D fit of the velocity field used to refine the initial position angle |
| `*.pdf`, `plotscripts/` | | the plots of the fit: parameters, moment maps, PV diagrams and channel maps, and the scripts that make them ({ref}`PLOTS <par-all-plots>`; in Python, `res.write(plots=...)`, and in memory `res.plot_parameters()`, `plot_kinmaps()`, `plot_pvs()`, `plot_pvslices()`, `plot_chanmaps()`, `plot_asymdrift()`; `res.write_allplots()` writes only the PDF files) |
| `_asymdrift.txt` | `asymdrift` | the asymmetric drift correction ({ref}`ADRIFT <par-3dfit-adrift>`) |
| `_<sampler>_results.npy` | `samples` | GLOBAL and BAYESIAN: the samples (for GLOBAL, `_optimize_results.npy`: the best fit and the draws of the Laplace approximation); `samples` is an astropy Table with one column per free parameter and equal weights |
| `_corner.pdf` | | GLOBAL and BAYESIAN: the corner plot (in Python `res.fitter.corner_plot()`) |
| `_params.par` | | all the parameters of the run, defaults included, with METHOD |

The `Result` also has `rotation_curve` (the radius and VROT, with the errors, of the rings fitted), `data`
(the cube that was fitted), `residual` (data minus model, in the mask), `method` (`"CLASSIC"`, `"GLOBAL"` or
`"BAYESIAN"`) and `fitter` (the fitting engine: `Galfit` for CLASSIC, {class}`~bbarolo.Bayesian3DFIT` for GLOBAL and
BAYESIAN). `models` and `asymdrift` exist with CLASSIC only. The plots of the Result (`plot_rings`, `plot_maps`,
`plot_pv`, `plot_parameters`, ...) work with every method.
Its ring tables have all the rings, at full precision, with a column `fitok` (False where the fit of the
ring failed); the files have 3 decimals and leave out the failed rings, unless
{ref}`BADOUT <par-3dfit-badout>`.

With GLOBAL and BAYESIAN there is no first stage and no normalisation of the model by NORM; `rings` holds the best
fit (the median of the posterior for BAYESIAN) with the lower and upper errors, `E_VROT1`, `E_VROT2`, `E_DISP1`, ...
A parameter free in each ring has its own errors in each ring, and one shared by all the rings (`_SINGLE`) has the
same errors in all of them.

The rings files have one row per ring: the radius (arcsec), VROT, VDISP, INC, PA, Z0 (arcsec), DENS,
XPOS, YPOS, VSYS and VRAD, and, with {ref}`FLAGERRORS <par-3dfit-flagerrors>`, the lower and upper errors
of the free parameters.

## More

- {doc}`parameters`: all the parameters of 3DFIT.
- {doc}`algorithm`: METHOD CLASSIC: the model, the residual function, the minimisation, the two stages, the errors,
  and the accuracy of the fits on simulated disks.
- {doc}`likelihood`: METHOD GLOBAL and BAYESIAN: the likelihood, the noise, the priors, the density profile, the
  minimiser, the samplers and the tests on simulated disks.
- {doc}`../windfit/index` and {doc}`../slitfit/index`: the same methods GLOBAL and BAYESIAN for a bi-conical wind and
  for a long-slit spectrum.

```{toctree}
:hidden:
:maxdepth: 1

parameters
algorithm
likelihood
```
