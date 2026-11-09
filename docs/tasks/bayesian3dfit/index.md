# BAYESIAN3DFIT

BAYESIAN3DFIT fits the same 3D tilted-ring model as {doc}`3DFIT <../3dfit/index>`, but it does not look for
one best fit. It samples the posterior probability distribution of the free parameters of all the rings
together with nested sampling, and gives the median and the 68% credible interval of each parameter, with
the correlations between them. Use it when the uncertainties matter, for example on data of low
signal-to-noise ratio or with few resolution elements, or when a parameter such as the inclination is shared
by all the rings and should be constrained by the whole disk. It also allows priors on the parameters. It is
slower than 3DFIT (minutes to hours instead of seconds), and it needs the package
[dynesty](https://dynesty.readthedocs.io) or [nautilus](https://nautilus-sampler.readthedocs.io).

In Python: {class}`bbarolo.Bayesian3DFIT`.

## Example

This example fits the rotation velocity of four rings of 120 arcsec of
{doc}`NGC 2403 <../../getting-started/example-data>`, one velocity dispersion for all the rings and the
noise per voxel ({ref}`FREE <par-bayesian3dfit-free>`: 6 parameters). The centre, the systemic velocity, the
inclination, the position angle and the thickness stay at the given values. The priors of VROT and VDISP are
uniform, between 50 and 250 km/s and between 0 and 30 km/s ({ref}`PRIORS <par-bayesian3dfit-priors>`; the
default ones follow the initial values: VROT from 0 to twice the initial VROT, VDISP from 0 to 50 km/s).
The sampler is dynesty ({ref}`SAMPLER <par-bayesian3dfit-sampler>`; the default is nautilus) with 100 live
points ({ref}`SAMPLEROPTS <par-bayesian3dfit-sampleropts>`) and
stops when the remaining evidence is below
{ref}`RUNOPTS <par-bayesian3dfit-runopts>` `dlogz=1`. SIGMA is free because a tilted-ring model does not
reproduce NGC 2403 down to the noise: with SIGMA the credible intervals include this misfit, without it they
would be formal errors of a few 0.01 km/s.

::::{tab-set}
:sync-group: lang

:::{tab-item} Parameter file
:sync: par

```text
FITSFILE        ngc2403.fits
BAYESIAN3DFIT   true
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
from bbarolo import Bayesian3DFIT

f = Bayesian3DFIT("ngc2403.fits", radii=[120, 240, 360, 480], xpos=77, ypos=77, vsys=132.8,
                  vrot=100, vdisp=10, inc=60, pa=123.7, z0=10, mask="SEARCH")
f.priors["vrot"] = dict(name="uniform", loc=50, scale=200)      # km/s
f.priors["vdisp"] = dict(name="uniform", loc=0, scale=30)       # km/s
f.run(freepar="VROT VDISP_SINGLE SIGMA", threads=4, method="dynesty",
      sampler_kwargs=dict(nlive=100, seed=1), run_kwargs=dict(dlogz=1))

f.print_stats()           # the median and the 68% interval of each free parameter
f.corner_plot()           # the figure below
f.plot_profiles()         # the rings against radius, with their errors
f.write()                 # the files of BAYESIAN3DFIT, in output/ngc2403/
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

## How it differs from 3DFIT

- 3DFIT fits the rings one at a time with a downhill simplex and finds one solution; BAYESIAN3DFIT builds
  one model of the whole disk, fits all the free parameters together, and returns their distribution.
- The errors come from the posterior, so they include the correlations between parameters (e.g. INC and VROT).
- Parameters can be shared by all the rings (`NAME_SINGLE`) or described by a function of radius
  (`NAME_FUNC`), and any of them can have a prior (by default, uniform around the initial values).
- There is no second stage and no normalisation of the model by NORM: the density profile is computed from
  the data, or fitted.

On simulated disks, the medians were more accurate than those of 3DFIT, and the credible intervals were
conservative (the true value is inside the 68% interval more often than 68% of the time): see
{doc}`algorithm` for the tests and the known biases (VDISP is biased low).

## Outputs

The files are written in the output folder ({ref}`OUTFOLDER <par-all-outfolder>`, by default
`output/<object>/`) with the prefix `<object>`, the OBJECT of the FITS header. The class
{class}`~bbarolo.Bayesian3DFIT` does not return a {class}`~bbarolo.Result`: `run()` writes nothing, and
the results are attributes and methods of the object. `f.write()` writes all the files of the task (with
{ref}`PLOTS <par-all-plots>`, or `write(plots=...)`), `f.write_bestmodel()` only the best model with its
outputs, and `f.save_results(file)` only the samples.

| File | Python | Content |
|---|---|---|
| `_rings_final1.txt` | `f.outri`, `f.ring_errors()` | the best-fit rings, with the lower and upper errors of the free parameters |
| `_mod_none.fits` | `write_bestmodel()` | the best model cube (the median of the posterior) |
| `_densprof.txt` | `write_bestmodel()` | the observed radial profile of the data in the best-fit rings |
| `_densprof_deconvolved.txt` | `f.dens_profile` | the density profile used by the model (DENS not free), deconvolved from the total intensity map, with its errors |
| `_<sampler>_results.npy` | `save_results(file)`, `load_results(file)` | the samples and their weights, the free parameters, the rings, the priors and the options |
| `maps/`, `pvs/` | `write_bestmodel()` | moment maps and position–velocity slices of the data and of the model |
| `_corner.pdf` | `corner_plot()` | the corner plot (in Python it returns the figure) |
| `_profiles.pdf` | `plot_profiles()` | the parameters of the rings against radius, with their errors (in Python it returns the figure) |
| `_maps_none.pdf`, `_parameters.pdf`, `plotscripts/` | `write_bestmodel()` | the plots of 3DFIT: moment maps, parameters, and the scripts that make them |
| `_params.par` | | all the parameters of the run, defaults included (parameter files only) |

The rings file has the columns of the 3DFIT rings file (radius in arcsec, VROT and VDISP in km/s, INC and PA
in degrees, Z0 in arcsec, DENS, XPOS and YPOS in pixels, VSYS and VRAD in km/s), followed by the lower
(negative) and upper errors of the free parameters, `E_VROT1`, `E_VROT2`, `E_DISP1`, ... A parameter free in
each ring has its own errors in each ring, and one shared by all the rings (`_SINGLE`) has the same errors in
all of them.

## More

- {doc}`parameters`: all the parameters of BAYESIAN3DFIT.
- {doc}`algorithm`: the likelihood, the noise, the priors, the density profile, the samplers, the outputs
  and the tests on simulated disks.
- {doc}`../3dfit/index`: the same model, fitted ring by ring.

```{toctree}
:hidden:
:maxdepth: 1

parameters
algorithm
```
