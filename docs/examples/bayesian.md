# A Bayesian fit of NGC 2403

The page of the task, {doc}`BAYESIAN3DFIT <../tasks/bayesian3dfit/index>`, fits only the rotation velocity
and the velocity dispersion. Here the geometry is free too: the inclination and the position angle are
shared by all the rings and are sampled with the other parameters, together with the edge of the disk and
the noise. The fit gives the rotation curve with its credible intervals, which we compare with the
{doc}`3DFIT <../tasks/3dfit/index>` of the same rings, and we save the result and read it back.

The fits of this page were run on the cube of {doc}`NGC 2403 <../getting-started/example-data>` with four
processes. Times are wall-clock times on that machine; they depend on it.

## The fit

### Rings and free parameters

The rings must cover the whole disk (the fit below finds gas out to at least 900 arcsec): with rings that stop at 480 arcsec,
the code warns that 23% of the flux of the total intensity map lies beyond the last ring and that the model
cannot reproduce it. We use three rings of 300 arcsec, at 300, 600 and 900 arcsec, which is coarse (the
beam is 59 arcsec) but keeps the fit short: the time grows with the number of rings and of free parameters,
and with the number of voxels in the region of the likelihood, which grows with the radius of the last ring.

The free parameters ({ref}`FREE <par-bayesian3dfit-free>`) are 8:

| Name | Number | Meaning |
|---|---|---|
| `VROT` | 3 | the rotation velocity of each ring (km/s) |
| `VDISP_SINGLE` | 1 | one velocity dispersion for all the rings (km/s) |
| `INC_SINGLE` | 1 | one inclination for all the rings (degrees) |
| `PA_SINGLE` | 1 | one position angle for all the rings (degrees) |
| `SIGMA` | 1 | the noise per voxel (Jy/beam), see below |
| `RADMAX` | 1 | the outer edge of the disk (arcsec), see below |

The centre, VSYS and Z0 stay at the given values. Z0 is 10 arcsec and the model is a Gaussian layer.
`SIGMA` is free because a tilted-ring model does not reproduce NGC 2403 down to the noise: the credible intervals
then include the misfit of the model. `FRACERR` (a fractional error of the model) is the other way to do it.

### Priors

The default priors follow the initial values (see {ref}`PRIORS <par-bayesian3dfit-priors>`); here we set the
ones that matter by hand, as uniform distributions of `scipy.stats` (`loc` is the lower limit, `scale` the
width): VROT between 50 and 250 km/s, VDISP between 0 and 30 km/s, INC between 40° and 65°, PA between 110°
and 140°. The priors of SIGMA and RADMAX are the default ones: SIGMA is log-uniform between 0.5 and 1000
times the measured noise of the cube (0.00047 Jy/beam), and RADMAX is uniform from one beam
below the last radius to twice the last radius, kept within the rings (up to the outer edge of the last ring,
as this example gives its rings and they are not extended). A prior that is too tight cuts the posterior: look at the
corner plot, and widen the prior if the posterior piles up against one of its limits.

### Sampler

The sampler is [nautilus](https://nautilus-sampler.readthedocs.io), with 500 live points. Its run stops
when the effective sample size of the posterior reaches `n_eff`, which we set to 3000 (the default, 10000,
would take a long time here). The dynesty alternative is discussed at the end of the page.

::::{tab-set}
:sync-group: lang

:::{tab-item} Parameter file
:sync: par

```text
FITSFILE        ngc2403.fits
BAYESIAN3DFIT   true
RADII           300 600 900
XPOS            77
YPOS            77
VSYS            132.8
VROT            100
VDISP           10
INC             55
PA              123.7
Z0              10
FREE            VROT VDISP_SINGLE INC_SINGLE PA_SINGLE SIGMA RADMAX
PRIORS          VROT=(uniform,loc=50,scale=200) VDISP=(uniform,loc=0,scale=30) INC=(uniform,loc=40,scale=25) PA=(uniform,loc=110,scale=30)
SAMPLER         NAUTILUS
SAMPLEROPTS     nlive=500 seed=1
RUNOPTS         n_eff=3000
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
from bbarolo import BayesianBBarolo

f = BayesianBBarolo("ngc2403.fits", radii=[300, 600, 900], xpos=77, ypos=77, vsys=132.8,
                    vrot=100, vdisp=10, inc=55, pa=123.7, z0=10, mask="SEARCH")
f.priors["vrot"] = dict(name="uniform", loc=50, scale=200)      # km/s
f.priors["vdisp"] = dict(name="uniform", loc=0, scale=30)       # km/s
f.priors["inc"] = dict(name="uniform", loc=40, scale=25)        # degrees
f.priors["phi"] = dict(name="uniform", loc=110, scale=30)       # degrees (phi is PA)
f.run(freepar="VROT VDISP_SINGLE INC_SINGLE PA_SINGLE SIGMA RADMAX", threads=4, method="nautilus",
      sampler_kwargs=dict(nlive=500, seed=1), run_kwargs=dict(n_eff=3000))
f.print_stats()
```
:::
::::

The fit took 11 minutes (655 seconds). In Python the parameter PA is called `phi` in `priors` and in the printed
statistics.

### Result

`print_stats()` prints the median and the 68% interval of each parameter. The result of this run:

| Parameter | Median | 68% interval |
|---|---|---|
| VROT, ring 1 (300 arcsec) | 118.14 km/s | −0.35, +0.36 |
| VROT, ring 2 (600 arcsec) | 130.23 km/s | −0.45, +0.44 |
| VROT, ring 3 (900 arcsec) | 132.82 km/s | −0.95, +0.90 |
| VDISP, all rings | 12.86 km/s | −0.11, +0.12 |
| INC | 59.11° | −0.21, +0.21 |
| PA | 125.14° | −0.09, +0.11 |
| SIGMA | 0.0117 Jy/beam | −0.0001, +0.0001 |
| RADMAX | 958.95 arcsec | −0.61, +0.28 |

SIGMA is 25 times the noise of the cube: it is the rms of the residuals, so it measures how far the model is
from the data, and the reduced χ² of the best fit is 1.00 by construction. The errors are therefore the
statistical errors of a model that is known to be imperfect, scaled to its misfit. Do not read 0.2° on INC
as the accuracy of the inclination of NGC 2403: it is the precision of this model on these data. In
particular the warp of the outer disk and the structure of the gas are not in the model.

VDISP is biased low in the tests on simulated disks (by 0.4 to 1.6 km/s at signal-to-noise ratios of 20 to
40; see {doc}`../tasks/bayesian3dfit/algorithm`): the interval above is smaller than that bias. The same is
true of the other intervals: they do not contain the systematic errors.

## The posterior of the geometry

The corner plot shows the one- and two-dimensional marginal posteriors. We pass the samples of the
four parameters that are not rotation velocities (`f.samples_equal` has one column per free parameter, in the
order of `f.freepar_names`):

```python
names = list(f.freepar_names)                   # ['vrot1', 'vrot2', 'vrot3', 'vdisp', 'inc', 'phi', 'sigma', 'radmax']
cols = [names.index(k) for k in ("inc", "phi", "radmax", "vdisp")]
fig = f.corner_plot(samples=f.samples_equal[:, cols], labels=["INC", "PA", "RADMAX", "VDISP"])
```

```{figure} /figures/bayesian_corner.png
:width: 75%
:alt: Corner plot of the inclination, position angle, edge and velocity dispersion of NGC 2403

Marginal posteriors of INC (degrees), PA (degrees), RADMAX (arcsec) and VDISP (km/s). The dashed lines are the
16th, 50th and 84th percentiles.
```

INC and PA are not correlated with each other or with VDISP here. The posterior of RADMAX is cut at the
right: its upper limit (959 arcsec) is the outer edge of the last ring, beyond which the edge has no effect.
The data want gas out to the end of the last ring or beyond it, so the edge of the disk is not measured: it
is at least 959 arcsec. To measure it, put rings beyond it.

### RADMAX: the outer edge of the disk

The free parameter RADMAX is the radius at which the gas of the model ends. Without it, the model has gas
out to the last ring and nothing beyond, whatever the data. The behaviour is chosen by the option `radmax`
of {ref}`LIKEOPTS <par-bayesian3dfit-likeopts>`:

- `edge` (the default): the rings stay where they are and the density of the model is multiplied by a
  weight that goes from 1 to 0 across less than one pixel at the edge, so the likelihood is continuous in RADMAX;
- `taper`: a Gaussian edge, with a width `radmax_width`;
- `ring`: RADMAX is the radius of the last ring, as in the original Bayesian module of pyBBarolo.

With `edge`, a ring beyond the edge has no gas in the model: its values are extrapolations, not measurements.
`f.in_disk` says which rings are inside the edge and `f.disk_edge` gives it (here all three are inside, and
the edge is 959 arcsec). Rings beyond it have INDISK = 0 in the rings file. See
{doc}`../tasks/bayesian3dfit/algorithm` for the formulas.

## The rotation curve and 3DFIT

The credible intervals of the rotation curve come from `f.ring_errors("vrot")` (the lower and upper errors,
positive, one value per ring), and `f.outri.r["vrot"]` has the best-fit rotation curve. The figure compares it
with 3DFIT on the same rings and the same initial values:

```python
from bbarolo import FitMod3D

rings = dict(radii=[300, 600, 900], xpos=77, ypos=77, vsys=132.8, vrot=100, vdisp=10, inc=55, pa=123.7,
             z0=10, mask="SEARCH")
fit_a = FitMod3D("ngc2403.fits", **rings, free="VROT VDISP INC PA").run(threads=4)
inc, pa = f.outri.r["inc"][0], f.outri.r["phi"][0]       # the posterior medians
fit_b = FitMod3D("ngc2403.fits", **{**rings, "inc": inc, "pa": pa}, free="VROT VDISP").run(threads=4)

lo, hi = f.ring_errors("vrot")
```

```{figure} /figures/bayesian_vrot.png
:width: 70%
:alt: Rotation curve of NGC 2403 from BAYESIAN3DFIT with credible intervals, and from 3DFIT

Rotation velocity of the three rings: median and 68% interval of BAYESIAN3DFIT (circles), and 3DFIT with its
own inclination and position angle (squares) and with those of the posterior medians (triangles). The
thin lines are 150 curves drawn from the posterior. The dotted line is the edge of the disk (RADMAX).
```

The numbers, in km/s:

| Ring (arcsec) | BAYESIAN3DFIT | 3DFIT, INC and PA free | 3DFIT at the posterior INC, PA |
|---|---|---|---|
| 300 | 118.1 ± 0.4 | 122.5 | 120.0 |
| 600 | 130.2 ± 0.4 | 130.4 | 128.2 |
| 900 | 132.8 ± 0.9 | 138.5 | 135.3 |

3DFIT, with INC and PA free, found INC = 56.5° and PA = 124.2° (the posterior has 59.1° and 125.1°), and the
rotation curve rises more steeply in the outer rings. Fixing 3DFIT at the posterior geometry brings its
rotation curve closer, as expected, because the rotation velocity scales as 1/sin INC, but the two codes
still differ by 2 to 3 km/s, several times the intervals of the posterior. The two methods fit the data in
different ways (3DFIT fits each ring separately, BAYESIAN3DFIT the whole disk with one likelihood), and the
intervals of the posterior do not include that difference. The scatter between the two methods is the
better measure of the systematic uncertainty of the curve: about 3 km/s, 2% of VROT.

## Saving and reading the results

`save_results` writes the samples, the weights, the free parameters, the rings, the priors and the options
of the likelihood in one file. A new object reads them with `load_results`, with no sampling: it repeats
the setup, recomputes the best-fit rings and the density profile, and gives back all the outputs. The cube must be
given to the new object, as it is not saved. With nautilus, say it in `sampler`:

```python
f.save_results("bayes_results.npy")

g = BayesianBBarolo("ngc2403.fits", outfolder="output/ngc2403/")
g.load_results("bayes_results.npy", sampler="nautilus")      # about 6 seconds
g.print_stats()                                               # the same statistics
g.ring_errors("vrot"), g.in_disk, g.disk_edge                 # and the other outputs
g.write()                                                     # the files of the task, see below
```

A saved result can be plotted again, or written out, on another day or machine. `write()` writes the files of
BAYESIAN3DFIT in the output folder (`output/ngc2403/` by default): the best-fit rings with the errors
(`ngc2403_rings_final1.txt`), the best model cube and its maps and position–velocity slices, the deconvolved
density profile, the samples, and, with `plots=1`, the corner plot and the profiles. The page of the
{doc}`task <../tasks/bayesian3dfit/index>` lists the files.

## dynesty or nautilus

The sampler is chosen by `method` (`SAMPLER` in a parameter file). Both are nested samplers.

- **dynesty** stops when the remaining evidence is below `dlogz` (1 in the examples of the
  task page).
- **nautilus** (the default) learns the shape of the posterior with neural networks. It needs more live points (500 here),
  and it stops when the effective sample size reaches `n_eff`, which is the main control of its run time.

`nlive` (`n_live` in nautilus), `dlogz` and `seed` have the same name in both. As a test, on a different set of
rings (160, 320 and 480 arcsec, so the model was missing the outer disk and INC was low, 47°; not shown
here), the same 8 parameters with the same priors took 506 seconds with dynesty (`nlive=100`, `dlogz=1`) and
323 seconds with nautilus (`nlive=500`, `dlogz=1`, default `n_eff`). The medians agreed to within 0.1 of
the interval (VROT of ring 1: 108.04 and 108.09 km/s), and the effective sample sizes were 467 and 10021. These
runs shared the machine with other jobs, so the times are indicative only, and one run of each is not a
benchmark. If a fit takes too long, reduce the number of rings, shrink the priors to what the data need, and
free SIGMA and RADMAX only when you need them.

## More

- {doc}`../tasks/bayesian3dfit/index` and {doc}`../tasks/bayesian3dfit/parameters`: the task and all its parameters.
- {doc}`../tasks/bayesian3dfit/algorithm`: the likelihood, the priors, RADMAX, the samplers, and the tests on
  simulated disks, with their known biases.
- {doc}`../tasks/3dfit/index`: the same model, fitted ring by ring.
