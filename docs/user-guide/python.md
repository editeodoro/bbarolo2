# The Python interface

Every task of BBarolo is a Python class, with the same parameters as the parameter files, and runs the
same code as the command line. A task computes in memory: `run()` returns its outputs as astropy objects
and writes nothing, and `write()` writes the files of the command line when they are wanted.

```python
import numpy as np
from bbarolo import FitMod3D

f = FitMod3D("ngc2403.fits", radii=np.arange(15, 1200, 30), xpos=77, ypos=77, vsys=132.8,
             vrot=120, vdisp=8, inc=60, pa=123.7, z0=10, free="VROT VDISP", mask="SEARCH")
res = f.run(threads=4)       # in memory: nothing is written
res.rings                    # the fitted rings: an astropy Table
res.write()                  # the files of the command line, in output/ngc2403/
```

The classes are listed in {doc}`../api/index`. The fits (3DFIT, WINDFIT, SLITFIT) have three
methods, chosen by `method=`: see [below](#the-fits-and-their-methods). The engines behind them, for custom
likelihoods and interactive work, are described [at the end](#the-engines-bayesian3dfit-bayesianwind-bayesianslitfit).

## The cube and the parameters

The first argument is the cube: the name of a FITS file, an astropy HDU (or an HDUList, of which the first
HDU with data is used), or a tuple `(data, header)`. `beam=(bmaj, bmin, bpa)`, in arcsec and degrees (or `beam=60`, or a string such as
`beam='0.01d 0.01d 30'`), is the keyword {ref}`BEAM <par-all-beam>` and replaces the beam of the header.

The other arguments are the parameters: the keywords of the parameter files, case-insensitive (`vrot` or
`VROT`), or their aliases. A ring parameter is one value for all the rings, or one value per ring (a list
or a numpy array). A wrong value is an error, with a message that names the parameter. The class sets the
flag of its task; the flag of another task is an error (use its class, or {func}`bbarolo.run` for several
tasks), except `contsub=True`, which subtracts the continuum first.

- `task.update(**params)` changes parameters (`None` restores the default) and returns the task;
  `run(**params)` changes them too, before running.
- `task.get(name)` is the value of a parameter, as set or the default; `task.parameters` is a dict of the
  parameters that are set.
- `task.to_parfile("ngc2403.par")` writes the parameters as a parameter file, and
  `FitMod3D.from_parfile("ngc2403.par")` reads one (its FITSFILE is the cube; the flags of other tasks
  are ignored).
- `task.help()`, or `bbarolo.help("3DFIT")`, prints the parameters of the task, and `bbarolo.help("free")`
  one parameter.

## Running: the Result

`res = task.run()` runs the task in memory and returns a {class}`~bbarolo.Result` (also kept as
`task.result`). Its attributes depend on the task: the section *Outputs* of the page of each task lists
them, and `res.keys()` gives their names. `print(res)` shows them with their types, e.g. for 3DFIT:

```text
<Result of 3DFIT: in memory, not written
   rings          Table (41 rows: radii, vrot, vdisp, inc, pa, z0, ...)
   rings_stage1   Table (41 rows: radii, vrot, vdisp, inc, pa, z0, ...)
   rings_initial  Table (41 rows: radii, vrot, vdisp, inc, pa, z0, ...)
   rotation_curve Table (41 rows: radii, vrot)
   model          HDU 62x150x150
   densprof       Table (computed when used)
   densprof_deconvolved  Table (the density profile deconvolved from the beam)
   maps           dict of HDUs (computed when used)
   pvs            dict of HDUs (computed when used)
   fitter         Galfit
   mask           HDU 62x150x150
   data           HDU 62x150x150 (computed when used)
   residual       HDU 62x150x150 (computed when used)>
```

- Cubes and maps are astropy HDUs, with the headers of the files that `write()` writes; several maps are
  a dict of HDUs (`res.maps["0mom"]`).
- Tables are astropy Tables at full precision, with the names of the columns in lower case. The ring
  tables have all the rings: those of 3DFIT have a column `fitok` (False where the fit of the ring failed),
  and the 2DFIT table has NaN in the rings not fitted. The ring files have 3 decimals and leave out the
  failed rings (unless {ref}`BADOUT <par-3dfit-badout>`).
- Some outputs are computed when first used: the moment maps and position–velocity slices of a model,
  the cube that was analysed (`data`) and the residuals.
- The engine of the task (`res.fitter` of the fits, `res.diskmodel` of GALMOD, ...) gives access to the rest.
- `res["model"]` and `"model" in res` work as for a dict.

A Result also makes plots, each a matplotlib Figure, with nothing written. For 3DFIT (every method) these
are the plots of the command line ({ref}`the PDF files <outputs-plots>`), drawn from the outputs in memory:

```python
res.plot_parameters()             # <prefix>_parameters.pdf: the rings against radius, with their errors
res.plot_chanmaps()               # <prefix>_chanmaps_<norm>.pdf: channel maps of data and model
res.plot_pvs()                    # <prefix>_pv_<norm>.pdf: PV slices along the major and minor axes
res.plot_pvslices()               # <prefix>_pvslices_<norm>.pdf: PV slices at several angles and offsets
res.plot_kinmaps()                # <prefix>_maps_<norm>.pdf: moment maps of data, model and residuals
res.plot_asymdrift()              # <prefix>_asymmetricdrift.pdf (ADRIFT)
res.plot_chanmaps(model="local")  # another model of res.models
res.write_allplots()              # all of them as PDF files (the names of the command line), nothing else
```

They are the functions of `bbarolo.plots` (`plot_parameters`, `plot_chanmaps`, ...), which draw a plot from
arrays, and the plotting scripts written with the PDF files contain the same functions, to edit and run again.
Each figure keeps the style of BBarolo (it does not change the rcParams of the session). For any task with
rings or a model, `res.plot_rings()` is a quick plot of the ring parameters against radius, with their errors
(`plot_rings(["vrot", "inc"])` for some of them), and, for the models of 3DFIT and GALMOD, `res.plot_maps()`
shows the moment maps of the data and of the model and `res.plot_pv()` the position–velocity slices along the
major and the minor axis.

## Writing: write()

```python
files = res.write()                    # in OUTFOLDER, or output/<object>/
files = res.write("fits/ngc2403/")     # in another folder
files = res.write(plots=False)         # without the plots
f.write()                              # the same, for the last run of the task
```

`res.write(outfolder=None, plots=None)` writes the files of the task, exactly those of the command line
with the same parameters, and `<prefix>_params.par`, a parameter file with all the parameters of the run,
defaults included, which runs the same task from the command line. It returns the list of the files
written; afterwards `res.files` lists all the files written and `res.outfolder` is the folder. The folder is
OUTFOLDER if it was given, otherwise `output/<object>/` in the current folder, `<object>` being the OBJECT
of the FITS header; a folder that already exists is used as it is, and its files of the same names are
overwritten (with a warning). The plots, PDF files and the scripts that make them, follow
{ref}`PLOTS <par-all-plots>` (by default the plots of the task), and the argument `plots` changes it:
`True`, `False`, or a value of PLOTS.

A cube given in memory (an HDU or a tuple), or a cube with a new `beam`, is written as
`<object>_input.fits` in the same folder: the plots read the data from it, and `<prefix>_params.par` refers
to it.

## Models: GalMod and GalMod3D

`GalMod(...).run()` builds the model in memory: `res.model` is the model cube, normalised as
{ref}`NORM <par-galmod-norm>`, and `res.diskmodel` the model engine (a {class}`bbarolo.DiskModel`), which
builds the models of other rings on the same grid. With {ref}`NOISERMS <par-galmod-noiserms>` the model
with noise is `res.models["noise"]`, and `run(seed=...)` sets the seed of the noise.

```python
from bbarolo import GalMod

g = GalMod("ngc2403.fits", nradii=41, radsep=30, xpos=77, ypos=77, vsys=132.8, vrot=120,
           vdisp=8, inc=60, pa=123.7, z0=10, dens=1, norm="NONE")
model = g.run(threads=4).model                                 # an astropy PrimaryHDU
noisy = g.run(noiserms=0.001, seed=1).models["noise"]           # the same model, with noise
```

{class}`bbarolo.GalMod3D` builds a model cube from 3D fields of the density and the velocity of the gas;
it is not a task, and `GalMod3D(...).model()` returns the cube (see {doc}`../tasks/galmod3d/index`).

## Parameter files: bbarolo.run

`bbarolo.run("ngc2403.par", threads=8)` runs a parameter file with all its tasks in the same Python
process, and writes the outputs, as the command line does. Keyword arguments change the parameters of the
file, or replace the file: `bbarolo.run(fitsfile="ngc2403.fits", search=True, totalmap=True)`. It returns
a dict `{task: Result}` (a list of them when {ref}`FITSLIST <par-all-fitslist>` gives several cubes), with
the outputs in memory and the files written in `res.files`.

## The fits and their methods

The three fits, {class}`~bbarolo.FitMod3D` (3DFIT), {class}`~bbarolo.WindFit` (WINDFIT) and
{class}`~bbarolo.SlitFit` (SLITFIT), are tasks like the others, and `method=` is the keyword
{ref}`METHOD <par-3dfit-method>` (so are `sampler=` and `minimiser=`): `"classic"` (3DFIT only, and its default:
the rings fitted one at a time), `"global"` (all the parameters together, at the maximum of the likelihood, with
the errors of the Laplace approximation; the default of WindFit and SlitFit) or `"bayesian"` (nested sampling of
the posterior). GLOBAL and BAYESIAN share `free`, `priors`, `likeopts`, `sampleropts` and `runopts`, so changing
between them is one word. `run(**params)` changes any parameter, the method included, and returns a Result;
`write()` writes the files.

```python
from bbarolo import FitMod3D, WindFit, SlitFit

f = FitMod3D("ngc2403.fits", radii=[120, 240, 360, 480], xpos=77, ypos=77, vsys=132.8, vrot=100, vdisp=10,
             inc=60, pa=123.7, z0=10, mask="SEARCH", free="VROT VDISP_SINGLE INC_SINGLE PA_SINGLE")
res = f.run(method="global", minimiser="powell", threads=16)   # a minimisation, in seconds
res = f.run(method="bayesian", sampler="dynesty", threads=16)  # the sampling of the same likelihood
res.method                   # "BAYESIAN"
res.rings                    # the best-fit rings, with the errors (lower and upper) of the free parameters
res.samples                  # an astropy Table: one column per free parameter, equal weights
res.fitter                   # the engine (Bayesian3DFIT)
res.plot_rings()             # the plots of the Result work with every method
res.write()
```

Every Result of a fit has `method` (`"CLASSIC"`, `"GLOBAL"` or `"BAYESIAN"`) and `fitter` (the engine: `Galfit` for
CLASSIC, {class}`~bbarolo.Bayesian3DFIT` for GLOBAL and BAYESIAN; it replaces the old `res.galfit` and
`res.bayes`). The keys of the Result of 3DFIT:

- all the methods: `rings` (with the errors), `rotation_curve`, `rings_initial`, `model`, `densprof`, `maps`, `pvs`,
  `data`, `residual`, `mask`; `densprof_deconvolved` (with GLOBAL and BAYESIAN only when DENS is not free);
- CLASSIC only: `rings_stage1`, `models`, `asymdrift`;
- GLOBAL and BAYESIAN only: `samples` (for GLOBAL, the draws of the Laplace approximation).

The Result of {class}`~bbarolo.WindFit` has `model` (the model cube, an HDU), `parameters` (a Table with `name`,
`value`, `err_lo` and `err_hi`), `shells` (a Table), `samples` and `fitter` ({class}`~bbarolo.BayesianWind`). That of
{class}`~bbarolo.SlitFit` has `rings`, `model` (the HDU of the slit), `samples` and `fitter`
({class}`~bbarolo.BayesianSlitFit`). `bbarolo.run(parfile)` returns `{"3DFIT": Result}` for every method of 3DFIT,
and `{"WINDFIT": ...}` or `{"SLITFIT": ...}`. See {doc}`../tasks/3dfit/index`, {doc}`../tasks/windfit/index` and
{doc}`../tasks/slitfit/index`.

## The engines: Bayesian3DFIT, BayesianWind, BayesianSlitFit

Behind the methods GLOBAL and BAYESIAN are the classes {class}`bbarolo.Bayesian3DFIT`,
{class}`bbarolo.BayesianWind` and {class}`bbarolo.BayesianSlitFit`. They take the cube and the parameters as the
tasks do, but keep their results themselves: there is no Result. Use them for a custom likelihood
(`log_like=`), `run_geometry`, or interactive work on one object. Their `run(freepar=..., ...)` follows METHOD
(`method=`; the default is BAYESIAN for Bayesian3DFIT and GLOBAL for BayesianWind and BayesianSlitFit, as for their
tasks; CLASSIC is an error) and returns the object, writing nothing. The two methods are also explicit:
`f.sample(freepar=None, sampler=None, threads=None, ...)` samples the posterior (`sampler=` is one of `"nautilusbb"`,
`"nautilus"` and `"dynesty"`), and `f.optimize(freepar=None, minimiser="L-BFGS-B", ...)` finds the best fit by a
classic minimisation of the same likelihood, with the errors of the Laplace approximation. Changing only METHOD,
SAMPLER, MINIMISER, SAMPLEROPTS or RUNOPTS does not reset the rings of the engine.

```python
f = Bayesian3DFIT("ngc2403.fits", radii=[120, 240, 360, 480], vrot=100, vdisp=10, inc=60, pa=123.7, ...)
f.sample(freepar="VROT VDISP_SINGLE SIGMA", sampler="dynesty", threads=4)        # or f.run(sampler="dynesty")
f.optimize(freepar="VROT VDISP_SINGLE", minimiser="powell")      # or f.run(method="global", minimiser="powell")
```

The results of a Bayesian3DFIT are in memory: `f.print_stats()`, `f.outri` (the best-fit rings), `f.ring_errors()`,
`f.corner_plot()` and `f.plot_profiles()`, and `f.plot_parameters()`, `f.plot_chanmaps()`, `f.plot_pvs()`,
`f.plot_pvslices()` and `f.plot_kinmaps()`, the plots of the best model that the task writes (its model and maps are
computed for the first of them; `f.write_allplots()` writes them as PDF files). `f.write(outfolder=None,
plots=None)` writes what the task writes: the best model with its maps and slices, the rings with their errors, the
samples and the plots. After `optimize()`, `f.optimum` holds the details of the minimisation.

In memory, a BayesianWind has `f.print_stats()`, `f.best` (the wind at the medians), `f.model` (the best model
cube), `f.emission` (the emissivities of its cells), `f.shells_table()`, `f.emission_errors()` and `f.corner_plot()`;
`f.log_likelihood(theta)` evaluates the marginal likelihood of a vector of the free parameters. A BayesianSlitFit
has `f.print_stats()`, `f.ring_table()` (the rings with their errors, an astropy Table), `f.best` (the rings, the
model of the slit and, with NORM LOCAL, the amplitudes of the positions), `f.prep` (the prepared slit: data, noise
map, continuum, the fits of the lines along the slit), `f.plot_pv()`, `f.plot_parameters()` and `f.corner_plot()`.
Both have `f.write(outfolder=None, plots=None)`.

Migrating from the earlier interface: `Bayesian3DFIT(...).run(method="dynesty")` is now
`.sample(sampler="dynesty")` (or `.run(sampler="dynesty")`); `.run(method="optimise",
sampler_kwargs=dict(method="powell"))` is `.optimize(minimiser="powell")` or `.run(method="global",
minimiser="powell")`; `BayesianWind(...).run()` now minimises, so use `.sample()` or `method="bayesian"` to sample;
and the parameter files with `BAYESIAN3DFIT true`, `SAMPLER OPTIMISE` and `SAMPLEROPTS method=powell` are now
`3DFIT true`, `METHOD GLOBAL` and `MINIMISER powell`.
