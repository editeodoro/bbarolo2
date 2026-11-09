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

The classes are listed in {doc}`../api/index`. {class}`~bbarolo.Bayesian3DFIT` (BAYESIAN3DFIT) works
a little differently: see [below](#bayesian3dfit).

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
   galfit         Galfit
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
- The engine of the task (`res.galfit`, `res.diskmodel`, ...) gives access to the rest.
- `res["model"]` and `"model" in res` work as for a dict.

A Result also makes quick plots, each a matplotlib Figure: `res.plot_rings()` the ring parameters against
radius, with their errors (`plot_rings(["vrot", "inc"])` for some of them), and, for the models of 3DFIT and
GALMOD, `res.plot_maps()` the moment maps of the data and of the model and `res.plot_pv()` the
position–velocity slices along the major and the minor axis.

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

## Bayesian3DFIT

{class}`bbarolo.Bayesian3DFIT` (BAYESIAN3DFIT) takes the cube and the parameters as the other tasks,
but it keeps its results itself: there is no Result. `f.run(freepar=..., ...)` samples the posterior and
writes nothing. Then `f.print_stats()`, `f.outri` (the best-fit rings), `f.ring_errors()`,
`f.corner_plot()` and `f.plot_profiles()` give the results in memory, and
`f.write(outfolder=None, plots=None)` writes what the BAYESIAN3DFIT task writes: the best model with its
maps and slices, the rings with their errors, the samples and the plots. See
{doc}`../tasks/bayesian3dfit/index`.

## BayesianWind

{class}`bbarolo.BayesianWind` (BAYESIANWIND) works in the same way for bi-conical winds: `f.run(freepar=...)`
samples the posterior and writes nothing. In memory are `f.print_stats()`, `f.best` (the wind at the medians),
`f.model` (the best model cube), `f.emission` (the emissivities of its cells), `f.shells_table()`,
`f.emission_errors()` and `f.corner_plot()`. `f.write(outfolder=None, plots=None)` writes what the task writes.
`f.log_likelihood(theta)` evaluates the marginal likelihood of a vector of the free parameters. See
{doc}`../tasks/bayesianwind/index`.
