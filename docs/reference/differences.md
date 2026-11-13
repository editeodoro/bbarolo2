# Differences from BBarolo 1.8

This page is for users of the C++ BBarolo 1.8 and of its Python wrapper pyBBarolo. BBarolo 2 is a port of
the C++ code (version 1.8.1dev) and of pyBBarolo 1.8.0 to pure Python. A parameter file written for
BBarolo 1.8 runs unchanged, and the tasks give the same results, with the exceptions listed here. Each
task page of the documentation ends its page *how it works* with a section *Differences from BBarolo 1.8*:
the links are in [the last section](#differences-by-task).

## Installation and names

- Nothing is compiled. The numerical kernels are written with numba and compiled when they are first
  used (the first run of a task is slower, the later ones use a cache). See {doc}`../getting-started/installation`.
- The package is `bbarolo` (`import bbarolo`), and the command is still `BBarolo`. The package `pyBBarolo`
  does not exist any more: `import pyBBarolo` fails, unless the old C++-based package is still installed
  in the same environment. Uninstall it, and change the imports (see
  [the table below](#from-pybbarolo-to-bbarolo)).
- The command line has the same options as the C++ executable (`-p`, `-f`, `-c`, `-d`, `-t`, `-l`,
  `-v`, `-q`, `-h` and the FITS utilities). `KEY=value` after `-p file.par` overrides the file.
- The MPI executable `BBarolo_MPI` does not exist. Run several `BBarolo -p` processes instead, for
  instance with `xargs -P`. (The Bayesian fit can use MPI from Python: see
  {doc}`../user-guide/parallel`.)

## What gives the same results

Every task of 1.8 is available; SLITFIT, which did not work, is replaced by the new task
{doc}`BAYESIANSLITFIT <../tasks/bayesianslitfit/index>`. The table compares the
outputs of the Python code with those of the C++ code on the example cubes of BBarolo (numbers from
`README_cowork.md` of the repository, measured when the port was made).

| Task | Agreement with C++ 1.8 |
|---|---|
| SEARCH (catalogue, detection maps, mask) | identical |
| 2DFIT | ring table identical; model map equal to float precision |
| ELLPROF (profile) | identical (apart from the bugs fixed below) |
| Moment maps, RMS map, S/N map | equal to float precision (relative difference about $10^{-7}$) |
| SMOOTH, SMOOTHSPEC, REND3D, CONTSUB | equal to float precision |
| PVSLICE and the PV files of 3DFIT | same slice geometry; flux along the slice within 0.4% (the slices are interpolated now) |
| GALMOD | same algorithm and normalisation; same total flux; differences at the level of the Monte Carlo noise |
| GALWIND (cylinders) | total flux equal; spectra correlate at 0.99999 |
| 3DFIT, NGC 2403 | rms difference 1.0 km/s in VROT and 0.4 km/s in VDISP; PA 123.95° (Python) and 123.90° (C++) |

The Monte Carlo models cannot be identical to those of C++, because the random numbers are different. For
the other 3DFIT examples the differences are those between two runs of the C++ code with different seeds:

| Example | VROT rms, C++ against C++ (another seed) | VROT rms, C++ against Python |
|---|---:|---:|
| NGC 5633 | 5.0 km/s | 5.5 km/s |
| U3_25160 | 14.5 km/s | 10.2 km/s |
| zcos_z1_202 | 13.9 km/s | 12.2 km/s |

On simulated galaxies (3304 fits, `performance/` in the repository), ring by ring the two codes differ by
2% in VROT, 14% in VDISP and 0.9° in INC, the level of the Monte Carlo noise of the models, and their
statistics are the same. The same fit of NGC 2403 with BAYESIAN3DFIT (3 rings, 4 free parameters, dynesty
with 100 live points) gave VROT 103.8 ± 9.7 and 128.8 ± 9.2 km/s in the last two rings with the C++-based
pyBBarolo, and 103.4 ± 9.5 and 128.3 ± 9.0 km/s with the Python code of that time. A 3DFIT of NGC 2403 took 66.8 s in C++ and 22.3 s in Python, on
two cores (README_cowork.md; the models and the convolutions of the fit are the part that is faster).

Reproducibility: the random numbers of each ring of 3DFIT (used with FLAGERRORS, and when a trial value
falls outside its bounds) are now seeded by the stage and the ring. In 1.8 each thread had its own stream,
so with `THREADS` above 1 two identical runs could give different rings (about 1.4 km/s rms in VROT for
NGC 5633 with errors). The ring tables are now the same for any number of threads. The Monte Carlo noise
of the final model cube still depends on THREADS.

Header differences: the WCS values are written in double precision (C++ truncated them to float), and
the PV slices carry RESTFRQ and SPECSYS. The gnuplot scripts are not written: all the plots are made
with matplotlib, with the same scripts as in C++.

## Changed defaults and conventions

The defaults below are those of `keywords.py`. Run `BBarolo -d 3DFIT` or `bbarolo.help("ringmode")` to
see the value and the description of a parameter.

| What | BBarolo 1.8 | BBarolo 2 |
|---|---|---|
| Ring convention of the models ({ref}`RINGMODE <par-3dfit-ringmode>`) | the first and last radius are moved by RADSEP/2 | `CENTRED`: each ring is an annulus centred on its radius, with the value of the ring. `RINGMODE SHIFTED` gives the models of 1.8, `NODES` those of GIPSY's GALMOD |
| Z0 when not given ({ref}`Z0 <par-3dfit-z0>`) | 0, a disk with no thickness | one pixel, a thin disk (3DFIT, BAYESIAN3DFIT, SPACEPAR and the estimates of GUESSTYPE FIT; GALMOD needs Z0) |
| Initial estimates ({ref}`GUESSTYPE <par-3dfit-guesstype>`) | one algorithm | `LEGACY` (the default) is that algorithm; `ELLIPSE` corrects it and `FIT` is new |
| DISTANCE | converts the outputs to kpc | removed (see below) |
| THREADS | all the cores | the number of cores, at most 16 |
| VERBOSE | `true` or `false` | also 2; the default (1) prints about 140 lines for the NGC 2403 3DFIT instead of about 400 |
| Names of the output files | some without the prefix, some with it but with no underscore | every file is `<prefix>_<what>` |
| A misspelt keyword | ignored | an error if it is very close to a real one (VDSIP), otherwise a warning that proposes the closest keywords |
| BAYESIAN3DFIT and the Python class `Bayesian3DFIT`: GALMODTYPE | (pyBBarolo) MC models | `ANALYTIC`; the other tasks keep `MC` |
| BAYESIAN3DFIT: SAMPLER | (pyBBarolo) dynesty | nautilus (`SAMPLER DYNESTY` or `run(method='dynesty')` selects dynesty) |
| BAYESIAN3DFIT: dynesty | (pyBBarolo) dynamic sampler | static sampler (`SAMPLEROPTS dynamic=true` selects the dynamic one) |
| BAYESIAN3DFIT: RADMAX | (pyBBarolo) the radius of the last ring | the outer edge of the disk (`LIKEOPTS radmax=edge`; `ring` is the old meaning) |
| BAYESIAN3DFIT: priors | (pyBBarolo) fixed wide ranges | follow the initial rings (given or estimated) |

### RINGMODE

3DFIT fits every ring as an annulus centred on its radius, reaching halfway to its neighbours, and its ring
tables give those radii. The final model of 1.8 was built after moving the first radius inwards and the
last one outwards by RADSEP/2, which covers the right area but bends the profiles: with rings at 30, 90,
150 and 210 arcsec and VROT 60, 100, 120 and 130 km/s, the model of 1.8 has VROT 73 km/s at 30 arcsec and
127 km/s at 210 arcsec. The default `CENTRED` keeps the value of each ring at its radius. The ring tables
of 3DFIT do not change. The final models, model maps and PV slices of 3DFIT and GALMOD change in the first
and last ring: on the examples by 0.03% of the model flux for the NGC 2403 fit (41 rings), 0.02% for its AZIM
GALMOD model, and 0.5% for NGC 5633 (15 rings of 1.7 arcsec, where the rotation curve is steep). Models of evenly
spaced rings with constant values do not change. With `RINGMODE SHIFTED` all the outputs are those of the
previous version of the port, and the comparisons with the C++ code use it. The algorithm is described in
{doc}`../tasks/galmod/algorithm`.

### Units: arcsec only, no DISTANCE

In 1.8, DISTANCE only converted arcsec to kpc in the outputs: no fit, model or likelihood used it. It is
gone, and so is what it gave.

- DISTANCE in a parameter file is an unknown keyword: it is ignored, with a warning (checked). `distance=`
  in a Python class is an error.
- The ring files (`<prefix>_rings_final1.txt`, ...) have no `RAD(Kpc)` and `Z0(pc)` columns. The columns
  are `RAD(arcs)`, `VROT`, `DISP`, `INC`, `P.A.`, `Z0(arcs)`, `DENS`, `XPOS`, `YPOS`, `VSYS`, `VRAD`, then the
  errors. VROT is column 2, not column 3. `RINGFILE` finds the columns by their names, so it reads the
  ring files of both versions. A parameter read with `file(name,n)` from a ring file of BBarolo 2 must
  come from its own column: asking for another quantity is an error that gives the right column.
- The tables of the results (`res.rings`) have no `rad_kpc` and `z0_pc`.
- SMOOTH has no LINEAR (a resolution in kpc). The new beam is NEWBEAM or FACTOR. LINEAR is still
  the spectral resolution of the models when SPECRES is not given.

### Other changes of meaning

- **SPECRES** is new: the spectral resolution of the models (σ, in channels). LINEAR sets it when SPECRES
  is not given, as in 1.8.
- **FREEANGLE** is new. 2DFIT read the angle left out around the minor axis from DVDZ; it now reads FREEANGLE
  (default 15°), and DVDZ is no longer used by 2DFIT.
- The position angles of the beams are always angles: `NEWBEAM 150 100 -1` is −1°, and a BEAM angle of 0 is
  0°, not "the angle of the old beam" and "the angle of the header". When the BPA of NEWBEAM is not given
  the new beam has the angle of the old one.
- **CHECKCUBE** writes the cleaned cube in the output folder as `<prefix>_clean.fits`, not next to the
  input file.
- **A velocity axis without CUNIT3** is taken as m/s if its channels are wider than 100, otherwise as km/s.
- **Spatial axes in arcsec or arcmin** are converted to degrees when the cube is read, and the outputs are
  written in degrees.
- **Output headers** have no CD matrix: the WCS is written as CDELT, plus PC when the axes are rotated.
- **Relative paths** (FITSFILE, FITSLIST, OUTFOLDER) become absolute in the plotting scripts and in
  `<prefix>_params.par`.
- **`<prefix>_params.par`** is new: every run writes the parameters it used, defaults included, and
  `BBarolo -p <prefix>_params.par` repeats it.
- **The estimated ring parameters** of 3DFIT and GALMOD no longer change the parameters seen by the next
  tasks of the same run (as in C++).
- **Ring parameters** can be lists, one value per ring, like RADII.
- **Several emission lines** (RESTWAVE or RESTFREQ with RELINT): each line has its own reference frequency
  (see the bugs below).

The complete list, with the numbers measured for each change, is in {doc}`changes`.

## Removed and renamed keywords and options

All the keywords of the C++ parameter table are known to BBarolo 2 except DISTANCE (checked by comparing
the keywords read by `Arrays/param.cpp` with `bbarolo/keywords.py`). The keywords that are new in BBarolo 2
are BAYESIAN3DFIT, PRIORS, SAMPLER, SAMPLEROPTS, RUNOPTS and LIKEOPTS (see the Bayesian task), GALMODTYPE,
RINGMODE, GUESSTYPE, FREEANGLE and SPECRES.

| Keyword | What happened |
|---|---|
| DISTANCE | removed (above) |
| LINEAR of SMOOTH | removed (above); LINEAR of the models stays |
| BMAJ, BMIN, BPA, BEAMFWHM, OBMAJ, OBMIN, OBPA | replaced by {ref}`BEAM <par-all-beam>` (the beam of the data, which replaces that of the header) and {ref}`NEWBEAM <par-smooth-newbeam>` (the new beam of SMOOTH and of the SMOOTH masks); the old keywords are errors that name BEAM. In Python, `beam=` is the keyword BEAM, and `Smooth(cube, newbeam=150)` replaces `bmaj=150, bmin=150, bpa=0` |
| DVDZ of 2DFIT | no longer read; use FREEANGLE |
| SLITFIT | an error that names BAYESIANSLITFIT, the new task of the fit of long-slit spectra (SLITWIDTH is one of its keywords) |
| NORM `AZIMUTHAL` | accepted before, but used as LOCAL; it is AZIM now |
| the long forms of SIDE (`APPROACHING`, `RECEDING`, `BOTH`, `APP`, ...) | work in 3DFIT, 2DFIT, SPACEPAR and ELLPROF (1.8 ignored them) |
| `POLYN` | still accepted, as an alias of REGTYPE (as in C++) |

The keywords of the Python classes are those of the parameter files, case-insensitive, with a few aliases
(`pa` or `phi`, `errors` for FLAGERRORS, `fit3d` for 3DFIT, `polyn`, `deltaphi`).

Python-only options that existed in the old pyBBarolo and are gone or renamed: `smooth=` of FitMod3D is
`sm=`; `outfolder`, `ltype`, `deltainc`, `ftype`, `wfunc`, `bweight`, `tol`, `mask`, `norm`, `free`, `side`,
`twostage`, `startrad`, `redshift` and `restwave` keep their names.

## New tasks and features

- **BAYESIAN3DFIT** is a task of the parameter files (`BBarolo -p file.par`), with the keywords above. In
  1.8 the Bayesian fit was the Python module `pyBBarolo.bayesian`, built on the C++ library. See
  {doc}`../tasks/bayesian3dfit/index`. The likelihood, the priors, the sampler defaults and RADMAX differ
  from that module: the list is in {doc}`../tasks/bayesian3dfit/algorithm`. Requires dynesty or nautilus
  (`pip install "bbarolo[bayesian]"`).
- **GalMod3D** (Python only): model cubes from 3D fields of the density and velocity of the gas, e.g.
  thick lagging disks, bars and spiral arms. See {doc}`../tasks/galmod3d/index`.
- **GALMODTYPE** `MC` (default, the classic Monte Carlo GALMOD) or `ANALYTIC`, which has no Monte Carlo noise
  (also for GALWIND models, GalMod3D and the fits). See {doc}`../tasks/galmod/algorithm`.
- **RINGMODE**, **GUESSTYPE**, **SPECRES**, **FREEANGLE**: above.
- **Deconvolved density profile** in ELLPROF (`<prefix>_ellprof_deconvolved.txt`).
- **SIDE in BAYESIAN3DFIT**, and `INDISK` in the ring files of BAYESIAN3DFIT.
- **Errors**: problems in the input raise `bbarolo.BBaroloError` (a `ValueError`) with a message that names
  the parameter; the library never exits the Python session, and the command line exits with code 1.
  Parameters are checked before the run, e.g. unknown names in FREE and a VSYS outside the velocity range
  of the cube.
- **Python**: every task is a class; cubes can be given in memory (a FITS file, an astropy HDU or
  `(data, header)`); the outputs are astropy HDUs and Tables; and a task computes in memory and writes
  files only when asked. See {doc}`../user-guide/python`.
- **Help**: `bbarolo.help("free")` and `BBarolo -d TASK` describe each parameter; the same text is in the
  parameter tables of this documentation.

## From pyBBarolo to bbarolo

The old interface had two layers: the task classes of `pyBBarolo.pyBBarolo` (`init`, `set_options`,
`compute`) with their own defaults, and `pyBBarolo.wrapper` (`BBaroloWrapper`), which ran the C++
executable in a subprocess. They covered 7 of the 18 tasks. In BBarolo 2 every task has a class, the classes
take the cube and all the parameters in one call, `run()` computes in memory and returns a
{class}`~bbarolo.Result`, and `write()` writes the files. The same example in both:

::::{tab-set}
:sync-group: lang

:::{tab-item} pyBBarolo 1.8
:sync: old

```python
from pyBBarolo import FitMod3D

f3d = FitMod3D("ngc2403.fits")
f3d.init(radii=np.arange(15, 1200, 30), xpos=77, ypos=77, vsys=132.8, vrot=120,
         vdisp=8, vrad=0, z0=10, inc=60, phi=123.7)
f3d.set_options(mask="SEARCH", free="VROT VDISP", distance=3.2, outfolder="output/ngc2403")
bfrings, bestmod = f3d.compute(threads=4)
```
:::

:::{tab-item} BBarolo 2
:sync: py

```python
from bbarolo import FitMod3D

f3d = FitMod3D("ngc2403.fits", radii=np.arange(15, 1200, 30), xpos=77, ypos=77, vsys=132.8,
               vrot=120, vdisp=8, z0=10, inc=60, phi=123.7, mask="SEARCH", free="VROT VDISP",
               outfolder="output/ngc2403")
res = f3d.run(threads=4)           # in memory
bfrings, bestmod = res.rings, res.model
res.write()                        # the files of the command line, in outfolder
```
:::
::::

Note that `distance` is gone, and that the defaults are those of a parameter file, not those of the old
class (below). Old call and new call:

| pyBBarolo 1.8 | BBarolo 2 |
|---|---|
| `from pyBBarolo import FitMod3D` | `from bbarolo import FitMod3D` |
| `t = Task(fitsname)`, then `t.init(...)`, `t.set_options(...)` | `t = Task(fitsname, **params)`: the arguments of `init` and the options in one call; `t.update(**params)` changes them later |
| `t.compute(threads=4)` | `res = t.run(threads=4)`: returns a `Result`; nothing is written |
| (the files were written by `compute`) | `res.write()` or `t.write()`: the files of the command line; `res.write("folder/", plots=False)` |
| `t.show_options()`, `t.show_arguments()` | `t.help()`, `bbarolo.help("free")`, `t.parameters` and `t.get("free")` |
| `FitMod3D.compute` returns `(bfit, outmodel)`, an array and an HDU | `res.rings` (an astropy Table with column names, all the rings, plus `res.rings_stage1`), `res.model` (an HDU); also `res.maps`, `res.pvs`, `res.densprof`, `res.mask`, `res.rotation_curve` |
| `f3d.plot_model()` | `res.write(plots=True)` writes the plots; `res.plot_rings()`, `res.plot_maps()` and `res.plot_pv()` return matplotlib figures |
| `GalMod(fits).init(...)`; `compute()`; `smooth()` | `GalMod(fits, **params).run().model`: the model is convolved with the beam unless `sm=False` (checked: peak 24.8 against 43.0 on NGC 2403) |
| `GalWind`, `Search`, `FitMod2D`, `Ellprof` | the same names; `Search.search()` is `run()`, with `res.detections`, `res.table`, `res.cube`, `res.labels`, `res.detmap`; `FitMod2D`: `res.rings`, `res.model`, `res.velfield`; `Ellprof`: `res.profile`, `res.map` |
| `SpectralSmoothing(fits).smooth(window_type, window_size)`, `writeto()` | `SmoothSpec(fits, window_type="hanning", window_size=3).run().cube`, then `write()` |
| `Model3D.smooth()`, `set_beam(bmaj, bmin, bpa)` | `sm=` and `beam=(bmaj, bmin, bpa)` (arcsec and degrees) as arguments |
| `Param`, `Rings`, `FitsCube` of pyBBarolo | `bbarolo.Param`, `bbarolo.Rings`, `bbarolo.Cube` (not the same classes) |
| `t.outmodel` with `norm`, files read back from `rings_final1.txt` | in memory: `res.rings` has all the rings and a column `fitok` (False where the fit failed), at full precision; the text files have 3 decimals and leave out the failed rings unless BADOUT |
| `pyBBarolo.wrapper.BBaroloWrapper(params).run()`, `.write_parameterfile()` | `bbarolo.run("file.par", **params)` or `bbarolo.run(fitsfile=..., search=True)`; `task.to_parfile("file.par")`; `bbarolo.read_parfile` reads a file. Python runs in the same process and returns `{task: Result}`; `exe="/path/to/BBarolo"` runs another executable (e.g. the C++ one) in a subprocess |
| `wrapper.GenericTask` classes (`PVSlice`, `MakeMask`, `SpatialSmoothing`, ...) | `bbarolo.PVSlice`, `MakeMask`, `Smooth`, `Maps`, `SpacePar`, `Rend3D`, `ContSub`, `Stats`, all with `run()` |
| `pyBBarolo.utils`: `SimulatedGalaxyCube`, `emptyfits`, `build_path`, `extract_pv_path`, `extract_pv` | `bbarolo.SimulatedGalaxyCube` (builds the model in memory, with `run()`, then `write_model(fileout)`; no executable), `bbarolo.emptyfits`; `bbarolo.pv_path`, `pv_sample`, `pv_extract` for the PV slices |
| `pyBBarolo.bayesian.BayesianBBarolo`: `init`, `set_options`, `compute`, `compute_geometry`, `write_bestmodel`, `save_results`, `load_results` | `bbarolo.Bayesian3DFIT(cube, **params)`; `run(freepar=..., method=...)`, `run_geometry`, `write()`; `save_results`, `load_results`, `print_stats`, `corner_plot` and `plot_profiles` are kept. `run()` writes nothing. `useBBres=True` is `like_kwargs={"bbres": True}`. The class also runs from a parameter file: `Bayesian3DFIT.from_parfile` |

Defaults of the old `FitMod3D` class, which are not those of a parameter file and have changed to the
parameter-file ones: LTYPE 2 (now 1), FREE `VROT VDISP` (now `VROT VDISP INC PA`), MASK `SMOOTH` (now
`SEARCH`), REGTYPE `bezier` (now `auto`), `z0=0` (now one pixel). Add them to the call to repeat an old
run. The new defaults are checked with `bbarolo.help("ltype")` and similar calls.

The old module `BayesianBBarolo_test` and the C++ function `Galfit::getModel_BBB` are not ported.

## C++ bugs fixed

The bugs below are in the C++ code (and in pyBBarolo where stated); they are fixed in BBarolo 2, so the
results of the affected tasks differ from those of 1.8. The entries of {doc}`changes` marked **(C++)** are
the bugs of 1.8; the others were introduced by the port and fixed before this release. The table is a
summary: {doc}`changes` has the details and the size of each change.

| Task | Bug in 1.8 |
|---|---|
| 3DFIT | `FREE ALL` with TWOSTAGE: DENS free in the second stage but not counted (degenerate simplex, misaligned error columns) |
| 3DFIT | The PA refinement read beyond the end of the 2D-fit arrays; the outermost ring could get a random PA |
| 3DFIT | The second stage regularised all the rings, also those not fitted; the rings that failed kept their DENS in the final model; ADRIFT used them too |
| 3DFIT | VDISP trial values out of [MINVDISP, MAXVDISP] were redrawn across the other bound |
| 3DFIT | REVERSE: the beam-wide rings took the initial values of the ring with the same index, not interpolated ones |
| 3DFIT | A fixed VROT was replaced by that of the 2D fit when PA was free |
| 3DFIT | RADSEP without NRADII: the extent of the rings scaled with RADSEP |
| 3DFIT, GALMOD, SPACEPAR | `NORM AZIMUTHAL` was used as LOCAL; a NORM with two values made GALMOD write no model |
| 3DFIT, BAYESIAN3DFIT | MC models in a box were not crops of the model of the whole cube (the residuals jumped by the Monte Carlo noise) |
| 3DFIT | `writeModel` switched FLUXCONVERT off and never restored it |
| GALMOD | `DENSFLUX false` was ignored (the task went through the 3DFIT class, which sets it) |
| SPACEPAR | The PA and VROT of the given rings were replaced by those of a 2D fit; the models were always normalised LOCAL; FTYPE and WFUNC out of range gave zero residuals |
| SPACEPAR | With NORMALCUBE the noise was divided by the scale factor once per pixel when the cube was restored |
| GALWIND | Spherical shells (WTYPE 1) read DENS from VWIND and VROT from OPENANG; the flux differs by DENS/VWIND |
| GALWIND | Cylinders: a VROT column was read from DENS; ISEED never reached the cylinders (all shared the same random numbers); `VWIND > 0` was an inflow for the cylinders and an outflow for the shells; the smoothed model was multiplied by the beam area; the layers used a Gaussian profile, not boxes; list values used only their first element |
| 2DFIT | The model velocity field: radius of the pixels between the rings computed with the geometry of the ring of the previous pixel; RADII cut to the estimated number of rings; a ring not fitted stopped the run |
| ELLPROF | The ring-overlap correction was applied to blank pixels only; SIDE A and R swapped for most galaxies; blank pixels of a 2D map read as 0 |
| Maps | `derv_gauss` for MAPTYPE GAUSSIAN used $e^{-x^2}$ instead of $e^{-x^2/2}$; with several lines (RESTWAVE, RESTFREQ) the maps mixed all the lines (now: the first line only, and a joint Gaussian fit with MAPTYPE GAUSSIAN); the HI mass constant (8794 against 8846) differed between MASSDENSMAP and ELLPROF; MASSDENSMAP and RMSMAP used a different mask; RMSMAP of spectra with fewer than 5 channels was 1E10 |
| SMOOTH | A beam with major axis smaller than the minor one was rotated by 90°; cubes in mJy/beam, uJy/beam and W.U. were not rescaled; a cube without a beam was not smoothed |
| SMOOTHSPEC | A window of one channel (HANNING2, BARTLETT, WELCH, BLACKMAN, FLATTOP) gave a cube of NaN |
| SEARCH, MAKEMASK | ITERNOISE always ran 20 iterations; the checks of SEARCHTYPE and of the growth threshold ran only with SEARCH true; the HISTORY card GROWTHTHRESHOLD had the wrong value; with SNMAP the S/N maps overwrote those of TOTALMAP |
| Statistics | min and max initialised with `array[0]` even when that pixel was masked |
| REND3D, REDUCE | CRPIX3 = 0 gave each plane an angle one degree too large; CRPIX of a spatially reduced cube was rounded |
| CONTSUB | EXCLWIND items that were not channels were ignored, and a channel outside the cube wrote outside the array |
| CHECKCUBE | The thresholds of the row and column tests were swapped |
| FITS utilities | `--fitscopy` of a reversed axis flipped only CDELT; `--jybeam2k` and `--k2jybeam` failed with `file[ext]`; mJy/beam taken as Jy/beam; K converted again |
| All tasks, models | Spatial axes in arcsec or arcmin gave a wrong beam area, ring width and noise correlation; spectral axes in kHz, MHz, GHz or nm, `CUNIT3 = kms` and radio velocities on a wavelength axis gave wrong velocities or empty models; several lines on a frequency axis with REDSHIFT were misplaced; several lines were shifted by a fixed velocity |
| All tasks, headers | A rotated WCS (PC or CD matrix) was lost when the WCS was rebuilt; a Stokes axis with an empty CUNIT3 was taken as km/s; centres given as `07h36m51s` or `+65d35m12s` were misread; a path of a ring file was lowercased |
| Parameters | `RADII` as `Rmin~Rmax:step` could miss the last radius; the long forms of SIDE were not used |
| ADRIFT | ADRIFTPOL1 < 0 left the dispersion profile uninitialised |
| Initial estimates | The simplex of the inclination estimate did not keep the folded point (INC above 90°), so the estimate could end at 80° |
| pyBBarolo | Default `redshift=-1` gave $1/(1+z) = \infty$; `write_bestmodel` called a missing function; `useBBres=True` returned an undefined variable; `verbose=` raised a TypeError; `plot_profiles` failed unless all nine parameters were free; parallel sampling failed with the `spawn` start method of macOS; the wrapper did not skip comment lines of parameter files |
| BAYESIAN3DFIT | The `lambdas` penalties used the wrong separations; `update()` discarded the priors given as distributions; a negative parameter of a function of radius had zero likelihood; `RUNOPTS dlogz` failed with dynesty's dynamic sampler, and with nautilus gave `f_live = dlogz`; a failed or interrupted sampling left the worker processes running; the initial rings changed during a run |

## Differences by task

The section *Differences from BBarolo 1.8* of the page *how it works* of each task:
[3DFIT](../tasks/3dfit/algorithm.md#differences-from-bbarolo-18),
[BAYESIAN3DFIT](../tasks/bayesian3dfit/algorithm.md#differences-from-bbarolo-18),
[GALMOD](../tasks/galmod/algorithm.md#differences-from-bbarolo-18),
[GalMod3D](../tasks/galmod3d/algorithm.md#differences-from-bbarolo-18),
[GALWIND](../tasks/galwind/algorithm.md#differences-from-bbarolo-18),
[2DFIT](../tasks/2dfit/algorithm.md#differences-from-bbarolo-18),
[SPACEPAR](../tasks/spacepar/algorithm.md#differences-from-bbarolo-18),
[SEARCH](../tasks/search/algorithm.md#differences-from-bbarolo-18),
[MAKEMASK](../tasks/makemask/algorithm.md#differences-from-bbarolo-18),
[SMOOTH](../tasks/smooth/algorithm.md#differences-from-bbarolo-18),
[SMOOTHSPEC](../tasks/smoothspec/algorithm.md#differences-from-bbarolo-18),
[CONTSUB](../tasks/contsub/algorithm.md#differences-from-bbarolo-18),
[STATS](../tasks/stats/algorithm.md#differences-from-bbarolo-18),
[Moment maps and profiles](../tasks/maps/algorithm.md#differences-from-bbarolo-18),
[PVSLICE](../tasks/pvslice/algorithm.md#differences-from-bbarolo-18),
[ELLPROF](../tasks/ellprof/algorithm.md#differences-from-bbarolo-18),
[REND3D](../tasks/rend3d/algorithm.md#differences-from-bbarolo-18) and
[FITS utilities](../tasks/fitsutils/algorithm.md#differences-from-bbarolo-18).

## Not ported

- **SLITFIT**: the task did not work in the C++ code. It is replaced by the new
  {doc}`BAYESIANSLITFIT <../tasks/bayesianslitfit/index>`.
- **gnuplot output**: removed. All plots are made with matplotlib.
- **The MPI executable `BBarolo_MPI`**: see above.
- **The unfinished `BBauto` routine** of the C++ code.
- **The test module `BayesianBBarolo_test`** of pyBBarolo and `Galfit::getModel_BBB`.
- **The old Python layers** `api.py` and `wrapper.py` of pyBBarolo and the package name `pyBBarolo`.
