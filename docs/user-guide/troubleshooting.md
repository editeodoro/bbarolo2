# Troubleshooting and FAQ

The messages below are copied from real runs of BBarolo 2.0.0 with the example cube of NGC 2403. Errors stop
the run (the command line prints the message and exits with status 1; Python raises an exception). Warnings
do not.

## Problems with the input

### `unknown parameter 'NRADIII'`

```text
 BBAROLO ERROR: unknown parameter 'NRADIII' (did you mean 'NRADII' or 'RADII'?)
```

A keyword of the parameter file is not known, usually a typo. BBarolo does not ignore it. In Python, the same
mistake raises `bbarolo.keywords.UnknownParameterError: unknown parameter 'incl' (did you mean 'INC'?)`. The
parameters of each task are listed on its page and with `BBarolo -d 3DFIT`; keywords are not case-sensitive.

### `Could not open the parameter file` and `is not a readable FITS file`

```text
 BBAROLO ERROR: Could not open the parameter file nofile.par.
```

```text
[Errno 2] No such file or directory: '/.../nothere.fits'
nothere.fits is not a readable FITS file!
Exiting ...
```

Paths in a parameter file are relative to the folder where you start BBarolo, not to the parameter file.

### `No beam information available for the data`

```text
HEADER WARNING: BMAJ-BMMAJ-CLBMMAJ keywords not found.
HEADER WARNING: BMIN-BMMIN-CLBMMIN keywords not found.
HEADER WARNING: BPA-BMPA keywords not found.
 BBAROLO ERROR: No beam information available for the data: set it with BEAM (BMAJ BMIN BPA, e.g. BEAM 30 20 45; BMAJ and BMIN in arcsec, or in degrees with a final d; one value is a circular beam), or beam=(bmaj, bmin, bpa) in Python.
```

The fits and models need the beam: the header has none (BBarolo also looks for it in `HISTORY` cards). Give
it with {ref}`BEAM <par-all-beam>`: `BEAM 30 20 45` (BMAJ and BMIN in arcsec, or in degrees with a final `d`;
BPA in degrees), or `BEAM 30` for a circular beam; in Python, `beam=(30, 20, 45)`. BBarolo 1.8 assumed a beam of 30 arcsec instead.

### Header without units, or with odd ones

A cube without `CUNIT3` or `BUNIT` still runs, with warnings:

```text
HEADER WARNING: CUNIT3 keywords not found. Assuming m/s (from CDELT3=-5152.86).
HEADER WARNING: BUNIT keyword not found.
...
Flux units are NONE.
  No known conversion to Jy.
```

A velocity axis without `CUNIT3` is taken as m/s when the channel is wider than 100 and as km/s otherwise: check
that this is right, and add the keyword to the file if not. Without `BUNIT` the fluxes cannot be converted to
Jy (only the catalogue and the maps need it; the fits with `NORM LOCAL` or `AZIM` do not). A positive `CDELT1`
gives

```text
HEADER CHECK: BBAROLO expects a negative CDELT1.
BBAROLO WARNING: Something seems wrong with the header. Fix it before going on.
```

The other conventions of the headers are in {doc}`conventions`.

### `VSYS=132800 km/s is outside the velocity range of the cube`

```text
 BBAROLO ERROR: VSYS=132800 km/s is outside the velocity range of the cube (-24.7 to 289.6 km/s): is it in m/s?
```

`VSYS` is in km/s. The fits stop with this error; GALMOD only warns. For a cube with a frequency or wavelength
axis, `VSYS` is relative to the rest frequency or wavelength, and `REDSHIFT`, `RESTFREQ` and `RESTWAVE` set the
zero (see {ref}`the spectral axes <par-all-veldef>` and {doc}`conventions`).

## Problems with the fit

### No source is found

```text
Final object count = 0
```

The source finder found nothing. When SEARCH is a task of its own it only writes an empty catalogue, but when
it builds the mask of 3DFIT (the default {ref}`MASK <par-3dfit-mask>` `SEARCH`) the run stops:

```text
 Creating mask (SEARCH) ...
 BBAROLO ERROR: MASKING error: No sources detected in the datacube. Cannot build mask!!!
```

(here with `THRESHOLD 100`, far above the data). Lower {ref}`SNRCUT <par-search-snrcut>` or
{ref}`THRESHOLD <par-search-threshold>`; look at the criteria of a detection, {ref}`MINPIX <par-search-minpix>`,
`MINCHANNELS` and `MINVOXELS`: with `MINPIX 100000` the same cube also has 0 objects, after finding 43 candidates
(`Intermediate list has 43 objects`). A cube with very few channels or pixels needs lower values. Another
mask can be chosen (`SMOOTH`, `THRESHOLD`, `FILE(mask.fits)`): see the {doc}`../tasks/search/index` and
{doc}`../tasks/makemask/index` pages.

### Rings that fail

The table of each stage of 3DFIT ends every row with the outcome of the ring. Rings far out, with few pixels
in the mask, show it clearly (here a run with a high threshold mask):

```text
     #       RAD     VROT    VDISP       fmin  outcome
    13    750.00   134.37     6.31      4.347  ok
    14    810.00  1117.86     2.72      0.995  rejected: VROT > 182 km/s (the band)
    ...
    18   1050.00        -        -          -  skipped: few pixels in the mask
```

- `skipped: few pixels in the mask`: the ring has fewer than 20 pixels in the mask (or fewer than 70% of its
  pixels are inside the cube). Use fewer rings, a larger {ref}`STARTRAD <par-3dfit-startrad>`, or a wider mask.
- `rejected: VROT > ... km/s (the band)`: the fitted rotation velocity is larger than the spectral band
  allows: the ring has no emission to fit. Remove the outer rings (`NRADII`).
- `not converged`: the minimiser did not converge (the message `Can not achieve convergence in ring #n` appears
  with `VERBOSE 2`; this one was not reproduced). Change the initial values (`VROT`, `INC`, `PA`) or the
  function to minimise.

Failed rings are left out of the ring files, and empty in the final model; `BADOUT true` keeps them in the
files, with `FITOK` 0. Bad initial values, in particular the inclination and the centre, are the usual cause of
bad fits: the {doc}`3DFIT page <../tasks/3dfit/index>` has advice.

## Time and resources

### The first run is slow

The first run compiles the numba kernels (about 5 s for a 3DFIT, up to a minute for all the tasks); the next
ones load them from the cache. See {doc}`parallel`.

### The fit is slow

Set {ref}`THREADS <par-all-threads>`; with the default (all the cores, at most 16) a shared machine may be busy.
The fits take seconds to a minute, METHOD BAYESIAN of 3DFIT minutes to hours. Plots take a few seconds; `PLOTS false` skips them.

(ts-numba-cache)=
### Strange errors after an update, or after moving the package

numba caches its kernels in the folder `__pycache__` of the package, and the cache records the names of the
modules. A stale cache (for example from before an update that renamed modules, or from another copy of the
code) can make the code import the wrong module, with errors that mention a module or a function that should
not exist. Delete the `__pycache__` folders of the package and of the folder you run it from:

```bash
find $(python3 -c "import bbarolo, os; print(os.path.dirname(bbarolo.__file__))") -name __pycache__ -exec rm -r {} +
```

This was not reproduced here. If you set `NUMBA_CACHE_DIR`, delete that folder instead.

## Outputs and results

### Where are the outputs?

On the command line, in the folder `output/<object>/` under the current folder, where `<object>` is the
`OBJECT` keyword of the header (`ngc2403`), unless {ref}`OUTFOLDER <par-all-outfolder>` is given. If the folder
exists, BBarolo warns and overwrites the files:

```text
 BBAROLO WARNING: the output folder output/ngc2403/ already exists: its files will be overwritten.
```

In Python, `run()` computes in memory and **writes nothing**: call `res.write()` to write the files of the
command line (see {doc}`python`). The list of the files is on the {doc}`outputs` page.

### My results differ from BBarolo 1.8 (C++)

Small differences are expected: the Monte Carlo models use different random numbers, so the models, and the fits
that use them, differ at the level of the noise of the model; and some defaults changed on purpose
(the ring convention `CENTRED`, `Z0` of one pixel, `THREADS` at most 16, a beam is required). To compare, use
`RINGMODE SHIFTED` and read {doc}`../reference/differences`. The results also depend on `THREADS` (the final
model only): see {doc}`parallel`.

## Reporting a problem

Look at the [issues of the repository](https://github.com/editeodoro/bbarolo2/issues) first, then open a
new one with:

- the version (`BBarolo -v`, or `bbarolo.__version__`) and the output of `BBarolo -v` for the libraries;
- the parameter file, or the Python lines, and the complete message (the output of the run, with `VERBOSE 2` if
  it is a fit);
- the header of the cube (`BBarolo` prints the beam and the axes at the start; or `fitsheader cube.fits`), or
  a small cube that shows the problem.

Questions about the method are welcome there too. To cite BBarolo, see {doc}`../reference/citing`.
