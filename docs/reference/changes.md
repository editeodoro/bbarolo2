# Changes from BBarolo 1.8

BBarolo 2 is BBarolo 1.8 (the C++ code and its Python wrapper pyBBarolo) rewritten in pure Python. The
parameter files of 1.8 run unchanged and give the same results, within the Monte Carlo noise of the models,
except where a bug of 1.8 was fixed. This page lists the main improvements and differences. The details for users
of 1.8 are in {doc}`differences`, and every change, with the numbers measured, is in
[CHANGES.md](https://github.com/editeodoro/bbarolo2/blob/main/CHANGES.md) of the repository.

## Main improvements

- **Global and Bayesian fits.** Besides the ring-by-ring fit of 1.8 (METHOD CLASSIC, the default), 3DFIT fits
  all the rings together: METHOD GLOBAL finds the maximum of the likelihood, METHOD BAYESIAN samples the
  posterior. They handle edge-on disks, parameters shared by the whole disk or given as functions of radius, the
  outer edge of the gas, a thick layer with a lag, smoothness priors and a model error, and their errors include
  the correlations of the parameters. They run from parameter files and from Python, and the nested sampler
  (nautilus) is included. See {doc}`../tasks/3dfit/likelihood`.
- **New fits.** {doc}`WINDFIT <../tasks/windfit/index>` for bi-conical winds and
  {doc}`SLITFIT <../tasks/slitfit/index>` for long-slit spectra.
- **Models without Monte Carlo noise.** `GALMODTYPE ANALYTIC` builds the models from analytic line profiles
  (the default of the global and Bayesian fits). {doc}`GalMod3D <../tasks/galmod3d/index>` builds model cubes
  from 3D fields of the density and velocity of the gas.
- **A Python interface.** Every task is a class that takes a FITS file or a cube in memory, computes in memory
  and returns astropy HDUs and Tables and matplotlib figures; files are written only when asked. See
  {doc}`../user-guide/python`.
- **Nothing to compile.** Install with pip; the numerical kernels are compiled at run time by numba. A 3DFIT of
  NGC 2403 takes 22 s instead of 67 s (on two cores).
- **Bugs of 1.8 fixed**, in 3DFIT, GALMOD, SPACEPAR, GALWIND, 2DFIT, ELLPROF and the other tasks: the list is in
  {doc}`differences`.
- **Reproducible fits.** The ring tables of 3DFIT no longer depend on the number of threads.
- **Clearer input checks.** Wrong values are errors with a message that names the parameter, checked before the
  run; a misspelt keyword is an error or a warning that proposes the right one.
- **New options**, among others: {doc}`SPECRES <../user-guide/spectral-resolution>` (the spectral resolution of
  the models), new initial estimates (GUESSTYPE), a deconvolved surface-density profile, ring parameters given
  as lists or `range()`, and a `<prefix>_params.par` file with all the parameters of each run.

## Main differences

- **Package and command.** The Python package is `bbarolo`, not `pyBBarolo`, and its classes and methods have
  changed ([the table](differences.md#from-pybbarolo-to-bbarolo)). The command `BBarolo` keeps its options; there is no
  `BBarolo_MPI`.
- **Rings of the models.** The first and last rings of the models are annuli centred on their radii
  (`RINGMODE CENTRED`); 1.8 moved the first and last radius by half a ring. The ring tables do not change;
  `RINGMODE SHIFTED` gives the models of 1.8.
- **Units.** Everything is in arcsec: DISTANCE is gone, and so are the columns in kpc of the ring files.
- **Beams.** BEAM (the beam of the data) and NEWBEAM (the new beam of SMOOTH) replace BMAJ, BMIN, BPA, BEAMFWHM,
  OBMAJ, OBMIN and OBPA.
- **Defaults.** Z0 is one pixel when not given (1.8: 0); THREADS is at most 16; every output file is named
  `<prefix>_<what>`.
- **Plots** are made with matplotlib only (no gnuplot), and can be redrawn from Python.
- **The Bayesian fit of pyBBarolo** is now 3DFIT with METHOD BAYESIAN, with a different likelihood, priors that
  follow the initial rings, nautilus as the default sampler and analytic models.
