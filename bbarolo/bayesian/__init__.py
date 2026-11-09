"""
Bayesian BBarolo (BBB): fit a 3D tilted-ring model to a datacube by sampling the
posterior distribution of the ring parameters with nested sampling (dynesty or
nautilus). Pure-Python port of bayesian.py of the original C++-based pyBBarolo (BBB 0.2).

The interface follows the other bbarolo tasks (it builds on bbarolo.FitMod3D): the
cube and any BBarolo parameter are given to the constructor, the rings with the usual
keywords (radii or nradii/radsep, xpos, ypos, vsys, vrot, vdisp, inc, pa, z0, vrad, dens,
...; single values or one value per ring; the values not given are estimated or set as in 3DFIT):

    >>> from bbarolo import Bayesian3DFIT
    >>> f = Bayesian3DFIT("ngc2403.fits", radii=np.arange(30,600,60), xpos=77, ypos=77,
    ...                   vsys=132.8, vrot=120, vdisp=10, inc=60, pa=123.7, z0=10,
    ...                   mask="SEARCH", outfolder="output/bbb")
    >>> f.priors['vrot'] = dict(name='uniform', loc=0, scale=300)      # optional
    >>> f.run(freepar=['vrot','vdisp_single','inc_single'], method='dynesty', threads=4)
    >>> f.print_stats(); f.write_bestmodel(); f.corner_plot()

Parameter files (``BBarolo -p``, ``bbarolo.run``): the task BAYESIAN3DFIT, with the free
parameters in FREE (e.g. VROT VDISP PA_SINGLE INC_SINGLE RADMAX; by default VROT VDISP and the
geometry, Z0, RADMAX and FRACERR), the priors in PRIORS (e.g.
VROT=(uniform,loc=0,scale=250) INC=(norm,60,5)), and SAMPLER, SAMPLEROPTS, RUNOPTS and
THREADS; it samples, writes the outputs and the samples, prints the statistics (VERBOSE)
and makes the plots (PLOTS): see run_task and Bayesian3DFIT.write. These keywords are accepted by Bayesian3DFIT
too (defaults of run()), and Bayesian3DFIT.from_parfile reads a parameter file.

Free parameters are given as names: 'vrot' (one value per ring), 'vrot_single' (one value
for all rings), 'vrot_func' (parametric function of radius, see ``funcs``), plus
'sigma' (noise of the likelihood; default prior log-uniform between 0.5 and 1000 times the
measured noise), 'fracerr' (fractional model error: the variance of a voxel is
sigma^2 + (fracerr*model)^2) and 'radmax' (the outer edge of the disk; like_kwargs radmax='ring'
moves the last ring instead, as in the original module). Priors are frozen scipy.stats
distributions (``priors`` dictionary or ``prior_distr``). The default priors follow the initial
rings, given or estimated as in 3DFIT (e.g. INC +- 10 deg, PA +- 15 deg, VROT 0 to twice the
largest initial VROT; see Bayesian3DFIT._set_default_priors and the keyword PRIORS).

The options of the likelihood are given in run(like_kwargs=...) (LIKEOPTS in parameter files;
the options of the fit are in f.like_kwargs): likemask, dilation, noisecorr, densprof_smooth,
bbres (BBarolo's 3DFIT residuals), radmax and radmax_width (what RADMAX is), sigma and fracerr
(fixed values) and lambdas.

Correlated noise: the log-likelihood is multiplied by 1/(number of voxels per independent
measurement). By default (noisecorr='measured') this is the correlation volume of the noise
(spatial x spectral), measured from the voxels without emission (see noise_correlation), and
conservative: at least one beam area spatially, and summed over the positive core of the
autocorrelation where it has negative lobes (the noise of synthesis images);
noisecorr='beam' uses the beam area in pixels instead, and noisecorr=<number> sets it.

Voxels of the likelihood (likemask): 'dilated' (default) compares the model with the data,
not masked, in the mask dilated by one beam and three channels (dilation); 'none' with the
data in all the voxels of the likelihood region (see _setup_likemask). The original module
compared it with the data times the mask (zero below the threshold of the mask), which
penalised the faint model emission outside the mask and biased the fits.

Density profile when DENS is not fitted: the densities whose beam-convolved model reproduces
the total intensity map (non-negative least squares with a smoothness term, densprof_smooth;
see _deconvolved_profile), with errors (deconvolved_profile_errors). The original module used
the average of the observed map in each ring (Ellprof), which is smoothed by the beam twice
(by the data and by the model); that profile is now only an output (<prefix>_densprof.txt).

Differences from the original module (other than running without the C++ library):
  * the interface of the bbarolo tasks: Bayesian3DFIT(cube, **params), update(**params)
    and run(freepar=..., ...) replace init(), set_options() and compute();
    run_geometry() replaces compute_geometry(). The model engine always uses one thread
    (and TWOSTAGE is off); run(threads=n) samples with n processes (default 1, or THREADS
    if given);
  * a model evaluation is ~3x faster (pure-Python GALMOD/convolution engine, compiled
    residual kernel, empty channels skipped in the convolution);
  * parallel sampling (``threads>1``) uses a fork-based process pool whose workers
    inherit the fitter, so nothing big is pickled and it also works on macOS (where the
    default 'spawn' start method broke the original global-function approach). Under MPI,
    schwimmbad's MPIPool is used as before;
  * the models are ANALYTIC by default (GALMODTYPE=MC for the Monte-Carlo models): the
    likelihood has no Monte-Carlo noise, and is about twice faster;
  * the likelihood uses the measured noise per voxel when 'sigma' is not fitted (instead
    of sigma=1 in the units of the data), is multiplied by 1/(correlation volume of the
    noise) (each independent noise measurement counts once; noisecorr='beam' uses the beam
    area in pixels), and is computed in a fixed region (independent of the box of the
    model); with 'sigma' free, the normalisation counts all the voxels; 'fracerr' adds
    a model error proportional to the model; the data are not masked (likemask) and the
    density profile is corrected for the beam. The credible intervals are thus uncertainties in
    physical units. like_kwargs=dict(sigma=1, noisecorr=1) gives the noise conventions of the
    original module (sigma = 1 in the units of the data, independent voxels);
  * fixed: ``compute(verbose=...)`` raised a TypeError; ``useBBres=True`` returned an
    undefined variable; ``write_bestmodel`` called a missing ``_normalize_model``;
    ``plot_profiles`` failed unless all nine parameters were free and did not handle
    parametric ('_func') parameters; ``write_bestmodel`` failed with the density fitted;
    ``save_results``/``load_results`` now also work with nautilus;
  * the posterior errors are written in <prefix>_rings_final1.txt (as the 3DFIT error columns) and
    shown in the _parameters.pdf plot of write_bestmodel.

The package: fitter.py (Bayesian3DFIT: parameters, rings, priors, setup, run), likelihood.py (the
voxels, the region and the noise of the likelihood, the log-likelihoods), density.py (the deconvolved
density profile), sampling.py (dynesty, nautilus and the process pool), outputs.py (files, plots,
statistics, save and load), noise.py (the correlation of the noise), rings.py, task.py (BAYESIAN3DFIT).
"""

########################################################################
# Copyright (C) 2024 Enrico Di Teodoro
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
########################################################################

from .fitter import Bayesian3DFIT
from .wind import BayesianWind
from .noise import noise_correlation, dilate_mask
from .task import run_task, write_task

__all__ = ["Bayesian3DFIT", "BayesianWind", "noise_correlation", "dilate_mask", "run_task", "write_task"]
