"""
Bayesian BBarolo (BBB): fit a 3D tilted-ring model to a datacube by sampling the
posterior distribution of the ring parameters with nested sampling (dynesty or
nautilus). Pure-Python port of bayesian.py of the original C++-based pyBBarolo (BBB 0.2).

The interface follows the other bbarolo tasks (it builds on bbarolo.FitMod3D): the
cube and any BBarolo parameter are given to the constructor, the rings with the usual
keywords (radii or nradii/radsep, xpos, ypos, vsys, vrot, vdisp, inc, pa, z0, vrad, dens,
...; single values or one value per ring; geometric parameters not given are estimated):

    >>> from bbarolo import BayesianBBarolo
    >>> f = BayesianBBarolo("ngc2403.fits", radii=np.arange(30,600,60), xpos=77, ypos=77,
    ...                     vsys=132.8, vrot=120, vdisp=10, inc=60, pa=123.7, z0=10,
    ...                     mask="SEARCH", outfolder="output/bbb")
    >>> f.priors['vrot'] = dict(name='uniform', loc=0, scale=300)      # optional
    >>> f.run(freepar=['vrot','vdisp_single','inc_single'], method='dynesty', threads=4)
    >>> f.print_stats(); f.write_bestmodel(); f.corner_plot()

Parameter files (``BBarolo -p``, ``bbarolo.run``): the task BAYESIAN3DFIT, with the free
parameters in FREE (required; e.g. VROT VDISP PA_SINGLE INC_SINGLE RADMAX), the priors in PRIORS (e.g.
VROT=(uniform,loc=0,scale=250) INC=(norm,60,5)), and SAMPLER, SAMPLEROPTS, RUNOPTS and
THREADS; it samples, writes the outputs and the samples, prints the statistics (VERBOSE)
and makes the plots (PLOTS): see run_task. These keywords are accepted by BayesianBBarolo
too (defaults of run()), and BayesianBBarolo.from_parfile reads a parameter file.

Free parameters are given as names: 'vrot' (one value per ring), 'vrot_single' (one value
for all rings), 'vrot_func' (parametric function of radius, see ``funcs``), plus
'sigma' (noise of the likelihood; default prior log-uniform between 0.5 and 1000 times the
measured noise), 'fracerr' (fractional model error: the variance of a voxel is
sigma^2 + (fracerr*model)^2) and 'radmax' (outer radius). Priors are frozen scipy.stats
distributions (``priors`` dictionary or ``prior_distr``).

The options of the likelihood are given in run(like_kwargs=...) (LIKEOPTS in parameter files;
the options of the fit are in f.like_kwargs): likemask, dilation, noisecorr, densprof_smooth,
bbres (BBarolo's 3DFIT residuals), sigma and fracerr (fixed values) and lambdas.

Correlated noise: the log-likelihood is multiplied by 1/(number of voxels per independent
measurement). By default (noisecorr='measured') this is the correlation volume of the noise
(spatial x spectral), measured from the voxels without emission (see noise_correlation);
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
(by the data and by the model); that profile is now only an output (densprof.txt).

Differences from the original module (other than running without the C++ library):
  * the interface of the bbarolo tasks: BayesianBBarolo(cube, **params), update(**params)
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
  * the posterior errors are written in rings_final1.txt (as the 3DFIT error columns) and
    shown in the _parameters.pdf plot of write_bestmodel.
"""

########################################################################
# Copyright (C) 2024 Enrico Di Teodoro
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
########################################################################

import os
import sys
import math
import time
import copy
import contextlib
import multiprocessing as mp
import numpy as np
import numba as nb
from scipy import stats

from .tasks import FitMod3D, build_param, _checked, _FromFile

# A function to print only if verbose is True
vprint = lambda verbose, *args, **kwargs: print(*args, **kwargs) if verbose else None


def isIterable(p):
    """Check if p is an iteratable (list,tuple or numpy array). """
    return isinstance(p, (list, tuple, np.ndarray))

try:
    import dynesty as dyn
    from dynesty import utils as dyfunc
except ImportError:                                   # dynesty is needed only for method='dynesty'
    dyn = dyfunc = None

# MPI is used only when the script is launched with mpirun/mpiexec
isMPI = "OMPI_COMM_WORLD_SIZE" in os.environ or "PMI_RANK" in os.environ
if isMPI:
    try:
        from mpi4py import MPI
        from schwimmbad import MPIPool
    except ImportError:
        raise ImportError("ERROR! To run in parallel with MPI, you need to install mpi4py and schwimmbad.")

PARAMS = ('radii', 'xpos', 'ypos', 'vsys', 'vrot', 'vdisp', 'vrad', 'vvert', 'dvdz', 'zcyl', 'dens', 'z0', 'inc', 'phi')


def get_distribution(distr_name, *params, **kwargs):
    """Create a frozen scipy.stats distribution given its name and parameters."""
    try:
        distr = getattr(stats, distr_name)
        return distr(*params, **kwargs)
    except AttributeError:
        raise ValueError(f"ERROR: Distribution '{distr_name}' not found in scipy.stats.")


@nb.njit(cache=True, nogil=True)
def _sumsq_residuals(model, data, mask, maskmode, region):
    """sum over voxels of (d - model)^2, skipping NaN data (as np.nansum) and, if region
    (ny,nx) is not empty, the pixels outside region. maskmode: 0 = all the voxels, d = data;
    1 (or True) = d = data*mask (zero outside the mask); 2 = only the voxels in the mask,
    d = data (the other voxels are skipped)."""
    nz, ny, nx = model.shape
    useregion = region.shape[0] > 0
    s = 0.
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                if useregion and not region[j, i]:
                    continue
                d = data[k, j, i]
                if d != d:
                    continue
                if maskmode != 0 and not mask[k, j, i]:
                    if maskmode == 2:
                        continue
                    d = 0.
                r = d - model[k, j, i]
                s += r * r
    return s


@nb.njit(cache=True, nogil=True)
def _residuals_fracerr(model, data, mask, maskmode, region, sigma2, eps2):
    """As _sumsq_residuals (same voxels, mask and region), with a variance per voxel
    sigma^2 + eps^2 * model^2. Returns (sum of r^2 / variance, sum over the voxels where the
    model is not zero of log(variance / sigma^2))."""
    nz, ny, nx = model.shape
    useregion = region.shape[0] > 0
    s = 0.
    lg = 0.
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                if useregion and not region[j, i]:
                    continue
                d = data[k, j, i]
                if d != d:
                    continue
                if maskmode != 0 and not mask[k, j, i]:
                    if maskmode == 2:
                        continue
                    d = 0.
                m = model[k, j, i]
                r = d - m
                if m != 0.:
                    v = sigma2 + eps2 * m * m
                    s += r * r / v
                    lg += math.log(v / sigma2)
                else:
                    s += r * r / sigma2
    return s, lg


def _dilate(m, rz, ry, rx):
    """ Binary dilation of m by a box of half-sizes (rz, ry, rx) (separable, O(N)). """
    from scipy.ndimage import maximum_filter1d
    out = np.ascontiguousarray(m, dtype=bool).view(np.uint8)
    for axis, r in ((0, rz), (1, ry), (2, rx)):
        if r > 0:
            out = maximum_filter1d(out, size=2 * r + 1, axis=axis, mode='constant', cval=0)
    return out.astype(bool)


def dilate_mask(mask, radius, nchan):
    """ Dilation of a mask (nz, ny, nx) by a cylinder: a disk of the given radius (pixels)
    in each channel, then +-nchan channels. """
    from scipy.ndimage import distance_transform_edt, maximum_filter1d
    m = np.asarray(mask, bool)
    out = m.copy()
    if radius > 0:
        for k in np.flatnonzero(m.any(axis=(1, 2))):
            out[k] = distance_transform_edt(~m[k]) <= radius
    if nchan > 0:
        out = maximum_filter1d(out.view(np.uint8), size=2 * int(nchan) + 1, axis=0, mode='constant',
                               cval=0).astype(bool)
    return out


def noise_correlation(data, mask=None, beam_fwhm=None, maxlag_xy=None, maxlag_z=6, minvox=2000, minfree=20000):
    """ Correlation of the noise of a cube, measured where there is no emission.

    data: cube (nz, ny, nx); mask: the emission (same shape), dilated by about one beam and
    two channels to exclude the faint wings; beam_fwhm: beam FWHM in pixels; maxlag_xy:
    radius (pixels) of the spatial lags summed (default 2.5 beam FWHM); maxlag_z: largest
    spectral lag (channels).

    The correlation area is the sum of the normalised autocorrelation of the noise over the
    lags: the number of voxels per independent noise measurement. The noise is assumed
    separable, and the autocorrelations are computed with FFTs, weighting by the voxels
    used:
      * spatial (within the channel maps): in the channels without emission if they contain
        at least minfree noise voxels, otherwise in all the voxels outside the dilated mask;
      * spectral (along the spectra): in the runs of emission-free channels if they are at
        least three channels long and give at least minfree/2 pairs at lag 1 (lags up to the
        longest run - 1); otherwise in the pixels without emission in any channel, if they
        have at least minfree voxels; otherwise in the voxels outside the dilated mask.
    Emission-free channels have at most 0.1% of their pixels in the mask and are at least
    two channels away from those with more than 1%. Emission below the detection threshold
    of the mask adds spurious correlation, hence the preference for these channels; smaller
    samples give noisy sums (tens of per cent).

    For white noise convolved with a Gaussian beam the spatial area is twice the beam area
    in pixels; for noise whose correlation is the beam itself, as in interferometric images,
    one beam area. Hanning smoothing along the spectral axis gives 8/3 channels.

    Returns None if there are fewer than minvox noise voxels in all, otherwise a dict with
    spatial (pixels), spectral (channels), volume (= spatial * spectral, voxels), rho_spectral
    (autocorrelation at the spectral lags used), nnoise (number of noise voxels),
    spatial_from and spectral_from (the voxels used: 'free channels', 'free pixels' or
    'voxels outside the mask').
    """
    from scipy import fft
    data = np.asarray(data)
    nz, ny, nx = data.shape
    valid = np.isfinite(data)
    bp = float(beam_fwhm) if beam_fwhm is not None and beam_fwhm > 0 else 2.
    if mask is not None and np.any(mask):
        m = np.asarray(mask, bool)
        r = max(1, int(math.ceil(bp)))
        emis = _dilate(m, 2, r, r)
        noise = valid & ~emis
        frac = m.reshape(nz, -1).mean(axis=1)
        chanfree = frac <= 0.001
        for k in np.flatnonzero(frac > 0.01):
            chanfree[max(k - 1, 0):k + 2] = False
        pixfree = ~emis.any(axis=0)
        del emis
    else:
        noise = valid
        chanfree = np.ones(nz, bool)
        pixfree = np.ones((ny, nx), bool)
    nnoise = int(noise.sum())
    if nnoise < minvox:
        return None
    vals = data[noise]
    if vals.size > 2000000:
        vals = vals[np.random.default_rng(0).choice(vals.size, 2000000, replace=False)]
    med = float(np.median(vals))
    del vals

    def centred(sl, w):
        """ data - median in the selected voxels, 0 elsewhere (float64) """
        return np.where(w, np.asarray(data[sl], np.float64) - med, 0.)

    # spatial autocorrelation (zero spectral lag), averaged over the channels
    free_ch = int(noise[chanfree].sum()) >= minfree
    use = chanfree if free_ch else np.ones(nz, bool)
    R = float(maxlag_xy) if maxlag_xy else 2.5 * bp
    L = max(1, min(int(math.ceil(R)), ny - 1, nx - 1))
    fy, fx = fft.next_fast_len(ny + L), fft.next_fast_len(nx + L)
    pn = np.zeros((fy, fx // 2 + 1))
    pw = np.zeros((fy, fx // 2 + 1))
    chans = np.flatnonzero(use & noise.any(axis=(1, 2)))
    for k in range(0, len(chans), 8):
        cc = chans[k:k + 8]
        w = noise[cc]
        a = fft.rfft2(centred(cc, w), s=(fy, fx))
        b = fft.rfft2(w.astype(np.float64), s=(fy, fx))
        pn += (a.real ** 2 + a.imag ** 2).sum(axis=0)
        pw += (b.real ** 2 + b.imag ** 2).sum(axis=0)
    cn, cw = fft.irfft2(pn, s=(fy, fx)), fft.irfft2(pw, s=(fy, fx))
    lags = np.arange(-L, L + 1)
    iy, ix = np.ix_(lags % fy, lags % fx)
    with np.errstate(invalid='ignore', divide='ignore'):
        c = np.where(cw[iy, ix] > 0.5, cn[iy, ix] / cw[iy, ix], 0.)
    inside = np.hypot(*np.meshgrid(lags, lags, indexing='ij')) <= R
    spatial = float(np.sum(c[inside]) / c[L, L]) if c[L, L] > 0 else 1.
    # spectral autocorrelation (zero spatial lag)
    K = max(0, min(int(maxlag_z), nz - 1))
    runs = []
    k = 0
    while k < nz:
        if chanfree[k]:
            k2 = k
            while k2 + 1 < nz and chanfree[k2 + 1]:
                k2 += 1
            runs.append((k, k2 + 1))
            k = k2 + 1
        else:
            k += 1
    longest = max([b - a for a, b in runs], default=0)
    pairs1 = sum(int((noise[a:b - 1] & noise[a + 1:b]).sum()) for a, b in runs if b - a >= 2)
    free_px = int((valid & pixfree[None]).sum()) >= minfree
    if longest >= 3 and pairs1 >= minfree // 2:
        spectral_from = 'free channels'
        K = min(K, longest - 1)
        cz, wz = np.zeros(K + 1), np.zeros(K + 1)
        for a, b in runs:
            ok = noise[a:b]
            seg = centred(slice(a, b), ok)
            for t in range(min(K, b - a - 1) + 1):
                pr = ok[:b - a - t] & ok[t:]
                cz[t] += float(np.sum((seg[:b - a - t] * seg[t:])[pr]))
                wz[t] += float(pr.sum())
    else:
        spectral_from = 'free pixels' if free_px else 'voxels outside the mask'
        fz = fft.next_fast_len(nz + K)
        qn = np.zeros(fz // 2 + 1)
        qw = np.zeros(fz // 2 + 1)
        for j in range(0, ny, 32):
            w = (valid[:, j:j + 32] & pixfree[None, j:j + 32]) if free_px else noise[:, j:j + 32]
            a = fft.rfft(centred((slice(None), slice(j, j + 32)), w), n=fz, axis=0)
            b = fft.rfft(w.astype(np.float64), n=fz, axis=0)
            qn += (a.real ** 2 + a.imag ** 2).sum(axis=(1, 2))
            qw += (b.real ** 2 + b.imag ** 2).sum(axis=(1, 2))
        cz, wz = fft.irfft(qn, n=fz)[:K + 1], fft.irfft(qw, n=fz)[:K + 1]
    with np.errstate(invalid='ignore', divide='ignore'):
        rz = np.where(wz > 0.5, cz / wz, 0.)
    rz = rz / rz[0] if rz[0] > 0 else np.r_[1., np.zeros(K)]
    spectral = float(1. + 2. * np.sum(rz[1:]))
    spatial, spectral = max(spatial, 1.), max(spectral, 1.)
    return dict(spatial=spatial, spectral=spectral, volume=spatial * spectral, rho_spectral=rz, nnoise=nnoise,
                maxlag_xy=R, maxlag_z=K, spatial_from='free channels' if free_ch else 'voxels outside the mask',
                spectral_from=spectral_from)


# --------------------------------------------------------------------------
# Process pool: the workers inherit the fitter (fork) and only receive the parameter
# vectors. The functions must be importable module-level functions.
# --------------------------------------------------------------------------
_WORKER = {}


def _worker_init(key):
    _WORKER['f'] = _WORKER_REGISTRY[key]


_WORKER_REGISTRY = {}


def _worker_loglike(theta):
    # in a worker process 'f' is set by the initializer; samplers may also call the
    # functions in the main process (e.g. nautilus for the prior transform)
    f, like_kwargs = _WORKER['f'] if 'f' in _WORKER else _WORKER['parent']
    return f._log_like_func(theta, **like_kwargs)


def _worker_prior(u):
    f, _ = _WORKER['f'] if 'f' in _WORKER else _WORKER['parent']
    return f._prior_transform(u)


def _parent_loglike(theta):
    f, like_kwargs = _WORKER['parent']
    return f._log_like_func(theta, **like_kwargs)


def _parent_prior(u):
    f, _ = _WORKER['parent']
    return f._prior_transform(u)


def _make_pool(fitter, like_kwargs, nproc):
    key = id(fitter)
    _WORKER_REGISTRY[key] = (fitter, like_kwargs)
    try:
        ctx = mp.get_context("fork")
    except ValueError:
        print("WARNING: process pools need the 'fork' start method, not available on this system. "
              "Running serially.")
        return None
    return ctx.Pool(nproc, initializer=_worker_init, initargs=(key,))


RING_PARAMS = dict(radii='RADII', xpos='XPOS', ypos='YPOS', vsys='VSYS', vrot='VROT', vdisp='VDISP', vrad='VRAD',
                   vvert='VVERT', dvdz='DVDZ', zcyl='ZCYL', dens='DENS', z0='Z0', inc='INC', phi='PA')


class _RingDict(object):
    """A set of rings as a dictionary of arrays (r), with the Rings object of the model
    engine (object)."""

    def __init__(self, nrings):
        self.nr = int(nrings)
        self.r = {k: None for k in PARAMS}
        self.object = None

    @classmethod
    def from_rings(cls, rings):
        rd = cls(rings.nr)
        for k in PARAMS:
            rd.modify_parameter(k, getattr(rings, k))
        rd.make_object()
        return rd

    def copy(self):
        rd = _RingDict(self.nr)
        rd.r = {k: np.array(v) for k, v in self.r.items()}
        rd.make_object()
        return rd

    def set_rings_from_dict(self, rings):
        """ Define rings given an input dictionary of arrays (same keys as self.r) """
        for k in PARAMS:
            self.modify_parameter(k, rings[k])
        self.make_object()

    def modify_parameter(self, pname, pvalue, makeobj=False):
        """ Modifies one of the ring parameters (e.g., vrot, vdisp, etc.) """
        if pname not in self.r:
            raise ValueError("ERROR: Unknown ring parameter %s" % pname)
        self.r[pname] = np.array(pvalue, dtype=np.float32) if isIterable(pvalue) \
            else np.full(self.nr, pvalue, dtype=np.float32)
        if len(self.r[pname]) != self.nr:
            raise ValueError("All parameters must have size = %i" % self.nr)
        if makeobj:
            self.make_object()

    def make_object(self):
        """ Creates and stores the Rings object of the model engine """
        from .rings import Rings
        r = Rings()
        r.setRings(**{k: np.asarray(v, dtype=np.float64) for k, v in self.r.items()})
        r.radsep = float(np.float32(self.r['radii'][-1] - self.r['radii'][-2])) if self.nr > 1 else 0.
        self.object = r
        return r


class BayesianBBarolo(FitMod3D):
    """ Fit a galaxy model to a 3D datacube in a Bayesian framework (nested sampling).

    BayesianBBarolo(cube, **params): the cube (FITS file, HDU or (data, header)) and any
    BBarolo parameter, including the rings (radii or nradii/radsep, xpos, ypos, vsys, vrot,
    vdisp, inc, pa, z0, vrad, dens, vvert, dvdz, zcyl: single values or one value per ring).
    Ring parameters not fitted are kept fixed. Geometric parameters that are not given are
    estimated as in 3DFIT (VDISP=8, Z0=0, VRAD=0, DENS=1 if not given).

    The parameters of the BAYESIAN3DFIT task of parameter files are accepted too: FREE (the
    free parameters, e.g. 'VROT VDISP_SINGLE INC_SINGLE'; run() needs them, as FREE or as
    freepar), SAMPLER, SAMPLEROPTS and RUNOPTS are the defaults of run(); PRIORS sets entries
    of priors (e.g. 'VROT=(uniform,loc=0,scale=300)'). BayesianBBarolo.from_parfile(file)
    reads a parameter file.

    Methods
    -------
    update(**params):        change BBarolo parameters (and the rings).
    run(freepar, method, threads, log_like, like_kwargs, sampler_kwargs, run_kwargs, verbose,
        dynamic, **params):  sample the posterior ('dynesty' or 'nautilus').
    run_geometry(freepar, **kwargs): fit only the geometry on the total intensity map.
    write_bestmodel(plots, norm, **kwargs): write the best model, maps, PVs and plots.
    corner_plot(...), plot_profiles(...), print_stats(...), print_priors(), chi2_reduced()
    ring_errors(param):      1-sigma errors of a ring parameter in each ring of the best fit.
    save_results(outfile), load_results(inputfile, sampler)

    Attributes: priors (dict of prior definitions, see the module documentation), funcs
    (parametric functions of radius), lambdas (smoothness penalties), like_kwargs (the options
    of the likelihood of the fit, with the defaults), params (best-fit values of the free
    parameters), samples, samples_equal, quantiles, outri (best-fit rings).
    """

    TASK = "BAYESIAN3DFIT"              # its task flag in parameter files (the model engine is that of 3DFIT)
    _KEYWORD_OPTIONS = ("FREE", "SAMPLER", "PRIORS", "SAMPLEROPTS", "RUNOPTS", "LIKEOPTS")

    def __init__(self, cube=None, **params):
        self.taskname = "BAYESIAN3DFIT"
        self._method = None
        self.freepar_idx = None
        self.ndim = 0
        self.freepar_names = None
        self.priors = {}
        self.prior_distr = None
        self.funcs = {}
        self._galfit = self._ellprof = None
        self._cube_obj = None
        self._par = None
        self.modCalculated = False
        self._inri = self.outri = None
        self._default_pri = {}
        self.results = None
        self.sampler = None
        self.noise_correlation = None
        self._likemask_key = None
        self.dens_profile = None
        self.useBBres = False
        self._set_like_options({})              # like_kwargs: the options of the likelihood (defaults)
        super(BayesianBBarolo, self).__init__(cube, **params)
        if self._cube is not None:
            self._init_rings()

    # ------------------------------------------------------------------
    def update(self, beam=None, **params):
        """ Change BBarolo parameters (None restores the default). Changing the rings
        resets them (and the default priors that depend on them). FREE, SAMPLER, SAMPLEROPTS
        and RUNOPTS set the defaults of run(); PRIORS sets entries of self.priors (see
        bbarolo.params.bayes_priors). Returns self. """
        from .keywords import lookup
        given = {}
        for k, v in params.items():              # checked before they are set
            kw = lookup(k)
            if kw is not None and kw.name in self._KEYWORD_OPTIONS and v is not None:
                given[kw.name] = self._keyword_option(kw.name, v)
        super(BayesianBBarolo, self).update(beam=beam, **params)
        if not hasattr(self, "priors"):              # called by the base constructor
            return self
        if "PRIORS" in given:
            self.priors.update(given["PRIORS"])
            if self.prior_distr is not None:
                for k in given["PRIORS"]:
                    self.prior_distr[k] = None       # frozen again by the next run
        if params or beam is not None:
            self._galfit = self._ellprof = None
            self._cube_obj = None
            if self._inri is not None:
                self._init_rings()
        return self

    def _keyword_option(self, name, value):
        """ The value of FREE (list of free parameters), SAMPLER ('dynesty' or 'nautilus'),
        PRIORS (dict of priors), SAMPLEROPTS, RUNOPTS or LIKEOPTS (dict of options), from a
        string as in a parameter file. Raises ValueError. """
        from .params import bayes_freepar, bayes_priors, bayes_options, bayes_likeopts, BAYES_FUNC_PARS
        s = str(value)
        try:
            if name == "FREE":
                return bayes_freepar(s, funcs=set(BAYES_FUNC_PARS) | set(getattr(self, 'funcs', {})))
            if name == "LIKEOPTS":
                return bayes_likeopts(s)
            if name == "PRIORS":
                return bayes_priors(s)
            if name == "SAMPLER":
                m = s.strip().lower()
                if m not in ("dynesty", "nautilus"):
                    raise ValueError(f"unknown sampler {s.strip()} (DYNESTY or NAUTILUS)")
                return m
            return bayes_options(s)
        except ValueError as e:
            raise ValueError(f"BAYESIAN3DFIT: {name}: {e}") from None

    def _run_defaults(self):
        """ The defaults of run() given as parameters: FREE, SAMPLER, SAMPLEROPTS, RUNOPTS, LIKEOPTS. """
        return {k: self._keyword_option(k, self._params[k])
                for k in ("FREE", "SAMPLER", "SAMPLEROPTS", "RUNOPTS", "LIKEOPTS") if self._params.get(k) is not None}

    def _set_like_options(self, opts, custom=False):
        """ The options of the likelihood (run(like_kwargs=...) and LIKEOPTS; see
        bbarolo.params.bayes_like_option), checked and with the defaults. Sets like_kwargs (all
        the options of the fit) and _like_kwargs (those passed to the log-likelihood at each
        call: sigma, fracerr, lambdas and, with user-defined residuals or log-likelihood
        (custom=True), any other keyword; otherwise an unknown option raises ValueError). """
        from .params import bayes_like_option, LIKE_OPTIONS, LIKE_SETUP
        out = {k: LIKE_OPTIONS[k] for k in LIKE_SETUP}
        call = {}
        for k, v in dict(opts or {}).items():
            kv = bayes_like_option(k, v)
            if kv is None:
                if not custom:
                    raise ValueError(f"BayesianBBarolo: unknown option of the likelihood '{k}' (options: "
                                     + ", ".join(LIKE_OPTIONS) + "; any other only with user-defined residuals "
                                     "or log-likelihood)")
                kv = (k, v)
            key, val = kv
            if key not in LIKE_SETUP:
                if val is None:
                    out.pop(key, None)
                    call.pop(key, None)
                    continue
                call[key] = val
            out[key] = val
        self.like_kwargs, self._like_kwargs = out, call
        return out

    def _all_params(self, outfolder=None, fitsfile=None):
        # The defaults of the Bayesian fit that differ from 3DFIT (keywords.TASK_DEFAULTS): ANALYTIC
        # models, whose likelihood has no Monte-Carlo noise (with sigma = the noise, the noise of MC
        # models makes it rough and the sampling slow) and which are about twice faster. A GALMODTYPE
        # that is given (MC) is kept.
        from .keywords import TASK_DEFAULTS
        prm = super(BayesianBBarolo, self)._all_params(outfolder=outfolder, fitsfile=fitsfile)
        for k, v in TASK_DEFAULTS[self.TASK].items():
            prm.setdefault(k, v)
        return prm

    def _outfolder(self):
        of = self._params.get("OUTFOLDER")
        of = os.path.expanduser(str(of).strip()) if of not in (None, "") else self._default_outfolder()
        return of if of.endswith("/") else of + "/"

    def _make_par(self):
        """ The Param of the model engine (verbose off, one thread: the samples are computed
        in parallel by processes). """
        of = self._outfolder()
        prm = self._all_params(outfolder=of, fitsfile=self._input_file(of))
        prm.pop(self.TASK, None)                 # the engine is that of 3DFIT...
        prm.pop("FREE", None)                    # ...whose FREE is not that of BAYESIAN3DFIT
        prm["3DFIT"] = True
        prm["VERBOSE"] = False
        prm["THREADS"] = 1
        prm["TWOSTAGE"] = False                  # one set of rings (rings_final1.txt), also for the plots
        return _checked(build_param(prm))

    def _load_cube(self):
        """ Read the cube (with the current parameters). """
        from .cube import Cube
        par = self._make_par()
        c = Cube(par=par)
        p = c.pars()
        if p.outFolder == "":
            p.setOutfolder(self._outfolder())
        if not c.readCube(p.imageFile, printInfo=False):
            raise IOError(f"Cannot read {p.imageFile}")
        self._cube_obj, self._par = c, p
        return c

    def _init_rings(self):
        """ The initial rings from the parameters (estimating the missing geometry as in
        3DFIT), and the default priors. """
        from .rings import readRings
        from .paramguess import EstimateInitial
        from .galfit import ringfile_values
        c = self._cube_obj if self._cube_obj is not None else self._load_cube()
        par = copy.deepcopy(c.pars().gf)
        ringfile_values(par)                     # RINGFILE: the values not given
        toEstimate = ((par.RADII == "-1" and (par.NRADII == -1 or par.RADSEP == -1)) or par.XPOS == "-1"
                      or par.YPOS == "-1" or par.VSYS == "-1" or par.VROT == "-1" or par.PHI == "-1"
                      or par.INC == "-1")
        if toEstimate:
            c.checkBeam()
            if not c.MaskAll():
                c.BlankMask()
            ip = EstimateInitial(c, par)
            if par.RADII == "-1":
                if par.NRADII == -1: par.NRADII = ip.nrings
                if par.RADSEP == -1: par.RADSEP = ip.radsep
            if par.XPOS == "-1": par.XPOS = repr(float(ip.xcentre))
            if par.YPOS == "-1": par.YPOS = repr(float(ip.ycentre))
            if par.VSYS == "-1": par.VSYS = repr(float(ip.vsystem))
            if par.VROT == "-1": par.VROT = repr(float(ip.vrot))
            if par.INC == "-1": par.INC = repr(float(ip.inclin))
            if par.PHI == "-1": par.PHI = repr(float(ip.posang))
        for k, v in (("VDISP", "8.0"), ("Z0", "0.0"), ("VRAD", "0.0"), ("DENS", "1.0"), ("VVERT", "0.0"),
                     ("DVDZ", "0.0"), ("ZCYL", "0.0")):
            if getattr(par, k) == "-1":
                setattr(par, k, v)
        inR = readRings(par, c.Head())
        if inR.nr == 0:
            raise ValueError("BAYESIAN3DFIT ERROR: the number of radii must be > 0.")
        if toEstimate:
            print(f"\n{self.taskname}: initial rings ({inR.nr} rings every {inR.radsep:.2f} arcsec) with "
                  "estimated parameters:")
            for k, n in (("xpos", "XPOS"), ("ypos", "YPOS"), ("vsys", "VSYS"), ("vrot", "VROT"), ("inc", "INC"),
                         ("phi", "PA")):
                print(f"  {n:5s} = {np.median(getattr(inR, k)):.3f}")
        self._inri = _RingDict.from_rings(inR)
        self._set_default_priors()

    def _set_default_priors(self):
        """ Default priors (uniform), parametric functions and smoothness penalties. The
        priors that depend on the rings follow them, unless they were changed. """
        rr = {k: np.asarray(v, dtype=float) for k, v in self._inri.r.items()}
        radsep = float(np.abs(rr['radii'][-1] - rr['radii'][-2])) if len(rr['radii']) > 1 else float(rr['radii'][0])
        new = dict(
            xpos=dict(name='uniform', loc=rr['xpos'][0] - 5, scale=10),       # +- 5 pixels
            ypos=dict(name='uniform', loc=rr['ypos'][0] - 5, scale=10),       # +- 5 pixels
            vsys=dict(name='uniform', loc=rr['vsys'][0] - 20, scale=40),      # +- 20 km/s
            vrot=dict(name='uniform', loc=0, scale=350),                      # 0-350 km/s
            vdisp=dict(name='uniform', loc=0, scale=50),                      # 0-50 km/s
            vrad=dict(name='uniform', loc=0, scale=50),                       # 0-50 km/s
            inc=dict(name='uniform', loc=0, scale=90),                        # 0-90 degrees
            phi=dict(name='uniform', loc=0, scale=360),                       # 0-360 degrees
            vvert=dict(name='uniform', loc=0, scale=30),                      # 0-30 km/s
            dvdz=dict(name='uniform', loc=0, scale=2),                        # 0-2 km/s/pc
            zcyl=dict(name='uniform', loc=0, scale=5),                        # 0-5 arcsec
            dens=dict(name='uniform', loc=0.1, scale=200),                    # 0.1-200 1E20 cm^-2
            z0=dict(name='uniform', loc=0, scale=10),                         # 0-10 arcsec
            sigma=dict(name='loguniform', a=None, b=None),                    # None: 0.5 and 1000 x noise rms
            fracerr=dict(name='loguniform', a=1e-3, b=1.),                    # fractional model error
            radmax=dict(name='uniform', loc=float(rr['radii'][-1]) - radsep / 2., scale=radsep),
            vrot_p1=dict(name='uniform', loc=1, scale=350),
            vrot_p2=dict(name='uniform', loc=1, scale=1000),
            vdisp_p1=dict(name='uniform', loc=0, scale=10),
            vdisp_p2=dict(name='uniform', loc=-10, scale=20),
            dens_p1=dict(name='uniform', loc=1, scale=20),
            dens_p2=dict(name='uniform', loc=1, scale=200))
        for k, v in new.items():
            if k not in self.priors or self.priors[k] == self._default_pri.get(k):
                self.priors[k] = dict(v)
        self._default_pri = {k: dict(v) for k, v in new.items()}

        # Parametric functions of radius (default only for dens, vrot and vdisp)
        self.funcs.setdefault('vrot', lambda R, p1, p2: 2. / np.pi * p1 * np.arctan(R / p2))   # arctan curve
        self.funcs.setdefault('vdisp', lambda R, p1, p2: p1 + p2 * R)                          # linear
        self.funcs.setdefault('dens', lambda R, p1, p2: p1 * np.exp(-R / p2))                  # exponential

        self.prior_distr = {key: None for key in self.priors.keys()}
        # Lambda parameters for the smoothness penalty in the likelihood
        if not hasattr(self, 'lambdas'):
            self.lambdas = {key: 0.0 for key in ['vrot', 'vdisp', 'dens', 'vrad', 'phi', 'inc', 'xpos', 'ypos', 'z0']}

    # ------------------------------------------------------------------
    def _setup(self, freepar, **kwargs):
        """ Set: ndim, freepar_idx, freepar_names, _freepar_inp, prior_distr, the Galfit and
        Ellprof engines, mask, data and noise, with the options of the likelihood in
        like_kwargs (see run). """
        if kwargs:
            self.update(**{k: v for k, v in kwargs.items() if k not in ('verbose', 'dynamic')})
        if self._inri is None:
            self._init_rings()
        self.useBBres = bool(self.like_kwargs.get('bbres', False))
        self.freepar_idx = {}
        self.ndim = 0
        self.freepar_names = []
        self._freepar_inp = freepar
        dup = {key: False for key in self.priors.keys()}
        okpars = ['sigma', 'radmax', 'fracerr']
        for pname in freepar:
            s = pname.split('_')
            if s[0] not in self._inri.r and s[0] not in okpars:
                raise ValueError(f"ERROR! The requested free parameter is unknown: {s[0]}")
            if dup.get(s[0], False):
                raise ValueError(f"ERROR! Multiple entries for {s[0]} parameter. Please choose one!")
            dup[s[0]] = True
            if len(s) == 2 and 'single' in pname:
                self.freepar_idx[s[0]] = np.array([self.ndim])
                self.ndim += 1
                self.freepar_names.append(s[0])
            elif len(s) == 2 and 'func' in pname:
                if s[0] not in self.funcs:
                    raise ValueError(f"ERROR! The functional form for {s[0]} is unknown. Please define it first.")
                nparams = self.funcs[s[0]].__code__.co_argcount - 1
                for i in range(nparams):
                    self.freepar_names.append(f'{s[0]}_p{i + 1}')
                    self.freepar_idx[f'{s[0]}_p{i + 1}'] = np.arange(self.ndim, self.ndim + 1, dtype='int')
                    self.ndim += 1
            elif len(s) == 1:
                if s[0] in okpars:
                    self.freepar_idx[s[0]] = np.array([self.ndim])
                    self.ndim += 1
                    self.freepar_names.append(s[0])
                else:
                    self.freepar_idx[s[0]] = np.arange(self.ndim, self.ndim + self._inri.nr, dtype='int')
                    self.ndim += self._inri.nr
                    for i in range(self._inri.nr):
                        self.freepar_names.append(f'{s[0]}{i + 1}')
            else:
                raise ValueError(f"ERROR! Cannot interpret free parameter '{pname}'.")

        # The 3DFIT engine
        if self._galfit is None:
            from .galfit import Galfit
            c = self._cube_obj if self._cube_obj is not None else self._load_cube()
            self._galfit = Galfit(c, self._inri.object, self._par)
            self.mask = c.Mask()
            self.data = c.array
            self._setup_noise()

        # Priors (sigma: None values are relative to the measured noise, see _prior_kwargs). A
        # distribution frozen here is frozen again when its definition in self.priors has
        # changed (e.g. between two runs); one given in self.prior_distr is kept.
        if self.prior_distr is None:
            self.prior_distr = {}
        frozen = self.__dict__.setdefault('_frozen_priors', {})
        for key in self.freepar_idx:
            d = self.prior_distr.get(key)
            if d is not None and key in frozen and frozen[key][0] is d \
                    and frozen[key][1] != self._prior_definition(key):
                d = None
            if d is None:
                try:
                    d = get_distribution(self.priors[key]['name'], **self._prior_kwargs(key))
                except Exception:
                    raise ValueError(f"Something is wrong in the prior for parameter '{key}'. Please double check!")
                self.prior_distr[key] = d
                frozen[key] = (d, self._prior_definition(key))
            elif not isinstance(d, stats._distn_infrastructure.rv_continuous_frozen):
                raise ValueError(f"ERROR! Prior for {key} should be None or a scipy.stats distribution.")
        self._setup_likemask()
        self._setup_likebox()
        self._set_likefactor()
        self.map_rms = self._map_noise(self._nvox_map)

        # If the density is not fitted, the model is normalized with the azimuthal profile
        self.useNorm = not any('dens' in sub for sub in self.freepar_names)
        self.update_prof = False
        if self.useNorm and not self.useBBres:
            self._set_density(self._inri)
            # The profile changes during the fit only if the geometry (or the thickness) is fitted
            self.update_prof = any(sub in string for string in self.freepar_names
                                   for sub in ['inc', 'phi', 'xpos', 'ypos', 'radmax', 'z0'])

    def _setup_noise(self):
        """ Noise and correlation scale of the data, used by the likelihood.

        noiserms:   rms noise per voxel, from the median absolute deviation of the data
                    outside the mask;
        beam_area:  area of the (Gaussian) beam in pixels, pi*bmaj*bmin/(4 ln2)/pixel area:
                    the number of pixels that share the same noise (1 if the beam is unknown);
        likefactor: multiplicative factor of the log-likelihood (see _set_likefactor): by
                    default 1/(correlation volume of the noise), so that each independent
                    noise measurement, not each voxel, counts once.
        Also: the voxels of the likelihood (see _setup_likemask), and the maps of their
        squared data and number, used with the likelihood region (see _setup_likebox).
        """
        out = self.data[(self.mask == 0) & np.isfinite(self.data)]
        if out.size == 0:
            out = self.data[np.isfinite(self.data)]
        self.noiserms = float(1.4826 * np.median(np.abs(out - np.median(out))))
        if not self.noiserms > 0:
            self.noiserms = float(np.nanstd(out)) if out.size else 1.
        h = self._cube_obj.Head()
        bmaj, bmin = abs(h.Bmaj()), abs(h.Bmin())
        pixarea = abs(h.Cdelt(0) * h.Cdelt(1))
        if bmaj > 0 and bmin > 0 and pixarea > 0:
            self.beam_area = float(np.pi * bmaj * bmin / (4. * np.log(2.)) / pixarea)
        else:
            print("BAYESIAN WARNING: the beam of the cube is unknown, the pixels are treated as independent.")
            self.beam_area = 1.
        self.noise_correlation = None
        self._set_likefactor()
        self._setup_likemask(force=True)
        self.map_rms = self._map_noise(self._nvox_map)
        _, ys, xs = self.data.shape
        self._likebox = ([0, 0], [xs, ys])
        self._region = np.ones((ys, xs), bool)
        self._set_region_sums()

    def _prior_kwargs(self, key):
        """ Parameters of the prior of key (self.priors without 'name'). For sigma, None values
        are relative to the measured noise rms: loguniform a = 0.5 rms, b = 1000 rms (the
        default prior: scale-invariant between half and 1000 times the noise); scale = rms
        (e.g. halfcauchy, the default before November 2026). """
        kw = {k: v for k, v in self.priors[key].items() if k != "name"}
        if key == 'sigma':
            for k, f in (('a', 0.5), ('b', 1000.), ('scale', 1.)):
                if k in kw and kw[k] is None:
                    kw[k] = f * float(self.noiserms)
        return kw

    def _prior_definition(self, key):
        """ The distribution and the parameters of the prior of key (None if not defined). """
        try:
            return self.priors[key]['name'], self._prior_kwargs(key)
        except Exception:
            return None

    def _setup_likemask(self, force=False):
        """ The voxels of the likelihood, from like_kwargs['likemask']:

        'dilated' (default): the data, not masked, in the voxels of the mask dilated by
                           like_kwargs['dilation'] = (radius in units of the beam FWHM, number
                           of channels), default (1, 3): a disk of 1 FWHM radius in each
                           channel, then +-3 channels. The other voxels are not used;
        'none':            the data, not masked, in all the voxels of the likelihood region.

        The model is compared with the data in a fixed set of voxels: data selected through
        the mask of their neighbours, not their own values, and not clipped. (The original
        module compared it with the data times the mask, i.e. with zero below the threshold
        of the mask, which penalised the faint model emission there and biased the fits at
        low S/N: VDISP too low, INC too high.) The dilation should include the faint wings:
        with channels much narrower than the velocity dispersion, increase the number of
        channels (about 1.5 VDISP / channel width).

        Sets _likemask (the mask passed to the residuals), _maskmode (2: only the voxels in
        _likemask), the maps of the squared data and of the number of valid voxels of the
        likelihood in each pixel (_dm2_map, _nvox_map), and the total intensity map of the
        same voxels (data_mom0: run_geometry, the density profile). When the choice changes,
        the Ellprof engine of the outputs is reset.
        """
        lk = self.like_kwargs
        mode = str(lk['likemask']).lower()
        if mode not in ('dilated', 'none'):
            raise ValueError(f"ERROR! likemask must be 'dilated' or 'none', not {lk['likemask']!r}.")
        try:
            nbeam, nchan = (float(x) for x in lk['dilation'])
        except Exception:
            raise ValueError(f"ERROR! dilation must be (radius in beams, number of channels), not {lk['dilation']!r}.")
        if not (nbeam >= 0 and nchan >= 0):
            raise ValueError(f"ERROR! dilation must be >= 0: {lk['dilation']!r}.")
        nchan = int(round(nchan))
        key = (mode, nbeam, nchan) if mode == 'dilated' else (mode,)
        if key == self._likemask_key and not force:
            return
        m = np.asarray(self.mask, bool)
        if mode == 'dilated':
            h = self._cube_obj.Head()
            bpix = abs(h.Bmaj() / h.Cdelt(0)) if h.Bmaj() and h.Cdelt(0) else 1.
            lm, self._maskmode = dilate_mask(m, nbeam * bpix, nchan), 2
            self._likemask_note = (f"data in the mask dilated by {nbeam:g} beam (radius {nbeam * bpix:.1f} pixels) "
                                   f"and +-{nchan} channels")
        else:
            lm, self._maskmode = np.ones(m.shape, bool), 2
            self._likemask_note = "all the voxels of the region"
        self._likemask = lm
        nz, ny, nx = self.data.shape
        self._dm2_map = np.zeros((ny, nx))
        self._nvox_map = np.zeros((ny, nx), np.int64)
        mom0 = np.zeros((ny, nx))
        for k in range(0, nz, 16):                     # in chunks of channels (no full float64 copies)
            v = lm[k:k + 16] & np.isfinite(self.data[k:k + 16])
            d = np.where(v, self.data[k:k + 16], 0.).astype(np.float64)
            self._dm2_map += np.sum(d * d, axis=0)
            self._nvox_map += v.sum(axis=0)
            mom0 += d.sum(axis=0)
        self.data_mom0 = mom0
        if self._likemask_key is not None and key != self._likemask_key:
            self._ellprof = None
        self._likemask_key = key

    def _likemask_map(self):
        """ Total intensity map of the data in the voxels of the likelihood, in the units of
        the zeroth moment (MomentMap): sum x channel width (Jy/beam km/s, or as the cube), NaN
        in the pixels without such voxels. The map of the observed density profile of the
        outputs (see _observed_profile). """
        from .bbutils import DeltaVel, FluxtoJyBeam
        c = self._cube_obj
        dv = abs(DeltaVel(c.Head())) if c.HeadDef() else 1.
        m0 = self.data_mom0
        if c.HeadDef() and c.pars().fluxConvert:
            m0 = FluxtoJyBeam(m0, c.Head())
        return np.where(self._nvox_map > 0, m0 * dv, np.nan).astype(np.float32)

    def _map_noise(self, nchan):
        """ rms noise of the sum of nchan channels (array), for the total intensity map of
        run_geometry: noiserms * sqrt(nchan) for independent channels; with the measured
        correlation of the noise (noisecorr='measured'), the channels are assumed contiguous,
        variance = noiserms^2 * (n + 2 sum_l (n - l) rho_l) over the spectral lags l. """
        n = np.asarray(nchan, np.float64)
        var = n.copy()
        nc = self.noise_correlation if str(self.like_kwargs['noisecorr']).lower() == 'measured' else None
        if nc is not None:
            rho = np.asarray(nc['rho_spectral'], float)
            for lag in range(1, len(rho)):
                var += 2. * np.clip(n - lag, 0., None) * rho[lag]
        return self.noiserms * np.sqrt(np.clip(var, 0., None))

    def _set_region_sums(self):
        """ Summed-area table of the squared data of the likelihood in the region, and nvox. """
        a = np.where(self._region, self._dm2_map, 0.)
        self._dm2_sat = np.zeros((a.shape[0] + 1, a.shape[1] + 1))
        self._dm2_sat[1:, 1:] = np.cumsum(np.cumsum(a, axis=0), axis=1)
        self._dm2_box = float(self._dm2_sat[-1, -1])
        self.nvox = int(np.sum(self._nvox_map[self._region]))

    def _set_likefactor(self):
        """ Multiplicative factor of the log-likelihood for the correlation of the noise:
        1/(number of voxels per independent measurement), from like_kwargs['noisecorr']:

        'measured' (default): 1/(correlation volume of the noise, measured from the voxels
                          without emission, spatial x spectral; see noise_correlation), and
                          1/(spatial correlation area) for the likelihood of run_geometry;
                          if it cannot be measured (too few voxels without emission, e.g.
                          MASK=NONE), the beam area;
        'beam':           1/(beam area in pixels), spatial correlation only (the default
                          before November 2026);
        a number N > 0:   N voxels (and N pixels of the total intensity map) per independent
                          noise value: the log-likelihood x 1/N (N=1: independent voxels).
        Sets likefactor (3D likelihood) and likefactor2d (total intensity map). """
        mode = self.like_kwargs['noisecorr']
        A = A2 = max(self.beam_area, 1.)
        self._likefactor_note = f"1/beam area in pixels = 1/{self.beam_area:.2f}"
        if isinstance(mode, (int, float, np.integer, np.floating)) and not isinstance(mode, (bool, np.bool_)):
            if not (np.isfinite(mode) and mode > 0):
                raise ValueError(f"ERROR! noisecorr must be 'beam', 'measured' or a number > 0, not {mode!r}.")
            A = A2 = float(mode)
            self._likefactor_note = f"1/{A:g} (noisecorr given: voxels per independent noise value)"
        elif str(mode).lower() == 'measured':
            if self.noise_correlation is None:
                h = self._cube_obj.Head()
                bpix = abs(h.Bmaj() / h.Cdelt(0)) if h.Bmaj() and h.Cdelt(0) else None
                self.noise_correlation = noise_correlation(self.data, self.mask, bpix)
            nc = self.noise_correlation
            if nc is None:
                print("BAYESIAN WARNING: too few voxels without emission to measure the correlation of the "
                      "noise:\n         using the beam area (like_kwargs noisecorr='beam').")
            else:
                A, A2 = nc['volume'], nc['spatial']
                self._likefactor_note = (f"1/{A:.2f} (measured noise correlation: {nc['spatial']:.2f} pixels x "
                                         f"{nc['spectral']:.2f} channels; beam area {self.beam_area:.2f} pixels)")
        elif str(mode).lower() != 'beam':
            raise ValueError(f"ERROR! noisecorr must be 'beam', 'measured' or a number, not {mode!r}.")
        self.likefactor = 1. / A
        self.likefactor2d = 1. / A2

    @staticmethod
    def _boxsum(S, blo, bhi):
        return float(S[bhi[1], bhi[0]] - S[blo[1], bhi[0]] - S[bhi[1], blo[0]] + S[blo[1], blo[0]])

    def _setup_likebox(self):
        """ Likelihood region: the pixels where the model can be non-zero for any value of the
        free parameters allowed by the priors. It is the projected disk within the outermost
        radius (the upper bound of radmax if fitted, for the lowest inclination and all the
        position angles allowed), widened by 3*z0 + 0.5 beam FWHM (+ the allowed shift of
        the centre), within the union of the model boxes (DiskModel.box). The region
        contains >99.8% of the flux of the (convolved) model.

        The residuals are computed in this fixed region (the model is zero outside its own
        box), so that the likelihood does not depend on the box in which the model is
        calculated. Data beyond the outermost ring are not part of the fit (they would
        otherwise inflate a fitted sigma when only the inner rings are modelled). nvox,
        the number of valid voxels of the likelihood within the region (in the mask, or in
        the dilated mask, or all, see _setup_likemask), is the number of measurements of
        the normalisation term.
        """
        _, ys, xs = self.data.shape

        def upper(key, default):
            d = self.prior_distr.get(key) if key in self.freepar_idx else None
            if d is None:
                return default
            return float(d.ppf(1.)) if np.isfinite(d.ppf(1.)) else float(d.ppf(1 - 1e-6))

        def lower(key, default):
            d = self.prior_distr.get(key) if key in self.freepar_idx else None
            if d is None:
                return default
            return float(d.ppf(0.)) if np.isfinite(d.ppf(0.)) else float(d.ppf(1e-6))
        z0free = any(k.startswith('z0') for k in self.freepar_idx)
        if z0free and 'z0' not in self.freepar_idx:            # z0 as a function of radius
            blo, bhi = [0, 0], [xs, ys]
        else:
            rings = self._inri.object.copy()
            rings.radii[-1] = upper('radmax', float(rings.radii[-1]))
            if 'radmax' in self.freepar_idx and rings.nr > 1:
                rings.radsep = float(np.float32(rings.radii[-1] - rings.radii[-2]))
            rings.z0[-1] = upper('z0', float(np.max(rings.z0)))
            xp, yp = float(rings.xpos[-1]), float(rings.ypos[-1])
            corners = []
            for x in (lower('xpos', xp), upper('xpos', xp)):
                for y in (lower('ypos', yp), upper('ypos', yp)):
                    rings.xpos[-1], rings.ypos[-1] = x, y
                    corners.append(self._galfit.model.box(rings, final=True))
            blo = [min(c[0][0] for c in corners), min(c[0][1] for c in corners)]
            bhi = [max(c[1][0] for c in corners), max(c[1][1] for c in corners)]
            blo = [max(int(blo[0]), 0), max(int(blo[1]), 0)]
            bhi = [min(int(bhi[0]), xs), min(int(bhi[1]), ys)]
        self._likebox = (blo, bhi)
        self._region = np.zeros((ys, xs), bool)
        self._region[blo[1]:bhi[1], blo[0]:bhi[0]] = True
        if not (z0free and 'z0' not in self.freepar_idx):
            self._region[blo[1]:bhi[1], blo[0]:bhi[0]] = self._disk_region(blo, bhi, upper, lower)
        self._set_region_sums()

    def _disk_region(self, blo, bhi, upper, lower):
        """ Pixels of the box [blo,bhi) within the projected outer disk, widened (see _setup_likebox). """
        from scipy.ndimage import distance_transform_edt
        h = self._cube_obj.Head()
        pix = 0.5 * (abs(h.Cdelt(0)) + abs(h.Cdelt(1))) * 3600.          # arcsec
        r = self._inri.r
        radii = np.asarray(r['radii'], float)
        radsep = float(radii[-1] - radii[-2]) if len(radii) > 1 else 0.
        rout = upper('radmax', float(radii[-1]))
        if 'radmax' in self.freepar_idx and len(radii) > 1:
            radsep = rout - float(radii[-2])
        rout += radsep / 2.
        z0max = upper('z0', float(np.max(r['z0'])))
        inc = np.asarray(r['inc'], float)
        incmin = min(float(inc.min()), lower('inc', float(inc.min())))
        incmin = min(max(incmin, 0.), 89.)
        phis = np.asarray(r['phi'], float)
        if 'phi' in self.freepar_idx:
            plo, phi_hi = lower('phi', float(phis.min())), upper('phi', float(phis.max()))
            phis = np.linspace(plo, phi_hi, 73) if phi_hi - plo < 180. else np.array([0.])
            if phi_hi - plo >= 180.:
                incmin = 0.                                              # any orientation: a circle
        xc, yc = float(r['xpos'][-1]), float(r['ypos'][-1])
        shift = max(abs(upper('xpos', xc) - xc), abs(lower('xpos', xc) - xc),
                    abs(upper('ypos', yc) - yc), abs(lower('ypos', yc) - yc)) * pix
        yy, xx = np.mgrid[blo[1]:bhi[1], blo[0]:bhi[0]]
        dx, dy = (xx - xc) * pix, (yy - yc) * pix
        cinc = np.cos(np.radians(incmin))
        disk = np.zeros(xx.shape, bool)
        for ph in np.unique(phis):
            sp, cp = np.sin(np.radians(ph)), np.cos(np.radians(ph))
            xr = -dx * sp + dy * cp
            yr = (-dx * cp - dy * sp) / cinc
            disk |= xr * xr + yr * yr <= rout * rout
        if not disk.any():
            return np.ones(xx.shape, bool)
        margin = 3. * z0max + 0.5 * abs(h.Bmaj()) * 3600. + shift + 0.5 * pix
        return distance_transform_edt(~disk) * pix <= margin

    def _model_box(self, rings):
        """ Box [blo,bhi) of the model for the given rings, within the likelihood region. """
        blo, bhi = self._galfit.model.box(rings.object, final=True)
        Lb, Hb = self._likebox
        blo = [max(blo[0], Lb[0]), max(blo[1], Lb[1])]
        bhi = [max(min(bhi[0], Hb[0]), blo[0]), max(min(bhi[1], Hb[1]), blo[1])]
        return blo, bhi

    def _sumsq_outside(self, blo, bhi):
        """ Sum of the squared masked data in the likelihood region outside the model box
        [blo,bhi), where the model is zero. """
        return max(self._dm2_box - self._boxsum(self._dm2_sat, blo, bhi), 0.)

    # ------------------------------------------------------------------
    def _calculate_model(self, rings, fullcube=False):
        """ Unnormalized (convolved) model for a set of rings.

        Returns (mod, bhi, blo, None): mod is the model in the box [blo,bhi) (the full cube
        if fullcube=True). The last element (a C++ pointer in the original) is kept for
        compatibility.
        """
        _, ys, xs = self.data.shape
        bhi, blo = [xs, ys], [0, 0]
        if not fullcube:
            # Model built in a smaller box containing the galaxy within the last ring
            blo, bhi = self._model_box(rings)
        mod = self._galfit.model.array(rings.object, blo, bhi)
        return mod, np.array(bhi), np.array(blo), None

    def _calculate_residuals(self, model, data, mask=None, **kwargs):
        """ Default residuals (-log-likelihood): Gaussian with noise sigma per voxel. Can be overridden.

        kwargs: sigma (default: the measured noise rms), region (pixels of the box that are in
        the likelihood region, default all), maskmode (how the mask is used: 2 = only the
        voxels in the mask, the default, as in the likelihood, see _setup_likemask; 1 =
        data*mask, zero outside the mask, as the original module; 0 = mask not used), ss_out (sum of the squared data of the likelihood in the
        region outside the model box, default 0), nvox (number of measurements for the
        normalisation term, default the voxels of the likelihood in the region), fracerr
        (fractional model error eps: the variance of a voxel is sigma^2 + (eps*model)^2,
        default 0).
        """
        sigma = kwargs.get('sigma')
        if sigma is None:
            sigma = self.noiserms
        if not sigma > 0:
            return np.inf
        region = kwargs.get('region')
        if region is None:
            region = np.zeros((0, 0), bool)
        maskmode = int(kwargs.get('maskmode', 2)) if mask is not None else 0
        eps = kwargs.get('fracerr')
        if eps is not None and eps > 0:
            return self._residuals_fracerr(model, data, mask, region, sigma, float(eps),
                                           ss_out=kwargs.get('ss_out', 0.), nvox=kwargs.get('nvox'),
                                           maskmode=maskmode)
        if model.ndim == 3 and data.shape == model.shape and (mask is None or mask.shape == model.shape):
            ss = _sumsq_residuals(np.asarray(model, np.float64), data,
                                  mask if mask is not None else data, maskmode, region)
        else:
            d, w = self._masked(data, mask, maskmode)
            r2 = np.where(w, (d - model) ** 2, 0.)
            ss = np.nansum(r2 * region[None]) if region.size else np.nansum(r2)
        ss += kwargs.get('ss_out', 0.)
        nvox = kwargs.get('nvox', getattr(self, 'nvox', 1))
        return 0.5 * ss / sigma ** 2 + 0.5 * nvox * np.log(2. * np.pi * sigma ** 2)

    @staticmethod
    def _masked(data, mask, maskmode):
        """ (data used, voxels used) for maskmode (see _calculate_residuals). """
        if mask is None or maskmode == 0:
            return data, True
        if maskmode == 2:
            return data, np.asarray(mask, bool)
        return data * mask, True

    def _residuals_fracerr(self, model, data, mask, region, sigma, eps, ss_out=0., nvox=None, maskmode=None):
        """ -log-likelihood with the fractional model error eps (see _calculate_residuals):
        0.5 * sum[r^2 / (sigma^2 + eps^2 m^2) + log(2 pi (sigma^2 + eps^2 m^2))] over the
        voxels, where the voxels outside the model box (ss_out, nvox) have variance sigma^2. """
        s2, e2 = float(sigma) ** 2, float(eps) ** 2
        if maskmode is None:
            maskmode = 2 if mask is not None else 0
        if model.ndim == 3 and data.shape == model.shape and (mask is None or mask.shape == model.shape):
            ss, lg = _residuals_fracerr(np.asarray(model, np.float64), data, mask if mask is not None else data,
                                        int(maskmode), region, s2, e2)
        else:
            d, used = self._masked(data, mask, maskmode)
            v = s2 + e2 * np.asarray(model, np.float64) ** 2
            w = (region[None] if region.size else 1.) * np.asarray(used, float)
            ss = float(np.nansum(np.where(w > 0, (d - model) ** 2 / v, 0.) * w))
            lg = float(np.nansum(np.where(np.isfinite(d) & (model != 0) & (w > 0), np.log(v / s2), 0.)))
        ss += ss_out / s2
        if nvox is None:
            nvox = getattr(self, 'nvox', 1)
        return 0.5 * ss + 0.5 * nvox * np.log(2. * np.pi * s2) + 0.5 * lg

    def _log_likelihood(self, theta, **kwargs):
        """ Log-likelihood of the parameters theta.

        Gaussian likelihood of the data in the voxels of the likelihood (likemask: by default
        the data, not masked, in the dilated mask; see _setup_likemask), with noise sigma per
        voxel: sigma is a free parameter if 'sigma' is in freepar, otherwise the measured noise
        rms (self.noiserms) or the value given in like_kwargs.
        Since the noise is correlated, the log-likelihood is multiplied by self.likefactor:
        by default 1/(correlation volume of the noise measured from the voxels without
        emission, spatial x spectral), so that each independent noise measurement counts
        once; with noisecorr='beam', 1/(beam area in pixels); with noisecorr=N, 1/N (see
        _set_likefactor).

        With 'fracerr' free (or given in like_kwargs), the variance of each voxel is
        sigma^2 + (fracerr * model)^2: a model error proportional to the model.

        Accepted kwargs (the options of like_kwargs used at each call): sigma (fixed noise per
        voxel, default self.noiserms); fracerr (fixed fractional model error, default 0);
        lambdas (dict of smoothness penalties, default self.lambdas, not multiplied by
        likefactor); any keyword accepted by user-defined residuals (_calculate_residuals).
        """
        a = ('vrot', 'vdisp', 'inc', 'xpos', 'ypos', 'z0', 'dens', 'radmax')
        for k in self.freepar_idx:
            if k.startswith(a) and np.any(theta[self.freepar_idx[k]] < 0):
                return -np.inf
        rings = self._update_rings(self._inri, theta)
        if self.useBBres:
            return -self._galfit.calculateResiduals(rings.object)

        if self.useNorm and self.update_prof:
            self._set_density(rings)
        model, bhi, blo = self._calculate_model(rings)[:3]
        mask = self._likemask[:, blo[1]:bhi[1], blo[0]:bhi[0]]
        data = self.data[:, blo[1]:bhi[1], blo[0]:bhi[0]]
        if 'sigma' in self.freepar_names:
            kwargs['sigma'] = theta[self.freepar_idx['sigma']][0]
        elif kwargs.get('sigma') is None:
            kwargs['sigma'] = self.noiserms
        if 'fracerr' in self.freepar_idx:
            kwargs['fracerr'] = theta[self.freepar_idx['fracerr']][0]
        kwargs.setdefault('maskmode', self._maskmode)
        kwargs.setdefault('ss_out', self._sumsq_outside(blo, bhi))
        kwargs.setdefault('region', self._region[blo[1]:bhi[1], blo[0]:bhi[0]])
        logL = self.likefactor * self._calculate_residuals(model, data, mask, **kwargs)

        # Smoothness penalty for ring-by-ring parameters
        lambdas = kwargs.get('lambdas', self.lambdas)
        if any(lambdas[key] > 0 for key in lambdas):
            dr = rings.r['radii'][1] - rings.r['radii'][0]
            for key in self.freepar_idx:
                if key in lambdas and lambdas[key] > 0 and len(self.freepar_idx[key]) > 1:
                    v = rings.r[key]
                    if key == 'vrot':                     # force the rotation curve to start from zero
                        v = np.concatenate([[0.0], v])
                    d2v = (v[2:] - 2 * v[1:-1] + v[:-2]) / dr ** 2
                    logL += 0.5 * lambdas[key] * np.sum(d2v ** 2)
        return -logL

    def _log_likelihood_geometry(self, theta, **kwargs):
        """ Likelihood on the total intensity map (geometry fit, Cannubi-style).

        Gaussian likelihood of the total intensity map (data_mom0: the sum of the data in the
        voxels of the likelihood along each spectrum, see _setup_likemask), with the noise of
        the map in each pixel (map_rms: noiserms*sqrt(number of channels summed), larger with
        the measured spectral correlation of the noise, see _map_noise), computed in the
        fixed likelihood region and multiplied by self.likefactor2d (1/the measured spatial
        correlation area of the noise, 1/beam area in pixels with noisecorr='beam', 1/N with
        noisecorr=N), as in _log_likelihood.
        """
        a = ('inc', 'xpos', 'ypos', 'dens', 'z0', 'radmax')
        for k in self.freepar_idx:
            if k.startswith(a) and np.any(theta[self.freepar_idx[k]] < 0):
                return -np.inf
        rings = self._update_rings(self._inri, theta)
        if self.useNorm and self.update_prof:
            self._set_density(rings)
        model, bhi, blo = self._calculate_model(rings, fullcube=False)[:3]
        if model.ndim == 4:                 # (non-convolved, convolved) models
            model = model[-1]
        Lb, Hb = self._likebox
        map_data = self.data_mom0[Lb[1]:Hb[1], Lb[0]:Hb[0]]
        map_rms = self.map_rms[Lb[1]:Hb[1], Lb[0]:Hb[0]]
        map_mod = np.zeros_like(map_data, dtype=np.float64)
        map_mod[blo[1] - Lb[1]:bhi[1] - Lb[1], blo[0] - Lb[0]:bhi[0] - Lb[0]] = np.nansum(model, axis=0)
        ok = (map_rms > 0) & self._region[Lb[1]:Hb[1], Lb[0]:Hb[0]]
        res = 0.5 * np.nansum(((map_data[ok] - map_mod[ok]) / map_rms[ok]) ** 2)
        return -self.likefactor2d * res

    def chi2_reduced(self, theta=None, fracerr=False):
        """ Reduced chi-square of the (Gaussian) likelihood for the parameters theta (default:
        the best-fit parameters): sum of the squared residuals / (sigma^2 * nvox), with sigma
        the fitted noise if 'sigma' is free, otherwise the one of like_kwargs or the measured
        noise rms. Values much
        larger than 1 mean that the model does not describe the data down to the noise,
        so that the credible intervals obtained with sigma fixed to the noise are too small:
        in this case fit 'sigma' or 'fracerr' as well (freepar=[..., 'fracerr']).
        fracerr=True: with the variance sigma^2 + (fracerr*model)^2 of each voxel, using the
        fitted fractional model error, or the one of like_kwargs (or a number: that value). """
        theta = self.params if theta is None else np.asarray(theta, float)
        rings = self._inri.copy()
        rings = self._update_rings(rings, theta)
        if self.useNorm and self.update_prof:
            self._set_density(rings)
        model, bhi, blo = self._calculate_model(rings)[:3]
        if model.ndim == 4:
            model = model[-1]
        mask = self._likemask[:, blo[1]:bhi[1], blo[0]:bhi[0]]
        data = self.data[:, blo[1]:bhi[1], blo[0]:bhi[0]]
        ss = _sumsq_residuals(np.asarray(model, np.float64), data, mask, self._maskmode,
                              self._region[blo[1]:bhi[1], blo[0]:bhi[0]]) + self._sumsq_outside(blo, bhi)
        lk = getattr(self, '_like_kwargs', {})
        sigma = theta[self.freepar_idx['sigma']][0] if 'sigma' in self.freepar_idx else \
            (lk.get('sigma') or self.noiserms)
        if fracerr is not False and fracerr is not None:
            eps = theta[self.freepar_idx['fracerr']][0] if fracerr is True and 'fracerr' in self.freepar_idx \
                else (float(lk.get('fracerr') or 0.) if fracerr is True else float(fracerr))
            if eps > 0:
                ss, _ = _residuals_fracerr(np.asarray(model, np.float64), data, mask, self._maskmode,
                                           self._region[blo[1]:bhi[1], blo[0]:bhi[0]], sigma ** 2, eps ** 2)
                ss += self._sumsq_outside(blo, bhi) / sigma ** 2
                return float(ss / max(self.nvox, 1))
        return float(ss / (sigma ** 2 * max(self.nvox, 1)))

    def _prior_transform(self, u):
        """ Transform the unit cube into the parameter space using the prior distributions """
        p = np.zeros_like(u)
        for key in self.freepar_idx:
            p[self.freepar_idx[key]] = self.prior_distr[key].ppf(u[self.freepar_idx[key]])
        return p

    # ------------------------------------------------------------------
    def run(self, freepar=None, method=None, threads=None, *, log_like=None, like_kwargs: dict = {},
            sampler_kwargs: dict = {}, run_kwargs: dict = {}, **kwargs):
        """ Sample the posterior.

        Args:
          threads (int):        number of processes for the sampler (default: THREADS if it was
                                given, otherwise 1 = serial)
          freepar (list):       the free parameters (required, here or as the parameter FREE), e.g.
                                ['vrot','vdisp_single','inc_single','sigma'], or a string as FREE in
                                a parameter file ('VROT VDISP_SINGLE INC_SINGLE SIGMA'). Names are
                                case-insensitive (PA is phi, DISP is vdisp)
          method (str):         'dynesty' or 'nautilus' (default: SAMPLER if given, otherwise 'dynesty')
          log_like (callable):  a custom log-likelihood f(theta, **like_kwargs) (default _log_likelihood)
          like_kwargs (dict):   the options of the likelihood, added to LIKEOPTS if given (they are
                                kept in self.like_kwargs, with the defaults; see
                                bbarolo.params.bayes_like_option):
                                likemask ('dilated', default: the data in the mask dilated by
                                dilation; 'none': all the voxels of the region; see _setup_likemask),
                                dilation ((radius in beams, channels), default (1, 3)),
                                noisecorr ('measured', default: the log-likelihood x 1/correlation
                                volume of the noise measured from the voxels without emission;
                                'beam': x 1/beam area in pixels; a number N: x 1/N, N=1 for
                                independent voxels; see _set_likefactor),
                                densprof_smooth (smoothness of the deconvolved density profile,
                                default 0.1), bbres (True: BBarolo's 3DFIT residuals instead of
                                the Gaussian likelihood), sigma and fracerr (fixed noise per voxel
                                and fractional model error, when not free), lambdas (smoothness
                                penalties); with user-defined residuals or log_like, also their
                                own keywords
          sampler_kwargs (dict): arguments for the sampler (e.g. nlive), added to SAMPLEROPTS if
                                given (there, dynamic=False: the static sampler of dynesty; seed=N:
                                a reproducible sampling, also with dynesty)
          run_kwargs (dict):    arguments for the run (e.g. dlogz, maxiter), added to RUNOPTS if given
          kwargs:               verbose (bool), dynamic (bool, dynesty only), and any BBarolo parameter
        """
        moved = [k for k in ('useBBres', 'noisecorr', 'likemask') if k in kwargs]
        if moved:
            raise TypeError(f"BayesianBBarolo.run: {', '.join(moved)} are options of like_kwargs, e.g. "
                            "like_kwargs=dict(likemask='none', noisecorr='beam', bbres=True).")
        params = {k: v for k, v in kwargs.items() if k not in ('verbose', 'dynamic')}
        if params:
            self.update(**params)
        kwargs = {k: v for k, v in kwargs.items() if k in ('verbose', 'dynamic')}
        dflt = self._run_defaults()
        if freepar is None:
            freepar = dflt.get('FREE')
            if freepar is None:
                raise ValueError("BayesianBBarolo.run: give the free parameters, as run(freepar=[...]) or as the "
                                 "parameter FREE (e.g. free='VROT VDISP_SINGLE INC_SINGLE').")
        elif isinstance(freepar, str):
            freepar = self._keyword_option('FREE', freepar)
        else:
            from .params import bayes_name
            freepar = [bayes_name(p) for p in freepar]
        method = str(dflt.get('SAMPLER', 'dynesty') if method is None else method).lower()
        self._method = method
        sampler_kwargs = {**dflt.get('SAMPLEROPTS', {}), **sampler_kwargs}
        if 'dynamic' in sampler_kwargs:              # dynesty: dynamic (default) or static sampler
            kwargs.setdefault('dynamic', bool(sampler_kwargs.pop('dynamic')))
        if method == 'dynesty' and 'seed' in sampler_kwargs:     # (nautilus: its own option seed)
            seed = sampler_kwargs.pop('seed')
            sampler_kwargs.setdefault('rstate', np.random.default_rng(None if seed is None else int(seed)))
        run_kwargs = {**dflt.get('RUNOPTS', {}), **run_kwargs}
        if threads is None:
            threads = int(self._params.get("THREADS", 1))
        # residuals replaced by the user (f._calculate_residuals = my_function)? a custom log-likelihood?
        custom = '_calculate_residuals' in self.__dict__
        own = log_like is not None and log_like != self._log_likelihood and log_like != self._log_likelihood_geometry
        self._set_like_options({**dflt.get('LIKEOPTS', {}), **like_kwargs}, custom=custom or own)
        like_kwargs = self._like_kwargs                      # (those of each call of the log-likelihood)
        useBBres = self.like_kwargs['bbres']
        self._setup(freepar, **kwargs)
        self._log_like_func = log_like if log_like is not None else self._log_likelihood
        verbose = kwargs.get('verbose', True)
        if verbose:
            print(f"\nFitting {self.ndim} parameters: {self.freepar_names}")
            if custom and not useBBres and log_like is None:
                print(f"Likelihood: user-defined residuals (_calculate_residuals); log-likelihood x "
                      f"{self._likefactor_note}")
            elif (not useBBres and log_like is None) or log_like == self._log_likelihood_geometry:
                sg = like_kwargs.get('sigma')
                sg = "free parameter" if 'sigma' in self.freepar_idx else \
                    (f"{sg:.4g} (given)" if sg is not None else f"{self.noiserms:.4g} (measured noise rms)")
                if log_like == self._log_likelihood_geometry:
                    fc = f"1/{1. / self.likefactor2d:.4g} (spatial correlation of the noise)"
                else:
                    fc = self._likefactor_note
                fe = like_kwargs.get('fracerr')
                fe = "; model error: fractional, free parameter 'fracerr'" if 'fracerr' in self.freepar_idx else \
                    (f"; model error: fractional, fracerr = {fe:.4g} (given)" if fe else "")
                if log_like == self._log_likelihood_geometry:
                    print(f"Likelihood: total intensity map ({self._likemask_note}), noise per voxel sigma = "
                          f"{sg}; log-likelihood x {fc}")
                else:
                    print(f"Likelihood: noise per voxel sigma = {sg}{fe}; log-likelihood x {fc}; "
                          f"{self.nvox} voxels ({self._likemask_note})")
            if self.useNorm and not useBBres:
                print(f"Density profile: deconvolved (fit of the total intensity map with the beam-convolved "
                      f"rings, smoothness {self.like_kwargs['densprof_smooth']:g})")
            self.print_priors()
        self._check_likelihood(like_kwargs, custom, verbose)

        # Functions for the sampler and (optional) parallel pool
        pool = None
        _WORKER['parent'] = (self, like_kwargs)
        loglike, prior = _parent_loglike, _parent_prior
        if isMPI:
            _WORKER['f'] = (self, like_kwargs)
            pool = MPIPool()
            if not pool.is_master():
                pool.wait()
                sys.exit(0)
            loglike, prior = _worker_loglike, _worker_prior
        elif threads > 1:
            pool = _make_pool(self, like_kwargs, threads)
            if pool is not None:
                loglike, prior = _worker_loglike, _worker_prior

        toc = time.time()
        if method == 'dynesty':
            if dyn is None:
                raise ImportError("BayesianBBarolo with method='dynesty' requires the package 'dynesty'.")
            Sampler = dyn.DynamicNestedSampler if kwargs.get('dynamic', True) else dyn.NestedSampler
            skw = dict(bound='multi')
            if pool is not None:
                # several proposals per worker keep the processes busy
                skw.update(pool=pool, queue_size=4 * threads)
            skw.update(sampler_kwargs)
            self.sampler = Sampler(loglike, prior, ndim=self.ndim, **skw)
            self.sampler.run_nested(**{'print_progress': verbose, **run_kwargs})
            self.results = self.sampler.results
            samples = self.results.samples
            weights = np.exp(self.results.logwt - self.results.logz[-1])
        elif method == 'nautilus':
            try:
                import nautilus
            except ImportError:
                raise ImportError("BayesianBBarolo requires the package nautilus when method='nautilus'.")
            run_kwargs = dict(run_kwargs)
            if 'dlogz' in run_kwargs and 'f_live' not in run_kwargs:
                run_kwargs['f_live'] = run_kwargs.pop('dlogz')
            skw = dict(pass_dict=False)
            if pool is not None:
                skw['pool'] = pool
            skw.update(sampler_kwargs)
            self.sampler = nautilus.Sampler(prior, loglike, n_dim=self.ndim, **skw)
            self.sampler.run(**{'verbose': verbose, **run_kwargs})
            samples, logw, _ = self.sampler.posterior()
            weights = np.exp(logw - logw.max())
            self.results = None
        else:
            if pool is not None:
                pool.close()
            raise ValueError(f"ERROR! Unknown method '{method}'.")

        dt = time.time() - toc
        dt = f'{dt:.2f} seconds' if dt < 60.00 else f'{dt / 60.00:.2f} minutes'
        vprint(verbose, f'\nSampling with {method} took {dt} to run.')
        if pool is not None:
            pool.close()
            if not isMPI:
                pool.join()
        _WORKER_REGISTRY.pop(id(self), None)

        self.modCalculated = True
        self._set_sampling_stats(samples, weights)
        if verbose and self.ess < 20:
            print(f"WARNING: the posterior is concentrated on {self.ess:.1f} samples (effective sample size), "
                  "so the\n         credible intervals are not reliable: the sampling stopped early "
                  "(maxiter, maxcall), or\n         the likelihood is extremely narrow (check that the noise "
                  "sigma is right, or make it\n         a free parameter).")
        if verbose and self._log_like_func == self._log_likelihood and not self.useBBres and not custom:
            try:
                chi2 = self.chi2_reduced()
                msg = f"Reduced chi-square of the best fit: {chi2:.2f}"
                if 'fracerr' in self.freepar_idx:
                    msg += f" ({self.chi2_reduced(fracerr=True):.2f} with the fitted model error)"
                print(msg)
                if chi2 > 2 and 'sigma' not in self.freepar_idx and 'fracerr' not in self.freepar_idx \
                        and not like_kwargs.get('fracerr'):
                    print("WARNING: the model does not describe the data down to the noise, so the credible "
                          "intervals\n         (computed with sigma = noise rms) underestimate the uncertainties. "
                          "Consider adding\n         'fracerr' (model error) or 'sigma' to the free parameters.")
            except Exception:
                pass

        # Best-fit rings (rings_final1.txt, with the errors from the posterior), with the
        # density profile at the best-fit geometry if DENS is not fitted
        self.outri = self._inri.copy()
        self.outri = self._update_rings(self.outri, self.params)
        if self.useNorm and not self.useBBres:
            self._set_density(self.outri)
        self._galfit.outr = self.outri.object.copy()
        if self.deconvolved_profile_errors() is not None and verbose:
            dp = self.dens_profile
            print(f"Deconvolved density profile: errors from {dp['nsamples']} posterior samples (geometry) and "
                  f"{dp['nmc']} noise realisations (see f.dens_profile, print_stats)")
        with self._errors_in_outputs():
            pass
        return self

    def _check_likelihood(self, like_kwargs, custom, verbose=True):
        """ Warnings before sampling, from the log-likelihood at the centre of the priors: a free
        'sigma' that the user-defined residuals do not use, and (if verbose) log-likelihoods
        below -1e6 (which dynesty prints as -inf, with nan errors). """
        try:
            theta = np.asarray(self._prior_transform(np.full(self.ndim, 0.5)), float)
            logl = self._log_like_func(theta, **like_kwargs)
            for key, what in (('sigma', 'the noise per voxel as\n         the keyword argument sigma (the free '
                                         'parameter, or the measured noise)'),
                              ('fracerr', 'the fractional model error as\n         the keyword argument fracerr')):
                if custom and key in self.freepar_idx and self._log_like_func == self._log_likelihood \
                        and not self.useBBres:
                    t2 = theta.copy()
                    t2[self.freepar_idx[key]] = self.prior_distr[key].ppf(0.9)
                    if self._log_like_func(t2, **like_kwargs) == logl:
                        print(f"WARNING: '{key}' is a free parameter, but the user-defined residuals do not depend "
                              f"on it.\n         _calculate_residuals(model, data, mask, **kwargs) receives {what}.\n")
            if verbose and np.isfinite(logl) and logl < -1e6:
                print(f"NOTE: the log-likelihood at the centre of the priors is {logl:.3g}. dynesty prints "
                      "values below\n      -1e6 as -inf (and the error of logz as nan): they are finite. "
                      "Such values mean\n      that the residuals are much larger than the noise sigma: the "
                      "posterior will be\n      extremely narrow unless 'sigma' or 'fracerr' is free (and used).\n")
        except Exception:
            pass

    def run_geometry(self, freepar=('xpos_single', 'ypos_single', 'inc_single', 'phi_single'), **kwargs):
        """ Fit only the geometry of the model on the total intensity map (Cannubi-style) """
        from .params import bayes_name
        allowed = {'xpos_single', 'ypos_single', 'inc_single', 'phi_single', 'z0_single', 'dens', 'dens_single',
                   'dens_func', 'radmax'}
        freepar = [bayes_name(p) for p in (freepar.replace(',', ' ').split() if isinstance(freepar, str) else freepar)]
        if not (freepar and set(freepar).issubset(allowed)):
            raise ValueError("ERROR! The function run_geometry only accepts the following free parameters: "
                             + ", ".join(sorted(allowed)))
        return self.run(freepar=freepar, log_like=self._log_likelihood_geometry, **kwargs)

    def _set_sampling_stats(self, samples, weights):
        """ Statistics of the sampling results """
        self.samples = samples
        self.weights = weights / np.nansum(weights)
        # effective number of independent samples of the posterior
        self.ess = float(1. / np.nansum(self.weights ** 2))
        # equal-weight posterior samples (fixed seed, so that statistics are reproducible)
        if dyfunc is not None:
            self.samples_equal = dyfunc.resample_equal(self.samples, self.weights, rstate=np.random.default_rng(42))
        else:
            idx = np.random.default_rng(42).choice(len(samples), size=len(samples), p=self.weights)
            self.samples_equal = samples[idx]
        self.modCalculated = True
        self.quantiles = np.quantile(self.samples_equal, [0.16, 0.50, 0.84], axis=0)
        self.params = self.quantiles[1]

    def _update_rings(self, rings, theta):
        """ Update rings with the parameters theta """
        freepar_idx = copy.copy(self.freepar_idx)
        if 'radmax' in freepar_idx:
            rings.r['radii'][-1] = theta[freepar_idx['radmax']][0]
        whatFunctional = np.unique([key.split('_')[0] for key in freepar_idx if '_p' in key])
        for p in whatFunctional:
            keys = [key for key in freepar_idx if key.startswith(p + '_p')]
            fvalue = [theta[freepar_idx[key]] for key in keys]
            rings.modify_parameter(p, self.funcs[p](rings.r['radii'], *fvalue))
            for key in keys:
                freepar_idx.pop(key)
        for key in freepar_idx:
            if key in ('sigma', 'radmax', 'fracerr'):
                continue
            pvalue = theta[freepar_idx[key]]
            if len(pvalue) == 1:
                pvalue = pvalue[0]
            rings.modify_parameter(key, pvalue)
        rings.make_object()
        return rings

    def _set_density(self, rings):
        """ Sets the densities of the rings, when DENS is not fitted, to the deconvolved profile:
        the densities whose beam-convolved model reproduces the total intensity map of the
        data (see _deconvolved_profile). The original module used the face-on average of the
        observed map in each ring (Ellprof), which the beam smooths twice (in the data and in
        the model): for marginally resolved disks, the inner rings were too faint and the outer
        ones too bright, which biased INC and VDISP. That profile is now only an output
        (densprof.txt, see _observed_profile). """
        dens = self._deconvolved_profile(rings)
        if dens is None:
            raise ValueError("BAYESIAN3DFIT ERROR: too few pixels with data in the likelihood region to compute "
                             "the density profile.\n  Check the mask and the rings, or fit DENS.")
        rings.modify_parameter("dens", dens, makeobj=True)

    def _observed_profile(self, rings):
        """ The Ellprof engine with the radial profile of the total intensity map of the voxels
        of the likelihood along the rings: the observed profile of the outputs (densprof.txt,
        the plots and the DENS_OBS column of densprof_deconvolved.txt). Not used in the fit. """
        if self._ellprof is None:
            from .ellprof import Ellprof
            self._ellprof = Ellprof(self._cube_obj, rings.object)
            self._ellprof.im.array = self._likemask_map()
        self._ellprof.update_rings(rings.object)
        self._ellprof.RadialProfile()
        return self._ellprof

    def _deconvolved_profile(self, rings):
        """ Densities of the rings whose model reproduces the total intensity map of the data
        (data_mom0: the sum of the data in the voxels of the likelihood, see _setup_likemask).
        The total intensity map of the model is linear in the densities, sum_r dens_r K_r,
        where K_r is the map of ring r with unit density convolved with the beam
        (DiskModel.ring_maps: exact for the ANALYTIC model, the expectation of the MC one), and
        it depends only on the geometry, not on the kinematics. The densities are the
        non-negative weighted least-squares solution in the pixels of the likelihood region
        with data, weights 1/map_rms^2. This corrects the profile for the beam, unlike the
        average of the observed map in each ring.

        Two terms make it robust on real galaxies:
          * an exterior annulus, from the outer edge of the rings to two beams beyond, with
            its own density: it takes the emission beyond the last ring that falls in the
            likelihood region, which the last ring would otherwise absorb (its density is
            not used);
          * a smoothness prior: rings narrower than the beam are nearly degenerate once
            convolved, and noise or non-axisymmetric structure make the solution oscillate
            from ring to ring. The squared second differences of the profile (including the
            exterior annulus) are penalised with a weight densprof_smooth (default 0.1) times
            the mean data weight of a ring (trace(A^T A) / trace(L^T L)), independent of the
            units and of the S/N (0 = no prior).
        Returns None if there are too few pixels. """
        sysm = self._profile_system(rings)
        if sysm is None:
            return None
        A, b, _, Rc = sysm
        return self._solve_profile(A, b, Rc)[:rings.nr]

    def _profile_system(self, rings):
        """ The weighted least-squares system of the deconvolved profile (see
        _deconvolved_profile): (A, b, ok, Rc), with A (npix, nr+1) the beam-convolved maps of
        the rings and of the exterior annulus with unit density and b the total intensity map
        of the data, both divided by the map noise, in the pixels ok (a mask of the likelihood
        box), and Rc the radii of the rings and of the exterior annulus. None if there are
        fewer than 2 nr pixels. """
        from .galmod import outer_edge
        from .rings import Rings
        Lb, Hb = self._likebox
        model = self._galfit.model
        r = rings.object
        K = model.ring_maps(r, Lb, Hb)
        h = self._cube_obj.Head()
        R = np.asarray(rings.r['radii'], float)
        w = 2. * abs(h.Bmaj()) * 3600. if h.Bmaj() else 2. * float(np.max(np.diff(R), initial=R[-1]))
        rext = outer_edge(r, model.ringmode) + w / 2.
        ext = Rings().setRings(radii=[rext], xpos=r.xpos[-1], ypos=r.ypos[-1], vsys=r.vsys[-1], vrot=r.vrot[-1],
                               vdisp=r.vdisp[-1], z0=r.z0[-1], inc=r.inc[-1], phi=r.phi[-1], dens=1., radsep=w)
        K = np.concatenate([K, model.ring_maps(ext, Lb, Hb, ringmode="CENTRED")])
        sl = (slice(Lb[1], Hb[1]), slice(Lb[0], Hb[0]))
        y, s = self.data_mom0[sl], self.map_rms[sl]
        ok = self._region[sl] & (self._nvox_map[sl] > 0) & (s > 0) & np.isfinite(y)
        if ok.sum() < 2 * rings.nr:
            return None
        return K[:, ok].T / s[ok, None], y[ok] / s[ok], ok, np.r_[R, rext]

    def _solve_profile(self, A, b, radii):
        """ Non-negative least-squares solution of A dens = b (all the columns, including the
        exterior annulus), with the smoothness prior of densprof_smooth on the profile at the
        radii (see _deconvolved_profile). """
        from scipy.optimize import nnls
        n = A.shape[1]
        alpha = float(self.like_kwargs.get('densprof_smooth') or 0.)
        if alpha > 0 and n >= 3:
            R = np.asarray(radii, float)
            L = np.zeros((n - 2, n))
            for i in range(1, n - 1):                   # second differences (any spacing)
                d1, d2 = R[i] - R[i - 1], R[i + 1] - R[i]
                L[i - 1, i - 1:i + 2] = np.array([1. / d1, -1. / d1 - 1. / d2, 1. / d2]) * 2. / (d1 + d2)
            L *= math.sqrt(alpha * np.sum(A * A) / max(np.sum(L * L), 1e-300))
            A, b = np.vstack([A, L]), np.r_[b, np.zeros(n - 2)]
        q, r = np.linalg.qr(A)                         # nnls on the small (n x n) system
        return nnls(r, q.T @ b)[0]

    def deconvolved_profile_errors(self, nsamples=200, nmc=200, seed=0):
        """ Uncertainties of the deconvolved density profile of the best fit (DENS not fitted),
        from two sources combined in quadrature:

          geometry: the profile computed for nsamples samples of the posterior (16th-84th
                    percentiles around their median); zero if the geometry is not fitted;
          noise:    the spread of the profile fitted, at the best-fit geometry, to nmc
                    realisations of the best-fit total intensity map plus noise with the map
                    rms of each pixel and the spatial correlation of the noise (Gaussian, with
                    the correlation area 1/likefactor2d pixels).

        Called at the end of run() and load_results(). Sets and returns self.dens_profile, a
        dict with radii, dens (the profile of the best fit, as in outri and rings_final1.txt),
        err_lo, err_hi (total 1-sigma errors), err_geom_lo, err_geom_hi, err_noise_lo,
        err_noise_hi, nsamples and nmc; None if DENS is fitted. The errors are also written in rings_final1.txt (E_DENS) and
        densprof_deconvolved.txt (write_bestmodel). """
        from scipy.ndimage import gaussian_filter
        self.dens_profile = None
        if not (getattr(self, 'useNorm', False) and not self.useBBres and self.outri is not None):
            return None
        R = np.asarray(self.outri.r['radii'], float)
        d0 = np.asarray(self.outri.r['dens'], float)
        nr = len(R)
        rng = np.random.default_rng(seed)

        def spread(prof):
            p16, p50, p84 = np.percentile(np.asarray(prof), [16, 50, 84], axis=0)
            return p50 - p16, p84 - p50
        # geometry: the profile for samples of the posterior
        geo_lo = geo_hi = np.zeros(nr)
        ngeo = 0
        if self.update_prof and getattr(self, 'samples_equal', None) is not None and nsamples > 0:
            S = np.asarray(self.samples_equal)
            idx = rng.choice(len(S), min(int(nsamples), len(S)), replace=False)
            prof = [p for p in (self._deconvolved_profile(self._update_rings(self._inri.copy(), S[i])) for i in idx)
                    if p is not None]
            if len(prof) > 1:
                geo_lo, geo_hi = spread(prof)
                ngeo = len(prof)
        # noise: refits of the best-fit map plus correlated noise, at the best-fit geometry
        noi_lo = noi_hi = np.zeros(nr)
        sysm = self._profile_system(self.outri) if nmc > 0 else None
        if sysm is not None:
            A, b, ok, Rc = sysm
            bmod = A @ self._solve_profile(A, b, Rc)          # the best-fit map (with the exterior annulus)
            sk = math.sqrt(max(1. / self.likefactor2d, 1.) / (4. * math.pi))   # Gaussian ACF of that area
            prof = []
            for _ in range(int(nmc)):
                g = rng.standard_normal(ok.shape)
                if sk > 0.3:
                    g = gaussian_filter(g, sk, mode='wrap')
                    g /= g.std()
                prof.append(self._solve_profile(A, bmod + g[ok], Rc)[:nr])
            noi_lo, noi_hi = spread(prof)
        self.dens_profile = dict(radii=R, dens=d0, err_lo=np.hypot(geo_lo, noi_lo), err_hi=np.hypot(geo_hi, noi_hi),
                                 err_geom_lo=geo_lo, err_geom_hi=geo_hi, err_noise_lo=noi_lo, err_noise_hi=noi_hi,
                                 nsamples=ngeo, nmc=int(nmc) if sysm is not None else 0)
        return self.dens_profile

    # ------------------------------------------------------------------
    def _normalize_model(self, model, data, mask, norm='AZIM'):
        """ Normalization of the best model (called but not defined in the original module).

        'AZIM' (default): no change, i.e. the azimuthally-normalized model used in the
        likelihood; 'LOCAL': pixel-by-pixel normalization to the flux of the masked data.
        """
        if str(norm).upper() != 'LOCAL':
            return model
        modSum = np.nansum(model, axis=0)
        obsSum = np.nansum(data * mask, axis=0)
        with np.errstate(invalid='ignore', divide='ignore'):
            factor = np.where(modSum != 0, obsSum / modSum, 0.)
        return model * factor[None]

    def write_bestmodel(self, plots=True, norm='AZIM', **kwargs):
        """ Write the best model, its moment maps, PV diagrams and plots in the output folder.

        norm: 'AZIM' (model as used in the likelihood) or 'LOCAL' (pixel-by-pixel normalized),
        used only when the density is not fitted.

        The DENS column of rings_final1.txt is the density profile of the model (deconvolved,
        at the best-fit geometry, if DENS is not fitted), and densprof.txt the observed profile
        (the average of the total intensity map in each ring). If DENS is not fitted,
        densprof_deconvolved.txt lists both profiles at full precision, with the errors.
        """
        if not self.modCalculated:
            print("Sampler has not been run yet. Please run run() before running this function.")
            return
        verbose = kwargs.get('verbose', True)
        vprint(verbose, "\nWriting the best model to output directory ...", end=' ', flush=True)
        if self.useBBres:
            with self._errors_in_outputs():
                self._galfit.writeModel("AZIM", plots)
        else:
            if self.useNorm:
                self._set_density(self.outri)
            ell = self._observed_profile(self.outri)        # densprof.txt (outputs only)
            model = self._calculate_model(self.outri, fullcube=True)[0]
            if model.ndim == 4:             # (non-convolved, convolved) models
                model = model[-1]
            if self.useNorm:
                model = self._normalize_model(model, self.data, self.mask, norm)
            from .cube import Cube
            c = self._cube_obj
            mod = Cube.fromArray(np.asarray(model, dtype=np.float32), c.Head(), c.pars())
            with self._errors_in_outputs():
                self._galfit.writeOutputs(mod, ell, plots)
            if self.useNorm:
                self._write_deconvolved_profile()
        vprint(verbose, "Done!")

    def _write_deconvolved_profile(self):
        """ densprof_deconvolved.txt in the output folder: radius, the density profile of the
        model (deconvolved, at the best-fit geometry) with its 1-sigma errors (see
        deconvolved_profile_errors), the observed profile (face-on average of the total
        intensity map in each ring, the profile of the original module), and the errors from
        the geometry and from the noise, in the units of DENS. Needs the Ellprof profile along
        self.outri (see write_bestmodel). """
        fn = os.path.join(self._cube_obj.pars().outFolder, "densprof_deconvolved.txt")
        r = self.outri.r
        nr = self.outri.nr
        dp = getattr(self, 'dens_profile', None)
        z = np.zeros(nr)
        e = {k: (dp[k] if dp is not None else z) for k in ('err_lo', 'err_hi', 'err_geom_lo', 'err_geom_hi',
                                                            'err_noise_lo', 'err_noise_hi')}
        with open(fn, "w") as f:
            f.write("# Density profile of the model (DENS_MODEL: the densities whose beam-convolved model\n"
                    "# reproduces the total intensity map) and the observed profile (DENS_OBS: the face-on\n"
                    "# average of the map in each ring, the profile of the original module).\n"
                    "# E_LO, E_HI: 1-sigma errors of DENS_MODEL, the errors from the geometry (posterior samples:\n"
                    f"# {dp['nsamples'] if dp else 0}) and from the noise (realisations: {dp['nmc'] if dp else 0}) "
                    "added in quadrature.\n")
            f.write(f"{'#RAD(arcs)':<12s}" + "".join(f"{c:>14s}" for c in (
                'DENS_MODEL', 'E_LO', 'E_HI', 'DENS_OBS', 'E_GEOM_LO', 'E_GEOM_HI', 'E_NOISE_LO', 'E_NOISE_HI')) + "\n")
            for i in range(nr):
                vals = (r['dens'][i], e['err_lo'][i], e['err_hi'][i], abs(self._ellprof.getSurfDensFaceOn(i)),
                        e['err_geom_lo'][i], e['err_geom_hi'][i], e['err_noise_lo'][i], e['err_noise_hi'][i])
                f.write(f"{r['radii'][i]:<12.3f}" + "".join(f"{v:>14.6e}" for v in vals) + "\n")

    # Ring parameters in the order of the 3DFIT tables and error columns
    _RING_NAMES = ('vrot', 'vdisp', 'dens', 'z0', 'inc', 'phi', 'xpos', 'ypos', 'vsys', 'vrad')

    def _posterior_errors(self):
        """ The ring parameters that were sampled (flags in the order of _RING_NAMES) and
        their 1-sigma errors from the posterior, as the 3DFIT error columns:
        errors[ring, 0 = lower (negative) / 1 = upper (positive), k-th free parameter]. """
        base = {k.split('_')[0] for k in self.freepar_idx}
        if getattr(self, 'dens_profile', None) is not None:
            base.add('dens')                          # errors of the deconvolved profile
        mpar = [n in base for n in self._RING_NAMES]
        free = [n for n, m in zip(self._RING_NAMES, mpar) if m]
        errors = np.zeros((self.outri.nr, 2, len(free)))
        for k, n in enumerate(free):
            lo, hi = self.ring_errors(n)
            errors[:, 0, k] = -np.clip(np.nan_to_num(np.asarray(lo, float)), 0, None)
            errors[:, 1, k] = np.clip(np.nan_to_num(np.asarray(hi, float)), 0, None)
        return mpar, errors

    @contextlib.contextmanager
    def _errors_in_outputs(self):
        """ Writes rings_final1.txt with the errors from the posterior. Inside the context,
        the free parameters of the sampling (instead of FREE) and the error columns are used
        by the outputs of the 3DFIT engine, so that the plots (_parameters.pdf) show the
        errors. """
        gf = self._galfit
        old = (gf.mpar, gf.nfree, gf.par.flagERRORS)
        mpar, errors = self._posterior_errors()
        gf.mpar, gf.nfree, gf.par.flagERRORS = mpar, int(sum(mpar)), bool(any(mpar))
        try:
            gf.writeRingFile("rings_final1.txt", self.outri.object, errors)
            yield
        finally:
            gf.mpar, gf.nfree, gf.par.flagERRORS = old

    # ------------------------------------------------------------------
    def corner_plot(self, samples=None, labels=None, truths=None, quantiles=[0.15865, 0.50, 0.84135],
                    plot_dynesty=False, **kwargs):
        """ Corner plot of the posterior distributions (needs 'corner', or dynesty's plotting) """
        if samples is None and not self.modCalculated:
            print("Sampler has not been run yet. Please run run() before running this function "
                  "or load samples with load_results().")
            return
        defaults = dict(max_n_ticks=5, color='darkblue', show_titles=True)
        if self.results is None:
            plot_dynesty = False
        if plot_dynesty:
            from dynesty import plotting as dyplot
            samples = self.results
            cornerplot = dyplot.cornerplot
        else:
            try:
                import corner
            except ImportError:
                raise ImportError("BayesianBB requires the package 'corner' to produce corner plots.")
            cornerplot = corner.corner
            defaults['plot_datapoints'] = False
        if samples is None:
            samples = self.samples_equal
        if labels is None:
            labels = self.freepar_names
        if truths is not None:
            kwargs['truths'] = truths
        c = cornerplot(samples, title_quantiles=quantiles, quantiles=quantiles, labels=labels,
                       label_kwargs=dict(fontsize=20), **{**defaults, **kwargs})
        return c[0] if plot_dynesty else c

    _ERR_ALIASES = dict(pa='phi', disp='vdisp')

    def ring_errors(self, param=None):
        """ Lower and upper 1-sigma errors of a ring parameter in each ring of the best fit
        (self.outri), as two arrays of positive values (lower, upper):

          sampled per ring or '_single':  the 16th and 84th percentiles of the posterior;
          '_func':                        the percentiles of the function of radius over the
                                          posterior samples;
          DENS not fitted:                the errors of the deconvolved
                                          profile (geometry and noise, see
                                          deconvolved_profile_errors);
          otherwise (fixed):              zeros.

        param: vrot, vdisp (or disp), dens, z0, inc, phi (or pa), xpos, ypos, vsys, vrad,
        vvert, dvdz, zcyl, case-insensitive. Without param: a dictionary with the errors of
        vrot, vdisp, dens, z0, inc, phi, xpos, ypos, vsys and vrad. These are the errors of
        rings_final1.txt and of plot_profiles. Needs run() or load_results(). """
        if not self.modCalculated or self.outri is None:
            raise ValueError("ring_errors: run the sampling (run) or load the results (load_results) first.")
        if param is None:
            return {n: self.ring_errors(n) for n in self._RING_NAMES}
        a = str(param).strip().lower()
        a = self._ERR_ALIASES.get(a, a)
        if a not in PARAMS or a == 'radii':
            raise ValueError(f"ring_errors: unknown ring parameter {param!r}.")
        nr = self.outri.nr
        if a in self.freepar_idx:
            q = self.quantiles[:, self.freepar_idx[a]]
            lo, hi = q[1] - q[0], q[2] - q[1]
        else:
            keys = [k for k in self.freepar_idx if k.startswith(a + '_p')]
            if keys:
                R = self.outri.r['radii']
                prof = np.array([self.funcs[a](R, *[s[self.freepar_idx[k]] for k in keys])
                                 for s in self.samples_equal])
                q = np.quantile(prof, [0.16, 0.5, 0.84], axis=0)
                lo, hi = self.outri.r[a] - q[0], q[2] - self.outri.r[a]
            elif a == 'dens' and getattr(self, 'dens_profile', None) is not None:
                lo, hi = self.dens_profile['err_lo'], self.dens_profile['err_hi']      # deconvolved profile
            else:
                lo = hi = 0.
        lo = np.clip(np.broadcast_to(np.asarray(lo, float), (nr,)), 0., None)
        hi = np.clip(np.broadcast_to(np.asarray(hi, float), (nr,)), 0., None)
        return lo, hi

    def plot_profiles(self, **kwargs):
        """ Plot the best-fit radial profiles of the ring parameters (with 1-sigma errors) """
        import matplotlib.pyplot as plt
        if not self.modCalculated:
            print("Sampler has not been run yet. Please run run() before running this function "
                  "or load samples with load_results().")
            return
        rings = self.outri.r
        toplot = ['vrot', 'vdisp', 'dens', 'inc', 'phi', 'z0', 'xpos', 'ypos', 'vsys']
        labels = [r'$V_\mathrm{rot}$ (km/s)', r'$\sigma$ (km/s)', 'Density', 'Inclination (deg)',
                  'Position angle (deg)', r'$z_0$ (arcsec)', 'Xpos (pix)', 'Ypos (pix)', r'$V_\mathrm{sys}$ (km/s)']
        fig, axes = plt.subplots(3, 3, figsize=(12, 9), sharex=True,
                                 gridspec_kw=dict(height_ratios=[1.6, 1, 1], hspace=0.05, wspace=0.3))
        for k, a in enumerate(axes.flat):
            a.tick_params(direction='in', top=True, right=True)
            if k >= 6:
                a.set_xlabel('Radius (arcsec)', fontsize=14)
        axes[0, 0].set_xlim(0, rings['radii'][-1] * 1.05)
        ax = axes.ravel()
        for i, a in enumerate(toplot):
            err_low, err_up = self.ring_errors(a)
            ax[i].errorbar(rings['radii'], rings[a], yerr=[np.asarray(err_low), np.asarray(err_up)], fmt='o',
                           **{'c': '#B22222', 'ecolor': '#B22222', **kwargs})
            ax[i].set_ylabel(labels[i])
            lo, hi = np.nanmin(rings[a] - err_low), np.nanmax(rings[a] + err_up)
            if hi > lo:
                pad = 0.1 * (hi - lo)
                ax[i].set_ylim(lo - pad, hi + pad)
            if a == 'vrot':
                ax[i].set_ylim(0, None)
        return fig

    def print_stats(self, quantiles=[0.15865, 0.50, 0.84135]):
        """ Print the median and 1-sigma errors of the posteriors (and the maximum likelihood) """
        if not self.modCalculated:
            print("Sampler has not been run yet. Please run run() before running this function.")
            return
        if not isIterable(quantiles) or len(quantiles) != 3:
            raise ValueError("ERROR! Percentiles must be a list of three values.")
        results = self.results
        lk = getattr(self, '_like_kwargs', {})          # the likelihood options of run()
        logl = self._log_like_func(self.params, **lk) if hasattr(self, '_log_like_func') else \
            self._log_likelihood(self.params, **lk)
        print(f"\nBest-fit parameters (log_likelihood = {logl:.2f}):")
        if getattr(self, 'ess', np.inf) < 20:
            print(f"  (effective sample size of the posterior: {self.ess:.1f}: the errors are not reliable)")
        q = np.quantile(self.samples_equal, quantiles, axis=0)
        for i in range(len(self.freepar_names)):
            d = np.diff(q[:, i])
            print("  %10s = %10.3f %+10.3f %+10.3f" % (self.freepar_names[i], q[1, i], -d[0], d[1]))
        dp = getattr(self, 'dens_profile', None)
        if dp is not None:
            print("\nDeconvolved density profile (DENS, not fitted; errors from the geometry and the noise):")
            for i in range(len(dp['radii'])):
                print("  R = %8.2f   %10.4g %+10.3g %+10.3g" % (dp['radii'][i], dp['dens'][i], -dp['err_lo'][i],
                                                                dp['err_hi'][i]))
        if results is not None:
            max_index = results.logl.argmax()
            print(f"\nMaximum logl: {results.logl.max()} at:")
            for i in range(len(self.freepar_names)):
                print("  %10s = %10.3f " % (self.freepar_names[i], results.samples[max_index][i]))

    def print_priors(self):
        """ Print the priors for each parameter """
        print("\nPrior distributions for free parameters:")
        for key in self.freepar_idx:
            print(f"{key:>10s} = {self.prior_distr[key].dist.name:10s}", self.prior_distr[key].kwds)
        print("\n")

    def save_results(self, outfile):
        """ Save the results as a dictionary in a numpy (.npy) file (with the options of the
        likelihood, like_kwargs; those that cannot be saved, e.g. functions, are left out). """
        import pickle
        like = {}
        for k, v in self.like_kwargs.items():
            try:
                pickle.dumps(v)
                like[k] = v
            except Exception:
                pass
        tosave = dict(results=self.results, samples=self.samples, weights=self.weights,
                      freepar_inp=self._freepar_inp, freepar_names=self.freepar_names, freepar_idx=self.freepar_idx,
                      inrings=self._inri.r, options={k: str(v) if isinstance(v, _FromFile) else v
                                                     for k, v in self.parameters.items()},
                      priors=self.priors,
                      prior_distr=self.prior_distr, lambdas=self.lambdas, like_kwargs=like)
        np.save(outfile, tosave)

    def load_results(self, inputfile, sampler='dynesty'):
        """ Load results saved with save_results() (dynesty or nautilus). """
        r = np.load(inputfile, allow_pickle=True).item()
        rr = r['inrings']
        self._galfit = self._ellprof = None
        # options of the saved fit (also files written by the original module: lower-case keys)
        from .keywords import lookup
        opts = {k: v for k, v in dict(r.get('options') or {}).items()
                if lookup(k) is not None and lookup(k).name not in ('FITSFILE', 'FITSLIST', '3DFIT')}
        rings = {RING_PARAMS[k]: np.asarray(rr[k], dtype=float) for k in PARAMS}
        rings['NRADII'] = None
        rings['RADSEP'] = None
        opts = {lookup(k).name: v for k, v in opts.items()}
        opts.update(rings)
        super(BayesianBBarolo, self).update(**opts)
        self._cube_obj = None
        self._init_rings()
        for k in ('priors', 'prior_distr', 'lambdas'):
            if r.get(k) is not None:
                getattr(self, k).update(r[k])
        # the options of the likelihood of the saved fit (in files saved before like_kwargs existed:
        # separate entries; without noisecorr, the beam area)
        like = r.get('like_kwargs')
        if like is None:
            like = dict(noisecorr=r.get('noisecorr', 'beam'),
                        likemask=r['likemask'] if r.get('likemask') in ('dilated', 'none') else 'dilated',
                        dilation=tuple(r.get('likemask_dilation', (1., 3))),
                        densprof_smooth=r.get('densprof_smooth', 0.1))
        self._set_like_options(like, custom=True)
        self._setup(r['freepar_inp'])
        if sampler == 'dynesty' and r.get('results') is not None:
            self.results = r['results']
            weights = np.exp(self.results.logwt - self.results.logz[-1])
            self._set_sampling_stats(self.results.samples, weights)
        elif sampler in ('dynesty', 'nautilus') and r.get('weights') is not None:
            self.results = None
            self._set_sampling_stats(r['samples'], r['weights'])
        else:
            raise ValueError(f"Error: cannot load results for sampler {sampler} from {inputfile}")
        self.outri = self._inri.copy()
        self.outri = self._update_rings(self.outri, self.params)
        if self.useNorm and not self.useBBres:
            self._set_density(self.outri)
        self._galfit.outr = self.outri.object.copy()
        self.deconvolved_profile_errors()


# ----------------------------------------------------------------------
# The BAYESIAN3DFIT task of parameter files (BBarolo -p, bbarolo.run)
# ----------------------------------------------------------------------
def _task_fitter(c, par):
    """ The BayesianBBarolo of the BAYESIAN3DFIT task: the parameters of par as given (but the
    flags of the tasks and the input files), the output folder of c, and the data of c. """
    from .keywords import lookup
    p = c.pars()
    prm = {}
    for k, v in getattr(par, "_given", {}).items():
        kw = lookup(k)
        if kw is None or kw.flag or k in ("FITSFILE", "FITSLIST", "AUTO", "OUTFOLDER"):
            continue
        prm[k] = _FromFile(v)
    prm["OUTFOLDER"] = p.outFolder
    f = BayesianBBarolo(None, **prm)
    f._set_cube(p.imageFile)
    cb = f._load_cube()
    if cb.array.shape == c.array.shape and not np.array_equal(cb.array, c.array, equal_nan=True):
        cb.array[...] = c.array                   # changed by the tasks run before (CONTSUB, CHECKCUBE)
    f._init_rings()
    return f


def run_task(c, par):
    """ The BAYESIAN3DFIT task: samples the posterior with the parameters of par (a checked
    Param; its values as given, see Param._given) on the cube c read by BBarolo (with the
    changes of the tasks run before it, e.g. CONTSUB). The free parameters are FREE (required),
    the sampler SAMPLER, with SAMPLEROPTS and RUNOPTS, and the priors PRIORS (the others: the
    defaults of BayesianBBarolo); the sampler runs with THREADS processes.

    Then it writes in OUTFOLDER the outputs of BayesianBBarolo.write_bestmodel (best model,
    maps, PV diagrams, rings_final1.txt with the errors, densprof.txt and, when DENS is not
    fitted, densprof_deconvolved.txt; with PLOTS the plots of 3DFIT) and the samples
    (<prefix>_<sampler>_results.npy, see load_results); with PLOTS also the corner plot
    (<prefix>_corner.pdf) and the radial profiles (<prefix>_profiles.pdf); with VERBOSE it
    prints the statistics of the posterior (print_stats). Returns the BayesianBBarolo. """
    from .params import bayes_options
    f = _task_fitter(c, par)
    p = c.pars()
    gf = p.gf
    verbose = bool(p.verbose)
    method = gf.SAMPLER.strip().lower()
    run_kwargs = {}
    if method == 'dynesty' and not p.showbar and 'print_progress' not in bayes_options(gf.RUNOPTS):
        run_kwargs['print_progress'] = False      # e.g. LOGFILE: no progress line
    f.run(freepar=gf.FREE, method=method, threads=p.threads, verbose=verbose, run_kwargs=run_kwargs)

    if p.plots >= 0:                              # (as 3DFIT: PLOTS=-1 writes no model)
        f.write_bestmodel(plots=p.plots > 0, verbose=verbose)
    cp = f._cube_obj.pars()
    of, pre = cp.outFolder, cp.outPrefix
    fn = os.path.join(of, f"{pre}_{method}_results.npy")
    f.save_results(fn)
    vprint(verbose, f"Posterior samples written in {fn} (see BayesianBBarolo.load_results)")
    if p.plots > 0:
        import logging
        import warnings
        import matplotlib.pyplot as plt
        for name, make in (("corner", lambda: f.corner_plot(plot_dynesty=(method == 'dynesty'))),
                           ("profiles", f.plot_profiles)):
            out = os.path.join(of, f"{pre}_{name}.pdf")
            root = logging.getLogger()
            level = root.level
            try:
                # (no warnings of the plotting packages, e.g. for a posterior on very few samples)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    root.setLevel(logging.ERROR)
                    fig = make()
                    fig.savefig(out, bbox_inches='tight')
                plt.close(fig)
                vprint(verbose, f"Plot written in {out}")
            except Exception as e:
                plt.close('all')
                print(f"BAYESIAN3DFIT WARNING: no {name} plot ({e})")
            finally:
                root.setLevel(level)
    if verbose:
        f.print_stats()
    return f
