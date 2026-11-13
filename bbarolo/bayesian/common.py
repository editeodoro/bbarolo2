"""Small helpers of Bayesian3DFIT."""

import contextlib
import weakref
import numpy as np
from scipy import stats

# Threads of BLAS and OpenMP during the sampling and the errors of the density profile (None: no limit)
BLAS_THREADS = 1


# Apple Accelerate (the BLAS of numpy on macOS): threadpoolctl does not know it, and its threads (GCD) do not survive a
# fork: a worker of a fork-based pool whose parent ran a threaded BLAS product crashed in its first one (SIGSEGV in
# cblas_dgemv -> dispatch_apply), and the pool waited forever. macOS 15+ has a switch, BLASSetThreading (vecLib
# thread_api.h: BLAS_THREADING_MULTI_THREADED = 0, BLAS_THREADING_SINGLE_THREADED = 1).
_ACCELERATE_SINGLE = 1
_accelerate_lib = []


def _accelerate():
    """ Accelerate's BLASGetThreading and BLASSetThreading (a ctypes library), or None (not macOS, or macOS < 15). """
    if not _accelerate_lib:
        lib = None
        import sys
        if sys.platform == "darwin":
            try:
                import ctypes
                lib = ctypes.CDLL("/System/Library/Frameworks/Accelerate.framework/Accelerate")
                lib.BLASGetThreading.restype, lib.BLASGetThreading.argtypes = ctypes.c_uint, []
                lib.BLASSetThreading.restype, lib.BLASSetThreading.argtypes = ctypes.c_int, [ctypes.c_uint]
            except (OSError, AttributeError):
                lib = None
        _accelerate_lib.append(lib)
    return _accelerate_lib[0]


class _BlasLimits:
    """ BLAS and OpenMP limited to n threads from the creation of the object (as threadpoolctl's threadpool_limits),
    restored by restore() or at the end of a with block: threadpoolctl (OpenBLAS, MKL, OpenMP), and Accelerate on
    macOS when n is 1 (it has only single and multi-threaded). """

    def __init__(self, n):
        self._tp = self._acc = None
        try:
            from threadpoolctl import threadpool_limits
            self._tp = threadpool_limits(limits=int(n))
        except ImportError:
            pass
        acc = _accelerate()
        if acc is not None and int(n) == 1:
            self._acc = (acc, int(acc.BLASGetThreading()))
            acc.BLASSetThreading(_ACCELERATE_SINGLE)

    def restore(self):
        if self._acc is not None:
            acc, old = self._acc
            acc.BLASSetThreading(old)
            self._acc = None
        if self._tp is not None:
            self._tp.restore_original_limits()
            self._tp = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.restore()
        return False


def blas_threads():
    """ A context in which BLAS and OpenMP use BLAS_THREADS threads, from its creation: with threadpoolctl (if it is
    installed) and, on macOS, Accelerate single-threaded when BLAS_THREADS is 1 (see _accelerate). The linear algebra
    of the likelihood (the deconvolved density profile) works on small matrices, for which many threads are slower:
    with MKL on 224 cores, the likelihood with a free geometry took 1.6-2.2 times longer, and the processes of a
    parallel sampling competed for the cores. The process pools fork in it, so that the workers inherit one thread
    (Accelerate's threads do not survive a fork). The values do not change. """
    if BLAS_THREADS is None:
        return contextlib.nullcontext()
    return _BlasLimits(BLAS_THREADS)

def show_bar(fitter):
    """ SHOWBAR for the progress bars of a fit (Bayesian3DFIT, BayesianWind): the value of its task (_task_showbar, set
    by the task of parameter files: off with LOGFILE), otherwise the parameter as given (default true). The engine of
    the fit runs with VERBOSE off, which turns its own SHOWBAR off. """
    v = getattr(fitter, "_task_showbar", None)
    if v is not None:
        return bool(v)
    s = fitter._params.get("SHOWBAR")
    if s is None:
        return True
    from ..tasks import build_param
    return bool(build_param({"SHOWBAR": s}).showbar)


# A function to print only if verbose is True
vprint = lambda verbose, *args, **kwargs: print(*args, **kwargs) if verbose else None


def banner(task=None):
    """The line that opens (with the name of the task) or closes (task None) the output of a fit, 69 characters wide
    as that of 3DFIT."""
    return "\n" + f" {task} ".center(69, "=") if task else "=" * 69 + "\n"


def isIterable(p):
    """Check if p is an iteratable (list,tuple or numpy array). """
    return isinstance(p, (list, tuple, np.ndarray))


def get_distribution(distr_name, *params, **kwargs):
    """Create a frozen scipy.stats distribution given its name and parameters."""
    try:
        distr = getattr(stats, distr_name)
        return distr(*params, **kwargs)
    except AttributeError:
        raise ValueError(f"ERROR: Distribution '{distr_name}' not found in scipy.stats.")


def resample_equal(samples, weights, rng):
    """ Equal-weight samples of the weighted samples (samples, weights; rng a numpy Generator): the systematic
    resampling of dynesty's utils.resample_equal (Hol, Schon & Gustafsson 2006), the same samples in the same order
    for the same generator, without needing dynesty. Each sample appears floor or ceil(weight x number) times, then
    the samples are shuffled. """
    cumulative_sum = np.cumsum(weights)
    cumulative_sum /= cumulative_sum[-1]
    n = len(weights)
    positions = (rng.random() + np.arange(n)) / n
    idx = np.searchsorted(cumulative_sum, positions, side='right')
    return rng.permutation(np.asarray(samples)[idx])


_FAST_PPF = weakref.WeakKeyDictionary()


def fast_ppf(distr):
    """ The quantile function (ppf) of a frozen scipy.stats distribution, as scipy computes it inside (0, 1) (the same
    values, bit by bit) but without the checks of its arguments, which made the prior transforms slow: ~70 us for a
    ppf, called for each parameter at every point of the samplers, serially in the main process. Values of q outside
    (0, 1), and distributions that are not continuous or have their own ppf, go to scipy's ppf. Cached for each
    distribution. """
    try:
        return _FAST_PPF[distr]
    except (KeyError, TypeError):
        pass
    func = distr.ppf
    dist = getattr(distr, 'dist', None)
    if isinstance(dist, stats.rv_continuous) and type(dist).ppf is stats.rv_continuous.ppf:
        try:
            shapes, loc, scale = dist._parse_args(*distr.args, **distr.kwds)
            ok = bool(np.all(dist._argcheck(*shapes)) and np.all(np.asarray(scale) > 0) and np.all(loc == loc))
        except Exception:
            ok = False
        if ok:
            _ppf, slow = dist._ppf, distr.ppf
            if all(np.ndim(a) == 0 for a in (*shapes, scale, loc)):
                # (scalar parameters, the usual case: rv_continuous.ppf broadcasts them to the shape of q, which
                # changes no value)
                shapes, scale, loc = [float(a) for a in shapes], float(scale), float(loc)

                def func(q):
                    q = np.asarray(q, dtype=float)
                    q1 = q.reshape(-1)
                    if q1.size == 0 or not ((q1 > 0) & (q1 < 1)).all():
                        return slow(q)
                    out = _ppf(q1, *[np.full(q1.shape, a) for a in shapes]) * scale + loc
                    return out[0] if q.ndim == 0 else out.reshape(q.shape)
            else:
                def func(q):
                    q = np.asarray(q, dtype=float)
                    if q.size == 0 or not np.all((q > 0) & (q < 1)):
                        return slow(q)
                    # (as rv_continuous.ppf: 1-D arguments broadcast together, then the shape of q)
                    q1, *args = np.broadcast_arrays(*np.atleast_1d(q, *shapes, scale, loc))
                    out = _ppf(q1.ravel(), *[a.ravel() for a in args[:-2]]) * args[-2].ravel() + args[-1].ravel()
                    return out.reshape(q.shape)[()] if q.ndim == 0 else out.reshape(q.shape)
    try:
        _FAST_PPF[distr] = func
    except TypeError:
        pass
    return func


# the quantiles of the median and of the 1-sigma interval (of a Gaussian), of the posteriors and of the errors of
# the deconvolved profile (the tables used 0.16 and 0.84, print_stats and the corner plot these)
SIGMA_QUANTILES = (0.15865, 0.50, 0.84135)


# the units of the smoothness tolerances (LIKEOPTS smooth): those of the ring parameters, DENS in dex
SMOOTH_UNITS = dict(vrot="km/s", vdisp="km/s", vrad="km/s", vsys="km/s", vvert="km/s", dvdz="km/s/arcsec",
                    inc="deg", phi="deg", xpos="pix", ypos="pix", z0="arcsec", zcyl="arcsec", dens="dex")


def smoothness_bends(radii, values, beam, from_zero=False, log=False):
    """The bends over one beam of a ring profile (LIKEOPTS smooth) at its inner rings, and their weights.

    The bend at a ring is p'' B^2/8: the departure of the profile from the straight line through its values half a
    beam either side (p'' the second derivative with respect to radius on the radii of the rings, B the beam FWHM, both
    in arcsec). The weight of an inner ring j is the fraction of a beam of radius it stands for, (h_j + h_{j+1})/2/B
    (h the separations of the rings), so that the sum over the rings does not depend on their number.

    values: one profile (n,) or profiles in columns (n, m) (the bends are linear in them); from_zero: the profile
    starts from 0 at R = 0 (VROT: a node is added, so that a rise linear from the centre is not penalised); log: the
    bends of log10 of the values (DENS, in dex; values below 1e-6 of the largest count as that). Returns (bends,
    weights), empty with fewer than 3 nodes."""
    v, x = np.asarray(values, float), np.asarray(radii, float)
    if log:
        top = np.max(v, axis=0)
        v = np.log10(np.maximum(v, np.maximum(1e-6 * top, 1e-300)))
    if from_zero:
        v, x = np.concatenate([np.zeros((1,) + v.shape[1:]), v]), np.concatenate([[0.0], x])
    if v.shape[0] < 3:
        return np.zeros((0,) + v.shape[1:]), np.zeros(0)
    h1, h2 = np.diff(x)[:-1], np.diff(x)[1:]
    sh = (-1,) + (1,) * (v.ndim - 1)
    d2v = 2. * ((v[2:] - v[1:-1]) / h2.reshape(sh) - (v[1:-1] - v[:-2]) / h1.reshape(sh)) / (h1 + h2).reshape(sh)
    return d2v * beam ** 2 / 8., (h1 + h2) / (2. * beam)


def smoothness_penalty(radii, values, s, beam, from_zero=False, log=False):
    """The smoothness penalty of a ring parameter (LIKEOPTS smooth), subtracted from ln L: half the sum over the inner
    rings of (bend / s)^2, each weighted by the fraction of a beam it stands for (see smoothness_bends). It is a
    Gaussian prior of standard deviation s on the bend of the profile over each beam of radius, whatever the number of
    rings. s in the units of the parameter (dex with log, DENS), beam the FWHM (arcsec). Not multiplied by the factor
    of the correlation of the noise."""
    b, w = smoothness_bends(radii, values, beam, from_zero, log)
    return 0.5 * float(np.sum(w * b * b)) / (s * s)


def smoothness_summary(radii, values, s, beam, from_zero=False, log=False, cov=None):
    """What the smoothness prior of a ring parameter (LIKEOPTS smooth) does to a fit, at the profile values: a dict
    with n (the rings), bend (the rms bend over a beam, weighted as in the penalty, in the units of s), penalty (in
    ln L) and ndof: with cov (the posterior covariance of the values of the rings, n x n) the effective number of
    independent values of the profile, n - trace(cov H), H the Hessian of the penalty (with log, linearised at the
    values): n when the data alone fix the rings, 2 (a straight line; 1 from_zero) when the prior does; None
    without cov."""
    v = np.asarray(values, float)
    n = v.size
    b, w = smoothness_bends(radii, v, beam, from_zero, log)
    out = dict(n=n, penalty=0.5 * float(np.sum(w * b * b)) / (s * s),
               bend=float(np.sqrt(np.sum(w * b * b) / np.sum(w))) if w.size else 0., ndof=None)
    if cov is not None and w.size:
        D = smoothness_bends(radii, np.eye(n), beam, from_zero)[0]        # (inner rings, n): linear in the values
        if log:
            D = D / (np.maximum(v, max(1e-6 * v.max(), 1e-300)) * np.log(10.))[None, :]
        H = D.T @ (w[:, None] * D) / (s * s)
        out["ndof"] = float(min(max(n - np.sum(np.asarray(cov, float).reshape(n, n) * H), 0.), n))
    return out


def smoothness_text(used, given=None, per="beam"):
    """The line describing the smoothness priors (LIKEOPTS smooth) before a fit: the tolerances used ({parameter: s},
    those of the parameters free in each ring) per beam (or per what), and those given but not used (given: all the
    tolerances given), e.g. 'vrot 5, vdisp 3 km/s per beam (inc ignored: not free in each ring)'."""
    items = list(used.items())
    parts = []
    for i, (k, v) in enumerate(items):
        u = SMOOTH_UNITS.get(k, "")
        last = i == len(items) - 1 or SMOOTH_UNITS.get(items[i + 1][0], "") != u
        parts.append(f"{k} {v:g}" + (f" {u}" if last and u else ""))
    txt = ", ".join(parts) + f" per {per}" if parts else "none"
    unused = [k for k in (given or {}) if k not in used]
    if unused:
        txt += f" ({', '.join(unused)} ignored: not free in each ring)"
    return txt


def smoothness_result(stats):
    """The line on the smoothness priors after a fit: the effective number of independent values of each smoothed
    profile (stats: {parameter: smoothness_summary}), e.g. 'independent values vrot 4.9 of 8 rings, vdisp 5.0 of 8';
    '' without them (no covariance)."""
    parts = [f"{k} {st['ndof']:.1f} of {st['n']}" for k, st in stats.items() if st.get("ndof") is not None]
    if not parts:
        return ""
    parts[0] += " rings"
    return "independent values " + ", ".join(parts)
