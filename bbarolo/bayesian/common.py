"""Small helpers of Bayesian3DFIT."""

import contextlib
import weakref
import numpy as np
from scipy import stats

# Threads of BLAS and OpenMP during the sampling and the errors of the density profile (None: no limit)
BLAS_THREADS = 1


def blas_threads():
    """ A context in which BLAS and OpenMP use BLAS_THREADS threads, set with threadpoolctl (if it is
    installed; otherwise nothing changes). The linear algebra of the likelihood (the deconvolved
    density profile) works on small matrices, for which many threads are slower: with MKL on 224
    cores, the likelihood with a free geometry took 1.6-2.2 times longer, and the processes of a
    parallel sampling competed for the cores. The values do not change. """
    if BLAS_THREADS is None:
        return contextlib.nullcontext()
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        return contextlib.nullcontext()
    return threadpool_limits(limits=int(BLAS_THREADS))

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
