"""Small helpers of BayesianBBarolo."""

import contextlib
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

# A function to print only if verbose is True
vprint = lambda verbose, *args, **kwargs: print(*args, **kwargs) if verbose else None


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


# the quantiles of the median and of the 1-sigma interval (of a Gaussian), of the posteriors and of the errors of
# the deconvolved profile (the tables used 0.16 and 0.84, print_stats and the corner plot these)
SIGMA_QUANTILES = (0.15865, 0.50, 0.84135)
