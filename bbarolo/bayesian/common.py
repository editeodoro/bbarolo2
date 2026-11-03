"""Small helpers of BayesianBBarolo."""

import numpy as np
from scipy import stats

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
