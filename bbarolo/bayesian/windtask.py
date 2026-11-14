"""The WINDFIT task of parameter files (BBarolo -p, bbarolo.run)."""

import numpy as np

from ..tasks import _FromFile
from .wind import BayesianWind


def _task_wind(c, par):
    """The BayesianWind of the WINDFIT task: the parameters of par as given (but the flags of the tasks and the
    input files), the output folder of c, and the data of c (with the changes of the tasks run before it)."""
    from ..keywords import lookup
    p = c.pars()
    prm = {}
    for k, v in getattr(par, "_given", {}).items():
        kw = lookup(k)
        if kw is None or kw.flag or k in ("FITSFILE", "FITSLIST", "AUTO", "OUTFOLDER"):
            continue
        prm[k] = _FromFile(v)
    prm["OUTFOLDER"] = p.outFolder
    f = BayesianWind(None, **prm)
    src = getattr(c, "_source_hdu", None)         # (a cube in memory: the Python task class)
    f._set_cube(src if src is not None else p.imageFile)
    cb = f._load_cube()
    if cb.array.shape == c.array.shape and not np.array_equal(cb.array, c.array, equal_nan=True):
        cb.array[...] = c.array
        f.data = np.asarray(cb.array, dtype=np.float64)
    return f


def run_task(c, par):
    """The WINDFIT task: the fit with the parameters of par (METHOD GLOBAL: the maximum of the likelihood, with
    MINIMISER; BAYESIAN: the posterior, with SAMPLER; SAMPLEROPTS, RUNOPTS, PRIORS, LIKEOPTS), with THREADS processes.
    With VERBOSE it prints the statistics. Nothing is written (see write_task). Returns the BayesianWind."""
    from .common import task_fit
    f = _task_wind(c, par)
    p = c.pars()
    task_fit(f, p, "WINDFIT")
    if p.verbose:
        f.print_stats()
    return f


def write_task(f, c):
    """The outputs of the WINDFIT task f in the output folder of the cube c, with its PLOTS."""
    p = c.pars()
    f.write(outfolder=p.outFolder, plots=p.plots, verbose=bool(p.verbose))
