"""The BAYESIANWIND task of parameter files (BBarolo -p, bbarolo.run)."""

import numpy as np

from ..tasks import _FromFile
from .wind import BayesianWind


def _task_wind(c, par):
    """The BayesianWind of the BAYESIANWIND task: the parameters of par as given (but the flags of the tasks and the
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
    f._set_cube(p.imageFile)
    cb = f._load_cube()
    if cb.array.shape == c.array.shape and not np.array_equal(cb.array, c.array, equal_nan=True):
        cb.array[...] = c.array
        f.data = np.asarray(cb.array, dtype=np.float64)
    return f


def run_task(c, par):
    """The BAYESIANWIND task: samples the posterior with the parameters of par (FREE required; SAMPLER, SAMPLEROPTS,
    RUNOPTS, PRIORS, LIKEOPTS), with THREADS processes. With VERBOSE it prints the statistics. Nothing is written
    (see write_task). Returns the BayesianWind."""
    from ..params import bayes_options
    f = _task_wind(c, par)
    p = c.pars()
    gf = p.gf
    verbose = bool(p.verbose)
    method = gf.SAMPLER.strip().lower()
    run_kwargs = {}
    if method == "dynesty" and not p.showbar and "print_progress" not in bayes_options(gf.RUNOPTS):
        run_kwargs["print_progress"] = False
    f._task_showbar = p.showbar                   # (off with LOGFILE: the progress bars of SAMPLER OPTIMISE)
    f.run(method=method, threads=p.threads, verbose=verbose, run_kwargs=run_kwargs)
    if verbose:
        f.print_stats()
    return f


def write_task(f, c):
    """The outputs of the BAYESIANWIND task f in the output folder of the cube c, with its PLOTS."""
    p = c.pars()
    f.write(outfolder=p.outFolder, plots=p.plots, verbose=bool(p.verbose))
