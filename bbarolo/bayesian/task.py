"""The BAYESIAN3DFIT task of parameter files (BBarolo -p, bbarolo.run)."""

import numpy as np

from ..tasks import _FromFile
from .fitter import BayesianBBarolo


def _task_fitter(c, par):
    """ The BayesianBBarolo of the BAYESIAN3DFIT task: the parameters of par as given (but the
    flags of the tasks and the input files), the output folder of c, and the data of c. """
    from ..keywords import lookup
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
    f._init_rings(show_initial=bool(p.verbose))
    return f


def run_task(c, par):
    """ The BAYESIAN3DFIT task: samples the posterior with the parameters of par (a checked
    Param; its values as given, see Param._given) on the cube c read by BBarolo (with the
    changes of the tasks run before it, e.g. CONTSUB). The free parameters are FREE (required),
    the sampler SAMPLER, with SAMPLEROPTS and RUNOPTS, and the priors PRIORS (the others: the
    defaults of BayesianBBarolo); the sampler runs with THREADS processes. With VERBOSE it prints
    the statistics of the posterior (print_stats). Nothing is written (see write_task). Returns the
    BayesianBBarolo. """
    from ..params import bayes_options
    f = _task_fitter(c, par)
    p = c.pars()
    gf = p.gf
    verbose = bool(p.verbose)
    method = gf.SAMPLER.strip().lower()
    run_kwargs = {}
    if method == 'dynesty' and not p.showbar and 'print_progress' not in bayes_options(gf.RUNOPTS):
        run_kwargs['print_progress'] = False      # e.g. LOGFILE: no progress line
    f.run(freepar=gf.FREE, method=method, threads=p.threads, verbose=verbose, run_kwargs=run_kwargs)
    if verbose:
        f.print_stats()
    return f


def write_task(f, c):
    """ The outputs of the BAYESIAN3DFIT task f (see BayesianBBarolo.write) in the output folder of the cube
    c, with its PLOTS. """
    p = c.pars()
    f._cube_obj.pars().setOutfolder(p.outFolder)
    f.write(plots=p.plots, verbose=bool(p.verbose))
