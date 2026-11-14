"""3DFIT with METHOD GLOBAL or BAYESIAN in parameter files (BBarolo -p, bbarolo.run)."""

import numpy as np

from ..tasks import _FromFile
from .fitter import Bayesian3DFIT


def _task_fitter(c, par):
    """ The Bayesian3DFIT of 3DFIT with METHOD GLOBAL or BAYESIAN: the parameters of par as given (but the
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
    f = Bayesian3DFIT(None, **prm)
    src = getattr(c, "_source_hdu", None)         # (a cube in memory: the Python task class)
    f._set_cube(src if src is not None else p.imageFile)
    cb = f._load_cube()
    if cb.array.shape == c.array.shape and not np.array_equal(cb.array, c.array, equal_nan=True):
        cb.array[...] = c.array                   # changed by the tasks run before (CONTSUB, CHECKCUBE)
    f._init_rings(show_initial=bool(p.verbose))
    return f


def run_task(c, par):
    """ 3DFIT with METHOD GLOBAL or BAYESIAN: the fit with the parameters of par (a checked Param; its values as given,
    see Param._given) on the cube c read by BBarolo (with the changes of the tasks run before it, e.g. CONTSUB). The
    free parameters are FREE (default keywords.BAYES_DEFAULT_FREE), the priors PRIORS (the others: the defaults of
    Bayesian3DFIT); METHOD GLOBAL finds the maximum of the likelihood (MINIMISER), BAYESIAN samples the posterior
    (SAMPLER), with SAMPLEROPTS and RUNOPTS, in THREADS processes. With VERBOSE it prints the statistics of the
    posterior (print_stats). Nothing is written (see write_task). Returns the Bayesian3DFIT. """
    from .common import task_fit
    f = _task_fitter(c, par)
    p = c.pars()
    free = p.gf.FREE if "FREE" in getattr(par, "_given", {}) else None     # None: the default FREE
    task_fit(f, p, "3DFIT", free)
    if p.verbose:
        f.print_stats()
    return f


def write_task(f, c):
    """ The outputs of 3DFIT with METHOD GLOBAL or BAYESIAN, f (see Bayesian3DFIT.write), in the output folder of the cube
    c, with its PLOTS. """
    p = c.pars()
    f._cube_obj.pars().setOutfolder(p.outFolder)
    f.write(plots=p.plots, verbose=bool(p.verbose))
