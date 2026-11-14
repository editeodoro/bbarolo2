"""The SLITFIT task of parameter files (BBarolo -p, bbarolo.run)."""

from ..tasks import _FromFile
from .slit import BayesianSlitFit


def _task_slit(c, par):
    """The BayesianSlitFit of the SLITFIT task: the parameters of par as given (but the flags of the tasks other
    than CONTSUB, the continuum of the slit, and the input files), the output folder of c and the slit of the parameter
    file (FITSFILE)."""
    from ..keywords import lookup
    p = c.pars()
    prm = {}
    for k, v in getattr(par, "_given", {}).items():
        kw = lookup(k)
        if kw is None or (kw.flag and k != "CONTSUB") or k in ("FITSFILE", "FITSLIST", "AUTO", "OUTFOLDER"):
            continue
        prm[k] = _FromFile(v)
    prm["OUTFOLDER"] = p.outFolder
    src = getattr(c, "_source_hdu", None)         # (a slit in memory: the Python task class)
    return BayesianSlitFit(src if src is not None else p.imageFile, **prm)


def run_task(c, par):
    """The SLITFIT task: the fit with the parameters of par (METHOD GLOBAL: the maximum of the likelihood, with
    MINIMISER; BAYESIAN: the posterior, with SAMPLER; SAMPLEROPTS, RUNOPTS, PRIORS, LIKEOPTS), with THREADS processes.
    With VERBOSE it prints the statistics. Nothing is written (see write_task). Returns the BayesianSlitFit."""
    from .common import task_fit
    f = _task_slit(c, par)
    p = c.pars()
    task_fit(f, p, "SLITFIT")
    if p.verbose:
        f.print_stats()
    return f


def write_task(f, c):
    """The outputs of the SLITFIT task f in the output folder of c, with its PLOTS."""
    p = c.pars()
    return f.write(outfolder=p.outFolder, plots=p.plots, verbose=bool(p.verbose))
