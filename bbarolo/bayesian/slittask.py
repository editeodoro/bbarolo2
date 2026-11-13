"""The BAYESIANSLITFIT task of parameter files (BBarolo -p, bbarolo.run)."""

from ..tasks import _FromFile
from .slit import BayesianSlitFit


def _task_slit(c, par):
    """The BayesianSlitFit of the BAYESIANSLITFIT task: the parameters of par as given (but the flags of the tasks and
    the input files), the output folder of c and the slit of the parameter file (FITSFILE)."""
    from ..keywords import lookup
    p = c.pars()
    prm = {}
    for k, v in getattr(par, "_given", {}).items():
        kw = lookup(k)
        if kw is None or kw.flag or k in ("FITSFILE", "FITSLIST", "AUTO", "OUTFOLDER"):
            continue
        prm[k] = _FromFile(v)
    prm["OUTFOLDER"] = p.outFolder
    return BayesianSlitFit(p.imageFile, **prm)


def run_task(c, par):
    """The BAYESIANSLITFIT task: samples the posterior (or finds the best fit, SAMPLER OPTIMISE) with the parameters of
    par, with THREADS processes. With VERBOSE it prints the statistics. Nothing is written (see write_task). Returns the
    BayesianSlitFit."""
    from ..params import bayes_options
    f = _task_slit(c, par)
    p = c.pars()
    gf = p.gf
    verbose = bool(p.verbose)
    method = gf.SAMPLER.strip().lower()
    run_kwargs = {}
    if method == "dynesty" and not p.showbar and "print_progress" not in bayes_options(gf.RUNOPTS):
        run_kwargs["print_progress"] = False
    f._task_showbar = p.showbar
    f.run(method=method, threads=p.threads, verbose=verbose, run_kwargs=run_kwargs)
    if verbose:
        f.print_stats()
    return f


def write_task(f, c):
    """The outputs of the BAYESIANSLITFIT task f in the output folder of c, with its PLOTS."""
    p = c.pars()
    return f.write(outfolder=p.outFolder, plots=p.plots, verbose=bool(p.verbose))
