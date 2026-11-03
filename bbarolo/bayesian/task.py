"""The BAYESIAN3DFIT task of parameter files (BBarolo -p, bbarolo.run)."""

import os
import numpy as np

from ..tasks import _FromFile
from .fitter import BayesianBBarolo
from .common import vprint


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
    f._init_rings()
    return f


def run_task(c, par):
    """ The BAYESIAN3DFIT task: samples the posterior with the parameters of par (a checked
    Param; its values as given, see Param._given) on the cube c read by BBarolo (with the
    changes of the tasks run before it, e.g. CONTSUB). The free parameters are FREE (required),
    the sampler SAMPLER, with SAMPLEROPTS and RUNOPTS, and the priors PRIORS (the others: the
    defaults of BayesianBBarolo); the sampler runs with THREADS processes.

    Then it writes in OUTFOLDER the outputs of BayesianBBarolo.write_bestmodel (best model,
    maps, PV diagrams, <prefix>_rings_final1.txt with the errors, <prefix>_densprof.txt and, when DENS is not
    fitted, <prefix>_densprof_deconvolved.txt; with PLOTS the plots of 3DFIT) and the samples
    (<prefix>_<sampler>_results.npy, see load_results); with PLOTS also the corner plot
    (<prefix>_corner.pdf) and the radial profiles (<prefix>_profiles.pdf); with VERBOSE it
    prints the statistics of the posterior (print_stats). Returns the BayesianBBarolo. """
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

    if p.plots >= 0:                              # (as 3DFIT: PLOTS=-1 writes no model)
        f.write_bestmodel(plots=p.plots > 0, verbose=verbose)
    cp = f._cube_obj.pars()
    of, pre = cp.outFolder, cp.outPrefix
    fn = os.path.join(of, f"{pre}_{method}_results.npy")
    f.save_results(fn)
    vprint(verbose, f"Posterior samples written in {fn} (see BayesianBBarolo.load_results)")
    if p.plots > 0:
        import logging
        import warnings
        import matplotlib.pyplot as plt
        for name, make in (("corner", lambda: f.corner_plot(plot_dynesty=(method == 'dynesty'))),
                           ("profiles", f.plot_profiles)):
            out = os.path.join(of, f"{pre}_{name}.pdf")
            root = logging.getLogger()
            level = root.level
            try:
                # (no warnings of the plotting packages, e.g. for a posterior on very few samples)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    root.setLevel(logging.ERROR)
                    fig = make()
                    fig.savefig(out, bbox_inches='tight')
                plt.close(fig)
                vprint(verbose, f"Plot written in {out}")
            except Exception as e:
                plt.close('all')
                print(f"BAYESIAN3DFIT WARNING: no {name} plot ({e})")
            finally:
                root.setLevel(level)
    if verbose:
        f.print_stats()
    return f
