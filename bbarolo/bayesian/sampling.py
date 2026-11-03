"""
The samplers of BayesianBBarolo (dynesty, nautilus) and the parallel pool.

Parallel sampling (threads > 1) uses a fork-based process pool: the workers inherit the fitter, so
only the parameter vectors are sent to them (nothing big is pickled), and it also works on macOS.
The functions given to the samplers must then be importable module-level functions. Under MPI
(a script launched with mpirun or mpiexec), schwimmbad's MPIPool is used instead.
"""

import os
import sys
import math
import time
import multiprocessing as mp
import numpy as np

try:
    import dynesty as dyn
    from dynesty import utils as dyfunc
except ImportError:                                   # dynesty is needed only for method='dynesty'
    dyn = dyfunc = None

# MPI is used only when the script is launched with mpirun/mpiexec
isMPI = "OMPI_COMM_WORLD_SIZE" in os.environ or "PMI_RANK" in os.environ
if isMPI:
    try:
        from mpi4py import MPI
        from schwimmbad import MPIPool
    except ImportError:
        raise ImportError("ERROR! To run in parallel with MPI, you need to install mpi4py and schwimmbad.")


# --------------------------------------------------------------------------
# Process pool: the workers inherit the fitter (fork) and only receive the parameter
# vectors. The functions must be importable module-level functions.
# --------------------------------------------------------------------------
_WORKER = {}


def _worker_init(key):
    _WORKER['f'] = _WORKER_REGISTRY[key]


_WORKER_REGISTRY = {}


def _worker_loglike(theta):
    # in a worker process 'f' is set by the initializer; samplers may also call the
    # functions in the main process (e.g. nautilus for the prior transform)
    f, like_kwargs = _WORKER['f'] if 'f' in _WORKER else _WORKER['parent']
    return f._log_like_func(theta, **like_kwargs)


def _worker_prior(u):
    f, _ = _WORKER['f'] if 'f' in _WORKER else _WORKER['parent']
    return f._prior_transform(u)


def _parent_loglike(theta):
    f, like_kwargs = _WORKER['parent']
    return f._log_like_func(theta, **like_kwargs)


def _parent_prior(u):
    f, _ = _WORKER['parent']
    return f._prior_transform(u)


def _queue_size(pool, threads):
    """ The number of proposals that dynesty evaluates at once: 4 per worker (the size of an MPI pool,
    otherwise the number of processes, threads). """
    return 4 * int(getattr(pool, 'size', None) or threads)


def _make_pool(fitter, like_kwargs, nproc):
    key = id(fitter)
    _WORKER_REGISTRY[key] = (fitter, like_kwargs)
    try:
        ctx = mp.get_context("fork")
    except ValueError:
        print("WARNING: process pools need the 'fork' start method, not available on this system. "
              "Running serially.")
        return None
    return ctx.Pool(nproc, initializer=_worker_init, initargs=(key,))


def sampler_options(method, dynamic, sampler_kwargs, run_kwargs):
    """ Checks the options of the sampler (SAMPLEROPTS, sampler_kwargs) and of its run (RUNOPTS,
    run_kwargs) against the arguments that the sampler accepts, before the setup: BBaroloError for
    an unknown option. The names of dynesty are accepted by all the samplers: nlive (n_live of
    nautilus) and dlogz, the stopping criterion of the static sampler of dynesty (dlogz_init for
    its dynamic sampler, f_live = 1 - exp(-dlogz) for nautilus: the fraction of the evidence left
    in the live points). Returns the options of the sampler and of the run. """
    import inspect
    from ..bbutils import BBaroloError
    sampler_kwargs, run_kwargs = dict(sampler_kwargs), dict(run_kwargs)
    if method == 'dynesty':
        if dyn is None:
            raise ImportError("BayesianBBarolo with method='dynesty' requires the package 'dynesty'.")
        Sampler = dyn.DynamicNestedSampler if dynamic else dyn.NestedSampler
        runf, own = Sampler.run_nested, {'loglikelihood', 'prior_transform', 'ndim'}
        if dynamic and 'dlogz' in run_kwargs:
            run_kwargs.setdefault('dlogz_init', run_kwargs.pop('dlogz'))
    elif method == 'nautilus':
        try:
            import nautilus
        except ImportError:
            raise ImportError("BayesianBBarolo requires the package nautilus when method='nautilus'.")
        Sampler, runf, own = nautilus.Sampler, nautilus.Sampler.run, {'prior', 'likelihood', 'n_dim'}
        if 'nlive' in sampler_kwargs:
            sampler_kwargs.setdefault('n_live', sampler_kwargs.pop('nlive'))
        if 'dlogz' in run_kwargs:
            run_kwargs.setdefault('f_live', 1. - math.exp(-float(run_kwargs.pop('dlogz'))))
    else:
        raise ValueError(f"ERROR! Unknown method '{method}' (dynesty or nautilus).")
    for what, func, opts in (("SAMPLEROPTS", Sampler, sampler_kwargs), ("RUNOPTS", runf, run_kwargs)):
        ok = set(inspect.signature(func).parameters) - {'self'} - own
        bad = sorted(set(opts) - ok)
        if bad:
            kind = (" (dynamic sampler)" if dynamic else " (static sampler)") if method == 'dynesty' else ""
            raise BBaroloError(f"BAYESIAN3DFIT: {what}: unknown option(s) {', '.join(bad)} of {method}{kind}. "
                               f"Options: {', '.join(sorted(ok))}.")
    return sampler_kwargs, run_kwargs


def sample(fitter, like_kwargs, method, threads, dynamic, sampler_kwargs, run_kwargs, verbose):
    """ Samples the posterior of the fitter (its _log_like_func with like_kwargs, its _prior_transform,
    its ndim) with method ('dynesty' or 'nautilus', the options checked by sampler_options), in
    threads processes (or an MPI pool). The pool is closed, or terminated if the sampling fails, and
    no workers are left running. Sets fitter.sampler (as soon as it exists) and fitter.results
    (dynesty; None for nautilus). Returns the samples, their weights and the time of the sampling
    (seconds). """
    pool = None
    failed = True
    _WORKER['parent'] = (fitter, like_kwargs)
    try:
        loglike, prior = _parent_loglike, _parent_prior
        if isMPI:
            _WORKER['f'] = (fitter, like_kwargs)
            pool = MPIPool()
            if not pool.is_master():
                pool.wait()
                sys.exit(0)
            loglike, prior = _worker_loglike, _worker_prior
        elif threads > 1:
            pool = _make_pool(fitter, like_kwargs, threads)
            if pool is not None:
                loglike, prior = _worker_loglike, _worker_prior

        toc = time.time()
        if method == 'dynesty':
            Sampler = dyn.DynamicNestedSampler if dynamic else dyn.NestedSampler
            skw = dict(bound='multi')
            if pool is not None:
                # several proposals per worker keep the processes busy
                skw.update(pool=pool, queue_size=_queue_size(pool if isMPI else None, threads))
            skw.update(sampler_kwargs)
            fitter.sampler = Sampler(loglike, prior, ndim=fitter.ndim, **skw)
            fitter.sampler.run_nested(**{'print_progress': verbose, **run_kwargs})
            fitter.results = fitter.sampler.results
            samples = fitter.results.samples
            weights = np.exp(fitter.results.logwt - fitter.results.logz[-1])
        else:
            import nautilus
            skw = dict(pass_dict=False)
            if pool is not None:
                skw['pool'] = pool
            skw.update(sampler_kwargs)
            fitter.sampler = nautilus.Sampler(prior, loglike, n_dim=fitter.ndim, **skw)
            fitter.sampler.run(**{'verbose': verbose, **run_kwargs})
            samples, logw, _ = fitter.sampler.posterior()
            weights = np.exp(logw - logw.max())
            fitter.results = None
        failed = False
    finally:
        if pool is not None:
            if failed and not isMPI:
                pool.terminate()            # (a failed sampling leaves no workers running)
            else:
                pool.close()
            if not isMPI:
                pool.join()
        _WORKER_REGISTRY.pop(id(fitter), None)
        _WORKER.pop('parent', None)
        _WORKER.pop('f', None)
    return samples, weights, time.time() - toc
