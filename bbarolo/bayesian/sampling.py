"""
The samplers of Bayesian3DFIT (nautilusbb, nautilus, dynesty) and the parallel pool.

nautilusbb (SAMPLER NAUTILUSBB, the default) is the copy of nautilus in bbarolo/bayesian/nautilus, changed to
need no scikit-learn and to be faster with a pool; nautilus (SAMPLER NAUTILUS) is the package nautilus-sampler.

Parallel sampling (threads > 1) uses a fork-based process pool: the workers inherit the fitter, so
only the parameter vectors are sent to them (nothing big is pickled), and it also works on macOS.
The functions given to the samplers must then be importable module-level functions. Under MPI
(a script launched with mpirun or mpiexec), schwimmbad's MPIPool is used instead.
"""

import os
import sys
import math
import contextlib
import time
import warnings
import multiprocessing as mp
import numpy as np

from . import common

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
    _WORKER['blas'] = common.blas_threads()          # (also inherited from the parent)


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


def _nautilus_batch(pool, threads):
    """ The number of likelihood evaluations of each step of nautilus (its n_batch) with a pool: at least 4 per
    worker, as the proposals of dynesty, otherwise nautilus's default (the smallest multiple of the workers that is
    at least 100), so it differs only above 33 workers. With 2 points per worker (64 workers), the work of the
    sampler between the steps, serial in the main process while the workers wait, took 9% more time on NGC 2403.
    (the size of an MPI pool, otherwise the number of processes, threads) """
    size = int(getattr(pool, 'size', None) or threads)
    return max(-(-100 // size) * size, 4 * size)


# The training of the neural networks of nautilus's bounds: at most NAUTILUS_MAX_ITER epochs (nautilus: 10000, with
# early stopping after 10 epochs without improvement, which took ~80 epochs). On UGC 1281 (24 parameters, 64
# processes) the run took 2.3x less time (training 600 s -> 90 s) for 14% more likelihood calls, with posterior means
# within 0.01 sigma and widths within 1% (as two seeds); on NGC 2403 (7 parameters) the calls did not change.
NAUTILUS_MAX_ITER = 20


def _nautilus_kwargs(pool, threads, sampler_kwargs):
    """ The arguments of nautilus.Sampler (but the prior, the likelihood and n_dim): the pool and its n_batch
    (_nautilus_batch) if there is a pool, at most NAUTILUS_MAX_ITER epochs for the training of the networks, then
    the options of the user (sampler_kwargs; the keys of its neural_network_kwargs update those of BBarolo). """
    skw = dict(pass_dict=False)
    if pool is not None:
        skw.update(pool=pool, n_batch=_nautilus_batch(pool if isMPI else None, threads))
    skw.update(sampler_kwargs)
    nn = sampler_kwargs.get('neural_network_kwargs') or {}
    skw['neural_network_kwargs'] = {'max_iter': NAUTILUS_MAX_ITER, **nn}
    return skw


@contextlib.contextmanager
def _quiet_training(method):
    """ A context without the ConvergenceWarning of scikit-learn, which nautilus's networks give when their
    training stops at max_iter (NAUTILUS_MAX_ITER: on purpose), one per network. The process pool started in it
    inherits the filter. """
    with warnings.catch_warnings():
        if method == 'nautilus':
            try:
                from sklearn.exceptions import ConvergenceWarning
                warnings.simplefilter('ignore', ConvergenceWarning)
            except ImportError:
                pass
        yield


from ..bbutils import BBaroloError


class WorkerDied(BBaroloError):
    """ A worker process of the pool died (a crash or a kill) while it computed the likelihood (a BBaroloError: the
    command line prints the message, with what to do). """


class _ForkPool:
    """ A fork-based multiprocessing.Pool whose map raises WorkerDied when a worker dies. (multiprocessing.Pool
    replaces a dead worker, but the task it was computing is lost and map() waits for it forever: so it went on macOS,
    where workers crashed in Accelerate.) map waits for the results in steps, checking the workers started with the
    pool (they never exit while the pool is open). After a death close() terminates the pool (a closed pool would
    wait for the lost task when joined). The rest is that of the pool (terminate, join, _processes, ...). """
    _CHECK = 0.5                        # (seconds between the checks of the workers)

    def __init__(self, ctx, nproc, key):
        self._pool = ctx.Pool(nproc, initializer=_worker_init, initargs=(key,))
        self._workers = list(self._pool._pool)
        self._broken = False

    def map(self, func, iterable, chunksize=None):
        res = self._pool.map_async(func, iterable, chunksize)
        while not res.ready():
            res.wait(self._CHECK)
            dead = [w for w in self._workers if w.exitcode is not None]
            if dead and not res.ready():
                self._broken = True
                raise WorkerDied(_died_message(dead))
        return res.get()

    def close(self):
        (self._pool.terminate if self._broken else self._pool.close)()

    def __getattr__(self, name):
        if name in ("_pool", "_workers"):              # (not set: the pool did not start)
            raise AttributeError(name)
        return getattr(self._pool, name)


def _died_message(dead):
    import signal
    how = []
    for w in dead:
        c = w.exitcode
        try:
            how.append(f"signal {signal.Signals(-c).name}" if c < 0 else f"exit code {c}")
        except ValueError:
            how.append(f"exit code {c}")
    mac = (" On macOS this is Apple's Accelerate (the BLAS of numpy), whose threads do not survive the fork of the "
           "workers: run with VECLIB_MAXIMUM_THREADS=1 in the environment, or THREADS 1.") \
        if sys.platform == "darwin" else ""
    return (f"THREADS > 1: a worker process of the pool died while it computed the likelihood "
            f"({', '.join(sorted(set(how)))}): its work was lost, so the fit stopped.{mac}")


def _make_pool(fitter, like_kwargs, nproc):
    key = id(fitter)
    _WORKER_REGISTRY[key] = (fitter, like_kwargs)
    try:
        ctx = mp.get_context("fork")
    except ValueError:
        print("WARNING: process pools need the 'fork' start method, not available on this system. "
              "Running serially.")
        return None
    return _ForkPool(ctx, nproc, key)


def _nautilus_module(method):
    """ The nautilus of method: the copy in bbarolo (nautilusbb) or the package nautilus-sampler (nautilus). """
    if method == 'nautilusbb':
        from . import nautilus as nautilusbb
        return nautilusbb
    try:
        import nautilus
    except ImportError:
        raise ImportError("SAMPLER NAUTILUS (method='nautilus') needs the package nautilus-sampler "
                          "(pip install nautilus-sampler); NAUTILUSBB, the default, needs nothing else.")
    return nautilus


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
            raise ImportError("Bayesian3DFIT with method='dynesty' requires the package 'dynesty'.")
        Sampler = dyn.DynamicNestedSampler if dynamic else dyn.NestedSampler
        runf, own = Sampler.run_nested, {'loglikelihood', 'prior_transform', 'ndim'}
        if dynamic and 'dlogz' in run_kwargs:
            run_kwargs.setdefault('dlogz_init', run_kwargs.pop('dlogz'))
    elif method in ('nautilusbb', 'nautilus'):
        Sampler = _nautilus_module(method).Sampler
        runf, own = Sampler.run, {'prior', 'likelihood', 'n_dim'}
        if 'nlive' in sampler_kwargs:
            sampler_kwargs.setdefault('n_live', sampler_kwargs.pop('nlive'))
        if 'dlogz' in run_kwargs:
            run_kwargs.setdefault('f_live', 1. - math.exp(-float(run_kwargs.pop('dlogz'))))
    else:
        raise ValueError(f"ERROR! Unknown method '{method}' (nautilusbb, nautilus or dynesty).")
    for what, func, opts in (("SAMPLEROPTS", Sampler, sampler_kwargs), ("RUNOPTS", runf, run_kwargs)):
        ok = set(inspect.signature(func).parameters) - {'self'} - own
        bad = sorted(set(opts) - ok)
        if bad:
            kind = (" (dynamic sampler)" if dynamic else " (static sampler)") if method == 'dynesty' else ""
            raise BBaroloError(f"BAYESIAN3DFIT: {what}: unknown option(s) {', '.join(bad)} of {method}{kind}. "
                               f"Options: {', '.join(sorted(ok))}.")
    if method == 'nautilusbb':          # (its networks take the options of scikit-learn's that apply to them)
        from .nautilus.mlp import MLPRegressor
        ok = set(inspect.signature(MLPRegressor).parameters) - {'random_state'}
        bad = sorted(set(sampler_kwargs.get('neural_network_kwargs') or {}) - ok)
        if bad:
            raise BBaroloError(f"BAYESIAN3DFIT: SAMPLEROPTS: unknown option(s) {', '.join(bad)} of the networks of "
                               f"nautilusbb (neural_network_kwargs). Options: {', '.join(sorted(ok))}.")
    return sampler_kwargs, run_kwargs


def sample(fitter, like_kwargs, method, threads, dynamic, sampler_kwargs, run_kwargs, verbose):
    """ Samples the posterior of the fitter (its _log_like_func with like_kwargs, its _prior_transform,
    its ndim) with method ('nautilusbb', 'nautilus' or 'dynesty', the options checked by sampler_options), in
    threads processes (or an MPI pool), with one BLAS thread in each (common.blas_threads). The
    pool is closed, or terminated if the sampling fails, and no workers are left running. Sets
    fitter.sampler (as soon as it exists) and fitter.results (dynesty; None for nautilus and nautilusbb). Returns
    the samples, their weights and the time of the sampling (seconds). """
    with common.blas_threads(), _quiet_training(method):
        return _sample(fitter, like_kwargs, method, threads, dynamic, sampler_kwargs, run_kwargs, verbose)


def _open_pool(fitter, like_kwargs, threads):
    """ The pool of the evaluations of the log-likelihood of fitter (its _log_like_func with like_kwargs): an MPI
    pool under MPI (the other ranks wait in it and exit), threads fork-based processes if threads > 1, otherwise
    None (serial). Returns the pool and the functions of the log-likelihood and of the prior transform to give the
    samplers (module-level functions, run in the workers with a pool). """
    _WORKER['parent'] = (fitter, like_kwargs)
    pool = None
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
    return pool, loglike, prior


def _close_pool(fitter, pool, failed):
    """ Closes the pool of _open_pool (terminated if the work failed: no workers are left running). """
    if pool is not None:
        if failed and not isMPI:
            pool.terminate()
        else:
            pool.close()
        if not isMPI:
            pool.join()
    _WORKER_REGISTRY.pop(id(fitter), None)
    _WORKER.pop('parent', None)
    _WORKER.pop('f', None)
    _WORKER.pop('blas', None)


@contextlib.contextmanager
def evaluator(fitter, like_kwargs, threads):
    """ A context that gives a function computing the log-likelihood of fitter (its _log_like_func with like_kwargs)
    at a list of parameter vectors, an array of the values: in threads processes (or an MPI pool), as the sampling,
    with one BLAS thread in each (common.blas_threads). Used by the optimisation (optimize.py). """
    with common.blas_threads():
        pool, loglike, _ = _open_pool(fitter, like_kwargs, threads)
        failed = True
        try:
            if pool is None:
                yield lambda thetas: np.array([loglike(np.asarray(t, float)) for t in thetas], float)
            else:
                # (chunks of about four vectors per worker and call keep the processes busy)
                size = int(getattr(pool, 'size', None) or threads)
                yield lambda thetas: np.array(pool.map(loglike, [np.asarray(t, float) for t in thetas],
                                                       **({} if isMPI else
                                                          dict(chunksize=max(1, len(thetas) // (4 * size))))),
                                              float)
            failed = False
        finally:
            _close_pool(fitter, pool, failed)


def _sample(fitter, like_kwargs, method, threads, dynamic, sampler_kwargs, run_kwargs, verbose):
    pool = None
    failed = True
    try:
        pool, loglike, prior = _open_pool(fitter, like_kwargs, threads)
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
            nautilus = _nautilus_module(method)
            skw = _nautilus_kwargs(pool, threads, sampler_kwargs)
            fitter.sampler = nautilus.Sampler(prior, loglike, n_dim=fitter.ndim, **skw)
            # (nautilusbb: its line of progress, with the time left, updated in place on a terminal with SHOWBAR,
            # otherwise a line at intervals)
            nverbose = ((verbose and 'lines') if method == 'nautilusbb' and not common.show_bar(fitter) else verbose)
            fitter.sampler.run(**{'verbose': nverbose, **run_kwargs})
            samples, logw, _ = fitter.sampler.posterior()
            weights = np.exp(logw - logw.max())
            fitter.results = None
        failed = False
    finally:
        _close_pool(fitter, pool, failed)
    return samples, weights, time.time() - toc
