"""
The optimisation of Bayesian3DFIT and BayesianWind (their method optimize): the best fit found by a classic
minimisation of -log L, with the likelihood of the sampling (the same function, options and free parameters), for
testing the samplers and quick fits. The result is the maximum likelihood within the bounds of the priors (their
support), or with posterior=True the maximum of the posterior (-log L - log prior).

The search runs in the unit cube of the priors, the space of the samplers: theta = the prior transform of u, u in
[0, 1]^ndim (1e-9 from an infinite end of a prior). Every parameter then has a range of one, the bounds are those of
the priors, and a log-uniform prior (SIGMA, FRACERR, EMSCALE, ...) is searched in its logarithm.

Methods (method):
  'L-BFGS-B' (the default), 'TNC', 'SLSQP', 'trust-constr': quasi-Newton methods with bounds, the gradient by
             central differences of step `step` (in units of the range of each prior), its 2 ndim + 1 likelihoods
             computed at once in `threads` processes (the pool of the sampling);
  'Powell', 'Nelder-Mead', 'COBYLA', 'COBYQA': without derivatives (one likelihood at a time: serial, and slow with
             many parameters, e.g. Nelder-Mead beyond ~10);
  'differential_evolution': a global search (scipy), its population computed in the processes, then polished with
             L-BFGS-B.
The options of the method are given in `options` (scipy's, e.g. maxiter); nstart > 1 starts the local methods also
from nstart - 1 points drawn from the priors (seed), and keeps the best. With verbose, the progress (the likelihoods
computed and the best log-likelihood) is printed after 10 seconds and then at intervals of a tenth of the time elapsed
(the minimisation has no known length), and the Hessian shows the progress bar of BBarolo (showbar: SHOWBAR). In
parameter files (and as the default of run()), SAMPLER OPTIMISE (or OPTIMIZE) runs optimize, with its arguments in
SAMPLEROPTS and the options of the method in RUNOPTS (see sampler_options).

The errors (errors=True): the Laplace approximation, the inverse of the Hessian of the objective at the best fit (in
the parameters), computed by finite differences (about ndim^2 likelihoods, steps of about four conditional sigma, see
_TARGET), with draws of the Gaussian truncated to the supports of the priors (nsamples) that serve as the samples of
the posterior in the outputs (errors, tables, corner plot). Directions that the likelihood does not constrain
(eigenvalues of the Hessian below that of the prior, 12 in u) get the width of the prior; a parameter closer to a
bound of its prior than its error is flagged (the Laplace approximation does not hold there).
"""

import math
import time
import numpy as np
from scipy import optimize as sopt
from scipy.optimize import OptimizeResult

from ..bbutils import BBaroloError
from .common import vprint
from .sampling import evaluator

LOCAL_METHODS = ('L-BFGS-B', 'TNC', 'SLSQP', 'trust-constr', 'Powell', 'Nelder-Mead', 'COBYLA', 'COBYQA')
GRADIENT_METHODS = ('L-BFGS-B', 'TNC', 'SLSQP', 'trust-constr')
GLOBAL_METHODS = ('differential_evolution',)
METHODS = LOCAL_METHODS + GLOBAL_METHODS

# SAMPLER OPTIMISE (parameter files, run(method='optimise')): its names, and the options of SAMPLEROPTS (the arguments
# of optimize; RUNOPTS are the options of the method)
SAMPLER_NAMES = ('optimise', 'optimize')
SAMPLER_OPTIONS = ('method', 'x0', 'posterior', 'errors', 'nstart', 'seed', 'step', 'nsamples')

_BIG = 1e30                 # the objective where the likelihood is not finite
_U_EDGE = 1e-9              # the bound of u at an infinite end of a prior
# the change of the objective at the steps of the Hessian: 8, about four conditional sigma. The likelihood of the
# discretised models has a jitter (~0.2 in log L on a mock of NGC 2403) that makes smaller steps measure too large a
# curvature: the errors of INC and PA were 2.5 times too small with steps of one sigma, 20% with two, 5-10% with four
# (against the posterior of nautilus); those of VROT and VDISP did not change.
_TARGET = 8.
_FTOL = 1e-3                # the tolerance of Powell on -log L (scipy's is relative, 1e-4: ~2 for log L ~ 2e4)
# the progress (verbose): the minimisation has no known length, so a line after _PROGRESS_FIRST seconds and then every
# max(_PROGRESS_FIRST, a tenth of the time elapsed) seconds; the Hessian, whose number of likelihoods is known, has the
# progress bar of BBarolo (bbutils.ProgressBar). The likelihoods are computed in chunks of _CHUNK, so that the long
# batches (the Hessian, the populations of differential evolution) show the progress too.
_PROGRESS_FIRST = 10.
_CHUNK = 128


def method_name(method):
    """ The name of an optimisation method (case-insensitive; 'lbfgsb', 'nelder-mead', 'de', ... are accepted).
    Raises BBaroloError for an unknown method. """
    key = str(method).strip().lower().replace('_', '').replace('-', '')
    for m in METHODS:
        if key == m.lower().replace('_', '').replace('-', ''):
            return m
    if key in ('de', 'differentialevolution'):
        return 'differential_evolution'
    if key in ('neldermead', 'simplex'):
        return 'Nelder-Mead'
    raise BBaroloError(f"optimize: unknown method {method!r} (methods: {', '.join(METHODS)}).")


def sampler_options(sampler_kwargs, run_kwargs=None, task="BAYESIAN3DFIT"):
    """ The arguments of optimize() for SAMPLER OPTIMISE: those of SAMPLEROPTS (sampler_kwargs: method, x0 ('centre',
    or values separated by commas), posterior, errors, nstart, seed, step, nsamples) and, as options, RUNOPTS
    (run_kwargs: the options of the method, e.g. maxiter; checked by the method when it starts). BBaroloError for an
    unknown option or method. """
    kw = dict(sampler_kwargs or {})
    bad = sorted(set(kw) - set(SAMPLER_OPTIONS))
    if bad:
        raise BBaroloError(f"{task}: SAMPLEROPTS: unknown option(s) {', '.join(bad)} of SAMPLER OPTIMISE. Options: "
                           f"{', '.join(SAMPLER_OPTIONS)} (those of the method in RUNOPTS).")
    if kw.get('method') is not None:
        try:
            kw['method'] = method_name(kw['method'])
        except BBaroloError as e:
            raise BBaroloError(f"{task}: SAMPLEROPTS: {str(e).replace('optimize: ', '')}") from None
    if isinstance(kw.get('x0'), (int, float)):
        kw['x0'] = [kw['x0']]
    if run_kwargs:
        kw['options'] = dict(run_kwargs)
    return kw


class _Space:
    """ The unit cube of the priors of a fitter: theta(u), its prior transform; u(theta), the CDFs of the priors; the
    bounds of u; dtheta/du = 1/pdf; the log-prior. """

    def __init__(self, fitter):
        self.f = fitter
        n = fitter.ndim
        self.dists = [None] * n
        for key, idx in fitter.freepar_idx.items():
            for i in np.atleast_1d(idx):
                self.dists[int(i)] = fitter.prior_distr[key]
        self.names = list(fitter.freepar_names)
        self.lo = np.array([0. if np.isfinite(d.ppf(0.)) else _U_EDGE for d in self.dists])
        self.hi = np.array([1. if np.isfinite(d.ppf(1.)) else 1. - _U_EDGE for d in self.dists])
        self.tlo = np.array([d.ppf(a) for d, a in zip(self.dists, self.lo)])     # (the bounds in theta)
        self.thi = np.array([d.ppf(b) for d, b in zip(self.dists, self.hi)])

    def theta(self, u):
        return np.asarray(self.f._prior_transform(np.clip(np.asarray(u, float), self.lo, self.hi)), float)

    def u(self, theta):
        return np.array([d.cdf(t) for d, t in zip(self.dists, np.asarray(theta, float))])

    def jacobian(self, theta):
        """ dtheta/du at theta (1/pdf of each prior). """
        return np.array([1. / d.pdf(t) for d, t in zip(self.dists, np.asarray(theta, float))])

    def logprior(self, theta):
        return float(sum(d.logpdf(t) for d, t in zip(self.dists, np.asarray(theta, float))))

    def support(self):
        return np.array([d.support() for d in self.dists], float)


def _duration(t):
    return f"{t:.0f} s" if t < 120 else f"{t / 60.:.1f} min"


class _Progress:
    """ The progress of an optimisation (verbose): the time, the likelihoods computed and the best log-likelihood (or
    log-posterior), printed after _PROGRESS_FIRST seconds and then at intervals of a tenth of the time elapsed. """

    def __init__(self, verbose, posterior):
        self.on = bool(verbose)
        self.what = "log-posterior" if posterior else "log-likelihood"
        self.t0 = time.time()
        self.next = _PROGRESS_FIRST

    def update(self, nfev, fbest):
        if not self.on:
            return
        el = time.time() - self.t0
        if el >= self.next:
            print(f"  {_duration(el):>8s}: {nfev} likelihoods, best {self.what} {-fbest:.7g}", flush=True)
            self.next = el + max(_PROGRESS_FIRST, 0.1 * el)


class _Objective:
    """ The objective of the minimisation in u: -log L (or -log L - log prior), with the likelihoods of a list of
    points computed at once (batch, in the processes, in chunks of _CHUNK); non-finite values are _BIG. Counts the
    likelihoods, and shows the progress. """

    def __init__(self, space, loglikes, posterior, progress=None):
        self.space, self.loglikes, self.posterior = space, loglikes, bool(posterior)
        self.nfev = 0
        self.best = (np.inf, None)
        self.fmin = np.inf                     # (the best value computed, also by the Hessian: for the progress)
        self.progress = progress or _Progress(False, posterior)
        self.bar = None                        # (a ProgressBar and the likelihoods before it: the Hessian)

    def batch(self, us):
        """ The objective at a list of points in u. """
        us = [np.clip(np.asarray(u, float), self.space.lo, self.space.hi) for u in us]
        F = self.batch_theta([self.space.theta(u) for u in us])
        i = int(np.argmin(F))
        if F[i] < self.best[0]:
            self.best = (float(F[i]), us[i].copy())
        return F

    def batch_theta(self, thetas):
        """ The objective at a list of vectors of the parameters (within the bounds of the priors). """
        thetas = [np.clip(np.asarray(t, float), self.space.tlo, self.space.thi) for t in thetas]
        out = []
        for k in range(0, len(thetas), _CHUNK):
            chunk = thetas[k:k + _CHUNK]
            F = -np.asarray(self.loglikes(chunk), float)
            self.nfev += len(chunk)
            if self.posterior:
                F = F - np.array([self.space.logprior(t) for t in chunk])
            F = np.where(np.isfinite(F), F, _BIG)
            self.fmin = min(self.fmin, float(F.min())) if F.size else self.fmin
            if self.bar is not None:
                self.bar[0].update(self.nfev - self.bar[1])
            else:
                self.progress.update(self.nfev, self.fmin)
            out.append(F)
        return np.concatenate(out) if out else np.empty(0)

    def __call__(self, u):
        return float(self.batch([u])[0])

    def gradient_points(self, u, h):
        """ The steps of the central differences at u (one-sided at a bound): up (to hi) and down (to lo). """
        up = np.minimum(h, self.space.hi - u)
        dn = np.minimum(h, u - self.space.lo)
        return up, dn

    def value_and_grad(self, u, h):
        u = np.clip(np.asarray(u, float), self.space.lo, self.space.hi)
        n = u.size
        up, dn = self.gradient_points(u, h)
        pts = [u]
        for i in range(n):
            e = np.zeros(n)
            e[i] = up[i]
            pts.append(u + e)
        for i in range(n):
            e = np.zeros(n)
            e[i] = dn[i]
            pts.append(u - e)
        F = self.batch(pts)
        f0, fp, fm = F[0], F[1:n + 1], F[n + 1:]
        g = np.zeros(n)
        for i in range(n):
            okp, okm = up[i] > 0 and fp[i] < _BIG, dn[i] > 0 and fm[i] < _BIG
            if okp and okm:
                g[i] = (fp[i] - fm[i]) / (up[i] + dn[i])
            elif okp:
                g[i] = (fp[i] - f0) / up[i]
            elif okm:
                g[i] = (f0 - fm[i]) / dn[i]
        return float(f0), g


def _local(obj, method, u0, step, options):
    """ One local minimisation from u0: a scipy OptimizeResult (x in u). An option that the method does not know is an
    error (BBaroloError; scipy only warns). """
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="Unknown solver options", category=sopt.OptimizeWarning)
        try:
            return _local_run(obj, method, u0, step, options)
        except sopt.OptimizeWarning as e:
            raise BBaroloError(f"optimize: {method}: {e} (RUNOPTS, or options)") from None


def _local_run(obj, method, u0, step, options):
    sp = obj.space
    bounds = sopt.Bounds(sp.lo, sp.hi)
    opts = dict(options or {})
    if method == 'Powell' and 'ftol' not in opts:
        opts['ftol'] = _FTOL / max(abs(obj(u0)), 1.)
    if method in GRADIENT_METHODS:
        h = np.full(u0.size, float(step))
        return sopt.minimize(lambda u: obj.value_and_grad(u, h), u0, jac=True, method=method, bounds=bounds,
                             options=opts)
    if method == 'Nelder-Mead':
        # (an initial simplex of 5% of the range of each prior: the default, 5% of the value, is no scale here)
        if 'initial_simplex' not in opts:
            n = u0.size
            sim = np.repeat(u0[None], n + 1, axis=0)
            for i in range(n):
                d = 0.05 if u0[i] + 0.05 <= sp.hi[i] else -0.05
                sim[i + 1, i] += d
            opts['initial_simplex'] = sim
        opts.setdefault('maxfev', 200 * u0.size)
    return sopt.minimize(obj, u0, method=method, bounds=bounds, options=opts)


def _hessian(obj, x, verbose, showbar=True):
    """ The Hessian of the objective in theta at x (finite differences), with steps of about four conditional sigma (a
    change _TARGET of the objective), the stencil moved inside the bounds of the priors. The Hessian is that of the
    parameters, not of u: with steps of a few sigma, the curvature in u of a log-uniform (or any non-uniform) prior
    adds terms of fourth order (8% of the variance of a Gaussian of 10% width on a log-uniform prior over three
    decades). Returns the Hessian and the steps. """
    sp = obj.space
    n = x.size
    S = sp.jacobian(x)                     # (dtheta/du: the scale of the prior at x)
    rng = sp.thi - sp.tlo
    h = 1e-3 * S
    from ..bbutils import ProgressBar
    bar = ProgressBar(True, verbose, showbar)
    # (the likelihoods: the steps, 1 to 3 times 2n+1, then n^2 + n + 1; the bar counts the most)
    bar.init(" Estimating errors (Hessian)... ", n * n + n + 1 + 3 * (2 * n + 1))
    obj.bar = (bar, obj.nfev)
    try:
        return _hessian_steps(obj, x, n, S, rng, h)
    finally:
        obj.bar = None
        bar.fillSpace("Done.\n")


def _hessian_steps(obj, x, n, S, rng, h):
    sp = obj.space

    def centre(h):
        return np.clip(x, sp.tlo + h, sp.thi - h)

    def diagonal(c, h):
        E = np.eye(n)
        F = obj.batch_theta([c] + [c + h[i] * E[i] for i in range(n)] + [c - h[i] * E[i] for i in range(n)])
        return F[0], F[1:n + 1], F[n + 1:]

    for _ in range(3):                     # (the steps: a change _TARGET of the objective, from the curvature)
        h = np.minimum(h, 0.25 * rng)
        c = centre(h)
        f0, fp, fm = diagonal(c, h)
        d2 = (fp + fm - 2. * f0) / h ** 2
        new = np.where((d2 > 0) & (fp < _BIG) & (fm < _BIG), np.sqrt(2. * _TARGET / np.maximum(d2, 1e-300)), 10 * h)
        new = np.clip(new, 1e-8 * S, 0.25 * rng)
        done = np.all(np.abs(np.log(new / h)) < math.log(1.5))
        h = new
        if done:
            break
    c = centre(h)
    f0, fp, fm = diagonal(c, h)
    H = np.diag((fp + fm - 2. * f0) / h ** 2)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    if pairs:
        E = np.eye(n)
        pts = [c + h[i] * E[i] + h[j] * E[j] for i, j in pairs] + [c - h[i] * E[i] - h[j] * E[j] for i, j in pairs]
        F = obj.batch_theta(pts)
        npair = len(pairs)
        for k, (i, j) in enumerate(pairs):
            v = (F[k] + F[npair + k] - fp[i] - fm[i] - fp[j] - fm[j] + 2. * f0) / (2. * h[i] * h[j])
            H[i, j] = H[j, i] = v
    if np.any(np.concatenate([fp, fm, [f0]]) >= _BIG) or not np.all(np.isfinite(H)):
        H = np.where(np.isfinite(H) & (np.abs(H) < _BIG), H, 0.)
    return H, h


def _laplace(space, theta, H, nsamples, seed):
    """ The covariance of theta from the Hessian H (in theta) and nsamples draws of the Gaussian truncated to the
    supports of the priors. The directions that H does not constrain more than the prior (an eigenvalue of the Hessian
    in u, S H S with S = dtheta/du, below 12, the inverse of the variance of u under the prior) get the width of the
    prior. Returns (cov, the covariance in u, samples, regularised). """
    S = space.jacobian(theta)
    Hu = S[:, None] * (0.5 * (H + H.T)) * S[None, :]
    w, V = np.linalg.eigh(Hu)
    floor = 12.
    reg = bool(np.any(w < floor))
    w = np.maximum(w, floor)
    A = S[:, None] * (V / np.sqrt(w))               # cov = A A^T
    cov = A @ A.T
    cov_u = (V / w) @ V.T
    sup = space.support()
    rng = np.random.default_rng(seed)
    out, ntry = [], 0
    while sum(len(x) for x in out) < nsamples and ntry < 50:
        z = theta + rng.standard_normal((nsamples, theta.size)) @ A.T
        ok = np.all((z >= sup[:, 0]) & (z <= sup[:, 1]), axis=1)
        out.append(z[ok])
        ntry += 1
    s = np.concatenate(out) if out else np.empty((0, theta.size))
    if len(s) < nsamples:                  # (the best fit far beyond a bound: clipped)
        z = np.clip(theta + rng.standard_normal((nsamples - len(s), theta.size)) @ A.T, sup[:, 0], sup[:, 1])
        s = np.concatenate([s, z])
    return cov, cov_u, s[:nsamples], reg


def optimize(fitter, like_kwargs, theta0, method='L-BFGS-B', threads=1, posterior=False, errors=True, nstart=1,
             seed=None, step=1e-3, options=None, nsamples=4000, verbose=True, showbar=True):
    """ Minimise -log L of fitter (its _log_like_func with like_kwargs, its priors and free parameters; see the
    module documentation) from theta0 (a vector of the free parameters; NaN: the centre of the prior), with method,
    the likelihoods computed in threads processes. Returns an OptimizeResult: x (the best fit), logl (its
    log-likelihood), logprior, fun (the objective), u (x in the unit cube of the priors), method, success, message,
    nit, nfev (likelihoods), time (seconds), starts (the best -log L of each start), at_bound (the parameters closer to
    a bound of their prior than their error, or than 0.1% of the prior); with errors also cov (the covariance), errors
    (its square root on the diagonal), hessian (of the objective, in the units of the parameters), steps (of the
    Hessian), samples (the draws of the Laplace approximation) and regularised (directions without constraint: their
    width is that of the prior). """
    method = method_name(method)
    space = _Space(fitter)
    n = fitter.ndim
    th0 = np.asarray(theta0, float)
    if th0.shape != (n,):
        raise BBaroloError(f"optimize: x0 has {th0.size} values, the free parameters are {n}.")
    u0 = np.where(np.isfinite(th0), space.u(np.where(np.isfinite(th0), th0, 0.)), 0.5)
    u0 = np.clip(np.where(np.isfinite(u0), u0, 0.5), space.lo, space.hi)
    rng = np.random.default_rng(seed)
    toc = time.time()
    with evaluator(fitter, like_kwargs, int(threads)) as loglikes:
        obj = _Objective(space, loglikes, posterior, _Progress(verbose, posterior))
        if not obj(u0) < _BIG:
            raise BBaroloError("optimize: the likelihood is not finite at the starting point (x0).")
        starts = []
        if method in GLOBAL_METHODS:
            import inspect
            opts = dict(options or {})
            opts.setdefault('rng' if 'rng' in inspect.signature(sopt.differential_evolution).parameters else 'seed',
                            rng)
            opts.setdefault('polish', False)
            try:
                de = sopt.differential_evolution(obj, list(zip(space.lo, space.hi)), x0=u0, updating='deferred',
                                                 workers=lambda func, pop: obj.batch(list(pop)), **opts)
            except TypeError as e:
                if 'unexpected keyword' not in str(e):
                    raise
                raise BBaroloError(f"optimize: differential_evolution: {e} (RUNOPTS, or options)") from None
            res = _local(obj, 'L-BFGS-B', np.asarray(de.x, float), step, None)
            res.message = f"differential evolution: {de.message}; L-BFGS-B: {res.message}"
            res.nit = int(getattr(de, 'nit', 0)) + int(getattr(res, 'nit', 0) or 0)
            starts.append(float(res.fun))
        else:
            res = None
            for k in range(max(1, int(nstart))):
                us = u0 if k == 0 else rng.uniform(space.lo, space.hi)
                if k > 0 and not obj(us) < _BIG:
                    continue
                r = _local(obj, method, us, step, options)
                starts.append(float(r.fun))
                vprint(verbose and nstart > 1, f"  start {k + 1}: -log L = {float(r.fun):.6g} ({r.message})")
                if res is None or r.fun < res.fun:
                    res = r
        # (the best point evaluated, e.g. by a gradient: no worse than the result of the method)
        fbest, ubest = obj.best
        u = np.clip(np.asarray(res.x, float), space.lo, space.hi)
        if ubest is not None and fbest < obj(u) - 1e-9:
            u = ubest
        x = space.theta(u)
        out = OptimizeResult(x=x, u=u, method=method, success=bool(getattr(res, 'success', True)),
                             message=str(getattr(res, 'message', '')), nit=int(getattr(res, 'nit', 0) or 0),
                             starts=starts, posterior=bool(posterior))
        logl = float(loglikes([x])[0])
        obj.nfev += 1
        out.logl, out.logprior = logl, space.logprior(x)
        out.fun = -logl - (out.logprior if posterior else 0.)
        tol = np.full(n, 1e-3)                   # (at a bound: closer than its error, or 0.1% of the prior)
        if errors:
            H, h = _hessian(obj, x, verbose, showbar)
            cov, cov_u, samples, reg = _laplace(space, x, H, int(nsamples), seed)
            out.hessian, out.steps, out.cov, out.samples, out.regularised = H, h, cov, samples, reg
            out.errors = np.sqrt(np.diag(cov))
            tol = np.maximum(tol, np.sqrt(np.diag(cov_u)))
        else:
            out.hessian = out.steps = out.cov = out.errors = None
            out.samples = x[None].copy()
            out.regularised = False
        out.at_bound = [nm for nm, ui, a, b, t in zip(space.names, u, space.lo, space.hi, tol)
                        if ui - a < t or b - ui < t]
        out.nfev = obj.nfev
    out.time = time.time() - toc
    return out


def summary(res, verbose=True, taskname=""):
    """ Prints how the optimisation went (method, likelihoods, time, the parameters at a bound of their prior). """
    if not verbose:
        return
    dt = res.time
    dt = f"{dt:.1f} seconds" if dt < 60 else f"{dt / 60.:.2f} minutes"
    what = "maximum a posteriori" if res.posterior else "maximum likelihood"
    print(f"\n{taskname}Optimisation with {res.method} ({what}): {res.nfev} likelihoods in {dt}; "
          f"log-likelihood {res.logl:.6g}.\n  {res.message}")
    if len(res.starts) > 1:
        print("  -log L of the starts: " + ", ".join(f"{s:.6g}" for s in res.starts))
    if res.at_bound:
        print(f"WARNING: at a bound of the prior: {', '.join(res.at_bound)} (the errors of the Laplace "
              "approximation\n         are not reliable there; widen the prior?)")
    if res.cov is not None and res.regularised:
        print("WARNING: the likelihood does not constrain some directions of the parameters (the Hessian is not\n"
              "         positive definite there, e.g. a parameter without effect): their errors are the width of "
              "the prior.")
