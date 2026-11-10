"""
The emission of BAYESIANWIND: the amplitudes of a linear model (the emissivities of the cells of the wind, the
densities of the rings of a fixed disk), solved at every likelihood call, and the log-likelihood of the other
parameters marginalised over them.

For the data d (weights W: the inverse noise variance times the correlation factor of the likelihood) and a model
A x linear in the amplitudes x, with the normal equations G = A'WA, b = A'Wd, c0 = d'Wd,

    -2 ln L(x) = c0 - 2 b'x + x'Gx + const.

The default prior of the amplitudes (log_marginal_gauss) is Gaussian and independent: x_k ~ N(m_k, (kappa m_k)^2),
a mean emissivity m (EMSCALE, times a power law of radius) and a fractional scatter kappa (EMCLUMP: the clumpiness;
for clumps of one brightness filling a fraction f of the cells, kappa^2 = (1 - f)/f). The marginal likelihood is then
exact (a Gaussian integral, one Cholesky factorisation), smooth in the other parameters, and it carries the Occam
factor that breaks the degeneracy of a free emission: a cone too wide or a wind too fast has cells that the prior
expects to emit (mean m) and the data find empty, each costing about ln(S/N) plus 1/(2 kappa^2). The amplitudes are
not forced to be positive: their posterior means can be slightly negative in empty cells, within the noise (as the
pixels of the sources of strong-lensing inversions, Suyu et al. 2006). With the hyperparameters inferred, the
evidence peaks at the true geometry of patchy mock winds.

The alternative prior (log_marginal) is exponential, x_k >= 0 of mean mu_k (the maximum-entropy prior of a positive
quantity): the posterior is a truncated Gaussian, whose integral is approximated (Laplace at the non-negative
maximum a posteriori, solve_nonneg, with the truncation of each amplitude). The approximation treats the amplitudes
at zero as independent, which overestimates the evidence when many correlated cells are empty (weak data, cells
overlapping through the beam): with correlated noise it favoured cones wider than the truth. log_profile is the
maximum over x >= 0 (no prior, no Occam factor): it is flat for all the models that cover the data (the degeneracy of
the voxel weighting of MoKA-3D).
"""

import functools
import math
import numpy as np
from scipy import linalg
from scipy.special import erfcx, log_ndtr

LOG2PI = math.log(2. * math.pi)


def _chol(M):
    """Cholesky factor (lower) of the symmetric matrix M, with a tiny jitter if it is not positive definite."""
    try:
        return linalg.cholesky(M, lower=True, check_finite=False)
    except linalg.LinAlgError:
        d = np.diag(M)
        eps = 1e-10 * (np.max(d) if d.size and np.max(d) > 0 else 1.)
        for _ in range(8):
            try:
                return linalg.cholesky(M + eps * np.eye(M.shape[0]), lower=True, check_finite=False)
            except linalg.LinAlgError:
                eps *= 100.
        raise


def _solve_free(G, h, F):
    x = np.zeros(h.size)
    if np.any(F):
        L = _chol(G[np.ix_(F, F)])
        x[F] = linalg.cho_solve((L, True), h[F], check_finite=False)
    return x


def solve_nonneg(G, h, free=None, maxiter=None):
    """The minimum of x'Gx/2 - h'x for x >= 0 (G symmetric positive semi-definite, with positive diagonal), by block
    principal pivoting (Kim & Park 2011), from the free set free (a boolean array; default the amplitudes with
    h > 0). Returns (x, free)."""
    n = h.size
    if n == 0:
        return np.zeros(0), np.zeros(0, dtype=bool)
    F = (h > 0) if free is None else np.asarray(free, dtype=bool).copy()
    maxiter = 5 * n + 10 if maxiter is None else maxiter
    alpha, beta = 3, n + 1
    hmax = max(float(np.max(np.abs(h))), 1e-300)
    for _ in range(maxiter):
        x = _solve_free(G, h, F)
        y = G @ x - h
        bad_x = F & (x < -1e-12 * max(float(np.max(np.abs(x))), 1e-300))
        bad_y = ~F & (y < -1e-10 * hmax)
        nbad = int(np.sum(bad_x) + np.sum(bad_y))
        if nbad == 0:
            x[~F] = 0.
            x[F] = np.maximum(x[F], 0.)
            return x, F
        if nbad < beta:
            beta, alpha = nbad, 3
            F = F ^ (bad_x | bad_y)
        elif alpha > 0:
            alpha -= 1
            F = F ^ (bad_x | bad_y)
        else:                                           # (the backup rule: swap the last infeasible amplitude)
            k = np.flatnonzero(bad_x | bad_y)[-1]
            F[k] = ~F[k]
    # (no convergence: a projected solution of the last free set, at least feasible)
    x = _solve_free(G, h, F)
    x = np.maximum(x, 0.)
    return x, x > 0


def log_marginal(G, b, c0, mu, free=None):
    """The Laplace approximation of the log marginal likelihood over the amplitudes x >= 0 (see the module docstring),
    -c0/2 included, for exponential priors of means mu (> 0; amplitudes with mu = 0 or no data, G_kk = 0, are left
    out: they do not change Z). Returns a dict: logz, x (the maximum a posteriori), free (the free amplitudes),
    chi2 (c0 - 2b'x + x'Gx, at x), and the parts of logz: occam (logz + chi2/2) and prior (sum of ln p(x))."""
    n = b.size
    mu = np.asarray(mu, dtype=np.float64)
    use = (np.diag(G) > 0) & (mu > 0)
    x = np.zeros(n)
    out = dict(x=x, free=np.zeros(n, dtype=bool))
    if not np.any(use):
        out.update(logz=-0.5 * c0, chi2=c0, occam=0., prior=0.)
        return out
    idx = np.flatnonzero(use)
    Gu, bu, mu_u = G[np.ix_(idx, idx)], b[idx], mu[idx]
    h = bu - 1. / mu_u
    xu, F = solve_nonneg(Gu, h, None if free is None else np.asarray(free)[idx])
    q = 0.5 * xu @ Gu @ xu - h @ xu                     # (the minimum of x'Gx/2 - h'x)
    chi2 = c0 - 2. * bu @ xu + xu @ Gu @ xu
    logz = -0.5 * c0 - np.sum(np.log(mu_u)) - q
    nf = int(np.sum(F))
    I = ~F
    if nf:
        L = _chol(Gu[np.ix_(F, F)])
        logz += 0.5 * nf * LOG2PI - np.sum(np.log(np.diag(L)))
        Linv = linalg.solve_triangular(L, np.eye(nf), lower=True, check_finite=False)
        s = np.sqrt(np.sum(Linv * Linv, axis=0))        # (marginal posterior widths of the free amplitudes)
        logz += np.sum(log_ndtr(xu[F] / s))
        if np.any(I):
            X = Linv @ Gu[np.ix_(F, I)]
            c = np.diag(Gu)[I] - np.sum(X * X, axis=0)
        else:
            c = np.zeros(0)
    else:
        c = np.diag(Gu)[I]
    if np.any(I):
        g = (Gu @ xu - h)[I]
        c = np.maximum(c, 1e-12 * np.diag(Gu)[I])
        logz += np.sum(_log_halfgauss(g, c))
    x[idx] = xu
    out["free"][idx] = F
    prior = float(-np.sum(np.log(mu_u)) - np.sum(xu / mu_u))
    out.update(logz=float(logz), chi2=float(chi2), occam=float(logz + 0.5 * chi2), prior=prior)
    return out


def _log_halfgauss(g, c):
    """ln of the integral over t >= 0 of exp(-g t - c t^2/2), c > 0: ln(sqrt(pi/(2c)) erfcx(g/sqrt(2c)))."""
    z = g / np.sqrt(2. * c)
    return 0.5 * np.log(math.pi / (2. * c)) + np.log(erfcx(z))


def log_marginal_diag(g, b, c0, mu):
    """The exact log marginal likelihood for independent amplitudes (G diagonal, g = diag(G)): -c0/2 plus, for each
    amplitude, ln of the integral over x >= 0 of exp(b x - g x^2/2) exp(-x/mu)/mu. Returns (logz, x) with x the
    maximum a posteriori."""
    g = np.asarray(g, dtype=np.float64)
    use = (g > 0) & (mu > 0)
    x = np.zeros(g.size)
    logz = -0.5 * c0
    if np.any(use):
        gu, mu_u = g[use], np.asarray(mu, dtype=np.float64)[use]
        h = b[use] - 1. / mu_u
        x[use] = np.maximum(h / gu, 0.)
        # ln int_0^inf exp(h x - g x^2/2) dx = h^2/(2g) + ln sqrt(2 pi/g) + ln Phi(h/sqrt(g))
        z = h / np.sqrt(gu)
        logz += float(np.sum(0.5 * z * z + 0.5 * np.log(2 * math.pi / gu) + log_ndtr(z) - np.log(mu_u)))
    return logz, x


def log_profile(G, b, c0):
    """The profile log-likelihood: the maximum over x >= 0 of -(c0 - 2b'x + x'Gx)/2 (non-negative least squares, no
    prior). Returns (logL, x)."""
    n = b.size
    use = np.diag(G) > 0
    x = np.zeros(n)
    if not np.any(use):
        return -0.5 * c0, x
    idx = np.flatnonzero(use)
    xu, _ = solve_nonneg(G[np.ix_(idx, idx)], b[idx])
    x[idx] = xu
    chi2 = c0 - 2. * b[idx] @ xu + xu @ G[np.ix_(idx, idx)] @ xu
    return float(-0.5 * chi2), x


def log_marginal_gauss(G, b, c0, m, kappa):
    """The exact log marginal likelihood over the amplitudes x with independent Gaussian priors N(m_k, (kappa m_k)^2)
    (m > 0; amplitudes with m = 0 or no data, G_kk = 0, are left out: they do not change Z), -c0/2 included. Returns a
    dict: logz, x (the posterior mean, = the maximum a posteriori), sd (the posterior widths of the amplitudes),
    chi2 (c0 - 2b'x + x'Gx at x) and occam (logz + chi2/2)."""
    n = b.size
    m = np.asarray(m, dtype=np.float64)
    kappa = np.broadcast_to(np.asarray(kappa, dtype=np.float64), m.shape)
    use = (np.diag(G) > 0) & (m > 0)
    x = np.where(m > 0, m, 0.)
    sd = np.where(m > 0, kappa * m, 0.)
    if not np.any(use):
        chi2 = float(c0 - 2. * b @ x + x @ G @ x)
        return dict(logz=-0.5 * c0, x=x, sd=sd, chi2=chi2, occam=0.5 * (chi2 - c0))
    idx = np.flatnonzero(use)
    Gu, bu, mu, ku = G[np.ix_(idx, idx)], b[idx], m[idx], kappa[idx]
    P = 1. / (ku * mu) ** 2
    A = Gu + np.diag(P)
    h = bu + P * mu
    L = _chol(A)
    y = linalg.solve_triangular(L, h, lower=True, check_finite=False)
    xu = linalg.solve_triangular(L.T, y, lower=False, check_finite=False)
    logz = -0.5 * (c0 + np.sum(P * mu * mu)) + 0.5 * y @ y - np.sum(np.log(ku * mu)) - np.sum(np.log(np.diag(L)))
    Linv = linalg.solve_triangular(L, np.eye(idx.size), lower=True, check_finite=False)
    x[idx] = xu
    sd[idx] = np.sqrt(np.sum(Linv * Linv, axis=0))
    # (the amplitudes left out are at their prior means, which the data do not see: they do not enter chi2)
    xc = np.zeros(n)
    xc[idx] = xu
    chi2 = float(c0 - 2. * b @ xc + xc @ G @ xc)
    return dict(logz=float(logz), x=x, sd=sd, chi2=chi2, occam=float(logz + 0.5 * chi2))


def log_marginal_gauss_diag(g, b, c0, m, kappa):
    """log_marginal_gauss for independent amplitudes (G diagonal, g = diag(G)). Returns (logz, x)."""
    g = np.asarray(g, dtype=np.float64)
    m = np.broadcast_to(np.asarray(m, dtype=np.float64), g.shape)
    kappa = np.broadcast_to(np.asarray(kappa, dtype=np.float64), g.shape)
    use = (g > 0) & (m > 0)
    x = np.where(m > 0, m, 0.)
    logz = -0.5 * c0
    if np.any(use):
        gu, bu, mu, ku = g[use], b[use], m[use], kappa[use]
        P = 1. / (ku * mu) ** 2
        A = gu + P
        h = bu + P * mu
        x[use] = h / A
        logz += float(np.sum(-0.5 * P * mu * mu + 0.5 * h * h / A - np.log(ku * mu) - 0.5 * np.log(A)))
    return logz, x


# ---------------------------------------------------------------- the detection likelihood (emission voxel)
_DET_LX = (math.log(1e-6), math.log(1e4), 2000)     # ln(x) of the table: x = the expected flux / noise
_DET_LK = (math.log(1e-3), math.log(30.), 70)       # ln(kappa): the fractional scatter of the flux of a voxel
_DET_Z = np.linspace(-8., 8., 321)                  # the nodes of the log-normal (standard normal deviates)


@functools.lru_cache(maxsize=8)
def detection_table(snrcut):
    """The table of the detection likelihood for the threshold snrcut (in units of the noise): (ln x, ln kappa, ln p,
    ln(1 - p)) with p(x, kappa) = E[Phi(F - snrcut)], the probability that a voxel whose flux F (in units of the noise)
    is log-normal, of mean x and fractional scatter kappa, plus Gaussian noise of unit rms, is above snrcut. kappa -> 0
    is the probit Phi(x - snrcut) of a smooth emission; x -> 0 gives the false-positive rate Phi(-snrcut)."""
    from scipy.special import logsumexp
    lx = np.linspace(*_DET_LX)
    lk = np.linspace(*_DET_LK)
    lw = -0.5 * _DET_Z ** 2
    lw = lw - logsumexp(lw)
    LP, L1 = np.empty((lk.size, lx.size)), np.empty((lk.size, lx.size))
    for i, k in enumerate(lk):
        s2 = math.log1p(math.exp(2. * k))                  # the variance of ln F
        F = np.exp(lx[:, None] - 0.5 * s2 + math.sqrt(s2) * _DET_Z[None, :])
        LP[i] = logsumexp(log_ndtr(F - snrcut) + lw, axis=1)
        L1[i] = logsumexp(log_ndtr(snrcut - F) + lw, axis=1)
    return lx, lk, LP, L1


def log_detection(x, det, kappa, snrcut):
    """The log-likelihood of the detections det (bool, a voxel above snrcut times the noise) given the expected flux
    of each voxel x (in units of the noise, >= 0), each voxel independent: sum of ln p over the detections and of
    ln(1 - p) over the others, p = the probability of a detection of a voxel whose flux is log-normal of mean x and
    fractional scatter kappa (the clumpiness) plus the noise (detection_table; kappa below 1e-3 or above 30 is taken
    at those limits, x below 1e-6 at 1e-6: the false-positive rate). An undetected voxel where the model predicts
    flux costs ln(1 - p) (for kappa -> 0 about -(x - snrcut)^2/2 above the threshold): the penalty of the models that
    predict emission where the data have none. A detection where the model has no flux costs ln Phi(-snrcut)."""
    lx, lk, LP, L1 = detection_table(float(snrcut))
    j = min(max((math.log(kappa) - lk[0]) / (lk[1] - lk[0]), 0.), lk.size - 1.000001)
    j0 = int(j)
    a = j - j0
    lp = (1. - a) * LP[j0] + a * LP[j0 + 1]
    l1 = (1. - a) * L1[j0] + a * L1[j0 + 1]
    lxv = np.log(np.maximum(np.asarray(x, dtype=np.float64), math.exp(lx[0])))
    return float(np.sum(np.interp(lxv[det], lx, lp)) + np.sum(np.interp(lxv[~det], lx, l1)))
