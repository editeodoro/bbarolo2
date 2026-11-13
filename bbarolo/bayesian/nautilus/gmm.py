"""Module implementing a two-component Gaussian mixture.

It replaces `sklearn.mixture.GaussianMixture(n_components=2, n_init=...)` for
splitting multi-ellipsoidal bounds, with the same model and fit: full
covariances regularised by `reg_covar` on the diagonal, k-means initial
responsibilities (k-means++ seeding, Lloyd iterations), expectation
maximisation until the mean log-likelihood changes by less than `tol`, and the
best of `n_init` starts.
"""

import numba as nb
import numpy as np
from scipy.linalg import cholesky, solve_triangular


@nb.njit(cache=True, nogil=True)
def _assign(points, centers, labels):
    """Label of the nearest of the two centers of each point."""
    n, n_dim = points.shape
    for i in range(n):
        d0 = 0.0
        d1 = 0.0
        for j in range(n_dim):
            a = points[i, j] - centers[0, j]
            b = points[i, j] - centers[1, j]
            d0 += a * a
            d1 += b * b
        labels[i] = 0 if d0 <= d1 else 1


@nb.njit(cache=True, nogil=True)
def _lloyd(points, centers, tol, n_iter_max):
    """Lloyd iterations of a two-cluster k-means from the centers (changed in
    place) until the centers move by at most `tol` (sum of squares), then the
    labels of the final centers, as scikit-learn."""
    n, n_dim = points.shape
    labels = np.zeros(n, np.int64)
    for _ in range(n_iter_max):
        _assign(points, centers, labels)
        new = np.zeros_like(centers)
        count = np.zeros(2)
        for i in range(n):
            new[labels[i]] += points[i]
            count[labels[i]] += 1
        shift = 0.0
        for k in range(2):
            if count[k] > 0:
                new[k] /= count[k]
            else:
                new[k] = centers[k]
            shift += np.sum((new[k] - centers[k])**2)
        centers[:] = new
        if shift <= tol:
            break
    _assign(points, centers, labels)
    return labels


def _kmeans_labels(points, rng, n_iter_max=300):
    """Labels of a two-cluster k-means (greedy k-means++ seeding, Lloyd
    iterations with the tolerance of scikit-learn)."""
    n = len(points)
    first = points[rng.integers(n)]
    d2 = np.sum((points - first)**2, axis=1)
    # Greedy k-means++: the best of 2 + log(2) candidates for the second
    # center, as scikit-learn.
    n_trials = 2 + int(np.log(2))
    candidates = rng.choice(n, size=n_trials, p=d2 / np.sum(d2)) \
        if np.sum(d2) > 0 else rng.integers(n, size=n_trials)
    costs = [np.sum(np.minimum(d2, np.sum((points - points[c])**2, axis=1)))
             for c in candidates]
    centers = np.array([first, points[candidates[np.argmin(costs)]]])
    tol = 1e-4 * np.mean(np.var(points, axis=0))
    return _lloyd(np.ascontiguousarray(points), centers, tol, n_iter_max)


def _m_step(points, resp, reg_covar):
    """Weights, means and covariances from the responsibilities."""
    nk = resp.sum(axis=0) + 10 * np.finfo(float).eps
    means = (resp.T @ points) / nk[:, None]
    covariances = np.empty((2, points.shape[1], points.shape[1]))
    for k in range(2):
        diff = points - means[k]
        covariances[k] = (resp[:, k, None] * diff).T @ diff / nk[k]
        covariances[k].flat[::points.shape[1] + 1] += reg_covar
    return nk / len(points), means, covariances


def _log_prob(points, weights, means, covariances):
    """Logarithm of the weighted density of each component at each point."""
    n_dim = points.shape[1]
    log_prob = np.empty((len(points), 2))
    for k in range(2):
        chol = cholesky(covariances[k], lower=True)
        y = solve_triangular(chol, (points - means[k]).T, lower=True)
        log_prob[:, k] = (-0.5 * np.sum(y**2, axis=0) -
                          np.sum(np.log(np.diag(chol))) -
                          0.5 * n_dim * np.log(2 * np.pi) + np.log(weights[k]))
    return log_prob


def two_component_gaussian_mixture(points, n_init=10, rng=None, tol=1e-3,
                                   reg_covar=1e-6, max_iter=100):
    """Fit a Gaussian mixture with two components.

    Parameters
    ----------
    points : numpy.ndarray
        Points, shape (n_points, n_dim).
    n_init : int, optional
        Number of starts; the fit with the highest likelihood is kept.
        Default is 10.
    rng : None or numpy.random.Generator, optional
        Random number generator. Default is None.
    tol : float, optional
        Convergence threshold on the mean log-likelihood. Default is 1e-3.
    reg_covar : float, optional
        Added to the diagonal of the covariances. Default is 1e-6.
    max_iter : int, optional
        Maximum number of iterations of each start. Default is 100.

    Returns
    -------
    weights : numpy.ndarray
        Weights of the components, shape (2, ).
    means : numpy.ndarray
        Means, shape (2, n_dim).
    covariances : numpy.ndarray
        Covariances, shape (2, n_dim, n_dim).

    """
    if rng is None:
        rng = np.random.default_rng()
    points = np.asarray(points, dtype=float)
    best, best_lower_bound = None, -np.inf
    for _ in range(n_init):
        labels = _kmeans_labels(points, rng)
        resp = np.zeros((len(points), 2))
        resp[np.arange(len(points)), labels] = 1
        params = _m_step(points, resp, reg_covar)
        lower_bound = -np.inf
        for _ in range(max_iter):
            log_prob = _log_prob(points, *params)
            log_max = np.max(log_prob, axis=1)
            log_norm = log_max + np.log(np.sum(
                np.exp(log_prob - log_max[:, None]), axis=1))
            previous, lower_bound = lower_bound, np.mean(log_norm)
            resp = np.exp(log_prob - log_norm[:, None])
            params = _m_step(points, resp, reg_covar)
            if abs(lower_bound - previous) < tol:
                break
        if lower_bound > best_lower_bound or best is None:
            best, best_lower_bound = params, lower_bound
    return best
