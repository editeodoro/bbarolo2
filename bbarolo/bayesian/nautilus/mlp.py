"""Module implementing a small multi-layer perceptron regressor.

It replaces `sklearn.neural_network.MLPRegressor` for the emulators of the
neural network-based bounds, with the same model and training: ReLU hidden
layers and an identity output, squared loss with an L2 penalty `alpha`, Glorot
uniform initialisation, mini-batches of at most 200 points reshuffled every
epoch, the Adam optimiser, and the same stopping rule (`max_iter` epochs, or
`n_iter_no_change` epochs without an improvement of the loss by more than
`tol`). The training loop is compiled with numba, which removes the overhead
of scikit-learn for the small networks and batches used here.
"""

import numba as nb
import numpy as np


@nb.njit(cache=True, nogil=True)
def _fit(x, y, params, sizes, batch_size, learning_rate, beta_1, beta_2,
         epsilon, alpha, max_iter, tol, n_iter_no_change, seed):
    """Train the network whose weights and biases are `params` (in place).

    The weights of layer l are params[w_l:w_l + n_l * n_(l+1)] as an array of
    shape (n_l, n_(l+1)), after them the biases of all layers.

    Returns
    -------
    n_iter : int
        Number of epochs.
    loss_curve : numpy.ndarray
        Loss of each epoch.

    """
    np.random.seed(seed)
    n = x.shape[0]
    n_layers = len(sizes) - 1
    w_off = np.zeros(n_layers, np.int64)
    b_off = np.zeros(n_layers, np.int64)
    off = 0
    for i in range(n_layers):
        w_off[i] = off
        off += sizes[i] * sizes[i + 1]
    for i in range(n_layers):
        b_off[i] = off
        off += sizes[i + 1]

    m = np.zeros_like(params)
    v = np.zeros_like(params)
    grad = np.zeros_like(params)
    loss_curve = np.zeros(max_iter)
    best_loss = np.inf
    no_improvement = 0
    t = 0
    n_iter = 0

    for epoch in range(max_iter):
        order = np.random.permutation(n)
        accumulated_loss = 0.0
        for start in range(0, n, batch_size):
            stop = min(start + batch_size, n)
            n_batch = stop - start
            index = order[start:stop]

            # Forward pass.
            activations = [x[index]]
            for i in range(n_layers):
                w = params[w_off[i]:w_off[i] + sizes[i] * sizes[i + 1]].reshape(
                    (sizes[i], sizes[i + 1]))
                z = np.dot(activations[i], w) + params[
                    b_off[i]:b_off[i] + sizes[i + 1]]
                if i < n_layers - 1:
                    z = np.maximum(z, 0.0)
                activations.append(z)

            # Loss.
            diff = activations[n_layers][:, 0] - y[index]
            loss = 0.5 * np.mean(diff**2)
            if alpha > 0:
                loss += 0.5 * alpha * np.sum(params[:b_off[0]]**2) / n_batch
            accumulated_loss += loss * n_batch

            # Backward pass.
            delta = diff.reshape((n_batch, 1)).copy()
            for i in range(n_layers - 1, -1, -1):
                size = sizes[i] * sizes[i + 1]
                w = params[w_off[i]:w_off[i] + size].reshape(
                    (sizes[i], sizes[i + 1]))
                g = np.dot(activations[i].T, delta)
                if alpha > 0:
                    g += alpha * w
                grad[w_off[i]:w_off[i] + size] = g.ravel() / n_batch
                grad[b_off[i]:b_off[i] + sizes[i + 1]] = (
                    delta.sum(axis=0) / n_batch)
                if i > 0:
                    delta = np.dot(delta, w.T) * (activations[i] > 0)

            # Adam.
            t += 1
            lr = (learning_rate * np.sqrt(1 - beta_2**t) / (1 - beta_1**t))
            for k in range(params.size):
                m[k] = beta_1 * m[k] + (1 - beta_1) * grad[k]
                v[k] = beta_2 * v[k] + (1 - beta_2) * grad[k] * grad[k]
                params[k] -= lr * m[k] / (np.sqrt(v[k]) + epsilon)

        loss = accumulated_loss / n
        loss_curve[epoch] = loss
        n_iter = epoch + 1
        if loss > best_loss - tol:
            no_improvement += 1
        else:
            no_improvement = 0
        if loss < best_loss:
            best_loss = loss
        if no_improvement > n_iter_no_change:
            break

    return n_iter, loss_curve[:n_iter]


class MLPRegressor:
    """Multi-layer perceptron regressor with ReLU hidden layers.

    The arguments are those of `sklearn.neural_network.MLPRegressor` that
    apply to this model, with the same defaults.

    Attributes
    ----------
    coefs_ : list
        Weights of each layer.
    intercepts_ : list
        Biases of each layer.
    n_iter_ : int
        Number of epochs of the training.
    loss_curve_ : numpy.ndarray
        Loss of each epoch.

    """

    def __init__(self, hidden_layer_sizes=(100, ), alpha=1e-4,
                 batch_size='auto', learning_rate_init=1e-3, max_iter=200,
                 tol=1e-4, n_iter_no_change=10, beta_1=0.9, beta_2=0.999,
                 epsilon=1e-8, random_state=None):
        """Initialize the regressor."""
        self.hidden_layer_sizes = tuple(np.atleast_1d(hidden_layer_sizes))
        self.alpha = alpha
        self.batch_size = batch_size
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.tol = tol
        self.n_iter_no_change = n_iter_no_change
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.random_state = random_state

    def fit(self, x, y):
        """Train the network.

        Parameters
        ----------
        x : numpy.ndarray
            Input coordinates, shape (n_points, n_dim).
        y : numpy.ndarray
            Target values, shape (n_points, ).

        Returns
        -------
        self : MLPRegressor
            The trained regressor.

        """
        x = np.ascontiguousarray(x, dtype=float)
        y = np.ascontiguousarray(y, dtype=float).ravel()
        rng = np.random.default_rng(self.random_state)
        sizes = np.array((x.shape[1], ) + self.hidden_layer_sizes + (1, ),
                         dtype=np.int64)
        coefs, intercepts = [], []
        for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
            bound = np.sqrt(6.0 / (fan_in + fan_out))
            coefs.append(rng.uniform(-bound, bound, (fan_in, fan_out)))
            intercepts.append(rng.uniform(-bound, bound, fan_out))
        params = np.concatenate([c.ravel() for c in coefs] + intercepts)
        if self.batch_size == 'auto':
            batch_size = min(200, len(x))
        else:
            batch_size = int(np.clip(self.batch_size, 1, len(x)))
        self.n_iter_, self.loss_curve_ = _fit(
            x, y, params, sizes, batch_size, float(self.learning_rate_init),
            float(self.beta_1), float(self.beta_2), float(self.epsilon),
            float(self.alpha), int(self.max_iter), float(self.tol),
            int(self.n_iter_no_change), int(rng.integers(2**31 - 1)))
        self.n_layers_ = len(sizes)
        self.coefs_, self.intercepts_ = [], []
        off = 0
        for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
            self.coefs_.append(
                params[off:off + fan_in * fan_out].reshape(fan_in, fan_out))
            off += fan_in * fan_out
        for fan_out in sizes[1:]:
            self.intercepts_.append(params[off:off + fan_out])
            off += fan_out
        return self

    def predict(self, x):
        """Predict the target values.

        Parameters
        ----------
        x : numpy.ndarray
            Input coordinates, shape (n_points, n_dim).

        Returns
        -------
        y : numpy.ndarray
            Predicted values, shape (n_points, ).

        """
        a = np.asarray(x, dtype=float)
        for i, (w, b) in enumerate(zip(self.coefs_, self.intercepts_)):
            a = a @ w + b
            if i < len(self.coefs_) - 1:
                np.maximum(a, 0, out=a)
        return a[:, 0]
