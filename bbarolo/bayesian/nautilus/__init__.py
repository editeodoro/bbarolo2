"""Neural Network-Boosted Importance Sampling for Bayesian Statistics.

nautilus (Lange 2023, MNRAS 525, 3181; github.com/johannesulf/nautilus), version 1.0.6 with the
changes of BBarolo (SAMPLER NAUTILUSBB), MIT licence (LICENSE in this folder):

- no scikit-learn: the networks of the bounds are `mlp.MLPRegressor` (compiled with numba, the model
  and training of scikit-learn's) and the bounds are split with `gmm.two_component_gaussian_mixture`;
  threadpoolctl is optional;
- a faster minimum-volume enclosing ellipsoid (rank-one updates of the inverse, same result);
- one cached ThreadpoolController (`pool.limit_threads`) instead of a new one at every batch;
- the networks drop the state of their training, and the processes of a pool that sample a bound
  return only the points: much less data through the pool.

The SAMPLER NAUTILUS of BBarolo uses the package nautilus-sampler instead.
"""

from .prior import Prior
from .sampler import Sampler

__author__ = 'Johannes U. Lange'
__email__ = 'jlange@american.edu'
__version__ = '1.0.6+bbarolo'
__all__ = ['Prior', 'Sampler']
