# STATS: how it works

STATS computes the statistics of the voxels of a cube: mean, standard deviation, median, MADFM, maximum
and minimum. They describe the noise of the cube, and the same statistics give the thresholds of
{doc}`SEARCH <../search/algorithm>`, of the {doc}`masks <../makemask/algorithm>` of the source finder and
the noise of 3DFIT. The input is the cube; the output is `<prefix>_stats.txt`, also printed. If STATS
runs, its statistics are used by the following tasks of the run.

## The voxels

The statistics are computed over the voxels that are not blank: voxels with value 0 or NaN are excluded
(NaN values are read as 0). Let $F_k$, $k = 1, \dots, N$, be the values of the remaining voxels, in the
flux units of the cube.

## The estimators

$$
\bar F = \frac{1}{N}\sum_k F_k, \qquad
s = \sqrt{\frac{1}{N}\sum_k F_k^2 - \bar F^2} ,
$$

$$
\tilde F = \mathrm{median}(F_k), \qquad
\mathrm{MADFM} = \mathrm{median}\bigl(|F_k - \tilde F|\bigr) ,
$$

with the median of an even number of values the mean of the two central ones; $\max_k F_k$ and
$\min_k F_k$. All are computed in double precision and are in the flux units of the cube.

For a Gaussian distribution $\mathrm{MADFM} = 0.6744888\,\sigma$, so the MADFM is converted to a standard
deviation as $\mathrm{MADFM}/0.6744888$.

The tasks use a middle value $m$ and a spread $\sigma$, chosen by
{ref}`FLAGROBUSTSTATS <par-all-flagrobuststats>`:

$$
(m, \sigma) =
\begin{cases}
(\tilde F,\ \mathrm{MADFM}/0.6744888) & \text{robust statistics (default)},\\
(\bar F,\ s) & \text{otherwise.}
\end{cases}
$$

The median and the MADFM are less sensitive to the emission than the mean and the standard deviation,
since the emission occupies a tail of the distribution.

(stats-iternoise)=
## Iterative noise

With {ref}`ITERNOISE <par-stats-iternoise>` the emission is excluded iteratively. Starting from the
statistics $m_0$, $\sigma_0$ of all the voxels, at each iteration $j = 1, 2, \dots$ the voxels with

$$
|F_k| > m_{j-1} + 3\,\sigma_{j-1}
$$

are excluded, in addition to those excluded before, and the statistics $m_j$, $\sigma_j$ are computed
again on the voxels left. The iterations stop when

$$
\frac{|\sigma_j - \sigma_{j-1}|}{\sigma_{j-1}} < 10^{-3} ,
$$

or after 20 iterations. The cut uses the absolute value, so bright negative voxels (e.g. artefacts) are
excluded as well as the emission. All the statistics, the maximum and the minimum included, are then
those of the voxels left.

## The output

`<prefix>_stats.txt` has one line with the mean, median, standard deviation, MADFM, MADFM/0.6744888,
maximum and minimum, in the flux units of the cube. The thresholds of SEARCH are computed from $m$ and
$\sigma$ (see {doc}`../search/algorithm`).

## Differences from BBarolo 1.8

- **ITERNOISE** stops at $|\Delta\sigma|/\sigma < 10^{-3}$. The convergence test of 1.8 added $\sigma$
  to the relative change, so it always ran 20 iterations; on NGC 2403 it now stops after 10 iterations
  (4 with robust statistics), with a noise smaller by $4\times10^{-4}$ ($4\times10^{-5}$) in relative
  terms.
- The maximum and the minimum are those of the voxels used; 1.8 started them from the first voxel of the
  cube even when it was blank.
- The statistics are computed in double precision (1.8: in the precision of the data, usually single).

The full list is in {doc}`../../reference/changes`.

## References

- Whiting M. T., 2012, MNRAS, 421, 3242 (Duchamp, whose statistics BBarolo uses):
  [2012MNRAS.421.3242W](https://ui.adsabs.harvard.edu/abs/2012MNRAS.421.3242W)

## In the code

- `bbarolo/stats.py`: `findAllStats`, `madfmToSigma`, `Stats` (`calculate`, `getMiddle`, `getSpread`,
  `setThresholdSNR`, `tofile`).
- `bbarolo/cube.py`: `Cube.setCubeStats` (the blank voxels and ITERNOISE).
- `bbarolo/tasks.py`: `_step_stats`, `Stats`.
