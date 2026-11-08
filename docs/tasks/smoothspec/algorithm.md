# SMOOTHSPEC: how it works

SMOOTHSPEC degrades the spectral resolution of a datacube: every spectrum is convolved with a
one-dimensional window of a few channels. The input is the cube and the window; the output is the
smoothed cube, on the same grid, written as a FITS file, with its channels optionally averaged in blocks
(REDUCE). The spatial axes are not touched. The cube in memory is not replaced: the tasks that run after
SMOOTHSPEC in the same run analyse the original cube.

## The window

The window is a set of $k$ weights $w_j$, $j = 0, \dots, k-1$, chosen with
{ref}`WINDOW_TYPE <par-smoothspec-window-type>` (default `HANNING`) and
{ref}`WINDOW_SIZE <par-smoothspec-window-size>` (default 3). For all windows except `GAUSSIAN`,
$k$ is the integer part of WINDOW_SIZE, which must be odd. With

$$
h = \frac{k-1}{2}, \qquad u_j = j - h, \qquad N = k - 1,
$$

($u_j$ is the offset of weight $j$ from the central channel, in channels) the unnormalised weights are:

| Window | $\tilde w_j$ |
|---|---|
| `HANNING` | $\tfrac12 + \tfrac12 \cos\!\left( \dfrac{2\pi u_j}{k+1} \right)$ |
| `HANNING2` | $\tfrac12 - \tfrac12 \cos\!\left( \dfrac{2\pi j}{N} \right)$ |
| `BOXCAR`, `TOPHAT` | $1$ |
| `BARTLETT` | $1 - \left| \dfrac{j - N/2}{N/2} \right|$ |
| `WELCH` | $1 - \left( \dfrac{j - N/2}{N/2} \right)^2$ |
| `BLACKMAN` | $0.42659 - 0.49656 \cos\dfrac{2\pi j}{N} + 0.076849 \cos\dfrac{4\pi j}{N}$ |
| `FLATTOP` | see below |
| `GAUSSIAN` | $\exp\!\left( -\dfrac{u_j^2}{2\sigma^2} \right)$ |

For `GAUSSIAN`, WINDOW_SIZE is the FWHM in channels (any value, not necessarily an integer or odd):
$\sigma = {\rm FWHM}/(2\sqrt{2\ln 2})$ channels and $k = 2\lfloor 5\sigma \rfloor + 1$, i.e. the
Gaussian is truncated at about $5\sigma$. The `FLATTOP` window is

$$
\tilde w_j = 0.21557895 - 0.41663158 \cos\frac{2\pi j}{N} + 0.277263158 \cos\frac{4\pi j}{N}
- 0.083578957 \cos\frac{6\pi j}{N} + 0.006947368 \cos\frac{8\pi j}{N}.
$$

All the weights are then normalised to a unit sum,

$$
w_j = \frac{\tilde w_j}{\sum_{l=0}^{k-1} \tilde w_l}.
$$

The windows `HANNING2`, `BARTLETT` and `WELCH` are zero at both ends, and `BLACKMAN` nearly so
($\tilde w_0 \approx 0.007$), so their effective width is $k-2$ channels: `HANNING2` of 5 channels has
the weights 1/4, 1/2, 1/4 of `HANNING` of 3 channels. `FLATTOP` has small negative weights. Examples of
the normalised weights:

- `HANNING`, $k=3$: 1/4, 1/2, 1/4; $k=5$: 1/12, 1/4, 1/3, 1/4, 1/12;
- `BOXCAR`, $k=5$: 1/5 each;
- `BARTLETT`, $k=5$: 0, 1/4, 1/2, 1/4, 0.

:::{note}
With WINDOW_SIZE 1 all the windows but `GAUSSIAN` leave the cube unchanged (a single weight of 1; in
BBarolo 1.8 `HANNING2`, `BARTLETT`, `WELCH`, `BLACKMAN` and `FLATTOP` had $N = 0$ in the formulae above and
gave a cube of NaN). `GAUSSIAN` with WINDOW_SIZE 1 is a Gaussian with a FWHM of one channel, over 5
channels (weights of about 0.06, 0.89, 0.06). WINDOW_SIZE must be positive.
:::

## The convolution

Each spectrum is smoothed as

$$
S'(x, y, z) = \sum_{j=0}^{k-1} w_j \, S(x, y, z + j - h),
$$

where $S$ is the input intensity (units of the cube), $z$ the channel and $S'$ the smoothed intensity,
in the same units. All the windows are symmetric, so this is a convolution. The channels beyond the ends
of the spectrum count as 0: the first and last $h$ channels are pulled towards zero. Blank voxels (NaN)
count as 0, and the output has no blanks: a NaN voxel of the input becomes the weighted sum of its
neighbours. Since $\sum_j w_j = 1$, the integral of each spectrum over the channels is conserved
(apart from the edges), and the units do not change.

## REDUCE and the output

With {ref}`REDUCE <par-smoothspec-reduce>` on (any value except `false`) and $k > 1$, the smoothed cube is
averaged over blocks of $M$ channels,

$$
M = \begin{cases} \lfloor (k+1)/2 \rfloor & \texttt{HANNING},\\ \lfloor k \rfloor & \text{other windows,}
\end{cases}
\qquad
S''(x, y, Z) = \frac{1}{M} \sum_{q=0}^{M-1} S'(x, y, M Z + q),
$$

with $k$ = WINDOW_SIZE (for `GAUSSIAN`, the FWHM, not the length of the truncated window) and $Z$ the
channel of the reduced cube. A Hanning window of $k$ channels has a FWHM of $(k+1)/2$ channels, so the
reduced channels are about as wide as the spectral resolution. A block whose sum is exactly 0, or contains
a NaN, is NaN; the channels at the end that do not fill a block are dropped. The header has
${\rm CDELT3}' = M \,{\rm CDELT3}$ and ${\rm CRPIX3}' = \tfrac12 + ({\rm CRPIX3} - \tfrac12)/M$.

The cube is written in {ref}`OUTFOLDER <par-all-outfolder>` as `<prefix>_h<k>.fits`, $k$ being the integer
part of WINDOW_SIZE (`<prefix>_h<k>_red.fits` with REDUCE), with the header of the input and a HISTORY card
naming the window. The same window smooths the cube for the mask `SMOOTHSPEC&SEARCH` (see
{ref}`MASK <par-makemask-mask>`).

## Differences from BBarolo 1.8

The windows, the convolution and REDUCE are those of BBarolo 1.8 and give the same cubes to floating-point
precision, with one exception: a window of one channel leaves the cube unchanged (`HANNING2`, `BARTLETT`,
`WELCH`, `BLACKMAN` and `FLATTOP` gave a cube of NaN). As in 1.8, an even WINDOW_SIZE (except for
`GAUSSIAN`) and an unknown WINDOW_TYPE stop the run in the check of the parameters, and so does a
WINDOW_SIZE that is not positive.

## References

- Di Teodoro E. M., Fraternali F., 2015, MNRAS, 451, 3021:
  [2015MNRAS.451.3021D](https://ui.adsabs.harvard.edu/abs/2015MNRAS.451.3021D)
- Harris F. J., 1978, Proceedings of the IEEE, 66, 51 (window functions):
  [doi:10.1109/PROC.1978.10837](https://doi.org/10.1109/PROC.1978.10837)

## In the code

- `bbarolo/bbutils.py`: `window_kernel` (the weights), `Smooth1D` (the convolution along one axis).
- `bbarolo/smooth.py`: `SpectralSmooth3D` (`smooth`, `fitswrite`).
- `bbarolo/cube.py`: `Cube.Reduce` (REDUCE).
- `bbarolo/tasks.py`: `_compute_smoothspec`, `_write_smoothspec`.
