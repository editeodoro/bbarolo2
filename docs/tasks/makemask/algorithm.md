# MAKEMASK: how it works

A mask selects the voxels of a cube that contain genuine emission. MAKEMASK builds the mask set by
{ref}`MASK <par-makemask-mask>` and writes it in `<prefix>_mask.fits`. The same masks are built by every
task that needs one: 3DFIT (every METHOD), GALMOD and SPACEPAR compare model and data within the mask;
2DFIT, ELLPROF, REND3D and the maps use the data within it. This page describes all the types of mask.
The input is the cube; the output is a cube of 1 (in the mask) and 0 (outside), with the parameters of
the mask in the HISTORY cards of its header.

Throughout the page, $F(x, y, z)$ is the value of voxel $(x, y, z)$ in the flux units of the cube, with
$x$, $y$ the pixel and $z$ the channel; blank voxels (NaN) are read as 0. $m$ and $\sigma$ are a middle
value and a spread, measured as described in {doc}`../stats/algorithm`: the median and
MADFM/0.6744888 with {ref}`FLAGROBUSTSTATS <par-all-flagrobuststats>` true (default), otherwise the mean
and the standard deviation, both in flux units.

## Types of mask

### Masks of the source finder

`SEARCH`, `SMOOTH&SEARCH` and `SMOOTHSPEC&SEARCH` run the source finder of SEARCH, with all its
parameters (thresholds, merging, growth, rejection; see {doc}`../search/algorithm`), and the mask is
made of the voxels of the detections:

- `SEARCH` (default): the source finder on the cube. If SEARCH has already run in the same run, its
  detections are used. The noise is that of the cube, with {ref}`ITERNOISE <par-search-iternoise>` if
  set.
- `SMOOTH&SEARCH`: the source finder on a copy of the cube smoothed spatially (see
  {doc}`../smooth/algorithm`) from the beam of the data, $(B_{\rm maj}, B_{\rm min}, B_{\rm pa})$, to the
  beam
  $$
  B'_{\rm maj} = f\,B_{\rm maj}, \qquad B'_{\rm min} = f\,B_{\rm min}, \qquad B'_{\rm pa} = B_{\rm pa} ,
  $$
  with $f$ = {ref}`FACTOR <par-makemask-factor>` (2 if not given), or to the beam given by
  {ref}`NEWBEAM <par-smooth-newbeam>` (`NEWBEAM bmaj bmin bpa`, arcsec; its BPA, if not given, is that of the
  beam of the data, which is the header beam or {ref}`BEAM <par-all-beam>`). The noise, the thresholds and the defaults that depend on the beam
  (THRESHSPATIAL, MINPIX, MINVOXELS) are those of the smoothed cube and of its beam $B'$.
- `SMOOTHSPEC&SEARCH`: the source finder on a copy of the cube smoothed along the spectral axis with
  {ref}`WINDOW_TYPE <par-smoothspec-window-type>` and {ref}`WINDOW_SIZE <par-smoothspec-window-size>`
  (see {doc}`../smoothspec/algorithm`); the beam does not change.

The smoothed copies are used only to find the voxels: the mask is applied to the original cube.

**All the detections or the largest.** The mask holds either all the detections or only the largest
one, the detection with most voxels:

- MAKEMASK, the maps (`TOTALMAP`, `VELOCITYMAP`, `DISPERSIONMAP`, `MASSDENSMAP`, `RMSMAP`), 2DFIT and
  ELLPROF use all the detections;
- 3DFIT (every METHOD), GALMOD, SPACEPAR and REND3D use the largest.

The types ending in `LARGEST` (`SEARCHLARGEST`, `SMOOTH&SEARCHLARGEST`, `SMOOTHSPEC&SEARCHLARGEST`) keep
only the largest detection in every case. If nothing is detected, BBarolo stops with an error.

### `SMOOTH`

The cube is smoothed spatially to the beam $B'$ above (FACTOR, or NEWBEAM). On the smoothed cube
$F_s$, the statistics $m_s$, $\sigma_s$ are measured over the voxels that are not blank (0 or NaN),
without ITERNOISE, and the mask is

$$
M(x, y, z) = \bigl[\, F_s(x, y, z) > m_s + n_b\,\sigma_s \,\bigr] ,
$$

with $n_b$ = {ref}`BLANKCUT <par-makemask-blankcut>`. Every voxel above the threshold is kept, connected
to others or not.

### `NEGATIVE`

The noise is measured in each channel from its negative values only. For channel $z$, let
$N_z = \{F(x, y, z) < 0\}$; the sample $N_z \cup (-N_z)$, symmetric about zero, has $m_z = 0$ and a spread
$\sigma_z$ (MADFM/0.6744888 or standard deviation). The mask is

$$
M(x, y, z) = \bigl[\, F(x, y, z) > m_z + n_b\,\sigma_z \,\bigr] ,
$$

with $n_b$ = BLANKCUT. A channel without negative values is left out of the mask. The emission does not
bias $\sigma_z$, as long as it is positive. With `MASK NEGATIVE`, 3DFIT weights the residuals of each
channel by its noise $\sigma_z$ (see {doc}`../3dfit/algorithm`).

### `THRESHOLD`

$$
M(x, y, z) = \bigl[\, F(x, y, z) > T \,\bigr] ,
$$

with $T$ = {ref}`THRESHOLD <par-search-threshold>`, in the flux units of the cube (0, i.e. all the
positive voxels, if not given).

### `FILE(mask.fits)`

The mask is read from a FITS file with as many voxels as the cube (degenerate axes are dropped): the
voxels with a non-zero value are in the mask; NaN values are outside.

### `NONE`

All the voxels are in the mask.

## Enlarging the mask

`_ENLARGE` followed by an integer $N$ after the type (e.g. `SEARCH_ENLARGE5`; $N = 3$ if omitted)
widens the mask in each channel: a voxel is added if a voxel of the same channel within
$\max(|\Delta x|, |\Delta y|) \le N$ pixels is in the mask (a square of $2N + 1$ pixels). The channels
do not change.

## The output

The mask is written in `<prefix>_mask.fits` (16-bit integers, 1 in the mask, 0 outside), with the header
of the cube. The HISTORY cards record MASK and the input file, and the parameters of the type: FACTOR and
the new beam for the smoothed masks (BLANKCUT for `SMOOTH`), WINDOW_TYPE and WINDOW_SIZE for
`SMOOTHSPEC&SEARCH`, the parameters of the source finder for the masks of the source finder, THRESHOLD
for `THRESHOLD`. A mask built once is used by all the following tasks of the run.

## Differences from BBarolo 1.8

- The **position angle of the new beam** of the smoothed masks is that of the beam of the data when the BPA of NEWBEAM
  is not given (it was 0°).
- With a header without a beam, where BEAM gives the beam of the data (in 1.8, BMAJ and BMIN), the `SMOOTH`
  and `SMOOTH&SEARCH` masks smooth to FACTOR times that beam (in 1.8 they smoothed to the same beam, i.e. not
  at all). The beam of the smoothed masks is NEWBEAM (in 1.8, BMAJ, BMIN and BPA).
- The checks of SEARCHTYPE and of the growth threshold now apply also when the source finder only builds
  a mask; the HISTORY card GROWTHTHRESHOLD has the growth threshold (it had the primary threshold).
- **ITERNOISE**, used by the masks of the source finder, stops at convergence (see
  {doc}`../stats/algorithm`).
- A `FILE(...)` mask needs the same number of voxels as the cube, not the same axes; a `NEGATIVE`
  channel without negative values is left out of the mask.
- The mask is written as `<prefix>_mask.fits` (1.8: `mask.fits`).

The full list is in {doc}`../../reference/changes`.

## References

- Whiting M. T., 2012, MNRAS, 421, 3242 (Duchamp):
  [2012MNRAS.421.3242W](https://ui.adsabs.harvard.edu/abs/2012MNRAS.421.3242W)
- Di Teodoro E. M., Fraternali F., 2015, MNRAS, 451, 3021:
  [2015MNRAS.451.3021D](https://ui.adsabs.harvard.edu/abs/2015MNRAS.451.3021D)

## In the code

- `bbarolo/cube.py`: `Cube.BlankMask` (all the types, `_ENLARGE` and the output), `Cube._maskBeam` (the
  new beam of the smoothed masks), `Cube.search`, `Cube.setCubeStats`.
- `bbarolo/search.py`: `Search.LargestDetection`. `bbarolo/smooth.py`: `Smooth3D`, `SpectralSmooth3D`.
- `bbarolo/tasks.py`: `_compute_makemask`, `write_step` (writes the mask), `MakeMask`. `bbarolo/params.py`: the check of MASK in
  `checkPars`.
