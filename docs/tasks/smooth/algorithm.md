# SMOOTH: how it works

SMOOTH degrades the spatial resolution of a datacube: every channel map is convolved with the elliptical
Gaussian kernel that turns the beam of the data (the old beam) into a larger beam (the new beam). The
input is the cube and the two beams; the output is the smoothed cube, with the new beam in its header,
written as a FITS file. The values are rescaled so that a cube in Jy/beam stays in Jy/beam of the new
beam. The spectral axis is not touched. The cube in memory is not replaced: the tasks that run after
SMOOTH in the same run analyse the original cube.

The task runs in six steps: the two beams are set; the convolution kernel is computed from them; the
kernel is sampled on the pixel grid; the flux scale factor is computed; the channel maps are convolved
with Fast Fourier Transforms and the blanks restored; the cube is optionally averaged in blocks of
pixels (REDUCE) and written.

## The beams

A beam is an elliptical Gaussian given by its major and minor axes $a \ge b$ (FWHM, arcsec) and the
position angle $\theta$ of its major axis (degrees, anticlockwise from North). Subscripts $\rm o$, $\rm n$
and $\rm k$ denote the old beam, the new beam and the convolution kernel.

- **Old beam** $(a_{\rm o}, b_{\rm o}, \theta_{\rm o})$: the beam of the header (BMAJ, BMIN, BPA), each
  value replaced by {ref}`OBMAJ <par-smooth-obmaj>`, {ref}`OBMIN <par-smooth-obmin>` and
  {ref}`OBPA <par-smooth-obpa>` when given. A header without a beam needs OBMAJ and OBMIN, or
  {ref}`BEAMFWHM <par-all-beamfwhm>`; without them the task stops with an error.
- **New beam** $(a_{\rm n}, b_{\rm n}, \theta_{\rm n})$: {ref}`BMAJ <par-smooth-bmaj>`,
  {ref}`BMIN <par-smooth-bmin>` and {ref}`BPA <par-smooth-bpa>`. An axis not given is
  $f$ times the old one, with $f$ = {ref}`FACTOR <par-smooth-factor>` (default 2); BMIN not given with
  BMAJ given is equal to BMAJ; $\theta_{\rm n} = \theta_{\rm o}$ when BPA is not given.

If $a < b$ for either beam, the two axes are swapped and the position angle is increased by 90°: the
angle given is that of the axis given as BMAJ. Then the new beam must be at least as large as the old
one along both axes, $a_{\rm n} \ge a_{\rm o}$ and $b_{\rm n} \ge b_{\rm o}$, otherwise the task stops
with an error.

## The convolution kernel

The convolution of two Gaussians is a Gaussian whose second-moment matrix is the sum of theirs. The
kernel $K$ that satisfies $G_{\rm o} * K = G_{\rm n}$ is then the Gaussian with the difference of the
moment matrices of the new and the old beam, if that difference is positive semi-definite. Its axes and
angle follow from the deconvolution formulae of Wild (1970). With

$$
D_{\rm n} = a_{\rm n}^2 - b_{\rm n}^2, \qquad D_{\rm o} = a_{\rm o}^2 - b_{\rm o}^2, \qquad
D_{\rm k} = \sqrt{D_{\rm n}^2 + D_{\rm o}^2 - 2\,D_{\rm n} D_{\rm o} \cos 2(\theta_{\rm n} - \theta_{\rm o})},
$$

the kernel has the axes (FWHM, arcsec) and position angle

$$
a_{\rm k}^2 = \tfrac12 \left( a_{\rm n}^2 + b_{\rm n}^2 - a_{\rm o}^2 - b_{\rm o}^2 + D_{\rm k} \right),
\qquad
b_{\rm k}^2 = \tfrac12 \left( a_{\rm n}^2 + b_{\rm n}^2 - a_{\rm o}^2 - b_{\rm o}^2 - D_{\rm k} \right),
$$

$$
\theta_{\rm k} = \tfrac12 \,\mathrm{atan2}\!\left(
D_{\rm n} \sin 2\theta_{\rm n} - D_{\rm o} \sin 2\theta_{\rm o},\;
D_{\rm n} \cos 2\theta_{\rm n} - D_{\rm o} \cos 2\theta_{\rm o} \right),
$$

with $\theta_{\rm k} = 0$ when both arguments of atan2 are zero. If $b_{\rm k}^2 < 0$ (beyond a relative
tolerance of $10^{-9}$, below which it is set to 0) the new beam cannot be obtained from the old one by a
convolution, for example a new beam larger along both axes but rotated with respect to an elongated old
beam, and the task stops with an error. When the two beams are equal, $a_{\rm k} = b_{\rm k} = 0$ and
the kernel is a single pixel.

## The kernel on the pixel grid

The kernel is sampled at the centres of the pixels, on a grid of $(2X+1) \times (2Y+1)$ pixels centred on
the central pixel:

$$
K_{ij} \propto \exp\!\left[ -4 \ln 2 \left(
\frac{x'^2}{a_{\rm k}^2} + \frac{y'^2}{b_{\rm k}^2} \right) \right],
\qquad
\begin{aligned}
x' &= i\,\Delta x \cos\varphi + j\,\Delta y \sin\varphi,\\
y' &= -i\,\Delta x \sin\varphi + j\,\Delta y \cos\varphi,
\end{aligned}
$$

where $i \in [-X, X]$ and $j \in [-Y, Y]$ are the pixel offsets along the x and y axes of the image,
$\Delta x = |{\rm CDELT1}|$ and $\Delta y = |{\rm CDELT2}|$ are the pixel sizes (arcsec), and
$\varphi = \theta_{\rm k} + 90^\circ - {\rm CROTA}$ is the angle of the major axis from the x axis of
the image (CROTA, degrees, the rotation of the sky axes in the header, 0 if absent). The kernel is
truncated where it falls below $10^{-4}$ of its peak: the half-extent of the ellipse at that level is
$r = \tfrac12 a \sqrt{\ln 10^4 / \ln 2} \approx 1.82\,a$ along each axis, and

$$
X = \mathrm{round}\!\left( \frac{\max(|r_a \cos\varphi|,\, |r_b \sin\varphi|)}{\Delta x} \right), \qquad
Y = \mathrm{round}\!\left( \frac{\max(|r_a \sin\varphi|,\, |r_b \cos\varphi|)}{\Delta y} \right),
$$

with $r_a$, $r_b$ from $a_{\rm k}$, $b_{\rm k}$. Values below the cut are set to 0, and the sampled
kernel is normalised to a unit sum, $\sum_{ij} K_{ij} = 1$. A kernel of more than $512 \times 512$
elements is an error.

```{note}
When $b_{\rm k} = 0$, e.g. when an elliptical beam is made circular with the size of its major axis, the
term $y'^2/b_{\rm k}^2$ is left out, as in BBarolo 1.8: the kernel is constant across its minor axis
within the box. Along an axis of the grid ($\varphi$ = 0° or 90°) the box is one pixel wide and the
kernel is a line, as it should be; at another angle it is a band as wide as the box, and the resulting
beam is wider than the new beam of the header (smoothing 30″ × 20″ at PA 45° to 30″ × 30″ gives a beam
of about 62″ × 28″).
```

## The flux scale factor

A unit-sum kernel conserves the integral of each channel map, i.e. the flux in units per pixel. For data
per beam the values must be multiplied by the ratio of the beam areas. SMOOTH computes this factor on the
same grid: it samples the old beam with unit peak, $G_{\rm o}$, with the same truncation and the angle
$\theta_{\rm o} + 90^\circ - {\rm CROTA}$, and takes

$$
s = \frac{1}{\sum_{ij} G_{{\rm o},ij}\, K_{-i,-j}},
$$

the inverse of the peak of $G_{\rm o} * K$. For well-sampled beams
$s \simeq a_{\rm n} b_{\rm n} / (a_{\rm o} b_{\rm o})$, the ratio of the beam areas.

The factor $s$ is used for data per beam: units Jy/beam, mJy/beam, uJy/beam, W.U., or units BBarolo does
not recognise. For data in K or per pixel (Jy, mJy) $s = 1$. {ref}`SCALEFACTOR <par-smooth-scalefactor>`,
when given, replaces $s$ for any unit.

## The convolution

Each channel $z$ of the cube, or of the sub-cube selected by {ref}`BOX <par-smooth-box>`, is convolved
with the kernel and multiplied by $s$:

$$
S'(x, y, z) = s \sum_{i=-X}^{X} \sum_{j=-Y}^{Y} K_{ij}\, S(x - i,\, y - j,\, z),
$$

where $S$ is the input intensity (units of the cube) and $S'$ the smoothed one, in the same units. The
pixels outside the cube (or the box) count as 0, so the maps are darkened near their edges, within the
half-size of the kernel. Blank pixels (NaN; BBarolo reads NaN as 0) enter the sum as 0.

The convolution is computed for all channels at once with real Fast Fourier Transforms. Each map and the
kernel are zero-padded to at least $n_y + Y$ by $n_x + X$ pixels ($n_x$, $n_y$: the map size), which makes
the circular convolution equal to the linear one over the map: the result is the exact sum above, to
floating-point precision. Empty channels are skipped.

After the convolution, the pixels that are blank in the input (0 or NaN) are blanked again: they are set
to NaN, or to 0 with {ref}`FFT <par-smooth-fft>` false. Emission does not spread into blanked regions,
and an input pixel that is exactly 0 is treated as blank.

## REDUCE and the output

With {ref}`REDUCE <par-smooth-reduce>` on and $N = \lfloor f \rfloor > 1$ ($f$ = FACTOR, also when the new
beam is given by BMAJ and BMIN), the smoothed cube is averaged over blocks of $N \times N$ pixels:

$$
S''(X', Y', z) = \frac{1}{N^2} \sum_{p=0}^{N-1} \sum_{q=0}^{N-1} S'(N X' + p,\, N Y' + q,\, z),
$$

with $X'$, $Y'$ the pixels of the reduced grid. A block whose sum is exactly 0, or contains a NaN, is
NaN. The pixels at the upper edges that do not fill a block are dropped. The header has the pixel sizes
multiplied by $N$ and the reference pixels ${\rm CRPIX}' = \tfrac12 + ({\rm CRPIX} - \tfrac12)/N$ for
the two spatial axes. The values remain per beam, the beam in arcsec does not change, and its area in
pixels is divided by $N^2$. Averaging after smoothing to $N$ times the beam keeps about the same number of
pixels per beam as in the input.

The cube is written in {ref}`OUTFOLDER <par-all-outfolder>` as `<prefix>_s<M>.fits`, $M$ being $a_{\rm n}$
rounded to an integer arcsec (`<prefix>_s<M>red.fits` with REDUCE), or in
{ref}`SMOOTHOUTPUT <par-smooth-smoothoutput>`. Its header has BMAJ, BMIN and BPA of the new beam, the size
of the box with CRPIX shifted by the lower corner of the box, and two HISTORY cards with the new and the
old beam.

The same steps smooth the cube for the masks `SMOOTH` and `SMOOTH&SEARCH`
(see {ref}`MASK <par-makemask-mask>`) and smooth the projections of
{doc}`REND3D <../rend3d/algorithm>`.

## Differences from BBarolo 1.8

- The convolution is always computed with FFTs. FFT false no longer selects the direct convolution: it
  only writes the blanks as 0 instead of NaN. (The direct convolution of the C++ code spread the NaN
  blanks over the size of the kernel.)
- The position angle of the new beam is that of the old beam when BPA is not given (it was 0°, which
  made the kernel of an elliptical beam skewed or impossible). Any value of BPA and OBPA is an angle:
  BPA −1 and OBPA 0 no longer mean "the old angle" and "the angle of the header".
- Cubes in mJy/beam, uJy/beam and W.U. are rescaled by $s$, as cubes in Jy/beam (C++ used $s = 1$).
- With a header without a beam, the beam of the data comes from OBMAJ, OBMIN, OBPA or BEAMFWHM, and it is
  an error without them (C++ took BMAJ and BMIN, the new beam, so nothing was smoothed).
- A new beam smaller than the old one is an error.
- A beam given with BMAJ < BMIN has its major axis at BPA + 90° (C++ kept the angle, so the beam was
  rotated by 90°), and the axes of the new beam are put in order before it is compared with the old one
  (C++ compared first, so BMAJ 15 BMIN 25 against an old beam of 20″ × 10″ was an error).
- LINEAR (a resolution in kpc) no longer sets the new beam.
- REDUCE: the reference pixel of the reduced cube is exact, $\tfrac12 + ({\rm CRPIX} - \tfrac12)/N$; C++
  rounded it to an integer.

## References

- Di Teodoro E. M., Fraternali F., 2015, MNRAS, 451, 3021:
  [2015MNRAS.451.3021D](https://ui.adsabs.harvard.edu/abs/2015MNRAS.451.3021D)
- Wild J. P., 1970, Australian Journal of Physics, 23, 113 (deconvolution of elliptical Gaussian beams):
  [1970AuJPh..23..113W](https://ui.adsabs.harvard.edu/abs/1970AuJPh..23..113W)

## In the code

- `bbarolo/smooth.py`: `convpars` (the kernel beam), `fillgauss2d` (the sampled Gaussian),
  `gaussian_scalefactor` (the factor $s$), `perBeam` (units per beam), `FFTConvolver` (the FFT
  convolution), `Smooth3D` (`cubesmooth`, `smooth`, `fitswrite`).
- `bbarolo/cube.py`: `Cube.Reduce` (REDUCE).
- `bbarolo/tasks.py`: `_compute_smooth`, `_write_smooth`.
