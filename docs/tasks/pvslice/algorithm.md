# PVSLICE: how it works

PVSLICE cuts a datacube along a straight line on the sky and returns a position–velocity (PV) diagram:
an image whose first axis is the offset along the line and whose second axis is the spectral axis of
the cube. Each column of the image is the spectrum of the cube at one point of the line, interpolated
between the pixels and, optionally, averaged across a slit. The input is the cube and the definition
of the line; the output is a FITS image, `<prefix>_pv_<PA>.fits`, in the flux unit of the cube. The
same code makes the PV diagrams of {doc}`3DFIT <../3dfit/algorithm>`, {doc}`GALMOD <../galmod/algorithm>`
and GALWIND.

The task is run by {ref}`PVSLICE <par-pvslice-pvslice>`. It has three steps: the points of the line,
the sampling of the cube at those points, and the header of the image.

## The line

Positions are in pixels of the cube, counted from 0: the centre of the first pixel is $(0, 0)$, and the
field is the rectangle $0 \le x \le N_x - 1$, $0 \le y \le N_y - 1$. The line is defined in one of two
ways. When all four coordinates of {ref}`P1_PV <par-pvslice-p1-pv>` and
{ref}`P2_PV <par-pvslice-p2-pv>` are 0 or more, the two points are used; otherwise the centre and the
angle.

### Through a centre, at an angle

The centre $(x_0, y_0)$ is {ref}`XPOS_PV <par-pvslice-xpos-pv>`, {ref}`YPOS_PV <par-pvslice-ypos-pv>`
(pixels, or world coordinates converted to pixels with the WCS of the cube), and must lie inside the
field. The angle $\phi$ is {ref}`PA_PV <par-pvslice-pa-pv>` (degrees), measured anticlockwise from the
$+y$ axis of the image, i.e. from North through East for an image with RA decreasing along $x$. The
direction of the line is

$$
\hat{u} = (-\sin\phi,\; \cos\phi),
$$

and its points are

$$
(x_j, y_j) = (x_0, y_0) + s_j\, \hat{u}, \qquad s_j = j \;\text{(pixels)}, \quad j \in \mathbb{Z},
$$

for all the integers $j$ that keep the point inside the field. The line therefore crosses the whole
field, with points exactly one pixel apart and one point at the centre ($s = 0$). The points are then
ordered so that the offset increases with $x$ (with $y$ for a vertical line, $\phi = 0$ or $180°$):
when $\hat{u}$ points towards decreasing $x$, the order is reversed and $s \to -s$. The slices at
$\phi$ and $\phi + 180°$ are thus the same image.

### Between two points

For $P_1 = (x_1, y_1)$ and $P_2 = (x_2, y_2)$ (pixels), with length $L = |P_2 - P_1| > 0$, the line has
the direction $\hat{u} = (P_2 - P_1)/L$ and the centre at the midpoint $P_{\rm c} = (P_1 + P_2)/2$.
Its points are

$$
(x_j, y_j) = P_{\rm c} + s_j\, \hat{u}, \qquad s_j = j, \qquad -\tfrac{L}{2} \le s_j \le \tfrac{L}{2},
$$

cut at the edges of the field. The offset is 0 at the midpoint and increases towards $P_2$. The end
points are included only when $L/2$ is an integer number of pixels.

A slice with fewer than two points inside the field is an error.

## Sampling the cube

Each point is in general between pixels. The value of channel $k$ at $(x, y)$ is taken as set by
{ref}`ANTIALIAS <par-pvslice-antialias>` $A$:

- $A = 0$: the nearest pixel, $I_k(\mathrm{round}(x), \mathrm{round}(y))$.
- $A \ne 0$ (default $0.5$): bilinear interpolation between the four nearest pixels. With
  $i = \lfloor x \rfloor$, $l = \lfloor y \rfloor$, $t_x = x - i$ and $t_y = y - l$,

$$
I_k(x, y) = \frac{\sum_{(a, b)} w_{ab}\, I_k(i + a,\, l + b)}{\sum_{(a, b)} w_{ab}},
\qquad
w_{ab} = \left[a\, t_x + (1 - a)(1 - t_x)\right] \left[b\, t_y + (1 - b)(1 - t_y)\right],
$$

with $a, b \in \{0, 1\}$. The sums run over the neighbours that are inside the cube and not blank, so
the weights are renormalised near the edges; a point whose valid neighbours carry less than half of
the total weight is blank. For points inside the field and data without blanks the denominator is 1
and this is the usual bilinear interpolation.

The same positions are used for all the channels: the spectral axis is not resampled.

## The slit

A slice can be averaged across a slit of full width $W$ (pixels), perpendicular to the line. $W$ is
$w/p$, with $w$ the width {ref}`WIDTH_PV <par-pvslice-width-pv>` (arcsec, 0 by default: a single line)
and $p$ the pixel size of the cube (the mean of the two axes, arcsec). With ANTIALIAS an integer
$A \ge 1$, the slit is at least $2A + 1$ pixels wide:

$$
W = \max\left(\frac{w}{p},\; 2A + 1\right).
$$

For $W > 0$ the cube is sampled at $n = \lceil 2W \rceil + 1$ offsets across the slit,

$$
d_q = -\frac{W}{2} + q\,\frac{W}{n - 1}, \qquad q = 0, \dots, n - 1,
$$

at most half a pixel apart, along the normal $\hat{n} = (\hat u_y, -\hat u_x)$ of the line. The value
of the slice at point $j$ is the mean of the valid samples,

$$
P_k(s_j) = \frac{1}{n_{\rm v}} \sum_{q\,\in\,\text{valid}} I_k\!\left(x_j + d_q \hat n_x,\; y_j + d_q \hat n_y\right),
$$

with $n_{\rm v}$ the number of samples that are not blank (samples outside the cube are blank). A
point with no valid sample is blank. All the samples have the same weight: the slit is a top hat.

## The output image

The image has $N_s$ columns, the points of the line, and $N_z$ rows, the channels of the cube. Its
values are in the flux unit of the cube (an average, not a sum, of the data). The header has:

- **Axis 1**, `Offset`: CRVAL1 = 0 at the centre (or midpoint); CDELT1 = $|\Delta x|$, the size of a
  pixel along $x$ of the cube; CUNIT1 that of the spatial axes of the cube (degrees). The offset of
  point $j$ is $s_j |\Delta x|$. For pixels that are not square, the offset along a slanted line is in
  units of the $x$ pixel size.
- **Axis 2**: the spectral axis of the cube, copied (CTYPE3, CRPIX3, CRVAL3, CDELT3, CUNIT3), with the
  rest frequency and the beam of the cube.
- A HISTORY card with the slice: the world coordinates of the centre and the angle, or the two points.

The file name contains PA_PV rounded to an integer (`<prefix>_pv_<PA>.fits`), also for a slice between
two points, where PA_PV is not used (0 by default).

## Differences from BBarolo 1.8

- **Sampling.** BBarolo 1.8 sampled the line from one end of the field, at steps of about one pixel
  (the number of points was the rounded length of the line), and took a weighted mean of the
  $(2A+1)^2$ pixels nearest to each point, with weights $1 - d/d_{\max}$ ($d$ the distance from the point,
  $d_{\max} = (A+1)\sqrt{2}$). The points are now exactly one pixel apart, with one at the centre, and
  are interpolated bilinearly. ANTIALIAS 0 selects the nearest pixel; values $\ge 1$ widen the slit
  instead of the neighbourhood.
- **WIDTH_PV** is the full width of the slit, sampled every half pixel or less. In 1.8 it was converted
  to a half-width rounded to whole pixels.
- **Offset axis.** CDELT1 is the pixel size of the cube; 1.8 derived it from the world coordinates of
  the end points. The offset is 0 at the centre in both.
- **Output name**: `<prefix>_pv_<PA>.fits` (1.8: `<prefix>pv_<PA>.fits`).

Flux along the slices of NGC 2403 agrees with 1.8 within 0.4%. The full list is in
{doc}`../../reference/changes`.

## References

- Di Teodoro E. M., Fraternali F., 2015, MNRAS, 451, 3021:
  [2015MNRAS.451.3021D](https://ui.adsabs.harvard.edu/abs/2015MNRAS.451.3021D)

## In the code

- `bbarolo/pvslice.py`: `pv_path` (the points of the line: straight, between two points, or following
  the rings for 3DFIT), `_line_range` (the part of the line inside the field), `pv_sample` (nearest,
  bilinear or spline sampling, the slit), `pv_extract` (both), `PvSlice` (the task: parameters,
  `slice`, `define_header`) and `PositionVelocity`.
- `bbarolo/tasks.py`: `_compute_pvslice`, `_write_pvslice` (the task) and the `PVSlice` class.
