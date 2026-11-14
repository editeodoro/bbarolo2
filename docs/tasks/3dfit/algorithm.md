# 3DFIT: how it works

3DFIT with METHOD CLASSIC (the default; see {doc}`index` for GLOBAL and BAYESIAN, and {doc}`likelihood` for their
algorithm) fits a tilted-ring model of a rotating disk directly to an emission-line datacube (Di Teodoro &
Fraternali 2015). The disk is a set of concentric rings. For each ring, the GALMOD engine builds a model
cube of that ring alone, convolves it with the beam and compares it, voxel by voxel, with the data in
the mask. The free parameters of the ring (by default the rotation velocity, the velocity dispersion, the
inclination and the position angle) are those that minimise a residual function, found with a downhill
simplex. The rings are fitted independently of each other. By default a second stage then makes the
geometry a smooth function of radius and fits the kinematics again. Because the comparison is made in
three dimensions, the model includes the beam smearing, the channel width, the instrumental broadening
and the thickness of the disk.

The inputs are the cube, the ring parameters (given, read from a ring file or estimated from the data)
and the mask. The outputs are the tables of the fitted rings, a model cube with its moment maps and
position–velocity slices, the radial profile of the data and, optionally, the errors of the parameters
and the asymmetric drift correction.

The task runs in this order:

1. the mask is built ({ref}`MASK <par-3dfit-mask>`);
2. the initial rings are set: values given, ring file, estimates from the data, defaults;
3. first stage: every ring is fitted, and the results are written in `<prefix>_rings_final1.txt`;
4. second stage ({ref}`TWOSTAGE <par-3dfit-twostage>`): the geometry is regularised and fixed, and the
   rings are fitted again (`<prefix>_rings_final2.txt`);
5. the final model of all the rings is built, normalised and written, with the other outputs.

## The tilted-ring model

Each ring $k$ has a centre $(x_0, y_0)$ (pixels), a systemic velocity $V_\mathrm{sys}$, a rotation velocity
$V_\mathrm{rot}$, a radial velocity $V_\mathrm{rad}$, a vertical velocity $V_\mathrm{vert}$ and a velocity
dispersion $\sigma$ (all in km/s), a face-on surface density $\Sigma$ ({ref}`DENS <par-3dfit-dens>`), a
scale height $z_0$ (arcsec), an inclination $i$ and a position angle $\phi$ (degrees; $\phi$ is the angle
of the receding half of the major axis, from North towards East). The free parameters are chosen with
{ref}`FREE <par-3dfit-free>`; the others stay at their initial values. {ref}`VVERT <par-3dfit-vvert>`,
{ref}`DVDZ <par-3dfit-dvdz>` and {ref}`ZCYL <par-3dfit-zcyl>` are used by the models but never fitted.

A pixel $(x, y)$ of the sky (in pixels) has, in the plane of a ring with this geometry, the coordinates

$$
x_r = -(x - x_0)\sin\phi + (y - y_0)\cos\phi, \qquad
y_r = \frac{-(x - x_0)\cos\phi - (y - y_0)\sin\phi}{\cos i},
$$

the radius $r = \sqrt{x_r^2 + y_r^2}$ (pixels) and the azimuth $\theta = \operatorname{atan2}(y_r, x_r)$
($\theta = 0$ on the receding half of the major axis, $|\theta| > 90^\circ$ on the approaching side; $\theta =
0$ for $r < 0.1$ pixels). Gas at azimuth $\theta$ in the plane of the ring has the line-of-sight velocity

$$
V_\mathrm{los} = V_\mathrm{sys} + \left(V_\mathrm{rot}\cos\theta + V_\mathrm{rad}\sin\theta\right)\sin i
+ V_\mathrm{vert}\cos i ,
$$

and a Gaussian line with dispersion $\sqrt{\sigma^2 + \sigma_\mathrm{instr}^2}$, where $\sigma_\mathrm{instr}$
is {ref}`SPECRES <par-3dfit-specres>` times the channel width. The gas is spread vertically with the
profile {ref}`LTYPE <par-3dfit-ltype>` and the scale height $z_0$. How the models are built from these
quantities (clouds in sub-rings, Monte Carlo or analytic line profiles, flux units) is described in
{doc}`../galmod/algorithm`.

### Rings in the fit and in the models

The rings are given by their radii $R_1 < R_2 < \dots < R_N$ (arcsec): {ref}`RADII <par-3dfit-radii>`, or
$R_k = (k - \tfrac12)\,\Delta R$ from {ref}`NRADII <par-3dfit-nradii>` and
{ref}`RADSEP <par-3dfit-radsep>` $= \Delta R$. In the fit, ring $k$ is the annulus

$$
R_k - w_k^- \le R \le R_k + w_k^+, \qquad
w_k^- = \frac{R_k - R_{k-1}}{2}, \quad w_k^+ = \frac{R_{k+1} - R_k}{2},
$$

with $w_1^- = w_1^+ = (R_2 - R_1)/2$, $w_N^- = w_N^+ = (R_N - R_{N-1})/2$ and, for a single ring,
$w^\pm = \Delta R/2$ (or $R_1/2$ without RADSEP); the inner edge is at least 0. During the fit all the
parameters are constant across the annulus. The tables report $R_k$ and the fitted values: each value
belongs to the annulus centred on $R_k$.

The final model of 3DFIT (and the models of GALMOD) interpolate the values of the rings linearly in radius
between the $R_k$, with the ring convention {ref}`RINGMODE <par-3dfit-ringmode>`. With the default,
`CENTRED`, the disk extends from $R_1 - w_1^-$ to $R_N + w_N^+$, and the values of the first and last ring
are kept constant in their outer halves, so that the model matches the annuli of the fit. The final model
is a smooth version of the fitted annuli, not the sum of the models of single rings used in the fit.
RINGMODE does not change the fit.

## Initial parameters

Every ring parameter needs an initial value in every ring: the starting point of the fit for the free
parameters, the fixed value for the others. The values come, in order of precedence, from:

1. the values given: one value for all the rings, one value per ring, or a column of a file (see
   {ref}`VROT <par-3dfit-vrot>`);
2. {ref}`RINGFILE <par-3dfit-ringfile>`: the columns of a BBarolo ring file, found by their names in the
   header, for the parameters not given (and the radii, unless RADII or both NRADII and RADSEP are given);
3. estimates from the data, for the centre, $V_\mathrm{sys}$, $V_\mathrm{rot}$, $i$, $\phi$, NRADII and
   RADSEP (method {ref}`GUESSTYPE <par-3dfit-guesstype>`, below);
4. defaults: $\sigma = 8$ km/s (estimated with GUESSTYPE `FIT`), $z_0$ one pixel (in arcsec), $V_\mathrm{rad}
   = 0$, $\Sigma = 1$, $V_\mathrm{vert} = 0$, DVDZ $= 0$, ZCYL $= 0$.

An estimated value is the same in all the rings. The initial rings are written in
`<prefix>_rings_initial.txt`, before the beam-wide rings of REVERSE and the refinement of $\phi$ (below); the
screen summary marks each value as given (i), default (d) or estimated (e); NRADII and RADSEP are given
when RADII gives the rings. METHOD GLOBAL and BAYESIAN build the initial rings with the same code, and prints the same
summary. Then, with $\phi$ free, a 2D fit
may refine $\phi$ and $V_\mathrm{rot}$ ring by ring (see below).

### Estimates from the data

The estimates use the largest source found by the source finder in the cube, with the parameters of the
SEARCH task (e.g. {ref}`SNRCUT <par-search-snrcut>`), or the voxels of the mask with `MASK FILE(...)`.
From the voxels of the source come the global profile (the flux summed in each channel), the total
intensity map and the velocity field $V(x, y)$, the intensity-weighted mean velocity of each pixel.

**LEGACY** (the default; the algorithm of BBarolo 1.8):

- $V_\mathrm{sys}$: from the velocities where the global profile crosses 20% and 50% of its peak,
  $v_{20}^\pm$ and $v_{50}^\pm$, with the widths $W_{20} = |v_{20}^+ - v_{20}^-|$ and
  $W_{50} = |v_{50}^+ - v_{50}^-|$:

  $$
  V_\mathrm{sys} =
  \begin{cases}
  \tfrac12\left(v_{20}^- + v_{20}^+\right) & \text{if } W_{20} > 1.5\,W_{50},\\[2pt]
  \tfrac14\left(v_{20}^- + v_{20}^+ + v_{50}^- + v_{50}^+\right) & \text{otherwise.}
  \end{cases}
  $$

- Centre: the mean of the flux-weighted centroid of the source and of the unweighted mean position of its
  voxels.
- $\phi$: among the lines through the centre, at angles in steps of 0.5°, the one along which the median of
  $|V - V_\mathrm{sys}|$ in the velocity field is largest; the side where $V > V_\mathrm{sys}$ sets the
  receding half.
- $i$ and the outer radius $R_\mathrm{max}$: first from the extent of the velocity field along the major
  and minor axes, $a$ and $b$, with $\cos i = b/a$ and $R_\mathrm{max} = a$. Then an ellipse with semi-major
  axis $R_\mathrm{max}$ and inclination $i$ (centre and $\phi$ fixed) is fitted to the footprint of the
  velocity field with a downhill simplex, which minimises the number of pixels of the ellipse without a
  velocity minus the number of pixels with one. Finally the axes are corrected for the beam, of FWHM $B$
  (arcsec), by subtracting it:

  $$
  \cos i = \frac{R_\mathrm{max}\cos i - B}{R_\mathrm{max} - B},
  $$

  applied when $R_\mathrm{max}\cos i - B > B$ and, above 75°, also $R_\mathrm{max} - B > 9B$. The
  inclination becomes 80° if $2R_\mathrm{max}\cos i < B$, and 25° if it is below 10°.
- Rings: $\Delta R = B$, or $B/2$ if that gives fewer than 20 rings within $R_\mathrm{max}$; $N =
  \operatorname{round}(R_\mathrm{max}/\Delta R)$, minus one if $N > 4$. With RADSEP given alone, $\Delta R$
  is RADSEP and $N$ follows from the same rule, so the rings cover the estimated extent of the disk.
- $V_\mathrm{rot} = W_{50}/2$, divided by $\sin i$ only if $i > 40^\circ$.
- With more than 3 rings, the estimates are refined with a 2D tilted-ring fit of the velocity field
  (Begeman 1989) on rings one pixel wide, with $V_\mathrm{sys}$, $V_\mathrm{rot}$, $\phi$ and the centre free,
  weights $\cos^2\theta$ and the pixels within 15° of the minor axis left out. The medians of the fitted
  values over the rings replace the centre, $V_\mathrm{sys}$ and $\phi$; then $i$ (the ellipse) and
  $V_\mathrm{rot}$ are estimated again. In LEGACY these intermediate steps use the estimates also of the
  parameters that are given (e.g. $V_\mathrm{rot}$ is deprojected with the estimated inclination); only the
  parameters not given take the estimated values. The 2D fit is written in `<prefix>_2drings.txt`.

**ELLIPSE** is the same algorithm, corrected: the beam is removed in quadrature from the semi-axes $a =
R_\mathrm{max}$ and $b = R_\mathrm{max}\cos i$,

$$
\cos i = \sqrt{\frac{\max\left(b^2 - (kB)^2,\, 0\right)}{a^2 - (kB)^2}}, \qquad k = 0.75,
$$

with $i$ kept within 5° and 89° (no jumps); $V_\mathrm{rot} = W_{50}/(2\sin\max(i, 10^\circ))$ always; and the
values given are kept in the intermediate steps.

**FIT** starts from the ELLIPSE estimates and refines them in three steps:

1. A beam-convolved disk (ANALYTIC model, rings half a beam wide out to $R_\mathrm{max}$, $z_0$ given or one
   pixel) is fitted to the map of the cube summed over the channels of the source (not masked), within 1.5
   beams of the footprint of the source. The centre (within 2 beams), $i$ (5° to 89°) and $\phi$ (±60°) are
   found by non-linear least squares; for each geometry, the densities of the rings are the solution of a
   regularised linear least-squares problem (the deconvolved radial profile). An inclination below 15° is
   replaced by the ELLIPSE value, since a nearly round map hardly constrains it.
2. A 2D tilted-ring fit of the velocity field on rings one beam wide ($V_\mathrm{rot}$, $i$, $\phi$ and
   $V_\mathrm{sys}$ free) gives $\phi$ and $V_\mathrm{sys}$, the medians over the rings; $V_\mathrm{rot} =
   W_{50}/(2\sin i)$.
3. The velocity dispersion comes from the second moment $m_2(x, y)$ of the source, in the pixels beyond
   $R_\mathrm{max}/2$ (where the beam smearing is smaller), corrected for the channel width $\Delta v$ (km/s):

   $$
   \sigma = \sqrt{\max\left(\operatorname{median}(m_2)^2 - \frac{\Delta v^2}{12},\; 4\right)} \ \text{km/s}.
   $$

### Refinement of the position angle

When $\phi$ is free and there are at least 5 rings, a 2D tilted-ring fit of a velocity field, on the rings
of 3DFIT, refines the initial $\phi$ and $V_\mathrm{rot}$ of each ring. The velocity field is that of the
source when the parameters were estimated, otherwise the first moment of the cube within the mask. Only
$V_\mathrm{rot}$ and $\phi$ are free, with weights $\cos^2\theta$ and the pixels within 15° of the minor axis
left out. The fitted $\phi$ and $V_\mathrm{rot}$ replace the initial values of rings 2 to $N$ where they are
finite (and $V_\mathrm{rot} > 0$); ring 1 takes the $\phi$ of ring 2. $V_\mathrm{rot}$ is replaced only if it
is free: a fixed rotation velocity is kept. The fit is written in
`<prefix>_2drings.txt`; `<prefix>_rings_initial.txt` has the values before this step.

### Reverse-cumulative rings

With {ref}`REVERSE <par-3dfit-reverse>` `true`, or `auto` and a median initial inclination above 75°, the
rings are fitted from the outside inwards (see below). If the rings are narrower than the beam ($\Delta R <
B$), they are replaced by $\lceil R_N/B \rceil$ rings of width $B$ at the radii $(k - \tfrac12)B$, with the
initial values of the given rings interpolated linearly at the new radii (the values of the first or last ring
beyond them; $\phi$ is unwrapped first).

## The mask

The mask $M_{xyz}$ (1 for emission, 0 elsewhere) is built before the fit with
{ref}`MASK <par-3dfit-mask>` (by default the largest source of the source finder), unless an earlier task
of the run built it. It enters the fit in five places:

- the residuals: the voxels in the mask are always compared with the model; outside it, only the voxels
  where the model has emission, which count as *blanks* (see below);
- the normalisation of the models (`LOCAL`, `AZIM`): the flux of the data is summed within the mask;
- the noise of each channel: with `MASK NEGATIVE`, $\sigma_z$ is the noise of channel $z$ measured from its
  negative values; with the other masks $\sigma_z = 1$;
- the rings with too little emission: a pixel is in the 2D mask if any voxel of its spectrum is; a ring
  with fewer than 20 such pixels, and either none or fewer than 70% of its pixels, is skipped (not in
  REVERSE mode);
- the final model: the `LOCAL` normalisation, the radial profile of the `AZIM` normalisation and the
  maps of the data use the data within the mask. The maps are moments, or Gaussian fits with
  {ref}`MAPTYPE <par-maps-maptype>` `GAUSSIAN`, for the data and the model alike.

The noise of the cube $\sigma_D$, used in the residuals where the data are not positive, is the spread of
all the voxels of the cube: MADFM/0.6745 or the standard deviation (see
{ref}`FLAGROBUSTSTATS <par-all-flagrobuststats>`).

## The model of a ring

At each evaluation of the residuals, the model of ring $k$ alone is built with the trial values: an annulus
from $R_k - w_k^-$ to $R_k + w_k^+$ with constant parameters, filled with sub-rings 0.75 pixels wide
({doc}`../galmod/algorithm`, with {ref}`GALMODTYPE <par-3dfit-galmodtype>` `MC` or `ANALYTIC`). Monte Carlo
models use the same seed ({ref}`ISEED <par-3dfit-iseed>`) at every evaluation, so that the residual is a
deterministic function of the parameters. The clouds that fall outside the box (below) still draw their
random velocity deviates, so the Monte Carlo model in the box is the crop of the model of the whole cube with
the same seed. With {ref}`EMPTY <par-3dfit-empty>` `false`, the model of a ring
would be a disk filled from the centre: keep it `true` for the fits.

**The neighbouring rings.** The model holds only ring $k$. The emission of the neighbouring rings that the
beam spreads into the pixels of ring $k$ is in the data but not in the model; the normalisation (below)
absorbs part of it. In REVERSE mode, the model of ring $k$ is added, before the convolution, to the model of
the rings already fitted outside it ($k+1$ to $N$, with their fitted values interpolated between their
radii).

**The box.** The model is built in a box of the cube around the centre of the ring, with half-size

$$
e = R_k + w_k^+ + \frac{w_k^- + w_k^+}{2} + 3 z_0 \quad \text{(arcsec)},
$$

rounded up to whole pixels on each axis and cut at the edges of the cube; in REVERSE mode, the box of all
the rings ($e = R_N + \Delta R/2 + 3z_{0,N}$). All the channels of the cube are modelled.

**The convolution.** Each channel of the model is convolved with the beam, a 2D Gaussian with the FWHM
BMAJ × BMIN at the angle BPA (from the header, or from the keyword {ref}`BEAM <par-all-beam>`), sampled on the
pixels down to $10^{-4}$ of its peak and normalised to a unit sum, so that the flux is conserved. Values
below $10^{-12}$ are then set to zero, which decides where the model "has emission". With
{ref}`SM <par-3dfit-sm>` `false` the models are not convolved. In the fits, $\Sigma$ is always a flux per
unit area (flux units of the cube times km/s per arcsec²).

**The normalisation.** {ref}`NORM <par-3dfit-norm>` takes up to three values: for the first stage, the second
stage and the final model (one value is used for all; with two, the second is also that of the final model).
`AZIMUTHAL` is the same as `AZIM`. `BOTH` is accepted only as the third value: in the first two it becomes
`LOCAL`, with a warning, so `NORM BOTH` gives `LOCAL` fits and a `LOCAL` final model. Before the residuals are
computed, the model $m_{xyz}$ of the ring is scaled:

- `LOCAL`: each spectrum is scaled to the positive flux of the data in the mask in that pixel,

  $$
  m_{xyz} \leftarrow m_{xyz}\,\frac{\sum_z M_{xyz}\, D^+_{xyz}}{\sum_z m_{xyz}}, \qquad D^+ = \max(D, 0),
  $$

  (zero if the model spectrum is empty);
- `AZIM`: the whole ring is scaled to the flux of the data in the mask within the pixels $A$ of the ring,

  $$
  m_{xyz} \leftarrow m_{xyz}\,
  \frac{\sum_{(x,y)\in A}\sum_z M_{xyz} D_{xyz}}{\sum_{(x,y)\in A}\sum_z m_{xyz}} ;
  $$

- `NONE`: no scaling, the flux of the model is set by $\Sigma$.

With `LOCAL` and `AZIM`, $\Sigma$ has no effect on the residuals; `FREE ALL`, which also frees DENS, is then
degenerate in it.

## The residual function

The residual of ring $k$ is computed on the pixels $A$ of the box whose radius $r$, computed with the trial
geometry of the ring, lies in the annulus, $(R_k - w_k^-)/p \le r \le (R_k + w_k^+)/p$, where $p$ is the mean
pixel size (arcsec); with {ref}`SIDE <par-3dfit-side>` `R` only those with $|\theta| \le 90^\circ$, with `A`
only those with $|\theta| \ge 90^\circ$. For each pixel, all the channels $z$ of the cube are considered:
those in the mask, and those outside it where the model has emission (the blanks). With $D_{xyz}$ the data,
$m_{xyz}$ the normalised model and

$$
o_{xyz} = \begin{cases} D_{xyz} & D_{xyz} > 0,\\ \sigma_D & \text{otherwise,}\end{cases}
$$

the function minimised is

$$
f = \left(1 + \frac{N_b}{N}\right)^{\beta}\,\frac{1}{N - N_b}
\sum_{(x,y)\in A} w(\theta_{xy}) \sum_{z \in V_{xy}} \frac{\rho(m_{xyz}, o_{xyz})}{\sigma_z},
$$

where:

- $V_{xy}$ is the set of the channels of pixel $(x, y)$ with $M_{xyz} = 1$, or with $M_{xyz} = 0$ and
  $m_{xyz} \neq 0$; $N$ is the total number of voxels in the sum and $N_b$ the number of blanks among them
  ($M = 0$), so that $N - N_b$ is the number of voxels of the ring in the mask ($f = \infty$ if it is 0);
- $\beta$ is {ref}`BWEIGHT <par-3dfit-bweight>`: the factor $(1 + N_b/N)^\beta$ penalises models with
  emission where the data have none (Di Teodoro & Fraternali 2015, section 2.4);
- $\rho$ is set by {ref}`FTYPE <par-3dfit-ftype>`:

  | FTYPE | $\rho(m, o)$ |
  |---|---|
  | 1 | $(m - o)^2/\sqrt{o}$ |
  | 2 (default) | $\lvert m - o\rvert$ |
  | 3 | $\lvert m - o\rvert/(m + o)$ |
  | 4 | $(m - o)^2$ |

- $w(\theta) = |\cos\theta|^{n}$ for {ref}`WFUNC <par-3dfit-wfunc>` $= n \ge 0$, $|\sin\theta|^{|n|}$ for
  $n < 0$ (default $\cos^2\theta$, which favours the pixels near the major axis);
- $\sigma_z$ is the noise of channel $z$ with `MASK NEGATIVE`, otherwise 1 (it divides each term; it is
  not squared, also for FTYPE 1 and 4).

$f$ is thus a weighted mean residual per voxel of the mask, in the units of $\rho$ (flux units of the cube
for FTYPE 2).

**Blanks of ANALYTIC models.** A Monte Carlo model is zero wherever no sub-cloud fell, while the line wings
of an ANALYTIC model never vanish: counted as blanks, they would always pay the penalty of their full
wings and bias $\sigma$ high. A voxel of an ANALYTIC model outside the mask therefore counts as a blank (and
enters the sum) only where the expected number of Monte Carlo sub-clouds that contribute to it,

$$
\lambda_{xyz} = \frac{1}{u\,c_z}\sum_{(x',y') \in K_{xy}} m^\mathrm{pre}_{x'y'z}
\;\ge\; e^{-\gamma} \approx 0.561,
$$

where $m^\mathrm{pre}$ is the model before the convolution, $K_{xy}$ the footprint of the beam kernel
around the pixel, $u$ the mean flux of one sub-cloud for the model's {ref}`CDENS <par-3dfit-cdens>` and
{ref}`NV <par-3dfit-nv>`, $c_z$ the flux conversion of channel $z$, and $\gamma$ the Euler constant. With this
threshold the number of blanks equals on average that of Monte Carlo models. The model cubes written to
disk are not truncated.

**NORMALCUBE.** With {ref}`NORMALCUBE <par-3dfit-normalcube>` (the default), during each stage the data and
$\sigma_D$ are multiplied by $10/D_\mathrm{max}$, where $D_\mathrm{max}$ is the maximum of the cube; the cube
is restored afterwards. This changes the scale of $f$, not the minimum, except with `NORM NONE`, where
$\Sigma$ must then be in the units of the scaled cube.

## The minimisation

Each ring is fitted with the downhill simplex of Nelder & Mead (1965), in the implementation of Numerical
Recipes (Press et al. 2007), with reflection, expansion (factor 2), contraction (factor 0.5) and shrinking
towards the best vertex. With $n$ free parameters, the starting simplex has the vertices $P_0$, the initial
values of the ring, and $P_j = P_0 + \delta_j e_j$ ($j = 1, \dots, n$), with the steps

| Parameter | Step $\delta_j$ |
|---|---|
| VROT | 30 km/s if $V_\mathrm{rot} > 40$ km/s, else 5 km/s |
| VDISP | 20 km/s if $\sigma > 80$; 10 if $20 < \sigma \le 80$; 0.2 if $\sigma \le 2$; else 5 km/s |
| DENS, Z0 | 20% of the initial value |
| INC, PA | 5°, 10° |
| XPOS, YPOS | a quarter of their allowed range (below) |
| VSYS | a quarter of the velocity range of the cube |
| VRAD | 25 km/s |

A free $z_0$ must therefore start above 0.

### Bounds

Each trial point is checked before the model is built. A value outside its interval is moved, and the
vertex of the simplex keeps the moved value. With $p^0$ the initial value of the ring and $u$ a uniform random
number in $[0, 1)$:

| Parameter | Interval | Value outside it |
|---|---|---|
| VROT | $p^0 \pm$ {ref}`DELTAVROT <par-3dfit-deltavrot>`, $\ge 0$ | the bound $\mp u\,$DELTAVROT |
| VDISP | {ref}`MINVDISP <par-3dfit-minvdisp>`–{ref}`MAXVDISP <par-3dfit-maxvdisp>` | bound $\mp u\,s$ |
| INC | $p^0 \pm$ {ref}`DELTAINC <par-3dfit-deltainc>`, within 0° to 90° | the bound |
| PA | $p^0 \pm$ {ref}`DELTAPA <par-3dfit-deltapa>`, within −360° to 360° | the bound |
| Z0 | 0 to 1000 arcsec | $p^0$ |
| XPOS, YPOS | 10 pixels beyond the extreme initial centres, within the cube | $p^0$ |
| VSYS | the velocity range of the cube | $p^0$ |
| VRAD | −500 to 500 km/s | $p^0$ |
| DENS | none | |

For VDISP, $s = \min(\sigma^0,\ \mathrm{MAXVDISP} - \mathrm{MINVDISP})$, so the new value stays within the
interval; MINVDISP must be smaller than MAXVDISP (an error of the checks otherwise).

The random numbers of each ring come from a generator seeded with the stage and the ring, so that the
results do not depend on the number of threads.

### Convergence and failed rings

The simplex stops when the residuals at its best and worst vertices satisfy

$$
\frac{2\,|f_\mathrm{hi} - f_\mathrm{lo}|}{|f_\mathrm{hi}| + |f_\mathrm{lo}| + 10^{-10}} < \mathrm{TOL}
$$

({ref}`TOL <par-3dfit-tol>`); the best vertex is the result. If the number of evaluations reaches $200\,n$
first, the ring is *not converged*; it is not restarted. A ring can end in four ways:

- *skipped*: too few pixels in the mask (see *The mask*), before any fit;
- *not converged*;
- *rejected*: the fitted rotation exceeds what the spectral band allows,
  $V_\mathrm{rot} > \max(|v_1 - V_\mathrm{sys}|, |v_{N_z} - V_\mathrm{sys}|)/\sin i$, with $v_1$ and $v_{N_z}$
  the
  velocities of the first and last channel;
- *ok*: the values are stored, and the errors computed if requested.

The rings not ok, and those inside STARTRAD, are left out of the ring files unless
{ref}`BADOUT <par-3dfit-badout>` (then with FITOK = 0 and their initial values); after the last stage their
DENS is set to 0, so they are empty in the final model.

### Order of the rings

The rings are independent: each starts from its own initial values, not from the result of a neighbour.
They are fitted in parallel, {ref}`THREADS <par-all-threads>` at a time, from the ring of index
{ref}`STARTRAD <par-3dfit-startrad>` (counted from 0) outwards. In REVERSE mode they are fitted one after the
other, from the outermost ring inwards to STARTRAD, and the model of each includes the rings already fitted
outside it (see *The model of a ring*).

## The second stage

With {ref}`TWOSTAGE <par-3dfit-twostage>` (the default) and at least one of INC, PA, Z0, XPOS, YPOS and VSYS
free, the geometry of the first stage is regularised: each of these free parameters is replaced by a smooth
function of radius chosen with {ref}`REGTYPE <par-3dfit-regtype>`. The function is fitted to the values $p_k$
of the rings that were fitted (*ok*) in the first stage, and in this section $N$ is their number and $k$ runs
over them; the rings skipped, not converged, rejected or inside STARTRAD are left out. The function is then
evaluated at the radii of all the rings:

- `median`: the median of the $p_k$;
- an integer $d$: the polynomial $\sum_{j=0}^{d} c_j R^j$ fitted by least squares to $(R_k, p_k)$; the rings
  with the largest and the smallest value are left out when $N - 2 > d + 1$; with $N \le d + 1$ a constant
  is used;
- `bezier`: the Bézier curve with the control points $(R_k, p_k)$,

  $$
  \mathbf{B}(t) = \sum_{j=0}^{N-1} \binom{N-1}{j}\, t^j (1 - t)^{N-1-j}\,(R_{j+1}, p_{j+1}),
  $$

  whose value at $t_k = (k-1)/(N-1)$ is assigned to ring $k$ (for evenly spaced radii the curve passes at
  $R = R_k$ there); when some rings were not fitted, the points of the curve at the $t_k$ are interpolated
  linearly in $R$ at the radii of all the rings (the end values beyond the first and last ring fitted);
- `auto` (the default): for INC and PA, a constant (degree 0) with $N \le 4$; otherwise, with the robust
  scatter $s = \mathrm{MADFM}/0.6745$ of the $p_k$, a straight line ($N < 10$) or a Bézier curve
  ($N \ge 10$) if $s > 3^\circ$, the median if $s \le 3^\circ$. VSYS, XPOS, YPOS and Z0: the median.

These parameters are then fixed at the regularised values, and all the rings (from STARTRAD) are fitted
again for the free VROT, VDISP and VRAD (and DENS with `FREE ALL`), with the second value of NORM. Each ring
starts from its first-stage values, which are also the centres of the DELTAVROT intervals. The results, with
the regularised geometry, are written in `<prefix>_rings_final2.txt`. If a polynomial fit fails, or no ring
was fitted in the first stage, there is no second stage: the first stage is final, and the rings not fitted
in it are empty (DENS = 0) in the final model.

## Errors

With {ref}`FLAGERRORS <par-3dfit-flagerrors>`, the errors of the free parameters of each ring are estimated
after its fit (in both stages), one parameter $p$ at a time, with the others at their best values:

1. 100 values of $p$ are drawn uniformly in $\hat p \pm \hat p/3$ for VROT (above 0), $\hat\sigma \pm 15$ km/s
   for VDISP (above 0), or the global interval of the fit for the others (as in *Bounds*, around all the
   initial values), and the residual $f$ is computed for each.
2. With $\Delta = p - \hat p$, the sampled range of $\Delta$ is divided in 40 bins; in each, $y$ is the lowest
   value of $100\,(f - f_\mathrm{min})/f_\mathrm{min}$ (the increase of the residual in percent, at most
   $10^4$; 0 for an empty bin).
3. The cubic $y = a_2\Delta^2 + a_3\Delta^3$ (no constant and linear terms, so that the minimum is at the best
   fit) is fitted by least squares to the bins with $0 < y \le y_c$, where $y_c$ starts at 15 and grows by 5
   until enough bins are used.
4. The errors are the roots of $a_3\Delta^3 + a_2\Delta^2 = 5$, the change of $p$ that raises the residual by
   5%, within the sampled range: the lower error is the root closest to 0 below it, the upper error the one
   closest above; with roots on one side only, the interval is symmetric.
5. An error not found is replaced by the mean of the lower (or upper) errors of the other free parameters of
   the ring, whatever their units. VDISP errors smaller than half the dispersion of a channel,
   $\Delta v/(4\sqrt{2\ln 2})$, are replaced by that value plus a random number between 0 and 1 km/s. With
   `FREE ALL` the errors of DENS are not meaningful, since DENS has no interval to sample.

The lower (negative) and upper errors are written in the columns `E_VROT1`, `E_VROT2`, ... of the ring files.
These intervals are not statistical confidence intervals: they measure how fast the residual grows when
one parameter moves. METHOD BAYESIAN gives posterior distributions instead (and GLOBAL the errors of the Laplace approximation).

## The final model and the outputs

The final rings are those of the second stage (of the first without it). The model of all the rings is built
with the ring convention RINGMODE, convolved with the beam and normalised with the last value of NORM:

- `AZIM`: the model built with the densities DENS of the deconvolved profile (below), as it is. It is no
  longer rescaled to the observed profile of `_densprof.txt`, $\Sigma_k = \cos i_k\, S_k/A_k$, where $S_k$
  is the sum of the total intensity map within the mask over the ring and $A_k$ its area on the sky (arcsec²),
  as in ELLPROF. If the deconvolved profile cannot be computed (no beam, too few pixels), the observed profile
  is used. The rings with DENS = 0 stay empty. (Internally the densities are multiplied by a power of
  ten that brings the smallest one between 0.1 and 10, so that Monte Carlo models have a sensible number of
  clouds, and the model is divided by it afterwards.)
- `LOCAL`: each spectrum of the model is scaled to the flux of the data in the mask in the same pixel,
  $\sum_z M_{xyz} D_{xyz} / \sum_z m_{xyz}$, starting from the model of `AZIM` (the deconvolved densities).
- `NONE`: the model with the DENS of the rings, which are the deconvolved densities (the fitted ones with
  FREE ALL, below).
- `BOTH` (or {ref}`DEBUG <par-all-debug>`): the `AZIM` and the `LOCAL` models.

**The deconvolved profile.** At the end of the run, 3DFIT computes the density profile deconvolved from the
beam at the geometry of the final rings, by the same method as the deconvolved profile of
{doc}`ELLPROF <../ellprof/algorithm>` and of {doc}`METHOD GLOBAL and BAYESIAN <likelihood>`: the densities
whose beam-convolved model reproduces the total intensity map of the data in the mask dilated by one beam,
with a smoothness prior on the profile. The errors come from 200 noise realisations. With SIDE `A` or `R`
only one half of the galaxy is used. It is written to `<prefix>_densprof_deconvolved.txt`, in the format of
`_ellprof_deconvolved.txt`: RADIUS, SDENS_FO, E_LO, E_HI, SDENS_FO_OBS (the observed face-on profile), DENS
and, for HI data in Jy, HIMASSDENS, E_LO, E_HI and HIMASSDENS_OBS. In Python it is
`res.densprof_deconvolved`, an astropy Table (`radii`, `sdens_fo`, `e_sdens_fo_lo`, `e_sdens_fo_hi`,
`sdens_fo_obs`, `dens`, ...). With `FREE ALL` (DENS fitted) the profile is computed and written, but not
used. The densities DENS of the profile are those of the final models; the ring normalisation inside the
fit is unchanged. On NGC 2403 the `AZIM` model holds 96% of the flux of the data in the mask, against 90%
with the observed profile. GALMOD is unchanged: it still uses the observed profile for `AZIM`. The
surface-density panel of `<prefix>_parameters.pdf` shows the face-on deconvolved profile, with its errors,
and the observed face-on profile (open grey circles), in map units per arcsec².

With {ref}`NOISERMS <par-3dfit-noiserms>`, a copy with Gaussian noise (convolved with the beam, then
rescaled to that rms) is also written. The outputs, in the output folder:

| File | Content |
|---|---|
| `<prefix>_rings_initial.txt` | the initial rings |
| `<prefix>_2drings.txt` | the 2D fits of the initial estimates and of the PA refinement |
| `<prefix>_rings_final1.txt`, `_rings_final2.txt` | the fitted rings of the two stages, with the errors |
| `<prefix>_densprof.txt` | the radial profile of the data in the final rings |
| `<prefix>_densprof_deconvolved.txt` | the density profile deconvolved from the beam, in the final rings |
| `<prefix>_mod_<norm>.fits` | the final model (`azim`, `local` or `none`), and `_mod_noise.fits` |
| `maps/` | maps 0, 1, 2 (moments, or Gaussian fits with MAPTYPE GAUSSIAN) of the data (in the mask) and of the model |
| `pvs/` | position–velocity slices of data, model and mask along the major and minor axes |
| `<prefix>_asymdrift.txt` | the asymmetric drift correction (ADRIFT) |
| plots | channel maps, position–velocity diagrams and ring parameters ({ref}`PLOTS <par-all-plots>`) |

### Asymmetric drift

With {ref}`ADRIFT <par-3dfit-adrift>`, the circular velocity is computed from the final rings that were fitted
(*ok*), from STARTRAD outwards (at least two are needed), following Iorio et al. (2017):

$$
V_c^2 = V_\mathrm{rot}^2 + V_A^2, \qquad
V_A^2 = -R\,\sigma^2\left(\frac{\mathrm{d}\ln\sigma^2}{\mathrm{d}R}
+ \frac{\mathrm{d}\ln\Sigma}{\mathrm{d}R}\right),
$$

where $R$ is the radius (arcsec, so the derivatives are per arcsec and $V_A^2$ is in (km/s)²), $\sigma$ the
VDISP of the rings and $\Sigma$ the face-on surface density, the median of the total intensity map in each
ring times $\cos i$. Both are regularised before the derivatives: $\sigma^2(R)$ by a least-squares
polynomial of degree {ref}`ADRIFTPOL1 <par-3dfit-adriftpol1>` (a negative value uses the VDISP of the rings
and takes $\mathrm{d}\sigma^2/\mathrm{d}R = 0$), $\Sigma(R)$ by one of degree
{ref}`ADRIFTPOL2 <par-3dfit-adriftpol2>`; with $n$ rings the degrees are at most $n - 2$. $V_c$ is undefined
(NaN) where $V_c^2 < 0$. The file has $R$, $V_c$, $V_A^2$, $\sigma^2$ and $\Sigma$ with their regularised
values; if a polynomial fit fails nothing is written, with a warning.

## Accuracy and known biases

Tests on simulated disks (3304 fits; the report is in `performance/report` of the repository) show:

- $V_\mathrm{rot}$ is robust: with the geometry known, at 5 beams or more across the disk and a peak S/N of 10
  or more, the median error is between −2% and 0%, with a scatter of 3–6% from ring to ring.
- $\sigma$ is hard: 20–40% scatter, and a bias that follows the S/N, from −50% at a peak S/N of 5 to +7% at
  40. At low S/N the mask cuts the line wings, which biases $\sigma$ low and $i$ high.
- $i$ stays close to its initial value (DELTAINC); $\phi$ is recovered to about 2°.
- At high S/N, `NORM AZIM` scales a model that lacks the beam-smeared emission of the neighbouring rings:
  $i$ is biased low (−7° at S/N 200) and $\sigma$ high. `NORM LOCAL` removes this bias. (The deconvolved
  profile addresses this for the final model, whose densities now account for the beam smearing; the
  normalisation of the rings in the fit itself is unchanged, and these tests do not use it.)

These are properties of the algorithm, shared by BBarolo 1.8.

## Differences from BBarolo 1.8

- **Ring convention of the final model.** The final model uses RINGMODE `CENTRED` by default; BBarolo 1.8
  moved the first and last radius by RADSEP/2 (`SHIFTED`), which bends the profiles in the first and last
  ring. The fit and the ring tables do not change.
- **ANALYTIC models** (GALMODTYPE) are new, with the rule for their blanks described above. The default is
  still `MC`. Monte Carlo models use another random generator, so results agree with 1.8 to the level of the
  Monte Carlo noise, not bit for bit; the ring tables do not depend on THREADS.
- **Default Z0**: one pixel, a thin disk, instead of 0.
- **Initial estimates**: GUESSTYPE `ELLIPSE` and `FIT` are new; `LEGACY` (the default) is the algorithm of
  1.8.
- **Bugs fixed** that also affect 1.8: with `FREE ALL` and TWOSTAGE, DENS stayed free in the second stage
  without being counted, so the simplex was degenerate (and the error columns misaligned); the refinement of
  the PA read beyond the end of the 2D fit for the last ring; cubes with spatial axes in arcsec or arcmin
  gave wrong beams and ring widths. Also: the second stage regularised all the rings, also those not fitted,
  with their initial or rejected values, and these rings kept their DENS in the final model when the
  regularisation failed (ADRIFT used them too); VDISP values out of bounds were redrawn as the bound
  $\mp u\sigma^0$, which could cross the other bound; the beam-wide rings of REVERSE took the initial values
  of the ring with the same index; Monte Carlo models in a box were not crops of the model of the whole cube
  when clouds fell outside the box; the PA refinement replaced a fixed VROT; RADSEP alone (without NRADII)
  kept the number of the estimated rings, so their extent scaled with RADSEP; `NORM AZIMUTHAL` was used as
  `LOCAL`.
- **Outputs**: no DISTANCE, so the ring files have no RAD(Kpc) and Z0(pc) columns (VROT is column 2); all the
  file names start with `<prefix>_`; RINGFILE finds the columns by their names (files of 1.8 are read).
- **Checks**: a VSYS outside the velocity range of the cube, unknown names in FREE and MINVDISP ≥ MAXVDISP
  are errors; rejected rings say why.

The full list is in {doc}`../../reference/changes`.

## References

- Begeman K. G., 1989, A&A, 223, 47: tilted-ring fits of velocity fields.
  [ADS](https://ui.adsabs.harvard.edu/abs/1989A&A...223...47B)
- Di Teodoro E. M., Fraternali F., 2015, MNRAS, 451, 3021: the 3D fitting algorithm of BBarolo.
  [ADS](https://ui.adsabs.harvard.edu/abs/2015MNRAS.451.3021D)
- Iorio G., Fraternali F., Nipoti C., Di Teodoro E., Read J. I., Battaglia G., 2017, MNRAS, 466, 4159:
  asymmetric drift correction. [ADS](https://ui.adsabs.harvard.edu/abs/2017MNRAS.466.4159I)
- Nelder J. A., Mead R., 1965, Computer Journal, 7, 308: the downhill simplex.
  [ADS](https://ui.adsabs.harvard.edu/abs/1965CompJ...7..308N)
- Press W. H., Teukolsky S. A., Vetterling W. T., Flannery B. P., 2007, Numerical Recipes, 3rd edn,
  Cambridge University Press.
- Rogstad D. H., Lockhart I. A., Wright M. C. H., 1974, ApJ, 193, 309: the tilted-ring model.
  [ADS](https://ui.adsabs.harvard.edu/abs/1974ApJ...193..309R)

## In the code

- `bbarolo/galfit.py`: `initial_rings` (initial rings, RINGFILE, defaults); the class `Galfit`: `setup`
  (mask, bounds, normalisations), `galfit` (one stage), `_fit_one_ring`, `minimize` and `func3D` (simplex and
  bounds), `getFuncValue` and the numba kernel `_norm` (residuals), `fit_reverse`, `SecondStage` and
  `regularizeParams`, `getErrors`.
- `bbarolo/paramguess.py`: `EstimateInitial` and `ParamGuess` (GUESSTYPE).
- `bbarolo/galmod.py`: `DiskModel` (`box`, `array`; ring conventions in `model_rings`).
- `bbarolo/galfit_out.py`: `writeModel` (final model and outputs), `radialProfile`, `asymmetricDrift`.
- `bbarolo/cube.py`: `Cube.BlankMask` (the mask); `bbarolo/ringmodel.py`: `Ringmodel` (2D fits).
- `bbarolo/tasks.py`: the task (`_compute_3dfit`, `_write_3dfit`) and the Python class `FitMod3D`.
