# GalMod3D (Python only): how it works

GalMod3D builds a model datacube of a galaxy from three-dimensional fields of its gas: the density
$\rho$, the velocity $\mathbf{v}$ and the isotropic velocity dispersion $\sigma_{\rm gas}$, given as
functions of $(R, z)$, $(R, \phi, z)$ or $(x, y, z)$ in the frame of the galaxy. The model is observed
on the grid of a given cube (pixels, channels, spectral axis, beam) with the geometry of
{doc}`GALMOD <../galmod/algorithm>`: a centre, a systemic velocity, an inclination and a position angle.
Where GALMOD describes the gas by rings with a fixed vertical layer, GalMod3D takes any distribution:
a flaring or lagging thick disk, a bar or spiral arms with streaming motions, a warp written in the
density, or the fields of a simulation on a grid. The output is the model cube, as an astropy HDU with
the header of the input cube, scaled to a total flux or not. GalMod3D is a Python class,
{class}`bbarolo.GalMod3D`, not a task of the command line.

A model is built in five steps: the fields are sampled on a regular Cartesian grid in the frame of the
galaxy; each grid point is projected on the sky and given its line-of-sight velocity; it is deposited
in its pixel with a Gaussian line integrated over the channels (or as random clouds); the cube is
convolved with the beam; the model is scaled to the requested flux.

## Coordinates and conventions

The frame of the galaxy has its origin at the centre, the $x$ axis along the receding half of the major
axis, the $y$ axis in the plane of the galaxy towards its far side (away from the observer), and the $z$
axis along the rotation axis, with $z > 0$ on the side of the plane away from the observer. $x$, $y$,
$z$ are in arcsec. The cylindrical coordinates are

$$
R = \sqrt{x^2 + y^2}, \qquad \phi = \operatorname{atan2}(y, x) \bmod 2\pi, \qquad
x = R\cos\phi, \quad y = R\sin\phi,
$$

with $R$ in arcsec and the azimuth $\phi$ in radians, in $[0, 2\pi)$: $\phi = 0$ is the receding half
of the major axis and $\phi = \pi/2$ the far side of the minor axis. This is the azimuth $\theta$ of the
{ref}`geometry of GALMOD <galmod-geometry>`.

The argument `coords` sets the arguments of the fields and the names of the velocity components:

| `coords` (aliases) | Fields of | Velocities (km/s) |
|---|---|---|
| `"Rz"` (`"axisymmetric"`, the default) | $(R, z)$ | `vr`, `vphi`, `vz` |
| `"Rphiz"` (`"cylindrical"`) | $(R, \phi, z)$ | `vr`, `vphi`, `vz` |
| `"xyz"` (`"cartesian"`) | $(x, y, z)$ | `vx`, `vy`, `vz` |

$v_R$ (`vr`) is positive outwards, $v_\phi$ (`vphi`) positive towards increasing $\phi$ (a disk with
$v_\phi > 0$ rotates as the disks of GALMOD: the receding side is at the position angle), and $v_z$
(`vz`) positive towards $z > 0$, away from the observer. A velocity not given is 0. The names are not
case sensitive; a velocity of another system of coordinates (e.g. `vx` with `"Rz"`) is an error.

## The fields

The fields are $\rho$ (`dens`, required), the velocity components and $\sigma_{\rm gas}$ (`vdisp`,
km/s, default 0). Each is one of:

- **a number**: a constant field;
- **a function** of the coordinates, in the order of the table, vectorised: it is called with arrays
  of the coordinates of many points at once and returns an array of the same shape (or one that
  broadcasts to it, e.g. a scalar);
- **a grid**: a tuple `(axes, values)`, with one 1D array of increasing values (at least two) for each
  coordinate and `values` of shape `(len(axes[0]), len(axes[1]), ...)`. The field is interpolated
  linearly in all the coordinates. Outside the grid the density is 0 and the other fields take the value
  at the nearest edge of the grid. The axis of $\phi$ is periodic: a point at $\phi$ is taken at
  $\phi_0 + [(\phi - \phi_0) \bmod 2\pi]$, with $\phi_0$ the first value of the axis; if the axis spans
  less than $2\pi$, the values at $\phi_0$ are repeated at $\phi_0 + 2\pi$, so the interpolation is
  continuous across $\phi_0$. An axis of $\phi$ wider than $2\pi$ is an error.

The density is in any units; it must be $\ge 0$ (a negative value is an error, a value that is not
finite is taken as 0). $\rho$ is a volume density, per arcsec³, except for a razor-thin model
($z_{\max} = 0$, below), where it is a surface density, per arcsec².

## The sampling grid

The fields are sampled on a regular Cartesian grid in the frame of the galaxy, with spacing $\Delta_{xy}$
in $x$ and $y$ and $\Delta_z$ in $z$ (arcsec), the argument `step`: a pair $(\Delta_{xy}, \Delta_z)$ or
one value for both. By default $\Delta_{xy} = \Delta_z = p/3$, with $p = \min(|\Delta x|, |\Delta y|)$
the smaller pixel size of the cube (arcsec). The points are the centres of the cells

$$
x_a = \left(a + \tfrac12 - \tfrac{n}{2}\right) \Delta_{xy}, \quad a = 0, \dots, n - 1, \qquad
n = \left\lceil \frac{2 R_{\max}}{\Delta_{xy}} \right\rceil,
$$

the same for $y_b$, and

$$
z_c = \left(c + \tfrac12 - \tfrac{n_z}{2}\right) \Delta_z, \quad c = 0, \dots, n_z - 1, \qquad
n_z = \left\lceil \frac{2 z_{\max}}{\Delta_z} \right\rceil,
$$

symmetric about the centre. The grid covers the box $|x|, |y| \le R_{\max}$, $|z| \le z_{\max}$ (it is
square in $x$, $y$: the points at $R > R_{\max}$ in its corners are included). $R_{\max}$ (`rmax`,
arcsec) is by default the distance on the sky from the centre to the farthest corner of the cube, plus
one pixel $p$. $z_{\max}$ (`zmax`, arcsec) is by default $R_{\max}$; for a disk a smaller value makes
the model much faster. $z_{\max} = 0$ gives a single plane at $z = 0$: a razor-thin model.

:::{note}
The grid is cut in the frame of the galaxy, not on the sky. Along the minor axis an inclined disk
projects to $|y|\cos i$, so with the default $R_{\max}$ the disk may extend beyond the grid at
$|y| > R_{\max}$ and still fall inside the cube: give `rmax` explicitly for strongly inclined disks.
$\Delta_z$ must resolve the vertical structure of the density.
:::

Each point carries the weight

$$
w = \rho(x, y, z)\, \Delta V, \qquad
\Delta V = \begin{cases} \Delta_{xy}^2\, \Delta_z & z_{\max} > 0,\\ \Delta_{xy}^2 & z_{\max} = 0, \end{cases}
$$

so that $\sum w$ is the integral of $\rho$ over the volume (or the area) of the grid, in units of
$\rho$ × arcsec³ (or arcsec²). Points with $\rho = 0$ are skipped.

## Projection on the sky

A point $(x, y, z)$ of the galaxy, seen at inclination $i$ (`inc`, degrees; $i = 0$ face-on), lies on
the sky at $x$ along the major axis and

$$
y_s = y\cos i - z\sin i
$$

along the minor axis (arcsec). Its pixel, counted from 0, is

$$
p_x = x_0 + \frac{x\sin\Gamma - y_s\cos\Gamma}{\Delta x}, \qquad
p_y = y_0 + \frac{x\cos\Gamma + y_s\sin\Gamma}{\Delta y}, \qquad \Gamma = {\rm PA} + {\rm CROTA2},
$$

with $(x_0, y_0)$ the centre (`xpos`, `ypos`: pixels counted from 0, or world coordinates as in GALMOD,
e.g. `'123.4d'` or `'07h36m51s'`), PA (`pa`, degrees) the position angle of the receding half of the
major axis from North towards East, and $\Delta x$, $\Delta y$ the signed pixel sizes (CDELT1, CDELT2,
in arcsec). These are the equations of GALMOD.

The line-of-sight velocity (km/s, positive away from the observer) is

$$
v_{\rm los} = V_{\rm sys} + v_y \sin i + v_z \cos i,
$$

with $V_{\rm sys}$ (`vsys`, km/s, default 0) the systemic velocity, in the frame of the spectral axis
(see {ref}`galmod-spectral`), and $v_y$ the component of the velocity along $y$:

$$
v_y = v_R \sin\phi + v_\phi \cos\phi \quad (\texttt{"Rz"}, \texttt{"Rphiz"}), \qquad
v_y = v_y \quad (\texttt{"xyz"}).
$$

The component along $x$ (`vx`, or $v_R\cos\phi - v_\phi\sin\phi$) lies in the plane of the sky and does
not contribute. All the fields are evaluated at the point itself, so any dependence on $z$ (e.g. a lag
$\partial v_\phi/\partial z$) or on $\phi$ (e.g. streaming motions in a bar) enters the model directly.

## The deposit: `galmodtype="ANALYTIC"`

The default model (`model(galmodtype="ANALYTIC")`) follows the {ref}`ANALYTIC models of GALMOD
<galmod-analytic>`, with the grid points in place of the samples of the sub-rings. The point is deposited
in its nearest pixel, $(\operatorname{round}(p_x), \operatorname{round}(p_y))$; points outside the cube
are lost. Its line is a Gaussian of dispersion

$$
\sigma = \sqrt{\sigma_{\rm gas}^2 + \sigma_{\rm instr}^2}
$$

(at least $10^{-6}$ of the narrowest channel), centred on $v_{\rm los}$, integrated over each channel
$k$:

$$
f_k = \left| \Phi\!\left(\frac{u_{k+1/2} - v_{\rm los}}{\sigma}\right)
  - \Phi\!\left(\frac{u_{k-1/2} - v_{\rm los}}{\sigma}\right) \right|,
$$

with $\Phi$ the standard normal cumulative distribution (tabulated as in GALMOD) and $u_{k\pm1/2}$ the
velocities of the edges of channel $k$. The channel $k$ of the pixel receives $w\, f_{k,\ell}\, I_\ell$ for
each line $\ell$ of relative intensity $I_\ell$ (one line: $I = 1$), where $f_{k,\ell}$ uses the channel
edges in the velocity frame of line $\ell$, computed from its own reference frequency: the lines of a point
share its Doppler factor, as in GALMOD. The profile is computed only within
$K = \lfloor 5\sigma/\Delta v_{\min} \rfloor + 2$ channels of the central one ($\Delta v_{\min}$: the
narrowest channel). The model before the convolution is therefore

$$
M(p_x, p_y, k) = \sum_{\ell} I_\ell \sum_{\text{points in } (p_x, p_y)} \rho\, \Delta V\, f_{k,\ell},
$$

and $\sum_{x,y,k} M = \sum_\ell I_\ell \sum w$, apart from the points outside the cube and the flux
outside the spectral range. Unlike GALMOD, no factor $1/\Delta v_{\rm ch}$ is applied: the sum of the
pixels, not the sum times the channel width, is the integral of the density.

## The deposit: `galmodtype="MC"`

With `model(galmodtype="MC")` the model is made of $N$ random clouds (`nclouds`, default $10^6$), as the
{ref}`MC models of GALMOD <galmod-mc>`:

1. The cells of the grid are drawn with probability proportional to $\rho$ at their centre (that is, to
   $w$); each cloud is placed uniformly within its cell, at $(x_a, y_b, z_c) + (u_1\Delta_{xy},
   u_2\Delta_{xy}, u_3\Delta_z)$ with $u_{1,2,3}$ uniform in $[-1/2, 1/2)$ ($z = 0$ for a razor-thin
   model). The density is constant within a cell.
2. Every cloud has the same weight $W = \sum w / N$, so the expected model has the flux of the ANALYTIC
   one.
3. The velocity and the dispersion are evaluated at the position of the cloud; the cloud gets the
   velocity $v_{\rm los} + \sigma g$, with $g$ a standard normal deviate and $\sigma$ as above, and adds
   $W I_\ell$, for each line $\ell$, to the channel of line $\ell$ nearest to that velocity in its nearest
   pixel (the channel of the frequency $\nu_\ell(v_{\rm los} + \sigma g)$, see GALMOD).

The random numbers come from numpy's default generator (PCG64), started from `seed` (`model(seed=...)`):
the same seed gives the same model. The Monte Carlo noise falls as $N^{-1/2}$.

## Spectral axis, beam and noise

The spectral axis, the conversion from velocity to channel, several lines and the instrumental
broadening are those of GALMOD ({ref}`galmod-spectral`), set by the options
{ref}`SPECRES <par-galmod-specres>` (or {ref}`LINEAR <par-galmod-linear>`; by default
$\sigma_{\rm instr}$ gives a FWHM of one channel), {ref}`RESTWAVE <par-galmod-restwave>`,
{ref}`RESTFREQ <par-galmod-restfreq>`, {ref}`RELINT <par-galmod-relint>` and
{ref}`REDSHIFT <par-galmod-redshift>`, given as keyword arguments (e.g. `specres=0.5`). The other
options are {ref}`MASK <par-galmod-mask>` (for `flux="measured"`), {ref}`THREADS <par-all-threads>`
and {ref}`VERBOSE <par-all-verbose>`.

With `model(smooth=True)` (the default) every channel is convolved with the beam, by the
{ref}`convolution of GALMOD <galmod-convolution>`: a 2D Gaussian kernel normalised to unit sum, applied
with zero padding, so the flux is conserved apart from the emission spread outside the cube. The beam
is that of the header of the cube, or `beam=(bmaj, bmin, bpa)` (arcsec, arcsec, degrees; the keyword {ref}`BEAM <par-all-beam>`). After the
convolution, values below $10^{-12}$ of the peak are set to 0.

With `model(noise=σn)`, Gaussian noise of rms $\sigma_n$ per voxel (in the units of the final model)
is added after the flux scaling. With `smooth=True`, as in GALMOD, it is first convolved with the beam and
rescaled to the standard deviation $\sigma_n$; with `smooth=False` it is white noise.

## Flux scaling

The argument `flux` sets the total flux of the final (convolved) model $M$:

- **`None`** (default): no scaling. $\sum M$ is $\sum w$, the integral of $\rho$ over the grid (in units
  of $\rho$ × arcsec³), less the emission lost outside the cube.
- **a number** $F$ (Jy km/s): the model is multiplied by

  $$
  s = \frac{F}{\mathcal{J}\!\left(\sum_k \Delta v_k \sum_{x,y} M(x, y, k)\right)},
  $$

  with $\Delta v_k = |u_{k+1/2} - u_{k-1/2}|$ the width of channel $k$ (km/s) and $\mathcal{J}$ the
  conversion from the units of the cube (BUNIT) × km/s to Jy km/s: for Jy/beam (also mJy/beam,
  µJy/beam) a division by the area of the beam in pixels, $\Omega_b/\bar p^{\,2}$, with
  $\Omega_b = \pi b_{\rm maj} b_{\rm min}/(4\ln 2)$ and $\bar p = (|\Delta x| + |\Delta y|)/2$; for Jy
  (per pixel) none; for K the Rayleigh-Jeans conversion at the rest frequency of the cube (that of the
  header, or `RESTFREQ`);
  for W.U. a factor $5\times10^{-3}$ Jy/beam per W.U. Other units are an error.
- **`"measured"`**: the model is scaled to the flux of the data $D$ of the cube within its mask $\mu$
  (1 inside, 0 outside), built as set by {ref}`MASK <par-galmod-mask>` (default `SEARCH`):

  $$
  s = \frac{\sum_k \Delta v_k \sum_{x,y} D(x, y, k)\, \mu(x, y, k)}{\sum_k \Delta v_k \sum_{x,y} M(x, y, k)},
  $$

  with blank voxels of the data ignored. The data must have positive flux in the mask.

The scaling multiplies the whole model: the shape of the cube depends only on the fields and the
geometry. A model with no emission inside the cube is an error (check the centre, $\rho$, $R_{\max}$
and $z_{\max}$).

## Validation against GALMOD

A GALMOD disk can be written as fields: with rings given as nodes (`RINGMODE NODES`), a Gaussian layer of
scale height $z_0$ and $\Sigma(R)$, $V_{\rm rot}(R)$ interpolated linearly in radius,

$$
\rho(R, z) = \Sigma(R)\, e^{-z^2/(2 z_0^2)}, \qquad v_\phi = V_{\rm rot}(R), \quad
v_R = V_{\rm rad}, \quad v_z = V_{\rm vert}, \quad \sigma_{\rm gas} = \sigma .
$$

On a test cube of 64 × 64 pixels of 10″ and 40 channels of 10 km/s, with a 30″ beam, $z_0 = 8″$,
$V_{\rm rad} = 15$ km/s, $V_{\rm vert} = 5$ km/s, $i = 60°$ and $\Delta = (10/3, 2)″$, GalMod3D
reproduces the GALMOD ANALYTIC model, normalised to the same flux, within 0.1% of the peak (rms) and 3% of
the peak in any voxel, with a correlation coefficient above 0.9999. The largest differences are in the
central pixels, where GALMOD samples the innermost sub-ring coarsely. The test suite also checks:

- the conventions: a compact blob placed on the $x$, $y$ or $z$ axis with a velocity along $y$ or $z$
  falls at the expected pixel and velocity ($v_y\sin i$, $v_z\cos i$) within 0.1 pixel or km/s;
- that the same disk written as a function of $(R, z)$, $(R, \phi, z)$ or $(x, y, z)$ gives the same
  cube (within $10^{-6}$ of the peak), and as values on a grid the same within the interpolation
  ($2\times10^{-3}$ of the peak); a grid in $(R, \phi, z)$ constant in $\phi$ gives the axisymmetric
  model, through the periodic axis;
- that the MC model has the flux of the ANALYTIC one (within $10^{-5}$) and its shape, and is
  reproducible with a seed; that `flux` gives the requested flux in Jy km/s and `"measured"` the flux of
  the data.

## Differences from BBarolo 1.8

GalMod3D is new in BBarolo 2: it has no counterpart in BBarolo 1.8, and GALMOD does not change. With
respect to GALMOD, a user should know that GalMod3D has no rings and no ring convention (`RINGMODE`),
no vertical layer (`LTYPE`, `Z0`: the vertical structure is in $\rho$), no column densities
(`DENSFLUX`), no normalisation to the data other than the total flux (no `NORM AZIM` or `LOCAL`), and
that its MC models use numpy's generator, not that of GALMOD (`ISEED`, `CDENS`, `NV` are not used).

The full list of changes is in {doc}`../../reference/changes`.

## References

- Di Teodoro E. M., Fraternali F., 2015, MNRAS, 451, 3021:
  [2015MNRAS.451.3021D](https://ui.adsabs.harvard.edu/abs/2015MNRAS.451.3021D)
- van der Hulst J. M., Terlouw J. P., Begeman K. G., Zwitser W., Roelfsema P. R., 1992, ASP Conf. Ser.
  25, 131 (GIPSY): [1992ASPC...25..131V](https://ui.adsabs.harvard.edu/abs/1992ASPC...25..131V)

## In the code

- `bbarolo/galmod3d.py`: `GalMod3D` (`model`, `_grid`, `_args`, `_project`, `_model_mc`,
  `_flux_scale`); `_Field` (numbers, functions, grids); the numba kernels `_deposit_analytic` and
  `_deposit_mc`. The grid is processed in groups of $z$ planes of about $2\times10^6$ points.
- `bbarolo/galmod.py`: `DiskModel` and `GalmodSetup` (spectral axis, beam kernel, `addnoise`),
  `_line_profile`, `_velgrid`, `model_hdu`.
- `examples/galmod3d.py`: a thick lagging disk, a bar with spiral arms, a warped disk on a Cartesian grid.
- Tests: `tests/test_bbarolo.py`, `test_galmod3d_*`.
