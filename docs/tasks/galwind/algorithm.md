# GALWIND: how it works

GALWIND builds a model datacube of a bi-conical outflow on the grid of the input cube. The outflow is made of
two opposite cones with a common apex at the centre of a disk, along its rotation axis: the gas moves away
from the centre along the cones, may rotate around the axis and has a velocity dispersion. The cones are
built either as stacks of cylinders, each a thick disk modelled by GALMOD with vertical and radial velocities
({ref}`WTYPE <par-galwind-wtype>` 0, the default), or as spherical shells filled within the opening angle
(WTYPE 1). The model is projected with the inclination and position angle of the disk, convolved with the
beam and written with its moment maps and a position–velocity slice. GALWIND does not fit: it makes models
to compare with data. The model of a disk, its units and its spectral axis are described in
{doc}`../galmod/algorithm`.

## Geometry

The apex is at the centre $(x_0, y_0)$ ({ref}`XPOS <par-galwind-xpos>`, {ref}`YPOS <par-galwind-ypos>`: pixels
or world coordinates). The disk has inclination $i$ ({ref}`INC <par-galwind-inc>`) and position angle $\phi$
({ref}`PA <par-galwind-pa>`, the receding half of the major axis, from North towards East), in degrees. In the
frame of the disk, $(x, y)$ are in the plane, with $x$ along the major axis, and $z$ is along the rotation
axis. A point at $(x, y, z)$ (arcsec) projects on the sky at

$$
x_\mathrm{sky} = x, \qquad y_\mathrm{sky} = y\cos i - z\sin i ,
$$

along and across the major axis, rotated by $\phi$ (and by the rotation of the WCS) onto the pixels of the
cube, as in GALMOD. The axis of the cones therefore projects onto the minor axis of the disk, and a height $z$
moves a point by $z\sin i$ along it. A gas element with velocity $(v_x, v_y, v_z)$ has the line-of-sight
velocity

$$
V_\mathrm{los} = V_\mathrm{sys} + v_y \sin i + v_z \cos i
$$

($V_\mathrm{sys}$ is {ref}`VSYS <par-galwind-vsys>`, km/s). The cones have the full opening angle
$\theta_\mathrm{o}$ ({ref}`OPENANG <par-galwind-openang>`, degrees) and extend to $H$
({ref}`HTOT <par-galwind-htot>`, arcsec). The outflow velocity is $V_\mathrm{w}$
({ref}`VWIND <par-galwind-vwind>`, km/s; positive values are an outflow), the rotation velocity around the
axis $V_\mathrm{rot}$ ({ref}`VROT <par-galwind-vrot>`, 0 by default) and the velocity dispersion $\sigma$
({ref}`VDISP <par-galwind-vdisp>`), to which the instrumental broadening is added in quadrature, as in GALMOD.
Several emission lines ({ref}`RESTWAVE <par-galmod-restwave>` or {ref}`RESTFREQ <par-galmod-restfreq>` with
{ref}`RELINT <par-galmod-relint>`) are modelled as in GALMOD: each line has its own reference frequency, so
the lines of a cloud share its Doppler factor; on a velocity axis only one line is modelled, with a warning.

## Cylinders (WTYPE 0)

Each cone is a stack of $N$ ({ref}`NTOT <par-galwind-ntot>`, default 25) cylinders of height $\Delta z = H/N$.
Cylinder $k$ ($k = 1, \dots, N$, from the centre outwards) has the half-height and outer radius

$$
z_0 = \frac{\Delta z}{2} = \frac{H}{2N}, \qquad
R_{\mathrm{out},k} = k\,w, \qquad w = \Delta z \tan\frac{\theta_\mathrm{o}}{2} \quad \text{(arcsec)},
$$

so that the edges of the cylinders follow the walls of the cone. Its mid-plane is at the height
$z_k = \pm(2k - 1)\,z_0$, one cylinder in each cone. Each cylinder is a GALMOD disk with:

- its centre moved to the projection of the point $(0, 0, z_k)$ of the axis, i.e. by $z_k\sin i$ along the
  minor axis; the inclination and position angle are those of the disk;
- the half-height $z_0$, with a box vertical profile (uniform within $\pm z_0$ of its mid-plane, LTYPE 5): the
  cylinders of each cone stack exactly, from the plane to $H$;
- rings at the radii $R_j = R_\mathrm{in} + j\,(R_{\mathrm{out},k} - R_\mathrm{in})/n$, $j = 0, \dots, n$,
  with the values set by {ref}`DENSTYPE <par-galwind-denstype>`:

  | DENSTYPE | Radii | Surface density $\Sigma_k$ | Angle of the velocity $\alpha_j$ |
  |---|---|---|---|
  | 0 | $R_\mathrm{in} = 0$, $n = k$ | $\Sigma/k^2$ | $(j/k)\,\theta_\mathrm{o}/2$ |
  | 1 (default) | $R_\mathrm{in} = 0$, $n = k$ | $\Sigma$ | $(j/k)\,\theta_\mathrm{o}/2$ |
  | 2 | $R_\mathrm{in} = (k-1)\,w$, $n = 1$ | $\Sigma$ | $\theta_\mathrm{o}/2$ |

  where $\Sigma$ is {ref}`DENS <par-galwind-dens>`. With DENSTYPE 0 and 1 the cylinders are filled discs (the
  rings are $w$ apart); with 2 each cylinder is a hollow annulus between the radii of cylinders $k-1$ and $k$,
  and the gas lies on the wall of the cone;
- the velocities, at the ring radii $R_j$ (and interpolated linearly in radius between them, as GALMOD does),

  $$
  V_\mathrm{rad} = V_\mathrm{w}\sin\alpha_j, \qquad V_\mathrm{vert} = \pm V_\mathrm{w}\cos\alpha_j ,
  $$

  with the sign of $z_k$: the gas moves away from the centre, along the axis at the centre of a cylinder and
  at $\theta_\mathrm{o}/2$ from it at the edge (with DENSTYPE 2, at $\theta_\mathrm{o}/2$ everywhere). With
  the rotation $V_\mathrm{rot}$, a cloud at azimuth $\theta$ in its cylinder has, as in GALMOD,

  $$
  V_\mathrm{los} = V_\mathrm{sys} + \left(V_\mathrm{rot}\cos\theta + V_\mathrm{rad}\sin\theta\right)\sin i
  + V_\mathrm{vert}\cos i .
  $$

$V_\mathrm{w}$, $\sigma$, $\Sigma$ and $V_\mathrm{rot}$ can be one value or one value per cylinder (a list or
a file column with exactly $N$ values); $\theta_\mathrm{o}$ is one value. $\Sigma$ is a flux per unit area, as
in GALMOD with {ref}`DENSFLUX <par-galmod-densflux>` true (the default): flux units of the cube times km/s
per arcsec² of the face-on cylinder; with DENSFLUX false it is a column density in $10^{20}$ cm⁻², as in
GALMOD. The clouds of each cylinder are those of GALMOD: Monte Carlo (sub-rings 0.75 pixels
wide, {ref}`CDENS <par-galwind-cdens>` clouds per pixel, at least 10, each split into
{ref}`NV <par-galwind-nv>` sub-clouds), or analytic with {ref}`GALMODTYPE <par-galwind-galmodtype>`
`ANALYTIC`. The $2N$ cylinders are built independently and summed; the cylinder $k$ of each cone has its own
seed, derived from {ref}`ISEED <par-galwind-iseed>`, $k$ and the cone, so the model does not depend on the
number of threads (apart from rounding).

## Spherical shells (WTYPE 1)

The cones are made of $N$ spherical shells centred on the apex, at the radii $r_s = (s - \tfrac12)\,H/N$
($s = 1, \dots, N$) or at the radii given with RADII (a list or a file column; $N$ is then their number). Each
shell has its own $V_\mathrm{rot}$, $V_\mathrm{w}$, $\sigma$, $\Sigma$ and $\theta_\mathrm{o}$ (one value for
all, or one per shell). As in GALMOD, the shells are divided into sub-shells of width $\delta = 0.75$ pixels,
from $r_1 + \delta/2$ to $r_N$, whose values are interpolated linearly between the shells.

A sub-shell of radius $r$ has

$$
n_\mathrm{c} = \operatorname{round}\left(C\,\Sigma\,\frac{2\pi r\,\delta}{A_\mathrm{pix}}\right)
$$

clouds, where $C$ is CDENS and $A_\mathrm{pix}$ the area of a pixel (no minimum: a sub-shell with no clouds is
empty, with a warning). Each cloud is placed at

- a distance $\rho$ from the apex drawn uniformly in area between $r - \delta/2$ and $r + \delta/2$;
- an azimuth $\varphi$ around the axis, uniform in $[0, 2\pi)$;
- a polar angle $\vartheta$ from the axis, uniform in $[0, \theta_\mathrm{o}/2]$, in the cone $z > 0$ or, for
  every other cloud, in the opposite one ($\vartheta \to \pi - \vartheta$). The clouds are uniform in angle,
  not in solid angle: their density per unit solid angle grows towards the axis as $1/\sin\vartheta$.

Its position in the frame of the disk is $(x, y, z) = (\rho\sin\vartheta\cos\varphi,\ \rho\sin\vartheta
\sin\varphi,\ \rho\cos\vartheta)$, projected as above. The gas moves radially from the apex at $V_\mathrm{w}$
and rotates around the axis at $V_\mathrm{rot}$:

$$
V_\mathrm{los} = V_\mathrm{sys}
+ V_\mathrm{w}\left(\cos\vartheta\cos i + \sin\vartheta\sin\varphi\sin i\right)
+ V_\mathrm{rot}\cos\varphi\sin i .
$$

Each cloud is split into NV sub-clouds (20 if NV is not given; one where $\Sigma = 0$) with velocities drawn
from a Gaussian of dispersion $\sqrt{\sigma^2 + \sigma_\mathrm{instr}^2}$, and each sub-cloud adds its flux to
the channel of its velocity. The flux of a sub-shell is that of a face-on annulus of radius $r$, width
$\delta$ and column density $\Sigma$: here $\Sigma$ is a column density in $10^{20}$ atoms cm$^{-2}$,
converted to flux as in GALMOD with DENSFLUX false (see {doc}`../galmod/algorithm`, *Units of the model*).
This flux does not depend on $\theta_\mathrm{o}$. The spherical model is always Monte Carlo (GALMODTYPE is
ignored) and does not use DENSTYPE; the sub-shells are split into THREADS chunks, each with its own seed, so
the model depends on the number of threads.

## Convolution and outputs

With {ref}`SM <par-galwind-sm>` (the default), every channel of the model is convolved with the beam of the
cube (BMAJ, BMIN, BPA), a 2D Gaussian normalised to a unit sum. The shells, and the cylinders with DENSFLUX
false, are a flux in Jy per pixel: they are then multiplied by the area of the beam in pixels,
$\Omega_\mathrm{b}/(\Delta x\,\Delta y) \approx 1.1331\,b_\mathrm{maj} b_\mathrm{min}/|\Delta x\,\Delta y|$
(exactly, the inverse of the peak of the normalised kernel), which turns a flux per pixel into a flux per
beam. The cylinders with DENSFLUX true are already in the units of the cube, and keep their total flux, as
the models of GALMOD. Values below $10^{-12}$ are set to zero.

The outputs, in the output folder:

| File | Content |
|---|---|
| `<prefix>_wind.fits` | the model cube, on the grid of the input cube |
| `<prefix>_wind_0mom.fits`, `_1mom.fits`, `_2mom.fits` | the moment maps 0, 1 and 2 of the model (no mask) |
| `<prefix>_wind_pv.fits` | slice through $(x_0, y_0)$ at PA $\phi - 90^\circ$, along the cone axis |
| `<prefix>_wind.pdf`, `_wind_maps.pdf` | channel maps; moment maps and slice ({ref}`PLOTS <par-all-plots>`) |

The plots are made by the script `pyscript_w.py` in the output folder.

In Python, the result of {class}`bbarolo.GalWind` holds the cube (`model`), the moment maps (`maps`) and the
slice (`pv`) in memory; `write()` writes the files and the plots.

## Differences from BBarolo 1.8

- **Direction of the cylinders**: in BBarolo 1.8 the cylinders (WTYPE 0) moved towards the plane for
  $V_\mathrm{w} > 0$ (radial velocity $-V_\mathrm{w}\sin\alpha$, vertical velocity towards the plane), an
  inflow, while the shells were an outflow. $V_\mathrm{w} > 0$ is an outflow in both now. The cylinders are
  also centred on the projected axis for cubes with rotated or mirrored axes.
- **Parameters read from the wrong place**: the shells took their densities from VWIND and their rotation
  from OPENANG; a file column of VROT for the cylinders was read from the DENS column. With the same
  parameters, the flux of the shells differs from BBarolo 1.8 by the factor DENS/VWIND.
- **Random numbers**: in BBarolo 1.8 all the cylinders and cones shared the same random numbers.
- **Lists**: VWIND, VDISP, DENS, VROT (and OPENANG and RADII for the shells) given as lists took only their
  first value; they give one value per cylinder or shell now. OPENANG as a file column with cylinders is an
  error (BBarolo 1.8 used 0).
- **NV** 1 is one sub-cloud for the shells (BBarolo 1.8 took it as 20); CDENS and NV given to the Python class
  are used.
- **Several lines** were shifted from the first by fixed velocity offsets, exact only at velocity 0; the
  lines of a cloud now share its Doppler factor (see {doc}`../galmod/algorithm`).
- **Flux of the convolved cylinders**: BBarolo 1.8 multiplied them by the area of the beam in pixels, as the
  shells, although their $\Sigma$ is a flux per unit area in the units of the cube: they were about 10 times
  too bright for a beam of 3 pixels.
- **Layers of the cylinders**: BBarolo 1.8 used the vertical profile of GALMOD's LTYPE (a Gaussian by
  default; GALWIND's own LTYPE 5 was never used), so about a third of the flux of each cylinder fell outside
  it, into its neighbours, beyond $H$ and into the other cone. They are boxes now.
- **ANALYTIC** cylinders (GALMODTYPE) are new.

## References

- Di Teodoro E. M., Fraternali F., 2015, MNRAS, 451, 3021: BBarolo and its disk models.
  [ADS](https://ui.adsabs.harvard.edu/abs/2015MNRAS.451.3021D)

## In the code

- `bbarolo/galwind.py`: the class `GalWind`: `compute`, `_cylindrical` (cylinders), `_shells` and `_spherical`
  (shells), `smooth`, `writeFITS`, `writeMomentMaps`, `writePV`, `makePlots`.
- `bbarolo/galmod.py`: `GalmodSetup._build` (the GALMOD disks of the cylinders) and the numba kernel
  `_galmod_wind` (the clouds of the shells).
- `bbarolo/smooth.py`: `Smooth3D` (the convolution).
- `bbarolo/tasks.py`: the task (`_compute_galwind`, `_write_galwind`) and the Python class `GalWind`.
