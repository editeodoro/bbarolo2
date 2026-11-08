# 2DFIT: how it works

2DFIT fits a tilted-ring model to a velocity field, ring by ring, with weighted least squares: the classic 2D
method of GIPSY's ROTCUR (Begeman 1987, 1989). The input is a datacube, whose velocity field is its first
moment within the mask, or a 2D velocity field. Each ring has its own centre, systemic velocity, rotation
velocity, radial (expansion) velocity, position angle and inclination; the free ones are those that best
reproduce the line-of-sight velocities of the pixels of the ring. The outputs are a table of the fitted rings
and a model velocity field. The method is fast, but a velocity field carries no information on the beam:
2DFIT does not correct for beam smearing, and at low spatial resolution its rotation velocities are biased
low (inner rings) and its inclinations uncertain. {doc}`../3dfit/algorithm` fits the cube instead.

The task runs in this order:

1. the velocity field is computed (or read);
2. the initial rings are set: values given, estimates from the data;
3. every ring is fitted, independently of the others;
4. the table and the model velocity field are written.

## The velocity field

If the input is a cube with more than one channel, the velocity field is its first moment within the mask
$M_{xyz}$ ({ref}`MASK <par-2dfit-mask>`, by default all the sources of the source finder):

$$
V(x, y) = \frac{\sum_z M_{xyz} D_{xyz}\, v_z}{\sum_z M_{xyz} D_{xyz}} ,
$$

where $D_{xyz}$ is the data and $v_z$ the velocity of channel $z$ (km/s). Pixels with
$\sum_z M_{xyz} D_{xyz} = 0$ are blank (NaN). The field is written to `<prefix>_map_1st.fits`. If the input is
a 2D image (or a cube with one channel), the image is the velocity field, in km/s. Blank pixels never enter
the fit.

## Rings and initial values

The rings come from {ref}`RADII <par-2dfit-radii>`, or from {ref}`NRADII <par-2dfit-nradii>` and
{ref}`RADSEP <par-2dfit-radsep>` $= \Delta R$ (radii $R_k = (k - \tfrac12)\Delta R$). The radii (arcsec) are
converted to pixels with the pixel size of the header. With $N$ radii, 2DFIT fits $N - 1$ rings: ring $k$
($k = 1, \dots, N-1$) is the annulus

$$
r_\mathrm{in} = \max\left(R_k - \tfrac12 w_k,\ 0\right) < r < r_\mathrm{out} = R_k + \tfrac12 w_k ,
\qquad w_k = R_{k+1} - R_k ,
$$

in pixels, centred on $R_k$. For evenly spaced radii the annuli tile the disk; otherwise they may overlap or
leave gaps. These are not the annuli of 3DFIT, which reach halfway to both neighbouring radii.

Each ring needs initial values of the centre $(x_0, y_0)$ ({ref}`XPOS <par-2dfit-xpos>`,
{ref}`YPOS <par-2dfit-ypos>`, pixels), the systemic velocity $V_\mathrm{sys}$
({ref}`VSYS <par-2dfit-vsys>`), the rotation velocity $V_\mathrm{rot}$ ({ref}`VROT <par-2dfit-vrot>`), the
expansion velocity $V_\mathrm{exp}$ ({ref}`VRAD <par-2dfit-vrad>`, 0 if not given), all in km/s, the position
angle $\phi$ ({ref}`PA <par-2dfit-pa>`) and the inclination $i$ ({ref}`INC <par-2dfit-inc>`), in degrees. The
values given are used (one value for all the rings or one per ring; the centre is one value, that of the
first ring when XPOS and YPOS are given per ring). If any of
the rings, the centre, $V_\mathrm{sys}$, $V_\mathrm{rot}$, $\phi$ or $i$ is not given, those not given are
estimated from the cube as in 3DFIT ({ref}`GUESSTYPE <par-2dfit-guesstype>`; see
{doc}`../3dfit/algorithm`, *Estimates from the data*). An estimated value is the same in all the rings.

## The model of a ring

A pixel $(x, y)$ (pixels, counted from 0) has, in the plane of a ring with centre $(x_0, y_0)$, position angle
$\phi$ and inclination $i$, the coordinates

$$
x_r = -(x - x_0)\sin\phi + (y - y_0)\cos\phi, \qquad
y_r = \frac{-(x - x_0)\cos\phi - (y - y_0)\sin\phi}{\cos i},
$$

the radius $r = \sqrt{x_r^2 + y_r^2}$ (pixels) and the azimuth $\theta = \operatorname{atan2}(y_r, x_r)$
(degrees; $\theta = 0$ on the receding half of the major axis, $\theta = 0$ for $r < 0.1$ pixels). The model
line-of-sight velocity at that pixel is that of gas in circular and radial motion in a flat ring:

$$
V_\mathrm{mod}(x, y) = V_\mathrm{sys}
+ \left(V_\mathrm{rot}\cos\theta + V_\mathrm{exp}\sin\theta\right)\sin i .
$$

There is no thickness, no velocity dispersion and no beam.

## The pixels and their weights

The pixels of ring $k$ are those of the velocity field (within its bounds and not blank) that satisfy, for
the current values of the geometry,

- $r_\mathrm{in} < r < r_\mathrm{out}$ (the annulus above);
- $|\cos\theta| > \sin\theta_\mathrm{f}$, i.e. they lie more than $\theta_\mathrm{f}$ from the minor axis, in
  the plane of the ring, where $\theta_\mathrm{f}$ is {ref}`FREEANGLE <par-2dfit-freeangle>` (15° by default;
  a value outside $0 \le \theta_\mathrm{f} < 90^\circ$ is an error). Near the minor axis
  $\cos\theta \approx 0$: the line-of-sight velocity hardly depends on $V_\mathrm{rot}$, and small errors of
  $\phi$ give large errors of $V_\mathrm{rot}$;
- with {ref}`SIDE <par-2dfit-side>` `R`, $|\theta| \le 90^\circ$ (the receding half); with `A`,
  $|\theta| \ge 90^\circ$ (the approaching half); with `B`, both.

The geometry of the selection is computed in single precision, as in BBarolo 1.8, so that the pixels on the
edges of the annuli are the same. Each pixel has the weight set by {ref}`WFUNC <par-2dfit-wfunc>` $= n$:

$$
w(\theta) = \begin{cases} |\cos\theta|^{n} & n \ge 0,\\ |\sin\theta|^{|n|} & n < 0, \end{cases}
$$

uniform for $n = 0$ and $\cos^2\theta$ by default ($n = 2$), which favours the pixels near the major axis.
The weights are not divided by any error of the velocities: the velocity field has none.

## The fit of a ring

### The function minimised

With $\mathbf{p}$ the vector of the seven ring parameters, of which $n_\mathrm{f}$ are free
({ref}`FREE <par-2dfit-free>`), and the $N_k$ pixels $j$ of the ring with velocity $V_j$, weight $w_j$ and
model $V_{\mathrm{mod},j}(\mathbf{p})$, the fit minimises

$$
\chi^2(\mathbf{p}) = \sum_{j=1}^{N_k} w_j \left[V_j - V_{\mathrm{mod},j}(\mathbf{p})\right]^2
\quad \text{(km/s)}^2 ,
$$

and measures the quality of a solution by the weighted rms residual

$$
q = \sqrt{\frac{\chi^2}{N_k - n_\mathrm{f}}} \quad \text{km/s}
$$

($q$ is infinite if $N_k \le n_\mathrm{f}$). FREE can include `VROT`, `INC`, `PA`, `XPOS`, `YPOS`, `VSYS`,
`VRAD` (or `VEXP`) and `ALL` (all seven); the other parameters stay at their initial values. With SIDE `A` or
`R`, $V_\mathrm{sys}$, $x_0$ and $y_0$ are always fixed, because one half of the galaxy does not constrain
them. A free centre is fitted in each ring.

### Two nested iterations

The set of pixels of a ring depends on its geometry ($x_0$, $y_0$, $\phi$, $i$), so $\chi^2$ is not a smooth
function of the parameters. As in ROTCUR, the fit therefore alternates two steps:

1. **Least squares on fixed pixels.** For the pixels selected with the current parameters $\mathbf{p}$,
   the free parameters that minimise $\chi^2$ are found by non-linear least squares, $\hat{\mathbf{p}}$.
   The minimiser is the trust-region reflective method of SciPy (Branch et al. 1999), with the analytic
   derivatives of $V_\mathrm{mod}$ with respect to the seven parameters, a relative tolerance on $\chi^2$ of
   $10^{-3}$ and at most $100\,(n_\mathrm{f} + 1)$ evaluations.
2. **Line search with new pixels.** With the step $\Delta\mathbf{p} = \hat{\mathbf{p}} - \mathbf{p}$, the
   trial points $\mathbf{p} + \lambda\,\Delta\mathbf{p}$ are tried with
   $\lambda = 1, -1, \tfrac12, -\tfrac12, \tfrac14, -\tfrac14, \dots$; for each, the pixels are selected again
   with the trial geometry and $q$ is computed. An inclination above 90° becomes $i - 180^\circ$. The first
   trial point with a $q$ lower than that of $\mathbf{p}$ is accepted, and step 1 starts again from it.

Let $q_0$ be the residual of the current point. The search stops, and the current point is the result, when
a trial point does not lower $q$ and

- $|q - q_0|/q_0 < 10^{-3}$, during the first 50 iterations;
- after 50 iterations, every component of $\lambda\,\Delta\mathbf{p}$ is below 0.1 (in the units of the
  parameters: km/s, degrees, pixels).

The tolerance $10^{-3}$ is fixed. If the search has not stopped after 100 iterations, or the least squares
fail, the ring is not fitted (see below).

### Errors

The errors come from the covariance matrix of the last least-squares step. With $\mathsf{J}$ the matrix of
the derivatives $\partial V_{\mathrm{mod},j}/\partial p_l$ of the free parameters at $\hat{\mathbf{p}}$ and
$\mathsf{W} = \mathrm{diag}(w_j)$,

$$
\sigma_l = \sqrt{\left[\left(\mathsf{J}^\mathsf{T}\mathsf{W}\mathsf{J}\right)^{-1}\right]_{ll}\,
\frac{\chi^2}{N_k - n_\mathrm{f}}} ,
$$

where $N_k$ counts only the pixels with $w_j > 0$. This is the formal error of a least-squares fit whose
data errors are unknown and estimated from the scatter of the residuals, as in GIPSY's `lsqfit`. It assumes
independent pixels: the pixels of a velocity field are correlated over a beam, so these errors are too small
by about the square root of the number of pixels per beam. Errors above 999.99 are set to 999.99.

### Failed rings

A ring is not fitted when there are no free parameters, when it has no more pixels than free parameters, when
the iterations do not converge, or when the matrix $\mathsf{J}^\mathsf{T}\mathsf{W}\mathsf{J}$ is singular or
has a non-positive diagonal; a message gives the reason. Its values are then NaN, and the ring is left out
of the model velocity field. The rings are independent:
each starts from its own initial values, and they are fitted in parallel on {ref}`THREADS <par-all-threads>`
threads.

## Outputs

| File | Content |
|---|---|
| `<prefix>_map_1st.fits` | the velocity field that was fitted (from a cube) |
| `<prefix>_2dtrm.txt` | the rings: radius (pix, arcsec), VSYS, VROT, VEXP (km/s), PA, INC (°), centre (pix) |
| `<prefix>_2d_mod.fits` | the model velocity field |

The table has no errors. In Python, the result of {class}`bbarolo.FitMod2D` holds the table (`rings`, at full
precision, with NaN in the rings whose fit failed), the model (`model`), the velocity field (`velfield`) and the
fitting engine (`ringmodel`), whose errors are `vsyse`, `vrote`, `vexpe`, `posae`, `incle`, `xpose` and
`ypose`; `write()` writes the files.

**The model velocity field.** For each ring, $V_\mathrm{mod}$ with its fitted values is computed at the pixels
of its annulus, with the selection above but without the free angle ($\theta_\mathrm{f} = 0$); SIDE still
applies, and only pixels where the velocity field is defined are filled. A pixel that belongs to two
rings (the fitted geometries can make the annuli overlap) takes the mean of the two values (for more rings,
each new value is averaged with the current one). Then the empty pixels within the outer edge of the last
ring fitted, with its geometry, take the model of the ring whose radius, computed with that same geometry,
is closest to their own radius.

## Differences from BBarolo 1.8

- **The least squares** use SciPy's trust-region reflective method instead of the Levenberg–Marquardt
  routine of GIPSY (`lsqfit`) ported in BBarolo 1.8. The outer iterations, the selection of the pixels and the
  errors are the same; the table of NGC 2403 is identical to that of BBarolo 1.8.
- **FREEANGLE** is new: BBarolo 1.8 read the free angle from DVDZ, so the DVDZ of a lagging disk changed it.
  DVDZ is no longer read by 2DFIT.
- **SIDE**: the long forms (`APPROACHING`, `RECEDING`, `APP`, ...) are accepted; BBarolo 1.8 fitted both sides
  with them.
- **RADII with an estimated geometry**: BBarolo 1.8 cut RADII to the estimated number of rings.
- **Parameters seen by 2DFIT after 3DFIT** in the same run: 3DFIT no longer writes its estimated and default
  values into the parameters, so 2DFIT sees the parameters as given.
- **Output names** start with `<prefix>_` (e.g. `<prefix>_map_1st.fits`).
- **The model velocity field**: the radius of the pixels between the rings is computed with the geometry of
  the last ring; BBarolo 1.8 used that of the ring of the previous pixel filled, so with fitted PA, INC or
  centre some pixels took the velocity of another ring (302 of 6964 pixels of NGC 2403, by up to 9 km/s).
  A ring that was not fitted is left out (the port stopped with an error; 1.8 left it out too, and did not
  fill the empty pixels when the last ring was not fitted).

## References

- Begeman K. G., 1987, PhD thesis, University of Groningen: HI rotation curves of spiral galaxies, and the
  ROTCUR method.
- Begeman K. G., 1989, A&A, 223, 47: tilted-ring fits of velocity fields.
  [ADS](https://ui.adsabs.harvard.edu/abs/1989A&A...223...47B)
- Branch M. A., Coleman T. F., Li Y., 1999, SIAM Journal on Scientific Computing, 21, 1: the trust-region
  reflective method.
- Rogstad D. H., Lockhart I. A., Wright M. C. H., 1974, ApJ, 193, 309: the tilted-ring model.
  [ADS](https://ui.adsabs.harvard.edu/abs/1974ApJ...193..309R)
- van der Hulst J. M., Terlouw J. P., Begeman K. G., Zwitser W., Roelfsema P. R., 1992, ASP Conf. Ser. 25,
  131: GIPSY. [ADS](https://ui.adsabs.harvard.edu/abs/1992ASPC...25..131V)
- Virtanen P. et al., 2020, Nature Methods, 17, 261: SciPy.
  [ADS](https://ui.adsabs.harvard.edu/abs/2020NatMe..17..261V)

## In the code

- `bbarolo/ringmodel.py`: the class `Ringmodel`: `_from_cube` and `setfromCube` (initial rings, velocity
  field), `getdat` (selection, weights, $q$), `rotfit` (outer iterations), `ringfit` (all the rings),
  `printfinal` (table), `writeModel` (model field); the functions `vfunc` (model), `vderv` (derivatives) and
  `lsqfit` (least squares and errors).
- `bbarolo/moments.py`: `MomentMap` (the velocity field).
- `bbarolo/tasks.py`: the task (`_compute_2dfit`, `_write_2dfit`) and the Python class `FitMod2D`.
