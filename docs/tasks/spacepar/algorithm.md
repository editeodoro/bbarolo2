# SPACEPAR: how it works

SPACEPAR computes the residual function of {doc}`../3dfit/algorithm` on a regular grid of two ring
parameters, ring by ring, with all the other parameters fixed. It does not fit: it maps the function that
3DFIT minimises, so that one can see the shape of its minimum, the degeneracies between two parameters
(e.g. $V_\mathrm{rot}$ and $i$), secondary minima and whether the minimum found by 3DFIT is the right one.
The inputs are the cube, the mask and the rings, as for 3DFIT, and the two parameters with their ranges. The
output is a cube with, for each ring, the relative increase of the residual over the grid, and its plot.

## Rings, mask and fixed parameters

SPACEPAR sets up the fit as 3DFIT does, with the same parameters:

- the mask ({ref}`MASK <par-spacepar-mask>`), the noise of the cube and of the channels;
- the rings and their initial values: values given, {ref}`RINGFILE <par-spacepar-ringfile>`, estimates from
  the data ({ref}`GUESSTYPE <par-spacepar-guesstype>`) and defaults (see {doc}`../3dfit/algorithm`,
  *Initial parameters*). Unlike 3DFIT, SPACEPAR does not refine the initial $\phi$ and $V_\mathrm{rot}$ with a
  2D fit of the velocity field: the rings are those given or estimated.

The two parameters of the grid, $P_1$ ({ref}`P1 <par-spacepar-p1>`) and $P_2$ ({ref}`P2 <par-spacepar-p2>`),
are two different names among `VROT`, `VDISP`, `Z0`, `INC`, `PA`, `XPOS`, `YPOS`, `VSYS` and `VRAD`. All the
other ring parameters of ring $k$, including DENS, VVERT, DVDZ and ZCYL, keep their initial values in ring
$k$. FREE has no effect.

## The grid

{ref}`P1PAR <par-spacepar-p1par>` gives the minimum $a_1$, the maximum $b_1$ and the step $\delta_1$ of $P_1$,
in the units of the parameter (km/s; degrees for INC and PA; arcsec for Z0; pixels for XPOS and YPOS). The
grid has

$$
n_1 = \left\lfloor \frac{|b_1 - a_1|}{\delta_1} + 1 \right\rfloor, \qquad
P_{1,i} = a_1 + i\,\delta_1, \quad i = 0, \dots, n_1 - 1 ,
$$

values, and the same for $P_2$ with {ref}`P2PAR <par-spacepar-p2par>` ($n_2$ values $P_{2,j}$). The grid
values are used as they are: the bounds of 3DFIT (DELTAVROT, DELTAINC, ...) do not apply.

## The residual at a grid point

For each ring $k$, from the ring of index {ref}`STARTRAD <par-spacepar-startrad>` (counted from 0) outwards,
and for each grid point $(P_{1,i}, P_{2,j})$, SPACEPAR builds the model of ring $k$ alone with
$P_1 = P_{1,i}$, $P_2 = P_{2,j}$ and the fixed values of the other parameters, and computes the residual
$f_k(P_{1,i}, P_{2,j})$. Model and residual are exactly those of a 3DFIT evaluation, described in
{doc}`../3dfit/algorithm`:

- the ring is the annulus $R_k - w_k^- \le R \le R_k + w_k^+$, reaching halfway to the neighbouring radii;
- the model is built with GALMOD ({ref}`GALMODTYPE <par-spacepar-galmodtype>` `MC` or `ANALYTIC`), in a box
  around the ring, convolved with the beam ({ref}`SM <par-spacepar-sm>`), and normalised with the *first*
  value of {ref}`NORM <par-spacepar-norm>`, the normalisation of the first stage of 3DFIT;
- the residual is

  $$
  f = \left(1 + \frac{N_b}{N}\right)^{\beta}\,\frac{1}{N - N_b}
  \sum_{(x,y)\in A} w(\theta_{xy}) \sum_{z \in V_{xy}} \frac{\rho(m_{xyz}, o_{xyz})}{\sigma_z},
  $$

  with the function $\rho$ of {ref}`FTYPE <par-spacepar-ftype>`, the weights $w(\theta)$ of
  {ref}`WFUNC <par-spacepar-wfunc>`, the penalty for blanks $\beta$ of
  {ref}`BWEIGHT <par-spacepar-bweight>`, the pixels $A$ of the ring (and of the half chosen with
  {ref}`SIDE <par-spacepar-side>`) and the voxels $V_{xy}$ in the mask or with model emission (the symbols are
  defined in {doc}`../3dfit/algorithm`, *The residual function*);
- with {ref}`NORMALCUBE <par-spacepar-normalcube>`, the data and their noise are multiplied by
  $10/D_\mathrm{max}$ during the computation, as in 3DFIT.

Monte Carlo models are all built with the same seed ({ref}`ISEED <par-spacepar-iseed>`), so the differences
between neighbouring grid points come from the parameters, not from the random clouds; the clouds that fall
outside the box still draw their random deviates, so the model in the box is a crop of the model of the whole
cube and does not change realisation when clouds cross the edge of the box. The map is still
granular at the level of the Monte Carlo noise; with GALMODTYPE `ANALYTIC` it is smooth.

The grid points of a ring are computed in parallel on {ref}`THREADS <par-all-threads>` threads. The cost is
$n_1 n_2$ model evaluations per ring (the example of NGC 2403, a 76 × 31
grid on 10 rings, about 24 000 models, takes about 20 s on 2 threads).

## The output map

For each ring the minimum over the grid, $f_{\min,k} = \min_{i,j} f_k(P_{1,i}, P_{2,j})$, and its position are
printed. The output is the relative increase of the residual over that minimum,

$$
\varepsilon_k(P_{1,i}, P_{2,j}) = \frac{\left|f_k(P_{1,i}, P_{2,j}) - f_{\min,k}\right|}{f_{\min,k}} ,
$$

dimensionless, 0 at the minimum. It is written to `<prefix>_spacepar.fits`, a cube with the axes $P_1$
(`CTYPE1` the name of P1, `CRVAL1` $= a_1$, `CDELT1` $= \delta_1$), $P_2$ (likewise) and the ring
(`CTYPE3 = RING_RAD`, in arcsec), with `BTYPE = FMIN_<FTYPE>`. The cube has one plane per ring; the planes of
the rings inside STARTRAD are NaN.

:::{note}
The third axis has, as in BBarolo 1.8, `CRVAL3` equal to the radius of the ring STARTRAD and `CDELT3` equal
to the mean separation of the rings, while its first plane is the first ring: with STARTRAD > 0, or with
unevenly spaced radii, the radii given by the header (and the titles of the plot) are not those of the
planes. BUNIT is that of the cube, although $\varepsilon$ is dimensionless.
:::

$\varepsilon$ connects to the errors of 3DFIT ({ref}`FLAGERRORS <par-3dfit-flagerrors>`), which are the
changes of one parameter that raise the residual by 5%: the contour $\varepsilon = 0.05$ is the
two-parameter version of that criterion, with the other parameters fixed instead of at their best values.
As for those errors, $\varepsilon$ is not a statistical confidence level.

**The plot.** The script `plotscripts/spacepar.py` in the output folder draws `<prefix>_spacepar.pdf`, with
{ref}`PLOTS <par-all-plots>`: one panel per ring, four per row, with $100\,\varepsilon$ (percent) in colour,
$P_1$ on the horizontal axis and $P_2$ on the vertical one, contours at 2, 5, 10 and 30% (drawn on the map
smoothed with a Gaussian of one grid cell), and a cross at the minimum. In Python, the result of
{class}`bbarolo.SpacePar` holds the cube (`space`) and the engine (`spacepar`), and `write()` writes the files
and the plot.

:::{note}
The map of a ring holds only that ring, with the other parameters fixed: the minimum of SPACEPAR is that of
the first stage of 3DFIT when the free parameters of 3DFIT are exactly $P_1$ and $P_2$ and 3DFIT starts from
the same rings. A different minimum in 3DFIT can come from its other free parameters, from its initial 2D fit
of $\phi$ and $V_\mathrm{rot}$, or from its bounds (DELTAVROT, DELTAINC, DELTAPA).
:::

## Differences from BBarolo 1.8

- **Normalisation**: SPACEPAR uses the first value of NORM (AZIM by default), as the first stage of 3DFIT;
  BBarolo 1.8 always normalised the models LOCAL, so its maps were not those of the function 3DFIT minimises
  with the default NORM.
- **Rings**: BBarolo 1.8 replaced the initial $\phi$ and $V_\mathrm{rot}$ of the rings (with PA in FREE and at
  least 5 rings) with those of a 2D fit of the velocity field, also when they were given; SPACEPAR keeps them.
- **FTYPE and WFUNC** out of range are replaced by 2 and 1, as in 3DFIT (BBarolo 1.8 gave zero residuals).
- **SIDE** accepts the long forms (`APPROACHING`, ...), which BBarolo 1.8 ignored.
- **NORMALCUBE**: the noise of the cube is restored correctly after the computation.
- **Monte Carlo models in the box** are crops of the model of the whole cube (in BBarolo 1.8 the clouds
  outside the box drew no velocity deviates, so the rest of the ring was another realisation, and the
  residuals jumped by the Monte Carlo noise when clouds crossed the edge of the box).
- **ANALYTIC models** (GALMODTYPE) are new; the default Z0 is one pixel, as in 3DFIT.
- **Outputs**: `<prefix>_spacepar.fits` and `<prefix>_spacepar.pdf` (BBarolo 1.8: `spacepar_<name>.fits`);
  the axis unit of XPOS and YPOS is `pix` (BBarolo 1.8 wrote `km/s`).

## References

- Di Teodoro E. M., Fraternali F., 2015, MNRAS, 451, 3021: the 3D fitting algorithm of BBarolo and its
  residual function. [ADS](https://ui.adsabs.harvard.edu/abs/2015MNRAS.451.3021D)

## In the code

- `bbarolo/spacepar.py`: the class `Spacepar` (a subclass of `Galfit`): `calculate` (the grid and the
  residuals), `_dring` (the ring of a grid point), `setOutSpacepar` (the output header), `plotAll_Python`
  (the cube of $\varepsilon$ and the plot script).
- `bbarolo/galfit.py`: `Galfit._init_from_cube` and `setup` (rings, mask, normalisation), `getFuncValue` and
  the numba kernel `_norm` (the residual).
- `bbarolo/tasks.py`: the task (`_compute_spacepar`, `_write_spacepar`) and the Python class `SpacePar`.
