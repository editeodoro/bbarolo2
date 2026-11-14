# ELLPROF: how it works

ELLPROF measures the radial profile of a map in elliptical rings: the projections on the sky of
concentric circular rings in an inclined disk. For each ring it gives the statistics of the map values
in the ring (sum, mean, median, scatter), the surface density on the sky and its face-on value, and,
for HI maps in Jy, the HI column density and mass surface density. The algorithm is that of GIPSY's
ELLINT. The input is a 2D map, or a cube, of which the total intensity map is used, and the rings;
the outputs are a table, `<prefix>_ellprof.txt`, and the map that was used, `<prefix>_ellprof_map.fits`.
With a beam, ELLPROF also gives the profile deconvolved from the beam, `<prefix>_ellprof_deconvolved.txt`
([below](#the-profile-deconvolved-from-the-beam)). The task is run by {ref}`ELLPROF <par-ellprof-ellprof>`.

## The map

- **A cube** (more than one channel): the map is its total intensity map within the mask, computed
  as the moment map of {doc}`../maps/algorithm` (always as a moment, not a Gaussian fit), in Jy/beam
  km/s with {ref}`FLUXCONVERT <par-all-fluxconvert>`. The mask is set by
  {ref}`MASK <par-ellprof-mask>` (with {ref}`BLANKCUT <par-ellprof-blankcut>` and
  {ref}`FACTOR <par-ellprof-factor>`), with all the detections of the source finder unless the type ends
  in `LARGEST`; a mask built by an earlier task of the run is reused. Pixels with no emission in the
  mask are blank (NaN).
- **A 2D map**: the map itself, in its own unit (BUNIT). Blank pixels of the file (NaN) are blank.

Below, $u$ is the unit of the map, $\Delta x$, $\Delta y$ (arcsec) the absolute pixel sizes and
$a_{\rm pix} = \Delta x\, \Delta y$ (arcsec²) the area of a pixel.

## The rings

The rings are centred on the radii $R_r$ (arcsec), $r = 1, \dots, N$, given by
{ref}`RADII <par-ellprof-radii>`, or by {ref}`NRADII <par-ellprof-nradii>` and
{ref}`RADSEP <par-ellprof-radsep>` ($R_r = (r - 1/2)\,$RADSEP). The width of ring $r$ is the distance
to the next radius, and to the previous one for the last ring:

$$
W_r = R_{r+1} - R_r \;\; (r < N), \qquad W_N = R_N - R_{N-1},
$$

and, for a single ring, RADSEP if given, otherwise $R_1/2$. Ring $r$ covers the radii

$$
R_r^{\rm in} = \max\left(R_r - \tfrac{1}{2} W_r,\; 0\right) \le R < R_r^{\rm out} = R_r + \tfrac{1}{2} W_r .
$$

With equally spaced radii the rings tile the disk; otherwise neighbouring rings can overlap or leave
gaps (see the overlap correction below).

Each ring has its inclination $i_r$ ({ref}`INC <par-ellprof-inc>`) and position angle $\phi_r$
({ref}`PA <par-ellprof-pa>`, degrees, anticlockwise from the $+y$ axis of the map, reduced to
$0$–$360°$). An inclination with $|\cos i_r| \le 10^{-4}$ (edge-on) is replaced by $89°$. The centre
$(x_{\rm c}, y_{\rm c})$ (pixels) is the same for all the rings: the mean of
{ref}`XPOS <par-ellprof-xpos>` and of {ref}`YPOS <par-ellprof-ypos>` over the rings. It must lie inside
the map. The centre, INC and PA are required: they are not estimated.

## Pixels in a ring

Each pixel is divided into $2 \times 2$ sub-pixels, at a quarter of a pixel from its centre. The
sub-pixel of pixel $(i, l)$ at the offset $(\delta_x, \delta_y) \in \{\pm 1/4\}^2$ has the sky
coordinates, relative to the centre,

$$
x' = \Delta x\,(i - x_{\rm c} + \delta_x), \qquad y' = \Delta y\,(l - y_{\rm c} + \delta_y) \qquad \text{(arcsec)},
$$

with $x'$ increasing with the pixel index $i$. In the frame of ring $r$, its coordinates along the
major and the minor axis, and its radius in the plane of the disk, are

$$
X = -x' \sin\phi_r + y' \cos\phi_r, \qquad
Y = -\frac{x' \cos\phi_r + y' \sin\phi_r}{\cos i_r}, \qquad
R = \sqrt{X^2 + Y^2} .
$$

$X > 0$ is the half of the major axis at the position angle $\phi_r$, which is the receding half when
PA follows the convention of BBarolo. The sub-pixel belongs to ring $r$ if
$R_r^{\rm in} \le R < R_r^{\rm out}$. Each sub-pixel carries the value $v$ of its pixel. The value of
a sub-pixel that falls in $n_{\rm ov} > 1$ rings is divided by $n_{\rm ov}$, so that each pixel is counted
once in total (the overlap correction). With the sub-pixels, a pixel crossed by the edge of a ring
contributes in proportion (to a quarter of a pixel) to the rings on the two sides.

## Sides

{ref}`SIDE <par-ellprof-side>` `A` or `R` restricts the profile to one half of the map, divided by the
minor axis; `B` (default) uses all of it. The half of a sub-pixel is set by its position angle on the
sky relative to the major axis of the ring,

$$
\theta = \operatorname{atan2}(y', x') + 270° - \phi_r \pmod{360°},
$$

which is 0 along $X > 0$ and increases in the same sense as PA. The two halves are
$\theta \in [270°, 360°) \cup [0°, 90°)$, the half with $X > 0$, and $\theta \in [90°, 270°)$, the
half with $X < 0$. The first contains the position angle PA of the rings, i.e. the receding half: SIDE
`R` selects it and SIDE `A` the second one, at any PA.

## Statistics in each ring

For ring $r$ (and the half selected by SIDE), let $v_q$, $q = 1, \dots, n$, be the values of its
sub-pixels that are not blank (after the overlap correction), and $n_{\rm b}$ the number of its blank
sub-pixels. With 4 sub-pixels per pixel, the table gives:

$$
\Sigma_r = \frac{1}{4} \sum_{q=1}^{n} v_q \quad (u), \qquad
A_r = \frac{n}{4}, \qquad
B_r = \frac{n_{\rm b}}{4} \quad \text{(pixels)},
$$

$$
\bar{v}_r = \frac{\Sigma_r}{A_r} = \frac{1}{n} \sum_q v_q, \qquad
s_r = \sqrt{\frac{\sum_q v_q^2 - n\, \bar{v}_r^2}{n - 1}}, \qquad
{\rm MAD}_r = {\rm median}_q\, |v_q - \tilde{v}_r| ,
$$

where $\Sigma_r$ is the sum of the map in the ring, $A_r$ the number of valid pixels and $B_r$ that of
blank pixels (both effective numbers, in quarters of a pixel), $\bar v_r$ the mean, $s_r$ the standard
deviation, $\tilde v_r$ the median and ${\rm MAD}_r$ the median absolute deviation from the median (not
scaled to a standard deviation). All are in $u$. The median, the standard deviation and the MAD are
those of the sub-pixel values, so a pixel counts four times. $s_r = 0$ for $n \le 1$; all the values
are 0 for an empty ring.

## Surface densities

The surface density on the sky is the sum over the area of the valid pixels,

$$
\Sigma^{\rm sky}_r = \frac{\Sigma_r}{A_r\, a_{\rm pix}} = \frac{\bar v_r}{a_{\rm pix}}
\quad (u\ \text{arcsec}^{-2}),
\qquad
\epsilon_r = \frac{s_r}{a_{\rm pix}},
$$

where $\epsilon_r$, the column `ERR_SD`, is the scatter of the surface density of single pixels in
the ring, not the error of its mean. The face-on surface density corrects the projected area of the
ring,

$$
\Sigma^{\rm fo}_r = \cos i_r \; \Sigma^{\rm sky}_r .
$$

The same quantities including the blank pixels, which then count as zero, are

$$
\Sigma^{\rm sky, bl}_r = \frac{\Sigma_r}{(A_r + B_r)\, a_{\rm pix}}, \qquad
\Sigma^{\rm fo, bl}_r = \cos i_r \; \Sigma^{\rm sky, bl}_r .
$$

For the total map of a cube, the blank pixels are those with no emission in the mask, so
$\Sigma^{\rm sky}_r$ is the mean surface density of the pixels with detected emission, and
$\Sigma^{\rm sky, bl}_r$ the mean over the whole ring.

### HI column density and mass surface density

When the unit of the map contains `Jy`, the table also gives the face-on HI mass surface density and
column density, assuming the map is the HI line integrated in velocity (Jy/beam km/s or Jy km/s). With
the relation $M_{\rm HI} = 2.356 \times 10^5\, D^2 \int S\, {\rm d}v$ (M☉, $D$ in Mpc, $S$ in Jy,
$v$ in km/s), the mass surface density is

$$
\Sigma_{\rm HI} = \frac{2.356 \times 10^{-7}}{\alpha^2} \; \frac{\Sigma^{\rm fo}_r}{\Omega_{\rm b}}
= 1.002 \times 10^4 \; \frac{\Sigma^{\rm fo}_r}{\Omega_{\rm b}}
\quad ({\rm M_\odot\,pc^{-2}}),
$$

with $\alpha = \pi/648000$ (one arcsec in radians) and $\Sigma^{\rm fo}_r$ in Jy km/s arcsec⁻²
(Jy/beam km/s arcsec⁻² for a map per beam). For a map per beam (a unit with `/b`), $\Omega_{\rm b}$ is
the beam area in pixels,

$$
\Omega_{\rm b} = \frac{\pi\, \theta_{\rm maj}\, \theta_{\rm min}}{4 \ln 2\; p^2},
$$

with $\theta_{\rm maj}$, $\theta_{\rm min}$ the FWHM of the beam and $p$ the mean pixel size; otherwise
$\Omega_{\rm b} = 1$. The column density is

$$
N_{\rm HI} = \frac{\Sigma_{\rm HI}}{0.0841186 \times 3.0857^2} = 1.2485\; \Sigma_{\rm HI}
\quad (10^{20}\ {\rm cm^{-2}}),
$$

the number of hydrogen atoms in 1 M☉ pc⁻². Helium is not included. A map in mJy or µJy is converted to Jy.
Maps in K have no HI columns. The constant is that of {ref}`MASSDENSMAP <par-maps-massdensmap>`.

## The profile deconvolved from the beam

The profile above is that of the observed map, which the beam has smoothed: for a disk a few beams across,
the inner rings come out too faint and the outer ones too bright. When the header has a beam, ELLPROF also
computes the face-on surface densities of the rings whose model, convolved with the beam, reproduces the
map. The method is that of 3DFIT with METHOD GLOBAL or BAYESIAN when DENS is not free ({doc}`../3dfit/likelihood`, *The
density profile when DENS is not free*), here at the fixed geometry of the rings. The map of the model is
linear in the densities $\Sigma_j$ of the rings,

$$
M(\mathbf x) = \sum_{j=1}^{N+1} \Sigma_j\, K_j(\mathbf x),
$$

where $K_j$ is the map of ring $j$ with unit density and the other rings empty, convolved with the beam.
The density is interpolated linearly between the radii (RINGMODE CENTRED), so $K_j$ is a "hat" that peaks
at $R_j$; the rings have the centre, INC and PA given, and the thickness Z0 (thin if not given). Ring
$N+1$ is an exterior annulus, from the outer edge of the rings to two beams beyond, which takes the emission
outside the rings; its density is not used. The densities are the non-negative least-squares solution
weighted by the noise $s(\mathbf x)$ of the map, with the smoothness term of 3DFIT with METHOD GLOBAL or BAYESIAN, of weight
$\alpha = 0.1$ times the mean data weight of a ring:

$$
\hat{\boldsymbol\Sigma} = \underset{\boldsymbol\Sigma \ge 0}{\arg\min}
\left[\sum_{\mathbf x}\left(\frac{D(\mathbf x) - M(\mathbf x)}{s(\mathbf x)}\right)^2
+ \alpha\,\frac{\lVert A\rVert^2}{\lVert L\rVert^2}\,\lVert L\boldsymbol\Sigma\rVert^2\right],
$$

with $A_{\mathbf xj} = K_j(\mathbf x)/s(\mathbf x)$ and $L$ the second differences of the profile on the
radii. At least $2N$ pixels are needed. The data map $D$, the maps $K_j$ and the noise depend on the input:

- **A cube**: $D(\mathbf x)$ is the sum of the data, not clipped, in the mask dilated by one beam FWHM and
  by ±3 channels, as the voxels of 3DFIT with METHOD GLOBAL or BAYESIAN: the faint emission around the mask, which the model
  reproduces, is included. $K_j$ is the channel sum of the ANALYTIC model of GALMOD, which does not
  depend on the kinematics. The noise is $s(\mathbf x) = \sigma_\mathrm n\sqrt{n(\mathbf x)\,L_z}$, with
  $\sigma_\mathrm n$ the scaled median absolute deviation of the data outside the mask, $n(\mathbf x)$ the
  number of channels summed and $L_z$ the spectral correlation of the noise (the conservative measure of
  3DFIT with METHOD GLOBAL or BAYESIAN, at least one channel). The pixels are those with at least one channel summed.
- **A 2D map** (e.g. a moment-0 map): $D$ is the map itself, without the blank pixels, with uniform
  weights. $K_j$ is the ring projected on the sky, thin or with the vertical profile
  {ref}`LTYPE <par-3dfit-ltype>` of scale height Z0, and convolved with the beam (a normalised kernel, so
  that a map per beam keeps its surface brightness); a uniform face-on disk of unit density gives
  $1/\cos i$ in each pixel. For the same rings it gives the profile of the cube (within 4% on simulated
  disks).

In both cases only the pixels within the outer edge of the exterior annulus are fitted: an ellipse with the
centre, PA and INC of the last ring, its minor axis widened by twice Z0 projected. The emission beyond, which
the model does not describe, pulled the solution (BBarolo 2.0.0: NGC 2403 with one ring 60" wide was 11 times
too low). With {ref}`SIDE <par-ellprof-side>` A or R only the pixels on that side of the minor axis (through
the median centre, at the median position angle of the rings, unwrapped across 0/360°) are used.

The errors are those of the noise: the profile is refitted to 200 realisations of the best-fit map plus
Gaussian noise, and E_LO and E_HI are the distances of the 15.87th and 84.13th percentiles (±1σ) from the
median. The
noise is correlated as that of the data: white noise smoothed by a Gaussian of standard deviation
$\sqrt{A_{xy}/4\pi}$ pixels, with $A_{xy}$ the spatial correlation area measured as in 3DFIT with METHOD GLOBAL or BAYESIAN (at
least one beam area; the beam area for a 2D map). Its rms is $s(\mathbf x)$ for a cube, and for a 2D map the
rms of the residuals of the fit, which includes the structure that the rings do not describe (spiral arms,
asymmetries), so that the errors are conservative. The geometry is fixed: the errors do not include the
uncertainty of the centre, INC and PA.

The densities are converted to face-on surface densities in the units of the observed profile
($u$/arcsec², as SDENS_FO), and to HI mass surface densities for maps in Jy, as above. For a cube the factor
is the mean surface brightness of a face-on uniform disk of unit density: the channel sum of its model divided by
the area of its sub-rings (BBarolo 2.0.0 divided by a disk that included its empty centre: 2% low). On a
simulated disk a
few beams across, the deconvolved profile is within 2% of the true one in the centre, where the observed one
is 30% too low. Without a beam in the header, or with too few pixels, there is no deconvolved profile (a
message says so).

## Outputs

`<prefix>_ellprof.txt` has a header with the unit of the map and the pixel area, and one line per ring:

| Column | Name | Quantity | Unit |
|---|---|---|---|
| 1 | RADIUS | $R_r$ | arcsec |
| 2–6 | SUM, MEAN, MEDIAN, STDDEV, MAD | $\Sigma_r$, $\bar v_r$, $\tilde v_r$, $s_r$, ${\rm MAD}_r$ | $u$ |
| 7 | NPIX | $A_r$ | pixels |
| 8–10 | SDENS, ERR_SD, SDENS_FO | $\Sigma^{\rm sky}_r$, $\epsilon_r$, $\Sigma^{\rm fo}_r$ | $u$/arcsec² |
| 11 | NPIXBL | $B_r$ | pixels |
| 12–13 | SDENSBL, SDENSBL_FO | $\Sigma^{\rm sky, bl}_r$, $\Sigma^{\rm fo, bl}_r$ | $u$/arcsec² |
| 14–15 | HIDENS_FO, HIMASSDENS | $N_{\rm HI}$, $\Sigma_{\rm HI}$ (maps in Jy only) | $10^{20}$ cm⁻², M☉/pc² |

`<prefix>_ellprof_deconvolved.txt` (with a beam) has a header with the unit of the map and the beam, and
one line per ring:

| Column | Name | Quantity | Unit |
|---|---|---|---|
| 1 | RADIUS | $R_r$ | arcsec |
| 2–4 | SDENS_FO, E_LO, E_HI | the deconvolved face-on surface density, its lower and upper errors | $u$/arcsec² |
| 5 | SDENS_FO_OBS | $\Sigma^{\rm fo}_r$, the observed one (SDENS_FO of `<prefix>_ellprof.txt`) | $u$/arcsec² |
| 6 | DENS | $\hat\Sigma_r$, as DENS of GALMOD (a 2D map: face-on, $u$ per pixel) | |
| 7–10 | HIMASSDENS, E_LO, E_HI, HIMASSDENS_OBS | the same as HI mass surface densities (maps in Jy only) | M☉/pc² |

`<prefix>_ellprof_map.fits` is the map that was used. In Python, the `Ellprof` class returns the table
(`profile`), the deconvolved profile (`profile_deconvolved`) and the map (`map`) in memory, and `write()`
writes the files.

## Differences from BBarolo 1.8

- **Overlapping rings.** BBarolo 1.8 applied the overlap correction to the blank pixels instead of to
  the pixels shared by several rings, so the sums of overlapping rings counted those pixels more than
  once. With equally spaced radii the rings do not overlap and the profile is identical to that of 1.8.
- **SIDE** accepts the long forms (APPROACHING, RECEDING, BOTH, APP, REC, ...); 1.8 used both sides
  with them. `A` is the approaching half and `R` the receding one at any PA: 1.8 swapped them when the mean
  PA was at most 180°.
- **Blank pixels of a 2D map** are blank; 1.8 read them as 0, so NPIXBL was 0 and they lowered the mean.
- **HI columns**: the constant is $2.356 \times 10^5$ (1.8: $2.36 \times 10^5$, 0.8% from the
  $8794/(\theta_{\rm maj}\theta_{\rm min})$ of MASSDENSMAP; both are now $8846$), and a map in mJy or µJy
  is converted to Jy (1.8 took it as Jy).
- **Outputs** are `<prefix>_ellprof.txt` and `<prefix>_ellprof_map.fits` (1.8: `<prefix>_densprof.txt`
  and `<prefix>_densmap.fits`; the name `<prefix>_densprof.txt` is now the density profile of 3DFIT).
- **The deconvolved profile**, `<prefix>_ellprof_deconvolved.txt`, is new.

The full list is in {doc}`../../reference/changes`.

## References

- Di Teodoro E. M., Fraternali F., 2015, MNRAS, 451, 3021:
  [2015MNRAS.451.3021D](https://ui.adsabs.harvard.edu/abs/2015MNRAS.451.3021D)
- van der Hulst J. M., Terlouw J. P., Begeman K. G., Zwitser W., Roelfsema P. R., 1992, ASP Conf. Ser.
  25, 131 (GIPSY, whose task ELLINT is the origin of the algorithm):
  [1992ASPC...25..131V](https://ui.adsabs.harvard.edu/abs/1992ASPC...25..131V)

## In the code

- `bbarolo/ellprof.py`: `Ellprof` (`setFromCube`: the map and the sides; `update_rings`: the rings,
  widths and centre; `RadialProfile`: sub-pixels, ring membership, overlap correction, statistics and
  surface densities; `printProfile`: the table and the HI columns; `writeMap`), `_in_segment` (the
  sides).
- `bbarolo/moments.py`: `MomentMap.ZeroMoment` (the total map of a cube). `bbarolo/rings.py`:
  `readRings`.
- `bbarolo/tasks.py`: `_compute_ellprof`, `_write_ellprof` (the task) and the `Ellprof` class.
- `bbarolo/profile.py`: `ellprof_deconvolved` (the deconvolved profile), `MapRingModel` (the rings of a
  2D map), `profile_system`, `solve_profile` (shared with 3DFIT with METHOD GLOBAL or BAYESIAN).
