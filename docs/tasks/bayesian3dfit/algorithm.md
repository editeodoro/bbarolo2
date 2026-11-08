# BAYESIAN3DFIT: how it works

BAYESIAN3DFIT fits a 3D tilted-ring model to a datacube, as {doc}`3DFIT <../3dfit/algorithm>` does, but
instead of minimising the residuals it samples the posterior probability distribution of the free
parameters with nested sampling. The inputs are the cube, its mask, the rings (given or estimated as in
3DFIT), the free parameters ({ref}`FREE <par-bayesian3dfit-free>`) and their priors
({ref}`PRIORS <par-bayesian3dfit-priors>`). The outputs are the posterior samples, the median and the 68%
credible interval of each free parameter, the best-fit rings with these errors, the best model with its
maps and position–velocity slices, and the plots. In Python the same fit is
{class}`bbarolo.BayesianBBarolo` (method `run`), which also offers a fit of the geometry on the total
intensity map (`run_geometry`).

## The idea

3DFIT fits the rings one at a time, each against the data of its own annulus, with a downhill simplex.
BAYESIAN3DFIT builds one model of the whole disk from all the rings and treats all the free parameters
together, as one vector $\boldsymbol\theta$ of dimension $n$. By Bayes' theorem

$$
p(\boldsymbol\theta \mid \mathbf d) = \frac{\mathcal L(\mathbf d \mid \boldsymbol\theta)\,
\pi(\boldsymbol\theta)}{Z},
\qquad
Z = \int \mathcal L(\mathbf d \mid \boldsymbol\theta)\,\pi(\boldsymbol\theta)\,\mathrm d^n\theta ,
$$

where $\mathbf d$ is the data cube, $\mathcal L$ the likelihood, $\pi$ the prior and $Z$ the evidence.
Nested sampling (Skilling 2006) follows a set of live points drawn from the prior, replacing the one of
lowest likelihood with a new point of higher likelihood at each step, so that it moves through shells of
decreasing prior volume. It returns weighted samples of the posterior and an estimate of $\ln Z$. The
credible intervals include the correlations between parameters (e.g. INC and VROT), and a parameter shared
by all the rings is constrained by the whole disk.

The fit runs in four steps:

1. the initial rings (printed, with {ref}`VERBOSE <par-all-verbose>`, as the table of the initial parameters
   of 3DFIT) and the priors;
2. the setup of the likelihood: the noise of the cube and its correlation, the voxels and the region of
   the likelihood;
3. the sampling: for each $\boldsymbol\theta$ proposed by the sampler, the rings are updated, the density
   profile is recomputed if needed, the model cube is built and $\ln\mathcal L$ is evaluated;
4. the posterior statistics, the best-fit rings and the outputs.

## Free parameters and priors

### The free parameters

FREE lists the free parameters (in Python `run(freepar=...)`). A ring parameter (VROT, VDISP, DENS, Z0,
INC, PA, XPOS, YPOS, VSYS, VRAD, VVERT, DVDZ, ZCYL) enters $\boldsymbol\theta$ in one of three forms; three
more parameters are single values:

| In FREE | Dimensions | Value in ring $j$ |
|---|---|---|
| `NAME` | $N_\mathrm r$ | one value per ring, $p_j$ |
| `NAME_SINGLE` | 1 | the same value $p$ in all the rings |
| `NAME_FUNC` | 2 | a function of radius, $g(R_j; p_1, p_2)$ |
| `SIGMA`, `FRACERR`, `RADMAX` | 1 each | the noise per voxel, the model error, the outer edge (below) |

$N_\mathrm r$ is the number of rings and $R_j$ the radius of ring $j$ (arcsec). The functions of radius are

$$
V_\mathrm{rot}(R) = \frac{2}{\pi}\,p_1 \arctan\frac{R}{p_2},
\qquad
V_\mathrm{disp}(R) = p_1 + p_2 R,
\qquad
\Sigma(R) = p_1\,e^{-R/p_2},
$$

with $p_1$ in km/s for VROT and VDISP and in the units of {ref}`DENS <par-bayesian3dfit-dens>` for DENS,
$p_2$ in arcsec for VROT and DENS and in km/s/arcsec for VDISP. In Python, `f.funcs[name]` can be any
function $g(R, p_1, \dots, p_k)$ of any ring parameter; $k$ is read from its signature.

The ring parameters that are not free keep their initial values. The initial rings are built by the code of
3DFIT, and are those of 3DFIT for the same parameters: the values given (or read from RINGFILE), the
estimates of {ref}`GUESSTYPE <par-bayesian3dfit-guesstype>` (the rings, the centre, VSYS, VROT, INC and PA,
and VDISP from the second moment with GUESSTYPE FIT), and otherwise VDISP 8 km/s, Z0 one pixel, DENS 1 and
VRAD, VVERT, DVDZ, ZCYL 0. (3DFIT then changes its rings for REVERSE and, with PA free, refines PA and VROT
with a 2D fit of the velocity field: these are steps of its fit, which BAYESIAN3DFIT does not take.) A point
$\boldsymbol\theta$ for which a free VROT, VDISP, INC, XPOS, YPOS, Z0 or DENS is negative in some ring, or
RADMAX is negative, has $\mathcal L = 0$. The parameters of a function of radius may be negative (e.g. the
slope of VDISP).

### Priors

The priors are independent, $\pi(\boldsymbol\theta) = \prod_k \pi_k(\theta_k)$. The samplers draw points
$u_k$ in the unit hypercube and map them to the parameters with the prior transform
$\theta_k = F_k^{-1}(u_k)$, where $F_k$ is the cumulative distribution of $\pi_k$. A parameter with one
value per ring has the same prior in every ring. Any continuous distribution of `scipy.stats` can be set
with {ref}`PRIORS <par-bayesian3dfit-priors>` (in Python, `f.priors[name]`, or a frozen distribution in
`f.prior_distr`).

The default priors are uniform, and those of the ring parameters follow the initial rings (given or
estimated) and the beam and channels of the cube. One prior serves all the rings, so "initial ± $w$" spans
the initial values of all the rings, from the smallest − $w$ to the largest + $w$:

| Parameter | Default prior |
|---|---|
| INC | initial ± 10°, within 0–90° |
| PA | initial ± 15° (one sense of rotation: PA + 180° is excluded) |
| VROT | 0 to twice the largest initial VROT |
| VDISP | 0 to twice the largest initial VDISP, at least 0–50 km/s |
| XPOS, YPOS | initial ± 2 beam FWHM |
| VSYS | initial ± 5 channels, at least ± 20 km/s |
| Z0 | 0 to one beam FWHM |

The others (VRAD, VVERT, DVDZ, ZCYL, DENS and the parameters of the functions of radius) are fixed ranges,
listed under PRIORS. VROT uses the largest initial value, so that the prior of every ring holds the flat part
of the rotation curve: the value of each ring would give the inner rings of a rising curve a prior narrower
than their uncertainty. The priors follow the rings when they change (in Python, `update()`), but those given
are kept. Where the data constrain a parameter poorly (Z0, the INC of a nearly face-on disk), the posterior
is as wide as the prior; and the prior of INC can miss the true INC when the estimate is 10° off (estimates
below 45° scatter by 10–12°, see GUESSTYPE): give INC, or a wider prior, for a galaxy seen nearly face-on.
Three priors are part of the algorithm:

- SIGMA: log-uniform between $0.5\,\sigma_\mathrm n$ and $1000\,\sigma_\mathrm n$, where $\sigma_\mathrm n$
  is the measured noise (below). The prior is scale-invariant, and wide because the misfit of real
  galaxies can be large (65 times the noise for NGC 2403);
- FRACERR: log-uniform between $10^{-3}$ and 1;
- RADMAX: uniform within ±1 beam FWHM of the last radius $R_{N_\mathrm r}$, within the rings. With `radmax`
  `edge` or `taper` (below), the edge of the disk, it stops at the outer edge of the last ring (the last
  radius with {ref}`RINGMODE <par-bayesian3dfit-ringmode>` NODES), beyond which the edge has no effect; with
  `radmax=ring`, the last ring itself, it stays beyond the middle of the last two radii.

The bounds of the priors also set the region of the likelihood (below): a prior wider than needed makes the
region, and the computation, larger.

## The model

At each evaluation the rings take the values of $\boldsymbol\theta$, and the model cube
$m(\boldsymbol\theta)$ is built by the GALMOD engine of all the tasks ({doc}`../galmod/algorithm`): the
ring values are interpolated linearly between the radii, with the ring convention of RINGMODE, the gas of
each sub-ring is projected and given its line profile, and the cube is convolved with the beam. Compared
with 3DFIT:

- the models are ANALYTIC by default ({ref}`GALMODTYPE <par-bayesian3dfit-galmodtype>`): the expectation of
  the Monte-Carlo model, without its noise. The likelihood is then smooth in $\boldsymbol\theta$, which the
  sampler needs when $\sigma$ is the noise of the data, and a call is about twice as fast. `GALMODTYPE MC`
  gives Monte-Carlo models, with the same seed in every evaluation (the clouds outside the box below still
  draw their random deviates, so the model in the box is a crop of the model of the whole cube);
- the model is one disk of all the rings, with one set of rings (no second stage) and the engine on one
  thread (the parallelism is in the processes of the sampler);
- the model is computed in a box around the centre of the last ring that contains the disk out to its outer
  edge plus three scale heights; it is zero outside;
- NORM is not used: the flux of the model comes from DENS, free or deconvolved from the data (below).

### The density profile when DENS is not free

When DENS is not in FREE, the densities of the rings are not sampled: they are computed from the data for
the geometry of each $\boldsymbol\theta$. The total intensity map of the model is linear in the densities
$\Sigma_j$ of the rings,

$$
M(\mathbf x) = \sum_{j} \Sigma_j\, K_j(\mathbf x),
$$

where $\mathbf x$ is a pixel and $K_j$ the channel sum of the beam-convolved model of ring $j$ with unit
density and the other rings empty (with the linear interpolation, a "hat" profile that peaks at $R_j$).
$K_j$ is the exact channel sum of the ANALYTIC model, and the expectation of the MC one. It depends on the
geometry (centre, INC, PA, Z0, layer type, RADMAX), not on the kinematics, as long as the lines fall within
the band of the cube.

The data map $D(\mathbf x)$ is the sum of the data over the voxels of the likelihood in the spectrum at
$\mathbf x$ (below; units of the cube), with noise $s(\mathbf x)$ (see [the noise of the
map](#noise-of-the-cube)). The densities are the solution of a non-negative weighted least-squares problem
with a smoothness term:

$$
\hat{\boldsymbol\Sigma} = \underset{\boldsymbol\Sigma \ge 0}{\arg\min}
\left[\sum_{\mathbf x}\left(\frac{D(\mathbf x) - \sum_{j=1}^{N_\mathrm r+1}\Sigma_j K_j(\mathbf x)}
{s(\mathbf x)}\right)^2
+ \alpha\,\frac{\lVert A\rVert^2}{\lVert L\rVert^2}\,\lVert L\boldsymbol\Sigma\rVert^2\right].
$$

- The pixels are those of the likelihood region with at least one voxel of the likelihood and
  $s(\mathbf x) > 0$; at least $2N_\mathrm r$ are needed.
- Column $j = N_\mathrm r + 1$ is an exterior annulus, from the outer edge of the disk $R_\mathrm{out}$ to
  two beams beyond it (centred at $R_{N_\mathrm r+1} = R_\mathrm{out} + F$, where $F$ is the beam FWHM in
  arcsec), with the geometry of the last ring. $R_\mathrm{out}$ is the outer edge of the rings, or RADMAX
  when it is the edge of the disk (`radmax=edge` or `taper`, below) and lies inside it. The annulus takes the
  emission beyond the disk that falls in the region, which the last ring would otherwise absorb. Its density
  is not used.
- $A_{\mathbf xj} = K_j(\mathbf x)/s(\mathbf x)$ and $\lVert\cdot\rVert$ is the Frobenius norm.
  $L$ is the $(N_\mathrm r - 1)\times(N_\mathrm r + 1)$ matrix of second differences on the radii, with
  $h_i = R_i - R_{i-1}$:

  $$
  (L\boldsymbol\Sigma)_i = \frac{2}{h_i + h_{i+1}}
  \left[\frac{\Sigma_{i-1}}{h_i} - \left(\frac{1}{h_i} + \frac{1}{h_{i+1}}\right)\Sigma_i
  + \frac{\Sigma_{i+1}}{h_{i+1}}\right],
  \qquad i = 2, \dots, N_\mathrm r .
  $$

- $\alpha$ is `densprof_smooth` of {ref}`LIKEOPTS <par-bayesian3dfit-likeopts>` (default 0.1; 0: no
  smoothness term). The factor $\lVert A\rVert^2/\lVert L\rVert^2$ makes the weight $\alpha$ times the mean
  data weight of a ring, independent of the units and of the signal-to-noise ratio. Rings narrower than the
  beam are nearly degenerate once convolved, and without this term noise and non-axisymmetric structure make
  the profile oscillate from ring to ring.

The data map sums the voxels of the likelihood, while the maps $K_j$ sum all the channels: the profile
assumes that the voxels of the likelihood hold the whole line. This holds with the default dilated mask (the
signal outside it is below 0.1%), not with `dilation` (0, 0) (about 20% outside).

The problem is reduced by a QR factorisation to an $(N_\mathrm r+1)$-square system and solved with the NNLS
algorithm (Lawson & Hanson 1974). The profile is recomputed at every evaluation when INC, PA, XPOS, YPOS,
Z0 or RADMAX is free, otherwise once at the setup, and once more at the best-fit geometry at the end.

The beam smooths the observed map: the average of the observed map along each ring (the profile of the
original Bayesian module, and of {doc}`ELLPROF <../ellprof/algorithm>`) is smoothed a second time by the
model, so that for marginally resolved disks the inner rings come out too faint and the outer ones too
bright, which biases INC and VDISP (see [Validation](#validation)). That observed profile is now only an
output.

The errors of the profile combine two terms in quadrature, separately for the lower and the upper error:

- geometry: the profile recomputed for 200 random samples of the posterior; the error is the distance of
  the 16th and 84th percentiles from the median (zero if the geometry is fixed);
- noise: at the best-fit geometry, the profile refitted to 200 realisations of the best-fit map plus
  Gaussian noise with the map noise $s(\mathbf x)$ of each pixel and the spatial correlation of the noise
  (white noise smoothed by a Gaussian of standard deviation $\sqrt{A_{xy}/4\pi}$ pixels, which has
  correlation area $A_{xy}$; see [Correlated noise](#correlated-noise)), with the same percentiles.

## The likelihood

### Noise of the cube

The noise per voxel is measured as the scaled median absolute deviation of the data $d$ outside the mask
({ref}`MASK <par-bayesian3dfit-mask>`):

$$
\sigma_\mathrm n = 1.4826\;\mathrm{median}\,\bigl|\,d - \mathrm{median}(d)\,\bigr| ,
$$

in the units of the cube (e.g. Jy/beam). The noise of the data map at pixel $\mathbf x$, the sum of
$n(\mathbf x)$ contiguous channels, is

$$
s(\mathbf x) = \sigma_\mathrm n\left[n + 2\sum_{l \ge 1}\max(n - l, 0)\,\rho_z(l)\right]^{1/2},
$$

where $\rho_z(l)$ is the measured spectral autocorrelation of the noise at a lag of $l$ channels (below),
summed over its positive core, up to the first lag where $\rho_z(l) \le 0$ (conservative, as the likelihood);
with `noisecorr` other than `measured`, $s = \sigma_\mathrm n\sqrt n$.

### Voxels of the likelihood

The model is compared with the data in a fixed set of voxels $\mathcal V$, chosen with `likemask` of
LIKEOPTS:

- `dilated` (the default): the voxels of the mask dilated by a disk of radius $n_\mathrm b F_\mathrm{pix}$
  pixels in each channel ($F_\mathrm{pix}$: the beam FWHM in pixels), then by $\pm n_\mathrm c$ channels;
  `dilation` $= (n_\mathrm b, n_\mathrm c) = (1, 3)$ by default;
- `none`: all the voxels of the likelihood region.

The data in these voxels are used as they are, neither masked nor clipped: the faint emission below the
threshold of the mask (line wings, beam-smeared edges) is compared with the model, and a voxel is selected
by the mask of its neighbours, not by its own value. With channels much narrower than the velocity
dispersion, more channels are needed (about $1.5\,V_\mathrm{disp}$ divided by the channel width). Voxels
with non-finite data are skipped. $N_\mathrm v$ is the number of voxels of $\mathcal V$ within the
likelihood region.

### Likelihood region

The residuals are computed in a fixed set of pixels, the same for every $\boldsymbol\theta$: those where the
model can be non-zero for any value allowed by the priors. The region holds the pixels within a distance

$$
3\,z_{0,\max} + F/2 + \delta_\mathrm c + p/2
$$

(arcsec) of the projected disk of radius $R_{N_\mathrm r} + \Delta/2$, at the lowest inclination allowed
(the lower bound of the prior of INC, or the lowest initial INC) and for all the position angles allowed by
the prior of PA (any angle if its range is 180° or more), within the union of the model boxes at the
extreme centres. Here $z_{0,\max}$ is the upper bound of the prior of Z0 (or the largest Z0),
$\delta_\mathrm c$ the largest shift of the centre allowed by the priors of XPOS and YPOS, and $p$ the
pixel size. The model is zero outside its own box, where the squared data still count. The likelihood thus
does not depend on the box of the model, and the data beyond the outermost ring are not part of the fit.

With {ref}`SIDE <par-bayesian3dfit-side>` A or R, the region holds only the pixels of the approaching or of
the receding half of the galaxy: those on that side of the minor axis of the initial rings, through their
median centre at their median position angle (the pixels on the axis belong to both halves). The half is
fixed during the sampling, also when the centre or PA are free, so that the likelihood always compares the
model with the same voxels; the density profile, computed in the region, uses that half too.

Before sampling, a warning is printed when more than 5% of the flux of the total intensity map lies beyond
the outermost ring plus half a beam, at the initial geometry: the model cannot reproduce it, and since the
region is that of the lowest inclination allowed, with a wide prior of INC it biases INC low.

### The Gaussian likelihood

$$
\ln\mathcal L(\boldsymbol\theta) = -f \sum_{i\in\mathcal V}
\left[\frac{\bigl(d_i - m_i(\boldsymbol\theta)\bigr)^2}{2 v_i} + \frac{1}{2}\ln(2\pi v_i)\right]
- P(\boldsymbol\theta),
\qquad
v_i = \sigma^2 + \epsilon^2 m_i^2(\boldsymbol\theta),
$$

where $d_i$ and $m_i$ are the data and the model in voxel $i$ (units of the cube), and

- $\sigma$ is the noise per voxel (units of the cube): $\sigma_\mathrm n$ by default, the value of `sigma`
  in LIKEOPTS, or the free parameter SIGMA;
- $\epsilon$ is the fractional model error (dimensionless): 0 by default, `fracerr` in LIKEOPTS, or the free
  parameter FRACERR. It adds a model error proportional to the model, which absorbs a mismatch where the
  model is bright (e.g. non-axisymmetric structure); its posterior measures the misfit per voxel;
- $f$ is the factor for the correlation of the noise ([Correlated noise](#correlated-noise));
- $P$ is the optional smoothness penalty (below).

Without $\epsilon$ the likelihood is

$$
\ln\mathcal L = -f\left[\frac{S}{2\sigma^2} + \frac{N_\mathrm v}{2}\ln(2\pi\sigma^2)\right] - P,
\qquad S = \sum_{i\in\mathcal V}(d_i - m_i)^2 .
$$

With SIGMA free its maximum is at $\sigma^2 = S/N_\mathrm v$, the mean square of the residuals, and its
relative width is about $1/\sqrt{2fN_\mathrm v}$. At the end of the run the reduced chi-square of the best
fit, $\chi^2_\mathrm{red} = S/(\sigma^2 N_\mathrm v)$, is printed. Values much larger than 1 mean that the
model does not describe the data down to the noise, and that the credible intervals obtained with
$\sigma = \sigma_\mathrm n$ are too small: a warning suggests adding SIGMA or FRACERR to the free
parameters. With $\sigma = \sigma_\mathrm n$ the posterior of a real galaxy can be extremely narrow (NGC
2403: $\chi^2_\mathrm{red} \approx 4650$, formal errors of 0.02 km/s on VROT).

### Correlated noise

Neighbouring voxels share their noise (the beam, spectral smoothing), so the $N_\mathrm v$ voxels are not
$N_\mathrm v$ independent measurements. The log-likelihood is multiplied by $f$ = 1/(number of voxels per
independent noise value), i.e. the likelihood is $\mathcal L^f$, so that each independent measurement
counts once. `noisecorr` of LIKEOPTS sets $f$:

- `measured` (the default): $f = 1/(A_{xy}L_z)$, the inverse of the correlation volume of the noise measured
  in the cube, conservative: at least one beam area (below). If fewer than 2000 voxels without emission are
  available (e.g. `MASK NONE`), the beam area is used;
- `beam`: $f = 1/\Omega_\mathrm b$, with $\Omega_\mathrm b = \pi F_\mathrm{maj}F_\mathrm{min}/(4\ln 2\,p^2)$
  the beam area in pixels (spatial correlation only);
- a number $N > 0$: $f = 1/N$ ($N = 1$: independent voxels).

The correlation is measured in the voxels without emission: those outside the mask dilated by
$\lceil F_\mathrm{pix}\rceil$ pixels and 2 channels, with the median subtracted. The autocorrelations are
computed with FFTs, each lag normalised by the number of pairs of noise voxels:

$$
S_{xy} = \sum_{\lVert\boldsymbol\ell\rVert \le 2.5F_\mathrm{pix}} \rho_{xy}(\boldsymbol\ell),
\qquad
S_z = 1 + 2\sum_{l=1}^{K}\rho_z(l),
$$

where $\rho_{xy}(\boldsymbol\ell)$ is the normalised autocorrelation within the channel maps at a spatial
lag $\boldsymbol\ell$ (pixels), $\rho_z(l)$ that along the spectra at a lag of $l$ channels, and $K = 6$
channels (fewer if the runs of channels used are shorter). The sums are the correlation area and length of
the noise, but the noise of synthesis images without short spacings has an autocorrelation with a negative
ring, whose sum can be a fraction of the beam (0.18 beam areas for a WSRT cube), and with it the posteriors
are too narrow. The measure is therefore conservative:

$$
A_{xy} = \max\left[S_{xy},\ S^+_{xy},\ \Omega_\mathrm b,\ 1\right],
\qquad
L_z = \max\left[S_z,\ S^+_z,\ 1\right],
$$

where $S^+_{xy}$ and $S^+_z$ are the same sums over the positive core of the autocorrelation, out to the
first radius (lag) where its azimuthal mean ($\rho_z$) is $\le 0$. $A_{xy}$ is in pixels and $L_z$ in
channels. A warning says when $S_{xy}$ is below half the beam area: the noise has a negative ring, or the beam
in the header is wrong.
The spatial correlation is measured in the emission-free channels (at most 0.1% of their pixels in the mask,
and at least two channels from those with more than 1%) if they hold at least 20 000 noise voxels, otherwise
in all the noise voxels; the spectral one along runs of at least three emission-free channels if they give
enough pairs, otherwise along the spectra of the pixels without emission, otherwise in all the noise voxels.
Emission below the threshold of the mask adds spurious correlation, hence the preference for empty
channels. For reference: white noise convolved with a Gaussian beam has $S_{xy} = 2\Omega_\mathrm b$, noise
correlated as the beam itself $S_{xy} = \Omega_\mathrm b$, and Hanning smoothing $S_z = 8/3$; for these
$A_{xy} = S_{xy}$ and $L_z = S_z$. For NGC 2403, $S_{xy} = 18.0$ and $S^+_{xy} = 20.3$ pixels
($\Omega_\mathrm b = 15.4$), so $A_{xy} = 20.3$, and $L_z = 2.6$: 52 voxels per independent value instead
of 15, and errors 1.8 times larger than with `beam`.

The factor is exact for a signal smoother than the beam; for models with structure on the scale of the beam
it is conservative (see [Validation](#validation)).

### Smoothness penalties (Python only)

In Python, `like_kwargs=dict(lambdas={'vrot': λ, ...})` (or `f.lambdas`) adds a penalty on the second
derivative, with respect to radius, of the parameters that are free in each ring:

$$
P(\boldsymbol\theta) = \frac{1}{2}\sum_k \lambda_k \sum_j
\left[\frac{2}{h_j + h_{j+1}}\left(\frac{p_{k,j+1} - p_{k,j}}{h_{j+1}}
- \frac{p_{k,j} - p_{k,j-1}}{h_j}\right)\right]^2 ,
$$

where $p_{k,j}$ is parameter $k$ in ring $j$, $h_j = R_j - R_{j-1}$ (arcsec) and $j$ runs over the inner
rings. For VROT a node $p_{k,0} = 0$ at $R_0 = 0$ is added, which pulls the rotation curve towards zero at the
centre without penalising a curve that rises linearly from it. $\lambda_k \ge 0$ (default
0) is in arcsec⁴ per squared unit of the parameter. The penalty is not multiplied by $f$.

### The residuals of 3DFIT (`bbres`)

With `bbres=true` in LIKEOPTS, $\ln\mathcal L(\boldsymbol\theta) = -F_\mathrm{3D}(\boldsymbol\theta)$, the
residual function that 3DFIT minimises, for all the rings, with its {ref}`FTYPE <par-3dfit-ftype>`,
{ref}`WFUNC <par-3dfit-wfunc>`, {ref}`BWEIGHT <par-3dfit-bweight>`, {ref}`SIDE <par-3dfit-side>` and
{ref}`NORM <par-3dfit-norm>`. There is then no noise, no factor $f$, no model error, no penalty and no
deconvolved profile, and the width of the posterior reflects the scale of the residual function, not the
noise of the data.

## RADMAX: the outer edge of the disk

The free parameter RADMAX, $R_\mathrm e$ (arcsec), is chosen with `radmax` of LIKEOPTS:

- `edge` (the default): the outer edge of the disk. The rings do not move; the density of each sub-ring of
  the model, at radius $R$ and of width $\delta$ (three quarters of the smallest pixel size), is multiplied
  by its fraction within the edge,

  $$
  w(R) = \min\left[1,\ \max\left(0,\ \frac{R_\mathrm e - R + \delta/2}{\delta}\right)\right],
  $$

  so that the model, and the likelihood, are continuous in $R_\mathrm e$;
- `taper`: a Gaussian edge,
  $w(R) = \tfrac12\,\mathrm{erfc}\bigl[(R - R_\mathrm e)/(\sqrt2\,s_\mathrm e)\bigr]$,
  with $s_\mathrm e$ = `radmax_width` (arcsec; default $F/4$);
- `ring`: the radius of the last ring, $R_{N_\mathrm r} = R_\mathrm e$, as in the original Bayesian module:
  the last ring moves with its annulus.

The edge applies to the maps $K_j$ of the density profile too, which is recomputed at every evaluation.
With the default prior (above) the edge lies within the last ring. The values of a ring beyond the edge,
$R_j > \hat R_\mathrm e$, are not measurements: the model has no gas there, and they are extrapolations of
the profiles inside the edge (the ring only sets their slope inside it). Such rings have INDISK = 0 in
`<prefix>_rings_final1.txt` (the column is written only with `edge` or `taper`), the region beyond the edge
is shaded in the plots of the parameters, `print_stats` names them, and in Python `f.in_disk` and
`f.disk_edge` give them.

## Sampling

{ref}`SAMPLER <par-bayesian3dfit-sampler>` chooses the sampler:

- `DYNESTY` (the default; Speagle 2020): the static nested sampler, with multi-ellipsoid bounds
  (`bound=multi`, Feroz et al. 2009) and the other defaults of dynesty; `SAMPLEROPTS dynamic=true` selects
  the dynamic sampler;
- `NAUTILUS` (Lange 2023): importance nested sampling with boundaries learnt by neural networks.

{ref}`SAMPLEROPTS <par-bayesian3dfit-sampleropts>` and {ref}`RUNOPTS <par-bayesian3dfit-runopts>` pass
options to the sampler and to its run; they are checked against the arguments of the sampler before the
setup. `nlive` (`n_live` of nautilus), `dlogz` (`dlogz_init` of the dynamic sampler of dynesty,
$f_\mathrm{live} = 1 - e^{-\mathrm{dlogz}}$ for nautilus) and `seed` (a reproducible sampling) are accepted
by both samplers.

The sampling runs on {ref}`THREADS <par-all-threads>` processes (in Python, `run(threads=n)`; default 1,
or THREADS if given). The pool is started by fork: the workers inherit the fitter, with the cube and the
setup, and receive only the parameter vectors. dynesty evaluates 4 proposals per process at a time. Each
process runs the model on one thread and limits BLAS and OpenMP to one thread: the small matrices of the
density profile are slower with many threads (on NGC 2403 with the geometry free, 22 ms per call instead
of 37 ms). A Python script launched with `mpirun` or `mpiexec` uses an MPI pool instead (it needs mpi4py
and schwimmbad).

## Fit of the geometry on the total intensity map (Python only)

`run_geometry(freepar, ...)` samples only the geometry, on the total intensity map. The free parameters are
a subset of XPOS, YPOS, INC, PA and Z0 (as `_single`), DENS (any form), RADMAX, SIGMA and FRACERR; the
default is the centre, INC and PA. The priors, the region, the voxels, the density profile, the samplers and
the outputs are those of `run`. The likelihood is

$$
\ln\mathcal L_\mathrm{map}(\boldsymbol\theta) = -\frac{f_\mathrm{map}}{2}\sum_{\mathbf x}
\left[\frac{\bigl(D(\mathbf x) - M(\mathbf x;\boldsymbol\theta)\bigr)^2}{u(\mathbf x)}
+ \ln\frac{u(\mathbf x)}{s^2(\mathbf x)}\right],
\qquad
u(\mathbf x) = s^2(\mathbf x)\,\frac{\sigma^2}{\sigma_\mathrm n^2}
+ \epsilon^2 M^2(\mathbf x;\boldsymbol\theta),
$$

where $M$ is the channel sum of the model cube, $D$ and $s$ are the data map and its noise (above), the sum
runs over the pixels of the region with $s > 0$, and $f_\mathrm{map} = 1/A_{xy}$ (`measured`),
$1/\Omega_\mathrm b$ (`beam`) or $1/N$. Without SIGMA and FRACERR, $u = s^2$. SIGMA scales the noise of all
the pixels and is recommended for real galaxies, whose maps have structure that a disk does not reproduce
(NGC 2403: residuals about 20 times the noise; with SIGMA free INC = 50.9 ± 0.5°, PA = 122.4 ± 0.7°, and no
convergence in 20 minutes without it). FRACERR does not suit the map: it weights the faint outer disk. The
map does not change for PA → PA + 180°, so the prior of PA should cover half a circle.

## Outputs

### Posterior statistics

The weights of the samples, $w_k$ (from the evidence for dynesty, from the posterior of nautilus), are
normalised to unit sum. The effective sample size is $\mathrm{ESS} = 1/\sum_k w_k^2$; below 20 a warning
says that the intervals are not reliable. The samples are resampled to equal weights with a fixed random
seed, so that the statistics are reproducible, and each parameter gets its 16th, 50th and 84th percentiles,
$q_{16}$, $q_{50}$, $q_{84}$. The best fit $\hat{\boldsymbol\theta}$ is the vector of the marginal
medians $q_{50}$, with errors $q_{50} - q_{16}$ and $q_{84} - q_{50}$. With dynesty, `print_stats` also
gives the sample of maximum likelihood.

### Best-fit rings and errors

The best-fit rings take the values of $\hat{\boldsymbol\theta}$ (a function of radius at the medians of its
parameters), with the density profile recomputed at the best-fit geometry when DENS is not free. They are
written in `<prefix>_rings_final1.txt`, with the columns of 3DFIT and the error columns of the sampled
parameters: a parameter free in each ring has its own percentiles in each ring; a `_SINGLE` one the same
errors in all the rings; for a `_FUNC` one, the percentiles of $g(R_j)$ over the posterior samples, relative
to the best-fit value; DENS not free, the errors of the deconvolved profile. The column INDISK marks the
rings within RADMAX.

### Files

The task writes in {ref}`OUTFOLDER <par-all-outfolder>`, with the prefix
{ref}`OUTPREFIX <par-all-outprefix>`:

- `<prefix>_rings_final1.txt`: the best-fit rings with their errors;
- `<prefix>_mod_none.fits`, the best model, with its moment maps, position–velocity slices and the plots of
  3DFIT (with the posterior errors) as for 3DFIT;
- `<prefix>_densprof.txt`: the observed profile, the face-on average of the total intensity map of the
  voxels of the likelihood along the best-fit rings (an output only);
- `<prefix>_densprof_deconvolved.txt` (DENS not free): the deconvolved profile with its lower and upper
  errors, the observed profile, and the errors from the geometry and from the noise;
- `<prefix>_<sampler>_results.npy`: the samples and their weights (and the results of dynesty), the free
  parameters, the initial rings, the parameters, the priors and the options of the likelihood;
- with {ref}`PLOTS <par-all-plots>`, `<prefix>_corner.pdf` (the one- and two-dimensional marginal
  posteriors, with the 16th, 50th and 84th percentiles) and `<prefix>_profiles.pdf` (VROT, VDISP, DENS,
  INC, PA, Z0, XPOS, YPOS and VSYS against radius, with their errors).

With `PLOTS -1` only the rings and the samples are written, no model. With
{ref}`VERBOSE <par-all-verbose>` the statistics of the posterior are printed. In Python, `run()` writes
nothing, and `write(outfolder=None, plots=None)` writes these files as the task does. `write_bestmodel`
writes only the best model with its outputs and the rings; `corner_plot` and `plot_profiles` return the
figures; `print_stats`,
`ring_errors` and `chi2_reduced` give the statistics; `save_results` writes the samples, and `load_results`
reads them back into an object made with the same cube (`BayesianBBarolo(cube)`, since the cube is
not saved): it takes the saved parameters, repeats the setup with the saved options of the
likelihood and recomputes the best-fit rings, the density profile and its errors, without sampling again.

## Validation

The defaults were tested on simulated disks (`performance/bayes_coverage.py`, `performance/radmax.py`): GALMOD
models (ANALYTIC) with noise convolved with the beam and Hanning-smoothed, flat disks with INC 30–75°, radius
3, 5 and 8 beams and peak S/N 5–40, plus warped, thick and lagging disks; 12 disks per case (144 fits, one
failed: at S/N 5 SEARCH found no source), fitted with nautilus, VROT in each ring, VDISP, INC and PA free, the
centre, VSYS and Z0 fixed at the true values, with uniform priors set by the tests, not the defaults (VROT
0–400 km/s, VDISP 0–60 km/s, INC ±20° and PA ±30° around the true values). These numbers come from
simulated disks, which the model describes exactly; real galaxies add model mismatch.

- The medians are more accurate than 3DFIT on the same cubes: at 3 beams and S/N 20 (12 fits), the scatter of
  the errors is 6.5 km/s on VROT (3DFIT: 10.1), 1.2° on INC (2.3°) and 0.7° on PA (1.9°). At 5 and 8 beams the
  INC of 3DFIT is biased by −1.8° and −3.0°, the posterior median by −0.3° and −0.4°.
- The credible intervals are conservative. The true VROT falls within the 68% interval in 92–100% of the
  cases: the intervals are 2–3 times wider than the actual errors. Those of INC and PA are up to twice too
  wide at 3 beams. The factor $f$ is exact only for a signal smoother than the beam.
- With noise correlated as in synthesis images, whose autocorrelation has a negative ring
  (`bayes_coverage.py --noise interf`), the plain sum of the autocorrelation gave intervals that were too
  narrow: only 30–50% of the true values fell within the 68% intervals. With the conservative measure of
  the correlation (above), 79–92% do (VDISP: 71% at S/N 20 and 42% at S/N 40, from its low bias). With
  beam-smoothed noise the two measures are the same.
- VDISP is biased low, by 0.4–1.6 km/s at S/N 20–40 (coverage 0.25–0.75), with the data in the dilated mask
  or in all the voxels, and also with the true density profile, and by 3.7 ± 1.6 km/s at S/N 10 (true VDISP
  10–22 km/s; coverage 0.83; 3DFIT −0.9 ± 3.4 km/s); the cause is not yet known. INC is biased by +2.3° at
  S/N 10 and by −7.5° at S/N 5 (3DFIT: +5.0°).
- Monte-Carlo fits, thick and lagging disks and warps (INC and PA free in each ring) behave as flat disks.
- On the same disks, the likelihood of the original Bayesian module (the data times the mask, the observed
  density profile) gave VDISP 51% too low and INC about 2° too low, with $\chi^2_\mathrm{red} \approx 5$ at
  the true parameters. The data in the dilated mask with the deconvolved profile give VDISP −3%, INC −0.2°
  and $\chi^2_\mathrm{red} = 1.03$. The errors of the deconvolved profile are 0.9–1.3 times its scatter over
  noise realisations at S/N 20 and 0.7–1.0 times at S/N 10.

RADMAX was tested on 12 disks with a sharp edge drawn within the last ring, at 3 and 5 beams and S/N 20,
with rings half a beam wide ($w$); the table gives the error of the edge (median and scatter, in units of
$w$), the coverage of its 68% interval, and the coverage of the VROT of the last ring:

| RADMAX | Edge, 3 beams | Edge, 5 beams | VROT of the last ring |
|---|---|---|---|
| `ring` | −0.02 ± 0.30 $w$, 0.75 | −0.04 ± 0.27 $w$, 0.83 | 0.83, 0.92 |
| `edge` (default) | −0.02 ± 0.27 $w$, 0.75 | −0.13 ± 0.28 $w$, 0.75 | 1.00, 1.00 |
| `taper` | −0.08 ± 0.30 $w$, 0.75 | −0.16 ± 0.31 $w$, 0.67 | 1.00, 1.00 |
| evidence of $N_\mathrm r \pm 1$, $N_\mathrm r$ rings | −0.21 ± 0.69 $w$ | +0.86 ± 0.72 $w$ | 0.83, 1.00 |

`ring` and `edge` locate the edge equally well, but with `ring` the VROT of the last ring refers to a radius
that changes from sample to sample.

## Differences from BBarolo 1.8

The C++ BBarolo 1.8 has no BAYESIAN3DFIT task: the Bayesian fit was the module `bayesian.py` of pyBBarolo,
on top of the C++ library. With respect to it:

- the fit runs from parameter files (BAYESIAN3DFIT, FREE, PRIORS, SAMPLER, SAMPLEROPTS, RUNOPTS, LIKEOPTS);
  in Python, `run` and `run_geometry` replace `compute` and `compute_geometry`;
- the models are ANALYTIC by default (they were MC);
- $\sigma$ is the measured noise when SIGMA is not free (it was 1 in the units of the data), the
  normalisation term counts all the voxels (it was $\tfrac12\ln 2\pi\sigma^2$ once, so a free $\sigma$ had
  no meaning), and the log-likelihood is multiplied by $f$ (it was 1): the credible intervals are now in
  physical units. `LIKEOPTS sigma=1 noisecorr=1` gives the old noise conventions;
- the default prior of SIGMA is log-uniform between 0.5 and 1000 times the noise (a half-Cauchy of scale
  0.5 in the units of the data);
- the data are not masked: the model is compared with the data in the dilated mask (it was compared with
  the data times the mask, i.e. with zero below the threshold of the mask), in a fixed region (it was the box
  of the model, which ignored the data outside it);
- the density profile, when DENS is not free, is deconvolved from the total intensity map (it was the
  average of the observed map in each ring), with errors;
- RADMAX is the outer edge of the disk (it was the radius of the last ring: `LIKEOPTS radmax=ring`);
- new: FRACERR, the measured correlation of the noise, SIDE in the likelihood, the errors in
  `<prefix>_rings_final1.txt` and in the plots, INDISK, `run_geometry` with the noise of the map in each
  pixel (it used an rms of 1 and data scaled by 1/10);
- the default priors follow the initial rings (they were fixed wide ranges: INC 0–90°, PA 0–360°, VROT
  0–350 km/s, Z0 0–10″, XPOS and YPOS ±5 pixels);
- `write` writes all the outputs of the task (the module had only `write_bestmodel` and `save_results`);
- dynesty runs its static sampler by default (it was the dynamic one);
- the smoothness penalties (`lambdas`) are second derivatives on the radii of the rings, with VROT 0 at
  $R = 0$ (they used the separation of the first two rings for all the rings, and put the zero of VROT one
  separation inside the first ring);
- MC models in the box of the likelihood are crops of the model of the whole cube (the clouds outside the
  box drew no velocity deviates, so the rest of the disk was another realisation);
- the initial rings no longer change during the sampling (each evaluation wrote its values into them, so a
  second run started from the last sample), and the parameters of a function of radius may be negative.

## References

- Di Teodoro E. M., Fraternali F., 2015, MNRAS, 451, 3021
  ([ADS](https://ui.adsabs.harvard.edu/abs/2015MNRAS.451.3021D))
- Feroz F., Hobson M. P., Bridges M., 2009, MNRAS, 398, 1601
  ([ADS](https://ui.adsabs.harvard.edu/abs/2009MNRAS.398.1601F))
- Lange J. U., 2023, MNRAS, 525, 3181 ([ADS](https://ui.adsabs.harvard.edu/abs/2023MNRAS.525.3181L))
- Lawson C. L., Hanson R. J., 1974, *Solving Least Squares Problems*, Prentice-Hall
- Skilling J., 2006, Bayesian Analysis, 1, 833 ([ADS](https://ui.adsabs.harvard.edu/abs/2006BayAn...1..833S))
- Speagle J. S., 2020, MNRAS, 493, 3132 ([ADS](https://ui.adsabs.harvard.edu/abs/2020MNRAS.493.3132S))

## In the code

- `bbarolo/bayesian/fitter.py`: `BayesianBBarolo` (`run`, `run_geometry`, the setup `_setup`, the initial
  rings and default priors `_init_rings`, `_set_default_priors`, the free parameters `_parse_freepar`,
  `_prior_transform`, `_update_rings`, `_setup_radmax`).
- `bbarolo/bayesian/likelihood.py`: `_log_likelihood`, `_log_likelihood_geometry`, `_calculate_residuals`,
  the noise `_setup_noise`, `_map_noise`, `_set_likefactor`, the voxels `_setup_likemask`, the region
  `_setup_likebox`, `_disk_region`.
- `bbarolo/bayesian/density.py` and `bbarolo/profile.py`: the deconvolved profile (`_deconvolved_profile`,
  `profile_system`, `solve_profile`) and its errors (`deconvolved_profile_errors`).
- `bbarolo/bayesian/noise.py`: `noise_correlation`, `dilate_mask`.
- `bbarolo/bayesian/sampling.py`: `sample`, `sampler_options`, the process pool.
- `bbarolo/bayesian/outputs.py`: `write_bestmodel`, `ring_errors`, `in_disk`, `corner_plot`,
  `plot_profiles`, `print_stats`, `save_results`, `load_results`.
- `bbarolo/bayesian/task.py`: `run_task` (the task of parameter files); `bbarolo/bayesian/rings.py`.
- `bbarolo/params.py`: `bayes_freepar`, `bayes_priors`, `bayes_like_option`, `LIKE_OPTIONS`.
- `bbarolo/galmod.py`: `edge_weights`, `DiskModel.ring_maps`.
