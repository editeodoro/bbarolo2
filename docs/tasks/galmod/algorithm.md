# GALMOD: how it works

GALMOD builds a model datacube of a rotating gas disk, observed on the grid of a given cube: the same
pixels, channels, spectral axis and beam. The disk is a set of concentric rings, each with its own
kinematics, density, thickness and orientation, as in GIPSY's GALMOD (van der Hulst et al. 1992),
from which the algorithm comes. The input is the cube (only its header is needed for the model; the data
are used to normalise it) and the ring parameters; the output is the model cube, normalised to the data
or not, with its moment maps and position-velocity slices. The same engine makes every disk model of
BBarolo: the models of {doc}`3DFIT <../3dfit/algorithm>`, {doc}`SPACEPAR <../spacepar/algorithm>`
and {doc}`3DFIT with METHOD GLOBAL or BAYESIAN <../3dfit/likelihood>` are the ones described here.

A model is built in five steps: the rings become a disk with a ring convention; the disk is cut into
thin sub-rings; the sub-rings are filled with emission, by random clouds (`MC`) or on a regular grid
(`ANALYTIC`), each projected on the sky and given a line profile on the spectral axis; the cube is
convolved with the beam; the final model is normalised to the data.

(galmod-rings)=
## The rings

The rings are given at the radii $R_1 < R_2 < \dots < R_N$ (arcsec) by {ref}`RADII <par-galmod-radii>`,
or by {ref}`NRADII <par-galmod-nradii>` and {ref}`RADSEP <par-galmod-radsep>`, or read from a
{ref}`RINGFILE <par-galmod-ringfile>`. Each ring $i$ has the values

- $V_{\rm rot}$, $V_{\rm rad}$, $V_{\rm vert}$, $\sigma_{\rm gas}$, $V_{\rm sys}$ (km/s):
  {ref}`VROT <par-galmod-vrot>`, {ref}`VRAD <par-galmod-vrad>`, {ref}`VVERT <par-galmod-vvert>`,
  {ref}`VDISP <par-galmod-vdisp>`, {ref}`VSYS <par-galmod-vsys>`;
- $\Sigma$, the face-on surface density, {ref}`DENS <par-galmod-dens>` (units below);
- $z_0$ (arcsec), the scale height, {ref}`Z0 <par-galmod-z0>`; $\partial V/\partial z$ (km/s/arcsec)
  and $z_{\rm cyl}$ (arcsec), {ref}`DVDZ <par-galmod-dvdz>` and {ref}`ZCYL <par-galmod-zcyl>`;
- $i$ and $\phi$ (degrees), {ref}`INC <par-galmod-inc>` and {ref}`PA <par-galmod-pa>`;
- $x_0$, $y_0$ (pixels, counted from 0), the centre, {ref}`XPOS <par-galmod-xpos>` and
  {ref}`YPOS <par-galmod-ypos>`.

The model builder treats the radii as **nodes**: it interpolates every value linearly in radius between
consecutive radii and fills the disk from $R_1$ to $R_N$. How a set of rings becomes a set of nodes is
the ring convention, {ref}`RINGMODE <par-galmod-ringmode>`:

- `CENTRED` (default): ring $i$ is an annulus centred on $R_i$, reaching halfway to its neighbours, as
  the rings fitted by 3DFIT. Two nodes are added: one at $\max(R_1 - w_1, 0)$ with the values of the
  first ring and one at $R_N + w_N$ with those of the last ring, where $w_1 = (R_2-R_1)/2$ and
  $w_N = (R_N - R_{N-1})/2$ (a single ring: $w = $ RADSEP$/2$, or $R_1/2$ without RADSEP). The disk then
  extends half a ring beyond the first and last radius, with constant values there, and takes the ring
  values at the radii. A node is not added if the extension is not wider than one sub-ring (below).
- `NODES`: the radii are the nodes; the disk covers $R_1$ to $R_N$, as GIPSY's GALMOD and TiRiFiC
  (Józsa et al. 2007).
- `SHIFTED`: the first and last radius are moved to $\max(R_1 - s/2, 0)$ and $R_N + s/2$, with
  $s$ = RADSEP (the mean separation when RADII is given), keeping their values: the models of BBarolo
  1.x. The disk covers the area of `CENTRED`, but the values in the first and last ring are interpolated
  between the shifted radii.

The convention applies to the final models (GALMOD, the final models of 3DFIT with every METHOD). In the
fits of 3DFIT and SPACEPAR each ring is modelled alone, as two nodes at its edges with the values of the
ring, so the ring tables do not depend on RINGMODE.

(galmod-subrings)=
## The sub-rings

The disk between the nodes is cut into sub-rings of width

$$
\delta = 0.75 \, \min(|\Delta x|, |\Delta y|),
$$

where $\Delta x$, $\Delta y$ are the pixel sizes (arcsec, CDELT1 and CDELT2). Consecutive nodes must be
more than $\delta$ apart, and in increasing order, otherwise the model stops with an error. The
sub-ring $j$ is the annulus $r_j \pm \delta/2$, with centres

$$
r_j = r_1 + (j-1)\,\delta, \qquad r_j < R_N, \qquad
r_1 = \begin{cases} R_1 + \delta/2 & \text{EMPTY true and } R_1 > 0,\\ \delta/2 & \text{otherwise,}\end{cases}
$$

where $R_1$, $R_N$ are now the first and last node. With {ref}`EMPTY <par-galmod-empty>` true (the
default) the disk has a hole inside $R_1$; with EMPTY false it is filled from the centre, and the
sub-rings inside $R_1$ take the values of the first node. Every value $q$ of a sub-ring is interpolated
linearly between the nodes that enclose it:

$$
q(r_j) = q_{k-1} + (q_k - q_{k-1}) \, \frac{r_j - R_{k-1}}{R_k - R_{k-1}}, \qquad R_{k-1} \le r_j < R_k .
$$

This holds for all the values, the geometry included: a warp in $i$ or $\phi$ is linear between the
nodes. The face-on area of a sub-ring is $A_j = 2\pi r_j \delta$ (arcsec²).

**Outer edge.** 3DFIT with METHOD GLOBAL or BAYESIAN can fit the outer edge $E$ of the disk (RADMAX, see
{ref}`LIKEOPTS <par-3dfit-likeopts>`), with the rings fixed. The density of each sub-ring is
then multiplied by the fraction of the sub-ring inside the edge, or by a Gaussian taper of width $w_E$
(`radmax=taper`):

$$
w_j = \min\!\left[\max\!\left(\frac{E - (r_j - \delta/2)}{\delta}, 0\right), 1\right]
\qquad \text{or} \qquad
w_j = \tfrac12 \operatorname{erfc}\!\left(\frac{r_j - E}{\sqrt2\, w_E}\right).
$$

The sharp edge is continuous in $E$, so the likelihood is too.

(galmod-geometry)=
## Geometry: from the disk to the sky

A point of the disk at radius $R$, azimuth $\theta$ and height $z$ above the plane (both $R$ and $z$
in arcsec) lies, in the plane of the sky, at

$$
x = R\cos\theta, \qquad y = R\sin\theta\cos i - z\sin i ,
$$

with $x$ along the receding half of the major axis and $y$ along the minor axis. Its pixel is

$$
p_x = x_0 + \frac{x\sin\phi' - y\cos\phi'}{\Delta x}, \qquad
p_y = y_0 + \frac{x\cos\phi' + y\sin\phi'}{\Delta y}, \qquad \phi' = \phi + \text{CROTA2},
$$

with $\Delta x$, $\Delta y$ signed (CDELT1 is usually negative: East to the left) and $\phi'$ in
radians. $p_x$ and $p_y$ are rounded to the nearest integer: all the flux of a cloud goes to one pixel.
Clouds outside the cube are lost.

The conventions follow. $\phi$ is the position angle of the receding half of the major axis, from North
towards East. $\theta$ is measured in the plane of the disk from the receding half of the major axis,
and grows towards the far side of the disk: $\theta = 90^\circ$ is the far side of the minor axis, which
lies at position angle $\phi - 90^\circ$ on the sky (for CROTA2 = 0 and East to the left). $z > 0$ is
the side of the plane away from the observer.

The line-of-sight velocity of the point (positive: away from the observer) is

$$
v = V_{\rm sys} + \left[ V_\phi(z)\cos\theta + V_{\rm rad}\sin\theta \right] \sin i + V_{\rm vert}\cos i,
$$

$$
V_\phi(z) = \begin{cases}
  V_{\rm rot} - \dfrac{\partial V}{\partial z}\,(|z| - z_{\rm cyl}) & |z| > z_{\rm cyl},\\[4pt]
V_{\rm rot} & |z| \le z_{\rm cyl}. \end{cases}
$$

The gas rotates from the receding major axis towards the far side; $V_{\rm rad} > 0$ is an outflow
in the plane (receding on the far side); $V_{\rm vert}$ is a motion perpendicular to the plane, the same
on both sides of it, positive away from the observer. $V_{\rm sys}$ is a velocity in the frame of the
spectral axis (see {ref}`below <galmod-spectral>`).

(galmod-vertical)=
## Vertical structure

The gas of each sub-ring is spread in height with the density profile $\rho(z)$ set by
{ref}`LTYPE <par-galmod-ltype>`, with scale height $z_0$:

| LTYPE | Layer | $\rho(z) \propto$ | Quantile $Q(p)$ of $z/z_0$, with $x = 2p - 1$ |
|---|---|---|---|
| 1 (default) | Gaussian | $\exp[-z^2/(2z_0^2)]$ | $\sqrt2\,\mathrm{erf}^{-1}(x)$ |
| 2 | sech² | $\mathrm{sech}^2(z/z_0)$ | $\operatorname{artanh} x$ |
| 3 | exponential | $\exp(-\lvert z\rvert/z_0)$ | $-\ln(1-x)$ for $x \ge 0$, $\ln(1+x)$ for $x < 0$ |
| 4 | Lorentzian | $[1 + (z/z_0)^2]^{-1}$ | $\tan(\pi x/2)$ |
| 5 | box | 1 for $\lvert z\rvert \le z_0$ | $x$ |

The MC models draw random heights $z = z_0\, Q(p)$ with $p$ uniform in $[0, 1)$ (the Gaussian layer:
$z = z_0\, g$, with $g$ a standard normal deviate); the ANALYTIC models use evenly spaced heights, each
weighted with the fraction of the layer around it (below). $z_0 = 0$ gives a disk with no thickness.

(galmod-mc)=
## The Monte Carlo model (`GALMODTYPE MC`)

The default model ({ref}`GALMODTYPE <par-galmod-galmodtype>` `MC`) is the GALMOD scheme of GIPSY and
BBarolo 1.8. Each sub-ring $j$ is filled with

$$
N_c = \operatorname{round}\!\left(C\, \Sigma_j^{\,m} \, \frac{A_j}{\Delta x\Delta y}\right)
$$

clouds, with $C$ = {ref}`CDENS <par-galmod-cdens>` (default 10) and $m$ = {ref}`CMODE <par-galmod-cmode>`
(default 1); $A_j/(\Delta x \Delta y)$ is the face-on area of the sub-ring in pixels. If this gives fewer
than 10 clouds per pixel (after rounding), $N_c = \operatorname{round}(10\,A_j/\Delta x\Delta y)$.
Sub-rings with $\Sigma_j = 0$ are empty. Each cloud has

- a radius $R = \sqrt{(r_j - \delta/2)^2 + 2\,\delta\, r_j\, u_1}$ and an azimuth $\theta = 2\pi u_2$,
  with $u_1, u_2$ uniform in $[0, 1]$: the clouds are uniform in area within the sub-ring;
- a height $z$ drawn from the layer (table above).

The cloud is projected on its pixel and its velocity $v$ computed as above. It is then split into
$n_v$ sub-clouds ({ref}`NV <par-galmod-nv>`; by default the number of channels of the cube; 1 if the
sub-ring has $\sigma_{\rm gas} = 0$), each with the velocity

$$
v_s = v + \sigma\, g, \qquad \sigma^2 = \sigma_{\rm gas}^2 + \sigma_{\rm instr}^2 ,
$$

with $g$ a standard normal deviate and $\sigma_{\rm instr}$ the instrumental broadening
({ref}`below <galmod-spectral>`). The sub-cloud adds its flux to the channel nearest to $v_s$ in its pixel.
On average, rounding to the nearest channel integrates the Gaussian line over the channel. The flux of a
sub-cloud is

$$
F = \frac{\Sigma_j\, A_j}{N_c\, n_v}\, \kappa_k \, I_\ell ,
$$

where $\kappa_k$ converts the density to the units of the cube in channel $k$
({ref}`Units <galmod-units>`) and $I_\ell$ is the relative intensity of line $\ell$. The flux of the model
does not depend on CDENS, CMODE or NV: they set only the Monte Carlo noise, which falls as
$(N_c n_v)^{-1/2}$.

**Random numbers.** The uniform deviates come from a xoroshiro128+ generator (Blackman & Vigna 2021),
initialised from the seed with splitmix64; the normal deviates from the ziggurat method of Marsaglia &
Tsang (2000). The generator is started from {ref}`ISEED <par-galmod-iseed>` at every model: the same
parameters give the same model, and all the models of a fit share the same random numbers, so that the
residuals change smoothly with the parameters. A cloud that falls outside the box in which the model is
built (the cube, or the box of a single ring in the fits) still draws the $n_v$ normal deviates of its
sub-clouds, so the model in a box is the crop of the model of the whole cube with the same seed.

**Threads.** The final models are built with {ref}`THREADS <par-all-threads>` threads: the sub-rings are
split into as many consecutive chunks (when there are at least two sub-rings per thread), chunk $k$
uses the seed ISEED $+ 7919\,k$ (modulo $2^{32}$), and the chunks are added in order. The final MC
models therefore depend on THREADS (a different realisation of the same noise). The models of single
rings in the fits use one thread.

(galmod-analytic)=
## The analytic model (`GALMODTYPE ANALYTIC`)

`ANALYTIC` (new in BBarolo 2) is the expectation value of the MC model, without its noise. The clouds lie
on a stratified grid and each deposits its line profile integrated over the channels. In each sub-ring:

- **Radius**: 2 samples, $R_{1,2} = \sqrt{(r_j - \delta/2)^2 + 2\,\delta\, r_j\, u}$ with $u = 1/4, 3/4$:
  the centres, in area, of the inner and outer half of the sub-ring.
- **Azimuth**: $n_\theta = \max[\operatorname{round}(3 \cdot 2\pi r_j / p), 16]$ samples, with
  $p = \sqrt{|\Delta x\Delta y|}$, at $\theta_m = 2\pi (m + 3/4)/n_\theta$ for the inner radius and
  $2\pi (m + 1/4)/n_\theta$ for the outer one (staggered by half a step). With $\delta = 0.75\,p$ this
  is about 8 samples per pixel of face-on area.
- **Height**: one sample ($z = 0$, weight 1) when $z_0 = 0$ or the disk is face-on; otherwise heights
  evenly spaced by

  $$
  \Delta z = \min\!\left(\frac{s\,p}{|\sin i|},\ \frac{\sigma}{2\,|\partial V/\partial z|\,|\sin i|}\right)
  $$

  ($s$ pixels once projected and, with a vertical gradient, half a dispersion in velocity; the
  second term only when $\partial V/\partial z \ne 0$). $s = 3/4$ when the model is convolved with a beam
  whose minor axis is at least 3 pixels, which smooths out the aliasing of the samples on the pixel grid
  (within 0.8% of the peak of the finer sampling), and $s = 1/4$ otherwise. The samples are symmetric about
  the plane and offset from it by half a step, $z_q = \pm(q + 1/2)\,\Delta z$ (a sample in the plane would fall on the edges of the
  pixels when the centre is at a pixel edge). Where the density of the layer is below $10^{-3}$ of its
  peak the bins are $4\Delta z$ wide. Each sample carries the fraction of the layer in its bin,
  $w_q = F(z_q + \Delta z/2) - F(z_q - \Delta z/2)$, with $F$ the cumulative distribution of the layer. The
  samples reach the height beyond which the layer has less than $10^{-6}$ of its flux ($5\,z_0$ Gaussian,
  $7.6\,z_0$ sech², $14\,z_0$ exponential, $z_0$ box), or $z_{\max} = (r_j + \delta + d)/|\sin i|$, with
  $d$ the distance (arcsec) from the centre to the farthest corner of the box of the model, beyond which
  no cloud can fall in it: the only limit of the Lorentzian layer, whose flux beyond it is not deposited,
  as the MC clouds outside the field. At most 1024 samples ($\Delta z$ grows if needed). The heights do
  not depend on the box, so a model in a box is the crop of the full model. (BBarolo 2.0.0 used
  equal-weight samples at the quantiles of the layer, several pixels apart in its tails at any number of
  samples.)

Each sample is projected on its nearest pixel, as a MC cloud, and carries the flux
$\Sigma_j A_j w_q/(2 n_\theta)$; consecutive heights that fall in the same pixel are deposited together. Its line is a Gaussian of dispersion
$\sigma = \sqrt{\sigma_{\rm gas}^2 + \sigma_{\rm instr}^2}$ centred on its velocity $v$, integrated over each
channel:

$$
f_k = \left| \Phi\!\left(\frac{u_{k+1/2} - v}{\sigma}\right)
  - \Phi\!\left(\frac{u_{k-1/2} - v}{\sigma}\right) \right|,
$$

where $\Phi$ is the standard normal cumulative distribution and $u_{k\pm1/2}$ are the velocities of the
edges of channel $k$ (the spectral coordinate at pixel $k \pm 1/2$ converted to velocity, in the frame of line
$\ell$ with several lines, below; so the channels need not be uniform in velocity). The channel $k$ receives
$\Sigma_j A_j w_q/(2 n_\theta) \, f_k \, \kappa_k\, I_\ell$. The profile is computed for the channels
within $K = \lfloor 5\sigma/\Delta v_{\min} \rfloor + 2$ of the central one ($\Delta v_{\min}$: the narrowest
channel); beyond $5\sigma$ the line holds less than $3\times10^{-7}$ of its flux. $\Phi$ is tabulated in
steps of $1/512$ for $|t| \le 6$ and interpolated linearly (error below $5\times10^{-7}$). A line with
$\sigma = 0$ falls in one channel. ANALYTIC models do not use CDENS, CMODE, NV or ISEED (3DFIT uses CDENS
and NV only to emulate the blanks of MC models in its residuals, see {doc}`../3dfit/algorithm`).

(galmod-spectral)=
## Line profile and spectral axis

**Velocity to channel.** A velocity $v$ (m/s) falls at the channel (counted from 0)

$$
k(v) = \frac{w(v) - \text{CRVAL3}}{\text{CDELT3}} + \text{CRPIX3} - 1,
$$

with CRVAL3 and CDELT3 converted to m/s, Hz or m (CUNIT3 m/s, km/s, kms; Hz, kHz, MHz, GHz; µm, nm,
Å). For a velocity axis $w = v$: the velocities of the model are those of the axis, whatever its
definition. For a frequency axis $w = \nu(v)$, and for a wavelength axis $w = c/\nu(v)$, with

$$
\nu(v) = \nu_0 \left(1 - \frac{v}{c}\right) \ \text{(radio)}, \qquad
\nu(v) = \frac{\nu_0}{1 + v/c} \ \text{(optical)}, \qquad
\nu(v) = \nu_0 \sqrt{\frac{c - v}{c + v}} \ \text{(relativistic)} .
$$

The definition is {ref}`VELDEF <par-all-veldef>` (by default radio for a frequency axis, relativistic
for a wavelength axis). The zero of the velocities is the redshifted line ($z$ =
{ref}`REDSHIFT <par-galmod-redshift>`): $\nu_0 = \nu_{\rm rest}/(1+z)$, with $\nu_{\rm rest}$ =
{ref}`RESTFREQ <par-galmod-restfreq>`, or the rest frequency of the header, or 1.420405752 GHz; for a
wavelength axis $\nu_0 = c/[\lambda_{\rm rest}(1+z)]$ with $\lambda_{\rm rest}$ =
{ref}`RESTWAVE <par-galmod-restwave>`, or $\nu_0 = c/\text{CRVAL3}$ without RESTWAVE.

**Several lines.** With several values of RESTWAVE (or RESTFREQ), every cloud emits each line $\ell$
with the relative intensity $I_\ell$ ({ref}`RELINT <par-galmod-relint>`; $I = 1$ for a single line).
Each line has its own reference frequency, $\nu_{0,\ell} = c/[\lambda_\ell(1+z)]$ or
$\nu_{0,\ell} = \nu_\ell/(1+z)$, and the line $\ell$ of a cloud at velocity $v$ falls at $\nu_\ell(v)$, the
formula above with $\nu_{0,\ell}$ in place of $\nu_0$: all the lines of a cloud share its Doppler factor,
and the velocities refer to the first line ($\nu_{0,1} = \nu_0$). In the MC model the sub-clouds of a cloud
have the same velocity deviate in all the lines; in the ANALYTIC model the channel edges $u_{k\pm1/2}$ of
line $\ell$ are computed with $\nu_{0,\ell}$. On a velocity axis only one line is modelled, with intensity 1
and a warning.

**Instrumental broadening.** The line of every cloud is a Gaussian of dispersion
$\sigma = \sqrt{\sigma_{\rm gas}^2 + \sigma_{\rm instr}^2}$, with

$$
\sigma_{\rm instr} = s\, \Delta v_{\rm ch}, \qquad
s = \text{SPECRES, else LINEAR, else } \frac{1}{2\sqrt{2\ln 2}} \approx 0.425 \text{ channels},
$$

({ref}`SPECRES <par-galmod-specres>`, {ref}`LINEAR <par-galmod-linear>`): by default a profile with a
FWHM of one channel. $\Delta v_{\rm ch}$ (km/s) is the channel width in velocity, the mean over the
axis for frequency and wavelength axes.

(galmod-units)=
## Units of the model

The conversion $\kappa_k$ depends on {ref}`DENSFLUX <par-galmod-densflux>`.

**DENS as a flux** (DENSFLUX true, the default). $\Sigma$ is an integrated flux per unit area: flux units
of the cube × km/s per arcsec² of face-on disk, and

$$
\kappa_k = \frac{1}{\Delta v_{\rm ch}} .
$$

The model, summed over its pixels and channels and multiplied by $\Delta v_{\rm ch}$, is
$\sum_j \Sigma_j A_j \sum_\ell I_\ell$ (the clouds that fall outside the cube excepted), before and after
the convolution. The model has the units of the cube; for a cube in Jy/beam this means that $\Sigma$ is
the sum of the pixel values (× km/s) per arcsec², the same quantity as the total intensity map of the
data.

**DENS as a column density** (DENSFLUX false; GALMOD with NORM NONE only). $\Sigma$ is the column
density in $10^{20}$ atoms cm$^{-2}$, converted to flux with the relation of the HI 21-cm line,
$N_{\rm HI} = 1.823\times10^{18} \int T_B\, dv$ (cm$^{-2}$, K, km/s), and the Rayleigh-Jeans brightness
$I_\nu = 2 k_B T_B/\lambda^2$. A column $\Sigma$ in one channel gives

$$
\kappa_k = \frac{2 k_B}{10^{-26}\, a\, \lambda_k^2\, |\text{CDELT3}|} \left(\frac{\pi}{180 \cdot 60}\right)^2
\quad \text{Jy arcmin}^{-2} \text{ per } 10^{20}\ \text{cm}^{-2},
$$

with $k_B = 1.3806488\times10^{-23}$ J/K, $\lambda_k$ the wavelength of channel $k$ (m), $A_j$ in
arcmin², and $a$ the HI relation per unit of the axis: $1.823\times10^{-5}$ (K m/s, velocity axis),
$3.8475\times10^{-6}$ (K Hz, frequency), $2.588\times10^{4}$ (K m, wavelength). The code has only this
HI conversion. Each pixel then holds a flux density in Jy; after the convolution the model is multiplied
by the area of the beam in pixels, $\Omega_b/(\Delta x\Delta y) \approx 1.1331\, b_{\rm maj} b_{\rm min}
/ |\Delta x\Delta y|$ (exactly, the inverse of the peak of the normalised kernel), so the model is in
Jy/beam, whatever the units of the cube (Jy per pixel with SM false).

(galmod-convolution)=
## Convolution with the beam

With {ref}`SM <par-galmod-sm>` true (the default) every channel is convolved with the beam of the cube:
the beam of the header (BMAJ, BMIN, BPA), or {ref}`BEAM <par-all-beam>` when given. The kernel is the beam, a 2D
Gaussian

$$
K(x', y') \propto
  \exp\!\left[-4\ln 2\left(\frac{x'^2}{b_{\rm maj}^2} + \frac{y'^2}{b_{\rm min}^2}\right)\right],
$$

with $x'$ along the major axis of the beam, at position angle BPA (North towards East, corrected for
CROTA2), and $b_{\rm maj}$, $b_{\rm min}$ the FWHMs (arcsec). It is sampled at the pixel centres, set to
zero below $10^{-4}$ of its peak (out to about 1.82 FWHM from the centre), and normalised to unit sum
(at most 512 × 512 pixels; if no kernel can be built, the model is not convolved, with a warning). The
convolution is linear, done with real FFTs on arrays padded with zeros: emission is not wrapped around
the edges, and emission outside the cube is lost. Channels with no emission are skipped. Values below
$10^{-12}$ are set to 0 after the convolution. The normalised kernel conserves the flux; with DENS as a
column density the model is then multiplied by the beam area in pixels (above).

(galmod-normalisation)=
## Normalisation of the final model

The final model is normalised to the data as set by {ref}`NORM <par-galmod-norm>` (default `AZIM`; with two
values, the second is used; `AZIMUTHAL` is the same as `AZIM`, other values become `LOCAL`). The
normalisations use the data within the mask ({ref}`MASK <par-galmod-mask>`), which GALMOD builds when it
writes its outputs or when NORM is not `NONE`.

- **NONE**: no normalisation; the flux is set by DENS.
- **AZIM** (azimuthal): the radial profile of the data in the rings is measured as by
  {doc}`ELLPROF <../ellprof/algorithm>`: $S_i$ is the mean of the total intensity map of the masked data
  (flux × km/s per pixel), over the pixels of ring $i$ where it is defined, per arcsec² of sky, and
  $\Sigma_i^{\rm fo} = S_i \cos i_i$ the face-on surface density. The model is built with
  $\Sigma_i = \Sigma_i^{\rm fo}$ for every ring with DENS $> 0$ (rings with DENS = 0 stay empty) and
  DENS as a flux. Since the number of MC clouds depends on $\Sigma$, the densities are first multiplied
  by a power of ten $s$ that brings the smallest positive $\Sigma_i^{\rm fo}$ between 0.1 and 10, and the
  model is divided by $s$: the flux does not change.
- **LOCAL**: each spectrum of the model is scaled to the flux of the data in the mask in the same
  pixel,

  $$
  M'(x, y, k) = M(x, y, k)\, \frac{\sum_{k'} D(x, y, k')\, \mu(x, y, k')}{\sum_{k'} M(x, y, k')},
  $$

  with $D$ the data and $\mu$ the mask (1 inside, 0 outside); $M' = 0$ where the model spectrum is 0.
  The model then has the total intensity map of the masked data, and its own line shapes.
- **BOTH**: the AZIM and the LOCAL model are both written (and the LOCAL one is the final one). With
  {ref}`DEBUG <par-all-debug>`, GALMOD writes both.

**Noise.** With {ref}`NOISERMS <par-galmod-noiserms>` $= \sigma_n \ne 0$, Gaussian noise of rms
$\sigma_n$ per voxel is added to the final model. With SM the noise is first convolved with the same
kernel and rescaled to the standard deviation $\sigma_n$, so it is correlated on the scale of the beam,
as in the data. The noise is different at each run (in Python, `GalMod(...).run(seed=...)` makes it
reproducible).

(galmod-outputs)=
## The GALMOD task and its outputs

The task reads the cube, checks the beam, builds the mask (if needed), reads the rings
(RADII or NRADII and RADSEP, XPOS, YPOS, VSYS, VROT, VDISP, INC, PA and Z0 are required unless a
RINGFILE gives them; DENS defaults to 1, VRAD, VVERT, DVDZ and ZCYL to 0), warns if VSYS is outside the
velocity range of the cube, and builds the final model of the rings with RINGMODE, GALMODTYPE, SM and
THREADS on the whole cube, normalised as NORM. It writes, in {ref}`OUTFOLDER <par-all-outfolder>`, with
the prefix {ref}`OUTPREFIX <par-all-outprefix>`:

- `<prefix>_rings_model.txt`: the rings used;
- `<prefix>_mod_azim.fits`, `<prefix>_mod_local.fits` or `<prefix>_mod_none.fits`: the model;
  `<prefix>_mod_noise.fits`, the model with noise (NOISERMS);
- `<prefix>_densprof.txt`: the radial profile of the data in the rings (as above);
- `<prefix>_mask.fits`: the mask (with {ref}`PLOTS <par-all-plots>` ≥ 0);
- `maps/`: the moment maps (0, 1, 2) of the masked data, `<prefix>_0mom.fits`, ..., and of the model,
  `<prefix>_mod_0mom_<norm>.fits`, ... (see {doc}`../maps/algorithm`);
- `pvs/`: position-velocity slices of the data, of the model and of the mask along the major axis
  (`_pv_a`) and the minor axis (`_pv_b`), through the median centre at the median position angle of the
  rings (see {doc}`../pvslice/algorithm`).

In Python, `GalMod(...).run()` returns these outputs in memory, without writing any file: the model is
`model`, and the model with noise `models["noise"]`; `write()` writes the files. The maps, the slices and the
radial profile are computed only when they are used (or written). `bbarolo.DiskModel` gives the model engine
itself, to build models of any rings on the grid of a cube.

**GalMod3D.** The Python class `GalMod3D` builds model cubes from 3D fields of the density and velocity
of the gas, as functions of $(R, z)$, $(R, \theta, z)$ or $(x, y, z)$, instead of rings. It uses the
geometry, the spectral axis, the several lines, the instrumental broadening and the beam convolution
described on this page: see {doc}`../galmod3d/index`.

## Differences from BBarolo 1.8

- **Ring convention.** The final models of BBarolo 1.8 moved the first and last radius by RADSEP/2
  (`RINGMODE SHIFTED`). The default is now `CENTRED`, which changes the models in the first and last
  ring (e.g. 0.02% of the flux of the AZIM model of NGC 2403); `SHIFTED` gives the models of 1.8.
- **ANALYTIC models** (`GALMODTYPE`) are new.
- **Random numbers.** MC models use another generator, so they are not identical to those of 1.8 (the
  differences are at the level of the Monte Carlo noise; the total flux is the same). The final MC models
  depend on THREADS.
- **DENSFLUX false** was ignored by GALMOD in 1.8; it now applies with NORM NONE. A NORM with two values
  made 1.8 write no model; the second value is used.
- **Spectral axes.** Fixed in BBarolo 2: several lines on a frequency axis with REDSHIFT (placed at
  $\nu_\ell(1+z)$ instead of $\nu_\ell/(1+z)$); frequency axes in kHz, MHz, GHz and wavelength axes in
  nm; CUNIT3 = kms (taken as cm/s, an empty model); the channel width of radio velocities on a wavelength
  axis. Other velocity units are an error.
- **Several lines** were shifted from the first by fixed velocity offsets, exact only for a cloud at velocity
  0: each line now has its own reference frequency, so the lines of a cloud share its Doppler factor.
- **MC models in a box**: clouds outside the box drew no velocity deviates, so the model of a box (the
  single rings of 3DFIT CLASSIC and the models of METHOD GLOBAL and BAYESIAN) was not a crop of the model of the whole cube. Models of disks
  inside the field with LTYPE 1–3 do not change.
- **NORM AZIMUTHAL** was taken as `LOCAL`; it is `AZIM`.
- **SPECRES** is new; LINEAR still sets the instrumental broadening when SPECRES is not given.
- **Outputs** are named `<prefix>_<what>` (e.g. `<prefix>_mod_azim.fits`, `<prefix>_rings_model.txt`;
  1.8: `<prefix>mod_azim.fits`, `rings_model.txt`).

The full list is in {doc}`../../reference/changes`.

## References

- Di Teodoro E. M., Fraternali F., 2015, MNRAS, 451, 3021:
  [2015MNRAS.451.3021D](https://ui.adsabs.harvard.edu/abs/2015MNRAS.451.3021D)
- van der Hulst J. M., Terlouw J. P., Begeman K. G., Zwitser W., Roelfsema P. R., 1992, ASP Conf. Ser.
  25, 131 (GIPSY): [1992ASPC...25..131V](https://ui.adsabs.harvard.edu/abs/1992ASPC...25..131V)
- Józsa G. I. G., Kenn F., Klein U., Oosterloo T. A., 2007, A&A, 468, 731 (TiRiFiC):
  [2007A&A...468..731J](https://ui.adsabs.harvard.edu/abs/2007A&A...468..731J)
- Marsaglia G., Tsang W. W., 2000, Journal of Statistical Software, 5, 8 (the ziggurat method):
  [doi:10.18637/jss.v005.i08](https://doi.org/10.18637/jss.v005.i08)
- Blackman D., Vigna S., 2021, ACM Transactions on Mathematical Software, 47, 36 (xoroshiro128+):
  [doi:10.1145/3460772](https://doi.org/10.1145/3460772)

## In the code

- `bbarolo/galmod.py`: `DiskModel` (the engine: `array`, `cube`, `hdu`, `addnoise`, `ring_maps`);
  `model_rings` and `outer_edge` (RINGMODE); `GalmodSetup` (header quantities, `subrings`, `build`,
  `_NHItoRAD`, `_velocity_edges`); `edge_weights`; the numba kernels `_galmod_mc`, `_galmod_analytic`,
  `_line_profile`, `_zgrid` (the vertical samples), `_layer_cdf`; `_conv_sparse` (convolution of single-ring
  models).
- `bbarolo/_rng.py`: the random generator. `bbarolo/smooth.py`: the kernel (`fillgauss2d`,
  `Smooth3D.defineBeam_Gaussian`) and `FFTConvolver`.
- `bbarolo/galfit_out.py`: `writeModel`, `finalModel`, `radialProfile` (AZIM), `_local` (LOCAL).
- `bbarolo/tasks.py`: `_galmod` (the task), `GalMod`. `bbarolo/galfit.py`: `initial_rings`, `final_norm`.
