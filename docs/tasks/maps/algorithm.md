# Moment maps and profiles: how it works

The MAPS tasks reduce a datacube to two-dimensional maps and to a one-dimensional spectrum. From the
spectrum of each spatial pixel they compute the total intensity (moment 0), the velocity (moment 1) and
the velocity dispersion (moment 2), either as moments or from a Gaussian fitted to the spectrum; the
HI mass surface density; and the noise of each spectrum. The total intensity map can come with its
noise and S/N maps. The global profile is the sum of each channel. The input is the cube and, for all
the maps except the global profile, the mask of the emission; the outputs are FITS images on the
spatial grid of the cube (same pixels, same WCS) and a text file for the profile.

Each map is requested by its own keyword: {ref}`TOTALMAP <par-maps-totalmap>`,
{ref}`VELOCITYMAP <par-maps-velocitymap>`, {ref}`DISPERSIONMAP <par-maps-dispersionmap>`,
{ref}`MASSDENSMAP <par-maps-massdensmap>`, {ref}`RMSMAP <par-maps-rmsmap>` and
{ref}`GLOBALPROFILE <par-maps-globalprofile>`. They run in this order, on the same cube and mask.

## Notation

For the spatial pixel $(x, y)$ and the channel $k = 0, \dots, N_z - 1$:

- $I_k$ is the value of the cube, in its flux unit (BUNIT: e.g. Jy/beam, mJy/beam, Jy, K). Blank (NaN)
  voxels are set to 0 when the cube is read.
- $m_k \in \{0, 1\}$ is the mask: 1 for the voxels in the mask, 0 elsewhere (all 1 without a mask).
- $v_k$ (km/s) is the velocity of the channel centre. A velocity axis in m/s is divided by 1000; a
  frequency or wavelength axis is converted with the rest frequency or wavelength of the line, redshifted
  by REDSHIFT, and the velocity definition {ref}`VELDEF <par-all-veldef>` (radio, optical or
  relativistic).
- $\Delta v$ (km/s) is the absolute channel width. It is exact for velocity axes and for frequency
  axes with the radio definition, where velocity is linear in frequency; otherwise it is the mean width
  of the channels across the band.

## The mask

The maps use the mask set by {ref}`MASK <par-maps-mask>` (with {ref}`BLANKCUT <par-maps-blankcut>`
and {ref}`FACTOR <par-maps-factor>` for the SMOOTH masks; see {doc}`../makemask/algorithm`). With
`MASK NONE` no mask is used and every channel of every spectrum enters the sums. Otherwise:

- A mask built by an earlier task of the same run is used as it is.
- If there is none, the maps build it with all the detections of the source finder (only the largest
  one for the types that end in `LARGEST`), and share it.

GLOBALPROFILE never uses the mask.

## Moment maps

With {ref}`MAPTYPE <par-maps-maptype>` `MOMENT` (the default), the maps are the moments of the masked
spectrum (of its first line, with several lines: see [Several lines](#several-lines)). Let

$$
S = \sum_{k} m_k I_k
$$

be the sum of the spectrum within the mask (flux unit of the cube). Then

$$
M_0 = f \, \Delta v \, S, \qquad
M_1 = \frac{\sum_k m_k I_k v_k}{S}, \qquad
M_2 = \sqrt{\frac{\sum_k m_k I_k \,(v_k - M_1)^2}{S}} ,
$$

where $M_0$ is the total intensity, $M_1$ (km/s) the intensity-weighted mean velocity and $M_2$ (km/s)
the intensity-weighted dispersion around it. $f$ is the flux conversion below. All the channels in
the mask contribute with their sign, so noise in the mask enters the moments; where the weighted
variance comes out negative, $M_2$ is blank. Pixels with $S = 0$, including those with no voxel in
the mask, are blank (NaN) in all three maps.

$M_2$ is the dispersion of the observed line. It includes the instrumental broadening and the width
of the channels and is not corrected for either. A mask that cuts the wings of the line biases $M_2$
low.

### Units of the total map

With {ref}`FLUXCONVERT <par-all-fluxconvert>` true (the default) and a known flux unit, the sum is
converted to Jy/beam and $M_0$ is in Jy/beam km/s:

| BUNIT of the cube | $f$ |
|---|---|
| Jy/beam | 1 |
| mJy/beam, μJy/beam | $10^{-3}$, $10^{-6}$ |
| Jy (per pixel; also mJy, μJy with their factor) | $\Omega_{\rm b}$, the beam area in pixels |
| K | $\theta_{\rm maj}\,\theta_{\rm min} / (1360\, \lambda^2)$ |
| W.U. | $5 \times 10^{-3}$ |

Here

$$
\Omega_{\rm b} = \frac{\pi\, \theta_{\rm maj}\, \theta_{\rm min}}{4 \ln 2 \; p^2}
$$

with $\theta_{\rm maj}$, $\theta_{\rm min}$ the FWHM of the beam and $p$ the mean of the two pixel sizes,
in the same units. The conversion from K is the Rayleigh–Jeans relation of the 21-cm line, with the
beam axes in arcsec and $\lambda = 21.106$ cm, whatever the line of the cube. With an unknown flux
unit, or with FLUXCONVERT false, $f = 1$ and $M_0$ is in the unit of the cube times km/s.

## Gaussian maps

With `MAPTYPE GAUSSIAN` each masked spectrum $m_k I_k$ is fitted with a Gaussian

$$
G(v) = A \exp\!\left[-\frac{(v - v_0)^2}{2\sigma^2}\right],
$$

with the amplitude $A$ (flux unit of the cube), the centre $v_0$ and the dispersion $\sigma$ (km/s) as
free parameters. The fit minimises $\chi^2 = \sum_k \left[m_k I_k - G(v_k)\right]^2$ over all the
channels, with equal weights. The channels outside the mask enter the fit as zeros, so the mask
constrains the shape of the line.

- **Start:** $A$ is the maximum of the masked spectrum, $v_0$ the velocity of the channel of the
  maximum, $\sigma = 10$ km/s.
- **Minimisation:** Levenberg–Marquardt, with the analytic derivatives of $G$, run on all the spectra
  at once. The damped normal equations are
  $(J^{\rm T}J + \lambda\, {\rm diag}\, J^{\rm T}J)\,\delta p = J^{\rm T} r$, with $J$ the Jacobian and
  $r$ the residuals. A step is accepted if it lowers $\chi^2$ (and $\lambda$ is divided by 10),
  otherwise $\lambda$ is multiplied by 10. $\lambda$ starts at $10^{-3}$. The fit has converged when an
  accepted step lowers $\chi^2$ by less than $10^{-3}$ of its value, or when $\lambda$ exceeds $10^{10}$
  (as in BBarolo 1.8). All the spectra are iterated 50 times, then those not yet converged up to 1000
  times.
- **Maps:** the total intensity is the integral of the Gaussian, $M_0 = \sqrt{2\pi}\,|\sigma|\, f A$
  (units as for the moment maps); the velocity map is $v_0$; the dispersion map is $\sigma$, as
  fitted: $G$ depends only on $\sigma^2$, so a fit can end with $\sigma < 0$, which the map keeps.

Spectra whose maximum is 0 (e.g. no voxel in the mask) are blank, as are those whose fit does not
converge, or has a parameter with no effect on $\chi^2$ (a zero diagonal of $J^{\rm T}J$), and all the
spectra of a cube with as many channels as free parameters or fewer (3 with one line). Every other spectrum
gets a fit, even without emission, and its values are then meaningless: use a mask.

## Several lines

With several values of {ref}`RESTWAVE <par-galmod-restwave>` on a wavelength axis, or of
{ref}`RESTFREQ <par-galmod-restfreq>` on a frequency axis, the maps are those of the first line, the one the
velocities refer to. Let $v^{(\ell)}_k$ be the velocity of channel $k$ in the frame of line $\ell$ (its rest
value at {ref}`REDSHIFT <par-galmod-redshift>`); $v^{(1)}_k = v_k$.

- **Moment maps:** the sums above run only over the channels of the first line, those nearer to it than to
  any other line: $|v^{(1)}_k - v_c| < |v^{(\ell)}_k - v_c|$ for every $\ell > 1$, so the window ends halfway
  to the next lines. $v_c$ is the flux-weighted velocity of the first line in the total spectrum (the sum
  of the masked cube over the pixels): it starts from the mean velocity of all the lines corrected for their
  offsets, weighted with {ref}`RELINT <par-galmod-relint>`, and is iterated with the window until the window
  no longer changes. $M_0$ is then the flux of the first line only. When the lines are blended (an edge of
  the window is within $2\sigma$ of $v_c$, $\sigma$ the dispersion of the total spectrum in the window), the
  window cuts the wings of the line: a warning says so, and `MAPTYPE GAUSSIAN` is the better choice.
- **Gaussian maps:** each spectrum is fitted with one Gaussian per line,
  $\sum_\ell A_\ell \exp[-(v^{(\ell)} - v_0)^2 / 2\sigma^2]$, with one velocity $v_0$ and one dispersion
  $\sigma$ for all the lines and a free amplitude $A_\ell$ each, so the observed line ratios can differ from
  RELINT. Lines whose centre at $v_c$ is outside the cube are left out. The fit starts from the channel of
  the first line where the lines, weighted with RELINT, line up best (the largest
  $\sum_\ell {\rm RELINT}_\ell\, m I$ at the channel of each line), with each $A_\ell$ the masked spectrum
  at its line and $\sigma$ the larger of 10 km/s and one channel. $M_0 = \sqrt{2\pi}\,|\sigma|\, f A_1$ is
  the flux of the first line.

The global profile and the PV slices keep all the lines.

## Noise and S/N of the total map

With {ref}`SNMAP <par-maps-snmap>` and TOTALMAP, the task also writes the noise of each pixel of the
total map and the ratio of the two, following the appendices of Verheijen & Sancisi (2001) and Lelli
et al. (2014). The noise of a sum of $N$ channels depends on how the continuum was subtracted and on the
spectral taper of the observations.

- $\sigma$ is the noise of the cube (flux unit of the cube): the standard deviation, or MADFM/0.6745
  with {ref}`FLAGROBUSTSTATS <par-all-flagrobuststats>`, of all the non-zero voxels; with
  {ref}`ITERNOISE <par-search-iternoise>` it is computed iteratively (see {doc}`../stats/algorithm`).
- $N_{\rm l}$ and $N_{\rm r}$ are the numbers of line-free channels on the two sides of the band that
  were used to fit the continuum, {ref}`CONTCHANS <par-maps-contchans>` (one value is used for both).
  The default, $10^6$, makes the correction negligible.
- $N(x, y) = \sum_k m_k$ is the number of channels in the mask at the pixel ($N_z$ without a mask).

The noise of the total map is

$$
\sigma_{M_0}(x, y) = f\, \Delta v \; c\; \sigma_{\rm BC} \sqrt{N - b + a N^2},
$$

where $\sigma_{\rm BC}$ is the noise before the continuum subtraction, and $a$, $b$, $c$ depend on
{ref}`TAPER <par-maps-taper>`:

| TAPER | $a$ | $b$ | $c$ | $\sigma_{\rm BC}$ |
|---|---|---|---|---|
| `uniform` | $\frac{1}{4}\left(\frac{1}{N_{\rm l}} + \frac{1}{N_{\rm r}}\right)$ | 0 | 1 | $\sigma/\sqrt{1+a}$ |
| `hanning1` | $\frac{1}{4}\left(\frac{N_{\rm r} - 3/4}{N_{\rm r}^2} + \frac{N_{\rm l} - 3/4}{N_{\rm l}^2}\right)$ | 3/4 | $4/\sqrt{6}$ | $\sigma/\sqrt{1+4a/3}$ |
| `hanning2` | as `hanning1` | 3/4 | $4/(\sqrt{2}\sqrt{6})$ | $\sigma/\sqrt{1+4a/3}$ |

`hanning1` is a Hanning taper with all the channels kept, `hanning2` one with every other channel
discarded. For uniform taper and no continuum subtraction, $\sigma_{M_0} = f\,\Delta v\,\sigma\sqrt{N}$.
The flux conversion $f$ and $\Delta v$ are those of the total map. The S/N map is
$M_0 / \sigma_{M_0}$, with the total map as written (moment or Gaussian). Pixels with no channel in the
mask ($\sigma_{M_0} = 0$) are blank in the noise map.

## HI mass surface density

MASSDENSMAP converts the masked sum of each spectrum into the surface density of HI, for the 21-cm
line only. The beam is required. For a cube in Jy/beam (also mJy/beam and μJy/beam, converted to Jy/beam),

$$
\Sigma_{\rm HI}(x, y) = 8846\; \frac{\Delta v \sum_k m_k S_k}{\theta_{\rm maj}\, \theta_{\rm min}}
\quad {\rm M_\odot\,pc^{-2}},
$$

with $S_k$ the flux density in Jy/beam, $\Delta v$ in km/s and the FWHM of the beam axes
$\theta_{\rm maj}$, $\theta_{\rm min}$ in arcsec. The constant follows from
$N_{\rm HI} = 1.8224 \times 10^{18} \int T\, {\rm d}v$ and the Rayleigh–Jeans relation below; it is the
$M_{\rm HI} = 2.356 \times 10^5\, D^2 \int S\, {\rm d}v$ of {doc}`ELLPROF <../ellprof/algorithm>`. For a
cube in K, the brightness temperature $T_k$ is
first converted to Jy/beam with the Rayleigh–Jeans relation at $\lambda = 21.106114$ cm,

$$
S_k = \frac{T_k\, \theta_{\rm maj}\, \theta_{\rm min}}{1360\, \lambda^2},
\qquad\text{so that}\qquad
\Sigma_{\rm HI} = 0.01460\; \Delta v \sum_k m_k T_k ,
$$

with $T_k$ in K and $\lambda$ in cm. Other flux units are an error. The map is the projected (observed)
surface density: it is not corrected for inclination, and it does not include helium. It is always
the sum of the spectrum, also with `MAPTYPE GAUSSIAN`, and does not depend on FLUXCONVERT.

## Noise of each spectrum

RMSMAP measures the noise of each spectrum, in the flux unit of the cube. The noise of a set of
values is their standard deviation, or with FLAGROBUSTSTATS

$$
\sigma = \frac{{\rm median}\left(|I_k - {\rm median}(I_k)|\right)}{0.6745}.
$$

- **With a mask** (MASK other than NONE), $\sigma$ is computed from the channels outside the mask. A
  spectrum entirely in the mask is blank.
- **Without a mask**, it is found iteratively, removing the emission. Let $\sigma_0$ be the noise of
  the whole spectrum, and $\sigma_{j}$ that of the values left after removing all the values above
  $1.5\,\sigma_{j-1}$. The iteration stops at the first $j$ for which
  $(\sigma_{j-1} - \sigma_j)/\sigma_{j-1} < 0.1$, or when fewer than 5 values are left, or after
  100 iterations; the map gets $\sigma_{j-1}$ (with fewer than 5 channels, $\sigma_0$). The cut is one-sided: only positive values (emission)
  are removed.

## Global profile

GLOBALPROFILE sums each channel over the whole field,

$$
F_k = \sum_{x, y} I_k(x, y),
$$

and writes $F_k$ against the spectral coordinate of the channel, in the units of the spectral axis
(not converted to velocity). $F_k$ is in the flux unit of the cube: neither the mask nor FLUXCONVERT
apply. For a cube in Jy/beam, $F_k / \Omega_{\rm b}$ is the flux density in Jy.

## Outputs

| Keyword | File | Unit |
|---|---|---|
| TOTALMAP | `<prefix>_map_0th.fits` | Jy/beam km/s (FLUXCONVERT), or BUNIT km/s |
| SNMAP | `<prefix>_map_0th_RMS.fits`, `<prefix>_map_0th_SN.fits` | as the total map; none |
| VELOCITYMAP | `<prefix>_map_1st.fits` | km/s |
| DISPERSIONMAP | `<prefix>_map_2nd.fits` | km/s |
| MASSDENSMAP | `<prefix>_map_massdens.fits` | M☉/pc² |
| RMSMAP | `<prefix>_map_RMS.fits` | BUNIT |
| GLOBALPROFILE | `<prefix>_spectrum.txt` | spectral unit, BUNIT |

The maps have the spatial header of the cube (two axes) with BUNIT and BTYPE set to the quantity. In
Python, the `Maps` class makes the total, velocity and dispersion maps when no map is requested.

## Differences from BBarolo 1.8

- **Flux units mJy/beam and μJy/beam** are converted to Jy/beam with their factor in the total map
  (with FLUXCONVERT) and in the mass surface density map; BBarolo 1.8 took them as Jy/beam.
- **Spectral axes in kHz, MHz or GHz, wavelength axes in nm**, and radio velocities on a wavelength
  axis gave wrong velocities and channel widths in 1.8, hence wrong velocity, dispersion and total
  maps; they are now converted correctly.
- **Gaussian maps** use another minimiser (Levenberg–Marquardt on all the spectra at once). The
  derivative of the Gaussian with respect to its parameters was wrong in 1.8 ($e^{-x^2}$ instead of
  $e^{-x^2/2}$). As in 1.8, a fit that does not converge, or cannot be done, is blank.
- **Spatial axes in arcsec or arcmin** are converted to degrees when the cube is read, so the beam area
  and the conversions from Jy and K are right for such cubes.
- **Outputs** are named `<prefix>_map_0th.fits`, ..., `<prefix>_spectrum.txt` (1.8: `<prefix>map_0th.fits`,
  `spectrum.txt`). The S/N maps of SEARCH have their own names and no longer overwrite those of TOTALMAP.
- **MASSDENSMAP and RMSMAP** use the mask of the other maps, with all the detections; in 1.8 they used the
  largest detection when they built the mask, so the maps depended on which map ran first.
- **The HI constant** of MASSDENSMAP is 8846 (1.8: 8794, 0.6% low, and 0.8% from the $2.36 \times 10^5$
  of ELLPROF).
- **RMSMAP** of spectra with fewer than 5 channels is their noise (1.8: $10^{10}$).

The moment, rms and S/N maps are otherwise equal to those of 1.8 to float precision. The full list is in
{doc}`../../reference/changes`.

## References

- Di Teodoro E. M., Fraternali F., 2015, MNRAS, 451, 3021:
  [2015MNRAS.451.3021D](https://ui.adsabs.harvard.edu/abs/2015MNRAS.451.3021D)
- Verheijen M. A. W., Sancisi R., 2001, A&A, 370, 765 (noise of the total map, appendix):
  [2001A&A...370..765V](https://ui.adsabs.harvard.edu/abs/2001A&A...370..765V)
- Lelli F., Verheijen M., Fraternali F., 2014, A&A, 566, A71 (noise of the total map, appendix):
  [2014A&A...566A..71L](https://ui.adsabs.harvard.edu/abs/2014A&A...566A..71L)

## In the code

- `bbarolo/moments.py`: `compute_moments` (the three moments), `gaussian_moments` (the Gaussian fits),
  `MomentMap` (`storeMap`, `SNMap`, `HIMassDensityMap`, `SumMap`, `RMSMap`, `setHead`), the numba
  kernels `_rms_iterative` and `_rms_one`, and `getAllMoments` (the maps used by other tasks).
- `bbarolo/image.py`: `Image2D` (headers and FITS output, `extractGlobalSpectrum`).
- `bbarolo/bbutils.py`: `AlltoVel`, `DeltaVel` (velocities and channel width), `FluxtoJyBeam`.
- `bbarolo/tasks.py`: `_compute_maps`, `_write_maps` (the task) and the `Maps` class.
