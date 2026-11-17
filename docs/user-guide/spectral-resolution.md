# Spectral resolution: SPECRES

{ref}`SPECRES <par-3dfit-specres>` (or LINEAR, its name in BBarolo 1.8) is the instrumental broadening of the
models, in channels. Every line of a model is a Gaussian of dispersion

$$
\sigma^2 = \mathrm{VDISP}^2 + (\mathrm{SPECRES}\,\Delta v)^2 ,
$$

with $\Delta v$ the channel width of the cube. The same SPECRES is used by every task that builds models: 3DFIT,
GALMOD, SPACEPAR, WINDFIT, GALWIND and SLITFIT. Its value matters when the dispersion of the gas is not much larger
than a channel: a SPECRES too large gives a VDISP too small, and the reverse.

## What the model already includes

The models do not take the value of the Gaussian at the centre of each channel: they integrate it over the
channel. In the Monte Carlo model each cloud goes into the channel that contains its velocity
({ref}`galmod-mc`); in the analytic model the Gaussian is integrated between the edges of each channel
({ref}`galmod-analytic`). Integrating over a channel is a convolution with a boxcar one channel wide, whose variance
is $\Delta v^2/12$. This part of the line spread function is already in the model, so SPECRES must leave it out.
In particular, **averaging channels together does not go into SPECRES**: the model of the averaged cube is
integrated over the wider channels.

## The rule

Let $\sigma_\mathrm{LSF}$ be the dispersion of the line spread function of the cube: the profile that a line much
narrower than the resolution has in the final cube. With a Gaussian of the same variance in place of the line
spread function (after averaging or smoothing it is not Gaussian),

$$
\mathrm{SPECRES}^2 = \left(\frac{\sigma_\mathrm{LSF}}{\Delta v}\right)^2 - \frac{1}{12} ,
$$

with $\Delta v$ the channel width of the final cube. $\sigma_\mathrm{LSF}^2$ is the sum of the variances of the steps
that made the cube, each in the channels of that step:

| Step | Adds to $\sigma_\mathrm{LSF}^2$ |
|---|---|
| The instrument, at its native channels | the variance of its line spread function: $(F/2.355)^2$ for a FWHM $F$ measured on unresolved lines |
| Smoothing with a Gaussian of FWHM $F$ | $(F/2.355)^2$ |
| Hanning smoothing (weights 1/4, 1/2, 1/4) | $\Delta v^2/2$ |
| Smoothing with any kernel of weights $w_j$ at $j$ channels ($\sum w_j = 1$) | $\Delta v^2 \sum_j w_j\, j^2$ |
| Running mean over $N$ channels | $(N^2 - 1)\,\Delta v^2/12$ |
| Averaging $N$ channels of width $\Delta v_n$ into one | $(N^2 - 1)\,\Delta v_n^2/12$ |
| Decimation (one channel kept out of $N$) | 0 |

Smoothing changes the line spread function and goes into SPECRES. Averaging also widens the line spread function,
but in the final cube by no more than the integration over the new channel, which the model already does. If
SPECRES² comes out negative (a cube decimated from channels much narrower than the resolution), use 0: the lines
of the data are then narrower than a channel, and no model can follow them. `SPECRES 0` gives no instrumental
broadening, only the integration over the channel.

## Common cases

The default, $1/2.355 = 0.425$ channels, is a Gaussian with a FWHM of one channel, integrated over the channel:
a cube at the native channels of an instrument whose resolution is one channel (an unresolved line then has a FWHM
of 1.21 channels). Starting from such native channels, with $N$ the factor of averaging, smoothing or decimation:

| | The cube | SPECRES | For $N \gg 1$ |
|---|---|---|---|
| 1 | native channels | $1/2.355 = 0.425$ | |
| 2 | averaged over $N$ channels | $0.425/N$ | 0 |
| 3 | smoothed with a Gaussian of FWHM $N$ channels | $\sqrt{1 + N^2}/2.355$ | |
| 4 | as 3, then averaged over $N$ channels | $\sqrt{1 + N^2}/(2.355\,N)$ | 0.425 |
| 5 | as 3, then decimated by $N$ | $\sqrt{(1 + N^2)/2.355^2 - (N^2 - 1)/12}\,/N$ | 0.311 |
| 6 | smoothed with a running mean over $N$ channels | $\sqrt{1/2.355^2 + (N^2 - 1)/12}$ | |
| 7 | as 6, then averaged over $N$ channels | $\sqrt{1/2.355^2 + (N^2 - 1)/12}\,/N$ | 0.289 |
| 8 | as 6, then decimated by $N$ (the same as 2) | $0.425/N$ | 0 |
| 9 | Hanning-smoothed | $\sqrt{1/2.355^2 + 1/2} = 0.825$ | |
| 10 | averaged over $N$ channels, then Hanning-smoothed | $\sqrt{0.425^2/N^2 + 1/2}$ | 0.707 |

With $N = 1$, cases 2, 6, 7 and 8 are case 1, cases 4 and 5 are case 3, and case 10 is case 9. When the documentation of the data gives the FWHM $F$ (in
channels) of the line spread function of the cube, measured on unresolved lines (the arc lines of a spectrograph, or
the spectral resolution of a correlator), SPECRES $= \sqrt{(F/2.355)^2 - 1/12}$: 0.311 for $F = 1$, 0.80 for
$F = 2$, 1.24 for $F = 3$. (Hanning smoothing, case 9, gives a line with $F = 2.06$ in the same approximation: 0.80
and 0.825 differ by less than the approximation.)

Some data:

- **Radio cubes at the native channels**: 0.425, the default. The response of a correlator without weighting is a
  sinc with a FWHM of 1.21 channels, as that of the default. With Hanning smoothing, which doubles the resolution,
  0.825 (case 9), as for NGC 891 ({doc}`../examples/ngc891`).
- **ALMA cubes averaged to wide channels**: Hanning smoothing in the correlator and the online averaging make the
  line spread function of the delivered data, which goes into SPECRES; the averaging to the final channels (in
  `mstransform` or `tclean`) does not. With final channels much wider than the native resolution, SPECRES is close to 0 (case 2).
  A cube averaged and then Hanning-smoothed has case 10: ALESS 073.1 ({doc}`../examples/aless73`), with native
  channels of 3.55 km/s averaged to 14.2 km/s ($N = 4$), has 0.715.
- **Optical and near-infrared spectrographs**: from the FWHM of the line spread function. The V500 setup of CALIFA
  has a FWHM of 6 Å on channels of 2 Å ($F = 3$): 1.24 ({doc}`../examples/ngc5633` uses 1.2).

## How much it matters

A change $\delta$ of SPECRES² changes the fitted VDISP² by $-\delta\,\Delta v^2$, that is VDISP by about
$\delta\,\Delta v^2/(2\,\mathrm{VDISP})$. For example, 0.849 (a Gaussian of FWHM 2 channels) in place of 0.707
($\delta = 0.22$) makes VDISP 12% smaller when VDISP is one channel, and 1% smaller when it is three channels.
The error matters for dispersions of one or two channels, and is negligible for well-resolved lines.
