# BAYESIANSLITFIT: how it works

## The model seen through the slit

A slit samples a strip of the sky. The model is built on a thin 3D grid around it (bbarolo/bayesian/slitmodel.py):

- x runs along the slit, with the pixels of the data (or an integer fraction of them, if they are larger than half
  the seeing FWHM), over the positions of the likelihood plus four seeing dispersions;
- y runs across the slit, with square pixels, over ±(`SLITWIDTH`/2 + 4σ of the seeing): all the light that the
  seeing can bring into the slit;
- z holds the channels of the spectral window, with the spectral axis of the data: GALMOD models every line of
  `RESTWAVE` at its own rest wavelength with the same Doppler factor, the instrumental dispersion added in
  quadrature, the line integrated over each channel.

The disk is set in the frame of the slit, so that no interpolation is needed: its position angle on the grid is
90° − `SLITANGLE`, its centre at `XPOS` along the slit and at `YPOS` from the slit across it. The model is the
ANALYTIC GALMOD model (deterministic: the likelihood has no Monte-Carlo noise), not convolved.

The seeing and the slit are then applied by a projection that is exact for a circular Gaussian seeing, which is
separable:

    P(x, z) = g_x ⊛ Σ_y w_y M(y, x, z)

where w_y is the fraction of the light of a point at y that the seeing brings into the slit (the Gaussian convolved
with the box of the slit, a difference of error functions) and g_x the Gaussian of the seeing along the slit,
integrated over the pixels. P is the flux that enters the slit at each position. On a mock, it agrees with a full
cube convolved in 2D by FFT and summed over the rows of the slit to 0.4% of the peak, for a slit on the major axis
and for one 20° off and 0.4 arcsec from the centre.

## The data

The data are prepared once (bbarolo/bayesian/slitdata.py):

1. The emission is found: the lines are stacked in velocity at each position (each line weighted by RELINT, within
   ±1000 km/s of REDSHIFT), after a first straight continuum away from the lines, and a Gaussian is fitted where the
   stack peaks above 4 times its noise. Isolated detections and velocities far from those of the neighbouring
   positions (a line taken for another one) are rejected. The velocity range of the emission sets the line channels
   and the spectral window of the fit.
2. The continuum (`CONTSUB`) is a polynomial of degree `CONTORDER` at each position, fitted with a 3σ clipping in the
   continuum windows (the window away from the lines, `EXCLWIND` and `MASKWAVE`), and the residual sky (`SKYRESID`)
   the median of each channel over the positions without emission.
3. The noise: σ²(s, λ) = σ_col(λ)² + σ_obj(s)². σ_col is the robust scatter of each channel over the positions
   without emission, pooling neighbouring channels to at least 100 values (with 15 empty positions, the scatter of a
   single channel varies by 30% from channel to channel, which biases the likelihood: on a mock the pulls of the
   parameters had an rms of 2); σ_obj is the excess scatter of each position in the continuum windows over that of
   the sky. Without enough empty positions, the noise of each position only.
4. The correlation of the noise: the sums of the autocorrelation of the normalised residuals along the spectra and
   along the slit (out to the first lag where it is not positive), after removing the median of each channel and of
   each position (the systematic residuals of the sky and of the continuum). The log-likelihood is divided by their
   product, the number of pixels per independent measurement (about 1.5 × 2 for the SALT example).

## The likelihood

The likelihood is Gaussian over the pixels of the window at the positions within the outer edge of the rings plus
two seeing FWHM of any XPOS of the prior, with the noise map (times `SIGMA` if free):

- `NORM AZIM`: the model of the rings with the densities that reproduce the flux of the lines along the slit. The
  flux of each ring with unit density is the projection of its total intensity map (DiskModel.ring_maps), the data
  are summed over the line channels (the blanks filled along the spectrum), and the densities are the non-negative
  least-squares solution with the smoothness prior and the exterior annulus of BAYESIAN3DFIT (bbarolo.profile). They
  depend on XPOS only: computed again when it changes.
- `NORM LOCAL`: the AZIM model of each position u_j (normalised to unit flux) times an amplitude a_j ≥ 0 with a flat
  prior, integrated out exactly: with g = Σ W u², b = Σ W d u, c = Σ W d² (W the inverse variance times the
  correlation factor),

      ln L_j = −c/2 + b²/(2g) + ½ ln(2π/g) + ln Φ(b/√g).

  The integral, not the maximum over a_j (the profile likelihood), carries the Occam factor ½ ln(2π/g), which
  favours narrower lines less than the profile likelihood does.
- `NORM NONE`: the model of the rings with their densities.

The smoothness penalties of `LIKEOPTS lambdas` are subtracted from ln L, as in BAYESIAN3DFIT: for each parameter free
in each ring, λ/2 times the sum of its squared second derivatives with respect to radius on the radii of the rings
(VROT from 0 at R = 0), not divided by the number of pixels per independent measurement.

The sampling and the minimisation are those of BAYESIAN3DFIT (bbarolo/bayesian/sampling.py and optimize.py).

## Validation

Mock slits made by the brute-force path (a full GALMOD cube, convolved in 2D, summed over the slit; Hα and the
[NII] doublet, a linear continuum, S/N 20 at the peak of the lines, 12 rings as wide as the seeing), fitted with
`SAMPLER OPTIMISE` (performance numbers of November 2026):

| Mock | NORM | VROT, VDISP: rms of the pulls | VSYS (true 15 km/s) | XPOS (true 70.4) |
|---|---|---|---|---|
| symmetric | AZIM | 1.0, 1.2 | 15.4 ± 0.3 | 70.41 ± 0.03 |
| symmetric | LOCAL | 1.1, 0.8 | 15.6 ± 0.3 | 70.43 ± 0.03 |
| flux 50% asymmetric | AZIM | 1.9, 1.4 | 26.6 ± 0.3 | 71.56 ± 0.02 |
| flux 50% asymmetric | LOCAL | 0.9, 0.7 | 15.3 ± 0.5 | 70.26 ± 0.04 |
| slit 20° off the major axis, 0.5" from the centre, given | AZIM | 1.0, 1.3 | 14.9 ± 0.3 | 70.38 ± 0.03 |
| the same, assumed on the major axis | AZIM | 10.8, 2.2 | −0.9 ± 0.3 | 70.81 ± 0.03 |

A misaligned or offset slit that is not given as such biases VROT low (here by 20 to 50 km/s) and VSYS: SLITANGLE
and YPOS matter, and should come from the observation log and the images.

## Differences from SLITFIT of BBarolo 1.8

The SLITFIT task of the C++ code was never finished (it modelled the HI line only, set no beam, did not subtract the
continuum and minimised the 3DFIT residuals of a slit extracted by summing the model over the slit, with errors in the
loops over the rows). BAYESIANSLITFIT is a new task: the C++ keyword SLITFIT is now an error that names it.
