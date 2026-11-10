# BAYESIANWIND: how it works

## The problem: where along the line of sight

A wind fills a volume. Every line of sight through the cones crosses them over a range of depths, so the
flux of a spaxel says how much gas there is along it, not where. The velocity field maps the depth to the
line-of-sight velocity, so the line profile of the spaxel does carry the distribution along the line of
sight, but only once the geometry and the kinematics are known, and those are what the fit is for.

- A parametric law of the density (uniform, a power law) is wrong for a patchy wind, and the fit then bends
  the geometry to reproduce the clumps.
- A normalisation in each spaxel (NORM LOCAL of 3DFIT) absorbs the patchiness on the sky, not along the
  line of sight.
- MoKA-3D (Marconcini et al. 2023, A&A 677, A58) weights the clouds of each voxel by the observed flux over
  the model flux. The weighted model then equals the data wherever the model reaches and is zero elsewhere.
  The fit sees only which voxels are reachable, so any model whose position-position-velocity support covers
  the data fits as well as the truth: fast winds at any inclination, hollow cones. The paper (its section 2.6)
  cures this with cuts on the line profiles: it discards the models whose unweighted integrated spectrum has the
  1st or 99th percentile velocity more than 5% away from the observed one, and it fixes the position angle and the
  opening angle from the data. The current code (github.com/cosimomarconcini/moka_3d) fits only the 1st and 99th
  percentile velocities of the unweighted model in shells on the sky (or a CRPS of 19 percentiles), and weights
  the clouds by the observed flux only after the fit. BAYESIANWIND's `emission=voxel` keeps the weighting with a
  likelihood of the detections instead (see below).

## The emission as linear amplitudes

BAYESIANWIND keeps the idea that the data, not a law, set the emission, within a forward model:

1. The emission is a set of emissivities on cells of the cones: radial bins, polar bins from the axis to the
   largest opening angle of the priors, and azimuthal sectors, each cone its own. The cells are a fixed grid
   in the frame of the wind, about `cellsize` (the beam) across, with fewer sectors near the apex. The current
   cone clips them: the quadrature points of a cell fill its clipped part, so the model moves continuously
   with the geometry, and a cell outside the cone has no volume.
2. The model cube is linear in the emissivities, M = A ε. Each column of A is the model of one cell with unit
   emissivity: its points, each with its line integrated over the channels (as the ANALYTIC models of GALMOD),
   deposited bilinearly and convolved with the beam (minus the 1/6 pixel² of the bilinear weights), kept as a
   dense box of the cube.
3. At every evaluation, the normal equations G = AᵀWA and b = AᵀWd (W the inverse noise variance times the
   correlation factor of the noise) are accumulated over the voxels of the boxes.

## The marginal likelihood

The emissivities have independent Gaussian priors: mean `EMSCALE` (times (r/r0)^−`EMSLOPE`, and times
`EMRATIO` in the approaching cone) and dispersion `EMCLUMP` times the mean. EMCLUMP is the clumpiness: for clumps of one brightness filling a fraction f of the
cells, EMCLUMP² = (1 − f)/f. The likelihood of the other parameters is the integral over the emissivities.
For a Gaussian prior and a model linear in them, this is a Gaussian integral, exact with one Cholesky
factorisation of G + diag(1/(EMCLUMP·mean)²).

It has the Occam factor that the best-fit emission (the profile likelihood, `marginal=false`) lacks. A cone
wider than the truth, or a wind faster, reaches cells the prior expects to emit but the data find empty.
Each costs about ln(S/N) plus 1/(2 EMCLUMP²). On a mock wind with half of its cells empty and noise smoothed
by the beam:

- the profile likelihood was flat to 3 in ln L from an opening angle of 60° (the truth) to 85°;
- the marginal likelihood, with EMSCALE and EMCLUMP fitted, peaked at 58–60° and fell by 31 at 70° and by 81
  at 85°;
- it recovered the mean emissivity of the mock and EMCLUMP = 1, the clumpiness of half-empty cells.

EMSCALE, EMCLUMP, EMRATIO and EMSLOPE are always fitted. Fixed at a rough value they bias the geometry: too low a
mean favours wide cones (the flux is spread over more cells), too high a mean narrow ones. With one mean for
both cones, a wind whose receding cone was empty came out with OPENANG 8° too narrow, VDISP 9 km/s too high
and INC 5° too low: the empty cone penalised every larger cone. With EMRATIO fitted (it went to about 250),
VDISP and INC were recovered, and OPENANG was 6° low: the 100 clumps of that mock did not reach the walls
of its single cone (the opening angle of a patchy wind is that of its emission). EMSLOPE is fitted too: with
an emission falling as r^-2, EMSLOPE fixed at 0 gave VWIND 24% low, VDISP 19 km/s high and INC 11° low; fitted
(it came out at 1.4–1.8), the biases were those of the mocks of uniform emission. A wind whose other side is
not seen can also be fitted with one cone only (LIKEOPTS `cones=approaching` or `receding`): on the mocks with only
the approaching cone it gave the same results as both cones with EMRATIO free, 2.4 times faster.

The emissivities are not forced to be positive: their posterior means can be slightly negative in empty
cells, within the noise. The alternative prior `exp` (non-negative exponential emissivities, the
maximum-entropy prior of a positive quantity) needs an approximate integral over a truncated Gaussian. The
Laplace approximation treats the cells at zero as independent, and overestimates the evidence of wide cones
when many weakly constrained, overlapping cells are empty. With correlated noise it favoured cones 10–17°
wider than the truth.

## The detection likelihood (`emission=voxel`)

`emission=voxel` is the weighting of MoKA-3D: the flux of the model is the data wherever the cone reaches, so the
flux of the data says nothing about the cone, only where the data have emission. The cone is fitted to that: a
voxel is a detection if the data are above `snrcut` (default 3, as MoKA-3D) times the noise, and each voxel is
detected with the probability

  p = E[Φ(F/σ − snrcut)],

where F is the flux of the voxel, log-normal with mean the uniform cone (EMSCALE, times (r/r0)^−EMSLOPE and EMRATIO
in the approaching cone, convolved with the beam and the line) and fractional scatter EMCLUMP, and Φ includes the
Gaussian noise σ. The log-likelihood is the sum of ln p over the detections and of ln(1 − p) over the other voxels
of the region, times the correlation factor of the noise. A voxel where the cone predicts emission and the data have
none costs ln(1 − p), about −(F/σ − snrcut)²/2 for a smooth emission: this is the penalty of the cones too wide and
the winds too fast that the weighting alone lacks (it fitted even the noise of every voxel the cone reached, so
every parameter ran to the edge of its prior). A detection outside the cone costs ln Φ(−snrcut), −6.6 at 3σ.
EMCLUMP → 0 is the probit Φ(EMSCALE·M/σ − snrcut), the likelihood of the detections of a smooth cone plus noise.

The threshold matters: it hides the faint wings of each clump, so the detections are narrower than the emission.
A link that ignores it (the probability of at least one clump in the voxel, 1 − exp(−n) with n proportional to
the flux of the cone) expects detections in the faint wings of the cone: in profiles of the likelihood of a clumpy
mock it peaked at VDISP 0 and at cones 10° too narrow at 3σ (5° at 2σ, 15° at 4σ). The log-normal flux above the
threshold peaked at the true opening angle, VWIND, VDISP and INC of the clumpy mocks (EMCLUMP near 0) and of the
log-normal field (EMCLUMP about 0.7).

## The region, the noise and the sampling

The likelihood uses all the voxels of a fixed region: the pixels within `grow` beams of the projected cones of
the widest priors (the corners of the box of the priors of the centre, INC, PA, OPENANG and CONELEN), all the
channels. There is no mask, because the empty voxels are what tell a cone that is too large. The noise rms is
measured outside the mask of the emission (`MASK`), and the log-likelihood is divided by the correlation volume
of the noise, as in {doc}`BAYESIAN3DFIT <../bayesian3dfit/algorithm>`. Values of the columns below `prune`
times the peak of their cell (the faint wings, where the tails of the line and of the beam multiply, about
0.3% of the flux) are left out of the normal equations, which makes the likelihood about ten times faster.

The sampling runs on `THREADS` processes, as BAYESIAN3DFIT. A likelihood evaluation of a wind 8 beams long in a
48×48×40 cube takes 15–60 ms with about 500 cells.

## Validation

`performance/wind_recovery.py` fits mocks of `performance/wind_mocks.py`. They are made with GalMod3D, an engine
independent of the fit, with 100 Gaussian clumps (as the mocks of MoKA-3D) or a log-normal field in the cones,
noise smoothed by the beam and a peak S/N of 10 (5 for `lowsnr`). The configurations are filled and hollow cones,
a single cone, inclinations of 30° and 85°, an accelerating and a rotating wind, and a log-normal field, with 4
realisations of the patchiness each. `performance/report/wind_recovery.md` has the bias, the pulls and the coverage
of the 68% intervals of every case. With the defaults (but EMSLOPE fixed at 0, the slope of those mocks: it is fitted
by default since the test of a declining emission):

- PA is recovered with the right errors (rms pull 1.1, coverage 0.67), INC within about 1° (rms pull 1.4), the
  opening angle of a hollow cone and the acceleration and rotation of the wind within a few per cent.
- The opening angle of the clumpy mocks comes out 2.5–4.7° narrow: the clumps are cut at the walls of the cone,
  so their emission ends inside them, and the fit measures the extent of the emission. With the log-normal field,
  which fills the cone up to its walls, the opening angle is unbiased (−0.8°, coverage 1). Edge-on, where the
  width of the cones is seen on the sky, it is too (−1.1°).
- VWIND is a few per cent low and VDISP 2–9 km/s high, with the intervals about twice too narrow (rms pulls
  about 1.9, coverage 0.35 instead of 0.68). Edge-on, VWIND is 14% low. A narrower, slower wind with broader lines
  gives the same velocity spread. Clumps make lumpier spectra than any smooth description of the emission, which
  pushes the fit that way. The bias is the same with cells of 22 arcsec, with one emissivity per shell and with a
  normalisation in each spaxel, and a model error proportional to the data (`datafrac` 0.3) widens the intervals
  by only 30–40%. With the log-normal field VWIND is unbiased and VDISP 4 km/s high.

The profile likelihood (the best-fit emission, no marginalisation) has the degeneracy of the free emission. On the
filled cone it found cones 20° too wide, winds 11% too fast and almost no dispersion (rms pulls up to 10). On the
same mocks, one emissivity per shell (`shells`) or per spaxel (`spaxel`) gave about the same results as the cells,
four to six times faster.

The weighting of MoKA-3D with the χ² of the voxels the cone does not reach (the earlier `voxel`, without the cuts of
MoKA-3D) ran every parameter to the edge of its prior: VWIND +83 km/s, VDISP +55 km/s, OPENANG +25°. With the
detection likelihood (`voxel`, EMSLOPE fixed at 0 as the cells of the table) the filled and hollow cones and the
log-normal field came out as well as with the cells, about ten times faster (0.7 minutes per fit): VWIND −0.5 to
−4.6 km/s, VDISP +4 to +9 km/s, INC and PA within 0.6°, OPENANG 1–2° narrow, OPENIN +0.8°, with rms pulls mostly 0.4–1.4
(the cells: up to 2.3). Edge-on it does worse than the cells: VWIND 24% low and VDISP 24 km/s high (the cells: 14% and
8 km/s), the same trade of a slower wind for broader lines, which the detections, blind to the flux, constrain less.
