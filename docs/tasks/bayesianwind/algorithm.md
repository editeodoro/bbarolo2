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
  the data fits as well as the truth: fast winds at any inclination, hollow cones. The paper cures this with
  percentile cuts on the line profiles; the current code compares velocity percentiles of the unweighted
  model per shell.

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

The emissivities have independent Gaussian priors: mean `EMSCALE` (times (r/r0)^−`EMSLOPE`) and dispersion
`EMCLUMP` times the mean. EMCLUMP is the clumpiness: for clumps of one brightness filling a fraction f of the
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

EMSCALE and EMCLUMP are always fitted. Fixed at a rough value they bias the geometry: too low a mean favours
wide cones (the flux is spread over more cells), too high a mean narrow ones.

The emissivities are not forced to be positive: their posterior means can be slightly negative in empty
cells, within the noise. The alternative prior `exp` (non-negative exponential emissivities, the
maximum-entropy prior of a positive quantity) needs an approximate integral over a truncated Gaussian. The
Laplace approximation treats the cells at zero as independent, and overestimates the evidence of wide cones
when many weakly constrained, overlapping cells are empty. With correlated noise it favoured cones 10–17°
wider than the truth.

## The region, the noise and the sampling

The likelihood uses all the voxels of a fixed region: the pixels within `grow` beams of the projected cones of
the widest priors (the corners of the box of the priors of the centre, INC, PA, OPENANG and HTOT), all the
channels. There is no mask, because the empty voxels are what tell a cone that is too large. The noise rms is
measured outside the mask of the emission (`MASK`), and the log-likelihood is divided by the correlation volume
of the noise, as in {doc}`BAYESIAN3DFIT <../bayesian3dfit/algorithm>`. Values of the columns below `prune`
times the peak of their cell (the faint wings, where the tails of the line and of the beam multiply, about
0.3% of the flux) are left out of the normal equations, which makes the likelihood about ten times faster.

The sampling runs on `THREADS` processes, as BAYESIAN3DFIT. A likelihood evaluation of a wind 8 beams long in a
48×48×40 cube takes 15–60 ms with about 500 cells.

## Validation

`performance/wind_recovery.py` fits mocks of `performance/wind_mocks.py`: GalMod3D, an engine independent of the
fit, with random Gaussian clumps (as the mocks of MoKA-3D) or a log-normal field in the cones, and noise smoothed
by the beam. It covers filled and hollow cones, one or two cones, inclinations of 30–85°, accelerating and
rotating winds and S/N of 5–10. For each configuration and emission basis it reports the bias, the pulls and
the coverage of the 68% intervals over several realisations of the patchiness (`performance/report/
wind_recovery.md`).
