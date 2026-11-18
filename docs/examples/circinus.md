# Circinus: an ionised nuclear wind

The Circinus galaxy, a spiral at 4.2 Mpc with a Seyfert 2 nucleus, has a cone of ionised gas that
leaves the nucleus towards the north-west: the gas pushed out by the active nucleus, seen in [OIII] 5007 by
MUSE. The example fits a cone to it with {doc}`WINDFIT <../tasks/windfit/index>` and `METHOD GLOBAL`, with the
geometry of Marconcini et al. (2023), who modelled the same cone with MoKA-3D. It shows how to keep a fit to a wind
when the cube holds much more than the wind (a mask), and compares the descriptions of the emission of WINDFIT,
among them the statistic of MoKA-3D.

## The data

| | |
|---|---|
| Object | Circinus galaxy, Seyfert 2, redshift 0.001448 (cz = 434 km/s), 4.2 Mpc (1 arcsec = 20.4 pc) |
| Data | [OIII] 5007, MUSE at the VLT (ESO programme 094.B-0321(A)): a reduced cube from the public ESO Science Archive. These observations are studied by Mingozzi et al. (2019) and Marconcini et al. (2023) |
| Cube | 173 × 238 pixels of 0.2 arcsec, 41 channels of 1.25 Å (75 km/s), from −1720 to +1270 km/s around [OIII] 5007: the part of the MUSE cube on the side of the cone |
| Beam | 0.9 arcsec (the seeing), in the header |
| Files | [circinus_OIII.fits](https://github.com/editeodoro/bbarolo2/raw/main/examples/circinus_OIII.fits) (6.8 MB), [circinus_OIII_mask.fits](https://github.com/editeodoro/bbarolo2/raw/main/examples/circinus_OIII_mask.fits) (1.7 MB), [circinus_WINDFIT_GLOBAL.par](https://github.com/editeodoro/bbarolo2/raw/main/examples/circinus_WINDFIT_GLOBAL.par), [circinus_mask.py](https://github.com/editeodoro/bbarolo2/raw/main/examples/circinus_mask.py) |

The cube holds more than the wind: [OIII] from the disk of the galaxy over the whole field, the lines
[Fe III] 4987 (at −1170 km/s from [OIII] 5007) and He I 5016 (at +530 km/s, faint), and the gas on the far side
of the nucleus. WINDFIT compares the cone with every voxel it is given, and the empty voxels are what tell a cone
that is too large: everything else in the cube pulls the cone towards it (fitted to the whole cube, the cone
becomes a wide blob that covers the disk). A mask keeps the fit to the cone.

## The mask

`circinus_mask.py` makes `circinus_OIII_mask.fits`, a cube of 0 and 1 with the shape of the data:

- the channels from −900 to +800 km/s around [OIII] 5007 ([Fe III] is out);
- the spaxels within 65° of the axis of the cone, at position angle −65° on the sky from the nucleus (the
  direction of Marconcini et al. 2023, 10° wider on each side than their cone: the bright filaments extend north of
  it);
- of these, the spaxels where the line is bright: a peak signal-to-noise ratio of at least 30 in the cube smoothed to
  1.8 arcsec. The faint disk emission around the cone is out.

The mask is spaxels × channels, not the detections: inside it the empty voxels are data too. WINDFIT uses
it with {ref}`MASK <par-windfit-mask>` `FILE(circinus_OIII_mask.fits)` and
{ref}`LIKEOPTS <par-windfit-likeopts>` `likemask=dilated dilation=0,0`: only the voxels of the mask, not dilated,
are voxels of the likelihood. The noise is measured on the whole cube, outside the mask. (Blanking the cube outside
the mask does the same to the likelihood, but then the noise is measured among the voxels of the cone, and it came
out 25% too high.)

```{figure} /figures/examples/circinus_mask.png
:width: 55%
:alt: The integrated [OIII] map of Circinus with the mask of the fit

The [OIII] emission from −900 to +800 km/s, with the spaxels of the mask (red), the nucleus (cross) and the axis
of the cone (dashed). The white disks are foreground stars, blanked in the cube.
```

::::{dropdown} circinus_mask.py
```{literalinclude} ../../examples/circinus_mask.py
:language: python
```
::::

## Running the example

::::{tab-set}
:sync-group: lang

:::{tab-item} Parameter file
:sync: par

```{literalinclude} ../../examples/circinus_WINDFIT_GLOBAL.par
:language: text
```

```bash
BBarolo -p circinus_WINDFIT_GLOBAL.par
```
:::

:::{tab-item} Python
:sync: py

```python
from bbarolo import WindFit

f = WindFit("circinus_OIII.fits", method="global", mask="FILE(circinus_OIII_mask.fits)", restwave=5006.843,
            redshift=0.001448, specres=0.85, xpos="14:13:09.90", ypos="-65:20:19.77", vsys=0, inc=70, pa=25,
            openang=110, conelen=52, vwind=400, vdisp=0, free="VWIND_SINGLE INC",
            priors="INC=(uniform,30,60) VWIND=(uniform,0,1600)", sampleropts="nstart=6 seed=1", flagerrors=True,
            likeopts="emission=cells cellsize=3.6 datafrac=0.3 cones=approaching likemask=dilated dilation=0,0")
res = f.run(threads=16)     # in memory
res.parameters              # a Table: name, value, err_lo, err_hi
res.shells                  # the shells of the best fit
res.fitter.print_stats()    # the free parameters, with the errors
res.fitter.corner_plot()    # the corner plot (a matplotlib Figure)
res.write()                 # the files of the command line, in output/circinus/
```
:::
::::

On our server, with 16 cores (16 processes for the gradient of the minimiser), the run takes **16 minutes**: 12.5
minutes for 2921 likelihoods of 3192 cells on the 536 000 voxels of the mask (six starts), then the errors.

## The setup

- **The line.** The cube is in wavelengths in air (`CTYPE3 AWAV`): {ref}`RESTWAVE <par-windfit-restwave>` is
  the wavelength of [OIII] 5007 in air, and {ref}`REDSHIFT <par-windfit-redshift>` the redshift of Circinus, so that
  VSYS is the velocity from the systemic one. {ref}`SPECRES <par-windfit-specres>` is 0.85 channels: the line spread
  function of MUSE has σ ≈ 0.9 channels (1.1 Å), and the averaging over a channel (1/12) is already in the model, see
  {doc}`../user-guide/spectral-resolution`.
- **The cone of Marconcini et al. (2023).** WINDFIT describes the wind as {doc}`GALWIND <../tasks/galwind/index>`
  does: the axis of the cones is the rotation axis of a disk of inclination INC and position angle PA, and the cone
  that approaches us projects on the sky along PA − 90°. Their angles become:

  | Marconcini et al. (2023) | WINDFIT |
  |---|---|
  | β = 84° (axis to the line of sight) | INC 84 |
  | γ = 65° (the position angle of the axis on the sky) | PA 25 (the cone along position angle −65°) |
  | θ = 55° (half opening angle) | OPENANG 110 |
  | R₀ = 1.06 kpc | CONELEN 52 arcsec |
  | v₀ = 550 km/s, v_sys = −70 km/s | VWIND, VSYS |

  As in their fit, the apex (XPOS, YPOS: the nucleus), PA, OPENANG and CONELEN are fixed, and there is no
  intrinsic velocity dispersion (VDISP 0): the width of the lines comes from the spread of the directions in the
  cone and from SPECRES. `cones=approaching` leaves the receding cone out of the model: it is behind the disk.
- **Free parameters.** One velocity of the wind for the whole cone (`VWIND_SINGLE`) and the inclination. VSYS is
  fixed at the systemic velocity of the galaxy: with VSYS free, the inclination and VSYS trade off (see below).
  {ref}`PRIORS <par-windfit-priors>`: INC between 30° and 90°, VWIND between 0 and 1600 km/s (the arguments of
  `scipy.stats.uniform` are `loc` and `scale`). {ref}`SAMPLEROPTS <par-windfit-sampleropts>` `nstart=6` starts the minimiser from the initial wind
  and from five random points of the priors.
- **The emission.** `emission=cells` (the default) describes the emission as cells of the cone with free
  emissivities, marginalised. Cells of one beam would be 0.9 arcsec: tens of thousands of cells, several seconds
  per likelihood. `cellsize=3.6` (4 beams) makes 3192 cells. Coarser cells bias the wind low, because a cell spreads
  its emission over the velocities of all its directions and cannot make the narrow lines of a fast cone: with cells
  of 8 and 5.4 arcsec the fit gives 281 and 300 km/s; with cells of 2.7 and 2.0 arcsec (7418 and 17 440 cells, two to
  eight times slower) 339 and 316 km/s, with the same inclination as 3.6 arcsec. The [OIII] of Circinus has a
  signal-to-noise ratio of thousands, and no smooth cone follows its filaments to the noise: `datafrac=0.3` adds a
  model error of 30% of the data to the variance of each voxel.
- **Errors.** {ref}`FLAGERRORS <par-windfit-flagerrors>` computes the errors of the Laplace approximation.

(circinus-results)=
## Results

| Parameter | Best fit |
|---|---|
| VWIND | 338 ± 4 km/s |
| INC | 71.6° ± 0.4° |
| EMSCALE, EMCLUMP, EMSLOPE | 117, 2.1, 3.4 (the prior of the emission: its mean, scatter and radial slope) |

The six starts end at the same point. The errors of the Laplace approximation are those of this likelihood, model
error included: they are much smaller than the differences between the descriptions of the emission below, which
are the real uncertainty of the fit.

Marconcini et al. (2023) found a faster wind with an axis closer to the plane of the sky: v₀ = 550 ± 20 km/s,
β = 84° ± 2°. Their data were not this cube: they fitted a model of the line, the sum of the Gaussian components
fitted to [OIII] in each spaxel, with the line spread function removed and above a signal-to-noise ratio of 3. The integrated [OIII] of this cube, in the
mask, has its 1st and 99th percentile velocities at −451 and +235 km/s; a cone with their parameters spreads from
−556 to +407 km/s on this cube, 40% wider (and the statistic of MoKA-3D rejects it, see below). The highest
velocities of the data are not in the model at any radius: see {ref}`circinus-high-velocities`.

```{figure} /figures/examples/circinus_maps.png
:width: 70%
:alt: Moment maps of the data, the model and the residuals of the cone of Circinus

The moment maps (`_maps_wind.pdf`) of the data in the mask, of the model and their difference. The model is
smoother than the filaments of the data. The dispersion of the data is noisy: the mask holds every channel of the
window, the empty ones too.
```

```{figure} /figures/examples/circinus_pv.png
:width: 70%
:alt: Position-velocity diagrams of Circinus along and across the cone, data and model

Position–velocity diagrams along the axis of the cone and across it, half way along (`_pv_wind.pdf`): data in grey
with blue contours, model in red. The model has the bright part of the lines but not their faint wings, from
−850 to +500 km/s, at any radius, and it is fainter than the bright line at 0 to +200 km/s along the axis between
10 and 25 arcsec.
```

```{figure} /figures/examples/circinus_pvslices.png
:alt: Position-velocity slices through the cone of Circinus, data and model

Position–velocity slices through the cone (`_pvslices_wind.pdf`): through the nucleus at 10 angles, 18° apart from
the projected axis of the cone (top), across the axis at 5 distances from the nucleus and parallel to it at 5
offsets (bottom), with their paths on the total intensity map. Data in blue, model in red contours.
```

```{figure} /figures/examples/circinus_chanmaps.png
:width: 70%
:alt: Channel maps of Circinus, data and model

Channel maps of the data (blue, the whole field: outside the mask, the [OIII] of the disk) and of the model (red,
the cone only). The channels beyond about −450 and +300 km/s are empty in the model: a cone at 338 km/s, inclined
by 72°, has line-of-sight velocities from −324 to +201 km/s, broadened by SPECRES.
```

```{figure} /figures/examples/circinus_corner.png
:width: 60%
:alt: Corner plot of the Laplace approximation of the fit of Circinus

The corner plot (`_wind_corner.pdf`): the draws of the Laplace approximation around the best fit.
```

(circinus-high-velocities)=
## The highest velocities

The lines of the cone have a bright core and faint wings, a few per cent of the flux, out to −850 and +500 km/s, at
every radius. A cone of one radial velocity V and half opening angle 55°, inclined by INC, has line-of-sight
velocities between −V cos(INC − 55°) and −V cos(INC + 55°): to reach the wings it needs V of about 550 km/s and an
axis close to the plane of the sky, which spreads the cores too. Two more fits (with the same file, as in
{ref}`circinus-other-emission`) show the trade-off:

| Fit | VWIND (km/s) | VDISP (km/s) | INC | log-likelihood |
|---|---|---|---|---|
| one VWIND (above) | 338 | 0 | 71.6° | |
| VWIND fixed at 550 (`VWIND=550 FREE=INC`) | 550 | 0 | 81.9° | −535 |
| VDISP free (`FREE=VWIND_SINGLE INC VDISP_SINGLE`) | 148 | 95 | 65.2° | +1129 |

With VWIND fixed at 550 km/s the inclination goes to 82°, close to the 84° of Marconcini et al. (2023), and the
wings beyond 10 arcsec are reproduced, but the cores are not, and the likelihood is much lower. With VDISP free the
cone becomes slow and turbulent: the cores are better, the wings still missing.

The likelihood of WINDFIT weighs how the flux is distributed in velocity: the bright cores, most of the flux, set
the wind. The statistic of MoKA-3D used by Marconcini et al. (2023) measures something else. κ is the flux of the
voxels that the cone does not reach, so it rewards a cone that reaches the emission, whatever it predicts elsewhere,
and its best cone is the fastest one that the cut on the 1st and 99th percentile velocities of the integrated line
allows: the velocity of the envelope of the line, its wings. On this cube `emission=moka` gives 384 km/s (below); in
a test with their recipe (a model of the line without noise, the sum of the Gaussian components fitted in each
spaxel with the line spread function removed, and VSYS free) it gives 408 km/s at INC 77°. Their 550 km/s needs a
wider integrated line than this cube has (see {ref}`circinus-results`).

```{figure} /figures/examples/circinus_spectra.png
:alt: Spectra of Circinus in rings around the nucleus, data and three models

The integrated spectra in the mask, in rings around the nucleus (logarithmic scale): the data (black), the fit with
one VWIND (red), with VWIND fixed at 550 km/s (green) and with VDISP free (purple). Within 3 arcsec, the bright
nucleus is not in any model.
```

(circinus-other-emission)=
## Other descriptions of the emission

WINDFIT has other descriptions of the emission ({ref}`LIKEOPTS <par-windfit-likeopts>` `emission`, see
{doc}`../tasks/windfit/algorithm`). The same parameter file runs them with another `LIKEOPTS` on the command line
(parameters after the file replace those in it):

```bash
BBarolo -p circinus_WINDFIT_GLOBAL.par FLAGERRORS=false OUTFOLDER=output/circinus_voxel/ \
        "LIKEOPTS=emission=voxel cones=approaching likemask=dilated dilation=0,0"
BBarolo -p circinus_WINDFIT_GLOBAL.par FLAGERRORS=false OUTFOLDER=output/circinus_shells/ \
        "LIKEOPTS=emission=shells datafrac=0.3 cones=approaching likemask=dilated dilation=0,0"
BBarolo -p circinus_WINDFIT_GLOBAL.par FLAGERRORS=false OUTFOLDER=output/circinus_moka/ \
        "LIKEOPTS=emission=moka cones=approaching likemask=dilated dilation=0,0"
```

| `emission` | VWIND (km/s) | INC | |
|---|---|---|---|
| `cells` (above) | 338 | 71.6° | the emissivities of 3192 cells, marginalised |
| `voxel` | 210 | 66.3° | the detections (S/N > 3) and the voxels without; EMCLUMP at the edge of its prior |
| `shells` | 233 | 84.6° | one emissivity per shell; three of the six starts end at worse minima |
| `moka` | 384 | 71.4° | κ of MoKA-3D (no likelihood: no errors); two starts end among rejected models |
| MoKA-3D, Marconcini et al. (2023) | 550 | 84° | their data (see above), v_sys = −70 km/s |

They took 4 to 8.5 minutes each. The four agree on a cone inclined by 66° to 85°, with a wind of 200 to 400 km/s:

- **`voxel`**, the weighting of MoKA-3D turned into a likelihood (the probability of each detection, and of each
  voxel without one), gives the slowest wind; its clumpiness EMCLUMP is at the edge of its prior, a sign that its
  description of the emission is strained.
- **`shells`** gives each shell one emissivity, the same in every direction: it cannot follow the filaments, and its
  solutions depend on the start.
- **`moka`**, the statistic κ of MoKA-3D (the flux of the bright voxels the clouds of the cone miss), only rewards a
  cone that reaches every bright voxel, within the 1st and 99th percentile velocities of the data: the fastest
  wind of the four. Its solution has percentiles −470 and +223 km/s (the data: −451 and +235). κ never penalises
  a cone where the data are empty, so it depends on the mask (with a wider mask the cone widens to cover the flux in
  it) and needs the opening angle fixed (free, the cone opens until it reaches every voxel).

```{figure} /figures/examples/circinus_compare.png
:alt: Position-velocity slices of Circinus with the cones of the four descriptions of the emission

Position–velocity slices of the data in the mask (grey, blue contours) along the axis of the cone and across it 10
arcsec from the nucleus, one beam wide, with the model cubes of the four fits (red contours, at the levels of the
data). The models of `voxel` and `moka` are the data where their cone reaches, as in MoKA-3D, so the dashed line
shows the cone itself: a uniform cone with the same geometry and wind (made with GALWIND). The model of `voxel`
keeps the data where the expected emission of its cone is above 1/1000 of its peak: with the emission falling as
r^−4.35 (EMSLOPE), only within 3 arcsec of the nucleus.
```

**VSYS free.** With VSYS free as well (`FREE=VWIND_SINGLE INC VSYS`, PRIORS VSYS between −300 and +300 km/s), the
fit with cells goes to VSYS = +288 km/s, INC 46° and VWIND 588 km/s, with a log-likelihood higher by 859. The inclination and VSYS trade off: a cone closer to the line of sight approaches faster, and a
redshifted VSYS brings it back. The data alone do not fix VSYS: here it is the systemic velocity of the galaxy.

## Going further: the velocity as a function of radius

A single VWIND is the velocity of the whole cone. `VWIND_FUNC` makes it a function of radius, by default an
arctangent that rises to a terminal velocity p1 with a scale radius p2:

```bash
BBarolo -p circinus_WINDFIT_GLOBAL.par FLAGERRORS=false OUTFOLDER=output/circinus_func/ "FREE=VWIND_FUNC INC" \
        "PRIORS=INC=(uniform,30,60) VWIND_P1=(uniform,0,1600) VWIND_P2=(uniform,1,50)"
```

The best fit has p1 = 418 km/s and p2 = 5.5 arcsec: the wind accelerates from 80 km/s near the nucleus to 300 km/s
at 12 arcsec and 360 km/s at the edge of the field (about 26 arcsec along the axis; beyond, the profile is the
function, not the data). The inclination barely changes (70.4°), and the log-likelihood is higher by 96 than with
one velocity. One velocity per shell (`FREE VWIND`) has too much freedom: the velocities jump by hundreds of km/s
from a shell to the next.

```{figure} /figures/examples/circinus_func_shells.png
:alt: The velocity of the wind of Circinus as a function of radius

The shells of the fit with `VWIND_FUNC` (`_wind_shells.pdf`): the velocity of the wind, the dispersion and the
rotation (fixed at 0) against radius, and the flux of the model in radial bins.
```

```{figure} /figures/examples/circinus_func_pv.png
:width: 70%
:alt: Position-velocity diagrams of Circinus with the model of a wind accelerating with radius

The position–velocity diagrams of the fit with `VWIND_FUNC`, as for the fit with one velocity above: the wind is
slow near the nucleus, where the lines of the data are the widest, and faster outwards; the faint wings of the lines
are missing at every radius, as with one velocity.
```

```{figure} /figures/examples/circinus_func_maps.png
:width: 70%
:alt: Moment maps of Circinus with the model of a wind accelerating with radius

The moment maps of the fit with `VWIND_FUNC`.
```

## References

- Marconcini, C., et al. 2023, A&A, 677, A58:
  [MOKA3D: An innovative approach to 3D gas kinematic modelling. I. Application to AGN ionised outflows](https://ui.adsabs.harvard.edu/abs/2023A%26A...677A..58M)
  (the model of the cone of Circinus with MoKA-3D, and its statistic).
- Mingozzi, M., et al. 2019, A&A, 622, A146:
  [The MAGNUM survey: different gas properties in the outflowing and disk components in nearby active galaxies with MUSE](https://ui.adsabs.harvard.edu/abs/2019A%26A...622A.146M)
  (the MUSE observations of Circinus in the MAGNUM survey; the distance).
- {doc}`../tasks/windfit/index` and {doc}`../tasks/windfit/algorithm`: the model, the likelihoods and the options.
