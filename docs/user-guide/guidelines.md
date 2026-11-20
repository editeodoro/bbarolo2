# Guidelines for a successful fit

A 3D fit is only as good as its setup. These are the steps to follow with a new datacube, especially one of low
resolution, before trusting a rotation curve. The {doc}`examples <../examples/index>` show them on real data.

## 1. The observational parameters

The model is convolved with the beam and broadened by the spectral resolution, so both must be right: they are what
corrects the fit for beam smearing.

- **Beam.** BBarolo reads it from the header (BMAJ, BMIN, BPA). If the header has none, as for most optical and
  near-infrared cubes, give it with {ref}`BEAM <par-all-beam>` (the FWHM of the PSF or of the seeing, in arcsec;
  one value for a circular beam).
- **Spectral resolution.** {ref}`SPECRES <par-3dfit-specres>` is the instrumental broadening, in channels: the
  dispersion of the line spread function, less the integration over the channel that the model already does. The
  default, 0.425 channels, fits a cube at the native channels of an instrument with a resolution of one channel.
  A cube averaged over $N$ channels needs 0.425/$N$ (averaging is not in SPECRES), Hanning-smoothed data about 0.8,
  and optical and near-infrared spectrographs usually more (see {doc}`../examples/ngc5633` and
  {doc}`../examples/u3-25160`). How to compute it for your data: {doc}`spectral-resolution`.
- **Spectral axis.** For an axis in wavelength or frequency, give the rest wavelength or frequency of the line
  ({ref}`RESTWAVE <par-galmod-restwave>`, {ref}`RESTFREQ <par-galmod-restfreq>`) and the {ref}`REDSHIFT <par-galmod-redshift>`;
  {ref}`VELDEF <par-all-veldef>` sets the definition of the velocity. Several lines (Hα and [NII]) are fitted together
  with {ref}`RELINT <par-galmod-relint>`.

The log of each run prints what BBarolo understood of the header: check the channel width, the beam and the units
there. Headers without units are in {doc}`troubleshooting`.

## 2. The mask

The mask tells the fit which voxels are emission. It should be as large as possible, to include the faint emission
of the outer disk, but not so large that it is mostly noise. {ref}`MASK <par-3dfit-mask>` `SEARCH` (the default) uses
the detections of the source finder; `SMOOTH&SEARCH` follows faint extended emission better. Try the types with
{doc}`MAKEMASK <../tasks/makemask/index>`, which builds only the mask, and compare each with the channel maps of the
data; when one looks right, give 3DFIT the same parameters. Look also for emission that is not part of the galaxy:
companions, or the HI of the Milky Way in the foreground ({doc}`../examples/ngc2403`).

## 3. The initial values

The parameter space is degenerate, and the fit depends on where it starts, above all on the **inclination**: with
METHOD CLASSIC each ring can move only by {ref}`DELTAINC <par-3dfit-deltainc>` (5° by default) from the initial
inclination, and on simulated disks the fitted inclination stays close to its start
({doc}`../tasks/3dfit/algorithm`). The rotation velocity scales as $1/\sin i$.

BBarolo estimates every parameter that is not given ({ref}`GUESSTYPE <par-3dfit-guesstype>`), and the log marks them
with `(e)` in the table of the initial parameters (with METHOD CLASSIC they are also in `_rings_initial.txt`, and
`res.rings_initial` in Python). The estimates are simple and can fail: always look at them before trusting the fit.
Better, give your own:

- **Centre and inclination** (XPOS, YPOS, INC) from the total intensity map ({doc}`MAPS <../tasks/maps/index>`), or,
  better, from images at higher resolution (optical or near-infrared).
- **Position angle** (PA) from the velocity field: by eye, or with a 2D fit ({doc}`2DFIT <../tasks/2dfit/index>`).
- **Systemic velocity** (VSYS) from the midpoint of the global profile. The estimate of BBarolo is usually good, but
  check it.
- **Rotation velocity and dispersion**: any sensible value with METHOD CLASSIC. With METHOD GLOBAL and BAYESIAN they
  also set the default priors (VROT between 0 and twice the initial value), so give a value that is not too low,
  or set {ref}`PRIORS <par-3dfit-priors>`.

At low inclination (below 30–40°) the inclination is hard to measure from the kinematics: fix it, at a value from
other data ({doc}`../examples/pgc53639`), or, with METHOD BAYESIAN, give it a prior ({doc}`../examples/aless73`). Close to edge-on, it trades off with the thickness of the disk
({doc}`../examples/ngc891`).

## 4. The rings

Rings about one beam wide are independent; narrower rings sample the rotation curve more finely, but neighbouring
rings are correlated. The rings must cover the emission (with METHOD GLOBAL and BAYESIAN, BBarolo warns when
more than 5% of the flux of the total intensity map lies outside the disk of the rings). Give them with {ref}`NRADII <par-3dfit-nradii>` and {ref}`RADSEP <par-3dfit-radsep>`,
or with {ref}`RADII <par-3dfit-radii>` (a list, or `range(Rmin,Rmax,step)`). The innermost ring, within a beam of the
centre, is the least reliable.

## 5. What to fit

Start with few free parameters ({ref}`FREE <par-3dfit-free>`) and add more only if the data need them.

- **Low resolution** (a few beams across the disk): fix the geometry and fit only VROT and VDISP.
- **Well resolved disks**: INC and PA can be free; with METHOD CLASSIC, {ref}`TWOSTAGE <par-3dfit-twostage>`
  (the default) regularises them as functions of radius and fits the velocities again. A warp shows as a PA or an
  INC that change smoothly with radius ({doc}`../examples/pgc53639`).
- **Parameters shared by the disk** (one inclination, one centre, one dispersion) and functions of radius are
  possible with METHOD GLOBAL and BAYESIAN (`NAME_SINGLE`, `NAME_FUNC`), as are the thickness, the outer edge and a
  lagging thick layer.

## 6. The method

{ref}`METHOD <par-3dfit-method>` `CLASSIC` (the default) fits the rings one at a time in seconds: start with it.
`GLOBAL` fits all the parameters together by maximum likelihood, in seconds to minutes: use it for edge-on disks,
whose rings overlap along the line of sight ({doc}`../examples/ugc1281`), for shared parameters, and to test a setup
before sampling. `BAYESIAN` samples the posterior, in minutes to hours: use it when the uncertainties and the
correlations of the parameters matter, as for data of low resolution or low signal-to-noise ratio
({doc}`../examples/u3-25160`). See {doc}`../tasks/3dfit/index`.

## 7. Check the outputs

Look at the model, not only at the rotation curve: the position–velocity diagrams along the major and the minor
axes, the channel maps and the residual maps of the PDF plots. The model should follow the data in all of them; a
systematic pattern of the residuals (one side always brighter, a twist of the velocity field) points to a wrong
centre, position angle, inclination, or to motions that the model does not have. If the model is not good, change
the initial values or the options, and fit again.

For example, the residual velocity field (the velocity field of the data minus that of the model, in the `_maps`
plots) shows most of these errors at a glance. The figure shows it for a simulated disk fitted with VROT and VDISP
free and one of the other parameters wrong.

```{figure} /figures/guidelines_residuals.png
:alt: Residual velocity fields of a simulated disk fitted with a wrong centre, inclination, position angle or systemic velocity, or with motions that the model does not have

Residual velocity fields of 3DFIT on a simulated disk (inclination 60°, PA 40°, a beam of 30 arcsec, the grey dot;
rotation velocity 200 km/s in the outer disk, noise added), with VROT and VDISP free and the other parameters fixed.
The cross and the dashed line are the centre and the major axis of the model. (a) The velocity field of the data.
(b) The right model: only the noise is left. (c, d) The centre is one beam off, along the major or the minor axis:
the residuals are largest near the centre, where the velocity changes fastest. (e) The inclination is 15° too low:
the fit raises VROT to match the major axis, and four lobes remain, of opposite signs in neighbouring quadrants; an
inclination too high gives the same lobes with the opposite signs. (f) The PA is 10° too large: the residuals have
opposite signs on the two sides of the major axis and grow outwards. (g) VSYS is 10 km/s too high: the same residual
everywhere. (h) The gas of the data flows inwards at 25 km/s, and the model has no radial motions: the residuals have
opposite signs on the two sides of the major axis, as in (f), but are largest along the minor axis, down to the
centre. (i) The data have a warp (the PA grows by 15° beyond 200 arcsec): the inner disk is right, the outer disk
has the residuals of a wrong PA.
```

What to do in each case: for (c, d), take the centre from the total intensity map or from an image of higher
resolution, or set XPOS and YPOS free; for (e), give a better inclination (section 3); for (f), set PA free; for
(g), set VSYS free or take it from the global profile; for (h), set VRAD free; for (i), set PA (and INC) free, so
that they can change with radius. The patterns add up when more than one parameter is wrong, and beam smearing
blurs them in data of low resolution: with a few beams across the disk, look also at the position–velocity diagrams.

The position–velocity diagrams along the major and the minor axes (the `_pv` plots) show the same errors in another
way, and tell a wrong PA from radial motions better than the velocity field does.

```{figure} /figures/guidelines_pv.png
:alt: Position-velocity diagrams along the major and the minor axes of the simulated disk, fitted with a wrong centre, systemic velocity or position angle, or with an inflow that the model does not have

Position–velocity diagrams of the simulated disk of the figure above, along the major axis (first and third rows)
and the minor axis (second and fourth rows) of the model, through its centre, as in the `_pv` plots: the data in
grey with blue contours, the model in red, the fitted rotation velocities, projected, in gold. The velocities are
relative to the VSYS of the model, and the diagrams of the minor axis are on a smaller scale. (a) The right model.
(b) The centre is one beam off along the major axis: on the major axis, the data cross VSYS one beam from the centre
of the model; the minor axis now crosses the receding half of the disk, and all its data are at higher velocities.
(c) The centre is one beam off along the minor axis: the major axis hardly changes, and the data of the minor axis
are shifted by one beam (the residual velocity field shows this error better). (d) VSYS is 10 km/s too high: the
data lie 10 km/s below the model in both diagrams, and the two halves of the rotation curve are not symmetric about
VSYS. (e) The PA is 10° too large: the major axis hardly changes, as VROT absorbs the error, but the data of the
minor axis are tilted: their velocity changes smoothly through the centre and grows outwards. (f) The gas flows
inwards at 25 km/s: the major axis is unchanged, and the two halves of the minor axis are shifted in opposite
directions, by VRAD sin i (22 km/s), with a step at the centre.
```

The step of (f) is sharp because the inflow of the simulated disk is the same at all radii: radial motions that grow
outwards, as the rotation does, look more like a wrong PA. Set PA and VRAD free in turn to tell them apart.

Read the errors for what they are. The errors of METHOD CLASSIC ({ref}`FLAGERRORS <par-3dfit-flagerrors>`) measure
how fast the residuals of a ring grow when one parameter moves, the others fixed; those of GLOBAL and BAYESIAN include
the correlations, but none includes the systematic errors of the model. A velocity dispersion smaller than the
spectral resolution and the channel width is hard to measure, whatever its error says.

## 8. Test the setup on a simulated cube

To know what a fit can recover at the resolution and signal-to-noise ratio of your data, build a model galaxy on the
grid of your cube, with known parameters and the same noise, and fit it with the same setup:
{doc}`../examples/simulated`.
