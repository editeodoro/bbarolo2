# Conventions

The conventions BBarolo follows for positions, angles, rings, units and spectral axes. They are the same
for every task, on the command line and in Python. Where they differ from BBarolo 1.8 (C++) or from other
programs, the text says so.

## Geometry of the disk

```{figure} /figures/conventions_ring.png
:width: 55%
:alt: A tilted ring on the sky, with North up, East to the left, the position angle and the receding half

A ring on the sky, here with a position angle of 40 degrees and an inclination of 60 degrees. The red half
of the major axis is the receding half, and the position angle is measured from North to it, anticlockwise.
The arrow shows the sense of rotation of the models: clockwise on the sky.
```

**Position of the centre.** {ref}`XPOS <par-3dfit-xpos>` and {ref}`YPOS <par-3dfit-ypos>` are the pixel
coordinates of the centre of the rings, **counted from 0**: the first pixel of the cube is 0, as in Python and
in DS9's "image" coordinates minus one (GIPSY counts from 1). They can also be world coordinates, as degrees
with a `d` or as sexagesimal values (`XPOS 114.2056d`, `XPOS 07:36:49.3`). Right ascensions in sexagesimal
form are in hours. Both must be given in the same form.

```text
XPOS  77                 # pixels
YPOS  77
XPOS  114.20557d         # the same position, in degrees
YPOS  +65.59778d
XPOS  07:36:49.34        # and in sexagesimal form
YPOS  +65:35:52.0
```

A GALMOD model of NGC 2403 with these three forms has its centroid at pixel (77.10, 76.97) in each
case (to 0.001 pixel), and `XPOS 78` moves it by one pixel to 78.10: the pixel (77, 77) is the one with index 77 of the numpy
array `cube[channel, y, x]`.

**Position angle.** {ref}`PA <par-3dfit-pa>` is the position angle of the **receding half of the major
axis**, in degrees, measured **anticlockwise from North**, i.e. through East. North is up and East is to
the left in an ordinary image with a negative `CDELT1`. A model with `PA 0` has its approaching side to
the South; with `PA 90`, to the West. In a GALMOD test, the pixels at 4 or more pixels from the centre in the
direction of the PA had a mean velocity of 199 km/s with `VSYS 132.8`, `VROT 100` and `INC 60`, and the
opposite ones 66 km/s, whatever the PA (0, 45 or 90).

**Inclination.** {ref}`INC <par-3dfit-inc>` is 0 for a face-on disk and 90 for an edge-on disk, in degrees.
It is always between 0 and 90: the near and the far side of the disk are set by the next paragraph, not
by the inclination.

**Sense of rotation.** The velocity field of a rotating disk does not tell which side is the near one: the
two senses of rotation give the same line-of-sight velocities. BBarolo takes the side of the disk at
PA − 90° (West for `PA 0`) as the far one. The gas then rotates **clockwise on the sky**, from the receding
half of the major axis towards the far side. The choice only matters for the motions that are not
circular: with {ref}`VRAD <par-3dfit-vrad>` positive (away from the centre), the minor-axis half of the
far side is redshifted. In a GALMOD model with `PA 0`, `VRAD 50` and no rotation, the western half of the
minor axis had a mean velocity of 175.5 km/s and the eastern one 90.0 km/s, around `VSYS 132.8`. The
vertical motion {ref}`VVERT <par-3dfit-vvert>` is positive away from the observer.

## Rings

The rings are given by their radii, in arcsec from the centre: {ref}`RADII <par-3dfit-radii>`, or
{ref}`NRADII <par-3dfit-nradii>` and {ref}`RADSEP <par-3dfit-radsep>`. With `NRADII` and `RADSEP` the
radii are `RADSEP`/2, 3·`RADSEP`/2, ..., so the first ring is centred at half a ring separation from the
centre: `NRADII 41`, `RADSEP 30` gives 15, 45, ..., 1215 arcsec. Each radius is the **centre** of an
annulus that reaches halfway to the neighbouring radii. The ring tables (`_rings_final*.txt`) list the centre
radius of each ring, and the fit gives one value of each parameter per ring.

A model needs a value of every parameter at every radius, and the way the radii are turned into a disk is the
ring convention {ref}`RINGMODE <par-galmod-ringmode>`. It applies to the final models of 3DFIT, to GALMOD
and to 3DFIT with METHOD GLOBAL or BAYESIAN, not to the fits themselves (the ring tables of 3DFIT do not depend on it):

| `RINGMODE` | The disk of the model | Where |
|------------|-----------------------|-------|
| `CENTRED` (default) | each ring is an annulus centred on its radius; the disk reaches half a ring beyond the first and the last radius, with the values of those rings | as the rings fitted by 3DFIT |
| `NODES` | the radii are the nodes of the profiles: the disk goes from the first to the last radius | GIPSY's GALMOD, TiRiFiC |
| `SHIFTED` | the first and the last radius are moved outwards by `RADSEP`/2, keeping their values | the models of BBarolo 1.x (C++) |

The values are always interpolated linearly in radius between the nodes. The algorithm pages
define the conventions precisely: {ref}`the ring convention of GALMOD <galmod-subrings>` and the
{doc}`3DFIT algorithm <../tasks/3dfit/algorithm>`. As an example, a model with four rings at 30, 90, 150 and
210 arcsec (`RADSEP 60`), face-on, without convolution, filled the following radii:

| `RINGMODE` | Radii with emission (arcsec) | Total flux of the model |
|------------|------------------------------|------------------------:|
| `CENTRED` | 0 to 245 | 35118 |
| `NODES` | 23 to 218 | 26338 |
| `SHIFTED` | 0 to 245 | 35118 |

(the edges are known to the size of a pixel, 16 arcsec, and the fluxes are for `DENS 1`, `NORM NONE`). The
`CENTRED` and `SHIFTED` models cover the same area, from 0 to 240 arcsec, but they differ in the
first and last ring, where `SHIFTED` interpolates between shifted radii. **To compare with C++ results**,
use `RINGMODE SHIFTED`; this is what the comparison scripts of the repository do (see
{doc}`../reference/differences`).

## Units

| Quantity | Unit |
|----------|------|
| radii (`RADII`, `RADSEP`), `Z0`, `ZCYL`, beams (`BEAM`, `NEWBEAM`) | arcsec, or degrees with a final `d` |
| `XPOS`, `YPOS` | pixels (counted from 0), or degrees or sexagesimal |
| `INC`, `PA`, the position angle of the beams | degrees |
| `VSYS`, `VROT`, `VDISP`, `VRAD`, `VVERT` | km/s |
| `DVDZ` | km/s per arcsec |
| `DENS` | flux units of the cube × km/s per arcsec² (see below) |
| `RESTFREQ` | Hz, whatever the unit of the axis |
| `RESTWAVE`, `LINEAR` | the unit of the spectral axis; `LINEAR` is a number of channels |

The arcsec are used everywhere, also when the cube has its axes in degrees (the usual case), arcmin or
arcsec: BBarolo converts the header at reading time. The velocities are in km/s whatever the unit of the
spectral axis (m/s in many cubes).

**DENS.** The face-on surface density of the gas is an integrated flux per unit area. With the default
{ref}`DENSFLUX <par-galmod-densflux>` `true`, the model summed over its pixels and channels, times the
channel width in km/s, is `DENS` times the face-on area of the disk in arcsec². The flux units are
those of the cube (`BUNIT`, e.g. Jy/beam). `DENS` sets the flux of the model only with `NORM NONE`: with
`LOCAL` and `AZIM` the model is scaled to the data. With `DENSFLUX false` (GALMOD with `NORM NONE`) it is a
column density in 10²⁰ atoms/cm². GALWIND's spherical shells always use the column density.

**Z0.** The scale height is in arcsec and the meaning of its value depends on the layer
({ref}`LTYPE <par-galmod-ltype>`). If it is not given, the fits use **one pixel** (a thin disk), where
BBarolo 1.8 used 0. GALMOD needs it.

## Spectral axes

BBarolo reads the spectral axis of the cube (the third one; the fourth if the third is a Stokes axis, in
which case only the first Stokes plane is read) and works in km/s. The axis type is found from `CTYPE3` and
`CUNIT3`:

| Axis | Recognised from | Velocity definition |
|------|-----------------|---------------------|
| velocity (radio) | `CTYPE3` with `VEL` or `VRAD`, or a unit of m/s | radio |
| velocity (optical) | `VOPT` or `FELO` | optical |
| frequency | `FREQ` in `CTYPE3`, or a unit in Hz | `radio`, unless {ref}`VELDEF <par-all-veldef>` says otherwise |
| wavelength | `WAVE`, or units of um, nm, Angstrom, micron | `relativistic`, unless `VELDEF` says otherwise |

A velocity axis keeps its own definition, and `VELDEF` (`radio`, `optical`, `relativistic` or `auto`, the
default) is ignored. For a frequency or a wavelength axis, the default is announced by a warning:

```text
BBAROLO WARNING: I will use a radio definition for velocity. This can be changed by using VELDEF parameter.
```

A frequency cube made from NGC 2403 (the same data, with `CTYPE3 FREQ`, `CUNIT3 Hz` and the frequencies of
the radio velocities) gives with `VELDEF auto` a model identical to that of the velocity cube; the same
cube with `VELDEF optical` has a channel of 5.157 km/s instead of 5.153 km/s, and a model shifted by a
fraction of a channel.

The zero of the velocities, and so the meaning of `VSYS`, is set by the rest frequency or wavelength:

- {ref}`RESTFREQ <par-galmod-restfreq>` (Hz) replaces the rest frequency of the header (`RESTFRQ`, `RESTFREQ` or
  `FREQ0`); without any of them it is 1.4204057 GHz, the HI line, with a warning. The zero is at
  `RESTFREQ`/(1 + {ref}`REDSHIFT <par-galmod-redshift>`).
- {ref}`RESTWAVE <par-galmod-restwave>`, in the units of the axis, sets the zero at `RESTWAVE` × (1 + `REDSHIFT`).
  Without it, the zero is at the reference pixel of the axis and `VSYS` must be given in that frame.
- With `REDSHIFT` right, `VSYS` is close to 0.

Several lines can be modelled together by giving several values of `RESTFREQ` or `RESTWAVE` and the
relative intensities `RELINT`. The velocities refer to the first line, and the moment maps (of MAPS, 3DFIT and
the other tasks) are those of the first line ({doc}`../tasks/maps/algorithm`). The example cube of NGC 2403 has a heliocentric radio velocity axis in m/s,
`CRVAL3` 289.6 km/s and a channel of −5.15 km/s: the channel width is negative, as in many cubes, and BBarolo
handles both orders. The cube information printed at the start of every run lists the axis, the channel
width and the rest frequency.

## Cube headers

- **Spatial axes.** `CDELT1` should be negative (East to the left); a positive one gives the warning
  `HEADER CHECK: BBAROLO expects a negative CDELT1.` Axes in arcsec or arcmin are converted to degrees.
- **Beam.** Taken from `BMAJ`, `BMIN`, `BPA` (in degrees, as in the FITS standard), or from `HISTORY` cards
  (AIPS), or from the keyword {ref}`BEAM <par-all-beam>` (`BEAM bmaj bmin bpa`, which replaces the beam of the header);
  in Python, `beam=(bmaj, bmin, bpa)`, in arcsec and degrees.
- **Flux units.** BBarolo uses `BUNIT` to convert the fluxes of the SEARCH catalogue and of the maps
  to Jy ({ref}`FLUXCONVERT <par-all-fluxconvert>`). It recognises `JY/BEAM` (also `Jy beam-1`, `mJy/beam` and
  `uJy/beam`), `Jy` (divided by the beam area, which needs the beam), `K` and `K (Tmb)` (converted for the
  21-cm line and the beam) and the units of WSRT (`W.U.`). With `NORM LOCAL` or `AZIM` the fits do not need
  a known unit, because the model is scaled to the data. With another unit, the run says
  `No known conversion to Jy.` and keeps the fluxes in the units of the cube.
- The values `NaN` are read as 0 (blanks).

## Masks

A mask is a cube of 0 and 1 that selects the voxels of the galaxy: the fits compare the model with the data
only inside the mask, and the maps use it to select the emission. The default is `SEARCH`, the detections of the
source finder; the other masks (`SMOOTH&SEARCH`, `SMOOTH`, `THRESHOLD`, `NEGATIVE`, `NONE`, `FILE`) are described with the
{ref}`MASK <par-3dfit-mask>` parameter, and the task {doc}`../tasks/makemask/index` writes a mask without
doing anything else.
