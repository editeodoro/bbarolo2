"""
The documentation of the parameters of BBarolo: for each keyword of keywords.KEYWORDS, its full
description (doc, Markdown), the section of the parameter pages of its tasks (group) and, when the
default value is a placeholder such as -1, the default in words (default). Used by help('KEYWORD') and by
the website (docs/_ext/bbparams.py), which lists the parameters of each task by section, in the order
of GROUPS.
"""

# The sections of the parameter pages, in page order
GROUPS = ("Task", "Input", "Rings", "Free parameters and fit", "Bayesian sampling", "Model", "Spectral lines",
          "Mask", "Wind", "Parameter space", "Thresholds", "Merging", "Detections", "New beam", "Old beam",
          "Window", "Maps", "Slice", "Outputs", "Options", "Messages and computation")

DOCS = {
    'FITSFILE': dict(
        group='Input',
        default='none: required (or FITSLIST)',
        doc="""\
The input FITS file: a datacube, or a 2D image for the tasks that accept one (e.g. `SEARCH`,
`2DFIT`, `ELLPROF`). BBarolo reads the data and the header of the primary HDU. Blank values (NaN)
are set to 0, and of a 4D cube only the first plane of the fourth axis (e.g. the first Stokes
plane) is used.

Required by all the tasks, unless `FITSLIST` is given. A relative path is relative to the current
directory, and `~` is expanded. In Python the cube is the first argument of the task classes (a
file name, an astropy HDU or `(data, header)`), not a keyword; `bbarolo.run` accepts `fitsfile=`.
"""),
    'FITSLIST': dict(
        group='Input',
        default='none',
        doc="""\
A text file with a list of FITS files, one per line: all the tasks requested run on each file in
turn, and a file that cannot be read is skipped. When FITSLIST is given, `FITSFILE` is ignored.

The paths are relative to the current directory. Without `OUTFOLDER`, each cube has its own output
folder, `output/<OBJECT>/`. Only for parameter files, the command line and `bbarolo.run` (which then
returns a list of results, one per cube): the Python task classes take one cube.
"""),
    'VERBOSE': dict(
        group='Messages and computation',
        doc="""\
The amount of messages on the screen:

- `0` (or `false`): no messages (apart from some warnings) and no progress bars;
- `1` (or `true`, the default): the main messages, the progress bars and, for 3DFIT, one line per
  ring with its outcome (e.g. why a ring was rejected);
- `2`: everything, as `VERBOSE true` in BBarolo 1.8: the list of the parameters at the start and
  the details of every ring.
"""),
    'OUTFOLDER': dict(
        group='Outputs',
        default='output/<OBJECT>/ in the current directory',
        doc="""\
The directory where the outputs are written. It is created if it does not exist; if it exists and
contains files, they are overwritten (a warning says so). A relative path is relative to the current
directory, and `~` is expanded.

If not given, the outputs go to `output/<OBJECT>/` in the current directory, where `<OBJECT>` is the
OBJECT keyword of the header (`NONE` if it is missing). Every run also writes there
`<prefix>_params.par`, a parameter file with all the parameters of the run. In Python,
`run(write=False)` writes into a temporary folder instead, deleted with the result.
"""),
    'OUTPREFIX': dict(
        group='Outputs',
        default='the OBJECT of the header',
        doc="""\
The prefix of the names of the output files: every output is `<prefix>_<what>`, e.g.
`ngc2403_rings_final1.txt` or `ngc2403_mask.fits`. If not given, it is the OBJECT keyword of the
header, with spaces replaced by `_` and slashes and parentheses by `-` (`NONE` if the keyword is
missing).

Unlike BBarolo 1.8, all the outputs have the prefix followed by an underscore (some names had no
prefix, others no underscore).
"""),
    'LOGFILE': dict(
        group='Messages and computation',
        doc="""\
With `true`, the messages (standard output and errors) are written to a log file in the output
folder, `BBlog_0.txt` (or `BBlog_1.txt`, ... if it already exists), instead of the screen. Progress
bars are turned off. With `VERBOSE`, the log starts with the list of the parameters.
"""),
    'THREADS': dict(
        group='Messages and computation',
        default='the number of cores, at most 16',
        doc="""\
The number of threads for the parallel parts of the tasks: e.g. 3DFIT fits the rings in parallel,
and the models, the convolutions, SPACEPAR, 2DFIT and the RMS map use several threads. For
BAYESIAN3DFIT it is the number of processes of the sampler. Values below 1 are set to 1.

The ring tables of 3DFIT do not depend on THREADS; the Monte Carlo noise of the final MC models does
(one random seed per chunk). In BBarolo 1.8 the default was all the cores.
"""),
    'DEBUG': dict(
        group='Messages and computation',
        doc="""\
Debugging mode. With `true`, 3DFIT and GALMOD write the final model with both normalisations,
`<prefix>_mod_azim.fits` and `<prefix>_mod_local.fits` (with their moment maps and
position-velocity slices), whatever `NORM`.
"""),
    'SHOWBAR': dict(
        group='Messages and computation',
        doc="""\
Whether to show progress bars. They are drawn only on a terminal, and never with `VERBOSE 0` or
`LOGFILE true`.
"""),
    'PLOTS': dict(
        group='Outputs',
        doc="""\
Whether plots are made (with matplotlib), and how many:

- `-1`: no plots; 3DFIT and BAYESIAN3DFIT write no final model (only the ring tables), and no task
  writes the mask;
- `0` (or `false`): no plots;
- `1` (or `true`, the default): the plots of the tasks, e.g. the channel maps, position-velocity
  slices and parameters of 3DFIT, the parameter spaces of SPACEPAR, the detections of SEARCH
  (with `CUBELETS`);
- `2`: also a plot of the initial estimates of the geometry, when BBarolo estimates it,
  `<prefix>_initialguesses.pdf`;
- `3`: also `<prefix>_initialguesses_0.pdf`, the estimates before their refinement.

In Python, `run(write=False)` makes no plots unless PLOTS is given.
"""),
    'BEAMFWHM': dict(
        group='Input',
        default='none (the beam of the header)',
        doc="""\
The FWHM of a circular beam, in arcsec, for a cube without the beam in its header. The beam is
taken, in order of priority, from the header (BMAJ, BMIN and BPA, or HISTORY cards such as those of
AIPS), from `BMAJ`, `BMIN` and `BPA`, and last from BEAMFWHM.

The tasks that need the beam (e.g. 3DFIT, GALMOD, SMOOTH, MAKEMASK, SEARCH with the default
`MINPIX`) stop with an error if none is available (BBarolo 1.8 assumed 30 arcsec). A beam smaller
than a quarter of a pixel is an error (the value is in arcsec, not degrees). In Python,
`beam=(bmaj, bmin, bpa)` replaces the beam of the header.
"""),
    'CHECKCUBE': dict(
        group='Input',
        doc="""\
Checks the cube for bad channels, rows or columns and erases them (sets them to 0) before the other
tasks. A channel (row, column) is bad when it is blank, or when its noise (the MADFM of its
non-blank pixels) is more than 1.5 times that of the whole cube; a row or a column only if more than 70%
of its pixels are not blank. Values:

- `0`: no check (default);
- `1`: channels, rows and columns;
- `2`: only channels;
- `3`: rows and columns;
- `4`: only columns (y);
- `5`: only rows (x).

If something is erased, the statistics of the cube are computed again, and the cleaned cube, without
the bad planes at its edges, is written in `OUTFOLDER` as `<prefix>_clean.fits` (BBarolo 1.8 wrote it
next to the input file). Other values are set to 0.
"""),
    'AUTO': dict(
        group='Task',
        doc="""\
Automated 3DFIT: `true` runs the 3DFIT task, as `BBarolo -f cube.fits`. As always in 3DFIT, the ring
parameters that are not given (the rings, centre, `VSYS`, `INC`, `PA`, `VROT`, ...) are estimated from
the data, with the mask set by `MASK` (by default the detections of the source finder). The
estimates can be poor for data of low resolution or low S/N: check the results.

With `BAYESIAN3DFIT true`, AUTO does not turn on 3DFIT. The Python task classes do not accept it
(use `FitMod3D`, or `bbarolo.run(auto=True)`).
"""),
    'FLUXCONVERT': dict(
        group='Options',
        doc="""\
Whether to convert the fluxes to Jy: the total intensity maps (`TOTALMAP`, and the noise map of
`SNMAP`) are then in Jy/beam km/s, and the integrated fluxes of the SEARCH detections in Jy km/s. The
conversion is known for Jy/beam (also mJy/beam and uJy/beam), Jy per pixel, K (for the 21-cm line)
and W.U.; other units are left as they are, and a total map is then in the unit of the cube times
km/s.

The moment maps of the models of 3DFIT and GALMOD (folder `maps/`) are never converted.
"""),
    'FLAGROBUSTSTATS': dict(
        group='Options',
        doc="""\
Whether to use robust statistics for the noise of the data: the median and the MADFM (the median
absolute deviation from the median, divided by 0.6745 to give a standard deviation) instead of the
mean and the standard deviation. Robust estimators are less sensitive to the emission.

They are used for the noise of the cube (the thresholds of SEARCH, STATS, `SNMAP`, the noise of
3DFIT), the `SMOOTH` and `NEGATIVE` masks and the RMS map. Blank voxels (0 or NaN) are excluded.
"""),
    'CONTSUB': dict(
        group='Task',
        doc="""\
Subtracts the continuum: a polynomial of degree `CONTORDER` is fitted to each spectrum, excluding the
channels of `EXCLWIND`, and subtracted from it. `true` runs it.

It runs before all the other tasks, which then analyse the subtracted cube. The cube is written in
`<prefix>_contsub.fits`. In Python: the `ContSub` class, or `contsub=True` given to any other task.
"""),
    'EXCLWIND': dict(
        group='Options',
        default='none (all the channels are used)',
        doc="""\
The channels excluded from the continuum fit of `CONTSUB`, i.e. those with line emission: single
channels or ranges `first:last` (also `first~last`, both included), separated by spaces or commas, e.g.
`EXCLWIND 20:45 60`. Channels are counted from 0. Any other item, or a channel outside the cube, is an
error, as is a window that excludes all the channels.

If not given, the continuum is fitted to all the channels, including those of the line.
"""),
    'CONTORDER': dict(
        group='Options',
        doc="""\
The degree of the polynomial fitted to each spectrum by `CONTSUB`: `0` a constant, `1` a straight
line (the default), `2` a parabola, ... Negative values are set to 1.
"""),
    'MAKEMASK': dict(
        group='Task',
        doc="""\
Builds the mask of the cube, as set by `MASK`, and writes it in `<prefix>_mask.fits` (1 for the voxels
in the mask, 0 outside; the parameters of the mask are in the HISTORY of the header). `true` runs
it.

With the masks of the source finder (`SEARCH`, `SMOOTH&SEARCH`, `SMOOTHSPEC&SEARCH`) all the
detections are in the mask; the types ending in `LARGEST` keep only the largest. MAKEMASK runs before
the fits and the maps, which then use this mask.
"""),
    'MASK': dict(
        group='Mask',
        doc="""\
The mask of the emission, i.e. the voxels with genuine emission. The fits and models (3DFIT,
BAYESIAN3DFIT, GALMOD, SPACEPAR) compare model and data within it; the maps (not `GLOBALPROFILE`),
2DFIT and ELLPROF use the data within it. It is written in `<prefix>_mask.fits`. Values
(case-insensitive):

- `SEARCH` (default): the detections of the source finder, with the parameters of the SEARCH task
  (`SNRCUT`, `GROWTHCUT`, `MINPIX`, ...);
- `SMOOTH&SEARCH`: as `SEARCH`, on the cube smoothed to `FACTOR` times the beam (or to `BMAJ`, `BMIN`,
  `BPA`);
- `SMOOTHSPEC&SEARCH`: as `SEARCH`, on the cube smoothed spectrally (`WINDOW_TYPE`, `WINDOW_SIZE`);
- `SMOOTH`: the voxels of the smoothed cube (as above) above `BLANKCUT` times its noise;
- `NEGATIVE`: in each channel, the voxels above `BLANKCUT` times the noise measured from the negative
  values of the channel (3DFIT then weights the residuals of each channel by its noise);
- `THRESHOLD`: the voxels above `THRESHOLD`, in the flux units of the cube (0 if not given);
- `NONE`: no mask, all the voxels;
- `FILE(mask.fits)`: a FITS file with as many voxels as the cube, non-zero in the mask (the path is
  relative to the current directory and keeps its case).

With the source finder, the fits, the models, `MASSDENSMAP` and `RMSMAP` use the largest detection,
while `MAKEMASK`, `TOTALMAP`, `VELOCITYMAP`, `DISPERSIONMAP`, 2DFIT and ELLPROF use all of them,
unless the type ends in `LARGEST` (`SEARCHLARGEST`, `SMOOTH&SEARCHLARGEST`,
`SMOOTHSPEC&SEARCHLARGEST`: only the largest). A mask built by a task is used by the next tasks of
the run. If nothing is detected, BBarolo stops with an error.

`_ENLARGE` and a number N after the type (e.g. `SEARCH_ENLARGE5`) widen the mask by N pixels in each
channel (3 without N). An unknown type is replaced by `SMOOTH`, with an error message.
"""),
    'BLANKCUT': dict(
        group='Mask',
        doc="""\
The S/N threshold of the `SMOOTH` and `NEGATIVE` masks (see `MASK`), in units of the noise. `SMOOTH`:
the voxels of the smoothed cube above its middle plus BLANKCUT times its noise (median and MADFM, or
mean and standard deviation, see `FLAGROBUSTSTATS`). `NEGATIVE`: the voxels of each channel above
BLANKCUT times the noise of the channel, measured from its negative values.

The other types of mask do not use it.
"""),
    'STATS': dict(
        group='Task',
        doc="""\
Computes the statistics of the cube: the mean, median, standard deviation, MADFM (also as a standard
deviation, MADFM/0.6745), maximum and minimum of the voxels that are not blank (0 or NaN). They are
printed (with `VERBOSE`) and written in `<prefix>_stats.txt`. `true` runs it.

`ITERNOISE` excludes the emission iteratively. In Python, the result of the `Stats` class has
`stats` and `text`.
"""),
    'GALMODTYPE': dict(
        group='Model',
        default='MC (ANALYTIC for BAYESIAN3DFIT)',
        doc="""\
The kind of disk models. `MC`: the Monte Carlo models of GIPSY's GALMOD and BBarolo 1.8, made of random
clouds (see `CDENS`, `NV`, `ISEED`). `ANALYTIC`: models without Monte Carlo noise, whose clouds lie on a
regular grid in radius, azimuth and height, with line profiles integrated over the channels; they are
usually faster, and their residuals change smoothly with the parameters.

The default is `MC`, except for BAYESIAN3DFIT, whose sampling needs a likelihood without noise
(`ANALYTIC`). In GALWIND only the cylinders can be ANALYTIC. ANALYTIC is new in BBarolo 2.
"""),
    'RINGMODE': dict(
        group='Model',
        doc="""\
How a set of rings becomes a model, in the final models of 3DFIT and the models of GALMOD and
BAYESIAN3DFIT; the values are interpolated linearly between the radii.

- `CENTRED` (the default): each ring is an annulus centred on its radius, reaching halfway to the
  neighbouring rings, as the rings fitted by 3DFIT; the disk extends half a ring beyond the first and
  the last radius, with the values of those rings.
- `NODES`: the values are at the radii, and the disk goes from the first to the last radius, as in
  GIPSY's GALMOD and TiRiFiC.
- `SHIFTED`: the models of BBarolo 1.x: the first and the last radius are moved by `RADSEP`/2, inwards
  and outwards. The disk covers the area of `CENTRED`, but the profiles are bent in the first and last
  ring.

The fits of single rings in 3DFIT and SPACEPAR do not depend on it. `CENTERED` is also accepted.
"""),
    'SEARCH': dict(
        group='Task',
        doc="""\
Runs the source finder, derived from Duchamp (Whiting 2012), on a cube or a 2D image. The voxels
above the primary threshold (`SNRCUT` times the noise of the cube, or `THRESHOLD`) are joined into
objects, in each channel map or along each spectrum (`SEARCHTYPE`); close objects are merged
(`FLAGADJACENT`, `THRESHVELOCITY`), grown down to a secondary threshold (`FLAGGROWTH`) and rejected if
too small (`MINPIX`, `MINCHANNELS`, ...). `true` runs it.

Outputs: `<prefix>_detections.txt` (the detections and their properties), the cube of the detections
(`<prefix>_detections.fits.gz`), a cube with the number of each detection
(`<prefix>_detections_num.fits.gz`), the number of detected channels in each pixel
(`<prefix>_detectmap.fits.gz`) and the moment maps of the detections (`<prefix>_mom0th.fits`,
`<prefix>_mom1st.fits`, `<prefix>_mom2nd.fits`). The same search, with the same parameters, builds
the masks of the source finder (see `MASK`).
"""),
    'SEARCHTYPE': dict(
        group='Options',
        doc="""\
How the objects are found:

- `spatial` (default): in each channel map (pixels connected also diagonally), then merged across
  the channels;
- `spectral`: as runs of consecutive channels along each spectrum, then merged;
- `spatialsmooth`: as `spatial`, on a cube smoothed with the parameters of the SMOOTH task (`BMAJ`,
  `BMIN`, `BPA` or `FACTOR`); the noise and the thresholds are those of the smoothed cube, the fluxes
  are measured on the original cube.

Case-insensitive. Another value is replaced by `spatial`, with a message (also when the source finder
only builds a mask, `MASK SEARCH`). A 2D image is always searched spatially.
"""),
    'ITERNOISE': dict(
        group={'STATS': 'Options', '*': 'Thresholds'},
        doc="""\
Whether to estimate the noise of the cube iteratively: the voxels whose absolute value exceeds the
middle plus 3 times the noise are excluded and the statistics computed again, until the noise changes
by less than 0.1% (at most 20 times). This reduces the contribution of the emission to the noise when
it fills a large part of the cube.

It changes the statistics of the cube wherever they are used: the thresholds of SEARCH (and of the
masks of the source finder), STATS, `SNMAP` and the noise of 3DFIT.
"""),
    'FLAGADJACENT': dict(
        group='Merging',
        doc="""\
How close two objects must be on the sky to be merged. With `true` (default), they are merged when
they touch (adjacent pixels, also diagonally) and their channels are at most `THRESHVELOCITY` apart.
With `false`, when their pixels are closer than `THRESHSPATIAL` pixels (and within `THRESHVELOCITY`
channels).
"""),
    'THRESHSPATIAL': dict(
        group='Merging',
        default='the beam major axis (pixels)',
        doc="""\
The maximum distance on the sky, in pixels, between two objects to be merged, with
`FLAGADJACENT false`: objects whose pixels are closer than THRESHSPATIAL are merged (if they are also
within `THRESHVELOCITY` channels). Ignored with `FLAGADJACENT true`.

Default: the major axis of the beam in pixels, rounded up.
"""),
    'THRESHVELOCITY': dict(
        group='Merging',
        doc="""\
The maximum separation, in channels, between two objects to be merged: objects that are close on the
sky (see `FLAGADJACENT`) and at most THRESHVELOCITY channels apart become one.
"""),
    'MINCHANNELS': dict(
        group='Detections',
        doc="""\
The minimum number of consecutive channels of a detection: an object is kept only if at least
MINCHANNELS of its channels are adjacent. The default of `MINVOXELS` depends on it.
"""),
    'MINVOXELS': dict(
        group='Detections',
        default='MINPIX × MINCHANNELS',
        doc="""\
The minimum number of voxels of a detection. Default: `MINPIX` times `MINCHANNELS`.
"""),
    'MINPIX': dict(
        group='Detections',
        default='the beam area (pixels)',
        doc="""\
The minimum number of pixels of a detection on the sky (the pixels it covers in any channel).
Default: the area of the beam in pixels, rounded up; this needs the beam (from the header, `BMAJ` and
`BMIN`, or `BEAMFWHM`).
"""),
    'MAXCHANNELS': dict(
        group='Detections',
        default='no limit',
        doc="""\
The maximum number of channels of a detection: objects with MAXCHANNELS channels or more are
rejected. Zero or negative values (the default) set no limit.
"""),
    'MAXANGSIZE': dict(
        group='Detections',
        default='no limit',
        doc="""\
The maximum angular size of a detection, in arcmin: objects whose largest extent along the x or y
axis is MAXANGSIZE or more are rejected. Zero or negative values (the default) set no limit.
"""),
    'REJECTBEFOREMERGE': dict(
        group='Merging',
        doc="""\
Whether the objects that do not meet the requirements of size (`MINPIX`, `MINCHANNELS`, `MINVOXELS`,
`MAXCHANNELS`, `MAXANGSIZE`) are rejected also before they are merged. They are rejected in any case
at the end, after the merging and the growth.
"""),
    'TWOSTAGEMERGING': dict(
        group='Merging',
        doc="""\
Whether objects are merged in two stages: during the search, each new object is merged into the first
object found that is close to it, and all the objects are then merged at the end. With `false`, they
are merged only at the end.

With `false`, set also `REJECTBEFOREMERGE false`: otherwise the pieces of a source in single channels
(or single spectra) are rejected before they can be merged, and usually nothing is found.
"""),
    'SNRCUT': dict(
        group='Thresholds',
        doc="""\
The primary threshold of the source finder, in units of the noise of the cube: the voxels above the
middle plus SNRCUT times the noise start a detection (median and MADFM, or mean and standard
deviation, see `FLAGROBUSTSTATS` and `ITERNOISE`). Ignored if `THRESHOLD` is given.
"""),
    'THRESHOLD': dict(
        group='Thresholds',
        default='none (SNRCUT is used)',
        doc="""\
The primary threshold of the source finder in the flux units of the cube, instead of `SNRCUT`.

It is also the threshold of `MASK THRESHOLD` (the voxels above it; 0, i.e. all the positive voxels, if
not given).
"""),
    'FLAGGROWTH': dict(
        group='Thresholds',
        doc="""\
Whether the detections are grown, after merging, to include the voxels connected to them above the
secondary threshold (`GROWTHCUT` or `GROWTHTHRESHOLD`). Objects joined by the growth are merged.
"""),
    'GROWTHCUT': dict(
        group='Thresholds',
        doc="""\
The secondary threshold for the growth of the detections (`FLAGGROWTH`), in units of the noise of the
cube (as `SNRCUT`). It must not exceed `SNRCUT`: otherwise the growth is turned off, with a message
(for SEARCH and for the masks of the source finder). Ignored if `GROWTHTHRESHOLD` is given.
"""),
    'GROWTHTHRESHOLD': dict(
        group='Thresholds',
        default='none (GROWTHCUT is used)',
        doc="""\
The secondary threshold for the growth of the detections in the flux units of the cube, instead of
`GROWTHCUT`. With `THRESHOLD`, it must not exceed it: otherwise the growth is turned off, with a
message (for SEARCH and for the masks of the source finder).
"""),
    'CUBELETS': dict(
        group='Outputs',
        doc="""\
Whether to write a sub-cube (cubelet) of each detection, in the folder `sources/` of the output
folder: one folder per detection, `<prefix>_<n>` (just `<prefix>` if there is only one), with the
cubelet, its mask, its moment maps (`_mom0.fits`, `_mom1.fits`, `_mom2.fits`), its spectrum
(`_spectrum.dat`: velocity in km/s, flux in Jy) and, with `PLOTS`, plots of the detections.

The cubelets extend `EDGES` pixels beyond the detection on the sky, and 10 channels along the spectral
axis.
"""),
    'EDGES': dict(
        group='Outputs',
        doc="""\
The margin of the cubelets of `CUBELETS` on the sky, in pixels, around each detection (within the
edges of the cube). Along the spectral axis the margin is 10 channels.
"""),
    'SORTSOURCES': dict(
        group='Outputs',
        doc="""\
The order of the detections in the outputs: descending, or ascending with a `-` in front (e.g.
`-vel`). Values (case-insensitive): `xvalue`, `yvalue`, `zvalue` (the centre in pixels), `ra`, `dec`,
`vel`, `w20`, `w50` (the widths of the line at 20% and 50% of the peak), `iflux` (integrated flux),
`pflux` (peak flux), `snr` (peak S/N), `npix` (pixels on the sky), `nvox` (voxels, the default).

`ra`, `dec`, `vel`, `w20`, `w50` and `iflux` need a valid WCS, otherwise the detections are not
sorted. An unknown value leaves them unsorted, with a warning.
"""),
    'PBCORR': dict(
        group='Options',
        doc="""\
Correction of the integrated fluxes of the detections for the attenuation of the primary beam.
Models of the primary beam exist for WSRT, VLA, ATCA and GMRT (from the TELESCOP keyword of the
header); the pointing centre is CRVAL1, CRVAL2. Values:

- `0`: no correction (default);
- `1`: corrected, and 0 beyond the cutoff of the primary beam;
- `2`: corrected, and beyond the cutoff corrected with the response at the cutoff;
- `3`: corrected, and NaN beyond the cutoff;
- `4`: corrected also beyond the cutoff.

For other telescopes the fluxes are not corrected. The widths W20 and W50 are measured on the
uncorrected fluxes.
"""),
    'GLOBALPROFILE': dict(
        group='Task',
        doc="""\
Writes the global profile of the cube, the sum of all the pixels of each channel, in
`<prefix>_spectrum.txt`: two columns, the spectral coordinate (in the units of the spectral axis) and
the sum (in the units of the cube: for a cube in Jy/beam, divide by the area of the beam in pixels to
obtain Jy). `true` runs it.

The whole field is summed: neither `MASK` nor `FLUXCONVERT` apply. In Python:
`Maps(cube, globalprofile=True)`, with `Result.spectrum`.
"""),
    'TOTALMAP': dict(
        group='Task',
        doc="""\
Writes the total intensity map (moment 0) in `<prefix>_map_0th.fits`: the sum of each spectrum within
the mask (`MASK`; the whole spectrum with `MASK NONE`) times the channel width. Units: Jy/beam km/s
with `FLUXCONVERT` (when the flux unit is known), otherwise the unit of the cube times km/s. Pixels
whose sum is zero (e.g. outside the mask) are blank (NaN). With `MAPTYPE GAUSSIAN`, the integral of a
Gaussian fitted to each spectrum. `true` runs it.

`SNMAP` adds its noise and S/N maps. In Python, the `Maps` class (which makes the total, velocity and
dispersion maps when no map is requested) gives `Result.totalmap`.
"""),
    'MASSDENSMAP': dict(
        group='Task',
        doc="""\
Writes the HI mass surface density map in `<prefix>_map_massdens.fits`, in Msun/pc2: 8794 times the
sum over the channels within the mask (`MASK`) of the flux in Jy/beam times the channel width in
km/s, divided by the product of the axes of the beam in arcsec. Helium is not included. `true` runs
it.

Only for the 21-cm line of HI: the cube must be in Jy/beam (also mJy/beam, uJy/beam) or in K
(converted with the Rayleigh-Jeans law at 21 cm); other units are an error. The beam is needed.
"""),
    'VELOCITYMAP': dict(
        group='Task',
        doc="""\
Writes the velocity field (moment 1) in `<prefix>_map_1st.fits`: the intensity-weighted mean velocity
of each spectrum within the mask (`MASK`), in km/s. A frequency or wavelength axis is converted to
velocities as set by `VELDEF`. Pixels without emission in the mask are blank (NaN). With
`MAPTYPE GAUSSIAN`, the centre of a Gaussian fitted to each spectrum. `true` runs it.
"""),
    'DISPERSIONMAP': dict(
        group='Task',
        doc="""\
Writes the velocity dispersion field (moment 2) in `<prefix>_map_2nd.fits`: the intensity-weighted
standard deviation of the velocities of each spectrum around its moment 1, within the mask (`MASK`),
in km/s. It is not corrected for the channel width or the instrumental broadening. With
`MAPTYPE GAUSSIAN`, the dispersion of a Gaussian fitted to each spectrum. `true` runs it.
"""),
    'RMSMAP': dict(
        group='Task',
        doc="""\
Writes the noise of each spectrum in `<prefix>_map_RMS.fits`, in the units of the cube. With a mask
(`MASK` other than `NONE`), it is computed from the channels outside the mask; with `MASK NONE`,
iteratively, discarding the values above 1.5 times the noise until it changes by less than 10%. The
noise is the MADFM/0.6745 around the median, or the standard deviation (`FLAGROBUSTSTATS`). `true`
runs it.
"""),
    'MAPTYPE': dict(
        group='Maps',
        doc="""\
How the maps are derived: `MOMENT` (default), the moments of the spectra, or `GAUSSIAN`, a Gaussian
fitted to each spectrum within the mask, starting from its peak and a dispersion of 10 km/s. With
`GAUSSIAN` the total map is the integral of the Gaussian, the velocity map its centre and the
dispersion map its dispersion; spectra without emission give meaningless values, so use a mask.
Case-insensitive; another value is replaced by `MOMENT`, with a warning.

Only for `TOTALMAP`, `VELOCITYMAP` and `DISPERSIONMAP`.
"""),
    'SNMAP': dict(
        group={'SEARCH': 'Outputs', '*': 'Maps'},
        doc="""\
Whether to compute also the noise and the S/N of the total intensity map, following Verheijen &
Sancisi (2001) and Lelli et al. (2014): the noise of each pixel is that of the sum of the channels in
the mask at that pixel (all the channels with `MASK NONE`), from the noise of the cube, corrected for
the continuum subtraction (`CONTCHANS`) and the spectral taper (`TAPER`). It writes
`<prefix>_map_0th_RMS.fits` (the noise, in the units of the total map) and `<prefix>_map_0th_SN.fits`
(the S/N).

With `TOTALMAP` the maps refer to the total map of the cube; with `SEARCH`, to the total map of the
detections (`<prefix>_mom0th.fits`), written as `<prefix>_mom0th_RMS.fits` and `<prefix>_mom0th_SN.fits`,
and to that of each cubelet (`CUBELETS`).
"""),
    'TAPER': dict(
        group='Maps',
        doc="""\
The spectral taper applied to the data when they were taken, for the noise of `SNMAP`: `uniform` (no
taper, the default), `hanning1` (Hanning taper, all the channels kept) or `hanning2` (Hanning taper,
half of the channels discarded). Case-insensitive; another value is replaced by `uniform`, with a
warning. Only used with `SNMAP`.
"""),
    'CONTCHANS': dict(
        group='Maps',
        default='1000000 (no continuum subtracted)',
        doc="""\
The numbers of line-free channels at the low- and high-velocity ends of the cube that were used to
fit and subtract the continuum, for the noise of `SNMAP`. One value is used for both ends. Large
numbers (the default) mean that no continuum was subtracted. Only used with `SNMAP`.
"""),
    'VELDEF': dict(
        group='Input',
        doc="""\
The definition of velocity used to convert a frequency or wavelength axis into velocities: `radio`,
`optical`, `relativistic` or `auto` (the default; case-insensitive). With `auto`, `radio` for a
frequency axis and `relativistic` for a wavelength axis (a warning says so).

It applies wherever velocities come from the spectral axis: the models of 3DFIT and GALMOD, the maps,
SEARCH. A velocity axis keeps its own definition (radio for VELO or VRAD, optical for VOPT or FELO),
and VELDEF is then ignored. Another value is replaced by `auto`, with a warning.
"""),
    '2DFIT': dict(
        group='Task',
        doc="""\
Fits a tilted-ring model to the velocity field, ring by ring, with weighted least squares (the classic
2D method, as GIPSY's ROTCUR). `true` runs the task. It is fast but does not correct for beam smearing,
so it is unreliable for data at low spatial resolution. If `FITSFILE` is a cube, the velocity field is
its first moment within the mask (`MASK`); if it is a 2D map, the map is the velocity field. The field
is written as `<prefix>_map_1st.fits`.

The rings come from `RADII`, or `NRADII` and `RADSEP`, with `XPOS`, `YPOS`, `VSYS`, `VROT`, `VRAD`,
`INC` and `PA`: those not given are estimated as in `3DFIT` (`GUESSTYPE`), and `VRAD` is 0. Each ring
is centred on a radius, with a width equal to the distance to the next radius, so N radii give N−1
rings. `FREE` selects the free parameters among `VROT`, `INC`, `PA`, `XPOS`, `YPOS`, `VSYS`, `VRAD`
and `ALL` (the others are ignored); a free centre is fitted in each ring.

The pixels within 15° of the minor axis are not used (`DVDZ`, if given, sets this angle, as in C++),
and the others are weighted as set by `WFUNC`. `SIDE` `A` or `R` fits one half of the galaxy, with
`VSYS`, `XPOS` and `YPOS` fixed. The rings are fitted in parallel on `THREADS` threads.

The outputs are the table `<prefix>_2dtrm.txt` (radius in pixels and arcsec, VSYS, VROT, VEXP (the
radial velocity), PA, INC, XPOS and YPOS of each ring) and the model velocity field
`<prefix>_2d_mod.fits`.
"""),
    'ELLPROF': dict(
        group='Task',
        doc="""\
Derives the radial profile of the intensity of a galaxy in elliptical rings (as GIPSY's ELLINT).
`true` runs the task. If `FITSFILE` is a cube, the profile is that of its total intensity map within
the mask (`MASK`); if it is a 2D map, that of the map itself.

The rings are `RADII`, or `NRADII` and `RADSEP`, with `XPOS`, `YPOS`, `INC` and `PA`, which are
required (they are not estimated). `INC` and `PA` can change from ring to ring; the centre is the mean
of `XPOS` and `YPOS`. Each ring is centred on a radius, with a width equal to the distance to the next
radius (to the previous one for the last ring), and the pixels are sampled on 2×2 sub-pixels. `SIDE`
`A` or `R` gives the profile of the approaching or of the receding half only.

The outputs are `<prefix>_ellprof.txt` and the map, `<prefix>_ellprof_map.fits`. For each ring the
table has the radius (arcsec); the sum, mean, median, standard deviation and median absolute deviation
of the map; the number of pixels; the surface density (per arcsec²), its error and its face-on value
(times cos `INC`); the number of blank pixels and the surface densities including them; for maps in
Jy, also the face-on HI column density (1E20 atoms/cm2) and mass surface density (Msun/pc2). In
BBarolo 1.8 the files were `<prefix>_densprof.txt` and `<prefix>_densmap.fits`.
"""),
    '3DFIT': dict(
        group='Task',
        doc="""\
Fits a 3D tilted-ring model to the datacube of `FITSFILE`: the rings are fitted one at a time, in
parallel threads, by comparing the data in the mask (`MASK`) with models of the ring convolved with the
beam (Di Teodoro & Fraternali 2015). `true` runs the task.

The ring parameters that are not given are estimated from the data (`GUESSTYPE`) or take their
defaults, and `FREE` lists those that are fitted. With `TWOSTAGE`, the geometry is regularised and the
rings are fitted again. The outputs are the ring tables `<prefix>_rings_initial.txt` (the values given
or estimated, before the 2D refinement of PA and VROT, in `<prefix>_2drings.txt`, and before the
beam-wide rings of `REVERSE`), `<prefix>_rings_final1.txt` and `<prefix>_rings_final2.txt`, the model
`<prefix>_mod_<norm>.fits` with its moment maps and position-velocity slices, the radial profile
`<prefix>_densprof.txt` and the plots.

3DFIT does not run with `BAYESIAN3DFIT`, which turns it off; when both 3DFIT and `GALMOD` are set,
GALMOD is turned off.
"""),
    'BAYESIAN3DFIT': dict(
        group='Task',
        doc="""\
Fits a 3D tilted-ring model to the datacube by sampling the posterior distribution of the free
parameters with nested sampling (see `SAMPLER`), instead of minimising the residuals as `3DFIT` does.
`true` runs the task. It needs the free parameters in `FREE` (e.g. `FREE VROT VDISP_SINGLE INC_SINGLE`)
and the Python package of the sampler (`pip install "bbarolo[bayesian]"` installs dynesty and nautilus).

The rings are given as in `3DFIT`. The parameters that are not free stay fixed, and the geometry that
is not given (rings, `XPOS`, `YPOS`, `VSYS`, `VROT`, `INC`, `PA`) is estimated as in `3DFIT`
(`GUESSTYPE`); `VDISP` is 8 km/s, `Z0` one pixel, `VRAD` 0 and `DENS` 1 when not given. The models are
ANALYTIC by default (`GALMODTYPE`), with one set of rings (no second stage), and `NORM` is not used:
unless `DENS` is free, the density profile is deconvolved from the total intensity map.

The likelihood is Gaussian, with the noise rms measured outside the mask, on the data in the mask
(`MASK`) dilated by one beam and 3 channels, and the log-likelihood is divided by the correlation
volume of the noise; `LIKEOPTS` changes these choices, `PRIORS` the priors, `SAMPLEROPTS` and
`RUNOPTS` the sampler. The sampling runs on `THREADS` processes.

The best fit is the median of the posterior. BBarolo writes the outputs of `3DFIT` for it (model,
maps, PV diagrams, `<prefix>_rings_final1.txt` with the errors from the 16th and 84th percentiles,
`<prefix>_densprof.txt` and, when `DENS` is not free, `<prefix>_densprof_deconvolved.txt`), the samples
in `<prefix>_<sampler>_results.npy`, with `PLOTS` the corner plot `<prefix>_corner.pdf` and the radial
profiles `<prefix>_profiles.pdf`, and with `VERBOSE` the statistics of the posterior. `3DFIT` and
`GALMOD` cannot run with it and are turned off. In Python the same fit is `bbarolo.BayesianBBarolo`.
"""),
    'GALMOD': dict(
        group='Task',
        doc="""\
Builds a model datacube of a rotating disk from a set of rings, on the grid of `FITSFILE` (its header,
beam and spectral axis), with the algorithm of GIPSY's GALMOD. `true` runs the task.

The rings need `RADII` (or `NRADII` and `RADSEP`), `XPOS`, `YPOS`, `VSYS`, `VROT`, `VDISP`, `INC`, `PA`
and `Z0`, unless they are read from `RINGFILE`; `DENS`, `VRAD`, `VVERT`, `DVDZ` and `ZCYL` are optional.
The model is normalised as set by `NORM` (`AZIM` and `LOCAL` use the data in the mask `MASK`) and
written to `<prefix>_mod_<norm>.fits`, with the rings (`<prefix>_rings_model.txt`), the moment maps and
position-velocity slices of the model and of the data, and `<prefix>_densprof.txt`. A `VSYS` outside the
velocity range of the cube gives a warning.

GALMOD is turned off when `3DFIT` or `BAYESIAN3DFIT` run.
"""),
    'GALWIND': dict(
        group='Task',
        doc="""\
Builds a model datacube of a bi-conical outflow on the grid of `FITSFILE`: two opposite cones with
their apex at (`XPOS`, `YPOS`), along the rotation axis of a disk with inclination `INC` and position
angle `PA`, extending to `HTOT` with opening angle `OPENANG`. The gas moves along the cones at `VWIND`,
rotates around the axis at `VROT` and has velocity dispersion `VDISP`. `true` runs the task.

The cones are made of cylinders (`WTYPE` 0) or of spherical shells (`WTYPE` 1). The model is convolved
with the beam if `SM` is true and written to `<prefix>_wind.fits`, with its moment maps
(`<prefix>_wind_0mom.fits`, ...), a position-velocity slice along the axis (`<prefix>_wind_pv.fits`) and
plots. `XPOS`, `YPOS`, `VSYS`, `VDISP`, `INC`, `PA`, `DENS`, `OPENANG`, `VWIND` and `HTOT` are
required.
"""),
    'SPACEPAR': dict(
        group='Task',
        doc="""\
Computes the residuals of 3DFIT over a grid of two ring parameters, `P1` and `P2`, in each ring: it
shows the shape of the parameter space and whether the minimum found by 3DFIT is the right one. It can
be slow. `true` runs the task.

The rings are those of 3DFIT (given or estimated, without the 2D refinement of PA and VROT that 3DFIT
does), from the ring `STARTRAD`; the residuals use `FTYPE`, `WFUNC`, `BWEIGHT`, `SIDE` and `NORMALCUBE`,
with the normalisation of the first stage of 3DFIT (`NORM`). The outputs are
`<prefix>_spacepar.fits`, a cube with, in each ring, the relative difference of the residuals from their
minimum, |f − f_min|/f_min, and its plot `<prefix>_spacepar.pdf`.
"""),
    'NRADII': dict(
        group='Rings',
        default='estimated from the data (GALMOD and ELLPROF: required without RADII)',
        doc="""\
The number of rings. Without `RADII`, the rings are `RADSEP` arcsec apart, at the radii `RADSEP`/2,
3·`RADSEP`/2, ..., (NRADII − 1/2)·`RADSEP`. With `RADII`, or with ring parameters given as lists of
values, a smaller NRADII keeps only the first NRADII rings (in Python, it must equal the number of
values).

When neither NRADII nor `RADII` is given, 3DFIT, BAYESIAN3DFIT, SPACEPAR and 2DFIT estimate the number
of rings from the extent of the emission (see `GUESSTYPE`). GALMOD and ELLPROF need `NRADII` and
`RADSEP`, or `RADII`.
"""),
    'RADII': dict(
        group='Rings',
        default='from NRADII and RADSEP',
        doc="""\
The radii of the rings (arcsec), in increasing order, as an alternative to `NRADII` and `RADSEP`. It can
be a list of values separated by spaces or commas (`30 60 90 120`; in Python, a list or an array);
`Rmin~Rmax:step`, e.g. `60~180:60` for 60, 120 and 180 arcsec (up to Rmax, included); or a column of a text
file, `file(name,column)` or `file(name,column,rows)` (see `VROT`).

The separation of the rings is then the mean difference between consecutive radii (`RADSEP` is
ignored, except with a single radius R, where it is `RADSEP` if given, otherwise R/2). In the fits,
each ring is an annulus reaching halfway to the neighbouring radii (see also `RINGMODE`).
"""),
    'RADSEP': dict(
        group='Rings',
        default='estimated from the beam',
        doc="""\
The separation between consecutive rings (arcsec), which is also the width of each ring, used with
`NRADII` when `RADII` is not given (with `RADII` it is ignored, except for a single radius).

If it is not given, 3DFIT, BAYESIAN3DFIT, SPACEPAR and 2DFIT use the FWHM of the major axis of the
beam, or half of it when that gives fewer than 20 rings across the emission. Given without `NRADII`, the
number of rings is estimated so that rings of RADSEP cover the emission. GALMOD and ELLPROF need it
(or `RADII`).
"""),
    'VROT': dict(
        group={'GALWIND': 'Wind', '*': 'Rings'},
        default='estimated from the data (GALMOD: required; GALWIND: 0)',
        doc="""\
The rotation velocity of the rings (km/s): the initial value of the fits, or the value of the model
(GALMOD). Like the other ring parameters, it can be one value for all the rings; one value per ring,
separated by spaces or commas (in Python, a list or an array); or a column of a text file, `file(name,column)`
with the column counted from 1, `file(name,column,N)` from row N to the end, or
`file(name,column,N:M)` rows N to M. The rows are the lines of the file, counted from 1, and the lines
without a number in that column are skipped. A column of a BBarolo ring file (e.g.
`<prefix>_rings_final1.txt`) must be that of the same quantity, otherwise it is an error. Lists of
values must have the same length (in a parameter file the shortest one sets the number of rings).

The models interpolate the ring values linearly in radius. When VROT is not given, 3DFIT,
BAYESIAN3DFIT, SPACEPAR and 2DFIT estimate it from the width of the global profile, deprojected with the
inclination (see `GUESSTYPE`); GALMOD needs it. In 3DFIT the fitted VROT of each ring stays within
`DELTAVROT` of its initial value.

In GALWIND, VROT is the rotation of the outflowing gas around the axis of the cones (default 0): one
value, or one value per cylinder or shell (a list or a file column).
"""),
    'VRAD': dict(
        group='Rings',
        default='0',
        doc="""\
The radial velocity in the plane of the disk (km/s), the same at all the azimuths of a ring: positive
values are motions away from the centre. A cloud at azimuth θ from the receding major axis gets
VRAD·sin(θ)·sin(`INC`) in its line-of-sight velocity. One value, one value per ring or
`file(name,column)` (see `VROT`); 0 if not given.

It is fitted when `FREE` includes it: by 3DFIT within ±500 km/s, and by 2DFIT, where it is also called
VEXP.
"""),
    'Z0': dict(
        group='Rings',
        default='one pixel (GALMOD: required)',
        doc="""\
The scale height of the gas layer (arcsec), i.e. the thickness of the disk perpendicular to its plane;
its meaning depends on `LTYPE` (e.g. the dispersion of a Gaussian layer, the half-thickness of a box).
One value, one value per ring or `file(name,column)` (see `VROT`).

When it is not given, 3DFIT, BAYESIAN3DFIT, SPACEPAR and the estimates of `GUESSTYPE` `FIT` use one
pixel, a thin disk; GALMOD needs it. When `FREE` includes Z0, 3DFIT fits it between 0 and 1000 arcsec.
BBarolo 1.8 used 0 (a disk with no thickness) when Z0 was not given.
"""),
    'VVERT': dict(
        group='Rings',
        default='0',
        doc="""\
A velocity perpendicular to the plane of the disk (km/s), the same above and below the plane: it adds
VVERT·cos(`INC`) to the line-of-sight velocity of all the gas (positive values: away from the observer).
One value, one value per ring or `file(name,column)` (see `VROT`); 0 if not given.

BAYESIAN3DFIT fits it when `FREE` includes it. The models of 3DFIT and SPACEPAR also use the value
given, fixed.
"""),
    'DVDZ': dict(
        group='Rings',
        default='0',
        doc="""\
The vertical gradient of the rotation velocity (km/s per arcsec): at heights |z| above `ZCYL` the gas
rotates at `VROT` − DVDZ·(|z| − `ZCYL`), so a positive value gives a thick disk that lags behind the
plane (it needs `Z0` > 0). One value, one value per ring or `file(name,column)` (see `VROT`); 0 if not
given.

BAYESIAN3DFIT fits it when `FREE` includes it; the models of 3DFIT and SPACEPAR use the value given,
fixed. (2DFIT of BBarolo 1.8 read DVDZ as the angle around the minor axis left out of the fit, now
`FREEANGLE`.)
"""),
    'ZCYL': dict(
        group='Rings',
        default='0',
        doc="""\
The height above the plane of the disk (arcsec) where the vertical gradient of rotation `DVDZ` begins:
closer to the plane the gas rotates at `VROT`. One value, one value per ring or `file(name,column)` (see
`VROT`); 0 if not given.

BAYESIAN3DFIT fits it when `FREE` includes it; the models of 3DFIT and SPACEPAR use the value given,
fixed.
"""),
    'VDISP': dict(
        group={'GALWIND': 'Wind', '*': 'Rings'},
        default='8 km/s, or estimated with GUESSTYPE FIT (GALMOD, GALWIND: required)',
        doc="""\
The velocity dispersion of the gas (km/s), the intrinsic width (σ) of the line: the models add in
quadrature the instrumental broadening `SPECRES` (or `LINEAR`). One value, one value per ring or
`file(name,column)` (see `VROT`).

When it is not given, 3DFIT, SPACEPAR and BAYESIAN3DFIT start from 8 km/s; with `GUESSTYPE` `FIT`,
3DFIT and SPACEPAR use instead an estimate from the second moment of the emission (in the outer half of
the disk, corrected for the channel width). GALMOD and GALWIND need it (in GALWIND: one value, or one
value per cylinder or shell, as a list or a file column). When `FREE` includes VDISP, 3DFIT fits it between
`MINVDISP` and `MAXVDISP`.
"""),
    'XPOS': dict(
        group={'GALWIND': 'Wind', '*': 'Rings'},
        default='estimated from the data (GALMOD, ELLPROF, GALWIND: required)',
        doc="""\
The x coordinate of the centre of the rings: a pixel, counted from 0 (unlike GIPSY), or a world
coordinate, in degrees with a `d` (`114.2142d`) or sexagesimal (`07:36:51.4` or `07h36m51.4s`: hours on
a right ascension axis, degrees otherwise). A pixel can also be one value per ring or
`file(name,column)` (see `VROT`); a world coordinate is one position for all the rings, and GALWIND
takes one value. Give `XPOS` and `YPOS` in the same form, both pixels or both world coordinates; when
one is a list or a file column, the other can be a single value in pixels, used for all the rings.

When it is not given, 3DFIT, BAYESIAN3DFIT, SPACEPAR and 2DFIT estimate it from the largest source in
the cube (see `GUESSTYPE`); GALMOD, ELLPROF and GALWIND need it. When `FREE` includes XPOS, 3DFIT fits
it within 10 pixels of the initial values.
"""),
    'YPOS': dict(
        group={'GALWIND': 'Wind', '*': 'Rings'},
        default='estimated from the data (GALMOD, ELLPROF, GALWIND: required)',
        doc="""\
The y coordinate of the centre of the rings: a pixel, counted from 0, or a world coordinate, in degrees
with a `d` (`65.6025d`) or sexagesimal (`+65:36:09` or `+65d36m09s`). The same forms as `XPOS`, and
given in the same form as `XPOS`: one value, one value per ring or `file(name,column)` for pixels, one
position for all the rings for world coordinates.

When it is not given, 3DFIT, BAYESIAN3DFIT, SPACEPAR and 2DFIT estimate it from the largest source in
the cube (see `GUESSTYPE`); GALMOD, ELLPROF and GALWIND need it. When `FREE` includes YPOS, 3DFIT fits
it within 10 pixels of the initial values.
"""),
    'VSYS': dict(
        group={'GALWIND': 'Wind', '*': 'Rings'},
        default='estimated from the data (GALMOD, GALWIND: required)',
        doc="""\
The systemic velocity (km/s), in the velocity frame of the cube: on frequency and wavelength axes it is
relative to the rest frequency or wavelength of the line redshifted by `REDSHIFT` (see `RESTFREQ`,
`RESTWAVE`), with the velocity definition `VELDEF`. One value, one value per ring or
`file(name,column)` (see `VROT`).

When it is not given, 3DFIT, BAYESIAN3DFIT, SPACEPAR and 2DFIT estimate it from the global profile of
the largest source in the cube, the midpoint of its widths at 20% and 50% of the peak (see
`GUESSTYPE`); GALMOD and GALWIND need it. A VSYS outside the velocity range of the cube (often a value
in m/s instead of km/s) is an error for the fits (3DFIT, BAYESIAN3DFIT, SPACEPAR) and a warning for
GALMOD.
"""),
    'INC': dict(
        group={'GALWIND': 'Wind', '*': 'Rings'},
        default='estimated from the data (GALMOD, ELLPROF, GALWIND: required)',
        doc="""\
The inclination of the disk (degrees): 0 face-on, 90 edge-on. One value, one value per ring or
`file(name,column)` (see `VROT`).

When it is not given, 3DFIT, BAYESIAN3DFIT, SPACEPAR and 2DFIT estimate it from the shape of the
emission (see `GUESSTYPE`); GALMOD, ELLPROF and GALWIND need it. The fit of 3DFIT is sensitive to the
initial inclination: when `FREE` includes INC, the value of each ring stays within `DELTAINC` of its
initial value, and between 0 and 90.

In GALWIND, INC is the inclination of the plane perpendicular to the axis of the cones (0: the axis is
along the line of sight).
"""),
    'PA': dict(
        group={'GALWIND': 'Wind', '*': 'Rings'},
        default='estimated from the data (GALMOD, ELLPROF, GALWIND: required)',
        doc="""\
The position angle of the receding half of the major axis (degrees), measured anticlockwise from North,
i.e. towards East. One value, one value per ring or `file(name,column)` (see `VROT`).

When it is not given, 3DFIT, BAYESIAN3DFIT, SPACEPAR and 2DFIT estimate it from the velocity field
(see `GUESSTYPE`); GALMOD, ELLPROF and GALWIND need it. When `FREE` includes PA, 3DFIT keeps the value of
each ring within `DELTAPA` of its initial value.

In GALWIND, PA is the position angle of the plane perpendicular to the cones, whose axis is projected
along PA ± 90°.
"""),
    'DENS': dict(
        group={'GALWIND': 'Wind', '*': 'Rings'},
        default='1 (GALWIND: required)',
        doc="""\
The face-on surface density of the gas in the rings. By default (`DENSFLUX` true) it is an integrated
flux per unit area, in the flux units of the cube times km/s per arcsec²: the model summed over its
pixels and channels, times the channel width in km/s, is DENS times the face-on area of the disk in
arcsec². With `DENSFLUX` false (GALMOD with `NORM` `NONE` only) it is a column density in 1E20
atoms/cm², converted to flux as in GIPSY's GALMOD. One value, one value per ring or `file(name,column)`
(see `VROT`); 1 if not given.

DENS sets the flux of the models only with `NORM` `NONE`. With `AZIM`, the final models of 3DFIT and
GALMOD take the densities from the radial profile of the data (the rings with DENS = 0 stay empty); with
`LOCAL`, each spectrum of the model is scaled to the data. 3DFIT fits DENS only with `FREE` `ALL`.
BAYESIAN3DFIT fits it when `FREE` includes it, otherwise it uses a density profile deconvolved from the
total intensity map of the data.

In GALWIND, DENS is the density of the cylinders (`WTYPE` 0, as in GALMOD) or of the spherical shells
(`WTYPE` 1, a column density in 1E20 atoms/cm²): one value, or one value per cylinder or shell (a list
or a file column).
"""),
    'CDENS': dict(
        group='Model',
        doc="""\
The surface density of clouds of the Monte Carlo models (`GALMODTYPE` `MC`), an integer: each sub-ring
of the model (0.75 pixels wide) has CDENS·DENS^`CMODE` clouds per pixel of its face-on area, and at least
10. The flux of the clouds is scaled so that the flux of the model does not depend on CDENS: more
clouds give less Monte Carlo noise and slower models.

ANALYTIC models do not use it, except to decide which model voxels count as blanks in the residuals of
3DFIT (see `BWEIGHT`). The spherical shells of GALWIND have CDENS·`DENS` clouds per pixel, with no
minimum (shells with no clouds are left empty, with a warning).
"""),
    'CMODE': dict(
        group='Model',
        doc="""\
How the number of clouds of the Monte Carlo models (`GALMODTYPE` `MC`) depends on the density of the
rings: there are `CDENS`·DENS^CMODE clouds per pixel (at least 10), so 0 gives `CDENS` clouds per pixel
whatever the density, 1 (the default) a number proportional to `DENS`, 2 to its square. Other values are
replaced by 1, with a warning. ANALYTIC models do not use it.
"""),
    'ISEED': dict(
        group='Model',
        doc="""\
The seed of the random numbers of the Monte Carlo models (`GALMODTYPE` `MC`), any integer. The default,
−1, is a fixed seed like any other: the models are reproducible. In the fits, every model is built with
the same seed, so that the residuals change smoothly with the parameters.

The final models are built in `THREADS` parallel chunks, each with a seed derived from ISEED, so they
depend on the number of threads. GALWIND derives a seed for each cylinder and cone, or for each chunk of
shells.
"""),
    'NV': dict(
        group='Model',
        default='the number of channels of the cube',
        doc="""\
The number of sub-clouds into which each cloud of the Monte Carlo models (`GALMODTYPE` `MC`) is split,
with velocities drawn from its line profile (`VDISP` plus the instrumental broadening); rings with
VDISP = 0 have one. Fewer sub-clouds give faster and noisier models.

ANALYTIC models do not use it, except to decide which model voxels count as blanks in the residuals of
3DFIT (see `BWEIGHT`). For the spherical shells of GALWIND (`WTYPE` 1) the default is 20 (BBarolo 1.8
also took 1 as 20).
"""),
    'SM': dict(
        group='Model',
        doc="""\
Whether the models are convolved with the beam of the cube (that of the header, or `BMAJ`, `BMIN`,
`BPA`, `BEAMFWHM`), to the spatial resolution of the data. With `false`, no model is smoothed: also the
models of single rings in the fits of 3DFIT and SPACEPAR. If no convolution kernel can be built from the
beam, the models are not smoothed, with a warning. GALWIND convolves its final cube.
"""),
    'LTYPE': dict(
        group='Model',
        doc="""\
The vertical distribution of the gas: the density as a function of the height z from the plane, with
the scale height `Z0`. 1: Gaussian, exp(−z²/(2·Z0²)) (the default); 2: sech²(z/Z0); 3: exponential,
exp(−|z|/Z0); 4: Lorentzian, 1/(1 + (z/Z0)²); 5: box, uniform within ±Z0. Other values are replaced by
1, with a warning.
"""),
    'REDSHIFT': dict(
        group='Spectral lines',
        doc="""\
The redshift of the galaxy, for cubes with a frequency or wavelength axis: the zero of the velocities is
at the rest frequency divided by (1 + REDSHIFT), or at the rest wavelength times (1 + REDSHIFT) (see
`RESTFREQ`, `RESTWAVE`). `VSYS` is then relative to the redshifted line, and close to 0 when REDSHIFT is
right. It does not change velocity axes. Values ≤ −1 are taken as 0.
"""),
    'RESTWAVE': dict(
        group='Spectral lines',
        default='not set',
        doc="""\
The rest wavelength(s) of the emission line(s), for cubes with a wavelength axis, in the units of that
axis (e.g. `6562.8` for Hα with `CUNIT3` in Angstrom). The zero of the velocities is at RESTWAVE times
(1 + `REDSHIFT`); without RESTWAVE it is at the reference pixel of the spectral axis, and `VSYS` must be
given in that frame.

Several values, separated by spaces, model several lines at once (e.g. Hα and the [NII] doublet) with
the relative intensities `RELINT`: the velocities refer to the first line, and each line of a cloud is at
its own rest wavelength times the same Doppler factor. On a velocity axis only one line is modelled,
with a warning.
"""),
    'RESTFREQ': dict(
        group='Spectral lines',
        default='the rest frequency of the header',
        doc="""\
The rest frequency (or frequencies) of the emission line(s), in Hz whatever the unit of the spectral
axis, for cubes with a frequency axis. It replaces the rest frequency of the header (`RESTFRQ`,
`RESTFREQ` or `FREQ0`; without them, 1.4204057 GHz, the HI line). The zero of the velocities is at
RESTFREQ/(1 + `REDSHIFT`).

Several values, separated by spaces, model several lines at once with the relative intensities
`RELINT`: the velocities refer to the first line, and each line of a cloud is at its own rest frequency
times the same Doppler factor (BBarolo 1.8 shifted the lines by fixed velocities, which was exact only at
velocity 0).
"""),
    'RELINT': dict(
        group='Spectral lines',
        doc="""\
The relative intensities of the lines given in `RESTWAVE` or `RESTFREQ`, one value per line, separated
by spaces (e.g. `1 3`): the flux of each line in the models is multiplied by its value. RELINT must have
as many values as `RESTWAVE` or `RESTFREQ`, otherwise it is an error; with a single line it is not
used.
"""),
    'NOISERMS': dict(
        group='Model',
        doc="""\
The rms (flux units of the cube) of Gaussian noise added to the final model of 3DFIT and GALMOD: if it
is not 0, the model with noise is also written to `<prefix>_mod_noise.fits`. The noise is convolved with
the beam when the model is (`SM`), then scaled back to this rms per voxel; it changes at each run. In
Python, `GalMod.model()` returns the model with noise (its `seed` makes the noise reproducible).
SPACEPAR does not use it.
"""),
    'RINGFILE': dict(
        group='Rings',
        default='none',
        doc="""\
A ring file written by BBarolo, e.g. `<prefix>_rings_final1.txt`, `<prefix>_rings_final2.txt`,
`<prefix>_rings_initial.txt` or `<prefix>_rings_model.txt` (also those of BBarolo 1.8), from which the
ring parameters that are not given are read: the radii, `VROT`, `VDISP`, `INC`, `PA`, `Z0`, `DENS`,
`XPOS`, `YPOS`, `VSYS` and `VRAD`. The parameters given take precedence; the radii are read unless
`RADII`, or both `NRADII` and `RADSEP`, are given.

The columns are found by their names in the header of the file; a file without a header must have the
columns of BBarolo 1.8 (the radii in arcsec in column 2, VROT in column 3, ...). With RINGFILE, GALMOD
does not require the ring parameters; those missing from the file take their defaults or are estimated.
"""),
    'EMPTY': dict(
        group='Model',
        doc="""\
Whether the model is empty inside the inner edge of the first ring. With `false` the disk is filled from
the centre with the values of the first ring. The inner edge depends on `RINGMODE` (half a ring inside
the first radius with `CENTRED`): when it is at the centre there is no hole anyway.

EMPTY applies to all the models, also to those of single rings in the fits of 3DFIT and SPACEPAR, where
`false` adds the emission of a whole disk inside the ring: keep it `true` for the fits.
"""),
    'DENSFLUX': dict(
        group='Model',
        doc="""\
Whether `DENS` is an integrated flux per unit area, in the flux units of the cube times km/s per arcsec²
(`true`, the default), or a column density in 1E20 atoms/cm² converted to flux as in GIPSY's GALMOD
(`false`). GALMOD uses `false` only with `NORM` `NONE`, since the normalisations `AZIM` and `LOCAL` work
with fluxes; 3DFIT always takes `DENS` as a flux. BBarolo 1.8 ignored `DENSFLUX false` in GALMOD.
"""),
    'DELTAINC': dict(
        group='Free parameters and fit',
        doc="""\
The largest change of the inclination (degrees) in the fit of each ring, when `FREE` includes `INC`:
3DFIT keeps INC within the initial value of the ring ± DELTAINC, and between 0 and 90; trial values
outside are moved to the boundary. Letting the inclination vary over the whole range from 0 to 90 is
not advisable.
"""),
    'DELTAPA': dict(
        group='Free parameters and fit',
        doc="""\
The largest change of the position angle (degrees) in the fit of each ring, when `FREE` includes `PA`:
3DFIT keeps PA within the initial value of the ring ± DELTAPA (and between −360 and 360); trial values
outside are moved to the boundary.
"""),
    'DELTAVROT': dict(
        group='Free parameters and fit',
        doc="""\
The largest change of the rotation velocity (km/s) in the fit of each ring, when `FREE` includes
`VROT`: 3DFIT keeps VROT within the initial value of the ring ± DELTAVROT, and above 0; a trial value
outside is replaced by a random value inside. In the second stage (`TWOSTAGE`) the interval is centred
on the values of the first stage. The default, 1000 km/s, is in practice no limit.
"""),
    'MINVDISP': dict(
        group='Free parameters and fit',
        doc="""\
The lowest velocity dispersion (km/s) allowed in the fit of 3DFIT, when `FREE` includes `VDISP`; a trial
value below it is replaced by a random value between the two limits. It must be smaller than `MAXVDISP`.
"""),
    'MAXVDISP': dict(
        group='Free parameters and fit',
        doc="""\
The highest velocity dispersion (km/s) allowed in the fit of 3DFIT, when `FREE` includes `VDISP`; a
trial value above it is replaced by a random value between the two limits. See also `MINVDISP`.
"""),
    'FTYPE': dict(
        group='Free parameters and fit',
        doc="""\
The function of the residuals that 3DFIT minimises (and SPACEPAR computes) in each ring, summed over the
voxels, where m is the model and o the data (the noise rms where the data are not positive):

- 1: (m − o)²/√o, a χ²-like function;
- 2: |m − o| (the default);
- 3: |m − o|/(m + o);
- 4: (m − o)².

Each term is weighted by `WFUNC`; the sum is divided by the number of voxels and multiplied by the
penalty for blanks (`BWEIGHT`). 3DFIT and SPACEPAR replace other values with 2, with a warning.
BAYESIAN3DFIT uses it only with `LIKEOPTS` `bbres=true`.
"""),
    'WFUNC': dict(
        group='Free parameters and fit',
        doc="""\
The weight of each pixel in the residuals, as a function of its azimuthal angle θ in the plane of the
disk (0 on the receding major axis): 0 uniform; 1 |cos θ|; 2 cos²θ (the default); −1 |sin θ|;
−2 sin²θ. The cosine weights favour the pixels near the major axis, which carry most of the information
on the rotation. Used by 3DFIT, SPACEPAR and 2DFIT (on the velocity field); 3DFIT and SPACEPAR replace
values beyond ±2 with 1, with a warning.
BAYESIAN3DFIT uses it only with `LIKEOPTS` `bbres=true`.
"""),
    'TOL': dict(
        group='Free parameters and fit',
        doc="""\
The tolerance of the minimisation of each ring in 3DFIT: the downhill simplex stops when the residuals
at its vertices differ by less than TOL in relative terms, 2|f_max − f_min|/(|f_max| + |f_min|). A ring
that does not converge within about 200 evaluations of the residuals per free parameter is marked as not
converged (see `BADOUT`).
"""),
    'FREE': dict(
        group='Free parameters and fit',
        default='VROT VDISP INC PA (BAYESIAN3DFIT: required)',
        doc="""\
The parameters that are fitted, case-insensitive, separated by spaces or commas; the other ring
parameters stay at their initial values.

3DFIT: any of `VROT`, `VDISP` (or `DISP`), `INC`, `PA` (or `PHI`), `Z0`, `XPOS`, `YPOS`, `VSYS` and
`VRAD`, or `ALL`, which also frees `DENS` (`DENS` alone is not fitted). When `PA` is free and there are at
least 5 rings, the initial PA (and VROT, if free) of the rings are first refined by a 2D fit of the
velocity field.
2DFIT: `VROT`, `INC`, `PA`, `XPOS`, `YPOS`, `VSYS`, `VRAD` (or `VEXP`) or `ALL`, the others are ignored;
with `SIDE` `A` or `R` the centre and VSYS stay fixed.

BAYESIAN3DFIT requires FREE. A ring parameter (`VROT`, `VDISP`, `DENS`, `Z0`, `INC`, `PA`, `XPOS`,
`YPOS`, `VSYS`, `VRAD`, `VVERT`, `DVDZ`, `ZCYL`) has one value per ring; `NAME_SINGLE` has one value
for all the rings; `NAME_FUNC` is a function of radius with two parameters: `VROT_FUNC` an arctangent,
2/π·p1·arctan(R/p2), `VDISP_FUNC` a line, p1 + p2·R, `DENS_FUNC` an exponential, p1·exp(−R/p2).
`SIGMA` (the noise per voxel), `FRACERR` (a model error proportional to the model) and `RADMAX` (the
outer edge of the disk, see `LIKEOPTS`) are single values. Each parameter appears once, and its prior is
set by `PRIORS`; e.g. `FREE VROT VDISP_SINGLE INC_SINGLE PA_SINGLE`.
"""),
    'GUESSTYPE': dict(
        group='Rings',
        doc="""\
How the ring parameters that are not given are estimated by 3DFIT, BAYESIAN3DFIT, SPACEPAR and 2DFIT:
the centre (`XPOS`, `YPOS`), `VSYS`, `VROT`, `INC`, `PA` and the rings (`NRADII`, `RADSEP`). The
estimates use the largest source found by the source finder (with the parameters of SEARCH), or the
mask of `MASK` `file(...)`. The values given are used as they are. Other values of GUESSTYPE are an
error.

`LEGACY` (the default) is the algorithm of BBarolo 1.8: VSYS from the global profile, the centre from
the position of the source, PA from its velocity field, INC from the axes of an ellipse fitted to its
extent, corrected for the beam, rings as wide as the beam (half of it if that gives fewer than 20
rings), VROT from the width of the global profile; a 2D tilted-ring fit of the velocity field then
refines them. `ELLIPSE` is the same algorithm, corrected: the beam is subtracted in quadrature (BBarolo
1.8 subtracted a whole beam), INC has no jumps, VROT is always deprojected, and the values given are
also kept in the intermediate steps.

`FIT` starts from `ELLIPSE`, then fits a beam-convolved disk to the total intensity map (centre, INC,
PA) and a 2D tilted-ring model to the velocity field (PA, VSYS), and estimates `VDISP` from the second
moment (3DFIT and SPACEPAR). On simulated disks its initial INC is biased by at most 2.6°, also below
30° and above 75°, where `LEGACY` is biased by +6° and −7°.
"""),
    'SIDE': dict(
        group={'ELLPROF': 'Options', '*': 'Free parameters and fit'},
        doc="""\
The half of the galaxy that is used, divided by the minor axis: `A` the approaching side, `R` the
receding side, `B` both (the default). 3DFIT and SPACEPAR use only the pixels of that side in the
residuals, 2DFIT in the fit of the velocity field (with one side it keeps the centre and `VSYS` fixed),
ELLPROF in the radial profile. All these tasks also accept `APPROACHING`, `RECEDING`, `BOTH` and their
abbreviations (`APP`, `REC`, ...); other values are taken as `B`, with a warning. BAYESIAN3DFIT uses it
only with `LIKEOPTS` `bbres=true`.
"""),
    'FREEANGLE': dict(
        group='Free parameters and fit',
        doc="""\
The angle (degrees) on either side of the minor axis, in the plane of the disk, within which 2DFIT
leaves the pixels of the velocity field out of the fit: near the minor axis the line-of-sight velocity
carries little information on the rotation. Between 0 (no pixel left out) and 90; 15 by default. In
BBarolo 1.8, 2DFIT read this angle from `DVDZ`.
"""),
    'BWEIGHT': dict(
        group='Free parameters and fit',
        doc="""\
The weight of the blank voxels in the residuals of 3DFIT and SPACEPAR: the voxels where the model has
emission and the mask (`MASK`) has none. The residual of a ring is multiplied by
(1 + N_blank/N)^BWEIGHT, where N is the number of voxels compared (section 2.4 of Di Teodoro &
Fraternali 2015). Larger values penalise more the models that extend beyond the observed emission; 0
ignores the blanks. BAYESIAN3DFIT uses it only with `LIKEOPTS` `bbres=true`.
"""),
    'TWOSTAGE': dict(
        group='Free parameters and fit',
        doc="""\
Whether 3DFIT fits the rings a second time after regularising the geometry. After the first stage, the
free parameters among `INC`, `PA`, `Z0`, `XPOS`, `YPOS` and `VSYS` are replaced by smooth functions of
radius (see `REGTYPE`), fitted to the rings of the first stage that were fitted, and fixed; the second
stage fits again the free `VROT`, `VDISP` and `VRAD` (and
`DENS` with `FREE` `ALL`) and writes `<prefix>_rings_final2.txt`.

The second stage runs only if one of those geometric parameters is free; it is skipped if the
regularisation fails (or no ring was fitted), and the rings not fitted are then empty in the final model.
"""),
    'FLAGERRORS': dict(
        group='Free parameters and fit',
        doc="""\
Whether 3DFIT estimates the errors of the free parameters of each ring. For each parameter, the
residuals are computed for 100 random values around the best fit (VROT ± VROT/3, VDISP ± 15 km/s, the
others within their limits of the fit), with the other parameters at their best values; a cubic fitted
to them gives the interval where the residuals are within 5% of the minimum. The lower and upper errors
are written in the columns `E_VROT1`, `E_VROT2`, ... of the ring files. This slows the fit down
considerably.

As in BBarolo 1.8: an error that cannot be estimated (e.g. too few points near the minimum) is replaced
by the mean of the errors of the ring, all the parameters together; a VDISP error smaller than half the
dispersion of a channel (ΔV/2.355/2) is raised to it plus a random 0–1 km/s; with `FREE` `ALL` the
errors of DENS are not meaningful (its interval of random values is empty).
"""),
    'NORM': dict(
        group={'3DFIT': 'Free parameters and fit', '*': 'Model'},
        doc="""\
The normalisation of the models to the data: `LOCAL`, each spectrum of the model is scaled to the flux
of the data in the mask in the same pixel; `AZIM` (azimuthal), the model of a ring is scaled to the
total flux of the data in the ring, and the final model takes the densities of the rings from the radial
profile of the data; `NONE`, no normalisation: the flux of the model is set by `DENS`.

3DFIT takes up to three values: the normalisation of the first stage, of the second stage
(`TWOSTAGE`) and of the final model; one value is used for all, and with two the second is also that of
the final model. The final model can also be `BOTH`, which writes the `AZIM` and the `LOCAL` model: in
3DFIT only as the third value (e.g. `NORM AZIM AZIM BOTH`), since in the first two values it is not a
normalisation of the fit and becomes `LOCAL`, with a warning, so `NORM BOTH` gives LOCAL fits and a LOCAL
final model.
GALMOD takes one value (the second, if more are given): `AZIM`, `LOCAL`, `NONE` or `BOTH`. `AZIMUTHAL`
is the same as `AZIM`. Values not recognised become `LOCAL`, with a warning. SPACEPAR uses the first
value, the normalisation of the first stage of 3DFIT (BBarolo 1.8 always used `LOCAL`).
BAYESIAN3DFIT uses it only with `LIKEOPTS` `bbres=true`.
"""),
    'REGTYPE': dict(
        group='Free parameters and fit',
        doc="""\
How the second stage of 3DFIT (`TWOSTAGE`) regularises the parameters fitted in the first: `auto` (the
default); `bezier`, a Bézier curve; `median`, the median of the rings; or an integer n, a polynomial of
degree n in radius (0: a constant), fitted without the largest and the smallest value when there are
enough rings. A single value applies to INC and PA. Different choices are given as `key=value` pairs for
`INC`, `PA`, `VSYS`, `XPOS`, `YPOS` and `Z0`, e.g. `REGTYPE INC=1 PA=median VSYS=bezier`; the others are
`auto`.

With `auto`, INC and PA are a constant with at most 4 rings; otherwise their median, or, when they
scatter by more than 3° (robust standard deviation), a straight line (fewer than 10 rings) or a Bézier
curve. VSYS, XPOS, YPOS and Z0 become their median. Invalid values become `auto`, with a warning.
"""),
    'STARTRAD': dict(
        group='Free parameters and fit',
        doc="""\
The index of the first ring fitted by 3DFIT, counted from 0: the rings inside it are not fitted (they
are not written in the ring files, unless `BADOUT`, and are empty in the final model). With `REVERSE`,
the fit goes from the outermost ring inwards to this one. A value beyond the last ring is taken as 0.
SPACEPAR also starts from this ring, and the asymmetric drift (`ADRIFT`) uses the rings from it.
"""),
    'ADRIFT': dict(
        group='Outputs',
        doc="""\
Whether 3DFIT computes the asymmetric drift correction of the rotation curve (Iorio et al. 2017). The
circular velocity is √(VROT² + V_A²), with V_A² = −R·σ²·(d ln σ²/dR + d ln Σ/dR), where σ is the velocity
dispersion and Σ the face-on surface density (the median of the total intensity map in each ring times
cos INC), regularised with polynomials of degree `ADRIFTPOL1` and `ADRIFTPOL2`. The rings fitted, from
`STARTRAD`, are used.

The results (radius, VCIRC, V_A², σ² and Σ, with their regularised values) are written in
`<prefix>_asymdrift.txt` and plotted; if a polynomial fit fails nothing is written, with a warning.
"""),
    'ADRIFTPOL1': dict(
        group='Outputs',
        doc="""\
The degree of the polynomial in radius fitted to the square of the velocity dispersion, VDISP², for the
asymmetric drift correction (`ADRIFT`); with n rings it is lowered to n − 2 if needed. A negative value
uses the VDISP of the rings, not regularised, and then takes the radial derivative of VDISP² as zero.
"""),
    'ADRIFTPOL2': dict(
        group='Outputs',
        doc="""\
The degree of the polynomial in radius fitted to the face-on surface density Σ (the median of the total
intensity map in each ring times cos INC) for the asymmetric drift correction (`ADRIFT`); with n rings it
is lowered to n − 2 if needed. It must be 0 or more: a negative value becomes 3, with an error message.
"""),
    'REVERSE': dict(
        group='Free parameters and fit',
        doc="""\
Whether 3DFIT fits the rings in reverse-cumulative order, which may help with highly inclined galaxies:
from the outermost ring inwards, the model of each ring includes the rings already fitted outside it
(the rings are fitted one after the other, in one thread). `true`, `false` (the default) or `auto`:
reverse if the median initial inclination is above 75°.

In reverse mode, rings narrower than the major axis of the beam are replaced by rings as wide as the
beam, with the initial values of the given rings interpolated at their radii. Other values become
`auto`, with an error message.
"""),
    'NORMALCUBE': dict(
        group='Free parameters and fit',
        doc="""\
Whether the data are scaled during the fits of 3DFIT and SPACEPAR so that their maximum is 10 (the data
and their noise are multiplied by 10/maximum), which helps the convergence with very small flux values.
The cube and the outputs do not change. With `NORM` `NONE` the models are compared with the scaled
data, so `DENS` must then be in the units of the scaled cube, or NORMALCUBE `false`.
"""),
    'BADOUT': dict(
        group='Outputs',
        doc="""\
Whether the ring files of 3DFIT (`<prefix>_rings_final1.txt`, `<prefix>_rings_final2.txt`) also include
the rings that were not fitted: not converged, rejected (a VROT larger than the spectral band allows) or
skipped (too few pixels in the mask, or inside `STARTRAD`). An extra column FITOK flags them with 0 (1
for the good rings). By default they are left out. These rings are empty in the final model in either
case.
"""),
    'PLOTMASK': dict(
        group='Outputs',
        doc="""\
Whether the contour of the mask is drawn on the channel maps and on the position-velocity plots of 3DFIT
and BAYESIAN3DFIT.
"""),
    'PLOTMINCON': dict(
        group='Outputs',
        default='from the noise or the thresholds of the mask',
        doc="""\
The lowest contour (flux units of the cube) of the channel maps and position-velocity plots of 3DFIT
and BAYESIAN3DFIT: the contours are at 1, 2, 4, ..., 64 times this value, plus one negative contour at
−PLOTMINCON. If it is not given (or not positive), it is the growth threshold of the source finder with
`MASK` `SEARCH` (its primary threshold without growth), otherwise 2.5 times the noise rms of the cube.
"""),
    'SAMPLER': dict(
        group='Bayesian sampling',
        doc="""\
The nested sampler of `BAYESIAN3DFIT` (case-insensitive):

- `DYNESTY`: the dynesty package, with its static sampler and multi-ellipsoid bounds (`bound=multi`);
  `SAMPLEROPTS dynamic=true` selects its dynamic sampler.
- `NAUTILUS`: the nautilus package (`nautilus-sampler` on PyPI).

The package must be installed: an unknown sampler, or one that is not installed, is an error before
the run. The options of the sampler are given with `SAMPLEROPTS` and `RUNOPTS`.
"""),
    'PRIORS': dict(
        group='Bayesian sampling',
        default='uniform, listed below',
        doc="""\
The priors of the free parameters of `BAYESIAN3DFIT`, as `NAME=(distribution,arguments)` (or
`NAME=distribution(arguments)`) separated by spaces, e.g. `PRIORS VROT=(uniform,loc=0,scale=300)
INC=(norm,60,5)`. The distribution is a continuous distribution of `scipy.stats` (`uniform`, `norm`,
`loguniform`, `truncnorm`, `halfcauchy`, ...), with its arguments as `key=value` or by position (the
shape parameters, then `loc` and `scale`, as in `scipy.stats`): `uniform` is uniform between `loc` and
`loc`+`scale`. A wrong entry is an error before the run.

NAME is a free parameter as in `FREE` (case-insensitive; `NAME_SINGLE` is the same as NAME, and the
prior of a parameter with one value per ring applies to each ring), or `VROT_P1`, `VROT_P2`,
`VDISP_P1`, ... for the parameters of a function of radius (`NAME_FUNC` in `FREE`). For `SIGMA`, the
arguments `a`, `b` and `scale` can be `none`: 0.5, 1000 and 1 times the measured noise rms.

The priors that are not given are uniform:

- `VROT` 0–350 km/s, `VDISP` 0–50 km/s, `VRAD` 0–50 km/s, `VVERT` 0–30 km/s, `INC` 0–90°, `PA` 0–360°;
- `XPOS` and `YPOS` ±5 pixels, and `VSYS` ±20 km/s, around the initial values;
- `Z0` 0–10 arcsec, `ZCYL` 0–5 arcsec, `DVDZ` 0–2, `DENS` 0.1–200.1 (a flux per unit area: the flux
  units of the cube × km/s per arcsec², see `DENS`; give a prior suited to the units of the cube);
- `RADMAX` within half a ring separation of the last radius (only the inner half with `RINGMODE NODES`,
  unless `LIKEOPTS radmax=ring`);
- `SIGMA` log-uniform between 0.5 and 1000 times the noise rms, `FRACERR` log-uniform between 0.001 and 1;
- the functions of radius R (arcsec): `VROT` = 2/π · P1 · arctan(R/P2), with P1 1–351 km/s and P2
  1–1001 arcsec; `VDISP` = P1 + P2 · R, with P1 0–10 km/s and P2 −10 to 10 km/s/arcsec; `DENS` =
  P1 · exp(−R/P2), with P1 1–21 and P2 1–201 arcsec.
"""),
    'SAMPLEROPTS': dict(
        group='Bayesian sampling',
        default='the defaults of the sampler',
        doc="""\
Options of the sampler of `BAYESIAN3DFIT`, as `key=value` separated by spaces, e.g.
`SAMPLEROPTS nlive=200 seed=1`: the arguments of `dynesty.NestedSampler` (or
`dynesty.DynamicNestedSampler`) or of `nautilus.Sampler`. The values are read as integers, numbers,
`true`/`false`, `none` or strings; values separated by commas form a tuple. An option that the sampler
does not accept is an error before the setup, with the list of the accepted ones.

Some options are handled by BBarolo:

- `dynamic=true` (dynesty): the dynamic sampler instead of the static one, the default;
- `seed=N`: a reproducible sampling (for dynesty BBarolo makes the random generator; nautilus has its
  own `seed`);
- `nlive`: the number of live points, accepted also by nautilus (where it is `n_live`).

The other options keep the defaults of the sampler, apart from `bound=multi` for dynesty. With
several processes (`THREADS`) BBarolo gives the sampler its pool of processes.
"""),
    'RUNOPTS': dict(
        group='Bayesian sampling',
        default='the defaults of the sampler',
        doc="""\
Options of the sampling run of `BAYESIAN3DFIT`, as `key=value` separated by spaces: the arguments of
`run_nested` of dynesty (of the static or dynamic sampler, see `SAMPLEROPTS`) or of `Sampler.run` of
nautilus, e.g. `RUNOPTS dlogz=0.5 maxiter=20000` (dynesty) or `RUNOPTS f_live=0.01` (nautilus). The
values are read as in `SAMPLEROPTS`, and an option that the sampler does not accept is an error before
the setup.

`dlogz`, the stopping criterion of the static sampler of dynesty (the estimated log-evidence left in
the live points), is accepted by all the samplers: it becomes `dlogz_init` for the dynamic sampler and
`f_live` = 1 − exp(−dlogz) for nautilus. The progress of the sampling is shown with `VERBOSE`
(`print_progress` of dynesty, also off when `SHOWBAR` is false; `verbose` of nautilus). The other
options keep the defaults of the sampler.
"""),
    'LIKEOPTS': dict(
        group='Bayesian sampling',
        default='the defaults listed below',
        doc="""\
Options of the likelihood of `BAYESIAN3DFIT`, as `key=value` separated by spaces, e.g.
`LIKEOPTS likemask=none noisecorr=beam dilation=1.5,4`. A wrong name or value is an error before the
run, and `none` gives the default. The options, with their defaults in brackets:

- `likemask` (`dilated`): the voxels compared with the model. `dilated`: the data, not masked, in the
  mask (`MASK`) dilated by `dilation`; `none`: all the voxels of the likelihood region, the projected
  disk within the outermost ring for any geometry allowed by the priors, widened by 3 `Z0` and half a
  beam.
- `dilation` (`1,3`): the dilation of the mask, as a radius in units of the beam FWHM and a number of
  channels on each side. With channels much narrower than the velocity dispersion, more channels
  include the faint wings of the lines.
- `noisecorr` (`measured`): the correlation of the noise; the log-likelihood is divided by the number
  of voxels per independent noise value. `measured`: the correlation volume measured in the voxels
  without emission (the beam area if they are too few); `beam`: the beam area in pixels; a number
  N > 0: N voxels (1: independent voxels).
- `densprof_smooth` (`0.1`): the weight of the smoothness term of the density profile deconvolved from
  the total intensity map, used when `DENS` is not free (0: none).
- `bbres` (`false`): `true` uses the residuals minimised by `3DFIT` (with its `FTYPE`, `WFUNC`, `SIDE`,
  `BWEIGHT` and `NORM`) instead of the Gaussian likelihood.
- `radmax` (`edge`): what the free parameter `RADMAX` is. `edge`: the outer edge of the disk, sharp,
  with the rings fixed (no emission beyond it); `taper`: the same edge, Gaussian, of width
  `radmax_width`; `ring`: the radius of the last ring, which moves.
- `radmax_width` (a quarter of the beam FWHM): the width (σ, arcsec) of the edge with `radmax=taper`.
- `sigma` (the measured noise rms): a fixed noise per voxel, in the units of the data, when `SIGMA` is
  not free.
- `fracerr` (0): a fixed fractional model error, when `FRACERR` is not free: the variance of a voxel is
  sigma² + (fracerr × model)².

The smoothness penalties of the ring parameters (`lambdas`) are available only in Python
(`like_kwargs` of `BayesianBBarolo.run`).
"""),
    'HTOT': dict(
        group='Wind',
        default='required',
        doc="""\
The extent of each cone of the outflow (arcsec) in GALWIND: the height of the stack of `NTOT` cylinders
along the axis (`WTYPE` 0), or the outer radius of the spherical shells (`WTYPE` 1). Required.
"""),
    'OPENANG': dict(
        group='Wind',
        default='required',
        doc="""\
The full opening angle of the cones of GALWIND (degrees). With cylinders (`WTYPE` 0), the cylinder k,
counted from the centre, has a radius of k·(`HTOT`/`NTOT`)·tan(OPENANG/2); with spherical shells
(`WTYPE` 1), the gas fills the polar angles from 0 to OPENANG/2 from the axis, uniformly in angle. One
value (an error otherwise with `WTYPE` 0); with `WTYPE` 1 also one value per shell (a list or a file
column). Required.
"""),
    'NTOT': dict(
        group='Wind',
        doc="""\
The number of cylinders in each cone of GALWIND (`WTYPE` 0), or the number of spherical shells (`WTYPE`
1), each `HTOT`/NTOT thick. With `WTYPE` 1 the radii of the shells can be read instead from a file
column or a list given with `RADII`. With cylinders, `VWIND`, `VDISP`, `DENS` and `VROT` given as lists
or file columns must have NTOT values; with shells, at least as many values as shells.
"""),
    'VWIND': dict(
        group='Wind',
        default='required',
        doc="""\
The velocity of the gas of the outflow along the cones (km/s). With spherical shells (`WTYPE` 1) it is
radial, away from the centre. With cylinders (`WTYPE` 0) it is directed along the lines through the
centre, away from it: along the axis on the axis and at `OPENANG`/2 from it at the edge of the cylinders
(see `DENSTYPE`). A positive VWIND is an outflow in both models (in BBarolo 1.8 the cylinders had the
opposite velocities, an inflow). One value, or one value per cylinder or shell (a list or a file
column). Required.
"""),
    'DENSTYPE': dict(
        group='Wind',
        doc="""\
How the density is distributed in the cylinders of GALWIND (`WTYPE` 0):

- 0: the cylinder k, counted from the centre, has density `DENS`/k²;
- 1: all the cylinders have density `DENS` (the default);
- 2: the density `DENS`, on the wall of the cone only: each cylinder is a ring between the radii of the
  cylinders k − 1 and k, and the gas moves at `OPENANG`/2 from the axis.

In the filled cylinders (0 and 1) the direction of the velocity goes from the axis, at the centre of the
cylinder, to `OPENANG`/2 from it, at the edge. The spherical model does not use DENSTYPE.
"""),
    'WTYPE': dict(
        group='Wind',
        doc="""\
The geometry of the outflow of GALWIND: 0 (the default), a stack of `NTOT` cylinders in each cone, each a
disk model as in GALMOD with scale height `HTOT`/(2·`NTOT`) and the vertical and radial velocities of the
wind; 1, `NTOT` spherical shells, with clouds within `OPENANG`/2 of the axis moving radially. Any value
other than 0 gives shells.

In the spherical model `DENS` is a column density (1E20 atoms/cm²), `NV` defaults to 20, `DENSTYPE` is
not used and the models are always Monte Carlo (`GALMODTYPE` is ignored).
"""),
    'P1': dict(
        group='Parameter space',
        doc="""\
The first parameter of the space explored by SPACEPAR (the x axis of the plots): one of `VROT`,
`VDISP`, `Z0`, `INC`, `PA`, `XPOS`, `YPOS`, `VSYS` and `VRAD`, case-insensitive, different from `P2`.
Its values are set by `P1PAR`. For each ring, SPACEPAR computes the residuals of 3DFIT on the grid of P1
and `P2`, with the other parameters fixed at the initial values of the ring (given or estimated).
"""),
    'P1PAR': dict(
        group='Parameter space',
        doc="""\
The range and step of `P1`: three numbers, the minimum, the maximum and the step, in the units of the
parameter (km/s; degrees for INC and PA; arcsec for Z0; pixels for XPOS and YPOS). The grid starts at the
minimum and has int((maximum − minimum)/step) + 1 values: `0 100 2` gives 0, 2, ..., 100.
"""),
    'P2': dict(
        group='Parameter space',
        doc="""\
The second parameter of the space explored by SPACEPAR (the y axis of the plots): one of `VROT`,
`VDISP`, `Z0`, `INC`, `PA`, `XPOS`, `YPOS`, `VSYS` and `VRAD`, case-insensitive, different from `P1`.
Its values are set by `P2PAR`.
"""),
    'P2PAR': dict(
        group='Parameter space',
        doc="""\
The range and step of `P2`: three numbers, the minimum, the maximum and the step, in the units of the
parameter (km/s; degrees for INC and PA; arcsec for Z0; pixels for XPOS and YPOS). The grid starts at the
minimum and has int((maximum − minimum)/step) + 1 values: `0 100 2` gives 0, 2, ..., 100.
"""),
    'SMOOTH': dict(
        group='Task',
        doc="""\
Smooths the datacube spatially: each channel map is convolved with the elliptical Gaussian that turns
the beam of the data (that of the header, or `OBMAJ`, `OBMIN`, `OBPA`) into a larger beam, given by
`BMAJ`, `BMIN` and `BPA` or by `FACTOR` times the old beam. `true` runs the task. The new beam must be
at least as large as the old one along both axes (an error otherwise), and the data are multiplied by
`SCALEFACTOR` to keep the values in Jy/beam.

The smoothed cube is written in `OUTFOLDER` as `<prefix>_s<N>.fits` (N: the major axis of the new beam
in arcsec, rounded), with the new beam in its header, or in `SMOOTHOUTPUT`. `BOX` smooths a sub-region
and `REDUCE` averages the pixels of the smoothed cube. Blank pixels of the input (NaN or 0) stay
blank. `LINEAR` no longer sets the new beam: it was in kpc and needed `DISTANCE`, which BBarolo 2
does not have.
"""),
    'BOX': dict(
        group='Input',
        default='the whole cube',
        doc="""\
The sub-region of the cube that `SMOOTH` smooths and writes, as six integers
`xmin xmax ymin ymax zmin zmax` (pixels and channels, 0-based; each maximum is excluded). A value of
−1, or one outside the cube, takes the full extent along that axis (0 for a minimum, the size of the
axis for a maximum); missing values are −1. The output cube has the size of the box, with its
reference pixels shifted accordingly.
"""),
    'BMAJ': dict(
        group={'SMOOTH': 'New beam', '*': 'Input'},
        default='the beam of the header (SMOOTH: FACTOR times the old beam)',
        doc="""\
The major axis (FWHM, arcsec) of a beam, with two uses:

- for every task except `SMOOTH`, the beam of the data when the FITS header has none (no BMAJ/BMIN
  keywords and no beam in the HISTORY). With only one of `BMAJ` and `BMIN` the beam is circular; they
  take precedence over `BEAMFWHM`. When the header has a beam, they do not change it.
- for `SMOOTH`, the major axis of the new beam (by default `FACTOR` times the old one; if only `BMAJ` is
  given, `BMIN` is the same). It is also the beam to which the SMOOTH mask (`MASK SMOOTH` or
  `SMOOTH&SEARCH`) smooths the cube before building the mask, unless BMAJ is the beam of the data (a
  header without a beam): the mask then uses `FACTOR` times it.

With `SMOOTH` and a header without a beam, the beam of the data is given by `OBMAJ`, `OBMIN` and `OBPA`
(or `BEAMFWHM`), never by `BMAJ` and `BMIN`. In Python, the beam of the data can also be given as
`beam=(bmaj, bmin, bpa)`.
"""),
    'BMIN': dict(
        group={'SMOOTH': 'New beam', '*': 'Input'},
        default='the beam of the header (SMOOTH: FACTOR times the old beam)',
        doc="""\
The minor axis (FWHM, arcsec) of a beam, with the two uses of `BMAJ`: the beam of the data when the
FITS header has none (equal to `BMAJ` if only that is given), and for `SMOOTH` and the SMOOTH mask the
minor axis of the new beam (by default `FACTOR` times the old one, or `BMAJ` if only that is given to
`SMOOTH`).
"""),
    'BPA': dict(
        group={'SMOOTH': 'New beam', '*': 'Input'},
        default='0 (SMOOTH: the angle of the old beam)',
        doc="""\
The position angle of the major axis of a beam (degrees, anticlockwise from North), with the two uses
of `BMAJ`: the beam of the data when the FITS header has none, and for `SMOOTH` and the SMOOTH mask
the position angle of the new beam.

If not given, the beam of the data has a position angle of 0°, and the new beam of `SMOOTH` and of the
SMOOTH mask that of the old beam. Any value given is an angle (in BBarolo 1.8 the default was 0° also
for the new beam, and −1 kept the old angle).
"""),
    'OBMAJ': dict(
        group='Old beam',
        default='the beam of the header',
        doc="""\
The major axis (FWHM, arcsec) of the beam of the input cube, for `SMOOTH`: it replaces that of the
header. If not given, the beam of the header. When the header has no beam, `OBMAJ` and `OBMIN` (or
`BEAMFWHM`) give it, for `SMOOTH` and for the other tasks of the same run; without them BBarolo stops
with an error (`BMAJ` and `BMIN` are the new beam of `SMOOTH`).
"""),
    'OBMIN': dict(
        group='Old beam',
        default='the beam of the header',
        doc="""\
The minor axis (FWHM, arcsec) of the beam of the input cube, for `SMOOTH`: it replaces that of the
header, as `OBMAJ`. If not given, the minor axis of the beam of the header.
"""),
    'OBPA': dict(
        group='Old beam',
        default='the BPA of the header (0 if the header has no beam)',
        doc="""\
The position angle (degrees, anticlockwise from North) of the beam of the input cube, for `SMOOTH`: it
replaces that of the header. Any value given is an angle, also 0 (in BBarolo 1.8, 0 meant the angle of
the header).
"""),
    'LINEAR': dict(
        group='Model',
        default='not set',
        doc="""\
The spectral resolution of the models (σ of the instrumental line profile, in channels) when `SPECRES`
is not given, as in BBarolo 1.8: e.g. 0.85 for Hanning-smoothed data (a FWHM of 2 channels). See
`SPECRES`. SMOOTH no longer uses LINEAR (it was a resolution in kpc there).
"""),
    'SPECRES': dict(
        group='Model',
        default='LINEAR, or 1/2.355 channels',
        doc="""\
The spectral resolution of the instrument: the dispersion σ (in channels) of the instrumental line
profile, which the models add in quadrature to `VDISP`: the lines of the models have
σ² = VDISP² + (SPECRES × channel width)². If it is not given, `LINEAR` is used or, without it,
1/2.355 ≈ 0.42 channels (a FWHM of one channel). For Hanning-smoothed data (a FWHM of 2 channels) use
0.85; 0 gives no instrumental broadening.
"""),
    'FACTOR': dict(
        group={'SMOOTH': 'New beam', '*': 'Mask'},
        doc="""\
The factor by which the beam is enlarged when no new beam is given with `BMAJ` and `BMIN`:

- `SMOOTH`: the new beam is `FACTOR` times the old one along both axes, at the angle of the old beam
  (or `BPA`); an axis given with `BMAJ` or `BMIN` takes that value instead. `SMOOTH` needs either the
  new beam or `FACTOR`.
- the SMOOTH mask (`MASK SMOOTH` or `SMOOTH&SEARCH`, used by `3DFIT`, `BAYESIAN3DFIT`, `2DFIT`,
  `ELLPROF`, the maps and `MAKEMASK`): the cube is smoothed to `FACTOR` times its beam before the mask is
  built (−1 is taken as 2).
- `REDUCE`, with `SMOOTH` or on its own: the integer part of `FACTOR` is the number of pixels (or
  channels) averaged along each axis, also when `SMOOTH` has a new beam from `BMAJ` and `BMIN`. `REDUCE`
  on its own needs FACTOR ≥ 1.
"""),
    'SCALEFACTOR': dict(
        group='Options',
        default='computed from the beams',
        doc="""\
The factor by which `SMOOTH` multiplies the smoothed cube. If not given, BBarolo computes it from the
old and the new beam (about the ratio of their areas), so that the values stay per beam: for data in
Jy/beam, mJy/beam, uJy/beam, W.U. or in units BBarolo does not know. For data in K or per pixel (Jy,
mJy) it is 1.
"""),
    'FFT': dict(
        group='Options',
        doc="""\
Whether `SMOOTH` convolves with Fast Fourier Transforms. In BBarolo 2 the convolution is always
computed with FFTs: `FFT false` only writes the blank pixels of the input (NaN or 0) as 0 in the
smoothed cube instead of NaN. (The direct convolution of the C++ code wrote 0 for the blanks that were 0,
and spread the NaN blanks over the size of the kernel.)
"""),
    'REDUCE': dict(
        group='Options',
        doc="""\
Whether to average blocks of pixels or channels, to reduce the size of the cube: `false` (the default),
`true`, `spatial` or `spectral` (any other value is taken as `true`, with a warning). The blocks that
do not fit at the edges are dropped, and the header (pixel size, channel width, reference pixel) is
updated.

- With `SMOOTH`, the smoothed cube is averaged over blocks of N×N pixels, N being the integer part of
  `FACTOR` (if larger than 1), whatever the value, and `red` is added to its name (e.g.
  `<prefix>_s120red.fits`).
- With `SMOOTHSPEC`, the smoothed cube is averaged over blocks of (`WINDOW_SIZE`+1)/2 channels for
  `HANNING` and `WINDOW_SIZE` channels for the other windows, and written as `<prefix>_h<N>_red.fits`.
- Without `SMOOTH` and `SMOOTHSPEC`, `REDUCE` averages the input cube over N×N pixels (`true` or
  `spatial`) or N channels (`spectral`), N being the integer part of `FACTOR`, and writes
  `<prefix>_red.fits` in `OUTFOLDER`.
"""),
    'SMOOTHOUTPUT': dict(
        group='Outputs',
        default='`<prefix>_s<N>.fits` in `OUTFOLDER`',
        doc="""\
The name of the FITS file of the cube smoothed by `SMOOTH`, as a path (relative to the current folder,
not to `OUTFOLDER`). By default the cube is written in `OUTFOLDER` as `<prefix>_s<N>.fits`, N being the
major axis of the new beam in arcsec (rounded), with `red` added to the name when `REDUCE` is on.
"""),
    'SMOOTHSPEC': dict(
        group='Task',
        doc="""\
Smooths the datacube along the spectral axis: each spectrum is convolved with a window of type
`WINDOW_TYPE` and size `WINDOW_SIZE` (by default a Hanning window of 3 channels). `true` runs the task.
The smoothed cube is written in `OUTFOLDER` as `<prefix>_h<N>.fits`, N being `WINDOW_SIZE`; with
`REDUCE` its channels are also averaged (`<prefix>_h<N>_red.fits`). Blank voxels (NaN) are taken as 0,
and the spectra are padded with zeros beyond their ends.
"""),
    'WINDOW_TYPE': dict(
        group='Window',
        doc="""\
The window of the spectral smoothing of `SMOOTHSPEC` (case-insensitive):

- `HANNING` (the default): the Hanning window, e.g. weights 1/4, 1/2, 1/4 for 3 channels;
- `HANNING2`: a Hann window that is zero at its ends (5 channels give the weights of `HANNING` with 3);
- `BOXCAR` (or `TOPHAT`): equal weights;
- `BARTLETT` (triangular), `WELCH` (parabolic), `BLACKMAN` and `FLATTOP`: the classic windows, zero or
  nearly zero at their ends;
- `GAUSSIAN`: a Gaussian of FWHM `WINDOW_SIZE` channels, truncated at 5σ.

The windows are normalised to a unit sum. An unknown window is an error. The same window is used by
the mask `MASK SMOOTHSPEC&SEARCH`.
"""),
    'WINDOW_SIZE': dict(
        group='Window',
        doc="""\
The size of the window of `SMOOTHSPEC`, in channels: an odd integer for all the windows but `GAUSSIAN`
(an even size is an error, and a fractional part is ignored), and the FWHM in channels for `GAUSSIAN`.
It also sets the number of channels averaged by `REDUCE` with `SMOOTHSPEC`, and the window of the mask
`MASK SMOOTHSPEC&SEARCH`.
"""),
    'SLITFIT': dict(
        group='Task',
        doc="""\
The fit of long-slit spectroscopic data of the C++ BBarolo. It is not available in BBarolo 2 (it was
broken in C++ too): `true` only prints a message. As in C++, it still needs `SLITWIDTH` > 0 (an error
otherwise), and it is turned off, with a warning, when `3DFIT` runs.
"""),
    'SLITWIDTH': dict(
        group='Options',
        default='none',
        doc="""\
The width of the slit for `SLITFIT` (arcsec in the C++ code). Not used: `SLITFIT` is not available in
BBarolo 2, and `SLITWIDTH` is only checked to be positive when `SLITFIT` is `true`.
"""),
    'PVSLICE': dict(
        group='Task',
        doc="""\
Extracts a position-velocity (PV) slice from the datacube and writes it as a FITS image,
`<prefix>_pv_<PA>.fits` in `OUTFOLDER` (`<PA>`: `PA_PV` rounded, 0 for a slice between two points).
`true` runs the task. The slice is defined either by two points, `P1_PV` and `P2_PV`, or by a centre
and an angle, `XPOS_PV`, `YPOS_PV` and `PA_PV`; when all the coordinates of `P1_PV` and `P2_PV` are
given, the two points are used.

The cube is sampled at steps of one pixel along the slice, by bilinear interpolation (see `ANTIALIAS`),
and averaged across a slit of width `WIDTH_PV`. The first axis of the image is the offset along the
slice, in the units of the spatial axes of the cube: 0 at the centre (or at the midpoint of the two
points), increasing with X for a slice through a centre (with Y if it is vertical) and towards `P2_PV`
for two points. The second axis is the spectral axis of the cube.
"""),
    'XPOS_PV': dict(
        group='Slice',
        default='none',
        doc="""\
The X coordinate of the centre of the slice of `PVSLICE` (with `YPOS_PV` and `PA_PV`): a pixel
(0-based) or a world coordinate, as for `XPOS`: degrees with a `d` (e.g. `114.2271d`) or sexagesimal
(e.g. `07h36m51s` or `07:36:51`). `XPOS_PV` and `YPOS_PV` must be both pixels or both world
coordinates, and the centre must lie inside the cube. Ignored when `P1_PV` and `P2_PV` are given.
"""),
    'YPOS_PV': dict(
        group='Slice',
        default='none',
        doc="""\
The Y coordinate of the centre of the slice of `PVSLICE`, as `XPOS_PV`: a pixel (0-based) or a world
coordinate (e.g. `65.5889d`, `+65d35m20s` or `65:35:20`).
"""),
    'PA_PV': dict(
        group='Slice',
        doc="""\
The position angle of the slice of `PVSLICE` through (`XPOS_PV`, `YPOS_PV`), in degrees, anticlockwise
from the +Y axis of the image (from North through East for a standard image, as `PA`). The slice
crosses the whole field. The angle, rounded, is also in the name of the output file. Ignored when
`P1_PV` and `P2_PV` are given.
"""),
    'WIDTH_PV': dict(
        group='Slice',
        doc="""\
The full width of the slit of `PVSLICE`, in arcsec: the slice is averaged over positions across the
slit, from −`WIDTH_PV`/2 to +`WIDTH_PV`/2, spaced by at most half a pixel. 0 (the default) samples a
single line. With `ANTIALIAS` ≥ 1 the slit is at least 2·`ANTIALIAS`+1 pixels wide.
"""),
    'P1_PV': dict(
        group='Slice',
        default='none',
        doc="""\
The first point of a slice of `PVSLICE` between two points, as `x y` in pixels (0-based), e.g.
`P1_PV 10 30`. The slice is the segment from `P1_PV` to `P2_PV`, cut at the edges of the cube, with
the offset 0 at its midpoint and increasing towards `P2_PV`. The two points are used, instead of
`XPOS_PV`, `YPOS_PV` and `PA_PV`, when all their coordinates are 0 or more.
"""),
    'P2_PV': dict(
        group='Slice',
        default='none',
        doc="""\
The second point of a slice of `PVSLICE` between two points, as `x y` in pixels (0-based); see
`P1_PV`.
"""),
    'ANTIALIAS': dict(
        group='Slice',
        doc="""\
How `PVSLICE` samples the cube along the slice: `0` takes the nearest pixel; any other value
interpolates bilinearly between the four nearest pixels (`0.5`, the default), and an integer N ≥ 1
also averages across a slit at least 2N+1 pixels wide (see `WIDTH_PV`). Only 0, 0.5 and integers are
accepted: other values are replaced by 0.5, with a message.
"""),
    'REND3D': dict(
        group='Task',
        doc="""\
Writes a 3D rendering of the datacube: projections of the cube within the mask (`MASK`), seen from
directions that turn around the Y axis of the image, in the plane of the X axis and the spectral axis
(pixels and channels taken as equal lengths), one every degree from 0° (the sum over the channels) to
`RENDANGLE`. `true` runs the task. Each projection is divided by the number of voxels summed in it,
and the projections are smoothed to 1.2 times the beam and written as the planes of the cube
`<prefix>_3D.fits` in `OUTFOLDER`, whose third axis is the angle in degrees.
"""),
    'RENDANGLE': dict(
        group='Options',
        doc="""\
The largest azimuthal angle of the rendering of `REND3D`, in degrees: the output has one projection
per degree, at 0, 1, ..., `RENDANGLE`−1 degrees (the integer part). The default, 360, is a full turn.
"""),
}
