# SEARCH: how it works

SEARCH is the source finder of BBarolo. It is derived from Duchamp (Whiting 2012) and finds the
groups of connected voxels that stand above the noise of a cube (or of a 2D image). The input is the
cube; the outputs are a catalogue of the detections with their positions, sizes, line widths and
fluxes, cubes and maps of the detected voxels and, optionally, a sub-cube (cubelet) of each detection.
The same search, with the same parameters, builds the masks `SEARCH`, `SMOOTH&SEARCH` and
`SMOOTHSPEC&SEARCH` used by the fits and the maps (see {doc}`../makemask/algorithm`).

The search runs in six steps: the noise of the cube and the thresholds; the objects found above the
primary threshold, in each channel or in each spectrum; a first merging while they are found; a first
rejection of small objects; the final merging; the growth down to the secondary threshold and a last
rejection. The parameters of the detections are then measured on the cube.

Throughout the page, a voxel is $(x, y, z)$, with $x$, $y$ the pixel and $z$ the channel (counted from
0), and $F(x, y, z)$ is its value in the flux units of the cube (`BUNIT`). Blank voxels (NaN) are read as
0.

(search-noise)=
## Noise and thresholds

The noise of the cube is measured on all the voxels that are not blank (0 or NaN), as described in
{doc}`../stats/algorithm`: a middle value $m$ and a spread $\sigma$, both in the flux units of the
cube,

$$
m = \mathrm{median}(F), \qquad \sigma = \frac{\mathrm{MADFM}(F)}{0.6744888}
$$

with robust statistics ({ref}`FLAGROBUSTSTATS <par-all-flagrobuststats>`, the default), otherwise the
mean and the standard deviation. With {ref}`ITERNOISE <par-search-iternoise>` the bright voxels are
excluded iteratively before the final values ({ref}`STATS <stats-iternoise>`). If STATS ran before in the
same run, its statistics are used.

The primary threshold is

$$
T_1 = m + n_1\,\sigma ,
$$

with $n_1$ = {ref}`SNRCUT <par-search-snrcut>`, or $T_1$ = {ref}`THRESHOLD <par-search-threshold>` when
it is given (in flux units). The secondary threshold, used only by the growth, is

$$
T_2 = m + n_2\,\sigma ,
$$

with $n_2$ = {ref}`GROWTHCUT <par-search-growthcut>`, or $T_2$ =
{ref}`GROWTHTHRESHOLD <par-search-growththreshold>` when it is given. The growth is turned off, with a
message, if $n_2 > n_1$ (or, with THRESHOLD, if GROWTHTHRESHOLD exceeds THRESHOLD).

A voxel is *detected* if

$$
F(x, y, z) > T_1 .
$$

### `SEARCHTYPE spatialsmooth`

With {ref}`SEARCHTYPE <par-search-searchtype>` `spatialsmooth` the search runs on a copy of the cube
smoothed with the parameters of SMOOTH (see {doc}`../smooth/algorithm`). The noise $m$, $\sigma$ and the
thresholds are then measured on the smoothed cube, without ITERNOISE; the parameters of the detections
(below) are measured on the original cube, and their peak S/N uses the noise of the original cube.

(search-objects)=
## Objects

The detected voxels are joined into objects, according to {ref}`SEARCHTYPE <par-search-searchtype>`:

- `spatial` (default): in each channel $z$, the detected pixels are grouped into 8-connected objects:
  two pixels belong to the same object if they are linked by a chain of detected pixels, each touching
  the next along a side or a corner, $\max(|\Delta x|, |\Delta y|) \le 1$. The channels are scanned in
  increasing $z$.
- `spectral`: in each spectrum $(x, y)$, every run of consecutive detected channels is an object. The
  spectra are scanned row by row.

A 2D image (one channel) is always searched spatially.

(search-close)=
## Closeness of two objects

Two voxels $p$ and $q$ are *close* if they are near in velocity,

$$
|z_p - z_q| \le t_v ,
$$

with $t_v$ = {ref}`THRESHVELOCITY <par-search-threshvelocity>` (channels), and near on the sky:

- with {ref}`FLAGADJACENT <par-search-flagadjacent>` true (default), if the pixels coincide or touch,
  $\max(|x_p - x_q|, |y_p - y_q|) \le 1$;
- with FLAGADJACENT false, if
  $$
  \sqrt{(x_p - x_q)^2 + (y_p - y_q)^2} < t_s ,
  $$
  with $t_s$ = {ref}`THRESHSPATIAL <par-search-threshspatial>` (pixels). Its default is the major axis of
  the beam in pixels, $t_s = \lceil B_{\rm maj}/\Delta \rceil$, where $\Delta$ is the mean of the two
  pixel sizes.

Two objects are close if at least one voxel of the first is close to a voxel of the second.

(search-merging)=
## Merging

### First stage

With {ref}`TWOSTAGEMERGING <par-search-twostagemerging>` true (default), the objects are merged while
they are found: each new object joins the first object of the list (the one found earliest) that is
close to it, or starts a new object if none is. This stage does not join two objects already in the
list, even when the new one is close to both: that is left to the final merging.

With TWOSTAGEMERGING false, every object found starts a new entry, and all the merging happens in the
final stage.

### Rejection before merging

With {ref}`REJECTBEFOREMERGE <par-search-rejectbeforemerge>` true (default), the objects of the list that
do not meet the size requirements (see [Rejection](search-rejection)) are discarded at this point, before
the final merging and the growth. With TWOSTAGEMERGING false, the objects are then single channel maps (or
single spectra), which fail MINCHANNELS: set also REJECTBEFOREMERGE false in that case.

### Final merging

All the objects that are close are merged, transitively: if $A$ is close to $B$ and $B$ to $C$, then
$A$, $B$ and $C$ become one object, even when $A$ and $C$ are not close. The objects are the connected
components of the graph whose nodes are the objects and whose edges join the pairs of close objects.

(search-growth)=
## Growth

With {ref}`FLAGGROWTH <par-search-flaggrowth>` true (default), each object is grown into the voxels
around it that are above the secondary threshold, $F > T_2$. The growth follows chains of voxels in which
each voxel is within the box

$$
|\Delta x| \le s, \qquad |\Delta y| \le s, \qquad |\Delta z| \le t_v
$$

of the previous one, with $s = 1$ with FLAGADJACENT true, $s = t_s$ otherwise (a square of side
$2t_s + 1$ pixels, not the circle of the merging). A grown object is the set of voxels reached from its
voxels in this way. Objects that reach each other through the growth, or that come within the box of
each other, become one object. Voxels above $T_2$ that are not connected to a detection are not
included.

(search-rejection)=
## Rejection

After the merging and the growth (and also before the final merging, with REJECTBEFOREMERGE), an object
is kept only if it meets all the following requirements:

- $N_{\rm adj} \ge$ {ref}`MINCHANNELS <par-search-minchannels>`, where $N_{\rm adj}$ is the largest
  number of consecutive channels among the channels of the object;
- $N_{\rm pix} \ge$ {ref}`MINPIX <par-search-minpix>`, where $N_{\rm pix}$ is the number of distinct
  pixels $(x, y)$ the object covers in any channel. Default: $\lceil \Omega_{\rm b} \rceil$, the area of
  the beam in pixels,
  $$
  \Omega_{\rm b} = \frac{\pi\, B_{\rm maj} B_{\rm min}}{4 \ln 2\; \Delta^2} ,
  $$
  with $B_{\rm maj}$, $B_{\rm min}$ the FWHM axes of the beam and $\Delta$ the pixel size (same units);
- $N_{\rm vox} \ge$ {ref}`MINVOXELS <par-search-minvoxels>`, where $N_{\rm vox}$ is the number of
  voxels. Default: MINCHANNELS × MINPIX;
- if {ref}`MAXCHANNELS <par-search-maxchannels>` $> 0$: $N_{\rm chan} <$ MAXCHANNELS, where
  $N_{\rm chan}$ is the number of distinct channels of the object;
- if {ref}`MAXANGSIZE <par-search-maxangsize>` $> 0$: $\max(x_{\max} - x_{\min},\, y_{\max} - y_{\min})
  < \theta_{\max}/\Delta$, with $\theta_{\max}$ = MAXANGSIZE (arcmin) and $\Delta$ the pixel size in
  arcmin.

The objects that pass are the detections. They are sorted by
{ref}`SORTSOURCES <par-search-sortsources>` (by default by number of voxels, largest first) and numbered
from 1 in that order.

(search-parameters)=
## Parameters of the detections

For a detection $D$ with voxels $(x_k, y_k, z_k)$ and values $F_k$:

**Total flux and centre.** The total flux is $F_{\rm tot} = \sum_k F_k$ (flux units of the cube) and the
centre is the flux-weighted centroid, in pixels and channels,

$$
x_c = \frac{\sum_k x_k F_k}{F_{\rm tot}}, \qquad
y_c = \frac{\sum_k y_k F_k}{F_{\rm tot}}, \qquad
z_c = \frac{\sum_k z_k F_k}{F_{\rm tot}} .
$$

The world coordinates of the centre (right ascension and declination, in degrees, from the WCS of the
header) and its velocity $v(z_c)$ (km/s, with the definition of {ref}`VELDEF <par-all-veldef>`) follow
from $(x_c, y_c, z_c)$.

**Peak.** The peak flux $F_{\rm peak} = \max_k F_k$ (flux units of the cube) and its peak S/N,

$$
\mathrm{S/N}_{\rm peak} = \frac{F_{\rm peak}}{\sigma} ,
$$

with $\sigma$ the noise of the cube (the middle value is not subtracted).

**Integrated flux.** With $v(z)$ the velocity of channel $z$ (km/s), the channel width at channel $z$ is
$\Delta v_z = |v(z+1) - v(z-1)|/2$, and the integrated flux is

$$
S_{\rm int} = \frac{1}{\Omega_{\rm b}} \sum_k \frac{F_k}{P_k}\, \Delta v_{z_k} ,
$$

in Jy km/s, where the sum gives Jy/beam km/s after the conversion of the flux unit to Jy/beam (for a
cube in Jy/beam, mJy/beam, K, ...; see {ref}`FLUXCONVERT <par-all-fluxconvert>`) and $\Omega_{\rm b}$ is
the beam area in pixels. With FLUXCONVERT false, or a flux unit without a known conversion, $S_{\rm int}$
is the sum alone, in the units of the cube times km/s. $P_k$ is the primary-beam response (below); $P_k = 1$
without {ref}`PBCORR <par-search-pbcorr>`.

**Line widths.** The integrated spectrum of the detection is

$$
S(z) = \sum_{k:\, z_k = z} F_k ,
$$

the sum of the observed (not primary-beam corrected) values of its voxels in channel $z$; it is zero in
the channels without voxels of $D$. Let $S_{\max}$ be its maximum in $[z_{\min}, z_{\max}]$, the
channel range of $D$. For a fraction $f$ (0.5 or 0.2), the lower edge starts at $z_{\min}$ and moves
towards lower channels while $S > f S_{\max}$, or towards the peak while $S < f S_{\max}$; the upper edge
does the same from $z_{\max}$. Each edge is interpolated linearly between the two channels on either side
of the level $f S_{\max}$ and converted to a velocity, $v_{f,\min}$ and $v_{f,\max}$ (km/s). The widths
are

$$
W_{50} = |v_{50,\max} - v_{50,\min}|, \qquad W_{20} = |v_{20,\max} - v_{20,\min}| ,
$$

and the systemic velocity is the mean of the midpoints of the two widths,

$$
V_{\rm sys} = \frac{1}{2}\left[\frac{v_{20,\min} + v_{20,\max}}{2} + \frac{v_{50,\min} + v_{50,\max}}{2}\right] ,
$$

or the midpoint of $W_{20}$ alone if $W_{20} > 1.5\,W_{50}$.

(search-pbcorr)=
### Primary-beam correction

With {ref}`PBCORR <par-search-pbcorr>` $\ge 1$, each value is divided by the response of the primary beam
$P(r, \nu)$ at the angular distance $r$ of its pixel from the pointing centre (CRVAL1, CRVAL2) and at the
frequency $\nu$ of its channel (from its velocity and the rest frequency). The models are those of the
telescope in the TELESCOP keyword: for WSRT, $P = \cos^6(c\,\nu\,r)$, with $\nu$ in GHz, $r$ in degrees
and $c$ = 61.18, 66.4 or 62.9 depending on the wavelength; for the VLA, ATCA and GMRT, polynomials in
$(r\nu)^2$ (with $r$ in arcmin), with coefficients for each band. For other telescopes no correction is
applied. Below the cutoff of the model, $P < P_{\min}$ ($P_{\min} = 0.023$; for ATCA, the response at
$r\nu$ = 50 arcmin GHz), the corrected value is 0 (`1`), $F/P_{\min}$ (`2`), NaN (`3`) or $F/P$ (`4`).
Only $S_{\rm int}$ is corrected.

(search-outputs)=
## Outputs

### The catalogue

`<prefix>_detections.txt` starts with the noise $\sigma$ and the two thresholds $T_1$ and $T_2$ (not with
`spatialsmooth`), in flux units, and has one line per detection, with the columns:

| Column | Content | Unit |
|---|---|---|
| Source | number of the detection, in the order of SORTSOURCES | |
| Center | centroid $x_c$, $y_c$, $z_c$, rounded | pixel, pixel, channel |
| Center (WCS) | right ascension and declination of the centroid, $v(z_c)$ | deg, deg, km/s |
| Xwidth, Ywidth | $x_{\max} - x_{\min} + 1$, $y_{\max} - y_{\min} + 1$ | pixels |
| Zwidth | $z_{\max} - z_{\min} + 1$ | channels |
| NumPix | $N_{\rm pix}$, distinct pixels on the sky | pixels |
| NumVox | $N_{\rm vox}$, voxels | |
| Vsys | $V_{\rm sys}$ | km/s |
| W20 | $W_{20}$ | km/s |
| Flux | $S_{\rm int}$ | Jy km/s (or `BUNIT` km/s) |
| PeakSNR | $\mathrm{S/N}_{\rm peak}$ | |
| PeakFlux | $F_{\rm peak}$ | `BUNIT` |

The WCS column is present when the cube has a header.

### Cubes and maps

- `<prefix>_detections.fits.gz`: the cube with the values of the detected voxels, 0 elsewhere;
- `<prefix>_detections_num.fits.gz`: the number of the detection of each voxel, 0 elsewhere;
- `<prefix>_detectmap.fits.gz`: the number of detected channels in each pixel;
- `<prefix>_mom0th.fits`, `<prefix>_mom1st.fits`, `<prefix>_mom2nd.fits`: the moment maps of the
  cube within the detected voxels (see {doc}`../maps/algorithm`); with
  {ref}`SNMAP <par-search-snmap>`, also the noise and S/N of the total map, `<prefix>_mom0th_RMS.fits`
  and `<prefix>_mom0th_SN.fits`.

### Cubelets

With {ref}`CUBELETS <par-search-cubelets>`, each detection gets a folder `sources/<prefix>_<n>` (just
`<prefix>` if there is one detection) with a sub-cube of the data covering

$$
[x_{\min} - E,\; x_{\max} + E] \times [y_{\min} - E,\; y_{\max} + E] \times [z_{\min} - 10,\; z_{\max} + 10] ,
$$

clipped to the edges of the cube, with $E$ = {ref}`EDGES <par-search-edges>` (pixels). The folder holds
the cubelet, its mask (1 in the voxels of the detection), its moment maps (`_mom0`, `_mom1`, `_mom2`),
its spectrum (`_spectrum.dat`: the velocity in km/s and $S(z)/\Omega_{\rm b}$, the flux of the detected
voxels in each channel converted to Jy) and, with PLOTS, plots of the detections.

## Differences from BBarolo 1.8

- The algorithm is the same, but objects are found by connected-component labelling and merged as
  connected components of a graph, instead of the object lists of Duchamp. The catalogue, the detection
  maps and the mask are identical to those of the C++ code on the test cubes.
- **ITERNOISE** stops when the noise changes by less than 0.1%; in 1.8 the convergence test was wrong
  and it always ran 20 iterations (see {doc}`../stats/algorithm`).
- The checks of SEARCHTYPE and of the growth threshold also apply when the source finder only builds
  a mask (`MASK SEARCH`).
- **Output names** have the prefix: `<prefix>_detections.txt`, `<prefix>_detections.fits.gz`,
  `<prefix>_detections_num.fits.gz`, `<prefix>_detectmap.fits.gz` (1.8: `detections.txt`, ...,
  `DetectMap.fits.gz`). With SNMAP, the noise and S/N maps are `<prefix>_mom0th_RMS.fits` and
  `<prefix>_mom0th_SN.fits`: in 1.8 they had the names of those of TOTALMAP, which overwrote them.

The full list is in {doc}`../../reference/changes`.

## References

- Whiting M. T., 2012, MNRAS, 421, 3242 (Duchamp):
  [2012MNRAS.421.3242W](https://ui.adsabs.harvard.edu/abs/2012MNRAS.421.3242W)
- Di Teodoro E. M., Fraternali F., 2015, MNRAS, 451, 3021:
  [2015MNRAS.451.3021D](https://ui.adsabs.harvard.edu/abs/2015MNRAS.451.3021D)

## In the code

- `bbarolo/search.py`: `Search` (`search`, `_search3D` and `_first_close` for the objects and the first
  stage, `_merger`, `_merge_labels` and `_neighbour_pairs` for the merging, `_grow`, `_finalise` for the
  rejection); `Detection` (`calcFluxes`, `calcWCSparams`, `calcIntegFlux`, `calcVelWidths`, `getVsys`);
  `SortDetections`.
- `bbarolo/cube.py`: `Cube.setCubeStats`, `Cube.search` (defaults of THRESHSPATIAL, MINPIX, MINVOXELS,
  MAXANGSIZE), `printDetections`, `writeDetections`, `extractCubelet`, `writeCubelets`.
- `bbarolo/stats.py`: `Stats`. `bbarolo/bbutils.py`: `Pbcor`, `FluxtoJy`.
- `bbarolo/tasks.py`: `_compute_search`, `_write_search`, `Search`.
