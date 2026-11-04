# Changes

Changes to BBarolo 2 since the version developed in Claude Cowork (2.0.0, November 2026).
Bugs marked **(C++)** are also in the C++ BBarolo 1.8; the others were introduced by the port.

## Unreleased

### Fixed

- **BayesianBBarolo: the rings of a run carried over (C++).** Every evaluation of the likelihood
  wrote the sampled values (and RADMAX and the density profile) into the initial rings. A second
  run on the same object then kept the parameters that were free in the first one at the last
  sampled values, not at the initial ones, and `save_results` saved those rings. The initial rings
  no longer change.
- **BayesianBBarolo: other bugs.**
  - `update()` discarded the priors given as distributions in `prior_distr` **(C++)**.
  - A negative parameter of a function of radius had zero likelihood, e.g. the slope of
    `vdisp_func`: half of its default prior **(C++)**. Only the values in the rings must now be
    positive (VROT, VDISP, INC, XPOS, YPOS, Z0, DENS), as before for the other free parameters.
  - `RUNOPTS dlogz` with dynesty's dynamic sampler, the default, raised a TypeError: it is
    `dlogz_init` there **(C++)**. With nautilus, dlogz becomes f_live = 1 − exp(−dlogz) (it was
    f_live = dlogz: the sampling stopped at once for dlogz ≥ 1), and `nlive` is accepted as
    `n_live`. An unknown option of SAMPLEROPTS or RUNOPTS is an error before the setup, with the
    options of the sampler (the samplers raised it after the setup).
  - A sampling that failed, or was interrupted, left the worker processes running **(C++)**.
  - `lambdas` with one ring raised an IndexError **(C++)**.
  - `load_results` of a `run_geometry` fit used the likelihood of the cube (e.g. in `print_stats`).
  - With MPI, dynesty's `queue_size` came from THREADS instead of the number of MPI workers.
- **Spatial axes in arcsec or arcmin (C++).** BMAJ and BMIN are in degrees in FITS files, but the
  code compared them with CDELT in the units of the axes. For such cubes the beam area, the beam in
  pixels, the ring width estimated from the beam (0.008 arcsec: a blind 3DFIT stopped with "Radius
  separation too small"), the region and the noise correlation of the Bayesian likelihood, and the
  header warnings were wrong. The spatial axes are now converted to degrees when the cube is read,
  so the results are those of the same cube in degrees, and the outputs are written in degrees.
- **3DFIT and GALMOD changed the parameters seen by later tasks.** The ring parameters that
  are estimated or defaulted (XPOS, VSYS, DVDZ, ZCYL, the RINGFILE columns, ...) were written
  into the parameters of the cube, so the next tasks of the same run used them. For example, a
  2DFIT after a 3DFIT used DVDZ=0 (its free angle around the minor axis) instead of 15°. They
  are now filled in a copy, as in C++. Results of runs with one task do not change.
- **Initial inclination of nearly edge-on galaxies.** The simplex of the initial-guess fit did
  not keep the point folded back by the fitted function (inclinations above 90° become
  180° − inc), so the estimate could end above 90° and fall back to 80°. The folded point is
  now kept, as in C++.
- **REDUCE (spatial): CRPIX of the reduced cube** was rounded to an integer, so the WCS
  was off by up to half a reduced pixel (the port also rounded halves to even, unlike C++). It
  is now the exact 0.5 + (CRPIX − 0.5)/FACTOR, as for the spectral axis. **(C++)**
- **ANALYTIC models of lagging disks (DVDZ ≠ 0)** had too few vertical samples, and the line
  profiles of the samples formed a comb: the global spectrum of a thick lagging disk was off by
  11–15% of its peak. Consecutive samples are now at most about 0.5 σ apart in velocity: 1.4–2.1%,
  as MC models with the default CDENS. These models are 3–6 times slower than before.
- **ANALYTIC models with no line width** (VDISP=0 and LINEAR=0) raised ZeroDivisionError. The
  line now falls in one channel, as in MC models.
- **Several emission lines on a frequency axis with REDSHIFT > 0** were placed at the rest
  frequencies times (1+z) instead of divided by it, and the model was empty. **(C++)**
- **Spectral axes in kHz, MHz or GHz, and wavelength axes in nm** gave wrong velocities and
  channel widths (frequencies were compared with rest frequencies in Hz, and kHz and nm were not
  converted). All the spectral axis types, units and velocity definitions now give the same
  velocities, channel widths and models (tested). **(C++)**
- **CUNIT3 = kms** was taken as cm/s (with the factor of mm/s) by the model, which came out
  empty. A velocity unit other than m/s, ms, km/s or kms is now an error. **(C++)**
- **Radio velocities on a wavelength axis** (VELDEF radio): the channel width was
  −c·λ₀/Δλ instead of c·Δλ/λ₀. **(C++)**
- **Flux units mJy/beam and uJy/beam** were taken as Jy/beam in the conversions to Jy, the mass
  surface density map (MASSDENSMAP) and `--jybeam2k`. **(C++)**
- **FREE=ALL with TWOSTAGE**: in the second stage DENS stayed free but was not counted among
  the free parameters, so the simplex was degenerate and, with FLAGERRORS, the fit crashed (in
  C++ the error columns were misaligned). **(C++)**
- **NORMALCUBE** left the data scaled after a fitting stage, for any later residual computed with
  the same fitter.
- **Initial estimates with ring parameters given as `file(...)` columns** (e.g. VSYS from a ring
  file, with the inclination to be estimated) raised ValueError. C++ read them as 0. **(C++)**
- **2DFIT with RADII and estimated geometry** cut RADII to the estimated number of rings (20 of
  the 40 radii of NGC 2403). **(C++)**
- **SEARCH with PBCORR**: W20 and W50 were measured on the PB-corrected spectrum, which is NaN
  beyond the cutoff (option 3). They come again from the observed fluxes, as in C++.
- **2D images**: the spectral coordinates of a 2D header (e.g. SEARCH on an image) raised
  IndexError.
- **Rotated WCS (PCi_j or CDi_j matrices).** The rotation was lost whenever the WCS was rebuilt
  (e.g. after swapping a Stokes axis), and output headers copied the CD cards of the input next to
  the new CDELT values, which is wrong after a spatial REDUCE. The rotation is now kept as a PC
  matrix with the CDELTs, and written as such. **(C++)**
- **REDUCE of an empty cube** crashed when writing DATAMAX = NaN.
- **Cubes with axes RA, DEC, STOKES, FREQ**: an empty CUNIT3 of the Stokes axis was taken as km/s,
  so the spectral axis was typed as velocity, and with several Stokes planes the data mixed them.
  Now the first Stokes plane and the right spectral axis. **(C++)**
- **`Rings()` started with RADSEP = −1** (C++: 0), which shrank by 0.5″ the disks of rings built
  one by one with RINGMODE=SHIFTED.
- **RADII given as Rmin~Rmax:step** could miss the last radius for steps that are not exact in
  binary (0~1:0.1 gave 10 radii instead of 11). **(C++)**
- **NaN ring parameters** put their flux in pixel or channel 0 on ARM processors (Apple Silicon);
  such clouds are now skipped on every platform.
- **ANALYTIC models with a Lorentzian layer (LTYPE 4)** put the far tail of the layer closer to
  the plane than it is: +1.4% of flux against MC models in a thick, inclined disk. Only the heights
  that can fall in the field are sampled now (+0.5%, the remaining sampling error).
- **Final models with many threads** used one full-cube buffer per thread: about 140 GB for a
  1024²×256 cube with 64 threads. At most 16 buffers (and 2 GB) are used now, with the same
  results (the same chunks, seeds and order of the sums). The same for GALWIND.
- **XPOS or YPOS from a ring file** (`file(path,col)`): the path was lowercased, so ring files in
  folders with capital letters were not found. **(C++)**
- **Relative paths of FITSFILE, FITSLIST entries and OUTFOLDER** (e.g. `../../Data/HI/n5055.fits`)
  were written as given in the plotting scripts, which then failed when run from another folder
  (e.g. `plot_all.py` to redo the plots). They become absolute paths (relative to the current
  folder) when the parameters are checked.
- **Centres given as 07h36m51s or +65d35m12s**: the first was taken as pixels (an error), the second as
  65°, without the minutes. **(C++)**
- **VERBOSE 2 was lost** after the mask was built (the level was restored as "true").
- **A failed asymmetric drift fit (ADRIFT)** was silent without VERBOSE; it is a warning now.
- **Rings rejected by 3DFIT** ("does not look good") now say why: the fitted VROT exceeds what the
  spectral band allows at the ring's VSYS and INC.
- **A 3DFIT with no ring fitted** gave an empty `Result.rings` without columns (`res.rings['vrot']`
  was a KeyError); the columns are kept, with a warning.
- **Memory of long Bayesian samplings with a free centre, Z0 or RADMAX**: a kernel transform was
  cached for every new model box size, without limit. Only the 32 most recent ones are kept now.
- **Parameter files**: `inf` or `nan` given to an integer parameter (CDENS, NV, BOX, ...) raised
  an error; they are unreadable values, as in C++.

### Added

- **GalMod3D: model cubes from 3D fields** (Python only, `bbarolo.GalMod3D`; GALMOD does not change).
  The gas is described by its density, velocity and isotropic velocity dispersion as functions of
  (R, z) (axisymmetric), (R, φ, z) or (x, y, z) in the frame of the galaxy, with the geometry of GALMOD
  (XPOS, YPOS, VSYS, INC, PA): e.g. flaring and lagging thick disks, bars and spiral arms with streaming
  motions, or the fields of a simulation. Each field is a number, a function of the coordinates, or
  values on a grid of the coordinates (interpolated linearly; φ periodic). The density is in any units;
  the model is scaled to a total flux in Jy km/s, or to the flux of the data in the mask of the cube
  (`flux="measured"`). The fields are sampled on a Cartesian grid in the frame of the galaxy (a third of
  a pixel by default) and each point is deposited with its Gaussian line integrated over the channels,
  as the ANALYTIC models (or as random clouds, `galmodtype="MC"`), with the spectral axis, several lines,
  the instrumental broadening and the beam convolution of GALMOD. A GALMOD disk written as fields gives
  the GALMOD ANALYTIC model within 0.05% of the peak (rms), apart from the central pixels (2%, where
  GALMOD samples the innermost sub-ring coarsely). A thick disk on the NGC 2403 grid (150×150×62,
  |z| ≤ 60") takes 1.7 s. `examples/galmod3d.py` makes three models: a thick lagging disk, a bar with
  spiral arms and a warped disk on a Cartesian grid.
- **The parameters of each run** in `<prefix>_params.par`, in the output folder (and in the `files`
  of the result): the parameters given (FITSFILE and OUTFOLDER as absolute paths), then the defaults of
  the other parameters of the tasks that ran (and of SEARCH or MAKEMASK when they make the mask), with
  the date and the folder of the run. `BBarolo -p <prefix>_params.par` runs it again with the same
  parameters. `Param.to_parfile(tasks)` gives it as a string.
- **`run_geometry` with a model error**: `sigma` and `fracerr` (free, or given in `like_kwargs`)
  in the likelihood of the total intensity map, as in that of the cube; without them the values
  do not change. The map of a real galaxy has structure that an axisymmetric disk does not
  reproduce: on NGC 2403 the residuals are about 20 times the noise in each pixel, so without a
  model error the posterior is extremely narrow (the centre within 0.006 pixels), the errors
  meaningless and the sampling very slow (100 live points: no convergence in 20 minutes on 8
  processes). With `sigma` free (it scales the noise of all the pixels) the fit converges in
  81 s on 16 processes: INC 50.9 ± 0.5°, PA 122.4 ± 0.7°, σ = 17.5 times the noise. `fracerr`
  does not suit the map: it weights the faint outer disk, and drove INC to 0.
- **Warning when the rings do not cover the disk** (BayesianBBarolo): more than 5% of the flux of
  the total intensity map beyond the outermost ring (+ half a beam). The region of the likelihood
  is that of the lowest INC allowed by the prior, so with a wide prior that emission biases INC
  low: on NGC 2403, with rings to 720" (8% of the flux beyond), the map gave INC 30–45° instead
  of 51°. A total intensity map does not change with PA → PA + 180°: restrict the prior of PA in
  `run_geometry` (the default, 0–360°, gives two modes).
- **`performance/bayes_coverage.py`**: warped, thick and lagging disks (`--kind`), Monte-Carlo
  fits (`--galmodtype MC`), 3DFIT on the same cubes (`--fit3d`), and the errors of the medians
  in the summary. Validation of the defaults (12 disks per case; S/N 5–40; 3, 5 and 8 beams; flat,
  warped, thick and lagging disks; ANALYTIC and MC fits; 144 fits, one failed: at S/N 5 SEARCH found no source):
  * the medians are more accurate than 3DFIT on the same cubes: at 3 beams and S/N 20, scatter of
    VROT 6.5 against 10.1 km/s, INC 1.2° against 2.3°, PA 0.7° against 1.9°; 3DFIT's INC is
    biased by −1.8° and −3.0° at 5 and 8 beams (the posterior: −0.3° and −0.4°);
  * the credible intervals of VROT are 2–3 times too wide (coverage of the 68% intervals
    0.92–1.00, pulls ± 0.3–0.5), those of INC and PA up to twice at 3 beams: the factor
    1/(correlation volume of the noise) is exact only for a signal smoother than the beam;
  * VDISP is biased low by 0.3–1.6 km/s at S/N 20–40 (coverage 0.25–0.75), with the data in the
    dilated mask or in all the voxels, and with the true density profile; at S/N 5 INC is
    biased by −7.5° (3DFIT: +5°);
  * MC fits, thick and lagging disks, and warps (INC and PA in each ring) behave as flat disks
    with ANALYTIC fits.
- **GUESSTYPE: new initial estimates** of the parameters that are not given (3DFIT, 2DFIT,
  SPACEPAR, BAYESIAN3DFIT). For now the default is LEGACY, the C++ algorithm.
  - **FIT**: starts from the ELLIPSE estimates and refines them. (1) A beam-convolved disk is fitted
    to the total intensity map (centre, INC, PA). The densities of its rings are deconvolved for each
    geometry (the new `bbarolo.profile`, shared with BayesianBBarolo), the map is the cube summed over
    the channels of the detection (not masked), and Z0 is used if given. (2) A 2D tilted-ring fit of
    the velocity field on rings one beam wide gives the kinematic PA and VSYS (and radial profiles of
    INC and PA, `rad_profile`, `inc_profile`, `pa_profile` of the estimates). (3) VDISP comes from the
    second moment, corrected for the channel width, instead of the fixed 8 km/s.
  - **ELLIPSE**: the C++ algorithm, corrected. The beam correction of the axes of the mask is
    k·FWHM in quadrature (C++: a full FWHM subtracted, which over-corrected the minor axis at low
    inclination, and no correction above 75°); no jumps (INC = 80 when the minor axis is smaller than
    the beam, 25 below 10); VROT always deprojected (C++: only above 40°); the values given are not
    replaced by the refinement with the 2D fit.
  - Validation (`performance/estimates.py`, the simulated disks of the performance suite: 240 flat
    disks at 3, 5 and 8 beams, INC 15–85°, peak S/N 5–40, and 80 warped disks). Error of the final
    3DFIT INC (median ± robust scatter; centre, VSYS and Z0 given, FREE VROT VDISP INC PA):

    | Estimates | 3 beams | 5 beams | 8 beams | warps | above 75° (3, 5, 8 beams) |
    |---|---|---|---|---|---|
    | LEGACY | +0.2 ± 9.1° | +1.7 ± 6.8° | +0.9 ± 4.9° | +0.8 ± 6.8° | −10.3, −4.7, −3.3° |
    | FIT | +2.0 ± 5.1° | +0.9 ± 4.6° | −0.4 ± 4.9° | −0.1 ± 4.9° | +3.1, +0.2, −0.7° |

    The bias of the initial INC of FIT (nothing given) is at most 2.6° in every bin of true INC
    (LEGACY: +5.8° at 15–30°, −6.9° above 75°) and does not depend on the peak S/N. Below 30° all the
    estimates scatter by 7–11°: a nearly round map hardly constrains INC. Started from ELLIPSE, 3DFIT
    scatters as much as from LEGACY (6–7°), with −3.3° on the warps. Starting 3DFIT from the
    radial INC and PA profiles did not recover warps better, so it is not an option. On NGC 2403 all
    the estimates give INC = 50–52° (53–54° after 3DFIT).
- **`performance/estimates.py`**: the bench of the initial estimates (simulated cubes, estimates with
  any GUESSTYPE or code version, and 3DFIT started from them).
- **Errors**: `bbarolo.BBaroloError` (a `ValueError`) for problems in the input, with a message that
  says what is wrong. The library never exits the Python session any more (a cube without a beam
  raised `SystemExit`), and errors no longer surface later as unrelated `AttributeError`s (SMOOTH to
  a smaller beam, PVSLICE outside the cube, GALWIND with inconsistent sizes). The command line
  prints the message and exits with code 1 (it exited with 0 after unacceptable parameters).
- **Checks before running**, with messages that name the parameter: ring values that are not
  numbers, lists of numbers, `file(...)` or WCS positions (also when a task class is built: e.g.
  `inc="sixty"`); unknown names in FREE (they were ignored); VSYS outside the velocity range of the
  cube (an error for 3DFIT, with a hint when it looks like m/s; a warning for GALMOD); a beam from
  BMAJ/BMIN/BEAMFWHM smaller than a quarter of a pixel (degrees given instead of arcsec).
- **Result** of 3DFIT (and GALMOD, BAYESIAN3DFIT): `data` and `residual` (data − model, NaN outside
  the mask), computed when first used; `rotation_curve` (radius, VROT and its errors as positive
  numbers); `plot_rings()`, `plot_maps()` and `plot_pv()`, which return matplotlib figures.
- **VERBOSE levels**: 0 (false) nothing; 1 (true, the default) the main messages, a progress bar
  and one line per ring in order of radius, with the outcome of the ring (e.g. why it was
  rejected); 2 everything, as before (the parameters, every ring in detail).
- **SPECRES**: the spectral resolution of the models (σ, in channels). LINEAR still sets it when
  SPECRES is not given, as in C++.

### Changed

- **Screen output.** Paths are relative to the current folder (unless that goes up more than two
  folders); the outputs, e.g. the plotting scripts, keep absolute paths. The notes are warnings
  (BBAROLO WARNING), like the other messages. With VERBOSE 1, the table of the second stage of 3DFIT
  has the columns of the first: the parameters regularised after the first stage, and then fixed, are
  marked (r), e.g. INC(r), with their regularised values, and a line between the stages says how they
  were regularised (e.g. `INC median = 52.85 deg (auto), PA Bezier curve`); the radii are RAD. The
  coefficients of the regularising polynomials are printed with VERBOSE 2 only.
- **Default Z0: one pixel.** When Z0 is not given, 3DFIT, BAYESIAN3DFIT, SPACEPAR and the initial
  estimates (GUESSTYPE FIT) use a thin disk, one pixel thick, instead of a disk with no thickness (C++:
  Z0 = 0). GALMOD still needs Z0. The examples give Z0, so their outputs do not change. On NGC 2403
  (pixel 16", beam 59", INC, PA, VROT and VDISP free, one stage) the medians over the 42 rings change
  by −0.2° in INC, +0.6 km/s in VROT and less than 0.05 in PA and VDISP.
- **No DISTANCE: the outputs are in arcsec.** DISTANCE only converted arcsec to kpc in the outputs
  (no fit, model or likelihood used it; without it the distance was VSYS/70 Mpc). It is gone, and so
  is what it gave:
  - the ring files of 3DFIT, BAYESIAN3DFIT and GALMOD (`rings_final1.txt`, ...) have no RAD(Kpc) and
    Z0(pc) columns. The columns are RAD(arcs), VROT, DISP, INC, P.A., Z0(arcs), DENS, XPOS, YPOS,
    VSYS, VRAD, then the errors: VROT is column 2, not 3 as in the files of C++;
  - the ring tables of the results have no `rad_kpc` and `z0_pc` (nor has `rotation_curve`), the
    screen output has the radii and Z0 in arcsec, and the note on the distance from the Hubble flow
    is gone;
  - SMOOTH has no LINEAR (a resolution in kpc): the new beam is BMAJ, BMIN, BPA or FACTOR. LINEAR is
    still the spectral resolution of the models when SPECRES is not given.

  DISTANCE in a parameter file is an unknown keyword (ignored, with a warning); `distance=` in Python
  is an error. RINGFILE now finds the columns by their names in the header, so it reads the ring files
  of bbarolo2 and of C++ (and `rings_initial.txt`, whose columns it read wrongly before); a file without
  a header must have the columns of C++, and the plotting scripts read the ring files by the names of
  their columns. A parameter read with `file(name,n)` from a BBarolo ring file
  must come from its own column: e.g. `VROT file(rings_final1.txt,3)` on a file of bbarolo2 (column 3
  is DISP) is an error that gives the right column, so the column numbers of the C++ files cannot read
  another quantity without notice.
- **BayesianBBarolo: RADMAX is the outer edge of the disk.** In the original module RADMAX moved the
  radius of the last ring, so the posterior of that ring mixed different annuli and the model
  extended half a ring beyond RADMAX. Now the rings stay, and the model has no emission beyond
  RADMAX (a sharp edge, continuous in RADMAX: each sub-ring of the model is weighted by its fraction
  within the edge), also in the density profile. The option `radmax` of `like_kwargs` / LIKEOPTS
  chooses: `edge` (the default), `taper` (a Gaussian edge of width `radmax_width`, by default a
  quarter of the beam FWHM) or `ring` (the RADMAX of the original module; results saved before
  the option existed are loaded with it). The default prior is still the last ring, its radius ±
  half the ring separation; with NODES rings, whose disk ends at the last radius, the edge lies
  in the half of it inside the last radius. The model engine has the edge too (`Rings.edge`,
  `Rings.edgewidth`). `performance/radmax.py` compares the ways of fitting the edge on simulated
  disks with a sharp edge inside the last ring (12 disks at 3 and 5 beams, S/N 20; errors in
  units of the ring width w):

  | RADMAX | edge: error, coverage (3 beams) | (5 beams) | VROT of the last ring: coverage |
  |---|---|---|---|
  | ring (before) | −0.02 ± 0.30 w, 0.75 | −0.04 ± 0.27 w, 0.83 | 0.83, 0.92 |
  | edge (default) | −0.02 ± 0.27 w, 0.75 | −0.13 ± 0.28 w, 0.75 | 1.00, 1.00 |
  | taper | −0.08 ± 0.30 w, 0.75 | −0.16 ± 0.31 w, 0.67 | 1.00, 1.00 |
  | evidence of N−1, N, N+1 rings | −0.21 ± 0.69 w | +0.86 ± 0.72 w | 0.83, 1.00 |

  RADMAX locates the edge as well with `ring` as with `edge`, but with `ring` the model extends
  half a ring beyond RADMAX and the last ring's VROT refers to a radius that changes from sample
  to sample. `taper` brings nothing for sharp edges, and the evidence is coarse (it chose an
  empty extra ring in 5 of 12 disks at 5 beams). Times are the same.
- **BayesianBBarolo: faster sampling on machines with many cores.** BLAS (e.g. MKL with 112
  threads on a 224-core server) gave many threads to the small matrices of the density profile in
  each likelihood call, and the processes of a parallel sampling competed for the cores. The
  sampling and the errors of the density profile now run with one BLAS thread per process
  (threadpoolctl, added to the `bayesian` extras; without it nothing changes). On NGC 2403, with
  the geometry free, a likelihood call takes 22 ms instead of 37 ms (MC: 32 instead of 70), and a
  sampling makes 202 likelihood calls per second instead of 76 with 8 processes, 491 instead of
  155 with 32. The values do not change. `performance/bench.py --sampling 1,8,32` times a
  sampling.
- **BayesianBBarolo is a package**, `bbarolo.bayesian` (it was a 2200-line module): the fitter,
  the likelihood, the density profile, the samplers and the pool, the outputs, the noise and the
  task in separate modules, and the setup of a fit in explicit steps (`_setup`). The imports
  (`from bbarolo.bayesian import BayesianBBarolo, noise_correlation, dilate_mask`) and the
  results do not change.
- **Output file names**: every output is now `<prefix>_<what>` (the prefix is OUTPREFIX, by default
  the OBJECT of the header). Before, some names had no prefix and others had it without the
  underscore. The contents do not change. With the prefix `ngc2403`:

  | Before | Now |
  |---|---|
  | `rings_initial.txt`, `rings_model.txt`, `rings_final1.txt`, `rings_final2.txt` | `ngc2403_rings_initial.txt`, ... |
  | `densprof.txt`, `asymdrift.txt`, `mask.fits` | `ngc2403_densprof.txt`, `ngc2403_asymdrift.txt`, `ngc2403_mask.fits` |
  | `ngc2403mod_azim.fits` (and `_local`, `_none`, `_noise`) | `ngc2403_mod_azim.fits`, ... |
  | `maps/ngc2403mod_0mom_azim.fits`, ... | `maps/ngc2403_mod_0mom_azim.fits`, ... |
  | `pvs/ngc2403mod_pv_a_azim.fits`, `pvs/ngc2403mask_pv_a.fits`, ... | `pvs/ngc2403_mod_pv_a_azim.fits`, `pvs/ngc2403_mask_pv_a.fits`, ... |
  | MAPS, 2DFIT: `ngc2403map_0th.fits`, `ngc2403map_1st.fits`, ..., `spectrum.txt` | `ngc2403_map_0th.fits`, ..., `ngc2403_spectrum.txt` |
  | SEARCH: `detections.txt`, `detections.fits.gz`, `detections_num.fits.gz`, `DetectMap.fits.gz` | `ngc2403_detections.txt`, ..., `ngc2403_detectmap.fits.gz` |
  | PVSLICE: `ngc2403pv_123.fits` | `ngc2403_pv_123.fits` |
  | ELLPROF: `ngc2403_densprof.txt`, `ngc2403_densmap.fits` | `ngc2403_ellprof.txt`, `ngc2403_ellprof_map.fits` (the 3DFIT profile is now `ngc2403_densprof.txt`) |
  | SPACEPAR: `spacepar_ngc2403.fits`, `.pdf` | `ngc2403_spacepar.fits`, `.pdf` |
  | `initialguesses_ngc2403.pdf`, `densprof_deconvolved.txt` (BAYESIAN3DFIT) | `ngc2403_initialguesses.pdf`, `ngc2403_densprof_deconvolved.txt` |

  The plotting scripts and `Result` follow the new names; the keys of a Result (`maps['mod_0mom_azim']`,
  `pvs['pv_a']`, ...) do not change. Scripts that read the files by name need the new names.
- **THREADS** defaults to the number of cores, at most 16 (all the cores of a large shared server
  were too many). The ring tables do not depend on THREADS; MC final models do (one seed per chunk).
- **Default output** is much shorter (VERBOSE 1, above): about 140 lines for the NGC 2403 3DFIT
  instead of about 400. Progress bars are drawn on a terminal only (in a log file their backspaces
  piled up). A warning says when an existing output folder is overwritten.
- **A misspelled keyword in a parameter file** (very close to a real one, e.g. VDSIP) is an error;
  other unknown keywords are still ignored with a warning. All the keywords of the C++ code are
  known, so valid parameter files are not affected.
- **A velocity axis without CUNIT3** is taken as m/s if its channels are wider than 100, otherwise as
  km/s (C++: always km/s, or m/s when no CUNIT is given at all).
- **Help**: `-d TASK` lists the beam keywords; `bbarolo.help('dist')` prints the close
  matches instead of raising; clearer help for FREE; the FitMod3D docstring lists the columns of the
  results and the convention of the errors.
- **BayesianBBarolo and BAYESIAN3DFIT use ANALYTIC models by default** (GALMODTYPE; 3DFIT, GALMOD
  and the other tasks keep MC). Their likelihood has no Monte-Carlo noise: with sigma equal to the
  noise, the noise of MC models made it rough and the sampling slow. A likelihood call is also about
  twice faster (NGC 2403, 6 rings: 18 ms against 33 ms). `GALMODTYPE MC` gives the previous models;
  `examples/compare_bayesian.py`, which compares with the C++-based module, sets it.
- The input file is recorded with its absolute path (in the HISTORY of the mask, the plotting
  scripts and the printed parameters), and `Result.files` and `Result.outfolder` are absolute.
- Output headers no longer carry the CD matrix of the input: the WCS is written as CDELT, plus PC
  when the axes are rotated (the same coordinates).
