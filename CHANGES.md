# Changes

Changes to BBarolo 2 since the version developed in Claude Cowork (2.0.0, November 2026).
Bugs marked **(C++)** are also in the C++ BBarolo 1.8; the others were introduced by the port.

## Unreleased

### Fixed

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
- **Parameter files**: `inf` or `nan` given to an integer parameter (CDENS, NV, BOX, ...) raised
  an error; they are unreadable values, as in C++.

### Added

- **SPECRES**: the spectral resolution of the models (σ, in channels). LINEAR still sets it when
  SPECRES is not given, as in C++, but LINEAR is also the new resolution of SMOOTH (in kpc): the
  parameter check now warns when one run uses LINEAR for both.

### Changed

- The input file is recorded with its absolute path (in the HISTORY of the mask, the plotting
  scripts and the printed parameters), and `Result.files` and `Result.outfolder` are absolute.
- Output headers no longer carry the CD matrix of the input: the WCS is written as CDELT, plus PC
  when the axes are rotated (the same coordinates).
