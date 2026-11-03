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
- **Spectral axes in kHz, MHz or GHz** were compared with rest frequencies in Hz: wrong
  velocities and channel widths. **(C++)**
- **CUNIT3 = kms** was taken as cm/s (with the factor of mm/s) by the model, which came out
  empty. A velocity unit other than m/s, ms, km/s or kms is now an error. **(C++)**
- **Radio velocities on a wavelength axis** (VELDEF radio): the channel width was
  −c·λ₀/Δλ instead of c·Δλ/λ₀. **(C++)**
- **Flux units mJy/beam and uJy/beam** were taken as Jy/beam in the conversions to Jy, the mass
  surface density map (MASSDENSMAP) and `--jybeam2k`. **(C++)**
- **FREE=ALL with TWOSTAGE**: in the second stage DENS stayed free but was not counted among
  the free parameters, so the simplex was degenerate and, with FLAGERRORS, the fit crashed (in
  C++ the error columns were misaligned). **(C++)**
