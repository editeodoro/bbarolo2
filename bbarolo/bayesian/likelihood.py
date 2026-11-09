"""The likelihood of BayesianBBarolo: the voxels and the region of the likelihood, the noise and its
correlation, the residuals and the log-likelihoods of the cube and of the total intensity map."""

import math
import numpy as np
import numba as nb

from .noise import dilate_mask, noise_correlation
from ..galmod import _half_widths


@nb.njit(cache=True, nogil=True)
def _sumsq_residuals(model, data, mask, maskmode, region):
    """sum over voxels of (d - model)^2, skipping NaN data (as np.nansum) and, if region
    (ny,nx) is not empty, the pixels outside region. maskmode: 0 = all the voxels, d = data;
    1 (or True) = d = data*mask (zero outside the mask); 2 = only the voxels in the mask,
    d = data (the other voxels are skipped)."""
    nz, ny, nx = model.shape
    useregion = region.shape[0] > 0
    s = 0.
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                if useregion and not region[j, i]:
                    continue
                d = data[k, j, i]
                if d != d:
                    continue
                if maskmode != 0 and not mask[k, j, i]:
                    if maskmode == 2:
                        continue
                    d = 0.
                r = d - model[k, j, i]
                s += r * r
    return s


@nb.njit(cache=True, nogil=True)
def _residuals_fracerr(model, data, mask, maskmode, region, sigma2, eps2):
    """As _sumsq_residuals (same voxels, mask and region), with a variance per voxel
    sigma^2 + eps^2 * model^2. Returns (sum of r^2 / variance, sum over the voxels where the
    model is not zero of log(variance / sigma^2))."""
    nz, ny, nx = model.shape
    useregion = region.shape[0] > 0
    s = 0.
    lg = 0.
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                if useregion and not region[j, i]:
                    continue
                d = data[k, j, i]
                if d != d:
                    continue
                if maskmode != 0 and not mask[k, j, i]:
                    if maskmode == 2:
                        continue
                    d = 0.
                m = model[k, j, i]
                r = d - m
                if m != 0.:
                    v = sigma2 + eps2 * m * m
                    s += r * r / v
                    lg += math.log(v / sigma2)
                else:
                    s += r * r / sigma2
    return s, lg


class LikelihoodMixin(object):
    """ The likelihood of BayesianBBarolo (see the documentation of bbarolo.bayesian). """

    def _setup_likelihood(self):
        """ The likelihood of the fit, from like_kwargs and the priors: its voxels (_setup_likemask),
        its region (_setup_likebox, from the bounds of the priors), the factor for the correlation of
        the noise (_set_likefactor) and the noise of the total intensity map (map_rms). """
        self._setup_likemask()
        self._setup_likebox()
        self._set_likefactor()
        self.map_rms = self._map_noise(self._nvox_map)

    def _setup_noise(self):
        """ Noise and correlation scale of the data, used by the likelihood.

        noiserms:   rms noise per voxel, from the median absolute deviation of the data
                    outside the mask;
        beam_area:  area of the (Gaussian) beam in pixels, pi*bmaj*bmin/(4 ln2)/pixel area:
                    the number of pixels that share the same noise (1 if the beam is unknown);
        likefactor: multiplicative factor of the log-likelihood (see _set_likefactor): by
                    default 1/(correlation volume of the noise), so that each independent
                    noise measurement, not each voxel, counts once.
        Also: the voxels of the likelihood (see _setup_likemask), and the maps of their
        squared data and number, used with the likelihood region (see _setup_likebox).
        """
        out = self.data[(self.mask == 0) & np.isfinite(self.data)]
        if out.size == 0:
            out = self.data[np.isfinite(self.data)]
        self.noiserms = float(1.4826 * np.median(np.abs(out - np.median(out))))
        if not self.noiserms > 0:
            self.noiserms = float(np.nanstd(out)) if out.size else 1.
        h = self._cube_obj.Head()
        bmaj, bmin = abs(h.Bmaj()), abs(h.Bmin())
        pixarea = abs(h.Cdelt(0) * h.Cdelt(1))
        if bmaj > 0 and bmin > 0 and pixarea > 0:
            self.beam_area = float(np.pi * bmaj * bmin / (4. * np.log(2.)) / pixarea)
        else:
            print("BAYESIAN WARNING: the beam of the cube is unknown, the pixels are treated as independent.")
            self.beam_area = 1.
        self.noise_correlation = None
        self._set_likefactor()
        self._setup_likemask(force=True)
        self.map_rms = self._map_noise(self._nvox_map)
        _, ys, xs = self.data.shape
        self._likebox = ([0, 0], [xs, ys])
        self._region = self._side_region(np.ones((ys, xs), bool))
        self._set_region_sums()

    def _setup_likemask(self, force=False):
        """ The voxels of the likelihood, from like_kwargs['likemask']:

        'dilated' (default): the data, not masked, in the voxels of the mask dilated by
                           like_kwargs['dilation'] = (radius in units of the beam FWHM, number
                           of channels), default (1, 3): a disk of 1 FWHM radius in each
                           channel, then +-3 channels. The other voxels are not used;
        'none':            the data, not masked, in all the voxels of the likelihood region.

        The model is compared with the data in a fixed set of voxels: data selected through
        the mask of their neighbours, not their own values, and not clipped. (The original
        module compared it with the data times the mask, i.e. with zero below the threshold
        of the mask, which penalised the faint model emission there and biased the fits at
        low S/N: VDISP too low, INC too high.) The dilation should include the faint wings:
        with channels much narrower than the velocity dispersion, increase the number of
        channels (about 1.5 VDISP / channel width).

        Sets _likemask (the mask passed to the residuals), _maskmode (2: only the voxels in
        _likemask), the maps of the squared data and of the number of valid voxels of the
        likelihood in each pixel (_dm2_map, _nvox_map), and the total intensity map of the
        same voxels (data_mom0: run_geometry, the density profile). When the choice changes,
        the Ellprof engine of the outputs is reset.
        """
        lk = self.like_kwargs
        mode = str(lk['likemask']).lower()
        if mode not in ('dilated', 'none'):
            raise ValueError(f"ERROR! likemask must be 'dilated' or 'none', not {lk['likemask']!r}.")
        try:
            nbeam, nchan = (float(x) for x in lk['dilation'])
        except Exception:
            raise ValueError(f"ERROR! dilation must be (radius in beams, number of channels), not {lk['dilation']!r}.")
        if not (nbeam >= 0 and nchan >= 0):
            raise ValueError(f"ERROR! dilation must be >= 0: {lk['dilation']!r}.")
        nchan = int(round(nchan))
        key = (mode, nbeam, nchan) if mode == 'dilated' else (mode,)
        if key == self._likemask_key and not force:
            return
        m = np.asarray(self.mask, bool)
        if mode == 'dilated':
            h = self._cube_obj.Head()
            bpix = abs(h.Bmaj() / h.Cdelt(0)) if h.Bmaj() and h.Cdelt(0) else 1.
            lm, self._maskmode = dilate_mask(m, nbeam * bpix, nchan), 2
            self._likemask_note = (f"data in the mask dilated by {nbeam:g} beam (radius {nbeam * bpix:.1f} pixels) "
                                   f"and +-{nchan} channels")
        else:
            lm, self._maskmode = np.ones(m.shape, bool), 2
            self._likemask_note = "all the voxels of the region"
        self._likemask = lm
        nz, ny, nx = self.data.shape
        self._dm2_map = np.zeros((ny, nx))
        self._nvox_map = np.zeros((ny, nx), np.int64)
        mom0 = np.zeros((ny, nx))
        for k in range(0, nz, 16):                     # in chunks of channels (no full float64 copies)
            v = lm[k:k + 16] & np.isfinite(self.data[k:k + 16])
            d = np.where(v, self.data[k:k + 16], 0.).astype(np.float64)
            self._dm2_map += np.sum(d * d, axis=0)
            self._nvox_map += v.sum(axis=0)
            mom0 += d.sum(axis=0)
        self.data_mom0 = mom0
        if self._likemask_key is not None and key != self._likemask_key:
            self._ellprof = None
        self._likemask_key = key

    def _likemask_map(self):
        """ Total intensity map of the data in the voxels of the likelihood, in the units of
        the zeroth moment (MomentMap): sum x channel width (Jy/beam km/s, or as the cube), NaN
        in the pixels without such voxels. The map of the observed density profile of the
        outputs (see _observed_profile). """
        from ..bbutils import DeltaVel, FluxtoJyBeam
        c = self._cube_obj
        dv = abs(DeltaVel(c.Head())) if c.HeadDef() else 1.
        m0 = self.data_mom0
        if c.HeadDef() and c.pars().fluxConvert:
            m0 = FluxtoJyBeam(m0, c.Head())
        return np.where(self._nvox_map > 0, m0 * dv, np.nan).astype(np.float32)

    def _map_noise(self, nchan):
        """ rms noise of the sum of nchan channels (array), for the total intensity map of
        run_geometry: noiserms * sqrt(nchan) for independent channels; with the measured
        correlation of the noise (noisecorr='measured'), the channels are assumed contiguous,
        variance = noiserms^2 * (n + 2 sum_l (n - l) rho_l) over the spectral lags l of the positive
        core (up to the first lag with rho_l <= 0). """
        n = np.asarray(nchan, np.float64)
        var = n.copy()
        nc = self.noise_correlation if str(self.like_kwargs['noisecorr']).lower() == 'measured' else None
        if nc is not None:
            rho = np.asarray(nc['rho_spectral'], float)
            for lag in range(1, len(rho)):
                if rho[lag] <= 0:               # (the positive core only: conservative, as the likelihood)
                    break
                var += 2. * np.clip(n - lag, 0., None) * rho[lag]
        return self.noiserms * np.sqrt(np.clip(var, 0., None))

    def _set_region_sums(self):
        """ Summed-area table of the squared data of the likelihood in the region, and nvox. """
        a = np.where(self._region, self._dm2_map, 0.)
        self._dm2_sat = np.zeros((a.shape[0] + 1, a.shape[1] + 1))
        self._dm2_sat[1:, 1:] = np.cumsum(np.cumsum(a, axis=0), axis=1)
        self._dm2_box = float(self._dm2_sat[-1, -1])
        self.nvox = int(np.sum(self._nvox_map[self._region]))

    def _set_likefactor(self):
        """ Multiplicative factor of the log-likelihood for the correlation of the noise:
        1/(number of voxels per independent measurement), from like_kwargs['noisecorr']:

        'measured' (default): 1/(correlation volume of the noise, measured from the voxels
                          without emission, spatial x spectral; see noise_correlation), and
                          1/(spatial correlation area) for the likelihood of run_geometry;
                          if it cannot be measured (too few voxels without emission, e.g.
                          MASK=NONE), the beam area. The measure is conservative (wider
                          posteriors rather than narrower): the spatial area is the largest
                          of the sum of the autocorrelation of the noise, its sum over the
                          positive core and the beam area; the spectral one the largest of
                          the sum, the positive core and one channel. (The sum alone can be
                          much smaller than the beam for the noise of synthesis images,
                          whose autocorrelation has a negative ring: e.g. 0.2 beam areas.)
        'beam':           1/(beam area in pixels), spatial correlation only (the default
                          before November 2026);
        a number N > 0:   N voxels (and N pixels of the total intensity map) per independent
                          noise value: the log-likelihood x 1/N (N=1: independent voxels).
        Sets likefactor (3D likelihood) and likefactor2d (total intensity map). """
        mode = self.like_kwargs['noisecorr']
        A = A2 = max(self.beam_area, 1.)
        self._likefactor_note = f"1/beam area in pixels = 1/{self.beam_area:.2f}"
        self._corr_below_beam = False                  # (set by 'measured' only: not kept from a previous setup)
        if isinstance(mode, (int, float, np.integer, np.floating)) and not isinstance(mode, (bool, np.bool_)):
            if not (np.isfinite(mode) and mode > 0):
                raise ValueError(f"ERROR! noisecorr must be 'beam', 'measured' or a number > 0, not {mode!r}.")
            A = A2 = float(mode)
            self._likefactor_note = f"1/{A:g} (noisecorr given: voxels per independent noise value)"
        elif str(mode).lower() == 'measured':
            if self.noise_correlation is None:
                h = self._cube_obj.Head()
                bpix = abs(h.Bmaj() / h.Cdelt(0)) if h.Bmaj() and h.Cdelt(0) else None
                self.noise_correlation = noise_correlation(self.data, self.mask, bpix)
            nc = self.noise_correlation
            if nc is None:
                print("BAYESIAN WARNING: too few voxels without emission to measure the correlation of the "
                      "noise:\n         using the beam area (like_kwargs noisecorr='beam').")
            else:
                sp = max(nc['spatial'], nc.get('spatial_core', nc['spatial']), self.beam_area)
                sz = max(nc['spectral'], nc.get('spectral_core', nc['spectral']), 1.)
                A, A2 = sp * sz, sp
                self._corr_below_beam = nc['spatial'] < 0.5 * self.beam_area
                self._likefactor_note = (f"1/{A:.2f} (measured noise correlation: {sp:.2f} pixels x {sz:.2f} "
                                         f"channels, at least one beam area, {self.beam_area:.2f} pixels; sum of the "
                                         f"autocorrelation {nc['spatial']:.2f} pixels x {nc['spectral']:.2f} channels)")
        elif str(mode).lower() != 'beam':
            raise ValueError(f"ERROR! noisecorr must be 'beam', 'measured' or a number, not {mode!r}.")
        self.likefactor = 1. / A
        self.likefactor2d = 1. / A2

    @staticmethod
    def _boxsum(S, blo, bhi):
        return float(S[bhi[1], bhi[0]] - S[blo[1], bhi[0]] - S[bhi[1], blo[0]] + S[blo[1], blo[0]])

    def _setup_likebox(self):
        """ Likelihood region: the pixels where the model can be non-zero for any value of the
        free parameters allowed by the priors. It is the projected disk within the outermost
        radius (the upper bound of radmax if fitted, for the lowest inclination and all the
        position angles allowed), widened by 3*z0 + 0.5 beam FWHM (+ the allowed shift of
        the centre), within the union of the model boxes (DiskModel.box). The region
        contains about 99% of the flux of the (convolved) model.

        The residuals are computed in this fixed region (the model is zero outside its own
        box), so that the likelihood does not depend on the box in which the model is
        calculated. Data beyond the outermost ring are not part of the fit (they would
        otherwise inflate a fitted sigma when only the inner rings are modelled). nvox,
        the number of valid voxels of the likelihood within the region (in the mask, or in
        the dilated mask, or all, see _setup_likemask), is the number of measurements of
        the normalisation term.
        """
        _, ys, xs = self.data.shape

        def upper(key, default):
            d = self.prior_distr.get(key) if key in self.freepar_idx else None
            if d is None:
                return default
            return float(d.ppf(1.)) if np.isfinite(d.ppf(1.)) else float(d.ppf(1 - 1e-6))

        def lower(key, default):
            d = self.prior_distr.get(key) if key in self.freepar_idx else None
            if d is None:
                return default
            return float(d.ppf(0.)) if np.isfinite(d.ppf(0.)) else float(d.ppf(1e-6))
        z0free = any(k.startswith('z0') for k in self.freepar_idx)
        if z0free and 'z0' not in self.freepar_idx:            # z0 as a function of radius
            blo, bhi = [0, 0], [xs, ys]
        else:
            rings = self._inri.object.copy()
            if self._radmax_mode == 'ring':             # (otherwise the edge is within the rings)
                rings.radii[-1] = upper('radmax', float(rings.radii[-1]))
                if 'radmax' in self.freepar_idx and rings.nr > 1:
                    rings.radsep = float(np.float32(rings.radii[-1] - rings.radii[-2]))
            rings.z0[-1] = upper('z0', float(np.max(rings.z0)))
            xp, yp = float(rings.xpos[-1]), float(rings.ypos[-1])
            corners = []
            for x in (lower('xpos', xp), upper('xpos', xp)):
                for y in (lower('ypos', yp), upper('ypos', yp)):
                    rings.xpos[-1], rings.ypos[-1] = x, y
                    corners.append(self._galfit.model.box(rings, final=True))
            blo = [min(c[0][0] for c in corners), min(c[0][1] for c in corners)]
            bhi = [max(c[1][0] for c in corners), max(c[1][1] for c in corners)]
            blo = [max(int(blo[0]), 0), max(int(blo[1]), 0)]
            bhi = [min(int(bhi[0]), xs), min(int(bhi[1]), ys)]
        self._likebox = (blo, bhi)
        self._region = np.zeros((ys, xs), bool)
        self._region[blo[1]:bhi[1], blo[0]:bhi[0]] = True
        if not (z0free and 'z0' not in self.freepar_idx):
            self._region[blo[1]:bhi[1], blo[0]:bhi[0]] = self._disk_region(blo, bhi, upper, lower)
        self._region = self._side_region(self._region)
        self._set_region_sums()

    def _side(self):
        """ The half of the galaxy of the likelihood (SIDE): 'A' (approaching), 'R' (receding) or None (both). """
        c = self._cube_obj
        side = str(c.pars().gf.SIDE).strip().upper()[:1] if c is not None else "B"
        return side if side in ("A", "R") else None

    def _side_region(self, region):
        """ region (a map of the pixels of the likelihood) restricted to the half of the galaxy of SIDE: the
        pixels on the approaching (A) or receding (R) side of the minor axis of the initial rings (their median
        centre and position angle; the pixels on the axis are in both halves). The half is fixed during the
        sampling, so that the likelihood always compares the model with the same voxels. """
        side = self._side()
        if side is None or self._inri is None:
            return region
        r = self._inri.r
        xc, yc = float(np.median(r['xpos'])), float(np.median(r['ypos']))
        pa = float(np.median(np.unwrap(np.radians(r['phi']))))        # (PAs across 0/360: unwrapped)
        ys, xs = region.shape
        yy, xx = np.mgrid[0:ys, 0:xs]
        xr = -(xx - xc) * np.sin(pa) + (yy - yc) * np.cos(pa)       # along the major axis: > 0 receding
        return region & (xr >= 0 if side == "R" else xr <= 0)

    def _disk_region(self, blo, bhi, upper, lower):
        """ Pixels of the box [blo,bhi) within the projected outer disk, widened (see _setup_likebox). """
        from scipy.ndimage import distance_transform_edt
        h = self._cube_obj.Head()
        pix = 0.5 * (abs(h.Cdelt(0)) + abs(h.Cdelt(1))) * 3600.          # arcsec
        r = self._inri.r
        radii = np.asarray(r['radii'], float)
        # (one ring: its whole width, RADSEP, or R/2 to 3R/2 as the model; it was 0, the region ended at R)
        radsep = float(radii[-1] - radii[-2]) if len(radii) > 1 else 2. * _half_widths(self._inri.object)[1]
        rout = upper('radmax', float(radii[-1])) if self._radmax_mode == 'ring' else float(radii[-1])
        if 'radmax' in self.freepar_idx and len(radii) > 1 and self._radmax_mode == 'ring':
            radsep = rout - float(radii[-2])
        rout += radsep / 2.
        z0max = upper('z0', float(np.max(r['z0'])))
        inc = np.asarray(r['inc'], float)
        incmin = min(float(inc.min()), lower('inc', float(inc.min())))
        incmin = min(max(incmin, 0.), 89.)
        phis = np.asarray(r['phi'], float)
        if 'phi' in self.freepar_idx:
            plo, phi_hi = lower('phi', float(phis.min())), upper('phi', float(phis.max()))
            phis = np.linspace(plo, phi_hi, 73) if phi_hi - plo < 180. else np.array([0.])
            if phi_hi - plo >= 180.:
                incmin = 0.                                              # any orientation: a circle
        xc, yc = float(r['xpos'][-1]), float(r['ypos'][-1])
        shift = max(abs(upper('xpos', xc) - xc), abs(lower('xpos', xc) - xc),
                    abs(upper('ypos', yc) - yc), abs(lower('ypos', yc) - yc)) * pix
        yy, xx = np.mgrid[blo[1]:bhi[1], blo[0]:bhi[0]]
        dx, dy = (xx - xc) * pix, (yy - yc) * pix
        cinc = np.cos(np.radians(incmin))
        disk = np.zeros(xx.shape, bool)
        for ph in np.unique(phis):
            sp, cp = np.sin(np.radians(ph)), np.cos(np.radians(ph))
            xr = -dx * sp + dy * cp
            yr = (-dx * cp - dy * sp) / cinc
            disk |= xr * xr + yr * yr <= rout * rout
        if not disk.any():
            return np.ones(xx.shape, bool)
        margin = 3. * z0max + 0.5 * abs(h.Bmaj()) * 3600. + shift + 0.5 * pix
        return distance_transform_edt(~disk) * pix <= margin

    def _model_box(self, rings):
        """ Box [blo,bhi) of the model for the given rings, within the likelihood region. """
        blo, bhi = self._galfit.model.box(rings.object, final=True)
        Lb, Hb = self._likebox
        blo = [max(blo[0], Lb[0]), max(blo[1], Lb[1])]
        bhi = [max(min(bhi[0], Hb[0]), blo[0]), max(min(bhi[1], Hb[1]), blo[1])]
        return blo, bhi

    def _sumsq_outside(self, blo, bhi):
        """ Sum of the squared masked data in the likelihood region outside the model box
        [blo,bhi), where the model is zero. """
        return max(self._dm2_box - self._boxsum(self._dm2_sat, blo, bhi), 0.)

    def _calculate_model(self, rings, fullcube=False):
        """ Unnormalized (convolved) model for a set of rings.

        Returns (mod, bhi, blo, None): mod is the model in the box [blo,bhi) (the full cube
        if fullcube=True). The last element (a C++ pointer in the original) is kept for
        compatibility.
        """
        _, ys, xs = self.data.shape
        bhi, blo = [xs, ys], [0, 0]
        if not fullcube:
            # Model built in a smaller box containing the galaxy within the last ring
            blo, bhi = self._model_box(rings)
        mod = self._galfit.model.array(rings.object, blo, bhi)
        return mod, np.array(bhi), np.array(blo), None

    def _calculate_residuals(self, model, data, mask=None, **kwargs):
        """ Default residuals (-log-likelihood): Gaussian with noise sigma per voxel. Can be overridden.

        kwargs: sigma (default: the measured noise rms), region (pixels of the box that are in
        the likelihood region, default all), maskmode (how the mask is used: 2 = only the
        voxels in the mask, the default, as in the likelihood, see _setup_likemask; 1 =
        data*mask, zero outside the mask, as the original module; 0 = mask not used), ss_out (sum
        of the squared data of the likelihood in the region outside the model box, default 0),
        nvox (number of measurements for the
        normalisation term, default the voxels of the likelihood in the region), fracerr
        (fractional model error eps: the variance of a voxel is sigma^2 + (eps*model)^2,
        default 0).
        """
        sigma = kwargs.get('sigma')
        if sigma is None:
            sigma = self.noiserms
        if not sigma > 0:
            return np.inf
        region = kwargs.get('region')
        if region is None:
            region = np.zeros((0, 0), bool)
        maskmode = int(kwargs.get('maskmode', 2)) if mask is not None else 0
        eps = kwargs.get('fracerr')
        if eps is not None and eps > 0:
            return self._residuals_fracerr(model, data, mask, region, sigma, float(eps),
                                           ss_out=kwargs.get('ss_out', 0.), nvox=kwargs.get('nvox'),
                                           maskmode=maskmode)
        if model.ndim == 3 and data.shape == model.shape and (mask is None or mask.shape == model.shape):
            ss = _sumsq_residuals(np.asarray(model, np.float64), data,
                                  mask if mask is not None else data, maskmode, region)
        else:
            d, w = self._masked(data, mask, maskmode)
            r2 = np.where(w, (d - model) ** 2, 0.)
            ss = np.nansum(r2 * region[None]) if region.size else np.nansum(r2)
        ss += kwargs.get('ss_out', 0.)
        nvox = kwargs.get('nvox', getattr(self, 'nvox', 1))
        return 0.5 * ss / sigma ** 2 + 0.5 * nvox * np.log(2. * np.pi * sigma ** 2)

    @staticmethod
    def _masked(data, mask, maskmode):
        """ (data used, voxels used) for maskmode (see _calculate_residuals). """
        if mask is None or maskmode == 0:
            return data, True
        if maskmode == 2:
            return data, np.asarray(mask, bool)
        return data * mask, True

    def _residuals_fracerr(self, model, data, mask, region, sigma, eps, ss_out=0., nvox=None, maskmode=None):
        """ -log-likelihood with the fractional model error eps (see _calculate_residuals):
        0.5 * sum[r^2 / (sigma^2 + eps^2 m^2) + log(2 pi (sigma^2 + eps^2 m^2))] over the
        voxels, where the voxels outside the model box (ss_out, nvox) have variance sigma^2. """
        s2, e2 = float(sigma) ** 2, float(eps) ** 2
        if maskmode is None:
            maskmode = 2 if mask is not None else 0
        if model.ndim == 3 and data.shape == model.shape and (mask is None or mask.shape == model.shape):
            ss, lg = _residuals_fracerr(np.asarray(model, np.float64), data, mask if mask is not None else data,
                                        int(maskmode), region, s2, e2)
        else:
            d, used = self._masked(data, mask, maskmode)
            v = s2 + e2 * np.asarray(model, np.float64) ** 2
            w = (region[None] if region.size else 1.) * np.asarray(used, float)
            ss = float(np.nansum(np.where(w > 0, (d - model) ** 2 / v, 0.) * w))
            lg = float(np.nansum(np.where(np.isfinite(d) & (model != 0) & (w > 0), np.log(v / s2), 0.)))
        ss += ss_out / s2
        if nvox is None:
            nvox = getattr(self, 'nvox', 1)
        return 0.5 * ss + 0.5 * nvox * np.log(2. * np.pi * s2) + 0.5 * lg

    # Ring parameters that cannot be negative
    _POSITIVE = ('vrot', 'vdisp', 'inc', 'xpos', 'ypos', 'z0', 'dens')

    def _negative(self, rings, theta):
        """ True if RADMAX, or a value in the rings of a free VROT, VDISP, INC, XPOS, YPOS, Z0 or DENS
        (per ring, single or function of radius), is negative: the likelihood is then zero. The
        parameters of a function of radius may be negative (e.g. the slope of VDISP). """
        if 'radmax' in self.freepar_idx and theta[self.freepar_idx['radmax'][0]] < 0:
            return True
        return any(np.any(rings.r[k] < 0) for k in self._free_positive)

    def _log_likelihood(self, theta, **kwargs):
        """ Log-likelihood of the parameters theta.

        Gaussian likelihood of the data in the voxels of the likelihood (likemask: by default
        the data, not masked, in the dilated mask; see _setup_likemask), with noise sigma per
        voxel: sigma is a free parameter if 'sigma' is in freepar, otherwise the measured noise
        rms (self.noiserms) or the value given in like_kwargs.
        Since the noise is correlated, the log-likelihood is multiplied by self.likefactor:
        by default 1/(correlation volume of the noise measured from the voxels without
        emission, spatial x spectral), so that each independent noise measurement counts
        once; with noisecorr='beam', 1/(beam area in pixels); with noisecorr=N, 1/N (see
        _set_likefactor).

        With 'fracerr' free (or given in like_kwargs), the variance of each voxel is
        sigma^2 + (fracerr * model)^2: a model error proportional to the model.

        Accepted kwargs (the options of like_kwargs used at each call): sigma (fixed noise per
        voxel, default self.noiserms); fracerr (fixed fractional model error, default 0);
        lambdas (dict of smoothness penalties, default self.lambdas, not multiplied by
        likefactor); any keyword accepted by user-defined residuals (_calculate_residuals).
        """
        rings = self._update_rings(self._work, theta)
        if self._negative(rings, theta):
            return -np.inf
        if self.useBBres:
            return -self._galfit.calculateResiduals(rings.object)

        if self.useNorm and self.update_prof:
            self._set_density(rings)
        model, bhi, blo = self._calculate_model(rings)[:3]
        mask = self._likemask[:, blo[1]:bhi[1], blo[0]:bhi[0]]
        data = self.data[:, blo[1]:bhi[1], blo[0]:bhi[0]]
        if 'sigma' in self.freepar_names:
            kwargs['sigma'] = theta[self.freepar_idx['sigma']][0]
        elif kwargs.get('sigma') is None:
            kwargs['sigma'] = self.noiserms
        if 'fracerr' in self.freepar_idx:
            kwargs['fracerr'] = theta[self.freepar_idx['fracerr']][0]
        kwargs.setdefault('maskmode', self._maskmode)
        kwargs.setdefault('ss_out', self._sumsq_outside(blo, bhi))
        kwargs.setdefault('region', self._region[blo[1]:bhi[1], blo[0]:bhi[0]])
        logL = self.likefactor * self._calculate_residuals(model, data, mask, **kwargs)

        # Smoothness penalty for ring-by-ring parameters
        lambdas = kwargs.get('lambdas', self.lambdas)
        if rings.nr > 1 and any(lambdas[key] > 0 for key in lambdas):
            # second derivatives on the radii of the rings (pyBBarolo used the first separation for all the
            # rings, and put the zero of VROT one separation inside the first ring, not at R = 0)
            radii = np.asarray(rings.r['radii'], float)
            for key in self.freepar_idx:
                if key in lambdas and lambdas[key] > 0 and len(self.freepar_idx[key]) > 1:
                    v, x = np.asarray(rings.r[key], float), radii
                    if key == 'vrot':                     # force the rotation curve to start from zero
                        v, x = np.concatenate([[0.0], v]), np.concatenate([[0.0], x])
                    d1, d2 = np.diff(x)[:-1], np.diff(x)[1:]
                    d2v = 2. * ((v[2:] - v[1:-1]) / d2 - (v[1:-1] - v[:-2]) / d1) / (d1 + d2)
                    logL += 0.5 * lambdas[key] * np.sum(d2v ** 2)
        return -logL

    def _log_likelihood_geometry(self, theta, **kwargs):
        """ Likelihood on the total intensity map (geometry fit, Cannubi-style).

        Gaussian likelihood of the total intensity map (data_mom0: the sum of the data in the
        voxels of the likelihood along each spectrum, see _setup_likemask), with the noise of
        the map in each pixel (map_rms: noiserms*sqrt(number of channels summed), larger with
        the measured spectral correlation of the noise, see _map_noise), computed in the
        fixed likelihood region and multiplied by self.likefactor2d (1/the measured spatial
        correlation area of the noise, 1/beam area in pixels with noisecorr='beam', 1/N with
        noisecorr=N), as in _log_likelihood.

        The map of a real galaxy has structure that an axisymmetric disk does not reproduce, often
        far above the noise (NGC 2403: about 20 times the noise in each pixel): the posterior is
        then extremely narrow (the errors are too small) and the sampling slow. As in
        _log_likelihood, a model error widens it: 'sigma' (the noise per voxel, free or given; the
        noise of the map is map_rms * sigma / noiserms) and 'fracerr' (free or given: the variance
        of a pixel is that of the noise plus (fracerr * model)^2), with the normalisation of the
        Gaussian (relative to map_rms, so that the values without them do not change). 'sigma'
        scales the noise of all the pixels: the best geometry is that of the likelihood without
        it, and the posterior gets a realistic width. 'fracerr' instead weights the pixels where the
        model is faint, the outer disk: on NGC 2403 it drove INC to zero (and fracerr to 1).
        """
        rings = self._update_rings(self._work, theta)
        if self._negative(rings, theta):
            return -np.inf
        if self.useNorm and self.update_prof:
            self._set_density(rings)
        model, bhi, blo = self._calculate_model(rings, fullcube=False)[:3]
        if model.ndim == 4:                 # (non-convolved, convolved) models
            model = model[-1]
        Lb, Hb = self._likebox
        map_data = self.data_mom0[Lb[1]:Hb[1], Lb[0]:Hb[0]]
        map_rms = self.map_rms[Lb[1]:Hb[1], Lb[0]:Hb[0]]
        map_mod = np.zeros_like(map_data, dtype=np.float64)
        map_mod[blo[1] - Lb[1]:bhi[1] - Lb[1], blo[0] - Lb[0]:bhi[0] - Lb[0]] = np.nansum(model, axis=0)
        ok = (map_rms > 0) & self._region[Lb[1]:Hb[1], Lb[0]:Hb[0]]
        sigma = theta[self.freepar_idx['sigma']][0] if 'sigma' in self.freepar_idx else kwargs.get('sigma')
        eps = theta[self.freepar_idx['fracerr']][0] if 'fracerr' in self.freepar_idx else kwargs.get('fracerr')
        if sigma is None and not eps:
            res = 0.5 * np.nansum(((map_data[ok] - map_mod[ok]) / map_rms[ok]) ** 2)
        else:
            if sigma is not None and not sigma > 0:
                return -np.inf
            s2 = map_rms[ok] ** 2
            v = s2 * (float(sigma) / self.noiserms) ** 2 if sigma is not None else s2
            if eps:
                v = v + (float(eps) * map_mod[ok]) ** 2
            res = 0.5 * np.nansum((map_data[ok] - map_mod[ok]) ** 2 / v + np.log(v / s2))
        return -self.likefactor2d * res

    def _flux_beyond_rings(self):
        """ The fraction of the flux of the total intensity map of the likelihood (data_mom0, in the
        region) beyond the outer edge of the initial rings plus half a beam, at their geometry (that
        of the last ring). The model cannot reproduce it, and the region of the likelihood is that
        of the lowest INC allowed by the prior: with a wide prior of INC, it biases INC low (NGC
        2403, rings to 720" with 8% of the flux beyond them: the INC of the map 30-45 instead of
        51 deg, and 0 with fracerr). """
        from ..galmod import outer_edge
        h = self._cube_obj.Head()
        pix = 0.5 * (abs(h.Cdelt(0)) + abs(h.Cdelt(1))) * 3600.
        r = self._inri.r
        edge = outer_edge(self._inri.object, self._galfit.model.ringmode) + 0.5 * abs(h.Bmaj()) * 3600.
        ny, nx = self.data_mom0.shape
        yy, xx = np.mgrid[:ny, :nx]
        dx, dy = (xx - float(r['xpos'][-1])) * pix, (yy - float(r['ypos'][-1])) * pix
        sp, cp = math.sin(math.radians(float(r['phi'][-1]))), math.cos(math.radians(float(r['phi'][-1])))
        cinc = max(math.cos(math.radians(float(r['inc'][-1]))), 1e-3)
        rr = np.hypot(-dx * sp + dy * cp, (-dx * cp - dy * sp) / cinc)
        m = self._region & np.isfinite(self.data_mom0)
        tot = float(np.sum(self.data_mom0[m]))
        return float(np.sum(self.data_mom0[m & (rr > edge)]) / tot) if tot > 0 else 0.

    def chi2_reduced(self, theta=None, fracerr=False):
        """ Reduced chi-square of the (Gaussian) likelihood for the parameters theta (default:
        the best-fit parameters): sum of the squared residuals / (sigma^2 * nvox), with sigma
        the fitted noise if 'sigma' is free, otherwise the one of like_kwargs or the measured
        noise rms. Values much
        larger than 1 mean that the model does not describe the data down to the noise,
        so that the credible intervals obtained with sigma fixed to the noise are too small:
        in this case fit 'sigma' or 'fracerr' as well (freepar=[..., 'fracerr']).
        fracerr=True: with the variance sigma^2 + (fracerr*model)^2 of each voxel, using the
        fitted fractional model error, or the one of like_kwargs (or a number: that value). """
        theta = self.params if theta is None else np.asarray(theta, float)
        rings = self._update_rings(self._fit0.copy(), theta)
        if self.useNorm and self.update_prof:
            self._set_density(rings)
        model, bhi, blo = self._calculate_model(rings)[:3]
        if model.ndim == 4:
            model = model[-1]
        mask = self._likemask[:, blo[1]:bhi[1], blo[0]:bhi[0]]
        data = self.data[:, blo[1]:bhi[1], blo[0]:bhi[0]]
        ss = _sumsq_residuals(np.asarray(model, np.float64), data, mask, self._maskmode,
                              self._region[blo[1]:bhi[1], blo[0]:bhi[0]]) + self._sumsq_outside(blo, bhi)
        lk = getattr(self, '_like_kwargs', {})
        sigma = theta[self.freepar_idx['sigma']][0] if 'sigma' in self.freepar_idx else \
            (lk.get('sigma') or self.noiserms)
        if fracerr is not False and fracerr is not None:
            eps = theta[self.freepar_idx['fracerr']][0] if fracerr is True and 'fracerr' in self.freepar_idx \
                else (float(lk.get('fracerr') or 0.) if fracerr is True else float(fracerr))
            if eps > 0:
                ss, _ = _residuals_fracerr(np.asarray(model, np.float64), data, mask, self._maskmode,
                                           self._region[blo[1]:bhi[1], blo[0]:bhi[0]], sigma ** 2, eps ** 2)
                ss += self._sumsq_outside(blo, bhi) / sigma ** 2
                return float(ss / max(self.nvox, 1))
        return float(ss / (sigma ** 2 * max(self.nvox, 1)))
