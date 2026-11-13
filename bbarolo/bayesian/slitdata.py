"""
The data of BAYESIANSLITFIT: a long-slit spectrum (a 2D image, position along the slit x spectral axis), read and
prepared for the fit.

    * reading (SlitData.read): the spectral axis is the one whose CTYPE is spectral (WAVE, AWAV, FREQ, VELO, VRAD,
      VOPT, FELO; or whose CUNIT is a wavelength, frequency or velocity), the other one is the position along the
      slit, whose scale (CDELT, CUNIT in arcsec, arcmin or deg) is required. Either order of the axes is accepted.
      A wavelength axis in metres is read in Angstrom (the unit of RESTWAVE and of the wavelengths of the keywords
      then); other units are kept. Blanks (NaN, or exactly 0) are not data;
    * the spectral window of the fit (WAVERANGE, or the lines over the velocity range of the emission, padded);
    * the continuum (CONTSUB): a polynomial of degree CONTORDER (default 1, a straight line) per position, fitted in
      the window away from the lines (and from the channels of EXCLWIND) with a 3-sigma clipping, and subtracted;
    * the residual sky (SKYRESID): the median of each column over the positions without line emission, subtracted;
    * the noise map, sigma^2(s, lambda) = sigma_col(lambda)^2 + sigma_obj(s)^2: the scatter of each column over the
      positions without line emission (the sky, larger at sky lines) plus that of each position in the continuum
      windows in excess of the sky (the Poisson noise of a bright galaxy);
    * the correlation of the noise along the slit and along the spectral axis (the number of pixels per independent
      measurement);
    * the initial estimate: a Gaussian line (all the lines, with RELINT, the same velocity) fitted at each position,
      which gives the velocity, the dispersion and the flux along the slit.

Spectral arrays are (nspec, npos): the spectral axis first, as the channels of the model cubes.
"""

import math
import numpy as np

from ..bbutils import BBaroloError, Wave2Vel, Vel2Wave, Freq2Vel, Vel2Freq

C_KMS = 299792.458
_WAVE_UNITS = {"m": 1E10, "um": 1E04, "mum": 1E04, "micron": 1E04, "nm": 10., "nanom": 10., "a": 1., "ang": 1.,
               "angstrom": 1.}
_FREQ_UNITS = {"hz": 1., "khz": 1E03, "mhz": 1E06, "ghz": 1E09}
_VEL_UNITS = {"m/s": 1E-3, "ms": 1E-3, "km/s": 1., "kms": 1., "km s-1": 1., "m s-1": 1E-3}
_SPEC_CTYPES = ("WAVE", "AWAV", "FREQ", "VELO", "VRAD", "VOPT", "FELO", "WAVELENGTH", "LAMBDA")

VSEARCH = 1000.         # km/s: the range around each line where the emission is searched for
VPAD = 600.             # km/s: the continuum on either side of the lines in the default window


def _spec_kind(ctype, cunit):
    """'wave', 'freq', 'velo' or None (not a spectral axis), from CTYPE and CUNIT."""
    ct, cu = str(ctype).upper().strip(), str(cunit).lower().strip()
    if cu in _WAVE_UNITS and cu != "m" or ct.startswith(("WAVE", "AWAV", "LAMBDA")):
        return "wave"
    if cu in _FREQ_UNITS or ct.startswith("FREQ"):
        return "freq"
    if cu in _VEL_UNITS or ct.startswith(("VELO", "VRAD", "VOPT", "FELO")):
        return "velo"
    return None


class SpecAxis:
    """The spectral axis of a slit: values (in its unit) and velocities of the channels for each line.

    kind 'wave' (unit a wavelength), 'freq' (unit a frequency) or 'velo' (km/s); crval, cdelt, crpix as in FITS
    (crpix 1-based); veldef 'radio', 'optical' or 'relativistic'; rest: the observed reference of each line
    (wavelength RESTWAVE*(1+z), frequency RESTFREQ/(1+z), or None for a velocity axis)."""

    def __init__(self, kind, ctype, cunit, crval, cdelt, crpix, n):
        self.kind, self.ctype, self.cunit = kind, ctype, cunit
        self.crval, self.cdelt, self.crpix, self.n = float(crval), float(cdelt), float(crpix), int(n)
        self.veldef = "relativistic" if kind == "wave" else ("radio" if kind == "freq" else "optical")
        self.rest = None

    def values(self, chan=None):
        """The spectral values of the channels chan (0-based; default all)."""
        k = np.arange(self.n) if chan is None else np.asarray(chan, float)
        return self.crval + (k + 1 - self.crpix) * self.cdelt

    def channel(self, value):
        """The (fractional, 0-based) channel of a spectral value."""
        return (np.asarray(value, float) - self.crval) / self.cdelt + self.crpix - 1

    def vel(self, value, line=0):
        """The velocity (km/s) of spectral values for the line (index)."""
        if self.kind == "velo":
            return np.asarray(value, float) * _VEL_UNITS.get(self.cunit.lower(), 1.)
        if self.kind == "wave":
            return Wave2Vel(value, self.rest[line], self.veldef)
        return Freq2Vel(np.asarray(value, float) * _FREQ_UNITS[self.cunit.lower()],
                        self.rest[line] * _FREQ_UNITS[self.cunit.lower()], self.veldef)

    def value(self, v, line=0):
        """The spectral value of the velocity v (km/s) of the line."""
        v = np.asarray(v, float)
        if self.kind == "velo":
            return v / _VEL_UNITS.get(self.cunit.lower(), 1.)
        if self.kind == "wave":
            return Vel2Wave(v, self.rest[line], self.veldef)
        return Vel2Freq(v, self.rest[line], self.veldef)

    def chan_of_vel(self, v, line=0):
        return self.channel(self.value(v, line))

    def dv(self, line=0):
        """The width of a channel (km/s) at the centre of the line."""
        if self.kind == "velo":
            return abs(self.cdelt) * _VEL_UNITS.get(self.cunit.lower(), 1.)
        c0 = self.chan_of_vel(0., line)
        return float(abs(self.vel(self.values(c0 + 0.5), line) - self.vel(self.values(c0 - 0.5), line)))

    def subset(self, k0, k1):
        """The axis of the channels k0 to k1 - 1."""
        a = SpecAxis(self.kind, self.ctype, self.cunit, self.crval, self.cdelt, self.crpix - k0, k1 - k0)
        a.veldef, a.rest = self.veldef, self.rest
        return a


class SlitData:
    """A long-slit spectrum: data (nspec, npos) float64 with the blanks as NaN, the spectral axis (SpecAxis), the
    scale of the positions (pix, arcsec) and the header of the file. transposed: the spectral axis was the second
    axis of the FITS image (NAXIS2), as written back by to_hdu."""

    def __init__(self, data, spec, pix, header=None, transposed=False, name="NONE", posaxis=None):
        self.data = np.asarray(data, np.float64)
        self.spec = spec
        self.pix = float(pix)
        self.header = header
        self.transposed = transposed
        self.name = name
        self.posaxis = posaxis or {}           # (CTYPE, CRPIX, CRVAL, CDELT, CUNIT of the position axis, as read)
        self.nspec, self.npos = self.data.shape

    @classmethod
    def read(cls, src):
        """The slit from a FITS file name (file.fits[ext] for an extension), an astropy HDU or (data, header)."""
        from astropy.io import fits
        if isinstance(src, str):
            from ..header import read_fits_image
            hdr, data = read_fits_image(src)
        elif isinstance(src, (fits.PrimaryHDU, fits.ImageHDU)):
            hdr, data = src.header, src.data
        elif isinstance(src, tuple) and len(src) == 2:
            data, hdr = src
            hdr = fits.Header(hdr)
        else:
            raise TypeError("BAYESIANSLITFIT: the slit must be a FITS file name, an astropy HDU or (data, header)")
        data = np.squeeze(np.asarray(data, np.float64))
        if data.ndim != 2:
            raise BBaroloError(f"BAYESIANSLITFIT: the slit must be a 2D image (position x spectral axis), not an "
                               f"array of shape {np.shape(data)}")
        hdr = fits.Header(hdr)
        naxes = [i for i in range(1, int(hdr.get("NAXIS", 2)) + 1) if int(hdr.get(f"NAXIS{i}", 1)) > 1]
        if len(naxes) != 2:
            naxes = [1, 2]
        kinds = [_spec_kind(hdr.get(f"CTYPE{i}", ""), hdr.get(f"CUNIT{i}", "")) for i in naxes]
        if sum(k is not None for k in kinds) != 1:
            raise BBaroloError("BAYESIANSLITFIT: cannot tell the spectral axis of the slit from CTYPE1/2 and "
                               f"CUNIT1/2 ({[hdr.get(f'CTYPE{i}') for i in naxes]}, "
                               f"{[hdr.get(f'CUNIT{i}') for i in naxes]}): one must be WAVE, AWAV, FREQ or VELO")
        isp = 0 if kinds[0] is not None else 1
        sax, pax = naxes[isp], naxes[1 - isp]
        kind = kinds[isp]
        # the position axis: its scale in arcsec
        pu = str(hdr.get(f"CUNIT{pax}", "")).strip().lower()
        pdelt = hdr.get(f"CDELT{pax}", hdr.get(f"CD{pax}_{pax}"))
        if pdelt is None or not pu or not any(u in pu for u in ("deg", "arcm", "arcs")):
            raise BBaroloError(f"BAYESIANSLITFIT: the position axis of the slit (axis {pax}) has no scale: give "
                               f"CDELT{pax} with CUNIT{pax} in arcsec, arcmin or deg")
        from ..bbutils import arcsconv
        pix = abs(float(pdelt)) * arcsconv(pu)
        # the spectral axis
        cunit = str(hdr.get(f"CUNIT{sax}", "")).strip()
        ctype = str(hdr.get(f"CTYPE{sax}", "")).strip()
        crval = float(hdr.get(f"CRVAL{sax}", 0.))
        cdelt = float(hdr.get(f"CDELT{sax}", hdr.get(f"CD{sax}_{sax}", 1.)))
        crpix = float(hdr.get(f"CRPIX{sax}", 1.))
        if kind == "wave":
            cu = cunit.lower()
            if cu == "" or cu not in _WAVE_UNITS:
                raise BBaroloError(f"BAYESIANSLITFIT: unknown unit of the wavelength axis CUNIT{sax}={cunit!r} "
                                   "(m, um, nm or Angstrom)")
            if cu == "m":                       # (metres: read in Angstrom)
                crval, cdelt, cunit = crval * 1E10, cdelt * 1E10, "Angstrom"
        elif kind == "freq" and cunit.lower() not in _FREQ_UNITS:
            raise BBaroloError(f"BAYESIANSLITFIT: unknown unit of the frequency axis CUNIT{sax}={cunit!r}")
        elif kind == "velo" and cunit.lower() not in _VEL_UNITS:
            raise BBaroloError(f"BAYESIANSLITFIT: unknown unit of the velocity axis CUNIT{sax}={cunit!r}")
        # data (nspec, npos): numpy's axes are reversed with respect to FITS
        arr = data if sax == naxes[1] else data.T
        arr = np.array(arr, np.float64)
        arr[~np.isfinite(arr) | (arr == 0)] = np.nan
        spec = SpecAxis(kind, ctype, cunit, crval, cdelt, crpix, arr.shape[0])
        from ..header import object_name
        posaxis = {k: hdr.get(f"{k}{pax}") for k in ("CTYPE", "CRPIX", "CRVAL", "CDELT", "CUNIT")}
        return cls(arr, spec, pix, header=hdr, transposed=(sax == naxes[1]),
                   name=object_name(hdr.get("OBJECT", "NONE")) or "NONE", posaxis=posaxis)

    def to_hdu(self, arr, k0=0, bunit=None, extra=None):
        """A PrimaryHDU of an array (nspec_window, npos) of this slit, from channel k0, in the orientation and with the
        axes of the file (the spectral axis in its unit, Angstrom for metres)."""
        from astropy.io import fits
        arr = np.asarray(arr, np.float32)
        data = arr if self.transposed else arr.T
        h = fits.Header()
        sp, ps = (2, 1) if self.transposed else (1, 2)
        h[f"CTYPE{sp}"], h[f"CUNIT{sp}"] = self.spec.ctype, self.spec.cunit
        h[f"CRPIX{sp}"], h[f"CRVAL{sp}"], h[f"CDELT{sp}"] = self.spec.crpix - k0, self.spec.crval, self.spec.cdelt
        pa = self.posaxis
        h[f"CTYPE{ps}"] = pa.get("CTYPE") or "POSITION"
        h[f"CUNIT{ps}"] = pa.get("CUNIT") or "arcsec"
        h[f"CRPIX{ps}"] = pa.get("CRPIX") if pa.get("CRPIX") is not None else 1.
        h[f"CRVAL{ps}"] = pa.get("CRVAL") if pa.get("CRVAL") is not None else 0.
        h[f"CDELT{ps}"] = pa.get("CDELT") if pa.get("CDELT") is not None else self.pix
        h["OBJECT"] = self.name
        if bunit is None and self.header is not None and "BUNIT" in self.header:
            bunit = self.header["BUNIT"]
        if bunit:
            h["BUNIT"] = bunit
        for k, v in (extra or {}).items():
            h[k] = v
        return fits.PrimaryHDU(data=data, header=h)


# ---------------------------------------------------------------------------------------------- preparation
def _polyfit(y, x, w, order):
    """Weighted polynomial of degree order of y (n, npos) against x (n,) with weights w (n, npos), for each position:
    its values (n, npos) (0 where fewer than order + 2 points)."""
    V = np.vander(x, order + 1, increasing=True)                     # (n, order + 1)
    yw = np.where(w > 0, y, 0.)
    G = np.einsum("ip,ij,ik->pjk", w, V, V)                         # (npos, order + 1, order + 1)
    b = np.einsum("ip,ij,ip->pj", w, V, yw)                         # (npos, order + 1)
    ok = w.sum(0) >= order + 2
    c = np.zeros_like(b)
    if np.any(ok):
        c[ok] = (np.linalg.pinv(G[ok]) @ b[ok][:, :, None])[:, :, 0]
    return V @ c.T


def fit_continuum(data, x, use, order=1, nclip=3):
    """The continuum of each position: a polynomial of degree order in x fitted to data (nspec, npos) in the channels
    use (nspec,) bool, with 3-sigma clipping (nclip iterations). Returns the continuum (nspec, npos)."""
    good = np.isfinite(data) & use[:, None]
    w = good.astype(np.float64)
    xc = x - np.mean(x[use]) if np.any(use) else x - np.mean(x)
    h = float(np.max(np.abs(xc[use]))) if np.any(use) else 1.
    xc = xc / (h if h > 0 else 1.)                                  # (in [-1, 1] in the channels of the fit)
    for _ in range(nclip + 1):
        cont = _polyfit(data, xc, w, int(order))
        r = np.where(good, data - cont, np.nan)
        with np.errstate(all="ignore"):
            med = np.nanmedian(r, axis=0)
            s = 1.4826 * np.nanmedian(np.abs(r - med), axis=0)
        neww = (good & (np.abs(r - med) <= 3 * np.where(s > 0, s, np.inf))).astype(np.float64)
        if np.array_equal(neww, w):
            break
        w = neww
    return cont


def _mad(a, axis=None):
    with np.errstate(all="ignore"):
        med = np.nanmedian(a, axis=axis, keepdims=True)
        return 1.4826 * np.squeeze(np.nanmedian(np.abs(a - med), axis=axis, keepdims=True), axis=axis)


def _ranges_mask(values, ranges):
    """The channels whose values fall in any of the ranges (pairs (lo, hi), either order)."""
    m = np.zeros(values.shape, bool)
    for lo, hi in ranges:
        lo, hi = min(lo, hi), max(lo, hi)
        m |= (values >= lo) & (values <= hi)
    return m


def _pairs(vec, name):
    v = [float(x) for x in (vec or [])]
    if len(v) % 2:
        raise BBaroloError(f"BAYESIANSLITFIT: {name} needs pairs of values (ranges lo hi), not {len(v)} values")
    return [(v[i], v[i + 1]) for i in range(0, len(v), 2)]


def line_stack(data, spec, relint, vgrid):
    """The lines stacked in velocity: for each position, sum_l relint_l data(lambda_l(v)) / sum_l relint_l^2 on the
    velocity grid vgrid (linear interpolation of each spectrum; NaN where a line falls outside the data or on a
    blank). Returns (nv, npos)."""
    relint = np.asarray(relint, float)
    out = np.zeros((vgrid.size, data.shape[1]))
    norm = 0.
    k = np.arange(data.shape[0])
    for l, a in enumerate(relint):
        if a == 0:
            continue
        c = spec.chan_of_vel(vgrid, l)
        i0 = np.floor(c).astype(int)
        f = c - i0
        ok = (i0 >= 0) & (i0 + 1 < k.size)
        i0c = np.clip(i0, 0, k.size - 2)
        s = data[i0c] * (1 - f)[:, None] + data[i0c + 1] * f[:, None]
        s[~ok] = np.nan
        out += a * s
        norm += a * a
    return out / norm


def _gauss(v, a, v0, s):
    return a * np.exp(-0.5 * ((v - v0) / s) ** 2)


def fit_lines(stack, vgrid, noise, sig_instr, snr=4.):
    """A Gaussian fitted at each position of the stacked lines (nv, npos), where its peak, smoothed to the instrumental
    resolution, is above snr times the noise of the stack (noise: per position). Returns a dict of arrays (npos,):
    ok, v (km/s), sigma (observed dispersion, km/s), flux (amplitude x sigma x sqrt(2 pi), in data units x km/s),
    snr (the peak signal-to-noise)."""
    from scipy.optimize import curve_fit
    from scipy.ndimage import gaussian_filter1d
    nv, npos = stack.shape
    dv = abs(vgrid[1] - vgrid[0])
    res = {k: np.full(npos, np.nan) for k in ("v", "sigma", "flux", "snr")}
    res["ok"] = np.zeros(npos, bool)
    st = np.where(np.isfinite(stack), stack, 0.)
    sm = gaussian_filter1d(st, max(sig_instr / dv, 0.5), axis=0, mode="constant")
    # the noise of the smoothed stack, per position
    nsm = noise * math.sqrt(1. / (2. * math.sqrt(math.pi) * max(sig_instr / dv, 0.5)))
    for j in range(npos):
        y = stack[:, j]
        good = np.isfinite(y)
        if good.sum() < 8 or not nsm[j] > 0:
            continue
        k = int(np.argmax(np.where(good, sm[:, j], -np.inf)))
        res["snr"][j] = sm[k, j] / nsm[j]
        if res["snr"][j] < snr:
            continue
        p0 = (max(y[k] if np.isfinite(y[k]) else sm[k, j], 1e-30), vgrid[k], max(2 * sig_instr, 2 * dv))
        try:
            p, _ = curve_fit(_gauss, vgrid[good], y[good], p0=p0, maxfev=2000,
                             bounds=([0., vgrid[0], 0.5 * dv], [np.inf, vgrid[-1], 0.5 * (vgrid[-1] - vgrid[0])]))
        except Exception:
            continue
        a, v0, s = p
        if not (a > 0 and np.isfinite(v0) and s < 0.4 * (vgrid[-1] - vgrid[0])):
            continue
        res["ok"][j] = True
        res["v"][j], res["sigma"][j], res["flux"][j] = v0, s, a * s * math.sqrt(2 * math.pi)
    return res


def _smooth_outliers(v, ok, tol, half=5):
    """The positions whose value v is within tol of the median of the detections within half positions (itself
    excluded), iterated until nothing changes: a bool array (False for the outliers and where not ok)."""
    keep = ok.copy()
    n = v.size
    for _ in range(5):
        new = keep.copy()
        for j in np.flatnonzero(keep):
            nb = [i for i in range(max(j - half, 0), min(j + half + 1, n)) if i != j and keep[i]]
            if len(nb) >= 2 and abs(v[j] - np.median(v[nb])) > tol:
                new[j] = False
        if np.array_equal(new, keep):
            break
        keep = new
    return keep


class PreparedSlit:
    """The slit prepared for the fit (see prepare): attributes

      slit        the SlitData
      k0, k1      the channels of the window, [k0, k1)
      spec        the SpecAxis of the window
      data        (nz, npos) the data of the window, continuum and residual sky subtracted (NaN: blanks)
      continuum   (nz, npos) the continuum subtracted (0 without CONTSUB)
      sky         (nz,) the residual sky subtracted (0 without SKYRESID)
      noise       (nz, npos) the noise map
      valid       (nz, npos) bool: the pixels of the data (not blank, not in MASKWAVE)
      linechan    (nz,) bool: the channels of the lines (the velocity range of the emission)
      contchan    (nz,) bool: the channels of the continuum (the window away from the lines, EXCLWIND and MASKWAVE)
      freepos     (npos,) bool: the positions without line emission
      lines       dict of the fits of the lines along the slit (see fit_lines), with pos (pixels)
      vrange      (vlo, vhi): the velocity range of the emission (km/s)
      corr        dict: spatial and spectral correlation lengths (pixels, channels), likefactor
    """


def prepare(slit, restwave, relint, redshift, veldef, sig_instr, beam_fwhm, contsub=False, contorder=1,
            exclwind=None, waverange=None, skyresid=False, maskwave=None, verbose=True):
    """Prepare the slit for the fit (see the module documentation). restwave: the rest wavelengths (or frequencies,
    for a frequency axis; ignored for a velocity axis) in the units of the axis; relint: their relative intensities;
    sig_instr: the instrumental dispersion (km/s); beam_fwhm: the seeing (arcsec); contsub, contorder: the continuum
    subtracted, a polynomial of degree contorder; exclwind: the channels that are not continuum (EXCLWIND, channels
    of the slit counted from 0); waverange, maskwave: lists of values of the spectral axis (pairs). Returns a
    PreparedSlit."""
    from ..cube import exclwind_channels
    spec = slit.spec
    sp = SpecAxis(spec.kind, spec.ctype, spec.cunit, spec.crval, spec.cdelt, spec.crpix, spec.n)
    if veldef and veldef != "auto":
        sp.veldef = veldef
    if sp.kind == "velo":
        rest = [None]
        relint = [1.]
    else:
        rw = [float(x) for x in (restwave or []) if float(x) > 0]
        if not rw:
            raise BBaroloError(f"BAYESIANSLITFIT: RESTWAVE{' (RESTFREQ)' if sp.kind == 'freq' else ''} is required "
                               f"for a {'wavelength' if sp.kind == 'wave' else 'frequency'} axis")
        rest = [w * (1 + redshift) for w in rw] if sp.kind == "wave" else [f / (1 + redshift) for f in rw]
        relint = list(relint) if relint is not None and len(relint) == len(rw) else [1.] * len(rw)
    sp.rest = rest
    out = PreparedSlit()
    out.slit = slit
    nsp, npos = slit.data.shape
    allv = sp.values()
    data = slit.data
    blank = ~np.isfinite(data)
    maskr = _pairs(maskwave, "MASKWAVE")
    masked = _ranges_mask(allv, maskr) if maskr else np.zeros(nsp, bool)
    notcont = masked | exclwind_channels("NONE" if exclwind is None else exclwind, nsp,
                                         "the spectral axis of the slit")
    # the channels near the lines (the search range)
    near = np.zeros(nsp, bool)
    for l in range(len(rest)):
        c = np.sort(sp.chan_of_vel(np.array([-VSEARCH, VSEARCH]), l))
        near[max(int(np.floor(c[0])), 0):max(int(np.ceil(c[1])) + 1, 0)] = True
    if not np.any(near):
        raise BBaroloError("BAYESIANSLITFIT: the lines (RESTWAVE at REDSHIFT) fall outside the spectral axis of the "
                           f"slit ({allv[0]:.6g} to {allv[-1]:.6g} {sp.cunit})")
    if waverange:
        wr = _pairs(waverange, "WAVERANGE")
        if len(wr) != 1:
            raise BBaroloError("BAYESIANSLITFIT: WAVERANGE is one range (two values)")
        inwin = _ranges_mask(allv, wr)
        if inwin.sum() < 10:
            raise BBaroloError("BAYESIANSLITFIT: WAVERANGE holds fewer than 10 channels of the slit")
    else:
        inwin = None
    dvch = sp.dv(0)
    pad = int(math.ceil(VPAD / dvch))
    # first pass: the continuum away from the lines (+-VSEARCH/2), in a window around the search range
    k = np.flatnonzero(near)
    lo1, hi1 = max(k[0] - pad, 0), min(k[-1] + pad + 1, nsp)
    sl1 = slice(lo1, hi1)
    # (the channels away from the lines: +-600 km/s, less if that leaves fewer than 20 channels on a short spectrum)
    for vex in (0.6 * VSEARCH, 0.4 * VSEARCH, 0.3 * VSEARCH):
        vlines = np.zeros(nsp, bool)
        for l in range(len(rest)):
            c = np.sort(sp.chan_of_vel(np.array([-vex, vex]), l))
            vlines[max(int(np.floor(c[0])), 0):max(int(np.ceil(c[1])) + 1, 0)] = True
        if np.sum(~vlines[sl1] & ~notcont[sl1]) >= 20:
            break
    x = allv
    cont = np.zeros_like(data)
    if contsub:
        use = ~vlines[sl1] & ~notcont[sl1]
        if use.sum() < max(10, 3 * (contorder + 1)):
            raise BBaroloError("BAYESIANSLITFIT: too few channels to fit the continuum (CONTSUB): the spectrum is "
                               "too short around the lines, or EXCLWIND and MASKWAVE leave out too many channels")
        cont[sl1] = fit_continuum(data[sl1], x[sl1], use, contorder)
    work = np.where(masked[:, None], np.nan, data - cont)
    # the stacked lines and their fits along the slit
    vgrid = np.arange(-VSEARCH, VSEARCH + 0.5 * dvch, dvch)
    stack = line_stack(work, sp, relint, vgrid)
    edge = np.abs(vgrid) > vex
    snoise = _mad(np.where(edge[:, None], stack, np.nan), axis=0)
    if not np.all(np.isfinite(snoise)):
        snoise = np.where(np.isfinite(snoise), snoise, np.nanmedian(snoise))
    fits_ = fit_lines(stack, vgrid, snoise, sig_instr, snr=4.)
    ok = fits_["ok"]
    if ok.sum() < 3:
        raise BBaroloError("BAYESIANSLITFIT: no line emission detected along the slit (check RESTWAVE, REDSHIFT and "
                           "the spectral axis; CONTSUB true if the slit has a continuum)")
    # isolated detections are spurious: keep the positions with a detected neighbour; so are the velocities far
    # from those of the nearby positions (e.g. a line taken for another one of the stack)
    ok &= np.r_[False, ok[:-1]] | np.r_[ok[1:], False]
    ok &= _smooth_outliers(fits_["v"], ok, max(3. * sig_instr, 100.))
    ok &= fits_["sigma"] < 300.
    fits_["ok"] = ok
    fits_["pos"] = np.arange(npos, dtype=float)
    vlo = float(np.nanmin(fits_["v"][ok] - 2.5 * fits_["sigma"][ok]))
    vhi = float(np.nanmax(fits_["v"][ok] + 2.5 * fits_["sigma"][ok]))
    out.vrange = (vlo, vhi)
    linechan = np.zeros(nsp, bool)
    for l in range(len(rest)):
        c = np.sort(sp.chan_of_vel(np.array([vlo, vhi]), l))
        linechan[max(int(np.floor(c[0])), 0):max(int(np.ceil(c[1])) + 1, 0)] = True
    # the window of the fit
    if inwin is None:
        k = np.flatnonzero(linechan)
        inwin = np.zeros(nsp, bool)
        inwin[max(k[0] - pad, 0):min(k[-1] + pad + 1, nsp)] = True
    kw = np.flatnonzero(inwin)
    k0, k1 = int(kw[0]), int(kw[-1]) + 1
    sl = slice(k0, k1)
    # second pass of the continuum: the window away from the lines
    contchan = np.zeros(nsp, bool)
    contchan[sl] = ~linechan[sl]
    contchan &= ~notcont
    if contsub:
        if contchan.sum() < max(6, 3 * (contorder + 1)):
            raise BBaroloError("BAYESIANSLITFIT: too few channels to fit the continuum (CONTSUB) in the window of the "
                               "fit: widen WAVERANGE, or exclude fewer channels (EXCLWIND, MASKWAVE)")
        cont[:] = 0.
        cont[sl] = fit_continuum(data[sl], x[sl], contchan[sl], contorder)
    # the positions without line emission: not detected, and more than one seeing FWHM from a detection
    bpix = beam_fwhm / slit.pix
    det = np.flatnonzero(ok)
    dist = np.min(np.abs(np.arange(npos)[:, None] - det[None, :]), axis=1)
    freepos = ~ok & (dist > max(bpix, 2.)) & (np.isfinite(fits_["snr"]) & (np.abs(fits_["snr"]) < 3.))
    # residual sky
    w = data[sl] - cont[sl]
    sky = np.zeros(k1 - k0)
    if skyresid:
        if freepos.sum() < 5:
            raise BBaroloError(f"BAYESIANSLITFIT: SKYRESID needs positions without line emission along the slit "
                               f"(found {int(freepos.sum())})")
        with np.errstate(all="ignore"):
            sky = np.nanmedian(w[:, freepos], axis=1)
        sky = np.where(np.isfinite(sky), sky, 0.)
        w = w - sky[:, None]
    valid = np.isfinite(w) & ~masked[sl, None]
    # the noise map
    out.noise = noise_map(w, valid, contchan[sl], freepos)
    valid &= np.isfinite(out.noise) & (out.noise > 0)
    out.k0, out.k1 = k0, k1
    out.spec = sp.subset(k0, k1)
    out.data = np.where(valid, w, np.nan)
    out.continuum = cont[sl]
    out.sky = sky
    out.valid = valid
    out.linechan = linechan[sl]
    out.contchan = contchan[sl]
    out.freepos = freepos
    out.lines = fits_
    out.relint = np.asarray(relint, float)
    out.corr = noise_corr(out.data / out.noise, valid & out.contchan[:, None] | valid & freepos[None, :])
    return out


def _clipped_std(x, nsig=4.):
    """A robust standard deviation of the finite values of x: the standard deviation within nsig times the MAD scatter
    of the median (iterated twice; more efficient than the MAD itself), NaN for fewer than 3 values."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 3:
        return np.nan
    for _ in range(2):
        med = np.median(x)
        s = 1.4826 * np.median(np.abs(x - med))
        if not s > 0:
            return float(np.std(x))
        x = x[np.abs(x - med) <= nsig * s]
    return float(np.std(x, ddof=1)) if x.size > 2 else np.nan


NOISE_MIN_SAMPLES = 100     # (the values of the noise of a channel: neighbouring channels are pooled to reach them)


def noise_map(w, valid, contchan, freepos):
    """The noise of each pixel of the window w (nz, npos): sigma_col(lambda)^2 + sigma_obj(s)^2 (see the module
    documentation); with fewer than 10 positions without emission, the noise of each position in the continuum
    channels only (sigma_obj). The noise of a channel pools the neighbouring channels to reach NOISE_MIN_SAMPLES
    values (a robust standard deviation, see _clipped_std): with few empty positions, that of a single channel is
    noisy (15 positions: 30% scatter), which biases the likelihood."""
    nz, npos = w.shape
    wv = np.where(valid, w, np.nan)
    srow = None
    cc = contchan & np.any(valid, axis=1)
    if cc.sum() >= 5:
        srow = np.array([_clipped_std(wv[cc, j]) for j in range(npos)])  # (npos,): each position, continuum channels
    nfree = int(freepos.sum())
    if nfree >= 10:
        h = max(0, int(math.ceil((NOISE_MIN_SAMPLES / nfree - 1) / 2.)))
        wf = wv[:, freepos]
        scol = np.array([_clipped_std(wf[max(k - h, 0):k + h + 1]) for k in range(nz)])   # (nz,)
        scol = np.where(np.isfinite(scol) & (scol > 0), scol, np.nanmedian(scol))
        if srow is not None:
            sky = float(np.nanmedian(scol[cc]))
            obj2 = np.maximum(np.where(np.isfinite(srow), srow, sky) ** 2 - sky ** 2, 0.)
            return np.sqrt(scol[:, None] ** 2 + obj2[None, :])
        return np.repeat(scol[:, None], npos, axis=1)
    if srow is None:
        s = _mad(wv)
        return np.full((nz, npos), s)
    srow = np.where(np.isfinite(srow) & (srow > 0), srow, np.nanmedian(srow))
    return np.repeat(srow[None, :], nz, axis=0)


def noise_corr(z, use, maxlag=8):
    """The correlation of the noise of the normalised residuals z (nz, npos) in the pixels use, along the spectral axis
    and along the slit: the sums of the autocorrelation over the lags (out to the first lag where it is <= 0, at most
    maxlag), which are the numbers of channels and of positions per independent measurement (at least 1).
    Returns dict(spectral, spatial, likefactor = 1 / (spectral * spatial), rho_spectral, rho_spatial)."""
    zz = np.where(use & np.isfinite(z), z, np.nan)
    # (the median of each channel and of each position out: the systematic residuals of the sky and of the continuum,
    # constant along the slit or along a spectrum, are not the correlation of the noise)
    with np.errstate(all="ignore"):
        for ax in (1, 0, 1):
            med = np.nanmedian(zz, axis=ax, keepdims=True)
            zz = zz - np.where(np.isfinite(med), med, 0.)

    def acf(a, axis):
        rho = [1.]
        a0 = np.nan_to_num(a)
        m = np.isfinite(a).astype(float)
        v = np.sum(a0 * a0) / max(np.sum(m), 1.)
        for L in range(1, maxlag + 1):
            if axis == 0:
                p, n = np.sum(a0[L:] * a0[:-L]), np.sum(m[L:] * m[:-L])
            else:
                p, n = np.sum(a0[:, L:] * a0[:, :-L]), np.sum(m[:, L:] * m[:, :-L])
            if n < 50 or v <= 0:
                break
            r = p / n / v
            if r <= 0:
                break
            rho.append(r)
        return np.array(rho)
    rs, rp = acf(zz, 0), acf(zz, 1)
    A_s = max(1., 1. + 2. * float(np.sum(rs[1:])))
    A_p = max(1., 1. + 2. * float(np.sum(rp[1:])))
    return dict(spectral=A_s, spatial=A_p, likefactor=1. / (A_s * A_p), rho_spectral=rs, rho_spatial=rp)
