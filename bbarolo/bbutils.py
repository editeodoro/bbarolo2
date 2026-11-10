"""
General utilities (port of Utilities/utils.cpp, string.cpp, interpolation.cpp,
statistics.cpp helpers and progressbar.cpp).

Spectral conversions, flux conversions, coordinate parsing, ring reading,
random adjectives & quotes and a few numerical helpers.  Most functions accept
numpy arrays as well as scalars.
"""

import os
import sys
import time
import random
import numpy as np

class BBaroloError(ValueError):
    """An error in the input of BBarolo (data, parameters) that stops a task, with a message that
    says what is wrong. The command line prints it and exits with code 1."""


C_KMS = 299792.458          # speed of light in km/s
C_MS = 2.99792458e08        # speed of light in m/s


# --------------------------------------------------------------------------
# Misc
# --------------------------------------------------------------------------
def mkdirp(path):
    if path:
        os.makedirs(path, exist_ok=True)
    return True


def shortpath(path):
    """A path for the messages on screen: relative to the current folder, unless that goes up more than
    two folders (then absolute). The outputs (e.g. the plotting scripts) keep absolute paths, which work
    from any folder."""
    if not path:
        return path
    s = str(path)
    try:
        rel = os.path.relpath(s)
    except ValueError:                          # (another drive, on Windows)
        return s
    if rel.split(os.sep)[:3] == [os.pardir] * 3:
        return s
    if s.endswith(os.sep) and not rel.endswith(os.sep):
        rel += os.sep
    return rel


def fexists(filename):
    return os.path.isfile(filename)


# --------------------------------------------------------------------------
# Spectral conversions
# --------------------------------------------------------------------------
def Freq2Vel(v, freq0, veldef):
    """Frequency -> velocity in km/s."""
    v = np.asarray(v, dtype=float) if not np.isscalar(v) else float(v)
    if veldef == "radio":
        return C_KMS * (freq0 - v) / freq0
    if veldef == "optical":
        return C_KMS * (freq0 - v) / v
    return C_KMS * (freq0 * freq0 - v * v) / (freq0 * freq0 + v * v)


def Wave2Vel(v, wave0, veldef):
    """Wavelength -> velocity in km/s."""
    v = np.asarray(v, dtype=float) if not np.isscalar(v) else float(v)
    if veldef == "radio":
        return C_KMS * (v - wave0) / v
    if veldef == "optical":
        return C_KMS * (v - wave0) / wave0
    return C_KMS * (v * v - wave0 * wave0) / (v * v + wave0 * wave0)


def Vel2Freq(v, freq0, veldef):
    """Velocity in km/s -> frequency in the same units as freq0."""
    if veldef == "radio":
        return freq0 * (1 - v / C_KMS)
    if veldef == "optical":
        return freq0 / (1 + v / C_KMS)
    return freq0 * np.sqrt((1 - v / C_KMS) / (1 + v / C_KMS))


def Vel2Wave(v, wave0, veldef):
    return C_KMS / Vel2Freq(v, C_KMS / wave0, veldef)


def _spec_unit(h):
    cunit2 = h.Cunit(h.NumAx() - 1).lower()
    if h.NumAx() > 3:
        cunit2 = h.Cunit(2).lower()
    return cunit2


def _wave_rest(h, zcent_default):
    z_cent = zcent_default
    restw, reds = h.Wave0(), h.Redshift()
    if restw != -1:
        z_cent = (restw * (1 + reds) - h.Crval(2)) / h.Cdelt(2) + h.Crpix(2) - 1
    return (z_cent + 1 - h.Crpix(2)) * h.Cdelt(2) + h.Crval(2)


_WAVE_UNITS = ("mum", "um", "micron", "nm", "nanom", "a", "ang", "angstrom")
# frequency units of the spectral axis -> Hz (rest frequencies are always in Hz)
_FREQ_UNITS = {"hz": 1., "khz": 1E03, "mhz": 1E06, "ghz": 1E09}


def AlltoVel(inval, h):
    """Convert a spectral value (in units of the spectral axis) to km/s."""
    cunit2 = _spec_unit(h)
    if cunit2 in ("km/s", "kms"):
        return inval
    if cunit2 in ("m/s", "ms"):
        return inval / 1000.
    if cunit2 in _FREQ_UNITS:
        frest = h.Freq0() / (1 + h.Redshift())
        return Freq2Vel(inval * _FREQ_UNITS[cunit2], frest, h.VelDef())
    if cunit2 in _WAVE_UNITS:
        wrest = _wave_rest(h, h.Crpix(2) - 1)
        return Wave2Vel(inval, wrest, h.VelDef())
    return inval


def Vel2Spec(inval, h):
    """Convert a velocity in km/s to the units of the spectral axis."""
    cunit2 = _spec_unit(h)
    if cunit2 in ("km/s", "kms"):
        return inval
    if cunit2 in ("m/s", "ms"):
        return inval * 1000.
    if cunit2 in _FREQ_UNITS:
        frest = h.Freq0() / (1 + h.Redshift())
        return Vel2Freq(inval, frest, h.VelDef()) / _FREQ_UNITS[cunit2]
    if cunit2 in _WAVE_UNITS:
        wrest = _wave_rest(h, h.Crpix(2) - 1)
        return Vel2Wave(inval, wrest, h.VelDef())
    return inval


def DeltaVel(h):
    """Channel width in km/s."""
    deltaV = h.Cdelt(2)
    zdim = h.DimAx(2) if h.NumAx() > 2 else 1
    cunit2 = h.Cunit(2).lower() if h.NumAx() > 2 else ""
    if cunit2 in ("km/s", "kms"):
        return deltaV
    if cunit2 in ("m/s", "ms"):
        return deltaV / 1000.
    i = np.arange(zdim - 1)
    ipix = (i + 1 - h.Crpix(2)) * h.Cdelt(2) + h.Crval(2)
    spix = (i + 2 - h.Crpix(2)) * h.Cdelt(2) + h.Crval(2)
    if cunit2 in _FREQ_UNITS:
        fc = _FREQ_UNITS[cunit2]
        frest = h.Freq0() / (1 + h.Redshift())
        if h.VelDef() == "radio":
            return -C_KMS * deltaV * fc / frest
        return float(np.sum(Freq2Vel(spix * fc, frest, h.VelDef()) - Freq2Vel(ipix * fc, frest, h.VelDef()))
                     / (zdim - 1))
    if cunit2 in _WAVE_UNITS:
        # (radio velocities of a wavelength axis are not linear in wavelength: averaged as the others)
        wrest = _wave_rest(h, h.DimAx(2) / 2. - 1)
        return float(np.sum(Wave2Vel(spix, wrest, h.VelDef()) - Wave2Vel(ipix, wrest, h.VelDef())) / (zdim - 1))
    return deltaV


# --------------------------------------------------------------------------
# Flux conversions
# --------------------------------------------------------------------------
def isFluxUnitKnown(h):
    b = h.Bunit().lower().replace(" ", "")
    return ("w.u." in b or "wu" in b) or ("jy" in b) or b in ("k", "k(tmb)")


def _jy_prefix(b):
    """Factor of a prefixed Jansky unit (mJy, uJy) to Jy (C++ took mJy as Jy)."""
    if b.startswith("mjy"):
        return 1E-3
    if b.startswith(("ujy", "µjy", "μjy", "mujy")):
        return 1E-6
    return 1.


def FluxtoJyBeam(inval, h):
    b = h.Bunit().lower().replace(" ", "")
    if "w.u." in b or "wu" in b:
        return inval * 5E-3
    if "jy/b" in b or "j/b" in b:
        return inval * _jy_prefix(b)
    if "jy" in b and h.BeamArea() != 0:
        return inval * _jy_prefix(b) * h.BeamArea()
    if b in ("k", "k(tmb)"):
        return inval * (h.Bmaj() * 3600. * h.Bmin() * 3600.) / (1360. * 21.106 * 21.106)
    return inval


def FluxtoJy(inval, h):
    f = FluxtoJyBeam(inval, h)
    if h.BeamArea() != 0:
        f = f / h.BeamArea()
    return f


# --------------------------------------------------------------------------
# Angles and coordinates
# --------------------------------------------------------------------------
def arcsconv(cunit):
    c = cunit.lower()
    if "deg" in c: return 3600.
    if "arcm" in c: return 60.
    if "arcs" in c: return 1.
    raise ValueError(f"Conversion error (unknown CUNIT for RA-DEC): {cunit}. Cannot convert to ARCSEC.")


def degconv(cunit):
    c = cunit.lower()
    if "deg" in c: return 1.
    if "arcm" in c: return 1. / 60.
    if "arcs" in c: return 1. / 3600.
    raise ValueError(f"Conversion error (unknown CUNIT for RA-DEC): {cunit}. Cannot convert to DEGREE.")


def dmsToDec(dms):
    """'+12:23:34.45' -> decimal degrees (sign preserved)."""
    isNeg = dms.strip().startswith("-")
    parts = (dms.split(":") + ["0", "0"])[:3]
    vals = []
    for p in parts:
        try:
            vals.append(float(p))
        except ValueError:
            vals.append(0.)
    dec = abs(vals[0]) + vals[1] / 60. + vals[2] / 3600.
    return -dec if isNeg else dec


def decToDMS(dec, type="RA", decPrecision=2):
    precision = max(0, decPrecision)
    thisDec = dec
    T = "RA"
    for k in ("RA", "DEC", "GLON", "GLAT"):
        if k in type:
            T = k
            break
    degSize = 2
    sign = ""
    if T == "RA":
        precision += 1
    if T in ("RA", "GLON"):
        if T == "GLON":
            degSize = 3
        thisDec = thisDec % 360.
        if T == "RA":
            thisDec /= 15.
    else:
        sign = "-" if thisDec < 0 else "+"
    dec_abs = abs(thisDec)
    degD = np.floor(dec_abs)
    minD = (dec_abs - degD) * 60.
    minDint = np.floor(minD)
    sec = (minD - minDint) * 60.
    deg, mn = int(degD), int(minDint)
    if abs(sec - 60.) < 10 ** (-precision):
        sec = 0.
        mn += 1
        if mn == 60:
            mn = 0
            deg += 1
    if precision > 0:
        secs = f"{sec:0{precision + 3}.{precision}f}"
    else:
        secs = f"{int(sec):02d}"
    return f"{sign}{deg:0{degSize}d}:{mn:02d}:{secs}"


def angularSeparation(ra1, dec1, ra2, dec2):
    d2r = np.pi / 180.
    phi1, phi2 = dec1 * d2r, dec2 * d2r
    dphi = (dec2 - dec1) * d2r
    dlam = (ra2 - ra1) * d2r
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    a = np.clip(a, 0, 1)
    return 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)) / d2r


def getCenterCoordinates(pos, h):
    """Parse a pair of (x,y) center strings: pixels, degrees ('123.4d') or sexagesimal."""
    import re
    world = [0., 0.]
    isPOS = [False, False]
    for i in range(2):
        s = str(pos[i]).strip().lower()
        ctyp = h.Ctype(i).lower()
        if re.search(r"\d[hms]", s) or (re.search(r"\dd", s) and re.search(r"[ms]", s)):
            # 07h36m51s, +65d35m12s (C++ read 65d35m as 65 degrees)
            from astropy.coordinates import Angle
            world[i] = float(Angle(s, unit="hourangle" if "h" in s else "deg").deg)
        elif "d" in s:
            world[i] = float(s[:s.find("d")])
        elif ":" in s:
            v = dmsToDec(s)
            if "ra" in ctyp:
                v *= 15
            world[i] = v
        else:
            isPOS[i] = True
    if isPOS[0] and isPOS[1]:
        return float(pos[0]), float(pos[1])
    if not isPOS[0] and not isPOS[1]:
        pix = h.wcsToPix(np.array([[world[0], world[1], h.Crval(2) if h.NumAx() > 2 else 0.]]), keep_spec=True)
        return float(pix[0, 0]), float(pix[0, 1])
    raise ValueError("WCS ERROR: please provide both center coordinates.")


# --------------------------------------------------------------------------
# Primary beam correction (GIPSY PRIBEAM)
# --------------------------------------------------------------------------
def Pbcor(x, y, z, flux, h, cutoffOption=1):
    tel = h.Telesc()
    haveCorr = tel == "WSRT" or "VLA" in tel or tel == "ATCA" or tel == "GMRT"
    if not haveCorr or cutoffOption == 0:
        return flux
    try:
        world = h.pixToWCS(np.array([[x, y, 0.]]))[0]
    except Exception:
        return flux
    angdist = angularSeparation(world[0], world[1], h.Crval(0), h.Crval(1))
    vel = AlltoVel(h.getZphys(z), h)
    freq = Vel2Freq(vel, h.Freq0(), "relativistic") / 1E09
    if freq < 0:
        sys.stderr.write(" PBCORR ERROR: Negative frequency. Not applying any correction.")
        return flux
    RF = angdist * 60 * freq
    RF2 = RF * RF
    minval = 0.023
    pbc = 1.
    if tel == "WSRT":
        lam_cm = 299792458. / (freq * 1E9) * 100
        calib = 61.18 if 0 < lam_cm < 35. else (66.4 if 35. <= lam_cm < 70. else 62.9)
        pbc = np.cos(calib * freq * angdist * np.pi / 180.) ** 6
    elif "VLA" in tel:
        a0, a4, legacy = 1., 0., False
        if freq <= 1.43: a1, a2, a3 = -1.329E-03, +6.445E-07, -1.146E-10
        elif 1.43 <= freq < 1.73: a1, a2, a3 = -1.343E-03, +6.579E-07, -1.186E-10
        elif 4.4 <= freq <= 5.1: a1, a2, a3 = -1.372E-03, +6.940E-07, -1.309E-10
        elif 7.9 <= freq <= 8.9: a1, a2, a3 = -1.306E-03, +6.253E-07, -1.100E-10
        elif 14.3 <= freq <= 15.5: a1, a2, a3 = -1.305E-03, +6.155E-07, -1.030E-10
        elif 21.9 <= freq <= 24.1: a1, a2, a3 = -1.417E-03, +7.332E-07, -1.352E-10
        elif 39.9 <= freq <= 50.1: a1, a2, a3 = -1.321E-03, +6.185E-07, -0.983E-10
        else:
            a0, a1, a2, a3, a4 = 0.9920378, 0.9956885E-03, 0.3814573E-05, -0.5311695E-08, 0.3980963E-11
            legacy = True
        polyn = a0 + RF2 * (a1 + RF2 * (a2 + RF2 * (a3 + RF2 * a4)))
        pbc = ((1. / polyn) if polyn > 0 else 0.) if legacy else polyn
        pbc = min(1., max(0., pbc))
    elif tel == "ATCA":
        a0, a1, a2, a3, a4, valid = 1., 0., 0., 0., 0., True
        if 1.25 < freq <= 1.78: a1, a2, a3, a4 = 8.99E-04, 2.15E-06, -2.23E-09, 1.56E-12
        elif 2.2 < freq <= 2.5: a1, a2, a3, a4 = 1.02E-03, 9.48E-07, -3.68E-10, 4.88E-13
        elif 4.4 < freq <= 6.1: a1, a2, a3, a4 = 1.08E-03, 1.31E-06, -1.17E-09, 1.07E-12
        elif 8.0 < freq <= 9.2: a1, a2, a3, a4 = 1.04E-03, 8.36E-07, -4.68E-10, 5.50E-13
        else: valid = False
        polyn = a0 + RF2 * (a1 + RF2 * (a2 + RF2 * (a3 + RF2 * a4)))
        pbc = min(1., (1. / polyn) if polyn > 0 else 0.)
        if valid:
            R = 2500.0
            minval = 1.0 / (a0 + R * (a1 + R * (a2 + R * (a3 + R * a4))))
    elif tel == "GMRT":
        if freq <= 0.175: a = (-4.04, 76.2, -68.8, 22.03)
        elif freq <= 0.277: a = (-3.366, 46.159, -29.963, 7.529)
        elif freq <= 0.480: a = (-3.397, 47.192, -30.931, 7.803)
        elif freq <= 0.900: a = (-3.486, 47.749, -35.203, 10.399)
        else: a = (-2.27961, 21.4611, -9.7929, 1.80153)
        polyn = 1 + RF2 * (a[0] * 1E-03 + RF2 * (a[1] * 1E-07 + RF2 * (a[2] * 1E-10 + RF2 * a[3] * 1E-13)))
        pbc = min(1., max(0., polyn))
    fluxcorr = flux / pbc if pbc > 0 else flux
    if pbc < minval:
        if cutoffOption == 1: fluxcorr = 0
        if cutoffOption == 2: fluxcorr = flux / minval
        if cutoffOption == 3: fluxcorr = np.nan
    return fluxcorr


# --------------------------------------------------------------------------
# Reading data from text files
# --------------------------------------------------------------------------
def getDataColumn(filestring):
    """Parse 'file(name.dat,n,m:k)' and return the column as a list, or None."""
    low = filestring.lower()
    found = low.find("file(")
    if found < 0:
        return None
    s = filestring[found + 5:]
    if s.endswith(")"):
        s = s[:-1]
    parts = s.split(",")
    filename = parts[0].strip()
    col, row_low, row_high = 0, 0, sys.maxsize
    if len(parts) > 1:
        col = int(parts[1]) - 1
    if len(parts) > 2:
        rows = parts[2]
        if ":" not in rows:
            row_low = int(rows) - 1
        else:
            lo, hi = rows.split(":", 1)
            row_low = int(lo) - 1 if lo.strip() else -1
            if hi.strip():
                row_high = int(hi) - 1
    filename = os.path.expanduser(filename)
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"\nERROR: {filename} doesn't exist! \n")
    data = []
    with open(filename) as f:
        for row_n, line in enumerate(f):
            if row_low <= row_n <= row_high:
                toks = line.split()
                if col < len(toks):
                    try:
                        data.append(float(toks[col]))
                    except ValueError:
                        pass
    return data


def getData(filename):
    """Read all numeric data from a text file (list of rows)."""
    out = []
    with open(filename) as f:
        for line in f:
            row = []
            for t in line.split():
                try:
                    row.append(float(t))
                except ValueError:
                    break
            out.append(row)
    return out


# --------------------------------------------------------------------------
# Interpolation / numerics
# --------------------------------------------------------------------------
def polyfit_lsq(x, y, ncoeff, w=None, free=None):
    """Weighted least-squares polynomial y = sum_k c_k x^k (k = 0..ncoeff-1).

    w: weights (points with w <= 0 are ignored); free: which coefficients are fitted (the
    others are fixed to 0). Returns (coeff [increasing order], errors, ok); the errors are
    sqrt(diag((A^T W A)^-1) * chi2 / (n - nfree)), as in GIPSY's lsqfit."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    w = np.ones_like(x) if w is None else np.asarray(w, float)
    free = np.ones(max(ncoeff, 0), bool) if free is None else np.asarray(free, bool)
    coeff, err = np.zeros(max(ncoeff, 1)), np.zeros(max(ncoeff, 1))
    good = w > 0
    nuse, nfree = int(good.sum()), int(free.sum())
    if ncoeff < 1 or nfree == 0 or nfree >= nuse:
        return coeff, err, False
    sw = np.sqrt(w[good])
    A = np.vander(x[good], ncoeff, increasing=True)[:, free] * sw[:, None]
    b = y[good] * sw
    # columns scaled to unit norm (powers of large radii span many orders of magnitude)
    scale = np.linalg.norm(A, axis=0)
    if np.any(scale == 0) or not np.all(np.isfinite(scale)):
        return coeff, err, False
    As = A / scale
    sol, _, rank, _ = np.linalg.lstsq(As, b, rcond=None)
    if rank < nfree:
        return coeff, err, False
    res = b - As @ sol
    try:
        cov = np.linalg.inv(As.T @ As) * (res @ res) / (nuse - nfree)
    except np.linalg.LinAlgError:
        return coeff, err, False
    coeff[free] = sol / scale
    err[free] = np.sqrt(np.clip(np.diag(cov), 0, None)) / scale
    return coeff, err, True


def bezier_interp(x_in, y_in, ns=None):
    """Bezier curve of degree n through n datapoints, evaluated at ns uniform t.
    Returns (x_out, y_out)."""
    x_in = np.asarray(x_in, dtype=float)
    y_in = np.asarray(y_in, dtype=float)
    npnt = len(x_in)
    if ns is None:
        ns = npnt
    from scipy.special import gammaln
    n = npnt - 1
    j = np.arange(npnt)
    logbc = gammaln(n + 1) - gammaln(j + 1) - gammaln(n - j + 1)
    t = np.arange(ns) / (ns - 1) if ns > 1 else np.zeros(1)
    xo, yo = np.empty(ns), np.empty(ns)
    for i, sr in enumerate(t):
        if sr == 0.0:
            xo[i], yo[i] = x_in[0], y_in[0]
        elif sr == 1.0:
            xo[i], yo[i] = x_in[n], y_in[n]
        else:
            u = np.exp(logbc + n * np.log(1 - sr) + j * (np.log(sr) - np.log(1 - sr)))
            xo[i], yo[i] = np.dot(x_in, u), np.dot(y_in, u)
    return xo, yo


def RotMatrices(alpha, beta, gamma):
    """Three rotation matrices (row-major, standard) about the x, y, z axes."""
    out = []
    for i, a in enumerate(np.radians([alpha, beta, gamma])):
        e = np.zeros(3)
        e[i] = 1
        c, s = np.cos(a), np.sin(a)
        K = np.array([[0, -e[2], e[1]], [e[2], 0, -e[0]], [-e[1], e[0], 0]])
        out.append(c * np.eye(3) + (1 - c) * np.outer(e, e) + s * K)
    return out


def RingRegion(r, h):
    """2D region covered by a set of rings (azimuthal angle of each pixel, NaN outside)."""
    nx, ny = h.DimAx(0), h.DimAx(1)
    pscale = h.PixScale() * arcsconv(h.Cunit(0))
    reg = np.full((ny, nx), np.nan)
    radsep = r.radsep
    if radsep < 0:
        radsep = r.radii[1] - r.radii[0] if r.nr > 1 else r.radii[0] / 2.
    R1 = max((r.radii[0] - radsep / 2.) / pscale, 0.)
    R2 = (r.radii[-1] + radsep / 2.) / pscale
    phi, inc, z0 = r.phi[-1], r.inc[-1], 3 * r.z0[-1] / pscale
    x0, y0 = r.xpos[-1] - 1, r.ypos[-1] - 1
    M = RotMatrices(inc, 0., -phi - 90)
    rot = M[2].T @ M[0].T     # C++ MatrixProduct reads the matrices column-major
    xyr, zr = int(np.ceil(R2)), int(np.ceil(z0))
    g = np.arange(-xyr, xyr + 1)
    Y, X = np.meshgrid(g, g, indexing="ij")
    rr = np.hypot(X, Y)
    sel = (rr <= R2) & (rr >= R1)
    X, Y, rr = X[sel], Y[sel], rr[sel]
    theta = np.where(rr < 0.1, 0., np.degrees(np.arctan2(Y, X)))
    for z in range(-zr, zr + 1):
        c = rot @ np.vstack([X, Y, np.full(X.shape, z)])
        xr = np.rint(c[0] + x0).astype(int)
        yr = np.rint(c[1] + y0).astype(int)
        ok = (xr >= 0) & (xr < nx) & (yr >= 0) & (yr < ny)
        for xx, yy, th in zip(xr[ok], yr[ok], theta[ok]):
            if np.isnan(reg[yy, xx]):
                reg[yy, xx] = th
    return reg


def SimulateNoise(stddev, size, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    return rng.normal(0., stddev, size)


_WINDOWS = ("HANNING", "HANNING2", "BOXCAR", "TOPHAT", "FLATTOP", "BARTLETT", "WELCH", "BLACKMAN", "GAUSSIAN")


def window_kernel(windowType, windowSize):
    """Normalised 1D smoothing kernel (port of the kernel built in Smooth1D)."""
    windowType = windowType.upper()
    if windowType not in _WINDOWS:
        sys.stderr.write("Smoothing 1D: window type unknown -> setting to HANNING.\n")
        windowType = "HANNING"
    k = int(windowSize)
    if k % 2 == 0 and windowType != "GAUSSIAN":
        sys.stderr.write(f"Smoothing 1D: need an odd number for the window size. Changing {windowSize} to {windowSize + 1}.\n")
        k += 1
    sigma = 0.
    if windowType == "GAUSSIAN":
        sigma = windowSize / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        k = 2 * int(5 * sigma) + 1
    if k == 1:
        return np.ones(1)               # (a window of one channel: no smoothing; C++ divided by N = 0)
    half = k // 2
    j = np.arange(k, dtype=float)
    x = j - half
    N = k - 1
    scale = (k + 1.) / 2.
    if windowType == "GAUSSIAN": c = np.exp(-x * x / (2 * sigma * sigma))
    elif windowType == "HANNING": c = 0.5 + 0.5 * np.cos(x * np.pi / scale)
    elif windowType == "HANNING2": c = 0.5 - 0.5 * np.cos(j * 2 * np.pi / N)
    elif windowType == "BARTLETT": c = 1 - np.abs((j - N / 2.) / (N / 2.))
    elif windowType == "WELCH": c = 1 - ((j - N / 2.) / (N / 2.)) ** 2
    elif windowType == "BLACKMAN": c = 0.42659 - 0.49656 * np.cos(j * 2 * np.pi / N) + 0.076849 * np.cos(j * 4 * np.pi / N)
    elif windowType == "FLATTOP":
        c = (0.21557895 - 0.416631580 * np.cos(j * 2 * np.pi / N) + 0.277263158 * np.cos(j * 4 * np.pi / N)
             - 0.083578957 * np.cos(j * 6 * np.pi / N) + 0.006947368 * np.cos(j * 8 * np.pi / N))
    else: c = np.ones(k)
    return c / c.sum()


def Smooth1D(inarray, windowType, windowSize, axis=-1):
    """Smooth along one axis with the BBarolo windows (zero-padded edges)."""
    from scipy.ndimage import correlate1d
    kern = window_kernel(windowType, windowSize)
    a = np.asarray(inarray, dtype=float)
    return correlate1d(a, kern, axis=axis, mode="constant", cval=0.0)


# --------------------------------------------------------------------------
# Strings and fun
# --------------------------------------------------------------------------
_GOOD = ["trustful", "awesome", "stunning", "creative", "flabbergasting", "fabulous", "gorgeous",
         "unbelievable", "extraordinary", "breathtaking", "astonishing", "stonking", "brilliant",
         "marvellous", "ungodly", "incredible", "wondrous", "magnificent", "glorious", "splendiferous",
         "wonderful", "phantasmagoric", "phenomenal", "excellent", "exceptional", "refulgent",
         "ambitious", "exuberant", "frank", "witty", "amiable", "fearless", "honest", "ineffable",
         "arcadian", "didactic", "efficacious", "judicious", "propitious", "sagacious",
         "zealous", "flamboyant", "sincere", "exuberant", "charming", "communicative", "enthusiastic",
         "pioneering", "astonishing", "enlightening", "edifying", "illuminating", "homiletic", "indescribable",
         "bucolic", "dazzling", "thrilling", "superior", "outstanding", "empyrean", "sublime", "breezy"]
_BAD = ["flummoxed", "cranky", "pernicious", "modest", "shameful", "wobbling", "knackered", "flippant",
        "wonky", "bellicose", "caustic", "calamitous", "crapulous", "dowdy", "execrable", "fastidious",
        "guileless", "hubristic", "insidious", "insolent", "irksome", "mendacious", "meretricious",
        "noxious", "obtuse", "recalcitrant", "risible", "strident", "wheedling", "withering", "pauciloquent",
        "preposterous", "brazen", "aberrant", "abortive", "barbarous", "bilious", "brash", "capricious",
        "ridiculous", "pointless", "ludicrous", "impudent", "acrimonious", "outmoded", "inelegant", "frowzy",
        "egregious", "discourteous", "malapert", "scornful", "bamboozled", "disastrous", "ruinous", "woeful",
        "deplorable", "grievous", "flagitious", "deleterious", "detrimental", "insalubrious", "inconsequential"]


def randomAdjective(type=1):
    return random.choice(_GOOD if type == 1 else _BAD)


def randomQuoting():
    A = randomAdjective
    quotes = [
        f"BBarolo is such a {A(1)} code", "Why watching TV when you can use BBarolo?",
        "BBarolo is certainly the best code I have ever used", "We all must be thankful for having BBarolo",
        f"BBarolo, a better and {A(1)} alternative to galaxy kinematics",
        f"I literally spend entire nights playing with BBarolo, a {A(1)} software",
        f"Before knowing BBarolo, my life was just {A(2)}", "Try BBarolo, you will not regret it!",
        "The best software experience I ever had", "If you feel down, BBarolo will cheer you up",
        "I was amazed when I discovered I could actually fit 3D kinematic models with BBarolo",
        "You won't believe what BBarolo can do before you see it", "BBarolo is addictive, be aware!",
        "Nothing better than spending a Saturday night with BBarolo", "If I only had discovered BBarolo earlier!",
        "If you like Barolo, you will love BBarolo", "It is simply the best code you will ever use",
        "Yesterday my wife caught me again while I was fitting galaxies with BBarolo",
        f"Just one word can describe BBarolo: {A(1)}!", "BBarolo can't be beaten!",
        "BBarolo is kinematic modelling at its finest level", "I can't believe life existed before BBarolo",
        "Nothing makes me happier than a well attained fit with BBarolo", "BBarolo is a whole new world",
        "Did you know BBarolo is now fully written in Python? Amazing...", "If you don't know how, BBarolo is the answer",
        "BBarolo takes you into a magical, undiscovered world", "Can't think of anything better than BBarolo!",
        "Astronomy will be a better science thanks to BBarolo", f"Simply {A(1)}!",
        "And now we have pyBBarolo too!", "Don't waste your time with 2D models, go straight to 3D with BBarolo",
        "I was so depressed because of beam smearing, BBarolo changed my life", "BBarolo is a dream come true",
        "Luckily BBarolo is free, because it would be priceless", "BBarolo, not a regular code",
        "Emission-line data have no secrets for BBarolo!", "BBarolo should be installed on any modern computer",
        "3D modelling is the new 2D", "If you feel flat, try 3D-Barolo", "BBarolo is not for the faint-hearted!",
        "History should be divided in two epochs: Before BBarolo (BBB) and After BBarolo (ABB)",
        "BBarolo is the medicine for beam-smearing anxiety", "And now you can even ask BBarolo for random quotes!",
        "I have been told that BBarolo is full of Easter eggs...", "Take everything from me, but leave me BBarolo!",
        "BBarolo has been ranked amongst the top 10 most influential codes ever written", "When you go BBarolo, you never go back!"]
    wise = ["Leonardo da Vinci", "Isaac Newton", "Socrates", "Galileo Galilei", "Aristotle", "Albert Einstein",
            "Nikola Tesla", "Archimedes", "Steve Vai", "Michael Schumacher", "Plato", "Marie Curie", "Pythagoras",
            "Johann Sebastian Bach", "Charles Darwin", "Alan Turing", "Hippocrates", "Friedrich Nietzsche", "Euclid",
            "Immanuel Kant", "Confucius", "Homer", "Blaise Pascal", "Sun Tzu", "Voltaire", "Leo Tolstoy", "Enrico Fermi",
            "Dante Alighieri", "Louis Pasteur", "Niccolò Machiavelli", "Isaac Asimov", "Democritus", "Sophocles",
            "Buddha", "Epicurus", "Ptolemy", "Arthur Schopenhauer", "Reinhold Messner", "Bertrand Russell", "Augustus",
            "Frederic Chopin", "Hypathia", "Marcus Tullius Cicero", "Henry Ford", "Guglielmo Marconi", "Aeschylus",
            "Maria De Filippi", "Rosseau", "Carl Linnaeus", "Giordano Bruno", "Giacomo Leopardi", "Marco Polo",
            "Napoleon", "Lao Tsu", "Franz Schubert", "Franz Liszt", "Caravaggio", "Franz Kafka", "Plutarch", "Popeye",
            "Aristarchus of Samos", "Kakaroth", "Amerigo Vespucci", "Donald Duck", "Charlie Brown", "Cleopatra",
            "Enrico Di Teodoro", "Santa Claus", "Easter Bunny", "Mohammed Ali", "Averroes", "Louis Cauchy",
            "Leonhard Euler", "Carl Friedrich Gauss", "Don Matteo", "Edmond Dantès", "Filippo Fraternali", "Anne Boleyn",
            "Lucia Armillotta"]
    return f'"{random.choice(quotes)}" (cit. {random.choice(wise)})'


class ProgressBar:
    """A minimal progress bar (port of Utilities/progressbar.cpp)."""

    def __init__(self, Time=False, Verbose=True, Showbar=True, length=20, s="#"):
        self.ltime = Time
        self.verbose = Verbose
        # (drawn on a terminal only: in a log file the backspaces would pile up)
        self.showbar = Showbar and Verbose and sys.stdout.isatty()
        self.length = length
        self.s = s
        self.size = 1
        self.nvis = 0
        self.t0 = time.time()

    def init(self, title, size):
        if not self.verbose:
            return
        self.size = max(int(size), 1)
        self.nvis = 0
        self.t0 = time.time()
        sys.stdout.write(title)
        if self.showbar:
            sys.stdout.write("|" + " " * self.length + "|")
        sys.stdout.flush()

    def update(self, num):
        if not self.showbar:
            return
        n = int(self.length * min(num, self.size) / self.size)
        if n != self.nvis:
            self.nvis = n
            bar = "|" + self.s * n + " " * (self.length - n) + "|"
            extra = ""
            if self.ltime and num > 0:
                eta = (time.time() - self.t0) / num * (self.size - num)
                extra = f" {int(eta // 60):3d}m{int(eta % 60):02d}s"
            sys.stdout.write("\b" * (self.length + 2 + (len(self._extra) if hasattr(self, "_extra") else 0)) + bar + extra)
            self._extra = extra
            sys.stdout.flush()

    def fillSpace(self, s):
        if not self.verbose:
            return
        if self.showbar:
            n = self.length + 2 + (len(self._extra) if hasattr(self, "_extra") else 0)
            sys.stdout.write("\b" * n + " " * n + "\b" * n)
        sys.stdout.write(s)
        sys.stdout.flush()


def printSpace(n):
    return " " * max(n, 0)
