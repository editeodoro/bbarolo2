"""
Moment maps (port of Tasks/moment.cpp). All the maps are computed with vectorised numpy
operations over the whole cube. Position-velocity slices are in pvslice.py.
"""

import math
import numpy as np
import warnings
import numba as nb

from .image import Image2D
from .bbutils import (AlltoVel, DeltaVel, FluxtoJyBeam, arcsconv, Wave2Vel, Freq2Vel, _spec_unit, _WAVE_UNITS,
                      _FREQ_UNITS)


def _velocities(c, nz=None):
    nz = c.DimZ() if nz is None else nz
    if c.HeadDef():
        return np.asarray(AlltoVel(c.getZphys(np.arange(nz)), c.Head()), dtype=np.float64)
    return np.arange(nz, dtype=np.float64)


_WARNED_BLEND = set()        # (the cubes already warned about blended lines)


def line_velocities(c):
    """Several lines in the cube (more than one RESTWAVE on a wavelength axis, or RESTFREQ on a frequency axis):
    (V, relint), with V (nlines, nz) the velocity of the channels in the frame of each line (rest value at
    REDSHIFT) and relint the relative intensities (RELINT; equal if they do not match the lines). The first line
    is the main one: V[0] is _velocities(c), the velocities of the maps. None with one line."""
    if not c.HeadDef():
        return None
    h, gf = c.Head(), c.pars().gf
    sp, unit, z = h.SpectralType(), _spec_unit(h), h.Redshift()
    x = np.asarray(c.getZphys(np.arange(c.DimZ(), dtype=float)), dtype=np.float64)
    if sp == "wave" and unit in _WAVE_UNITS and len(gf.RESTWAVE) > 1:
        rest = [Wave2Vel(x, w * (1 + z), h.VelDef()) for w in gf.RESTWAVE[1:]]
    elif sp == "freq" and unit in _FREQ_UNITS and len(gf.RESTFREQ) > 1:
        rest = [Freq2Vel(x * _FREQ_UNITS[unit], f / (1 + z), h.VelDef()) for f in gf.RESTFREQ[1:]]
    else:
        return None
    V = np.array([_velocities(c)] + rest, dtype=np.float64)
    relint = np.asarray(gf.RELINT, float) if len(gf.RELINT) == len(V) else np.ones(len(V))
    return V, relint


def line_window(c, mask=None, lv=None, warn=True):
    """Several lines (line_velocities): the channels of the main line, (nz,) bool, and its velocity vc. A channel
    belongs to the line whose centre (at vc, in the frame of each line) is the nearest in velocity, so the window
    ends halfway to the next lines. vc is the flux-weighted velocity of the main line in the total spectrum
    (within mask), starting from that of all the lines corrected for their offsets (with RELINT) and iterated
    with the window. Warns (warn) once per cube when the lines are blended (the window cuts the main line: the total
    spectrum reaches within 2 sigma of its edges). None with one line."""
    lv = line_velocities(c) if lv is None else lv
    if lv is None:
        return None
    V, relint = lv
    a = np.nan_to_num(c.array.astype(np.float64))
    T = (a if mask is None else a * mask).sum(axis=(1, 2))
    # offset of each line in the frame of the main line, for gas at velocity 0
    off = np.array([np.interp(0., Vk[np.argsort(Vk)], V[0][np.argsort(Vk)]) for Vk in V])
    tot = T.sum()
    vc = (T * V[0]).sum() / tot - (relint * off).sum() / relint.sum() if tot > 0 else 0.
    win = None
    for _ in range(20):
        new = np.argmin(np.abs(V - vc), axis=0) == 0
        if not new.any():
            return np.zeros(len(T), bool), vc
        Tw = np.where(new, T, 0.)
        if Tw.sum() > 0:
            vc = (Tw * V[0]).sum() / Tw.sum()
        if win is not None and np.array_equal(new, win):
            break
        win = new
    Tw = np.clip(np.where(win, T, 0.), 0, None)
    if warn and Tw.sum() > 0 and c.Head().fitsname not in _WARNED_BLEND:
        sig = np.sqrt((Tw * (V[0] - vc) ** 2).sum() / Tw.sum())
        vw = V[0][win]
        if min(vc - vw.min(), vw.max() - vc) < 2 * sig:
            _WARNED_BLEND.add(c.Head().fitsname)
            warnings.warn("MOMENT MAPS: the lines (RESTWAVE/RESTFREQ) are blended: the moment maps of the main line "
                          f"use the channels from {vw.min():.0f} to {vw.max():.0f} km/s, which cut its wings. "
                          "MAPTYPE=GAUSSIAN fits all the lines together.", stacklevel=2)
    return win, vc


def compute_moments(c, mask=None):
    """Moment 0,1,2 for all pixels (port of MomentMap::calculateMoments). Returns 3 arrays. With several lines
    (RESTWAVE/RESTFREQ), the moments of the main line (the first): only the channels of line_window()."""
    a = c.array.astype(np.float64)
    vels = _velocities(c)[:, None, None]
    I = a if mask is None else a * mask
    lw = line_window(c, mask)
    if lw is not None:
        I = I * lw[0][:, None, None]
    denom = I.sum(axis=0)
    num = (I * vels).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        m1 = num / denom
        m2 = np.sqrt((I * (vels - m1[None]) ** 2).sum(axis=0) / denom)
    dv = abs(DeltaVel(c.Head())) if c.HeadDef() else 1.
    if c.HeadDef() and c.pars().fluxConvert:
        m0 = FluxtoJyBeam(denom, c.Head()) * dv
    else:
        m0 = denom * dv
    bad = denom == 0
    m0 = np.where(bad, np.nan, m0)
    m1 = np.where(bad, np.nan, m1)
    m2 = np.where(bad, np.nan, m2)
    return m0, m1, m2


def gaussian_moments(c, mask=None, niter=50, maxiter=1000, tol=1e-3):
    """Gaussian fit of all spectra (vectorised Levenberg-Marquardt): niter iterations for all the spectra,
    then up to maxiter for those not converged (as C++ Lsqfit: the chi^2 decreased by less than tol in
    relative terms, or the damping exceeded 1E10). Returns integrated intensity, central velocity,
    dispersion: NaN where the fit did not converge, where a column of the Jacobian is 0, with as many channels
    as parameters or fewer, and for empty spectra (C++ flagged these failures as well).
    With several lines (RESTWAVE/RESTFREQ), one Gaussian per line with one velocity and one dispersion for all
    (in the frame of each line) and free amplitudes, starting from a dispersion of at least one channel; the lines
    with no centre in the cube are left out. The intensity is that of the main line (the first)."""
    a = c.array.astype(np.float64)
    if mask is not None:
        a = a * mask
    nz, ny, nx = a.shape
    S = a.reshape(nz, -1).T
    lv = line_velocities(c)
    if lv is None:
        V = _velocities(c)[None]
        imax = np.argmax(S, axis=1)
        smax = S[np.arange(S.shape[0]), imax]
        amp = [smax]
    else:
        vc = line_window(c, mask, lv, warn=False)[1]
        keep = [0] + [k for k in range(1, len(lv[0])) if lv[0][k].min() <= vc <= lv[0][k].max()]
        V, relint = lv[0][keep], lv[1][keep]
        # the channel of each line for gas at the velocity of each channel of the main line (-1: out of the cube)
        ich = []
        for Vk in V:
            o = np.argsort(Vk)
            i = np.rint(np.interp(V[0], Vk[o], np.arange(nz, dtype=float)[o])).astype(int)
            ich.append(np.where((V[0] < Vk.min()) | (V[0] > Vk.max()), -1, i))
        # starting velocity: where the lines (weighted with RELINT) line up best
        score = sum(r * np.where(i >= 0, S[:, i], 0.) for r, i in zip(relint, ich))
        imax = np.argmax(score, axis=1)
        smax = S[np.arange(S.shape[0]), imax]
        amp = [smax] + [np.where(i[imax] >= 0, S[np.arange(S.shape[0]), i[imax]], 0.) for i in ich[1:]]
    nl = len(V)
    npar = nl + 2
    v0 = V[0][imax]
    # starting dispersion: 10 km/s (as C++); several lines: at least a channel (a narrower start ends in spikes
    # on one channel with the wide channels of optical cubes)
    sig0 = 10.
    if lv is not None and nz > 1:
        sig0 = max(sig0, float(np.abs(np.diff(V[0])).max()))
    p = np.stack(amp + [v0, np.full(S.shape[0], sig0)], 1)
    lam = np.full(S.shape[0], 1e-3)

    def gauss(p):
        arg = (V[None] - p[:, nl, None, None]) / p[:, nl + 1, None, None]
        return arg, np.exp(-arg * arg / 2.)

    def model(p):
        return (p[:, :nl, None] * gauss(p)[1]).sum(axis=1)

    chi = np.sum((S - model(p)) ** 2, axis=1)
    ok = (smax != 0) & (nz > npar)
    conv = np.zeros(S.shape[0], bool)
    fail = np.zeros(S.shape[0], bool)
    act = np.arange(S.shape[0])                 # (all the spectra for niter iterations, as before)
    eye = np.eye(npar)[None]
    for it in range(maxiter):
        if it == niter:
            act = np.flatnonzero(ok & ~conv & ~fail)
        if act.size == 0:
            break
        pa, Sa, la, ca = p[act], S[act], lam[act], chi[act]
        arg, ex = gauss(pa)
        Aex = pa[:, :nl, None] * ex
        sg = pa[:, nl + 1, None, None]
        J = np.concatenate([ex.transpose(0, 2, 1), (Aex * arg / sg).sum(axis=1)[..., None],
                            (Aex * arg * arg / sg).sum(axis=1)[..., None]], 2)
        r = Sa - Aex.sum(axis=1)
        JTJ = np.einsum("nki,nkj->nij", J, J)
        JTr = np.einsum("nki,nk->ni", J, r)
        fail[act] |= ~np.all(np.diagonal(JTJ, axis1=1, axis2=2) > 0, 1)
        A = JTJ + la[:, None, None] * JTJ * eye
        A += 1e-30 * eye
        try:
            dp = np.linalg.solve(A, JTr[..., None])[..., 0]
        except np.linalg.LinAlgError:
            break
        pn = pa + dp
        cn = np.sum((Sa - model(pn)) ** 2, axis=1)
        better = (cn < ca) & np.all(np.isfinite(pn), 1)
        p[act] = np.where(better[:, None], pn, pa)
        chi[act] = np.where(better, cn, ca)
        lam[act] = np.where(better, la / 10, la * 10)
        conv[act] |= (better & (ca - cn <= tol * ca)) | (lam[act] > 1e10)
    ok &= conv & ~fail & np.all(np.isfinite(p), 1)
    sig = p[:, nl + 1]
    integ = np.sqrt(2 * np.pi) * np.abs(sig) * p[:, 0]
    if c.pars().fluxConvert:
        integ = np.sqrt(2 * np.pi) * np.abs(sig) * FluxtoJyBeam(p[:, 0], c.Head())
    out = [integ, p[:, nl], sig]
    out = [np.where(ok, o, np.nan).reshape(ny, nx) for o in out]
    return out


@nb.njit(cache=True, error_model="numpy")
def _rms_one(row, rob, sncut, level):
    sp = row.copy()
    orms = 1E10
    rms = 1E10
    count = 0
    while True:
        if sp.size == 0:
            break
        if rob:
            med = np.median(sp)
            rms = np.median(np.abs(sp - med)) / 0.6745
        else:
            rms = sp.std()
        rat = (orms - rms) / orms
        if rat < level or count > 100 or sp.size < 5:
            break
        count += 1
        orms = rms
        sp = sp[~(sp > sncut * rms)]
    # (with fewer than 5 channels the loop stops at once: the rms of all of them; C++ returned 1E10)
    return rms if count == 0 else orms


@nb.njit(parallel=True, cache=True)
def _rms_iterative(S, rob, sncut, level):
    """Iterative clipped RMS of each spectrum (rows of S), as MomentMap::RMSMap without mask."""
    n = S.shape[0]
    out = np.empty(n)
    for i in nb.prange(n):
        out[i] = _rms_one(S[i], rob, sncut, level)
    return out


class MomentMap(Image2D):
    """A moment map (0: intensity, 1: velocity, 2: dispersion, 3: rms/mass)."""

    def __init__(self):
        super().__init__()
        self.storedtype = -1
        self.cin = None
        self.mask = None

    def input(self, c, mask=None, blo=None, bhi=None):
        self.cin = c
        self.mask = mask
        self.blo = [0, 0, 0] if blo is None else list(blo)
        self.bhi = [c.DimX(), c.DimY(), c.DimZ()] if bhi is None else list(bhi)
        self.array = np.zeros((self.bhi[1] - self.blo[1], self.bhi[0] - self.blo[0]), np.float32)
        self.par = c.pars()

    def setHead(self, type):
        c = self.cin
        if not c.HeadDef():
            self.headDefined = False
            return False
        h = c.Head()
        self.copyHeader(h)
        self.head.setCrpix(0, h.Crpix(0) - self.blo[0])
        self.head.setCrpix(1, h.Crpix(1) - self.blo[1])
        if type == 0:
            self.head.setBtype("intensity")
            bunit = h.Bunit() + " * KM/S"
            if c.pars().fluxConvert and FluxtoJyBeam(1, h) != 1:
                bunit = "JY/BEAM * KM/S"
            self.head.setBunit(bunit)
        elif type in (1, 2):
            if type == 1:
                sp = h.SpectralType()
                if "velo" in sp:
                    self.head.setBtype(sp)
                else:
                    self.head.setBtype({"radio": "velo-radio", "optical": "velo-opt",
                                        "relativistic": "velo-rel"}.get(h.VelDef(), "velo"))
            else:
                self.head.setBtype("dispersion")
            self.head.setBunit("KM/S")
        elif type == 3:
            self.head.setBtype("rms")
        self.head.calcArea()
        self.headDefined = True
        return True

    def _getmask(self, msk):
        c = self.cin
        if msk and self.mask is None:
            if not c.MaskAll():
                c.BlankMask(None, "LARGEST" in c.pars().MaskType)
            self.mask = c.Mask()
        return self.mask if msk else None

    def storeMap(self, msk, whichmap, map_type="MOMENT"):
        if not self.setHead(whichmap) and self.cin.pars().verbose:
            print("MOMENT MAPS warning: cannot create new header.")
        m = self._getmask(msk)
        names = {0: " Extracting intensity map ", 1: " Extracting velocity map ", 2: " Extracting velocity dispersion map "}
        verb = self.cin.pars().verbose
        if verb:
            print(names[whichmap] + ("(GAUSSIAN)... " if map_type == "GAUSSIAN" else "(MOMENT)... "), end="", flush=True)
        mm = gaussian_moments(self.cin, m) if map_type == "GAUSSIAN" else compute_moments(self.cin, m)
        b0, b1 = self.blo, self.bhi
        self.array = mm[whichmap][b0[1]:b1[1], b0[0]:b1[0]].astype(np.float32)
        self.storedtype = whichmap
        if verb:
            print("Done.")

    def ZeroMoment(self, msk, mtype="MOMENT"): self.storeMap(msk, 0, mtype)
    def FirstMoment(self, msk, mtype="MOMENT"): self.storeMap(msk, 1, mtype)
    def SecondMoment(self, msk, mtype="MOMENT"): self.storeMap(msk, 2, mtype)

    def SumMap(self, msk):
        hd = self.cin.headDefined
        self.cin.headDefined = False
        self.ZeroMoment(msk)
        self.cin.headDefined = hd
        self.setHead(0)
        self.head.setBunit(self.cin.Head().Bunit())

    def HIMassDensityMap(self, msk):
        c = self.cin
        bunit = c.Head().Bunit().lower()
        isJY, isK = "/b" in bunit, bunit in ("k", "kelvin")
        if not (isJY or isK):
            raise ValueError("MOMENT MAPS ERROR: Input datacube must be in JY/BEAM or K.")
        self._getmask(msk)          # (the mask of the other maps: it depended on what ran before)
        v = c.pars().verbose
        c.pars().verbose = False
        if v:
            print(" Extracting Mass surfance-density map... ", end="", flush=True)
        self.SumMap(msk)
        c.pars().verbose = v
        if v:
            print("Done. ")
        c.checkBeam()
        bmaj = c.Head().Bmaj() * arcsconv(c.Head().Cunit(0))
        bmin = c.Head().Bmin() * arcsconv(c.Head().Cunit(1))
        dvel = abs(DeltaVel(c.Head()))
        val = self.array.astype(np.float64)
        if isJY:
            val = FluxtoJyBeam(val, c.Head())      # (the sum map is in the units of the cube, e.g. mJy/beam)
        if isK:
            val = val / (1360 * 21.106114 * 21.106114 / (bmaj * bmin))
        # 8846 Msun/pc2 per Jy/beam km/s for a 1x1 arcsec beam (= 2.356E5 D^2 S, as ELLPROF; C++ used 8794)
        self.array = (8846 * val * dvel / (bmaj * bmin)).astype(np.float32)
        self.head.setBunit("Msun/pc2")
        self.head.setBtype("Mass_surfdens")
        self.storedtype = 3

    def RMSMap(self, msk, sncut=1.5, level=0.1):
        c = self.cin
        self.setHead(3)
        rob = c.pars().flagRobustStats
        mask = self._getmask(msk)   # (the mask of the other maps: it depended on what ran before)
        a = c.array
        nz, ny, nx = a.shape
        out = np.zeros((ny, nx), np.float32)
        verb = c.pars().verbose
        if verb:
            print(" Computing RMS map... ", end="", flush=True)
        if msk:
            good = ~np.isnan(a) & ~mask
            arr = np.where(good, a, np.nan).astype(np.float64)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                if rob:
                    med = np.nanmedian(arr, axis=0)
                    out = np.nanmedian(np.abs(arr - med[None]), axis=0) / 0.6745
                else:
                    out = np.nanstd(arr, axis=0)
            out = out.astype(np.float32)
        else:
            S = np.ascontiguousarray(np.where(np.isnan(a), 0, a).astype(np.float64).reshape(nz, -1).T)
            nb.set_num_threads(max(1, min(int(c.pars().threads), nb.config.NUMBA_NUM_THREADS)))
            out = _rms_iterative(S, bool(rob), float(sncut), float(level)).reshape(ny, nx).astype(np.float32)
        self.array = out
        self.storedtype = 4
        if verb:
            print("Done.")

    def SNMap(self, msk, outfolder="", name="_map_0th"):
        """Noise and S/N maps of the total map, written in <outfolder><name>_RMS.fits and <name>_SN.fits
        (outfolder: OUTFOLDER + prefix by default)."""
        c = self.cin
        n_im, s_im = self.SNMaps(msk)
        if outfolder == "":
            outfolder = c.pars().outFolder + c.pars().outPrefix
        n_im.fitswrite_2d(outfolder + name + "_RMS.fits")
        s_im.fitswrite_2d(outfolder + name + "_SN.fits")

    def SNMaps(self, msk):
        """The noise and S/N maps of the total map (two Image2D): the noise expected in each pixel from the
        noise of the cube, the number of channels summed and the taper (MAPS parameters)."""
        c = self.cin
        if self.storedtype != 0:
            self.ZeroMoment(msk, "MOMENT")
        m = self._getmask(msk)
        if not c.StatsDef():
            c.setCubeStats()
        sigma = c.stat().getSpread()
        cc = c.pars().ma.contChans
        Nleft = float(cc[0])
        Nright = float(cc[1]) if len(cc) > 1 else Nleft
        tap = c.pars().ma.taper
        if tap == "hanning1":
            a = 0.25 * ((Nright - 0.75) / (Nright ** 2) + (Nleft - 0.75) / (Nleft ** 2)); b = 0.75; cf = 4. / math.sqrt(6.)
            sigmaBC = sigma / math.sqrt(1. + 4. / 3. * a)
        elif tap == "hanning2":
            a = 0.25 * ((Nright - 0.75) / (Nright ** 2) + (Nleft - 0.75) / (Nleft ** 2)); b = 0.75
            cf = 4. / math.sqrt(2.) / math.sqrt(6.)
            sigmaBC = sigma / math.sqrt(1. + 4. / 3. * a)
        else:
            a = 0.25 * (1. / Nright + 1. / Nleft); b = 0.; cf = 1.
            sigmaBC = sigma / math.sqrt(1. + a)
        lw = line_window(c, m if msk else None)       # (several lines: the channels of the main line)
        chans = np.ones(c.DimZ(), bool) if lw is None else lw[0]
        nchan = (m * chans[:, None, None]).sum(axis=0) if msk else np.full(self.array.shape, chans.sum())
        noise = np.sqrt(nchan - b + a * nchan * nchan) * cf * sigmaBC
        if c.pars().fluxConvert:
            noise = FluxtoJyBeam(noise, c.Head())
        noise = noise * abs(DeltaVel(c.Head()))
        nmap = np.where(noise != 0, noise, np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            snmap = self.array / noise
        n_im = Image2D.fromArray(nmap.astype(np.float32), c.Head())
        n_im.Head().setBtype("noise")
        n_im.Head().setBunit(self.head.Bunit())
        s_im = Image2D.fromArray(snmap.astype(np.float32), c.Head())
        s_im.Head().setBtype("S/N")
        s_im.Head().setBunit("NONE")
        return n_im, s_im


def getAllMoments(c, usemask=True, mask=None, mtype="MOMENT"):
    """Return [mom0, mom1, mom2] MomentMap objects."""
    if mask is None and usemask:
        if not c.MaskAll():
            c.BlankMask()
        mask = c.Mask()
    verb = c.pars().verbose
    if verb:
        print(" Deriving kinematic maps... ", end="", flush=True)
    mm = gaussian_moments(c, mask if usemask else None) if mtype == "GAUSSIAN" else \
        compute_moments(c, mask if usemask else None)
    maps = []
    for i in range(3):
        m = MomentMap()
        m.input(c, mask)
        m.setHead(i)
        m.storedtype = i
        m.array = mm[i].astype(np.float32)
        maps.append(m)
    if verb:
        print("Done.")
    return maps
