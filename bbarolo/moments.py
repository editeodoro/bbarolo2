"""
Moment maps (port of Tasks/moment.cpp). All the maps are computed with vectorised numpy
operations over the whole cube. Position-velocity slices are in pvslice.py.
"""

import math
import numpy as np
import warnings
import numba as nb

from .image import Image2D
from .bbutils import (AlltoVel, DeltaVel, FluxtoJyBeam, arcsconv)


def _velocities(c, nz=None):
    nz = c.DimZ() if nz is None else nz
    if c.HeadDef():
        return np.asarray(AlltoVel(c.getZphys(np.arange(nz)), c.Head()), dtype=np.float64)
    return np.arange(nz, dtype=np.float64)


def compute_moments(c, mask=None):
    """Moment 0,1,2 for all pixels (port of MomentMap::calculateMoments). Returns 3 arrays."""
    a = c.array.astype(np.float64)
    vels = _velocities(c)[:, None, None]
    I = a if mask is None else a * mask
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
    dispersion: NaN where the fit did not converge, where a column of the Jacobian is 0, with 3 channels or
    fewer, and for empty spectra (C++ flagged these failures as well)."""
    a = c.array.astype(np.float64)
    if mask is not None:
        a = a * mask
    nz, ny, nx = a.shape
    S = a.reshape(nz, -1).T
    v = _velocities(c)
    imax = np.argmax(S, axis=1)
    smax = S[np.arange(S.shape[0]), imax]
    p = np.stack([smax, v[imax], np.full(S.shape[0], 10.)], 1)
    lam = np.full(S.shape[0], 1e-3)

    def model(p):
        arg = (v[None] - p[:, 1:2]) / p[:, 2:3]
        return p[:, 0:1] * np.exp(-arg * arg / 2.)

    def chi2(p):
        return np.sum((S - model(p)) ** 2, axis=1)
    chi = chi2(p)
    ok = (smax != 0) & (nz > 3)
    conv = np.zeros(S.shape[0], bool)
    fail = np.zeros(S.shape[0], bool)
    act = np.arange(S.shape[0])                 # (all the spectra for niter iterations, as before)
    for it in range(maxiter):
        if it == niter:
            act = np.flatnonzero(ok & ~conv & ~fail)
        if act.size == 0:
            break
        pa, Sa, la, ca = p[act], S[act], lam[act], chi[act]
        arg = (v[None] - pa[:, 1:2]) / pa[:, 2:3]
        ex = np.exp(-arg * arg / 2.)
        J = np.stack([ex, pa[:, 0:1] * ex * arg / pa[:, 2:3], pa[:, 0:1] * ex * arg * arg / pa[:, 2:3]], 2)
        r = Sa - pa[:, 0:1] * ex
        JTJ = np.einsum("nki,nkj->nij", J, J)
        JTr = np.einsum("nki,nk->ni", J, r)
        fail[act] |= ~np.all(np.diagonal(JTJ, axis1=1, axis2=2) > 0, 1)
        A = JTJ + la[:, None, None] * JTJ * np.eye(3)[None]
        A += 1e-30 * np.eye(3)[None]
        try:
            dp = np.linalg.solve(A, JTr[..., None])[..., 0]
        except np.linalg.LinAlgError:
            break
        pn = pa + dp
        arg = (v[None] - pn[:, 1:2]) / pn[:, 2:3]
        cn = np.sum((Sa - pn[:, 0:1] * np.exp(-arg * arg / 2.)) ** 2, axis=1)
        better = (cn < ca) & np.all(np.isfinite(pn), 1)
        p[act] = np.where(better[:, None], pn, pa)
        chi[act] = np.where(better, cn, ca)
        lam[act] = np.where(better, la / 10, la * 10)
        conv[act] |= (better & (ca - cn <= tol * ca)) | (lam[act] > 1e10)
    ok &= conv & ~fail & np.all(np.isfinite(p), 1)
    integ = np.sqrt(2 * np.pi) * np.abs(p[:, 2]) * p[:, 0]
    if c.pars().fluxConvert:
        integ = np.sqrt(2 * np.pi) * np.abs(p[:, 2]) * FluxtoJyBeam(p[:, 0], c.Head())
    out = [integ, p[:, 1], p[:, 2]]
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
        nchan = m.sum(axis=0) if msk else np.full(self.array.shape, c.DimZ())
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
