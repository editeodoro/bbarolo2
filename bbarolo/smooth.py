"""
Spatial and spectral smoothing (port of Tasks/smooth3D.cpp and Utilities/conv2D.cpp).

The 2D convolution kernels and scale factors are computed exactly as in the C++
code (Wild 1970 convolution beam, Gaussian truncated at 1E-4 of the peak), while
the convolution itself is done with batched real FFTs (``scipy.fft``) over all
channels at once, with the kernel transform cached for each array shape.
"""

import math
import threading
import collections
import numpy as np
import scipy.fft as sfft

from .bbutils import BBaroloError, arcsconv, FluxtoJyBeam, Smooth1D, shortpath

CUTOFFRATIO = 1.0 / 10000.0
MAXCONV = 512 * 512


# --------------------------------------------------------------------------
# Beam algebra
# --------------------------------------------------------------------------
def convpars(oldbeam, newbeam):
    """Convolution beam to go from oldbeam to newbeam (Wild 1970). Angles in deg wrt +X.
    Returns (bmaj, bmin, bpa) or None if impossible."""
    d2r = math.atan(1.) / 45.
    a2, b2 = oldbeam[0] / 2, oldbeam[1] / 2
    a0, b0 = newbeam[0] / 2, newbeam[1] / 2
    D0 = a0 * a0 - b0 * b0
    D2 = a2 * a2 - b2 * b2
    th2, th0 = oldbeam[2] * d2r, newbeam[2] * d2r
    D1 = math.sqrt(max(D0 * D0 + D2 * D2 - 2 * D0 * D2 * math.cos(2 * (th0 - th2)), 0.))
    arg = 0.5 * (a0 * a0 + b0 * b0 - a2 * a2 - b2 * b2 + D1)
    if arg < 0:
        print("SMOOTHING error: unsuitable new beam parameters!")
        return None
    a1 = math.sqrt(arg)
    arg = 0.5 * (a0 * a0 + b0 * b0 - a2 * a2 - b2 * b2 - D1)
    if arg < 0:
        if arg > -1e-9 * max(a0 * a0, 1e-30):
            arg = 0.
        else:
            print("SMOOTHING error: unsuitable new beam parameters!")
            return None
    b1 = math.sqrt(arg)
    nom = D0 * math.sin(2 * th0) - D2 * math.sin(2 * th2)
    den = D0 * math.cos(2 * th0) - D2 * math.cos(2 * th2)
    th1 = 0. if (den == 0 and nom == 0) else math.atan2(nom, den) / 2
    return (2 * a1, 2 * b1, th1 / d2r)


def perBeam(h):
    """Whether the flux unit of the header h is per beam (Jy/beam, mJy/beam, uJy/beam, W.U., or a unit not
    known), so that SMOOTH rescales the data by the ratio of the beam areas. Units per pixel (Jy, mJy) and K
    are not."""
    b = h.Bunit().lower().replace(" ", "")
    return FluxtoJyBeam(1., h) == 1 or "jy/b" in b or "j/b" in b or "w.u." in b or "wu" in b


def fillgauss2d(beam, gridspace, ampl=1.0, norm=True, cutoff=CUTOFFRATIO):
    """Rotated 2D Gaussian sampled on the pixel grid (port of Fillgauss2d).
    beam=(bmaj,bmin,pa[deg, wrt +X]) in the same units as gridspace. Returns array (ny,nx)."""
    amplitude = 1.0 if norm else ampl
    phi = beam[2] * math.pi / 180.
    cs, sn = math.cos(phi), math.sin(phi)
    bm = (abs(beam[0]), abs(beam[1]))
    extend = math.sqrt(-1.0 * math.log(cutoff) / math.log(2.0))
    xr, yr = 0.5 * bm[0] * extend, 0.5 * bm[1] * extend
    x = max(abs(xr * cs), abs(-yr * sn))
    y = max(abs(xr * sn), abs(yr * cs))
    gs = (abs(gridspace[0]), abs(gridspace[1]))
    Xmax, Ymax = int(round(x / gs[0])), int(round(y / gs[1]))
    if (2 * Xmax + 1) * (2 * Ymax + 1) > MAXCONV:
        print("SMOOTH error: Convolution function too big for buffer.")
        return None
    jj, ii = np.mgrid[-Ymax:Ymax + 1, -Xmax:Xmax + 1]
    xx, yy = ii * gs[0], jj * gs[1]
    xrr = xx * cs + yy * sn
    yrr = -xx * sn + yy * cs
    argX = xrr / bm[0] if bm[0] != 0 else 0 * xrr
    argY = yrr / bm[1] if bm[1] != 0 else 0 * yrr
    c = np.exp(-4.0 * math.log(2.0) * (argX * argX + argY * argY))
    c = np.where(c >= cutoff, c * amplitude, 0.)
    if norm:
        c = c / c.sum()
    return c


def gaussian_scalefactor(oldbeam_rot, confie, gridspace):
    """Scale factor 1/(G_old * K)(0) (port of the scale computation in defineBeam_Gaussian)."""
    g = fillgauss2d(oldbeam_rot, gridspace, 1.0, norm=False)
    ky, kx = confie.shape
    gy, gx = g.shape
    # value at the centre of the convolution of g with confie (both centred)
    hy, hx = min(ky, gy) // 2, min(kx, gx) // 2
    cg = g[gy // 2 - hy:gy // 2 + hy + 1, gx // 2 - hx:gx // 2 + hx + 1]
    ck = confie[ky // 2 - hy:ky // 2 + hy + 1, kx // 2 - hx:kx // 2 + hx + 1]
    val = float(np.sum(cg * ck[::-1, ::-1]))
    return 1.0 / val if val != 0 else 1.0


# --------------------------------------------------------------------------
# FFT convolution engine ("same" linear convolution, as LINEAR_SAME in conv2D)
# --------------------------------------------------------------------------
class FFTConvolver:
    """Convolve stacks of 2D images with a fixed kernel. Kernel FFTs are cached per shape (the
    MAXCACHE most recent shapes: with free centre, thickness or RADMAX, the likelihood of
    Bayesian3DFIT makes models in boxes of many sizes)."""

    MAXCACHE = 32

    def __init__(self, kernel, workers=1):
        self.kernel = np.asarray(kernel, dtype=np.float64)
        self.workers = workers
        self._cache = collections.OrderedDict()
        self._lock = threading.Lock()

    def __getstate__(self):
        # the lock is not picklable, and the cached kernel FFTs are recomputed when needed
        return {"kernel": self.kernel, "workers": self.workers}

    def __setstate__(self, st):
        self.__init__(st["kernel"], st["workers"])

    def _kfft(self, ny, nx):
        key = (ny, nx)
        with self._lock:
            k = self._cache.get(key)
            if k is not None:
                self._cache.move_to_end(key)
                return k
        ky, kx = self.kernel.shape
        Ny = sfft.next_fast_len(ny + ky // 2, real=True)
        Nx = sfft.next_fast_len(nx + kx // 2, real=True)
        kf = sfft.rfft2(self.kernel, s=(Ny, Nx))
        k = (Ny, Nx, kf)
        with self._lock:
            self._cache[key] = k
            while len(self._cache) > self.MAXCACHE:
                self._cache.popitem(last=False)
        return k

    def __call__(self, data, workers=None):
        """data: (..., ny, nx). Returns the 'same' convolution (float64)."""
        data = np.ascontiguousarray(data, dtype=np.float64)
        if data.ndim == 3:
            # Only non-empty planes need transforming (models are empty outside the line)
            nz = data.shape[0]
            full = data.reshape(nz, -1).any(axis=1)
            nfull = int(full.sum())
            if nfull < nz:
                out = np.zeros(data.shape, np.float64)
                if nfull:
                    out[full] = self(data[full], workers)
                return out
        ny, nx = data.shape[-2:]
        Ny, Nx, kf = self._kfft(ny, nx)
        ky, kx = self.kernel.shape
        w = self.workers if workers is None else workers
        f = sfft.rfft2(data, s=(Ny, Nx), workers=w)
        f *= kf
        r = sfft.irfft2(f, s=(Ny, Nx), workers=w)
        return r[..., ky // 2:ky // 2 + ny, kx // 2:kx // 2 + nx]


# --------------------------------------------------------------------------
# Smooth3D
# --------------------------------------------------------------------------
class Smooth3D:
    """Spatial smoothing of a datacube to a new Gaussian beam."""

    def __init__(self):
        self.usescalefac = True
        self.useBlanks = True
        self.scalefac = -1.
        self.fft = True
        self.array = None
        self.confie = None
        self.NconX = self.NconY = 0
        self.oldbeam = self.newbeam = self.conbeam = None
        self.blo = [0, 0, 0]
        self.bhi = None

    def setUseScalefac(self, b): self.usescalefac = b
    def setUseBlanks(self, b): self.useBlanks = b
    def Array(self): return self.array
    def Scalefac(self): return self.scalefac
    def Confie(self): return self.confie

    def setfromCube(self, c, oldbeam, newbeam, bhi=None, blo=None):
        self.cube = c
        h = c.Head()
        unittoarc = arcsconv(h.Cunit(0))
        self.gridspace = (h.Cdelt(0) * unittoarc, h.Cdelt(1) * unittoarc)
        dims = [c.DimX(), c.DimY(), c.DimZ()]
        self.bhi = list(bhi) if bhi is not None else dims
        self.blo = list(blo) if blo is not None else [0, 0, 0]
        self.crota = h.Crota()
        return self.defineBeam_Gaussian(oldbeam, newbeam)

    def defineBeam_Gaussian(self, oldbeam, newbeam):
        ob, nb = list(oldbeam), list(newbeam)
        # a beam given with BMAJ < BMIN: the axes are swapped, and the major axis is at PA + 90 (C++ kept the
        # PA, and compared the new beam with the old one before swapping its axes)
        for b in (ob, nb):
            if b[0] < b[1]:
                print("Major axis < minor axis. Inverting...", end="")
                b[0], b[1], b[2] = b[1], b[0], b[2] + 90.
        if not (nb[0] >= ob[0] and nb[1] >= ob[1]):
            raise BBaroloError(f"SMOOTH: the new beam ({nb[0]:g} x {nb[1]:g} arcsec) is smaller than the old beam "
                               f"({ob[0]:g} x {ob[1]:g} arcsec).")
        self.oldbeam, self.newbeam = tuple(ob), tuple(nb)
        cb = convpars(ob, nb)
        if cb is None:
            raise BBaroloError(f"SMOOTH: cannot calculate the convolution from the beam {ob} to the beam {nb}.")
        cb = [cb[0], cb[1], cb[2] - (self.crota - 90)]
        conf = fillgauss2d(cb, self.gridspace, 1.0, norm=True)
        if conf is None:
            print("SMOOTH error: Convolution region to big!")
            return False
        self.confie = conf
        self.NconY, self.NconX = conf.shape
        cb[2] += self.crota
        self.conbeam = tuple(cb)
        if self.scalefac == -1:
            obr = (ob[0], ob[1], ob[2] + 90 - self.crota)
            self.scalefac = gaussian_scalefactor(obr, conf, self.gridspace)
        self.convolver = FFTConvolver(conf)
        return True

    def smooth(self, c, oldbeam, newbeam, oldarray=None, bhi=None, blo=None, threads=None):
        """Smooth cube c (or oldarray with the same shape) from oldbeam to newbeam (arcsec, deg).
        Returns the smoothed array (also stored in self.array)."""
        if not self.setfromCube(c, oldbeam, newbeam, bhi, blo):
            return None
        verb = c.pars().verbose and oldarray is None
        if verb:
            self._print()
        src = c.array if oldarray is None else oldarray
        b0, b1 = self.blo, self.bhi
        sub = src[b0[2]:b1[2], b0[1]:b1[1], b0[0]:b1[0]]
        data = np.nan_to_num(np.asarray(sub, dtype=np.float64), nan=0.0)
        scal = self.scalefac if self.usescalefac else 1.0
        nth = threads if threads is not None else c.pars().threads
        out = self.convolver(data, workers=nth) * scal
        if self.useBlanks:
            csub = c.array[b0[2]:b1[2], b0[1]:b1[1], b0[0]:b1[0]]
            blank = (csub == 0) | np.isnan(csub)
            out[blank] = np.nan if self.fft else 0.
        self.array = out.astype(np.float32 if src.dtype == np.float32 else np.float64)
        return self.array

    def _print(self):
        ob, nb, cb = self.oldbeam, self.newbeam, self.conbeam
        print("=" * 22 + " SMOOTH " + "=" * 24)
        print(f"{'Convert old beam: ':>20}{ob[0]:7.2f},{ob[1]:7.2f}  arcsec{ob[2]:7.2f} deg")
        print(f"{'to the new beam: ':>20}{nb[0]:7.2f},{nb[1]:7.2f}  arcsec{nb[2]:7.2f} deg")
        print(f"{'with conv. beam: ':>20}{cb[0]:7.2f},{cb[1]:7.2f}  arcsec{cb[2] + 90:7.2f} deg\n")
        print(f"{'Scale factor: ':>20}{self.scalefac:7.4f}")
        print("=" * 54 + "\n")

    def cubesmooth(self, c):
        """Smooth a cube using the SMOOTH parameters stored in c.pars()."""
        p = c.pars()
        h = c.Head()
        unittoarc = arcsconv(h.Cunit(0))
        # the old beam: that of the cube (BEAM, if given, replaced the beam of the header when it was read)
        OB = [h.Bmaj() * unittoarc, h.Bmin() * unittoarc, h.Bpa()]
        if OB[0] == 0 or OB[1] == 0:
            raise BBaroloError("SMOOTH: the cube has no beam: give it with BEAM (BMAJ BMIN BPA, in arcsec and "
                               "degrees).")
        nb = p.newbeam
        NB = [p.factor * OB[0], p.factor * OB[1], OB[2]] if nb is None else \
            [nb[0], nb[1], OB[2] if nb[2] is None else nb[2]]
        dims = [c.DimX(), c.DimY(), c.DimZ()]
        Blo, Bhi = [0, 0, 0], list(dims)
        for j in range(3):
            lo, hi = p.BOX[2 * j], p.BOX[2 * j + 1]
            Blo[j] = 0 if (lo < 0 or lo > dims[j]) else lo
            Bhi[j] = dims[j] if (hi < 0 or hi > dims[j]) else hi
        self.fft = p.flagFFT
        self.scalefac = p.scalefactor
        if self.scalefac == -1 and not perBeam(h):
            self.scalefac = 1                   # (units not per beam: the data are not rescaled)
        return self.smooth(c, OB, NB, bhi=Bhi, blo=Blo)

    def result(self):
        """The smoothed cube (a Cube) as the SMOOTH task gives it: the new beam and the HISTORY cards in the
        header, cropped to the BOX, and reduced by FACTOR with REDUCE."""
        from .cube import Cube
        c = self.cube
        p = c.pars()
        out = Cube.fromArray(self.array, c.Head(), p)
        h = out.Head()
        for i in range(3):
            h.setDimAx(i, self.array.shape[2 - i])
            h.setCrpix(i, c.Head().Crpix(i) - self.blo[i])
        h.setBmaj(self.newbeam[0] / 3600.)
        h.setBmin(self.newbeam[1] / 3600.)
        h.setBpa(self.newbeam[2])
        h.calcArea()
        nb = f"{self.newbeam[0]:.6f} x {self.newbeam[1]:.6f} arcsec, pa = {self.newbeam[2]:.6f} deg"
        ob = f"{self.oldbeam[0]:.6f} x {self.oldbeam[1]:.6f} arcsec, pa = {self.oldbeam[2]:.6f} deg"
        h.Keys().append("HISTORY BBAROLO SMOOTHING: New beam " + nb)
        h.Keys().append("HISTORY BBAROLO SMOOTHING: Old beam " + ob)
        factor = int(math.floor(p.factor))
        if p.getflagReduce() and factor > 1:
            return out.Reduce(factor, "spatial")
        out.Head().setMinMax(np.nanmin(out.array), np.nanmax(out.array))
        return out

    def outname(self):
        """The file of the smoothed cube: SMO_OUT, or <outfolder><prefix>_s<new bmaj>[red].fits."""
        p = self.cube.pars()
        name = p.smo_out
        if name == "NONE":
            name = p.outFolder + p.outPrefix + f"_s{int(round(self.newbeam[0]))}"
            if p.getflagReduce():
                name += "red"
            name += ".fits"
        return name

    def fitswrite(self, outname=None):
        name = self.outname() if outname is None else outname
        self.result().fitswrite_3d(name, True)
        return name


class SpectralSmooth3D:
    """Smoothing along the spectral axis with Hanning/Boxcar/... windows."""

    def __init__(self, wtype="HANNING", wsize=3):
        self.windowtype = wtype.upper()
        self.windowsize = wsize
        self.array = None

    def Array(self):
        return self.array

    def smooth(self, c, verbose=None):
        verb = c.pars().verbose if verbose is None else verbose
        if verb:
            print(f" Spectral smoothing ({self.windowtype}) ...", end="", flush=True)
        self.array = Smooth1D(np.nan_to_num(c.array), self.windowtype, self.windowsize, axis=0).astype(c.array.dtype)
        if verb:
            print(" Done!")
        return self.array

    def result(self, templ):
        """The smoothed cube (a Cube, with the header of the cube templ that was smoothed) as the SMOOTHSPEC
        task gives it: a HISTORY card, and reduced spectrally with REDUCE."""
        from .cube import Cube
        p = templ.pars()
        out = Cube.fromArray(self.array, templ.Head(), p)
        out.Head().Keys().append(f"HISTORY BBAROLO SPECTRAL SMOOTHING: {self.windowtype} window of size {int(self.windowsize)} channels")
        if p.getflagReduce() and self.windowsize > 1:
            bins = int((self.windowsize + 1) / 2) if self.windowtype == "HANNING" else int(self.windowsize)
            return out.Reduce(bins, "spectral")
        out.Head().setMinMax(np.nanmin(out.array), np.nanmax(out.array))
        return out

    def outname(self, templ):
        """The file of the smoothed cube: <outfolder><prefix>_h<window size>[_red].fits."""
        p = templ.pars()
        outname = p.outFolder + p.outPrefix + f"_h{int(self.windowsize)}"
        if p.getflagReduce():
            outname += "_red"
        return outname + ".fits"

    def fitswrite(self, templ, outname="", result=None):
        """Write the smoothed cube (result: the Cube of result(templ), if already computed)."""
        p = templ.pars()
        if outname == "":
            outname = self.outname(templ)
        (result if result is not None else self.result(templ)).fitswrite_3d(outname, True)
        if p.verbose:
            print(f" Spectrally-smoothed datacube written in {shortpath(outname)}")
        return outname
