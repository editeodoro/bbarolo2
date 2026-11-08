"""
REND3D task: 3D rendering of a datacube, i.e. projections of the (masked) cube
seen from different azimuthal angles around the spatial y axis (port of
Tasks/rendering3D.hh). The projection kernel is compiled with numba and the
angles are processed in parallel.
"""

import math
import numpy as np
import numba as nb

from .smooth import Smooth3D


@nb.njit(inline="always", cache=True)
def _lround(x):
    if x >= 0:
        return int(x + 0.5)
    return -int(-x + 0.5)


@nb.njit(parallel=True, cache=True)
def _render(data, mask, nang):
    nz, ny, nx = data.shape
    out = np.zeros((nang, ny, nx), np.float64)
    cx, cz = nx / 2., nz / 2.
    for i in nb.prange(nang):
        c = math.cos(i * math.pi / 180.)
        s = math.sin(i * math.pi / 180.)
        count = 0
        for z in range(nz):
            zc = z - cz
            for x in range(nx):
                xc = x - cx
                x1 = _lround(xc * c + zc * s + cx)
                z1 = _lround(-xc * s + zc * c + cz)
                if x1 < 0 or x1 >= nx or z1 < 0 or z1 >= nz:
                    continue
                for y in range(ny):
                    if mask[z1, y, x1]:
                        out[i, y, x] += data[z1, y, x1]
                        count += 1
        if count > 0:
            for y in range(ny):
                for x in range(nx):
                    out[i, y, x] /= count
        else:
            for y in range(ny):
                for x in range(nx):
                    out[i, y, x] = np.nan
    return out


class Rendering3D:

    def __init__(self, c):
        self.cin = c
        self.out = None

    def compute(self, azangle=360., smoothfactor=1.2):
        """The rendered cube (self.out): one plane per line-of-sight angle (azangle degrees in all, one
        degree apart), smoothed to smoothfactor times the beam (1: not smoothed)."""
        from .cube import Cube
        c = self.cin
        if not c.MaskAll():
            c.BlankMask()
        nang = int(azangle)
        verb = c.pars().verbose
        if verb:
            print(" Rendering 3D... ", end="", flush=True)
        nb.set_num_threads(max(1, min(int(c.pars().threads), nb.config.NUMBA_NUM_THREADS)))
        data = np.where(np.isnan(c.array), 0, c.array).astype(np.float32)
        out = _render(data, np.ascontiguousarray(c.Mask()), nang)
        h = c.Head().copy()
        h.setCrpix(2, 1)                 # (plane 1 is the angle 0; C++ wrote CRPIX3 = 0)
        h.setCrval(2, 0)
        h.setCdelt(2, 1)
        h.setCunit(2, "DEGREES")
        h.setCtype(2, "L.O.S. ANGLE")
        self.out = Cube.fromArray(out.astype(np.float32), h, c.pars())
        if smoothfactor > 1:
            h = c.Head()
            sm = Smooth3D()
            sm.setUseBlanks(False)
            res = sm.smooth(self.out, (h.Bmaj() * 3600, h.Bmin() * 3600, h.Bpa()),
                            (smoothfactor * h.Bmaj() * 3600, smoothfactor * h.Bmin() * 3600, h.Bpa()),
                            oldarray=self.out.array, threads=c.pars().threads)
            if res is not None:
                self.out.array = np.asarray(res, np.float32)
        if verb:
            print("Done.")

    def writefits(self, fname=""):
        c = self.cin
        if not fname:
            fname = c.pars().outFolder + c.pars().outPrefix + "_3D.fits"
        self.out.fitswrite_3d(fname)
