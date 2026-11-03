"""
2D images (port of Arrays/image.hh and image.cpp).
"""

import os
import warnings
import numpy as np
from astropy.io import fits

from .header import Header
from .params import Param
from .stats import Stats


class Image2D:

    def __init__(self, shape=None, head=None, par=None):
        self.array = None if shape is None else np.zeros((shape[1], shape[0]), np.float32)
        self.head = Header()
        self.headDefined = False
        self.par = par.copy() if par is not None else Param()
        self.stats = Stats()
        self.statsDefined = False
        if head is not None:
            self.copyHeader(head)

    @classmethod
    def fromArray(cls, array, head=None, par=None):
        im = cls(par=par)
        im.array = np.asarray(array)
        if head is not None:
            im.copyHeader(head)
            im.head.dimAxes = [im.DimX(), im.DimY()]
        return im

    def DimX(self): return self.array.shape[-1]
    def DimY(self): return self.array.shape[0] if self.array.ndim > 1 else 1
    def NumPix(self): return self.array.size
    def Head(self): return self.head
    def Array(self): return self.array
    def pars(self): return self.par
    def setHeadDef(self, b): self.headDefined = b
    def saveHead(self, h): self.head = h.copy(); self.headDefined = True

    def copyHeader(self, h):
        """Copy the first two axes and the main keywords of h (port of copyHeader)."""
        nh = Header(warning=False)
        nh.setNumAx(2)
        for i in range(2):
            nh.dimAxes[i] = h.DimAx(i)
            nh.cdelt[i] = h.Cdelt(i)
            nh.crpix[i] = h.Crpix(i)
            nh.crval[i] = h.Crval(i)
            nh.cunit[i] = h.Cunit(i)
            nh.ctype[i] = h.Ctype(i)
        nh.bunit = h.Bunit()
        nh.bmaj, nh.bmin, nh.bpa = h.Bmaj(), h.Bmin(), h.Bpa()
        nh.epoch = h.epoch
        nh.object = h.Name()
        nh.telescope = h.Telesc()
        nh.freq0 = h.Freq0()
        nh.radesys = h.radesys
        nh.specsys = h.specsys
        nh.veldef = h.veldef
        nh.wave0, nh.redshift = h.wave0, h.redshift
        nh.sptype = h.sptype
        nh._hdr = getattr(h, "_hdr", None) or fits.Header()
        nh.calcArea()
        self.head = nh
        self.headDefined = True

    def readImage(self, fname):
        self.par.imageFile = fname
        if not self.head.header_read(fname):
            return False
        self.headDefined = True
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d = np.asarray(fits.getdata(fname))
        nx, ny = self.head.DimAx(0), self.head.DimAx(1)
        self.array = d.reshape(-1)[:nx * ny].reshape(ny, nx).astype(np.float32)
        return True

    def fitswrite_2d(self, outname, fullHead=False):
        arr = self.array
        if arr.dtype == np.bool_:
            arr = arr.astype(np.int16)
        hdr = self.head.toAstropy(2, fullHead) if self.headDefined else fits.Header()
        d = os.path.dirname(outname)
        if d:
            os.makedirs(d, exist_ok=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fits.PrimaryHDU(data=arr, header=hdr).writeto(outname, overwrite=True, output_verify="silentfix")
        return True

    def setImageStats(self):
        self.stats.setRobust(self.par.flagRobustStats)
        self.stats.calculate(self.array)
        self.stats.setThresholdSNR(self.par.se.snrCut)
        self.statsDefined = True

    def extractGlobalSpectrum(self, cube):
        self.array = cube.array.astype(np.float64).sum(axis=(1, 2)).astype(np.float32)
        return self.array
