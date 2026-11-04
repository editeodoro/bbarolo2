"""
The datacube class (port of Arrays/cube.hh and cube.cpp).

Data are stored in a numpy array of shape (nz, ny, nx) (FITS order), so that the
flat C-order index equals the C++ index ``x + y*nx + z*nx*ny``.
"""

import os
import sys
import math
import shutil
import warnings
import numpy as np
from astropy.io import fits

from .header import Header
from .params import Param, stringize
from .stats import Stats, findMADFM
from .bbutils import (BBaroloError, arcsconv, DeltaVel, FluxtoJy, FluxtoJyBeam, isFluxUnitKnown, AlltoVel,
                      ProgressBar, mkdirp, randomAdjective, shortpath)


class Cube:

    def __init__(self, fname=None, par=None, printInfo=False, array=None, head=None):
        self.par = par.copy() if par is not None else Param()
        self.head = head.copy() if head is not None else Header()
        self.headDefined = head is not None
        self.array = None
        self.mask = None
        self.stats = Stats(self.par.flagRobustStats)
        self.statsDefined = False
        self.sources = None
        self.isSearched = False
        if array is not None:
            self.array = np.asarray(array)
        if fname is not None:
            self.par.imageFile = fname
            if not self.readCube(fname, printInfo):
                raise IOError(f"Cannot read FITS file {fname}")

    @classmethod
    def fromArray(cls, array, head, par=None):
        c = cls(par=par, array=np.asarray(array), head=head)
        if head is not None:
            nd = c.array.ndim
            for i in range(min(3, c.head.NumAx())):
                c.head.setDimAx(i, c.array.shape[nd - 1 - i] if i < nd else 1)
        return c

    def copy(self):
        c = Cube(par=self.par, head=self.head if self.headDefined else None)
        c.headDefined = self.headDefined
        c.array = None if self.array is None else self.array.copy()
        c.mask = None if self.mask is None else self.mask.copy()
        c.stats = self.stats.copy()
        c.statsDefined = self.statsDefined
        c.sources, c.isSearched = self.sources, self.isSearched
        return c

    # ------------------------------------------------------------------ accessors
    def DimX(self): return self.array.shape[2]
    def DimY(self): return self.array.shape[1]
    def DimZ(self): return self.array.shape[0]
    def AxisDim(self): return [self.DimX(), self.DimY(), self.DimZ()]
    def AxesDim(self, i): return self.AxisDim()[i]
    def NumPix(self): return self.array.size
    def Head(self): return self.head
    def pars(self): return self.par
    def stat(self): return self.stats
    def Mask(self): return self.mask
    def MaskAll(self): return self.mask is not None
    def StatsDef(self): return self.statsDefined
    def HeadDef(self): return self.headDefined
    def getSources(self): return self.sources
    def getNumObj(self): return 0 if self.sources is None else self.sources.getNumObj()
    def pObject(self, i): return self.sources.pObject(i)
    def saveHead(self, h): self.head = h.copy(); self.headDefined = True
    def saveParam(self, p): self.par = p.copy()

    def nPix(self, x, y, z): return x + y * self.DimX() + z * self.DimX() * self.DimY()
    def getZphys(self, z): return self.head.getZphys(z)
    def getZgrid(self, v): return self.head.getZgrid(v)
    def getXphys(self, x): return (x + 1 - self.head.Crpix(0)) * self.head.Cdelt(0) + self.head.Crval(0)
    def getYphys(self, y): return (y + 1 - self.head.Crpix(1)) * self.head.Cdelt(1) + self.head.Crval(1)

    def setBeam(self, a, b, c): self.head.setBeam(a, b, c)

    # ------------------------------------------------------------------ I/O
    def checkBeam(self):
        h, p = self.head, self.par
        if h.BeamArea() == 0:
            bmaj, bmin, bpa = p.bmaj / 3600., p.bmin / 3600., p.bpa
            if p.bmaj < 0: bmaj = -1
            if p.bmin < 0: bmin = -1
            if bmaj > 0 and bmin < 0: bmin = bmaj
            if bmin > 0 and bmaj < 0: bmaj = bmin
            if bmaj < 0 and bmin < 0 and p.beamFWHM > 0:
                bmaj = bmin = p.beamFWHM
            if bmaj < 0 and bmin < 0:
                raise BBaroloError("No beam information available for the data: set the beam with BMAJ, BMIN "
                                   "and BPA, or BEAMFWHM (in arcsec), or beam=(bmaj, bmin, bpa) in Python.")
            print(f"\n WARNING: Beam not available in the header: using a {bmaj * 3600.:.5g}x{bmin * 3600.:.5g}"
                  f" arcsec (BPA={bpa:.5g}).\n"
                  "          You can set the beam with BMAJ/BMIN/BPA or BeamFWHM params (in arcsec).\n")
            pix = h.PixScale()
            if min(bmaj, bmin) < 0.25 * pix:
                raise BBaroloError(f"The beam ({bmaj * 3600.:.4g}x{bmin * 3600.:.4g} arcsec) is smaller than a "
                                   f"quarter of a pixel ({pix * 3600.:.4g} arcsec): BMAJ, BMIN and BEAMFWHM are in "
                                   "arcsec.")
            h.setBmaj(bmaj)
            h.setBmin(bmin)
            h.setBpa(bpa)
            h.calcArea()
        elif min(h.Bmaj(), h.Bmin()) < h.PixScale() and not getattr(self, "_beamwarned", False):
            self._beamwarned = True
            print(f"\n WARNING: the beam ({h.Bmaj() * 3600.:.4g}x{h.Bmin() * 3600.:.4g} arcsec) is smaller than a "
                  f"pixel ({h.PixScale() * 3600.:.4g} arcsec).\n")

    def readCube(self, fname, printInfo=True):
        p = self.par
        p.imageFile = fname
        h = self.head
        h.warning = p.verbose
        if not h.header_read(fname):
            return False

        def getdata():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return fits.getdata(fname, memmap=False)
        return self._readHeaderData(fname, getdata, printInfo)

    def readHDU(self, hdu, name="in-memory cube", printInfo=False):
        """Like readCube, but from an astropy HDU (header and data in memory, no file)."""
        p = self.par
        p.imageFile = name
        h = self.head
        h.warning = p.verbose
        h.fitsname = name
        if not h.fromAstropy(hdu.header.copy()):
            return False
        return self._readHeaderData(name, lambda: np.asarray(hdu.data), printInfo)

    def _readHeaderData(self, fname, getdata, printInfo):
        """The part of readCube after the header has been read (getdata() gives the array)."""
        p = self.par
        h = self.head
        if not h.checkHeader():
            print("\nBBAROLO WARNING: Something seems wrong with the header. Fix it before going on. ")
        beamNeeded = (p.flagSmooth or p.makeMask or p.gf.flagGALFIT or p.gm.flagGALMOD or p.flagSpace
                      or (p.se.flagSearch and p.se.minPix == -1))
        if beamNeeded:
            self.checkBeam()
        self.headDefined = True
        if p.getRestfreq() != -1:
            h.setFreq0(p.getRestfreq())
        dims = [h.DimAx(i) if i < h.NumAx() else 1 for i in range(3)]
        stokes3 = h.NumAx() > 3 and h.Ctype(2).lower() == "stokes"
        if stokes3:
            h.setCrpix(2, h.Crpix(3)); h.setCrval(2, h.Crval(3)); h.setCdelt(2, h.Cdelt(3))
            h.setCtype(2, h.Ctype(3)); h.setCunit(2, h.Cunit(3))
            h.setDimAx(2, h.DimAx(3))
            dims[2] = h.DimAx(2)
            h.updateWCS()
        h.setRedshift(p.getRedshift())
        h.setWave0(p.getRestwave())
        sptype = h.SpectralType()
        if sptype == "velo-radio":
            h.setVelDef("radio")
        elif sptype == "velo-opt":
            h.setVelDef("optical")
        else:
            if p.ma.veldef == "auto":
                if sptype == "freq":
                    h.setVelDef("radio")
                    print("\nBBAROLO WARNING: I will use a radio definition for velocity. "
                          "This can be changed by using VELDEF parameter.")
                elif sptype == "wave":
                    h.setVelDef("relativistic")
                    print("\nBBAROLO WARNING: I will use a relativistic definition of velocity. "
                          "This can be changed by using VELDEF parameter.")
            else:
                h.setVelDef(p.ma.veldef)

        if p.verbose:
            npx = dims[0] * dims[1] * dims[2]
            print(f"\nOpening file {shortpath(fname)}")
            print(f"Reading {dims[0]} x {dims[1]} x {dims[2]} pixels FITS file ({4 * npx / 1048576.:.1f} MB)... ", end="")
        data = np.asarray(getdata())
        if data.ndim == 4:
            # the first Stokes plane (or the first plane of the 4th axis): (FREQ, STOKES, y, x) when
            # the third axis is the Stokes one, otherwise (STOKES, FREQ, y, x)
            data = data[:, 0] if stokes3 else data[0]
        n = dims[0] * dims[1] * dims[2]
        data = data.reshape(-1)[:n].reshape(dims[2], dims[1], dims[0])
        data = data.astype(np.float32 if data.dtype.itemsize <= 4 else np.float64, copy=True)
        data[np.isnan(data)] = 0
        self.array = np.ascontiguousarray(data)
        if p.verbose:
            print("Done. \n")
        if p.outPrefix == "":
            p.outPrefix = h.Name()

        if printInfo and p.verbose:
            if dims[2] > 1:
                s = "Spectral axis is "
                if sptype == "velo-radio": s += "VELOCITY (RADIO).\n"
                elif sptype == "velo-opt": s += "VELOCITY (OPTICAL).\n"
                elif sptype == "freq":
                    s += f"FREQUENCY.\n  Rest frequency  = {h.Freq0():.6f} {h.Cunit(2)}\n"
                    if p.getRedshift() != 0: s += f"  Redshift (z)    = {p.getRedshift():.6f}\n"
                elif sptype == "wave":
                    s += "WAVELENGTH.\n"
                    if h.Wave0() != -1:
                        s += f"  Rest wavelength = {h.Wave0():.6f} {h.Cunit(2)}\n"
                        if p.getRedshift() != 0: s += f"  Redshift (z)    = {p.getRedshift():.6f}\n"
                else:
                    s += "UNKNOWN.\n"
                s += f"  Channel width   = {h.Cdelt(2):.6f} {h.Cunit(2)} = {DeltaVel(h):.6f} km/s"
                s += f" ({h.VelDef()}).\n" if sptype in ("wave", "freq") else ".\n"
                print(s, end="")
            print(f"Flux units are {h.Bunit()}.")
            if isFluxUnitKnown(h):
                a, b = FluxtoJyBeam(1., h), FluxtoJy(1., h)
                f = (lambda v: f"{v:e}") if (a < 1E-05 or b < 1E-05) else (lambda v: f"{v:.6f}")
                print(f"  Conversion to Jy/beam: 1 {h.Bunit()} = {f(a)} Jy/beam.")
                print(f"  Conversion to Jy     : 1 {h.Bunit()} = {f(b)} Jy.")
            else:
                print("  No known conversion to Jy.")
            if h.BeamArea() == 0:
                print("Beam information has not been found in the header.\n")
            else:
                print(f"Beam size is {h.Bmaj() * 3600:.6f}\" x {h.Bmin() * 3600:.6f}\" (angle = {h.Bpa():.6f} deg).")
                print(f"Beam area is {h.BeamArea():.6f} pixels.\n")
        return True

    def fitswrite_3d(self, outfile, fullHead=False, dtype=None):
        arr = self.array if dtype is None else self.array.astype(dtype)
        if arr.dtype == np.bool_:
            arr = arr.astype(np.int16)
        if arr.ndim == 2:
            arr = arr[None]
        hdr = fits.Header()
        if self.headDefined:
            nd = 2 if self.head.NumAx() == 2 else 3
            hdr = self.head.toAstropy(nd, fullHead)
        hdu = fits.PrimaryHDU(data=arr, header=hdr)
        dname = os.path.dirname(outfile)
        if dname:
            os.makedirs(dname, exist_ok=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hdu.writeto(outfile, overwrite=True, output_verify="silentfix")
        return True

    # ------------------------------------------------------------------ stats
    def setCubeStats(self):
        p = self.par
        if p.verbose:
            print("Calculating statistics for the cube... ", end="", flush=True)
        a = self.array
        good = ~((a == 0) | np.isnan(a))
        self.stats.setRobust(p.flagRobustStats)
        self.stats.calculate(a, good)
        if p.se.iternoise:
            for _ in range(20):
                thresh = self.stats.getMiddle() + 3 * self.stats.getSpread()
                good &= ~(np.abs(a) > thresh)
                olds = self.stats.getSpread()
                self.stats.calculate(a, good)
                ftol = abs(olds - self.stats.getSpread()) / olds + self.stats.getSpread()
                if ftol < 0.001:
                    break
        self.stats.setThresholdSNR(p.se.snrCut)
        if p.verbose:
            print("Done.")
            print(self.stats)
        self.statsDefined = True

    # ------------------------------------------------------------------ search
    def search(self):
        from .search import Search, SortDetections
        from .smooth import Smooth3D
        h, par = self.head, self.par
        p = par.se
        if not self.statsDefined:
            self.setCubeStats()
        if par.verbose:
            thresh = p.threshold if p.UserThreshold else self.stats.getThreshold()
            f = (lambda v: f"{v:.5e}") if thresh < 1E-04 else (lambda v: f"{v:.5f}")
            print(f"Using flux threshold of: {f(thresh)} {h.Bunit()}")
            print(f"{'':25s}(middle = {f(self.stats.getMiddle())}, spread = {f(self.stats.getSpread())})")
        if p.minPix == -1:
            self.checkBeam()
        import copy
        sp = copy.deepcopy(p)
        ps = h.PixScale()
        sp.threshSpatial = p.threshSpatial if p.threshSpatial != -1 else int(math.ceil(h.Bmaj() / ps))
        sp.minPix = p.minPix if p.minPix != -1 else int(math.ceil(h.BeamArea()))
        sp.minVoxels = p.minVoxels if p.minVoxels != -1 else p.minChannels * sp.minPix
        sp.maxAngSize = p.maxAngSize / (h.PixScale() * arcsconv(h.Cunit(1)) / 60.)
        self.sources = Search(sp)
        if p.searchType == "spatialsmooth":
            sm = Smooth3D()
            arr = sm.cubesmooth(self)
            self.sources.search(arr, None, par.flagRobustStats, par.threads, par.verbose, par.showbar)
        else:
            self.sources.search(self.array, self.stats, par.flagRobustStats, par.threads, par.verbose, par.showbar)
        self.isSearched = True
        for o in self.sources.objectList:
            o.calcAllParams(self.array, h, p.pbcorr, par.fluxConvert)
            o.peakSNR = o.getPeakFlux() / self.stats.getSpread()
        self.sources.objectList = SortDetections(self.sources.objectList, p.sortsrcs)

    def printDetections(self, stream=None):
        st = stream if stream is not None else sys.stdout
        h, p, stats = self.head, self.par, self.stats
        m, k = 10, 29
        st.write("#\n")
        st.write(f"# Detections for {h.Name() if self.headDefined else p.imageFile}\n#\n")
        thresh1 = p.se.threshold if p.se.UserThreshold else stats.getThreshold()
        thresh2 = p.se.growthThreshold if p.se.growthThreshold != 0 else stats.getMiddle() + stats.getSpread() * p.se.growthCut
        fmt = (lambda v: f"{v:e}") if (thresh1 < 1E-03 or thresh2 < 1E-03) else (lambda v: f"{v:.6f}")
        st.write(f"# RMS Noise: {fmt(stats.getSpread())} {h.Bunit()}\n")
        if p.se.searchType != "spatialsmooth":
            st.write(f"# Primary threshold: {fmt(thresh1)} {h.Bunit()}\n# Secondary threshold: {fmt(thresh2)} {h.Bunit()}\n")
        st.write("#" + "_" * 149 + " \n#\n")
        line = "#" + f"{'  Source':<{m - 2}}" + f"{'Center    ':>{m + 6}}"
        if self.headDefined: line += f"{'Center (WCS)       ':>{k}}"
        line += (f"{'Xwidth':>{m - 3}}{'Ywidth':>{m - 3}}{'Zwidth':>{m - 3}}{'NumPix':>{m}}{'NumVox':>{m}}"
                 f"{'Vsys ':>{m}}{'W20  ':>{m}}{'Flux ':>{m + 5}}{'PeakSNR':>{m - 2}}{'PeakFlux':>{m}}")
        st.write(line + "\n")
        line = "#" + f"{'    [#]':<{m - 2}}" + f"{'[pix,pix,chan]':>{m + 7}}"
        if self.headDefined: line += f"{'[' + h.Cunit(0) + ',' + h.Cunit(1) + ',KM/S]     ':>{k}}"
        bu = "[" + h.Bunit() + "*km/s]"
        if p.fluxConvert and FluxtoJy(1, h) != 1: bu = "[Jy*km/s]"
        line += (f"{'[pix] ':>{m - 3}}{'[pix] ':>{m - 3}}{'[chan]':>{m - 4}}{'[pix] ':>{m}}{'[pix] ':>{m}}"
                 f"{'[km/s]':>{m}}{'[km/s]':>{m}}{bu:>{m + 5}}{' ':>{m - 2}}{'[' + h.Bunit() + ']':>{m}}")
        st.write(line + "\n")
        st.write("#" + "_" * (149 if self.headDefined else 103) + " \n#\n")
        for i in range(self.getNumObj()):
            o = self.pObject(i)
            xc, yc, zc = o.getXcentre(), o.getYcentre(), o.getZcentre()
            line = "      " + f"{i + 1:<{m - 7}}" + f"{xc:6.0f}{yc:5.0f}{zc:5.0f}"
            if self.headDefined:
                try:
                    w = h.pixToWCS(np.array([[xc, yc, zc]]))[0]
                    RA, DEC = w[0], w[1]
                except Exception:
                    RA = DEC = -1
                VEL = AlltoVel(self.getZphys(zc), h)
                if RA < 0: RA += 360
                elif RA > 360: RA -= 360
                line += f"{RA:11.3f}{DEC:9.3f}{VEL:8.1f}"
            line += (f"{abs(o.getXmax() - o.getXmin()) + 1:>{m - 3}d}{abs(o.getYmax() - o.getYmin()) + 1:>{m - 3}d}"
                     f"{abs(o.getZmax() - o.getZmin()) + 1:>{m - 3}d}{o.getSpatialSize():>{m}d}{o.getSize():>{m}d}")
            line += (f"{o.getVsys():>{m}.1f}{o.getW20():>{m}.1f}{o.getIntegFlux():>{m + 5}.3f}"
                     f"{o.getPeakFlux() / stats.getSpread():>{m - 2}.1f}{o.getPeakFlux():>{m}.3f}")
            st.write(line + "\n")
        st.write("\n")

    def detectionMask(self):
        m = np.zeros(self.array.shape, bool)
        for o in self.sources.objectList:
            m[o.z, o.y, o.x] = True
        return m

    def writeDetections(self):
        from .moments import getAllMoments
        from .image import Image2D
        if self.getNumObj() == 0:
            return
        p, h = self.par, self.head
        isObj = self.detectionMask()
        det = Cube.fromArray(np.where(isObj, self.array, 0).astype(self.array.dtype), h, p)
        det.Head().setMinMax(0, 0)
        det.fitswrite_3d(p.outFolder + p.outPrefix + "_detections.fits.gz")
        num = np.zeros(self.array.shape, np.int32)
        for i, o in enumerate(self.sources.objectList):
            num[o.z, o.y, o.x] = i + 1
        dn = Cube.fromArray(num, h, p)
        dn.Head().setMinMax(0, 0)
        dn.fitswrite_3d(p.outFolder + p.outPrefix + "_detections_num.fits.gz")
        dm = Image2D.fromArray(isObj.sum(axis=0).astype(np.int32), h)
        dm.Head().setMinMax(0, 0)
        dm.Head().setBtype("detected_chan")
        dm.fitswrite_2d(p.outFolder + p.outPrefix + "_detectmap.fits.gz")
        maps = getAllMoments(self, True, isObj, "MOMENT")
        maps[0].fitswrite_2d(p.outFolder + p.outPrefix + "_mom0th.fits")
        maps[1].fitswrite_2d(p.outFolder + p.outPrefix + "_mom1st.fits")
        maps[2].fitswrite_2d(p.outFolder + p.outPrefix + "_mom2nd.fits")
        if p.ma.SNmap:
            maps[0].SNMap(True)

    def extractCubelet(self, obj, edges):
        dims = self.AxisDim()
        omin = [obj.getXmin(), obj.getYmin(), obj.getZmin()]
        omax = [obj.getXmax(), obj.getYmax(), obj.getZmax()]
        starts, stops = [0] * 3, [0] * 3
        for i in range(3):
            starts[i] = max(omin[i] - edges, 0)
            stops[i] = min(omax[i] + edges, dims[i] - 1)
        starts[2] = max(omin[2] - 10, 0)
        stops[2] = min(omax[2] + 10, dims[2] - 1)
        sub = self.array[starts[2]:stops[2] + 1, starts[1]:stops[1] + 1, starts[0]:stops[0] + 1].copy()
        c = Cube.fromArray(sub, self.head, self.par)
        h = c.Head()
        pix = np.array([[round(obj.getXaverage()), round(obj.getYaverage()), round(starts[2])]], float)
        try:
            world = self.head.pixToWCS(pix)[0]
            world[2] = self.getZphys(starts[2])
            ok = True
        except Exception:
            ok = False
        for i in range(3):
            h.setDimAx(i, stops[i] - starts[i] + 1)
            if ok:
                h.setCrpix(i, pix[0, i] - starts[i] + 1)
                h.setCrval(i, world[i])
            else:
                h.setCrpix(i, self.head.Crpix(i) - starts[i])
        h.updateWCS()
        h.setMinMax(0, 0)
        h.keys = ["HISTORY BBAROLO SOURCE FINDER DATA PRODUCT"]
        return c, starts

    def writeCubelets(self):
        from .moments import getAllMoments
        n = self.getNumObj()
        if n == 0:
            return
        p = self.par
        outfold = p.outFolder + "sources/"
        obj = p.outPrefix if p.outPrefix != "" else "source"
        shutil.rmtree(outfold, ignore_errors=True)
        mkdirp(outfold)
        bar = ProgressBar(True, p.verbose, p.showbar)
        bar.init(" Writing cubelets... ", n)
        for i in range(n):
            bar.update(i + 1)
            src = obj
            if n > 1:
                nd = int(math.ceil(math.log10(n)))
                src += "_" + str(i + 1).zfill(nd)
            d = outfold + src + "/"
            mkdirp(d)
            o = self.pObject(i)
            o.calcWCSparams(self.head)
            c, st = self.extractCubelet(o, p.se.edges)
            c.Head().setName(src)
            c.fitswrite_3d(d + src + ".fits", True)
            isObj = np.zeros(c.array.shape, bool)
            isObj[o.z - st[2], o.y - st[1], o.x - st[0]] = True
            m = Cube.fromArray(isObj.astype(np.int16), c.Head(), c.pars())
            m.Head().setMinMax(0, 0)
            m.fitswrite_3d(d + src + "_mask.fits", True)
            c.pars().verbose = False
            maps = getAllMoments(c, True, isObj, "MOMENT")
            maps[0].fitswrite_2d(d + src + "_mom0.fits")
            maps[1].fitswrite_2d(d + src + "_mom1.fits")
            maps[2].fitswrite_2d(d + src + "_mom2.fits")
            if p.ma.SNmap:
                maps[0].SNMap(True, d + src)
            vel = AlltoVel(c.getZphys(np.arange(c.DimZ())), c.Head())
            spec = FluxtoJy(np.where(isObj, c.array, 0).astype(np.float64).sum(axis=(1, 2)), c.Head())
            with open(d + src + "_spectrum.dat", "w") as f:
                f.write("#Velocity(km/s)     Flux\n")
                for v, s in zip(np.atleast_1d(vel), np.atleast_1d(spec)):
                    f.write(f"{v:<18g}  {s:g}\n")
        bar.fillSpace("Done.\n")

    def plotDetections(self):
        from .plotscripts import write_sources_script, run_scripts
        if self.getNumObj() == 0:
            return 1
        script = write_sources_script(self)
        ret = 0
        if self.par.plots:
            if self.par.verbose:
                print(f" Making {randomAdjective(1)} plots for sources...", end="", flush=True)
            ret = run_scripts([script])
            if self.par.verbose:
                print(" Done." if ret == 0 else " Something went wrong! Check plot_sources.py in the output folder.")
        return ret

    # ------------------------------------------------------------------ masking
    def BlankCube(self, arr):
        if self.mask is None:
            self.BlankMask()
        return arr * self.mask

    def BlankMask(self, channel_noise=None, onlyLargest=True):
        """Build the mask (self.mask) according to the MASK parameter."""
        from .smooth import Smooth3D, SpectralSmooth3D
        p = self.par
        h = self.head
        MASK = p.MaskType
        mask = np.zeros(self.array.shape, bool)
        verb = p.verbose
        if verb:
            print(f" Creating mask ({MASK}) ...", end="", flush=True)
            p.verbose = False
        factor = nbmaj = nbmin = nbpa = 0.
        try:
            if "SMOOTH" in MASK and "SEARCH" in MASK:
                sm_c = Cube(par=p, head=h)
                if "SMOOTH&SEARCH" in MASK:
                    bmaj = h.Bmaj() * arcsconv(h.Cunit(0))
                    bmin = h.Bmin() * arcsconv(h.Cunit(0))
                    bpa = h.Bpa()
                    factor = 2 if p.factor == -1 else p.factor
                    nbmaj = factor * bmaj if p.bmaj == -1 else p.bmaj
                    nbmin = factor * bmin if p.bmin == -1 else p.bmin
                    nbpa = bpa if p.bpa == -1 else p.bpa
                    sm = Smooth3D()
                    arr = sm.smooth(self, (bmaj, bmin, bpa), (nbmaj, nbmin, nbpa))
                    sm_c.array = np.nan_to_num(arr).astype(self.array.dtype)
                    sm_c.Head().setBmaj(nbmaj / 3600.)
                    sm_c.Head().setBmin(nbmin / 3600.)
                    sm_c.Head().setBpa(nbpa)
                    sm_c.Head().calcArea()
                else:   # SMOOTHSPEC&SEARCH
                    sm = SpectralSmooth3D(p.window_type, p.window_size)
                    sm_c.array = sm.smooth(self, verbose=False)
                sm_c.setCubeStats()
                sm_c.search()
                n = sm_c.getNumObj()
                if n == 0:
                    raise BBaroloError("MASKING error: No sources detected in the datacube. Cannot build mask!!! ")
                objs = [sm_c.getSources().LargestDetection()] if (onlyLargest or n == 1) else sm_c.getSources().objectList
                for o in objs:
                    mask[o.z, o.y, o.x] = True
                self._maskSources = sm_c.getSources()
            elif "SEARCH" in MASK:
                if not self.isSearched:
                    self.search()
                n = self.getNumObj()
                if n == 0:
                    raise BBaroloError("MASKING error: No sources detected in the datacube. Cannot build mask!!! ")
                objs = [self.sources.LargestDetection()] if (onlyLargest or n == 1) else self.sources.objectList
                for o in objs:
                    mask[o.z, o.y, o.x] = True
                self._maskSources = self.sources
            elif "SMOOTH" in MASK:
                bmaj = h.Bmaj() * arcsconv(h.Cunit(0))
                bmin = h.Bmin() * arcsconv(h.Cunit(0))
                bpa = h.Bpa()
                factor = 2 if p.factor == -1 else p.factor
                nbmaj = factor * bmaj if p.bmaj == -1 else p.bmaj
                nbmin = factor * bmin if p.bmin == -1 else p.bmin
                nbpa = bpa if p.bpa == -1 else p.bpa
                sm = Smooth3D()
                arr = sm.smooth(self, (bmaj, bmin, bpa), (nbmaj, nbmin, nbpa))
                st = Stats(p.flagRobustStats)
                good = ~((arr == 0) | np.isnan(arr))
                st.calculate(arr, good)
                st.setThresholdSNR(p.blankCut)
                with np.errstate(invalid="ignore"):
                    mask = arr > st.getThreshold()
            elif "THRESHOLD" in MASK:
                mask = self.array > p.se.threshold
            elif "NEGATIVE" in MASK:
                st = Stats(p.flagRobustStats)
                for z in range(self.DimZ()):
                    ch = self.array[z]
                    neg = ch[ch < 0]
                    onlyneg = np.concatenate([neg, -neg])
                    if onlyneg.size == 0:
                        continue
                    st.calculate(onlyneg)
                    st.setThresholdSNR(p.blankCut)
                    mask[z] = ch > st.getThreshold()
                    if channel_noise is not None:
                        channel_noise[z] = st.getSpread()
            elif "FILE(" in MASK:
                first, last = MASK.find("("), MASK.rfind(")")
                if first < 0 or last < 0:
                    raise ValueError("\n  ERROR: MASK parameter is not correct. Provide file(Maskfitsfile)")
                fname = MASK[first + 1:last]
                if not os.path.isfile(fname):
                    raise IOError(f"\n ERROR: Mask {fname} is not a readable FITS image! Exiting ...")
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    md = np.asarray(fits.getdata(fname))
                md = np.squeeze(md)
                if md.size != self.array.size:
                    raise ValueError(f"\n ERROR: Mask file and data file have different dimensions.{fname} Exiting ...")
                mask = np.nan_to_num(md.reshape(self.array.shape)) != 0
            elif MASK == "NONE":
                mask[:] = True

            found = MASK.find("ENLARGE")
            if found >= 0:
                Ns = MASK[found + 7:]
                N = int(Ns) if Ns.strip() else 3
                from scipy.ndimage import binary_dilation
                st2 = np.zeros((1, 2 * N + 1, 2 * N + 1), bool)
                st2[0] = True
                mask = binary_dilation(mask, structure=st2)
        finally:
            p.verbose = verb                     # (the level, VERBOSE 1 or 2)

        # Writing mask to FITS file
        mh = h.copy()
        mh.setMinMax(0, 0)
        sh = "HISTORY BBAROLO MASKING: "
        s = [sh + "mask built with MASK=" + MASK, sh + "FITSFILE=" + p.imageFile]
        if "SMOOTH" in MASK:
            if "SPEC" in MASK:
                s += [sh + "WINDOW_TYPE=" + p.window_type, sh + f"WINDOW_SIZE={p.window_size:.6f}"]
            else:
                s += [sh + f"FACTOR={factor:.6f}", sh + f"NEWBMAJ={nbmaj:.6f}", sh + f"NEWBMIN={nbmin:.6f}",
                      sh + f"NEWBPA={nbpa:.6f}"]
                if MASK == "SMOOTH":
                    s.append(sh + f"BLANKCUT={p.blankCut:.6f}")
        if "SEARCH" in MASK:
            se = p.se
            s.append(sh + "SEARCHTYPE=" + se.searchType)
            s.append(sh + (f"THRESHOLD={se.threshold:.6f}" if se.threshold != 0 else f"SNRCUT={se.snrCut:.6f}"))
            s.append(sh + "FLAGGROWTH=" + stringize(se.flagGrowth))
            if se.flagGrowth:
                s.append(sh + (f"GROWTHTHRESHOLD={se.threshold:.6f}" if se.growthThreshold != 0 else f"GROWTHCUT={se.growthCut:.6f}"))
            s.append(sh + "REJECTBEFOREMERGE=" + stringize(se.RejectBeforeMerge))
            s.append(sh + "TWOSTAGEMERGING=" + stringize(se.TwoStageMerging))
            s.append(sh + f"THRESHVELOCITY={se.threshVelocity}")
            s.append(sh + ("THRESHSPATIAL=1" if se.flagAdjacent else f"THRESHSPATIAL={se.threshSpatial}"))
            s.append(sh + f"MINCHANNELS={se.minChannels}")
            s.append(sh + ("MINPIX=beam_area" if se.minPix == -1 else f"MINPIX={se.minPix}"))
            s.append(sh + ("MINVOXELS=MINPIX*MINCHANNELS" if se.minVoxels == -1 else f"MINVOXELS={se.minVoxels}"))
            if se.maxChannels != -1: s.append(sh + f"MAXCHANNELS={se.maxChannels}")
            if se.maxAngSize != -1: s.append(sh + f"MAXANGSIZE={se.maxAngSize:.6f}")
        if MASK == "THRESHOLD":
            s.append(sh + f"THRESHOLD={p.se.threshold:.6f}")
        mh.keys = s
        if p.plots >= 0:
            mc = Cube(par=p, head=mh)
            mc.array = mask.astype(np.int16)
            mc.fitswrite_3d(p.outFolder + p.outPrefix + "_mask.fits", True)
        if verb:
            print(" Done.")
        self.mask = mask
        return mask

    # ------------------------------------------------------------------ misc
    def Reduce(self, fac, rtype="spatial"):
        p = self.par
        if p.verbose:
            print(" Reducing...", end="", flush=True)
        a = self.array.astype(np.float64)
        nz, ny, nx = a.shape
        h = self.head.copy()
        if rtype == "spectral":
            nzn = nz // fac
            red = a[:nzn * fac].reshape(nzn, fac, ny, nx)
            s = red.sum(axis=1)
            out = s / float(fac)
            out[s == 0] = np.nan
            h.setCdelt(2, fac * self.head.Cdelt(2))
            h.setCrpix(2, 0.5 + (self.head.Crpix(2) - 0.5) / float(fac))
            o = f"  Old channel width: {self.head.Cdelt(2):.6f} {self.head.Cunit(2)}"
            nn = f"  New channel width: {h.Cdelt(2):.6f} {self.head.Cunit(2)}"
            h.addKey("HISTORY BBAROLO SPECTRAL AVERAGING: " + nn)
            h.addKey("HISTORY BBAROLO SPECTRAL AVERAGING: " + o)
        else:
            nyn, nxn = ny // fac, nx // fac
            red = a[:, :nyn * fac, :nxn * fac].reshape(nz, nyn, fac, nxn, fac)
            s = red.sum(axis=(2, 4))
            out = s / float(fac * fac)
            out[s == 0] = np.nan
            h.setCdelt(0, fac * self.head.Cdelt(0))
            h.setCdelt(1, fac * self.head.Cdelt(1))
            # pixel k of the reduced grid covers pixels (k-1)*fac+1 ... k*fac of the original one
            h.setCrpix(0, 0.5 + (self.head.Crpix(0) - 0.5) / float(fac))
            h.setCrpix(1, 0.5 + (self.head.Crpix(1) - 0.5) / float(fac))
            h.calcArea()
            o = f"  Old beam size: {self.head.BeamArea():.6f} pixels"
            nn = f"  New beam size: {h.BeamArea():.6f} pixels"
            h.addKey("HISTORY BBAROLO SPATIAL AVERAGING: " + nn)
            h.addKey("HISTORY BBAROLO SPATIAL AVERAGING: " + o)
        if p.verbose:
            print("OK\n" + o + "\n" + nn)
        r = Cube.fromArray(out.astype(self.array.dtype), h, p)
        if np.any(np.isfinite(out)):            # (an empty cube is all NaN: no DATAMAX/DATAMIN)
            r.Head().setDataMax(np.nanmax(out))
            r.Head().setDataMin(np.nanmin(out))
        return r

    def CheckCube(self, type=1):
        if type < 1 or type > 5:
            raise ValueError("CheckCube() ERROR: acceptable 'type' values are 1-5")
        if not self.statsDefined:
            self.setCubeStats()
        a = self.array
        nz, ny, nx = a.shape
        p = self.par
        madfm_all = self.stats.getMadfm()
        bad_x, bad_y, bad_z = [], [], []

        def mad(v):
            v = v[~((v == 0) | np.isnan(v))]
            if v.size == 0:
                return None, 0
            return findMADFM(v), v.size

        print("\n")
        if type in (1, 2):
            for z in range(nz):
                m, n = mad(a[z].ravel())
                if m is None:
                    bad_z.append(z); continue
                if m > 1.5 * madfm_all:
                    bad_z.append(z); a[z] = 0
            print(" All channels are good.\n" if not bad_z else
                  " Following channels have been erased: " + " ".join(str(z + 1) for z in bad_z) + "\n")
        if type in (1, 3, 4):
            for y in range(ny):
                m, n = mad(a[:, y, :].ravel())
                if m is None:
                    bad_y.append(y); continue
                if m > 1.5 * madfm_all and n / nz > 0.7 * ny:
                    bad_y.append(y); a[:, y, :] = 0
            print(" All pixel columns are good.\n" if not bad_y else
                  " Following pixel columns have been erased: " + " ".join(str(y + 1) for y in bad_y) + "\n")
        if type in (1, 3, 5):
            for x in range(nx):
                m, n = mad(a[:, :, x].ravel())
                if m is None:
                    bad_x.append(x); continue
                if m > 1.5 * madfm_all and n / nz > 0.7 * nx:
                    bad_x.append(x); a[:, :, x] = 0
            print(" All pixel rows are good.\n" if not bad_x else
                  " Following pixel rows have been erased: " + " ".join(str(x + 1) for x in bad_x) + "\n")
        if not (bad_x or bad_y or bad_z):
            return

        def edges(bad, dim):
            start = 0
            while start in bad: start += 1
            end = 0
            while (dim - 1 - end) in bad and end < dim: end += 1
            return start, end
        cx0, cx1 = edges(bad_x, nx)
        cy0, cy1 = edges(bad_y, ny)
        cz0, cz1 = edges(bad_z, nz)
        oldv = p.verbose
        if oldv:
            print(" Re-calculating statistics for cleaned cube ...", end="")
        p.verbose = False
        self.setCubeStats()
        p.verbose = oldv
        if oldv:
            print("\n" + str(self.stats))
        name = p.imageFile
        f = name.lower().find(".fits")
        name = name[:f] + "_clean" + name[f:]
        sub = a[cz0:nz - cz1, cy0:ny - cy1, cx0:nx - cx1].copy()
        sub[(sub == 0)] = np.nan
        out = Cube.fromArray(sub, self.head, p)
        h = out.Head()
        h.setMinMax(0, 0)
        h.setCrpix(0, self.head.Crpix(0) - cx0)
        h.setCrpix(1, self.head.Crpix(1) - cy0)
        h.setCrpix(2, self.head.Crpix(2) - cz0)
        out.fitswrite_3d(name, True)

    def continuumSubtract(self):
        """Fit and subtract a polynomial continuum from each spectrum (vectorised)."""
        p = self.par
        nz = self.DimZ()
        toex = np.zeros(nz, bool)
        for w in str(p.exclude_windows).split():
            kv = None
            for d in (":", "~"):
                if d in w:
                    kv = w.split(d, 1)
                    break
            if kv is not None:
                if not (kv[0].isdigit() and kv[1].isdigit()):
                    continue
                a, b = sorted((int(kv[0]), int(kv[1])))
                toex[a:b + 1] = True
            elif w.isdigit():
                toex[int(w)] = True
        z = np.arange(nz, dtype=np.float64)
        ncoeff = p.cont_order + 1
        V = np.vander(z, ncoeff, increasing=True)
        A = V[~toex]
        a = self.array.astype(np.float64)
        Y = a[~toex].reshape(A.shape[0], -1)
        coeff, *_ = np.linalg.lstsq(A, Y, rcond=None)
        cont = (V @ coeff).reshape(a.shape)
        self.array = (a - cont).astype(self.array.dtype)
