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

from .header import Header, read_fits_image
from .params import Param, stringize
from .stats import Stats, findMADFM
from .bbutils import (BBaroloError, arcsconv, DeltaVel, FluxtoJy, FluxtoJyBeam, isFluxUnitKnown, AlltoVel,
                      ProgressBar, mkdirp, randomAdjective, shortpath)



def write_outputs(c, outputs):
    """Write the outputs {suffix: text, Figure or HDU} (see Cube.add_output) in the output folder of the cube c.
    Returns the files written."""
    p = c.pars()
    files = []
    for suffix, v in outputs.items():
        fn = p.outFolder + p.outPrefix + suffix
        if isinstance(v, str):
            mkdirp(os.path.dirname(fn) or ".")
            with open(fn, "w") as f:
                f.write(v)
        elif hasattr(v, "savefig"):
            mkdirp(os.path.dirname(fn) or ".")
            v.savefig(fn, bbox_inches="tight")
        else:
            write_hdu(v, fn)
        files.append(fn)
    return files


def write_hdu(hdu, outfile):
    """Write an HDU to a FITS file (overwriting it, creating its folder), as BBarolo writes its outputs."""
    dname = os.path.dirname(outfile)
    if dname:
        os.makedirs(dname, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        hdu.writeto(outfile, overwrite=True, output_verify="silentfix")



def exclwind_channels(exclwind, nz, what="the cube"):
    """The channels excluded from the continuum fit by EXCLWIND (single channels or ranges first:last or first~last,
    counted from 0, separated by spaces or commas; NONE: none), as a bool array (nz,). Raises BBaroloError for a wrong
    item, a channel beyond nz or all the channels excluded."""
    import re
    toex = np.zeros(nz, bool)
    ex = str(exclwind).strip()
    if ex.upper() in ("NONE", ""):
        return toex
    for w in ex.replace(",", " ").split():
        m = re.fullmatch(r"(\d+)(?:[:~](\d+))?", w)
        if m is None:
            raise BBaroloError(f"EXCLWIND {ex}: {w} is not a channel or a range of channels (first:last), "
                               "counted from 0.")
        a, b = sorted((int(m.group(1)), int(m.group(2) or m.group(1))))
        if b >= nz:
            raise BBaroloError(f"EXCLWIND {ex}: channel {b} is outside {what} (channels 0 to {nz - 1}).")
        toex[a:b + 1] = True
    if toex.all():
        raise BBaroloError(f"EXCLWIND {ex}: all the channels are excluded, no continuum to fit.")
    return toex


class Cube:

    def __init__(self, fname=None, par=None, printInfo=False, array=None, head=None):
        self.par = par.copy() if par is not None else Param()
        self.head = head.copy() if head is not None else Header()
        self.headDefined = head is not None
        self.array = None
        self.mask = None
        self.mask_header = None              # (the header of the mask built by BlankMask)
        self.mask_written = False
        self.extra_outputs = {}              # (outputs made while computing, written by the task: see add_output)
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
        c.mask_header, c.mask_written = self.mask_header, self.mask_written
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
    def applyBeam(self):
        """BEAM (par.beam), if given, replaces the beam of the header (BMAJ, BMIN in arcsec, BPA in degrees,
        0 if not given). Called once, when the cube is read."""
        h, b = self.head, getattr(self.par, "beam", None)
        if b is None:
            return
        bmaj, bmin, bpa = b[0] / 3600., b[1] / 3600., (0. if b[2] is None else float(b[2]))
        pix = h.PixScale()
        if pix and min(bmaj, bmin) < 0.25 * pix:
            raise BBaroloError(f"BEAM ({b[0]:.4g}x{b[1]:.4g} arcsec) is smaller than a quarter of a pixel "
                               f"({pix * 3600.:.4g} arcsec): BMAJ and BMIN are in arcsec, or in degrees with a final "
                               "d (e.g. BEAM 0.01d 0.01d 30).")
        if h.BeamArea() > 0 and self.par.verbose and (abs(h.Bmaj() - bmaj) > 1e-6 * bmaj
                                                      or abs(h.Bmin() - bmin) > 1e-6 * bmin
                                                      or abs(h.Bpa() - bpa) > 1e-6):
            print(f"\n BEAM: using a {b[0]:.5g}x{b[1]:.5g} arcsec beam (BPA={bpa:.5g}) instead of that of the header "
                  f"({h.Bmaj() * 3600.:.5g}x{h.Bmin() * 3600.:.5g} arcsec, BPA={h.Bpa():.5g}).")
        h.setBmaj(bmaj)
        h.setBmin(bmin)
        h.setBpa(bpa)
        h.calcArea()

    def checkBeam(self):
        h = self.head
        if h.BeamArea() == 0:
            raise BBaroloError("No beam information available for the data: set it with BEAM (BMAJ BMIN BPA, "
                               "e.g. BEAM 30 20 45; BMAJ and BMIN in arcsec, or in degrees with a final d; one "
                               "value is a circular beam), or beam=(bmaj, bmin, bpa) in Python.")
        elif min(h.Bmaj(), h.Bmin()) < h.PixScale() and not getattr(self, "_beamwarned", False):
            self._beamwarned = True
            print(f"\nWARNING: the beam ({h.Bmaj() * 3600.:.4g}x{h.Bmin() * 3600.:.4g} arcsec) is smaller than a "
                  f"pixel ({h.PixScale() * 3600.:.4g} arcsec).\n")

    def readCube(self, fname, printInfo=True):
        p = self.par
        p.imageFile = fname
        h = self.head
        h.warning = p.verbose
        if not h.header_read(fname):
            return False

        return self._readHeaderData(fname, lambda: read_fits_image(fname)[1], printInfo)

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
        self.applyBeam()
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
        nan = np.isnan(data)
        # (the blank pixels of a 2D map, for ELLPROF; the data are 0 there, as the blanks of cubes)
        self.blanks2d = nan[0].copy() if dims[2] == 1 and nan.any() else None
        data[nan] = 0
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

    def to_hdu(self, fullHead=False, dtype=None, copy=False):
        """The cube as an astropy PrimaryHDU: the data and header that fitswrite_3d writes (fullHead: all the
        keywords of the original header; dtype: the data type, default that of the array; copy: a copy of the
        data, not a view of the array of the cube)."""
        arr = self.array if dtype is None else self.array.astype(dtype)
        if copy and arr is self.array:
            arr = arr.copy()
        if arr.dtype == np.bool_:
            arr = arr.astype(np.int16)
        if arr.ndim == 2:
            arr = arr[None]
        hdr = fits.Header()
        if self.headDefined:
            nd = 2 if self.head.NumAx() == 2 else 3
            hdr = self.head.toAstropy(nd, fullHead)
        hdu = fits.PrimaryHDU(data=arr, header=hdr)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hdu.verify("silentfix")
        return hdu

    def fitswrite_3d(self, outfile, fullHead=False, dtype=None):
        write_hdu(self.to_hdu(fullHead, dtype), outfile)
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
                ftol = abs(olds - self.stats.getSpread()) / olds
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
        if p.searchType == "spatialsmooth":
            # the cube smoothed with the parameters of SMOOTH, and its statistics (C++ printed the threshold
            # of the original cube)
            arr = Smooth3D().cubesmooth(self)
            stats = Stats(par.flagRobustStats)
            stats.calculate(arr, ~((arr == 0) | np.isnan(arr)))
            stats.setThresholdSNR(p.snrCut)
        else:
            arr, stats = self.array, self.stats
        if par.verbose:
            thresh = p.threshold if p.UserThreshold else stats.getThreshold()
            f = (lambda v: f"{v:.5e}") if thresh < 1E-04 else (lambda v: f"{v:.5f}")
            print(f"Using flux threshold of: {f(thresh)} {h.Bunit()}")
            print(f"{'':25s}(middle = {f(stats.getMiddle())}, spread = {f(stats.getSpread())})")
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
        self.sources.search(arr, stats, par.flagRobustStats, par.threads, par.verbose, par.showbar)
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

    def detectionProducts(self):
        """The products of the source finder as HDUs, {file suffix: HDU} in the order they are written: the
        cube of the detections (_detections.fits.gz), the cube of their numbers (_detections_num.fits.gz),
        the number of channels detected in each pixel (_detectmap.fits.gz), the moment maps of the detections
        (_mom0th.fits, _mom1st.fits, _mom2nd.fits) and, with SNMAP, the noise and S/N maps of the total map
        (_mom0th_RMS.fits, _mom0th_SN.fits). Empty without detections."""
        from .moments import getAllMoments
        from .image import Image2D
        if self.getNumObj() == 0:
            return {}
        p, h = self.par, self.head
        out = {}
        isObj = self.detectionMask()
        det = Cube.fromArray(np.where(isObj, self.array, 0).astype(self.array.dtype), h, p)
        det.Head().setMinMax(0, 0)
        out["_detections.fits.gz"] = det.to_hdu()
        num = np.zeros(self.array.shape, np.int32)
        for i, o in enumerate(self.sources.objectList):
            num[o.z, o.y, o.x] = i + 1
        dn = Cube.fromArray(num, h, p)
        dn.Head().setMinMax(0, 0)
        out["_detections_num.fits.gz"] = dn.to_hdu()
        dm = Image2D.fromArray(isObj.sum(axis=0).astype(np.int32), h)
        dm.Head().setMinMax(0, 0)
        dm.Head().setBtype("detected_chan")
        out["_detectmap.fits.gz"] = dm.to_hdu()
        maps = getAllMoments(self, True, isObj, "MOMENT")
        for i, s in enumerate(("_mom0th.fits", "_mom1st.fits", "_mom2nd.fits")):
            out[s] = maps[i].to_hdu()
        if p.ma.SNmap:
            n_im, s_im = maps[0].SNMaps(True)         # (TOTALMAP writes _map_0th_RMS.fits, _map_0th_SN.fits)
            out["_mom0th_RMS.fits"], out["_mom0th_SN.fits"] = n_im.to_hdu(), s_im.to_hdu()
        return out

    def writeDetections(self, products=None):
        """Write the products of the source finder (products: those of detectionProducts(), if computed)."""
        p = self.par
        for suffix, hdu in (products if products is not None else self.detectionProducts()).items():
            write_hdu(hdu, p.outFolder + p.outPrefix + suffix)

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

    def _maskBeam(self, bmaj, bmin, bpa):
        """FACTOR and the new beam (arcsec, deg) of the SMOOTH masks, from the beam of the data: NEWBEAM if
        given (its BPA, if not given, that of the data), otherwise FACTOR (default 2) times the beam, at the
        same angle."""
        p = self.par
        factor = 2 if p.factor == -1 else p.factor
        nb = getattr(p, "newbeam", None)
        if nb is not None:
            return factor, nb[0], nb[1], (bpa if nb[2] is None else nb[2])
        return factor, factor * bmaj, factor * bmin, bpa

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
                    factor, nbmaj, nbmin, nbpa = self._maskBeam(bmaj, bmin, bpa)
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
                factor, nbmaj, nbmin, nbpa = self._maskBeam(bmaj, bmin, bpa)
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
                    raise BBaroloError("\nERROR: MASK parameter is not correct. Provide file(Maskfitsfile)")
                fname = MASK[first + 1:last]
                try:
                    md = read_fits_image(fname)[1]
                except Exception:
                    md = None
                if md is None:
                    raise BBaroloError(f"\nERROR: Mask {fname} is not a readable FITS image! Exiting ...")
                md = np.squeeze(md)
                if md.size != self.array.size:
                    raise BBaroloError(f"\nERROR: Mask file and data file have different dimensions.{fname} "
                                       "Exiting ...")
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
                s.append(sh + (f"GROWTHTHRESHOLD={se.growthThreshold:.6f}" if se.growthThreshold != 0
                               else f"GROWTHCUT={se.growthCut:.6f}"))
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
        if verb:
            print(" Done.")
        self.mask = mask
        self.mask_header = mh
        self.mask_written = False
        return mask

    def add_output(self, suffix, value):
        """Keep an output made while computing (e.g. the table of the 2D fit of the initial estimates), to be
        written by the task as <outfolder><prefix><suffix>: a string (text file), a matplotlib Figure (saved)
        or an astropy HDU."""
        self.extra_outputs[suffix] = value

    def pop_outputs(self):
        """The outputs kept by add_output since the last call ({suffix: value})."""
        out, self.extra_outputs = self.extra_outputs, {}
        return out

    def mask_hdu(self):
        """The mask built by BlankMask as an HDU (int16, 1 inside), with the header of <prefix>_mask.fits;
        None if there is no mask."""
        if self.mask is None or getattr(self, "mask_header", None) is None:
            return None
        mc = Cube(par=self.par, head=self.mask_header)
        mc.array = self.mask.astype(np.int16)
        return mc.to_hdu(True)

    def writeMask(self, force=False):
        """Write <prefix>_mask.fits in the output folder (once for each mask built, unless force)."""
        if getattr(self, "mask_header", None) is None or (self.mask_written and not force):
            return None
        p = self.par
        fn = p.outFolder + p.outPrefix + "_mask.fits"
        write_hdu(self.mask_hdu(), fn)
        self.mask_written = True
        return fn

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
        """Erase (set to 0) the noisy channels (type 1, 2), pixel columns (1, 3, 4) and rows (1, 3, 5) of the
        cube, in place. Returns the cleaned cube cropped of the erased edges, with the blanks NaN (written as
        <prefix>_clean.fits by the CHECKCUBE task), or None if nothing was erased."""
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
                if m > 1.5 * madfm_all and n / nz > 0.7 * nx:         # (a line of nx pixels; C++ had ny)
                    bad_y.append(y); a[:, y, :] = 0
            print(" All pixel columns are good.\n" if not bad_y else
                  " Following pixel columns have been erased: " + " ".join(str(y + 1) for y in bad_y) + "\n")
        if type in (1, 3, 5):
            for x in range(nx):
                m, n = mad(a[:, :, x].ravel())
                if m is None:
                    bad_x.append(x); continue
                if m > 1.5 * madfm_all and n / nz > 0.7 * ny:         # (a line of ny pixels; C++ had nx)
                    bad_x.append(x); a[:, :, x] = 0
            print(" All pixel rows are good.\n" if not bad_x else
                  " Following pixel rows have been erased: " + " ".join(str(x + 1) for x in bad_x) + "\n")
        if not (bad_x or bad_y or bad_z):
            return None

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
        sub = a[cz0:nz - cz1, cy0:ny - cy1, cx0:nx - cx1].copy()
        sub[(sub == 0)] = np.nan
        out = Cube.fromArray(sub, self.head, p)
        h = out.Head()
        h.setMinMax(0, 0)
        h.setCrpix(0, self.head.Crpix(0) - cx0)
        h.setCrpix(1, self.head.Crpix(1) - cy0)
        h.setCrpix(2, self.head.Crpix(2) - cz0)
        return out

    def continuumSubtract(self):
        """Fit and subtract a polynomial continuum from each spectrum (vectorised)."""
        p = self.par
        nz = self.DimZ()
        toex = exclwind_channels(p.exclude_windows, nz)
        z = np.arange(nz, dtype=np.float64)
        ncoeff = p.cont_order + 1
        V = np.vander(z, ncoeff, increasing=True)
        A = V[~toex]
        a = np.nan_to_num(self.array.astype(np.float64))    # (blanks are 0, as in the cubes read from FITS)
        Y = a[~toex].reshape(A.shape[0], -1)
        coeff, *_ = np.linalg.lstsq(A, Y, rcond=None)
        cont = (V @ coeff).reshape(a.shape)
        self.array = (a - cont).astype(self.array.dtype)
