"""
FITS header handling (port of Arrays/header.hh, header.cpp).

The :class:`Header` collects the information BBarolo needs from a FITS header
and wraps an ``astropy.wcs.WCS`` object (the C++ version uses WCSLIB directly).
Axis indices follow the FITS/C++ convention: 0=x (RA), 1=y (DEC), 2=spectral.
"""

import copy
import math
import warnings
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS



def _lower(s):
    return str(s).lower()


class Header:

    def __init__(self, fname=None, warning=True):
        self.bitpix = -32
        self.numAxes = 0
        self.dimAxes = []
        self.crpix, self.crval, self.cdelt = [], [], []
        self.ctype, self.cunit = [], []
        self.bmaj = self.bmin = self.bpa = 0.
        self.beamArea = 0.
        self.freq0 = 0.
        self.wave0 = -1.
        self.datamin = self.datamax = 0.
        self.redshift = 0.
        self.crota = 0.
        self.drval3 = 0.
        self.dunit3 = ""
        self.radesys = ""
        self.specsys = ""
        self.object = "NONE"
        self.veldef = "auto"
        self.bunit = "NONE"
        self.btype = "NONE"
        self.telescope = "NONE"
        self.sptype = "unknown"
        self.bzero, self.bscale, self.blank, self.epoch = 0., 1., 0., 0.
        self.fitsname = ""
        self.specUnits = ""
        self.specConv = 1.
        self.keys = []                  # list of header cards (strings)
        self.wcs = None
        self.wcsIsGood = False
        self.warning = warning
        if fname is not None:
            if not self.header_read(fname):
                raise IOError(f"Cannot read header of {fname}")

    # ------------------------------------------------------------------ getters
    def NumAx(self): return self.numAxes
    def DimAx(self, i): return self.dimAxes[i]
    def Crpix(self, i): return self.crpix[i]
    def Crval(self, i): return self.crval[i]
    def Cdelt(self, i): return self.cdelt[i]
    def Ctype(self, i): return self.ctype[i]
    def Cunit(self, i): return self.cunit[i]
    def Bmaj(self): return self.bmaj
    def Bmin(self): return self.bmin
    def Bpa(self): return self.bpa
    def BeamArea(self): return self.beamArea
    def Freq0(self): return self.freq0
    def Wave0(self): return self.wave0
    def Redshift(self): return self.redshift
    def VelDef(self): return self.veldef
    def Bunit(self): return self.bunit
    def Btype(self): return self.btype
    def Name(self): return self.object
    def Telesc(self): return self.telescope
    def SpectralType(self): return self.sptype
    def Crota(self): return self.crota
    def Keys(self): return self.keys
    def PixScale(self): return (abs(self.cdelt[0]) + abs(self.cdelt[1])) / 2.
    def isWCS(self): return self.wcsIsGood

    # ------------------------------------------------------------------ setters
    def setCrpix(self, i, v): self.crpix[i] = float(v)
    def setCrval(self, i, v): self.crval[i] = float(v)
    def setCdelt(self, i, v): self.cdelt[i] = float(v)
    def setCtype(self, i, v): self.ctype[i] = v
    def setCunit(self, i, v): self.cunit[i] = v
    def setDimAx(self, i, v): self.dimAxes[i] = int(v)
    def setBmaj(self, v): self.bmaj = float(v)
    def setBmin(self, v): self.bmin = float(v)
    def setBpa(self, v): self.bpa = float(v)
    def setBunit(self, v): self.bunit = v
    def setBtype(self, v): self.btype = v
    def setName(self, v): self.object = v
    def setFreq0(self, v): self.freq0 = float(v)
    def setWave0(self, v): self.wave0 = float(v)
    def setRedshift(self, v): self.redshift = float(v)
    def setVelDef(self, v): self.veldef = v
    def setMinMax(self, mn, mx): self.datamin, self.datamax = float(mn), float(mx)
    def setDataMin(self, v): self.datamin = float(v)
    def setDataMax(self, v): self.datamax = float(v)
    def addKey(self, s): self.keys.append(s)

    def setBeam(self, a, b, c):
        self.bmaj, self.bmin, self.bpa = a, b, c
        self.calcArea()

    def setNumAx(self, n):
        self.numAxes = n
        self.dimAxes = [1] * n
        self.crpix, self.crval, self.cdelt = [0.] * n, [0.] * n, [0.] * n
        self.ctype, self.cunit = [""] * n, [""] * n

    def copy(self):
        h = copy.copy(self)
        for k in ("dimAxes", "crpix", "crval", "cdelt", "ctype", "cunit", "keys"):
            setattr(h, k, list(getattr(self, k)))
        h.wcs = self.wcs.deepcopy() if self.wcs is not None else None
        return h

    def calcArea(self):
        avpix = (abs(self.cdelt[0]) + abs(self.cdelt[1])) / 2.
        if avpix == 0:
            self.beamArea = 0.
            return
        self.beamArea = math.pi * (self.bmaj / avpix / 2.) * (self.bmin / avpix / 2.) / math.log(2)

    def Warning(self, s):
        if self.warning:
            print(s)

    # ------------------------------------------------------------------ reading
    def header_read(self, fname):
        self.fitsname = fname
        try:
            with fits.open(fname) as hdul:
                hdr = hdul[0].header.copy()
        except Exception as e:
            print(e)
            return False
        return self.fromAstropy(hdr)

    def fromAstropy(self, hdr):
        """Fill the Header from an astropy Header (port of Header::header_read)."""
        g = hdr.get
        self.bitpix = int(g("BITPIX", -32))
        n = int(g("NAXIS", 0))
        self.setNumAx(n)
        self.dimAxes = [int(g(f"NAXIS{i + 1}", 1)) for i in range(n)]
        self.keys = [c.image for c in hdr.cards if c.keyword not in ("", "END")]

        cdelt = [g(f"CDELT{i + 1}") for i in range(n)]
        if all(c is None for c in cdelt):
            self.Warning("HEADER WARNING: CDELTs keywords not found.")
            for i in range(n):
                v = g(f"CD{i + 1}_{i + 1}")
                if v is None:
                    self.Warning("HEADER ERROR: No pixel spacing keywords found.")
                    return False
                cdelt[i] = v
            self.Warning("HEADER WARNING: Found CDn_n keywords instead!")
        self.cdelt = [float(c if c is not None else (g(f"CD{i + 1}_{i + 1}") or 1.)) for i, c in enumerate(cdelt)]
        crpix = [g(f"CRPIX{i + 1}") for i in range(n)]
        if all(c is None for c in crpix):
            self.Warning("HEADER ERROR: CRPIXs keywords not found.")
            return False
        self.crpix = [float(c) if c is not None else 1. for c in crpix]
        crval = [g(f"CRVAL{i + 1}") for i in range(n)]
        if all(c is None for c in crval):
            self.Warning("HEADER ERROR: CRVALSs keywords not found.")
            return False
        self.crval = [float(c) if c is not None else 0. for c in crval]
        self.drval3 = float(g("DRVAL3", 0.))

        ctype = [g(f"CTYPE{i + 1}") for i in range(n)]
        if all(c is None for c in ctype):
            self.Warning("HEADER WARNING: CTYPEs keywords not found. Assuming [RA,DEC,VRAD].")
            dflt = ["RA---NCP", "DEC--NCP", "VRAD"]
            ctype = [dflt[i] if i < 3 else "" for i in range(n)]
        self.ctype = [str(c) if c is not None else "" for c in ctype]
        if n > 0 and "RA" in self.ctype[0] and self.crval[0] < 0:
            self.crval[0] += 360.

        cunit = [g(f"CUNIT{i + 1}") for i in range(n)]
        if all(c is None for c in cunit):
            dflt = ["deg", "deg", "hz" if (n > 2 and "freq" in self.ctype[2].lower()) else "m/s"]
            cunit = [dflt[i] if i < 3 else "" for i in range(n)]
            self.Warning(f"HEADER WARNING: CUNITs keywords not found. Assuming [deg,deg,{cunit[2] if n > 2 else ''}]")
        else:
            cunit = [str(c) if c is not None else "" for c in cunit]
            for i in range(n):
                if "km s-1" in cunit[i]: cunit[i] = "km/s"
                if "m s-1" in cunit[i]: cunit[i] = "m/s"
                if cunit[i] == "":
                    cunit[i] = "km/s" if i == 2 else ("stokes" if i == 3 else "deg")
                    self.Warning(f"HEADER WARNING: CUNIT{i + 1} keywords not found. Assuming {cunit[i]}.")
        self.cunit = cunit

        self.dunit3 = str(g("DUNIT3", "NONE"))
        if "BUNIT" in hdr:
            self.bunit = str(g("BUNIT"))
        else:
            self.Warning("HEADER WARNING: BUNIT keyword not found.")
            self.bunit = "NONE"
        if "beam-1" in self.bunit.lower() and "jy" in self.bunit.lower():
            self.bunit = "JY/BEAM"
        self.btype = str(g("BTYPE", "NONE"))
        self.bzero = float(g("BZERO", 0.))
        self.bscale = float(g("BSCALE", 1.))
        self.blank = float(g("BLANK", 0.)) if isinstance(g("BLANK", 0.), (int, float)) else 0.
        ep = g("EPOCH", g("EQUINOX", 0.))
        try:
            self.epoch = float(ep)
        except (TypeError, ValueError):
            self.epoch = 0.
        rs = g("RADECSYS", g("RADESYS"))
        if rs is None:
            rs = "FK5" if self.epoch > 1984 else ("FK4" if self.epoch != 0 else "ICRS")
        self.radesys = str(rs)
        self.crota = float(g("CROTA", 0.))
        self.datamin = float(g("DATAMIN", 0.))
        self.datamax = float(g("DATAMAX", 0.))

        if n == 0:
            return True
        cu2 = self.cunit[n - 1].lower()
        ct2 = self.ctype[n - 1].lower()
        if n > 3:
            cu2 = self.cunit[2].lower()
            ct2 = self.ctype[2].lower()
        self.specsys = ""
        if "SPECSYS" in hdr:
            self.specsys = str(g("SPECSYS"))
        elif "VELREF" not in hdr:
            if "lsr" in ct2: self.specsys = "LSRK"
            elif "hel" in ct2: self.specsys = "BARYCENT"
            elif "top" in ct2: self.specsys = "TOPOCENT"

        if ("wav" in ct2 or "um" in cu2 or "nm" in cu2 or "ang" in cu2 or "micr" in cu2):
            self.sptype = "wave"
        elif "freq" in ct2 or "hz" in cu2:
            self.sptype = "freq"
        elif "vopt" in ct2 or "felo" in ct2:
            self.sptype = "velo-opt"
        elif "vel" in ct2 or "vrad" in ct2 or "m/s" in cu2:
            self.sptype = "velo-radio"
        else:
            self.sptype = "unknown"

        f0 = None
        for k in ("FREQ0", "RESTFREQ", "RESTFRQ"):
            if k in hdr:
                f0 = float(g(k))
                break
        if f0 is None:
            if self.dunit3 == "NONE" or self.drval3 == 0 or ("hz" in cu2 and "m/s" in cu2):
                self.Warning("HEADER WARNING: FREQ0-RESTFREQ keyword not found. Assuming 1.4204057 GHz.")
                f0 = 0.1420405751786E10
            else:
                d = 0.
                if self.dunit3.lower() == "km/s": d = self.drval3 * 1000
                elif self.dunit3.lower() == "m/s": d = self.drval3
                c3 = 0.
                if cu2 == "hz": c3 = self.crval[2]
                elif cu2 == "mhz": c3 = self.crval[2] * 1E06
                elif cu2 == "ghz": c3 = self.crval[2] * 1E09
                f0 = c3 * math.sqrt((299792458. + d) / (299792458. - d))
        self.freq0 = f0

        if "OBJECT" in hdr:
            obj = str(g("OBJECT"))
        else:
            self.Warning("HEADER WARNING: OBJECT keyword not found.")
            obj = "NONE"
        for a, b in (("/", "-"), ("\\", "-"), ("(", "-"), (")", "-")):
            obj = obj.replace(a, b)
        self.object = "".join("_" if ch.isspace() else ch for ch in obj)

        tel = g("TELESCOP", g("INSTRUME"))
        if tel is None:
            self.Warning("HEADER WARNING: TELESCOP-INSTRUME keywords not found.")
            tel = "NONE"
        self.telescope = str(tel)

        clbmaj = clbmin = 0.
        bmaj = g("BMAJ", g("BMMAJ"))
        if bmaj is None:
            clbmaj = float(g("CLBMMAJ", 0.))
            if clbmaj == 0: self.Warning("HEADER WARNING: BMAJ-BMMAJ-CLBMMAJ keywords not found.")
            bmaj = 0.
        bmin = g("BMIN", g("BMMIN"))
        if bmin is None:
            clbmin = float(g("CLBMMIN", 0.))
            if clbmin == 0: self.Warning("HEADER WARNING: BMIN-BMMIN-CLBMMIN keywords not found.")
            bmin = 0.
        bpa = g("BPA", g("BMPA"))
        if bpa is None:
            self.Warning("HEADER WARNING: BPA-BMPA keywords not found.")
            bpa = 0.
        self.bmaj, self.bmin, self.bpa = float(bmaj), float(bmin), float(bpa)
        if self.bmaj == 0 and self.bmin == 0 and self.bpa == 0 and clbmaj == 0 and clbmin == 0:
            # Search HISTORY cards (e.g. AIPS) for beam information
            for k in self.keys:
                for kw, attr in (("BMAJ=", "bmaj"), ("BMIN=", "bmin"), ("BPA=", "bpa")):
                    f = k.find(kw)
                    if f >= 0:
                        try:
                            setattr(self, attr, float(k[f + len(kw):].split()[0]))
                        except (ValueError, IndexError):
                            pass
            if self.bmaj != 0 and self.bmin != 0 and self.warning:
                print("\n--------> WARNING: beam information found in HISTORY keywords: <--------\n"
                      f" BMAJ = {self.bmaj:.5g} {self.cunit[0]}  BMIN = {self.bmin:.5g} {self.cunit[0]}"
                      f"  BPA = {self.bpa:.5g} DEGREE\n"
                      " It is heartily recommended to check these values before going on!!\n")
        for kw, attr in (("BMMAJ", "bmaj"), ("BMMIN", "bmin")):
            v = hdr.get(kw)
            if isinstance(v, str) and "D" in v:
                setattr(self, attr, getattr(self, attr) / 3600.)
        if clbmaj != 0: self.bmaj = clbmaj / 3600.
        if clbmin != 0: self.bmin = clbmin / 3600.

        self._hdr = hdr
        self.updateWCS(hdr)
        cs = cu2
        if cs == "km/s": self.specConv = 1 / 1000.
        if cs == "mhz": self.specConv = 1 / 1E06
        if cs == "ghz": self.specConv = 1 / 1E09
        if cs in ("micron", "mum", "um"): self.specConv = 1E06
        if cs in ("a", "ang", "angstrom"): self.specConv = 1E10
        self.calcArea()
        return True

    # ------------------------------------------------------------------ WCS
    def _wcs_header(self):
        """Build a minimal FITS header describing the WCS of this Header."""
        h = fits.Header()
        n = self.numAxes
        h["NAXIS"] = n
        for i in range(n):
            h[f"NAXIS{i + 1}"] = self.dimAxes[i]
            h[f"CRPIX{i + 1}"] = self.crpix[i]
            h[f"CRVAL{i + 1}"] = self.crval[i]
            h[f"CDELT{i + 1}"] = self.cdelt[i]
            h[f"CTYPE{i + 1}"] = self.ctype[i]
            cu = self.cunit[i]
            if cu.upper() in ("M/S", "KM/S", "CM/S", "HZ", "KHZ", "MHZ", "GHZ"):
                cu = {"M/S": "m/s", "KM/S": "km/s", "CM/S": "cm/s", "HZ": "Hz",
                      "KHZ": "kHz", "MHZ": "MHz", "GHZ": "GHz"}[cu.upper()]
            elif cu.lower() in ("mum", "micron"):
                cu = "um"
            elif cu.lower() in ("a", "ang", "angstrom"):
                cu = "Angstrom"
            elif cu.lower() == "stokes":
                cu = ""
            h[f"CUNIT{i + 1}"] = cu
        for k in ("RESTFRQ", "RESTFREQ", "RESTWAV", "SPECSYS", "EQUINOX", "RADESYS",
                  "LONPOLE", "LATPOLE", "PV2_1", "PV2_2", "VELREF", "DATE-OBS", "MJD-OBS"):
            if hasattr(self, "_hdr") and k in self._hdr:
                h[k] = self._hdr[k]
        if self.freq0 and "RESTFRQ" not in h and "RESTFREQ" not in h:
            h["RESTFRQ"] = self.freq0
        return h

    def updateWCS(self, hdr=None):
        if hdr is None:
            hdr = self._wcs_header()
        else:
            # Use the full header but with sanitized units
            wh = self._wcs_header()
            hdr = hdr.copy()
            for k in wh:
                hdr[k] = wh[k]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                self.wcs = WCS(hdr, relax=True, fix=True)
                self.wcs.wcs.set()
            except Exception:
                try:
                    self.wcs = WCS(self._wcs_header(), relax=True, fix=True)
                    self.wcs.wcs.set()
                except Exception:
                    self.wcs = None
        if self.wcs is not None:
            self.wcsIsGood = self.wcs.wcs.lng != -1 and self.wcs.wcs.lat != -1
            spec = self.wcs.wcs.spec
            self.specUnits = str(self.wcs.wcs.cunit[spec]).lower() if spec >= 0 else ""
        else:
            self.wcsIsGood = False

    def _specaxis(self):
        if self.wcs is None:
            return 2
        s = self.wcs.wcs.spec
        nax = self.wcs.wcs.naxis
        if s < 0: s = 2
        if s >= nax: s = nax - 1
        return s

    def pixToWCS(self, pix):
        """pix: (N,3) 0-based (x,y,z) -> (N,3) world (lng,lat,spec) in WCSLIB units."""
        pix = np.atleast_2d(np.asarray(pix, dtype=float))
        w = self.wcs
        nax = w.wcs.naxis
        spec = self._specaxis()
        full = np.zeros((pix.shape[0], nax))
        full[:, w.wcs.lng] = pix[:, 0]
        full[:, w.wcs.lat] = pix[:, 1]
        if nax > 2:
            full[:, spec] = pix[:, 2]
        world = w.wcs_pix2world(full, 0)
        out = np.zeros((pix.shape[0], 3))
        out[:, 0] = world[:, w.wcs.lng]
        out[:, 1] = world[:, w.wcs.lat]
        if nax > 2:
            out[:, 2] = world[:, spec]
        return out

    def wcsToPix(self, world, keep_spec=False):
        """world: (N,3) (lng,lat,spec) -> (N,3) 0-based pixels."""
        world = np.atleast_2d(np.asarray(world, dtype=float))
        w = self.wcs
        nax = w.wcs.naxis
        spec = self._specaxis()
        full = np.tile(np.array(w.wcs.crval, dtype=float), (world.shape[0], 1))
        full[:, w.wcs.lng] = world[:, 0]
        full[:, w.wcs.lat] = world[:, 1]
        if nax > 2:
            if keep_spec:
                full[:, spec] = w.wcs.crval[spec]
            else:
                full[:, spec] = world[:, 2]
        pix = w.wcs_world2pix(full, 0)
        out = np.zeros((world.shape[0], 3))
        out[:, 0] = pix[:, w.wcs.lng]
        out[:, 1] = pix[:, w.wcs.lat]
        if nax > 2:
            out[:, 2] = pix[:, spec]
        return out

    def getZphys(self, z, convert=True):
        """Channel (0-based) -> spectral value in the units of the spectral axis."""
        z = np.asarray(z, dtype=float)
        lin = (z + 1 - self.crpix[2]) * self.cdelt[2] + self.crval[2]
        if self.wcs is None or self.wcs.wcs.spec < 0:
            return lin if lin.ndim else float(lin)
        try:
            w = self.wcs
            nax = w.wcs.naxis
            spec = w.wcs.spec
            zz = np.atleast_1d(z)
            full = np.zeros((zz.size, nax))
            full[:, spec] = zz
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                world = w.wcs_pix2world(full, 0)[:, spec]
            if not np.all(np.isfinite(world)):
                raise ValueError
            if convert:
                world = world * self.specConv
            return world.reshape(z.shape) if z.ndim else float(world[0])
        except Exception:
            return lin if lin.ndim else float(lin)

    def getZgrid(self, specvalue, convert=True):
        """Spectral value (units of spectral axis) -> 0-based channel."""
        s = np.asarray(specvalue, dtype=float)
        lin = (s - self.crval[2]) / self.cdelt[2] + self.crpix[2] - 1
        if self.wcs is None or self.wcs.wcs.spec < 0:
            return lin if lin.ndim else float(lin)
        try:
            w = self.wcs
            w.wcs.naxis
            spec = w.wcs.spec
            ss = np.atleast_1d(s) / (self.specConv if convert else 1.)
            full = np.tile(np.array(w.wcs.crval, dtype=float), (ss.size, 1))
            full[:, spec] = ss
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pix = w.wcs_world2pix(full, 0)[:, spec]
            if not np.all(np.isfinite(pix)):
                raise ValueError
            return pix.reshape(s.shape) if s.ndim else float(pix[0])
        except Exception:
            return lin if lin.ndim else float(lin)

    def lngtype(self):
        t = str(self.wcs.wcs.lngtyp).strip() if self.wcs is not None else ""
        return t if t else self.ctype[0][:4]

    def lattype(self):
        t = str(self.wcs.wcs.lattyp).strip() if self.wcs is not None else ""
        return t if t else self.ctype[1][:4]

    # ------------------------------------------------------------------ checks
    def checkHeader(self):
        allgood = True
        if self.cdelt[0] > 0:
            self.Warning("HEADER CHECK: BBAROLO expects a negative CDELT1.")
            allgood = False
        lowval, highval = 1E-06, 0.5
        fac = 0
        c1, c2 = self.cunit[0].lower(), self.cunit[1].lower() if self.numAxes > 1 else ""
        if "deg" in c1 and "deg" in c2: fac = 1
        elif "arcm" in c1 and "arcm" in c2: fac = 60
        elif "arcs" in c1 and "arcs" in c2: fac = 3600
        if fac > 0 and (abs(self.cdelt[0]) < fac * lowval or abs(self.cdelt[0]) > fac * highval
                        or abs(self.cdelt[1]) < fac * lowval or abs(self.cdelt[1]) > fac * highval):
            self.Warning("HEADER CHECK: CDELT1 and/or CDELT2 values are suspiciously large or small. ")
            allgood = False
        if self.sptype == "velo-opt" and self.numAxes > 2:
            if self.crpix[2] < -50 or self.crpix[2] > self.dimAxes[2] + 50:
                v1 = self.getZphys(0)
                v2 = self.getZphys(1)
                dv = v2 - v1
                for s in ("HEADER CHECK: CRPIX3 is very outside the channel range and CDELT3 is not constant.",
                          "              This may create issues when building up a kinematical model.",
                          "              Please update the data with a spectral reference channel inside cube.",
                          "              I will fix it for this run, but you should change it in your data. ",
                          "              For example, change to : CRPIX3 = 1",
                          f"                                       CRVAL3 = {v1:.6f}",
                          f"                                       CDELT3 = {dv:.6f}"):
                    self.Warning(s)
                allgood = False
                self.crpix[2], self.crval[2], self.cdelt[2], self.cunit[2] = 1, v1, dv, "m/s"
                self.updateWCS()
        if self.beamArea != 0 and self.numAxes > 1:
            r1, r2 = abs(self.bmaj / self.cdelt[0]), abs(self.bmin / self.cdelt[1])
            if r1 < 1 or r2 < 1 or r1 > 30 or r2 > 30:
                self.Warning("HEADER CHECK: BMAJ and/or BMIN values are suspiciously large or small. ")
                allgood = False
        if self.numAxes > 2 and "wavelength" in self.ctype[2].lower():
            self.Warning("HEADER CHECK: CYTPE3=WAVELENGTH is not FITS standard. Change it to WAVE!")
            allgood = False
        return allgood

    # ------------------------------------------------------------------ writing
    def toAstropy(self, numDim=3, fullHead=False, shape=None):
        """Return an astropy Header (port of Header::headwrite)."""
        h = fits.Header()
        for i in range(min(numDim, self.numAxes)):
            k = i + 1
            h[f"CRPIX{k}"] = self.crpix[i]
            h[f"CRVAL{k}"] = self.crval[i]
            h[f"CDELT{k}"] = self.cdelt[i]
            h[f"CTYPE{k}"] = self.ctype[i]
            h[f"CUNIT{k}"] = self.cunit[i]
        h["BUNIT"] = self.bunit
        if self.btype != "NONE": h["BTYPE"] = self.btype
        if self.bmaj != 0: h["BMAJ"] = self.bmaj
        if self.bmin != 0: h["BMIN"] = self.bmin
        h["BPA"] = self.bpa
        if self.object != "NONE": h["OBJECT"] = self.object
        if self.epoch != 0: h["EQUINOX"] = self.epoch
        h["RADESYS"] = self.radesys
        if self.specsys != "": h["SPECSYS"] = self.specsys
        if self.telescope != "NONE": h["TELESCOP"] = self.telescope
        if self.freq0 != 0: h["RESTFRQ"] = self.freq0
        if self.datamax != 0: h["DATAMAX"] = self.datamax
        if self.datamin != 0: h["DATAMIN"] = self.datamin
        if fullHead:
            skip = {"SIMPLE", "BITPIX", "NAXIS", "EXTEND", "BZERO", "BSCALE", "BLANK", "END", "COMMENT", ""}
            for card in self.keys:
                if card.startswith("HISTORY"):
                    h.add_history(card[7:].strip() if card[7:8] == " " else card[7:])
                    continue
                f = card.find("=")
                if f < 0:
                    continue
                kw = card[:f].strip()
                if kw in skip or kw.startswith("NAXIS") or kw in h:
                    continue
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        c = fits.Card.fromstring(card)
                        c.verify("silentfix")
                        h.append(c)
                except Exception:
                    pass
        return h

    def __str__(self):
        return "\n".join(self.keys)
