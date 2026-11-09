"""
Parameter handling for BBarolo (port of Arrays/param.hh and param.cpp).

All BBarolo input parameters live in a :class:`Param` instance. Parameters can be
read from a parameter file (``readParamFile``), from a string (``readParamString``),
from the command line (``readParamCL``, ``KEY=value``) or set programmatically
with :meth:`Param.set` / :meth:`Param.update`.

Task specific parameters are stored in the sub-structures ``gm`` (GALMOD), ``gf``
(3DFIT), ``gw`` (GALWIND), ``se`` (SEARCH) and ``ma`` (MAPS), mirroring the C++
``GALMOD_PAR``, ``GALFIT_PAR``, ``GALWIND_PAR``, ``SEARCH_PAR`` and ``MAPS_PAR``.
"""

import os
import io
import re
import math
import copy
import sys
import importlib.util
from dataclasses import dataclass, field
from typing import List

from ._version import __version__

BBVERSION = __version__

TASKS_LIST = [
    ("3DFIT",         "Fit a 3D tilted-ring model to an emission-line datacube."),
    ("BAYESIAN3DFIT", "Fit a 3D tilted-ring model by nested sampling of the posterior (needs dynesty or nautilus)."),
    ("GALMOD",        "Make a 3D model observation datacube of an emission line."),
    ("SEARCH",        "Search for sources in a 3D datacube or in a 2D image."),
    ("2DFIT",         "Fit a 2D tilted-ring model to a velocity field."),
    ("SMOOTH",        "Smooth spatially of a datacube with a 2D Gaussian kernel."),
    ("SMOOTHSPEC",    "Smooth spectrally of a datacube with a given window."),
    ("ELLPROF",       "Derive radial profiles in elliptical rings from a map."),
    ("MAKEMASK",      "Define a mask for emission-line data."),
    ("SPACEPAR",      "Compute the full 2D space for any pair of 3DFIT parameters."),
    ("GALWIND",       "Make a 3D emission-line model of a bi-conical outflow."),
    ("STATS",         "Simply calculate and print statistics for a cube."),
    ("CONTSUB",       "Subtract a polynomial continuum from a cube."),
    ("SLITFIT",       "Fit the kinematics from long-slit data (NOT PORTED: broken in C++ too)."),
    ("GLOBALPROFILE", "Extract a total spectral profile from a datacube."),
    ("TOTALMAP",      "Compute a total intensity map from a datacube."),
    ("VELOCITYMAP",   "Compute a velocity field from a datacube."),
    ("DISPERSIONMAP", "Compute a velocity dispersion field from a datacube."),
    ("PVSLICE",       "Extracting a position-velocity slice from a datacube."),
    ("REND3D",        "Writing a 3D view of a datacube."),
]

FITSUTILS_LIST = [
    ("listhead",  "List header keywords in a FITS file."),
    ("modhead",   "Modify/print a header keyword in a FITS file."),
    ("remhead",   "Delete a header keyword in a FITS file."),
    ("fitscopy",  "Copy a input file to a new FITS file with filtering."),
    ("fitsarith", "Arithmetic operations with FITS files."),
    ("jybeam2k",  "Convert flux units from Jy/beam to K."),
    ("k2jybeam",  "Convert flux units from K to Jy/beam."),
]


# Parsing helpers (port of Utilities/string.cpp) and the keyword table
from .keywords import (boolify, stringize, readint, readfloat, readflag, readflagorint,  # noqa: F401
                       readstr, readvec, isNumber, splitString, _tokens, _first)
from . import keywords as _kw
from .bbutils import BBaroloError


def checkHome(s):
    if s and s[0] == "~":
        return os.path.expanduser("~") + s[1:]
    return s


_FREE_NAMES = ("vrot", "disp", "dens", "z0", "inc", "pa", "phi", "xpos", "ypos", "vsys", "vrad", "all")
_FREE_NAMES_2D = ("vrot", "inc", "pa", "phi", "xpos", "ypos", "vsys", "vrad", "vexp", "all")      # (2DFIT)
_NUM = r"[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?"


def unknown_keyword(name):
    """An unknown keyword of a parameter file (or KEY=value): an error if it is very close to a known
    one (a typo, e.g. VDSIP) or a keyword removed in BBarolo 2 (e.g. BMAJ), otherwise a warning (it is
    ignored, as in C++)."""
    import difflib
    _kw.check_removed(name)
    close = difflib.get_close_matches(str(name).lower(), list(_kw.LOOKUP), n=1, cutoff=0.8)
    if close:
        from .bbutils import BBaroloError
        raise BBaroloError(f"{_kw.unknown_message(name)}")
    sys.stderr.write(f" BBAROLO WARNING: {_kw.unknown_message(name)}. It is ignored.\n")


def ring_value_problem(key, value):
    """None if value is acceptable for the ring parameter key (RADII, VROT, XPOS, ...), otherwise a
    message: a number, one number per ring, file(name,col) or, for RADII, Rmin~Rmax:step; for XPOS and
    YPOS also a WCS value (123.4d, 07:36:51, 07h36m51s)."""
    import re
    v = str(value).strip()
    if v in ("", "-1") or "file(" in v.lower():
        return None
    toks = v.replace(",", " ").split()
    if key == "RADII" and len(toks) == 1 and re.fullmatch(_NUM + "~" + _NUM + ":" + _NUM, toks[0]):
        return None
    ok = re.compile(_NUM + "$")
    wcs = re.compile(r"[+-]?[\d.:]+(d|deg|h)?([\d.]+m)?([\d.]+s)?$|[+-]?\d+[hd]\d*m?[\d.]*s?$")
    for t in toks:
        if ok.match(t) or (key in ("XPOS", "YPOS") and wcs.match(t.lower())):
            continue
        return (f"{key}={v!r} is not valid: give a number, one number per ring, file(name,column)"
                + (" or Rmin~Rmax:step" if key == "RADII" else "")
                + (", or a WCS position (e.g. 123.4d, 07:36:51, 07h36m51s)" if key in ("XPOS", "YPOS") else ""))
    return None


def free_problem(free, task="3DFIT"):
    """None if every name of FREE (of 3DFIT, or of 2DFIT) is known, otherwise a message."""
    import re
    names = _FREE_NAMES_2D if task == "2DFIT" else _FREE_NAMES
    bad = [t for t in re.split(r"[\s,]+", str(free).strip()) if t and not any(k in t.lower() for k in names)]
    if bad:
        acc = ("VROT, INC, PA, XPOS, YPOS, VSYS, VRAD (or VEXP)" if task == "2DFIT" else
               "VROT, VDISP, DENS, Z0, INC, PA, XPOS, YPOS, VSYS, VRAD")
        return f"unknown free parameter(s) {', '.join(bad)} in FREE (accepted: {acc} or ALL)"
    return None


# --------------------------------------------------------------------------
# Parameter structures
# --------------------------------------------------------------------------
@dataclass
class GalmodPar:
    flagGALMOD: bool = False
    NRADII: int = -1
    RADII: str = "-1"
    XPOS: str = "-1"
    YPOS: str = "-1"
    RADSEP: float = -1
    VSYS: str = "-1"
    VROT: str = "-1"
    VRAD: str = "-1"
    VVERT: str = "-1"
    VDISP: str = "-1"
    INC: str = "-1"
    PHI: str = "-1"
    Z0: str = "-1"
    DENS: str = "-1"
    DVDZ: str = "-1"
    ZCYL: str = "-1"
    CDENS: int = 10
    CMODE: int = 1
    ISEED: int = -1
    LTYPE: int = 1
    NV: int = -1
    REDSHIFT: float = 0.0
    EMPTY: bool = True
    DENSFLUX: bool = True
    RESTWAVE: List[float] = field(default_factory=lambda: [-1.0])
    RESTFREQ: List[float] = field(default_factory=lambda: [-1.0])
    RELINT: List[float] = field(default_factory=lambda: [1.0])
    SM: bool = True
    NOISERMS: float = 0.0
    ringfile: str = ""


@dataclass
class GalfitPar(GalmodPar):
    flagGALFIT: bool = False
    DELTAINC: float = 5.0
    DELTAPHI: float = 15.0
    DELTAVROT: float = 1000.0
    MAXVDISP: float = 1000.0
    MINVDISP: float = 0.1
    FTYPE: int = 2
    WFUNC: int = 2
    TOL: float = 1e-3
    NORM: str = "AZIM"
    FREE: str = "VROT VDISP INC PA"
    GUESSTYPE: str = "LEGACY"        # initial estimates: LEGACY (as C++), ELLIPSE (corrected) or FIT
    SIDE: str = "B"
    FREEANGLE: float = 15.0          # 2DFIT: the angle around the minor axis left out of the fit (degrees)
    TWOSTAGE: bool = True
    REGTYPE: str = "auto"
    BWEIGHT: float = 1.0
    STARTRAD: int = 0
    flagERRORS: bool = False
    flagADRIFT: bool = False
    ADRIFTPOL1: int = 3
    ADRIFTPOL2: int = 3
    REVERSE: str = "false"
    NORMALCUBE: bool = True
    flagBADOUT: bool = False
    PLOTMASK: bool = False
    PLOTMINCON: float = -1.0
    # BAYESIAN3DFIT (Bayesian3DFIT)
    flagBAYESIAN: bool = False
    SAMPLER: str = "NAUTILUS"
    PRIORS: str = ""
    SAMPLEROPTS: str = ""
    RUNOPTS: str = ""
    LIKEOPTS: str = ""


@dataclass
class GalwindPar(GalmodPar):
    flagGALWIND: bool = False
    flagBAYESIAN: bool = False          # (BAYESIANWIND)
    VWIND: str = "-1"
    VROT: str = "0"
    OPENANG: str = "-1"
    OPENIN: str = "0"
    HTOT: float = -1.0
    NTOT: int = 25
    LTYPE: int = 5
    DENSTYPE: int = 1
    WTYPE: int = 0
    DISKRINGS: str = ""


@dataclass
class SearchPar:
    flagSearch: bool = False
    searchType: str = "spatial"
    snrCut: float = 4.0
    iternoise: bool = False
    threshold: float = 0.0
    UserThreshold: bool = False
    flagAdjacent: bool = True
    threshSpatial: int = -1
    threshVelocity: int = 2
    minChannels: int = 2
    RejectBeforeMerge: bool = True
    TwoStageMerging: bool = True
    minVoxels: int = -1
    minPix: int = -1
    maxChannels: int = -1
    maxAngSize: float = -1.0
    flagGrowth: bool = True
    growthCut: float = 2.5
    flagUserGrowthT: bool = False
    growthThreshold: float = 0.0
    cubelets: bool = False
    edges: int = 20
    sortsrcs: str = "nvox"
    pbcorr: int = 0


@dataclass
class MapsPar:
    maptype: str = "MOMENT"
    globprof: bool = False
    massdensmap: bool = False
    totalmap: bool = False
    velocitymap: bool = False
    dispersionmap: bool = False
    rmsmap: bool = False
    SNmap: bool = False
    contChans: List[int] = field(default_factory=lambda: [1000000])
    taper: str = "uniform"
    veldef: str = "auto"


# --------------------------------------------------------------------------
# The Param class
# --------------------------------------------------------------------------
class Param:
    """Container of all input parameters (port of the C++ ``Param`` class)."""

    def __init__(self, paramfile=None, **kwargs):
        self.defaultValues()
        if paramfile is not None:
            if not self.readParamFile(paramfile):
                raise FileNotFoundError(f"Could not open parameter file {paramfile}")
        if kwargs:
            self.update(**kwargs)

    # ------------------------------------------------------------------
    def defaultValues(self):
        self._given = {}           # the parameters given, as strings {KEYWORD: value} (see Keyword.apply)
        self.imageFile = ""
        self.imageList = "NONE"
        self.images = []
        self.outFolder = ""
        self.outPrefix = ""
        self.logFile = False
        self.verbose = 1             # 0 quiet, 1 normal, 2 everything (VERBOSE)
        self.showbar = True
        self.plots = 1
        self.beam = None              # BEAM: [bmaj, bmin, bpa] (arcsec, deg; bpa None: 0), replaces the header beam
        self.checkCube = 0
        self.flagStats = False
        self.flagRobustStats = True
        self.fluxConvert = True

        self.contsub = False
        self.exclude_windows = "NONE"
        self.cont_order = 1

        self.makeMask = False
        self.MaskType = "SEARCH"
        self.blankCut = 3.0

        self.flagRing = False

        self.flagSmooth = False
        self.flagFFT = True
        self.newbeam = None           # NEWBEAM: [bmaj, bmin, bpa] of SMOOTH (bpa None: that of the data)
        self.linear = -1.0
        self.specres = -1.0           # spectral resolution of the models (sigma, channels); -1: LINEAR
        self.factor = 2.0
        self.scalefactor = -1.0
        self.reduce = "false"
        self.smo_out = "NONE"
        self.BOX = [-1] * 6

        self.flagSmoothSpectral = False
        self.window_type = "HANNING"
        self.window_size = 3.0

        self.flagSlitfit = False
        self.slitwidth = -1.0

        self.flagSpace = False
        self.P1 = "VROT"
        self.P2 = "VDISP"
        self.P1p = [0.0, 100.0, 2.0]
        self.P2p = [0.0, 100.0, 2.0]

        self.flagPV = False
        self.XPOS_PV = "-1"
        self.YPOS_PV = "-1"
        self.PA_PV = 0.0
        self.WIDTH_PV = 0.0
        self.P1_PV = [-1.0, -1.0]
        self.P2_PV = [-1.0, -1.0]
        self.ANTIALIAS = 0.5

        self.flagEllProf = False
        self.flagRend3D = False
        self.rendangle = 360.0

        self.debug = False
        self.AUTO = False
        self.threads = min(os.cpu_count() or 1, 16)   # (all the cores is too many on a shared server)

        # New options of the python port
        self.galmodtype = "MC"   # MC = classic Monte Carlo GALMOD, ANALYTIC = noiseless profiles
        self.ringmode = "CENTRED"  # rings of the models: CENTRED annuli, NODES or SHIFTED (BBarolo 1.x)

        self.gm = GalmodPar()
        self.gf = GalfitPar()
        self.gw = GalwindPar()
        self.se = SearchPar()
        self.ma = MapsPar()

    # ------------------------------------------------------------------
    def copy(self):
        return copy.deepcopy(self)

    # Convenience accessors (same names as C++ getters, for readability)
    def getMaps(self):
        m = self.ma
        return m.globprof or m.totalmap or m.velocitymap or m.dispersionmap or m.massdensmap or m.rmsmap

    def getflagReduce(self):
        return self.reduce != "false"

    def setOutfolder(self, s):
        if s != "" and not s.endswith("/"):
            s += "/"
        self.outFolder = s

    def getRedshift(self):
        # z <= -1 is unphysical: it is used as "not set" (e.g. the default -1 of the original pyBBarolo)
        z = self.gf.REDSHIFT
        return 0. if z <= -1 else z

    def getRestwave(self):
        return self.gf.RESTWAVE[0]

    def getRestfreq(self):
        return self.gf.RESTFREQ[0]

    # ------------------------------------------------------------------
    # Reading parameters
    # ------------------------------------------------------------------
    def readParamFile(self, paramfile):
        if not os.path.isfile(paramfile):
            return False
        with open(paramfile, "r") as fin:
            for line in fin:
                line = line.rstrip("\n")
                if line and line[0] not in "#/":
                    self.setParam(line)
        return True

    def readParamString(self, paramstring):
        for line in io.StringIO(paramstring):
            line = line.rstrip("\n")
            if line and line[0] not in "#/":
                self.setParam(line)
        return True

    def readParamCL(self, parstr):
        found = parstr.find("=")
        if found < 0:
            return False
        self.setParam(parstr[:found] + " " + parstr[found + 1:])
        return True

    def update(self, **kwargs):
        """Set parameters from keyword arguments, e.g. ``update(vrot=100, free='VROT VDISP')``.
        Keywords are the parameter-file keywords (case-insensitive) or their aliases;
        see :func:`bbarolo.help`."""
        for k, v in kwargs.items():
            self.set(k, v)

    def set(self, key, value):
        """Set a single parameter from a Python value (key: a parameter-file keyword or an
        alias, case-insensitive). Numbers, strings, booleans and sequences (for lists of
        values) are accepted; None restores the default value. Raises KeyError for an
        unknown keyword and ValueError/TypeError for a value of the wrong type."""
        kw = _kw.lookup(key)
        if kw is None:
            _kw.check_removed(key)
            raise _kw.UnknownParameterError(_kw.unknown_message(key))
        if value is None:
            d = _default_param()
            for t in kw.targets:
                _kw._setattr(self, t, copy.deepcopy(_kw._getattr(d, t)))
            for t in kw.also:
                _kw._setattr(self, t, copy.deepcopy(_kw._getattr(d, t)) if _hasattr(d, t) else False)
            self._given.pop(kw.name, None)
            return
        kw.apply(self, value_to_string(kw, value))

    def setParam(self, parstr):
        """Parse a 'KEYWORD value' string (port of Param::setParam). Unknown keywords
        are ignored with a warning."""
        parstr = parstr.strip()
        if not parstr:
            return
        parts = parstr.split(None, 1)
        rest = parts[1] if len(parts) > 1 else ""
        kw = _kw.lookup(parts[0])
        if kw is None:
            unknown_keyword(parts[0])
            return
        kw.apply(self, rest)

    # ------------------------------------------------------------------
    def checkPars(self, quiet=False):
        """Check parameters for consistency (port of Param::checkPars)."""
        out = (lambda *a, **k: None) if quiet else print
        good = True
        gf, gm, gw, se = self.gf, self.gm, self.gw, self.se

        self.images = []
        if self.imageList != "NONE":
            if not os.path.isfile(self.imageList):
                out(f"\n Error opening list file: {self.imageList} doesn't exist.\n")
                return False
            with open(self.imageList) as f:
                for s in f:
                    s = checkHome(s.strip())
                    if s:
                        self.images.append(os.path.abspath(s))
        else:
            self.imageFile = checkHome(self.imageFile)
            if self.imageFile != "":
                # absolute paths (relative to the current folder): they are written in the plotting
                # scripts and other outputs, which must work from any folder
                self.imageFile = os.path.abspath(self.imageFile)
                self.images.append(self.imageFile)

        if len(self.images) == 0:
            out("\n BBAROLO ERROR: no FITSFILE provided.")
            return False

        self.outFolder = checkHome(self.outFolder)
        if self.outFolder != "":
            self.outFolder = os.path.abspath(self.outFolder) + "/"

        if not self.verbose:
            self.showbar = False

        # MASK
        ms = self.MaskType.upper()
        maskav = ["SMOOTH", "SEARCH", "SMOOTH&SEARCH", "SMOOTHSPEC&SEARCH", "SEARCHLARGEST",
                  "SMOOTH&SEARCHLARGEST", "SMOOTHSPEC&SEARCHLARGEST", "THRESHOLD", "NEGATIVE", "NONE"]
        maskOK = False
        if "FILE(" in ms:
            first = ms.find("(")
            secon = ms.rfind(")")
            self.MaskType = ms[:first] + self.MaskType[first:secon] + ms[secon:]
            maskOK = True
        else:
            for m in maskav:
                if ms == m or (m + "_ENLARGE") in ms:
                    maskOK = True
                    self.MaskType = ms
        if not maskOK:
            out(f"\n ERROR: Unknown type of mask: {self.MaskType}\n Setting to SMOOTH")
            self.MaskType = "SMOOTH"

        if self.contsub and self.cont_order < 0:
            out("CONTORDER must be >0. Setting to 1.")
            self.cont_order = 1

        if self.checkCube < 0 or self.checkCube > 5:
            out("CHECKCUBE must >=0 and <=5. Setting it to False (=0).")
            self.checkCube = 0

        if se.flagSearch or "SEARCH" in str(self.MaskType).upper():     # (also the masks of the source finder)
            if se.searchType not in ("spatial", "spatialsmooth", "spectral"):
                out(f'You have requested a search type of "{se.searchType}".\n'
                    'Only "spectral", "spatial" and "spatialsmooth" are accepted. Setting to "spatial".')
                se.searchType = "spatial"
            if se.flagGrowth:
                bad = False
                if se.UserThreshold and (se.threshold < se.growthThreshold or se.snrCut < se.growthCut):
                    out(f'Your "growthThreshold" parameter{se.growthThreshold} is larger than your "threshold"{se.threshold}')
                    bad = True
                if not se.UserThreshold and se.snrCut < se.growthCut:
                    out(f'Your "growthCut" parameter {se.growthCut} is larger than your "snrCut"{se.snrCut}')
                    bad = True
                if bad:
                    out("The growth function is being turned off.")
                    se.flagGrowth = False

        if gf.flagBAYESIAN:
            if gf.flagGALFIT:
                out("BAYESIAN3DFIT warning: 3DFIT and BAYESIAN3DFIT can not be run at the same time (they write "
                    "the same outputs). Turning off 3DFIT.")
                gf.flagGALFIT = False
            if gm.flagGALMOD:
                out("BAYESIAN3DFIT warning: GALMOD and BAYESIAN3DFIT can not be run at the same time. "
                    "Turning off GALMOD.")
                gm.flagGALMOD = False
            gf.SAMPLER = gf.SAMPLER.strip().upper()
            if gf.SAMPLER not in ("DYNESTY", "NAUTILUS"):
                out(f"BAYESIAN3DFIT error: unknown SAMPLER {gf.SAMPLER} (DYNESTY or NAUTILUS).")
                good = False
            elif importlib.util.find_spec(gf.SAMPLER.lower()) is None:
                out(f"BAYESIAN3DFIT error: SAMPLER {gf.SAMPLER} needs the Python package {gf.SAMPLER.lower()} "
                    f"(pip install {gf.SAMPLER.lower()}).")
                good = False
            if "FREE" not in getattr(self, "_given", {}):
                out("BAYESIAN3DFIT error: FREE is not an optional parameter. Give the free parameters, e.g. "
                    "FREE VROT VDISP_SINGLE INC_SINGLE (see bbarolo.help('FREE')).")
                good = False
            for key, check in (("FREE", bayes_freepar), ("PRIORS", bayes_priors), ("SAMPLEROPTS", bayes_options),
                               ("RUNOPTS", bayes_options), ("LIKEOPTS", bayes_likeopts)):
                if key == "FREE" and "FREE" not in getattr(self, "_given", {}):
                    continue
                try:
                    check(getattr(gf, key))
                except ValueError as e:
                    out(f"BAYESIAN3DFIT error: {key}: {e}")
                    good = False

        if gf.flagGALFIT or gf.flagBAYESIAN or self.flagSpace or gm.flagGALMOD or self.flagSlitfit:
            msg = " is not an optional parameter. Please specify it in the input file or set flagSearch=true"
            if gf.flagGALFIT and gm.flagGALMOD:
                out("3DFIT warning: 3DFIT and GALMOD can not be run at the same time. Turning off GALMOD.")
                gm.flagGALMOD = False
            if gm.flagGALMOD and not gm.ringfile:
                if gf.NRADII == -1 and gf.RADII == "-1":
                    out("3DFIT error: NRADII or RADII" + msg); good = False
                for k, n in (("XPOS", "XPOS"), ("YPOS", "YPOS"), ("VSYS", "VSYS"), ("VROT", "VROT"),
                             ("VDISP", "VDISP"), ("INC", "INC"), ("PHI", "PHI"), ("Z0", "Z0")):
                    if getattr(gm, k) == "-1":
                        out(f"3DFIT error: {n}" + msg); good = False
                if gm.RADSEP == -1 and gm.RADII == "-1":
                    out("3DIT error: RADSEP" + msg); good = False

            for n in gf.NORM.split():
                if n not in ("NONE", "LOCAL", "AZIM", "AZIMUTHAL", "BOTH"):
                    out(f"3DFIT error: Unknown type of normalization: {n}")
                    good = False
            # (AZIMUTHAL is AZIM: C++ accepted it here, then used LOCAL)
            gf.NORM = " ".join("AZIM" if n == "AZIMUTHAL" else n for n in gf.NORM.split())

            if gf.flagGALFIT:
                if gf.FREE == "":
                    out("3DFIT error: FREE" + msg); good = False
                elif free_problem(gf.FREE):
                    out("3DFIT error: " + free_problem(gf.FREE)); good = False
                if gf.MINVDISP >= gf.MAXVDISP:
                    out(f"3DFIT error: MINVDISP ({gf.MINVDISP:g}) must be smaller than MAXVDISP ({gf.MAXVDISP:g}).")
                    good = False
            if (gf.flagGALFIT or self.flagSpace) and gf.STARTRAD < 0:
                out(f"3DFIT error: STARTRAD ({gf.STARTRAD}) is the index of the first ring to fit: 0 or more.")
                good = False
            if self.flagSpace:
                for name, (lo, hi, step) in (("P1PAR", self.P1p[:3]), ("P2PAR", self.P2p[:3])):
                    # (C++ explored min to min + |max - min| when min > max, and divided by a step 0)
                    if not (lo <= hi and step > 0):
                        out(f"SPACEPAR error: {name} ({lo:g} {hi:g} {step:g}) must be the minimum, the maximum "
                            "and a positive step.")
                        good = False
            if gf.flagGALFIT or self.flagSpace:          # (SPACEPAR computes the residuals of 3DFIT)
                if gf.FTYPE < 1 or gf.FTYPE > 4:
                    out("3DFIT warning: Not valid argument for FTYPE parameter. Assuming 2 (|mod-obs|).")
                    gf.FTYPE = 2
                if abs(gf.WFUNC) > 2:
                    out("3DFIT warning: Not valid argument for WFUNC parameter. Assuming 1 (|cos(θ)| weighting function).")
                    gf.WFUNC = 1
            if gf.flagGALFIT:
                # REGTYPE checking
                polyn = gf.REGTYPE.split()
                keep = []
                for w in polyn:
                    kv = splitString(w, "=")
                    if kv is not None:
                        key, val = kv
                        if key not in ("inc", "pa", "vsys", "xpos", "ypos", "z0"):
                            sys.stderr.write(f"3DFIT warning: invalid REGTYPE key of '{key}'. I will ignore it. \n")
                            continue
                    else:
                        key = val = w
                        if len(polyn) != 1:
                            sys.stderr.write(f"3DFIT warning: invalid REGTYPE key of '{w}'. I will ignore it. \n")
                            continue
                    if not isNumber(val) and val not in ("bezier", "median", "auto"):
                        sys.stderr.write(f"3DFIT warning: invalid REGTYPE value of '{val}' for {key}. Changing it to 'auto'. \n")
                        val = "auto"
                    keep.append(val if key == val else f"{key}={val}")
                gf.REGTYPE = " ".join(keep) if keep else "auto"

            if gf.LTYPE < 1 or gf.LTYPE > 5:
                out("3DFIT warning: Not valid argument for LTYPE parameter. Assuming 1 (gaussian layer).")
                gf.LTYPE = 1
                gm.LTYPE = 1

            if self.flagSlitfit:
                if gf.flagGALFIT:
                    sys.stderr.write("3DFIT WARNING: Galfit and Slitfit cannot be run at the same time. Switching off Slitfit. \n")
                    self.flagSlitfit = False
                elif self.slitwidth < 0:
                    sys.stderr.write("SLITFIT WARNING: SLITWIDTH has to be specified and > 0. \n")
                    good = False

            for s in (gf, gm):
                if len(s.RESTWAVE) == 0: s.RESTWAVE = [-1.0]
                if len(s.RESTFREQ) == 0: s.RESTFREQ = [-1.0]
                if len(s.RELINT) == 0: s.RELINT = [-1.0]
            if gf.RESTWAVE[0] != -1 and len(gf.RELINT) != len(gf.RESTWAVE):
                sys.stderr.write("3DFIT ERROR: RESTWAVE and RELINT must have same size.\n")
                good = False
            if gf.RESTFREQ[0] != -1 and len(gf.RELINT) != len(gf.RESTFREQ):
                sys.stderr.write("3DFIT ERROR: RESTFREQ and RELINT must have same size.\n")
                good = False
            if not any(k in gf.REVERSE for k in ("auto", "true", "false")):
                sys.stderr.write("3DFIT ERROR: accepted values for REVERSE are 'true', 'false' or 'auto'. "
                                 "Setting it to 'auto'.\n")
                gf.REVERSE = "auto"
            if gf.flagADRIFT and gf.ADRIFTPOL2 < 0:
                sys.stderr.write("3DFIT ERROR: ADRIFTPOL2 needs to be >0. Defaulting it to 3.\n")
                gf.ADRIFTPOL2 = 3

        if gf.flagGALFIT or gf.flagBAYESIAN or self.flagSpace or self.flagRing or self.flagEllProf:
            # the long forms of SIDE, for all the tasks that use it (C++ never converted them)
            S = gf.SIDE
            if S in ("A", "APP", "APPR", "APPROACHING"): S = "A"
            elif S in ("R", "REC", "RECED", "RECEDING"): S = "R"
            elif S in ("B", "BOTH"): S = "B"
            elif S in ("BS", "S", "SINGLE", "BOTH SINGLE"): S = "S"
            else:
                out("3DFIT WARNING: Not valid argument for SIDE parameter. Assuming B (fitting both sides of the "
                    "galaxy).")
                S = "B"
            gf.SIDE = S

        if gf.flagGALFIT or gf.flagBAYESIAN or self.flagSpace or gm.flagGALMOD or self.flagRing or self.flagEllProf:
            st = gm if gm.flagGALMOD else gf
            for key in _RING_KEYS:
                attr = "PHI" if key == "PA" else key
                if hasattr(st, attr):
                    prob = ring_value_problem(key, getattr(st, attr))
                    if prob:
                        out("INPUT error: " + prob); good = False

        if self.flagRing and not gf.flagGALFIT and free_problem(gf.FREE, "2DFIT"):    # (3DFIT: checked above)
            out("2DFIT error: " + free_problem(gf.FREE, "2DFIT")); good = False
        if self.flagRing and not 0 <= gf.FREEANGLE < 90:
            out(f"2DFIT error: FREEANGLE must be between 0 and 90 degrees, not {gf.FREEANGLE:g}."); good = False

        if self.flagEllProf:
            msg = " is not an optional parameter. Please specify it in the input file."
            if (gf.NRADII == -1 or gf.RADSEP == -1) and gf.RADII == "-1":
                out("ELLPROF error: NRADII or RADII" + msg); good = False
            for k, n in (("XPOS", "XPOS"), ("YPOS", "YPOS"), ("INC", "INC"), ("PHI", "PA")):
                if getattr(gf, k) == "-1":
                    out(f"ELLPROF ERROR: {n}" + msg); good = False

        if gw.flagBAYESIAN:
            from .bayesian.wind import wind_freepar, wind_priors, wind_likeopts
            if gf.flagBAYESIAN:
                out("BAYESIANWIND error: BAYESIAN3DFIT and BAYESIANWIND can not be run at the same time (they share "
                    "FREE, PRIORS and the options of the sampler).")
                good = False
            gf.SAMPLER = gf.SAMPLER.strip().upper()
            if gf.SAMPLER not in ("DYNESTY", "NAUTILUS"):
                out(f"BAYESIANWIND error: unknown SAMPLER {gf.SAMPLER} (DYNESTY or NAUTILUS).")
                good = False
            elif importlib.util.find_spec(gf.SAMPLER.lower()) is None:
                out(f"BAYESIANWIND error: SAMPLER {gf.SAMPLER} needs the Python package {gf.SAMPLER.lower()} "
                    f"(pip install {gf.SAMPLER.lower()}).")
                good = False
            if "FREE" not in getattr(self, "_given", {}):
                out("BAYESIANWIND error: FREE is not an optional parameter. Give the free parameters, e.g. "
                    "FREE VWIND_SINGLE VDISP_SINGLE INC PA OPENANG (see bbarolo.help('BAYESIANWIND')).")
                good = False
            for key, check in (("FREE", wind_freepar), ("PRIORS", wind_priors), ("SAMPLEROPTS", bayes_options),
                               ("RUNOPTS", bayes_options), ("LIKEOPTS", wind_likeopts)):
                if key == "FREE" and "FREE" not in getattr(self, "_given", {}):
                    continue
                try:
                    check(getattr(gf, key))
                except ValueError as e:
                    out(f"BAYESIANWIND error: {key}: {e}")
                    good = False
            msg = " is not an optional parameter. Please specify it in the input file (the initial wind)."
            for k in ("XPOS", "YPOS", "VSYS", "INC", "PHI", "OPENANG", "VWIND"):
                if getattr(gw, k) == "-1":
                    out(f"BAYESIANWIND ERROR: {k}" + msg); good = False
            if gw.HTOT == -1:
                out("BAYESIANWIND error: HTOT" + msg); good = False

        if gw.flagGALWIND:
            msg = " is not an optional parameter. Please specify it in the input file."
            for k in ("XPOS", "YPOS", "VSYS", "VDISP", "INC", "PHI", "DENS", "OPENANG", "VWIND"):
                if getattr(gw, k) == "-1":
                    out(f"GALWIND ERROR: {k}" + msg); good = False
            if gw.HTOT == -1:
                out("GALWIND error: HTOT" + msg); good = False

        if self.flagSmooth:
            if self.newbeam is None and self.factor == -1:
                out("SMOOTH error: you need to specify either the new beam (NEWBEAM) or FACTOR.")
                good = False

        if self.flagSmoothSpectral:
            s = self.window_type
            if not math.isfinite(self.window_size) or self.window_size <= 0:
                out(f"SMOOTHSPEC error: WINDOW_SIZE must be positive, not {self.window_size:g}")
                good = False
            elif int(self.window_size) % 2 == 0 and s != "GAUSSIAN":
                out("SMOOTHSPEC error: smoothing window must be an odd number")
                good = False
            if s not in ("HANNING", "HANNING2", "BOXCAR", "TOPHAT", "FLATTOP", "BARTLETT",
                         "WELCH", "BLACKMAN", "GAUSSIAN"):
                out(f"SMOOTHSPEC error: unknown window type {s}")
                good = False

        if self.reduce not in ("true", "false", "spatial", "spectral"):
            out("GENERAL PARAM warning: REDUCE parameter can only be TRUE, FALSE, SPECTRAL or SPATIAL. Defaulting to TRUE.")
            self.reduce = "true"
        if self.getflagReduce() and not (self.flagSmooth or self.flagSmoothSpectral) and self.factor < 1:
            out(f"REDUCE error: FACTOR (the pixels or channels averaged) must be at least 1, not {self.factor:g}.")
            good = False

        if self.ma.maptype not in ("GAUSSIAN", "MOMENT"):     # (MAPS and the maps of 3DFIT, BAYESIAN3DFIT, GALMOD)
            out("MAP warning: MAPTYPE is either MOMENT or GAUSSIAN. Reverting to MOMENT.")
            self.ma.maptype = "MOMENT"

        if self.ma.veldef not in ("radio", "optical", "relativistic", "auto"):
            out(f"MAP warning: unknown velocity definition '{self.ma.veldef}'. Defaulting to 'auto'. ")
            self.ma.veldef = "auto"
        if self.ma.taper not in ("uniform", "hanning1", "hanning2"):
            out(f"MAP warning: unknown tapering '{self.ma.taper}'. Defaulting to 'uniform'. ")
            self.ma.taper = "uniform"

        if self.flagPV:
            try:
                isSingle = float(self.XPOS_PV.replace("d", "")) >= 0 and float(self.YPOS_PV.replace("d", "")) >= 0
            except ValueError:
                isSingle = self.XPOS_PV != "-1" and self.YPOS_PV != "-1"
            isDouble = all(v >= 0 for v in self.P1_PV + self.P2_PV)
            if not isSingle and not isDouble:
                out("PVSLICE error: define slice through either (XPOS_PV,YPOS_PV,PA_PV) or (P1_PV,P2_PV) ")
                good = False
            if not math.isfinite(self.ANTIALIAS) or (int(self.ANTIALIAS) != self.ANTIALIAS and self.ANTIALIAS != 0.5):
                out("PVSLICE error: ANTIALIAS can only be a positive integer or 0.5. Defaulting to 0.5 ")
                self.ANTIALIAS = 0.5

        if gf.GUESSTYPE not in ("LEGACY", "ELLIPSE", "FIT"):
            out(f"GUESSTYPE error: unknown type of initial estimates {gf.GUESSTYPE} (FIT, ELLIPSE or LEGACY).")
            good = False

        if self.galmodtype not in ("MC", "ANALYTIC"):
            out("GALMODTYPE can only be MC or ANALYTIC. Setting to MC.")
            self.galmodtype = "MC"

        if self.ringmode == "CENTERED":
            self.ringmode = "CENTRED"
        if self.ringmode not in ("CENTRED", "NODES", "SHIFTED"):
            out("RINGMODE can only be CENTRED, NODES or SHIFTED. Setting to CENTRED.")
            self.ringmode = "CENTRED"

        if self.threads < 1:
            self.threads = 1

        return good

    # ------------------------------------------------------------------
    # Printing
    # ------------------------------------------------------------------
    def __str__(self):
        s = io.StringIO()
        s.write("\n-------------------------- Parameters -------------------------\n\n")
        printParams(s, self, defaults=False)
        return s.getvalue()

    def printDefaults(self, stream=None, wtask="ALL"):
        stream = stream or sys.stdout
        whichtask = wtask.upper()
        if whichtask not in [t[0] for t in TASKS_LIST]:
            whichtask = "ALL"
        if whichtask == "ALL":
            stream.write("\n---------------------- Default parameters ----------------------\n\n")
        else:
            stream.write(f"\n----------------- Default parameters for {whichtask} -----------------\n\n")
        printParams(stream, Param(), defaults=True, whichtask=whichtask)

    def write_parameterfile(self, fileout, tasks=()):
        """Write the parameter file of to_parfile(tasks) in fileout."""
        with open(fileout, "w") as f:
            f.write(self.to_parfile(tasks))

    def to_parfile(self, tasks=()):
        """A parameter file (a string) with all the parameters of the tasks (names as in keywords, e.g.
        '3DFIT') and the general ones: first those given, as given (FITSFILE, the cube of this run, and
        OUTFOLDER as absolute paths), then the defaults of the others. BBarolo -p runs it again with the
        same parameters (paths in other parameters, e.g. RINGFILE, are relative to the folder of the run,
        written in the header)."""
        import datetime
        given = {k: v for k, v in getattr(self, "_given", {}).items()
                 if k not in ("FITSFILE", "FITSLIST", "OUTFOLDER")}
        uses = {t: {k.name for k in _kw.keywords_for(t)} for t in tasks}
        names = set().union(*uses.values()) if uses else set()
        when = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        lines = [f"// BBarolo parameter file of a run of bbarolo {BBVERSION} ({when}, in {os.getcwd()}):",
                 "// the parameters given, then the defaults of the others" + (f" (tasks: {', '.join(tasks)})"
                                                                              if tasks else "") + ".", ""]
        w = max(len(k.name) for k in _kw.KEYWORDS) + 1
        if self.imageFile:
            lines.append(f"{'FITSFILE':<{w}}{os.path.abspath(self.imageFile)}")
        if self.outFolder:
            lines.append(f"{'OUTFOLDER':<{w}}{os.path.abspath(self.outFolder)}/")
        lines += [f"{k:<{w}}{v}" for k, v in given.items()]
        lines += ["", "// Defaults"]
        d = _default_param()
        for kw in _kw.KEYWORDS:
            # (not the switches of the tasks, nor the keywords that do more than set their value, e.g.
            # THRESHOLD, which also replaces SNRCUT)
            if kw.name in given or kw.flag or kw.also or kw.name in ("FITSFILE", "FITSLIST", "OUTFOLDER"):
                continue
            if "ALL" not in kw.tasks and kw.name not in names:
                continue
            if len({str(_kw._getattr(d, t)) for t in kw.targets}) > 1:
                continue                         # (a default that depends on the task, e.g. VROT of GALWIND)
            v = kw.value(d)
            users = [t for t in tasks if kw.name in uses[t]]
            tdef = {_kw._fmtval(_kw.TASK_DEFAULTS.get(t, {}).get(kw.name, v)) for t in users}
            if len(tdef) > 1:
                continue                         # (e.g. GALMODTYPE: ANALYTIC for BAYESIAN3DFIT, MC for 3DFIT)
            s = tdef.pop() if tdef else _kw._fmtval(v)
            if s.strip() and s != '""':
                lines.append(f"{kw.name:<{w}}{s}")
        return "\n".join(lines) + "\n"

    def createTemplate(self, fname="param.par"):
        m = 12
        with open(fname, "w") as p:
            p.write("// This is a template input file for the 3DFIT task.\n")
            p.write("// Lines beginning with the double-slash or hash and blank \n"
                    "// lines are not read by the program.\n\n")
            def kv(k, v):
                p.write(f"{k:<{m}}{v}\n")
            p.write("// Name of the fitsfile to be modeled.\n"); kv("FITSFILE", "/yourpath/yourfile.fits\n")
            p.write("// Using the 3DFIT utility? Must be true!!\n"); kv("3DFIT", "true\n")
            p.write("// Number of radii to be modeled.\n"); kv("NRADII", "10\n")
            p.write("// Separation between radii in arcsec.\n"); kv("RADSEP", "30\n")
            p.write("// Systemic velocity of the galaxy (in km/s).\n"); kv("VSYS", "132\n")
            p.write("// X-Y coordinates of the galaxy center (in pixel).\n"); kv("XPOS", "256"); kv("YPOS", "256\n")
            p.write("// Initial global values for parameters:\n// Rotation and dispersion velocities (in km/s),\n"
                    "// inclination and position angles [measured\n// anti-clockwise from north] (in degrees),\n"
                    "// height scale of the disk (in arcsec).\n")
            kv("VROT", "100"); kv("VRAD", "0"); kv("VDISP", "10"); kv("INC", "60"); kv("PA", "100"); kv("Z0", "20\n")
            p.write("// Free parameters for the minimization.\n"); kv("FREE", "VROT VDISP INC PA\n")
            p.write(f"// OPTIONAL: Function to be minimized (default is {self.gf.FTYPE}):\n// = 1: chi-squared.\n"
                    "// = 2: |mod-obs|.\n// = 3: |mod-obs|/|mod+obs|.\n"); kv("FTYPE", "1\n")
            p.write(f"// OPTIONAL: Weighting function (default is {self.gf.WFUNC}):\n// = 0: uniform weight.\n"
                    "// = 1: |cos(θ)|.\n// = 2: cos(θ)^2.\n// θ is the azimuthal angle.\n"); kv("WFUNC", "1\n")
            p.write(f"// OPTIONAL: Layer type along z (default is {self.gf.LTYPE}):\n// = 1: gaussian layer.\n"
                    "// = 2: sech2 layer.\n// = 3: exponential layer.\n// = 4: Lorentzian layer.\n// = 5: box layer.;\n")
            kv("LTYPE", "1\n")
            p.write("// OPTIONAL: Number of subcloud in a velocity profile.\n// (default is = total number of channels).\n")
            kv("NV", "60\n")
            p.write(f"// OPTIONAL: Surface density of clouds in the plane of ring (1e20).\n// (default is = {self.gf.CDENS}):\n")
            kv("CDENS", "10\n")
            p.write(f"// OPTIONAL: Tolerance for the minimization (default is {self.gf.TOL}):\n"); kv("TOL", "1E-03\n")
            p.write(f"// OPTIONAL: Using a mask for the minimization (default is {self.MaskType}):\n"); kv("MASK", "SMOOTH\n")
            p.write(f"// OPTIONAL: Normalization type (default is {self.gf.NORM}):\n"); kv("NORM", "AZIM\n")
            p.write(f"// OPTIONAL: Side of the galaxy to be fitted (default is {self.gf.SIDE}):\n"
                    "// = A: Approaching.\n// = R: Receding.\n// = B: Both.\n"); kv("SIDE", "B\n")
            p.write(f"// OPTIONAL: Using a two stages minimization (default is {stringize(self.gf.TWOSTAGE)}):\n"); kv("TWOSTAGE", "false\n")
            p.write(f"// OPTIONAL: Type of regularization to use for second stage (default is {self.gf.REGTYPE}):\n"); kv("REGTYPE", "3\n")
            p.write(f"// OPTIONAL: Enabling error estimation (default is {stringize(self.gf.flagERRORS)}):\n"); kv("flagErrors", "false\n")
            p.write("// OPTIONAL (pure-Python only): GALMOD type, MC (classic) or ANALYTIC (fast, noiseless).\n")
            kv("GALMODTYPE", "MC\n")
            p.write("// OPTIONAL (pure-Python only): rings of the models, CENTRED (annuli centred on the radii),\n"
                    "// NODES (values at the radii) or SHIFTED (BBarolo 1.x).\n")
            kv("RINGMODE", "CENTRED\n")
        print('\n A template parameter input file ("param.par") for BBarolo has been generated.\n')


_DEFAULT = None


def _default_param():
    """A Param with all the default values (cached; do not modify it)."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = Param()
    return _DEFAULT


def _hasattr(obj, path):
    try:
        _kw._getattr(obj, path)
        return True
    except AttributeError:
        return False


def _beamstr(b, none):
    """A beam [bmaj, bmin, bpa] (BEAM, NEWBEAM) as printed in the parameter list; none if not given."""
    if b is None:
        return none
    return f"{b[0]:g} {b[1]:g}" + ("" if b[2] is None else f" {b[2]:g}")


def value_to_string(kw, value):
    """Convert a Python value into the parameter-file string for keyword kw, checking its
    type: booleans become true/false (1/0 for integer keywords), sequences become
    space-separated lists (only for keywords that accept lists), numbers must be numbers."""
    import numpy as np
    name = kw.name
    if isinstance(value, (bool, np.bool_)):
        if kw.kind in ("int", "cdens"):
            return "1" if value else "0"
        if kw.kind in ("float",) or kw.kind in _kw.VECTOR_KINDS:
            raise TypeError(f"{name}: expected a number, not {value!r}")
        return stringize(bool(value))
    if kw.kind == "beam":           # (numbers, or strings such as '0.03d'; a None BPA is not given)
        items = list(np.ravel(np.asarray(value, dtype=object))) if not isinstance(value, str) else [value]
        if items and items[-1] is None:
            items = items[:-1]
        for v in items:
            if isinstance(v, (bool, np.bool_)) or not isinstance(v, (str, int, float, np.integer, np.floating)):
                raise TypeError(f"{name}: expected (bmaj, bmin, bpa), not {value!r}")
        out = " ".join(v.strip() if isinstance(v, str) else _numstr(v) for v in items)
        try:
            _kw.readbeam(out)
        except ValueError as e:
            raise BBaroloError(f"{name}: {e}") from None
        return out
    if isinstance(value, (list, tuple, np.ndarray)):
        flat = np.asarray(value, dtype=object).ravel().tolist() if isinstance(value, np.ndarray) else list(value)
        if kw.kind in _kw.SCALAR_KINDS:
            if len(flat) != 1:
                raise TypeError(f"{name}: expected a single value, not a sequence")
            return value_to_string(kw, flat[0])
        if kw.kind in _kw.VECTOR_KINDS or kw.name in _RING_KEYS:
            for v in flat:
                if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, float, np.integer, np.floating)):
                    raise TypeError(f"{name}: the values of a list must be numbers, not {v!r}")
            return " ".join(_numstr(v) for v in flat)
        raise TypeError(f"{name}: a list of values is not accepted")
    if isinstance(value, (int, float, np.integer, np.floating)):
        if kw.kind in ("flag",):
            if value not in (0, 1):
                raise BBaroloError(f"{name}: expected True or False, not {value!r}")
            return stringize(bool(value))           # (1.0 written as "1.0" was read as false)
        return _numstr(value)
    if isinstance(value, str):
        s = value.strip()
        if kw.kind in _kw.NUMERIC_KINDS:
            try:
                float(s)
            except ValueError:
                raise BBaroloError(f"{name}: expected a number, not {value!r}") from None
        elif kw.kind == "flag" and s.lower() not in ("true", "false", "1", "0"):
            raise BBaroloError(f"{name}: expected True or False, not {value!r}")
        return s
    raise TypeError(f"{name}: unsupported value {value!r}")


def _numstr(v):
    """Number -> string that reads back to the same value."""
    import numpy as np
    if isinstance(v, (bool, np.bool_)):
        return "1" if v else "0"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    return repr(float(v)) if not isinstance(v, np.float32) else str(v)


# --------------------------------------------------------------------------
# BAYESIAN3DFIT: free parameters, priors and options of the sampler (FREE, PRIORS,
# SAMPLEROPTS and RUNOPTS, as written in parameter files)
# --------------------------------------------------------------------------
# ring parameters that can be fitted, the other free parameters, and the parameters with a
# function of radius (Bayesian3DFIT.funcs: VROT arctan, VDISP linear, DENS exponential)
BAYES_RING_PARS = ("vrot", "vdisp", "dens", "z0", "inc", "phi", "xpos", "ypos", "vsys", "vrad", "vvert", "dvdz",
                   "zcyl")
BAYES_OTHER_PARS = ("sigma", "fracerr", "radmax")
BAYES_FUNC_PARS = ("vrot", "vdisp", "dens")
_BAYES_ALIASES = {"pa": "phi", "disp": "vdisp"}


def bayes_name(name, aliases=None):
    """The name of a free parameter (or of a prior) as Bayesian3DFIT uses it: lower case,
    PA -> phi and DISP -> vdisp, e.g. 'PA_SINGLE' -> 'phi_single', 'VROT' -> 'vrot' (aliases: other
    aliases, e.g. those of BayesianWind)."""
    al = _BAYES_ALIASES if aliases is None else aliases
    base, sep, suffix = str(name).strip().lower().partition("_")
    return al.get(base, base) + sep + suffix


def bayes_freepar(free, funcs=BAYES_FUNC_PARS, ring_pars=BAYES_RING_PARS, other_pars=BAYES_OTHER_PARS, aliases=None):
    """The free parameters of BAYESIAN3DFIT from FREE (as in a parameter file), as a list for
    Bayesian3DFIT.run: 'VROT VDISP PA_SINGLE INC_SINGLE RADMAX' -> ['vrot', 'vdisp',
    'phi_single', 'inc_single', 'radmax']. A ring parameter (VROT, VDISP, DENS, Z0, INC, PA,
    XPOS, YPOS, VSYS, VRAD, VVERT, DVDZ, ZCYL) has one value per ring, NAME_SINGLE one value
    for all the rings, NAME_FUNC a function of radius (funcs: the parameters with a function,
    by default VROT, VDISP and DENS); SIGMA (noise), FRACERR (fractional model error) and
    RADMAX (outer radius) are single values. Names are case-insensitive; PHI is the same as
    PA and DISP as VDISP. Raises ValueError."""
    out, seen = [], set()
    for tok in re.split(r"[\s,;]+", str(free).strip()):
        if not tok:
            continue
        base, _, suffix = bayes_name(tok, aliases).partition("_")
        if base in other_pars:
            if suffix:
                raise ValueError(f"{tok}: {base.upper()} is a single parameter (no _{suffix.upper()})")
            name = base
        elif base in ring_pars:
            if suffix not in ("", "single", "func"):
                raise ValueError(f"unknown free parameter {tok} (NAME, NAME_SINGLE or NAME_FUNC)")
            if suffix == "func" and base not in funcs:
                raise ValueError(f"{tok}: {base.upper()} has no function of radius (only "
                                 f"{', '.join(sorted(f.upper() for f in funcs))})")
            name = base + ("_" + suffix if suffix else "")
        else:
            raise ValueError(f"unknown free parameter {tok}")
        if base in seen:
            raise ValueError(f"{base.upper()} is given more than once")
        seen.add(base)
        out.append(name)
    if not out:
        raise ValueError("no free parameters")
    return out


# NAME=(distribution,arguments) or NAME=distribution(arguments)
_PRIOR_RE = re.compile(r"([A-Za-z]\w*)\s*=\s*(?:([A-Za-z_]\w*)\s*)?\(([^()]*)\)")


def _check_gap(gap, what):
    g = gap.strip(" \t\n,;")
    if g:
        raise ValueError(f"cannot read '{g}' (the format is {what})")


def _number(v, where):
    v = v.strip()
    if v.lower() == "none":
        return None
    try:
        return float(v)
    except ValueError:
        raise ValueError(f"{where}: '{v}' is not a number") from None


def bayes_priors(priors, ring_pars=BAYES_RING_PARS, other_pars=BAYES_OTHER_PARS, func_pars=BAYES_FUNC_PARS,
                 aliases=None):
    """The priors of BAYESIAN3DFIT from PRIORS (as in a parameter file), as entries of
    Bayesian3DFIT.priors:

        'VROT=(uniform,loc=0,scale=300) INC=(norm,60,5)'  ->
        {'vrot': {'name': 'uniform', 'loc': 0.0, 'scale': 300.0},
         'inc': {'name': 'norm', 'loc': 60.0, 'scale': 5.0}}

    Each entry is NAME=(distribution,arguments) (or NAME=distribution(arguments)): a
    continuous distribution of scipy.stats and its arguments, as key=value or by position
    (the shape parameters of the distribution, then loc and scale, as in scipy.stats).
    NAME is a free parameter (as in FREE; NAME_SINGLE is the same as NAME), or a parameter
    of a function of radius (VROT_P1, VROT_P2, VDISP_P1, ...). Entries are separated by
    spaces (or commas); names are case-insensitive. For SIGMA, a or b = none are 0.5 and
    1000 times the noise (see Bayesian3DFIT._prior_kwargs). Raises ValueError."""
    import numpy as np
    from scipy import stats
    s = str(priors).strip()
    fmt = "NAME=(distribution,key=value,...)"
    out = {}
    pos = 0
    for m in _PRIOR_RE.finditer(s):
        _check_gap(s[pos:m.start()], fmt)
        pos = m.end()
        label = m.group(1)
        base, _, suffix = bayes_name(label, aliases).partition("_")
        if base in ring_pars and suffix in ("", "single"):
            key = base
        elif base in other_pars and not suffix:
            key = base
        elif base in func_pars and re.fullmatch(r"p\d+", suffix):
            key = f"{base}_{suffix}"
        else:
            raise ValueError(f"{label}: unknown parameter")
        items = [x.strip() for x in m.group(3).split(",")]
        if m.group(2):
            items.insert(0, m.group(2))
        dname = items[0].lower()
        if not dname:
            raise ValueError(f"{label}: no distribution")
        dist = getattr(stats, dname, None)
        if not isinstance(dist, stats.rv_continuous):
            raise ValueError(f"{label}: unknown distribution '{items[0]}' (a continuous distribution of scipy.stats, "
                             "e.g. uniform, norm, loguniform, truncnorm, halfcauchy)")
        names = [x.strip() for x in (dist.shapes or "").split(",") if x.strip()] + ["loc", "scale"]
        args, npos = {}, 0
        for it in items[1:]:
            if not it:
                continue
            k, eq, v = it.partition("=")
            if eq:
                k = k.strip().lower()
                if k in args:
                    raise ValueError(f"{label}: {k} is given more than once")
                args[k] = _number(v, label)
            else:
                if len(args) > npos:
                    raise ValueError(f"{label}: values by position must come before key=value")
                if npos >= len(names):
                    raise ValueError(f"{label}: too many values for {dname} (arguments: {', '.join(names)})")
                args[names[npos]] = _number(it, label)
                npos += 1
        desc = f"{dname}({', '.join(f'{k}={v:g}' if v is not None else f'{k}=None' for k, v in args.items())})"
        if any(v is None for v in args.values()):
            if key != "sigma" or any(k not in ("a", "b", "scale") for k, v in args.items() if v is None):
                raise ValueError(f"{label}: none is accepted only for a, b or scale of SIGMA")
        else:
            try:
                ok = bool(np.isfinite(dist(**args).ppf(0.5)))
            except Exception as e:
                raise ValueError(f"{label}: {desc} is not valid ({e})") from None
            if not ok:
                raise ValueError(f"{label}: {desc} is not a valid distribution")
        out[key] = dict(name=dname, **args)
    _check_gap(s[pos:], fmt)
    return out


def bayes_value_string(name, value):
    """A Python value of FREE, PRIORS, SAMPLER, SAMPLEROPTS, RUNOPTS or LIKEOPTS (BAYESIAN3DFIT) as it is written in a
    parameter file: FREE a list of names (['vrot', 'inc_single'] -> 'vrot inc_single'); PRIORS a dict as
    Bayesian3DFIT.priors, {name: {'name': distribution, argument: value}} or {name: (distribution, values by
    position)}; the options a dict {key: value} (a tuple: values separated by commas). A string is returned as it
    is. Raises TypeError."""
    import numpy as np

    def opt(v):
        if v is None:
            return "none"
        if isinstance(v, (bool, np.bool_)):
            return "true" if v else "false"
        if isinstance(v, dict):
            return "(" + ",".join(f"{k}={opt(x)}" for k, x in v.items()) + ")"
        if isinstance(v, (list, tuple, np.ndarray)):
            return ",".join(opt(x) for x in v)
        if isinstance(v, (int, float, np.integer, np.floating)):
            return _numstr(v)
        return str(v)
    if isinstance(value, str):
        return value
    if name == "FREE" and isinstance(value, (list, tuple, np.ndarray)) and all(isinstance(v, str) for v in value):
        return " ".join(value)
    if name in ("PRIORS", "SAMPLEROPTS", "RUNOPTS", "LIKEOPTS") and isinstance(value, dict):
        if name != "PRIORS":
            return " ".join(f"{k}={opt(v)}" for k, v in value.items())
        items = []
        for k, d in value.items():
            if isinstance(d, dict) and "name" in d:
                args = [str(d["name"])] + [f"{a}={opt(v)}" for a, v in d.items() if a != "name"]
            elif isinstance(d, (list, tuple)) and d and isinstance(d[0], str):
                args = [d[0]] + [opt(v) for v in d[1:]]
            else:
                raise TypeError(f"BAYESIAN3DFIT: PRIORS: {k}: give {{'name': distribution, ...}} or (distribution, "
                                f"values...), not {d!r}")
            items.append(f"{k}=({','.join(args)})")
        return " ".join(items)
    raise TypeError(f"BAYESIAN3DFIT: {name}: expected a string as in a parameter file" + (
        ", or a list of names" if name == "FREE" else ", or a dict" if name != "SAMPLER" else "") + f", not {value!r}")


def _optvalue(v, none=True):
    if len(v) > 1 and v[0] + v[-1] in ("()", "[]"):
        v = v[1:-1]
        if "=" in v:                                    # (key=value, ...): a dict
            return bayes_options(v, none)
    if "," in v:
        return tuple(_optvalue(x, none) for x in v.split(",") if x)
    lv = v.lower()
    if lv in ("true", "false"):
        return lv == "true"
    if lv == "none" and none:
        return None
    try:
        x = float(v)
    except ValueError:
        return v
    if re.fullmatch(r"[+-]?\d+", v) or ("." not in v and "e" in lv and x.is_integer() and abs(x) < 2 ** 63):
        return int(x)
    return x


def bayes_options(opts, none=True):
    """Options of the sampler (SAMPLEROPTS), of the sampling (RUNOPTS) or of the likelihood
    (LIKEOPTS) from 'key=value ...' (as in a parameter file): 'nlive=200 bound=multi
    dynamic=false' -> {'nlive': 200, 'bound': 'multi', 'dynamic': False}. Values are
    integers, numbers, true/false, none (None; a string with none=False), strings, or several
    of them separated by commas (a tuple: 'dilation=1,3' -> (1, 3)), or key=value pairs in brackets (a dict:
    'lambdas=(vrot=3000, vdisp=1000)' -> {'vrot': 3000, 'vdisp': 1000}). Raises ValueError."""
    s = re.sub(r"\s*([=,])\s*", r"\1", str(opts).strip())
    toks, start, depth = [], 0, 0                    # split at spaces, ';' and ',key=', outside brackets
    for i, c in enumerate(s):
        if c in "([":
            depth += 1
        elif c in ")]":
            depth = max(depth - 1, 0)
        elif depth == 0 and (c.isspace() or c == ";" or (c == "," and re.match(r"[A-Za-z_]\w*=", s[i + 1:]))):
            toks.append(s[start:i])
            start = i + 1
    toks.append(s[start:])
    out = {}
    for tok in toks:
        if not tok:
            continue
        k, eq, v = tok.partition("=")
        if not eq or not k or not v:
            raise ValueError(f"cannot read '{tok}' (the format is key=value)")
        if not k.isidentifier():
            raise ValueError(f"'{k}' is not the name of an option")
        out[k.lower()] = _optvalue(v, none)
    return out


# Options of the likelihood (Bayesian3DFIT.run(like_kwargs=...), LIKEOPTS) and their defaults.
# Those of LIKE_SETUP set up the likelihood; sigma, fracerr and lambdas are passed to the
# log-likelihood at each call (None: not given).
LIKE_OPTIONS = dict(likemask="dilated", dilation=(1., 3), noisecorr="measured", densprof_smooth=0.1, bbres=False,
                    radmax="edge", radmax_width=None, sigma=None, fracerr=None, lambdas=None)
LIKE_SETUP = ("likemask", "dilation", "noisecorr", "densprof_smooth", "bbres", "radmax", "radmax_width")
_LIKE_ALIASES = {"usebbres": "bbres", "likemask_dilation": "dilation"}


def bayes_like_option(name, value):
    """An option of the likelihood of Bayesian3DFIT (run(like_kwargs=...), LIKEOPTS), checked:
    returns (name, value), the name in lower case; None if name is not an option. Raises
    ValueError for a wrong value. None gives the default.

      likemask         the voxels of the likelihood: 'dilated' (default: the data in the mask
                       dilated by dilation) or 'none' (all the voxels of the region)
      dilation         (radius in beams, channels) of the dilation of the mask, default (1, 3)
      noisecorr        the correlation of the noise: 'measured' (default), 'beam' (the beam
                       area), or the number of voxels per independent measurement (1:
                       independent voxels); the log-likelihood is multiplied by its inverse
      densprof_smooth  smoothness of the deconvolved density profile (default 0.1; 0: none)
      bbres            True: the residuals of BBarolo's 3DFIT instead of the Gaussian
                       likelihood (also usebbres)
      radmax           what the free parameter RADMAX is: 'edge' (default: the outer edge of
                       the disk, with the rings fixed: no emission beyond it), 'taper' (a
                       Gaussian edge of width radmax_width) or 'ring' (the radius of the last
                       ring, which moves, as in the original module)
      radmax_width     the width (sigma, arcsec) of the edge with radmax='taper' (default a
                       quarter of the beam FWHM)
      sigma            fixed noise per voxel (default: the measured noise rms)
      fracerr          fixed fractional model error (default 0)
      lambdas          dict of smoothness penalties of the ring parameters, e.g. {'vrot': 1}
                       (LIKEOPTS: lambdas=(vrot=1, vdisp=0.5))"""
    import numpy as np
    key = _LIKE_ALIASES.get(str(name).strip().lower(), str(name).strip().lower())
    if key not in LIKE_OPTIONS:
        return None
    if value is None or (key != "likemask" and isinstance(value, str) and value.strip().lower() == "none"):
        return key, LIKE_OPTIONS[key]

    def number(x, positive=False):
        try:
            if isinstance(x, (bool, np.bool_)):
                raise TypeError
            x = float(x)
        except (TypeError, ValueError):
            raise ValueError(f"{key}: expected a number, not {value!r}") from None
        if not np.isfinite(x) or x < 0 or (positive and x == 0):
            raise ValueError(f"{key}: expected a number {'>' if positive else '>='} 0, not {value!r}")
        return x
    if key == "likemask":
        v = str(value).strip().lower()
        if v not in ("dilated", "none"):
            raise ValueError(f"likemask: 'dilated' or 'none', not {value!r}")
        return key, v
    if key == "dilation":
        if isinstance(value, str):
            value = _optvalue(re.sub(r"\s+", "", value))
        try:
            nb, nc = value
        except (TypeError, ValueError):
            raise ValueError(f"dilation: (radius in beams, channels), not {value!r}") from None
        return key, (number(nb), int(round(number(nc))))
    if key == "noisecorr":
        if isinstance(value, str) and value.strip().lower() in ("measured", "beam"):
            return key, value.strip().lower()
        try:
            return key, number(value, positive=True)
        except ValueError:
            raise ValueError(f"noisecorr: 'measured', 'beam' or a number > 0, not {value!r}") from None
    if key == "bbres":
        if isinstance(value, str) and value.strip().lower() in ("true", "false", "1", "0"):
            return key, value.strip().lower() in ("true", "1")
        if isinstance(value, (bool, np.bool_, int, np.integer)) and value in (0, 1):
            return key, bool(value)
        raise ValueError(f"bbres: true or false, not {value!r}")
    if key == "radmax":
        v = str(value).strip().lower()
        if v not in ("ring", "edge", "taper"):
            raise ValueError(f"radmax: 'ring', 'edge' or 'taper', not {value!r}")
        return key, v
    if key == "lambdas":
        if isinstance(value, str):
            value = _optvalue(value.strip())
        if not isinstance(value, dict):
            raise ValueError(f"lambdas: smoothness penalties of ring parameters, e.g. (vrot=3000, vdisp=1000) or "
                             f"{{'vrot': 3000}}, not {value!r}")
        out = {}
        for k, v in value.items():
            name = bayes_name(k)
            if name not in BAYES_RING_PARS:
                raise ValueError(f"lambdas: '{k}' is not a ring parameter (" + ", ".join(BAYES_RING_PARS) + ")")
            try:
                out[name] = number(v)
            except ValueError:
                raise ValueError(f"lambdas: {k}: expected a number >= 0, not {v!r}") from None
        return key, out
    return key, number(value, positive=(key == "sigma"))         # densprof_smooth, sigma, fracerr, radmax_width


def bayes_likeopts(opts):
    """The options of the likelihood of BAYESIAN3DFIT from LIKEOPTS (as in a parameter file):
    'likemask=none noisecorr=beam dilation=1.5,4' -> {'likemask': 'none', 'noisecorr': 'beam',
    'dilation': (1.5, 4)}; 'lambdas=(vrot=3000, vdisp=1000)' -> {'lambdas': {'vrot': 3000.,
    'vdisp': 1000.}}. The options are those of bayes_like_option. Raises ValueError."""
    out = {}
    for k, v in bayes_options(opts, none=False).items():
        kv = bayes_like_option(k, v)
        if kv is None:
            raise ValueError(f"unknown option '{k}' (options: " + ", ".join(LIKE_OPTIONS) + ")")
        out[kv[0]] = kv[1]
    return out


# Ring parameters: a list of numbers gives one value per ring
_RING_KEYS = ("RADII", "VROT", "VRAD", "Z0", "VVERT", "DVDZ", "ZCYL", "VDISP", "XPOS", "YPOS", "VSYS", "INC",
              "PA", "DENS")


def _fmt(v):
    if isinstance(v, bool):
        return stringize(v)
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def recordParam(stream, name, desc, value):
    width = 60
    widthText = width - len(name)
    stream.write(f"{desc:.<{widthText}}{name}  =  {_fmt(value)}\n")


def printParams(Str, p, defaults=False, whichtask="ALL"):
    """Port of printParams(): writes a formatted parameter list on stream Str."""
    isAll = whichtask == "ALL" and defaults
    gf, gm, gw, se, ma = p.gf, p.gm, p.gw, p.se, p.ma
    R = lambda name, desc, val: recordParam(Str, name, desc, val)

    from .bbutils import shortpath                 # (paths relative to the current folder, on screen)
    if defaults:
        R("[FITSFILE]", "FITS file to be analysed", p.imageFile)
        R("[FITSLIST]", "List of FITS files to be analysed", p.imageList)
    else:
        if p.imageList == "NONE": R("[FITSFILE]", "FITS file to be analysed", shortpath(p.imageFile))
        else: R("[FITSLIST]", "List of FITS files to be analysed", shortpath(p.imageList))
    R("[THREADS]", "Number of threads", p.threads)
    R("[PLOTS]", "Producing output plots?", stringize(p.plots))
    R("[SHOWBAR]", "Showing progress bars?", stringize(p.showbar))
    R("[VERBOSE]", "Output messages (0 none, 1 main, 2 all)?", int(p.verbose))
    if p.outFolder != "" or defaults: R("[OUTFOLDER]", "Directory where outputs are written", shortpath(p.outFolder))
    if p.outPrefix != "" or defaults: R("[OUTPREFIX]", "Prefix for output files", p.outPrefix)
    R("[LOGFILE]", "Redirect output messages to a file?", stringize(p.logFile))
    R("[flagRobustStats]", "Using robust statistics?", stringize(p.flagRobustStats))
    R("[FLUXCONVERT]", "Whether to convert flux to Jy?", stringize(p.fluxConvert))
    if p.checkCube or (defaults and isAll): R("[checkCube]", "Check cube for bad channels/rows/columns?", p.checkCube)
    if defaults or p.beam is not None:
        R("[BEAM]", "Beam of the data (arcsec, arcsec, deg)", _beamstr(p.beam, "header"))
    if p.debug: R("[DEBUG]", "Debugging mode?", stringize(p.debug))
    Str.write("-----------------\n")

    if isAll or p.flagStats or (defaults and whichtask == "STATS"):
        R("[STATS]", "Calculate and print statistics?", stringize(p.flagStats))
        R("[ITERNOISE]", "   Estimating noise with iterative algorithm?", stringize(se.iternoise))
    if isAll or p.contsub or (defaults and whichtask == "CONTSUB"):
        R("[CONTSUB]", "Subtract continuum from datacube?", stringize(p.contsub))
        R("[EXCLWIND]", "   Channel window(s) to exclude for continuum fit", p.exclude_windows)
        R("[CONTORDER]", "   Order of polynomial for continuum fit", p.cont_order)
    if isAll or p.makeMask or (defaults and whichtask == "MAKEMASK"):
        R("[MAKEMASK]", "Writing the mask to a fitsfile", stringize(p.makeMask))
        R("[MASK]", "   Type of mask", p.MaskType)
    if isAll or se.flagSearch or (defaults and whichtask == "SEARCH"):
        R("[SEARCH]", "Searching for sources in cube?", stringize(se.flagSearch))
        R("[SEARCHTYPE]", "   Type of searching performed", se.searchType)
        R("[SORTSOURCES]", "   Sorting detections based on?", se.sortsrcs)
        R("[ITERNOISE]", "   Estimating noise with iterative algorithm?", stringize(se.iternoise))
        R("[CUBELETS]", "   Writing sub-cubes of detections?", stringize(se.cubelets))
        if se.cubelets or defaults: R("[EDGES]", "     Number of pixels at the edges of cubelets?", se.edges)
        R("[PBCORR]", "   Correct flux for primary beam attenuation?", se.pbcorr)
        if defaults:
            R("[threshold]", "   Detection Threshold", se.threshold)
            R("[snrCut]", "   SNR Threshold (in sigma)", se.snrCut)
        else:
            if se.UserThreshold: R("[threshold]", "   Detection Threshold", se.threshold)
            else: R("[snrCut]", "   SNR Threshold (in sigma)", se.snrCut)
        R("[flagGrowth]", "   Growing objects after detection?", stringize(se.flagGrowth))
        if se.flagGrowth or defaults:
            if defaults:
                R("[growthThreshold]", "   Threshold for growth", se.growthThreshold)
                R("[growthCut]", "   SNR Threshold for growth", se.growthCut)
            else:
                if se.flagUserGrowthT: R("[growthThreshold]", "   Threshold for growth", se.growthThreshold)
                else: R("[growthCut]", "   SNR Threshold for growth", se.growthCut)
        R("[minPix]", "   Minimum # Pixels in a detection", se.minPix)
        R("[minChannels]", "   Minimum # Channels in a detection", se.minChannels)
        R("[minVoxels]", "   Minimum # Voxels in a detection", se.minVoxels)
        R("[maxChannels]", "   Maximum # Channels in a detection", se.maxChannels)
        R("[maxAngsize]", "   Max angular size of a detection in arcmin", se.maxAngSize)
        R("[flagAdjacent]", "   Using Adjacent-pixel criterion?", stringize(se.flagAdjacent))
        if not se.flagAdjacent or defaults:
            R("[threshSpatial]", "   Max. spatial separation for merging", se.threshSpatial)
        R("[threshVelocity]", "   Max. velocity separation for merging", se.threshVelocity)
        R("[RejectBeforeMerge]", "   Reject objects before merging?", stringize(se.RejectBeforeMerge))
        R("[TwoStageMerging]", "   Merge objects in two stages?", stringize(se.TwoStageMerging))
    if isAll or p.flagSmooth or (defaults and whichtask == "SMOOTH"):
        R("[SMOOTH]", "Smoothing the datacube?", stringize(p.flagSmooth))
        box = " ".join(str(b) for b in p.BOX if b != -1) or "NONE"
        R("[BOX]", "   Sub-region to be used?", box)
        if defaults or p.newbeam is not None:
            R("[NEWBEAM]", "   New beam (arcsec, arcsec, deg)", _beamstr(p.newbeam, "FACTOR x old"))
        if defaults or p.newbeam is None:
            R("[FACTOR]", "   New beam factor (times old beam)", p.factor)
        R("[FFT]", "   Using FFT for convolution?", stringize(p.flagFFT))
        R("[REDUCE]", "   Averaging pixels to reduce datacube?", p.reduce)
    if isAll or p.flagSmoothSpectral or (defaults and whichtask == "SMOOTHSPEC"):
        R("[SMOOTHSPEC]", "Spectral smoothing of the datacube?", stringize(p.flagSmoothSpectral))
        R("[WINDOW_TYPE]", "   Type of smoothing window ", p.window_type)
        R("[WINDOW_SIZE]", "   Size of smoothing window (channels)", p.window_size)
        R("[REDUCE]", "   Averaging channels to reducing datacube?", p.reduce)

    isGalfit = isAll or gf.flagGALFIT or (defaults and whichtask == "3DFIT")
    isBayes = isAll or gf.flagBAYESIAN or (defaults and whichtask == "BAYESIAN3DFIT")
    isGalmod = isAll or gm.flagGALMOD or (defaults and whichtask == "GALMOD")
    isSpacepar = isAll or p.flagSpace or (defaults and whichtask == "SPACEPAR")
    if isGalfit or isBayes or isGalmod or isSpacepar:
        if isGalmod: R("[GALMOD]", "Writing a 3D model?", stringize(gm.flagGALMOD))
        if isGalfit: R("[3DFIT]", "Fitting a 3D model to the datacube?", stringize(gf.flagGALFIT))
        if isBayes: R("[BAYESIAN3DFIT]", "Fitting a 3D model by nested sampling?", stringize(gf.flagBAYESIAN))
        R("[REDSHIFT]", "   Redshift of the galaxy?", gf.REDSHIFT)
        if gf.RESTWAVE[0] != -1 or defaults: R("[RESTWAVE]", "   Transition wavelength at rest?", " ".join(f"{v:.5f}" for v in gf.RESTWAVE))
        if gf.RESTFREQ[0] != -1 or defaults: R("[RESTFREQ]", "   Transition frequency at rest?", " ".join(f"{v:.2f}" for v in gf.RESTFREQ))
        if len(gf.RELINT) > 1 or defaults: R("[RELINT]", "   Relative intensities of lines?", " ".join(f"{v:.2f}" for v in gf.RELINT))
        R("[RINGFILE]", "   A BB output file with rings", gf.ringfile)
        R("[SPECRES]", "   Spectral resolution in sigma (channels)", p.specres if p.specres != -1 else p.linear)
        if defaults or gf.RADII == "-1":
            if defaults: R("[RADII]", "   Radii for rings", gf.RADII)
            R("[NRADII]", "   Number of radii", gf.NRADII)
            R("[RADSEP]", "   Separation between radii (arcsec)", gf.RADSEP)
        else:
            R("[RADII]", "   Radii for rings", gf.RADII)
        R("[XPOS]", "   X center of the galaxy (pixel)", gf.XPOS)
        R("[YPOS]", "   Y center of the galaxy (pixel)", gf.YPOS)
        R("[VSYS]", "   Systemic velocity of the galaxy (km/s)", gf.VSYS)
        R("[VROT]", "   Initial rotation velocity (km/s)", gf.VROT)
        if isGalfit: R("[DELTAVROT]", "   Max rot. velocity variation from VROT (km/s)", gf.DELTAVROT)
        R("[VRAD]", "   Initial radial velocity (km/s)", gf.VRAD)
        R("[VDISP]", "   Initial velocity dispersion (km/s)", gf.VDISP)
        if isGalfit:
            R("[MINVDISP]", "   Minimum acceptable velocity dispersion (km/s)", gf.MINVDISP)
            R("[MAXVDISP]", "   Maximum acceptable velocity dispersion (km/s)", gf.MAXVDISP)
        R("[INC]", "   Initial inclination (degrees)", gf.INC)
        if isGalfit: R("[DELTAINC]", "   Max inclination variation from INC (degrees)", gf.DELTAINC)
        R("[PA]", "   Initial position angle (degrees)", gf.PHI)
        if isGalfit: R("[DELTAPA]", "   Max position angle variation from PA (degrees)", gf.DELTAPHI)
        R("[Z0]", "   Scale height of the disk (arcsec)", gf.Z0)
        R("[DENS]", "   Gas column density (atoms/cm2)", gf.DENS)
        if isGalmod:
            R("[DENSFLUX]", "   Whether DENS is integrated flux / arcs2", stringize(gf.DENSFLUX))
            R("[VVERT]", "   Vertical velocity (km/s)", gm.VVERT)
            R("[DVDZ]", "   Vertical gradient of rotation velocity (km/s/arcs)", gm.DVDZ)
            R("[ZCYL]", "   Height where the gradient begins (arcs)", gm.ZCYL)
        if isGalfit or isBayes:
            free = gf.FREE if (isGalfit or "FREE" in getattr(p, "_given", {})) else "(required)"
            R("[FREE]", "   Parameters to be minimized", free)
        R("[MASK]", "   Type of mask", p.MaskType)
        if isGalfit or isGalmod or isSpacepar: R("[NORM]", "   Type of normalization", gf.NORM)
        if isGalfit: R("[SIDE]", "   Side of the galaxy to be used", gf.SIDE)
        typ = {1: "gaussian", 2: "sech2", 3: "exponential", 4: "Lorentzian", 5: "box"}.get(gf.LTYPE, "")
        R("[LTYPE]", "   Layer type along z direction", typ)
        gmt = p.galmodtype
        if (defaults and whichtask == "BAYESIAN3DFIT") or \
                (not defaults and gf.flagBAYESIAN and "GALMODTYPE" not in getattr(p, "_given", {})):
            gmt = "ANALYTIC"                    # the default of BAYESIAN3DFIT (keywords.TASK_DEFAULTS)
        R("[GALMODTYPE]", "   Type of model (MC or ANALYTIC)", gmt)
        R("[RINGMODE]", "   Ring convention (CENTRED, NODES, SHIFTED)", p.ringmode)
        R("[EMPTY]", "   Whether to leave empty inner part", stringize(gf.EMPTY))
        if isGalfit:
            typ = {1: "chi-squared", 2: "|m-o|", 3: "|m-o|/|m+o|", 4: "(m-o)^2"}.get(gf.FTYPE, "")
            R("[FTYPE]", "   Residuals to minimize", typ)
            typ = {0: "uniform", 1: "|cos(θ)|", 2: "cos(θ)^2", -1: "|sin(θ)|", -2: "sin(θ)^2"}.get(gf.WFUNC, "")
            R("[WFUNC]", "   Weighting function", typ)
            R("[BWEIGHT]", "   Weight for blank pixels", gf.BWEIGHT)
            R("[TOL]", "   Minimization tolerance", gf.TOL)
            R("[TWOSTAGE]", "   Two stages minimization?", stringize(gf.TWOSTAGE))
            if gf.TWOSTAGE or defaults: R("[REGTYPE]", "     Degree of polynomial fitting angles?", gf.REGTYPE)
            R("[FLAGERRORS]", "   Estimating errors?", stringize(gf.flagERRORS))
            R("[ADRIFT]", "   Computing asymmetric drift correction?", stringize(gf.flagADRIFT))
            if gf.flagADRIFT or defaults:
                R("[ADRIFTPOL1]", "     Polynomial for VDISP2 regularisation?", gf.ADRIFTPOL1)
                R("[ADRIFTPOL2]", "     Polynomial for SIGMA regularisation?", gf.ADRIFTPOL2)
            R("[REVERSE]", "   Using reverse-cumulative fitting?", gf.REVERSE)
            R("[NORMALCUBE]", "   Normalizing cube to help convergence?", stringize(gf.NORMALCUBE))
            R("[BADOUT]", "   Write unconverged rings in output (with flag)?", stringize(gf.flagBADOUT))
            R("[VELDEF]", "   Definition for velocity conversion?", ma.veldef)
            R("[PLOTMASK]", "   Overlaying mask to output plots?", stringize(gf.PLOTMASK))
            R("[PLOTMINCON]", "   Minimum flux contour for output plots", gf.PLOTMINCON)
        if isBayes:
            R("[SAMPLER]", "   Nested sampler", gf.SAMPLER)
            R("[PRIORS]", "   Priors of the free parameters", gf.PRIORS or "default")
            R("[SAMPLEROPTS]", "   Options of the sampler", gf.SAMPLEROPTS or "default")
            R("[RUNOPTS]", "   Options of the sampling", gf.RUNOPTS or "default")
            R("[LIKEOPTS]", "   Options of the likelihood", gf.LIKEOPTS or "default")
        if isGalfit or isGalmod or isSpacepar: R("[NOISERMS]", "   RMS noise to add to the model", gf.NOISERMS)
        if isSpacepar:
            R("[SPACEPAR]", "Full parameter space for a pair of parameters", stringize(p.flagSpace))
            R("[P1]", "   First parameter to explore", p.P1)
            R("[P2]", "   Second parameter to explore", p.P2)
            R("[P1PAR]", "   Range and step for P1", " ".join(f"{v:.2f}" for v in p.P1p))
            R("[P2PAR]", "   Range and step for P2", " ".join(f"{v:.2f}" for v in p.P2p))

    if isAll or gw.flagGALWIND or (defaults and whichtask == "GALWIND"):
        R("[GALWIND]", "Generating a 3D datacube with a wind model?", stringize(gw.flagGALWIND))
        R("[VWIND]", "   Radial velocity of the wind (km/s)", gw.VWIND)
        R("[VROT]", "   Azimuthal velocity of the wind (km/s)", gw.VROT)
        R("[OPENANG]", "   Wind opening angle (degrees)", gw.OPENANG)
        R("[HTOT]", "   Wind maximum distance (arcsec)", gw.HTOT)
        R("[XPOS]", "   X center of the galaxy (pixel)", gw.XPOS)
        R("[YPOS]", "   Y center of the galaxy (pixel)", gw.YPOS)
        R("[VSYS]", "   Systemic velocity of the galaxy (km/s)", gw.VSYS)
        R("[VDISP]", "   Global velocity dispersion (km/s)", gw.VDISP)
        R("[INC]", "   Global inclination (degrees)", gw.INC)
        R("[PA]", "   Global position angle (degrees)", gw.PHI)
        R("[DENS]", "   Global column density of gas (atoms/cm2)", gw.DENS)
        R("[NTOT]", "   Number of layers/cylinder for each cone", gw.NTOT)
        R("[DENSTYPE]", "   How to distribute density in layers", gw.DENSTYPE)
        R("[WTYPE]", "   Cylindrical or spherical model", gw.WTYPE)
        R("[GALMODTYPE]", "   Type of model (MC or ANALYTIC)", p.galmodtype)
        R("[SM]", "   Whether to smooth the model", stringize(gf.SM))

    if isAll or p.flagRing or (defaults and whichtask == "2DFIT"):
        R("[2DFIT]", "Fitting velocity field with a ring model?", stringize(p.flagRing))
        if defaults or gf.RADII == "-1":
            if defaults: R("[RADII]", "   Radii for rings", gf.RADII)
            R("[NRADII]", "   Number of radii", gf.NRADII)
            R("[RADSEP]", "   Separation between radii (arcsec)", gf.RADSEP)
        else:
            R("[RADII]", "   Radii for rings", gf.RADII)
        R("[XPOS]", "   X center of the galaxy (pixel)", gf.XPOS)
        R("[YPOS]", "   Y center of the galaxy (pixel)", gf.YPOS)
        R("[VSYS]", "   Systemic velocity of the galaxy (km/s)", gf.VSYS)
        R("[VROT]", "   Rotation velocity (km/s)", gf.VROT)
        R("[VRAD]", "   Radial velocity (km/s)", gf.VRAD)
        R("[INC]", "   Inclination angle (degrees)", gf.INC)
        R("[PA]", "   Position angle (degrees)", gf.PHI)
        R("[FREE]", "   Parameters to be fit", gf.FREE)
        R("[MASK]", "   Type of mask for velocity map", p.MaskType)
        R("[SIDE]", "   Side of the galaxy to be used", gf.SIDE)
        typ = {0: "uniform", 1: "|cos(θ)|", 2: "cos(θ)^2", -1: "|sin(θ)|", -2: "sin(θ)^2"}.get(gf.WFUNC, "")
        R("[WFUNC]", "   Weighting function", typ)
        R("[FREEANGLE]", "   Angle around the minor axis left out (deg)", gf.FREEANGLE)

    if isAll or p.flagEllProf or (defaults and whichtask == "ELLPROF"):
        R("[ELLPROF]", "Deriving radial intensity profile?", stringize(p.flagEllProf))
        if defaults or gf.RADII == "-1":
            if defaults: R("[RADII]", "   Radii for rings", gf.RADII)
            R("[NRADII]", "   Number of radii", gf.NRADII)
            R("[RADSEP]", "   Separation between radii (arcsec)", gf.RADSEP)
        else:
            R("[RADII]", "   Radii for rings", gf.RADII)
        R("[XPOS]", "   X center of the galaxy (pixel)", gf.XPOS)
        R("[YPOS]", "   Y center of the galaxy (pixel)", gf.YPOS)
        R("[INC]", "   Inclination angle (degrees)", gf.INC)
        R("[PA]", "   Position angle (degrees)", gf.PHI)
        R("[MASK]", "   Type of mask for intensity map", p.MaskType)
        R("[SIDE]", "   Side of the galaxy to be used", gf.SIDE)

    if ma.globprof or (defaults and isAll):
        R("[globalProfile]", "Saving the global profile?", stringize(ma.globprof))
    if ma.totalmap or (defaults and isAll):
        R("[totalMap]", "Saving integrated intensity map to FITS file?", stringize(ma.totalmap))
        R("[SNMap]", "   Computing also S/N map for intensity map?", stringize(ma.SNmap))
        if ma.SNmap or (defaults and isAll):
            R("[taper]", "     Type of velocity taper used", ma.taper)
            R("[contChans]", "     Channels used for continuum subtraction", " ".join(str(c) for c in ma.contChans))
    if ma.massdensmap or (defaults and isAll):
        R("[massdensMap]", "Saving HI mass density map to FITS file?", stringize(ma.massdensmap))
    if ma.velocitymap or (defaults and isAll):
        R("[velocityMap]", "Saving velocity map to FITS file?", stringize(ma.velocitymap))
    if ma.dispersionmap or (defaults and isAll):
        R("[dispersionMap]", "Saving velocity dispersion map to FITS file?", stringize(ma.dispersionmap))
    if ma.rmsmap or (defaults and isAll):
        R("[rmsMap]", "Saving RMS map to FITS file?", stringize(ma.rmsmap))
    if p.getMaps() or (defaults and isAll):
        R("[MASK]", "   Mask used for maps and profile?", p.MaskType)
    if ma.totalmap or ma.velocitymap or ma.dispersionmap or (defaults and isAll):
        R("[MAPTYPE]", "   How to extract the map (gaussian or moment)?", ma.maptype)

    if isAll or p.flagPV or (defaults and whichtask in ("PVSLICE", "PV")):
        R("[PVSLICE]", "Extracting a position-velocity slice from a cube?", stringize(p.flagPV))
        R("[ANTIALIAS]", "   Type of anti-aliasing to use for slice", p.ANTIALIAS)
        R("[XPOS_PV]", "   X center of the slice", p.XPOS_PV)
        R("[YPOS_PV]", "   Y center of the slice", p.YPOS_PV)
        R("[PA_PV]", "   Position angle of the slice", p.PA_PV)
        R("[P1_PV]", "   Position 1 of a two-point slice", " ".join(f"{v:.1f}" for v in p.P1_PV))
        R("[P2_PV]", "   Position 2 of a two-point slice", " ".join(f"{v:.1f}" for v in p.P2_PV))
        R("[WIDTH_PV]", "   Width of the slice (arcsec)", p.WIDTH_PV)
    if isAll or p.flagRend3D or (defaults and whichtask == "REND3D"):
        R("[REND3D]", "Writing a 3D rendering of a datacube?", stringize(p.flagRend3D))
        R("[RENDANGLE]", "   Maximum azimuthal angle for rendering", p.rendangle)

    Str.write("\n-------------------------------------------------------------------\n\n")
