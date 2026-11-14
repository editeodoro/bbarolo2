"""
The table of BBarolo's input parameters.

Every keyword is defined once here: how it is read, which attribute(s) of
:class:`bbarolo.params.Param` it sets, the tasks that use it, a short description and
its aliases. Parameter files, ``KEY=value`` on the command line and keyword arguments in
Python all go through this table. Keywords are case-insensitive. Default values are
those of a new :class:`~bbarolo.params.Param` (``bbarolo.help('free')`` shows them).
"""

import re
import math
import difflib


# --------------------------------------------------------------------------
# Small parsing helpers (port of Utilities/string.cpp)
# --------------------------------------------------------------------------
def boolify(s):
    """'1' or 'true' (any case) -> True, anything else -> False."""
    s = str(s).strip()
    return s == "1" or s.lower() == "true"


def stringize(b):
    return "true" if b else "false"


def _tokens(s):
    return str(s).split()


def _first(s):
    t = _tokens(s)
    return t[0] if t else ""


_int_re = re.compile(r"^[+-]?\d+")
_flt_re = re.compile(r"^[+-]?((\d+\.?\d*)|(\.\d+))([eE][+-]?\d+)?")


def readint(s, default=0):
    """Emulates C++ ``ss >> int`` (reads the leading integer of the first token)."""
    m = _int_re.match(_first(s))
    return int(m.group(0)) if m else default


def readfloat(s, default=0.0):
    """Emulates C++ ``ss >> float``."""
    tok = _first(s)
    m = _flt_re.match(tok)
    if m:
        return float(m.group(0))
    try:
        return float(tok)   # nan, inf
    except ValueError:
        return default


def _int_or_zero(x):
    """int(x), or 0 (as an unreadable integer) if x is infinite or NaN."""
    return int(x) if x == x and abs(x) != float("inf") else 0


def readflag(s):
    return boolify(_first(s))


def readflagorint(s):
    tok = _first(s)
    if "true" in tok.lower() or "false" in tok.lower():
        return int(boolify(tok))
    return readint(tok)


def readbeam(s):
    """A beam, 'BMAJ BMIN BPA' (BEAM, NEWBEAM): BMAJ and BMIN in arcsec, or in degrees with a final d
    (0.03d), BPA in degrees. One value is a circular beam, two values have no BPA. Returns
    [bmaj, bmin, bpa] in arcsec and degrees (bpa None if not given), or None for an empty string."""
    from .bbutils import BBaroloError
    toks = str(s).replace(",", " ").split()
    if not toks:
        return None
    if len(toks) > 3:
        raise BBaroloError(f"a beam is 'BMAJ BMIN BPA' (or one value, a circular beam), not {s!r}")
    vals = []
    for i, t in enumerate(toks):
        deg = i < 2 and t[-1:] in ("d", "D")
        try:
            x = float(t[:-1] if deg else t)
        except ValueError:
            raise BBaroloError(f"a beam is 'BMAJ BMIN BPA' (arcsec, or degrees with a final d, e.g. 0.03d; "
                               f"BPA in degrees), not {s!r}") from None
        if not math.isfinite(x) or (i < 2 and x <= 0):
            raise BBaroloError(f"the axes of a beam must be positive numbers, not {s!r}")
        vals.append(x * 3600. if deg else x)
    bmaj = vals[0]
    bmin = vals[1] if len(vals) > 1 else bmaj
    return [bmaj, bmin, vals[2] if len(vals) > 2 else None]


def readstr(s):
    """readFilename: the rest of the line, stripped."""
    return str(s).strip()


def readvec(s, typ=float):
    """The numbers of a list, separated by spaces or commas (C++ read only the first of a comma-separated
    list), up to the first that is not a number."""
    out = []
    for t in str(s).replace(",", " ").split():
        try:
            out.append(typ(float(t)) if typ is int else typ(t))
        except (ValueError, OverflowError):        # (inf, nan: not integers, as for C++ >> int)
            break
    return out


def isNumber(w):
    return len(w) > 0 and all(c.isdigit() for c in w)


def splitString(s, delimiter):
    """Return (key,val) if delimiter is in s, otherwise None."""
    found = s.find(delimiter)
    if found < 0:
        return None
    return s[:found], s[found + 1:]


# --------------------------------------------------------------------------
# Readers (parameter-file semantics: as the C++ ``ss >> value``)
# --------------------------------------------------------------------------
READERS = {
    "str": readstr,
    "lstr": lambda s: readstr(s).lower(),
    # lower case (WCS values such as 07h36m51s), except a file(...) column: its path keeps its case
    # (C++ lowercased it too, so paths with capital letters were not found)
    "pos": lambda s: readstr(s) if "file(" in readstr(s).lower() else readstr(s).lower(),
    "ustr": lambda s: readstr(s).upper(),
    "flag": readflag,
    "int": readint,
    "float": readfloat,
    "flagorint": readflagorint,
    "ivec": lambda s: readvec(s, int),
    "fvec": lambda s: readvec(s, float),
    "cdens": lambda s: _int_or_zero(readfloat(s)),
    "fvec3": lambda s: readvec(s, float)[:3],    # (the values not given keep the current ones: Keyword.apply)
    "ivec6": lambda s: (readvec(s, int) + [-1] * 6)[:6],
    "fvec2": lambda s: (readvec(s, float) + [-1, -1])[:2],
    "beam": readbeam,
}

# Kinds whose value is one number / flag (a sequence is not accepted in Python)
SCALAR_KINDS = ("flag", "int", "float", "flagorint", "cdens")
NUMERIC_KINDS = ("int", "float", "cdens")
VECTOR_KINDS = ("ivec", "fvec", "fvec3", "ivec6", "fvec2", "beam")

G3 = ("gm", "gf", "gw")     # GALMOD, 3DFIT and GALWIND structures
G2 = ("gm", "gf")           # GALMOD and 3DFIT structures

# Groups of tasks (for the documentation of each keyword). A fit task with a METHOD (3DFIT, WINDFIT, SLITFIT) is
# given alone (all its methods) or as TASK/METHOD (only with that METHOD), e.g. "3DFIT/CLASSIC".
ALL = ("ALL",)
FITS = ("3DFIT", "WINDFIT", "SLITFIT")       # the tasks with a METHOD
CLASSIC = ("3DFIT/CLASSIC",)                 # 3DFIT with METHOD CLASSIC
LIKE3 = ("3DFIT/GLOBAL", "3DFIT/BAYESIAN")   # 3DFIT with the likelihood (METHOD GLOBAL or BAYESIAN)
LIKE = LIKE3 + ("WINDFIT", "SLITFIT")        # the options of the likelihood fits
SAMPLE = ("3DFIT/BAYESIAN", "WINDFIT/BAYESIAN", "SLITFIT/BAYESIAN")    # METHOD BAYESIAN (the sampler)
MINIM = ("3DFIT/GLOBAL", "WINDFIT/GLOBAL", "SLITFIT/GLOBAL")           # METHOD GLOBAL (the minimiser)
MOD = ("3DFIT", "GALMOD", "SPACEPAR")
RING = ("3DFIT", "GALMOD", "SPACEPAR", "2DFIT", "ELLPROF")
SLIT = ("SLITFIT",)
WIND = ("GALWIND", "WINDFIT")
MOD3 = ("3DFIT/CLASSIC", "GALMOD", "SPACEPAR")        # not 3DFIT with METHOD GLOBAL or BAYESIAN
MAPS = ("TOTALMAP", "VELOCITYMAP", "DISPERSIONMAP", "MASSDENSMAP", "RMSMAP", "GLOBALPROFILE")
MAPSM = ("TOTALMAP", "VELOCITYMAP", "DISPERSIONMAP", "MASSDENSMAP", "RMSMAP")   # the maps that use the mask


class Keyword:
    """One input parameter of BBarolo.

    name:     the keyword (as written in parameter files)
    kind:     how the value is read (see READERS)
    targets:  attribute(s) of Param that are set, e.g. ('gf.FREE',)
    tasks:    tasks that use the keyword ('ALL' = general parameter)
    help:     short description
    aliases:  other accepted names
    also:     extra attributes set to a constant when the keyword is given
              (e.g. THRESHOLD also sets se.UserThreshold = True)
    flag:     True if the keyword switches a task on

    From bbarolo/keywords_doc.py (the documentation of the parameters):
    doc:      the full description (Markdown), shown by help('KEYWORD') and on the website
    group:    the section of the parameter pages of its tasks (a name, or {task: name, "*": name})
    default_text: the default as words, when the default value is a placeholder (e.g. -1)
    """
    __slots__ = ("name", "kind", "targets", "tasks", "help", "aliases", "also", "flag", "doc", "group",
                 "default_text")

    def __init__(self, name, kind, targets, tasks, help, aliases=(), also=None, flag=False):
        self.name = name
        self.kind = kind
        self.targets = tuple(targets)
        self.tasks = tuple(tasks)
        self.help = help
        self.aliases = tuple(aliases)
        self.also = dict(also or {})
        self.flag = flag
        self.doc = self.group = self.default_text = None

    def group_for(self, task=None):
        """The section of the parameter page of task where the keyword is listed."""
        g = self.group
        if isinstance(g, dict):
            return g.get(task, g.get("*"))
        return g

    def read(self, rest):
        if self.kind == "beam":
            try:
                return readbeam(rest)
            except ValueError as e:
                raise type(e)(f"{self.name}: {e}") from None
        return READERS[self.kind](rest)

    def apply(self, par, rest):
        """Set the value read from the string rest (parameter-file semantics). The string is
        also recorded in par._given (the parameters as given, in the order of their last setting)."""
        val = self.read(rest)
        if self.kind == "fvec3":
            # the values not given keep the current ones, as C++ readArray (P1PAR 0 100 keeps the step)
            val = list(val) + list(self.value(par))[len(val):3]
        for t in self.targets:
            _setattr(par, t, val)
        for t, v in self.also.items():
            _setattr(par, t, v)
        given = getattr(par, "_given", None)
        if given is not None:
            given.pop(self.name, None)
            given[self.name] = str(rest)

    def value(self, par):
        """Current value in par (of the first target)."""
        return _getattr(par, self.targets[0])

    def __repr__(self):
        return f"Keyword({self.name})"


def _getattr(obj, path):
    for a in path.split("."):
        obj = getattr(obj, a)
    return obj


def _setattr(obj, path, val):
    parts = path.split(".")
    for a in parts[:-1]:
        obj = getattr(obj, a)
    if isinstance(val, list):
        val = list(val)
    setattr(obj, parts[-1], val)


def _g(structs, attr):
    return tuple(f"{s}.{attr}" for s in structs)


K = Keyword
KEYWORDS = [
    # ---------------------------------------------------------------- general
    K("FITSFILE", "str", ["imageFile"], ALL, "FITS file to be analysed"),
    K("FITSLIST", "str", ["imageList"], ALL, "Text file with a list of FITS files to be analysed"),
    K("VERBOSE", "flagorint", ["verbose"], ALL, "Output messages: 0 (false) none, 1 (true, the default) the main "
      "ones and one line per ring, 2 all (the parameters, the details of every ring, as in C++)"),
    K("OUTFOLDER", "str", ["outFolder"], ALL, "Directory where outputs are written (default ./output/<OBJECT>/)"),
    K("OUTPREFIX", "str", ["outPrefix"], ALL, "Prefix for output files (default: OBJECT keyword of the header)"),
    K("LOGFILE", "flag", ["logFile"], ALL, "Redirect output messages to a file?"),
    K("THREADS", "int", ["threads"], ALL, "Number of threads (default: the number of cores, at most 16)"),
    K("DEBUG", "flag", ["debug"], ALL, "Debugging mode?"),
    K("SHOWBAR", "flag", ["showbar"], ALL, "Showing progress bars?"),
    K("PLOTS", "flagorint", ["plots"], ALL, "Producing output plots?"),
    K("BEAM", "beam", ["beam"], ALL, "Beam of the data, BMAJ BMIN BPA (arcsec, or degrees with a final d; "
      "BPA in degrees; one value: a circular beam): replaces the beam of the header"),
    K("CHECKCUBE", "int", ["checkCube"], ALL, "Check cube for bad channels/rows/columns (0-5)?"),
    K("AUTO", "flag", ["AUTO"], ALL, "Automated 3DFIT (same as BBarolo -f)"),
    K("FLUXCONVERT", "flag", ["fluxConvert"], ALL, "Whether to convert flux to Jy?"),
    K("FLAGROBUSTSTATS", "flag", ["flagRobustStats"], ALL, "Using robust statistics?", aliases=["robuststats"]),
    K("CONTSUB", "flag", ["contsub"], ("CONTSUB",) + SLIT, "Subtract continuum from datacube (SLITFIT: "
      "from each position of the slit)?", flag=True),
    K("EXCLWIND", "str", ["exclude_windows"], ("CONTSUB",) + SLIT, "Channel window(s) to exclude for continuum fit"),
    K("CONTORDER", "int", ["cont_order"], ("CONTSUB",) + SLIT, "Order of polynomial for continuum fit"),
    K("MAKEMASK", "flag", ["makeMask"], ("MAKEMASK",), "Writing the mask to a fitsfile", flag=True),
    K("MASK", "str", ["MaskType"], ("MAKEMASK", "3DFIT", "WINDFIT", "GALMOD", "SPACEPAR",
                                    "2DFIT", "ELLPROF", "REND3D") + MAPSM,
      "Type of mask: SMOOTH, SEARCH, SMOOTH&SEARCH, THRESHOLD, NEGATIVE, NONE or FILE(mask.fits)"),
    K("BLANKCUT", "float", ["blankCut"], ("MAKEMASK", "3DFIT", "GALMOD", "SPACEPAR", "2DFIT", "ELLPROF",
                                          "REND3D") + MAPSM,
      "S/N cut for the SMOOTH mask"),
    K("STATS", "flag", ["flagStats"], ("STATS",), "Calculate and print statistics?", flag=True),
    K("GALMODTYPE", "ustr", ["galmodtype"], ("3DFIT", "GALMOD", "SPACEPAR", "GALWIND"),
      "Type of model: MC (Monte Carlo) or ANALYTIC (noiseless). Default MC, but "
      "ANALYTIC for 3DFIT with METHOD GLOBAL or BAYESIAN"),
    K("RINGMODE", "ustr", ["ringmode"], ("3DFIT", "GALMOD", "SLITFIT"),
      "Rings of the models: CENTRED (each ring is an annulus centred on its radius, reaching "
      "halfway to the next rings, as fitted by 3DFIT), NODES (values at the radii, disk from the "
      "first to the last radius) or SHIFTED (BBarolo 1.x: first and last radius moved by RADSEP/2)"),

    # ---------------------------------------------------------------- SEARCH
    K("SEARCH", "flag", ["se.flagSearch"], ("SEARCH",), "Searching for sources in cube?", flag=True),
    K("SEARCHTYPE", "lstr", ["se.searchType"], ("SEARCH",), "Type of searching performed (spatial or spectral)"),
    K("ITERNOISE", "flag", ["se.iternoise"], ("SEARCH", "STATS"), "Estimating noise with iterative algorithm?"),
    K("FLAGADJACENT", "flag", ["se.flagAdjacent"], ("SEARCH",), "Using Adjacent-pixel criterion?",
      aliases=["adjacent"]),
    K("THRESHSPATIAL", "int", ["se.threshSpatial"], ("SEARCH",), "Max. spatial separation for merging",
      aliases=["thrspatial"]),
    K("THRESHVELOCITY", "int", ["se.threshVelocity"], ("SEARCH",), "Max. velocity separation for merging",
      aliases=["thrvelocity"]),
    K("MINCHANNELS", "int", ["se.minChannels"], ("SEARCH",), "Minimum # Channels in a detection"),
    K("MINVOXELS", "int", ["se.minVoxels"], ("SEARCH",), "Minimum # Voxels in a detection"),
    K("MINPIX", "int", ["se.minPix"], ("SEARCH",), "Minimum # Pixels in a detection", aliases=["minpixels"]),
    K("MAXCHANNELS", "int", ["se.maxChannels"], ("SEARCH",), "Maximum # Channels in a detection"),
    K("MAXANGSIZE", "float", ["se.maxAngSize"], ("SEARCH",), "Max angular size of a detection in arcmin"),
    K("REJECTBEFOREMERGE", "flag", ["se.RejectBeforeMerge"], ("SEARCH",), "Reject objects before merging?"),
    K("TWOSTAGEMERGING", "flag", ["se.TwoStageMerging"], ("SEARCH",), "Merge objects in two stages?",
      aliases=["twostagemerge"]),
    K("SNRCUT", "float", ["se.snrCut"], ("SEARCH",), "SNR Threshold (in sigma)"),
    K("THRESHOLD", "float", ["se.threshold"], ("SEARCH",), "Detection Threshold (flux units)",
      also={"se.UserThreshold": True}),
    K("FLAGGROWTH", "flag", ["se.flagGrowth"], ("SEARCH",), "Growing objects after detection?",
      aliases=["growth"]),
    K("GROWTHCUT", "float", ["se.growthCut"], ("SEARCH",), "SNR Threshold for growth"),
    K("GROWTHTHRESHOLD", "float", ["se.growthThreshold"], ("SEARCH",), "Threshold for growth (flux units)",
      aliases=["growththresh"], also={"se.flagUserGrowthT": True}),
    K("CUBELETS", "flag", ["se.cubelets"], ("SEARCH",), "Writing sub-cubes of detections?"),
    K("EDGES", "int", ["se.edges"], ("SEARCH",), "Number of pixels at the edges of cubelets?"),
    K("SORTSOURCES", "lstr", ["se.sortsrcs"], ("SEARCH",), "Sorting detections based on?"),
    K("PBCORR", "int", ["se.pbcorr"], ("SEARCH",), "Correct flux for primary beam attenuation?"),

    # ---------------------------------------------------------------- MAPS
    K("GLOBALPROFILE", "flag", ["ma.globprof"], ("GLOBALPROFILE",), "Saving the global profile?", flag=True),
    K("TOTALMAP", "flag", ["ma.totalmap"], ("TOTALMAP",), "Saving integrated intensity map to FITS file?", flag=True),
    K("MASSDENSMAP", "flag", ["ma.massdensmap"], ("MASSDENSMAP",), "Saving HI mass density map to FITS file?",
      flag=True),
    K("VELOCITYMAP", "flag", ["ma.velocitymap"], ("VELOCITYMAP",), "Saving velocity map to FITS file?", flag=True),
    K("DISPERSIONMAP", "flag", ["ma.dispersionmap"], ("DISPERSIONMAP",),
      "Saving velocity dispersion map to FITS file?", flag=True),
    K("RMSMAP", "flag", ["ma.rmsmap"], ("RMSMAP",), "Saving RMS map to FITS file?", flag=True),
    K("MAPTYPE", "ustr", ["ma.maptype"], ("TOTALMAP", "VELOCITYMAP", "DISPERSIONMAP", "3DFIT", "WINDFIT",
                                          "GALMOD"),
      "How to extract the maps (MOMENT or GAUSSIAN)?"),
    K("SNMAP", "flag", ["ma.SNmap"], ("TOTALMAP", "SEARCH"), "Computing also S/N map for intensity map?"),
    K("TAPER", "lstr", ["ma.taper"], ("TOTALMAP",), "Type of velocity taper used for the S/N map"),
    K("CONTCHANS", "ivec", ["ma.contChans"], ("TOTALMAP",), "Channels used for continuum subtraction (S/N map)"),
    K("VELDEF", "lstr", ["ma.veldef"], ALL, "Definition for velocity conversion (radio, optical, relativistic, auto)"),

    # ---------------------------------------------------------------- task flags
    K("2DFIT", "flag", ["flagRing"], ("2DFIT",), "Fitting velocity field with a ring model?", aliases=["fit2d"],
      flag=True),
    K("ELLPROF", "flag", ["flagEllProf"], ("ELLPROF",), "Deriving radial intensity profile?", flag=True),
    K("3DFIT", "flag", ["gf.flagGALFIT"], ("3DFIT",), "Fitting a 3D model to the datacube?",
      aliases=["galfit", "fit3d"], flag=True),
    K("GALMOD", "flag", ["gm.flagGALMOD"], ("GALMOD",), "Writing a 3D model?", flag=True),
    K("GALWIND", "flag", ["gw.flagGALWIND"], ("GALWIND",), "Generating a 3D datacube with a wind model?",
      flag=True),
    K("WINDFIT", "flag", ["gw.flagWINDFIT"], ("WINDFIT",), "Fitting a bi-conical wind to the datacube?",
      flag=True),
    K("SLITFIT", "flag", ["flagSlitFit"], SLIT, "Fitting the kinematics of a long-slit spectrum (position x "
      "wavelength)?", flag=True),
    K("SPACEPAR", "flag", ["flagSpace"], ("SPACEPAR",), "Full parameter space for a pair of parameters", flag=True),

    # ---------------------------------------------------------------- rings (GALMOD/3DFIT/GALWIND)
    K("NRADII", "int", _g(G2, "NRADII"), RING + SLIT, "Number of radii"),
    K("RADII", "str", _g(G3, "RADII"), RING + WIND + SLIT,
      "Radii for rings (arcsec): list of values, Rmin~Rmax:step or file()"),
    K("RADSEP", "float", _g(G2, "RADSEP"), RING + SLIT, "Separation between radii (arcsec)"),
    K("VROT", "str", _g(G3, "VROT"), RING + WIND + SLIT, "Rotation velocity (km/s)"),
    K("VRAD", "str", _g(G2, "VRAD"), RING + SLIT, "Radial velocity (km/s)"),
    K("Z0", "str", _g(G2, "Z0"), MOD + SLIT, "Scale height of the disk (arcsec); the fits use one pixel if not given "
      "(a thin disk), GALMOD needs it"),
    K("VVERT", "str", _g(G2, "VVERT"), MOD, "Vertical velocity (km/s)"),
    K("DVDZ", "str", _g(G2, "DVDZ"), MOD, "Vertical gradient of rotation velocity (km/s/arcs)"),
    K("ZCYL", "str", _g(G2, "ZCYL"), MOD, "Height where the gradient begins (arcs)"),
    K("VDISP", "str", _g(G3, "VDISP"), MOD + WIND + SLIT, "Velocity dispersion (km/s)"),
    K("XPOS", "pos", _g(G3, "XPOS"), RING + WIND + SLIT, "X center of the galaxy (pixel, or WCS with a unit)"),
    K("YPOS", "pos", _g(G3, "YPOS"), RING + WIND + SLIT, "Y center of the galaxy (pixel, or WCS with a unit)"),
    K("VSYS", "str", _g(G3, "VSYS"), RING + WIND + SLIT, "Systemic velocity of the galaxy (km/s)"),
    K("INC", "str", _g(G3, "INC"), RING + WIND + SLIT, "Inclination angle (degrees)"),
    K("PA", "str", _g(G3, "PHI"), RING + WIND,
      "Position angle of the receding half of the major axis (degrees, anticlockwise from North)", aliases=["phi"]),
    K("DENS", "str", _g(G3, "DENS"), MOD + ("GALWIND",) + SLIT, "Gas surface density (a flux per arcsec2 or a column "
      "density, see DENSFLUX)"),
    K("CDENS", "cdens", _g(G3, "CDENS"), MOD + ("GALWIND",), "Surface density of the clouds of MC models (per pixel)"),
    K("CMODE", "int", _g(G3, "CMODE"), MOD, "Mode of clouds-surface density"),
    K("ISEED", "int", _g(G3, "ISEED"), MOD + ("GALWIND",), "Seed for random number generator"),
    K("NV", "int", _g(G3, "NV"), MOD + ("GALWIND",), "Number of subclouds per profile"),
    K("SM", "flag", _g(G3, "SM"), MOD + ("GALWIND",), "Whether to smooth the model to the beam"),
    K("LTYPE", "int", _g(G2, "LTYPE"), MOD + SLIT, "Layer type along z (1=gaussian, 2=sech2, 3=exp, 4=Lorentz, 5=box)"),
    K("REDSHIFT", "float", _g(G2, "REDSHIFT"), MOD + WIND + SLIT, "Redshift of the galaxy"),
    K("RESTWAVE", "fvec", _g(G2, "RESTWAVE"), MOD + WIND + SLIT, "Rest wavelength(s) of the line(s)"),
    K("RESTFREQ", "fvec", _g(G2, "RESTFREQ"), MOD + WIND + SLIT, "Rest frequency(ies) of the line(s)"),
    K("RELINT", "fvec", _g(G2, "RELINT"), MOD + WIND + SLIT, "Relative intensities of the lines"),
    K("NOISERMS", "float", _g(G2, "NOISERMS"), ("3DFIT/CLASSIC", "GALMOD"), "RMS noise to add to the model"),
    K("RINGFILE", "str", _g(G2, "ringfile"), MOD, "A BBarolo output file with the rings"),
    K("EMPTY", "flag", _g(G3, "EMPTY"), MOD, "Whether to leave empty inner part"),
    K("DENSFLUX", "flag", _g(G3, "DENSFLUX"), ("GALMOD",), "Whether DENS is integrated flux / arcs2"),

    # ---------------------------------------------------------------- 3DFIT
    K("DELTAINC", "float", ["gf.DELTAINC"], CLASSIC, "Max inclination variation from INC (degrees)"),
    K("DELTAPA", "float", ["gf.DELTAPHI"], CLASSIC, "Max position angle variation from PA (degrees)",
      aliases=["deltaphi"]),
    K("DELTAVROT", "float", ["gf.DELTAVROT"], CLASSIC, "Max rot. velocity variation from VROT (km/s)"),
    K("MINVDISP", "float", ["gf.MINVDISP"], CLASSIC, "Minimum acceptable velocity dispersion (km/s)"),
    K("MAXVDISP", "float", ["gf.MAXVDISP"], CLASSIC, "Maximum acceptable velocity dispersion (km/s)"),
    K("FTYPE", "int", ["gf.FTYPE"], ("3DFIT", "SPACEPAR"),
      "Residuals to minimize (1=chi2, 2=|m-o|, 3=|m-o|/|m+o|, 4=(m-o)^2)"),
    K("WFUNC", "int", ["gf.WFUNC"], ("3DFIT", "SPACEPAR", "2DFIT"),
      "Weighting function (0=uniform, 1=|cos|, 2=cos^2, -1=|sin|, -2=sin^2)"),
    K("TOL", "float", ["gf.TOL"], CLASSIC, "Minimization tolerance"),
    K("FREE", "str", ["gf.FREE"], FITS + ("2DFIT",),
      "Free parameters, case-insensitive. 3DFIT (METHOD CLASSIC) and 2DFIT: any of VROT, VDISP, DENS, Z0, INC, PA, "
      "XPOS, YPOS, VSYS, VRAD, or ALL. 3DFIT with METHOD GLOBAL or BAYESIAN: a name alone has one value per ring, "
      "NAME_SINGLE one "
      "value for all the rings, NAME_FUNC a function of radius (VROT, VDISP, DENS), plus SIGMA, FRACERR and "
      "RADMAX (default: VROT VDISP XPOS_SINGLE YPOS_SINGLE VSYS_SINGLE INC_SINGLE PA_SINGLE Z0_SINGLE RADMAX "
      "FRACERR)"),
    K("GUESSTYPE", "ustr", ["gf.GUESSTYPE"], ("3DFIT", "2DFIT", "SPACEPAR"),
      "Initial estimates of the parameters that are not given: FIT (a beam-convolved disk fitted to the "
      "total map and a 2D tilted-ring fit of the velocity field), ELLIPSE (the C++ algorithm, corrected) or "
      "LEGACY (the C++ algorithm)"),
    K("SIDE", "ustr", ["gf.SIDE"], ("3DFIT", "SPACEPAR", "2DFIT", "ELLPROF"),
      "Side of the galaxy to be used (A=approaching, R=receding, B=both)"),
    K("FREEANGLE", "float", ["gf.FREEANGLE"], ("2DFIT",),
      "Angle around the minor axis left out of the fit of the velocity field (degrees)"),
    K("BWEIGHT", "float", ["gf.BWEIGHT"], ("3DFIT", "SPACEPAR"), "Weight for blank pixels"),
    K("TWOSTAGE", "flag", ["gf.TWOSTAGE"], CLASSIC, "Two stages minimization?"),
    K("FLAGERRORS", "flag", ["gf.flagERRORS"], CLASSIC, "Estimating errors?", aliases=["errors"]),
    K("NORM", "ustr", ["gf.NORM"], MOD3 + SLIT, "Type of normalization (LOCAL, AZIM or NONE)"),
    K("REGTYPE", "lstr", ["gf.REGTYPE"], CLASSIC, "Regularization of the second stage (auto, bezier, median, "
      "degree of polynomial)", aliases=["polyn"]),
    K("STARTRAD", "int", ["gf.STARTRAD"], ("3DFIT/CLASSIC", "SPACEPAR"), "Starting radius (ring index)"),
    K("ADRIFT", "flag", ["gf.flagADRIFT"], CLASSIC, "Computing asymmetric drift correction?"),
    K("ADRIFTPOL1", "int", ["gf.ADRIFTPOL1"], CLASSIC, "Polynomial for VDISP2 regularisation"),
    K("ADRIFTPOL2", "int", ["gf.ADRIFTPOL2"], CLASSIC, "Polynomial for SIGMA regularisation"),
    K("REVERSE", "lstr", ["gf.REVERSE"], CLASSIC, "Using reverse-cumulative fitting (true, false, auto)?"),
    K("NORMALCUBE", "flag", ["gf.NORMALCUBE"], ("3DFIT/CLASSIC", "SPACEPAR"), "Normalizing cube to help convergence?"),
    K("BADOUT", "flag", ["gf.flagBADOUT"], CLASSIC, "Write unconverged rings in output (with flag)?"),
    K("PLOTMASK", "flag", ["gf.PLOTMASK"], ("3DFIT",), "Overlaying mask to output plots?"),
    K("PLOTMINCON", "float", ["gf.PLOTMINCON"], ("3DFIT",), "Minimum flux contour for output plots"),

    # ---------------------------------------------------------------- METHOD (3DFIT, WINDFIT, SLITFIT)
    K("METHOD", "ustr", ["gf.METHOD"], FITS,
      "Fitting method: CLASSIC (3DFIT only: ring by ring, minimising the residuals), GLOBAL (all the rings "
      "together: the maximum of the likelihood, with the errors of its curvature) or BAYESIAN (the posterior, "
      "by nested sampling). Default CLASSIC for 3DFIT, GLOBAL for WINDFIT and SLITFIT"),
    K("SAMPLER", "ustr", ["gf.SAMPLER"], SAMPLE,
      "Nested sampler of METHOD BAYESIAN: NAUTILUSBB (nautilus, built in), NAUTILUS (package nautilus-sampler) or "
      "DYNESTY"),
    K("MINIMISER", "str", ["gf.MINIMISER"], MINIM,
      "Minimiser of METHOD GLOBAL: L-BFGS-B, TNC, SLSQP, trust-constr, Powell, Nelder-Mead, COBYLA, COBYQA or "
      "differential_evolution", aliases=["minimizer"]),
    K("PRIORS", "str", ["gf.PRIORS"], LIKE,
      "Priors of the free parameters, NAME=(distribution,arguments) with the distributions of scipy.stats, "
      "e.g. VROT=(uniform,loc=0,scale=300) INC=(norm,60,5) (default: see bbarolo.Bayesian3DFIT)"),
    K("SAMPLEROPTS", "str", ["gf.SAMPLEROPTS"], LIKE,
      "Options of the sampler (BAYESIAN) or of the minimisation (GLOBAL), key=value, e.g. nlive=200 dynamic=true "
      "seed=1 (dynesty; the static sampler unless dynamic=true) or n_live=1000 seed=1 (nautilus, nautilusbb) or "
      "nstart=3 seed=1 errors=true (GLOBAL)",
      aliases=["sampler_kwargs"]),
    K("RUNOPTS", "str", ["gf.RUNOPTS"], LIKE,
      "Options of the sampling, key=value, e.g. dlogz=0.5 maxiter=20000 (dynesty) or f_live=0.01 "
      "(nautilus, nautilusbb) or "
      "maxiter=500 (GLOBAL: the options of the minimiser of scipy)",
      aliases=["run_kwargs"]),
    K("LIKEOPTS", "str", ["gf.LIKEOPTS"], LIKE,
      "Options of the likelihood, key=value (defaults first): likemask=dilated|none, dilation=1,3 (radius in "
      "beams, channels), noisecorr=measured|beam|<voxels per independent noise value>, densprof_smooth=0.1, "
      "bbres=false|true (3DFIT residuals), radmax=edge|taper|ring (the free RADMAX: the outer edge of the disk, "
      "sharp or Gaussian of width radmax_width, or the radius of the last ring), sigma=<fixed noise, default: "
      "measured>, fracerr=<fixed model error, default: 0>",
      aliases=["like_kwargs"]),

    # ---------------------------------------------------------------- GALWIND
    K("CONELEN", "float", ["gw.CONELEN"], WIND, "Length of each cone of the wind, from the centre (arcsec)"),
    K("OPENANG", "str", ["gw.OPENANG"], WIND, "Wind opening angle (degrees)"),
    K("OPENIN", "str", ["gw.OPENIN"], WIND, "Wind inner opening angle (degrees): hollow cones (WTYPE 2 and "
      "WINDFIT; default 0)"),
    K("NLAYERS", "int", ["gw.NLAYERS"], WIND, "Number of layers (cylinders or shells) in each cone (WINDFIT: "
      "number of shells)"),
    K("VWIND", "str", ["gw.VWIND"], WIND, "Radial velocity of the wind (km/s)"),
    K("DENSTYPE", "int", ["gw.DENSTYPE"], ("GALWIND",), "How to distribute density in layers",
      aliases=["dtype"]),
    K("WTYPE", "int", ["gw.WTYPE"], ("GALWIND",), "Cylindrical (0), spherical (1) or analytic cones (2) model"),
    K("DISKRINGS", "str", ["gw.DISKRINGS"], ("WINDFIT",), "A rings file of a fixed disk added to the wind "
      "model (its ring densities are fitted with the emission of the wind)"),

    # ---------------------------------------------------------------- SPACEPAR
    K("P1", "str", ["P1"], ("SPACEPAR",), "First parameter to explore"),
    K("P1PAR", "fvec3", ["P1p"], ("SPACEPAR",), "Range and step for P1 (min max step)"),
    K("P2", "str", ["P2"], ("SPACEPAR",), "Second parameter to explore"),
    K("P2PAR", "fvec3", ["P2p"], ("SPACEPAR",), "Range and step for P2 (min max step)"),

    # ---------------------------------------------------------------- SMOOTH / SMOOTHSPEC
    K("SMOOTH", "flag", ["flagSmooth"], ("SMOOTH",), "Smoothing the datacube?", flag=True),
    K("BOX", "ivec6", ["BOX"], ("SMOOTH",), "Sub-region to be used (xmin xmax ymin ymax zmin zmax)"),
    K("NEWBEAM", "beam", ["newbeam"], ("SMOOTH", "MAKEMASK", "3DFIT", "SPACEPAR", "2DFIT",
                                       "ELLPROF", "REND3D") + MAPSM,
      "New beam of SMOOTH and of the SMOOTH masks, BMAJ BMIN BPA (as BEAM; BPA not given: that of the data)"),
    K("LINEAR", "float", ["linear"], ("3DFIT", "GALMOD", "SPACEPAR") + WIND + SLIT,
      "Spectral resolution of the models (sigma, channels) when SPECRES is not given (as in C++)"),
    K("SPECRES", "float", ["specres"], ("3DFIT", "GALMOD", "SPACEPAR") + WIND + SLIT,
      "Spectral resolution of the models (sigma, in channels; default: LINEAR if given, else "
      "1/2.355, i.e. a FWHM of one channel)"),
    K("FACTOR", "float", ["factor"], ("SMOOTH", "MAKEMASK", "3DFIT", "SPACEPAR", "2DFIT", "ELLPROF",
                                      "REND3D") + MAPSM,
      "New beam factor (times old beam), also used by the SMOOTH mask"),
    K("SCALEFACTOR", "float", ["scalefactor"], ("SMOOTH",), "Scaling factor for the smoothed cube"),
    K("FFT", "flag", ["flagFFT"], ("SMOOTH",), "Using FFT for convolution?"),
    K("REDUCE", "lstr", ["reduce"], ("SMOOTH", "SMOOTHSPEC"), "Averaging pixels to reduce datacube?"),
    K("SMOOTHOUTPUT", "str", ["smo_out"], ("SMOOTH",), "Output file name for the smoothed cube"),
    K("SMOOTHSPEC", "flag", ["flagSmoothSpectral"], ("SMOOTHSPEC",), "Spectral smoothing of the datacube?",
      flag=True),
    K("WINDOW_TYPE", "ustr", ["window_type"], ("SMOOTHSPEC",),
      "Type of smoothing window (HANNING, HANNING2, BOXCAR, BARTLETT, WELCH, BLACKMAN, FLATTOP, GAUSSIAN)"),
    K("WINDOW_SIZE", "float", ["window_size"], ("SMOOTHSPEC",), "Size of smoothing window (channels)"),

    # ---------------------------------------------------------------- SLITFIT
    K("SLITWIDTH", "float", ["slitwidth"], SLIT, "Width of the slit (arcsec)"),
    K("SLITANGLE", "ustr", ["slitangle"], SLIT, "Angle from the slit (increasing position) to the receding half of "
      "the major axis, counterclockwise (degrees), or AUTO (0 or 180 from the data)"),
    K("WAVERANGE", "fvec", ["waverange"], SLIT, "Spectral range of the fit (two values in the units of the spectral "
      "axis; default: the lines over the velocity range of the emission)"),
    K("SKYRESID", "flag", ["skyresid"], SLIT, "Subtracting the median of the positions without line emission from "
      "each channel (a residual sky)?"),
    K("MASKWAVE", "fvec", ["maskwave"], SLIT, "Spectral ranges left out of the fit (pairs of values, e.g. sky "
      "lines)"),

    # ---------------------------------------------------------------- PVSLICE
    K("PVSLICE", "flag", ["flagPV"], ("PVSLICE",), "Extracting a position-velocity slice from a cube?",
      aliases=["flagpv", "pv"], flag=True),
    K("XPOS_PV", "lstr", ["XPOS_PV"], ("PVSLICE",), "X center of the slice"),
    K("YPOS_PV", "lstr", ["YPOS_PV"], ("PVSLICE",), "Y center of the slice"),
    K("PA_PV", "float", ["PA_PV"], ("PVSLICE",), "Position angle of the slice"),
    K("WIDTH_PV", "float", ["WIDTH_PV"], ("PVSLICE",), "Width of the slice (arcsec)"),
    K("P1_PV", "fvec2", ["P1_PV"], ("PVSLICE",), "Position 1 of a two-point slice (x y)"),
    K("P2_PV", "fvec2", ["P2_PV"], ("PVSLICE",), "Position 2 of a two-point slice (x y)"),
    K("ANTIALIAS", "float", ["ANTIALIAS"], ("PVSLICE",), "Type of anti-aliasing to use for slice"),

    # ---------------------------------------------------------------- REND3D
    K("REND3D", "flag", ["flagRend3D"], ("REND3D",), "Writing a 3D rendering of a datacube?", flag=True),
    K("RENDANGLE", "float", ["rendangle"], ("REND3D",), "Maximum azimuthal angle for rendering"),
]
del K

# Lookup table: lower-case name or alias -> Keyword
LOOKUP = {}
for _k in KEYWORDS:
    for _n in (_k.name,) + _k.aliases:
        if _n.lower() in LOOKUP:
            raise RuntimeError(f"Duplicated keyword {_n}")
        LOOKUP[_n.lower()] = _k

TASK_FLAGS = [k for k in KEYWORDS if k.flag]

# Keywords removed in BBarolo 2: an error that names the replacement
_BEAM_MSG = ("{} is no longer a parameter: give the beam of the data with BEAM (BMAJ BMIN BPA, e.g. BEAM 30 20 45; "
             "BMAJ and BMIN in arcsec, or in degrees with a final d, e.g. BEAM 0.01d 0.01d 45; one value is a "
             "circular beam), and the new beam of SMOOTH with NEWBEAM.")
REMOVED = {n.lower(): _BEAM_MSG.format(n) for n in ("BMAJ", "BMIN", "BPA", "BEAMFWHM", "OBMAJ", "OBMIN", "OBPA")}
REMOVED.update({old.lower(): f"{old} is no longer a parameter: it was renamed {new} (GALWIND, WINDFIT)."
                for old, new in (("HTOT", "CONELEN"), ("NTOT", "NLAYERS"))})
REMOVED.update({old.lower(): f"{old} is no longer a parameter: {new}." for old, new in (
    ("BAYESIAN3DFIT", "3DFIT with METHOD GLOBAL or BAYESIAN"), ("BAYESIANWIND", "WINDFIT"),
    ("BAYESIANSLITFIT", "SLITFIT"))})


def check_removed(name):
    """Raise BBaroloError if name is a keyword removed in BBarolo 2 (the message names its replacement)."""
    msg = REMOVED.get(str(name).strip().lower())
    if msg:
        from .bbutils import BBaroloError
        raise BBaroloError(msg)

# The full descriptions of the keywords (doc, group, default_text)
from .keywords_doc import DOCS as _DOCS, GROUPS  # noqa: E402
for _k in KEYWORDS:
    _e = _DOCS.get(_k.name, {})
    _k.doc, _k.group, _k.default_text = _e.get("doc"), _e.get("group"), _e.get("default")


def lookup(name):
    """The Keyword for a name or alias (case-insensitive), or None."""
    return LOOKUP.get(str(name).strip().lower())


def suggest(name, n=3):
    """Close matches of an unknown keyword."""
    return difflib.get_close_matches(str(name).lower(), list(LOOKUP), n=n, cutoff=0.6)


class UnknownParameterError(KeyError):
    """An unknown BBarolo parameter."""
    def __str__(self):
        return str(self.args[0]) if self.args else ""


def unknown_message(name):
    s = suggest(name)
    msg = f"unknown parameter '{name}'"
    if s:
        msg += " (did you mean " + " or ".join(f"'{lookup(x).name}'" if lookup(x).name.lower() == x
                                                 else f"'{x}' (={lookup(x).name})" for x in s) + "?)"
    return msg


# --------------------------------------------------------------------------
# Help
# --------------------------------------------------------------------------
TASK_NAMES = ("3DFIT", "WINDFIT", "GALMOD", "GALWIND", "SEARCH", "2DFIT", "ELLPROF", "SMOOTH",
              "SMOOTHSPEC", "MAPS", "PVSLICE", "SPACEPAR", "REND3D", "CONTSUB", "STATS", "MAKEMASK", "SLITFIT")
_CLASS_TASK = {"fitmod3d": "3DFIT", "galmod": "GALMOD", "galwind": "GALWIND", "search": "SEARCH",
               "fitmod2d": "2DFIT", "ellprof": "ELLPROF", "smooth": "SMOOTH", "smoothspec": "SMOOTHSPEC",
               "maps": "MAPS", "pvslice": "PVSLICE", "spacepar": "SPACEPAR", "rend3d": "REND3D",
               "contsub": "CONTSUB", "stats": "STATS", "makemask": "MAKEMASK",
               "windfit": "WINDFIT", "slitfit": "SLITFIT",
               "bayesian3dfit": "3DFIT", "bayesianwind": "WINDFIT", "bayesianslitfit": "SLITFIT"}

# The methods of the fit tasks (METHOD), and the default of each task
METHODS = {"3DFIT": ("CLASSIC", "GLOBAL", "BAYESIAN"), "WINDFIT": ("GLOBAL", "BAYESIAN"),
           "SLITFIT": ("GLOBAL", "BAYESIAN")}
DEFAULT_METHOD = {"3DFIT": "CLASSIC", "WINDFIT": "GLOBAL", "SLITFIT": "GLOBAL"}


def split_tag(tag):
    """(task, method) of an entry of Keyword.tasks: '3DFIT/CLASSIC' -> ('3DFIT', 'CLASSIC'), 'GALMOD' ->
    ('GALMOD', None) (all the methods)."""
    task, _, method = tag.partition("/")
    return task, method or None


def methods_for(kw, task):
    """The methods of the fit task with which the keyword kw is used (a tuple, in the order of METHODS), or None
    if it is used with all of them (or task has no METHOD, or kw is not used by task)."""
    ms = set()
    for t, m in map(split_tag, kw.tasks):
        if t == task:
            if m is None:
                return None
            ms.add(m)
    return tuple(m for m in METHODS.get(task, ()) if m in ms) or None


def uses(kw, task, method=None):
    """Whether the keyword kw is used by task (with METHOD method, if given)."""
    return any(t == task and (m is None or method is None or m == method) for t, m in map(split_tag, kw.tasks))


def _fmtval(v):
    if v is None:
        return ""                                   # (not given: no line in parameter files)
    if isinstance(v, bool):
        return stringize(v)
    if isinstance(v, float):
        return f"{v:g}"
    if isinstance(v, (list, tuple)):
        return " ".join(_fmtval(x) for x in v)
    s = str(v)
    return s if s != "" else '""'


def _task_of(name):
    """Task name for a task, a class or a class name; None otherwise."""
    if not isinstance(name, str):
        cls = name if isinstance(name, type) else type(name)
        if not hasattr(cls, "TASK") or not hasattr(cls, "FLAGS"):
            return None
        return _CLASS_TASK.get(cls.__name__.lower(), getattr(cls, "TASK", None) or "MAPS")
    n = str(name).strip()
    if n.upper() in TASK_NAMES:
        return n.upper()
    return _CLASS_TASK.get(n.lower())


# Defaults that differ for a task, or a task with a METHOD (TASK/METHOD; applied by the classes of the fits, e.g.
# Bayesian3DFIT._all_params): see task_defaults
# FREE of 3DFIT with METHOD GLOBAL or BAYESIAN when not given (the FREE of Param is that of METHOD CLASSIC)
BAYES_DEFAULT_FREE = "VROT VDISP XPOS_SINGLE YPOS_SINGLE VSYS_SINGLE INC_SINGLE PA_SINGLE Z0_SINGLE RADMAX FRACERR"
# FREE of SLITFIT when not given
SLIT_DEFAULT_FREE = "VROT VDISP VSYS XPOS"
_LIKE3_DEFAULTS = {"GALMODTYPE": "ANALYTIC", "FREE": BAYES_DEFAULT_FREE}
TASK_DEFAULTS = {"3DFIT/GLOBAL": _LIKE3_DEFAULTS, "3DFIT/BAYESIAN": _LIKE3_DEFAULTS,
                 "SLITFIT": {"FREE": SLIT_DEFAULT_FREE, "NORM": "AZIM"}}


def task_defaults(task, method=None):
    """The defaults of the keywords that differ for task with METHOD method (None: the default method), as
    {keyword: value as in a parameter file}."""
    method = method or DEFAULT_METHOD.get(task)
    return {**TASK_DEFAULTS.get(task, {}), **TASK_DEFAULTS.get(f"{task}/{method}", {})}


def _used_by(kw):
    """The tasks of kw as text, with the methods of the fit tasks: '3DFIT (METHOD CLASSIC), SPACEPAR'."""
    out = []
    for tag in kw.tasks:
        t = split_tag(tag)[0]
        if t == "ALL" or any(o.startswith(t) for o in out):
            continue
        ms = methods_for(kw, t)
        out.append(t + (f" (METHOD {' or '.join(ms)})" if ms else ""))
    return ", ".join(out)


def describe(kw, default=None):
    """One keyword as text: name, aliases, default, tasks and description (the full one, kw.doc, if the
    keyword has it, else its one-line help)."""
    import textwrap
    s = f"{kw.name}"
    if kw.aliases:
        s += f" (also: {', '.join(kw.aliases)})"
    if kw.doc:
        paras = [p.strip() for p in kw.doc.strip().split("\n\n")]
        out = []
        for p in paras:
            if p.lstrip().startswith(("- ", "* ", "1. ")):        # lists: keep their lines
                out.append("\n".join("    " + ln.strip() for ln in p.splitlines()))
            else:
                out.append(textwrap.fill(" ".join(p.split()), 96, initial_indent="    ", subsequent_indent="    "))
        s += "\n" + "\n\n".join(out)
    else:
        s += f"\n    {kw.help}."
    if kw.default_text:
        s += f"\n    Default: {kw.default_text}."
    elif default is not None:
        s += f"\n    Default: {_fmtval(default)}"
    tasks = "all tasks" if kw.tasks == ("ALL",) else _used_by(kw)
    s += f"   Used by: {tasks}"
    return s


def keywords_for(task, method=None):
    """The keywords used by a task (task name as in TASK_NAMES), with METHOD method if given (otherwise with any
    of its methods)."""
    task = task.upper()
    method = method.upper() if method else None
    maps = task == "MAPS"
    out = []
    for k in KEYWORDS:
        if maps and any(t in MAPS for t in k.tasks):
            out.append(k)
        elif uses(k, task, method):
            out.append(k)
    return out


def help(name=None, stream=None):
    """Help on BBarolo's parameters.

    help()          all the keywords, with their defaults
    help('free')    one keyword (name or alias, case-insensitive)
    help('3DFIT')   the keywords of a task (also a class, e.g. help(FitMod3D), or 'FitMod3D')
    help(obj)       Python's help for any other object
    """
    import sys
    import builtins
    from .params import _default_param
    if name is not None and not isinstance(name, str) and _task_of(name) is None:
        return builtins.help(name)                  # any other Python object
    st = stream or sys.stdout
    d = _default_param()
    if name is None:
        st.write("\nBBarolo parameters (keyword: description; default). Keywords are case-insensitive.\n"
                 "help('KEYWORD') gives the details of one parameter, help('TASK') the parameters of a task.\n\n")
        for k in KEYWORDS:
            st.write(f"  {k.name:18s}{k.help} [{_fmtval(k.value(d))}]\n")
        st.write("\n")
        return
    task = _task_of(name)
    kw = lookup(name) if isinstance(name, str) else None
    if task is not None:
        st.write(f"\nParameters of {task}" + (" (task flag set automatically by its class)" if task != "MAPS" else "")
                 + ":\n\n")
        tdef = task_defaults(task)
        if task in DEFAULT_METHOD:
            tdef.setdefault("METHOD", DEFAULT_METHOD[task])
        kws = [k for k in keywords_for(task) if k.name != task]
        sections = [None] + sorted({methods_for(k, task) for k in kws} - {None},
                                   key=lambda ms: [METHODS[task].index(m) for m in ms])
        for ms in sections:
            if ms is not None:
                st.write(f"\n  With METHOD {' or '.join(ms)} only:\n")
                tdef = task_defaults(task, ms[0])
            for k in kws:
                if methods_for(k, task) == ms:
                    st.write(f"  {k.name:18s}{k.help} [{_fmtval(tdef.get(k.name, k.value(d)))}]\n")
        st.write("\nGeneral parameters: " + ", ".join(k.name for k in KEYWORDS if k.tasks[0] == "ALL") + "\n\n")
        if kw is not None and kw.name != task:
            st.write(describe(kw, kw.value(d)) + "\n\n")
        return
    if kw is None:
        st.write("\n" + unknown_message(name)[0].upper() + unknown_message(name)[1:] + ".\n\n")
        return
    st.write("\n" + describe(kw, kw.value(d)) + "\n\n")
