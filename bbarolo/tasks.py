"""
The Python interface of BBarolo: one class per task, and :func:`run`.

Every class takes a datacube and any BBarolo parameter as keyword arguments (the same
keywords as in a parameter file, case-insensitive; see :func:`bbarolo.help`):

    >>> from bbarolo import FitMod3D
    >>> f = FitMod3D("ngc2403.fits", radii=np.arange(15, 1200, 30), xpos=77, ypos=77,
    ...              vsys=132.8, vrot=120, vdisp=8, z0=10, inc=60, pa=123.7,
    ...              free="VROT VDISP", mask="SEARCH", distance=3.2, threads=4)
    >>> res = f.run()
    >>> res.rings, res.model, res.maps

The tasks run exactly the same code as the command line ``BBarolo -p file.par``: a
class with some parameters gives the same results as a parameter file with the same
parameters (and the task flag). The outputs are written in OUTFOLDER as usual and are
also returned in a :class:`Result`. With ``run(write=False)`` they go to a temporary
folder that is deleted with the Result (and the plots are off unless PLOTS is set).

:func:`run` runs a parameter file (or parameters) with all its tasks, like the
command line, in the same Python process.
"""

import os
import io
import sys
import copy
import time
import shutil
import tempfile
import contextlib
import numpy as np
from astropy.io import fits

from .params import Param, value_to_string
from . import keywords as _kw
from .bbutils import mkdirp, BBaroloError

__all__ = ['FitMod3D', 'GalMod', 'GalWind', 'Search', 'FitMod2D', 'Ellprof', 'Smooth', 'SmoothSpec', 'Maps',
           'PVSlice', 'SpacePar', 'Rend3D', 'ContSub', 'Stats', 'MakeMask', 'run', 'Result', 'Task',
           'read_parfile']


# ==========================================================================
# The engine: the steps of BBarolo, used by both the command line and the classes
# ==========================================================================
def _step_contsub(c, par):
    p = c.pars()
    c.continuumSubtract()
    c.fitswrite_3d(p.outFolder + p.outPrefix + "_contsub.fits", True)
    return {}


def _step_checkcube(c, par):
    c.CheckCube(par.checkCube)
    return {}


def _step_stats(c, par):
    p = c.pars()
    c.setCubeStats()
    c.stat().tofile(p.outFolder + p.outPrefix + "_stats.txt")
    return {"stats": c.stat()}


def _step_makemask(c, par):
    c.BlankMask(None, "LARGEST" in par.MaskType)
    return {}


def _step_smooth(c, par):
    from .smooth import Smooth3D
    sm = Smooth3D()
    sm.cubesmooth(c)
    sm.fitswrite()
    return {}


def _step_smoothspec(c, par):
    from .smooth import SpectralSmooth3D
    sm = SpectralSmooth3D(par.window_type, par.window_size)
    sm.smooth(c)
    sm.fitswrite(c)
    return {}


def _step_search(c, par):
    p = c.pars()
    c.search()
    with open(p.outFolder + "detections.txt", "w") as f:
        c.printDetections(f)
    c.writeDetections()
    if p.se.cubelets:
        c.writeCubelets()
        c.plotDetections()
    return {"detections": list(c.sources.objectList), "table": _detections_table(c)}


def _step_3dfit(c, par):
    from .galfit import Galfit
    fit = Galfit(c)
    fit.galfit()
    if fit.par.TWOSTAGE:
        fit.SecondStage()
    if par.plots >= 0:
        fit.writeModel("BOTH" if par.debug else fit.Norms(2), par.plots)
    return {"galfit": fit}


def _step_bayesian(c, par):
    from .bayesian import run_task
    f = run_task(c, par)
    return {"bayes": f, "galfit": f._galfit}


def _galmod(c, par, write=True, seed=None):
    """The GALMOD task on the cube c: the rings from the parameters (initial_rings), the model
    engine (DiskModel, ring convention RINGMODE) and the final model, normalised as NORM. write=True: all the outputs (model, maps, PV
    slices, densprof.txt, ...); write=False: no files, and the mask only if it is needed.
    Returns (model Cube, DiskModel, rings, NORM)."""
    from .galmod import DiskModel
    from .galfit import initial_rings, final_norm
    from . import galfit_out
    p = c.pars()
    gf = copy.deepcopy(p.gf)          # initial_rings fills it in; the next tasks see the parameters as given
    norm_par = final_norm(gf.NORM, "GALMOD", extra=("BOTH",))
    norm = "BOTH" if (write and par.debug) else norm_par
    c.checkBeam()
    if (write or norm != "NONE") and not c.MaskAll():
        c.BlankMask()
    rings = initial_rings(c, gf, galmod=True, write=write).rings
    from .galfit import check_vsys
    check_vsys(c, rings, fatal=False)
    # DENS is a flux (DENSFLUX=false: a column density, used with NORM=NONE only, since the
    # normalisations to the data work with fluxes)
    dm = DiskModel(c, par=gf, densflux=gf.DENSFLUX if norm == "NONE" else True)
    if write:
        mod = galfit_out.writeModel(c, dm, rings, norm, c.Mask(), par=gf)
    else:
        mod = galfit_out.finalModel(c, dm, rings, norm, c.Mask(), par=gf, seed=seed)
    return mod, dm, rings, norm_par


def _step_galmod(c, par):
    _, dm, rings, norm = _galmod(c, par, write=True)
    return {"diskmodel": dm, "rings": rings, "norm": norm}


def _step_spacepar(c, par):
    from .spacepar import Spacepar
    sp = Spacepar(c)
    sp.calculate()
    sp.plotAll_Python()
    return {"spacepar": sp}


def _step_galwind(c, par):
    from .galwind import GalWind as _GalWind
    p = c.pars()
    w = _GalWind(c)
    w.compute()
    if p.gf.SM:
        w.smooth()
    w.writeFITS()
    w.writeMomentMaps()
    w.writePV()
    if par.plots:
        w.makePlots()
    return {"galwind": w}


def _step_2dfit(c, par):
    from .ringmodel import Ringmodel
    p = c.pars()
    of, pre = p.outFolder, p.outPrefix
    tr = Ringmodel(c)
    tr.ringfit(p.threads, p.verbose, p.showbar)
    with open(of + pre + "_2dtrm.txt", "w") as f:
        tr.printfinal(f, c.Head())
    if p.verbose:
        tr.printfinal(sys.stdout, c.Head())
    tr.writeModel(of + pre + "_2d_mod.fits", c.Head())
    return {"ringmodel": tr}


def _step_maps(c, par):
    from .moments import MomentMap
    from .image import Image2D
    p = c.pars()
    of, pre = p.outFolder, p.outPrefix
    s = of + pre
    masking = par.MaskType != "NONE"
    ma = p.ma
    m = MomentMap()
    m.input(c)
    if ma.totalmap:
        m.ZeroMoment(masking, ma.maptype)
        m.fitswrite_2d(s + "map_0th.fits")
        if ma.SNmap:
            m.SNMap(masking)
    if ma.velocitymap:
        m.FirstMoment(masking, ma.maptype)
        m.fitswrite_2d(s + "map_1st.fits")
    if ma.dispersionmap:
        m.SecondMoment(masking, ma.maptype)
        m.fitswrite_2d(s + "map_2nd.fits")
    if ma.massdensmap:
        m.HIMassDensityMap(masking)
        m.fitswrite_2d(s + "map_massdens.fits")
    if ma.rmsmap:
        m.RMSMap(masking)
        m.fitswrite_2d(s + "map_RMS.fits")
    if ma.globprof:
        spec = Image2D().extractGlobalSpectrum(c)
        zphys = np.array([c.getZphys(i) for i in range(c.DimZ())])
        with open(of + "spectrum.txt", "w") as f:
            for z, v in zip(zphys, spec):
                f.write(f"{z:g} {v:g}\n")
    return {}


def _step_pvslice(c, par):
    from .pvslice import PvSlice
    p = c.pars()
    pv = PvSlice(c)
    pv.slice()
    pv.fitswrite_2d(p.outFolder + p.outPrefix + f"pv_{par.PA_PV:.0f}.fits", True)
    return {}


def _step_reduce(c, par):
    p = c.pars()
    red = c.Reduce(int(np.floor(par.factor)), "spectral" if par.reduce == "spectral" else "spatial")
    red.fitswrite_3d(p.outFolder + p.outPrefix + "_red.fits", True)
    return {}


def _step_slitfit(c, par):
    print(" SLITFIT task is not available in the Python version (it was broken in C++ BBarolo too).")
    return {}


def _step_ellprof(c, par):
    from .ellprof import Ellprof as _Ellprof
    p = c.pars()
    of, pre = p.outFolder, p.outPrefix
    ell = _Ellprof(c)
    ell.RadialProfile()
    with open(of + pre + "_densprof.txt", "w") as f:
        ell.printProfile(f, ell.getNseg() - 1)
    if p.verbose:
        ell.printProfile(sys.stdout, ell.getNseg() - 1)
    ell.writeMap(of + pre + "_densmap.fits")
    return {"ellprof": ell}


def _step_rend3d(c, par):
    from .rendering3d import Rendering3D
    r3d = Rendering3D(c)
    r3d.compute(par.rendangle)
    r3d.writefits()
    return {}


# (name, is it requested?, function), in the order in which BBarolo runs them.
# p is the Param of the cube (a copy of par, with the output folder and prefix).
STEPS = [
    ("CONTSUB", lambda par, p: par.contsub, _step_contsub),
    ("CHECKCUBE", lambda par, p: par.checkCube, _step_checkcube),
    ("STATS", lambda par, p: par.flagStats, _step_stats),
    ("MAKEMASK", lambda par, p: par.makeMask, _step_makemask),
    ("SMOOTH", lambda par, p: par.flagSmooth, _step_smooth),
    ("SMOOTHSPEC", lambda par, p: par.flagSmoothSpectral, _step_smoothspec),
    ("SEARCH", lambda par, p: p.se.flagSearch, _step_search),
    ("3DFIT", lambda par, p: p.gf.flagGALFIT, _step_3dfit),
    ("BAYESIAN3DFIT", lambda par, p: p.gf.flagBAYESIAN, _step_bayesian),
    ("GALMOD", lambda par, p: p.gm.flagGALMOD, _step_galmod),
    ("SPACEPAR", lambda par, p: par.flagSpace, _step_spacepar),
    ("GALWIND", lambda par, p: p.gw.flagGALWIND, _step_galwind),
    ("2DFIT", lambda par, p: par.flagRing, _step_2dfit),
    ("MAPS", lambda par, p: p.getMaps(), _step_maps),
    ("PVSLICE", lambda par, p: par.flagPV, _step_pvslice),
    ("REDUCE", lambda par, p: par.getflagReduce() and not (par.flagSmooth or par.flagSmoothSpectral), _step_reduce),
    ("SLITFIT", lambda par, p: par.flagSlitfit, _step_slitfit),
    ("ELLPROF", lambda par, p: par.flagEllProf, _step_ellprof),
    ("REND3D", lambda par, p: par.flagRend3D, _step_rend3d),
]


def run_steps(c, par, collect=False):
    """Run all the requested tasks on the cube c (port of the task part of BBcore).
    With collect=True, return {step name: (outputs in memory, files written)}."""
    p = c.pars()
    if par.AUTO and not p.gf.flagBAYESIAN:          # (BBarolo -f cube.fits BAYESIAN3DFIT=true: no 3DFIT)
        p.gf.flagGALFIT = True
    out = {}
    for name, requested, step in STEPS:
        if not requested(par, p):
            continue
        before = _snapshot(p.outFolder) if collect else None
        res = step(c, par)
        if collect:
            out[name] = (res, _written(p.outFolder, before))
    return out


@contextlib.contextmanager
def _logfile(par, outfolder):
    if not par.logFile:
        yield
        return
    from .cli import welcomeMessage
    i = 0
    logfile = outfolder + "BBlog_0.txt"
    while os.path.exists(logfile):
        i += 1
        logfile = outfolder + f"BBlog_{i}.txt"
    with open(logfile, "w") as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
        if par.verbose:
            welcomeMessage()
            print(par)
        yield


def BBcore(par, collect=False):
    """Run all the tasks requested in par on par.imageFile (port of BBcore).
    Returns False if the cube can not be read; otherwise True, or with collect=True
    the outputs of each step (see run_steps)."""
    from .cube import Cube
    from .header import Header
    c = Cube(par=par)
    p = c.pars()
    outfolder = p.outFolder
    if outfolder == "":
        try:
            h = Header(p.imageFile, warning=False)
        except Exception:
            print(f"{p.imageFile} is not a readable FITS file!")
            return False
        outfolder = os.getcwd() + "/output/" + h.Name() + "/"
        p.setOutfolder(outfolder)
    if p.verbose and os.path.isdir(outfolder) and os.listdir(outfolder):
        print(f" NOTE: the output folder {outfolder} already exists: its files will be overwritten.")
    mkdirp(outfolder)
    with _logfile(par, outfolder):
        if par.logFile:
            p.showbar = False
        try:
            ok = c.readCube(p.imageFile)
        except BBaroloError:
            raise                                # (e.g. no beam: a clear message, not "not readable")
        except Exception as e:
            print(e)
            ok = False
        if not ok:
            print(f"{p.imageFile} is not a readable FITS file!")
            return False
        out = run_steps(c, par, collect)
    return (out, c) if collect else True


def run_par(par, banner=True, collect=False):
    """Run a checked Param on all its images, as the command line does (welcome message,
    parameters and execution time if verbose). Returns (ok, list of outputs per image)."""
    t0 = time.time()
    from .cli import welcomeMessage
    verbose = par.verbose
    if banner and verbose and not par.logFile:
        welcomeMessage()
        if verbose >= 2:
            print(par)
    images = list(par.images)
    allout = []
    for im, img in enumerate(images):
        if len(images) > 1 and verbose and not par.logFile:
            print("\n" + "_" * 69 + "\n")
            print(f"Working on {img} ".rjust(70) + "\n")
            print(f" File {im + 1} of {len(images)}\n")
        par.imageFile = img
        r = BBcore(par, collect)
        if not r:
            if len(images) - im > 1:
                print("Skipping to next file...")
            else:
                print("Exiting ...\n")
                return False, allout
        allout.append(r)
    t = time.time() - t0
    if verbose and banner:
        if t < 1:
            print(f"\nExecution time: {t:.2g} sec.")
        else:
            print(f"\nExecution time: {int(t / 60)} min and {int(t) % 60} sec.")
    return True, allout


# ==========================================================================
# Outputs
# ==========================================================================
def _snapshot(folder):
    out = {}
    for root, _, files in os.walk(folder):
        for f in files:
            fn = os.path.join(root, f)
            try:
                st = os.stat(fn)
                out[fn] = (st.st_mtime_ns, st.st_size)
            except OSError:
                pass
    return out


def _written(folder, before):
    after = _snapshot(folder)
    return sorted(f for f, s in after.items() if before.get(f) != s)


def load_fits(fname):
    """The first HDU of a FITS file, with the data in memory (the file is closed)."""
    with fits.open(fname, memmap=False) as h:
        return fits.PrimaryHDU(data=None if h[0].data is None else np.array(h[0].data),
                               header=h[0].header.copy())


_COLRENAME = {("rad", "kpc"): "rad_kpc", ("rad", "arcs"): "radii", ("radius", "pix"): "rad_pix",
              ("radius", "arcs"): "radii", ("radius", "arcsec"): "radii", ("z0", "pc"): "z0_pc",
              ("z0", "arcs"): "z0", ("disp", None): "vdisp", ("p.a.", None): "pa", ("incl.", None): "inc",
              ("vexp", None): "vrad", ("xcenter", None): "xpos", ("ycenter", None): "ypos",
              ("sig", "e20"): "dens", ("e_disp1", None): "e_vdisp1", ("e_disp2", None): "e_vdisp2"}
_UNITS = {"km/s": "km/s", "kms": "km/s", "deg": "deg", "arcs": "arcsec", "arcsec": "arcsec", "pc": "pc",
          "kpc": "kpc", "pix": "pix", "msun/pc2": "solMass/pc2", "1e20/cm2": "1e20/cm2"}


def read_table(fname):
    """A BBarolo text table (rings, profiles, 2D fit) as an astropy Table. Column names
    are lower case (e.g. radii, vrot, vdisp, inc, pa, xpos, ypos, vsys, e_vrot1, ...)."""
    from astropy.table import Table
    import astropy.units as u
    comments, rows = [], []
    with open(fname) as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith("#"):
                comments.append(s[1:].split())
            else:
                rows.append(s.split())
    if rows:
        ncol = len(rows[0])
        data = np.array([[float(x) for x in r[:ncol]] for r in rows if len(r) >= ncol])
    else:
        # no rows (e.g. no ring could be fitted): the columns of the header, with no values
        head = next((c for c in reversed(comments) if c), None)
        if not head:
            return Table()
        ncol = len(head)
        data = np.zeros((0, ncol))
    # the header is the last comment line with one word per column, or the last two (names
    # and units) when they are consecutive
    cand = [i for i, c in enumerate(comments) if len(c) == ncol]
    names, units = None, [None] * ncol
    if cand:
        i = cand[-1]
        if i - 1 in cand:
            names, units = comments[i - 1], [x.strip("[]") for x in comments[i]]
        else:
            names = comments[i]
    if names is None:
        names = [f"col{i + 1}" for i in range(ncol)]
    out, seen = [], set()
    for n, un in zip(names, units):
        if "(" in n and n.endswith(")"):
            n, un = n[:n.index("(")], n[n.index("(") + 1:-1]
        key = n.lower()
        uu = un.lower() if un else None
        new = _COLRENAME.get((key, uu), _COLRENAME.get((key, None), key))
        if new in seen:
            new = f"{new}_{uu}" if uu else new + "_2"
        seen.add(new)
        out.append((new, un))
    t = Table(data, names=[n for n, _ in out])
    for (n, un) in out:
        un = _UNITS.get(un.lower()) if un else None
        if un:
            try:
                t[n].unit = u.Unit(un)
            except ValueError:
                pass
    return t


def _detections_table(c):
    """Table of the detections of SEARCH (the columns of detections.txt)."""
    from astropy.table import Table
    from .bbutils import AlltoVel
    h, stats = c.Head(), c.stat()
    rows = []
    for i in range(c.getNumObj()):
        o = c.pObject(i)
        xc, yc, zc = o.getXcentre(), o.getYcentre(), o.getZcentre()
        try:
            w = h.pixToWCS(np.array([[xc, yc, zc]]))[0]
            ra, dec = float(w[0]) % 360., float(w[1])
        except Exception:
            ra = dec = np.nan
        rows.append((i + 1, xc, yc, zc, ra, dec, AlltoVel(c.getZphys(zc), h),
                     abs(o.getXmax() - o.getXmin()) + 1, abs(o.getYmax() - o.getYmin()) + 1,
                     abs(o.getZmax() - o.getZmin()) + 1, o.getSpatialSize(), o.getSize(), o.getVsys(),
                     o.getW20(), o.getIntegFlux(), o.getPeakFlux() / stats.getSpread(), o.getPeakFlux()))
    names = ("id", "xcenter", "ycenter", "zcenter", "ra", "dec", "vel", "xwidth", "ywidth", "zwidth", "npix",
             "nvox", "vsys", "w20", "flux", "peaksnr", "peakflux")
    if not rows:
        return Table(names=names)
    return Table(rows=rows, names=names)


class Result:
    """The outputs of a BBarolo task.

    The attributes depend on the task (``keys()`` lists them): astropy HDUs for cubes and
    maps (dictionaries of HDUs for several maps), astropy Tables for text tables, plus
    ``files`` (the files written, full paths) and ``outfolder``. ``open(name)`` loads any
    written FITS file whose name contains ``name``. With ``run(write=False)`` the files
    are in a temporary folder that is deleted with the Result.
    """

    def __init__(self, task, outfolder, files=(), tmpdir=None):
        self.task = task
        self.outfolder = outfolder
        self.files = list(files)
        self._tmpdir = tmpdir
        self._keys = []
        self._lazy = {}

    def _set(self, name, value):
        if value is None:
            return
        setattr(self, name, value)
        if name not in self._keys:
            self._keys.append(name)

    def _set_lazy(self, name, fn, what):
        """An attribute computed when it is first used (e.g. a cube): fn() gives it, what describes it."""
        self._lazy[name] = (fn, what)
        if name not in self._keys:
            self._keys.append(name)

    def __getattr__(self, name):
        lazy = self.__dict__.get("_lazy", {})
        if name in lazy:
            v = lazy.pop(name)[0]()
            setattr(self, name, v)
            return v
        raise AttributeError(f"'Result' object has no attribute '{name}'")

    def keys(self):
        return list(self._keys)

    def __getitem__(self, name):
        if name not in self._keys:
            raise KeyError(name)
        return getattr(self, name)

    def __contains__(self, name):
        return name in self._keys

    def open(self, name):
        """Load the written FITS file whose name contains ``name`` (e.g. 'pv_a', 'mod_local')."""
        cand = [f for f in self.files if name in os.path.basename(f) and f.endswith((".fits", ".fits.gz"))]
        if not cand:
            raise FileNotFoundError(f"No FITS output with '{name}' in its name. Files: "
                                    + ", ".join(os.path.basename(f) for f in self.files))
        cand.sort(key=lambda f: (os.path.basename(f) != name, len(f)))
        return load_fits(cand[0])

    def plot(self):
        """Make (again) the plots of a 3DFIT result; returns the PDF files."""
        fit = getattr(self, "galfit", None)
        if fit is None:
            raise ValueError(f"No plots available for the {self.task} task.")
        before = _snapshot(self.outfolder)
        fit.cin.pars().plots = 1
        fit.plotAll_Python()
        new = _written(self.outfolder, before)
        self.files = sorted(set(self.files) | set(new))
        return [f for f in self.files if f.endswith(".pdf")]

    @staticmethod
    def _describe(v):
        if isinstance(v, (fits.PrimaryHDU, fits.ImageHDU)):
            return "HDU " + ("x".join(str(s) for s in v.data.shape) if v.data is not None else "empty")
        if hasattr(v, "colnames") and hasattr(v, "__len__"):
            return f"Table ({len(v)} rows: {', '.join(v.colnames[:6])}{', ...' if len(v.colnames) > 6 else ''})"
        if isinstance(v, dict):
            return "dict of " + str(len(v)) + (": " + ", ".join(list(v)[:6]) + (", ..." if len(v) > 6 else "")
                                               if v else "")
        if isinstance(v, list):
            return f"list ({len(v)})"
        return type(v).__name__

    def __repr__(self):
        lines = [f"<Result of {self.task}: {len(self.files)} files in {self.outfolder}"]
        w = max([14] + [len(k) for k in self._keys])
        for k in self._keys:
            d = self._lazy[k][1] + " (computed when used)" if k in self._lazy else self._describe(getattr(self, k))
            lines.append(f"   {k:{w}s} {d}")
        return "\n".join(lines) + ">"

    # ------------------------------------------------------------------ plots
    def _need(self, *names):
        miss = [n for n in names if n not in self]
        if miss:
            raise ValueError(f"No {', '.join(miss)} in this Result of {self.task}.")

    def plot_rings(self, params=None):
        """Ring parameters against radius (with the errors, if any): VROT, VDISP and the other free
        parameters, or the list params (e.g. ['vrot', 'inc']). Returns a matplotlib Figure."""
        import matplotlib.pyplot as plt
        self._need("rings")
        t = self.rings
        if params is None:
            params = ["vrot", "vdisp"] + [k for k in ("inc", "pa", "z0", "vsys", "xpos", "ypos", "vrad", "dens")
                                          if f"e_{k}1" in t.colnames or (k in t.colnames and len(t) > 1
                                                                           and np.ptp(np.asarray(t[k])) > 0)]
        units = dict(vrot="km/s", vdisp="km/s", vrad="km/s", vsys="km/s", inc="deg", pa="deg", z0="arcsec",
                     xpos="pix", ypos="pix", dens="")
        fig, axs = plt.subplots(len(params), 1, sharex=True, figsize=(6, 1.9 * len(params) + 0.6), squeeze=False)
        r = np.asarray(t["radii"])
        for ax, k in zip(axs[:, 0], params):
            v = np.asarray(t[k])
            err = None
            if f"e_{k}1" in t.colnames:
                err = [-np.asarray(t[f"e_{k}1"]), np.asarray(t[f"e_{k}2"])]
            ax.errorbar(r, v, yerr=err, fmt="o-", ms=3, lw=1, capsize=2)
            ax.set_ylabel(f"{k.upper()}" + (f" ({units.get(k)})" if units.get(k) else ""))
        axs[-1, 0].set_xlabel("Radius (arcsec)")
        fig.tight_layout()
        return fig

    def _map_pair(self, kind):
        """(data, model) HDUs of a map ('0mom', ...) or PV slice ('pv_a', 'pv_b')."""
        d = self.maps if kind.endswith("mom") else self.pvs
        mod = sorted(k for k in d if k == "mod_" + kind or k.startswith("mod_" + kind + "_"))   # e.g. mod_0mom_azim
        return d[kind], (d[mod[0]] if mod else None)

    def plot_maps(self):
        """The moment maps of the data (top) and of the model (bottom): total intensity, velocity and
        dispersion. Returns a matplotlib Figure."""
        import matplotlib.pyplot as plt
        self._need("maps")
        kinds = [k for k in ("0mom", "1mom", "2mom") if k in self.maps]
        fig, axs = plt.subplots(2, len(kinds), figsize=(3.6 * len(kinds), 6.4), squeeze=False)
        titles = {"0mom": "Total intensity", "1mom": "Velocity (km/s)", "2mom": "Dispersion (km/s)"}
        for j, k in enumerate(kinds):
            d, m = self._map_pair(k)
            dd = np.asarray(d.data, float)
            fin = np.isfinite(dd)
            lo, hi = (np.nanpercentile(dd[fin], [1, 99]) if fin.any() else (0, 1))
            for i, (h, lab) in enumerate(((d, "data"), (m, "model"))):
                ax = axs[i, j]
                if h is not None:
                    im = ax.imshow(np.asarray(h.data, float), origin="lower", vmin=lo, vmax=hi,
                                   cmap="RdBu_r" if k == "1mom" else "viridis")
                    fig.colorbar(im, ax=ax, shrink=0.8)
                ax.set_title(f"{titles[k]}, {lab}", fontsize=9)
                ax.set_xticks([]), ax.set_yticks([])
        fig.tight_layout()
        return fig

    def plot_pv(self):
        """Position-velocity slices along the major (pv_a) and minor (pv_b) axes: the data in grey and
        the model in red contours. Returns a matplotlib Figure."""
        import matplotlib.pyplot as plt
        self._need("pvs")
        kinds = [k for k in ("pv_a", "pv_b") if k in self.pvs]
        fig, axs = plt.subplots(1, len(kinds), figsize=(5.2 * len(kinds), 4), squeeze=False)
        for ax, k in zip(axs[0], kinds):
            d, m = self._map_pair(k)
            h = d.header
            ny, nx = d.data.shape
            off = (np.arange(nx) + 1 - h.get("CRPIX1", 1)) * h.get("CDELT1", 1) + h.get("CRVAL1", 0)
            off = off * (3600. if str(h.get("CUNIT1", "deg")).lower().startswith("deg") else 1.)
            vel = (np.arange(ny) + 1 - h.get("CRPIX2", 1)) * h.get("CDELT2", 1) + h.get("CRVAL2", 0)
            if str(h.get("CUNIT2", "")).lower() in ("m/s", "ms"):
                vel = vel / 1000.
            dd = np.asarray(d.data, float)
            mm = np.asarray(m.data, float) if m is not None else None
            if vel[-1] < vel[0]:                 # velocity increasing upwards
                vel, dd = vel[::-1], dd[::-1]
                mm = mm[::-1] if mm is not None else None
            ext = [off[0], off[-1], vel[0], vel[-1]]
            fin = np.isfinite(dd) & (dd != 0)
            vmax = np.nanpercentile(dd[fin], 99.5) if fin.any() else 1.
            ax.imshow(dd, origin="lower", extent=ext, aspect="auto", cmap="Greys", vmin=0, vmax=vmax)
            if mm is not None and np.nanmax(mm) > 0:
                ax.contour(mm, levels=np.nanmax(mm) * np.array([0.05, 0.15, 0.3, 0.5, 0.75]), origin="lower",
                           extent=ext, colors="red", linewidths=0.8)
            ax.set_xlabel("Offset (arcsec)")
            ax.set_ylabel("Velocity (km/s)")
            ax.set_title("Major axis" if k == "pv_a" else "Minor axis")
        fig.tight_layout()
        return fig


def _base(f, prefix):
    """File name without folder, prefix and extension."""
    b = os.path.basename(f)
    for ext in (".fits.gz", ".fits", ".txt"):
        if b.endswith(ext):
            b = b[:-len(ext)]
    if prefix and b.startswith(prefix):
        b = b[len(prefix):]
    return b.lstrip("_") or b


def _rotation_curve(t):
    """The rotation curve of a ring table: radius (arcsec, and kpc), VROT and its errors (lower and upper,
    as positive numbers: the table has e_vrot1 < 0 and e_vrot2 > 0)."""
    from astropy.table import Table
    rc = Table()
    for k in ("radii", "rad_kpc", "vrot"):
        if k in t.colnames:
            rc[k] = t[k]
    if "e_vrot1" in t.colnames:
        rc["e_vrot_lo"] = -t["e_vrot1"]
        rc["e_vrot_hi"] = t["e_vrot2"]
    return rc


def _add_data(res, c):
    """res.data (the cube that was analysed, after CONTSUB if any) and res.residual (data - model,
    NaN outside the mask), computed when first used."""
    if c is None or getattr(c, "array", None) is None:
        return
    def data():
        return fits.PrimaryHDU(data=c.array, header=c.Head().toAstropy(3, True))

    def residual():
        r = np.asarray(res.data.data, np.float32) - np.asarray(res.model.data, np.float32)
        if "mask" in res:
            r[~np.asarray(res.mask.data, bool)] = np.nan
        h = res.model.header.copy()
        h["BTYPE"] = "Residual"
        return fits.PrimaryHDU(data=r, header=h)
    shape = "x".join(str(n) for n in c.array.shape)
    res._set_lazy("data", data, f"HDU {shape}")
    if "model" in res:
        res._set_lazy("residual", residual, f"HDU {shape}")


def _fill(res, step, mem, files, prefix, cube=None):
    """Fill a Result with the outputs of one step (files written + objects in memory). cube: the Cube
    that was analysed (for Result.data and Result.residual)."""
    fitsf = [f for f in files if f.endswith((".fits", ".fits.gz"))]
    base = lambda f: os.path.basename(f)
    find = lambda pred: [f for f in fitsf if pred(base(f))]
    if "mask.fits" in [base(f) for f in files] and step != "SEARCH":
        res._set("mask", load_fits([f for f in files if base(f) == "mask.fits"][0]))

    def folder_maps(sub):
        return {_base(f, prefix): load_fits(f) for f in fitsf if os.path.basename(os.path.dirname(f)) == sub}

    if step in ("3DFIT", "GALMOD", "BAYESIAN3DFIT"):
        txt = {base(f): f for f in files if f.endswith(".txt")}
        if step == "3DFIT":
            last = txt.get("rings_final2.txt") or txt.get("rings_final1.txt")
            res._set("rings", read_table(last) if last else None)
            if last and not len(res.rings):
                import warnings
                warnings.warn("3DFIT: no ring could be fitted (see the warnings of the fit): Result.rings is "
                              "empty.")
            if "rings_final2.txt" in txt and "rings_final1.txt" in txt:
                res._set("rings_stage1", read_table(txt["rings_final1.txt"]))
            if "rings_initial.txt" in txt:
                res._set("rings_initial", read_table(txt["rings_initial.txt"]))
        elif step == "BAYESIAN3DFIT":
            if "rings_final1.txt" in txt:
                res._set("rings", read_table(txt["rings_final1.txt"]))
            if "densprof_deconvolved.txt" in txt:
                res._set("densprof_deconvolved", read_table(txt["densprof_deconvolved.txt"]))
            res._set("bayes", mem.get("bayes"))
            res._set("samples", next((f for f in files if f.endswith("_results.npy")), None))
        else:
            if "rings_model.txt" in txt:
                res._set("rings", read_table(txt["rings_model.txt"]))
        top = res.outfolder.rstrip("/")
        models = {_base(f, prefix).replace("mod_", "", 1): load_fits(f) for f in fitsf
                  if base(f).startswith(prefix + "mod_") and os.path.dirname(f).rstrip("/") == top}
        if models:
            fit = mem.get("galfit")
            if step == "3DFIT":
                want = fit.Norms(2).lower() if fit is not None else None
            else:
                want = str(mem.get("norm", "")).lower() or None
            if want not in models:
                nonoise = [k for k in models if k != "noise"]
                want = nonoise[-1] if nonoise else list(models)[-1]
            res._set("model", models[want])
            if len(models) > 1:
                res._set("models", models)
        if "densprof.txt" in txt:
            res._set("densprof", read_table(txt["densprof.txt"]))
        if "asymdrift.txt" in txt:
            res._set("asymdrift", read_table(txt["asymdrift.txt"]))
        res._set("maps", folder_maps("maps") or None)
        res._set("pvs", folder_maps("pvs") or None)
        res._set("galfit", mem.get("galfit"))
        res._set("diskmodel", mem.get("diskmodel"))
        if "rings" in res and "vrot" in res.rings.colnames and step != "GALMOD":
            res._set("rotation_curve", _rotation_curve(res.rings))
        _add_data(res, cube)
        pdfs = [f for f in files if f.endswith(".pdf")]
        res._set("plots", pdfs or None)
    elif step == "GALWIND":
        for f in fitsf:
            b = _base(f, prefix)
            if b == "wind":
                res._set("model", load_fits(f))
            elif b == "wind_pv":
                res._set("pv", load_fits(f))
        res._set("maps", {_base(f, prefix).replace("wind_", ""): load_fits(f)
                          for f in find(lambda b: "_wind_" in b and "mom" in b)} or None)
        res._set("galwind", mem.get("galwind"))
    elif step == "SEARCH":
        res._set("detections", mem.get("detections"))
        res._set("table", mem.get("table"))
        for f in fitsf:
            b = base(f)
            if b.startswith("detections.fits"):
                res._set("cube", load_fits(f))
            elif b.startswith("detections_num"):
                res._set("labels", load_fits(f))
            elif b.startswith("DetectMap"):
                res._set("detmap", load_fits(f))
        top = res.outfolder.rstrip("/")
        res._set("maps", {_base(f, prefix): load_fits(f) for f in fitsf if base(f).startswith(prefix + "_mom")
                          and os.path.dirname(f).rstrip("/") == top} or None)
        res._set("cubelets", sorted({os.path.dirname(f) for f in fitsf if "/sources/" in f}) or None)
    elif step == "2DFIT":
        for f in files:
            b = base(f)
            if b.endswith("_2dtrm.txt"):
                res._set("rings", read_table(f))
            elif b.endswith("_2d_mod.fits"):
                res._set("model", load_fits(f))
            elif b.endswith("map_1st.fits"):
                res._set("velfield", load_fits(f))
        res._set("ringmodel", mem.get("ringmodel"))
    elif step == "ELLPROF":
        for f in files:
            b = base(f)
            if b.endswith("_densprof.txt"):
                res._set("profile", read_table(f))
            elif b.endswith("_densmap.fits"):
                res._set("map", load_fits(f))
        res._set("ellprof", mem.get("ellprof"))
    elif step == "MAPS":
        names = {"map_0th": "totalmap", "map_1st": "velocitymap", "map_2nd": "dispersionmap",
                 "map_massdens": "massdensmap", "map_RMS": "rmsmap", "map_0th_SN": "snmap",
                 "map_0th_RMS": "totalmap_rms"}
        maps = {}
        for f in fitsf:
            b = _base(f, prefix)
            if b in names:
                maps[names[b]] = load_fits(f)
        res._set("maps", maps or None)
        for k, v in maps.items():
            res._set(k, v)
        sp = [f for f in files if base(f) == "spectrum.txt"]
        if sp:
            from astropy.table import Table
            a = np.loadtxt(sp[0], ndmin=2)
            res._set("spectrum", Table(a, names=("spectral", "flux")))
    elif step in ("SMOOTH", "SMOOTHSPEC", "CONTSUB", "REDUCE"):
        cubes = [f for f in fitsf if base(f) != "mask.fits"]
        if cubes:
            # CONTSUB before another task: the subtracted cube is 'contsub'
            name = "contsub" if step == "CONTSUB" and res.task != "CONTSUB" else "cube"
            res._set(name, load_fits(cubes[-1]))
    elif step == "PVSLICE":
        pv = find(lambda b: b.startswith(prefix + "pv_"))
        if pv:
            res._set("pv", load_fits(pv[-1]))
    elif step == "SPACEPAR":
        sp = find(lambda b: b.startswith("spacepar_"))
        if sp:
            res._set("space", load_fits(sp[-1]))
        res._set("spacepar", mem.get("spacepar"))
        pdfs = [f for f in files if f.endswith(".pdf")]
        res._set("plots", pdfs or None)
    elif step == "REND3D":
        im = find(lambda b: b.endswith("_3D.fits"))
        if im:
            res._set("image", load_fits(im[-1]))
    elif step == "STATS":
        res._set("stats", mem.get("stats"))
        st = [f for f in files if f.endswith("_stats.txt")]
        if st:
            with open(st[0]) as fh:
                res._set("text", fh.read())
    return res


# ==========================================================================
# Parameters
# ==========================================================================
class _FromFile(str):
    """A value read from a parameter file (applied with the parameter-file rules)."""


def read_parfile(fname):
    """The parameters of a parameter file as a dict {KEYWORD: value string}, in order
    (unknown keywords are ignored with a warning, as BBarolo does)."""
    if not os.path.isfile(fname):
        raise FileNotFoundError(f"Could not open parameter file {fname}")
    out = {}
    with open(fname) as f:
        for line in f:
            line = line.rstrip("\n").strip()
            if not line or line[0] in "#/":
                continue
            parts = line.split(None, 1)
            kw = _kw.lookup(parts[0])
            if kw is None:
                from .params import unknown_keyword
                unknown_keyword(parts[0])           # (an error if it is a typo of a keyword)
                continue
            out.pop(kw.name, None)
            out[kw.name] = _FromFile(parts[1] if len(parts) > 1 else "")
    return out


def _canonical(params):
    """{any accepted name: value} -> {KEYWORD: value}, checking names and values."""
    out = {}
    for k, v in params.items():
        kw = _kw.lookup(k)
        if kw is None:
            raise _kw.UnknownParameterError(_kw.unknown_message(k))
        if v is not None and not isinstance(v, _FromFile):
            value_to_string(kw, v)             # type check now, not at run time
        out[kw.name] = v
    return out


def build_param(params, quiet=True):
    """A Param from {KEYWORD: value} (defaults for the others)."""
    par = Param()
    for k, v in params.items():
        if isinstance(v, _FromFile):
            _kw.lookup(k).apply(par, str(v))
        else:
            par.set(k, v)
    return par


def _checked(par):
    """par.checkPars(), raising ValueError with BBarolo's messages if the parameters are
    not acceptable."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ok = par.checkPars()
    msg = buf.getvalue().strip()
    if not ok:
        from .bbutils import BBaroloError
        raise BBaroloError("Unacceptable input parameters:\n" + msg)
    if msg and par.verbose:
        print(msg)
    return par


def _to_str(kw, v):
    return str(v) if isinstance(v, _FromFile) else value_to_string(kw, v)


# ==========================================================================
# Tasks
# ==========================================================================
class Task:
    """Base class of the BBarolo tasks.

    Task(cube, **params)
        cube:   a FITS file name, an astropy HDU (or HDUList), or a tuple (data, header).
        params: any BBarolo parameter (parameter-file keyword, case-insensitive, or an
                alias; see bbarolo.help()). Ring parameters (radii, vrot, inc, ...) can be
                single values or one value per ring.
        beam:   (bmaj, bmin, bpa) in arcsec and degrees: replaces the beam of the cube.

    Methods: run(**params) -> Result; update(**params) -> self; get(name);
    from_parfile(file, **params); to_parfile(file=None); help().
    """
    TASK = None                 # task flag set by the class
    FLAGS = ()                  # task flags accepted (besides CONTSUB)
    DEFAULT_FLAGS = ()          # flags set if none of FLAGS is given

    def __init__(self, cube=None, beam=None, **params):
        self._params = {}
        self._cube = None
        self.beam = None
        self.result = None
        self._set_cube(cube)
        self.update(beam=beam, **params)

    # ------------------------------------------------------------------ parameters
    @classmethod
    def from_parfile(cls, parfile, cube=None, **params):
        """A task with the parameters of a parameter file (FITSFILE is the cube unless cube is
        given). Flags of other tasks in the file are ignored (bbarolo.run runs them all).
        params override the file."""
        fpar = read_parfile(parfile)
        ignored = []
        for kname in list(fpar):
            kw = _kw.lookup(kname)
            if kname in ("FITSFILE", "FITSLIST"):
                v = fpar.pop(kname)
                if kname == "FITSFILE" and cube is None:
                    cube = str(v).strip()
                elif kname == "FITSLIST":
                    ignored.append(kname)
            elif kname == cls.TASK:
                fpar.pop(kname)                      # set by the class anyway
            elif (kw.flag or kname == "AUTO") and not cls._flag_ok(kname):
                if _kw.READERS[kw.kind](fpar[kname]):
                    ignored.append(kname)
                fpar.pop(kname)
        if ignored:
            print(f" {cls.__name__}: {', '.join(ignored)} in {parfile} ignored "
                  "(bbarolo.run runs all the tasks of a parameter file).")
        t = cls(cube, **fpar)          # (all the parameters at once: BayesianBBarolo sets its rings here)
        t.update(**params)
        return t

    @classmethod
    def _flag_ok(cls, name):
        return name == cls.TASK or name in cls.FLAGS or name == "CONTSUB"

    def update(self, beam=None, **params):
        """Set or change parameters (None restores the default). Returns the task."""
        if beam is not None:
            b = list(np.ravel(beam).astype(float))
            if len(b) == 1:
                b = [b[0], b[0], 0.]
            elif len(b) == 2:
                b.append(0.)
            if len(b) != 3:
                raise ValueError("beam must be (bmaj, bmin, bpa) in arcsec and degrees")
            self.beam = tuple(float(x) for x in b)
        new = _canonical(params)
        for k, v in new.items():
            kw = _kw.lookup(k)
            if k in ("FITSFILE", "FITSLIST"):
                raise ValueError(f"{type(self).__name__}: give the cube as the first argument, not {k}.")
            if (kw.flag or k == "AUTO") and not self._flag_ok(k):
                hint = " Did you mean SM (smooth the model)?" if k == "SMOOTH" else ""
                raise ValueError(f"{type(self).__name__} runs the {self.TASK or 'MAPS'} task: {k} is the flag of "
                                 f"another task (use its class, or bbarolo.run for several tasks).{hint}")
            if v is None:
                self._params.pop(k, None)
            else:
                from .params import ring_value_problem, _RING_KEYS, free_problem
                if isinstance(v, str) and not isinstance(v, _FromFile):
                    prob = ring_value_problem(k, v) if k in _RING_KEYS else (
                        free_problem(v) if k == "FREE" and self.TASK == "3DFIT" else None)
                    if prob:
                        raise BBaroloError(f"{type(self).__name__}: {prob}")
                self._params[k] = v
        self._check_rings()
        return self

    def _check_rings(self):
        """Ring parameters given as lists must have the same number of values (= NRADII)."""
        from .params import _RING_KEYS
        lens = {}
        for k, v in self._params.items():
            if k in _RING_KEYS and isinstance(v, (list, tuple, np.ndarray)) and np.size(v) > 1:
                lens[k] = int(np.size(v))
        if len(set(lens.values())) > 1:
            raise ValueError("Ring parameters with different numbers of values: "
                             + ", ".join(f"{k}={n}" for k, n in lens.items()))
        nr = self._params.get("NRADII")
        if lens and nr is not None and not isinstance(nr, _FromFile) and int(nr) != list(lens.values())[0]:
            raise ValueError(f"NRADII={nr} but the ring parameters have {list(lens.values())[0]} values")

    def get(self, name):
        """Value of a parameter (as set, or the default)."""
        kw = _kw.lookup(name)
        if kw is None:
            raise _kw.UnknownParameterError(_kw.unknown_message(name))
        if kw.name in self._params:
            return self._params[kw.name]
        return kw.value(self.par)

    @property
    def parameters(self):
        """The parameters that are set (dict KEYWORD -> value)."""
        return dict(self._params)

    def _all_params(self, outfolder=None, fitsfile=None):
        prm = {}
        if fitsfile is not None:
            prm["FITSFILE"] = fitsfile
        if self.DEFAULT_FLAGS and not any(f in self._params and _truthy(self._params[f]) for f in self.FLAGS):
            for f in self.DEFAULT_FLAGS:
                prm[f] = True
        prm.update(self._params)
        if self.TASK:
            prm.pop(self.TASK, None)
            prm[self.TASK] = True
        if outfolder is not None:
            prm["OUTFOLDER"] = outfolder
        return prm

    @property
    def par(self):
        """The full set of parameters (a bbarolo.params.Param, with the defaults); print it
        to see the parameters used by the task."""
        return build_param(self._all_params(fitsfile=self._cube if isinstance(self._cube, str) else None))

    def to_parfile(self, fname=None):
        """The parameters as a parameter file (a string; also written to fname if given)."""
        lines = [f"// BBarolo parameter file for the {self.TASK or 'MAPS'} task (written by bbarolo {_version()})"]
        if isinstance(self._cube, str):
            lines.append(f"{'FITSFILE':14s}{self._cube}")
        else:
            lines.append("// FITSFILE: an in-memory cube (written in the output folder by run())")
        for k, v in self._all_params().items():
            lines.append(f"{k:14s}{_to_str(_kw.lookup(k), v)}")
        s = "\n".join(lines) + "\n"
        if fname:
            with open(fname, "w") as f:
                f.write(s)
        return s

    def help(self):
        """Print the parameters of this task, with the values set and the defaults."""
        from .keywords import help as _help
        _help(self.TASK or "MAPS")

    def __repr__(self):
        cube = self._cube if isinstance(self._cube, str) else ("in-memory cube" if self._cube is not None else None)
        s = f"{type(self).__name__}({cube!r}"
        for k, v in self._params.items():
            if isinstance(v, (list, tuple, np.ndarray)) and np.size(v) > 4:
                sv = f"<{np.size(v)} values>"
            else:
                sv = repr(str(v) if isinstance(v, _FromFile) else (v.tolist() if isinstance(v, np.ndarray) else v))
                sv = sv if len(sv) < 40 else sv[:36] + "..."
            s += f", {k.lower()}={sv}"
        return s + (f", beam={self.beam}" if self.beam else "") + ")"

    # ------------------------------------------------------------------ the cube
    def _set_cube(self, cube):
        if cube is None:
            self._cube = None
        elif isinstance(cube, (str, os.PathLike)):
            self._cube = os.fspath(cube)
        elif isinstance(cube, fits.HDUList):
            hdu = next((h for h in cube if h.data is not None), None)
            if hdu is None:
                raise ValueError("The HDUList has no data")
            self._cube = fits.PrimaryHDU(data=hdu.data, header=hdu.header)
        elif isinstance(cube, (fits.PrimaryHDU, fits.ImageHDU)):
            self._cube = fits.PrimaryHDU(data=cube.data, header=cube.header)
        elif isinstance(cube, tuple) and len(cube) == 2:
            self._cube = fits.PrimaryHDU(data=np.asarray(cube[0]), header=fits.Header(cube[1]))
        else:
            raise TypeError("cube must be a FITS file name, an astropy HDU or a tuple (data, header)")

    def _input_hdu(self):
        """The cube as an HDU, with the beam if given (None: the user's file, read as it is)."""
        if self._cube is None:
            raise ValueError(f"{type(self).__name__}: no cube given.")
        if isinstance(self._cube, str) and self.beam is None:
            return None
        if isinstance(self._cube, str):
            with fits.open(self._cube, memmap=False) as h:
                hdu = fits.PrimaryHDU(data=np.array(h[0].data), header=h[0].header.copy())
        else:
            hdu = fits.PrimaryHDU(data=self._cube.data, header=self._cube.header.copy())
        if self.beam is not None:
            hdu.header["BMAJ"] = self.beam[0] / 3600.
            hdu.header["BMIN"] = self.beam[1] / 3600.
            hdu.header["BPA"] = self.beam[2]
        return hdu

    def _input_file(self, outfolder):
        """The FITS file to read: the user's file, or a copy (in-memory cube, or new beam)
        written in the output folder."""
        hdu = self._input_hdu()
        if hdu is None:
            return self._cube
        name = str(hdu.header.get("OBJECT", "cube")).strip() or "cube"
        mkdirp(outfolder)
        fn = os.path.join(outfolder, name.replace("/", "_") + "_input.fits")
        hdu.writeto(fn, overwrite=True)
        return fn

    def _default_outfolder(self):
        from .header import Header
        if isinstance(self._cube, str):
            name = Header(self._cube, warning=False).Name()
        else:
            name = str(self._cube.header.get("OBJECT", "NONE")).strip() or "NONE"
        return os.getcwd() + "/output/" + name + "/"

    # ------------------------------------------------------------------ running
    def run(self, write=True, **params):
        """Run the task. params: parameters to change first (as update()). write=False:
        outputs in a temporary folder deleted with the Result (plots off unless PLOTS is
        set). Returns a Result (also in self.result)."""
        if params:
            self.update(**params)
        tmp = None
        if write:
            of = self._params.get("OUTFOLDER")
            of = os.path.expanduser(str(of).strip()) if of not in (None, "") else self._default_outfolder()
        else:
            tmp = tempfile.TemporaryDirectory(prefix="bbarolo_")
            of = tmp.name
        of = os.path.abspath(of) + "/"
        prm = self._all_params(outfolder=of, fitsfile=self._input_file(of))
        if not write and "PLOTS" not in self._params:
            prm["PLOTS"] = False
        par = _checked(build_param(prm))
        r = BBcore(par, collect=True)
        if not r:
            raise IOError(f"{par.imageFile} is not a readable FITS file!")
        outs, c = r
        res = Result(self.TASK or "MAPS", of, tmpdir=tmp)
        if prm["FITSFILE"] != self._cube:            # the copy of an in-memory cube (or with a new beam)
            res.files.append(prm["FITSFILE"])
        prefix = c.pars().outPrefix
        for name, (mem, files) in outs.items():
            res.files = sorted(set(res.files) | set(files))
            _fill(res, name, mem, files, prefix, cube=c)
        self.result = res
        return res


def _truthy(v):
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    return _kw.readflag(str(v))


def _version():
    from ._version import __version__
    return __version__


def _doc(cls, text):
    cls.__doc__ = text + "\n\n" + Task.__doc__
    return cls


class FitMod3D(Task):
    TASK = "3DFIT"


class GalMod(Task):
    TASK = "GALMOD"

    def model(self, seed=None, **params):
        """The model cube, built in memory without writing any file: an astropy PrimaryHDU
        with the data and header of run().model. Only what the model needs is computed: no
        statistics, maps or PV slices, and the mask only for NORM=LOCAL or AZIM. With
        NOISERMS, the model with noise (seed: seed of the noise). params: parameters to
        change first (as update()).

        The model engine is kept in self.diskmodel (a bbarolo.galmod.DiskModel), to build
        other models on the same grid: self.diskmodel.hdu(rings)."""
        from .cube import Cube
        from .galmod import model_hdu
        if params:
            self.update(**params)
        hdu = self._input_hdu()
        with tempfile.TemporaryDirectory(prefix="bbarolo_") as tmp:
            prm = self._all_params(outfolder=tmp + "/", fitsfile=self._cube if hdu is None else "in-memory cube")
            prm["PLOTS"] = -1                       # no mask.fits
            par = _checked(build_param(prm))
            c = Cube(par=par)
            ok = c.readCube(self._cube, printInfo=False) if hdu is None else c.readHDU(hdu, name=prm["FITSFILE"])
            if not ok:
                raise IOError(f"{prm['FITSFILE']} is not a readable FITS file!")
            mod, self.diskmodel, _, _ = _galmod(c, par, write=False, seed=seed)
        return model_hdu(mod, mod.Head().Name())


class GalWind(Task):
    TASK = "GALWIND"


class Search(Task):
    TASK = "SEARCH"


class FitMod2D(Task):
    TASK = "2DFIT"


class Ellprof(Task):
    TASK = "ELLPROF"


class Smooth(Task):
    TASK = "SMOOTH"


class SmoothSpec(Task):
    TASK = "SMOOTHSPEC"


class Maps(Task):
    TASK = None
    FLAGS = ("TOTALMAP", "VELOCITYMAP", "DISPERSIONMAP", "MASSDENSMAP", "RMSMAP", "GLOBALPROFILE")
    DEFAULT_FLAGS = ("TOTALMAP", "VELOCITYMAP", "DISPERSIONMAP")


class PVSlice(Task):
    TASK = "PVSLICE"


class SpacePar(Task):
    TASK = "SPACEPAR"


class Rend3D(Task):
    TASK = "REND3D"


class ContSub(Task):
    TASK = "CONTSUB"


class Stats(Task):
    TASK = "STATS"


class MakeMask(Task):
    TASK = "MAKEMASK"


_doc(FitMod3D, """Fit a 3D tilted-ring model to a datacube (3DFIT).

    Result:
      rings          Table of the final rings: radii (arcsec), rad_kpc, vrot, vdisp, vrad, vsys
                     (km/s), inc, pa (deg), z0 (arcsec), z0_pc, dens, xpos, ypos (pixels); with
                     FLAGERRORS, the errors e_<name>1 (lower, negative) and e_<name>2 (upper) of
                     the free parameters
      rotation_curve radii, rad_kpc, vrot and its errors e_vrot_lo, e_vrot_hi (positive)
      rings_stage1, rings_initial   the rings of the first stage and the initial rings
      model          HDU of the model (NORM of the last stage); models: all the models written
      data, residual HDUs of the data analysed and of data - model (NaN outside the mask),
                     computed when first used
      maps, pvs      dicts of HDUs: moment maps and PV slices of the data and of the model
      densprof, mask, galfit (the fitting engine), plots (PDF files)
    Methods: plot_rings(), plot_maps(), plot_pv() (matplotlib Figures), plot() (the PDF plots
    again). Geometric parameters that are not given are estimated, as in BBarolo.

    >>> f = FitMod3D("ngc2403.fits", radii=np.arange(15, 1200, 30), xpos=77, ypos=77,
    ...              vsys=132.8, vrot=120, vdisp=8, z0=10, inc=60, pa=123.7,
    ...              free="VROT VDISP", mask="SEARCH", distance=3.2)
    >>> res = f.run(threads=4)""")
_doc(GalMod, """Make a 3D model of a disk galaxy on the grid of a datacube (GALMOD).

    The rings (single values or one value per ring): RADII (or NRADII and RADSEP), XPOS,
    YPOS, VSYS, VROT, VDISP, INC, PA and Z0 are needed (or a RINGFILE); DENS, VRAD, VVERT,
    DVDZ and ZCYL are optional. Each ring is an annulus centred on its radius (RINGMODE, see
    bbarolo.help('RINGMODE')). The model is normalized as set by NORM (default AZIM, i.e. to
    the azimuthal profile of the data): use norm='NONE' for the model with the given DENS.

    run() -> Result: model (HDU), rings, maps, pvs, densprof, mask, diskmodel (the model
    engine). model() -> the model only (HDU), in memory: no files, maps or statistics.

    >>> gm = GalMod("ngc2403.fits", radii=np.arange(15, 1200, 30), xpos=77, ypos=77, vsys=132.8,
    ...             vrot=120, vdisp=8, z0=10, inc=60, pa=123.7, dens=1, norm="NONE")
    >>> hdu = gm.model()""")
_doc(GalWind, """Make a 3D model of a bi-conical outflow (GALWIND).

    Result: model (HDU), maps (0mom, 1mom, 2mom), pv (HDU).""")
_doc(Search, """Search for sources in a datacube (SEARCH).

    Result: detections (list of the detected objects), table (Table of the detections),
    cube (data of the detections), labels (cube with the number of each detection),
    detmap (number of detected channels per pixel), maps (moment maps).""")
_doc(FitMod2D, """Fit a 2D tilted-ring model to a velocity field (2DFIT).

    Result: rings (Table), model (HDU of the model velocity field), velfield (HDU of the
    velocity field that was fitted), mask.""")
_doc(Ellprof, """Radial profiles in elliptical rings (ELLPROF).

    Result: profile (Table), map (HDU of the map used), mask.""")
_doc(Smooth, """Smooth a datacube spatially to a new beam (SMOOTH): BMAJ, BMIN, BPA (arcsec,
    degrees), or FACTOR (times the old beam), or LINEAR (kpc) with DISTANCE.

    Result: cube (HDU of the smoothed cube).""")
_doc(SmoothSpec, """Smooth a datacube spectrally (SMOOTHSPEC): WINDOW_TYPE, WINDOW_SIZE.

    Result: cube (HDU of the smoothed cube).""")
_doc(Maps, """Maps from a datacube: TOTALMAP, VELOCITYMAP, DISPERSIONMAP (the default if none is
    given), MASSDENSMAP, RMSMAP, GLOBALPROFILE (flags), with MASK, MAPTYPE, SNMAP, ...

    Result: maps (dict of HDUs: totalmap, velocitymap, dispersionmap, massdensmap, rmsmap,
    snmap), each also as an attribute, spectrum (Table, for GLOBALPROFILE).""")
_doc(PVSlice, """Extract a position-velocity slice (PVSLICE): XPOS_PV, YPOS_PV, PA_PV or P1_PV,
    P2_PV, WIDTH_PV. See also bbarolo.pv_extract for slices in memory.

    Result: pv (HDU).""")
_doc(SpacePar, """Explore the residuals of a 3D model over a grid of two parameters (SPACEPAR):
    P1, P1PAR, P2, P2PAR, plus the ring parameters.

    Result: space (HDU of the residuals), plots.""")
_doc(Rend3D, """3D rendering of a datacube (REND3D).

    Result: image (HDU).""")
_doc(ContSub, """Subtract the continuum from a datacube (CONTSUB): EXCLWIND, CONTORDER.

    CONTSUB=True can also be given to any other task, to subtract the continuum first.
    Result: cube (HDU).""")
_doc(Stats, """Statistics of a datacube (STATS).

    Result: stats (the statistics object), text (the statistics as written).""")
_doc(MakeMask, """Build the mask of a datacube (MAKEMASK) as set by MASK.

    Result: mask (HDU).""")


# ==========================================================================
# run(): a parameter file, or parameters, with all the tasks
# ==========================================================================
def run(parfile=None, exe=None, stdout=None, **params):
    """Run BBarolo on a parameter file and/or parameters, with all the tasks requested,
    exactly as the command line ``BBarolo -p parfile KEY=value ...`` (in this Python
    process).

        >>> bbarolo.run("ngc2403.par", threads=8)
        >>> bbarolo.run(fitsfile="ngc2403.fits", search=True, totalmap=True)

    params override the file. Returns a dict {task: Result} (a list of them if FITSLIST
    gives several cubes).

    exe: path of another BBarolo executable (e.g. the C++ one) to run instead, in a
    subprocess; stdout: None (screen), a file name or 'null'. Returns its exit code.
    """
    if exe is not None:
        return _run_external(exe, parfile, stdout, params)
    par = Param()
    if parfile is not None and not par.readParamFile(parfile):
        raise FileNotFoundError(f"Could not open parameter file {parfile}")
    par.update(**params)
    par = _checked(par)
    with _stdout_to(stdout):
        ok, outs = run_par(par, banner=True, collect=True)
    results = []
    for r in outs:
        out, c = r
        p = c.pars()
        d = {}
        for name, (mem, files) in out.items():
            d[name] = _fill(Result(name, p.outFolder, files), name, mem, files, p.outPrefix, cube=c)
        results.append(d)
    if not ok and not results:
        raise RuntimeError("BBarolo could not run (see the messages above).")
    return results[0] if len(results) == 1 else results


@contextlib.contextmanager
def _stdout_to(stdout):
    if stdout is None:
        yield
        return
    target = os.devnull if str(stdout).lower() == "null" else stdout
    with open(target, "w") as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
        yield


def _run_external(exe, parfile, stdout, params):
    import subprocess
    cmd = list(exe) if isinstance(exe, (list, tuple)) else [str(exe)]
    if isinstance(exe, (str, os.PathLike)) and not os.path.exists(exe) and shutil.which(str(exe)) is None:
        raise FileNotFoundError(f"{exe} does not exist.")
    lines = [f"{k}  {_to_str(_kw.lookup(k), v)}" for k, v in _canonical(params).items()]
    tmp = None
    if parfile and lines:
        # parameters after the file override it (also for executables without overrides)
        tmp = tempfile.NamedTemporaryFile("w", suffix=".par", delete=False)
        with open(parfile) as f:
            tmp.write(f.read().rstrip("\n") + "\n")
        tmp.write("\n".join(lines) + "\n")
        tmp.close()
        cmd += ["-p", tmp.name]
    elif parfile:
        cmd += ["-p", parfile]
    else:
        cmd += ["-c"] + [ln.replace("  ", "=", 1) for ln in lines]
    out = None
    if stdout is not None:
        out = subprocess.DEVNULL if str(stdout).lower() == "null" else open(stdout, "w")
    try:
        return subprocess.call(cmd, stdout=out, stderr=out)
    finally:
        if hasattr(out, "close"):
            out.close()
        if tmp is not None:
            os.unlink(tmp.name)
