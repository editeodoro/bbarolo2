"""
3D tilted-ring fitting (port of Tasks/galfit.cpp, galfit_min.cpp, galfit_errors.cpp
and galfit_out.cpp).

The rings are fitted one at a time (in parallel threads) with the same downhill
simplex, parameter boundaries, normalisation and residual functions as BBarolo.
The models are built and convolved by the model engine (galmod.DiskModel, in
Galfit.model); the residuals are compiled with numba; the ring loop runs in a thread
pool (the numba kernels and the FFTs release the GIL). The initial rings
(initial_rings) and the outputs (galfit_out) are shared with the GALMOD task.
"""

import sys
import io
import copy
import math
import time
import types
import threading
import numpy as np
import numba as nb
from concurrent.futures import ThreadPoolExecutor

from .rings import Rings, readRings, ring_file_columns
from .galmod import DiskModel
from .stats import findMedian, findMADFM, madfmToSigma
from .bbutils import (BBaroloError, arcsconv, AlltoVel, DeltaVel, bezier_interp, polyfit_lsq,
                      ProgressBar, mkdirp)

VROT, VDISP, DENS, Z0, INC, PA, XPOS, YPOS, VSYS, VRAD, MAXPAR = range(11)
PARNAMES = ["VROT", "VDISP", "DENS", "Z0", "INC", "PA", "XPOS", "YPOS", "VSYS", "VRAD"]
NORMCODE = {"LOCAL": 0, "AZIM": 1, "NONE": 2}
SIDECODE = {"R": 1, "A": 2}


# --------------------------------------------------------------------------
# Compiled residual functions (norm_local, norm_azim, norm_none)
# --------------------------------------------------------------------------
@nb.njit(nogil=True, cache=True)
def _norm(model, data, mask, blo0, blo1, x0, y0, phi, inc, r1, r2, wpow, side, ftype,
          chan_noise, data_noise, bweight, normtype, pre, ilo, ihi, thr, P):
    """BBarolo residual function. If pre (the non-convolved model, (ny,nx,nz)) is not
    empty, a model voxel outside the mask is counted as a blank only if the sum of pre
    over the footprint of the convolution kernel (rows j: columns x+ilo[j]..x+ihi[j]) is
    >= thr[z] (ANALYTIC models: see Galfit._support_args). P is a work array of shape
    (ky, nx+1, nz) for the cumulative sums of pre along x of the last ky rows."""
    bsy, bsx, nz = model.shape
    trunc = pre.shape[0] > 0
    ky = ilo.shape[0]
    hy = ky // 2
    lam = np.zeros(nz)
    nextrow = 0
    F = math.pi / 180.
    sphi, cphi, cinc = math.sin(F * phi), math.cos(F * phi), math.cos(F * inc)
    numBlanks = 0
    numPix_tot = 0
    minfunc = 0.0
    usesin = wpow < 0
    apow = abs(wpow)
    # inside-ring map and weights
    inside = np.zeros((bsy, bsx), np.bool_)
    wmap = np.zeros((bsy, bsx))
    for y in range(bsy):
        for x in range(bsx):
            xr = -(x - x0) * sphi + (y - y0) * cphi
            yr = (-(x - x0) * cphi - (y - y0) * sphi) / cinc
            r = math.sqrt(xr * xr + yr * yr)
            if r < 0.1:
                th = 0.0
            else:
                th = math.atan2(yr, xr) / F
            if not (r >= r1 and r <= r2):
                continue
            if side == 1 and not (abs(th) <= 90.0):
                continue
            if side == 2 and not (abs(th) >= 90.0):
                continue
            inside[y, x] = True
            if usesin:
                wf = abs(math.sin(th * F))
            else:
                wf = abs(math.cos(th * F))
            wmap[y, x] = wf ** apow
    factor = 1.0
    if normtype == 1:
        modSum = 0.0
        obsSum = 0.0
        for y in range(bsy):
            for x in range(bsx):
                if not inside[y, x]:
                    continue
                for z in range(nz):
                    modSum += model[y, x, z]
                    if mask[y + blo1, x + blo0, z]:
                        obsSum += data[y + blo1, x + blo0, z]
        factor = obsSum / modSum if modSum != 0 else 0.0
    for y in range(bsy):
        if trunc:
            # cumulative sums along x of the rows y-hy..y+hy of pre (rolling buffer)
            anyin = False
            for x in range(bsx):
                if inside[y, x]:
                    anyin = True
                    break
            if anyin:
                if nextrow < y - hy:
                    nextrow = y - hy
                while nextrow <= min(y + hy, bsy - 1):
                    Pr = P[nextrow % ky]
                    pr = pre[nextrow]
                    Pr[0, :] = 0.
                    for x in range(bsx):
                        for z in range(nz):
                            Pr[x + 1, z] = Pr[x, z] + pr[x, z]
                    nextrow += 1
        for x in range(bsx):
            if not inside[y, x]:
                continue
            if normtype == 0:
                modSum = 0.0
                obsSum = 0.0
                for z in range(nz):
                    modSum += model[y, x, z]
                    o = data[y + blo1, x + blo0, z]
                    if o > 0 and mask[y + blo1, x + blo0, z]:
                        obsSum += o
                factor = obsSum / modSum if modSum != 0 else 0.0
            wi = wmap[y, x]
            if trunc:
                # expected number of MC sub-clouds (x thr/MC_SUPPORT_NMIN) for the voxels
                # that may count as blanks
                ca = nz
                cb = -1
                for z in range(nz):
                    if model[y, x, z] != 0 and not mask[y + blo1, x + blo0, z]:
                        if ca == nz:
                            ca = z
                        cb = z
                for z in range(ca, cb + 1):
                    lam[z] = 0.
                if cb >= ca:
                    for j in range(ky):
                        yy = y + hy - j
                        if yy < 0 or yy >= bsy:
                            continue
                        xa = max(x + ilo[j], 0)
                        xb = min(x + ihi[j] + 1, bsx)
                        if xb <= xa:
                            continue
                        p1 = P[yy % ky, xb]
                        p0 = P[yy % ky, xa]
                        for z in range(ca, cb + 1):
                            lam[z] += p1[z] - p0[z]
            for z in range(nz):
                if normtype != 2:
                    model[y, x, z] *= factor
                o = data[y + blo1, x + blo0, z]
                obs = o if o > 0 else data_noise
                mod = model[y, x, z]
                if not mask[y + blo1, x + blo0, z]:
                    if mod == 0:
                        continue
                    if trunc and lam[z] < thr[z]:
                        continue
                    numBlanks += 1
                numPix_tot += 1
                nw = chan_noise[z]
                if ftype == 1:
                    minfunc += wi * (mod - obs) ** 2 / math.sqrt(obs) / nw
                elif ftype == 2:
                    minfunc += wi * abs(mod - obs) / nw
                elif ftype == 3:
                    minfunc += wi * abs(mod - obs) / (mod + obs) / nw
                elif ftype == 4:
                    minfunc += wi * (mod - obs) ** 2 / nw
    if numPix_tot - numBlanks == 0:
        return np.inf
    return (1 + numBlanks / numPix_tot) ** bweight * minfunc / (numPix_tot - numBlanks)


# Expected number of Monte-Carlo sub-clouds below which a voxel of an ANALYTIC model is not
# counted as a blank in the residuals: exp(-Euler gamma) makes the number of blanks equal
# to the expected one of MC models, sum(1 - exp(-N)), in the line wings.
MC_SUPPORT_NMIN = math.exp(-0.5772156649015329)
_NO_SUPPORT = (np.zeros((0, 0, 0)), np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0),
               np.zeros((0, 0, 0)))


@nb.njit(nogil=True, cache=True)
def _count_ring_pixels(mask2D, blo0, blo1, bhi0, bhi1, x0, y0, phi, inc, r1, r2):
    F = math.pi / 180.
    sphi, cphi, cinc = math.sin(F * phi), math.cos(F * phi), math.cos(F * inc)
    nTot = 0
    nIn = 0
    for y in range(blo1, bhi1 + 1):
        for x in range(blo0, bhi0 + 1):
            xx = x - blo0
            yy = y - blo1
            xr = -(xx - x0) * sphi + (yy - y0) * cphi
            yr = (-(xx - x0) * cphi - (yy - y0) * sphi) / cinc
            r = math.sqrt(xr * xr + yr * yr)
            if r >= r1 and r <= r2:
                nTot += 1
                if y < mask2D.shape[0] and x < mask2D.shape[1] and mask2D[y, x] > 0:
                    nIn += 1
    return nTot, nIn


def _fmt(v, prec=3):
    return f"{v:.{prec}f}"


# --------------------------------------------------------------------------
# Initial rings and normalisation (shared by 3DFIT and GALMOD)
# --------------------------------------------------------------------------
NORMS = ("LOCAL", "AZIM", "NONE")


def final_norm(norm, task="GALMOD", extra=()):
    """The normalisation of the final model for the NORM parameter: the last of the (one or
    two) values given; LOCAL if not recognised. extra: other accepted values."""
    v = ["AZIM" if n == "AZIMUTHAL" else n for n in str(norm).upper().split()]
    n = v[1] if len(v) > 1 else (v[0] if v else "")
    if n not in NORMS + tuple(extra):
        sys.stderr.write(f"{task} WARNING: Normalization function not recognized. Using LOCAL.\n")
        n = "LOCAL"
    return n


def initialRingsText(inr):
    """The initial rings as written in rings_initial.txt (3DFIT) and rings_model.txt (GALMOD)."""
    m = 11
    s = "#" + "=" * 46 + " Initial parameters " + "=" * 45 + " \n"
    s += "".join(f"{s:<{m}}" for s in ("#RAD(arcs)", "VROT(km/s)", "DISP(km/s)", "INC(deg)", "P.A.(deg)",
                                       "Z0(arcs)", "SIG(E20)", "XPOS(pix)", "YPOS(pix)", "VSYS(km/s)")) + "\n"
    for i in range(inr.nr):
        s += "".join(f"{v:<{m}.3f}" for v in (inr.radii[i], inr.vrot[i], inr.vdisp[i], inr.inc[i], inr.phi[i],
                                              inr.z0[i], inr.dens[i], inr.xpos[i], inr.ypos[i], inr.vsys[i])) + "\n"
    return s


def ring_table(r, mpar=None, errors=None, fitok=None, extra=None):
    """The rings r (a Rings object) as an astropy Table at full precision, with the columns of the ring files
    (rings_final1.txt, ...): radii (arcsec), vrot, vdisp (km/s), inc, pa (deg), z0 (arcsec), dens, xpos, ypos
    (pixels), vsys, vrad (km/s); with errors (array nr x 2 x number of free parameters, the free parameters
    mpar), the lower (negative) and upper errors e_<name>1, e_<name>2 of the free parameters; with fitok, a
    column fitok (False for the rings whose fit failed); extra: {name: values} as more columns."""
    from astropy.table import Table
    import astropy.units as u
    kms, arcs = u.km / u.s, u.arcsec
    n = r.nr
    cols = [("radii", r.radii, arcs), ("vrot", r.vrot, kms), ("vdisp", r.vdisp, kms), ("inc", r.inc, u.deg),
            ("pa", r.phi, u.deg), ("z0", r.z0, arcs), ("dens", r.dens, None), ("xpos", r.xpos, u.pix),
            ("ypos", r.ypos, u.pix), ("vsys", r.vsys, kms), ("vrad", r.vrad, kms)]
    t = Table()
    for name, v, unit in cols:
        t[name] = np.asarray(v, float)[:n]
        if unit is not None:
            t[name].unit = unit
    if errors is not None and mpar is not None:
        names = [("vrot", kms), ("vdisp", kms), ("dens", None), ("z0", arcs), ("inc", u.deg), ("pa", u.deg),
                 ("xpos", u.pix), ("ypos", u.pix), ("vsys", kms), ("vrad", kms)]
        kk = 0
        for i, (name, unit) in enumerate(names):
            if i < len(mpar) and mpar[i]:
                for j in (0, 1):
                    t[f"e_{name}{j + 1}"] = np.asarray(errors, float)[:n, j, kk]
                    if unit is not None:
                        t[f"e_{name}{j + 1}"].unit = unit
                kk += 1
    if fitok is not None:
        t["fitok"] = np.asarray(fitok, bool)[:n]
    for name, v in (extra or {}).items():
        t[name] = np.asarray(v)[:n]
    return t


def ringfile_values(par):
    """RINGFILE: the ring parameters that are not given (par, a GalfitPar, changed in place)
    are read from the columns of a BBarolo ring file (e.g. rings_final1.txt), found by their names in
    the header (files of bbarolo2 and of C++ BBarolo, whose columns differ); a file without a header
    must have the columns of the ring files of C++ BBarolo."""
    if not par.ringfile:
        return
    rf = par.ringfile
    cols, _ = ring_file_columns(rf)
    if cols is None:
        cols = dict(RADII=2, VROT=3, VDISP=4, INC=5, PHI=6, Z0=8, DENS=9, XPOS=10, YPOS=11, VSYS=12, VRAD=13)
    if par.RADII == "-1" and (par.NRADII == -1 or par.RADSEP == -1) and "RADII" in cols:
        par.RADII = f"file({rf},{cols['RADII']})"
    for k in ("VROT", "VDISP", "INC", "PHI", "Z0", "DENS", "XPOS", "YPOS", "VSYS", "VRAD"):
        if getattr(par, k) == "-1" and k in cols:
            setattr(par, k, f"file({rf},{cols[k]})")


def check_vsys(c, rings, fatal=True):
    """VSYS of the rings must be in the velocity range of the cube: a BBaroloError if fatal (3DFIT),
    otherwise a warning (GALMOD). A VSYS given in m/s instead of km/s is the usual cause."""
    h = c.Head()
    if c.DimZ() < 2:
        return
    v = [AlltoVel(c.getZphys(0), h), AlltoVel(c.getZphys(c.DimZ() - 1), h)]
    lo, hi = min(v), max(v)
    out = [x for x in np.atleast_1d(rings.vsys) if not lo <= x <= hi]
    if not out:
        return
    msg = (f"VSYS={out[0]:g} km/s is outside the velocity range of the cube ({lo:.1f} to {hi:.1f} km/s)"
           + (": is it in m/s?" if lo <= out[0] / 1000. <= hi else "."))
    if fatal:
        raise BBaroloError(msg)
    print(f" WARNING: {msg}")


def default_z0(h):
    """The default scale height of the fits (arcsec), when Z0 is not given: one pixel of the header h, a
    thin disk. (C++ uses 0, a disk with no thickness.)"""
    return float(h.PixScale() * arcsconv(h.Cunit(0)))


def initial_rings(c, par, galmod=False, write=True, buildmask=False, task="3DFIT"):
    """The initial rings of 3DFIT and BAYESIAN3DFIT, or the rings of GALMOD (galmod=True), from the
    parameters par (a GalfitPar; changed in place: the values that are not given are set).

    Values not given are read from RINGFILE, estimated from the data (centre, systemic and
    rotation velocity, inclination, position angle, number and width of the rings), or set
    to the defaults (VDISP=8, or its estimate with GUESSTYPE FIT; Z0=one pixel, VRAD=0, DENS=1,
    VVERT=0, DVDZ=0, ZCYL=0). The estimates use the mask of the cube: 3DFIT builds it before;
    buildmask=True builds it here (MASK; BAYESIAN3DFIT, only when something is estimated), otherwise
    only a mask from a file is built (GALMOD). With write=True the rings are kept in the cube for
    rings_model.txt (GALMOD) or rings_initial.txt (3DFIT), see Cube.add_output. task: the name in
    the error messages; with BAYESIAN3DFIT, the estimated rings are one beam wide (EstimateInitial
    beamrings), not half a beam for fewer than 20 rings.

    Returns a namespace with: rings (a Rings object), fromfile (the rings were read from a
    file), estimated (dict: which parameters were not given), toEstimate (bool) and ip (the
    ParamGuess object of the estimates, or None)."""
    from .paramguess import EstimateInitial
    p = c.pars()
    ringfile_values(par)
    toEstimate = ((par.RADII == "-1" and (par.NRADII == -1 or par.RADSEP == -1)) or par.XPOS == "-1"
                  or par.YPOS == "-1" or par.VSYS == "-1" or par.VROT == "-1" or par.PHI == "-1" or par.INC == "-1")
    estimated = estimatedParameters(par)
    ip = None
    if toEstimate:
        if buildmask:                                       # the mask of the fit, as 3DFIT
            c.checkBeam()
            if not c.MaskAll():
                c.BlankMask()
        elif "FILE(" in p.MaskType and not c.MaskAll():    # the estimates use a mask from a file
            c.BlankMask()
        ip = EstimateInitial(c, par, beamrings=task == "BAYESIAN3DFIT")
        if par.RADII == "-1":
            if par.NRADII == -1: par.NRADII = ip.nrings
            if par.RADSEP == -1: par.RADSEP = ip.radsep
        if par.XPOS == "-1": par.XPOS = repr(float(ip.xcentre))
        if par.YPOS == "-1": par.YPOS = repr(float(ip.ycentre))
        if par.VSYS == "-1": par.VSYS = repr(float(ip.vsystem))
        if par.VROT == "-1": par.VROT = repr(float(ip.vrot))
        if par.INC == "-1": par.INC = repr(float(ip.inclin))
        if par.PHI == "-1": par.PHI = repr(float(ip.posang))
    if par.VDISP == "-1":
        # (GUESSTYPE FIT: from the second moment; otherwise 8 km/s, as in C++)
        par.VDISP = repr(float(ip.vdisp)) if ip is not None and getattr(ip, "vdisp", None) else "8.0"
    if par.Z0 == "-1": par.Z0 = repr(default_z0(c.Head()))
    if par.VRAD == "-1": par.VRAD = "0.0"
    if par.DENS == "-1": par.DENS = "1.0"
    if par.VVERT == "-1": par.VVERT = "0.0"
    if par.DVDZ == "-1": par.DVDZ = "0.0"
    if par.ZCYL == "-1": par.ZCYL = "0.0"

    inR, fromfile = readRings(par, c.Head(), return_fromfile=True)
    if inR.nr == 0:
        raise BBaroloError(f"\n {task} ERROR: The number of radii must be > 0! ")
    if write:
        c.add_output("_rings_model.txt" if galmod else "_rings_initial.txt", initialRingsText(inR))
    return types.SimpleNamespace(rings=inR, fromfile=fromfile, estimated=estimated, toEstimate=toEstimate, ip=ip)


def estimatedParameters(par):
    """Which ring parameters of par (a GalfitPar, before the values not given are filled in) are not given:
    {NRADII, RADSEP, XPOS, ...: True if estimated or default}."""
    noradii = par.RADII == "-1"
    return dict(NRADII=noradii and par.NRADII == -1, RADSEP=noradii and par.RADSEP == -1, XPOS=par.XPOS == "-1",
                YPOS=par.YPOS == "-1", VSYS=par.VSYS == "-1", VROT=par.VROT == "-1", INC=par.INC == "-1",
                PHI=par.PHI == "-1", Z0=par.Z0 == "-1", VDISP=par.VDISP == "-1", VRAD=par.VRAD == "-1")


def initialParametersText(inr, est):
    """The table of the initial parameters of the rings inr shown by 3DFIT and BAYESIAN3DFIT: each value marked
    (i) given, (e) estimated or (d) default, as in est (see estimatedParameters)."""
    def tag(k, d="(e)"): return d if est.get(k, False) else "(i)"
    def tagd(k): return "(d)" if est.get(k, False) else "(i)"
    m, n = 9, 12
    s = io.StringIO()
    s.write("\n" + "=" * 24 + " Initial parameters " + "=" * 25 + "\n")
    s.write("    (i) input by the user    (d) default    (e) estimated by me\n\n")
    s.write(f"   Fitting #{inr.nr}{tag('NRADII')} rings of width {inr.radsep:.2f}{tag('RADSEP')} arcsec\n\n")
    s.write(f"{'    Xpos ' + tag('XPOS'):<{n}}{'= ':>3}{inr.xpos[0]:>{m}.2f}{'  pix':<{m}}")
    s.write(f"{'        Ypos ' + tag('YPOS'):<{n + 4}}{'= ':>3}{inr.ypos[0]:>{m - 1}.2f}{'  pix':<{m}}\n")
    s.write(f"{'    Vsys ' + tag('VSYS'):<{n}}{'= ':>3}{inr.vsys[0]:>{m}.2f}{'  km/s':<{m}}")
    s.write(f"{'        Vrot ' + tag('VROT'):<{n + 4}}{'= ':>3}{inr.vrot[0]:>{m - 1}.2f}{'  km/s':<{m}}\n")
    s.write(f"{'    Inc ' + tag('INC'):<{n}}{'= ':>3}{inr.inc[0]:>{m}.2f}{'  deg':<{m}}")
    s.write(f"{'        PA ' + tag('PHI'):<{n + 4}}{'= ':>3}{inr.phi[0]:>{m - 1}.2f}{'  deg':<{m}}\n")
    s.write(f"{'    Z0 ' + tagd('Z0'):<{n}}{'= ':>3}{inr.z0[0]:>{m}.2f}{'  arcs':<{m}}")
    s.write(f"{'        Disp ' + tagd('VDISP'):<{n + 4}}{'= ':>3}{inr.vdisp[0]:>{m - 1}.2f}{'  km/s':<{m}}\n")
    s.write(f"{'    Vrad ' + tagd('VRAD'):<{n}}{'= ':>3}{inr.vrad[0]:>{m}.2f}{'  km/s':<{m}}\n\n\n")
    return s.getvalue()


class Galfit:
    """3D tilted-ring fit of a datacube (3DFIT task)."""

    refine2D = True         # with PA free, the initial PA and VROT of the rings from a 2D fit (not SPACEPAR)

    # ------------------------------------------------------------------
    def __init__(self, cube=None, inrings=None, par=None, gfpar=None):
        self._lock = threading.Lock()
        self.reverse = False
        self.second = False
        self.n_galfit = 0
        if cube is None:
            return
        if inrings is not None:
            # constructor with Cube + Rings + Param (used by Bayesian3DFIT)
            p = par if par is not None else cube.pars()
            if p.outPrefix == "":
                p.outPrefix = cube.Head().Name()
            cube.par = p
            p.checkPars()
            cube.Head().setRedshift(p.getRedshift())
            cube.Head().setWave0(p.getRestwave())
            self.setup(cube, inrings, gfpar if gfpar is not None else p.gf)
        else:
            self._init_from_cube(cube)

    def _init_from_cube(self, c):
        self.cin = c
        # a copy, as in C++: initial_rings fills in the parameters that are not given, and the
        # tasks that come next (2DFIT, SPACEPAR, ...) must see the parameters as given
        par = copy.deepcopy(c.pars().gf)
        p = c.pars()
        self.verb = p.verbose
        c.checkBeam()
        if not c.MaskAll():
            c.BlankMask()
        isgalmod = p.gm.flagGALMOD
        ini = initial_rings(c, par, galmod=isgalmod)
        inR, toEstimate, ip = ini.rings, ini.toEstimate, ini.ip
        self.rings_initial = inR.copy()          # (the rings of rings_initial.txt)
        self.stages = {}                         # (the outcome of each stage, see galfit())
        self._estimated = ini.estimated
        if not isgalmod and not ini.fromfile and self.verb:
            self.showInitial(inR, sys.stdout, par)

        if "auto" in par.REVERSE and not isgalmod:
            if findMedian(inR.inc) > 75:
                if self.verb:
                    print("\n 3DFIT WARNING: because the galaxy is highly inclined, I recommend to try \n"
                          " a reverse-cumulative fitting scheme (REVERSE=true) with ring width=beam size.")
                self.reverse = True
                print()
        elif "true" in par.REVERSE:
            self.reverse = True
        else:
            self.reverse = False
        if self.reverse:
            beamarcs = c.Head().Bmaj() * arcsconv(c.Head().Cunit(0))
            if inR.radsep < beamarcs:
                if self.verb:
                    print(" To this end, I will set ring widths >= beam major axis.")
                new_nr = int(math.ceil(inR.radii[-1] / beamarcs))
                new_r = np.arange(new_nr) * beamarcs + beamarcs / 2.

                def at(v):          # the initial values at the new radii (C++ took those of ring i)
                    return np.interp(new_r, inR.radii, np.asarray(v, float))
                phi = np.degrees(np.unwrap(np.radians(np.asarray(inR.phi, float))))
                nr = Rings()
                nr.radsep = beamarcs
                cols = [at(v) for v in (inR.xpos, inR.ypos, inR.vsys, inR.vrot, inR.vdisp, inR.vrad, inR.vvert,
                                        inR.dvdz, inR.zcyl, inR.dens, inR.z0, inR.inc, phi)]
                for i in range(new_nr):
                    nr.addRing(new_r[i], *(float(col[i]) for col in cols))
                inR = nr

        # A little trick: fit a 2D model first to get better PA variation (not for SPACEPAR, which keeps
        # the rings given)
        free = par.FREE.lower()
        if self.refine2D and "pa" in free and inR.nr >= 5 and not isgalmod:
            from .ringmodel import Ringmodel
            topix = c.Head().PixScale() * arcsconv(c.Head().Cunit(0))
            radii = inR.radii / topix
            wids = np.full(inR.nr, inR.radsep / topix)
            tr = Ringmodel()
            if toEstimate:
                tr.set(inR.nr, radii, wids, inR.vsys, inR.vrot, inR.vrad, inR.phi, inR.inc, inR.xpos[0], inR.ypos[0])
                tr.setfield(ip.Vemap, c.DimX(), c.DimY())
            else:
                tr.setfromCube(c, inR)
            tr.setoption([False, True, False, True, False, False, False], 3, 2, 15.)
            tr.ringfit(p.threads, False, False)
            buf = io.StringIO()
            tr.printfinal(buf, c.Head())
            c.add_output("_2drings.txt", buf.getvalue())
            # NB: the C++ code loops up to inR->nr, reading beyond the end of the
            # 2D-fit arrays (which have nr-1 elements) for the last ring.
            for i in range(1, min(inR.nr, tr.nrad)):
                pa_i, vr_i = tr.getPosaf(i), tr.getVrotf(i)
                if not np.isnan(pa_i) and vr_i > 0:
                    inR.phi[i] = pa_i
                if not np.isnan(vr_i) and vr_i > 0 and "vrot" in free:      # (a fixed VROT is kept)
                    inR.vrot[i] = vr_i
            inR.phi[0] = inR.phi[1]
        self.setup(c, inR, par)

    # ------------------------------------------------------------------
    def setup(self, c, inrings, gfpar):
        self.cin = c
        self.par = copy.deepcopy(gfpar)
        par = self.par
        self.inr = inrings.copy()
        self.verb = c.pars().verbose
        for ir in range(self.inr.nr):
            if ir and self.inr.radii[ir] <= self.inr.radii[ir - 1]:
                raise BBaroloError("3DFIT ERROR: Radii not in increasing order.")
            if self.inr.radii[ir] < 0:
                raise BBaroloError("3DFIT ERROR: Negative radius!!!")
        c.checkBeam()
        h = c.Head()
        self.arcconv = arcsconv(h.Cunit(0))
        check_vsys(c, self.inr, fatal=True)
        self.chan_noise = np.ones(c.DimZ(), np.float64)
        if not c.StatsDef():
            c.setCubeStats()
        self.data_noise = c.stat().getSpread()
        self.wpow = par.WFUNC
        self.setFree()
        if not c.MaskAll() or c.pars().MaskType == "NEGATIVE":
            cn = np.ones(c.DimZ(), np.float32)
            c.BlankMask(cn)
            self.chan_noise = cn.astype(np.float64)
        self.mask = c.Mask()
        self.mask_T = np.ascontiguousarray(self.mask.transpose(1, 2, 0))
        self.mask2D = self.mask.sum(axis=0).astype(np.int32)
        norms = ["AZIM" if n == "AZIMUTHAL" else n for n in par.NORM.split()]
        if len(norms) == 1:
            norms.append(norms[0])
        for i in range(2):
            if norms[i] not in ("NONE", "AZIM", "LOCAL"):
                sys.stderr.write(f"3DFIT WARNING: Normalization function not recognized for step {i + 1}. Using LOCAL.\n")
                norms[i] = "LOCAL"
        if len(norms) == 2:
            norms.append(norms[1])
        self.norms = norms
        par.DENSFLUX = True
        inr = self.inr
        self.maxs = np.zeros(MAXPAR)
        self.mins = np.zeros(MAXPAR)
        self.maxs[VROT] = inr.vrot.max() + par.DELTAVROT
        self.mins[VROT] = inr.vrot.min() - par.DELTAVROT
        self.maxs[VDISP], self.mins[VDISP] = par.MAXVDISP, par.MINVDISP
        self.maxs[Z0], self.mins[Z0] = 1000, 0.
        self.maxs[INC] = inr.inc.max() + par.DELTAINC
        self.mins[INC] = inr.inc.min() - par.DELTAINC
        self.maxs[PA] = inr.phi.max() + par.DELTAPHI
        self.mins[PA] = inr.phi.min() - par.DELTAPHI
        self.mins[VROT] = max(self.mins[VROT], 0)
        self.maxs[INC] = min(self.maxs[INC], 90)
        self.mins[INC] = max(self.mins[INC], 0)
        self.maxs[PA] = min(self.maxs[PA], 360)
        self.mins[PA] = max(self.mins[PA], -360)
        self.maxs[XPOS] = min(inr.xpos.max() + 10, c.DimX())
        self.mins[XPOS] = max(inr.xpos.min() - 10, 0)
        self.maxs[YPOS] = min(inr.ypos.max() + 10, c.DimY())
        self.mins[YPOS] = max(inr.ypos.min() - 10, 0)
        self.maxs[VSYS] = AlltoVel(c.getZphys(c.DimZ() - 1), h)
        self.mins[VSYS] = AlltoVel(c.getZphys(0), h)
        if self.maxs[VSYS] < self.mins[VSYS]:
            self.maxs[VSYS], self.mins[VSYS] = self.mins[VSYS], self.maxs[VSYS]
        self.maxs[VRAD], self.mins[VRAD] = 500, -500
        # The model engine (quantities derived from the header and beam kernel)
        self.model = DiskModel(c, par=par, densflux=True)
        if not self.model.smooth:
            par.SM = False
        self.outr = self.inr.copy()
        self.pixScale = ((abs(h.Cdelt(0)) * self.arcconv) + (abs(h.Cdelt(1)) * self.arcconv)) / 2.
        self.cdeltarc = (abs(h.Cdelt(0)) * self.arcconv, abs(h.Cdelt(1)) * self.arcconv)
        self.scaled_data = c.array
        self.data_T = np.ascontiguousarray(c.array.transpose(1, 2, 0))
        self._rngs = {}
        self._tls = threading.local()
        self._ringstat = {}             # outcome of the fit of each ring (for the summary)

    def __getstate__(self):
        st = self.__dict__.copy()
        st.pop("_tls", None)            # thread-local work arrays and the lock are not picklable
        st.pop("_lock", None)
        return st

    def __setstate__(self, st):
        self.__dict__.update(st)
        self._tls = threading.local()
        self._lock = threading.Lock()

    def setFree(self):
        f = self.par.FREE.lower()
        m = [False] * MAXPAR
        if "vrot" in f: m[VROT] = True
        if "disp" in f: m[VDISP] = True
        if "z0" in f: m[Z0] = True
        if "inc" in f: m[INC] = True
        if "pa" in f or "phi" in f: m[PA] = True
        if "xpos" in f: m[XPOS] = True
        if "ypos" in f: m[YPOS] = True
        if "vsys" in f: m[VSYS] = True
        if "vrad" in f: m[VRAD] = True
        if "all" in f: m = [True] * MAXPAR
        if not any(m):
            raise BBaroloError(" 3DFIT ERROR: NO free parameters!")
        self.mpar = m
        self.nfree = sum(m)

    def Norms(self, i=2):
        return self.norms[i]

    def Inrings(self): return self.inr
    def Outrings(self): return self.outr

    # ------------------------------------------------------------------
    # Model & residuals
    # ------------------------------------------------------------------
    def _support_args(self, support):
        """Arguments of _norm that emulate the support of Monte-Carlo models for ANALYTIC
        models. In BBarolo's residuals, the model voxels outside the mask count as blanks
        only where the model is non-zero. A MC model is non-zero only where at least one
        sub-cloud contributes (within the footprint of the convolution kernel), while the
        analytic line wings never vanish: without this correction an ANALYTIC model would
        always be penalised for its full +-5 sigma wings, independently of the dispersion,
        biasing the fitted dispersions high. A voxel of an ANALYTIC model is thus counted as
        a blank only if at least MC_SUPPORT_NMIN sub-clouds are expected to contribute to
        it; with this choice the number of blanks matches the average one of MC models."""
        if not support or support[1] <= 0:
            return _NO_SUPPORT
        pre, unit = support
        dm = self.model
        cd2i = dm.setup.cd2i
        if dm.conv is not None:
            K = dm.cfield
            # kernel cells where one sub-cloud gives a value above the 1E-12 threshold
            kmin = 1.E-12 / (unit * float(cd2i.max()) * float(dm.scalefactor))
            fp = K > max(kmin, 0.)
            ky, kx = K.shape
            hx = kx // 2
            anyr = fp.any(axis=1)
            i0 = np.where(anyr, fp.argmax(axis=1), 0)
            i1 = np.where(anyr, kx - 1 - fp[:, ::-1].argmax(axis=1), -1)
            # output (y,x) receives the input row y+ky//2-j, columns x+ilo[j]..x+ihi[j]
            ilo = np.where(anyr, hx - i1, 1).astype(np.int64)
            ihi = np.where(anyr, hx - i0, 0).astype(np.int64)
        else:
            ilo = np.zeros(1, np.int64)
            ihi = np.zeros(1, np.int64)
        thr = np.ascontiguousarray(MC_SUPPORT_NMIN * unit * cd2i, dtype=np.float64)
        # work array (reused by each thread, to avoid allocating fresh memory at each call)
        ky = len(ilo)
        ny, nx, nz = pre.shape
        n = ky * (nx + 1) * nz
        buf = getattr(self._tls, "pbuf", None)
        if buf is None or buf.size < n:
            buf = self._tls.pbuf = np.empty(n)
        return pre, ilo, ihi, thr, buf[:n].reshape(ky, nx + 1, nz)

    def _norm_value(self, dring, model, blo, support=None):
        par = self.par
        ps = self.pixScale
        normtype = NORMCODE[self.norms[self.n_galfit - 1]] if self.n_galfit else 0
        return _norm(model, self.data_T, self.mask_T, blo[0], blo[1],
                     dring.xpos[-1] - blo[0], dring.ypos[-1] - blo[1], dring.phi[-1], dring.inc[-1],
                     dring.radii[0] / ps, dring.radii[-1] / ps, float(self.wpow), SIDECODE.get(par.SIDE, 0),
                     par.FTYPE, self.chan_noise, float(self.data_noise), float(par.BWEIGHT), normtype,
                     *self._support_args(support))

    def getFuncValue(self, dring, modsoFar=None):
        if self.reverse:
            blo, bhi = self.model.box(self.outr)
        else:
            blo, bhi = self.model.box(dring)
        support = []
        mod = self.model.array(dring, blo, bhi, final=False, modsoFar=modsoFar, layout="yxz", support=support)
        return self._norm_value(dring, mod, blo, support)

    def calculateResiduals(self, r):
        return self.getFuncValue(r)

    # ------------------------------------------------------------------
    # Minimisation (port of minimize, mtry, func3D)
    # ------------------------------------------------------------------
    def _rng(self):
        # The random numbers of a ring fit come from a generator seeded by the stage and the
        # ring (set in _fit_ring), so that the results do not depend on the number of
        # threads, on which thread fits which ring, or on what ran before in the process.
        r = getattr(self._tls, "rng", None)
        if r is not None:
            return r
        # outside a ring fit (e.g. func3D called directly): one generator per thread
        tid = threading.get_ident()
        with self._lock:
            r = self._rngs.get(tid)
            if r is None:
                r = np.random.default_rng([12345, len(self._rngs)])
                self._rngs[tid] = r
        return r

    def func3D(self, dring, zpar, modsoFar=None):
        par, inr, mins, maxs = self.par, self.inr, self.mins, self.maxs
        w = dring.id
        fran = self._rng().random
        vals = dict(vrot=dring.vrot[0], vdisp=dring.vdisp[0], dens=dring.dens[0], z0=dring.z0[0],
                    inc=dring.inc[0], phi=dring.phi[0], xpos=dring.xpos[0], ypos=dring.ypos[0],
                    vsys=dring.vsys[0], vrad=dring.vrad[0])
        np_ = 0
        m = self.mpar
        if m[VROT]:
            minv = max(inr.vrot[w] - par.DELTAVROT, mins[VROT])
            maxv = min(inr.vrot[w] + par.DELTAVROT, maxs[VROT])
            z = zpar[np_]
            if z > maxv: v = maxv - fran() * par.DELTAVROT
            elif z < minv: v = minv + fran() * par.DELTAVROT
            else: v = z
            vals["vrot"] = v; zpar[np_] = v; np_ += 1
        if m[VDISP]:
            z = zpar[np_]
            # (redrawn within the bounds: C++ used u·VDISP, which crossed the other bound when the initial
            # VDISP exceeded MAXVDISP - MINVDISP)
            span = min(inr.vdisp[w], maxs[VDISP] - mins[VDISP])
            if z > maxs[VDISP]: v = maxs[VDISP] - fran() * span
            elif z < mins[VDISP]: v = mins[VDISP] + fran() * span
            else: v = z
            vals["vdisp"] = v; zpar[np_] = v; np_ += 1
        if m[DENS]:
            # (a negative DENS goes back to the initial one, as Z0: C++ had no bound and stopped)
            z = zpar[np_]
            v = inr.dens[w] if z < 0 else z
            vals["dens"] = v; zpar[np_] = v; np_ += 1
        if m[Z0]:
            z = zpar[np_]
            v = inr.z0[w] if (z < mins[Z0] or z > maxs[Z0]) else z
            vals["z0"] = v; zpar[np_] = v; np_ += 1
        if m[INC]:
            mini = max(inr.inc[w] - par.DELTAINC, mins[INC])
            maxi = min(inr.inc[w] + par.DELTAINC, maxs[INC])
            z = zpar[np_]
            v = maxi if z > maxi else (mini if z < mini else z)
            vals["inc"] = v; zpar[np_] = v; np_ += 1
        if m[PA]:
            minp = max(inr.phi[w] - par.DELTAPHI, mins[PA])
            maxp = min(inr.phi[w] + par.DELTAPHI, maxs[PA])
            z = zpar[np_]
            v = maxp if z > maxp else (minp if z < minp else z)
            vals["phi"] = v; zpar[np_] = v; np_ += 1
        for idx, key in ((XPOS, "xpos"), (YPOS, "ypos"), (VSYS, "vsys"), (VRAD, "vrad")):
            if m[idx]:
                z = zpar[np_]
                v = getattr(inr, key)[w] if (z < mins[idx] or z > maxs[idx]) else z
                vals[key] = v; zpar[np_] = v; np_ += 1
        for k, v in vals.items():
            getattr(dring, k)[:] = v
        return self.getFuncValue(dring, modsoFar)

    def minimize(self, dring, modsoFar=None):
        """Downhill simplex (Nelder & Mead, NR implementation). Returns (ok, minimum, pmin)."""
        TINY = 1.0e-10
        tol = self.par.TOL
        ndim = self.nfree
        if ndim == 0:
            # nothing left to fit: the second stage when only INC, PA, Z0, XPOS, YPOS or VSYS were free
            # (regularised, then fixed). The ring is the model at its values (C++ read beyond the simplex).
            return True, float(self.func3D(dring, np.empty(0), modsoFar)), np.empty(0)
        NMAX = 200 * ndim
        mpts = ndim + 1
        m = self.mpar
        point, dels = [], []
        if m[VROT]:
            dels.append(30. if dring.vrot[0] > 40 else 5); point.append(dring.vrot[0])
        if m[VDISP]:
            v = dring.vdisp[0]
            dels.append(20. if v > 80 else (10. if 20. < v <= 80. else (0.2 if v <= 2 else 5)))
            point.append(v)
        if m[DENS]: dels.append(0.2 * dring.dens[0]); point.append(dring.dens[0])
        if m[Z0]: dels.append(0.2 * dring.z0[0]); point.append(dring.z0[0])
        if m[INC]: dels.append(5.); point.append(dring.inc[0])
        if m[PA]: dels.append(10.); point.append(dring.phi[0])
        if m[XPOS]: dels.append((self.maxs[XPOS] - self.mins[XPOS]) / 4.); point.append(dring.xpos[0])
        if m[YPOS]: dels.append((self.maxs[YPOS] - self.mins[YPOS]) / 4.); point.append(dring.ypos[0])
        if m[VSYS]: dels.append((self.maxs[VSYS] - self.mins[VSYS]) / 4.); point.append(dring.vsys[0])
        if m[VRAD]: dels.append(25); point.append(dring.vrad[0])
        p = np.tile(np.array(point, float), (mpts, 1))
        for i in range(1, mpts):
            p[i, i - 1] += dels[i - 1]
        y = np.empty(mpts)
        for i in range(mpts):
            x = p[i].copy()
            y[i] = self.func3D(dring, x, modsoFar)
        nfunc = 0
        psum = p.sum(axis=0)
        while True:
            ilo = 0
            if y[0] > y[1]:
                ihi, inhi = 0, 1
            else:
                ihi, inhi = 1, 0
            for i in range(mpts):
                if y[i] <= y[ilo]: ilo = i
                if y[i] > y[ihi]:
                    inhi = ihi
                    ihi = i
                elif y[i] > y[inhi] and i != ihi:
                    inhi = i
            rtol = 2.0 * abs(y[ihi] - y[ilo]) / (abs(y[ihi]) + abs(y[ilo]) + TINY)
            if rtol < tol:
                y[0], y[ilo] = y[ilo], y[0]
                p[[0, ilo]] = p[[ilo, 0]]
                return True, float(y[0]), p[0].copy()
            if nfunc >= NMAX:
                return False, float(y[ilo]), p[ilo].copy()
            nfunc += 2
            ytry = self._mtry(dring, p, y, psum, ihi, -1.0, modsoFar)
            if ytry <= y[ilo]:
                ytry = self._mtry(dring, p, y, psum, ihi, 2.0, modsoFar)
            elif ytry >= y[inhi]:
                ysave = y[ihi]
                ytry = self._mtry(dring, p, y, psum, ihi, 0.5, modsoFar)
                if ytry >= ysave:
                    for i in range(mpts):
                        if i != ilo:
                            p[i] = 0.5 * (p[i] + p[ilo])
                            ps_ = p[i].copy()
                            y[i] = self.func3D(dring, ps_, modsoFar)
                            # in the C++ code psum is (ab)used as work array: the clamped values
                            # are written back to it, not to the simplex
                    nfunc += ndim
                    psum = p.sum(axis=0)
            else:
                nfunc -= 1

    def _mtry(self, dring, p, y, psum, ihi, fac, modsoFar):
        ndim = self.nfree
        fac1 = (1.0 - fac) / ndim
        fac2 = fac1 - fac
        ptry = psum * fac1 - p[ihi] * fac2
        ytry = self.func3D(dring, ptry, modsoFar)
        if ytry < y[ihi]:
            y[ihi] = ytry
            psum += ptry - p[ihi]
            p[ihi] = ptry
        return ytry

    # ------------------------------------------------------------------
    # Fitting loops
    # ------------------------------------------------------------------
    def _make_dring(self, ir):
        inr = self.inr
        if inr.nr == 1:
            w1 = w2 = inr.radsep / 2. if inr.radsep > 0 else inr.radii[0] / 2.
        elif ir == 0:
            w1 = w2 = (inr.radii[1] - inr.radii[0]) / 2.
        elif ir == inr.nr - 1:
            w1 = w2 = (inr.radii[ir] - inr.radii[ir - 1]) / 2.
        else:
            w1 = (inr.radii[ir] - inr.radii[ir - 1]) / 2.
            w2 = (inr.radii[ir + 1] - inr.radii[ir]) / 2.
        w1, w2 = np.float32(w1), np.float32(w2)
        drads = [max(float(np.float32(inr.radii[ir] - w1)), 0.), max(float(np.float32(inr.radii[ir] + w2)), 0.)]
        d = Rings()
        d.setRings(radii=drads, xpos=inr.xpos[ir], ypos=inr.ypos[ir], vsys=inr.vsys[ir], vrot=inr.vrot[ir],
                   vdisp=inr.vdisp[ir], vrad=inr.vrad[ir], vvert=inr.vvert[ir], dvdz=inr.dvdz[ir],
                   zcyl=inr.zcyl[ir], dens=inr.dens[ir], z0=inr.z0[ir], inc=inr.inc[ir], phi=inr.phi[ir])
        d.radsep = drads[1] - drads[0]
        d.id = ir
        return d

    def _store(self, ir, pmin):
        k = 0
        o = self.outr
        for idx, key in ((VROT, "vrot"), (VDISP, "vdisp"), (DENS, "dens"), (Z0, "z0"), (INC, "inc"),
                         (PA, "phi"), (XPOS, "xpos"), (YPOS, "ypos"), (VSYS, "vsys"), (VRAD, "vrad")):
            if self.mpar[idx]:
                getattr(o, key)[ir] = pmin[k]
                k += 1

    def _max_vrot(self, ir):
        """The largest VROT whose projection still falls in the spectral band (for VSYS and INC of
        the ring ir)."""
        c, h, o = self.cin, self.cin.Head(), self.outr
        maxv = abs(AlltoVel(c.getZphys(c.DimZ() - 1), h) - o.vsys[ir])
        minv = abs(AlltoVel(c.getZphys(0), h) - o.vsys[ir])
        sini = math.sin(o.inc[ir] * math.pi / 180.)
        return max(maxv / sini, minv / sini) if sini != 0 else np.inf

    def _check_vrot(self, ir):
        return self.outr.vrot[ir] <= self._max_vrot(ir)

    def _fit_ring(self, ir, errors, fitok, nthreads, modsoFar=None):
        # the generator of this ring is used only during its fit (see _rng)
        self._tls.rng = np.random.default_rng([12345, self.n_galfit, ir])
        try:
            self._fit_one_ring(ir, errors, fitok, nthreads, modsoFar)
        finally:
            self._tls.rng = None
            bar = getattr(self, "_bar", None)
            if bar is not None:
                with self._lock:
                    self._ndone += 1
                    bar.update(self._ndone)

    def _fit_one_ring(self, ir, errors, fitok, nthreads, modsoFar=None):
        inr = self.inr
        if self.verb >= 2 and nthreads == 1:
            print(f"\n Working on ring #{ir + 1} at radius {inr.radii[ir]:.2f} arcsec... "
                  f"[{time.strftime('%H:%M:%S')}]")
        dring = self._make_dring(ir)
        if modsoFar is None and not self.reverse:
            blo, bhi = self.model.box(dring)
            ps = self.pixScale
            nTot, nIn = _count_ring_pixels(self.mask2D, blo[0], blo[1], bhi[0], bhi[1],
                                           dring.xpos[-1] - blo[0], dring.ypos[-1] - blo[1],
                                           dring.phi[-1], dring.inc[-1], dring.radii[0] / ps, dring.radii[-1] / ps)
            if nIn < 20 and (nIn == 0 or nIn < 0.7 * nTot):
                if self.verb >= 2:
                    WarningMessage(sys.stdout, f" Not enough pixels to fit in ring #{ir + 1}. I will skip it.")
                self._ringstat[ir] = ("skipped: few pixels in the mask", None)
                fitok[ir] = False
                return
        ok, minimum, pmin = self.minimize(dring, modsoFar)
        fitok[ir] = ok
        if not ok:
            self._ringstat[ir] = ("not converged", minimum)
            if self.verb >= 2:
                WarningMessage(sys.stdout, f" Can not achieve convergence in ring #{ir + 1}. I'll keep going, but \n"
                                           " parameters for this ring are wrong! Please, try to change initial \n"
                                           " conditions and/or the function to minimize.")
            return
        self._store(ir, pmin)
        if not self._check_vrot(ir):
            self._ringstat[ir] = (f"rejected: VROT > {self._max_vrot(ir):.0f} km/s (the band)", minimum)
            if self.verb >= 2:
                o = self.outr
                WarningMessage(sys.stdout, f" Ring #{ir + 1}: the fitted VROT ({o.vrot[ir]:.1f} km/s) is larger "
                                           f"than the band allows\n (at most {self._max_vrot(ir):.1f} km/s for "
                                           f"VSYS={o.vsys[ir]:.1f} km/s and INC={o.inc[ir]:.1f} deg).\n The ring "
                                           "is treated as a non-converged fit (try other initial values,\n e.g. "
                                           "a lower VROT, or a wider band).")
            fitok[ir] = False
            return
        self._ringstat[ir] = ("ok", minimum)
        if self.verb >= 2:
            with self._lock:
                printRing(sys.stdout, self.outr, ir, minimum, self.mpar, nthreads)
        if self.par.flagERRORS:
            self.getErrors(dring, errors[ir], ir, minimum)

    def galfit(self):
        """One fitting stage (port of Galfit::galfit)."""
        self.n_galfit = 2 if self.n_galfit == 1 else 1
        n = self.n_galfit
        c = self.cin
        p = c.pars()
        if self.verb:
            p.verbose = False
            print("\n" + "=" * 31 + " 3DFIT " + "=" * 31)
        self.outr = self.inr.copy()
        scaling = 1.
        dn_orig, dT_orig = self.data_noise, self.data_T
        if self.par.NORMALCUBE:
            # the data, and their noise, are scaled during the fit only (restored below)
            scaling = 10. / c.stat().getMax()
            self.scaled_data = (c.array * scaling).astype(c.array.dtype)
            self.data_T = np.ascontiguousarray(self.scaled_data.transpose(1, 2, 0))
        else:
            self.scaled_data = c.array
        self.data_noise = self.data_noise * scaling
        nr = self.inr.nr
        errors = np.zeros((nr, 2, self.nfree))
        fitok = [False] * nr
        nthreads = p.threads
        self._ringstat, self._ndone, self._bar = {}, 0, None
        start = self.par.STARTRAD if self.par.STARTRAD < nr else 0
        if self.verb == 1:                       # (VERBOSE 2: every ring, as in C++)
            self._bar = ProgressBar(True, True, p.showbar)
            self._bar.init(f" Stage {n}: fitting {nr - start} rings with {max(1, nthreads)} threads ", nr - start)
        try:
            if self.reverse:
                self.fit_reverse(errors, fitok)
            else:
                start = self.par.STARTRAD if self.par.STARTRAD < nr else 0
                if nthreads <= 1:
                    for ir in range(start, nr):
                        self._fit_ring(ir, errors, fitok, 1)
                else:
                    with ThreadPoolExecutor(nthreads) as ex:
                        futs = [ex.submit(self._fit_ring, ir, errors, fitok, nthreads) for ir in range(start, nr)]
                        for f in futs:
                            f.result()
        finally:
            self.data_noise, self.data_T = dn_orig, dT_orig
            self.scaled_data = c.array
            if self._bar is not None:
                self._bar.fillSpace(" Done.\n")
                self._bar = None
        if self.verb == 1:
            self._print_rings(range(start, nr))
        self.fitok = fitok
        self.errors = errors
        fout = io.StringIO()
        writeHeader(fout, self.mpar, self.par.flagERRORS, self.par.flagBADOUT)
        for ir in range(nr):
            if fitok[ir] or self.par.flagBADOUT:
                writeRing(fout, self.outr, ir, self.nfree, self.par.flagERRORS, errors,
                          self.par.flagBADOUT, fitok[ir])
        # the outcome of the stage (the rings as fitted, before the failed ones are emptied), and the text of
        # <prefix>_rings_final<n>.txt, written by the task
        if not hasattr(self, "stages"):
            self.stages = {}
        self.stages[n] = types.SimpleNamespace(rings=self.outr.copy(), errors=errors.copy(), fitok=list(fitok),
                                               mpar=list(self.mpar), text=fout.getvalue())
        c.add_output(f"_rings_final{n}.txt", fout.getvalue())
        m = self.mpar
        is2 = m[INC] or m[PA] or m[Z0] or m[XPOS] or m[YPOS] or m[VSYS]
        lastround = n == 2 or (n == 1 and not (self.par.TWOSTAGE and is2))
        if lastround:
            for ir in range(nr):
                if not fitok[ir]:
                    self.outr.dens[ir] = 0
        if self.verb:
            print("=" * 69 + "\n")
            p.verbose = self.verb

    def _print_rings(self, rings):
        """The rings of a stage, in order of radius: the free parameters, fmin and the outcome."""
        names = [("VROT", VROT, "vrot"), ("VDISP", VDISP, "vdisp"), ("DENS", DENS, "dens"), ("Z0", Z0, "z0"),
                 ("INC", INC, "inc"), ("PA", PA, "phi"), ("XPOS", XPOS, "xpos"), ("YPOS", YPOS, "ypos"),
                 ("VSYS", VSYS, "vsys"), ("VRAD", VRAD, "vrad")]
        # (the second stage shows the columns of the first: the parameters that are not fitted any more,
        # regularised, are marked (r))
        shown = getattr(self, "_shown", None) or self.mpar
        cols = [x for x in names if shown[x[1]]]
        o = self.outr
        head = f"   {'#':>3s} {'RAD':>9s}" + "".join(f" {x[0] + ('' if self.mpar[x[1]] else '(r)'):>8s}" for x in cols)
        print(head + f" {'fmin':>10s}  outcome")
        for ir in rings:
            st, fmin = self._ringstat.get(ir, ("not fitted", None))
            line = f"   {ir + 1:3d} {o.radii[ir]:9.2f}"
            fitted = st == "ok" or st.startswith("rejected") or st == "not converged"
            for _, idx, attr in cols:
                v = getattr(o, attr)[ir]
                line += f" {v:8.2f}" if fitted or not self.mpar[idx] else f" {'-':>8s}"
            line += f" {fmin:10.4g}" if fmin is not None else f" {'-':>10s}"
            print(line + "  " + st)
        sys.stdout.flush()

    def fit_reverse(self, errors, fitok):
        inr = self.inr
        start = self.par.STARTRAD if self.par.STARTRAD < inr.nr else 0
        for ir in range(inr.nr - 1, start - 1, -1):
            modsoFar = None
            if ir != inr.nr - 1:
                k = ir + 1
                d2 = self.outr.subset(slice(k, inr.nr))
                d2.id = ir + 1
                blo, bhi = self.model.box(self.outr)
                modsoFar = self.model.setup.build(d2, blo, bhi, layout="yxz")
            self._fit_ring(ir, errors, fitok, 1, modsoFar=modsoFar)

    # ------------------------------------------------------------------
    # Second stage
    # ------------------------------------------------------------------
    def SecondStage(self):
        m = self.mpar
        isNeeded = m[INC] or m[PA] or m[Z0] or m[XPOS] or m[YPOS] or m[VSYS]
        if not isNeeded:
            self.par.TWOSTAGE = False
            return isNeeded
        self.second = True
        nr = self.outr.nr
        # only the rings fitted in the first stage are regularised (C++ used also the rings skipped, not
        # converged, rejected or inside STARTRAD, with their initial or rejected values)
        good = np.asarray(getattr(self, "fitok", [True] * nr), bool)
        ngood = int(good.sum())
        reg = [-3] * 6
        regstr = ["inc", "pa", "vsys", "xpos", "ypos", "z0"]

        def getval(s):
            if s.isdigit(): return 1 + int(s)
            if s == "bezier": return -1
            if s == "median": return -2
            return -3
        auto = [False] * 6
        Polyn = self.par.REGTYPE.lower().split()
        keys = [w.split("=", 1) for w in Polyn if "=" in w]
        if len(Polyn) == 1 and not keys:
            reg[0] = reg[1] = getval(Polyn[0])
        else:
            for k, v in keys:
                if k in regstr:
                    reg[regstr.index(k)] = getval(v)
        for i in range(6):
            if reg[i] == -3:
                auto[i] = True
                if i < 2:
                    if ngood <= 4:
                        reg[i] = 1
                    else:
                        vec = (self.outr.inc if i == 0 else self.outr.phi)[good]
                        med = findMedian(vec)
                        if madfmToSigma(findMADFM(vec, med)) > 3:
                            reg[i] = 2 if ngood < 10 else -1
                        else:
                            reg[i] = -2
                else:
                    reg[i] = -2
        self.inr = self.outr.copy()
        proceed = ngood > 0
        how = []
        for flag, key, r, name, short, unit, a in (
                (m[INC], "inc", reg[0], "inclination", "INC", "deg", auto[0]),
                (m[PA], "phi", reg[1], "position angle", "PA", "deg", auto[1]),
                (m[VSYS], "vsys", reg[2], "systemic velocity", "VSYS", "km/s", auto[2]),
                (m[XPOS], "xpos", reg[3], "X center", "XPOS", "pix", auto[3]),
                (m[YPOS], "ypos", reg[4], "Y center", "YPOS", "pix", auto[4]),
                (m[Z0], "z0", reg[5], "disk thickness", "Z0", "arcs", auto[5])):
            if flag and proceed:
                x, y = self.inr.radii, getattr(self.inr, key)
                if good.all():
                    ok, out = self.regularizeParams(x, y, r, name)
                else:                               # (fitted on the good rings, evaluated at all the radii)
                    ok, out = self.regularizeParams(x[good], y[good], r, name, xout=x)
                if ok:
                    setattr(self.inr, key, out)
                    deg = (max(r - 1, 0) if ngood > r else 0) if r > 0 else None
                    how.append(f"{short} " + (f"median = {out[0]:.2f} {unit}" if r == -2 else "Bezier curve" if r == -1
                                               else f"constant = {out[0]:.2f} {unit}" if deg == 0
                                               else f"polynomial of degree {deg}") + (" (auto)" if a else ""))
                proceed = proceed and ok
        if proceed and self.verb:
            print(" Regularised for the second stage (then fixed): " + ", ".join(how) + ".")
        oldmpar, oldnfree = list(self.mpar), self.nfree
        for i in (INC, PA, VSYS, XPOS, YPOS, Z0):
            self.mpar[i] = False
        # VROT, VDISP, VRAD and DENS (free with FREE=ALL) stay free (C++ forgot DENS here, so with
        # FREE=ALL the simplex was degenerate and the errors misaligned)
        self.nfree = sum(self.mpar)
        if proceed:
            self._shown = oldmpar
            try:
                self.galfit()
            finally:
                self._shown = None
        else:
            if self.verb:
                sys.stderr.write("3DFIT ERROR: Regularization failed. I will not proceed to second fitting stage. \n")
            # the first stage is final: the rings not fitted are empty in the final model (C++ kept their DENS)
            for ir in range(nr):
                if not good[ir]:
                    self.outr.dens[ir] = 0
        self.mpar, self.nfree = oldmpar, oldnfree
        return isNeeded

    def regularizeParams(self, x, y, rtype, whichpar="", xout=None):
        """Regularise the values y at the radii x: median (rtype -2), Bezier curve (-1) or polynomial with
        rtype coefficients. Returns (ok, values at the radii x, or at the radii xout if given)."""
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        xo = x if xout is None else np.asarray(xout, float)
        n = x.size
        if rtype == -2:
            return True, np.full(xo.size, findMedian(y))
        if rtype == -1:
            xb, yo = bezier_interp(x, y)
            return True, (yo if xout is None else np.interp(xo, xb, yo))
        if n <= rtype:
            if self.verb:
                sys.stderr.write("3DFIT WARNING - Second stage: too few degree of freedom. Using a constant value.\n")
            rtype = 1
        xx, yy = list(x), list(y)
        if n - 2 > rtype:
            # deleting maximum and minimum values (reproducing the C++ index handling)
            w = int(np.argmax(yy)); del yy[w]; del xx[w]
            w = int(np.argmin(yy)); del yy[w]; del xx[w]
        coeff, coefferr, ok = polyfit_lsq(np.array(xx), np.array(yy), rtype)
        if not ok:
            if self.verb:
                sys.stderr.write(f"3DFIT ERROR: cannot least-square fit {whichpar}.\n")
            return False, y
        yout = np.polyval(coeff[::-1], xo)
        if self.verb >= 2:
            print(f"  Best parameters for {whichpar}:")
            for i in range(rtype):
                print(f"{'a' + str(i) + ' = ':>43}{coeff[i]:9.4g}{' ±':>4}{coefferr[i]:9.4g}")
        return True, yout

    # ------------------------------------------------------------------
    # Errors (port of getErrors)
    # ------------------------------------------------------------------
    def getErrors(self, dr, err, ir, minimum):
        err[:] = 0.
        o = self.outr
        keys = {VROT: "vrot", VDISP: "vdisp", DENS: "dens", Z0: "z0", INC: "inc", PA: "phi",
                XPOS: "xpos", YPOS: "ypos", VSYS: "vsys", VRAD: "vrad"}

        def reset():
            for k in keys.values():
                getattr(dr, k)[:] = getattr(o, k)[ir]
        reset()
        free_var = [i for i in range(MAXPAR) if self.mpar[i]]
        nfree = len(free_var)
        maxval = np.full(nfree, -1.E04)
        minval = np.full(nfree, 1.E04)
        midval = np.array([getattr(o, keys[f])[ir] for f in free_var], float)
        n_models, n_bin = 100, 40
        rng = self._rng()
        bar = ProgressBar(True, self.verb and self.cin.pars().threads == 1, self.cin.pars().showbar)
        bar.init(" Estimating errors... ", n_models * nfree)
        cc = 1
        for nf in range(nfree - 1, -1, -1):
            reset()
            fv = free_var[nf]
            if fv == VROT:
                maxv = o.vrot[ir] + o.vrot[ir] / 3.
                minv = max(o.vrot[ir] - o.vrot[ir] / 3., 0)
            elif fv == VDISP:
                maxv = o.vdisp[ir] + 15
                minv = max(o.vdisp[ir] - 15, 0)
            else:
                minv, maxv = self.mins[fv], self.maxs[fv]
            var_val = np.empty(n_models)
            var_func = np.empty(n_models)
            for nm in range(n_models - 1, -1, -1):
                bar.update(cc); cc += 1
                val = rng.random() * (maxv - minv) + minv
                getattr(dr, keys[fv])[:] = val
                var_val[nm] = val - midval[nf]
                maxval[nf] = max(maxval[nf], var_val[nm])
                minval[nf] = min(minval[nf], var_val[nm])
                var_func[nm] = self.getFuncValue(dr)
            bin_range = (maxval[nf] - minval[nf]) / n_bin
            xx_bin = np.empty(n_bin)
            yy_bin = np.empty(n_bin)
            for ib in range(n_bin):
                b0 = minval[nf] + ib * bin_range
                b1 = b0 + bin_range
                xx_bin[ib] = b0 + bin_range / 2.
                sel = (var_val >= b0) & (var_val < b1)
                yy_bin[ib] = np.min(100 * (var_func[sel] - minimum) / minimum) if sel.any() else 0
                if sel.any() and yy_bin[ib] > 10000:
                    yy_bin[ib] = 10000
            order = 3
            ndp, y_interv = 0, 15
            ww = np.zeros(n_bin)
            it = 0
            while ndp < 2 * order and it < 1000:
                ww = np.where((yy_bin > y_interv) | (yy_bin <= 0), 0., 1.)
                ndp += int(ww.sum())
                y_interv += 5
                it += 1
            # cubic a2*x^2 + a3*x^3 (constant and linear terms fixed to zero)
            coeff, _, ok = polyfit_lsq(xx_bin, yy_bin, order + 1, ww, free=[False, False, True, True])
            if not ok:
                continue
            threshold = 5.
            a = coeff[2] / coeff[3]
            b = coeff[1] / coeff[3]
            cc_ = -threshold / coeff[3]
            Q = (a * a - 3 * b) / 9.
            R = (2 * a * a * a - 9 * a * b + 27 * cc_) / 54.
            low_err = upp_err = 0.
            if R * R < Q * Q * Q:
                TT = math.acos(R / math.sqrt(Q * Q * Q))
                xs = [-2 * math.sqrt(Q) * math.cos(TT / 3.) - a / 3.,
                      -2 * math.sqrt(Q) * math.cos((TT + 2 * math.pi) / 3.) - a / 3.,
                      -2 * math.sqrt(Q) * math.cos((TT - 2 * math.pi) / 3.) - a / 3.]
                inb = [minval[nf] <= x_ <= maxval[nf] for x_ in xs]
                if all(inb):
                    rs = sorted(xs)
                    npos = sum(r_ > 0 for r_ in rs)
                    if npos == 3: low_err, upp_err = -rs[0], rs[0]
                    elif npos == 2: low_err, upp_err = rs[0], rs[1]
                    elif npos == 1: low_err, upp_err = rs[1], rs[2]
                    else: low_err, upp_err = rs[2], -rs[2]
                elif sum(inb) == 2:
                    x1, x2 = sorted([x_ for x_, b_ in zip(xs, inb) if b_])
                    if x1 > 0 and x2 > 0: low_err, upp_err = -x1, x1
                    elif x1 > 0 or x2 > 0: low_err, upp_err = x1, x2
                elif sum(inb) == 1:
                    x1 = [x_ for x_, b_ in zip(xs, inb) if b_][0]
                    low_err, upp_err = (-x1, x1) if x1 > 0 else (x1, -x1)
                else:
                    continue
            else:
                A = -np.cbrt(R + math.sqrt(R * R - Q * Q * Q))
                B = Q / A if A != 0 else 0
                x1 = (A + B) - a / 3.
                if not (minval[nf] <= x1 <= maxval[nf]):
                    continue
                low_err, upp_err = (-x1, x1) if x1 > 0 else (x1, -x1)
            err[0][nf] = low_err
            err[1][nf] = upp_err
        for nf in range(nfree - 1, -1, -1):
            if err[0][nf] == 0: err[0][nf] = np.mean(err[0][:nfree])
            if err[1][nf] == 0: err[1][nf] = np.mean(err[1][:nfree])
            if free_var[nf] == VDISP:
                minerr_vd = DeltaVel(self.cin.Head()) / (2 * math.sqrt(2 * math.log(2)))
                if abs(err[0][nf]) < minerr_vd / 2.: err[0][nf] = -abs(minerr_vd / 2. + rng.random())
                if abs(err[1][nf]) < minerr_vd / 2.: err[1][nf] = abs(minerr_vd / 2. + rng.random())
        bar.fillSpace("Done.\n")

    # ------------------------------------------------------------------
    # Asymmetric drift
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Outputs
    # ------------------------------------------------------------------
    def writeRingFile(self, filename, r, errors=None, extra=None):
        """<prefix>_<filename> in the output folder: the rings r (and the errors of the free parameters).
        extra: {NAME: integer value of each ring}, written as the last columns (e.g. INDISK of
        Bayesian3DFIT)."""
        fileo = self.cin.pars().outFolder + self.cin.pars().outPrefix + "_" + filename
        extra = extra or {}
        with open(fileo, "w") as fout:
            writeHeader(fout, self.mpar, self.par.flagERRORS, False, extra=list(extra))
            for i in range(r.nr):
                writeRing(fout, r, i, self.nfree, self.par.flagERRORS and errors is not None, errors, False, False,
                          extra=[v[i] for v in extra.values()])

    def showInitial(self, inr, stream, pp=None):
        stream.write(initialParametersText(inr, getattr(self, "_estimated", {})))


    def getErrorColumns(self):
        free = [j for j in range(MAXPAR) if self.mpar[j]]
        nc = [-1] * MAXPAR
        for j in range(MAXPAR):
            if self.par.flagERRORS and self.mpar[j]:
                nc[j] = NCOLS + 2 * free.index(j)
        return nc

    def stage_table(self, n):
        """The rings fitted in stage n (1 or 2) as an astropy Table at full precision (see ring_table): all the
        rings, with the errors of the free parameters (FLAGERRORS) and fitok (False: the fit of the ring
        failed)."""
        st = self.stages[n]
        errs = st.errors if self.par.flagERRORS else None
        return ring_table(st.rings, st.mpar, errs, st.fitok)

    def model_outputs(self, normtype):
        """The final model of the fit and its outputs in memory (see galfit_out.model_outputs)."""
        from .galfit_out import model_outputs
        return model_outputs(self.cin, self.model, self.outr, normtype, self.mask, par=self.par, fit=self)

    def writeModel(self, normtype, makeplots=True):
        """Compute and write the final model of the fit and its outputs (see galfit_out.writeModel)."""
        from .galfit_out import writeModel
        return writeModel(self.cin, self.model, self.outr, normtype, self.mask, par=self.par,
                          makeplots=makeplots, fit=self)

    def writeOutputs(self, mod, ell, makeplots=True):
        """Write a model computed elsewhere and its outputs (see galfit_out.writeOutputs)."""
        from .galfit_out import writeOutputs
        return writeOutputs(self.cin, mod, ell, self.outr, self.mask, makeplots=makeplots, fit=self)

    def plotAll_Python(self, runner=None):
        from .plotscripts import plotAll_Python
        return plotAll_Python(self, runner)


# --------------------------------------------------------------------------
# Writing rings
# --------------------------------------------------------------------------
# The columns of the ring files of the fits (rings_final1.txt, ...), followed by the errors of the free
# parameters. The files of C++ BBarolo also have RAD(Kpc), first, and Z0(pc), before Z0(arcs).
RING_COLUMNS = ("RAD(arcs)", "VROT(km/s)", "DISP(km/s)", "INC(deg)", "P.A.(deg)", "Z0(arcs)", "DENS(un)",
                "XPOS(pix)", "YPOS(pix)", "VSYS(km/s)", "VRAD(km/s)")
NCOLS = len(RING_COLUMNS)


def writeHeader(fout, mpar, writeErrors, writeBadRings, extra=()):
    m = 10
    s = (f"{'#RAD(arcs)':<{m + 1}}{'VROT(km/s)':<{m + 1}}{'DISP(km/s)':<{m + 1}}{'INC(deg)':<{m}}"
         f"{'P.A.(deg)':<{m}}{'Z0(arcs)':<{m}}{'DENS(un)':<{m}}{'XPOS(pix)':<{m}}{'YPOS(pix)':<{m}}"
         f"{'VSYS(km/s)':<{m + 1}}{'VRAD(km/s)':<{m + 1}}")
    if writeErrors:
        names = ["VROT", "DISP", "DENS", "Z0", "INC", "PA", "XPOS", "YPOS", "VSYS", "VRAD"]
        for i, nm in enumerate(names):
            if mpar[i]:
                s += f"{'E_' + nm + '1':<{m}}{'E_' + nm + '2':<{m}}"
    if writeBadRings:
        s += f"{'FITOK':<{m}}"
    for nm in extra:
        s += f"{nm:<{m}}"
    fout.write(s + "\n")


def writeRing(fout, r, i, nfree, writeErrors, errors, writeBadRings, fitOK, extra=()):
    m = 10
    s = (f"{r.radii[i]:<{m + 1}.3f}{r.vrot[i]:<{m + 1}.3f}{r.vdisp[i]:<{m + 1}.3f}"
         f"{r.inc[i]:<{m}.3f}{r.phi[i]:<{m}.3f}{r.z0[i]:<{m}.3f}"
         f"{r.dens[i]:<{m}.1e}{r.xpos[i]:<{m}.3f}{r.ypos[i]:<{m}.3f}{r.vsys[i]:<{m + 1}.3f}{r.vrad[i]:<{m + 1}.3f}")
    if writeErrors:
        for kk in range(nfree):
            s += f"{errors[i][0][kk]:<{m}.3f}{errors[i][1][kk]:<{m}.3f}"
    if writeBadRings:
        s += f"{int(fitOK):<{m}d}"
    for v in extra:
        s += f"{int(v):<{m}d}"
    fout.write(s + "\n")


def printRing(fout, r, i, minimum, mpar, nthreads):
    k, n = 8, 11
    s = ""
    if nthreads > 1:
        s += f"\n Ring #{i + 1} at radius {r.radii[i]:.2f} arcsec... \n"
    s += f"  Best parameters for ring #{i + 1} (fmin = {minimum:.3e}):\n"

    def f(name, idx, val, unit, first):
        nm = name + ("" if mpar[idx] else "(f)")
        if first:
            return f"{nm:<{n}}{'= ':>3}{val:>{k}.2f}{unit:<{k}}"
        return f"{nm:<{n + 4}}{'= ':>3}{val:>{k - 1}.2f}{unit:<{k}}\n"
    s += f("    Vrot", VROT, r.vrot[i], "  km/s", True) + f("        Disp", VDISP, r.vdisp[i], "  km/s", False)
    s += f("    Vrad", VRAD, r.vrad[i], "  km/s", True) + f("        Vsys", VSYS, r.vsys[i], "  km/s", False)
    s += f("    Inc", INC, r.inc[i], "  deg", True) + f("        PA", PA, r.phi[i], "  deg", False)
    s += f("    Xpos", XPOS, r.xpos[i], "  pix", True) + f("        Ypos", YPOS, r.ypos[i], "  pix", False)
    s += f("    Z0", Z0, r.z0[i], "  arcs", True) + "\n"
    fout.write(s)
    fout.flush()


def WarningMessage(fout, msg):
    fout.write("\n ========================== 3DFIT WARNING ==========================\n"
               + msg + "\n ===================================================================\n\n")
    fout.flush()
