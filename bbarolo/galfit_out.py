"""
Outputs of the 3DFIT and GALMOD tasks (port of Tasks/galfit_out.cpp: writeModel,
writeOutputs, writeKinematicMaps, writePVs). Plotting scripts are in plotscripts.py.

The functions take the data cube, the model engine (galmod.DiskModel), the rings and
the mask, so that 3DFIT (through its Galfit object) and GALMOD (without any fit) share
them. finalModel() gives the final model of writeModel in memory, without writing files.
"""

import sys
import math
import contextlib
import numpy as np

from .ellprof import Ellprof
from .moments import getAllMoments
from .pvslice import PositionVelocity
from .stats import findMedian
from .bbutils import mkdirp, randomAdjective, polyfit_lsq

FLT_MAX = float(np.finfo(np.float32).max)


def writeKinematicMaps(c, mod, suffix=""):
    """Moment maps of the data (masked) and of the model in the maps/ folder."""
    p = c.pars()
    outfold, prefix = p.outFolder, p.outPrefix
    mkdirp(outfold + "maps/")
    v = p.verbose
    p.verbose = False
    allmaps = getAllMoments(c, True, None, "MOMENT")
    for i, s in enumerate(("_0mom", "_1mom", "_2mom")):
        allmaps[i].fitswrite_2d(outfold + "maps/" + prefix + s + ".fits")
    mod.pars().verbose = False
    modmaps = getAllMoments(mod, False, None, "MOMENT")
    for i, s in enumerate(("mod_0mom", "mod_1mom", "mod_2mom")):
        modmaps[i].fitswrite_2d(outfold + "maps/" + prefix + s + suffix + ".fits")
    p.verbose = v


def writePVs(c, mod, rings, mask, suffix=""):
    """PV slices of the data, of the model and of the mask along the major and minor axis
    (median position angle and centre of the rings) in the pvs/ folder."""
    from .cube import Cube
    p = c.pars()
    outfold, prefix = p.outFolder + "pvs/", p.outPrefix
    mkdirp(outfold)
    o = rings
    meanPA = float(np.float32(findMedian(o.phi)))
    meanX = float(np.float32(findMedian(o.xpos)))
    meanY = float(np.float32(findMedian(o.ypos)))
    meanPAp90 = meanPA + 90 if meanPA + 90 < 360 else meanPA - 90
    PositionVelocity(c, meanX, meanY, meanPA).fitswrite_2d(outfold + prefix + "_pv_a.fits")
    PositionVelocity(c, meanX, meanY, meanPAp90).fitswrite_2d(outfold + prefix + "_pv_b.fits")
    PositionVelocity(mod, meanX, meanY, meanPA).fitswrite_2d(outfold + prefix + "mod_pv_a" + suffix + ".fits")
    PositionVelocity(mod, meanX, meanY, meanPAp90).fitswrite_2d(outfold + prefix + "mod_pv_b" + suffix + ".fits")
    m = Cube.fromArray(mask.astype(np.int16), c.Head(), p)
    m.Head().setMinMax(0., 0)
    pa = PositionVelocity(m, meanX, meanY, meanPA, order=0)        # nearest pixel for the mask
    pa.array = pa.array.astype(np.int16)
    pa.fitswrite_2d(outfold + prefix + "mask_pv_a.fits")
    pb = PositionVelocity(m, meanX, meanY, meanPAp90, order=0)
    pb.array = pb.array.astype(np.int16)
    pb.fitswrite_2d(outfold + prefix + "mask_pv_b.fits")


# --------------------------------------------------------------------------
# The final model and its normalisation
# --------------------------------------------------------------------------
@contextlib.contextmanager
def _outputs_mode(p):
    """While the outputs are computed: not verbose, fluxes not converted."""
    verb, oldfc = p.verbose, p.fluxConvert
    p.verbose = False
    p.fluxConvert = False
    try:
        yield verb
    finally:
        p.verbose = verb
        p.fluxConvert = oldfc


def radialProfile(c, rings, normtype):
    """Radial profile of the data in the rings (Ellprof, within the mask). For AZIM (and
    BOTH) the DENS of the rings is set (in place) to the face-on surface density of the
    data, scaled by a power of ten to between 0.1 and 10. Returns (ell, scaling factor)."""
    ell = Ellprof(c, rings, True)
    ell.RadialProfile()
    scaling_factor = 1.
    if normtype in ("AZIM", "BOTH"):
        profmin = FLT_MAX
        for i in range(rings.nr):
            mean = ell.getSurfDensFaceOn(i)
            if not np.isnan(mean) and profmin > mean > 0:
                profmin = mean
        scaling_factor = np.float32(1.)
        while profmin < 0.1:
            profmin *= 10
            scaling_factor *= 10
        while profmin > 10:
            profmin /= 10
            scaling_factor /= 10
        scaling_factor = float(scaling_factor)
        for i in range(rings.nr):
            if rings.dens[i] > 0:
                rings.dens[i] = scaling_factor * abs(ell.getSurfDensFaceOn(i))
    return ell, scaling_factor


def _model(c, model, rings):
    """The final model (a Cube named <name>mod) and its values as float64."""
    mod = model.cube(rings.copy(), final=True)
    mod.Head().setMinMax(0., 0.)
    mod.Head().setName(c.Head().Name() + "mod")
    return mod, mod.array.astype(np.float64)


def _local(out, c, mask):
    """LOCAL normalisation: each spectrum of the model scaled to the flux of the data in the mask."""
    data = c.array.astype(np.float64)
    modSum = out.sum(axis=0)
    obsSum = (data * mask).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        factor = np.where(modSum != 0, obsSum / modSum, 0.)
    return (out * factor[None]).astype(np.float32)


def finalModel(c, model, rings, normtype, mask=None, par=None, seed=None):
    """The final model of writeModel in memory (no files written): the model of the rings
    (DiskModel model, with its ring convention) normalised as normtype (AZIM, LOCAL or
    NONE; BOTH gives the LOCAL model of writeModel; AZIM changes the DENS of the rings in
    place), plus noise if par.NOISERMS is not 0 (seed: seed of the noise). mask: the mask
    (needed by LOCAL and AZIM). Returns a Cube."""
    p = c.pars()
    par = p.gf if par is None else par
    with _outputs_mode(p):
        sf = 1.
        if normtype in ("AZIM", "BOTH"):
            _, sf = radialProfile(c, rings, normtype)
        mod, out = _model(c, model, rings)
        if normtype == "AZIM":
            mod.array = (out / sf).astype(np.float32)
        elif normtype in ("LOCAL", "BOTH"):
            mod.array = _local(out, c, mask)
        if par.NOISERMS != 0:
            mod.array = model.addnoise(mod.array, par.NOISERMS, seed=seed)
    return mod


def writeModel(c, model, rings, normtype, mask, par=None, makeplots=False, fit=None):
    """Write the final model and the outputs of 3DFIT and GALMOD in the output folder.

    c: the data cube; model: the model engine (DiskModel); rings: the final rings (AZIM
    changes their DENS in place); normtype: AZIM, LOCAL, NONE or BOTH (AZIM and LOCAL);
    mask: the mask (bool array); par: the GALMOD/3DFIT parameters (NOISERMS, ADRIFT, ...;
    default c.pars().gf); fit: the Galfit object of a 3DFIT, for the plots (makeplots).

    Written: densprof.txt, <prefix>mod_<norm>.fits, the moment maps and PV slices of the
    model, the data and the mask, the noisy model (NOISERMS), asymdrift.txt (ADRIFT) and
    the plots. Returns the (last) model Cube."""
    p = c.pars()
    par = p.gf if par is None else par
    outfold, prefix = p.outFolder, p.outPrefix
    # Start loading the plotting libraries in a helper process while the outputs are written
    runner = None
    if makeplots and fit is not None and p.plots:
        from .plotscripts import PlotRunner
        runner = PlotRunner()
    with _outputs_mode(p) as verb:
        if verb:
            print(" Preparing a bunch of cool outputs...")
            print(f"    Deriving {randomAdjective(1)} radial profile...", end="", flush=True)
        ell, scaling_factor = radialProfile(c, rings, normtype)
        with open(outfold + "densprof.txt", "w") as f:
            ell.printProfile(f, ell.getNseg() - 1)
        if verb:
            print(" Done.")
            print("    Calculating the very last model...", end="", flush=True)
        mod, out = _model(c, model, rings)
        if normtype in ("AZIM", "BOTH"):
            mod.array = (out / scaling_factor).astype(np.float32)
            if verb:
                print(" Done.")
                print(f"    Writing {randomAdjective(1)} azimuthally-normalized model...", end="", flush=True)
            mod.fitswrite_3d(outfold + prefix + "mod_azim.fits")
            writePVs(c, mod, rings, mask, "_azim")
            if verb:
                print(" Done.")
                print(f"    Writing {randomAdjective(2)} kinematic maps...", end="", flush=True)
            writeKinematicMaps(c, mod, "_azim")
            if verb:
                print(" Done.")
        if normtype in ("LOCAL", "BOTH"):
            mod.array = _local(out, c, mask)
            if verb and normtype != "BOTH":
                print(" Done.")
            if verb:
                print(f"    Writing {randomAdjective(1)} locally-normalized model...", end="", flush=True)
            mod.fitswrite_3d(outfold + prefix + "mod_local.fits")
            writePVs(c, mod, rings, mask, "_local")
            if verb:
                print(" Done.")
                s = "    Writing " + ("even more " if normtype == "BOTH" else "")
                print(s + f"{randomAdjective(2)} kinematic maps...", end="", flush=True)
            writeKinematicMaps(c, mod, "_local")
            if verb:
                print(" Done.")
        if normtype == "NONE":
            if verb:
                print(" Done.")
                print("    Writing model...", end="", flush=True)
            mod.fitswrite_3d(outfold + prefix + "mod_none.fits")
            writePVs(c, mod, rings, mask, "_none")
            if verb:
                print(" Done.")
                print(f"    Writing {randomAdjective(1)} kinematic maps...", end="", flush=True)
            writeKinematicMaps(c, mod, "_none")
            if verb:
                print(" Done.")
        if par.NOISERMS != 0:
            if verb:
                print(f"    Writing {randomAdjective(1)} noisy model...", end="", flush=True)
            mod.array = model.addnoise(mod.array, par.NOISERMS)
            mod.fitswrite_3d(outfold + prefix + "mod_noise.fits")
            if verb:
                print(" Done.")
        if par.flagADRIFT:
            if verb:
                print("    Computing asymmetric drift correction...", end="", flush=True)
            strad = par.STARTRAD if par.STARTRAD < rings.nr else 0
            dens_m = np.array([ell.getMedian(i) for i in range(rings.nr)])
            ok = asymmetricDrift(c, par, rings.radii[strad:], dens_m[strad:], rings.vdisp[strad:],
                                 rings.vrot[strad:], rings.inc[strad:])
            if verb:
                print(" Done." if ok else " Failed.")
    if makeplots and fit is not None:
        if verb:
            print(f"    Producing {randomAdjective(1)} plots...", end="", flush=True)
        ret = fit.plotAll_Python(runner)
        if verb:
            print(" Done." if ret == 0 else " Something went wrong! Check plotting scripts in the output folder.")
    if verb:
        print(" All done!")
    return mod


def writeOutputs(c, mod, ell, rings, mask, makeplots=True, fit=None):
    """Write a model computed elsewhere (a Cube, e.g. the best model of BayesianBBarolo)
    with its outputs: densprof.txt (from the Ellprof ell), the model, its moment maps and
    PV slices, and the plots of the fit."""
    p = c.pars()
    suffix = "_none"
    outfold, prefix = p.outFolder, p.outPrefix
    with open(outfold + "densprof.txt", "w") as f:
        ell.printProfile(f, ell.getNseg() - 1)
    mod.Head().setMinMax(0., 0.)
    mod.Head().setName(prefix + "mod")
    mod.fitswrite_3d(outfold + prefix + "mod" + suffix + ".fits")
    writePVs(c, mod, rings, mask, suffix)
    writeKinematicMaps(c, mod, suffix)
    if makeplots and fit is not None:
        fit.plotAll_Python()


# --------------------------------------------------------------------------
# Asymmetric drift
# --------------------------------------------------------------------------
def asymmetricDrift(c, par, rad, densprof, dispprof, rotcur, inc):
    """Asymmetric drift correction from the density and dispersion profiles (regularised
    with polynomials of degree ADRIFTPOL1 and ADRIFTPOL2), written in asymdrift.txt.
    Returns False (and sets par.flagADRIFT = False) if a polynomial fit fails."""
    p = c.pars()
    nn = len(rad)
    rad = np.asarray(rad, float)
    npar1 = par.ADRIFTPOL1 + 1
    disp2 = np.asarray(dispprof, float) ** 2
    cdisp = None
    if npar1 > 0:
        if npar1 >= nn: npar1 = nn - 1
        cdisp, _, ok = polyfit_lsq(rad, disp2, npar1)
        if not ok:
            if p.verbose: sys.stderr.write("3DFIT ERROR: cannot least-square fit the dispersion for asymmetric drift.\n")
            par.flagADRIFT = False
            return False
    SigmaFO = np.asarray(densprof, float) * np.cos(np.radians(inc))
    npar2 = par.ADRIFTPOL2 + 1
    if npar2 >= nn: npar2 = nn - 1
    cfun, _, ok = polyfit_lsq(rad, SigmaFO, npar2)
    if not ok:
        if p.verbose: sys.stderr.write("3DFIT ERROR: cannot least-square fit the Sigma for asymmetric drift.\n")
        par.flagADRIFT = False
        return False
    P = np.polynomial.polynomial
    m = 20
    with open(p.outFolder + "asymdrift.txt", "w") as f:
        f.write("#" + f"{'RAD(arcs)':>{m - 1}}" + "".join(f"{s:>{m}}" for s in
                ("VCIRC(km/s)", "ASYMDRIFT_SQ(km/s)2", "DISP2(km/s)2", "DISP2_REG(km/s)2", "DENSPROF", "DENSPROF_REG")) + "\n")
        for i in range(nn):
            d2r = P.polyval(rad[i], cdisp) if npar1 > 0 else disp2[i]
            Sr = P.polyval(rad[i], cfun)
            d2d = P.polyval(rad[i], P.polyder(cdisp)) if (npar1 > 0 and len(cdisp) > 1) else 0.
            Sd = P.polyval(rad[i], P.polyder(cfun)) if len(cfun) > 1 else 0.
            ad2 = -rad[i] * d2r * (d2d / d2r + Sd / Sr)
            vc = math.sqrt(rotcur[i] ** 2 + ad2) if rotcur[i] ** 2 + ad2 >= 0 else float("nan")
            f.write("".join(f"{v:>{m}.4f}" for v in (rad[i], vc, ad2, disp2[i], d2r, SigmaFO[i], Sr)) + "\n")
    return True
