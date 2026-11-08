"""
Outputs of the 3DFIT and GALMOD tasks (port of Tasks/galfit_out.cpp: writeModel,
writeOutputs, writeKinematicMaps, writePVs). Plotting scripts are in plotscripts.py.

The functions take the data cube, the model engine (galmod.DiskModel), the rings and
the mask, so that 3DFIT (through its Galfit object) and GALMOD (without any fit) share
them. model_outputs() computes the final model and its outputs in memory (a ModelOutputs),
ModelOutputs.write() writes them; finalModel() gives the final model only.
"""

import sys
import warnings
import math
import contextlib
import numpy as np

from .ellprof import Ellprof
from .moments import getAllMoments
from .pvslice import PositionVelocity
from .stats import findMedian
from .bbutils import mkdirp, randomAdjective, polyfit_lsq

FLT_MAX = float(np.finfo(np.float32).max)


def kinematic_maps(c, mod, mask=None, suffix=""):
    """The moment maps of the data (within the mask: mask, or that of the cube) and of the model mod, as HDUs:
    {'0mom', '1mom', '2mom', 'mod_0mom<suffix>', ...} (the files of the maps/ folder without the prefix)."""
    p = c.pars()
    v = p.verbose
    p.verbose = False
    try:
        allmaps = getAllMoments(c, True, mask, "MOMENT")
        out = {s: allmaps[i].to_hdu() for i, s in enumerate(("0mom", "1mom", "2mom"))}
        mod.pars().verbose = False
        modmaps = getAllMoments(mod, False, None, "MOMENT")
        out.update({s + suffix: modmaps[i].to_hdu() for i, s in enumerate(("mod_0mom", "mod_1mom", "mod_2mom"))})
    finally:
        p.verbose = v
    return out


def pv_slices(c, mod, rings, mask, suffix=""):
    """PV slices of the data, of the model and of the mask along the major and minor axis (median position
    angle and centre of the rings), as HDUs: {'pv_a', 'pv_b', 'mod_pv_a<suffix>', 'mod_pv_b<suffix>',
    'mask_pv_a', 'mask_pv_b'} (the files of the pvs/ folder without the prefix)."""
    from .cube import Cube
    p = c.pars()
    o = rings
    meanPA = float(np.float32(findMedian(o.phi)))
    meanX = float(np.float32(findMedian(o.xpos)))
    meanY = float(np.float32(findMedian(o.ypos)))
    meanPAp90 = meanPA + 90 if meanPA + 90 < 360 else meanPA - 90
    out = {"pv_a": PositionVelocity(c, meanX, meanY, meanPA).to_hdu(),
           "pv_b": PositionVelocity(c, meanX, meanY, meanPAp90).to_hdu(),
           "mod_pv_a" + suffix: PositionVelocity(mod, meanX, meanY, meanPA).to_hdu(),
           "mod_pv_b" + suffix: PositionVelocity(mod, meanX, meanY, meanPAp90).to_hdu()}
    m = Cube.fromArray(mask.astype(np.int16), c.Head(), p)
    m.Head().setMinMax(0., 0)
    for k, pa in (("mask_pv_a", meanPA), ("mask_pv_b", meanPAp90)):
        pv = PositionVelocity(m, meanX, meanY, pa, order=0)        # nearest pixel for the mask
        pv.array = pv.array.astype(np.int16)
        out[k] = pv.to_hdu()
    return out


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


class ModelOutputs:
    """The outputs of the final model of 3DFIT, GALMOD and BAYESIAN3DFIT, in memory.

    mask: the mask (bool array); ell: the radial profile of the data (an Ellprof; densprof() as a Table);
    both can be None, and are then computed when needed (the mask of the cube, built if needed, and the
    radial profile in the rings); models: {norm: model Cube} (azim, local or none, in the order they are
    written); noise: the noisy model (a Cube, NOISERMS) or None; adrift: the text of <prefix>_asymdrift.txt
    (ADRIFT) or None. The moment maps (maps()) and the PV slices (pvs()) of the data, the models and the
    mask are computed when first asked for. write() writes them all, as the command line does."""

    def __init__(self, c, rings, mask, ell, models, noise=None, adrift=None, fluxconv=False):
        self.c = c
        self.rings = rings.copy()
        self._mask = mask
        self._ell = ell
        self.models = models
        self.noise = noise
        self.adrift = adrift
        self._fluxconv = fluxconv                          # (maps with the fluxes converted, as FLUXCONVERT)
        self._maps = self._pvs = None

    @property
    def mask(self):
        if self._mask is None:
            c = self.c
            if not c.MaskAll():
                with _outputs_mode(c.pars()):
                    c.BlankMask()
            self._mask = c.Mask()
        return self._mask

    @property
    def cmask(self):
        """The mask of the data maps: that of the cube."""
        return self.mask

    @property
    def ell(self):
        if self._ell is None:
            self.mask
            with _outputs_mode(self.c.pars()):
                self._ell = radialProfile(self.c, self.rings.copy(), "NONE")[0]
        return self._ell

    def densprof_text(self):
        import io
        buf = io.StringIO()
        self.ell.printProfile(buf, self.ell.getNseg() - 1)
        return buf.getvalue()

    def densprof(self):
        """The radial profile of the data in the rings (<prefix>_densprof.txt) as an astropy Table."""
        import io
        from .tasks import read_table
        return read_table(io.StringIO(self.densprof_text()))

    def asymdrift(self):
        """The asymmetric drift correction (<prefix>_asymdrift.txt) as an astropy Table, or None."""
        import io
        from .tasks import read_table
        return read_table(io.StringIO(self.adrift)) if self.adrift else None

    def model_hdus(self):
        """{norm: HDU} of the models, with 'noise' for the noisy model."""
        out = {k: m.to_hdu() for k, m in self.models.items()}
        if self.noise is not None:
            out["noise"] = self.noise.to_hdu()
        return out

    def _context(self):
        return contextlib.nullcontext() if self._fluxconv else _outputs_mode(self.c.pars())

    def maps(self):
        """The moment maps of the data and of each model: {'0mom', ..., 'mod_0mom_<norm>', ...} of HDUs."""
        if self._maps is None:
            with self._context():
                self._maps = {}
                for k, mod in self.models.items():
                    self._maps.update(kinematic_maps(self.c, mod, self.cmask, "_" + k))
        return self._maps

    def pvs(self):
        """The PV slices of the data, of each model and of the mask: {'pv_a', ..., 'mod_pv_a_<norm>', ...,
        'mask_pv_a', ...} of HDUs."""
        if self._pvs is None:
            with self._context():
                self._pvs = {}
                for k, mod in self.models.items():
                    self._pvs.update(pv_slices(self.c, mod, self.rings, self.mask, "_" + k))
        return self._pvs

    def write(self, makeplots=False, fit=None, runner=None):
        """Write the outputs in the output folder of the data cube: <prefix>_densprof.txt, the models
        (<prefix>_mod_<norm>.fits), their PV slices (pvs/) and moment maps (maps/), the noisy model, the
        asymmetric drift and, with makeplots, the plots of the fit (fit: its Galfit object)."""
        from .cube import write_hdu
        p = self.c.pars()
        of, pre = p.outFolder, p.outPrefix
        verb = p.verbose
        mkdirp(of)
        self.mask
        if p.plots >= 0:
            self.c.writeMask()                # (the plots read it; the mask may have been built only now)
        with open(of + pre + "_densprof.txt", "w") as f:
            f.write(self.densprof_text())
        if verb:
            print(f"    Writing {randomAdjective(1)} models, kinematic maps and PV slices...", end="", flush=True)
        maps, pvs = self.maps(), self.pvs()
        for k, mod in self.models.items():
            write_hdu(mod.to_hdu(), of + pre + f"_mod_{k}.fits")
            for name in ("pv_a", "pv_b", f"mod_pv_a_{k}", f"mod_pv_b_{k}", "mask_pv_a", "mask_pv_b"):
                write_hdu(pvs[name], of + "pvs/" + pre + "_" + name + ".fits")
            for name in ("0mom", "1mom", "2mom", f"mod_0mom_{k}", f"mod_1mom_{k}", f"mod_2mom_{k}"):
                write_hdu(maps[name], of + "maps/" + pre + "_" + name + ".fits")
        if self.noise is not None:
            write_hdu(self.noise.to_hdu(), of + pre + "_mod_noise.fits")
        if self.adrift:
            with open(of + pre + "_asymdrift.txt", "w") as f:
                f.write(self.adrift)
        if verb:
            print(" Done.")
        if makeplots and fit is not None:
            if verb:
                print(f"    Producing {randomAdjective(1)} plots...", end="", flush=True)
            ret = fit.plotAll_Python(runner)
            if verb:
                print(" Done." if ret == 0 else " Something went wrong! Check plotting scripts in the output folder.")


def model_outputs(c, model, rings, normtype, mask, par=None, fit=None, seed=None):
    """The final model and the outputs of 3DFIT and GALMOD, in memory (a ModelOutputs; nothing is written).

    c: the data cube; model: the model engine (DiskModel); rings: the final rings (AZIM changes their DENS in
    place); normtype: AZIM, LOCAL, NONE or BOTH (AZIM and LOCAL); mask: the mask (bool array; None: built
    when needed, NONE only); par: the
    GALMOD/3DFIT parameters (NOISERMS, ADRIFT, ...; default c.pars().gf); fit: the Galfit object of a 3DFIT
    (its rings fitted, for the asymmetric drift); seed: the seed of the noise of the noisy model (NOISERMS)."""
    p = c.pars()
    par = p.gf if par is None else par
    with _outputs_mode(p) as verb:
        if verb:
            print(" Preparing a bunch of cool outputs...")
        ell, scaling_factor = None, 1.
        if normtype != "NONE" or mask is not None or par.flagADRIFT:
            if verb:
                print(f"    Deriving {randomAdjective(1)} radial profile...", end="", flush=True)
            ell, scaling_factor = radialProfile(c, rings, normtype)
            if verb:
                print(" Done.")
        if verb:
            print("    Calculating the very last model...", end="", flush=True)
        mod, out = _model(c, model, rings)
        models = {}
        if normtype in ("AZIM", "BOTH"):
            mod.array = (out / scaling_factor).astype(np.float32)
            models["azim"] = mod
        if normtype in ("LOCAL", "BOTH"):
            m = mod.copy() if models else mod
            m.array = _local(out, c, mask)
            models["local"] = m
        if normtype == "NONE":
            models["none"] = mod
        noise = None
        if par.NOISERMS != 0:
            last = list(models.values())[-1]
            noise = last.copy()
            noise.array = model.addnoise(last.array, par.NOISERMS, seed=seed)
        if verb:
            print(" Done.")
        adrift = None
        if par.flagADRIFT:
            if verb:
                print("    Computing asymmetric drift correction...", end="", flush=True)
            strad = par.STARTRAD if par.STARTRAD < rings.nr else 0
            dens_m = np.array([ell.getMedian(i) for i in range(rings.nr)])
            # the rings fitted, from STARTRAD (C++ used also those that were not fitted)
            sel = np.arange(rings.nr) >= strad
            if getattr(fit, "fitok", None) is not None:
                sel &= np.asarray(fit.fitok, bool)
            if sel.sum() >= 2:
                adrift = asymmetricDrift(c, par, rings.radii[sel], dens_m[sel], rings.vdisp[sel], rings.vrot[sel],
                                         rings.inc[sel])
            if verb:
                print(" Done." if adrift else " Failed.")
    return ModelOutputs(c, rings, mask, ell, models, noise, adrift or None)


def writeModel(c, model, rings, normtype, mask, par=None, makeplots=False, fit=None):
    """Compute (model_outputs) and write (ModelOutputs.write) the final model and the outputs of 3DFIT and
    GALMOD in the output folder. Returns the ModelOutputs."""
    p = c.pars()
    # Start loading the plotting libraries in a helper process while the outputs are computed
    runner = None
    if makeplots and fit is not None and p.plots:
        from .plotscripts import PlotRunner
        runner = PlotRunner()
    mo = model_outputs(c, model, rings, normtype, mask, par=par, fit=fit)
    mo.write(makeplots, fit, runner)
    if p.verbose:
        print(" All done!")
    return mo


def writeOutputs(c, mod, ell, rings, mask, makeplots=True, fit=None):
    """Write a model computed elsewhere (a Cube, e.g. the best model of BayesianBBarolo)
    with its outputs: densprof.txt (from the Ellprof ell), the model, its moment maps and
    PV slices, and the plots of the fit. Returns the ModelOutputs."""
    p = c.pars()
    mod.Head().setMinMax(0., 0.)
    mod.Head().setName(p.outPrefix + "mod")
    mo = ModelOutputs(c, rings, mask, ell, {"none": mod}, fluxconv=True)
    mo.write(makeplots, fit)
    return mo


# --------------------------------------------------------------------------
# Asymmetric drift
# --------------------------------------------------------------------------
def asymmetricDrift(c, par, rad, densprof, dispprof, rotcur, inc):
    """Asymmetric drift correction from the density and dispersion profiles (regularised
    with polynomials of degree ADRIFTPOL1 and ADRIFTPOL2): the text of <prefix>_asymdrift.txt.
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
            warnings.warn("3DFIT: cannot fit the dispersion profile for the asymmetric drift correction (ADRIFT): "
                          "<prefix>_asymdrift.txt is not written.")
            par.flagADRIFT = False
            return False
    SigmaFO = np.asarray(densprof, float) * np.cos(np.radians(inc))
    npar2 = par.ADRIFTPOL2 + 1
    if npar2 >= nn: npar2 = nn - 1
    cfun, _, ok = polyfit_lsq(rad, SigmaFO, npar2)
    if not ok:
        warnings.warn("3DFIT: cannot fit the Sigma profile for the asymmetric drift correction (ADRIFT): "
                      "<prefix>_asymdrift.txt is not written.")
        par.flagADRIFT = False
        return False
    P = np.polynomial.polynomial
    m = 20
    import io
    with io.StringIO() as f:
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
        return f.getvalue()
