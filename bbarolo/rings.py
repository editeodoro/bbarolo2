"""
Tilted rings (port of Arrays/rings.hh and readRings in utils.cpp).
"""

import numpy as np

from .bbutils import getDataColumn, getCenterCoordinates

RING_KEYS = ("radii", "xpos", "ypos", "vsys", "vrot", "vdisp", "vrad", "vvert",
             "dvdz", "zcyl", "dens", "z0", "inc", "phi")


class Rings:
    """A set of tilted rings. Each parameter is a float64 numpy array of length nr.

    edge (arcsec, default None): the outer edge of the disk of the model, within the rings: the
    density of the model beyond it is zero (edgewidth = 0: a sharp edge) or falls as
    1/2 erfc((R - edge) / (sqrt(2) edgewidth)) (edgewidth > 0, arcsec: a Gaussian edge). See
    galmod.edge_weights. Used by BayesianBBarolo for RADMAX."""

    def __init__(self, nr=0):
        self.radsep = 0.            # (as in C++; -1 shrank SHIFTED disks by 0.5)
        self.id = 0
        self.edge, self.edgewidth = None, 0.
        for k in RING_KEYS:
            setattr(self, k, np.zeros(nr))

    @property
    def nr(self):
        return len(self.radii)

    def copy(self):
        r = Rings()
        r.radsep, r.id = self.radsep, self.id
        r.edge, r.edgewidth = self.edge, self.edgewidth
        for k in RING_KEYS:
            setattr(r, k, np.array(getattr(self, k), dtype=float, copy=True))
        return r

    def addRing(self, radii, xpos, ypos, vsys, vrot, vdisp, vrad, vvert, dvdz, zcyl, dens, z0, inc, phi):
        vals = (radii, xpos, ypos, vsys, vrot, vdisp, vrad, vvert, dvdz, zcyl, dens, z0, inc, phi)
        for k, v in zip(RING_KEYS, vals):
            setattr(self, k, np.append(getattr(self, k), float(v)))

    def setRings(self, **kw):
        """Set all ring arrays: scalars are broadcast to the length of radii."""
        radii = np.atleast_1d(np.asarray(kw["radii"], dtype=float))
        n = len(radii)
        dflt = dict(xpos=0, ypos=0, vsys=0, vrot=0, vdisp=0, vrad=0, vvert=0, dvdz=0, zcyl=0,
                    dens=1, z0=0, inc=0, phi=0)
        self.radii = radii.copy()
        for k in RING_KEYS[1:]:
            v = kw.get(k, dflt[k])
            v = np.asarray(v, dtype=float)
            self.__dict__[k] = np.full(n, float(v)) if v.ndim == 0 else v.astype(float).copy()
        if "radsep" in kw:
            self.radsep = kw["radsep"]
        else:
            self.radsep = radii[1] - radii[0] if n > 1 else 0
        return self

    def subset(self, idx):
        r = Rings()
        r.radsep, r.id = self.radsep, self.id
        r.edge, r.edgewidth = self.edge, self.edgewidth
        for k in RING_KEYS:
            setattr(r, k, np.atleast_1d(np.asarray(getattr(self, k))[idx]).astype(float))
        return r

    def deleteRing(self, n):
        for k in RING_KEYS:
            setattr(self, k, np.delete(getattr(self, k), n))

    def printRing(self, n, stream=None):
        s = (f"\n Ring #{n + 1} (idx={n}) at radius {self.radii[n]} arcsec \n"
             f"    Vrot   = {self.vrot[n]:8.2f}  km/s          Disp   = {self.vdisp[n]:7.2f}  km/s\n"
             f"    Vrad   = {self.vrad[n]:8.2f}  km/s          Vsys   = {self.vsys[n]:7.2f}  km/s\n"
             f"    Inc    = {self.inc[n]:8.2f}  deg           PA     = {self.phi[n]:7.2f}  deg\n"
             f"    Xpos   = {self.xpos[n]:8.2f}  pix           Ypos   = {self.ypos[n]:7.2f}  pix\n"
             f"    Z0     = {self.z0[n]:8.2f}  arcs          Dens   = {self.dens[n]:7.2f} 1E20 1/cm2\n"
             f"    VVERT  = {self.vvert[n]:8.2f}  km/s          ZCYL   = {self.zcyl[n]:7.2f} arcs\n"
             f"    DVDZ   = {self.dvdz[n]:8.2f}  km/s/arcs\n")
        if stream is None:
            print(s, end="")
        else:
            stream.write(s)

    def __str__(self):
        import io
        s = io.StringIO()
        for i in range(self.nr):
            self.printRing(i, s)
        return s.getvalue()

    def as_dict(self):
        return {k: getattr(self, k).copy() for k in RING_KEYS}


def ringList(s):
    """The values of a ring parameter given as a list of numbers ('120 125 130': one
    value per ring), or None if s is a single value (or anything else)."""
    t = str(s).split()
    if len(t) < 2:
        return None
    try:
        return [float(x) for x in t]
    except ValueError:
        return None


def firstValue(s):
    """The first value of a ring parameter given as a list of numbers, else s itself."""
    lst = ringList(s)
    return repr(lst[0]) if lst is not None else s


def readRings(par, h, return_fromfile=False):
    """Build a Rings object from a GALFIT/GALMOD parameter structure (port of readRings)."""
    fr = {}
    for key, pkey in (("radii", "RADII"), ("xpos", "XPOS"), ("ypos", "YPOS"), ("vsys", "VSYS"),
                      ("vrot", "VROT"), ("vrad", "VRAD"), ("vvert", "VVERT"), ("dvdz", "DVDZ"),
                      ("zcyl", "ZCYL"), ("vdisp", "VDISP"), ("z0", "Z0"), ("dens", "DENS"),
                      ("inc", "INC"), ("phi", "PHI")):
        val = str(getattr(par, pkey))
        fr[key] = getDataColumn(val) if "file(" in val.lower() else None
    fromfile = any(v is not None for v in fr.values())
    for key, pkey in (("xpos", "XPOS"), ("ypos", "YPOS"), ("vsys", "VSYS"), ("vrot", "VROT"), ("vrad", "VRAD"),
                      ("vvert", "VVERT"), ("dvdz", "DVDZ"), ("zcyl", "ZCYL"), ("vdisp", "VDISP"), ("z0", "Z0"),
                      ("dens", "DENS"), ("inc", "INC"), ("phi", "PHI")):
        if fr[key] is None:
            fr[key] = ringList(getattr(par, pkey))      # one value per ring
    radii_b = fr["radii"] is not None

    if not radii_b and str(par.RADII) != "-1":
        s = str(par.RADII).split()
        if len(s) == 1 and "~" in s[0] and ":" in s[0]:
            f1, f2 = s[0].find("~"), s[0].find(":")
            Rmin, Rmax, step = float(s[0][:f1]), float(s[0][f1 + 1:f2]), float(s[0][f2 + 1:])
            if Rmin > Rmax or step < 0:
                raise ValueError(' INPUT ERROR: RADII parameter format is "Rmin~Rmax:step".')
            # Rmin, Rmin+step, ... up to Rmax included (a tolerance for steps such as 0.1, which
            # are not exact in binary: C++ sums floats, and 0~1:0.1 gave 10 or 11 radii)
            n = int(np.floor((Rmax - Rmin) / step + 1e-6)) + 1 if step > 0 else 1
            fr["radii"] = list(Rmin + np.arange(n) * step)
        else:
            fr["radii"] = [float(x) for x in s]
        radii_b = len(fr["radii"]) > 0

    sizes = [len(v) for v in fr.values() if v is not None and len(v) > 0]
    max_size = min(sizes) if sizes else 2 ** 31

    def pval(s, dflt):
        s = str(s)
        if s == "-1" or "file(" in s.lower():
            return dflt
        try:
            return float(s.split()[0])
        except ValueError:
            return dflt

    defaults = dict(vsys=pval(par.VSYS, 0), vrot=pval(par.VROT, 0), vrad=pval(par.VRAD, 0),
                    vvert=pval(par.VVERT, 0), dvdz=pval(par.DVDZ, 0), zcyl=pval(par.ZCYL, 0),
                    vdisp=pval(par.VDISP, 0), z0=pval(par.Z0, 0), dens=pval(par.DENS, 1),
                    inc=pval(par.INC, 0), phi=pval(par.PHI, 0))
    xpos = ypos = 0.
    if str(par.XPOS) != "-1" and str(par.YPOS) != "-1" and fr["xpos"] is None and fr["ypos"] is None:
        xpos, ypos = getCenterCoordinates([par.XPOS, par.YPOS], h)
    elif str(par.XPOS) != "-1" and str(par.YPOS) != "-1":
        try:
            xpos, ypos = getCenterCoordinates([par.XPOS, par.YPOS], h)
        except Exception:
            pass
    defaults["xpos"], defaults["ypos"] = xpos, ypos

    nr = par.NRADII
    radsep = par.RADSEP
    nr = nr if (0 < nr < max_size) else max_size
    if radii_b:
        rr = fr["radii"]
        if nr == 1:
            if radsep == -1:
                radsep = rr[0] / 2.
        else:
            radsep = float(np.mean(np.diff(rr[:]))) if len(rr) > 1 else radsep
    if nr >= 2 ** 31:
        nr = 0

    r = Rings(nr)
    r.radsep = radsep
    for k in RING_KEYS:
        if k == "radii":
            vals = np.asarray(fr["radii"][:nr], float) if radii_b else np.arange(nr) * radsep + radsep / 2.
        else:
            vals = np.asarray(fr[k][:nr], float) if fr[k] is not None else np.full(nr, float(defaults[k]))
        setattr(r, k, vals)
    if return_fromfile:
        return r, fromfile
    return r
