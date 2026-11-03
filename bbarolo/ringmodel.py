"""
2D tilted-ring fitting of a velocity field (port of Tasks/ringmodel.cpp, i.e. GIPSY's ROTCUR).
"""

import sys
import math
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from scipy.optimize import least_squares

from .bbutils import arcsconv, ProgressBar

VSYS, VROT, VEXP, PA, INC, X0, Y0, MAXPAR = range(8)
FLT_MAX = np.finfo(np.float32).max


def _nint(x):
    return int(x + 0.5) if x > 0.0 else int(x - 0.5)


def vfunc(c, p):
    """Line-of-sight velocity at sky positions c (n,2) for ring parameters p."""
    F = math.pi / 180.
    sinp, cosp = math.sin(F * p[PA]), math.cos(F * p[PA])
    sini, cosi = math.sin(F * p[INC]), math.cos(F * p[INC])
    x = c[:, 0] - p[X0]
    y = c[:, 1] - p[Y0]
    x1 = -x * sinp + y * cosp
    y1 = (-x * cosp - y * sinp) / cosi
    r = np.sqrt(x1 * x1 + y1 * y1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cost1 = x1 / r
        sint1 = y1 / r
    return p[VSYS] + (p[VROT] * cost1 + p[VEXP] * sint1) * sini


def vderv(c, p):
    F = math.pi / 180.
    vc, vr = p[VROT], p[VEXP]
    sinp1, cosp1 = math.sin(F * p[PA]), math.cos(F * p[PA])
    cosi1, sini1 = math.cos(F * p[INC]), math.sin(F * p[INC])
    cosi2, sini2 = cosi1 * cosi1, sini1 * sini1
    x = c[:, 0] - p[X0]
    y = c[:, 1] - p[Y0]
    x1 = -x * sinp1 + y * cosp1
    y1 = (-x * cosp1 - y * sinp1) / cosi1
    r = np.sqrt(x1 * x1 + y1 * y1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cost1, sint1 = x1 / r, y1 / r
    cost2, sint2 = cost1 * cost1, sint1 * sint1
    d = np.empty((c.shape[0], MAXPAR))
    d[:, VSYS] = 1.0
    d[:, VROT] = sini1 * cost1
    d[:, VEXP] = sini1 * sint1
    d[:, PA] = (F * vc * (1.0 - sini2 * sint2) * sint1 * sini1 / cosi1
                - F * vr * (1.0 - sini2 * cost2) * cost1 * sini1 / cosi1)
    d[:, INC] = (F * vc * (cosi2 - sini2 * sint2) * cost1 / cosi1
                 + F * vr * (cosi2 + sini2 * cost2) * sint1 / cosi1)
    with np.errstate(invalid="ignore", divide="ignore"):
        d[:, X0] = (vc * (sint1 * sinp1 - cost1 * cosp1 / cosi1) * sint1 * sini1 / r
                    - vr * (sint1 * sinp1 - cost1 * cosp1 / cosi1) * cost1 * sini1 / r)
        d[:, Y0] = (-vc * (sint1 * cosp1 + cost1 * sinp1 / cosi1) * sint1 * sini1 / r
                    + vr * (sint1 * cosp1 + cost1 * sinp1 / cosi1) * cost1 * sini1 / r)
    return d


def lsqfit(X, y, w, p, mask, tol, maxiter):
    """Weighted least-squares fit of vfunc to the velocities y at positions X, varying the
    parameters p with mask=True (scipy's trust-region least squares with the analytic
    derivatives of vderv). Returns (code, parameters, errors): code > 0 on success,
    otherwise one of GIPSY lsqfit's error codes (-2 no free parameters, -3 not enough
    points, -4 too many iterations, -6 singular matrix). The errors are
    sqrt(diag((J^T W J)^-1) * chi2 / (n - nfree)), as in GIPSY's lsqfit."""
    free = np.flatnonzero(mask)
    good = w > 0
    nuse, nfree = int(good.sum()), free.size
    zero = np.zeros_like(p)
    if nfree == 0:
        return -2, p, zero
    if nfree >= nuse:
        return -3, p, zero
    Xg, yg, sw = X[good], y[good], np.sqrt(w[good])

    def full(q):
        pp = p.copy()
        pp[free] = q
        return pp

    def resid(q):
        return sw * (vfunc(Xg, full(q)) - yg)

    def jac(q):
        return sw[:, None] * vderv(Xg, full(q))[:, free]
    try:
        with np.errstate(invalid="ignore", divide="ignore"):
            r = least_squares(resid, p[free], jac=jac, method="trf", ftol=tol, max_nfev=maxiter * (nfree + 1))
    except (ValueError, np.linalg.LinAlgError):
        return -6, p, zero
    if r.status <= 0 or not np.all(np.isfinite(r.x)):
        return -4, p, zero
    J = r.jac
    JTJ = J.T @ J
    if np.any(np.diag(JTJ) <= 0):
        return -5, p, zero
    try:
        cov = np.linalg.inv(JTJ)
    except np.linalg.LinAlgError:
        return -6, p, zero
    d = np.diag(cov)
    if np.any(d <= 0) or not np.all(np.isfinite(d)):
        return -7, p, zero
    err = zero.copy()
    err[free] = np.sqrt(d * 2 * r.cost / (nuse - nfree))
    return max(r.nfev, 1), full(r.x), err


class Ringmodel:
    """Tilted-ring model fit of a 2D velocity field."""

    def __init__(self, cube=None):
        self.tol = 0.001
        self.cor = [-1, -1]
        self.mask = np.ones(MAXPAR, bool)
        self.side = 3
        self.wpow = 2
        self.thetaf = 15.
        self.vfield = None
        if cube is not None:
            self._from_cube(cube)

    def _from_cube(self, c):
        from .paramguess import EstimateInitial
        from .rings import readRings
        import copy
        par = copy.deepcopy(c.pars().gf)
        toEstimate = ((par.RADII == "-1" and (par.NRADII == -1 or par.RADSEP == -1)) or par.XPOS == "-1"
                      or par.YPOS == "-1" or par.VSYS == "-1" or par.VROT == "-1" or par.PHI == "-1" or par.INC == "-1")
        if toEstimate:
            ip = EstimateInitial(c, par)
            if par.NRADII == -1: par.NRADII = ip.nrings
            if par.RADSEP == -1: par.RADSEP = ip.radsep
            if par.XPOS == "-1": par.XPOS = repr(float(ip.xcentre))
            if par.YPOS == "-1": par.YPOS = repr(float(ip.ycentre))
            if par.VSYS == "-1": par.VSYS = repr(float(ip.vsystem))
            if par.VROT == "-1": par.VROT = repr(float(ip.vrot))
            if par.INC == "-1": par.INC = repr(float(ip.inclin))
            if par.PHI == "-1": par.PHI = repr(float(ip.posang))
        if par.VRAD == "-1":
            par.VRAD = "0"
        r = readRings(par, c.Head())
        self.setfromCube(c, r, par)

    def set(self, nrings, radii, widths, vsys, vrot, vexp, posang, incl, xcenter, ycenter):
        def arr(v):
            v = np.asarray(v, float)
            return np.full(nrings, float(v)) if v.ndim == 0 else v[:nrings].astype(float).copy()
        self.nrad = nrings
        self.rads, self.wids = arr(radii), arr(widths)
        self.vsysi, self.vroti, self.vexpi = arr(vsys), arr(vrot), arr(vexp)
        self.posai, self.incli = arr(posang), arr(incl)
        self.xposi, self.yposi = float(xcenter), float(ycenter)
        n = nrings
        for k in ("vsys", "vrot", "vexp", "posa", "incl", "xpos", "ypos"):
            setattr(self, k + "f", np.full(n, np.nan))
            setattr(self, k + "e", np.full(n, np.nan))
        self.npts = np.zeros(n, int)
        self.chis = np.zeros(n)
        self.elp = np.zeros((n, 4))

    def setfromCube(self, c, r, p=None):
        from .moments import MomentMap
        p = c.pars().gf if p is None else p
        h = c.Head()
        nrings = r.nr - 1
        conv = arcsconv(h.Cunit(0)) * h.PixScale()
        radii = r.radii[:nrings] / conv
        widths = (r.radii[1:nrings + 1] - r.radii[:nrings]) / conv
        self.set(nrings, radii, widths, r.vsys, r.vrot, r.vrad, r.phi, r.inc, r.xpos[0], r.ypos[0])
        f = p.FREE.lower()
        m = [False] * MAXPAR
        if "vrot" in f: m[VROT] = True
        if "inc" in f: m[INC] = True
        if "pa" in f or "phi" in f: m[PA] = True
        if "xpos" in f: m[X0] = True
        if "ypos" in f: m[Y0] = True
        if "vsys" in f: m[VSYS] = True
        if "vrad" in f or "vexp" in f: m[VEXP] = True
        if "all" in f: m = [True] * MAXPAR
        hside = 1 if p.SIDE == "R" else (2 if p.SIDE == "A" else 3)
        try:
            freeang = 15. if p.DVDZ == "-1" else float(str(p.DVDZ).split()[0])
        except ValueError:
            freeang = 15.
        self.setoption(m, hside, p.WFUNC, freeang)
        mp = MomentMap()
        mp.input(c)
        if c.Head().NumAx() > 2 and c.DimZ() > 1:
            mp.FirstMoment(True)
        else:
            mp.array = c.array[0].copy()
            mp.setHead(1)
        mp.fitswrite_2d(c.pars().outFolder + c.pars().outPrefix + "map_1st.fits")
        self.setfield(mp.array, mp.DimX(), mp.DimY())

    def setoption(self, maskpar, hside, wfunc, freeangle):
        self.side = hside
        if self.side not in (1, 2, 3):
            print("Not allowed half of galaxy. Setting to both.")
            self.side = 3
        self.thetaf = freeangle
        self.wpow = wfunc
        self.mask = np.array(maskpar, bool).copy()
        if self.side != 3:
            self.mask[VSYS] = self.mask[X0] = self.mask[Y0] = False
        if not self.mask.any():
            print("NO free parameters!")

    def setfield(self, array, xsize, ysize, boxup=None, boxlow=None):
        a = np.asarray(array, float).reshape(ysize, xsize)
        if boxup is None:
            boxup, boxlow = (xsize - 1, ysize - 1), (0, 0)
        self.blo, self.bup = list(boxlow), list(boxup)
        self.vfield = a[self.blo[1]:self.bup[1] + 1, self.blo[0]:self.bup[0] + 1].copy()

    # ------------------------------------------------------------------
    def getdat(self, p, ri, ro, nfr):
        F = math.pi / 180.
        phi, inc, x0, y0 = p[PA], p[INC], p[X0], p[Y0]
        free = abs(math.sin(F * self.thetaf))
        sinp, cosp = math.sin(F * phi), math.cos(F * phi)
        sini, cosi = math.sin(F * inc), math.cos(F * inc)
        a = math.sqrt(1.0 - cosp * cosp * sini * sini)
        b = math.sqrt(1.0 - sinp * sinp * sini * sini)
        llo = max(self.blo[0], _nint(x0 - a * ro) - 1)
        lup = min(self.bup[0], _nint(x0 + a * ro) + 1)
        mlo = max(self.blo[1], _nint(y0 - b * ro) - 1)
        mup = min(self.bup[1], _nint(y0 + b * ro) + 1)
        empty = np.zeros((0, 2)), np.zeros(0), np.zeros(0)
        if llo > lup or mlo > mup:
            return 0, empty[0], empty[1], empty[2], FLT_MAX
        rx, ry = np.mgrid[llo:lup, mlo:mup]      # same pixel order as C++ (x outer loop)
        v = self.vfield[ry - self.blo[1], rx - self.blo[0]]
        # Single precision geometry, as in the C++ code (ring membership of the
        # pixels lying exactly on the ring boundaries depends on it)
        f4 = np.float32
        x0f, y0f = f4(x0), f4(y0)
        sinpf, cospf, cosif = f4(sinp), f4(cosp), f4(cosi)
        dx, dy = rx.astype(f4) - x0f, ry.astype(f4) - y0f
        xr = -dx * sinpf + dy * cospf
        yr = (-dx * cospf - dy * sinpf) / cosif
        r = np.sqrt(xr * xr + yr * yr)
        theta = np.where(r >= 0.1, (np.arctan2(yr, xr) / F).astype(f4), f4(0.)).astype(float)
        costh = np.abs(np.cos(F * theta)).astype(f4)
        wf = (np.abs(np.sin(F * theta)) if self.wpow < 0 else np.abs(np.cos(F * theta))).astype(f4).astype(float)
        sel = ~np.isnan(v) & (r > f4(ri)) & (r < f4(ro)) & (costh > f4(free))
        if self.side == 1:
            sel &= np.abs(theta) <= 90.0
        elif self.side == 2:
            sel &= np.abs(theta) >= 90.0
        n = int(sel.sum())
        X = np.stack([rx[sel], ry[sel]], 1).astype(float)
        y = v[sel]
        w = wf[sel] ** abs(self.wpow)
        if n:
            s = y - vfunc(X, p)
            q = float(np.sum(s * s * w))
        else:
            q = 0.
        q = math.sqrt(q / (n - nfr)) if n > nfr else FLT_MAX
        return n, X, y, w, q

    def rotfit(self, ri, ro, p):
        t, eps = 100, 0.1
        nfr = int(self.mask.sum())
        n, X, y, w, q = self.getdat(p, ri, ro, nfr)
        e = np.zeros(MAXPAR)
        stop = False
        h = 0
        nrt = 0
        while not stop and h < t:
            h += 1
            chi = q
            nrt, pf0, e = lsqfit(X, y, w, p, self.mask, self.tol, t)
            if nrt < 0:
                break
            df = (pf0 - p).astype(np.float32).astype(float)
            flip = 1.0
            while True:
                pf = flip * df + p
                if pf[INC] > 90.0:
                    pf[INC] -= 180.0
                n, X, y, w, q = self.getdat(pf, ri, ro, nfr)
                if q < chi:
                    p = pf.copy()
                    break
                if 2 * h > t:
                    stop = bool(np.all(np.abs(flip * df) < eps))
                else:
                    stop = True if (q == chi and chi == 0.0) else abs(q - chi) / chi < self.tol
                if stop:
                    q = chi
                    break
                flip = -flip if flip > 0.0 else flip * -0.5
        if stop:
            ier = h
        elif nrt < 0:
            ier = nrt
        else:
            ier = -4
        return ier, p, e, n, q

    def _fit_one(self, ir):
        p = np.array([self.vsysi[ir], self.vroti[ir], self.vexpi[ir], self.posai[ir], self.incli[ir],
                      self.xposi, self.yposi], float)
        ri = max(self.rads[ir] - 0.5 * self.wids[ir], 0.0)
        ro = self.rads[ir] + 0.5 * self.wids[ir]
        ret, p, e, n, q = self.rotfit(ri, ro, p)
        return ir, ret, p, e, n, q

    def ringfit(self, nthreads=1, verbose=True, showbar=True):
        if self.vfield is None:
            print("2DFIT ERROR: Arrays are not allocated!")
            return
        bar = ProgressBar(True, verbose, showbar)
        bar.init(" Fitting 2D tilted-ring model... ", self.nrad)
        msgs = {-1: "Too many free parameters!", -2: "No free parameters!", -3: "Not enough degrees of freedom!",
                -4: "Maximum number of iterations too small!", -5: "Diagonal of matrix contains zeroes!",
                -6: "Deter. of the coeff. matrix is zero!", -7: "Square root of negative number!"}
        if nthreads > 1:
            with ThreadPoolExecutor(nthreads) as ex:
                results = list(ex.map(self._fit_one, range(self.nrad)))
        else:
            results = []
            for ir in range(self.nrad):
                results.append(self._fit_one(ir))
                bar.update(ir + 1)
        for ir, ret, p, e, n, q in results:
            if ret > 0:
                for k, idx in (("vsys", VSYS), ("vrot", VROT), ("vexp", VEXP), ("posa", PA), ("incl", INC),
                               ("xpos", X0), ("ypos", Y0)):
                    getattr(self, k + "f")[ir] = p[idx]
                    getattr(self, k + "e")[ir] = e[idx] if e[idx] < 999.99 else 999.99
                self.npts[ir] = n
                self.chis[ir] = q
            else:
                for k in ("vsys", "vrot", "vexp", "posa", "incl", "xpos", "ypos"):
                    getattr(self, k + "f")[ir] = np.nan
                    getattr(self, k + "e")[ir] = np.nan
                if verbose:
                    sys.stderr.write("Error fitting ring model: " + msgs.get(ret, "unknown error") + "\n")
        bar.fillSpace("Done.\n")

    # accessors as in C++
    def getRadius(self, i): return self.rads[i]
    def getVsysf(self, i): return self.vsysf[i]
    def getVrotf(self, i): return self.vrotf[i]
    def getVexpf(self, i): return self.vexpf[i]
    def getPosaf(self, i): return self.posaf[i]
    def getInclf(self, i): return self.inclf[i]
    def getXposf(self, i): return self.xposf[i]
    def getYposf(self, i): return self.yposf[i]

    def printfinal(self, stream, h):
        m = 10
        stream.write("#" + "".join(f"{s:>{m}}" for s in ("Radius", "Radius", "Vsys ", "Vrot ", "Vexp ", "P.A.",
                                                          "Incl.", "Xcenter", "Ycenter")) + "\n")
        stream.write("#" + "".join(f"{s:>{m}}" for s in ("[pix] ", "[arcs]", "[KM/S]", "[KM/S]", "[KM/S]", "[deg]",
                                                          "[deg]", "[pix] ", "[pix] ")) + "\n")
        conv = h.PixScale() * arcsconv(h.Cunit(0))
        for i in range(self.nrad):
            vals = (self.rads[i], self.rads[i] * conv, self.vsysf[i], self.vrotf[i], self.vexpf[i],
                    self.posaf[i], self.inclf[i], self.xposf[i], self.yposf[i])
            stream.write("".join(f"{v:>{m}.2f}" for v in vals) + "\n")

    def writeModel(self, fname, h):
        from .image import Image2D
        nx, ny = h.DimAx(0), h.DimAx(1)
        model = np.full((ny, nx), np.nan)
        F = math.pi / 180.
        ot = self.thetaf
        self.thetaf = 0
        for ir in range(self.nrad):
            p = np.array([self.vsysf[ir], self.vrotf[ir], self.vexpf[ir], self.posaf[ir], self.inclf[ir],
                          self.xposf[ir], self.yposf[ir]])
            ri = max(self.rads[ir] - 0.5 * self.wids[ir], 0.0)
            ro = self.rads[ir] + 0.5 * self.wids[ir]
            n, X, y, w, q = self.getdat(p, ri, ro, 0)
            if n == 0:
                continue
            vv = vfunc(X, p)
            for (xx, yy), v in zip(X.astype(int), vv):
                if xx < nx and yy < ny:
                    model[yy, xx] = v if np.isnan(model[yy, xx]) else (model[yy, xx] + v) / 2.
        il = self.nrad - 1
        p = np.array([self.vsysf[il], self.vrotf[il], self.vexpf[il], self.posaf[il], self.inclf[il],
                      self.xposf[il], self.yposf[il]])
        rl = self.rads[il] + 0.5 * self.wids[il]
        n, X, y, w, q = self.getdat(p, 0, rl, 0)
        for (xx, yy) in X.astype(int):
            if xx < nx and yy < ny and np.isnan(model[yy, xx]):
                xr = -(xx - p[X0]) * math.sin(F * p[PA]) + (yy - p[Y0]) * math.cos(F * p[PA])
                yr = (-(xx - p[X0]) * math.cos(F * p[PA]) - (yy - p[Y0]) * math.sin(F * p[PA])) / math.cos(F * p[INC])
                R = math.hypot(xr, yr)
                ind = int(np.argmin(np.abs(R - self.rads)))
                p = np.array([self.vsysf[ind], self.vrotf[ind], self.vexpf[ind], self.posaf[ind], self.inclf[ind],
                              self.xposf[ind], self.yposf[ind]])
                model[yy, xx] = vfunc(np.array([[xx, yy]], float), p)[0]
        img = Image2D()
        img.array = model.astype(np.float32)
        img.saveHead(h)
        img.fitswrite_2d(fname)
        self.thetaf = ot
