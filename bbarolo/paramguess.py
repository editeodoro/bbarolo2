"""
Estimation of initial parameters for 3DFIT/2DFIT (port of Utilities/paramguess.cpp and
the NelderMead class of Utilities/optimization.hh).
"""

import math
import numpy as np

from .bbutils import BBaroloError, AlltoVel, arcsconv, DeltaVel, getCenterCoordinates
from .stats import findMedian

FLT_MAX = float(np.finfo(np.float32).max)
# GUESSTYPE ELLIPSE: the semi-axes of the mask are widened by about BEAMCORR * FWHM of the beam, in
# quadrature (calibrated on simulated disks, performance/estimates.py: 0.5 to 1 give similar results)
BEAMCORR = 0.75


def _lround(x):
    return int(math.floor(x + 0.5)) if x >= 0 else -int(math.floor(-x + 0.5))


class NelderMead:
    """Downhill simplex (Numerical Recipes 2007 implementation, as in optimization.hh)."""

    def __init__(self, ftol=1E-05):
        self.ftol = ftol

    def minimize(self, point, dels, func, NMAX=5000):
        ndim = len(point)
        p = np.tile(np.asarray(point, float), (ndim + 1, 1))
        if np.isscalar(dels):
            dels = [dels] * ndim
        for i in range(1, ndim + 1):
            p[i, i - 1] += dels[i - 1]
        TINY = 1.0e-10
        y = np.array([func(list(p[i])) for i in range(ndim + 1)], float)
        nfunc = 0
        psum = p.sum(axis=0)

        def functry(ihi, fac):
            fac1 = (1.0 - fac) / ndim
            fac2 = fac1 - fac
            ptry = psum * fac1 - p[ihi] * fac2
            pl = list(ptry)
            ytry = func(pl)
            ptry = np.asarray(pl, float)       # func may clamp the point in place (as in C++, keep it)
            if ytry < y[ihi]:
                y[ihi] = ytry
                psum[:] += np.asarray(ptry) - p[ihi]
                p[ihi] = ptry
            return ytry
        while True:
            ilo = 0
            if y[0] > y[1]:
                ihi, inhi = 0, 1
            else:
                ihi, inhi = 1, 0
            for i in range(ndim + 1):
                if y[i] <= y[ilo]: ilo = i
                if y[i] > y[ihi]:
                    inhi = ihi; ihi = i
                elif y[i] > y[inhi] and i != ihi:
                    inhi = i
            rtol = 2.0 * abs(y[ihi] - y[ilo]) / (abs(y[ihi]) + abs(y[ilo]) + TINY)
            if rtol < self.ftol:
                y[0], y[ilo] = y[ilo], y[0]
                p[[0, ilo]] = p[[ilo, 0]]
                self.fmin = y[0]
                return p[0].copy()
            if nfunc >= NMAX:
                raise RuntimeError("NELDER-MEAD Optimizer: Maximum number of iterations exceeded")
            nfunc += 2
            ytry = functry(ihi, -1.0)
            if ytry <= y[ilo]:
                functry(ihi, 2.0)
            elif ytry >= y[inhi]:
                ysave = y[ihi]
                ytry = functry(ihi, 0.5)
                if ytry >= ysave:
                    for i in range(ndim + 1):
                        if i != ilo:
                            p[i] = 0.5 * (p[i] + p[ilo])
                            y[i] = func(list(p[i]))
                    nfunc += ndim
                    psum = p.sum(axis=0)
            else:
                nfunc -= 1


def _given(s):
    """The (first) value of a ring parameter that is given: a number, one value per ring, or a
    column of a file, file(name,col)."""
    s = str(s)
    if "file(" in s.lower():
        from .bbutils import getDataColumn
        col = getDataColumn(s)
        if not col:
            raise ValueError(f"3DFIT ERROR: cannot read the values of {s}")
        return float(col[0])
    return float(s.split()[0])


def funcEllipse(mypar, c, pa, xcen, ycen, Vf):
    if mypar[0] < 0: mypar[0] = abs(mypar[0])
    if mypar[1] > 90: mypar[1] = 180 - mypar[1]
    R = mypar[0] / (c.Head().PixScale() * arcsconv(c.Head().Cunit(0)))
    inc = math.radians(mypar[1])
    pa = math.radians(pa)
    yy, xx = np.mgrid[0:c.DimY(), 0:c.DimX()]
    xr = -(xx - xcen) * math.sin(pa) + (yy - ycen) * math.cos(pa)
    yr = (-(xx - xcen) * math.cos(pa) - (yy - ycen) * math.sin(pa)) / math.cos(inc)
    r = np.sqrt(xr * xr + yr * yr)
    isIn = r <= R
    nanv = np.isnan(Vf)
    return float(np.sum(isIn & nanv) - np.sum(isIn & ~nanv))


class ParamGuess:
    """Initial guesses for geometrical and kinematical parameters from a detection."""

    def __init__(self, c, obj):
        self.cin = c
        self.obj = obj
        if not obj.haveParams:
            obj.calcAllParams(c.array, c.Head())
        nx, ny = c.DimX(), c.DimY()
        f = c.array[obj.z, obj.y, obj.x].astype(np.float32)
        zph = np.asarray(c.getZphys(obj.z.astype(float)), dtype=np.float64)
        fluxsum = np.zeros(ny * nx, np.float64)
        fluxint = np.zeros(ny * nx, np.float64)
        idx = obj.y * nx + obj.x
        np.add.at(fluxsum, idx, f)
        np.add.at(fluxint, idx, f * zph)
        self.Intmap = fluxsum.reshape(ny, nx).astype(np.float32)
        self.totflux_obs = float(np.float32(fluxsum.sum()))
        with np.errstate(invalid="ignore", divide="ignore"):
            self.Vemap = np.asarray(AlltoVel(fluxint / fluxsum, c.Head()), dtype=np.float64).reshape(ny, nx).astype(np.float32)
        self.radsep = c.Head().Bmaj() * arcsconv(c.Head().Cunit(0))
        self.xcentre = self.ycentre = self.vsystem = self.inclin = self.posang = self.Rmax = self.vrot = 0.
        self.nrings = 0
        self.vdisp = None             # (GUESSTYPE FIT) velocity dispersion from the second moment
        self.z0_given = 0.            # (FIT) the thickness of the map model (arcsec): Z0, or its default
        self.rad_profile, self.inc_profile, self.pa_profile = [], [], []   # (FIT) radial profiles
        self.mode = "LEGACY"         # GUESSTYPE: LEGACY (as C++), ELLIPSE (C++ corrected) or FIT
        self.major_max = [0, 0]; self.major_min = [0, 0]
        self.minor_max = [0, 0]; self.minor_min = [0, 0]

    def findAll(self):
        self.findSystemicVelocity()
        self.findCentre()
        self.findPositionAngle()
        self.findInclination()
        self.findRings()
        self.findRotationVelocity()

    def findCentre(self):
        o = self.obj
        self.xcentre = (o.getXcentre() + o.getXaverage()) / 2.
        self.ycentre = (o.getYcentre() + o.getYaverage()) / 2.

    def findSystemicVelocity(self):
        self.vsystem = self.obj.getVsys()

    def findRotationVelocity(self):
        self.vrot = abs(self.obj.getW50() / 2.)
        if self.mode == "LEGACY":
            if self.inclin > 40:
                self.vrot /= math.sin(math.radians(self.inclin))
        else:
            # always deprojected (C++: only above 40 deg, so VROT was too low at low inclination)
            self.vrot /= math.sin(math.radians(max(self.inclin, 10.)))

    def findPositionAngle(self, algorithm=1):
        c, o, V = self.cin, self.obj, self.Vemap.ravel()
        nx = c.DimX()
        velmin = AlltoVel(c.getZphys(0), c.Head())
        velmax = AlltoVel(c.getZphys(c.DimZ() - 1), c.Head())
        if velmin > velmax:
            velmin, velmax = velmax, velmin
        Xmin, Ymin, Xmax, Ymax = o.getXmin(), o.getYmin(), o.getXmax(), o.getYmax()
        xc, yc, vs = self.xcentre, self.ycentre, self.vsystem
        if algorithm == 1:
            maxdev = bestpa = vl = vr = 0.
            p = 0.
            while p < 180:
                vdev = []
                sl = sr = 0.
                if 45 < p < 135:
                    for y in range(Ymin, Ymax + 1):
                        x = _lround(1 / math.tan(math.radians(p)) * (y - yc) + xc)
                        if p == 90: x = _lround(xc)
                        if not (Xmin <= x <= Xmax): continue
                        v = V[y * nx + x]
                        if np.isnan(v) or not (velmin <= v <= velmax): continue
                        vdev.append(abs(v - vs))
                        if p <= 90:
                            if y < yc: sl += v - vs
                            else: sr += v - vs
                        else:
                            if x < xc: sl += v - vs
                            else: sr += v - vs
                else:
                    for x in range(Xmin, Xmax + 1):
                        y = _lround(math.tan(math.radians(p)) * (x - xc) + yc)
                        if not (Ymin <= y <= Ymax): continue
                        v = V[y * nx + x]
                        if np.isnan(v) or not (velmin <= v <= velmax): continue
                        vdev.append(abs(v - vs))
                        if x < xc: sl += v - vs
                        else: sr += v - vs
                med = findMedian(np.array(vdev, np.float32)) if vdev else np.nan
                if med > maxdev and abs(med) < 1E16:
                    maxdev, bestpa, vl, vr = med, p, sl, sr
                p += 0.5
            if vl < vr:
                self.posang = 270 + bestpa if bestpa <= 90 else 90 + bestpa
            else:
                self.posang = 90 + bestpa if bestpa <= 90 else bestpa - 90
        elif algorithm == 2:
            vel_high = vel_low = vs
            rng = int(math.ceil(c.Head().Bmaj() / c.Head().PixScale()))
            ch, cl = [0, 0], [0, 0]
            xsize, ysize = abs(Xmax - Xmin) + 1, abs(Ymax - Ymin) + 1
            Vm = self.Vemap
            # Median velocity in (2rng+1)^2 boxes around each pixel (vectorised): the
            # lowest/highest medians within [velmin, velmax] (first in row-major order).
            ny_, nx_ = ysize - 2 * rng, xsize - 2 * rng
            if ny_ > 0 and nx_ > 0:
                sub = Vm[Ymin:Ymin + ysize, Xmin:Xmin + xsize]
                win = np.lib.stride_tricks.sliding_window_view(sub, (2 * rng + 1, 2 * rng + 1))[:ny_, :nx_]
                meds = np.median(win.reshape(ny_, nx_, -1), axis=2)
                okc = ~np.isnan(sub[rng:rng + ny_, rng:rng + nx_])
                with np.errstate(invalid="ignore"):
                    lowc = okc & (meds < vel_low) & (meds >= velmin)
                    highc = okc & (meds > vel_high) & (meds <= velmax)
                if lowc.any():
                    k = np.argmin(np.where(lowc, meds, np.inf))
                    y, x = divmod(int(k), nx_)
                    vel_low, cl = meds[y, x], [x + rng + Xmin, y + rng + Ymin]
                if highc.any():
                    k = np.argmax(np.where(highc, meds, -np.inf))
                    y, x = divmod(int(k), nx_)
                    vel_high, ch = meds[y, x], [x + rng + Xmin, y + rng + Ymin]
            xx = np.array([cl[0], ch[0], _lround(xc)], float)
            yy = np.array([cl[1], ch[1], _lround(yc)], float)
            m = np.polyfit(xx, yy, 1)[0]
            ang = math.atan(m)
            if ch[0] >= xc:
                self.posang = math.degrees(3 * math.pi / 2 + ang) if ang < math.pi / 2 else math.degrees(math.pi / 2 + ang)
            else:
                self.posang = math.degrees(math.pi / 2 + ang) if ang < math.pi / 2 else math.degrees(ang - math.pi / 2)
        else:
            raise ValueError(f"PARAMGUESS ERROR: unknown algorithm value {algorithm}")

    def setAxesLine(self, xcen, ycen, pa):
        m = pa - 90
        while m >= 180: m -= 180
        while m < 0: m += 180
        maj0 = math.tan(math.radians(m))
        maj = [maj0, ycen - maj0 * xcen]
        min0 = -1 / maj0 if maj0 != 0 else -1e30
        mnr = [min0, ycen - min0 * xcen]
        return maj, mnr

    def findAxisLength(self, lpar):
        c, o, V = self.cin, self.obj, self.Vemap.ravel()
        nx = c.DimX()
        up, low = [0, 0], [0, 0]
        arl = arr = 0.
        Xmin, Ymin, Xmax, Ymax = o.getXmin(), o.getYmin(), o.getXmax(), o.getYmax()
        xc, yc = self.xcentre, self.ycentre
        p = math.degrees(math.atan(lpar[0]))
        while p >= 180: p -= 180
        while p < 0: p += 180
        if 45 < p < 135:
            for y in range(Ymin, Ymax + 1):
                x = _lround((y - lpar[1]) / lpar[0]) if lpar[0] != 0 else _lround(xc)
                if abs(90. - p) < 0.1: x = _lround(xc)
                if not (Xmin <= x <= Xmax) or np.isnan(V[y * nx + x]): continue
                r = math.hypot(x - xc, y - yc)
                if p - 90 < 0.5:
                    if r > arl and y <= yc: arl, up = r, [x, y]
                    if r > arr and y > yc: arr, low = r, [x, y]
                else:
                    if r > arl and x <= xc: arl, up = r, [x, y]
                    if r > arr and x > xc: arr, low = r, [x, y]
        else:
            for x in range(Xmin, Xmax + 1):
                y = _lround(lpar[0] * x + lpar[1])
                if not (Ymin <= y <= Ymax) or np.isnan(V[y * nx + x]): continue
                r = math.hypot(x - xc, y - yc)
                if r > arl and x <= xc: arl, up = r, [x, y]
                if r > arr and x > xc: arr, low = r, [x, y]
        return (arr + arl) / 2., up, low

    def findInclination(self, algorithm=2):
        c = self.cin
        pmaj, pmin = self.setAxesLine(self.xcentre, self.ycentre, self.posang)
        axmaj, self.major_max, self.major_min = self.findAxisLength(pmaj)
        axmin, self.minor_max, self.minor_min = self.findAxisLength(pmin)
        if axmin > axmaj:
            print("---------------> WARNING - Finding initial parameters <--------------\n"
                  " The major axis is shorter than the minor axis. They will be swapped\n"
                  " for estimating the inclination.\n"
                  " The galaxy seems to be less elongated in the kinematical axis!!\n")
            axmin, axmaj = axmaj, axmin
        self.inclin = math.degrees(math.acos(axmin / axmaj)) if axmaj > 0 else 45.
        self.Rmax = axmaj * c.Head().PixScale() * arcsconv(c.Head().Cunit(0))
        if algorithm == 1:
            return
        if algorithm in (2, 3):
            point = [self.Rmax, self.inclin]
            dels = [0.1 * v for v in point]
            point = [v - 0.05 * v for v in point]
            if algorithm == 3:
                func = lambda par: funcIncfromMap(par, c, self.radsep, self.Rmax, self.posang, self.xcentre,
                                                  self.ycentre, self.vsystem, self.Intmap, self.totflux_obs)
            else:
                func = lambda par: funcEllipse(par, c, self.posang, self.xcentre, self.ycentre, self.Vemap)
            try:
                mymin = NelderMead().minimize(point, dels, func)
            except RuntimeError:
                raise BBaroloError("PARAMGUESS ERROR: Error while estimating inclination.")
            self.Rmax, self.inclin = float(mymin[0]), float(mymin[1])
            bm = c.Head().Bmaj() * 3600.
            if self.mode != "LEGACY":
                self._beam_corrected_inclination(bm)
                return
            Axmin = self.Rmax * math.cos(math.radians(self.inclin))
            Axmin_corr = Axmin - bm
            Axmaj_corr = self.Rmax - bm
            if self.inclin < 75:
                if Axmin_corr > 0 and Axmaj_corr > 0 and Axmin_corr / bm > 1:
                    self.inclin = math.degrees(math.acos(Axmin_corr / Axmaj_corr))
            else:
                if Axmin_corr / bm > 1 and Axmaj_corr / bm > 9:
                    self.inclin = math.degrees(math.acos(Axmin_corr / Axmaj_corr))
            if 2 * Axmin < bm:
                self.inclin = 80
        else:
            raise ValueError(f"PARAMGUESS ERROR: unknown algorithm value {algorithm}")
        if self.inclin < 10:
            self.inclin = 25

    def _beam_corrected_inclination(self, bm):
        """INC from the semi-axes of the ellipse fitted to the mask, corrected for the beam: both are
        widened by about k*FWHM, in quadrature (C++ subtracted a full FWHM from both, which over-
        corrected the minor axis at low inclination, and did not correct above 75 deg). No jumps."""
        k = BEAMCORR
        a, b = self.Rmax, self.Rmax * math.cos(math.radians(self.inclin))
        a2, b2 = a * a - (k * bm) ** 2, b * b - (k * bm) ** 2
        if a2 > 0:
            self.inclin = math.degrees(math.acos(min(math.sqrt(max(b2, 0.) / a2), 1.)))
        self.inclin = min(max(self.inclin, 5.), 89.)

    def findRings(self):
        self.nrings = _lround(self.Rmax / self.radsep)
        if self.nrings < 20:
            self.radsep /= 2.
            self.nrings = _lround(self.Rmax / self.radsep)
        if self.nrings > 4:
            self.nrings -= 1

    def tuneWithTiltedRing(self, given=None):
        from .ringmodel import Ringmodel, VEXP, INC
        c = self.cin
        rmax = self.Rmax / (c.Head().PixScale() * arcsconv(c.Head().Cunit(0))) - 2
        rwidth = 1                   # (rings of one pixel: their median PA is good also at low resolution)
        nr = int(rmax / rwidth)
        if nr < 4:
            return
        radii = 2 + np.arange(nr) * rwidth
        tr = Ringmodel()
        tr.set(nr, radii, rwidth, self.vsystem, self.vrot, 0, self.posang, self.inclin, self.xcentre, self.ycentre)
        tr.setfield(self.Vemap, c.DimX(), c.DimY())
        mpar = [True] * 7
        mpar[VEXP] = False
        mpar[INC] = False
        tr.setoption(mpar, 3, 2, 15.)
        tr.ringfit(c.pars().threads if self.mode == "LEGACY" else 1, False, False)
        if self.mode == "LEGACY":
            with open(c.pars().outFolder + c.pars().outPrefix + "_2drings.txt", "w") as f:
                tr.printfinal(f, c.Head())

        def med(a):
            a = np.asarray(a[1:])
            a = a[~np.isnan(a)]
            return float(np.median(a.astype(np.float32))) if a.size > 1 else None
        for attr, arr in (("xcentre", tr.xposf), ("ycentre", tr.yposf), ("vsystem", tr.vsysf),
                          ("inclin", tr.inclf), ("posang", tr.posaf)):
            m = med(arr)
            # (ELLIPSE, FIT: the parameters given are kept; C++ replaced them too, here and in the
            # inclination estimated from them)
            if m is not None and not (self.mode != "LEGACY" and given and given.get(attr)):
                setattr(self, attr, m)
        self.findInclination(2)
        self.findRotationVelocity()

    # ------------------------------------------------------------------ GUESSTYPE FIT
    def fitGeometry(self, free):
        """GUESSTYPE FIT: refine the estimates (from the ELLIPSE algorithm) with (1) a beam-convolved
        disk fitted to the total intensity map of the detection (centre, INC, PA; the densities of the
        rings are deconvolved for each geometry, bbarolo.profile), (2) a 2D tilted-ring fit of the
        velocity field on rings one beam wide (PA, VSYS, VROT, and radial profiles of INC and PA),
        (3) the velocity dispersion from the second moment. free: {name: True if not given}."""
        self._fit_map(free)
        self._fit_velfield(free)
        self._estimate_vdisp()

    def _beam_pix(self):
        h = self.cin.Head()
        return max(h.Bmaj() / h.PixScale(), 1.)

    def _fit_map(self, free):
        from scipy.optimize import least_squares
        from scipy.ndimage import binary_dilation
        from .galmod import DiskModel
        from .rings import Rings
        from .profile import profile_system, solve_profile
        names = [n for n in ("xcentre", "ycentre", "inclin", "posang") if free.get(n, True)]
        if not names:
            return
        c, h = self.cin, self.cin.Head()
        arcs = h.PixScale() * arcsconv(h.Cunit(0))           # arcsec per pixel
        bpix = self._beam_pix()
        foot = np.asarray(self.Intmap) != 0
        if foot.sum() < 20:
            return
        # the data: the cube summed over the channels of the detection, not masked (the mask cuts the
        # faint outer emission at low S/N, which biases INC high); the noise is the same everywhere
        o = self.obj
        data = c.array[o.getZmin():o.getZmax() + 1].astype(np.float64).sum(axis=0)
        rad = int(math.ceil(1.5 * bpix))
        yy, xx = np.mgrid[-rad:rad + 1, -rad:rad + 1]
        ok = binary_dilation(foot, structure=(xx * xx + yy * yy) <= rad * rad)   # zeros within a beam too
        ys, xs = np.nonzero(ok)
        m = 2 * rad
        blo = (max(int(xs.min()) - m, 0), max(int(ys.min()) - m, 0))
        bhi = (min(int(xs.max()) + m + 1, c.DimX()), min(int(ys.max()) + m + 1, c.DimY()))
        sl = (slice(blo[1], bhi[1]), slice(blo[0], bhi[0]))
        dmap, okb = data[sl], ok[sl]
        w = max(h.Bmaj() * 3600. / 2., 1.5 * arcs)              # rings of half a beam
        nr = max(int(math.ceil(self.Rmax / w)), 2)
        radii = (np.arange(nr) + 0.5) * w
        dm = DiskModel(c, par=c.pars().gf, ringmode="CENTRED", galmodtype="ANALYTIC", smooth=True)
        start = {n: getattr(self, n) for n in ("xcentre", "ycentre", "inclin", "posang")}
        rms = np.ones_like(dmap)

        def model(th):
            g = dict(start)
            g.update(zip(names, th))
            r = Rings().setRings(radii=radii, xpos=g["xcentre"], ypos=g["ycentre"], vsys=self.vsystem, vrot=100.,
                                 vdisp=10., z0=self.z0_given, inc=g["inclin"], phi=g["posang"], dens=1., radsep=w)
            sysm = profile_system(dm, r, (blo, bhi), dmap, rms, okb, beam_arcsec=h.Bmaj() * 3600.)
            if sysm is None:
                return None
            A, b, _, Rc = sysm
            return A @ solve_profile(A, b, Rc, 0.1) - b

        lo = {"xcentre": start["xcentre"] - 2 * bpix, "ycentre": start["ycentre"] - 2 * bpix, "inclin": 5.,
              "posang": start["posang"] - 60.}
        hi = {"xcentre": start["xcentre"] + 2 * bpix, "ycentre": start["ycentre"] + 2 * bpix, "inclin": 89.,
              "posang": start["posang"] + 60.}
        x0 = np.array([min(max(start[n], lo[n] + 1e-6), hi[n] - 1e-6) for n in names])
        r0 = model(x0)
        if r0 is None:
            return
        try:
            res = least_squares(lambda t: model(t), x0, bounds=([lo[n] for n in names], [hi[n] for n in names]),
                                x_scale=[1. if n in ("xcentre", "ycentre") else 2. for n in names],
                                diff_step=1e-3, max_nfev=60 * len(names))
        except Exception:
            return
        ell_inc = self.inclin
        for n, v in zip(names, res.x):
            setattr(self, n, float(v))
        if "inclin" in names and self.inclin < 15.:
            # nearly round maps hardly constrain INC: the fit drifts to its lower bound (5 deg), where
            # 3DFIT then stays; the ELLIPSE value is better there (on simulated disks)
            self.inclin = ell_inc
        self.posang %= 360.
        self.map_fit_cost = float(res.cost)

    def _fit_velfield(self, free):
        from .ringmodel import Ringmodel, VSYS, VROT, VEXP, PA, INC, X0, Y0
        c, h = self.cin, self.cin.Head()
        bpix = self._beam_pix()
        arcs = h.PixScale() * arcsconv(h.Cunit(0))
        rmax = self.Rmax / arcs
        w = max(bpix, 2.)
        nr = int(rmax / w)
        if nr < 2:
            w = max(bpix / 2., 1.5)
            nr = int(rmax / w)
        if nr < 2:
            return
        radii = (np.arange(nr) + 0.5) * w
        tr = Ringmodel()
        tr.set(nr, radii, w, self.vsystem, max(self.vrot, 20.), 0., self.posang, self.inclin, self.xcentre,
               self.ycentre)
        tr.setfield(self.Vemap, c.DimX(), c.DimY())
        mpar = [False] * 7
        mpar[VROT] = True
        mpar[PA] = free.get("posang", True)
        mpar[INC] = True
        mpar[VSYS] = free.get("vsystem", True)
        tr.setoption(mpar, 3, 2, 15.)
        with np.errstate(all="ignore"):
            tr.ringfit(1, False, False)
        ok = np.isfinite(tr.vrotf) & np.isfinite(tr.posaf) & np.isfinite(tr.inclf) & (tr.vrotf > 0)
        if ok.sum() < 2:
            return
        self.rad_profile = (radii[ok] * arcs).tolist()
        self.inc_profile = np.clip(tr.inclf[ok], 5., 89.).tolist()
        self.pa_profile = (tr.posaf[ok] % 360.).tolist()
        if free.get("posang", True):
            # the kinematic PA: the median over the rings, near the PA of the map fit (same sense)
            pa = np.asarray(self.pa_profile)
            d = (pa - self.posang + 180.) % 360. - 180.
            self.posang = float((self.posang + np.median(d)) % 360.)
        if free.get("vsystem", True) and np.isfinite(tr.vsysf[ok]).any():
            self.vsystem = float(np.median(tr.vsysf[ok]))
        if free.get("vrot", True):
            # W50/2 deprojected with the final INC (the velocity field is smoothed by the beam, and its
            # rotation velocities were 20-25% low on simulated disks)
            self.findRotationVelocity()

    def _estimate_vdisp(self):
        """VDISP from the second moment of the detection, in its outer half (less beam smearing),
        corrected for the channel width."""
        c, o = self.cin, self.obj
        f = c.array[o.z, o.y, o.x].astype(np.float64)
        v = np.asarray(AlltoVel(c.getZphys(o.z.astype(float)), c.Head()), float)
        nx = c.DimX()
        idx = o.y * nx + o.x
        n = c.DimY() * nx
        s0 = np.bincount(idx, f, n)
        s1 = np.bincount(idx, f * v, n)
        s2 = np.bincount(idx, f * v * v, n)
        with np.errstate(invalid="ignore", divide="ignore"):
            m1 = s1 / s0
            m2 = np.sqrt(np.maximum(s2 / s0 - m1 * m1, 0.))
        h = c.Head()
        arcs = h.PixScale() * arcsconv(h.Cunit(0))
        yy, xx = np.divmod(np.arange(n), nx)
        pa, inc = math.radians(self.posang), math.radians(self.inclin)
        xr = -(xx - self.xcentre) * math.sin(pa) + (yy - self.ycentre) * math.cos(pa)
        yr = (-(xx - self.xcentre) * math.cos(pa) - (yy - self.ycentre) * math.sin(pa)) / max(math.cos(inc), 0.1)
        R = np.hypot(xr, yr) * arcs
        sel = (s0 > 0) & np.isfinite(m2) & (R > 0.5 * self.Rmax) & (m2 > 0)
        if sel.sum() < 5:
            return
        chw = abs(DeltaVel(h))
        disp2 = np.median(m2[sel]) ** 2 - chw * chw / 12.
        self.vdisp = float(math.sqrt(max(disp2, 4.)))

    def plotGuess(self, outfile="initial_guesses.pdf"):
        """Diagnostic plot of the initial guesses (direct matplotlib version)."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        c, o = self.cin, self.obj
        b = 10
        x0, x1 = max(o.getXmin() - b, 0), min(o.getXmax() + b, c.DimX() - 1)
        y0, y1 = max(o.getYmin() - b, 0), min(o.getYmax() + b, c.DimY() - 1)
        f = self.Intmap[y0:y1 + 1, x0:x1 + 1].copy()
        f[f == 0] = np.nan
        v = self.Vemap[y0:y1 + 1, x0:x1 + 1]
        axmaj = self.Rmax / (c.Head().PixScale() * arcsconv(c.Head().Cunit(0)))
        axmin = axmaj * math.cos(math.radians(self.inclin))
        xp, yp = self.xcentre - x0, self.ycentre - y0
        t = np.linspace(0, 2 * np.pi, 100)
        pr = math.radians(self.posang - 90)
        xt = xp + axmaj * math.cos(pr) * np.cos(t) - axmin * math.sin(pr) * np.sin(t)
        yt = yp + axmaj * math.sin(pr) * np.cos(t) + axmin * math.cos(pr) * np.sin(t)
        spec = c.array[:, y0:y1 + 1, x0:x1 + 1].astype(float).sum(axis=(1, 2))
        vel = AlltoVel(c.getZphys(np.arange(c.DimZ())), c.Head())
        fig = plt.figure(figsize=(10, 10))
        ax0 = fig.add_axes([0.00, 0.42, 0.3, 0.3])
        ax1 = fig.add_axes([0.31, 0.42, 0.3, 0.3])
        ax2 = fig.add_axes([0.00, 0.1, 0.4, 0.3])
        ax0.imshow(f, origin="lower", cmap="Spectral_r", aspect="auto")
        ax1.imshow(v, origin="lower", cmap=plt.get_cmap("RdBu_r", 25), aspect="auto")
        for a in (ax0, ax1):
            a.plot(xt, yt, "k")
            a.plot(xp, yp, "x", color="k", markersize=7, mew=1.5)
            a.set_xlim(0, x1 - x0 + 1); a.set_ylim(0, y1 - y0 + 1)
        ax2.plot(vel, spec, ".-")
        ax2.axvline(self.vsystem, c="green")
        ax2.set_xlabel("V$_\\mathrm{LOS}$ (km/s)")
        pstr = [f"X$_\\mathrm{{pos}}$ = {self.xcentre:.2f} pix", f"Y$_\\mathrm{{pos}}$ = {self.ycentre:.2f} pix",
                f"$\\phi$ = {self.posang:.0f}$^\\circ$", f"inc = {self.inclin:.1f}$^\\circ$",
                f"R$_\\mathrm{{max}}$ = {axmaj:.2f} pix", f"V$_\\mathrm{{sys}}$ = {self.vsystem:.2f} km/s"]
        for i, s in enumerate(pstr):
            ax2.text(1.1, 0.97 - i * 0.08, s, va="top", transform=ax2.transAxes)
        fig.savefig(c.pars().outFolder + outfile, bbox_inches="tight")
        plt.close(fig)
        return 0


def funcIncfromMap(mypar, c, radsep, Rmax, pa, xcen, ycen, vsys, Imap, totflux):
    from .rings import Rings
    from .moments import MomentMap
    from .ellprof import Ellprof
    from .galmod import DiskModel
    verb = c.pars().verbose
    c.pars().verbose = False
    if mypar[0] < 0: mypar[0] = 2 * radsep
    if mypar[0] > 1.5 * Rmax: mypar[0] = Rmax
    if mypar[1] < 0: mypar[1] = 1
    if mypar[1] > 90: mypar[1] = 89
    RMAX, inc = mypar[0], mypar[1]
    size = int((RMAX - 0) / radsep + 1)
    r = Rings().setRings(radii=np.arange(size) * radsep, xpos=xcen, ypos=ycen, vsys=vsys,
                         vrot=10 * DeltaVel(c.Head()), vdisp=8, vrad=0, vvert=0, dvdz=0, zcyl=0, dens=1, z0=0,
                         inc=inc, phi=pa, radsep=radsep)
    tm = MomentMap()
    tm.input(c)
    tm.SumMap(True)
    tm.array = np.asarray(Imap, np.float32)
    ell = Ellprof(image=tm, rings=r)
    ell.RadialProfile()
    profmin = FLT_MAX
    for i in range(r.nr):
        m = ell.getMean(i)
        if not np.isnan(m) and profmin > m > 0:
            profmin = m
    factor = 1.
    while profmin < 0.1:
        profmin *= 10; factor *= 10
    while profmin > 10:
        profmin /= 10; factor /= 10
    r.dens = factor * np.abs(np.array([ell.getMean(i) for i in range(r.nr)]))
    r.dens[r.dens == 0] = profmin
    # the model smoothed to the beam, with the values given at the radii (NODES)
    dm = DiskModel(c, par=c.pars().gm, ringmode="NODES", smooth=True)
    map_mod = dm.array(r, threads=1).sum(axis=0)
    fac = totflux / map_mod.sum()
    res = float(np.sum(np.abs(Imap - map_mod * fac)))
    c.pars().verbose = verb
    return res


def EstimateInitial(c, p):
    """Estimate the initial parameters not given in the GALFIT_PAR structure p."""
    from .search import Detection
    largest = None
    if c.MaskAll() and "FILE(" in c.pars().MaskType:
        z, y, x = np.nonzero(c.Mask())
        largest = Detection(z, y, x)
        largest.calcAllParams(c.array, c.Head())
    else:
        if not c.isSearched:
            c.search()
        largest = c.getSources().LargestDetection() if c.getSources() is not None else None
    verb = c.pars().verbose
    c.pars().verbose = False
    if verb:
        print("\n Estimating initial parameters... ", end="", flush=True)
    if largest is None:
        raise BBaroloError(" 3DFIT ERROR: No sources detected in the datacube. Cannot fit!!! ")
    ip = ParamGuess(c, largest)
    ip.mode = getattr(p, "GUESSTYPE", "LEGACY")
    if p.VSYS != "-1": ip.vsystem = _given(p.VSYS)
    else: ip.findSystemicVelocity()
    if p.XPOS != "-1" and p.YPOS != "-1":
        from .rings import firstValue
        def first(s):
            return repr(_given(s)) if "file(" in str(s).lower() else firstValue(s)
        ip.xcentre, ip.ycentre = getCenterCoordinates([first(p.XPOS), first(p.YPOS)], c.Head())
    else:
        ip.findCentre()
    if p.PHI != "-1": ip.posang = _given(p.PHI)
    else: ip.findPositionAngle(1)
    ip.findInclination(2)
    if p.INC != "-1": ip.inclin = _given(p.INC)
    if p.NRADII != -1 and p.RADSEP != -1:
        ip.radsep, ip.nrings = p.RADSEP, p.NRADII
    else:
        ip.findRings()
    if p.VROT != "-1": ip.vrot = _given(p.VROT)
    else: ip.findRotationVelocity()
    if c.pars().plots >= 3:
        ip.plotGuess(c.pars().outPrefix + "_initialguesses_0.pdf")
    given = dict(xcentre=p.XPOS != "-1" and p.YPOS != "-1", inclin=p.INC != "-1", posang=p.PHI != "-1",
                 vsystem=p.VSYS != "-1", vrot=p.VROT != "-1")
    given["ycentre"] = given["xcentre"]
    if ip.nrings > 3:
        ip.tuneWithTiltedRing(given)         # (FIT: the ELLIPSE estimates are its starting point)
        if ip.mode != "LEGACY":
            # the tuning estimates INC and VROT again: the values given stay (C++ replaced them in its
            # internal estimates; only the parameters not given are used, but the others are derived)
            if given["inclin"]: ip.inclin = _given(p.INC)
            if given["vrot"]: ip.vrot = _given(p.VROT)
            elif given["inclin"]: ip.findRotationVelocity()
    if ip.mode == "FIT":
        from .galfit import default_z0
        ip.z0_given = _given(p.Z0) if p.Z0 != "-1" else default_z0(c.Head())
        ip.fitGeometry({k: not v for k, v in given.items()})
    if c.pars().plots >= 2:
        ip.plotGuess(c.pars().outPrefix + "_initialguesses.pdf")
    if verb:
        print("Done.")
    c.pars().verbose = verb
    return ip
