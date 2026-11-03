"""
Estimation of initial parameters for 3DFIT/2DFIT (port of Utilities/paramguess.cpp and
the NelderMead class of Utilities/optimization.hh).
"""

import math
import numpy as np

from .bbutils import AlltoVel, arcsconv, DeltaVel, getCenterCoordinates
from .stats import findMedian

FLT_MAX = float(np.finfo(np.float32).max)


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
        if self.inclin > 40:
            self.vrot /= math.sin(math.radians(self.inclin))

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
                raise RuntimeError("PARAMGUESS ERROR: Error while estimating inclination.")
            self.Rmax, self.inclin = float(mymin[0]), float(mymin[1])
            bm = c.Head().Bmaj() * 3600.
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

    def findRings(self):
        self.nrings = _lround(self.Rmax / self.radsep)
        if self.nrings < 20:
            self.radsep /= 2.
            self.nrings = _lround(self.Rmax / self.radsep)
        if self.nrings > 4:
            self.nrings -= 1

    def tuneWithTiltedRing(self):
        from .ringmodel import Ringmodel, VEXP, INC
        c = self.cin
        rmax = self.Rmax / (c.Head().PixScale() * arcsconv(c.Head().Cunit(0))) - 2
        rwidth = 1
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
        tr.ringfit(c.pars().threads, False, False)
        with open(c.pars().outFolder + c.pars().outPrefix + "_2drings.txt", "w") as f:
            tr.printfinal(f, c.Head())

        def med(a):
            a = np.asarray(a[1:])
            a = a[~np.isnan(a)]
            return float(np.median(a.astype(np.float32))) if a.size > 1 else None
        for attr, arr in (("xcentre", tr.xposf), ("ycentre", tr.yposf), ("vsystem", tr.vsysf),
                          ("inclin", tr.inclf), ("posang", tr.posaf)):
            m = med(arr)
            if m is not None:
                setattr(self, attr, m)
        self.findInclination(2)
        self.findRotationVelocity()

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
        raise RuntimeError(" 3DFIT ERROR: No sources detected in the datacube. Cannot fit!!! ")
    ip = ParamGuess(c, largest)
    if p.VSYS != "-1": ip.vsystem = float(str(p.VSYS).split()[0])
    else: ip.findSystemicVelocity()
    if p.XPOS != "-1" and p.YPOS != "-1":
        from .rings import firstValue
        ip.xcentre, ip.ycentre = getCenterCoordinates([firstValue(p.XPOS), firstValue(p.YPOS)], c.Head())
    else:
        ip.findCentre()
    if p.PHI != "-1": ip.posang = float(str(p.PHI).split()[0])
    else: ip.findPositionAngle(1)
    ip.findInclination(2)
    if p.INC != "-1": ip.inclin = float(str(p.INC).split()[0])
    if p.NRADII != -1 and p.RADSEP != -1:
        ip.radsep, ip.nrings = p.RADSEP, p.NRADII
    else:
        ip.findRings()
    if p.VROT != "-1": ip.vrot = float(str(p.VROT).split()[0])
    else: ip.findRotationVelocity()
    if c.pars().plots >= 3:
        ip.plotGuess("initialguesses_" + c.pars().outPrefix + "_0.pdf")
    if ip.nrings > 3:
        ip.tuneWithTiltedRing()
    if c.pars().plots >= 2:
        ip.plotGuess("initialguesses_" + c.pars().outPrefix + ".pdf")
    if verb:
        print("Done.")
    c.pars().verbose = verb
    return ip
