"""
Radial profiles in elliptical rings (port of Tasks/ellprof.cpp, from GIPSY's ELLINT).
The pixel/sub-pixel/ring loops are vectorised with numpy.
"""

import sys
import math
import numpy as np

from .moments import MomentMap
from .rings import readRings
from .stats import findMADFM
from .bbutils import arcsconv, _jy_prefix

FLT_MAX = float(np.finfo(np.float32).max)


def _toangle_scalar(a):
    while a < 0.0:
        a += 360.0
    while a > 360.0:
        a -= 360.0
    return a


def _in_segment(angle, s1, s2):
    if abs(s2 - s1) >= 360.0:
        return np.ones_like(angle, bool)
    if s1 < s2:
        return (angle >= s1) & (angle < s2)
    if s1 > s2:
        return (angle >= s1) | (angle < s2)
    return angle == s1


class Ellprof:
    """Radial profile of a moment map (or of a cube, via its total intensity map)."""

    def __init__(self, c=None, inR=None, mask=True, image=None, rings=None, nseg=1, segments=None):
        self.Overlap = True
        self.Range = [-FLT_MAX, FLT_MAX]
        self.subpix = [2, 2]
        self.maprotation = 0.
        self.Rmax = 0.
        if c is not None:
            if inR is None:
                inR = readRings(c.pars().gf, c.Head())
            self.setFromCube(c, inR, mask)
        elif image is not None:
            self.im = image
            self.update_rings(rings, nseg, segments)

    def setFromCube(self, c, inR, mask=True):
        meanPA = float(np.mean(inR.phi))
        nseg = 1
        segments = [0, 360., 0., 0]
        side = c.pars().gf.SIDE
        # the azimuth of the rings is 0 on the receding half of the major axis, at any PA (C++ took the
        # opposite halves, and swapped them for a mean PA above 180 degrees: right only then)
        if side == "A":
            nseg = 2; segments[2], segments[3] = 90, -90
        elif side == "R":
            nseg = 2; segments[2], segments[3] = -90, 90
        im = MomentMap()
        im.input(c)
        if c.Head().NumAx() > 2 and c.DimZ() > 1:
            im.ZeroMoment(mask)
        else:
            im.array = c.array[0].copy()
            blank = getattr(c, "blanks2d", None)        # (the blank pixels of a 2D map, read as 0)
            if blank is not None and blank.shape == im.array.shape:
                im.array[blank] = np.nan
            im.setHead(0)
            im.Head().setBunit(c.Head().Bunit())
        self.im = im
        self.update_rings(inR, nseg, segments)

    def update_rings(self, rings, nseg=1, segments=None):
        self.Nseg = nseg
        nr = rings.nr
        self.Nrad = nr
        self.Radius = np.asarray(rings.radii, float).copy()
        self.Phi = np.array([_toangle_scalar(float(p)) for p in rings.phi])
        self.Inc = np.asarray(rings.inc, float).copy()
        W = np.zeros(nr)
        for r in range(nr):
            if nr == 1:
                W[r] = rings.radsep if rings.radsep > 0 else rings.radii[0] / 2.
            elif r == nr - 1:
                W[r] = rings.radii[r] - rings.radii[r - 1]
            else:
                W[r] = rings.radii[r + 1] - rings.radii[r]
        self.Width = W
        ang = np.radians(-1.0 * (self.Phi + self.maprotation + 90.0))
        self.Cosphi, self.Sinphi = np.cos(ang), np.sin(ang)
        self.Cosinc = np.cos(np.radians(self.Inc))
        bad = np.abs(self.Cosinc) <= 0.0001
        self.Inc[bad] = 89.
        self.Cosinc[bad] = math.cos(math.radians(89.))
        self.Annuli = np.stack([np.maximum(self.Radius - 0.5 * W, 0.), np.maximum(self.Radius + 0.5 * W, 0.)], 1)
        self.Rmax = max(self.Rmax, float(self.Annuli[:, 1].max()))
        segs = [0.0, 360.0]
        if segments is not None:
            istart = 0
            if not (segments[0] == 0.0) and (segments[1] == 360.0):
                self.Nseg += 1
                istart = 1
            segs = [0.0, 360.0] * self.Nseg
            for i in range(istart, self.Nseg):
                segs[2 * i] = _toangle_scalar(segments[2 * i])
                segs[2 * i + 1] = _toangle_scalar(segments[2 * i + 1])
        self.Segments = segs
        self.Position = [float(np.mean(rings.xpos)), float(np.mean(rings.ypos))]
        im = self.im
        if not (0 <= self.Position[0] < im.DimX() and 0 <= self.Position[1] < im.DimY()):
            raise ValueError("ELLPROF ERROR: The center of galaxy must be inside the map. Exiting...")
        h = im.Head()
        self.Dx = h.Cdelt(0) * arcsconv(h.Cunit(0))
        self.Dy = h.Cdelt(1) * arcsconv(h.Cunit(1))
        if self.Dx == 0 or self.Dy == 0:
            raise ValueError("ELLPROF ERROR: Null spacing. Check CUNIT and CDELT keywords in the header.")
        self.stepxy = [abs(self.Dx) / float(self.subpix[0]), abs(self.Dy) / float(self.subpix[1])]

    def setOptions(self, overlap, rng, subp):
        self.Overlap = overlap
        self.Range = list(rng)
        self.subpix = list(subp)
        self.stepxy = [abs(self.Dx) / float(self.subpix[0]), abs(self.Dy) / float(self.subpix[1])]

    # ------------------------------------------------------------------
    def RadialProfile(self):
        Nrad, Nseg = self.Nrad, self.Nseg
        im = self.im.array
        adx, ady = abs(self.Dx), abs(self.Dy)
        P = self.Position
        x0 = max(0, int(P[0] - self.Rmax / adx - 1))
        x1 = min(self.im.DimX() - 1, int(P[0] + self.Rmax / adx + 1))
        y0 = max(0, int(P[1] - self.Rmax / ady - 1))
        y1 = min(self.im.DimY() - 1, int(P[1] + self.Rmax / ady + 1))
        ys, xs = np.mgrid[y0:y1 + 1, x0:x1 + 1]
        xs, ys = xs.ravel(), ys.ravel()
        imval = im[ys, xs].astype(np.float32)
        Xr = (adx * (xs - P[0])).astype(np.float32)
        Yr = (ady * (ys - P[1])).astype(np.float32)

        def offsets(step, absd):
            o = []
            pos = np.float32(0.5 * (step - absd))
            while pos < np.float32(0.5 * absd):
                o.append(float(pos))
                pos = np.float32(pos + np.float32(step))
            return np.array(o)
        ox, oy = offsets(self.stepxy[0], adx), offsets(self.stepxy[1], ady)
        OX, OY = np.meshgrid(ox, oy, indexing="ij")
        posX = (Xr[:, None] + OX.ravel()[None]).astype(np.float32)        # (npix, nsub)
        posY = (Yr[:, None] + OY.ravel()[None]).astype(np.float32)
        npix, nsub = posX.shape
        rng = self.Range
        v = imval
        if rng[0] < rng[1]:
            valid = (v >= rng[0]) & (v <= rng[1])
        else:
            valid = (v < rng[1]) | (v > rng[0])
        valid &= ~np.isnan(v)

        # inside each ring (npix, nsub, nrad)
        cp, sp, ci = self.Cosphi, self.Sinphi, self.Cosinc
        Xx = posX[..., None] * cp - posY[..., None] * sp
        Yy = (posX[..., None] * sp + posY[..., None] * cp) / ci
        R2 = (Xx * Xx + Yy * Yy).astype(np.float32)
        A = self.Annuli.astype(np.float32)
        inside = (A[:, 0] * A[:, 0] <= R2) & (R2 < A[:, 1] * A[:, 1])
        overcount = inside.sum(axis=2)
        ivo = np.broadcast_to(v[:, None], (npix, nsub)).astype(np.float64)
        if self.Overlap:
            ivo = np.where(overcount > 1, ivo / np.maximum(overcount, 1), ivo)
        th0 = np.where((posX == 0) & (posY == 0), 0., np.degrees(np.arctan2(posY, posX)))
        self.Sum = np.zeros((Nrad, Nseg)); self.Sumsqr = np.zeros((Nrad, Nseg))
        self.Num = np.zeros((Nrad, Nseg), np.int64); self.Numblanks = np.zeros((Nrad, Nseg), np.int64)
        self.Datamin = np.full((Nrad, Nseg), FLT_MAX); self.Datamax = np.full((Nrad, Nseg), -FLT_MAX)
        self.Contrib = np.zeros((Nrad, Nseg), np.int64)
        self.Median = np.zeros((Nrad, Nseg)); self.MAD = np.zeros((Nrad, Nseg))
        validb = np.broadcast_to(valid[:, None], (npix, nsub))
        vb = np.broadcast_to(v[:, None], (npix, nsub))
        for r in range(Nrad):
            ins = inside[..., r]
            if not ins.any():
                continue
            theta = np.mod(th0 + 270.0 - self.Phi[r] - self.maprotation, 360.0)
            for s in range(Nseg):
                sel = ins & _in_segment(theta, self.Segments[2 * s], self.Segments[2 * s + 1])
                good = sel & validb
                vals = ivo[good]
                self.Sum[r, s] = vals.sum()
                self.Sumsqr[r, s] = np.sum(vals * vals)
                self.Num[r, s] = vals.size
                self.Numblanks[r, s] = int((sel & ~validb).sum())
                if vals.size:
                    self.Datamax[r, s] = max(float(vb[good].max()), -FLT_MAX)
                    self.Datamin[r, s] = min(float(vb[good].min()), FLT_MAX)
                    self.Median[r, s] = np.median(vals)
                    self.MAD[r, s] = findMADFM(vals, self.Median[r, s])
                self.Contrib[r, s] = int(good.any(axis=1).sum())
        subtot = self.subpix[0] * self.subpix[1]
        self.Sum /= subtot
        self.Area = self.Num / subtot
        with np.errstate(invalid="ignore", divide="ignore"):
            self.Mean = np.where(self.Num > 0, self.Sum / np.where(self.Area > 0, self.Area, 1), 0.)
            self.Var = np.where(self.Num > 1, (self.Sumsqr - self.Num * self.Mean ** 2) / np.maximum(self.Num - 1, 1), 0.)
        empty = self.Num == 0
        self.Datamin[empty] = 0
        self.Datamax[empty] = 0
        self.Blankarea = self.Numblanks / subtot
        pa = abs(self.Dx * self.Dy)
        area = self.Area * pa
        area_bl = self.Blankarea * pa
        with np.errstate(invalid="ignore", divide="ignore"):
            self.Surfdens = np.where(area == 0, 0., self.Sum / np.where(area == 0, 1, area))
            self.Surfdens_Bl = np.where(area + area_bl == 0, 0., self.Sum / np.where(area + area_bl == 0, 1, area + area_bl))

    # ---- getters
    def getNrad(self): return self.Nrad
    def getNseg(self): return self.Nseg
    def getRadius(self, i): return self.Radius[i]
    def getSum(self, i, j=0): return self.Sum[i, j]
    def getNum(self, i, j=0): return self.Num[i, j]
    def getMean(self, i, j=0): return self.Mean[i, j]
    def getMedian(self, i, j=0): return self.Median[i, j]
    def getRMS(self, i, j=0): return math.sqrt(abs(self.Var[i, j]))
    def getArea(self, i, j=0): return self.Area[i, j]
    def getSurfDens(self, i, j=0): return self.Surfdens[i, j]
    def getSurfDensFaceOn(self, i, j=0): return self.Cosinc[i] * self.Surfdens[i, j]
    def getSurfDens_Bl(self, i, j=0): return self.Surfdens_Bl[i, j]
    def getSurfDensFaceOn_Bl(self, i, j=0): return self.Cosinc[i] * self.Surfdens_Bl[i, j]

    def writeMap(self, fname):
        self.im.fitswrite_2d(fname)

    def printProfile(self, stream=None, seg=0):
        st = stream if stream is not None else sys.stdout
        h = self.im.Head()
        unit = h.Bunit().replace(" ", "")
        isJy = "jy" in unit.lower()
        m = 9
        pa = abs(self.Dx * self.Dy)
        st.write(f"# ELLPROF results for {h.Name()}\n")
        st.write(f"#\n# Map units: u = {unit}\n# Pixel area: {pa:.5f} arcs2\n#\n")
        st.write("# Columns 2-6  : ring statistics (sum, mean, median, standard deviation and median absolute deviation from the median) in map units (see above).\n")
        st.write("# Column  7    : effective number of valid pixels within the ring.\n")
        st.write("# Columns 8-10 : surface density, its error and face-on surface density (inclination corrected).\n")
        st.write("# Columns 11   : effective number of blank pixels within the ring.\n")
        st.write("# Columns 12-13: surface density and face-on surface density including area covered by blank pixels (total area: NPIX + NPIXBL).\n")
        if isJy:
            st.write("# Columns 14-15: (ONLY FOR HI DATA!) face-on (inclination corrected) number surface density and mass surface density.\n")
        cols = ["SUM", "MEAN", "MEDIAN", "STDDEV", "MAD", "NPIX", "SDENS", "ERR_SD", "SDENS_FO", "NPIXBL", "SDENSBL"]
        s = "#" + f"{'RADIUS':>{m - 1}} " + "".join(f"{c:>{m}} " for c in cols) + f"{'SDENSBL_FO':>{m + 1}} "
        if isJy:
            s += f"{'HIDENS_FO':>{m + 1}} {'HIMASSDENS':>{m + 1}} "
        st.write(s + "\n")
        units = ["u", "u", "u", "u", "u", "pix", "u/arcs2", "u/arcs2", "u/arcs2", "pix", "u/arcs2"]
        s = "#" + f"{'arcsec':>{m - 1}} " + "".join(f"{c:>{m}} " for c in units) + f"{'u/arcs2':>{m + 1}} "
        if isJy:
            s += f"{'1E20/cm2':>{m + 1}} {'Msun/pc2':>{m + 1}} "
        st.write(s + "\n#\n")
        h.Bmaj() * h.Bmin() * abs(arcsconv(h.Cunit(0)) * arcsconv(h.Cunit(1)))
        arctorad = 1 / 3600. * math.pi / 180.

        def e(x):
            return f"{x:>{m}.3e}"
        for i in range(self.Nrad):
            rms = math.sqrt(abs(self.Var[i, seg]))
            line = (f"{self.Radius[i]:>{m}.3f} " + e(self.Sum[i, seg]) + " " + e(self.Mean[i, seg]) + " "
                    + e(self.Median[i, seg]) + " " + e(rms) + " " + e(self.MAD[i, seg]) + " " + e(self.Area[i, seg]) + " "
                    + e(self.Surfdens[i, seg]) + " " + e(rms / pa) + " " + e(self.getSurfDensFaceOn(i, seg)) + " "
                    + e(self.Blankarea[i, seg]) + " " + e(self.Surfdens_Bl[i, seg]) + " "
                    + f"{self.getSurfDensFaceOn_Bl(i, seg):>{m + 1}.3e} ")
            if isJy:
                nd, mds = self.hiDensities(self.getSurfDensFaceOn(i, seg))
                line += f"{nd:>{m + 1}.3e} {mds:>{m + 1}.3e} "
            st.write(line + "\n")

    def hiDensities(self, sdfo):
        """(ONLY FOR HI DATA, map in Jy) the face-on number (1E20/cm2) and mass (Msun/pc2) surface densities of a
        face-on surface density sdfo in the units of the map per arcsec2."""
        h = self.im.Head()
        unit = h.Bunit().replace(" ", "")
        arctorad = 1 / 3600. * math.pi / 180.
        # (2.356E5 D^2 S Msun, the constant of MASSDENSMAP; C++ used 2.36E5, and Jy for mJy and uJy)
        mds = 2.356E-07 * sdfo / (arctorad * arctorad)
        mds *= _jy_prefix(unit.lower())
        if "/b" in unit.lower() and h.BeamArea() != 0:
            mds /= h.BeamArea()
        nd = mds / (8.41185687E-02 * 3.0857 * 3.0857)
        return nd, mds

    def deconvolvedText(self, dp, seg=0, task="ELLPROF", obsfile="_ellprof.txt", note=""):
        """The deconvolved profile dp (see profile.ellprof_deconvolved) as written in <prefix>_ellprof_deconvolved.txt
        (task ELLPROF) or <prefix>_densprof_deconvolved.txt (3DFIT, obsfile _densprof.txt), with the observed
        face-on surface density of segment seg (column SDENS_FO of obsfile). note: lines added to the header."""
        h = self.im.Head()
        unit = h.Bunit().replace(" ", "")
        isJy = "jy" in unit.lower()
        m = 12
        s = (f"# {task} deconvolved profile for {h.Name()}\n#\n" + note + f"# Map units: u = {unit}\n"
             "# SDENS_FO: the face-on surface density whose beam-convolved model reproduces the total intensity map\n"
             f"#   (beam {dp['beam']:.2f} arcsec), with its 1-sigma errors E_LO, E_HI from the noise "
             f"({dp['nmc']} realisations of the " + ("scatter of the residuals" if dp.get("map2d") else "noise")
             + "; the rings are fixed);\n"
             f"# SDENS_FO_OBS: the observed face-on surface density (column SDENS_FO of {obsfile}), smoothed by "
             "the beam;\n# DENS: " + ("the face-on surface density in map units per pixel.\n" if dp.get("map2d") else
                                      "the densities of the model, as DENS of GALMOD.\n"))
        if isJy:
            s += ("# HIMASSDENS, HIMASSDENS_OBS: (ONLY FOR HI DATA!) the face-on mass surface densities of SDENS_FO and "
                  "SDENS_FO_OBS.\n")
        cols = ["SDENS_FO", "E_LO", "E_HI", "SDENS_FO_OBS", "DENS"]
        units = ["u/arcs2", "u/arcs2", "u/arcs2", "u/arcs2", ""]
        if isJy:
            cols += ["HIMASSDENS", "E_LO", "E_HI", "HIMASSDENS_OBS"]
            units += ["Msun/pc2"] * 4
        s += "#" + f"{'RADIUS':>{m - 1}} " + "".join(f"{c:>{m}} " for c in cols) + "\n"
        s += "#" + f"{'arcsec':>{m - 1}} " + "".join(f"{c:>{m}} " for c in units) + "\n#\n"
        for i in range(len(dp["radii"])):
            obs = self.getSurfDensFaceOn(i, seg)
            vals = [dp["sdens_fo"][i], dp["err_lo"][i], dp["err_hi"][i], obs, dp["dens"][i]]
            if isJy:
                vals += [self.hiDensities(v)[1] for v in (dp["sdens_fo"][i], dp["err_lo"][i], dp["err_hi"][i], obs)]
            s += f"{dp['radii'][i]:>{m}.3f} " + "".join(f"{v:>{m}.4e} " for v in vals) + "\n"
        return s

    def deconvolvedTable(self, dp, seg=0):
        """The deconvolved profile dp as an astropy Table at full precision (columns radii, sdens_fo, e_sdens_fo_lo,
        e_sdens_fo_hi, sdens_fo_obs, dens and, for HI data in Jy, himassdens, e_himassdens_lo, e_himassdens_hi,
        himassdens_obs)."""
        from astropy.table import Table
        import astropy.units as u
        h = self.im.Head()
        isJy = "jy" in h.Bunit().replace(" ", "").lower()
        nr = len(dp["radii"])
        obs = np.array([self.getSurfDensFaceOn(i, seg) for i in range(nr)])
        t = Table()
        t["radii"] = np.asarray(dp["radii"], float) * u.arcsec
        for k, v in (("sdens_fo", dp["sdens_fo"]), ("e_sdens_fo_lo", dp["err_lo"]), ("e_sdens_fo_hi", dp["err_hi"]),
                     ("sdens_fo_obs", obs), ("dens", dp["dens"])):
            t[k] = np.asarray(v, float)
        if isJy:
            for k, v in (("himassdens", dp["sdens_fo"]), ("e_himassdens_lo", dp["err_lo"]),
                         ("e_himassdens_hi", dp["err_hi"]), ("himassdens_obs", obs)):
                t[k] = np.array([self.hiDensities(x)[1] for x in np.asarray(v, float)]) * u.Msun / u.pc ** 2
        return t
