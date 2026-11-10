"""
Bi-conical wind models whose emission is a set of non-negative amplitudes on cells of the cones (WindModel):
the model engine of BAYESIANWIND (bbarolo.BayesianWind) and of GALWIND WTYPE 2.

Geometry and conventions are those of GALWIND WTYPE 1 (galmod._galmod_wind). The two cones share the apex
(XPOS, YPOS); their axis is the rotation axis z of a disk with inclination INC and position angle PA, so
that the projected axis lies along PA - 90. A point at distance r (arcsec) from the apex, at polar angle
theta from the axis of its cone and azimuth phi, has
    R = r sin(theta),  z = c r cos(theta),  x = R cos(phi),  y = R sin(phi)
with c = +1 for the cone along +z (receding for INC < 90) and c = -1 for the other (approaching); on the sky
    y_sky = y cos(INC) - z sin(INC)   (x along PA, as the major axis of GALMOD)
and its line-of-sight velocity is
    v_los = VSYS + v_r(r) (c cos(theta) cos(INC) + sin(theta) sin(phi) sin(INC)) + v_rot(r) cos(phi) sin(INC),
for a radial (outflow if v_r > 0) wind VWIND rotating around the axis with VROT, both functions of r
interpolated linearly between the shells (as GALWIND), with a Gaussian line of dispersion VDISP(r) plus the
instrumental dispersion.

The emission: a fixed grid of cells in the frame of the wind (WindCells), each cone with its own cells:
radial bins, polar bins from the axis to the largest half opening angle, azimuthal sectors. The current
cone (inner and outer half opening angles OPENIN/2 and OPENANG/2, outer radius CONELEN) clips the cells: the
quadrature points of a cell fill its clipped part, so the model moves continuously with the geometry and a
cell outside the cone has no volume. The model is linear in the amplitudes of the cells (emissivity per
unit volume): M = A(geometry, kinematics) . amplitudes. The columns of A, each point deposited with its line
integrated over the channels (as the ANALYTIC models of GALMOD) and with the beam evaluated at its exact
position (no pixelisation of the positions), are kept as dense boxes of the cube, one per cell (Boxes):
from them the normal equations of the fit (AtWA, AtWd) and the model cube.
"""

import math
import numpy as np
import numba as nb

from .bbutils import BBaroloError, arcsconv, getCenterCoordinates
from .galmod import GalmodSetup, _lround, _velgrid, _line_profile, _PHI_TAB

PSF_CUTOFF = 1.0e-4                 # (as CUTOFFRATIO of SMOOTH: the beam is zero below this fraction of its peak)
MAX_POINTS = 2_000_000              # (the most quadrature points: beyond, the cones are sampled more coarsely)


# --------------------------------------------------------------------------
# Parameters and cells
# --------------------------------------------------------------------------
class WindPars:
    """The parameters of a wind model: centre (xpos, ypos: pixels), vsys (km/s), inc and pa (degrees, as GALWIND),
    openang and openin (full outer and inner opening angles, degrees), conelen (outer radius, arcsec), and per
    shell (radii: arcsec) vwind, vdisp and vrot (km/s)."""

    NAMES = ("xpos", "ypos", "vsys", "inc", "pa", "openang", "openin", "conelen")
    SHELL = ("vwind", "vdisp", "vrot")

    def __init__(self, radii, vwind, vdisp, vrot=0., xpos=0., ypos=0., vsys=0., inc=60., pa=0., openang=60.,
                 openin=0., conelen=None):
        self.radii = np.atleast_1d(np.asarray(radii, dtype=np.float64))
        n = self.radii.size
        if n == 0 or np.any(np.diff(self.radii) <= 0):
            raise BBaroloError("WindModel: the radii of the shells must be increasing")
        for name, v in (("vwind", vwind), ("vdisp", vdisp), ("vrot", vrot)):
            a = np.asarray(v, dtype=np.float64).ravel()
            if a.size == 1:
                a = np.full(n, a[0])
            if a.size != n:
                raise BBaroloError(f"WindModel: {name} has {a.size} values, not one or {n} (one per shell)")
            setattr(self, name, a)
        self.xpos, self.ypos, self.vsys = float(xpos), float(ypos), float(vsys)
        self.inc, self.pa = float(inc), float(pa)
        self.openang, self.openin = float(openang), float(openin)
        self.conelen = float(self.radii[-1] if conelen is None else conelen)

    def copy(self):
        p = WindPars.__new__(WindPars)
        p.__dict__ = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in self.__dict__.items()}
        return p

    def check(self):
        """Raise a BBaroloError if the parameters are not a valid wind."""
        if not (0 <= self.openin < self.openang <= 180.):
            raise BBaroloError(f"WindModel: the opening angles must be 0 <= OPENIN < OPENANG <= 180 "
                               f"(OPENIN={self.openin:g}, OPENANG={self.openang:g})")
        if not self.conelen > 0:
            raise BBaroloError(f"WindModel: CONELEN must be positive (CONELEN={self.conelen:g})")
        if np.any(self.vdisp < 0):
            raise BBaroloError("WindModel: negative velocity dispersion")

    def __repr__(self):
        g = ", ".join(f"{k}={getattr(self, k):g}" for k in self.NAMES)
        return f"<WindPars {g}, {self.radii.size} shells>"


class WindCells:
    """The fixed grid of cells of the emission in the frame of the wind.

    redges: the edges of the radial bins (arcsec, increasing, from 0 or an inner radius); ntheta polar bins of
    the half opening angle from 0 to thmax (degrees: the largest half opening angle of the cone); nphi
    azimuthal sectors; cones: +1 (the cone along +z, receding for INC < 90), -1 (the other) or both (1, -1).
    The cells are ordered by cone, radius, polar angle and azimuth: cell = ((icone*nr + ir)*ntheta + it)*nphi + ip.
    """

    def __init__(self, redges, ntheta=1, nphi=1, thmax=90., cones=(1, -1)):
        self.redges = np.asarray(redges, dtype=np.float64).ravel()
        if self.redges.size < 2 or np.any(np.diff(self.redges) <= 0) or self.redges[0] < 0:
            raise BBaroloError("WindCells: the radial edges must be increasing and non-negative")
        self.ntheta, self.nphi = int(ntheta), int(nphi)
        if self.ntheta < 1 or self.nphi < 1:
            raise BBaroloError("WindCells: ntheta and nphi must be at least 1")
        if not 0 < thmax <= 90.:
            raise BBaroloError(f"WindCells: the largest half opening angle must be in (0, 90] degrees, not {thmax:g}")
        self.thmax = float(thmax)
        self.cones = tuple(int(c) for c in np.atleast_1d(cones))
        if not self.cones or any(c not in (1, -1) for c in self.cones) or len(set(self.cones)) != len(self.cones):
            raise BBaroloError("WindCells: cones must be 1, -1 or both")
        self.nr = self.redges.size - 1
        tedges = np.linspace(0., math.radians(self.thmax), self.ntheta + 1)
        pedges = np.linspace(0., 2 * math.pi, self.nphi + 1)
        rows = []
        for c in self.cones:
            for ir in range(self.nr):
                for it in range(self.ntheta):
                    for ip in range(self.nphi):
                        rows.append((c, self.redges[ir], self.redges[ir + 1], tedges[it], tedges[it + 1],
                                     pedges[ip], pedges[ip + 1]))
        self.cells = np.ascontiguousarray(rows, dtype=np.float64)
        self.ijk = np.array([(ic, ir, it, ip) for ic in range(len(self.cones)) for ir in range(self.nr)
                             for it in range(self.ntheta) for ip in range(self.nphi)], dtype=np.int64).reshape(-1, 4)
        self.qn = None
        self.qoff = None

    @classmethod
    def resolved(cls, rmax, thmax, size, cones=(1, -1), rmin=0., minphi=1):
        """A grid of cells about size (arcsec, e.g. the beam) across: radial bins of width ~size from rmin to rmax,
        and in each radial bin polar bins (from 0 to thmax, degrees) and azimuthal sectors as many as fit with
        that size at the outer radius of the bin (at least minphi sectors), so that the cells near the apex are
        not smaller than the resolution."""
        nr = max(1, int(round((rmax - rmin) / size)))
        redges = np.linspace(rmin, rmax, nr + 1)
        self = cls(redges, 1, 1, thmax=thmax, cones=cones)
        th = math.radians(thmax)
        rows, ijk = [], []
        for ic, c in enumerate(self.cones):
            for ir in range(nr):
                ro = redges[ir + 1]
                nt = max(1, int(round(ro * th / size)))
                tedges = np.linspace(0., th, nt + 1)
                for it in range(nt):
                    npp = max(int(minphi), int(round(2 * math.pi * ro * math.sin(tedges[it + 1]) / size)))
                    pedges = np.linspace(0., 2 * math.pi, npp + 1)
                    for ip in range(npp):
                        rows.append((c, redges[ir], redges[ir + 1], tedges[it], tedges[it + 1], pedges[ip],
                                     pedges[ip + 1]))
                        ijk.append((ic, ir, it, ip))
        self.cells = np.ascontiguousarray(rows, dtype=np.float64)
        self.ijk = np.array(ijk, dtype=np.int64)
        self.ntheta = self.nphi = None               # (they depend on the radial bin)
        return self

    @property
    def ncell(self):
        return self.cells.shape[0]

    def index(self, cone, ir, it=0, ip=0):
        """The index of the cell (cone, radial bin, polar bin, azimuthal sector)."""
        ic = self.cones.index(cone)
        hit = np.flatnonzero((self.ijk[:, 0] == ic) & (self.ijk[:, 1] == ir) & (self.ijk[:, 2] == it)
                             & (self.ijk[:, 3] == ip))
        if hit.size == 0:
            raise BBaroloError(f"WindCells: no cell ({cone}, {ir}, {it}, {ip})")
        return int(hit[0])

    def set_quadrature(self, spacing, vmax, sigmin, cap=16, maxpoints=None):
        """The number of quadrature points of each cell in r, theta and phi: points at most spacing (arcsec) apart
        on the sky (as many as a cell needs: one cell can span a whole cone), and with line-of-sight velocities at
        most half the narrowest line dispersion sigmin (km/s) apart for winds up to vmax (km/s: the largest |VWIND| +
        |VROT|), at most cap per dimension for the velocities. With maxpoints, the spacing is enlarged until the cells
        have at most maxpoints points. Sets qn, qoff and spacing (the spacing used, arcsec)."""
        c = self.cells
        dr = c[:, 2] - c[:, 1]
        dth = c[:, 4] - c[:, 3]
        dph = c[:, 6] - c[:, 5]
        st = np.sin(c[:, 4])
        dv = max(0.5 * float(sigmin), 1e-3)
        nt_v = np.minimum(np.ceil(vmax * dth / dv), cap)
        np_v = np.minimum(np.ceil(vmax * st * dph / dv), cap)
        sp = float(spacing)
        while True:
            nr = np.ceil(dr / sp)
            nt = np.maximum(np.ceil(c[:, 2] * dth / sp), nt_v)
            npp = np.maximum(np.ceil(c[:, 2] * st * dph / sp), np_v)
            qn = np.maximum(np.vstack([nr, nt, npp]).T, 1).astype(np.int64)
            tot = int(np.sum(np.prod(qn, axis=1)))
            if maxpoints is None or tot <= maxpoints:
                break
            sp *= max(1.02, (tot / maxpoints) ** (1. / 3.))
        self.spacing = sp
        self.qn = np.ascontiguousarray(qn)
        self.qoff = np.zeros(self.ncell + 1, dtype=np.int64)
        np.cumsum(np.prod(qn, axis=1), out=self.qoff[1:])
        return self

    @property
    def npoints(self):
        return int(self.qoff[-1]) if self.qoff is not None else 0

    def centres(self):
        """(cone, r, theta, phi) of the centres of the cells (arcsec, degrees)."""
        c = self.cells
        return (c[:, 0].astype(int), 0.5 * (c[:, 1] + c[:, 2]), np.degrees(0.5 * (c[:, 3] + c[:, 4])),
                np.degrees(0.5 * (c[:, 5] + c[:, 6])))


# --------------------------------------------------------------------------
# numba kernels
# --------------------------------------------------------------------------
@nb.njit(inline="always", cache=True)
def _interp(r, xs, ys):
    """ys(xs) at r, linear, constant beyond the ends (xs increasing)."""
    n = xs.shape[0]
    if n == 1 or r <= xs[0]:
        return ys[0]
    if r >= xs[n - 1]:
        return ys[n - 1]
    j = np.searchsorted(xs, r)
    t = (r - xs[j - 1]) / (xs[j] - xs[j - 1])
    return ys[j - 1] + t * (ys[j] - ys[j - 1])


@nb.njit(cache=True, nogil=True)
def _wind_points(cells, qn, qoff, radii, vw, vd, vr, x0, y0, vsys, sinc, cinc, spa, cpa, cd0, cd1, th_in, th_out,
                 rmax, sig_instr, gx, gy, vel, sig, w, rr, vol, rbar):
    """The quadrature points of the cells clipped by the cone (th_in <= theta <= th_out, r <= rmax; radians and
    arcsec): sky pixels (gx, gy), line-of-sight velocity and line dispersion (m/s, the instrumental dispersion
    included), weight (arcsec^3: the clipped volume of the cell shared among its points, 0 for an empty cell)
    and radius; vol and rbar: the clipped volume and the mean radius of each cell."""
    for k in range(cells.shape[0]):
        cs = cells[k, 0]
        r0 = cells[k, 1]
        r1 = min(cells[k, 2], rmax)
        t0 = max(cells[k, 3], th_in)
        t1 = min(cells[k, 4], th_out)
        p0, p1 = cells[k, 5], cells[k, 6]
        nqr, nqt, nqp = qn[k, 0], qn[k, 1], qn[k, 2]
        i0, i1 = qoff[k], qoff[k + 1]
        if r1 <= r0 or t1 <= t0:
            for i in range(i0, i1):
                gx[i] = -1e9
                gy[i] = -1e9
                vel[i] = 0.
                sig[i] = 1.
                w[i] = 0.
                rr[i] = r0
            vol[k] = 0.
            rbar[k] = 0.5 * (r0 + r1) if r1 > r0 else r0
            continue
        dr = (r1 - r0) / nqr
        dt = (t1 - t0) / nqt
        dp = (p1 - p0) / nqp
        i = i0
        wsum = 0.
        for a in range(nqr):
            r = r0 + (a + 0.5) * dr
            vrad = _interp(r, radii, vw) * 1000.
            disp = _interp(r, radii, vd) * 1000.
            vrot = _interp(r, radii, vr) * 1000.
            s = math.sqrt(disp * disp + sig_instr * sig_instr)
            for b in range(nqt):
                th = t0 + (b + 0.5) * dt
                st, ct = math.sin(th), math.cos(th)
                for c in range(nqp):
                    ph = p0 + (c + 0.5) * dp
                    sp, cp = math.sin(ph), math.cos(ph)
                    R = r * st
                    z = cs * r * ct
                    x = R * cp
                    ys = R * sp * cinc - z * sinc
                    gx[i] = x0 + (x * spa - ys * cpa) / cd0
                    gy[i] = y0 + (x * cpa + ys * spa) / cd1
                    vel[i] = vsys + vrad * (cs * ct * cinc + st * sp * sinc) + vrot * cp * sinc
                    sig[i] = s
                    w[i] = r * r * st
                    rr[i] = r
                    wsum += w[i]
                    i += 1
        V = (r1 ** 3 - r0 ** 3) / 3. * (math.cos(t0) - math.cos(t1)) * (p1 - p0)
        vol[k] = V
        f = V / wsum if wsum > 0 else 0.
        rs = 0.
        for i in range(i0, i1):
            w[i] *= f
            rs += w[i] * rr[i]
        rbar[k] = rs / V if V > 0 else 0.5 * (r0 + r1)


@nb.njit(cache=True, nogil=True)
def _raw_boxes(qoff, gx, gy, vel, sig, w, lim, chmin, axtyp, velsys, lfreq0, crval3, cdelt3, crpix3, rboxes):
    """rboxes[k] = (x0, x1, y0, y1, k0, k1), the pixels and channels [lo, hi) of the bilinear deposit of the points
    of cell k (lines within 5 dispersions), within lim (the same six numbers); x1 = x0 if none."""
    nlines = lfreq0.shape[0]
    for k in range(qoff.shape[0] - 1):
        bx0, bx1, by0, by1, bk0, bk1 = 1 << 30, -(1 << 30), 1 << 30, -(1 << 30), 1 << 30, -(1 << 30)
        for i in range(qoff[k], qoff[k + 1]):
            if w[i] <= 0. or not (abs(gx[i]) < 1e8 and abs(gy[i]) < 1e8):
                continue
            ix, iy = int(math.floor(gx[i])), int(math.floor(gy[i]))
            if ix < bx0: bx0 = ix
            if ix + 2 > bx1: bx1 = ix + 2
            if iy < by0: by0 = iy
            if iy + 2 > by1: by1 = iy + 2
            K = int(5.0 * sig[i] / chmin) + 2
            for nl in range(nlines):
                kc = _lround(_velgrid(vel[i], axtyp, velsys, lfreq0[nl], crval3, cdelt3, crpix3))
                if kc - K < bk0: bk0 = kc - K
                if kc + K + 1 > bk1: bk1 = kc + K + 1
        bx0, bx1 = max(bx0, lim[0]), min(bx1, lim[1])
        by0, by1 = max(by0, lim[2]), min(by1, lim[3])
        bk0, bk1 = max(bk0, lim[4]), min(bk1, lim[5])
        if bx1 <= bx0 or by1 <= by0 or bk1 <= bk0:
            bx0 = bx1 = lim[0]
            by0 = by1 = lim[2]
            bk0 = bk1 = lim[4]
        rboxes[k, 0], rboxes[k, 1], rboxes[k, 2] = bx0, bx1, by0
        rboxes[k, 3], rboxes[k, 4], rboxes[k, 5] = by1, bk0, bk1


@nb.njit(cache=True, nogil=True)
def _deposit_raw(rbuf, roffs, rboxes, qoff, gx, gy, vel, sig, w, vedges, relint, ones, ptab, chmin, axtyp, velsys,
                 lfreq0, crval3, cdelt3, crpix3, nz):
    """Deposit the points of each cell k in its box (rbuf[roffs[k]:roffs[k+1]], layout (channel, y, x)): each point
    with its line integrated over the channels, for each line, on the four nearest pixels (bilinear weights)."""
    nlines = lfreq0.shape[0]
    prof = np.empty(nz)
    for k in range(rboxes.shape[0]):
        x0, x1, y0, y1, k0, k1 = rboxes[k, 0], rboxes[k, 1], rboxes[k, 2], rboxes[k, 3], rboxes[k, 4], rboxes[k, 5]
        if x1 <= x0 or y1 <= y0 or k1 <= k0:
            continue
        bx, by = x1 - x0, y1 - y0
        base = roffs[k]
        for i in range(qoff[k], qoff[k + 1]):
            if w[i] <= 0. or not (abs(gx[i]) < 1e8 and abs(gy[i]) < 1e8):
                continue
            ix, iy = int(math.floor(gx[i])), int(math.floor(gy[i]))
            fx, fy = gx[i] - ix, gy[i] - iy
            s = sig[i]
            if s < 1e-6 * chmin:
                s = 1e-6 * chmin
            K = int(5.0 * s / chmin) + 2
            for nl in range(nlines):
                kc = _lround(_velgrid(vel[i], axtyp, velsys, lfreq0[nl], crval3, cdelt3, crpix3))
                kk0, nk = _line_profile(prof, vel[i], s, vedges[nl], ones, ptab, kc, K, nz)
                amp = w[i] * relint[nl]
                for j in range(nk):
                    kz = kk0 + j
                    if kz < k0 or kz >= k1:
                        continue
                    pv = amp * prof[j]
                    if pv == 0.:
                        continue
                    for dy in range(2):
                        yy = iy + dy
                        if yy < y0 or yy >= y1:
                            continue
                        wy = fy if dy == 1 else 1. - fy
                        row = base + ((kz - k0) * by + (yy - y0)) * bx - x0
                        for dx in range(2):
                            xx = ix + dx
                            if xx < x0 or xx >= x1:
                                continue
                            wx = fx if dx == 1 else 1. - fx
                            rbuf[row + xx] += pv * wx * wy


@nb.njit(cache=True, nogil=True)
def _convolve_boxes(rbuf, roffs, rboxes, kern, kx, ky, separable, buf, offs, boxes):
    """Convolve each box of rbuf with the beam kern (2D, (2hy+1, 2hx+1); separable: the 1D kernels kx and ky) into
    the boxes of buf (the boxes of rbuf grown by the kernel and clipped)."""
    hy, hx = kern.shape[0] // 2, kern.shape[1] // 2
    for k in range(rboxes.shape[0]):
        rx0, rx1, ry0 = rboxes[k, 0], rboxes[k, 1], rboxes[k, 2]
        ry1, rk0, rk1 = rboxes[k, 3], rboxes[k, 4], rboxes[k, 5]
        x0, x1, y0, y1, k0, k1 = boxes[k, 0], boxes[k, 1], boxes[k, 2], boxes[k, 3], boxes[k, 4], boxes[k, 5]
        if rx1 <= rx0 or ry1 <= ry0 or rk1 <= rk0 or x1 <= x0 or y1 <= y0 or k1 <= k0:
            continue
        rbx, rby = rx1 - rx0, ry1 - ry0
        bx, by = x1 - x0, y1 - y0
        if separable:
            tmp = np.zeros((ry1 - ry0, bx))          # (one channel: the x pass, then the y pass)
            for kz in range(max(rk0, k0), min(rk1, k1)):
                tmp[:, :] = 0.
                for y in range(ry0, ry1):
                    rrow = roffs[k] + ((kz - rk0) * rby + (y - ry0)) * rbx - rx0
                    for x in range(rx0, rx1):
                        v = rbuf[rrow + x]
                        if v == 0.:
                            continue
                        for jx in range(-hx, hx + 1):
                            xx = x + jx
                            if xx < x0 or xx >= x1:
                                continue
                            tmp[y - ry0, xx - x0] += v * kx[jx + hx]
                for y in range(ry0, ry1):
                    for jy in range(-hy, hy + 1):
                        yy = y + jy
                        if yy < y0 or yy >= y1:
                            continue
                        f = ky[jy + hy]
                        row = offs[k] + ((kz - k0) * by + (yy - y0)) * bx
                        for xx in range(bx):
                            buf[row + xx] += f * tmp[y - ry0, xx]
        else:
            for kz in range(max(rk0, k0), min(rk1, k1)):
                for y in range(ry0, ry1):
                    rrow = roffs[k] + ((kz - rk0) * rby + (y - ry0)) * rbx - rx0
                    for x in range(rx0, rx1):
                        v = rbuf[rrow + x]
                        if v == 0.:
                            continue
                        for jy in range(-hy, hy + 1):
                            yy = y + jy
                            if yy < y0 or yy >= y1:
                                continue
                            row = offs[k] + ((kz - k0) * by + (yy - y0)) * bx - x0
                            for jx in range(-hx, hx + 1):
                                xx = x + jx
                                if xx < x0 or xx >= x1:
                                    continue
                                buf[row + xx] += v * kern[jy + hy, jx + hx]


@nb.njit(cache=True, nogil=True)
def _box_dot(bufa, offa, boxa, bufb, offb, boxb, wm):
    """Sum over the intersection of the boxes boxa and boxb of bufa*bufb*wm (wm: (nz, ny, nx) weights)."""
    x0, x1 = max(boxa[0], boxb[0]), min(boxa[1], boxb[1])
    y0, y1 = max(boxa[2], boxb[2]), min(boxa[3], boxb[3])
    k0, k1 = max(boxa[4], boxb[4]), min(boxa[5], boxb[5])
    if x1 <= x0 or y1 <= y0 or k1 <= k0:
        return 0.
    bxa, bya = boxa[1] - boxa[0], boxa[3] - boxa[2]
    bxb, byb = boxb[1] - boxb[0], boxb[3] - boxb[2]
    s = 0.
    for kz in range(k0, k1):
        for y in range(y0, y1):
            ra = offa + ((kz - boxa[4]) * bya + (y - boxa[2])) * bxa - boxa[0]
            rb = offb + ((kz - boxb[4]) * byb + (y - boxb[2])) * bxb - boxb[0]
            for x in range(x0, x1):
                va = bufa[ra + x]
                if va != 0.:
                    s += va * bufb[rb + x] * wm[kz, y, x]
    return s


@nb.njit(cache=True, nogil=True)
def _normal(buf, offs, boxes, wd, wm, G, b):
    """The normal equations of the boxes: G[k, l] = sum(A_k A_l wm), b[k] = sum(A_k wd) (wd = wm * data)."""
    n = boxes.shape[0]
    for k in range(n):
        x0, x1, y0, y1, k0, k1 = boxes[k, 0], boxes[k, 1], boxes[k, 2], boxes[k, 3], boxes[k, 4], boxes[k, 5]
        b[k] = 0.
        G[k, k] = 0.
        if x1 <= x0 or y1 <= y0 or k1 <= k0:
            continue
        bx, by = x1 - x0, y1 - y0
        s, s2 = 0., 0.
        for kz in range(k0, k1):
            for y in range(y0, y1):
                row = offs[k] + ((kz - k0) * by + (y - y0)) * bx - x0
                for x in range(x0, x1):
                    v = buf[row + x]
                    if v != 0.:
                        s += v * wd[kz, y, x]
                        s2 += v * v * wm[kz, y, x]
        b[k] = s
        G[k, k] = s2
    for k in range(n):
        for l in range(k + 1, n):
            v = _box_dot(buf, offs[k], boxes[k], buf, offs[l], boxes[l], wm)
            G[k, l] = v
            G[l, k] = v


@nb.njit(cache=True, nogil=True)
def _voxel_lists(buf, offs, boxes, wm, lim, rel):
    """The values of the boxes above rel times the peak of their box, with wm != 0, grouped by voxel (of the box lim =
    (x0, x1, y0, y1, k0, k1)): start (one more than the voxels of lim), cell and value of each entry, the cells of a
    voxel in increasing order."""
    lx, ly = lim[1] - lim[0], lim[3] - lim[2]
    nvox = lx * ly * (lim[5] - lim[4])
    start = np.zeros(nvox + 1, dtype=np.int64)
    thr = np.zeros(boxes.shape[0])
    for k in range(boxes.shape[0]):
        m = 0.
        for j in range(offs[k], offs[k + 1]):
            a = abs(buf[j])
            if a > m:
                m = a
        thr[k] = rel * m
    for k in range(boxes.shape[0]):
        x0, x1, y0, y1, k0, k1 = boxes[k, 0], boxes[k, 1], boxes[k, 2], boxes[k, 3], boxes[k, 4], boxes[k, 5]
        if x1 <= x0 or y1 <= y0 or k1 <= k0:
            continue
        bx, by = x1 - x0, y1 - y0
        for kz in range(k0, k1):
            for y in range(y0, y1):
                row = offs[k] + ((kz - k0) * by + (y - y0)) * bx - x0
                vrow = ((kz - lim[4]) * ly + (y - lim[2])) * lx - lim[0]
                for x in range(x0, x1):
                    v = buf[row + x]
                    if v != 0. and abs(v) >= thr[k] and wm[kz, y, x] != 0.:
                        start[vrow + x + 1] += 1
    for i in range(nvox):
        start[i + 1] += start[i]
    ncell = np.empty(start[nvox], dtype=np.int32)
    val = np.empty(start[nvox], dtype=np.float64)
    fill = start[:nvox].copy()
    for k in range(boxes.shape[0]):
        x0, x1, y0, y1, k0, k1 = boxes[k, 0], boxes[k, 1], boxes[k, 2], boxes[k, 3], boxes[k, 4], boxes[k, 5]
        if x1 <= x0 or y1 <= y0 or k1 <= k0:
            continue
        bx, by = x1 - x0, y1 - y0
        for kz in range(k0, k1):
            for y in range(y0, y1):
                row = offs[k] + ((kz - k0) * by + (y - y0)) * bx - x0
                vrow = ((kz - lim[4]) * ly + (y - lim[2])) * lx - lim[0]
                for x in range(x0, x1):
                    v = buf[row + x]
                    if v != 0. and abs(v) >= thr[k] and wm[kz, y, x] != 0.:
                        j = fill[vrow + x]
                        ncell[j] = k
                        val[j] = v
                        fill[vrow + x] = j + 1
    return start, ncell, val


@nb.njit(cache=True, nogil=True)
def _normal_lists(start, ncell, val, wd, wm, lim, G, b):
    """G[k, l] = sum(A_k A_l wm), b[k] = sum(A_k wd) from the voxel lists of _voxel_lists."""
    lx, ly = lim[1] - lim[0], lim[3] - lim[2]
    nvox = start.shape[0] - 1
    for i in range(nvox):
        j0, j1 = start[i], start[i + 1]
        if j1 == j0:
            continue
        x = lim[0] + i % lx
        y = lim[2] + (i // lx) % ly
        kz = lim[4] + i // (lx * ly)
        w = wm[kz, y, x]
        d = wd[kz, y, x]
        for p in range(j0, j1):
            kp = ncell[p]
            ap = val[p]
            b[kp] += ap * d
            wa = w * ap
            for q in range(p, j1):
                G[kp, ncell[q]] += wa * val[q]
    n = G.shape[0]
    for k in range(n):
        for l in range(k + 1, n):
            G[l, k] = G[k, l]


@nb.njit(cache=True, nogil=True)
def _cross_lists(starta, cella, vala, startb, cellb, valb, wm, lim, Gab):
    """Gab[k, l] = sum(A_k B_l wm) from the voxel lists of two sets of columns (the same lim)."""
    lx, ly = lim[1] - lim[0], lim[3] - lim[2]
    for i in range(starta.shape[0] - 1):
        a0, a1 = starta[i], starta[i + 1]
        b0, b1 = startb[i], startb[i + 1]
        if a1 == a0 or b1 == b0:
            continue
        x = lim[0] + i % lx
        y = lim[2] + (i // lx) % ly
        kz = lim[4] + i // (lx * ly)
        w = wm[kz, y, x]
        for p in range(a0, a1):
            wa = w * vala[p]
            for q in range(b0, b1):
                Gab[cella[p], cellb[q]] += wa * valb[q]


@nb.njit(cache=True, nogil=True)
def _normal_cross(bufa, offa, boxa, bufb, offb, boxb, wm, Gab):
    """Gab[k, l] = sum(A_k B_l wm) for the boxes of two sets."""
    for k in range(boxa.shape[0]):
        for l in range(boxb.shape[0]):
            Gab[k, l] = _box_dot(bufa, offa[k], boxa[k], bufb, offb[l], boxb[l], wm)


@nb.njit(cache=True, nogil=True)
def _scatter(out, buf, offs, boxes, amp):
    """out (nz, ny, nx) += sum_k amp[k] A_k."""
    for k in range(boxes.shape[0]):
        a = amp[k]
        if a == 0.:
            continue
        x0, x1, y0, y1, k0, k1 = boxes[k, 0], boxes[k, 1], boxes[k, 2], boxes[k, 3], boxes[k, 4], boxes[k, 5]
        if x1 <= x0 or y1 <= y0 or k1 <= k0:
            continue
        bx, by = x1 - x0, y1 - y0
        for kz in range(k0, k1):
            for y in range(y0, y1):
                row = offs[k] + ((kz - k0) * by + (y - y0)) * bx - x0
                for x in range(x0, x1):
                    out[kz, y, x] += a * buf[row + x]


# --------------------------------------------------------------------------
# Boxes: the columns of the linear model
# --------------------------------------------------------------------------
class Boxes:
    """The columns of a linear model, each a dense box of the cube: buf[offs[k]:offs[k+1]] holds column k in the
    box boxes[k] = (x0, x1, y0, y1, k0, k1) (pixels and channels of the cube, [lo, hi)), with layout
    (channel, y, x)."""

    def __init__(self, buf, offs, boxes):
        self.buf, self.offs, self.boxes = buf, offs, boxes

    @property
    def n(self):
        return self.boxes.shape[0]

    @staticmethod
    def from_boxes(boxes):
        boxes = np.ascontiguousarray(boxes, dtype=np.int64)
        size = (boxes[:, 1] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 2]) * (boxes[:, 5] - boxes[:, 4])
        offs = np.zeros(boxes.shape[0] + 1, dtype=np.int64)
        np.cumsum(np.maximum(size, 0), out=offs[1:])
        return Boxes(np.zeros(int(offs[-1]), dtype=np.float64), offs, boxes)

    @staticmethod
    def from_cubes(cubes, tiny=0.):
        """Boxes of a list of (nz, ny, nx) arrays: the bounding box of the values above tiny of each."""
        bx = []
        for a in cubes:
            nzv = np.argwhere(np.abs(a) > tiny)
            if nzv.size == 0:
                bx.append((0, 0, 0, 0, 0, 0))
            else:
                lo, hi = nzv.min(axis=0), nzv.max(axis=0) + 1
                bx.append((lo[2], hi[2], lo[1], hi[1], lo[0], hi[0]))
        B = Boxes.from_boxes(np.array(bx, dtype=np.int64).reshape(-1, 6))
        for k, a in enumerate(cubes):
            x0, x1, y0, y1, k0, k1 = B.boxes[k]
            if x1 > x0:
                B.buf[B.offs[k]:B.offs[k + 1]] = a[k0:k1, y0:y1, x0:x1].ravel()
        return B

    def column(self, k, shape):
        """Column k as a full (nz, ny, nx) array."""
        out = np.zeros(shape)
        x0, x1, y0, y1, k0, k1 = self.boxes[k]
        if x1 > x0 and y1 > y0 and k1 > k0:
            out[k0:k1, y0:y1, x0:x1] = self.buf[self.offs[k]:self.offs[k + 1]].reshape(k1 - k0, y1 - y0, x1 - x0)
        return out

    def dense(self, shape, mask=None):
        """The matrix A (number of voxels, n): all the voxels of shape, or those of the boolean mask."""
        cols = [self.column(k, shape).ravel() for k in range(self.n)]
        A = np.array(cols).T if cols else np.zeros((int(np.prod(shape)), 0))
        return A[np.ravel(mask)] if mask is not None else A

    def model(self, amp, shape):
        out = np.zeros(shape)
        _scatter(out, self.buf, self.offs, self.boxes, np.ascontiguousarray(amp, dtype=np.float64))
        return out

    def lim(self):
        """The box (x0, x1, y0, y1, k0, k1) containing all the columns."""
        b = self.boxes
        ok = (b[:, 1] > b[:, 0]) & (b[:, 3] > b[:, 2]) & (b[:, 5] > b[:, 4])
        if not np.any(ok):
            return np.zeros(6, dtype=np.int64)
        b = b[ok]
        return np.array([b[:, 0].min(), b[:, 1].max(), b[:, 2].min(), b[:, 3].max(), b[:, 4].min(), b[:, 5].max()],
                        dtype=np.int64)

    def voxel_lists(self, wm, lim=None, rel=0.):
        """The values of the columns (where wm != 0) grouped by voxel of lim (default self.lim()): the tuple (lim,
        start, cell, value). rel > 0 leaves out the values below rel times the peak of their column (the faint wings,
        where the tails of the line and of the beam multiply: with rel = 1e-3, about 85% of the values and 0.3% of
        the flux of a column)."""
        lim = self.lim() if lim is None else np.asarray(lim, dtype=np.int64)
        return (lim,) + _voxel_lists(self.buf, self.offs, self.boxes, wm, lim, float(rel))

    def normal(self, wd, wm, lists=None):
        """(G, b) of the normal equations: G = AtWA, b = AtWd for the weights wm and the weighted data wd = wm*d
        ((nz, ny, nx) arrays), accumulated over the non-zero voxels of the columns (lists: voxel_lists(wm))."""
        n = self.n
        G, b = np.zeros((n, n)), np.zeros(n)
        lim, start, cell, val = self.voxel_lists(wm) if lists is None else lists
        _normal_lists(start, cell, val, wd, wm, lim, G, b)
        return G, b

    def cross(self, other, wm, lists=None, other_lists=None):
        """Gab = AtWB with the columns B of other (their voxel lists on the same lim, if given)."""
        Gab = np.zeros((self.n, other.n))
        if lists is None or other_lists is None:
            la, lb = self.lim(), other.lim()
            lim = np.array([min(la[0], lb[0]), max(la[1], lb[1]), min(la[2], lb[2]), max(la[3], lb[3]),
                            min(la[4], lb[4]), max(la[5], lb[5])], dtype=np.int64)
            lists, other_lists = self.voxel_lists(wm, lim), other.voxel_lists(wm, lim)
        if not np.array_equal(lists[0], other_lists[0]):
            raise BBaroloError("Boxes.cross: the voxel lists must be on the same box")
        _cross_lists(*lists[1:], *other_lists[1:], wm, lists[0], Gab)
        return Gab


# --------------------------------------------------------------------------
# The engine
# --------------------------------------------------------------------------
class WindModel:
    """Models of bi-conical winds on the grid of a datacube (see the module docstring).

    WindModel(cube, cells, smooth=True)
      cube:   a bbarolo Cube (its header and parameters: spectral axis, lines RESTWAVE/RESTFREQ/RELINT, beam);
      cells:  the WindCells of the emission (their quadrature is set by set_quadrature or quadrature());
      smooth: spread the points with the beam of the cube (default), else bilinearly on the pixels.

    Then, for the parameters p (WindPars):
      points(p)        the quadrature points (sky pixels, velocities, weights, cell of each point);
      boxes(p, lim)    the columns of the model (Boxes), within lim = (x0, x1, y0, y1, k0, k1) (default the cube);
      cube(p, amp)     the model cube (nz, ny, nx) for the amplitudes amp of the cells (emissivity per arcsec^3).
    The model is a flux per pixel and channel in the units of the amplitudes times arcsec^3 (the line integrates
    to 1 over the channels, the beam to 1 over the pixels): see GALWIND WTYPE 2 for the units of DENS.
    """

    def __init__(self, cube, cells, smooth=True):
        self.cin = cube
        self.cells = cells
        self.setup = GalmodSetup(cube, densflux=True)
        su = self.setup
        self.shape = (cube.DimZ(), cube.DimY(), cube.DimX())
        self.cdelt = (su.cdelt[0] * 60., su.cdelt[1] * 60.)          # arcsec per pixel, with the sign of CDELT
        self.pixsize = min(abs(self.cdelt[0]), abs(self.cdelt[1]))
        self.ones = np.ones(self.shape[0])
        self.psfq = np.zeros(5)
        self.hx = self.hy = 0
        self.kern = np.ones((1, 1))             # (no beam: the bilinear deposit only)
        self.kx = self.ky = np.ones(1)
        self.separable = True
        self.beam = None
        self.sigbeam = None
        h = cube.Head()
        if smooth and h.Bmaj() > 0 and h.Bmin() > 0:
            self.beam = (h.Bmaj() * 3600., h.Bmin() * 3600., h.Bpa())
            self._set_psf()

    def _set_psf(self):
        """The beam, with the conventions of SMOOTH (fillgauss2d: the major axis at BPA + 90 - CROTA from the x axis):
        psfq, the quadratic form of the beam in the pixel offsets (a, b, c: exp(-(a dx^2 + 2 b dx dy + c dy^2)/2),
        normalisation, cut-off), and the kernel convolved with the bilinear deposit of the points: the Gaussian
        whose covariance is that of the beam minus the 1/6 pixel^2 of the bilinear weights on each axis (so that
        the model has the beam of the data), sampled on the pixels and normalised to 1, zero below PSF_CUTOFF of
        its peak."""
        bmaj, bmin, bpa = self.beam
        crota = self.cin.Head().Crota()
        phi = math.radians(bpa + 90. - crota)
        cs, sn = math.cos(phi), math.sin(phi)
        g0, g1 = abs(self.cdelt[0]), abs(self.cdelt[1])
        k = 8. * math.log(2.)
        a = k * g0 * g0 * (cs * cs / bmaj ** 2 + sn * sn / bmin ** 2)
        b = k * g0 * g1 * cs * sn * (1. / bmaj ** 2 - 1. / bmin ** 2)
        c = k * g1 * g1 * (sn * sn / bmaj ** 2 + cs * cs / bmin ** 2)
        if abs(b) < 1e-12 * math.sqrt(a * c):
            b = 0.
        det = a * c - b * b
        qcut = -2. * math.log(PSF_CUTOFF)
        self.psfq = np.array([a, b, c, math.sqrt(det) / (2. * math.pi), qcut])
        # (the dispersion of the beam along its minor axis, arcsec)
        self.sigbeam = min(bmaj, bmin) / (2. * math.sqrt(2. * math.log(2.)))
        # covariance of the beam (pixels^2) minus that of the bilinear weights
        S = np.linalg.inv(np.array([[a, b], [b, c]])) - np.eye(2) / 6.
        if np.min(np.linalg.eigvalsh(S)) <= 0.05:
            S = np.linalg.inv(np.array([[a, b], [b, c]]))        # (a beam of about one pixel: no correction)
        M = np.linalg.inv(S)
        ka, kb, kc = M[0, 0], M[0, 1], M[1, 1]
        kdet = ka * kc - kb * kb
        self.hx = int(math.ceil(math.sqrt(qcut * kc / kdet)))
        self.hy = int(math.ceil(math.sqrt(qcut * ka / kdet)))
        yy, xx = np.mgrid[-self.hy:self.hy + 1, -self.hx:self.hx + 1].astype(float)
        q = ka * xx * xx + 2 * kb * xx * yy + kc * yy * yy
        kern = np.where(q <= qcut, np.exp(-0.5 * q), 0.)
        self.kern = np.ascontiguousarray(kern / kern.sum())
        self.separable = (b == 0.)
        if self.separable:
            kx = np.exp(-0.5 * ka * np.arange(-self.hx, self.hx + 1.) ** 2)
            ky = np.exp(-0.5 * kc * np.arange(-self.hy, self.hy + 1.) ** 2)
            self.kx, self.ky = kx / kx.sum(), ky / ky.sum()

    def quadrature(self, vmax, sigmin, spacing=None, cap=16):
        """Set the quadrature of the cells for winds up to vmax (km/s, |VWIND| + |VROT|) and lines at least sigmin
        (km/s, VDISP; the instrumental dispersion is added) wide: points at most spacing (arcsec) apart on the sky
        (default the dispersion of the beam, at most one pixel without a beam) and at most one line dispersion apart
        in velocity (the sum of Gaussians one dispersion apart is flat to 1e-8)."""
        if spacing is None:
            spacing = self.pixsize if self.beam is None else self.sigbeam
        sig = math.hypot(sigmin, self.setup.sig_instr / 1000.)
        self.cells.set_quadrature(spacing, vmax, 2. * sig, cap=cap, maxpoints=MAX_POINTS)
        if self.cells.spacing > 2. * spacing:
            # (Gaussians 2 dispersions apart sum to a ripple of 1.5%; coarser, the model is lumpy)
            import sys
            sys.stderr.write(f" WindModel WARNING: the cones are sampled every {self.cells.spacing:.3g} arcsec, more "
                             f"than twice the {spacing:.3g} arcsec needed (at most {MAX_POINTS} quadrature points): the "
                             "model is lumpier than the beam. Use shorter or narrower cones (CONELEN, OPENANG), or a "
                             "cube with fewer pixels per beam.\n")
        return self

    # ------------------------------------------------------------------
    def center(self, xpos, ypos):
        """The pixel coordinates of a centre given as pixels or WCS strings."""
        return getCenterCoordinates([xpos, ypos], self.cin.Head())

    def points(self, p):
        """The quadrature points for the parameters p (WindPars): a dict of arrays gx, gy (pixels), vel and sig
        (m/s), w (arcsec^3), r (arcsec), cell (index of the cell of each point), and of the cells vol (clipped
        volume, arcsec^3) and rbar (mean radius, arcsec)."""
        if self.cells.qoff is None:
            raise BBaroloError("WindModel: the quadrature of the cells is not set (call quadrature first)")
        p.check()
        su = self.setup
        c = self.cells
        n = c.npoints
        out = {k: np.empty(n) for k in ("gx", "gy", "vel", "sig", "w", "r")}
        vol, rbar = np.empty(c.ncell), np.empty(c.ncell)
        inc = math.radians(p.inc)
        ang = math.radians(p.pa) + su.crota2
        _wind_points(c.cells, c.qn, c.qoff, p.radii, p.vwind, p.vdisp, p.vrot, p.xpos, p.ypos, p.vsys * 1000.,
                     math.sin(inc), math.cos(inc), math.sin(ang), math.cos(ang), self.cdelt[0], self.cdelt[1],
                     math.radians(0.5 * p.openin), math.radians(0.5 * p.openang), p.conelen, su.sig_instr,
                     out["gx"], out["gy"], out["vel"], out["sig"], out["w"], out["r"], vol, rbar)
        out["vol"], out["rbar"] = vol, rbar
        return out

    def boxes(self, p, lim=None, pts=None, weight=None):
        """The columns of the model (Boxes) for the parameters p, within lim = (x0, x1, y0, y1, k0, k1) (pixels and
        channels of the cube, [lo, hi); default all). pts: the points (default points(p)); weight: an extra weight
        of each point (e.g. a radial profile). The points are deposited bilinearly in the box of their cell, which
        is then convolved with the beam (kern)."""
        su = self.setup
        if pts is None:
            pts = self.points(p)
        w = pts["w"] if weight is None else pts["w"] * weight
        nz, ny, nx = self.shape
        lim = np.array([0, nx, 0, ny, 0, nz] if lim is None else lim, dtype=np.int64)
        hx, hy = self.hx, self.hy
        rlim = lim + np.array([-hx, hx, -hy, hy, 0, 0])
        c = self.cells
        rboxes = np.empty((c.ncell, 6), dtype=np.int64)
        spec = (su.axtyp, su.velsys, su.lfreq0, su.crval3, su.cdelt3, su.crpix3)
        _raw_boxes(c.qoff, pts["gx"], pts["gy"], pts["vel"], pts["sig"], w, rlim, su._chmin, *spec, rboxes)
        R = Boxes.from_boxes(rboxes)
        _deposit_raw(R.buf, R.offs, R.boxes, c.qoff, pts["gx"], pts["gy"], pts["vel"], pts["sig"], w,
                     su._vedges_lines, su.relint, self.ones, _PHI_TAB, su._chmin, *spec, nz)
        if hx == 0 and hy == 0:
            return Boxes(R.buf, R.offs, np.ascontiguousarray(np.clip(rboxes, np.repeat(lim[::2], 2),
                                                                     np.repeat(lim[1::2], 2))))
        boxes = rboxes + np.array([-hx, hx, -hy, hy, 0, 0])
        boxes[:, 0:2] = np.clip(boxes[:, 0:2], lim[0], lim[1])
        boxes[:, 2:4] = np.clip(boxes[:, 2:4], lim[2], lim[3])
        empty = (rboxes[:, 1] <= rboxes[:, 0]) | (boxes[:, 1] <= boxes[:, 0]) | (boxes[:, 3] <= boxes[:, 2])
        boxes[empty] = (lim[0], lim[0], lim[2], lim[2], lim[4], lim[4])
        B = Boxes.from_boxes(boxes)
        _convolve_boxes(R.buf, R.offs, R.boxes, self.kern, self.kx, self.ky, self.separable, B.buf, B.offs, B.boxes)
        return B

    def cube(self, p, amp=None, weight=None, lim=None):
        """The model cube (nz, ny, nx) of the parameters p with the amplitudes amp of the cells (emissivity per
        arcsec^3; default 1 for all: a uniform wind), and weight an extra weight of each point."""
        B = self.boxes(p, lim=lim, weight=weight)
        if amp is None:
            amp = np.ones(B.n)
        return B.model(amp, self.shape)

    def footprint(self, plist, grow=0):
        """The pixels (a boolean (ny, nx) map) within the beam of any point of the winds of the parameters plist (a
        list of WindPars), grown by grow pixels."""
        from scipy import ndimage
        nz, ny, nx = self.shape
        m = np.zeros((ny, nx), dtype=bool)
        for p in plist:
            pts = self.points(p)
            ok = pts["w"] > 0
            ix, iy = np.rint(pts["gx"][ok]).astype(np.int64), np.rint(pts["gy"][ok]).astype(np.int64)
            inside = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
            m[iy[inside], ix[inside]] = True
        hx, hy = max(self.hx, 1) + int(grow), max(self.hy, 1) + int(grow)
        yy, xx = np.mgrid[-hy:hy + 1, -hx:hx + 1]
        if self.beam is not None:
            a, b, c, _, qcut = self.psfq
            g = 1. + grow / max(self.hx, self.hy, 1)
            st = (a * xx * xx + 2 * b * xx * yy + c * yy * yy) <= qcut * g * g
        else:
            st = np.ones_like(xx, dtype=bool)
        return ndimage.binary_dilation(m, structure=st) if m.any() else m
