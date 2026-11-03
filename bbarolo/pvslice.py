"""
Position-velocity (PV) slices: the single implementation used everywhere in bbarolo
(the PVSLICE task, the PV FITS files of 3DFIT and GALWIND, and the 3DFIT plots).

Two steps, which can be used separately:

    s, x, y = pv_path(x0, y0, pa, shape=cube.shape[1:])     # straight slice through the field
    s, x, y = pv_path(x0, y0, pa_rings, radii=r_rings, ...)  # slice following the rings
    s, x, y = pv_path(p1=(xa, ya), p2=(xb, yb), ...)         # segment between two points
    pv = pv_sample(cube, x, y, width=3, order=1)             # (nchan, npoints) array

or both at once with pv_extract(cube, ...). PvSlice wraps them for Cube objects and
returns a 2D image with its FITS header (offset and spectral axes).

pv_sample interpolates each spatial plane at the path points with gathers of whole
spectra (nearest or bilinear, NaN-aware), so it costs a few array operations per
slit position regardless of the number of channels. Spline interpolation (order >= 2)
uses scipy.ndimage.map_coordinates channel by channel.

pv_path and pv_sample need only numpy (and scipy for curved paths / order >= 2): their
source is also copied in the plot_utils.py script written by 3DFIT, so that the plotting
scripts stay standalone.
"""

import math
import sys
import numpy as np

from .image import Image2D
from .bbutils import BBaroloError, arcsconv, getCenterCoordinates


def _line_range(x0, y0, dx, dy, nx, ny):
    """Range [smin, smax] of s for which (x0 + s*dx, y0 + s*dy) is inside [0,nx-1] x [0,ny-1]."""
    lo, hi = -np.inf, np.inf
    for p, d, n in ((x0, dx, nx), (y0, dy, ny)):
        if abs(d) < 1e-12:
            if p < 0 or p > n - 1:
                return 1., -1.
            continue
        a, b = (0 - p) / d, (n - 1 - p) / d
        lo, hi = max(lo, min(a, b)), min(hi, max(a, b))
    return lo, hi


def pv_path(x0=None, y0=None, pa=None, radii=None, shape=None, p1=None, p2=None, step=1.0):
    """Points of a slice through a datacube, in pixel coordinates (0-based), equally spaced
    by `step` pixels along the slice.

    - Straight slice: through (x0, y0) at position angle pa (degrees, from the +y axis
      towards -x, i.e. North through East for a standard image), across the whole field.
    - Curved slice: if pa is an array with one angle per radius in radii (pixels), the
      slice follows the rings: a spline through the points at +-radii along the ring
      position angles, extended along its end tangents to the edges of the field.
    - Segment: from p1=(x, y) to p2=(x, y), offset 0 at its midpoint.

    shape: (ny, nx) of the field; the points outside it are dropped (required for the
    first two forms). Returns s, x, y: the offset of each point along the slice (pixels, 0
    at the centre, increasing towards pa or towards p2) and its coordinates.
    """
    step = float(step)
    if p1 is not None:
        (xa, ya), (xb, yb) = p1, p2
        L = math.hypot(xb - xa, yb - ya)
        if L == 0:
            raise ValueError("pv_path: p1 and p2 coincide.")
        dx, dy = (xb - xa) / L, (yb - ya) / L
        xc, yc = 0.5 * (xa + xb), 0.5 * (ya + yb)
        lo, hi = -0.5 * L, 0.5 * L
        if shape is not None:
            a, b = _line_range(xc, yc, dx, dy, shape[1], shape[0])
            lo, hi = max(lo, a), min(hi, b)
        k = np.arange(math.ceil(lo / step - 1e-9), math.floor(hi / step + 1e-9) + 1)
        s = k * step
        return s, xc + s * dx, yc + s * dy
    ny, nx = shape[-2], shape[-1]
    pa_arr = np.atleast_1d(np.asarray(pa, float))
    curved = radii is not None and pa_arr.size > 1 and np.ptp(pa_arr) > 1e-6
    if not curved:
        t = math.radians(float(pa_arr[0]))
        dx, dy = -math.sin(t), math.cos(t)
        lo, hi = _line_range(x0, y0, dx, dy, nx, ny)
        if hi < lo:
            return np.zeros(0), np.zeros(0), np.zeros(0)
        k = np.arange(math.ceil(lo / step - 1e-9), math.floor(hi / step + 1e-9) + 1)
        s = k * step
        return s, x0 + s * dx, y0 + s * dy
    # Curved slice: spline through the ring points, from -R_max to +R_max
    r = np.asarray(radii, float)
    o = np.argsort(r)
    r, t = r[o], np.radians(pa_arr[o])
    keep = r > 0
    r, t = r[keep], t[keep]
    xs = np.concatenate([x0 + r[::-1] * np.sin(t[::-1]), [x0], x0 - r * np.sin(t)])
    ys = np.concatenate([y0 - r[::-1] * np.cos(t[::-1]), [y0], y0 + r * np.cos(t)])
    kdeg = min(3, xs.size - 1)
    uu = np.linspace(0, 1, 20 * xs.size)
    try:
        from scipy.interpolate import make_splprep          # scipy >= 1.15
        spl, _ = make_splprep(np.vstack([xs, ys]), s=0, k=kdeg)
        xc, yc = spl(uu)
        t0, t1 = np.asarray(spl(0.0, nu=1)), np.asarray(spl(1.0, nu=1))
    except ImportError:
        from scipy.interpolate import splprep, splev
        tck, _ = splprep([xs, ys], s=0, k=kdeg)
        xc, yc = (np.asarray(v) for v in splev(uu, tck))
        t0, t1 = np.asarray(splev(0.0, tck, der=1)), np.asarray(splev(1.0, tck, der=1))
    t0, t1 = t0 / np.hypot(*t0), t1 / np.hypot(*t1)
    # extensions along the end tangents, to the field edges
    ext = float(np.hypot(nx, ny))
    e = np.linspace(0, ext, int(ext / 0.25) + 2)[1:]
    xc = np.concatenate([xc[0] - e[::-1] * t0[0], xc, xc[-1] + e * t1[0]])
    yc = np.concatenate([yc[0] - e[::-1] * t0[1], yc, yc[-1] + e * t1[1]])
    # arc length, zero at the centre (the spline goes through (x0,y0))
    sc = np.concatenate([[0.], np.cumsum(np.hypot(np.diff(xc), np.diff(yc)))])
    sc -= sc[int(np.argmin((xc - x0) ** 2 + (yc - y0) ** 2))]
    k = np.arange(math.ceil(sc[0] / step), math.floor(sc[-1] / step) + 1)
    s = k * step
    x, y = np.interp(s, sc, xc), np.interp(s, sc, yc)
    inside = (x >= 0) & (x <= nx - 1) & (y >= 0) & (y <= ny - 1)
    # keep the stretch of the slice inside the field that contains the centre
    i0 = int(np.argmin(np.abs(s)))
    if inside[i0]:
        out = np.flatnonzero(~inside)
        lo = out[out < i0].max() + 1 if np.any(out < i0) else 0
        hi = out[out > i0].min() if np.any(out > i0) else s.size
        return s[lo:hi], x[lo:hi], y[lo:hi]
    return s[inside], x[inside], y[inside]


def pv_sample(cube, x, y, width=0., order=1, zoversample=1):
    """Sample a datacube (nz, ny, nx) along a slice: returns an array (nz, npoints).

    x, y: pixel coordinates of the slice points (e.g. from pv_path). width: full width of
    the slit in pixels; the slice is averaged over positions across the slit, spaced by
    at most 0.5 pixels, along the local normal of the slice. order: 0 nearest pixel,
    1 bilinear, >= 2 spline (scipy.ndimage.map_coordinates). NaN values (and points
    outside the cube) are ignored: bilinear weights are renormalised over the valid
    neighbours (NaN if they carry less than half of the weight). zoversample: number of
    output channels per input channel (linear interpolation along the spectral axis).
    """
    a = np.asarray(cube)
    nz, ny, nx = a.shape
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = x.size
    if n == 0:
        return np.zeros((nz, 0))
    flat = a.reshape(nz, ny * nx)
    # positions across the slit
    if width > 0 and n > 1:
        gx, gy = np.gradient(x), np.gradient(y)
        g = np.hypot(gx, gy)
        g[g == 0] = 1.
        nxv, nyv = gy / g, -gx / g
        offs = np.linspace(-width / 2., width / 2., int(math.ceil(width / 0.5)) + 1)
    else:
        nxv = nyv = np.zeros(n)
        offs = np.zeros(1)
    acc = np.zeros((nz, n))
    cnt = np.zeros((nz, n))
    for off in offs:
        xs, ys = x + off * nxv, y + off * nyv
        if order == 0:
            ix, iy = np.rint(xs).astype(np.int64), np.rint(ys).astype(np.int64)
            ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
            v = np.full((nz, n), np.nan)
            v[:, ok] = flat[:, iy[ok] * nx + ix[ok]]
        elif order == 1:
            fx0, fy0 = np.floor(xs), np.floor(ys)
            tx, ty = xs - fx0, ys - fy0
            ix0, iy0 = fx0.astype(np.int64), fy0.astype(np.int64)
            num = np.zeros((nz, n))
            den = np.zeros((nz, n))
            for ddx, ddy, w in ((0, 0, (1 - tx) * (1 - ty)), (1, 0, tx * (1 - ty)),
                                (0, 1, (1 - tx) * ty), (1, 1, tx * ty)):
                ix, iy = ix0 + ddx, iy0 + ddy
                ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny) & (w > 0)
                if not ok.any():
                    continue
                allok = bool(ok.all())
                idx = iy * nx + ix if allok else iy[ok] * nx + ix[ok]
                vals = flat[:, idx].astype(float, copy=False)
                wk = (w if allok else w[ok])[None, :]
                bad = np.isnan(vals)
                if bad.any():
                    vals = np.where(bad, 0., vals)
                    wk = wk * ~bad
                if allok:
                    num += vals * wk
                    den += wk
                else:
                    num[:, ok] += vals * wk
                    den[:, ok] += wk
            with np.errstate(invalid="ignore", divide="ignore"):
                v = np.where(den >= 0.5, num / den, np.nan)
        else:
            from scipy.ndimage import map_coordinates
            inside = (xs > -0.5) & (xs < nx - 0.5) & (ys > -0.5) & (ys < ny - 0.5)
            v = np.full((nz, n), np.nan)
            if inside.any():
                coords = np.vstack([ys[inside], xs[inside]])
                for k in range(nz):
                    plane = np.asarray(a[k], float)
                    bad = ~np.isfinite(plane)
                    vk = map_coordinates(np.where(bad, 0., plane), coords, order=int(order), mode="nearest")
                    if bad.any():
                        vk[map_coordinates(bad.astype(float), coords, order=0, mode="nearest") > 0] = np.nan
                    v[k, inside] = vk
        good = np.isfinite(v)
        acc += np.where(good, v, 0.)
        cnt += good
    with np.errstate(invalid="ignore", divide="ignore"):
        pv = np.where(cnt > 0, acc / cnt, np.nan)
    if zoversample and zoversample != 1:
        zf = np.arange(0, nz, 1.0 / zoversample)
        if order == 0:
            pv = pv[np.minimum(np.rint(zf).astype(np.int64), nz - 1)]
        else:
            i0 = np.floor(zf).astype(np.int64)
            i1 = np.minimum(i0 + 1, nz - 1)
            tz = (zf - i0)[:, None]
            pv = (1 - tz) * pv[i0] + tz * pv[i1]
    return pv


def pv_extract(cube, x0=None, y0=None, pa=None, radii=None, p1=None, p2=None, width=0., order=1,
               step=1.0, zoversample=1):
    """pv_path + pv_sample in one call. Returns s, x, y, pv (see the two functions)."""
    s, x, y = pv_path(x0, y0, pa, radii=radii, shape=np.shape(cube)[1:], p1=p1, p2=p2, step=step)
    return s, x, y, pv_sample(cube, x, y, width=width, order=order, zoversample=zoversample)


# --------------------------------------------------------------------------
# PV slice of a Cube with its FITS header
# --------------------------------------------------------------------------
class PvSlice(Image2D):
    """PV slice of a Cube, as a 2D image (axis 1: offset along the slice, axis 2: the
    spectral axis of the cube) with its header.

    The slice is defined by its centre and position angle (x0, y0, angle: pixels and
    degrees), by two points (x1, y1, x2, y2), or, if none is given, by the PVSLICE
    parameters (XPOS_PV, YPOS_PV, PA_PV or P1_PV, P2_PV; WIDTH_PV; ANTIALIAS).
    radii (pixels) with angle an array: slice following the rings (see pv_path).
    width: full slit width in pixels. order: 0 nearest, 1 bilinear (default), >= 2 spline.
    step: sampling along the slice, in pixels.

    As in BBarolo, the offset axis of a straight slice defined by an angle increases with
    x (with y for a vertical slice), the offset is 0 at (x0, y0) and the slice crosses the
    whole field; a slice between two points goes from the first to the second point.
    """

    def __init__(self, c, x0=None, y0=None, angle=None, x1=None, y1=None, x2=None, y2=None, width=0.,
                 order=1, step=1.0, radii=None):
        super().__init__()
        self.cin = c
        self.width = float(width)
        self.order = order
        self.step = float(step)
        self.radii = radii
        if x0 is not None:
            self.x0, self.y0, self.angle = float(x0), float(y0), angle
            self.isAngle = True
        elif x1 is not None:
            self.x1, self.y1, self.x2, self.y2 = map(float, (x1, y1, x2, y2))
            self.isAngle = False
        else:
            p = c.pars()
            if all(v >= 0 for v in list(p.P1_PV) + list(p.P2_PV)):
                self.x1, self.y1 = p.P1_PV
                self.x2, self.y2 = p.P2_PV
                self.isAngle = False
            elif p.XPOS_PV != "-1" and p.YPOS_PV != "-1":
                self.x0, self.y0 = getCenterCoordinates([p.XPOS_PV, p.YPOS_PV], c.Head())
                self.angle = p.PA_PV
                self.isAngle = True
            else:
                raise ValueError("PVSLICE ERROR: no slice has been defined!")
            pix = c.Head().PixScale() * arcsconv(c.Head().Cunit(0))
            self.width = float(p.WIDTH_PV) / pix if p.WIDTH_PV > 0 else 0.
            na = float(p.ANTIALIAS)
            # ANTIALIAS: 0 = nearest pixel; otherwise bilinear interpolation, and values >= 1
            # also average over a slit at least 2*ANTIALIAS+1 pixels wide
            self.order = 0 if na == 0 else 1
            if na >= 1:
                self.width = max(self.width, 2 * na + 1)

    def slice(self):
        """Extract the slice; returns False if it does not cross the cube."""
        c = self.cin
        a = c.array
        if self.isAngle:
            if self.radii is None:
                if not (0 <= self.x0 < c.DimX() and 0 <= self.y0 < c.DimY()):
                    raise BBaroloError(f"PVSLICE: the centre ({self.x0:g}, {self.y0:g}) is outside the cube "
                                       f"({c.DimX()} x {c.DimY()} pixels).")
            s, x, y = pv_path(self.x0, self.y0, self.angle, radii=self.radii, shape=a.shape[1:], step=self.step)
            if s.size > 1 and np.ndim(self.angle) == 0:
                # BBarolo's orientation: offset increasing with x (with y if vertical)
                if x[-1] < x[0] - 1e-9 or (abs(x[-1] - x[0]) <= 1e-9 and y[-1] < y[0]):
                    s, x, y = -s[::-1], x[::-1], y[::-1]
        else:
            s, x, y = pv_path(p1=(self.x1, self.y1), p2=(self.x2, self.y2), shape=a.shape[1:], step=self.step)
        if s.size < 2:
            raise BBaroloError("PVSLICE: the slice does not cross the cube.")
        self.s, self.x_locus, self.y_locus = s, x, y
        pv = pv_sample(a, x, y, width=self.width, order=self.order)
        self.array = pv.astype(np.float32)
        self.define_header()
        return True

    def define_header(self):
        c = self.cin
        h = c.Head()
        self.copyHeader(h)
        nh = self.head
        i0 = int(np.argmin(np.abs(self.s)))
        nh.setCtype(0, "Offset")
        nh.setCrpix(0, i0 + 1 - self.s[i0] / self.step)
        nh.setCrval(0, 0)
        nh.setCdelt(0, self.step * abs(h.Cdelt(0)))
        nh.setCunit(0, h.Cunit(0))
        nh.setCtype(1, h.Ctype(2))
        nh.setCrpix(1, h.Crpix(2))
        nh.setCrval(1, h.Crval(2))
        nh.setCdelt(1, h.Cdelt(2))
        nh.setCunit(1, h.Cunit(2))
        nh.btype = h.Btype()
        nh.setMinMax(0., 0.)
        nh.dimAxes = [self.array.shape[1], self.array.shape[0]]
        self.headDefined = True
        if self.isAngle:
            crt = h.pixToWCS(np.array([[self.x0, self.y0, 0]]))[0]
            ang = float(np.median(np.atleast_1d(self.angle)))
            s = f"RA={crt[0]:.6f} deg, DEC={crt[1]:.6f} deg, PA={ang:.1f} deg"
            nh.setName(h.Name() + f"_pv{ang:.0f}")
        else:
            s = f"P1=({self.x1:.1f},{self.y1:.1f}) P2=({self.x2:.1f},{self.y2:.1f})"
            nh.setName(h.Name() + "_pv")
        nh.keys.append("HISTORY BBAROLO PVSLICE: " + s)


def PositionVelocity(c, x0, y0, Phi, width=0., order=1):
    """Straight PV slice of the Cube c through (x0, y0) at position angle Phi (degrees)."""
    pv = PvSlice(c, x0=x0, y0=y0, angle=Phi, width=width, order=order)
    pv.slice()
    return pv
