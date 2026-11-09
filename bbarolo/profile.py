"""
Density profiles corrected for the beam (deconvolved), shared by Bayesian3DFIT and the initial
estimates (GUESSTYPE FIT).

The total intensity map of a disk model is linear in the densities of its rings, sum_r dens_r K_r,
where K_r is the map of ring r with unit density convolved with the beam
(DiskModel.ring_maps: exact for the ANALYTIC model, the expectation of the MC one). It depends only
on the geometry. For a given geometry the densities that reproduce an observed map are a
non-negative weighted least-squares problem, with:

  * an exterior annulus, from the outer edge of the disk (the edge of the rings, or RADMAX when it
    is the edge of the disk) to two beams beyond, with its own density: it takes the emission beyond
    the disk, which the last ring would otherwise absorb (its density is not used);
  * a smoothness prior: rings narrower than the beam are nearly degenerate once convolved, and
    noise or non-axisymmetric structure make the solution oscillate from ring to ring. The squared
    second differences of the profile (including the exterior annulus) are penalised with a weight
    smooth (0.1 by default) times the mean data weight of a ring (trace(A^T A) / trace(L^T L)),
    independent of the units and of the S/N (0 = no prior).
"""

import math
import numpy as np


def profile_system(model, rings, box, data_map, rms_map, ok_map, beam_arcsec=None):
    """The weighted least-squares system of the deconvolved profile: (A, b, ok, Rc).

    model: the DiskModel of the cube; rings: a Rings object; box: (blo, bhi) of the maps (pixels,
    as for DiskModel.ring_maps); data_map, rms_map, ok_map: the total intensity map of the data, its
    noise and the pixels to use, all with the shape of the box. A (npix, nr+1) holds the
    beam-convolved maps of the rings and of the exterior annulus with unit density, b the data map,
    both divided by the noise, in the pixels ok; Rc are the radii of the rings and of the exterior
    annulus. None if there are fewer than 2 nr pixels."""
    from .galmod import outer_edge
    from .rings import Rings
    Lb, Hb = box
    r = rings
    K = model.ring_maps(r, Lb, Hb)
    R = np.asarray(r.radii, float)
    if beam_arcsec:
        w = 2. * beam_arcsec
    else:
        w = 2. * float(np.max(np.diff(R), initial=R[-1]))
    edge = outer_edge(r, model.ringmode)
    if getattr(r, "edge", None) is not None:          # (RADMAX as the edge of the disk, also inside the last ring)
        edge = min(edge, float(r.edge))
    rext = edge + w / 2.
    ext = Rings().setRings(radii=[rext], xpos=r.xpos[-1], ypos=r.ypos[-1], vsys=r.vsys[-1], vrot=r.vrot[-1],
                           vdisp=r.vdisp[-1], z0=r.z0[-1], inc=r.inc[-1], phi=r.phi[-1], dens=1., radsep=w)
    K = np.concatenate([K, model.ring_maps(ext, Lb, Hb, ringmode="CENTRED")])
    y, s = data_map, rms_map
    ok = ok_map & (s > 0) & np.isfinite(y)
    if ok.sum() < 2 * r.nr:
        return None
    return K[:, ok].T / s[ok, None], y[ok] / s[ok], ok, np.r_[R, rext]


def solve_profile(A, b, radii, smooth=0.1):
    """Non-negative least-squares solution of A dens = b (all the columns, including the exterior
    annulus), with the smoothness prior of weight smooth on the profile at the radii."""
    from scipy.optimize import nnls
    n = A.shape[1]
    alpha = float(smooth or 0.)
    if alpha > 0 and n >= 3:
        R = np.asarray(radii, float)
        L = np.zeros((n - 2, n))
        for i in range(1, n - 1):                   # second differences (any spacing)
            d1, d2 = R[i] - R[i - 1], R[i + 1] - R[i]
            L[i - 1, i - 1:i + 2] = np.array([1. / d1, -1. / d1 - 1. / d2, 1. / d2]) * 2. / (d1 + d2)
        L *= math.sqrt(alpha * np.sum(A * A) / max(np.sum(L * L), 1e-300))
        A, b = np.vstack([A, L]), np.r_[b, np.zeros(n - 2)]
    q, r = np.linalg.qr(A)                         # nnls on the small (n x n) system
    return nnls(r, q.T @ b)[0]


def deconvolved_profile(model, rings, box, data_map, rms_map, ok_map, smooth=0.1, beam_arcsec=None):
    """The densities of the rings whose beam-convolved model reproduces data_map (see the module
    docstring), or None if there are too few pixels."""
    sysm = profile_system(model, rings, box, data_map, rms_map, ok_map, beam_arcsec=beam_arcsec)
    if sysm is None:
        return None
    A, b, _, Rc = sysm
    return solve_profile(A, b, Rc, smooth)[:rings.nr]



def _layer_offsets(ltype, n):
    """n stratified offsets (units of Z0) of the vertical profile of the layers (LTYPE, as the model: 1 gaussian,
    2 sech2, 3 exponential, 4 Lorentzian, 5 box)."""
    q = (np.arange(n) + 0.5) / n
    if ltype == 1:
        from scipy.special import ndtri
        return ndtri(q)
    x = 2. * q - 1.
    if ltype == 2:
        return np.arctanh(x)
    if ltype == 3:
        return np.sign(x) * -np.log(np.abs(x))
    if ltype == 4:
        return np.tan(0.5 * np.pi * np.clip(x, -0.95, 0.95))
    return x


class MapRingModel:
    """Total intensity maps of rings for a 2D map (no spectral axis), with the interface of DiskModel.ring_maps
    used by profile_system: each ring with unit density, its face-on surface density interpolated linearly between
    the radii with the ring convention ringmode (as the model of the cubes), projected on the sky (thin, or with
    the vertical profile LTYPE of scale height Z0) and convolved with the beam (normalised kernel: the units of a
    map per beam keep the surface brightness of extended emission). A uniform face-on disk of unit density gives
    1/cos(INC) in each pixel: the densities are face-on surface densities in the units of the map per pixel.

    h: the Header of the map; ltype: the vertical profile; nper: clouds per pixel of each ring."""

    def __init__(self, h, ltype=2, ringmode="CENTRED", nper=16):
        from .smooth import fillgauss2d, FFTConvolver
        from .bbutils import arcsconv
        self.ringmode = ringmode
        self.ltype = int(ltype)
        self.nper = nper
        u = arcsconv(h.Cunit(0))
        self.arcs = abs(h.Cdelt(0)) * u
        self.conv = None
        if h.Bmaj() > 0:
            gs = (h.Cdelt(0) * u, h.Cdelt(1) * u)
            k = fillgauss2d((h.Bmaj() * 3600., h.Bmin() * 3600., h.Bpa() + 90. - h.Crota()), gs, 1.0, norm=True)
            self.conv = FFTConvolver(k) if k is not None else None

    def ring_maps(self, rings, blo, bhi, ringmode=None):
        from .galmod import model_rings
        mode = self.ringmode if ringmode is None else ringmode
        nx, ny = bhi[0] - blo[0], bhi[1] - blo[1]
        out = np.zeros((rings.nr, ny, nx))
        a = self.arcs
        for r in range(rings.nr):
            e = rings.copy()
            e.dens = np.zeros(rings.nr)
            e.dens[r] = 1.
            nodes = model_rings(e, mode, 0.)
            Rn, Dn = np.asarray(nodes.radii, float), np.asarray(nodes.dens, float)
            nz = np.flatnonzero(Dn > 0)
            if nz.size == 0:
                continue
            Rlo, Rhi = Rn[max(nz[0] - 1, 0)], Rn[min(nz[-1] + 1, len(Rn) - 1)]
            if nz[0] == 0:
                Rlo = Rn[0]
            if nz[-1] == len(Rn) - 1:
                Rhi = Rn[-1]
            area = math.pi * (Rhi * Rhi - Rlo * Rlo) / (a * a)             # pixels
            z0 = float(rings.z0[r]) / a
            nzl = 8 if z0 > 0.05 else 1
            ntot = int(min(max(self.nper * area, 4000), 2000000))
            nrad = max(int(math.sqrt(ntot / nzl / math.pi)), 8)
            nth = max(int(ntot / nzl / nrad), 16)
            q = (np.arange(nrad) + 0.5) / nrad
            R = np.sqrt(Rlo * Rlo + (Rhi * Rhi - Rlo * Rlo) * q)              # uniform in area
            w = np.interp(R, Rn, Dn) * area / (nrad * nth * nzl)
            th = (np.arange(nth) + 0.5) / nth * 2. * math.pi
            zz = _layer_offsets(self.ltype, nzl) * z0 if nzl > 1 else np.zeros(1)
            inc, phi = math.radians(float(rings.inc[r])), math.radians(float(rings.phi[r]))
            X = (R[:, None, None] / a) * np.cos(th)[None, :, None] + 0. * zz[None, None, :]        # along the major axis
            Yd = (R[:, None, None] / a) * np.sin(th)[None, :, None]
            Y = Yd * math.cos(inc) + zz[None, None, :] * math.sin(inc)                     # along the minor axis
            x = float(rings.xpos[r]) - X * math.sin(phi) - Y * math.cos(phi) - blo[0]
            y = float(rings.ypos[r]) + X * math.cos(phi) - Y * math.sin(phi) - blo[1]
            W = np.broadcast_to(w[:, None, None], X.shape).ravel()
            x, y = x.ravel(), y.ravel()
            i0, j0 = np.floor(x).astype(int), np.floor(y).astype(int)
            fx, fy = x - i0, y - j0
            m = out[r]
            for di, dj, ww in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)), (0, 1, (1 - fx) * fy),
                               (1, 1, fx * fy)):
                ii, jj = i0 + di, j0 + dj
                ok = (ii >= 0) & (ii < nx) & (jj >= 0) & (jj < ny)
                np.add.at(m, (jj[ok], ii[ok]), (W * ww)[ok])
        if self.conv is not None:
            out = self.conv(out)
        return out


def ellprof_deconvolved(c, rings, side="B", smooth=0.1, nmc=200, seed=0, model=None):
    """The deconvolved radial profile of the ELLPROF task: the face-on surface densities of the rings whose
    beam-convolved model reproduces the total intensity map of the cube c (or the 2D map c), as in
    Bayesian3DFIT (see the module docstring), at the fixed geometry of the rings (a Rings object: radii,
    centre, INC, PA, Z0).

    A 2D map is the data itself, with the model of MapRingModel (the rings projected and convolved with the
    beam, without spectral axis), uniform weights, and errors from the scatter of the residuals of the fit
    (which includes the structure that the rings do not describe: conservative), correlated over a beam.

    The data map is the sum of the data, not clipped, in the mask of the cube dilated by one beam and +-3
    channels (the mask cuts the faint outer emission, which the model cannot reproduce); its noise is the rms
    of the data outside the mask times the square root of the number of channels summed and of the spectral
    correlation of the noise. The correlation of the noise is the conservative measure of Bayesian3DFIT (at
    least one beam area spatially). With side 'A' or 'R', only the pixels of that half of the galaxy are used
    (divided by the minor axis of the rings: their median centre and PA). The errors come from nmc refits of the
    best-fit map plus noise with that rms and spatial correlation (the geometry is fixed: no errors from it).

    model: the DiskModel of the cube whose densities are wanted (3DFIT: the engine of the fit, which then builds
    the final model with them; its ring convention and flux units); default an ANALYTIC model with CENTRED rings.

    Returns None for a map without a beam or with too few pixels; otherwise a dict
    with radii (arcsec), dens (the densities of the model, as DENS of GALMOD), sdens_fo (the face-on surface
    density in the units of the map of ELLPROF per arcsec^2, as its column SDENS_FO), err_lo, err_hi (1-sigma
    errors of sdens_fo), nmc, beam (FWHM, arcsec) and map2d (a 2D map: dens is then the face-on surface density in
    the units of the map per pixel)."""
    import copy
    from .galmod import DiskModel
    from .rings import Rings
    from .bbutils import arcsconv, DeltaVel, FluxtoJyBeam
    from .bayesian.noise import dilate_mask, noise_correlation
    h = c.Head()
    if not h.Bmaj() or not h.Cdelt(0):
        return None
    bpix = abs(h.Bmaj() / h.Cdelt(0))
    pixarea = abs(h.Cdelt(0) * h.Cdelt(1))
    beam_area = math.pi * abs(h.Bmaj() * h.Bmin()) / (4. * math.log(2.)) / pixarea if h.Bmin() else 1.
    is2d = c.DimZ() < 2
    if is2d:
        img = np.asarray(c.array[0], np.float64)
        blank2 = ~np.isfinite(img)
        b2d = getattr(c, "blanks2d", None)                 # (the blank pixels of a 2D map, read as 0)
        if b2d is not None and b2d.shape == img.shape:
            blank2 |= b2d
        dmap = np.where(blank2, 0., img)
        ok = ~blank2
        rms = np.ones(img.shape)
        area = beam_area
    else:
        data = np.asarray(c.array, np.float64)
        blank = (data == 0) | ~np.isfinite(data)
        mask = np.asarray(c.Mask(), bool) if c.MaskAll() else ~blank
        lm = dilate_mask(mask, bpix, 3) & ~blank
        dmap = np.where(lm, data, 0.).sum(axis=0)
        nvox = lm.sum(axis=0)
        out = data[~mask & ~blank]
        if out.size < 100:
            out = data[~blank]
        sigma = float(1.4826 * np.median(np.abs(out - np.median(out))))
        nc = noise_correlation(np.where(blank, np.nan, data), mask, bpix)
        area, szc = beam_area, 1.
        if nc is not None:
            area = max(nc['spatial'], nc['spatial_core'], beam_area)
            szc = max(nc['spectral'], nc['spectral_core'], 1.)
        rms = sigma * np.sqrt(nvox * szc)
        ok = nvox > 0
    ny, nx = ok.shape
    s = str(side).strip().upper()[:1]
    if s in ("A", "R"):
        xc, yc = float(np.median(rings.xpos)), float(np.median(rings.ypos))
        pa = float(np.median(np.unwrap(np.radians(np.asarray(rings.phi, float)))))     # (PAs across 0/360)
        yy, xx = np.mgrid[0:ny, 0:nx]
        xr = -(xx - xc) * math.sin(pa) + (yy - yc) * math.cos(pa)          # along the major axis: > 0 receding
        ok &= (xr >= 0) if s == "R" else (xr <= 0)
    if not ok.any():
        return None
    # the box: the pixels used, and the disk out to two beams beyond the last ring
    arcs = h.PixScale() * arcsconv(h.Cunit(0))
    rpix = (float(np.max(rings.radii)) + 0.5 * float(rings.radsep)) / arcs + 3. * bpix
    xc0, yc0 = float(np.median(rings.xpos)), float(np.median(rings.ypos))
    ys, xs = np.nonzero(ok)
    blo = (max(int(min(xs.min(), xc0 - rpix)), 0), max(int(min(ys.min(), yc0 - rpix)), 0))
    bhi = (min(int(max(xs.max() + 1, xc0 + rpix + 1)), nx), min(int(max(ys.max() + 1, yc0 + rpix + 1)), ny))
    sl = (slice(blo[1], bhi[1]), slice(blo[0], bhi[0]))
    nr = rings.nr
    z0 = np.where(np.asarray(rings.z0, float) > 0, rings.z0, 0.)
    r = Rings().setRings(radii=np.asarray(rings.radii, float), xpos=rings.xpos, ypos=rings.ypos, vsys=rings.vsys,
                         vrot=0., vdisp=8., z0=z0, inc=rings.inc, phi=rings.phi, dens=1., radsep=rings.radsep)
    par = copy.deepcopy(c.pars().gf)
    if is2d:
        dm = MapRingModel(h, ltype=getattr(par, "LTYPE", 2) or 2)
    elif model is not None:
        dm = model
    else:
        dm = DiskModel(c, par=par, ringmode="CENTRED", galmodtype="ANALYTIC", smooth=True)
    # only the pixels that the model describes: within the outer edge of its exterior annulus (two beams beyond the
    # edge of the rings), with the geometry of the last ring and its thickness (all the pixels with data were fitted,
    # so emission beyond the annulus pulled the solution: NGC 2403 with one ring 60" wide was 11 times too low)
    from .galmod import outer_edge
    lim = (outer_edge(r, dm.ringmode) + 2. * abs(h.Bmaj()) * 3600.) / arcs           # (pixels)
    inc = math.radians(min(abs(float(r.inc[-1])), 90.))
    bmin = min(lim, lim * math.cos(inc) + 2. * float(z0[-1]) / arcs * math.sin(inc))
    pa = math.radians(float(r.phi[-1]))
    yy, xx = np.mgrid[blo[1]:bhi[1], blo[0]:bhi[0]]
    dx, dy = xx - float(r.xpos[-1]), yy - float(r.ypos[-1])
    x1 = -dx * math.sin(pa) + dy * math.cos(pa)                         # along the major axis
    y1 = -dx * math.cos(pa) - dy * math.sin(pa)
    within = (x1 / lim) ** 2 + (y1 / max(bmin, 0.5)) ** 2 <= 1.
    sysm = profile_system(dm, r, (blo, bhi), dmap[sl], rms[sl], ok[sl] & within, beam_arcsec=abs(h.Bmaj()) * 3600.)
    if sysm is None:
        return None
    A, b, oks, Rc = sysm
    sol = solve_profile(A, b, Rc, smooth)
    dens = sol[:nr]
    noise = 1.                                          # (the noise of b: 1, as b is divided by the rms)
    if is2d:                                            # (uniform weights: the scatter of the residuals)
        res = b - A @ sol
        nz = b != 0                                     # (a clipped map: its zeros are not data, and fit too well)
        if nz.sum() > 2 * A.shape[1] and nz.sum() < 0.9 * nz.size:
            res = res[nz]
        noise = float(np.std(res) * math.sqrt(res.size / max(res.size - A.shape[1], 1)))
    # errors: refits of the best-fit map plus noise with the spatial correlation of the data
    lo = hi = np.zeros(nr)
    if nmc > 0:
        from scipy.ndimage import gaussian_filter
        rng = np.random.default_rng(seed)
        bmod = A @ sol
        sk = math.sqrt(area / (4. * math.pi))          # (the Gaussian autocorrelation of that area)
        prof = []
        for _ in range(int(nmc)):
            g = rng.standard_normal(oks.shape)
            if sk > 0.3:
                g = gaussian_filter(g, sk, mode='wrap')
                g /= g.std()
            prof.append(solve_profile(A, bmod + noise * g[oks], Rc, smooth)[:nr])
        p16, p50, p84 = np.quantile(np.asarray(prof), (0.15865, 0.50, 0.84135), axis=0)     # (1 sigma)
        lo, hi = p50 - p16, p84 - p50
    if is2d:
        unit = 1. / (arcs * arcs)                       # (the densities: face-on, in map units per pixel)
    else:
        # the face-on surface brightness of unit density (per pixel, in the units of the sum of the channels):
        # the flux of a face-on uniform disk of unit density divided by its area (the clouds of the model sit on
        # sub-rings, so single pixels are not uniform), at the centre of the cube. The area is that of the
        # sub-rings of the model (pi redge^2 included the empty centre of the disk: 2% low)
        from .galmod import model_rings
        xm, ym = nx // 2, ny // 2
        w = max(min(nx, ny) / 8., 2.) * arcs
        uni = Rings().setRings(radii=np.array([w, 2 * w, 3 * w]), xpos=xm, ypos=ym, vsys=float(rings.vsys[0]),
                               vrot=0., vdisp=8., z0=0., inc=0., phi=0., dens=1., radsep=w)
        su = dm.setup
        sub = su.subrings(model_rings(uni, "CENTRED", su.radsep))[0][0]           # (arcmin)
        area = float(np.sum(2. * math.pi * sub * su.radsep)) / (arcs / 60.) ** 2   # (pixels)
        k = float(dm.ring_maps(uni, ringmode="CENTRED").sum()) / area
        dv = abs(DeltaVel(h)) if c.HeadDef() else 1.
        conv = (FluxtoJyBeam(1., h) if c.pars().fluxConvert else 1.) * dv
        unit = k * conv / (arcs * arcs)                 # (u/arcsec^2 of the map of ELLPROF per unit of DENS)
    return dict(radii=np.asarray(rings.radii, float), dens=dens, sdens_fo=dens * unit, err_lo=lo * unit,
                err_hi=hi * unit, dens_err_lo=lo, dens_err_hi=hi, nmc=int(nmc), beam=abs(h.Bmaj()) * 3600.,
                map2d=is2d)
