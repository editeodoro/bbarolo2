"""
Density profiles corrected for the beam (deconvolved), shared by BayesianBBarolo and the initial
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


def ellprof_deconvolved(c, rings, side="B", smooth=0.1, nmc=200, seed=0):
    """The deconvolved radial profile of the ELLPROF task: the face-on surface densities of the rings whose
    beam-convolved model reproduces the total intensity map of the cube c, as in BayesianBBarolo (see the
    module docstring), at the fixed geometry of the rings (a Rings object: radii, centre, INC, PA, Z0).

    The data map is the sum of the data, not clipped, in the mask of the cube dilated by one beam and +-3
    channels (the mask cuts the faint outer emission, which the model cannot reproduce); its noise is the rms
    of the data outside the mask times the square root of the number of channels summed and of the spectral
    correlation of the noise. The correlation of the noise is the conservative measure of BayesianBBarolo (at
    least one beam area spatially). With side 'A' or 'R', only the pixels of that half of the galaxy are used
    (divided by the minor axis of the rings: their median centre and PA). The errors come from nmc refits of the
    best-fit map plus noise with that rms and spatial correlation (the geometry is fixed: no errors from it).

    Returns None for a cube with fewer than 2 channels, without a beam, or with too few pixels; otherwise a dict
    with radii (arcsec), dens (the densities of the model, as DENS of GALMOD), sdens_fo (the face-on surface
    density in the units of the map of ELLPROF per arcsec^2, as its column SDENS_FO), err_lo, err_hi (1-sigma
    errors of sdens_fo), nmc and beam (FWHM, arcsec)."""
    import copy
    from .galmod import DiskModel
    from .rings import Rings
    from .bbutils import arcsconv, DeltaVel, FluxtoJyBeam
    from .bayesian.noise import dilate_mask, noise_correlation
    h = c.Head()
    if c.DimZ() < 2 or not h.Bmaj() or not h.Cdelt(0):
        return None
    bpix = abs(h.Bmaj() / h.Cdelt(0))
    pixarea = abs(h.Cdelt(0) * h.Cdelt(1))
    beam_area = math.pi * abs(h.Bmaj() * h.Bmin()) / (4. * math.log(2.)) / pixarea if h.Bmin() else 1.
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
        pa = math.radians(float(np.median(rings.phi)))
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
    dm = DiskModel(c, par=par, ringmode="CENTRED", galmodtype="ANALYTIC", smooth=True)
    sysm = profile_system(dm, r, (blo, bhi), dmap[sl], rms[sl], ok[sl], beam_arcsec=abs(h.Bmaj()) * 3600.)
    if sysm is None:
        return None
    A, b, oks, Rc = sysm
    sol = solve_profile(A, b, Rc, smooth)
    dens = sol[:nr]
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
            prof.append(solve_profile(A, bmod + g[oks], Rc, smooth)[:nr])
        p16, p50, p84 = np.percentile(np.asarray(prof), [16, 50, 84], axis=0)
        lo, hi = p50 - p16, p84 - p50
    # the face-on surface brightness of unit density (per pixel, in the units of the sum of the channels): the
    # flux of a face-on uniform disk of unit density divided by its area (the clouds of the model sit on sub-rings,
    # so single pixels are not uniform), at the centre of the cube
    from .galmod import outer_edge
    xm, ym = nx // 2, ny // 2
    w = max(min(nx, ny) / 8., 2.) * arcs
    uni = Rings().setRings(radii=np.array([w, 2 * w, 3 * w]), xpos=xm, ypos=ym, vsys=float(rings.vsys[0]), vrot=0.,
                           vdisp=8., z0=0., inc=0., phi=0., dens=1., radsep=w)
    redge = outer_edge(uni, "CENTRED") / arcs
    k = float(dm.ring_maps(uni).sum()) / (math.pi * redge * redge)
    dv = abs(DeltaVel(h)) if c.HeadDef() else 1.
    conv = (FluxtoJyBeam(1., h) if c.pars().fluxConvert else 1.) * dv
    unit = k * conv / (arcs * arcs)                     # (u/arcsec^2 of the map of ELLPROF per unit of DENS)
    return dict(radii=np.asarray(rings.radii, float), dens=dens, sdens_fo=dens * unit, err_lo=lo * unit,
                err_hi=hi * unit, dens_err_lo=lo, dens_err_hi=hi, nmc=int(nmc), beam=abs(h.Bmaj()) * 3600.)
