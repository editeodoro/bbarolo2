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
