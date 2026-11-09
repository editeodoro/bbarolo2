"""
3D kinematic models of rotating disks (port of Tasks/galmod.cpp, i.e. GIPSY's GALMOD).

Two model types are available (parameter ``GALMODTYPE``):

* ``MC`` (default) -- the classic GALMOD Monte Carlo scheme: clouds are randomly
  placed in each (sub)ring and each cloud is split in ``NV`` sub-clouds with
  Gaussian velocity deviates. Same algorithm, defaults and normalisations as the
  C++ code. The kernel is compiled with numba and the random generator is seeded
  with ``ISEED`` at each model evaluation (common random numbers across the
  evaluations of a fit, as in BBarolo).
* ``ANALYTIC`` -- a noise-free version: clouds are placed on a regular
  (stratified) grid in radius/azimuth/height and each cloud deposits the
  analytic integral of its Gaussian line profile over each channel. This is the
  expectation value of the Monte-Carlo model and gives smooth chi-square surfaces.

The engine of all the tasks is DiskModel (at the end of this module): it holds the
quantities derived from the header of a cube (GalmodSetup, which builds the models
before the convolution) and the beam kernel, and builds models of a set of rings with
a ring convention (parameter ``RINGMODE``, see model_rings): by default each ring is an
annulus centred on its radius (``CENTRED``), as the rings fitted by 3DFIT.
"""

import sys
import math
import numpy as np
import numba as nb
from concurrent.futures import ThreadPoolExecutor

from .bbutils import BBaroloError, DeltaVel, C_MS, arcsconv
from ._rng import rng_init, rng_uniform, rng_normal
from .smooth import Smooth3D, FFTConvolver


# --------------------------------------------------------------------------
# numba helpers
# --------------------------------------------------------------------------
@nb.njit(inline="always", cache=True)
def _lround(x):
    if not abs(x) < 1e18:
        # NaN or infinite (e.g. a NaN ring parameter): int() of them is undefined (0 on ARM, which
        # put the flux in pixel or channel 0); a value far outside any grid is skipped by the callers
        return -(1 << 62)
    if x >= 0:
        return int(x + 0.5)
    return -int(-x + 0.5)


@nb.njit(inline="always", cache=True)
def _velgrid(v, axtyp, velsys, freq0, crval3, cdelt3, crpix3):
    if axtyp == 4:
        w = v
    else:
        if velsys == 1:
            w = freq0 * (1 - v / C_MS)
        elif velsys == 2:
            w = freq0 * (C_MS / (C_MS + v))
        else:
            w = freq0 * math.sqrt((C_MS - v) / (C_MS + v))
        if axtyp == 2:
            w = C_MS / w
    return (w - crval3) / cdelt3 + crpix3 - 1


@nb.njit(inline="always", cache=True)
def _fdev(ltype, st):
    if ltype == 1:
        return rng_normal(st)
    x = 2.0 * rng_uniform(st) - 1.0
    if ltype == 2:
        return math.atanh(x)
    elif ltype == 3:
        if x >= 0.0:
            return -math.log(x)
        return math.log(-x)
    elif ltype == 4:
        return math.tan(0.5 * math.pi * x)
    return x


@nb.njit(nogil=True, cache=True)
def _galmod_mc(out, sr, ring_lo, ring_hi, seed, radsep, nv, cdens, cmode, pixarea, cdelt0, cdelt1,
               crota2, blo0, blo1, bhi0, bhi1, ltype, sig_instr, axtyp, velsys, lfreq0, crval3,
               cdelt3, crpix3, relint, cd2i):
    """Monte-Carlo GALMOD kernel. sr: (14, nsub) array of sub-ring values in internal units:
    radius, vrot, vrad, vvert, vdisp, dens, z0, dvdz, zcyl, inc, phi, xpos, ypos, vsys. lfreq0, relint: the
    reference frequency (the rest frequency at REDSHIFT) and the relative intensity of each line."""
    st = rng_init(seed)
    nz = out.shape[2]
    nlines = lfreq0.shape[0]
    linear = axtyp == 4 and nlines == 1
    ia = 1.0 / cdelt3
    ib = -crval3 / cdelt3 + crpix3 - 1
    for ir in range(ring_lo, ring_hi):
        R0 = sr[0, ir]
        dens = sr[5, ir]
        if dens == 0:
            continue
        ring_area_pix = 2 * math.pi * R0 * radsep / pixarea
        nc = _lround(cdens * dens ** cmode * ring_area_pix)
        if _lround(nc / ring_area_pix) < 10.:
            nc = _lround(10 * ring_area_pix)
        vrot, vrad, vvert = sr[1, ir], sr[2, ir], sr[3, ir]
        vdisptmp = math.sqrt(sr[4, ir] * sr[4, ir] + sig_instr * sig_instr)
        z0, dvdz, zcyl = sr[6, ir], sr[7, ir], sr[8, ir]
        sinc, cinc = math.sin(sr[9, ir]), math.cos(sr[9, ir])
        phi = sr[10, ir]
        spa = math.sin(phi) * math.cos(crota2) + math.cos(phi) * math.sin(crota2)
        cpa = math.cos(phi) * math.cos(crota2) - math.sin(phi) * math.sin(crota2)
        xpos, ypos, vsys0 = sr[11, ir], sr[12, ir], sr[13, ir]
        nvr = nv[ir]
        fluxsc = np.float32(dens * ring_area_pix * pixarea / (nc * nvr))
        r_inner = R0 - 0.5 * radsep
        r_inner_sq = r_inner * r_inner
        r_cross = 2.0 * radsep * R0
        for ic in range(nc):
            R = math.sqrt(r_inner_sq + r_cross * abs(2.0 * rng_uniform(st) - 1.0))
            az = 2 * math.pi * abs(2.0 * rng_uniform(st) - 1.0)
            saz = math.sin(az)
            caz = math.cos(az)
            z = _fdev(ltype, st) * z0
            x = R * caz
            y = R * saz * cinc - z * sinc
            g0 = _lround(xpos + (x * spa - y * cpa) / cdelt0)
            g1 = _lround(ypos + (x * cpa + y * spa) / cdelt1)
            if g0 < blo0 or g0 >= bhi0 or g1 < blo1 or g1 >= bhi1:
                # the velocity deviates are drawn all the same, so that the model in a box is the crop of
                # the model of the whole cube (C++ did not draw them)
                for iv in range(nvr):
                    rng_normal(st)
                continue
            px = g0 - blo0
            py = g1 - blo1
            az_ = abs(z)
            if az_ > zcyl:
                vs = vsys0 + ((vrot - dvdz * (az_ - zcyl)) * caz + vrad * saz) * sinc
            else:
                vs = vsys0 + (vrot * caz + vrad * saz) * sinc
            vs += vvert * cinc
            if linear:
                # velocity axis, single line: linear velocity->channel mapping
                g0 = vs * ia + ib
                sg = vdisptmp * ia
                val = relint[0] * fluxsc
                for iv in range(nvr):
                    isubs = _lround(g0 + rng_normal(st) * sg)
                    if isubs < 0 or isubs >= nz:
                        continue
                    out[py, px, isubs] += val * cd2i[isubs]
            else:
                for iv in range(nvr):
                    v = vs + rng_normal(st) * vdisptmp
                    for nl in range(nlines):
                        isubs = _lround(_velgrid(v, axtyp, velsys, lfreq0[nl], crval3, cdelt3, crpix3))
                        if isubs < 0 or isubs >= nz:
                            continue
                        out[py, px, isubs] += relint[nl] * fluxsc * cd2i[isubs]
    return out


# Tabulated standard normal CDF, Phi(t) = (1 + erf(t/sqrt(2)))/2, for the analytic models.
# Linear interpolation on a 1/512 grid: |error| < 5e-7. Beyond +-6 sigma Phi is 0 or 1.
_PHI_TMAX = 6.0
_PHI_ISTEP = 512.0
_WARNED_LINES = set()         # (the cubes already warned about several lines on a velocity axis)
_PHI_TAB = np.array([0.5 * math.erfc(-(-_PHI_TMAX + k / _PHI_ISTEP) / math.sqrt(2.0))
                     for k in range(int(2 * _PHI_TMAX * _PHI_ISTEP) + 2)])


@nb.njit(inline="always", cache=True)
def _phi(t, tab):
    x = (t + 6.0) * 512.0
    if x <= 0.0:
        return 0.0
    k = int(x)
    if k >= tab.shape[0] - 1:
        return 1.0
    return tab[k] + (x - k) * (tab[k + 1] - tab[k])


@nb.njit(inline="always", cache=True)
def _line_profile(prof, v, sig, vedges, cd2i, ptab, kc, K, nz):
    """Channel-integrated Gaussian line (centre v, dispersion sig) times cd2i, for the
    channels kc-K..kc+K (shared channel edges). Returns the first channel and the number
    of channels written in prof."""
    k0 = kc - K
    if k0 < 0:
        k0 = 0
    k1 = kc + K + 1
    if k1 > nz:
        k1 = nz
    if k1 <= k0:
        return 0, 0
    inv = 1.0 / sig
    pp = _phi((vedges[k0] - v) * inv, ptab)
    for k in range(k0, k1):
        pn = _phi((vedges[k + 1] - v) * inv, ptab)
        f = pn - pp
        prof[k - k0] = (f if f >= 0 else -f) * cd2i[k]
        pp = pn
    return k0, k1 - k0


@nb.njit(inline="always", cache=True)
def _layer_cdf(ltype, x):
    """Fraction of the flux of the vertical layer below the height x (in units of z0), for the layers of
    _fdev: 1 Gaussian, 2 sech2, 3 exponential, 4 Lorentzian, other: box."""
    if ltype == 1:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
    if ltype == 2:
        return 0.5 * (1.0 + math.tanh(x))
    if ltype == 3:
        return 1.0 - 0.5 * math.exp(-x) if x >= 0 else 0.5 * math.exp(x)
    if ltype == 4:
        return 0.5 + math.atan(x) / math.pi
    return 0.5 * (min(max(x, -1.0), 1.0) + 1.0)


@nb.njit(inline="always", cache=True)
def _layer_extent(ltype):
    """The height (units of z0) beyond which the layer has less than 1E-6 of its flux (its edge for the box;
    none for the Lorentzian layer, limited by the box only)."""
    if ltype == 1:
        return 5.0
    if ltype == 2:
        return 7.6
    if ltype == 3:
        return 14.0
    if ltype == 4:
        return 1e30
    return 1.0


@nb.njit(inline="always", cache=True)
def _layer_core(ltype):
    """The height (units of z0) beyond which the density of the layer is below 1E-3 of its peak (none for the
    box)."""
    if ltype == 1:
        return 3.717
    if ltype == 2:
        return 4.147
    if ltype == 3:
        return 6.908
    if ltype == 4:
        return 31.61
    return 1e30


ZSAMPLES_MAX = 1024         # (the maximum number of vertical samples of a ring of the ANALYTIC models)
# The spacing of the vertical samples of the ANALYTIC models once projected (pixels; see _zgrid and
# DiskModel.zstep): ZSTEP_FINE, or ZSTEP_COARSE for models convolved with a beam of at least ZSTEP_BEAM pixels
# (minor axis). Odd multiples of 1/4: the samples, half a step from the plane, are then at odd multiples of 1/8
# pixel from it, never at the half pixels where a centre at a pixel edge would put them on the edges of the
# pixels (1 pixel: 2.4% of the peak in the total intensity maps after the beam). With fewer samples per pixel the
# nearest-pixel deposit aliases on the pixel grid, which the beam smooths out: at 0.75 pixels the model is within
# 0.8% of the peak of the 0.25-pixel one after a beam of 3 pixels (2% at 2 pixels, 12-48% without a beam), and the
# model is 1.1 and ring_maps 1.8 times faster (NGC 2403, Z0 = 10" on 16" pixels).
ZSTEP_FINE = 0.25
ZSTEP_COARSE = 0.75
ZSTEP_BEAM = 3.0


@nb.njit(nogil=True, cache=True)
def _zgrid(zq, wq, nmax, ltype, z0, sinc, pixsize, dvdz, sig, R0, radsep, xpos, ypos, cdelt0, cdelt1,
           blo0, blo1, bhi0, bhi1, zstep):
    """The vertical samples of a sub-ring of the ANALYTIC models: heights zq[:n] (in units of z0) and weights
    wq[:n] (the fraction of the flux of the layer around each height). Returns n.

    The heights are evenly spaced, at most zstep pixels apart once projected (pixsize; ZSTEP_FINE or ZSTEP_COARSE)
    and, with a vertical
    gradient dvdz, 0.5 sig apart in velocity (or the line profiles form a comb), symmetric about the plane, offset
    by half a step from it (a sample in the plane, or at multiples of half a pixel from it, would fall on the edges of
    the pixels when the centre is at a pixel edge, and the rounding would give its flux to either side); each
    weighs the flux of the layer between the midpoints with its neighbours. They reach the height beyond which the
    layer has less than 1E-6 of its flux, or no cloud can fall in the box [blo, bhi): a cloud of the ring centred at
    (xpos, ypos) (pixels) higher than (R0 + radsep + distance to the farthest corner of the box) / |sin i| is outside
    it; that flux is not deposited, as the clouds of the MC models outside the field. The spacing does not depend
    on the box, so the samples of a box model are those of the full model in it (up to nmax samples).
    (Equal-weight samples at the quantiles of the layer were up to several pixels apart in its tails, at any number
    of samples: models of thick or Lorentzian layers were off by up to 20% of the peak.)
    A face-on or thin layer has one sample."""
    s = abs(sinc)
    if z0 <= 0 or nmax < 2 or s < 1e-6:
        zq[0] = 0.0
        wq[0] = 1.0
        return 1
    dx = max(abs(blo0 - 0.5 - xpos), abs(bhi0 - 0.5 - xpos)) * abs(cdelt0)
    dy = max(abs(blo1 - 0.5 - ypos), abs(bhi1 - 0.5 - ypos)) * abs(cdelt1)
    zlim = min(_layer_extent(ltype), (R0 + radsep + math.sqrt(dx * dx + dy * dy)) / (s * z0))
    step = zstep * pixsize / (z0 * s)
    if dvdz != 0:
        step = min(step, 0.5 * sig / (abs(dvdz) * z0 * s))
    # the bins of one side, from the plane: step wide up to the core of the layer, 4 steps beyond (the far tail,
    # where the density is below 1E-3 of the peak: the long tail of the Lorentzian layer); edges at multiples of step
    zc = _layer_core(ltype)
    while True:
        nin = max(int(math.ceil(min(zc, zlim) / step)), 1)
        nout = max(int(math.ceil((zlim - nin * step) / (4.0 * step))), 0)
        if 2 * (nin + nout) <= nmax:
            break
        step *= 1.01 * (nin + nout) / (nmax // 2)
    half = nin + nout
    for j in range(half):
        lo = j * step if j < nin else (nin + 4 * (j - nin)) * step
        hi = min(lo + (step if j < nin else 4.0 * step), zlim)
        w = _layer_cdf(ltype, hi) - _layer_cdf(ltype, lo)
        zq[half + j], wq[half + j] = 0.5 * (lo + hi), w
        zq[half - 1 - j], wq[half - 1 - j] = -0.5 * (lo + hi), w
    n = 2 * half
    return n


@nb.njit(nogil=True, cache=True)
def _galmod_analytic(out, sr, ring_lo, ring_hi, radsep, cdens, pixarea, cdelt0, cdelt1, crota2,
                     blo0, blo1, bhi0, bhi1, ltype, sig_instr, vedges, relint, cd2i, nzq,
                     axtyp, velsys, lfreq0, crval3, cdelt3, crpix3, chwidth, ptab, zstep):
    """Deterministic GALMOD: clouds on a regular grid in radius/azimuth/height (nearest-pixel
    deposit, as in the Monte-Carlo GALMOD) with analytic channel-integrated line profiles.

    The line profile of an (R, azimuth) sample is computed once and shared by its vertical
    samples (unless the rotation depends on height), consecutive samples falling in the
    same pixel are deposited together, and the Gaussian integrals use a tabulated CDF
    with shared channel edges."""
    nz = out.shape[2]
    nlines = lfreq0.shape[0]           # (vedges: the channel edges in the velocity of each line)
    pixsize = math.sqrt(pixarea)
    zq = np.empty(max(nzq, 1))
    wq = np.empty(max(nzq, 1))
    prof = np.empty(nz + 1)
    gx = np.empty(max(nzq, 1), np.int64)
    gy = np.empty(max(nzq, 1), np.int64)
    for ir in range(ring_lo, ring_hi):
        R0 = sr[0, ir]
        dens = sr[5, ir]
        if dens == 0:
            continue
        ring_area_pix = 2 * math.pi * R0 * radsep / pixarea
        circ_pix = 2 * math.pi * R0 / math.sqrt(pixarea)
        naz = max(int(3 * circ_pix + 0.5), 16)
        nrad = 2
        vrot, vrad, vvert = sr[1, ir], sr[2, ir], sr[3, ir]
        # (VDISP=0 with no instrumental broadening: a line narrower than a channel, in one channel)
        sig = max(math.sqrt(sr[4, ir] * sr[4, ir] + sig_instr * sig_instr), 1e-6 * abs(chwidth))
        K = int(5.0 * sig / chwidth) + 2      # beyond 5 sigma: < 3e-7 of the line
        z0, dvdz, zcyl = sr[6, ir], sr[7, ir], sr[8, ir]
        sinc, cinc = math.sin(sr[9, ir]), math.cos(sr[9, ir])
        phi = sr[10, ir]
        spa = math.sin(phi) * math.cos(crota2) + math.cos(phi) * math.sin(crota2)
        cpa = math.cos(phi) * math.cos(crota2) - math.sin(phi) * math.sin(crota2)
        xpos, ypos, vsys0 = sr[11, ir], sr[12, ir], sr[13, ir]
        # vertical samples (heights zq, weights wq: see _zgrid)
        nzz = _zgrid(zq, wq, nzq, ltype, z0, sinc, pixsize, dvdz, sig, R0, radsep, xpos, ypos, cdelt0, cdelt1,
                     blo0, blo1, bhi0, bhi1, zstep)
        fluxsc = dens * ring_area_pix * pixarea / (naz * nrad)
        # does the velocity depend on height for some vertical sample?
        zdep = False
        if dvdz != 0 and nzz > 1:
            for kz in range(nzz):
                if abs(zq[kz] * z0) > zcyl:
                    zdep = True
                    break
        r_inner = R0 - 0.5 * radsep
        daz = 2 * math.pi / naz
        cd, sd = math.cos(daz), math.sin(daz)
        for kr in range(nrad):
            u = (kr + 0.5) / nrad
            R = math.sqrt(r_inner * r_inner + 2.0 * radsep * R0 * u)
            az0 = 2 * math.pi * (0.5 * ((kr + 1) % 2) + 0.25) / naz
            caz, saz = math.cos(az0), math.sin(az0)
            for ka in range(naz):
                if ka > 0:
                    caz, saz = caz * cd - saz * sd, saz * cd + caz * sd
                x = R * caz
                yb = R * saz * cinc
                xs = x * spa
                xc = x * cpa
                # pixels of the vertical samples
                for kz in range(nzz):
                    z = zq[kz] * z0 if nzz > 1 else 0.0
                    y = yb - z * sinc
                    gx[kz] = _lround(xpos + (xs - y * cpa) / cdelt0)
                    gy[kz] = _lround(ypos + (xc + y * spa) / cdelt1)
                vplane = vsys0 + (vrot * caz + vrad * saz) * sinc + vvert * cinc
                for nl in range(nlines):
                    a = relint[nl] * fluxsc
                    if not zdep:
                        v = vplane
                        kc = _lround(_velgrid(v, axtyp, velsys, lfreq0[nl], crval3, cdelt3, crpix3))
                        k0, nk = _line_profile(prof, v, sig, vedges[nl], cd2i, ptab, kc, K, nz)
                        if nk == 0:
                            continue
                        kz = 0
                        while kz < nzz:
                            g0, g1 = gx[kz], gy[kz]
                            ws = wq[kz]
                            w = 1
                            while kz + w < nzz and gx[kz + w] == g0 and gy[kz + w] == g1:
                                ws += wq[kz + w]
                                w += 1
                            kz += w
                            if g0 < blo0 or g0 >= bhi0 or g1 < blo1 or g1 >= bhi1:
                                continue
                            aw = a * ws
                            o = out[g1 - blo1, g0 - blo0]
                            for j in range(nk):
                                o[k0 + j] += aw * prof[j]
                    else:
                        # the velocity depends on |z|: the samples at +z and -z share the profile
                        for kz in range((nzz + 1) // 2):
                            kz2 = nzz - 1 - kz
                            in1 = not (gx[kz] < blo0 or gx[kz] >= bhi0 or gy[kz] < blo1 or gy[kz] >= bhi1)
                            in2 = kz2 != kz and not (gx[kz2] < blo0 or gx[kz2] >= bhi0 or gy[kz2] < blo1
                                                     or gy[kz2] >= bhi1)
                            if not (in1 or in2):
                                continue
                            az_ = abs(zq[kz] * z0)
                            if az_ > zcyl:
                                v = vsys0 + ((vrot - dvdz * (az_ - zcyl)) * caz + vrad * saz) * sinc + vvert * cinc
                            else:
                                v = vplane
                            kc = _lround(_velgrid(v, axtyp, velsys, lfreq0[nl], crval3, cdelt3, crpix3))
                            k0, nk = _line_profile(prof, v, sig, vedges[nl], cd2i, ptab, kc, K, nz)
                            if nk == 0:
                                continue
                            if in1 and in2 and gx[kz] == gx[kz2] and gy[kz] == gy[kz2]:
                                o = out[gy[kz] - blo1, gx[kz] - blo0]
                                aw = a * (wq[kz] + wq[kz2])
                                for j in range(nk):
                                    o[k0 + j] += aw * prof[j]
                                continue
                            if in1:
                                o = out[gy[kz] - blo1, gx[kz] - blo0]
                                aw = a * wq[kz]
                                for j in range(nk):
                                    o[k0 + j] += aw * prof[j]
                            if in2:
                                o = out[gy[kz2] - blo1, gx[kz2] - blo0]
                                aw = a * wq[kz2]
                                for j in range(nk):
                                    o[k0 + j] += aw * prof[j]
    return out


@nb.njit(nogil=True, cache=True)
def _ring_maps_analytic(out, sr, widx, wval, radsep, pixarea, cdelt0, cdelt1, crota2, blo0, blo1, bhi0, bhi1,
                        ltype, nzq, sig_instr, chwidth, zstep):
    """Total intensity (before the channel factor cd2i and the convolution) of the ANALYTIC
    GALMOD deposit of the sub-rings sr with unit density, accumulated in out[ring] (nr, ny, nx)
    with the interpolation weights of each sub-ring: wval[s, m] for the ring widx[s, m] (m = 0, 1;
    -1: none). Same samples as _galmod_analytic (its vertical samples depend also on the velocity
    dispersion and the vertical gradient: sig_instr and chwidth as there)."""
    pixsize = math.sqrt(pixarea)
    zq = np.empty(max(nzq, 1))
    wq = np.empty(max(nzq, 1))
    gx = np.empty(max(nzq, 1), np.int64)
    gy = np.empty(max(nzq, 1), np.int64)
    for ir in range(sr.shape[1]):
        if widx[ir, 0] < 0 and widx[ir, 1] < 0:
            continue
        R0 = sr[0, ir]
        ring_area_pix = 2 * math.pi * R0 * radsep / pixarea
        circ_pix = 2 * math.pi * R0 / math.sqrt(pixarea)
        naz = max(int(3 * circ_pix + 0.5), 16)
        nrad = 2
        z0 = sr[6, ir]
        sinc, cinc = math.sin(sr[9, ir]), math.cos(sr[9, ir])
        phi = sr[10, ir]
        spa = math.sin(phi) * math.cos(crota2) + math.cos(phi) * math.sin(crota2)
        cpa = math.cos(phi) * math.cos(crota2) - math.sin(phi) * math.sin(crota2)
        xpos, ypos = sr[11, ir], sr[12, ir]
        sig = max(math.sqrt(sr[4, ir] * sr[4, ir] + sig_instr * sig_instr), 1e-6 * abs(chwidth))
        nzz = _zgrid(zq, wq, nzq, ltype, z0, sinc, pixsize, sr[7, ir], sig, R0, radsep, xpos, ypos, cdelt0,
                     cdelt1, blo0, blo1, bhi0, bhi1, zstep)
        fluxsc = ring_area_pix * pixarea / (naz * nrad)
        r_inner = R0 - 0.5 * radsep
        daz = 2 * math.pi / naz
        cd, sd = math.cos(daz), math.sin(daz)
        for kr in range(nrad):
            u = (kr + 0.5) / nrad
            R = math.sqrt(r_inner * r_inner + 2.0 * radsep * R0 * u)
            az0 = 2 * math.pi * (0.5 * ((kr + 1) % 2) + 0.25) / naz
            caz, saz = math.cos(az0), math.sin(az0)
            for ka in range(naz):
                if ka > 0:
                    caz, saz = caz * cd - saz * sd, saz * cd + caz * sd
                x = R * caz
                yb = R * saz * cinc
                xs = x * spa
                xc = x * cpa
                for kz in range(nzz):
                    z = zq[kz] * z0 if nzz > 1 else 0.0
                    y = yb - z * sinc
                    gx[kz] = _lround(xpos + (xs - y * cpa) / cdelt0)
                    gy[kz] = _lround(ypos + (xc + y * spa) / cdelt1)
                kz = 0
                while kz < nzz:                 # (consecutive samples in the same pixel together)
                    g0, g1 = gx[kz], gy[kz]
                    ws = wq[kz]
                    kz += 1
                    while kz < nzz and gx[kz] == g0 and gy[kz] == g1:
                        ws += wq[kz]
                        kz += 1
                    if g0 < blo0 or g0 >= bhi0 or g1 < blo1 or g1 >= bhi1:
                        continue
                    for m in range(2):
                        r = widx[ir, m]
                        if r >= 0:
                            out[r, g1 - blo1, g0 - blo0] += fluxsc * ws * wval[ir, m]
    return out


@nb.njit(nogil=True, cache=True)
def _galmod_wind(out, sr, ring_lo, ring_hi, seed, sep, nv, cdens, cmode, pixarea, cdelt0, cdelt1,
                 crota2, blo0, blo1, bhi0, bhi1, sig_instr, axtyp, velsys, lfreq0, crval3, cdelt3,
                 crpix3, relint, cd2i):
    """GALMOD for spherical (bi-conical) winds. sr rows: radius, vrot, vsph, vdisp, dens, inc,
    pa, openang, xpos, ypos, vsys (internal units)."""
    st = rng_init(seed)
    nz = out.shape[2]
    nlines = lfreq0.shape[0]
    for ir in range(ring_lo, ring_hi):
        rtmp = sr[0, ir]
        nc = _lround(cdens * sr[4, ir] ** cmode * 2 * math.pi * rtmp * sep / pixarea)
        if nc == 0:
            continue
        vrottmp, vsphtmp = sr[1, ir], sr[2, ir]
        opentmp = sr[7, ir] / 2.
        vdisptmp = math.sqrt(sr[3, ir] * sr[3, ir] + sig_instr * sig_instr)
        vsystmp = sr[10, ir]
        sinc, cinc = math.sin(sr[5, ir]), math.cos(sr[5, ir])
        pa = sr[6, ir]
        spa = math.sin(pa) * math.cos(crota2) + math.cos(pa) * math.sin(crota2)
        cpa = math.cos(pa) * math.cos(crota2) - math.sin(pa) * math.sin(crota2)
        nvr = nv[ir]
        fluxsc = sr[4, ir] * 2 * math.pi * rtmp * sep / (nc * nvr)
        for ic in range(nc):
            rad = math.sqrt((rtmp - 0.5 * sep) ** 2 + 2 * sep * rtmp * abs(2.0 * rng_uniform(st) - 1.0))
            az = 2 * math.pi * abs(2.0 * rng_uniform(st) - 1.0)
            saz, caz = math.sin(az), math.cos(az)
            ph = opentmp * abs(2.0 * rng_uniform(st) - 1.0)
            if ic % 2 == 0:
                ph = math.pi - ph
            sphi, cphi = math.sin(ph), math.cos(ph)
            R = rad * sphi
            z = rad * cphi
            x = R * caz
            y = R * saz * cinc - z * sinc
            g0 = _lround(sr[8, ir] + (x * spa - y * cpa) / cdelt0)
            g1 = _lround(sr[9, ir] + (x * cpa + y * spa) / cdelt1)
            if g0 < blo0 or g0 >= bhi0 or g1 < blo1 or g1 >= bhi1:
                for iv in range(nvr):              # (as in _galmod_mc: the box model is a crop)
                    rng_normal(st)
                continue
            px, py = g0 - blo0, g1 - blo1
            vs = vsystmp + vsphtmp * (cphi * cinc + sphi * saz * sinc) + vrottmp * caz * sinc
            for iv in range(nvr):
                v = vs + rng_normal(st) * vdisptmp
                for nl in range(nlines):
                    isubs = _lround(_velgrid(v, axtyp, velsys, lfreq0[nl], crval3, cdelt3, crpix3))
                    if isubs < 0 or isubs >= nz:
                        continue
                    out[py, px, isubs] += relint[nl] * fluxsc * cd2i[isubs]
    return out


def _seed(iseed):
    return int(iseed) & 0xFFFFFFFF


# --------------------------------------------------------------------------
# Model setup
# --------------------------------------------------------------------------
def _relint(relint, nlines, name):
    """The relative intensities of nlines lines (RELINT), checked: one per line (the kernels read one per
    line: a shorter RELINT read beyond the array, in C++ too for GALWIND)."""
    if len(relint) != nlines:
        raise BBaroloError(f"GALMOD ERROR: {name} gives {nlines} lines, but RELINT has {len(relint)} values: give "
                           "one relative intensity per line.")
    return np.array(relint, float)


class GalmodSetup:
    """All the header-derived quantities needed to build models for a given cube.
    Computed once and reused for every model evaluation (port of Galmod::initialize)."""

    def __init__(self, cube, gmpar=None, galmodtype=None, densflux=None):
        h = cube.Head()
        p = cube.pars()
        gm = p.gm if gmpar is None else gmpar
        self.nsubs = cube.DimZ()
        self.dimx, self.dimy = cube.DimX(), cube.DimY()
        arcmconv = 0
        cdelt = [0., 0.]
        for i in range(2):
            cu = h.Cunit(i).lower()
            if "deg" in cu: arcmconv = 60.
            elif "arcs" in cu: arcmconv = 1 / 60.
            elif "arcm" in cu: arcmconv = 1.
            else:
                raise BBaroloError(f"GALMOD error (unknown CUNIT for RA-DEC): cannot convert to ARCMIN.\n{cu}")
            cdelt[i] = h.Cdelt(i) * arcmconv
        self.arcmconv = arcmconv
        self.cdelt = cdelt
        self.pixsize = [abs(cdelt[0]), abs(cdelt[1])]
        self.pixarea = self.pixsize[0] * self.pixsize[1]
        self.crota2 = h.Crota() * math.pi / 180.
        self.zstep = ZSTEP_FINE         # (vertical samples of the ANALYTIC models: see DiskModel.zstep)
        reds = p.getRedshift()
        cunit3 = h.Cunit(2).lower()
        self.cdelt3 = h.Cdelt(2)
        self.crval3 = h.Crval(2)
        self.crpix3 = h.Crpix(2)
        self.velsys = {"radio": 1, "optical": 2, "relativistic": 3}.get(h.VelDef(), 1)
        sptype = h.SpectralType()
        # several lines: the reference frequency of each (its rest frequency at REDSHIFT), so that every line
        # of a cloud has the same Doppler factor (C++ added a fixed velocity offset to each line, which did
        # not scale with the velocity of the cloud)
        lfreq0 = None
        self.relint = np.array([1.])
        self.freq0 = 0.1420405751786E10
        if sptype == "wave":
            self.axtyp = 2
            if cunit3 in ("um", "mum", "micron"): mconv = 1.0E-06
            elif cunit3 in ("nm", "nanom"): mconv = 1.0E-09
            elif cunit3 in ("a", "ang", "angstrom"): mconv = 1.0E-10
            else:
                raise BBaroloError(f"GALMOD ERROR: unknown CUNIT3={cunit3}. Cannot convert to meters.")
            self.crval3 *= mconv
            self.cdelt3 *= mconv
            restw = p.getRestwave()
            self.freq0 = C_MS / (restw * (1 + reds) * mconv) if restw != -1 else C_MS / self.crval3
            if len(gm.RESTWAVE) > 1:
                lfreq0 = C_MS / (np.asarray(gm.RESTWAVE, float) * (1 + reds) * mconv)
                self.relint = _relint(gm.RELINT, len(gm.RESTWAVE), "RESTWAVE")
        elif sptype == "freq":
            self.axtyp = 3
            hz = {"hz": 1, "khz": 1E03, "mhz": 1E06, "ghz": 1E09}.get(cunit3)
            if hz is None:
                raise BBaroloError(f"GALMOD ERROR: unknown CUNIT3={cunit3}. Cannot convert to Hz.")
            self.crval3 *= hz
            self.cdelt3 *= hz
            self.freq0 = h.Freq0() / (1 + reds)
            if self.freq0 == 0:
                self.freq0 = 0.1420405751786E10
            if len(gm.RESTFREQ) > 1:
                # observed frequency of each line: f/(1+z) (C++ used f*(1+z))
                lfreq0 = np.asarray(gm.RESTFREQ, float) / (1 + reds)
                self.relint = _relint(gm.RELINT, len(gm.RESTFREQ), "RESTFREQ")
        elif "velo" in sptype:
            self.axtyp = 4
            # the units of AlltoVel and DeltaVel (C++ took any other unit, e.g. kms, as cm/s)
            msconv = {"m/s": 1., "ms": 1., "km/s": 1.0E03, "kms": 1.0E03}.get(cunit3)
            if msconv is None:
                raise BBaroloError(f"GALMOD ERROR: unknown CUNIT3={cunit3}. Cannot convert to m/s.")
            self.freq0 = h.Freq0() / (1 + reds)
            if self.freq0 == 0:
                self.freq0 = 0.1420405751786E10
            self.crval3 *= msconv
            self.cdelt3 *= msconv
            if max(len(gm.RESTWAVE), len(gm.RESTFREQ)) > 1 and h.fitsname not in _WARNED_LINES:
                _WARNED_LINES.add(h.fitsname)
                sys.stderr.write(" GALMOD WARNING: several lines (RESTWAVE/RESTFREQ) on a velocity axis: only one "
                                 "line is modelled.\n")
        else:
            raise BBaroloError("GALMOD ERROR: unknown Spectral axis type. Check CTYPE3 and CUNIT3 keywords.")
        nch = p.specres if getattr(p, "specres", -1) != -1 else p.linear     # (LINEAR: as in C++)
        if nch == -1:
            nch = 1. / (2 * math.sqrt(2 * math.log(2)))
        self.chwidth = abs(DeltaVel(h)) * 1000
        self.sig_instr = nch * self.chwidth

        self.ltype = gm.LTYPE
        self.cmode = gm.CMODE
        if self.cmode < 0 or self.cmode > 2:
            print("GALMOD warning: CMODE must be 0,1 or 2. Assuming 1.")
            self.cmode = 1
        self.cdens = float(gm.CDENS)
        if self.cdens < 0:
            print("GALMOD warning: CDENS must greater than 0. Assuming 1.0.")
            self.cdens = 1.
        self.iseed = gm.ISEED
        self.empty = gm.EMPTY
        self.densIsFlux = gm.DENSFLUX if densflux is None else densflux
        self.nvpar = gm.NV
        self.galmodtype = (galmodtype or getattr(p, "galmodtype", "MC")).upper()
        self.radsep = 0.75 * min(self.pixsize)
        if self.densIsFlux:
            self.cd2i = np.full(self.nsubs, np.float32(3600. / self.chwidth * 1000.), dtype=np.float64)
        else:
            self.cd2i = self._NHItoRAD()
        self.lfreq0 = np.ascontiguousarray([self.freq0] if lfreq0 is None else lfreq0, dtype=np.float64)
        self.relint = np.ascontiguousarray(self.relint, dtype=np.float64)
        self._vedges = self._velocity_edges()
        # the channel edges in the velocity of each line (ANALYTIC models)
        self._vedges_lines = np.ascontiguousarray([self._velocity_edges(f) for f in self.lfreq0])
        # narrowest channel (km/s -> m/s units as vedges), to size the line-profile windows
        self._chmin = float(np.min(np.abs(np.diff(self._vedges)))) if self.nsubs > 0 else self.chwidth

    def _NHItoRAD(self):
        jy, K = 1E-26, 1.3806488E-23
        isubs = np.arange(self.nsubs)
        w = self.crval3 + self.cdelt3 * (isubs + 1 - self.crpix3)
        if self.axtyp == 2:
            lab, fac = w, 2.588E04
        elif self.axtyp == 3:
            lab, fac = C_MS / w, 3.8475E-06
        else:
            f = self.freq0 * (1 - w / C_MS)
            if self.velsys == 2: f = self.freq0 * (C_MS / (C_MS + w))
            if self.velsys == 3: f = self.freq0 * np.sqrt((C_MS - w) / (C_MS + w))
            lab, fac = C_MS / f, 1.823E-05
        ddum = 2 * (K / jy) / (fac * lab * lab) / abs(self.cdelt3)
        return (ddum * (math.pi / (180.0 * 60.0)) ** 2).astype(np.float32).astype(np.float64)

    def _velocity_edges(self, freq0=None):
        """Velocity (m/s) of the channel boundaries (inverse of velgrid at k-0.5), for the line of reference
        frequency freq0 (default: the first line)."""
        g = np.arange(self.nsubs + 1) - 0.5
        w = (g + 1 - self.crpix3) * self.cdelt3 + self.crval3
        if self.axtyp == 4:
            return w
        f = C_MS / w if self.axtyp == 2 else w
        r = f / (self.freq0 if freq0 is None else freq0)
        if self.velsys == 1:
            v = C_MS * (1 - r)
        elif self.velsys == 2:
            v = C_MS * (1 / r - 1)
        else:
            v = C_MS * (1 - r * r) / (1 + r * r)
        return np.ascontiguousarray(v, dtype=np.float64)

    # ------------------------------------------------------------------
    def subrings(self, rings):
        """Port of Galmod::ringIO: interpolate the input rings onto sub-rings 0.75 pix wide.
        Returns (sr array (14,nsub), nv array)."""
        nr = rings.nr
        radsep = self.radsep
        rad = np.asarray(rings.radii, float) / 60.
        for i in range(1, nr):
            if rad[i - 1] + radsep >= rad[i]:
                if rad[i - 1] > rad[i]:
                    raise BBaroloError("GALMOD ERROR: Radii not in increasing order.")
                raise BBaroloError("GALMOD ERROR: Radius separation too small.")
        if np.any(np.asarray(rings.vdisp) < 0):
            raise ValueError("GALMOD ERROR: Negative velocity dispersion not allowed.")
        if np.any(np.asarray(rings.dens) < 0):
            raise ValueError("GALMOD ERROR: Negative column-density or flux are not allowed.")
        if np.any(np.asarray(rings.z0) < 0):
            raise ValueError("GALMOD ERROR: Negative scale height not allowed.")
        # values in internal units: radius(arcmin) handled separately
        V = np.vstack([np.asarray(rings.vrot) * 1000., np.asarray(rings.vrad) * 1000.,
                       np.asarray(rings.vvert) * 1000., np.asarray(rings.vdisp) * 1000.,
                       np.asarray(rings.dens, float), np.asarray(rings.z0) / 60.,
                       np.asarray(rings.dvdz) * 1000 * 60, np.asarray(rings.zcyl) / 60.,
                       np.radians(np.asarray(rings.inc, float)), np.radians(np.asarray(rings.phi, float)),
                       np.asarray(rings.xpos, float), np.asarray(rings.ypos, float),
                       np.asarray(rings.vsys) * 1000.]).astype(float)
        # sub-ring radii, emulating the float accumulation of the C++ code
        f32 = np.float32
        rads = []
        segs = []          # (i-1, dr) for interpolation; -1 => constant ring 0
        r = f32(rad[0] + radsep / 2.0) if (rad[0] != 0 and self.empty) else f32(radsep / 2.0)
        if not (rad[0] != 0 and self.empty):
            while r < f32(rad[0]):
                rads.append(r); segs.append((0, None))
                r = f32(float(r) + radsep)
        for i in range(1, nr):
            ri = f32(rad[i])
            while r < ri:
                rads.append(r); segs.append((i, float(r) - float(f32(rad[i - 1]))))
                r = f32(float(r) + radsep)
        nsub = len(rads)
        sr = np.empty((14, nsub))
        sr[0] = np.asarray(rads, float)
        if nsub:
            idx = np.array([s[0] for s in segs])
            dr = np.array([0. if s[1] is None else s[1] for s in segs])
            const = np.array([s[1] is None for s in segs])
            i1 = np.maximum(idx, 1)
            dur = rad[i1] - rad[i1 - 1] if nr > 1 else np.ones(nsub)
            slope = (V[:, i1] - V[:, i1 - 1]) / dur if nr > 1 else np.zeros((13, nsub))
            vals = V[:, i1 - 1] + slope * dr
            vals[:, const] = V[:, [0]]
            sr[1:] = vals
        edge = getattr(rings, "edge", None)
        if edge is not None and nsub:
            sr[5] *= edge_weights(sr[0] * 60., radsep * 60., float(edge), float(getattr(rings, "edgewidth", 0.)))
        nvtmp = self.nvpar if self.nvpar >= 1 else self.nsubs
        nv = np.where(sr[4] == 0, 1, nvtmp).astype(np.int64)
        return np.ascontiguousarray(sr), nv

    def build(self, rings, blo=(0, 0), bhi=None, out=None, seed=None, nthreads=1, galmodtype=None, layout="zyx",
              return_unit=False):
        """Compute a (non-convolved) model in the box [blo,bhi).
        Returns an array (nz, ny, nx) if layout='zyx' or (ny, nx, nz) if layout='yxz'.
        With return_unit=True, returns (model, unit), where unit is the mean flux of one
        Monte-Carlo sub-cloud (see subcloud_flux)."""
        if bhi is None:
            bhi = (self.dimx, self.dimy)
        info = [] if return_unit else None
        out = self._build(rings, blo, bhi, out, seed, nthreads, galmodtype, info)
        if layout == "zyx":
            out = np.ascontiguousarray(out.transpose(2, 0, 1))
        return (out, info[0]) if return_unit else out

    def subcloud_flux(self, sr, nv):
        """Flux (before the cd2i factor) deposited in one channel by one sub-cloud of the
        Monte-Carlo GALMOD, for the sub-rings sr: flux-weighted harmonic mean over the
        sub-rings (so that model/unit is the expected number of sub-clouds), times the
        mean intensity of the emission lines. Used to emulate the support of MC models."""
        R0, dens = sr[0], sr[5]
        area_pix = 2 * np.pi * R0 * self.radsep / self.pixarea
        with np.errstate(divide="ignore", invalid="ignore"):
            nc = np.floor(self.cdens * dens ** self.cmode * area_pix + 0.5)
            low = np.floor(nc / area_pix + 0.5) < 10.
            nc = np.where(low, np.floor(10 * area_pix + 0.5), nc)
            w = dens * area_pix
            ok = (w > 0) & (nc > 0)
            if not np.any(ok):
                return 0.
            fsc = w[ok] * self.pixarea / (nc[ok] * nv[ok])
            unit = w[ok].sum() / np.sum(w[ok] / fsc)
        ri = self.relint
        return float(unit * np.sum(ri * ri) / np.sum(ri))

    def _build(self, rings, blo, bhi, out, seed, nthreads, galmodtype, info=None):
        bsize = (bhi[0] - blo[0], bhi[1] - blo[1])
        sr, nv = self.subrings(rings)
        if info is not None:
            info.append(self.subcloud_flux(sr, nv) if sr.shape[1] else 0.)
        if out is None:
            out = np.zeros((bsize[1], bsize[0], self.nsubs), np.float64)
        seed = _seed(self.iseed if seed is None else seed)
        gtype = (galmodtype or self.galmodtype).upper()
        nsub = sr.shape[1]
        if nsub == 0:
            return out
        if gtype == "ANALYTIC":
            nzq = 1 if not np.any(sr[6] > 0) else ZSAMPLES_MAX           # (vertical samples: see _zgrid)
            args = (sr, self.radsep, self.cdens, self.pixarea, self.cdelt[0], self.cdelt[1], self.crota2,
                    blo[0], blo[1], bhi[0], bhi[1], self.ltype, self.sig_instr, self._vedges_lines,
                    self.relint, self.cd2i, nzq, self.axtyp, self.velsys, self.lfreq0,
                    self.crval3, self.cdelt3, self.crpix3, self._chmin, _PHI_TAB, self.zstep)
            if nthreads <= 1 or nsub < 2 * nthreads:
                _galmod_analytic(out, sr, 0, nsub, *args[1:])
            else:
                chunks = np.array_split(np.arange(nsub), nthreads)

                def work_a(k, buf):
                    c = chunks[k]
                    if len(c):
                        _galmod_analytic(buf, sr, int(c[0]), int(c[-1]) + 1, *args[1:])
                _sum_chunks(out, nthreads, work_a)
            return out
        args = (self.radsep, nv, self.cdens, self.cmode, self.pixarea, self.cdelt[0], self.cdelt[1],
                self.crota2, blo[0], blo[1], bhi[0], bhi[1], self.ltype, self.sig_instr, self.axtyp,
                self.velsys, self.lfreq0, self.crval3, self.cdelt3, self.crpix3, self.relint, self.cd2i)
        if nthreads <= 1 or nsub < 2 * nthreads:
            _galmod_mc(out, sr, 0, nsub, seed, *args)
        else:
            # Split the sub-rings among threads (each chunk with its own seed)
            chunks = np.array_split(np.arange(nsub), nthreads)

            def work(k, buf):
                c = chunks[k]
                if len(c):
                    _galmod_mc(buf, sr, int(c[0]), int(c[-1]) + 1, (seed + 7919 * k) & 0xFFFFFFFF, *args)
            _sum_chunks(out, nthreads, work)
        return out


# at most this many bytes of chunk buffers at a time (and at most _MAXBUF buffers)
_BUFMEM = 2 * 1024 ** 3
_MAXBUF = 16


def _sum_chunks(out, nchunks, work):
    """out += the sum of the chunks 0..nchunks-1, where work(k, buf) deposits chunk k into the zeroed
    buffer buf (an array like out). The chunks run in parallel in groups, each with its own buffer,
    and are added to out in the order of k: the same result as one buffer per chunk, but at most
    min(_MAXBUF, _BUFMEM / out.nbytes) buffers exist at a time (one per thread was hundreds of GB for
    large cubes with many threads)."""
    nb_ = int(max(1, min(nchunks, _MAXBUF, _BUFMEM // max(out.nbytes, 1))))
    bufs = [np.empty_like(out) for _ in range(nb_)]
    with ThreadPoolExecutor(nb_) as ex:
        for k0 in range(0, nchunks, nb_):
            ks = list(range(k0, min(k0 + nb_, nchunks)))

            def run(i):
                bufs[i][...] = 0.
                work(ks[i], bufs[i])
            list(ex.map(run, range(len(ks))))
            for i in range(len(ks)):
                out += bufs[i]
    return out


# --------------------------------------------------------------------------
# Sparse 2D convolution (models of single rings)
# --------------------------------------------------------------------------
@nb.njit(nogil=True, cache=True)
def _conv_sparse(model, kern, scale):
    """'Same' 2D convolution of each channel of a sparse (ny,nx,nz) model with kern.
    Equivalent to the FFT convolution (LINEAR_SAME) of the C++ code, followed by the
    same thresholding (values < 1E-12 set to zero). Only the non-zero spectral range of
    each pixel is convolved, and only the pixels/channels reached are thresholded."""
    ny, nx, nz = model.shape
    ky, kx = kern.shape
    hy, hx = ky // 2, kx // 2
    # non-zero kernel cells
    nk = 0
    for j in range(ky):
        for i in range(kx):
            if kern[j, i] != 0:
                nk += 1
    kj = np.empty(nk, np.int64)
    ki = np.empty(nk, np.int64)
    kv = np.empty(nk)
    nk = 0
    for j in range(ky):
        for i in range(kx):
            if kern[j, i] != 0:
                kj[nk] = j - hy
                ki[nk] = i - hx
                kv[nk] = kern[j, i]
                nk += 1
    out = np.zeros_like(model)
    zl = np.full((ny, nx), nz, np.int64)
    zh = np.full((ny, nx), -1, np.int64)
    for y in range(ny):
        for x in range(nx):
            m = model[y, x]
            zlo = -1
            zhi = -1
            for z in range(nz):
                if m[z] != 0:
                    if zlo < 0:
                        zlo = z
                    zhi = z
            if zlo < 0:
                continue
            for n in range(nk):
                yy = y + kj[n]
                xx = x + ki[n]
                if yy < 0 or yy >= ny or xx < 0 or xx >= nx:
                    continue
                if zlo < zl[yy, xx]:
                    zl[yy, xx] = zlo
                if zhi > zh[yy, xx]:
                    zh[yy, xx] = zhi
                k = kv[n]
                o = out[yy, xx]
                for z in range(zlo, zhi + 1):
                    o[z] += k * m[z]
    for y in range(ny):
        for x in range(nx):
            o = out[y, x]
            for z in range(zl[y, x], zh[y, x] + 1):
                v = o[z]
                if v < 1.E-12:
                    o[z] = 0.
                elif scale != 1.0:
                    o[z] = v * scale
    return out


# --------------------------------------------------------------------------
# Ring conventions (parameter RINGMODE)
# --------------------------------------------------------------------------
RINGMODES = ("CENTRED", "NODES", "SHIFTED")


def ringmode_name(mode):
    """Canonical name of a ring convention (CENTERED is accepted for CENTRED)."""
    m = str(mode).strip().upper()
    m = "CENTRED" if m == "CENTERED" else m
    if m not in RINGMODES:
        raise ValueError(f"Unknown ring convention {mode!r}: RINGMODE can be CENTRED, NODES or SHIFTED.")
    return m


def _half_widths(rings):
    """Half widths of the first and last ring for CENTRED: half the distance to the
    neighbouring radius (as the annuli fitted by 3DFIT); a single ring is RADSEP wide
    (or extends from R/2 to 3R/2 if RADSEP is not set)."""
    R = rings.radii
    if rings.nr == 1:
        w = rings.radsep / 2. if rings.radsep > 0 else R[0] / 2.
        return w, w
    return (R[1] - R[0]) / 2., (R[-1] - R[-2]) / 2.


def model_rings(rings, mode="CENTRED", subring=0.):
    """The rings that give the model of a set of rings with a ring convention.

    The model builder (GalmodSetup.build) interpolates the ring values linearly between
    the radii and fills the disk from the first radius (from the centre if EMPTY=false)
    to the last one: the radii are nodes. The conventions (RINGMODE) are:

    CENTRED  each ring is an annulus centred on its radius, reaching halfway to the
             neighbouring rings, as the rings fitted by 3DFIT: the disk extends beyond
             the first and last radius by half a ring, where the values of the first
             and last ring are kept constant. The model takes the ring values at the
             radii.
    NODES    the values are given at the radii (as GIPSY's GALMOD and TiRiFiC): the disk
             covers the first to the last radius.
    SHIFTED  BBarolo 1.x (final models of 3DFIT and GALMOD): the first and last radius are
             moved inwards/outwards by half the ring separation RADSEP. The disk covers
             the same area as CENTRED, but the values in the first and last ring are
             interpolated between the shifted radii.

    subring: width of the sub-rings of the model in arcmin (GalmodSetup.radsep). With
    CENTRED, an extension that is not wider than one sub-ring is not added (the model
    builder needs radii further apart than one sub-ring).
    Returns a new Rings object (or rings itself for NODES)."""
    mode = ringmode_name(mode)
    if mode == "NODES" and rings.nr == 1:
        raise BBaroloError("RINGMODE NODES needs two radii at least: the disk covers the first to the last radius "
                           "(one ring gave an empty model).")
    if mode == "SHIFTED" and rings.nr == 1 and not rings.radsep > 0:
        raise BBaroloError("RINGMODE SHIFTED with one ring needs RADSEP > 0: the width of the ring.")
    if mode == "NODES" or rings.nr == 0:
        return rings
    r = rings.copy()
    if mode == "SHIFTED":
        if r.nr == 1:
            r.addRing(r.radii[0] + r.radsep / 2., r.xpos[0], r.ypos[0], r.vsys[0], r.vrot[0], r.vdisp[0],
                      r.vrad[0], r.vvert[0], r.dvdz[0], r.zcyl[0], r.dens[0], r.z0[0], r.inc[0], r.phi[0])
        else:
            r.radii[-1] += r.radsep / 2.
        r.radii[0] = max(r.radii[0] - r.radsep / 2., 0.)
        return r
    R = r.radii
    w0, wn = _half_widths(r)
    rin, rout = max(R[0] - w0, 0.), R[-1] + wn
    idx, rad = list(range(r.nr)), list(R)
    if rin / 60. + subring < R[0] / 60.:            # inner half of the first ring
        idx.insert(0, 0)
        rad.insert(0, rin)
    if R[-1] / 60. + subring < rout / 60.:          # outer half of the last ring
        idx.append(r.nr - 1)
        rad.append(rout)
    out = r.subset(np.array(idx))
    out.radii = np.array(rad, dtype=float)
    return out


def edge_weights(R, dr, edge, width=0.):
    """Weights of the density of sub-rings at radii R (arcsec, each dr wide) for a disk with an
    outer edge at the radius edge (arcsec): with width = 0 (a sharp edge), the fraction of each
    sub-ring, R - dr/2 to R + dr/2, within edge (continuous in edge); with width > 0 (arcsec), a
    Gaussian edge, 1/2 erfc((R - edge) / (sqrt(2) width))."""
    from scipy.special import erfc
    R = np.asarray(R, float)
    if width > 0:
        return 0.5 * erfc((R - edge) / (math.sqrt(2.) * width))
    return np.clip((edge - (R - dr / 2.)) / dr, 0., 1.) if dr > 0 else (R <= edge).astype(float)


def outer_edge(rings, mode="CENTRED"):
    """Outer radius (arcsec) of the disk of the rings with a ring convention."""
    mode = ringmode_name(mode)
    if mode == "NODES":
        return float(rings.radii[-1])
    if mode == "SHIFTED":
        return float(rings.radii[-1] + rings.radsep / 2.)
    return float(rings.radii[-1] + _half_widths(rings)[1])


# --------------------------------------------------------------------------
# The model engine
# --------------------------------------------------------------------------
class DiskModel:
    """Models of a rotating disk on the grid of a datacube: the GALMOD engine used by the
    GALMOD, 3DFIT and SPACEPAR tasks, BayesianBBarolo and SimulatedGalaxyCube.

    DiskModel(cube, par=None, ringmode=None, densflux=None, galmodtype=None, smooth=None)
      cube:       a bbarolo Cube; only its header and parameters are used.
      par:        the GALMOD parameters (LTYPE, CMODE, CDENS, ISEED, NV, EMPTY, SM, ...);
                  default the 3DFIT/GALMOD parameters of the cube (cube.pars().gf).
      ringmode:   ring convention, CENTRED, NODES or SHIFTED (see model_rings); default
                  the RINGMODE parameter of the cube.
      densflux:   True if DENS is a flux (integrated flux / arcsec2), False if it is a
                  column density (1E20 atoms/cm2); default par.DENSFLUX.
      galmodtype: MC or ANALYTIC; default the GALMODTYPE parameter of the cube.
      smooth:     convolve the models with the beam of the cube; default par.SM.

    The quantities derived from the header (GalmodSetup) and the beam kernel are
    computed once. Then:
      array(rings)   the model as a float64 array (nz, ny, nx), in the whole cube or in a
                     box [blo, bhi) of it;
      cube(rings)    the model as a Cube; hdu(rings) as an astropy PrimaryHDU;
      addnoise(a, rms)  a model plus noise smoothed to the beam;
      box(rings)     the box of the cube containing the model;
      ring_maps(rings)  the total intensity map of each ring with unit density (the map of
                     the model is linear in the densities).
    rings is a bbarolo.rings.Rings object (e.g. from bbarolo.rings.readRings or
    Rings().setRings(radii=..., vrot=..., ...)).
    """

    def __init__(self, cube, par=None, ringmode=None, densflux=None, galmodtype=None, smooth=None):
        p = cube.pars()
        self.cin = cube
        self.par = p.gf if par is None else par
        self.setup = GalmodSetup(cube, gmpar=self.par, galmodtype=galmodtype,
                                 densflux=self.par.DENSFLUX if densflux is None else densflux)
        self.ringmode = ringmode_name(getattr(p, "ringmode", "CENTRED") if ringmode is None else ringmode)
        self.arcconv = arcsconv(cube.Head().Cunit(0))
        self.cfield = None              # beam kernel (2D)
        self.conv = None                # FFT convolver with the kernel
        self.scalefactor = 1.           # flux scaling of the convolution (column densities)
        self.smooth = bool(self.par.SM if smooth is None else smooth)
        if self.smooth and not self._set_kernel():
            sys.stderr.write(" WARNING: can not set an appropriate convolution field. "
                             "Turning off the convolution step.\n")
            self.smooth = False
        self.setup.zstep = self.zstep

    @property
    def zstep(self):
        """The spacing (pixels, once projected) of the vertical samples of the ANALYTIC models: ZSTEP_COARSE
        (0.75) if the model is convolved with a beam whose minor axis is at least ZSTEP_BEAM (3) pixels, which
        smooths out the aliasing of the samples on the pixel grid, otherwise ZSTEP_FINE (0.25)."""
        if self.conv is None:
            return ZSTEP_FINE
        h = self.cin.Head()
        pix = max(self.setup.pixsize) / 60.                      # (degrees)
        bmin = min(abs(h.Bmaj()), abs(h.Bmin()))
        return ZSTEP_COARSE if pix > 0 and bmin / pix >= ZSTEP_BEAM else ZSTEP_FINE

    def _set_kernel(self):
        c = self.cin
        h = c.Head()
        sm = Smooth3D()
        if not sm.setfromCube(c, (0, 0, 0), (h.Bmaj() * 3600., h.Bmin() * 3600., h.Bpa())):
            return False
        self.cfield = sm.Confie()
        self.conv = FFTConvolver(self.cfield)
        self.scalefactor = 1.0 if self.setup.densIsFlux else sm.Scalefac()
        return True

    def __repr__(self):
        h = self.cin.Head()
        return (f"<DiskModel on a {self.cin.DimX()}x{self.cin.DimY()}x{self.cin.DimZ()} grid ({h.Name()}): "
                f"{self.setup.galmodtype}, RINGMODE={self.ringmode}, "
                f"{'smoothed to the beam' if self.conv is not None else 'not smoothed'}, "
                f"DENS as {'flux' if self.setup.densIsFlux else 'column density'}>")

    # ------------------------------------------------------------------
    def box(self, rings, final=False):
        """Box [blo, bhi) of the cube (pixels) containing the model of the rings: within the
        last radius plus half the ring separation RADSEP, plus 3 scale heights, around the
        centre of the last ring. final=True: also within the outer edge of the disk with the
        ring convention, if larger."""
        h = self.cin.Head()
        ext = rings.radii[-1] + rings.radsep / 2.
        if final:
            ext = max(ext, outer_edge(rings, self.ringmode))
        ext = ext + 3 * rings.z0[-1]
        xdis = int(math.ceil(ext / (abs(h.Cdelt(0)) * self.arcconv)))
        ydis = int(math.ceil(ext / (abs(h.Cdelt(1)) * self.arcconv)))
        xpos = int(math.ceil(rings.xpos[-1]))
        ypos = int(math.ceil(rings.ypos[-1]))
        blo = [max(xpos - xdis, 0), max(ypos - ydis, 0)]
        bhi = [min(xpos + xdis + 1, self.cin.DimX()), min(ypos + ydis + 1, self.cin.DimY())]
        return blo, bhi

    def array(self, rings, blo=None, bhi=None, final=True, ringmode=None, modsoFar=None, seed=None,
              galmodtype=None, layout="zyx", support=None, threads=None):
        """The model of the rings (convolved with the beam if smoothing is on) in the box
        [blo, bhi) of the cube (default: the whole cube), as a float64 array (nz, ny, nx),
        or (ny, nx, nz) with layout='yxz'.

        final=True: the model of a set of rings, with the ring convention (ringmode,
        default self.ringmode) and THREADS threads. final=False: the rings are used as
        they are (nodes) with one thread, as the models of single rings in the fits;
        modsoFar (an array (ny, nx, nz) of the box) is added before the convolution.
        seed: seed of the Monte Carlo model (default ISEED). galmodtype: MC or ANALYTIC
        (default that of the setup). support: see Galfit._support_args."""
        if blo is None or bhi is None:
            blo, bhi = (0, 0), (self.cin.DimX(), self.cin.DimY())
        if final:
            rings = model_rings(rings, self.ringmode if ringmode is None else ringmode, self.setup.radsep)
        nth = threads if threads is not None else (self.cin.pars().threads if final else 1)
        want = support is not None and (galmodtype or self.setup.galmodtype).upper() == "ANALYTIC"
        mod = self.setup.build(rings, blo, bhi, seed=seed, nthreads=nth, galmodtype=galmodtype, layout="yxz",
                               return_unit=want)
        if want:
            mod, unit = mod
        if modsoFar is not None and not final:
            mod += modsoFar
        if want:
            support.extend([mod if self.conv is not None else mod.copy(), unit])
        if self.conv is not None:
            if final or modsoFar is not None:
                # Dense model: batched FFT convolution
                m = self.conv(mod.transpose(2, 0, 1), workers=nth)
                m[m < 1.E-12] = 0.
                if self.scalefactor != 1:
                    m *= self.scalefactor
                return np.ascontiguousarray(m) if layout == "zyx" else np.ascontiguousarray(m.transpose(1, 2, 0))
            mod = _conv_sparse(mod, self.cfield, float(self.scalefactor))
        if layout == "zyx":
            return np.ascontiguousarray(mod.transpose(2, 0, 1))
        return mod

    def _ring_weights(self, rings, ringmode):
        """Weights of the densities of the rings in the sub-rings of the model (the densities
        are interpolated linearly between the radii, with the ring convention): arrays
        (nsub, 2) of ring indices (-1: none) and weights. Cached by radii and convention."""
        key = (tuple(np.asarray(rings.radii, float)), float(rings.radsep), ringmode, getattr(rings, "edge", None),
               float(getattr(rings, "edgewidth", 0.)))
        cache = self.__dict__.setdefault("_wcache", {})
        if key in cache:
            return cache[key]
        st = self.setup
        cols = []
        for r in range(rings.nr):
            rr = rings.copy()
            rr.dens = np.zeros(rings.nr)
            rr.dens[r] = 1.
            cols.append(st.subrings(model_rings(rr, ringmode, st.radsep))[0][5])
        W = np.array(cols).T                                  # (nsub, nr)
        nsub = W.shape[0]
        widx = np.full((nsub, 2), -1, np.int64)
        wval = np.zeros((nsub, 2))
        for s in range(nsub):
            nz = np.flatnonzero(W[s] != 0)
            if len(nz) > 2:
                raise RuntimeError("ring_maps: a sub-ring depends on more than two rings.")
            widx[s, :len(nz)] = nz
            wval[s, :len(nz)] = W[s, nz]
        if len(cache) > 64:
            cache.clear()
        cache[key] = (widx, wval)
        return widx, wval

    def ring_maps(self, rings, blo=None, bhi=None, ringmode=None):
        """Total intensity maps (sums over the channels, convolved with the beam if smoothing
        is on) of the model of each ring with unit density, in the box [blo, bhi) of the cube:
        an array (nr, ny, nx). The model is linear in the densities (interpolated linearly
        between the radii), so the total intensity map of the model of the rings,
        array(rings).sum(axis=0), is sum_r rings.dens[r] * maps[r], with the same ring
        convention (ringmode, default self.ringmode). The clouds are placed as in the ANALYTIC
        model (the expectation of the Monte-Carlo one), and the lines are assumed to fall
        within the spectral range of the cube."""
        if blo is None or bhi is None:
            blo, bhi = (0, 0), (self.cin.DimX(), self.cin.DimY())
        mode = self.ringmode if ringmode is None else ringmode_name(ringmode)
        st = self.setup
        mr = model_rings(rings, mode, st.radsep)
        mr = mr.copy() if mr is rings else mr
        mr.dens = np.ones(mr.nr)
        sr, _ = st.subrings(mr)
        widx, wval = self._ring_weights(rings, mode)
        out = np.zeros((rings.nr, bhi[1] - blo[1], bhi[0] - blo[0]))
        if sr.shape[1]:
            nzq = 1 if not np.any(sr[6] > 0) else ZSAMPLES_MAX
            _ring_maps_analytic(out, sr, widx, wval, st.radsep, st.pixarea, st.cdelt[0], st.cdelt[1], st.crota2,
                                blo[0], blo[1], bhi[0], bhi[1], st.ltype, nzq, st.sig_instr, st._chmin, st.zstep)
        # sum of the channel-integrated line profiles: cd2i (almost constant across the
        # channels) times the intensities of the lines
        out *= float(np.mean(st.cd2i)) * float(np.sum(st.relint))
        if self.conv is not None:
            out = self.conv(out) * self.scalefactor
        return out

    def cube(self, rings, blo=None, bhi=None, final=True, **kwargs):
        """The model (see array) as a Cube with the header of the cube (the reference pixel
        moved for a box)."""
        from .cube import Cube
        c = self.cin
        if blo is None or bhi is None:
            blo, bhi = (0, 0), (c.DimX(), c.DimY())
        arr = self.array(rings, blo, bhi, final=final, **kwargs)
        h = c.Head().copy()
        h.setCrpix(0, c.Head().Crpix(0) - blo[0])
        h.setCrpix(1, c.Head().Crpix(1) - blo[1])
        return Cube.fromArray(arr.astype(np.float32), h, c.pars())

    def hdu(self, rings, blo=None, bhi=None, final=True, **kwargs):
        """The model (see array) as an astropy PrimaryHDU, with the header that BBarolo
        writes for models (OBJECT = name of the cube + 'mod')."""
        return model_hdu(self.cube(rings, blo, bhi, final=final, **kwargs), self.cin.Head().Name() + "mod")

    def addnoise(self, array, rms, rng=None, seed=None):
        """array + Gaussian noise with the given rms per voxel; the noise is smoothed to the
        beam (and rescaled to rms) if smoothing is on. rng: a numpy Generator (default one
        seeded with seed). Returns a float32 array."""
        a = np.asarray(array)
        if rms == 0:
            return a.astype(np.float32)
        rng = np.random.default_rng(seed) if rng is None else rng
        noise = rng.normal(0., rms, a.shape)
        if self.conv is not None:
            noise = self.conv(noise, workers=self.cin.pars().threads)
            noise *= rms / np.std(noise)
        return (a + noise).astype(np.float32)


def model_hdu(mod, name):
    """An astropy PrimaryHDU of a model Cube, with the header written by BBarolo."""
    from astropy.io import fits
    h = mod.Head()
    h.setMinMax(0., 0.)
    h.setName(name)
    return fits.PrimaryHDU(data=mod.array, header=h.toAstropy(3))
