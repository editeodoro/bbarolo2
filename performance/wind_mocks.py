"""
Mock datacubes of bi-conical winds with a patchy emission, for the tests of BAYESIANWIND (performance/
wind_recovery.py, tests/test_wind.py).

The mocks are built with GalMod3D (bbarolo.galmod3d: a Cartesian grid in the frame of the wind, the fields
interpolated, the points deposited with their lines), not with the engine of the fit (bbarolo.windmodel), so that
the fits are not tested on their own model. The geometry is that of GALWIND and BAYESIANWIND: the cones along the
rotation axis z of a disk of inclination INC and position angle PA (GalMod3D with the same INC and PA), a radial
velocity v(r), a rotation around the axis and a dispersion.

The emission inside the cones (the half opening angles OPENIN/2 to OPENANG/2, out to CONELEN):
  'smooth'     uniform, times (r/conelen)^-slope;
  'clumps'     nclumps Gaussian clumps (dispersion clump arcsec, +-50%) at random positions in the volume of the
               cones, with log-normal amplitudes (sigma 0.5) times (r/conelen)^-slope (as the 100-clump mocks of
               MoKA-3D, Marconcini et al. 2023);
  'lognormal'  a log-normal random field exp(s G - s^2/2), G a Gaussian field of correlation length corr (arcsec),
               times (r/conelen)^-slope;
optionally only one cone (cones=(1,) the receding one, (-1,) the approaching one). Noise: Gaussian, smoothed to the
beam (the noise of DiskModel.addnoise), of rms peak/snr.

    >>> from wind_mocks import mock_wind
    >>> hdu, truth = mock_wind(kind="clumps", seed=3, snr=10)      # an astropy HDU and the true parameters
"""

import math
import numpy as np
from astropy.io import fits

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from bbarolo.utils import emptyfits                  # noqa: E402

DEFAULT_TRUTH = dict(xpos=23.5, ypos=23.5, vsys=0., inc=60., pa=30., openang=60., openin=0., conelen=200.,
                     radii=[50., 100., 150., 200.], vwind=[200., 200., 200., 200.], vdisp=20., vrot=0.)


def template(shape=(48, 48, 40), pix=10., chan=15., beam=30., bunit="JY/BEAM", obj="WINDMOCK"):
    """An empty cube (nx, ny, nz) with pixels of pix arcsec, channels of chan km/s and a circular beam (arcsec)."""
    return emptyfits(list(shape), [-pix / 3600., pix / 3600., chan], beam=beam / 3600., bunit=bunit, obj=obj)


def _frame_func(truth):
    """The fields of the wind in the frame of GalMod3D (x, y, z: arcsec; z the axis of the cones)."""
    R = np.asarray(truth["radii"], float)
    vw = np.broadcast_to(np.asarray(truth["vwind"], float), R.shape)
    vr = np.broadcast_to(np.asarray(truth.get("vrot", 0.), float), R.shape)
    vd = np.broadcast_to(np.asarray(truth.get("vdisp", 20.), float), R.shape)

    def comp(i):
        def f(x, y, z):
            r = np.sqrt(x * x + y * y + z * z)
            rr = np.where(r > 0, r, 1.)
            Rc = np.hypot(x, y)
            Rr = np.where(Rc > 0, Rc, 1.)
            v = np.interp(r, R, vw)
            rot = np.interp(r, R, vr)
            if i == 0:
                return v * x / rr - rot * y / Rr
            if i == 1:
                return v * y / rr + rot * x / Rr
            return v * z / rr
        return f
    disp = lambda x, y, z: np.interp(np.sqrt(x * x + y * y + z * z), R, vd)
    return comp(0), comp(1), comp(2), disp


def _in_cones(x, y, z, truth, cones):
    r = np.sqrt(x * x + y * y + z * z)
    with np.errstate(invalid="ignore", divide="ignore"):
        ct = np.where(r > 0, z / np.where(r > 0, r, 1.), 1.)
    t1, t0 = math.radians(truth["openang"] / 2.), math.radians(truth.get("openin", 0.) / 2.)
    ok = np.zeros(r.shape, bool)
    for c in cones:
        cc = c * ct
        ok |= (cc >= math.cos(t1)) & (cc <= math.cos(t0) + 1e-12)
    return ok & (r <= truth["conelen"])


def random_cone_points(n, truth, cones, rng):
    """n points uniform in the volume of the cones (x, y, z arcsec), and the cone of each."""
    t1, t0 = math.radians(truth["openang"] / 2.), math.radians(truth.get("openin", 0.) / 2.)
    r = truth["conelen"] * rng.random(n) ** (1. / 3.)
    ct = rng.uniform(math.cos(t1), math.cos(t0), n)
    st = np.sqrt(1 - ct * ct)
    ph = rng.uniform(0, 2 * math.pi, n)
    c = rng.choice(np.asarray(cones), n)
    return r * st * np.cos(ph), r * st * np.sin(ph), c * r * ct, c


def mock_wind(kind="clumps", truth=None, seed=0, snr=10., cones=(1, -1), nclumps=100, clump=15., sigma_ln=1.0,
              corr=20., slope=0., shape=(48, 48, 40), pix=10., chan=15., beam=30., step=None, noise=True):
    """A mock wind (see the module docstring): returns (hdu, truth) with truth the parameters (and 'noise', 'peak',
    'kind', 'seed')."""
    from bbarolo import GalMod3D
    tr = dict(DEFAULT_TRUTH)
    tr.update(truth or {})
    rng = np.random.default_rng(seed)
    hdu = template(shape, pix, chan, beam)
    conelen = tr["conelen"]
    radial = (lambda r: (np.maximum(r, 0.05 * conelen) / conelen) ** (-slope)) if slope else (lambda r: 1.)
    if kind == "smooth":
        dens = lambda x, y, z: _in_cones(x, y, z, tr, cones) * radial(np.sqrt(x * x + y * y + z * z))
    elif kind == "clumps":
        cx, cy, cz, _ = random_cone_points(nclumps, tr, cones, rng)
        cs = clump * rng.uniform(0.5, 1.5, nclumps)
        ca = np.exp(rng.normal(0., 0.5, nclumps)) * radial(np.sqrt(cx * cx + cy * cy + cz * cz))

        def dens(x, y, z):
            out = np.zeros(np.shape(x))
            for k in range(nclumps):
                d2 = (x - cx[k]) ** 2 + (y - cy[k]) ** 2 + (z - cz[k]) ** 2
                near = d2 < (4 * cs[k]) ** 2
                out[near] += ca[k] * np.exp(-0.5 * d2[near] / cs[k] ** 2)
            return out * _in_cones(x, y, z, tr, cones)
    elif kind == "lognormal":
        n = int(math.ceil(2 * conelen / (corr / 3.))) + 1
        ax = np.linspace(-conelen, conelen, n)
        g = rng.normal(size=(n, n, n))
        k = np.fft.fftfreq(n, d=ax[1] - ax[0]) * 2 * math.pi
        kk = k[:, None, None] ** 2 + k[None, :, None] ** 2 + k[None, None, :] ** 2
        g = np.real(np.fft.ifftn(np.fft.fftn(g) * np.exp(-0.25 * kk * corr ** 2)))
        g /= g.std()
        field = np.exp(sigma_ln * g - 0.5 * sigma_ln ** 2)
        from scipy.interpolate import RegularGridInterpolator
        itp = RegularGridInterpolator((ax, ax, ax), field, bounds_error=False, fill_value=0.)

        def dens(x, y, z):
            v = itp(np.column_stack([np.ravel(x), np.ravel(y), np.ravel(z)])).reshape(np.shape(x))
            return v * _in_cones(x, y, z, tr, cones) * radial(np.sqrt(x * x + y * y + z * z))
    else:
        raise ValueError(f"kind: smooth, clumps or lognormal, not {kind!r}")
    vx, vy, vz, disp = _frame_func(tr)
    g = GalMod3D(hdu, coords="xyz", xpos=tr["xpos"], ypos=tr["ypos"], vsys=tr["vsys"], inc=tr["inc"], pa=tr["pa"],
                 rmax=conelen, zmax=conelen, step=step or pix / 3., dens=dens, vx=vx, vy=vy, vz=vz, vdisp=disp)
    model = g.model().data.astype(np.float64)
    peak = float(model.max())
    rms = peak / snr
    if noise:
        data = g.diskmodel.addnoise(model, rms, rng=rng)
    else:
        data = model
    out = fits.PrimaryHDU(np.asarray(data, np.float32), hdu.header)
    tr.update(noise=rms, peak=peak, kind=kind, seed=seed, cones=tuple(cones), snr=snr)
    return out, tr
