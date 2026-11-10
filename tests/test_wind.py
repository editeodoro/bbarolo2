"""Tests of the wind models (bbarolo.windmodel), of the emission (bbarolo.bayesian.emission) and of BAYESIANWIND."""

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "performance"))

from bbarolo.utils import emptyfits                        # noqa: E402
from bbarolo.cube import Cube                              # noqa: E402
from bbarolo.params import Param                           # noqa: E402
from bbarolo.bbutils import BBaroloError                   # noqa: E402
from bbarolo.windmodel import WindModel, WindCells, WindPars, Boxes    # noqa: E402
from bbarolo.bayesian.emission import solve_nonneg, log_marginal, log_marginal_diag, log_profile  # noqa: E402


def _cube(nx=32, nz=30, pix=10., chan=15., beam=30., obj="WIND"):
    hdu = emptyfits([nx, nx, nz], [-pix / 3600., pix / 3600., chan], beam=None if beam is None else beam / 3600.,
                    bunit="JY/BEAM", obj=obj)
    p = Param()
    p.verbose = False
    c = Cube(par=p)
    c.readHDU(hdu)
    return c, hdu


GEO = dict(xpos=15.5, ypos=15.5, vsys=0., inc=60., pa=45., openang=60., conelen=140.)


def _uniform_cone_funcs(conelen, oa, vw):
    th = math.radians(oa / 2)

    def dens(x, y, z):
        r = np.sqrt(x * x + y * y + z * z)
        return ((r <= conelen) & (np.abs(z) >= r * math.cos(th))).astype(float)

    def comp(i):
        def f(x, y, z):
            r = np.sqrt(x * x + y * y + z * z)
            return vw * (x, y, z)[i] / np.where(r > 0, r, 1.)
        return f
    return dens, comp(0), comp(1), comp(2)


def test_windmodel_vs_galmod3d():
    """A uniform filled bicone with a radial wind: the engine of BAYESIANWIND gives the GalMod3D model (a Cartesian
    grid in the frame of the wind, a different engine) within 0.3% of the peak rms, the same flux and the same
    velocity field: the same geometry, projection and signs."""
    from bbarolo import GalMod3D
    c, hdu = _cube()
    cells = WindCells.resolved(150., 40., 30.)
    wm = WindModel(c, cells).quadrature(vmax=150, sigmin=10)
    p = WindPars(radii=[70.], vwind=150., vdisp=10., **GEO)
    m = wm.cube(p)
    dens, vx, vy, vz = _uniform_cone_funcs(GEO["conelen"], GEO["openang"], 150.)
    g = GalMod3D(hdu, coords="xyz", xpos=15.5, ypos=15.5, vsys=0., inc=60., pa=45., rmax=140., zmax=140.,
                 step=10 / 3., dens=dens, vx=vx, vy=vy, vz=vz, vdisp=10.).model().data.astype(float)
    assert abs(m.sum() / g.sum() - 1) < 3e-3
    assert np.sqrt(np.mean((m - g) ** 2)) < 3e-3 * g.max()
    assert np.corrcoef(m.ravel(), g.ravel())[0, 1] > 0.9999
    v = np.arange(m.shape[0])
    v1 = [np.where(a.sum(0) > 1e-2 * a.sum(0).max(), (a * v[:, None, None]).sum(0) / np.maximum(a.sum(0), 1e-30),
                   np.nan) for a in (m, g)]
    ok = np.isfinite(v1[0]) & np.isfinite(v1[1])
    assert np.nanmax(np.abs(v1[0][ok] - v1[1][ok])) < 0.3          # (channels)


def test_windmodel_galwind_conventions():
    """The cones of the engine have the conventions of GALWIND WTYPE 1 (the Monte-Carlo shells of C++): the same side
    of the sky recedes, the same velocity field."""
    from bbarolo import GalWind
    c, hdu = _cube()
    cells = WindCells([0., 140.], 1, 1, thmax=30.)
    wm = WindModel(c, cells).quadrature(vmax=150, sigmin=10)
    m = wm.cube(WindPars(radii=[70.], vwind=150., vdisp=10., **GEO))
    res = GalWind(hdu, xpos=15.5, ypos=15.5, vsys=0, inc=60, pa=45, openang=60, conelen=140, vwind=150, vdisp=10,
                  dens=10, nlayers=10, wtype=1, cdens=50, nv=10, iseed=3, verbose=False).run()
    g = np.asarray(res.model.data, float)
    v = np.arange(m.shape[0]) - (m.shape[0] / 2.)
    v1 = []
    for a in (m, g):
        tot = a.sum(0)
        v1.append(np.where(tot > 0.05 * tot.max(), (a * v[:, None, None]).sum(0) / np.maximum(tot, 1e-30), np.nan))
    ok = np.isfinite(v1[0]) & np.isfinite(v1[1])
    assert ok.sum() > 100
    assert np.corrcoef(v1[0][ok], v1[1][ok])[0, 1] > 0.95
    assert np.corrcoef(m.sum(0).ravel(), g.sum(0).ravel())[0, 1] > 0.9


def test_wind_boxes_normal_equations():
    """The normal equations from the voxel lists of the boxes are A'WA and A'Wd of the dense columns, the model is A x,
    and the columns integrate to the clipped volumes of their cells; the rotated elliptical beam (not separable) gives
    the model smoothed by the FFT of BBarolo."""
    c, _ = _cube(nx=24, nz=20)
    cells = WindCells.resolved(110., 40., 35.)
    wm = WindModel(c, cells).quadrature(vmax=150, sigmin=10)
    p = WindPars(radii=[50.], vwind=150., vdisp=10., xpos=11.5, ypos=11.5, inc=60, pa=45, openang=60, conelen=100.)
    pts = wm.points(p)
    B = wm.boxes(p, pts=pts)
    rng = np.random.default_rng(0)
    A = B.dense(wm.shape)
    x = rng.exponential(1., B.n)
    d = A @ x + rng.normal(0, 0.1, A.shape[0])
    W = rng.uniform(0.5, 2., wm.shape)
    G, b = B.normal(W * d.reshape(wm.shape), W)
    assert np.allclose(G, A.T @ (W.ravel()[:, None] * A), rtol=1e-12, atol=1e-12 * np.abs(G).max())
    assert np.allclose(b, A.T @ (W.ravel() * d), rtol=1e-12, atol=1e-12 * np.abs(b).max())
    assert np.allclose(B.model(x, wm.shape).ravel(), A @ x)
    inside = np.abs(A.sum(axis=0) / np.where(pts["vol"] > 0, pts["vol"], 1.) - 1)[pts["vol"] > 0]
    assert np.median(inside) < 1e-3
    # rotated elliptical beam
    hdu2 = emptyfits([24, 24, 20], [-10. / 3600., 10. / 3600., 15.], beam=[40. / 3600., 20. / 3600., 30.],
                     bunit="JY/BEAM", obj="ELL")
    pp = Param()
    pp.verbose = False
    c2 = Cube(par=pp)
    c2.readHDU(hdu2)
    w2 = WindModel(c2, WindCells.resolved(110., 40., 35.)).quadrature(vmax=150, sigmin=10)
    assert not w2.separable
    raw = WindModel(c2, WindCells.resolved(110., 40., 35.), smooth=False).quadrature(vmax=150, sigmin=10,
                                                                                     spacing=2.).cube(p)
    from bbarolo.galmod import DiskModel
    sm = DiskModel(c2, densflux=True, smooth=True).conv(raw)
    m2 = w2.cube(p)
    # (0.4%: the reference has the bilinear weights on top of the beam, 1/6 pixel^2 against a minor axis of 2 pixels;
    # the beam at the wrong angle, BPA 120 or -30 degrees, differs by 3%)
    assert np.sqrt(np.mean((m2 - sm) ** 2)) < 1e-2 * sm.max()
    hdu3 = emptyfits([24, 24, 20], [-10. / 3600., 10. / 3600., 15.], beam=[40. / 3600., 20. / 3600., 120.],
                     bunit="JY/BEAM", obj="ELL")
    c3 = Cube(par=pp)
    c3.readHDU(hdu3)
    m3 = WindModel(c3, WindCells.resolved(110., 40., 35.)).quadrature(vmax=150, sigmin=10).cube(p)
    assert np.sqrt(np.mean((m3 - sm) ** 2)) > 2e-2 * sm.max()


def test_wind_model_continuous():
    """The columns change continuously with the geometry: cells clipped by the cone, quadrature points moving with
    it (a small change of OPENANG, INC or CONELEN changes the normal equations by a small amount)."""
    c, _ = _cube(nx=24, nz=20)
    wm = WindModel(c, WindCells.resolved(110., 40., 30.)).quadrature(vmax=150, sigmin=10)
    W = np.ones(wm.shape)
    base = dict(radii=[50.], vwind=150., vdisp=10., xpos=11.5, ypos=11.5, inc=60, pa=45, openang=60, conelen=100.)
    d = wm.cube(WindPars(**base))

    def Gb(**kw):
        q = dict(base)
        q.update(kw)
        return wm.boxes(WindPars(**q)).normal(W * d, W)
    G0, b0 = Gb()
    for k, dv in (("openang", 0.1), ("inc", 0.1), ("conelen", 0.2)):
        G1, b1 = Gb(**{k: base[k] + dv})
        G2, b2 = Gb(**{k: base[k] + 2 * dv})
        d1, d2 = np.abs(b1 - b0).max(), np.abs(b2 - b0).max()
        assert d1 < 0.05 * np.abs(b0).max(), k
        assert 1.2 < d2 / max(d1, 1e-30) < 3.0, k           # (about linear: no jumps)


def test_emission_solver_and_evidence():
    """solve_nonneg is the NNLS solution; log_marginal is exact for one amplitude and close for two (the integrals of
    the Laplace approximation with the truncation of each amplitude: within 0.15); log_marginal_diag is exact."""
    from scipy.optimize import nnls
    from scipy import integrate
    rng = np.random.default_rng(3)
    for n in (5, 60):
        A = np.abs(rng.normal(size=(4 * n, n))) * (rng.random((4 * n, n)) < 0.3) + 0.01
        d = A @ (rng.exponential(1, n) * (rng.random(n) < 0.6)) + rng.normal(0, 0.3, 4 * n)
        x, F = solve_nonneg(A.T @ A, A.T @ d)
        assert np.allclose(x, nnls(A, d, maxiter=50 * n)[0], atol=1e-9)
    for trial in range(6):
        n = 1 + trial % 2
        M = rng.normal(size=(n + 2, n))
        G = M.T @ M * rng.uniform(0.5, 20)
        xt = rng.exponential(1, n) * (rng.random(n) < 0.6)
        b = G @ xt + rng.normal(size=n) @ np.linalg.cholesky(G).T
        mu = rng.uniform(0.3, 3, n)
        c0 = float(xt @ G @ xt + 1.)
        f = lambda *x: math.exp(-0.5 * c0 + b @ np.array(x) - 0.5 * np.array(x) @ G @ np.array(x)
                                - np.sum(np.array(x) / mu)) / np.prod(mu)
        top = np.maximum(np.linalg.solve(G, b), 0) + 12 / np.sqrt(np.diag(G))
        val = integrate.nquad(f, [[0, top[i]] for i in range(n)], opts={"epsrel": 1e-8})[0]
        r = log_marginal(G, b, c0, mu)
        assert abs(r["logz"] - math.log(val)) < (1e-6 if n == 1 else 0.15)      # (two: about 0.1 at most)
        if n == 1:
            assert abs(log_marginal_diag(np.diag(G), b, c0, mu)[0] - math.log(val)) < 1e-6
    lp, x = log_profile(G, b, c0)
    assert lp >= r["logz"] - 50 and np.all(x >= 0)


def test_wind_parsers_and_keywords():
    """The free parameters, priors and options of BAYESIANWIND; its keywords and the checks of the task."""
    from bbarolo.bayesian.wind import wind_freepar, wind_priors, wind_likeopts
    from bbarolo.keywords import lookup, keywords_for
    assert wind_freepar("VWIND_SINGLE vdisp INC PA OPENANG EMSCALE vrot_func") == \
        ["vwind_single", "vdisp", "inc", "pa", "openang", "emscale", "vrot_func"]
    assert wind_freepar("phi disp") == ["pa", "vdisp"]
    with pytest.raises(ValueError):
        wind_freepar("vrad")
    with pytest.raises(ValueError):
        wind_freepar("sigma_single")
    pr = wind_priors("INC=(uniform,40,30) VWIND_P1=(uniform,0,500)")
    assert pr["inc"] == dict(name="uniform", loc=40., scale=30.) and "vwind_p1" in pr
    assert wind_likeopts("emission=shells marginal=false datafrac=0.2") == \
        dict(emission="shells", marginal=False, datafrac=0.2)
    with pytest.raises(ValueError):
        wind_likeopts("emission=pixels")
    with pytest.raises(ValueError):
        wind_likeopts("likemask=none")
    names = {k.name for k in keywords_for("BAYESIANWIND")}
    assert {"OPENIN", "DISKRINGS", "FREE", "PRIORS", "LIKEOPTS", "VWIND", "OPENANG", "CONELEN", "XPOS"} <= names
    assert lookup("BAYESIANWIND").flag
    # the checks of the task: FREE required, LIKEOPTS checked
    from bbarolo.tasks import build_param
    import io
    import contextlib
    base = dict(FITSFILE="x.fits", BAYESIANWIND=True, XPOS=10, YPOS=10, VSYS=0, INC=60, PA=30, OPENANG=60,
                CONELEN=100, VWIND=200)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ok = build_param(dict(base)).checkPars()
    assert not ok and "FREE is not an optional parameter" in buf.getvalue()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ok = build_param(dict(base, FREE="VWIND_SINGLE INC", LIKEOPTS="emission=pixels")).checkPars()
    assert not ok and "LIKEOPTS" in buf.getvalue()


def test_htot_ntot_renamed():
    """HTOT and NTOT of BBarolo 1.8 are now CONELEN and NLAYERS: the old names are errors that name the new ones, in
    parameter files and in Python."""
    from bbarolo import GalWind, BayesianWind
    _, hdu = _cube()
    for old, new in (("HTOT", "CONELEN"), ("NTOT", "NLAYERS")):
        with pytest.raises(BBaroloError, match=f"{old}.*renamed {new}"):
            Param().readParamString(f"FITSFILE x.fits\nGALWIND true\n{old} 10\n")
        for task in (GalWind, BayesianWind):
            with pytest.raises(BBaroloError, match=new):
                task(hdu, **{old.lower(): 10})
    p = Param()
    p.readParamString("FITSFILE x.fits\nGALWIND true\nCONELEN 120\nNLAYERS 6\n")
    assert p.gw.CONELEN == 120. and p.gw.NLAYERS == 6


def test_galwind_wtype2():
    """GALWIND WTYPE 2: the cones of the engine (OPENIN, DENS per shell as an emissivity), noise-free; the flux is the
    emissivity times the volume of the cones (over the channel width), a hollow cone has less."""
    from bbarolo import GalWind
    _, hdu = _cube()
    kw = dict(xpos=15.5, ypos=15.5, vsys=0, inc=60, pa=45, openang=60, conelen=140, vwind=150, vdisp=10, dens=2.,
              nlayers=4, wtype=2, verbose=False)
    m = np.asarray(GalWind(hdu, **kw).run().model.data, float)
    vol = 2 * 2 * math.pi / 3 * 140. ** 3 * (1 - math.cos(math.radians(30)))
    assert abs(m.sum() * 15. / (2. * vol) - 1) < 0.01
    h = np.asarray(GalWind(hdu, **dict(kw, openin=30)).run().model.data, float)
    volh = 2 * 2 * math.pi / 3 * 140. ** 3 * (math.cos(math.radians(15)) - math.cos(math.radians(30)))
    assert abs(h.sum() * 15. / (2. * volh) - 1) < 0.01
    with pytest.raises(BBaroloError):
        GalWind(hdu, **dict(kw, openang="60 70 80 90")).run()


def test_bayesian_wind_quick(tmp_path):
    """BAYESIANWIND on a tiny mock wind (GalMod3D clumps): the marginal likelihood prefers the true geometry to wider or
    narrower cones and slower or faster winds, EMSCALE, EMCLUMP, EMRATIO and EMSLOPE are added to the free parameters,
    a short sampling runs (dynesty, few live points: only the pipeline, not the accuracy, which is that of
    performance/wind_recovery.py) and the outputs are written: the model, the maps and PV slices of data and model in
    the layout of 3DFIT (maps/, pvs/), and the plots (the scripts in plotscripts/ run)."""
    pytest.importorskip("dynesty")
    from wind_mocks import mock_wind
    from bbarolo import BayesianWind
    hdu, tr = mock_wind(kind="clumps", seed=2, snr=15, shape=(24, 24, 20), pix=12., chan=25., beam=36.,
                        truth=dict(xpos=11.5, ypos=11.5, conelen=110., radii=[55., 110.], vwind=200.), nclumps=40,
                        clump=20.)
    f = BayesianWind(hdu, xpos=11.5, ypos=11.5, vsys=0, inc=55, pa=30, openang=55, conelen=110, vwind=180, vdisp=20,
                     nlayers=2, outfolder=str(tmp_path) + "/")
    f.like_kwargs = dict(f.like_kwargs, cellsize=40.)
    names = f._with_hyper(["vwind_single", "openang"], verbose=False)
    assert names == ["vwind_single", "openang", "emscale", "emclump", "emratio", "emslope"]
    f._setup(names, verbose=False)
    ll = lambda vw, oa: max(f.log_likelihood([vw, oa, s, k, 1., 0.]) for s in np.exp(np.linspace(-3, 3, 7)) * f._s0
                            for k in (0.5, 1., 2.))
    best = ll(200., 60.)
    for vw, oa in ((200., 80.), (200., 40.), (140., 60.), (260., 60.)):
        assert ll(vw, oa) < best - 3, (vw, oa)
    f.run(freepar=["vwind_single", "openang"], method="dynesty", sampler_kwargs=dict(nlive=10, seed=1),
          run_kwargs=dict(maxiter=40), like_kwargs=dict(cellsize=40.), verbose=False)
    assert f.ndim == 6 and np.all(np.isfinite(f.params)) and f.model.shape == hdu.data.shape
    files = f.write(plots=1, verbose=False)
    assert all(os.path.isfile(x) for x in files)
    names = {os.path.relpath(x, str(tmp_path)) for x in files}
    pre = f._cube_obj.pars().outPrefix
    for n in ("_wind_shells.txt", "_mod_wind.fits", "_mask.fits", "_chanmaps_wind.pdf", "_maps_wind.pdf",
              "_pv_wind.pdf"):
        assert pre + n in names, n
    for n in ("0mom", "1mom", "2mom", "mod_0mom_wind", "mod_1mom_wind", "mod_2mom_wind"):
        assert f"maps/{pre}_{n}.fits" in names, n
    for n in ("pv_a", "pv_b", "mod_pv_a_wind", "mod_pv_b_wind", "mask_pv_a", "mask_pv_b"):
        assert f"pvs/{pre}_{n}.fits" in names, n


def test_emission_gauss_exact():
    """log_marginal_gauss (the default prior of BAYESIANWIND: Gaussian emissivities of mean m and scatter kappa m) is
    the exact integral (brute force for 1-2 correlated amplitudes, the Gaussian of the data for 6), and
    log_marginal_gauss_diag its diagonal version; the posterior mean solves the regularised normal equations."""
    from scipy import integrate
    from bbarolo.bayesian.emission import log_marginal_gauss, log_marginal_gauss_diag
    rng = np.random.default_rng(5)
    for n in (1, 2):
        M = rng.normal(size=(n + 2, n)) + 0.8
        G = M.T @ M * 3
        b = rng.normal(size=n) * 2 + 1
        m = rng.uniform(0.5, 2, n)
        k, c0 = 0.8, 5.
        f = lambda *x: math.exp(-0.5 * c0 + b @ np.array(x) - 0.5 * np.array(x) @ G @ np.array(x)
                                - 0.5 * np.sum(((np.array(x) - m) / (k * m)) ** 2)) \
            / np.prod(np.sqrt(2 * np.pi) * k * m)
        val = integrate.nquad(f, [[m[i] - 12 * k * m[i], m[i] + 12 * k * m[i]] for i in range(n)],
                              opts={"epsrel": 1e-9})[0]
        r = log_marginal_gauss(G, b, c0, m, k)
        assert abs(math.log(val) - r["logz"]) < 1e-5
        P = np.diag(1. / (k * m) ** 2)
        assert np.allclose(r["x"], np.linalg.solve(G + P, b + P @ m))
    # in the space of the data: the marginal is the Gaussian N(d; A m, s2 I + A S A') (s2 the noise variance)
    from scipy.stats import multivariate_normal
    A = np.abs(rng.normal(size=(30, 6))) + 0.2
    m = rng.uniform(0.5, 2., 6)
    k, s2 = 0.7, 0.5
    d = A @ (m * (1 + k * rng.normal(size=6))) + rng.normal(0., math.sqrt(s2), 30)
    r = log_marginal_gauss(A.T @ A / s2, A.T @ d / s2, d @ d / s2, m, k)
    C = s2 * np.eye(30) + A @ np.diag((k * m) ** 2) @ A.T
    assert abs(r["logz"] - 0.5 * 30 * math.log(2 * math.pi * s2) - multivariate_normal(A @ m, C).logpdf(d)) < 1e-8
    g, bb, mm = np.array([2., 3.]), np.array([1., -1.]), np.array([1., 2.])
    ld = log_marginal_gauss_diag(g, bb, 4., mm, 0.7)[0]
    assert abs(ld - log_marginal_gauss(np.diag(g), bb, 4., mm, 0.7)["logz"]) < 1e-12


def test_bayesian_wind_fixed_disk(tmp_path):
    """DISKRINGS: a mock wind on top of a rotating disk (GALMOD rings written to a rings file). With the disk the
    likelihood is higher at the true opening angle than with wider cones, and the densities of the rings are recovered
    within 30%; without the disk, wider cones absorb its emission."""
    from wind_mocks import mock_wind
    from bbarolo import BayesianWind, GalMod
    from astropy.io import fits
    from bbarolo.bayesian import common
    hdu, tr = mock_wind(kind="clumps", seed=4, snr=15, noise=False, shape=(28, 28, 24), pix=12., chan=25., beam=36.,
                        truth=dict(xpos=13.5, ypos=13.5, conelen=130., radii=[65., 130.], vwind=200.), nclumps=50,
                        clump=20.)
    radii = np.arange(15., 151., 30.)
    vrot = 150 * np.tanh(radii / 40.)
    dens = 3e3 * np.exp(-radii / 60.)
    rf = str(tmp_path / "disk_rings.txt")
    with open(rf, "w") as fh:
        fh.write("#  RAD(arcs)  RAD(arcs)  VROT(km/s)  DISP(km/s)  INC(deg)  P.A.(deg)  Z0(pc)  Z0(arcs)  SIG(E20)  "
                 "XPOS(pix)  YPOS(pix)  VSYS(km/s)  VRAD(km/s)\n")
        for r, v in zip(radii, vrot):
            fh.write(f" {r:8.2f} {r:8.2f} {v:8.2f} 10.0 {tr['inc']:6.1f} {tr['pa']:6.1f} 0 10 1.0 13.5 13.5 0.0 0.0\n")
    disk = GalMod(hdu, radii=" ".join(map(str, radii)), xpos=13.5, ypos=13.5, vsys=0,
                  vrot=" ".join(f"{v:.3f}" for v in vrot), vdisp=10, inc=tr["inc"], pa=tr["pa"], z0=10,
                  dens=" ".join(f"{d:.3f}" for d in dens),
                  galmodtype="ANALYTIC", norm="NONE", mask="NONE", verbose=False).run().model.data
    data = hdu.data + disk + np.random.default_rng(1).normal(0, 0.05 * hdu.data.max(), hdu.data.shape)
    mock = fits.PrimaryHDU(data.astype(np.float32), hdu.header)
    with common.blas_threads():
        res = {}
        for with_disk in (True, False):
            f = BayesianWind(mock, xpos=13.5, ypos=13.5, vsys=0, inc=tr["inc"], pa=tr["pa"], openang=60, conelen=130,
                             vwind=200, vdisp=20, nlayers=2, outfolder=str(tmp_path) + "/",
                             **({"diskrings": rf} if with_disk else {}))
            f.like_kwargs = dict(f.like_kwargs, cellsize=40., noisecorr=1.)
            f._setup(f._with_hyper(["openang"], verbose=False), verbose=False)
            ss = np.exp(np.linspace(-2, 1, 7)) * f._s0
            res[with_disk] = {oa: max(f.log_likelihood([oa, s, 1., 1., 0.]) for s in ss) for oa in (60., 80.)}
            if with_disk:
                s = max(ss, key=lambda s: f.log_likelihood([60., s, 1., 1., 0.]))
                x = f.log_likelihood([60., s, 1., 1., 0.], details=True)[1]["x"][-f._disk.n:]
                assert np.all(np.abs(x / dens - 1) < 0.3), x / dens
    assert res[True][60.] > res[True][80.]
    assert res[False][80.] > res[False][60.]


def test_bayesian_wind_one_cone(tmp_path):
    """LIKEOPTS cones: a mock wind with only its approaching cone (GalMod3D clumps), fitted with cones=approaching. The
    model has the cells of that cone only and no EMRATIO (an error if given); the likelihood prefers the true opening
    angle and velocity; the outputs and the plots (one cone: one slice across it) are written."""
    pytest.importorskip("dynesty")
    from wind_mocks import mock_wind
    from bbarolo import BayesianWind
    from bbarolo.bayesian.wind import wind_likeopts
    assert wind_likeopts("cones=approaching") == dict(cones="approaching")
    with pytest.raises(ValueError):
        wind_likeopts("cones=near")
    hdu, tr = mock_wind(kind="clumps", seed=3, snr=15, shape=(24, 24, 20), pix=12., chan=25., beam=36., cones=(-1,),
                        truth=dict(xpos=11.5, ypos=11.5, conelen=110., radii=[55., 110.], vwind=200.), nclumps=40,
                        clump=20.)
    f = BayesianWind(hdu, xpos=11.5, ypos=11.5, vsys=0, inc=55, pa=30, openang=55, conelen=110, vwind=180, vdisp=20,
                     nlayers=2, outfolder=str(tmp_path) + "/")
    f.like_kwargs = dict(f.like_kwargs, cellsize=40., cones="approaching")
    with pytest.raises(ValueError):
        f._with_hyper(["openang", "emratio"], verbose=False)
    names = f._with_hyper(["vwind_single", "openang"], verbose=False)
    assert names == ["vwind_single", "openang", "emscale", "emclump", "emslope"]
    f._setup(names, verbose=False)
    assert f._wm.cells.cones == (-1,) and set(f._wm.cells.cells[:, 0]) == {-1.}
    ll = lambda vw, oa: max(f.log_likelihood([vw, oa, s, k, 0.]) for s in np.exp(np.linspace(-3, 3, 7)) * f._s0
                            for k in (0.5, 1., 2.))
    best = ll(200., 60.)
    for vw, oa in ((200., 85.), (200., 35.), (130., 60.), (270., 60.)):
        assert ll(vw, oa) < best - 3, (vw, oa)
    f.run(freepar=["vwind_single", "openang"], method="dynesty", sampler_kwargs=dict(nlive=10, seed=1),
          run_kwargs=dict(maxiter=40), like_kwargs=dict(cellsize=40., cones="approaching"), verbose=False)
    assert f.ndim == 5
    files = f.write(plots=1, verbose=False)
    pre = f._cube_obj.pars().outPrefix
    for n in ("_pv_wind.pdf", "_maps_wind.pdf", "_chanmaps_wind.pdf", "_wind_shells.txt"):
        assert os.path.isfile(str(tmp_path) + "/" + pre + n), n


def test_log_detection():
    """log_detection (the likelihood of emission=voxel): its table is the integral of Phi(F - t) over a log-normal F of
    mean x and fractional scatter kappa (brute force), the probit Phi(x - t) for kappa -> 0 and the false-positive rate
    Phi(-t) for x -> 0; the sum over detected and undetected voxels; flux predicted in undetected voxels lowers it."""
    from scipy import integrate
    from scipy.special import ndtr
    from bbarolo.bayesian.emission import log_detection
    t = 3.
    one = lambda x, k, d: log_detection(np.array([x]), np.array([d]), k, t)
    for x, k in ((0.3, 0.2), (2.5, 0.5), (4., 1.3), (10., 4.)):
        s2 = math.log1p(k * k)
        p = integrate.quad(lambda z: math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)
                           * ndtr(math.exp(math.log(x) - 0.5 * s2 + math.sqrt(s2) * z) - t), -12, 12, epsabs=1e-12)[0]
        assert abs(math.exp(one(x, k, True)) / p - 1) < 2e-3, (x, k)
        assert abs(math.exp(one(x, k, False)) / (1 - p) - 1) < 2e-3, (x, k)
    for x in (0.5, 3., 6.):
        assert abs(math.exp(one(x, 1e-4, True)) - ndtr(x - t)) < 1e-4
    assert abs(one(0., 1., True) - math.log(ndtr(-t))) < 1e-5
    x = np.array([0., 1., 4., 7.])
    det = np.array([True, False, True, False])
    tot = sum(one(v, 0.7, d) for v, d in zip(x, det))
    assert abs(log_detection(x, det, 0.7, t) - tot) < 1e-9
    assert log_detection(x + np.array([0, 1, 0, 1.]), det, 0.7, t) < log_detection(x, det, 0.7, t)


def test_bayesian_wind_voxel(tmp_path):
    """LIKEOPTS emission=voxel (MoKA-3D: the flux of the model is the data wherever the cone reaches) with the
    detection likelihood: EMSCALE, EMCLUMP, EMRATIO and EMSLOPE are added to the free parameters, SIGMA and DISKRINGS
    are errors; the likelihood prefers the true opening angle and velocity to wider or faster cones (which the old
    voxel weighting, without a penalty for the voxels it predicts and the data find empty, preferred) and to narrower or
    slower ones; a short sampling runs and the outputs are written, the model being the data where the cone reaches."""
    pytest.importorskip("dynesty")
    from wind_mocks import mock_wind
    from bbarolo import BayesianWind
    from bbarolo.bayesian.wind import wind_likeopts
    assert wind_likeopts("emission=voxel snrcut=2.5") == dict(emission="voxel", snrcut=2.5)
    with pytest.raises(ValueError):
        wind_likeopts("snrcut=0")
    hdu, tr = mock_wind(kind="clumps", seed=2, snr=15, shape=(24, 24, 20), pix=12., chan=25., beam=36.,
                        truth=dict(xpos=11.5, ypos=11.5, conelen=110., radii=[55., 110.], vwind=200.), nclumps=40,
                        clump=20.)
    kw = dict(xpos=11.5, ypos=11.5, vsys=0, inc=55, pa=30, openang=55, conelen=110, vwind=180, vdisp=20, nlayers=2,
              outfolder=str(tmp_path) + "/")
    f = BayesianWind(hdu, **kw)
    f.like_kwargs = dict(f.like_kwargs, emission="voxel")
    with pytest.raises(ValueError):
        f._with_hyper(["openang", "sigma"], verbose=False)
    names = f._with_hyper(["vwind_single", "openang"], verbose=False)
    assert names == ["vwind_single", "openang", "emscale", "emclump", "emratio", "emslope"]
    f._setup(names, verbose=False)
    ll = lambda vw, oa: max(f.log_likelihood([vw, oa, s, k, 1., 0.]) for s in np.exp(np.linspace(-3, 3, 7)) * f._s0
                            for k in (0.1, 0.5, 2.))
    best = ll(200., 60.)
    for vw, oa in ((200., 80.), (260., 60.)):                     # wider, faster: the voxels predicted and empty
        assert ll(vw, oa) < best - 5, (vw, oa)
    for vw, oa in ((200., 40.), (140., 60.)):                     # narrower, slower: the detections left out
        assert ll(vw, oa) < best - 2, (vw, oa)
    g = BayesianWind(hdu, diskrings=str(tmp_path / "rings.txt"), **kw)
    g.like_kwargs = dict(g.like_kwargs, emission="voxel")
    with pytest.raises(BBaroloError):
        g._setup(names, verbose=False)
    f.run(freepar=["vwind_single", "openang"], method="dynesty", sampler_kwargs=dict(nlive=10, seed=1),
          run_kwargs=dict(maxiter=40), like_kwargs=dict(emission="voxel"), verbose=False)
    assert f.ndim == 6 and np.all(np.isfinite(f.params))
    cov = f.best_details["covered"]
    assert np.array_equal(f.model, np.where(cov, f._dsafe, 0.))
    f.write(plots=0, verbose=False)
    pre = f._cube_obj.pars().outPrefix
    with open(str(tmp_path) + "/" + pre + "_wind_params.txt") as fh:
        assert "emission voxel, detection likelihood" in fh.readline()
