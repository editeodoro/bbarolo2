"""
Tests for the pure-Python BBarolo (package bbarolo).

    cd pyBBarolo_pure && python -m pytest -q tests

Tests that need the example datacubes (ngc2403.fits) look for them in
$BBAROLO_EXAMPLES or in ../examples (i.e. the BBarolo distribution folder) and are
skipped if not found. Reference numbers come from the C++ BBarolo 1.8.1dev.
"""

import os
import sys
import subprocess
import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from bbarolo.params import Param                    # noqa: E402
from bbarolo.cube import Cube                       # noqa: E402
from bbarolo.rings import Rings                     # noqa: E402
from bbarolo.galmod import GalmodSetup              # noqa: E402
from bbarolo.smooth import Smooth3D                 # noqa: E402
from bbarolo.utils import emptyfits                 # noqa: E402

EXAMPLES = os.environ.get("BBAROLO_EXAMPLES", os.path.join(os.path.dirname(ROOT), "examples"))
N2403 = os.path.join(EXAMPLES, "ngc2403.fits")
needs_data = pytest.mark.skipif(not os.path.isfile(N2403), reason="example cube ngc2403.fits not found")


@pytest.fixture
def synthetic_cube(tmp_path):
    hdu = emptyfits([64, 64, 40], [-10. / 3600., 10. / 3600., 10.], beam=30. / 3600., bunit="JY/BEAM", obj="SYNTH")
    fname = str(tmp_path / "empty.fits")
    hdu.writeto(fname)
    p = Param()
    p.verbose = False
    return Cube(fname, par=p)


def _rings(nr=8):
    return Rings().setRings(radii=np.arange(nr) * 20. + 10., xpos=31.5, ypos=31.5, vsys=0., vrot=150.,
                            vdisp=10., z0=0., dens=1., inc=60., phi=45.)


def test_param_parsing():
    p = Param()
    p.readParamString("FITSFILE  a.fits\n3DFIT true\nNRADII 10\nFREE VROT VDISP\n// comment\nTHREADS 3\n")
    assert p.gf.flagGALFIT and p.gf.NRADII == 10 and p.gf.FREE == "VROT VDISP" and p.threads == 3
    p.readParamCL("NRADII=12")
    assert p.gf.NRADII == 12
    assert p.checkPars(quiet=True)


def test_rng():
    import numba as nb
    from bbarolo._rng import rng_init, rng_normal, rng_uniform

    @nb.njit
    def draw(n):
        s = rng_init(12345)
        a = np.empty(n)
        b = np.empty(n)
        for i in range(n):
            a[i] = rng_normal(s)
            b[i] = rng_uniform(s)
        return a, b
    a, b = draw(400000)
    assert abs(a.mean()) < 0.01 and abs(a.std() - 1) < 0.01
    assert abs(b.mean() - 0.5) < 0.005 and b.min() >= 0 and b.max() < 1


def test_galmod_mc_vs_analytic(synthetic_cube):
    setup = GalmodSetup(synthetic_cube, densflux=True)
    r = _rings()
    mc = setup.build(r, galmodtype="MC")
    an = setup.build(r, galmodtype="ANALYTIC")
    assert mc.shape == an.shape == synthetic_cube.array.shape
    assert abs(mc.sum() / an.sum() - 1) < 0.02
    # velocity centroid along the major axis is symmetric around vsys
    spec_mc, spec_an = mc.sum(axis=(1, 2)), an.sum(axis=(1, 2))
    assert np.corrcoef(spec_mc, spec_an)[0, 1] > 0.99


def test_ring_maps(synthetic_cube):
    """The total intensity map of a model is sum_r dens_r * ring_maps[r] (ANALYTIC), with the
    ring conventions, thick disks and warps, in a box of the cube."""
    from bbarolo.galmod import DiskModel
    for mode, z0 in (("CENTRED", 10.), ("NODES", 0.), ("SHIFTED", 20.)):
        m = DiskModel(synthetic_cube, ringmode=mode, galmodtype="ANALYTIC")
        r = _rings()
        r.z0[:] = z0
        r.dens = np.linspace(3., 0.5, r.nr)
        r.inc = np.linspace(50., 70., r.nr)
        blo, bhi = (5, 8), (60, 58)
        K = m.ring_maps(r, blo, bhi)
        mod = m.array(r, blo, bhi).sum(axis=0)
        assert K.shape == (r.nr, 50, 55) and mod.max() > 0
        assert np.abs(np.tensordot(r.dens, K, axes=1) - mod).max() < 1e-6 * mod.max()


def test_bayesian_deconvolved_profile(tmp_path):
    """The density profile when DENS is not fitted: the densities whose beam-convolved model
    reproduces the total intensity map. On a model cube it recovers the true profile; the
    average of the observed map in each ring (Ellprof, already smoothed by the beam; the
    profile of the original module, now an output) does not."""
    from astropy.io import fits
    from bbarolo.galmod import DiskModel
    from bbarolo.bayesian import BayesianBBarolo
    hdu = emptyfits([64, 64, 40], [-10. / 3600., 10. / 3600., 10.], beam=30. / 3600., bunit="JY/BEAM", obj="MOCK")
    hdu.writeto(str(tmp_path / "empty.fits"))
    p = Param()
    p.verbose = False
    c = Cube(str(tmp_path / "empty.fits"), par=p)
    radii = np.arange(8) * 20. + 10.                    # rings narrower than the beam (30")
    dens = 2. * np.exp(-radii / 50.)
    kw = dict(radii=radii, xpos=31.5, ypos=31.5, vsys=0., vrot=150., vdisp=10., z0=0., inc=60.)
    m = DiskModel(c, galmodtype="ANALYTIC").array(Rings().setRings(**kw, phi=45., dens=dens))
    noise = np.random.default_rng(2).normal(0., m.max() / 300., m.shape)
    fn = str(tmp_path / "mock.fits")
    fits.PrimaryHDU((m + noise).astype(np.float32), header=hdu.header).writeto(fn)
    f = BayesianBBarolo(fn, **kw, pa=45., mask="SEARCH", galmodtype="ANALYTIC", outfolder=str(tmp_path) + "/")
    f.like_kwargs['likemask'] = 'none'
    # the observed profile (outputs only): too faint in the inner rings, and a model with it
    # does not fit the data
    f._setup(['vrot_single'])
    ell = f._observed_profile(f._inri)
    obs = np.abs([ell.getSurfDensFaceOn(i) for i in range(8)]) / dens
    assert obs[0] < 0.8
    dec = np.array(f._inri.r['dens'])
    f._inri.modify_parameter("dens", obs * dens, makeobj=True)
    assert f.chi2_reduced(np.array([150.])) > 3
    f._inri.modify_parameter("dens", dec, makeobj=True)
    for smooth in (0., 0.1):
        f.like_kwargs['densprof_smooth'] = smooth
        f._setup(['vrot_single'])
        assert np.all(np.abs(f._inri.r['dens'] / dens - 1) < 0.08)
        assert f.chi2_reduced(np.array([150.])) < 1.05
    # saved and loaded
    f._set_sampling_stats(np.full((3, 1), 150.), np.ones(3))
    f.save_results(str(tmp_path / "res.npy"))
    g = BayesianBBarolo(fn, outfolder=str(tmp_path) + "/")
    g.load_results(str(tmp_path / "res.npy"), sampler='nautilus')
    assert g.like_kwargs['densprof_smooth'] == 0.1 and np.allclose(g._inri.r['dens'], f._inri.r['dens'])
    # outputs: the profile of the model in rings_final1.txt (2 digits) and, at full precision
    # with the observed one, in densprof_deconvolved.txt; densprof.txt is the observed profile
    assert np.all(np.abs(g.outri.r['dens'] / dens - 1) < 0.08)
    # errors of the profile (load_results and run): geometry fixed here, so only the noise
    dp = g.dens_profile
    assert dp is not None and dp['nsamples'] == 0 and dp['nmc'] == 200
    assert np.all(dp['err_geom_lo'] == 0) and np.allclose(dp['err_lo'], dp['err_noise_lo'])
    assert np.all((dp['err_lo'] > 0) & (dp['err_hi'] > 0) & (dp['err_hi'] < 0.1 * dp['dens']))
    lo, hi = g.ring_errors('DENS')
    assert np.array_equal(lo, dp['err_lo']) and np.array_equal(hi, dp['err_hi'])
    # ring_errors: from the posterior for the sampled parameters (vrot_single here), zeros if fixed
    lo, hi = g.ring_errors('vrot')
    q = g.quantiles[:, 0]
    assert lo.shape == (8,) and np.allclose(lo, q[1] - q[0]) and np.allclose(hi, q[2] - q[1])
    assert not np.any(g.ring_errors('PA')[0]) and set(g.ring_errors()) == set(g._RING_NAMES)
    with pytest.raises(ValueError):
        g.ring_errors('radii')
    with pytest.raises(ValueError):
        BayesianBBarolo(fn, **kw, pa=45., outfolder=str(tmp_path) + "/").ring_errors('vrot')
    g.write_bestmodel(plots=False, verbose=False)
    ring_dens = np.loadtxt(str(tmp_path / "rings_final1.txt"), usecols=8)
    assert np.allclose(ring_dens, g.outri.r['dens'], rtol=0.06)
    head = open(str(tmp_path / "rings_final1.txt")).readline().split()
    assert head[13:] == ["E_VROT1", "E_VROT2", "E_DENS1", "E_DENS2"]
    e_dens = np.loadtxt(str(tmp_path / "rings_final1.txt"), usecols=(15, 16))
    assert np.allclose(e_dens, np.c_[-dp['err_lo'], dp['err_hi']], rtol=1e-2, atol=1e-3)
    t = np.loadtxt(str(tmp_path / "densprof_deconvolved.txt"))
    assert np.allclose(t[:, 0], radii) and np.allclose(t[:, 1], g.outri.r['dens'], rtol=1e-5)
    assert np.allclose(t[:, 2], dp['err_lo'], rtol=1e-5) and np.allclose(t[:, 3], dp['err_hi'], rtol=1e-5)
    assert np.allclose(t[:, 4] / dens, obs, rtol=1e-3) and os.path.isfile(str(tmp_path / "densprof.txt"))
    # too few pixels to compute the profile: an error
    g._region[:] = False
    with pytest.raises(ValueError):
        g._set_density(g._inri)


def test_pv_engine():
    """pvslice: path geometry, exact bilinear sampling of a linear field, slit width,
    NaN handling, curved paths through the ring points, segments."""
    from bbarolo.pvslice import pv_path, pv_sample
    nz, ny, nx = 3, 40, 50
    z, yy, xx = np.mgrid[0:nz, 0:ny, 0:nx].astype(float)
    cube = 2 * xx + 3 * yy + 10 * z
    x0, y0, pa = 20.3, 15.7, 30.
    s, x, y = pv_path(x0, y0, pa, shape=(ny, nx), step=0.5)
    i0 = int(np.argmin(np.abs(s)))
    assert s[i0] == 0 and np.isclose(x[i0], x0) and np.isclose(y[i0], y0)
    d = np.array([x[i0 + 1] - x0, y[i0 + 1] - y0]) / 0.5
    assert np.allclose(d, [-np.sin(np.radians(pa)), np.cos(np.radians(pa))])
    assert x.min() >= 0 and x.max() <= nx - 1 and y.min() >= 0 and y.max() <= ny - 1
    exact = 2 * x[None] + 3 * y[None] + 10 * np.arange(nz)[:, None]
    assert np.allclose(pv_sample(cube, x, y, order=1), exact)
    inner = (x > 2) & (x < nx - 3) & (y > 2) & (y < ny - 3)                 # whole slit inside the field
    assert np.allclose(pv_sample(cube, x, y, order=1, width=4)[:, inner], exact[:, inner])
    near = pv_sample(cube, x, y, order=0)
    assert np.allclose(near, cube[:, np.rint(y).astype(int), np.rint(x).astype(int)])
    c2 = cube.copy()
    c2[:, 15, 20] = np.nan
    v = pv_sample(c2, np.array([20.0, 20.3, 20.6]), np.array([15.0, 15.0, 15.0]), order=1)
    # NaN where the valid neighbours carry < 1/2 of the weight, otherwise renormalised over them
    assert np.isnan(v[:, :2]).all() and np.allclose(v[:, 2], 2 * 21 + 45 + 10 * np.arange(nz))
    # curved slice through the ring points
    radii = np.array([4., 8., 12., 16.])
    pas = np.array([30., 40., 50., 60.])
    s, x, y = pv_path(25., 20., pas, radii=radii, shape=(ny, nx), step=0.1)
    for r, p_ in zip(radii, pas):
        t = np.radians(p_)
        for sign in (1, -1):
            px, py = 25. - sign * r * np.sin(t), 20. + sign * r * np.cos(t)
            assert np.min(np.hypot(x - px, y - py)) < 0.06
    # segment: offset 0 at the midpoint, from p1 to p2
    s, x, y = pv_path(p1=(5., 5.), p2=(25., 5.), shape=(ny, nx))
    assert np.isclose(x[0], 5.) and np.isclose(x[-1], 25.) and np.isclose(x[np.argmin(np.abs(s))], 15.)


def test_pvslice_header(synthetic_cube):
    from bbarolo.pvslice import PvSlice
    c = synthetic_cube
    zz, yy, xx = np.mgrid[0:c.DimZ(), 0:c.DimY(), 0:c.DimX()]
    c.array = (xx + 0.0 * yy + 0 * zz).astype(np.float32)
    pv = PvSlice(c, x0=31.5, y0=20., angle=45.)
    assert pv.slice()
    h = pv.Head()
    assert h.Ctype(0) == "Offset" and np.isclose(h.Cdelt(0), abs(c.Head().Cdelt(0)))
    i0 = int(round(h.Crpix(0))) - 1
    assert np.isclose(pv.x_locus[i0], 31.5) and np.isclose(pv.y_locus[i0], 20.)
    assert np.all(np.diff(pv.x_locus) > 0)                 # BBarolo's orientation: x increasing
    assert np.allclose(pv.array[0], pv.x_locus, atol=1e-4)  # bilinear sampling of x


def test_smoothing_preserves_flux(synthetic_cube):
    c = synthetic_cube
    c.array[:, 32, 32] = 1.
    sm = Smooth3D()
    sm.setUseScalefac(False)
    sm.setUseBlanks(False)
    out = np.asarray(sm.smooth(c, (0., 0., 0.), (30., 30., 0.), oldarray=c.array))
    assert out.shape == c.array.shape
    assert abs(out.sum() / c.array.sum() - 1) < 1e-3


def test_plot_runner(tmp_path):
    """Plotting scripts run in processes forked from a helper with the libraries loaded."""
    from bbarolo.plotscripts import run_scripts
    (tmp_path / "util_mod.py").write_text("VALUE = 2\n")
    (tmp_path / "ok.py").write_text("import matplotlib.pyplot as plt\nfrom util_mod import VALUE\n"
                                    "print('stdout is discarded')\nfig = plt.figure()\nplt.plot([0, VALUE])\n"
                                    "fig.savefig(__file__.replace('.py', '.pdf'))\n")
    (tmp_path / "bad.py").write_text("raise RuntimeError('broken script')\n")
    ok, bad = str(tmp_path / "ok.py"), str(tmp_path / "bad.py")
    assert run_scripts([ok, ok]) == 0 and (tmp_path / "ok.pdf").is_file()
    assert run_scripts([ok, bad]) == 1


def test_cli_version():
    out = subprocess.run([sys.executable, os.path.join(ROOT, "run_bbarolo.py"), "-v"], capture_output=True, text=True)
    from bbarolo import __version__
    assert out.returncode == 0 and f"BBarolo version {__version__}" in out.stdout


@needs_data
def test_search_matches_cpp(tmp_path):
    p = Param()
    p.readParamString(f"FITSFILE {N2403}\nSEARCH true\nSEARCHTYPE spatial\nSNRCUT 10\nGROWTHCUT 3\n"
                      f"MINPIX 30\nMINCHANNELS 5\nTHREADS 2\nVERBOSE false\nOUTFOLDER {tmp_path}/\n")
    assert p.checkPars(quiet=True)
    c = Cube(N2403, par=p)
    c.search()
    assert c.getNumObj() == 1
    o = c.pObject(0)
    assert o.getSize() == 142786                       # C++: NumVox = 142786
    assert o.getSpatialSize() == 13039                 # C++: NumPix = 13039


@needs_data
def test_2dfit_matches_cpp(tmp_path):
    from bbarolo import FitMod2D
    rm = FitMod2D(N2403, radii=np.arange(15, 1200, 30), xpos=77, ypos=77, vsys=132.8, vrot=120, inc=60, pa=123.7,
                  wfunc=2, free="VROT PA INC", mask="SEARCH", threads=2, verbose=False)
    rings = rm.run(write=False).rings
    # first rings of the C++ run (examples/test/n2403_2dfit.par)
    assert np.allclose(rings['vrot'][:3], [27.58, 49.78, 66.09], atol=0.011)
    assert np.allclose(rings['pa'][:3], [130.22, 125.63, 123.42], atol=0.011)
    assert np.allclose(rings['inc'][:3], [54.84, 61.30, 59.00], atol=0.011)


@needs_data
def test_3dfit_quick(tmp_path):
    from bbarolo import FitMod3D
    f3d = FitMod3D(N2403, radii=np.arange(105, 400, 90), xpos=77, ypos=77, vsys=132.8, vrot=120, vdisp=8, vrad=0,
                   z0=10, inc=60, pa=123.7, mask="SEARCH", free="VROT VDISP", wfunc=2, distance=3.2,
                   twostage=False, threads=2, plots=False, verbose=False)
    res = f3d.run(outfolder=str(tmp_path))
    vrot = np.asarray(res.rings['vrot'])
    # C++ (41 rings run): ~80 km/s at 105", ~115 km/s at 285", ~125 km/s at 375"
    assert np.all(np.diff(vrot) > 0) and 60 < vrot[0] < 100 and 110 < vrot[-1] < 135
    assert res.model.data.shape == (62, 150, 150)
    assert set(res.maps) >= {"0mom", "1mom", "mod_0mom_azim"} and "pv_a" in res.pvs


@needs_data
def test_python_equals_cli(tmp_path):
    """A task class gives the same outputs as the command line with the same parameters."""
    from bbarolo import Ellprof
    par = tmp_path / "ell.par"
    par.write_text(f"FITSFILE {N2403}\nELLPROF true\nNRADII 10\nRADSEP 30\nXPOS 77\nYPOS 77\nINC 60\n"
                   f"PA 123.7\nMASK SEARCH\nVERBOSE false\n")
    subprocess.run([sys.executable, os.path.join(ROOT, "run_bbarolo.py"), "-p", str(par),
                    f"OUTFOLDER={tmp_path}/cli/"], check=True, stdout=subprocess.DEVNULL)
    res = Ellprof.from_parfile(str(par), outfolder=str(tmp_path / "py")).run()
    a = open(tmp_path / "cli" / "ngc2403_densprof.txt").read()
    b = open(tmp_path / "py" / "ngc2403_densprof.txt").read()
    assert a == b and len(res.profile) == 10
    from astropy.io import fits
    assert np.array_equal(fits.getdata(tmp_path / "cli" / "ngc2403_densmap.fits"), res.map.data, equal_nan=True)


def test_simulated_galaxy_cube(tmp_path, monkeypatch):
    """SimulatedGalaxyCube: the model is built in memory (no files), equal to the GALMOD
    task; write_model() writes it."""
    from astropy.io import fits
    from bbarolo import SimulatedGalaxyCube, GalMod
    monkeypatch.chdir(tmp_path)
    s = SimulatedGalaxyCube(axisDim=[41, 41, 32], cdelts=[-20 / 3600., 20 / 3600., 10.], beam=60 / 3600.,
                            bunit='JY/BEAM', obj='sim')
    s.define_galaxy(radii=np.arange(0, 300, 20.), vsys=0, vdisp=10., inc=60., pa=30., dens=10., z0=20., vrot=150.,
                    warpinc=False, warppa=False, flare=False, radmotion=False)
    m = s.run(noise=0.01, seed=1, threads=1)
    assert os.listdir(tmp_path) == []
    assert m is s.model and m.header['OBJECT'] == 'sim' and m.data.shape == (32, 41, 41)
    assert abs(np.std(s.model.data - s.model_nonoise.data) - 0.01) < 1e-3
    g = GalMod(s.f, radii=s.radii, vrot=s.vrot, vrad=s.vrad, vdisp=s.disp, inc=s.inc, pa=s.pa, z0=s.z0,
               dens=s.dens, xpos=s.xpos, ypos=s.ypos, vsys=s.vsys, norm='NONE', mask='NONE', empty=False, linear=0,
               cdens=10, threads=1, verbose=False)
    assert np.array_equal(g.model().data, s.model_nonoise.data)
    # the task reads a copy of the template written to disk (header values with <= 20 characters)
    assert np.allclose(g.run(write=False).model.data, s.model_nonoise.data, rtol=0,
                       atol=1e-12 * s.model_nonoise.data.max())
    fn = s.write_model("out/sim.fits", params=True)
    assert np.array_equal(fits.getdata(fn), s.model.data) and os.path.isfile("out/sim_params.txt")
    with pytest.raises(TypeError, match="executable"):
        s.run(exe="BBarolo")


def test_ring_conventions():
    """RINGMODE: CENTRED annuli (half a ring beyond the first and last radius, with the values
    of the first and last ring), NODES (the rings as given), SHIFTED (BBarolo 1.x)."""
    from bbarolo import Rings
    from bbarolo.galmod import model_rings, outer_edge
    r = Rings().setRings(radii=[30., 90, 150, 210], vrot=[60, 100, 120, 130], dens=[8, 4, 2, 1], radsep=60.)
    c = model_rings(r, "CENTRED")
    assert list(c.radii) == [0, 30, 90, 150, 210, 240] and list(c.vrot) == [60, 60, 100, 120, 130, 130]
    assert list(c.dens) == [8, 8, 4, 2, 1, 1]
    s = model_rings(r, "SHIFTED")
    assert list(s.radii) == [0, 90, 150, 240] and list(s.vrot) == [60, 100, 120, 130]
    assert model_rings(r, "NODES") is r and model_rings(r, "centered").nr == 6
    assert outer_edge(r, "CENTRED") == 240 and outer_edge(r, "NODES") == 210 and outer_edge(r, "SHIFTED") == 240
    one = Rings().setRings(radii=[30.], vrot=[60.], radsep=20.)
    assert list(model_rings(one, "CENTRED").radii) == [20, 30, 40]
    with pytest.raises(ValueError):
        model_rings(r, "EDGES")


@needs_data
def test_diskmodel(tmp_path, monkeypatch):
    """DiskModel = GalMod.model() = GalMod.run().model; model() writes no files; with
    constant ring values CENTRED and SHIFTED give the same disk; DENSFLUX=false with NORM=NONE."""
    import pickle
    from bbarolo import Cube, DiskModel, Rings, GalMod, Param
    monkeypatch.chdir(tmp_path)
    p = Param()
    p.threads, p.verbose = 1, False
    radii = np.arange(15, 400, 30.)
    vrot = 150 * np.tanh(radii / 100.)
    r = Rings().setRings(radii=radii, xpos=77, ypos=77, vsys=132.8, vrot=vrot, vdisp=8, z0=10, inc=60, phi=123.7,
                         dens=1)
    dm = DiskModel(Cube(N2403, par=p))
    gm = GalMod(N2403, radii=radii, xpos=77, ypos=77, vsys=132.8, vrot=vrot, vdisp=8, z0=10, inc=60, pa=123.7,
                dens=1, norm="NONE", threads=1, verbose=False)
    m = gm.model()
    assert os.listdir(tmp_path) == []
    h = dm.hdu(r)
    assert np.array_equal(m.data, h.data) and m.header == h.header
    assert np.array_equal(m.data, gm.run(write=False).model.data)
    assert not np.array_equal(m.data, gm.model(ringmode="SHIFTED").data)
    rc = Rings().setRings(radii=radii, xpos=77, ypos=77, vsys=132.8, vrot=120, vdisp=8, z0=10, inc=60, phi=123.7)
    assert np.array_equal(dm.array(rc), dm.array(rc, ringmode="SHIFTED"))
    assert np.array_equal(pickle.loads(pickle.dumps(dm)).array(r), dm.array(r))
    assert gm.model(ringmode="CENTRED", densflux=False).data.sum() < 0.1 * m.data.sum()   # 1E20 atoms/cm2


@needs_data
def test_ring_arrays_equal_ring_file(tmp_path):
    """Ring parameters given as arrays (one value per ring) == the same values in a ring file."""
    from bbarolo import GalMod
    radii = np.arange(12) * 30 + 15.
    vrot = 150 * np.tanh(radii / 200.)
    np.savetxt(tmp_path / "rings.txt", np.column_stack([radii, vrot]))
    kw = dict(vsys=132.8, xpos=77, ypos=77, vdisp=8, inc=60, pa=123.7, z0=10, dens=1, norm="NONE", mask="NONE",
              threads=1, verbose=False)
    a = GalMod(N2403, radii=radii, vrot=vrot, **kw).run(write=False)
    b = GalMod(N2403, radii=f"file({tmp_path}/rings.txt,1)", vrot=f"file({tmp_path}/rings.txt,2)", **kw).run(
        write=False)
    assert np.array_equal(a.model.data, b.model.data)
    assert np.allclose(a.rings["vrot"], vrot, atol=1e-3)


def test_task_parameters(tmp_path):
    from bbarolo import FitMod3D, GalMod, help
    from bbarolo.keywords import UnknownParameterError, KEYWORDS, lookup
    f = FitMod3D("cube.fits", vrot=120, phi=123.7, errors=True, free="VROT VDISP")
    assert f.parameters == {"VROT": 120, "PA": 123.7, "FLAGERRORS": True, "FREE": "VROT VDISP"}
    assert f.get("ltype") == 1 and f.get("PA") == 123.7
    with pytest.raises(UnknownParameterError, match="VROT"):
        FitMod3D("cube.fits", vrott=1)
    with pytest.raises(ValueError, match="LTYPE"):
        FitMod3D("cube.fits", ltype="sech2")
    with pytest.raises(ValueError, match="another task"):
        FitMod3D("cube.fits", search=True)
    with pytest.raises(ValueError, match="different numbers"):
        GalMod("cube.fits", vrot=[1, 2], inc=[1, 2, 3])
    s = f.to_parfile(str(tmp_path / "f.par"))
    g = FitMod3D.from_parfile(str(tmp_path / "f.par"))
    assert g.to_parfile().split("\n")[1:] == s.split("\n")[1:] and "3DFIT" in s
    names = [n.lower() for k in KEYWORDS for n in (k.name,) + k.aliases]
    assert len(names) == len(set(names)) and lookup("Fit3D").name == "3DFIT"
    help("free")


def test_parameter_table_sets_attributes():
    """Every keyword of the table sets the attributes it declares."""
    from bbarolo.keywords import KEYWORDS, _getattr
    for k in KEYWORDS:
        p = Param()
        before = [_getattr(p, t) for t in k.targets]
        val = {"flag": "true", "int": "7", "float": "7.5", "flagorint": "7", "cdens": "7", "str": "x7",
               "lstr": "x7", "ustr": "x7"}.get(k.kind, "7 8")
        p.setParam(f"{k.name} {val}")
        after = [_getattr(p, t) for t in k.targets]
        assert all(a != b for a, b in zip(after, before)) or all(a == after[0] for a in after), k.name


def test_launchers(tmp_path):
    launcher = os.path.join(ROOT, "bin", "BBarolo")
    os.symlink(launcher, str(tmp_path / "BB"))
    # bin/BBarolo is both a shell and a Python script (python bin/BBarolo works too); links are followed
    for cmd in ([launcher], [sys.executable, launcher], [str(tmp_path / "BB")], [sys.executable, str(tmp_path / "BB")]):
        out = subprocess.run(cmd + ["-v"], capture_output=True, text=True)
        assert "BBarolo version" in out.stdout, (cmd, out.stderr)


@needs_data
def test_analytic_residuals_match_mc(tmp_path):
    """The 3DFIT residuals of ANALYTIC models (with the emulated MC support of the
    blanks) agree with the average residuals of MC models."""
    from bbarolo.galfit import Galfit
    p = Param()
    p.readParamString(f"FITSFILE {N2403}\n3DFIT true\nNRADII 20\nRADSEP 30\nXPOS 77\nYPOS 77\nVSYS 132.8\n"
                      f"VROT 120\nVDISP 8\nINC 60\nPA 123.7\nZ0 10\nMASK SEARCH\nFTYPE 2\nWFUNC 2\n"
                      f"DISTANCE 3.2\nVERBOSE false\nOUTFOLDER {tmp_path}/\n")
    assert p.checkPars(quiet=True)
    g = Galfit(Cube(N2403, par=p))
    g.n_galfit = 1
    r = g.inr.subset([8, 9])
    f = {}
    for vd in (6., 25.):
        r.vdisp[:] = vd
        g.model.setup.galmodtype = "MC"
        blo, bhi = g.model.box(r)
        mc = [g._norm_value(r, g.model.array(r, blo, bhi, final=False, seed=s, layout="yxz"), blo)
              for s in range(6)]
        g.model.setup.galmodtype = "ANALYTIC"
        f[vd] = (np.mean(mc), g.getFuncValue(r))
        assert abs(f[vd][1] / f[vd][0] - 1) < 0.03      # without the MC support: 9-17% higher
    assert (f[6.][0] < f[25.][0]) == (f[6.][1] < f[25.][1])


@needs_data
def test_bayesian_quick(tmp_path):
    pytest.importorskip("dynesty")
    from bbarolo.bayesian import BayesianBBarolo
    f = BayesianBBarolo(N2403, radii=np.arange(60, 400, 120), xpos=77, ypos=77, vsys=132.8, vrot=100, vdisp=10,
                        inc=60, pa=123.7, z0=10, mask="SEARCH", outfolder=str(tmp_path) + "/", ringmode="SHIFTED")
    assert f.like_kwargs['likemask'] == 'dilated'                 # the default
    f._set_like_options(dict(noisecorr=1))                         # independent voxels
    f._setup(['vrot', 'vdisp_single'])
    # with sigma=1 and no beam factor (as the original BayesianBBarolo) and the rings of
    # BBarolo 1.x, the log-likelihood along vrot of ring 2 peaks at ~110 km/s
    scan = [f._log_likelihood(np.array([80., v, 120., 13.]), sigma=1.) for v in (90., 110., 130.)]
    assert scan[1] > scan[0] and scan[1] > scan[2]
    f.update(ringmode="CENTRED")                         # centred annuli (default): same peak
    f._setup(['vrot', 'vdisp_single'])
    scan = [f._log_likelihood(np.array([80., v, 120., 13.]), sigma=1.) for v in (90., 110., 130.)]
    assert scan[1] > scan[0] and scan[1] > scan[2]
    f.run(freepar=['vrot_single', 'vdisp_single'], threads=1, dynamic=False, verbose=False,
              sampler_kwargs=dict(nlive=30, rstate=np.random.default_rng(1)), run_kwargs=dict(maxiter=150))
    assert f.params.shape == (2,) and np.all(np.isfinite(f.params))
    assert not f._galfit.par.TWOSTAGE                   # the plots then read rings_final1.txt
    # the posterior errors are written as the 3DFIT error columns (and FREE is restored)
    # (with the errors of the deconvolved density profile)
    head = open(os.path.join(str(tmp_path), "rings_final1.txt")).readline().split()
    assert head[13:] == ["E_VROT1", "E_VROT2", "E_DISP1", "E_DISP2", "E_DENS1", "E_DENS2"]
    assert not f._galfit.par.flagERRORS
    f.write_bestmodel(plots=False, verbose=False)
    assert os.path.isfile(os.path.join(str(tmp_path), "ngc2403mod_none.fits"))
    f.save_results(str(tmp_path / "res.npy"))
    g = BayesianBBarolo(N2403, outfolder=str(tmp_path) + "/")
    g.load_results(str(tmp_path / "res.npy"))
    assert np.allclose(g.params, f.params)
    assert g.like_kwargs['noisecorr'] == f.like_kwargs['noisecorr'] == 'measured'
    assert np.isclose(g.likefactor, f.likefactor)
    old = np.load(str(tmp_path / "res.npy"), allow_pickle=True).item()
    lk = old.pop('like_kwargs')                              # a file saved before like_kwargs and noisecorr
    old.update(likemask=lk['likemask'], likemask_dilation=lk['dilation'], densprof_smooth=lk['densprof_smooth'])
    np.save(str(tmp_path / "old.npy"), old)
    g.load_results(str(tmp_path / "old.npy"))
    assert g.like_kwargs['noisecorr'] == 'beam' and np.isclose(g.likefactor, 1. / g.beam_area)
    assert np.array_equal(g._inri.r['vrot'], f._inri.r['vrot']) and g.get("mask") == "SEARCH"


@needs_data
def test_bayesian_likelihood_checks(tmp_path, capsys):
    """Warnings: a free sigma not used by user-defined residuals; a collapsed posterior."""
    from bbarolo.bayesian import BayesianBBarolo
    f = BayesianBBarolo(N2403, radii=np.arange(60, 400, 120), xpos=77, ypos=77, vsys=132.8, vrot=100, vdisp=10,
                        inc=60, pa=123.7, z0=10, mask="SEARCH", outfolder=str(tmp_path) + "/")
    f._setup(['vrot', 'sigma'])
    f._log_like_func = f._log_likelihood
    f._calculate_residuals = lambda model, data, mask, sigma, **kw: 0.5 * np.nansum((data * mask - model) ** 2)
    f._check_likelihood({}, True)
    assert "do not depend on it" in capsys.readouterr().out
    f._calculate_residuals = lambda model, data, mask, sigma, **kw: 0.5 * np.nansum((data * mask - model) ** 2) \
        / sigma ** 2
    f._check_likelihood({}, True)
    assert "do not depend on it" not in capsys.readouterr().out
    f._set_sampling_stats(np.ones((3, f.ndim)), np.array([1., 1e-30, 1e-30]))
    assert f.ess < 1.01


@needs_data
def test_bayesian_likelihood_noise_and_beam(tmp_path):
    """Default likelihood: sigma = measured noise, log-likelihood x 1/(beam area in pixels),
    independent of the box in which the model is computed."""
    from bbarolo.bayesian import BayesianBBarolo
    f = BayesianBBarolo(N2403, radii=np.arange(60, 400, 120), xpos=77, ypos=77, vsys=132.8, vrot=100, vdisp=10,
                        inc=60, pa=123.7, z0=10, mask="SEARCH", outfolder=str(tmp_path) + "/",
                        galmodtype="ANALYTIC")
    f.like_kwargs['noisecorr'] = 'beam'                    # the default is 'measured'
    f._setup(['vrot', 'vdisp_single'])
    out = f.data[(f.mask == 0) & np.isfinite(f.data)]
    assert np.isclose(f.noiserms, 1.4826 * np.median(np.abs(out - np.median(out))))
    assert abs(f.beam_area - 15.39) < 0.05 and np.isclose(f.likefactor, 1. / f.beam_area)
    th = np.array([80., 108., 120., 13.])
    L = f._log_likelihood(th)
    f.like_kwargs['noisecorr'] = 1.                        # independent voxels
    f._set_likefactor()
    assert f.likefactor == 1. and np.isclose(L, f._log_likelihood(th) / f.beam_area)
    f.like_kwargs['noisecorr'] = 'beam'
    f._set_likefactor()
    assert np.isclose(L, f._log_likelihood(th, sigma=f.noiserms))
    # the same likelihood with the model computed in the whole likelihood region
    f._model_box = lambda rings: f._likebox
    assert np.isclose(f._log_likelihood(th), L, rtol=1e-9)


def test_noise_correlation():
    """Correlation area of the noise: twice the beam area for white noise convolved with the
    beam, one beam area when the correlation is the beam itself, 8/3 channels after Hanning."""
    from scipy.ndimage import gaussian_filter
    from bbarolo.bayesian import noise_correlation
    rng = np.random.default_rng(5)
    b = 4.
    s = b / (2 * np.sqrt(2 * np.log(2)))
    barea = np.pi * b * b / (4 * np.log(2))
    n = gaussian_filter(rng.standard_normal((40, 96, 96)), sigma=(0, s, s), mode="wrap")
    nc = noise_correlation(n, None, beam_fwhm=b)
    assert abs(nc['spatial'] / (2 * barea) - 1) < 0.12 and abs(nc['spectral'] - 1) < 0.05
    m = gaussian_filter(rng.standard_normal((40, 96, 96)), sigma=(0, s / np.sqrt(2), s / np.sqrt(2)), mode="wrap")
    assert abs(noise_correlation(m, None, beam_fwhm=b)['spatial'] / barea - 1) < 0.12
    h = 0.25 * np.roll(n, 1, 0) + 0.5 * n + 0.25 * np.roll(n, -1, 0)
    assert abs(noise_correlation(h, None, beam_fwhm=b)['spectral'] / (8 / 3) - 1) < 0.06
    # the emission (mask) is excluded
    e = n.copy()
    mask = np.zeros(n.shape, bool)
    mask[10:30, 30:60, 30:60] = True
    e[mask] += 10.
    ne = noise_correlation(e, mask, beam_fwhm=b)
    assert abs(ne['spatial'] / (2 * barea) - 1) < 0.15 and ne['spatial_from'] == 'free channels'
    eh = h.copy()
    eh[mask] += 10.
    ne = noise_correlation(eh, mask, beam_fwhm=b)
    assert abs(ne['spectral'] / (8 / 3) - 1) < 0.06 and ne['spectral_from'] == 'free channels'
    assert noise_correlation(n[:2, :20, :20], None, beam_fwhm=b) is None        # too few voxels


@needs_data
def test_bayesian_noisecorr_and_fracerr(tmp_path):
    """The fractional model error ('fracerr', variance sigma^2 + (fracerr*model)^2) and the
    correlation factor of the likelihood (noisecorr = 'beam', 'measured' or a number)."""
    from bbarolo.bayesian import BayesianBBarolo
    kw = dict(radii=np.arange(60, 400, 120), xpos=77, ypos=77, vsys=132.8, vrot=100, vdisp=10, inc=60, pa=123.7,
              z0=10, mask="SEARCH", outfolder=str(tmp_path) + "/", galmodtype="ANALYTIC")
    f = BayesianBBarolo(N2403, **kw)
    f._setup(['vrot', 'vdisp_single', 'fracerr'])
    g = BayesianBBarolo(N2403, **kw)
    g._setup(['vrot', 'vdisp_single'])
    th = np.array([80., 108., 120., 13.])
    assert np.isclose(f._log_likelihood(np.r_[th, 1e-12]), g._log_likelihood(th), rtol=1e-10)
    L = f._log_likelihood(np.r_[th, 0.1])
    assert np.isclose(L, g._log_likelihood(th, fracerr=0.1), rtol=1e-12)
    # direct calculation
    model, bhi, blo = g._calculate_model(g._update_rings(g._inri, th))[:3]
    d = g.data[:, blo[1]:bhi[1], blo[0]:bhi[0]].astype(float)
    reg = g._region[blo[1]:bhi[1], blo[0]:bhi[0]][None] & np.isfinite(d) & g._likemask[:, blo[1]:bhi[1], blo[0]:bhi[0]]
    s2 = g.noiserms ** 2
    v = s2 + 0.01 * model ** 2
    nll = 0.5 * np.sum(((d - model) ** 2 / v)[reg]) + 0.5 * np.sum(np.log(v / s2)[reg & (model != 0)]) \
        + 0.5 * g._sumsq_outside(blo, bhi) / s2 + 0.5 * g.nvox * np.log(2 * np.pi * s2)
    assert np.isclose(L, -g.likefactor * nll, rtol=1e-9)
    assert L > g._log_likelihood(th) and g.chi2_reduced(th, fracerr=0.1) < g.chi2_reduced(th)
    assert f.priors['fracerr']['name'] == 'loguniform' and 'fracerr' not in f._update_rings(f._inri, np.r_[th, .1]).r
    # correlation of the noise: measured by default, the beam area with noisecorr='beam'
    assert g.like_kwargs['noisecorr'] == 'measured'
    nc = g.noise_correlation
    # interferometric (THINGS) cube with Hanning smoothing: correlation ~ beam, 0.6-0.7 at lag 1
    assert 0.9 < nc['spatial'] / g.beam_area < 1.6 and 2.2 < nc['spectral'] < 3.2
    assert nc['spatial_from'] == 'free channels' and nc['spectral_from'] == 'free channels'
    assert np.isclose(g.likefactor, 1. / nc['volume']) and np.isclose(g.likefactor2d, 1. / nc['spatial'])
    g.like_kwargs['noisecorr'] = 'beam'
    g._set_likefactor()
    assert np.isclose(g.likefactor, 1. / g.beam_area) and np.isclose(g.likefactor2d, 1. / g.beam_area)
    for n in (50., 0.5):                                   # N voxels per independent value: x 1/N
        g.like_kwargs['noisecorr'] = n
        g._set_likefactor()
        assert np.isclose(g.likefactor, 1. / n) and np.isclose(g.likefactor2d, 1. / n)
    for bad in ('pixels', 0.):
        g.like_kwargs['noisecorr'] = bad
        with pytest.raises(ValueError):
            g._set_likefactor()
    # the options of the likelihood are in like_kwargs (checked)
    with pytest.raises(TypeError, match="like_kwargs"):
        f.run(freepar=['vrot_single'], noisecorr='beam')
    for bad in (dict(factor=1.), dict(likemask='mask'), dict(noisecorr=-1), dict(dilation=1)):
        with pytest.raises(ValueError):
            f.run(freepar=['vrot_single'], like_kwargs=bad)
    # a short run with both
    pytest.importorskip("dynesty")
    f.run(freepar=['vrot_single', 'fracerr'], like_kwargs=dict(noisecorr='measured'), threads=1, dynamic=False,
          verbose=False, sampler_kwargs=dict(nlive=30, rstate=np.random.default_rng(1)), run_kwargs=dict(maxiter=150))
    assert f.freepar_names == ['vrot', 'fracerr'] and np.all(np.isfinite(f.params))
    assert np.isclose(f.likefactor, 1. / f.noise_correlation['volume'])


@needs_data
def test_bayesian_likemask(tmp_path):
    """The voxels of the likelihood (likemask): 'dilated' (default: data not masked, in the
    dilated mask), 'none' (all the voxels of the region); the map of run_geometry and of the
    observed density profile; the default prior of sigma."""
    from bbarolo.bayesian import BayesianBBarolo, dilate_mask
    from bbarolo.moments import compute_moments
    # dilation by a cylinder: a disk in each channel, then +-n channels
    m = np.zeros((9, 21, 21), bool)
    m[4, 10, 10] = True
    dm = dilate_mask(m, 3., 2)
    yy, xx = np.mgrid[:21, :21]
    assert np.array_equal(dm[2:7], np.broadcast_to((xx - 10) ** 2 + (yy - 10) ** 2 <= 9, (5, 21, 21)))
    assert not dm[:2].any() and not dm[7:].any()
    kw = dict(radii=np.arange(60, 400, 120), xpos=77, ypos=77, vsys=132.8, vrot=100, vdisp=10, inc=60, pa=123.7,
              z0=10, mask="SEARCH", outfolder=str(tmp_path) + "/", galmodtype="ANALYTIC")
    f = BayesianBBarolo(N2403, **kw)
    free = ['vrot', 'vdisp_single']
    th = np.array([80., 108., 120., 13.])
    assert f.like_kwargs['likemask'] == 'dilated'                 # the default
    f._setup(free)
    s2 = f.noiserms ** 2

    def direct(g, use, d):
        """ log-likelihood and reduced chi2 over the voxels use, data d """
        rings = g._update_rings(g._inri.copy(), th)
        model = np.asarray(g._calculate_model(rings, fullcube=True)[0], float)
        assert g.nvox == use.sum()
        ss = np.sum((d[use] - model[use]) ** 2) / s2
        return -g.likefactor * (0.5 * ss + 0.5 * use.sum() * np.log(2 * np.pi * s2)), ss / use.sum()
    use = f._region[None] & np.isfinite(f.data)
    # dilated by 1 beam FWHM and 3 channels: only those voxels, data not masked
    h = f._cube_obj.Head()
    lm = dilate_mask(f.mask, abs(h.Bmaj() / h.Cdelt(0)), 3)
    assert np.array_equal(f._likemask, lm) and f._maskmode == 2 and lm.sum() > 2 * f.mask.astype(bool).sum()
    L = f._log_likelihood(th)
    Ld, chi2 = direct(f, lm & use, f.data)
    assert np.isclose(L, Ld, rtol=1e-8) and np.isclose(f.chi2_reduced(th), chi2, rtol=1e-8)
    # independent of the box in which the model is computed
    box = f._model_box
    f._model_box = lambda rings: f._likebox
    assert np.isclose(f._log_likelihood(th), L, rtol=1e-9)
    f._model_box = box
    # the map of run_geometry and of the density profiles: the same voxels, in the units of the
    # zeroth moment of MomentMap for the observed profile of the outputs (Ellprof)
    mom0 = np.where(lm & np.isfinite(f.data), f.data, 0.).sum(axis=0)
    assert np.allclose(f.data_mom0, mom0)
    m0 = compute_moments(f._cube_obj, lm)[0]
    assert np.allclose(f._likemask_map(), m0, rtol=1e-4, atol=1e-6, equal_nan=True)
    assert f._ellprof is None                                # not used in the fit
    ell = f._observed_profile(f._inri)
    assert np.array_equal(ell.im.array, f._likemask_map(), equal_nan=True) and np.all(np.isfinite(ell.Surfdens))
    # map noise with the measured spectral correlation of the noise (contiguous channels)
    n = int(f._nvox_map.max())
    rho = f.noise_correlation['rho_spectral']
    var = n + 2 * sum((n - k) * rho[k] for k in range(1, len(rho)))
    assert np.isclose(f.map_rms.max(), f.noiserms * np.sqrt(var)) and var > 2 * n
    # with the beam factor, the channels are independent
    f.like_kwargs['noisecorr'] = 'beam'
    f._setup(free)
    assert np.isclose(f.map_rms.max(), f.noiserms * np.sqrt(n))
    f.like_kwargs['noisecorr'] = 'measured'
    # all the voxels of the region
    f.like_kwargs['likemask'] = 'none'
    f._setup(free)
    assert f._likemask.all() and f.nvox == use.sum() and f._ellprof is None
    assert np.isclose(f._log_likelihood(th), direct(f, use, f.data)[0], rtol=1e-8)
    # back to 'dilated': the same likelihood as before
    f.like_kwargs['likemask'] = 'dilated'
    f._setup(free)
    assert np.isclose(f._log_likelihood(th), L, rtol=1e-12)
    for bad in ('dilate', 'mask'):                          # 'mask' (data x mask) was removed
        f.like_kwargs['likemask'] = bad
        with pytest.raises(ValueError):
            f._setup(free)
    # saved and loaded (older files: 'dilated')
    f.like_kwargs.update(likemask='dilated', dilation=(1.5, 4))
    f._setup(free)
    f._set_sampling_stats(np.tile(th, (3, 1)), np.ones(3))
    f.save_results(str(tmp_path / "res.npy"))
    g = BayesianBBarolo(N2403, outfolder=str(tmp_path) + "/")
    g.load_results(str(tmp_path / "res.npy"), sampler='nautilus')
    assert g.like_kwargs['likemask'] == 'dilated' and g.like_kwargs['dilation'] == (1.5, 4) and g.nvox == f.nvox
    old = np.load(str(tmp_path / "res.npy"), allow_pickle=True).item()
    lk = old.pop('like_kwargs')                              # a file saved before like_kwargs
    old.update(noisecorr=lk['noisecorr'], likemask='mask', densprof_smooth=lk['densprof_smooth'])
    np.save(str(tmp_path / "old.npy"), old)
    g.load_results(str(tmp_path / "old.npy"), sampler='nautilus')
    assert g.like_kwargs['likemask'] == 'dilated' and g.like_kwargs['dilation'] == (1., 3) and g._maskmode == 2
    # default prior of sigma: log-uniform between 0.5 and 1000 times the measured noise
    f._setup(['vrot', 'sigma'])
    p = f.prior_distr['sigma']
    assert np.isclose(p.ppf(0.), 0.5 * f.noiserms) and np.isclose(p.ppf(1.), 1000. * f.noiserms)
    assert f.priors['sigma']['a'] is None                      # stays relative to the noise


@needs_data
def test_bayesian_errors_on_mock_data(tmp_path):
    """On mock cubes with noise correlated over the beam, the posterior width of vrot matches
    the scatter of the estimates (it is ~4x too small without the beam factor)."""
    from scipy.ndimage import gaussian_filter
    from bbarolo.bayesian import BayesianBBarolo
    f = BayesianBBarolo(N2403, radii=np.arange(60, 400, 120), xpos=77, ypos=77, vsys=132.8, vrot=100, vdisp=10,
                        inc=60, pa=123.7, z0=10, mask="SEARCH", outfolder=str(tmp_path) + "/",
                        galmodtype="ANALYTIC")
    f._setup(['vrot', 'vdisp_single'])
    truth = np.array([80., 108., 120., 13.])
    model = f._calculate_model(f._update_rings(f._inri, truth), fullcube=True)[0]
    h = f._cube_obj.Head()
    sk = np.sqrt(h.Bmaj() * h.Bmin()) / abs(h.Cdelt(0)) / np.sqrt(2) / 2.3548   # correlation = beam
    rng = np.random.default_rng(3)
    dv = np.array([-0.2, 0., 0.2])
    est, wid = [], []
    for _ in range(12):
        n = np.stack([gaussian_filter(c, sk, mode="wrap") for c in rng.standard_normal(f.data.shape)])
        f.data = (model + 0.005 * n / n.std()).astype(np.float32)
        f._setup_noise()
        f._setup_likebox()
        L = [f._log_likelihood(truth + np.array([0, d, 0, 0])) for d in dv]
        a, b, _ = np.polyfit(dv, L, 2)                       # Gaussian posterior
        est.append(-b / (2 * a))
        wid.append(np.sqrt(-1 / (2 * a)))
    ratio = np.mean(wid) / np.std(est, ddof=1)
    assert 0.8 < ratio < 1.8
    assert ratio * np.sqrt(f.likefactor) < 0.5                # without the beam factor
    # default noisecorr='measured': the noise is correlated as the beam, so its correlation
    # volume is about one beam area
    assert f.like_kwargs['noisecorr'] == 'measured' and 0.8 < f.noise_correlation['volume'] / f.beam_area < 1.3


def test_bayesian_keywords(capsys):
    """BAYESIAN3DFIT in parameter files: FREE, PRIORS, SAMPLER, SAMPLEROPTS and RUNOPTS."""
    import io
    from bbarolo.params import bayes_freepar, bayes_priors, bayes_options, bayes_name, TASKS_LIST
    from bbarolo import help
    assert [bayes_name(n) for n in ("PA_SINGLE", "Disp", "vrot_FUNC", "phi")] == ['phi_single', 'vdisp', 'vrot_func',
                                                                                 'phi']
    assert bayes_freepar("VROT VDISP PA_SINGLE INC_SINGLE RADMAX") == ['vrot', 'vdisp', 'phi_single', 'inc_single',
                                                                      'radmax']
    assert bayes_freepar("vrot_func, disp phi_single sigma FRACERR") == ['vrot_func', 'vdisp', 'phi_single', 'sigma',
                                                                         'fracerr']
    for bad in ("VROT VROT_SINGLE", "VROTT", "INC_FUNC", "SIGMA_SINGLE", "VROT_P1", ""):
        with pytest.raises(ValueError):
            bayes_freepar(bad)
    assert bayes_freepar("INC_FUNC", funcs=("inc",)) == ['inc_func']
    pr = bayes_priors("vrot=(uniform,loc=0,scale=250) vdisp=(uniform,loc=1,scale=30) ")
    assert pr == {'vrot': dict(name='uniform', loc=0., scale=250.), 'vdisp': dict(name='uniform', loc=1., scale=30.)}
    pr = bayes_priors("INC=(norm,60,5), PA_SINGLE = truncnorm(-2, 2, loc=100, scale=10) vrot_p2=(loguniform,1,100) "
                      "sigma=(loguniform,a=none,b=none)")
    assert pr == {'inc': dict(name='norm', loc=60., scale=5.), 'phi': dict(name='truncnorm', a=-2., b=2., loc=100.,
                                                                           scale=10.),
                  'vrot_p2': dict(name='loguniform', a=1., b=100.), 'sigma': dict(name='loguniform', a=None, b=None)}
    for bad in ("vrot=(unif,0,1)", "vrot=(uniform,0,1,2)", "vrot=(uniform,loc=0,1)", "vrott=(uniform,0,1)",
                "vrot=(uniform,0,-1)", "vrot=uniform", "vrot=(uniform,locc=0)", "inc=(norm,loc=none)",
                "vrot=(uniform,0,1) junk", "vrot=(uniform,x,1)"):
        with pytest.raises(ValueError):
            bayes_priors(bad)
    assert bayes_options("nlive=200 bound = multi dynamic=FALSE dlogz=0.5 maxiter=1e5") == \
        dict(nlive=200, bound='multi', dynamic=False, dlogz=0.5, maxiter=100000)
    for bad in ("nlive", "nlive=", "=3", "1a=2"):
        with pytest.raises(ValueError):
            bayes_options(bad)
    from bbarolo.params import bayes_likeopts
    assert bayes_likeopts("LIKEMASK=None, noisecorr = 2.5 dilation=(1.5, 4) sigma=0.01 usebbres=false") == \
        dict(likemask='none', noisecorr=2.5, dilation=(1.5, 4), sigma=0.01, bbres=False)
    for bad in ("likemask=mask", "noisecorr=0", "dilation=1", "factor=1", "sigma=0", "lambdas=1", "bbres=2"):
        with pytest.raises(ValueError):
            bayes_likeopts(bad)
    # Param: the values as given, the checks, the printing, the task list and the help
    p = Param()
    p.readParamString("FITSFILE m.fits\nBAYESIAN3DFIT true\n3DFIT true\nFREE VROT INC_SINGLE\n"
                      "PRIORS inc=(uniform,40,40)\nSAMPLER nautilus\n")
    assert p._given["FREE"] == "VROT INC_SINGLE" and p.gf.flagBAYESIAN and p.gf.SAMPLER == "NAUTILUS"
    assert p.checkPars() and not p.gf.flagGALFIT                # 3DFIT turned off
    assert "Turning off 3DFIT" in capsys.readouterr().out
    s = str(p)
    assert "[BAYESIAN3DFIT]  =  true" in s and "inc=(uniform,40,40)" in s and "[3DFIT]" not in s
    assert "[NORM]" not in s and "[NOISERMS]" not in s and "VROT INC_SINGLE" in s
    # FREE is required (in any case)
    p.set("free", None)
    assert "FREE" not in p._given and p.gf.FREE == "VROT VDISP INC PA"
    assert not p.checkPars() and "FREE is not an optional parameter" in capsys.readouterr().out
    p.set("free", "vrot pa_single Inc_Single")
    assert p.checkPars()
    capsys.readouterr()
    st = io.StringIO()
    Param().printDefaults(st, "BAYESIAN3DFIT")
    assert "[FREE]  =  (required)" in st.getvalue() and "[NORM]" not in st.getvalue()
    p = Param()
    p.readParamString("FITSFILE m.fits\nBAYESIAN3DFIT true\nSAMPLER emcee\nFREE VROT INC_FUNC\n"
                      "PRIORS vrot=(xx,1)\nSAMPLEROPTS nlive\nLIKEOPTS factor=1\n")
    assert not p.checkPars()
    out = capsys.readouterr().out
    assert all(k in out for k in ("SAMPLER EMCEE", "FREE: INC_FUNC", "PRIORS: vrot", "SAMPLEROPTS: cannot",
                                  "LIKEOPTS: unknown option 'factor'"))
    assert "BAYESIAN3DFIT" in [t[0] for t in TASKS_LIST]
    help("BAYESIAN3DFIT")
    out = capsys.readouterr().out
    assert "SAMPLER" in out and "PRIORS" in out and "RUNOPTS" in out and "VVERT" in out
    assert "\n  NORM " not in out and "\n  NOISERMS " not in out


def _mock_cube(tmp_path, name="mock"):
    """A small model cube (64x64x40, beam 30") with noise; returns its file and rings."""
    from astropy.io import fits
    from bbarolo.galmod import DiskModel
    hdu = emptyfits([64, 64, 40], [-10. / 3600., 10. / 3600., 10.], beam=30. / 3600., bunit="JY/BEAM", obj="MOCK")
    hdu.writeto(str(tmp_path / "empty.fits"), overwrite=True)
    p = Param()
    p.verbose = False
    c = Cube(str(tmp_path / "empty.fits"), par=p)
    radii = np.arange(4) * 40. + 20.
    kw = dict(radii=radii, xpos=31.5, ypos=31.5, vsys=0., vrot=150., vdisp=10., z0=0., inc=60.)
    m = DiskModel(c, galmodtype="ANALYTIC").array(Rings().setRings(**kw, phi=45., dens=2. * np.exp(-radii / 60.)))
    noise = np.random.default_rng(4).normal(0., m.max() / 100., m.shape)
    fn = str(tmp_path / f"{name}.fits")
    fits.PrimaryHDU((m + noise).astype(np.float32), header=hdu.header).writeto(fn)
    return fn


def test_bayesian_task(tmp_path, capsys):
    """The BAYESIAN3DFIT task of parameter files: BBarolo -p (the samples, the outputs, the
    statistics and the plots), bbarolo.run and BayesianBBarolo.from_parfile."""
    pytest.importorskip("dynesty")
    from bbarolo import BayesianBBarolo
    from bbarolo.bayesian import _task_fitter
    from bbarolo.cli import main
    import bbarolo
    fn = _mock_cube(tmp_path)
    par = tmp_path / "bbb.par"
    par.write_text(f"FITSFILE {fn}\nOUTFOLDER {tmp_path}/cli\nBAYESIAN3DFIT true\nNRADII 4\nRADSEP 40\nXPOS 31.5\n"
                   "YPOS 31.5\nVSYS 0\nVROT 120\nVDISP 15\nZ0 0\nINC 60\nPA 45\nMASK SEARCH\nGALMODTYPE ANALYTIC\n"
                   "LINEAR 0\nFREE VROT_SINGLE VDISP_SINGLE\nPRIORS vrot=(uniform,100,100) vdisp=(uniform,1,30)\n"
                   "SAMPLEROPTS nlive=20 dynamic=false seed=2\nRUNOPTS dlogz=1\nTHREADS 1\nPLOTS true\n"
                   "LIKEOPTS likemask=dilated dilation=1,3 noisecorr=measured\n")
    # the fitter: the parameters as given, the data of the cube read by BBarolo (here changed in memory)
    p = Param(str(par))
    assert p.checkPars(quiet=True)
    c = Cube(fn, par=p)
    c.array *= 2.
    f = _task_fitter(c, p)
    assert np.array_equal(f._cube_obj.array, c.array) and np.allclose(f._inri.r['radii'], [20, 60, 100, 140])
    assert f.priors['vrot'] == dict(name='uniform', loc=100., scale=100.)
    assert not any(k in f._params for k in ("FITSFILE", "BAYESIAN3DFIT", "3DFIT"))
    assert f._par.gf.flagGALFIT and not f._par.gf.flagBAYESIAN and f._par.gf.FREE == "VROT VDISP INC PA"
    assert f._run_defaults()['FREE'] == ['vrot_single', 'vdisp_single']
    # the command line
    assert main(["-p", str(par)]) == 0
    out = capsys.readouterr().out
    assert "Fitting 2 parameters: ['vrot', 'vdisp']" in out and "Best-fit parameters" in out
    assert "Deconvolved density profile" in out
    cli = tmp_path / "cli"
    for name in ("rings_final1.txt", "densprof.txt", "densprof_deconvolved.txt", "MOCKmod_none.fits",
                 "MOCK_dynesty_results.npy", "MOCK_corner.pdf", "MOCK_profiles.pdf", "MOCK_parameters.pdf"):
        assert (cli / name).is_file(), name
    head = open(cli / "rings_final1.txt").readline().split()
    assert head[13:] == ["E_VROT1", "E_VROT2", "E_DISP1", "E_DISP2", "E_DENS1", "E_DENS2"]
    g = BayesianBBarolo(fn, outfolder=str(tmp_path / "re"))
    g.load_results(str(cli / "MOCK_dynesty_results.npy"))
    assert g.freepar_names == ['vrot', 'vdisp'] and 100 < g.params[0] < 200
    assert g.like_kwargs == dict(likemask='dilated', dilation=(1., 3), noisecorr='measured', densprof_smooth=0.1,
                                 bbres=False)
    capsys.readouterr()
    # bbarolo.run (the same parameters, quiet, no plots): a Result with the fitter
    res = bbarolo.run(str(par), outfolder=str(tmp_path / "run"), verbose=False, plots=False,
                      runopts="maxiter=40", free="vrot_single Disp_Single")["BAYESIAN3DFIT"]
    assert isinstance(res.bayes, BayesianBBarolo) and len(res.rings) == 4 and "e_vrot1" in res.rings.colnames
    assert res.bayes.freepar_names == ['vrot', 'vdisp']
    assert res.samples.endswith("MOCK_dynesty_results.npy") and len(res.densprof_deconvolved) == 4
    assert not any(x.endswith(".pdf") for x in res.files) and capsys.readouterr().out.strip() == ""
    # from_parfile: the rings of the file; FREE, SAMPLER, SAMPLEROPTS, RUNOPTS: the defaults of run()
    h = BayesianBBarolo.from_parfile(str(par), sampler="nautilus")
    assert np.allclose(h._inri.r['vrot'], 120) and h._run_defaults()['SAMPLER'] == 'nautilus'
    assert h.get("priors").startswith("vrot=(uniform") and h.priors['vdisp']['scale'] == 30.
    with pytest.raises(ValueError, match="BAYESIAN3DFIT: FREE"):
        h.update(free="VROT_P1")
    with pytest.raises(ValueError, match="another task"):
        h.update(**{"3dfit": True})
    h.update(priors="inc=(norm,60,5)")
    assert h.priors['inc'] == dict(name='norm', loc=60., scale=5.) and h.priors['vrot']['loc'] == 100.
    # a prior changed after a run is used by the next run; a distribution given in prior_distr is kept
    from scipy import stats
    h._setup(['vrot_single'])
    assert h.prior_distr['vrot'].kwds == dict(loc=100., scale=100.)
    h.priors['vrot'] = dict(name='uniform', loc=50., scale=10.)
    h._setup(['vrot_single'])
    assert h.prior_distr['vrot'].kwds == dict(loc=50., scale=10.)
    h.prior_distr['vrot'] = stats.norm(150., 5.)
    h._setup(['vrot_single'])
    assert h.prior_distr['vrot'].dist.name == 'norm'
    # the free parameters are required; names in any case (PA is phi); run_geometry's default
    seen = []

    def fake_setup(freepar, **kwargs):
        seen.append(list(freepar))
        raise RuntimeError("stop")
    h._setup = fake_setup
    for args in (dict(freepar=['VROT_SINGLE', 'Disp_Single', 'PA']), dict(freepar="vrot_single, INC_SINGLE"), {}):
        with pytest.raises(RuntimeError):
            h.run(**args)
    with pytest.raises(RuntimeError):
        h.run_geometry()
    with pytest.raises(RuntimeError):
        h.run_geometry("INC_SINGLE PA_SINGLE")
    assert seen == [['vrot_single', 'vdisp_single', 'phi'], ['vrot_single', 'inc_single'],
                    ['vrot_single', 'vdisp_single'], ['xpos_single', 'ypos_single', 'inc_single', 'phi_single'],
                    ['inc_single', 'phi_single']]
    h.update(free=None)
    with pytest.raises(ValueError, match="give the free parameters"):
        h.run()
