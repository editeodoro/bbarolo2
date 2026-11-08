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
    ell = f._observed_profile(f._fit0)
    obs = np.abs([ell.getSurfDensFaceOn(i) for i in range(8)]) / dens
    assert obs[0] < 0.8
    dec = np.array(f._fit0.r['dens'])
    f._fit0.modify_parameter("dens", obs * dens, makeobj=True)
    assert f.chi2_reduced(np.array([150.])) > 3
    f._fit0.modify_parameter("dens", dec, makeobj=True)
    for smooth in (0., 0.1):
        f.like_kwargs['densprof_smooth'] = smooth
        f._setup(['vrot_single'])
        assert np.all(np.abs(f._fit0.r['dens'] / dens - 1) < 0.08)
        assert f.chi2_reduced(np.array([150.])) < 1.05
    # saved and loaded
    f._set_sampling_stats(np.full((3, 1), 150.), np.ones(3))
    f.save_results(str(tmp_path / "res.npy"))
    g = BayesianBBarolo(fn, outfolder=str(tmp_path) + "/")
    g.load_results(str(tmp_path / "res.npy"), sampler='nautilus')
    assert g.like_kwargs['densprof_smooth'] == 0.1 and np.allclose(g._fit0.r['dens'], f._fit0.r['dens'])
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
    ring_dens = np.loadtxt(str(tmp_path / "MOCK_rings_final1.txt"), usecols=6)
    assert np.allclose(ring_dens, g.outri.r['dens'], rtol=0.06)
    head = open(str(tmp_path / "MOCK_rings_final1.txt")).readline().split()
    assert head[11:] == ["E_VROT1", "E_VROT2", "E_DENS1", "E_DENS2"]
    e_dens = np.loadtxt(str(tmp_path / "MOCK_rings_final1.txt"), usecols=(13, 14))
    assert np.allclose(e_dens, np.c_[-dp['err_lo'], dp['err_hi']], rtol=1e-2, atol=1e-3)
    t = np.loadtxt(str(tmp_path / "MOCK_densprof_deconvolved.txt"))
    assert np.allclose(t[:, 0], radii) and np.allclose(t[:, 1], g.outri.r['dens'], rtol=1e-5)
    assert np.allclose(t[:, 2], dp['err_lo'], rtol=1e-5) and np.allclose(t[:, 3], dp['err_hi'], rtol=1e-5)
    assert np.allclose(t[:, 4] / dens, obs, rtol=1e-3) and os.path.isfile(str(tmp_path / "MOCK_densprof.txt"))
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
                   z0=10, inc=60, pa=123.7, mask="SEARCH", free="VROT VDISP", wfunc=2,
                   twostage=False, threads=2, plots=False, verbose=False)
    res = f3d.run(outfolder=str(tmp_path))
    vrot = np.asarray(res.rings['vrot'])
    # C++ (41 rings run): ~80 km/s at 105", ~115 km/s at 285", ~125 km/s at 375"
    assert np.all(np.diff(vrot) > 0) and 60 < vrot[0] < 100 and 110 < vrot[-1] < 135
    assert res.model.data.shape == (62, 150, 150)
    assert set(res.maps) >= {"0mom", "1mom", "mod_0mom_azim"} and "pv_a" in res.pvs


@needs_data
def test_3dfit_galmod_keep_parameters(tmp_path):
    """3DFIT and GALMOD fill the ring parameters that are not given in a copy: the parameters of
    the cube, which the next tasks read, are unchanged (as in C++). Before, a 2DFIT after a 3DFIT
    fitted with DVDZ=0 (its free angle) set by 3DFIT instead of the default 15 degrees."""
    import copy
    from bbarolo.galfit import Galfit
    from bbarolo.tasks import _galmod
    p = Param()
    p.readParamString(f"FITSFILE {N2403}\n3DFIT true\nRADII 105 195 285\nXPOS 77\nYPOS 77\nVSYS 132.8\n"
                      f"VROT 120\nINC 60\nPA 123.7\nFREE VROT VDISP\nMASK SEARCH\nTHREADS 2\nVERBOSE false\n"
                      f"OUTFOLDER {tmp_path}/\n")
    assert p.checkPars(quiet=True)
    c = Cube(N2403, par=p)
    before = copy.deepcopy(c.pars().gf)
    Galfit(c)
    assert vars(c.pars().gf) == vars(before)
    _galmod(c, c.pars(), write=False)
    assert vars(c.pars().gf) == vars(before)


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
    a = open(tmp_path / "cli" / "ngc2403_ellprof.txt").read()
    b = open(tmp_path / "py" / "ngc2403_ellprof.txt").read()
    assert a == b and len(res.profile) == 10
    from astropy.io import fits
    assert np.array_equal(fits.getdata(tmp_path / "cli" / "ngc2403_ellprof_map.fits"), res.map.data, equal_nan=True)


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
                      f"VERBOSE false\nOUTFOLDER {tmp_path}/\n")
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
    head = open(os.path.join(str(tmp_path), "ngc2403_rings_final1.txt")).readline().split()
    assert head[11:] == ["E_VROT1", "E_VROT2", "E_DISP1", "E_DISP2", "E_DENS1", "E_DENS2"]
    assert not f._galfit.par.flagERRORS
    f.write_bestmodel(plots=False, verbose=False)
    assert os.path.isfile(os.path.join(str(tmp_path), "ngc2403_mod_none.fits"))
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
    model, bhi, blo = g._calculate_model(g._update_rings(g._fit0.copy(), th))[:3]
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
        rings = g._update_rings(g._fit0.copy(), th)
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
    model = f._calculate_model(f._update_rings(f._fit0.copy(), truth), fullcube=True)[0]
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
    from bbarolo.bayesian.task import _task_fitter
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
    for name in ("MOCK_rings_final1.txt", "MOCK_densprof.txt", "MOCK_densprof_deconvolved.txt", "MOCK_mod_none.fits",
                 "MOCK_dynesty_results.npy", "MOCK_corner.pdf", "MOCK_profiles.pdf", "MOCK_parameters.pdf"):
        assert (cli / name).is_file(), name
    head = open(cli / "MOCK_rings_final1.txt").readline().split()
    assert head[11:] == ["E_VROT1", "E_VROT2", "E_DISP1", "E_DISP2", "E_DENS1", "E_DENS2"]
    g = BayesianBBarolo(fn, outfolder=str(tmp_path / "re"))
    g.load_results(str(cli / "MOCK_dynesty_results.npy"))
    assert g.freepar_names == ['vrot', 'vdisp'] and 100 < g.params[0] < 200
    assert g.like_kwargs == dict(likemask='dilated', dilation=(1., 3), noisecorr='measured', densprof_smooth=0.1,
                                 bbres=False, radmax='edge', radmax_width=None)
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


# --------------------------------------------------------------------------
# Fixes of November 2026 (CHANGES.md)
# --------------------------------------------------------------------------
def test_nelder_mead_keeps_clamped_point():
    """The simplex of the initial guesses keeps the point clamped by the function, as in C++
    (funcEllipse folds inc > 90 to 180 - inc): the minimum at 89.5 is found, not 90.5."""
    from bbarolo.paramguess import NelderMead

    def f(par):
        if par[1] > 90:
            par[1] = 180 - par[1]
        return (par[0] - 10) ** 2 + 0.01 * (par[1] - 89.5) ** 2
    res = NelderMead().minimize([9.5, 84.], [1., 8.4], f)
    assert abs(res[1] - 89.5) < 0.01 and abs(res[0] - 10) < 0.01


def test_reduce_spatial_crpix(synthetic_cube):
    """Spatial REDUCE: the WCS of a reduced pixel is that of the centre of the pixels it averages."""
    c = synthetic_cube
    h = c.Head()
    h.setCrpix(0, 33.)
    h.setCrpix(1, 33.)
    for fac in (2, 3):
        r = c.Reduce(fac, "spatial").Head()
        for i in (0, 1):
            # reduced pixel 1 averages the original pixels 1..fac, centred at (1 + fac) / 2
            x_old = (1 + fac) / 2.
            assert np.isclose((1 - r.Crpix(i)) * r.Cdelt(i), (x_old - h.Crpix(i)) * h.Cdelt(i))


@needs_data
def test_analytic_dvdz_and_zero_width():
    """ANALYTIC models: a lagging thick disk is sampled finely enough in height (its global
    spectrum agrees with a converged MC model), and VDISP=0 with LINEAR=0 works."""
    from bbarolo import DiskModel
    c = Cube(N2403, par=Param(verbose=False))
    dm = DiskModel(c, smooth=False)
    r = Rings().setRings(radii=np.arange(15, 600, 30), xpos=77, ypos=77, vsys=132.8, vrot=120, vdisp=8, z0=30,
                         inc=80, phi=123.7, dens=1, dvdz=2., zcyl=5.)
    a = dm.array(r, galmodtype="ANALYTIC").sum((1, 2))
    dm.setup.cdens = 400.
    ref = dm.array(r, galmodtype="MC", seed=3, threads=8).sum((1, 2))
    assert np.abs(a - ref).max() / ref.max() < 0.04             # 0.146 before
    c.pars().linear = 0.
    dm = DiskModel(c, smooth=False)
    r = Rings().setRings(radii=np.arange(15, 600, 30), xpos=77, ypos=77, vsys=132.8, vrot=120, vdisp=0, z0=0,
                         inc=60, phi=123.7, dens=1)
    assert np.isclose(dm.array(r, galmodtype="ANALYTIC").sum(), dm.array(r, galmodtype="MC").sum(), rtol=1e-6)


def _freq_cube(path, unit, scale, f1, z):
    from astropy.io import fits
    h = fits.Header()
    for k, v in dict(NAXIS=3, NAXIS1=64, NAXIS2=64, NAXIS3=120, CTYPE1="RA---SIN", CTYPE2="DEC--SIN", CTYPE3="FREQ",
                     CRPIX1=32.5, CRPIX2=32.5, CRPIX3=40, CRVAL1=150., CRVAL2=30., CRVAL3=f1 / (1 + z) / scale,
                     CDELT1=-10 / 3600., CDELT2=10 / 3600., CDELT3=-2e4 / scale, CUNIT1="deg", CUNIT2="deg",
                     CUNIT3=unit, BMAJ=30 / 3600., BMIN=30 / 3600., BPA=0., BUNIT="JY/BEAM", RESTFREQ=f1,
                     OBJECT="FREQ").items():
        h[k] = v
    fits.writeto(path, np.zeros((120, 64, 64), np.float32), h, overwrite=True)
    return path


def test_frequency_axes_lines_and_units(tmp_path):
    """Frequency cubes: several lines at REDSHIFT > 0 sit at their velocities (C++ multiplied the rest
    frequencies by 1+z and the model was empty), and MHz axes give the same model as Hz axes."""
    from bbarolo import DiskModel
    from bbarolo.bbutils import DeltaVel
    f1, z, C = 1.420405751786e9, 0.01, 299792.458
    f2 = f1 * (1 - 300 / C)
    specs = []
    for unit, scale in (("Hz", 1.), ("MHz", 1e6)):
        p = Param()
        p.readParamString(f"RESTFREQ {f1} {f2}\nRELINT 1 1\nREDSHIFT {z}\nVERBOSE false\n")
        c = Cube(_freq_cube(str(tmp_path / f"f{unit}.fits"), unit, scale, f1, z), par=p)
        assert np.isclose(DeltaVel(c.Head()), C * 2e4 / (f1 / (1 + z)), rtol=1e-4)
        dm = DiskModel(c, smooth=False)
        assert np.allclose(dm.setup.lfreq0, np.array([f1, f2]) / (1 + z))
        r = Rings().setRings(radii=np.arange(15, 200, 30), xpos=31.5, ypos=31.5, vsys=0., vrot=100, vdisp=10, z0=0,
                             inc=60, phi=45, dens=1)
        specs.append(dm.array(r, galmodtype="ANALYTIC").sum((1, 2)))
    assert specs[0].sum() > 0 and np.allclose(specs[0], specs[1], rtol=1e-6)


@needs_data
def test_velocity_flux_and_wavelength_units(tmp_path):
    """CUNIT3=kms is km/s (C++ took it as cm/s: empty model), mJy/beam and uJy/beam are scaled to
    Jy/beam, and radio velocities of a wavelength axis have the right channel width."""
    from astropy.io import fits
    from bbarolo import DiskModel
    from bbarolo.bbutils import DeltaVel, FluxtoJyBeam
    d, h0 = fits.getdata(N2403), fits.getheader(N2403)
    assert h0["CUNIT3"].lower() == "m/s"
    r = Rings().setRings(radii=np.arange(15, 600, 30), xpos=77, ypos=77, vsys=132.8, vrot=120, vdisp=8, z0=10, inc=60,
                         phi=123.7, dens=1)
    flux = []
    for cu, sc in (("m/s", 1.), ("kms", 1e-3)):
        h = h0.copy()
        h["CUNIT3"], h["CRVAL3"], h["CDELT3"] = cu, h0["CRVAL3"] * sc, h0["CDELT3"] * sc
        fits.writeto(tmp_path / "u.fits", d, h, overwrite=True)
        flux.append(DiskModel(Cube(str(tmp_path / "u.fits"), par=Param(verbose=False)), smooth=False).array(r).sum())
    assert flux[1] > 0 and np.isclose(flux[0], flux[1])
    for bu, fac in (("JY/BEAM", 1.), ("mJy/beam", 1e-3), ("uJy/beam", 1e-6)):
        h = h0.copy()
        h["BUNIT"] = bu
        fits.writeto(tmp_path / "u.fits", d, h, overwrite=True)
        assert FluxtoJyBeam(1., Cube(str(tmp_path / "u.fits"), par=Param(verbose=False)).Head()) == fac
    h = fits.Header()
    for k, v in dict(NAXIS=3, NAXIS1=8, NAXIS2=8, NAXIS3=50, CTYPE1="RA---SIN", CTYPE2="DEC--SIN", CTYPE3="WAVE",
                     CUNIT1="deg", CUNIT2="deg", CUNIT3="Angstrom", CRPIX3=25, CRVAL3=6563., CDELT3=0.5, CDELT1=-1e-4,
                     CDELT2=1e-4, BMAJ=3e-4, BMIN=3e-4, BUNIT="JY/BEAM").items():
        h[k] = v
    fits.writeto(tmp_path / "w.fits", np.zeros((50, 8, 8), np.float32), h, overwrite=True)
    p = Param(verbose=False)
    p.gf.RESTWAVE = [6563.]
    p.ma.veldef = "radio"
    assert np.isclose(DeltaVel(Cube(str(tmp_path / "w.fits"), par=p).Head()), 299792.458 * 0.5 / 6563., rtol=1e-3)


@needs_data
def test_free_all_two_stages_with_errors(tmp_path):
    """FREE=ALL with two stages and FLAGERRORS: DENS stays free in the second stage and the errors
    have one column per free parameter (C++: degenerate simplex; the port crashed)."""
    from bbarolo import FitMod3D
    res = FitMod3D(N2403, radii=[90, 210, 330], xpos=77, ypos=77, vsys=132.8, vrot=100, vdisp=8, z0=10, inc=60,
                   pa=123.7, free="ALL", mask="SEARCH", flagerrors=True, twostage=True, threads=3, plots=False,
                   verbose=False).run(outfolder=str(tmp_path))
    cols = res.rings.colnames
    assert {"e_vrot1", "e_vdisp1", "e_dens1", "e_vrad1"} <= set(cols) and len(res.rings) == 3


@needs_data
def test_normalcube_restores_data(tmp_path):
    """NORMALCUBE scales the data during a fitting stage only."""
    from bbarolo.galfit import Galfit
    p = Param()
    p.readParamString(f"FITSFILE {N2403}\n3DFIT true\nRADII 150 250\nXPOS 77\nYPOS 77\nVSYS 132.8\nVROT 120\n"
                      f"VDISP 8\nINC 60\nPA 123.7\nZ0 10\nFREE VROT\nNORMALCUBE true\nTHREADS 2\nVERBOSE false\n"
                      f"OUTFOLDER {tmp_path}/\n")
    assert p.checkPars(quiet=True)
    c = Cube(N2403, par=p)
    fit = Galfit(c)
    fit.galfit()
    assert np.array_equal(fit.data_T, c.array.transpose(1, 2, 0))
    assert getattr(fit._tls, "rng", None) is None        # the generator of a ring lives only during its fit


@needs_data
def test_estimates_with_values_from_files(tmp_path):
    """Initial estimates with some ring parameters given as file(...) columns (it raised ValueError)."""
    from bbarolo.galfit import initial_rings
    rf = tmp_path / "rings.txt"
    rf.write_text("# rad vsys vrot pa\n100 133.5 110 124.\n200 133.5 120 124.\n")
    p = Param()
    p.readParamString(f"FITSFILE {N2403}\nVSYS file({rf},2)\nVROT file({rf},3)\nPA file({rf},4)\nTHREADS 2\n"
                      f"VERBOSE false\nOUTFOLDER {tmp_path}/\n")
    assert p.checkPars(quiet=True)
    c = Cube(N2403, par=p)
    r = initial_rings(c, c.pars().gf, write=False).rings
    assert np.allclose([r.vsys[0], r.vrot[0], r.phi[0]], [133.5, 110., 124.])


@needs_data
def test_2dfit_keeps_given_radii(tmp_path):
    """2DFIT with RADII given and the geometry estimated keeps all the radii (they were cut to the
    estimated number of rings)."""
    from bbarolo import FitMod2D
    rm = FitMod2D(N2403, radii=np.arange(15, 1200, 30), vsys=132.8, vrot=120, free="VROT", mask="SEARCH",
                  threads=2, verbose=False)
    radii = np.asarray(rm.run(write=False).rings["radii"])
    assert len(radii) >= 38 and np.allclose(radii, np.arange(15, 1200, 30)[:len(radii)])


def test_detection_widths_with_pbcorr(synthetic_cube, monkeypatch):
    """W20/W50 come from the spectrum of the observed fluxes, also with PBCORR (as in C++): the
    corrected fluxes can be NaN beyond the cutoff."""
    import bbarolo.search as bs
    c = synthetic_cube
    z = np.arange(10, 30)
    a = np.zeros_like(c.array)
    a[z, 32, 32] = np.exp(-0.5 * ((z - 20) / 3.) ** 2)
    det = bs.Detection(z, np.full(z.size, 32), np.full(z.size, 32))
    det.calcIntegFlux(a, c.Head(), pbcorr=0, fluxconvert=False)
    w20, w50 = det.getW20(), det.getW50()
    monkeypatch.setattr(bs, "Pbcor", lambda x, y, z, f, h, opt: np.nan)
    det.calcIntegFlux(a, c.Head(), pbcorr=3, fluxconvert=False)
    assert np.isfinite(w20) and (det.getW20(), det.getW50()) == (w20, w50)


def test_2d_header_spectral_coordinates(tmp_path):
    """A 2D image has no spectral axis: getZphys/getZgrid return the channel (it raised IndexError)."""
    from bbarolo.header import Header
    hdu = emptyfits([32, 32], [-10. / 3600., 10. / 3600.], beam=30. / 3600., bunit="JY/BEAM", obj="IMG")
    hdu.writeto(str(tmp_path / "img.fits"))
    h = Header(str(tmp_path / "img.fits"), warning=False)
    assert h.getZphys(0) == 0. and h.getZgrid(0.) == 0.


def test_rotated_wcs_kept(tmp_path):
    """A rotated CD matrix is kept as PC + CDELT: when the WCS is rebuilt, and in the header of a
    spatially REDUCEd cube (which had the old CD cards next to the new CDELT)."""
    from astropy.io import fits
    from astropy.wcs import WCS
    hdu = emptyfits([40, 40, 8], [-10. / 3600., 10. / 3600., 10.], beam=30. / 3600., bunit="JY/BEAM", obj="ROT")
    h = hdu.header
    th = np.radians(30.)
    for k in ("CDELT1", "CDELT2"):
        del h[k]
    h["CD1_1"], h["CD1_2"] = -10. / 3600. * np.cos(th), -10. / 3600. * np.sin(th)
    h["CD2_1"], h["CD2_2"] = -10. / 3600. * np.sin(th), 10. / 3600. * np.cos(th)
    hdu.data[:] = np.random.default_rng(1).random(hdu.data.shape)
    fn = str(tmp_path / "rot.fits")
    hdu.writeto(fn)
    w0 = WCS(fits.getheader(fn)).celestial
    c = Cube(fn, par=Param(verbose=False))
    pix = np.array([[3., 7., 0.], [25., 31., 0.]])
    before = c.Head().pixToWCS(pix)[:, :2]
    assert np.allclose(before, w0.wcs_pix2world(pix[:, :2], 0), atol=1e-9)
    c.Head().updateWCS()
    assert np.allclose(c.Head().pixToWCS(pix)[:, :2], before, atol=1e-9)
    c.Reduce(2, "spatial").fitswrite_3d(str(tmp_path / "red.fits"), True)
    Cube.fromArray(np.zeros_like(c.array), c.Head().copy(), c.pars()).Reduce(2, "spatial")   # all NaN: no crash
    hr = fits.getheader(str(tmp_path / "red.fits"))
    assert not any(k.startswith("CD") and "_" in k for k in hr)
    # reduced pixel (0, 0) (0-based) is the centre of original pixels 0-1: 0.5
    assert np.allclose(WCS(hr).celestial.wcs_pix2world([[0., 0.]], 0), w0.wcs_pix2world([[0.5, 0.5]], 0), atol=1e-9)


def test_stokes_third_axis(tmp_path):
    """A cube with axes RA, DEC, STOKES, FREQ (no CUNIT3): the first Stokes plane, and a frequency axis
    (C++ took the empty CUNIT3 as km/s)."""
    from astropy.io import fits
    from bbarolo.bbutils import DeltaVel
    data = np.zeros((20, 2, 16, 16), np.float32)          # (FREQ, STOKES, y, x)
    data[:, 0] = 1.
    data[:, 1] = 5.
    h = fits.Header()
    for k, v in dict(NAXIS=4, NAXIS1=16, NAXIS2=16, NAXIS3=2, NAXIS4=20, CTYPE1="RA---SIN", CTYPE2="DEC--SIN",
                     CTYPE3="STOKES", CTYPE4="FREQ", CRPIX1=8, CRPIX2=8, CRPIX3=1, CRPIX4=10, CRVAL1=150., CRVAL2=30.,
                     CRVAL3=1., CRVAL4=1.42e9, CDELT1=-10 / 3600., CDELT2=10 / 3600., CDELT3=1., CDELT4=-2e4,
                     CUNIT1="deg", CUNIT2="deg", CUNIT4="Hz", BMAJ=30 / 3600., BMIN=30 / 3600., BPA=0.,
                     BUNIT="JY/BEAM", RESTFREQ=1.420405751786e9).items():
        h[k] = v
    fits.writeto(str(tmp_path / "s.fits"), data, h)
    c = Cube(str(tmp_path / "s.fits"), par=Param(verbose=False))
    assert c.array.shape == (20, 16, 16) and np.all(c.array == 1.)
    assert c.Head().SpectralType() == "freq" and np.isclose(abs(DeltaVel(c.Head())), 4.22, atol=0.05)


def test_rings_defaults_and_ranges():
    """Rings() starts with radsep 0 (as C++; -1 shrank SHIFTED disks), RADII Rmin~Rmax:step includes
    Rmax also for steps that are not exact in binary, and a NaN position or velocity does not land in
    pixel or channel 0."""
    from bbarolo.galmod import _lround
    from bbarolo.rings import readRings
    assert Rings().radsep == 0.
    p = Param()
    p.readParamString("RADII 0~1:0.1\nXPOS 10\nYPOS 10\nVSYS 0\nVROT 100\nVDISP 10\nINC 60\nPA 30\nZ0 0\n")
    h = emptyfits([32, 32, 10], [-10. / 3600., 10. / 3600., 10.], beam=30. / 3600.)
    from bbarolo.header import Header
    hh = Header()
    hh.fromAstropy(h.header)
    r = readRings(p.gf, hh)
    assert r.nr == 11 and np.isclose(r.radii[-1], 1.)
    assert _lround(np.nan) < -1e15 and _lround(np.inf) < -1e15 and _lround(2.5) == 3 and _lround(-2.5) == -3


@needs_data
def test_analytic_lorentzian_layer():
    """ANALYTIC models with a Lorentzian layer (LTYPE 4) keep only the part of the layer that can fall
    in the field, as MC models do (the far tail was put closer to the plane: +1.4% of flux)."""
    from bbarolo import DiskModel
    p = Param(verbose=False)
    p.gf.LTYPE = p.gm.LTYPE = 4
    dm = DiskModel(Cube(N2403, par=p), smooth=False)
    r = Rings().setRings(radii=np.arange(15, 600, 30), xpos=77, ypos=77, vsys=132.8, vrot=120, vdisp=8, z0=30,
                         inc=80, phi=123.7, dens=1)
    a = dm.array(r, galmodtype="ANALYTIC").sum()
    dm.setup.cdens = 200.
    m = dm.array(r, galmodtype="MC", seed=3, threads=8).sum()
    assert abs(a / m - 1) < 0.008                             # 0.0138 before


def test_specres_keyword(synthetic_cube, capsys):
    """SPECRES sets the spectral resolution of the models; LINEAR still does when SPECRES is not given
    (as in C++). LINEAR is no longer a resolution (kpc) of SMOOTH."""
    from bbarolo import DiskModel
    c = synthetic_cube
    r = _rings()
    models = {}
    for k, v in (("default", None), ("linear", ("linear", 2.)), ("specres", ("specres", 2.))):
        c.pars().linear, c.pars().specres = -1., -1.
        if v:
            setattr(c.pars(), v[0], v[1])
        models[k] = DiskModel(c, smooth=False).array(r, galmodtype="ANALYTIC")
    assert np.array_equal(models["linear"], models["specres"]) and not np.allclose(models["default"], models["linear"])
    p = Param()
    from bbarolo.keywords import lookup
    p.readParamString(f"FITSFILE {N2403}\nSMOOTH true\nLINEAR 0.5\nGALMOD true\nRADII 100 200\n"
                      "XPOS 77\nYPOS 77\nVSYS 130\nVROT 100\nVDISP 8\nINC 60\nPA 120\nZ0 0\n")
    assert p.checkPars() is not False
    assert "LINEAR" not in capsys.readouterr().out and "SMOOTH" not in lookup("linear").tasks



def test_ring_files_in_arcsec(tmp_path, capsys):
    """No DISTANCE: the ring files have the radii and Z0 in arcsec only (C++ also wrote RAD(Kpc), first, and
    Z0(pc)). RINGFILE finds the columns by their names, in the files of bbarolo2 and of C++, and
    file(name,n) of a ring file must read the column of the parameter (old column numbers are an error)."""
    import io
    from bbarolo.galfit import writeHeader, writeRing, ringfile_values, RING_COLUMNS
    from bbarolo.params import GalfitPar
    from bbarolo.rings import readRings
    from bbarolo.bbutils import BBaroloError, getDataColumn
    from bbarolo.keywords import lookup, UnknownParameterError
    assert lookup("distance") is None
    with pytest.raises(UnknownParameterError):
        Param().set("distance", 3.2)
    Param().readParamString("DISTANCE 3.2\n")                 # an old parameter file: ignored, with a warning
    assert "DISTANCE" in capsys.readouterr().err.upper()
    r = _rings(3)
    r.vrot = np.array([100., 120., 130.])
    f = io.StringIO()
    writeHeader(f, [False] * 10, False, False)
    for i in range(r.nr):
        writeRing(f, r, i, 0, False, None, False, False)
    new = tmp_path / "rings_new.txt"
    new.write_text(f.getvalue())
    assert f.getvalue().split("\n")[0].lstrip("#").split() == list(RING_COLUMNS)
    cpp = tmp_path / "rings_cpp.txt"
    cpp.write_text("#RAD(Kpc) RAD(arcs)  VROT(km/s) DISP(km/s) INC(deg)  P.A.(deg) Z0(pc)    Z0(arcs)  DENS(un)  "
                   "XPOS(pix) YPOS(pix) VSYS(km/s) VRAD(km/s)\n" + "".join(
                       f"{R * 0.0155:.3f} {R:.3f} {v:.3f} 10.0 60.0 45.0 0.0 0.0 1.0e+00 31.5 31.5 0.0 0.0\n"
                       for R, v in zip(r.radii, r.vrot)))
    for fn, col in ((new, 2), (cpp, 3)):
        par = GalfitPar()
        par.ringfile = str(fn)
        ringfile_values(par)
        assert par.VROT == f"file({fn},{col})" and getDataColumn(par.VROT) == [100., 120., 130.]
        assert getDataColumn(par.RADII) == list(r.radii) and getDataColumn(par.INC) == [60.] * 3
    par = GalfitPar()
    par.RADII, par.VROT = f"file({cpp},2)", f"file({cpp},2)"
    with pytest.raises(BBaroloError, match="column 2 .* is RAD\\(arcs\\); VROT is column 3"):
        readRings(par, None)
    par.RADII, par.VROT = f"file({new},1)", f"file({new},3)"
    with pytest.raises(BBaroloError, match="DISP\\(km/s\\); VROT is column 2"):
        readRings(par, None)

_C, _F0, _W0, _NZ, _DV, _V0 = 299792.458, 1.420405751786e9, 6562.8, 60, 10., 1000.
_b = _V0 / _C
# (name, CTYPE3, CUNIT3, CRVAL3, CDELT3, extra cards, VELDEF, RESTWAVE, expected DeltaVel)
_AXES = [(f"VRAD {u}", "VRAD", u, _V0 * s, _DV * s, {}, "radio", None, _DV) for u, s in
         (("m/s", 1e3), ("km/s", 1.), ("kms", 1.))] + \
        [(f"FREQ {u}", "FREQ", u, _F0 * (1 - _b) / s, -_F0 * _DV / _C / s, dict(RESTFRQ=_F0), "radio", None, _DV)
         for u, s in (("Hz", 1.), ("kHz", 1e3), ("MHz", 1e6), ("GHz", 1e9))] + \
        [(f"WAVE {u}", "WAVE", u, _W0 * (1 + _b) / s, _W0 * _DV / _C / s, {}, "optical", _W0 / s, _DV)
         for u, s in (("Angstrom", 1.), ("nm", 10.), ("um", 1e4))] + \
        [("FREQ optical", "FREQ", "Hz", _F0 / (1 + _b), -_F0 * _DV / _C, dict(RESTFRQ=_F0), "optical", None,
          _DV * (1 + _b) ** 2),
         ("WAVE radio", "WAVE", "Angstrom", _W0 / (1 - _b), _W0 * _DV / _C, {}, "radio", _W0, _DV * (1 - _b) ** 2),
         ("FREQ relativistic", "FREQ", "Hz", _F0 * np.sqrt((1 - _b) / (1 + _b)), -_F0 * _DV / _C,
          dict(RESTFRQ=_F0), "relativistic", None, None),
         ("WAVE relativistic", "WAVE", "Angstrom", _W0 * np.sqrt((1 + _b) / (1 - _b)), _W0 * _DV / _C, {},
          "relativistic", _W0, None)]


@pytest.mark.parametrize("axis", _AXES, ids=[a[0] for a in _AXES])
def test_spectral_axes(axis, tmp_path):
    """The same galaxy on spectral axes of every type, unit and velocity definition: the velocity of the
    reference channel, the channel width, channel <-> spectral value round trips, and a model centred on
    VSYS (kHz and nm axes were wrong)."""
    from astropy.io import fits
    from bbarolo import DiskModel
    from bbarolo.bbutils import DeltaVel, AlltoVel, Vel2Spec
    name, ct, cu, crval, cdelt, extra, veldef, restw, dv = axis
    h = fits.Header()
    for k, v in dict(NAXIS=3, NAXIS1=48, NAXIS2=48, NAXIS3=_NZ, CTYPE1="RA---SIN", CTYPE2="DEC--SIN", CTYPE3=ct,
                     CRPIX1=24.5, CRPIX2=24.5, CRPIX3=_NZ // 2, CRVAL1=150., CRVAL2=30., CRVAL3=crval,
                     CDELT1=-10 / 3600., CDELT2=10 / 3600., CDELT3=cdelt, CUNIT1="deg", CUNIT2="deg", CUNIT3=cu,
                     BMAJ=30 / 3600., BMIN=30 / 3600., BPA=0., BUNIT="JY/BEAM", OBJECT="AX", **extra).items():
        h[k] = v
    fits.writeto(str(tmp_path / "ax.fits"), np.zeros((_NZ, 48, 48), np.float32), h)
    p = Param(verbose=False)
    p.ma.veldef = veldef
    if restw:
        p.gf.RESTWAVE = p.gm.RESTWAVE = [restw]
    c = Cube(str(tmp_path / "ax.fits"), par=p)
    hh = c.Head()
    v = np.array([AlltoVel(hh.getZphys(z), hh) for z in range(_NZ)])
    assert np.isclose(v[_NZ // 2 - 1], _V0, atol=1e-3)
    if dv is not None:
        assert np.isclose(abs(DeltaVel(hh)), dv, rtol=2e-3)
    assert np.allclose([hh.getZgrid(Vel2Spec(x, hh)) for x in v], np.arange(_NZ), atol=1e-6)
    r = Rings().setRings(radii=np.arange(15, 200, 30), xpos=23.5, ypos=23.5, vsys=_V0, vrot=100, vdisp=10, z0=0,
                         inc=60, phi=45, dens=1)
    spec = DiskModel(c, smooth=False).array(r, galmodtype="ANALYTIC").sum((1, 2))
    assert spec.sum() > 0 and abs((spec * v).sum() / spec.sum() - _V0) < 0.5


def test_parser_non_finite_integers():
    """inf or nan given to an integer parameter is unreadable (it raised OverflowError/ValueError)."""
    p = Param()
    p.readParamString("FITSFILE x.fits\nBOX inf 1 2\nCDENS nan\nNV inf\n")
    assert p.BOX[0] == -1 and p.gf.CDENS == 0 and p.checkPars(quiet=True)


def test_xpos_from_file_keeps_path_case():
    """XPOS/YPOS are read in lower case (WCS values), but a file(...) path keeps its case (C++ lowercased
    it: ring files in folders with capital letters were not found)."""
    p = Param()
    p.readParamString("XPOS file(/Data/Rings.txt,10)\nYPOS 07H36M51S\n")
    assert p.gf.XPOS == "file(/Data/Rings.txt,10)" and p.gf.YPOS == "07h36m51s"


@needs_data
def test_relative_paths(tmp_path, monkeypatch):
    """FITSFILE and OUTFOLDER relative to the current folder (e.g. ../../Data/HI/cube.fits) become
    absolute paths: they are written in the plotting scripts, which failed when run from another
    folder (plot_all.py)."""
    import shutil
    from bbarolo import Ellprof
    (tmp_path / "Data" / "HI").mkdir(parents=True)
    shutil.copy(N2403, tmp_path / "Data" / "HI" / "g.fits")
    (tmp_path / "a" / "b").mkdir(parents=True)
    monkeypatch.chdir(tmp_path / "a" / "b")
    p = Param()
    p.readParamString("FITSFILE ../../Data/HI/g.fits\nOUTFOLDER out\n")
    assert p.checkPars(quiet=True)
    assert p.imageFile == str(tmp_path / "Data" / "HI" / "g.fits") and p.outFolder == str(tmp_path / "a" / "b" / "out") + "/"
    res = Ellprof("../../Data/HI/g.fits", nradii=5, radsep=60, xpos=77, ypos=77, inc=60, pa=123.7, verbose=False).run(
        outfolder="out")
    assert len(res.profile) == 5 and all(os.path.isabs(f) for f in res.files)


def test_fft_kernel_cache_is_bounded():
    """The cache of kernel transforms keeps the most recent shapes only (it grew with every new box
    size, e.g. in BayesianBBarolo with a free centre or RADMAX), and gives the same convolutions."""
    from bbarolo.smooth import FFTConvolver
    rng = np.random.default_rng(2)
    cv = FFTConvolver(rng.random((7, 7)))
    a = rng.random((3, 20, 24))
    ref = cv(a)
    for n in range(10, 10 + 3 * FFTConvolver.MAXCACHE):
        cv(rng.random((n, n)))
    assert len(cv._cache) == FFTConvolver.MAXCACHE and np.array_equal(cv(a), ref)


@needs_data
def test_bayesian_default_analytic(tmp_path):
    """BayesianBBarolo and BAYESIAN3DFIT use ANALYTIC models by default; a GALMODTYPE that is given is
    kept. 3DFIT keeps MC."""
    from bbarolo import FitMod3D
    from bbarolo.bayesian import BayesianBBarolo
    kw = dict(radii=[100, 200], xpos=77, ypos=77, vsys=132.8, vrot=100, vdisp=10, inc=60, pa=123.7, z0=10,
              outfolder=str(tmp_path) + "/")
    assert BayesianBBarolo(N2403, **kw)._make_par().galmodtype == "ANALYTIC"
    assert BayesianBBarolo(N2403, galmodtype="MC", **kw)._make_par().galmodtype == "MC"
    assert FitMod3D(N2403, **kw).par.galmodtype == "MC"
    par = tmp_path / "b.par"
    par.write_text(f"FITSFILE {N2403}\nBAYESIAN3DFIT true\nFREE VROT\nRADII 100 200\nXPOS 77\nYPOS 77\nVSYS 132.8\n"
                   "VROT 100\nVDISP 10\nINC 60\nPA 123.7\nZ0 10\n")
    assert BayesianBBarolo.from_parfile(str(par))._make_par().galmodtype == "ANALYTIC"
    par.write_text(par.read_text() + "GALMODTYPE MC\n")
    assert BayesianBBarolo.from_parfile(str(par))._make_par().galmodtype == "MC"


def test_errors_and_exit_codes(tmp_path):
    """Errors in the input raise BBaroloError (a ValueError) with a clear message, never SystemExit; the
    command line prints it and exits with 1 (it exited with 0 after unacceptable parameters, and ran
    the task after an unknown option)."""
    from bbarolo import FitMod3D, Smooth, PVSlice, BBaroloError
    nob = emptyfits([32, 32, 10], [-10. / 3600., 10. / 3600., 10.], bunit="JY/BEAM", obj="NOBEAM")
    nob.data[:] = 1.
    fn = str(tmp_path / "nobeam.fits")
    nob.writeto(fn)
    with pytest.raises(BBaroloError, match="No beam"):
        FitMod3D(fn, radii=[10, 20], xpos=15, ypos=15, vsys=0, vrot=50, vdisp=8, inc=60, pa=30, z0=0,
                 verbose=False).run(write=False)
    beam = emptyfits([32, 32, 10], [-10. / 3600., 10. / 3600., 10.], beam=30. / 3600., bunit="JY/BEAM", obj="B")
    beam.data[:] = 1.
    fb = str(tmp_path / "beam.fits")
    beam.writeto(fb)
    with pytest.raises(BBaroloError, match="smaller than the old beam"):
        Smooth(fb, bmaj=10, bmin=10, verbose=False).run(write=False)
    with pytest.raises(BBaroloError, match="outside the cube"):
        PVSlice(fb, xpos_pv=500, ypos_pv=500, pa_pv=30, verbose=False).run(write=False)
    cli = [sys.executable, os.path.join(ROOT, "run_bbarolo.py")]
    run = lambda *a: subprocess.run(cli + list(a), capture_output=True, text=True, cwd=str(tmp_path))
    r = run("-p", "nofile.par")
    assert r.returncode == 1 and "Could not open the parameter file" in r.stderr
    r = run("-c", f"fitsfile={fb}", "3dfit=true", "norm=GLOBAL", "radii=10", "xpos=15", "ypos=15", "vsys=0",
            "vrot=50", "vdisp=8", "inc=60", "pa=30", "z0=0")
    assert r.returncode == 1 and "Unacceptable input parameters" in r.stderr
    r = run("-f", fb, "--bogus")
    assert r.returncode == 1 and "--bogus is unacceptable" in r.stderr and not (tmp_path / "output").exists()
    r = run("-c", f"fitsfile={fn}", "3dfit=true", "radii=10", "xpos=15", "ypos=15", "vsys=0", "vrot=50", "vdisp=8",
            "inc=60", "pa=30", "z0=0")
    assert r.returncode == 1 and "No beam" in r.stderr and "Traceback" not in r.stderr
    assert run("-v").returncode == 0


def test_empty_ring_table(tmp_path):
    """A ring table without rows (no ring fitted) keeps its columns (Result.rings['vrot'] was a KeyError)."""
    from bbarolo.tasks import read_table
    from bbarolo.galfit import writeHeader
    with open(tmp_path / "r.txt", "w") as f:
        writeHeader(f, [True] * 2 + [False] * 8, False, False)
    t = read_table(str(tmp_path / "r.txt"))
    assert len(t) == 0 and {"radii", "vrot", "vdisp", "inc", "pa"} <= set(t.colnames)


def test_input_checks(tmp_path, capsys):
    """Checks before running: misspelled keywords, ring values, FREE names, VSYS in the band, beams in
    degrees, missing CUNIT3, sexagesimal centres."""
    from astropy.io import fits
    from bbarolo import FitMod3D, BBaroloError
    from bbarolo.header import Header
    from bbarolo.bbutils import getCenterCoordinates
    # a typo of a keyword is an error, another unknown keyword a warning
    with pytest.raises(BBaroloError, match="VDISP"):
        Param().readParamString("VDSIP 10\n")
    Param().readParamString("NOTAKEYWORD 1\n")
    assert "It is ignored" in capsys.readouterr().err
    # ring values and FREE, when the task is built
    hdu = emptyfits([32, 32, 20], [-10. / 3600., 10. / 3600., 10.], beam=30. / 3600., bunit="JY/BEAM", obj="CHK")
    hdu.data[:] = 1.
    fn = str(tmp_path / "c.fits")
    hdu.writeto(fn)
    kw = dict(radii=[10, 20], xpos=15, ypos=15, vsys=0, vrot=50, vdisp=8, inc=60, pa=30, z0=0, mask="NONE",
              verbose=False)
    with pytest.raises(BBaroloError, match="INC='sixty' is not valid"):
        FitMod3D(fn, **dict(kw, inc="sixty"))
    with pytest.raises(BBaroloError, match="unknown free parameter.*FOO"):
        FitMod3D(fn, free="VROT FOO", **kw)
    p = Param()
    p.readParamString(f"FITSFILE {fn}\n3DFIT true\nRADII 10 20\nXPOS 15\nYPOS 15\nVSYS 0\nVROT 50\nVDISP 8\n"
                      "INC sixty\nPA 30\nZ0 0\nFREE VROT\n")
    assert not p.checkPars(quiet=True)
    # VSYS outside the band (given in m/s)
    with pytest.raises(BBaroloError, match="is it in m/s"):
        FitMod3D(fn, **dict(kw, vsys=50000)).run(write=False)
    # a beam given in degrees instead of arcsec
    del hdu.header["BMAJ"], hdu.header["BMIN"], hdu.header["BPA"]
    hdu.writeto(str(tmp_path / "nob.fits"))
    with pytest.raises(BBaroloError, match="smaller than a quarter of a pixel"):
        FitMod3D(str(tmp_path / "nob.fits"), bmaj=0.0083, bmin=0.0083, **kw).run(write=False)
    # missing CUNIT3: m/s if the channels are wider than 100
    for cdelt, unit in ((5000., "m/s"), (5., "km/s")):
        h = fits.Header()
        for k, v in dict(NAXIS=3, NAXIS1=8, NAXIS2=8, NAXIS3=10, CTYPE1="RA---SIN", CTYPE2="DEC--SIN", CTYPE3="VRAD",
                         CRPIX1=4, CRPIX2=4, CRPIX3=5, CRVAL1=150., CRVAL2=30., CRVAL3=0., CDELT1=-1e-3, CDELT2=1e-3,
                         CDELT3=cdelt, CUNIT1="deg", CUNIT2="deg").items():
            h[k] = v
        hh = Header()
        hh.warning = False
        hh.fromAstropy(h)
        assert hh.Cunit(2) == unit
    # sexagesimal centres (07h36m51s was taken as pixels, +65d35m as 65 deg)
    if os.path.isfile(N2403):
        h2 = Header(N2403, warning=False)
        a = getCenterCoordinates(["07h36m51s", "+65d35m"], h2)
        b = getCenterCoordinates([f"{(7 + 36 / 60 + 51 / 3600) * 15}d", f"{65 + 35 / 60}d"], h2)
        assert np.allclose(a, b)


@needs_data
def test_verbosity_levels(tmp_path, capsys):
    """VERBOSE 0: nothing; 1 (default): one line per ring, in order of radius; 2: every ring in detail, as
    in C++."""
    from bbarolo import FitMod3D
    kw = dict(radii=[150, 250, 350], xpos=77, ypos=77, vsys=132.8, vrot=120, vdisp=8, inc=60, pa=123.7, z0=10,
              free="VROT VDISP", twostage=False, threads=3, plots=False)
    out = {}
    for v in (0, 1, 2):
        FitMod3D(N2403, verbose=v, **kw).run(outfolder=str(tmp_path / f"v{v}"))
        out[v] = capsys.readouterr().out
    assert "Best parameters" not in out[0] and "outcome" not in out[0] and len(out[0].splitlines()) < 5
    lines = [ln.split() for ln in out[1].splitlines() if ln.strip().endswith(" ok")]
    assert [int(x[0]) for x in lines] == [1, 2, 3] and "Best parameters" not in out[1]
    assert out[2].count("Best parameters for ring") == 3 and "outcome" not in out[2]


@needs_data
def test_result_data_residual_curve_plots(tmp_path):
    """Result of 3DFIT: data and residual (computed when used), the rotation curve with positive errors,
    and the plotting methods (matplotlib Figures)."""
    import matplotlib
    matplotlib.use("Agg")
    from bbarolo import FitMod3D, GalMod
    kw = dict(radii=[150, 250, 350], xpos=77, ypos=77, vsys=132.8, vrot=120, vdisp=8, inc=60, pa=123.7, z0=10,
              twostage=False, threads=3, plots=False, verbose=0)
    res = FitMod3D(N2403, free="VROT VDISP", flagerrors=True, **kw).run(write=False)
    assert "data" in res and "data" in res._lazy                 # not computed yet
    assert res.data.data.shape == res.model.data.shape == res.residual.data.shape
    m = np.asarray(res.mask.data, bool)
    assert np.all(np.isnan(res.residual.data[~m]))
    assert np.allclose(res.residual.data[m], (res.data.data - res.model.data)[m])
    rc = res.rotation_curve
    assert rc.colnames == ["radii", "vrot", "e_vrot_lo", "e_vrot_hi"] and np.all(rc["e_vrot_lo"] >= 0)
    for f in (res.plot_rings(), res.plot_maps(), res.plot_pv()):
        assert isinstance(f, matplotlib.figure.Figure)
    g = GalMod(N2403, norm="AZIM", **kw).run(write=False)
    assert g.residual.data.shape == g.model.data.shape and "rotation_curve" not in g


def _mock_disk(tmp_path, inc=70., pa=40., snr=20., seed=3):
    """A simulated disk (5 beams across, as in performance/run_suite.py) with noise; returns the file."""
    from bbarolo import GalMod
    n, beam, pix, dv = 41, 48., 12., 10.
    tmpl = emptyfits([n, n, 60], [-pix / 3600., pix / 3600., dv], beam=beam / 3600., bunit="JY/BEAM", obj="MOCKD")
    rs = (np.arange(24) + 0.5) * 10.
    gm = GalMod(tmpl, radii=rs, xpos=20, ypos=20, vsys=0., vrot=150 * 2 / np.pi * np.arctan(rs / 40.), vdisp=12.,
                z0=0., inc=inc, pa=pa, dens=np.exp(-rs / 90.), norm="NONE", mask="NONE", galmodtype="ANALYTIC",
                threads=1, verbose=False)
    clean = gm.model()
    noisy = gm.diskmodel.addnoise(clean.data, float(clean.data.max()) / snr, seed=seed)
    from astropy.io import fits
    fn = str(tmp_path / "mockdisk.fits")
    fits.PrimaryHDU(data=noisy, header=tmpl.header).writeto(fn, overwrite=True)
    return fn


@pytest.mark.parametrize("guesstype", ["ELLIPSE", "FIT"])
def test_initial_estimates(tmp_path, guesstype):
    """GUESSTYPE ELLIPSE (the C++ algorithm, corrected) and FIT (beam-convolved disk fitted to the map, 2D
    fit of the velocity field) on a simulated disk at INC 70; given values stay as given."""
    from bbarolo.paramguess import EstimateInitial
    fn = _mock_disk(tmp_path)
    tol = dict(ELLIPSE=6., FIT=3.)[guesstype]
    p = Param()
    p.readParamString(f"FITSFILE {fn}\nGUESSTYPE {guesstype}\nTHREADS 1\nVERBOSE false\nOUTFOLDER {tmp_path}/\n")
    assert p.checkPars(quiet=True)
    ip = EstimateInitial(Cube(fn, par=p), p.gf)
    assert abs(ip.inclin - 70) < tol and abs((ip.posang - 40 + 180) % 360 - 180) < 3
    assert np.hypot(ip.xcentre - 20, ip.ycentre - 20) < 1. and abs(ip.vsystem) < 5 and 100 < ip.vrot < 200
    if guesstype == "FIT":
        assert 6 < ip.vdisp < 20 and len(ip.rad_profile) >= 2
    p.readParamString("INC 65\nVSYS 1\n")
    p.checkPars(quiet=True)
    ip = EstimateInitial(Cube(fn, par=p), p.gf)
    assert ip.inclin == 65 and ip.vsystem == 1


def test_spatial_axes_in_arcsec(tmp_path):
    """Spatial axes in arcsec or arcmin (BMAJ in degrees, as in FITS) are converted to degrees on
    reading: the same results as in degrees (C++ mixed the units: the beam area, the estimated ring
    width (0.008 arcsec: 3DFIT stopped), the region of the Bayesian likelihood)."""
    from astropy.io import fits
    import bbarolo
    from bbarolo.bayesian import BayesianBBarolo
    fn = _mock_cube(tmp_path, "deg")
    h, d = fits.getheader(fn), fits.getdata(fn)
    for name, f, u in (("arcs", 3600., "arcsec"), ("arcm", 60., "arcmin")):
        h2 = h.copy()
        for k in ("CDELT1", "CDELT2", "CRVAL1", "CRVAL2"):
            h2[k] = h[k] * f
        h2["CUNIT1"] = h2["CUNIT2"] = u
        fits.PrimaryHDU(d, header=h2).writeto(str(tmp_path / f"{name}.fits"))
    out = {}
    for name in ("deg", "arcs", "arcm"):
        fn = str(tmp_path / f"{name}.fits")
        hd = Cube(fn, par=Param()).Head()
        r = bbarolo.FitMod3D(fn, free="vrot inc", mask="SEARCH", galmodtype="ANALYTIC", outfolder=str(tmp_path / name),
                             verbose=0, plots=0, twostage=False, threads=1).run()
        mh = fits.getheader(str(tmp_path / name / "MOCK_mod_azim.fits"))
        f = BayesianBBarolo(fn, radii=[20, 60, 100, 140], xpos=31.5, ypos=31.5, vsys=0, vrot=120, vdisp=15, z0=0, inc=60,
                            pa=45, mask="SEARCH", outfolder=str(tmp_path / ("b" + name)))
        f._setup(['vrot', 'vdisp_single'])
        out[name] = (hd.Cdelt(0), hd.Crval(0), hd.BeamArea(), mh["CUNIT1"], mh["CDELT1"], f.beam_area, f.nvox,
                     f._log_likelihood(np.array([140., 150, 150, 150, 10.])), np.asarray(r.rings["radii"]),
                     np.asarray(r.rings["vrot"]), np.asarray(r.rings["inc"]))
    for name in ("arcs", "arcm"):
        for a, b in zip(out[name], out["deg"]):
            assert np.allclose(a, b, rtol=1e-9, atol=0) if not isinstance(a, str) else a == b == "deg"


def _bayes_mock(tmp_path, **kw):
    from bbarolo.bayesian import BayesianBBarolo
    fn = str(tmp_path / "mock.fits")
    if not os.path.isfile(fn):
        _mock_cube(tmp_path)
    prm = dict(radii=[20, 60, 100, 140], xpos=31.5, ypos=31.5, vsys=0, vrot=120, vdisp=15, z0=0, inc=60, pa=45,
               mask="SEARCH", outfolder=str(tmp_path / "bbb"))
    return BayesianBBarolo(fn, **{**prm, **kw})


def test_bayesian_rings_do_not_carry_over(tmp_path):
    """The likelihood evaluations of a run do not change the initial rings: the parameters fixed in a
    second run (and the rings saved with the results) are the initial ones, not the last sample."""
    pytest.importorskip("dynesty")
    f = _bayes_mock(tmp_path)
    r0 = {k: np.array(v) for k, v in f._inri.r.items()}
    f._setup(['vrot', 'inc_single', 'radmax'])
    f._log_likelihood(np.array([150., 150, 150, 150, 50., 135.]))
    assert all(np.array_equal(f._inri.r[k], r0[k]) for k in r0)
    run = dict(threads=1, verbose=False, dynamic=False, sampler_kwargs=dict(nlive=15, seed=1),
               run_kwargs=dict(maxiter=60))
    f.run(freepar=['vrot_single'], **run)
    f.run(freepar=['vdisp_single'], **run)
    assert np.allclose(f.outri.r['vrot'], 120.) and np.allclose(f._inri.r['vrot'], 120.)
    f.save_results(str(tmp_path / "res.npy"))
    assert np.allclose(np.load(str(tmp_path / "res.npy"), allow_pickle=True).item()['inrings']['vrot'], 120.)


def test_bayesian_priors_and_options(tmp_path):
    """A prior given as a distribution survives update(); the slope of a parametric function may be
    negative (only the values of the rings must be positive); unknown sampler and run options are
    errors; dlogz works with the dynamic sampler of dynesty (as dlogz_init); one ring with lambdas."""
    pytest.importorskip("dynesty")
    from scipy import stats
    from bbarolo.bbutils import BBaroloError
    f = _bayes_mock(tmp_path)
    d = stats.norm(150., 10.)
    f.prior_distr['vrot'] = d
    f.update(vdisp=12.)
    f._setup(['vrot_single'])
    assert f.prior_distr['vrot'] is d
    f._setup(['vdisp_func'])                                   # p1 + p2 R
    assert np.isfinite(f._log_likelihood(np.array([20., -0.05])))
    assert f._log_likelihood(np.array([5., -0.1])) == -np.inf      # negative at the outer rings
    run = dict(threads=1, verbose=False, sampler_kwargs=dict(nlive=15, seed=1))
    with pytest.raises(BBaroloError, match="RUNOPTS.*dlogzz"):
        f.run(freepar=['vrot_single'], run_kwargs=dict(dlogzz=1.), **run)
    with pytest.raises(BBaroloError, match="SAMPLEROPTS.*nlivee"):
        f.run(freepar=['vrot_single'], threads=1, verbose=False, sampler_kwargs=dict(nlivee=15))
    f.run(freepar=['vrot_single'], dynamic=True, run_kwargs=dict(dlogz=50., maxiter=60), **run)
    assert np.isfinite(f.params[0])
    g = _bayes_mock(tmp_path, radii=[60.], outfolder=str(tmp_path / "one"))
    g._setup(['vrot'])
    assert np.isfinite(g._log_likelihood(np.array([150.]), lambdas=dict(vrot=1.)))


def test_bayesian_pool_and_loaded_likelihood(tmp_path):
    """A sampling that fails leaves no worker processes; load_results restores the likelihood of the
    fit (run_geometry: the total intensity map; RADMAX of the last ring for files saved before the
    option radmax); the queue of an MPI pool follows its size."""
    pytest.importorskip("dynesty")
    import multiprocessing as mp
    from bbarolo.bayesian import sampling
    f = _bayes_mock(tmp_path)

    def bad(theta, **kw):
        raise RuntimeError("failed likelihood")
    with pytest.raises(RuntimeError, match="failed likelihood"):
        f.run(freepar=['vrot_single'], threads=2, log_like=bad, verbose=False, sampler_kwargs=dict(nlive=15, seed=1))
    assert not mp.active_children() and not sampling._WORKER_REGISTRY and 'parent' not in sampling._WORKER
    f.run_geometry(freepar=['inc_single'], threads=1, verbose=False, dynamic=False,
                   sampler_kwargs=dict(nlive=15, seed=1), run_kwargs=dict(maxiter=60))
    f.save_results(str(tmp_path / "geo.npy"))
    g = _bayes_mock(tmp_path, outfolder=str(tmp_path / "re"))
    g.load_results(str(tmp_path / "geo.npy"))
    assert g._log_like_func == g._log_likelihood_geometry
    assert np.isclose(g._log_like_func(g.params), f._log_like_func(f.params))
    assert g.like_kwargs['radmax'] == 'edge'
    r = np.load(str(tmp_path / "geo.npy"), allow_pickle=True).item()
    del r['like_kwargs']['radmax']
    np.save(str(tmp_path / "old.npy"), r)
    g.load_results(str(tmp_path / "old.npy"))
    assert g.like_kwargs['radmax'] == 'ring' and g._radmax_mode == 'ring'

    class FakeMPIPool:
        size = 7
    assert sampling._queue_size(FakeMPIPool(), 1) == 28 and sampling._queue_size(None, 3) == 12


def test_bayesian_one_blas_thread(tmp_path):
    """The sampling (serial and in the worker processes) runs with one BLAS thread: the linear
    algebra of the likelihood is on small matrices, and many threads made it 1.6-3x slower."""
    pytest.importorskip("dynesty")
    threadpoolctl = pytest.importorskip("threadpoolctl")
    if not any(d["user_api"] == "blas" for d in threadpoolctl.threadpool_info()):
        pytest.skip("no BLAS library found by threadpoolctl")
    f = _bayes_mock(tmp_path)

    def check(theta, **kw):                       # an unusable likelihood if BLAS has more threads
        n = max(d["num_threads"] for d in threadpoolctl.threadpool_info() if d["user_api"] == "blas")
        return -0.5 * ((theta[0] - 150.) / 10.) ** 2 if n == 1 else -1e10
    for threads in (1, 2):
        f.run(freepar=['vrot_single'], threads=threads, log_like=check, verbose=False, dynamic=False,
              sampler_kwargs=dict(nlive=15, seed=1), run_kwargs=dict(maxiter=40))
        assert np.all(f.results.logl > -1e9)


def test_bayesian_geometry_model_error(tmp_path):
    """run_geometry accepts a model error: 'fracerr' and 'sigma' (free or given) in the likelihood of
    the total intensity map, with the normalisation of the Gaussian; without them the values do not
    change."""
    pytest.importorskip("dynesty")
    f = _bayes_mock(tmp_path)
    f._setup(['inc_single', 'fracerr', 'sigma'])
    g = _bayes_mock(tmp_path, outfolder=str(tmp_path / "g"))
    g._setup(['inc_single'])
    L0 = g._log_likelihood_geometry(np.array([55.]))
    assert L0 == g._log_likelihood_geometry(np.array([55.]), fracerr=0.)
    assert np.isclose(f._log_likelihood_geometry(np.array([55., 0., f.noiserms])), L0, rtol=1e-12)
    # by hand, with the model map of the deconvolved profile
    rings = f._update_rings(f._fit0.copy(), np.array([55., 0.1, 2. * f.noiserms]))
    f._set_density(rings)
    model, bhi, blo = f._calculate_model(rings)[:3]
    Lb, Hb = f._likebox
    d, s = f.data_mom0[Lb[1]:Hb[1], Lb[0]:Hb[0]], f.map_rms[Lb[1]:Hb[1], Lb[0]:Hb[0]]
    m = np.zeros_like(d)
    m[blo[1] - Lb[1]:bhi[1] - Lb[1], blo[0] - Lb[0]:bhi[0] - Lb[0]] = model.sum(axis=0)
    ok = (s > 0) & f._region[Lb[1]:Hb[1], Lb[0]:Hb[0]]
    v = 4. * s[ok] ** 2 + (0.1 * m[ok]) ** 2
    expect = -f.likefactor2d * 0.5 * np.sum((d[ok] - m[ok]) ** 2 / v + np.log(v / s[ok] ** 2))
    assert np.isclose(f._log_likelihood_geometry(np.array([55., 0.1, 2. * f.noiserms])), expect, rtol=1e-9)
    assert f._log_likelihood_geometry(np.array([55., 0.1, -1.])) == -np.inf
    g.run_geometry("inc_single fracerr", threads=1, verbose=False, dynamic=False,
                   sampler_kwargs=dict(nlive=15, seed=1), run_kwargs=dict(maxiter=60))
    assert g.freepar_names == ['inc', 'fracerr'] and np.all(np.isfinite(g.params))


def test_bayesian_radmax_edge(tmp_path):
    """RADMAX as the outer edge of the disk (like_kwargs radmax='edge', the default, or 'taper'): the
    rings do not move, the model has no emission beyond the edge (also with the deconvolved density
    profile), and the likelihood is continuous in RADMAX; radmax='ring' moves the last ring. The
    default prior is the last ring; with NODES rings, whose disk ends at the last radius, the half of
    it inside the last radius."""
    pytest.importorskip("dynesty")
    from bbarolo.galmod import edge_weights
    assert np.allclose(edge_weights([10., 20., 30.], 10., 20.), [1., 0.5, 0.])
    assert np.allclose(edge_weights([20.], 10., 20., 5.), 0.5)
    f = _bayes_mock(tmp_path)
    f._setup(['vrot_single', 'radmax'])
    assert f.like_kwargs['radmax'] == 'edge' and f._radmax_mode == 'edge'
    assert f.prior_distr['radmax'].ppf(0.) == 120. and f.prior_distr['radmax'].ppf(1.) == 160.
    r0 = np.array(f._fit0.r['radii'])
    L = [f._log_likelihood(np.array([150., e])) for e in np.linspace(125., 160., 36)]
    assert np.all(np.isfinite(L)) and np.max(np.abs(np.diff(L))) < 0.2 * np.ptp(L)
    rings = f._update_rings(f._fit0.copy(), np.array([150., 130.]))
    f._set_density(rings)
    assert np.array_equal(rings.r['radii'], r0) and rings.object.edge == 130.
    model = f._galfit.model.array(rings.object)
    yy, xx = np.mgrid[:64, :64]
    dx, dy = (xx - 31.5) * 10., (yy - 31.5) * 10.
    sp, cp = np.sin(np.radians(45.)), np.cos(np.radians(45.))
    rr = np.hypot(-dx * sp + dy * cp, (-dx * cp - dy * sp) / np.cos(np.radians(60.)))
    reach = {}                                   # the largest radius with emission (not convolved)
    for e in (100., 130., None):
        r = rings.object.copy()
        r.edge = e
        raw = f._galfit.model.setup.build(r, (0, 0), (64, 64)).sum(axis=0)
        reach[e] = rr[raw > 0].max()
    # (pixels of 10" at INC 60: the centre of a pixel is up to 2 x 7" from the points of the disk in it)
    assert reach[100.] <= 115. and reach[100.] < reach[130.] <= 145. and reach[130.] < reach[None]
    assert 0 < model.sum()
    g = _bayes_mock(tmp_path, outfolder=str(tmp_path / "g"))
    g._set_like_options(dict(radmax='taper', radmax_width=8.))
    g._setup(['vrot_single', 'radmax'])
    assert g._radmax_width == 8. and np.isfinite(g._log_likelihood(np.array([150., 140.])))
    h = _bayes_mock(tmp_path, outfolder=str(tmp_path / "h"))
    h._set_like_options(dict(radmax='ring'))
    h._setup(['vrot_single', 'radmax'])                       # 'ring': the last ring moves
    assert h._update_rings(h._fit0.copy(), np.array([150., 130.])).r['radii'][-1] == 130.
    n = _bayes_mock(tmp_path, outfolder=str(tmp_path / "n"), ringmode="NODES")
    n._setup(['vrot_single', 'radmax'])
    assert n.prior_distr['radmax'].ppf(0.) == 120. and n.prior_distr['radmax'].ppf(1.) == 140.
    n._set_like_options(dict(radmax='ring'))
    n._setup(['vrot_single', 'radmax'])
    assert n.prior_distr['radmax'].ppf(0.) == 120. and n.prior_distr['radmax'].ppf(1.) == 160.


def test_bayesian_flux_beyond_rings(tmp_path, capsys):
    """A warning when the rings do not cover the disk: the emission beyond them, in the region of the
    likelihood (that of the lowest INC allowed), biases the geometry."""
    pytest.importorskip("dynesty")
    f = _bayes_mock(tmp_path)
    f._setup(['vrot_single'])
    assert f._flux_beyond_rings() < 0.05
    g = _bayes_mock(tmp_path, radii=[20., 60.], outfolder=str(tmp_path / "g"))
    g._setup(['vrot_single'])
    assert g._flux_beyond_rings() > 0.05
    g.run(freepar=['vrot_single'], threads=1, dynamic=False, sampler_kwargs=dict(nlive=15, seed=1),
          run_kwargs=dict(maxiter=20, print_progress=False))
    assert "lies beyond the outermost ring" in capsys.readouterr().out


def test_z0_default_one_pixel(tmp_path):
    """Z0 not given: the fits (3DFIT and BayesianBBarolo) use a thin disk, one pixel (10" here), not a disk
    with no thickness; GALMOD needs Z0."""
    pytest.importorskip("dynesty")
    import copy
    from bbarolo.galfit import initial_rings
    f = _bayes_mock(tmp_path, z0=None)
    assert np.allclose(f._inri.r['z0'], 10.)
    fn = str(tmp_path / "mock.fits")
    rings = f"RADII 20 60\nXPOS 31.5\nYPOS 31.5\nVSYS 0\nVROT 120\nVDISP 15\nINC 60\nPA 45\n"
    p = Param()
    p.readParamString(f"FITSFILE {fn}\n3DFIT true\n{rings}MASK NONE\nVERBOSE 0\nOUTFOLDER {tmp_path}/\n")
    c = Cube(fn, par=p)
    ini = initial_rings(c, copy.deepcopy(p.gf), write=False)
    assert np.allclose(ini.rings.z0, 10.) and ini.estimated["Z0"]
    g = Param()
    g.readParamString(f"FITSFILE {fn}\nGALMOD true\n{rings}")
    assert not g.checkPars(quiet=True)


def test_plot_scripts_ring_columns():
    """The plotting scripts read the ring files by the names of their columns (galfit.RING_COLUMNS), not by
    fixed numbers: with the columns of C++ two scripts read past the last column (or the errors)."""
    import re
    import bbarolo.plotscripts as ps
    from bbarolo.galfit import RING_COLUMNS
    src = open(ps.__file__).read()
    assert not re.findall(r"(?:rings_final[12]\.txt'|frile),usecols=\(\d", src)
    for k, name in zip(ps._RING_KEYS, ("RAD(", "VROT(", "DISP(", "INC(", "P.A.(", "Z0(", "DENS(", "XPOS(",
                                       "YPOS(", "VSYS(", "VRAD(")):
        assert RING_COLUMNS[int(ps._uc(k)[1:-1])].startswith(name)


@needs_data
def test_two_stage_table_and_run_parameters(tmp_path, capsys):
    """VERBOSE 1: the second stage shows the columns of the first, the regularised parameters marked (r),
    after a line on how they were regularised. Every run writes <prefix>_params.par, with all its parameters
    (as given, then the defaults), which gives the same parameters when read again. Paths on screen are
    relative to the current folder, unless far from it."""
    from bbarolo import FitMod3D
    from bbarolo.bbutils import shortpath
    from bbarolo import keywords as K
    from bbarolo.tasks import STEPS
    assert shortpath(os.path.abspath("a/b.fits")) == os.path.join("a", "b.fits")
    assert shortpath(os.path.abspath("a") + "/") == "a/" and shortpath(str(tmp_path / "x")) == str(tmp_path / "x")
    kw = dict(radii=[150, 250, 350, 450, 550], xpos=77, ypos=77, vsys=132.8, vrot=120, vdisp=8, inc=60, pa=123.7,
              z0=10, free="VROT INC", twostage=True, regtype="median", threads=4, plots=False)
    for i in range(2):
        res = FitMod3D(N2403, **kw).run(outfolder=str(tmp_path / "f"))
    out = capsys.readouterr().out
    assert "BBAROLO WARNING: the output folder" in out and "NOTE" not in out
    assert "Regularised for the second stage (then fixed): INC median = " in out
    heads = [ln.split() for ln in out.splitlines() if ln.strip().endswith("outcome")]
    assert heads[-2][:4] == ["#", "RAD", "VROT", "INC"] and heads[-1][:4] == ["#", "RAD", "VROT", "INC(r)"]
    fn = str(tmp_path / "f" / "ngc2403_params.par")
    assert fn in res.files
    q = Param()
    q.readParamFile(fn)
    assert q.checkPars(quiet=True) and q.imageFile == os.path.abspath(N2403) and q.gf.flagGALFIT
    assert q.gf.FREE == "VROT INC" and q.gf.REGTYPE == "median" and q.threads == 4
    # the parameter file of a parameter file: the same values
    p = Param()
    p.readParamString(f"FITSFILE {N2403}\n3DFIT true\nRADII 150 250\nVROT 120\nFREE VROT VDISP\nMASK SEARCH\n"
                      "THRESHOLD 0.01\nREGTYPE bezier\n")
    p.checkPars(quiet=True)
    q = Param()
    q.readParamString(p.to_parfile([n for n, r, _ in STEPS if r(p, p)] + ["SEARCH"]))
    q.checkPars(quiet=True)
    assert all(K._getattr(p, t) == K._getattr(q, t) for k in K.KEYWORDS for t in k.targets)


def _g3d_cube():
    from bbarolo import emptyfits
    return emptyfits([64, 64, 40], [-10 / 3600., 10 / 3600., 10.], beam=30 / 3600., bunit="JY/BEAM", obj="SYNTH")


def test_galmod3d_vs_galmod():
    """GalMod3D with the fields of a GALMOD disk (Gaussian layer, VRAD, VVERT, NODES rings: the values
    interpolated linearly in radius) gives the GALMOD ANALYTIC model (the largest differences, ~2% of the
    peak, are in the central pixels, where GALMOD samples the innermost sub-ring coarsely)."""
    from bbarolo import GalMod3D, DiskModel
    hdu = _g3d_cube()
    p = Param()
    p.verbose = False
    c = Cube(par=p)
    c.readHDU(hdu)
    radii = np.arange(0, 261, 10.)
    sig = np.exp(-radii / 80.) * np.clip((250 - radii) / 50., 0, 1)
    vrot = 150 * np.tanh(radii / 40.)
    p.gf.LTYPE = 1
    dm = DiskModel(c, ringmode="NODES", galmodtype="ANALYTIC", densflux=True)
    g = dm.hdu(Rings().setRings(radii=radii, xpos=31.5, ypos=31.5, vsys=0., vrot=vrot, vrad=15., vvert=5.,
                                vdisp=10., z0=8., dens=sig, inc=60, phi=45)).data.astype(float)
    m = GalMod3D(hdu, coords="Rz", xpos=31.5, ypos=31.5, inc=60, pa=45, rmax=260, zmax=40, step=(10 / 3., 2.),
                 dens=lambda R, z: np.interp(R, radii, sig, right=0) * np.exp(-z * z / 128.),
                 vphi=lambda R, z: np.interp(R, radii, vrot), vr=15., vz=5., vdisp=10.)
    f = m.model().data.astype(float)
    f *= g.sum() / f.sum()
    assert np.sqrt(np.mean((f - g) ** 2)) < 1e-3 * g.max() and np.abs(f - g).max() < 0.03 * g.max()
    assert np.corrcoef(f.ravel(), g.ravel())[0, 1] > 0.9999


def test_galmod3d_conventions_coords_grids():
    """The geometry of GALMOD: phi=0 is the receding half of the major axis (at PA), y > 0 the far side,
    v_los = vsys + v_y sin(i) + v_z cos(i). The same disk as a function of (R, z), (R, phi, z) or (x, y, z)
    gives the same cube, and as values on a grid the same within the interpolation."""
    from bbarolo import GalMod3D
    hdu = _g3d_cube()
    h = hdu.header
    vaxis = h["CRVAL3"] + (np.arange(40) + 1 - h["CRPIX3"]) * h["CDELT3"]
    geo = dict(xpos=31.5, ypos=31.5, vsys=0, inc=60, pa=45)
    si, ci = np.sin(np.radians(60)), np.cos(np.radians(60))

    def blob(x0, y0, z0, **v):
        d = GalMod3D(hdu, coords="xyz", rmax=150, zmax=40, **geo, **v,
                     dens=lambda x, y, z: np.exp(-((x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2) / 50.)
                     ).model(smooth=False).data.astype(float)
        _, yy, xx = np.indices(d.shape)
        return np.array([(d * xx).sum(), (d * yy).sum(), (d.sum(axis=(1, 2)) * vaxis).sum()]) / d.sum()
    s45 = np.sin(np.radians(45))
    assert np.allclose(blob(100, 0, 0, vy=100.), [31.5 - 10 * s45, 31.5 + 10 * s45, 100 * si], atol=0.1)
    assert np.allclose(blob(0, 100, 0, vy=50.), [31.5 + 10 * ci * s45, 31.5 + 10 * ci * s45, 50 * si], atol=0.1)
    assert np.allclose(blob(0, 0, 30, vz=40.), [31.5 - 3 * si * s45, 31.5 - 3 * si * s45, 40 * ci], atol=0.1)
    vp = lambda R: 150 * np.tanh(R / 40.)
    S = lambda R, z: np.exp(-R / 80.) * np.exp(-z * z / 128.)
    kw = dict(**geo, rmax=250, zmax=30, vdisp=10.)
    a = GalMod3D(hdu, coords="Rz", dens=S, vphi=lambda R, z: vp(R), vr=10., **kw).model().data
    b = GalMod3D(hdu, coords="cylindrical", dens=lambda R, p, z: S(R, z), vphi=lambda R, p, z: vp(R), vr=10.,
                 **kw).model().data

    def vxy(x, y, z):
        R = np.hypot(x, y)
        return (10 * x - vp(R) * y) / np.where(R > 0, R, 1), (10 * y + vp(R) * x) / np.where(R > 0, R, 1)
    c = GalMod3D(hdu, coords="xyz", dens=lambda x, y, z: S(np.hypot(x, y), z), vx=lambda *p: vxy(*p)[0],
                 vy=lambda *p: vxy(*p)[1], **kw).model().data
    assert np.allclose(a, b, rtol=0, atol=1e-6 * a.max()) and np.allclose(a, c, rtol=0, atol=1e-6 * a.max())
    Rg, zg = np.linspace(0, 360, 181), np.linspace(-30, 30, 61)
    RR, ZZ = np.meshgrid(Rg, zg, indexing="ij")
    g = GalMod3D(hdu, coords="Rz", dens=((Rg, zg), S(RR, ZZ)), vphi=((Rg, zg), vp(RR)), vr=10., **kw).model().data
    assert np.abs(g - a).max() < 2e-3 * a.max()
    # (R, phi, z) on a grid, periodic in phi: a constant phi pattern gives the axisymmetric model
    pg = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    R3, P3, Z3 = np.meshgrid(Rg, pg, zg, indexing="ij")
    g3 = GalMod3D(hdu, coords="Rphiz", dens=((Rg, pg, zg), S(R3, Z3)), vphi=((Rg, pg, zg), vp(R3)), vr=10.,
                  **kw).model().data
    assert np.abs(g3 - g).max() < 1e-5 * a.max()


def test_galmod3d_mc_flux_errors():
    """MC: the flux of the deterministic model, the same shape, reproducible with a seed. flux: a number
    (Jy km/s) or 'measured' (the data of the cube in its mask). Errors name what is wrong."""
    from bbarolo import GalMod3D, BBaroloError
    from bbarolo.bbutils import FluxtoJy
    hdu = _g3d_cube()
    kw = dict(coords="Rz", xpos=31.5, ypos=31.5, inc=60, pa=45, rmax=250, zmax=30, vdisp=10.,
              dens=lambda R, z: np.exp(-R / 80.) * np.exp(-z * z / 128.), vphi=lambda R, z: 150 * np.tanh(R / 40.))
    a = GalMod3D(hdu, **kw).model().data
    m = GalMod3D(hdu, **kw)
    mc = m.model(galmodtype="MC", nclouds=1e6, seed=2).data
    assert abs(mc.sum() / a.sum() - 1) < 1e-5 and np.corrcoef(mc.ravel(), a.ravel())[0, 1] > 0.999
    assert np.array_equal(mc, m.model(galmodtype="MC", nclouds=1e6, seed=2).data)
    f = GalMod3D(hdu, flux=100., **kw)
    d = f.model().data.astype(float)
    assert abs(float(FluxtoJy(d.sum(), f.cube.Head())) * 10. - 100.) < 1e-3
    data = hdu.copy()
    data.data = (2 * d + np.random.default_rng(1).normal(0, 1e-4, d.shape)).astype(np.float32)
    meas = GalMod3D(data, flux="measured", mask="NONE", **kw).model().data
    assert np.isclose(meas.sum(), data.data.sum(), rtol=1e-5)
    with pytest.raises(BBaroloError, match="coords"):
        GalMod3D(hdu, **{**kw, "coords": "spherical"})
    with pytest.raises(BBaroloError, match="vx is not a velocity"):
        GalMod3D(hdu, vx=10., **kw)
    with pytest.raises(BBaroloError, match="grid of dens has 3 axes"):
        GalMod3D(hdu, **{**kw, "dens": (([0, 1], [0, 1], [0, 1]), np.ones((2, 2, 2)))})
    with pytest.raises(BBaroloError, match="no emission inside the cube"):
        GalMod3D(hdu, **{**kw, "xpos": 500}).model()
    with pytest.raises(BBaroloError, match="no positive flux"):
        GalMod3D(hdu, flux="measured", mask="NONE", **kw).model()
    with pytest.raises(TypeError):
        GalMod3D(hdu, vrot=10., **kw)


def test_bayesian_rings_beyond_radmax(tmp_path, capsys):
    """With RADMAX as the edge of the disk, the rings beyond it are marked: in_disk, INDISK = 0 in
    rings_final1.txt, a note of print_stats and the region beyond RADMAX shaded in the plots of the
    parameters. load_results keeps the output folder given to the object."""
    pytest.importorskip("dynesty")
    from bbarolo.plotscripts import writeScripts_Python
    from bbarolo.tasks import read_table
    f = _bayes_mock(tmp_path)
    f.priors['radmax'] = dict(name='uniform', loc=80., scale=50.)
    f.run(freepar=['vrot_single', 'radmax'], threads=1, verbose=False, dynamic=False,
          sampler_kwargs=dict(nlive=15, seed=1), run_kwargs=dict(maxiter=60))
    assert 80 <= f.disk_edge <= 130 and list(f.in_disk) == [True, True, True, False]
    f.write_bestmodel(plots=False, verbose=False)
    fn = tmp_path / "bbb" / "MOCK_rings_final1.txt"
    t = read_table(str(fn))
    assert list(t["indisk"]) == [1, 1, 1, 0] and t.colnames[-1] == "indisk"
    capsys.readouterr()
    f.print_stats()
    assert "Ring 4 (R = 140) lies beyond it" in capsys.readouterr().out.replace("\n", " ")
    with f._errors_in_outputs():
        writeScripts_Python(f._galfit)
    assert "axvspan(radmax" in (tmp_path / "bbb" / "plotscripts" / "plot_parameters.py").read_text()
    f.save_results(str(tmp_path / "res.npy"))
    g = _bayes_mock(tmp_path, outfolder=str(tmp_path / "other"))
    g.load_results(str(tmp_path / "res.npy"))
    assert g._outfolder() == str(tmp_path / "other") + "/" and list(g.in_disk) == [True, True, True, False]
    h = _bayes_mock(tmp_path, outfolder=str(tmp_path / "ring"))
    h.priors['radmax'] = dict(name='uniform', loc=120., scale=40.)
    h.run(freepar=['vrot_single', 'radmax'], like_kwargs=dict(radmax='ring'), threads=1, verbose=False,
          dynamic=False, sampler_kwargs=dict(nlive=15, seed=1), run_kwargs=dict(maxiter=40))
    assert h.disk_edge is None and h.in_disk.all()


def test_keyword_docs():
    """Every keyword has its full description (keywords_doc.py, shown by help('KEYWORD') and on the website)
    and a section of the parameter page of each of its tasks."""
    import io
    from bbarolo import keywords as K
    from bbarolo.keywords_doc import GROUPS, DOCS
    assert set(DOCS) == {k.name for k in K.KEYWORDS}
    for k in K.KEYWORDS:
        assert k.doc and len(k.doc.split()) >= 8, k.name
        assert all(len(ln) <= 110 for ln in k.doc.splitlines()), k.name
        assert k.default_text is None or k.default_text.strip(), k.name
        if isinstance(k.group, dict):
            assert "*" in k.group and set(k.group) - {"*"} <= set(k.tasks), k.name
        for t in k.tasks:
            assert k.group_for(t) in GROUPS, (k.name, t)
    s = io.StringIO()
    K.help("free", stream=s)
    assert "NAME_SINGLE" in s.getvalue() and "Default: " in s.getvalue()


def test_smooth_scale_factor_of_prefixed_units(tmp_path):
    """SMOOTH rescales the data in mJy/beam and uJy/beam by the ratio of the beam areas, as those in Jy/beam
    (they were not rescaled since mJy were no longer taken as Jy), and also those in W.U. (C++ did not); data
    per pixel and in K are not rescaled."""
    factors = {}
    for bunit in ("JY/BEAM", "mJy/beam", "uJy/beam", "W.U.", "Jy", "K"):
        hdu = emptyfits([32, 32, 3], [-10. / 3600., 10. / 3600., 10.], beam=30. / 3600., bunit=bunit)
        fn = str(tmp_path / "unit.fits")
        hdu.writeto(fn, overwrite=True)
        p = Param()
        p.verbose = False
        p.readParamString(f"FITSFILE {fn}\nSMOOTH true\nFACTOR 2\nOUTFOLDER {tmp_path}\n")
        sm = Smooth3D()
        sm.cubesmooth(Cube(fn, par=p))
        factors[bunit] = sm.scalefac
    assert abs(factors["JY/BEAM"] - 4) < 0.01
    assert factors["mJy/beam"] == factors["uJy/beam"] == factors["W.U."] == factors["JY/BEAM"]
    assert factors["Jy"] == factors["K"] == 1


def test_mask_header_growth_threshold(tmp_path):
    """The HISTORY card GROWTHTHRESHOLD of the mask of the source finder has the growth threshold, not the
    threshold (C++ wrote THRESHOLD there)."""
    from astropy.io import fits
    fn = _mock_cube(tmp_path)
    p = Param()
    p.verbose = False
    p.readParamString(f"FITSFILE {fn}\nMASK SEARCH\nTHRESHOLD 0.02\nGROWTHTHRESHOLD 0.005\nOUTFOLDER {tmp_path}/\n")
    c = Cube(fn, par=p)
    c.BlankMask()
    hist = [str(s) for s in fits.getheader(str(tmp_path / "MOCK_mask.fits"))["HISTORY"]]
    assert any("GROWTHTHRESHOLD=0.005000" in s for s in hist), hist


def test_ring_lists_with_commas_and_mixed_centre():
    """Lists of ring values can be separated by commas (VROT 100,120,130 became 0, RADII 30,60,90 a
    ValueError); with XPOS (or YPOS) given per ring and the other one a single value in pixels, that value
    is used (it became 0), and a single WCS value is an error."""
    from bbarolo.rings import readRings
    from bbarolo.header import Header
    from bbarolo.bbutils import BBaroloError
    hh = Header()
    hh.fromAstropy(emptyfits([32, 32, 10], [-10. / 3600., 10. / 3600., 10.], beam=30. / 3600.).header)
    base = "XPOS 10\nYPOS 12\nVSYS 0\nVROT 100\nVDISP 10\nINC 60\nPA 30\nZ0 0\n"

    def rings(extra):
        p = Param()
        p.readParamString(base + extra)
        return readRings(p.gf, hh)
    r = rings("RADII 30,60,90\nVROT 100,120, 130\nINC 60, 61, 62\n")
    assert list(r.radii) == [30, 60, 90] and list(r.vrot) == [100, 120, 130] and list(r.inc) == [60, 61, 62]
    r = rings("RADII 30 60 90\nXPOS 10 11 12\n")
    assert list(r.xpos) == [10, 11, 12] and list(r.ypos) == [12, 12, 12]
    r = rings("RADII 30 60 90\nYPOS 10,11,12\n")
    assert list(r.xpos) == [10, 10, 10] and list(r.ypos) == [10, 11, 12]
    with pytest.raises(BBaroloError, match="YPOS"):
        rings("RADII 30 60 90\nXPOS 10 11 12\nYPOS 12.5d\n")


def test_galwind_lists_and_openang(synthetic_cube):
    """GALWIND takes lists of values, one per cylinder or shell (only the first value was used), and
    OPENANG as a file column with cylinders is an error that names it."""
    from bbarolo.galwind import GalWind
    from bbarolo.bbutils import BBaroloError
    c = synthetic_cube
    gw = c.pars().gw
    for k, v in dict(XPOS="31.5", YPOS="31.5", VSYS="0", VWIND="100", VROT="0", VDISP="10", DENS="10",
                     INC="60", PHI="30", OPENANG="60", HTOT=400., NTOT=4, SM=False, ISEED=1).items():
        setattr(gw, k, v)

    def model(**kw):
        w = GalWind(c)
        for k, v in kw.items():
            setattr(w.par, k, v)
        w.compute()
        return w.out.array
    for wtype in (0, 1):
        one = model(WTYPE=wtype)
        assert np.array_equal(one, model(WTYPE=wtype, VWIND="100 100 100 100"))
        assert not np.array_equal(one, model(WTYPE=wtype, VWIND="100, 150, 200, 250"))
    with pytest.raises(BBaroloError, match="NTOT=4"):
        model(WTYPE=0, VWIND="100 150")
    with pytest.raises(BBaroloError, match="OPENANG"):
        model(WTYPE=0, OPENANG="file(rings.txt,2)")


def _smooth_run(tmp_path, fn, pars):
    """Smooth3D.cubesmooth of the cube fn with the parameters pars (a parameter-file string)."""
    p = Param()
    p.verbose = False
    p.readParamString(f"FITSFILE {fn}\nOUTFOLDER {tmp_path}/\n" + pars)
    sm = Smooth3D()
    sm.cubesmooth(Cube(fn, par=p))
    return sm


def test_smooth_cube_without_beam(tmp_path):
    """With no beam in the header, SMOOTH takes the beam of the data from OBMAJ/OBMIN/OBPA (or BEAMFWHM):
    BMAJ/BMIN, the new beam, became also the old one, so nothing was smoothed, and OBMAJ alone stopped with
    'No beam'. The SMOOTH mask of a fit with BMAJ as the beam of the data uses FACTOR times it."""
    from bbarolo.bbutils import BBaroloError
    from astropy.io import fits
    hdu = emptyfits([32, 32, 3], [-10. / 3600., 10. / 3600., 10.], bunit="JY/BEAM", obj="NOBEAM")
    hdu.data[:, 16, 16] = 1.
    fn = str(tmp_path / "nobeam.fits")
    hdu.writeto(fn)
    sm = _smooth_run(tmp_path, fn, "SMOOTH true\nOBMAJ 30\nOBMIN 30\nBMAJ 60\nBMIN 60\n")
    assert sm.oldbeam[:2] == (30, 30) and sm.newbeam[:2] == (60, 60) and abs(sm.scalefac - 4) < 0.01
    assert np.nanmax(sm.array) < 0.3
    sm = _smooth_run(tmp_path, fn, "SMOOTH true\nOBMAJ 30\nOBMIN 30\n")
    assert sm.oldbeam[:2] == (30, 30) and sm.newbeam[:2] == (60, 60)
    with pytest.raises(BBaroloError, match="OBMAJ"):
        _smooth_run(tmp_path, fn, "SMOOTH true\nBMAJ 60\nBMIN 60\n")
    p = Param()
    p.verbose = False
    p.readParamString(f"FITSFILE {fn}\nOUTFOLDER {tmp_path}/\nMAKEMASK true\nMASK SMOOTH\nBMAJ 30\nBMIN 30\n")
    c = Cube(fn, par=p)
    c.BlankMask()
    hist = [str(x) for x in fits.getheader(str(tmp_path / "NOBEAM_mask.fits"))["HISTORY"]]
    assert any("NEWBMAJ=60.0" in x for x in hist), hist


def test_smooth_new_beam_angle(tmp_path):
    """The new beam keeps the angle of the old one when BPA is not given (it was 0, also with FACTOR alone),
    and OBPA 0 is an angle of 0 degrees (it meant the BPA of the header)."""
    hdu = emptyfits([32, 32, 3], [-10. / 3600., 10. / 3600., 10.], bunit="JY/BEAM", obj="ELL")
    hdu.header["BMAJ"], hdu.header["BMIN"], hdu.header["BPA"] = 40. / 3600., 25. / 3600., 60.
    fn = str(tmp_path / "ell.fits")
    hdu.writeto(fn)
    sm = _smooth_run(tmp_path, fn, "SMOOTH true\nFACTOR 1.2\n")
    assert sm.oldbeam[2] == sm.newbeam[2] == 60 and sm.conbeam is not None
    sm = _smooth_run(tmp_path, fn, "SMOOTH true\nFACTOR 1.5\nBPA 0\nOBPA 0\n")
    assert sm.oldbeam[2] == 0 and sm.newbeam[2] == 0


def test_iternoise_converges(synthetic_cube):
    """ITERNOISE stops when the noise changes by less than 0.1% (the test added the noise itself, so it
    ran all the 20 iterations for noise above 0.001)."""
    c = synthetic_cube
    c.array[:] = np.random.default_rng(3).normal(0., 1., c.array.shape)
    c.pars().se.iternoise = True
    calls = []
    calc = c.stats.calculate
    c.stats.calculate = lambda *a, **k: (calls.append(1), calc(*a, **k))[1]
    c.setCubeStats()
    assert len(calls) < 10 and abs(c.stats.getSpread() - 1) < 0.05


def test_search_snmap_names(tmp_path):
    """SEARCH with SNMAP writes its noise and S/N maps as <prefix>_mom0th_RMS.fits and _mom0th_SN.fits: they
    had the names of those of TOTALMAP, which overwrote them in the same run."""
    from bbarolo.tasks import BBcore
    fn = _mock_cube(tmp_path)
    p = Param()
    p.verbose = False
    p.readParamString(f"FITSFILE {fn}\nOUTFOLDER {tmp_path}/out/\nSEARCH true\nSNMAP true\nTOTALMAP true\n"
                      "MASK SEARCH\nPLOTS false\n")
    assert p.checkPars(quiet=True)
    BBcore(p)
    for n in ("mom0th", "mom0th_RMS", "mom0th_SN", "map_0th", "map_0th_RMS", "map_0th_SN"):
        assert (tmp_path / "out" / f"MOCK_{n}.fits").exists(), n


def test_search_checks_for_masks(tmp_path, capsys):
    """The checks of SEARCHTYPE and of the growth threshold run also when the source finder only builds a
    mask (MASK SEARCH), not only with SEARCH true."""
    p = Param()
    p.readParamString("FITSFILE a.fits\nMAKEMASK true\nMASK SEARCH\nSEARCHTYPE spatail\nSNRCUT 2\n")
    p.checkPars(quiet=False)
    assert p.se.searchType == "spatial" and not p.se.flagGrowth


def test_norm_free_side_checks():
    """NORM AZIMUTHAL is AZIM (accepted by the checks, then LOCAL); FTYPE and WFUNC are checked also for
    SPACEPAR (FTYPE 7 gave zero residuals); the long forms of SIDE work for every task that uses it (only
    3DFIT converted them)."""
    from bbarolo.galfit import final_norm
    assert final_norm("AZIMUTHAL") == "AZIM" and final_norm("LOCAL AZIMUTHAL") == "AZIM"
    base = ("FITSFILE a.fits\nRADII 30 60\nXPOS 10\nYPOS 10\nVSYS 0\nVROT 100\nVDISP 10\nINC 60\nPA 30\nZ0 1\n"
            "P1 VROT\nP2 VDISP\n")
    p = Param()
    p.readParamString(base + "3DFIT true\nNORM AZIMUTHAL LOCAL\n")
    p.checkPars(quiet=True)
    assert p.gf.NORM == "AZIM LOCAL"
    p = Param()
    p.readParamString(base + "SPACEPAR true\nFTYPE 7\nWFUNC 5\nSIDE APPROACHING\n")
    p.checkPars(quiet=True)
    assert p.gf.FTYPE == 2 and p.gf.WFUNC == 1 and p.gf.SIDE == "A"
    for task, side, short in (("2DFIT", "RECEDING", "R"), ("ELLPROF", "APP", "A"), ("2DFIT", "BOTH", "B")):
        p = Param()
        p.readParamString(base + f"{task} true\nSIDE {side}\n")
        p.checkPars(quiet=True)
        assert p.gf.SIDE == short, (task, side)


@needs_data
def test_spacepar_rings_and_norm(tmp_path):
    """SPACEPAR keeps the PA and VROT of the rings given (with PA in FREE, as by default, it replaced them
    with a 2D fit) and normalises the models as the first stage of 3DFIT (NORM; it used LOCAL); 3DFIT keeps
    a fixed VROT when it refines the initial PA with the 2D fit."""
    from bbarolo.spacepar import Spacepar
    from bbarolo.galfit import Galfit
    base = (f"FITSFILE {N2403}\nNRADII 6\nRADSEP 60\nVSYS 132.8\nXPOS 77\nYPOS 77\nVROT 120\nVDISP 8\nINC 60\n"
            f"PA 123.7\nZ0 10\nP1 VROT\nP2 VDISP\nP1PAR 100 140 20\nP2PAR 5 15 5\nTHREADS 2\nVERBOSE false\n"
            f"MASK SEARCH\nOUTFOLDER {tmp_path}/\n")

    def make(cls, extra):
        p = Param()
        p.readParamString(base + extra)
        assert p.checkPars(quiet=True)
        return cls(Cube(N2403, par=p))
    res = {}
    for norm in ("AZIM", "LOCAL"):
        sp = make(Spacepar, f"SPACEPAR true\nNORM {norm}\n")
        assert np.all(sp.inr.phi == 123.7) and np.all(sp.inr.vrot == 120)
        res[norm] = sp.getFuncValue(sp._dring(3, 120., 8.))
    assert res["AZIM"] != res["LOCAL"]
    gf = make(Galfit, "3DFIT true\nFREE INC PA\n")
    assert np.all(gf.inr.vrot == 120) and not np.all(gf.inr.phi == 123.7)
    gf = make(Galfit, "3DFIT true\nFREE VROT INC PA\n")
    assert not np.all(gf.inr.vrot == 120)


@needs_data
def test_radsep_alone_keeps_extent(tmp_path):
    """With RADSEP but no NRADII, the rings reach out to the estimated extent of the disk (the number of
    rings was that of the beam-wide rings, so the extent scaled with RADSEP)."""
    from bbarolo.galfit import initial_rings
    last = {}
    for radsep in ("", "RADSEP 30\n", "RADSEP 120\n"):
        p = Param()
        p.readParamString(f"FITSFILE {N2403}\nTHREADS 2\nVERBOSE false\nOUTFOLDER {tmp_path}/\n" + radsep)
        assert p.checkPars(quiet=True)
        r = initial_rings(Cube(N2403, par=p), p.gf, write=False).rings
        last[radsep] = r.radii[-1] + r.radsep / 2
    assert abs(last["RADSEP 30\n"] - last[""]) < 60 and abs(last["RADSEP 120\n"] - last[""]) < 120, last


def test_2dfit_freeangle(synthetic_cube, tmp_path):
    """2DFIT takes the angle left out around the minor axis from FREEANGLE (it read DVDZ, so the DVDZ of
    a lagging disk changed it)."""
    from bbarolo.ringmodel import Ringmodel
    c = synthetic_cube
    c.array[:] = 0.
    c.array[20, 24:40, 24:40] = 1.
    for extra, angle in (("DVDZ 0.5\n", 15.), ("DVDZ 0.5\nFREEANGLE 20\n", 20.)):
        c.pars().readParamString(f"2DFIT true\nMASK NONE\nOUTFOLDER {tmp_path}/\n" + extra)
        tr = Ringmodel()
        tr.setfromCube(c, _rings(4))
        assert tr.thetaf == angle
    c.pars().readParamString("FITSFILE a.fits\nFREEANGLE 95\n")
    assert not c.pars().checkPars(quiet=True)


def test_galwind_outflow_signs_and_nv(synthetic_cube):
    """With VWIND > 0 the cylinders of GALWIND (WTYPE 0) are an outflow, as the shells (WTYPE 1): they had the
    opposite line-of-sight velocities on each side of the centre. NV 1 is one sub-cloud for the shells (any
    NV up to 1 was 20)."""
    from bbarolo.galwind import GalWind
    c = synthetic_cube
    gw = c.pars().gw
    for k, v in dict(XPOS="31.5", YPOS="31.5", VSYS="0", VWIND="100", VROT="0", VDISP="10", DENS="10",
                     INC="60", PHI="0", OPENANG="60", HTOT=400., NTOT=4, SM=False, ISEED=1).items():
        setattr(gw, k, v)

    def model(**kw):
        w = GalWind(c)
        for k, v in kw.items():
            setattr(w.par, k, v)
        w.compute()
        return w.out.array.astype(float)
    h = c.Head()
    v = h.Crval(2) + (np.arange(c.DimZ()) + 1 - h.Crpix(2)) * h.Cdelt(2)
    east = {}
    for wtype in (0, 1):
        a = model(WTYPE=wtype)[:, :, :31]           # (the east half: x < XPOS)
        east[wtype] = (a * v[:, None, None]).sum() / a.sum()
    assert east[0] > 20 and east[1] > 20, east
    assert not np.array_equal(model(WTYPE=1, NV=1), model(WTYPE=1, NV=20))


def test_exclwind_lists_and_errors(synthetic_cube):
    """EXCLWIND accepts commas between the windows (they were ignored), and a channel outside the cube or an
    item that is not a channel is an error that names EXCLWIND (IndexError, or ignored)."""
    from bbarolo.bbutils import BBaroloError
    c = synthetic_cube
    z = np.arange(c.DimZ(), dtype=np.float32)
    line = np.where((z >= 10) & (z <= 15), 5., 0.) + np.where(z == 30, 5., 0.)
    data = (1. + 0.01 * z + line)[:, None, None] * np.ones(c.array.shape, np.float32)
    out = {}
    for ex in ("10:15 30", "10:15,30", "10~15, 30"):
        c.array = data.copy()
        c.pars().exclude_windows = ex
        c.continuumSubtract()
        out[ex] = c.array.copy()
        assert abs(out[ex][0, 0, 0]) < 1e-4 and abs(out[ex][12, 0, 0] - 5) < 1e-4
    for ex in ("10:15 45", "10:15;30", "ten"):
        c.array = data.copy()
        c.pars().exclude_windows = ex
        with pytest.raises(BBaroloError, match="EXCLWIND"):
            c.continuumSubtract()


def test_checkcube_lines_and_output(tmp_path):
    """CHECKCUBE erases noisy pixel lines also on cubes that are not square (the thresholds of the x and y
    lines were swapped, so on cubes wider than tall no x line was erased), and writes the cleaned cube in
    OUTFOLDER as <prefix>_clean.fits (it was written next to the input file)."""
    hdu = emptyfits([48, 24, 10], [-10. / 3600., 10. / 3600., 10.], beam=30. / 3600., bunit="JY/BEAM", obj="CHK")
    rng = np.random.default_rng(1)
    hdu.data[:] = rng.normal(0., 1., hdu.data.shape)
    hdu.data[:, :, 5] *= 5.
    hdu.data[:, 7, :] *= 5.
    os.makedirs(tmp_path / "data")
    fn = str(tmp_path / "data" / "chk.fits")
    hdu.writeto(fn)
    p = Param()
    p.verbose = False
    p.readParamString(f"FITSFILE {fn}\nOUTFOLDER {tmp_path}/out/\n")
    os.makedirs(tmp_path / "out")
    c = Cube(fn, par=p)
    c.CheckCube(1)
    assert np.all(c.array[:, :, 5] == 0) and np.all(c.array[:, 7, :] == 0)
    assert (tmp_path / "out" / "CHK_clean.fits").exists() and not (tmp_path / "data" / "chk_clean.fits").exists()


def test_reduce_factor_check():
    """REDUCE on its own needs FACTOR >= 1 (FACTOR 0 raised ZeroDivisionError)."""
    p = Param()
    p.readParamString("FITSFILE a.fits\nREDUCE true\nFACTOR 0\n")
    assert not p.checkPars(quiet=True)
    p.readParamString("FACTOR 2\n")
    assert p.checkPars(quiet=True)


def _galfit_n2403(tmp_path, extra):
    """A Galfit on NGC 2403 with the rings given (nothing estimated), set up but not run."""
    from bbarolo.galfit import Galfit
    p = Param()
    p.readParamString(f"FITSFILE {N2403}\n3DFIT true\nVSYS 132.8\nXPOS 77\nYPOS 77\nVDISP 8\nINC 60\nPA 123.7\n"
                      f"Z0 10\nMASK SEARCH\nTHREADS 2\nVERBOSE false\nOUTFOLDER {tmp_path}/\n" + extra)
    assert p.checkPars(quiet=True)
    return Galfit(Cube(N2403, par=p))


@needs_data
def test_vdisp_trials_within_bounds(tmp_path):
    """A VDISP trial value out of [MINVDISP, MAXVDISP] is redrawn within the interval (as bound -/+ u·VDISP
    it crossed the other bound when VDISP > MAXVDISP - MINVDISP, and a negative dispersion stopped the fit);
    MINVDISP must be smaller than MAXVDISP."""
    gf = _galfit_n2403(tmp_path, "RADII 60 120 180\nVROT 120\nFREE VDISP\nMINVDISP 1\nMAXVDISP 5\n")
    dring = gf._make_dring(1)
    for trial in (50., -50.) * 10:
        z = np.array([trial])
        gf.func3D(dring, z)
        assert 1 <= z[0] <= 5
    p = Param()
    p.readParamString("FITSFILE a.fits\n3DFIT true\nMINVDISP 10\nMAXVDISP 5\n")
    assert not p.checkPars(quiet=True)


@needs_data
def test_reverse_rings_by_radius(tmp_path):
    """REVERSE replaces rings narrower than the beam with beam-wide rings, whose initial values are those of
    the given rings at the new radii (they were those of the ring with the same index, at a smaller radius)."""
    radii, vrot = np.array([15., 45., 75., 105., 135., 165.]), np.array([10., 40., 70., 100., 130., 160.])
    gf = _galfit_n2403(tmp_path, f"RADII {' '.join(map(str, radii))}\nVROT {' '.join(map(str, vrot))}\n"
                                 "FREE VROT\nREVERSE true\n")
    assert gf.inr.nr == 3 and gf.inr.radsep > 55
    assert np.allclose(gf.inr.vrot, np.interp(gf.inr.radii, radii, vrot))


@needs_data
def test_second_stage_and_adrift_skip_failed_rings(tmp_path, monkeypatch):
    """The second stage of 3DFIT regularises only the rings fitted in the first stage (C++ used also the
    rings not fitted, with their initial values), evaluated at all the radii; when the regularisation fails,
    the rings not fitted are empty in the final model (they kept their DENS); ADRIFT uses only the rings
    fitted."""
    from bbarolo import galfit_out
    gf = _galfit_n2403(tmp_path, "RADII 60 120 180 240 300 360\nVROT 120\nFREE VROT INC\nREGTYPE 0\nADRIFT true\n")
    gf.outr = gf.inr.copy()
    gf.outr.inc[:] = [30., 30., 60., 61., 62., 63.]
    gf.fitok = [False, False, True, True, True, True]
    monkeypatch.setattr(gf, "galfit", lambda: None)
    gf.SecondStage()
    assert np.allclose(gf.inr.inc, 61.5)              # (the mean of 61 and 62: max and min left out)
    dens = gf.outr.dens.copy()
    monkeypatch.setattr(gf, "regularizeParams", lambda *a, **k: (False, None))
    gf.SecondStage()
    assert np.all(gf.outr.dens[:2] == 0) and np.all(gf.outr.dens[2:] == dens[2:])
    seen = {}
    monkeypatch.setattr(galfit_out, "asymmetricDrift", lambda c, par, rad, *a: seen.setdefault("rad", rad) is None)
    gf.outr.dens[:] = 1.
    gf.writeModel("LOCAL", makeplots=False)
    assert list(seen["rad"]) == [180., 240., 300., 360.]


def test_mc_model_in_box_is_crop(synthetic_cube):
    """An MC model built in a box is the crop of the model of the whole cube, also when clouds fall outside
    the box (a Lorentzian layer): they did not draw their velocity deviates, so the rest of the ring was
    another realisation."""
    setup = GalmodSetup(synthetic_cube, densflux=True)
    setup.ltype = 4
    r = _rings(3)
    r.z0[:] = 30.
    full = setup.build(r, galmodtype="MC", seed=7)
    blo, bhi = (12, 12), (52, 52)
    box = setup.build(r, blo=blo, bhi=bhi, galmodtype="MC", seed=7)
    crop = full[..., blo[1]:bhi[1], blo[0]:bhi[0]] if full.shape[-2:] == synthetic_cube.array.shape[1:] else None
    assert crop is not None and full.sum() > box.sum() > 0
    assert np.array_equal(box, crop)


def test_several_lines_scale_with_velocity(tmp_path):
    """Several lines: each line of a cloud is at its own rest wavelength times the same Doppler factor (each
    line had a fixed velocity offset from the first, so with optical velocities and VSYS = 3000 km/s
    [NII]6583 was 0.2 A, 2 channels, too blue), in the MC and ANALYTIC models."""
    from astropy.io import fits
    from bbarolo import DiskModel
    C = 299792.458
    h = fits.Header()
    for k, v in dict(NAXIS=3, NAXIS1=32, NAXIS2=32, NAXIS3=400, CTYPE1="RA---SIN", CTYPE2="DEC--SIN",
                     CTYPE3="WAVE", CRPIX1=16.5, CRPIX2=16.5, CRPIX3=1, CRVAL1=150., CRVAL2=30., CRVAL3=6620.,
                     CDELT1=-1 / 3600., CDELT2=1 / 3600., CDELT3=0.1, CUNIT1="deg", CUNIT2="deg", CUNIT3="Angstrom",
                     BMAJ=3 / 3600., BMIN=3 / 3600., BPA=0., BUNIT="JY/BEAM", OBJECT="LINES").items():
        h[k] = v
    fn = str(tmp_path / "lines.fits")
    fits.writeto(fn, np.zeros((400, 32, 32), np.float32), h)
    p = Param()
    p.readParamString("RESTWAVE 6562.8 6583.45\nRELINT 1 1\nVELDEF optical\nVERBOSE false\n")
    c = Cube(fn, par=p)
    lam = 6620. + 0.1 * np.arange(400)
    r = Rings().setRings(radii=[2., 4., 6.], xpos=15.5, ypos=15.5, vsys=3000., vrot=0., vdisp=5., z0=0., inc=30.,
                         phi=0., dens=1.)
    for gtype in ("MC", "ANALYTIC"):
        spec = DiskModel(c, smooth=False).array(r, galmodtype=gtype).sum((1, 2))
        nii = lam > 6640.
        centre = np.sum(spec[nii] * lam[nii]) / spec[nii].sum()
        assert abs(centre - 6583.45 * (1 + 3000. / C)) < 0.03, (gtype, centre)


def test_profile_annulus_starts_at_radmax(synthetic_cube):
    """The exterior annulus of the deconvolved density profile starts at the outer edge of the disk: RADMAX
    when it is the edge (RADMAX edge or taper), also inside the last ring (it started at the outer edge of
    the rings, so the truncated last ring took the emission between the two)."""
    from bbarolo.galmod import DiskModel
    from bbarolo.profile import profile_system
    model = DiskModel(synthetic_cube, galmodtype="ANALYTIC")
    r = _rings(4)                                    # radii 10-70", CENTRED edges up to 80"
    maps = np.ones((64, 64)), np.ones((64, 64)), np.ones((64, 64), bool)
    Rc = profile_system(model, r, ((0, 0), (64, 64)), *maps, beam_arcsec=30.)[3]
    assert np.isclose(Rc[-1], 80. + 30.)
    r.edge = 62.
    Rc = profile_system(model, r, ((0, 0), (64, 64)), *maps, beam_arcsec=30.)[3]
    assert np.isclose(Rc[-1], 62. + 30.)


@needs_data
def test_bayesian_smoothness_penalty_spacing(tmp_path):
    """The smoothness penalties (lambdas) are second derivatives on the radii of the rings, with VROT 0 at
    R = 0: a rotation curve rising linearly from the centre has no penalty, also with uneven rings (the first
    separation was used for all the rings, and the zero of VROT put one separation inside the first ring)."""
    from bbarolo.bayesian import BayesianBBarolo
    f = BayesianBBarolo(N2403, radii=[60., 120., 240., 300.], xpos=77, ypos=77, vsys=132.8, vrot=100, vdisp=10,
                        inc=60, pa=123.7, z0=10, mask="SEARCH", outfolder=str(tmp_path) + "/")
    f._setup(['vrot', 'vdisp_single'])
    th = np.array([30., 60., 120., 150., 10.])
    L0 = f._log_likelihood(th, sigma=1.)
    assert np.isclose(f._log_likelihood(th, sigma=1., lambdas=dict(f.lambdas, vrot=1.)), L0, rtol=0, atol=1e-9)
    th2 = np.array([30., 60., 90., 150., 10.])
    assert f._log_likelihood(th2, sigma=1., lambdas=dict(f.lambdas, vrot=1.)) < f._log_likelihood(th2, sigma=1.)
