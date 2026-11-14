"""Tests of SLITFIT (bbarolo.bayesian.slit): the reader of long-slit spectra, the preparation of the data, the
model seen through the slit, the likelihood, the fit and the task."""

import math
import os
import sys

import numpy as np
import pytest
from astropy.io import fits

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from bbarolo.bbutils import BBaroloError                                     # noqa: E402
from bbarolo.rings import Rings                                               # noqa: E402
from bbarolo.tasks import build_param                                         # noqa: E402

REST = [6564.61, 6549.86, 6585.27]
RELINT = [1., 0.117, 0.35]
Z = 0.05


def _vrot(R):
    return 200. * np.tanh(R / 2.)


def _vdisp(R):
    return 25. + 0. * R


def _brute_slit(npos=90, nz=300, pix=0.3, width=1.0, fwhm=1.2, slitangle=0., yoff=0., xpos=45.3, vsys=10., inc=60.,
                specres=1.2, dlam=0.45, asym=0.):
    """A slit (nz, npos) by the brute-force path: a full GALMOD cube (ANALYTIC), optionally asymmetric along the slit,
    convolved with the seeing in 2D (FFT) and summed over the rows of the slit; and its spectral axis."""
    from bbarolo.cube import Cube
    from bbarolo.header import Header
    from bbarolo.galmod import DiskModel
    R = (np.arange(8) + 0.5) * 1.2
    crval = REST[0] * (1 + Z) - dlam * nz / 2
    ny = 2 * int(12 / pix) + 1
    yc = (ny - 1) / 2.
    h = fits.Header()
    h["NAXIS"] = 3
    h["NAXIS1"], h["NAXIS2"], h["NAXIS3"] = npos, ny, nz
    h["CTYPE1"], h["CTYPE2"], h["CTYPE3"] = "RA---CAR", "DEC--CAR", "WAVE"
    h["CUNIT1"], h["CUNIT2"], h["CUNIT3"] = "deg", "deg", "Angstrom"
    h["CDELT1"] = h["CDELT2"] = pix / 3600.
    h["CDELT3"] = dlam
    h["CRPIX1"] = h["CRPIX2"] = h["CRPIX3"] = 1.
    h["CRVAL1"] = h["CRVAL2"] = 0.
    h["CRVAL3"] = crval
    h["BMAJ"] = h["BMIN"] = fwhm / 3600.
    h["BPA"] = 0.
    par = build_param(dict(RESTWAVE=REST, RELINT=RELINT, REDSHIFT=Z, SPECRES=specres))
    hd = Header(warning=False)
    hd.fromAstropy(h)
    hd.setRedshift(Z)
    hd.setWave0(REST[0])
    hd.setVelDef("relativistic")
    c = Cube.fromArray(np.broadcast_to(np.zeros(1, np.float32), (nz, ny, npos)), hd, par)
    r = Rings().setRings(radii=R, xpos=xpos, ypos=yc - yoff / pix, vsys=vsys, vrot=_vrot(R), vdisp=_vdisp(R), z0=pix,
                         dens=np.exp(-R / 3.), inc=inc, phi=90. - slitangle, radsep=R[1] - R[0])
    dm = DiskModel(c, par=par.gf, densflux=True, galmodtype="ANALYTIC", smooth=True)
    if asym:
        m0 = DiskModel(c, par=par.gf, densflux=True, galmodtype="ANALYTIC", smooth=False).array(r)
        m0 *= (1. + asym * np.tanh((np.arange(npos) - xpos) * pix / 2.))[None, None, :]
        m = dm.conv(m0)
    else:
        m = dm.array(r)
    yy = (np.arange(ny) - yc) * pix
    frac = np.clip(np.minimum(yy + pix / 2, width / 2) - np.maximum(yy - pix / 2, -width / 2), 0, None) / pix
    S = np.tensordot(frac, m, axes=(0, 1))
    return S * (100. / S.max()), crval, dlam, R


def _write_slit(fname, S, crval, dlam, pix=0.3, noise=4., cont=40., seed=1, transpose=False, unit="Angstrom",
                curv=0.):
    """The FITS file of a slit (nz, npos) with a linear continuum (curv: plus a parabola) and Gaussian noise: axis 1
    the wavelength, axis 2 the position (transpose: the other way round); unit 'm' writes the wavelengths in metres."""
    nz, npos = S.shape
    rng = np.random.default_rng(seed)
    lam = crval + dlam * np.arange(nz)
    prof = cont * np.exp(-0.5 * ((np.arange(npos) - npos / 2) * pix / 2.) ** 2)
    t = (lam[:, None] - lam.mean()) / (lam[-1] - lam[0])
    data = S + prof[None, :] * (1 + 0.3 * t + curv * t * t) + rng.normal(0, noise, S.shape)
    arr = data if transpose else data.T
    hdu = fits.PrimaryHDU(data=np.ascontiguousarray(arr).astype(np.float32))
    h = hdu.header
    sp, ps = (2, 1) if transpose else (1, 2)
    f = 1e-10 if unit == "m" else 1.
    h[f"CTYPE{sp}"], h[f"CUNIT{sp}"], h[f"CRPIX{sp}"], h[f"CRVAL{sp}"], h[f"CDELT{sp}"] = \
        "WAVE", unit, 1., crval * f, dlam * f
    h[f"CTYPE{ps}"], h[f"CUNIT{ps}"], h[f"CRPIX{ps}"], h[f"CRVAL{ps}"], h[f"CDELT{ps}"] = "PIXEL", "arcsec", 1., 0., pix
    h["OBJECT"] = "MOCKSLIT"
    hdu.writeto(fname, overwrite=True)
    return fname


@pytest.fixture(scope="module")
def mock_slit(tmp_path_factory):
    S, crval, dlam, R = _brute_slit()
    fn = str(tmp_path_factory.mktemp("slit") / "mock.fits")
    return _write_slit(fn, S, crval, dlam), R


def _fit(fname, **kw):
    from bbarolo.bayesian.slit import BayesianSlitFit
    prm = dict(slitwidth=1., beam=1.2, specres=1.2, inc=60., redshift=Z, restwave=REST, relint=RELINT, contsub=True)
    prm.update(kw)
    return BayesianSlitFit(fname, **prm)


def test_slit_reader(tmp_path):
    """The spectral axis is found by its CTYPE in either order of the axes; metres are read in Angstrom; blanks (0,
    NaN) are NaN; a position axis without a scale is an error."""
    from bbarolo.bayesian.slitdata import SlitData
    S = np.ones((20, 10))
    S[3, 4] = 0.
    a = SlitData.read(_write_slit(str(tmp_path / "a.fits"), S, 6800., 0.4, noise=0., cont=0.))
    b = SlitData.read(_write_slit(str(tmp_path / "b.fits"), S, 6800., 0.4, noise=0., cont=0., transpose=True,
                                  unit="m"))
    assert a.data.shape == b.data.shape == (20, 10) and not a.transposed and b.transposed
    assert b.spec.cunit == "Angstrom" and b.spec.crval == pytest.approx(6800.) and b.spec.cdelt == pytest.approx(0.4)
    assert np.isnan(a.data[3, 4]) and np.isnan(b.data[3, 4]) and a.pix == pytest.approx(0.3)
    hdr = fits.getheader(str(tmp_path / "a.fits"))
    del hdr["CUNIT2"]
    with pytest.raises(BBaroloError, match="no scale"):
        SlitData.read((np.ones((10, 20)), hdr))
    # the outputs keep the orientation and the axes of the file
    h = b.to_hdu(b.data, k0=2).header
    assert h["CTYPE2"] == "WAVE" and h["CRPIX2"] == pytest.approx(-1.) and h["CDELT1"] == pytest.approx(0.3)


def test_slit_projection_vs_brute_force():
    """The model seen through the slit (separable projection of the seeing and of the slit) agrees with a full cube
    convolved in 2D and summed over the rows of the slit, for a slit along the major axis and for one misaligned and
    off the centre."""
    from bbarolo.bayesian.slitdata import SlitData, SpecAxis, PreparedSlit
    from bbarolo.bayesian.slitmodel import SlitModel
    from bbarolo.bayesian.rings import _RingDict
    for slitangle, yoff in ((0., 0.), (20., 0.4)):
        S, crval, dlam, R = _brute_slit(slitangle=slitangle, yoff=yoff)
        nz, npos = S.shape
        sp = SpecAxis("wave", "WAVE", "Angstrom", crval, dlam, 1., nz)
        sp.rest = [r * (1 + Z) for r in REST]
        prep = PreparedSlit()
        prep.slit, prep.spec = SlitData(np.zeros((nz, npos)), sp, 0.3), sp
        par = build_param(dict(RESTWAVE=REST, RELINT=RELINT, REDSHIFT=Z, SPECRES=1.2))
        m = SlitModel(prep, par, (0, npos), 1.0, 1.2, yoffset=yoff)
        r = Rings().setRings(radii=R, xpos=45.3, ypos=0., vsys=10., vrot=_vrot(R), vdisp=_vdisp(R), z0=0.3,
                             dens=np.exp(-R / 3.), inc=60., phi=90. - slitangle, radsep=R[1] - R[0])
        P = m.model(m.rings(_RingDict.from_rings(r)))
        P *= 100. / P.max()
        assert np.abs(P - S).max() < 1.5                  # (the discretisation of the pixels: below 1.5% of the peak)
        assert P.sum() / S.sum() == pytest.approx(1., abs=1e-2)


def test_slit_basis_and_local_marginal(mock_slit):
    """The models of single rings add up to the model of the rings, and the LOCAL likelihood of a position is the
    integral of the likelihood over its amplitude (flat prior, non-negative)."""
    from scipy.integrate import quad
    fname, R = mock_slit
    f = _fit(fname, norm="LOCAL", radii=list(R))
    f._fit_options(None, {}, f._run_defaults())
    f._setup(f._run_defaults()["FREE"], verbose=False)
    th = f._initial_theta()
    M, rd, dens = f.model_of(th, details=True)
    rt = f._model.rings(rd)
    rt.dens = dens
    B = f._model.basis(rt)
    assert np.allclose(np.tensordot(dens, B, axes=(0, 0)), M, rtol=0, atol=1e-6 * M.max())
    # one position, by numerical integration
    j = int(np.argmax(M.sum(axis=0)))
    u = M[:, j] / M[:, j].sum()
    W, d = f._W[:, j], f._d[:, j]
    chi = lambda a: float(np.sum(W * (d - a * u) ** 2))
    a0 = max(float(np.sum(W * d * u) / np.sum(W * u * u)), 0.)
    num = math.log(quad(lambda a: math.exp(-0.5 * (chi(a) - chi(a0))), 0, 20 * a0 + 1, points=[a0], limit=200)[0]) \
        - 0.5 * chi(a0)
    g, b, c = np.sum(W * u * u), np.sum(W * d * u), np.sum(W * d * d)
    from scipy.special import log_ndtr
    z = b / math.sqrt(g)
    ana = -0.5 * c + 0.5 * z * z + 0.5 * math.log(2 * math.pi / g) + float(log_ndtr(z))
    assert ana == pytest.approx(num, abs=1e-6)


def test_slit_prepare_and_noise(mock_slit):
    """The preparation: the emission found along the slit, the continuum removed (CONTSUB), the noise map close to the
    noise of the mock (pooled channels) and the noise uncorrelated."""
    fname, R = mock_slit
    f = _fit(fname)
    p = f._load()
    assert p.lines["ok"].sum() > 30
    assert np.nanmedian(p.noise[p.valid]) == pytest.approx(4., rel=0.08)
    assert np.percentile(p.noise[p.valid], 90) / np.percentile(p.noise[p.valid], 10) < 1.4
    assert abs(np.nanmedian(p.data[p.contchan])) < 0.5
    assert p.corr["likefactor"] > 0.6
    ini = f.initial
    assert ini["slitangle"] == 0. and abs(ini["xpos"] - 45.3) < 2 and abs(ini["vsys"] - 10.) < 15


def test_slit_fit_optimise(mock_slit):
    """METHOD GLOBAL (the default) on the mock recovers the rotation curve, the dispersion, VSYS and XPOS within the errors of the
    Laplace approximation (and the noise map: reduced chi-square close to 1)."""
    fname, R = mock_slit
    f = _fit(fname, radii=list(R))
    f.run(threads=4, verbose=False)
    assert f._method == "optimize"
    t = f.ring_table()
    pull = (np.asarray(t["VROT"]) - _vrot(R)) / (0.5 * (t["E_VROT_LO"] + t["E_VROT_HI"]))
    assert np.sqrt(np.mean(np.asarray(pull[1:]) ** 2)) < 2.5
    assert np.all(np.abs(np.asarray(t["VDISP"])[:6] - 25.) < 8)
    i, k = f.freepar_names.index("vsys"), f.freepar_names.index("xpos")
    assert abs(f.params[i] - 10.) < 2 and abs(f.params[k] - 45.3) < 0.3
    assert 0.8 < f.best["chi2_reduced"] < 1.2
    # the PV plot: the rings where they cross the slit lie on the line-of-sight velocities of the rings, with the
    # errors of VROT projected; the systemic velocity of each line
    from bbarolo.bayesian.slit_outputs import ring_points, vsys_values, los_curves
    s, w, elo, ehi = ring_points(f)
    assert s.size == 2 * len(R) and np.all(np.diff(s) > 0)
    cs, cw = los_curves(f, npts=4000)[0][0]
    assert np.allclose(w, np.interp(s, cs, cw), atol=2e-3)
    sp = f.prep.spec
    dwdv = abs(float(sp.value(1., 0)) - float(sp.value(0., 0)))
    sini = math.sin(math.radians(60.))
    assert np.allclose(np.sort(np.r_[elo, ehi]),
                       np.sort(np.r_[t["E_VROT_LO"], t["E_VROT_HI"], t["E_VROT_LO"], t["E_VROT_HI"]]) * sini * dwdv,
                       rtol=1e-3)
    vs = vsys_values(f)
    assert len(vs) == 3 and vs[0] == pytest.approx(float(sp.value(f.params[i], 0)))


def test_slit_task_parfile(mock_slit, tmp_path):
    """The task in a parameter file (BBarolo -p) runs the fit and writes the rings, the FITS files of the slit and the
    plots with their scripts."""
    import bbarolo
    fname, R = mock_slit
    par = tmp_path / "slit.par"
    par.write_text(f"FITSFILE {fname}\nSLITFIT true\nSLITWIDTH 1\nBEAM 1.2\nSPECRES 1.2\nINC 60\n"
                   f"REDSHIFT {Z}\nRESTWAVE {' '.join(map(str, REST))}\nRELINT {' '.join(map(str, RELINT))}\n"
                   f"CONTSUB true\nNORM LOCAL\nFREE VROT VDISP_SINGLE VSYS XPOS\nMETHOD GLOBAL\n"
                   f"SAMPLEROPTS errors=false\nTHREADS 4\nPLOTS 1\nVERBOSE 0\nOUTFOLDER {tmp_path}/out\n")
    res = bbarolo.run(str(par))
    out = tmp_path / "out"
    for suffix in ("_slitfit_rings.txt", "_slit_data.fits", "_slit_model.fits", "_slit_resid.fits",
                   "_slit_noise.fits", "_slit_cont.fits", "_slitfit_pv.pdf", "_slitfit_parameters.pdf"):
        assert (out / ("MOCKSLIT" + suffix)).is_file(), suffix
    assert (out / "plotscripts" / "plot_slitfit_pv.py").is_file()
    assert len(res["SLITFIT"].rings) == len(R) and res["SLITFIT"].method == "GLOBAL"
    rings = np.loadtxt(out / "MOCKSLIT_slitfit_rings.txt")
    assert rings.shape[0] == len(R) and np.all(np.abs(rings[:, 2] - rings[0, 2]) < 1e-6)   # (VDISP_SINGLE)
    from astropy.io import fits
    assert fits.getheader(out / "MOCKSLIT_slit_model.fits")["BBTASK"] == "SLITFIT"
    # SlitFit: the same fit in memory, also of a slit in memory (an HDU)
    from bbarolo import SlitFit, BayesianSlitFit
    hdu = fits.open(fname)[0]
    kw = dict(slitwidth=1, beam=1.2, specres=1.2, inc=60, redshift=Z, restwave=list(REST), relint=list(RELINT),
              contsub=True, norm="LOCAL", free="VROT VDISP_SINGLE VSYS XPOS", sampleropts="errors=false", threads=4,
              verbose=False)
    for slit in (fname, hdu):
        r = SlitFit(slit, **kw).run()
        assert r.method == "GLOBAL" and isinstance(r.fitter, BayesianSlitFit) and len(r.rings) == len(R)
        assert np.allclose(r.rings["VROT"], res["SLITFIT"].rings["VROT"])
        assert r.model.data.shape == fits.getdata(out / "MOCKSLIT_slit_model.fits").shape


def test_slit_contsub(tmp_path):
    """CONTSUB (the CONTSUB of datacubes, at each position of the slit): a polynomial of degree CONTORDER, fitted away
    from the lines and from the channels of EXCLWIND, which are not continuum (nor in the noise)."""
    S, crval, dlam, R = _brute_slit()
    fn = _write_slit(str(tmp_path / "curved.fits"), S, crval, dlam, curv=3.)
    off = {}
    for order in (1, 2):
        p = _fit(fn, contorder=order)._load()
        j = int(np.argmax(p.continuum.sum(axis=0)))     # (the continuum channels at the brightest positions)
        off[order] = float(np.mean(np.abs(np.nanmean(p.data[p.contchan][:, j - 3:j + 4], axis=1))))
    assert off[2] < 1.5 < 2.2 < off[1]                   # (a straight line leaves the curvature; noise: ~1.2)
    f = _fit(fn, contorder=2, exclwind="10:30 35")
    p = f._load()
    ch = np.arange(p.k0, p.k1)
    assert not np.any(p.contchan[(ch >= 10) & (ch <= 30) | (ch == 35)])
    assert np.any(p.contchan[ch < 10]) or p.k0 >= 10
    with pytest.raises(BBaroloError, match="EXCLWIND"):
        _fit(fn, exclwind="10-30")._load()


def test_slit_smooth(mock_slit):
    """LIKEOPTS smooth (as 3DFIT with METHOD GLOBAL or BAYESIAN): the smoothness prior of the parameters free in each
    ring, over one seeing FWHM, is subtracted from log L; the keys are the ring parameters of the slit."""
    from bbarolo.bayesian.slit import slit_likeopts
    from bbarolo.bayesian.common import smoothness_penalty
    assert slit_likeopts("smooth=(vrot=5, disp=3) noisecorr=1") == \
        {"smooth": {"vrot": 5., "vdisp": 3.}, "noisecorr": 1.}
    with pytest.raises(ValueError, match="ring parameter"):
        slit_likeopts("smooth=(inc=10)")
    with pytest.raises(ValueError, match="unknown"):
        slit_likeopts("lambdas=(vrot=10)")
    fname, R = mock_slit
    f = _fit(fname, radii=list(R), free="VROT VDISP_SINGLE VSYS XPOS")
    f._fit_options(None, {}, f._run_defaults())
    f._setup(f._run_defaults()["FREE"], verbose=False)
    th = f._initial_theta()
    th[f.freepar_idx["vrot"]] += np.random.default_rng(3).normal(0, 20, len(R))
    l0 = f.log_likelihood(th)
    f._fit_options(None, {"smooth": {"vrot": 5., "vdisp": 1.}}, f._run_defaults())
    assert f._smooth_used() == {"vrot": 5.}                 # (VDISP_SINGLE: one value, no prior)
    rad = np.asarray(f._fit0.r["radii"], float)
    pen = smoothness_penalty(rad, th[f.freepar_idx["vrot"]], 5., f._beam, True)
    assert pen > 0 and f.log_likelihood(th) == pytest.approx(l0 - pen)


def test_slit_xpos_smooth(mock_slit):
    """The model is smooth in XPOS: it is built at the whole pixel of the template below XPOS and the fraction of a
    pixel moves the seeing along the slit. (The clouds of GALMOD fall in their nearest pixel: a model built at any XPOS
    was a staircase of tiny steps in XPOS, whose numerical gradient was noise, and L-BFGS-B stopped near the start,
    ABNORMAL.) The model moves by the step of XPOS and is continuous at a whole pixel."""
    fname, R = mock_slit
    f = _fit(fname, radii=list(R), free="VROT VDISP_SINGLE VSYS XPOS")
    f._fit_options(None, {}, f._run_defaults())
    f._setup(f._run_defaults()["FREE"], verbose=False)
    th = f._initial_theta()
    ix = f.freepar_idx["xpos"][0]
    th[ix] += 0.7                                           # (off the best XPOS: a clear slope)

    def at(dx):
        t = th.copy()
        t[ix] += dx
        return t
    L = lambda dx: f.log_likelihood(at(dx))
    g = [(L(h) - L(-h)) / (2 * h) for h in (0.1, 0.01, 0.001)]
    assert abs(g[0]) > 1 and np.allclose(g, g[0], rtol=0.02), g
    steps = np.diff([L(dx) for dx in np.linspace(-0.02, 0.02, 21)])
    assert np.all(np.sign(steps) == np.sign(g[0])), steps

    def model(dx):
        rd = f._update_rings(f._work, at(dx))
        rt = f._model.rings(rd)
        rt.dens = np.ones(rt.nr)
        return f._model.model(rt).sum(axis=0)
    cen = lambda M: float(np.sum(M * np.arange(M.size)) / np.sum(M))
    assert cen(model(0.3)) - cen(model(0.)) == pytest.approx(0.3, abs=1e-3)
    xt = float(f._model.xpos_template(at(0.)[ix]))
    dn = (math.ceil(xt) - xt) / f._model.f                 # (to the next whole pixel of the template)
    assert np.allclose(model(dn - 1e-9), model(dn + 1e-9), rtol=0, atol=1e-6 * model(dn).max())


def test_slit_parameter_listing():
    """The parameters of SLITFIT in the lists of BBarolo -d SLITFIT (the defaults of the task: FREE,
    NORM AZIM) and of a parameter file (VERBOSE 2: the values given; CONTSUB is that of the slit, not of a cube)."""
    import io
    from bbarolo.params import Param
    s = io.StringIO()
    Param().printDefaults(s, "SLITFIT")
    txt = s.getvalue()
    for k in ("[SLITWIDTH]", "[SLITANGLE]", "[CONTSUB]", "[CONTORDER]", "[EXCLWIND]", "[SKYRESID]", "[MASKWAVE]",
              "[WAVERANGE]", "[SPECRES]", "[LIKEOPTS]", "[METHOD]", "[MINIMISER]", "[SAMPLER]"):
        assert k in txt, k
    assert "VROT VDISP VSYS XPOS" in txt and "[NORM]  =  AZIM" in txt.replace(".", "")
    p = Param()
    p.readParamString("FITSFILE a.fits\nSLITFIT true\nINC 60\nSLITWIDTH 1.5\nBEAM 1.5\nCONTSUB true\n"
                      "CONTORDER 2\nNORM LOCAL\nMASKWAVE 7150 7152\n")
    txt = str(p)
    assert "Width of the slit (arcsec)" in txt and "1.5" in txt and "7150 7152" in txt
    assert "[NORM]  =  LOCAL" in txt.replace(".", "") and "datacube" not in txt


def test_slit_keywords():
    """The checks of the task: SLITFIT is gone (an error naming SLITFIT), and SLITFIT needs SLITWIDTH, a
    circular BEAM and INC, and runs alone, with METHOD GLOBAL or BAYESIAN; the geometry cannot be free."""
    from bbarolo.keywords import check_removed
    from bbarolo.params import Param
    from bbarolo.bayesian.slit import slit_freepar
    with pytest.raises(BBaroloError, match="BAYESIANSLITFIT is no longer a parameter: SLITFIT"):
        check_removed("BAYESIANSLITFIT")
    p = Param()
    p.readParamString("FITSFILE a.fits\nSLITFIT true\nINC 60\nSLITWIDTH 1\nBEAM 1.5\n")
    assert p.checkPars(quiet=True) and p.gf.METHOD == "GLOBAL"
    for extra, ok in (("BEAM 1.5 1.0 0\n", False), ("SLITWIDTH 0\n", False), ("3DFIT true\n", False),
                      ("SLITANGLE abc\n", False), ("NORM LOCAL\n", True), ("MASKWAVE 1 2 3\n", False),
                      ("CONTSUB true\nCONTORDER 2\nEXCLWIND 10:20\n", True),
                      ("LIKEOPTS smooth=(vrot=5)\n", True), ("LIKEOPTS smooth=(xpos=1)\n", False),
                      ("LIKEOPTS lambdas=(vrot=100)\n", False), ("METHOD CLASSIC\n", False),
                      ("METHOD bayesian\n", True), ("MINIMISER simplex\n", True), ("MINIMISER foo\n", False)):
        p = Param()
        p.readParamString("FITSFILE a.fits\nSLITFIT true\nINC 60\nSLITWIDTH 1\nBEAM 1.5\n" + extra)
        assert p.checkPars(quiet=True) == ok, extra
    assert slit_freepar("VROT VDISP_SINGLE VSYS_SINGLE XPOS") == ["vrot", "vdisp_single", "vsys", "xpos"]
    with pytest.raises(ValueError, match="fixed"):
        slit_freepar("VROT INC")
