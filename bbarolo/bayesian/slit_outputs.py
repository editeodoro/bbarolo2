"""
The outputs of SLITFIT (BayesianSlitFit.write), in the output folder, named after the OBJECT of the slit:

  <prefix>_slitfit_rings.txt          the rings of the best fit with the 1-sigma errors of the free parameters
  <prefix>_slit_data.fits             the spectral window of the fit (continuum and residual sky subtracted)
  <prefix>_slit_model.fits            the best model (the positions of the likelihood; NaN elsewhere)
  <prefix>_slit_resid.fits            data - model
  <prefix>_slit_noise.fits            the noise map
  <prefix>_slit_cont.fits             the continuum subtracted (CONTSUB)
  <prefix>_slitfit_<method>_results.npy   the samples of the posterior and the options of the fit
  <prefix>_slitfit_pv.pdf, <prefix>_slitfit_parameters.pdf, <prefix>_slitfit_corner.pdf   the plots (PLOTS > 0),
                                      with their scripts in plotscripts/
"""

import os
import math
import numpy as np

from ..bbutils import mkdirp

_COLS = (("RAD(arcs)", "radii"), ("VROT(km/s)", "vrot"), ("VDISP(km/s)", "vdisp"), ("VRAD(km/s)", "vrad"),
         ("VSYS(km/s)", "vsys"), ("XPOS(pix)", "xpos"), ("INC(deg)", "inc"), ("SLITANGLE(deg)", "slitangle"),
         ("Z0(arcs)", "z0"), ("DENS", "dens"))


def rings_text(f):
    """The text of <prefix>_slitfit_rings.txt: one line per ring, a column per parameter, then the lower and upper
    1-sigma errors of the free ones (E_<NAME>_LO, E_<NAME>_HI; 0 for the fixed ones)."""
    t = f.ring_table()
    names = [n for n, _ in _COLS]
    keys = [k for _, k in _COLS]
    cols = [np.asarray(t[k.upper()] if k != "radii" else t["RADII"], float) for k in keys]
    for n, k in _COLS[1:6]:
        K = k.upper()
        names += [f"E_{K}_LO", f"E_{K}_HI"]
        lo = np.asarray(t[f"E_{K}_LO"], float) if f"E_{K}_LO" in t.colnames else np.zeros(len(t))
        hi = np.asarray(t[f"E_{K}_HI"], float) if f"E_{K}_HI" in t.colnames else np.zeros(len(t))
        cols += [lo, hi]
    out = ["#" + "".join(f"{n:>15s}" for n in names)]
    for i in range(len(t)):
        out.append(" " + "".join(f"{c[i]:15.6g}" for c in cols))
    return "\n".join(out) + "\n"


def los_curves(f, rings=None, npts=400):
    """The line-of-sight velocities of the best rings along the slit (the plane of the disk, without the seeing), in
    the spectral values of each line: a list of (offsets (arcsec, from XPOS), spectral values), and the velocities."""
    rd = rings if rings is not None else f.best["rings"]
    r = rd.r
    ini = f.initial
    radii = np.asarray(r["radii"], float)
    inc = math.radians(float(np.median(r["inc"])))
    d = math.radians(ini["slitangle"])
    yoff = ini["yoffset"]
    p0, p1 = f.region
    pix = f.prep.slit.pix
    xpos = float(np.median(r["xpos"]))
    s = np.linspace((p0 - xpos) * pix, (p1 - 1 - xpos) * pix, npts)
    xm = s * math.cos(d) + yoff * math.sin(d)
    ym = -s * math.sin(d) + yoff * math.cos(d)
    X, Y = xm, ym / math.cos(inc)
    R = np.hypot(X, Y)
    with np.errstate(all="ignore"):
        caz, saz = np.where(R > 0, X / R, 0.), np.where(R > 0, Y / R, 0.)
    vr = np.interp(R, np.r_[0., radii], np.r_[0., np.asarray(r["vrot"], float)])
    vd = np.interp(R, np.r_[0., radii], np.r_[0., np.asarray(r["vrad"], float)])
    v = float(np.median(r["vsys"])) + (vr * caz + vd * saz) * math.sin(inc)
    sp = f.prep.spec
    curves = [(s, sp.value(v, l)) for l in range(len(sp.rest) if sp.rest and sp.rest[0] is not None else 1)]
    return curves, s, v


def ring_points(f):
    """The rotation velocities of the best rings where they cross the slit, in the spectral values of the first line:
    (offsets (arcsec, from XPOS), spectral values, lower errors, upper errors), sorted by offset, from the 1-sigma
    errors of VROT (0 if VROT is fixed). A ring crosses the slit at two offsets (one if it only touches it, none if the slit is beyond
    it, YPOS)."""
    rd = f.best["rings"]
    r = rd.r
    ini = f.initial
    radii = np.asarray(r["radii"], float)
    vrot, vrad = np.asarray(r["vrot"], float), np.asarray(r["vrad"], float)
    t = f.ring_table()
    elo = np.asarray(t["E_VROT_LO"], float) if "E_VROT_LO" in t.colnames else np.zeros(radii.size)
    ehi = np.asarray(t["E_VROT_HI"], float) if "E_VROT_HI" in t.colnames else np.zeros(radii.size)
    sini = math.sin(math.radians(float(np.median(r["inc"]))))
    ci = math.cos(math.radians(float(np.median(r["inc"]))))
    d = math.radians(ini["slitangle"])
    yoff = ini["yoffset"]
    vsys = float(np.median(r["vsys"]))
    # R^2 = a s^2 + 2 b s + c along the slit (see los_curves)
    a = math.cos(d) ** 2 + math.sin(d) ** 2 / ci ** 2
    b = math.sin(d) * math.cos(d) * yoff * (1. - 1. / ci ** 2)
    c = yoff ** 2 * (math.sin(d) ** 2 + math.cos(d) ** 2 / ci ** 2)
    out = [[], [], [], []]
    sp = f.prep.spec
    for i, R in enumerate(radii):
        disc = b * b - a * (c - R * R)
        if disc < 0:
            continue
        for s_ in sorted({(-b - math.sqrt(disc)) / a, (-b + math.sqrt(disc)) / a}):
            X = s_ * math.cos(d) + yoff * math.sin(d)
            Y = (-s_ * math.sin(d) + yoff * math.cos(d)) / ci
            caz, saz = (X / R, Y / R) if R > 0 else (0., 0.)
            v = vsys + (vrot[i] * caz + vrad[i] * saz) * sini
            lo, hi = (elo[i], ehi[i]) if caz >= 0 else (ehi[i], elo[i])
            w, wlo, whi = (float(x) for x in sp.value(np.array([v, v - lo * abs(caz) * sini,
                                                                v + hi * abs(caz) * sini]), 0))
            out[0].append(s_)
            out[1].append(w)
            out[2].append(abs(w - wlo))
            out[3].append(abs(whi - w))
    o = np.argsort(out[0])
    return tuple(np.asarray(x, float)[o] for x in out)


def vsys_values(f):
    """The spectral values of the systemic velocity of the best fit, one per line."""
    sp = f.prep.spec
    vsys = float(np.median(f.best["rings"].r["vsys"]))
    nl = len(sp.rest) if sp.rest and sp.rest[0] is not None else 1
    return [float(sp.value(vsys, l)) for l in range(nl)]


def pv_inputs(f):
    """The arguments of plots.slitfit.plot_slit_pv for the best fit of f (positions of the likelihood)."""
    prep = f.prep
    p0, p1 = f.region
    pix = prep.slit.pix
    data = prep.data[:, p0:p1]
    model = np.asarray(f.best["model"], float)
    resid = (data - model) / (prep.noise[:, p0:p1] * f.best["scale"])
    xpos = float(np.median(f.best["rings"].r["xpos"]))
    offsets = (np.arange(p0, p1) - xpos) * pix
    curves, _, _ = los_curves(f)
    L = prep.lines
    ok = L["ok"][p0:p1]
    pts = ((np.arange(p0, p1)[ok] - xpos) * pix, prep.spec.value(L["v"][p0:p1][ok], 0)) if np.any(ok) else None
    sp = prep.spec
    return dict(data=data, model=model, resid=resid, offsets=offsets, spec=sp.values(), specunit=sp.cunit,
                curves=curves, points=pts, rings=ring_points(f), vsys=vsys_values(f), gname=prep.slit.name,
                lines=_lines_text(f))


def _lines_text(f):
    sp = f.prep.spec
    if sp.kind == "velo" or not sp.rest:
        return ""
    gf = f._par.gf
    rest = gf.RESTWAVE if sp.kind == "wave" else gf.RESTFREQ
    return "lines at " + ", ".join(f"{x:.6g}" for x in rest if x > 0) + f" {sp.cunit}, z = {f._par.getRedshift():g}"


def _scalar_text(f, key, fmt):
    if key in f.freepar_idx:
        i = int(f.freepar_idx[key][0])
        lo, med, hi = f.quantiles[i]
        val = f.params[i]
        return f"{val:{fmt}} (-{med - lo:.2g}, +{hi - med:.2g})"
    return f"{float(np.median(f.best['rings'].r[key])):{fmt}} (fixed)"


def parameter_inputs(f):
    """The arguments of plots.slitfit.plot_slit_parameters for the best fit of f."""
    t = f.ring_table()
    rings = {"rad": np.asarray(t["RADII"], float)}
    for k in ("VROT", "VDISP", "VRAD", "DENS"):
        rings[k.lower()] = np.asarray(t[k], float)
        if f"E_{k}_LO" in t.colnames:
            rings[f"e_{k.lower()}_lo"] = np.asarray(t[f"E_{k}_LO"], float)
            rings[f"e_{k.lower()}_hi"] = np.asarray(t[f"E_{k}_HI"], float)
    scal = {"VSYS": _scalar_text(f, "vsys", ".2f") + " km/s", "XPOS": _scalar_text(f, "xpos", ".2f") + " pix",
            "INC": f"{f.initial['inc']:g}", "SLITANGLE": f"{f.initial['slitangle']:g}",
            "NORM": f._norm()}
    return dict(rings=rings, scalars=scal, gname=f.prep.slit.name)


def write_outputs(f, outfolder=None, plots=None, verbose=True):
    """Write the outputs of the fit f (see the module documentation) in outfolder (default: OUTFOLDER of f); plots
    (default PLOTS): > 0 also the plots and their scripts, < 0 only the rings and the results. Returns the files
    written."""
    from ..tasks import write_hdu
    of = outfolder if outfolder is not None else f._outfolder()
    of = os.path.expanduser(str(of))
    of = of if of.endswith("/") else of + "/"
    mkdirp(of)
    plots = f._par.plots if plots is None else int(plots)
    prep = f.prep
    slit = prep.slit
    pre = of + slit.name
    files = []
    fn = pre + "_slitfit_rings.txt"
    with open(fn, "w") as fh:
        fh.write(rings_text(f))
    files.append(fn)
    # the results
    fn = pre + f"_slitfit_{f._method}_results.npy"
    np.save(fn, dict(samples=f.samples, weights=f.weights, freepar_names=list(f.freepar_names), params=f.params,
                     quantiles=f.quantiles, priors=dict(f.priors), like_kwargs=dict(f.like_kwargs),
                     method=f._method, region=f.region, initial={k: v for k, v in f.initial.items()}),
            allow_pickle=True)
    files.append(fn)
    if plots >= 0:
        model = model_array(f)
        extra = {"BBTASK": "SLITFIT"}
        for suffix, arr in (("data", prep.data), ("model", model), ("resid", prep.data - model),
                            ("noise", prep.noise * f.best["scale"])):
            fn = pre + f"_slit_{suffix}.fits"
            write_hdu(slit.to_hdu(arr, k0=prep.k0, extra=extra), fn)
            files.append(fn)
        if f._par.contsub:
            fn = pre + "_slit_cont.fits"
            write_hdu(slit.to_hdu(prep.continuum, k0=prep.k0, extra=extra), fn)
            files.append(fn)
    if plots > 0:
        files += _write_plots(f, of)
    if verbose:
        print(f" SLITFIT: {len(files)} files written in {of}")
    return files


def model_array(f):
    """The best model of the slit f on the grid of its prepared data (NaN outside the region of the fit)."""
    p0, p1 = f.region
    model = np.full(f.prep.data.shape, np.nan)
    model[:, p0:p1] = f.best["model"]
    return model


def _write_plots(f, of):
    import matplotlib
    import matplotlib.pyplot as plt
    from ..plots import slitfit as sp
    from ..plots.scripts import script, plot_utils_text
    from .. import plots
    files = []
    pre = of + f.prep.slit.name
    try:
        fig = sp.plot_slit_pv(**pv_inputs(f))
        fig.savefig(pre + "_slitfit_pv.pdf", bbox_inches="tight")
        plt.close(fig)
        files.append(pre + "_slitfit_pv.pdf")
        fig = sp.plot_slit_parameters(**parameter_inputs(f))
        fig.savefig(pre + "_slitfit_parameters.pdf", bbox_inches="tight")
        plt.close(fig)
        files.append(pre + "_slitfit_parameters.pdf")
    except Exception as e:                                  # (the plots must not lose the other outputs)
        print(f" SLITFIT WARNING: the plots failed: {e}")
    if np.all(np.ptp(np.asarray(f.samples_equal), axis=0) > 0):   # (no errors, METHOD GLOBAL, errors=false: none)
        try:
            import corner                                       # noqa: F401
            fig = f.corner_plot()
            fig.savefig(pre + "_slitfit_corner.pdf", bbox_inches="tight")
            plt.close(fig)
            files.append(pre + "_slitfit_corner.pdf")
        except ImportError:
            pass
    # the scripts
    sd = of + "plotscripts/"
    mkdirp(sd)
    with open(sd + "plot_utils.py", "w") as fh:
        fh.write(plot_utils_text())
    files.append(sd + "plot_utils.py")
    pin = pv_inputs(f)
    curves = [(np.asarray(c[0]), np.asarray(c[1])) for c in pin["curves"]]
    settings = [("outfolder", of, ""), ("outprefix", f.prep.slit.name, ""),
                ("region", tuple(int(x) for x in f.region), "the positions of the likelihood"),
                ("k0", int(f.prep.k0), "the first channel of the window in the slit"),
                ("xpos", float(np.median(f.best["rings"].r["xpos"])), "the centre along the slit (pixels)"),
                ("pix", float(f.prep.slit.pix), "arcsec per position"),
                ("transposed", bool(f.prep.slit.transposed), "the spectral axis is NAXIS2"),
                ("specunit", pin["specunit"], ""), ("gname", pin["gname"], ""), ("lines", pin["lines"], "")]
    from ..plots.scripts import Code
    arr = lambda x, nd: f"np.array({[float(v) for v in np.round(np.asarray(x, float), nd)]})"
    settings.append(("curves", Code("[" + ", ".join(f"({arr(c[0], 5)}, {arr(c[1], 6)})" for c in curves) + "]"),
                     "the velocities of the rings along the slit, per line"))
    pts = pin["points"]
    settings.append(("points", Code("None" if pts is None else f"({arr(pts[0], 5)}, {arr(pts[1], 6)})"),
                     "the velocities of the Gaussian fits along the slit (first line)"))
    rp = pin["rings"]
    settings.append(("rings", Code("(" + ", ".join(arr(x, 6) for x in rp) + ")"),
                     "the rotation velocities of the rings on the slit: offsets, values, errors (first line)"))
    settings.append(("vsys", [round(v, 6) for v in pin["vsys"]], "the systemic velocity, per line"))
    body = ("def read(suffix):\n"
            "    with fits.open(outfolder + outprefix + '_slit_' + suffix + '.fits') as h:\n"
            "        a, hd = h[0].data.astype(float), h[0].header\n"
            "    a = a if transposed else a.T\n"
            "    return a, hd\n\n"
            "data, hd = read('data')\nmodel, _ = read('model')\nnoise, _ = read('noise')\n"
            "sax = 2 if transposed else 1\n"
            "spec = hd[f'CRVAL{sax}'] + (np.arange(data.shape[0]) + 1 - hd[f'CRPIX{sax}']) * hd[f'CDELT{sax}']\n"
            "p0, p1 = region\n"
            "data, model, noise = data[:, p0:p1], model[:, p0:p1], noise[:, p0:p1]\n"
            "offsets = (np.arange(p0, p1) - xpos) * pix\n"
            "fig = plot_slit_pv(data, model, (data - model) / noise, offsets=offsets, spec=spec, specunit=specunit,\n"
            "                   curves=curves, points=points, rings=rings, vsys=vsys, gname=gname, lines=lines)\n"
            "fig.savefig(outfolder + outprefix + '_slitfit_pv.pdf', bbox_inches='tight')\n")
    fn = sd + "plot_slitfit_pv.py"
    with open(fn, "w") as fh:
        fh.write(script("This script plots the slit, its best model and the residuals", [sp.plot_slit_pv],
                        settings, body))
    files.append(fn)
    pin = parameter_inputs(f)
    settings = [("outfolder", of, ""), ("outprefix", f.prep.slit.name, ""), ("gname", pin["gname"], ""),
                ("scalars", pin["scalars"], "the single values of the fit")]
    body = ("rings = read_slit_rings(outfolder + outprefix + '_slitfit_rings.txt')\n"
            "for k in ('vrot', 'vdisp', 'vrad', 'vsys', 'xpos'):\n"
            "    if not np.any(rings.get(f'e_{k}_hi', 0)):\n"
            "        rings.pop(f'e_{k}_lo', None), rings.pop(f'e_{k}_hi', None)\n"
            "fig = plot_slit_parameters(rings, scalars=scalars, gname=gname)\n"
            "fig.savefig(outfolder + outprefix + '_slitfit_parameters.pdf', bbox_inches='tight')\n")
    fn = sd + "plot_slitfit_parameters.py"
    with open(fn, "w") as fh:
        fh.write(script("This script plots the rings of the best fit", [sp.read_slit_rings, sp.plot_slit_parameters],
                        settings, body))
    files.append(fn)
    return files
