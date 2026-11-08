#!/usr/bin/env python3
"""
Coverage test of the credible intervals of BayesianBBarolo on simulated galaxies.

Flat disks of the geometry experiment (run_suite.py, 30 <= INC < 75 deg) are simulated with
GALMOD (ANALYTIC) and noise convolved with the beam, optionally Hanning-smoothed along the
spectral axis (noise only), and fitted with BayesianBBarolo (nautilus), free parameters VROT
per ring, VDISP, INC and PA (centre, VSYS and Z0 fixed at the true values), with several
likelihood options ("configurations"):

  dilated_deconv          the defaults: noise correlation measured from the data, data not
                          masked in the mask dilated by 1 beam and 3 channels, deconvolved
                          density profile
  dilated_deconv_fracerr  + the fractional model error 'fracerr' as a free parameter
  dilated_deconv_sigma    + the noise 'sigma' as a free parameter
  none_deconv             all the voxels of the likelihood region (likemask='none')
  dilated_truedens        diagnostic: DENS fixed to the true profile (scaled to the flux of the
                          simulation) instead of the deconvolved profile

The results file also holds configurations of likelihood options that were removed on
3 November 2026, shown by --summary but no longer runnable: the data times the mask (zero
outside it) and the density profile of the observed total intensity map (Ellprof), as in the
original module, with log-likelihood x 1/(beam area) ('beam', 'fracerr' = beam + fracerr,
'sigma' = beam + sigma) or x 1/(measured noise correlation volume) ('measured', 'both' =
measured + fracerr); the data in the dilated mask with the observed profile ('dilated',
'dilated_fracerr', 'dilated_sigma', 'dilated_sigma_fracerr', 'none'); and the data times the
mask with the first version of the deconvolved profile ('deconv_v1'; 'dilated_deconv_v1' and
'dilated_deconv_fracerr_v1' are the first version of the current defaults).

Kinds of galaxies (--kind):

  flat    the flat disks above (the default)
  warp    the warped disks of the warp experiment (INC and PA change linearly beyond 0.3-0.6
          RMAX by up to 20 and 40 deg), with INC and PA fitted in each ring
  thick   flat disks three times thicker (Z0 = 0.15 RMAX)
  lag     thick disks with a vertical gradient of the rotation velocity, DVDZ = -0.1 VFLAT / Z0
          (10% slower at one scale height)

Z0 and DVDZ are fixed at their true values in the fits. --galmodtype MC fits Monte-Carlo models
to the ANALYTIC simulations, and --fit3d also fits 3DFIT to the same cubes (free VROT, VDISP, INC
and PA in each ring, one stage: its errors of the medians are shown with those of the posterior).

For each fit the 16/50/84% quantiles of the posterior are stored with the true values. The
summary gives, per configuration and parameter, the coverage (fraction of the true values
within the 68% interval, ideally 0.68) and the pulls (median - true) / (half width of the
68% interval), whose scatter should be ~1, and the errors of the medians (median - true).

    python bayes_coverage.py --ngal 12 --procs 2                  # run (resumable)
    python bayes_coverage.py --kind warp --fit3d --ngal 12 --procs 12
    python bayes_coverage.py --summary                            # tables
"""

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")             # one thread per process (several fits run in parallel)
import sys
import json
import time
import argparse
import tempfile
import warnings
import numpy as np
from multiprocessing import get_context

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import run_suite as rs                                 # noqa: E402

CONFIGS = {
    "dilated_deconv": dict(noisecorr="measured", fracerr=False, sigma=False, likemask="dilated"),
    "dilated_deconv_fracerr": dict(noisecorr="measured", fracerr=True, sigma=False, likemask="dilated"),
    "dilated_deconv_sigma": dict(noisecorr="measured", fracerr=False, sigma=True, likemask="dilated"),
    "none_deconv": dict(noisecorr="measured", fracerr=False, sigma=False, likemask="none"),
    "dilated_truedens": dict(noisecorr="measured", fracerr=False, sigma=False, likemask="dilated", truedens=True),
}
# configurations of options removed on 3 November 2026 (in the results file, shown by --summary)
HISTORICAL = ["beam", "measured", "fracerr", "both", "sigma", "dilated", "dilated_fracerr", "dilated_sigma",
              "dilated_sigma_fracerr", "none", "deconv_v1", "dilated_deconv_v1", "dilated_deconv_fracerr_v1"]


KINDS = ("flat", "warp", "thick", "lag")


def vertical(g, kind):
    """Scale height (arcsec) and vertical gradient of VROT (km/s/arcsec) of a galaxy of a kind."""
    z0 = 3. * rs.Z0 if kind in ("thick", "lag") else rs.Z0
    return z0, (-0.1 * g["vflat"] / z0 if kind == "lag" else 0.)


def simulate(g, cg, snr, seed, wd, hanning, kind="flat"):
    """Noiseless ANALYTIC model + noise convolved with the beam (rms = peak / snr), Hanning
    smoothed along the spectral axis if hanning (the noise only). Returns (file, centre, total flux
    of the noiseless model)."""
    from bbarolo import GalMod
    from bbarolo.utils import emptyfits
    from astropy.io import fits
    n, nz = cg["npix"], cg["nchan"]
    tmpl = emptyfits([n, n, nz], [-cg["pix"] / 3600., cg["pix"] / 3600., cg["dv"]],
                     beam=cg["beam"] / 3600., bunit="JY/BEAM", obj="cov")
    xc = (n - 1) / 2.
    dr = max(rs.RMAX / 30., cg["pix"])
    R = (np.arange(int(round(rs.RMAX / dr))) + 0.5) * dr
    t = rs.profiles(g, R)
    z0, dvdz = vertical(g, kind)
    gm = GalMod(tmpl, radii=R, xpos=xc, ypos=xc, vsys=0., vrot=t["vrot"], vdisp=t["vdisp"], vrad=0., z0=z0,
                dvdz=dvdz, inc=t["inc"], pa=t["pa"], dens=t["dens"], ltype=1, norm="NONE", mask="NONE",
                galmodtype="ANALYTIC", threads=1, verbose=False)
    clean = gm.run().model.data.astype(np.float64)
    rms = clean.max() / snr
    noise = gm.diskmodel.addnoise(np.zeros_like(clean), 1., seed=seed).astype(np.float64)
    if hanning:
        noise = 0.25 * np.roll(noise, 1, 0) + 0.5 * noise + 0.25 * np.roll(noise, -1, 0)
    noise *= rms / noise.std()
    fn = os.path.join(wd, "cube.fits")
    fits.PrimaryHDU(data=(clean + noise).astype(np.float32), header=tmpl.header).writeto(fn, overwrite=True)
    return fn, xc, float(clean.sum())


def fit(job):
    """One Bayesian fit; returns the record (quantiles and true values)."""
    warnings.simplefilter("ignore")
    from bbarolo.bayesian import BayesianBBarolo
    g, cg, cfg = job["galaxy"], job["cube"], CONFIGS[job["config"]]
    kind, gtype = job.get("kind", "flat"), job.get("galmodtype", "ANALYTIC")
    rec = dict(key=job["key"], config=job["config"], gid=g["id"], snr=job["snr"], hanning=job["hanning"],
               spatial=job["spatial"], inc=g["inc"], kind=kind, galmodtype=gtype)
    t0 = time.time()
    try:
        with tempfile.TemporaryDirectory(prefix="bbcov_") as wd:
            fn, xc, ftot = simulate(g, cg, job["snr"], job["seed"], wd, job["hanning"], kind)
            R = rs.fit_radii(cg["beam"])
            t = rs.profiles(g, R)
            z0, dvdz = vertical(g, kind)
            f = BayesianBBarolo(fn, radii=list(R), xpos=xc, ypos=xc, vsys=0., vrot=120., vdisp=15., inc=g["inc"],
                                pa=g["pa"], z0=z0, dvdz=dvdz, ltype=1, mask="SEARCH", galmodtype=gtype,
                                outfolder=os.path.join(wd, "out"), verbose=False)
            f.priors['vrot'] = dict(name='uniform', loc=0., scale=400.)
            f.priors['vdisp'] = dict(name='uniform', loc=0., scale=60.)
            ilo, ihi = min(t["inc"]), max(t["inc"])
            lo, hi = max(5., ilo - 20.), min(89., ihi + 20.)
            f.priors['inc'] = dict(name='uniform', loc=lo, scale=hi - lo)
            dpa = g.get("dpa", 0.)
            f.priors['phi'] = dict(name='uniform', loc=g["pa"] - 30. + min(dpa, 0.), scale=60. + abs(dpa))
            free = ['vrot', 'vdisp_single', 'inc', 'phi'] if kind == "warp" else \
                ['vrot', 'vdisp_single', 'inc_single', 'phi_single']
            if cfg["fracerr"]:
                free.append('fracerr')
            if cfg["sigma"]:
                free.append('sigma')
            like = dict(noisecorr=cfg["noisecorr"], likemask=cfg.get("likemask", "dilated"))
            if cfg.get("truedens"):                 # diagnostic: DENS = the true profile, not from the data
                f._set_like_options(like)
                f._setup(free)
                th = np.r_[t["vrot"], g["sigma"], g["inc"], g["pa"]].astype(float)
                rings = f._update_rings(f._inri.copy(), th)
                rings.modify_parameter("dens", np.asarray(t["dens"], float), makeobj=True)
                mt = np.asarray(f._calculate_model(rings, fullcube=True)[0], float)
                td = np.asarray(t["dens"], float) * ftot / mt.sum()
                rec["truedens_scale"] = float(ftot / mt.sum())

                def prof(rings, td=td):
                    rings.modify_parameter("dens", td, makeobj=True)
                f._set_density = prof
            f.run(freepar=free, method='nautilus', threads=1, verbose=False, like_kwargs=like,
                  sampler_kwargs=dict(n_live=job["nlive"], seed=job["seed"] % 2 ** 31),
                  run_kwargs=dict(n_eff=job["neff"]))
            q = np.asarray(f.quantiles)
            truth = {}
            for i, name in enumerate(f.freepar_names):
                if name.startswith('vrot'):
                    truth[name] = float(t["vrot"][int(name[4:]) - 1])
                elif name == 'vdisp':
                    truth[name] = float(g["sigma"])
                elif name == 'inc':
                    truth[name] = float(g["inc"])
                elif name == 'phi':
                    truth[name] = float(g["pa"])
                elif name[:3] in ('inc', 'phi') and name[3:].isdigit():          # per ring (warps)
                    truth[name] = float((t["inc"] if name[:3] == 'inc' else t["pa"])[int(name[3:]) - 1])
            rec.update(status="ok", names=list(f.freepar_names), q16=q[0].tolist(), q50=q[1].tolist(),
                       q84=q[2].tolist(), truth=truth, ess=float(f.ess), chi2=float(f.chi2_reduced()),
                       likefactor=float(f.likefactor), beam_area=float(f.beam_area), noiserms=float(f.noiserms),
                       likemask=str(f.like_kwargs['likemask']), nvox=int(f.nvox),
                       noise_correlation=None if f.noise_correlation is None else
                       {k: (float(v) if isinstance(v, (float, int, np.floating)) else
                            (v.tolist() if isinstance(v, np.ndarray) else v)) for k, v in f.noise_correlation.items()},
                       ncall=int(getattr(f.sampler, 'n_like', 0)))
            if job.get("fit3d"):
                rec["fit3d"] = fit3d(fn, R, xc, g, z0, dvdz, gtype, wd)
                rec["truth_rings"] = dict(radii=[float(x) for x in R], vrot=[float(x) for x in t["vrot"]],
                                          vdisp=[float(g["sigma"])] * len(R),
                                          inc=[float(x) for x in t["inc"]], phi=[float(x) for x in t["pa"]])
    except Exception as e:                       # a failed fit is a result too
        rec.update(status="failed", error=f"{type(e).__name__}: {e}"[:400])
    rec["time"] = time.time() - t0
    return rec


def fit3d(fn, R, xc, g, z0, dvdz, gtype, wd):
    """3DFIT on the same cube (free VROT, VDISP, INC, PA in each ring; one stage, starting from the
    true geometry, as the posterior): the fitted rings."""
    from bbarolo import FitMod3D
    r = FitMod3D(fn, radii=list(R), xpos=xc, ypos=xc, vsys=0., vrot=120., vdisp=15., inc=g["inc"], pa=g["pa"], z0=z0,
                 dvdz=dvdz, ltype=1, free="vrot vdisp inc pa", mask="SEARCH", galmodtype=gtype, twostage=False,
                 outfolder=os.path.join(wd, "fit3d"), verbose=0, plots=0, threads=1).run()
    t = r.rings[np.asarray(r.rings["fitok"], bool)]        # (the rings fitted)
    return {k: [float(x) for x in t[c]] for k, c in (("radii", "radii"), ("vrot", "vrot"), ("vdisp", "vdisp"),
                                                            ("inc", "inc"), ("phi", "pa"))}


def jobs(ngal, snrs, configs, spatial, hanning, nlive, neff, seed=2026, kind="flat", galmodtype="ANALYTIC",
         fit3d=False):
    if kind == "warp":
        gal = [dict(galaxy=rs.draw_galaxy("warp", k, seed), seed=int(np.random.default_rng([seed, 7, k]).integers(2 ** 31)))
               for k in range(ngal)]
    else:
        gal = [j for j in rs.make_jobs(["geometry"], 24, 200, 120, seed) if j["fitmode"] == "truth"
               and 30 <= j["galaxy"]["inc"] < 75][:ngal]
    pre = "" if kind == "flat" else f"{kind}/"
    post = "" if galmodtype == "ANALYTIC" else f"/{galmodtype.lower()}"
    out = []
    for snr in snrs:
        for j in gal:
            g = j["galaxy"]
            cg = rs.geometry_of_cube(g, spatial, rs.REF["spectral"])
            for c in configs:
                out.append(dict(key=f"{pre}{c}/snr{snr:g}/sp{spatial:g}/h{int(hanning)}/g{g['id']}{post}", config=c,
                                galaxy=g, cube=cg, snr=snr, spatial=spatial, hanning=hanning, nlive=nlive, neff=neff,
                                seed=j["seed"] ^ int(snr), kind=kind, galmodtype=galmodtype, fit3d=fit3d))
    return out


def _matches(name, p):
    """A free parameter name of the ring parameter p: p itself (single) or p<ring>."""
    return name == p or (name.startswith(p) and name[len(p):].isdigit())


def summary(path):
    recs = [json.loads(line) for line in open(path) if line.strip()]
    recs = list({r["key"]: r for r in recs}.values())      # (a fit run again: its last record)
    print(f"{len(recs)} fits, {sum(r['status'] != 'ok' for r in recs)} failed")
    key = lambda r: (r.get("kind", "flat"), r.get("galmodtype", "ANALYTIC"), r["snr"], r["spatial"], r["hanning"])
    pars = ("vrot", "vdisp", "inc", "phi")
    for grp in sorted({key(r) for r in recs}):
        kind, gtype, snr, sp, han = grp
        print(f"\n=== {kind} disks, {gtype} fits, S/N {snr:g}, disk radius {sp:g} beams, Hanning noise {bool(han)}")
        print(f"{'config':25s} {'fits':>4s} {'factor':>7s}  "
              + "  ".join(f"{p + ': cover  pulls':>22s}" for p in pars)
              + "   time  chi2  sigma/rms  fracerr")
        errs = {}
        for c in list(CONFIGS) + HISTORICAL:
            rr = [r for r in recs if r["config"] == c and key(r) == grp and r["status"] == "ok"]
            if not rr:
                continue
            cols = []
            for p in pars:
                cov, pull, err = [], [], []
                for r in rr:
                    for i, n in enumerate(r["names"]):
                        if _matches(n, p):
                            tv = r["truth"][n]
                            hw = 0.5 * (r["q84"][i] - r["q16"][i])
                            cov.append(r["q16"][i] <= tv <= r["q84"][i])
                            pull.append((r["q50"][i] - tv) / hw if hw > 0 else np.nan)
                            err.append(r["q50"][i] - tv)
                pull = np.array(pull)
                p16, p50, p84 = np.nanpercentile(pull, [16, 50, 84])
                cols.append(f"{np.mean(cov):5.2f} {p50:+5.1f}+-{(p84 - p16) / 2:4.1f}")
                errs.setdefault(c, {})[p] = err
            fac = np.median([1. / r["likefactor"] for r in rr])

            def fitted(name, scale=None):
                v = [r["q50"][r["names"].index(name)] / (r[scale] if scale else 1.) for r in rr
                     if name in r["names"] and (scale is None or scale in r)]
                return f"{np.median(v):9.3g}" if v else f"{'-':>9s}"
            print(f"{c:25s} {len(rr):4d} {fac:7.1f}  " + "  ".join(f"{x:>22s}" for x in cols)
                  + f"  {np.median([r['time'] for r in rr]):5.0f}s {np.median([r['chi2'] for r in rr]):5.2f}"
                  + f"  {fitted('sigma', 'noiserms')} {fitted('fracerr')}")
            f3 = [r for r in rr if "fit3d" in r]
            if f3:
                errs.setdefault("3DFIT (" + c + ")", {})
                for p in pars:
                    e = []
                    for r in f3:
                        per_ring = any(n.startswith(p) and n[len(p):].isdigit() for n in r["names"])
                        v, tv = np.asarray(r["fit3d"][p], float), np.asarray(r["truth_rings"][p], float)
                        if "radii" in r["fit3d"]:           # (3DFIT leaves out the rings without enough signal)
                            tv = tv[np.searchsorted(np.round(r["truth_rings"]["radii"], 3),
                                                    np.round(r["fit3d"]["radii"], 3))]
                        elif len(v) != len(tv):
                            continue
                        d = v - tv
                        if p == "phi":
                            d = (d + 180.) % 360. - 180.
                        # (a parameter with one value in the posterior: the median of the rings of 3DFIT)
                        e.extend(d.tolist() if per_ring else [float(np.median(d))])
                    errs["3DFIT (" + c + ")"][p] = e
        print("  (cover: fraction of the true values within the 68% interval; pulls: median and half 68% range of"
              " (median - true) / half-width; factor: voxels per independent measurement)")
        print(f"  errors of the medians (median - true, median +- half 68% range; VROT and VDISP in km/s, INC and PA"
              " in deg; 3DFIT: the median of its rings for a parameter with one value in the posterior):")
        for c, e in errs.items():
            cells = []
            for p in pars:
                v = np.asarray(e.get(p, []), float)
                v = v[np.isfinite(v)]
                if p == "phi":
                    v = (v + 180.) % 360. - 180.
                cells.append(f"{p} {np.median(v):+6.2f} +- {0.5 * np.subtract(*np.percentile(v, [84, 16])):5.2f}"
                             if v.size else f"{p} -")
            print(f"    {c:32s} " + "   ".join(cells))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=os.path.join(HERE, "results", "bayes_coverage.jsonl"))
    ap.add_argument("--ngal", type=int, default=12)
    ap.add_argument("--snr", type=float, nargs="+", default=[20.])
    ap.add_argument("--configs", nargs="+", default=["dilated_deconv"], choices=list(CONFIGS))
    ap.add_argument("--spatial", type=float, default=3., help="disk radius in beams")
    ap.add_argument("--kind", default="flat", choices=KINDS, help="the galaxies (see the documentation)")
    ap.add_argument("--galmodtype", default="ANALYTIC", choices=("ANALYTIC", "MC"), help="the models of the fit")
    ap.add_argument("--fit3d", action="store_true", help="also 3DFIT on the same cubes")
    ap.add_argument("--no-hanning", action="store_true", help="noise not Hanning-smoothed")
    ap.add_argument("--nlive", type=int, default=400, help="live points of nautilus")
    ap.add_argument("--neff", type=int, default=1500, help="effective sample size of nautilus")
    ap.add_argument("--procs", type=int, default=2)
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()
    if args.summary:
        summary(args.out)
        return
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    done = set()
    if os.path.isfile(args.out):
        done = {json.loads(line)["key"] for line in open(args.out) if line.strip()}
    todo = [j for j in jobs(args.ngal, args.snr, args.configs, args.spatial, not args.no_hanning, args.nlive,
                            args.neff, kind=args.kind, galmodtype=args.galmodtype, fit3d=args.fit3d)
            if j["key"] not in done]
    print(f"{len(todo)} fits to run ({len(done)} done)", flush=True)
    t0 = time.time()
    with get_context("fork").Pool(args.procs, maxtasksperchild=4) as pool, open(args.out, "a") as fo:
        for i, r in enumerate(pool.imap_unordered(fit, todo), 1):
            fo.write(json.dumps(r) + "\n")
            fo.flush()
            el = time.time() - t0
            print(f"[{i}/{len(todo)}] {r['key']:40s} {r['status']} {r['time']:6.0f} s "
                  f"(elapsed {el / 60:.1f} min, left ~{el / i * (len(todo) - i) / 60:.0f} min) {r.get('error', '')}",
                  flush=True)


if __name__ == "__main__":
    main()
