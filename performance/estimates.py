#!/usr/bin/env python3
"""
Accuracy of the initial estimates of BBarolo (paramguess.EstimateInitial) on simulated galaxies.

The galaxies and cubes are those of run_suite.py (ANALYTIC GALMOD models plus Gaussian noise
smoothed to the beam): flat disks with random INC (15-85 deg), PA and peak S/N (5-40), at
several spatial resolutions, and warped disks. Nothing is given to the estimator but the cube
(as with BBarolo -f), unless --given centre,vsys.

    python estimates.py cubes --outdir est --n 120          # simulate the cubes (once)
    python estimates.py run --outdir est --label new          # estimates with the current code
    python estimates.py run --outdir est --label ellipse --guesstype ELLIPSE
    python estimates.py summary --outdir est new ellipse      # tables of the errors

Each run writes one JSON line per cube in OUTDIR/<label>.jsonl. --code runs another copy of
bbarolo (e.g. a git worktree of an older commit).
"""

import os
import sys
import io
import json
import time
import argparse
import contextlib
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SPATIAL_EST = (3., 5., 8.)            # disk radius / beam of the flat disks


def make_cubes(outdir, n, seed=7, procs=16):
    """n flat disks per spatial resolution and n warped disks (at 5 beams)."""
    from multiprocessing import Pool
    os.makedirs(os.path.join(outdir, "cubes"), exist_ok=True)
    jobs = []
    for sp in SPATIAL_EST:
        jobs += [("geometry", k, sp) for k in range(n)]
    jobs += [("warp", k, 5.) for k in range(n)]
    with Pool(procs) as p:
        rows = p.starmap(_make_cube, [(outdir, kind, k, sp, seed) for kind, k, sp in jobs])
    with open(os.path.join(outdir, "truth.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{len(rows)} cubes in {outdir}/cubes")


def _make_cube(outdir, kind, k, spatial, seed):
    sys.path.insert(0, HERE)
    import run_suite as rs
    warnings.simplefilter("ignore")
    g = rs.draw_galaxy(kind, k, seed)
    cg = rs.geometry_of_cube(g, spatial, 1.)
    name = f"{kind}{k:03d}_s{spatial:g}"
    wd = os.path.join(outdir, "cubes", name)
    os.makedirs(wd, exist_ok=True)
    with contextlib.redirect_stdout(io.StringIO()):
        fn, xc, rms, peak, _ = rs.simulate(g, cg, g.get("snr", rs.REF["snr"]), seed * 1000 + k, wd)
    R = rs.fit_radii(cg["beam"])
    t = rs.profiles(g, R)
    return dict(name=name, file=fn, kind=kind, spatial=spatial, snr=g.get("snr", rs.REF["snr"]), xc=xc, vsys=0.,
                inc=g["inc"], pa=g["pa"], dinc=g["dinc"], dpa=g["dpa"], vflat=g["vflat"], sigma=g["sigma"],
                beam=cg["beam"], pix=cg["pix"], rmax=rs.RMAX, radii=list(R), inc_r=list(t["inc"]),
                pa_r=list(t["pa"]), vrot_r=list(t["vrot"]))


def _estimate(args):
    row, code, guesstype, given = args
    sys.path.insert(0, code)
    warnings.simplefilter("ignore")
    os.environ.setdefault("MPLBACKEND", "Agg")
    from bbarolo.params import Param
    from bbarolo.cube import Cube
    from bbarolo.paramguess import EstimateInitial
    from bbarolo import keywords
    s = f"FITSFILE {row['file']}\nMASK SEARCH\nTHREADS 1\nVERBOSE false\nPLOTS false\n" \
        f"OUTFOLDER {os.path.dirname(row['file'])}/out/\n"
    if guesstype and keywords.lookup("GUESSTYPE") is not None:
        s += f"GUESSTYPE {guesstype}\n"
    if "centre" in given:
        s += f"XPOS {row['xc']}\nYPOS {row['xc']}\n"
    if "vsys" in given:
        s += f"VSYS {row['vsys']}\n"
    os.makedirs(os.path.join(os.path.dirname(row["file"]), "out"), exist_ok=True)
    p = Param()
    p.readParamString(s)
    p.checkPars(quiet=True)
    out = dict(name=row["name"])
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            c = Cube(row["file"], par=p)
            t0 = time.perf_counter()
            ip = EstimateInitial(c, c.pars().gf)
            out["time"] = time.perf_counter() - t0
        out.update(xc=ip.xcentre, yc=ip.ycentre, vsys=ip.vsystem, inc=ip.inclin, pa=ip.posang, vrot=ip.vrot,
                   radsep=ip.radsep, nrings=ip.nrings, rmax=getattr(ip, "Rmax", np.nan),
                   vdisp=getattr(ip, "vdisp", None),
                   inc_r=list(map(float, getattr(ip, "inc_profile", []))),
                   pa_r=list(map(float, getattr(ip, "pa_profile", []))),
                   rad_r=list(map(float, getattr(ip, "rad_profile", []))))
    except Exception as e:                       # (an estimate that fails is a result too)
        out["error"] = f"{type(e).__name__}: {e}"[:200]
    return out


def run(outdir, label, code, guesstype, given, procs):
    from multiprocessing import Pool
    rows = [json.loads(ln) for ln in open(os.path.join(outdir, "truth.jsonl"))]
    with Pool(procs) as p:
        res = p.map(_estimate, [(r, code, guesstype, given) for r in rows], chunksize=1)
    with open(os.path.join(outdir, f"{label}.jsonl"), "w") as f:
        for r in res:
            f.write(json.dumps(r) + "\n")
    print(f"{label}: {sum('error' in r for r in res)} failures of {len(res)}")


def _fit3d(args):
    row, code, opts = args
    sys.path.insert(0, code)
    sys.path.insert(0, HERE)
    warnings.simplefilter("ignore")
    os.environ.setdefault("MPLBACKEND", "Agg")
    import run_suite as rs
    from bbarolo import FitMod3D
    radii = rs.fit_radii(row["beam"])
    kw = dict(radii=radii, xpos=row["xc"], ypos=row["xc"], vsys=row["vsys"], z0=rs.Z0, ltype=1,
              free="VROT VDISP INC PA", mask="SEARCH", twostage=True, threads=1, plots=False, verbose=0)
    kw.update(opts)
    out = dict(name=row["name"])
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            t0 = time.perf_counter()
            res = FitMod3D(row["file"], **kw).run(write=False)
            out["time"] = time.perf_counter() - t0
        t = res.rings
        out.update(radii=list(map(float, t["radii"])), inc=list(map(float, t["inc"])), pa=list(map(float, t["pa"])),
                   vrot=list(map(float, t["vrot"])), vdisp=list(map(float, t["vdisp"])))
        if "rings_initial" in res:
            out.update(inc0=list(map(float, res.rings_initial["inc"])), pa0=list(map(float, res.rings_initial["pa"])))
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"[:200]
    return out


def fit3d(outdir, label, code, opts, kinds, spatial, procs):
    from multiprocessing import Pool
    rows = [json.loads(ln) for ln in open(os.path.join(outdir, "truth.jsonl"))]
    rows = [r for r in rows if r["kind"] in kinds and (r["kind"] == "warp" or r["spatial"] in spatial)]
    with Pool(procs) as p:
        res = p.map(_fit3d, [(r, code, opts) for r in rows], chunksize=1)
    with open(os.path.join(outdir, f"fit3d_{label}.jsonl"), "w") as f:
        for r in res:
            f.write(json.dumps(r) + "\n")
    print(f"fit3d_{label}: {sum('error' in r for r in res)} failures of {len(res)}")


def summary3d(outdir, labels):
    """Errors of the final 3DFIT rings: INC and PA (ring by ring), VROT/VROT_true - 1, VDISP/sigma - 1."""
    truth = {json.loads(ln)["name"]: json.loads(ln) for ln in open(os.path.join(outdir, "truth.jsonl"))}
    for lab in labels:
        res = [json.loads(ln) for ln in open(os.path.join(outdir, f"fit3d_{lab}.jsonl"))]
        ok = [r for r in res if "error" not in r and r["radii"]]
        print(f"\n=== 3DFIT from {lab}: {len(ok)}/{len(res)} fits, median time {np.median([r['time'] for r in ok]):.1f} s")
        for kind in ("geometry", "warp"):
            sel = [r for r in ok if truth[r["name"]]["kind"] == kind]
            if not sel:
                continue
            di, dp, dv, ds, di0 = [], [], [], [], []
            for r in sel:
                t = truth[r["name"]]
                R = np.asarray(r["radii"])
                ti = np.interp(R, t["radii"], t["inc_r"])
                tp = np.interp(R, t["radii"], np.degrees(np.unwrap(np.radians(t["pa_r"]))))
                tv = np.interp(R, t["radii"], t["vrot_r"])
                di += list(np.asarray(r["inc"]) - ti)
                dp += list(_dang(r["pa"], tp))
                dv += list(np.asarray(r["vrot"]) / tv - 1)
                ds += list(np.asarray(r["vdisp"]) / t["sigma"] - 1)
            f = lambda x: "{:+.2f} ± {:.2f}".format(*_rstat(x))
            print(f"  {kind:8s} ({len(sel)}): INC {f(di)} deg, PA {f(dp)} deg, VROT {f(dv)}, VDISP {f(ds)}")
            if kind == "geometry":
                bins = [(15, 30), (30, 45), (45, 60), (60, 75), (75, 86)]
                line = "     INC by true INC:"
                for lo, hi in bins:
                    x = []
                    for r in sel:
                        t = truth[r["name"]]
                        if lo <= t["inc"] < hi:
                            x += list(np.asarray(r["inc"]) - t["inc"])
                    line += f"  {lo}-{hi}: " + "{:+.1f} ± {:.1f}".format(*_rstat(x))
                print(line)


def _dang(a, b):
    """a - b for position angles, in (-180, 180]."""
    return (np.asarray(a) - np.asarray(b) + 180.) % 360. - 180.


def _rstat(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if not len(x):
        return np.nan, np.nan
    med = np.median(x)
    return med, 1.4826 * np.median(np.abs(x - med))


def summary(outdir, labels):
    truth = {json.loads(ln)["name"]: json.loads(ln) for ln in open(os.path.join(outdir, "truth.jsonl"))}
    bins = [(15, 30), (30, 45), (45, 60), (60, 75), (75, 86)]
    for lab in labels:
        res = [json.loads(ln) for ln in open(os.path.join(outdir, f"{lab}.jsonl"))]
        ok = [r for r in res if "error" not in r]
        print(f"\n=== {lab}: {len(ok)}/{len(res)} estimates, median time {np.median([r['time'] for r in ok]):.2f} s")
        flat = [r for r in ok if truth[r["name"]]["kind"] == "geometry"]
        print(f"  flat disks: INC error (median, robust scatter) by true INC; PA, centre (beams), VSYS (km/s), "
              f"VROT/VFLAT-1")
        line = "   "
        for lo, hi in bins:
            sel = [r for r in flat if lo <= truth[r["name"]]["inc"] < hi]
            m, s = _rstat([r["inc"] - truth[r["name"]]["inc"] for r in sel])
            line += f"  {lo}-{hi}: {m:+5.1f} ± {s:4.1f} ({len(sel)})"
        print(line)
        for sp in SPATIAL_EST:
            sel = [r for r in flat if truth[r["name"]]["spatial"] == sp]
            di = _rstat([r["inc"] - truth[r["name"]]["inc"] for r in sel])
            dp = _rstat(_dang([r["pa"] for r in sel], [truth[r["name"]]["pa"] for r in sel]))
            dc = _rstat([np.hypot(r["xc"] - truth[r["name"]]["xc"], r["yc"] - truth[r["name"]]["xc"])
                         * truth[r["name"]]["pix"] / truth[r["name"]]["beam"] for r in sel])
            dv = _rstat([r["vsys"] - truth[r["name"]]["vsys"] for r in sel])
            vr = _rstat([r["vrot"] / truth[r["name"]]["vflat"] - 1 for r in sel])
            print(f"   {sp:g} beams: INC {di[0]:+5.1f} ± {di[1]:4.1f}, PA {dp[0]:+5.1f} ± {dp[1]:4.1f}, "
                  f"centre {dc[0]:.2f} ± {dc[1]:.2f}, VSYS {dv[0]:+5.1f} ± {dv[1]:4.1f}, VROT {vr[0]:+.2f} ± {vr[1]:.2f}")
        bysnr = {}
        for r in flat:
            bysnr.setdefault(truth[r["name"]]["snr"], []).append(r["inc"] - truth[r["name"]]["inc"])
        print("   INC by peak S/N: " + ", ".join(f"{k:g}: {_rstat(v)[0]:+.1f} ± {_rstat(v)[1]:.1f}"
                                                for k, v in sorted(bysnr.items())))
        warp = [r for r in ok if truth[r["name"]]["kind"] == "warp"]
        if warp:
            prof = [r for r in warp if r.get("inc_r")]
            line = f"  warps ({len(warp)}): single INC vs mean true INC {_rstat([r['inc'] - np.mean(truth[r['name']]['inc_r']) for r in warp])[0]:+.1f}"
            if prof:
                # the radial profiles, interpolated at the radii of the truth: fraction of the warp recovered
                fi, fp = [], []
                for r in prof:
                    t = truth[r["name"]]
                    R = np.asarray(t["radii"])
                    ei = np.interp(R, r["rad_r"], r["inc_r"]) if r["rad_r"] else np.full(len(R), r["inc"])
                    ep = np.interp(R, r["rad_r"], np.unwrap(np.radians(r["pa_r"])) * 180 / np.pi) \
                        if r["rad_r"] else np.full(len(R), r["pa"])
                    if abs(t["dinc"]) > 5:
                        fi.append((ei[-1] - ei[0]) / (t["inc_r"][-1] - t["inc_r"][0]))
                    if abs(t["dpa"]) > 10:
                        fp.append(_dang(ep[-1], ep[0]) / _dang(t["pa_r"][-1], t["pa_r"][0]))
                line += (f"; fraction of the warp recovered by the profiles: INC {_rstat(fi)[0]:.2f} "
                         f"({len(fi)}), PA {_rstat(fp)[0]:.2f} ({len(fp)})")
            print(line)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=("cubes", "run", "summary", "fit3d", "summary3d"))
    ap.add_argument("labels", nargs="*")
    ap.add_argument("--outdir", default="estimates")
    ap.add_argument("--n", type=int, default=100, help="galaxies per configuration (cubes)")
    ap.add_argument("--label", default="new")
    ap.add_argument("--code", default=os.path.dirname(HERE), help="the bbarolo2 folder to use")
    ap.add_argument("--guesstype", default=None)
    ap.add_argument("--given", default="", help="parameters given to the estimator: centre,vsys")
    ap.add_argument("--procs", type=int, default=16)
    ap.add_argument("--opts", default="", help="fit3d: parameters of FitMod3D, key=value,key=value")
    ap.add_argument("--kinds", default="geometry,warp")
    ap.add_argument("--spatial", default="5", help="fit3d: spatial resolutions of the flat disks")
    a = ap.parse_args()
    if a.what == "cubes":
        make_cubes(a.outdir, a.n, procs=a.procs)
    elif a.what == "run":
        run(a.outdir, a.label, os.path.abspath(a.code), a.guesstype, a.given.split(","), a.procs)
    elif a.what == "fit3d":
        opts = dict(kv.split("=", 1) for kv in a.opts.split(",") if kv)
        fit3d(a.outdir, a.label, os.path.abspath(a.code), opts, a.kinds.split(","),
              [float(x) for x in a.spatial.split(",")], a.procs)
    elif a.what == "summary3d":
        summary3d(a.outdir, a.labels)
    else:
        summary(a.outdir, a.labels)


if __name__ == "__main__":
    main()
