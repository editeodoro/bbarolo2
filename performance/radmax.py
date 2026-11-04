#!/usr/bin/env python3
"""
RADMAX in BayesianBBarolo: four ways of fitting the outer edge of a disk, compared on simulated
disks with a known sharp edge.

The disks have a rising rotation curve (arctan, turnover radius 0.3-0.6 RMAX, so that VROT changes
across the last ring), constant VDISP, an exponential surface density and a sharp edge at a radius
Rt drawn within the last ring of the fit (RMAX - 0.9 w to RMAX - 0.1 w, rings of width w = half a
beam from the centre to RMAX). They are simulated with GALMOD (ANALYTIC, nodes up to Rt) plus noise
convolved with the beam, and fitted with BayesianBBarolo (nautilus): VROT in each ring, VDISP, INC
and PA (centre, VSYS and Z0 fixed at the true values), and

  ring     RADMAX free, the radius of the last ring (like_kwargs radmax='ring', the RADMAX of the
           original module): the last ring moves, and its annulus with it
  edge     RADMAX free, the outer edge of the disk (radmax='edge', the default of BayesianBBarolo
           since these tests): the rings stay, and the model has no emission beyond RADMAX
  taper    as edge, with a Gaussian edge a quarter of a beam wide (radmax='taper')
  evid     no RADMAX: N-1, N and N+1 rings (N: those up to RMAX), compared by their evidence

For each fit the posterior quantiles are stored with the truth, and the summary gives:
  * the edge of the disk: for 'ring' 1.5 RADMAX - 0.5 R(N-1) (the outer edge of the last annulus
    of CENTRED rings), for 'edge' and 'taper' RADMAX, for 'evid' the outer edge of the rings with
    the largest evidence: its error (median - Rt) in units of w, and the coverage of Rt;
  * the last ring: the posterior of VROT(N) - Vtrue(R(N)), with R(N) the radius of the last ring
    in each sample (it moves with 'ring'): coverage of 0 and pulls;
  * the coverage and pulls of the other parameters, and the time of the fits.

    python performance/radmax.py run --ngal 12 --procs 72 --out results/radmax.jsonl
    python performance/radmax.py summary --out results/radmax.jsonl
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

RMAX = rs.RMAX
METHODS = ("ring", "edge", "taper", "evid")


def draw(k, seed):
    rng = np.random.default_rng([seed, 11, k])
    return dict(id=k, vflat=rng.uniform(100, 250), rt=rng.uniform(0.3, 0.6), sigma=rng.uniform(10, 20),
                h=rng.uniform(0.3, 0.5), inc=rng.uniform(40, 70), pa=rng.uniform(0, 360), u=rng.uniform(0.1, 0.9),
                seed=int(rng.integers(2 ** 31)))


def vtrue(g, R):
    return 2. / np.pi * g["vflat"] * np.arctan(np.asarray(R, float) / (g["rt"] * RMAX))


def geometry(spatial):
    beam = RMAX / spatial
    w = beam / 2.
    n = int(round(RMAX / w))
    return beam, w, n


def simulate(g, spatial, snr, wd):
    from bbarolo import GalMod
    from bbarolo.utils import emptyfits
    from astropy.io import fits
    beam, w, n = geometry(spatial)
    pix = beam / rs.PIX_PER_BEAM
    dv = g["sigma"]
    npix = 2 * int(np.ceil((RMAX + 1.5 * beam) / pix)) + 1
    vext = g["vflat"] * np.sin(np.radians(g["inc"])) + 4. * np.hypot(g["sigma"], dv / 2.355) + 2. * dv
    nz = 2 * int(np.ceil(vext / dv)) + 1
    tmpl = emptyfits([npix, npix, nz], [-pix / 3600., pix / 3600., dv], beam=beam / 3600., bunit="JY/BEAM", obj="edge")
    xc = (npix - 1) / 2.
    rt = RMAX - g["u"] * w                             # the edge of the disk, within the last ring
    R = np.linspace(0., rt, int(np.ceil(rt / (2. * pix))) + 1)         # nodes: no emission beyond rt
    gm = GalMod(tmpl, radii=R, xpos=xc, ypos=xc, vsys=0., vrot=vtrue(g, R), vdisp=g["sigma"], vrad=0., z0=rs.Z0,
                inc=g["inc"], pa=g["pa"], dens=np.exp(-R / (g["h"] * RMAX)), ltype=1, norm="NONE", mask="NONE",
                galmodtype="ANALYTIC", ringmode="NODES", threads=1, verbose=False)
    clean = gm.model().data.astype(np.float64)
    rms = clean.max() / snr
    noise = gm.diskmodel.addnoise(np.zeros_like(clean), 1., seed=g["seed"]).astype(np.float64)
    noise *= rms / noise.std()
    fn = os.path.join(wd, "cube.fits")
    fits.PrimaryHDU(data=(clean + noise).astype(np.float32), header=tmpl.header).writeto(fn, overwrite=True)
    return fn, xc, rt


def fit(job):
    warnings.simplefilter("ignore")
    from bbarolo.bayesian import BayesianBBarolo
    g, method, nr = job["galaxy"], job["method"], job["nrings"]
    beam, w, n = geometry(job["spatial"])
    rec = dict(key=job["key"], gid=g["id"], method=method, nrings=nr, n=n, w=w, spatial=job["spatial"],
               snr=job["snr"])
    t0 = time.time()
    try:
        with tempfile.TemporaryDirectory(prefix="bbedge_") as wd:
            fn, xc, rt = simulate(g, job["spatial"], job["snr"], wd)
            R = (np.arange(nr) + 0.5) * w
            f = BayesianBBarolo(fn, radii=list(R), xpos=xc, ypos=xc, vsys=0., vrot=120., vdisp=15., inc=g["inc"],
                                pa=g["pa"], z0=rs.Z0, ltype=1, mask="SEARCH", galmodtype="ANALYTIC",
                                outfolder=os.path.join(wd, "out"), verbose=False)
            f.priors['vrot'] = dict(name='uniform', loc=0., scale=400.)
            f.priors['vdisp'] = dict(name='uniform', loc=0., scale=60.)
            lo, hi = max(5., g["inc"] - 20.), min(89., g["inc"] + 20.)
            f.priors['inc'] = dict(name='uniform', loc=lo, scale=hi - lo)
            f.priors['phi'] = dict(name='uniform', loc=g["pa"] - 30., scale=60.)
            free = ['vrot', 'vdisp_single', 'inc_single', 'phi_single']
            like = {}
            if method != "evid":
                free.append('radmax')
                like['radmax'] = method
            f.run(freepar=free, method='nautilus', threads=1, verbose=False, like_kwargs=like,
                  sampler_kwargs=dict(n_live=job["nlive"], seed=g["seed"] % 2 ** 31),
                  run_kwargs=dict(n_eff=job["neff"]))
            S = np.asarray(f.samples_equal)
            idx = f.freepar_idx
            q = lambda x: np.quantile(np.asarray(x, float), [0.16, 0.5, 0.84]).tolist()
            # the last ring: VROT(N) - Vtrue(R(N)), R(N) = its radius in each sample
            Rn = S[:, idx['radmax'][0]] if method == "ring" else np.full(len(S), R[-1])
            dv_last = S[:, idx['vrot'][-1]] - vtrue(g, Rn)
            if method == "ring":
                edge = 1.5 * S[:, idx['radmax'][0]] - 0.5 * R[-2]
            elif method in ("edge", "taper"):
                edge = S[:, idx['radmax'][0]]
            else:
                edge = np.full(len(S), R[-1] + w / 2.)
            names = list(f.freepar_names)
            truth = {}
            for i, nm in enumerate(names):
                if nm.startswith('vrot'):
                    truth[nm] = float(vtrue(g, R[int(nm[4:]) - 1]))
                elif nm in ('vdisp', 'inc', 'phi'):
                    truth[nm] = float(g[dict(vdisp="sigma", inc="inc", phi="pa")[nm]])
            qq = np.asarray(f.quantiles)
            rec.update(status="ok", rt=float(rt), edge=q(edge), dv_last=q(dv_last), names=names, q16=qq[0].tolist(),
                       q50=qq[1].tolist(), q84=qq[2].tolist(), truth=truth, logz=float(f.sampler.log_z),
                       ess=float(f.ess), ncall=int(getattr(f.sampler, 'n_like', 0)))
    except Exception as e:
        rec.update(status="failed", error=f"{type(e).__name__}: {e}"[:400])
    rec["time"] = time.time() - t0
    return rec


def jobs(ngal, spatial, snr, nlive, neff, seed=2026):
    out = []
    _, _, n = geometry(spatial)
    for k in range(ngal):
        g = draw(k, seed)
        for m in METHODS:
            for nr in ((n - 1, n, n + 1) if m == "evid" else (n,)):
                out.append(dict(key=f"{m}/n{nr}/sp{spatial:g}/snr{snr:g}/g{k}", galaxy=g, method=m, nrings=nr,
                                spatial=spatial, snr=snr, nlive=nlive, neff=neff))
    return out


def summary(path):
    recs = [json.loads(line) for line in open(path) if line.strip()]
    ok = [r for r in recs if r["status"] == "ok"]
    print(f"{len(recs)} fits, {len(recs) - len(ok)} failed")
    for sp, snr in sorted({(r["spatial"], r["snr"]) for r in ok}):
        rr = [r for r in ok if r["spatial"] == sp and r["snr"] == snr]
        print(f"\n=== disk radius {sp:g} beams, S/N {snr:g}; edge errors in units of the ring width w")
        print(f"{'method':8s} {'fits':>4s}  {'edge: err':>10s} {'+-':>5s} {'cover':>6s}   {'VROT(N): cover':>14s} "
              f"{'pulls':>11s}   {'vdisp':>11s} {'inc':>11s} {'phi':>11s} {'vrot1..N-1':>11s}   time")
        for m in METHODS:
            mm = [r for r in rr if r["method"] == m]
            if m == "evid":                         # the number of rings with the largest evidence
                best = {}
                for r in mm:
                    if r["gid"] not in best or r["logz"] > best[r["gid"]]["logz"]:
                        best[r["gid"]] = r
                mm = list(best.values())
                pick = np.bincount([r["nrings"] - r["n"] + 1 for r in mm], minlength=3)
            if not mm:
                continue
            e = np.array([(r["edge"][1] - r["rt"]) / r["w"] for r in mm])
            cov_e = np.mean([r["edge"][0] <= r["rt"] <= r["edge"][2] for r in mm])
            d = [r["dv_last"] for r in mm]
            cov_v = np.mean([x[0] <= 0 <= x[2] for x in d])
            pv = np.array([x[1] / (0.5 * (x[2] - x[0])) for x in d])

            def cp(name):
                c, p = [], []
                for r in mm:
                    for i, nm in enumerate(r["names"]):
                        if nm == name or (name == "vrot" and nm.startswith("vrot") and nm != f"vrot{r['nrings']}"):
                            tv, hw = r["truth"][nm], 0.5 * (r["q84"][i] - r["q16"][i])
                            c.append(r["q16"][i] <= tv <= r["q84"][i])
                            p.append((r["q50"][i] - tv) / hw)
                return f"{np.mean(c):4.2f} {np.median(p):+4.1f}~{0.5 * np.subtract(*np.percentile(p, [84, 16])):3.1f}"
            hw = lambda x: 0.5 * np.subtract(*np.percentile(x, [84, 16]))
            print(f"{m:8s} {len(mm):4d}  {np.median(e):+10.2f} {hw(e):5.2f} "
                  f"{cov_e:6.2f}   {cov_v:14.2f} {np.median(pv):+5.1f}~{hw(pv):3.1f}"
                  f"   {cp('vdisp'):>11s} {cp('inc'):>11s} {cp('phi'):>11s} {cp('vrot'):>11s}"
                  f"   {np.median([r['time'] for r in mm]):4.0f}s" + (f"   (N-1, N, N+1 chosen: {pick.tolist()})"
                                                                     if m == "evid" else ""))
        print("  (cover: fraction of the true values in the 68% interval; pulls: median ~ half 68% range of"
              " (median - true) / half-width; VROT(N): of VROT(N) - Vtrue(R(N)))")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("action", choices=("run", "summary"))
    ap.add_argument("--out", default=os.path.join(HERE, "results", "radmax.jsonl"))
    ap.add_argument("--ngal", type=int, default=12)
    ap.add_argument("--spatial", type=float, default=5., help="disk radius in beams")
    ap.add_argument("--snr", type=float, default=20.)
    ap.add_argument("--nlive", type=int, default=400)
    ap.add_argument("--neff", type=int, default=1500)
    ap.add_argument("--procs", type=int, default=2)
    args = ap.parse_args()
    if args.action == "summary":
        summary(args.out)
        return
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    done = set()
    if os.path.isfile(args.out):
        done = {json.loads(line)["key"] for line in open(args.out) if line.strip()}
    todo = [j for j in jobs(args.ngal, args.spatial, args.snr, args.nlive, args.neff) if j["key"] not in done]
    print(f"{len(todo)} fits to run ({len(done)} done)", flush=True)
    t0 = time.time()
    with get_context("fork").Pool(args.procs, maxtasksperchild=2) as pool, open(args.out, "a") as fo:
        for i, r in enumerate(pool.imap_unordered(fit, todo), 1):
            fo.write(json.dumps(r) + "\n")
            fo.flush()
            el = time.time() - t0
            print(f"[{i}/{len(todo)}] {r['key']:32s} {r['status']} {r['time']:6.0f} s (elapsed {el / 60:.1f} min) "
                  f"{r.get('error', '')}", flush=True)


if __name__ == "__main__":
    main()
