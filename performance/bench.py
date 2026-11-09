#!/usr/bin/env python3
"""
Benchmark of the model engine as used by BayesianBBarolo: the time of one likelihood call and of
one full model, on NGC 2403 (../examples/ngc2403.fits or --cube).

    python performance/bench.py -o before.json             # times and log-likelihood values
    python performance/bench.py --compare before.json      # the same, compared with a previous run

Each likelihood case evaluates the log-likelihood at N parameter points drawn from the priors (a
fixed seed), on one thread, as a sampler worker does. The values are saved, and --compare checks
that they are identical (optimisations must not change results) and prints the speed-ups.

    python performance/bench.py --sampling 1,8,32 [--nlive 500 --max-calls 10000] [--no-blas-limit]

times instead a short sampling (nautilus, fixed seed, stopped after --max-calls likelihood calls) of the
case "ANALYTIC, + inc + pa" with 1, 8 and 32 processes: the time of the sampling, the number of likelihood calls and the calls per
second. --no-blas-limit lets BLAS use all its threads (bbarolo.bayesian.common.BLAS_THREADS = None).
"""

import os
import sys
import json
import time
import argparse
import warnings
import contextlib
import io

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

CASES = {
    # name: (galmodtype, free parameters)
    "MC, vrot + vdisp": ("MC", ["vrot", "vdisp_single"]),
    "ANALYTIC, vrot + vdisp": ("ANALYTIC", ["vrot", "vdisp_single"]),
    "ANALYTIC, + inc + pa": ("ANALYTIC", ["vrot", "vdisp_single", "inc_single", "phi_single"]),
    "MC, + inc + pa": ("MC", ["vrot", "vdisp_single", "inc_single", "phi_single"]),
}


def likelihood_case(cube, galmodtype, freepar, n, seed=1):
    from bbarolo.bayesian import BayesianBBarolo
    with contextlib.redirect_stdout(io.StringIO()):
        f = BayesianBBarolo(cube, radii=np.arange(60, 700, 120), xpos=77, ypos=77, vsys=132.8, vrot=100, vdisp=10,
                            inc=60, pa=123.7, z0=10, mask="SEARCH", galmodtype=galmodtype, threads=1,
                            outfolder=os.path.join(os.environ.get("TMPDIR", "/tmp"), "bbbench") + "/", verbose=False)
        f._setup(freepar)
    rng = np.random.default_rng(seed)
    thetas = [np.asarray(f._prior_transform(rng.random(f.ndim)), float) for _ in range(n)]
    f._log_likelihood(thetas[0])                     # (compilation, caches)
    vals, times = [], []
    for th in thetas:
        t0 = time.perf_counter()
        vals.append(float(f._log_likelihood(th)))
        times.append(time.perf_counter() - t0)
    return dict(ms_median=1e3 * float(np.median(times)), ms_mean=1e3 * float(np.mean(times)), values=vals)


def sampling_case(cube, threads, nlive, max_calls, blas_limit=True, seed=1):
    from bbarolo.bayesian import BayesianBBarolo, common, fitter
    common.BLAS_THREADS = 1 if blas_limit else None
    with contextlib.redirect_stdout(io.StringIO()):
        f = BayesianBBarolo(cube, radii=np.arange(60, 700, 120), xpos=77, ypos=77, vsys=132.8, vrot=100, vdisp=10,
                            inc=60, pa=123.7, z0=10, mask="SEARCH", galmodtype="ANALYTIC",
                            outfolder=os.path.join(os.environ.get("TMPDIR", "/tmp"), "bbbench") + "/", verbose=False)
    spent = {}
    sample = fitter.sample

    def timed(*a, **k):                            # the time of the sampling only (not of the setup)
        t0 = time.perf_counter()
        out = sample(*a, **k)
        spent['s'] = time.perf_counter() - t0
        return out
    fitter.sample = timed
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            f.run(freepar=CASES["ANALYTIC, + inc + pa"][1], threads=threads, method='nautilus', verbose=False,
                  sampler_kwargs=dict(n_live=nlive, seed=seed), run_kwargs=dict(n_like_max=max_calls))
    finally:
        fitter.sample = sample
    ncall = int(f.sampler.n_like)
    logz = f.sampler.log_z
    return dict(seconds=spent['s'], ncall=ncall, calls_per_s=ncall / spent['s'],
                logz=float('nan') if logz is None else float(logz))


def model_case(cube, galmodtype, nrep=5):
    from bbarolo import Cube, DiskModel, Rings
    from bbarolo.params import Param
    c = Cube(cube, par=Param(verbose=False))
    dm = DiskModel(c)
    r = Rings().setRings(radii=np.arange(15, 1200, 30), xpos=77, ypos=77, vsys=132.8, vrot=120, vdisp=8, z0=10,
                         inc=60, phi=123.7, dens=1)
    m = dm.array(r, galmodtype=galmodtype, threads=1)
    times = []
    for _ in range(nrep):
        t0 = time.perf_counter()
        dm.array(r, galmodtype=galmodtype, threads=1)
        times.append(time.perf_counter() - t0)
    return dict(ms_median=1e3 * float(np.median(times)), checksum=float(np.sum(m, dtype=np.float64)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cube", default=os.path.join(os.path.dirname(REPO), "examples", "ngc2403.fits"))
    ap.add_argument("-n", type=int, default=200, help="likelihood calls per case")
    ap.add_argument("-o", "--output", help="JSON file for the results")
    ap.add_argument("--compare", help="JSON file of a previous run")
    ap.add_argument("--sampling", help="time a short sampling with these numbers of processes, e.g. 1,8,32")
    ap.add_argument("--nlive", type=int, default=500)
    ap.add_argument("--max-calls", type=int, default=10000, help="likelihood calls of the sampling (n_like_max)")
    ap.add_argument("--no-blas-limit", action="store_true", help="BLAS with all its threads")
    args = ap.parse_args()
    warnings.simplefilter("ignore")
    if args.sampling:
        print(f"{'processes':>9s} {'seconds':>9s} {'calls':>8s} {'calls/s':>9s} {'logz':>12s}")
        for n in (int(x) for x in args.sampling.split(",")):
            r = sampling_case(args.cube, n, args.nlive, args.max_calls, blas_limit=not args.no_blas_limit)
            print(f"{n:9d} {r['seconds']:9.2f} {r['ncall']:8d} {r['calls_per_s']:9.1f} {r['logz']:12.3f}")
        return 0
    res = {"likelihood": {}, "model": {}}
    for name, (gt, free) in CASES.items():
        res["likelihood"][name] = likelihood_case(args.cube, gt, free, args.n)
    for gt in ("MC", "ANALYTIC"):
        res["model"][f"full model, 40 rings, {gt}"] = model_case(args.cube, gt)
    old = None
    if args.compare:
        with open(args.compare) as f:
            old = json.load(f)
    print(f"{'case':38s} {'ms/call':>9s}" + (f" {'before':>9s} {'speed-up':>9s}  same values" if old else ""))
    for kind in ("likelihood", "model"):
        for name, r in res[kind].items():
            line = f"{name:38s} {r['ms_median']:9.2f}"
            if old and name in old.get(kind, {}):
                o = old[kind][name]
                same = (r["values"] == o["values"]) if kind == "likelihood" else (r["checksum"] == o["checksum"])
                line += f" {o['ms_median']:9.2f} {o['ms_median'] / r['ms_median']:8.2f}x  {'yes' if same else 'NO'}"
            print(line)
    if args.output:
        with open(args.output, "w") as f:
            json.dump(res, f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
