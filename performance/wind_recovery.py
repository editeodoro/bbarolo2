"""
Recovery of the parameters of patchy mock winds by BAYESIANWIND: bias, pulls and coverage of the 68% intervals.

Each fit is a mock of wind_mocks.py (GalMod3D: clumps or a log-normal field in the cones, smoothed noise) fitted with
BayesianWind (nautilus), with one emission basis of the likelihood (option emission, and marginal or profile). The
fits run as separate processes (each with its own pool of --threads processes), --jobs at a time; each writes one
line of JSON to the results file (default performance/results/wind_recovery.jsonl), and --report writes the summary
(performance/report/wind_recovery.md).

    python3 performance/wind_recovery.py --quick                       # 2 configurations x 2 seeds, cells
    python3 performance/wind_recovery.py --configs all --seeds 8 --jobs 12 --threads 8
    python3 performance/wind_recovery.py --configs filled,hollow --bases cells,cells-profile,shells,spaxel,voxel
    python3 performance/wind_recovery.py --report
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

RESULTS = os.path.join(HERE, "results", "wind_recovery.jsonl")
REPORT = os.path.join(HERE, "report", "wind_recovery.md")

# the configurations: the truth (over wind_mocks.DEFAULT_TRUTH), the mock and the free parameters
BASE_FREE = ["vwind_single", "vdisp_single", "inc", "pa", "openang", "emscale"]
CONFIGS = {
    "filled": dict(truth={}, mock=dict(kind="clumps"), free=BASE_FREE),
    "hollow": dict(truth=dict(openin=30., openang=70.), mock=dict(kind="clumps"), free=BASE_FREE + ["openin"]),
    "single": dict(truth={}, mock=dict(kind="clumps", cones=(-1,)), free=BASE_FREE),
    "lowinc": dict(truth=dict(inc=30.), mock=dict(kind="clumps"), free=BASE_FREE),
    "edgeon": dict(truth=dict(inc=85.), mock=dict(kind="clumps"), free=BASE_FREE),
    "accel": dict(truth=dict(vwind=[100., 160., 220., 280.]), mock=dict(kind="clumps"),
                  free=["vwind", "vdisp_single", "inc", "pa", "openang", "emscale"]),
    "rotation": dict(truth=dict(vrot=50.), mock=dict(kind="clumps"), free=BASE_FREE + ["vrot_single"]),
    "lognormal": dict(truth={}, mock=dict(kind="lognormal", sigma_ln=1.0, corr=20.), free=BASE_FREE),
    "lowsnr": dict(truth={}, mock=dict(kind="clumps", snr=5.), free=BASE_FREE),
}
BASES = {"cells": dict(emission="cells"), "cells-profile": dict(emission="cells", marginal=False),
         "shells": dict(emission="shells"), "cone": dict(emission="cone"), "spaxel": dict(emission="spaxel"),
         "voxel": dict(emission="voxel", marginal=False)}
# the initial wind of the fits: the truth offset (as a user's guess), and the priors of BayesianWind around it
OFFSETS = dict(inc=-5., pa=5., openang=-5., openin=5., vwind=-20., vdisp=5., vrot=10.)


def one_fit(config, seed, basis, threads, nlive, out):
    from wind_mocks import mock_wind
    from bbarolo import BayesianWind
    cfg = CONFIGS[config]
    mock = dict(snr=10.)
    mock.update(cfg["mock"])
    hdu, tr = mock_wind(truth=cfg["truth"], seed=seed, **mock)
    ini = {k: tr[k] for k in ("xpos", "ypos", "vsys", "inc", "pa", "openang", "openin", "htot")}
    ini["vwind"] = np.atleast_1d(tr["vwind"]).astype(float).tolist()
    ini["vdisp"] = float(np.atleast_1d(tr["vdisp"])[0])
    ini["vrot"] = float(np.atleast_1d(tr.get("vrot", 0.))[0])
    for k, d in OFFSETS.items():
        if k in ("vwind",):
            ini[k] = [max(10., v + d) for v in ini[k]]
        elif k in ini:
            ini[k] = ini[k] + d
    ini["openin"] = max(0., ini["openin"]) if tr.get("openin", 0.) > 0 else 0.
    ini["inc"] = min(max(ini["inc"], 1.), 89.)
    radii = tr["radii"]
    f = BayesianWind(hdu, radii=" ".join(str(r) for r in radii), vwind=" ".join(f"{v:g}" for v in ini["vwind"]),
                     **{k: v for k, v in ini.items() if k != "vwind"}, verbose=False, outfolder=os.path.join(HERE, "results", "wind_tmp") + "/")
    t0 = time.time()
    f.run(freepar=cfg["free"], threads=threads, like_kwargs=BASES[basis], sampler_kwargs=dict(n_live=nlive, seed=seed),
          run_kwargs=dict(verbose=False), verbose=False)
    res = dict(config=config, seed=seed, basis=basis, time=time.time() - t0, ess=f.ess, ndim=f.ndim,
               names=f.freepar_names, quantiles=f.quantiles.tolist(), snr=tr["snr"], kind=tr["kind"],
               truth={k: (np.asarray(v).tolist() if np.ndim(v) else v) for k, v in tr.items()
                      if k in ("xpos", "ypos", "vsys", "inc", "pa", "openang", "openin", "htot", "vwind", "vdisp",
                               "vrot", "radii", "noise", "peak")})
    with open(out, "a") as fh:
        fh.write(json.dumps(res) + "\n")
    return res


def truth_of(res, name):
    """The true value of the free parameter name of a result (None if it has none: EMSCALE)."""
    tr = res["truth"]
    base = name.rstrip("0123456789")
    if base in ("emscale", "emslope", "sigma"):
        return None
    if base in ("vwind", "vdisp", "vrot"):
        v = np.broadcast_to(np.asarray(tr.get(base, 0.), float), np.asarray(tr["radii"]).shape)
        if name == base:
            return float(np.mean(v))
        return float(v[int(name[len(base):]) - 1])
    return float(tr.get(name, 0.))


def report(path=RESULTS, out=REPORT):
    rows = [json.loads(x) for x in open(path)] if os.path.isfile(path) else []
    if not rows:
        print("No results.")
        return
    groups = {}
    for r in rows:
        groups.setdefault((r["config"], r["basis"]), []).append(r)
    lines = ["# BAYESIANWIND: recovery of patchy mock winds", "",
             "Mocks: performance/wind_mocks.py (GalMod3D, clumps or log-normal emission, smoothed noise); fits: "
             "BayesianWind with nautilus. For each parameter: the mean bias (posterior median - truth) and its "
             "scatter over the seeds, the rms of the pulls (bias / half-width of the 68% interval: 1 if the errors are "
             "right) and the coverage of the 68% intervals (the fraction of seeds whose interval contains the truth).",
             ""]
    for (config, basis), rs in sorted(groups.items()):
        lines.append(f"## {config}, {basis} ({len(rs)} seeds, {np.mean([r['time'] for r in rs]) / 60:.1f} min per fit)")
        lines.append("")
        lines.append("| parameter | truth | bias | scatter | pull rms | coverage 68% |")
        lines.append("|---|---|---|---|---|---|")
        for i, name in enumerate(rs[0]["names"]):
            tv = truth_of(rs[0], name)
            if tv is None:
                continue
            b, pulls, cov = [], [], []
            for r in rs:
                lo, med, hi = r["quantiles"][i]
                t = truth_of(r, name)
                b.append(med - t)
                hw = max(0.5 * (hi - lo), 1e-12)
                pulls.append((med - t) / hw)
                cov.append(lo <= t <= hi)
            lines.append(f"| {name} | {tv:.4g} | {np.mean(b):+.3g} | {np.std(b):.3g} | "
                         f"{math.sqrt(np.mean(np.square(pulls))):.2f} | {np.mean(cov):.2f} |")
        lines.append("")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configs", default="filled,hollow", help="comma-separated, or 'all'")
    ap.add_argument("--bases", default="cells", help="comma-separated: " + ", ".join(BASES))
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=4, help="fits at the same time")
    ap.add_argument("--threads", type=int, default=8, help="processes of each fit")
    ap.add_argument("--nlive", type=int, default=500)
    ap.add_argument("--out", default=RESULTS)
    ap.add_argument("--quick", action="store_true", help="2 configurations x 2 seeds, n_live 200")
    ap.add_argument("--report", action="store_true", help="only write the report of the results")
    ap.add_argument("--one", nargs=3, metavar=("CONFIG", "SEED", "BASIS"), help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.report:
        report(a.out)
        return
    if a.one:
        one_fit(a.one[0], int(a.one[1]), a.one[2], a.threads, a.nlive, a.out)
        return
    if a.quick:
        a.configs, a.seeds, a.nlive = "filled,hollow", 2, 200
    configs = list(CONFIGS) if a.configs == "all" else a.configs.split(",")
    bases = a.bases.split(",")
    for c in configs:
        if c not in CONFIGS:
            raise SystemExit(f"unknown configuration {c} ({', '.join(CONFIGS)})")
    for b in bases:
        if b not in BASES:
            raise SystemExit(f"unknown basis {b} ({', '.join(BASES)})")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    todo = [(c, s, b) for c in configs for b in bases for s in range(a.seed0, a.seed0 + a.seeds)]
    running = []
    t0 = time.time()
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    while todo or running:
        while todo and len(running) < a.jobs:
            c, s, b = todo.pop(0)
            cmd = [sys.executable, os.path.abspath(__file__), "--one", c, str(s), b, "--threads", str(a.threads),
                   "--nlive", str(a.nlive), "--out", a.out]
            running.append(((c, s, b), subprocess.Popen(cmd, env=env)))
        time.sleep(2)
        for job in list(running):
            if job[1].poll() is not None:
                running.remove(job)
                print(f" done {job[0]} (exit {job[1].returncode}) after {(time.time() - t0) / 60:.1f} min", flush=True)
    report(a.out)


if __name__ == "__main__":
    main()
