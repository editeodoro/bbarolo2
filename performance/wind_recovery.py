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
    # the same mocks with the current defaults (EMSLOPE fitted), and fitted with one cone only (cones=approaching)
    "single-both": dict(truth={}, mock=dict(kind="clumps", cones=(-1,)), free=BASE_FREE),
    "single-cone": dict(truth={}, mock=dict(kind="clumps", cones=(-1,)), free=BASE_FREE,
                        like=dict(cones="approaching")),
    "lowinc": dict(truth=dict(inc=30.), mock=dict(kind="clumps"), free=BASE_FREE),
    "edgeon": dict(truth=dict(inc=85.), mock=dict(kind="clumps"), free=BASE_FREE),
    "accel": dict(truth=dict(vwind=[100., 160., 220., 280.]), mock=dict(kind="clumps"),
                  free=["vwind", "vdisp_single", "inc", "pa", "openang", "emscale"]),
    "rotation": dict(truth=dict(vrot=50.), mock=dict(kind="clumps"), free=BASE_FREE + ["vrot_single"]),
    "lognormal": dict(truth={}, mock=dict(kind="lognormal", sigma_ln=1.0, corr=20.), free=BASE_FREE),
    "lowsnr": dict(truth={}, mock=dict(kind="clumps", snr=5.), free=BASE_FREE),
    # emission falling as r^-2 (the clumps and the cone), fitted with EMSLOPE fixed at 0 or free
    "slope": dict(truth={}, mock=dict(kind="clumps", slope=2., snr=20.), free=BASE_FREE, like=dict(emslope=0.)),
    "slope-free": dict(truth={}, mock=dict(kind="clumps", slope=2., snr=20.), free=BASE_FREE + ["emslope"]),
}
BASES = {"cells": dict(emission="cells"), "cells-profile": dict(emission="cells", marginal=False),
         "shells": dict(emission="shells", emslope=0.), "cone": dict(emission="cone"), "spaxel": dict(emission="spaxel"),
         "voxel": dict(emission="voxel", emslope=0.)}
# the initial wind of the fits: the truth offset (as a user's guess), and the priors of BayesianWind around it
OFFSETS = dict(inc=-5., pa=5., openang=-5., openin=5., vwind=-20., vdisp=5., vrot=10.)


def one_fit(config, seed, basis, threads, nlive, out, neff=2000):
    from wind_mocks import mock_wind
    from bbarolo import BayesianWind
    cfg = CONFIGS[config]
    mock = dict(snr=10.)
    mock.update(cfg["mock"])
    hdu, tr = mock_wind(truth=cfg["truth"], seed=seed, **mock)
    ini = {k: tr[k] for k in ("xpos", "ypos", "vsys", "inc", "pa", "openang", "openin", "conelen")}
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
                     **{k: v for k, v in ini.items() if k != "vwind"}, verbose=False,
                     outfolder=os.path.join(HERE, "results", "wind_tmp") + "/")
    t0 = time.time()
    f.run(freepar=cfg["free"], threads=threads, like_kwargs={**BASES[basis], **cfg.get("like", {})},
          sampler_kwargs=dict(n_live=nlive, seed=seed),
          run_kwargs=dict(verbose=False, n_eff=neff), verbose=False)
    res = dict(config=config, seed=seed, basis=basis, time=time.time() - t0, ess=f.ess, ndim=f.ndim,
               names=f.freepar_names, quantiles=f.quantiles.tolist(), snr=tr["snr"], kind=tr["kind"],
               truth={k: (np.asarray(v).tolist() if np.ndim(v) else v) for k, v in tr.items()
                      if k in ("xpos", "ypos", "vsys", "inc", "pa", "openang", "openin", "conelen", "vwind", "vdisp",
                               "vrot", "radii", "noise", "peak")})
    with open(out, "a") as fh:
        fh.write(json.dumps(res) + "\n")
    return res


def truth_of(res, name):
    """The true value of the free parameter name of a result (None if it has none: EMSCALE)."""
    tr = res["truth"]
    base = name.rstrip("0123456789")
    if base in ("emscale", "emclump", "emratio", "emslope", "sigma"):
        return None
    if base in ("vwind", "vdisp", "vrot"):
        v = np.broadcast_to(np.asarray(tr.get(base, 0.), float), np.asarray(tr["radii"]).shape)
        if name == base:
            return float(np.mean(v))
        return float(v[int(name[len(base):]) - 1])
    return float(tr.get(name, 0.))


def _stats(rs, name):
    """(mean bias, scatter of the bias, rms pull, coverage of the 68% interval) of a parameter over the results rs,
    the parameters per shell (vwind1, vwind2, ...) pooled under their base name; None if no result has it."""
    b, pulls, cov = [], [], []
    for r in rs:
        for i, n in enumerate(r["names"]):
            if n.rstrip("0123456789") != name:
                continue
            t = truth_of(r, n)
            if t is None:
                continue
            lo, med, hi = r["quantiles"][i]
            b.append(med - t)
            pulls.append((med - t) / max(0.5 * (hi - lo), 1e-12))
            cov.append(lo <= t <= hi)
    if not b:
        return None
    return np.mean(b), np.std(b), math.sqrt(np.mean(np.square(pulls))), np.mean(cov)


PARS = ("vwind", "vdisp", "vrot", "inc", "pa", "openang", "openin")


def _load(paths):
    rows = []
    for p in paths:
        if os.path.isfile(p):
            rows += [json.loads(x) for x in open(p) if x.strip()]
    return rows


def report(path=RESULTS, out=REPORT):
    """The report of the results (path, and the other results files wind_recovery*.jsonl of its folder: the tests of
    the cell size and of the model error), in Markdown (out)."""
    import glob
    paths = [path] + sorted(p for p in glob.glob(os.path.join(os.path.dirname(path), "wind_recovery*.jsonl"))
                            if os.path.abspath(p) != os.path.abspath(path))
    rows = _load(paths)
    if not rows:
        print("No results.")
        return
    groups = {}
    for r in rows:
        groups.setdefault((r["config"], r["basis"]), []).append(r)
    L = ["# BAYESIANWIND: recovery of patchy mock winds", "",
         "Mocks: performance/wind_mocks.py (GalMod3D, an engine independent of the fit; clumps or a log-normal field "
         "in the cones; noise smoothed by the beam, peak S/N 10 unless noted). Fits: BayesianWind with nautilus "
         "(300 live points, 2000 effective samples), the initial wind offset from the truth (INC -5, PA +5, OPENANG "
         "-5, VWIND -20 km/s, VDISP +5 km/s) and the default priors around it; EMSLOPE fixed at 0 (the slope of the "
         "mocks; fitted by default since) but in the configuration slope-free. For each parameter: the mean bias "
         "(posterior median - truth), the rms of the pulls (bias / half-width of the 68% interval: 1 if the "
         "intervals are right) and the coverage of the 68% intervals over the seeds. EMSCALE, EMCLUMP and EMRATIO "
         "have no true value and are left out.", ""]
    # 1. the configurations with the default basis
    L += ["## The configurations (emission cells, marginal likelihood: the defaults)", "",
          "| configuration | seeds | " + " | ".join(PARS) + " |", "|---|---|" + "---|" * len(PARS)]
    for (config, basis), rs in groups.items():
        if basis != "cells" or config.startswith("slope") or config in ("single-cone", "single-both"):
            continue
        cells = []
        for p in PARS:
            st = _stats(rs, p)
            cells.append("" if st is None else f"{st[0]:+.1f} ({st[2]:.1f}, {st[3]:.2f})")
        L.append(f"| {config} | {len(rs)} | " + " | ".join(cells) + " |")
    allc = [r for (c, b), rs in groups.items()
            if b == "cells" and not c.startswith("slope") and c not in ("single-cone", "single-both") for r in rs]
    tot = []
    for p in PARS:
        st = _stats(allc, p)
        if st is not None:
            tot.append(f"{p}: pull rms {st[2]:.2f}, coverage {st[3]:.2f}")
    L += ["", "Each entry: bias (rms pull, coverage). All the configurations: " + "; ".join(tot) + ".", ""]
    # 2. EMSLOPE
    if any(c.startswith("slope") for c, b in groups):
        L += ["## Emission falling as r^-2: EMSLOPE fixed at 0 or free", "",
              "| configuration | seeds | " + " | ".join(PARS[:6]) + " |", "|---|---|" + "---|" * 6]
        for config in ("slope", "slope-free"):
            rs = groups.get((config, "cells"))
            if not rs:
                continue
            cells = []
            for p in PARS[:6]:
                st = _stats(rs, p)
                cells.append("" if st is None else f"{st[0]:+.1f} ({st[2]:.1f}, {st[3]:.2f})")
            L.append(f"| {config} | {len(rs)} | " + " | ".join(cells) + " |")
        L += ["", "slope: the clump mock with its emission falling as r^-2 (peak S/N 20), EMSLOPE fixed at 0 (the "
              "prior mean uniform in radius); slope-free: EMSLOPE fitted (the default since).", ""]
    # 3. one cone
    if ("single-cone", "cells") in groups:
        L += ["## A wind with one cone: both cones in the model (EMRATIO) or one (cones=approaching)", "",
              "| configuration | seeds | min per fit | " + " | ".join(PARS[:6]) + " |", "|---|---|---|" + "---|" * 6]
        for config in ("single", "single-both", "single-cone"):
            rs = groups.get((config, "cells"))
            if not rs:
                continue
            cells = []
            for p in PARS[:6]:
                st = _stats(rs, p)
                cells.append("" if st is None else f"{st[0]:+.1f} ({st[2]:.1f}, {st[3]:.2f})")
            mins = np.mean([r['time'] for r in rs]) / 60
            L.append(f"| {config} | {len(rs)} | {mins:.1f} | " + " | ".join(cells) + " |")
        L += ["", "The same mocks (only the approaching cone emits): single and single-both fit both cones with "
              "EMRATIO free (single with EMSLOPE fixed at 0, single-both with the current defaults: EMSLOPE fitted); "
              "single-cone only the approaching cone (LIKEOPTS cones=approaching, EMSLOPE fitted).", ""]
    # 4. the bases of the emission
    L += ["## The descriptions of the emission (option emission, marginal or profile)", "",
          "| configuration | basis | seeds | min per fit | " + " | ".join(PARS[:6]) + " |",
          "|---|---|---|---|" + "---|" * 6]
    order = ["cells", "cells-profile", "shells", "spaxel", "voxel", "voxel-kappa", "cells22", "cells-df"]
    for config in ("filled", "hollow", "lognormal", "edgeon"):
        for basis in order:
            rs = groups.get((config, basis))
            if not rs:
                continue
            cells = []
            for p in PARS[:6]:
                st = _stats(rs, p)
                cells.append("" if st is None else f"{st[0]:+.1f} ({st[2]:.1f}, {st[3]:.2f})")
            L.append(f"| {config} | {basis} | {len(rs)} | {np.mean([r['time'] for r in rs]) / 60:.1f} | "
                     + " | ".join(cells) + " |")
    L += ["", "cells-profile: the maximum over the emission (no Occam factor); voxel: the weighting of MoKA-3D (the "
          "model is the data wherever the uniform cone reaches) with the detection likelihood (the voxels above 3 "
          "times the noise, each detected with the probability of a log-normal flux of mean the uniform cone, scatter "
          "EMCLUMP, plus the noise); voxel-kappa: the earlier voxel likelihood, the chi^2 of the voxels the cone does "
          "not reach (the kappa of MoKA-3D without its cuts on the percentile velocities; code before November 10, "
          "2026); cells22: cells of 22 arcsec instead of the beam (30 arcsec); cells-df: the model error datafrac = "
          "0.3.", ""]
    # 5. the details
    L += ["## Details", ""]
    for (config, basis), rs in sorted(groups.items()):
        L.append(f"### {config}, {basis} ({len(rs)} seeds, {np.mean([r['time'] for r in rs]) / 60:.1f} min per fit)")
        L += ["", "| parameter | truth | bias | scatter | pull rms | coverage 68% |", "|---|---|---|---|---|---|"]
        for name in rs[0]["names"]:
            tv = truth_of(rs[0], name)
            if tv is None:
                continue
            b, pulls, cov = [], [], []
            for r in rs:
                i = r["names"].index(name)
                lo, med, hi = r["quantiles"][i]
                t = truth_of(r, name)
                b.append(med - t)
                pulls.append((med - t) / max(0.5 * (hi - lo), 1e-12))
                cov.append(lo <= t <= hi)
            L.append(f"| {name} | {tv:.4g} | {np.mean(b):+.3g} | {np.std(b):.3g} | "
                     f"{math.sqrt(np.mean(np.square(pulls))):.2f} | {np.mean(cov):.2f} |")
        L.append("")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"Report written to {out} ({len(rows)} fits).")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configs", default="filled,hollow", help="comma-separated, or 'all'")
    ap.add_argument("--bases", default="cells", help="comma-separated: " + ", ".join(BASES))
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=4, help="fits at the same time")
    ap.add_argument("--threads", type=int, default=8, help="processes of each fit")
    ap.add_argument("--nlive", type=int, default=500)
    ap.add_argument("--neff", type=int, default=2000, help="effective sample size at which nautilus stops")
    ap.add_argument("--out", default=RESULTS)
    ap.add_argument("--quick", action="store_true", help="2 configurations x 2 seeds, n_live 200")
    ap.add_argument("--report", action="store_true", help="only write the report of the results")
    ap.add_argument("--one", nargs=3, metavar=("CONFIG", "SEED", "BASIS"), help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.report:
        report(a.out)
        return
    if a.one:
        one_fit(a.one[0], int(a.one[1]), a.one[2], a.threads, a.nlive, a.out, a.neff)
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
                   "--nlive", str(a.nlive), "--neff", str(a.neff), "--out", a.out]
            running.append(((c, s, b), subprocess.Popen(cmd, env=env)))
        time.sleep(2)
        for job in list(running):
            if job[1].poll() is not None:
                running.remove(job)
                print(f" done {job[0]} (exit {job[1].returncode}) after {(time.time() - t0) / 60:.1f} min", flush=True)
    report(a.out)


if __name__ == "__main__":
    main()
