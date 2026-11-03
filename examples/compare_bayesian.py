#!/usr/bin/env python3
"""
Run the same Bayesian fit (BayesianBBarolo, dynesty) with the original C++-based
pyBBarolo (ctypes + libBBarolo) and with the pure-Python bbarolo, and compare the
likelihood, the posteriors and the run times.

Run from the BBarolo folder (the one with pyBBarolo/libBBarolo.* and examples/):

    python pyBBarolo_pure/examples/compare_bayesian.py [--threads 2] [--nlive 100]

Each version runs in its own process. The original module has the old interface
(init/set_options/compute), the pure-Python one the interface of the bbarolo tasks.
Options: --cube, --cpp-dir (folder containing the original pyBBarolo/ package),
--threads, --nlive, --dlogz, --seed, --skip-cpp. See --help.

Both versions are run with sigma = 1 and no beam factor, as the original module (for the
pure-Python version: like_kwargs=dict(sigma=1, noisecorr=1)); by default it uses instead the
measured noise and a factor 1/(correlation volume of the noise). Its likelihood also differs
from the original in the data (not masked, in the mask dilated by one beam) and in the
density profile (corrected for the beam), so the comparison shows the effect of these
changes. Until 3 November 2026 these two could be switched back to the original ones, and the
likelihoods were then identical (see the README). The log-likelihood of the pure-Python
version includes a constant normalisation term, so the scans are shown relative to their
first value.
"""

import os
import sys
import json
import argparse
import subprocess
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                       # .../pyBBarolo_pure
BBDIR = os.path.dirname(ROOT)                      # .../BBarolo

WORKER = r'''
import sys, json, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
cfg = json.loads(sys.argv[1])
sys.path.insert(0, cfg["path"])
ring = dict(xpos=77, ypos=77, vsys=132.8, vrot=100, vdisp=10, inc=60, z0=10)


def make():
    if cfg["label"] == "C++":                 # original module: old interface
        from pyBBarolo.bayesian import BayesianBBarolo
        f = BayesianBBarolo(cfg["cube"])
        f.init(radii=np.array(cfg["radii"]), phi=123.7, **ring)
        f.set_options(mask="SEARCH", outfolder=cfg["outfolder"], plots=0)
    else:                                     # pure-Python bbarolo
        from bbarolo import BayesianBBarolo
        f = BayesianBBarolo(cfg["cube"], radii=np.array(cfg["radii"]), pa=123.7, mask="SEARCH",
                            outfolder=cfg["outfolder"], plots=0, ringmode="SHIFTED", **ring)   # C++ rings
    return f


def sample(f, **kw):
    if cfg["label"] == "C++":
        return f.compute(threads=cfg["threads"], **kw)
    return f.run(threads=cfg["threads"], **kw)


# likelihood scan through the vrot of the second ring
f = make()
LK = dict(sigma=1.)                     # sigma = 1 and no beam factor, as the original BayesianBBarolo
PY = dict(LK, noisecorr=1) if cfg["label"] != "C++" else LK     # (noisecorr=1: independent voxels)
if cfg["label"] != "C++":
    f._set_like_options(PY)
f._setup(cfg["freepar"])
th = np.array(cfg["theta0"])
scan = []
for v in cfg["scan"]:
    t = th.copy(); t[1] = v
    scan.append(float(f._log_likelihood(t, **LK)))
t0 = time.time()
for i in range(10): f._log_likelihood(th, **LK)
tcall = (time.time() - t0) / 10
f2 = make()
t0 = time.time()
sample(f2, freepar=cfg["freepar"], method="dynesty", dynamic=False, like_kwargs=PY,
       sampler_kwargs=dict(nlive=cfg["nlive"], rstate=np.random.default_rng(cfg["seed"])),
       run_kwargs=dict(dlogz=cfg["dlogz"]))
dt = time.time() - t0
q = np.quantile(f2.samples_equal, [0.16, 0.5, 0.84], axis=0)
print("JSON" + json.dumps(dict(names=f2.freepar_names, q=q.tolist(), time=dt, ncall=int(f2.results.ncall.sum()),
                               tcall=tcall, scan=scan, logz=float(f2.results.logz[-1]))))
'''


def run(label, path, cfg):
    cfg = dict(cfg, path=path, label=label, outfolder=os.path.join(cfg["outdir"], label))
    print(f" Running the {label} version...", flush=True)
    p = subprocess.run([sys.executable, "-c", WORKER, json.dumps(cfg)], capture_output=True, text=True)
    for line in p.stdout.splitlines():
        if line.startswith("JSON"):
            return json.loads(line[4:])
    print(p.stdout[-2000:], p.stderr[-3000:])
    sys.exit(f"The {label} run failed.")


def main():
    ap = argparse.ArgumentParser(description="C++ vs Python BayesianBBarolo",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--cube", default=os.path.join(BBDIR, "examples", "ngc2403.fits"))
    ap.add_argument("--cpp-dir", default=BBDIR, help="folder that contains the original pyBBarolo/ package")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--nlive", type=int, default=100)
    ap.add_argument("--dlogz", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--outdir", default="bayes_comparison")
    ap.add_argument("--skip-cpp", action="store_true", help="run only the Python version")
    args = ap.parse_args()
    cfg = dict(cube=os.path.abspath(args.cube), radii=[60., 180., 300.], freepar=['vrot', 'vdisp_single'],
               theta0=[80., 105., 120., 13.], scan=list(np.arange(60., 151., 15.)), threads=args.threads,
               nlive=args.nlive, dlogz=args.dlogz, seed=args.seed, outdir=os.path.abspath(args.outdir))
    res = {}
    if not args.skip_cpp:
        res["C++"] = run("C++", os.path.abspath(args.cpp_dir), cfg)
    res["Python"] = run("Python", ROOT, cfg)

    print("\n Log-likelihood scan (vrot of ring 2, other parameters fixed), relative to the first value:")
    print("   vrot2 " + "".join(f"{l:>12s}" for l in res))
    for i, v in enumerate(cfg["scan"]):
        print(f"   {v:5.0f} " + "".join(f"{res[l]['scan'][i] - res[l]['scan'][0]:12.4f}" for l in res))
    print("\n Posteriors (median, -/+ 1 sigma):")
    names = res["Python"]["names"]
    print("   " + " " * 8 + "".join(f"{l:>26s}" for l in res))
    for i, n in enumerate(names):
        row = ""
        for l in res:
            q = np.array(res[l]["q"])[:, i]
            row += f"{q[1]:12.2f} -{q[1] - q[0]:5.2f} +{q[2] - q[1]:5.2f}"
        print(f"   {n:8s}" + row)
    print("\n" + "".join(f" {l}: {r['time']:.1f} s, {r['ncall']} likelihood calls, "
                         f"{r['tcall'] * 1000:.1f} ms per call, logZ = {r['logz']:.2f}\n" for l, r in res.items()))
    if "C++" in res:
        print(" (logZ differs by a constant: the Python log-likelihood includes the normalisation of all\n"
              "  the voxels in the likelihood region, the C++-based one only 0.5*log(2*pi*sigma^2).)")
        print(f" Speed-up of the sampling: x{res['C++']['time'] / res['Python']['time']:.2f}")


if __name__ == "__main__":
    main()
