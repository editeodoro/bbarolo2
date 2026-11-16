#!/usr/bin/env python3
"""
3DFIT with REVERSE (the rings fitted again from the outside in, each with the model of the whole disk) against
the fits of single rings. Flat disks of the geometry experiment fitted from the true INC and PA with DELTAINC = 15
(as diag_incbias.py), FREE = VROT VDISP INC PA, two stages:

  mid    twelve disks with 45 <= INC < 75 deg (the first six are those of diag_incbias.py), at peak S/N 10, 20,
         40 and 200;
  steep  twelve disks of the geometry experiment with INC set to 72-80 deg, at peak S/N 20 and 40;
  edge   the same disks with INC set to 80-88 deg, at peak S/N 20 and 40.

Each with the region of the rings ANNULUS or STRIP (RINGREGION), and with other masks and normalisations. For
each fit: the median over the fitted rings of the INC offset, of the VDISP error and of the VROT error
(relative), and the time.

    python diag_context.py --procs 60          # writes results/diag_context.json and prints a summary
"""

import os
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
import diag_incbias as di                              # noqa: E402

TESTS = {                   # name: (label, changes; ringwidth in beams)
    "default": ("rings beam/2, one pass", {}),
    "w1": ("rings one beam wide, one pass", dict(ringwidth=1.)),
    "reverse": ("REVERSE", dict(reverse="true")),
    "reverse_sw2": ("REVERSE, SWEEPS 2", dict(reverse="true", sweeps=2)),
    "reverse_strip": ("REVERSE, RINGREGION STRIP", dict(reverse="true", ringregion="STRIP")),
    "reverse_sw2_strip": ("REVERSE, SWEEPS 2, RINGREGION STRIP", dict(reverse="true", sweeps=2, ringregion="STRIP")),
    "smooth": ("MASK SMOOTH&SEARCH", dict(mask="SMOOTH&SEARCH")),
    "reverse_smooth": ("REVERSE, MASK SMOOTH&SEARCH", dict(reverse="true", mask="SMOOTH&SEARCH")),
    "reverse_strip_smooth": ("REVERSE, STRIP, MASK SMOOTH&SEARCH",
                             dict(reverse="true", ringregion="STRIP", mask="SMOOTH&SEARCH")),
    "local": ("NORM LOCAL", dict(norm="LOCAL")),
    "reverse_local": ("REVERSE, NORM LOCAL", dict(reverse="true", norm="LOCAL")),
}
SETS = {"mid": dict(snrs=(10., 20., 40., 200.), tests=list(TESTS)),
        "steep": dict(snrs=(20., 40.), tests=list(TESTS)),
        "edge": dict(snrs=(20., 40.), tests=list(TESTS))}
NGAL = 12
EDGE_INC = tuple(np.linspace(80., 88., NGAL))
STEEP_INC = tuple(np.linspace(72., 80., NGAL))


def galaxies(kind):
    if kind == "mid":
        return [j["galaxy"] for j in di.galaxies(NGAL)]
    gs = []
    for k, inc in enumerate(EDGE_INC if kind == "edge" else STEEP_INC):
        g = rs.draw_galaxy("geometry", 100 + k, 2026)
        g.update(id=100 + k + (0 if kind == "edge" else 100), inc=inc)
        gs.append(g)
    return gs


def run_fit(args):
    kind, g, snr, name = args
    warnings.simplefilter("ignore")
    from bbarolo import FitMod3D
    ch = dict(TESTS[name][1])
    di.SNR = snr
    rs.PIX_PER_BEAM = 4
    cg = rs.geometry_of_cube(g, rs.REF["spatial"], rs.REF["spectral"])
    w = ch.pop("ringwidth", 0.5) * cg["beam"]
    with tempfile.TemporaryDirectory(prefix="bbctx_") as wd:
        fn, xc = di.simulate(g, cg, 1000 + g["id"], wd)
        kw = dict(nradii=int(round(rs.RMAX / w)), radsep=w, xpos=xc, ypos=xc, vsys=0., z0=rs.Z0, ltype=1,
                  mask=rs.MASK, free="VROT VDISP INC PA", twostage=True, inc=g["inc"], pa=g["pa"], deltainc=15.)
        kw.update(ch)
        t0 = time.time()
        res = FitMod3D(fn, threads=1, verbose=False, plots=-1, outfolder=os.path.join(wd, "out"), **kw).run()
        dt = time.time() - t0
        r = res["rings"]
        r = r[np.asarray(r["fitok"], bool)]
    R = np.asarray(r["radii"], float)
    t = rs.profiles(g, R)
    return dict(set=kind, snr=snr, test=name, gid=g["id"], inc=g["inc"], nfit=len(R), time=dt,
                dinc=float(np.median(np.asarray(r["inc"]) - g["inc"])) if len(R) else np.nan,
                ds=float(np.median(np.asarray(r["vdisp"]) / g["sigma"] - 1)) if len(R) else np.nan,
                dv=float(np.median(np.abs(np.asarray(r["vrot"]) / t["vrot"] - 1))) if len(R) else np.nan)


def summary(rows):
    out = []
    for kind, s in SETS.items():
        for snr in s["snrs"]:
            out.append(f"\n{kind} disks, peak S/N {snr:g} (median over the galaxies [range]):")
            out.append(f"  {'test':40s} {'dINC (deg)':>22s} {'dVDISP':>8s} {'|dVROT|':>8s} {'time':>7s}")
            for name in s["tests"]:
                x = [r for r in rows if r["set"] == kind and r["snr"] == snr and r["test"] == name]
                if not x:
                    continue
                d = np.array([r["dinc"] for r in x])
                ds, dv = 100 * np.nanmedian([r["ds"] for r in x]), 100 * np.nanmedian([r["dv"] for r in x])
                out.append(f"  {TESTS[name][0]:40s} {np.nanmedian(d):+6.1f} "
                           f"[{np.nanmin(d):+5.1f}, {np.nanmax(d):+5.1f}]"
                           f" {ds:+7.0f}% {dv:7.1f}% {np.median([r['time'] for r in x]):6.0f}s")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=os.path.join(HERE, "results", "diag_context.json"))
    ap.add_argument("--procs", type=int, default=2)
    ap.add_argument("--sets", nargs="+", default=list(SETS))
    ap.add_argument("--tests", nargs="+", help="only these tests")
    ap.add_argument("--snrs", nargs="+", type=float, help="only these peak S/N")
    args = ap.parse_args()
    jobs = [(kind, g, snr, t) for kind in args.sets for snr in SETS[kind]["snrs"] for t in SETS[kind]["tests"]
            if (not args.tests or t in args.tests) and (not args.snrs or snr in args.snrs) for g in galaxies(kind)]
    rows = []
    with get_context("fork").Pool(args.procs, maxtasksperchild=1) as pool:
        for i, r in enumerate(pool.imap_unordered(run_fit, jobs), 1):
            rows.append(r)
            print(f"[{i}/{len(jobs)}] {r['set']} S/N {r['snr']:g} {r['test']:22s} g{r['gid']:<3d} INC {r['inc']:5.1f}: "
                  f"dINC {r['dinc']:+.1f}, dVDISP {100 * r['ds']:+.0f}%, |dVROT| {100 * r['dv']:.1f}% "
                  f"({r['time']:.0f} s)", flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(rows, f, indent=1)
    print(summary(rows))
    print("->", args.out)


if __name__ == "__main__":
    main()
