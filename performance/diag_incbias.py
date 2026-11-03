#!/usr/bin/env python3
"""
Origin of the inclination bias of 3DFIT at high S/N (section "Origin of the biases" of the
report). Flat disks of the geometry experiment with 45 <= INC < 75 deg, at S/N = 200.

  A. 3DFIT fits from the true INC and PA, with DELTAINC = 15 (so that the minimum is not
     at the limit of the default DELTAINC = 5), changing one ingredient at a time.
  B. The 3DFIT objective function of single rings (the residuals that 3DFIT minimises) with
     VROT, VDISP and PA at the true values, scanned in INC: where is the minimum? For rings
     of different widths, for a uniform disk with a flat rotation curve, and with the rest
     of the disk added to the model of the ring.

    python diag_incbias.py --procs 2         # ~5 minutes with 2 processes

writes results/diag_incbias.json and report/tables/incbias.tex.
"""

import os
import re
import sys
import json
import argparse
import tempfile
import warnings
import numpy as np
from multiprocessing import get_context

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import run_suite as rs                                 # noqa: E402

SNR = 200.
GALAXIES = 6            # first flat disks of the geometry experiment with 45 <= INC < 75
SCAN_GALAXIES = 3
FIT_TESTS = {           # name: (label, changes)
    "default":  ("as the geometry experiment", {}),
    "mc":       ("cube simulated with the Monte-Carlo GALMOD", dict(sim_galmodtype="MC")),
    "thin":     ("razor-thin disk (Z0 = 0)", dict(z0=0.)),
    "analytic": ("noise-free models in the fit (GALMODTYPE = ANALYTIC)", dict(galmodtype="ANALYTIC")),
    "wfunc0":   ("uniform azimuthal weights (WFUNC = 0)", dict(wfunc=0)),
    "local":    ("local normalisation (NORM = LOCAL)", dict(norm="LOCAL")),
    "pix10":    ("10 pixels per beam", dict(ppb=10)),
    "spatial12": ("disk radius = 12 beams", dict(spatial=12.)),
    "radsep1":  ("rings one beam wide", dict(ringwidth=1.)),
    "reverse":  ("reverse-cumulative fit (REVERSE = true)", dict(reverse="true")),
}
SCAN_TESTS = {          # name: (label, ring width / beam, uniform disk, rest of the disk in the model)
    "w05":      ("ring alone, width = beam/2 (as 3DFIT)", 0.5, False, False),
    "w1":       ("ring alone, width = beam", 1., False, False),
    "w2":       ("ring alone, width = 2 beams", 2., False, False),
    "uniform":  ("ring alone, width = beam/2; uniform disk, flat rotation curve", 0.5, True, False),
    "rest":     ("ring + rest of the disk (true values), width = beam/2", 0.5, False, True),
}


def galaxies(n):
    jobs = [j for j in rs.make_jobs(["geometry"], 24, 200, 120, 2026) if j["fitmode"] == "truth"
            and 45 <= j["galaxy"]["inc"] < 75]
    return jobs[:n]


def simulate(g, cg, seed, wd, galmodtype="ANALYTIC", z0=rs.Z0):
    """As run_suite.simulate, with the model type and thickness as options."""
    from bbarolo import GalMod
    from bbarolo.utils import emptyfits
    from astropy.io import fits
    n, nz = cg["npix"], cg["nchan"]
    tmpl = emptyfits([n, n, nz], [-cg["pix"] / 3600., cg["pix"] / 3600., cg["dv"]],
                     beam=cg["beam"] / 3600., bunit="JY/BEAM", obj="diag")
    xc = (n - 1) / 2.
    dr = max(rs.RMAX / 30., cg["pix"])
    R = (np.arange(int(round(rs.RMAX / dr))) + 0.5) * dr
    t = rs.profiles(g, R)
    gm = GalMod(tmpl, radii=R, xpos=xc, ypos=xc, vsys=0., vrot=t["vrot"], vdisp=t["vdisp"], vrad=0., z0=z0,
                inc=t["inc"], pa=t["pa"], dens=t["dens"], ltype=1, norm="NONE", mask="NONE",
                galmodtype=galmodtype, threads=1, verbose=False)
    clean = gm.model()
    rms = float(clean.data.max()) / SNR
    fn = os.path.join(wd, "cube.fits")
    fits.PrimaryHDU(data=gm.diskmodel.addnoise(clean.data, rms, seed=seed), header=tmpl.header).writeto(fn)
    return fn, xc


def run_fit(args):
    """Part A: one fit (galaxy job, test name) -> per-ring median INC offset and VDISP error."""
    job, name = args
    warnings.simplefilter("ignore")
    from bbarolo import FitMod3D
    ch = dict(FIT_TESTS[name][1])
    g = job["galaxy"]
    rs.PIX_PER_BEAM = ch.pop("ppb", 4)
    cg = rs.geometry_of_cube(g, ch.pop("spatial", rs.REF["spatial"]), rs.REF["spectral"])
    z0 = ch.pop("z0", rs.Z0)
    wb = ch.pop("ringwidth", 0.5)
    with tempfile.TemporaryDirectory(prefix="bbdiag_") as wd:
        fn, xc = simulate(g, cg, job["seed"], wd, ch.pop("sim_galmodtype", "ANALYTIC"), z0)
        w = wb * cg["beam"]
        kw = dict(nradii=int(round(rs.RMAX / w)), radsep=w, xpos=xc, ypos=xc, vsys=0., z0=z0, ltype=1,
                  mask=rs.MASK, free="VROT VDISP INC PA", twostage=True, inc=g["inc"], pa=g["pa"], deltainc=15.)
        kw.update(ch)
        res = FitMod3D(fn, threads=1, verbose=False, plots=-1, outfolder=os.path.join(wd, "out"), **kw).run()
        r = res["rings"]
    return dict(test=name, gid=g["id"], inc=g["inc"], dinc=float(np.median(np.array(r["inc"]) - g["inc"])),
                ds=float(np.median(np.array(r["vdisp"]) / g["sigma"] - 1)))


def run_scan(args):
    """Part B: INC offset of the minimum of the objective of each ring (R >= 0.35 RMAX)."""
    job, name = args
    warnings.simplefilter("ignore")
    from bbarolo import FitMod3D
    from bbarolo.rings import Rings
    label, wb, uniform, rest = SCAN_TESTS[name]
    g = dict(job["galaxy"])
    if uniform:
        g.update(h=1e4, rt=1e-4)
    rs.PIX_PER_BEAM = 4
    cg = rs.geometry_of_cube(g, rs.REF["spatial"], rs.REF["spectral"])
    w = wb * cg["beam"]
    n = int(rs.RMAX / w)
    R = (np.arange(n) + 0.5) * w
    t = rs.profiles(g, R)
    out = []
    with tempfile.TemporaryDirectory(prefix="bbdiag_") as wd:
        fn, xc = simulate(g, cg, job["seed"], wd)
        # a quick fit only to set up the Galfit object (rings, mask, model engine)
        kw = dict(nradii=n, radsep=w, xpos=xc, ypos=xc, vsys=0., z0=rs.Z0, ltype=1, mask=rs.MASK, inc=g["inc"],
                  pa=g["pa"], vrot=list(t["vrot"]), vdisp=g["sigma"], free="VSYS", twostage=False,
                  galmodtype="ANALYTIC")
        gf = FitMod3D(fn, threads=1, verbose=False, plots=-1, outfolder=os.path.join(wd, "out"), **kw).run()["galfit"]

        def disk(r0, r1, blo, bhi):            # model of the disk between r0 and r1 (true values)
            dr = max(rs.RMAX / 30., cg["pix"])
            rr = np.arange(r0, r1 + 1e-6, dr)
            if rr[-1] < r1 - 1e-6:
                rr = np.append(rr, r1)
            tt = rs.profiles(g, rr)
            d = Rings()
            d.setRings(radii=list(rr), xpos=xc, ypos=xc, vsys=0., vrot=list(tt["vrot"]), vdisp=g["sigma"], vrad=0.,
                       vvert=0., dvdz=0., zcyl=0., dens=list(tt["dens"]), z0=rs.Z0, inc=g["inc"], phi=g["pa"])
            return gf.model.setup.build(d, blo, bhi, layout="yxz", galmodtype="ANALYTIC")

        dis = np.arange(-12., 6.01, 0.5)
        for ir in range(n):
            if R[ir] < 0.35 * rs.RMAX:
                continue
            d0 = gf._make_dring(ir)
            other = None
            if rest:
                blo, bhi = gf.model.box(d0)
                r1, r2 = float(d0.radii[0]), float(d0.radii[-1])
                other = disk(0., r1, blo, bhi)
                if r2 < rs.RMAX - 1:
                    other += disk(r2, rs.RMAX, blo, bhi)
            F = []
            for di in dis:
                d = gf._make_dring(ir)
                d.vrot[:] = t["vrot"][ir]
                d.vdisp[:] = g["sigma"]
                d.inc[:] = g["inc"] + di
                d.phi[:] = g["pa"]
                d.vsys[:] = 0.
                d.dens[:] = t["dens"][ir]
                F.append(gf.getFuncValue(d, None if other is None else other.copy()))
            out.append(float(dis[int(np.argmin(F))]))
    return dict(test=name, gid=g["id"], inc=g["inc"], dinc_rings=out, dinc=float(np.median(out)) if out else np.nan)


def table(fits, scans, path):
    def rng(v):
        v = np.asarray(v, float)
        return f"{np.median(v):+.1f} & [{v.min():+.1f}, {v.max():+.1f}]"
    lines = [r"\begin{tabular}{l rl r}", r"\toprule",
             r" & \multicolumn{2}{c}{INC offset ($^\circ$)} & $\sigma$ error \\",
             r"test & median & [range] & (\%) \\", r"\midrule",
             r"\multicolumn{4}{l}{\emph{A. 3DFIT from the true INC and PA, DELTAINC = 15$^\circ$ "
             r"(per galaxy: median over the rings)}} \\"]
    for name, (label, _) in FIT_TESTS.items():
        s = [f for f in fits if f["test"] == name]
        if s:
            lines.append(f"{label} & {rng([f['dinc'] for f in s])} & "
                         f"{100 * np.median([f['ds'] for f in s]):+.0f} \\\\")
    lines += [r"\addlinespace", r"\multicolumn{4}{l}{\emph{B. Minimum of the objective of single rings, other "
              r"parameters at the true values (per ring)}} \\"]
    for name, (label, *_) in SCAN_TESTS.items():
        s = [x for f in scans if f["test"] == name for x in f["dinc_rings"]]
        if s:
            lines.append(f"{label} & {rng(s)} & \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(path, "w") as f:
        f.write(re.sub(r"(?<=[\s&\[(])-(?=\d)", "$-$", "\n".join(lines) + "\n"))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=os.path.join(HERE, "results", "diag_incbias.json"))
    ap.add_argument("--table", default=os.path.join(HERE, "report", "tables", "incbias.tex"))
    ap.add_argument("--procs", type=int, default=2)
    ap.add_argument("--tests", nargs="+", help="run only these tests (the others are kept from --out)")
    ap.add_argument("--table-only", action="store_true", help="only rebuild the table from --out")
    args = ap.parse_args()
    if args.table_only:
        with open(args.out) as f:
            d = json.load(f)
        table(d["fits"], d["scans"], args.table)
        return
    fits, scans = [], []
    if args.tests and os.path.isfile(args.out):
        with open(args.out) as f:
            old = json.load(f)
        fits = [r for r in old["fits"] if r["test"] in FIT_TESTS and r["test"] not in args.tests]
        scans = [r for r in old["scans"] if r["test"] in SCAN_TESTS and r["test"] not in args.tests]
    sel = (lambda t: True) if not args.tests else (lambda t: t in args.tests)
    fjobs = [(j, t) for t in FIT_TESTS if sel(t) for j in galaxies(GALAXIES)]
    sjobs = [(j, t) for t in SCAN_TESTS if sel(t) for j in galaxies(SCAN_GALAXIES)]
    with get_context("fork").Pool(args.procs, maxtasksperchild=1) as pool:
        for i, r in enumerate(pool.imap_unordered(run_scan, sjobs), 1):
            scans.append(r)
            print(f"[scan {i}/{len(sjobs)}] {r['test']:8s} g{r['gid']:<3d} INC {r['inc']:5.1f}: "
                  f"minimum at {r['dinc']:+.1f} deg", flush=True)
        for i, r in enumerate(pool.imap_unordered(run_fit, fjobs), 1):
            fits.append(r)
            print(f"[fit {i}/{len(fjobs)}] {r['test']:9s} g{r['gid']:<3d} INC {r['inc']:5.1f}: "
                  f"dINC {r['dinc']:+.1f} deg, dsigma {100 * r['ds']:+.0f}%", flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(dict(snr=SNR, fits=fits, scans=scans), f, indent=1)
    os.makedirs(os.path.dirname(args.table), exist_ok=True)
    table(fits, scans, args.table)
    print("->", args.out, args.table)


if __name__ == "__main__":
    main()
