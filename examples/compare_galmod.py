#!/usr/bin/env python3
"""
Compare the C++ BBarolo and the pure-Python BBarolo on N random GALMOD models.

For each of N random disk galaxies (rotation curve, dispersion, geometry, thickness,
density profile, radial motions) the script:

  1. writes the rings to a text file and a GALMOD parameter file;
  2. runs the C++ executable:          BBarolo -p model.par
  3. runs the Python executable:       python run_bbarolo.py -p model.par   (same file)
  4. builds the same model with the Python interface (bbarolo.GalMod, ring arrays
     instead of the ring file, outputs in a temporary folder);
  5. optionally runs the C++ code again with another random seed, to measure the
     Monte-Carlo noise of GALMOD itself (the "baseline");
  6. compares the C++ and Python cubes: total flux, global spectrum, velocity and
     dispersion fields, and the voxel-by-voxel residual (to be compared with the
     C++ vs C++ baseline, since MC models never match bit for bit).

The Python models use the rings of the C++ code (RINGMODE=SHIFTED; see --ringmode).
It prints a table, saves results.csv and a summary figure in the output folder.

Usage (from anywhere; paths default to the BBarolo folder that contains pyBBarolo_pure):

    python pyBBarolo_pure/examples/compare_galmod.py -n 10
    python compare_galmod.py -n 20 --threads 4 --cpp /path/to/BBarolo \\
                             --template /path/to/cube.fits --outdir galmod_test
    python compare_galmod.py -n 5 --template synthetic      # a synthetic empty cube

Options: see ``--help``.
"""

import os
import sys
import csv
import time
import shutil
import argparse
import subprocess
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                     # .../pyBBarolo_pure
BBDIR = os.path.dirname(ROOT)                    # .../BBarolo
sys.path.insert(0, ROOT)                         # always use the pure-Python bbarolo

from astropy.io import fits                      # noqa: E402
from bbarolo import GalMod                       # noqa: E402
from bbarolo.header import Header                # noqa: E402
from bbarolo.bbutils import AlltoVel             # noqa: E402
from bbarolo.utils import emptyfits              # noqa: E402

COLS = ("RADII", "VROT", "VDISP", "INC", "PA", "Z0", "DENS", "XPOS", "YPOS", "VSYS", "VRAD")


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="C++ vs Python BBarolo on N random GALMOD models",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("-n", "--nmodels", type=int, default=10, help="number of random models")
    ap.add_argument("--cpp", default=os.environ.get("BBAROLO_CPP", os.path.join(BBDIR, "BBarolo")),
                    help="C++ BBarolo executable")
    ap.add_argument("--template", default=os.path.join(BBDIR, "examples", "ngc2403.fits"),
                    help="template datacube (only its header is used), or 'synthetic'")
    ap.add_argument("--outdir", default="galmod_comparison", help="output folder")
    ap.add_argument("--threads", type=int, default=1, help="THREADS for both codes")
    ap.add_argument("--seed", type=int, default=2024, help="seed of the random galaxy generator")
    ap.add_argument("--ltype", type=int, default=2, help="vertical layer type (LTYPE)")
    ap.add_argument("--cdens", type=int, default=10, help="CDENS for both codes")
    ap.add_argument("--nv", type=int, default=-1, help="NV for both codes (-1 = number of channels)")
    ap.add_argument("--no-smooth", action="store_true", help="do not convolve the models with the beam")
    ap.add_argument("--no-baseline", action="store_true",
                    help="skip the second C++ run that measures the Monte-Carlo noise")
    ap.add_argument("--no-api", action="store_true", help="skip the models made with the Python interface")
    ap.add_argument("--galmodtype", default="MC", choices=("MC", "ANALYTIC"),
                    help="GALMODTYPE for the Python models (the C++ code is always MC)")
    ap.add_argument("--ringmode", default="SHIFTED", choices=("SHIFTED", "CENTRED", "NODES"),
                    help="RINGMODE for the Python models (SHIFTED: the rings of the C++ code)")
    ap.add_argument("--cleanup", action="store_true", help="delete the model cubes at the end")
    return ap.parse_args()


def check_cpp(exe):
    if not os.path.isfile(exe):
        sys.exit(f"C++ BBarolo executable not found: {exe}  (use --cpp or $BBAROLO_CPP)")
    with open(exe, "rb") as f:
        if f.read(2) == b"#!":
            sys.exit(f"{exe} is a script, not the compiled C++ BBarolo (use --cpp)")
    return os.path.abspath(exe)


def make_template(path, outdir):
    if path != "synthetic":
        if not os.path.isfile(path):
            sys.exit(f"Template cube not found: {path}  (use --template, or --template synthetic)")
        return os.path.abspath(path)
    fn = os.path.join(outdir, "synthetic_template.fits")
    hdu = emptyfits([128, 128, 64], [-10. / 3600., 10. / 3600., 5.], crvals=[150., 30., 1000.],
                    beam=[30. / 3600., 30. / 3600., 0.], obj="SYNTH", bunit="JY/BEAM")
    hdu.writeto(fn, overwrite=True)
    return os.path.abspath(fn)


def random_galaxy(rng, h, vel):
    """Draw a random disk that fits in the template cube. Returns a dict of ring arrays."""
    nx, ny = h.DimAx(0), h.DimAx(1)
    pix = h.PixScale() * 3600.                                    # arcsec / pixel
    beam = max(h.Bmaj() * 3600., 2 * pix) if h.Bmaj() > 0 else 3 * pix
    vlo, vhi = np.min(vel), np.max(vel)
    halfband = 0.5 * (vhi - vlo)
    inc = rng.uniform(20., 80.)
    pa = rng.uniform(0., 360.)
    rmax = rng.uniform(0.25, 0.42) * min(nx, ny) * pix
    radsep = max(beam / 2., 2 * pix)
    nr = max(int(rmax / radsep), 4)
    radii = (np.arange(nr) + 0.5) * radsep
    vsys = 0.5 * (vlo + vhi) + rng.uniform(-0.1, 0.1) * halfband
    vdisp0, dslope = rng.uniform(6., 25.), rng.uniform(-6., 6.)
    vdisp = np.clip(vdisp0 + dslope * radii / radii[-1], 3., None)
    # flat rotation speed, limited so that the line stays inside the band
    vmax = (0.9 * halfband - abs(vsys - 0.5 * (vlo + vhi)) - 2 * vdisp.max()) / np.sin(np.radians(inc))
    vflat = rng.uniform(0.5, 1.0) * max(min(vmax, 300.), 50.)
    rt = rng.uniform(0.05, 0.4) * radii[-1]
    vrot = vflat * 2. / np.pi * np.arctan(radii / rt)
    vrad = np.full(nr, rng.uniform(-15., 15.) if rng.random() < 0.5 else 0.)
    z0 = np.full(nr, rng.uniform(0., 0.5) * beam)
    dens = np.exp(-radii / (rng.uniform(0.25, 0.7) * radii[-1]))
    if rng.random() < 0.3:                                        # sometimes a central hole
        dens *= 1. - np.exp(-(radii / (0.2 * radii[-1])) ** 2)
    xpos = (nx - 1) / 2. + rng.uniform(-0.05, 0.05) * nx
    ypos = (ny - 1) / 2. + rng.uniform(-0.05, 0.05) * ny
    return dict(RADII=radii, VROT=vrot, VDISP=vdisp, INC=np.full(nr, inc), PA=np.full(nr, pa), Z0=z0,
                DENS=dens, XPOS=np.full(nr, xpos), YPOS=np.full(nr, ypos), VSYS=np.full(nr, vsys), VRAD=vrad)


def write_inputs(gal, folder, template, iseed, args, outfolder):
    os.makedirs(folder, exist_ok=True)
    ringfile = os.path.join(folder, "rings.txt")
    np.savetxt(ringfile, np.column_stack([gal[c] for c in COLS]), fmt="%.6f",
               header="  ".join(COLS))
    lines = [f"FITSFILE   {template}", "GALMOD     true"]
    lines += [f"{c:<10s} file({ringfile},{i + 1})" for i, c in enumerate(COLS)]
    lines += [f"LTYPE      {args.ltype}", f"CDENS      {args.cdens}", f"NV         {args.nv}",
              f"ISEED      {iseed}", "NORM       NONE", "MASK       NONE",
              f"SM         {'false' if args.no_smooth else 'true'}", "PLOTS      false",
              "SHOWBAR    false", f"THREADS    {args.threads}", f"OUTFOLDER  {outfolder}/"]
    parfile = os.path.join(folder, "model.par")
    with open(parfile, "w") as f:
        f.write("\n".join(lines) + "\n")
    return parfile


# --------------------------------------------------------------------------
# Running the codes
# --------------------------------------------------------------------------
def run(cmd, logfile):
    t0 = time.perf_counter()
    with open(logfile, "w") as log:
        rc = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT)
    return rc, time.perf_counter() - t0


def find_model(folder):
    for f in sorted(os.listdir(folder)) if os.path.isdir(folder) else []:
        if f.endswith("mod_none.fits"):
            return os.path.join(folder, f)
    return None


def run_api(template, gal, iseed, args):
    """Same model as the GALMOD task, made with the Python interface (bbarolo.GalMod): the
    rings are given as arrays (one value per ring) instead of the ring file."""
    gm = GalMod(template, radii=gal["RADII"], xpos=gal["XPOS"], ypos=gal["YPOS"], vsys=gal["VSYS"], z0=gal["Z0"],
                inc=gal["INC"], pa=gal["PA"], vrot=gal["VROT"], vdisp=gal["VDISP"], dens=gal["DENS"],
                vrad=gal["VRAD"], ltype=args.ltype, cdens=args.cdens, nv=args.nv, iseed=iseed, norm="NONE",
                mask="NONE", sm=not args.no_smooth, galmodtype=args.galmodtype, ringmode=args.ringmode,
                threads=args.threads, plots=False, showbar=False, verbose=False)
    t0 = time.perf_counter()
    out = gm.run().model
    return np.array(out.data, dtype=np.float64), time.perf_counter() - t0


# --------------------------------------------------------------------------
# Comparison
# --------------------------------------------------------------------------
def moments(cube, vel, good):
    I = np.clip(np.nan_to_num(cube), 0, None)
    m0 = I.sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        m1 = (I * vel[:, None, None]).sum(0) / m0
        m2 = np.sqrt((I * (vel[:, None, None] - m1) ** 2).sum(0) / m0)
    return np.where(good, m1, np.nan), np.where(good, m2, np.nan)


def compare(a, b, vel):
    """Agreement metrics of cube b with respect to the reference cube a."""
    a, b = np.nan_to_num(a.astype(np.float64)), np.nan_to_num(b.astype(np.float64))
    sa, sb = a.sum(axis=(1, 2)), b.sum(axis=(1, 2))
    ma, mb = a.sum(0), b.sum(0)
    good = (ma > 0.05 * ma.max()) & (mb > 0.05 * mb.max())
    v1a, v2a = moments(a, vel, good)
    v1b, v2b = moments(b, vel, good)
    foot = (ma > 0.01 * ma.max()) | (mb > 0.01 * mb.max())
    ra, rb = a[:, foot], b[:, foot]
    return dict(flux_ratio=b.sum() / a.sum(),
                spec_corr=float(np.corrcoef(sa, sb)[0, 1]),
                spec_maxdiff=float(np.max(np.abs(sa - sb)) / np.max(np.abs(sa))),
                dvel=float(np.nanmedian(np.abs(v1a - v1b))),
                ddisp=float(np.nanmedian(np.abs(v2a - v2b))),
                resid=float(np.sqrt(np.sum((ra - rb) ** 2) / np.sum(ra ** 2))))


# --------------------------------------------------------------------------
def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    outdir = os.path.abspath(args.outdir)
    cpp = check_cpp(args.cpp)
    template = make_template(args.template, outdir)
    pyexe = [sys.executable, os.path.join(ROOT, "run_bbarolo.py")]
    h = Header(template, warning=False)
    vel = np.asarray(AlltoVel(h.getZphys(np.arange(h.DimAx(2))), h), dtype=np.float64)
    rng = np.random.default_rng(args.seed)

    print(f"\n C++ BBarolo     : {cpp}\n Python BBarolo  : {' '.join(pyexe)}\n Template cube   : {template}"
          f"\n Models          : {args.nmodels} random disks, THREADS={args.threads}, LTYPE={args.ltype}, "
          f"CDENS={args.cdens}, NV={args.nv}, smoothing={'off' if args.no_smooth else 'on'}"
          f"\n Output folder   : {outdir}\n")

    if not args.no_api:
        # warm-up: compile/load the numba kernels outside the timings
        run_api(template, random_galaxy(np.random.default_rng(0), h, vel), -1, args)

    hdr = (f"{'#':>3s} {'nr':>3s} {'Vflat':>6s} {'inc':>5s} {'PA':>6s} | {'t_C++':>6s} {'t_Py':>6s} {'t_API':>6s} | "
           f"{'flux':>6s} {'spec_r':>7s} {'dV1':>5s} {'dV2':>5s} {'resid':>6s} {'resid0':>6s} | ok")
    print(hdr + "\n" + "-" * len(hdr))
    rows = []
    for k in range(args.nmodels):
        gal = random_galaxy(rng, h, vel)
        iseed = -(k + 1)
        mdir = os.path.join(outdir, f"model_{k:03d}")
        out_c, out_p, out_c2 = (os.path.join(mdir, s) for s in ("cpp", "py", "cpp_seed2"))
        par = write_inputs(gal, mdir, template, iseed, args, out_c)
        rc_c, t_c = run([cpp, "-p", par], os.path.join(mdir, "cpp.log"))
        rc_p, t_p = run(pyexe + ["-p", par, f"OUTFOLDER={out_p}/", f"GALMODTYPE={args.galmodtype}",
                                  f"RINGMODE={args.ringmode}"],
                        os.path.join(mdir, "py.log"))
        if not args.no_baseline:
            run([cpp, "-p", par, f"OUTFOLDER={out_c2}/", f"ISEED={iseed - 1000}"], os.path.join(mdir, "cpp_seed2.log"))
        fc, fp, fc2 = find_model(out_c), find_model(out_p), find_model(out_c2)
        row = dict(model=k, nr=len(gal["RADII"]), vflat=float(gal["VROT"].max()), inc=float(gal["INC"][0]),
                   pa=float(gal["PA"][0]), z0=float(gal["Z0"][0]), vdisp=float(gal["VDISP"].mean()),
                   t_cpp=t_c, t_py=t_p, rc_cpp=rc_c, rc_py=rc_p)
        if fc is None or fp is None:
            print(f"{k:3d}  FAILED (C++ rc={rc_c}, Python rc={rc_p}) -- see logs in {mdir}")
            rows.append(row)
            continue
        A, B = fits.getdata(fc), fits.getdata(fp)
        m = compare(A, B, vel)
        row.update(m)
        if fc2 is not None:
            row.update({k2 + "_base": v for k2, v in compare(A, fits.getdata(fc2), vel).items()})
        if not args.no_api:
            api, t_api = run_api(template, gal, iseed, args)
            row["t_api"] = t_api
            row["api_vs_cli"] = float(np.max(np.abs(api - B)) / np.max(np.abs(B)))
        base = row.get("resid_base", np.nan)
        row["ok"] = bool(abs(m["flux_ratio"] - 1) < 0.03 and m["spec_corr"] > 0.995 and
                         (np.isnan(base) or m["resid"] < 1.5 * base + 0.02))
        rows.append(row)
        print(f"{k:3d} {row['nr']:3d} {row['vflat']:6.1f} {row['inc']:5.1f} {row['pa']:6.1f} | {t_c:6.2f} {t_p:6.2f} "
              f"{row.get('t_api', np.nan):6.2f} | {m['flux_ratio']:6.3f} {m['spec_corr']:7.4f} {m['dvel']:5.2f} "
              f"{m['ddisp']:5.2f} {m['resid']:6.3f} {base:6.3f} | {'yes' if row['ok'] else 'NO'}")
        if args.cleanup:
            for d in (out_c, out_p, out_c2):
                shutil.rmtree(d, ignore_errors=True)

    summarize(rows, outdir, args)


def summarize(rows, outdir, args):
    keys = sorted({k for r in rows for k in r})
    with open(os.path.join(outdir, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    done = [r for r in rows if "resid" in r]
    print("\n Columns: times in s (C++ executable, Python executable, Python interface); flux = total"
          "\n flux Py/C++; spec_r = correlation of the global spectra; dV1, dV2 = median |difference| of"
          "\n velocity and dispersion fields (km/s); resid = |Py - C++| / |C++| over the disk; resid0 = the"
          "\n same for two C++ runs with different seeds (Monte-Carlo noise of GALMOD itself).\n")
    if not done:
        print(" No model could be compared.")
        return
    g = lambda k: np.array([r.get(k, np.nan) for r in done], dtype=float)  # noqa: E731
    tc, tp, ta = g("t_cpp").sum(), g("t_py").sum(), np.nansum(g("t_api"))
    print(f" Total time  C++: {tc:.1f} s   Python: {tp:.1f} s  (x{tc / tp:.2f})"
          + (f"   Python interface: {ta:.1f} s (x{tc / ta:.2f})" if np.isfinite(g('t_api')).any() else ""))
    print(f" Flux ratio Py/C++      : median {np.median(g('flux_ratio')):.4f}  range [{g('flux_ratio').min():.4f}, "
          f"{g('flux_ratio').max():.4f}]" + (f"   (C++/C++: median {np.nanmedian(g('flux_ratio_base')):.4f})"
                                              if np.isfinite(g('flux_ratio_base')).any() else ""))
    print(f" Median |dV| velocity   : {np.median(g('dvel')):.2f} km/s" +
          (f"   (C++/C++: {np.nanmedian(g('dvel_base')):.2f})" if np.isfinite(g('dvel_base')).any() else ""))
    print(f" Median |dV| dispersion : {np.median(g('ddisp')):.2f} km/s" +
          (f"   (C++/C++: {np.nanmedian(g('ddisp_base')):.2f})" if np.isfinite(g('ddisp_base')).any() else ""))
    print(f" Voxel residual         : median {np.median(g('resid')):.3f}" +
          (f"   (C++/C++: {np.nanmedian(g('resid_base')):.3f})" if np.isfinite(g('resid_base')).any() else ""))
    if np.isfinite(g("api_vs_cli")).any():
        print(f" Python interface vs Python executable, max |diff|/peak: {np.nanmax(g('api_vs_cli')):.2e}")
    nok = sum(r.get("ok", False) for r in done)
    print(f"\n {nok}/{len(done)} models consistent with the C++ code within the Monte-Carlo noise"
          f" ({len(rows) - len(done)} failed runs).\n Results: {os.path.join(outdir, 'results.csv')}")
    try:
        plot(done, outdir, g)
    except Exception as e:                                   # plotting is optional
        print(f" (no plot: {e})")


def plot(done, outdir, g):
    import matplotlib
    import matplotlib.ticker
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    k = np.arange(len(done))
    fig, ax = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
    w = 0.27
    ax[0, 0].bar(k - w, g("t_cpp"), w, label="C++ executable", color="#4C72B0")
    ax[0, 0].bar(k, g("t_py"), w, label="Python executable", color="#DD8452")
    if np.isfinite(g("t_api")).any():
        ax[0, 0].bar(k + w, g("t_api"), w, label="Python interface", color="#55A868")
    ax[0, 0].set(xlabel="model", ylabel="time (s)", title="Run time per model")
    ax[0, 0].legend(fontsize=8)
    ax[0, 1].axhline(1, c="k", lw=0.8)
    ax[0, 1].plot(k, g("flux_ratio"), "o", c="#DD8452", label="Python / C++")
    if np.isfinite(g("flux_ratio_base")).any():
        ax[0, 1].plot(k, g("flux_ratio_base"), "s", mfc="none", c="#4C72B0", label="C++ (seed 2) / C++")
    lo = np.nanmin(np.r_[g("flux_ratio"), g("flux_ratio_base")])
    hi = np.nanmax(np.r_[g("flux_ratio"), g("flux_ratio_base")])
    ax[0, 1].set_ylim(min(lo, 0.995) - 0.001, max(hi, 1.005) + 0.001)
    ax[0, 1].ticklabel_format(axis="y", useOffset=False)
    ax[0, 1].set(xlabel="model", ylabel="total flux ratio", title="Total flux")
    ax[0, 1].legend(fontsize=8)
    ax[1, 0].plot(k, g("resid"), "o", c="#DD8452", label="Python vs C++")
    if np.isfinite(g("resid_base")).any():
        ax[1, 0].plot(k, g("resid_base"), "s", mfc="none", c="#4C72B0", label="C++ vs C++ (other seed)")
    ax[1, 0].set(xlabel="model", ylabel="|diff| / |C++ model|", title="Voxel-by-voxel residual")
    ax[1, 0].legend(fontsize=8)
    ax[1, 1].plot(k, g("dvel"), "o", c="#DD8452", label="velocity, Python vs C++")
    ax[1, 1].plot(k, g("ddisp"), "^", c="#C44E52", label="dispersion, Python vs C++")
    if np.isfinite(g("dvel_base")).any():
        ax[1, 1].plot(k, g("dvel_base"), "s", mfc="none", c="#4C72B0", label="velocity, C++ vs C++")
        ax[1, 1].plot(k, g("ddisp_base"), "v", mfc="none", c="#8172B3", label="dispersion, C++ vs C++")
    ax[1, 1].set(xlabel="model", ylabel="median |difference| (km/s)", title="Moment maps")
    ax[1, 1].legend(fontsize=8)
    for a in ax.flat:
        a.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    fn = os.path.join(outdir, "galmod_comparison.png")
    fig.savefig(fn, dpi=120)
    print(f" Figure : {fn}")


if __name__ == "__main__":
    main()
