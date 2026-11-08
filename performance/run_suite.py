#!/usr/bin/env python3
"""
Performance suite of 3DFIT on simulated galaxies.

Galaxies are simulated with GALMOD (bbarolo.GalMod, ANALYTIC models, i.e. without
Monte Carlo noise) on an empty cube, Gaussian noise smoothed to the beam is added, and
the cubes are fitted with 3DFIT (bbarolo.FitMod3D, Monte Carlo models as by default).
Three experiments:

  resolution  Fixed geometry (INC = 60 deg, PA, centre and VSYS given); FREE = VROT VDISP.
              Varied: the spatial resolution (disk radius in beams), the spectral resolution
              (channel width = instrumental FWHM, in units of the gas dispersion) and the
              peak S/N. A full spatial x spectral grid at the reference S/N, plus the S/N
              values along the two axes through the reference configuration. The same N
              galaxies are observed in every configuration.
  geometry    Flat disks with random inclination, position angle and S/N, at the reference
              resolution; FREE = VROT VDISP INC PA, two-stage fit; the initial INC, PA and
              VROT are estimated by BBarolo; centre, VSYS and rings are given. Each cube is
              fitted three times: with the default limits (DELTAINC = 5, DELTAPA = 15 deg
              around the initial values), with wider ones (15 and 30 deg), and (diagnostic)
              with the default limits around the true INC and PA instead of the estimates.
  warp        As geometry, for warped disks: INC and PA change linearly beyond a radius.
  cpp         Cross-check with the C++ BBarolo executable (--cpp), same cubes and parameters:
              the reference resolution configuration (S/N 10, 20, 40) and the flat disks of
              the geometry experiment started from the true INC and PA.
  mask        Diagnostic of the mask and of the normalisation: the cubes of the reference
              resolution configuration (all S/N) and the flat disks (true initial INC and PA)
              fitted again with MASK = SMOOTH&SEARCH, with MASK = NONE, with BWEIGHT = 0
              (model voxels outside the mask not weighted as blanks) and with NORM = LOCAL
              (instead of the default AZIM).

One JSON line per fit is appended to OUTDIR/results.jsonl (an interrupted run continues
where it stopped). analyse.py makes the figures and tables of the report.

Usage (from anywhere):
    python run_suite.py --outdir results                       # all experiments
    python run_suite.py --outdir results --experiments resolution --ngal 10 --procs 8
    python run_suite.py --outdir quick --quick                 # a few fits, to test
"""

import os
import sys
import json
import time
import argparse
import tempfile
import traceback
import warnings
import numpy as np
from multiprocessing import get_context

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))          # the bbarolo package next to this folder

# --------------------------------------------------------------------------
# Set-up of the experiments
# --------------------------------------------------------------------------
RMAX = 240.                     # outer radius of the disks (arcsec)
Z0 = 0.05 * RMAX                # scale height (arcsec, gaussian layer)
PIX_PER_BEAM = 4                # pixels per beam FWHM
SPATIAL = (1.5, 3., 5., 8.)     # disk radius / beam FWHM
SPECTRAL = (0.5, 1., 2., 4.)    # channel width (= instrumental FWHM) / gas dispersion
SNRS = (5., 10., 20., 40.)      # peak S/N of the cube (noise rms per voxel = peak / S/N)
REF = dict(spatial=5., spectral=1., snr=20.)
INC_RES, PA_RES = 60., 30.      # geometry of the resolution experiment
MASK = "SEARCH"                 # 3DFIT mask (the default)
EXPERIMENTS = ("resolution", "geometry", "warp", "cpp", "mask")
# fits of the geometry and warp experiments: default limits around the initial INC and PA
# (estimated by BBarolo), wider limits, and (as a diagnostic) default limits around the true values
FITMODES = {"default": {}, "wide": dict(deltainc=15., deltapa=30.), "truth": {}}
# mask experiment: other masks, blank weight and normalisation (the reference is MASK = SEARCH,
# BWEIGHT = 1, NORM = AZIM: the defaults)
MASKVARIANTS = {"smooth": dict(mask="SMOOTH&SEARCH"), "none": dict(mask="NONE"), "noblank": dict(bweight=0.),
                "local": dict(norm="LOCAL")}


def resolution_cells():
    """(spatial, spectral, snr) of the resolution experiment: the spatial x spectral grid at
    the reference S/N, and the other S/N values along the two axes through the reference."""
    cells = [(sp, sv, REF["snr"]) for sp in SPATIAL for sv in SPECTRAL]
    for snr in SNRS:
        if snr == REF["snr"]:
            continue
        cells += [(sp, REF["spectral"], snr) for sp in SPATIAL]
        cells += [(REF["spatial"], sv, snr) for sv in SPECTRAL if sv != REF["spectral"]]
    return cells


def draw_galaxy(kind, k, seed):
    """Intrinsic properties of galaxy k of an experiment (the same for every configuration)."""
    rng = np.random.default_rng([seed, ("resolution", "geometry", "warp").index(kind), k])
    g = dict(kind=kind, id=k, vflat=rng.uniform(80, 280), rt=rng.uniform(0.05, 0.25),
             sigma=rng.uniform(10, 25), h=rng.uniform(0.25, 0.5),
             inc=INC_RES, pa=PA_RES, dinc=0., dpa=0., rw=1.)
    if kind == "geometry":
        g.update(inc=rng.uniform(15, 85), pa=rng.uniform(0, 360), snr=float(rng.choice(SNRS)))
    elif kind == "warp":
        inc0 = rng.uniform(40, 75)
        dinc = rng.uniform(-20, 20)
        dinc = float(np.clip(dinc, 25 - inc0, 85 - inc0))
        g.update(inc=inc0, pa=rng.uniform(0, 360), rw=rng.uniform(0.3, 0.6), dinc=dinc,
                 dpa=rng.uniform(-40, 40), snr=REF["snr"])
    return g


def profiles(g, R):
    """True ring values at radii R (arcsec): rotation curve (arctan), constant dispersion,
    exponential surface density and (for warps) INC and PA changing linearly beyond rw*RMAX."""
    R = np.asarray(R, float)
    w = np.clip((R - g["rw"] * RMAX) / ((1 - g["rw"]) * RMAX), 0, 1) if g["rw"] < 1 else np.zeros_like(R)
    return dict(vrot=2. / np.pi * g["vflat"] * np.arctan(R / (g["rt"] * RMAX)),
                vdisp=np.full(R.shape, g["sigma"]),
                dens=np.exp(-R / (g["h"] * RMAX)),
                inc=g["inc"] + g["dinc"] * w,
                pa=(g["pa"] + g["dpa"] * w) % 360.)


def geometry_of_cube(g, spatial, spectral):
    """Beam, pixel, channel width and cube size for a galaxy and a resolution."""
    beam = RMAX / spatial
    pix = beam / PIX_PER_BEAM
    dv = spectral * g["sigma"]
    npix = 2 * int(np.ceil((RMAX + 1.5 * beam) / pix)) + 1
    sig_instr = dv / (2. * np.sqrt(2. * np.log(2.)))
    incmax = max(g["inc"], g["inc"] + g["dinc"])
    vext = g["vflat"] * np.sin(np.radians(incmax)) + 4. * np.hypot(g["sigma"], sig_instr) + 2. * dv
    nchan = 2 * int(np.ceil(vext / dv)) + 1
    return dict(beam=beam, pix=pix, dv=dv, npix=npix, nchan=nchan)


def fit_radii(beam):
    """Rings of the fit: width = half a beam, from the centre to RMAX."""
    w = beam / 2.
    n = int(round(RMAX / w))
    return (np.arange(n) + 0.5) * w


# --------------------------------------------------------------------------
# One simulation + fit
# --------------------------------------------------------------------------
def simulate(g, cg, snr, seed, workdir):
    """Simulated cube (FITS file in workdir): ANALYTIC GALMOD model with fine rings, plus
    Gaussian noise smoothed to the beam with rms = peak / snr. Returns (file, centre, rms, peak)."""
    from bbarolo import GalMod
    from bbarolo.utils import emptyfits
    n, nz = cg["npix"], cg["nchan"]
    tmpl = emptyfits([n, n, nz], [-cg["pix"] / 3600., cg["pix"] / 3600., cg["dv"]],
                     beam=cg["beam"] / 3600., bunit="JY/BEAM", obj=f"{g['kind']}{g['id']}")
    xc = (n - 1) / 2.
    dr = max(RMAX / 30., cg["pix"])            # fine rings (GALMOD needs them >= 0.75 pixels apart)
    rs = (np.arange(int(round(RMAX / dr))) + 0.5) * dr
    t = profiles(g, rs)
    gm = GalMod(tmpl, radii=rs, xpos=xc, ypos=xc, vsys=0., vrot=t["vrot"], vdisp=t["vdisp"], vrad=0., z0=Z0,
                inc=t["inc"], pa=t["pa"], dens=t["dens"], ltype=1, norm="NONE", mask="NONE",
                galmodtype="ANALYTIC", threads=1, verbose=False)
    clean = gm.run().model
    peak = float(clean.data.max())
    rms = peak / snr
    noisy = gm.diskmodel.addnoise(clean.data, rms, seed=seed)
    from astropy.io import fits
    fn = os.path.join(workdir, "cube.fits")
    fits.PrimaryHDU(data=noisy, header=tmpl.header).writeto(fn, overwrite=True)
    return fn, xc, rms, peak, ring_snr(clean.data, g, cg, xc, rms)


def ring_snr(clean, g, cg, xc, rms):
    """S/N of each fitted ring: median over the pixels of the ring of the peak of the
    noiseless spectrum, divided by the noise rms (BBarolo's projection of the rings)."""
    radii = fit_radii(cg["beam"])
    w = cg["beam"] / 2.
    pk = clean.max(axis=0)
    yy, xx = np.indices(pk.shape, dtype=float)
    t = profiles(g, radii)
    out = []
    for R, inc, pa in zip(radii, t["inc"], t["pa"]):
        sp, cp, ci = np.sin(np.radians(pa)), np.cos(np.radians(pa)), np.cos(np.radians(inc))
        xr = -(xx - xc) * sp + (yy - xc) * cp
        yr = (-(xx - xc) * cp - (yy - xc) * sp) / ci
        r = np.hypot(xr, yr) * cg["pix"]
        sel = (r >= R - w / 2) & (r <= R + w / 2)
        out.append(float(np.median(pk[sel]) / rms) if sel.any() else 0.)
    return out


def fit_parameters(job, xc):
    """The parameters of the 3DFIT of a job (parameter-file keywords)."""
    g, cg = job["galaxy"], job["cube"]
    w = cg["beam"] / 2.
    kw = dict(nradii=len(fit_radii(cg["beam"])), radsep=w, xpos=xc, ypos=xc, vsys=0., z0=Z0, ltype=1, mask=MASK)
    if g["kind"] == "resolution":
        kw.update(inc=g["inc"], pa=g["pa"], vrot=120., vdisp=15., free="VROT VDISP", twostage=False)
    else:            # INC, PA and VROT estimated by BBarolo, then fitted with VDISP
        kw.update(free="VROT VDISP INC PA", twostage=True, **FITMODES[job["fitmode"]])
        if job["fitmode"] == "truth":      # initial INC and PA = the true values of the rings
            R = fit_radii(cg["beam"])
            w = np.clip((R - g["rw"] * RMAX) / ((1 - g["rw"]) * RMAX), 0, 1) if g["rw"] < 1 else np.zeros_like(R)
            pa = g["pa"] + g["dpa"] * w                  # continuous (not wrapped at 360)
            if pa.max() >= 360:
                pa = pa - 360.
            kw.update(inc=list(g["inc"] + g["dinc"] * w), pa=list(pa))
    kw.update(MASKVARIANTS.get(job.get("variant"), {}))
    return kw


def tables(res):
    out = {}
    for name in ("rings", "rings_stage1", "rings_initial"):
        if name in res:
            t = res[name]
            if "fitok" in t.colnames:                  # (the rings fitted, as in the ring files)
                t = t[np.asarray(t["fitok"], bool)]
            out[name] = {c: [float(v) for v in t[c]] for c in ("radii", "vrot", "vdisp", "inc", "pa") if c in t.colnames}
    return out


def fit(job, fn, xc, workdir):
    """3DFIT of the simulated cube (Python); returns the tables of the result."""
    from bbarolo import FitMod3D
    kw = fit_parameters(job, xc)
    res = FitMod3D(fn, threads=1, verbose=False, plots=-1, outfolder=os.path.join(workdir, "out"), **kw).run()
    return tables(res)


def fit_cpp(job, fn, xc, workdir, exe):
    """3DFIT of the simulated cube with the C++ executable; returns the rings table."""
    import subprocess
    from bbarolo.tasks import read_table
    of = os.path.join(workdir, "outcpp") + "/"
    lines = [f"FITSFILE {fn}", "3DFIT true", f"OUTFOLDER {of}", "THREADS 1", "PLOTS false", "VERBOSE false"]
    for k, v in fit_parameters(job, xc).items():
        if isinstance(v, (list, tuple, np.ndarray)):       # one value per ring: constant here (flat disks)
            if np.ptp(v) > 1e-9:
                raise ValueError("the C++ cross-check needs constant ring values")
            v = float(v[0])
        lines.append(f"{k.upper()} {str(v).lower() if isinstance(v, bool) else v}")
    par = os.path.join(workdir, "cpp.par")
    with open(par, "w") as f:
        f.write("\n".join(lines) + "\n")
    subprocess.run([exe, "-p", par], cwd=workdir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=3600)
    out = {}
    for name, fn in (("rings", "rings_final2.txt"), ("rings_stage1", "rings_final1.txt")):
        if os.path.isfile(of + fn):
            t = read_table(of + fn)
            out[name] = {c: [float(v) for v in t[c]] for c in ("radii", "vrot", "vdisp", "inc", "pa")}
    if "rings" not in out:
        out["rings"] = out.pop("rings_stage1")
    return out


def run_job(job):
    """Simulate and fit one galaxy in one configuration (in a temporary folder)."""
    warnings.simplefilter("ignore")
    rec = dict(job)
    t0 = time.time()
    with tempfile.TemporaryDirectory(prefix="bbperf_") as wd:
        try:
            g, cg = job["galaxy"], job["cube"]
            fn, xc, rms, peak, rsnr = simulate(g, cg, job["snr"], job["seed"], wd)
            rec.update(noise=rms, peak=peak, ring_snr=rsnr, time_sim=time.time() - t0)
            t1 = time.time()
            if job["experiment"] == "cpp":
                rec["fit"] = fit_cpp(job, fn, xc, wd, job["exe"])
            else:
                rec["fit"] = fit(job, fn, xc, wd)
            rec["time_fit"] = time.time() - t1
            radii = fit_radii(cg["beam"])
            rec["true"] = {k: [float(v) for v in vals] for k, vals in profiles(g, radii).items()}
            rec["true"]["radii"] = [float(v) for v in radii]
            rec["status"] = "ok"
        except BaseException as e:            # (SystemExit included) a failed fit is a result too
            rec["status"] = "failed"
            rec["error"] = f"{type(e).__name__}: {e}"[:500]
            rec["traceback"] = traceback.format_exc()[-1500:]
    rec["time"] = time.time() - t0
    return rec


# --------------------------------------------------------------------------
# The jobs
# --------------------------------------------------------------------------
def make_jobs(experiments, ngal, ngeo, nwarp, seed, exe=None):
    jobs = []

    def res_key(sp, sv, snr, k):
        return f"resolution/sp{sp:g}/sv{sv:g}/snr{snr:g}/g{k}"
    if "resolution" in experiments:
        for (sp, sv, snr) in resolution_cells():
            for k in range(ngal):
                g = draw_galaxy("resolution", k, seed)
                jobs.append(dict(key=res_key(sp, sv, snr, k), noisekey=res_key(sp, sv, snr, k),
                                 experiment="resolution", galaxy=g, spatial=sp, spectral=sv, snr=snr,
                                 cube=geometry_of_cube(g, sp, sv)))
    if "cpp" in experiments and exe:          # the same cubes as the resolution experiment
        sp, sv = REF["spatial"], REF["spectral"]
        for snr in SNRS[1:]:
            for k in range(ngal):
                g = draw_galaxy("resolution", k, seed)
                jobs.append(dict(key="cpp/" + res_key(sp, sv, snr, k), noisekey=res_key(sp, sv, snr, k),
                                 experiment="cpp", galaxy=g, spatial=sp, spectral=sv, snr=snr, exe=exe,
                                 cube=geometry_of_cube(g, sp, sv)))
    if "cpp" in experiments and exe:          # flat disks of the geometry experiment, true initial INC, PA
        for k in range(ngeo):
            g = draw_galaxy("geometry", k, seed)
            jobs.append(dict(key=f"cpp/geometry/truth/g{k}", noisekey=f"geometry/g{k}", experiment="cpp",
                             fitmode="truth", galaxy=g, spatial=REF["spatial"], spectral=REF["spectral"],
                             snr=g["snr"], exe=exe, cube=geometry_of_cube(g, REF["spatial"], REF["spectral"])))
    if "mask" in experiments:                # the same cubes, other masks
        sp, sv = REF["spatial"], REF["spectral"]
        for v in MASKVARIANTS:
            for snr in SNRS:
                for k in range(ngal):
                    g = draw_galaxy("resolution", k, seed)
                    jobs.append(dict(key=f"mask/{v}/" + res_key(sp, sv, snr, k), noisekey=res_key(sp, sv, snr, k),
                                     experiment="mask", variant=v, galaxy=g, spatial=sp, spectral=sv, snr=snr,
                                     cube=geometry_of_cube(g, sp, sv)))
            for k in range(ngeo):
                g = draw_galaxy("geometry", k, seed)
                jobs.append(dict(key=f"mask/{v}/geometry/truth/g{k}", noisekey=f"geometry/g{k}", experiment="mask",
                                 variant=v, fitmode="truth", galaxy=g, spatial=sp, spectral=sv, snr=g["snr"],
                                 cube=geometry_of_cube(g, sp, sv)))
    for kind, n in (("geometry", ngeo), ("warp", nwarp)):
        if kind in experiments:
            for k in range(n):
                g = draw_galaxy(kind, k, seed)
                for mode in FITMODES:
                    jobs.append(dict(key=f"{kind}/{mode}/g{k}", noisekey=f"{kind}/g{k}", experiment=kind,
                                     fitmode=mode, galaxy=g, spatial=REF["spatial"], spectral=REF["spectral"],
                                     snr=g["snr"], cube=geometry_of_cube(g, REF["spatial"], REF["spectral"])))
    for j in jobs:      # seed of the noise (the same cube for the jobs that share noisekey)
        j["seed"] = int.from_bytes(j["noisekey"].encode(), "little") % (2 ** 31) ^ seed
    return jobs


def main():
    ap = argparse.ArgumentParser(description="3DFIT performance suite on simulated galaxies",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--outdir", default="results", help="output folder (results.jsonl)")
    ap.add_argument("--experiments", nargs="+", default=list(EXPERIMENTS), choices=EXPERIMENTS)
    ap.add_argument("--ngal", type=int, default=24, help="galaxies per configuration (resolution, cpp)")
    ap.add_argument("--ngeo", type=int, default=200, help="galaxies of the geometry experiment")
    ap.add_argument("--nwarp", type=int, default=120, help="galaxies of the warp experiment")
    ap.add_argument("--cpp", default=os.environ.get("BBAROLO_CPP"),
                    help="C++ BBarolo executable for the cpp cross-check (skipped if not given)")
    ap.add_argument("--procs", type=int, default=2, help="fits run in parallel (one thread each)")
    ap.add_argument("--seed", type=int, default=2026, help="seed of the galaxies and of the noise")
    ap.add_argument("--quick", action="store_true", help="a few fits only, to test the set-up")
    args = ap.parse_args()
    if args.quick:
        args.ngal, args.ngeo, args.nwarp = 1, 2, 2
    os.makedirs(args.outdir, exist_ok=True)
    out = os.path.join(args.outdir, "results.jsonl")
    done = set()
    if os.path.isfile(out):
        with open(out) as f:
            done = {json.loads(line)["key"] for line in f if line.strip()}
    exe = os.path.abspath(args.cpp) if args.cpp and os.path.isfile(args.cpp) else None
    if "cpp" in args.experiments and exe is None:
        print("cpp cross-check skipped: no C++ executable (--cpp or $BBAROLO_CPP)")
    jobs = [j for j in make_jobs(args.experiments, args.ngal, args.ngeo, args.nwarp, args.seed, exe)
            if j["key"] not in done]
    if args.quick:
        jobs = [j for j in jobs if j["experiment"] not in ("resolution", "cpp", "mask")
                or (j["spatial"], j["spectral"], j["snr"]) in ((1.5, 1., 20.), (5., 1., 20.), (5., 4., 5.))]
    print(f"{len(jobs)} fits to run ({len(done)} already done) with {args.procs} processes", flush=True)
    t0 = time.time()
    ctx = get_context("fork")
    with ctx.Pool(args.procs, maxtasksperchild=20) as pool, open(out, "a") as f:
        for i, rec in enumerate(pool.imap_unordered(run_job, jobs), 1):
            f.write(json.dumps(rec) + "\n")
            f.flush()
            el = time.time() - t0
            print(f"[{i}/{len(jobs)}] {rec['key']:40s} {rec['status']:6s} {rec['time']:6.1f} s "
                  f"(elapsed {el / 60:.1f} min, left ~{el / i * (len(jobs) - i) / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
