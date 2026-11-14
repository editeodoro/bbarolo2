#!/usr/bin/env python3
"""
Run all the example parameter files with the current code and with another git revision of
bbarolo2, and report the output files that differ.

    python performance/compare_examples.py               # current code vs HEAD
    python performance/compare_examples.py 941b993 -j 8   # vs a commit, 8 runs at a time
    python performance/compare_examples.py --examples old/examples --new-examples new/examples   # keywords changed

The parameter files are those of <examples>/*.par and <examples>/test/*.par, where <examples> is
../examples (next to this repository) or --examples. They are run from the folder that contains
<examples>, as their FITSFILE paths expect, with PLOTS=false. Known non-deterministic outputs are
ignored: the noisy model (*mod_noise.fits, unseeded noise), the parameters of the run (*_params.par: date
and folder) and the plot scripts (they contain the
output folder). For FITS files the report gives the largest data difference and the header
keywords that changed; for text files the number of lines that differ. With --new-examples, the new code runs the
parameter files of that folder instead (the same names, run from its parent folder), e.g. those of --examples
translated to changed keywords, while the git revision runs those of --examples.
"""

import os
import sys
import glob
import shutil
import argparse
import tempfile
import subprocess
from concurrent.futures import ThreadPoolExecutor

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IGNORE = ("mod_noise.fits", "/plotscripts/", "_params.par")


def run(code, parfile, outfolder, cwd, threads):
    os.makedirs(outfolder, exist_ok=True)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", MPLBACKEND="Agg")
    with open(outfolder.rstrip("/") + ".log", "w") as log:
        r = subprocess.run([sys.executable, os.path.join(code, "bin", "BBarolo"), "-p", parfile, "PLOTS=false",
                            f"THREADS={threads}", f"OUTFOLDER={outfolder}"], cwd=cwd, stdout=log, stderr=log, env=env)
    return r.returncode


def files_of(folder):
    out = set()
    for root, _, files in os.walk(folder):
        for f in files:
            out.add(os.path.relpath(os.path.join(root, f), folder))
    return out


def compare_file(a, b):
    """None if equal, otherwise a short description of the difference."""
    with open(a, "rb") as fa, open(b, "rb") as fb:
        ba, bb = fa.read(), fb.read()
    if a.endswith(".gz"):                   # (the gzip header holds a time stamp)
        import gzip
        ba, bb = gzip.decompress(ba), gzip.decompress(bb)
    if ba == bb:
        return None
    if a.endswith((".fits", ".fits.gz")):
        from astropy.io import fits
        with fits.open(a) as ha, fits.open(b) as hb:
            da, db = ha[0].data, hb[0].data
            hdr = sorted(k for k in set(ha[0].header) | set(hb[0].header)
                         if k not in ("HISTORY", "COMMENT", "DATE") and ha[0].header.get(k) != hb[0].header.get(k))
            if da is None or db is None or da.shape != db.shape:
                data = "shapes differ"
            else:
                d = np.abs(da.astype(float) - db.astype(float))
                data = "data equal" if np.array_equal(da, db, equal_nan=True) else \
                    f"max |diff| {np.nanmax(d):.3g} (max |data| {np.nanmax(np.abs(da)):.3g})"
            return data + (f"; header: {', '.join(hdr)}" if hdr else "")
    with open(a, errors="replace") as fa, open(b, errors="replace") as fb:
        la, lb = fa.readlines(), fb.readlines()
    n = sum(1 for x, y in zip(la, lb) if x != y) + abs(len(la) - len(lb))
    return f"{n} of {max(len(la), len(lb))} lines differ"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ref", nargs="?", default="HEAD", help="git revision to compare with (default HEAD)")
    ap.add_argument("--examples", default=os.path.join(os.path.dirname(REPO), "examples"))
    ap.add_argument("--new-examples", default=None, help="the parameter files of the new code (the names of those of "
                    "--examples; default: the same files), e.g. with changed keywords")
    ap.add_argument("--outdir", default=None, help="folder for the outputs (default: a temporary folder, deleted)")
    ap.add_argument("-j", "--jobs", type=int, default=4, help="runs at a time")
    ap.add_argument("-t", "--threads", type=int, default=4, help="THREADS of each run")
    ap.add_argument("-k", "--keep", action="store_true", help="keep the outputs")
    args = ap.parse_args()

    examples = os.path.abspath(args.examples)
    pars = sorted(glob.glob(os.path.join(examples, "*.par")) + glob.glob(os.path.join(examples, "test", "*.par")))
    if not pars:
        sys.exit(f"No parameter files in {examples}")
    cwd = os.path.dirname(examples)
    newex = os.path.abspath(args.new_examples) if args.new_examples else examples
    missing = [p for p in pars if not os.path.isfile(os.path.join(newex, os.path.relpath(p, examples)))]
    if missing:
        sys.exit(f"Not in {newex}: {', '.join(os.path.relpath(p, examples) for p in missing)}")
    outdir = args.outdir or tempfile.mkdtemp(prefix="bbcompare_")
    ref_code = os.path.join(outdir, "ref_code")
    subprocess.run(["git", "-C", REPO, "worktree", "add", "-q", "--detach", ref_code, args.ref], check=True)
    try:
        jobs = []
        for p in pars:
            name = os.path.relpath(p, examples).replace("/", "_")[:-4]
            pn = os.path.join(newex, os.path.relpath(p, examples))
            for tag, code, pf, wd in (("ref", ref_code, p, cwd), ("new", REPO, pn, os.path.dirname(newex))):
                jobs.append((tag, name, code, pf, os.path.join(outdir, tag, name) + "/", wd))
        with ThreadPoolExecutor(args.jobs) as ex:
            codes = list(ex.map(lambda j: run(j[2], j[3], j[4], j[5], args.threads), jobs))
        failed = [f"{j[0]}:{j[1]}" for j, c in zip(jobs, codes) if c != 0]
        ndiff = 0
        for p in pars:
            name = os.path.relpath(p, examples).replace("/", "_")[:-4]
            a, b = os.path.join(outdir, "ref", name), os.path.join(outdir, "new", name)
            fa, fb = files_of(a), files_of(b)
            lines = [f"   only in {args.ref}: {f}" for f in sorted(fa - fb)]
            lines += [f"   only in the new code: {f}" for f in sorted(fb - fa)]
            for f in sorted(fa & fb):
                if any(s in "/" + f for s in IGNORE):
                    continue
                d = compare_file(os.path.join(a, f), os.path.join(b, f))
                if d:
                    lines.append(f"   {f}: {d}")
            ndiff += len(lines)
            print(f"{name}: {'identical' if not lines else str(len(lines)) + ' differences'}")
            for ln in lines:
                print(ln)
        if failed:
            print("FAILED runs (see the .log files):", ", ".join(failed))
        print(f"\n{len(pars)} parameter files, {ndiff} differences" + (f", {len(failed)} failed runs" if failed else ""))
        return 1 if failed else 0
    finally:
        subprocess.run(["git", "-C", REPO, "worktree", "remove", "--force", ref_code])
        if args.outdir is None and not args.keep:
            shutil.rmtree(outdir, ignore_errors=True)
        elif args.keep or args.outdir:
            print(f"Outputs in {outdir}")


if __name__ == "__main__":
    sys.exit(main())
