"""The figures of the example pages (docs/examples/): run the parameter files of examples/ and convert the PDF plots
they write into PNG files in docs/figures/examples/ (committed: Read the Docs does not run fits).

    python3 docs/scripts/make_example_figures.py --run              # run all the examples, then the figures
    python3 docs/scripts/make_example_figures.py --run ngc2403      # only some (names: --list)
    python3 docs/scripts/make_example_figures.py                    # only the figures, from the outputs of a run

The examples run with the command line of this repository (bin/BBarolo -p <file> THREADS=<--threads>) in the
folder --work (default: examples_runs/ next to the repository), where the FITS files and the parameter files of
examples/ are linked, so their outputs go to <work>/output/. The wall-clock time of each run is written in
<work>/times.txt (the times on the example pages are those of a server with 16 cores, --threads 16).
Needs pdftoppm (poppler) to convert the PDF files.
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys
import time

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
EXAMPLES = os.path.join(REPO, "examples")
OUT = os.path.join(REPO, "docs", "figures", "examples")

# name: (parameter file, output folder in <work>, prefix of the files, [(PDF pattern, PNG name, width in pixels)])
FIT3D = [("_parameters.pdf", "parameters", 1000), ("_maps_*.pdf", "maps", 650), ("_pv_*.pdf", "pv", 900),
         ("_pvslices_*.pdf", "pvslices", 1300), ("_chanmaps_*.pdf", "chanmaps", 800)]
CORNER = [("_corner.pdf", "corner", 1100)]
EX = {
    "ngc2403": ("ngc2403_3DFIT.par", "output/ngc2403", "ngc2403", FIT3D),
    "ngc5633": ("NGC5633_3DFIT.par", "output/NGC5633", "NGC5633", FIT3D),
    "pgc53639": ("pgc53639_3DFIT.par", "output/PGC53639", "PGC53639", FIT3D),
    "ugc1281": ("ugc1281_3DFIT_GLOBAL.par", "output/UGC1281", "UGC1281", FIT3D + CORNER),
    "u3_25160": ("U3_25160_3DFIT_BAY.par", "output/U3_25160", "U3_25160", FIT3D + CORNER),
    "hd142527": ("HD142527_3DFIT_BAY.par", "output/hd142527", "hd142527", FIT3D + CORNER),
    "2masxj1022": ("2masxj1022_SLITFIT.par", "output/2masxj1022_slit", "2MASXJ1022",
                   [("_slitfit_parameters.pdf", "parameters", 800), ("_slitfit_pv.pdf", "pv", 1300)]),
}


def run(name, work, threads):
    par = EX[name][0]
    t0 = time.time()
    with open(os.path.join(work, f"{name}.log"), "w") as log:
        ok = subprocess.run([sys.executable, os.path.join(REPO, "bin", "BBarolo"), "-p", par,
                             f"THREADS={threads}", "SHOWBAR=false"], cwd=work, stdout=log,
                            stderr=subprocess.STDOUT).returncode == 0
    dt = time.time() - t0
    with open(os.path.join(work, "times.txt"), "a") as f:
        f.write(f"{name:12s} {dt:8.1f} s  THREADS={threads}  {'ok' if ok else 'FAILED'}  "
                f"{time.strftime('%Y-%m-%d %H:%M')}\n")
    print(f"  {name}: {dt:.1f} s{'' if ok else ' FAILED (see ' + name + '.log)'}")
    return ok


def figures(name, work):
    _, folder, prefix, plots = EX[name]
    for pattern, png, width in plots:
        pdfs = sorted(glob.glob(os.path.join(work, folder, prefix + pattern)))
        if not pdfs:
            print(f"  {name}: no {prefix + pattern} in {os.path.join(work, folder)}")
            continue
        dest = os.path.join(OUT, f"{name}_{png}")
        subprocess.run(["pdftoppm", "-png", "-singlefile", "-f", "1", "-scale-to-x", str(width), "-scale-to-y", "-1",
                        pdfs[0], dest], check=True)
        im = Image.open(dest + ".png").convert("RGB")        # 256 colours: 40% of the size, the same to the eye
        im.quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE).save(dest + ".png",
                                                                                                 optimize=True)
        print(f"    {os.path.relpath(dest, REPO)}.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="*", help="examples (default: all)")
    ap.add_argument("--run", action="store_true", help="run the examples first")
    ap.add_argument("--work", default=os.path.join(os.path.dirname(REPO), "examples_runs"),
                    help="folder of the runs (default: examples_runs/ next to the repository)")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--list", action="store_true", help="list the examples")
    a = ap.parse_args()
    if a.list:
        for k, v in EX.items():
            print(f"{k:12s} examples/{v[0]}")
        return
    names = a.names or list(EX)
    bad = [n for n in names if n not in EX]
    if bad:
        sys.exit(f"Unknown examples: {', '.join(bad)} (see --list)")
    if shutil.which("pdftoppm") is None:
        sys.exit("pdftoppm (poppler) is needed to convert the PDF plots")
    os.makedirs(OUT, exist_ok=True)
    if a.run:
        os.makedirs(a.work, exist_ok=True)
        for f in glob.glob(os.path.join(EXAMPLES, "*.fits")) + glob.glob(os.path.join(EXAMPLES, "*.par")):
            dst = os.path.join(a.work, os.path.basename(f))
            if os.path.lexists(dst):
                os.remove(dst)
            os.symlink(f, dst)
        for n in names:
            run(n, a.work, a.threads)
    for n in names:
        figures(n, a.work)


if __name__ == "__main__":
    main()
