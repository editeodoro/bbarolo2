"""The figures of the example pages (docs/examples/): run the parameter files of examples/ and convert the PDF plots
they write into PNG files in docs/figures/examples/ (committed: Read the Docs does not run fits).

    python3 docs/scripts/make_example_figures.py --run              # run all the examples, then the figures
    python3 docs/scripts/make_example_figures.py --run ngc2403      # only some (names: --list)
    python3 docs/scripts/make_example_figures.py                    # only the figures, from the outputs of a run

The examples run with the command line of this repository (bin/BBarolo -p <file> THREADS=<--threads>, and the
overrides of the example: the variants of Circinus) in the folder --work (default: examples_runs/ next to the repository), where the FITS files and the parameter files of
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

# name: (parameter file, output folder in <work>, prefix of the files, [(PDF pattern, PNG name, width in pixels)],
#        [KEY=value overrides of the command line])
FIT3D = [("_parameters.pdf", "parameters", 1000), ("_maps_*.pdf", "maps", 650), ("_pv_*.pdf", "pv", 900),
         ("_pvslices_*.pdf", "pvslices", 1300), ("_chanmaps_*.pdf", "chanmaps", 800)]
CORNER = [("_corner.pdf", "corner", 1100)]
WIND = [("_maps_wind.pdf", "maps", 650), ("_pv_wind.pdf", "pv", 800), ("_wind_shells.pdf", "shells", 1000),
        ("_chanmaps_wind.pdf", "chanmaps", 800), ("_pvslices_wind.pdf", "pvslices", 1300)]
CIRC = "cones=approaching likemask=dilated dilation=0,0"          # (the variants of the fit of Circinus)
EX = {
    "ngc2403": ("ngc2403_3DFIT.par", "output/ngc2403", "ngc2403", FIT3D),
    "ngc5633": ("NGC5633_3DFIT.par", "output/NGC5633", "NGC5633", FIT3D),
    "pgc53639": ("pgc53639_3DFIT.par", "output/PGC53639", "PGC53639", FIT3D),
    "ugc1281": ("ugc1281_3DFIT_GLOBAL.par", "output/UGC1281", "UGC1281", FIT3D + CORNER),
    "ngc891": ("ngc891_3DFIT_GLOBAL.par", "output/ngc891_1", "ngc891_1", FIT3D + CORNER),
    "u3_25160": ("U3_25160_3DFIT_BAY.par", "output/U3_25160", "U3_25160", FIT3D + CORNER),
    "hd142527": ("HD142527_3DFIT_BAY.par", "output/hd142527", "hd142527", FIT3D + CORNER),
    "aless73": ("ALESS73_3DFIT_BAY.par", "output/LESS_J033229.4-275619", "LESS_J033229.4-275619", FIT3D + CORNER),
    "2masxj1022": ("2masxj1022_SLITFIT.par", "output/2masxj1022_slit", "2MASXJ1022",
                   [("_slitfit_parameters.pdf", "parameters", 800), ("_slitfit_pv.pdf", "pv", 1300)]),
    "circinus": ("circinus_WINDFIT_GLOBAL.par", "output/circinus", "circinus",
                 [w for w in WIND if w[1] != "shells"] + [("_wind_corner.pdf", "corner", 700)]),
    "circinus_voxel": ("circinus_WINDFIT_GLOBAL.par", "output/circinus_voxel", "circinus", [],
                       ["OUTFOLDER=output/circinus_voxel/", "FLAGERRORS=false", f"LIKEOPTS=emission=voxel {CIRC}"]),
    "circinus_shells": ("circinus_WINDFIT_GLOBAL.par", "output/circinus_shells", "circinus", [],
                        ["OUTFOLDER=output/circinus_shells/", "FLAGERRORS=false",
                         f"LIKEOPTS=emission=shells datafrac=0.3 {CIRC}"]),
    "circinus_moka": ("circinus_WINDFIT_GLOBAL.par", "output/circinus_moka", "circinus", [],
                      ["OUTFOLDER=output/circinus_moka/", "FLAGERRORS=false", f"LIKEOPTS=emission=moka {CIRC}"]),
    "circinus_vsys": ("circinus_WINDFIT_GLOBAL.par", "output/circinus_vsys", "circinus", [],
                      ["OUTFOLDER=output/circinus_vsys/", "FLAGERRORS=false", "FREE=VWIND_SINGLE INC VSYS",
                       "PRIORS=INC=(uniform,30,60) VWIND=(uniform,0,1600) VSYS=(uniform,-300,600)"]),
    "circinus_v550": ("circinus_WINDFIT_GLOBAL.par", "output/circinus_v550", "circinus", [],
                      ["OUTFOLDER=output/circinus_v550/", "FLAGERRORS=false", "VWIND=550", "FREE=INC",
                       "PRIORS=INC=(uniform,30,60)"]),
    "circinus_vdisp": ("circinus_WINDFIT_GLOBAL.par", "output/circinus_vdisp", "circinus", [],
                       ["OUTFOLDER=output/circinus_vdisp/", "FLAGERRORS=false", "FREE=VWIND_SINGLE INC VDISP_SINGLE",
                        "PRIORS=INC=(uniform,30,60) VWIND=(uniform,0,1600) VDISP=(uniform,0,300)"]),
    "circinus_func": ("circinus_WINDFIT_GLOBAL.par", "output/circinus_func", "circinus",
                      [("_wind_shells.pdf", "shells", 1000), ("_maps_wind.pdf", "maps", 650),
                       ("_pv_wind.pdf", "pv", 800)],
                      ["OUTFOLDER=output/circinus_func/", "FLAGERRORS=false", "FREE=VWIND_FUNC INC",
                       "PRIORS=INC=(uniform,30,60) VWIND_P1=(uniform,0,1600) VWIND_P2=(uniform,1,50)"]),
    # (the Bayesian fit, priors narrowed around the global fit: run with --threads 64, 53 minutes)
    "circinus_bayes": ("circinus_WINDFIT_GLOBAL.par", "output/circinus_bayes", "circinus",
                       [("_wind_corner.pdf", "corner", 700)],
                       ["OUTFOLDER=output/circinus_bayes/", "METHOD=BAYESIAN", "SAMPLEROPTS=seed=1",
                        "PRIORS=INC=(uniform,60,25) VWIND=(uniform,200,300) EMSCALE=(loguniform,30,500) "
                        "EMCLUMP=(loguniform,0.5,10) EMSLOPE=(uniform,1,5)"]),
}
# figures made from the outputs of several examples: name -> (function, the examples it needs)
COMPOSITE = {"circinus_compare": ("circinus_compare", ["circinus", "circinus_voxel", "circinus_shells",
                                                       "circinus_moka"]),
             "circinus_mask": ("circinus_mask", []),
             "circinus_spectra": ("circinus_spectra", ["circinus", "circinus_v550", "circinus_vdisp"])}


def run(name, work, threads):
    par = EX[name][0]
    extra = EX[name][4] if len(EX[name]) > 4 else []
    t0 = time.time()
    with open(os.path.join(work, f"{name}.log"), "w") as log:
        ok = subprocess.run([sys.executable, os.path.join(REPO, "bin", "BBarolo"), "-p", par,
                             f"THREADS={threads}", "SHOWBAR=false"] + extra, cwd=work, stdout=log,
                            stderr=subprocess.STDOUT).returncode == 0
    dt = time.time() - t0
    with open(os.path.join(work, "times.txt"), "a") as f:
        f.write(f"{name:12s} {dt:8.1f} s  THREADS={threads}  {'ok' if ok else 'FAILED'}  "
                f"{time.strftime('%Y-%m-%d %H:%M')}\n")
    print(f"  {name}: {dt:.1f} s{'' if ok else ' FAILED (see ' + name + '.log)'}")
    return ok


def figures(name, work):
    if name in COMPOSITE:
        globals()[COMPOSITE[name][0]](work)
        return
    _, folder, prefix, plots = EX[name][:4]
    for pattern, png, width in plots:
        pdfs = sorted(glob.glob(os.path.join(work, folder, prefix + pattern)))
        if not pdfs:
            print(f"  {name}: no {prefix + pattern} in {os.path.join(work, folder)}")
            continue
        dest = os.path.join(OUT, f"{name}_{png}")
        subprocess.run(["pdftoppm", "-png", "-singlefile", "-f", "1", "-scale-to-x", str(width), "-scale-to-y", "-1",
                        pdfs[0], dest], check=True)
        _quantize(dest + ".png")
        print(f"    {os.path.relpath(dest, REPO)}.png")


def _quantize(png):
    im = Image.open(png).convert("RGB")                  # 256 colours: 40% of the size, the same to the eye
    im.quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE).save(png, optimize=True)


def _wind_params(path):
    """The best values of the free parameters in a <prefix>_wind_params.txt: {name: value}."""
    out = {}
    with open(path) as f:
        for line in f:
            w = line.split()
            if w and not line.startswith("#") and len(w) > 1:
                out[w[0]] = float(w[1])
    return out


def circinus_mask(work):
    """The integrated [OIII] map of Circinus (-900 to +800 km/s) with the mask of the fit and the axis of the cone."""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import PowerNorm
    from astropy.io import fits
    from astropy.wcs import WCS
    hdu = fits.open(os.path.join(EXAMPLES, "circinus_OIII.fits"))[0]
    h = hdu.header
    mask = fits.getdata(os.path.join(EXAMPLES, "circinus_OIII_mask.fits")).astype(bool)
    lam = h["CRVAL3"] + (np.arange(hdu.data.shape[0]) + 1 - h["CRPIX3"]) * h["CD3_3"]
    vel = (lam / (5006.843 * 1.001448) - 1) * 299792.458
    mom0 = np.nansum(hdu.data[(vel >= -900) & (vel <= 800)], axis=0) * abs(h["CD3_3"])
    pix = abs(h["CD2_2"]) * 3600.
    x0, y0 = WCS(h).celestial.world_to_pixel_values(213.291251, -65.338826)
    ny, nx = mom0.shape
    ext = [-x0 * pix, (nx - 1 - x0) * pix, -y0 * pix, (ny - 1 - y0) * pix]      # (arcsec from the nucleus; x: west)
    fig, ax = plt.subplots(figsize=(5.4, 6.2))
    for img, title in ((mom0, "[OIII] 5007, -900 to +800 km/s"),):
        ax.imshow(np.clip(np.nan_to_num(img), 0, None), origin="lower", cmap="Greys", extent=ext,
                  norm=PowerNorm(0.4, vmin=0, vmax=np.nanpercentile(img, 99.8)))
        ax.contour((np.arange(nx) - x0) * pix, (np.arange(ny) - y0) * pix, mask.any(axis=0), [0.5],
                   colors="#d62728", linewidths=1.2)
        a = np.radians(-65.)
        ax.plot([0, 40 * -np.sin(a)], [0, 40 * np.cos(a)], "--", color="#1f4e9c", lw=1)
        ax.plot(0, 0, "+", color="#ff7f0e", ms=14, mew=2)
        ax.set_xlim(ext[0], ext[1])
        ax.set_ylim(ext[2], ext[3])
        ax.set_xlabel("x offset (arcsec, west to the right)")
        ax.set_title(title, fontsize=11)
    ax.set_ylabel("y offset (arcsec, north up)")
    fig.tight_layout()
    dest = os.path.join(OUT, "circinus_mask.png")
    fig.savefig(dest, dpi=90)
    plt.close(fig)
    _quantize(dest)
    print(f"    {os.path.relpath(dest, REPO)}")


def circinus_spectra(work):
    """The integrated spectra of Circinus in rings around the nucleus (in the mask, log scale): the data and the
    models of the fit with one VWIND, with VWIND fixed at 550 km/s and with VDISP free."""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from astropy.io import fits
    from astropy.wcs import WCS
    hdu = fits.open(os.path.join(EXAMPLES, "circinus_OIII.fits"))[0]
    h = hdu.header
    mask = fits.getdata(os.path.join(EXAMPLES, "circinus_OIII_mask.fits")).astype(bool)
    lam = h["CRVAL3"] + (np.arange(hdu.data.shape[0]) + 1 - h["CRPIX3"]) * h["CD3_3"]
    vel = (lam / (5006.843 * 1.001448) - 1) * 299792.458
    chan = mask.any(axis=(1, 2))
    x0, y0 = WCS(h).celestial.world_to_pixel_values(213.291251, -65.338826)
    ny, nx = hdu.data.shape[1:]
    yy, xx = np.mgrid[0:ny, 0:nx]
    rad = np.hypot(xx - x0, yy - y0) * abs(h["CD2_2"]) * 3600.
    cubes = [(np.where(mask, np.nan_to_num(hdu.data.astype(np.float64)), 0.), "data", "k", 1.8)]
    for name, col in (("circinus", "#d62728"), ("circinus_v550", "#2ca02c"), ("circinus_vdisp", "#9467bd")):
        folder, prefix = EX[name][1], EX[name][2]
        modf = os.path.join(work, folder, prefix + "_mod_wind.fits")
        if not os.path.exists(modf):
            print(f"  circinus_spectra: no {modf}")
            return
        par = _wind_params(os.path.join(work, folder, prefix + "_wind_params.txt"))
        lab = (f"VWIND {par.get('vwind', 550.):.0f}" + (f", VDISP {par['vdisp']:.0f}" if "vdisp" in par else "")
               + f", INC {par['inc']:.0f}")
        cubes.append((np.where(mask, np.nan_to_num(fits.getdata(modf).astype(np.float64)), 0.), lab, col, 1.1))
    fig, axs = plt.subplots(1, 4, figsize=(16, 4.2), sharex=True)
    for ax, (r0, r1) in zip(axs, ((0, 3), (6, 10), (15, 20), (20, 26))):
        sel = (rad >= r0) & (rad < r1)
        for cube, lab, col, lw in cubes:
            sp = cube[:, sel].sum(axis=1)
            ax.step(vel[chan], np.clip(sp[chan], 1e-30, None), where="mid", color=col, lw=lw, label=lab)
        peak = cubes[0][0][:, sel].sum(axis=1).max()
        ax.set_yscale("log")
        ax.set_ylim(3e-3 * peak, 2 * peak)
        ax.axvline(0, color="0.6", lw=0.5)
        ax.set_title(f"{r0}-{r1} arcsec from the nucleus", fontsize=11)
        ax.set_xlabel("V$_{LOS}$ - V$_{sys}$ (km/s)")
    axs[0].set_ylabel("flux in the mask")
    axs[1].legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    dest = os.path.join(OUT, "circinus_spectra.png")
    fig.savefig(dest, dpi=80)
    plt.close(fig)
    _quantize(dest)
    print(f"    {os.path.relpath(dest, REPO)}")


def circinus_compare(work):
    """The fits of Circinus with the four descriptions of the emission (cells, voxel, shells, moka): position-velocity
    slices of the data in the mask (grey, blue contours) and of the models (red) along the axis of the cone and
    across it at 10 arcsec from the nucleus, 1 beam wide. The model cubes of the fits (_mod_wind.fits) in red
    contours at the levels of the data; those of voxel and moka are the data where their cone reaches (as MoKA-3D),
    so their cone is drawn too: the outline (dashed) of a uniform cone with the same geometry and wind (GALWIND), at
    1/1000 of its peak in the cube, the reach of the cone in the model of voxel."""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from astropy.io import fits
    from astropy.wcs import WCS
    from scipy.ndimage import map_coordinates
    sys.path.insert(0, REPO)
    from bbarolo import GalWind
    hdu = fits.open(os.path.join(EXAMPLES, "circinus_OIII.fits"))[0]
    h = hdu.header
    mask = fits.getdata(os.path.join(EXAMPLES, "circinus_OIII_mask.fits")).astype(bool)
    data = np.where(mask, np.nan_to_num(hdu.data.astype(np.float64)), np.nan)
    lam = h["CRVAL3"] + (np.arange(data.shape[0]) + 1 - h["CRPIX3"]) * h["CD3_3"]
    vel = (lam / (5006.843 * 1.001448) - 1) * 299792.458
    chan = mask.any(axis=(1, 2))
    pix = abs(h["CD2_2"]) * 3600.
    x0, y0 = WCS(h).celestial.world_to_pixel_values(213.291251, -65.338826)
    axis = np.radians(-65.)                                   # (position angle of the axis, east of north)
    ux, uy = -np.sin(axis), np.cos(axis)                      # (pixel x grows to the west)
    wx, wy = -uy, ux                                          # (across the axis)

    def slice_(cube, cx, cy, dx, dy, offs):
        """The slice along the direction (dx, dy) through (cx, cy), averaged over 1 beam across it."""
        out = np.zeros((int(chan.sum()), offs.size))
        n = 0
        for w in np.linspace(-0.45, 0.45, 5) / pix:
            px, py = cx + offs / pix * dx + w * dy, cy + offs / pix * dy - w * dx
            for i, k in enumerate(np.flatnonzero(chan)):
                out[i] += map_coordinates(cube[k], [py, px], order=1, cval=np.nan)
            n += 1
        return out / n

    along = np.arange(0., 30.01, 0.2)
    across = np.arange(-12., 12.01, 0.2)
    cx10, cy10 = x0 + 10. / pix * ux, y0 + 10. / pix * uy
    runs = [("circinus", "cells"), ("circinus_voxel", "voxel"), ("circinus_shells", "shells"),
            ("circinus_moka", "moka")]
    d_al = slice_(data, x0, y0, ux, uy, along)
    d_ac = slice_(data, cx10, cy10, wx, wy, across)
    lev = 3 * 20. * 2. ** np.arange(0, 12)                      # (3 sigma, doubling)
    fig, axs = plt.subplots(2, 4, figsize=(15, 7.2), sharey=True)
    for j, (name, label) in enumerate(runs):
        folder, prefix = EX[name][1], EX[name][2]
        modf = os.path.join(work, folder, prefix + "_mod_wind.fits")
        if not os.path.exists(modf):
            print(f"  circinus_compare: no {modf}")
            plt.close(fig)
            return
        model = np.where(mask, fits.getdata(modf).astype(np.float64), np.nan)
        par = _wind_params(os.path.join(work, folder, prefix + "_wind_params.txt"))
        cone = None
        if label in ("voxel", "moka"):                      # (the uniform cone of the fit)
            g = GalWind(os.path.join(EXAMPLES, "circinus_OIII.fits"), xpos="14:13:09.90", ypos="-65:20:19.77",
                        vsys=par.get("vsys", 0.), inc=par["inc"], pa=25, openang=110, conelen=52,
                        vwind=par["vwind"], vdisp=0, restwave=5006.843, redshift=0.001448, specres=0.85, dens=1,
                        verbose=False)
            cone = np.asarray(g.run(threads=4).model.data, np.float64)
            cone_lev = 1e-3 * np.nanmax(cone)               # (the reach of the cone: 1/1000 of its peak in the cube)
            cone = np.where(mask, cone, np.nan)
        for i, (d, m, c, offs, xl) in enumerate(
                ((d_al, slice_(model, x0, y0, ux, uy, along),
                  None if cone is None else slice_(cone, x0, y0, ux, uy, along), along, "offset along the axis (arcsec)"),
                 (d_ac, slice_(model, cx10, cy10, wx, wy, across),
                  None if cone is None else slice_(cone, cx10, cy10, wx, wy, across), across,
                  "offset across the axis, 10 arcsec out (arcsec)"))):
            ax = axs[i, j]
            ext = [offs[0], offs[-1], vel[chan][0], vel[chan][-1]]
            ax.imshow(d, origin="lower", aspect="auto", extent=ext, cmap="Greys", vmin=0, vmax=np.nanpercentile(d, 99.5))
            ax.contour(offs, vel[chan], np.nan_to_num(d), lev, colors="#1f4e9c", linewidths=0.7)
            if c is None:
                ax.contour(offs, vel[chan], np.nan_to_num(m), lev, colors="#d62728", linewidths=0.8)
            else:
                # (the model is the data where the cone reaches and 0 elsewhere: blanked there, so that its contours
                # stop at the last channel reached instead of running on to the next channel)
                ax.contour(offs, vel[chan], np.where(np.nan_to_num(m) != 0, m, np.nan), lev, colors="#d62728",
                           linewidths=0.8)
                ax.contour(offs, vel[chan], np.nan_to_num(c), [cone_lev], colors="#d62728", linewidths=1.4,
                           linestyles="--")
            ax.axhline(0, color="k", lw=0.5)
            ax.set_xlabel(xl)
            if j == 0:
                ax.set_ylabel("V$_{LOS}$ - V$_{sys}$ (km/s)")
            if i == 0:
                ax.set_title(f"{label}: VWIND {par.get('vwind', np.nan):.0f} km/s, INC {par.get('inc', np.nan):.1f}",
                             fontsize=11)
    fig.tight_layout()
    dest = os.path.join(OUT, "circinus_compare.png")
    fig.savefig(dest, dpi=90)
    plt.close(fig)
    _quantize(dest)
    print(f"    {os.path.relpath(dest, REPO)}")


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
            print(f"{k:16s} examples/{v[0]}" + (" " + " ".join(v[4]) if len(v) > 4 else ""))
        for k, v in COMPOSITE.items():
            print(f"{k:16s} (figure of {', '.join(v[1])})")
        return
    names = a.names or list(EX) + list(COMPOSITE)
    bad = [n for n in names if n not in EX and n not in COMPOSITE]
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
            if n in EX:
                run(n, a.work, a.threads)
    for n in names:
        figures(n, a.work)


if __name__ == "__main__":
    main()
