"""The figures of the documentation, made by running BBarolo on the example cube of NGC 2403.

The PNG files go to docs/figures/ and are committed: Read the Docs does not run fits.

    python3 docs/scripts/make_figures.py                 # all the figures
    python3 docs/scripts/make_figures.py 3dfit maps      # only some
    python3 docs/scripts/make_figures.py --list
    python3 docs/scripts/make_figures.py --out /tmp/figs # elsewhere, e.g. to compare with docs/figures/

The cube ngc2403.fits is looked for in --examples, $BBAROLO_EXAMPLES, or the folder examples/ next to
the repository (as the tests do). The code of each figure is the code shown on the task pages. The tasks
run in memory (run() writes nothing), so only the PNG files are written.
$BBAROLO_CODE runs another copy of the package (a folder that contains bbarolo/).
"""
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(REPO, "docs", "figures")
sys.path.insert(0, os.environ.get("BBAROLO_CODE", REPO))      # the code to run: this repository by default

import matplotlib                                    # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
import numpy as np                                   # noqa: E402

FIGURES = {}
CUBE = None                                          # the path of ngc2403.fits, set in main()


def figure(name):
    """Register a function that makes the figures `name`*.png."""
    def deco(fn):
        FIGURES[name] = fn
        return fn
    return deco


def save(fig, fname):
    path = os.path.join(OUT, fname)
    fig.savefig(path, dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"    {os.path.relpath(path, REPO)}")


# The parameters of NGC 2403 used on all the pages
N2403 = dict(xpos=77, ypos=77, vsys=132.8, vrot=120, vdisp=8, inc=60, pa=123.7, z0=10)


@figure("3dfit")
def fig_3dfit():
    from bbarolo import FitMod3D
    f = FitMod3D(CUBE, nradii=41, radsep=30, **N2403, free="VROT VDISP PA", mask="SEARCH", norm="LOCAL")
    res = f.run(threads=4)
    save(res.plot_rings(), "3dfit_rings.png")
    save(res.plot_pv(), "3dfit_pv.png")


# ---------------------------------------------------------------- models: BAYESIAN3DFIT, GALMOD, GalMod3D, GALWIND
def _moment_panels(hdus, titles, cmaps, fname, figsize=None):
    """Moment maps (astropy HDUs, 2D) side by side, in pixels, each with its colour bar."""
    fig, axs = plt.subplots(1, len(hdus), figsize=figsize or (3.4 * len(hdus), 3.6))
    for ax, hdu, title, cmap in zip(np.atleast_1d(axs), hdus, titles, cmaps):
        d = np.asarray(hdu.data, float)
        lo, hi = np.nanpercentile(d[np.isfinite(d) & (d != 0)], [1, 99])
        im = ax.imshow(d, origin="lower", vmin=lo, vmax=hi, cmap=cmap)
        fig.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("x (pixel)")
    np.atleast_1d(axs)[0].set_ylabel("y (pixel)")
    fig.tight_layout()
    save(fig, fname)


@figure("galmod")
def fig_galmod():
    from bbarolo import GalMod
    g = GalMod(CUBE, nradii=41, radsep=30, **N2403, norm="LOCAL", mask="SMOOTH")
    res = g.run(threads=4)
    save(res.plot_pv(), "galmod_pv.png")


@figure("galmod3d")
def fig_galmod3d():
    from bbarolo import GalMod3D, Maps

    def dens(R, phi, z):
        bar = 1 + 0.8 * np.exp(-(R / 150.) ** 2) * np.cos(2 * (phi - 0.5))
        arms = 1 + 0.5 * np.cos(2 * (phi - np.log(np.maximum(R, 1.) / 100.) / np.tan(np.radians(20))))
        return np.exp(-R / 400.) * bar * arms * np.exp(-z * z / 50.)

    def vphi(R, phi, z):
        return 130 * np.tanh(R / 60.) * (1 + 0.1 * np.exp(-(R / 150.) ** 2) * np.cos(2 * (phi - 0.5)))

    def vr(R, phi, z):
        return 15 * np.exp(-(R / 150.) ** 2) * np.sin(2 * (phi - 0.5))

    m = GalMod3D(CUBE, coords="Rphiz", xpos=77, ypos=77, vsys=132.8, inc=30, pa=123.7, dens=dens, vphi=vphi,
                 vr=vr, vdisp=8, zmax=30, flux="measured", threads=4)
    model = m.model()
    maps = Maps(model, mask="THRESHOLD", threshold=0.005, totalmap=True, velocitymap=True).run(threads=4)
    _moment_panels([maps.totalmap, maps.velocitymap], ["Total intensity (Jy/beam km/s)", "Velocity (km/s)"],
                   ["viridis", "RdBu_r"], "galmod3d_maps.png", figsize=(7, 3.6))


@figure("galwind")
def fig_galwind():
    from bbarolo import GalWind
    w = GalWind(CUBE, xpos=77, ypos=77, vsys=132.8, inc=60, pa=123.7, vwind=150, openang=60, htot=600,
                vdisp=30, dens=0.001)
    res = w.run(threads=4)
    _moment_panels([res.maps["0mom"], res.maps["1mom"], res.maps["2mom"]],
                   ["Total intensity (Jy/beam km/s)", "Velocity (km/s)", "Dispersion (km/s)"],
                   ["viridis", "RdBu_r", "magma"], "galwind_maps.png")
    h = res.pv.header
    d = np.asarray(res.pv.data, float)
    off = ((np.arange(h["NAXIS1"]) + 1 - h["CRPIX1"]) * h["CDELT1"] + h["CRVAL1"]) * 3600.
    vel = ((np.arange(h["NAXIS2"]) + 1 - h["CRPIX2"]) * h["CDELT2"] + h["CRVAL2"]) / 1000.
    if vel[-1] < vel[0]:
        vel, d = vel[::-1], d[::-1]
    fig, ax = plt.subplots(figsize=(5.2, 4))
    im = ax.imshow(d, origin="lower", aspect="auto", cmap="viridis", extent=[off[0], off[-1], vel[0], vel[-1]])
    fig.colorbar(im, ax=ax, label="Jy/beam")
    ax.set_xlabel("Offset along the cone axis (arcsec)")
    ax.set_ylabel("Velocity (km/s)")
    save(fig, "galwind_pv.png")


@figure("bayesian3dfit")
def fig_bayesian3dfit():
    from bbarolo import Bayesian3DFIT
    f = Bayesian3DFIT(CUBE, radii=[120, 240, 360, 480], xpos=77, ypos=77, vsys=132.8, vrot=100, vdisp=10,
                      inc=60, pa=123.7, z0=10, mask="SEARCH")
    f.priors["vrot"] = dict(name="uniform", loc=50, scale=200)      # km/s
    f.priors["vdisp"] = dict(name="uniform", loc=0, scale=30)       # km/s
    f.run(freepar="VROT VDISP_SINGLE SIGMA", threads=4, method="dynesty",
          sampler_kwargs=dict(nlive=100, seed=1), run_kwargs=dict(dlogz=1, print_progress=False))
    f.print_stats()
    save(f.corner_plot(), "bayesian3dfit_corner.png")


# (end of models)


# ---------------------------------------------------------------- 2DFIT, SPACEPAR, ELLPROF, PVSLICE
@figure("2dfit")
def fig_2dfit():
    from bbarolo import FitMod2D
    f = FitMod2D(CUBE, nradii=41, radsep=30, xpos=77, ypos=77, vsys=132.8, vrot=120, inc=60, pa=123.7,
                 free="VROT PA INC", mask="SEARCH", side="B", wfunc=2)
    res = f.run(threads=4)
    save(res.plot_rings(["vrot", "pa", "inc"]), "2dfit_rings.png")
    fig, axs = plt.subplots(1, 2, figsize=(8, 3.8), sharey=True)
    vf, vm = np.asarray(res.velfield.data, float), np.asarray(res.model.data, float)
    vm = np.where(np.isfinite(vf), vm, np.nan)
    lo, hi = np.nanpercentile(vf, [1, 99])
    for ax, v, t in zip(axs, (vf, vm), ("Velocity field of the data", "2D model")):
        im = ax.imshow(v, origin="lower", cmap="RdBu_r", vmin=lo, vmax=hi)
        ax.set_title(t, fontsize=10)
        ax.set_xlabel("x (pixel)")
    axs[0].set_ylabel("y (pixel)")
    fig.colorbar(im, ax=axs, shrink=0.85, label="Velocity (km/s)")
    save(fig, "2dfit_velfield.png")


@figure("spacepar")
def fig_spacepar():
    from bbarolo import SpacePar
    f = SpacePar(CUBE, p1="VROT", p2="INC", p1par="60 160 2", p2par="30 90 2", radii="450 750",
                 xpos=77, ypos=77, vsys=132.8, vrot=120, vdisp=8, inc=60, pa=123.7, z0=10, mask="SEARCH")
    res = f.run(threads=4)
    h = res.space.header
    d = 100 * np.asarray(res.space.data, float)
    vrot = h["CRVAL1"] + h["CDELT1"] * np.arange(h["NAXIS1"])
    inc = h["CRVAL2"] + h["CDELT2"] * np.arange(h["NAXIS2"])
    radii = [450, 750]
    fig, axs = plt.subplots(1, 2, figsize=(8, 3.6), sharey=True)
    for ax, m, r in zip(axs, d, radii):
        im = ax.pcolormesh(vrot, inc, np.clip(m, 0, 50), cmap="viridis_r", shading="nearest")
        ax.contour(vrot, inc, m, levels=[2, 5, 10, 30], colors="k", linewidths=0.7)
        j, i = np.unravel_index(np.nanargmin(m), m.shape)
        ax.plot(vrot[i], inc[j], "r+", ms=12, mew=2)
        ax.set_title(f"Ring at {r} arcsec\nminimum at VROT = {vrot[i]:.0f} km/s, INC = {inc[j]:.0f}$^\\circ$",
                     fontsize=9)
        ax.set_xlabel("VROT (km/s)")
    axs[0].set_ylabel("INC (degrees)")
    fig.colorbar(im, ax=axs, shrink=0.85, label="Increase of the residual over the minimum (%)")
    save(fig, "spacepar_map.png")


@figure("ellprof")
def fig_ellprof():
    from bbarolo import Ellprof
    f = Ellprof(CUBE, nradii=41, radsep=30, xpos=77, ypos=77, inc=60, pa=123.7, mask="SEARCH", side="B")
    res = f.run()
    t = res.profile
    r = np.asarray(t["radii"])
    sd = np.asarray(t["himassdens"])
    scatter = sd * np.asarray(t["stddev"]) / np.asarray(t["mean"])      # scatter of single pixels
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.fill_between(r, np.maximum(sd - scatter, 1e-3), sd + scatter, color="C0", alpha=0.25,
                    label="scatter of the pixels in the ring")
    ax.plot(r, sd, "o-", ms=3, lw=1, color="C0", label="HIMASSDENS, observed")
    d = res.profile_deconvolved                                         # the profile deconvolved from the beam
    ax.errorbar(np.asarray(d["radii"]), np.asarray(d["himassdens"]),
                yerr=[np.asarray(d["e_himassdens_lo"]), np.asarray(d["e_himassdens_hi"])],
                fmt="s-", ms=3, lw=1, capsize=0, color="C3", label="HIMASSDENS, deconvolved")
    ax.set_yscale("log")
    ax.set_xlabel("Radius (arcsec)")
    ax.set_ylabel("HI mass surface density (M$_\\odot$/pc$^2$)")
    ax.legend(fontsize=8)
    save(fig, "ellprof_profile.png")


@figure("pvslice")
def fig_pvslice():
    from bbarolo import PVSlice
    a = PVSlice(CUBE, xpos_pv=77, ypos_pv=77, pa_pv=123.7).run()
    b = PVSlice(CUBE, p1_pv="30 105", p2_pv="120 45", width_pv=60).run()
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.8))
    for ax, res, t in zip(axs, (a, b), ("Major axis (PA_PV = 123.7$^\\circ$)",
                                         "Between two points, 60 arcsec wide")):
        h, d = res.pv.header, np.asarray(res.pv.data, float)
        off = 3600 * (h["CDELT1"] * (np.arange(h["NAXIS1"]) + 1 - h["CRPIX1"]) + h["CRVAL1"])
        vel = (h["CDELT2"] * (np.arange(h["NAXIS2"]) + 1 - h["CRPIX2"]) + h["CRVAL2"]) / 1000.
        im = ax.pcolormesh(off, vel, d * 1000, cmap="Greys", shading="nearest")
        ax.set_title(t, fontsize=10)
        ax.set_xlabel("Offset (arcsec)")
        fig.colorbar(im, ax=ax, label="Flux (mJy/beam)")
    axs[0].set_ylabel("Velocity (km/s)")
    fig.tight_layout()
    save(fig, "pvslice_pv.png")


# (end of fits and profiles)


# ---------------------------------------------------------------- SEARCH, MAKEMASK, SMOOTH, SMOOTHSPEC, CONTSUB
def _vel_axis(h):
    """The velocity axis (km/s) of a cube header (VELO or FREQ not handled: the example is VELO-HEL)."""
    n = h["NAXIS3"]
    return (h["CRVAL3"] + (np.arange(n) + 1 - h["CRPIX3"]) * h["CDELT3"]) / 1000.0


def _offsets(h):
    """The offsets (arcsec) of the pixel edges of a cube header, for the extent of imshow."""
    ex = []
    for ax in (1, 2):
        n, d = h[f"NAXIS{ax}"], abs(h[f"CDELT{ax}"]) * 3600
        ex += [-(n / 2.0) * d, (n / 2.0) * d]
    return ex


@figure("search")
def fig_search():
    from bbarolo import Search
    res = Search(CUBE, searchtype="spatial", snrcut=10, growthcut=3, minpix=30, minchannels=5).run(threads=4)
    fig, ax = plt.subplots(1, 2, figsize=(9, 4.2))
    for a, hdu, lab, cm in [(ax[0], res.detmap, "detected channels", "viridis"),
                            (ax[1], res.maps["mom0th"], "total intensity (Jy/beam km/s)", "magma")]:
        d = np.array(hdu.data, dtype=float)
        if cm == "viridis":
            d = np.where(d > 0, d, np.nan)
        im = a.imshow(d, origin="lower", cmap=cm)
        fig.colorbar(im, ax=a, fraction=0.046, pad=0.03, label=lab)
        a.set_xlabel("x (pixel)")
        a.set_ylabel("y (pixel)")
    ax[0].set_title("detmap")
    ax[1].set_title("mom0th")
    fig.tight_layout()
    save(fig, "search_maps.png")


@figure("makemask")
def fig_makemask():
    from bbarolo import MakeMask
    from astropy.io import fits
    cube = fits.getdata(CUBE)
    h = fits.getheader(CUBE)
    z = 30
    fig, ax = plt.subplots(figsize=(5.6, 5))
    ax.imshow(cube[z], origin="lower", cmap="Greys", extent=_offsets(h), vmin=0, vmax=0.05)
    ext = _offsets(h)
    for mask, kw, col, lab in [("SEARCH", {}, "tab:blue", "SEARCH"),
                               ("SMOOTH&SEARCH", dict(factor=2), "tab:red", "SMOOTH&SEARCH, FACTOR 2"),
                               ("THRESHOLD", dict(threshold=0.003), "tab:green", "THRESHOLD, 0.003 Jy/beam")]:
        m = MakeMask(CUBE, mask=mask, **kw).run(threads=4).mask.data[z]
        ax.contour(m, levels=[0.5], colors=col, linewidths=1.3, origin="lower", extent=ext)
        ax.plot([], [], color=col, label=lab)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlabel("offset (arcsec)")
    ax.set_ylabel("offset (arcsec)")
    ax.set_title(f"channel {z} and the edge of three masks")
    save(fig, "makemask_masks.png")


@figure("smooth")
def fig_smooth():
    from bbarolo import Smooth
    from astropy.io import fits
    from matplotlib.patches import Ellipse
    h0 = fits.getheader(CUBE)
    d0 = fits.getdata(CUBE)
    res = Smooth(CUBE, newbeam=150, reduce=True).run(threads=4)
    h1, d1 = res.cube.header, res.cube.data
    z = 30
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 4.3), sharex=True, sharey=True)
    for a, d, h, t in [(ax[0], d0, h0, "original"), (ax[1], d1, h1, "SMOOTH, 150 arcsec, REDUCE")]:
        ext = _offsets(h)
        im = a.imshow(d[z], origin="lower", cmap="viridis", extent=ext, vmin=0, vmax=np.nanmax(d[z]))
        fig.colorbar(im, ax=a, fraction=0.046, pad=0.03, label="Jy/beam")
        bmaj, bmin, bpa = h["BMAJ"] * 3600, h["BMIN"] * 3600, h.get("BPA", 0)
        a.add_patch(Ellipse((ext[0] + 0.8 * bmaj, ext[2] + 0.8 * bmaj), bmin, bmaj, angle=bpa,
                            fc="white", ec="k"))
        a.set_title(t)
        a.set_xlabel("offset (arcsec)")
    ax[0].set_ylabel("offset (arcsec)")
    fig.tight_layout()
    save(fig, "smooth_channel.png")


@figure("smoothspec")
def fig_smoothspec():
    from bbarolo import SmoothSpec
    from astropy.io import fits
    h = fits.getheader(CUBE)
    d0 = fits.getdata(CUBE)
    res = SmoothSpec(CUBE, window_type="HANNING", window_size=11).run()
    v = _vel_axis(h)
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(v, d0[:, 77, 77], "o-", color="0.55", ms=3, label="original")
    ax.plot(v, res.cube.data[:, 77, 77], "-", color="tab:red", label="HANNING, 11 channels")
    ax.set_xlabel("velocity (km/s)")
    ax.set_ylabel("intensity (Jy/beam)")
    ax.set_title("spectrum of the pixel (77, 77)")
    ax.legend()
    save(fig, "smoothspec_spectrum.png")


@figure("contsub")
def fig_contsub():
    from astropy.io import fits
    from bbarolo import ContSub
    d0 = fits.getdata(CUBE)
    h = fits.getheader(CUBE)
    # A synthetic continuum, a slope of 0.0004 Jy/beam per channel from 0.02 Jy/beam, added to the cube
    z = np.arange(d0.shape[0])
    cont = 0.02 + 0.0004 * z
    hdu = fits.PrimaryHDU((d0 + cont[:, None, None]).astype("float32"), h)
    res = ContSub(hdu, contorder=1, exclwind="3:56").run(threads=4)
    v = _vel_axis(h)
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(v, hdu.data[:, 77, 77], "o-", color="0.55", ms=3, label="cube with continuum")
    ax.plot(v, d0[:, 77, 77] * 0 + hdu.data[:, 77, 77] - res.cube.data[:, 77, 77], "--", color="tab:blue",
            label="fitted continuum")
    ax.plot(v, res.cube.data[:, 77, 77], "-", color="tab:red", label="after CONTSUB")
    ax.axvspan(v[56], v[3], color="0.9", zorder=0, label="excluded from the fit (EXCLWIND)")
    ax.set_xlabel("velocity (km/s)")
    ax.set_ylabel("intensity (Jy/beam)")
    ax.set_title("spectrum of the pixel (77, 77), synthetic continuum")
    ax.legend(fontsize=8)
    save(fig, "contsub_spectrum.png")


# (end of cubes and masks)


# ---------------------------------------------------------------- maps, STATS, REND3D, FITS utilities
@figure("maps")
def fig_maps():
    from bbarolo import Maps
    m = Maps(CUBE, mask="THRESHOLD", threshold=0.01, totalmap=True, velocitymap=True,
             dispersionmap=True, globalprofile=True)
    res = m.run(threads=4)
    fig, axs = plt.subplots(1, 3, figsize=(10, 3.6))
    panels = [(res.totalmap, "Total intensity (Jy/beam km/s)", "viridis"),
              (res.velocitymap, "Velocity (km/s)", "RdBu_r"),
              (res.dispersionmap, "Dispersion (km/s)", "magma")]
    for ax, (hdu, title, cmap) in zip(axs, panels):
        d = np.asarray(hdu.data, float)
        lo, hi = np.nanpercentile(d, [1, 99])
        im = ax.imshow(d, origin="lower", vmin=lo, vmax=hi, cmap=cmap)
        fig.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("x (pixel)")
    axs[0].set_ylabel("y (pixel)")
    fig.tight_layout()
    save(fig, "maps_moments.png")
    sp = res.spectrum
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.plot(np.asarray(sp["spectral"]) / 1000, sp["flux"], color="C0")
    ax.set_xlabel("Velocity (km/s)")
    ax.set_ylabel("Flux (Jy/beam, summed over pixels)")
    ax.grid(alpha=0.3)
    save(fig, "maps_profile.png")


@figure("rend3d")
def fig_rend3d():
    from bbarolo import Rend3D
    res = Rend3D(CUBE, mask="THRESHOLD", threshold=0.01, rendangle=180).run(threads=4)
    cube = np.asarray(res.image.data, float)
    fig, axs = plt.subplots(1, 4, figsize=(10, 3))
    for ax, ang in zip(axs, (0, 60, 120, 179)):
        ax.imshow(cube[ang], origin="lower", cmap="viridis")
        ax.set_title(f"{ang} deg", fontsize=9)
        ax.set_xlabel("x (pixel)")
    axs[0].set_ylabel("y (pixel)")
    fig.tight_layout()
    save(fig, "rend3d_views.png")


# (end of maps and utilities)


# ---------------------------------------------------------------- examples: NGC 2403 3DFIT, simulated cubes
_N2403_FIT = {}


def _n2403_refined():
    """The refined fit of NGC 2403 (page ngc2403-3dfit): INC and PA free, TWOSTAGE, errors, ADRIFT."""
    if "res" not in _N2403_FIT:
        from bbarolo import FitMod3D
        f = FitMod3D(CUBE, nradii=20, radsep=60, xpos=77, ypos=77, vsys=132.8, vrot=120, vdisp=8, inc=60,
                     pa=123.7, z0=10, free="VROT VDISP INC PA", deltainc=15, mask="SEARCH", norm="LOCAL",
                     flagerrors=True, adrift=True)
        _N2403_FIT["res"] = f.run(threads=4)
    return _N2403_FIT["res"]


@figure("ngc2403_first")
def fig_ngc2403_first():
    from bbarolo import Maps, PVSlice
    m = Maps(CUBE, mask="THRESHOLD", threshold=0.002, totalmap=True, velocitymap=True).run(threads=4)
    pv = PVSlice(CUBE, xpos_pv=77, ypos_pv=77, pa_pv=123.7).run()
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.7))
    for ax, hdu, title, cmap in ((axs[0], m.totalmap, "Total intensity (Jy/beam km/s)", "viridis"),
                                 (axs[1], m.velocitymap, "Velocity (km/s)", "RdBu_r")):
        d = np.asarray(hdu.data, float)
        lo, hi = np.nanpercentile(d[np.isfinite(d) & (d != 0)], [1, 99])
        im = ax.imshow(d, origin="lower", vmin=lo, vmax=hi, cmap=cmap)
        fig.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("x (pixel)")
    axs[0].set_ylabel("y (pixel)")
    h, d = pv.pv.header, np.asarray(pv.pv.data, float)
    off = 3600 * (h["CDELT1"] * (np.arange(h["NAXIS1"]) + 1 - h["CRPIX1"]) + h["CRVAL1"])
    vel = (h["CDELT2"] * (np.arange(h["NAXIS2"]) + 1 - h["CRPIX2"]) + h["CRVAL2"]) / 1000.
    im = axs[2].pcolormesh(off, vel, d * 1000, cmap="Greys", shading="nearest")
    fig.colorbar(im, ax=axs[2], label="mJy/beam", shrink=0.8)
    axs[2].set_title("PV slice, PA = 123.7$^\\circ$", fontsize=9)
    axs[2].set_xlabel("Offset (arcsec)")
    axs[2].set_ylabel("Velocity (km/s)")
    fig.tight_layout()
    save(fig, "ngc2403_first.png")


@figure("ngc2403_geometry")
def fig_ngc2403_geometry():
    res = _n2403_refined()
    t1, t2 = res.rings_stage1, res.rings
    fig, axs = plt.subplots(4, 1, sharex=True, figsize=(6, 8))
    for ax, k, lab in zip(axs, ("vrot", "vdisp", "inc", "pa"),
                          ("VROT (km/s)", "VDISP (km/s)", "INC (deg)", "PA (deg)")):
        ax.plot(t1["radii"], t1[k], "o", ms=4, color="0.55", label="stage 1")
        ax.plot(t2["radii"], t2[k], "-o", ms=3, color="C3", label="stage 2")
        ax.set_ylabel(lab)
    axs[0].legend(fontsize=8)
    axs[-1].set_xlabel("Radius (arcsec)")
    fig.tight_layout()
    save(fig, "ngc2403_geometry.png")


@figure("ngc2403_check")
def fig_ngc2403_check():
    res = _n2403_refined()
    save(res.plot_pv(), "ngc2403_pv.png")
    data, model, resid = (np.asarray(x.data, float) for x in (res.data, res.model, res.residual))
    h = res.data.header
    vel = (np.arange(data.shape[0]) + 1 - h["CRPIX3"]) * h["CDELT3"] + h["CRVAL3"]
    if str(h.get("CUNIT3", "")).lower() in ("m/s", "ms"):
        vel = vel / 1000.
    chans = [14, 20, 26, 31, 36, 42]
    vmax = np.nanmax(data)
    fig, axs = plt.subplots(3, len(chans), figsize=(2.2 * len(chans), 6.8), sharex=True, sharey=True)
    for j, c in enumerate(chans):
        axs[0, j].imshow(data[c], origin="lower", cmap="Greys", vmin=0, vmax=0.6 * vmax)
        axs[0, j].contour(model[c], levels=vmax * np.array([0.1, 0.3, 0.6]), colors="red", linewidths=0.7)
        axs[0, j].set_title(f"{vel[c]:.0f} km/s", fontsize=9)
        im = axs[1, j].imshow(model[c], origin="lower", cmap="Greys", vmin=0, vmax=0.6 * vmax)
        im2 = axs[2, j].imshow(resid[c], origin="lower", cmap="RdBu_r", vmin=-0.3 * vmax, vmax=0.3 * vmax)
    for ax in axs.ravel():
        ax.set_xticks([]), ax.set_yticks([])
    for i, lab in enumerate(("data + model", "model", "residual")):
        axs[i, 0].set_ylabel(lab)
    fig.tight_layout()
    save(fig, "ngc2403_channels.png")


@figure("ngc2403_errors")
def fig_ngc2403_errors():
    res = _n2403_refined()
    t, ad = res.rings, res.asymdrift
    fig, ax = plt.subplots(figsize=(6, 3.8))
    r = np.asarray(t["radii"])
    ax.errorbar(r, t["vrot"], yerr=[-np.asarray(t["e_vrot1"]), np.asarray(t["e_vrot2"])], fmt="o-", ms=3,
                capsize=2, label="$V_\\mathrm{rot}$ (FLAGERRORS)")
    ax.plot(ad["radii"], ad["vcirc"], "s-", ms=3, color="C3", label="$V_\\mathrm{c}$ (ADRIFT)")
    ax.set_xlabel("Radius (arcsec)")
    ax.set_ylabel("Velocity (km/s)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save(fig, "ngc2403_errors.png")


def _simulate(seed, snr=20., path=None):
    """The simulated galaxy of the page `simulated`: returns (SimulatedGalaxyCube, true ring dict)."""
    from bbarolo import SimulatedGalaxyCube
    radii = np.arange(15, 360, 30.)                         # ring centres (arcsec)
    vrot = 200. * 2 / np.pi * np.arctan(radii / 60.)         # km/s
    s = SimulatedGalaxyCube(axisDim=[64, 64, 48], cdelts=[-20. / 3600., 20. / 3600., 10.], beam=60. / 3600.,
                            bunit="JY/BEAM", obj="sim")
    s.define_galaxy(radii=radii, xpos=31.5, ypos=31.5, vsys=0, vrot=vrot, vdisp=10., inc=60., pa=40.,
                    dens=np.exp(-radii / 200.), z0=10.,
                    warpinc=False, warppa=False, flare=False, radmotion=False)
    s.run(smooth=True, noise=0, threads=4, galmodtype="ANALYTIC")
    peak = float(s.model_nonoise.data.max())
    s.run(smooth=True, noise=peak / snr, seed=seed, threads=4, galmodtype="ANALYTIC")
    return s, radii, vrot


def _fit_sim(s, radii, **kw):
    from bbarolo import FitMod3D
    f = FitMod3D(s.model, radii=radii, xpos=31.5, ypos=31.5, vsys=0, z0=10, vrot=150, vdisp=15, inc=55, pa=35,
                 free="VROT VDISP INC PA", deltainc=15, deltapa=20, mask="SEARCH", norm="LOCAL", **kw)
    return f.run(threads=4)


@figure("simulated")
def fig_simulated():
    s, radii, vrot = _simulate(1)
    res = _fit_sim(s, radii, flagerrors=True)
    t = res.rings
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.8))
    err = lambda k: [-np.asarray(t[f"e_{k}1"]), np.asarray(t[f"e_{k}2"])]
    runs = []
    for seed in range(2, 22):
        s2, _, _ = _simulate(seed)
        r2 = _fit_sim(s2, radii)
        runs.append((np.asarray(r2.rings["vrot"]), np.asarray(r2.rings["vdisp"])))
        for ax, y in zip(axs, runs[-1]):
            ax.plot(radii, y, color="0.8", lw=0.8, zorder=1)
    for ax, k, true, lab in zip(axs, ("vrot", "vdisp"), (vrot, np.full(len(radii), 10.)),
                                ("VROT (km/s)", "VDISP (km/s)")):
        ax.errorbar(radii, t[k], yerr=err(k), fmt="o", ms=3, capsize=2, color="C0", zorder=3, label="fit, seed 1")
        ax.plot(radii, true, "-", color="C3", zorder=2, label="true")
        ax.set_xlabel("Radius (arcsec)")
        ax.set_ylabel(lab)
    axs[0].plot([], [], color="0.8", label="20 other noise seeds")
    axs[0].legend(fontsize=8)
    fig.tight_layout()
    save(fig, "simulated_rc.png")
    v = np.array([r[0] for r in runs]); d = np.array([r[1] for r in runs])
    print("median vrot error % per ring:", np.round(100 * (np.median(v, 0) / vrot - 1), 1))
    print("scatter vrot % per ring:", np.round(100 * np.std(v, 0) / vrot, 1))
    print("median vdisp per ring:", np.round(np.median(d, 0), 1))
    print("peak S/N check rms", s.noiserms)
    sa, _, _ = _simulate(1)
    print("REPRODUCIBLE", np.array_equal(s.model.data, sa.model.data))
    print("seed1 vrot", np.round(np.asarray(t["vrot"]), 1), "vdisp", np.round(np.asarray(t["vdisp"]), 1),
          "inc", float(t["inc"][0]), "pa", float(t["pa"][0]))
    print("errors vrot lo", np.round(-np.asarray(t["e_vrot1"]), 1), "hi", np.round(np.asarray(t["e_vrot2"]), 1))
    save(res.plot_pv(), "simulated_pv.png")


# (end of examples 1)


# ---------------------------------------------------------------- examples: Bayesian fit, GalMod3D
def _pv_axes(hdu):
    """Offset (arcsec), velocity (km/s, increasing) and data of a PV HDU."""
    h, d = hdu.header, np.asarray(hdu.data, float)
    off = 3600 * (h["CDELT1"] * (np.arange(h["NAXIS1"]) + 1 - h["CRPIX1"]) + h["CRVAL1"])
    vel = (h["CDELT2"] * (np.arange(h["NAXIS2"]) + 1 - h["CRPIX2"]) + h["CRVAL2"]) / 1000.
    if vel[-1] < vel[0]:
        vel, d = vel[::-1], d[::-1]
    return off, vel, d


@figure("galmod3d_lag")
def fig_galmod3d_lag():
    from bbarolo import GalMod3D, PVSlice
    geo = dict(xpos=77, ypos=77, vsys=132.8, inc=85, pa=123.7, flux="measured", threads=4)

    def vcirc(R):
        return 130 * np.tanh(R / 60.)

    def thin(R, z):
        return np.exp(-R / 400.) * np.exp(-z * z / (2 * 10. ** 2))

    def thick(R, z):
        h = 30 + R / 20.                                  # exponential layer, scale height 30-90 arcsec
        return np.exp(-R / 400.) * np.exp(-np.abs(z) / h) / h * 30

    m1 = GalMod3D(CUBE, coords="Rz", dens=thin, vphi=lambda R, z: vcirc(R), vdisp=8., zmax=100, **geo).model()
    m2 = GalMod3D(CUBE, coords="Rz", dens=thick, vphi=lambda R, z: vcirc(R) - 0.3 * np.abs(z), vdisp=8.,
                  zmax=300, **geo).model()
    pa = np.radians(123.7 + 90)                              # towards the minor axis
    dx, dy = -np.sin(pa) * 100 / 16., np.cos(pa) * 100 / 16.   # 100 arcsec, in pixels (16 arcsec)
    fig, axs = plt.subplots(2, 2, figsize=(8.4, 6.6), sharex=True, sharey=True)
    for j, (m, name) in enumerate(((m1, "Thin disk"), (m2, "Thick, lagging"))):
        for i, (shift, where) in enumerate(((0, "major axis"), (1, "z = 100 arcsec"))):
            pv = PVSlice(m, xpos_pv=77 + shift * dx, ypos_pv=77 + shift * dy, pa_pv=123.7, width_pv=48).run().pv
            off, vel, d = _pv_axes(pv)
            ax = axs[i, j]
            im = ax.pcolormesh(off, vel, d * 1000, cmap="Greys", shading="nearest", vmin=0, vmax=300)
            ax.axhline(132.8, color="C3", lw=0.6, ls=":")
            ax.set_title(f"{name}, {where}", fontsize=9)
            if i == 1:
                ax.set_xlabel("Offset (arcsec)")
            if j == 0:
                ax.set_ylabel("Velocity (km/s)")
    fig.colorbar(im, ax=axs, label="Flux (mJy/beam)", shrink=0.8)
    save(fig, "galmod3d_lag.png")


def _warp_fields():
    vc = lambda R: 130 * np.tanh(R / 60.)

    def zwarp(X, Y):
        R = np.hypot(X, Y)
        return 0.2 * np.maximum(R - 300, 0) * X / np.where(R > 0, R, 1.)   # line of nodes: the minor axis

    x = np.linspace(-2000, 2000, 201)
    z = np.linspace(-500, 500, 101)
    X, Y, Z = np.meshgrid(x, x, z, indexing="ij")
    R = np.hypot(X, Y)
    rho = np.exp(-R / 600.) * np.exp(-(Z - zwarp(X, Y)) ** 2 / (2 * 30. ** 2))
    v = vc(R) / np.where(R > 0, R, 1.)
    return (x, z), rho, v * X, -v * Y


@figure("galmod3d_warp")
def fig_galmod3d_warp():
    from bbarolo import GalMod3D, Maps
    (x, z), rho, vy, vx = _warp_fields()
    ax3 = (x, x, z)
    out = []
    for inc in (85, 60):
        m = GalMod3D(CUBE, coords="xyz", xpos=77, ypos=77, vsys=132.8, inc=inc, pa=123.7, dens=(ax3, rho),
                     vx=(ax3, vx), vy=(ax3, vy), vdisp=10., zmax=500, flux="measured", threads=4).model()
        out.append(Maps(m, mask="THRESHOLD", threshold=0.005, totalmap=True, velocitymap=True).run())
    fig, axs = plt.subplots(2, 2, figsize=(7.4, 7.0))
    for row, (res, inc) in enumerate(zip(out, (85, 60))):
        for col, (hdu, cmap, lab) in enumerate(((res.totalmap, "viridis", "Total intensity (Jy/beam km/s)"),
                                                 (res.velocitymap, "RdBu_r", "Velocity (km/s)"))):
            d = np.asarray(hdu.data, float)
            lo, hi = np.nanpercentile(d[np.isfinite(d) & (d != 0)], [1, 99])
            ax = axs[row, col]
            im = ax.imshow(d, origin="lower", vmin=lo, vmax=hi, cmap=cmap)
            fig.colorbar(im, ax=ax, shrink=0.8)
            ax.set_title(f"{lab}, i = {inc}$^\\circ$", fontsize=8)
    for ax in axs[1]:
        ax.set_xlabel("x (pixel)")
    for ax in axs[:, 0]:
        ax.set_ylabel("y (pixel)")
    fig.tight_layout()
    save(fig, "galmod3d_warp.png")


@figure("bayesian")
def fig_bayesian():
    """The fit of the page examples/bayesian: about 8 minutes on 4 processes."""
    from bbarolo import Bayesian3DFIT, FitMod3D
    rings = dict(radii=[300, 600, 900], xpos=77, ypos=77, vsys=132.8, vrot=100, vdisp=10, inc=55, pa=123.7,
                 z0=10, mask="SEARCH")
    f = Bayesian3DFIT(CUBE, **rings)
    f.priors["vrot"] = dict(name="uniform", loc=50, scale=200)      # km/s
    f.priors["vdisp"] = dict(name="uniform", loc=0, scale=30)       # km/s
    f.priors["inc"] = dict(name="uniform", loc=40, scale=25)        # degrees
    f.priors["phi"] = dict(name="uniform", loc=110, scale=30)       # degrees
    t0 = time.time()
    f.run(freepar="VROT VDISP_SINGLE INC_SINGLE PA_SINGLE SIGMA RADMAX", threads=4, method="nautilus",
          sampler_kwargs=dict(nlive=500, seed=1), run_kwargs=dict(dlogz=1, n_eff=3000))
    print(f"    fit: {time.time() - t0:.0f} s")
    f.print_stats()
    if os.environ.get("BAYES_SAVE"):
        f.save_results(os.environ["BAYES_SAVE"])
    names = list(f.freepar_names)
    cols = [names.index(k) for k in ("inc", "phi", "radmax", "vdisp")]
    fig = f.corner_plot(samples=f.samples_equal[:, cols], labels=["INC", "PA", "RADMAX", "VDISP"])
    save(fig, "bayesian_corner.png")

    # the rotation curve against 3DFIT: with its own geometry, and with that of the posterior medians
    fa = FitMod3D(CUBE, **rings, free="VROT VDISP INC PA").run(threads=4)
    inc, pa = f.outri.r["inc"][0], f.outri.r["phi"][0]
    fb = FitMod3D(CUBE, **{**rings, "inc": inc, "pa": pa}, free="VROT VDISP").run(threads=4)
    R = np.asarray(f.outri.r["radii"], float)
    lo, hi = f.ring_errors("vrot")
    vr = np.asarray(f.outri.r["vrot"], float)
    fig, ax = plt.subplots(figsize=(5.4, 4))
    curves = np.array([s[[names.index(f"vrot{i + 1}") for i in range(len(R))]] for s in f.samples_equal])
    for c in curves[:: max(1, len(curves) // 150)]:
        ax.plot(R, c, color="C0", alpha=0.04, lw=1)
    ax.errorbar(R, vr, [lo, hi], fmt="o", color="C0", capsize=3, label="BAYESIAN3DFIT: median, 68%")
    ax.plot(R + 6, fa.rings["vrot"], "s", color="C1", label=f"3DFIT, INC, PA free ({fa.rings['inc'][0]:.1f}$^\\circ$)")
    ax.plot(R - 6, fb.rings["vrot"], "^", color="C2", label=f"3DFIT, INC = {inc:.1f}$^\\circ$ (Bayesian)")
    if f.disk_edge is not None:
        ax.axvline(f.disk_edge, color="grey", ls=":")
        ax.text(f.disk_edge, 0.04, "edge ", color="grey", fontsize=8, ha="right", transform=ax.get_xaxis_transform())
    ax.set_xlabel("Radius (arcsec)")
    ax.set_ylabel("VROT (km/s)")
    ax.legend(fontsize=8, loc="lower right")
    save(fig, "bayesian_vrot.png")
    print("    3DFIT, own geometry:", fa.rings["vrot"], fa.rings["inc"][0], fa.rings["pa"][0])
    print("    3DFIT, Bayes geometry:", fb.rings["vrot"], fb.rings["vdisp"])
    print("    Bayes:", vr, lo, hi)


# (end of examples 2)


# ---------------------------------------------------------------- user guide
@figure("conventions")
def fig_conventions():
    """The geometry of a ring on the sky: position angle, receding half, near side, sense of rotation."""
    from matplotlib.patches import Arc, Ellipse
    pa, inc, a = 40.0, 60.0, 1.0
    b = a * np.cos(np.radians(inc))
    p = np.radians(pa)
    major = np.array([-np.sin(p), np.cos(p)])           # (right, up) of the receding half of the major axis
    near = np.array([-np.cos(p), -np.sin(p)])           # the near side, at PA + 90 degrees
    fig, ax = plt.subplots(figsize=(4.8, 4.6))
    ax.add_patch(Ellipse((0, 0), 2 * a, 2 * b, angle=90 + pa, fc="#f2f2f2", ec="k", lw=1.5))
    ax.plot([0, 1.55 * major[0]], [0, 1.55 * major[1]], c="#c0392b", lw=2)
    ax.plot([0, -1.2 * major[0]], [0, -1.2 * major[1]], c="#2471a3", lw=2)
    ax.text(*(1.17 * major + [0.06, 0.02]), "receding\nhalf (red)", color="#c0392b", fontsize=9, va="bottom")
    ax.text(*(-1.1 * major + [0.02, -0.06]), "approaching\nhalf (blue)", color="#2471a3", fontsize=9, va="top", ha="right")
    ax.text(*(0.86 * b * near + [0.04, -0.08]), "near side", fontsize=9, ha="center", va="top")
    ax.text(0.55, 0.62, "far side", fontsize=9, ha="center")
    ax.annotate("", xy=(0, 1.55), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color="0.4", ls="--"))
    ax.text(0.04, 1.6, "North", fontsize=9, color="0.3")
    ax.annotate("", xy=(-1.1, -1.15), xytext=(-0.2, -1.15), arrowprops=dict(arrowstyle="-|>", color="0.4"))
    ax.text(-0.65, -1.3, "East", fontsize=9, color="0.3", ha="center")
    ax.add_patch(Arc((0, 0), 1.2, 1.2, theta1=90, theta2=90 + pa, color="k", lw=1.2))
    ax.text(-0.2, 0.7, "PA", fontsize=10)
    # sense of rotation: clockwise on the sky, from the receding half towards the far side
    # the points of the ellipse in its own frame, rotated to the sky
    c, s = np.cos(np.radians(90 + pa)), np.sin(np.radians(90 + pa))
    t = np.radians(-np.linspace(200, 250, 30))                   # outside the ring, from the approaching end to the near side
    ex, ey = 1.14 * a * np.cos(t), 1.14 * b * np.sin(t)
    ax.plot(c * ex - s * ey, s * ex + c * ey, c="k", lw=1.2)
    ax.annotate("", xy=(c * ex[-1] - s * ey[-1], s * ex[-1] + c * ey[-1]),
                xytext=(c * ex[-3] - s * ey[-3], s * ex[-3] + c * ey[-3]),
                arrowprops=dict(arrowstyle="-|>", color="k"))
    ax.set_xlim(-1.7, 1.7)
    ax.set_ylim(-1.5, 1.8)
    ax.set_aspect("equal")
    ax.axis("off")
    save(fig, "conventions_ring.png")

# (end of user guide)


def find_cube(examples):
    cands = [examples, os.environ.get("BBAROLO_EXAMPLES"), os.path.join(os.path.dirname(REPO), "examples")]
    for d in cands:
        if d and os.path.isfile(os.path.join(d, "ngc2403.fits")):
            return os.path.join(d, "ngc2403.fits")
    sys.exit("ngc2403.fits not found: give its folder with --examples (see getting-started/example-data).")


def main():
    global CUBE, OUT
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="*", help="the figures to make (default: all)")
    ap.add_argument("--examples", help="the folder of ngc2403.fits")
    ap.add_argument("--out", help="the folder of the PNG files (default: docs/figures)")
    ap.add_argument("--list", action="store_true", help="list the figures and exit")
    a = ap.parse_args()
    if a.list:
        print("\n".join(FIGURES))
        return
    unknown = set(a.names) - set(FIGURES)
    if unknown:
        sys.exit(f"Unknown figures: {', '.join(sorted(unknown))}. Known: {', '.join(FIGURES)}")
    CUBE = find_cube(a.examples)
    if a.out:
        OUT = os.path.abspath(a.out)
    os.makedirs(OUT, exist_ok=True)
    for name in a.names or FIGURES:
        t = time.time()
        print(f"{name}:")
        FIGURES[name]()
        print(f"    ({time.time() - t:.0f} s)")


if __name__ == "__main__":
    main()
