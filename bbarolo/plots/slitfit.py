"""
The plots of BAYESIANSLITFIT. Each plot is a function that draws it from arrays and returns a matplotlib Figure,
writing nothing:

  plot_slit_pv          the slit: data, best model and residuals             <prefix>_slitfit_pv.pdf
  plot_slit_parameters  the rings of the best fit against radius            <prefix>_slitfit_parameters.pdf

The plotting scripts written by BayesianSlitFit.write (in <outfolder>/plotscripts/) contain a copy of these functions
and of read_slit_rings: they must use only the names imported by fit3d._IMPORTS.
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from astropy.visualization import ImageNormalize, PercentileInterval, PowerStretch

from .utils import StyledFigure


def read_slit_rings(fname):
    """The rings of a ring file of BAYESIANSLITFIT (<prefix>_slitfit_rings.txt) as a dict of arrays, by the names of the
    columns in lower case without their units (rad, vrot, e_vrot_lo, e_vrot_hi, vdisp, ...)."""
    with open(fname) as f:
        names = f.readline().lstrip("#").split()
    data = np.atleast_2d(np.loadtxt(fname, comments="#"))
    return {n.split("(")[0].lower(): data[:, i] for i, n in enumerate(names)}


def plot_slit_pv(data, model, resid, *, offsets, spec, specunit="", curves=(), points=None, cont=None, gname="",
                 lines=""):
    """The slit (<prefix>_slitfit_pv.pdf): the data (grey scale, blue contours), the data with the contours of the best
    model (red) and the residuals in units of the noise, against the offset along the slit from the centre (arcsec)
    and the spectral axis, with the line-of-sight velocities of the best rings (gold lines, one per spectral line).

    data, model, resid: arrays (nspec, npos) (resid: (data - model) / noise); offsets: the offsets of the positions
    (arcsec); spec: the spectral values of the channels (specunit: their unit); curves: a list of (offsets, spectral
    values) of the velocities of the rings, one per line; points: the velocities measured along the slit as (offsets,
    spectral values) (the first line), or None; cont: the lowest contour (default 3 times the noise of the data,
    estimated from the residuals); gname: the name of the galaxy; lines: a description of the lines. Returns the
    Figure."""
    style = {"xtick.direction": "in", "xtick.top": True, "ytick.direction": "in", "ytick.right": True,
             "contour.negative_linestyle": "solid", "font.family": "sans-serif", "font.size": 12,
             "text.usetex": False, "mathtext.fontset": "cm", "mathtext.default": "regular"}
    data = np.asarray(data, float)
    model = np.asarray(model, float)
    offsets, spec = np.asarray(offsets, float), np.asarray(spec, float)
    if cont is None:
        rd = data - model
        noise = 1.4826 * np.nanmedian(np.abs(rd - np.nanmedian(rd)))
        cont = 3 * noise if noise > 0 else 0.05 * np.nanmax(data)
    levels = np.array([1, 2, 4, 8, 16, 32, 64, 128]) * cont
    levels = levels[levels <= np.nanmax(np.r_[np.ravel(data[np.isfinite(data)]), cont])] if np.any(
        np.isfinite(data)) else levels[:1]
    dx = offsets[1] - offsets[0] if offsets.size > 1 else 1.
    ds = spec[1] - spec[0] if spec.size > 1 else 1.
    ext = [offsets[0] - dx / 2, offsets[-1] + dx / 2, spec[0] - ds / 2, spec[-1] + ds / 2]
    fin = data[np.isfinite(data)]
    norm = ImageNormalize(vmin=max(cont / 3., np.percentile(fin, 1) if fin.size else 0.),
                          vmax=np.percentile(fin, 99.8) if fin.size else 1., stretch=PowerStretch(0.5))
    with mpl.rc_context(style):
        fig = plt.figure(FigureClass=StyledFigure, style=style, figsize=(16, 6.5), dpi=150)
        axs = [fig.add_axes([0.06 + i * 0.31, 0.12, 0.27, 0.75]) for i in range(3)]
        titles = ("Data", "Data and best model", "Residuals / noise")
        for i, ax in enumerate(axs):
            if i < 2:
                ax.imshow(data, origin="lower", cmap="Greys", norm=norm, aspect="auto", interpolation="nearest",
                          extent=ext)
                ax.contour(data, levels, origin="lower", linewidths=0.8, colors="#00008B", extent=ext)
                if i == 1:
                    ax.contour(model, levels, origin="lower", linewidths=1.3, colors="#B22222", extent=ext)
            else:
                im = ax.imshow(resid, origin="lower", cmap="RdBu_r", vmin=-4, vmax=4, aspect="auto",
                               interpolation="nearest", extent=ext)
                cax = fig.add_axes([0.06 + 3 * 0.31 - 0.035, 0.12, 0.008, 0.75])
                fig.colorbar(im, cax=cax)
            for k, (cs, cw) in enumerate(curves):
                if i < 2:
                    ax.plot(cs, cw, "-", color="gold", lw=1.5, alpha=0.9, zorder=5)
            if points is not None and i == 0:
                ax.plot(points[0], points[1], ".", color="gold", ms=4, zorder=6)
            ax.axvline(0, color="k", lw=0.6)
            ax.set_xlim(ext[0], ext[1])
            ax.set_ylim(ext[2], ext[3])
            ax.set_title(titles[i], fontsize=13)
            ax.set_xlabel("Offset along the slit (arcsec)")
            if i == 0:
                ax.set_ylabel(f"Spectral axis ({specunit})" if specunit else "Spectral axis")
            else:
                ax.tick_params(labelleft=False)
        fig.text(0.5, 0.95, f"{gname}" + (f"  ({lines})" if lines else ""), ha="center", fontsize=15)
    return fig


def plot_slit_parameters(rings, *, scalars=None, gname=""):
    """The rings of the best fit against radius (<prefix>_slitfit_parameters.pdf): VROT, VDISP, VRAD and the density
    of the model, with their 1-sigma errors (e_<name>_lo, e_<name>_hi, if in rings), and the single values (scalars:
    a dict of name -> text, e.g. VSYS, XPOS, INC, SLITANGLE). rings: a dict of arrays (see read_slit_rings). Returns
    the Figure."""
    style = {"xtick.direction": "in", "xtick.top": True, "ytick.direction": "in", "ytick.right": True,
             "font.family": "sans-serif", "font.size": 11, "text.usetex": False, "mathtext.fontset": "cm",
             "mathtext.default": "regular", "errorbar.capsize": 0}
    color = "#B22222"
    rad = rings["rad"]
    rmax = 1.1 * np.nanmax(rad)

    def err(name):
        lo, hi = rings.get(f"e_{name}_lo"), rings.get(f"e_{name}_hi")
        return None if lo is None or hi is None else [lo, hi]
    panels = [("vrot", r"V$_\mathrm{rot}$ (km/s)"), ("vdisp", r"$\sigma_\mathrm{gas}$ (km/s)"),
              ("vrad", r"V$_\mathrm{rad}$ (km/s)"), ("dens", "Density of the model")]
    with mpl.rc_context(style):
        fig = plt.figure(FigureClass=StyledFigure, style=style, figsize=(11, 8), dpi=150)
        for i, (key, label) in enumerate(panels):
            ax = fig.add_axes([0.08 + (i % 2) * 0.47, 0.55 - (i // 2) * 0.42, 0.38, 0.36])
            y = rings.get(key)
            if y is not None and key == "vrad" and not np.any(y) and err(key) is None:
                y = None                                    # (VRAD fixed at 0)
            if y is None:
                ax.set_visible(False)
                continue
            ax.errorbar(rad, y, yerr=err(key), fmt="o", color=color)
            ax.set_xlim(0, rmax)
            if key in ("vrot", "vdisp", "dens"):
                ax.set_ylim(0, 1.2 * max(np.nanmax(np.asarray(y) + (err(key)[1] if err(key) is not None else 0)),
                                         1e-30))
            else:
                ax.axhline(0, color="k", lw=0.5)
            ax.set_ylabel(label)
            ax.set_xlabel("Radius (arcsec)")
        if scalars:
            fig.text(0.5, 0.965, "   ".join(f"{k} = {v}" for k, v in scalars.items()), ha="center", fontsize=11)
        fig.text(0.5, 0.995, gname, ha="center", va="top", fontsize=14)
    return fig
