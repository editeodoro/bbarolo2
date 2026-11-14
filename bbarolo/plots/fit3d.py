"""
The plots of 3DFIT (port of Galfit::writeScripts_Python of galfit_out.cpp). Each plot is a
function that draws it from arrays and returns a matplotlib Figure, writing nothing:

  plot_parameters   the parameters of the rings against radius           <prefix>_parameters.pdf
  plot_asymdrift    the asymmetric drift correction (ADRIFT)              <prefix>_asymmetricdrift.pdf
  plot_chanmaps     channel maps of the data and of a model               <prefix>_chanmaps_<norm>.pdf
  plot_pvs          PV slices along the major and minor axes              <prefix>_pv_<norm>.pdf
  plot_pvslices     PV slices at several angles and offsets               <prefix>_pvslices_<norm>.pdf
  plot_kinmaps      moment maps of the data, a model and the residuals    <prefix>_maps_<norm>.pdf

The rings are dicts of arrays as given by read_rings (rad, vrot, disp, ..., and the errors err_<name>).

write_scripts writes the plotting scripts of a fit in <outfolder>/plotscripts/: each contains a copy of the source
of its plotting function and of the readers below, the settings of the run, and the code that reads the output
files and saves the PDF. They need only numpy, matplotlib, astropy, scipy and plot_utils.py (written next to them),
not bbarolo, and can be edited and run again. Result.plot_parameters() etc. (bbarolo.tasks) draw the same plots
from the outputs in memory (see inmemory.py).

The functions here are copied in the scripts: they must use only the names imported by _IMPORTS.
"""

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colorbar import ColorbarBase
from astropy.io import fits
from astropy.visualization import ImageNormalize, PercentileInterval, PowerStretch
from copy import copy

from .utils import StyledFigure, defineaxis, calculate_padding, build_path, extract_pv

# The imports of the scripts (the names used by the functions copied in them)
_IMPORTS = ("import os\n"
            "import numpy as np\n"
            "import matplotlib as mpl\n"
            "import matplotlib.pyplot as plt\n"
            "from matplotlib import gridspec\n"
            "from matplotlib.colorbar import ColorbarBase\n"
            "from astropy.io import fits\n"
            "from astropy.visualization import ImageNormalize, PercentileInterval, PowerStretch\n"
            "from copy import copy\n"
            "from plot_utils import StyledFigure, defineaxis, calculate_padding, build_path, extract_pv\n")

# The keys of the columns 0-10 of the ring files, and those of the parameters 0-9 of 3DFIT (MAXPAR, the order of
# Galfit.getErrorColumns)
RING_KEYS = ("rad", "vrot", "disp", "inc", "pa", "z0", "dens", "xpos", "ypos", "vsys", "vrad")
PAR_KEYS = ("vrot", "disp", "dens", "z0", "inc", "pa", "xpos", "ypos", "vsys", "vrad")

# the labels of the rows of the maps plot (MAPTYPE MOMENT, GAUSSIAN)
MAPNAMES = (("MOMENT 0TH", "MOMENT 1ST", "MOMENT 2ND"), ("GAUSS INTENSITY", "GAUSS VELOCITY", "GAUSS DISPERSION"))


# ===================================================================================================== readers
def read_rings(fname, errcols=None):
    """The rings of a ring file of BBarolo (<prefix>_rings_final1.txt, ...) as a dict of arrays: rad (arcsec), vrot,
    disp (km/s), inc, pa (deg), z0 (arcsec), dens, xpos, ypos (pixels), vsys, vrad (km/s), the columns 0-10 of the
    file. errcols: {name: column of its lower error} adds err_<name>, the lower (negative) and upper errors."""
    keys = ("rad", "vrot", "disp", "inc", "pa", "z0", "dens", "xpos", "ypos", "vsys", "vrad")
    t = np.genfromtxt(fname, ndmin=2)
    rings = {k: t[:, i] for i, k in enumerate(keys)}
    for k, c in (errcols or {}).items():
        rings["err_" + k] = np.array([t[:, c], t[:, c + 1]])
    return rings


def read_columns(fname, usecols):
    """The columns usecols (numbers from 0) of a text file, as a list of arrays."""
    return list(np.genfromtxt(fname, usecols=usecols, ndmin=2).T)


def read_cube(fname, hdu=0):
    """The data of the FITS cube fname (HDU hdu) as an array (nz, ny, nx): the first plane of any further axis."""
    data = fits.getdata(fname, hdu)
    while data.ndim > 3:
        data = data[0]
    return data


def read_map(fname, hdu=0):
    """The data of the FITS image fname (HDU hdu) as an array (ny, nx)."""
    data = fits.getdata(fname, hdu)
    while data.ndim > 2:
        data = data[0]
    return data


def ring_errors(rings, key):
    """The errors of the parameter key of the rings for errorbar: [lower, upper], both positive (zero without
    errors)."""
    e = rings.get("err_" + key)
    if e is None:
        n = len(rings["rad"])
        return [np.zeros(n), np.zeros(n)]
    return [-np.asarray(e[0]), np.asarray(e[1])]


# ===================================================================================================== plots
def plot_parameters(rings, rings2=None, sdens=None, sdens_obs=None, sdens_label=None, faceon=False, radmax=None,
                    bunit="JY/BEAM * KM/S"):
    """The parameters of the rings against radius (<prefix>_parameters.pdf): VROT, VDISP, the surface density, INC,
    XPOS, VRAD, PA, YPOS and VSYS, with their errors.

    rings: the rings of the first stage (as read_rings, with the errors err_<name>); rings2: those of the second
    stage (TWOSTAGE), drawn on top of the first, or None. sdens: the surface density profile (radii in arcsec,
    values, errors: one array or [lower, upper], positive); sdens_obs: the profile of the data (radii, values;
    empty circles) or None; sdens_label: the label of sdens in the legend (no legend if None); faceon: whether the
    profiles are face-on (per arcsec^2); bunit: the units of the maps. radmax: the outer edge of the disk (arcsec,
    METHOD GLOBAL and BAYESIAN), beyond which the plot is shaded, or None. Returns the Figure."""
    style = {"xtick.direction": "in", "xtick.top": True, "ytick.direction": "in", "ytick.right": True,
             "contour.negative_linestyle": "solid", "font.family": "sans-serif", "font.serif": "Helvetica",
             "font.size": 10, "text.usetex": False, "mathtext.fontset": "cm", "mathtext.default": "regular",
             "errorbar.capsize": 0}
    lsize = 11
    color = color2 = "#B22222"
    rad, r2 = rings["rad"], rings2
    max_vrot, max_vdisp = np.nanmax(rings["vrot"]), np.nanmax(rings["disp"])
    max_rad = 1.1 * np.nanmax(rad)
    if r2 is not None:
        color = "#A0A0A0"
        max_rad = 1.1 * np.nanmax(r2["rad"])
        max_vrot, max_vdisp = np.maximum(max_vrot, np.nanmax(r2["vrot"])), np.maximum(max_vdisp, np.nanmax(r2["disp"]))

    with mpl.rc_context(style):
        # Defining figure and axes
        fig = plt.figure(FigureClass=StyledFigure, style=style, figsize=(11, 11), dpi=150)
        nrows, ncols = 3, 3
        xlen, ylen = 0.27, 0.13
        x_sep, y_sep = 0.07, 0.015
        bottom_corner = [0.1, 0.7]
        for i in range(nrows):
            bottom_corner[0], yl = 0.1, ylen
            if i == 0:
                yl *= 1.8
            for j in range(ncols):
                fig.add_axes([bottom_corner[0], bottom_corner[1], xlen, yl])
                fig.axes[-1].set_xlim(0, max_rad)
                fig.axes[-1].tick_params(labelbottom=(i == nrows - 1))
                bottom_corner[0] += xlen + x_sep
            bottom_corner[1] -= (ylen + y_sep)
        ax = fig.axes

        # Rotation velocity
        ax[0].set_ylim(0, 1.2 * max_vrot)
        ax[0].set_ylabel(r"V$_\mathrm{rot}$ (km/s)", fontsize=lsize)
        ax[0].errorbar(rad, rings["vrot"], yerr=ring_errors(rings, "vrot"), fmt="o", color=color)
        if r2 is not None:
            ax[0].errorbar(r2["rad"], r2["vrot"], yerr=ring_errors(r2, "vrot"), fmt="o", color=color2)

        # Velocity dispersion
        ax[1].set_ylim(0, 1.2 * max_vdisp)
        ax[1].set_ylabel(r"$\sigma_\mathrm{gas}$  (km/s)", fontsize=lsize)
        ax[1].errorbar(rad, rings["disp"], yerr=ring_errors(rings, "disp"), fmt="o", color=color)
        if r2 is not None:
            ax[1].errorbar(r2["rad"], r2["disp"], yerr=ring_errors(r2, "disp"), fmt="o", color=color2)

        # Surface density
        ax[2].set_xlim(0, max_rad)
        if faceon:
            ax[2].set_ylabel(r"$\Sigma$ face-on (" + bunit + " arcsec$^{-2}$)", fontsize=lsize - 1)
        else:
            ax[2].set_ylabel(r"$\Sigma$ (" + bunit + ")", fontsize=lsize)
        if sdens_obs is not None:
            ax[2].plot(sdens_obs[0], sdens_obs[1], "o", mfc="none", color="#A0A0A0", label="observed")
        if sdens is not None:
            ax[2].errorbar(sdens[0], sdens[1], yerr=sdens[2], fmt="o", color=color2, label=sdens_label)
        if sdens_label is not None:
            ax[2].legend(fontsize=8, frameon=False)

        # Inclination angle
        ax[3].set_ylabel("i (deg)", fontsize=lsize)
        ax[3].errorbar(rad, rings["inc"], yerr=ring_errors(rings, "inc"), fmt="o", color=color)
        if r2 is not None:
            ax[3].errorbar(r2["rad"], r2["inc"], yerr=ring_errors(r2, "inc"), fmt="o-", color=color2)

        # x-centre
        ax[4].set_ylabel(r"x$_0$ (pix)", fontsize=lsize)
        ax[4].errorbar(rad, rings["xpos"], yerr=ring_errors(rings, "xpos"), fmt="o", color=color)
        if r2 is not None:
            ax[4].errorbar(r2["rad"], r2["xpos"], yerr=ring_errors(r2, "xpos"), fmt="o-", color=color2)

        # Radial velocity
        ax[5].set_xlim(0, max_rad)
        ax[5].set_ylabel(r"V$_\mathrm{rad}$ (km/s)", fontsize=lsize)
        ax[5].errorbar(rad, rings["vrad"], yerr=ring_errors(rings, "vrad"), fmt="o", color=color)
        if r2 is not None:
            ax[5].errorbar(r2["rad"], r2["vrad"], yerr=ring_errors(r2, "vrad"), fmt="o", color=color2)

        # Position angle
        ax[6].set_ylabel(r"$\phi$ (deg)", fontsize=lsize)
        ax[6].set_xlabel("Radius (arcsec)", fontsize=lsize, labelpad=10)
        ax[6].errorbar(rad, rings["pa"], yerr=ring_errors(rings, "pa"), fmt="o", color=color)
        if r2 is not None:
            ax[6].errorbar(r2["rad"], r2["pa"], yerr=ring_errors(r2, "pa"), fmt="o-", color=color2)

        # y-centre
        ax[7].set_ylabel(r"y$_0$ (pix)", fontsize=lsize)
        ax[7].set_xlabel("Radius (arcsec)", fontsize=lsize, labelpad=10)
        ax[7].errorbar(rad, rings["ypos"], yerr=ring_errors(rings, "ypos"), fmt="o", color=color)
        if r2 is not None:
            ax[7].errorbar(r2["rad"], r2["ypos"], yerr=ring_errors(r2, "ypos"), fmt="o-", color=color2)

        # Systemic velocity
        ax[8].set_ylabel(r"v$_\mathrm{sys}$ (km/s)", fontsize=lsize)
        ax[8].set_xlabel("Radius (arcsec)", fontsize=lsize, labelpad=10)
        ax[8].errorbar(rad, rings["vsys"], yerr=ring_errors(rings, "vsys"), fmt="o", color=color)
        if r2 is not None:
            ax[8].errorbar(r2["rad"], r2["vsys"], yerr=ring_errors(r2, "vsys"), fmt="o", color=color2)

        # Beyond the outer edge of the disk (RADMAX) the model has no gas: the rings there are extrapolations
        if radmax is not None:
            for a in ax:
                a.axvspan(radmax, max(max_rad, radmax), color="0.9", zorder=0, lw=0)
                a.axvline(radmax, ls="--", lw=1, color="0.5", zorder=0)
            ax[0].annotate(" RADMAX", xy=(radmax, 0.97), xycoords=("data", "axes fraction"), ha="left", va="top",
                           fontsize=9, color="0.4")
    return fig


def plot_asymdrift(rings, rad, vcirc, va2, disp2r, fun, funr):
    """The asymmetric drift correction of the rotation curve (ADRIFT, <prefix>_asymmetricdrift.pdf): VROT, V_A and
    the circular velocity; the velocity dispersion and its regularised profile; the function Sigma*cos(i) used and
    its regularised profile. rings: the final rings (as read_rings); rad, vcirc, va2 (V_A^2), disp2r (sigma^2
    regularised), fun, funr: the columns 0, 1, 2, 4, 5, 6 of <prefix>_asymdrift.txt. Returns the Figure."""
    style = {"xtick.direction": "in", "xtick.top": True, "ytick.direction": "in", "ytick.right": True,
             "contour.negative_linestyle": "solid", "font.family": "sans-serif", "font.serif": "Helvetica",
             "font.size": 10, "text.usetex": False, "mathtext.fontset": "cm", "mathtext.default": "regular",
             "errorbar.capsize": 0}
    color2 = "#B22222"
    max_rad = 1.1 * np.nanmax(rings["rad"])
    with mpl.rc_context(style):
        fig = plt.figure(FigureClass=StyledFigure, style=style, figsize=(10, 10), dpi=150)
        ax1 = fig.add_axes([0.10, 0.1, 0.3, 0.25])
        ax2 = fig.add_axes([0.48, 0.1, 0.3, 0.25])
        ax3 = fig.add_axes([0.86, 0.1, 0.3, 0.25])
        ax1.set_xlim(0, max_rad)
        ax1.set_xlabel("Radius (arcsec)", fontsize=11, labelpad=5)
        ax1.set_ylabel("V (km/s)", fontsize=11)
        ax1.plot(rings["rad"], rings["vrot"], "-", color=color2, label=r"V$_\mathrm{rot}$", zorder=1)
        ax1.plot(rad, np.sqrt(va2), "--", color="#FABC11", label=r"V$_\mathrm{A}$", zorder=2)
        ax1.errorbar(rad, vcirc, yerr=ring_errors(rings, "vrot"), fmt="o", color="#43A8D4",
                     label=r"V$_\mathrm{circ}$", zorder=0)
        ax1.legend()
        ax2.set_xlim(0, max_rad)
        ax2.set_xlabel("Radius (arcsec)", fontsize=11, labelpad=5)
        ax2.set_ylabel(r"$\sigma$ (km/s)", fontsize=11)
        ax2.errorbar(rings["rad"], rings["disp"], yerr=ring_errors(rings, "disp"), fmt="o", color=color2,
                     label=r"$\sigma_\mathrm{gas}$", zorder=0)
        ax2.plot(rad, np.sqrt(disp2r), "-", color="#0FA45A", label=r"$\sigma_\mathrm{reg}$", zorder=1)
        ax2.legend()
        ax3.set_xlim(0, max_rad)
        ax3.set_xlabel("Radius (arcsec)", fontsize=11, labelpad=5)
        ax3.set_ylabel(r"$\Sigma_\mathrm{gas}\cos(i)$", fontsize=11)
        ax3.plot(rad, fun, "o", color=color2, label=r"$f_\mathrm{obs}$", zorder=0)
        ax3.plot(rad, funr, "-", color="#0FA45A", label=r"$f_\mathrm{reg}$", zorder=1)
        ax3.legend()
    return fig


def plot_chanmaps(data, model, rings, mask=None, *, box, cont, pixscale, cdelt3, crval3, beam, gname="", bunit="",
                  plotmask=False):
    """Channel maps of the data (grey scale, blue contours) and of a model (red contours) below them: 15 channels
    evenly spaced in the box (<prefix>_chanmaps_<norm>.pdf).

    data, model: cubes (nz, ny, nx); mask: the mask, a cube of the same size (its edge is drawn with plotmask);
    rings: the final rings (as read_rings: the cross marks their mean centre; the velocities are relative to their
    mean VSYS); box: (xmin, xmax, ymin, ymax, zmin, zmax) the region shown (pixels and channels of the cubes,
    inclusive); cont: the lowest contour (units of the data; the others at 2^n cont); pixscale: the size of the
    pixels (arcsec); cdelt3, crval3: the width of the channels and the velocity of channel 0 of the cubes (km/s);
    beam: (major, minor axis in pixels, position angle in degrees); gname: the name of the galaxy; bunit: the
    units of the data. Returns the Figure."""
    style = {"xtick.direction": "in", "xtick.top": True, "xtick.labelbottom": False, "ytick.direction": "in",
             "ytick.right": True, "ytick.labelleft": False, "contour.negative_linestyle": "solid",
             "font.family": "sans-serif", "font.serif": "Helvetica", "font.size": 10, "text.usetex": False,
             "mathtext.fontset": "cm", "mathtext.default": "regular"}
    xmin, xmax, ymin, ymax, zmin, zmax = box
    xcen_m, ycen_m, vsys_m = np.nanmean((rings["xpos"], rings["ypos"], rings["vsys"]), axis=1)
    data = data[zmin:zmax + 1, ymin:ymax + 1, xmin:xmax + 1]
    model = model[zmin:zmax + 1, ymin:ymax + 1, xmin:xmax + 1]
    if mask is not None:
        mask = mask[zmin:zmax + 1, ymin:ymax + 1, xmin:xmax + 1]
    zsize, ysize, xsize = data.shape
    pad_x, pad_y = calculate_padding(xsize, ysize)
    v = np.array([1, 2, 4, 8, 16, 32, 64]) * cont
    v_neg = [-cont]
    interval = PercentileInterval(99.5)
    vmax = interval.get_limits(data)[1]
    norm = ImageNormalize(vmin=cont, vmax=vmax, stretch=PowerStretch(0.5))
    xcen, ycen = xcen_m - xmin + pad_x[0], ycen_m - ymin + pad_y[0]

    with mpl.rc_context(style):
        fig = plt.figure(FigureClass=StyledFigure, style=style, figsize=(8.27, 11.69), dpi=150)
        grid = [gridspec.GridSpec(2, 5), gridspec.GridSpec(2, 5), gridspec.GridSpec(2, 5)]
        grid[0].update(top=0.90, bottom=0.645, left=0.05, right=0.95, wspace=0.0, hspace=0.0)
        grid[1].update(top=0.60, bottom=0.345, left=0.05, right=0.95, wspace=0.0, hspace=0.0)
        grid[2].update(top=0.30, bottom=0.045, left=0.05, right=0.95, wspace=0.0, hspace=0.0)

        num = 0
        for j in range(0, 3):
            for i in range(0, 5):
                chan = int(num * (zsize) / 15)
                z = np.pad(data[chan, :, :], (pad_y, pad_x), mode="constant", constant_values=np.nan)
                z_mod = np.pad(model[chan, :, :], (pad_y, pad_x), mode="constant", constant_values=np.nan)
                # New matplotlib draws wrong contours when no contours are found. This is a workaround.
                if np.all(z_mod < v[0]):
                    z_mod[:, :] = 0
                velo_kms = (chan + 1 - (1 - zmin)) * cdelt3 + crval3
                velo = " v = " + str(int(velo_kms - vsys_m)) + " km/s"
                ax = fig.add_subplot(grid[j][0, i])
                ax.set_title(velo, fontsize=10, loc="left")
                ax.imshow(z, origin="lower", cmap=mpl.cm.Greys, norm=norm, aspect="auto", interpolation=None)
                ax.contour(z, v, origin="lower", linewidths=0.7, colors="#00008B")
                ax.contour(z, v_neg, origin="lower", linewidths=0.1, colors="gray")
                ax.plot(xcen, ycen, "x", color="#0FB05A", markersize=7, mew=2)
                if plotmask and mask is not None:
                    z_mas = np.pad(mask[chan], (pad_y, pad_x), mode="constant", constant_values=0)
                    ax.contour(z_mas, levels=[0], origin="lower", linewidths=2, colors="k")
                if j == i == 0:
                    ax.text(0, 1.4, gname, transform=ax.transAxes, fontsize=15, va="center")
                    lbar = 0.5 * (xmax - xmin) * pixscale
                    ltex = "%.0f'' " % lbar if lbar > 10 else "%.2f'' " % lbar
                    if lbar > 600:
                        ltex = "%.0f' " % (lbar / 60.)
                    ax.annotate("", xy=(4.5, 1.4), xycoords="axes fraction", xytext=(5, 1.4),
                                arrowprops=dict(arrowstyle="<->", color="k"))
                    ax.text(4.75, 1.50, ltex, transform=ax.transAxes, fontsize=11, ha="center")
                    bmaj, bmin, bpa = beam[0] / float(xmax - xmin), beam[1] / float(ymax - ymin), beam[2]
                    ell = mpl.patches.Ellipse(xy=(3.5, 1.4), width=bmaj, height=bmin, angle=bpa + 90,
                                              color="#5605D0", clip_on=False, transform=ax.transAxes, alpha=0.2)
                    ax.add_artist(ell)
                    ax.text(3.6 + bmaj / 1.8, 1.4, "Beam", transform=ax.transAxes, fontsize=11, ha="left",
                            va="center")
                ax = fig.add_subplot(grid[j][1, i])
                ax.tick_params(axis="both", which="both", bottom=True, top=True, labelbottom=False,
                               labelleft=False)
                ax.imshow(z_mod, origin="lower", cmap=mpl.cm.Greys, norm=norm, aspect="auto", interpolation=None)
                ax.contour(z_mod, v, origin="lower", linewidths=0.7, colors="#B22222")
                ax.plot(xcen, ycen, "x", color="#0FB05A", markersize=7, mew=2)
                if i == 0 and j == 2:
                    clab = (r"Contour levels at 2$^n \, c_{min}$, where $c_{min}$ = %g " + bunit
                            + " and n = 0,1,..,8 ") % cont
                    ax.text(0.01, -0.16, clab, transform=ax.transAxes, fontsize=11, ha="left", va="center")
                num = num + 1
    return fig


def plot_pvs(data, model, rings, mom0, mom1, mask=None, *, box, cont, pixscale, vrange, gname="", plotmask=False):
    """Position-velocity slices of the data (grey scale, blue contours) and of a model (red contours) along the
    major axis (top; the gold points are the rotation curve of the rings, projected) and the minor axis (bottom, on
    the offset range of the major axis), following the position angles of the rings, beside the total intensity and velocity maps of the data with the
    paths of the slices (<prefix>_pv_<norm>.pdf).

    data, model, mask, rings, box, cont, pixscale, gname, plotmask: as for plot_chanmaps; mom0, mom1: the total
    intensity and velocity maps of the data (ny, nx); vrange: the velocities (km/s) of the outer edges of the
    channels zmin and zmax of the box. Returns the Figure."""
    style = {"xtick.direction": "in", "xtick.top": True, "ytick.direction": "in", "ytick.right": True,
             "contour.negative_linestyle": "solid", "font.family": "sans-serif", "font.serif": "Helvetica",
             "font.size": 18, "text.usetex": False, "mathtext.fontset": "cm", "mathtext.default": "regular"}
    labsize = 18
    xmin, xmax, ymin, ymax, zmin, zmax = box
    zmin_wcs, zmax_wcs = vrange
    cdeltsp = pixscale
    data = data[zmin:zmax + 1, ymin:ymax + 1, xmin:xmax + 1]
    data_mod = model[zmin:zmax + 1, ymin:ymax + 1, xmin:xmax + 1]
    data_mas = mask[zmin:zmax + 1, ymin:ymax + 1, xmin:xmax + 1] if mask is not None else None
    zsize, ysize, xsize = data.shape
    pad_x, pad_y = calculate_padding(xsize, ysize)
    v = np.array([1, 2, 4, 8, 16, 32, 64]) * cont
    v_neg = [-cont]
    mom0 = mom0[ymin:ymax + 1, xmin:xmax + 1]
    mom1 = mom1[ymin:ymax + 1, xmin:xmax + 1]
    mom0p = np.pad(mom0, (pad_y, pad_x), mode="constant", constant_values=np.nan)
    mom1p = np.pad(mom1, (pad_y, pad_x), mode="constant", constant_values=np.nan)
    extm = [0, mom0p.shape[1] - 1, 0, mom0p.shape[0] - 1]
    norm0 = ImageNormalize(mom0, interval=PercentileInterval(98.0), stretch=PowerStretch(0.5))
    norm1 = ImageNormalize(mom1, interval=PercentileInterval(99.5))

    rad, vrot, inc, pa = rings["rad"], rings["vrot"], rings["inc"], rings["pa"]
    xcen_m, ycen_m, vsys_m, inc_m, pa_m = np.nanmean((rings["xpos"], rings["ypos"], rings["vsys"], inc, pa), axis=1)
    xcen_m, ycen_m = xcen_m - xmin, ycen_m - ymin
    radii = np.concatenate((rad, -rad))
    vlos1 = vrot * np.sin(np.deg2rad(inc))
    vlos = np.concatenate((vlos1, -vlos1))

    def plot_pv(ax, rads, x0, y0, theta):
        s, x, y = build_path(x0, y0, rads / cdeltsp, theta, xsize, ysize)
        pv_data = extract_pv(data, x, y, order=0)
        pv_mod = extract_pv(data_mod, x, y)
        norm = ImageNormalize(vmin=cont, vmax=np.percentile(data, 99.8), stretch=PowerStretch(0.5))

        s_arcsec = s * cdeltsp
        ext = [s_arcsec[0], s_arcsec[-1], zmin_wcs - vsys_m, zmax_wcs - vsys_m]

        ax.imshow(pv_data, origin="lower", cmap="Greys", norm=norm, aspect="auto", interpolation="nearest", extent=ext)
        ax.contour(pv_data, v, origin="lower", linewidths=1.0, colors="#00008B", extent=ext)
        ax.contour(pv_data, v_neg, origin="lower", linewidths=0.2, colors="gray", extent=ext)
        ax.contour(pv_mod, v, origin="lower", linewidths=1.5, colors="#B22222", extent=ext)
        ax.axhline(y=0, color="black")
        ax.axvline(x=0, color="black")
        ax.grid(color="gray", linestyle="--", linewidth=0.3)

        if plotmask and data_mas is not None:
            pv_mask = extract_pv(data_mas, x, y)
            ax.contour(pv_mask, levels=[0], origin="lower", linewidths=3.5, colors="k", extent=ext)

        # x range: arc length along the (curved) slice out to the outermost ring, either side
        ir = np.argmax(rads)
        rp, th = rads[ir] / cdeltsp, np.radians(np.broadcast_to(theta, np.shape(rads))[ir])
        slen = rads[ir]
        if len(s) > 0:
            slen = cdeltsp * max(abs(s[np.argmin((x - x0 + sg * rp * np.sin(th)) ** 2
                                                 + (y - y0 - sg * rp * np.cos(th)) ** 2)]) for sg in (1, -1))
        ax.set_xlim(-1.3 * slen, 1.3 * slen)
        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=6))
        ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=6))
        ax.tick_params(labelbottom=True, labelleft=True)
        ax.tick_params(which="major", length=8, labelsize=labsize)
        ax.set_ylabel(r"$\mathrm{\Delta V_{LOS}}$ (km/s)", fontsize=labsize + 2)
        ax.set_xlabel("Offset (arcsec)", fontsize=labsize + 2)

        ax2 = ax.twinx()
        ax2.set_ylim([ext[2] + vsys_m, ext[3] + vsys_m])
        ax2.tick_params(which="major", length=8, labelsize=labsize)
        ax2.set_ylabel(r"$\mathrm{V_{LOS}}$ (km/s)", fontsize=labsize + 2)
        return s, x, y, slen

    with mpl.rc_context(style):
        fig = plt.figure(FigureClass=StyledFigure, style=style, figsize=(9, 9))
        fig, axes = defineaxis(2, 1, 1, 0.7, xsep=0, ysep=0.1, fig=fig)
        axes = np.ravel(axes)

        # Major axis
        s, x, y, _ = plot_pv(axes[0], rad, xcen_m, ycen_m, pa)
        axes[0].text(0, 1.05, gname, transform=axes[0].transAxes, fontsize=32)

        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        bbox = axes[0].get_tightbbox(renderer).transformed(fig.transFigure.inverted())
        pos = axes[0].get_position()
        ax_m = fig.add_axes([bbox.x1 + 0.14, pos.y1 - 0.4, 0.4, 0.4])

        ax_m.tick_params(labelbottom=False, labelleft=False, top=False, bottom=False, left=False, right=False)
        ax_m.imshow(mom0p, origin="lower", cmap="Greys", aspect="auto", interpolation="nearest", norm=norm0,
                    extent=extm)
        ax_m.imshow(mom1p, origin="lower", cmap="seismic", aspect="auto", interpolation="nearest", norm=norm1,
                    extent=extm, alpha=0.4)
        ax_m.contour(mom1p, levels=[vsys_m], extent=extm, colors="forestgreen", linewidths=3)
        ax_m.plot(xcen_m + pad_x[0], ycen_m + pad_y[0], "x", color="y", markersize=7, mew=1.5, zorder=10)
        ax_m.annotate("", xy=(0.8, -0.1), xycoords="axes fraction", xytext=(1.0, -0.1),
                      arrowprops=dict(arrowstyle="<->", color="k", lw=1.5))
        ax_m.text(0.9, -0.16, f"{0.2 * xsize * cdeltsp:.0f}''", transform=ax_m.transAxes, fontsize=12, ha="center")
        ax_m.axvline(xcen_m + pad_x[0], ls="dotted", c="k", alpha=0.2)
        ax_m.axhline(ycen_m + pad_y[1], ls="dotted", c="k", alpha=0.2)
        _, ax_m_xmax = ax_m.get_xlim()
        _, ax_m_ymax = ax_m.get_ylim()
        m = (x >= 0) & (x < ax_m_xmax) & (y >= 0) & (y < ax_m_ymax)
        ax_m.plot(x[m] + pad_x[0], y[m] + pad_y[0], "-", lw=1.5, c="#B22222")
        ax_m.plot(x[~m] + pad_x[0], y[~m] + pad_y[0], "-", lw=1.5, c="#B22222", alpha=0.3)

        # Rotation curve
        r = np.sqrt((x - xcen_m) ** 2 + (y - ycen_m) ** 2)
        r[s < 0] *= -1
        x_rad = np.interp(radii / cdeltsp, r, x)
        y_rad = np.interp(radii / cdeltsp, r, y)
        s_rad = np.interp(radii / cdeltsp, r, s)
        axes[0].plot(s_rad * cdeltsp, vlos, "o", c="gold", ms=10, alpha=0.8, mec="cornsilk")
        ax_m.scatter(x_rad + pad_x[0], y_rad + pad_y[0], c="gold", s=10, alpha=0.8)

        # Minor axis
        s, x, y, _ = plot_pv(axes[1], rad * np.cos(np.radians(inc)), xcen_m, ycen_m, pa + 90)
        m = (x >= 0) & (x < ax_m_xmax) & (y >= 0) & (y < ax_m_ymax)
        ax_m.plot(x[m] + pad_x[0], y[m] + pad_y[0], "--", lw=1.5, c="navy")
        ax_m.plot(x[~m] + pad_x[0], y[~m] + pad_y[0], "--", lw=1.5, c="navy", alpha=0.3)

        # the same offset range as the major axis, to compare the two slices on one scale
        axes[1].set_xlim(axes[0].get_xlim())
    return fig


def plot_pvslices(data, model, rings, mom0, mask=None, *, box, cont, pixscale, vrange, gname="", plotmask=False):
    """Position-velocity slices of the data (blue scale and contours) and of a model (red contours): through the
    centre at 10 angles from the major axis, 18 degrees apart (top), and parallel to the major and to the minor axis
    at 5 offsets each (bottom), with the total intensity map of the data and the paths of the slices
    (<prefix>_pvslices_<norm>.pdf). The arguments are those of plot_pvs (mom0: the total intensity map of the
    data). Returns the Figure."""
    style = {"xtick.direction": "in", "xtick.top": True, "ytick.direction": "in", "ytick.right": True,
             "contour.negative_linestyle": "solid", "font.family": "sans-serif", "font.serif": "Helvetica",
             "font.size": 13, "text.usetex": False, "mathtext.fontset": "cm", "mathtext.default": "regular"}
    labsize = 13
    xmin, xmax, ymin, ymax, zmin, zmax = box
    zmin_wcs, zmax_wcs = vrange
    cdeltsp = pixscale
    data = data[zmin:zmax + 1, ymin:ymax + 1, xmin:xmax + 1]
    data_mod = model[zmin:zmax + 1, ymin:ymax + 1, xmin:xmax + 1]
    data_mas = mask[zmin:zmax + 1, ymin:ymax + 1, xmin:xmax + 1] if mask is not None else None
    zsize, ysize, xsize = data.shape
    pad_x, pad_y = calculate_padding(xsize, ysize)
    v = np.array([1, 2, 4, 8, 16, 32, 64]) * cont
    v_neg = [-cont]
    mom0 = mom0[ymin:ymax + 1, xmin:xmax + 1]
    mom0p = np.pad(mom0, (pad_y, pad_x), mode="constant", constant_values=np.nan)
    extm = [0, mom0p.shape[1] - 1, 0, mom0p.shape[0] - 1]
    norm = ImageNormalize(data, interval=PercentileInterval(99.9), stretch=PowerStretch(1.0))
    norm0 = ImageNormalize(mom0, interval=PercentileInterval(99.0), stretch=PowerStretch(0.5))

    rad = rings["rad"]
    xcen_m, ycen_m, vsys_m, inc_m, pa_m = np.nanmean((rings["xpos"], rings["ypos"], rings["vsys"], rings["inc"],
                                                      rings["pa"]), axis=1)
    xcen_m, ycen_m = xcen_m - xmin, ycen_m - ymin

    s_maj, x_maj, y_maj = build_path(xcen_m, ycen_m, rad / cdeltsp, np.full(len(rad), pa_m), xsize, ysize)
    s_min, x_min, y_min = build_path(xcen_m, ycen_m, rad / cdeltsp, np.full(len(rad), pa_m + 90), xsize, ysize)
    s_maj = s_maj * cdeltsp
    s_min = s_min * cdeltsp
    maj_offsets = np.linspace(0, rad[-1], 3, endpoint=True)
    min_offsets = np.linspace(0, rad[-1] * np.cos(np.radians(inc_m)), 3, endpoint=True)
    maj_offsets = np.concatenate([-maj_offsets[1:][::-1], maj_offsets])
    min_offsets = np.concatenate([-min_offsets[1:][::-1], min_offsets])
    x_maj_s = np.interp(min_offsets, s_min, x_min)
    y_maj_s = np.interp(min_offsets, s_min, y_min)
    x_min_s = np.interp(maj_offsets, s_maj, x_maj)
    y_min_s = np.interp(maj_offsets, s_maj, y_maj)

    cmap = plt.cm.PuBu_r
    colors = cmap(np.linspace(0, 1, 13))

    def plot_pv(i, ax, ax_m, x0, y0, theta, **kwargs):
        s, x, y = build_path(x0, y0, rad / cdeltsp, np.full(len(rad), theta), xsize, ysize)
        pv_data = extract_pv(data, x, y)
        pv_mod = extract_pv(data_mod, x, y)
        s_arcsec = s * cdeltsp
        ext = [s_arcsec[0], s_arcsec[-1], zmin_wcs - vsys_m, zmax_wcs - vsys_m]

        ax.imshow(pv_data, origin="lower", cmap="Blues", norm=norm, aspect="auto", interpolation="nearest", extent=ext)
        ax.contour(pv_data, v, origin="lower", linewidths=0.7, colors="#00008B", extent=ext)
        ax.contour(pv_data, v_neg, origin="lower", linewidths=0.1, colors="gray", extent=ext)
        ax.contour(pv_mod, v, origin="lower", linewidths=1.2, colors="#B22222", extent=ext)
        ax.axhline(y=0, color="black")
        ax.axvline(x=0, color="black")
        ax.grid(color="gray", linestyle="--", linewidth=0.3)

        if plotmask and data_mas is not None:
            pv_mask = extract_pv(data_mas, x, y)
            ax.contour(pv_mask, levels=[0], origin="lower", linewidths=1.5, colors="k", extent=ext)

        ax.set_xlim(-1.3 * rad[-1], 1.3 * rad[-1])
        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=5))
        ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=5))
        if i == 0 or i == 5:
            ax.tick_params(labelbottom=True, labelleft=True)
            ax.set_ylabel(r"$\mathrm{\Delta V_{LOS}}$ (km/s)", fontsize=labsize + 2)
        else:
            ax.tick_params(labelbottom=True, labelleft=False)
        ax.set_xlabel("Offset (arcsec)", fontsize=labsize)
        ax_m.plot(x + pad_x[0], y + pad_y[0], **kwargs)

    with mpl.rc_context(style):
        fig = plt.figure(FigureClass=StyledFigure, style=style, figsize=(10, 10))
        fig, axes = defineaxis(4, 5, 0.3, 0.3, xsep=0.01, ysep=0.1, fig=fig)
        axes = np.ravel(axes)

        pos = axes[4].get_position()
        ax_m1 = fig.add_axes([pos.x1 + 0.03, pos.y0 - 0.2, 0.3, 0.3])
        pos = axes[14].get_position()
        ax_m2 = fig.add_axes([pos.x1 + 0.03, pos.y0 - 0.2, 0.3, 0.3])

        # Total intensity map
        for ax in [ax_m1, ax_m2]:
            ax.set_prop_cycle(color=colors)
            ax.tick_params(labelbottom=False, labelleft=False, top=False, bottom=False, left=False, right=False)
            ax.imshow(mom0p, origin="lower", cmap="Greys", aspect="auto", interpolation="nearest", norm=norm0,
                      extent=extm)
            ax.set_title("Slices")
            ax.plot(xcen_m + pad_x[0], ycen_m + pad_y[0], "x", color="yellow", markersize=7, mew=1.5, zorder=12)
            ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=5))
            ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=5))

        pos = ax_m2.get_position()
        ax_m2.set_position([pos.x0, pos.y0 - 0.07, pos.width, pos.height])

        # Slices through the centre
        dtheta = 18
        for i, ax in enumerate(axes[0:10]):
            theta = pa_m + i * dtheta
            ls = "solid" if i < 5 else "dashed"
            plot_pv(i, ax, ax_m1, xcen_m, ycen_m, theta, ls=ls, label=rf"{i * dtheta:.0f}$^\circ$")
            ax.text(0.03, 1.03, rf"$\theta$ = {i * dtheta:.0f}$^\circ$", transform=ax.transAxes, fontsize=labsize)

        # Slices parallel to the major and minor axes
        for i, ax in enumerate(axes[10:]):
            pos = ax.get_position()
            ax.set_position([pos.x0, pos.y0 - 0.05, pos.width, pos.height])
            if i < 5:
                xi, yi, off, theta = x_maj_s[i], y_maj_s[i], min_offsets[i], pa_m
                ls = "solid"
            else:
                xi, yi, off, theta = x_min_s[i - 5], y_min_s[i - 5], maj_offsets[i - 5], pa_m + 90
                ls = "dashed"
            plot_pv(i, ax, ax_m2, xi, yi, theta, ls=ls, label=rf"{off:.1f}''")
            ax.text(0.03, 1.03, rf"offset = {off:.1f}''", transform=ax.transAxes, fontsize=labsize)
            ax_m2.plot(xi + pad_x[0], yi + pad_y[0], "o", c="royalblue", zorder=11)
        leg_opts = dict(ncol=2, loc="upper left", bbox_to_anchor=(0.0, -0.05), frameon=False)
        ax_m1.legend(**leg_opts, handlelength=3.3)
        ax_m2.legend(**leg_opts, handlelength=2)
    return fig


def plot_kinmaps(maps, models, rings, *, box, pixscale, bunit="",
                 mapnames=("MOMENT 0TH", "MOMENT 1ST", "MOMENT 2ND")):
    """The total intensity, velocity and dispersion maps (rows) of the data, of a model and of the residuals
    (columns), with the major axis, the outer ring and the position angles of the rings
    (<prefix>_maps_<norm>.pdf).

    maps, models: the three maps (mom0, mom1, mom2: arrays ny x nx) of the data and of the model; rings: the final
    rings (as read_rings: the centre, PA, INC and VSYS are their means); box: (xmin, xmax, ymin, ymax, ...) the
    region shown (pixels, inclusive); pixscale: the size of the pixels (arcsec); bunit: the units of the total
    intensity map; mapnames: the labels of the rows. Returns the Figure."""
    style = {"xtick.direction": "in", "ytick.direction": "in", "font.family": "sans-serif", "font.serif": "Helvetica",
             "font.size": 10, "text.usetex": False, "mathtext.fontset": "cm", "mathtext.default": "regular"}
    xmin, xmax, ymin, ymax = box[:4]
    bunit = bunit.replace(" ", "").lower()
    mom0, mom1, mom2 = (m[ymin:ymax + 1, xmin:xmax + 1] for m in maps)
    mom0_mod, mom1_mod, mom2_mod = (m[ymin:ymax + 1, xmin:xmax + 1] for m in models)
    maskmap = np.copy(mom1)
    maskmap[mom1 == mom1] = 1
    ysize, xsize = mom0.shape
    pad_x, pad_y = calculate_padding(xsize, ysize)

    rad, pa = rings["rad"], rings["pa"]
    xcen_m, ycen_m, inc_m, pa_m, vsys_m = np.nanmean((rings["xpos"], rings["ypos"], rings["inc"], pa, rings["vsys"]),
                                                     axis=1)
    xcen, ycen = xcen_m - xmin + pad_x[0], ycen_m - ymin + pad_y[0]

    cmaps = [plt.get_cmap("Spectral_r"), plt.get_cmap("coolwarm"), plt.get_cmap("PuOr_r")]
    barlab = ["Intensity (" + bunit + ")", r"V$_\mathrm{LOS}$ (km/s)", r"$\sigma_\mathrm{obs}$ (km/s)"]
    barlab2 = [r"I$_\mathrm{res}$ (" + bunit + ")", r"V$_\mathrm{res}$ (km/s)", r"$\sigma_\mathrm{res}$ (km/s)"]
    titles = ["DATA", "MODEL", "RESIDUALS"]
    x = np.arange(0, xmax - xmin, 0.1)
    y = np.tan(np.radians(pa_m - 90)) * (x - xcen) + ycen
    ext = [0, xmax - xmin + pad_x[0] + pad_x[1], 0, ymax - ymin + pad_y[0] + pad_y[1]]
    rad_pix = rad / pixscale
    nr = len(rad_pix)
    interval = PercentileInterval(99.5)
    to_plot = [[mom0, mom1 - vsys_m, mom2], [mom0_mod, mom1_mod - vsys_m, mom2_mod],
               [mom0 - mom0_mod, mom1 - mom1_mod, mom2 - mom2_mod]]

    with mpl.rc_context(style):
        nrows, ncols, x_len, y_len = 3, 3, 0.2, 0.2
        fig = plt.figure(FigureClass=StyledFigure, style=style, figsize=(11, 11))
        fig, ax = defineaxis(nrows, ncols, x_len, y_len, xsep=0.00, ysep=0.08, fig=fig)

        for i in range(ax.shape[0]):
            cmap = copy(cmaps[i])
            cmap.set_bad("w", 1.)
            vmin, vmax = interval.get_limits(to_plot[1][i])
            vmin, vmax = (-1.1 * np.nanmax(vmax), 1.1 * np.nanmax(vmax)) if i == 1 else (vmin, vmax)
            norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
            cbax = fig.add_axes([ax[i][0].get_position().x0, ax[i][0].get_position().y0 - 0.025, 2 * x_len - 0.003,
                                 0.02])
            cb1 = ColorbarBase(cbax, orientation="horizontal", cmap=cmap, norm=norm)
            cb1.set_label(barlab[i], fontsize=13)
            for j in range(ax.shape[1]):
                axis = ax[i][j]
                axis.tick_params(labelbottom=False, labelleft=False, right=True, top=True)
                axis.set_xlim(ext[0], ext[1])
                axis.set_ylim(ext[2], ext[3])
                if j == 2:
                    vmax = np.nanmax(interval.get_limits(to_plot[j][i]))
                    norm = mpl.colors.Normalize(vmin=-vmax, vmax=vmax)
                pp = np.pad(to_plot[j][i] * maskmap, (pad_y, pad_x), mode="constant", constant_values=np.nan)
                axis.imshow(pp, origin="lower", cmap=cmap, norm=norm, aspect="auto", extent=ext,
                            interpolation="nearest")
                axis.plot(xcen, ycen, "x", color="#000000", markersize=7, mew=1.5, zorder=10)

                if i == 0:
                    axis.text(0.5, 1.05, titles[j], ha="center", transform=axis.transAxes, fontsize=15)
                    axis.plot(x, y, "--", color="k", linewidth=1)
                    if j != 2 and nr > 3:
                        axmaj = rad_pix[-1]
                        axmin = axmaj * np.cos(np.radians(inc_m))
                        posa = np.radians(pa_m - 90)
                        t = np.linspace(0, 2 * np.pi, 100)
                        xt = xcen + axmaj * np.cos(posa) * np.cos(t) - axmin * np.sin(posa) * np.sin(t)
                        yt = ycen + axmaj * np.sin(posa) * np.cos(t) + axmin * np.cos(posa) * np.sin(t)
                        axis.plot(xt, yt, "-", c="k", lw=0.8)
                elif i == 1:
                    axis.plot(x, y, "--", color="k", linewidth=1)
                    if nr < 10:
                        x_pix = rad_pix * np.cos(np.radians(pa_m - 90))
                        y_pix = rad_pix * np.sin(np.radians(pa_m - 90))
                        axis.scatter(x_pix + xcen, y_pix + ycen, c="grey", s=12)
                        axis.scatter(xcen - x_pix, ycen - y_pix, c="grey", s=12)
                    if nr > 5 and not all(np.diff(pa) == 0):
                        x_pix = rad_pix * np.cos(np.radians(pa - 90))
                        y_pix = rad_pix * np.sin(np.radians(pa - 90))
                        axis.plot(xcen - x_pix, ycen - y_pix, "-", color="grey", lw=1)
                        axis.plot(x_pix + xcen, y_pix + ycen, "-", color="grey", lw=1)
                    if j != 2:
                        cmax = np.nanmax(np.abs(to_plot[0][i]))
                        levels = np.linspace(0.166 * cmax, cmax, 6)
                        axis.contour(pp, levels=[0], colors="forestgreen", origin="lower", extent=ext)
                        axis.contour(pp, levels=-levels[::-1], colors="navy", linewidths=0.7, origin="lower",
                                     extent=ext, linestyles="solid")
                        axis.contour(pp, levels=levels, colors="darkred", linewidths=0.7, origin="lower",
                                     extent=ext)
                if j == 0:
                    axis.text(-0.12, 0.5, mapnames[i], va="center", rotation=90, transform=axis.transAxes,
                              fontsize=15)

            cbax = fig.add_axes([ax[i][2].get_position().x0 + 0.003, ax[i][2].get_position().y0 - 0.025,
                                 x_len - 0.003, 0.02])
            cb2 = ColorbarBase(cbax, orientation="horizontal", cmap=cmap, norm=norm)
            cb2.set_label(barlab2[i], fontsize=13)
            cb2.ax.locator_params(nbins=3)
            for cb in [cb1, cb2]:
                cb.solids.set_edgecolor("face")
                cb.outline.set_linewidth(0)
    return fig
