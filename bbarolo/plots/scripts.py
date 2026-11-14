"""
The plotting scripts of 3DFIT (port of Galfit::writeScripts_Python of galfit_out.cpp), written in
<outfolder>/plotscripts/ and run by plotscripts.plotAll_Python.

Each script is made of: the imports; a copy of the source of its plotting function and of the readers of fit3d.py
(so it runs without bbarolo, with plot_utils.py, the copy of utils.py); the settings of the run (values computed
by plot_settings); and the code that reads the output files, calls the function and saves the PDF.
"""

import inspect

import numpy as np

from ..stats import findMedian
from ..header import fits_image_hdu
from ..bbutils import DeltaVel, AlltoVel, FluxtoJy, Vel2Spec, mkdirp
from . import fit3d


class Code(str):
    """A value of a setting already written as Python code."""


def source(*objs):
    """The source code of functions and classes, one after the other."""
    return "\n\n".join(inspect.getsource(o).rstrip() + "\n" for o in objs)


def lit(v):
    """A value of a setting as Python code: floats with 6 significant digits, as written by C++ BBarolo."""
    from ..plotscripts import F
    if isinstance(v, Code):
        return str(v)
    if v is None or isinstance(v, (bool, np.bool_)):
        return repr(None if v is None else bool(v))
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return "np.nan" if np.isnan(v) else F(v) if np.isfinite(v) else ("np.inf" if v > 0 else "-np.inf")
    if isinstance(v, str):
        return repr(v)
    if isinstance(v, tuple):
        return "(" + ", ".join(lit(x) for x in v) + ("," if len(v) == 1 else "") + ")"
    if isinstance(v, list):
        return "[" + ", ".join(lit(x) for x in v) + "]"
    if isinstance(v, dict):
        return "{" + ", ".join(lit(k) + ": " + lit(x) for k, x in v.items()) + "}"
    if isinstance(v, np.ndarray):
        return "np.array([" + ",".join("%.6e" % float(x) for x in np.ravel(v)) + "])"
    raise TypeError(f"no literal for {type(v).__name__}")


def script(title, funcs, settings, body):
    """The text of a plotting script: a banner with the title, the imports (fit3d._IMPORTS), the source of funcs,
    the settings ((name, value, comment) with comment possibly empty) and the body (the code of the run)."""
    bar = "#" * (len(title) + 10)
    out = [f"{bar}\n#### {title} ####\n{bar}\n",
           "# Written by BBarolo. The functions below draw the plots (the same as bbarolo.plots); the settings of\n"
           "# the run and the code that reads the files and saves the plots are at the bottom. Edit any of them and\n"
           "# run the script again from the folder where BBarolo was run (cubefile may be relative to it). It needs\n"
           "# numpy, matplotlib, astropy, scipy and plot_utils.py (in the same folder), not bbarolo.\n",
           fit3d._IMPORTS, "\n\n", source(*funcs), "\n\n# Settings of this run\n"]
    for name, value, comment in settings:
        line = f"{name} = {lit(value)}"
        out.append(line + (" " * max(1, 40 - len(line)) + "# " + comment if comment else "") + "\n")
    out.append("\n" + body.strip("\n") + "\n")
    return "".join(out)


def plot_utils_text():
    """The source of plot_utils.py, the helpers of the plotting scripts (3DFIT, WINDFIT)."""
    from .. import pvslice
    from . import utils
    return ("###############################################################\n"
            "#### This script contains some utilities for plotting      ####\n"
            "###############################################################\n"
            "import math\nimport numpy as np\nimport matplotlib as mpl\nimport matplotlib.pyplot as plt\n"
            "from matplotlib.figure import Figure\n\n\n"
            + source(utils.StyledFigure, utils.defineaxis, utils.calculate_padding) + "\n"
            + "# PV slices: copy of bbarolo/pvslice.py (pv_path, pv_sample)\n"
            + source(pvslice._line_range, pvslice.pv_path, pvslice.pv_sample, utils.build_path, utils.extract_pv))


def _lround(x):
    return int(np.floor(x + 0.5)) if x >= 0 else -int(np.floor(-x + 0.5))


def plot_settings(gf):
    """The settings of the plots of the fit gf (a Galfit after the fit, also that of METHOD GLOBAL or BAYESIAN), as a dict:
    gname, outfolder, outprefix, twostage, plotmask; box: (xmin, xmax, ymin, ymax, zmin, zmax) the region of the
    cube shown (pixels and channels, inclusive: around the largest source of the SEARCH mask, the mask of
    SMOOTH&SEARCH, or the rings and their velocities); cont: the lowest contour; pixscale (arcsec), cdelt3,
    crval3 (km/s, channel width and velocity of channel 0), beam (pixels, pixels, degrees), vrange (km/s, the
    velocities of the outer edges of the channels of the box); bunit (of the maps) and head_bunit (of the cube),
    mapnames; errcols1, errcols2: the columns of the errors in the ring files of the two stages; dens: the
    density panel ('plain', 'deconvolved' (3DFIT) or 'model' (3DFIT, METHOD GLOBAL or BAYESIAN)) and model_dens (the profile of the
    model, METHOD GLOBAL and BAYESIAN); radmax: the edge of the disk (3DFIT, METHOD GLOBAL or BAYESIAN) or None; cubefile, cubehdu: the cube read
    by the scripts; and disp, pa_av, pa_min, reverse, arcconv (for plot_pvs_old.py)."""
    c = gf.cin
    h = c.Head()
    p = c.pars()
    par = gf.par
    outr = gf.outr
    arcconv = gf.arcconv
    MASK = p.MaskType

    cdelt3 = float(np.float32(DeltaVel(h)))
    crval3 = float(np.float32(AlltoVel(c.getZphys(0), h)))
    bunit = "JY/BEAM * KM/S"
    if FluxtoJy(1, h) == 1:
        bunit = h.Bunit() + " * KM/S"

    xpos = int(findMedian(outr.xpos))
    ypos = int(findMedian(outr.ypos))
    zmin = disp = 0
    zmax = c.DimZ() - 1
    vsys_av = float(np.float32(findMedian(outr.vsys)))
    se = p.se
    if par.PLOTMINCON > 0:
        cont = par.PLOTMINCON
    elif MASK == "SEARCH":
        if se.flagUserGrowthT:
            cont = se.growthThreshold
        elif se.UserThreshold:
            cont = se.threshold
        else:
            cont = (se.growthCut if se.flagGrowth else se.snrCut) * c.stat().getSpread()
    else:
        if not c.StatsDef():
            c.setCubeStats()
        cont = 2.5 * c.stat().getSpread()
    cont = float(np.float32(cont))
    bm_pix = h.Bmaj() / h.PixScale()
    if MASK == "SEARCH" and c.getSources() is not None and c.getNumObj() > 0:
        larg = c.getSources().LargestDetection()
        ext = [abs(xpos - _lround(larg.getXmin() - 2 * bm_pix)), abs(xpos - _lround(larg.getXmax() + 2 * bm_pix)),
               abs(ypos - _lround(larg.getYmin() - 2 * bm_pix)), abs(ypos - _lround(larg.getYmax() + 2 * bm_pix))]
        disp = int(max(ext))
        zmin, zmax = larg.getZmin() - 3, larg.getZmax() + 3
    elif MASK == "SMOOTH&SEARCH":
        zz, yy, xx = np.nonzero(gf.mask)
        if zz.size:
            ext = [abs(xpos - _lround(xx.min() - 2 * bm_pix)), abs(xpos - _lround(xx.max() + 2 * bm_pix)),
                   abs(ypos - _lround(yy.min() - 2 * bm_pix)), abs(ypos - _lround(yy.max() + 2 * bm_pix))]
            disp = int(max(ext))
            zmin, zmax = int(zz.min()) - 3, int(zz.max()) + 3
    else:
        max_v = float(np.max(outr.vrot * np.sin(np.radians(outr.inc)) + outr.vdisp))
        z_vsys = _lround(float(c.getZgrid(Vel2Spec(vsys_av, h))))
        disp_v = int(np.ceil((1.5 * max_v) / abs(cdelt3)))
        zmin = z_vsys - 2 * disp_v if z_vsys - 2 * disp_v > 0 else 0
        zmax = z_vsys + 2 * disp_v if z_vsys + 2 * disp_v < c.DimZ() else c.DimZ() - 1
        disp = int(abs((outr.radii[-1] / arcconv + 2 * h.Bmaj()) / h.PixScale()))
    xmin = xpos - disp if xpos - disp >= 0 else 0
    xmax = xpos + disp if xpos + disp < c.DimX() else c.DimX() - 1
    ymin = ypos - disp if ypos - disp >= 0 else 0
    ymax = ypos + disp if ypos + disp < c.DimY() else c.DimY() - 1
    zmin = max(int(zmin), 0)
    zmax = min(int(zmax), c.DimZ() - 1)

    nc = gf.getErrorColumns()
    errcols1 = {k: nc[i] for i, k in enumerate(fit3d.PAR_KEYS) if nc[i] > 0}
    errcols2 = {k: nc[i] - (6 if k == "vrad" else 0) for i, k in enumerate(fit3d.PAR_KEYS)
                if k in ("vrot", "disp", "vrad") and nc[i] > 0}

    cubefile, cubehdu = p.imageFile, 0
    if p.contsub:
        cubefile = p.outFolder + p.outPrefix + "_contsub.fits"
    else:
        try:                                    # (file.fits[ext], or an image in an extension)
            cubefile, cubehdu = fits_image_hdu(cubefile)
        except Exception:
            pass

    pa_av = float(np.float32(findMedian(outr.phi)))
    model_dens = getattr(gf, "_model_dens_profile", None)
    dens = ("deconvolved" if getattr(gf, "_densprof_deconvolved", False) else "model" if model_dens else "plain")
    edge = getattr(gf, "_disk_edge", None)
    return dict(
        gname=h.Name(), outfolder=p.outFolder, outprefix=p.outPrefix, twostage=bool(par.TWOSTAGE),
        plotmask=bool(par.PLOTMASK), box=(int(xmin), int(xmax), int(ymin), int(ymax), zmin, zmax), cont=cont,
        pixscale=h.PixScale() * arcconv, cdelt3=cdelt3, crval3=crval3,
        beam=(float(np.float32(bm_pix)), float(np.float32(h.Bmin() / h.PixScale())), float(np.float32(h.Bpa()))),
        vrange=(float(np.float32(AlltoVel(c.getZphys(zmin - 0.5), h))),
                float(np.float32(AlltoVel(c.getZphys(zmax + 0.5), h)))),
        bunit=bunit, head_bunit=h.Bunit(), mapnames=fit3d.MAPNAMES[p.ma.maptype == "GAUSSIAN"],
        errcols1=errcols1, errcols2=errcols2, dens=dens, model_dens=model_dens,
        radmax=None if edge is None else float(edge), adrift=bool(par.flagADRIFT), cubefile=cubefile,
        cubehdu=cubehdu, disp=disp, pa_av=pa_av, pa_min=pa_av + 90 if pa_av + 90 < 360 else pa_av - 90,
        reverse=bool(45 <= pa_av < 225), arcconv=arcconv)


# ===================================================================================================== scripts
def _common(s):
    return [("gname", s["gname"], ""), ("outfolder", s["outfolder"], ""), ("outprefix", s["outprefix"], ""),
            ("twostage", s["twostage"], "two stages of fit (rings_final2.txt)")]


_READ_RINGS = "rings = read_rings(outfolder + outprefix + ('_rings_final2.txt' if twostage else '_rings_final1.txt'))\n"

_MODELS = ("files_mod = [f for f in sorted(os.listdir(outfolder)) if outprefix + '_mod' in f]\n"
           "if len(files_mod) == 0:\n"
           "    raise FileNotFoundError('ERROR: no model in output directory')\n")


def _parameters_script(s):
    settings = [("outfolder", s["outfolder"], ""), ("outprefix", s["outprefix"], ""),
                ("twostage", s["twostage"], "two stages of fit (rings_final2.txt)"),
                ("errcols1", s["errcols1"], "columns of the errors in rings_final1.txt"),
                ("errcols2", s["errcols2"], "columns of the errors in rings_final2.txt"),
                ("bunit", s["bunit"], "units of the maps"),
                ("radmax", None if s["radmax"] is None else Code(repr(s["radmax"])),
                 "outer edge of the disk (arcsec), shaded")]
    if s["dens"] == "deconvolved":
        dens = ("rad_dc, sd_dc, sd_dclo, sd_dchi, sd_obs = read_columns(outfolder + outprefix + "
                "'_densprof_deconvolved.txt', (0, 1, 2, 3, 4))\n"
                "sdens, sdens_obs, sdens_label, faceon = (rad_dc, sd_dc, [sd_dclo, sd_dchi]), (rad_dc, sd_obs), "
                "'deconvolved', True\n")
    elif s["dens"] == "model":
        mp = s["model_dens"]
        settings += [("rad_dm", np.asarray(mp["radii"], float), "surface density of the model"),
                     ("sd_dm", np.asarray(mp["sdens"], float), ""),
                     ("sd_dmlo", np.asarray(mp["lo"], float), ""), ("sd_dmhi", np.asarray(mp["hi"], float), ""),
                     ("sd_label", mp["label"], "")]
        dens = ("rad_sd, sd_obs = read_columns(outfolder + outprefix + '_densprof.txt', (0, 9))\n"
                "sdens, sdens_obs, sdens_label, faceon = (rad_dm, sd_dm, [sd_dmlo, sd_dmhi]), (rad_sd, sd_obs), "
                "sd_label, True\n")
    else:
        dens = ("rad_sd, surfdens, sd_err = read_columns(outfolder + outprefix + '_densprof.txt', (0, 3, 4))\n"
                "sdens, sdens_obs, sdens_label, faceon = (rad_sd, surfdens, sd_err), None, None, False\n")
    body = ("rings = read_rings(outfolder + outprefix + '_rings_final1.txt', errcols1)\n"
            "rings2 = read_rings(outfolder + outprefix + '_rings_final2.txt', errcols2) if twostage else None\n"
            + dens +
            "fig = plot_parameters(rings, rings2, sdens=sdens, sdens_obs=sdens_obs, sdens_label=sdens_label, "
            "faceon=faceon,\n                      radmax=radmax, bunit=bunit)\n"
            "fig.savefig(outfolder + outprefix + '_parameters.pdf', bbox_inches='tight')\n")
    funcs = [fit3d.read_rings, fit3d.read_columns, fit3d.ring_errors, fit3d.plot_parameters]
    if s["adrift"]:
        funcs.append(fit3d.plot_asymdrift)
        body += ("\n# Asymmetric drift correction\n"
                 "if os.path.exists(outfolder + outprefix + '_asymdrift.txt'):\n"
                 "    rad_a, vcirc, va2, disp2_a, disp2r, fun, funr = read_columns(outfolder + outprefix + "
                 "'_asymdrift.txt', (0, 1, 2, 3, 4, 5, 6))\n"
                 "    fig = plot_asymdrift(rings2 if twostage else rings, rad_a, vcirc, va2, disp2r, fun, funr)\n"
                 "    fig.savefig(outfolder + outprefix + '_asymmetricdrift.pdf', bbox_inches='tight')\n")
    return script("This script produces plots of the best-fit parameters", funcs, settings, body)


def _cube_settings(s):
    return _common(s) + [
        ("cubefile", s["cubefile"], "the data"), ("cubehdu", s["cubehdu"], ""),
        ("box", s["box"], "xmin, xmax, ymin, ymax, zmin, zmax (pixels, channels)"),
        ("cont", s["cont"], "lowest contour"), ("pixscale", s["pixscale"], "arcsec"),
        ("plotmask", s["plotmask"], "draw the edge of the mask")]


def _chanmaps_script(s):
    settings = _cube_settings(s) + [
        ("cdelt3", s["cdelt3"], "channel width (km/s)"), ("crval3", s["crval3"], "velocity of channel 0 (km/s)"),
        ("beam", s["beam"], "major, minor axis (pixels), PA (deg)"), ("bunit", s["head_bunit"], "units of the cube")]
    body = (_READ_RINGS +
            "data = read_cube(cubefile, cubehdu)\n"
            "mask = read_cube(outfolder + outprefix + '_mask.fits')\n" + _MODELS +
            "for f in files_mod:\n"
            "    model = read_cube(outfolder + f)\n"
            "    fig = plot_chanmaps(data, model, rings, mask, box=box, cont=cont, pixscale=pixscale, cdelt3=cdelt3,\n"
            "                        crval3=crval3, beam=beam, gname=gname, bunit=bunit, plotmask=plotmask)\n"
            "    ntype = f[f.rfind('_'):].replace('.fits', '')\n"
            "    fig.savefig(outfolder + outprefix + '_chanmaps' + ntype + '.pdf', orientation='portrait', "
            "format='pdf')\n"
            "    plt.close(fig)\n")
    return script("This script writes a plot of channel maps of model and data",
                  [fit3d.read_rings, fit3d.read_cube, fit3d.plot_chanmaps], settings, body)


def _pvs_script(s, slices):
    settings = _cube_settings(s) + [("vrange", s["vrange"], "velocities of the edges of the box (km/s)")]
    name, func, out = (("plot_pvslices", fit3d.plot_pvslices, "_pvslices") if slices else
                       ("plot_pvs", fit3d.plot_pvs, "_pv"))
    maps = "mom0" if slices else "mom0, mom1"
    body = (_READ_RINGS +
            "data = read_cube(cubefile, cubehdu)\n"
            "mask = read_cube(outfolder + outprefix + '_mask.fits')\n"
            "mom0 = read_map(outfolder + 'maps/' + outprefix + '_0mom.fits')\n"
            + ("" if slices else "mom1 = read_map(outfolder + 'maps/' + outprefix + '_1mom.fits')\n")
            + _MODELS +
            "for f in files_mod:\n"
            "    model = read_cube(outfolder + f)\n"
            f"    fig = {name}(data, model, rings, {maps}, mask, box=box, cont=cont, pixscale=pixscale,\n"
            f"    {' ' * (len(name) + 7)}vrange=vrange, gname=gname, plotmask=plotmask)\n"
            "    ntype = f[f.rfind('_'):].replace('.fits', '')\n"
            f"    fig.savefig(outfolder + outprefix + '{out}' + ntype + '.pdf', bbox_inches='tight')\n"
            "    plt.close(fig)\n")
    title = ("This script writes a plot of P-V slices across several cuts" if slices else
             "This script writes a plot of position-velocity slices of model and data")
    return script(title, [fit3d.read_rings, fit3d.read_cube, fit3d.read_map, func], settings, body)


def _kinmaps_script(s):
    settings = _common(s) + [("box", s["box"], "xmin, xmax, ymin, ymax, zmin, zmax (pixels, channels)"),
                             ("pixscale", s["pixscale"], "arcsec"), ("mapnames", s["mapnames"], "")]
    body = (_READ_RINGS +
            "maps = [read_map(outfolder + 'maps/' + outprefix + '_%dmom.fits' % i) for i in range(3)]\n"
            "bunit = fits.getheader(outfolder + 'maps/' + outprefix + '_0mom.fits').get('BUNIT', '')\n"
            "files_mod0 = [f for f in sorted(os.listdir(outfolder + 'maps/')) if outprefix + '_mod_0mom' in f]\n"
            "if len(files_mod0) == 0:\n"
            "    raise FileNotFoundError('ERROR: no model maps in output directory')\n"
            "for f in files_mod0:\n"
            "    models = [read_map(outfolder + 'maps/' + f.replace('_mod_0mom', '_mod_%dmom' % i))\n"
            "              for i in range(3)]\n"
            "    fig = plot_kinmaps(maps, models, rings, box=box, pixscale=pixscale, bunit=bunit, mapnames=mapnames)\n"
            "    ntype = f[f.rfind('_'):].replace('.fits', '')\n"
            "    fig.savefig(outfolder + outprefix + '_maps' + ntype + '.pdf', bbox_inches='tight')\n"
            "    plt.close(fig)\n")
    return script("This script writes a plot of kinematic maps of model and data",
                  [fit3d.read_rings, fit3d.read_map, fit3d.plot_kinmaps], settings, body)


def write_fit3d_scripts(gf, settings=None):
    """Write the plotting scripts of the fit gf in <outfolder>/plotscripts/: plot_utils.py and the scripts of
    the plots (returned, in the order they are run). settings: those of plot_settings(gf), if computed."""
    s = settings or plot_settings(gf)
    folder = s["outfolder"] + "plotscripts/"
    mkdirp(folder)
    texts = {"plot_utils.py": plot_utils_text(), "plot_parameters.py": _parameters_script(s),
             "plot_chanmaps.py": _chanmaps_script(s), "plot_pvs.py": _pvs_script(s, False),
             "plot_pvslices.py": _pvs_script(s, True), "plot_kinmaps.py": _kinmaps_script(s)}
    for name, text in texts.items():
        with open(folder + name, "w") as f:
            f.write(text)
    return [n for n in texts if n != "plot_utils.py"]
