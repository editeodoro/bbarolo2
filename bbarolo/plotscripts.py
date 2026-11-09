"""
Python plotting scripts for 3DFIT (port of Galfit::writeScripts_Python in galfit_out.cpp)
and for the SEARCH task (Cube::plotDetections).

As in BBarolo, the scripts are written in the ``plotscripts`` sub-directory of the output
folder (so that they can be edited and re-run by the user) and then executed in parallel.
The script text is generated from the same templates used by the C++ code. To avoid
loading matplotlib & co. once per script, the scripts run in processes forked from a
helper process that has already imported them (PlotRunner, _plotserver.py).
"""

import os
import sys
import json
import math
import subprocess
import numpy as np

from .stats import findMedian
from .header import fits_image_hdu
from .bbutils import DeltaVel, AlltoVel, FluxtoJy, FluxtoJyBeam, Vel2Spec, mkdirp

MAXPAR = 10

# The columns of the ring files read by the scripts (0-based), from galfit.RING_COLUMNS
_RING_KEYS = ("rad", "vrot", "disp", "inc", "pa", "z0", "dens", "xpos", "ypos", "vsys", "vrad")


def _edge_marks(gf):
    """Code of plot_parameters.py that shades the radii beyond the outer edge of the disk (RADMAX of
    Bayesian3DFIT, gf._disk_edge; nothing without it): the rings there are not measurements."""
    edge = getattr(gf, "_disk_edge", None)
    if edge is None:
        return ""
    return ("\n# Beyond the outer edge of the disk (RADMAX) the model has no gas: the rings there are extrapolations"
            f"\nradmax = {float(edge)!r} \nfor a in ax: \n  a.axvspan(radmax, max(max_rad, radmax), color='0.9', "
            "zorder=0, lw=0) \n  a.axvline(radmax, ls='--', lw=1, color='0.5', zorder=0) \n"
            "ax[0].annotate(' RADMAX', xy=(radmax, 0.97), xycoords=('data', 'axes fraction'), ha='left', va='top', "
            "fontsize=9, color='0.4') \n")


def _deconv_panel(bunit):
    """Code of plot_parameters.py for the panel of the surface density of 3DFIT: the face-on profile deconvolved
    from the beam (<prefix>_densprof_deconvolved.txt, with its errors) and the observed one (SDENS_FO_OBS)."""
    return ("rad_dc, sd_dc, sd_dclo, sd_dchi, sd_obs = np.atleast_2d(np.genfromtxt(outfolder+outprefix+"
            "'_densprof_deconvolved.txt',usecols=(0,1,2,3,4))).T \n"
            "ax[2].set_ylabel(r'$\\Sigma$ face-on (" + F(bunit) + " arcsec$^{-2}$)', fontsize=lsize-1) \n"
            "ax[2].plot(rad_dc, sd_obs, 'o', mfc='none', color='#A0A0A0', label='observed') \n"
            "ax[2].errorbar(rad_dc, sd_dc, yerr=[sd_dclo,sd_dchi], fmt='o', color=color2, label='deconvolved') \n"
            "ax[2].legend(fontsize=8, frameon=False) \n")


def _uc(*keys):
    """The usecols of the ring parameters keys (e.g. 'rad', 'vrot') in a ring file, as text: '(0,1)'."""
    from .galfit import RING_COLUMNS
    assert len(_RING_KEYS) == len(RING_COLUMNS)
    return "(" + ",".join(str(_RING_KEYS.index(k)) for k in keys) + ")"


def lround(x):
    return int(math.floor(x + 0.5)) if x >= 0 else -int(math.floor(-x + 0.5))


def F(v):
    """Format a value as the C++ default ostream would (6 significant digits)."""
    if isinstance(v, (bool, np.bool_)):
        return str(int(v))
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        if np.isnan(v):
            return "nan"
        if np.isinf(v):
            return "inf" if v > 0 else "-inf"
        s = "%g" % v
        return s
    return str(v)


def _run_subprocesses(paths, cwd=None):
    """Run each script in its own interpreter, all at the same time; return 0 if all succeeded."""
    procs = []
    env = dict(os.environ)
    env.setdefault("MPLBACKEND", "Agg")
    for p in paths:
        procs.append(subprocess.Popen([sys.executable, p], cwd=cwd or os.getcwd(),
                                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env))
    ret = 0
    for pr in procs:
        ret |= pr.wait()
    return ret


def _use_plot_server():
    """The fast runner needs os.fork (not available on Windows). BBAROLO_PLOT_SUBPROCESS=1
    forces the old behaviour (one new interpreter per script)."""
    return hasattr(os, "fork") and os.environ.get("BBAROLO_PLOT_SUBPROCESS", "0") in ("", "0")


class PlotRunner:
    """Runs the plotting scripts in processes forked from a helper process that has
    already imported numpy, matplotlib, astropy and scipy (see _plotserver.py), instead
    of starting a new interpreter for each script. The scripts and their outputs are the
    same; only the start-up cost changes.

    The helper starts when the runner is created, so it can be created early (e.g. before
    the final model is written) to load the libraries in the background. Call run(paths)
    once; close() stops a helper that was not used.
    """

    def __init__(self, cwd=None):
        self.cwd = cwd or os.getcwd()
        self.proc = None
        if not _use_plot_server():
            return
        env = dict(os.environ)
        env["MPLBACKEND"] = "Agg"
        try:
            self.proc = subprocess.Popen([sys.executable, os.path.join(os.path.dirname(__file__), "_plotserver.py")],
                                         cwd=self.cwd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                         stderr=subprocess.DEVNULL, env=env)
        except OSError:
            self.proc = None

    def run(self, paths, nproc=None):
        """Run the scripts; return 0 if all of them succeeded."""
        paths = [os.path.abspath(p) for p in paths]
        if not paths:
            self.close()
            return 0
        if self.proc is None:
            return _run_subprocesses(paths, self.cwd)
        proc, self.proc = self.proc, None
        try:
            out, _ = proc.communicate(json.dumps({"paths": paths, "nproc": nproc or os.cpu_count() or 1}).encode())
            codes = json.loads(out.decode() or "null")
            if proc.returncode != 0 or not isinstance(codes, list) or len(codes) != len(paths):
                raise RuntimeError("plot helper failed")
        except Exception:
            # Something went wrong with the helper: run the scripts the old way
            if proc.poll() is None:
                proc.kill()
            return _run_subprocesses(paths, self.cwd)
        return 0 if all(c == 0 for c in codes) else 1

    def close(self):
        if self.proc is not None:
            try:
                self.proc.stdin.close()                    # the helper exits on an empty request
                self.proc.wait(timeout=10)
            except Exception:
                self.proc.kill()
            self.proc = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def run_scripts(paths, cwd=None, runner=None):
    """Run python scripts concurrently; return 0 if all succeeded.

    With the default fast runner (PlotRunner) the scripts run in processes forked from a
    helper that has the plotting libraries already loaded; set BBAROLO_PLOT_SUBPROCESS=1
    to start one new interpreter per script as in BBarolo. runner: a PlotRunner created
    beforehand (it is used once)."""
    if runner is None:
        if len(paths) <= 1:                       # nothing to share between scripts
            return _run_subprocesses(paths, cwd)
        runner = PlotRunner(cwd)
    return runner.run(paths)


def plotAll_Python(gf, runner=None):
    names = writeScripts_Python(gf)
    folder = gf.cin.pars().outFolder + "plotscripts/"
    if gf.cin.pars().plots == 0:
        if runner is not None:
            runner.close()
        return 0
    return run_scripts([folder + n for n in names], runner=runner)


def _pv_utils_source():
    """Source of the PV-slice functions for plot_utils.py: the same code as pvslice.py
    (copied, so that the plotting scripts do not need bbarolo), plus the two helpers
    used by the plotting scripts."""
    import inspect
    from . import pvslice
    src = "\n\n".join(inspect.getsource(f) for f in (pvslice._line_range, pvslice.pv_path, pvslice.pv_sample))
    return ("# PV slices: copy of bbarolo/pvslice.py (pv_path, pv_sample)\n" + src + "\n\n"
            "def build_path(x0, y0, rad_pix, pa_deg, nx, ny, oversampling=1):\n"
            "  return pv_path(x0, y0, pa_deg, radii=rad_pix, shape=(ny, nx), step=1.0/oversampling)\n\n\n"
            "def extract_pv(array, x, y, order=1, width=1, oversampling=1):\n"
            "  return pv_sample(array, x, y, width=width, order=order, zoversample=oversampling)\n")


def _plot_utils_text():
    """The source of plot_utils.py, the helpers of the plotting scripts (3DFIT, BAYESIAN3DFIT, BAYESIANWIND)."""
    return ("###############################################################\n#### This script contains some utilities for plotting      ####\n###############################################################\nimport math \nimport numpy as np \nimport matplotlib.pyplot as plt \n\ndef defineaxis(nrows,ncols,xlen,ylen,xsep=0,ysep=0,fig_width=8.27,fig_heigth=11.69, fig=None):\n\n  if not fig: fig = plt.figure(figsize=(fig_width, fig_heigth)) \n  fig_width, fig_heigth = fig.get_size_inches() \n  fig_ratio = fig_width / fig_heigth \n\n  ax = [] \n  for i in range(nrows):\n    row = []\n    y = -(i * (ylen + ysep) * fig_ratio)\n    for j in range(ncols):\n      x = j * (xlen + xsep)\n      row.append(fig.add_axes([x, y, xlen, ylen * fig_ratio])) \n    ax.append(row)\n  return fig, np.squeeze(ax)\n\n"
            + "def calculate_padding(xsize,ysize):\n  n = max(ysize, xsize) \n  pad_y = ((n - ysize) // 2, n - ysize - (n - ysize) // 2) \n  pad_x = ((n - xsize) // 2, n - xsize - (n - xsize) // 2) \n  return (pad_x,pad_y)\n\n"
            + _pv_utils_source())


# the labels of the rows of the maps plot (MAPTYPE MOMENT, GAUSSIAN)
_MAPNAMES = (['MOMENT 0TH', 'MOMENT 1ST', 'MOMENT 2ND'], ['GAUSS INTENSITY', 'GAUSS VELOCITY', 'GAUSS DISPERSION'])


def writeScripts_Python(gf):
    c = gf.cin
    h = c.Head()
    p = c.pars()
    par = gf.par
    outr = gf.outr
    arcconv = gf.arcconv
    outfolder = p.outFolder
    outprefix = p.outPrefix
    gname = h.Name()
    head_bunit = h.Bunit()
    twostage = bool(par.TWOSTAGE)
    plotmask = bool(par.PLOTMASK)
    deconv = bool(getattr(gf, "_densprof_deconvolved", False))     # (3DFIT: <prefix>_densprof_deconvolved.txt)
    pixscale_arcs = h.PixScale() * arcconv

    cdelt3_kms = float(np.float32(DeltaVel(h)))
    crval3_kms_pix1 = float(np.float32(AlltoVel(c.getZphys(0), h)))
    bmaj = float(np.float32(h.Bmaj() / h.PixScale()))
    bmin = float(np.float32(h.Bmin() / h.PixScale()))
    bpa = float(np.float32(h.Bpa()))
    bunit = "JY/BEAM * KM/S"
    if FluxtoJy(1, h) == 1:
        bunit = h.Bunit() + " * KM/S"

    cont = 0.
    xpos = int(findMedian(outr.xpos))
    ypos = int(findMedian(outr.ypos))
    xmin = ymin = zmin = disp = 0
    xmax, ymax, zmax = c.DimX() - 1, c.DimY() - 1, c.DimZ() - 1
    vsys_av = float(np.float32(findMedian(outr.vsys)))
    se = p.se
    MASK = p.MaskType
    if par.PLOTMINCON > 0:
        cont = par.PLOTMINCON
    else:
        if MASK == "SEARCH":
            if se.flagUserGrowthT: cont = se.growthThreshold
            elif se.UserThreshold: cont = se.threshold
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
        ext = [abs(xpos - lround(larg.getXmin() - 2 * bm_pix)), abs(xpos - lround(larg.getXmax() + 2 * bm_pix)),
               abs(ypos - lround(larg.getYmin() - 2 * bm_pix)), abs(ypos - lround(larg.getYmax() + 2 * bm_pix))]
        disp = int(max(ext))
        zmin, zmax = larg.getZmin() - 3, larg.getZmax() + 3
    elif MASK == "SMOOTH&SEARCH":
        m = gf.mask
        zz, yy, xx = np.nonzero(m)
        if zz.size:
            ext = [abs(xpos - lround(xx.min() - 2 * bm_pix)), abs(xpos - lround(xx.max() + 2 * bm_pix)),
                   abs(ypos - lround(yy.min() - 2 * bm_pix)), abs(ypos - lround(yy.max() + 2 * bm_pix))]
            disp = int(max(ext))
            zmin, zmax = int(zz.min()) - 3, int(zz.max()) + 3
    else:
        maxv = outr.vrot * np.sin(np.radians(outr.inc)) + outr.vdisp
        max_v = float(np.max(maxv))
        vsys_spec = Vel2Spec(vsys_av, h)
        z_vsys = lround(float(c.getZgrid(vsys_spec)))
        disp_v = int(math.ceil((1.5 * max_v) / abs(cdelt3_kms)))
        zmin = z_vsys - 2 * disp_v if z_vsys - 2 * disp_v > 0 else 0
        zmax = z_vsys + 2 * disp_v if z_vsys + 2 * disp_v < c.DimZ() else c.DimZ() - 1
        disp = int(abs((outr.radii[-1] / arcconv + 2 * h.Bmaj()) / h.PixScale()))
    xmin = xpos - disp if xpos - disp >= 0 else 0
    xmax = xpos + disp if xpos + disp < c.DimX() else c.DimX() - 1
    ymin = ypos - disp if ypos - disp >= 0 else 0
    ymax = ypos + disp if ypos + disp < c.DimY() else c.DimY() - 1
    if zmin < 0: zmin = 0
    if zmax >= c.DimZ(): zmax = c.DimZ() - 1

    nc = gf.getErrorColumns()
    folder = outfolder + "plotscripts/"
    mkdirp(folder)
    scriptnames = []
    buf = []

    def w(*parts):
        buf.append("".join(parts))

    def flush(name):
        with open(folder + name, "w") as f:
            f.write("".join(buf))
        buf.clear()

    # ---------------------------------------------------------------- plot_utils.py
    w(_plot_utils_text())
    flush("plot_utils.py")

    # ---------------------------------------------------------------- plot_parameters.py
    scriptnames.append("plot_parameters.py")
    w("###############################################################\n#### This script produces plots of the best-fit parameters ####\n###############################################################\nimport numpy as np \nimport matplotlib as mpl \nimport matplotlib.pyplot as plt \nlsize = 11 \nmpl.rc('xtick',direction='in',top=True) \nmpl.rc('ytick',direction='in',right=True) \nmpl.rcParams['contour.negative_linestyle'] = 'solid' \nplt.rc('font',family='sans-serif',serif='Helvetica',size=10) \nparams = {'text.usetex': False, 'mathtext.fontset': 'cm', 'mathtext.default': 'regular', 'errorbar.capsize': 0} \nplt.rcParams.update(params) \n\noutprefix = '",
      F(outprefix),
      "' \noutfolder = '",
      F(outfolder),
      "' \ntwostage = ",
      F(twostage),
      " \n\n")
    w("# Reading in best-fit parameters \nrad_sd, surfdens, sd_err = np.genfromtxt(outfolder+outprefix+'_densprof.txt', usecols=(0,3,4),unpack=True) \nrad,vrot,disp,inc,pa,z0,xpos,ypos,vsys,vrad = np.genfromtxt(outfolder+outprefix+'_rings_final1.txt',usecols=" + _uc("rad", "vrot", "disp", "inc", "pa", "z0", "xpos", "ypos", "vsys", "vrad") + ",unpack=True) \n")
    if outr.nr == 1:
        w("rad,vrot,disp,inc,pa,z0,xpos,ypos,vsys,vrad = np.array([rad]),np.array([vrot]),np.array([disp]),np.array([inc]),np.array([pa]),np.array([z0]),np.array([xpos]),np.array([ypos]),np.array([vsys]),np.array([vrad])\n")
    w("try: err1_l, err1_h = np.zeros(shape=(",
      F(MAXPAR),
      ",len(rad))), np.zeros(shape=(",
      F(MAXPAR),
      ",len(rad)))\nexcept: err1_l, err1_h = np.zeros(shape=(",
      F(MAXPAR),
      ",1)), np.zeros(shape=(",
      F(MAXPAR),
      ",1))\ncolor=color2='#B22222' \nmax_vrot,max_vdisp = np.nanmax(vrot),np.nanmax(disp) \nmax_rad = 1.1*np.nanmax(rad) \n")
    for i in range(MAXPAR):
        if nc[i] > 0:
            w("err1_l[",
              F(i),
              "], err1_h[",
              F(i),
              "] = np.genfromtxt(outfolder+outprefix+'_rings_final1.txt',usecols=(",
              F(nc[i]),
              ",",
              F(nc[i]+1),
              "),unpack=True) \n")
    w("\nif twostage: \n  rad2, vrot2,disp2,inc2,pa2,z02,xpos2,ypos2,vsys2, vrad2 = np.genfromtxt(outfolder+outprefix+'_rings_final2.txt',usecols=" + _uc("rad", "vrot", "disp", "inc", "pa", "z0", "xpos", "ypos", "vsys", "vrad") + ",unpack=True)\n")
    if outr.nr == 1:
        w("  rad2,vrot2,disp2,inc2,pa2,z02,xpos2,ypos2,vsys2,vrad2 = np.array([rad2]),np.array([vrot2]),np.array([disp2]),np.array([inc2]),np.array([pa2]),np.array([z02]),np.array([xpos2]),np.array([ypos2]),np.array([vsys2]),np.array([vrad2])\n")
    w("  err2_l, err2_h = np.zeros(shape=(",
      F(MAXPAR),
      ",len(rad2))), np.zeros(shape=(",
      F(MAXPAR),
      ",len(rad2)))\n  color='#A0A0A0' \n  max_rad = 1.1*np.nanmax(rad2) \n  max_vrot,max_vdisp = np.maximum(max_vrot,np.nanmax(vrot2)),np.maximum(max_vdisp,np.nanmax(disp2)) \n")
    for i in range(MAXPAR):
        if (i == 0 or i == 1 or i == 9) and nc[i] > 0:
            ff = 6 if i == 9 else 0
            w("  err2_l[",
              F(i),
              "], err2_h[",
              F(i),
              "] = np.genfromtxt(outfolder+outprefix+'_rings_final2.txt',usecols=(",
              F(nc[i]-ff),
              ",",
              F(nc[i]+1-ff),
              "),unpack=True) \n")
    w("\n# Defining figure and axes \nfig=plt.figure(figsize=(11,11),dpi=150) \nnrows, ncols = 3,3 \nxlen, ylen = 0.27, 0.13 \nx_sep, y_sep = 0.07,0.015 \nbottom_corner = [0.1,0.7] \nfor i in range (nrows): \n  bottom_corner[0], yl = 0.1, ylen \n  if i==0: yl *= 1.8 \n  for j in range (ncols): \n    fig.add_axes([bottom_corner[0],bottom_corner[1],xlen,yl]) \n    fig.axes[-1].set_xlim(0,max_rad) \n    if i==nrows-1: fig.axes[-1].tick_params(labelbottom=True) \n    else: fig.axes[-1].tick_params(labelbottom=False) \n    bottom_corner[0]+=xlen+x_sep \n  bottom_corner[1]-=(ylen+y_sep) \n\nax = fig.axes \n\n# Plotting rotation velocity \nax[0].set_ylim(0,1.2*max_vrot) \nax[0].set_ylabel(r'V$_\\mathrm{rot}$ (km/s)', fontsize=lsize) \nax[0].errorbar(rad,vrot, yerr=[-err1_l[0],err1_h[0]],fmt='o', color=color) \nif twostage: ax[0].errorbar(rad2,vrot2, yerr=[-err2_l[0],err2_h[0]],fmt='o', color=color2) \n\n# Plotting velocity dispersion \nax[1].set_ylim(0,1.2*max_vdisp) \nax[1].set_ylabel(r'$\\sigma_\\mathrm{gas}$  (km/s)', fontsize=lsize) \nax[1].errorbar(rad,disp, yerr=[-err1_l[1],err1_h[1]],fmt='o', color=color) \nif twostage: ax[1].errorbar(rad2,disp2, yerr=[-err2_l[1],err2_h[1]],fmt='o', color=color2) \n\n# Plotting surface density \nax[2].set_xlim(0,max_rad) \n" + (_deconv_panel(bunit) if deconv else "ax[2].set_ylabel(r'$\\Sigma$ (" + F(bunit) + ")', fontsize=lsize) \nax[2].errorbar(rad_sd,surfdens, yerr=sd_err,fmt='o', color=color2) \n") + "\n# Plotting inclination angle \nax[3].set_ylabel('i (deg)', fontsize=lsize) \nax[3].errorbar(rad,inc, yerr=[-err1_l[4],err1_h[4]],fmt='o', color=color) \nif twostage: ax[3].errorbar(rad2,inc2,yerr=[-err2_l[4],err2_h[4]], fmt='o-', color=color2) \n\n# Plotting x-center \nax[4].set_ylabel(r'x$_0$ (pix)', fontsize=lsize) \nax[4].errorbar(rad,xpos, yerr=[-err1_l[6],err1_h[6]],fmt='o', color=color) \nif twostage: ax[4].errorbar(rad2,xpos2,yerr=[-err2_l[6],err2_h[6]],fmt='o-', color=color2) \n\n# Plotting radial velocity \nax[5].set_xlim(0,max_rad) \nax[5].set_ylabel(r'V$_\\mathrm{rad}$ (km/s)', fontsize=lsize) \nax[5].errorbar(rad,vrad, yerr=[-err1_l[9],err1_h[9]],fmt='o', color=color) \nif twostage: ax[5].errorbar(rad2,vrad2,yerr=[-err2_l[9],err2_h[9]],fmt='o', color=color2) \n\n# Plotting position angle \nax[6].set_ylabel(r'$\\phi$ (deg)', fontsize=lsize) \nax[6].set_xlabel('Radius (arcsec)', fontsize=lsize, labelpad=10) \nax[6].errorbar(rad,pa, yerr=[-err1_l[5],err1_h[5]],fmt='o', color=color) \nif twostage: ax[6].errorbar(rad2,pa2,yerr=[-err2_l[5],err2_h[5]], fmt='o-', color=color2) \n\n# Plotting y-center \nax[7].set_ylabel(r'y$_0$ (pix)', fontsize=lsize) \nax[7].set_xlabel('Radius (arcsec)', fontsize=lsize, labelpad=10) \nax[7].errorbar(rad,ypos, yerr=[-err1_l[7],err1_h[7]],fmt='o', color=color) \nif twostage: ax[7].errorbar(rad2,ypos2, yerr=[-err2_l[7],err2_h[7]],fmt='o-', color=color2) \n\n# Plotting systemic velocity \nax[8].set_ylabel(r'v$_\\mathrm{sys}$ (km/s)', fontsize=lsize) \nax[8].set_xlabel('Radius (arcsec)', fontsize=lsize, labelpad=10) \nax[8].errorbar(rad,vsys, yerr=[-err1_l[8],err1_h[8]],fmt='o', color=color) \nif twostage: ax[8].errorbar(rad2,vsys2,yerr=[-err2_l[8],err2_h[8]],fmt='o', color=color2) \n" + _edge_marks(gf) + "\nfig.savefig(outfolder+'%s_parameters.pdf'%outprefix,bbox_inches='tight') \n")
    if par.flagADRIFT:
        w("\n#Asymmetric drift correction\nrad_a, vcirc, va2, disp2_a, disp2r, fun, funr = np.genfromtxt(outfolder+outprefix+'_asymdrift.txt',usecols=(0,1,2,3,4,5,6),unpack=True)\nif twostage: \n  rad, vrot, disp = rad2, vrot2, disp2 \n  err1_l, err1_h = err2_l, err2_h\nfig=plt.figure(figsize=(10,10), dpi=150) \nax1 = fig.add_axes([0.10,0.1,0.3,0.25])\nax2 = fig.add_axes([0.48,0.1,0.3,0.25])\nax3 = fig.add_axes([0.86,0.1,0.3,0.25])\nax1.set_xlim(0,max_rad)\nax1.set_xlabel('Radius (arcsec)', fontsize=11, labelpad=5)\nax1.set_ylabel('V (km/s)', fontsize=11)\nax1.plot(rad,vrot,'-', color=color2,label=r'V$_\\mathrm{rot}$',zorder=1)\nax1.plot(rad_a,np.sqrt(va2),'--', color='#FABC11',label=r'V$_\\mathrm{A}$',zorder=2)\nax1.errorbar(rad_a,vcirc, yerr=[-err1_l[0],err1_h[0]],fmt='o', color='#43A8D4',label=r'V$_\\mathrm{circ}$',zorder=0)\nax1.legend()\nax2.set_xlim(0,max_rad)\nax2.set_xlabel('Radius (arcsec)', fontsize=11, labelpad=5)\nax2.set_ylabel(r'$\\sigma$ (km/s)', fontsize=11)\nax2.errorbar(rad,disp, yerr=[-err1_l[1],err1_h[1]],fmt='o', color=color2,label=r'$\\sigma_\\mathrm{gas}$',zorder=0)\nax2.plot(rad_a,np.sqrt(disp2r),'-', color='#0FA45A',label=r'$\\sigma_\\mathrm{reg}$',zorder=1)\nax2.legend()\nax3.set_xlim(0,max_rad)\nax3.set_xlabel('Radius (arcsec)', fontsize=11, labelpad=5)\nax3.set_ylabel(r'$\\Sigma_\\mathrm{gas}\\cos(i)$', fontsize=11)\nax3.plot(rad_a,fun,'o', color=color2,label=r'$f_\\mathrm{obs}$',zorder=0)\nax3.plot(rad_a,funr,'-', color='#0FA45A',label=r'$f_\\mathrm{reg}$',zorder=1)\nax3.legend()\nfig.savefig(outfolder+'%s_asymmetricdrift.pdf'%outprefix,bbox_inches='tight')\n")
    flush("plot_parameters.py")

    # ---------------------------------------------------------------- plot_chanmaps.py
    scriptnames.append("plot_chanmaps.py")
    cubefile, cubehdu = p.imageFile, 0
    if p.contsub:
        cubefile = outfolder + outprefix + "_contsub.fits"
    else:
        try:                                    # (file.fits[ext], or an image in an extension)
            cubefile, cubehdu = fits_image_hdu(cubefile)
        except Exception:
            pass
    w("#####################################################################\n#### This script writes a plot of channel maps of model and data ####\n#####################################################################\nimport numpy as np \nimport os \nimport matplotlib as mpl \nimport matplotlib.pyplot as plt \nimport matplotlib.gridspec as gridspec \nfrom plot_utils import calculate_padding \nfrom astropy.io import fits \nfrom astropy.visualization import PowerStretch \nfrom astropy.visualization.mpl_normalize import ImageNormalize \nfrom astropy.visualization import PercentileInterval \nmpl.rc('xtick',direction='in',top=True,labelbottom=False) \nmpl.rc('ytick',direction='in',right=True,labelleft=False) \nmpl.rcParams['contour.negative_linestyle'] = 'solid' \nplt.rc('font',family='sans-serif',serif='Helvetica',size=10) \nparams = {'text.usetex': False, 'mathtext.fontset': 'cm', 'mathtext.default': 'regular'} \nplt.rcParams.update(params) \n\ngname = '",
      F(gname),
      "' \noutfolder = '",
      F(outfolder),
      "' \noutprefix = '",
      F(outprefix),
      "' \ntwostage = ",
      F(twostage),
      " \nplotmask = ",
      F(plotmask),
      " \n\nif twostage: xpos,ypos,vsys = np.genfromtxt(outfolder+outprefix+'_rings_final2.txt',usecols=" + _uc("xpos", "ypos", "vsys") + ",unpack=True) \nelse: xpos,ypos,vsys = np.genfromtxt(outfolder+outprefix+'_rings_final1.txt',usecols=" + _uc("xpos", "ypos", "vsys") + ",unpack=True) \nxcen_m,ycen_m,vsys_m = np.nanmean((xpos,ypos,vsys),axis=1) \n\n# CHANNEL MAPS: Setting all the needed variables \nimage = fits.open('",
      F(cubefile),
      "') \nimage_mas = fits.open(outfolder+outprefix+'_mask.fits') \nxmin, xmax = ",
      F(xmin),
      ", ",
      F(xmax),
      "\nymin, ymax = ",
      F(ymin),
      ", ",
      F(ymax),
      "\nzmin, zmax = ",
      F(zmin),
      ", ",
      F(zmax),
      "\ndata = image[" + str(cubehdu) + "].data[")
    if h.NumAx() > 3:
        for i in range(h.NumAx() - 3):
            w("0,")
    w("zmin:zmax+1,ymin:ymax+1,xmin:xmax+1] \ndata_mas = image_mas[0].data[zmin:zmax+1,ymin:ymax+1,xmin:xmax+1] \nhead = image[" + str(cubehdu) + "].header \nzsize, ysize, xsize = data.shape \npad_x, pad_y = calculate_padding(xsize,ysize) \ncdeltsp=",
      F(pixscale_arcs),
      "\ncont = ",
      F(cont),
      "\nv = np.array([1,2,4,8,16,32,64])*cont \nv_neg = [-cont] \ninterval = PercentileInterval(99.5) \nvmax = interval.get_limits(data)[1] \nnorm = ImageNormalize(vmin=cont, vmax=vmax, stretch=PowerStretch(0.5)) \nxcen, ycen = xcen_m-xmin+pad_x[0], ycen_m-ymin+pad_y[0] \n\nfiles_mod = [f for f in sorted(os.listdir(outfolder)) if outprefix+'_mod' in f]  \nif len(files_mod)==0: raise FileNotFoundError('ERROR: no model in output directory') \n\n")
    w("# Beginning channel map plot \nfor k in range (len(files_mod)): \n  image_mod = fits.open(outfolder+files_mod[k]) \n  data_mod = image_mod[0].data[zmin:zmax+1,ymin:ymax+1,xmin:xmax+1] \n  fig = plt.figure(figsize=(8.27, 11.69), dpi=150) \n  grid = [gridspec.GridSpec(2,5),gridspec.GridSpec(2,5),gridspec.GridSpec(2,5)] \n  grid[0].update(top=0.90, bottom=0.645, left=0.05, right=0.95, wspace=0.0, hspace=0.0) \n  grid[1].update(top=0.60, bottom=0.345, left=0.05, right=0.95, wspace=0.0, hspace=0.0) \n  grid[2].update(top=0.30, bottom=0.045, left=0.05, right=0.95, wspace=0.0, hspace=0.0) \n\n  num = 0 \n  for j in range (0,3): \n    for i in range (0,5): \n      chan = int(num*(zsize)/15) \n      z = np.pad(data[chan,:,:],(pad_y, pad_x),mode='constant',constant_values=np.nan) \n      z_mod = np.pad(data_mod[chan,:,:],(pad_y, pad_x),mode='constant',constant_values=np.nan) \n      #New matplotlib draws wrong contours when no contours are found. This is a workaround.\n      if np.all(z_mod<v[0]): z_mod[:,:] =0\n      velo_kms = (chan+1-",
      F(1-zmin),
      ")*",
      F(cdelt3_kms),
      "+",
      F(crval3_kms_pix1),
      "\n      velo = ' v = ' + str(int(velo_kms-vsys_m)) + ' km/s' \n      ax = plt.subplot(grid[j][0,i]) \n      ax.set_title(velo, fontsize=10,loc='left') \n      ax.imshow(z,origin='lower',cmap = mpl.cm.Greys,norm=norm,aspect='auto',interpolation=None) \n      ax.contour(z,v,origin='lower',linewidths=0.7,colors='#00008B') \n      ax.contour(z,v_neg,origin='lower',linewidths=0.1,colors='gray') \n      ax.plot(xcen,ycen,'x',color='#0FB05A',markersize=7,mew=2) \n      if plotmask: \n        z_mas = np.pad(data_mas[chan],(pad_y, pad_x),mode='constant',constant_values=0) \n        ax.contour(z_mas,levels=[0],origin='lower',linewidths=2,colors='k') \n      if (j==i==0): \n        ax.text(0, 1.4, gname, transform=ax.transAxes,fontsize=15,va='center') \n        lbar = 0.5*(xmax-xmin)*cdeltsp \n        ltex = \"%.0f'' \"%lbar if lbar>10 else \"%.2f'' \"%lbar \n        if lbar>600: ltex = \"%.0f' \"%(lbar/60.) \n        ax.annotate('', xy=(4.5, 1.4), xycoords='axes fraction', xytext=(5, 1.4),arrowprops=dict(arrowstyle='<->', color='k'))\n        ax.text(4.75,1.50,ltex,transform=ax.transAxes,fontsize=11, ha='center')\n        bmaj, bmin, bpa = ",
      F(bmaj),
      "/float(xmax-xmin), ",
      F(bmin),
      "/float(ymax-ymin),",
      F(bpa),
      "\n        beam = mpl.patches.Ellipse(xy=(3.5, 1.4), width=bmaj, height=bmin, angle=bpa+90, \\\n                                   color='#5605D0', clip_on=False, transform=ax.transAxes, alpha=0.2) \n        ax.add_artist(beam) \n        ax.text(3.6+bmaj/1.8,1.4,'Beam',transform=ax.transAxes,fontsize=11, ha='left',va='center') \n      ax = plt.subplot(grid[j][1,i]) \n      ax.tick_params(axis='both',which='both',bottom=True,top=True,labelbottom=False,labelleft=False) \n      ax.imshow(z_mod,origin='lower',cmap = mpl.cm.Greys,norm=norm,aspect='auto',interpolation=None) \n      ax.contour(z_mod,v,origin='lower',linewidths=0.7,colors='#B22222') \n      ax.plot(xcen,ycen,'x',color='#0FB05A',markersize=7,mew=2) \n      if (i==0 and j==2): \n        clab = r'Contour levels at 2$^n \\, c_{min}$, where $c_{min}$ = %s ",
      F(head_bunit),
      " and n = 0,1,..,8 '%cont \n        ax.text(0.01,-0.16,clab,transform=ax.transAxes,fontsize=11, ha='left',va='center') \n      num = num+1 \n\n  outfile = '%s_chanmaps'%outprefix \n  ntype   = files_mod[k][files_mod[k].rfind('_'):].replace('.fits','') \n  fig.savefig(outfolder+outfile+ntype+'.pdf', orientation = 'portrait', format = 'pdf') \n  image_mod.close() \n\nimage.close() \n")
    flush("plot_chanmaps.py")

    zmin_wcs = float(np.float32(AlltoVel(c.getZphys(zmin - 0.5), h)))
    zmax_wcs = float(np.float32(AlltoVel(c.getZphys(zmax + 0.5), h)))
    pa_av = float(np.float32(findMedian(outr.phi)))
    pa_min = pa_av + 90 if pa_av + 90 < 360 else pa_av - 90
    reverse = (pa_av >= 45 and pa_av < 225)

    # ---------------------------------------------------------------- plot_pvs_old.py
    w("#################################################################################\n#### This script writes a plot of position-velocity slices of model and data ####\n#################################################################################\nimport numpy as np \nimport os \nimport matplotlib as mpl \nimport matplotlib.pyplot as plt \nfrom plot_utils import defineaxis \nfrom astropy.io import fits \nfrom astropy.visualization import PowerStretch \nfrom astropy.visualization.mpl_normalize import ImageNormalize \nfrom astropy.visualization import PercentileInterval \nmpl.rc('xtick',direction='in') \nmpl.rc('ytick',direction='in') \nmpl.rcParams['contour.negative_linestyle'] = 'solid' \nplt.rc('font',family='sans-serif',serif='Helvetica',size=10) \nparams = {'text.usetex': False, 'mathtext.fontset': 'cm', 'mathtext.default': 'regular'} \nplt.rcParams.update(params) \n\ngname = '",
      F(gname),
      "' \noutfolder = '",
      F(outfolder),
      "' \noutprefix = '",
      F(outprefix),
      "' \ntwostage = ",
      F(twostage),
      " \nplotmask = ",
      F(plotmask),
      " \nzmin, zmax = ",
      F(zmin),
      ", ",
      F(zmax),
      "\n\nif twostage: rad,vrot,inc,pa,vsys = np.genfromtxt(outfolder+outprefix+'_rings_final2.txt',usecols=" + _uc("rad", "vrot", "inc", "pa", "vsys") + ",unpack=True) \nelse: rad,vrot,inc,pa,vsys = np.genfromtxt(outfolder+outprefix+'_rings_final1.txt',usecols=" + _uc("rad", "vrot", "inc", "pa", "vsys") + ",unpack=True) \n\nfiles_pva_mod = [f for f in sorted(os.listdir(outfolder+'pvs/')) if outprefix+'_mod_pv_a' in f] \nfiles_pvb_mod = [f for f in sorted(os.listdir(outfolder+'pvs/')) if outprefix+'_mod_pv_b' in f] \nif len(files_pva_mod)==0 or len(files_pvb_mod)==0: raise FileNotFoundError('ERROR: no PV model in output directory') \n\n\nimage_maj     = fits.open(outfolder+'pvs/'+outprefix+'_pv_a.fits') \nimage_min     = fits.open(outfolder+'pvs/'+outprefix+'_pv_b.fits') \nimage_mas_maj = fits.open(outfolder+'pvs/'+outprefix+'_mask_pv_a.fits') \nimage_mas_min = fits.open(outfolder+'pvs/'+outprefix+'_mask_pv_b.fits') \nhead = [image_maj[0].header,image_min[0].header] \ncrpixpv = np.array([head[0]['CRPIX1'],head[1]['CRPIX1']]) \ncdeltpv = np.array([head[0]['CDELT1'],head[1]['CDELT1']]) \ncrvalpv = np.array([head[0]['CRVAL1'],head[1]['CRVAL1']]) \nxminpv, xmaxpv = np.floor(crpixpv-1-",
      F(disp),
      "), np.ceil(crpixpv-1 +",
      F(disp),
      ") \nif xminpv[0]<0: xminpv[0]=0 \nif xminpv[1]<0: xminpv[1]=0 \nif xmaxpv[0]>=head[0]['NAXIS1']: xmaxpv[0]=head[0]['NAXIS1']-1 \nif xmaxpv[1]>=head[1]['NAXIS1']: xmaxpv[1]=head[1]['NAXIS1']-1 \ndata_maj = image_maj[0].data[zmin:zmax+1,int(xminpv[0]):int(xmaxpv[0])+1] \ndata_min = image_min[0].data[zmin:zmax+1,int(xminpv[1]):int(xmaxpv[1])+1] \ndata_mas_maj = image_mas_maj[0].data[zmin:zmax+1,int(xminpv[0]):int(xmaxpv[0])+1] \ndata_mas_min = image_mas_min[0].data[zmin:zmax+1,int(xminpv[1]):int(xmaxpv[1])+1] \nxmin_wcs = ((xminpv+1-0.5-crpixpv)*cdeltpv+crvalpv)*",
      F(arcconv),
      "\nxmax_wcs = ((xmaxpv+1+0.5-crpixpv)*cdeltpv+crvalpv)*",
      F(arcconv),
      "\nzmin_wcs, zmax_wcs = ",
      F(zmin_wcs),
      ", ",
      F(zmax_wcs),
      "\ncont = ",
      F(cont),
      "\nv = np.array([1,2,4,8,16,32,64])*cont \nv_neg = [-cont] \ninterval = PercentileInterval(99.5) \nvmax = interval.get_limits(data_maj)[1] \nnorm = ImageNormalize(vmin=cont, vmax=vmax, stretch=PowerStretch(0.5)) \n\n")
    w("radius = np.concatenate((rad,-rad)) \npamaj_av = ",
      F(pa_av),
      "\npamin_av = ",
      F(pa_min),
      "\ncosth = np.cos(np.deg2rad(np.abs(pa-pamaj_av))) \nvlos1 = vsys+vrot*np.sin(np.deg2rad(inc))*costh \nvlos2 = vsys-vrot*np.sin(np.deg2rad(inc))*costh \n")
    if reverse:
        w("reverse = True \n")
    else:
        w("reverse = False \n")
    w("if reverse: vlos1, vlos2 = vlos2, vlos1 \nvlos = np.concatenate((vlos1,vlos2)) \nvsys_m = np.nanmean(vsys) \next = [[xmin_wcs[0],xmax_wcs[0],zmin_wcs-vsys_m,zmax_wcs-vsys_m],\\\n       [xmin_wcs[1],xmax_wcs[1],zmin_wcs-vsys_m,zmax_wcs-vsys_m]] \nlabsize = 14 \npalab = [r'$\\phi = $%i$^\\circ$'%np.round(pamaj_av), r'$\\phi = $%i$^\\circ$'%np.round(pamin_av)] \n\n")
    w("# Beginning PV plot \nfor k in range (len(files_pva_mod)): \n  image_mod_maj = fits.open(outfolder+'pvs/'+files_pva_mod[k]) \n  image_mod_min = fits.open(outfolder+'pvs/'+files_pvb_mod[k]) \n  data_mod_maj = image_mod_maj[0].data[zmin:zmax+1,int(xminpv[0]):int(xmaxpv[0])+1] \n  data_mod_min = image_mod_min[0].data[zmin:zmax+1,int(xminpv[1]):int(xmaxpv[1])+1] \n  toplot = [[data_maj,data_min],[data_mod_maj,data_mod_min],[data_mas_maj,data_mas_min]] \n\n  fig, ax = defineaxis(2,1,0.6,0.42,xsep=0.00,ysep=0.08,fig_width=10,fig_heigth=10) \n\n  for i in range (2): \n    axis = ax[i] \n    axis.tick_params(which='major',length=8, labelsize=labsize) \n    axis.set_xlabel('Offset (arcsec)',fontsize=labsize+2) \n    axis.set_ylabel(r'$\\mathrm{\\Delta V_{LOS}}$ (km/s)',fontsize=labsize+2) \n    axis.text(1, 1.02,palab[i],ha='right',transform=axis.transAxes,fontsize=labsize+4) \n    axis2 = axis.twinx() \n    axis2.set_xlim([ext[i][0],ext[i][1]]) \n    axis2.set_ylim([ext[i][2]+vsys_m,ext[i][3]+vsys_m]) \n    axis2.tick_params(which='major',length=8, labelsize=labsize) \n    axis2.set_ylabel(r'$\\mathrm{V_{LOS}}$ (km/s)',fontsize=labsize+2) \n    axis.imshow(toplot[0][i],origin='lower',cmap = mpl.cm.Greys,norm=norm,extent=ext[i],aspect='auto') \n    axis.contour(toplot[0][i],v,origin='lower',linewidths=0.7,colors='#00008B',extent=ext[i]) \n    axis.contour(toplot[0][i],v_neg,origin='lower',linewidths=0.1,colors='gray',extent=ext[i]) \n    axis.contour(toplot[1][i],v,origin='lower',linewidths=1,colors='#B22222',extent=ext[i]) \n    axis.axhline(y=0,color='black') \n    axis.axvline(x=0,color='black') \n    axis.grid(color='gray', linestyle='--', linewidth=0.3) \n    if plotmask: \n      axis.contour(toplot[2][i],levels=[0],origin='lower',linewidths=2,colors='k',extent=ext[i]) \n    if i==0 : \n      axis2.plot(radius,vlos,'yo') \n      axis.text(0, 1.1, gname, transform=axis.transAxes,fontsize=22) \n\n  outfile = '%s_pv'%outprefix \n  ntype   = files_pva_mod[k][files_pva_mod[k].rfind('_'):].replace('.fits','') \n  fig.savefig(outfolder+outfile+ntype+'.pdf', bbox_inches='tight') \n  image_mod_maj.close() \n  image_mod_min.close() \n\nimage_maj.close() \nimage_min.close() \n")
    flush("plot_pvs_old.py")

    # ---------------------------------------------------------------- plot_pvs.py
    scriptnames.append("plot_pvs.py")
    w("#################################################################################\n#### This script writes a plot of position-velocity slices of model and data ####\n#################################################################################\nimport numpy as np \nimport os \nimport matplotlib as mpl \nimport matplotlib.pyplot as plt \nfrom plot_utils import *\nfrom astropy.io import fits \nfrom astropy.visualization import ImageNormalize, PercentileInterval, PowerStretch\n\nlabsize = 18\nmpl.rc('xtick',direction='in',top=True) \nmpl.rc('ytick',direction='in',right=True) \nmpl.rcParams['contour.negative_linestyle'] = 'solid' \nplt.rc('font',family='sans-serif',serif='Helvetica',size=labsize) \nparams = {'text.usetex': False, 'mathtext.fontset': 'cm', 'mathtext.default': 'regular'} \nplt.rcParams.update(params) \n\ngname = '",
      F(gname),
      "' \noutfolder = '",
      F(outfolder),
      "' \noutprefix = '",
      F(outprefix),
      "' \ntwostage = ",
      F(twostage),
      " \nplotmask = ",
      F(plotmask),
      " \n\nimage = fits.open('",
      F(cubefile),
      "') \nimage_mas = fits.open(outfolder+outprefix+'_mask.fits') \nxmin, xmax = ",
      F(xmin),
      ", ",
      F(xmax),
      "\nymin, ymax = ",
      F(ymin),
      ", ",
      F(ymax),
      "\nzmin, zmax = ",
      F(zmin),
      ", ",
      F(zmax),
      "\nzmin_wcs, zmax_wcs = ",
      F(zmin_wcs),
      ", ",
      F(zmax_wcs),
      "\ndata = image[" + str(cubehdu) + "].data[")
    if h.NumAx() > 3:
        for i in range(h.NumAx() - 3):
            w("0,")
    w("zmin:zmax+1,ymin:ymax+1,xmin:xmax+1] \ndata_mas = image_mas[0].data[zmin:zmax+1,ymin:ymax+1,xmin:xmax+1] \nzsize, ysize, xsize = data.shape \npad_x, pad_y = calculate_padding(xsize,ysize) \ncdeltsp=",
      F(pixscale_arcs),
      "\ncont = ",
      F(cont),
      "\nv = np.array([1,2,4,8,16,32,64])*cont \nv_neg = [-cont] \nmom0 = fits.open(outfolder+'/maps/'+outprefix+'_0mom.fits')[0].data[ymin:ymax+1,xmin:xmax+1] \nmom1 = fits.open(outfolder+'/maps/'+outprefix+'_1mom.fits')[0].data[ymin:ymax+1,xmin:xmax+1] \nmom0p = np.pad(mom0,(pad_y, pad_x),mode='constant',constant_values=np.nan) \nmom1p = np.pad(mom1,(pad_y, pad_x),mode='constant',constant_values=np.nan) \nextm = [0,mom0p.shape[1]-1,0,mom0p.shape[0]-1]\nnorm0 = ImageNormalize(mom0, interval=PercentileInterval(98.0), stretch=PowerStretch(0.5))\nnorm1 = ImageNormalize(mom1, interval=PercentileInterval(99.5))\n\nfrile = outfolder + outprefix + ('_rings_final2.txt' if twostage else '_rings_final1.txt')\nrings = np.genfromtxt(frile,usecols=" + _uc("rad", "vrot", "disp", "inc", "pa", "xpos", "ypos", "vsys") + ",unpack=True)\nrad,vrot,vdisp,inc,pa,xpos,ypos,vsys = rings\nxcen_m,ycen_m,vsys_m,inc_m,pa_m = np.nanmean((xpos,ypos,vsys,inc,pa),axis=1) \nxcen_m, ycen_m = xcen_m-xmin, ycen_m-ymin \nproj_vmax = np.nanmax(vrot)*np.sin(np.radians(inc_m))\nmax_vdisp = np.nanmax(vdisp)\nradii = np.concatenate((rad,-rad)) \nvlos1 = vrot*np.sin(np.deg2rad(inc))\nvlos = np.concatenate((vlos1,-vlos1)) \n\nfiles_mod = [f for f in sorted(os.listdir(outfolder)) if outprefix+'_mod' in f] \nif len(files_mod)==0: raise FileNotFoundError('ERROR: no model in output directory') \n\n\ndef plot_pv(ax,rads,x0,y0,theta,**kwargs):\n\n  s,x,y = build_path(x0, y0, rads/cdeltsp, theta, xsize, ysize)\n  pv_data = extract_pv(data,x,y,order=0)\n  pv_mod  = extract_pv(data_mod,x,y)\n  norm = ImageNormalize(vmin=cont, vmax=np.percentile(data, 99.8), stretch=PowerStretch(0.5)) \n  #norm = ImageNormalize(data, interval=PercentileInterval(99.9), stretch=PowerStretch(1.0))\n\n  s_arcsec = s * cdeltsp\n  ext = [s_arcsec[0], s_arcsec[-1], zmin_wcs-vsys_m,zmax_wcs-vsys_m]\n\n  ax.imshow(pv_data,origin='lower',cmap='Greys',norm=norm,aspect='auto',interpolation='nearest',extent=ext)\n  ax.contour(pv_data,v,origin='lower',linewidths=1.0,colors='#00008B',extent=ext) \n  ax.contour(pv_data,v_neg,origin='lower',linewidths=0.2,colors='gray',extent=ext) \n  ax.contour(pv_mod,v,origin='lower',linewidths=1.5,colors='#B22222',extent=ext) \n  ax.axhline(y=0,color='black') \n  ax.axvline(x=0,color='black') \n  ax.grid(color='gray', linestyle='--', linewidth=0.3) \n\n  if plotmask:\n    pv_mask = extract_pv(data_mas,x,y) \n    ax.contour(pv_mask,levels=[0],origin='lower',linewidths=3.5,colors='k',extent=ext)\n\n  ax.set_xlim(-1.3*rads[-1],1.3*rads[-1])\n  #ax.set_ylim(-1.1*(proj_vmax+2*max_vdisp),1.1*(proj_vmax+2*max_vdisp))\n  ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=6))\n  ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=6))\n  ax.tick_params(labelbottom=True,labelleft=True)\n  ax.tick_params(which='major',length=8, labelsize=labsize) \n  ax.set_ylabel(r'$\\mathrm{\\Delta V_{LOS}}$ (km/s)',fontsize=labsize+2) \n  ax.set_xlabel('Offset (arcsec)',fontsize=labsize+2)\n\n  ax2 = ax.twinx()\n  ax2.set_ylim([ext[2]+vsys_m,ext[3]+vsys_m]) \n  ax2.tick_params(which='major',length=8, labelsize=labsize) \n  ax2.set_ylabel(r'$\\mathrm{V_{LOS}}$ (km/s)',fontsize=labsize+2) \n\n  return s,x,y\n\n\n# Beginning position-velocity plot\nfor k in range (len(files_mod)): \n  image_mod = fits.open(outfolder+files_mod[k]) \n  data_mod = image_mod[0].data[zmin:zmax+1,ymin:ymax+1,xmin:xmax+1]\n\n  fig, axes = defineaxis(2,1,1,0.7,xsep=0,ysep=0.1,fig_width=9,fig_heigth=9)\n  axes = np.ravel(axes)\n\n  # Major axis\n  s,x,y = plot_pv(axes[0],rad,xcen_m,ycen_m,pa)\n  axes[0].text(0, 1.05, gname, transform=axes[0].transAxes,fontsize=32) \n\n  fig.canvas.draw()\n  renderer = fig.canvas.get_renderer() \n  bbox = axes[0].get_tightbbox(renderer).transformed(fig.transFigure.inverted())\n  pos = axes[0].get_position()\n  ax_m = fig.add_axes([bbox.x1+0.14,pos.y1-0.4,0.4,0.4])\n\n  ax_m.tick_params(labelbottom=False,labelleft=False,top=False,bottom=False,left=False,right=False) \n  ax_m.imshow(mom0p,origin='lower',cmap='Greys',aspect='auto',interpolation='nearest',norm=norm0,extent=extm)\n  ax_m.imshow(mom1p,origin='lower',cmap='seismic',aspect='auto',interpolation='nearest',norm=norm1,extent=extm, alpha=0.4)\n  ax_m.contour(mom1p,levels=[vsys_m],extent=extm, colors='forestgreen',linewidths=3)\n  ax_m.plot(xcen_m+pad_x[0],ycen_m+pad_y[0],'x',color='y',markersize=7,mew=1.5,zorder=10)\n  ax_m.annotate('',xy=(0.8,-0.1),xycoords='axes fraction',xytext=(1.0, -0.1),arrowprops=dict(arrowstyle='<->', color='k', lw=1.5))\n  ax_m.text(0.9,-0.16,f\"{0.2*xsize*cdeltsp:.0f}''\",transform=ax_m.transAxes,fontsize=12, ha='center')\n  ax_m.axvline(xcen_m+pad_x[0],ls='dotted',c='k',alpha=0.2)\n  ax_m.axhline(ycen_m+pad_y[1],ls='dotted',c='k',alpha=0.2)\n  _,ax_m_xmax = ax_m.get_xlim()\n  _,ax_m_ymax = ax_m.get_ylim()\n  m = (x >= 0) & (x < ax_m_xmax) & (y >= 0) & (y < ax_m_ymax)\n  ax_m.plot(x[m]+pad_x[0],y[m]+pad_y[0],'-',lw=1.5,c='#B22222') \n  ax_m.plot(x[~m]+pad_x[0],y[~m]+pad_y[0],'-',lw=1.5,c='#B22222',alpha=0.3) \n\n  # Plotting rotation curve \n  r = np.sqrt((x-xcen_m)**2+(y-ycen_m)**2)\n  r[s<0] *= -1\n  x_rad = np.interp(radii/cdeltsp,r,x)\n  y_rad = np.interp(radii/cdeltsp,r,y)\n  s_rad = np.interp(radii/cdeltsp,r,s)\n  axes[0].plot(s_rad*cdeltsp,vlos,'o',c='gold',ms=10,alpha=0.8,mec='cornsilk') \n  ax_m.scatter(x_rad+pad_x[0],y_rad+pad_y[0],c='gold',s=10,alpha=0.8) \n\n  # Minor axis\n  s,x,y = plot_pv(axes[1],rad*np.cos(np.radians(inc)),xcen_m,ycen_m,pa+90)\n  m = (x >= 0) & (x < ax_m_xmax) & (y >= 0) & (y < ax_m_ymax)\n  ax_m.plot(x[m]+pad_x[0],y[m]+pad_y[0],'--',lw=1.5,c='navy') \n  ax_m.plot(x[~m]+pad_x[0],y[~m]+pad_y[0],'--',lw=1.5,c='navy',alpha=0.3) \n\n  xran_min = rad[-1]*np.sin(np.radians(inc_m))\n  axes[1].set_xlim(-1.3*xran_min,1.3*xran_min)\n\n  outfile = '%s_pv'%outprefix\n  ntype   = files_mod[k][files_mod[k].rfind('_'):].replace('.fits','') \n  fig.savefig(outfolder+outfile+ntype+'.pdf', bbox_inches='tight') \n  image_mod.close() \n\nimage.close() \nimage_mas.close() \n")
    flush("plot_pvs.py")

    # ---------------------------------------------------------------- plot_pvslices.py
    scriptnames.append("plot_pvslices.py")
    w("#######################################################################\n#### This script writes a plot of P-V slices across several cuts   ####\n#######################################################################\nimport numpy as np \nimport os \nimport matplotlib as mpl \nimport matplotlib.pyplot as plt \nfrom plot_utils import *\nfrom astropy.io import fits \nfrom astropy.visualization import ImageNormalize, PercentileInterval, PowerStretch\n\nlabsize = 13\nmpl.rc('xtick',direction='in',top=True) \nmpl.rc('ytick',direction='in',right=True) \nmpl.rcParams['contour.negative_linestyle'] = 'solid' \nplt.rc('font',family='sans-serif',serif='Helvetica',size=labsize) \nparams = {'text.usetex': False, 'mathtext.fontset': 'cm', 'mathtext.default': 'regular'} \nplt.rcParams.update(params) \n\ngname = '",
      F(gname),
      "' \noutfolder = '",
      F(outfolder),
      "' \noutprefix = '",
      F(outprefix),
      "' \ntwostage = ",
      F(twostage),
      " \nplotmask = ",
      F(plotmask),
      " \n\nimage = fits.open('",
      F(cubefile),
      "') \nimage_mas = fits.open(outfolder+outprefix+'_mask.fits') \nxmin, xmax = ",
      F(xmin),
      ", ",
      F(xmax),
      "\nymin, ymax = ",
      F(ymin),
      ", ",
      F(ymax),
      "\nzmin, zmax = ",
      F(zmin),
      ", ",
      F(zmax),
      "\nzmin_wcs, zmax_wcs = ",
      F(zmin_wcs),
      ", ",
      F(zmax_wcs),
      "\ndata = image[" + str(cubehdu) + "].data[")
    if h.NumAx() > 3:
        for i in range(h.NumAx() - 3):
            w("0,")
    w("zmin:zmax+1,ymin:ymax+1,xmin:xmax+1] \ndata_mas = image_mas[0].data[zmin:zmax+1,ymin:ymax+1,xmin:xmax+1] \nzsize, ysize, xsize = data.shape \npad_x, pad_y = calculate_padding(xsize,ysize) \ncdeltsp=",
      F(pixscale_arcs),
      "\ncont = ",
      F(cont),
      "\nv = np.array([1,2,4,8,16,32,64])*cont \nv_neg = [-cont] \nmom0 = fits.open(outfolder+'/maps/'+outprefix+'_0mom.fits')[0].data[ymin:ymax+1,xmin:xmax+1] \nmom0p = np.pad(mom0,(pad_y, pad_x),mode='constant',constant_values=np.nan) \nextm = [0,mom0p.shape[1]-1,0,mom0p.shape[0]-1]\nnorm = ImageNormalize(data, interval=PercentileInterval(99.9), stretch=PowerStretch(1.0))\nnorm0 = ImageNormalize(mom0, interval=PercentileInterval(99.0), stretch=PowerStretch(0.5))\n\nfrile = outfolder + outprefix + ('_rings_final2.txt' if twostage else '_rings_final1.txt')\nrings = np.genfromtxt(frile,usecols=" + _uc("rad", "vrot", "disp", "inc", "pa", "xpos", "ypos", "vsys") + ",unpack=True)\nrad,vrot,vdisp,inc,pa,xpos,ypos,vsys = rings\nxcen_m,ycen_m,vsys_m,inc_m,pa_m = np.nanmean((xpos,ypos,vsys,inc,pa),axis=1) \nxcen_m,ycen_m = xcen_m-xmin, ycen_m-ymin \nproj_vmax = np.nanmax(vrot)*np.sin(np.radians(inc_m))\nmax_vdisp = np.nanmax(vdisp)\n\ns_maj,x_maj,y_maj = build_path(xcen_m, ycen_m, rad/cdeltsp, np.full(len(rad),pa_m), xsize, ysize)\ns_min,x_min,y_min = build_path(xcen_m, ycen_m, rad/cdeltsp, np.full(len(rad),pa_m+90), xsize, ysize)\ns_maj = s_maj*cdeltsp\ns_min = s_min*cdeltsp\nmaj_offsets = np.linspace(0,rad[-1],3,endpoint=True)\nmin_offsets = np.linspace(0,rad[-1]*np.cos(np.radians(inc_m)),3,endpoint=True)\nmaj_offsets = np.concatenate([-maj_offsets[1:][::-1],maj_offsets,])\nmin_offsets = np.concatenate([-min_offsets[1:][::-1],min_offsets,])\nx_maj_s = np.interp(min_offsets, s_min, x_min)\ny_maj_s = np.interp(min_offsets, s_min, y_min)\nx_min_s = np.interp(maj_offsets, s_maj, x_maj)\ny_min_s = np.interp(maj_offsets, s_maj, y_maj)\n\nfiles_mod = [f for f in sorted(os.listdir(outfolder)) if outprefix+'_mod' in f] \nif len(files_mod)==0: raise FileNotFoundError('ERROR: no model in output directory') \n\ncmap = plt.cm.PuBu_r \ncolors = cmap(np.linspace(0, 1, 13)) \n\n\ndef plot_pv(i,ax,ax_m,x0,y0,theta,**kwargs):\n\n  s,x,y = build_path(x0, y0, rad/cdeltsp, np.full(len(rad),theta), xsize, ysize)\n  pv_data = extract_pv(data,x,y)\n  pv_mod  = extract_pv(data_mod,x,y)\n  s_arcsec = s * cdeltsp\n  ext = [s_arcsec[0], s_arcsec[-1], zmin_wcs-vsys_m,zmax_wcs-vsys_m]\n\n  ax.imshow(pv_data,origin='lower',cmap='Blues',norm=norm,aspect='auto',interpolation='nearest',extent=ext)\n  ax.contour(pv_data,v,origin='lower',linewidths=0.7,colors='#00008B',extent=ext) \n  ax.contour(pv_data,v_neg,origin='lower',linewidths=0.1,colors='gray',extent=ext) \n  ax.contour(pv_mod,v,origin='lower',linewidths=1.2,colors='#B22222',extent=ext) \n  ax.axhline(y=0,color='black') \n  ax.axvline(x=0,color='black') \n  ax.grid(color='gray', linestyle='--', linewidth=0.3) \n\n  if plotmask:\n    pv_mask = extract_pv(data_mas,x,y) \n    ax.contour(pv_mask,levels=[0],origin='lower',linewidths=1.5,colors='k',extent=ext)\n\n  ax.set_xlim(-1.3*rad[-1],1.3*rad[-1])\n  #ax.set_ylim(-1.1*(proj_vmax+2*max_vdisp),1.1*(proj_vmax+2*max_vdisp))\n  ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=5))\n  ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=5))\n  if i==0 or i==5: \n    ax.tick_params(labelbottom=True,labelleft=True)\n    ax.set_ylabel(r'$\\mathrm{\\Delta V_{LOS}}$ (km/s)',fontsize=labsize+2) \n  else: ax.tick_params(labelbottom=True,labelleft=False)\n  ax.set_xlabel('Offset (arcsec)',fontsize=labsize)\n  ax_m.plot(x+pad_x[0],y+pad_y[0],**kwargs) \n\n")
    w("# Beginning position-velocity plot \nfor k in range (len(files_mod)): \n  image_mod = fits.open(outfolder+files_mod[k]) \n  data_mod = image_mod[0].data[zmin:zmax+1,ymin:ymax+1,xmin:xmax+1]\n\n  fig, axes = defineaxis(4,5,0.3,0.3,xsep=0.01,ysep=0.1,fig_width=10,fig_heigth=10)\n  axes = np.ravel(axes)\n\n  pos = axes[4].get_position()\n  ax_m1 = fig.add_axes([pos.x1+0.03,pos.y0-0.2,0.3,0.3])\n  pos = axes[14].get_position()\n  ax_m2 = fig.add_axes([pos.x1+0.03,pos.y0-0.2,0.3,0.3])\n\n  # Plotting moment map\n  pm = np.pad(mom0,(pad_y, pad_x),mode='constant',constant_values=np.nan) \n  for ax in [ax_m1,ax_m2]:\n    ax.set_prop_cycle(color=colors)\n    ax.tick_params(labelbottom=False,labelleft=False,top=False,bottom=False,left=False,right=False)\n    ax.imshow(pm,origin='lower',cmap='Greys',aspect='auto',interpolation='nearest',norm=norm0,extent=extm)\n    ax.set_title('Slices')\n    ax.plot(xcen_m+pad_x[0],ycen_m+pad_y[0],'x',color='yellow',markersize=7,mew=1.5,zorder=12) \n    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=5))\n    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=5))\n\n  pos = ax_m2.get_position()\n  ax_m2.set_position([pos.x0,pos.y0 - 0.07,pos.width,pos.height])\n\n  dtheta = 18\n  for i,ax in enumerate(axes[0:10]):\n    theta = pa_m + i*dtheta  \n    ls = 'solid' if i<5 else 'dashed'\n    plot_pv(i,ax,ax_m1,xcen_m,ycen_m,theta,ls=ls,label=rf'{i*dtheta:.0f}$^\\circ$')\n    ax.text(0.03,1.03,rf\"$\\theta$ = {i*dtheta:.0f}$^\\circ$\",transform=ax.transAxes,fontsize=labsize)\n    theta += 15.\n\n  for i,ax in enumerate(axes[10:]):\n    pos = ax.get_position()\n    ax.set_position([pos.x0,pos.y0 - 0.05,pos.width,pos.height])\n\n    if i<5:\n      xi,yi,off,theta = x_maj_s[i],y_maj_s[i],min_offsets[i],pa_m\n      ls = 'solid'\n    else:\n      xi,yi,off,theta = x_min_s[i-5],y_min_s[i-5],maj_offsets[i-5],pa_m + 90\n      ls = 'dashed'\n\n    plot_pv(i,ax,ax_m2,xi,yi,theta,ls=ls,label=rf\"{off:.1f}''\")\n    ax.text(0.03,1.03,rf\"offset = {off:.1f}''\",transform=ax.transAxes,fontsize=labsize)\n    ax_m2.plot(xi+pad_x[0],yi+pad_y[0],'o',c='royalblue',zorder=11)\n  leg_opts = dict(ncol=2,loc='upper left',bbox_to_anchor=(0.0, -0.05),frameon=False)\n  ax_m1.legend(**leg_opts,handlelength=3.3)\n  ax_m2.legend(**leg_opts,handlelength=2)\n\n  outfile = '%s_pvslices'%outprefix \n  ntype   = files_mod[k][files_mod[k].rfind('_'):].replace('.fits','') \n  fig.savefig(outfolder+outfile+ntype+'.pdf', bbox_inches='tight') \n  image_mod.close() \n\nimage.close() \nimage_mas.close() \n")
    flush("plot_pvslices.py")

    # ---------------------------------------------------------------- plot_kinmaps.py
    scriptnames.append("plot_kinmaps.py")
    w("#######################################################################\n#### This script writes a plot of kinematic maps of model and data ####\n#######################################################################\nimport numpy as np \nimport os \nimport matplotlib as mpl \nimport matplotlib.pyplot as plt \nfrom matplotlib.colorbar import ColorbarBase \nfrom plot_utils import calculate_padding, defineaxis \nfrom astropy.io import fits \nfrom astropy.visualization import PercentileInterval \nfrom copy import copy \nmpl.rc('xtick',direction='in') \nmpl.rc('ytick',direction='in') \nplt.rc('font',family='sans-serif',serif='Helvetica',size=10) \nparams = {'text.usetex': False, 'mathtext.fontset': 'cm', 'mathtext.default': 'regular'} \nplt.rcParams.update(params) \n\ngname = '",
      F(gname),
      "' \noutfolder = '",
      F(outfolder),
      "' \noutprefix = '",
      F(outprefix),
      "' \ntwostage = ",
      F(twostage),
      " \nxmin, xmax = ",
      F(xmin),
      ", ",
      F(xmax),
      "\nymin, ymax = ",
      F(ymin),
      ", ",
      F(ymax),
      "\n\n# Opening maps and retrieving intensity map units\nf0 = fits.open(outfolder+'/maps/'+outprefix+'_0mom.fits') \nf1 = fits.open(outfolder+'/maps/'+outprefix+'_1mom.fits') \nf2 = fits.open(outfolder+'/maps/'+outprefix+'_2mom.fits') \nbunit = f0[0].header['BUNIT'] \nbunit = bunit.replace(' ', '').lower() \n# Now plotting moment maps \nmom0 = f0[0].data[ymin:ymax+1,xmin:xmax+1] \nmom1 = f1[0].data[ymin:ymax+1,xmin:xmax+1] \nmom2 = f2[0].data[ymin:ymax+1,xmin:xmax+1] \nmaskmap = np.copy(mom1) \nmaskmap[mom1==mom1] = 1 \nysize, xsize = mom0.shape \npad_x, pad_y = calculate_padding(xsize,ysize) \n\nif twostage: rad,inc,pa,xpos,ypos,vsys = np.genfromtxt(outfolder+outprefix+'_rings_final2.txt',usecols=" + _uc("rad", "inc", "pa", "xpos", "ypos", "vsys") + ",unpack=True) \nelse: rad,inc,pa,xpos,ypos,vsys = np.genfromtxt(outfolder+outprefix+'_rings_final1.txt',usecols=" + _uc("rad", "inc", "pa", "xpos", "ypos", "vsys") + ",unpack=True) \nxcen_m,ycen_m,inc_m,pa_m,vsys_m=np.nanmean((xpos,ypos,inc,pa,vsys),axis=1) \nxcen, ycen = xcen_m-xmin+pad_x[0], ycen_m-ymin+pad_y[0] \n\nfiles_mod0 = [f for f in sorted(os.listdir(outfolder+'maps/')) if outprefix+'_mod_0mom' in f] \nfiles_mod1 = [f for f in sorted(os.listdir(outfolder+'maps/')) if outprefix+'_mod_1mom' in f] \nfiles_mod2 = [f for f in sorted(os.listdir(outfolder+'maps/')) if outprefix+'_mod_2mom' in f] \nif len(files_mod0)==0 or len(files_mod1)==0 or len(files_mod2)==0 : raise FileNotFoundError('ERROR: no model maps in output directory') \n\n\ncmaps = [plt.get_cmap('Spectral_r'),plt.get_cmap('coolwarm'),plt.get_cmap('PuOr_r')] \nbarlab = ['Intensity ('+bunit+')', r'V$_\\mathrm{LOS}$ (km/s)', r'$\\sigma_\\mathrm{obs}$ (km/s)'] \nbarlab2 = [r'I$_\\mathrm{res}$ ('+bunit+')', r'V$_\\mathrm{res}$ (km/s)', r'$\\sigma_\\mathrm{res}$ (km/s)'] \ntitles = ['DATA', 'MODEL','RESIDUALS'] \nmapname = " + repr(_MAPNAMES[p.ma.maptype == "GAUSSIAN"]) + " \nx = np.arange(0,xmax-xmin,0.1) \ny = np.tan(np.radians(pa_m-90))*(x-xcen)+ycen \next = [0,xmax-xmin+pad_x[0]+pad_x[1],0, ymax-ymin+pad_y[0]+pad_y[1]] \nrad_pix = rad/",
      F(pixscale_arcs),
      "\ntry: nr = len(rad_pix) \nexcept: nr = 1 \ninterval = PercentileInterval(99.5) \n\nfor k in range (len(files_mod0)): \n  mom0_mod = fits.open(outfolder+'/maps/'+files_mod0[k])[0].data[ymin:ymax+1,xmin:xmax+1] \n  mom1_mod = fits.open(outfolder+'/maps/'+files_mod1[k])[0].data[ymin:ymax+1,xmin:xmax+1] \n  mom2_mod = fits.open(outfolder+'/maps/'+files_mod2[k])[0].data[ymin:ymax+1,xmin:xmax+1] \n  to_plot = [[mom0,mom1-vsys_m,mom2],[mom0_mod,mom1_mod-vsys_m,mom2_mod],[mom0-mom0_mod,mom1-mom1_mod,mom2-mom2_mod]] \n\n  nrows, ncols, x_len, y_len = 3, 3, 0.2, 0.2 \n  fig, ax = defineaxis(nrows,ncols,x_len,y_len,xsep=0.00,ysep=0.08,fig_width=11,fig_heigth=11) \n\n  for i in range (ax.shape[0]): \n    cmap = copy(cmaps[i]) \n    cmap.set_bad('w',1.) \n    vmin, vmax = interval.get_limits(to_plot[1][i]) \n    vmin, vmax = (-1.1*np.nanmax(vmax),1.1*np.nanmax(vmax)) if i==1 else (vmin,vmax) \n    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax) \n    cbax = fig.add_axes([ax[i][0].get_position().x0,ax[i][0].get_position().y0-0.025,2*x_len-0.003,0.02]) \n    cb1 = ColorbarBase(cbax, orientation='horizontal', cmap=cmap, norm=norm) \n    cb1.set_label(barlab[i],fontsize=13) \n    for j in range (ax.shape[1]): \n      axis = ax[i][j] \n      axis.tick_params(labelbottom=False,labelleft=False,right=True,top=True) \n      axis.set_xlim(ext[0],ext[1]) \n      axis.set_ylim(ext[2],ext[3]) \n      if j==2: \n        vmax = np.nanmax(interval.get_limits(to_plot[j][i])) \n        norm = mpl.colors.Normalize(vmin=-vmax, vmax=vmax) \n      pp = np.pad(to_plot[j][i]*maskmap,(pad_y, pad_x),mode='constant',constant_values=np.nan) \n      axis.imshow(pp,origin='lower',cmap=cmap,norm=norm,aspect='auto',extent=ext,interpolation='nearest') \n      axis.plot(xcen,ycen,'x',color='#000000',markersize=7,mew=1.5,zorder=10) \n\n      if i==0: \n        axis.text(0.5,1.05,titles[j],ha='center',transform=axis.transAxes,fontsize=15) \n        axis.plot(x,y,'--',color='k',linewidth=1) \n        if j!=2 and nr>3:  \n          axmaj = rad_pix[-1] \n          axmin = axmaj*np.cos(np.radians(inc_m))  \n          posa = np.radians(pa_m-90)  \n          t = np.linspace(0,2*np.pi,100)  \n          xt = xcen+axmaj*np.cos(posa)*np.cos(t)-axmin*np.sin(posa)*np.sin(t)  \n          yt = ycen+axmaj*np.sin(posa)*np.cos(t)+axmin*np.cos(posa)*np.sin(t)  \n          axis.plot(xt,yt,'-',c='k',lw=0.8)  \n      elif i==1: \n        axis.plot(x,y,'--',color='k',linewidth=1) \n        if nr<10: \n          x_pix = rad_pix*np.cos(np.radians(pa_m-90)) \n          y_pix = rad_pix*np.sin(np.radians(pa_m-90)) \n          axis.scatter(x_pix+xcen,y_pix+ycen,c='grey',s=12) \n          axis.scatter(xcen-x_pix,ycen-y_pix,c='grey',s=12) \n        if nr>5 and not all(np.diff(pa)==0): \n          x_pix = rad_pix*np.cos(np.radians(pa-90)) \n          y_pix = rad_pix*np.sin(np.radians(pa-90)) \n          axis.plot(xcen-x_pix,ycen-y_pix,'-',color='grey',lw=1) \n          axis.plot(x_pix+xcen,y_pix+ycen,'-',color='grey',lw=1) \n        if j!=2: \n          cmax = np.nanmax(np.abs(to_plot[0][i])) \n          levels = np.linspace(0.166 * cmax, cmax, 6) \n          axis.contour(pp,levels=[0],colors='forestgreen',origin='lower',extent=ext) \n          axis.contour(pp,levels=-levels[::-1],colors='navy',linewidths=0.7,origin='lower',extent=ext,linestyles='solid') \n          axis.contour(pp,levels=levels,colors='darkred',linewidths=0.7,origin='lower',extent=ext) \n      if j==0: axis.text(-0.12,0.5,mapname[i],va='center',rotation=90,transform=axis.transAxes,fontsize=15) \n\n    cbax = fig.add_axes([ax[i][2].get_position().x0+0.003,ax[i][2].get_position().y0-0.025,x_len-0.003,0.02]) \n    cb2 = ColorbarBase(cbax, orientation='horizontal', cmap=cmap, norm=norm) \n    cb2.set_label(barlab2[i],fontsize=13) \n    cb2.ax.locator_params(nbins=3) \n    for c in [cb1,cb2]: \n      c.solids.set_edgecolor('face') \n      c.outline.set_linewidth(0) \n\n  outfile = '%s_maps'%outprefix \n  ntype   = files_mod0[k][files_mod0[k].rfind('_'):].replace('.fits','') \n  fig.savefig(outfolder+outfile+ntype+'.pdf', bbox_inches = 'tight') \n\n")
    flush("plot_kinmaps.py")

    # ---------------------------------------------------------------- plot_all.py
    w("########################################################################\n"
      "#### This script simply calls all other python scripts for plotting ####\n"
      "########################################################################\n"
      "# Run it from the folder where BBarolo was run. plot_pvs_old.py (the previous style of\n"
      "# PV plot, which writes the same file as plot_pvs.py) is not run: run it by hand if needed.\n"
      "import os, sys, subprocess \n\n"
      "scriptdir = '", F(outfolder), "plotscripts/' \n"
      "skip = ('plot_all.py', 'plot_utils.py', 'plot_pvs_old.py') \n"
      "paths = [os.path.join(scriptdir, f) for f in sorted(os.listdir(scriptdir)) if f.endswith('.py') and f not in skip] \n\n"
      "if hasattr(os, 'fork'): \n"
      "  # Load the plotting libraries once, then run each script in a forked process \n"
      "  import runpy \n"
      "  import matplotlib \n"
      "  matplotlib.use('Agg') \n"
      "  import matplotlib.pyplot, astropy.io.fits, astropy.visualization, scipy.ndimage, scipy.interpolate \n"
      "  pids = [] \n"
      "  for p in paths: \n"
      "    sys.stdout.flush(); sys.stderr.flush() \n"
      "    pid = os.fork() \n"
      "    if pid == 0: \n"
      "      code = 0 \n"
      "      try: \n"
      "        sys.argv = [p] \n"
      "        sys.path.insert(0, os.path.dirname(os.path.abspath(p))) \n"
      "        runpy.run_path(p, run_name='__main__') \n"
      "      except BaseException: \n"
      "        import traceback; traceback.print_exc(); code = 1 \n"
      "      sys.stdout.flush(); sys.stderr.flush() \n"
      "      os._exit(code) \n"
      "    pids.append(pid) \n"
      "  for pid in pids: os.waitpid(pid, 0) \n"
      "else: \n"
      "  procs = [subprocess.Popen([sys.executable, p]) for p in paths] \n"
      "  for pr in procs: pr.wait() \n")
    flush("plot_all.py")

    return scriptnames


def write_sources_script(cube):
    """Write the plot_sources.py script for the SEARCH task (port of Cube::plotDetections)."""
    h = cube.Head()
    p = cube.pars()
    bmaj = float(np.float32(h.Bmaj() / h.PixScale()))
    bmin = float(np.float32(h.Bmin() / h.PixScale()))
    bpa = float(np.float32(h.Bpa()))
    bu = "(" + h.Bunit() + "*km/s)"
    if p.fluxConvert and FluxtoJyBeam(1, h) != 1:
        bu = "(Jy*km/s)"
    outfolder = p.outFolder + "sources/"
    mkdirp(outfolder)
    txt = ("#########################################################################\n"
           "#### This script writes single/global plots of all detected sources  ####\n"
           "#########################################################################\n"
           "import os \nimport numpy as np \nimport matplotlib.pyplot as plt \n"
           "from matplotlib.patches import Ellipse \nfrom astropy.io import fits \nfsize=10 \n"
           "plt.rc('font',family='sans-serif',serif='Helvetica',size=fsize) \n\n"
           f"outdir = '{outfolder}' \ngname  = '{p.outPrefix}' \nsingle = True \n\n\n"
           "def addrowplot(thisfig,nrow): \n"
           "  xlen, ylen, xsep, ysep = 0.2,0.2,0.03,0.04 \n"
           "  ystart = 1-nrow*(ylen+ysep) \n"
           "  for i in range(4): \n"
           "    xstart = i*(xlen+xsep) \n"
           "    thisfig.add_axes([xstart,ystart,xlen,ylen]) \n"
           "    thisfig.add_axes([xstart,ystart-0.005,xlen,0.015]) \n"
           "  thisfig.delaxes(thisfig.axes[-1]) \n\n\n"
           "def plot_source(ax,m0,m1,m2,spec): \n\n"
           "  for a in ax: \n"
           "    a.tick_params(right=True,top=True,labelleft=False,labelbottom=False,direction='in',labelsize=fsize) \n"
           "  common = dict(origin='lower',aspect='auto',extent=[-10,10,-10,10],interpolation='nearest') \n\n"
           "  # Plotting intensity \n"
           "  im = ax[0].imshow(m0,**common,cmap=plt.get_cmap('Spectral_r')) \n"
           "  cb = plt.colorbar(im,cax=ax[1],orientation='horizontal') \n"
           "  ax[0].text(-0.1,0.5,'%s'%s,ha='center',va='center',rotation=90,transform=ax[0].transAxes,fontsize=fsize+2) \n"
           f"  bmaj, bmin, bpa = {F(bmaj)}/m0.shape[0], {F(bmin)}/m0.shape[1],{F(bpa)}\n"
           "  beam = Ellipse((0.95-bmaj/2.,0.1+bmaj/2.),width=bmaj,height=bmin,angle=bpa+90,color='gray',clip_on=False,transform=ax[0].transAxes,alpha=0.2)\n"
           "  ax[0].add_artist(beam)\n\n"
           "  # Plotting velocity \n"
           "  im = ax[2].imshow(m1,**common,cmap=plt.get_cmap('RdBu_r',25)) \n"
           "  cb = plt.colorbar(im,cax=ax[3],orientation='horizontal') \n\n"
           "  # Plotting dispersion \n"
           "  im = ax[4].imshow(m2,**common,cmap=plt.get_cmap('PuOr_r')) \n"
           "  cb = plt.colorbar(im,cax=ax[5],orientation='horizontal') \n\n"
           "  # Plotting spectrum \n"
           "  vel,val = spec \n"
           "  ax[6].plot(vel,val,'o-',c='gray',ms=5) \n"
           "  ax[6].tick_params(labelright=True,labelbottom=True) \n\n"
           "  for a in ax[1::2]: a.locator_params(nbins=4) \n\n\n"
           "def setaxislabels(ax): \n"
           f"  ax[0].text(0.5,-0.2,'Int flux {bu}',ha='center',transform=ax[0].transAxes,fontsize=fsize+1) \n"
           "  ax[1].text(0.5,-0.2,'Velocity (km/s)',ha='center',transform=ax[1].transAxes,fontsize=fsize+1) \n"
           "  ax[2].text(0.5,-0.2,'Line width (km/s)',ha='center',transform=ax[2].transAxes,fontsize=fsize+1) \n"
           "  ax[3].text(0.5,-0.2,'Velocity (km/s)',ha='center',transform=ax[3].transAxes,fontsize=fsize+1) \n\n\n"
           "# Start plotting \n"
           "fig = plt.figure(figsize=(10,10)) \n"
           "for i,s in enumerate(sorted(os.listdir(outdir))): \n"
           "  if not os.path.isdir('%s/%s'%(outdir,s)): continue \n"
           "  prefix = '%s/%s/%s'%(outdir,s,s) \n\n"
           "  # Read in necessary files \n"
           "  m0 = fits.open('%s_mom0.fits'%prefix)[0].data \n"
           "  m1 = fits.open('%s_mom1.fits'%prefix)[0].data \n"
           "  m2 = fits.open('%s_mom2.fits'%prefix)[0].data \n"
           "  vel, val = np.genfromtxt('%s_spectrum.dat'%prefix,unpack=True) \n"
           "  ysize, xsize = m0.shape \n"
           "  n = max(ysize, xsize) \n"
           "  pad_y = ((n - ysize) // 2, n - ysize - (n - ysize) // 2) \n"
           "  pad_x = ((n - xsize) // 2, n - xsize - (n - xsize) // 2) \n"
           "  m0 = np.pad(m0,(pad_y, pad_x),mode='constant',constant_values=np.nan) \n"
           "  m1 = np.pad(m1,(pad_y, pad_x),mode='constant',constant_values=np.nan) \n"
           "  m2 = np.pad(m2,(pad_y, pad_x),mode='constant',constant_values=np.nan) \n\n"
           "  # Plotting summary plot \n"
           "  addrowplot(fig,i) \n"
           "  ax = fig.axes[-7:] \n"
           "  plot_source(ax,m0,m1,m2,(vel,val)) \n\n"
           "  if single: \n"
           "    # Plotting for a single source \n"
           "    fig_sin = plt.figure(figsize=(10,10)) \n"
           "    addrowplot(fig_sin,0) \n"
           "    plot_source(fig_sin.axes,m0,m1,m2,(vel,val)) \n"
           "    setaxislabels(fig_sin.axes[-7::2]) \n"
           "    fig_sin.savefig('%s.pdf'%prefix,bbox_inches='tight') \n"
           "    plt.close(fig_sin) \n\n"
           "setaxislabels(fig.axes[-7::2]) \n"
           "fig.savefig('%s/%s_sources.pdf'%(outdir,gname),bbox_inches='tight') \n\n")
    path = outfolder + "plot_sources.py"
    with open(path, "w") as f:
        f.write(txt)
    return path
