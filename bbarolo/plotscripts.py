"""
Running the plotting scripts; the scripts of 3DFIT (port of Galfit::writeScripts_Python in galfit_out.cpp)
and of the SEARCH task (Cube::plotDetections).

As in BBarolo, the scripts are written in the ``plotscripts`` sub-directory of the output
folder (so that they can be edited and re-run by the user) and then executed in parallel.
The plots of 3DFIT are functions of bbarolo/plots/fit3d.py, callable in Python,
whose source is copied in their scripts (plots/scripts.py). To avoid
loading matplotlib & co. once per script, the scripts run in processes forked from a
helper process that has already imported them (PlotRunner, _plotserver.py).
"""

import os
import sys
import json
import subprocess
import numpy as np

from .bbutils import FluxtoJyBeam, mkdirp

# The columns of the ring files read by the scripts (0-based), from galfit.RING_COLUMNS
_RING_KEYS = ("rad", "vrot", "disp", "inc", "pa", "z0", "dens", "xpos", "ypos", "vsys", "vrad")


def _uc(*keys):
    """The usecols of the ring parameters keys (e.g. 'rad', 'vrot') in a ring file, as text: '(0,1)'."""
    from .galfit import RING_COLUMNS
    assert len(_RING_KEYS) == len(RING_COLUMNS)
    return "(" + ",".join(str(_RING_KEYS.index(k)) for k in keys) + ")"


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


def writeScripts_Python(gf):
    """Write the plotting scripts of the fit gf (a Galfit, also that of METHOD GLOBAL or BAYESIAN) in <outfolder>/plotscripts/:
    those of plots/scripts.py (plot_utils.py, plot_parameters.py, plot_chanmaps.py, plot_pvs.py, plot_pvslices.py,
    plot_kinmaps.py), plot_pvs_old.py (the previous style of PV plot, not run) and plot_all.py. Returns the names
    of the scripts to run."""
    from .plots.scripts import plot_settings, write_fit3d_scripts
    s = plot_settings(gf)
    scriptnames = write_fit3d_scripts(gf, s)
    gname, outfolder, outprefix = s["gname"], s["outfolder"], s["outprefix"]
    twostage, plotmask, cont, disp, arcconv = s["twostage"], s["plotmask"], s["cont"], s["disp"], s["arcconv"]
    zmin, zmax = s["box"][4:]
    zmin_wcs, zmax_wcs = s["vrange"]
    pa_av, pa_min, reverse = s["pa_av"], s["pa_min"], s["reverse"]
    folder = outfolder + "plotscripts/"
    buf = []

    def w(*parts):
        buf.append("".join(parts))

    def flush(name):
        with open(folder + name, "w") as f:
            f.write("".join(buf))
        buf.clear()

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
