"""
The plotting scripts of WINDFIT, in the style of those of 3DFIT (bbarolo/plotscripts.py) and with their helpers
(plot_utils.py): written in <outfolder>/plotscripts/ by BayesianWind.write and run with PLOTS.

  plot_chanmaps_wind.py  channel maps of the data with the contours of the model -> <prefix>_chanmaps_wind.pdf
  plot_kinmaps_wind.py   moment maps of the data, the model and the residuals    -> <prefix>_maps_wind.pdf
  plot_pvs_wind.py       position-velocity slices along the projected axis of the cones (pvs/<prefix>_pv_b, through
                         the apex) and across each cone at half its length, with the contours of the model
                         -> <prefix>_pv_wind.pdf

The scripts read the outputs of BayesianWind.write (the data cube, the mask, <prefix>_mod_wind.fits, maps/, pvs/);
the geometry of the best fit (the apex, VSYS, the projected axis and the far rims of the cones) is written in them,
as there is no ring file of a wind. They can be edited and run again from their folder.
"""

import math
import numpy as np

from ..plotscripts import run_scripts
from ..plots.scripts import plot_utils_text as _plot_utils_text
from ..bbutils import DeltaVel, AlltoVel, FluxtoJy


def _geometry(f):
    """The projected axis of the cones (pixels: the points at CONELEN along +z and -z), their far rims (the circles of
    radius CONELEN sin(OPENANG/2) at CONELEN cos(OPENANG/2) along the axis), and the slices across each cone at half its
    length (the ends of segments along the x axis of the wind, its PA, out to 1.3 times the radius of the cone there
    plus one beam), for the best wind of f."""
    b = f.best
    su = f._wm.setup
    cd0, cd1 = f._wm.cdelt
    inc = math.radians(b.inc)
    ang = math.radians(b.pa) + su.crota2
    spa, cpa = math.sin(ang), math.cos(ang)
    si, ci = math.sin(inc), math.cos(inc)

    def sky(x, yd, z):
        ys = yd * ci - z * si
        return b.xpos + (x * spa - ys * cpa) / cd0, b.ypos + (x * cpa + ys * spa) / cd1
    cones = f._wm.cells.cones
    axis = [sky(0., 0., c * b.conelen) for c in cones]
    if len(axis) == 1:
        axis.insert(0, sky(0., 0., 0.))                 # (one cone: the axis from the apex)
    th = math.radians(0.5 * b.openang)
    phi = np.linspace(0., 2 * math.pi, 73)
    rims = []
    for c in cones:
        R, z = b.conelen * math.sin(th), c * b.conelen * math.cos(th)
        gx, gy = sky(R * np.cos(phi), R * np.sin(phi), z)
        rims.append((np.round(gx, 2).tolist(), np.round(gy, 2).tolist()))
    x0, y0 = sky(0., 0., 0.)
    ux, uy = sky(1., 0., 0.)
    ux, uy = ux - x0, uy - y0
    nu = math.hypot(ux, uy)
    h = f._cube_obj.Head()
    half = 1.3 * (0.5 * b.conelen * math.tan(th) + h.Bmaj() * 3600.)           # (arcsec)
    cuts = []
    for c in cones:
        cx, cy = sky(0., 0., 0.5 * c * b.conelen)
        cuts.append(((cx - half * ux, cy - half * uy), (cx + half * ux, cy + half * uy)))
    labels = ["across the %s cone, at CONELEN/2 (PA)" % ("receding" if c > 0 else "approaching") for c in cones]
    return axis, rims, [[[float(v) for v in p] for p in cut] for cut in cuts], labels


def _box(f, mod):
    """The box of the plots: the region of the likelihood, the channels with model emission and 3 more on each
    side."""
    x0, x1, y0, y1, k0, k1 = (int(v) for v in f._lim)
    prof = np.nansum(np.abs(mod), axis=(1, 2))
    nz = mod.shape[0]
    if np.any(prof > 1e-3 * prof.max()):
        ks = np.flatnonzero(prof > 1e-3 * prof.max())
        k0, k1 = max(int(ks[0]) - 3, 0), min(int(ks[-1]) + 4, nz)
    return x0, x1 - 1, y0, y1 - 1, k0, k1 - 1


def _lowest_contour(f):
    """The lowest contour of the plots: PLOTMINCON, or as 3DFIT: the threshold of the SEARCH mask (growth or S/N
    cut times the noise), 2.5 times the noise otherwise."""
    c = f._cube_obj
    p = c.pars()
    if p.gf.PLOTMINCON > 0:
        return float(p.gf.PLOTMINCON)
    se = p.se
    if str(p.MaskType).upper() == "SEARCH":
        if se.flagUserGrowthT:
            return float(se.growthThreshold)
        if se.UserThreshold:
            return float(se.threshold)
        return float((se.growthCut if se.flagGrowth else se.snrCut) * f.noiserms)
    return 2.5 * f.noiserms


_IMPORTS = """import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colorbar import ColorbarBase
from plot_utils import calculate_padding, defineaxis
from astropy.io import fits
from astropy.visualization import PowerStretch, PercentileInterval
from astropy.visualization.mpl_normalize import ImageNormalize
from copy import copy
mpl.rcParams['contour.negative_linestyle'] = 'solid'
plt.rc('font',family='sans-serif',serif='Helvetica',size=10)
plt.rcParams.update({'text.usetex': False, 'mathtext.fontset': 'cm', 'mathtext.default': 'regular'})
"""

_CHANMAPS = """
# CHANNEL MAPS of the data (contours: blue) and of the model (red)
image = fits.open(cubefile)
data = image[cubehdu].data
while data.ndim > 3: data = data[0]
data = data[zmin:zmax+1,ymin:ymax+1,xmin:xmax+1]
data_mas = fits.open(outfolder+outprefix+'_mask.fits')[0].data[zmin:zmax+1,ymin:ymax+1,xmin:xmax+1]
data_mod = fits.open(outfolder+outprefix+'_mod_wind.fits')[0].data[zmin:zmax+1,ymin:ymax+1,xmin:xmax+1]
zsize, ysize, xsize = data.shape
pad_x, pad_y = calculate_padding(xsize,ysize)
v = np.array([1,2,4,8,16,32,64])*cont
v_neg = [-cont]
vmax = PercentileInterval(99.5).get_limits(data)[1]
norm = ImageNormalize(vmin=cont, vmax=vmax, stretch=PowerStretch(0.5))
xcen, ycen = xcen_m-xmin+pad_x[0], ycen_m-ymin+pad_y[0]
ax_x = np.array(axis_x)-xmin+pad_x[0]
ax_y = np.array(axis_y)-ymin+pad_y[0]

fig = plt.figure(figsize=(8.27, 11.69), dpi=150)
grid = [gridspec.GridSpec(2,5),gridspec.GridSpec(2,5),gridspec.GridSpec(2,5)]
grid[0].update(top=0.90, bottom=0.645, left=0.05, right=0.95, wspace=0.0, hspace=0.0)
grid[1].update(top=0.60, bottom=0.345, left=0.05, right=0.95, wspace=0.0, hspace=0.0)
grid[2].update(top=0.30, bottom=0.045, left=0.05, right=0.95, wspace=0.0, hspace=0.0)
num = 0
for j in range (0,3):
  for i in range (0,5):
    chan = int(num*(zsize)/15)
    z = np.pad(data[chan],(pad_y, pad_x),mode='constant',constant_values=np.nan)
    z_mod = np.pad(data_mod[chan],(pad_y, pad_x),mode='constant',constant_values=np.nan)
    if np.all(~(z_mod>=v[0])): z_mod[:,:] = 0
    velo = ' v = %d km/s'%int(crval3_kms+(zmin+chan)*cdelt3_kms-vsys_m)
    for row, zz, col in ((0, z, '#00008B'), (1, z_mod, '#B22222')):
      ax = plt.subplot(grid[j][row,i])
      ax.tick_params(axis='both',which='both',bottom=True,top=True,left=True,right=True,direction='in',
                     labelbottom=False,labelleft=False)
      if row==0: ax.set_title(velo, fontsize=10,loc='left')
      ax.imshow(zz,origin='lower',cmap=mpl.cm.Greys,norm=norm,aspect='auto',interpolation='none')
      ax.contour(zz,v,origin='lower',linewidths=0.7,colors=col)
      if row==0: ax.contour(zz,v_neg,origin='lower',linewidths=0.1,colors='gray')
      ax.plot(ax_x,ax_y,'--',color='#0FB05A',lw=0.6)
      ax.plot(xcen,ycen,'x',color='#0FB05A',markersize=7,mew=2)
      ax.set_xlim(0,zz.shape[1]-1)
      ax.set_ylim(0,zz.shape[0]-1)
      if plotmask and row==0:
        z_mas = np.pad(data_mas[chan],(pad_y, pad_x),mode='constant',constant_values=0)
        ax.contour(z_mas,levels=[0],origin='lower',linewidths=2,colors='k')
      if row==0 and j==i==0:
        ax.text(0, 1.4, gname, transform=ax.transAxes,fontsize=15,va='center')
        bm = mpl.patches.Ellipse(xy=(3.5, 1.4), width=bmaj/float(xmax-xmin), height=bmin/float(ymax-ymin),
                                 angle=bpa+90, color='#5605D0', clip_on=False, transform=ax.transAxes, alpha=0.2)
        ax.add_artist(bm)
        ax.text(3.6+bmaj/float(xmax-xmin)/1.8,1.4,'Beam',transform=ax.transAxes,fontsize=11,ha='left',va='center')
      if row==1 and i==0 and j==2:
        clab = r'Contour levels at 2$^n \\, c_{min}$, where $c_{min}$ = %.4g %s and n = 0,1,..,6 '%(cont, bunit)
        ax.text(0.01,-0.16,clab,transform=ax.transAxes,fontsize=11, ha='left',va='center')
    num = num+1
fig.savefig(outfolder+outprefix+'_chanmaps_wind.pdf', orientation='portrait', format='pdf')
image.close()
"""

_KINMAPS = """
# MOMENT MAPS of the data, the model and the residuals
mom = [fits.open(outfolder+'maps/'+outprefix+'_%dmom.fits'%i)[0].data[ymin:ymax+1,xmin:xmax+1] for i in range(3)]
mod = [fits.open(outfolder+'maps/'+outprefix+'_mod_%dmom_wind.fits'%i)[0].data[ymin:ymax+1,xmin:xmax+1]
       for i in range(3)]
bunit_map = fits.open(outfolder+'maps/'+outprefix+'_0mom.fits')[0].header.get('BUNIT', '').replace(' ', '').lower()
maskmap = np.where(np.isfinite(mom[1]), 1., np.nan)
ysize, xsize = mom[0].shape
pad_x, pad_y = calculate_padding(xsize,ysize)
xcen, ycen = xcen_m-xmin+pad_x[0], ycen_m-ymin+pad_y[0]
ax_x = np.array(axis_x)-xmin+pad_x[0]
ax_y = np.array(axis_y)-ymin+pad_y[0]
to_plot = [[mom[0],mom[1]-vsys_m,mom[2]],[mod[0],mod[1]-vsys_m,mod[2]],[mom[0]-mod[0],mom[1]-mod[1],mom[2]-mod[2]]]
cmaps = [plt.get_cmap('Spectral_r'),plt.get_cmap('coolwarm'),plt.get_cmap('PuOr_r')]
barlab = ['Intensity ('+bunit_map+')', r'V$_\\mathrm{LOS}$ - V$_\\mathrm{sys}$ (km/s)',
          r'$\\sigma_\\mathrm{obs}$ (km/s)']
titles = ['DATA', 'MODEL','RESIDUALS']
ext = [0,xsize+pad_x[0]+pad_x[1],0,ysize+pad_y[0]+pad_y[1]]
interval = PercentileInterval(99.5)
fig, ax = defineaxis(3,3,0.2,0.2,xsep=0.00,ysep=0.08,fig_width=11,fig_heigth=11)
for i in range (3):
  cmap = copy(cmaps[i])
  cmap.set_bad('w',1.)
  vmin, vmax = interval.get_limits(to_plot[1][i][np.isfinite(to_plot[1][i])]) if np.any(np.isfinite(to_plot[1][i])) \\
               else (0, 1)
  if i==1: vmin, vmax = -1.1*max(abs(vmin),abs(vmax)), 1.1*max(abs(vmin),abs(vmax))
  norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
  cbax = fig.add_axes([ax[i][0].get_position().x0,ax[i][0].get_position().y0-0.025,2*0.2-0.003,0.02])
  cb1 = ColorbarBase(cbax, orientation='horizontal', cmap=cmap, norm=norm)
  cb1.set_label(barlab[i],fontsize=13)
  for j in range (3):
    axis = ax[i][j]
    axis.tick_params(labelbottom=False,labelleft=False,right=True,top=True,direction='in')
    nrm = norm
    if j==2:
      fin = np.isfinite(to_plot[j][i])
      vv = np.nanmax(np.abs(interval.get_limits(to_plot[j][i][fin]))) if np.any(fin) else 1.
      nrm = mpl.colors.Normalize(vmin=-vv, vmax=vv)
    pp = np.pad(to_plot[j][i]*maskmap,(pad_y, pad_x),mode='constant',constant_values=np.nan)
    axis.imshow(pp,origin='lower',cmap=cmap,norm=nrm,aspect='auto',extent=ext,interpolation='nearest')
    axis.plot(ax_x,ax_y,'--',color='k',linewidth=1)
    for rx, ry in rims:
      axis.plot(np.array(rx)-xmin+pad_x[0],np.array(ry)-ymin+pad_y[0],'-',color='k',lw=0.6,alpha=0.6)
    axis.plot(xcen,ycen,'x',color='#000000',markersize=7,mew=1.5,zorder=10)
    axis.set_xlim(ext[0],ext[1])
    axis.set_ylim(ext[2],ext[3])
    if i==0: axis.text(0.5,1.05,titles[j],ha='center',transform=axis.transAxes,fontsize=15)
ax[0][0].text(0, 1.2, gname, transform=ax[0][0].transAxes,fontsize=18)
fig.savefig(outfolder+outprefix+'_maps_wind.pdf', bbox_inches='tight')
"""

_PVS = """
# POSITION-VELOCITY SLICES: along the projected axis of the cones (pvs/..._pv_b: PA+90, through the apex), and across
# each cone at half its length (along PA); data (grey, blue contours) and model (red contours)
from plot_utils import pv_path, pv_sample
labsize = 14
v = np.array([1,2,4,8,16,32,64])*cont
v_neg = [-cont]
image = fits.open(cubefile)
cube = image[cubehdu].data
while cube.ndim > 3: cube = cube[0]
cube_mod = fits.open(outfolder+outprefix+'_mod_wind.fits')[0].data
cube_mas = fits.open(outfolder+outprefix+'_mask.fits')[0].data
npan = 1 + len(cuts)
fig, ax = defineaxis(npan,1,0.6,0.32,xsep=0.00,ysep=0.08,fig_width=10,
                     fig_heigth=4.5*npan)
ax = np.atleast_1d(ax)
labels = ['along the axis of the cones (PA+90)'] + list(cut_labels)
for i in range(npan):
  if i == 0:
    img = fits.open(outfolder+'pvs/'+outprefix+'_pv_b.fits')
    hd = img[0].header
    dat = img[0].data
    mod = fits.open(outfolder+'pvs/'+outprefix+'_mod_pv_b_wind.fits')[0].data
    mas = fits.open(outfolder+'pvs/'+outprefix+'_mask_pv_b.fits')[0].data
    off = ((np.arange(dat.shape[1])+1-hd['CRPIX1'])*hd['CDELT1']+hd['CRVAL1'])*arcconv
    keep = np.abs(off) <= halfsize
    if keep.sum() < 2: keep[:] = True
    i0, i1 = np.flatnonzero(keep)[[0,-1]]
    dat, mod, mas, off = dat[:,i0:i1+1], mod[:,i0:i1+1], mas[:,i0:i1+1], off[i0:i1+1]
    img.close()
  else:
    p1, p2 = cuts[i-1]
    s, x, y = pv_path(p1=tuple(p1), p2=tuple(p2), shape=cube.shape[1:], step=0.5)
    dat, mod = pv_sample(cube, x, y), pv_sample(cube_mod, x, y)
    mas = pv_sample(cube_mas.astype(float), x, y, order=0)
    off = s*cdeltsp
  dat, mod, mas = dat[zmin:zmax+1], mod[zmin:zmax+1], mas[zmin:zmax+1]
  d_o = abs(off[1]-off[0]) if len(off) > 1 else cdeltsp
  ext = [off[0]-0.5*d_o, off[-1]+0.5*d_o, zmin_wcs-vsys_m, zmax_wcs-vsys_m]
  fin = dat[np.isfinite(dat)]
  vmax = PercentileInterval(99.5).get_limits(fin)[1] if fin.size else 1.
  norm = ImageNormalize(vmin=cont, vmax=max(vmax, 1.01*cont), stretch=PowerStretch(0.5))
  axis = ax[i]
  axis.tick_params(which='major',length=8, labelsize=labsize, direction='in')
  axis.set_xlabel('Offset (arcsec)',fontsize=labsize+2)
  axis.set_ylabel(r'$\\mathrm{\\Delta V_{LOS}}$ (km/s)',fontsize=labsize+2)
  axis.text(1, 1.02, labels[i], ha='right', transform=axis.transAxes, fontsize=labsize)
  axis.imshow(dat,origin='lower',cmap=mpl.cm.Greys,norm=norm,extent=ext,aspect='auto')
  axis.contour(dat,v,origin='lower',linewidths=0.7,colors='#00008B',extent=ext)
  axis.contour(dat,v_neg,origin='lower',linewidths=0.1,colors='gray',extent=ext)
  if np.any(mod >= v[0]): axis.contour(mod,v,origin='lower',linewidths=1,colors='#B22222',extent=ext)
  if plotmask and np.any(mas > 0): axis.contour(mas,levels=[0.5],origin='lower',linewidths=2,colors='k',extent=ext)
  axis.axhline(y=0,color='black')
  axis.axvline(x=0,color='black')
  axis.grid(color='gray', linestyle='--', linewidth=0.3)
  axis2 = axis.twinx()
  axis2.set_ylim([ext[2]+vsys_m,ext[3]+vsys_m])
  axis2.tick_params(which='major',length=8, labelsize=labsize, direction='in')
  axis2.set_ylabel(r'$\\mathrm{V_{LOS}}$ (km/s)',fontsize=labsize+2)
  if i==0: axis.text(0, 1.1, gname, transform=axis.transAxes,fontsize=22)
fig.savefig(outfolder+outprefix+'_pv_wind.pdf', bbox_inches='tight')
image.close()
"""


def write_scripts(f, outfolder, prefix, cubefile, cubehdu=0):
    """Write the plotting scripts of the BayesianWind f (after its outputs, in outfolder with prefix; cubefile: the
    data cube that the scripts read). Returns their paths."""
    import os
    c = f._cube_obj
    h = c.Head()
    p = c.pars()
    folder = outfolder + "plotscripts/"
    os.makedirs(folder, exist_ok=True)
    with open(folder + "plot_utils.py", "w") as fh:
        fh.write(_plot_utils_text())
    axis, rims, cuts, cut_labels = _geometry(f)
    xmin, xmax, ymin, ymax, zmin, zmax = _box(f, f.model)
    pixscale = abs(h.Cdelt(0)) * 3600.
    b = f.best
    halfsize = 1.15 * max(math.hypot((x - b.xpos) * pixscale, (y - b.ypos) * pixscale)
                          for x in (xmin, xmax) for y in (ymin, ymax))
    bunit = h.Bunit() + " * KM/S" if FluxtoJy(1, h) == 1 else "JY/BEAM * KM/S"
    vals = dict(gname=h.Name(), outfolder=outfolder, outprefix=prefix, cubefile=cubefile, cubehdu=int(cubehdu),
                plotmask=bool(p.gf.PLOTMASK), xmin=int(xmin), xmax=int(xmax), ymin=int(ymin), ymax=int(ymax),
                zmin=int(zmin), zmax=int(zmax), xcen_m=float(b.xpos), ycen_m=float(b.ypos), vsys_m=float(b.vsys),
                cont=float(np.float32(_lowest_contour(f))), cdelt3_kms=float(DeltaVel(h)),
                crval3_kms=float(AlltoVel(c.getZphys(0), h)), zmin_wcs=float(AlltoVel(c.getZphys(zmin - 0.5), h)),
                zmax_wcs=float(AlltoVel(c.getZphys(zmax + 0.5), h)), bmaj=float(h.Bmaj() / h.PixScale()),
                bmin=float(h.Bmin() / h.PixScale()), bpa=float(h.Bpa()), bunit=h.Bunit(), arcconv=3600.,
                halfsize=float(halfsize), axis_x=[float(a[0]) for a in axis], axis_y=[float(a[1]) for a in axis],
                rims=rims, cuts=cuts, cut_labels=cut_labels, cdeltsp=float(pixscale))
    if vals["zmax_wcs"] < vals["zmin_wcs"]:
        vals["zmin_wcs"], vals["zmax_wcs"] = vals["zmax_wcs"], vals["zmin_wcs"]
    header = "".join(f"{k} = {v!r}\n" for k, v in vals.items())
    paths = []
    for name, body, what in (("plot_chanmaps_wind.py", _CHANMAPS, "channel maps of data and model"),
                             ("plot_kinmaps_wind.py", _KINMAPS, "moment maps of data, model and residuals"),
                             ("plot_pvs_wind.py", _PVS, "position-velocity slices of data and model")):
        path = folder + name
        with open(path, "w") as fh:
            fh.write(f"#### WINDFIT: {what} (BBarolo; edit and run from this folder) ####\n")
            fh.write(_IMPORTS + "\n" + header + body)
        paths.append(path)
    return paths


def run(paths):
    """Run the scripts (as the plots of 3DFIT). Returns 0 if they all ran."""
    return run_scripts(paths)
