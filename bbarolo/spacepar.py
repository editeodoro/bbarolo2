"""
SPACEPAR task: residuals over the full 2D space of any pair of 3DFIT parameters,
ring by ring (port of Tasks/spacepar.cpp).

The grid points of each ring are evaluated in parallel (THREADS). Each model uses
the same random seed, so the parameter space is free of Monte-Carlo jitter between
neighbouring grid points (and with GALMODTYPE=ANALYTIC it is noise-free).
"""

import numpy as np
from concurrent.futures import ThreadPoolExecutor

from .galfit import Galfit, VROT, VDISP, Z0, INC, PA, XPOS, YPOS, VSYS, VRAD
from .header import Header
from .bbutils import AlltoVel, ProgressBar, randomAdjective, mkdirp, BBaroloError
from .plotscripts import run_scripts

ALLOWED = ("VROT", "VDISP", "Z0", "INC", "PA", "XPOS", "YPOS", "VSYS", "VRAD")
_ATTR = dict(VROT="vrot", VDISP="vdisp", Z0="z0", INC="inc", PA="phi", XPOS="xpos",
             YPOS="ypos", VSYS="vsys", VRAD="vrad")


class Spacepar(Galfit):

    refine2D = False        # (the rings as given, or as estimated)

    def __init__(self, c):
        super().__init__(c)
        self.n_galfit = 1   # (the normalisation of the first stage of 3DFIT, NORM)
        p = c.pars()
        self.p1, self.p2 = p.P1.upper(), p.P2.upper()
        for i, q in enumerate((self.p1, self.p2)):
            if q not in ALLOWED:
                raise BBaroloError(f"SPACEPAR ERROR: Unknown parameter {i + 1} {q}")
        if self.p1 == self.p2:
            raise BBaroloError("SPACEPAR ERROR: p1=p2!!")
        self.mpar = [False] * len(self.mpar)
        self.minp1, self.maxp1, self.delp1 = (float(v) for v in p.P1p[:3])
        self.minp2, self.maxp2, self.delp2 = (float(v) for v in p.P2p[:3])
        mx, mn = self.maxs, self.mins
        mx[VROT], mn[VROT] = 10000, 0
        mx[VDISP], mn[VDISP] = 10000, 0
        mx[Z0], mn[Z0] = 10000, 0
        mx[INC], mn[INC] = 90, 0
        mx[PA], mn[PA] = 360, 0
        mx[XPOS], mn[XPOS] = c.DimX() - 1, 0
        mx[YPOS], mn[YPOS] = c.DimY() - 1, 0
        mx[VRAD], mn[VRAD] = 10000, -10000
        mx[VSYS] = AlltoVel(c.getZphys(c.DimZ() - 1), c.Head())
        mn[VSYS] = AlltoVel(c.getZphys(0), c.Head())
        self.parspace = None

    # ------------------------------------------------------------------
    def _dring(self, ir, v1, v2):
        inr = self.inr
        if inr.nr == 1:     # (as 3DFIT: half RADSEP, or half the radius; C++ read beyond the rings)
            w1 = w2 = inr.radsep / 2. if inr.radsep > 0 else inr.radii[0] / 2.
        elif ir == 0:
            w1 = w2 = (inr.radii[1] - inr.radii[0]) / 2.
        elif ir == inr.nr - 1:
            w1 = w2 = (inr.radii[ir] - inr.radii[ir - 1]) / 2.
        else:
            w1 = (inr.radii[ir] - inr.radii[ir - 1]) / 2.
            w2 = (inr.radii[ir + 1] - inr.radii[ir]) / 2.
        d = inr.subset([ir, ir])
        d.radii = np.array([max(inr.radii[ir] - w1, 0.), max(inr.radii[ir] + w2, 0.)])
        d.radsep = d.radii[1] - d.radii[0]
        getattr(d, _ATTR[self.p1])[:] = v1
        getattr(d, _ATTR[self.p2])[:] = v2
        d.id = ir
        return d

    def calculate(self):
        c = self.cin
        p = c.pars()
        verb = p.verbose
        if verb:
            p.verbose = False
            print("\n" + " SPACEPAR ".rjust(40, "=") + " " * 34 + "\n")
            print(f" Exploring parameter space: \n  {self.p1:<5s} in [{self.minp1:.2f}, {self.maxp1:.2f}] "
                  f"with delta = {self.delp1:.2f}\n  {self.p2:<5s} in [{self.minp2:.2f}, {self.maxp2:.2f}] "
                  f"with delta = {self.delp2:.2f}\n")
        n1 = int(np.float32(abs(self.maxp1 - self.minp1) / self.delp1 + 1))
        n2 = int(np.float32(abs(self.maxp2 - self.minp2) / self.delp2 + 1))
        p1steps = self.minp1 + np.arange(n1) * self.delp1
        p2steps = self.minp2 + np.arange(n2) * self.delp2
        self.setOutSpacepar(n1, n2)
        start_rad = self.par.STARTRAD if self.par.STARTRAD < self.inr.nr else 0
        scaling = 1.
        if self.par.NORMALCUBE:
            scaling = 10. / c.stat().getMax()
            self.data_T = self.data_T * scaling
            self.data_noise *= scaling
        nthreads = max(1, int(p.threads))
        ex = ThreadPoolExecutor(nthreads)
        for ir in range(start_rad, self.inr.nr):
            if verb:
                print(f"\n Working on ring #{ir + 1} at radius {self.inr.radii[ir]:.2f} arcsec: ")
            bar = ProgressBar(True, verb, p.showbar)
            bar.init("   Calculating models...", n1 * n2)
            done = [0]

            def work(k, ir=ir):
                i, j = divmod(k, n2)
                v = self.getFuncValue(self._dring(ir, p1steps[i], p2steps[j]))
                done[0] += 1
                bar.update(done[0])
                return v
            vals = np.array(list(ex.map(work, range(n1 * n2)))).reshape(n1, n2)
            self.parspace[ir] = vals.T.astype(np.float32)
            imin = np.unravel_index(np.argmin(vals), vals.shape)
            if verb:
                bar.fillSpace("OK.\n")
                print(f"   Minimum (fmin={vals[imin]:.6f}) at: {self.p1} = {p1steps[imin[0]]:.1f}, "
                      f"{self.p2} = {p2steps[imin[1]]:.1f}")
        ex.shutdown()
        p.verbose = verb
        if self.par.NORMALCUBE:
            self.data_T = np.ascontiguousarray(c.array.transpose(1, 2, 0))
            self.data_noise /= scaling

    def setOutSpacepar(self, n1, n2):
        from .cube import Cube
        inr = self.inr
        start_rad = self.par.STARTRAD if self.par.STARTRAD < inr.nr else 0
        h = Header()
        h.setNumAx(3)
        h.dimAxes = [n1, n2, inr.nr]
        for i in range(3):
            h.setCrpix(i, 1)
        h.setCrval(0, self.minp1); h.setCrval(1, self.minp2); h.setCrval(2, inr.radii[start_rad])
        h.setCdelt(0, self.delp1); h.setCdelt(1, self.delp2); h.setCdelt(2, inr.radsep)
        h.setCtype(0, self.p1); h.setCtype(1, self.p2); h.setCtype(2, "RING_RAD")

        def unit(q):
            if q in ("INC", "PA"): return "deg"
            if q in ("XPOS", "YPOS"): return "pix"
            if q == "Z0": return "arcsec"
            return "km/s"
        h.setCunit(0, unit(self.p1)); h.setCunit(1, unit(self.p2)); h.setCunit(2, "arcsec")
        h.setBtype(f"FMIN_{self.par.FTYPE}")
        h.setBunit(self.cin.Head().Bunit())
        h.setName(self.cin.Head().Name())
        self.parspace = np.zeros((inr.nr, n2, n1), np.float32)
        self._pshead = h
        self._Cube = Cube

    def space_hdu(self):
        """The residuals over the grid of the two parameters, relative to their minimum in each ring
        (|f - fmin| / fmin; axes P1, P2 and the ring radius), as an astropy HDU (<prefix>_spacepar.fits)."""
        ps = self.parspace.astype(np.float64)
        mins = ps.reshape(ps.shape[0], -1).min(axis=1)[:, None, None]
        with np.errstate(invalid="ignore", divide="ignore"):
            rel = np.abs(ps - mins) / mins
        return self._Cube.fromArray(rel.astype(np.float32), self._pshead, self.cin.pars()).to_hdu()

    def plotAll_Python(self, hdu=None):
        """Write <prefix>_spacepar.fits (hdu: that of space_hdu(), if already computed) and the plotting script,
        and make the plots (PLOTS)."""
        if self.parspace is None:
            return
        from .cube import write_hdu
        p = self.cin.pars()
        verb = p.verbose
        outfold, obj = p.outFolder, p.outPrefix
        fname = outfold + obj + "_spacepar.fits"
        write_hdu(hdu if hdu is not None else self.space_hdu(), fname)
        mkdirp(outfold + "plotscripts/")
        fout = outfold + obj + "_spacepar.pdf"
        script = outfold + "plotscripts/spacepar.py"
        with open(script, "w") as f:
            f.write(f"""# A script to produce output plots for SPACEPAR task.
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.convolution import convolve,Gaussian2DKernel
matplotlib.rc('xtick',direction='in')
matplotlib.rc('ytick',direction='in')
matplotlib.rc('font',family='sans-serif',size=10)

f = fits.open('{fname}')[0]
d, h = f.data, f.header
p1, p2 = h['CTYPE1'], h['CTYPE2']
p1u, p2u, ru = h['CUNIT1'], h['CUNIT2'], h['CUNIT3']
p1ran = (np.arange(0,d.shape[2])+1-h['CRPIX1'])*h['CDELT1']+h['CRVAL1']
p2ran = (np.arange(0,d.shape[1])+1-h['CRPIX2'])*h['CDELT2']+h['CRVAL2']
rings = (np.arange(0,d.shape[0])+1-h['CRPIX3'])*h['CDELT3']+h['CRVAL3']

nrad = d.shape[0]
ncols = 4
nrows = int(np.ceil(nrad/float(ncols)))
ext = [p1ran[0],p1ran[-1],p2ran[0],p2ran[-1]]
cmap = plt.get_cmap('nipy_spectral')

fig = plt.figure(figsize=(8,8))
x_axis_len, y_axis_len = 0.27, 0.27
x_sep, y_sep = 0.07, 0.08
count = 0
axis, bottom_corner = [], [0.1,0.7]
for i in range (nrows):
\tbottom_corner[0] = 0.1
\tfor j in range (ncols):
\t\tif (count>=nrad): break
\t\taxis.append(fig.add_axes([bottom_corner[0],bottom_corner[1],x_axis_len,y_axis_len]))
\t\tbottom_corner[0]+=x_axis_len+x_sep
\t\tcount += 1
\tbottom_corner[1]-=(y_axis_len+y_sep)

for i in range (nrad):
\tnr = int(i/ncols) + 1
\tnc = i - (nr-1)*ncols + 1
\ttoplot = 100*d[i]
\ttoplot_sm = convolve(toplot,Gaussian2DKernel(x_stddev=1),boundary='extend')
\ta = np.unravel_index(np.argmin(toplot),toplot.shape)
\tp1min, p2min = p1ran[a[1]], p2ran[a[0]]
\tax = axis[i]
\tax.set_xlim(ext[0],ext[1])
\tax.set_ylim(ext[2],ext[3])
\tax.imshow(toplot,origin='lower',extent=ext,aspect='auto',cmap=cmap)
\tcs = ax.contour(toplot_sm,levels=[2,5,10,30],colors='w',origin='lower',extent=ext)
\tax.clabel(cs, cs.levels, inline=True, fmt='%.0f', fontsize=9)
\tax.plot(p1min,p2min,'x',mew=2,ms=8,c='w')
\tradstr = 'R = %.2f %s'%(rings[i],ru)
\tminstr = 'min = (%.1f %s, %.1f %s)'%(p1min,p1u,p2min,p2u)
\tax.text(0.01,1.1,radstr,transform=ax.transAxes)
\tax.text(0.01,1.03,minstr,transform=ax.transAxes)
\tif nc==1: ax.set_ylabel(p2+' ('+p2u+')')
\tif (nr==nrows) or (nr==nrows-1 and nrad%ncols!=0 and nc>nrad%ncols):
\t\tax.set_xlabel(p1+' ('+p1u+')')

fig.savefig('{fout}',bbox_inches='tight')
""")
        if p.plots:
            if verb:
                print(f"\n\nProducing {randomAdjective(1)} parameter space plots...", end="", flush=True)
            ret = run_scripts([script])
            if verb:
                print(" Done." if ret == 0 else " Something went wrong! Check spacepar.py in the output folder.")
