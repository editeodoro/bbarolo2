"""
GALWIND task: 3D models of bi-conical outflows (port of Tasks/galwind.cpp).

The cone is built either as a stack of ``NTOT`` cylinders (``WTYPE=0``, Lelli's
approach, each cylinder being a GALMOD disk with vertical/radial velocities) or as
a set of spherical shells (``WTYPE=1``, Galmod_wind).

Differences with the C++ code (all bugs in the original):
  * cylindrical model: VROT is read from VROT (C++ read the DENS column instead);
  * spherical model: shell densities are taken from DENS (C++ used VWIND) and the
    rotation velocities from VROT (C++ used OPENANG);
  * each cylinder/cone gets its own random seed (C++ set ISEED on the 3DFIT
    parameters, which were not used, so all cylinders shared the same random numbers);
  * CDENS and NV given to the GalWind constructor are honoured (C++ ignored them
    for cylindrical models and used the GALMOD values);
  * cylindrical model: VWIND > 0 is an outflow, as in the spherical model (C++ gave the
    cylinders the opposite velocities, an inflow), and the cylinders are centred on the
    projected axis of the disk also for rotated or mirrored axes;
  * cylindrical model: the layers are boxes (LTYPE 5 of the wind; C++ used the Gaussian layer of
    GALMOD), and the smoothed model is not multiplied by the beam area (its DENS is a flux per
    unit area, as in GALMOD; C++ made it about 10 times too bright for a 3-pixel beam);
  * spherical model: NV 1 is one sub-cloud (C++ took any NV up to 1 as 20);
  * VWIND, VDISP, DENS, VROT, OPENANG and RADII also as lists, one value per cylinder or
    shell (C++ used the first value).
"""

import math
import copy
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from .galmod import GalmodSetup, _galmod_wind, _seed, _sum_chunks
from .rings import Rings, ringList
from .smooth import Smooth3D
from .moments import MomentMap
from .pvslice import PositionVelocity
from .bbutils import BBaroloError, arcsconv, getCenterCoordinates, getDataColumn, AlltoVel, DeltaVel, \
    ProgressBar, randomAdjective
from .plotscripts import run_scripts, F


def _column(s):
    """Return the list of values of a parameter given as a file column or as a list of numbers (one value
    per cylinder or shell, separated by spaces or commas), else None."""
    s = str(s)
    return getDataColumn(s) if "file(" in s.lower() else ringList(s)


def _values(name, s, n):
    """The n values of the GALWIND parameter name (value s): a file column or a list with at least n values,
    or a single value for all."""
    col = _column(s)
    if col is None:
        try:
            return np.full(n, float(str(s).strip()))
        except ValueError:
            raise BBaroloError(f"GALWIND: {name} {s} is not a number, a list of numbers or file(name,column)") from None
    if len(col) < n:
        raise BBaroloError(f"GALWIND: {name} has {len(col)} values, fewer than the {n} cylinders or shells.")
    return np.asarray(col[:n], float)


class GalWind:
    """Bi-conical outflow model.

    Three ways to build it, as in C++::

        GalWind(cube)                     # parameters from cube.pars().gw
        GalWind(cube, gwpar)              # a GalwindPar structure
        GalWind(cube, x0, y0, pa, inc, disp, dens, vsys, vw, openang, htot, denstype,
                ntot=25, cdens=10, nv=10, nthreads=1)
    """

    def __init__(self, cube, *args, **kwargs):
        self.cin = cube
        self.out = None
        if len(args) <= 1 and not kwargs:
            self.par = copy.deepcopy(args[0] if args else cube.pars().gw)
        else:
            names = ("x0", "y0", "pa", "inc", "disp", "dens", "vsys", "vw", "openang", "htot",
                     "denstype", "ntot", "cdens", "nv", "nthreads")
            vals = dict(ntot=25, cdens=10, nv=10, nthreads=1)
            vals.update(dict(zip(names, args)))
            vals.update(kwargs)
            from .params import GalwindPar
            p = GalwindPar()
            p.flagGALWIND = True
            def text(v):                # (a list: one value per cylinder or shell)
                return " ".join(str(x) for x in np.ravel(v)) if np.ndim(v) else str(v)
            p.XPOS, p.YPOS = str(vals["x0"]), str(vals["y0"])
            p.PHI, p.INC = str(vals["pa"]), str(vals["inc"])
            p.VDISP, p.DENS = text(vals["disp"]), text(vals["dens"])
            p.VSYS, p.VWIND = str(vals["vsys"]), text(vals["vw"])
            p.OPENANG, p.HTOT = text(vals["openang"]), float(vals["htot"])
            p.DENSTYPE, p.NTOT = int(vals["denstype"]), int(vals["ntot"])
            p.CDENS, p.NV = int(vals["cdens"]), int(vals["nv"])
            self.par = p
            cube.pars().threads = int(vals["nthreads"])
        cube.checkBeam()

    # ------------------------------------------------------------------
    def getOut(self): return self.out
    def getArray(self): return self.out.array
    def getPar(self): return self.par

    def _center(self):
        return getCenterCoordinates([self.par.XPOS, self.par.YPOS], self.cin.Head())

    def compute(self):
        from .cube import Cube
        out = np.zeros((self.cin.DimY(), self.cin.DimX(), self.cin.DimZ()), np.float64)
        ok = self._cylindrical(out) if self.par.WTYPE == 0 else self._spherical(out)
        self.out = Cube.fromArray(np.ascontiguousarray(out.transpose(2, 0, 1)).astype(np.float32),
                                  self.cin.Head().copy(), self.cin.pars())
        return ok

    # ------------------------------------------------------------------
    def _gmpar(self):
        """GALMOD parameters for the cylinders: those of the cube with CDENS/NV of the wind, and the vertical
        layer of the wind, LTYPE 5 (a box: the cylinders of height 2 z0 stack exactly; C++ used the layer of
        GALMOD, Gaussian by default, so neighbouring cylinders overlapped)."""
        gm = copy.deepcopy(self.cin.pars().gm)
        gm.CDENS, gm.NV = self.par.CDENS, self.par.NV
        gm.ISEED = self.par.ISEED
        gm.LTYPE = self.par.LTYPE
        return gm

    def _cylindrical(self, out):
        c, par = self.cin, self.par
        try:
            OpA = float(str(par.OPENANG).strip())
        except ValueError:
            raise BBaroloError(f"GALWIND: OPENANG {par.OPENANG}: with cylinders (WTYPE 0) the opening angle is "
                               "one value.") from None
        ntot = int(par.NTOT)
        z0 = par.HTOT / (2 * ntot)                          # half-height of cylinders (arcsec)
        width = (2 * z0) * math.tan(OpA / 2. * math.pi / 180.)
        PA, inc, Vsys = float(par.PHI), float(par.INC), float(par.VSYS)
        x0, y0 = self._center()

        def column(name, s):
            col = _column(s)
            if col is not None and len(col) != ntot:
                raise BBaroloError(f"GALWIND: VWIND, VDISP, DENS and VROT must have NTOT={ntot} values (one per "
                                   f"cylinder): {name} has {len(col)}.")
            return list(_values(name, s, ntot))
        Vwind, Vdisp = column("VWIND", par.VWIND), column("VDISP", par.VDISP)
        Dens, Vrot = column("DENS", par.DENS), column("VROT", par.VROT)

        setup = GalmodSetup(c, gmpar=self._gmpar())
        # the cylinders are a flux per unit area (DENSFLUX true, as GALMOD): no beam-area factor when smoothed
        self._usescale = not setup.densIsFlux
        gtype = getattr(c.pars(), "galmodtype", "MC")
        # pixel offset of the point at height z = 1 arcsec on the axis of the disk, projected as the clouds
        # of GALMOD are (sky y = -z sin(inc)): the cylinder k of each cone is centred at z = ±(2k-1)·z0
        sinc = math.sin(math.radians(inc))
        phi = math.radians(PA) + setup.crota2
        dx = sinc * math.cos(phi) / (60. * setup.cdelt[0])
        dy = -sinc * math.sin(phi) / (60. * setup.cdelt[1])
        base = _seed(par.ISEED)

        def cylinder(k):
            Rin = (k - 1) * width if par.DENSTYPE == 2 else 0.
            Rout = k * width
            dens = Dens[k - 1] / (k * k) if par.DENSTYPE == 0 else Dens[k - 1]
            nr = 2 if par.DENSTYPE == 2 else k + 1
            radsep = (Rout - Rin) / (nr - 1)
            F_ = np.ones(nr) if par.DENSTYPE == 2 else np.arange(nr) / (nr - 1)
            ang = F_ * OpA / 2. * math.pi / 180.
            r = Rings().setRings(radii=Rin + np.arange(nr) * radsep, vsys=Vsys, vdisp=Vdisp[k - 1],
                                 inc=inc, phi=PA, z0=z0, dens=dens, vrot=Vrot[k - 1], radsep=radsep)
            # an outflow for VWIND > 0 (as the shells of WTYPE 1): along the lines through the centre, away
            # from it (C++ had the opposite signs, an inflow)
            r.vrad = Vwind[k - 1] * np.sin(ang)
            cones = []
            for sgn in (1, -1):                 # the cone above the plane, then the one below
                z = sgn * (2. * k - 1) * z0
                r.xpos[:] = x0 + z * dx
                r.ypos[:] = y0 + z * dy
                r.vvert = sgn * Vwind[k - 1] * np.cos(ang)
                cones.append(r.copy())
            return cones

        nthreads = max(1, int(c.pars().threads))
        verb = c.pars().verbose
        bar = ProgressBar(False, verb, c.pars().showbar)
        bar.init(" Generating outflow model... ", ntot)
        nch = max(1, min(nthreads, ntot))          # chunk t: the cylinders t+1, t+1+nch, ...
        done = [0]

        def work(t, buf):
            for k in range(t + 1, ntot + 1, nch):
                for ic, r in enumerate(cylinder(k)):
                    setup._build(r, (0, 0), (c.DimX(), c.DimY()), buf, (base + 2 * k + ic) & 0xFFFFFFFF, 1, gtype)
                done[0] += 1
                bar.update(done[0])
        _sum_chunks(out, nch, work)
        bar.fillSpace(" Done.\n")
        return True

    def _shells(self, setup):
        """Port of compute_spherical + Galmod_wind::shellIO. Returns (sr (11,ns), nv, sep)."""
        par = self.par
        PA, inc, Vsys = float(par.PHI), float(par.INC), float(par.VSYS)
        x0, y0 = self._center()
        Radii = _column(par.RADII)
        nr = int(par.NTOT)
        radsep = par.HTOT / nr
        if Radii is not None:
            radsep = np.mean(np.diff(Radii[1:])) if len(Radii) > 2 else radsep
            nr = len(Radii)
        radii = np.asarray(Radii, float) if Radii is not None else np.arange(nr) * radsep + radsep / 2.

        def column(name):
            return _values(name, getattr(par, name), nr)
        U = np.vstack([column("VROT") * 1000., column("VWIND") * 1000., column("VDISP") * 1000.,
                       column("DENS"), np.full(nr, math.radians(inc)), np.full(nr, math.radians(PA)),
                       np.radians(column("OPENANG")), np.full(nr, x0), np.full(nr, y0),
                       np.full(nr, Vsys * 1000.)])
        if np.any(U[2] < 0):
            raise BBaroloError("GALMOD error: Negative velocity dispersion not allowed.")
        if np.any(U[3] < 0):
            raise BBaroloError("GALMOD error: Negative column-density not allowed.")
        urad = radii / 60.
        sep = 0.75 * min(setup.pixsize)
        if np.any(urad[:-1] + sep >= urad[1:]):
            raise BBaroloError("GALMOD error: Radii not in increasing order or separation too small.")
        f32 = np.float32
        rads, vals = [], []
        r = f32(urad[0] + sep / 2.) if urad[0] != 0 else f32(sep / 2.)
        for i in range(1, nr):
            slope = (U[:, i] - U[:, i - 1]) / (urad[i] - urad[i - 1])
            while r < f32(urad[i]):
                rads.append(float(r))
                vals.append(U[:, i - 1] + slope * (float(r) - urad[i - 1]))
                r = f32(float(r) + sep)
        sr = np.empty((11, len(rads)))
        if rads:
            sr[0] = rads
            sr[1:] = np.array(vals).T
        nvpar = par.NV if par.NV > 0 else 20                # (NV not given: 20; C++ also took NV 1 as 20)
        nv = np.where(sr[3] == 0, 1, nvpar if nvpar >= 1 else setup.nsubs).astype(np.int64)
        return np.ascontiguousarray(sr), nv, sep

    def _spherical(self, out):
        c, par = self.cin, self.par
        gm = self._gmpar()
        gm.CMODE, gm.DENSFLUX = 1, False
        setup = GalmodSetup(c, gmpar=gm, densflux=False)
        self._usescale = True          # (a column density: a flux per pixel, scaled to a flux per beam)
        sr, nv, sep = self._shells(setup)
        # rows of the kernel: radius, vrot, vsph, vdisp, dens, inc, pa, openang, xpos, ypos, vsys
        sr = np.ascontiguousarray(sr[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
        ns = sr.shape[1]
        nc = np.rint(setup.cdens * sr[4] * 2 * math.pi * sr[0] * sep / setup.pixarea)
        if ns and np.any(nc == 0):
            print(" GALWIND WARNING: no clouds in some shells. Choose higher CDENS.")
        nthreads = max(1, min(int(c.pars().threads), ns))
        chunks = np.array_split(np.arange(ns), nthreads)
        args = (sep, nv, setup.cdens, 1, setup.pixarea, setup.cdelt[0], setup.cdelt[1], setup.crota2,
                0, 0, c.DimX(), c.DimY(), setup.sig_instr, setup.axtyp, setup.velsys, setup.lfreq0,
                setup.crval3, setup.cdelt3, setup.crpix3, setup.relint, setup.cd2i)
        seed = _seed(par.ISEED)

        def work(t, buf):
            ch = chunks[t]
            if len(ch):
                _galmod_wind(buf, sr, int(ch[0]), int(ch[-1]) + 1, (seed + 7919 * t) & 0xFFFFFFFF, *args)
        if c.pars().verbose:
            print(" Modeling outflow... ", end="", flush=True)
        _sum_chunks(out, nthreads, work)
        if c.pars().verbose:
            print("OK.")
        return True

    # ------------------------------------------------------------------
    def smooth(self, scalefac=True):
        if self.out is None or not self.par.SM:
            return False
        h = self.cin.Head()
        arcconv = arcsconv(h.Cunit(0))
        sm = Smooth3D()
        # (C++ scaled the cylinders too, about 10 times too bright with a beam of 3 pixels)
        sm.setUseScalefac(scalefac and getattr(self, "_usescale", True))
        sm.setUseBlanks(False)
        res = sm.smooth(self.out, (0., 0., 0.), (h.Bmaj() * arcconv, h.Bmin() * arcconv, h.Bpa()),
                        oldarray=self.out.array, threads=self.cin.pars().threads)
        res = np.asarray(res, np.float32)
        res[res < 1.E-12] = 0.
        self.out.array = res
        return True

    def writeFITS(self, fname="", fullHead=False):
        if self.out is None:
            return False
        p = self.cin.pars()
        fn = fname or p.outFolder + p.outPrefix + "_wind.fits"
        return self.out.fitswrite_3d(fn, fullHead)

    def momentMaps(self):
        """The total, velocity and dispersion maps of the model (three astropy HDUs, no mask)."""
        m = MomentMap()
        m.input(self.out)
        out = []
        for f in (m.ZeroMoment, m.FirstMoment, m.SecondMoment):
            f(False)
            out.append(m.to_hdu())
        return out

    def pv(self):
        """The position-velocity slice of the model through the centre, along PHI - 90 (an Image2D)."""
        x0, y0 = self._center()
        return PositionVelocity(self.out, x0, y0, float(self.par.PHI) - 90)

    def writeMomentMaps(self, maps=None):
        """Write <prefix>_wind_<i>mom.fits (maps: the HDUs of momentMaps(), if already computed)."""
        if self.out is None:
            return False
        from .cube import write_hdu
        p = self.cin.pars()
        fn = p.outFolder + p.outPrefix + "_wind"
        for i, hdu in enumerate(maps if maps is not None else self.momentMaps()):
            write_hdu(hdu, fn + f"_{i}mom.fits")
        return True

    def writePV(self, fname="", pv=None):
        """Write <prefix>_wind_pv.fits (pv: the HDU of the slice, if already computed)."""
        if self.out is None:
            return False
        from .cube import write_hdu
        p = self.cin.pars()
        fn = fname or p.outFolder + p.outPrefix + "_wind_pv.fits"
        write_hdu(pv if pv is not None else self.pv().to_hdu(), fn)
        return True

    def makePlots(self):
        if self.out is None:
            return False
        p = self.cin.pars()
        h = self.cin.Head()
        PA, Vsys = float(self.par.PHI), float(self.par.VSYS)
        x0, y0 = self._center()
        s = f"""import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.visualization import PowerStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from astropy.visualization import PercentileInterval
mpl.style.use('classic')
mpl.rc('font',family='serif')
params = {{'text.usetex': False, 'mathtext.fontset': 'stix'}}
plt.rcParams.update(params)

# PARAMETERS: plotting the fit parameters
gname = '{p.outPrefix}'
outfolder = '{p.outFolder}'

nc = 7
xlen, ylen = 0.15, 0.15
xsep, ysep = 0.01, 0.01
cmap = plt.get_cmap('Greys')
interval = PercentileInterval(99)
# Plotting channel maps
fig = plt.figure(figsize=(10, 10))
ax = []
for jj in range (nc):
\tax.append(fig.add_axes([jj*(xlen+xsep),1,xlen,ylen]))

fitc = fits.open(outfolder+gname+'_wind.fits')[0]
d, h = fitc.data, fitc.header
lims = interval.get_limits(d)
norm = ImageNormalize(vmin=lims[0], vmax=lims[1], stretch=PowerStretch(1))
cont = np.array([1,2,4,16,64,256,1024])*0.04
crpix3_kms = {F(h.Crpix(2))}
cdelt3_kms = {F(DeltaVel(h))}
crval3_kms = {F(AlltoVel(h.Crval(2), h))}
velos = (np.arange(0,d.shape[0])+1-crpix3_kms)*cdelt3_kms+crval3_kms

for k in range(nc):
\ta = ax[k]
\tchan = int(k*(d.shape[0])/nc)
\ttopl = d[chan]
\ta.tick_params(top=False,right=False,bottom=False,left=False,labelbottom=False,labelleft=False)
\tif np.all(topl<cont[0]): topl[:,:] = 0
\ta.imshow(topl,origin='lower',cmap=cmap,norm=norm)
\tif np.any(topl>=cont[0]): a.contour(topl,levels=cont,colors='#009999')
\ttext = r'$V_\\mathrm{{LOS}}$ = %.1f km/s'%(velos[chan])
\ta.text(0.5,1.05,text,transform=a.transAxes,ha='center')
fig.savefig(outfolder+gname+'_wind.pdf',bbox_inches='tight')

# Plotting kinematic maps
fig = plt.figure(figsize=(10, 10))
ax, axcb = [], []
for jj in range (4):
\tif jj==3: ax.append(fig.add_axes([jj*(xlen+xsep),1,1.5*xlen,ylen]))
\telse: ax.append(fig.add_axes([jj*(xlen+xsep),1,xlen,ylen]))
\tif jj<3:
\t\taxcb.append(fig.add_axes([jj*(xlen+xsep),1-0.019,xlen,0.015]))

pv   = fits.open(outfolder+gname+'_wind_pv.fits')[0].data
mom0 = fits.open(outfolder+gname+'_wind_0mom.fits')[0].data
mom1 = fits.open(outfolder+gname+'_wind_1mom.fits')[0].data
mom2 = fits.open(outfolder+gname+'_wind_2mom.fits')[0].data
mom0 /= np.nanmax(mom0)
topl = [mom0,mom1,mom2,pv]
cmaps = [plt.get_cmap('afmhot'),plt.get_cmap('Spectral_r'),plt.get_cmap('viridis'),plt.get_cmap('Greys')]
norms = [ImageNormalize(vmin=0,vmax=1),ImageNormalize(vmin=np.nanmin(mom1),vmax=np.nanmax(mom1)),ImageNormalize(vmin=np.nanmin(mom2),vmax=np.nanmax(mom2))]
titls = ['Intensity','Velocity','Dispersion','Position-Velocity']
labs  = ['Normalized $I$', r'$V_\\mathrm{{LOS}}$ (km/s)', r'$\\sigma$ (km/s)', 'Offset (arbitrary units)']
for k in range(len(topl)):
\ta = ax[k]
\ta.tick_params(top=False,right=False,bottom=False,left=False,labelbottom=False,labelleft=False)
\ta.text(0.5,1.1,titls[k],transform=a.transAxes,ha='center',va='center')
\tif k==1:
\t\txcen, ycen, phi = {F(x0)},{F(y0)},{F(PA)}
\t\tif phi==90: a.axvline(xcen,c='k',ls='--',alpha=0.7)
\t\telse:
\t\t\txx = np.arange(d.shape[-1])
\t\t\tyy = np.tan(np.radians(phi))*(xx-xcen)+ycen
\t\t\ta.plot(xx,yy,c='k',ls='--',alpha=0.7)

\tif k==3:
\t\text = [-0.99,0.99 ,velos[0],velos[-1]]
\t\ta.set_ylim(ext[2]-2*abs(cdelt3_kms),ext[3]+2*abs(cdelt3_kms))
\t\ta.tick_params(top=True,left=True,bottom=True,right=True,labelright=True)
\t\ta.tick_params(axis='x', pad=7)
\t\ta.imshow(topl[k],origin='lower',cmap=cmaps[k],aspect='auto',extent=ext)
\t\tif np.nanmax(topl[k])>=cont[0]: a.contour(topl[k],levels=cont,colors='#769BCB',extent=ext)
\t\ta.axhline({F(Vsys)},c='k',ls='-',alpha=0.5)
\t\ta.axvline(0,c='k',ls='-',alpha=0.5)
\t\ta.tick_params(labelbottom=True)
\t\ta.text(1.25, 0.5,r'$V_\\mathrm{{LOS}}$ (km/s)',transform=a.transAxes,ha='center',va='center',rotation=90)
\telse:
\t\tm = a.imshow(topl[k],origin='lower',cmap=cmaps[k],aspect='auto',norm=norms[k])
\t\tfor axis in ['top','bottom','left','right']: axcb[k].spines[axis].set_linewidth(0)
\t\tcbar = plt.colorbar(mappable=m,cax=axcb[k],orientation='horizontal')
\t\tcbar.solids.set_edgecolor('face')
\t\tcbar.outline.set_linewidth(0)
\t\tcbar.locator = mpl.ticker.MaxNLocator(nbins=5)
\t\tcbar.update_ticks()
\ta.text(0.5,-0.35,labs[k],transform=a.transAxes,ha='center')
fig.savefig(outfolder+gname+'_wind_maps.pdf',bbox_inches='tight')
"""
        fn = p.outFolder + "pyscript_w.py"
        with open(fn, "w") as f:
            f.write(s)
        if p.verbose:
            print(f" Producing {randomAdjective(1)} plots...", end="", flush=True)
        ret = run_scripts([fn])
        if p.verbose:
            print(" Done." if ret == 0 else " Something went wrong! Check pyscript_w.py in the output folder.")
        return True
