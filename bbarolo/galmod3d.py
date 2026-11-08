"""
Models of a datacube from 3D fields of the gas (GalMod3D): the density, the velocity and the
velocity dispersion as functions of (R, z), (R, phi, z) or (x, y, z) in the frame of the galaxy,
projected on the sky with the geometry of GALMOD (centre, VSYS, INC, PA).

GALMOD (bbarolo.GalMod, galmod.DiskModel) describes the gas with tilted rings: every parameter is a
function of radius and the vertical structure is a fixed layer. GalMod3D takes any 3D distribution
instead, e.g. a flaring or lagging thick disk, a bar, spiral arms, a warp written in the density, or
the fields of a simulation on a grid. Python only (it is not a task of the command line).

The model is built on a regular Cartesian grid in the frame of the galaxy (spacing ``step``): the
density at each point, times the volume of the point, is the weight of the point; the point is
projected on the sky, and deposited with its line of sight velocity and a Gaussian line, integrated
over each channel (as the ANALYTIC models of GALMOD), or as random clouds (``galmodtype="MC"``, as
the MC models). The cube is then convolved with the beam, with the convolution of GALMOD.
"""

import math
import numpy as np
import numba as nb

from .bbutils import BBaroloError, FluxtoJy, isFluxUnitKnown, getCenterCoordinates
from .galmod import _lround, _velgrid, _line_profile, _PHI_TAB, model_hdu

# The coordinate systems: the arguments of the fields and the names of the velocity components
_COORDS = {"rz": ("Rz", ("R", "z"), ("vr", "vphi", "vz")),
           "rphiz": ("Rphiz", ("R", "phi", "z"), ("vr", "vphi", "vz")),
           "xyz": ("xyz", ("x", "y", "z"), ("vx", "vy", "vz"))}
_COORDS_ALIASES = {"axisymmetric": "rz", "cylindrical": "rphiz", "cartesian": "xyz"}
_VELOCITIES = {"vr", "vphi", "vx", "vy", "vz"}
# options of the cube (BBarolo parameters) that GalMod3D passes on
_CUBE_OPTIONS = ("specres", "linear", "restwave", "restfreq", "relint", "redshift", "mask", "threads", "verbose")


# --------------------------------------------------------------------------
# Fields
# --------------------------------------------------------------------------
class _Field:
    """A field of the model: a constant, a vectorised function of the coordinates, or values on a grid
    of the coordinates, (axes, values), interpolated linearly. Outside the grid: zero for the density
    (outside=0), the value at the nearest edge for the other fields (outside=None). The axis of phi
    (if any) is periodic."""

    def __init__(self, name, value, argnames, outside=None):
        self.name = name
        self.const = None
        self.func = None
        self.interp = None
        if callable(value):
            self.func = value
        elif isinstance(value, (tuple, list)) and len(value) == 2 and not np.isscalar(value[0]):
            self._set_grid(value, argnames, outside)
        else:
            try:
                self.const = float(value)
            except (TypeError, ValueError):
                raise BBaroloError(f"GalMod3D: {name} must be a number, a function of ({', '.join(argnames)}) "
                                   f"or a tuple (axes, values) of a grid, not {type(value).__name__}")
            if not math.isfinite(self.const):
                raise BBaroloError(f"GalMod3D: {name} = {value} is not finite")

    def _set_grid(self, value, argnames, outside):
        from scipy.interpolate import RegularGridInterpolator
        axes, values = value
        if len(axes) != len(argnames):
            raise BBaroloError(f"GalMod3D: the grid of {self.name} has {len(axes)} axes; the fields are functions "
                               f"of ({', '.join(argnames)})")
        axes = [np.asarray(a, dtype=float).ravel() for a in axes]
        values = np.asarray(values, dtype=float)
        if values.shape != tuple(len(a) for a in axes):
            raise BBaroloError(f"GalMod3D: the values of {self.name} have shape {values.shape}, the axes "
                               f"{tuple(len(a) for a in axes)}")
        for a, n in zip(axes, argnames):
            if len(a) < 2 or np.any(np.diff(a) <= 0):
                raise BBaroloError(f"GalMod3D: the axis {n} of the grid of {self.name} must have at least two "
                                   "increasing values")
        self.iphi = argnames.index("phi") if "phi" in argnames else None
        if self.iphi is not None:
            # periodic in phi: the values at the first angle also at the first angle + 2 pi
            a = axes[self.iphi]
            if a[-1] - a[0] > 2 * math.pi + 1e-9:
                raise BBaroloError(f"GalMod3D: the axis phi of the grid of {self.name} spans more than 2 pi")
            if a[-1] < a[0] + 2 * math.pi - 1e-9:
                axes[self.iphi] = np.append(a, a[0] + 2 * math.pi)
                first = np.take(values, [0], axis=self.iphi)
                values = np.concatenate([values, first], axis=self.iphi)
        self.axes = axes
        self.outside = outside
        self.interp = RegularGridInterpolator(axes, values, method="linear", bounds_error=False,
                                              fill_value=outside)

    def __call__(self, args):
        """The field at the points of args (a tuple of arrays of the coordinates, of the same shape)."""
        shape = np.shape(args[0])
        if self.const is not None:
            return np.full(shape, self.const)
        if self.func is not None:
            out = np.asarray(self.func(*args), dtype=float)
            if out.shape != shape:
                try:
                    out = np.broadcast_to(out, shape)
                except ValueError:
                    raise BBaroloError(f"GalMod3D: the function of {self.name} returned an array of shape "
                                       f"{out.shape} for coordinates of shape {shape}")
            return out
        pts = []
        for i, a in enumerate(args):
            a = np.asarray(a, dtype=float)
            if i == self.iphi:
                a = (a - self.axes[i][0]) % (2 * math.pi) + self.axes[i][0]
            elif self.outside is None:
                a = np.clip(a, self.axes[i][0], self.axes[i][-1])     # (the nearest value)
            pts.append(a.ravel())
        return self.interp(np.column_stack(pts)).reshape(shape)


# --------------------------------------------------------------------------
# numba kernels
# --------------------------------------------------------------------------
@nb.njit(cache=True, nogil=True)
def _deposit_analytic(out, gx, gy, vel, sig, w, sig_instr, vedges, relint, ones, ptab, chmin,
                      axtyp, velsys, lfreq0, crval3, cdelt3, crpix3):
    """Deposit the points (gx, gy: pixels; vel: m/s; sig: m/s; w: weights) in the nearest pixel of out
    (ny, nx, nz), as GALMOD, each with its Gaussian line (dispersion sqrt(sig^2 + sig_instr^2)) integrated
    over the channels, for each line (reference frequency lfreq0, channel edges vedges, intensity relint)."""
    ny, nx, nz = out.shape
    nlines = lfreq0.shape[0]
    prof = np.empty(nz)
    for i in range(gx.shape[0]):
        ix, iy = _lround(gx[i]), _lround(gy[i])
        if ix < 0 or ix >= nx or iy < 0 or iy >= ny:
            continue
        s = math.sqrt(sig[i] * sig[i] + sig_instr * sig_instr)
        if s < 1e-6 * chmin:
            s = 1e-6 * chmin
        K = int(5.0 * s / chmin) + 2
        for nl in range(nlines):
            v = vel[i]
            kc = _lround(_velgrid(v, axtyp, velsys, lfreq0[nl], crval3, cdelt3, crpix3))
            k0, nk = _line_profile(prof, v, s, vedges[nl], ones, ptab, kc, K, nz)
            if nk == 0:
                continue
            a = w[i] * relint[nl]
            o = out[iy, ix]
            for j in range(nk):
                o[k0 + j] += a * prof[j]


@nb.njit(cache=True, nogil=True)
def _deposit_mc(out, gx, gy, vel, w, relint, axtyp, velsys, lfreq0, crval3, cdelt3, crpix3):
    """Deposit the clouds (gx, gy: pixels; vel: m/s, with their random deviates; w: weights) in the nearest
    pixel and channel of out (ny, nx, nz), for each line (reference frequency lfreq0, intensity relint)."""
    ny, nx, nz = out.shape
    for i in range(gx.shape[0]):
        ix, iy = _lround(gx[i]), _lround(gy[i])
        if ix < 0 or ix >= nx or iy < 0 or iy >= ny:
            continue
        for nl in range(lfreq0.shape[0]):
            k = _lround(_velgrid(vel[i], axtyp, velsys, lfreq0[nl], crval3, cdelt3, crpix3))
            if 0 <= k < nz:
                out[iy, ix, k] += w[i] * relint[nl]


# --------------------------------------------------------------------------
# The model
# --------------------------------------------------------------------------
class GalMod3D:
    """A model datacube of a galaxy from 3D fields of the gas, on the grid of a cube.

    GalMod3D(cube, coords="Rz", xpos, ypos, vsys=0, inc, pa, dens, <velocities>, vdisp=0, rmax=None,
             zmax=None, step=None, flux=None, beam=None, **options)

    cube:   the grid of the model: a FITS file, an astropy HDU or (data, header), as the tasks. Its beam
            (or beam=(bmaj, bmin, bpa), arcsec and degrees) is the beam of the model.
    coords: the coordinates of the fields, in the frame of the galaxy:
              "Rz"     functions of (R, z): an axisymmetric system (also "axisymmetric");
              "Rphiz"  functions of (R, phi, z) (also "cylindrical");
              "xyz"    functions of (x, y, z) (also "cartesian").
            R, x, y, z in arcsec; phi in radians, in the plane of the galaxy from the receding half of
            the major axis towards the far side (x = R cos phi, y = R sin phi; x is the major axis, y
            points away from the observer, z along the rotation axis).
    xpos, ypos: the centre (pixels, 0-based, or WCS as '123.4d' or '07h36m51s', as GALMOD).
    vsys, inc, pa: systemic velocity (km/s), inclination and position angle (degrees, of the receding
            half of the major axis, from north to east), as in GALMOD.

    The fields (km/s for the velocities): each is a number, a vectorised function of the coordinates
    (e.g. lambda R, z: ...), or a tuple (axes, values): values on the grid of the coordinates (one 1D
    array per coordinate, increasing), interpolated linearly (outside the grid: 0 for the density,
    the nearest value for the others; phi is periodic).
      dens:   the density of the gas (required), in any units >= 0;
      vr, vphi, vz (coords Rz, Rphiz): the radial, azimuthal (positive: rotation towards increasing
              phi, i.e. the receding side at PA) and vertical velocities;
      vx, vy, vz (coords xyz): the velocity components along x, y, z;
      vdisp:  the isotropic velocity dispersion (default 0: a line as wide as the spectral resolution).
    The line of sight velocity of a point is vsys + v_y sin(inc) + v_z cos(inc), with v_y the
    component along y (vr sin phi + vphi cos phi).

    The grid of the model, in the frame of the galaxy:
      rmax:   |x|, |y| <= rmax (arcsec); default: the distance from the centre to the farthest corner
              of the cube (along the minor axis an inclined disk may extend further: give rmax).
      zmax:   |z| <= zmax (arcsec); default rmax. For disks give a smaller value: the model is much
              faster. zmax=0: a razor-thin disk (then dens is a surface density).
      step:   (dxy, dz), or one value for both, in arcsec: the spacing of the grid; default a third
              of a pixel. dz must resolve the vertical structure of the density.

    flux:   the total flux of the model: a number (Jy km/s, converted to the units of the cube, BUNIT),
            "measured" (the flux of the data of the cube in its mask, see mask), or None (default: the
            sum of the model is the integral of dens over the volume, in units of dens x arcsec^3).
    options: BBarolo parameters of the cube: specres (spectral resolution of the model, sigma in
            channels; default a FWHM of one channel), restwave/restfreq, relint and redshift (several
            lines, as GALMOD), mask (the mask of flux="measured": SEARCH, the default, SMOOTH, NONE,
            ...), threads, verbose.

    Methods: model(galmodtype="ANALYTIC", nclouds=1e6, seed=None, noise=0, smooth=True) -> PrimaryHDU.

    >>> m = GalMod3D("ngc2403.fits", coords="Rz", xpos=77, ypos=77, vsys=132.8, inc=60, pa=123.7,
    ...              dens=lambda R, z: np.exp(-R / 200) * np.exp(-np.abs(z) / 10),
    ...              vphi=lambda R, z: 130 * np.tanh(R / 60), vdisp=8, zmax=60, flux="measured")
    >>> hdu = m.model()
    """

    def __init__(self, cube, coords="Rz", xpos=None, ypos=None, vsys=0., inc=None, pa=None, dens=None,
                 vdisp=0., rmax=None, zmax=None, step=None, flux=None, beam=None, **kwargs):
        key = str(coords).strip().lower()
        key = _COORDS_ALIASES.get(key, key)
        if key not in _COORDS:
            raise BBaroloError(f"GalMod3D: coords must be 'Rz', 'Rphiz' or 'xyz', not {coords!r}")
        self.coords, self.argnames, self.velnames = _COORDS[key]
        options, vel = {}, {}
        for k, v in kwargs.items():
            kl = k.lower()
            if kl in _VELOCITIES:
                if kl not in self.velnames:
                    raise BBaroloError(f"GalMod3D: {k} is not a velocity of coords={self.coords!r}: the velocities "
                                       f"are {', '.join(self.velnames)}")
                vel[kl] = v
            elif kl in _CUBE_OPTIONS:
                options[kl] = v
            else:
                raise TypeError(f"GalMod3D() got an unexpected keyword argument {k!r}")
        if dens is None:
            raise BBaroloError("GalMod3D: dens (the density of the gas) is required")
        for name, v in (("xpos", xpos), ("ypos", ypos), ("inc", inc), ("pa", pa)):
            if v is None:
                raise BBaroloError(f"GalMod3D: {name} is required")
        self.fields = {"dens": _Field("dens", dens, self.argnames, outside=0.),
                       "vdisp": _Field("vdisp", vdisp, self.argnames)}
        for n in self.velnames:
            self.fields[n] = _Field(n, vel.get(n, 0.), self.argnames)
        self.vsys, self.inc, self.pa = float(vsys), float(inc), float(pa)
        self.rmax, self.zmax, self.step = rmax, zmax, step
        if isinstance(flux, str) and flux.strip().lower() != "measured":
            raise BBaroloError(f"GalMod3D: flux must be a number, 'measured' or None, not {flux!r}")
        self.flux = flux
        self._load(cube, beam, options)
        h = self.cube.Head()
        self.xpos, self.ypos = getCenterCoordinates([xpos, ypos], h)

    # ------------------------------------------------------------------ the cube
    def _load(self, cube, beam, options):
        """The cube (a bbarolo Cube, self.cube) with the parameters of the options, and the engine of
        GALMOD on its grid (self.diskmodel: spectral axis, beam convolution, noise)."""
        import tempfile
        from .tasks import Task, build_param, _checked
        from .cube import Cube
        from .galmod import DiskModel
        prm = {k.upper(): v for k, v in options.items()}
        prm.setdefault("MASK", "SEARCH")
        prm.setdefault("VERBOSE", False)
        prm["PLOTS"] = -1                                   # (no mask.fits in the temporary folder)
        t = Task(cube, beam=beam, **prm)
        hdu = t._input_hdu()
        with tempfile.TemporaryDirectory(prefix="bbarolo_") as tmp:
            fits_in = t._cube if hdu is None else "in-memory cube"
            par = _checked(build_param(t._all_params(outfolder=tmp + "/", fitsfile=fits_in)))
            c = Cube(par=par)
            ok = c.readCube(t._cube, printInfo=False) if hdu is None else c.readHDU(hdu, name=fits_in)
        if not ok:
            raise IOError(f"{cube} is not a readable FITS file!")
        c.checkBeam()
        self.cube = c
        self.diskmodel = DiskModel(c, densflux=True, smooth=True)
        # arcsec per pixel (with the sign of CDELT)
        self._cdelt = (self.diskmodel.setup.cdelt[0] * 60., self.diskmodel.setup.cdelt[1] * 60.)

    # ------------------------------------------------------------------ the grid of the galaxy
    def _grid(self):
        """The axes of the grid in the frame of the galaxy: x (= y) and z centres (arcsec), and the
        volume of a point."""
        pix = min(abs(self._cdelt[0]), abs(self._cdelt[1]))
        step = self.step
        if step is None:
            dxy = dz = pix / 3.
        else:
            st = np.ravel(step).astype(float)
            dxy, dz = (st[0], st[0]) if st.size == 1 else (st[0], st[1])
        if not (dxy > 0 and dz > 0):
            raise BBaroloError(f"GalMod3D: step must be positive, not {step}")
        rmax = self.rmax
        if rmax is None:
            nx, ny = self.cube.DimX(), self.cube.DimY()
            corners = [(0, 0), (nx - 1, 0), (0, ny - 1), (nx - 1, ny - 1)]
            rmax = max(math.hypot((cx - self.xpos) * self._cdelt[0], (cy - self.ypos) * self._cdelt[1])
                       for cx, cy in corners) + pix
        zmax = rmax if self.zmax is None else float(self.zmax)
        if not rmax > 0 or zmax < 0:
            raise BBaroloError(f"GalMod3D: rmax must be > 0 and zmax >= 0 (rmax={rmax}, zmax={zmax})")
        n = max(int(math.ceil(2 * rmax / dxy)), 1)
        xs = -0.5 * n * dxy + (np.arange(n) + 0.5) * dxy
        if zmax == 0:
            zs, vol = np.zeros(1), dxy * dxy
        else:
            nzg = max(int(math.ceil(2 * zmax / dz)), 1)
            zs = -0.5 * nzg * dz + (np.arange(nzg) + 0.5) * dz
            vol = dxy * dxy * dz
        return xs, zs, vol, (dxy, dz if zmax > 0 else 0.)

    def _args(self, x, y, z):
        """The coordinates of the fields at the points (x, y, z) of the frame of the galaxy."""
        if self.coords == "xyz":
            return (x, y, z)
        R = np.hypot(x, y)
        if self.coords == "Rz":
            return (R, z)
        return (R, np.arctan2(y, x) % (2 * math.pi), z)

    def _project(self, x, y, z, args):
        """Sky pixels (gx, gy), line of sight velocity (m/s) and dispersion (m/s) of the points."""
        si, ci = math.sin(math.radians(self.inc)), math.cos(math.radians(self.inc))
        ang = math.radians(self.pa) + self.diskmodel.setup.crota2
        spa, cpa = math.sin(ang), math.cos(ang)
        ys = y * ci - z * si
        gx = self.xpos + (x * spa - ys * cpa) / self._cdelt[0]
        gy = self.ypos + (x * cpa + ys * spa) / self._cdelt[1]
        f = self.fields
        if self.coords == "xyz":
            vy = f["vy"](args)
        else:
            R = args[0]
            with np.errstate(invalid="ignore", divide="ignore"):
                sphi = np.where(R > 0, y / np.where(R > 0, R, 1.), 0.)
                cphi = np.where(R > 0, x / np.where(R > 0, R, 1.), 1.)
            vy = f["vr"](args) * sphi + f["vphi"](args) * cphi
        vlos = (self.vsys + vy * si + f["vz"](args) * ci) * 1000.
        sig = np.abs(f["vdisp"](args)) * 1000.
        return gx, gy, vlos, sig

    def _chunks(self, xs, zs):
        """Groups of z planes of the grid, with about 2 million points each."""
        nplane = xs.size * xs.size
        per = max(1, 2_000_000 // nplane)
        for k in range(0, zs.size, per):
            yield zs[k:k + per]

    def _plane_points(self, xs, zc):
        """The grid points (x, y, z) of the z planes zc, as flat arrays."""
        x, y, z = np.meshgrid(xs, xs, zc, indexing="ij")
        return x.ravel(), y.ravel(), z.ravel()

    def _density(self, args):
        d = np.array(self.fields["dens"](args), dtype=float)
        bad = ~np.isfinite(d)
        if np.any(d[~bad] < 0):
            raise BBaroloError("GalMod3D: the density is negative somewhere")
        d[bad] = 0.
        return d

    # ------------------------------------------------------------------ the model
    def model(self, galmodtype="ANALYTIC", nclouds=1e6, seed=None, noise=0., smooth=True):
        """The model cube (an astropy PrimaryHDU with the header of the cube).

        galmodtype: "ANALYTIC" (default: every point of the grid with its line integrated over the
                    channels, no noise) or "MC" (nclouds random clouds drawn from the density, each
                    with a random velocity from its line, as the MC models of GALMOD; seed: their seed).
        noise:      rms of a Gaussian noise added to the model, smoothed to the beam (seed: its seed).
        smooth:     convolve the model with the beam (default True)."""
        typ = str(galmodtype).strip().upper()
        if typ not in ("ANALYTIC", "MC"):
            raise BBaroloError(f"GalMod3D: galmodtype must be ANALYTIC or MC, not {galmodtype!r}")
        rng = np.random.default_rng(seed)
        su = self.diskmodel.setup
        c = self.cube
        out = np.zeros((c.DimY(), c.DimX(), c.DimZ()), dtype=np.float64)
        xs, zs, vol, steps = self._grid()
        spec = (su.axtyp, su.velsys, su.lfreq0, su.crval3, su.cdelt3, su.crpix3)
        if typ == "ANALYTIC":
            ones = np.ones(c.DimZ())
            for zc in self._chunks(xs, zs):
                x, y, z = self._plane_points(xs, zc)
                args = self._args(x, y, z)
                d = self._density(args)
                keep = d > 0
                if not np.any(keep):
                    continue
                x, y, z, d = x[keep], y[keep], z[keep], d[keep]
                args = tuple(a[keep] for a in args)
                gx, gy, vlos, sig = self._project(x, y, z, args)
                _deposit_analytic(out, gx, gy, vlos, sig, d * vol, su.sig_instr, su._vedges_lines, su.relint,
                                  ones, _PHI_TAB, su._chmin, *spec)
        else:
            self._model_mc(out, xs, zs, vol, steps, int(nclouds), rng, spec)
        total = out.sum()
        if not total > 0:
            raise BBaroloError("GalMod3D: the model has no emission inside the cube (check the centre, the "
                               "density, rmax and zmax)")
        m = out.transpose(2, 0, 1)
        dm = self.diskmodel
        if smooth and dm.conv is not None:
            m = dm.conv(m, workers=c.pars().threads)
            m[m < 1e-12 * np.max(m)] = 0.
        m = np.ascontiguousarray(m)
        m *= self._flux_scale(m)
        if noise:
            m = dm.addnoise(m, float(noise), rng=rng)
        from .cube import Cube
        mod = Cube.fromArray(m.astype(np.float32), c.Head().copy(), c.pars())
        return model_hdu(mod, c.Head().Name() + "mod")

    def _model_mc(self, out, xs, zs, vol, steps, nclouds, rng, spec):
        """MC: nclouds clouds drawn from the density on the grid (cells chosen with probability
        dens x volume, uniformly within the cell), each with a velocity drawn from its line."""
        su = self.diskmodel.setup
        if nclouds < 1:
            raise BBaroloError(f"GalMod3D: nclouds must be >= 1, not {nclouds}")
        chunks = list(self._chunks(xs, zs))
        totals = []
        for zc in chunks:                                   # (first pass: the weight of each chunk)
            totals.append(float(self._density(self._args(*self._plane_points(xs, zc))).sum()))
        totals = np.array(totals)
        if not totals.sum() > 0:
            return
        counts = rng.multinomial(nclouds, totals / totals.sum())
        w = totals.sum() * vol / nclouds
        dxy, dz = steps
        for zc, n in zip(chunks, counts):
            if n == 0:
                continue
            x, y, z = self._plane_points(xs, zc)
            d = self._density(self._args(x, y, z))
            idx = rng.choice(d.size, size=n, p=d / d.sum())
            x = x[idx] + (rng.random(n) - 0.5) * dxy
            y = y[idx] + (rng.random(n) - 0.5) * dxy
            z = z[idx] + (rng.random(n) - 0.5) * dz
            args = self._args(x, y, z)
            gx, gy, vlos, sig = self._project(x, y, z, args)
            vlos = vlos + rng.standard_normal(n) * np.sqrt(sig * sig + su.sig_instr ** 2)
            _deposit_mc(out, gx, gy, vlos, np.full(n, w), su.relint, *spec)

    def _flux_scale(self, m):
        """The factor that scales the model m (nz, ny, nx) to the total flux of self.flux."""
        if self.flux is None:
            return 1.
        c = self.cube
        dv = np.abs(np.diff(self.diskmodel.setup._vedges)) / 1000.          # km/s of each channel
        model_sum = float(np.sum(m.sum(axis=(1, 2)) * dv))
        if isinstance(self.flux, str):                                     # "measured": the data in the mask
            if not c.MaskAll():
                c.BlankMask()
            data = np.where(c.Mask() > 0, c.array, 0.)
            data_sum = float(np.nansum(np.nansum(data, axis=(1, 2)) * dv))
            if not data_sum > 0:
                raise BBaroloError("GalMod3D: flux='measured': the data of the cube have no positive flux in the "
                                   f"mask (MASK={c.pars().MaskType})")
            return data_sum / model_sum
        h = c.Head()
        if not isFluxUnitKnown(h):
            raise BBaroloError(f"GalMod3D: cannot convert a flux in Jy km/s to BUNIT={h.Bunit()!r}: give "
                               "flux='measured' or None")
        return float(self.flux) / float(FluxtoJy(model_sum, h))
