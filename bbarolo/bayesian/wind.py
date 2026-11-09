"""
BayesianWind (the task BAYESIANWIND): fit a bi-conical wind to a datacube by sampling the posterior of its
geometry and kinematics with nested sampling (dynesty or nautilus), the emission of the wind marginalised.

The model is that of GALWIND (bbarolo.windmodel: the cones along the rotation axis of a disk of inclination INC and
position angle PA, opening angles OPENIN to OPENANG, out to HTOT, a radial velocity VWIND, a dispersion VDISP and a
rotation VROT around the axis, per shell), and its emission is not assumed: it is a set of emissivities on cells of
the cones (shells x polar bins x azimuthal sectors, each cone its own; option emission, see below), linear in the
model, solved at every evaluation of the likelihood and marginalised, with Gaussian priors of mean
EMSCALE (r/r0)^-EMSLOPE and fractional scatter EMCLUMP (the clumpiness; see bbarolo.bayesian.emission). A patchy
wind is then a set of emissivities, not a bias of the geometry, and the marginalisation penalises the cones and
velocities larger than the data need (cells that the prior expects to emit and the data find empty), which the
profile likelihood (marginal=False) does not: with correlated noise the profile likelihood was flat over 25 degrees
of opening angle of a patchy mock, the marginal one peaked at the truth.

    >>> from bbarolo import BayesianWind
    >>> f = BayesianWind("wind.fits", xpos=31.5, ypos=31.5, vsys=0, inc=60, pa=30, openang=60, htot=200,
    ...                  vwind=200, vdisp=20, outfolder="output/wind")
    >>> f.run(freepar=['vwind_single', 'vdisp_single', 'inc', 'pa', 'openang', 'emscale'], threads=8)
    >>> f.print_stats(); f.write()

Parameter files: the task BAYESIANWIND, with the initial wind as for GALWIND (XPOS, YPOS, VSYS, INC, PA, OPENANG,
OPENIN, HTOT, VWIND, VDISP, VROT; the shells from RADII or NTOT, by default one per beam), the free parameters in FREE
(e.g. VWIND_SINGLE VDISP_SINGLE INC PA OPENANG EMSCALE), the priors in PRIORS, the sampler in SAMPLER, SAMPLEROPTS,
RUNOPTS and THREADS, the options of the likelihood in LIKEOPTS, and an optional fixed disk in DISKRINGS.

Free parameters: VWIND, VDISP, VROT (one value per shell, NAME_SINGLE one value for all, NAME_FUNC a function of
radius, see funcs); XPOS, YPOS, VSYS, INC, PA, OPENANG, OPENIN, HTOT (single values); SIGMA (the noise); EMSCALE,
EMCLUMP, EMRATIO and EMSLOPE (the prior of the emissivities: its mean at r0, the median radius of the cells, in the
receding cone (+z), its fractional scatter, the ratio of the mean of the approaching cone to that of the receding one,
and its power-law slope in radius). They are always fitted, added to the free parameters unless fixed with the
options emscale, emclump, emratio and emslope: fixed at a rough value they bias the geometry (the two cones of a wind
often differ, e.g. the far one extincted, and its emission fades outwards: EMSLOPE fixed at 0 for an emission falling
as r^-2 gave VWIND 24% low and INC 11 degrees low).

Options of the likelihood (LIKEOPTS, run(like_kwargs=...)):
  emission   the basis of the emission: 'cells' (default: cells about cellsize across, fewer near the apex),
             'shells' (one emissivity per shell and cone), 'cone' (one per cone, a power law r^-EMSLOPE), 'spaxel'
             (a uniform cone normalised in each spaxel, as NORM LOCAL), 'voxel' (MoKA-3D: the model equals the
             data in the voxels the uniform cone reaches; for comparison only: it is degenerate)
  cellsize   the size of the cells (arcsec, default the beam FWHM)
  marginal   true (default): the likelihood marginalised over the emission; false: its maximum (profile)
  prior      the prior of the emissivities: 'gauss' (default: Gaussian, exact marginal likelihood) or 'exp'
             (exponential, non-negative emissivities: an approximate marginal likelihood, see emission.py)
  noisecorr  the correlation of the noise, as BAYESIAN3DFIT: 'measured' (default), 'beam' or a number of voxels
  sigma      fixed noise per voxel (default: the measured rms)
  datafrac   a model error proportional to the data: the variance of a voxel is sigma^2 + (datafrac*data)^2
             (default 0), for the structure that the cells cannot follow
  grow       the region of the likelihood: the projected cones of the widest prior grown by grow beams (default 1)
  prune      the values of the columns of the model below prune times their peak are left out of the likelihood (the
             faint wings, where the tails of the line and of the beam multiply; default 1e-3: about 0.3% of the flux
             of a cell, and the likelihood ten times faster; 0 keeps them all)
  emscale    a fixed EMSCALE (the mean emissivity at r0): a number, or 'fit' (the value that maximises the marginal
             likelihood at each evaluation, about 20 times slower); default: free
  emclump    a fixed EMCLUMP (prior 'gauss'); default: free
  emratio    a fixed EMRATIO (e.g. 1: the cones have the same mean emissivity); default: free
  emslope    a fixed EMSLOPE (e.g. 0: a mean emissivity uniform in radius); default: free

The likelihood uses all the voxels of its region (no mask: the empty voxels are what tell a cone that is too
large). The fixed disk (DISKRINGS: a rings file of 3DFIT or BAYESIAN3DFIT) adds one model per ring with its
geometry and kinematics, whose densities are non-negative amplitudes solved and marginalised with the emission.
"""

import os
import math
import copy
import numpy as np

from ..tasks import Task, build_param, _checked
from ..bbutils import BBaroloError, getCenterCoordinates, DeltaVel
from ..windmodel import WindModel, WindCells, WindPars, Boxes
from .emission import log_marginal, log_marginal_diag, log_profile, log_marginal_gauss, log_marginal_gauss_diag
from .common import vprint, get_distribution, SIGMA_QUANTILES
from .sampling import sample, sampler_options

WIND_SHELL_PARS = ("vwind", "vdisp", "vrot")
WIND_SCALAR_PARS = ("xpos", "ypos", "vsys", "inc", "pa", "openang", "openin", "htot")
WIND_OTHER_PARS = ("sigma", "emscale", "emclump", "emratio", "emslope")
WIND_FUNC_PARS = WIND_SHELL_PARS
_WIND_ALIASES = {"phi": "pa", "disp": "vdisp"}
EMISSION_BASES = ("cells", "shells", "cone", "spaxel", "voxel")

# the default functions of radius (arcsec) of VWIND_FUNC, VDISP_FUNC and VROT_FUNC (BayesianWind.funcs)
DEFAULT_WIND_FUNCS = dict(vwind=lambda R, p1, p2: 2. / np.pi * p1 * np.arctan(R / p2),   # accelerating to p1
                          vdisp=lambda R, p1, p2: p1 + p2 * R,                         # linear
                          vrot=lambda R, p1, p2: p1 + p2 * R)                          # linear

WIND_LIKE_OPTIONS = dict(emission="cells", cellsize=None, marginal=True, prior="gauss", noisecorr="measured",
                         sigma=None, datafrac=0., grow=1., emscale=None, emclump=None, emratio=None, emslope=None,
                         prune=1e-3)
# the hyperparameters of the prior of the emission, free unless fixed (LIKEOPTS emscale, emclump, emratio, emslope)
HYPER = dict(gauss=("emscale", "emclump", "emratio", "emslope"), exp=("emscale", "emratio", "emslope"))


def wind_like_option(name, value):
    """An option of the likelihood of BayesianWind, checked: (name, value); None if name is not an option. Raises
    ValueError for a wrong value (None gives the default)."""
    from ..params import _optvalue
    key = str(name).strip().lower()
    if key not in WIND_LIKE_OPTIONS:
        return None
    if value is None or (isinstance(value, str) and value.strip().lower() == "none"):
        return key, WIND_LIKE_OPTIONS[key]
    if key == "emission":
        v = str(value).strip().lower()
        if v not in EMISSION_BASES:
            raise ValueError(f"emission: {', '.join(EMISSION_BASES)}, not {value!r}")
        return key, v
    if key == "prior":
        v = str(value).strip().lower()
        if v not in ("gauss", "exp"):
            raise ValueError(f"prior: gauss or exp, not {value!r}")
        return key, v
    if key == "marginal":
        if isinstance(value, str):
            value = _optvalue(value.strip())
        if isinstance(value, (bool, np.bool_)) or value in (0, 1):
            return key, bool(value)
        raise ValueError(f"marginal: true or false, not {value!r}")
    if key == "noisecorr":
        if isinstance(value, str) and value.strip().lower() in ("measured", "beam"):
            return key, value.strip().lower()
    if key == "emscale" and isinstance(value, str) and value.strip().lower() == "fit":
        return key, "fit"
    if key == "emslope":                            # (any sign)
        try:
            x = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"emslope: expected a number, not {value!r}") from None
        if not np.isfinite(x):
            raise ValueError(f"emslope: expected a finite number, not {value!r}")
        return key, x
    try:
        x = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{key}: expected a number, not {value!r}") from None
    pos = ("cellsize", "sigma", "noisecorr", "emscale", "emclump", "emratio")
    if not np.isfinite(x) or x < 0 or (x == 0 and key in pos):
        raise ValueError(f"{key}: expected a number {'>' if key in pos else '>='} 0, not {value!r}")
    return key, x


def wind_likeopts(opts):
    """The options of the likelihood of BAYESIANWIND from LIKEOPTS (as in a parameter file). Raises ValueError."""
    from ..params import bayes_options
    out = {}
    for k, v in bayes_options(opts, none=False).items():
        kv = wind_like_option(k, v)
        if kv is None:
            raise ValueError(f"unknown option '{k}' (options: {', '.join(WIND_LIKE_OPTIONS)})")
        out[kv[0]] = kv[1]
    return out


def wind_freepar(free, funcs=WIND_FUNC_PARS):
    """The free parameters of BAYESIANWIND from FREE: 'VWIND_SINGLE INC PA OPENANG' -> ['vwind_single', 'inc', 'pa',
    'openang']. Raises ValueError."""
    from ..params import bayes_freepar
    return bayes_freepar(free, funcs=funcs, ring_pars=WIND_SHELL_PARS + WIND_SCALAR_PARS,
                         other_pars=WIND_OTHER_PARS, aliases=_WIND_ALIASES)


def wind_priors(priors):
    """The priors of BAYESIANWIND from PRIORS (as BAYESIAN3DFIT, with the names of the wind). Raises ValueError."""
    from ..params import bayes_priors
    return bayes_priors(priors, ring_pars=WIND_SHELL_PARS + WIND_SCALAR_PARS, other_pars=WIND_OTHER_PARS,
                        func_pars=WIND_FUNC_PARS, aliases=_WIND_ALIASES)


class BayesianWind(Task):
    """Fit a bi-conical wind to a datacube in a Bayesian framework (see the module documentation).

    BayesianWind(cube, **params): the cube (FITS file, HDU or (data, header)) and any BBarolo parameter: the initial
    wind as for GALWIND (XPOS, YPOS, VSYS, INC, PA, OPENANG, HTOT, VWIND are required; OPENIN, VDISP, VROT, RADII,
    NTOT, DISKRINGS optional) and the options FREE, PRIORS, SAMPLER, SAMPLEROPTS, RUNOPTS and LIKEOPTS (the defaults of
    run()).

    Methods: run(freepar, method, threads, like_kwargs, sampler_kwargs, run_kwargs, verbose, dynamic, **params) samples
    the posterior (nothing is written); write(outfolder, plots) writes the outputs; print_stats(), corner_plot(),
    best_model(), emission_errors(), shells_table(); log_likelihood(theta) and the parameters of theta (wind_pars).

    Attributes: priors (definitions, see bbarolo.params.bayes_priors), funcs (functions of radius), like_kwargs, the
    results samples, weights, samples_equal, quantiles, params (the medians), best (the WindPars of the medians),
    emission (the emissivities of the cells at the best fit), model (the best model cube).
    """

    TASK = "BAYESIANWIND"
    _KEYWORD_OPTIONS = ("FREE", "SAMPLER", "PRIORS", "SAMPLEROPTS", "RUNOPTS", "LIKEOPTS")

    def __init__(self, cube=None, **params):
        self.taskname = "BAYESIANWIND"
        self.freepar_idx = {}
        self.freepar_names = []
        self.ndim = 0
        self.priors = {}
        self.prior_distr = None
        self._default_pri = {}
        self.funcs = dict(DEFAULT_WIND_FUNCS)
        self.like_kwargs = dict(WIND_LIKE_OPTIONS)
        self.results = self.sampler = None
        self.samples = self.weights = self.samples_equal = self.quantiles = self.params = None
        self.best = self.emission = self.model = None
        self.modCalculated = False
        self._method = None
        self._reset()
        super(BayesianWind, self).__init__(cube, **params)

    def _reset(self):
        self._cube_obj = self._par = None
        self._init = None
        self._wm = None
        self._setup_key = None

    # ------------------------------------------------------------------ parameters
    def update(self, **params):
        """Change BBarolo parameters (None restores the default); FREE, SAMPLER, SAMPLEROPTS, RUNOPTS and LIKEOPTS set
        the defaults of run(), PRIORS entries of self.priors (strings as in parameter files, or Python values).
        Returns self."""
        from ..keywords import lookup
        from ..params import bayes_value_string
        given = {}
        for k, v in list(params.items()):
            kw = lookup(k)
            if kw is not None and kw.name in self._KEYWORD_OPTIONS and v is not None and not isinstance(v, str):
                params[k] = bayes_value_string(kw.name, v)
        for k, v in params.items():
            kw = lookup(k)
            if kw is not None and kw.name in self._KEYWORD_OPTIONS and v is not None:
                given[kw.name] = self._keyword_option(kw.name, v)
        super(BayesianWind, self).update(**params)
        if not hasattr(self, "priors"):
            return self
        if "PRIORS" in given:
            self.priors.update(given["PRIORS"])
            if self.prior_distr is not None:
                for k in given["PRIORS"]:
                    self.prior_distr.pop(k, None)
        if params:
            self._reset()
        return self

    def _keyword_option(self, name, value):
        s = str(value)
        try:
            if name == "FREE":
                return wind_freepar(s, funcs=set(WIND_FUNC_PARS) | set(getattr(self, "funcs", {})))
            if name == "LIKEOPTS":
                return wind_likeopts(s)
            if name == "PRIORS":
                return wind_priors(s)
            if name == "SAMPLER":
                m = s.strip().lower()
                if m not in ("dynesty", "nautilus"):
                    raise ValueError(f"unknown sampler {s.strip()} (DYNESTY or NAUTILUS)")
                return m
            from ..params import bayes_options
            return bayes_options(s)
        except ValueError as e:
            raise ValueError(f"BAYESIANWIND: {name}: {e}") from None

    def _run_defaults(self):
        return {k: self._keyword_option(k, self._params[k])
                for k in self._KEYWORD_OPTIONS if k != "PRIORS" and self._params.get(k) is not None}

    def _outfolder(self):
        of = self._params.get("OUTFOLDER")
        of = os.path.expanduser(str(of).strip()) if of not in (None, "") else self._default_outfolder()
        return of if of.endswith("/") else of + "/"

    def _make_par(self):
        """The Param of the fit (the task flag and FREE out: the checks of the task need a FREE that run() may give)."""
        hdu = self._input_hdu()
        prm = self._all_params(outfolder=self._outfolder(), fitsfile=self._cube if hdu is None else "in-memory cube")
        prm.pop(self.TASK, None)
        prm.pop("FREE", None)
        prm["VERBOSE"] = False
        prm["THREADS"] = 1
        return _checked(build_param(prm))

    def _load_cube(self):
        """Read the cube, build the mask of its emission (MASK, default SEARCH: all the detections) and measure its
        noise."""
        par = self._make_par()
        c = self._read_cube(par)
        p = c.pars()
        if p.outFolder == "":
            p.setOutfolder(self._outfolder())
        self._cube_obj, self._par = c, p
        self.data = np.asarray(c.array, dtype=np.float64)
        if str(p.MaskType).upper() != "NONE":
            if not c.StatsDef():
                c.setCubeStats()
            if not c.MaskAll():
                c.BlankMask(None, False)
        self.mask = c.Mask() if c.MaskAll() else np.zeros(self.data.shape, bool)
        out = self.data[(~self.mask.astype(bool)) & np.isfinite(self.data)]
        if out.size == 0:
            out = self.data[np.isfinite(self.data)]
        self.noiserms = float(1.4826 * np.median(np.abs(out - np.median(out))))
        if not self.noiserms > 0:
            self.noiserms = float(np.nanstd(out)) if out.size else 1.
        h = c.Head()
        self._pix = abs(h.Cdelt(0)) * 3600.
        self._chw = abs(DeltaVel(h))
        bmaj, bmin = abs(h.Bmaj()) * 3600., abs(h.Bmin()) * 3600.
        self._beam = math.sqrt(bmaj * bmin) if bmaj > 0 and bmin > 0 else self._pix
        self.beam_area = float(np.pi * bmaj * bmin / (4 * np.log(2)) / self._pix ** 2) if bmaj > 0 else 1.
        self.noise_correlation = None
        return c

    def _initial(self):
        """The initial wind from the parameters (as GALWIND): a dict of the scalars (pixels, km/s, degrees, arcsec) and
        the shells (radii, vwind, vdisp, vrot)."""
        from ..rings import ringList, getDataColumn as _gdc
        c = self._cube_obj if self._cube_obj is not None else self._load_cube()
        gw = self._par.gw

        def need(name, s):
            v = str(s).strip()
            if v in ("", "-1"):
                raise BBaroloError(f"BAYESIANWIND: {name} is not an optional parameter (the initial wind, as GALWIND).")
            return v

        def values(name, s, n=None):
            s = str(s)
            col = _gdc(s) if "file(" in s.lower() else ringList(s)
            if col is None:
                col = [float(s.strip())]
            a = np.asarray(col, float)
            if n is not None and a.size not in (1, n):
                raise BBaroloError(f"BAYESIANWIND: {name} has {a.size} values, not one or {n} (one per shell)")
            return a if n is None or a.size == n else np.full(n, a[0])
        x0, y0 = getCenterCoordinates([need("XPOS", gw.XPOS), need("YPOS", gw.YPOS)], c.Head())
        ini = dict(xpos=x0, ypos=y0, vsys=float(need("VSYS", gw.VSYS)), inc=float(need("INC", gw.INC)),
                   pa=float(need("PA", gw.PHI)), openang=float(values("OPENANG", need("OPENANG", gw.OPENANG))[0]),
                   openin=float(values("OPENIN", getattr(gw, "OPENIN", "0") or "0")[0]),
                   htot=float(gw.HTOT))
        if not ini["htot"] > 0:
            raise BBaroloError("BAYESIANWIND: HTOT is not an optional parameter (the length of the cones, arcsec).")
        radii = None
        rs = str(gw.RADII).strip()
        if rs not in ("", "-1"):
            radii = values("RADII", rs)
        else:
            n = int(gw.NTOT) if "NTOT" in self._params else max(1, min(15, int(round(ini["htot"] / self._beam))))
            radii = (np.arange(n) + 0.5) * ini["htot"] / n
        n = radii.size
        ini["radii"] = np.asarray(radii, float)
        ini["vwind"] = values("VWIND", need("VWIND", gw.VWIND), n)
        vd = str(gw.VDISP).strip()
        ini["vdisp"] = values("VDISP", vd, n) if vd not in ("", "-1") else np.full(n, 20.)
        vr = str(gw.VROT).strip()
        ini["vrot"] = values("VROT", vr, n) if vr not in ("", "-1") else np.zeros(n)
        if not 0 <= ini["inc"] <= 90:
            raise BBaroloError(f"BAYESIANWIND: INC must be between 0 and 90 degrees (the cone along +z recedes; PA "
                               f"sets its side), not {ini['inc']:g}")
        self._init = ini
        self._set_default_priors()
        return ini

    # ------------------------------------------------------------------ priors
    def _set_default_priors(self):
        """The default priors, from the initial wind: XPOS, YPOS +- one beam; VSYS +- 5 channels (at least 20 km/s);
        INC +- 15 deg within 0-90; PA +- 20 deg; OPENANG +- 30 deg within 1-180; OPENIN 0 to OPENANG - 1; HTOT 0.5 to
        1.5 times; VWIND 0 to twice the largest (at least 100 km/s); VDISP 0 to 3 times the largest (at least 50); VROT
        +- twice the largest (at least 50); SIGMA log-uniform 0.5-1000 times the noise; EMSCALE log-uniform over six
        decades around the mean emissivity of the data; EMSLOPE -2 to 6. A prior given (priors) is kept."""
        ini = self._init
        b = self._beam / self._pix
        vmax = max(100., 2 * float(np.max(np.abs(ini["vwind"]))))
        dmax = max(50., 3 * float(np.max(ini["vdisp"])))
        rmax = max(50., 2 * float(np.max(np.abs(ini["vrot"]))))
        U = lambda lo, hi: dict(name="uniform", loc=float(lo), scale=float(hi - lo))
        dv = max(5 * self._chw, 20.)
        pri = dict(xpos=U(ini["xpos"] - b, ini["xpos"] + b), ypos=U(ini["ypos"] - b, ini["ypos"] + b),
                   vsys=U(ini["vsys"] - dv, ini["vsys"] + dv),
                   inc=U(max(0., ini["inc"] - 15), min(90., ini["inc"] + 15)),
                   pa=U(ini["pa"] - 20, ini["pa"] + 20),
                   openang=U(max(1., ini["openang"] - 30), min(180., ini["openang"] + 30)),
                   openin=U(0., max(1., ini["openang"] - 1.)), htot=U(0.5 * ini["htot"], 1.5 * ini["htot"]),
                   vwind=U(0., vmax), vdisp=U(0., dmax), vrot=U(-rmax, rmax),
                   sigma=dict(name="loguniform", a=0.5 * self.noiserms, b=1000. * self.noiserms),
                   emscale=dict(name="loguniform", a=1e-3 * self._s0, b=1e3 * self._s0) if self._s0 else None,
                   emslope=U(-2., 6.), emclump=dict(name="loguniform", a=0.1, b=10.),
                   emratio=dict(name="loguniform", a=1e-3, b=1e3),
                   vwind_p1=U(0., vmax), vwind_p2=U(0.02 * ini["htot"], ini["htot"]),
                   vdisp_p1=U(0., dmax), vdisp_p2=U(-dmax / ini["htot"], dmax / ini["htot"]),
                   vrot_p1=U(-rmax, rmax), vrot_p2=U(-rmax / ini["htot"], rmax / ini["htot"]))
        for k, v in pri.items():
            if v is None:
                continue
            if k not in self.priors or self.priors.get(k) == self._default_pri.get(k):
                self.priors[k] = v
                self._default_pri[k] = v

    @property
    def _s0(self):
        """The mean emissivity of the data: the flux in the mask over the volume of the initial cones (per arcsec^3, in
        the units of the model: flux per pixel and channel)."""
        ini = self._init
        if ini is None:
            return None
        flux = float(np.nansum(np.where(self.mask.astype(bool), self.data, 0.)))
        th1, th0 = math.radians(ini["openang"] / 2), math.radians(ini["openin"] / 2)
        vol = 2 * 2 * math.pi / 3 * ini["htot"] ** 3 * (math.cos(th0) - math.cos(th1))
        return abs(flux) / vol if vol > 0 and flux != 0 else 1.

    def _parse_freepar(self, freepar):
        """The free parameters (names as in run): ndim, freepar_idx, freepar_names."""
        self.freepar_idx, self.freepar_names, self.ndim = {}, [], 0
        nsh = self._init["radii"].size
        for name in wind_freepar(" ".join(freepar) if not isinstance(freepar, str) else freepar,
                                 funcs=set(WIND_FUNC_PARS) | set(self.funcs)):
            base, _, suffix = name.partition("_")
            if suffix == "func":
                if base not in self.funcs:
                    raise ValueError(f"BAYESIANWIND: no function of radius for {base.upper()} (see funcs)")
                npar = self.funcs[base].__code__.co_argcount - 1
                for i in range(npar):
                    key = f"{base}_p{i + 1}"
                    self.freepar_idx[key] = np.array([self.ndim])
                    self.freepar_names.append(key)
                    self.ndim += 1
            elif base in WIND_SHELL_PARS and suffix == "":
                self.freepar_idx[base] = np.arange(self.ndim, self.ndim + nsh)
                self.freepar_names += [f"{base}{i + 1}" for i in range(nsh)]
                self.ndim += nsh
            else:
                self.freepar_idx[base] = np.array([self.ndim])
                self.freepar_names.append(base)
                self.ndim += 1
        self._free_bases = {k.split("_")[0] for k in self.freepar_idx}

    def _freeze_priors(self):
        self.prior_distr = {} if self.prior_distr is None else self.prior_distr
        for key in self.freepar_idx:
            base = key
            if key not in self.priors:
                raise ValueError(f"BAYESIANWIND: no prior for {key}")
            d = self.priors[key]
            kw = {k: v for k, v in d.items() if k != "name"}
            if key == "sigma":
                for k, f in (("a", 0.5), ("b", 1000.), ("scale", 1.)):
                    if k in kw and kw[k] is None:
                        kw[k] = f * self.noiserms
            try:
                self.prior_distr[base] = get_distribution(d["name"], **kw)
            except Exception:
                raise ValueError(f"BAYESIANWIND: something is wrong in the prior of {key}: {d}")

    def _prior_transform(self, u):
        p = np.zeros_like(u)
        for key, idx in self.freepar_idx.items():
            p[idx] = self.prior_distr[key].ppf(u[idx])
        return p

    def _bounds(self, key):
        """The range of a parameter: the support of its prior (99.9% for unbounded priors) if free, else its value."""
        if key in self.freepar_idx:
            d = self.prior_distr[key]
            lo, hi = d.support()
            if not np.isfinite(lo):
                lo = d.ppf(1e-3)
            if not np.isfinite(hi):
                hi = d.ppf(1 - 1e-3)
            return float(lo), float(hi)
        v = self._init[key]
        return (float(np.min(v)), float(np.max(v))) if np.ndim(v) else (float(v), float(v))

    # ------------------------------------------------------------------ setup
    def _setup(self, freepar, verbose=True):
        """The setup of a fit with the free parameters freepar: the initial wind and its default priors, the free
        parameters, the priors (frozen), the engine (the cells of the emission and their quadrature, from the bounds
        of the priors), the region of the likelihood and its weights, and the fixed disk."""
        if self._init is None:
            self._initial()
        self._parse_freepar(freepar)
        self._freeze_priors()
        lk = self.like_kwargs
        ini = self._init
        # (the bounds of the free parameters: the cells cover the largest cones, the quadrature the fastest wind)
        rmax = self._bounds("htot")[1]
        thmax = min(90., 0.5 * self._bounds("openang")[1])
        vw = max(abs(x) for x in self._shell_range("vwind"))
        vr = max(abs(x) for x in self._shell_range("vrot"))
        sigmin = max(0., min(self._shell_range("vdisp")))
        size = float(lk.get("cellsize") or self._beam)
        basis = lk["emission"]
        if basis == "cells":
            cells = WindCells.resolved(rmax, thmax, size)
        elif basis == "shells":
            r = ini["radii"]
            edges = np.concatenate([[0.], 0.5 * (r[1:] + r[:-1]), [max(rmax, r[-1] + 0.5 * (r[-1] - (r[-2] if r.size > 1
                                                                                                  else 0.)))]])
            cells = WindCells(edges, 1, 1, thmax=thmax)
        else:
            cells = WindCells([0., rmax], 1, 1, thmax=thmax)
        self._wm = WindModel(self._cube_obj, cells).quadrature(vmax=vw + vr, sigmin=sigmin)
        self._r0 = float(np.median(0.5 * (cells.cells[:, 1] + cells.cells[:, 2])))
        self._setup_region()
        self._setup_disk()
        self._setup_key = (tuple(self.freepar_names), basis, size)
        if verbose:
            vprint(verbose, f" BAYESIANWIND: {cells.ncell} {basis} ({cells.npoints} quadrature points), region of "
                            f"{int(self._region2d.sum())} pixels x {self._lim[5] - self._lim[4]} channels, "
                            f"likelihood x {self.likefactor:.3g} ({self._likefactor_note})")

    def _shell_range(self, base):
        """The range of the values of a shell parameter allowed by the priors (free) or given (fixed)."""
        ini = self._init
        if base in self.freepar_idx:
            return self._bounds(base)
        if f"{base}_p1" in self.freepar_idx:
            R = np.linspace(0, self._bounds("htot")[1], 50)
            f = self.funcs[base]
            npar = f.__code__.co_argcount - 1
            rng = np.random.default_rng(0)
            vals = [f(R, *[self.prior_distr[f"{base}_p{i + 1}"].ppf(rng.random()) for i in range(npar)])
                    for _ in range(200)]
            return float(np.min(vals)), float(np.max(vals))
        return float(np.min(ini[base])), float(np.max(ini[base]))

    def _corner_pars(self):
        """WindPars at the corners of the box of the priors of the geometry (for the region of the likelihood)."""
        import itertools
        ini = self._init
        keys = ("xpos", "ypos", "inc", "pa")
        ranges = [self._bounds(k) for k in keys]
        rmax = self._bounds("htot")[1]
        thmax = min(180., self._bounds("openang")[1])
        out = []
        for combo in itertools.product(*[(lo, hi) if hi > lo else (lo,) for lo, hi in ranges]):
            kw = dict(zip(keys, combo))
            for pa in np.linspace(min(ranges[3]), max(ranges[3]), 5):
                kw["pa"] = pa
                out.append(WindPars(radii=ini["radii"], vwind=ini["vwind"], vdisp=ini["vdisp"], vrot=ini["vrot"],
                                    vsys=ini["vsys"], openang=thmax, openin=0., htot=rmax, **kw))
        return out

    def _setup_region(self):
        """The region of the likelihood (all its voxels): the pixels within grow beams of the projected cones of the
        widest prior, all the channels with data; the weights W = likefactor/sigma^2 (or with the model error of
        datafrac) and the constants of the likelihood."""
        from .likelihood import LikelihoodMixin
        lk = self.like_kwargs
        grow = int(math.ceil(float(lk.get("grow", 1.)) * self._beam / self._pix))
        self._region2d = self._wm.footprint(self._corner_pars(), grow=grow)
        if not self._region2d.any():
            raise BBaroloError("BAYESIANWIND: the cones of the priors are outside the cube (check XPOS, YPOS, HTOT)")
        ys, xs = np.nonzero(self._region2d)
        nz = self.data.shape[0]
        self._lim = np.array([xs.min(), xs.max() + 1, ys.min(), ys.max() + 1, 0, nz], dtype=np.int64)
        self._ok = self._region2d[None, :, :] & np.isfinite(self.data)
        # the correlation factor of the noise, as BAYESIAN3DFIT (LikelihoodMixin._set_likefactor needs like_kwargs)
        holder = _LikeFactor(self, lk.get("noisecorr", "measured"))
        LikelihoodMixin._set_likefactor(holder)
        self.likefactor, self._likefactor_note = holder.likefactor, holder._likefactor_note
        self.noise_correlation = holder.noise_correlation
        self._dsafe = np.where(self._ok, self.data, 0.)
        self._set_weights(self._sigma_fixed())

    def _sigma_fixed(self):
        s = self.like_kwargs.get("sigma")
        return float(s) if s else self.noiserms

    def _set_weights(self, sigma):
        """The weights of the voxels for the noise sigma: W, W*data, c0 = sum W data^2 and the normalisation of the
        Gaussian likelihood (times the correlation factor)."""
        f = float(self.like_kwargs.get("datafrac") or 0.)
        var = sigma * sigma + (f * np.maximum(self._dsafe, 0.)) ** 2
        W = np.where(self._ok, self.likefactor / var, 0.)
        self._W, self._Wd = W, W * self._dsafe
        self._c0 = float(np.sum(self._Wd * self._dsafe))
        self._lognorm = float(-0.5 * self.likefactor * np.sum(np.log(2 * np.pi * var[self._ok])))
        self._wsigma = sigma

    def _setup_disk(self):
        """The fixed disk of DISKRINGS: one model per ring (unit density, the geometry and kinematics of the file), as
        Boxes in the region, and the default prior of their densities."""
        self._disk = None
        fn = str(getattr(self._par.gw, "DISKRINGS", "") or "").strip()
        if not fn:
            return
        from ..rings import ring_file_columns, Rings
        from ..galmod import DiskModel
        cols, _ = ring_file_columns(fn)
        if cols is None:
            raise BBaroloError(f"BAYESIANWIND: DISKRINGS {fn} is not a rings file of BBarolo")
        tab = np.atleast_2d(np.loadtxt(os.path.expanduser(fn), comments="#"))
        get = lambda k, d=0.: tab[:, cols[k] - 1] if k in cols else np.full(tab.shape[0], d)
        rings = Rings().setRings(radii=get("RADII"), xpos=get("XPOS"), ypos=get("YPOS"), vsys=get("VSYS"),
                                 vrot=get("VROT"), vdisp=get("VDISP"), inc=get("INC"), phi=get("PHI"), z0=get("Z0"),
                                 dens=np.ones(tab.shape[0]), vrad=get("VRAD"))
        dm = DiskModel(self._cube_obj, galmodtype="ANALYTIC", densflux=True)
        cubes = []
        for i in range(rings.nr):
            r = rings.copy()
            r.dens[:] = 0.
            r.dens[i] = 1.
            cubes.append(np.asarray(dm.array(r), dtype=np.float64))
        B = Boxes.from_cubes(cubes, tiny=1e-12 * max(float(np.max(a)) for a in cubes))
        self._disk = B
        self._disk_rings = rings
        self._disk_lists = B.voxel_lists(self._W, self._lim, rel=float(self.like_kwargs.get("prune") or 0.))
        Gd, bd = B.normal(self._Wd, self._W, self._disk_lists)
        x = log_profile(Gd, bd, self._c0)[1]
        top = float(np.max(x)) if x.size and np.max(x) > 0 else 1.
        self._disk_mu = np.maximum(x, 0.1 * top)            # (the prior of the densities: around the initial fit)

    # ------------------------------------------------------------------ the likelihood
    def wind_pars(self, theta):
        """The WindPars and the hyperparameters (sigma, emscale, emslope) of the vector of parameters theta."""
        ini = self._init
        R = ini["radii"]
        kw = {k: ini[k] for k in WIND_SCALAR_PARS}
        sh = {k: np.array(ini[k], float) for k in WIND_SHELL_PARS}
        hyp = dict(sigma=None, emscale=None, emclump=None, emratio=None, emslope=None)
        for key, idx in self.freepar_idx.items():
            base, _, suf = key.partition("_")
            if suf.startswith("p"):
                continue
            v = theta[idx]
            if base in WIND_SHELL_PARS:
                sh[base] = np.full(R.size, v[0]) if idx.size == 1 else np.array(v, float)
            elif base in WIND_SCALAR_PARS:
                kw[base] = float(v[0])
            else:
                hyp[base] = float(v[0])
        for base in WIND_FUNC_PARS:
            if f"{base}_p1" in self.freepar_idx:
                f = self.funcs[base]
                npar = f.__code__.co_argcount - 1
                sh[base] = np.asarray(f(R, *[theta[self.freepar_idx[f"{base}_p{i + 1}"]][0] for i in range(npar)]),
                                      float) * np.ones(R.size)
        return WindPars(radii=R, **sh, **kw), hyp

    def _valid(self, p):
        return (0. <= p.inc <= 90. and 0. <= p.openin < p.openang <= 180. and p.htot > 0
                and np.all(p.vdisp >= 0) and np.all(np.isfinite(p.vwind)))

    def _amplitude_system(self, p, hyp, pts=None):
        """The normal equations of the emission (and of the fixed disk) for the wind p: (G, b, mu, B, pts)."""
        wm = self._wm
        basis = self.like_kwargs["emission"]
        pts = wm.points(p) if pts is None else pts
        weight = None
        slope = self._slope(hyp)
        if basis == "cone" and slope:
            weight = (np.maximum(pts["r"], 1e-3 * self._r0) / self._r0) ** (-slope)
        B = wm.boxes(p, lim=self._lim, pts=pts, weight=weight)
        lists = B.voxel_lists(self._W, self._lim, rel=float(self.like_kwargs.get("prune") or 0.))
        G, b = B.normal(self._Wd, self._W, lists)
        s = hyp.get("emscale") or self._emscale_fixed()
        if basis == "cone":
            rb = np.full(B.n, self._r0)
        else:
            rb = np.maximum(pts["rbar"], 1e-3 * self._r0)
        mu = s * (rb / self._r0) ** (-slope if basis != "cone" else 0.)
        ratio = hyp.get("emratio") or float(self.like_kwargs.get("emratio") or 1.)
        mu = np.where(wm.cells.cells[:, 0] < 0, ratio * mu, mu)            # (the approaching cone)
        mu = np.where(pts["vol"] > 0, mu, 0.)
        if self._disk is not None:
            Gx = B.cross(self._disk, self._W, lists, self._disk_lists)
            Gd, bd = self._disk_normal()
            G = np.block([[G, Gx], [Gx.T, Gd]])
            b = np.concatenate([b, bd])
            mu = np.concatenate([mu, self._disk_mu])
        return G, b, mu, B, pts

    def _disk_normal(self):
        key = self._wsigma
        if getattr(self, "_disk_cache", (None,))[0] != key:
            self._disk_cache = (key,) + self._disk.normal(self._Wd, self._W, self._disk_lists)
        return self._disk_cache[1], self._disk_cache[2]

    def _log_like_func(self, theta, **kwargs):
        try:
            return self.log_likelihood(theta)
        except BBaroloError:
            return -np.inf

    def log_likelihood(self, theta, details=False):
        """The log-likelihood of theta (a vector of the free parameters), the emission marginalised (marginal, the
        default) or maximised; details=True: also a dict with the emission (x), the wind (p), and the normal
        equations."""
        p, hyp = self.wind_pars(np.asarray(theta, float))
        if not self._valid(p):
            return (-np.inf, {}) if details else -np.inf
        if hyp["sigma"] is not None and hyp["sigma"] != self._wsigma:
            self._set_weights(hyp["sigma"])
        basis = self.like_kwargs["emission"]
        if basis in ("spaxel", "voxel"):
            out = self._loglike_smooth(p, hyp, basis)
        else:
            G, b, mu, B, pts = self._amplitude_system(p, hyp)
            if self.like_kwargs.get("marginal", True):
                if hyp["emscale"] is None and self.like_kwargs.get("emscale") == "fit":
                    r = self._best_scale(G, b, mu, hyp)
                else:
                    r = self._marginal(G, b, mu, hyp)
                logl, x = r["logz"], r["x"]
            else:
                logl, x = log_profile(G, b, self._c0)
            out = (logl, dict(x=x, p=p, B=B, pts=pts, G=G, b=b, mu=mu, hyp=hyp))
        logl = out[0] + self._lognorm
        return (logl, out[1]) if details else logl

    def _slope(self, hyp):
        """EMSLOPE: free (hyp), fixed (LIKEOPTS emslope) or 0."""
        s = hyp.get("emslope")
        return float(s) if s is not None else float(self.like_kwargs.get("emslope") or 0.)

    def _kappa(self, hyp, n):
        """The fractional scatter of the prior of each amplitude: EMCLUMP for the cells of the wind, 3 (a broad prior
        around the densities of the initial fit) for the rings of the fixed disk."""
        k = hyp.get("emclump") or float(self.like_kwargs.get("emclump") or 1.)
        kap = np.full(n, k)
        if self._disk is not None:
            kap[n - self._disk.n:] = 3.
        return kap

    def _marginal(self, G, b, mu, hyp):
        if self.like_kwargs.get("prior", "gauss") == "gauss":
            return log_marginal_gauss(G, b, self._c0, mu, self._kappa(hyp, mu.size))
        return log_marginal(G, b, self._c0, mu)

    def _emscale_fixed(self):
        """EMSCALE when it is not free: the option emscale (a number), or the mean emissivity of the data."""
        s = self.like_kwargs.get("emscale")
        return float(s) if s not in (None, "fit") else (self._s0 or 1.)

    def _best_scale(self, G, b, mu, hyp):
        """The marginal likelihood at the EMSCALE that maximises it (emscale='fit'; mu: the means at the fixed
        EMSCALE)."""
        from scipy.optimize import minimize_scalar
        s0 = 1.
        nw = mu.size - (self._disk.n if self._disk is not None else 0)
        scl = lambda ls: np.concatenate([mu[:nw] * math.exp(ls), mu[nw:]])
        f = lambda ls: -self._marginal(G, b, scl(ls), hyp)["logz"]
        grid = np.log(s0) + np.linspace(-7, 7, 8)
        vals = [f(x) for x in grid]
        i = int(np.argmin(vals))
        lo, hi = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
        res = minimize_scalar(f, bounds=(lo, hi), method="bounded", options=dict(xatol=0.05))
        best = res.x if res.fun < vals[i] else grid[i]
        r = self._marginal(G, b, scl(best), hyp)
        r["emscale"] = math.exp(best)
        return r

    def _loglike_smooth(self, p, hyp, basis):
        """'spaxel' (a uniform cone normalised in each spaxel, marginalised) and 'voxel' (MoKA-3D: the model is the
        data where the uniform cone reaches, zero elsewhere: the profile likelihood of the voxels it does not reach)."""
        wm = self._wm
        pts = wm.points(p)
        B = wm.boxes(p, lim=self._lim, pts=pts)
        M = B.model(np.ones(B.n), self.data.shape)
        if basis == "voxel":
            thr = 1e-3 * float(np.max(M)) if np.max(M) > 0 else 0.
            covered = M > thr
            logl = -0.5 * float(np.sum(np.where(covered, 0., self._Wd * self._dsafe)))
            return logl, dict(x=None, p=p, covered=covered)
        g = np.sum(self._W * M * M, axis=0).ravel()
        bb = np.sum(self._Wd * M, axis=0).ravel()
        s = hyp.get("emscale") or self._emscale_fixed()
        gauss = self.like_kwargs.get("prior", "gauss") == "gauss"
        k = hyp.get("emclump") or float(self.like_kwargs.get("emclump") or 1.)
        marg = (lambda sc: log_marginal_gauss_diag(g, bb, self._c0, np.full(g.size, sc), k)) if gauss else \
            (lambda sc: log_marginal_diag(g, bb, self._c0, np.full(g.size, sc)))
        if hyp.get("emscale") is None and self.like_kwargs.get("emscale") == "fit":
            from scipy.optimize import minimize_scalar
            res = minimize_scalar(lambda ls: -marg(math.exp(ls))[0], bounds=(-15, 15), method="bounded")
            s = math.exp(res.x)
        logz, x = marg(s)
        if not self.like_kwargs.get("marginal", True):
            ok = g > 0
            x = np.where(ok, np.maximum(bb / np.where(ok, g, 1.), 0.), 0.)
            logz = -0.5 * float(self._c0 - 2 * bb @ x + g @ (x * x))
        return logz, dict(x=x, p=p, smooth=M)

    # ------------------------------------------------------------------ sampling
    def run(self, freepar=None, method=None, threads=None, *, like_kwargs: dict = {}, sampler_kwargs: dict = {},
            run_kwargs: dict = {}, verbose=True, dynamic=False, **params):
        """Sample the posterior of the free parameters freepar (names as in FREE; default FREE) with method
        ('nautilus', default, or 'dynesty'; SAMPLER), in threads processes (default THREADS, or 1), with the options of
        the likelihood like_kwargs (LIKEOPTS), of the sampler (SAMPLEROPTS) and of its run (RUNOPTS). Nothing is
        written (see write). Returns self."""
        if params:
            self.update(**params)
        dflt = self._run_defaults()
        freepar = freepar if freepar is not None else dflt.get("FREE")
        if not freepar:
            raise ValueError("BAYESIANWIND: no free parameters (give freepar or FREE)")
        method = (method or dflt.get("SAMPLER") or "nautilus").lower()
        if threads is None:
            threads = int(self._params.get("THREADS", 1)) if "THREADS" in self._params else 1
        lk = dict(WIND_LIKE_OPTIONS)
        lk.update(dflt.get("LIKEOPTS", {}))
        for k, v in dict(like_kwargs or {}).items():
            kv = wind_like_option(k, v)
            if kv is None:
                raise ValueError(f"BAYESIANWIND: unknown option of the likelihood '{k}' "
                                 f"(options: {', '.join(WIND_LIKE_OPTIONS)})")
            lk[kv[0]] = kv[1]
        self.like_kwargs = lk
        skw = {**dflt.get("SAMPLEROPTS", {}), **dict(sampler_kwargs or {})}
        rkw = {**dflt.get("RUNOPTS", {}), **dict(run_kwargs or {})}
        dynamic = bool(skw.pop("dynamic", dynamic))
        if method == "dynesty" and "seed" in skw:          # (nautilus: its own option seed)
            seed = skw.pop("seed")
            skw.setdefault("rstate", np.random.default_rng(None if seed is None else int(seed)))
        skw, rkw = sampler_options(method, dynamic, skw, rkw)
        freepar = self._with_hyper(freepar, verbose)
        self._setup(freepar, verbose=verbose)
        theta0 = self._prior_transform(np.full(self.ndim, 0.5))
        l0 = self.log_likelihood(theta0)
        if not np.isfinite(l0):
            raise BBaroloError("BAYESIANWIND: the likelihood at the centre of the priors is not finite (check the "
                               "initial wind and the priors)")
        self._method = method
        vprint(verbose, f" BAYESIANWIND: sampling {self.ndim} parameters with {method} ({threads} processes)...")
        samples, weights, dt = sample(self, {}, method, int(threads), dynamic, skw, rkw, verbose)
        self._set_sampling_stats(samples, weights)
        self.time = dt
        self._best_fit()
        vprint(verbose, f" BAYESIANWIND: done in {dt / 60.:.1f} minutes.")
        return self

    def _with_hyper(self, freepar, verbose=True):
        """The free parameters with the hyperparameters of the prior of the emission (EMSCALE, EMCLUMP) that are not
        fixed (options emscale, emclump): they are always fitted (the emission marginalised with them fixed at a rough
        value biases the geometry)."""
        names = wind_freepar(" ".join(freepar) if not isinstance(freepar, str) else freepar,
                             funcs=set(WIND_FUNC_PARS) | set(self.funcs))
        lk = self.like_kwargs
        if lk["emission"] == "voxel" or not lk.get("marginal", True):
            return names
        added = []
        for h in HYPER[lk.get("prior", "gauss")]:
            if h in ("emratio", "emslope") and lk["emission"] == "spaxel":
                continue                                # (the uniform cone of each spaxel: no cones, no radial law)
            if h not in names and lk.get(h) is None:
                names.append(h)
                added.append(h.upper())
        if added:
            vprint(verbose, f" BAYESIANWIND: {', '.join(added)} (the prior of the emission) added to the free "
                            "parameters (fix them with LIKEOPTS emscale=, emclump=, emratio=, emslope=)")
        return names

    def _set_sampling_stats(self, samples, weights):
        w = np.asarray(weights, float)
        w = w / np.sum(w)
        self.samples, self.weights = np.asarray(samples), w
        self.ess = float(1. / np.sum(w * w))
        rng = np.random.default_rng(12345)
        n = max(2000, int(self.ess))
        self.samples_equal = self.samples[rng.choice(len(w), size=n, p=w)]
        self.quantiles = np.array([[_wquantile(self.samples[:, i], w, q) for q in SIGMA_QUANTILES]
                                   for i in range(self.ndim)])
        self.params = self.quantiles[:, 1].copy()
        self.modCalculated = True

    def _best_fit(self):
        """The best wind (the medians), its emission (maximum a posteriori) and the model cube."""
        logl, det = self.log_likelihood(self.params, details=True)
        self.best_logl = logl
        self.best = det.get("p")
        self.best_details = det
        self.emission = det.get("x")
        self.model = self._model_cube(det)

    def _model_cube(self, det):
        """The model cube (nz, ny, nx) of the details of log_likelihood."""
        shape = self.data.shape
        basis = self.like_kwargs["emission"]
        if basis == "voxel":
            return np.where(det["covered"], self._dsafe, 0.)
        if basis == "spaxel":
            ny, nx = shape[1:]
            return det["smooth"] * det["x"].reshape(ny, nx)[None]
        p, x, hyp = det["p"], det["x"], det["hyp"]
        pts = det["pts"]
        weight = None
        slope = self._slope(hyp)
        if basis == "cone" and slope:
            weight = (np.maximum(pts["r"], 1e-3 * self._r0) / self._r0) ** (-slope)
        B = self._wm.boxes(p, pts=pts, weight=weight)
        nw = B.n
        m = B.model(x[:nw], shape)
        if self._disk is not None:
            m += self._disk.model(x[nw:], shape)
        return m

    # ------------------------------------------------------------------ results
    def print_stats(self, stream=None):
        """The medians and the 1-sigma intervals of the free parameters."""
        import sys
        st = stream or sys.stdout
        if not self.modCalculated:
            st.write(" BAYESIANWIND: run() first.\n")
            return
        st.write(f"\n BAYESIANWIND posterior ({self._method}, emission {self.like_kwargs['emission']}, "
                 f"{'marginal' if self.like_kwargs.get('marginal', True) else 'profile'} likelihood):\n")
        for name, (lo, med, hi) in zip(self.freepar_names, self.quantiles):
            st.write(f"   {name:12s} = {med:12.4g}  -{med - lo:<10.3g} +{hi - med:<10.3g}\n")
        st.write("\n")

    def corner_plot(self, labels=None, truths=None, quantiles=SIGMA_QUANTILES, **kwargs):
        """Corner plot of the posterior (needs the package corner)."""
        import corner
        defaults = dict(max_n_ticks=5, color="darkblue", show_titles=True, plot_datapoints=False)
        if truths is not None:
            kwargs["truths"] = truths
        return corner.corner(self.samples_equal, labels=labels or self.freepar_names, quantiles=quantiles,
                             title_quantiles=quantiles, **{**defaults, **kwargs})

    def emission_errors(self, nsamp=50, seed=0):
        """The emission of each shell and cone at the best fit and its 1-sigma interval over nsamp posterior samples:
        a dict with radii (bin centres), and for each cone (1: +z, receding; -1: approaching) the flux (emissivity
        times volume: the units of the cube times pixels and channels) median, low and high."""
        if self.like_kwargs["emission"] in ("spaxel", "voxel"):
            return None
        rng = np.random.default_rng(seed)
        cells = self._wm.cells
        nr = cells.nr
        out = []
        for th in self.samples_equal[rng.choice(len(self.samples_equal), size=min(nsamp, len(self.samples_equal)),
                                                replace=False)]:
            logl, det = self.log_likelihood(th, details=True)
            if not np.isfinite(logl):
                continue
            out.append(self._shell_flux(det))
        out = np.array(out)
        best = self._shell_flux(self.best_details)
        redges = cells.redges
        res = dict(radii=0.5 * (redges[1:] + redges[:-1]), edges=redges)
        for ic, c in enumerate(cells.cones):
            q = np.percentile(out[:, ic], [100 * SIGMA_QUANTILES[0], 100 * SIGMA_QUANTILES[2]], axis=0) if len(out) \
                else np.full((2, nr), np.nan)
            res[c] = dict(best=best[ic], lo=q[0], hi=q[1])
        return res

    def _shell_flux(self, det):
        cells = self._wm.cells
        nw = cells.ncell
        flux = det["x"][:nw] * det["pts"]["vol"]
        out = np.zeros((len(cells.cones), cells.nr))
        np.add.at(out, (cells.ijk[:, 0], cells.ijk[:, 1]), flux)
        return out

    def shells_table(self):
        """The shells of the best fit: radius, VWIND, VDISP, VROT, each with its 1-sigma errors (lower, upper; 0 for
        fixed parameters): a dict of arrays."""
        R = self._init["radii"]
        tab = dict(radii=R)
        eq = self.samples_equal
        for base in WIND_SHELL_PARS:
            best = getattr(self.best, base)
            vals = None
            if base in self.freepar_idx:
                idx = self.freepar_idx[base]
                vals = eq[:, idx] if idx.size == R.size else np.repeat(eq[:, idx], R.size, axis=1)
            elif f"{base}_p1" in self.freepar_idx:
                f = self.funcs[base]
                npar = f.__code__.co_argcount - 1
                vals = np.array([f(R, *[th[self.freepar_idx[f"{base}_p{i + 1}"]][0] for i in range(npar)])
                                 * np.ones(R.size) for th in eq[:2000]])
            if vals is None:
                lo = hi = np.zeros(R.size)
            else:
                ql, qh = np.percentile(vals, [100 * SIGMA_QUANTILES[0], 100 * SIGMA_QUANTILES[2]], axis=0)
                lo, hi = np.maximum(best - ql, 0.), np.maximum(qh - best, 0.)
            tab[base], tab[base + "_lo"], tab[base + "_hi"] = best, lo, hi
        return tab

    def write(self, outfolder=None, plots=None, verbose=True):
        """Write the outputs of the fit (see bbarolo.bayesian.wind_outputs.write_outputs)."""
        from .wind_outputs import write_outputs
        return write_outputs(self, outfolder=outfolder, plots=plots, verbose=verbose)


class _LikeFactor:
    """The attributes that LikelihoodMixin._set_likefactor reads, for BayesianWind."""

    def __init__(self, f, mode):
        self.like_kwargs = dict(noisecorr=mode)
        self.beam_area = f.beam_area
        self.noise_correlation = f.noise_correlation
        self._cube_obj = f._cube_obj
        self.data = f.data
        self.mask = f.mask


def _wquantile(x, w, q):
    i = np.argsort(x)
    cw = np.cumsum(w[i])
    cw /= cw[-1]
    return float(np.interp(q, cw, x[i]))
