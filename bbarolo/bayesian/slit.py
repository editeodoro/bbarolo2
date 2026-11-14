"""
BayesianSlitFit (the task SLITFIT): fit the kinematics of a rotating disk to a long-slit spectrum (a 2D image,
position along the slit x wavelength), by sampling the posterior with nested sampling (the nautilus included in
BBarolo, NAUTILUSBB, the default; nautilus or dynesty) or by a minimisation of the same likelihood (METHOD GLOBAL).

The model is a GALMOD disk (rings of VROT, VDISP, VRAD; INC and Z0 fixed) built on a thin 3D grid around the slit,
with the spatial and spectral sampling of the slit, and seen through the slit: the seeing (BEAM, a circular Gaussian)
and the width of the slit (SLITWIDTH) are applied to it, and the instrumental resolution (SPECRES, the dispersion in
channels) to its lines (see bbarolo.bayesian.slitmodel). Several lines (RESTWAVE, with the relative intensities
RELINT) are modelled together at the same velocities. The geometry is that of the slit: SLITANGLE (the angle from
the slit, in the direction of increasing position, to the receding half of the major axis; AUTO: 0 or 180, from the
data), XPOS (the centre along the slit, pixels) and YPOS (the distance of the slit from the centre, arcsec, fixed);
the position angle on the sky plays no role. The data are prepared first (bbarolo.bayesian.slitdata): a spectral
window around the lines, an optional continuum (CONTSUB, a polynomial of degree CONTORDER at each position) and
residual sky (SKYRESID) subtracted, masked ranges (MASKWAVE), a noise map and the correlation of the noise.

    >>> from bbarolo import BayesianSlitFit
    >>> f = BayesianSlitFit("slit.fits", slitwidth=1, beam=1.5, specres=2.44, inc=71, redshift=0.09143,
    ...                     restwave=[6564.61, 6549.86, 6585.27], relint=[1, 0.34, 1], contsub=True)
    >>> f.run(threads=16)
    >>> f.print_stats(); f.write()

Free parameters (FREE): VROT, VDISP, VRAD (one value per ring, NAME_SINGLE one value for all, NAME_FUNC a function of
radius, see funcs), VSYS and XPOS (single values), DENS (NORM NONE only) and SIGMA (a factor of the noise map). The
default is VROT VDISP VSYS XPOS. The geometry (INC, Z0, SLITANGLE, YPOS) is fixed: a slit does not constrain it.

The normalisation of the model (NORM):
  AZIM   (default) the densities of the rings are those whose model reproduces the flux of the lines along the slit
         (the data summed over the channels of the lines), a non-negative least-squares fit with a smoothness prior
         (LIKEOPTS densprof_smooth), as 3DFIT with METHOD GLOBAL or BAYESIAN deconvolves the total intensity map;
  LOCAL  the AZIM model times one amplitude per position of the slit (an asymmetric or clumpy disk), marginalised
         in the likelihood (a flat prior, non-negative);
  NONE   the densities of the rings (DENS, given or free).

Options of the likelihood (LIKEOPTS): noisecorr ('measured', default: the correlation of the noise along the slit and
along the spectra, measured; or a number of pixels per independent measurement, 1 for independent pixels),
densprof_smooth (default 0.1), sigma (a fixed factor of the noise map, default 1), smooth (the smoothness priors of
the parameters free in each ring, as 3DFIT with METHOD GLOBAL or BAYESIAN: the bend allowed to each profile over one seeing FWHM, in the units
of the parameter, DENS in dex: smooth=(vrot=5, vdisp=3) in LIKEOPTS, a dict in Python).
"""

import os
import math
import copy
import numpy as np

from ..tasks import Task, build_param, _checked
from ..bbutils import BBaroloError
from .common import (vprint, banner, get_distribution, fast_ppf, show_bar, smoothness_penalty, smoothness_summary,
                     smoothness_text, smoothness_result, SIGMA_QUANTILES,
                     fit_method, global_arguments, check_sampler, _resets)
from .sampling import sample, sampler_options

SLIT_RING_PARS = ("vrot", "vdisp", "vrad", "dens")
SLIT_SCALAR_PARS = ("vsys", "xpos")
SLIT_OTHER_PARS = ("sigma",)
SLIT_FUNC_PARS = ("vrot", "vdisp", "vrad", "dens")
_SLIT_ALIASES = {"disp": "vdisp"}
_FIXED_GEOMETRY = ("inc", "phi", "pa", "z0", "ypos")

# the default functions of radius (arcsec) of VROT_FUNC, VDISP_FUNC, VRAD_FUNC and DENS_FUNC (BayesianSlitFit.funcs)
DEFAULT_SLIT_FUNCS = dict(vrot=lambda R, p1, p2: 2. / np.pi * p1 * np.arctan(R / p2),     # rising to p1
                          vdisp=lambda R, p1, p2: p1 + p2 * R,                           # linear
                          vrad=lambda R, p1, p2: p1 + p2 * R,                            # linear
                          dens=lambda R, p1, p2: p1 * np.exp(-R / p2))                   # exponential

SLIT_LIKE_OPTIONS = dict(noisecorr="measured", densprof_smooth=0.1, sigma=None, smooth=None)


def slit_like_option(name, value):
    """An option of the likelihood of SLITFIT, checked: (name, value); None if name is not an option. Raises
    ValueError for a wrong value (None gives the default)."""
    k = str(name).strip().lower()
    if k not in SLIT_LIKE_OPTIONS:
        return None
    if value is None:
        return k, SLIT_LIKE_OPTIONS[k]
    if k == "noisecorr":
        v = str(value).strip().lower()
        if v == "measured":
            return k, v
        try:
            x = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"noisecorr: 'measured' or a number of pixels per independent value, not {value!r}")
        if not x >= 1:
            raise ValueError(f"noisecorr: the number of pixels per independent value must be >= 1, not {value}")
        return k, x
    if k == "smooth":
        from ..params import bayes_like_option
        sm = bayes_like_option(k, value)[1]
        bad = [n for n in sm if n not in SLIT_RING_PARS]
        if bad:
            raise ValueError(f"smooth: '{bad[0]}' is not a ring parameter of SLITFIT ("
                             + ", ".join(SLIT_RING_PARS) + ")")
        return k, sm
    try:
        x = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{k}: a number, not {value!r}") from None
    if k == "densprof_smooth" and not x >= 0:
        raise ValueError(f"densprof_smooth must be >= 0, not {value}")
    if k == "sigma" and not x > 0:
        raise ValueError(f"sigma (a factor of the noise map) must be > 0, not {value}")
    return k, x


def slit_likeopts(opts):
    """The options of the likelihood of SLITFIT from LIKEOPTS (as in a parameter file). Raises ValueError."""
    from ..params import bayes_options
    out = {}
    for k, v in bayes_options(opts, none=False).items():
        kv = slit_like_option(k, v)
        if kv is None:
            raise ValueError(f"unknown option '{k}' (options: {', '.join(SLIT_LIKE_OPTIONS)})")
        out[kv[0]] = kv[1]
    return out


def slit_freepar(free, funcs=SLIT_FUNC_PARS):
    """The free parameters of SLITFIT from FREE: 'VROT VDISP_SINGLE VSYS XPOS' -> ['vrot', 'vdisp_single',
    'vsys', 'xpos'] (VSYS, XPOS: single values, also as VSYS_SINGLE, XPOS_SINGLE). Raises ValueError."""
    from ..params import bayes_freepar, bayes_name
    toks = str(free).replace(",", " ").split()
    for t in toks:
        if bayes_name(t, _SLIT_ALIASES).partition("_")[0] in _FIXED_GEOMETRY:
            raise ValueError(f"{t}: the geometry (INC, PA, Z0, YPOS, SLITANGLE) is fixed in SLITFIT (a slit "
                             "does not constrain it)")
    out = []
    for n in bayes_freepar(free, funcs=funcs, ring_pars=SLIT_RING_PARS + SLIT_SCALAR_PARS,
                           other_pars=SLIT_OTHER_PARS, aliases=_SLIT_ALIASES):
        base, _, suf = n.partition("_")
        if base in SLIT_SCALAR_PARS:
            if suf == "func":
                raise ValueError(f"{base.upper()} is a single value in SLITFIT (no _FUNC)")
            n = base
        out.append(n)
    return out


def slit_priors(priors):
    """The priors of SLITFIT from PRIORS (as 3DFIT with METHOD GLOBAL or BAYESIAN, with its free parameters). Raises ValueError."""
    from ..params import bayes_priors
    return bayes_priors(priors, ring_pars=SLIT_RING_PARS + SLIT_SCALAR_PARS, other_pars=SLIT_OTHER_PARS,
                        func_pars=SLIT_FUNC_PARS, aliases=_SLIT_ALIASES)


def _wquantile(x, w, q):
    i = np.argsort(x)
    cw = np.cumsum(w[i])
    cw /= cw[-1]
    return float(np.interp(q, cw, x[i]))


def _ring_values(s):
    """A ring parameter as given (parameter-file string): None (not given), a float or a list of floats."""
    from ..rings import ringList, ringTokens
    from ..bbutils import getDataColumn
    s = str(s).strip()
    if s in ("", "-1"):
        return None
    if "file(" in s.lower():
        return [float(x) for x in getDataColumn(s)]
    lst = ringList(s)
    if lst is not None:
        return lst
    try:
        return float(ringTokens(s)[0])
    except (ValueError, IndexError):
        raise BBaroloError(f"SLITFIT: cannot read the ring parameter {s!r}") from None


class BayesianSlitFit(Task):
    """Fit the kinematics of a disk to a long-slit spectrum in a Bayesian framework (see the module documentation).

    BayesianSlitFit(slit, **params): the slit (FITS file, HDU or (data, header)) and any BBarolo parameter: SLITWIDTH,
    BEAM (the seeing), INC, SPECRES (or LINEAR) are required; REDSHIFT, RESTWAVE, RELINT for the lines; the rings
    (RADII or NRADII and RADSEP, VROT, VDISP, VRAD, VSYS, XPOS, YPOS, Z0, DENS: estimated from the data if not given);
    SLITANGLE, NORM, CONTSUB, CONTORDER, EXCLWIND, WAVERANGE, SKYRESID, MASKWAVE; and the options FREE, PRIORS, SAMPLER,
    SAMPLEROPTS, RUNOPTS and LIKEOPTS (the defaults of run()).

    Methods: run(freepar, method, threads, like_kwargs, sampler_kwargs, run_kwargs, verbose, dynamic, **params) samples
    the posterior (nothing is written); optimize(...) finds the best fit by a minimisation of the same likelihood;
    write(outfolder, plots) writes the outputs; print_stats(), corner_plot(), ring_table(), plot_pv(),
    plot_parameters(); log_likelihood(theta) and the model of a vector of parameters (model_of).

    Attributes: priors, funcs, like_kwargs; prep (the prepared slit: data, noise, continuum, ...), initial (the initial
    rings); the results samples, weights, samples_equal, quantiles, params (the medians, or the best fit of
    optimize), optimum (the details of optimize), best (a dict: the rings, the model of the slit, the amplitudes).
    """

    TASK = "SLITFIT"
    DEFAULT_METHOD = "GLOBAL"           # the METHOD of run() when it is not given (as the task SLITFIT)
    _KEYWORD_OPTIONS = ("FREE", "SAMPLER", "PRIORS", "SAMPLEROPTS", "RUNOPTS", "LIKEOPTS")

    def __init__(self, slit=None, **params):
        self.taskname = "SLITFIT"
        self.freepar_idx = {}
        self.freepar_names = []
        self.ndim = 0
        self.priors = {}
        self.prior_distr = None
        self._default_pri = {}
        self.funcs = dict(DEFAULT_SLIT_FUNCS)
        self.like_kwargs = dict(SLIT_LIKE_OPTIONS)
        self.results = self.sampler = None
        self.optimum = None
        self.samples = self.weights = self.samples_equal = self.quantiles = self.params = None
        self.best = None
        self.modCalculated = False
        self._method = None
        self._reset()
        super(BayesianSlitFit, self).__init__(slit, **params)

    def _reset(self):
        self._par = None
        self.prep = None
        self.initial = None
        self._model = None
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
        super(BayesianSlitFit, self).update(**params)
        if not hasattr(self, "priors"):
            return self
        if "PRIORS" in given:
            self.priors.update(given["PRIORS"])
            if self.prior_distr is not None:
                for k in given["PRIORS"]:
                    self.prior_distr.pop(k, None)
        if _resets(params):
            self._reset()
        return self

    def _keyword_option(self, name, value):
        s = str(value)
        try:
            if name == "FREE":
                return slit_freepar(s, funcs=set(SLIT_FUNC_PARS) | set(getattr(self, "funcs", {})))
            if name == "LIKEOPTS":
                return slit_likeopts(s)
            if name == "PRIORS":
                return slit_priors(s)
            if name == "SAMPLER":
                return check_sampler(s, "SLITFIT")
            from ..params import bayes_options
            return bayes_options(s)
        except ValueError as e:
            raise ValueError(f"SLITFIT: {name}: {str(e).replace('SLITFIT: ', '')}") from None

    def _run_defaults(self):
        from ..keywords import TASK_DEFAULTS
        out = {k: self._keyword_option(k, self._params[k])
               for k in self._KEYWORD_OPTIONS if k != "PRIORS" and self._params.get(k) is not None}
        out.setdefault("FREE", self._keyword_option("FREE", TASK_DEFAULTS[self.TASK]["FREE"]))
        return out

    def _all_params(self, outfolder=None, fitsfile=None):
        from ..keywords import TASK_DEFAULTS
        prm = super(BayesianSlitFit, self)._all_params(outfolder=outfolder, fitsfile=fitsfile)
        for k, v in TASK_DEFAULTS[self.TASK].items():
            prm.setdefault(k, v)
        return prm

    def _outfolder(self):
        of = self._params.get("OUTFOLDER")
        of = os.path.expanduser(str(of).strip()) if of not in (None, "") else self._default_outfolder()
        return of if of.endswith("/") else of + "/"

    def _make_par(self):
        """The Param of the fit (checked as the task; verbose off, one thread: the samples are computed in parallel by
        processes)."""
        hdu = self._input_hdu()
        prm = self._all_params(outfolder=self._outfolder(), fitsfile=self._cube if hdu is None else "in-memory slit")
        prm["VERBOSE"] = False
        prm["THREADS"] = 1
        par = build_param(prm)
        par._given = {k: True for k in self._params}
        return _checked(par)

    # ------------------------------------------------------------------ the data and the initial rings
    def _load(self, verbose=False):
        """Read and prepare the slit, and estimate the initial rings (once, until the parameters change)."""
        if self.prep is not None:
            return self.prep
        from .slitdata import SlitData, prepare
        par = self._make_par()
        self._par = par
        hdu = self._input_hdu()
        slit = SlitData.read(os.path.abspath(self._cube) if hdu is None else hdu)
        gf = par.gf
        sp = slit.spec
        if sp.kind == "freq":
            rest = [x for x in (gf.RESTFREQ or []) if x > 0]
            if not rest and par.gf.RESTFREQ == [-1.0]:
                rest = []
        else:
            rest = [x for x in (gf.RESTWAVE or []) if x > 0]
        nl = max(len(rest), 1)
        relint = [x for x in (gf.RELINT or [])] if gf.RELINT and gf.RELINT[0] != -1 else [1.] * nl
        if len(relint) != nl:
            raise BBaroloError(f"SLITFIT: RELINT must have one value per line ({nl})")
        nch = par.specres if par.specres != -1 else par.linear
        if nch == -1:
            raise BBaroloError("SLITFIT: give the instrumental resolution of the spectra, SPECRES (the "
                               "dispersion of the line spread function, in channels)")
        self._nch = float(nch)
        self._beam = float(par.beam[0])
        self._width = float(par.slitwidth)
        veldef = par.ma.veldef
        # (the dispersion in km/s, for the estimates: the channel width at the first line)
        sp.rest = None
        from .slitdata import SpecAxis
        tmp = SpecAxis(sp.kind, sp.ctype, sp.cunit, sp.crval, sp.cdelt, sp.crpix, sp.n)
        if veldef and veldef != "auto":
            tmp.veldef = veldef
        z = par.getRedshift()
        tmp.rest = [r * (1 + z) for r in rest] if sp.kind == "wave" else [r / (1 + z) for r in rest] or None
        sig_instr = self._nch * tmp.dv(0)
        self.prep = prepare(slit, rest, relint, z, veldef, sig_instr, self._beam, contsub=par.contsub,
                            contorder=max(int(par.cont_order), 0), exclwind=par.exclude_windows,
                            waverange=par.waverange, skyresid=par.skyresid, maskwave=par.maskwave, verbose=verbose)
        self.prep.sig_instr = sig_instr
        self.noiserms = float(np.nanmedian(self.prep.noise[self.prep.valid]))
        self._init_rings()
        self._set_default_priors()
        return self.prep

    def _init_rings(self):
        """The initial rings: the values given, or estimated from the fits of the lines along the slit (see
        slitdata.fit_lines): XPOS the peak of the continuum (CONTSUB) or the kinematic centre, VSYS the mean of the
        velocities at the same distance on the two sides, SLITANGLE (AUTO) the side of the receding velocities, VROT
        and VDISP the deprojected velocities and the dispersions (minus the instrumental one) averaged in the rings;
        RADII every seeing FWHM out to the farthest emission (RADSEP and NRADII if given). Sets self.initial (a dict)
        and the rings _fit0 (a _RingDict)."""
        from .rings import _RingDict
        from ..rings import Rings, ringTokens
        par, prep = self._par, self.prep
        gf = par.gf
        L = prep.lines
        ok = L["ok"]
        pos, v, sg = L["pos"], L["v"], L["sigma"]
        pix = prep.slit.pix
        inc = _ring_values(gf.INC)
        inc = float(np.median(inc)) if isinstance(inc, list) else float(inc)
        if not 0 < inc < 90:
            raise BBaroloError(f"SLITFIT: INC must be between 0 and 90 degrees (not included), not {inc:g}")
        sini = math.sin(math.radians(inc))
        est = {}
        # the centre along the slit
        xpos = _ring_values(gf.XPOS)
        if xpos is None:
            xpos = self._estimate_xpos()
            est["XPOS"] = xpos
        else:
            xpos = float(np.median(xpos)) if isinstance(xpos, list) else float(xpos)
        # the systemic velocity
        vsys = _ring_values(gf.VSYS)
        if vsys is None:
            vsys = _folded_vsys(pos[ok], v[ok], xpos)
            est["VSYS"] = vsys
        else:
            vsys = float(np.median(vsys)) if isinstance(vsys, list) else float(vsys)
        # the direction of the receding side
        sa = str(par.slitangle).strip().upper()
        if sa == "AUTO":
            sgn = np.sum((pos[ok] - xpos) * (v[ok] - vsys))
            slitangle = 0. if sgn >= 0 else 180.
            est["SLITANGLE"] = slitangle
        else:
            slitangle = float(sa)
        d = math.radians(slitangle)
        ci = math.cos(math.radians(inc))
        q = math.sqrt(math.cos(d) ** 2 + math.sin(d) ** 2 / ci ** 2)
        # the radius of a position along the slit and the cosine of its azimuth in the disk
        s = (pos - xpos) * pix
        R = np.abs(s) * q
        cphi = np.sign(s) * math.cos(d) / q
        # the rings
        radii = _ring_values(gf.RADII)
        if isinstance(radii, float) or (radii is None and str(gf.RADII).strip() not in ("", "-1")):
            tok = ringTokens(gf.RADII)
            if len(tok) == 1 and "~" in tok[0] and ":" in tok[0]:
                a, rest_ = tok[0].split("~")
                b, st = rest_.split(":")
                a, b, st = float(a), float(b), float(st)
                radii = list(a + np.arange(int(np.floor((b - a) / st + 1e-6)) + 1) * st)
            else:
                radii = [float(radii)]
        if radii is None:
            sep = gf.RADSEP if gf.RADSEP > 0 else self._beam
            if gf.NRADII > 0:
                nr = int(gf.NRADII)
            else:
                rmax = float(np.max(R[ok])) if np.any(ok) else 5 * sep
                nr = max(2, int(round(rmax / sep)))
                est["NRADII"] = nr
                if gf.RADSEP <= 0:
                    est["RADSEP"] = sep
            radii = list((np.arange(nr) + 0.5) * sep)
        radii = np.asarray(radii, float)
        nr = radii.size
        edges = np.r_[0., 0.5 * (radii[1:] + radii[:-1]), radii[-1] + 0.5 * (radii[-1] - radii[-2] if nr > 1 else
                                                                                radii[0])]

        def per_ring(vals, use):
            out = np.full(nr, np.nan)
            for i in range(nr):
                m = use & (R >= edges[i]) & (R < edges[i + 1]) & np.isfinite(vals)
                if np.any(m):
                    out[i] = np.median(vals[m])
            good = np.isfinite(out)
            if not np.any(good):
                return None
            return np.interp(radii, radii[good], out[good])

        given = {}
        for key in ("vrot", "vdisp", "vrad", "z0", "dens"):
            given[key] = _ring_values(getattr(gf, key.upper()))
        if given["vrot"] is None:
            with np.errstate(all="ignore"):
                vr = (v - vsys) / (sini * cphi)
            use = ok & (np.abs(cphi) > 0.2)
            vrot = per_ring(np.abs(vr), use)
            vrot = np.maximum(vrot if vrot is not None else np.full(nr, 100.), 0.)
            est["VROT"] = float(np.median(vrot))
        else:
            vrot = given["vrot"]
        if given["vdisp"] is None:
            with np.errstate(all="ignore"):
                vd = np.sqrt(np.maximum(sg ** 2 - prep.sig_instr ** 2, 25.))
            vdisp = per_ring(vd, ok)
            vdisp = np.clip(vdisp if vdisp is not None else np.full(nr, 20.), 5., 150.)
            est["VDISP"] = float(np.median(vdisp))
        else:
            vdisp = given["vdisp"]
        z0 = given["z0"] if given["z0"] is not None else pix          # (a thin disk: one pixel, as the 3D fits)
        r = Rings().setRings(radii=radii, xpos=xpos, ypos=0., vsys=vsys, vrot=vrot, vdisp=vdisp,
                             vrad=given["vrad"] if given["vrad"] is not None else 0., z0=z0,
                             dens=given["dens"] if given["dens"] is not None else 1., inc=inc, phi=90. - slitangle,
                             radsep=float(radii[1] - radii[0]) if nr > 1 else float(radii[0]))
        self._fit0 = _RingDict.from_rings(r)
        self._work = self._fit0.copy()
        yp = str(gf.YPOS).strip()
        try:
            yoff = 0. if yp in ("", "-1") else float(yp)
        except ValueError:
            raise BBaroloError(f"SLITFIT: YPOS (the distance of the slit from the centre, arcsec) must be a "
                               f"number, not {yp}") from None
        self.initial = dict(radii=radii, xpos=xpos, vsys=vsys, vrot=np.asarray(vrot, float) * np.ones(nr),
                            vdisp=np.asarray(vdisp, float) * np.ones(nr), inc=inc, slitangle=slitangle, yoffset=yoff,
                            estimated=est)

    def _estimate_xpos(self):
        """The centre along the slit: the peak of the continuum (CONTSUB, if it stands out of the noise), otherwise the
        kinematic centre (where the velocities cross their midrange, near the flux-weighted centre of the lines)."""
        prep = self.prep
        L = prep.lines
        ok = L["ok"]
        if self._par.contsub:
            c = np.nanmedian(np.where(prep.contchan[:, None], prep.continuum, np.nan), axis=0)
            n = np.nanmedian(prep.noise) / math.sqrt(max(prep.contchan.sum(), 1))
            if np.any(np.isfinite(c)):
                k = int(np.nanargmax(c))
                if c[k] > 20 * n and 0 < k < c.size - 1 and np.all(np.isfinite(c[k - 1:k + 2])):
                    a, b, d = c[k - 1], c[k], c[k + 1]
                    den = a - 2 * b + d
                    return float(k + (0.5 * (a - d) / den if den < 0 else 0.))
        pos, v = L["pos"][ok], L["v"][ok]
        f = np.where(np.isfinite(L["flux"][ok]), L["flux"][ok], 0.)
        xc = float(np.sum(pos * f) / np.sum(f)) if np.sum(f) > 0 else float(np.median(pos))
        vs = np.convolve(v, np.ones(3) / 3, mode="same") if v.size > 3 else v
        mid = 0.5 * (np.percentile(vs, 10) + np.percentile(vs, 90))
        x = vs - mid
        cross = [pos[i] - x[i] * (pos[i + 1] - pos[i]) / (x[i + 1] - x[i]) for i in range(len(x) - 1)
                 if x[i] * x[i + 1] < 0 and pos[i + 1] - pos[i] < 3]
        if cross:
            return float(min(cross, key=lambda c_: abs(c_ - xc)))
        return xc

    def _set_default_priors(self):
        """The default priors, from the initial rings: VROT 0 to twice the largest (at least 100 km/s); VDISP 0 to three
        times the largest (at least 100); VRAD +- half the largest VROT (at least 50); VSYS +- 150 km/s (at least 5
        channels); XPOS +- two seeing FWHM; SIGMA log-uniform 0.3-30; DENS 0 to 3 times the largest initial density;
        the parameters of the functions of radius (VROT_FUNC: 0 to the VROT range and 2% to 100% of the last radius;
        the linear ones +- their ranges). A prior given (priors) is kept."""
        ini = self.initial
        U = lambda lo, hi: dict(name="uniform", loc=float(lo), scale=float(hi - lo))
        vmax = max(100., 2 * float(np.max(ini["vrot"])))
        dmax = max(100., 3 * float(np.max(ini["vdisp"])))
        rmax = max(50., 0.5 * float(np.max(ini["vrot"])))
        dv = max(150., 5 * self.prep.spec.dv(0))
        b = 2 * self._beam / self.prep.slit.pix
        R = float(ini["radii"][-1])
        pri = dict(vrot=U(0., vmax), vdisp=U(0., dmax), vrad=U(-rmax, rmax),
                   vsys=U(ini["vsys"] - dv, ini["vsys"] + dv), xpos=U(ini["xpos"] - b, ini["xpos"] + b),
                   sigma=dict(name="loguniform", a=0.3, b=30.),
                   vrot_p1=U(0., vmax), vrot_p2=U(0.02 * R, R),
                   vdisp_p1=U(0., dmax), vdisp_p2=U(-dmax / R, dmax / R),
                   vrad_p1=U(-rmax, rmax), vrad_p2=U(-rmax / R, rmax / R))
        for k, v in pri.items():
            if k not in self.priors or self.priors.get(k) == self._default_pri.get(k):
                self.priors[k] = v
                self._default_pri[k] = v

    # ------------------------------------------------------------------ free parameters and priors
    def _parse_freepar(self, freepar):
        """The free parameters (names as in FREE): ndim, freepar_idx, freepar_names."""
        self.freepar_idx, self.freepar_names, self.ndim = {}, [], 0
        nr = self._fit0.nr
        names = slit_freepar(" ".join(freepar) if not isinstance(freepar, str) else freepar,
                             funcs=set(SLIT_FUNC_PARS) | set(self.funcs))
        norm = self._norm()
        for name in names:
            base, _, suffix = name.partition("_")
            if base == "dens" and norm != "NONE":
                raise ValueError(f"SLITFIT: DENS is free only with NORM NONE (with NORM {norm} the densities "
                                 "come from the data)")
            if suffix == "func":
                if base not in self.funcs:
                    raise ValueError(f"SLITFIT: no function of radius for {base.upper()} (see funcs)")
                npar = self.funcs[base].__code__.co_argcount - 1
                for i in range(npar):
                    key = f"{base}_p{i + 1}"
                    self.freepar_idx[key] = np.array([self.ndim])
                    self.freepar_names.append(key)
                    self.ndim += 1
            elif base in SLIT_RING_PARS and suffix == "":
                self.freepar_idx[base] = np.arange(self.ndim, self.ndim + nr)
                self.freepar_names += [f"{base}{i + 1}" for i in range(nr)]
                self.ndim += nr
            else:
                self.freepar_idx[base] = np.array([self.ndim])
                self.freepar_names.append(base)
                self.ndim += 1

    def _freeze_priors(self):
        self.prior_distr = {} if self.prior_distr is None else self.prior_distr
        for key in self.freepar_idx:
            if key not in self.priors:
                raise ValueError(f"SLITFIT: no prior for {key} (give it in PRIORS)")
            d = self.priors[key]
            kw = {k: v for k, v in d.items() if k != "name"}
            try:
                self.prior_distr[key] = get_distribution(d["name"], **kw)
            except Exception:
                raise ValueError(f"SLITFIT: something is wrong in the prior of {key}: {d}")

    def _prior_transform(self, u):
        """ The unit cube into the parameter space with the priors (their ppf, as common.fast_ppf computes it). """
        p = np.zeros_like(u)
        for key, idx in self.freepar_idx.items():
            p[idx] = fast_ppf(self.prior_distr[key])(u[idx])
        return p

    def _bounds(self, key):
        """The range of a parameter: the support of its prior (99.9% for unbounded priors) if free, else its value."""
        if key in self.freepar_idx:
            d = self.prior_distr[key]
            lo, hi = d.support()
            lo = d.ppf(1e-3) if not np.isfinite(lo) else lo
            hi = d.ppf(1 - 1e-3) if not np.isfinite(hi) else hi
            return float(lo), float(hi)
        v = self.initial[key]
        return float(v), float(v)

    def _norm(self):
        return str(self._par.gf.NORM).strip().upper() if self._par is not None else "AZIM"

    # ------------------------------------------------------------------ setup
    def _setup(self, freepar, verbose=True):
        """The setup of a fit: the data and the initial rings, the free parameters and their priors, the region of the
        slit of the likelihood (the positions within the outer edge of the rings, plus two seeing FWHM, of any XPOS of
        the prior), the model engine and the weights of the likelihood."""
        from ..galmod import outer_edge
        from .slitmodel import SlitModel
        self._load(verbose)
        self._parse_freepar(freepar)
        self._freeze_priors()
        prep = self.prep
        pix = prep.slit.pix
        xlo, xhi = self._bounds("xpos")
        ext = (outer_edge(self._fit0.object, self._par.ringmode) + 2 * self._beam) / pix
        p0 = max(int(math.floor(xlo - ext)), 0)
        p1 = min(int(math.ceil(xhi + ext)) + 1, prep.slit.npos)
        key = (p0, p1, tuple(self._fit0.r["radii"]))
        if self._setup_key != key:
            self._model = SlitModel(prep, self._par, (p0, p1), self._width, self._beam,
                                    yoffset=self.initial["yoffset"], ringmode=self._par.ringmode)
            self._setup_key = key
        self.region = (p0, p1)
        sl = slice(p0, p1)
        self._d = np.where(prep.valid[:, sl], prep.data[:, sl], 0.)
        self._valid = prep.valid[:, sl]
        sig = np.where(self._valid, prep.noise[:, sl], 1.)
        nc = self.like_kwargs.get("noisecorr", "measured")
        self.likefactor = float(prep.corr["likefactor"]) if nc == "measured" else 1. / float(nc)
        self._W = np.where(self._valid, self.likefactor / sig ** 2, 0.)
        self._lognorm = float(-0.5 * self.likefactor * np.sum(np.log(2 * np.pi * sig[self._valid] ** 2)))
        self.nvalid = int(self._valid.sum())
        # the flux of the lines along the slit (AZIM): the data in the channels of the lines, the blanks filled along
        # the spectra
        lc = prep.linechan
        dl = prep.data[lc][:, sl]
        nl = prep.noise[lc][:, sl]
        filled = dl.copy()
        good = np.isfinite(dl)
        k = np.arange(dl.shape[0])
        for j in range(dl.shape[1]):
            g = good[:, j]
            if g.sum() >= 2 and not g.all():
                filled[~g, j] = np.interp(k[~g], k[g], dl[g, j])
        self._prof_ok = good.mean(axis=0) >= 0.5
        self._prof = np.where(self._prof_ok, np.nansum(np.where(np.isfinite(filled), filled, 0.), axis=0), 0.)
        self._prof_rms = np.sqrt(np.nansum(np.where(np.isfinite(nl), nl, 0.) ** 2, axis=0) *
                                 prep.corr["spectral"])
        self._prof_ok &= self._prof_rms > 0
        self._ringmap_key = None
        self._free_bases = {k.split("_")[0] for k in self.freepar_idx}

    # ------------------------------------------------------------------ the likelihood
    def _update_rings(self, rd, theta):
        """The rings rd (a _RingDict) with the parameters theta."""
        fixed = {}
        for p in {k.split("_")[0] for k in self.freepar_idx if "_p" in k}:
            keys = [k for k in self.freepar_idx if k.startswith(p + "_p")]
            vals = [float(theta[self.freepar_idx[k]][0]) for k in keys]
            rd.modify_parameter(p, self.funcs[p](np.asarray(rd.r["radii"], float), *vals))
            fixed[p] = True
        for key, idx in self.freepar_idx.items():
            if "_p" in key or key == "sigma":
                continue
            val = theta[idx]
            rd.modify_parameter(key, val[0] if val.size == 1 else val)
        rd.make_object()
        return rd

    def _profile_basis(self, rt):
        """The flux along the slit of each ring with unit density and of an exterior annulus (two seeing FWHM beyond the
        outer edge), (nr + 1, npos), and the radii of their centres (the maps of the rings cached by the whole pixel
        of the centre and the radii; the fraction of a pixel, rt.xshift, moves their projection)."""
        from ..galmod import outer_edge
        from ..rings import Rings
        m = self._model
        key = (float(rt.xpos[0]), tuple(rt.radii))
        if self._ringmap_key == key:
            K, Rc = self._ringmap
            return m.project_maps(K, getattr(rt, "xshift", 0.)), Rc
        K = m.disk.ring_maps(rt)
        edge = outer_edge(rt, m.ringmode)
        w = 2. * self._beam
        rext = edge + w / 2.
        ext = Rings().setRings(radii=[rext], xpos=rt.xpos[-1], ypos=rt.ypos[-1], vsys=rt.vsys[-1], vrot=rt.vrot[-1],
                               vdisp=rt.vdisp[-1], z0=rt.z0[-1], inc=rt.inc[-1], phi=rt.phi[-1], dens=1., radsep=w)
        K = np.concatenate([K, m.disk.ring_maps(ext, ringmode="CENTRED")])
        self._ringmap = (K, np.r_[np.asarray(rt.radii, float), rext])
        self._ringmap_key = key
        return m.project_maps(K, getattr(rt, "xshift", 0.)), self._ringmap[1]

    def _azim_density(self, rt):
        """The densities of the rings whose model reproduces the flux of the lines along the slit (NORM AZIM, LOCAL)."""
        from ..profile import solve_profile
        A, Rc = self._profile_basis(rt)
        ok = self._prof_ok
        if ok.sum() < 2 * rt.nr:
            raise BBaroloError("SLITFIT: too few positions with data to compute the density profile")
        s = self._prof_rms[ok]
        dens = solve_profile(A[:, ok].T / s[:, None], self._prof[ok] / s, Rc,
                             self.like_kwargs.get("densprof_smooth") or 0.)
        return dens[:rt.nr]

    def model_of(self, theta, details=False):
        """The model of the slit (nz, npos of the region) for the parameters theta (the AZIM/NONE model; LOCAL: before
        the amplitudes of the positions), and with details the rings (a _RingDict, XPOS in pixels of the slit) and
        their densities."""
        rd = self._update_rings(self._work, np.asarray(theta, float))
        for k in ("vdisp", "dens"):
            if np.any(np.asarray(rd.r[k]) < 0):
                raise BBaroloError(f"SLITFIT: negative {k.upper()}")
        rt = self._model.rings(rd)
        if self._norm() != "NONE":
            rt.dens = self._azim_density(rt)
        M = self._model.model(rt)
        if details:
            return M, rd, np.asarray(rt.dens, float)
        return M

    def _scale(self, theta):
        if "sigma" in self.freepar_idx:
            return float(theta[self.freepar_idx["sigma"]][0])
        return float(self.like_kwargs.get("sigma") or 1.)

    def log_likelihood(self, theta, details=False):
        """The log-likelihood of the parameters theta (see the module documentation), minus the smoothness penalties
        (LIKEOPTS smooth); with details also a dict: the model (nz, npos of the region), the rings, the densities and,
        with NORM LOCAL, the amplitudes of the positions."""
        from scipy.special import log_ndtr
        theta = np.asarray(theta, float)
        M, rd, dens = self.model_of(theta, details=True)
        k = self._scale(theta)
        W = self._W / (k * k)
        d = self._d
        lognorm = self._lognorm - self.likefactor * self.nvalid * math.log(k)
        amp = None
        if self._norm() == "LOCAL":
            tot = M.sum(axis=0)
            use = tot > 1e-12 * max(float(np.max(tot)), 1e-300)
            U = np.where(use[None, :], M / np.where(use, tot, 1.)[None, :], 0.)
            g = np.sum(W * U * U, axis=0)
            b = np.sum(W * d * U, axis=0)
            c = np.sum(W * d * d, axis=0)
            gp = g > 0
            gs = np.where(gp, g, 1.)
            z = b / np.sqrt(gs)
            logz = np.where(gp, -0.5 * c + 0.5 * z * z + 0.5 * np.log(2 * np.pi / gs) + log_ndtr(z), -0.5 * c)
            logl = float(np.sum(logz)) + lognorm
            if details:
                amp = np.where(gp, np.maximum(b / gs, 0.), 0.)
                M = U * amp[None, :]
        else:
            logl = float(-0.5 * np.sum(W * (d - M) ** 2)) + lognorm
        for key, sk in self._smooth_used().items():
            logl -= smoothness_penalty(rd.r["radii"], rd.r[key], sk, self._beam, from_zero=(key == "vrot"),
                                       log=(key == "dens"))
        if not details:
            return logl
        return logl, dict(model=M, rings=rd.copy(), dens=dens, amplitudes=amp, scale=k)

    def _smooth_used(self):
        """The smoothness tolerances used (LIKEOPTS smooth, {parameter: s}): those of the parameters free in each
        ring."""
        sm = self.like_kwargs.get("smooth") or {}
        return {k: float(v) for k, v in sm.items() if k in self.freepar_idx and self.freepar_idx[k].size > 1}

    def smoothness_stats(self):
        """What the smoothness priors (LIKEOPTS smooth) do to the best fit: {parameter: dict(n, bend, penalty, ndof)}
        (see common.smoothness_summary: the rms bend over a seeing FWHM of the best-fit profile, the penalty in ln L and
        the effective number of independent values of the profile, from the posterior covariance of its rings, that of
        the Laplace approximation after optimize); {} without smoothness priors."""
        sm = self._smooth_used()
        if not sm or not self.modCalculated or self.best is None:
            return {}
        rd = self.best["rings"]
        opt = self.optimum if self._method == "optimize" else None
        out = {}
        for k, sk in sm.items():
            idx = np.asarray(self.freepar_idx[k])
            if opt is None:
                cov = np.cov(np.asarray(self.samples)[:, idx].T, aweights=self.weights)
            else:
                cov = None if opt.cov is None else np.asarray(opt.cov)[np.ix_(idx, idx)]
            out[k] = smoothness_summary(rd.r["radii"], rd.r[k], sk, self._beam, from_zero=(k == "vrot"),
                                        log=(k == "dens"), cov=cov)
        return out

    def _log_like_func(self, theta, **kwargs):
        try:
            v = self.log_likelihood(theta)
        except (BBaroloError, ValueError):
            return -np.inf
        return v if np.isfinite(v) else -np.inf

    # ------------------------------------------------------------------ fitting
    def run(self, freepar=None, threads=None, *, like_kwargs: dict = {}, sampler_kwargs: dict = {},
            run_kwargs: dict = {}, verbose=True, dynamic=False, **params):
        """Fit with METHOD (a parameter, method=...; default GLOBAL, as the task SLITFIT): GLOBAL finds the maximum of the
        likelihood (optimize(), with MINIMISER, the other arguments of optimize in SAMPLEROPTS and sampler_kwargs, the
        options of the minimiser in RUNOPTS and run_kwargs), BAYESIAN samples the posterior (sample(), with SAMPLER).
        params: any BBarolo parameter (e.g. method, sampler, minimiser). Nothing is written (see write). Returns
        self."""
        if params:
            self.update(**params)
        if fit_method(self, self.DEFAULT_METHOD) == "GLOBAL":
            okw = global_arguments(self, self._run_defaults(), sampler_kwargs, run_kwargs, "SLITFIT")
            return self.optimize(freepar, threads=threads, like_kwargs=like_kwargs, verbose=verbose, **okw)
        return self.sample(freepar, threads=threads, like_kwargs=like_kwargs, sampler_kwargs=sampler_kwargs,
                           run_kwargs=run_kwargs, verbose=verbose, dynamic=dynamic)

    def sample(self, freepar=None, sampler=None, threads=None, *, like_kwargs: dict = {}, sampler_kwargs: dict = {},
               run_kwargs: dict = {}, verbose=True, dynamic=False, **params):
        """Sample the posterior of the free parameters freepar (names as in FREE; default FREE) with sampler
        ('nautilusbb', default: the nautilus included in BBarolo, 'nautilus' or 'dynesty'; SAMPLER), in threads
        processes (default THREADS, or 1), with the options of the likelihood like_kwargs (LIKEOPTS), of the sampler
        (SAMPLEROPTS) and of its run (RUNOPTS) (METHOD BAYESIAN). Nothing is written (see write). Returns self."""
        if params:
            self.update(**params)
        dflt = self._run_defaults()
        freepar = self._fit_options(freepar, like_kwargs, dflt)
        method = check_sampler(sampler, "SLITFIT") if sampler is not None else dflt.get("SAMPLER", "nautilusbb")
        threads = self._threads(threads)
        skw = {**dflt.get("SAMPLEROPTS", {}), **dict(sampler_kwargs or {})}
        rkw = {**dflt.get("RUNOPTS", {}), **dict(run_kwargs or {})}
        dynamic = bool(skw.pop("dynamic", dynamic))
        if method == "dynesty" and "seed" in skw:
            seed = skw.pop("seed")
            skw.setdefault("rstate", np.random.default_rng(None if seed is None else int(seed)))
        skw, rkw = sampler_options(method, dynamic, skw, rkw)
        self._prepare_fit(freepar, verbose)
        self._method = method
        self.optimum = None
        vprint(verbose, f" SLITFIT: sampling {self.ndim} parameters with {method} ({threads} processes)...")
        samples, weights, dt = sample(self, {}, method, int(threads), dynamic, skw, rkw, verbose)
        self._set_sampling_stats(samples, weights)
        self.time = dt
        self._best_fit()
        vprint(verbose, f" SLITFIT: done in {dt / 60.:.1f} minutes.")
        vprint(verbose, banner())
        return self

    def optimize(self, freepar=None, minimiser="L-BFGS-B", threads=None, *, like_kwargs: dict = {}, x0=None,
                 posterior=False, errors=True, nstart=1, seed=None, step=1e-3, options: dict = {}, nsamples=4000,
                 verbose=True, **params):
        """Find the best fit by a classic minimisation of -log L (METHOD GLOBAL), with the likelihood of sample() (the same free
        parameters, options and priors), searched in the unit cube of the priors (see bbarolo.bayesian.optimize and
        Bayesian3DFIT.optimize: method, x0 None (the initial rings) or 'centre' or a vector, posterior, errors from the
        Laplace approximation, nstart, seed, step, options, nsamples). Nothing is written. Returns self."""
        from .optimize import optimize as _optimize, summary, method_name
        method = method_name(minimiser)
        if params:
            self.update(**params)
        dflt = self._run_defaults()
        freepar = self._fit_options(freepar, like_kwargs, dflt)
        threads = self._threads(threads)
        self._prepare_fit(freepar, verbose)
        self.results = self.sampler = None
        if isinstance(x0, str):
            if x0.strip().lower() not in ("centre", "center"):
                raise BBaroloError(f"SLITFIT: optimize: x0={x0!r} (None, 'centre' or a vector of the free "
                                   "parameters)")
            theta0 = np.full(self.ndim, np.nan)
        else:
            theta0 = self._initial_theta() if x0 is None else np.asarray(x0, float)
        vprint(verbose, f" SLITFIT: optimising {self.ndim} parameters with {method} ({threads} "
                        f"process{'es' if threads > 1 else ''})...")
        res = _optimize(self, {}, theta0, minimiser=method, threads=int(threads), posterior=posterior, errors=errors,
                        nstart=nstart, seed=seed, step=step, options=options, nsamples=nsamples, verbose=verbose,
                        showbar=show_bar(self))
        summary(res, verbose, taskname=" SLITFIT: ")
        self.optimum = res
        self._method = "optimize"
        self._set_sampling_stats(res.samples, np.ones(len(res.samples)))
        self.params = np.asarray(res.x, float).copy()
        self.time = res.time
        self._best_fit()
        vprint(verbose, banner())
        return self

    def _fit_options(self, freepar, like_kwargs, dflt):
        freepar = freepar if freepar is not None else dflt.get("FREE")
        if not freepar:
            raise ValueError("SLITFIT: no free parameters (give freepar or FREE)")
        lk = dict(SLIT_LIKE_OPTIONS)
        lk.update(dflt.get("LIKEOPTS", {}))
        for k, v in dict(like_kwargs or {}).items():
            kv = slit_like_option(k, v)
            if kv is None:
                raise ValueError(f"SLITFIT: unknown option of the likelihood '{k}' "
                                 f"(options: {', '.join(SLIT_LIKE_OPTIONS)})")
            lk[kv[0]] = kv[1]
        self.like_kwargs = lk
        return freepar

    def _threads(self, threads):
        if threads is None:
            threads = int(self._params.get("THREADS", 1)) if "THREADS" in self._params else 1
        return int(threads)

    def _prepare_fit(self, freepar, verbose):
        vprint(verbose, banner("SLITFIT"))
        self._setup(freepar, verbose=verbose)
        if verbose:
            self.print_setup()
        theta0 = self._prior_transform(np.full(self.ndim, 0.5))
        l0 = self._log_like_func(theta0)
        if not np.isfinite(l0):
            raise BBaroloError("SLITFIT: the likelihood at the centre of the priors is not finite (check the "
                               "initial rings and the priors)")

    def print_setup(self, stream=None):
        """The data, the rings and the likelihood of the fit, in sections separated by blank lines."""
        import sys
        import textwrap
        st = stream or sys.stdout
        prep, ini, par = self.prep, self.initial, self._par
        sp = prep.spec
        w0, w1 = sp.values([0, sp.n - 1])
        det = np.flatnonzero(prep.lines["ok"])
        row = lambda label, text: st.write(f"  {label:15s}{text}\n")
        st.write("\nSlit\n")
        row("positions", f"{prep.slit.npos} of {prep.slit.pix:.4g} arcsec; {self.region[0]} to {self.region[1] - 1} "
                         "in the likelihood")
        row("window", f"{w0:.6g} to {w1:.6g} {sp.cunit}: {sp.n} channels of {sp.dv(0):.3g} km/s")
        row("emission", f"positions {int(det[0])} to {int(det[-1])}; velocities {prep.vrange[0]:.0f} to "
                        f"{prep.vrange[1]:.0f} km/s")
        row("seeing", f"{self._beam:.3g} arcsec (FWHM)")
        row("slit width", f"{self._width:.3g} arcsec")
        row("instrumental", f"{prep.sig_instr:.3g} km/s ({self._nch:.3g} channels, dispersion)")
        order = max(int(par.cont_order), 0)
        row("continuum", f"polynomial of degree {order}, subtracted" if par.contsub else "none")
        ex = str(par.exclude_windows).strip()
        if ex.upper() not in ("", "NONE"):
            row("not continuum", f"channels {ex} (EXCLWIND)")
        row("residual sky", "subtracted" if par.skyresid else "none")
        if par.maskwave:
            mw = par.maskwave
            row("masked", ", ".join(f"{mw[k]:g} to {mw[k + 1]:g}" for k in range(0, len(mw) - 1, 2)) +
                f" {sp.cunit} (MASKWAVE)")
        est = ini["estimated"]
        mark = lambda k: " (estimated)" if k in est else ""
        st.write(f"\nGeometry: INC {ini['inc']:g}, SLITANGLE {ini['slitangle']:g}{mark('SLITANGLE')}, YPOS "
                 f"{ini['yoffset']:g} arcsec; XPOS {ini['xpos']:.2f}{mark('XPOS')}, VSYS {ini['vsys']:.1f} km/s"
                 f"{mark('VSYS')}\n")
        radii = " ".join(f"{r:.4g}" for r in ini["radii"])
        st.write("\n" + textwrap.fill(f"Rings: {len(ini['radii'])}, radii (arcsec): {radii}", width=110,
                                      subsequent_indent="  ") + "\n")
        st.write(f"Fitting {self.ndim} parameters: {self.freepar_names}\n")
        nc = self.like_kwargs.get("noisecorr")
        corr = prep.corr
        st.write("\nLikelihood\n")
        row("normalisation", self._norm())
        if self._norm() != "NONE":
            row("density", f"deconvolved from the flux along the slit (smoothness "
                           f"{self.like_kwargs.get('densprof_smooth') or 0.:g})")
        row("noise", f"map, median {self.noiserms:.4g}" + (", times SIGMA (free)" if "sigma" in self.freepar_idx
                                                            else ""))
        row("log L x", f"1/{1 / self.likefactor:.3g} " +
            (f"(noise correlation: {corr['spectral']:.2f} channels x {corr['spatial']:.2f} positions)"
             if nc == "measured" else "(given)"))
        row("pixels", f"{self.nvalid}")
        if self.like_kwargs.get("smooth"):
            row("smoothness", smoothness_text(self._smooth_used(), self.like_kwargs["smooth"], per="seeing FWHM"))
        st.write("\nPriors\n")
        for key in self.freepar_idx:
            d = self.priors[key]
            args = ", ".join(f"{k}={v:.4g}" for k, v in d.items() if k != "name")
            st.write(f"  {key:15s}{d['name']}({args})\n")
        st.write("\n")

    def _initial_theta(self):
        """The free parameters at the initial rings (the value of each ring, or their median for NAME_SINGLE); SIGMA the
        fixed factor (1); NaN (the centre of the prior) for the parameters of the functions of radius."""
        th = np.full(self.ndim, np.nan)
        r = self._fit0.r
        for key, idx in self.freepar_idx.items():
            if "_p" in key:
                continue
            if key == "sigma":
                th[idx] = self.like_kwargs.get("sigma") or 1.
            elif key in ("vsys", "xpos"):
                th[idx] = float(np.median(r[key]))
            else:
                v = np.asarray(r[key], float)
                th[idx] = v if idx.size == v.size else np.median(v)
        if "dens" in self.freepar_idx and np.all(np.asarray(r["dens"]) == 1.):
            th[self.freepar_idx["dens"]] = np.nan
        return th

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
        """The best fit (params): its rings, densities, model of the slit and amplitudes (NORM LOCAL)."""
        logl, det = self.log_likelihood(self.params, details=True)
        det["logl"] = logl
        self.best = det
        chi2 = float(np.sum(self._W * (self._d - det["model"]) ** 2) / det["scale"] ** 2 / self.likefactor)
        self.best["chi2_reduced"] = chi2 / max(self.nvalid - self.ndim, 1)

    # ------------------------------------------------------------------ results
    def ring_table(self):
        """The best-fit rings with the 1-sigma errors of the free parameters: an astropy Table (radii in arcsec; VROT,
        VDISP, VRAD, VSYS in km/s; XPOS in pixels of the slit; DENS the densities of the model; E_<NAME>_LO/HI the
        errors), from the posterior samples (or the Laplace draws of optimize)."""
        from astropy.table import Table
        rd = self.best["rings"]
        nr = rd.nr
        t = Table()
        t["RADII"] = np.asarray(rd.r["radii"], float)
        cols = (("VROT", "vrot"), ("VDISP", "vdisp"), ("VRAD", "vrad"), ("VSYS", "vsys"), ("XPOS", "xpos"),
                ("INC", "inc"), ("SLITANGLE", "phi"), ("Z0", "z0"), ("DENS", "dens"))
        for name, key in cols:
            if key == "phi":
                t[name] = np.full(nr, self.initial["slitangle"])
            elif key == "dens":
                t[name] = np.asarray(self.best["dens"], float)
            else:
                t[name] = np.asarray(rd.r[key], float)
            lo, hi = self._ring_errors(key, nr)
            if lo is not None:
                t[f"E_{name}_LO"], t[f"E_{name}_HI"] = lo, hi
        return t

    def _ring_errors(self, key, nr):
        """The 1-sigma errors of a ring parameter in each ring (None if not free): the 16th-84th percentiles of its
        values in the rings of the posterior samples around their median."""
        if key not in self.freepar_idx and not any(k.startswith(key + "_p") for k in self.freepar_idx):
            return None, None
        S = np.asarray(self.samples_equal)
        S = S[np.random.default_rng(0).choice(len(S), min(len(S), 1000), replace=False)]
        vals = []
        rd = self._fit0.copy()
        for th in S:
            r = self._update_rings(rd, th)
            vals.append(np.asarray(r.r[key], float) * np.ones(nr))
        lo, med, hi = np.quantile(np.array(vals), SIGMA_QUANTILES, axis=0)
        return med - lo, hi - med

    def print_stats(self, stream=None):
        """The best fit and the 1-sigma intervals of the free parameters."""
        import sys
        st = stream or sys.stdout
        if not self.modCalculated:
            st.write(" SLITFIT: run() first.\n")
            return
        opt = self.optimum if self._method == "optimize" else None
        if opt is None:
            st.write(f"\n SLITFIT posterior ({self._method}, NORM {self._norm()}):\n")
        else:
            st.write(f"\n SLITFIT {'maximum a posteriori' if opt.posterior else 'maximum likelihood'} "
                     f"({opt.method}, NORM {self._norm()}; errors: "
                     f"{'Laplace approximation' if opt.cov is not None else 'not computed'}):\n")
        for name, val, (lo, med, hi) in zip(self.freepar_names, self.params, self.quantiles):
            val = med if opt is None else val
            st.write(f"   {name:12s} = {val:12.4g}  -{med - lo:<10.3g} +{hi - med:<10.3g}\n")
        if self.best is not None:
            st.write(f"   reduced chi-square of the best fit: {self.best['chi2_reduced']:.3g}\n")
        sm = smoothness_result(self.smoothness_stats())
        if sm:
            st.write(f"   smoothness: {sm}\n")
        st.write("\n")

    def corner_plot(self, labels=None, truths=None, quantiles=SIGMA_QUANTILES, **kwargs):
        """Corner plot of the posterior (needs the package corner)."""
        import corner
        defaults = dict(max_n_ticks=5, color="darkblue", show_titles=True, plot_datapoints=False)
        if truths is not None:
            kwargs["truths"] = truths
        return corner.corner(self.samples_equal, labels=labels or self.freepar_names, quantiles=quantiles,
                             title_quantiles=quantiles, **{**defaults, **kwargs})

    def write(self, outfolder=None, plots=None, verbose=True):
        """Write the outputs in outfolder (default OUTFOLDER, or ./output/<OBJECT>/): see slit_outputs.write_outputs.
        Returns the files written."""
        from .slit_outputs import write_outputs
        if not self.modCalculated:
            raise RuntimeError("BayesianSlitFit: nothing to write, run() first.")
        return write_outputs(self, outfolder, plots, verbose)

    def model_hdu(self):
        """The best model of the slit as an HDU (that of <name>_slit_model.fits: the grid of the prepared data, NaN
        outside the region of the fit). Needs run()."""
        from .slit_outputs import model_array
        prep = self.prep
        return prep.slit.to_hdu(model_array(self), k0=prep.k0, extra={"BBTASK": "SLITFIT"})

    def plot_pv(self, **kwargs):
        """The data, the best model and the residuals of the slit (a matplotlib Figure; nothing is written)."""
        from ..plots.slitfit import plot_slit_pv
        from .slit_outputs import pv_inputs
        return plot_slit_pv(**{**pv_inputs(self), **kwargs})

    def plot_parameters(self, **kwargs):
        """The rings of the best fit with their errors against radius (a matplotlib Figure; nothing is written)."""
        from ..plots.slitfit import plot_slit_parameters
        from .slit_outputs import parameter_inputs
        return plot_slit_parameters(**{**parameter_inputs(self), **kwargs})


def _folded_vsys(pos, v, xpos, maxd=None):
    """The systemic velocity: the median of the mean velocities of the pairs of positions at the same distance from
    xpos on the two sides of the slit (interpolated), or the midrange of the velocities if there are no pairs."""
    if pos.size < 3:
        return float(np.median(v)) if v.size else 0.
    o = np.argsort(pos)
    pos, v = pos[o], v[o]
    d = np.abs(pos - xpos)
    left, right = pos < xpos, pos > xpos
    if left.sum() >= 2 and right.sum() >= 2:
        dmax = min(d[left].max(), d[right].max())
        dd = np.linspace(min(1., dmax), dmax, 20)
        vl = np.interp(xpos - dd, pos[left], v[left])
        vr = np.interp(xpos + dd, pos[right], v[right])
        return float(np.median(0.5 * (vl + vr)))
    return float(0.5 * (np.percentile(v, 10) + np.percentile(v, 90)))
