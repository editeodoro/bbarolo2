"""
Bayesian3DFIT: the fitter (parameters, rings, priors, free parameters, setup and sampling). The
likelihood, the density profile and the outputs are in the mixins of likelihood.py, density.py and
outputs.py; the samplers and the pool in sampling.py.
"""

import os
import copy
import numpy as np
from scipy import stats

from ..tasks import FitMod3D, build_param, _checked
from ..bbutils import BBaroloError
from .common import vprint, get_distribution, show_bar, SIGMA_QUANTILES
from .rings import _RingDict
from .likelihood import LikelihoodMixin
from .density import DensityMixin
from .outputs import OutputsMixin
from .sampling import sample, sampler_options, dyfunc
from .optimize import SAMPLER_NAMES, sampler_options as optimize_options

# the default functions of radius of VROT_FUNC, VDISP_FUNC and DENS_FUNC (Bayesian3DFIT.funcs)
DEFAULT_FUNCS = dict(vrot=lambda R, p1, p2: 2. / np.pi * p1 * np.arctan(R / p2),     # arctan curve
                     vdisp=lambda R, p1, p2: p1 + p2 * R,                            # linear
                     dens=lambda R, p1, p2: p1 * np.exp(-R / p2))                    # exponential


class Bayesian3DFIT(OutputsMixin, DensityMixin, LikelihoodMixin, FitMod3D):
    """ Fit a galaxy model to a 3D datacube in a Bayesian framework (nested sampling).

    Bayesian3DFIT(cube, **params): the cube (FITS file, HDU or (data, header)) and any
    BBarolo parameter, including the rings (radii or nradii/radsep, xpos, ypos, vsys, vrot,
    vdisp, inc, pa, z0, vrad, dens, vvert, dvdz, zcyl: single values or one value per ring).
    Ring parameters not fitted are kept fixed. The initial rings are those of 3DFIT for the same
    parameters: the values not given are read from RINGFILE, estimated from the data as set by
    GUESSTYPE (the rings, centre, VSYS, VROT, INC, PA, and VDISP with GUESSTYPE FIT) or set to the
    defaults (VDISP=8, Z0=one pixel, VRAD=VVERT=DVDZ=ZCYL=0, DENS=1). The default priors follow
    the initial values (e.g. INC +- 10 deg, PA +- 15 deg, VROT 0 to twice the largest initial
    VROT: see _set_default_priors and the keyword PRIORS).

    The parameters of the BAYESIAN3DFIT task of parameter files are accepted too: FREE (the
    free parameters, e.g. 'VROT VDISP_SINGLE INC_SINGLE'; default
    keywords.BAYES_DEFAULT_FREE, used when neither FREE nor freepar is given), SAMPLER, SAMPLEROPTS and RUNOPTS are the defaults of run(); PRIORS sets entries
    of priors (e.g. 'VROT=(uniform,loc=0,scale=300)'). Bayesian3DFIT.from_parfile(file)
    reads a parameter file.

    Methods
    -------
    update(**params):        change BBarolo parameters (and the rings).
    run(freepar, method, threads, log_like, like_kwargs, sampler_kwargs, run_kwargs, verbose,
        dynamic, **params):  sample the posterior ('dynesty' or 'nautilus'); nothing is written.
    run_geometry(freepar, **kwargs): fit only the geometry on the total intensity map.
    write(outfolder, plots): write all the outputs, as the BAYESIAN3DFIT task.
    write_bestmodel(plots, norm, **kwargs): write the best model, maps, PVs and plots.
    corner_plot(...), plot_profiles(...), print_stats(...), print_priors(), chi2_reduced()
    plot_parameters(), plot_chanmaps(), plot_pvs(), plot_pvslices(), plot_kinmaps(): the plots of the
                             best model written by write(), as matplotlib Figures (nothing is written).
    write_allplots(outfolder): those plots as PDF files, without the other outputs.
    model_outputs(norm):     the best model with its maps and PV slices, in memory.
    ring_errors(param):      1-sigma errors of a ring parameter in each ring of the best fit.
    save_results(outfile), load_results(inputfile, sampler)

    Attributes: priors (dict of prior definitions, see the module documentation), funcs
    (parametric functions of radius), lambdas (smoothness penalties), like_kwargs (the options
    of the likelihood of the fit, with the defaults), params (best-fit values of the free
    parameters), samples, samples_equal, quantiles, outri (best-fit rings).
    """

    TASK = "BAYESIAN3DFIT"              # its task flag in parameter files (the model engine is that of 3DFIT)

    _KEYWORD_OPTIONS = ("FREE", "SAMPLER", "PRIORS", "SAMPLEROPTS", "RUNOPTS", "LIKEOPTS")

    def __init__(self, cube=None, **params):
        self.taskname = "BAYESIAN3DFIT"
        self._method = None
        self.freepar_idx = None
        self.ndim = 0
        self.freepar_names = None
        self.priors = {}
        self.prior_distr = None
        self.funcs = {}
        self._galfit = self._ellprof = None
        self._cube_obj = None
        self._par = None
        self.modCalculated = False
        self._inri = self.outri = None
        self._default_pri = {}
        self.results = None
        self.sampler = None
        self.optimum = None                            # (the details of a fit of optimize)
        self.noise_correlation = None
        self._noisecorr_failed = False                 # (the measure of the correlation failed: not retried)
        self._likemask_key = None
        self.dens_profile = None
        self.useBBres = False
        self._set_like_options({})              # like_kwargs: the options of the likelihood (defaults)
        super(Bayesian3DFIT, self).__init__(cube, **params)
        if self._cube is not None:
            self._init_rings()

    def update(self, **params):
        """ Change BBarolo parameters (None restores the default). Changing the rings
        resets them (and the default priors that depend on them). FREE, SAMPLER, SAMPLEROPTS
        and RUNOPTS set the defaults of run(); PRIORS sets entries of self.priors (see
        bbarolo.params.bayes_priors). These are strings as in a parameter file, or Python values:
        FREE a list of names, PRIORS a dict as self.priors, the options dicts (see
        bbarolo.params.bayes_value_string). Returns self. """
        from ..keywords import lookup
        from ..params import bayes_value_string
        given = {}
        for k, v in list(params.items()):        # (Python values: as in a parameter file)
            kw = lookup(k)
            if kw is not None and kw.name in self._KEYWORD_OPTIONS and v is not None and not isinstance(v, str):
                params[k] = bayes_value_string(kw.name, v)
        for k, v in params.items():              # checked before they are set
            kw = lookup(k)
            if kw is not None and kw.name in self._KEYWORD_OPTIONS and v is not None:
                given[kw.name] = self._keyword_option(kw.name, v)
        super(Bayesian3DFIT, self).update(**params)
        if not hasattr(self, "priors"):              # called by the base constructor
            return self
        if "PRIORS" in given:
            self.priors.update(given["PRIORS"])
            if self.prior_distr is not None:
                for k in given["PRIORS"]:
                    self.prior_distr[k] = None       # frozen again by the next run
        if params:
            self._galfit = self._ellprof = None
            self._cube_obj = None
            if self._inri is not None:
                self._init_rings()
        return self

    def _keyword_option(self, name, value):
        """ The value of FREE (list of free parameters), SAMPLER ('dynesty', 'nautilus' or 'optimise'),
        PRIORS (dict of priors), SAMPLEROPTS, RUNOPTS or LIKEOPTS (dict of options), from a
        string as in a parameter file. Raises ValueError. """
        from ..params import bayes_freepar, bayes_priors, bayes_options, bayes_likeopts, BAYES_FUNC_PARS
        s = str(value)
        try:
            if name == "FREE":
                return bayes_freepar(s, funcs=set(BAYES_FUNC_PARS) | set(getattr(self, 'funcs', {})))
            if name == "LIKEOPTS":
                return bayes_likeopts(s)
            if name == "PRIORS":
                return bayes_priors(s)
            if name == "SAMPLER":
                m = s.strip().lower()
                m = "optimise" if m == "optimize" else m
                if m not in ("dynesty", "nautilus", "optimise"):
                    raise ValueError(f"unknown sampler {s.strip()} (NAUTILUS, DYNESTY or OPTIMISE)")
                return m
            return bayes_options(s)
        except ValueError as e:
            raise ValueError(f"BAYESIAN3DFIT: {name}: {e}") from None

    def _run_defaults(self):
        """ The defaults of run() given as parameters: FREE, SAMPLER, SAMPLEROPTS, RUNOPTS, LIKEOPTS. """
        return {k: self._keyword_option(k, self._params[k])
                for k in ("FREE", "SAMPLER", "SAMPLEROPTS", "RUNOPTS", "LIKEOPTS") if self._params.get(k) is not None}

    def _set_like_options(self, opts, custom=False):
        """ The options of the likelihood (run(like_kwargs=...) and LIKEOPTS; see
        bbarolo.params.bayes_like_option), checked and with the defaults. Sets like_kwargs (all
        the options of the fit) and _like_kwargs (those passed to the log-likelihood at each
        call: sigma, fracerr, lambdas and, with user-defined residuals or log-likelihood
        (custom=True), any other keyword; otherwise an unknown option raises ValueError). """
        from ..params import bayes_like_option, LIKE_OPTIONS, LIKE_SETUP
        out = {k: LIKE_OPTIONS[k] for k in LIKE_SETUP}
        call = {}
        for k, v in dict(opts or {}).items():
            kv = bayes_like_option(k, v)
            if kv is None:
                if not custom:
                    raise ValueError(f"Bayesian3DFIT: unknown option of the likelihood '{k}' (options: "
                                     + ", ".join(LIKE_OPTIONS) + "; any other only with user-defined residuals "
                                     "or log-likelihood)")
                kv = (k, v)
            key, val = kv
            if key not in LIKE_SETUP:
                if val is None:
                    out.pop(key, None)
                    call.pop(key, None)
                    continue
                call[key] = val
            out[key] = val
        self.like_kwargs, self._like_kwargs = out, call
        return out

    def _all_params(self, outfolder=None, fitsfile=None):
        # The defaults of the Bayesian fit that differ from 3DFIT (keywords.TASK_DEFAULTS): ANALYTIC
        # models, whose likelihood has no Monte-Carlo noise (with sigma = the noise, the noise of MC
        # models makes it rough and the sampling slow) and which are about twice faster. A GALMODTYPE
        # that is given (MC) is kept.
        from ..keywords import TASK_DEFAULTS
        prm = super(Bayesian3DFIT, self)._all_params(outfolder=outfolder, fitsfile=fitsfile)
        for k, v in TASK_DEFAULTS[self.TASK].items():
            prm.setdefault(k, v)
        return prm

    def _outfolder(self):
        of = self._params.get("OUTFOLDER")
        of = os.path.expanduser(str(of).strip()) if of not in (None, "") else self._default_outfolder()
        return of if of.endswith("/") else of + "/"

    def _make_par(self):
        """ The Param of the model engine (verbose off, one thread: the samples are computed
        in parallel by processes). """
        of = self._outfolder()
        hdu = self._input_hdu()
        prm = self._all_params(outfolder=of, fitsfile=self._cube if hdu is None else "in-memory cube")
        prm.pop(self.TASK, None)                 # the engine is that of 3DFIT...
        prm.pop("FREE", None)                    # ...whose FREE is not that of BAYESIAN3DFIT
        prm["3DFIT"] = True
        prm["VERBOSE"] = False
        prm["THREADS"] = 1
        prm["TWOSTAGE"] = False                  # one set of rings (<prefix>_rings_final1.txt), also for the plots
        return _checked(build_param(prm))

    def _load_cube(self):
        """ Read the cube (with the current parameters). """
        from ..cube import Cube
        par = self._make_par()
        c = Cube(par=par)
        p = c.pars()
        if p.outFolder == "":
            p.setOutfolder(self._outfolder())
        hdu = self._input_hdu()
        ok = c.readCube(p.imageFile, printInfo=False) if hdu is None else c.readHDU(hdu, name=p.imageFile)
        if not ok:
            raise BBaroloError(f"Cannot read {p.imageFile}")
        self._cube_obj, self._par = c, p
        return c

    def _init_rings(self, show_initial=False):
        """ The initial rings from the parameters, as those of 3DFIT (galfit.initial_rings: the
        values not given are read from RINGFILE, estimated from the data with GUESSTYPE, or set to
        the defaults of 3DFIT; the mask of the fit is built only when something is estimated), and
        the default priors, which depend on them. show_initial: print the table of the initial
        parameters of 3DFIT (given, estimated or default); otherwise, the estimated values only.
        (3DFIT then changes its rings for REVERSE and, with PA free, refines PA and VROT with a 2D
        fit: those are steps of its fit, not of the initial rings.) """
        from ..galfit import initial_rings, initialParametersText
        c = self._cube_obj if self._cube_obj is not None else self._load_cube()
        c.checkBeam()                            # (as 3DFIT; the default priors use the beam)
        ini = initial_rings(c, copy.deepcopy(c.pars().gf), write=False, buildmask=True, task=self.TASK)
        inR, toEstimate = ini.rings, ini.toEstimate
        if show_initial:
            print(initialParametersText(inR, ini.estimated), end="")
        elif toEstimate and c.pars().verbose:          # (nothing at VERBOSE 0)
            print(f"\n{self.taskname}: initial rings ({inR.nr} rings every {inR.radsep:.2f} arcsec) with "
                  "estimated parameters:")
            for k, n in (("xpos", "XPOS"), ("ypos", "YPOS"), ("vsys", "VSYS"), ("vrot", "VROT"), ("inc", "INC"),
                         ("phi", "PA")):
                print(f"  {n:5s} = {np.median(getattr(inR, k)):.3f}")
        self._inri = _RingDict.from_rings(inR)
        self._inri_base = None                   # (the rings before their extension for RADMAX, see _rings_for_radmax)
        self._nradii_estimated = bool(ini.estimated.get('NRADII', False))
        self._radmax_guess = float(self._inri.r['radii'][-1])
        self._set_default_priors()

    def _rings_for_radmax(self, verbose=True):
        """ With RADMAX free as the edge of the disk (like_kwargs radmax 'edge' or 'taper') and the number of
        rings estimated (NRADII and RADII not given), the rings are extended, with the same separation and the
        values of the last ring, out to the estimated outer radius (_radmax_guess, the last radius of the
        estimated rings) plus one beam FWHM (_radmax_top), the upper end of the default prior of RADMAX: the
        edge can then be anywhere up
        to there. Otherwise the estimated rings are used. Returns True if the rings changed (the default
        priors, the engine and the free parameters must then be set again). """
        mode = str(self.like_kwargs.get('radmax') or 'edge')
        want = 'radmax' in self.freepar_idx and mode != 'ring' and getattr(self, '_nradii_estimated', False)
        if want == (getattr(self, '_inri_base', None) is not None):
            return False
        if want:
            base = self._inri
            r = {k: np.asarray(v, float) for k, v in base.r.items()}
            R = r['radii']
            sep = float(R[-1] - R[-2]) if base.nr > 1 else float(R[-1])
            if not sep > 0:
                return False
            nadd = int(np.ceil((self._radmax_top() - R[-1]) / sep - 1e-6))
            if nadd < 1:
                return False
            ext = _RingDict(base.nr + nadd)
            for k, v in r.items():
                new = R[-1] + sep * np.arange(1, nadd + 1) if k == 'radii' else np.full(nadd, v[-1])
                ext.modify_parameter(k, np.r_[v, new])
            ext.edge, ext.edgewidth = base.edge, base.edgewidth
            ext.make_object()
            self._inri, self._inri_base = ext, base
            vprint(verbose, f"\n{self.taskname}: RADMAX free: the {base.nr} estimated rings are extended to {ext.nr} (out to "
                   f"{float(ext.r['radii'][-1]):.4g} arcsec, the estimated radius {self._radmax_guess:.4g} "
                   "arcsec plus one beam)")
        else:
            self._inri, self._inri_base = self._inri_base, None
        self._galfit = self._ellprof = None
        self._set_default_priors()
        return True

    def _beam_fwhm(self):
        """ The beam FWHM of the cube (major axis, arcsec). """
        from ..bbutils import arcsconv
        h = self._cube_obj.Head()
        return abs(h.Bmaj()) * arcsconv(h.Cunit(0))

    def _radmax_top(self):
        """ The upper end of the default prior of RADMAX (arcsec): the estimated outer radius (_radmax_guess,
        the last initial radius) plus one beam FWHM. """
        return float(self._radmax_guess) + self._beam_fwhm()

    def _set_default_priors(self):
        """ Default priors (uniform), parametric functions and smoothness penalties.

        The default priors of the ring parameters follow the initial rings (given or estimated, see
        _init_rings), and change with them. A prior serves all the rings, so 'initial +- w' spans
        the initial values of every ring, from the smallest - w to the largest + w:
          INC     initial +- 10 deg, within 0-90 deg;
          PA      initial +- 15 deg (one sense of rotation: PA+180 is excluded);
          VROT    0 to 2 x the largest initial VROT (0-350 km/s if it is not positive): the largest,
                  because the prior of each ring must hold the outer, flat part of the curve; the
                  value of each ring would give the inner rings of a rising curve a narrow prior,
                  where the initial values are the least reliable (and an estimate is one value);
          XPOS, YPOS initial +- 2 beam FWHM (major axis, in pixels);
          VSYS    initial +- 5 channels, at least +- 20 km/s;
          VDISP   0 to 2 x the largest initial VDISP, at least 0-50 km/s;
          Z0      0 to one beam FWHM (arcsec);
          RADMAX  the estimated outer radius (_radmax_guess, the last initial radius) - one beam FWHM to
                  + one beam FWHM, within the rings (see _prior_kwargs and _rings_for_radmax);
          VROT_P1 (the amplitude of the arctan curve) 1 km/s to 2 x the largest initial VROT.
        The others do not depend on the rings: VRAD 0-50 km/s, VVERT 0-30 km/s, DVDZ 0-2 km/s/arcsec,
        ZCYL 0-5 arcsec, DENS 0.1-200.1, SIGMA and FRACERR log-uniform (see _prior_kwargs), and those of
        the functions of radius. A prior that was changed (in priors, or with PRIORS) is kept: only
        those equal to the previous defaults (_default_pri) follow the rings. """
        from ..bbutils import DeltaVel, arcsconv
        rr = {k: np.asarray(v, dtype=float) for k, v in self._inri.r.items()}
        h = self._cube_obj.Head()
        beam = abs(h.Bmaj()) * arcsconv(h.Cunit(0))              # the beam FWHM (arcsec)
        beampix = abs(h.Bmaj()) / h.PixScale()                    # (pixels)

        def around(v, w, lo=-np.inf, hi=np.inf):
            a, b = max(float(np.min(v)) - w, lo), min(float(np.max(v)) + w, hi)
            return dict(name='uniform', loc=a, scale=b - a)
        vtop = 2. * float(rr['vrot'].max()) if rr['vrot'].max() > 0 else 350.
        guess = float(getattr(self, '_radmax_guess', rr['radii'][-1]))
        rlo = max(guess - beam, 0.)
        new = dict(
            xpos=around(rr['xpos'], 2. * beampix),                           # +- 2 beams
            ypos=around(rr['ypos'], 2. * beampix),
            vsys=around(rr['vsys'], max(5. * abs(DeltaVel(h)), 20.)),        # +- 5 channels, >= 20 km/s
            vrot=dict(name='uniform', loc=0., scale=vtop),                   # 0 - 2 x VROT
            vdisp=dict(name='uniform', loc=0., scale=max(2. * float(rr['vdisp'].max()), 50.)),
            vrad=dict(name='uniform', loc=0, scale=50),                       # 0-50 km/s
            inc=around(rr['inc'], 10., 0., 90.),                              # +- 10 degrees
            phi=around(np.degrees(np.unwrap(np.radians(rr['phi']))), 15.),    # +- 15 degrees
            vvert=dict(name='uniform', loc=0, scale=30),                      # 0-30 km/s
            dvdz=dict(name='uniform', loc=0, scale=2),                        # 0-2 km/s/arcsec
            zcyl=dict(name='uniform', loc=0, scale=5),                        # 0-5 arcsec
            dens=dict(name='uniform', loc=0.1, scale=200),                    # 0.1-200.1 (flux units x km/s/arcsec2)
            z0=dict(name='uniform', loc=0., scale=beam),                     # 0 - 1 beam
            sigma=dict(name='loguniform', a=None, b=None),                    # None: 0.5 and 1000 x noise rms
            fracerr=dict(name='loguniform', a=1e-3, b=1.),                    # fractional model error
            radmax=dict(name='uniform', loc=rlo, scale=guess + beam - rlo),        # guess +- 1 beam
            vrot_p1=dict(name='uniform', loc=1., scale=max(vtop, 2.) - 1.),
            vrot_p2=dict(name='uniform', loc=1, scale=1000),
            vdisp_p1=dict(name='uniform', loc=0, scale=10),
            vdisp_p2=dict(name='uniform', loc=-10, scale=20),
            dens_p1=dict(name='uniform', loc=1, scale=20),
            dens_p2=dict(name='uniform', loc=1, scale=200))
        for k, v in new.items():
            if k not in self.priors or self.priors[k] == self._default_pri.get(k):
                self.priors[k] = dict(v)
        self._default_pri = {k: dict(v) for k, v in new.items()}

        # Parametric functions of radius (default only for dens, vrot and vdisp)
        for k, f in DEFAULT_FUNCS.items():
            self.funcs.setdefault(k, f)

        if self.prior_distr is None:            # (the distributions given are kept; those frozen by _setup are
            self.prior_distr = {}               # frozen again when their definition in priors changes)
        # Lambda parameters for the smoothness penalty in the likelihood
        if not hasattr(self, 'lambdas'):
            self.lambdas = {key: 0.0 for key in ['vrot', 'vdisp', 'dens', 'vrad', 'phi', 'inc', 'xpos', 'ypos', 'z0']}

    def _setup(self, freepar, **kwargs):
        """ The setup of a fit with the free parameters freepar and the options of the likelihood in
        like_kwargs (see run), in this order (each step needs the previous ones):

          1. the parameters given (kwargs) and the initial rings (_init_rings: the estimates of the
             geometry not given, and the default priors);
          2. the free parameters (_parse_freepar): ndim, freepar_idx, freepar_names; with RADMAX free,
             the estimated rings extended out to their radius plus one beam (_rings_for_radmax);
          3. the 3DFIT engine of the cube, its data, mask and noise (_setup_engine, once per cube);
          4. the priors, frozen (_freeze_priors): prior_distr (the region of the likelihood depends
             on their bounds);
          5. the likelihood (_setup_likelihood): its voxels, its region, the factor for the
             correlation of the noise and the noise of the total intensity map;
          6. the rings of the fit (_setup_fit_rings): the initial rings with the density profile
             (computed in the region of the likelihood) and the rings of the evaluations. """
        if kwargs:
            self.update(**{k: v for k, v in kwargs.items() if k not in ('verbose', 'dynamic')})
        if self._inri is None:
            self._init_rings()
        self.useBBres = bool(self.like_kwargs.get('bbres', False))
        self._parse_freepar(freepar)
        if self._rings_for_radmax(kwargs.get('verbose', True)):
            self._parse_freepar(freepar)
        self._setup_engine()
        self._setup_radmax()
        self._freeze_priors()
        self._setup_likelihood()
        self._setup_fit_rings()

    def _parse_freepar(self, freepar):
        """ The free parameters (names as in run): ndim, freepar_idx (the indices of each one in the
        vector of parameters), freepar_names (one name per dimension) and _freepar_inp. """
        self.freepar_idx = {}
        self.ndim = 0
        self.freepar_names = []
        self._freepar_inp = freepar
        dup = {key: False for key in self.priors.keys()}
        okpars = ['sigma', 'radmax', 'fracerr']
        for pname in freepar:
            s = pname.split('_')
            if s[0] not in self._inri.r and s[0] not in okpars:
                raise ValueError(f"ERROR! The requested free parameter is unknown: {s[0]}")
            if dup.get(s[0], False):
                raise ValueError(f"ERROR! Multiple entries for {s[0]} parameter. Please choose one!")
            dup[s[0]] = True
            if len(s) == 2 and 'single' in pname:
                self.freepar_idx[s[0]] = np.array([self.ndim])
                self.ndim += 1
                self.freepar_names.append(s[0])
            elif len(s) == 2 and 'func' in pname:
                if s[0] not in self.funcs:
                    raise ValueError(f"ERROR! The functional form for {s[0]} is unknown. Please define it first.")
                nparams = self.funcs[s[0]].__code__.co_argcount - 1
                for i in range(nparams):
                    self.freepar_names.append(f'{s[0]}_p{i + 1}')
                    self.freepar_idx[f'{s[0]}_p{i + 1}'] = np.arange(self.ndim, self.ndim + 1, dtype='int')
                    self.ndim += 1
            elif len(s) == 1:
                if s[0] in okpars:
                    self.freepar_idx[s[0]] = np.array([self.ndim])
                    self.ndim += 1
                    self.freepar_names.append(s[0])
                else:
                    self.freepar_idx[s[0]] = np.arange(self.ndim, self.ndim + self._inri.nr, dtype='int')
                    self.ndim += self._inri.nr
                    for i in range(self._inri.nr):
                        self.freepar_names.append(f'{s[0]}{i + 1}')
            else:
                raise ValueError(f"ERROR! Cannot interpret free parameter '{pname}'.")

    def _setup_engine(self):
        """ The 3DFIT engine (Galfit) of the cube with the initial rings, the data, the mask and the
        noise (_setup_noise). Once per cube: update() resets it. """
        if self._galfit is None:
            from ..galfit import Galfit
            c = self._cube_obj if self._cube_obj is not None else self._load_cube()
            self._galfit = Galfit(c, self._inri.object, self._par)
            self.mask = c.Mask()
            self.data = c.array
            self._setup_noise()

    def _setup_radmax(self):
        """ What the free parameter RADMAX is (like_kwargs radmax): 'edge' (the default), the outer
        edge of the disk, with the rings fixed (the model has no emission beyond it: see
        galmod.edge_weights); 'taper', a Gaussian edge of width radmax_width (default a quarter of
        the beam FWHM); 'ring', the radius of the last ring (which moves, with its annulus: the
        RADMAX of the original module). The default prior is from the estimated outer radius - one beam
        FWHM to + one beam FWHM, within the rings (see _prior_kwargs and _rings_for_radmax). """
        self._radmax_mode = str(self.like_kwargs.get('radmax') or 'edge')
        w = self.like_kwargs.get('radmax_width')
        if self._radmax_mode != 'taper':
            w = 0.
        elif w is None:
            w = 0.25 * abs(self._cube_obj.Head().Bmaj()) * 3600.
        self._radmax_width = float(w)

    def _freeze_priors(self):
        """ The priors of the free parameters as frozen distributions, prior_distr (sigma: None
        values are relative to the measured noise, see _prior_kwargs). A distribution frozen here is
        frozen again when its definition in self.priors has changed (e.g. between two runs); one
        given in self.prior_distr is kept. """
        if self.prior_distr is None:
            self.prior_distr = {}
        frozen = self.__dict__.setdefault('_frozen_priors', {})
        for key in self.freepar_idx:
            d = self.prior_distr.get(key)
            if d is not None and key in frozen and frozen[key][0] is d \
                    and frozen[key][1] != self._prior_definition(key):
                d = None
            if d is None:
                try:
                    d = get_distribution(self.priors[key]['name'], **self._prior_kwargs(key))
                except Exception:
                    raise ValueError(f"Something is wrong in the prior for parameter '{key}'. Please double check!")
                self.prior_distr[key] = d
                frozen[key] = (d, self._prior_definition(key))
            elif not isinstance(d, stats._distn_infrastructure.rv_continuous_frozen):
                raise ValueError(f"ERROR! Prior for {key} should be None or a scipy.stats distribution.")

    def _setup_fit_rings(self):
        """ The rings of the fit: _fit0, the initial rings (self._inri, never changed by the fit) with
        the density profile when DENS is not fitted, and _work, the rings of the evaluations of the
        likelihood (overwritten at each call, so that nothing carries over to the next run or to the
        saved results). Also useNorm (DENS not fitted), update_prof (the profile changes with the
        free geometry) and the parameters that must be positive. """
        self._fit0 = self._inri.copy()
        self._free_positive = sorted({k.split('_')[0] for k in self.freepar_idx} & set(self._POSITIVE))
        # If the density is not fitted, the model is normalized with the azimuthal profile
        self.useNorm = not any('dens' in sub for sub in self.freepar_names)
        self.update_prof = False
        if self.useNorm and not self.useBBres:
            self._set_density(self._fit0)
            # The profile changes during the fit only if the geometry (or the thickness) is fitted
            self.update_prof = any(sub in string for string in self.freepar_names
                                   for sub in ['inc', 'phi', 'xpos', 'ypos', 'radmax', 'z0'])
        self._work = self._fit0.copy()

    def _prior_kwargs(self, key):
        """ Parameters of the prior of key (self.priors without 'name'). For sigma, None values
        are relative to the measured noise rms: loguniform a = 0.5 rms, b = 1000 rms (the
        default prior: scale-invariant between half and 1000 times the noise); scale = rms
        (e.g. halfcauchy, the default before November 2026). The default prior of RADMAX, from the
        estimated outer radius ± one beam FWHM, is kept within the rings: an edge (radmax 'edge' or 'taper') ends at
        the outer edge of the disk of the rings (galmod.outer_edge: the last radius with NODES rings,
        half a ring beyond it otherwise), beyond which it has no effect; the last ring (radmax 'ring')
        stays beyond the middle of the last two radii, so that the radii stay in order. """
        from ..galmod import outer_edge
        kw = {k: v for k, v in self.priors[key].items() if k != "name"}
        if key == 'radmax' and self.priors[key] == getattr(self, '_default_pri', {}).get(key) \
                and self._galfit is not None:
            lo, hi = kw['loc'], kw['loc'] + kw['scale']
            radii = np.asarray(self._inri.r['radii'], float)
            if getattr(self, '_radmax_mode', 'edge') == 'ring':
                if len(radii) > 1:
                    lo = max(lo, 0.5 * (radii[-2] + radii[-1]))
            else:
                hi = min(hi, outer_edge(self._inri.object, self._galfit.model.ringmode))
            kw['loc'], kw['scale'] = lo, hi - lo
        if key == 'sigma':
            for k, f in (('a', 0.5), ('b', 1000.), ('scale', 1.)):
                if k in kw and kw[k] is None:
                    kw[k] = f * float(self.noiserms)
        return kw

    def _prior_definition(self, key):
        """ The distribution and the parameters of the prior of key (None if not defined). """
        try:
            return self.priors[key]['name'], self._prior_kwargs(key)
        except Exception:
            return None

    def _prior_transform(self, u):
        """ Transform the unit cube into the parameter space using the prior distributions """
        p = np.zeros_like(u)
        for key in self.freepar_idx:
            p[self.freepar_idx[key]] = self.prior_distr[key].ppf(u[self.freepar_idx[key]])
        return p

    def run(self, freepar=None, method=None, threads=None, *, log_like=None, like_kwargs: dict = {},
            sampler_kwargs: dict = {}, run_kwargs: dict = {}, **kwargs):
        """ Sample the posterior.

        Args:
          threads (int):        number of processes for the sampler (default: THREADS if it was
                                given, otherwise 1 = serial)
          freepar (list):       the free parameters (here or as the parameter FREE; default
                                'VROT VDISP XPOS_SINGLE YPOS_SINGLE VSYS_SINGLE INC_SINGLE PA_SINGLE
                                Z0_SINGLE RADMAX FRACERR', keywords.BAYES_DEFAULT_FREE), e.g.
                                ['vrot','vdisp_single','inc_single','sigma'], or a string as FREE in
                                a parameter file ('VROT VDISP_SINGLE INC_SINGLE SIGMA'). Names are
                                case-insensitive (PA is phi, DISP is vdisp)
          method (str):         'nautilus' or 'dynesty' (default: SAMPLER if given, otherwise 'nautilus');
                                'optimise' (or 'optimize'): optimize(), with sampler_kwargs its arguments
                                (method, x0, posterior, errors, nstart, seed, step, nsamples) and run_kwargs
                                the options of its method (SAMPLER OPTIMISE in parameter files)
          log_like (callable):  a custom log-likelihood f(theta, **like_kwargs) (default _log_likelihood)
          like_kwargs (dict):   the options of the likelihood, added to LIKEOPTS if given (they are
                                kept in self.like_kwargs, with the defaults; see
                                bbarolo.params.bayes_like_option):
                                likemask ('dilated', default: the data in the mask dilated by
                                dilation; 'none': all the voxels of the region; see _setup_likemask),
                                dilation ((radius in beams, channels), default (1, 3)),
                                noisecorr ('measured', default: the log-likelihood x 1/correlation
                                volume of the noise measured from the voxels without emission;
                                'beam': x 1/beam area in pixels; a number N: x 1/N, N=1 for
                                independent voxels; see _set_likefactor),
                                densprof_smooth (smoothness of the deconvolved density profile,
                                default 0.1), bbres (True: BBarolo's 3DFIT residuals instead of
                                the Gaussian likelihood), sigma and fracerr (fixed noise per voxel
                                and fractional model error, when not free), lambdas (smoothness
                                penalties); with user-defined residuals or log_like, also their
                                own keywords
          sampler_kwargs (dict): arguments for the sampler (e.g. nlive), added to SAMPLEROPTS if
                                given (there, dynamic=True: the dynamic sampler of dynesty; seed=N:
                                a reproducible sampling, also with dynesty)
          run_kwargs (dict):    arguments for the run (e.g. dlogz, maxiter), added to RUNOPTS if given
          kwargs:               verbose (bool), dynamic (bool, dynesty only: the dynamic sampler;
                                default False, the static one), and any BBarolo parameter
        """
        params = {k: v for k, v in kwargs.items() if k not in ('verbose', 'dynamic')}
        self._check_moved(kwargs, 'run')
        if params:
            self.update(**params)
        kwargs = {k: v for k, v in kwargs.items() if k in ('verbose', 'dynamic')}
        dflt = self._run_defaults()
        method = str(dflt.get('SAMPLER', 'nautilus') if method is None else method).lower()
        sampler_kwargs = {**dflt.get('SAMPLEROPTS', {}), **sampler_kwargs}
        if method in SAMPLER_NAMES:                  # (SAMPLER OPTIMISE: the optimisation)
            okw = optimize_options(sampler_kwargs, {**dflt.get('RUNOPTS', {}), **run_kwargs})
            return self.optimize(freepar, threads=threads, log_like=log_like, like_kwargs=like_kwargs,
                                 verbose=kwargs.get('verbose', True), **okw)
        if 'dynamic' in sampler_kwargs:              # dynesty: static (default) or dynamic sampler
            kwargs.setdefault('dynamic', bool(sampler_kwargs.pop('dynamic')))
        if method == 'dynesty' and 'seed' in sampler_kwargs:     # (nautilus: its own option seed)
            seed = sampler_kwargs.pop('seed')
            sampler_kwargs.setdefault('rstate', np.random.default_rng(None if seed is None else int(seed)))
        run_kwargs = {**dflt.get('RUNOPTS', {}), **run_kwargs}
        sampler_kwargs, run_kwargs = sampler_options(method, kwargs.get('dynamic', False), sampler_kwargs,
                                                      run_kwargs)
        like_kwargs, threads, custom = self._start_fit(freepar, threads, log_like, like_kwargs, kwargs, 'run')
        verbose = kwargs.get('verbose', True)
        self._method = method
        self.optimum = None
        samples, weights, dt = sample(self, like_kwargs, method, threads, kwargs.get('dynamic', False), sampler_kwargs,
                                      run_kwargs, verbose)
        dt = f'{dt:.2f} seconds' if dt < 60.00 else f'{dt / 60.00:.2f} minutes'
        vprint(verbose, f'\nSampling with {method} took {dt} to run.')
        self._set_sampling_stats(samples, weights)
        if verbose and self.ess < 20:
            print(f"WARNING: the posterior is concentrated on {self.ess:.1f} samples (effective sample size), "
                  "so the\n         credible intervals are not reliable: the sampling stopped early "
                  "(maxiter, maxcall), or\n         the likelihood is extremely narrow (check that the noise "
                  "sigma is right, or make it\n         a free parameter).")
        return self._finish_fit(like_kwargs, custom, verbose)

    def optimize(self, freepar=None, method='L-BFGS-B', threads=None, *, log_like=None, like_kwargs: dict = {},
                 x0=None, posterior=False, errors=True, nstart=1, seed=None, step=1e-3, options: dict = {},
                 nsamples=4000, **kwargs):
        """ Find the best fit by a classic minimisation of -log L, with the likelihood of run() (the same free
        parameters, options and priors: see run), e.g. to test the sampling. Nothing is written (see write).

        The result is the maximum likelihood within the bounds of the priors (posterior=True: the maximum of the
        posterior, likelihood x prior), searched in the unit cube of the priors (see bbarolo.bayesian.optimize).
        Its errors (errors=True) are those of the Laplace approximation (the inverse Hessian of -log L at the best
        fit, about ndim^2 likelihoods), and nsamples draws of it serve as the samples of the posterior in the
        outputs (errors in the tables, corner plot, ring_errors, the errors of the deconvolved profile); errors=False:
        no errors (zero). The best fit is in params (not the median of the draws), the details in self.optimum (see
        bbarolo.bayesian.optimize.optimize).

        Args:
          freepar, threads, log_like, like_kwargs: as in run (threads: the processes of the likelihoods of a
                                gradient, of the Hessian and of a population)
          method (str):         'L-BFGS-B' (default: quasi-Newton, the gradient by central differences),
                                'TNC', 'SLSQP', 'trust-constr', 'Powell', 'Nelder-Mead', 'COBYLA', 'COBYQA'
                                (scipy.optimize.minimize, with the bounds of the priors) or
                                'differential_evolution' (global, then polished with L-BFGS-B)
          x0:                   the starting point: None (default: the initial rings, the noise for SIGMA, the
                                centre of the prior for FRACERR and the parameters of functions of radius),
                                'centre' (the centres of the priors) or a vector of the free parameters (NaN: the
                                centre of the prior)
          posterior (bool):     maximise the posterior instead of the likelihood (default False)
          errors (bool):        the errors from the Hessian (default True)
          nstart (int):         local methods: also start from nstart - 1 points drawn from the priors (seed), and
                                keep the best (default 1)
          step (float):         the step of the gradient, in units of the range of each prior (default 1e-3: the
                                likelihood of the discretised models is rough at much smaller steps)
          options (dict):       options of the method (scipy's, e.g. maxiter)
          kwargs:               verbose (bool) and any BBarolo parameter
        """
        from .optimize import optimize as _optimize, summary, method_name
        method = method_name(method)
        params = {k: v for k, v in kwargs.items() if k not in ('verbose',)}
        self._check_moved(kwargs, 'optimize')
        if params:
            self.update(**params)
        kwargs = {k: v for k, v in kwargs.items() if k == 'verbose'}
        like_kwargs, threads, custom = self._start_fit(freepar, threads, log_like, like_kwargs, kwargs, 'optimize')
        verbose = kwargs.get('verbose', True)
        self.results = self.sampler = None
        if isinstance(x0, str):
            if x0.strip().lower() not in ('centre', 'center'):
                raise BBaroloError(f"Bayesian3DFIT.optimize: x0={x0!r} (None, 'centre' or a vector of the free "
                                   "parameters)")
            theta0 = np.full(self.ndim, np.nan)
        else:
            theta0 = self._initial_theta() if x0 is None else np.asarray(x0, float)
        vprint(verbose, f"\nOptimising with {method} ({threads} process{'es' if threads > 1 else ''})...")
        res = _optimize(self, like_kwargs, theta0, method=method, threads=threads, posterior=posterior,
                        errors=errors, nstart=nstart, seed=seed, step=step, options=options, nsamples=nsamples,
                        verbose=verbose, showbar=show_bar(self))
        summary(res, verbose)
        self._set_optimum(res)
        return self._finish_fit(like_kwargs, custom, verbose)

    def _set_optimum(self, res):
        """ The results of an optimisation (optimize): optimum, the draws of its Laplace approximation as the
        samples (equal weights) and the best fit as params. """
        self.optimum = res
        self._method = 'optimize'
        self._set_sampling_stats(res.samples, np.ones(len(res.samples)))
        self.params = np.asarray(res.x, float).copy()

    def _initial_theta(self):
        """ The vector of the free parameters at the initial rings (with the density profile when DENS is not
        fitted): the value of each ring, or their median for NAME_SINGLE; RADMAX the estimated outer radius, SIGMA
        the noise (or the value given); NaN (the centre of the prior) for FRACERR and the parameters of the
        functions of radius. """
        th = np.full(self.ndim, np.nan)
        r = self._fit0.r
        for key, idx in self.freepar_idx.items():
            if key == 'sigma':
                th[idx] = self.like_kwargs.get('sigma') or self.noiserms
            elif key == 'radmax':
                th[idx] = self._radmax_guess if getattr(self, '_radmax_mode', 'edge') != 'ring' else r['radii'][-1]
            elif key in r and '_p' not in key:
                v = np.asarray(r[key], float)
                th[idx] = v if idx.size == v.size else np.median(v)
        return th

    @staticmethod
    def _check_moved(kwargs, caller):
        moved = [k for k in ('useBBres', 'noisecorr', 'likemask') if k in kwargs]
        if moved:
            raise TypeError(f"Bayesian3DFIT.{caller}: {', '.join(moved)} are options of like_kwargs, e.g. "
                            "like_kwargs=dict(likemask='none', noisecorr='beam', bbres=True).")

    def _start_fit(self, freepar, threads, log_like, like_kwargs, kwargs, caller):
        """ The start of a fit (run, optimize): the free parameters (freepar or FREE), the processes, the options of
        the likelihood (LIKEOPTS and like_kwargs), the setup, the log-likelihood (log_like or _log_likelihood) and,
        if verbose, the description of the likelihood and the priors. Returns the options of each call of the
        log-likelihood, the number of processes and whether the residuals are the user's. """
        dflt = self._run_defaults()
        if freepar is None:
            freepar = dflt.get('FREE')
            if freepar is None:
                from ..keywords import BAYES_DEFAULT_FREE
                freepar = self._keyword_option('FREE', BAYES_DEFAULT_FREE)
        elif isinstance(freepar, str):
            freepar = self._keyword_option('FREE', freepar)
        else:
            from ..params import bayes_name
            freepar = [bayes_name(p) for p in freepar]
        if len(freepar) == 0:                       # (it sampled zero dimensions)
            raise BBaroloError(f"Bayesian3DFIT.{caller}: no free parameters (freepar or FREE).")
        if threads is None:
            threads = int(self._params.get("THREADS", 1))
        # residuals replaced by the user (f._calculate_residuals = my_function)? a custom log-likelihood?
        custom = '_calculate_residuals' in self.__dict__
        own = log_like is not None and log_like != self._log_likelihood and log_like != self._log_likelihood_geometry
        self._set_like_options({**dflt.get('LIKEOPTS', {}), **like_kwargs}, custom=custom or own)
        like_kwargs = self._like_kwargs                      # (those of each call of the log-likelihood)
        useBBres = self.like_kwargs['bbres']
        self._setup(freepar, **kwargs)
        self._log_like_func = log_like if log_like is not None else self._log_likelihood
        verbose = kwargs.get('verbose', True)
        if verbose:
            import textwrap
            radii = " ".join(f"{r:.6g}" for r in np.asarray(self._inri.r['radii'], float))
            print("\n" + textwrap.fill(f"Rings: {self._inri.nr}, radii (arcsec): {radii}", width=110,
                                       subsequent_indent="  "))
            print(f"Fitting {self.ndim} parameters: {self.freepar_names}")
            if custom and not useBBres and log_like is None:
                print(f"\nLikelihood: user-defined residuals (_calculate_residuals); log-likelihood x "
                      f"{self._likefactor_note}")
            elif (not useBBres and log_like is None) or log_like == self._log_likelihood_geometry:
                sg = like_kwargs.get('sigma')
                sg = "free parameter" if 'sigma' in self.freepar_idx else \
                    (f"{sg:.4g} (given)" if sg is not None else f"{self.noiserms:.4g} (measured noise rms)")
                if log_like == self._log_likelihood_geometry:
                    fc = f"1/{1. / self.likefactor2d:.4g} (spatial correlation of the noise)"
                else:
                    fc = self._likefactor_note
                fe = like_kwargs.get('fracerr')
                fe = "; model error: fractional, free parameter 'fracerr'" if 'fracerr' in self.freepar_idx else \
                    (f"; model error: fractional, fracerr = {fe:.4g} (given)" if fe else "")
                if log_like == self._log_likelihood_geometry:
                    print(f"\nLikelihood: total intensity map ({self._likemask_note}), noise per voxel sigma = "
                          f"{sg}{fe}; log-likelihood x {fc}")
                else:
                    print(f"\nLikelihood: noise per voxel sigma = {sg}{fe}; log-likelihood x {fc}; "
                          f"{self.nvox} voxels ({self._likemask_note})")
                if getattr(self, '_corr_below_beam', False) and str(self.like_kwargs['noisecorr']).lower() == \
                        'measured' and log_like != self._log_likelihood_geometry:
                    print("WARNING: the sum of the autocorrelation of the noise is less than half the beam area: "
                          "the noise has\n         a negative ring (synthesis images without short spacings), or "
                          "the beam in the header\n         is wrong. The likelihood counts one independent value "
                          "per beam area at most.")
                if self._side() is not None:
                    print(f"           the {'approaching' if self._side() == 'A' else 'receding'} half of the "
                          "galaxy only (SIDE), divided by the minor axis of the initial rings")
            if self.useNorm and not useBBres:
                print(f"Density profile: deconvolved (fit of the total intensity map with the beam-convolved "
                      f"rings, smoothness {self.like_kwargs['densprof_smooth']:g})")
            if not useBBres and (log_like is None or log_like == self._log_likelihood_geometry):
                fb = self._flux_beyond_rings()
                if fb > 0.05:
                    print(f"WARNING: {fb:.0%} of the flux of the total intensity map lies beyond the outermost ring "
                          "(+ half a beam,\n         at the initial geometry): the model cannot reproduce it, and "
                          "with a wide prior\n         of INC it biases INC low. Extend the rings to the whole disk.")
            if 'radmax' in self.freepar_idx:
                print("\n" + dict(ring="RADMAX: the radius of the last ring (the ring moves)",
                           edge="RADMAX: the outer edge of the disk (sharp; the rings do not move)",
                           taper=f"RADMAX: the outer edge of the disk (Gaussian, width {self._radmax_width:.3g} "
                                 "arcsec; the rings do not move)")[self._radmax_mode])
            self.print_priors()
        self._check_likelihood(like_kwargs, custom, verbose)

        self._log_like_kind = ('geometry' if log_like == self._log_likelihood_geometry else
                               'cube' if log_like is None or log_like == self._log_likelihood else 'custom')
        return like_kwargs, threads, custom

    def _finish_fit(self, like_kwargs, custom, verbose):
        """ The end of a fit (run, optimize): the reduced chi-square, the best-fit rings (outri, with the density
        profile at the best-fit geometry if DENS is not fitted) and the errors of the deconvolved profile. """
        self.modCalculated = True
        if verbose and self._log_like_func == self._log_likelihood and not self.useBBres and not custom:
            try:
                chi2 = self.chi2_reduced()
                msg = f"Reduced chi-square of the best fit: {chi2:.2f}"
                if 'fracerr' in self.freepar_idx:
                    msg += f" ({self.chi2_reduced(fracerr=True):.2f} with the fitted model error)"
                print(msg)
                if chi2 > 2 and 'sigma' not in self.freepar_idx and 'fracerr' not in self.freepar_idx \
                        and not like_kwargs.get('fracerr'):
                    print("WARNING: the model does not describe the data down to the noise, so the credible "
                          "intervals\n         (computed with sigma = noise rms) underestimate the uncertainties. "
                          "Consider adding\n         'fracerr' (model error) or 'sigma' to the free parameters.")
            except Exception:
                pass

        # Best-fit rings (<prefix>_rings_final1.txt, with the errors from the posterior), with the
        # density profile at the best-fit geometry if DENS is not fitted
        self.outri = self._update_rings(self._fit0.copy(), self.params)
        if self.useNorm and not self.useBBres:
            self._set_density(self.outri)
        self._galfit.outr = self.outri.object.copy()
        if self.deconvolved_profile_errors() is not None and verbose:
            dp = self.dens_profile
            print(f"Deconvolved density profile: errors from {dp['nsamples']} posterior samples (geometry) and "
                  f"{dp['nmc']} noise realisations (see f.dens_profile, print_stats)")
        return self

    def _check_likelihood(self, like_kwargs, custom, verbose=True):
        """ Warnings before sampling, from the log-likelihood at the centre of the priors: a free
        'sigma' that the user-defined residuals do not use, and (if verbose) log-likelihoods
        below -1e6 (which dynesty prints as -inf, with nan errors). """
        try:
            theta = np.asarray(self._prior_transform(np.full(self.ndim, 0.5)), float)
            logl = self._log_like_func(theta, **like_kwargs)
            for key, what in (('sigma', 'the noise per voxel as\n         the keyword argument sigma (the free '
                                         'parameter, or the measured noise)'),
                              ('fracerr', 'the fractional model error as\n         the keyword argument fracerr')):
                if custom and key in self.freepar_idx and self._log_like_func == self._log_likelihood \
                        and not self.useBBres:
                    t2 = theta.copy()
                    t2[self.freepar_idx[key]] = self.prior_distr[key].ppf(0.9)
                    if self._log_like_func(t2, **like_kwargs) == logl:
                        print(f"WARNING: '{key}' is a free parameter, but the user-defined residuals do not depend "
                              f"on it.\n         _calculate_residuals(model, data, mask, **kwargs) receives {what}.\n")
            if verbose and np.isfinite(logl) and logl < -1e6:
                print(f"BBAROLO WARNING: the log-likelihood at the centre of the priors is {logl:.3g}. dynesty "
                      "prints\n                 values below -1e6 as -inf (and the error of logz as nan): they are "
                      "finite.\n                 Such values mean that the residuals are much larger than the "
                      "noise sigma:\n                 the posterior will be extremely narrow unless 'sigma' or "
                      "'fracerr' is free\n                 (and used).\n")
        except Exception:
            pass

    def run_geometry(self, freepar=('xpos_single', 'ypos_single', 'inc_single', 'phi_single'), **kwargs):
        """ Fit only the geometry of the model on the total intensity map (Cannubi-style; see
        _log_likelihood_geometry). On real galaxies:
          * add 'sigma' to the free parameters: the structure of the map that the model does not
            reproduce otherwise makes the posterior extremely narrow and the sampling slow
            (NGC 2403: 81 s on 16 processes with sigma, no convergence in 20 minutes without);
            'fracerr' is not suitable here (it weights the faint outer disk);
          * let the rings cover the whole disk (a warning tells when they do not): the emission
            beyond them biases INC low;
          * keep the prior of phi within one half: the map does not change with PA -> PA + 180
            (the default prior, the initial PA +- 15 deg, does; one 0-360 wide gives two modes).
        The data map sums the voxels of the likelihood, the model map all the channels (as in pyBBarolo):
        they agree when the likelihood voxels hold the whole line (with the default dilated mask the
        signal outside them is below 0.1%; with dilation (0, 0) about 20%). """
        from ..params import bayes_name
        allowed = {'xpos_single', 'ypos_single', 'inc_single', 'phi_single', 'z0_single', 'dens', 'dens_single',
                   'dens_func', 'radmax', 'sigma', 'fracerr'}
        freepar = [bayes_name(p) for p in (freepar.replace(',', ' ').split() if isinstance(freepar, str) else freepar)]
        if not (freepar and set(freepar).issubset(allowed)):
            raise ValueError("ERROR! The function run_geometry only accepts the following free parameters: "
                             + ", ".join(sorted(allowed)))
        return self.run(freepar=freepar, log_like=self._log_likelihood_geometry, **kwargs)

    def _set_sampling_stats(self, samples, weights):
        """ Statistics of the sampling results """
        self.samples = samples
        self.weights = weights / np.nansum(weights)
        # effective number of independent samples of the posterior
        self.ess = float(1. / np.nansum(self.weights ** 2))
        # equal-weight posterior samples (fixed seed, so that statistics are reproducible)
        if dyfunc is not None:
            self.samples_equal = dyfunc.resample_equal(self.samples, self.weights, rstate=np.random.default_rng(42))
        else:
            idx = np.random.default_rng(42).choice(len(samples), size=len(samples), p=self.weights)
            self.samples_equal = samples[idx]
        self.modCalculated = True
        self.quantiles = np.quantile(self.samples_equal, SIGMA_QUANTILES, axis=0)
        self.params = self.quantiles[1]

    def _update_rings(self, rings, theta):
        """ Update rings with the parameters theta """
        freepar_idx = copy.copy(self.freepar_idx)
        if 'radmax' in freepar_idx:
            if getattr(self, '_radmax_mode', 'edge') == 'ring':
                rings.r['radii'][-1] = theta[freepar_idx['radmax']][0]
            else:                                   # the outer edge of the disk (the rings do not move)
                rings.edge, rings.edgewidth = float(theta[freepar_idx['radmax']][0]), self._radmax_width
        whatFunctional = np.unique([key.split('_')[0] for key in freepar_idx if '_p' in key])
        for p in whatFunctional:
            keys = [key for key in freepar_idx if key.startswith(p + '_p')]
            fvalue = [theta[freepar_idx[key]] for key in keys]
            rings.modify_parameter(p, self.funcs[p](rings.r['radii'], *fvalue))
            for key in keys:
                freepar_idx.pop(key)
        for key in freepar_idx:
            if key in ('sigma', 'radmax', 'fracerr'):
                continue
            pvalue = theta[freepar_idx[key]]
            if len(pvalue) == 1:
                pvalue = pvalue[0]
            rings.modify_parameter(key, pvalue)
        rings.make_object()
        return rings
