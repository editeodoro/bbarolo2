"""
BayesianBBarolo: the fitter (parameters, rings, priors, free parameters, setup and sampling). The
likelihood, the density profile and the outputs are in the mixins of likelihood.py, density.py and
outputs.py; the samplers and the pool in sampling.py.
"""

import os
import copy
import numpy as np
from scipy import stats

from ..tasks import FitMod3D, build_param, _checked
from ..galmod import ringmode_name
from .common import vprint, get_distribution
from .rings import _RingDict
from .likelihood import LikelihoodMixin
from .density import DensityMixin
from .outputs import OutputsMixin
from .sampling import sample, sampler_options, dyfunc

class BayesianBBarolo(OutputsMixin, DensityMixin, LikelihoodMixin, FitMod3D):
    """ Fit a galaxy model to a 3D datacube in a Bayesian framework (nested sampling).

    BayesianBBarolo(cube, **params): the cube (FITS file, HDU or (data, header)) and any
    BBarolo parameter, including the rings (radii or nradii/radsep, xpos, ypos, vsys, vrot,
    vdisp, inc, pa, z0, vrad, dens, vvert, dvdz, zcyl: single values or one value per ring).
    Ring parameters not fitted are kept fixed. Geometric parameters that are not given are
    estimated as in 3DFIT (VDISP=8, Z0=0, VRAD=0, DENS=1 if not given).

    The parameters of the BAYESIAN3DFIT task of parameter files are accepted too: FREE (the
    free parameters, e.g. 'VROT VDISP_SINGLE INC_SINGLE'; run() needs them, as FREE or as
    freepar), SAMPLER, SAMPLEROPTS and RUNOPTS are the defaults of run(); PRIORS sets entries
    of priors (e.g. 'VROT=(uniform,loc=0,scale=300)'). BayesianBBarolo.from_parfile(file)
    reads a parameter file.

    Methods
    -------
    update(**params):        change BBarolo parameters (and the rings).
    run(freepar, method, threads, log_like, like_kwargs, sampler_kwargs, run_kwargs, verbose,
        dynamic, **params):  sample the posterior ('dynesty' or 'nautilus').
    run_geometry(freepar, **kwargs): fit only the geometry on the total intensity map.
    write_bestmodel(plots, norm, **kwargs): write the best model, maps, PVs and plots.
    corner_plot(...), plot_profiles(...), print_stats(...), print_priors(), chi2_reduced()
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
        self.noise_correlation = None
        self._likemask_key = None
        self.dens_profile = None
        self.useBBres = False
        self._set_like_options({})              # like_kwargs: the options of the likelihood (defaults)
        super(BayesianBBarolo, self).__init__(cube, **params)
        if self._cube is not None:
            self._init_rings()

    def update(self, beam=None, **params):
        """ Change BBarolo parameters (None restores the default). Changing the rings
        resets them (and the default priors that depend on them). FREE, SAMPLER, SAMPLEROPTS
        and RUNOPTS set the defaults of run(); PRIORS sets entries of self.priors (see
        bbarolo.params.bayes_priors). Returns self. """
        from ..keywords import lookup
        given = {}
        for k, v in params.items():              # checked before they are set
            kw = lookup(k)
            if kw is not None and kw.name in self._KEYWORD_OPTIONS and v is not None:
                given[kw.name] = self._keyword_option(kw.name, v)
        super(BayesianBBarolo, self).update(beam=beam, **params)
        if not hasattr(self, "priors"):              # called by the base constructor
            return self
        if "PRIORS" in given:
            self.priors.update(given["PRIORS"])
            if self.prior_distr is not None:
                for k in given["PRIORS"]:
                    self.prior_distr[k] = None       # frozen again by the next run
        if params or beam is not None:
            self._galfit = self._ellprof = None
            self._cube_obj = None
            if self._inri is not None:
                self._init_rings()
        return self

    def _keyword_option(self, name, value):
        """ The value of FREE (list of free parameters), SAMPLER ('dynesty' or 'nautilus'),
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
                if m not in ("dynesty", "nautilus"):
                    raise ValueError(f"unknown sampler {s.strip()} (DYNESTY or NAUTILUS)")
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
                    raise ValueError(f"BayesianBBarolo: unknown option of the likelihood '{k}' (options: "
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
        prm = super(BayesianBBarolo, self)._all_params(outfolder=outfolder, fitsfile=fitsfile)
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
        prm = self._all_params(outfolder=of, fitsfile=self._input_file(of))
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
        if not c.readCube(p.imageFile, printInfo=False):
            raise IOError(f"Cannot read {p.imageFile}")
        self._cube_obj, self._par = c, p
        return c

    def _init_rings(self):
        """ The initial rings from the parameters (estimating the missing geometry as in
        3DFIT), and the default priors. """
        from ..rings import readRings
        from ..paramguess import EstimateInitial
        from ..galfit import ringfile_values, default_z0
        c = self._cube_obj if self._cube_obj is not None else self._load_cube()
        par = copy.deepcopy(c.pars().gf)
        ringfile_values(par)                     # RINGFILE: the values not given
        toEstimate = ((par.RADII == "-1" and (par.NRADII == -1 or par.RADSEP == -1)) or par.XPOS == "-1"
                      or par.YPOS == "-1" or par.VSYS == "-1" or par.VROT == "-1" or par.PHI == "-1"
                      or par.INC == "-1")
        if toEstimate:
            c.checkBeam()
            if not c.MaskAll():
                c.BlankMask()
            ip = EstimateInitial(c, par)
            if par.RADII == "-1":
                if par.NRADII == -1: par.NRADII = ip.nrings
                if par.RADSEP == -1: par.RADSEP = ip.radsep
            if par.XPOS == "-1": par.XPOS = repr(float(ip.xcentre))
            if par.YPOS == "-1": par.YPOS = repr(float(ip.ycentre))
            if par.VSYS == "-1": par.VSYS = repr(float(ip.vsystem))
            if par.VROT == "-1": par.VROT = repr(float(ip.vrot))
            if par.INC == "-1": par.INC = repr(float(ip.inclin))
            if par.PHI == "-1": par.PHI = repr(float(ip.posang))
        for k, v in (("VDISP", "8.0"), ("Z0", repr(default_z0(c.Head()))), ("VRAD", "0.0"), ("DENS", "1.0"),
                     ("VVERT", "0.0"), ("DVDZ", "0.0"), ("ZCYL", "0.0")):
            if getattr(par, k) == "-1":
                setattr(par, k, v)
        inR = readRings(par, c.Head())
        if inR.nr == 0:
            raise ValueError("BAYESIAN3DFIT ERROR: the number of radii must be > 0.")
        if toEstimate:
            print(f"\n{self.taskname}: initial rings ({inR.nr} rings every {inR.radsep:.2f} arcsec) with "
                  "estimated parameters:")
            for k, n in (("xpos", "XPOS"), ("ypos", "YPOS"), ("vsys", "VSYS"), ("vrot", "VROT"), ("inc", "INC"),
                         ("phi", "PA")):
                print(f"  {n:5s} = {np.median(getattr(inR, k)):.3f}")
        self._inri = _RingDict.from_rings(inR)
        self._set_default_priors()

    def _set_default_priors(self):
        """ Default priors (uniform), parametric functions and smoothness penalties. The
        priors that depend on the rings follow them, unless they were changed. """
        rr = {k: np.asarray(v, dtype=float) for k, v in self._inri.r.items()}
        radsep = float(np.abs(rr['radii'][-1] - rr['radii'][-2])) if len(rr['radii']) > 1 else float(rr['radii'][0])
        new = dict(
            xpos=dict(name='uniform', loc=rr['xpos'][0] - 5, scale=10),       # +- 5 pixels
            ypos=dict(name='uniform', loc=rr['ypos'][0] - 5, scale=10),       # +- 5 pixels
            vsys=dict(name='uniform', loc=rr['vsys'][0] - 20, scale=40),      # +- 20 km/s
            vrot=dict(name='uniform', loc=0, scale=350),                      # 0-350 km/s
            vdisp=dict(name='uniform', loc=0, scale=50),                      # 0-50 km/s
            vrad=dict(name='uniform', loc=0, scale=50),                       # 0-50 km/s
            inc=dict(name='uniform', loc=0, scale=90),                        # 0-90 degrees
            phi=dict(name='uniform', loc=0, scale=360),                       # 0-360 degrees
            vvert=dict(name='uniform', loc=0, scale=30),                      # 0-30 km/s
            dvdz=dict(name='uniform', loc=0, scale=2),                        # 0-2 km/s/pc
            zcyl=dict(name='uniform', loc=0, scale=5),                        # 0-5 arcsec
            dens=dict(name='uniform', loc=0.1, scale=200),                    # 0.1-200 1E20 cm^-2
            z0=dict(name='uniform', loc=0, scale=10),                         # 0-10 arcsec
            sigma=dict(name='loguniform', a=None, b=None),                    # None: 0.5 and 1000 x noise rms
            fracerr=dict(name='loguniform', a=1e-3, b=1.),                    # fractional model error
            radmax=dict(name='uniform', loc=float(rr['radii'][-1]) - radsep / 2., scale=radsep),
            vrot_p1=dict(name='uniform', loc=1, scale=350),
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
        self.funcs.setdefault('vrot', lambda R, p1, p2: 2. / np.pi * p1 * np.arctan(R / p2))   # arctan curve
        self.funcs.setdefault('vdisp', lambda R, p1, p2: p1 + p2 * R)                          # linear
        self.funcs.setdefault('dens', lambda R, p1, p2: p1 * np.exp(-R / p2))                  # exponential

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
          2. the free parameters (_parse_freepar): ndim, freepar_idx, freepar_names;
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
        RADMAX of the original module). The default prior is the last ring, its radius +- half the
        ring separation; with NODES rings, whose disk ends at the last radius, the edge lies within
        half a ring separation inside it (see _prior_kwargs). """
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
        (e.g. halfcauchy, the default before November 2026). The default prior of an edge (radmax
        'edge' or 'taper') with NODES rings is the half of the last ring inside its radius. """
        kw = {k: v for k, v in self.priors[key].items() if k != "name"}
        if key == 'radmax' and self.priors[key] == getattr(self, '_default_pri', {}).get(key) \
                and self._galfit is not None and getattr(self, '_radmax_mode', 'edge') != 'ring' \
                and ringmode_name(self._galfit.model.ringmode) == 'NODES':
            kw['scale'] = 0.5 * kw['scale']     # (the disk of NODES rings ends at the last radius)
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
          freepar (list):       the free parameters (required, here or as the parameter FREE), e.g.
                                ['vrot','vdisp_single','inc_single','sigma'], or a string as FREE in
                                a parameter file ('VROT VDISP_SINGLE INC_SINGLE SIGMA'). Names are
                                case-insensitive (PA is phi, DISP is vdisp)
          method (str):         'dynesty' or 'nautilus' (default: SAMPLER if given, otherwise 'dynesty')
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
        moved = [k for k in ('useBBres', 'noisecorr', 'likemask') if k in kwargs]
        if moved:
            raise TypeError(f"BayesianBBarolo.run: {', '.join(moved)} are options of like_kwargs, e.g. "
                            "like_kwargs=dict(likemask='none', noisecorr='beam', bbres=True).")
        params = {k: v for k, v in kwargs.items() if k not in ('verbose', 'dynamic')}
        if params:
            self.update(**params)
        kwargs = {k: v for k, v in kwargs.items() if k in ('verbose', 'dynamic')}
        dflt = self._run_defaults()
        if freepar is None:
            freepar = dflt.get('FREE')
            if freepar is None:
                raise ValueError("BayesianBBarolo.run: give the free parameters, as run(freepar=[...]) or as the "
                                 "parameter FREE (e.g. free='VROT VDISP_SINGLE INC_SINGLE').")
        elif isinstance(freepar, str):
            freepar = self._keyword_option('FREE', freepar)
        else:
            from ..params import bayes_name
            freepar = [bayes_name(p) for p in freepar]
        method = str(dflt.get('SAMPLER', 'dynesty') if method is None else method).lower()
        self._method = method
        sampler_kwargs = {**dflt.get('SAMPLEROPTS', {}), **sampler_kwargs}
        if 'dynamic' in sampler_kwargs:              # dynesty: static (default) or dynamic sampler
            kwargs.setdefault('dynamic', bool(sampler_kwargs.pop('dynamic')))
        if method == 'dynesty' and 'seed' in sampler_kwargs:     # (nautilus: its own option seed)
            seed = sampler_kwargs.pop('seed')
            sampler_kwargs.setdefault('rstate', np.random.default_rng(None if seed is None else int(seed)))
        run_kwargs = {**dflt.get('RUNOPTS', {}), **run_kwargs}
        sampler_kwargs, run_kwargs = sampler_options(method, kwargs.get('dynamic', False), sampler_kwargs,
                                                      run_kwargs)
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
            print(f"\nFitting {self.ndim} parameters: {self.freepar_names}")
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
        samples, weights, dt = sample(self, like_kwargs, method, threads, kwargs.get('dynamic', False), sampler_kwargs,
                                      run_kwargs, verbose)
        dt = f'{dt:.2f} seconds' if dt < 60.00 else f'{dt / 60.00:.2f} minutes'
        vprint(verbose, f'\nSampling with {method} took {dt} to run.')

        self.modCalculated = True
        self._set_sampling_stats(samples, weights)
        if verbose and self.ess < 20:
            print(f"WARNING: the posterior is concentrated on {self.ess:.1f} samples (effective sample size), "
                  "so the\n         credible intervals are not reliable: the sampling stopped early "
                  "(maxiter, maxcall), or\n         the likelihood is extremely narrow (check that the noise "
                  "sigma is right, or make it\n         a free parameter).")
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
        with self._errors_in_outputs():
            pass
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
          * restrict the prior of phi to one half: the map does not change with PA -> PA + 180
            (the default prior, 0-360, gives two modes). """
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
        self.quantiles = np.quantile(self.samples_equal, [0.16, 0.50, 0.84], axis=0)
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
