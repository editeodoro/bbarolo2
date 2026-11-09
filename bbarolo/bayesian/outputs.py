"""The outputs of Bayesian3DFIT: the best model and its files, the errors of the rings, the plots, the
statistics, and the results saved and loaded."""

import os
import contextlib
import numpy as np

from ..tasks import FitMod3D, _FromFile
from .rings import PARAMS, RING_PARAMS, isIterable
from .common import vprint, SIGMA_QUANTILES


class OutputsMixin(object):
    """ The outputs of Bayesian3DFIT. """

    def _normalize_model(self, model, data, mask, norm='AZIM'):
        """ Normalization of the best model (called but not defined in the original module).

        'AZIM' (default): no change, i.e. the azimuthally-normalized model used in the
        likelihood; 'LOCAL': pixel-by-pixel normalization to the flux of the masked data.
        """
        if str(norm).upper() != 'LOCAL':
            return model
        modSum = np.nansum(model, axis=0)
        obsSum = np.nansum(data * mask, axis=0)
        with np.errstate(invalid='ignore', divide='ignore'):
            factor = np.where(modSum != 0, obsSum / modSum, 0.)
        return model * factor[None]

    def write(self, outfolder=None, plots=None, verbose=True):
        """ Write the outputs of the fit in outfolder (default: OUTFOLDER, or ./output/<OBJECT>/), as the
        BAYESIAN3DFIT task does: with plots >= 0 (default PLOTS) the best model and its outputs
        (write_bestmodel: model, maps, PV diagrams, <prefix>_rings_final1.txt with the errors,
        <prefix>_densprof.txt, ...; plots > 0: the plots of 3DFIT), otherwise only the rings; the samples
        (<prefix>_<sampler>_results.npy, see load_results); with plots > 0 also the corner plot
        (<prefix>_corner.pdf) and the radial profiles (<prefix>_profiles.pdf). Returns the files written. """
        from ..cube import write_outputs
        from ..tasks import _snapshot, _written, _input_copy
        from ..bbutils import shortpath
        if not self.modCalculated:
            raise RuntimeError("Bayesian3DFIT.write: no results to write (run() first).")
        c = self._cube_obj
        p = c.pars()
        if outfolder is not None:
            p.setOutfolder(os.path.abspath(os.path.expanduser(str(outfolder))) + "/")
        of, pre = p.outFolder, p.outPrefix
        plots = p.plots if plots is None else int(plots)
        os.makedirs(of, exist_ok=True)
        before = _snapshot(of)
        old_image = p.imageFile
        try:
            if plots > 0:
                p.imageFile = _input_copy(self, c, of)          # (the plots read the data from a file)
            write_outputs(c, c.extra_outputs)                   # (e.g. the 2D fit of the initial estimates)
            if plots >= 0:                                      # (as 3DFIT: PLOTS=-1 writes no model)
                c.writeMask(force=True)
                self.write_bestmodel(plots=plots > 0, verbose=verbose)
            else:
                with self._errors_in_outputs():
                    pass
            fn = os.path.join(of, f"{pre}_{self._method}_results.npy")
            self.save_results(fn)
            vprint(verbose, f"Posterior samples written in {shortpath(fn)} (see Bayesian3DFIT.load_results)")
            if plots > 0:
                self._save_plots(of, pre, verbose)
        finally:
            p.imageFile = old_image
        return _written(of, before)

    def _save_plots(self, of, pre, verbose):
        """ The corner plot and the radial profiles, as PDF files. """
        import logging
        import warnings
        import matplotlib.pyplot as plt
        from ..bbutils import shortpath
        for name, make in (("corner", lambda: self.corner_plot(plot_dynesty=(self._method == 'dynesty'))),
                           ("profiles", self.plot_profiles)):
            out = os.path.join(of, f"{pre}_{name}.pdf")
            root = logging.getLogger()
            level = root.level
            try:
                # (no warnings of the plotting packages, e.g. for a posterior on very few samples)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    root.setLevel(logging.ERROR)
                    fig = make()
                    fig.savefig(out, bbox_inches='tight')
                plt.close(fig)
                vprint(verbose, f"Plot written in {shortpath(out)}")
            except Exception as e:
                plt.close('all')
                print(f"BAYESIAN3DFIT WARNING: no {name} plot ({e})")
            finally:
                root.setLevel(level)

    def write_bestmodel(self, plots=True, norm='AZIM', **kwargs):
        """ Write the best model, its moment maps, PV diagrams and plots in the output folder.

        norm: 'AZIM' (model as used in the likelihood) or 'LOCAL' (pixel-by-pixel normalized),
        used only when the density is not fitted.

        The DENS column of <prefix>_rings_final1.txt is the density profile of the model (deconvolved,
        at the best-fit geometry, if DENS is not fitted), and <prefix>_densprof.txt the observed profile
        (the average of the total intensity map in each ring). If DENS is not fitted,
        <prefix>_densprof_deconvolved.txt lists both profiles at full precision, with the errors.
        """
        if not self.modCalculated:
            print("Sampler has not been run yet. Please run run() before running this function.")
            return
        verbose = kwargs.get('verbose', True)
        os.makedirs(self._cube_obj.pars().outFolder, exist_ok=True)
        vprint(verbose, "\nWriting the best model to output directory ...", end=' ', flush=True)
        if self.useBBres:
            with self._errors_in_outputs():
                self._galfit.writeModel("AZIM", plots)
        else:
            if self.useNorm:
                self._set_density(self.outri)
            ell = self._observed_profile(self.outri)        # <prefix>_densprof.txt (outputs only)
            model = self._calculate_model(self.outri, fullcube=True)[0]
            if model.ndim == 4:             # (non-convolved, convolved) models
                model = model[-1]
            if self.useNorm:
                model = self._normalize_model(model, self.data, self.mask, norm)
            from ..cube import Cube
            c = self._cube_obj
            mod = Cube.fromArray(np.asarray(model, dtype=np.float32), c.Head(), c.pars())
            with self._errors_in_outputs():
                self._galfit._model_dens_profile = self._model_dens_profile() if plots else None
                try:
                    self._galfit.writeOutputs(mod, ell, plots)
                finally:
                    self._galfit._model_dens_profile = None
            if self.useNorm:
                self._write_deconvolved_profile()
        vprint(verbose, "Done!")

    def _write_deconvolved_profile(self):
        """ <prefix>_densprof_deconvolved.txt in the output folder: radius, the density profile of the
        model (deconvolved, at the best-fit geometry) with its 1-sigma errors (see
        deconvolved_profile_errors), the observed profile (face-on average of the total
        intensity map in each ring, the profile of the original module), and the errors from
        the geometry and from the noise, in the units of DENS. Needs the Ellprof profile along
        self.outri (see write_bestmodel). """
        p_ = self._cube_obj.pars()
        fn = os.path.join(p_.outFolder, p_.outPrefix + "_densprof_deconvolved.txt")
        r = self.outri.r
        nr = self.outri.nr
        dp = getattr(self, 'dens_profile', None)
        z = np.zeros(nr)
        e = {k: (dp[k] if dp is not None else z) for k in ('err_lo', 'err_hi', 'err_geom_lo', 'err_geom_hi',
                                                            'err_noise_lo', 'err_noise_hi')}
        with open(fn, "w") as f:
            f.write("# Density profile of the model (DENS_MODEL: the densities whose beam-convolved model\n"
                    "# reproduces the total intensity map) and the observed profile (DENS_OBS: the face-on\n"
                    "# average of the map in each ring, the profile of the original module).\n"
                    "# E_LO, E_HI: 1-sigma errors of DENS_MODEL, the errors from the geometry (posterior samples:\n"
                    f"# {dp['nsamples'] if dp else 0}) and from the noise (realisations: {dp['nmc'] if dp else 0}) "
                    "added in quadrature.\n")
            f.write(f"{'#RAD(arcs)':<12s}" + "".join(f"{c:>14s}" for c in (
                'DENS_MODEL', 'E_LO', 'E_HI', 'DENS_OBS', 'E_GEOM_LO', 'E_GEOM_HI', 'E_NOISE_LO', 'E_NOISE_HI')) + "\n")
            for i in range(nr):
                vals = (r['dens'][i], e['err_lo'][i], e['err_hi'][i], abs(self._ellprof.getSurfDensFaceOn(i)),
                        e['err_geom_lo'][i], e['err_geom_hi'][i], e['err_noise_lo'][i], e['err_noise_hi'][i])
                f.write(f"{r['radii'][i]:<12.3f}" + "".join(f"{v:>14.6e}" for v in vals) + "\n")

    def _model_dens_profile(self):
        """ The density profile of the best model as a face-on surface density (map units per arcsec^2, as the
        observed one, SDENS_FO of <prefix>_densprof.txt), for the plot of the parameters: a dict with radii, sdens,
        lo, hi (1-sigma errors) and label, 'deconvolved' (DENS not fitted: the deconvolved profile, with the
        errors of deconvolved_profile_errors) or 'fitted' (the posterior errors, ring_errors). """
        from ..profile import sdens_unit
        r = self.outri.r
        unit = sdens_unit(self._cube_obj, self._galfit.model, float(np.median(r['vsys'])))
        nr = self.outri.nr
        if self.useNorm:
            dp = getattr(self, 'dens_profile', None)
            lo, hi = (dp['err_lo'], dp['err_hi']) if dp is not None else (np.zeros(nr), np.zeros(nr))
            label = 'deconvolved'
        else:
            lo, hi = self.ring_errors('dens')
            label = 'fitted'
        e = lambda v: np.clip(np.nan_to_num(np.asarray(v, float)), 0, None) * unit
        return dict(radii=np.asarray(r['radii'], float), sdens=np.asarray(r['dens'], float) * unit, lo=e(lo),
                    hi=e(hi), label=label)

    # Ring parameters in the order of the 3DFIT tables and error columns
    _RING_NAMES = ('vrot', 'vdisp', 'dens', 'z0', 'inc', 'phi', 'xpos', 'ypos', 'vsys', 'vrad')

    def _posterior_errors(self):
        """ The ring parameters that were sampled (flags in the order of _RING_NAMES) and
        their 1-sigma errors from the posterior, as the 3DFIT error columns:
        errors[ring, 0 = lower (negative) / 1 = upper (positive), k-th free parameter]. """
        base = {k.split('_')[0] for k in self.freepar_idx}
        if getattr(self, 'dens_profile', None) is not None:
            base.add('dens')                          # errors of the deconvolved profile
        mpar = [n in base for n in self._RING_NAMES]
        free = [n for n, m in zip(self._RING_NAMES, mpar) if m]
        errors = np.zeros((self.outri.nr, 2, len(free)))
        for k, n in enumerate(free):
            lo, hi = self.ring_errors(n)
            errors[:, 0, k] = -np.clip(np.nan_to_num(np.asarray(lo, float)), 0, None)
            errors[:, 1, k] = np.clip(np.nan_to_num(np.asarray(hi, float)), 0, None)
        return mpar, errors

    @property
    def disk_edge(self):
        """ The outer edge of the disk (arcsec) of the best fit: RADMAX, when it is free and is the edge of
        the disk (like_kwargs radmax='edge' or 'taper'); None otherwise. """
        if 'radmax' not in getattr(self, 'freepar_idx', {}) or getattr(self, '_radmax_mode', 'edge') == 'ring':
            return None
        e = getattr(getattr(self, 'outri', None), 'object', None)
        e = getattr(e, 'edge', None) if e is not None else None
        return None if e is None else float(e)

    @property
    def in_disk(self):
        """ For each ring of the best fit (outri): True if its radius is within the outer edge of the disk
        (disk_edge, RADMAX), False beyond it. The model has no gas beyond the edge: the values of a ring
        beyond it are those, at its radius, of the profiles interpolated between the rings, i.e. an
        extrapolation of the profiles of the gas inside the edge (by half a ring or more, the ring only
        sets their slope inside the edge). All True without an edge. """
        r = np.asarray(self.outri.r['radii'], float)
        edge = self.disk_edge
        return np.ones(r.size, bool) if edge is None else r <= edge

    @contextlib.contextmanager
    def _errors_in_outputs(self):
        """ Writes <prefix>_rings_final1.txt with the errors from the posterior (and, with RADMAX as the edge
        of the disk, the column INDISK: 1 for the rings within the edge, 0 beyond it, see in_disk). Inside
        the context, the free parameters of the sampling (instead of FREE) and the error columns are
        used by the outputs of the 3DFIT engine, so that the plots (_parameters.pdf) show the errors
        (and the region beyond the edge). """
        gf = self._galfit
        old = (gf.mpar, gf.nfree, gf.par.flagERRORS, getattr(gf, '_disk_edge', None))
        mpar, errors = self._posterior_errors()
        gf.mpar, gf.nfree, gf.par.flagERRORS = mpar, int(sum(mpar)), bool(any(mpar))
        gf._disk_edge = self.disk_edge
        extra = {} if self.disk_edge is None else {'INDISK': self.in_disk.astype(int)}
        try:
            gf.writeRingFile("rings_final1.txt", self.outri.object, errors, extra=extra)
            yield
        finally:
            gf.mpar, gf.nfree, gf.par.flagERRORS, gf._disk_edge = old

    def corner_plot(self, samples=None, labels=None, truths=None, quantiles=SIGMA_QUANTILES,
                    plot_dynesty=False, **kwargs):
        """ Corner plot of the posterior distributions (needs 'corner', or dynesty's plotting) """
        if samples is None and not self.modCalculated:
            print("Sampler has not been run yet. Please run run() before running this function "
                  "or load samples with load_results().")
            return
        defaults = dict(max_n_ticks=5, color='darkblue', show_titles=True)
        if self.results is None:
            plot_dynesty = False
        if plot_dynesty:
            from dynesty import plotting as dyplot
            samples = self.results
            cornerplot = dyplot.cornerplot
        else:
            try:
                import corner
            except ImportError:
                raise ImportError("BayesianBB requires the package 'corner' to produce corner plots.")
            cornerplot = corner.corner
            defaults['plot_datapoints'] = False
        if samples is None:
            samples = self.samples_equal
        if labels is None:
            labels = self.freepar_names
        if truths is not None:
            kwargs['truths'] = truths
        c = cornerplot(samples, title_quantiles=quantiles, quantiles=quantiles, labels=labels,
                       label_kwargs=dict(fontsize=20), **{**defaults, **kwargs})
        return c[0] if plot_dynesty else c

    _ERR_ALIASES = dict(pa='phi', disp='vdisp')

    def ring_errors(self, param=None):
        """ Lower and upper 1-sigma errors of a ring parameter in each ring of the best fit
        (self.outri), as two arrays of positive values (lower, upper):

          sampled per ring or '_single':  the 16th and 84th percentiles of the posterior;
          '_func':                        the percentiles of the function of radius over the
                                          posterior samples;
          DENS not fitted:                the errors of the deconvolved
                                          profile (geometry and noise, see
                                          deconvolved_profile_errors);
          otherwise (fixed):              zeros.

        param: vrot, vdisp (or disp), dens, z0, inc, phi (or pa), xpos, ypos, vsys, vrad,
        vvert, dvdz, zcyl, case-insensitive. Without param: a dictionary with the errors of
        vrot, vdisp, dens, z0, inc, phi, xpos, ypos, vsys and vrad. These are the errors of
        <prefix>_rings_final1.txt and of plot_profiles. Needs run() or load_results(). """
        if not self.modCalculated or self.outri is None:
            raise ValueError("ring_errors: run the sampling (run) or load the results (load_results) first.")
        if param is None:
            return {n: self.ring_errors(n) for n in self._RING_NAMES}
        a = str(param).strip().lower()
        a = self._ERR_ALIASES.get(a, a)
        if a not in PARAMS or a == 'radii':
            raise ValueError(f"ring_errors: unknown ring parameter {param!r}.")
        nr = self.outri.nr
        if a in self.freepar_idx:
            q = self.quantiles[:, self.freepar_idx[a]]
            lo, hi = q[1] - q[0], q[2] - q[1]
        else:
            keys = [k for k in self.freepar_idx if k.startswith(a + '_p')]
            if keys:
                R = self.outri.r['radii']
                prof = np.array([self.funcs[a](R, *[s[self.freepar_idx[k]] for k in keys])
                                 for s in self.samples_equal])
                q = np.quantile(prof, SIGMA_QUANTILES, axis=0)
                lo, hi = self.outri.r[a] - q[0], q[2] - self.outri.r[a]
            elif a == 'dens' and getattr(self, 'dens_profile', None) is not None:
                lo, hi = self.dens_profile['err_lo'], self.dens_profile['err_hi']      # deconvolved profile
            else:
                lo = hi = 0.
        lo = np.clip(np.broadcast_to(np.asarray(lo, float), (nr,)), 0., None)
        hi = np.clip(np.broadcast_to(np.asarray(hi, float), (nr,)), 0., None)
        return lo, hi

    def plot_profiles(self, **kwargs):
        """ Plot the best-fit radial profiles of the ring parameters (with 1-sigma errors) """
        import matplotlib.pyplot as plt
        if not self.modCalculated:
            print("Sampler has not been run yet. Please run run() before running this function "
                  "or load samples with load_results().")
            return
        rings = self.outri.r
        toplot = ['vrot', 'vdisp', 'dens', 'inc', 'phi', 'z0', 'xpos', 'ypos', 'vsys']
        labels = [r'$V_\mathrm{rot}$ (km/s)', r'$\sigma$ (km/s)', 'Density', 'Inclination (deg)',
                  'Position angle (deg)', r'$z_0$ (arcsec)', 'Xpos (pix)', 'Ypos (pix)', r'$V_\mathrm{sys}$ (km/s)']
        fig, axes = plt.subplots(3, 3, figsize=(12, 9), sharex=True,
                                 gridspec_kw=dict(height_ratios=[1.6, 1, 1], hspace=0.05, wspace=0.3))
        for k, a in enumerate(axes.flat):
            a.tick_params(direction='in', top=True, right=True)
            if k >= 6:
                a.set_xlabel('Radius (arcsec)', fontsize=14)
        axes[0, 0].set_xlim(0, rings['radii'][-1] * 1.05)
        ax = axes.ravel()
        for i, a in enumerate(toplot):
            err_low, err_up = self.ring_errors(a)
            ax[i].errorbar(rings['radii'], rings[a], yerr=[np.asarray(err_low), np.asarray(err_up)], fmt='o',
                           **{'c': '#B22222', 'ecolor': '#B22222', **kwargs})
            ax[i].set_ylabel(labels[i])
            lo, hi = np.nanmin(rings[a] - err_low), np.nanmax(rings[a] + err_up)
            if hi > lo:
                pad = 0.1 * (hi - lo)
                ax[i].set_ylim(lo - pad, hi + pad)
            if a == 'vrot':
                ax[i].set_ylim(0, None)
        return fig

    def print_stats(self, quantiles=SIGMA_QUANTILES):
        """ Print the median and 1-sigma errors of the posteriors (and the maximum likelihood) """
        if not self.modCalculated:
            print("Sampler has not been run yet. Please run run() before running this function.")
            return
        if not isIterable(quantiles) or len(quantiles) != 3:
            raise ValueError("ERROR! Percentiles must be a list of three values.")
        results = self.results
        lk = getattr(self, '_like_kwargs', {})          # the likelihood options of run()
        llf = getattr(self, '_log_like_func', self._log_likelihood)
        if llf is not None:
            print(f"\nBest-fit parameters (log_likelihood = {llf(self.params, **lk):.2f}):")
        else:
            print("\nBest-fit parameters (user-defined log-likelihood, not saved with the results):")
        if getattr(self, 'ess', np.inf) < 20:
            print(f"  (effective sample size of the posterior: {self.ess:.1f}: the errors are not reliable)")
        q = np.quantile(self.samples_equal, quantiles, axis=0)
        for i in range(len(self.freepar_names)):
            d = np.diff(q[:, i])
            print("  %10s = %10.3f %+10.3f %+10.3f" % (self.freepar_names[i], q[1, i], -d[0], d[1]))
        dp = getattr(self, 'dens_profile', None)
        if dp is not None:
            print("\nDeconvolved density profile (DENS, not fitted; errors from the geometry and the noise):")
            for i in range(len(dp['radii'])):
                print("  R = %8.2f   %10.4g %+10.3g %+10.3g" % (dp['radii'][i], dp['dens'][i], -dp['err_lo'][i],
                                                                dp['err_hi'][i]))
        edge = self.disk_edge
        out = np.flatnonzero(~self.in_disk)
        if edge is not None and out.size:
            import textwrap
            r = np.asarray(self.outri.r['radii'], float)
            rings = ", ".join(f"{i + 1} (R = {r[i]:g})" for i in out)
            many = out.size > 1
            msg = (f"RADMAX = {edge:.4g} arcsec is the outer edge of the disk: beyond it the model has no gas. "
                   f"Ring{'s' if many else ''} {rings} lie{'' if many else 's'} beyond it: "
                   f"{'their' if many else 'its'} values are extrapolations of the profiles of the gas inside the "
                   "edge, not measurements (INDISK = 0 in rings_final1.txt; see in_disk).")
            print("\n" + textwrap.fill(msg, 100))
        if results is not None:
            max_index = results.logl.argmax()
            print(f"\nMaximum logl: {results.logl.max()} at:")
            for i in range(len(self.freepar_names)):
                print("  %10s = %10.3f " % (self.freepar_names[i], results.samples[max_index][i]))

    def print_priors(self):
        """ Print the priors for each parameter """
        print("\nPrior distributions for free parameters:")
        for key in self.freepar_idx:
            print(f"{key:>10s} = {self.prior_distr[key].dist.name:10s}", self.prior_distr[key].kwds)
        print("\n")

    def save_results(self, outfile):
        """ Save the results as a dictionary in a numpy (.npy) file (with the options of the
        likelihood, like_kwargs; those that cannot be saved, e.g. functions, are left out). """
        import pickle
        like = {}
        for k, v in self.like_kwargs.items():
            try:
                pickle.dumps(v)
                like[k] = v
            except Exception:
                pass
        tosave = dict(results=self.results, samples=self.samples, weights=self.weights,
                      freepar_inp=self._freepar_inp, freepar_names=self.freepar_names, freepar_idx=self.freepar_idx,
                      inrings=self._inri.r, options={k: str(v) if isinstance(v, _FromFile) else v
                                                     for k, v in self.parameters.items()},
                      priors=self.priors,
                      prior_distr=self.prior_distr, lambdas=self.lambdas, like_kwargs=like,
                      loglike=getattr(self, '_log_like_kind', 'cube'), method=self._method,
                      funcs=self._funcs_used())
        np.save(outfile, tosave)

    def _funcs_used(self):
        """ The functions of radius of the free parameters NAME_FUNC: 'default', or the name of the function
        given in funcs (functions are not saved: load_results needs the same ones in funcs). """
        from .fitter import DEFAULT_FUNCS
        used = {p.split('_')[0] for p in (self._freepar_inp or []) if p.endswith('_func')}
        return {k: ('default' if f is DEFAULT_FUNCS.get(k) else getattr(f, '__qualname__', repr(f)))
                for k, f in self.funcs.items() if k in used}

    def load_results(self, inputfile, sampler=None):
        """ Load results saved with save_results() (dynesty or nautilus: the sampler of the fit, or 'dynesty' for
        files that do not record it): the samples, the priors, the rings and the parameters of the fit (BEAM
        included). The output folder is that of the fit unless this object has one (outfolder given). A fit with functions of radius of its own (funcs) needs the same functions in funcs
        (functions are not saved): BBaroloError otherwise. """
        from ..bbutils import BBaroloError
        from .fitter import DEFAULT_FUNCS
        r = np.load(inputfile, allow_pickle=True).item()
        if sampler is None:
            sampler = r.get('method') or 'dynesty'
        for k, name in dict(r.get('funcs') or {}).items():
            if name != 'default' and self.funcs.get(k, DEFAULT_FUNCS.get(k)) is DEFAULT_FUNCS.get(k):
                raise BBaroloError(f"load_results: the fit used its own function of radius for {k.upper()}_FUNC "
                                   f"({name}): set it in funcs['{k}'] before loading (functions are not saved).")
        rr = r['inrings']
        self._galfit = self._ellprof = None
        # options of the saved fit (also files written by the original module: lower-case keys)
        from ..keywords import lookup
        opts = {k: v for k, v in dict(r.get('options') or {}).items()
                if lookup(k) is not None and lookup(k).name not in ('FITSFILE', 'FITSLIST', '3DFIT')}
        rings = {RING_PARAMS[k]: np.asarray(rr[k], dtype=float) for k in PARAMS}
        rings['NRADII'] = None
        rings['RADSEP'] = None
        opts = {lookup(k).name: v for k, v in opts.items()}
        if self._params.get("OUTFOLDER") not in (None, ""):
            opts.pop("OUTFOLDER", None)        # (the output folder given to this object, not that of the fit)
        opts.update(rings)
        FitMod3D.update(self, **opts)     # (not that of Bayesian3DFIT)
        self._cube_obj = None
        self._init_rings()
        for k in ('priors', 'prior_distr', 'lambdas'):
            if r.get(k) is not None:
                getattr(self, k).update(r[k])
        # the options of the likelihood of the saved fit (in files saved before like_kwargs existed:
        # separate entries; without noisecorr, the beam area)
        like = r.get('like_kwargs')
        if like is None:
            like = dict(noisecorr=r.get('noisecorr', 'beam'),
                        likemask=r['likemask'] if r.get('likemask') in ('dilated', 'none') else 'dilated',
                        dilation=tuple(r.get('likemask_dilation', (1., 3))),
                        densprof_smooth=r.get('densprof_smooth', 0.1))
        if 'radmax' not in like:                       # (files saved before the option: the last ring)
            like = dict(like, radmax='ring')
        self._set_like_options(like, custom=True)
        self._setup(r['freepar_inp'])
        # the likelihood of the fit: of the cube, of the total intensity map (run_geometry), or one given
        # by the user (not saved: None)
        self._log_like_kind = r.get('loglike', 'cube')
        self._log_like_func = dict(cube=self._log_likelihood, geometry=self._log_likelihood_geometry).get(
            self._log_like_kind)
        if sampler == 'dynesty' and r.get('results') is not None:
            self.results = r['results']
            weights = np.exp(self.results.logwt - self.results.logz[-1])
            self._set_sampling_stats(self.results.samples, weights)
        elif sampler in ('dynesty', 'nautilus') and r.get('weights') is not None:
            self.results = None
            self._set_sampling_stats(r['samples'], r['weights'])
        else:
            raise BBaroloError(f"load_results: cannot load results for sampler {sampler} from {inputfile}")
        self._method = sampler
        self.outri = self._update_rings(self._fit0.copy(), self.params)
        if self.useNorm and not self.useBBres:
            self._set_density(self.outri)
        self._galfit.outr = self.outri.object.copy()
        self.deconvolved_profile_errors()
