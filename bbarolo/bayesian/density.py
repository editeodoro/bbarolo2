"""The density profile of BayesianBBarolo when DENS is not fitted: deconvolved from the total intensity
map (bbarolo.profile), with its errors, and the observed profile of the outputs."""

import math
import numpy as np

from .common import blas_threads


class DensityMixin(object):
    """ The density profile of BayesianBBarolo (DENS not fitted). """

    def _set_density(self, rings):
        """ Sets the densities of the rings, when DENS is not fitted, to the deconvolved profile:
        the densities whose beam-convolved model reproduces the total intensity map of the
        data (see _deconvolved_profile). The original module used the face-on average of the
        observed map in each ring (Ellprof), which the beam smooths twice (in the data and in
        the model): for marginally resolved disks, the inner rings were too faint and the outer
        ones too bright, which biased INC and VDISP. That profile is now only an output
        (<prefix>_densprof.txt, see _observed_profile). """
        dens = self._deconvolved_profile(rings)
        if dens is None:
            raise ValueError("BAYESIAN3DFIT ERROR: too few pixels with data in the likelihood region to compute "
                             "the density profile.\n  Check the mask and the rings, or fit DENS.")
        rings.modify_parameter("dens", dens, makeobj=True)

    def _observed_profile(self, rings):
        """ The Ellprof engine with the radial profile of the total intensity map of the voxels
        of the likelihood along the rings: the observed profile of the outputs (<prefix>_densprof.txt,
        the plots and the DENS_OBS column of <prefix>_densprof_deconvolved.txt). Not used in the fit. """
        if self._ellprof is None:
            from ..ellprof import Ellprof
            self._ellprof = Ellprof(self._cube_obj, rings.object)
            self._ellprof.im.array = self._likemask_map()
        self._ellprof.update_rings(rings.object)
        self._ellprof.RadialProfile()
        return self._ellprof

    def _deconvolved_profile(self, rings):
        """ Densities of the rings whose model reproduces the total intensity map of the data
        (data_mom0: the sum of the data in the voxels of the likelihood, see _setup_likemask).
        The total intensity map of the model is linear in the densities, sum_r dens_r K_r,
        where K_r is the map of ring r with unit density convolved with the beam
        (DiskModel.ring_maps: exact for the ANALYTIC model, the expectation of the MC one), and
        it depends only on the geometry, not on the kinematics. The densities are the
        non-negative weighted least-squares solution in the pixels of the likelihood region
        with data, weights 1/map_rms^2. This corrects the profile for the beam, unlike the
        average of the observed map in each ring.

        Two terms make it robust on real galaxies:
          * an exterior annulus, from the outer edge of the rings to two beams beyond, with
            its own density: it takes the emission beyond the last ring that falls in the
            likelihood region, which the last ring would otherwise absorb (its density is
            not used);
          * a smoothness prior: rings narrower than the beam are nearly degenerate once
            convolved, and noise or non-axisymmetric structure make the solution oscillate
            from ring to ring. The squared second differences of the profile (including the
            exterior annulus) are penalised with a weight densprof_smooth (default 0.1) times
            the mean data weight of a ring (trace(A^T A) / trace(L^T L)), independent of the
            units and of the S/N (0 = no prior).
        Returns None if there are too few pixels. """
        sysm = self._profile_system(rings)
        if sysm is None:
            return None
        A, b, _, Rc = sysm
        return self._solve_profile(A, b, Rc)[:rings.nr]

    def _profile_system(self, rings):
        """ The weighted least-squares system of the deconvolved profile (see
        bbarolo.profile.profile_system), in the likelihood box: the data map data_mom0 with its
        noise map_rms, in the pixels of the likelihood region with data. """
        from ..profile import profile_system
        Lb, Hb = self._likebox
        h = self._cube_obj.Head()
        sl = (slice(Lb[1], Hb[1]), slice(Lb[0], Hb[0]))
        ok = self._region[sl] & (self._nvox_map[sl] > 0)
        return profile_system(self._galfit.model, rings.object, (Lb, Hb), self.data_mom0[sl], self.map_rms[sl], ok,
                              beam_arcsec=abs(h.Bmaj()) * 3600. if h.Bmaj() else None)

    def _solve_profile(self, A, b, radii):
        """ bbarolo.profile.solve_profile with the smoothness densprof_smooth of like_kwargs. """
        from ..profile import solve_profile
        return solve_profile(A, b, radii, self.like_kwargs.get('densprof_smooth') or 0.)

    def deconvolved_profile_errors(self, nsamples=200, nmc=200, seed=0):
        """ Uncertainties of the deconvolved density profile of the best fit (DENS not fitted),
        from two sources combined in quadrature:

          geometry: the profile computed for nsamples samples of the posterior (16th-84th
                    percentiles around their median); zero if the geometry is not fitted;
          noise:    the spread of the profile fitted, at the best-fit geometry, to nmc
                    realisations of the best-fit total intensity map plus noise with the map
                    rms of each pixel and the spatial correlation of the noise (Gaussian, with
                    the correlation area 1/likefactor2d pixels).

        Called at the end of run() and load_results(). Sets and returns self.dens_profile, a
        dict with radii, dens (the profile of the best fit, as in outri and <prefix>_rings_final1.txt),
        err_lo, err_hi (total 1-sigma errors), err_geom_lo, err_geom_hi, err_noise_lo,
        err_noise_hi, nsamples and nmc; None if DENS is fitted. The errors are also written in
        <prefix>_rings_final1.txt (E_DENS) and <prefix>_densprof_deconvolved.txt (write_bestmodel). """
        with blas_threads():
            return self._deconvolved_profile_errors(nsamples, nmc, seed)

    def _deconvolved_profile_errors(self, nsamples, nmc, seed):
        from scipy.ndimage import gaussian_filter
        self.dens_profile = None
        if not (getattr(self, 'useNorm', False) and not self.useBBres and self.outri is not None):
            return None
        R = np.asarray(self.outri.r['radii'], float)
        d0 = np.asarray(self.outri.r['dens'], float)
        nr = len(R)
        rng = np.random.default_rng(seed)

        def spread(prof):
            p16, p50, p84 = np.percentile(np.asarray(prof), [16, 50, 84], axis=0)
            return p50 - p16, p84 - p50
        # geometry: the profile for samples of the posterior
        geo_lo = geo_hi = np.zeros(nr)
        ngeo = 0
        if self.update_prof and getattr(self, 'samples_equal', None) is not None and nsamples > 0:
            S = np.asarray(self.samples_equal)
            idx = rng.choice(len(S), min(int(nsamples), len(S)), replace=False)
            prof = [p for p in (self._deconvolved_profile(self._update_rings(self._fit0.copy(), S[i])) for i in idx)
                    if p is not None]
            if len(prof) > 1:
                geo_lo, geo_hi = spread(prof)
                ngeo = len(prof)
        # noise: refits of the best-fit map plus correlated noise, at the best-fit geometry
        noi_lo = noi_hi = np.zeros(nr)
        sysm = self._profile_system(self.outri) if nmc > 0 else None
        if sysm is not None:
            A, b, ok, Rc = sysm
            bmod = A @ self._solve_profile(A, b, Rc)          # the best-fit map (with the exterior annulus)
            sk = math.sqrt(max(1. / self.likefactor2d, 1.) / (4. * math.pi))   # Gaussian ACF of that area
            prof = []
            for _ in range(int(nmc)):
                g = rng.standard_normal(ok.shape)
                if sk > 0.3:
                    g = gaussian_filter(g, sk, mode='wrap')
                    g /= g.std()
                prof.append(self._solve_profile(A, bmod + g[ok], Rc)[:nr])
            noi_lo, noi_hi = spread(prof)
        self.dens_profile = dict(radii=R, dens=d0, err_lo=np.hypot(geo_lo, noi_lo), err_hi=np.hypot(geo_hi, noi_hi),
                                 err_geom_lo=geo_lo, err_geom_hi=geo_hi, err_noise_lo=noi_lo, err_noise_hi=noi_hi,
                                 nsamples=ngeo, nmc=int(nmc) if sysm is not None else 0)
        return self.dens_profile
