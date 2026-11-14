"""
The model of SLITFIT: a GALMOD disk on a thin 3D grid around the slit, seen through the slit.

The grid (the template) has the slit along its x axis: x covers the positions of the slit (pixels of the data, or
an integer fraction of them if they are larger than half the seeing FWHM), y a strip across the slit, wide enough
for all the light that the seeing brings into it (SLITWIDTH/2 + 4 sigma on each side), and z the channels of the
spectral window of the data, with their spectral axis (several lines are modelled as by GALMOD, each line of a
cloud at its own rest wavelength with the same Doppler factor). The geometry of the disk is then set in the frame
of the slit: the position angle of the model is 90 - SLITANGLE (SLITANGLE: the angle from the slit, in the
direction of increasing position, to the receding half of the major axis, counterclockwise), the centre is at XPOS
along the slit and YPOS (arcsec) across it, and the model is not convolved (DiskModel with no smoothing, ANALYTIC).

The seeing (a circular Gaussian, BEAM) and the slit are applied by a projection that is exact for a Gaussian:

    P(x, z) = g_x * sum_y w_y M(y, x, z)

w_y is the fraction of the light of a point at y that the seeing (in y) brings into the slit, the Gaussian
convolved with the slit box, and g_x the seeing along the slit (a 1D convolution). The model is the flux that
enters the slit per pixel of the data.
"""

import math
import numpy as np
from scipy.special import erf

from ..bbutils import BBaroloError


def _gauss_pix(sig, half, shift=0.):
    """The Gaussian of dispersion sig (pixels), centred at shift (pixels), integrated over the pixels -half..half."""
    k = np.arange(-half, half + 1, dtype=float) - shift
    s = math.sqrt(2.) * max(sig, 1e-6)
    g = 0.5 * (erf((k + 0.5) / s) - erf((k - 0.5) / s))
    return g / g.sum()


class SlitModel:
    """The model engine of a prepared slit (see the module documentation).

    SlitModel(prep, par, region, slitwidth, beam_fwhm, yoffset)
      prep:       the PreparedSlit (its spectral window and data);
      par:        the Param of the fit (GALMOD parameters: RESTWAVE, RELINT, REDSHIFT, SPECRES/LINEAR, LTYPE, ...);
      region:     (p0, p1): the positions of the data modelled (pixels of the slit, p1 excluded);
      slitwidth:  arcsec; beam_fwhm: the seeing FWHM (arcsec); yoffset: YPOS, the distance of the centre of the
                  galaxy from the slit, across it (arcsec; positive: the slit passes on the +y side of the centre,
                  the side 90 degrees counterclockwise from the direction of increasing position).
    Methods: rings(rd) the Rings object of the template from a _RingDict (XPOS in pixels of the slit); model(rings)
    the model (nz, npos) of a Rings object; basis(rings) the models of each ring with unit density (nr, nz, npos).

    XPOS in sub-pixels: the clouds of GALMOD fall in the nearest pixel, so a model built at any XPOS would jump each
    time a cloud crosses the edge of a pixel (a staircase in XPOS of tiny steps, on which minimisers stall). The
    model is built at the whole pixel of the template below XPOS (the same pattern of clouds for any XPOS), and the
    rest, a fraction of a pixel (rings: xshift), shifts the seeing along the slit, the Gaussian centred at that
    fraction: a model smooth in XPOS, and equal to the model one pixel further at a whole pixel.
    """

    def __init__(self, prep, par, region, slitwidth, beam_fwhm, yoffset=0., ringmode=None):
        from astropy.io import fits
        from ..cube import Cube
        from ..header import Header
        from ..galmod import DiskModel
        self.prep = prep
        slit = prep.slit
        self.pix = slit.pix                              # arcsec per position of the slit
        self.p0, self.p1 = int(region[0]), int(region[1])
        if not 0 <= self.p0 < self.p1 <= slit.npos:
            raise BBaroloError(f"SLITFIT: bad region of the slit {region}")
        self.width = float(slitwidth)
        self.fwhm = float(beam_fwhm)
        self.sigma = self.fwhm / (2. * math.sqrt(2. * math.log(2.)))
        self.yoffset = float(yoffset)
        # sub-pixels: the model pixels are at most half the seeing FWHM
        self.f = max(1, int(math.ceil(self.pix / (0.5 * self.fwhm))))
        pt = self.pix / self.f
        self.pt = pt
        ext = int(math.ceil(4. * self.sigma / self.pix)) + 1      # (positions beyond the region that reach it)
        self.tx0 = self.p0 - ext
        self.nxs = (self.p1 - self.p0) + 2 * ext                  # (positions of the template)
        nx = self.nxs * self.f
        half = int(math.ceil((0.5 * self.width + 4. * self.sigma) / pt))
        ny = 2 * half + 1
        self.yc = float(half)                                    # (the centre line of the slit, in template pixels)
        sp = prep.spec
        nz = sp.n
        h = fits.Header()
        h["NAXIS"] = 3
        h["NAXIS1"], h["NAXIS2"], h["NAXIS3"] = nx, ny, nz
        h["CTYPE1"], h["CTYPE2"] = "RA---CAR", "DEC--CAR"
        h["CUNIT1"], h["CUNIT2"] = "deg", "deg"
        h["CDELT1"], h["CDELT2"] = pt / 3600., pt / 3600.
        h["CRPIX1"], h["CRPIX2"], h["CRVAL1"], h["CRVAL2"] = 1., 1., 0., 0.
        h["CTYPE3"] = sp.ctype or {"wave": "WAVE", "freq": "FREQ", "velo": "VOPT"}[sp.kind]
        h["CUNIT3"] = sp.cunit
        h["CRVAL3"], h["CDELT3"], h["CRPIX3"] = sp.crval, sp.cdelt, sp.crpix
        h["BMAJ"] = h["BMIN"] = self.fwhm / 3600.
        h["BPA"] = 0.
        h["BUNIT"] = "NONE"
        h["OBJECT"] = slit.name
        if sp.kind == "freq" and sp.rest:
            h["RESTFREQ"] = float(np.asarray(par.gf.RESTFREQ, float)[0]) if par.gf.RESTFREQ else sp.rest[0]
        hd = Header(warning=False)
        hd.fromAstropy(h)
        hd.setRedshift(par.getRedshift())
        hd.setWave0(par.getRestwave())
        hd.setVelDef(sp.veldef)
        hd.fitsname = slit.name
        arr = np.broadcast_to(np.zeros(1, np.float32), (nz, ny, nx))
        self.cube = Cube.fromArray(arr, hd, par)
        self.disk = DiskModel(self.cube, par=par.gf, ringmode=ringmode, densflux=True, galmodtype="ANALYTIC",
                              smooth=False)
        self.ringmode = self.disk.ringmode
        self.nz, self.ny, self.nx = nz, ny, nx
        # the projection: across the slit (the seeing convolved with the slit box) and along it (the seeing)
        y = (np.arange(ny) - self.yc) * pt
        s = math.sqrt(2.) * self.sigma
        self.wy = 0.5 * (erf((y + 0.5 * self.width) / s) - erf((y - 0.5 * self.width) / s))
        self._khalf = int(math.ceil(4. * self.sigma / pt)) + 2
        self.kx = _gauss_pix(self.sigma / pt, self._khalf)
        self._kshift = (0., self.kx)
        self.sig_instr = self.disk.setup.sig_instr / 1000.       # km/s
        self._sr_key = None

    # ------------------------------------------------------------------ geometry
    def xpos_template(self, xpos):
        """The x pixel of the template of a position of the slit (pixels)."""
        return (np.asarray(xpos, float) - self.tx0 + 0.5) * self.f - 0.5

    def rings(self, rd):
        """The Rings object of the template for the rings rd (a _RingDict: XPOS in pixels of the slit, PHI the model
        position angle, already 90 - SLITANGLE). YPOS of rd is not used (the offset of the slit, yoffset). XPOS is
        that of the template less the fraction of a pixel of its first ring, kept as xshift (applied by the
        projection, see the class documentation)."""
        r = rd.object.copy()
        xt = np.asarray(self.xpos_template(r.xpos), float)
        r.xshift = float(xt.flat[0] - math.floor(xt.flat[0]))
        r.xpos = xt - r.xshift
        r.ypos = np.full(r.nr, self.yc - self.yoffset / self.pt)
        if getattr(rd.object, "edge", None) is not None:
            r.edge, r.edgewidth = rd.object.edge, rd.object.edgewidth
        return r

    # ------------------------------------------------------------------ models
    def _kernel(self, shift):
        """The seeing along the slit (pixels of the template), centred at shift (a fraction of a pixel)."""
        if self._kshift[0] != shift:
            self._kshift = (shift, _gauss_pix(self.sigma / self.pt, self._khalf, shift))
        return self._kshift[1]

    def project(self, m, shift=0.):
        """The projection of a template model m (ny, nx, nz) on the slit: (nz, npos) for the positions of the region,
        moved along the slit by shift (pixels of the template, the xshift of the rings)."""
        p = np.tensordot(self.wy, m, axes=(0, 0))                 # (nx, nz)
        from scipy.ndimage import convolve1d
        p = convolve1d(p, self._kernel(shift), axis=0, mode="constant")
        if self.f > 1:
            p = p.reshape(self.nxs, self.f, self.nz).sum(axis=1)
        ext = self.p0 - self.tx0
        return np.ascontiguousarray(p[ext:ext + self.p1 - self.p0].T)

    def project_maps(self, maps, shift=0.):
        """The projection on the slit of maps of the template (n, ny, nx), e.g. the total intensity maps of the rings
        (DiskModel.ring_maps): (n, npos) for the positions of the region, moved along the slit by shift (as project)."""
        from scipy.ndimage import convolve1d
        p = np.tensordot(maps, self.wy, axes=(1, 0))              # (n, nx)
        p = convolve1d(p, self._kernel(shift), axis=1, mode="constant")
        if self.f > 1:
            p = p.reshape(p.shape[0], self.nxs, self.f).sum(axis=2)
        ext = self.p0 - self.tx0
        return np.ascontiguousarray(p[:, ext:ext + self.p1 - self.p0])

    def model(self, rings):
        """The model of the rings (a Rings object of the template, see rings) seen through the slit: (nz, npos)."""
        m = self.disk.array(rings, final=True, layout="yxz", threads=1)
        return self.project(m, getattr(rings, "xshift", 0.))

    def _subrings(self, rings):
        """The sub-rings of the model of the rings with unit densities and the weights of each ring in them (cached by
        the radii and the ring convention: the densities of the sub-rings are linear in those of the rings)."""
        from ..galmod import model_rings
        st = self.disk.setup
        mr = model_rings(rings, self.ringmode, st.radsep)
        mr = mr.copy() if mr is rings else mr
        mr.dens = np.ones(mr.nr)
        sr, _ = st.subrings(mr)
        widx, wval = self.disk._ring_weights(rings, self.ringmode)
        return sr, widx, wval

    def basis(self, rings):
        """The models of each ring with unit density, seen through the slit: (nr, nz, npos). The model of the rings is
        sum_r dens_r basis_r (the densities are interpolated linearly between the radii, as in the model)."""
        from ..galmod import _galmod_analytic, ZSAMPLES_MAX, _PHI_TAB
        st = self.disk.setup
        sr, widx, wval = self._subrings(rings)
        nr = rings.nr
        out = np.zeros((nr, self.nz, self.p1 - self.p0))
        nsub = sr.shape[1]
        if nsub == 0:
            return out
        nzq = 1 if not np.any(sr[6] > 0) else ZSAMPLES_MAX
        buf = np.zeros((self.ny, self.nx, self.nz))
        args = (st.radsep, st.cdens, st.pixarea, st.cdelt[0], st.cdelt[1], st.crota2, 0, 0, self.nx, self.ny,
                st.ltype, st.sig_instr, st._vedges_lines, st.relint, st.cd2i, nzq, st.axtyp, st.velsys, st.lfreq0,
                st.crval3, st.cdelt3, st.crpix3, st._chmin, _PHI_TAB, st.zstep)
        dens0 = sr[5].copy()
        for r in range(nr):
            w = np.where(widx == r, wval, 0.).sum(axis=1)          # (nsub,): the weight of ring r in each sub-ring
            idx = np.flatnonzero(w)
            if idx.size == 0:
                continue
            sr[5] = dens0 * w
            buf[...] = 0.
            _galmod_analytic(buf, sr, int(idx[0]), int(idx[-1]) + 1, *args)
            out[r] = self.project(buf, getattr(rings, "xshift", 0.))
        sr[5] = dens0
        return out
