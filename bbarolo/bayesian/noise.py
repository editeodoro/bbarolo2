"""The noise of the cube for the likelihood of Bayesian3DFIT: its correlation, measured where there is
no emission, and the dilation of the mask."""

import math
import numpy as np


def _dilate(m, rz, ry, rx):
    """ Binary dilation of m by a box of half-sizes (rz, ry, rx) (separable, O(N)). """
    from scipy.ndimage import maximum_filter1d
    out = np.ascontiguousarray(m, dtype=bool).view(np.uint8)
    for axis, r in ((0, rz), (1, ry), (2, rx)):
        if r > 0:
            out = maximum_filter1d(out, size=2 * r + 1, axis=axis, mode='constant', cval=0)
    return out.astype(bool)


def dilate_mask(mask, radius, nchan):
    """ Dilation of a mask (nz, ny, nx) by a cylinder: a disk of the given radius (pixels)
    in each channel, then +-nchan channels. """
    from scipy.ndimage import distance_transform_edt, maximum_filter1d
    m = np.asarray(mask, bool)
    out = m.copy()
    if radius > 0:
        for k in np.flatnonzero(m.any(axis=(1, 2))):
            out[k] = distance_transform_edt(~m[k]) <= radius
    if nchan > 0:
        out = maximum_filter1d(out.view(np.uint8), size=2 * int(nchan) + 1, axis=0, mode='constant',
                               cval=0).astype(bool)
    return out


def noise_correlation(data, mask=None, beam_fwhm=None, maxlag_xy=None, maxlag_z=6, minvox=2000, minfree=20000):
    """ Correlation of the noise of a cube, measured where there is no emission.

    data: cube (nz, ny, nx); mask: the emission (same shape), dilated by about one beam and
    two channels to exclude the faint wings; beam_fwhm: beam FWHM in pixels; maxlag_xy:
    radius (pixels) of the spatial lags summed (default 2.5 beam FWHM); maxlag_z: largest
    spectral lag (channels).

    The correlation area is the sum of the normalised autocorrelation of the noise over the
    lags: the number of voxels per independent noise measurement. The noise is assumed
    separable, and the autocorrelations are computed with FFTs, weighting by the voxels
    used:
      * spatial (within the channel maps): in the channels without emission if they contain
        at least minfree noise voxels, otherwise in all the voxels outside the dilated mask;
      * spectral (along the spectra): in the runs of emission-free channels if they are at
        least three channels long and give at least minfree/2 pairs at lag 1 (lags up to the
        longest run - 1); otherwise in the pixels without emission in any channel, if they
        have at least minfree voxels; otherwise in the voxels outside the dilated mask.
    Emission-free channels have at most 0.1% of their pixels in the mask and are at least
    two channels away from those with more than 1%. Emission below the detection threshold
    of the mask adds spurious correlation, hence the preference for these channels; smaller
    samples give noisy sums (tens of per cent).

    For white noise convolved with a Gaussian beam the spatial area is twice the beam area
    in pixels; for noise whose correlation is the beam itself, as in interferometric images,
    one beam area. Hanning smoothing along the spectral axis gives 8/3 channels.

    Returns None if there are fewer than minvox noise voxels in all, otherwise a dict with
    spatial (pixels), spectral (channels), volume (= spatial * spectral, voxels), the alternative
    areas spatial_sq, spectral_sq (sums of the squared autocorrelation) and spatial_core,
    spectral_core (sums over the positive core, out to the first lag where the autocorrelation
    is <= 0; the likelihood uses the largest of the sum, the core and one beam area or one
    channel, see LikelihoodMixin._set_likefactor), rho_spectral
    (autocorrelation at the spectral lags used), nnoise (number of noise voxels),
    spatial_from and spectral_from (the voxels used: 'free channels', 'free pixels' or
    'voxels outside the mask').
    """
    from scipy import fft
    data = np.asarray(data)
    nz, ny, nx = data.shape
    valid = np.isfinite(data)
    bp = float(beam_fwhm) if beam_fwhm is not None and beam_fwhm > 0 else 2.
    if mask is not None and np.any(mask):
        m = np.asarray(mask, bool)
        r = max(1, int(math.ceil(bp)))
        emis = _dilate(m, 2, r, r)
        noise = valid & ~emis
        frac = m.reshape(nz, -1).mean(axis=1)
        chanfree = frac <= 0.001
        for k in np.flatnonzero(frac > 0.01):
            chanfree[max(k - 1, 0):k + 2] = False
        pixfree = ~emis.any(axis=0)
        del emis
    else:
        noise = valid
        chanfree = np.ones(nz, bool)
        pixfree = np.ones((ny, nx), bool)
    nnoise = int(noise.sum())
    if nnoise < minvox:
        return None
    vals = data[noise]
    if vals.size > 2000000:
        vals = vals[np.random.default_rng(0).choice(vals.size, 2000000, replace=False)]
    med = float(np.median(vals))
    del vals

    def centred(sl, w):
        """ data - median in the selected voxels, 0 elsewhere (float64) """
        return np.where(w, np.asarray(data[sl], np.float64) - med, 0.)

    # spatial autocorrelation (zero spectral lag), averaged over the channels
    free_ch = int(noise[chanfree].sum()) >= minfree
    use = chanfree if free_ch else np.ones(nz, bool)
    R = float(maxlag_xy) if maxlag_xy else 2.5 * bp
    L = max(1, min(int(math.ceil(R)), ny - 1, nx - 1))
    fy, fx = fft.next_fast_len(ny + L), fft.next_fast_len(nx + L)
    pn = np.zeros((fy, fx // 2 + 1))
    pw = np.zeros((fy, fx // 2 + 1))
    chans = np.flatnonzero(use & noise.any(axis=(1, 2)))
    for k in range(0, len(chans), 8):
        cc = chans[k:k + 8]
        w = noise[cc]
        a = fft.rfft2(centred(cc, w), s=(fy, fx))
        b = fft.rfft2(w.astype(np.float64), s=(fy, fx))
        pn += (a.real ** 2 + a.imag ** 2).sum(axis=0)
        pw += (b.real ** 2 + b.imag ** 2).sum(axis=0)
    cn, cw = fft.irfft2(pn, s=(fy, fx)), fft.irfft2(pw, s=(fy, fx))
    lags = np.arange(-L, L + 1)
    iy, ix = np.ix_(lags % fy, lags % fx)
    with np.errstate(invalid='ignore', divide='ignore'):
        c = np.where(cw[iy, ix] > 0.5, cn[iy, ix] / cw[iy, ix], 0.)
    rlag = np.hypot(*np.meshgrid(lags, lags, indexing='ij'))
    inside = rlag <= R
    spatial = float(np.sum(c[inside]) / c[L, L]) if c[L, L] > 0 else 1.
    # alternative areas: the sum of the squared autocorrelation, and the sum over the positive core (out to the
    # first radius where the azimuthal mean of the autocorrelation is <= 0)
    rho = c / c[L, L] if c[L, L] > 0 else np.zeros_like(c)
    spatial_sq = float(np.sum(rho[inside] ** 2)) if c[L, L] > 0 else 1.
    rbin = np.rint(rlag).astype(int)
    prof = np.array([rho[rbin == k].mean() for k in range(int(np.ceil(R)) + 1)])
    r0 = int(np.argmax(prof <= 0)) if np.any(prof <= 0) else len(prof)
    spatial_core = float(np.sum(rho[inside & (rbin < r0)])) if c[L, L] > 0 else 1.
    # spectral autocorrelation (zero spatial lag)
    K = max(0, min(int(maxlag_z), nz - 1))
    runs = []
    k = 0
    while k < nz:
        if chanfree[k]:
            k2 = k
            while k2 + 1 < nz and chanfree[k2 + 1]:
                k2 += 1
            runs.append((k, k2 + 1))
            k = k2 + 1
        else:
            k += 1
    longest = max([b - a for a, b in runs], default=0)
    pairs1 = sum(int((noise[a:b - 1] & noise[a + 1:b]).sum()) for a, b in runs if b - a >= 2)
    free_px = int((valid & pixfree[None]).sum()) >= minfree
    if longest >= 3 and pairs1 >= minfree // 2:
        spectral_from = 'free channels'
        K = min(K, longest - 1)
        cz, wz = np.zeros(K + 1), np.zeros(K + 1)
        for a, b in runs:
            ok = noise[a:b]
            seg = centred(slice(a, b), ok)
            for t in range(min(K, b - a - 1) + 1):
                pr = ok[:b - a - t] & ok[t:]
                cz[t] += float(np.sum((seg[:b - a - t] * seg[t:])[pr]))
                wz[t] += float(pr.sum())
    else:
        spectral_from = 'free pixels' if free_px else 'voxels outside the mask'
        fz = fft.next_fast_len(nz + K)
        qn = np.zeros(fz // 2 + 1)
        qw = np.zeros(fz // 2 + 1)
        for j in range(0, ny, 32):
            w = (valid[:, j:j + 32] & pixfree[None, j:j + 32]) if free_px else noise[:, j:j + 32]
            a = fft.rfft(centred((slice(None), slice(j, j + 32)), w), n=fz, axis=0)
            b = fft.rfft(w.astype(np.float64), n=fz, axis=0)
            qn += (a.real ** 2 + a.imag ** 2).sum(axis=(1, 2))
            qw += (b.real ** 2 + b.imag ** 2).sum(axis=(1, 2))
        cz, wz = fft.irfft(qn, n=fz)[:K + 1], fft.irfft(qw, n=fz)[:K + 1]
    with np.errstate(invalid='ignore', divide='ignore'):
        rz = np.where(wz > 0.5, cz / wz, 0.)
    rz = rz / rz[0] if rz[0] > 0 else np.r_[1., np.zeros(K)]
    spectral = float(1. + 2. * np.sum(rz[1:]))
    spectral_sq = float(1. + 2. * np.sum(rz[1:] ** 2))
    k0 = int(np.argmax(rz <= 0)) if np.any(rz <= 0) else len(rz)
    spectral_core = float(1. + 2. * np.sum(rz[1:k0]))
    spatial, spectral = max(spatial, 1.), max(spectral, 1.)
    return dict(spatial=spatial, spectral=spectral, volume=spatial * spectral, rho_spectral=rz, nnoise=nnoise,
                spatial_sq=max(spatial_sq, 1.), spectral_sq=max(spectral_sq, 1.),
                spatial_core=max(spatial_core, 1.), spectral_core=max(spectral_core, 1.),
                maxlag_xy=R, maxlag_z=K, spatial_from='free channels' if free_ch else 'voxels outside the mask',
                spectral_from=spectral_from)
