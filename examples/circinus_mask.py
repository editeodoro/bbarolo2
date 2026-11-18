"""
The mask of the ionised cone of Circinus for WINDFIT (examples/circinus_WINDFIT.par): the voxels that the fit
uses, out of the MUSE [OIII] cube circinus_OIII.fits.

The cube holds more than the wind: [OIII] of the disk over the whole field, the lines [Fe III] 4987 (-1170 km/s)
and He I 5016 (+530 km/s), and the far side of the nucleus. A mask keeps the fit to the cone:

  - the channels from -900 to +800 km/s (around [OIII] 5007; [Fe III] is out);
  - the spaxels within 65 degrees of the axis of the approaching cone (position angle -65 degrees on the sky,
    the gamma = 65 of Marconcini et al. 2023), from the nucleus;
  - of these, the spaxels where the line is bright: a peak signal-to-noise ratio of at least 30 in the cube
    smoothed to 1.8 arcsec (twice the seeing). The faint disk emission between the bright filaments is out.

Every voxel of the mask is data for the fit, the empty ones too (they tell a cone that is too large): the mask
is spaxels times channels, not the detections. WINDFIT uses it with
    MASK      FILE(circinus_OIII_mask.fits)
    LIKEOPTS  likemask=dilated dilation=0,0 ...

    python3 circinus_mask.py            ->  circinus_OIII_mask.fits (in the folder of the cube)
"""
import os
import sys

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import gaussian_filter

HERE = os.path.dirname(os.path.abspath(__file__))
CUBE = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "circinus_OIII.fits")
RA, DEC = 213.291251, -65.338826        # the nucleus (the apex of the cone): 14:13:09.90 -65:20:19.8
AXIS, HALF = -65., 65.                  # position angle of the axis of the cone and half width of the mask (deg)
VLO, VHI = -900., 800.                  # the channels of the fit (km/s)
SNRCUT, SMOOTH = 30., 1.8               # the peak S/N of a spaxel, in the cube smoothed to SMOOTH arcsec
REST, Z = 5006.843, 0.001448            # [OIII] 5007 in air (Angstrom) and the redshift of Circinus

hdu = fits.open(CUBE)[0]
h = hdu.header
d = np.nan_to_num(hdu.data.astype(np.float64))
nz, ny, nx = d.shape
lam = h["CRVAL3"] + (np.arange(nz) + 1 - h["CRPIX3"]) * h.get("CD3_3", h.get("CDELT3"))
v = (lam / (REST * (1 + Z)) - 1) * 299792.458

# the peak S/N of each spaxel around [OIII] 5007, in the cube smoothed to 1.8 arcsec (the noise: the median
# absolute deviation in the channels without lines, beyond -1350 and +850 km/s)
pix = abs(h.get("CD2_2", h.get("CDELT2"))) * 3600.
seeing = h["BMAJ"] * 3600.
s = np.sqrt(SMOOTH ** 2 - seeing ** 2) / (2 * np.sqrt(2 * np.log(2))) / pix
sm = np.array([gaussian_filter(ch, s) for ch in d])
free = (v < -1350) | (v > 850)
noise = 1.4826 * np.median(np.abs(sm[free] - np.median(sm[free])))
peak = sm[np.abs(v) < 800].max(axis=0) / noise

# the position angle of each spaxel from the nucleus (east of north)
x0, y0 = WCS(h).celestial.world_to_pixel_values(RA, DEC)
yy, xx = np.mgrid[0:ny, 0:nx]
pa = np.degrees(np.arctan2(-(xx - x0), yy - y0))            # (x grows to the west)
dpa = (pa - AXIS + 180) % 360 - 180
near = np.hypot(xx - x0, yy - y0) * pix < 1.                 # (the nucleus itself)

spax = (peak >= SNRCUT) & ((np.abs(dpa) <= HALF) | near)
chan = (v >= VLO) & (v <= VHI)
mask = (chan[:, None, None] & spax[None]).astype(np.uint8)
out = os.path.join(os.path.dirname(os.path.abspath(CUBE)), "circinus_OIII_mask.fits")
mh = h.copy()
mh["BUNIT"] = ""
mh.add_history("circinus_mask.py: the voxels of WINDFIT (1), the others 0")
fits.PrimaryHDU(mask, mh).writeto(out, overwrite=True)
print(f"noise of the smoothed cube {noise:.3g}; {spax.sum()} spaxels x {chan.sum()} channels -> {out}")
