# REND3D: how it works

REND3D makes a three-dimensional view of a datacube: it treats the cube as a volume with axes x
(pixels), y (pixels) and the spectral axis (channels), turns it around the y axis one degree at a time
and, at each angle, projects the emission within the mask onto the plane of the sky. The input is the
cube and its mask; the output is a cube of images, one per angle, written as a FITS file. Played as a
movie, the images show the emission as a solid object turning around the y axis, which helps to see the
structure of the line emission (e.g. a rotating disk, or emission outside it).

The task runs in three steps: the mask is built, the projections are computed, the projections are
smoothed and written.

## The mask

Only the voxels in the mask contribute to the projections. The mask is the one already built by a
previous task of the run, or the one set by {ref}`MASK <par-rend3d-mask>` (default `SEARCH`, the
detections of the source finder), with {ref}`BLANKCUT <par-rend3d-blankcut>` and
{ref}`FACTOR <par-rend3d-factor>` for the mask `SMOOTH`. Let $M(x, y, z) = 1$ in the mask and 0 outside,
and $S(x, y, z)$ the intensity (units of the cube), with blanks (NaN) taken as 0. Here $x$, $y$ are
the pixel indices and $z$ the channel index, all counted from 0, and the cube has $n_x \times n_y \times
n_z$ voxels.

## The projections

The volume is rotated in the plane of the x axis and the spectral axis, about the point
$(c_x, c_z) = (n_x/2, n_z/2)$, with one pixel and one channel taken as equal lengths. The angles are
$\phi_i = i$ degrees, $i = 0, 1, \dots, n_\phi - 1$, with $n_\phi$ the integer part of
{ref}`RENDANGLE <par-rend3d-rendangle>` (default 360, a full turn). For the image at angle $\phi_i$, the
output pixel $(x, y)$ collects the voxels along a line of sight sampled at the $n_z$ points
$z = 0, \dots, n_z - 1$, each rotated back into the cube and rounded to the nearest voxel (halves away
from zero):

$$
x_1 = \mathrm{round}\!\left[ (x - c_x)\cos\phi_i + (z - c_z)\sin\phi_i + c_x \right], \qquad
z_1 = \mathrm{round}\!\left[ -(x - c_x)\sin\phi_i + (z - c_z)\cos\phi_i + c_z \right].
$$

Points that fall outside the cube are skipped. The image is the sum of the masked voxels, divided by
their total number in the image:

$$
P_i(x, y) = \frac{1}{N_i} \sum_{z=0}^{n_z-1} M(x_1, y, z_1)\, S(x_1, y, z_1), \qquad
N_i = \sum_{x, y} \sum_{z=0}^{n_z-1} M(x_1, y, z_1).
$$

$N_i$ is one number for the whole image $i$, not a count per pixel: the images are sums along the line
of sight, scaled by a constant so that images at different angles have comparable values even when the
volume seen changes. $P_i$ is in units of the cube. An image with no masked voxel ($N_i = 0$) is empty.

At $\phi = 0^\circ$ the line of sight is the spectral axis and $P_0$ is the total intensity map of the
masked cube, divided by the number of voxels in the mask. At $\phi = 90^\circ$ the line of sight is the
x axis and the image is a position-velocity diagram along y summed over x, with the channels along the
horizontal axis in reverse order. Since the line of sight has $n_z$ samples of unit length centred on
the cube, a cube with $n_x \ne n_z$ is not seen in full at all angles: at $90^\circ$, for example, only
the central $\min(n_x, n_z)$ pixels along x and channels are used.

## Smoothing and output

Each image is then smoothed with the algorithm of {doc}`SMOOTH <../smooth/algorithm>`, from the beam of
the cube $(a, b, \theta)$ to $(1.2\,a, 1.2\,b, \theta)$, to soften the pixelation of the rotated volume.
The flux scale factor of SMOOTH (about $1.2^2 = 1.44$, the ratio of the beam areas) is always applied,
whatever the units, and no pixel is blanked. Empty images ($N_i = 0$) are 0 after the smoothing.

The images are written as the planes of the cube `<prefix>_3D.fits` in
{ref}`OUTFOLDER <par-all-outfolder>`, with the spatial header of the input. The third axis is the angle:
CTYPE3 `L.O.S. ANGLE`, CUNIT3 `DEGREES`, CDELT3 1, CRVAL3 0 and CRPIX3 1: the plane $p$ (counted from 1)
holds the projection at $p - 1$ degrees.

## Differences from BBarolo 1.8

- The projections are the same as in BBarolo 1.8 (equal to floating-point precision), with the angles
  computed in parallel.
- An image with no masked voxel is empty (0 in the output); C++ divided by zero.
- CRPIX3 is 1: with the CRPIX3 = 0 of C++ the header gave each plane an angle one degree too large.

## References

- Di Teodoro E. M., Fraternali F., 2015, MNRAS, 451, 3021:
  [2015MNRAS.451.3021D](https://ui.adsabs.harvard.edu/abs/2015MNRAS.451.3021D)

## In the code

- `bbarolo/rendering3d.py`: `_render` (the projections, a numba kernel parallel over the angles),
  `Rendering3D` (`compute`, `writefits`, which smooths with `Smooth3D` of `bbarolo/smooth.py`).
- `bbarolo/tasks.py`: `_step_rend3d`.
