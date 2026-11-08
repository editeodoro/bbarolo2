# REND3D

REND3D makes a three-dimensional view of a datacube. It treats the cube as a volume (two spatial axes
and the spectral axis), turns it around the y axis one degree at a time and projects the emission in the
mask onto the sky at each angle. The result is a cube of images, one per angle, which played as a movie
shows the emission as a solid object turning. It helps to see the structure of the emission, for
instance a rotating disk or gas outside it.

In Python: {class}`bbarolo.Rend3D`.

## Example

This example renders {doc}`NGC 2403 <../../getting-started/example-data>` on 180 degrees, with a mask of
the voxels brighter than 0.01 Jy/beam.

::::{tab-set}
:sync-group: lang

:::{tab-item} Parameter file
:sync: par

```text
FITSFILE    ngc2403.fits
THREADS     4
MASK        THRESHOLD
THRESHOLD   0.01
REND3D      true
RENDANGLE   180
```

```bash
BBarolo -p ngc2403_rend3d.par
```
:::

:::{tab-item} Python
:sync: py

```python
from bbarolo import Rend3D

r = Rend3D("ngc2403.fits", mask="THRESHOLD", threshold=0.01, rendangle=180)
res = r.run(threads=4)

res.image                 # the rendered cube: an astropy HDU, 180 x 150 x 150
```
:::
::::

The run takes less than a second. The figure shows the images at four angles. At 0 degrees the line of
sight is the spectral axis, so the image is the total intensity map. At other angles the image mixes
position and velocity: the rotation of the disk shows as the S-shaped structure at 60 degrees. The images are summed along the line of sight and normalised by the
number of voxels in the mask, so their values are in Jy/beam, not in Jy/beam km/s.

```{figure} /figures/rend3d_views.png
:alt: Four views of the rendered cube of NGC 2403 at 0, 60, 120 and 179 degrees

Planes 1, 61, 121 and 180 of the rendered cube of NGC 2403 (`res.image`, angles 0, 60, 120 and 179 degrees).
```

## Outputs

The files are written in the output folder ({ref}`OUTFOLDER <par-all-outfolder>`, by default
`output/<object>/`) with the prefix `<object>`, the OBJECT of the FITS header. In Python they are also
attributes of the {class}`~bbarolo.Result`:

| File | `Result` | Content |
|---|---|---|
| `_3D.fits` | `image` | the rendered cube: one image per angle |
| `_mask.fits` | `mask` | the mask used |
| `_params.par` | | all the parameters of the run, defaults included |

`_3D.fits` has the spatial grid of the input cube, and the angle as third axis (CTYPE3 `L.O.S. ANGLE`, in
degrees, one plane per degree starting at 0). With {ref}`RENDANGLE <par-rend3d-rendangle>` 180 it has
180 planes; the default is 360, a full turn.

## More

- {doc}`parameters`: all the parameters of REND3D.
- {doc}`algorithm`: the mask, the rotation and projection, the normalisation and the smoothing.
- {doc}`../maps/index`: the moment maps, the projection at 0 degrees.

```{toctree}
:hidden:
:maxdepth: 1

parameters
algorithm
```
