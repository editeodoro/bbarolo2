# GalMod3D (Python only)

GalMod3D builds a model datacube from three-dimensional fields of the gas: its density, its velocity and its
velocity dispersion, given as functions of $(R, z)$, $(R, \phi, z)$ or $(x, y, z)$ in the frame of the
galaxy. It makes models that a set of rings, as in {doc}`GALMOD <../galmod/index>`, cannot describe: a
flaring or lagging thick disk, a bar or spiral arms with streaming motions, a warp, or the fields of a
simulation on a grid. The model is observed on the grid of an input cube, with its beam, like the other
models. It is a Python class and not a task of the command line, so it has no parameter file and no
parameter page.

In Python: {class}`bbarolo.GalMod3D`.

## Example

This example puts a bar and two spiral arms in the disk of {doc}`NGC 2403 <../../getting-started/example-data>`,
seen at an inclination of 30° to make them visible. The density is a function of $(R, \phi, z)$ (R in
arcsec, $\phi$ in radians, measured in the plane of the galaxy from the receding half of the major axis):
an exponential disk times a bar and a logarithmic spiral pattern, with a Gaussian layer in z. The rotation
velocity is that of a flat curve, with a 10% modulation by the bar, plus a radial streaming velocity of
at most 15 km/s along the bar. `flux="measured"` scales the model to the flux of the data of the
cube in its mask; without it the sum of the model is the integral of the density.

```python
import numpy as np
from bbarolo import GalMod3D, Maps

def dens(R, phi, z):
    bar = 1 + 0.8 * np.exp(-(R / 150.) ** 2) * np.cos(2 * (phi - 0.5))
    arms = 1 + 0.5 * np.cos(2 * (phi - np.log(np.maximum(R, 1.) / 100.) / np.tan(np.radians(20))))
    return np.exp(-R / 400.) * bar * arms * np.exp(-z * z / 50.)

def vphi(R, phi, z):
    return 130 * np.tanh(R / 60.) * (1 + 0.1 * np.exp(-(R / 150.) ** 2) * np.cos(2 * (phi - 0.5)))

def vr(R, phi, z):
    return 15 * np.exp(-(R / 150.) ** 2) * np.sin(2 * (phi - 0.5))

m = GalMod3D("ngc2403.fits", coords="Rphiz", xpos=77, ypos=77, vsys=132.8, inc=30, pa=123.7,
             dens=dens, vphi=vphi, vr=vr, vdisp=8, zmax=30, flux="measured", threads=4)
model = m.model()          # an astropy PrimaryHDU, on the grid of ngc2403.fits
model.writeto("bar_spirals.fits", overwrite=True)

maps = Maps(model, mask="THRESHOLD", threshold=0.005, totalmap=True, velocitymap=True)
res = maps.run(threads=4, write=False)   # the moment maps of the model: the figure below
```

The fields are vectorised functions: they receive arrays of coordinates and return arrays. A field can also
be a number or a grid of values (`(axes, values)`, interpolated linearly). With `coords="Rz"` the
functions have two arguments and describe an axisymmetric system; with `coords="xyz"` the velocities are
`vx`, `vy` and `vz`. The model takes about 2 seconds here. `model()` writes no file: the cube is
an astropy HDU, which can be given to {class}`bbarolo.Maps`, {class}`bbarolo.PVSlice` or any other task.
The default model is the analytic one; `model(galmodtype="MC")` gives random clouds and
`model(noise=...)` adds noise.

The spiral arms show in the total intensity map. The velocity field is dominated by rotation; the streaming motions of
the bar change it only slightly, near the centre:

```{figure} /figures/galmod3d_maps.png
:width: 80%
:alt: Total intensity and velocity maps of a model with a bar and two spiral arms

Total intensity and velocity maps of the bar and spiral model (`GalMod3D(...).model()`, then
`Maps(...).run()`).
```

## Output

`GalMod3D.model()` returns the model cube as an `astropy.io.fits.PrimaryHDU` with the header of the input
cube; the units are those of the cube (for example Jy/beam), scaled to the total flux given by `flux`.
Nothing is written to the disk by the class.

## More

- {doc}`algorithm`: the coordinates, the fields, the sampling grid, the projection, the two ways of
  depositing the gas on the cube, the beam and the flux, and a validation against GALMOD.
- {doc}`../galmod/index`: models from rings, the usual way.
- {doc}`../galwind/index`: a model of a bi-conical outflow.

```{toctree}
:hidden:
:maxdepth: 1

algorithm
```
