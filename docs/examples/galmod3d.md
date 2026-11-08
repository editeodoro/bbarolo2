# Models from 3D fields with GalMod3D

{doc}`GalMod3D <../tasks/galmod3d/index>` builds a model cube from fields of density and velocity that you
write as Python functions or give as values on a grid. The page of the task shows a bar and spiral arms.
Here are three other cases that a set of rings cannot describe: a thick disk that flares and lags,
seen almost edge-on; a warp, given as fields on a grid as if they came from a simulation; and the two ways
of putting the gas on the cube. All the models are made on the grid of the cube of
{doc}`NGC 2403 <../getting-started/example-data>` (pixels of 16 arcsec, a beam of about 59 arcsec, 62 channels
of 5.15 km/s), with the flux of the data (`flux="measured"`). Because the beam is large, the scale heights
used here (30 to 90 arcsec) are larger than those of real disks: they are chosen to be seen at this
resolution.

## A thick disk that flares and lags

The gas of a thick disk above the plane often rotates more slowly than the gas in the plane (a vertical
gradient of the rotation velocity, a lag). In `coords="Rz"` the fields are functions of $(R, z)$, so both
effects are one line each. The first model is a thin disk (Gaussian layer, $\sigma_z$ = 10 arcsec), the second
an exponential layer whose scale height grows with radius, $h = 30 + R/20$ arcsec (a flare), and whose
rotation velocity falls by 0.3 km/s per arcsec of height, $v_\phi = V_\mathrm{c}(R) - 0.3\,|z|$:

```python
import numpy as np
from bbarolo import GalMod3D, PVSlice

def vcirc(R):                                   # km/s, R in arcsec
    return 130 * np.tanh(R / 60.)

def thin(R, z):
    return np.exp(-R / 400.) * np.exp(-z * z / (2 * 10. ** 2))

def thick(R, z):
    h = 30 + R / 20.                            # scale height (arcsec), growing with radius
    return np.exp(-R / 400.) * np.exp(-np.abs(z) / h) / h * 30

geo = dict(xpos=77, ypos=77, vsys=132.8, inc=85, pa=123.7, flux="measured", threads=4)
m1 = GalMod3D("ngc2403.fits", coords="Rz", dens=thin, vphi=lambda R, z: vcirc(R), vdisp=8., zmax=100, **geo)
m2 = GalMod3D("ngc2403.fits", coords="Rz", dens=thick, vphi=lambda R, z: vcirc(R) - 0.3 * np.abs(z),
              vdisp=8., zmax=300, **geo)
thin_cube, thick_cube = m1.model(), m2.model()

# PV diagram along the major axis, and along a line 100 arcsec (6.25 pixels) from it
pv = PVSlice(thick_cube, xpos_pv=77, ypos_pv=77, pa_pv=123.7, width_pv=48).run().pv
```

`zmax` is the half-height of the grid in which the fields are sampled (arcsec): it must contain the
layer, and a small value makes the model much faster. The thin disk took about 5 seconds and the thick one
about 11 on two threads (the box of $z$ is three times larger). Seen at 85°, the models differ in the
position–velocity diagrams:

```{figure} /figures/galmod3d_lag.png
:width: 85%
:alt: Position-velocity diagrams of a thin disk and of a thick lagging disk seen edge-on

Position–velocity diagrams of the thin disk (left) and of the thick disk with a flare and a lag (right),
along the major axis (top) and along a line parallel to it, 100 arcsec above the plane (bottom, the
slices are 48 arcsec wide). Both models have the same flux. The dotted line is the systemic velocity.
```

In the thin disk the emission along the major axis reaches the same velocity at all radii beyond the rise of
the curve and is concentrated in two bright blocks. In the thick disk it is spread towards the systemic velocity
(the gas above the plane is slower, so a line of sight through the disk has a tail to low velocities); at an
offset of −600 arcsec the intensity-weighted mean velocity is 234 km/s against 250 km/s for the thin disk, and the
peak is 0.14 against 0.31 Jy/beam. Above the plane, the thin disk has only the emission that the beam
(59 arcsec) spreads out of the plane, while the thick one has real gas, which rotates more slowly: the
envelope of the diagram at $z$ = 100 arcsec is lower in velocity than that of the major axis, and the
emission fills the diagram to the systemic velocity. These are the signatures used to detect lagging gas
in edge-on galaxies.

## A warp, from fields on a grid

A simulation gives its fields on a grid, not as functions. With `(axes, values)` GalMod3D interpolates them
linearly, so the same call works. Here the "simulation" is a disk whose plane bends beyond 300 arcsec, up on
one side of the major axis and down on the other, $z_\mathrm{w} = 0.2\,(R - 300)\,x/R$ (arcsec; $x$ is along
the major axis, so the line of nodes is the minor axis; a tilt of about 11° at large radii).
The fields are in Cartesian coordinates, `coords="xyz"`, with velocities `vx`
and `vy` (the tilt of the warp is neglected in the velocities, which are those of rotation in the plane
$z = 0$):

```python
x = np.linspace(-2000, 2000, 201)               # arcsec
z = np.linspace(-500, 500, 101)
X, Y, Z = np.meshgrid(x, x, z, indexing="ij")
R = np.hypot(X, Y)
zwarp = 0.2 * np.maximum(R - 300, 0) * X / np.where(R > 0, R, 1.)
rho = np.exp(-R / 600.) * np.exp(-(Z - zwarp) ** 2 / (2 * 30. ** 2))
v = vcirc(R) / np.where(R > 0, R, 1.)           # v_phi / R, so that vx = -v y and vy = v x

axes = (x, x, z)
m = GalMod3D("ngc2403.fits", coords="xyz", xpos=77, ypos=77, vsys=132.8, inc=85, pa=123.7,
             dens=(axes, rho), vx=(axes, -v * Y), vy=(axes, v * X), vdisp=10., zmax=500,
             flux="measured", threads=4)
cube = m.model()
```

The values are an array with one axis per entry of `axes`, in the same order as the arguments of the
fields. Outside the grid the density is 0 and the other fields take their value at the nearest edge, so the
grid must cover the disk. The model took about 40 seconds. The same disk written as functions of
$(x, y, z)$ took about 17 seconds and differs from the grid model by 0.02% of the peak (rms) and at most
0.7% in any voxel, which is the error of the linear interpolation.

```{figure} /figures/galmod3d_warp.png
:width: 70%
:alt: Total intensity and velocity maps of a warped disk at two inclinations

Total intensity and velocity maps of the warped disk (made with {class}`bbarolo.Maps`, as on the page of the
task) at 85° (top) and 60° (bottom). At 85° the warp bends the outer disk away from the major axis, one side
up and the other down.
```

## The two ways of depositing the gas

By default (`model(galmodtype="ANALYTIC")`) each point of the grid gives a Gaussian line, integrated over the
channels: the model is the expectation of the Monte-Carlo one, smooth and without noise. With
`model(galmodtype="MC")` the gas is `nclouds` random clouds, as in the Monte-Carlo models of GALMOD. They
have the same flux, and the Monte-Carlo noise falls as $N^{-1/2}$. For the warped disk written as functions
(rms difference from the ANALYTIC model, in per cent of its peak; the times are those of the model with
2 other jobs running on the same machine):

| `nclouds` | Rms difference | Time |
|---|---|---|
| $10^5$ | 0.32% | 6 s |
| $10^6$ (default) | 0.10% | 6 s |
| $10^7$ | 0.03% | 11 s |

The time of the clouds is mostly the sampling of the fields on the grid, so the Monte-Carlo model is not
faster than the ANALYTIC one (18 s here); its use is to give a model with grain, as GALMOD does, and
`seed` makes it reproducible. Use ANALYTIC unless you need that, for example to fit with a model that has
no noise: see {doc}`../tasks/galmod3d/algorithm` for both.

## Other points

- The model is a cube in memory: `model()` writes no file. Use `cube.writeto("model.fits", overwrite=True)` for
  one, and give it to {class}`bbarolo.Maps`, {class}`bbarolo.PVSlice` or {class}`bbarolo.FitMod3D` as it is.
- `model(noise=sigma)` adds noise of rms `sigma` per voxel (the units of the cube), smoothed by the beam.
- Along the minor axis an inclined disk projects to $|y|\cos i$, so the default `rmax` may be too small for
  an almost edge-on disk: give `rmax` and `zmax` to cover the whole disk, as above.
- A thin disk with a Gaussian layer is also what {doc}`GALMOD <../tasks/galmod/index>` makes from rings: use
  GALMOD for such a disk, and GalMod3D when the density or the velocity depend on $z$ or $\phi$ in a way
  that rings cannot describe.

## More

- {doc}`../tasks/galmod3d/index` and {doc}`../tasks/galmod3d/algorithm`: the coordinates, the fields,
  the grid, the projection and the validation against GALMOD.
- {doc}`simulated`: simulated cubes of galaxies made with GALMOD.
