# A complete 3DFIT of NGC 2403

This page goes through a whole analysis of the HI cube of {doc}`NGC 2403 <../getting-started/example-data>`
with {doc}`3DFIT <../tasks/3dfit/index>`, step by step: a first look at the data, an automatic fit, a fit with
the geometry free, the checks of the fit, the errors, and the circular velocity corrected for asymmetric
drift. The one-line example of the task page is not repeated here. The run times are those of a
four-core machine (`threads=4`).

The commands are shown in Python. Each one has a parameter-file equivalent, with the same keywords in capitals
(`FREE VROT VDISP INC PA`, `FLAGERRORS true`, ...), and `res.write()` writes the files that the command line
writes.

## 1. A first look

The fit needs three things from the cube: the noise, the extent of the emission and the rough geometry.
{doc}`STATS <../tasks/stats/index>` gives the noise (less than a second):

```python
from bbarolo import Stats

print(Stats("ngc2403.fits").run().text)
```

```text
 Min    = -5.61045e-03	 Max      =  2.59943e-01
 Mean   =  2.40527e-03	 Std.Dev. =  1.27112e-02
 Median =  6.81619e-05	 MADFM    =  3.73598e-04 (= 5.53898e-04 as std.dev.)
```

The noise is about 5.5e-4 Jy/beam and the peak is 0.26 Jy/beam, so the peak signal-to-noise ratio is about
470: this is a high-S/N cube, and a fit of it is limited by the model, not by the noise. The moment maps
({doc}`MAPS <../tasks/maps/index>`) and a position–velocity slice along the major axis
({doc}`PVSLICE <../tasks/pvslice/index>`) show a regular rotating disk. The
position–velocity slice rises steeply in the centre and becomes flat:

```python
from bbarolo import Maps, PVSlice

m = Maps("ngc2403.fits", mask="THRESHOLD", threshold=0.002, totalmap=True, velocitymap=True).run()
pv = PVSlice("ngc2403.fits", xpos_pv=77, ypos_pv=77, pa_pv=123.7).run()
```

```{figure} /figures/ngc2403_first.png
:alt: Total intensity map, velocity field and major-axis position-velocity slice of NGC 2403

The total intensity (`m.totalmap`) and the velocity field (`m.velocitymap`) of the voxels above
0.002 Jy/beam (about 3.6 times the noise), and the position–velocity slice along PA = 123.7° (`pv.pv`).
```

The patches at about 0 km/s outside the disk, in blue in the velocity field, are faint and narrow
emission at 1–6 km/s, most likely HI of the Milky Way in the foreground: they are not part of the galaxy.

## 2. The automatic fit

With no parameters but the cube, 3DFIT estimates everything from the data: the mask (the detections of the
source finder, {ref}`MASK <par-3dfit-mask>` `SEARCH`), the centre, the systemic velocity, the inclination, the
position angle, the rotation velocity and the rings. It is the same as `BBarolo -f ngc2403.fits` on the
command line (a parameter file with `AUTO true` does it too).

```python
from bbarolo import FitMod3D

f = FitMod3D("ngc2403.fits")
res = f.run(threads=4)
res.rings_initial          # the estimates
res.rings                  # the fit
```

It takes 17 to 22 seconds. The estimates printed by the run are:

```text
   Fitting #21(e) rings of width 59.33(e) arcsec

    Xpos (e) =     75.13  pix            Ypos (e) =    77.13  pix
    Vsys (e) =    132.72  km/s           Vrot (e) =   151.36  km/s
    Inc (e)  =     51.62  deg            PA (e)   =   124.44  deg
    Z0 (d)   =     16.00  arcs           Disp (d) =      8.00  km/s
```

The rings are one beam wide, and extend to the edge of the mask, 1216 arcsec. The scale height (Z0, one
pixel) and the dispersion (8 km/s) are defaults, not estimates. By default 3DFIT fits VROT, VDISP, INC and PA
({ref}`FREE <par-3dfit-free>`) and has two stages ({ref}`TWOSTAGE <par-3dfit-twostage>`). The
run regularised the geometry for the second stage to INC = 54.5° and PA = 124.2°, and the rotation curve is
flat at about 141 km/s beyond 700 arcsec.

The estimates are a good start, not a result: they come from the data without any knowledge of the galaxy
(here the centre is 2 pixels away from the one used on the other pages, for instance). Use them to see what
the algorithm finds, and give your own values for what you know.

:::{note}
The inclination of a galaxy that is not very inclined is hard to measure from a velocity field alone, and
it is the main uncertainty of the rotation velocity, which scales with $1/\sin i$. The next step lets the
inclination vary around a given value.
:::

## 3. A fit with the geometry free

We now give the centre and the systemic velocity, and start the fit with INC = 60° and PA = 123.7°. The rings
are 60 arcsec wide (one beam) and 20 of them cover the disk. INC and PA are free, in an interval of
±15° around the initial inclination ({ref}`DELTAINC <par-3dfit-deltainc>`) and of ±15° (the default) around
the initial position angle.

::::{tab-set}
:sync-group: lang

:::{tab-item} Parameter file
:sync: par

```text
FITSFILE    ngc2403.fits
3DFIT       true
NRADII      20
RADSEP      60
XPOS        77
YPOS        77
VSYS        132.8
VROT        120
VDISP       8
INC         60
PA          123.7
Z0          10
FREE        VROT VDISP INC PA
DELTAINC    15
MASK        SEARCH
NORM        LOCAL
THREADS     4
```

```bash
BBarolo -p ngc2403_free.par
```
:::

:::{tab-item} Python
:sync: py

```python
from bbarolo import FitMod3D

f = FitMod3D("ngc2403.fits", nradii=20, radsep=60, xpos=77, ypos=77, vsys=132.8, vrot=120,
             vdisp=8, inc=60, pa=123.7, z0=10, free="VROT VDISP INC PA", deltainc=15,
             mask="SEARCH", norm="LOCAL")
res = f.run(threads=4)

res.rings_stage1          # first stage: every ring fitted independently
res.rings                 # second stage: the regularised geometry, VROT and VDISP fitted again
```
:::
::::

The run takes 8 seconds. In the first stage every ring has its own inclination and position angle. They
scatter, mostly because a ring is a thin ellipse that the beam makes broad, and because the inner rings
hold few independent resolution elements: in the figure the inclination of stage 1 goes from 59° to 67° and
the position angle of the innermost rings from 119° to 134°. The second stage replaces
them by a smooth function of radius ({ref}`REGTYPE <par-3dfit-regtype>`, by default chosen automatically,
here the median: INC = 62.2°, PA = 124.2°), fixes them, and fits VROT and VDISP again. The geometry of the
disk is thus the same for all the rings, which is what the data support, and the rotation curve is
smoother (see {doc}`../tasks/3dfit/algorithm` for the details):

```{figure} /figures/ngc2403_geometry.png
:width: 70%
:alt: VROT, VDISP, INC and PA of the rings of NGC 2403 in the two stages of the fit

The rings of the first stage (grey) and of the second stage (red), from `res.rings_stage1` and `res.rings`.
```

The two fits of this page do not agree on the inclination (54.5° and 62.2°), and the flat part of the
rotation curve is 141 km/s in one and 134 km/s in the other, a difference of 5%. This is the degeneracy of
the note above: the data alone do not fix the inclination of this disk to better than several degrees. If you
have a better value from another source (the optical image, for instance), give it with a small
DELTAINC, or keep INC fixed.

## 4. Checking the fit

A fit must always be looked at. The checks go from the global to the local.

**Position–velocity diagrams.** `res.plot_pv()` draws the slices along the major and the minor axis: the
data in grey and the model in red contours.

```python
fig = res.plot_pv()
```

```{figure} /figures/ngc2403_pv.png
:alt: Position-velocity diagrams of NGC 2403 along the major and minor axes, data and model

Position–velocity slices along the major and the minor axis of the data (grey) and of the fit (red
contours).
```

Along the major axis the model follows the data from the rising part to the flat part, on both sides, and
its contours reach the velocities of the data. The minor axis is a more sensitive test of the velocity
dispersion and of the inclination: the model is centred in velocity, and its extent in velocity and offset
is similar to that of the data. The pointed contours at zero offset come from the beam smearing of the
steep inner rotation curve.

**The residual cube.** `res.residual` is the data minus the model, inside the mask of the fit (NaN outside),
as an astropy HDU, normalised as the model (here `NORM LOCAL`: each voxel of the model is scaled to the data in its
own pixel, so that the residuals show the kinematics, not the surface density). `res.data` and `res.model` are
the other two cubes. The figure shows six channels:

```python
data, model, resid = (res.data.data, res.model.data, res.residual.data)   # numpy arrays
res.plot_maps()                                                            # moment maps of data and model
```

```{figure} /figures/ngc2403_channels.png
:alt: Channel maps of NGC 2403: data with model contours, model and residual

Six channels (velocity above each column): the data (grey) with the model contours (red, at 10, 30 and 60% of
the maximum of the cube), the model, and the residual in the mask, from -30% (blue) to +30% (red) of the
maximum of the cube.
```

The model reproduces the shape of every channel shown, which is what a good geometry and rotation curve do. The residuals are clumps of positive and negative values of up to about 30% of the peak,
at the size of the beam or smaller, with no obvious systematic pattern with radius or azimuth in these channels: they look like the
clumpy HI structure that a smooth ring model does not have. Check all the channels (the PDF below) before
concluding. A pattern
(one side always positive, say) would instead tell that the centre, the position angle, a
warp or a radial motion is wrong.

`res.write()` writes the PDF plots of the command line, among them `<object>_chanmaps_local.pdf`, with all the
channels, and `<object>_pvslices_local.pdf`, with PV slices at several angles. Without `plots`, they follow
{ref}`PLOTS <par-all-plots>`.

## 5. Errors and asymmetric drift

**Errors.** {ref}`FLAGERRORS <par-3dfit-flagerrors>` samples the residual function of each free parameter of
each ring after its fit, and estimates the lower and upper error ({doc}`algorithm <../tasks/3dfit/algorithm>`).
It is the expensive step: here (20 rings, 4 free parameters, 4 threads) it adds about 20 seconds to the
8 of the fit (the run with errors and ADRIFT takes 34 seconds), and the cost grows with the number of rings and of
free parameters and with the size of the cube. For a large cube, ask for the errors only for the final fit, not
while exploring. The errors are in the ring table:

```python
f = FitMod3D("ngc2403.fits", nradii=20, radsep=60, xpos=77, ypos=77, vsys=132.8, vrot=120,
             vdisp=8, inc=60, pa=123.7, z0=10, free="VROT VDISP INC PA", deltainc=15,
             mask="SEARCH", norm="LOCAL", flagerrors=True, adrift=True)
res = f.run(threads=4)

res.rings["vrot", "e_vrot1", "e_vrot2"]        # the lower (negative) and upper errors
res.rotation_curve                             # radii, vrot, e_vrot_lo, e_vrot_hi (positive)
```

The errors of VROT are 1.7 to 5.0 km/s (lower) and 1.9 to 4.3 km/s (upper), and those of VDISP 1.0 to 3.2 km/s
and 1.0 to 2.8 km/s. They are not statistical confidence
intervals: they measure how fast the residual grows when one parameter moves, with the others fixed. They
do not include the uncertainty of the inclination, which is the largest here (section 3). For the posterior
distributions, see {doc}`3DFIT with METHOD BAYESIAN <../tasks/3dfit/index>`.

**Asymmetric drift.** Gas with a velocity dispersion is held up partly by its pressure, so its rotation velocity
is lower than the circular velocity of the potential. With {ref}`ADRIFT <par-3dfit-adrift>`, BBarolo computes
the circular velocity $V_\mathrm{c}$ from the dispersion and the density profile of the fitted rings, after
Iorio et al. (2017), and writes it in `res.asymdrift`, with the columns `radii`, `vcirc` and the terms of the
correction:

```python
ad = res.asymdrift
ad["radii", "vcirc"]
```

```{figure} /figures/ngc2403_errors.png
:width: 70%
:alt: Rotation velocity with errors and circular velocity of NGC 2403

The rotation velocity of the second stage, with the errors of FLAGERRORS, and the circular velocity after the
asymmetric drift correction.
```

For NGC 2403 the correction is small: in the first 18 rings $V_\mathrm{c}$ exceeds $V_\mathrm{rot}$ by 0.2 to
2.8 km/s (0.3 to 2.1%), because the dispersion is low (10 km/s) compared with the rotation. It is
larger for dwarf galaxies and at high redshift. The correction depends on the derivatives of the regularised
dispersion and density profiles, and is unreliable in the outer rings, where the density falls to
zero (in the last two rings $V_\mathrm{c}$ is lower than $V_\mathrm{rot}$ by 0.3 and 1.3 km/s): look at it before using it (the polynomial degrees are {ref}`ADRIFTPOL1 <par-3dfit-adriftpol1>` and
{ref}`ADRIFTPOL2 <par-3dfit-adriftpol2>`).

## Summary

| Step | Task | Time |
|---|---|---|
| noise, maps, PV slice | STATS, MAPS, PVSLICE | under 1 s each |
| automatic fit | `FitMod3D(cube)` | 17–22 s |
| fit with INC and PA free, two stages | `FitMod3D(..., free="VROT VDISP INC PA")` | 8 s |
| the same with FLAGERRORS and ADRIFT | `flagerrors=True, adrift=True` | 34 s |

The next steps are the {doc}`Bayesian fit <bayesian>` of the same galaxy, and the test of the algorithm on
{doc}`simulated cubes <simulated>`.
