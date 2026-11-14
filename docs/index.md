---
html_theme.sidebar_secondary.remove: true
---

# 3D-Barolo

:::{div} bb-hero
<p class="bb-tagline">A 3D fitting tool for the kinematics of galaxies</p>

**3D-Barolo** (3D-Based Analysis of Rotating Objects via Line Observations), or **BBarolo**, fits
3D tilted-ring models to emission-line datacubes. It derives the rotation curve, velocity
dispersion and geometry of any disk-like object (galaxy disks, circumnuclear disks, accretion and
protoplanetary disks) from any emission line: HI, CO and other molecular lines, recombination
lines. Fitting the datacube itself, rather than a velocity field, makes the results robust against
beam smearing, and reliable also for poorly resolved galaxies.

BBarolo {{ version }} is written in pure Python: nothing to compile, the same parameter files and
outputs as the C++ BBarolo, and a Python interface to every task.

```{button-ref} getting-started/index
:ref-type: doc
:color: primary
:class: sd-mr-2
Get started
```
```{button-ref} tasks/index
:ref-type: doc
:color: secondary
:outline:
The tasks
```
:::

```bash
pip install git+https://github.com/editeodoro/bbarolo2       # then: BBarolo -p param.par
```

## What it does

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} 3D tilted-ring fits
:link: tasks/3dfit/index
:link-type: doc
**3DFIT** fits rotation, dispersion and geometry ring by ring to the whole datacube, with the
beam and the spectral resolution in the model.
:::

:::{grid-item-card} Global and Bayesian fits
:link: tasks/3dfit/index
:link-type: doc
**3DFIT** with `METHOD GLOBAL` or `BAYESIAN` fits all the parameters at once: the maximum of the likelihood, or the
posterior by nested sampling, with errors, correlations, priors and the outer edge of the disk. **WINDFIT** and
**SLITFIT** fit winds and long-slit spectra the same way.
:::

:::{grid-item-card} Models of datacubes
:link: tasks/galmod/index
:link-type: doc
**GALMOD** builds model cubes of rotating disks; **GalMod3D** from any 3D density and velocity
field: bars, spirals, warps, simulations.
:::

:::{grid-item-card} Source finding and masks
:link: tasks/search/index
:link-type: doc
**SEARCH** finds the emission in a cube; **MAKEMASK** and **SMOOTH** build masks and change
the resolution.
:::

:::{grid-item-card} Maps and PV diagrams
:link: tasks/maps/index
:link-type: doc
Moment maps, global profiles, **PVSLICE** position-velocity diagrams, **ELLPROF** radial
profiles.
:::

:::{grid-item-card} Python and command line
:link: getting-started/index
:link-type: doc
Every task runs from a parameter file (`BBarolo -p`) or from Python (`FitMod3D(...).run()`),
with the results as astropy objects.
:::
::::

## News

- **BBarolo 2.0** (in development): the pure-Python BBarolo, with the global and Global and Bayesian fits
  (3DFIT METHOD GLOBAL and BAYESIAN, WINDFIT, SLITFIT), models from 3D fields (GalMod3D), new initial estimates and many fixes. See
  {doc}`reference/changes`.
- **BBarolo 1.8** (June 2026): the last version of the C++ code. Its documentation is
  {doc}`archived <reference/legacy>`.

## Citing BBarolo

If you use BBarolo in your research, please cite
[Di Teodoro & Fraternali (2015), MNRAS 451, 3021](https://ui.adsabs.harvard.edu/abs/2015MNRAS.451.3021D).
See {doc}`reference/citing`.

```{toctree}
:hidden:
:maxdepth: 2

getting-started/index
user-guide/index
tasks/index
api/index
examples/index
reference/index
```
