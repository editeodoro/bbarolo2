# BAYESIANWIND

BAYESIANWIND fits a bi-conical wind to an emission-line datacube by sampling the posterior distribution of
its geometry and kinematics with nested sampling (nautilus or dynesty). The wind is the model of
{doc}`GALWIND <../galwind/index>`: two opposite cones with their apex at the centre, along the rotation axis
of a disk of inclination `INC` and position angle `PA`, between the opening angles `OPENIN` and `OPENANG`,
out to `HTOT`, with a radial velocity, a velocity dispersion and a rotation around the axis in each shell.

The emission of the wind is **not assumed**. Winds are patchy: clumps and filaments fill the cones
unevenly, and a smooth density law would bias the geometry. BAYESIANWIND describes the emission as a set
of emissivities on cells of the cones (about one beam across, each cone its own). The emissivities are
solved at every evaluation of the likelihood and marginalised over, so the posterior of the geometry and
kinematics accounts for every distribution of the emission that the data allow. The marginalisation
also penalises cones wider and winds faster than the data need, which the best-fit emission alone does
not do (see {doc}`algorithm`).

In Python: {class}`bbarolo.BayesianWind`. It needs the package of the sampler
(`pip install "bbarolo[bayesian]"`).

## Example

A mock wind with a clumpy emission (performance/wind_mocks.py): cones of 60° full opening angle, 200 arcsec
long, inclined by 60°, with a radial velocity of 200 km/s and a dispersion of 20 km/s. The initial wind is
given as for GALWIND, with a guess of the geometry; the free parameters are the velocity of the wind, its
inclination and its opening angle. `EMSCALE` and `EMCLUMP`, the mean emissivity and the clumpiness of the
prior of the emission, are fitted with them.

::::{tab-set}
:sync-group: lang

:::{tab-item} Parameter file
:sync: par

```text
FITSFILE      wind_mock.fits
BAYESIANWIND  true
XPOS          23.5
YPOS          23.5
VSYS          0
INC           55
PA            30
OPENANG       55
HTOT          200
VWIND         180
VDISP         20
FREE          VWIND_SINGLE INC OPENANG
THREADS       8
```
:::

:::{tab-item} Python
:sync: py

```python
from bbarolo import BayesianWind

f = BayesianWind("wind_mock.fits", xpos=23.5, ypos=23.5, vsys=0, inc=55, pa=30, openang=55, htot=200,
                 vwind=180, vdisp=20)
f.run(freepar=["vwind_single", "inc", "openang"], threads=8)
f.print_stats()
f.write("output/wind/")
```
:::
::::

## Free parameters

`FREE` lists them, as for BAYESIAN3DFIT:

- `VWIND`, `VDISP`, `VROT`: one value per shell (the shells are `RADII`, or `NTOT` shells, by default one per
  beam), `NAME_SINGLE` one value for all the shells, `NAME_FUNC` a function of radius (VWIND: an arctangent
  rising to a terminal velocity; VDISP and VROT: linear);
- `XPOS`, `YPOS`, `VSYS`, `INC` (0 to 90°: the cone along +z recedes, `PA` sets the side it projects on),
  `PA`, `OPENANG`, `OPENIN`, `HTOT`;
- `EMSCALE`, `EMCLUMP`, `EMSLOPE`: the prior of the emissivities, its mean, its fractional scatter and its
  power-law slope in radius. EMSCALE and EMCLUMP are always fitted, added to the free parameters unless
  `LIKEOPTS` fixes them; EMSLOPE is 0 unless free;
- `SIGMA`: the noise, as BAYESIAN3DFIT.

The default priors follow the initial wind (INC ± 15°, PA ± 20°, OPENANG ± 30°, VWIND 0 to twice the
initial value, ...); `PRIORS` changes them, e.g. `PRIORS OPENANG=(uniform,20,100)`.

## Options of the likelihood

`LIKEOPTS` (`like_kwargs` in Python):

| option | meaning |
|---|---|
| `emission` | the description of the emission: `cells` (default), `shells` (one emissivity per shell and cone), `cone` (one per cone, a power law in radius), `spaxel` (a uniform cone normalised in each spaxel, as NORM LOCAL), `voxel` (the voxel weighting of MoKA-3D: for comparison only, it is degenerate) |
| `cellsize` | the size of the cells (arcsec; default the beam FWHM) |
| `marginal` | `true` (default): the likelihood marginalised over the emission; `false`: its maximum (the profile likelihood) |
| `prior` | `gauss` (default: Gaussian emissivities, exact marginal likelihood) or `exp` (non-negative exponential emissivities, approximate) |
| `emscale`, `emclump` | fixed values of EMSCALE and EMCLUMP (default: fitted) |
| `noisecorr`, `sigma` | the correlation and the value of the noise, as BAYESIAN3DFIT |
| `datafrac` | a model error proportional to the data (default 0) |
| `grow` | the region of the likelihood: the projected cones of the priors grown by `grow` beams (default 1) |
| `prune` | values of the model below `prune` times the peak of their cell are left out of the likelihood (default 1e-3) |

A fixed disk can be added with `DISKRINGS` (a rings file of 3DFIT or BAYESIAN3DFIT): its geometry and
kinematics are those of the file, and the densities of its rings are fitted with the emission of the wind.

## Outputs

In the output folder, with the prefix of the cube:

- `_wind_params.txt`: the free parameters, the median and the 1σ interval of their posterior, and the priors;
- `_wind_shells.txt`: the shells of the best fit with their errors, and the flux of each cone in radial bins;
- `_wind_cells.txt`: the cells of the emission and their emissivities at the best fit;
- `_windmod.fits`, `_windmod_0mom.fits`, `_windmod_1mom.fits`, `_windmod_2mom.fits`: the best model and its maps;
- `_wind_pv.fits`, `_windmod_pv.fits`: position-velocity slices of the data and of the model along the
  projected axis of the cones;
- `_<sampler>_windresults.npy`: the samples, their weights and the setup of the fit;
- with `PLOTS`: `_wind_corner.pdf`, `_wind_maps.pdf` (maps and slices of data and model), `_wind_shells.pdf`.

```{toctree}
:hidden:

parameters
algorithm
```
