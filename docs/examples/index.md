# Examples

## Example datasets

Real data of different kinds, each with a parameter file, from radio cubes of nearby galaxies to a
protoplanetary disk, a long-slit spectrum and the wind of an active nucleus. The cubes and the parameter files are in the folder
[`examples/`](https://github.com/editeodoro/bbarolo2/tree/main/examples) of the repository
(see {doc}`../getting-started/example-data`). Each page shows the parameter file and the same run in Python,
explains the choices of the setup, and shows the outputs: the rings, the moment maps, the position–velocity
diagrams and the channel maps of the data and of the model. The notebook
[`examples.ipynb`](https://github.com/editeodoro/bbarolo2/blob/main/examples/examples.ipynb), in the same folder, runs all the examples in Python, in memory, and shows their plots.

| Example | Data | Task | Time |
|---|---|---|---|
| {doc}`ngc2403` | HI, VLA, nearby spiral | 3DFIT CLASSIC, geometry free, errors | 52 s |
| {doc}`ngc5633` | Hα + [NII], CALIFA optical IFU | CONTSUB, 3DFIT CLASSIC with three lines | 25 s |
| {doc}`pgc53639` | HI, ATCA, warped disk at low inclination | 3DFIT CLASSIC, position angle free | 24 s |
| {doc}`ugc1281` | HI, WSRT, edge-on dwarf | 3DFIT GLOBAL, smoothness priors, errors | 40 s |
| {doc}`ngc891` | HI, WSRT, edge-on spiral with a lagging halo | 3DFIT GLOBAL, thin disk and thick layer with a lag | 18 min |
| {doc}`u3-25160` | Hα, KMOS, star-forming galaxy at z = 0.9 | 3DFIT BAYESIAN | 1 min 55 s |
| {doc}`aless73` | [CII], ALMA, starburst galaxy at z = 4.8 | 3DFIT BAYESIAN, prior on the inclination | 8 min 19 s |
| {doc}`hd142527` | ¹³CO, ALMA, protoplanetary disk | 3DFIT BAYESIAN, geometry and thickness free | 52 min |
| {doc}`2masxj1022` | Hα + [NII], SALT long slit, massive spiral | SLITFIT GLOBAL | 27 s |
| {doc}`circinus` | [OIII], MUSE, ionised nuclear wind of a Seyfert galaxy | WINDFIT GLOBAL with a mask, MoKA-3D compared | 16 min |

The times are the wall-clock times of `BBarolo -p` on our server (Intel Xeon Platinum 8480+) with 16 cores
(`THREADS 16`, the default), start-up and output files included. The server was shared with other jobs, so the
times are indicative.

```{toctree}
:maxdepth: 1
:hidden:

ngc2403
ngc5633
pgc53639
ugc1281
ngc891
u3-25160
aless73
hd142527
2masxj1022
circinus
```

## Tutorials

Models made by BBarolo itself: model cubes from 3D fields, and simulated cubes to test a fit.

```{toctree}
:maxdepth: 1

galmod3d
simulated
```
