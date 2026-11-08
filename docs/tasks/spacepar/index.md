# SPACEPAR

SPACEPAR computes the residual of a 3D model on a grid of two ring parameters, ring by ring, with all the
other parameters fixed. It does not fit anything: it shows the shape of the function that
{doc}`3DFIT <../3dfit/index>` minimises, so that one can see the degeneracies between two parameters (for
example the rotation velocity and the inclination), secondary minima and the size of the valley around
the best value.

In Python: {class}`bbarolo.SpacePar`.

## Example

This example explores the rotation velocity (60 to 160 km/s, in steps of 2 km/s) and the inclination
(30 to 90 degrees, in steps of 2 degrees) of two rings of
{doc}`NGC 2403 <../../getting-started/example-data>`, at 450 and 750 arcsec. The other parameters have
the values given. The rings are given with {ref}`RADII <par-spacepar-radii>`, to keep the run short.

::::{tab-set}
:sync-group: lang

:::{tab-item} Parameter file
:sync: par

```text
FITSFILE    ngc2403.fits
SPACEPAR    true
P1          VROT
P2          INC
P1PAR       60 160 2
P2PAR       30 90 2
RADII       450 750
XPOS        77
YPOS        77
VSYS        132.8
VROT        120
VDISP       8
INC         60
PA          123.7
Z0          10
MASK        SEARCH
```

```bash
BBarolo -p ngc2403_spacepar.par
```
:::

:::{tab-item} Python
:sync: py

```python
from bbarolo import SpacePar

f = SpacePar("ngc2403.fits", p1="VROT", p2="INC", p1par="60 160 2", p2par="30 90 2",
             radii="450 750", xpos=77, ypos=77, vsys=132.8, vrot=120, vdisp=8, inc=60,
             pa=123.7, z0=10, mask="SEARCH")
res = f.run(threads=4)

res.space.data            # the residuals: an array of shape (rings, INC, VROT)
```
:::
::::

The run takes about 20 seconds on four cores. The minimum of each ring is printed (and is marked in the
figure): VROT = 126 km/s and INC = 62 degrees at 450 arcsec, and 134 km/s and 58 degrees at
750 arcsec. The valley is long and curved: a lower inclination needs a higher rotation velocity to give
a similar model. The longer the valley, the less the data constrain the two parameters separately.

```{figure} /figures/spacepar_map.png
:alt: Maps of the residual over the grid of VROT and INC for two rings of NGC 2403, with the minimum marked

The increase of the residual over its minimum (per cent), for two rings of NGC 2403. The contours are at
2, 5, 10 and 30 per cent and the cross marks the minimum. Made from `res.space`.
```

## Outputs

The files are written in the output folder ({ref}`OUTFOLDER <par-all-outfolder>`, by default
`output/<object>/`) with the prefix `<object>`, the OBJECT of the FITS header. In Python, `run()` keeps
them in memory as attributes of the {class}`~bbarolo.Result`, and `res.write()` writes the files:

| File | `Result` | Content |
|---|---|---|
| `_spacepar.fits` | `space` | the residuals over the grid, one plane per ring |
| `_spacepar.pdf`, `plotscripts/spacepar.py` | | the plot of the maps (one panel per ring) and its script ({ref}`PLOTS <par-all-plots>`) |
| `_mask.fits` | `mask` | the mask |
| `_rings_initial.txt` | | the rings |
| `_params.par` | | all the parameters of the run, defaults included |

The `Result` also has `spacepar`, the engine that computed the maps. The plot is made by `write()`, as the
other files.

`_spacepar.fits` is a cube with the axes P1, P2 and the ring. Each plane holds
$|f - f_{\min}| / f_{\min}$ (dimensionless): 0 at the minimum of the ring, 0.05 where the residual is 5% higher.
The header gives the first value and the step of each axis (`CRVAL`, `CDELT`).

## More

- {doc}`parameters`: all the parameters of SPACEPAR.
- {doc}`algorithm`: the grid, the residual at a grid point, the output map and its plot.
- {doc}`../3dfit/index`: the fit of the same residual by minimisation.

```{toctree}
:hidden:
:maxdepth: 1

parameters
algorithm
```
