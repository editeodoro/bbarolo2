# STATS

STATS calculates and prints the statistics of the voxels of a cube: mean, median, standard deviation,
MADFM (median absolute deviation from the median), maximum and minimum. They describe the noise of the
cube. The same statistics give the thresholds of the source finder and the noise used by other tasks.

In Python: {class}`bbarolo.Stats`.

## Example

This example calculates the statistics of the cube of
{doc}`NGC 2403 <../../getting-started/example-data>`.

::::{tab-set}
:sync-group: lang

:::{tab-item} Parameter file
:sync: par

```text
FITSFILE    ngc2403.fits
STATS       true
```

```bash
BBarolo -p ngc2403_stats.par
```
:::

:::{tab-item} Python
:sync: py

```python
from bbarolo import Stats

res = Stats("ngc2403.fits").run()

res.text                  # the statistics as written in the file
res.stats                 # the statistics object
```
:::
::::

The statistics are printed in the flux unit of the cube, here Jy/beam. The run takes less than a second:

```text
Calculating statistics for the cube... Done.
 Min    = -5.61045e-03	 Max      =  2.59943e-01
 Mean   =  2.40527e-03	 Std.Dev. =  1.27112e-02
 Median =  6.81619e-05	 MADFM    =  3.73598e-04 (= 5.53898e-04 as std.dev.)
```

The mean and the standard deviation are driven by the emission, the median and the MADFM much less. The
noise of the cube is therefore about 5.5e-04 Jy/beam, the MADFM converted to a standard deviation. Blank
voxels (0 or NaN) are not used. With {ref}`ITERNOISE <par-stats-iternoise>` the emission is excluded
iteratively.

## Outputs

The file is written in the output folder ({ref}`OUTFOLDER <par-all-outfolder>`, by default
`output/<object>/`) with the prefix `<object>`, the OBJECT of the FITS header. In Python, `run()` keeps
it in memory as an attribute of the {class}`~bbarolo.Result`, and `res.write()` writes the file:

| File | `Result` | Content |
|---|---|---|
| `_stats.txt` | `text` | the statistics, as written in the file |
| | `stats` | the statistics object |
| `_params.par` | | all the parameters of the run, defaults included |

`_stats.txt` has a header line and one line of numbers, in the flux unit of the cube:

```text
# MEAN       MEDIAN       STDDEV       MADFM        MADasSTDDEV  MAX          MIN          
2.4053e-03   6.8162e-05   1.2711e-02   3.7360e-04   5.5390e-04   2.5994e-01   -5.6104e-03  
```

## More

- {doc}`parameters`: all the parameters of STATS.
- {doc}`algorithm`: the estimators, the robust statistics and the iterative noise.
- {doc}`../search/index`: the source finder, which uses these statistics for its thresholds.

```{toctree}
:hidden:
:maxdepth: 1

parameters
algorithm
```
