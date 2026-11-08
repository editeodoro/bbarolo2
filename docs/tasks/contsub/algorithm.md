# CONTSUB: how it works

CONTSUB removes the continuum emission from a datacube: in every spectrum it fits a polynomial in the
channel number to the channels without line emission and subtracts the polynomial from all the channels.
The input is the cube, the degree of the polynomial and the channels to exclude; the output is the
continuum-subtracted cube, on the same grid and in the same units, written as a FITS file. CONTSUB
replaces the cube in memory, and it runs before every other task: the tasks of the same run (3DFIT,
SEARCH, the maps, ...) analyse the subtracted cube.

## The channels of the fit

The channels are numbered $z = 0, \dots, n_z - 1$. {ref}`EXCLWIND <par-contsub-exclwind>` lists the
channels to exclude, single channels or ranges `first:last` (or `first~last`, both included), separated by
spaces or commas. The fit uses the set
$\mathcal{C}$ of the channels not excluded; by default $\mathcal{C}$ has all the channels, line included.
A channel outside the cube, an item that is not a channel or a range, or a list that excludes every
channel is an error. The same channels are excluded in all the spectra.

## The fit

For each spectrum $S(x, y, z)$ (units of the cube) the continuum is the polynomial of degree
$d$ = {ref}`CONTORDER <par-contsub-contorder>` (default 1, a straight line; a negative value is set to 1)

$$
C(x, y, z) = \sum_{m=0}^{d} c_m(x, y)\, z^m ,
$$

with the coefficients that minimise the unweighted sum of squares over the channels of the fit,

$$
\chi^2(x, y) = \sum_{z \in \mathcal{C}} \left[ S(x, y, z) - \sum_{m=0}^{d} c_m(x, y)\, z^m \right]^2 .
$$

This is a linear least-squares problem, $V \mathbf{c} = \mathbf{S}$, where $V_{zm} = z^m$ is the
Vandermonde matrix of the channels in $\mathcal{C}$. The matrix is the same for every spectrum, so all
the spectra are solved together, with a singular value decomposition of $V$; the solution is exact (no
iterations). All the channels have the same weight: the noise is assumed uniform along the spectrum.

The continuum is then evaluated at every channel, including the excluded ones, and subtracted:

$$
S'(x, y, z) = S(x, y, z) - \sum_{m=0}^{d} c_m(x, y)\, z^m , \qquad z = 0, \dots, n_z - 1 .
$$

Blank voxels (NaN in the FITS file) are read as 0 by BBarolo, so they enter the fit as zeros and are 0
minus the continuum in the output; NaN in a cube given in memory (in Python) are taken as 0 as well. The polynomial is in the channel number, not in frequency or velocity:
on a linear spectral axis the two are equivalent. The excluded channels are interpolated (or, at the ends
of the spectrum, extrapolated) by the polynomial; a high degree with a wide excluded window can make this
interpolation unstable.

:::{note}
Without EXCLWIND the line contributes to the fit, and a positive line raises the fitted continuum: the
subtracted cube then has a negative bias outside the line. Exclude the channels of the line, with a
margin, whenever the cube has a continuum to remove.
:::

## The output

The subtracted cube is written as `<prefix>_contsub.fits` in {ref}`OUTFOLDER <par-all-outfolder>`, with
the header of the input. In Python, the `ContSub` class runs the task alone, and `contsub=True` given to
any other task subtracts the continuum first.

## Differences from BBarolo 1.8

- The fit is a direct linear least-squares solution for all the spectra at once; C++ called its general
  non-linear least-squares fitter on each spectrum, which converges to the same solution (equal to
  floating-point precision on the examples). In C++ a spectrum whose fit failed was left unchanged.
- EXCLWIND: windows separated by commas are accepted. An item that is not a channel or a range, or a
  channel outside the cube, is an error (C++ ignored such items without warning, and wrote outside its
  array for channels beyond the cube); a list that excludes every channel is an error.

## References

- Di Teodoro E. M., Fraternali F., 2015, MNRAS, 451, 3021:
  [2015MNRAS.451.3021D](https://ui.adsabs.harvard.edu/abs/2015MNRAS.451.3021D)

## In the code

- `bbarolo/cube.py`: `Cube.continuumSubtract` (the channels, the fit and the subtraction).
- `bbarolo/tasks.py`: `_step_contsub` (runs it first and writes the cube).
