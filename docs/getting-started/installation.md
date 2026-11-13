# Installation

BBarolo needs Python 3.9 or later. It is written in pure Python: its numerical kernels are compiled
at run time by [numba](https://numba.pydata.org), so there is nothing to compile at installation.

## With pip

```bash
pip install git+https://github.com/editeodoro/bbarolo2
```

This installs the package `bbarolo` and the command `BBarolo`, with their requirements: numpy,
scipy, astropy, numba and matplotlib. Optional extras:

| Extra | Adds | For |
|---|---|---|
| `bayesian` | dynesty, nautilus-sampler, corner, threadpoolctl | the samplers DYNESTY and NAUTILUS of BAYESIAN3DFIT, its corner plot |
| `mpi` | mpi4py, schwimmbad | Bayesian samplings on several nodes with MPI |
| `test` | pytest | the tests |

```bash
pip install "bbarolo[bayesian] @ git+https://github.com/editeodoro/bbarolo2"
```

BAYESIAN3DFIT and BAYESIANWIND run without the extras: their default sampler (`SAMPLER NAUTILUSBB`) is part
of BBarolo.

## From a copy of the repository

```bash
git clone https://github.com/editeodoro/bbarolo2
cd bbarolo2
pip install -e ".[bayesian]"        # editable: changes to the code are used at once
```

Without installing, the launchers of the repository run BBarolo directly:
`python -m bbarolo -p param.par`, `bin/BBarolo -p param.par` or `python run_bbarolo.py -p param.par`.

## Checking the installation

```bash
BBarolo -v          # the version of BBarolo and of its requirements
BBarolo -l          # the list of tasks
```

```python
import bbarolo
print(bbarolo.__version__)
```

:::{note}
The first run compiles the numerical kernels, which takes up to a minute. They are cached in the
`__pycache__` folder of the package (or in the folder of the environment variable
`NUMBA_CACHE_DIR`), and the next runs start at once.
:::
