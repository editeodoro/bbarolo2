# Parallel runs

BBarolo uses the cores of one machine with threads, and with processes for the Bayesian sampler. There is
nothing to configure beyond the parameter {ref}`THREADS <par-all-threads>` (`threads=` in Python), and, for
a few cases, the environment variables below.

## THREADS

`THREADS` is the number of threads of the parallel parts of the tasks. Its default is the number of cores,
**at most 16** (BBarolo 1.8 used all of them). Values below 1 are set to 1. On a shared server, set it:
`BBarolo -p file.par THREADS=4`, or `res = f.run(threads=4)` in Python.

| Task | What runs in parallel |
|------|-----------------------|
| 3DFIT | the rings of each stage, one ring per thread |
| 3DFIT, GALMOD (final model) | the Monte Carlo model, in `THREADS` chunks of sub-rings |
| 2DFIT | the rings |
| SPACEPAR | the points of the grid |
| GALWIND | the cylinders or the shells |
| REND3D, maps (moments) | the angles, the pixels (numba threads, see below) |
| SMOOTH, convolution of the models | the FFT of the convolution (`scipy.fft` workers) |
| BAYESIAN3DFIT | the sampler: `THREADS` processes (see below) |
| SEARCH, MAKEMASK, ELLPROF, PVSLICE, STATS | nothing: one thread |

The rings of 3DFIT are fitted in Python threads, which work in parallel because the numba kernels release the
GIL. A run with fewer rings than threads leaves threads idle. The time of the 3DFIT of
{doc}`NGC 2403 <../getting-started/example-data>` (41 rings, two stages, `FREE VROT VDISP PA`, `run()` in
Python without writing, so without the plots) on a machine with many cores:

| `THREADS` | 1 | 2 | 4 |
|-----------|---|---|---|
| time (s) | 21.9 | 12.3 | 8.3 |

A fit of a bigger cube or with more free parameters scales in the same way, up to about the number of
rings.

### What depends on THREADS

- **The ring tables of 3DFIT do not.** Each ring is fitted with a random seed set by the stage and the ring,
  so the tables are identical for any number of threads: the fit above gave a maximum difference of 0.0 km/s in
  VROT between 1 and 4 threads.
- **The final Monte Carlo model does.** The final models of 3DFIT and GALMOD are built in `THREADS`
  chunks, each with its own random seed derived from {ref}`ISEED <par-galmod-iseed>`. The cube
  therefore changes with `THREADS` by the Monte Carlo noise. In a GALMOD of NGC 2403, with `NORM NONE`, the
  total flux was 913406.56 with 1 and 2 threads and 913406.31 with 4, and the largest difference between the cubes
  was 4.2 (1 against 2 threads) and 5.1 (2 against 4) (the peak of the model is 22). Two runs with the same number
  of threads are identical.
- With `GALMODTYPE ANALYTIC` the models have no noise: the cubes of 1, 2 and 4 threads agreed to 3·10⁻¹¹.

To reproduce a run, give the same `THREADS` (and `ISEED`, which is a fixed seed by default). The moment maps and
the other tasks do not depend on it.

## numba

The numerical kernels are compiled by [numba](https://numba.pydata.org) the first time they are used, and
cached on disk: in the folder `__pycache__` of the package, or in `NUMBA_CACHE_DIR` if you set it. The
first run is therefore slower. In the test above, with an empty cache, the first 3DFIT with four threads took
13.8 s instead of 8.3 s, and the first run of all the tasks takes longer (the installation page says up to a
minute). Set `NUMBA_CACHE_DIR` to a folder where you can write when the package is installed in a read-only
place, or when several users share it:

```bash
export NUMBA_CACHE_DIR=$HOME/.cache/bbarolo-numba
```

The cache is rewritten when the code of a kernel changes, for instance after an update. If a cache gets
stale, see {ref}`the troubleshooting page <ts-numba-cache>`.

Two kernels (moments, REND3D) use numba's own parallel threads: their number is `THREADS`, at most the
value of `NUMBA_NUM_THREADS` (by default, the number of cores). All the others are plain kernels, called from
Python threads.

## BLAS and other libraries

The linear algebra of numpy and scipy (BLAS, LAPACK) can start its own threads, set by `OMP_NUM_THREADS`,
`OPENBLAS_NUM_THREADS` or `MKL_NUM_THREADS`. BBarolo's kernels do not use it much, but the likelihood of
BAYESIAN3DFIT does, on small matrices, for which many threads are slower: with MKL on 224 cores it was
1.6 to 2.2 times slower, and the processes of the sampler competed for the cores. During the sampling and the
deconvolution of the density profile BBarolo therefore limits BLAS and OpenMP to **one thread**, with the
package [threadpoolctl](https://github.com/joblib/threadpoolctl) (installed with the `bayesian` extra; without it
the limit is not set). The results do not change. For the other tasks the libraries' defaults are used. If
`THREADS` threads each start a team of BLAS threads, the machine is oversubscribed: on a shared machine, set

```bash
export OMP_NUM_THREADS=1        # or OPENBLAS_NUM_THREADS / MKL_NUM_THREADS
```

before starting BBarolo, and use `THREADS` to choose the parallelism.

## BAYESIAN3DFIT: processes

{doc}`BAYESIAN3DFIT <../tasks/bayesian3dfit/index>` samples the posterior with `THREADS` **processes**, not
threads: the model engine of one likelihood runs on one thread. The pool is started with the `fork` method, so
the workers inherit the fitter without copying it, and the sampler (dynesty or nautilus) sends them only the
parameter vectors. dynesty evaluates four proposals per process at a time. Two consequences:

- the pool needs `fork`: on Windows, which has no `fork`, the task prints a warning
  (`WARNING: process pools need the 'fork' start method, not available on this system. Running serially.`)
  and runs serially;
- do not edit the package while a long run is going, since a new worker imports the modules from disk.

The default of `THREADS` in the task is the number of cores (at most 16). The Python class
{class}`bbarolo.Bayesian3DFIT` has `threads=1` by default: give `run(threads=n)`. When the script is
launched with `mpirun` or `mpiexec`, the sampler uses an MPI pool (needs `mpi4py` and `schwimmbad`) instead
of the process pool, so a fit can use several nodes.

The example of the {doc}`BAYESIAN3DFIT page <../tasks/bayesian3dfit/index>` takes between 2.5 and 3 minutes on
four processes, about 8 minutes of CPU time: the sampling scales almost linearly with the processes, but
the time grows quickly with the number of free parameters.

## Plots

The plots of a run are drawn by matplotlib scripts, started in processes forked from a helper that has
already imported the libraries. `BBAROLO_PLOT_SUBPROCESS=1` starts one new Python process per script instead:
use it if forking causes problems (macOS). On Windows, which has no `fork`, it is the only way and is
selected automatically.
