# bbarolo2: BBarolo in pure Python

Port of 3D-Barolo (C++ 1.8.1dev) and its ctypes wrapper pyBBarolo 1.8.0 to pure Python:
package `bbarolo`, version 2.0.0 (`bbarolo/_version.py`). Numerical kernels are numba, compiled
at run time; everything else is numpy, scipy, astropy and matplotlib. Nothing to compile.

`README.md` is a short public overview, to be expanded. The detailed documentation of the Cowork
phase is `README_cowork.md`, kept on the development machine and not in git: the Python interface,
deliberate differences from C++, C++ bugs fixed in the port, validation numbers, the Bayesian
likelihood. Read the relevant section of it before changing behaviour.

## Layout

- `bbarolo/tasks.py`: the engine shared by the command line and Python. `STEPS` lists the tasks
  in BBarolo's order, each a pair `compute(c, par) -> Outputs` (everything in memory, no file) and
  `write(c, par, out)` (the files); `BBcore`/`run_par`/`run_steps` run them, computing then writing.
  The `Task` classes (`FitMod3D`, `GalMod`, ...) build a `Param`, read the `Cube` and run
  `run_steps(write=False)` into a `Result`; `Result.write()` replays the writes. Engines must not
  write while computing: files made during a computation go to `Cube.add_output` (written by the task).
- `bbarolo/keywords.py`: the parameter table. Every keyword is defined once in `KEYWORDS`
  (reader, `Param` attributes, tasks, help, aliases). New keywords go there.
  `bbarolo/params.py`: `Param`, `checkPars()` (port of `Param::checkPars`), BAYESIAN3DFIT parsers.
- `bbarolo/galmod.py`: `DiskModel`, the only model engine (GALMOD, 3DFIT, SPACEPAR, Bayesian,
  `SimulatedGalaxyCube`), with `GALMODTYPE` MC/ANALYTIC and `RINGMODE` CENTRED/NODES/SHIFTED.
- `bbarolo/galfit.py` + `galfit_out.py`: 3DFIT. `bbarolo/bayesian/`: `Bayesian3DFIT`
  (BAYESIAN3DFIT): `fitter.py` (the class: parameters, rings, priors, `_setup` steps, `run`) with the
  mixins `likelihood.py`, `density.py` (deconvolved profile) and `outputs.py`; `sampling.py`
  (nautilusbb, nautilus, dynesty, process pool), `noise.py`, `rings.py`, `task.py`. The other modules map
  one-to-one to C++ sources: see the section "Map of the C++ sources" of `README_cowork.md`.
- `bbarolo/bayesian/nautilus/`: a modified copy of nautilus 1.0.6 (MIT licence: keep its `LICENSE`), the
  default sampler (`SAMPLER NAUTILUSBB`; `SAMPLER NAUTILUS` is the PyPI package). No scikit-learn: `mlp.py`
  (numba network) and `gmm.py` (Gaussian mixture) replace it. List every change in its `__init__.py`.
  Its upstream tests are in `tests/nautilusbb/` (~2 minutes).
- `bin/BBarolo`, `run_bbarolo.py`: launchers that need no install. `examples/`: tutorial and
  C++ comparison scripts. `performance/`: 3DFIT accuracy suite and Bayesian coverage, with the
  report in `performance/report/`.
- `tests/test_BBB/` is the user's own test folder, not part of the package: edit it only when
  asked (it is not tracked in git).

## Reference code and data

- The C++ original is github.com/editeodoro/Bbarolo (`src/` for C++, `pyBBarolo/` for the old
  wrapper). Treat it as read-only. On the development server a clone sits at `../reference/Bbarolo`.
- Example cubes and parameter files live in `../examples` (not in this repo, and not in the public
  Bbarolo repo either). Tests look there, or in `$BBAROLO_EXAMPLES`.

## Running and testing

```bash
python3 bin/BBarolo -p param.par            # or: python3 -m bbarolo -p param.par
python3 bin/BBarolo -d 3DFIT                # parameters and defaults of a task
python3 -m pytest -q                        # about 1 minute (pyproject.toml ignores tests/test_BBB)
python3 performance/compare_examples.py HEAD -j 15   # all example outputs: working tree vs a git ref
```

- Every fix gets a regression test that fails on the previous code (check it in a `git worktree`
  of the previous commit).
- `performance/compare_examples.py` must report "identical" for changes meant to keep results;
  otherwise it shows what changed (FITS data difference, header keywords, text lines).
- `bash examples/run_all_tests.sh` runs every example task; run it from the folder that contains
  `examples/` (the parent of this repo).
- Comparisons with C++ (`examples/compare_galmod.py`, `compare_bayesian.py`,
  `performance/run_suite.py --cpp`) need a compiled C++ `BBarolo` and use `RINGMODE=SHIFTED`.

## Rules from the user

- Ask before changing defaults or anything else that changes results (e.g. the known 3DFIT
  biases are documented, not fixed: fixing them changes results with respect to C++).
- Remove obsolete options cleanly: no deprecation shims and no warnings for old saved files
  (reading old formats silently is fine).
- Record every behavioural change and bug fix in `CHANGES.md` (under "Unreleased"), marking
  with **(C++)** the bugs that are also in the C++ code.
- Run the tests after every change.

## Conventions

- The package and the pip distribution are both `bbarolo`; the old name `pyBBarolo` is gone (no
  compatibility alias). Only `examples/compare_bayesian.py` imports `pyBBarolo`: the original
  C++-based module, for comparisons.
- Plain Python, lines up to 120 characters, few type hints. Docstrings in plain English that
  state behaviour and units. British spelling in documentation and keywords (CENTRED, normalised).
- numba kernels: `@nb.njit(cache=True, nogil=True)`, numpy arrays and scalars in, release the
  GIL (3DFIT fits rings in parallel threads).
- Errors in the input: raise `BBaroloError` (`bbutils`, a `ValueError`) with a message that names
  the parameter, never `sys.exit`/`SystemExit` or "print and return False" (the command line prints
  the message and exits with 1). BBarolo-style `"TASK error: ..."` messages in `checkPars`.
- `p.verbose` is a level: 0 quiet, 1 normal (default), 2 everything. Detailed output (every ring,
  the parameter list) goes under `verbose >= 2`; save and restore the level, not `True`.

## Pitfalls

- numba caches in `bbarolo/__pycache__` record module names. Stale caches (e.g. from before the
  rename from `pyBBarolo`) can make the code import the wrong module: delete `__pycache__`.
- Do not edit `bbarolo/*.py` while a long multi-process run is going (e.g.
  `performance/bayes_coverage.py --procs N`): new workers import the edited modules.
- MC models are not bit-identical to C++ (different random generators), and the final MC model
  depends on THREADS (each chunk has its own seed). Ring fits are seeded per stage and ring, so
  the ring tables do not depend on THREADS.
