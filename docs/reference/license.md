# License

BBarolo 2 is free software. It is distributed under the terms of the
[GNU General Public License](https://www.gnu.org/licenses/gpl-3.0.html), version 3 or (at your option) any
later version (SPDX identifier `GPL-3.0-or-later`, as declared in `pyproject.toml`). You can use, copy,
modify and redistribute it under the conditions of that licence, whose full text is in `LICENSE.md` at the top
of the repository. It comes without any warranty.

Copyright © Enrico Di Teodoro.

## Relation to the C++ BBarolo

The C++ BBarolo 1.8, of which BBarolo 2 is a port, is distributed under the GNU General Public License
version 2 or (at your option) any later version: this is the text of the header of its source files.
The "or any later version" clause allows a derived work to be distributed under version 3.

## Dependencies

BBarolo 2 needs these packages, which are installed with it and have their own licences:

| Package | Use |
|---|---|
| [numpy](https://numpy.org) | arrays |
| [scipy](https://scipy.org) | optimisation, FFTs, labelling of sources |
| [astropy](https://www.astropy.org) | FITS files and WCS |
| [matplotlib](https://matplotlib.org) | plots |
| [numba](https://numba.pydata.org) | compiled numerical kernels |

The optional extra `bayesian` adds [dynesty](https://dynesty.readthedocs.io) and
[nautilus-sampler](https://nautilus-sampler.readthedocs.io) (the nested samplers DYNESTY and NAUTILUS of
METHOD BAYESIAN of 3DFIT), corner and threadpoolctl; the extra `mpi` adds mpi4py and schwimmbad.

## nautilus

BBarolo includes a modified copy of [nautilus](https://github.com/johannesulf/nautilus) 1.0.6 (Lange 2023),
the sampler `NAUTILUSBB`, in `bbarolo/bayesian/nautilus`. nautilus is released under the MIT licence,
Copyright (c) 2022 Johannes U. Lange; the licence is in `bbarolo/bayesian/nautilus/LICENSE`, and the changes
are listed in `bbarolo/bayesian/nautilus/__init__.py`.

## The source finder

The source finder (SEARCH, and the masks that use it) derives from Duchamp, as the C++ code does: the
algorithms of the C++ class `Search` are based on the Duchamp source finder, described in
Whiting M. T., 2012, MNRAS, 421, 3242
([ADS](https://ui.adsabs.harvard.edu/abs/2012MNRAS.421.3242W)). See {doc}`../tasks/search/algorithm`. If
you use SEARCH, you may cite that paper as well as {doc}`citing`.
