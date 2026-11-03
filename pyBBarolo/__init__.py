"""
Old name of the package: ``import pyBBarolo`` still works for a while, but the package
is now called ``bbarolo`` (``from bbarolo import FitMod3D``). pyBBarolo.X is bbarolo.X.
"""
import sys
import warnings
import importlib
import importlib.abc
import importlib.util


def _imported_by_numba():
    """True if numba is importing this package: compiled code cached by numba before the
    package was renamed refers to its functions as pyBBarolo.X. That is not the user's
    code, so no warning is given (the functions are those of bbarolo)."""
    f = sys._getframe(1)
    while f is not None:
        fn = f.f_code.co_filename.replace("\\", "/")
        if "/numba/" in fn:
            return True
        f = f.f_back
    return False


if not _imported_by_numba():
    warnings.warn("The pyBBarolo package is now called 'bbarolo': please use 'import bbarolo' "
                  "(the old name will be removed in a future version).", FutureWarning, stacklevel=2)

import bbarolo as _bb                      # noqa: E402
from bbarolo import *                      # noqa: E402,F401,F403
from bbarolo import __version__            # noqa: E402,F401


class _Alias(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """pyBBarolo.X -> the module bbarolo.X (the same module object)."""

    def find_spec(self, name, path=None, target=None):
        if name.startswith("pyBBarolo."):
            return importlib.util.spec_from_loader(name, self)
        return None

    def create_module(self, spec):
        return importlib.import_module("bbarolo." + spec.name[len("pyBBarolo."):])

    def exec_module(self, module):
        pass


if not any(isinstance(f, _Alias) for f in sys.meta_path):
    sys.meta_path.insert(0, _Alias())


def __getattr__(name):
    return getattr(_bb, name)
