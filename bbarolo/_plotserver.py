"""
Helper process that runs BBarolo's plotting scripts (started by plotscripts.PlotRunner).

It runs as a standalone script in a fresh interpreter (it does not import bbarolo).
It imports numpy, matplotlib (Agg backend), astropy and scipy once, warms up
matplotlib's font and mathtext caches, then reads a JSON request from stdin:

    {"paths": [script, ...], "nproc": N}

Each script runs in a child process forked from this one, so the libraries are already
loaded. This is what makes it faster than starting a new interpreter for every script.
At most N children run at a time, longest scripts first. The scripts run as with
``python script.py``: as __main__, with their folder first in sys.path, in the current
working directory, with stdout/stderr discarded. The reply on stdout is a JSON list
with one exit code per script, in the order of the request.

The libraries are imported while the parent is still preparing the outputs, so the
cost of importing them is (partly) hidden.
"""

import os
import sys
import json


def _preload():
    os.environ.setdefault("MPLBACKEND", "Agg")
    try:
        import numpy                                       # noqa: F401
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.backends.backend_pdf             # noqa: F401
        import matplotlib.backends.backend_agg             # noqa: F401
        from matplotlib.colorbar import ColorbarBase       # noqa: F401
        import matplotlib.gridspec                         # noqa: F401
        import astropy.io.fits                             # noqa: F401
        import astropy.visualization                       # noqa: F401
        import scipy.ndimage                               # noqa: F401
        import scipy.interpolate                           # noqa: F401
    except Exception:                                      # the scripts import what they need
        return
    try:
        # Warm-up: font lookups, glyph caches, mathtext parser and PDF writer, with the
        # settings used by the scripts. rcParams are restored afterwards.
        import io
        with matplotlib.rc_context():
            plt.rc('font', family='sans-serif', serif='Helvetica', size=13)
            plt.rcParams.update({'text.usetex': False, 'mathtext.fontset': 'cm', 'mathtext.default': 'regular'})
            fig = plt.figure(figsize=(2, 2))
            ax = fig.add_subplot(111)
            ax.plot([0, 1], [0, 1])
            ax.contour([[0., 1.], [1., 2.]], [0.5, 1.5])
            ax.set_xlabel(r'$\mathrm{\Delta V_{LOS}}$ (km/s) 0123456789.-')
            ax.set_title(r'$\theta$ = 0$^\circ$ $\sigma_\mathrm{gas}$')
            fig.savefig(io.BytesIO(), format='pdf', bbox_inches='tight')
            plt.close(fig)
    except Exception:
        pass


def _run_child(path):
    """Body of a forked child: run the script as __main__, then exit (never returns)."""
    code = 1
    try:
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        import runpy
        sys.argv = [path]
        sys.path.insert(0, os.path.dirname(os.path.abspath(path)))
        runpy.run_path(path, run_name="__main__")
        code = 0
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
    except BaseException:
        code = 1
    finally:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
        os._exit(code)


# scripts that usually take longest are started first
_ORDER = ("pvslices", "chanmaps", "kinmaps", "pvs", "parameters")


def _priority(path):
    name = os.path.basename(path)
    for i, key in enumerate(_ORDER):
        if key in name:
            return i
    return len(_ORDER)


def run(paths, nproc):
    codes = [1] * len(paths)
    queue = sorted(range(len(paths)), key=lambda i: _priority(paths[i]))
    running = {}
    nproc = max(1, int(nproc))
    while queue or running:
        while queue and len(running) < nproc:
            i = queue.pop(0)
            sys.stdout.flush()
            sys.stderr.flush()
            pid = os.fork()
            if pid == 0:
                _run_child(paths[i])
            running[pid] = i
        pid, status = os.waitpid(-1, 0)
        if pid in running:
            i = running.pop(pid)
            codes[i] = os.waitstatus_to_exitcode(status) if hasattr(os, "waitstatus_to_exitcode") else status
    return codes


def main():
    _preload()
    req = sys.stdin.read()
    if not req.strip():                                   # nothing to do (e.g. the parent gave up)
        return 0
    req = json.loads(req)
    paths = req.get("paths", [])
    nproc = req.get("nproc") or os.cpu_count() or 1
    codes = run(paths, min(nproc, max(len(paths), 1)))
    sys.stdout.write(json.dumps(codes))
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
