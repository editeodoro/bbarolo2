"""
Command line interface of BBarolo (port of bbarolo.cpp and Param::getopts).

    BBarolo -p param.par                 # run with a parameter file
    BBarolo -f cube.fits                 # automated 3DFIT
    BBarolo -c fitsfile=cube.fits 3dfit=true ...
    BBarolo -p param.par THREADS=4       # parameters on the command line override the file
    BBarolo -d [TASK] | -t | -l | -v | -q
    BBarolo --listhead/--modhead/--remhead/--fitscopy/--fitsarith/--jybeam2k/--k2jybeam ...

The same entry point is available as ``python -m bbarolo``.
"""

import os
import sys
import numpy as np

from .params import Param, TASKS_LIST, FITSUTILS_LIST
from . import keywords as _kw
from ._version import __version__, __cpp_reference__
from .bbutils import randomQuoting, BBaroloError


# --------------------------------------------------------------------------
# Messages
# --------------------------------------------------------------------------
def welcomeMessage(stream=None):
    stream = stream or sys.stdout
    msg = f" Welcome to BBarolo v{__version__}"
    size = 70
    sizel = int((size - 10 + len(msg)) / 2.)
    sizer = size - sizel
    stream.write("\n" + "=" * (size - 1) + "\n"
                 + "=====" + "=====".rjust(size - 6) + "\n"
                 + "=====" + msg.rjust(sizel) + "=====".rjust(sizer - 6) + "\n"
                 + "=====" + "=====".rjust(size - 6) + "\n"
                 + "=" * (size - 1) + "\n\n")


def listTasks(stream=None):
    stream = stream or sys.stdout
    stream.write("     _ _                                                            \n"
                 ".-. | B |                                               .--.        \n"
                 "|A|_| A |                                         .-===-|==|---.    \n"
                 "|S| | R.|<\\            Available tasks            | Py  |  |___|-. \n"
                 "|T| | O | \\\\                                      |     |GR|---|=|\n"
                 "|R| | L |  \\\\                                     |     |  |___| |\n"
                 "|O| | O |   \\>                                    |-----|==|---|=| \n"
                 + '"' * 66 + "\n\n")
    for name, descr in TASKS_LIST:
        stream.write(f"{name}: \n   {descr}\n\n")
    stream.write('"' * 66 + "\n\n")


def listFitsUtils(stream=None):
    stream = stream or sys.stdout
    stream.write("\n A few utilities to handle FITS files are available in BBarolo \n\n"
                 "   Usage: BBarolo --utilityname [options] \n\n"
                 " where 'utilityname' is one of the following: \n\n")
    for name, descr in FITSUTILS_LIST:
        stream.write(f"   {name:<10s}:  {descr}\n")
    stream.write("\n")


def helpscreen(stream=None):
    stream = stream or sys.stdout
    m = 20

    def opt(name, lines, example=None):
        s = f"{name:<{m}}{lines[0]}\n"
        for l in lines[1:]:
            s += " " * m + l + "\n"
        if example:
            s += "\n" + " " * m + f"Example:      {example}\n"
        return s + "\n\n"
    stream.write(
        "\n             ____                                _____        __           \n"
        "   /\\      .'    /\\                             /.---.\\      (==)       \n"
        "  |K -----;     |  |                           | ````` |     |~~|          \n"
        "   \\/      '.____\\/                             \\     /      |  |       \n"
        "                   °.                            `-.-'       |()|          \n"
        "        __          °.                             |        /`  `\\        \n"
        "   __  |_/         .°.  _________________        __|__     /      \\       \n"
        "   \\_|\\\\ _          .°.                         `-----`   ;  ____  ;    \n"
        "      _\\(_)_           BBarolo quick guide                ||`    `||      \n"
        "     (_)_)(_)_      _________________________     ___     ||BAROLO||       \n"
        "    (_)(_)_)(_)                                  ||  \\\\   ||      ||     \n"
        "     (_)(_))_)                                   ||__// _ || 2015 ||       \n"
        "      (_(_(_)                                    ||  \\\\   | \\____/ |    \n"
        "       (_)_)                                     ||__//   |        |       \n"
        "        (_)                                               \\_.-\"\"-._/    \n "
        "                                                          `\"\"\"\"\"\"`   \n\n\n "
        "  Usage:           BBarolo <option> [arg] \n\n Options: \n\n")
    stream.write(opt("   -f, --fitsfile", ["BBarolo runs the 3DFIT task in automatic mode ",
                                           "using default parameters. [arg] is required   ",
                                           "and represents a FITS file to be analysed.    ",
                                           "BBarolo looks for sources, estimates initial  ",
                                           "parameters and fits a 3D tilted-ring model. "],
                     "BBarolo -f myfitsfile.fits"))
    stream.write(opt("   -p, --paramfile", ["BBarolo runs with a parameter file. [arg] is ",
                                            "required and is a text file with a list of   ",
                                            "parameters. Additional KEY=value parameters ",
                                            "on the command line override the file. "],
                     "BBarolo -p param.par       "))
    stream.write(opt("   -c, --cline", ["Read all parameters from the command line. ",
                                        "Input parameters are given as a list of ",
                                        "parametername=value. "],
                     "BBarolo -c fitsfile=file.fits 3dfit=true"))
    stream.write(opt("   -d, --defaults", ["Print on the screen a list all available par-",
                                           "ameters and their default values. A task name",
                                           "[arg] can optionally be given to print only  ",
                                           "parameters relevant for that task. "],
                     "BBarolo -d 3DFIT       "))
    stream.write(opt("   -l, --list", ["Print a list of available tasks.       "]))
    stream.write(opt("   -t, --template", ["Create a template input parameter file named ",
                                           "param.par for the 3DFIT task.        "]))
    stream.write(opt("   -v, --version", ["Version info    "]))
    stream.write(opt("   --fitsutils", ["List the available FITS utilities.   "]))


def versionInfo(stream=None):
    import numba, scipy, astropy
    stream = stream or sys.stdout
    stream.write(f"\nThis is BBarolo version {__version__}\n"
                 f"\nExecutable: {sys.argv[0]}\n"
                 f"Python {sys.version.split()[0]}, numpy {np.__version__}, scipy {scipy.__version__}, "
                 f"astropy {astropy.__version__}, numba {numba.__version__}\n\n")


# --------------------------------------------------------------------------
# Options
# --------------------------------------------------------------------------
FITSUTILS = ("listhead", "modhead", "remhead", "fitscopy", "fitsarith", "jybeam2k", "k2jybeam")


def getopts(argv):
    """Port of Param::getopts. Returns a checked Param or None if nothing to run."""
    from . import fitsutils
    if len(argv) < 1:
        helpscreen()
        return None
    par = Param()
    ok = False
    i = 0
    rest = []
    # (--paramfile=file, --fitsfile=file, --defaults=TASK: as --paramfile file; they were taken as KEY=value and
    # ignored, and BBarolo exited with nothing done)
    argv = [x for a in argv for x in (a.split("=", 1) if a.startswith("--") and "=" in a
                                      and a.split("=", 1)[0] in ("--paramfile", "--fitsfile", "--defaults") else [a])]
    while i < len(argv):
        a = argv[i]
        nxt = argv[i + 1] if i + 1 < len(argv) else None
        if a.startswith("--") and a[2:] in FITSUTILS:
            u = a[2:]
            args = argv[i + 1:]
            if u == "jybeam2k":
                fitsutils.ConvertFluxUnits(args, "j2k")
            elif u == "k2jybeam":
                fitsutils.ConvertFluxUnits(args, "k2j")
            else:
                getattr(fitsutils, u)(args)
            return None
        if a in ("-p", "--paramfile"):
            if nxt is None:
                raise BBaroloError("Option -p requires a parameter file: BBarolo -p paramfile")
            if not par.readParamFile(nxt):
                raise BBaroloError(f"Could not open the parameter file {nxt}.")
            ok = True
            i += 2
            continue
        if a in ("-f", "--fitsfile"):
            if nxt is None:
                raise BBaroloError("Option -f requires a FITS file: BBarolo -f fitsfile")
            # (as keywords: recorded in the parameter file of the run, which then runs the same with -p)
            _kw.lookup("FITSFILE").apply(par, nxt)
            _kw.lookup("AUTO").apply(par, "true")
            ok = True
            i += 2
            continue
        if a in ("-d", "--defaults"):
            if nxt is not None and not nxt.startswith("-") and "=" not in nxt:
                par.printDefaults(sys.stdout, nxt.upper())
                i += 2
            else:
                par.printDefaults(sys.stdout)
                i += 1
            continue
        if a in ("-c", "--cline"):
            ok = True
        elif a in ("-t", "--template"):
            par.createTemplate()
        elif a in ("-l", "--list"):
            listTasks()
        elif a in ("-v", "--version"):
            versionInfo()
        elif a in ("-q", "--quote"):
            n = int(nxt) if nxt is not None and nxt.isdigit() else 1
            for _ in range(n):
                print(f"\n {randomQuoting()}\n")
            return None
        elif a in ("-F", "--fitsutils"):
            listFitsUtils()
        elif a in ("-U", "--upgrade"):
            print("BBarolo automated upgrade is not implemented yet! Try again in next versions...")
        elif a in ("-h", "--help"):
            helpscreen()
        elif "=" in a and not a.startswith("-"):
            rest.append(a)
        else:
            raise BBaroloError(f"Option {a} is unacceptable. Try BBarolo -h for help.")
        i += 1
    if not ok:
        return None
    for s in rest:
        par.readParamCL(s)
    if not par.checkPars():
        raise BBaroloError("Unacceptable input parameters (see the messages above).")
    return par


# --------------------------------------------------------------------------
# Tasks: the engine is in tasks.py (shared with the Python classes)
# --------------------------------------------------------------------------
from .tasks import BBcore, run_steps, run_par  # noqa: E402,F401


def _set_process_name(name="BBarolo"):
    """Show the process as ``name`` instead of ``python`` in top/ps (best effort, never fails).

    With the optional ``setproctitle`` package the whole command line shown by ``ps``/``top -c``
    changes too; otherwise on Linux only the short name (what ``top`` shows by default) changes.
    Worker processes forked later (THREADS > 1) inherit the name.
    """
    try:
        import setproctitle
        setproctitle.setproctitle(" ".join([name] + sys.argv[1:]))
        return
    except Exception:
        pass
    if sys.platform.startswith("linux"):
        try:
            import ctypes
            libc = ctypes.CDLL(None, use_errno=True)
            libc.prctl(15, ctypes.c_char_p(name.encode()[:15]), 0, 0, 0)   # 15 = PR_SET_NAME
        except Exception:
            pass


def main(argv=None):
    """Entry point of the BBarolo command (returns the exit code: 0, or 1 if something failed)."""
    _set_process_name()
    try:
        return _main(argv)
    except BBaroloError as e:
        sys.stdout.flush()
        sys.stderr.write(f"\n BBAROLO ERROR: {str(e).strip()}\n\n")
        return 1
    except BrokenPipeError:
        # Output piped to a program that closed it early (e.g. "BBarolo -d | head")
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, sys.stdout.fileno())
        return 1


def _main(argv=None):
    # plots are written to files only (e.g. the corner plot of BAYESIAN3DFIT): no GUI backend
    os.environ.setdefault("MPLBACKEND", "Agg")
    argv = sys.argv[1:] if argv is None else list(argv)
    par = getopts(argv)
    if par is None:                         # (help, version, ... or no arguments: the help screen)
        return 0 if argv else 1
    ok, _ = run_par(par)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
