#!/usr/bin/env python3
"""
Launcher for the pure-Python BBarolo that works without installing the package:

    python /path/to/pyBBarolo_pure/run_bbarolo.py -p param.par

or, with the shell launcher next to it, /path/to/pyBBarolo_pure/bin/BBarolo -p param.par
(it always picks the bbarolo package next to this file).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bbarolo.cli import main  # noqa: E402

sys.exit(main())
