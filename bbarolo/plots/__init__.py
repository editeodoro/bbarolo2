"""
The plots of BBarolo as Python functions: each draws a plot from arrays and returns a matplotlib Figure, writing
nothing. The plotting scripts written in <outfolder>/plotscripts/ contain a copy of the same functions, so that the
plots can be edited and made again without bbarolo. Result.plot_parameters(), plot_chanmaps() etc. draw them from
the outputs of a task in memory.

3DFIT and BAYESIAN3DFIT (fit3d.py): plot_parameters, plot_asymdrift, plot_chanmaps, plot_pvs, plot_pvslices,
plot_kinmaps; read_rings, read_cube, read_map and read_columns read the output files for them.
"""

from .utils import StyledFigure
from .fit3d import (plot_parameters, plot_asymdrift, plot_chanmaps, plot_pvs, plot_pvslices, plot_kinmaps,
                    read_rings, read_cube, read_map, read_columns)

__all__ = ["StyledFigure", "plot_parameters", "plot_asymdrift", "plot_chanmaps", "plot_pvs", "plot_pvslices",
           "plot_kinmaps", "read_rings", "read_cube", "read_map", "read_columns"]
