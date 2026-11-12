"""
Helpers of the plots of BBarolo. The source of this module (with that of pv_path and pv_sample of
pvslice.py) is also written as plot_utils.py next to the plotting scripts, so that the scripts run
without bbarolo: the functions here must need only numpy, matplotlib (and scipy, for pv_sample).
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from ..pvslice import pv_path, pv_sample


class StyledFigure(Figure):
    """A matplotlib Figure with its own rcParams (style, a dict): they are in force while the figure is made
    (with matplotlib.rc_context) and whenever it is drawn (shown or saved), whatever the rcParams of the session.
    Make it with plt.figure(FigureClass=StyledFigure, style=style, ...)."""

    def __init__(self, *args, style=None, **kwargs):
        self.bb_style = dict(style or {})
        with mpl.rc_context(self.bb_style):
            super().__init__(*args, **kwargs)

    def draw(self, renderer):
        with mpl.rc_context(self.bb_style):
            return super().draw(renderer)

    def get_tightbbox(self, *args, **kwargs):
        with mpl.rc_context(self.bb_style):
            return super().get_tightbbox(*args, **kwargs)


def defineaxis(nrows, ncols, xlen, ylen, xsep=0, ysep=0, fig_width=8.27, fig_heigth=11.69, fig=None):
    """A grid of nrows x ncols axes of size xlen x ylen (fractions of the figure width; the height is scaled by
    the aspect ratio of the figure), separated by xsep and ysep, in fig (a new figure if None). Returns the figure
    and the axes (an array, squeezed)."""
    if not fig:
        fig = plt.figure(figsize=(fig_width, fig_heigth))
    fig_width, fig_heigth = fig.get_size_inches()
    fig_ratio = fig_width / fig_heigth

    ax = []
    for i in range(nrows):
        row = []
        y = -(i * (ylen + ysep) * fig_ratio)
        for j in range(ncols):
            x = j * (xlen + xsep)
            row.append(fig.add_axes([x, y, xlen, ylen * fig_ratio]))
        ax.append(row)
    return fig, np.squeeze(ax)


def calculate_padding(xsize, ysize):
    """The padding (before, after) along x and y that makes an image of xsize x ysize pixels square."""
    n = max(ysize, xsize)
    pad_y = ((n - ysize) // 2, n - ysize - (n - ysize) // 2)
    pad_x = ((n - xsize) // 2, n - xsize - (n - xsize) // 2)
    return (pad_x, pad_y)


def build_path(x0, y0, rad_pix, pa_deg, nx, ny, oversampling=1):
    """The points of a PV slice through (x0, y0), along the position angles pa_deg (degrees) of the rings of radii
    rad_pix (pixels), in a field of nx x ny pixels: s, x, y (see pv_path)."""
    return pv_path(x0, y0, pa_deg, radii=rad_pix, shape=(ny, nx), step=1.0 / oversampling)


def extract_pv(array, x, y, order=1, width=1, oversampling=1):
    """The PV slice of the cube array along the points x, y (see pv_sample)."""
    return pv_sample(array, x, y, width=width, order=order, zoversample=oversampling)
