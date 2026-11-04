#!/usr/bin/env python3
"""
GalMod3D: model cubes from 3D fields of the density and velocity of the gas, on the grid of NGC 2403.
Run it from the folder containing examples/ (the parent of this repository):

    python bbarolo2/examples/galmod3d.py

Three models, each written as a FITS file in ./output/galmod3d/ (bbarolo.Maps or bbarolo.PVSlice
work on them, e.g. bbarolo.Maps(hdu, totalmap=True, velocitymap=True).run()):
  1. an axisymmetric thick disk, functions of (R, z), that flares and lags in rotation above the plane;
  2. a disk with a bar and two spiral arms, functions of (R, phi, z), with streaming motions;
  3. Cartesian fields (x, y, z) on a grid, as from a simulation: a disk warped beyond 600 arcsec.
The total flux of each model is that of the data in its mask (flux="measured").
"""

import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))          # the bbarolo package next to this folder

from bbarolo import GalMod3D                       # noqa: E402

fitsname = "./examples/ngc2403.fits"
out = "./output/galmod3d/"
os.makedirs(out, exist_ok=True)
geo = dict(xpos=77, ypos=77, vsys=132.8, inc=60, pa=123.7, flux="measured", threads=8)


def vcirc(R):
    return 130 * np.tanh(R / 60.)                # km/s, R in arcsec


# 1. A flaring thick disk that lags in rotation: functions of (R, z)
def dens1(R, z):
    h = 10 + R / 50.                              # scale height (arcsec), growing with radius
    return np.exp(-R / 400.) * np.exp(-np.abs(z) / h) / h


m1 = GalMod3D(fitsname, coords="Rz", dens=dens1, vphi=lambda R, z: vcirc(R) - 0.5 * np.abs(z), vdisp=10.,
              zmax=80, **geo)
m1.model().writeto(out + "thick_lagging_disk.fits", overwrite=True)


# 2. A bar and two spiral arms: functions of (R, phi, z); phi from the receding major axis (radians)
def dens2(R, phi, z):
    bar = 1 + 0.8 * np.exp(-(R / 150.) ** 2) * np.cos(2 * (phi - 0.5))
    arms = 1 + 0.5 * np.cos(2 * (phi - np.log(np.maximum(R, 1.) / 100.) / np.tan(np.radians(20))))
    return np.exp(-R / 400.) * bar * arms * np.exp(-z * z / 50.)


def vphi2(R, phi, z):
    return vcirc(R) * (1 + 0.1 * np.exp(-(R / 150.) ** 2) * np.cos(2 * (phi - 0.5)))


m2 = GalMod3D(fitsname, coords="Rphiz", dens=dens2, vphi=vphi2,
              vr=lambda R, phi, z: 15 * np.exp(-(R / 150.) ** 2) * np.sin(2 * (phi - 0.5)), vdisp=8., zmax=30, **geo)
m2.model().writeto(out + "bar_spirals.fits", overwrite=True)

# 3. Cartesian fields on a grid (e.g. from a simulation): a disk warped beyond 600 arcsec (line of nodes
# along x), rotating in its plane (the tilt of the warp is neglected in the velocities)
x = np.linspace(-2000, 2000, 401)
z = np.linspace(-100, 100, 41)
X, Y, Z = np.meshgrid(x, x, z, indexing="ij")
R = np.hypot(X, Y)
zwarp = 0.1 * np.maximum(R - 600, 0) * Y / np.where(R > 0, R, 1.)
rho = np.exp(-R / 400.) * np.exp(-(Z - zwarp) ** 2 / 50.)
v = vcirc(R) / np.where(R > 0, R, 1.)
m3 = GalMod3D(fitsname, coords="xyz", dens=((x, x, z), rho), vx=((x, x, z), -v * Y), vy=((x, x, z), v * X),
              vdisp=10., zmax=100, **geo)
m3.model().writeto(out + "warped_disk.fits", overwrite=True)
print(f"Models written in {out}")
