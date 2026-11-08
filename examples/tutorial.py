#!/usr/bin/env python3
"""
The Python interface of bbarolo: the examples of examples/test/pyBB_test.py written
with the new classes. Run it from the BBarolo folder (the one containing examples/):

    python pyBBarolo_pure/examples/tutorial.py

Every class takes the cube and any BBarolo parameter (the keywords of a parameter
file, case-insensitive), runs the same code as the command line and returns a Result
with the outputs (astropy HDUs and Tables). bbarolo.help('KEYWORD') or
bbarolo.help('TASK') lists the parameters and their defaults.
"""

import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))          # the bbarolo package next to this folder

import bbarolo                                       # noqa: E402
from bbarolo import FitMod3D, GalMod, Ellprof, FitMod2D, Search, GalWind   # noqa: E402

fitsname = "./examples/ngc2403.fits"
out = "./output/tutorial/"

### 3D FIT #####################################################################
# The rings (single values or one value per ring) and the options, all at once.
# Parameters that are not given are estimated or take the parameter-file defaults.
f3d = FitMod3D(fitsname, radii=np.arange(15, 450, 30), xpos=77, ypos=77, vsys=132.8, vrot=120, vdisp=8,
               vrad=0, z0=10, inc=60, pa=123.7, mask="SEARCH", free="VROT VDISP", wfunc=2,
               ltype=2, outfolder=out + "3dfit", beam=(60, 60, -80), threads=4)
print(f3d)                      # the parameters that are set
res = f3d.run()                 # computes everything in memory: nothing is written
print(res)                      # what the Result contains
f3d.write()                     # the usual outputs (and plots) in outfolder, as the command line
print(res.rings["radii", "vrot", "vdisp"])
res.model.writeto("ngc2403_model.fits", overwrite=True)

### GALMOD #####################################################################
# norm='NONE': the model with the given DENS (the default AZIM normalizes it to the data).
# Each ring is an annulus centred on its radius (RINGMODE=CENTRED, as the rings of 3DFIT).
gm = GalMod(fitsname, radii=np.arange(15, 1200, 30), xpos=74, ypos=74, vsys=132.8, vrot=120, vdisp=8, z0=10,
            inc=60, pa=123.7, dens=1, ltype=1, iseed=-2, norm="NONE", verbose=False, threads=4)
mymodel = gm.run().model      # the model, in memory, smoothed to the beam of the data (SM=true)
mymodel.writeto("awesome_model.fits", overwrite=True)
# gm.write() would write the model, maps, PV slices, ... in the output folder

### ELLPROF ####################################################################
el = Ellprof(fitsname, radii=np.arange(15, 1200, 30), xpos=74, ypos=74, inc=60, pa=123.7, mask="SEARCH",
             outfolder=out + "ellprof", verbose=False)
prof = el.run().profile
print(prof["radii", "himassdens"])

### 2DFIT ######################################################################
rm = FitMod2D(fitsname, radii=np.arange(15, 1200, 30), xpos=74, ypos=74, vsys=132.8, vrot=120, inc=60,
              pa=123.7, wfunc=1, free="VROT INC", outfolder=out + "2dfit", threads=4, verbose=False)
rings = rm.run().rings
print(rings["radii", "vrot", "inc"])

### SEARCH #####################################################################
s = Search(fitsname, flaggrowth=True, outfolder=out + "search", threads=4)
r = s.run()
print(r.table)

### GALWIND ####################################################################
gw = GalWind(fitsname, xpos=74, ypos=74, pa=123.7, inc=60, vdisp=20, dens=1.25, vsys=132.8, vwind=100,
             openang=100, htot=500, threads=4, verbose=False)
wmodel = gw.run().model
wmodel.writeto("wind_model.fits", overwrite=True)
wmodel.data = wmodel.data + mymodel.data
wmodel.writeto("wind+galaxy.fits", overwrite=True)

### A whole parameter file, as the command line #################################
results = bbarolo.run("examples/test/n2403_moment.par", outfolder=out + "moment", verbose=False)
print(results["MAPS"].maps.keys(), results["PVSLICE"].pv.data.shape)
