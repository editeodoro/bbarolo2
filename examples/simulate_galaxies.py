#!/usr/bin/env python3
"""
Simulate an emission-line datacube of a disk galaxy with SimulatedGalaxyCube (bbarolo):
the new version of examples/pyBB_simulate_galaxies.py. The model is built in memory
(no BBarolo executable, no temporary files) and written in a 'models' folder.

    python pyBBarolo_pure/examples/simulate_galaxies.py
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))          # the bbarolo package next to this folder

from bbarolo import SimulatedGalaxyCube             # noqa: E402

# General parameters for datacube
xysize, vsize = 51, 64          # Number of pixels/channels
pixsize   = 20                  # Size of pixels (arcsec)
chwidth   = 10                  # Channel width (km/s)
beamFWHM  = 60                  # Beam size (arcsec)
modname   = 'model1'            # Name of the model
noiserms  = 0.03                # RMS noise in Jy/beam

# Basic parameters of the model
radmax  = 350
radii   = np.arange(0, radmax, pixsize)
dens    = 50*np.exp(-radii/400-100/(0.5*radii+100))
vrot    = 2./np.pi*200.*np.arctan(radii/10.)
vdisp   = np.full(len(radii), 10.)
pa      = np.full(len(radii), 0.)
inc     = np.full(len(radii), 75.)
z0      = np.full(len(radii), 85.)

# Below if we wanna define a warp or a flare (these are linear for example)
#rstart = 150
#pa[radii>rstart] = -(radii[radii>rstart]-rstart)/20.+pa[0]
#inc[radii>rstart] = -(radii[radii>rstart]-rstart)/80.+inc[0]
#z0[radii>rstart] = (radii[radii>rstart]-rstart)/4. + z0[0]

# Initializing a SimulatedGalaxyCube instance (the template cube)
s = SimulatedGalaxyCube(axisDim=[xysize, xysize, vsize],
                        cdelts=[-pixsize/3600., pixsize/3600., chwidth],
                        beam=beamFWHM/3600., bunit='JY/BEAM', obj=modname)

# Setting up galaxy parameters (if parameters are not given, they are random!)
s.define_galaxy(radii=radii, vsys=0, vdisp=vdisp, inc=inc, pa=pa,
                dens=dens, z0=z0, vrot=vrot, warpinc=False, warppa=False)

# Building the model: s.model is an astropy HDU (s.model_nonoise without noise).
# Any GALMOD parameter can be added, e.g. ltype=2, galmodtype='ANALYTIC'; seed=... for the noise.
print(f"Simulating {modname}...", flush=True, end='')
s.run(smooth=True, noise=noiserms)
print(" Done!")

# Writing the model and the galaxy parameters in the 'models' folder
fileout = s.write_model(f"models/{modname}.fits", params=True)
print(f"Model written in {fileout} (parameters in models/{modname}_params.txt). See you soon!")
