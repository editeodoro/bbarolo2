#!/usr/bin/env python3
"""
Example: Bayesian 3D fit of NGC 2403 with the pure-Python Bayesian3DFIT.

Run from the BBarolo folder (the one containing examples/ngc2403.fits):

    python pyBBarolo_pure/examples/bayesian_example.py [--threads 4]

It fits the rotation velocity of each of 4 rings, a single velocity dispersion, a
single inclination and the noise per voxel (7 parameters), then prints the posterior
statistics and writes the best model, the usual BBarolo plots, a corner plot and the
radial profiles in ./output/bayes_ngc2403/. The sampling takes a few minutes with 2
processes; the run time grows quickly with the number of free parameters.
"""

import os
import sys
import argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))                  # the pure-Python bbarolo

import matplotlib                                           # noqa: E402
matplotlib.use("Agg")
from bbarolo import Bayesian3DFIT                        # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--cube", default=os.path.join(os.path.dirname(os.path.dirname(HERE)), "examples", "ngc2403.fits"))
ap.add_argument("--threads", type=int, default=os.cpu_count() or 1)
ap.add_argument("--outfolder", default="./output/bayes_ngc2403/")
args = ap.parse_args()

# Rings (initial values; the parameters that are not fitted stay fixed) and any other
# BBarolo parameter. ANALYTIC models give a likelihood without Monte Carlo noise.
f = Bayesian3DFIT(args.cube, radii=np.arange(120, 500, 120), xpos=77, ypos=77, vsys=132.8, vrot=100, vdisp=10,
                  inc=60, pa=123.7, z0=10, mask="SEARCH", outfolder=args.outfolder, plots=1,
                  galmodtype="ANALYTIC")

# Priors: scipy.stats distributions, given by name and parameters...
f.priors['inc'] = dict(name='uniform', loc=45, scale=30)          # 45-75 deg
f.priors['vdisp'] = dict(name='uniform', loc=0, scale=40)         # 0-40 km/s
# ...or directly as frozen distributions: f.prior_distr['inc'] = scipy.stats.norm(60, 5)

# Free parameters: 'vrot' (one per ring), '_single' (one for all rings), '_func' (function of R).
# The likelihood uses the measured noise per voxel and the measured correlation of the noise
# (each independent noise measurement counts once; like_kwargs=dict(noisecorr='beam') counts
# each beam once).
# A tilted-ring model does not fit NGC 2403 down to the noise (reduced chi-square >> 1), so
# the noise per voxel 'sigma' is fitted too: the credible intervals then include the misfit
# of the model. Without 'sigma', they would be formal errors of a few 0.01 km/s.
# The likelihood compares the model with the data, not masked, in the mask dilated by one
# beam and 3 channels (the option likemask='dilated' of like_kwargs), and the density profile
# of the model is the one whose beam-convolved model reproduces the total intensity map: on
# mock cubes, VDISP and INC are then unbiased (see the README).
freepar = ['vrot', 'vdisp_single', 'inc_single', 'sigma']

f.run(freepar=freepar, threads=args.threads, method='nautilusbb',
      sampler_kwargs=dict(n_live=500), run_kwargs=dict(f_live=0.01))

f.print_stats()
f.save_results(os.path.join(args.outfolder, "bayes_results.npy"))
f.write_bestmodel(plots=True)
f.plot_profiles().savefig(os.path.join(args.outfolder, "bayes_profiles.pdf"), bbox_inches='tight')
try:
    f.corner_plot().savefig(os.path.join(args.outfolder, "bayes_corner.pdf"), bbox_inches='tight')
except ImportError as e:
    print(e)
print(f"\nOutputs written in {args.outfolder}")
