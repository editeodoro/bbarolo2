"""
BAYESIANWIND on a mock wind with a clumpy emission.

The mock (performance/wind_mocks.py) is made with GalMod3D: 100 Gaussian clumps at random positions in two cones
(60 degrees of full opening angle, 200 arcsec long, inclined by 60 degrees, PA 30), a radial wind of 200 km/s with a
dispersion of 20 km/s, and noise smoothed by the beam, peak S/N 10. The fit starts from a wrong guess of the
geometry, as a real one would, and samples the velocity of the wind, the inclination and the opening angle; the
mean emissivity, clumpiness, ratio of the cones and radial slope of the prior of the emission (EMSCALE, EMCLUMP,
EMRATIO, EMSLOPE) are fitted with them.

    python3 examples/wind_example.py [outfolder] [threads]
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "performance"))

from wind_mocks import mock_wind          # noqa: E402
from bbarolo import BayesianWind          # noqa: E402

out = sys.argv[1] if len(sys.argv) > 1 else "output/wind_example/"
threads = int(sys.argv[2]) if len(sys.argv) > 2 else 8

hdu, truth = mock_wind(kind="clumps", seed=1, snr=10)
f = BayesianWind(hdu, xpos=23.5, ypos=23.5, vsys=0, inc=55, pa=30, openang=55, conelen=200, vwind=180, vdisp=20,
                 outfolder=out)
f.run(freepar=["vwind_single", "inc", "openang"], threads=threads)
f.print_stats()
print(f" truth: VWIND {truth['vwind'][0]:g} km/s, INC {truth['inc']:g} deg, OPENANG {truth['openang']:g} deg")
f.write(plots=1)
