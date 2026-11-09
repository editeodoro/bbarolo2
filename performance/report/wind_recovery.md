# BAYESIANWIND: recovery of patchy mock winds

Mocks: performance/wind_mocks.py (GalMod3D, an engine independent of the fit; clumps or a log-normal field in the cones; noise smoothed by the beam, peak S/N 10 unless noted). Fits: BayesianWind with nautilus (300 live points, 2000 effective samples), the initial wind offset from the truth (INC -5, PA +5, OPENANG -5, VWIND -20 km/s, VDISP +5 km/s) and the default priors around it. For each parameter: the mean bias (posterior median - truth), the rms of the pulls (bias / half-width of the 68% interval: 1 if the intervals are right) and the coverage of the 68% intervals over the seeds. EMSCALE, EMCLUMP and EMRATIO have no true value and are left out.

## The configurations (emission cells, marginal likelihood: the defaults)

| configuration | seeds | vwind | vdisp | vrot | inc | pa | openang | openin |
|---|---|---|---|---|---|---|---|---|
| filled | 4 | -7.4 (2.0, 0.25) | +3.1 (1.4, 0.50) |  | -1.1 (1.6, 0.50) | +0.1 (1.2, 0.50) | -2.8 (2.1, 0.00) |  |
| hollow | 4 | -4.7 (2.0, 0.25) | +2.3 (1.4, 0.25) |  | -0.4 (1.0, 0.75) | -0.2 (1.5, 0.50) | -2.9 (2.3, 0.00) | +3.3 (1.8, 0.00) |
| single | 4 | -5.5 (1.8, 0.00) | +3.5 (1.2, 0.25) |  | -1.7 (1.3, 0.00) | -0.7 (1.2, 0.50) | -4.7 (3.2, 0.25) |  |
| lowinc | 4 | +0.1 (0.8, 0.75) | +0.1 (0.8, 0.75) |  | -0.3 (1.3, 0.50) | -0.2 (0.9, 0.75) | -4.1 (2.5, 0.00) |  |
| edgeon | 4 | -27.6 (2.8, 0.00) | +7.8 (3.2, 0.00) |  | -1.0 (1.4, 0.75) | +0.6 (1.2, 0.75) | -1.1 (0.7, 1.00) |  |
| accel | 4 | -8.2 (1.8, 0.44) | +9.3 (3.4, 0.00) |  | -1.3 (1.6, 0.50) | +0.3 (1.1, 0.75) | -3.9 (3.2, 0.00) |  |
| rotation | 4 | -9.1 (2.1, 0.25) | +2.9 (1.5, 0.50) | +1.7 (1.9, 0.75) | -1.5 (1.7, 0.50) | +0.5 (0.9, 0.75) | -3.5 (2.5, 0.00) |  |
| lognormal | 4 | -0.6 (1.4, 0.50) | +4.3 (1.5, 0.25) |  | -0.1 (1.6, 0.50) | -0.2 (0.7, 0.75) | -0.8 (0.5, 1.00) |  |
| lowsnr | 4 | -7.4 (1.3, 0.50) | +4.2 (1.0, 0.50) |  | -1.4 (1.3, 0.50) | +0.6 (0.8, 0.75) | -2.5 (1.1, 0.50) |  |

Each entry: bias (rms pull, coverage). All the configurations: vwind: pull rms 1.84, coverage 0.35; vdisp: pull rms 1.93, coverage 0.33; vrot: pull rms 1.95, coverage 0.75; inc: pull rms 1.43, coverage 0.50; pa: pull rms 1.09, coverage 0.67; openang: pull rms 2.25, coverage 0.31; openin: pull rms 1.84, coverage 0.00.

## The descriptions of the emission (option emission, marginal or profile)

| configuration | basis | seeds | min per fit | vwind | vdisp | vrot | inc | pa | openang |
|---|---|---|---|---|---|---|---|---|---|
| filled | cells | 4 | 6.5 | -7.4 (2.0, 0.25) | +3.1 (1.4, 0.50) |  | -1.1 (1.6, 0.50) | +0.1 (1.2, 0.50) | -2.8 (2.1, 0.00) |
| filled | cells-profile | 4 | 2.7 | +22.8 (2.6, 0.00) | -18.0 (10.5, 0.00) |  | +3.5 (2.9, 0.00) | +4.2 (2.6, 0.50) | +19.9 (6.0, 0.00) |
| filled | shells | 4 | 1.1 | -7.7 (2.3, 0.50) | +4.7 (1.7, 0.25) |  | -1.7 (2.0, 0.25) | +0.6 (1.8, 0.50) | -3.5 (3.3, 0.00) |
| filled | spaxel | 4 | 0.8 | -5.9 (2.3, 0.25) | +2.9 (1.5, 0.50) |  | -1.2 (2.8, 0.25) | +0.7 (1.7, 0.50) | -3.4 (3.2, 0.00) |
| filled | voxel | 4 | 1.5 | +82.5 (25.6, 0.00) | +54.8 (295.4, 0.00) |  | +2.6 (34.6, 0.00) | +0.2 (8.7, 0.50) | +24.9 (259.6, 0.00) |
| filled | cells22 | 4 | 10.3 | -7.2 (1.9, 0.25) | +3.2 (1.5, 0.50) |  | -1.2 (1.6, 0.50) | +0.1 (1.3, 0.50) | -2.9 (2.2, 0.00) |
| filled | cells-df | 4 | 6.4 | -7.7 (1.4, 0.50) | +4.0 (1.3, 0.50) |  | -1.7 (1.8, 0.50) | +0.3 (1.0, 0.75) | -3.6 (2.1, 0.00) |
| hollow | cells | 4 | 8.2 | -4.7 (2.0, 0.25) | +2.3 (1.4, 0.25) |  | -0.4 (1.0, 0.75) | -0.2 (1.5, 0.50) | -2.9 (2.3, 0.00) |
| hollow | cells-profile | 4 | 4.7 | +21.5 (3.2, 0.00) | -18.1 (11.1, 0.00) |  | +4.2 (2.0, 0.25) | +3.1 (2.9, 0.75) | +20.8 (6.1, 0.00) |
| hollow | shells | 4 | 1.6 | -3.8 (3.1, 0.50) | +2.1 (1.9, 0.50) |  | -0.2 (1.1, 0.50) | -0.1 (1.9, 0.25) | -2.3 (2.0, 0.00) |
| hollow | spaxel | 4 | 1.4 | -3.5 (2.1, 0.25) | +1.6 (1.4, 0.50) |  | -0.2 (1.4, 0.00) | -0.1 (1.7, 0.25) | -2.6 (2.3, 0.25) |
| hollow | voxel | 4 | 1.4 | +87.1 (17.4, 0.00) | +54.8 (307.4, 0.00) |  | +3.0 (17.9, 0.00) | -10.5 (15.9, 0.50) | +24.9 (240.7, 0.00) |
| lognormal | cells | 4 | 6.3 | -0.6 (1.4, 0.50) | +4.3 (1.5, 0.25) |  | -0.1 (1.6, 0.50) | -0.2 (0.7, 0.75) | -0.8 (0.5, 1.00) |
| lognormal | cells22 | 4 | 9.6 | -0.7 (1.4, 0.50) | +4.6 (1.6, 0.25) |  | -0.2 (1.6, 0.50) | -0.2 (0.7, 0.75) | -0.8 (0.7, 0.75) |
| edgeon | cells | 4 | 7.3 | -27.6 (2.8, 0.00) | +7.8 (3.2, 0.00) |  | -1.0 (1.4, 0.75) | +0.6 (1.2, 0.75) | -1.1 (0.7, 1.00) |
| edgeon | cells22 | 4 | 11.0 | -26.3 (2.9, 0.00) | +8.1 (3.3, 0.00) |  | -1.0 (1.5, 0.75) | +0.6 (1.3, 0.75) | -1.4 (0.9, 0.50) |
| edgeon | cells-df | 4 | 7.2 | -28.8 (2.1, 0.00) | +9.0 (2.1, 0.00) |  | -1.3 (1.3, 0.50) | +0.4 (1.0, 0.75) | -2.5 (1.2, 0.25) |

cells-profile: the maximum over the emission (no Occam factor); voxel: the weighting of MoKA-3D (the model is the data wherever the uniform cone reaches); cells22: cells of 22 arcsec instead of the beam (30 arcsec); cells-df: the model error datafrac = 0.3.

## Details

### accel, cells (4 seeds, 9.8 min per fit)

| parameter | truth | bias | scatter | pull rms | coverage 68% |
|---|---|---|---|---|---|
| vwind1 | 100 | +3.63 | 20.2 | 0.90 | 0.75 |
| vwind2 | 160 | +4.53 | 11.8 | 0.74 | 0.75 |
| vwind3 | 220 | -17.9 | 19.7 | 2.05 | 0.25 |
| vwind4 | 280 | -22.8 | 7.84 | 2.68 | 0.00 |
| vdisp | 20 | +9.33 | 2.16 | 3.37 | 0.00 |
| inc | 60 | -1.26 | 1.44 | 1.58 | 0.50 |
| pa | 30 | +0.341 | 0.873 | 1.14 | 0.75 |
| openang | 60 | -3.94 | 0.309 | 3.24 | 0.00 |

### edgeon, cells (4 seeds, 7.3 min per fit)

| parameter | truth | bias | scatter | pull rms | coverage 68% |
|---|---|---|---|---|---|
| vwind | 200 | -27.6 | 7.85 | 2.85 | 0.00 |
| vdisp | 20 | +7.78 | 2.74 | 3.24 | 0.00 |
| inc | 85 | -0.995 | 0.96 | 1.44 | 0.75 |
| pa | 30 | +0.559 | 0.743 | 1.18 | 0.75 |
| openang | 60 | -1.1 | 0.574 | 0.72 | 1.00 |

### edgeon, cells-df (4 seeds, 7.2 min per fit)

| parameter | truth | bias | scatter | pull rms | coverage 68% |
|---|---|---|---|---|---|
| vwind | 200 | -28.8 | 5.57 | 2.06 | 0.00 |
| vdisp | 20 | +9 | 3.28 | 2.10 | 0.00 |
| inc | 85 | -1.31 | 0.957 | 1.32 | 0.50 |
| pa | 30 | +0.372 | 0.773 | 0.97 | 0.75 |
| openang | 60 | -2.48 | 1.43 | 1.22 | 0.25 |

### edgeon, cells22 (4 seeds, 11.0 min per fit)

| parameter | truth | bias | scatter | pull rms | coverage 68% |
|---|---|---|---|---|---|
| vwind | 200 | -26.3 | 7.64 | 2.88 | 0.00 |
| vdisp | 20 | +8.1 | 2.72 | 3.31 | 0.00 |
| inc | 85 | -0.989 | 0.956 | 1.45 | 0.75 |
| pa | 30 | +0.58 | 0.759 | 1.31 | 0.75 |
| openang | 60 | -1.44 | 0.517 | 0.94 | 0.50 |

### filled, cells (4 seeds, 6.5 min per fit)

| parameter | truth | bias | scatter | pull rms | coverage 68% |
|---|---|---|---|---|---|
| vwind | 200 | -7.45 | 4.76 | 1.96 | 0.25 |
| vdisp | 20 | +3.1 | 1.52 | 1.43 | 0.50 |
| inc | 60 | -1.13 | 1.47 | 1.55 | 0.50 |
| pa | 30 | +0.132 | 0.98 | 1.21 | 0.50 |
| openang | 60 | -2.79 | 0.414 | 2.15 | 0.00 |

### filled, cells-df (4 seeds, 6.4 min per fit)

| parameter | truth | bias | scatter | pull rms | coverage 68% |
|---|---|---|---|---|---|
| vwind | 200 | -7.69 | 3.95 | 1.37 | 0.50 |
| vdisp | 20 | +4.02 | 2.39 | 1.27 | 0.50 |
| inc | 60 | -1.67 | 1.7 | 1.82 | 0.50 |
| pa | 30 | +0.297 | 0.92 | 1.00 | 0.75 |
| openang | 60 | -3.57 | 0.604 | 2.15 | 0.00 |

### filled, cells-profile (4 seeds, 2.7 min per fit)

| parameter | truth | bias | scatter | pull rms | coverage 68% |
|---|---|---|---|---|---|
| vwind | 200 | +22.8 | 1.58 | 2.65 | 0.00 |
| vdisp | 20 | -18 | 0.288 | 10.54 | 0.00 |
| inc | 60 | +3.5 | 4.62 | 2.90 | 0.00 |
| pa | 30 | +4.19 | 5.72 | 2.58 | 0.50 |
| openang | 60 | +19.9 | 1.54 | 5.99 | 0.00 |

### filled, cells22 (4 seeds, 10.3 min per fit)

| parameter | truth | bias | scatter | pull rms | coverage 68% |
|---|---|---|---|---|---|
| vwind | 200 | -7.18 | 4.58 | 1.94 | 0.25 |
| vdisp | 20 | +3.24 | 1.48 | 1.49 | 0.50 |
| inc | 60 | -1.18 | 1.42 | 1.60 | 0.50 |
| pa | 30 | +0.123 | 1.01 | 1.30 | 0.50 |
| openang | 60 | -2.87 | 0.603 | 2.20 | 0.00 |

### filled, shells (4 seeds, 1.1 min per fit)

| parameter | truth | bias | scatter | pull rms | coverage 68% |
|---|---|---|---|---|---|
| vwind | 200 | -7.67 | 6.66 | 2.31 | 0.50 |
| vdisp | 20 | +4.65 | 2.43 | 1.73 | 0.25 |
| inc | 60 | -1.66 | 1.83 | 2.00 | 0.25 |
| pa | 30 | +0.613 | 0.855 | 1.79 | 0.50 |
| openang | 60 | -3.46 | 1.16 | 3.26 | 0.00 |

### filled, spaxel (4 seeds, 0.8 min per fit)

| parameter | truth | bias | scatter | pull rms | coverage 68% |
|---|---|---|---|---|---|
| vwind | 200 | -5.89 | 4.9 | 2.27 | 0.25 |
| vdisp | 20 | +2.95 | 2.91 | 1.46 | 0.50 |
| inc | 60 | -1.22 | 1.91 | 2.81 | 0.25 |
| pa | 30 | +0.665 | 0.886 | 1.74 | 0.50 |
| openang | 60 | -3.38 | 0.694 | 3.25 | 0.00 |

### filled, voxel (4 seeds, 1.5 min per fit)

| parameter | truth | bias | scatter | pull rms | coverage 68% |
|---|---|---|---|---|---|
| vwind | 200 | +82.5 | 125 | 25.57 | 0.00 |
| vdisp | 20 | +54.8 | 0.0812 | 295.35 | 0.00 |
| inc | 60 | +2.6 | 12.4 | 34.61 | 0.00 |
| pa | 30 | +0.166 | 12.2 | 8.74 | 0.50 |
| openang | 60 | +24.9 | 0.00981 | 259.55 | 0.00 |

### hollow, cells (4 seeds, 8.2 min per fit)

| parameter | truth | bias | scatter | pull rms | coverage 68% |
|---|---|---|---|---|---|
| vwind | 200 | -4.71 | 4.44 | 1.99 | 0.25 |
| vdisp | 20 | +2.31 | 0.769 | 1.35 | 0.25 |
| inc | 60 | -0.441 | 0.691 | 1.01 | 0.75 |
| pa | 30 | -0.202 | 0.844 | 1.54 | 0.50 |
| openang | 70 | -2.86 | 0.296 | 2.26 | 0.00 |
| openin | 30 | +3.29 | 0.903 | 1.84 | 0.00 |

### hollow, cells-profile (4 seeds, 4.7 min per fit)

| parameter | truth | bias | scatter | pull rms | coverage 68% |
|---|---|---|---|---|---|
| vwind | 200 | +21.5 | 2.15 | 3.24 | 0.00 |
| vdisp | 20 | -18.1 | 0.218 | 11.11 | 0.00 |
| inc | 60 | +4.19 | 1.92 | 2.03 | 0.25 |
| pa | 30 | +3.08 | 5.58 | 2.87 | 0.75 |
| openang | 70 | +20.8 | 1.12 | 6.09 | 0.00 |
| openin | 30 | -20.5 | 1.23 | 3.47 | 0.00 |

### hollow, shells (4 seeds, 1.6 min per fit)

| parameter | truth | bias | scatter | pull rms | coverage 68% |
|---|---|---|---|---|---|
| vwind | 200 | -3.79 | 7.9 | 3.12 | 0.50 |
| vdisp | 20 | +2.1 | 2.85 | 1.95 | 0.50 |
| inc | 60 | -0.154 | 0.645 | 1.13 | 0.50 |
| pa | 30 | -0.103 | 0.805 | 1.92 | 0.25 |
| openang | 70 | -2.33 | 0.476 | 2.00 | 0.00 |
| openin | 30 | +1.96 | 2.14 | 2.08 | 0.50 |

### hollow, spaxel (4 seeds, 1.4 min per fit)

| parameter | truth | bias | scatter | pull rms | coverage 68% |
|---|---|---|---|---|---|
| vwind | 200 | -3.54 | 4.02 | 2.13 | 0.25 |
| vdisp | 20 | +1.62 | 1.87 | 1.37 | 0.50 |
| inc | 60 | -0.166 | 0.832 | 1.44 | 0.00 |
| pa | 30 | -0.0721 | 0.797 | 1.74 | 0.25 |
| openang | 70 | -2.62 | 0.867 | 2.30 | 0.25 |
| openin | 30 | +2.53 | 1.52 | 1.90 | 0.25 |

### hollow, voxel (4 seeds, 1.4 min per fit)

| parameter | truth | bias | scatter | pull rms | coverage 68% |
|---|---|---|---|---|---|
| vwind | 200 | +87.1 | 114 | 17.43 | 0.00 |
| vdisp | 20 | +54.8 | 0.0945 | 307.44 | 0.00 |
| inc | 60 | +3 | 11.3 | 17.87 | 0.00 |
| pa | 30 | -10.5 | 5.85 | 15.87 | 0.50 |
| openang | 70 | +24.9 | 0.00904 | 240.70 | 0.00 |
| openin | 30 | -8.93 | 4.74 | 1.02 | 0.50 |

### lognormal, cells (4 seeds, 6.3 min per fit)

| parameter | truth | bias | scatter | pull rms | coverage 68% |
|---|---|---|---|---|---|
| vwind | 200 | -0.616 | 7.45 | 1.44 | 0.50 |
| vdisp | 20 | +4.29 | 2.53 | 1.51 | 0.25 |
| inc | 60 | -0.137 | 2.18 | 1.58 | 0.50 |
| pa | 30 | -0.176 | 0.664 | 0.68 | 0.75 |
| openang | 60 | -0.765 | 1.22 | 0.53 | 1.00 |

### lognormal, cells22 (4 seeds, 9.6 min per fit)

| parameter | truth | bias | scatter | pull rms | coverage 68% |
|---|---|---|---|---|---|
| vwind | 200 | -0.676 | 7.25 | 1.45 | 0.50 |
| vdisp | 20 | +4.55 | 2.39 | 1.58 | 0.25 |
| inc | 60 | -0.205 | 2.13 | 1.58 | 0.50 |
| pa | 30 | -0.184 | 0.727 | 0.73 | 0.75 |
| openang | 60 | -0.847 | 1.54 | 0.71 | 0.75 |

### lowinc, cells (4 seeds, 4.5 min per fit)

| parameter | truth | bias | scatter | pull rms | coverage 68% |
|---|---|---|---|---|---|
| vwind | 200 | +0.0832 | 1.28 | 0.80 | 0.75 |
| vdisp | 20 | +0.0725 | 1.11 | 0.79 | 0.75 |
| inc | 30 | -0.333 | 1.28 | 1.26 | 0.50 |
| pa | 30 | -0.181 | 1.54 | 0.88 | 0.75 |
| openang | 60 | -4.07 | 1.16 | 2.54 | 0.00 |

### lowsnr, cells (4 seeds, 5.5 min per fit)

| parameter | truth | bias | scatter | pull rms | coverage 68% |
|---|---|---|---|---|---|
| vwind | 200 | -7.41 | 6.5 | 1.34 | 0.50 |
| vdisp | 20 | +4.24 | 3.06 | 1.02 | 0.50 |
| inc | 60 | -1.4 | 2.07 | 1.34 | 0.50 |
| pa | 30 | +0.572 | 0.886 | 0.84 | 0.75 |
| openang | 60 | -2.52 | 0.82 | 1.13 | 0.50 |

### rotation, cells (4 seeds, 7.0 min per fit)

| parameter | truth | bias | scatter | pull rms | coverage 68% |
|---|---|---|---|---|---|
| vwind | 200 | -9.1 | 4.14 | 2.12 | 0.25 |
| vdisp | 20 | +2.85 | 1.39 | 1.51 | 0.50 |
| inc | 60 | -1.48 | 1.02 | 1.71 | 0.50 |
| pa | 30 | +0.54 | 0.489 | 0.91 | 0.75 |
| openang | 60 | -3.47 | 0.394 | 2.51 | 0.00 |
| vrot | 50 | +1.71 | 7.14 | 1.95 | 0.75 |

### single, cells (4 seeds, 4.2 min per fit)

| parameter | truth | bias | scatter | pull rms | coverage 68% |
|---|---|---|---|---|---|
| vwind | 200 | -5.47 | 7.18 | 1.77 | 0.00 |
| vdisp | 20 | +3.48 | 1.2 | 1.22 | 0.25 |
| inc | 60 | -1.7 | 0.324 | 1.29 | 0.00 |
| pa | 30 | -0.659 | 1.16 | 1.16 | 0.50 |
| openang | 60 | -4.67 | 2.15 | 3.22 | 0.25 |

