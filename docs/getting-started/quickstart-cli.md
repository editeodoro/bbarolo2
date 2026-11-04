# First fit: the command line

BBarolo reads a **parameter file**: a text file with one parameter per line, `KEYWORD value`. Lines
that start with `#` or `//` are comments; keywords are case-insensitive. Save this file as
`ngc2403.par` next to `ngc2403.fits`:

```text
FITSFILE    ngc2403.fits
3DFIT       true
NRADII      20
RADSEP      30
XPOS        77
YPOS        77
VSYS        132.8
VROT        120
VDISP       8
INC         60
PA          123.7
Z0          10
FREE        VROT VDISP
MASK        SEARCH
```

and run it:

```bash
BBarolo -p ngc2403.par
```

BBarolo builds a mask of the emission (SEARCH), fits the rotation velocity and the dispersion in
20 rings of 30 arcsec, and writes the results in `output/ngc2403/` (the name is the OBJECT of the
FITS header): the fitted rings (`ngc2403_rings_final1.txt`), the model cube, moment maps,
position-velocity diagrams and plots. Every run also writes `ngc2403_params.par`, with all the
parameters that it used.

## Other ways to run

```bash
BBarolo -f ngc2403.fits                       # 3DFIT in automatic mode: everything estimated
BBarolo -p ngc2403.par THREADS=8 FREE="VROT VDISP INC PA"     # parameters that override the file
BBarolo -c fitsfile=ngc2403.fits 3dfit=true   # all the parameters on the command line
BBarolo -d 3DFIT                              # the parameters of a task and their defaults
BBarolo -t                                    # a template parameter file for 3DFIT
```

The parameters of each task are described in {doc}`../tasks/index`.
