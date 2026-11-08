# FITS utilities

BBarolo includes seven small tools for FITS files, run from the command line instead of a task: they
list and edit header keywords, copy a file or a section of it, do arithmetic between images, and convert
the flux unit between Jy/beam and K. They need no parameter file, and no Python class: in Python, the
functions are in `bbarolo.fitsutils` and take the list of the arguments of the command line.

## Example

Each tool is called as `BBarolo --<utility> [arguments]`. `BBarolo --fitsutils` lists them:

```bash
BBarolo --fitsutils
```

```text
   listhead  :  List header keywords in a FITS file.
   modhead   :  Modify/print a header keyword in a FITS file.
   remhead   :  Delete a header keyword in a FITS file.
   fitscopy  :  Copy a input file to a new FITS file with filtering.
   fitsarith :  Arithmetic operations with FITS files.
   jybeam2k  :  Convert flux units from Jy/beam to K.
   k2jybeam  :  Convert flux units from K to Jy/beam.
```

The examples below use a copy of `ngc2403.fits`, the cube of
{doc}`NGC 2403 <../../getting-started/example-data>`.

**Header keywords.** `--listhead` prints the header (here the first lines only), `--modhead` prints a
keyword, changes it or creates it, `--remhead` deletes it. `--modhead` and `--remhead` change the file in
place.

```bash
BBarolo --listhead ngc2403.fits
```

```text
Header listing for HDU #1
SIMPLE  =                    T / file does conform to FITS standard
BITPIX  =                  -32 / number of bits per data pixel
NAXIS   =                    3 / number of data axes
NAXIS1  =                  150 / length of data axis 1
NAXIS2  =                  150 / length of data axis 2
NAXIS3  =                   62 / length of data axis 3
...
```

```bash
BBarolo --modhead ngc2403.fits OBJECT
BBarolo --modhead ngc2403.fits TESTKEY 5 "a test"
BBarolo --remhead ngc2403.fits TESTKEY
```

```text
OBJECT  = 'ngc2403 '           /
Keyword does not exist
Keyword has been changed to:
TESTKEY =                    5 / a test
Keyword TESTKEY has been deleted.
```

**Copies.** `--fitscopy` copies a file, or a section of it. The section gives one range per axis
(axis 1 first, pixels counted from 1, both ends included, `*` for the whole axis). The reference pixel is
moved so that the world coordinates do not change. This cuts the central 70 x 70 pixels, all channels:

```bash
BBarolo --fitscopy "ngc2403.fits[41:110,41:110,*]" sub.fits
```

The output cube has 70 x 70 x 62 pixels and CRPIX1 and CRPIX2 are 36 (76 in the input). The tool
prints nothing when it succeeds.

**Arithmetic.** `--fitsarith` combines two images, or an image and a number, voxel by voxel, with `add`,
`sub`, `mul` or `div`. This multiplies the cube by 1000 and subtracts the cube from itself (the result is
0 everywhere):

```bash
BBarolo --fitsarith ngc2403.fits 1000 mul ngc2403_x1000.fits
BBarolo --fitsarith ngc2403.fits ngc2403.fits sub zero.fits
```

Both write the output file and print nothing. The output has the header of the first image, so the
BUNIT of the first command is still JY/BEAM: the factor of 1000 is not a unit conversion.

**Flux units.** `--jybeam2k` converts a cube from Jy/beam to K with the beam and the rest frequency of
the header, and `--k2jybeam` converts it back. Without an output file the input is overwritten.

```bash
BBarolo --jybeam2k ngc2403.fits ngc2403_K.fits
BBarolo --k2jybeam ngc2403_K.fits ngc2403_Jy.fits
```

```text
 Converting JY/BEAM --> K:

  Rest frequency = 1.42041 GHz
       Beam size = 59.3288" x 58.6054"
    Scale factor = 174.23

 New FITS file written to ngc2403_K.fits
```

```text
 Converting K --> JY/BEAM:

  Rest frequency = 1.42041 GHz
       Beam size = 59.3288" x 58.6054"
    Scale factor = 0.00573956

 New FITS file written to ngc2403_Jy.fits
```

The two commands also print the description of the file they open (size, spectral axis, units, beam),
left out here. A tool called with the wrong number of arguments prints its usage and examples.

## More

- {doc}`algorithm`: the file-name syntax (extensions and sections), the rules of each tool, the
  conversion formula and the differences from BBarolo 1.8.

```{toctree}
:hidden:
:maxdepth: 1

algorithm
```
