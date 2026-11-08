# FITS utilities: how it works

BBarolo includes seven small tools for FITS files, run from the command line instead of a task:
listing and editing header keywords, copying a file or a section of it, arithmetic between images, and
the conversion of flux units between Jy/beam and K. They are ports of the CFITSIO utilities of BBarolo
1.8, built on astropy. Each is called as

    BBarolo --<utility> [arguments]

(or `python -m bbarolo --<utility> ...`). The utility takes all the arguments that follow it, and no task
runs. Called with the wrong number of arguments, a utility prints its usage and examples.
`BBarolo --fitsutils` (or `-F`) lists the utilities.

## File names

The input file names accept part of the CFITSIO extended syntax:

- **extension**: `file.fits[n]` (number, 0 is the primary array) or `file.fits[NAME]` (EXTNAME);
  without it the primary array is used (all the extensions for `--listhead`);
- **image section** (`--fitscopy` only): `file.fits[x1:x2,y1:y2,...]`, one range per FITS axis
  (axis 1 first), with pixels counted from 1 and both ends included. `*` (or an empty range) is the
  whole axis, a single number one pixel, a leading `-` reverses the axis (also `x2:x1` with
  $x_2 < x_1$). Axes not listed are taken whole. Steps (`x1:x2:s`) are not supported.

The two can be combined: `file.fits[1][11:50,21:60]`.

## Header keywords

**`--listhead file.fits[ext]`** prints the header of one extension, or of every extension in turn
(`Header listing for HDU #n`, counted from 1), card by card, ending with `END`.

**`--modhead file.fits[ext] keyword [value] ["comment"]`** prints the card of `keyword`; with a value
it sets the keyword (creating it if it does not exist) and prints the new card. The value is typed
from its text: `T` or `F` is a logical, then an integer, then a real number, otherwise a string (quotes
removed). The comment is the fourth argument, otherwise the existing comment is kept. For `HISTORY` and
`COMMENT` the value is added as a new card (and without a value all those cards are printed). The
structural keywords (`SIMPLE`, `BITPIX`, `NAXISn`, `EXTEND`, `XTENSION`, `PCOUNT`, `GCOUNT`, the table
keywords `TFIELDS`, `TFORMn`, ...) are protected and cannot be changed. The file is modified in place.

**`--remhead file.fits[ext] keyword`** deletes the keyword from the header, in place. Protected keywords
cannot be deleted.

## Copies

**`--fitscopy in.fits[ext][section] out.fits`** copies the input to `out.fits` (overwritten if it
exists). Without extension and section the whole file is copied, all its extensions included.
Otherwise one HDU is written, as the primary array of the output, cut to the section. For each axis $j$
of the section, with first pixel $x_1$ and $n$ pixels, the reference pixel is moved so that the world
coordinates do not change:

$$
\text{CRPIX}_j' = \text{CRPIX}_j - x_1 + 1 , \qquad
\text{CRPIX}_j' = x_1 + n - \text{CRPIX}_j ,\ \ \text{CDELT}_j' = -\text{CDELT}_j \ \ \text{(reversed axis)} .
$$

For a reversed axis $k$ the direction is flipped in the matrix that the header uses: column $k$ of a CD
matrix, or of a PC matrix that is not diagonal, otherwise CDELT$_k$ as above. The same is done for the
alternate world coordinates (CRPIX$_k$A ... CRPIX$_k$Z). The header of the HDU is copied, also from an
extension (without its XTENSION, PCOUNT and GCOUNT cards).

## Arithmetic

**`--fitsarith image1 {image2|value} {add|sub|mul|div} out.fits`** computes, voxel by voxel,

$$
O = A \circ B, \qquad \circ \in \{+, -, \times, /\},
$$

with $A$ the first image and $B$ the second image (same shape, at most 3 axes) or a number. The operator
is read from its first letter (`a`, `s`, `m`, `d`). In the division of two images, $O = 0$ where
$B = 0$. A number $B = 0$ (or text that is neither a file nor a number) is rejected. The operation is
done in double precision; the output has the header and the data type (BITPIX) of the first image, and
integer types are rounded to the nearest integer.

## Flux units

**`--jybeam2k file.fits [out.fits]`** and **`--k2jybeam file.fits [out.fits]`** convert a cube or a map
between Jy/beam and brightness temperature (K), with the Rayleigh-Jeans relation

$$
T_B = \frac{1.22222\times10^{6}}{\nu^2\, b_{\rm maj}\, b_{\rm min}}\; S ,
$$

with $S$ in Jy/beam, $T_B$ in K, $\nu$ the rest frequency in GHz (RESTFREQ or RESTFRQ of the header,
derived from the spectral axis, or the HI line, 1.420406 GHz, when missing) and $b_{\rm maj}$,
$b_{\rm min}$ the FWHMs of the beam (arcsec, from BMAJ, BMIN in degrees). `--jybeam2k` multiplies the
data by this factor (by $10^{-3}$ or $10^{-6}$ more for mJy/beam or µJy/beam) and sets BUNIT to `K`;
`--k2jybeam` divides by it and sets BUNIT to `JY/BEAM`. A unit of a map, `*KM/S` or `*M/S`, is kept.
The factor is printed and recorded in a HISTORY card; the data are written in single precision to
`out.fits`, or over the input file if no output is given.

The input unit is checked first: without BUNIT the unit is assumed to be the one to convert from; a
BUNIT that is neither Jy/beam nor K is an error; `--k2jybeam` on a cube in Jy/beam does nothing.
`--jybeam2k` on a cube that is not in Jy/beam (e.g. already in K) does nothing. An extension can be given
as `in.fits[ext]`; the output file is then required.

## Differences from BBarolo 1.8

- The utilities use astropy instead of CFITSIO: of the extended file-name syntax only extensions and
  image sections are supported (no filters, no steps in sections). With an extension or a section,
  `--fitscopy` writes only the selected HDU, as the primary array (1.8 copied all the HDUs, with the
  selected one cut and its header kept).
- `--jybeam2k` takes mJy/beam and µJy/beam into account (1.8 treated them as Jy/beam).
- `--jybeam2k` on a cube in K does nothing: 1.8 recognised K only when followed by a space in BUNIT, so it
  converted a cube with BUNIT = `K` again.

## In the code

- `bbarolo/fitsutils.py`: `listhead`, `modhead`, `remhead`, `fitscopy`, `fitsarith`, `ConvertFluxUnits`
  (`--jybeam2k`, `--k2jybeam`), and the parsers of the file names, `_parse_name`, `_section_slices`.
  Each function takes the list of arguments of the command line (e.g. `fitsutils.listhead(["a.fits"])`).
- `bbarolo/cli.py`: `getopts` dispatches the options `--<utility>`; `listFitsUtils` (`--fitsutils`).
- `bbarolo/params.py`: `FITSUTILS_LIST`, the descriptions of the utilities.
