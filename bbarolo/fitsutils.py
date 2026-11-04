"""
Small utilities to handle FITS files (port of Utilities/fitsUtils.cpp), built on
astropy.io.fits instead of CFITSIO:

    BBarolo --listhead  file.fits[ext]
    BBarolo --modhead   file.fits[ext] keyword [newvalue] ["comment"]
    BBarolo --remhead   file.fits[ext] keyword
    BBarolo --fitscopy  in.fits[ext][x1:x2,y1:y2,...] out.fits
    BBarolo --fitsarith image1 {image2|value} {add|sub|mul|div} outimage
    BBarolo --jybeam2k  file.fits [out.fits]
    BBarolo --k2jybeam  file.fits [out.fits]

The CFITSIO extended file-name syntax is supported for extensions (``[n]`` or
``[NAME]``) and image sections (``[x1:x2,y1:y2]``, 1-based, inclusive, ``-``
to reverse an axis, ``*`` for the full axis).
"""

import os
import re
import sys
import numpy as np
from astropy.io import fits

from .bbutils import shortpath

STRUCT_KEYS = re.compile(r"^(SIMPLE|BITPIX|NAXIS\d*|EXTEND|XTENSION|PCOUNT|GCOUNT|TFIELDS|TFORM\d+|"
                         r"TBCOL\d+|THEAP|END|GROUPS|BLOCKED)$")


def _parse_name(fname):
    """Split 'file.fits[ext][section]' into (file, ext or None, section or None)."""
    m = re.match(r"^(.*?)((\[[^\]]*\])*)$", fname)
    base, brackets = m.group(1), re.findall(r"\[([^\]]*)\]", m.group(2) or "")
    ext, section = None, None
    for b in brackets:
        if any(ch in b for ch in ":,*"):
            section = b
        elif ext is None:
            ext = int(b) if b.strip().isdigit() else b.strip()
    return base, ext, section


def _section_slices(section, shape):
    """CFITSIO image section -> numpy slices (array axes are reversed w.r.t. FITS)."""
    parts = [s.strip() for s in section.split(",")]
    slices = []
    for i, s in enumerate(parts):
        n = shape[::-1][i]
        rev = s.startswith("-")
        s = s.lstrip("-")
        if s in ("*", ""):
            lo, hi = 1, n
        else:
            a = s.split(":")
            lo = int(a[0])
            hi = int(a[1]) if len(a) > 1 and a[1] else lo
        if lo > hi:
            lo, hi, rev = hi, lo, not rev
        sl = slice(lo - 1, hi)
        slices.append((sl, rev))
    slices += [(slice(None), False)] * (len(shape) - len(slices))
    return slices[::-1]


def listhead(argv):
    if len(argv) != 1:
        print("\n BBarolo's LISTHEAD FITS utility: \n\n"
              " List the FITS header keywords in a single extension, or, if [ext] is\n"
              " not given, list the keywords in all the extensions.\n\n"
              " Usage:\n   BBarolo --listhead filename[ext] \n\n"
              " Examples: \n"
              "   BBarolo --listhead file.fits      (list every header in the file) \n"
              "   BBarolo --listhead file.fits[0]   (list primary array header) \n"
              "   BBarolo --listhead file.fits[2]   (list header of 2nd extension) \n"
              "   BBarolo --listhead file.fits[GTI] (list header of GTI extension) \n")
        return 0
    fname, ext, _ = _parse_name(argv[0])
    with fits.open(fname) as hdul:
        hdus = [(hdul.index_of(ext), hdul[ext])] if ext is not None else list(enumerate(hdul))
        for i, hdu in hdus:
            print(f"Header listing for HDU #{i + 1}")
            for card in hdu.header.cards:
                print(card.image)
            print("END\n")
    return 0


def modhead(argv):
    if len(argv) not in (2, 3, 4):
        print("\n BBarolo's MODHEAD FITS utility: \n\n"
              " Write or modify the value of a header keyword.\n"
              " If a newvalue is not specified, just print the current value.\n\n"
              " Usage:\n   BBarolo --modhead filename[ext] keyword [newvalue] \"[comment]\"\n\n"
              " Examples: \n"
              "   BBarolo --modhead in.fits cunit1               (list the CUNIT1 keyword)\n"
              "   BBarolo --modhead in.fits cunit1 deg           (set CUNIT1 = 'deg')\n"
              "   BBarolo --modhead in.fits cunit1 deg \"RA unit\" (set CUNIT1 = 'deg' with comment)\n")
        return 0
    fname, ext, _ = _parse_name(argv[0])
    key = argv[1].upper()
    mode = "readonly" if len(argv) == 2 else "update"
    with fits.open(fname, mode=mode) as hdul:
        h = hdul[ext if ext is not None else 0].header
        isHistory = key in ("HISTORY", "COMMENT")
        exists = key in h
        if not exists:
            sys.stderr.write("Keyword does not exist\n")
        elif isHistory and len(argv) == 2:
            for card in h.cards:
                if card.keyword == key:
                    print(card.image)
        elif not isHistory:
            print(h.cards[key].image)
        if len(argv) >= 3:
            if isHistory:
                h.add_history(argv[2]) if key == "HISTORY" else h.add_comment(argv[2])
                print("Following entry has been added to the header:")
                print(f"{key} {argv[2]}")
            elif STRUCT_KEYS.match(key):
                sys.stderr.write("Protected keyword cannot be modified.\n")
            else:
                val = _autotype(argv[2])
                comment = argv[3] if len(argv) == 4 else (h.comments[key] if exists else None)
                h[key] = (val, comment) if comment else val
                print("Keyword has been changed to:")
                print(h.cards[key].image)
    return 0


def _autotype(s):
    t = s.strip()
    if t.upper() in ("T", "F"):
        return t.upper() == "T"
    for typ in (int, float):
        try:
            return typ(t)
        except ValueError:
            pass
    return t.strip("'")


def remhead(argv):
    if len(argv) != 2:
        print("\n BBarolo's REMHEAD FITS utility: \n\n"
              " Delete a header keyword from a FITS file.\n\n"
              " Usage:\n   BBarolo --remhead filename[ext] keyword \n\n"
              " Examples: \n"
              "   BBarolo --remhead in.fits object      (remove the OBJECT keyword)\n")
        return 0
    fname, ext, _ = _parse_name(argv[0])
    key = argv[1].upper()
    with fits.open(fname, mode="update") as hdul:
        h = hdul[ext if ext is not None else 0].header
        if key not in h:
            sys.stderr.write(f"Keyword {key} does not exist.\n")
        elif STRUCT_KEYS.match(key):
            sys.stderr.write(f"Protected keyword {key} cannot be deleted.\n")
        else:
            del h[key]
            print(f"Keyword {key} has been deleted.")
    return 0


def fitscopy(argv):
    if len(argv) != 2:
        print("\n BBarolo's FITSCOPY FITS utility: \n\n"
              " Copy an input file to an output file, optionally extracting an extension\n"
              " and/or an image section (1-based, inclusive; '-' reverses an axis).\n\n"
              " Usage:\n   BBarolo --fitscopy inputfile[ext][section] outputfile \n\n"
              " Examples: \n"
              "   BBarolo --fitscopy in.fits[11:50,21:60] out.fits      (copy a subimage)\n"
              "   BBarolo --fitscopy in.fits[-*,*] out.fits             (mirror reverse axis 0)\n"
              "   BBarolo --fitscopy in.fits[1] out.fits                (copy first extension)\n")
        return 0
    fname, ext, section = _parse_name(argv[0])
    out = argv[1]
    with fits.open(fname) as hdul:
        if ext is None and section is None:
            hdul.writeto(out, overwrite=True)
            return 0
        hdu = hdul[ext if ext is not None else 0]
        data, h = hdu.data, hdu.header.copy()
        if section is not None and data is not None:
            sls = _section_slices(section, data.shape)
            data = data[tuple(s for s, _ in sls)]
            for ax, (s, rev) in enumerate(sls):
                if rev:
                    data = np.flip(data, axis=ax)
            # update the reference pixel / increment of each FITS axis
            for i, (s, rev) in enumerate(sls[::-1]):
                k = i + 1
                if f"CRPIX{k}" in h and s.start is not None:
                    if rev:
                        n = data.shape[::-1][i]
                        h[f"CRPIX{k}"] = s.start + n - h[f"CRPIX{k}"] + 1
                        if f"CDELT{k}" in h:
                            h[f"CDELT{k}"] = -h[f"CDELT{k}"]
                    else:
                        h[f"CRPIX{k}"] = h[f"CRPIX{k}"] - s.start
        fits.PrimaryHDU(data, h if isinstance(hdu, fits.PrimaryHDU) else None).writeto(out, overwrite=True)
    return 0


def fitsarith(argv):
    if len(argv) != 4:
        print("\n BBarolo's FITSARITH FITS utility: \n\n"
              " Perform an operation between two images or between an image and a number. \n"
              " Supported operators are add, sub, mul, div (first character required). \n\n"
              " Usage:\n   BBarolo --fitsarith image1 { image2 | value } oper outimage \n\n"
              " Examples: \n"
              "   BBarolo --fitsarith in1.fits in2.fits add out.fits  (add the 2 files)\n"
              "   BBarolo --fitsarith in1.fits 1000.0 mul out.fits    (mult in1 by 1000)\n")
        return 0
    fa, ea, _ = _parse_name(argv[0])
    with fits.open(fa) as h1:
        hdu = h1[ea if ea is not None else 0]
        a = hdu.data.astype(np.float64)
        header = hdu.header.copy()
    fb, eb, _ = _parse_name(argv[1])
    if os.path.isfile(fb):
        with fits.open(fb) as h2:
            b = h2[eb if eb is not None else 0].data.astype(np.float64)
        if a.ndim > 3:
            sys.stderr.write("FITSARITH ERROR: images with > 3 dimensions are not supported.\n")
            return 1
        if a.shape != b.shape:
            sys.stderr.write("FITSARITH ERROR: input images don't have same size.\n")
            return 1
    else:
        try:
            b = float(argv[1])
        except ValueError:
            b = 0.
        if b == 0.:
            sys.stderr.write("FITSARITH ERROR: second argument is neither an image name nor a valid numerical value.\n")
            return 1
    op = argv[2][:1].lower()
    if op == "a":
        r = a + b
    elif op == "s":
        r = a - b
    elif op == "m":
        r = a * b
    elif op == "d":
        if np.isscalar(b):
            r = a / b
        else:
            with np.errstate(divide="ignore", invalid="ignore"):
                r = np.where(b != 0, a / np.where(b != 0, b, 1), 0.)
    else:
        sys.stderr.write(f"FITSARITH ERROR: unknown arithmetic operator {argv[2]}\n")
        return 1
    bitpix = header.get("BITPIX", -64)
    dtype = {8: np.uint8, 16: np.int16, 32: np.int32, 64: np.int64, -32: np.float32, -64: np.float64}[bitpix]
    if bitpix > 0:
        r = np.round(r)
    fits.PrimaryHDU(r.astype(dtype), header).writeto(argv[3], overwrite=True)
    return 0


def ConvertFluxUnits(argv, whichtype):
    from .cube import Cube
    if whichtype not in ("j2k", "k2j"):
        sys.stderr.write(f" ConvertFluxUnits function: '{whichtype}' is not an acceptable conversion type.\n")
        return 1
    taskname = "JYBEAM2K" if whichtype == "j2k" else "K2JYBEAM"
    if len(argv) not in (1, 2):
        print(f"\n BBarolo's {taskname} FITS utility: \n\n"
              " Convert flux units between Jy/Beam <-> Kelvin. \n"
              " NB: It requires the beam info (BMAJ, BMIN, BPA, in deg) \n"
              "     and RESTFREQ (in Hz) keywords to be present in the FITS header. \n\n"
              f" Usage:\n   BBarolo --{taskname.lower()} filename[ext] {{outputfile}}  \n\n"
              " Examples: \n"
              f"   BBarolo --{taskname.lower()} in.fits              \n"
              f"   BBarolo --{taskname.lower()} in.fits out.fits     \n")
        return 0
    bunit = fits.getheader(argv[0]).get("BUNIT")
    if bunit is None:
        bunit = "JY/BEAM" if whichtype == "j2k" else "K"
        sys.stderr.write(f"BUNIT Keyword does not exist. Assuming {bunit}\n")
    b = bunit.lower().replace(" ", "")
    if "jy/b" not in b and "j/b" not in b and "k" not in b:
        sys.stderr.write(f" {taskname} ERROR: unknown BUNIT={b}. Cannot make conversions!\n")
        return 0
    if whichtype == "k2j" and ("jy/b" in b or "j/b" in b):
        print(f" {taskname}: flux units are already in JY/BEAM. Doing nothing!!")
        return 0
    if whichtype == "j2k" and "k " in bunit.lower():
        print(f" {taskname}: flux units are already in KELVIN. Doing nothing!!")
        return 0
    c = Cube()
    if not c.readCube(argv[0]):
        sys.stderr.write(f"{argv[0]} is not a readable FITS file!\n")
        return 1
    h = c.Head()
    beam = h.Bmaj() * 3600. * h.Bmin() * 3600.
    freqGHz = h.Freq0() / 1E09
    factor = 1.22222E06 / (beam * freqGHz * freqGHz)
    if whichtype == "k2j":
        factor = 1 / factor
        endbunit = "JY/BEAM"
    else:
        from .bbutils import _jy_prefix
        factor *= _jy_prefix(b)             # mJy/beam, uJy/beam
        endbunit = "K"
    if "*km/s" in b:
        endbunit += " * KM/S"
    if "*m/s" in b and "*km/s" not in b:
        endbunit += " * M/S"
    print(f" Converting {h.Bunit()} --> {endbunit}:\n\n"
          f"  Rest frequency = {freqGHz:.6g} GHz\n"
          f"       Beam size = {h.Bmaj() * 3600:.6g}\" x {h.Bmin() * 3600:.6g}\" \n"
          f"    Scale factor = {factor:.6g}")
    c.array = (c.array * factor).astype(np.float32)
    h.setBunit(endbunit)
    h.keys.append(f"HISTORY BBAROLO {taskname}: Converted from {bunit} to {endbunit} with factor {factor:.4f}")
    outfile = argv[1] if len(argv) == 2 else argv[0]
    c.fitswrite_3d(outfile, True)
    print(f"\n New FITS file written to {shortpath(outfile)}")
    return 0
