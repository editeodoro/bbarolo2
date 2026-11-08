# Parameter files and the command line

{doc}`../getting-started/quickstart-cli` runs a first fit. This page describes the syntax of the parameter
file, how a file with several tasks runs, and the options of the `BBarolo` command. The parameters
themselves are described in {doc}`../tasks/general` (those of every task) and on the page of each task.

## The syntax

A parameter file is a text file with one parameter per line: the keyword, then the value, separated by
spaces or tabs.

```text
// Comments start with // or #, in the first column of the line
FITSFILE    ngc2403.fits
3DFIT       true
radsep      30
FREE        VROT VDISP
MASK        SEARCH
```

- **Comments and blank lines.** A line that starts with `#` or `/` is a comment, and blank lines are
  ignored. The comment must start in the first column: an indented `#` is read as a keyword, and BBarolo
  warns `unknown parameter '#'` (and goes on). A comment cannot follow a value on the same line: a text
  value takes the whole rest of the line, so `FITSFILE ngc2403.fits # data` looks for a file with that
  name, and `VROT 100 # note` is an error.
- **Keywords** are case-insensitive: `RADSEP`, `radsep` and `RadSep` are the same. Some have aliases (for
  example `GALFIT` for `3DFIT`): `bbarolo.help("3DFIT")` and the pages of the parameters give them. If a
  keyword is given twice, the last value is used.
- **Flags** (`true`/`false`): `true` or `1`, in any case, is true. Anything else is false, with no
  message: `PLOTS yes` switches the plots off.
- **Numbers**: BBarolo reads the first number of the value.
- **Text.** The value of the file names (`FITSFILE`, `OUTFOLDER`, `FITSLIST`, ...) is the rest of the line,
  so a name may contain spaces, without quotes (`OUTFOLDER  out multi/`). Names keep their case; the
  other text keywords (`MASK`, `GALMODTYPE`, ...) are not case-sensitive. A `~` at the start of a path is the home
  folder.
- **Lists of values**, such as `FREE`, `RADII` or `CONTCHANS`, are separated by spaces (`FREE VROT VDISP`).

### Ring parameters

The parameters of the rings (`RADII`, `VROT`, `VDISP`, `INC`, `PA`, `Z0`, `DENS`, `VRAD`, `VSYS`, `XPOS`,
`YPOS`, ...) accept one of these forms, in the same way in every task that uses them:

| Form | Example | Meaning |
|---|---|---|
| a number | `INC 60` | the same value for all the rings |
| a list | `VDISP 12 11 10 9 8` (commas are also accepted) | one value per ring |
| a column of a text file | `VROT file(curve.txt,2)` | column 2 of `curve.txt`, counted from 1 |
| rows of a column | `VROT file(curve.txt,2,3:5)` | rows 3 to 5 (`3` alone: from row 3 to the end) |
| a range of radii | `RADII 15~135:30` | `RADII` only: from 15 to 135 arcsec in steps of 30 |
| a position on the sky | `XPOS 07h36m51s`, `YPOS +65d35m20s` | `XPOS` and `YPOS` only: both are needed |

In `file(...)` the rows are the lines of the file, counted from 1, and the lines without a number in that
column (comments, the header) are skipped, so a ring file written by BBarolo can be read as it is. An
unreadable value is an error that names the parameter:

```text
INPUT error: VROT='abc' is not valid: give a number, one number per ring, file(name,column)
 BBAROLO ERROR: Unacceptable input parameters (see the messages above).
```

If the lists have different lengths, the shortest sets the number of rings. For the ring parameters that
are fitted or modelled see {doc}`../tasks/3dfit/index` and {doc}`../tasks/galmod/index`; the rules for each
parameter are in its description in the page of parameters of the task.

This file makes a model with five rings, with the rotation curve read from a file:

```text
FITSFILE  ngc2403.fits
GALMOD    true
RADII     15~135:30
XPOS      77
YPOS      77
VSYS      132.8
VROT      file(curve.txt,2)
VDISP     12 11 10 9 8
INC       60
PA        123.7
Z0        10
DENS      1
NORM      NONE
```

### Unknown and misspelt keywords

A keyword that BBarolo does not know is not used. If it is close to a known one, it is taken to be a
typo and the run stops before it starts, with the closest keywords:

```text
 BBAROLO ERROR: unknown parameter 'VDSIP' (did you mean 'VDISP' or 'MINVDISP' or 'MAXVDISP'?)
```

If it is not close to any keyword, BBarolo prints a warning, ignores the line and goes on. Check the
warnings at the top of a run.

## Several tasks in one file

A file can switch on any number of tasks, each with its flag (`3DFIT true`, `TOTALMAP true`, ...). They
all run on the same cube, one after the other, in a fixed order that does not depend on the order of the
lines of the file:

1. CONTSUB
2. the check of the cube (`CHECKCUBE`)
3. STATS
4. MAKEMASK
5. SMOOTH
6. SMOOTHSPEC
7. SEARCH
8. 3DFIT
9. BAYESIAN3DFIT
10. GALMOD
11. SPACEPAR
12. GALWIND
13. 2DFIT
14. the maps (TOTALMAP, VELOCITYMAP, DISPERSIONMAP, MASSDENSMAP, RMSMAP, GLOBALPROFILE)
15. PVSLICE
16. ELLPROF
17. REND3D

Some tasks change what the others see. CONTSUB replaces the cube in memory, so the tasks after it work
on the continuum-subtracted cube. SMOOTH and SMOOTHSPEC only write their cubes: the tasks that follow use
the original one. The tasks that need a mask (3DFIT, GALMOD, the maps, ...) build it as set by
{ref}`MASK <par-makemask-mask>`, with SEARCH or MAKEMASK if the mask needs them, and share it with the tasks
that follow. All the tasks write in the same output folder. For example, this file

```text
FITSFILE     ngc2403.fits
STATS        true
TOTALMAP     true
VELOCITYMAP  true
OUTFOLDER    multi
```

runs STATS, then builds the mask (the default is MASK SEARCH), then writes the total intensity map and the
velocity field. The folder `multi/` contains `ngc2403_stats.txt`, `ngc2403_map_0th.fits`,
`ngc2403_map_1st.fits`, `ngc2403_mask.fits` and `ngc2403_params.par`.

With {ref}`FITSLIST <par-all-fitslist>`, a text file with one FITS file per line, all the tasks run on each
file in turn (a file that cannot be read is skipped). Without OUTFOLDER, each cube has its own folder.

In Python, `bbarolo.run("file.par")` does the same in one process, and returns a dict `{task: Result}`
(see {doc}`python`).

## The command line

`BBarolo` is installed with the package. Without installing, `python3 bin/BBarolo` in the repository, or
`python3 -m bbarolo` (with the package in the path), take the same options. `BBarolo` with no argument
prints the help; `-h` too.

| Option | What it does |
|---|---|
| `-p file.par [KEY=value ...]` | runs the parameter file; the `KEY=value` that follow override the file |
| `-f cube.fits` | runs 3DFIT in automatic mode on the cube (BBarolo estimates the parameters); `KEY=value` can follow |
| `-c KEY=value ...` | all the parameters on the command line |
| `-d [TASK]` | prints the parameters and their defaults (of a task, if given) |
| `-t` | writes `param.par`, a template parameter file for 3DFIT, in the current folder |
| `-l` | lists the tasks |
| `-v` | prints the version of BBarolo and of numpy, scipy, astropy and numba |
| `-h` | prints the help |
| `--fitsutils` | lists the FITS utilities |
| `--listhead`, `--modhead`, `--remhead`, `--fitscopy`, `--fitsarith`, `--jybeam2k`, `--k2jybeam` | the FITS utilities, with their own arguments: see {doc}`../tasks/fitsutils/index` |

`KEY=value` takes a keyword, an `=` and a value, with no spaces around the `=`. Values that contain spaces
are quoted for the shell:

```bash
BBarolo -p ngc2403.par THREADS=8 FREE="VROT VDISP INC PA"
BBarolo -c fitsfile=ngc2403.fits stats=true outfolder=stats
BBarolo -f ngc2403.fits BAYESIAN3DFIT=true
```

An override replaces the value of the file. It can also switch a task on or off: `TOTALMAP=false` after
`-p` switches off a task of the file, `TOTALMAP=true` adds one. With `-c`, there is no file: a `KEY=value` that is not an option is a
parameter, and the keywords and their values are checked as in a file.

`BBarolo` exits with code 0 if all went well and 1 if it did not (a message `BBAROLO ERROR: ...` goes to
the standard error): an unknown option, a parameter file that cannot be opened, an unacceptable parameter
or a cube that cannot be read.

```text
 BBAROLO ERROR: Could not open the parameter file nofile.par.
```

### What every run writes

- **`<prefix>_params.par`**, in the output folder: a parameter file with all the parameters of the run (the
  ones given, then the defaults of the others, with the tasks that ran in the comments). The paths are
  absolute. It runs the same task again, `BBarolo -p output/ngc2403/ngc2403_params.par`, and it is the
  record of how a result was made.
- **The output folder**: {ref}`OUTFOLDER <par-all-outfolder>`, by default `output/<OBJECT>/` in the current
  folder. If it exists, its files with the same names are overwritten, with a warning. See {doc}`outputs`.
- **The messages.** {ref}`VERBOSE <par-all-verbose>` sets how many: `0` none (but the warnings and
  errors), `1` the default, with one line per ring for 3DFIT, `2` everything, including the parameters
  of the run. With {ref}`LOGFILE <par-all-logfile>` `true` the messages go to `BBlog_0.txt` in the output
  folder (`BBlog_1.txt` ... if it exists) and nothing is printed on the screen, except the errors of the
  command line. `SHOWBAR false` removes the progress bars, for example in a batch job.
- **The number of threads**, {ref}`THREADS <par-all-threads>` (default: the cores, at most 16): see
  {doc}`parallel`.

## Where to go next

- {doc}`outputs`: the files that the tasks write.
- {doc}`python`: the same parameters in Python.
- {doc}`../tasks/index`: the tasks, with their parameters.
