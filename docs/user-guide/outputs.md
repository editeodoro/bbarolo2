# Outputs

Every task writes its results in the **output folder**. This page describes the folder, the kinds of files
and how to control them, with a table of the files of each task.

The files are written by the command line (`BBarolo -p`, `-f`, `-c`) and by `Result.write()` in Python: a
task run from Python computes in memory and writes nothing until `write()` is called, and then it writes
exactly the files of the command line (see {doc}`python`).

## The folder and the prefix

- **Folder.** {ref}`OUTFOLDER <par-all-outfolder>`; by default `output/<OBJECT>/` in the current folder,
  `<OBJECT>` being the OBJECT keyword of the FITS header. It is created if needed. If it already has
  files, those with the same names are overwritten, with a warning.
- **Prefix.** {ref}`OUTPREFIX <par-all-outprefix>`; by default the OBJECT of the header. Every file is
  `<prefix>_<what>`: `ngc2403_rings_final1.txt`. Spaces in the OBJECT become `_`, and slashes and
  parentheses `-` (`NGC 2403 (HI)` gives `NGC_2403_-HI-`).
- **No OBJECT.** If the header has no OBJECT, the folder and the prefix are `NONE` (`output/NONE/NONE_stats.txt`).
  Set OUTFOLDER and OUTPREFIX to choose other names.

Files in sub-folders: `maps/` (moment maps), `pvs/` (position–velocity slices) and `plotscripts/`.
The folder can hold `BBlog_0.txt`, the messages of a run with {ref}`LOGFILE <par-all-logfile>`.

## Kinds of files

### Ring tables (`.txt`)

The rings of 3DFIT, GALMOD and BAYESIAN3DFIT (`_rings_initial.txt`, `_rings_final1.txt`, `_rings_final2.txt`,
`_rings_model.txt`) are text files with a header line and one row per ring. The columns, in this order:

| Column | Content | Unit |
|---|---|---|
| `RAD` | radius of the ring | arcsec |
| `VROT`, `DISP` | rotation velocity, velocity dispersion | km/s |
| `INC`, `P.A.` | inclination, position angle | degrees |
| `Z0` | scale height | arcsec |
| `DENS` | gas density (`SIG(E20)` in the initial rings) | the units of the model, see {doc}`../tasks/galmod/index` |
| `XPOS`, `YPOS` | centre | pixels |
| `VSYS`, `VRAD` | systemic and radial velocity | km/s |
| `E_VROT1`, `E_VROT2`, `E_DISP1`, ... | with {ref}`FLAGERRORS <par-3dfit-flagerrors>` or BAYESIAN3DFIT: the lower (negative) and upper errors of each free parameter | as the parameter |

`_rings_initial.txt` and `_rings_model.txt` have no `VRAD` column. The values have three decimals. The rows of the rings
whose fit failed are left out (unless {ref}`BADOUT <par-3dfit-badout>`). The other text tables
(`_densprof.txt`, `_ellprof.txt`, `_detections.txt`, `_stats.txt`, `_spectrum.txt`) describe their columns in
their header, and on the page of their task. In Python, `bbarolo.tasks.read_table(file)` reads one of them
as an astropy Table, with lower case column names.

### FITS files

Cubes (`_mod_<norm>.fits`, `_mask.fits`, `_contsub.fits`, ...), maps and position–velocity slices are
FITS files with the header of the input cube: the WCS, the beam (BMAJ, BMIN, BPA), the units (BUNIT) and
the other keywords. What changes:

| File | Header |
|---|---|
| model cube `_mod_azim.fits`, `_mod_local.fits`, `_mod_none.fits` | the OBJECT is `<object>mod`; BUNIT as the data |
| mask `_mask.fits` | 16-bit integers, 1 in the mask and 0 outside; HISTORY cards with the type of mask and its parameters |
| `maps/<prefix>_0mom.fits`, `_1mom.fits`, `_2mom.fits` (and `_mod_..._<norm>.fits`) | two axes (RA, DEC); BUNIT `JY/BEAM * KM/S` (0), `KM/S` (1 and 2); BTYPE `intensity`, `velo-radio` or `dispersion` |
| `pvs/<prefix>_pv_a.fits` (major axis), `_pv_b.fits` (minor axis), `_mod_pv_a_<norm>.fits`, `_mask_pv_a.fits`, ... | first axis: offset in degrees (CTYPE1 `Offset`), second axis: velocity (m/s for NGC 2403); OBJECT `<object>_pv<PA>` with the angle of the slice |

The maps and the PV slices of 3DFIT and GALMOD are those of the data (in the mask) and of the model: compare
them to check a fit. The maps written by the map tasks (`_map_0th.fits`, ...) are described in
{doc}`../tasks/maps/index`.

### Plots and `plotscripts/`

With the default {ref}`PLOTS <par-all-plots>` `1`, 3DFIT writes the
plots as PDF files: `_parameters.pdf` (the rings against radius), `_maps_<norm>.pdf` (moment maps of
the data and of the model), `_pv_<norm>.pdf` and `_pvslices_<norm>.pdf` (PV slices) and
`_chanmaps_<norm>.pdf` (channel maps). Each plot is made by a Python script in `plotscripts/`
(`plot_parameters.py`, `plot_kinmaps.py`, `plot_pvs.py`, `plot_pvslices.py`, `plot_chanmaps.py`), and
`plotscripts/plot_all.py` runs them all. To change a plot, edit its script and run it, **from the folder
where BBarolo was run**: the scripts read the cube with the name given in FITSFILE (a relative path
too) and the other files from the output folder.

```bash
python3 output/ngc2403/plotscripts/plot_all.py
```

(`plot_pvs_old.py`, the previous style of the PV plot, writes the same file as `plot_pvs.py` and is not run by
`plot_all.py`.)

**What turns the plots off.**

| PLOTS | Effect |
|---|---|
| `1`, `true` (default) | the plots and their scripts |
| `0`, `false` | no plots and no scripts: the other files are written as usual |
| `-1` | no plots, and 3DFIT and BAYESIAN3DFIT write no model, maps, slices or mask: only the ring tables |
| `2`, `3` | also the plots of the initial estimates of the geometry, when BBarolo makes them (`_initialguesses.pdf`) |

In Python, `res.write(plots=False)` writes without plots. A text that is not a number or `true` (`PLOTS no`) is
read as 0.

### `_params.par`

Every run writes `<prefix>_params.par`: all the parameters of the run, the given ones and then the
defaults. It runs the same tasks again with `BBarolo -p`. See {doc}`parameter-files`.

## The files of each task

| Task | Files | Described in |
|---|---|---|
| 3DFIT | rings (initial, final), `_densprof.txt`, `_densprof_deconvolved.txt`, `_mod_<norm>.fits`, mask, `maps/`, `pvs/`, plots, `_asymdrift.txt` | {doc}`../tasks/3dfit/index` |
| BAYESIAN3DFIT | `_rings_final1.txt` with errors, best model, `maps/`, `pvs/`, samples (`_dynesty_results.npy`), `_corner.pdf`, `_profiles.pdf` | {doc}`../tasks/bayesian3dfit/index` |
| GALMOD | `_rings_model.txt`, `_mod_<norm>.fits`, mask, `_densprof.txt`, `maps/`, `pvs/` | {doc}`../tasks/galmod/index` |
| GalMod3D | no files: `model()` returns an HDU | {doc}`../tasks/galmod3d/index` |
| GALWIND | `_wind.fits`, `_wind_<n>mom.fits`, `_wind_pv.fits`, plots | {doc}`../tasks/galwind/index` |
| 2DFIT | `_2dtrm.txt`, `_2d_mod.fits`, `_map_1st.fits`, mask | {doc}`../tasks/2dfit/index` |
| SPACEPAR | `_spacepar.fits`, `_spacepar.pdf`, mask, rings | {doc}`../tasks/spacepar/index` |
| SEARCH | `_detections.txt`, `_detections.fits.gz`, `_detections_num.fits.gz`, `_detectmap.fits.gz`, `_mom<n>.fits`, `sources/` | {doc}`../tasks/search/index` |
| MAKEMASK | `_mask.fits` | {doc}`../tasks/makemask/index` |
| SMOOTH | `_s<M>.fits` | {doc}`../tasks/smooth/index` |
| SMOOTHSPEC | `_h<k>.fits` | {doc}`../tasks/smoothspec/index` |
| CONTSUB | `_contsub.fits` | {doc}`../tasks/contsub/index` |
| STATS | `_stats.txt` | {doc}`../tasks/stats/index` |
| Maps and profiles | `_map_0th.fits`, `_map_1st.fits`, `_map_2nd.fits`, `_map_massdens.fits`, `_map_RMS.fits`, `_spectrum.txt`, mask | {doc}`../tasks/maps/index` |
| PVSLICE | `_pv_<PA>.fits` | {doc}`../tasks/pvslice/index` |
| ELLPROF | `_ellprof.txt`, `_ellprof_deconvolved.txt`, `_ellprof_map.fits`, mask | {doc}`../tasks/ellprof/index` |
| REND3D | `_3D.fits`, mask | {doc}`../tasks/rend3d/index` |
| FITS utilities | the file named in the command (`--fitscopy`, `--fitsarith`, `--jybeam2k`, ...) | {doc}`../tasks/fitsutils/index` |

The section *Outputs* of each of these pages lists the files, the columns and the corresponding attributes
of the Python `Result`.
