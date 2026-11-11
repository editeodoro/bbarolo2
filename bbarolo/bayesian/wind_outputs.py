"""
The outputs of BayesianWind (BAYESIANWIND), written by BayesianWind.write in the output folder, with the prefix of
the cube (<prefix>):

  <prefix>_wind_params.txt       the free parameters: median and 1-sigma interval of the posterior (optimize: the best
                                 fit and the interval of the Laplace approximation), prior
  <prefix>_wind_shells.txt       the shells of the best fit: radius, VWIND, VDISP, VROT (with errors) and the flux of
                                 each cone in the radial bins of the emission (with errors)
  <prefix>_wind_cells.txt        the cells of the emission: cone, ranges of radius, polar angle and azimuth, clipped
                                 volume and emissivity at the best fit
  <prefix>_mod_wind.fits         the best model cube (as <prefix>_mod_<norm>.fits of 3DFIT), and <prefix>_mask.fits
  maps/<prefix>_{0,1,2}mom.fits, maps/<prefix>_mod_{0,1,2}mom_wind.fits   the total intensity, velocity and
                                 dispersion maps of the data (in the mask) and of the model, as 3DFIT
  pvs/<prefix>_pv_{a,b}.fits, pvs/<prefix>_mod_pv_{a,b}_wind.fits, pvs/<prefix>_mask_pv_{a,b}.fits   position-velocity
                                 slices of the data, the model and the mask through the apex, across the axis of the
                                 cones (a: along PA) and along it (b: PA + 90), as 3DFIT
  <prefix>_<sampler>_windresults.npy   the samples, their weights, the free parameters, the priors and the options
                                 (optimize: <prefix>_optimize_windresults.npy, also the best fit and the optimisation)
  plotscripts/                   the plotting scripts (PLOTS >= 0; see wind_plots.py), run with PLOTS > 0:
                                 <prefix>_chanmaps_wind.pdf, <prefix>_maps_wind.pdf, <prefix>_pv_wind.pdf, and
                                 <prefix>_wind_corner.pdf, <prefix>_wind_shells.pdf
"""

import os
import numpy as np

from .common import vprint


def _prefix(f, outfolder):
    p = f._cube_obj.pars()
    of = outfolder or p.outFolder or f._outfolder()
    of = of if of.endswith("/") else of + "/"
    os.makedirs(of, exist_ok=True)
    return of + p.outPrefix


def write_outputs(f, outfolder=None, plots=None, verbose=True):
    """Write the outputs of a BayesianWind f after run() (see the module docstring). Returns the files written."""
    if not f.modCalculated:
        raise RuntimeError("BayesianWind: nothing to write, run() first.")
    from ..cube import Cube, write_hdu
    from ..moments import MomentMap
    from ..pvslice import PositionVelocity
    pre = _prefix(f, outfolder)
    plots = f._cube_obj.pars().plots if plots is None else plots
    files = []
    c = f._cube_obj
    h = c.Head()
    # the parameters
    fn = pre + "_wind_params.txt"
    with open(fn, "w") as out:
        opt = f.optimum if f._method == "optimize" else None
        if opt is None:
            out.write(f"# BAYESIANWIND: emission {f.like_kwargs['emission']}, "
                      f"{f._likelihood_name()} likelihood, sampler "
                      f"{f._method}, effective sample size {f.ess:.0f}, log-likelihood at the medians "
                      f"{f.best_logl:.6g}\n")
        else:
            out.write(f"# BAYESIANWIND: emission {f.like_kwargs['emission']}, {f._likelihood_name()} likelihood, "
                      f"{'maximum a posteriori' if opt.posterior else 'maximum likelihood'} by {opt.method} "
                      f"(errors: {'Laplace approximation' if opt.cov is not None else 'not computed'}), "
                      f"log-likelihood at the best fit {f.best_logl:.6g}\n")
        col = "MEDIAN" if opt is None else "BEST"
        out.write(f"#{'NAME':>13s} {col:>14s} {'ERR_LO':>12s} {'ERR_HI':>12s}   PRIOR\n")
        for name, val, (lo, med, hi) in zip(f.freepar_names, f.params, f.quantiles):
            key = name if name in f.priors else name.rstrip("0123456789")
            val = med if opt is None else val
            out.write(f" {name:>13s} {val:14.6g} {med - lo:12.5g} {hi - med:12.5g}   {f.priors.get(key, '')}\n")
    files.append(fn)
    # the shells
    tab = f.shells_table()
    em = f.emission_errors()
    fn = pre + "_wind_shells.txt"
    with open(fn, "w") as out:
        out.write("# RADIUS(arcsec) VWIND(km/s) E_LO E_HI VDISP(km/s) E_LO E_HI VROT(km/s) E_LO E_HI\n")
        for i, r in enumerate(tab["radii"]):
            out.write(f" {r:12.4f}" + "".join(f" {tab[b][i]:10.3f} {tab[b + '_lo'][i]:8.3f} {tab[b + '_hi'][i]:8.3f}"
                                              for b in ("vwind", "vdisp", "vrot")) + "\n")
        if em is not None:
            out.write("#\n# The flux of the cones in the radial bins of the emission (emissivity x volume: the units "
                      "of the cube x pixels x channels), best fit and 1-sigma interval\n")
            out.write("# R0(arcsec) R1(arcsec) FLUX_REC LO HI FLUX_APP LO HI\n")
            e = em["edges"]
            for i in range(len(e) - 1):
                row = f"# {e[i]:10.3f} {e[i + 1]:10.3f}"
                for cone in (1, -1):
                    d = em.get(cone)
                    row += (f" {d['best'][i]:12.5g} {d['lo'][i]:12.5g} {d['hi'][i]:12.5g}" if d is not None
                            else f" {'-':>12s} {'-':>12s} {'-':>12s}")
                out.write(row + "\n")
    files.append(fn)
    # the cells
    if f.like_kwargs["emission"] not in ("spaxel", "voxel"):
        cells = f._wm.cells
        fn = pre + "_wind_cells.txt"
        vol = f.best_details["pts"]["vol"]
        x = f.emission[:cells.ncell]
        with open(fn, "w") as out:
            out.write("# CONE R0(arcsec) R1(arcsec) THETA0(deg) THETA1(deg) PHI0(deg) PHI1(deg) VOLUME(arcsec3) "
                      "EMISSIVITY\n")
            for k in range(cells.ncell):
                cc = cells.cells[k]
                out.write(f" {int(cc[0]):3d} {cc[1]:10.3f} {cc[2]:10.3f} {np.degrees(cc[3]):8.3f} "
                          f"{np.degrees(cc[4]):8.3f} {np.degrees(cc[5]):8.3f} {np.degrees(cc[6]):8.3f} "
                          f"{vol[k]:12.5g} {x[k]:12.5g}\n")
        files.append(fn)
    # the model, its moment maps and PV slices, in the layout of 3DFIT (maps/, pvs/), with the functions of 3DFIT
    from ..galfit_out import kinematic_maps, pv_slices
    from ..rings import Rings
    of, prefix = os.path.dirname(pre) + "/", os.path.basename(pre)
    mod = Cube.fromArray(np.asarray(f.model, np.float32), h.copy(), c.pars())
    mod.Head().setMinMax(0., 0.)
    mod.Head().setName(prefix + "mod")
    fn = pre + "_mod_wind.fits"
    write_hdu(mod.to_hdu(), fn)
    files.append(fn)
    mask = _output_mask(c)
    fn = pre + "_mask.fits"
    write_hdu(c.mask_hdu(), fn)
    files.append(fn)
    os.makedirs(of + "maps", exist_ok=True)
    os.makedirs(of + "pvs", exist_ok=True)
    for name, hdu in kinematic_maps(c, mod, mask, "_wind").items():
        fn = of + "maps/" + prefix + "_" + name + ".fits"
        write_hdu(hdu, fn)
        files.append(fn)
    b = f.best
    apex = Rings().setRings(radii=[b.conelen], xpos=b.xpos, ypos=b.ypos, vsys=b.vsys, phi=b.pa, inc=b.inc)
    for name, hdu in pv_slices(c, mod, apex, mask, "_wind").items():
        fn = of + "pvs/" + prefix + "_" + name + ".fits"
        write_hdu(hdu, fn)
        files.append(fn)
    # the samples
    fn = pre + f"_{f._method}_windresults.npy"
    res = dict(samples=f.samples, weights=f.weights, freepar_names=f.freepar_names, freepar_idx=f.freepar_idx,
               quantiles=f.quantiles, priors=f.priors, like_kwargs=f.like_kwargs, init=f._init)
    if f._method == "optimize" and f.optimum is not None:
        res.update(params=f.params, optimum={k: v for k, v in f.optimum.items() if k != "samples"})
    np.save(fn, res, allow_pickle=True)
    files.append(fn)
    if plots is not None and plots >= 0:
        files += _plots(f, of, prefix, run=plots > 0, verbose=verbose)
    vprint(verbose, f" BAYESIANWIND: written {len(files)} files with the prefix {pre}")
    return files


def _output_mask(c):
    """The mask of the maps of the data: that of the cube (MASK), built if needed (as 3DFIT)."""
    if not c.MaskAll():
        from ..galfit_out import _outputs_mode
        with _outputs_mode(c.pars()):
            c.BlankMask()
    return np.asarray(c.Mask(), bool)


def _cube_file(f, c, of):
    """The data cube that the plotting scripts read: the user's file (and its HDU), or a copy of an in-memory cube
    in the output folder (<OBJECT>_input.fits), as the plots of the other tasks."""
    from ..tasks import _input_copy
    from ..header import fits_image_hdu
    fn = _input_copy(f, c, of)
    try:
        return fits_image_hdu(fn)
    except Exception:
        return fn, 0


def _plots(f, of, prefix, run=True, verbose=True):
    """The plots: the plotting scripts of wind_plots.py (channel maps, moment maps, PV slices), run if run, and the
    corner plot and the shells (with matplotlib, if run). Returns the files written."""
    from .wind_plots import write_scripts, run as run_scripts
    pre = of + prefix
    cubefile, cubehdu = _cube_file(f, f._cube_obj, of)
    paths = write_scripts(f, of, prefix, cubefile, cubehdu)
    out = list(paths) + [of + "plotscripts/plot_utils.py"]
    if not run:
        return out
    ret = run_scripts(paths)
    if ret != 0:
        vprint(verbose, " BAYESIANWIND: something went wrong with the plots: check the scripts in " + of + "plotscripts/")
    out += [pre + s for s in ("_chanmaps_wind.pdf", "_maps_wind.pdf", "_pv_wind.pdf") if os.path.isfile(pre + s)]
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    try:
        if f._method == "optimize" and f.optimum.get("cov") is None:
            raise ValueError("an optimisation without errors")
        fig = f.corner_plot()
        fn = pre + "_wind_corner.pdf"
        fig.savefig(fn, bbox_inches="tight")
        plt.close(fig)
        out.append(fn)
    except ImportError:
        vprint(verbose, " BAYESIANWIND: no corner plot (it needs the package corner)")
    except ValueError as e:                   # (e.g. a parameter without dynamic range in the samples)
        vprint(verbose, f" BAYESIANWIND: no corner plot ({e})")
    # the shells
    tab = f.shells_table()
    em = f.emission_errors()
    fig, ax = plt.subplots(1, 4, figsize=(16, 3.6))
    for a, base in zip(ax, ("vwind", "vdisp", "vrot")):
        a.errorbar(tab["radii"], tab[base], yerr=[tab[base + "_lo"], tab[base + "_hi"]], fmt="o-", color="firebrick")
        a.set_xlabel("r (arcsec)")
        a.set_ylabel(f"{base.upper()} (km/s)")
    if em is not None:
        for cone, col, lab in ((1, "firebrick", "receding"), (-1, "steelblue", "approaching")):
            dd = em.get(cone)
            if dd is None:
                continue
            ax[3].fill_between(em["radii"], dd["lo"], dd["hi"], color=col, alpha=0.3)
            ax[3].plot(em["radii"], dd["best"], color=col, label=lab)
        ax[3].set_xlabel("r (arcsec)")
        ax[3].set_ylabel("flux per bin")
        ax[3].legend()
    fn = pre + "_wind_shells.pdf"
    fig.savefig(fn, bbox_inches="tight")
    plt.close(fig)
    out.append(fn)
    return out
