"""
The outputs of BayesianWind (BAYESIANWIND), written by BayesianWind.write in the output folder, with the prefix of
the cube (<prefix>):

  <prefix>_wind_params.txt       the free parameters: median and 1-sigma interval of the posterior, prior
  <prefix>_wind_shells.txt       the shells of the best fit: radius, VWIND, VDISP, VROT (with errors) and the flux of
                                 each cone in the radial bins of the emission (with errors)
  <prefix>_wind_cells.txt        the cells of the emission: cone, ranges of radius, polar angle and azimuth, clipped
                                 volume and emissivity at the best fit
  <prefix>_windmod.fits          the best model cube
  <prefix>_windmod_{0,1,2}mom.fits  its total intensity, velocity and dispersion maps
  <prefix>_wind_pv.fits, <prefix>_windmod_pv.fits   position-velocity slices of the data and of the model along
                                 the projected axis of the cones (PA - 90, through the apex)
  <prefix>_<sampler>_windresults.npy   the samples, their weights, the free parameters, the priors and the options
  plots (PLOTS): <prefix>_wind_corner.pdf, <prefix>_wind_maps.pdf (data and model maps and slices),
                 <prefix>_wind_shells.pdf
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
        out.write(f"# BAYESIANWIND: emission {f.like_kwargs['emission']}, "
                  f"{'marginal' if f.like_kwargs.get('marginal', True) else 'profile'} likelihood, sampler "
                  f"{f._method}, effective sample size {f.ess:.0f}, log-likelihood at the medians {f.best_logl:.6g}\n")
        out.write(f"#{'NAME':>13s} {'MEDIAN':>14s} {'ERR_LO':>12s} {'ERR_HI':>12s}   PRIOR\n")
        for name, (lo, med, hi) in zip(f.freepar_names, f.quantiles):
            key = name if name in f.priors else name.rstrip("0123456789")
            out.write(f" {name:>13s} {med:14.6g} {med - lo:12.5g} {hi - med:12.5g}   {f.priors.get(key, '')}\n")
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
                out.write(f" {int(cc[0]):3d} {cc[1]:10.3f} {cc[2]:10.3f} {np.degrees(cc[3]):8.3f} {np.degrees(cc[4]):8.3f}"
                          f" {np.degrees(cc[5]):8.3f} {np.degrees(cc[6]):8.3f} {vol[k]:12.5g} {x[k]:12.5g}\n")
        files.append(fn)
    # the model, its maps and the slices
    mod = Cube.fromArray(np.asarray(f.model, np.float32), h.copy(), c.pars())
    fn = pre + "_windmod.fits"
    write_hdu(mod.to_hdu(copy=True), fn)
    files.append(fn)
    m = MomentMap()
    m.input(mod)
    for i, fun in enumerate((m.ZeroMoment, m.FirstMoment, m.SecondMoment)):
        fun(False)
        fn = pre + f"_windmod_{i}mom.fits"
        write_hdu(m.to_hdu(), fn)
        files.append(fn)
    b = f.best
    ang = b.pa - 90.
    for cube, tag in ((c, "_wind_pv.fits"), (mod, "_windmod_pv.fits")):
        fn = pre + tag
        write_hdu(PositionVelocity(cube, b.xpos, b.ypos, ang).to_hdu(), fn)
        files.append(fn)
    # the samples
    fn = pre + f"_{f._method}_windresults.npy"
    np.save(fn, dict(samples=f.samples, weights=f.weights, freepar_names=f.freepar_names,
                     freepar_idx=f.freepar_idx, quantiles=f.quantiles, priors=f.priors, like_kwargs=f.like_kwargs,
                     init=f._init), allow_pickle=True)
    files.append(fn)
    if plots and plots > 0:
        files += _plots(f, pre, mod)
    vprint(verbose, f" BAYESIANWIND: written {len(files)} files with the prefix {pre}")
    return files


def _plots(f, pre, mod):
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    out = []
    try:
        fig = f.corner_plot()
        fn = pre + "_wind_corner.pdf"
        fig.savefig(fn, bbox_inches="tight")
        plt.close(fig)
        out.append(fn)
    except ImportError:
        pass
    # maps and slices
    d = np.where(f._ok, f._dsafe, np.nan)
    m = np.where(f._ok, f.model, np.nan)
    from ..pvslice import PositionVelocity
    b = f.best
    pvd = PositionVelocity(f._cube_obj, b.xpos, b.ypos, b.pa - 90.).to_hdu().data
    pvm = PositionVelocity(mod, b.xpos, b.ypos, b.pa - 90.).to_hdu().data
    fig, ax = plt.subplots(2, 3, figsize=(13, 8))
    vaxis = np.arange(d.shape[0])
    for row, (cube, pv, lab) in enumerate(((d, pvd, "data"), (m, pvm, "model"))):
        tot = np.nansum(cube, axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            v1 = np.nansum(cube * vaxis[:, None, None], axis=0) / tot
        v1[tot <= 3 * f.noiserms * np.sqrt(3.)] = np.nan
        ax[row, 0].imshow(tot, origin="lower", cmap="afmhot")
        ax[row, 0].set_title(f"{lab}: total intensity")
        ax[row, 1].imshow(v1, origin="lower", cmap="RdBu_r")
        ax[row, 1].set_title(f"{lab}: velocity (channel)")
        ax[row, 2].imshow(pv, origin="lower", aspect="auto", cmap="Greys")
        ax[row, 2].contour(pvm, levels=np.nanmax(pvm) * np.array([0.05, 0.2, 0.5]), colors="firebrick", linewidths=0.7)
        ax[row, 2].set_title(f"{lab}: PV along the axis")
    fn = pre + "_wind_maps.pdf"
    fig.savefig(fn, bbox_inches="tight")
    plt.close(fig)
    out.append(fn)
    # shells
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
