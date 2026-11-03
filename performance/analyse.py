#!/usr/bin/env python3
"""
Figures, tables and numbers of the 3DFIT performance report, from the results of
run_suite.py.

    python analyse.py --results results/results.jsonl --report report

writes report/figures/*.pdf, report/tables/*.tex and report/numbers.tex (LaTeX macros with
the numbers quoted in the text); then build the report with
    cd report && pdflatex report && pdflatex report
"""

import os
import re
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_suite as rs                                 # noqa: E402

# --------------------------------------------------------------------------
# Style (validated palette: categorical blue/orange/aqua; ordinal blue ramp)
# --------------------------------------------------------------------------
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
RAMP4 = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]          # ordinal: light -> dark
DIVERGING = LinearSegmentedColormap.from_list("bias", ["#104281", "#3987e5", "#9ec5f4", "#f0efec",
                                                       "#f2a49f", "#e34948", "#a32a2a"])
SEQUENTIAL = LinearSegmentedColormap.from_list("seq", ["#f0efec", "#cde2fb", "#86b6ef", "#3987e5",
                                                       "#1c5cab", "#0d366b"])
MODES = ("default", "wide", "truth")
MODECOL = {"default": BLUE, "wide": ORANGE, "truth": AQUA}
MODEMARK = {"default": "o", "wide": "s", "truth": "D"}
MODELAB = {"default": r"estimates, default limits ($\pm$5$^\circ$ INC, $\pm$15$^\circ$ PA)",
           "wide": r"estimates, wide limits ($\pm$15$^\circ$ INC, $\pm$30$^\circ$ PA)",
           "truth": r"true values, default limits"}
MODESHORT = {"default": "estimates, default limits", "wide": "estimates, wide limits",
             "truth": "true values, default limits"}
MASKVARS = ("search",) + tuple(rs.MASKVARIANTS)   # search: the reference fits (MASK = SEARCH, BWEIGHT = 1)
MASKLAB = {"search": "defaults", "smooth": "MASK=SMOOTH&SEARCH", "none": "MASK=NONE", "noblank": "BWEIGHT=0",
           "local": "NORM=LOCAL"}
MASKTAB = {"search": "defaults", "smooth": "MASK = SMOOTH\\&SEARCH", "none": "MASK = NONE",
           "noblank": "BWEIGHT = 0", "local": "NORM = LOCAL"}
MASKCOL = {"search": BLUE, "smooth": ORANGE, "none": AQUA, "noblank": YELLOW, "local": "#e87ba4"}
MASKMARK = {"search": "o", "smooth": "s", "none": "D", "noblank": "^", "local": "v"}

plt.rcParams.update({
    "font.size": 8.5, "axes.labelsize": 8.5, "axes.titlesize": 9, "legend.fontsize": 7.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.titlecolor": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.6,
    "lines.markersize": 4.5, "legend.frameon": False, "figure.dpi": 150, "savefig.bbox": "tight",
    "pdf.fonttype": 42, "font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans",
})
TEXTW = 6.6         # inches (text width of the report)


def fmt_pct(x, signed=True):
    return "--" if not np.isfinite(x) else (f"{100 * x:+.1f}" if signed else f"{100 * x:.1f}")


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def load(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def wrap180(a):
    return (np.asarray(a, float) + 180.) % 360. - 180.


def tables_of(recs):
    """(fits, rings): one row per fit and one row per fitted ring."""
    fits, rings = [], []
    for r in recs:
        g, cg = r["galaxy"], r["cube"]
        nrt = len(rs.fit_radii(cg["beam"]))
        row = dict(key=r["key"], experiment=r["experiment"], fitmode=r.get("fitmode", ""), variant=r.get("variant", ""),
                   gid=g["id"], kind=g["kind"],
                   spatial=r["spatial"], spectral=r["spectral"], snr=r["snr"], status=r["status"],
                   inc_true=g["inc"], pa_true=g["pa"], vflat=g["vflat"], sigma=g["sigma"], dinc=g["dinc"],
                   dpa=g["dpa"], npix=cg["npix"], nchan=cg["nchan"], nrings=nrt, nfit=0,
                   time_fit=r.get("time_fit", np.nan), time_sim=r.get("time_sim", np.nan),
                   error=r.get("error", ""))
        if r["status"] == "ok":
            t, f = r["true"], r["fit"]["rings"]
            Rt = np.array(t["radii"])
            ini = r["fit"].get("rings_initial", {})
            row["inc_init"] = ini.get("inc", [np.nan])[0] if ini else np.nan
            row["pa_init"] = ini.get("pa", [np.nan])[0] if ini else np.nan
            row["nfit"] = len(f["radii"])
            for j, R in enumerate(f["radii"]):
                i = int(np.argmin(np.abs(Rt - R)))
                if abs(Rt[i] - R) > 1e-3 * cg["beam"]:
                    continue
                rr = dict(key=r["key"], experiment=r["experiment"], fitmode=row["fitmode"], variant=row["variant"],
                          gid=g["id"], kind=g["kind"],
                          spatial=r["spatial"], spectral=r["spectral"], snr=r["snr"], R=R, Rb=R / cg["beam"],
                          Rn=R / rs.RMAX, iring=i, ring_snr=r["ring_snr"][i], inc_galaxy=g["inc"],
                          v_true=t["vrot"][i], v_fit=f["vrot"][j], s_true=t["vdisp"][i], s_fit=f["vdisp"][j])
                if "inc" in f:
                    rr.update(inc_true=t["inc"][i], inc_fit=f["inc"][j], pa_true=t["pa"][i], pa_fit=f["pa"][j])
                rings.append(rr)
        fits.append(row)
    fits, rings = pd.DataFrame(fits), pd.DataFrame(rings)
    fits["completeness"] = fits["nfit"] / fits["nrings"]
    rings["dV"] = rings["v_fit"] / rings["v_true"] - 1
    rings["dS"] = rings["s_fit"] / rings["s_true"] - 1
    rings["dV_kms"] = rings["v_fit"] - rings["v_true"]
    rings["dS_kms"] = rings["s_fit"] - rings["s_true"]
    if "inc_fit" in rings:
        rings["dInc"] = rings["inc_fit"] - rings["inc_true"]
        rings["dPA"] = wrap180(rings["pa_fit"] - rings["pa_true"])
    return fits, rings


def st(x):
    """median, 16th and 84th percentiles, half 68% width, number."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return dict(med=np.nan, p16=np.nan, p84=np.nan, sig=np.nan, n=0)
    p16, med, p84 = np.percentile(x, [16, 50, 84])
    return dict(med=med, p16=p16, p84=p84, sig=(p84 - p16) / 2, n=x.size)


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def savefig(fig, outdir, name):
    os.makedirs(outdir, exist_ok=True)
    fig.savefig(os.path.join(outdir, name + ".pdf"))
    fig.savefig(os.path.join(outdir, name + ".png"), dpi=130)
    plt.close(fig)


def fig_examples(recs, figdir, gid):
    """One galaxy at the four spatial resolutions (reference spectral resolution and S/N)."""
    sel = {r["spatial"]: r for r in recs if r["experiment"] == "resolution" and r["galaxy"]["id"] == gid
           and r["spectral"] == rs.REF["spectral"] and r["snr"] == rs.REF["snr"] and r["status"] == "ok"}
    if not sel:
        return
    fig, ax = plt.subplots(2, len(rs.SPATIAL), figsize=(TEXTW, 3.3), sharex=True, sharey="row")
    Rf = np.linspace(0.5, rs.RMAX, 300)
    for c, sp in enumerate(rs.SPATIAL):
        if sp not in sel:
            continue
        r = sel[sp]
        g, f = r["galaxy"], r["fit"]["rings"]
        tt = rs.profiles(g, Rf)
        beam = r["cube"]["beam"]
        for row, (k, lab) in enumerate((("vrot", r"$V_\mathrm{rot}$ (km/s)"), ("vdisp", r"$\sigma$ (km/s)"))):
            a = ax[row, c]
            a.plot(Rf, tt[k], color=INK, lw=1.2, label="true")
            a.plot(f["radii"], f[k], "o", color=BLUE, mec="white", mew=0.6, ms=4.5, label="3DFIT")
            a.axvspan(0, beam, color="#f0efec", lw=0, zorder=0)
            if c == 0:
                a.set_ylabel(lab)
            if row == 0:
                a.set_title(f"$R_\\mathrm{{max}}$ = {sp:g} beams")
            else:
                a.set_xlabel("radius (arcsec)")
                a.set_ylim(0, max(2.2 * g["sigma"], 1.2 * max(f["vdisp"])))
    ax[0, 0].legend(loc="lower right")
    ax[1, 0].text(0.04, 0.06, "grey: within 1 beam", transform=ax[1, 0].transAxes, color=INK2, fontsize=6.5)
    fig.tight_layout(w_pad=0.6, h_pad=0.4)
    savefig(fig, figdir, "fig_examples")


def cell_stats(rings, fits, minsnr=0.):
    """Statistics per (spatial, spectral, snr) cell of the resolution experiment."""
    rr = rings[(rings.experiment == "resolution") & (rings.ring_snr >= minsnr)]
    ff = fits[fits.experiment == "resolution"]
    out = []
    for (sp, sv, snr), grp in ff.groupby(["spatial", "spectral", "snr"]):
        sub = rr[(rr.spatial == sp) & (rr.spectral == sv) & (rr.snr == snr)]
        v, s = st(sub.dV), st(sub.dS)
        ok = grp[grp.status == "ok"]
        out.append(dict(spatial=sp, spectral=sv, snr=snr, nfits=len(grp), nfail=int((grp.status != "ok").sum()),
                        completeness=float(ok.completeness.mean()) if len(ok) else np.nan,
                        v_med=v["med"], v_sig=v["sig"], v_n=v["n"], s_med=s["med"], s_sig=s["sig"],
                        v_good=float(np.mean(np.abs(sub.dV) < 0.1)) if len(sub) else np.nan,
                        s_good=float(np.mean(np.abs(sub.dS) < 0.2)) if len(sub) else np.nan,
                        time=float(grp.time_fit.median())))
    return pd.DataFrame(out)


def fig_maps(cells, figdir, name, title_note):
    """Heat maps of bias and scatter over spatial x spectral resolution (reference S/N)."""
    c = cells[cells.snr == rs.REF["snr"]]
    X, Y = list(rs.SPATIAL), list(rs.SPECTRAL)
    fig, ax = plt.subplots(2, 2, figsize=(TEXTW, 4.6))
    panels = (("v_med", r"$V_\mathrm{rot}$: median error", True, 0.25),
              ("v_sig", r"$V_\mathrm{rot}$: scatter (68%)", False, 0.25),
              ("s_med", r"$\sigma$: median error", True, 0.8),
              ("s_sig", r"$\sigma$: scatter (68%)", False, 0.8))
    for a, (col, lab, signed, lim) in zip(ax.ravel(), panels):
        Z = np.full((len(Y), len(X)), np.nan)
        for _, row in c.iterrows():
            Z[Y.index(row.spectral), X.index(row.spatial)] = row[col]
        if signed:
            im = a.imshow(Z, origin="lower", cmap=DIVERGING, vmin=-lim, vmax=lim, aspect="auto")
        else:
            im = a.imshow(Z, origin="lower", cmap=SEQUENTIAL, vmin=0, vmax=lim, aspect="auto")
        for iy in range(len(Y)):
            for ix in range(len(X)):
                z = Z[iy, ix]
                if np.isfinite(z):
                    dark = abs(z) / lim > 0.55
                    a.text(ix, iy, f"{100 * z:+.0f}%" if signed else f"{100 * z:.0f}%", ha="center", va="center",
                           fontsize=7.5, color="white" if dark else INK)
        a.set_xticks(range(len(X)), [f"{x:g}" for x in X])
        a.set_yticks(range(len(Y)), [f"{y:g}" for y in Y])
        a.set_xlabel(r"spatial resolution: $R_\mathrm{max}$ / beam")
        a.set_ylabel(r"channel width / $\sigma$")
        a.set_title(lab)
        a.grid(False)
        cb = fig.colorbar(im, ax=a, fraction=0.05, pad=0.03)
        cb.ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{100 * v:.0f}%"))
        cb.outline.set_visible(False)
    fig.suptitle(f"Peak S/N = {rs.REF['snr']:g}; {title_note}", fontsize=8.5, color=INK2, y=1.0)
    fig.tight_layout()
    savefig(fig, figdir, name)


def _sweep(ax, cells, xkey, fixed, ykey, skey, xlab, logx):
    for i, snr in enumerate(rs.SNRS):
        c = cells[(cells.snr == snr) & (cells[fixed[0]] == fixed[1])].sort_values(xkey)
        if c.empty:
            continue
        x = np.array(c[xkey], float)
        dx = (1 + 0.07 * (i - 1.5)) if logx else 0.16 * (i - 1.5)
        xx = x * dx if logx else x + dx
        ax.errorbar(xx, 100 * c[ykey], yerr=100 * c[skey], color=RAMP4[i], marker="o", ms=4, lw=1.4,
                    capsize=1.5, elinewidth=0.9, label=f"S/N = {snr:g}")
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xlabel(xlab)
    if logx:
        ax.set_xscale("log")
        ax.set_xticks(rs.SPECTRAL, [f"{v:g}" for v in rs.SPECTRAL])
        ax.minorticks_off()
    else:
        ax.set_xticks(rs.SPATIAL, [f"{v:g}" for v in rs.SPATIAL])


def fig_sweeps(cells, figdir, name, note):
    fig, ax = plt.subplots(2, 3, figsize=(TEXTW, 4.4))
    for row, (xkey, fixed, xlab, logx) in enumerate(
            (("spatial", ("spectral", rs.REF["spectral"]), r"$R_\mathrm{max}$ / beam", False),
             ("spectral", ("spatial", rs.REF["spatial"]), r"channel width / $\sigma$", True))):
        _sweep(ax[row, 0], cells, xkey, fixed, "v_med", "v_sig", xlab, logx)
        _sweep(ax[row, 1], cells, xkey, fixed, "s_med", "s_sig", xlab, logx)
        a = ax[row, 2]
        for i, snr in enumerate(rs.SNRS):
            c = cells[(cells.snr == snr) & (cells[fixed[0]] == fixed[1])].sort_values(xkey)
            a.plot(c[xkey], 100 * c["completeness"] * (1 - c["nfail"] / c["nfits"]), "o-", color=RAMP4[i],
                   ms=4, lw=1.4, label=f"S/N = {snr:g}")
        a.set_xlabel(xlab)
        a.set_ylim(0, 105)
        if logx:
            a.set_xscale("log")
            a.set_xticks(rs.SPECTRAL, [f"{v:g}" for v in rs.SPECTRAL])
            a.minorticks_off()
        else:
            a.set_xticks(rs.SPATIAL, [f"{v:g}" for v in rs.SPATIAL])
        ax[row, 0].set_ylabel(r"$V_\mathrm{rot}$ error (%)")
        ax[row, 1].set_ylabel(r"$\sigma$ error (%)")
        a.set_ylabel("rings fitted (%)")
        fx = (r"channel width = $\sigma$" if xkey == "spatial" else
              f"$R_\\mathrm{{max}}$ = {rs.REF['spatial']:g} beams")
        ax[row, 0].set_title(fx, loc="left", fontsize=8, color=INK2)
    ax[0, 0].set_ylim(-30, 30)
    ax[1, 0].set_ylim(-30, 30)
    ax[0, 1].set_ylim(-90, 110)
    ax[1, 1].set_ylim(-90, 110)
    ax[0, 2].legend(loc="lower right")
    fig.suptitle(note, fontsize=8, color=INK2, y=1.0)
    fig.tight_layout(w_pad=0.8, h_pad=1.0)
    savefig(fig, figdir, name)


SNRBINS = [0, 1, 2, 3, 5, 8, 13, 20, 40, 1e9]


def binned(df, xcol, ycol, bins):
    out = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        s = df[(df[xcol] >= lo) & (df[xcol] < hi)]
        if len(s) >= 5:
            q = st(s[ycol])
            out.append(dict(x=np.median(s[xcol]), **q))
    return pd.DataFrame(out)


def fig_ringsnr(rings, figdir):
    """Errors of each ring against its own S/N (all configurations of the resolution experiment,
    rings beyond one beam from the centre, so that beam smearing does not mix with the noise)."""
    rr = rings[(rings.experiment == "resolution") & (rings.Rb >= 1)]
    fig, ax = plt.subplots(1, 2, figsize=(TEXTW, 2.6))
    for a, (col, lab, lim) in zip(ax, (("dV", r"$V_\mathrm{rot}$ error (%)", (-60, 40)),
                                       ("dS", r"$\sigma$ error (%)", (-100, 100)))):
        for i, sv in enumerate(rs.SPECTRAL):
            b = binned(rr[rr.spectral == sv], "ring_snr", col, SNRBINS)
            if b.empty:
                continue
            a.fill_between(b.x, 100 * b.p16, 100 * b.p84, color=RAMP4[i], alpha=0.12, lw=0)
            a.plot(b.x, 100 * b.med, "o-", color=RAMP4[i], ms=3.5, lw=1.4, label=f"channel = {sv:g} $\\sigma$")
        a.axhline(0, color=INK2, lw=0.8)
        a.set_xscale("log")
        a.set_xticks([1, 2, 3, 5, 10, 20, 40], ["1", "2", "3", "5", "10", "20", "40"])
        a.minorticks_off()
        a.set_xlabel("S/N of the ring")
        a.set_ylabel(lab)
        a.set_ylim(*lim)
    if ax[1].get_legend_handles_labels()[0]:
        ax[1].legend(loc="upper left")
    fig.tight_layout(w_pad=1.2)
    savefig(fig, figdir, "fig_ringsnr")


def fig_radius(rings, figdir):
    """Errors against radius in beams, for the four spatial resolutions (reference spectral
    resolution, S/N = 20 and 40, rings with S/N >= 3)."""
    rr = rings[(rings.experiment == "resolution") & (rings.spectral == rs.REF["spectral"])
               & (rings.snr >= 20) & (rings.ring_snr >= 3)]
    fig, ax = plt.subplots(1, 2, figsize=(TEXTW, 2.6))
    for a, (col, lab, lim) in zip(ax, (("dV", r"$V_\mathrm{rot}$ error (%)", (-30, 80)),
                                       ("dS", r"$\sigma$ error (%)", (-60, 120)))):
        for i, sp in enumerate(rs.SPATIAL):
            s = rr[rr.spatial == sp]
            q = s.groupby("Rb")[col].agg([lambda x: np.percentile(x, 16), "median",
                                          lambda x: np.percentile(x, 84)])
            q.columns = ["p16", "med", "p84"]
            a.fill_between(q.index, 100 * q.p16, 100 * q.p84, color=RAMP4[i], alpha=0.12, lw=0)
            a.plot(q.index, 100 * q.med, "o-", color=RAMP4[i], ms=3.5, lw=1.4,
                   label=f"$R_\\mathrm{{max}}$ = {sp:g} beams")
        a.axhline(0, color=INK2, lw=0.8)
        a.set_xlabel("radius (beams)")
        a.set_ylabel(lab)
        a.set_xlim(0, 8.2)
        a.set_ylim(*lim)
    ax[1].legend(loc="upper right")
    fig.tight_layout(w_pad=1.2)
    savefig(fig, figdir, "fig_radius")


def galaxy_table(rings, fits, kind, variant=None):
    """One row per fit of the geometry/warp experiments (or the flat disks of the mask
    experiment, for a variant): median ring errors."""
    rr = rings[rings.experiment == kind]
    if variant is not None:
        rr = rr[(rr.variant == variant) & (rr.kind == "geometry")]
    rows = []
    for key, s in rr.groupby("key"):
        f = fits[fits.key == key].iloc[0]
        rows.append(dict(key=key, fitmode=f.fitmode, gid=f.gid, snr=f.snr, inc=f.inc_true, pa=f.pa_true,
                         dinc=f.dinc, dpa=f.dpa, inc_init=f.inc_init, pa_init=f.pa_init,
                         dInc=np.median(s.dInc), dPA=np.median(s.dPA), absInc=np.median(np.abs(s.dInc)),
                         absPA=np.median(np.abs(s.dPA)), dV=np.median(s.dV), dS=np.median(s.dS),
                         dV_out=np.median(s.dV[s.Rn > 0.3]), completeness=f.completeness))
    g = pd.DataFrame(rows)
    if not g.empty:
        g["dInc_init"] = g.inc_init - g.inc
        g["dPA_init"] = wrap180(g.pa_init - g.pa)
    return g


def fig_geometry(gal, figdir, mode="default"):
    """Flat disks, one fit mode: errors (median over the rings) against the true inclination,
    coloured by the peak S/N."""
    fig, ax = plt.subplots(2, 2, figsize=(TEXTW, 4.6))
    s = gal[gal.fitmode == mode]
    for i, snr in enumerate(rs.SNRS):
        q = s[s.snr == snr]
        kw = dict(color=RAMP4[i], ms=4.5, mec="white", mew=0.5, ls="none", marker="o")
        ax[0, 0].plot(q.inc, q.dInc, label=f"{snr:g}", **kw)
        ax[0, 1].plot(q.inc, q.dPA, **kw)
        ax[1, 0].plot(q.inc, 100 * q.dV_out, **kw)
        ax[1, 1].plot(q.inc, 100 * q.dS, **kw)
    ax[0, 0].set_ylabel(r"INC error (deg)")
    ax[0, 0].set_ylim(-15, 15)
    ax[0, 1].set_ylabel(r"PA error (deg)")
    ax[0, 1].set_ylim(-15, 15)
    ax[1, 0].set_ylabel(r"$V_\mathrm{rot}$ error (%)")
    ax[1, 0].set_ylim(-50, 50)
    ax[1, 1].set_ylabel(r"$\sigma$ error (%)")
    ax[1, 1].set_ylim(-100, 100)
    for a in ax.ravel():
        a.set_xlabel("true inclination (deg)")
        a.set_xlim(10, 90)
        a.axhline(0, color=INK2, lw=0.8)
    fig.legend(*ax[0, 0].get_legend_handles_labels(), loc="upper center", ncol=4, fontsize=7,
               bbox_to_anchor=(0.5, 1.04), title="peak S/N of the cube", title_fontsize=7)
    fig.tight_layout(w_pad=1.2, h_pad=1.0)
    savefig(fig, figdir, "fig_geometry")


def fig_geometry_init(gal, figdir):
    fig, ax = plt.subplots(1, 2, figsize=(TEXTW, 2.8))
    for a, (col, ini, lab, lim, d) in zip(ax, (("dInc", "dInc_init", "INC", 20, 5), ("dPA", "dPA_init", "PA", 25, 15))):
        for mode in MODES:
            s = gal[gal.fitmode == mode]
            a.plot(s[ini], s[col], ls="none", marker=MODEMARK[mode], ms=3.8, color=MODECOL[mode], mec="white",
                   mew=0.5, label="initial INC, PA: " + MODELAB[mode], alpha=0.9)
        x = np.array([-lim, lim])
        a.plot(x, x, color=INK2, lw=0.8)
        a.fill_between(x, x - d, x + d, color="#f0efec", lw=0, zorder=0)
        a.set_xlim(-lim, lim)
        a.set_ylim(-lim, lim)
        a.set_xlabel(f"error of the initial {lab} estimate (deg)")
        a.set_ylabel(f"error of the fitted {lab} (deg)")
        a.axhline(0, color=INK2, lw=0.5, ls=":")
    fig.legend(*ax[0].get_legend_handles_labels(), loc="upper center", ncol=3, fontsize=6.5,
               bbox_to_anchor=(0.5, 1.07))
    ax[0].text(0.97, 0.04, "grey band: initial $\\pm$5$^\\circ$", transform=ax[0].transAxes, ha="right",
               color=INK2, fontsize=6.5)
    ax[1].text(0.97, 0.04, "grey band: initial $\\pm$15$^\\circ$", transform=ax[1].transAxes, ha="right",
               color=INK2, fontsize=6.5)
    fig.tight_layout(w_pad=1.5)
    savefig(fig, figdir, "fig_geometry_init")


def fig_warp_examples(recs, figdir, ids):
    sel = {}
    for r in recs:
        if r["experiment"] == "warp" and r["status"] == "ok" and r["galaxy"]["id"] in ids:
            sel[(r["galaxy"]["id"], r["fitmode"])] = r
    ids = [i for i in ids if (i, "default") in sel]
    if not ids:
        return
    fig, ax = plt.subplots(2, len(ids), figsize=(TEXTW, 3.4), sharex=True)
    ax = np.atleast_2d(ax)
    Rf = np.linspace(0.5, rs.RMAX, 300)
    for c, gid in enumerate(ids):
        g = sel[(gid, "default")]["galaxy"]
        tt = rs.profiles(g, Rf)
        for row, k in enumerate(("inc", "pa")):
            a = ax[row, c]
            ref = tt[k][0]
            a.plot(Rf, wrap180(tt[k] - ref) if k == "pa" else tt[k], color=INK, lw=1.2, label="true")
            for mode in MODES:
                if (gid, mode) not in sel:
                    continue
                f = sel[(gid, mode)]["fit"]["rings"]
                y = wrap180(np.array(f[k]) - ref) if k == "pa" else f[k]
                a.plot(f["radii"], y, marker=MODEMARK[mode], ms=3.2, lw=1.0, color=MODECOL[mode], mec="white",
                       mew=0.5, label=MODESHORT[mode])
            if row == 0:
                a.set_title(f"warp {gid}: S/N {g['snr']:g}", fontsize=8)
            else:
                a.set_xlabel("radius (arcsec)")
        ax[0, 0].set_ylabel("INC (deg)")
        ax[1, 0].set_ylabel(r"PA $-$ PA(0) (deg)")
    fig.legend(*ax[0, 0].get_legend_handles_labels(), loc="upper center", ncol=4, fontsize=6.5,
               bbox_to_anchor=(0.5, 1.05))
    fig.tight_layout(w_pad=0.5, h_pad=0.4)
    savefig(fig, figdir, "fig_warp_examples")


def fig_warp_cdf(rings, figdir):
    """Distribution of the ring errors in INC and PA: flat vs warped disks, two fit limits."""
    fig, ax = plt.subplots(1, 3, figsize=(TEXTW, 2.4))
    styles = {(kind, mode): (MODECOL[mode], "-" if kind == "geometry" else "--")
              for kind in ("geometry", "warp") for mode in MODES}
    for (kind, mode), (col, ls) in styles.items():
        s = rings[(rings.experiment == kind) & (rings.fitmode == mode)]
        if kind == "geometry":
            s = s[s.snr >= 20]           # the warps have S/N = 20
        for a, (q, lab, lim) in zip(ax, ((np.abs(s.dInc), "|INC error| (deg)", 20),
                                         (np.abs(s.dPA), "|PA error| (deg)", 20),
                                         (np.abs(100 * s.dV), r"|$V_\mathrm{rot}$ error| (%)", 30))):
            x = np.sort(np.asarray(q))
            if x.size == 0:
                continue
            a.plot(x, np.arange(1, x.size + 1) / x.size, color=col, ls=ls, lw=1.4,
                   label=f"{'flat' if kind == 'geometry' else 'warped'}: {MODESHORT[mode]}")
            a.set_xlim(0, lim)
            a.set_xlabel(lab)
    ax[0].set_ylabel("fraction of rings")
    fig.legend(*ax[0].get_legend_handles_labels(), loc="upper center", ncol=2, fontsize=6.5,
               bbox_to_anchor=(0.5, 1.16))
    fig.tight_layout(w_pad=1.0)
    savefig(fig, figdir, "fig_warp_cdf")


def warp_amplitudes(rings):
    """Per warp fit: true and fitted change of INC and PA between the inner flat disk (the rings
    inside the warp radius) and the two outermost rings."""
    rows = []
    for key, s in rings[rings.experiment == "warp"].groupby("key"):
        s = s.sort_values("R")
        inner = s.inc_true.to_numpy() == s.inc_true.iloc[0]
        inner &= np.abs(wrap180(s.pa_true - s.pa_true.iloc[0])) < 1e-6
        if inner.sum() < 2 or len(s) < inner.sum() + 2:
            continue
        pt, pf = np.unwrap(np.radians(s.pa_true)), np.unwrap(np.radians(s.pa_fit))
        rows.append(dict(key=key, fitmode=s.fitmode.iloc[0], gid=s.gid.iloc[0],
                         dinc_true=s.inc_true.iloc[-2:].mean() - np.median(s.inc_true[inner]),
                         dinc_fit=s.inc_fit.iloc[-2:].mean() - np.median(s.inc_fit[inner]),
                         dpa_true=np.degrees(pt[-2:].mean() - np.median(pt[inner])),
                         dpa_fit=np.degrees(pf[-2:].mean() - np.median(pf[inner]))))
    return pd.DataFrame(rows)


def fig_warp_amp(wa, figdir):
    fig, ax = plt.subplots(1, 2, figsize=(TEXTW, 2.8))
    for a, (t, f, lab, lim) in zip(ax, (("dinc_true", "dinc_fit", "INC", 22), ("dpa_true", "dpa_fit", "PA", 42))):
        for mode in MODES:
            q = wa[wa.fitmode == mode]
            k = np.polyfit(q[t], q[f], 1)[0] if len(q) > 2 else np.nan
            a.plot(q[t], q[f], ls="none", marker=MODEMARK[mode], ms=3.6, color=MODECOL[mode], mec="white", mew=0.4,
                   alpha=0.9, label=f"{MODESHORT[mode]} (slope {k:.2f})")
        a.plot([-lim, lim], [-lim, lim], color=INK2, lw=0.8)
        a.axhline(0, color=INK2, lw=0.5, ls=":")
        a.set_xlim(-lim, lim)
        a.set_ylim(-lim, lim)
        a.set_xlabel(f"true {lab} warp (deg)")
        a.set_ylabel(f"fitted {lab} warp (deg)")
        a.legend(loc="upper left", fontsize=6.3, handletextpad=0.2, frameon=True, framealpha=0.9, edgecolor="none")
    fig.tight_layout(w_pad=1.5)
    savefig(fig, figdir, "fig_warp_amp")


def cpp_pairs(rings, fits):
    """Rings fitted by Python and C++ on the same cubes: the reference resolution cubes, and the
    flat disks (true initial INC, PA)."""
    py = rings[(rings.experiment == "resolution") & (rings.spatial == rs.REF["spatial"])
               & (rings.spectral == rs.REF["spectral"])]
    cp = rings[(rings.experiment == "cpp") & (rings.kind == "resolution")]
    m = py.merge(cp, on=["gid", "snr", "R"], suffixes=("_py", "_cpp"))
    pyg = rings[(rings.experiment == "geometry") & (rings.fitmode == "truth")]
    cpg = rings[(rings.experiment == "cpp") & (rings.kind == "geometry")]
    mg = pyg.merge(cpg, on=["gid", "snr", "R"], suffixes=("_py", "_cpp"))
    return m, mg


def fig_cpp(m, mg, figdir):
    if m.empty:
        return
    nrow = 2 if not mg.empty else 1
    fig, ax = plt.subplots(nrow, 3, figsize=(TEXTW, 2.4 * nrow))
    ax = np.atleast_2d(ax)
    for i, snr in enumerate(rs.SNRS):
        s = m[m.snr == snr]
        if s.empty:
            continue
        ax[0, 0].plot(100 * s.dV_cpp, 100 * s.dV_py, "o", ms=3, color=RAMP4[i], mec="white", mew=0.4,
                      label=f"S/N = {snr:g}")
        ax[0, 1].plot(100 * s.dS_cpp, 100 * s.dS_py, "o", ms=3, color=RAMP4[i], mec="white", mew=0.4)
    for a, lim, lab in zip(ax[0, :2], (40, 100), (r"$V_\mathrm{rot}$", r"$\sigma$")):
        a.plot([-lim, lim], [-lim, lim], color=INK2, lw=0.8)
        a.set_xlim(-lim, lim)
        a.set_ylim(-lim, lim)
        a.set_xlabel(f"{lab} error, C++ (%)")
        a.set_ylabel(f"{lab} error, Python (%)")
    ax[0, 0].set_title("FREE = VROT VDISP", loc="left", fontsize=8, color=INK2)
    ax[0, 0].legend(loc="upper left", fontsize=6.5)
    d = 100 * (m.dV_py - m.dV_cpp)
    ax[0, 2].hist(np.clip(d, -20, 20), bins=np.linspace(-20, 20, 41), color=BLUE, lw=0)
    ax[0, 2].set_xlabel(r"$V_\mathrm{rot}$: Python $-$ C++ (% of the true value)")
    ax[0, 2].set_ylabel("rings")
    if nrow == 2:
        gp = mg.groupby("gid").agg(snr=("snr", "first"), dInc_py=("dInc_py", "median"),
                                   dInc_cpp=("dInc_cpp", "median"), dV_py=("dV_py", "median"),
                                   dV_cpp=("dV_cpp", "median"), dS_py=("dS_py", "median"),
                                   dS_cpp=("dS_cpp", "median"))
        for i, snr in enumerate(rs.SNRS):
            s = gp[gp.snr == snr]
            kw = dict(ls="none", marker="o", ms=3.5, color=RAMP4[i], mec="white", mew=0.4)
            ax[1, 0].plot(s.dInc_cpp, s.dInc_py, label=f"S/N = {snr:g}", **kw)
            ax[1, 1].plot(100 * s.dV_cpp, 100 * s.dV_py, **kw)
            ax[1, 2].plot(100 * s.dS_cpp, 100 * s.dS_py, **kw)
        for a, lim, lab in zip(ax[1], (8, 40, 100), ("INC error (deg)", r"$V_\mathrm{rot}$ error (%)",
                                                     r"$\sigma$ error (%)")):
            a.plot([-lim, lim], [-lim, lim], color=INK2, lw=0.8)
            a.set_xlim(-lim, lim)
            a.set_ylim(-lim, lim)
            a.set_xlabel(f"{lab}, C++")
            a.set_ylabel(f"{lab}, Python")
        ax[1, 0].set_title("FREE = VROT VDISP INC PA (median of the rings)", loc="left", fontsize=8, color=INK2)
        ax[1, 0].legend(loc="upper left", fontsize=6.5)
    fig.tight_layout(w_pad=1.0, h_pad=1.2)
    savefig(fig, figdir, "fig_cpp")


def tex_cpp_table(mg, path):
    gp = mg.groupby("gid").agg(snr=("snr", "first"), inc=("inc_galaxy_py", "first"),
                               dInc_py=("dInc_py", "median"), dInc_cpp=("dInc_cpp", "median"),
                               dV_py=("dV_py", "median"), dV_cpp=("dV_cpp", "median"),
                               dS_py=("dS_py", "median"), dS_cpp=("dS_cpp", "median"))
    gp = gp[(gp.inc >= 30) & (gp.inc < 75)]
    lines = [r"\begin{tabular}{r r rr rr rr}", r"\toprule",
             r" & & \multicolumn{2}{c}{INC error ($^\circ$)} & \multicolumn{2}{c}{$V_\mathrm{rot}$ error (\%)} & "
             r"\multicolumn{2}{c}{$\sigma$ error (\%)} \\",
             r"S/N & galaxies & Python & C++ & Python & C++ & Python & C++ \\", r"\midrule"]
    for snr in rs.SNRS:
        s = gp[gp.snr == snr]
        if s.empty:
            continue
        lines.append(f"{snr:g} & {len(s)} & {np.median(s.dInc_py):+.1f} & {np.median(s.dInc_cpp):+.1f} & "
                     f"{fmt_pct(np.median(s.dV_py))} & {fmt_pct(np.median(s.dV_cpp))} & "
                     f"{fmt_pct(np.median(s.dS_py))} & {fmt_pct(np.median(s.dS_cpp))} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write_table(path, lines)


# --------------------------------------------------------------------------
# Mask experiment
# --------------------------------------------------------------------------
def mask_tables(rings, fits):
    """(rings, galaxies, failures) of the mask diagnostic: the rings of the reference resolution
    configuration and the per-galaxy medians of the flat disks fitted from the true INC and PA,
    for the default mask (from the resolution/geometry experiments) and the other variants."""
    msk = rings[rings.experiment == "mask"]
    if msk.empty:
        return pd.DataFrame(), pd.DataFrame(), {}
    ref = rings[(rings.experiment == "resolution") & (rings.spatial == rs.REF["spatial"])
                & (rings.spectral == rs.REF["spectral"])].assign(variant="search")
    rr = pd.concat([ref, msk[msk.kind == "resolution"]], ignore_index=True)
    g0 = galaxy_table(rings, fits, "geometry")
    gg = [g0[g0.fitmode == "truth"].assign(variant="search")]
    gg += [galaxy_table(rings, fits, "mask", v).assign(variant=v) for v in rs.MASKVARIANTS]
    gal = pd.concat([g for g in gg if not g.empty], ignore_index=True)
    fail = {}
    for v in MASKVARS:
        for kind in ("resolution", "geometry"):
            if v == "search":
                f = fits[(fits.experiment == kind) & ((fits.fitmode == "truth") if kind == "geometry" else
                                                      ((fits.spatial == rs.REF["spatial"]) &
                                                       (fits.spectral == rs.REF["spectral"])))]
            else:
                f = fits[(fits.experiment == "mask") & (fits.variant == v) & (fits.kind == kind)]
            for snr in rs.SNRS:
                q = f[f.snr == snr]
                if kind == "geometry":
                    q = q[(q.inc_true >= 30) & (q.inc_true < 75)]
                fail[(v, kind, snr)] = (int((q.status != "ok").sum()), len(q))
    return rr, gal, fail


def mask_stats(rr, gal):
    rows = []
    for v in MASKVARS:
        for snr in rs.SNRS:
            s = rr[(rr.variant == v) & (rr.snr == snr) & (rr.ring_snr >= 3)]
            g = gal[(gal.variant == v) & (gal.snr == snr) & (gal.inc >= 30) & (gal.inc < 75)]
            a, b, c, d, e = st(s.dV), st(s.dS), st(g.dInc), st(g.dS), st(g.dV_out)
            rows.append(dict(variant=v, snr=snr, v_med=a["med"], v_sig=a["sig"], s_med=b["med"], s_sig=b["sig"],
                             i_med=c["med"], i_sig=c["sig"], gs_med=d["med"], gv_med=e["med"], ngal=len(g)))
    return pd.DataFrame(rows)


def fig_mask(ms, figdir):
    """Median errors (bars: 68% of the rings or galaxies) against the peak S/N for the masks."""
    fig, ax = plt.subplots(1, 3, figsize=(TEXTW, 2.6))
    panels = (("v_med", "v_sig", 100, r"$V_\mathrm{rot}$ error (%)", (-40, 40)),
              ("s_med", "s_sig", 100, r"$\sigma$ error (%)", (-100, 150)),
              ("i_med", "i_sig", 1, "INC error (deg)", (-8, 8)))
    x0 = np.array(rs.SNRS)
    for i, v in enumerate(MASKVARS):
        q = ms[ms.variant == v].sort_values("snr")
        if q.empty:
            continue
        xx = np.array(q.snr) * (1 + 0.06 * (i - 1.5))
        for a, (mc, sc, k, lab, lim) in zip(ax, panels):
            y, e = k * np.array(q[mc], float), k * np.array(q[sc], float)
            inside = y <= lim[1]
            a.errorbar(xx[inside], y[inside], yerr=e[inside], color=MASKCOL[v], marker=MASKMARK[v], ms=4, lw=1.3,
                       capsize=1.5, elinewidth=0.9, mec="white", mew=0.4, label=MASKLAB[v])
            for xo, yo in zip(xx[~inside], y[~inside]):      # off the scale: arrow at the top
                a.plot(xo, lim[1] - 0.04 * (lim[1] - lim[0]), marker="^", color=MASKCOL[v], ms=5, mec="white",
                       mew=0.4)
                a.text(xo * 1.08, lim[1] - 0.05 * (lim[1] - lim[0]), f"{yo:+.0f}", color=INK2, fontsize=6,
                       va="center")
    for a, (mc, sc, k, lab, lim) in zip(ax, panels):
        a.axhline(0, color=INK2, lw=0.8)
        a.set_xscale("log")
        a.set_xticks(x0, [f"{v:g}" for v in x0])
        a.minorticks_off()
        a.set_xlim(4, 50)
        a.set_ylim(*lim)
        a.set_xlabel("peak S/N of the cube")
        a.set_ylabel(lab)
    ax[0].set_title("fixed geometry", loc="left", fontsize=8, color=INK2)
    ax[1].set_title("fixed geometry", loc="left", fontsize=8, color=INK2)
    ax[2].set_title("flat disks, from the true INC, PA", loc="left", fontsize=8, color=INK2)
    fig.tight_layout(w_pad=1.0)
    fig.legend(*ax[0].get_legend_handles_labels(), loc="lower center", ncol=5, fontsize=6.8,
               bbox_to_anchor=(0.5, 0.99), handletextpad=0.3, columnspacing=1.0)
    savefig(fig, figdir, "fig_mask")


def tex_mask_table(ms, fail, path):
    lines = [r"\begin{tabular}{l r rr rr rr r}", r"\toprule",
             r" & & \multicolumn{2}{c}{$V_\mathrm{rot}$ error (\%)} & \multicolumn{2}{c}{$\sigma$ error (\%)} & "
             r"\multicolumn{2}{c}{INC error ($^\circ$)} & failed \\",
             r" & & \multicolumn{4}{c}{fixed geometry} & \multicolumn{2}{c}{flat disks} & fits \\",
             r"\cmidrule(lr){3-6}\cmidrule(lr){7-8}",
             r"3DFIT options & peak S/N & median & scatter & median & scatter & median & scatter & \\", r"\midrule"]
    for v in MASKVARS:
        q = ms[ms.variant == v]
        if q.empty:
            continue
        for _, r in q.iterrows():
            nf = sum(fail.get((v, k, r.snr), (0, 0))[0] for k in ("resolution", "geometry"))
            nt = sum(fail.get((v, k, r.snr), (0, 0))[1] for k in ("resolution", "geometry"))
            name = MASKTAB[v] if r.snr == rs.SNRS[0] else ""
            lines.append(f"{name} & {r.snr:g} & {fmt_pct(r.v_med)} & {fmt_pct(r.v_sig, False)} & "
                         f"{fmt_pct(r.s_med)} & {fmt_pct(r.s_sig, False)} & {r.i_med:+.1f} & {r.i_sig:.1f} & "
                         f"{nf}/{nt} \\\\")
        if v != MASKVARS[-1]:
            lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write_table(path, lines)


# --------------------------------------------------------------------------
# LaTeX tables and numbers
# --------------------------------------------------------------------------
def write_table(path, lines):
    """Write a LaTeX table, with proper minus signs for the negative numbers."""
    text = re.sub(r"(?<=[\s&\[(])-(?=\d)", "$-$", "\n".join(lines) + "\n")
    with open(path, "w") as f:
        f.write(text)


def tex_resolution_table(cells, path):
    c = cells[cells.snr == rs.REF["snr"]]
    lines = [r"\begin{tabular}{cc rr rr r r}", r"\toprule",
             r" & & \multicolumn{2}{c}{$V_\mathrm{rot}$ error (\%)} & \multicolumn{2}{c}{$\sigma$ error (\%)} "
             r"& rings & time \\",
             r"$R_\mathrm{max}$/beam & $\Delta v/\sigma$ & median & scatter & median & scatter & fitted (\%) & (s) \\",
             r"\midrule"]
    for sp in rs.SPATIAL:
        for sv in rs.SPECTRAL:
            r = c[(c.spatial == sp) & (c.spectral == sv)]
            if r.empty:
                continue
            r = r.iloc[0]
            ok = 100 * r.completeness * (1 - r.nfail / r.nfits)
            lines.append(f"{sp:g} & {sv:g} & {fmt_pct(r.v_med)} & {fmt_pct(r.v_sig, False)} & {fmt_pct(r.s_med)} & "
                         f"{fmt_pct(r.s_sig, False)} & {ok:.0f} & {r.time:.1f} \\\\")
        if sp != rs.SPATIAL[-1]:
            lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write_table(path, lines)


def tex_snr_table(cells, path):
    c = cells[(cells.spatial == rs.REF["spatial"]) & (cells.spectral == rs.REF["spectral"])].sort_values("snr")
    lines = [r"\begin{tabular}{r rr rr rr r}", r"\toprule",
             r" & \multicolumn{2}{c}{$V_\mathrm{rot}$ error (\%)} & \multicolumn{2}{c}{$\sigma$ error (\%)} & "
             r"\multicolumn{2}{c}{within 10\% / 20\%} & failed \\",
             r"peak S/N & median & scatter & median & scatter & $V_\mathrm{rot}$ & $\sigma$ & fits \\", r"\midrule"]
    for _, r in c.iterrows():
        lines.append(f"{r.snr:g} & {fmt_pct(r.v_med)} & {fmt_pct(r.v_sig, False)} & {fmt_pct(r.s_med)} & "
                     f"{fmt_pct(r.s_sig, False)} & {100 * r.v_good:.0f}\\% & {100 * r.s_good:.0f}\\% & "
                     f"{int(r.nfail)}/{int(r.nfits)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write_table(path, lines)


def tex_geometry_table(gal, fits, path, kind):
    bins = [(15, 30), (30, 45), (45, 60), (60, 75), (75, 86)] if kind == "geometry" else [(0, 90)]
    lines = [r"\begin{tabular}{l l r rr rr rr}", r"\toprule",
             r" & & & \multicolumn{2}{c}{INC error ($^\circ$)} & \multicolumn{2}{c}{PA error ($^\circ$)} & "
             r"$V_\mathrm{rot}$ & $\sigma$ \\",
             r"limits & INC range & fits & median & scatter & median & scatter & error (\%) & error (\%) \\",
             r"\midrule"]
    for mode in MODES:
        for lo, hi in bins:
            s = gal[(gal.fitmode == mode) & (gal.inc >= lo) & (gal.inc < hi)]
            nfail = int(((fits.experiment == kind) & (fits.fitmode == mode) & (fits.status != "ok")
                         & (fits.inc_true >= lo) & (fits.inc_true < hi)).sum())
            if s.empty and not nfail:
                continue
            i, p, v, d = st(s.dInc), st(s.dPA), st(s.dV_out), st(s.dS)
            rng = "all" if kind != "geometry" else f"{lo}--{hi}$^\\circ$"
            lines.append(f"{MODESHORT[mode]} & {rng} & {len(s)}{'+' + str(nfail) + 'F' if nfail else ''} & "
                         f"{i['med']:+.1f} & {i['sig']:.1f} & {p['med']:+.1f} & {p['sig']:.1f} & "
                         f"{fmt_pct(v['med'])} & {fmt_pct(d['med'])} \\\\")
        if mode != MODES[-1]:
            lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write_table(path, lines)


def tex_geometry_snr_table(gal, fits, path):
    """Flat disks with 30 <= INC < 75 deg, by peak S/N and fit mode."""
    lines = [r"\begin{tabular}{l r r rr rr rr}", r"\toprule",
             r" & & & \multicolumn{2}{c}{INC error ($^\circ$)} & \multicolumn{2}{c}{PA error ($^\circ$)} & "
             r"$V_\mathrm{rot}$ & $\sigma$ \\",
             r"initial INC, PA & S/N & fits & median & scatter & median & scatter & error (\%) & error (\%) \\",
             r"\midrule"]
    for mode in MODES:
        for snr in rs.SNRS:
            s = gal[(gal.fitmode == mode) & (gal.snr == snr) & (gal.inc >= 30) & (gal.inc < 75)]
            nfail = int(((fits.experiment == "geometry") & (fits.fitmode == mode) & (fits.status != "ok")
                         & (fits.snr == snr) & (fits.inc_true >= 30) & (fits.inc_true < 75)).sum())
            i, p, v, d = st(s.dInc), st(s.dPA), st(s.dV_out), st(s.dS)
            lines.append(f"{MODESHORT[mode]} & {snr:g} & {len(s)}{'+' + str(nfail) + 'F' if nfail else ''} & "
                         f"{i['med']:+.1f} & {i['sig']:.1f} & {p['med']:+.1f} & {p['sig']:.1f} & "
                         f"{fmt_pct(v['med'])} & {fmt_pct(d['med'])} \\\\")
        if mode != MODES[-1]:
            lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write_table(path, lines)


def tex_numbers(path, values):
    with open(path, "w") as f:
        f.write("% numbers quoted in the text (written by analyse.py)\n")
        for k, v in values.items():
            v = re.sub(r"^-", r"\\ensuremath{-}", str(v))      # a minus sign in text and in math
            f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")


def main():
    ap = argparse.ArgumentParser(description="Figures and tables of the 3DFIT performance report")
    ap.add_argument("--results", default=os.path.join(HERE, "results", "results.jsonl"))
    ap.add_argument("--report", default=os.path.join(HERE, "report"))
    args = ap.parse_args()
    recs = load(args.results)
    figdir, tabdir = os.path.join(args.report, "figures"), os.path.join(args.report, "tables")
    os.makedirs(tabdir, exist_ok=True)
    fits, rings = tables_of(recs)
    fits.to_csv(os.path.join(args.report, "fits.csv"), index=False)
    rings.to_csv(os.path.join(args.report, "rings.csv"), index=False)

    # Resolution experiment
    cells_all = cell_stats(rings, fits, 0.)
    cells = cell_stats(rings, fits, 3.)
    cells.to_csv(os.path.join(tabdir, "resolution_cells.csv"), index=False)
    gids = sorted({r["galaxy"]["id"] for r in recs if r["experiment"] == "resolution"})
    if gids:
        vf = {r["galaxy"]["id"]: r["galaxy"]["vflat"] for r in recs if r["experiment"] == "resolution"}
        gmid = sorted(gids, key=lambda i: vf[i])[len(gids) // 2]
        fig_examples(recs, figdir, gmid)
        fig_maps(cells, figdir, "fig_maps", "rings with S/N $\\geq$ 3")
        fig_maps(cells_all, figdir, "fig_maps_all", "all rings")
        fig_sweeps(cells, figdir, "fig_sweeps", "rings with S/N $\\geq$ 3; error bars: 68% of the rings")
        fig_ringsnr(rings, figdir)
        fig_radius(rings, figdir)
        tex_resolution_table(cells, os.path.join(tabdir, "resolution.tex"))
        tex_snr_table(cells, os.path.join(tabdir, "snr.tex"))
        tex_snr_table(cells_all, os.path.join(tabdir, "snr_all.tex"))

    # Geometry and warps
    num = {}
    for kind in ("geometry", "warp"):
        gal = galaxy_table(rings, fits, kind)
        if gal.empty:
            continue
        gal.to_csv(os.path.join(tabdir, f"{kind}_galaxies.csv"), index=False)
        tex_geometry_table(gal, fits, os.path.join(tabdir, f"{kind}.tex"), kind)
        if kind == "geometry":
            fig_geometry(gal, figdir)
            fig_geometry_init(gal, figdir)
            tex_geometry_snr_table(gal, fits, os.path.join(tabdir, "geometry_snr.tex"))
        for mode in MODES:
            s = gal[gal.fitmode == mode]
            if s.empty:
                continue
            tag = ("Geo" if kind == "geometry" else "Warp") + {"default": "Def", "wide": "Wide", "truth": "True"}[mode]
            mid = s[(s.inc >= 30) & (s.inc < 75)] if kind == "geometry" else s
            num[f"{tag}IncMed"] = f"{st(mid.dInc)['med']:+.1f}"
            num[f"{tag}IncSig"] = f"{st(mid.dInc)['sig']:.1f}"
            num[f"{tag}PASig"] = f"{st(mid.dPA)['sig']:.1f}"
            num[f"{tag}VSig"] = f"{100 * st(mid.dV_out)['sig']:.0f}"
            num[f"{tag}VMed"] = f"{100 * st(mid.dV_out)['med']:+.1f}"
            num[f"{tag}Within"] = f"{100 * np.mean(np.abs(mid.dInc) < 3):.0f}"
            num[f"{tag}Move"] = f"{np.median(np.abs(mid.dInc - mid.dInc_init)):.1f}"
            ok = np.isfinite(mid.dInc_init) & np.isfinite(mid.dInc)
            if ok.sum() > 2 and np.std(mid.dInc_init[ok]) > 0:
                num[f"{tag}Corr"] = f"{np.corrcoef(mid.dInc_init[ok], mid.dInc[ok])[0, 1]:.2f}"
            num[f"{tag}InitSig"] = f"{st(mid.dInc_init)['sig']:.1f}"
            num[f"{tag}InitMed"] = f"{st(mid.dInc_init)['med']:+.1f}"
    if any(r["experiment"] == "warp" for r in recs):
        wids = sorted({r["galaxy"]["id"] for r in recs if r["experiment"] == "warp"})
        amp = {r["galaxy"]["id"]: abs(r["galaxy"]["dinc"]) + abs(r["galaxy"]["dpa"]) / 2 for r in recs
               if r["experiment"] == "warp"}
        big = sorted(wids, key=lambda i: -amp[i])[:4]
        fig_warp_examples(recs, figdir, big)
        fig_warp_cdf(rings, figdir)
        wa = warp_amplitudes(rings)
        if not wa.empty:
            wa.to_csv(os.path.join(tabdir, "warp_amplitudes.csv"), index=False)
            fig_warp_amp(wa, figdir)
            for mode in MODES:
                q = wa[wa.fitmode == mode]
                tag = "Warp" + {"default": "Def", "wide": "Wide", "truth": "True"}[mode]
                if len(q) > 2:
                    num[tag + "IncSlope"] = f"{np.polyfit(q.dinc_true, q.dinc_fit, 1)[0]:.2f}"
                    num[tag + "PASlope"] = f"{np.polyfit(q.dpa_true, q.dpa_fit, 1)[0]:.2f}"
                    num[tag + "IncRes"] = f"{np.std(q.dinc_fit - q.dinc_true):.1f}"
                    num[tag + "PARes"] = f"{np.std(q.dpa_fit - q.dpa_true):.1f}"

    # C++ cross-check
    m, mg = cpp_pairs(rings, fits)
    if not mg.empty:
        tex_cpp_table(mg, os.path.join(tabdir, "cpp_geometry.tex"))
        gp = mg.groupby("gid").agg(dInc_py=("dInc_py", "median"), dInc_cpp=("dInc_cpp", "median"))
        num.update(CppGeoN=f"{len(gp)}", CppGeoDiff=f"{st(gp.dInc_py - gp.dInc_cpp)['sig']:.1f}",
                   CppGeoCorr=f"{np.corrcoef(gp.dInc_py, gp.dInc_cpp)[0, 1]:.2f}")
    if not m.empty:
        fig_cpp(m, mg, figdir)
        d = st(100 * (m.dV_py - m.dV_cpp))
        ds = st(100 * (m.dS_py - m.dS_cpp))
        num.update(CppN=f"{len(m)}", CppVSig=f"{d['sig']:.1f}", CppVMed=f"{d['med']:+.1f}",
                   CppSSig=f"{ds['sig']:.1f}", CppSMed=f"{ds['med']:+.1f}",
                   CppVpySig=f"{100 * st(m.dV_py)['sig']:.1f}", CppVcppSig=f"{100 * st(m.dV_cpp)['sig']:.1f}",
                   CppSpyMed=f"{100 * st(m.dS_py)['med']:+.0f}", CppScppMed=f"{100 * st(m.dS_cpp)['med']:+.0f}")

    # Mask diagnostic
    mr, mgal, mfail = mask_tables(rings, fits)
    if not mr.empty:
        ms = mask_stats(mr, mgal)
        ms.to_csv(os.path.join(tabdir, "mask.csv"), index=False)
        fig_mask(ms, figdir)
        tex_mask_table(ms, mfail, os.path.join(tabdir, "mask.tex"))
        tags = {"search": "Search", "smooth": "Smooth", "none": "None", "noblank": "Noblank", "local": "Local"}
        snrw = {5.: "Five", 10.: "Ten", 20.: "Twenty", 40.: "Forty"}
        for _, r in ms.iterrows():
            t = "Mask" + tags[r.variant] + snrw[r.snr]
            num[t + "S"] = fmt_pct(r.s_med) if abs(r.s_med) < 2 else f"{100 * r.s_med:+.0f}"
            num[t + "V"] = fmt_pct(r.v_med)
            num[t + "Inc"] = f"{r.i_med:+.1f}"
        num["NMask"] = f"{int((fits.experiment == 'mask').sum())}"

    # Numbers for the text
    py = fits[~fits.experiment.isin(["cpp", "mask"])]
    num["NCpp"] = f"{int((fits.experiment == 'cpp').sum())}"
    num.update(NFits=f"{len(py)}", NFailed=f"{int((py.status != 'ok').sum())}",
               NRings=f"{int((rings.experiment != 'cpp').sum())}",
               NGal=f"{len(gids)}", NCells=f"{len(cells)}",
               NGeo=f"{fits[(fits.experiment == 'geometry') & (fits.fitmode == 'default')].shape[0]}",
               NWarp=f"{fits[(fits.experiment == 'warp') & (fits.fitmode == 'default')].shape[0]}",
               TimeMed=f"{py.time_fit.median():.1f}", TimeMax=f"{py.time_fit.max():.0f}",
               TimeTot=f"{(py.time_fit.sum() + py.time_sim.sum()) / 60:.0f}")
    num.update(NAll=f"{len(fits)}", TimeAll=f"{(fits.time_fit.sum() + fits.time_sim.sum()) / 60:.0f}")
    # Python and C++ fit times on the same cubes
    c = fits[(fits.experiment == "cpp") & (fits.status == "ok")].assign(ck=lambda d: d.key.str.replace("cpp/", "", regex=False))
    pt = fits[fits.experiment.isin(["resolution", "geometry"]) & (fits.status == "ok")][["key", "time_fit"]]
    mt = c.merge(pt.rename(columns={"key": "ck", "time_fit": "t_py"}), on="ck")
    for kind, tag in (("resolution", "Res"), ("geometry", "Geo")):
        q = mt[mt.ck.str.startswith(kind)]
        if len(q):
            num[f"TimePy{tag}"] = f"{q.t_py.median():.1f}"
            num[f"TimeCpp{tag}"] = f"{q.time_fit.median():.1f}"
    # Diagnostic of the inclination bias (diag_incbias.py)
    dpath = os.path.join(os.path.dirname(args.results), "diag_incbias.json")
    if os.path.isfile(dpath):
        with open(dpath) as f:
            dg = json.load(f)
        tags = dict(default="Default", mc="MC", thin="Thin", analytic="Analytic", wfunc0="Wfunc", local="Local",
                    pix10="Pix", spatial12="Spatial", radsep1="Wide", reverse="Reverse")
        for t, tag in tags.items():
            v = [r["dinc"] for r in dg["fits"] if r["test"] == t]
            if v:
                num["Diag" + tag] = f"{np.median(v):+.1f}"
                num["Diag" + tag + "S"] = f"{100 * np.median([r['ds'] for r in dg['fits'] if r['test'] == t]):+.0f}"
        for t, tag in dict(w05="Half", w1="One", w2="Two", uniform="Uniform", rest="Rest").items():
            v = [x for r in dg["scans"] if r["test"] == t for x in r["dinc_rings"]]
            if v:
                num["Scan" + tag] = f"{np.median(v):+.1f}"
    ref = cells[(cells.spatial == rs.REF["spatial"]) & (cells.spectral == rs.REF["spectral"])
                & (cells.snr == rs.REF["snr"])]
    if not ref.empty:
        r = ref.iloc[0]
        num.update(RefVMed=fmt_pct(r.v_med), RefVSig=fmt_pct(r.v_sig, False), RefSMed=fmt_pct(r.s_med),
                   RefSSig=fmt_pct(r.s_sig, False))
    for k, v in list(num.items()):
        num[k] = v.replace("%", r"\%")
    tex_numbers(os.path.join(args.report, "numbers.tex"), num)
    print(f"{len(recs)} records: {len(fits)} fits, {len(rings)} rings -> {args.report}")
    print("failures:", fits[fits.status != "ok"].groupby(["experiment"]).size().to_dict())


if __name__ == "__main__":
    main()
