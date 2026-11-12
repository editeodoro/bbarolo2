"""
The plots of 3DFIT and BAYESIAN3DFIT from the outputs in memory: the inputs of the functions of fit3d.py taken from
a Result (from_result) or from a Bayesian3DFIT after its fit (from_bayes) instead of the files, with the settings of
the scripts (scripts.plot_settings). Used by Result.plot_parameters(), plot_chanmaps(), ... and by the same
methods of Bayesian3DFIT.
"""

import numpy as np

from . import fit3d
from .scripts import plot_settings

# The columns of a ring table (galfit.ring_table) for the keys of fit3d.read_rings
_TABLE_COLUMNS = dict(rad="radii", disp="vdisp")


def rings_dict(t):
    """The rings of a ring table (an astropy Table of a Result) as a dict like fit3d.read_rings: only the rings
    fitted (column fitok), with the errors e_<name>1, e_<name>2 as err_<key>."""
    if "fitok" in t.colnames:
        t = t[np.asarray(t["fitok"], bool)]
    out = {}
    for k in fit3d.RING_KEYS:
        col = _TABLE_COLUMNS.get(k, k)
        out[k] = np.asarray(t[col], float)
        if f"e_{col}1" in t.colnames:
            out["err_" + k] = np.array([np.asarray(t[f"e_{col}1"], float), np.asarray(t[f"e_{col}2"], float)])
    return out


def from_result(res):
    """The plots of the Result res of 3DFIT or BAYESIAN3DFIT."""
    if "galfit" not in res:
        raise ValueError(f"No plots of 3DFIT for a Result of {res.task}: they are for 3DFIT and BAYESIAN3DFIT.")

    def get(name):
        if name.startswith("_"):
            return res._output(name)
        return res[name] if name in res else None
    return Fit3DPlots(res.galfit, get, bayes=res.bayes if "bayes" in res else None, what=f"Result of {res.task}")


def from_bayes(f):
    """The plots of the Bayesian3DFIT f after its fit (run, optimize or load_results). Its best model and its
    outputs (Bayesian3DFIT.model_outputs) are kept for the next plots, until the best fit changes."""
    from ..galfit import ring_table
    if not f.modCalculated:
        raise ValueError("BAYESIAN3DFIT: no best fit to plot yet: run() (or optimize(), load_results()) first.")
    key = (np.asarray(f.params, float).tobytes(), id(f.outri))
    cache = getattr(f, "_plots_cache", None)
    if cache is None or cache[0] != key:
        cache = (key, {})
        f._plots_cache = cache
    store = cache[1]

    def mo():
        if "mo" not in store:
            store["mo"] = f.model_outputs()
        return store["mo"]

    def get(name):
        if name == "rings":
            mpar, errors = f._posterior_errors()
            return ring_table(f.outri.object, mpar, errors)
        if name == "_model_outputs":
            return mo()
        if name == "_model_norm":
            return list(mo().models)[0]
        if name in ("densprof", "maps"):
            return getattr(mo(), name)()
        return None
    return Fit3DPlots(f._galfit, get, bayes=f, what="Bayesian3DFIT")


class Fit3DPlots:
    """The inputs of the plots of 3DFIT or BAYESIAN3DFIT from the outputs in memory: gf, the Galfit of the fit;
    get(name), the outputs as in a Result (rings, rings_stage1, densprof, densprof_deconvolved, asymdrift, maps,
    _model_outputs, _model_norm; None if absent); bayes: the Bayesian3DFIT, or None for 3DFIT."""

    def __init__(self, gf, get, bayes=None, what=""):
        self.gf, self.get, self.bayes, self.what = gf, get, bayes, what
        self.settings = plot_settings(gf)

    # ---------------------------------------------------------------- inputs
    def mo(self):
        """The ModelOutputs of the final model (galfit_out)."""
        return self.get("_model_outputs")

    def final_rings(self):
        return rings_dict(self.get("rings"))

    def norm(self, model=None):
        """The name of a model: model ('azim', 'local', 'none' or 'noise', a key of Result.models), or that of
        the model of the Result if None."""
        mo = self.mo()
        names = list(mo.models) + (["noise"] if mo.noise is not None else [])
        name = self.get("_model_norm") if model is None else str(model).lower()
        if name not in names:
            raise ValueError(f"No model {model!r} in this {self.what}: the models are "
                             f"{', '.join(repr(n) for n in names)}.")
        return name

    def model(self, model=None):
        """The data of a model (see norm)."""
        mo = self.mo()
        name = self.norm(model)
        return np.asarray((mo.noise if name == "noise" else mo.models[name]).array)

    def cube(self):
        return np.asarray(self.mo().c.array)

    def mask(self):
        return np.asarray(self.mo().mask, dtype=float)

    def cube_kwargs(self):
        s = self.settings
        return dict(box=s["box"], cont=s["cont"], pixscale=s["pixscale"], gname=s["gname"], plotmask=s["plotmask"])

    def _maps(self, model=None, with_model=True):
        """The moment maps of the data, of the model (if with_model) and the units of the total intensity map."""
        maps = self.get("maps")
        data = [np.asarray(maps[f"{i}mom"].data) for i in range(3)]
        mods = None
        if with_model:
            norm = self.norm(model)
            if f"mod_0mom_{norm}" not in maps:
                raise ValueError(f"No maps of the model {norm!r} in this {self.what}.")
            mods = [np.asarray(maps[f"mod_{i}mom_{norm}"].data) for i in range(3)]
        return data, mods, maps["0mom"].header.get("BUNIT", "")

    # ---------------------------------------------------------------- plots
    def parameters(self):
        stage1 = self.get("rings_stage1")
        rings = rings_dict(stage1 if stage1 is not None else self.get("rings"))
        rings2 = rings_dict(self.get("rings")) if stage1 is not None else None
        sdens_obs, sdens_label, faceon, radmax = None, None, False, None
        deconv = self.get("densprof_deconvolved")
        if self.bayes is not None:
            f = self.bayes
            mp = f._model_dens_profile()
            dp = self.get("densprof")
            sdens = (mp["radii"], mp["sdens"], [mp["lo"], mp["hi"]])
            sdens_obs, sdens_label, faceon = (np.asarray(dp["radii"]), np.asarray(dp["sdens_fo"])), mp["label"], True
            radmax = None if f.disk_edge is None else float(f.disk_edge)
        elif deconv is not None:
            r = np.asarray(deconv["radii"])
            sdens = (r, np.asarray(deconv["sdens_fo"]),
                     [np.asarray(deconv["e_sdens_fo_lo"]), np.asarray(deconv["e_sdens_fo_hi"])])
            sdens_obs, sdens_label, faceon = (r, np.asarray(deconv["sdens_fo_obs"])), "deconvolved", True
        else:
            dp = self.get("densprof")
            sdens = (np.asarray(dp["radii"]), np.asarray(dp["median"]), np.asarray(dp["stddev"]))
        return fit3d.plot_parameters(rings, rings2, sdens=sdens, sdens_obs=sdens_obs, sdens_label=sdens_label,
                                     faceon=faceon, radmax=radmax, bunit=self.settings["bunit"])

    def asymdrift(self):
        t = self.get("asymdrift")
        if t is None:
            raise ValueError(f"No asymmetric drift correction in this {self.what} (ADRIFT).")
        rad, vcirc, va2, _, disp2r, fun, funr = (np.asarray(t[c], float) for c in t.colnames[:7])
        return fit3d.plot_asymdrift(self.final_rings(), rad, vcirc, va2, disp2r, fun, funr)

    def chanmaps(self, model=None):
        s = self.settings
        return fit3d.plot_chanmaps(self.cube(), self.model(model), self.final_rings(), self.mask(),
                                   cdelt3=s["cdelt3"], crval3=s["crval3"], beam=s["beam"], bunit=s["head_bunit"],
                                   **self.cube_kwargs())

    def pvs(self, model=None):
        data, _, _ = self._maps(with_model=False)
        return fit3d.plot_pvs(self.cube(), self.model(model), self.final_rings(), data[0], data[1], self.mask(),
                              vrange=self.settings["vrange"], **self.cube_kwargs())

    def pvslices(self, model=None):
        data, _, _ = self._maps(with_model=False)
        return fit3d.plot_pvslices(self.cube(), self.model(model), self.final_rings(), data[0], self.mask(),
                                   vrange=self.settings["vrange"], **self.cube_kwargs())

    def kinmaps(self, model=None):
        s = self.settings
        data, mods, bunit = self._maps(model)
        return fit3d.plot_kinmaps(data, mods, self.final_rings(), box=s["box"], pixscale=s["pixscale"], bunit=bunit,
                                  mapnames=s["mapnames"])
