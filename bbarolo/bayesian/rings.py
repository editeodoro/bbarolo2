"""The rings of BayesianBBarolo: a dictionary of arrays with the Rings object of the model engine."""

import numpy as np

from .common import isIterable

PARAMS = ('radii', 'xpos', 'ypos', 'vsys', 'vrot', 'vdisp', 'vrad', 'vvert', 'dvdz', 'zcyl', 'dens', 'z0', 'inc', 'phi')

RING_PARAMS = dict(radii='RADII', xpos='XPOS', ypos='YPOS', vsys='VSYS', vrot='VROT', vdisp='VDISP', vrad='VRAD',
                   vvert='VVERT', dvdz='DVDZ', zcyl='ZCYL', dens='DENS', z0='Z0', inc='INC', phi='PA')


class _RingDict(object):
    """A set of rings as a dictionary of arrays (r), with the Rings object of the model
    engine (object)."""

    def __init__(self, nrings):
        self.nr = int(nrings)
        self.r = {k: None for k in PARAMS}
        self.object = None

    @classmethod
    def from_rings(cls, rings):
        rd = cls(rings.nr)
        for k in PARAMS:
            rd.modify_parameter(k, getattr(rings, k))
        rd.make_object()
        return rd

    def copy(self):
        rd = _RingDict(self.nr)
        rd.r = {k: np.array(v) for k, v in self.r.items()}
        rd.make_object()
        return rd

    def set_rings_from_dict(self, rings):
        """ Define rings given an input dictionary of arrays (same keys as self.r) """
        for k in PARAMS:
            self.modify_parameter(k, rings[k])
        self.make_object()

    def modify_parameter(self, pname, pvalue, makeobj=False):
        """ Modifies one of the ring parameters (e.g., vrot, vdisp, etc.) """
        if pname not in self.r:
            raise ValueError("ERROR: Unknown ring parameter %s" % pname)
        self.r[pname] = np.array(pvalue, dtype=np.float32) if isIterable(pvalue) \
            else np.full(self.nr, pvalue, dtype=np.float32)
        if len(self.r[pname]) != self.nr:
            raise ValueError("All parameters must have size = %i" % self.nr)
        if makeobj:
            self.make_object()

    def make_object(self):
        """ Creates and stores the Rings object of the model engine """
        from ..rings import Rings
        r = Rings()
        r.setRings(**{k: np.asarray(v, dtype=np.float64) for k, v in self.r.items()})
        r.radsep = float(np.float32(self.r['radii'][-1] - self.r['radii'][-2])) if self.nr > 1 else 0.
        self.object = r
        return r
