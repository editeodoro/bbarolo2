"""
Source finder (port of Tasks/search.cpp and the Duchamp-derived Map/ classes:
detection, object2D, object3D, objectgrower, scan, voxel).

The search proceeds exactly as in BBarolo/Duchamp:

1. threshold the cube and find 8-connected 2D objects in each channel
   (``spatial`` search) or 1D runs in each spectrum (``spectral`` search);
2. (two-stage merging) add each object to a running list, merging it into the
   *first* listed object it is close to;
3. optionally reject objects not meeting the size requirements;
4. merge all objects that are close (transitive closure);
5. optionally grow objects down to the secondary (growth) threshold, then merge
   again and reject objects not meeting the size requirements.

"Close" means: two voxels within ``threshVelocity`` channels and spatially
adjacent (or closer than ``threshSpatial`` if ``flagAdjacent=false``).
Connected-component labelling (Lutz algorithm in C++) is done with
``scipy.ndimage.label`` and merging/growth with vectorised numpy operations.
"""

import sys
import math
import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .stats import Stats
from .bbutils import AlltoVel, FluxtoJy, Pbcor, decToDMS, angularSeparation, arcsconv


# --------------------------------------------------------------------------
# Detection
# --------------------------------------------------------------------------
class Detection:
    """A detected object: a set of voxels (stored as sorted z, y, x index arrays)."""

    def __init__(self, z, y, x):
        z, y, x = np.asarray(z, np.int64), np.asarray(y, np.int64), np.asarray(x, np.int64)
        order = np.lexsort((x, y, z))
        self.z, self.y, self.x = z[order], y[order], x[order]
        self.haveParams = False
        self.flagWCS = False
        self.negSource = False
        self.totalFlux = self.peakFlux = self.intFlux = 0.
        self.xCentroid = self.yCentroid = self.zCentroid = 0.
        self.xpeak = self.ypeak = self.zpeak = 0
        self.ra = self.dec = self.vel = self.velMin = self.velMax = self.velWidth = 0.
        self.w20 = self.w50 = self.v20min = self.v20max = self.v50min = self.v50max = 0.
        self.raS = self.decS = ""
        self.raWidth = self.decWidth = self.majorAxis = self.minorAxis = self.posang = 0.

    # ---- geometry
    def getSize(self): return int(self.z.size)
    def getXmin(self): return int(self.x.min())
    def getXmax(self): return int(self.x.max())
    def getYmin(self): return int(self.y.min())
    def getYmax(self): return int(self.y.max())
    def getZmin(self): return int(self.z.min())
    def getZmax(self): return int(self.z.max())
    def getXaverage(self): return float(self.x.mean())
    def getYaverage(self): return float(self.y.mean())
    def getZaverage(self): return float(self.z.mean())

    def spatialPixels(self):
        return np.unique(self.y * (self.x.max() + 1) + self.x)

    def getSpatialSize(self):
        return int(self.spatialPixels().size)

    def getNumChannels(self):
        return int(np.unique(self.z).size)

    def getMaxAdjacentChannels(self):
        zs = np.unique(self.z)
        if zs.size == 0:
            return 0
        breaks = np.where(np.diff(zs) > 1)[0]
        runs = np.diff(np.concatenate(([-1], breaks, [zs.size - 1])))
        return int(runs.max())

    def hasEnoughChannels(self, minNumber):
        return self.getMaxAdjacentChannels() >= minNumber

    def isInObject(self, x, y, z):
        return bool(np.any((self.x == x) & (self.y == y) & (self.z == z)))

    def getXcentre(self, t="centroid"):
        return self.xpeak if t == "peak" else (self.getXaverage() if t == "average" else self.xCentroid)

    def getYcentre(self, t="centroid"):
        return self.ypeak if t == "peak" else (self.getYaverage() if t == "average" else self.yCentroid)

    def getZcentre(self, t="centroid"):
        return self.zpeak if t == "peak" else (self.getZaverage() if t == "average" else self.zCentroid)

    def getIntegFlux(self): return self.intFlux
    def getPeakFlux(self): return self.peakFlux
    def getTotalFlux(self): return self.totalFlux
    def getW20(self): return self.w20
    def getW50(self): return self.w50
    def getRA(self): return self.ra
    def getDec(self): return self.dec
    def getVel(self): return self.vel
    def isWCS(self): return self.flagWCS

    def getVsys(self):
        if self.w20 > 1.5 * self.w50:
            return (self.v20max + self.v20min) / 2.
        return ((self.v20max + self.v20min) / 2. + (self.v50max + self.v50min) / 2.) / 2.

    # ---- parameters
    def calcFluxes(self, array):
        f = array[self.z, self.y, self.x].astype(np.float64)
        self.totalFlux = float(f.sum())
        self.xCentroid = float(np.sum(self.x * f) / self.totalFlux)
        self.yCentroid = float(np.sum(self.y * f) / self.totalFlux)
        self.zCentroid = float(np.sum(self.z * f) / self.totalFlux)
        ip = int(np.argmin(f)) if self.negSource else int(np.argmax(f))
        self.peakFlux = float(f[ip])
        self.xpeak, self.ypeak, self.zpeak = int(self.x[ip]), int(self.y[ip]), int(self.z[ip])
        self.haveParams = True

    def _principal_axes(self):
        pix = self.spatialPixels()
        w = self.x.max() + 1
        xs, ys = (pix % w).astype(float), (pix // w).astype(float)
        n = xs.size
        mxx = np.sum(xs * xs) - xs.sum() ** 2 / n
        myy = np.sum(ys * ys) - ys.sum() ** 2 / n
        mxy = np.sum(xs * ys) - xs.sum() * ys.sum() / n
        if mxy == 0:
            theta = math.pi / 2. if mxx > myy else 0.
        else:
            theta = math.atan((mxx - myy + math.sqrt((mxx - myy) ** 2 + 4 * mxy * mxy)) / (2 * mxy))
        x0, y0 = xs.mean(), ys.mean()
        maj = (xs - x0 + 0.5) * math.cos(theta) + (ys - y0 + 0.5) * math.sin(theta)
        mnr = (xs - x0 + 0.5) * math.sin(theta) + (ys - y0 + 0.5) * math.cos(theta)
        axes = [max(abs(maj.max() - maj.min()), 0.5), max(abs(mnr.max() - mnr.min()), 0.5)]
        return axes, theta

    def calcWCSparams(self, head):
        if head.isWCS():
            pix = np.array([[self.getXcentre(), self.getYcentre(), self.getZcentre()],
                            [self.getXcentre(), self.getYcentre(), self.getZmin()],
                            [self.getXcentre(), self.getYcentre(), self.getZmax()],
                            [self.getXmin() - 0.5, self.getYmin() - 0.5, self.getZcentre()],
                            [self.getXmax() + 0.5, self.getYmax() + 0.5, self.getZcentre()]])
            try:
                world = head.pixToWCS(pix)
                self.ra, self.dec = float(world[0, 0]), float(world[0, 1])
                self.raS = decToDMS(self.ra, head.lngtype())
                self.decS = decToDMS(self.dec, head.lattype())
                self.raWidth = angularSeparation(world[3, 0], world[0, 1], world[4, 0], world[0, 1]) * 3600.
                self.decWidth = angularSeparation(world[0, 0], world[3, 1], world[0, 0], world[4, 1]) * 3600.
                axes, theta = self._principal_axes()
                ps = head.PixScale() * arcsconv(head.Cunit(1))
                self.majorAxis, self.minorAxis = max(axes) * ps, min(axes) * ps
                self.posang = theta * 180. / math.pi
                self.flagWCS = True
                self.haveParams = True
            except Exception:
                sys.stderr.write("BB calcWCSparams: Error in calculating the WCS for this object.")
        self.vel = float(AlltoVel(head.getZphys(self.getZcentre()), head))
        self.velMin = float(AlltoVel(head.getZphys(self.getZmin()), head))
        self.velMax = float(AlltoVel(head.getZphys(self.getZmax()), head))
        self.velWidth = abs(self.velMax - self.velMin)

    def calcIntegFlux(self, array, head, pbcorr=0, fluxconvert=True):
        f0 = array[self.z, self.y, self.x].astype(np.float64)
        f = f0
        if pbcorr:
            f = np.array([Pbcor(x, y, z, ff, head, pbcorr) for x, y, z, ff in zip(self.x, self.y, self.z, f0)])
        zmin, zmax = self.getZmin(), self.getZmax()
        zsize = zmax - zmin + 3
        zpts = np.arange(zsize) + zmin - 1
        world = np.asarray(AlltoVel(head.getZphys(zpts), head), dtype=float)
        dv = np.empty(zsize)
        dv[0] = world[1] - world[0]
        dv[-1] = world[-1] - world[-2]
        dv[1:-1] = (world[2:] - world[:-2]) / 2.
        self.intFlux = float(np.sum(f * np.abs(dv[self.z - zmin + 1])))
        # the line widths come from the spectrum of the observed fluxes, not PB-corrected (as in C++;
        # the corrected fluxes can be NaN beyond the cutoff)
        intSpec = np.zeros(array.shape[0])
        np.add.at(intSpec, self.z, f0)
        self.calcVelWidths(intSpec, head)
        if fluxconvert:
            self.intFlux = float(FluxtoJy(self.intFlux, head))

    def calcVelWidths(self, intSpec, head):
        zdim = intSpec.size
        zmin, zmax = self.getZmin(), self.getZmax()
        peak, peakLoc = 0., 0
        for z in range(zmin, zmax + 1):
            if z == 0 or peak < intSpec[z]:
                peak, peakLoc = intSpec[z], z

        def zphys(zpt):
            return float(AlltoVel(head.getZphys(zpt), head))

        def edge(frac, low):
            lev = peak * frac
            if low:
                z = zmin
                goLeft = intSpec[z] > lev
                if goLeft:
                    while z > 0 and intSpec[z] > lev: z -= 1
                else:
                    while z < peakLoc and intSpec[z] < lev: z += 1
                if z == 0:
                    return self.velMin
            else:
                z = zmax
                goLeft = intSpec[z] < lev
                if goLeft:
                    while z > peakLoc and intSpec[z] < lev: z -= 1
                else:
                    while z < zdim and intSpec[z] > lev: z += 1
                if z == zdim:
                    return self.velMax
            if goLeft:
                zp = z + (lev - intSpec[z]) / (intSpec[z + 1] - intSpec[z]) if z + 1 < zdim and intSpec[z + 1] != intSpec[z] else z
            else:
                zp = z - (lev - intSpec[z]) / (intSpec[z - 1] - intSpec[z]) if z - 1 >= 0 and intSpec[z - 1] != intSpec[z] else z
            return zphys(zp)

        self.v50min, self.v50max = edge(0.5, True), edge(0.5, False)
        self.v20min, self.v20max = edge(0.2, True), edge(0.2, False)
        self.w20 = abs(self.v20min - self.v20max)
        self.w50 = abs(self.v50min - self.v50max)

    def calcAllParams(self, array, head, pbcorr=0, fluxconvert=True):
        self.calcFluxes(array)
        self.calcWCSparams(head)
        self.calcIntegFlux(array, head, pbcorr, fluxconvert)


def SortDetections(objs, parameter="nvox"):
    keys = ("xvalue", "yvalue", "zvalue", "ra", "dec", "vel", "w20", "w50", "iflux", "pflux", "snr", "npix", "nvox")
    rev = parameter.startswith("-")
    p = parameter[1:] if rev else parameter
    if p not in keys:
        sys.stderr.write(f"SEARCH WARNING: Invalid sorting parameter: {parameter}. Not doing any sorting.\n")
        return objs
    if p in ("ra", "dec", "vel", "w20", "w50", "iflux") and not all(o.isWCS() for o in objs):
        return objs
    get = {"xvalue": lambda o: o.getXcentre(), "yvalue": lambda o: o.getYcentre(), "zvalue": lambda o: o.getZcentre(),
           "ra": lambda o: o.ra, "dec": lambda o: o.dec, "vel": lambda o: o.vel, "w20": lambda o: o.w20,
           "w50": lambda o: o.w50, "iflux": lambda o: o.intFlux, "pflux": lambda o: o.peakFlux,
           "snr": lambda o: getattr(o, "peakSNR", 0.), "npix": lambda o: o.getSpatialSize(), "nvox": lambda o: o.getSize()}[p]
    sgn = 1. if rev else -1.
    return sorted(objs, key=lambda o: sgn * get(o))


# --------------------------------------------------------------------------
# The Search class
# --------------------------------------------------------------------------
class Search:
    """Source finder. ``par`` is a SearchPar structure (see params.py)."""

    def __init__(self, par):
        import copy
        self.par = copy.deepcopy(par)
        self.objectList = []
        self.detectMap = None
        self.stats = None
        self.verbose = True

    def getNumObj(self):
        return len(self.objectList)

    def pObject(self, i):
        return self.objectList[i]

    def LargestDetection(self):
        if not self.objectList:
            return None
        sizes = [o.getSize() for o in self.objectList]
        return self.objectList[int(np.argmax(sizes))]

    # ------------------------------------------------------------------
    def search(self, array, stats=None, useRobust=True, nthreads=1, verbose=True, showbar=True):
        """Search a 3D array (nz,ny,nx). If stats is None, statistics are computed."""
        array = np.asarray(array)
        if array.ndim == 2:
            array = array[None]
        elif array.ndim == 1:
            array = array[:, None, None]
        self.array = array
        self.verbose = verbose
        if stats is None:
            stats = Stats(useRobust)
            good = ~((array == 0) | np.isnan(array))
            stats.calculate(array, good)
            stats.setThresholdSNR(self.par.snrCut)
        self.stats = stats.copy()
        if self.par.UserThreshold:
            self.stats.setThreshold(self.par.threshold)

        if verbose:
            print("\n\nStarting research for possible sources in the cube... ")
        labels, nobj = self._search3D()
        self.detectMap = None
        if verbose:
            s = "object" if nobj == 1 else "objects"
            print(f"  Intermediate list has {nobj} {s}.")
        objs = []
        if nobj > 0:
            if verbose:
                sys.stdout.write("  Merging and Rejecting...  ")
                sys.stdout.flush()
            objs = self._merger(labels, nobj)
            if verbose:
                print("Done.                      ")
        self.objectList = objs
        if verbose:
            print(f"  ... All done.\n\nFinal object count = {len(objs)}\n")
        return objs

    # ------------------------------------------------------------------
    def _footprint(self):
        p = self.par
        thV = int(math.ceil(p.threshVelocity))
        if p.flagAdjacent:
            g = 1
            sp = np.ones((3, 3), bool)
        else:
            g = int(math.ceil(p.threshSpatial))
            yy, xx = np.mgrid[-g:g + 1, -g:g + 1]
            # minSep between pixels < threshSpatial
            sp = np.hypot(xx, yy) < p.threshSpatial
        return thV, g, sp

    def _search3D(self):
        """First stage: find objects and (optionally) merge them into a list.
        Returns a label cube (int32, 0=background) and the number of objects."""
        arr = self.array
        nz, ny, nx = arr.shape
        st = self.stats
        det = arr > st.getThreshold()
        det &= ~np.isnan(arr)
        labels = np.zeros(arr.shape, np.int32)
        stype = self.par.searchType
        two = self.par.TwoStageMerging
        thV, g, sp = self._footprint()
        nobj = 0
        num = 0
        struct8 = np.ones((3, 3), bool)

        if "spatial" in stype or nz == 1:
            for z in range(nz):
                lab2, n2 = ndimage.label(det[z], structure=struct8)
                if n2 == 0:
                    continue
                num += n2
                slices = ndimage.find_objects(lab2)
                for i, sl in enumerate(slices):
                    m = lab2[sl] == (i + 1)
                    if two:
                        tgt = self._first_close(labels, z, z, sl, m, thV, g, sp)
                    else:
                        tgt = 0
                    if tgt == 0:
                        nobj += 1
                        tgt = nobj
                    labels[z][sl][m] = tgt
        elif stype == "spectral":
            for y in range(ny):
                for x in range(nx):
                    spec = det[:, y, x]
                    if not spec.any():
                        continue
                    d = np.diff(np.concatenate(([0], spec.astype(np.int8), [0])))
                    starts, ends = np.where(d == 1)[0], np.where(d == -1)[0] - 1
                    for z0, z1 in zip(starts, ends):
                        num += 1
                        sl = (slice(y, y + 1), slice(x, x + 1))
                        m = np.ones((1, 1), bool)
                        tgt = self._first_close(labels, z0, z1, sl, m, thV, g, sp) if two else 0
                        if tgt == 0:
                            nobj += 1
                            tgt = nobj
                        labels[z0:z1 + 1, y, x] = tgt
        else:
            print("Unknown search type : " + stype)
        if self.verbose:
            print(f"Searching in progress... Found {num} items.")
        return labels, nobj

    @staticmethod
    def _first_close(labels, z0, z1, sl, m, thV, g, sp):
        """Lowest label (=first in list) among labelled voxels close to object m."""
        nz, ny, nx = labels.shape
        ys, xs = sl
        ya, yb = max(ys.start - g, 0), min(ys.stop + g, ny)
        xa, xb = max(xs.start - g, 0), min(xs.stop + g, nx)
        za, zb = max(z0 - thV, 0), min(z1 + thV + 1, nz)
        sub = labels[za:zb, ya:yb, xa:xb]
        if not sub.any():
            return 0
        # dilate the 2D footprint of the object
        mm = np.zeros((yb - ya, xb - xa), bool)
        mm[ys.start - ya:ys.stop - ya, xs.start - xa:xs.stop - xa] = m
        dil = ndimage.binary_dilation(mm, structure=sp)
        vals = sub[:, dil]
        vals = vals[vals > 0]
        return int(vals.min()) if vals.size else 0

    # ------------------------------------------------------------------
    def _neighbour_pairs(self, labels, footprint=None):
        """All pairs of different labels that are 'close' (voxel based)."""
        thV, g, sp = self._footprint() if footprint is None else footprint
        # restrict to the bounding box of the labelled voxels (plus margins)
        nzi = np.nonzero(labels.any(axis=(1, 2)))[0]
        nyi = np.nonzero(labels.any(axis=(0, 2)))[0]
        nxi = np.nonzero(labels.any(axis=(0, 1)))[0]
        if nzi.size == 0:
            return np.zeros((0, 2), np.int64)
        labels = labels[nzi[0]:nzi[-1] + 1, nyi[0]:nyi[-1] + 1, nxi[0]:nxi[-1] + 1]
        nz, ny, nx = labels.shape
        pairs = []
        offs = [(dz, dy, dx) for dz in range(0, thV + 1) for dy in range(-g, g + 1)
                for dx in range(-g, g + 1) if sp[dy + g, dx + g] and (dz, dy, dx) > (0, 0, 0)]
        for dz, dy, dx in offs:
            a = labels[0:nz - dz, max(0, -dy):ny - max(0, dy), max(0, -dx):nx - max(0, dx)]
            b = labels[dz:nz, max(0, dy):ny + min(0, dy), max(0, dx):nx + min(0, dx)]
            sel = (a > 0) & (b > 0) & (a != b)
            if sel.any():
                pa, pb = a[sel], b[sel]
                pairs.append(np.unique(np.stack([pa, pb], 1), axis=0))
        if pairs:
            return np.unique(np.concatenate(pairs), axis=0)
        return np.zeros((0, 2), np.int64)

    def _merge_labels(self, labels, nlab, footprint=None):
        """Transitive merging of labels; return a relabelled cube and count."""
        pairs = self._neighbour_pairs(labels, footprint)
        if pairs.shape[0] == 0:
            return labels, nlab
        n = nlab + 1
        G = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
        ncomp, comp = connected_components(G, directed=False)
        # order the components by their first (lowest) label; label 0 stays alone
        first = np.full(ncomp, n, np.int64)
        np.minimum.at(first, comp[1:], np.arange(1, n))
        order = np.argsort(first, kind="stable")
        nvalid = int(np.sum(first < n))
        newid = np.zeros(ncomp, np.int32)
        newid[order[:nvalid]] = np.arange(1, nvalid + 1)
        lut = newid[comp]
        lut[0] = 0
        return lut[labels].astype(np.int32), nvalid

    def _objects_from_labels(self, labels, n):
        objs = []
        if n == 0:
            return objs
        idx = np.nonzero(labels)
        lab = labels[idx]
        order = np.argsort(lab, kind="stable")
        lab = lab[order]
        z, y, x = idx[0][order], idx[1][order], idx[2][order]
        bounds = np.searchsorted(lab, np.arange(1, n + 2))
        for i in range(n):
            a, b = bounds[i], bounds[i + 1]
            if b > a:
                objs.append(Detection(z[a:b], y[a:b], x[a:b]))
        return objs

    def _finalise(self, labels, n):
        """Reject objects not meeting the requirements; return new labels."""
        p = self.par
        if n == 0:
            return labels, 0
        keep = np.zeros(n + 1, bool)
        objs = self._objects_from_labels(labels, n)
        ids = np.unique(labels[labels > 0])
        for i, o in zip(ids, objs):
            nchan = o.getNumChannels()
            size = max(abs(o.getXmax() - o.getXmin()), abs(o.getYmax() - o.getYmin()))
            ok = (o.hasEnoughChannels(p.minChannels) and o.getSpatialSize() >= p.minPix and
                  o.getSize() >= p.minVoxels and (p.maxAngSize <= 0 or size < p.maxAngSize) and
                  (p.maxChannels <= 0 or nchan < p.maxChannels))
            keep[i] = ok
        lut = np.zeros(n + 1, np.int32)
        kept = np.where(keep)[0]
        lut[kept] = np.arange(1, kept.size + 1)
        return lut[labels], int(kept.size)

    def _merger(self, labels, n):
        p = self.par
        if p.RejectBeforeMerge:
            labels, n = self._finalise(labels, n)
        labels, n = self._merge_labels(labels, n)
        if p.flagGrowth and n > 0:
            gst = self.stats.copy()
            if p.flagUserGrowthT:
                gst.setThreshold(p.growthThreshold)
            else:
                gst.setThresholdSNR(p.growthCut)
            labels, n = self._grow(labels, n, gst)
        labels, n = self._finalise(labels, n)
        self.labels = labels
        objs = self._objects_from_labels(labels, n)
        # detection map: number of detected channels per spatial pixel
        self.detectMap = (labels > 0).sum(axis=0).astype(np.int16)
        return objs

    def _grow(self, labels, n, gst):
        """Grow objects through voxels above the growth threshold.

        In BBarolo each object is flood-filled through neighbouring voxels above
        the growth threshold and close objects are then merged: the result is the
        set of connected components (under the merging criterion) of the region
        reachable from the detections, which is what is computed here.
        """
        thV, g, sp = self._footprint()
        if not self.par.flagAdjacent:
            g = int(self.par.threshSpatial)
            sp = np.ones((2 * g + 1, 2 * g + 1), bool)
        arr = self.array
        seed = labels > 0
        allowed = (arr > gst.getThreshold()) & ~np.isnan(arr)
        allowed |= seed
        # connected components of the allowed region under the merging criterion: labelling with the
        # neighbours within the velocity/spatial thresholds (26-connectivity unless THRESHVELOCITY or the
        # spatial threshold is 0: then not across channels or pixels, as the growth box of C++), followed by
        # the merging of components that are closer than the thresholds
        dz, ds = min(thV, 1), min(g, 1)
        st = np.zeros((3, 3, 3), bool)
        st[1 - dz:2 + dz, 1 - ds:2 + ds, 1 - ds:2 + ds] = True
        lab0, n0 = ndimage.label(allowed, structure=st)
        lab_cc, ncc = self._merge_labels(lab0.astype(np.int32), n0, footprint=(thV, g, sp))
        # components containing detections, ordered by their first (lowest) original label
        first = np.full(ncc + 1, np.iinfo(np.int64).max, np.int64)
        np.minimum.at(first, lab_cc[seed], labels[seed].astype(np.int64))
        comps = np.where(first < np.iinfo(np.int64).max)[0]
        comps = comps[comps > 0]
        comps = comps[np.argsort(first[comps])]
        lut = np.zeros(ncc + 1, np.int32)
        lut[comps] = np.arange(1, comps.size + 1)
        return lut[lab_cc], int(comps.size)
