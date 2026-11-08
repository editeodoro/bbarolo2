"""
Robust statistics (port of Arrays/stats.hh, stats.cpp and Utilities/statistics.cpp).
"""

import sys
import numpy as np

correctionFactor = 0.6744888


def madfmToSigma(madfm):
    return madfm / correctionFactor


def sigmaToMADFM(sigma):
    return sigma * correctionFactor


def findMedian(a):
    a = np.asarray(a)
    return float(np.median(a)) if a.size else np.nan


def findMADFM(a, median=None):
    a = np.asarray(a, dtype=np.float64)
    if median is None:
        median = np.median(a)
    return float(np.median(np.abs(a - median))) if a.size else np.nan


def findAllStats(array, mask=None):
    """Return mean, stddev, median, madfm, max, min of array[mask]."""
    a = np.asarray(array)
    if mask is not None:
        a = a[mask]
    a = a.ravel()
    if a.size == 0:
        sys.stderr.write("Error in findAllStats: no good values!\n")
        return 0., 0., 0., 0., 0., 0.
    a64 = a.astype(np.float64, copy=False)
    mean = a64.mean()
    stddev = np.sqrt(max(np.mean(a64 * a64) - mean * mean, 0.))
    median = np.median(a64)
    madfm = np.median(np.abs(a64 - median))
    return mean, stddev, median, madfm, a64.max(), a64.min()


class Stats:
    """Statistics of an array with robust/standard estimators and a threshold."""

    def __init__(self, robust=True):
        self.useRobust = robust
        self.defined = False
        self.mean = self.stddev = self.median = self.madfm = 0.
        self.max_val = self.min_val = 0.
        self.threshold = 0.

    def copy(self):
        s = Stats(self.useRobust)
        s.__dict__.update(self.__dict__)
        return s

    def setRobust(self, b): self.useRobust = b
    def getRobust(self): return self.useRobust
    def getMean(self): return self.mean
    def getStddev(self): return self.stddev
    def getMedian(self): return self.median
    def getMadfm(self): return self.madfm
    def getMax(self): return self.max_val
    def getMin(self): return self.min_val
    def getThreshold(self): return self.threshold
    def setThreshold(self, t): self.threshold = t

    def getMiddle(self):
        return self.median if self.useRobust else self.mean

    def setMiddle(self, m):
        if self.useRobust: self.median = m
        else: self.mean = m

    def getSpread(self):
        return madfmToSigma(self.madfm) if self.useRobust else self.stddev

    def setSpread(self, s):
        if self.useRobust: self.madfm = sigmaToMADFM(s)
        else: self.stddev = s

    def getThresholdSNR(self):
        return (self.threshold - self.getMiddle()) / self.getSpread()

    def setThresholdSNR(self, snr):
        self.threshold = self.getMiddle() + snr * self.getSpread()

    def valueToSNR(self, v):
        return (v - self.getMiddle()) / self.getSpread()

    def snrToValue(self, snr):
        return snr * self.getSpread() + self.getMiddle()

    def scaleNoise(self, scale):
        snr = (self.threshold - self.getMiddle()) / self.getSpread()
        self.madfm *= scale
        self.stddev *= scale
        self.threshold = self.getMiddle() + snr * self.getSpread()

    def calculate(self, array, mask=None):
        (self.mean, self.stddev, self.median, self.madfm,
         self.max_val, self.min_val) = findAllStats(array, mask)
        self.defined = True

    def tofile(self, filename):
        with open(filename, "w") as f:
            m = 13
            f.write(f"{'# MEAN':<{m}}{'MEDIAN':<{m}}{'STDDEV':<{m}}{'MADFM':<{m}}{'MADasSTDDEV':<{m}}{'MAX':<{m}}{'MIN':<{m}}\n")
            vals = [self.mean, self.median, self.stddev, self.madfm, madfmToSigma(self.madfm), self.max_val, self.min_val]
            f.write("".join(f"{v:<{m}.4e}" for v in vals) + "\n")

    def __str__(self):
        return (f" Min    = {self.min_val:12.5e}\t Max      = {self.max_val:12.5e}\n"
                f" Mean   = {self.mean:12.5e}\t Std.Dev. = {self.stddev:12.5e}\n"
                f" Median = {self.median:12.5e}\t MADFM    = {self.madfm:12.5e} (= {madfmToSigma(self.madfm):.5e} as std.dev.)\n")
