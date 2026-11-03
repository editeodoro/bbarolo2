"""
Fast random numbers for the compiled kernels: xoroshiro128+ uniform generator
(seeded with splitmix64) and Marsaglia & Tsang's ziggurat for normal deviates.

They are ~5-10x faster than numba's np.random and, since the state is an explicit
array, each model evaluation has its own reproducible stream (seeded by ISEED),
independently of the thread that computes it.
"""

import math
import numpy as np
import numba as nb

_U64 = np.uint64


def _zig_tables():
    dn = 3.442619855899
    tn = dn
    vn = 9.91256303526217e-3
    m1 = 2147483648.0
    kn = np.zeros(128, np.int64)
    wn = np.zeros(128, np.float64)
    fn = np.zeros(128, np.float64)
    q = vn / math.exp(-.5 * dn * dn)
    kn[0] = int((dn / q) * m1)
    kn[1] = 0
    wn[0] = q / m1
    wn[127] = dn / m1
    fn[0] = 1.
    fn[127] = math.exp(-.5 * dn * dn)
    for i in range(126, 0, -1):
        dn = math.sqrt(-2. * math.log(vn / dn + math.exp(-.5 * dn * dn)))
        kn[i + 1] = int((dn / tn) * m1)
        tn = dn
        fn[i] = math.exp(-.5 * dn * dn)
        wn[i] = dn / m1
    return kn, wn, fn


KN, WN, FN = _zig_tables()


@nb.njit(inline="always", cache=True)
def _rotl(x, k):
    return (x << _U64(k)) | (x >> _U64(64 - k))


@nb.njit(cache=True)
def rng_init(seed):
    """Return a xoroshiro128+ state (2 x uint64) initialised from an integer seed."""
    s = np.empty(2, np.uint64)
    z = _U64(seed) + _U64(0x9E3779B97F4A7C15)
    for i in range(2):
        z = z + _U64(0x9E3779B97F4A7C15)
        x = z
        x = (x ^ (x >> _U64(30))) * _U64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> _U64(27))) * _U64(0x94D049BB133111EB)
        x = x ^ (x >> _U64(31))
        s[i] = x
    if s[0] == 0 and s[1] == 0:
        s[0] = _U64(1)
    return s


@nb.njit(inline="always", cache=True)
def rng_next(s):
    s0 = s[0]
    s1 = s[1]
    result = s0 + s1
    s1 ^= s0
    s[0] = _rotl(s0, 24) ^ s1 ^ (s1 << _U64(16))
    s[1] = _rotl(s1, 37)
    return result


@nb.njit(inline="always", cache=True)
def rng_uniform(s):
    """Uniform deviate in [0,1)."""
    return (rng_next(s) >> _U64(11)) * (1.0 / 9007199254740992.0)


@nb.njit(inline="always", cache=True)
def rng_normal(s):
    """Standard normal deviate (Marsaglia & Tsang ziggurat, fully inlined)."""
    while True:
        hz = np.int64(rng_next(s) >> _U64(32)) - 2147483648
        iz = hz & 127
        if abs(hz) < KN[iz]:
            return hz * WN[iz]
        x = hz * WN[iz]
        if iz == 0:
            while True:
                xx = -math.log(1.0 - rng_uniform(s)) * 0.2904764
                yy = -math.log(1.0 - rng_uniform(s))
                if yy + yy >= xx * xx:
                    break
            return 3.442620 + xx if hz > 0 else -3.442620 - xx
        if FN[iz] + rng_uniform(s) * (FN[iz - 1] - FN[iz]) < math.exp(-.5 * x * x):
            return x
