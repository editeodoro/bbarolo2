"""BBarolo's additions to nautilus (not upstream tests): the line of progress with the time left."""
import numpy as np
from scipy.stats import chi2

from bbarolo.bayesian.nautilus import Sampler
from bbarolo.bayesian.nautilus.sampler import _format_time, _format_time_left


def likelihood(x):
    return -0.5 * np.sum(((x - 0.5) / 0.05)**2)


def test_format_time():
    assert [_format_time(t) for t in (0, 44.6, 60, 185, 3599, 3600, 5100)] == [
        '0s', '45s', '1m00s', '3m05s', '59m59s', '1h00m', '1h25m']
    # the time left: a rough estimate, in minutes
    assert [_format_time_left(t) for t in (0, 59.9, 60, 89, 91, 185, 3569, 3599, 5100)] == [
        '<1m', '<1m', '~1m', '~1m', '~2m', '~3m', '~59m', '~1h00m', '~1h25m']


def test_time_left():
    # the exploration: the shrinking of the live set still needed (Gaussian near the peak) over its rate, from the
    # history; the sampling: the effective samples still needed over the efficiency and the rate of calls
    s = Sampler(lambda u: u, likelihood, n_dim=4, n_live=100, seed=1)
    assert s.time_left() is None                                # (outside run())
    s._progress_start(False, 0., 0.01, 1000)
    assert s.time_left() is None                                # (no history)
    rate, gap = 0.5, 30.
    s._progress['history'] = [(t, 100 * t, 0., -rate * t, 10., 10. - gap) for t in np.arange(1., 11.)]
    gap_end = chi2.ppf(0.01, 4) / 2
    assert np.isclose(s.time_left(), 2 * np.log(gap / gap_end) / rate)
    s._progress['history'][-1] = (10., 1000, 0., -5., 10., 10. - gap_end / 2)   # past the end: no time left
    assert s.time_left() == 0
    s._progress['history'] = s._progress['history'][:2]                       # (too early)
    assert s.time_left() is None


def test_progress_lines(capsys):
    # verbose 'lines' (and True when the output is not a terminal): a header, lines at intervals, the final line;
    # nothing updated in place
    s = Sampler(lambda u: u, likelihood, n_dim=3, n_live=200, seed=1)
    s.run(verbose='lines', n_eff=500)
    out = capsys.readouterr().out
    assert '\r' not in out
    lines = out.splitlines()
    assert lines[1].split() == ['Status', '|', 'Bounds', '|', 'Calls', '|', 'f_live', '|', 'N_eff', '|', 'log', 'Z',
                                '|', 'Time', '|', 'Left']
    assert lines[-1].startswith('Finished') and lines[-1] == lines[-1].rstrip()
    s = Sampler(lambda u: u, likelihood, n_dim=3, n_live=200, seed=1)
    s.run(verbose=True, n_eff=500)                       # (pytest: not a terminal)
    assert '\r' not in capsys.readouterr().out
