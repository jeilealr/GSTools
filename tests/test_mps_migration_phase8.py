"""Phase 8 regressions: deterministic parallel post-processing (cleanup).

The native cleanup pass (ABI rev 7) runs over a position-oriented dependency
DAG with the same level scheduler as the main engine, so its output is
bit-identical for any thread count (serial == parallel). These tests exercise
the parallel branch (>512-node cleanup paths clear the native small-path
cutoff) and pin thread-count invariance, including with zones.
"""

import numpy as np
import pytest

import gstools as gs
from gstools.mps import simulate as _sim


@pytest.fixture
def capture_cleanup_threads():
    """Record the effective thread count of each cleanup pass."""
    seen = []
    _sim._MPS_CLEANUP_STATS_HOOK = lambda stats: seen.append(
        stats["used_threads"]
    )
    try:
        yield seen
    finally:
        _sim._MPS_CLEANUP_STATS_HOOK = None


def _big_ti():
    rng = np.random.default_rng(2024)
    # Categorical TI; sim grid below is 24x24 = 576 nodes (> the 512 cutoff).
    return gs.mps.TrainingImage(
        (rng.random((30, 30)) < 0.45).astype(float), n_neighbors=16
    )


_BIG_POS = (np.arange(24.0), np.arange(24.0))


def test_cleanup_serial_equals_parallel(capture_cleanup_threads):
    ti = _big_ti()
    kw = dict(scan_fraction=0.4, threshold=0.05, post_processing=2)
    f1 = gs.mps.DirectSampling(gs.mps.MPSModel(ti, **kw))(
        _BIG_POS, seed=7, num_threads=1
    )
    f4 = gs.mps.DirectSampling(gs.mps.MPSModel(ti, **kw))(
        _BIG_POS, seed=7, num_threads=4
    )
    np.testing.assert_array_equal(f1, f4)
    # The 576-node run with num_threads=4 must have actually gone parallel
    # (otherwise serial==parallel would be a vacuous assertion); the
    # num_threads=1 passes stay at 1.
    assert capture_cleanup_threads == [1, 1, 4, 4]


def test_cleanup_parallel_is_reproducible():
    ti = _big_ti()
    kw = dict(scan_fraction=0.4, threshold=0.05, post_processing=2)

    def run():
        return gs.mps.DirectSampling(gs.mps.MPSModel(ti, **kw))(
            _BIG_POS, seed=13, num_threads=4
        )

    np.testing.assert_array_equal(run(), run())


def test_cleanup_serial_equals_parallel_random_post_path():
    # A random post path still orders the DAG by (per-pass) visit position, so
    # thread-count invariance must hold there too.
    ti = _big_ti()
    kw = dict(
        scan_fraction=0.4,
        threshold=0.05,
        post_processing=2,
        post_processing_path="random",
    )
    f1 = gs.mps.DirectSampling(gs.mps.MPSModel(ti, **kw))(
        _BIG_POS, seed=21, num_threads=1
    )
    f8 = gs.mps.DirectSampling(gs.mps.MPSModel(ti, **kw))(
        _BIG_POS, seed=21, num_threads=8
    )
    np.testing.assert_array_equal(f1, f8)


def test_cleanup_serial_equals_parallel_with_zones(capture_cleanup_threads):
    primary = _big_ti()
    rng = np.random.default_rng(99)
    zone_ti = gs.mps.TrainingImage(
        (rng.random((30, 30)) < 0.7).astype(float), n_neighbors=16
    )
    region = np.zeros((24, 24), dtype=bool)
    region[:12] = True  # top half scans the zone TI
    zone = gs.mps.Zone(zone_ti, region)
    kw = dict(
        scan_fraction=0.4, threshold=0.05, post_processing=1, zones=[zone]
    )
    f1 = gs.mps.DirectSampling(gs.mps.MPSModel(primary, **kw))(
        _BIG_POS, seed=5, num_threads=1
    )
    f4 = gs.mps.DirectSampling(gs.mps.MPSModel(primary, **kw))(
        _BIG_POS, seed=5, num_threads=4
    )
    np.testing.assert_array_equal(f1, f4)
    assert capture_cleanup_threads == [1, 4]
