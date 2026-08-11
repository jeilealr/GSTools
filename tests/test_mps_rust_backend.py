"""Regression tests: Rust MPS kernels produce output identical to the Python path.

Skipped automatically when gstools_core is not installed (e.g. CI without Rust build).
The flag gs.config.USE_GSTOOLS_CORE is restored after each test by the fixture.
"""

from __future__ import annotations

import numpy as np
import pytest

try:
    import gstools_core  # noqa: F401

    _CORE_AVAIL = True
except ImportError:
    _CORE_AVAIL = False

pytestmark = pytest.mark.skipif(
    not _CORE_AVAIL, reason="gstools_core not installed"
)

import gstools as gs
from gstools.mps import DirectSampling, MPSModel, TrainingImage
from gstools.mps.distance import (
    vec_categorical_dist,
    vec_l1_dist,
    vec_variation_dist,
)
from gstools.mps.training_image import Variable


@pytest.fixture(autouse=True)
def _restore_core_flag():
    """Restore USE_GSTOOLS_CORE after every test that changes it."""
    original = gs.config.USE_GSTOOLS_CORE
    yield
    gs.config.USE_GSTOOLS_CORE = original


def _cat_ti(shape=(60, 60)):
    gx, gy = np.meshgrid(
        np.arange(shape[0]), np.arange(shape[1]), indexing="ij"
    )
    return ((np.sin(gx / 5.0) + np.sin((gx + gy) / 8.0)) > 0).astype(np.int32)


def _cont_ti(shape=(60, 60)):
    gx, gy = np.meshgrid(
        np.arange(shape[0]), np.arange(shape[1]), indexing="ij"
    )
    return np.sin(gx / 6.0) * np.cos(gy / 8.0)


def _run(
    ti_data,
    categorical,
    distance,
    sg_shape,
    scan_fraction,
    threshold,
    seed,
    use_rust,
):
    gs.config.USE_GSTOOLS_CORE = use_rust
    ti = TrainingImage(ti_data, categorical=categorical, distance=distance)
    model = MPSModel(ti, scan_fraction=scan_fraction, threshold=threshold)
    ds = DirectSampling(model, seed=seed)
    return ds([np.arange(s, dtype=float) for s in sg_shape], store=False)


def _mixed_clean_masked_ds(num_threads=1):
    """Build the deterministic mixed-TI Stage 2 regression fixture."""
    clean = _cat_ti((32, 32)).astype(float)
    masked = clean.copy()
    masked[9:14, 10:15] = np.nan
    with pytest.warns(UserWarning, match="contains NaN"):
        ti = TrainingImage(
            [
                Variable(
                    "clean",
                    clean,
                    categorical=False,
                    distance="l1",
                    n_neighbors=8,
                    weight=0.5,
                ),
                Variable(
                    "masked",
                    masked,
                    categorical=True,
                    n_neighbors=8,
                    weight=0.5,
                ),
            ]
        )
    ds = DirectSampling(
        MPSModel(ti, scan_fraction=0.4, threshold=0.0), seed=20260809
    )
    ds.num_threads = num_threads
    # Each of the first two nodes constrains only one variable. The other
    # variable is simulated from the same selected TI anchor.
    ds.set_condition(
        [[1.0, 5.0, 8.0], [2.0, 6.0, 3.0]],
        {
            "clean": np.array([1.0, np.nan, 0.0]),
            "masked": np.array([np.nan, 0.0, 0.0]),
        },
    )
    return ds, [np.arange(12.0), np.arange(12.0)]


def _run_mixed_clean_masked(use_rust, num_threads=1):
    gs.config.USE_GSTOOLS_CORE = use_rust
    ds, pos = _mixed_clean_masked_ds(num_threads=num_threads)
    return ds(pos, store=False)


# ---------------------------------------------------------------------------
# 1. Categorical DSBC (threshold=0.0) — 20x20 SG, 60x60 TI
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [42, 1234, 99999])
def test_categorical_dsbc_output_identical(seed):
    """DSBC mode: Rust and Python outputs must be bit-identical."""
    ti = _cat_ti()
    py = _run(ti, True, "l1", (20, 20), 0.3, 0.0, seed, False)
    rs = _run(ti, True, "l1", (20, 20), 0.3, 0.0, seed, True)
    np.testing.assert_array_equal(py, rs)


# ---------------------------------------------------------------------------
# 2. Categorical DS mode (threshold=0.1) — 20x20 SG, 60x60 TI
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [42, 1234])
def test_categorical_ds_output_identical(seed):
    """DS (greedy threshold) mode: Rust and Python outputs must be bit-identical."""
    ti = _cat_ti()
    py = _run(ti, True, "l1", (20, 20), 0.3, 0.1, seed, False)
    rs = _run(ti, True, "l1", (20, 20), 0.3, 0.1, seed, True)
    np.testing.assert_array_equal(py, rs)


# ---------------------------------------------------------------------------
# 2a. Categorical labels may be arbitrary numeric values
# ---------------------------------------------------------------------------


def test_close_non_integer_categorical_labels_output_identical():
    """Distinct labels closer than 0.5 remain distinct in the Rust scan."""
    ti = np.where(_cat_ti((30, 30)) == 0, 0.1, 0.2)
    np.testing.assert_array_equal(np.unique(ti), np.array([0.1, 0.2]))

    py = _run(ti, True, "l1", (15, 15), 0.3, 0.0, 42, False)
    rs = _run(ti, True, "l1", (15, 15), 0.3, 0.0, 42, True)

    np.testing.assert_array_equal(py, rs)


# ---------------------------------------------------------------------------
# 3. Continuous L1 — 15x15 SG, 40x40 TI
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [42, 1234])
def test_l1_output_close(seed):
    """L1 distance: Rust and Python outputs must agree to rtol=1e-10."""
    ti = _cont_ti((40, 40))
    py = _run(ti, False, "l1", (15, 15), 0.3, 0.0, seed, False)
    rs = _run(ti, False, "l1", (15, 15), 0.3, 0.0, seed, True)
    np.testing.assert_allclose(py, rs, rtol=1e-10)


# ---------------------------------------------------------------------------
# 4. Continuous L2 — 15x15 SG, 40x40 TI
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [42, 1234])
def test_l2_output_close(seed):
    """L2 distance: Rust and Python outputs must agree to rtol=1e-10."""
    ti = _cont_ti((40, 40))
    py = _run(ti, False, "l2", (15, 15), 0.3, 0.0, seed, False)
    rs = _run(ti, False, "l2", (15, 15), 0.3, 0.0, seed, True)
    np.testing.assert_allclose(py, rs, rtol=1e-10)


# ---------------------------------------------------------------------------
# 5. Continuous Lp (p=3) — 15x15 SG, 40x40 TI
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [42])
def test_lp_output_close(seed):
    """Lp kernel with p=3: Rust and Python outputs must agree to rtol=1e-10."""
    ti = _cont_ti((40, 40))
    py = _run(ti, False, "l3", (15, 15), 0.3, 0.0, seed, False)
    rs = _run(ti, False, "l3", (15, 15), 0.3, 0.0, seed, True)
    np.testing.assert_allclose(py, rs, rtol=1e-10)


# ---------------------------------------------------------------------------
# 6. Variation distance — NaN-free Rust block kernel must match the Python path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [42, 1234])
@pytest.mark.parametrize("threshold", [0.0, 0.1])
def test_variation_output_close(seed, threshold):
    """Variation (default p=2), DSBC and DS: Rust kernel matches Python to rtol=1e-10."""
    ti = _cont_ti((40, 40))
    py = _run(ti, False, "variation", (12, 12), 0.3, threshold, seed, False)
    rs = _run(ti, False, "variation", (12, 12), 0.3, threshold, seed, True)
    np.testing.assert_allclose(py, rs, rtol=1e-10)


@pytest.mark.parametrize(
    "distance", ["variation1", "variation1.5", "variation2"]
)
def test_variation_p_variants_output_close(distance):
    """Variation with explicit exponents p=1, 1.5, 2 must match the Python path."""
    ti = _cont_ti((40, 40))
    py = _run(ti, False, distance, (12, 12), 0.3, 0.0, 42, False)
    rs = _run(ti, False, distance, (12, 12), 0.3, 0.0, 42, True)
    np.testing.assert_allclose(py, rs, rtol=1e-10)


def test_variation_wrapper_matches_python_randomized():
    """Compare the public PyO3 wrapper directly with the NumPy oracle."""
    rng = np.random.default_rng(20260809)
    for n in (2, 8, 17):
        for n_candidates in (1, 7, 32):
            for p in (0.5, 1.0, 1.5, 2.0, 3.0):
                de_sim = rng.normal(size=n)
                all_de_ti = rng.normal(size=(n_candidates, n))
                node_weights = rng.random(n)
                node_weights /= node_weights.sum()
                d_max = float(rng.uniform(0.1, 4.0))
                base = np.arange(n_candidates, dtype=np.int64) * n
                lags = np.arange(n, dtype=np.int64)

                expected = vec_variation_dist(
                    de_sim,
                    all_de_ti,
                    node_weights,
                    d_max,
                    p=p,
                    has_nan=False,
                )
                actual = gstools_core.mps_dist_block_variation(
                    de_sim,
                    all_de_ti.ravel(),
                    base,
                    lags,
                    node_weights,
                    d_max,
                    p,
                )
                np.testing.assert_allclose(
                    actual, expected, rtol=1e-12, atol=1e-13
                )


def test_variation_dispatch_calls_rust_kernel():
    """Prove the Rust variation kernel is actually invoked (not the Python fallback).

    A parity test can pass while silently taking the Python path, so spy on the
    module-level backend function and assert it was called.
    """
    from gstools.mps import scan as _scan

    calls = {"n": 0}
    original = _scan._mps_dist_block_variation_gsc

    def _spy(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    _scan._mps_dist_block_variation_gsc = _spy
    try:
        ti = _cont_ti((40, 40))
        _run(ti, False, "variation", (12, 12), 0.3, 0.0, 42, True)
    finally:
        _scan._mps_dist_block_variation_gsc = original
    assert calls["n"] > 0, "Rust variation kernel was never called"


def test_variation_dispatch_python_when_backend_off():
    """With the backend disabled, the Rust variation kernel must NOT be called."""
    from gstools.mps import scan as _scan

    calls = {"n": 0}
    original = _scan._mps_dist_block_variation_gsc

    def _spy(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    _scan._mps_dist_block_variation_gsc = _spy
    try:
        ti = _cont_ti((40, 40))
        _run(ti, False, "variation", (12, 12), 0.3, 0.0, 42, False)
    finally:
        _scan._mps_dist_block_variation_gsc = original
    assert calls["n"] == 0, "Rust variation kernel called despite backend off"


@pytest.mark.parametrize(
    ("distance", "backend_name"),
    [
        ("l1", "_mps_dist_block_l1_gsc"),
        ("l2", "_mps_dist_block_l2_gsc"),
        ("l3", "_mps_dist_block_lp_gsc"),
        ("variation", "_mps_dist_block_variation_gsc"),
    ],
)
def test_missing_continuous_core_export_falls_back(
    distance, backend_name, monkeypatch
):
    """Each missing optional export falls back independently to Python."""
    from gstools.mps import scan as _scan

    ti = _cont_ti((40, 40))
    expected = _run(ti, False, distance, (12, 12), 0.3, 0.0, 42, False)
    monkeypatch.setattr(_scan, backend_name, None)
    actual = _run(ti, False, distance, (12, 12), 0.3, 0.0, 42, True)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    ("distance", "backend_name"),
    [
        ("l1", "_mps_dist_block_l1_gsc"),
        ("l2", "_mps_dist_block_l2_gsc"),
        ("l3", "_mps_dist_block_lp_gsc"),
    ],
)
def test_continuous_dispatch_calls_expected_kernel(
    distance, backend_name, monkeypatch
):
    """Prove each existing continuous block kernel is reached."""
    from gstools.mps import scan as _scan

    calls = {"n": 0}
    original = getattr(_scan, backend_name)

    def _spy(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(_scan, backend_name, _spy)
    _run(
        _cont_ti((32, 32)),
        False,
        distance,
        (10, 10),
        0.3,
        0.0,
        42,
        True,
    )
    assert calls["n"] > 0, f"{backend_name} was never called"


def test_categorical_full_scan_dispatch_calls_rust(monkeypatch):
    """Prove the eligible categorical path reaches the full Rust scan."""
    from gstools.mps import scan as _scan

    calls = {"n": 0}
    original = _scan._mps_scan_node_cat_gsc

    def _spy(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(_scan, "_mps_scan_node_cat_gsc", _spy)
    _run(_cat_ti((32, 32)), True, "l1", (10, 10), 0.3, 0.0, 42, True)
    assert calls["n"] > 0, "full categorical Rust scan was never called"


def test_multivariate_categorical_dispatch_calls_block_kernel(monkeypatch):
    """Prove a joint categorical scan reaches the Rust block kernel."""
    from gstools.mps import scan as _scan

    calls = {"n": 0}
    original = _scan._mps_dist_block_cat_gsc

    def _spy(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(_scan, "_mps_dist_block_cat_gsc", _spy)
    a = _cat_ti((32, 32)).astype(float)
    b = (np.indices((32, 32)).sum(axis=0) % 3).astype(float)
    ti = TrainingImage(
        [
            Variable("a", a, categorical=True, n_neighbors=4),
            Variable("b", b, categorical=True, n_neighbors=4),
        ]
    )
    ds = DirectSampling(MPSModel(ti, scan_fraction=0.3), seed=42)
    gs.config.USE_GSTOOLS_CORE = True
    ds([np.arange(10.0), np.arange(10.0)], store=False)
    assert calls["n"] > 0, "categorical Rust block kernel was never called"


def test_variation_nan_ti_does_not_use_rust_kernel():
    """Masked variation is not yet in Rust: a NaN in the TI must fall back to Python.

    The NaN-free Rust variation kernel must never run on a TI containing NaN
    (it has no per-candidate common-support renormalization). Parity with the
    pure-Python masked path must still hold.
    """
    from gstools.mps import scan as _scan

    ti = _cont_ti((40, 40))
    ti[5:9, 5:9] = np.nan  # undefined patch → ti_has_nan True

    calls = {"n": 0}
    original = _scan._mps_dist_block_variation_gsc

    def _spy(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    _scan._mps_dist_block_variation_gsc = _spy
    try:
        py = _run(ti, False, "variation", (12, 12), 0.3, 0.0, 42, False)
        rs = _run(ti, False, "variation", (12, 12), 0.3, 0.0, 42, True)
    finally:
        _scan._mps_dist_block_variation_gsc = original
    assert calls["n"] == 0, "Rust variation kernel ran on a NaN TI"
    np.testing.assert_array_equal(py, rs)


# ---------------------------------------------------------------------------
# 6a. Stage 2 — per-variable NaN dispatch
# ---------------------------------------------------------------------------


def test_mixed_nan_dispatches_clean_and_masked_rust(monkeypatch):
    """Stage 3 accelerates the masked sibling without changing joint control."""
    from gstools.mps import scan as _scan

    rust_calls = {"clean_l1": 0, "masked_cat": 0}
    python_calls = []
    original_l1 = _scan._mps_dist_block_l1_gsc
    original_cat = _scan._mps_dist_block_cat_masked_gsc
    original_python = TrainingImage.vec_distance_var

    def _l1_spy(*args, **kwargs):
        rust_calls["clean_l1"] += 1
        return original_l1(*args, **kwargs)

    def _cat_spy(*args, **kwargs):
        rust_calls["masked_cat"] += 1
        return original_cat(*args, **kwargs)

    def _python_spy(self, var, *args, **kwargs):
        python_calls.append((var, kwargs.get("has_nan")))
        return original_python(self, var, *args, **kwargs)

    monkeypatch.setattr(_scan, "_mps_dist_block_l1_gsc", _l1_spy)
    monkeypatch.setattr(_scan, "_mps_dist_block_cat_masked_gsc", _cat_spy)
    monkeypatch.setattr(TrainingImage, "vec_distance_var", _python_spy)

    result = _run_mixed_clean_masked(True)

    assert rust_calls["clean_l1"] > 0
    assert rust_calls["masked_cat"] > 0
    assert python_calls == []
    assert np.isfinite(result["clean"]).all()
    assert np.isfinite(result["masked"]).all()


def test_mixed_nan_backend_parity_repeatability_and_parallelism():
    """Stage 2 preserves complete fields, RNG use, and the shared anchor."""
    python = _run_mixed_clean_masked(False, num_threads=1)
    rust_a = _run_mixed_clean_masked(True, num_threads=1)
    rust_b = _run_mixed_clean_masked(True, num_threads=1)
    rust_parallel = _run_mixed_clean_masked(True, num_threads=4)

    for variable in ("clean", "masked"):
        np.testing.assert_array_equal(python[variable], rust_a[variable])
        np.testing.assert_array_equal(rust_a[variable], rust_b[variable])
        np.testing.assert_array_equal(
            rust_a[variable], rust_parallel[variable]
        )
        assert np.isfinite(rust_a[variable]).all()

    # The two TI variables have the same defined values. Equality throughout
    # therefore proves that both targets were copied from one shared anchor,
    # including the two partially conditioned nodes above.
    np.testing.assert_array_equal(rust_a["clean"], rust_a["masked"])


def test_masked_univariate_categorical_uses_masked_block(monkeypatch):
    """Masked categorical uses its block kernel, never the full scan."""
    from gstools.mps import scan as _scan

    calls = {"full_scan": 0, "plain_block": 0, "masked_block": 0}
    original_full = _scan._mps_scan_node_cat_gsc
    original_plain = _scan._mps_dist_block_cat_gsc
    original_masked = _scan._mps_dist_block_cat_masked_gsc

    def _full_spy(*args, **kwargs):
        calls["full_scan"] += 1
        return original_full(*args, **kwargs)

    def _plain_spy(*args, **kwargs):
        calls["plain_block"] += 1
        return original_plain(*args, **kwargs)

    def _masked_spy(*args, **kwargs):
        calls["masked_block"] += 1
        return original_masked(*args, **kwargs)

    monkeypatch.setattr(_scan, "_mps_scan_node_cat_gsc", _full_spy)
    monkeypatch.setattr(_scan, "_mps_dist_block_cat_gsc", _plain_spy)
    monkeypatch.setattr(_scan, "_mps_dist_block_cat_masked_gsc", _masked_spy)
    data = _cat_ti((24, 24)).astype(float)
    data[7:10, 8:12] = np.nan
    with pytest.warns(UserWarning, match="contains NaN"):
        result = _run(
            data,
            True,
            "l1",
            (10, 10),
            0.4,
            0.0,
            42,
            True,
        )

    assert calls["full_scan"] == 0
    assert calls["plain_block"] == 0
    assert calls["masked_block"] > 0
    assert np.isfinite(result).all()


# ---------------------------------------------------------------------------
# 6b. Stage 3 — masked categorical and L1 block kernels
# ---------------------------------------------------------------------------


def test_masked_wrappers_match_python_randomized():
    """Direct PyO3 masked wrappers agree with the NumPy masked oracle."""
    rng = np.random.default_rng(20260809)
    for n in (2, 8, 17):
        for n_candidates in (1, 7, 32):
            weights = rng.random(n)
            weights /= weights.sum()
            base = np.arange(n_candidates, dtype=np.int64) * n
            lags = np.arange(n, dtype=np.int64)
            mask = rng.random((n_candidates, n)) < 0.25
            mask[0] = True

            de_cat = (rng.integers(0, 3, size=n) + 1) / 10.0
            ti_cat = (rng.integers(0, 3, size=(n_candidates, n)) + 1) / 10.0
            ti_cat[mask] = np.nan
            expected_cat = vec_categorical_dist(
                de_cat, ti_cat, weights, has_nan=True
            )
            actual_cat = gstools_core.mps_dist_block_cat_masked(
                de_cat, ti_cat.ravel(), base, lags, weights
            )
            np.testing.assert_allclose(
                actual_cat, expected_cat, rtol=1e-12, atol=1e-13
            )

            de_l1 = rng.normal(size=n)
            ti_l1 = rng.normal(size=(n_candidates, n))
            ti_l1[mask] = np.nan
            d_max = 4.0
            expected_l1 = vec_l1_dist(
                de_l1, ti_l1, weights, d_max, has_nan=True
            )
            actual_l1 = gstools_core.mps_dist_block_l1_masked(
                de_l1, ti_l1.ravel(), base, lags, weights, d_max
            )
            np.testing.assert_allclose(
                actual_l1, expected_l1, rtol=1e-12, atol=1e-13
            )


@pytest.mark.parametrize("threshold", [0.0, 0.1])
def test_masked_categorical_output_identical(threshold):
    """Masked arbitrary numeric labels preserve DSBC and DS output exactly."""
    data = np.where(_cat_ti((36, 36)) == 0, 0.1, 0.2)
    data[4:10, 7:13] = np.nan
    with pytest.warns(UserWarning, match="contains NaN"):
        python = _run(data, True, "l1", (13, 13), 0.4, threshold, 42, False)
    with pytest.warns(UserWarning, match="contains NaN"):
        rust = _run(data, True, "l1", (13, 13), 0.4, threshold, 42, True)
    np.testing.assert_array_equal(python, rust)


@pytest.mark.parametrize("threshold", [0.0, 0.1])
def test_masked_l1_output_identical(threshold):
    """Masked L1 preserves complete DSBC and DS fields."""
    data = _cont_ti((36, 36))
    data[4:10, 7:13] = np.nan
    with pytest.warns(UserWarning, match="contains NaN"):
        python = _run(data, False, "l1", (13, 13), 0.4, threshold, 42, False)
    with pytest.warns(UserWarning, match="contains NaN"):
        rust = _run(data, False, "l1", (13, 13), 0.4, threshold, 42, True)
    np.testing.assert_array_equal(python, rust)


def test_masked_l1_dispatch_calls_only_masked_kernel(monkeypatch):
    """Prove a masked L1 variable reaches the new Rust export."""
    from gstools.mps import scan as _scan

    calls = {"plain": 0, "masked": 0}
    original_plain = _scan._mps_dist_block_l1_gsc
    original_masked = _scan._mps_dist_block_l1_masked_gsc

    def _plain_spy(*args, **kwargs):
        calls["plain"] += 1
        return original_plain(*args, **kwargs)

    def _masked_spy(*args, **kwargs):
        calls["masked"] += 1
        return original_masked(*args, **kwargs)

    monkeypatch.setattr(_scan, "_mps_dist_block_l1_gsc", _plain_spy)
    monkeypatch.setattr(_scan, "_mps_dist_block_l1_masked_gsc", _masked_spy)
    data = _cont_ti((28, 28))
    data[8:12, 9:14] = np.nan
    with pytest.warns(UserWarning, match="contains NaN"):
        result = _run(data, False, "l1", (10, 10), 0.4, 0.0, 42, True)
    assert calls["plain"] == 0
    assert calls["masked"] > 0
    assert np.isfinite(result).all()


def test_masked_kernels_not_called_when_backend_off(monkeypatch):
    """Explicit opt-out retains the Python masked implementation."""
    from gstools.mps import scan as _scan

    def _unexpected(*args, **kwargs):
        raise AssertionError("masked Rust kernel called with backend disabled")

    monkeypatch.setattr(_scan, "_mps_dist_block_cat_masked_gsc", _unexpected)
    monkeypatch.setattr(_scan, "_mps_dist_block_l1_masked_gsc", _unexpected)
    cat = _cat_ti((24, 24)).astype(float)
    cont = _cont_ti((24, 24))
    cat[5:9, 7:11] = np.nan
    cont[11:15, 13:17] = np.nan
    with pytest.warns(UserWarning, match="contains NaN"):
        cat_result = _run(cat, True, "l1", (9, 9), 0.4, 0.0, 42, False)
    with pytest.warns(UserWarning, match="contains NaN"):
        l1_result = _run(cont, False, "l1", (9, 9), 0.4, 0.0, 42, False)
    assert np.isfinite(cat_result).all()
    assert np.isfinite(l1_result).all()


def test_multivariate_different_nan_masks_use_both_kernels(monkeypatch):
    """Different per-variable masks preserve one shared multivariate anchor."""
    from gstools.mps import scan as _scan

    calls = {"categorical": 0, "l1": 0}
    original_cat = _scan._mps_dist_block_cat_masked_gsc
    original_l1 = _scan._mps_dist_block_l1_masked_gsc

    def _cat_spy(*args, **kwargs):
        calls["categorical"] += 1
        return original_cat(*args, **kwargs)

    def _l1_spy(*args, **kwargs):
        calls["l1"] += 1
        return original_l1(*args, **kwargs)

    monkeypatch.setattr(_scan, "_mps_dist_block_cat_masked_gsc", _cat_spy)
    monkeypatch.setattr(_scan, "_mps_dist_block_l1_masked_gsc", _l1_spy)

    categorical = _cat_ti((30, 30)).astype(float)
    continuous = categorical.copy()
    categorical[4:9, 6:11] = np.nan
    continuous[17:22, 18:23] = np.nan
    with pytest.warns(UserWarning, match="contains NaN"):
        ti = TrainingImage(
            [
                Variable(
                    "categorical",
                    categorical,
                    categorical=True,
                    n_neighbors=8,
                    weight=0.5,
                ),
                Variable(
                    "l1",
                    continuous,
                    categorical=False,
                    distance="l1",
                    n_neighbors=8,
                    weight=0.5,
                ),
            ]
        )
    model = MPSModel(ti, scan_fraction=0.4, threshold=0.0)
    pos = [np.arange(11.0), np.arange(11.0)]

    gs.config.USE_GSTOOLS_CORE = False
    python = DirectSampling(model, seed=42)(pos, store=False)
    gs.config.USE_GSTOOLS_CORE = True
    rust = DirectSampling(model, seed=42)(pos, store=False)

    assert calls["categorical"] > 0
    assert calls["l1"] > 0
    for variable in ("categorical", "l1"):
        assert np.isfinite(rust[variable]).all()
        np.testing.assert_array_equal(python[variable], rust[variable])
    np.testing.assert_array_equal(rust["categorical"], rust["l1"])


@pytest.mark.parametrize(
    ("categorical", "backend_name"),
    [
        (True, "_mps_dist_block_cat_masked_gsc"),
        (False, "_mps_dist_block_l1_masked_gsc"),
    ],
)
def test_missing_masked_export_falls_back(
    categorical, backend_name, monkeypatch
):
    """Older cores without Stage 3 exports retain the Python path."""
    from gstools.mps import scan as _scan

    data = (
        _cat_ti((28, 28)).astype(float) if categorical else _cont_ti((28, 28))
    )
    data[8:12, 9:14] = np.nan
    with pytest.warns(UserWarning, match="contains NaN"):
        expected = _run(data, categorical, "l1", (10, 10), 0.4, 0.0, 42, False)
    monkeypatch.setattr(_scan, backend_name, None)
    with pytest.warns(UserWarning, match="contains NaN"):
        actual = _run(data, categorical, "l1", (10, 10), 0.4, 0.0, 42, True)
    np.testing.assert_array_equal(actual, expected)


# ---------------------------------------------------------------------------
# 7. Conditioned categorical — 3 conditioning points, SG 20x20
# ---------------------------------------------------------------------------


def test_conditioned_categorical():
    """Conditioning data is preserved bit-identically regardless of backend."""
    ti_data = _cat_ti()
    ti = TrainingImage(ti_data, categorical=True)
    model = MPSModel(ti, scan_fraction=0.3, threshold=0.0)
    sg_shape = (20, 20)
    rng = np.random.RandomState(7)
    xi = rng.randint(0, sg_shape[0], 3).astype(float)
    yi = rng.randint(0, sg_shape[1], 3).astype(float)
    vals = ti_data[xi.astype(int) % 60, yi.astype(int) % 60].astype(float)

    results = {}
    for use_rust in (False, True):
        gs.config.USE_GSTOOLS_CORE = use_rust
        ds = DirectSampling(model, seed=42)
        ds.set_condition([xi, yi], vals)
        results[use_rust] = ds(
            [np.arange(s, dtype=float) for s in sg_shape], store=False
        )

    np.testing.assert_array_equal(results[False], results[True])
    # Verify conditioning positions hold their values in the Rust output
    for x, y, v in zip(xi.astype(int), yi.astype(int), vals):
        assert results[True][x, y] == v, (
            f"Conditioning value at ({x}, {y}) not preserved: "
            f"got {results[True][x, y]}, expected {v}"
        )


# ---------------------------------------------------------------------------
# 8. Rust disabled — simulation still completes with expected shape
# ---------------------------------------------------------------------------


def test_rust_disabled_uses_python():
    """USE_GSTOOLS_CORE=False still produces a correctly shaped output."""
    ti = _cat_ti()
    gs.config.USE_GSTOOLS_CORE = False
    field = _run(ti, True, "l1", (10, 10), 0.3, 0.0, 42, False)
    assert field.shape == (10, 10)
    # All values must come from the TI (0 or 1 for this binary TI)
    assert set(np.unique(field)).issubset({0, 1})


# ---------------------------------------------------------------------------
# 9. Multivariate categorical — two variables, SG 10x10
# ---------------------------------------------------------------------------


def test_multivariate_categorical():
    """Multivariate categorical TI: Rust and Python outputs equal for both variables."""
    shape = (30, 30)
    gx, gy = np.meshgrid(
        np.arange(shape[0]), np.arange(shape[1]), indexing="ij"
    )
    facies = ((np.sin(gx / 5.0) + np.sin((gx + gy) / 8.0)) > 0).astype(
        np.int32
    )
    # Second variable: checkerboard pattern
    litho = ((gx + gy) % 3).astype(np.int32)

    ti = TrainingImage(
        [
            Variable(
                "facies", facies.astype(float), categorical=True, n_neighbors=4
            ),
            Variable(
                "litho", litho.astype(float), categorical=True, n_neighbors=4
            ),
        ]
    )
    model = MPSModel(ti, scan_fraction=0.3, threshold=0.0)
    pos = [np.arange(10, dtype=float)] * 2

    gs.config.USE_GSTOOLS_CORE = False
    ds_py = DirectSampling(model, seed=42)
    py_out = ds_py(pos, store=False)

    gs.config.USE_GSTOOLS_CORE = True
    ds_rs = DirectSampling(model, seed=42)
    rs_out = ds_rs(pos, store=False)

    for var_name in ("facies", "litho"):
        np.testing.assert_array_equal(
            py_out[var_name],
            rs_out[var_name],
            err_msg=f"Mismatch in variable {var_name!r}",
        )


# ---------------------------------------------------------------------------
# 10. Large categorical regression — 120x120 TI, 40x40 SG
# ---------------------------------------------------------------------------


def test_large_categorical_regression():
    """Large-case regression guard: Rust and Python outputs are bit-identical."""
    ti = _cat_ti((120, 120))
    py = _run(ti, True, "l1", (40, 40), 0.3, 0.0, 42, False)
    rs = _run(ti, True, "l1", (40, 40), 0.3, 0.0, 42, True)
    np.testing.assert_array_equal(py, rs)


# ---------------------------------------------------------------------------
# Additional parametrized medium-size tests (from brief)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [42, 1234, 99999])
def test_categorical_dsbc_medium(seed):
    """DSBC 30x30 SG, 60x60 TI — bit-identical across seeds."""
    ti = _cat_ti()
    py = _run(ti, True, "l1", (30, 30), 0.3, 0.0, seed, False)
    rs = _run(ti, True, "l1", (30, 30), 0.3, 0.0, seed, True)
    np.testing.assert_array_equal(py, rs)


@pytest.mark.parametrize("seed", [42, 1234])
def test_categorical_ds_mode(seed):
    """DS-mode 30x30 SG — bit-identical with threshold > 0."""
    ti = _cat_ti()
    py = _run(ti, True, "l1", (30, 30), 0.3, 0.1, seed, False)
    rs = _run(ti, True, "l1", (30, 30), 0.3, 0.1, seed, True)
    np.testing.assert_array_equal(py, rs)


@pytest.mark.parametrize("seed", [42, 1234])
def test_categorical_hiscan(seed):
    """High scan_fraction (0.8) — bit-identical."""
    ti = _cat_ti()
    py = _run(ti, True, "l1", (30, 30), 0.8, 0.0, seed, False)
    rs = _run(ti, True, "l1", (30, 30), 0.8, 0.0, seed, True)
    np.testing.assert_array_equal(py, rs)


@pytest.mark.parametrize("seed", [42, 1234])
def test_continuous_l1(seed):
    """L1 atol=1e-12 — 30x30 SG, 60x60 TI."""
    ti = _cont_ti()
    py = _run(ti, False, "l1", (30, 30), 0.3, 0.0, seed, False)
    rs = _run(ti, False, "l1", (30, 30), 0.3, 0.0, seed, True)
    np.testing.assert_allclose(py, rs, rtol=0, atol=1e-12)


@pytest.mark.parametrize("seed", [42, 1234])
def test_continuous_l2(seed):
    """L2 atol=1e-12 — 30x30 SG, 60x60 TI."""
    ti = _cont_ti()
    py = _run(ti, False, "l2", (30, 30), 0.3, 0.0, seed, False)
    rs = _run(ti, False, "l2", (30, 30), 0.3, 0.0, seed, True)
    np.testing.assert_allclose(py, rs, rtol=0, atol=1e-12)


@pytest.mark.parametrize("seed", [42])
def test_continuous_l3(seed):
    """Lp kernel with p=3 atol=1e-12 — 20x20 SG."""
    ti = _cont_ti()
    py = _run(ti, False, "l3", (20, 20), 0.3, 0.0, seed, False)
    rs = _run(ti, False, "l3", (20, 20), 0.3, 0.0, seed, True)
    np.testing.assert_allclose(py, rs, rtol=0, atol=1e-12)


def test_variation_distance_small_output_close():
    """Variation distance on a 20x20 SG agrees across both backends."""
    ti = _cont_ti()
    py = _run(ti, False, "variation", (20, 20), 0.3, 0.0, 42, False)
    rs = _run(ti, False, "variation", (20, 20), 0.3, 0.0, 42, True)
    np.testing.assert_allclose(py, rs, rtol=0, atol=1e-12)


def test_conditioned_categorical_15pts():
    """15-point conditioning — outputs bit-identical; conditioning values preserved."""
    ti_data = _cat_ti()
    ti = TrainingImage(ti_data, categorical=True)
    model = MPSModel(ti, scan_fraction=0.3, threshold=0.0)
    sg_shape = (20, 20)
    rng = np.random.RandomState(7)
    xi = rng.randint(0, sg_shape[0], 15).astype(float)
    yi = rng.randint(0, sg_shape[1], 15).astype(float)
    vals = ti_data[xi.astype(int) % 60, yi.astype(int) % 60].astype(float)

    results = {}
    for use_rust in (False, True):
        gs.config.USE_GSTOOLS_CORE = use_rust
        ds = DirectSampling(model, seed=42)
        ds.set_condition([xi, yi], vals)
        results[use_rust] = ds(
            [np.arange(s, dtype=float) for s in sg_shape], store=False
        )
    np.testing.assert_array_equal(results[False], results[True])
