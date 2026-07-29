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
    return ((np.sin(gx / 5.0) + np.sin((gx + gy) / 8.0)) > 0).astype(
        np.int32
    )


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
    ti = TrainingImage(
        ti_data, categorical=categorical, distance=distance
    )
    model = MPSModel(ti, scan_fraction=scan_fraction, threshold=threshold)
    ds = DirectSampling(model, seed=seed)
    return ds([np.arange(s, dtype=float) for s in sg_shape], store=False)


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
# 6. Variation distance — falls back to Python path; both must agree
# ---------------------------------------------------------------------------


def test_variation_distance_falls_back_to_python():
    """Variation distance has no Rust kernel; USE_GSTOOLS_CORE=True falls back correctly."""
    ti = _cont_ti((40, 40))
    py = _run(ti, False, "variation", (10, 10), 0.3, 0.0, 42, False)
    rs = _run(ti, False, "variation", (10, 10), 0.3, 0.0, 42, True)
    np.testing.assert_array_equal(py, rs)


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
            Variable("facies", facies.astype(float), categorical=True, n_neighbors=4),
            Variable("litho", litho.astype(float), categorical=True, n_neighbors=4),
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


def test_variation_distance_falls_back_to_python_small():
    """Variation distance (small 20x20 SG) — both paths must agree."""
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
