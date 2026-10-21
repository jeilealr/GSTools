"""Phase 4 regressions: native categorical penalty-matrix distance.

The Rust engine applies ``T[sim_code, ti_code]`` (zero diagonal, off-diagonal in
(0, 1], asymmetric allowed) in place of the 0/1 categorical mismatch. These
check the end-to-end simulation behaviour; the exact kernel arithmetic is pinned
by the Rust unit tests ``penalty_cost_*`` / ``categorical_penalty_*``.
"""

import numpy as np

import gstools as gs
from gstools.mps import Variable


def _ti_data():
    rng = np.random.default_rng(0)
    return rng.integers(0, 3, (24, 24)).astype(float)


def _run(penalty, seed=1):
    data = _ti_data()
    var = Variable("c", data, categorical=True, n_neighbors=12)
    if penalty is not None:
        var = var.replace(penalty_matrix=penalty)
    ti = gs.mps.TrainingImage(var)
    return gs.mps.DirectSampling(gs.mps.MPSModel(ti, scan_fraction=0.4))(
        [np.arange(10, dtype=float)] * 2, seed=seed
    )


def test_all_ones_penalty_matches_plain_categorical():
    # Off-diagonal all 1.0 reproduces the plain 0/1 mismatch -> identical field.
    ones = 1.0 - np.eye(3)
    np.testing.assert_array_equal(_run(None), _run(ones))


def test_penalty_changes_output_vs_plain():
    # Small off-diagonal costs make near-miss categories cheaper -> different.
    soft = np.array([[0.0, 0.2, 0.9], [0.2, 0.0, 0.2], [0.9, 0.2, 0.0]])
    assert not np.array_equal(_run(None), _run(soft))


def test_asymmetric_penalty_differs_from_transpose():
    # Orientation T[sim, ti] must matter: P and P.T give different fields.
    p = np.array([[0.0, 0.2, 0.9], [0.8, 0.0, 0.1], [0.3, 0.7, 0.0]])
    f_p = _run(p, seed=5)
    f_pt = _run(p.T.copy(), seed=5)
    assert not np.array_equal(f_p, f_pt)


def test_penalty_with_masked_ti_runs():
    # A NaN (undefined) cell in the TI still simulates with a penalty matrix.
    data = _ti_data()
    data[0, 0] = np.nan
    var = Variable("c", data, categorical=True, n_neighbors=12).replace(
        penalty_matrix=1.0 - np.eye(3)
    )
    ti = gs.mps.TrainingImage(var)
    field = gs.mps.DirectSampling(gs.mps.MPSModel(ti, scan_fraction=0.4))(
        [np.arange(8, dtype=float)] * 2, seed=2
    )
    assert np.all(np.isfinite(field))
    assert np.all(np.isin(field, [0.0, 1.0, 2.0]))


def test_multivariate_penalty_on_one_variable_runs():
    rng = np.random.default_rng(7)
    ti = gs.mps.TrainingImage(
        [
            Variable(
                "facies",
                rng.integers(0, 3, (20, 20)).astype(float),
                categorical=True,
                n_neighbors=8,
                penalty_matrix=np.array(
                    [[0.0, 0.3, 0.6], [0.3, 0.0, 0.3], [0.6, 0.3, 0.0]]
                ),
            ),
            Variable(
                "cont",
                rng.random((20, 20)),
                categorical=False,
                distance="l1",
                n_neighbors=8,
            ),
        ]
    )
    out = gs.mps.DirectSampling(gs.mps.MPSModel(ti, scan_fraction=0.3))(
        [np.arange(8, dtype=float)] * 2, seed=0
    )
    assert np.all(np.isin(out["facies"], [0.0, 1.0, 2.0]))
    assert np.all(np.isfinite(out["cont"]))
