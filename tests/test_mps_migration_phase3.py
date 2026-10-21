"""Phase 3 regressions: MPSModel.rotation/scale drive the simulation.

Before the reconciliation, ``MPSModel(rotation=, scale=)`` was validated but
ignored (``DirectSampling.set_nonstationary`` drove a parallel ``anis`` path).
These lock in that the model specs now reach the engine via
:func:`gstools.mps.nonstationary.resolve_spec`, and that scalar / stationary
component-vector / callable forms of the *same* rotation agree end to end.
"""

import numpy as np
import pytest

import gstools as gs


def _ti():
    rng = np.random.default_rng(0)
    return gs.mps.TrainingImage(rng.integers(0, 3, (24, 24)), n_neighbors=8)


def _run(**model_kw):
    ds = gs.mps.DirectSampling(gs.mps.MPSModel(_ti(), scan_fraction=0.3, **model_kw))
    return ds([np.arange(10, dtype=float)] * 2, seed=4)


def test_scalar_vector_callable_rotation_agree():
    theta = np.pi / 5
    f_scalar = _run(rotation=theta)
    # 2-D: no_of_angles == 1, so the stationary component vector is [theta].
    f_vector = _run(rotation=[theta])
    f_callable = _run(rotation=lambda x, y: np.full_like(x, theta))
    np.testing.assert_array_equal(f_scalar, f_vector)
    np.testing.assert_array_equal(f_scalar, f_callable)


def test_rotation_actually_changes_output_vs_plain():
    # The core Phase-3 fix: a model rotation must reach the engine.
    f_plain = _run()
    f_rot = _run(rotation=np.pi / 4)
    assert not np.array_equal(f_plain, f_rot)


def test_callable_scale_changes_output():
    f_plain = _run()
    # anisotropic absolute scale via a callable (compress axis 1)
    f_scale = _run(scale=lambda x, y: np.stack([np.ones_like(x), 0.5 * np.ones_like(y)], axis=-1))
    assert not np.array_equal(f_plain, f_scale)


def test_identity_rotation_scale_is_bit_identical_to_plain():
    f_plain = _run()
    f_id = _run(rotation=0.0, scale=1.0)
    np.testing.assert_array_equal(f_plain, f_id)


def test_rotation_rejected_on_1d_training_image():
    ti = gs.mps.TrainingImage(np.zeros(12))
    with pytest.raises(ValueError, match="rotation"):
        gs.mps.MPSModel(ti, rotation=0.5)


def test_scale_rejected_nonpositive_at_construction():
    ti = gs.mps.TrainingImage(np.zeros((9, 9)))
    with pytest.raises(ValueError, match="scale"):
        gs.mps.MPSModel(ti, scale=[1.0, 0.0])


def test_3d_scalar_rotation_runs_and_matches_padded_vector():
    rng = np.random.default_rng(1)
    data = rng.integers(0, 2, (12, 12, 12))

    def run(rot):
        ds = gs.mps.DirectSampling(
            gs.mps.MPSModel(
                gs.mps.TrainingImage(data, n_neighbors=4),
                scan_fraction=0.1,
                rotation=rot,
            )
        )
        return ds([np.arange(5, dtype=float)] * 3, seed=0)

    # set_angles convention: a 3-D scalar pads with zeros -> [theta, 0, 0].
    f_scalar = run(0.5)
    f_padded = run([0.5, 0.0, 0.0])
    np.testing.assert_array_equal(f_scalar, f_padded)
