"""F1 regression: event-node weight normalization under overflow/underflow.

``compute_node_weights`` must not collapse to a uniform event when a finite
``cond_weight`` overflows the raw sum, nor to an all-zero event when a large
``distance_power`` underflows the far nodes. Ordinary inputs keep the historical
direct normalization bit-for-bit. The native Rust helper carries the same
policy (see GSTools-Core mps_engine::compute_node_weights unit test).
"""

import numpy as np

from gstools.mps.distance import compute_node_weights


def test_f1_overflow_preserves_relative_weights():
    w = compute_node_weights(
        3,
        np.array([1.0, 1.0, 1.0]),
        distance_power=0.0,
        cond_mask=np.array([True, True, False]),
        cond_weight=1e308,
    )
    # Two hard nodes share the mass; the far node keeps a tiny positive weight.
    assert abs(w[0] - 0.5) < 1e-9
    assert abs(w[1] - 0.5) < 1e-9
    assert 0.0 < w[2] < 1e-300
    assert abs(w.sum() - 1.0) < 1e-12


def test_f1_underflow_keeps_informative_event():
    w = compute_node_weights(2, np.array([2.0, 3.0]), distance_power=2048.0)
    # Normalized weights tend to [1, 0] in float64, not an all-zero event.
    np.testing.assert_array_equal(w, [1.0, 0.0])


def test_f1_ordinary_path_is_bit_identical():
    w = compute_node_weights(
        3,
        np.array([1.0, 2.0, 3.0]),
        distance_power=1.0,
        cond_mask=np.array([False, True, False]),
        cond_weight=2.0,
    )
    raw = np.array([1.0, 2.0 * (2.0**-1.0), 3.0**-1.0])
    np.testing.assert_array_equal(w, raw / raw.sum())


def test_f1_candidate_selection_flips_to_correct_anchor():
    # Event [0, 0, 1] with two hard nodes (huge cond_weight) and one soft node.
    # Uniform weights would tie candidate B=[1,0,1] (distance 1/3) with
    # A=[0,0,0] (distance 1/3); correct weights make A win by a wide margin.
    w = compute_node_weights(
        3,
        np.array([1.0, 1.0, 1.0]),
        distance_power=0.0,
        cond_mask=np.array([True, True, False]),
        cond_weight=1e308,
    )
    event = np.array([0, 0, 1])
    dist_b = np.sum(w * (event != np.array([1, 0, 1])))
    dist_a = np.sum(w * (event != np.array([0, 0, 0])))
    assert dist_a < dist_b  # A is the correct anchor, not a uniform tie
