"""Node-weight computation for MPS pattern comparison.

The vectorized pattern distances themselves (categorical / L1 / L2 / Lp /
variation) are computed by the Rust backend (``gstools_core.mps_dist_block_*``);
this module keeps only the shared spatial-decay weighting used to build a data
event. See ``RUST_ONLY_MIGRATION.md`` for the Rust-only backend change.
"""

import numpy as np

__all__ = ["compute_node_weights"]


def compute_node_weights(
    n, lag_norms, distance_power, cond_mask=None, cond_weight=1.0
):
    """Compute normalized spatial-decay weights for a data event.

    Combines spatial decay (Mariethoz2010 Eq. 5) with conditioning data
    multipliers (Mariethoz2010 §3 ¶26).

    Parameters
    ----------
    n : int
        Number of neighbours in the data event.
    lag_norms : array-like or None, shape (n,)
        Euclidean norms ``‖h_i‖`` of each lag vector. ``None`` or
        ``distance_power == 0`` → uniform spatial weights. A zero lag-norm
        (collocated ``h=0`` entry) keeps the unit baseline weight and is not
        amplified by the spatial decay — its weight is scaled by ``cond_weight``.
    distance_power : float
        Exponent δ. ``0.0`` → uniform.
    cond_mask : array-like of bool, optional
        ``True`` where the neighbour is a conditioning datum.
    cond_weight : float, optional
        Bonus weight multiplier for conditioning nodes.

    Returns
    -------
    numpy.ndarray, shape (n,)
        Node weights normalized to sum to 1.
    """
    raw_w = np.ones(n, dtype=np.float64)
    if lag_norms is not None and distance_power != 0.0:
        norms = np.asarray(lag_norms, dtype=np.float64)
        # Only non-zero lags decay with distance.  A *true* zero lag-norm is a
        # collocated/conditioning entry (h=0, e.g. the multivariate same-node
        # constraint); it keeps the unit-cell baseline weight 1.0 rather than the
        # divergent norm**(-power), and its importance is governed by the
        # cond_weight multiplier below (it always carries cond_mask=True).
        nz = norms != 0.0
        raw_w[nz] = norms[nz] ** (-distance_power)

    if cond_mask is not None:
        # ``raw_w`` is a freshly allocated array (np.ones above, modified in
        # place), so no defensive copy is needed before scaling.
        raw_w[np.asarray(cond_mask, dtype=bool)] *= cond_weight

    total = raw_w.sum()
    if total == 0.0:
        # Every neighbour was zeroed out — the canonical case is an all-
        # conditioning data event with cond_weight == 0 (δ_c = 0 → conditioning
        # ignored entirely, Me13 p.323 → unconditional behaviour). Returning
        # zero weights makes the data event non-informative so the node is drawn
        # unconditionally, rather than re-weighting the ignored nodes uniformly.
        return np.zeros(n, dtype=np.float64)
    if not np.isfinite(total):
        # Defensive: non-finite weight sum should be unreachable for grid lags
        # (‖h‖ >= 1) — fall back to uniform rather than emit NaNs.
        return np.full(n, 1.0 / n, dtype=np.float64)
    return raw_w / total
