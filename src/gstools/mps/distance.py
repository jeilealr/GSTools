"""Node-weight computation for MPS pattern comparison.

The vectorized pattern distances themselves (categorical / L1 / L2 / Lp /
variation) are computed by the Rust backend (``gstools_core.mps_dist_block_*``);
this module keeps only the shared spatial-decay weighting used to build a data
event (MPS is Rust-only; the native kernels compute the pattern distances).
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

    cm = None if cond_mask is None else np.asarray(cond_mask, dtype=bool)
    if cm is not None:
        # ``raw_w`` is a freshly allocated array (np.ones above, modified in
        # place), so no defensive copy is needed before scaling.
        raw_w[cm] *= cond_weight

    with np.errstate(over="ignore", invalid="ignore"):
        total = raw_w.sum()
    if np.isfinite(total) and total > 0.0:
        # Direct normalization preserves the ordinary-case arithmetic order.
        # Use relative log weights only when the direct sum is degenerate.
        return raw_w / total

    # Degenerate direct sum: the raw weights overflowed to +inf
    # (e.g. a large finite ``cond_weight``) or every positive weight underflowed
    # to 0 (e.g. a large ``distance_power``). Both can arise from valid finite
    # inputs and must not collapse to a uniform or all-zero event, which would
    # discard the requested relative weighting / data-event information.
    #
    # A node is a *structural* zero only by intent: a conditioning node with
    # ``cond_weight == 0`` (δ_c = 0 → conditioning ignored, Me13 p.323). If every
    # node is structural (the canonical all-conditioning, zero-weight event),
    # keep the established all-zero, non-informative result.
    structural_zero = (
        np.zeros(n, dtype=bool)
        if cm is None or cond_weight != 0.0
        else cm
    )
    active = ~structural_zero
    if not active.any():
        return np.zeros(n, dtype=np.float64)

    # Rebuild the weights in the log domain so neither extreme overflows: a node
    # weight is ``r_i**(-power) * c_i`` with ``r_i = 1`` for the zero-lag
    # baseline. Subtract the min log-lag before scaling and the max log-weight
    # before exponentiating; underflowed entries become an exact 0 (the correct
    # float64 limit), and at least one active entry stays at unit scale.
    log_lag = np.zeros(n, dtype=np.float64)
    if lag_norms is not None and distance_power != 0.0:
        norms = np.asarray(lag_norms, dtype=np.float64)
        nz = norms != 0.0
        log_lag[nz] = np.log(norms[nz])
    log_w = -distance_power * (log_lag - log_lag[active].min())
    if cm is not None and cond_weight > 0.0:
        log_w[cm] += np.log(cond_weight)
    log_w[structural_zero] = -np.inf
    log_w -= log_w[active].max()
    weights = np.where(active, np.exp(log_w), 0.0)
    return weights / weights.sum()
