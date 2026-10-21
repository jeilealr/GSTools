"""Geometry precomputation for the Rust Direct Sampling engine.

Python prepares deterministic neighbour offsets and optional per-node lag
transform matrices. Neighbour selection and TI search windows run in Rust.
"""

import numpy as np

from gstools.tools.geometric import matrix_derotate, no_of_angles


def _precompute_offsets(shape, max_offset=None):
    """Neighbour offsets from the origin, sorted by Euclidean distance.

    Among equidistant offsets the order is a **canonical lexicographic**
    tie-break (by squared distance, then by each coordinate). This is
    platform- and NumPy-version-independent — unlike the default ``argsort``
    (unstable quicksort), whose tie order is implementation-defined — so that
    ``n_neighbors`` cutting mid-shell yields the same neighbour subset, and the
    simulation is reproducible across machines.

    Parameters
    ----------
    shape : tuple
        Simulation grid shape.
    max_offset : int, optional
        Maximum offset in any dimension.
        Default: ``max(shape)``.

    Returns
    -------
    numpy.ndarray, shape (N, dim)
    """
    dim = len(shape)
    if max_offset is None:
        max_offset = max(shape)
    estimated_elements = (2 * max_offset + 1) ** dim
    if estimated_elements > 5_000_000:
        raise ValueError(
            f"_precompute_offsets would allocate ~{estimated_elements * dim * 8 // 1_000_000} MB "
            f"(max_offset={max_offset}, dim={dim}). Set max_radius to bound the "
            "neighbourhood search radius."
        )
    rng_vals = np.arange(-max_offset, max_offset + 1)
    grid = np.array(np.meshgrid(*[rng_vals] * dim, indexing="ij"))
    offsets = grid.reshape(dim, -1).T
    offsets = offsets[np.any(offsets != 0, axis=1)]
    dist_sq = np.sum(offsets**2, axis=1)
    # lexsort: last key is primary. Primary = squared distance; ties broken by
    # coordinate 0, then 1, ... for a deterministic, portable ordering.
    keys = [offsets[:, d] for d in range(dim - 1, -1, -1)] + [dist_sq]
    idx = np.lexsort(keys)
    return offsets[idx]


def _lag_transform_matrix(dim, rotation_map, scale_map, x_i):
    """Detransform matrix M for a node's per-node rotation/scale.

    Consumes the canonical per-node maps produced by
    :func:`gstools.mps.nonstationary.resolve_spec` — always shaped
    ``grid_shape + (n_comp,)`` (stationary specs are zero-stride broadcast
    views), so indexing is uniform and no ndim heuristics are needed.

    Parameters
    ----------
    dim : int
        Spatial dimensionality of the simulation grid.
    rotation_map : numpy.ndarray or None
        Canonical rotation map, shape ``grid_shape + (no_of_angles(dim),)``.
        ``None`` → no rotation.
    scale_map : numpy.ndarray or None
        Canonical scale map, shape ``grid_shape + (dim,)``. ``None`` → no
        dilation (a uniform scalar value dilates every axis equally).
    x_i : numpy.ndarray, shape (dim,)
        Integer grid coordinates of the current simulation node.

    Returns
    -------
    M : numpy.ndarray, shape (dim, dim)
        ``diag(1 / s(x_i)) @ matrix_derotate(dim, θ(x_i))`` — the SG→TI inverse
        map (Mariethoz2010 §6.2: simulated structures are θ-rotated and
        s×-sized relative to the TI, so lags are pulled back by the inverse).
        Lags are stored as **row vectors** (shape ``(k, dim)``), so the frame
        map is applied as ``lags_ti = lags_sg @ M.T``.
    """
    idx = tuple(int(c) for c in x_i)
    angles = (
        rotation_map[idx]
        if rotation_map is not None
        else np.zeros(no_of_angles(dim))
    )
    scale = scale_map[idx] if scale_map is not None else np.ones(dim)
    inv_scale = 1.0 / np.asarray(scale, dtype=np.double)
    return inv_scale[:, None] * matrix_derotate(dim, angles)
