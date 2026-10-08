"""Geometry precomputation for the Rust Direct Sampling engine.

Python prepares deterministic neighbour offsets and optional per-node lag
transform matrices. Neighbour selection and TI search windows run in Rust.
"""

import numpy as np

from gstools.tools.geometric import matrix_isometrize, set_angles, set_anis


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


def _lag_transform_matrix(dim, rotation_map, anis_map, x_i):
    """Isometrization matrix M for a node's per-node rotation/anisotropy.

    Mirrors the inline build in the engine: a 1-D map is a stationary value
    broadcast to all nodes; a full-shape map is indexed at ``x_i``.

    Parameters
    ----------
    dim : int
        Spatial dimensionality of the simulation grid.
    rotation_map : numpy.ndarray or None
        Rotation angle(s).  ``None`` → no rotation (identity contribution).
        A 1-D array is a stationary multi-component angle vector applied to
        every node; a full-shape array is indexed at ``x_i`` for per-node
        angles.
    anis_map : numpy.ndarray or None
        Anisotropy ratio(s).  ``None`` → isotropic (ratio 1.0 in all
        transversal directions).  Same broadcast rules as ``rotation_map``.
    x_i : numpy.ndarray, shape (dim,)
        Integer grid coordinates of the current simulation node.  Used to
        index into per-node maps; ignored when the maps are stationary (1-D).

    Returns
    -------
    M : numpy.ndarray, shape (dim, dim)
        Isometrization matrix from :func:`gstools.tools.geometric.matrix_isometrize`.
        Lags are stored as **row vectors** (shape ``(k, dim)``), so the SG→TI
        frame map is applied as ``lags_ti = lags_sg @ M.T`` (right-multiply by
        the transpose) rather than as a left-multiply column-vector form.
    """
    angles_i = set_angles(
        dim,
        (rotation_map if rotation_map.ndim == 1 else rotation_map[tuple(x_i)])
        if rotation_map is not None
        else 0.0,
    )
    anis_i = set_anis(
        dim,
        (anis_map if anis_map.ndim == 1 else anis_map[tuple(x_i)])
        if anis_map is not None
        else 1.0,
    )
    return matrix_isometrize(dim, angles_i, anis_i)
