"""Direct Sampling simulation engine.

`ds_simulate` is the entry point; `_DirectSamplingEngine` holds one run's
state (output grids, conditioning masks, config) explicitly. The complete node
path executes in the Rust engine.
"""

import warnings

import numpy as np

from gstools import config

# MPS is Rust-only: the complete engine comes from gstools.mps._core, which
# hard-imports and version-checks gstools_core (see RUST_ONLY_MIGRATION.md).
from gstools.mps._core import mps_simulate as _mps_simulate_gsc
from gstools.mps.neighbors import _lag_transform_matrix, _precompute_offsets

# Python precomputes optional per-node lag transform matrices before the
# GIL-free Rust call.
_MPS_RUST_ENGINE_STATS_HOOK = None


def _build_path(unknown, path, rng_path, sim_shape):
    """Build the (N, dim) node visit order.

    Parameters
    ----------
    unknown : numpy.ndarray of bool, shape sim_shape
        Mask of nodes that require simulation (at least one uninformed variable).
    path : str or array-like
        ``"random"`` — uniformly shuffled raster order (default DS behaviour).
        ``"sequential"`` — lexicographic (raster) order; ``rng_path`` is not
        consumed, making the visit order deterministic without a ``path_seed``.
        An explicit ``(N, dim)`` integer array — the caller-supplied visit
        order.  Must include every unknown node; conditioned nodes present in
        the array are silently skipped so that a full-grid path (e.g. a spiral
        over all nodes) works unchanged with conditioning data.  Duplicate rows
        and missing unknown nodes are still errors.
    rng_path : numpy.random.RandomState
        Controls shuffling when ``path="random"``; unused for the other modes.
    sim_shape : tuple of int
        Simulation grid shape.

    Returns
    -------
    numpy.ndarray, shape (N, dim), dtype numpy.intp
        Node visit order.

    Raises
    ------
    ValueError
        For an unrecognised string or an invalid explicit array.
    """
    base = np.argwhere(
        unknown
    )  # raster (lexicographic) order over unknown nodes
    if isinstance(path, str):
        if path == "sequential":
            return base
        if path == "random":
            return base[rng_path.permutation(len(base))]
        raise ValueError(
            f"DirectSampling: path must be 'random', 'sequential', or an "
            f"(N, dim) array; got {path!r}"
        )
    arr = np.asarray(path)
    dim = len(sim_shape)
    # Shape check
    if arr.ndim != 2 or arr.shape[1] != dim:
        raise ValueError(
            f"DirectSampling: explicit path shape must be (N, dim={dim}); "
            f"got {arr.shape!r}"
        )
    # Integer check and bounds check
    if not np.issubdtype(arr.dtype, np.integer):
        # Allow float arrays with integer values (e.g. from argwhere stored as float)
        if not np.all(arr == np.floor(arr)):
            raise ValueError(
                "DirectSampling: explicit path must contain integer coordinates"
            )
        arr = arr.astype(np.intp)
    else:
        arr = arr.astype(np.intp, copy=False)
    for d in range(dim):
        if np.any(arr[:, d] < 0) or np.any(arr[:, d] >= sim_shape[d]):
            raise ValueError(
                f"DirectSampling: explicit path has out-of-bounds coordinates "
                f"along axis {d} for grid shape {sim_shape!r}"
            )
    # Validation: duplicates → error; conditioned extras → silently drop;
    # missing unknown nodes → error.
    flat_path = np.ravel_multi_index(arr.T, sim_shape)
    flat_base = (
        np.ravel_multi_index(base.T, sim_shape)
        if len(base)
        else np.empty(0, dtype=np.intp)
    )
    if len(np.unique(flat_path)) != len(flat_path):
        raise ValueError(
            "DirectSampling: explicit path contains duplicate nodes"
        )
    # Keep only unknown-set nodes (conditioned entries are silently dropped)
    unknown_mask = np.isin(flat_path, flat_base)
    arr = arr[unknown_mask]
    flat_path = flat_path[unknown_mask]
    # Every unknown node must appear in the (filtered) path
    if not np.array_equal(np.sort(flat_path), np.sort(flat_base)):
        raise ValueError(
            "DirectSampling: explicit path is missing one or more unknown nodes "
            "(every unset grid node must appear in the path)"
        )
    return arr


class _DirectSamplingEngine:
    """Holds one Direct Sampling run's state and orchestrates it."""

    def __init__(
        self,
        training_image,
        sim_shape,
        threshold,
        scan_fraction,
        rng_path,
        rng_nodes,
        conditions=None,
        cond_weight=1.0,
        boundary="strict",
        rotation_map=None,
        anis_map=None,
        path="random",
    ):
        self.training_image = training_image
        self.variables = [v.name for v in training_image.variables]
        self.weights = (
            {
                v.name: training_image.weights[v.name]
                for v in training_image.variables
            }
            if training_image.multivariate
            else {None: 1.0}
        )
        self.ti_shape = np.array(training_image.shape)
        self.sim_shape = sim_shape
        self.dim = len(sim_shape)
        self.threshold = threshold
        self.scan_fraction = scan_fraction
        self.cond_weight = cond_weight
        self.boundary = boundary
        self.rotation_map = rotation_map
        self.anis_map = anis_map

        self.n_k = {v.name: v.n_neighbors for v in training_image.variables}

        self.ti_matrix_f64 = np.ascontiguousarray(
            np.stack(
                [
                    np.asarray(v.data, dtype=np.float64).ravel()
                    for v in training_image.variables
                ],
                axis=0,
            )
        )

        self.sg = {v: np.full(sim_shape, np.nan) for v in self.variables}
        self.is_cond = {
            v: np.zeros(sim_shape, dtype=bool) for v in self.variables
        }
        if conditions:
            for idx, vd in conditions.items():
                for v, val in vd.items():
                    self.sg[v][idx] = val
                    self.is_cond[v][idx] = True

        self.max_radius_per_var = {
            v.name: v.max_radius for v in training_image.variables
        }
        _radii = [r for r in self.max_radius_per_var.values() if r is not None]
        global_max_radius = max(_radii) if _radii else None
        max_off_int = (
            int(np.ceil(global_max_radius))
            if global_max_radius is not None
            else None
        )
        self.offset_arr = _precompute_offsets(sim_shape, max_off_int)

        # Node path: every node with >= 1 uninformed variable.  Fully-conditioned
        # nodes need no simulation and are excluded (they stay -1 / always available).
        unknown = np.zeros(sim_shape, dtype=bool)
        for v in self.variables:
            unknown |= np.isnan(self.sg[v])
        self.path = _build_path(unknown, path, rng_path, sim_shape)
        n_nodes = len(self.path)
        self.u_start = rng_nodes.uniform(size=n_nodes)
        self.u_fallback = rng_nodes.uniform(size=(n_nodes, self.dim))

        # A joint fallback draw must have all variables defined.
        if training_image.has_nan:
            finite_all = np.ones(self.ti_shape, dtype=bool)
            for variable in training_image.variables:
                data = variable.data
                if np.issubdtype(data.dtype, np.floating):
                    finite_all &= ~np.isnan(data)
            if not finite_all.any():
                raise ValueError(
                    "TrainingImage has no cell defined in all variables (every "
                    "cell is NaN in at least one variable); cannot simulate."
                )

        self.var_categorical = {
            v.name: v.categorical for v in training_image.variables
        }
        self.var_has_nan = {
            v.name: v.has_nan for v in training_image.variables
        }
        self.var_p_norm = {v.name: v.p_norm for v in training_image.variables}
        self.var_d_max = {v.name: v.d_max for v in training_image.variables}
        self.var_variation_p = {
            v.name: v.variation_p_norm for v in training_image.variables
        }

    def _run_rust_engine(self, n_threads):
        """Run the complete node path inside one GIL-free Rust call."""
        fields = np.ascontiguousarray(
            np.stack([self.sg[v].reshape(-1) for v in self.variables])
        )
        conditioned = np.ascontiguousarray(
            np.stack(
                [self.is_cond[v].reshape(-1) for v in self.variables]
            ).astype(np.uint8)
        )
        metric_kinds = np.fromiter(
            (
                0
                if self.var_categorical[v]
                else 1
                if self.var_p_norm[v] is not None
                else 2
                for v in self.variables
            ),
            dtype=np.int64,
            count=len(self.variables),
        )
        p_norm = np.fromiter(
            (
                1.0
                if self.var_categorical[v]
                else self.var_p_norm[v]
                if self.var_p_norm[v] is not None
                else self.var_variation_p[v]
                for v in self.variables
            ),
            dtype=np.float64,
            count=len(self.variables),
        )
        lag_matrices = (
            np.ascontiguousarray(
                np.stack(
                    [
                        _lag_transform_matrix(
                            self.dim,
                            self.rotation_map,
                            self.anis_map,
                            node,
                        )
                        for node in self.path
                    ]
                )
            )
            if (self.rotation_map is not None or self.anis_map is not None)
            and len(self.path)
            else np.empty((0, self.dim, self.dim), dtype=np.float64)
        )
        (
            result,
            strict_fallbacks,
            level_count,
            max_ready_width,
            used_threads,
            collapsed_lags,
        ) = _mps_simulate_gsc(
            self.ti_matrix_f64,
            np.asarray(self.ti_shape, dtype=np.int64),
            fields,
            conditioned,
            np.asarray(self.sim_shape, dtype=np.int64),
            np.asarray(self.path, dtype=np.int64),
            lag_matrices,
            np.asarray(self.u_start, dtype=np.float64),
            np.asarray(self.u_fallback, dtype=np.float64),
            np.asarray(self.offset_arr, dtype=np.int64),
            np.fromiter(
                (self.n_k[v] for v in self.variables),
                dtype=np.int64,
                count=len(self.variables),
            ),
            np.fromiter(
                (
                    np.nan
                    if self.max_radius_per_var[v] is None
                    else self.max_radius_per_var[v]
                    for v in self.variables
                ),
                dtype=np.float64,
                count=len(self.variables),
            ),
            np.fromiter(
                (self.weights[v] for v in self.variables),
                dtype=np.float64,
                count=len(self.variables),
            ),
            metric_kinds,
            np.fromiter(
                (self.var_has_nan[v] for v in self.variables),
                dtype=np.uint8,
                count=len(self.variables),
            ),
            np.fromiter(
                (
                    1.0 if self.var_d_max[v] is None else self.var_d_max[v]
                    for v in self.variables
                ),
                dtype=np.float64,
                count=len(self.variables),
            ),
            p_norm,
            self.threshold,
            self.scan_fraction,
            self.training_image.distance_power,
            self.cond_weight,
            self.boundary == "partial",
            n_threads,
        )
        for row, variable in enumerate(self.variables):
            self.sg[variable][...] = result[row].reshape(self.sim_shape)
        self._rust_engine_stats = {
            "level_count": int(level_count),
            "max_ready_width": int(max_ready_width),
            "requested_threads": int(n_threads),
            "used_threads": int(used_threads),
        }
        if _MPS_RUST_ENGINE_STATS_HOOK is not None:
            _MPS_RUST_ENGINE_STATS_HOOK(self._rust_engine_stats.copy())
        if strict_fallbacks:
            warnings.warn(
                "gstools.mps: boundary='strict' could not fit the full data "
                "event inside the TI; falling back to partial mode (dropping "
                "the furthest neighbour(s)). Ensure the TI is at least as "
                "large as the data event extent to enforce strict mode.",
                RuntimeWarning,
                stacklevel=2,
            )
        if collapsed_lags:
            warnings.warn(
                "gstools.mps: anisotropy/rotation transform collapsed "
                "neighbour lag(s) onto duplicate TI positions; the "
                "duplicates are excluded from the data event. Reduce the "
                "anisotropy ratio or rotation to retain them.",
                RuntimeWarning,
                stacklevel=2,
            )
        return self.sg

    def run(self, num_threads=None):
        # The core is optional for other GSTools features but required for MPS.
        if not config.USE_GSTOOLS_CORE:
            raise RuntimeError(
                "GSTools MPS is Rust-only: gstools.config.USE_GSTOOLS_CORE must "
                "be True. It cannot be disabled for MPS (set it back to True)."
            )
        n_threads = (
            num_threads
            if num_threads is not None
            else (config.NUM_THREADS or 1)
        )
        return self._run_rust_engine(n_threads)


def ds_simulate(
    training_image,
    sim_shape,
    threshold,
    scan_fraction,
    rng_path,
    rng_nodes,
    conditions=None,
    cond_weight=1.0,
    boundary="strict",
    num_threads=None,
    rotation_map=None,
    anis_map=None,
    path="random",
):
    """Node-wise multivariate Direct Sampling (Mariethoz2010 §3, Eq. 8).

    Co-simulates every variable of a dict-valued :class:`TrainingImage` on one
    structured grid, treating all variables equally. The path visits every
    *node* with at least one unknown variable; at each node a single joint scan
    finds the TI cell ``y`` minimising the weighted distance ``Σ_k w_k d_k`` over
    all variables, and that one cell's whole vector ``{TI[v][y]}`` is copied to
    the node's *uninformed* variables. Because every variable at a node is drawn
    from the same TI cell, the joint (cross-variable) relationship is reproduced
    exactly. Variables already known at the node (only ever via conditioning
    data) act as collocated ``h = 0`` constraints, weighted by ``cond_weight``.

    Parameters
    ----------
    training_image : TrainingImage
        Training image; per-variable ``n_neighbors`` and ``max_radius`` are
        read from each :class:`Variable` object.
    sim_shape : tuple
        Simulation grid shape.
    threshold : float
        Distance threshold (Juda2022 §2). ``0.0`` -> DSBC mode.
    scan_fraction : float
        Fraction of the TI to scan per node, capped at the valid search window.
    rng_path : numpy.random.RandomState
        RandomState controlling the simulation path (node visit order).
        Only consumed when ``path="random"``; unused otherwise.
    rng_nodes : numpy.random.RandomState
        RandomState controlling TI scan entry points and fallback cells.
    path : str or array-like, optional
        Node visit order.  ``"random"`` (default) shuffles the unknown nodes
        uniformly using ``rng_path`` — the standard DS behaviour.
        ``"sequential"`` visits nodes in raster (lexicographic) order, which
        is deterministic and does not consume ``rng_path``.  An explicit
        integer array of shape ``(N, dim)`` provides a caller-supplied order
        and must include every unknown node (conditioned nodes are ignored;
        missing unknown nodes and duplicate rows raise ``ValueError``).
        Default: ``"random"``.
    conditions : dict, optional
        ``{node_index: {variable: value}}`` conditioning data.
    cond_weight : float, optional
        Weight delta for conditioning nodes (Mariethoz2010 §3 ¶26).
    boundary : str, optional
        ``"strict"`` (default) or ``"partial"`` search-window strategy.
    num_threads : int or None, optional
        Threads for the node-wise DAG. ``None`` -> ``config.NUM_THREADS``.
    rotation_map : numpy.ndarray or None, optional
        Per-node rotation angles, shape matching the simulation grid. ``None``
        → no rotation (stationary). Use ``DirectSampling.set_nonstationary``
        to produce this array from user-facing scalar or array inputs.
    anis_map : numpy.ndarray or None, optional
        Per-node anisotropy ratios, shape matching the simulation grid.
        ``None`` → isotropic (stationary). All values must be positive.
    Returns
    -------
    dict
        ``{variable: numpy.ndarray}`` — one simulated field per variable.
    """
    engine = _DirectSamplingEngine(
        training_image,
        sim_shape,
        threshold,
        scan_fraction,
        rng_path,
        rng_nodes,
        conditions=conditions,
        cond_weight=cond_weight,
        boundary=boundary,
        rotation_map=rotation_map,
        anis_map=anis_map,
        path=path,
    )
    return engine.run(num_threads=num_threads)
