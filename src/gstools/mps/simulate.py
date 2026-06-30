"""Direct Sampling simulation engine.

`ds_simulate` is the entry point; `_DirectSamplingEngine` holds one run's
state (output grids, informed masks, config) explicitly — what used to be
captured by closures inside `ds_simulate`. The engine orchestrates per-node
simulation by calling the stateless `neighbors`, `scan`, and `runner` modules.
"""

from concurrent.futures import ThreadPoolExecutor

import numpy as np

from gstools import config
from gstools.mps.neighbors import (
    _lag_transform_matrix,
    _precompute_offsets,
    _reduce_to_fit,
    _select_neighbors,
    _transform_lags,
    _window_bounds,
)
from gstools.mps.runner import _make_progress, _run_path
from gstools.mps.scan import _scan_for_match


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
        self.sim_shape_arr = np.array(sim_shape)
        self.dim = len(sim_shape)
        self.threshold = threshold
        self.scan_fraction = scan_fraction
        self.cond_weight = cond_weight
        self.boundary = boundary
        self.rotation_map = rotation_map
        self.anis_map = anis_map

        self.n_k = {v.name: v.n_neighbors for v in training_image.variables}
        self.ti_vars = {v.name: v.data for v in training_image.variables}

        self.sg = {v: np.full(sim_shape, np.nan) for v in self.variables}
        self.informed = {
            v: np.zeros(sim_shape, dtype=bool) for v in self.variables
        }
        self.is_cond = {
            v: np.zeros(sim_shape, dtype=bool) for v in self.variables
        }
        if conditions:
            for idx, vd in conditions.items():
                for v, val in vd.items():
                    self.sg[v][idx] = val
                    self.is_cond[v][idx] = True
                    self.informed[v][idx] = True

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
        self.max_radius = global_max_radius

        # Node path: every node with >= 1 uninformed variable.  Fully-conditioned
        # nodes need no simulation and are excluded (they stay -1 / always available).
        unknown = np.zeros(sim_shape, dtype=bool)
        for v in self.variables:
            unknown |= np.isnan(self.sg[v])
        self.path = _build_path(unknown, path, rng_path, sim_shape)
        n_nodes = len(self.path)
        self.u_start = rng_nodes.uniform(size=n_nodes)
        self.u_fallback = rng_nodes.uniform(size=(n_nodes, self.dim))

        sg_size = int(np.prod(sim_shape))
        path_flat = (
            np.ravel_multi_index(self.path.T, sim_shape)
            if len(self.path)
            else np.empty(0, dtype=np.intp)
        )
        # Per-variable position maps over the node path.  A path node carries its
        # node-path index; a cell conditioned in variable v is set to -1 in vmap[v]
        # (always available for v, like univariate conditioning) — essential for a
        # partially-conditioned node, whose known value must not be gated by path
        # order.  Cells absent from the path are already -1.
        self.vmap = {}
        for v in self.variables:
            m = np.full(sg_size, -1, dtype=np.intp)
            m[path_flat] = np.arange(len(path_flat))
            m[np.flatnonzero(self.is_cond[v].reshape(-1))] = -1
            self.vmap[v] = m

        # Masked-TI support: NaN cells are undefined. When present, distances
        # exclude them per-position and fallback draws are restricted to cells that
        # are defined in *every* variable (so a joint draw never yields NaN). The
        # ``ti_has_nan`` gate keeps the fully-defined path byte-identical.
        self.ti_has_nan = training_image.has_nan
        if self.ti_has_nan:
            finite_all = np.ones(self.ti_shape, dtype=bool)
            for a in self.ti_vars.values():
                if np.issubdtype(a.dtype, np.floating):
                    finite_all &= ~np.isnan(a)
            self.finite_flat = np.flatnonzero(finite_all.reshape(-1))
            if self.finite_flat.size == 0:
                raise ValueError(
                    "TrainingImage has no cell defined in all variables (every "
                    "cell is NaN in at least one variable); cannot simulate."
                )
        else:
            self.finite_flat = None

    def _rand_fallback(self, targets, u_fb_i):
        # Single random TI cell supplies the whole node-vector (preserves the
        # joint relationship); never an independent draw per variable. On a
        # masked TI the draw is restricted to cells defined in every variable.
        if self.ti_has_nan:
            flat = self.finite_flat[int(u_fb_i[0] * len(self.finite_flat))]
            cell = tuple(
                int(c) for c in np.unravel_index(flat, tuple(self.ti_shape))
            )
        else:
            cell = tuple(
                int(u_fb_i[d] * s) for d, s in enumerate(self.ti_shape)
            )
        return {v: float(self.ti_vars[v][cell]) for v in targets}

    # ------------------------------------------------------------------
    # Simulation pipeline helpers (called only from _simulate_node)
    # ------------------------------------------------------------------

    def _gather_neighborhood(self, x_i, x_i_t, curr_idx):
        """Seam 1 — build per-variable data-event arrays for node ``x_i``.

        For each variable, selects the ``n_k[var]`` closest already-informed
        neighbours, converts them to lag vectors, and prepends the collocated
        ``h=0`` row when the variable is already known at this node (only
        possible via conditioning).

        Parameters
        ----------
        x_i : numpy.ndarray, shape (dim,)
            Integer grid coordinates of the current node.
        x_i_t : tuple
            ``tuple(int(c) for c in x_i)`` — precomputed for indexing.
        curr_idx : int
            Path index of the current node (neighbours must have index < this).

        Returns
        -------
        lags_v : dict[str, ndarray shape (k, dim)]
            Lag vectors in SG frame (float64).
        de_v : dict[str, ndarray shape (k,)]
            Data-event values from the simulation grid.
        cm_v : dict[str, ndarray shape (k,)]
            Boolean conditioning mask (True = conditioning data).
        ln_v : dict[str, ndarray shape (k,)]
            Euclidean norms of the lag vectors.
        """
        lags_v, de_v, cm_v, ln_v = {}, {}, {}, {}
        for var in self.variables:
            # Parallel path: the DAG pass already computed the identical
            # _select_neighbors result (same call, informed=None, all earlier-
            # path deps guaranteed informed by DAG ordering). Reuse those coords
            # instead of recomputing.  Serial path (cache absent): compute normally.
            if self._neighbor_cache is not None:
                coords = self._neighbor_cache[curr_idx][var]
            else:
                coords, _ = _select_neighbors(
                    x_i,
                    self.offset_arr,
                    self.sim_shape_arr,
                    self.sim_shape,
                    self.vmap[var],
                    curr_idx,
                    self.informed[var],
                    self.max_radius_per_var[var],
                    self.n_k[var],
                )
            if len(coords):
                lv = (coords - x_i).astype(np.float64)
                dv = self.sg[var][tuple(coords.T)]
                cv = self.is_cond[var][tuple(coords.T)]
                lnv = np.linalg.norm(lv, axis=1)
            else:
                lv = np.empty((0, self.dim), dtype=np.float64)
                dv = np.empty(0)
                cv = np.empty(0, dtype=bool)
                lnv = np.empty(0)
            # collocated h=0 — a variable known at this very node (only possible
            # via conditioning, as the node's unknown variables are what we fill).
            # Never added for a target variable (uninformed here by definition).
            if self.informed[var][x_i_t]:
                lv = np.concatenate([np.zeros((1, self.dim)), lv], axis=0)
                dv = np.concatenate([[self.sg[var][x_i_t]], dv])
                cv = np.concatenate([[self.is_cond[var][x_i_t]], cv])
                lnv = np.concatenate([[0.0], lnv])
            lags_v[var], de_v[var], cm_v[var], ln_v[var] = lv, dv, cv, lnv
        return lags_v, de_v, cm_v, ln_v

    def _transform_and_reduce_lags(self, x_i, lags_v, de_v, cm_v, ln_v):
        """Seam 2 — map SG lags into TI frame (non-stationary) or copy (stationary).

        When ``rotation_map`` or ``anis_map`` is set, applies the per-node
        isometrization matrix, deduplicates collapsed lags
        (:func:`_transform_lags`), and globally drops lags that cannot fit the
        TI (:func:`_reduce_to_fit`, Mariethoz2010 para [43]).

        The stationary fast-path ``lags_ti_v = dict(lags_v)`` is a deliberate
        **shallow copy**: it creates a separate dict so that per-variable
        partial-mode truncation in seam 3 (``_intersect_search_windows``) never
        aliases back into ``lags_v`` itself.  Both the copy and the in-place
        mutation of ``lags_v`` / ``de_v`` / ``cm_v`` / ``ln_v`` on the
        non-stationary path must be preserved exactly.

        Parameters
        ----------
        x_i : numpy.ndarray, shape (dim,)
            Current node coordinates (used to index per-node maps).
        lags_v, de_v, cm_v, ln_v : dict
            Outputs of :meth:`_gather_neighborhood`; ``lags_v`` may be mutated
            in-place on the non-stationary path (dedup / reduce-to-fit).

        Returns
        -------
        lags_ti_v : dict[str, ndarray]
            TI-frame lag vectors (separate dict from ``lags_v``).
        lags_v, de_v, cm_v, ln_v : dict
            Parallel arrays, possibly trimmed to match ``lags_ti_v``.
        """
        # Geometric transform: map all per-variable SG lags into TI frame.
        # de_v / cm_v / ln_v stay on original SG geometry for SG-side ops.
        # dict(lags_v) creates a shallow-copy dict so partial-mode truncation
        # of lags_ti_v does not alias back into lags_v.
        if self.rotation_map is not None or self.anis_map is not None:
            M = _lag_transform_matrix(
                self.dim, self.rotation_map, self.anis_map, x_i
            )
            lags_ti_v = {}
            for v in self.variables:
                lv_ti, lags_v[v], de_v[v], cm_v[v], ln_v[v] = _transform_lags(
                    lags_v[v], M, lags_v[v], de_v[v], cm_v[v], ln_v[v]
                )
                # M10 para [43]: globally reduce an over-large transformed event
                # to fit the TI, dropping the most-outside node first (by TI
                # extent, regardless of SG rank). One reduction per simulation
                # node, independent of the TI scan position.
                lv_ti, lags_v[v], de_v[v], cm_v[v], ln_v[v] = _reduce_to_fit(
                    lv_ti, self.ti_shape, lags_v[v], de_v[v], cm_v[v], ln_v[v]
                )
                lags_ti_v[v] = lv_ti
        else:
            lags_ti_v = dict(lags_v)
        return lags_ti_v, lags_v, de_v, cm_v, ln_v

    def _intersect_search_windows(self, lags_ti_v, lags_v, de_v, cm_v, ln_v):
        """Seam 3 — compute and intersect per-variable TI search windows.

        Calls :func:`_window_bounds` for every variable, applies partial-mode
        truncation of all five per-variable arrays when fewer lags fit, and
        intersects the per-variable windows into a single ``(win_lo, win_hi)``
        pair.

        Because two conditions make the current node unfeasible for a scan
        (an infeasible single-variable window, or an empty intersection), this
        method signals those cases via a sentinel return rather than calling
        ``_rand_fallback`` directly — the caller (``_simulate_node``) holds the
        ``targets`` and ``u_fallback_i`` needed for the fallback and checks
        ``win_lo is None``.

        Parameters
        ----------
        lags_ti_v : dict[str, ndarray]
            TI-frame lag vectors (output of :meth:`_transform_and_reduce_lags`).
        lags_v, de_v, cm_v, ln_v : dict
            Parallel arrays in SG frame; may be truncated in-place here.

        Returns
        -------
        win_lo : numpy.ndarray or None
            Lower corner of the intersected window, or ``None`` when infeasible.
        win_hi : numpy.ndarray or None
            Upper corner of the intersected window, or ``None`` when infeasible.
        lags_ti_v, lags_v, de_v, cm_v, ln_v : dict
            Arrays after any partial-mode truncation (only mutated when
            ``win_lo`` is not ``None``).
        """
        # Per-variable search window computed via _window_bounds, then intersected.
        # h=0 lags are excluded inside _window_bounds — they map to y itself and
        # never constrain the window.
        win_lo = np.zeros(self.dim, dtype=int)
        win_hi = self.ti_shape - 1
        for var in self.variables:
            lv_ti = lags_ti_v[var]
            lo, hi, valid_count = _window_bounds(
                lv_ti, self.ti_shape, self.boundary
            )
            if valid_count == -1:
                # Infeasible: no subset of this variable's lags fits the TI.
                return None, None, lags_ti_v, lags_v, de_v, cm_v, ln_v
            if valid_count < len(lv_ti):
                # Truncate TI-frame and original arrays to the same keep count.
                # lags_ti_v uses a separate dict so this does not affect lags_v.
                lags_ti_v[var] = lv_ti[:valid_count]
                lags_v[var] = lags_v[var][:valid_count]
                de_v[var] = de_v[var][:valid_count]
                cm_v[var] = cm_v[var][:valid_count]
                ln_v[var] = ln_v[var][:valid_count]
            win_lo = np.maximum(win_lo, lo)
            win_hi = np.minimum(win_hi, hi)

        if np.any(win_lo > win_hi):
            # Infeasible: the per-variable windows do not intersect.
            return None, None, lags_ti_v, lags_v, de_v, cm_v, ln_v

        return win_lo, win_hi, lags_ti_v, lags_v, de_v, cm_v, ln_v

    def _scan_and_retrieve(
        self,
        win_lo,
        win_hi,
        lags_ti_v,
        de_v,
        cm_v,
        ln_v,
        targets,
        u_start_i,
        u_fallback_i,
    ):
        """Seam 4 — scan the TI window and copy the matched cell to targets.

        Calls :func:`_scan_for_match` with the intersected window.  On a
        masked TI where every candidate has distance ``NaN``, the scan returns
        ``None`` and this method falls back to a random draw.  Otherwise,
        gathers the target values from the matched TI cell, applies
        ``adjust_value`` (mean-shift for variation distance), and returns the
        result dict.

        Parameters
        ----------
        win_lo, win_hi : numpy.ndarray, shape (dim,)
            Intersected window corners (output of
            :meth:`_intersect_search_windows`).
        lags_ti_v : dict[str, ndarray]
            TI-frame lag vectors after truncation.
        de_v, cm_v, ln_v : dict
            SG-frame parallel arrays after truncation.
        targets : list[str]
            Variables that are uninformed at this node and need values.
        u_start_i : float
            Random scan entry point.
        u_fallback_i : numpy.ndarray, shape (dim,)
            Random coordinates for the fallback draw.

        Returns
        -------
        dict[str, float]
            ``{variable: value}`` for every target variable.
        """
        # Integer lags per variable (already exact integers as float64, incl.
        # the 0.0 h=0 row); reused for the scan and the mean-shift gather.
        int_lags = {
            v: lags_ti_v[v].astype(int)
            for v in self.variables
            if len(lags_ti_v[v])
        }
        y = _scan_for_match(
            win_lo,
            tuple(win_hi - win_lo + 1),
            int_lags,
            de_v,
            cm_v,
            ln_v,
            u_start_i,
            targets,
            variables=self.variables,
            weights=self.weights,
            ti_vars=self.ti_vars,
            ti_shape=self.ti_shape,
            scan_fraction=self.scan_fraction,
            threshold=self.threshold,
            cond_weight=self.cond_weight,
            distance_power=self.training_image.distance_power,
            vec_distance_var=self.training_image.vec_distance_var,
            ti_has_nan=self.ti_has_nan,
        )
        if y is None:
            # No candidate in the window was defined (masked TI): treat as the
            # empty-neighbourhood case and draw a defined TI cell. Avoids the
            # ``y + lag`` gather below, which an unconstrained cell could send
            # out of bounds.
            return self._rand_fallback(targets, u_fallback_i)
        y_t = tuple(int(c) for c in y)
        # Copy the single matched cell's vector to every uninformed variable.
        result = {}
        for v in targets:
            ti_val = float(self.ti_vars[v][y_t])
            il = int_lags.get(v)
            de_ti_v = (
                self.ti_vars[v][tuple((y + il).T)]
                if il is not None
                else np.empty(0)
            )
            result[v] = self.training_image.adjust_value(
                ti_val, de_v[v], de_ti_v, var=v
            )
        return result

    def _simulate_node(self, curr_idx, x_i, u_start_i, u_fallback_i):
        """Simulate one node: a short pipeline of four named steps."""
        x_i_t = tuple(int(c) for c in x_i)
        targets = [v for v in self.variables if np.isnan(self.sg[v][x_i_t])]

        # Step 1: collect informed neighbours and build per-variable data events.
        lags_v, de_v, cm_v, ln_v = self._gather_neighborhood(
            x_i, x_i_t, curr_idx
        )

        # Step 2: map SG lags into TI frame; stationary path is a shallow copy
        # so that step 3 truncations never alias back into lags_v.
        lags_ti_v, lags_v, de_v, cm_v, ln_v = self._transform_and_reduce_lags(
            x_i, lags_v, de_v, cm_v, ln_v
        )

        if all(len(lags_v[v]) == 0 for v in self.variables):
            return self._rand_fallback(targets, u_fallback_i)

        # Step 3: compute and intersect per-variable TI search windows.
        # win_lo is None when any early-exit condition fires (infeasible window).
        win_lo, win_hi, lags_ti_v, lags_v, de_v, cm_v, ln_v = (
            self._intersect_search_windows(lags_ti_v, lags_v, de_v, cm_v, ln_v)
        )
        if win_lo is None:
            return self._rand_fallback(targets, u_fallback_i)

        # Step 4: scan the TI window and paste the matched cell into targets.
        return self._scan_and_retrieve(
            win_lo,
            win_hi,
            lags_ti_v,
            de_v,
            cm_v,
            ln_v,
            targets,
            u_start_i,
            u_fallback_i,
        )

    def _write_result(self, node, result):
        for v, val in result.items():
            if np.isnan(val):
                raise ValueError(
                    f"Simulation produced NaN for {(v, node)}. Check TI data."
                )
            self.sg[v][node] = val
            self.informed[v][node] = True

    def run(self, num_threads=None, progress=None):
        n_threads = (
            num_threads
            if num_threads is not None
            else (config.NUM_THREADS or 1)
        )
        executor = (
            ThreadPoolExecutor(max_workers=n_threads)
            if n_threads > 1
            else None
        )
        update_progress, close_progress = _make_progress(
            progress, len(self.path), "DS"
        )

        # Parallel-only neighbour cache: the DAG build already calls
        # _select_neighbors per node×variable; on_cache_ready is called by
        # _run_path right after DAG build (before any node futures run) so
        # _gather_neighborhood can skip the redundant _select_neighbors call.
        # Serial mode (executor is None) leaves _neighbor_cache as None and
        # _gather_neighborhood computes normally.
        self._neighbor_cache = None

        def _on_cache_ready(coord_cache):
            self._neighbor_cache = coord_cache

        try:
            _run_path(
                self.path,
                self.u_start,
                self.u_fallback,
                lambda i, x_i, u_st, u_fb: self._simulate_node(
                    i, x_i, u_st, u_fb
                ),
                self._write_result,
                update_progress,
                executor,
                self.offset_arr,
                self.vmap,
                self.n_k,
                self.sim_shape,
                self.max_radius,
                on_cache_ready=_on_cache_ready
                if executor is not None
                else None,
            )
        finally:
            close_progress()
            if executor is not None:
                executor.shutdown(wait=True)

        return self.sg


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
    progress=None,
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
        and must be a permutation of exactly the unknown-node set (missing
        nodes, extra nodes, and duplicate rows all raise ``ValueError``).
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
    progress : bool or callable or None, optional
        Show simulation progress. ``True`` prints a plain percentage line
        (no third-party dependency); a callable is invoked as
        ``progress(n_done, n_total)`` once per completed node.
        ``None``/``False`` (default) disables it.

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
    return engine.run(num_threads=num_threads, progress=progress)
