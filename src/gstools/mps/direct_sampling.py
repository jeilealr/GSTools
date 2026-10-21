"""
GStools subpackage providing the Direct Sampling MPS simulation class.

.. currentmodule:: gstools.mps

The following classes and functions are provided

.. autosummary::
   DirectSampling
"""

import warnings

import numpy as np

from gstools.field.base import Field
from gstools.mps.model import MPSModel
from gstools.mps.nonstationary import build_zone_selector, resolve_spec
from gstools.mps.simulate import ds_simulate
from gstools.normalizer.tools import apply_mean_norm_trend
from gstools.random.rng import RNG
from gstools.tools.geometric import no_of_angles

__all__ = ["DirectSampling"]


def _is_unset(value):
    """True for an unset seed control: ``None`` or NaN (audit DS13)."""
    if value is None:
        return True
    try:
        return bool(np.isnan(value))
    except (TypeError, ValueError):
        return False


def _check_finite_positions(pos):
    """Raise ``ValueError`` if a conditioning position is NaN or +-inf."""
    if not np.isfinite(pos).all():
        raise ValueError(
            "DirectSampling.set_condition: conditioning positions must be "
            "finite."
        )


class DirectSampling(Field):
    """Multiple Point Statistics simulation using Direct Sampling.

    Subclasses :class:`gstools.field.base.Field`. Takes an :class:`MPSModel`
    that bundles a :class:`TrainingImage` and all algorithm parameters, and
    produces fields on structured grids.

    Parameters
    ----------
    model : MPSModel
        Algorithm configuration: training image, scan fraction, threshold,
        conditioning weight, and boundary strategy. Neighbour count and
        maximum search radius live on each :class:`Variable` inside the
        :class:`TrainingImage`. Build one with
        ``MPSModel(ti, scan_fraction=…, threshold=…)``.
    seed : int or nan, optional
        Master RNG seed. Default: nan.

    Notes
    -----
    Runtime thread count is set via :attr:`num_threads` or passed to
    :meth:`__call__`.

    See Also
    --------
    MPSModel : holds the algorithm parameters.
    TrainingImage : training data holder.
    """

    default_field_names = ["field"]

    def __init__(
        self,
        model,
        seed=np.nan,
    ):
        if not isinstance(model, MPSModel):
            raise TypeError(
                "DirectSampling requires an MPSModel as its first argument. "
                "Wrap a TrainingImage first: "
                "DirectSampling(MPSModel(ti, scan_fraction=…))"
            )

        super().__init__(model=None, dim=model.ti.ndim, value_type="scalar")
        # Retain the MPSModel as the single source of truth for search
        # parameters. It is NOT stored in Field._model (which stays None) —
        # Field.pre_pos/field_dim/latlon/temporal assume a CovModel there.
        self._mps_model = model
        self._ti = model.ti
        self._num_threads = None
        self._cond_pos = None
        self._cond_val = None
        self._mv_mean = {}
        self._mv_normalizer = {}
        self._mv_trend = {}
        self.rng = RNG(None if _is_unset(seed) else int(seed))
        if model.ti.multivariate:
            for v in model.ti.variables:
                if v.name in dir(self):
                    raise ValueError(
                        f"DirectSampling: variable name {v.name!r} collides with an "
                        f"existing attribute."
                    )

    def __call__(
        self,
        pos=None,
        seed=np.nan,
        path_seed=np.nan,
        node_seed=np.nan,
        path="random",
        mesh_type="structured",
        post_process=True,
        store=True,
        num_threads=None,
    ):
        """Generate the spatial random field via Direct Sampling.

        For a univariate training image the field is saved as ``self.field``
        and is also returned. For a multivariate training image no single
        ``self.field`` is set — each variable is stored under its own name
        (accessible via ``self[variable]`` / :attr:`all_fields`) and a
        ``{variable: numpy.ndarray}`` dict is returned.

        Parameters
        ----------
        pos : :class:`list`, optional
            The position tuple, containing main direction and transversal
            directions. Only structured grids are supported.
        seed : :class:`int`, optional
            Seed for the RNG. If ``np.nan``, the current seed is kept.
            Default: ``np.nan``
        path_seed : :class:`int` or :any:`numpy.nan`, optional
            Seed controlling the order in which simulation grid nodes are
            visited. If ``np.nan`` (default), derived from the master RNG
            together with ``node_seed``. Fix this while varying ``node_seed``
            to study the effect of TI search randomness under a constant
            visit order, or vice versa.  Only has an effect when
            ``path="random"``; inert for ``"sequential"`` and explicit arrays.
            Default: ``np.nan``
        node_seed : :class:`int` or :any:`numpy.nan`, optional
            Seed controlling the TI scan entry point and fallback cell for
            every node. If ``np.nan`` (default), derived from the master RNG.
            Default: ``np.nan``
        path : :class:`str` or array-like, optional
            Node visit order.  ``"random"`` (default) shuffles the unknown
            nodes uniformly using the ``path_seed``-controlled RNG — the
            standard DS behaviour.  ``"sequential"`` visits nodes in raster
            (lexicographic) order, which is deterministic regardless of
            ``path_seed``.  An explicit integer array of shape ``(N, dim)``
            provides a caller-supplied visit order and must include every
            unknown node (conditioned nodes are ignored; missing unknown
            nodes and duplicate rows raise :class:`ValueError`).
            Default: ``"random"``
        mesh_type : :class:`str`, optional
            Grid type. Must be ``"structured"``.
            Default: ``"structured"``
        post_process : :class:`bool`, optional
            Whether to apply post-processing transformations (mean,
            normalizer, trend) to the field. Default: :any:`True`
        store : :class:`bool` or :class:`str`, optional
            Whether to store the field (``True``), not store it (``False``),
            or store it under a custom name (string).
            Default: :any:`True`
        Returns
        -------
        field : :class:`numpy.ndarray` or :class:`dict`
            For a univariate training image, the simulated field (also saved as
            ``self.field``). For a multivariate training image, a
            ``{variable: numpy.ndarray}`` dict with all variables on equal
            footing — each is also stored as a named field accessible via
            ``self[variable]`` / :attr:`all_fields`.
        """
        if mesh_type != "structured":
            raise ValueError(
                "DirectSampling: only structured grids are supported."
            )
        if self._ti.multivariate and isinstance(store, str):
            # A multivariate run produces one field per variable; there is no
            # single privileged field to store under a custom name. Reject it
            # explicitly rather than silently dropping the name (the fields are
            # always stored under their variable names).
            raise ValueError(
                "DirectSampling: a custom store name is not supported for "
                "multivariate training images; each variable is stored under "
                "its own name. Use store=True/False instead."
            )
        name, save = self.get_store_config(store)
        pos, shape = self.pre_pos(pos, mesh_type)
        # Reject non-finite grid coordinates before snapping, coordinate maps or
        # stored field metadata can use them (audit DS15). Nonuniform *finite*
        # axes remain allowed; their spacing is interpreted in index space.
        for axis in self.pos:
            if not np.all(np.isfinite(np.asarray(axis, dtype=np.double))):
                raise ValueError(
                    "DirectSampling: simulation grid coordinates must be finite "
                    "(NaN/inf axis values are not allowed)."
                )
        # Resolve the MPSModel's grid-independent rotation/scale specs to
        # canonical per-node maps now that the grid is known (Mariethoz2010
        # §6.2). Non-callable arrays are re-validated here against the grid
        # shape, and callables are materialized and finiteness-checked.
        # Coordinate callables receive one flat array per axis (meshgridded in
        # C-order), so ``f(*grid_coords)`` yields one value per simulation node.
        grid_coords = [
            g.ravel() for g in np.meshgrid(*self.pos, indexing="ij")
        ]
        rotation_map, _ = resolve_spec(
            self._mps_model.rotation,
            shape,
            no_of_angles(len(shape)),
            grid_coords,
            "rotation",
            angles=True,
        )
        scale_map, _ = resolve_spec(
            self._mps_model.scale,
            shape,
            len(shape),
            grid_coords,
            "scale",
            positive=True,
        )
        # Resolve the zone masks against this grid into an integer selector
        # (0 = primary TI, i+1 = zones[i]); overlap/shape/dtype errors surface
        # here. The native engine maps it to a per-node domain and scans each
        # node's TI, with neighbours still gathered across zone boundaries.
        zones = self._mps_model.zones
        zone_selector = (
            build_zone_selector(zones, shape, grid_coords) if zones else None
        )
        conditions = self._conditions_to_grid(self.pos)
        # ``None`` is accepted as "unset" alongside ``nan`` for every seed
        # control (audit DS13), so callers need not know the nan sentinel.
        if not _is_unset(seed):
            self.rng.seed = int(seed)
        rng_path = (
            self.rng.random
            if _is_unset(path_seed)
            else RNG(int(path_seed)).random
        )
        rng_nodes = (
            self.rng.random
            if _is_unset(node_seed)
            else RNG(int(node_seed)).random
        )
        # Call-time num_threads overrides the instance default.
        n_threads = (
            num_threads if num_threads is not None else self._num_threads
        )
        # ds_simulate now accepts both univariate and multivariate TIs directly.
        # Univariate TIs produce {None: array}; multivariate produce {name: array}.
        # Conditions for univariate are {idx: value}; ds_simulate expects
        # {idx: {var: value}} so we remap using the sentinel key None.
        if self._ti.multivariate:
            sim_conditions = conditions  # already {idx: {var: val}}
        else:
            # Univariate: key conditions by the single variable's actual name
            # (``None`` for the anonymous bare-array sugar, else the given name).
            uv_name = self._ti.variables[0].name
            sim_conditions = (
                {idx: {uv_name: val} for idx, val in conditions.items()}
                if conditions
                else None
            )
        result = ds_simulate(
            training_image=self._ti,
            sim_shape=shape,
            threshold=self.threshold,
            scan_fraction=self.scan_fraction,
            rng_path=rng_path,
            rng_nodes=rng_nodes,
            conditions=sim_conditions,
            boundary=self.boundary,
            num_threads=n_threads,
            rotation_map=rotation_map,
            scale_map=scale_map,
            path=path,
            scan_path=self._mps_model.scan_path,
            zones=zones,
            zone_selector=zone_selector,
            post_processing=self._mps_model.post_processing,
            post_processing_factor=self._mps_model.post_processing_factor,
            post_processing_path=self._mps_model.post_processing_path,
            post_processing_variables=self._mps_model.post_processing_variables,
            post_processing_mask=self._mps_model.post_processing_mask,
        )
        # Branch only on the return type: multivariate → dict of named arrays;
        # univariate → bare array (unwrap the single None key).
        if self._ti.multivariate:
            # Equal treatment: every variable is a first-class named field
            # (no privileged primary). Returned as a dict keyed by variable name.
            # Per-variable transforms (mean/normalizer/trend) are applied here
            # using the dicts set via set_mv_transforms(); variables absent from
            # those dicts fall back to the inherited self.mean/normalizer/trend.
            # post_field is then called with process=False to handle storage only.
            out = {}
            for v in self._ti.variables:
                fld = result[v.name]
                if post_process:
                    mv_mean = self._mv_mean.get(v.name, self.mean)
                    mv_norm = self._mv_normalizer.get(v.name, self.normalizer)
                    mv_trend = self._mv_trend.get(v.name, self.trend)
                    fld = apply_mean_norm_trend(
                        pos=self.pos,
                        field=fld,
                        mesh_type=self.mesh_type,
                        value_type=self.value_type,
                        mean=mv_mean,
                        normalizer=mv_norm,
                        trend=mv_trend,
                        check_shape=False,
                        stacked=False,
                    )
                out[v.name] = self.post_field(
                    fld, name=v.name, process=False, save=save
                )
            return out
        return self.post_field(
            result[self._ti.variables[0].name], name, post_process, save
        )

    def _conditions_to_grid(self, axes):
        """Snap conditioning points to nearest grid nodes (Mariethoz2010 §3 ¶12).

        Univariate returns ``{idx: value}``; multivariate returns
        ``{idx: {variable: value}}`` with non-finite (NaN) entries skipped. When
        two points snap to the same node, the conflict is resolved
        **per variable**: for each variable the closest colliding point with a
        finite value wins. A farther point therefore still contributes its
        finite values for any variable the closer point left ``NaN``, so valid
        conditioning data is never silently discarded.

        Parameters
        ----------
        axes : tuple of numpy.ndarray
            Raw grid axis arrays (``self.pos``).

        Returns
        -------
        dict
        """
        if self._cond_pos is None:
            return {}
        # Points beyond half a cell outside the grid snap to its edge: warn.
        outside = 0
        for d, ax in enumerate(axes):
            ax = np.asarray(ax, dtype=np.double)
            tol = 0.5 * np.abs(np.diff(ax)).max() if len(ax) > 1 else 0.0
            p = np.asarray(self._cond_pos[d], dtype=np.double)
            outside |= (p < ax.min() - tol) | (p > ax.max() + tol)
        if np.any(outside):
            warnings.warn(
                f"DirectSampling: {int(np.sum(outside))} conditioning "
                "point(s) lie outside the simulation grid and were snapped "
                "to its nearest edge node.",
                UserWarning,
                stacklevel=3,
            )
        n_collided = 0  # finite values landing on an occupied (node, variable)
        if self._ti.multivariate and isinstance(self._cond_val, dict):
            if not self._cond_val:
                raise ValueError("DirectSampling: cond_val must not be empty")
            # idx -> {variable: (value, dist_sq)}; the closest finite value per
            # variable wins, so a farther point fills variables the closer one
            # left NaN (per-variable collision resolution).
            candidates = {}
            n_cond = len(next(iter(self._cond_val.values())))
            for k in range(n_cond):
                idx, dist_sq = self._snap_to_nearest(axes, k)
                var_best = candidates.setdefault(idx, {})
                for v in self._cond_val:
                    val = self._cond_val[v][k]
                    if not np.isfinite(val):
                        continue
                    if v in var_best:
                        n_collided += 1
                    if v not in var_best or dist_sq < var_best[v][1]:
                        var_best[v] = (float(val), dist_sq)
            result = {
                idx: {v: val for v, (val, _) in var_best.items()}
                for idx, var_best in candidates.items()
            }
        else:
            # univariate (NaN values were already dropped in set_condition)
            candidates = {}  # idx -> (val, dist_sq)
            for k in range(self._cond_val.shape[0]):
                idx, dist_sq = self._snap_to_nearest(axes, k)
                if idx in candidates:
                    n_collided += 1
                if idx not in candidates or dist_sq < candidates[idx][1]:
                    candidates[idx] = (self._cond_val[k], dist_sq)
            result = {idx: val for idx, (val, _) in candidates.items()}
        if n_collided:
            warnings.warn(
                f"DirectSampling: {n_collided} conditioning value(s) collided "
                "on an occupied grid node; the one closest to the node was "
                "kept (the first one on ties).",
                UserWarning,
                stacklevel=3,
            )
        return result

    def _snap_to_nearest(self, axes, k):
        """Return (grid_index_tuple, dist_sq) for the k-th conditioning point."""
        idx = tuple(
            int(np.argmin(np.abs(axes[d] - self._cond_pos[d][k])))
            for d in range(self.dim)
        )
        dist_sq = sum(
            (axes[d][idx[d]] - self._cond_pos[d][k]) ** 2
            for d in range(self.dim)
        )
        return idx, dist_sq

    def set_condition(self, cond_pos, cond_val):
        """Set the conditioning data for the simulation.

        Conditioning weight is configured per variable on the :class:`Variable`
        (``cond_weight=``), or via the univariate :class:`TrainingImage` sugar.

        Parameters
        ----------
        cond_pos : :class:`list`
            The position tuple of the conditioning data ``(x, [y, z])``.
        cond_val : :class:`numpy.ndarray` or :class:`dict`
            Univariate: values at the conditioning positions. Multivariate: a
            ``{variable: numpy.ndarray}`` mapping (use ``numpy.nan`` for a
            variable that is not conditioned at a given point).
        """
        # Conditioning updates are transactional (audit DS08): everything is
        # converted and validated into locals first, and ``self._cond_pos`` /
        # ``self._cond_val`` are reassigned only after every check passes, so a
        # rejected update never corrupts the state a later simulation reuses.
        if self._ti.multivariate and isinstance(cond_val, dict):
            if not cond_val:
                raise ValueError("DirectSampling: cond_val must not be empty")
            unknown = set(cond_val) - {v.name for v in self._ti.variables}
            if unknown:
                raise ValueError(
                    f"cond_val contains unknown variable(s): {sorted(unknown)}. "
                    f"TI variables are: {sorted(v.name for v in self._ti.variables)}"
                )
            cond_pos_arr = np.asarray(cond_pos, dtype=np.double).reshape(
                self.dim, -1
            )
            _check_finite_positions(cond_pos_arr)
            n_cond = cond_pos_arr.shape[1]
            local_val = {}
            for v, a in cond_val.items():
                arr = np.asarray(a, dtype=np.double)
                # Require exactly one scalar per conditioning position; reject an
                # (N, k) array up front instead of failing later in snapping with
                # an ambiguous-truth error (audit DS08).
                if arr.ndim != 1:
                    raise ValueError(
                        f"DirectSampling: cond_val[{v!r}] must be a 1-D array "
                        f"with one value per conditioning position; got shape "
                        f"{arr.shape!r}."
                    )
                if arr.shape[0] != n_cond:
                    raise ValueError(
                        "DirectSampling: all cond_val arrays must match the "
                        f"conditioning-position count ({n_cond}); got "
                        f"{arr.shape[0]} for {v!r}."
                    )
                local_val[v] = arr
            self._check_penalty_codes(local_val)  # DS09
            self._cond_pos = cond_pos_arr
            self._cond_val = local_val
        elif self._ti.multivariate:
            raise ValueError(
                "DirectSampling: cond_val must be a dict {variable: array} "
                "for multivariate TrainingImages."
            )
        else:
            from gstools.krige.tools import set_condition as _gs_set_condition

            # Validate into locals BEFORE storing so a non-finite position does
            # not leave invalid conditioning installed (audit DS08 univariate).
            local_pos, local_val = _gs_set_condition(
                cond_pos, cond_val, self.dim
            )
            _check_finite_positions(local_pos)
            uv_name = self._ti.variables[0].name
            self._check_penalty_codes(
                {uv_name: np.asarray(local_val, dtype=np.double)}
            )  # DS09
            self._cond_pos, self._cond_val = local_pos, local_val

    def _check_penalty_codes(self, values_by_name):
        """Screen hard category codes against active penalty matrices (DS09).

        A categorical penalty matrix is indexed by the hard code during the
        scan, so a non-integer, negative, or out-of-range code would silently
        become a full-mismatch (or, for the raw kernel, an invalid index). The
        effective matrices are the primary TI's and any cleanup
        ``post_processing_variables`` override's; a finite hard value must be a
        valid code ``[0, C)`` of each.
        """
        from gstools.mps.training_image import _check_category_codes

        checks = {}
        for var in self._ti.variables:
            if var.penalty_matrix is not None:
                checks.setdefault(var.name, []).append(
                    (var.penalty_matrix.shape[0], "penalty_matrix")
                )
        for ov in self._mps_model.post_processing_variables:
            if ov.penalty_matrix is not None:
                checks.setdefault(ov.name, []).append(
                    (
                        ov.penalty_matrix.shape[0],
                        "post_processing penalty_matrix",
                    )
                )
        for name, matrix_checks in checks.items():
            if name not in values_by_name:
                continue
            vals = np.asarray(values_by_name[name], dtype=np.double)
            finite = vals[np.isfinite(vals)]
            for ncat, label in matrix_checks:
                _check_category_codes(
                    finite,
                    ncat,
                    f"DirectSampling.set_condition: conditioning code for "
                    f"{name!r} (checked against {label})",
                )

    def set_mv_transforms(self, mean=None, normalizer=None, trend=None):
        """Set per-variable post-processing transforms for multivariate simulations.

        Only meaningful when the training image is multivariate. Variables not
        listed here fall back to the instance-level ``self.mean``,
        ``self.normalizer``, and ``self.trend``.

        Parameters
        ----------
        mean : dict of {str: scalar or callable}, optional
            Per-variable mean.
        normalizer : dict of {str: Normalizer}, optional
            Per-variable normalizer.
        trend : dict of {str: scalar or callable}, optional
            Per-variable trend (applied after denormalization).
        """
        from gstools.field.base import _set_mean_trend
        from gstools.normalizer.tools import _check_normalizer

        if mean is not None:
            self._mv_mean = {
                v: _set_mean_trend(m, self.dim) for v, m in mean.items()
            }
        if normalizer is not None:
            self._mv_normalizer = {
                v: _check_normalizer(n) for v, n in normalizer.items()
            }
        if trend is not None:
            self._mv_trend = {
                v: _set_mean_trend(t, self.dim) for v, t in trend.items()
            }

    @property
    def mps_model(self):
        """:any:`MPSModel`: the search-parameter configuration (training image + parameters)."""
        return self._mps_model

    @property
    def ti(self):
        """:any:`TrainingImage`: The training image model."""
        return self._ti

    def _collapse_per_var(self, attr):
        """Read ``attr`` from every TI Variable; scalar if all equal, else a dict.

        Read-only view over the variables' values. Used by the
        :attr:`n_neighbors` and :attr:`max_radius` getters.
        """
        vars_ = self._ti.variables
        vals = [getattr(v, attr) for v in vars_]
        if len(set(vals)) == 1:
            return vals[0]
        return {v.name: getattr(v, attr) for v in vars_}

    @property
    def n_neighbors(self):
        """:class:`int` or :class:`dict`: n_neighbors per variable (scalar if all equal).

        Read-only. Configure on the :class:`Variable` (``ti.variable(name).n_neighbors = …``)
        or at :class:`TrainingImage` construction.
        """
        return self._collapse_per_var("n_neighbors")

    @property
    def scan_fraction(self):
        """:class:`float`: Fraction of the TI scanned per node. Read-only; set on the MPSModel."""
        return self._mps_model.scan_fraction

    @property
    def threshold(self):
        """:class:`float`: Distance threshold (0.0 → DSBC mode). Read-only; set on the MPSModel."""
        return self._mps_model.threshold

    @property
    def cond_weight(self):
        """:class:`float` or :class:`dict`: Conditioning-node weight per variable (scalar if all equal).

        Read-only. Configure on the :class:`Variable` (``ti.variable(name).cond_weight``)
        or via the univariate :class:`TrainingImage` ``cond_weight=`` sugar.
        """
        return self._collapse_per_var("cond_weight")

    @property
    def boundary(self):
        """:class:`str`: Search-window strategy. Read-only; set on the MPSModel."""
        return self._mps_model.boundary

    @property
    def max_radius(self):
        """:class:`float` or :any:`None` or :class:`dict`: Euclidean neighbour cap (scalar if all equal).

        Read-only. Configure at :class:`Variable`/:class:`TrainingImage` construction.
        """
        return self._collapse_per_var("max_radius")

    @property
    def num_threads(self):
        """:class:`int` or :any:`None`: Number of threads for outer DAG parallelism."""
        return self._num_threads

    @num_threads.setter
    def num_threads(self, value):
        self._num_threads = None if value is None else int(value)

    def __repr__(self):
        parts = [
            f"DirectSampling(dim={self.dim}, ",
            f"n_neighbors={self.n_neighbors}, ",
            f"scan_fraction={self.scan_fraction}, ",
            f"threshold={self.threshold}, ",
            f"boundary={self.boundary!r}",
        ]
        if self._mps_model.rotation is not None:
            parts.append(f", rotation={self._mps_model.rotation!r}")
        if self._mps_model.scale is not None:
            parts.append(f", scale={self._mps_model.scale!r}")
        return "".join(parts) + ")"
