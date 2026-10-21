"""MPS configuration object — training image and search parameters."""

import warnings

import numpy as np

from gstools.mps.training_image import (
    TrainingImage,
    Variable,
    _check_category_codes,
    _post_pass_n_neighbors,
    _resolve_post_ti,
)
from gstools.mps.zone import Zone
from gstools.tools.geometric import no_of_angles

__all__ = ["MPSModel"]

_VALID_BOUNDARY = ("strict", "partial")
_VALID_SCAN_PATH = ("sequential", "random")
_VALID_POST_PATH = ("same", "random", "sequential")


def _validate_post_processing(value):
    """Validate post-processing pass count p >= 0 (Mariethoz2010 §4)."""
    v = int(value)
    if v < 0:
        raise ValueError(
            f"MPSModel: post_processing must be >= 0, got {value!r}"
        )
    return v


def _validate_post_processing_factor(value):
    """Validate p_f > 0 and finite (it divides the main f and n_neighbors)."""
    v = float(value)
    if not (np.isfinite(v) and v > 0.0 and np.isfinite(1.0 / v)):
        raise ValueError(
            "MPSModel: post_processing_factor must be finite and > 0, "
            f"got {value!r}"
        )
    return v


def _validate_post_processing_path(value):
    """None -> inherit main-path mode; str from _VALID_POST_PATH; else (N, dim) array."""
    if value is None or (isinstance(value, str) and value in _VALID_POST_PATH):
        return value
    if isinstance(value, str):
        raise ValueError(
            f"MPSModel: post_processing_path must be one of "
            f"{_VALID_POST_PATH!r}, an (N, dim) integer array, or None "
            f"(inherit the main path mode), got {value!r}"
        )
    arr = np.asarray(value)
    if arr.ndim != 2:
        raise ValueError(
            f"MPSModel: an explicit post_processing_path must be a 2-D "
            f"(N, dim) array, got shape {arr.shape!r}"
        )
    return arr


def _validate_post_processing_variables(
    post_processing_variables, primary_ti, zones=()
):
    """Validate a list[Variable] of post-processing overrides against primary_ti."""
    if post_processing_variables is None:
        return []
    post_processing_variables = list(post_processing_variables)
    if not all(isinstance(v, Variable) for v in post_processing_variables):
        raise TypeError(
            "MPSModel: post_processing_variables must be a list of Variable "
            "objects."
        )
    names = [v.name for v in post_processing_variables]
    if len(set(names)) != len(names):
        raise ValueError(
            f"MPSModel: post_processing_variables names must be unique, "
            f"got {names!r}."
        )
    primary_names = {v.name for v in primary_ti.variables}
    unknown = set(names) - primary_names
    if unknown:
        raise ValueError(
            f"MPSModel: post_processing_variables name(s) "
            f"{sorted(map(str, unknown))!r} not found in the primary TI's "
            f"variables {sorted(map(str, primary_names))!r}."
        )
    # An override changes search settings only: post-processing pastes from the
    # primary (and zone) data, so its data and categorical kind must match.
    for ov in post_processing_variables:
        pv = primary_ti.variable(ov.name)
        label = f"post_processing_variables[{ov.name!r}]"
        same_data = ov.data is pv.data or np.array_equal(
            ov.data, pv.data, equal_nan=True
        )
        if ov.categorical != pv.categorical or not same_data:
            raise ValueError(
                f"MPSModel: {label} must keep the primary Variable's data "
                "and categorical kind; build it with "
                "ti.variable(name).replace(...)."
            )
        if ov.penalty_matrix is not None:
            for i, z in enumerate(zones):
                _check_category_codes(
                    z.ti.variable(ov.name).data,
                    ov.penalty_matrix.shape[0],
                    f"MPSModel: zones[{i}].ti data (checked against "
                    f"{label}.penalty_matrix)",
                )
    # Early check of the post-pass TI (weights set on all variables or none).
    _resolve_post_ti(primary_ti, post_processing_variables)
    return post_processing_variables


def _validate_post_pass_effort(ti, post_processing, p_f, overrides):
    """Early check that post-pass neighbour counts are valid (variation >= 2)."""
    if post_processing > 0:
        _post_pass_n_neighbors(_resolve_post_ti(ti, overrides), p_f)


def _validate_post_processing_mask(value):
    """Light validation only; grid-dependent shape is re-checked at call time."""
    if value is None:
        return None
    arr = np.asarray(value)
    if arr.dtype != bool:
        raise TypeError(
            f"MPSModel: post_processing_mask must be a boolean array or "
            f"None, got dtype {arr.dtype!r}."
        )
    return arr


def _validate_boundary(value):
    """Validate boundary; return normalized value or raise ValueError."""
    if value not in _VALID_BOUNDARY:
        raise ValueError(
            f"MPSModel: boundary must be one of {_VALID_BOUNDARY!r}, "
            f"got {value!r}"
        )
    return value


def _validate_scan_path(value):
    """Validate scan_path; return normalized value or raise ValueError."""
    if value not in _VALID_SCAN_PATH:
        raise ValueError(
            f"MPSModel: scan_path must be one of {_VALID_SCAN_PATH!r}, "
            f"got {value!r}"
        )
    return value


def _validate_scan_fraction(value):
    """Validate scan_fraction in (0, 1]; return float or raise ValueError."""
    if not (0 < float(value) <= 1):
        raise ValueError(
            f"MPSModel: scan_fraction must be in (0, 1], got {value!r}"
        )
    return float(value)


def _validate_threshold(value):
    """Validate threshold: finite and >= 0, warn if > 1; return float.

    NaN/inf are rejected up front (a NaN threshold would otherwise slip past
    the ``< 0`` / ``> 1`` comparisons; cf. GeoStat-Framework/GSTools#416).
    """
    v = float(value)
    if not np.isfinite(v):
        raise ValueError(f"MPSModel: threshold must be finite, got {value!r}")
    if v < 0:
        raise ValueError(f"MPSModel: threshold must be >= 0, got {value!r}")
    if v > 1.0:
        warnings.warn(
            "MPSModel: threshold > 1.0 guarantees the first candidate is always accepted.",
            UserWarning,
            stacklevel=3,
        )
    return v


def _validate_rotation(spec, dim):
    """Validate a grid-independent rotation spec.

    Grid-dependent shape resolution happens at call time in
    :func:`gstools.mps.nonstationary.resolve_spec`; here we only reject what is
    knowable without the grid: rotation on a 1-D training image (no angles),
    and non-finite scalar/array values. Callables are stored unvalidated and
    checked when resolved.
    """
    if spec is None:
        return None
    if no_of_angles(dim) == 0:
        raise ValueError(
            f"MPSModel: rotation is undefined for a {dim}-D training image "
            "(1-D has no rotation angles)."
        )
    if callable(spec):
        return spec
    arr = np.asarray(spec, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        raise ValueError("MPSModel: all rotation values must be finite.")
    return arr


def _validate_scale(spec, dim):
    """Validate a grid-independent scale spec (finite and strictly positive).

    Supported in every dimension (uniform scale dilates all axes). Callables
    are stored unvalidated and checked when resolved.
    """
    if spec is None:
        return None
    if callable(spec):
        return spec
    arr = np.asarray(spec, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        raise ValueError("MPSModel: all scale values must be finite.")
    if not np.all(arr > 0):
        raise ValueError(
            "MPSModel: all scale values must be > 0 "
            f"(minimum found: {float(np.min(arr))!r})."
        )
    return arr


def _validate_zones(zones, primary_ti):
    """Every zone TI must share the primary TI's variable set and kinds."""
    if zones is None:
        return []
    zones = list(zones)
    primary_kinds = {v.name: v.categorical for v in primary_ti.variables}
    for i, z in enumerate(zones):
        if not isinstance(z, Zone):
            raise TypeError(
                f"MPSModel: zones[{i}] must be a Zone, got {type(z)!r}."
            )
        if z.ti.ndim != primary_ti.ndim:
            raise ValueError(
                f"MPSModel: zones[{i}].ti has dim {z.ti.ndim}, primary TI "
                f"has dim {primary_ti.ndim}."
            )
        zone_kinds = {v.name: v.categorical for v in z.ti.variables}
        if set(zone_kinds) != set(primary_kinds):
            raise ValueError(
                f"MPSModel: zones[{i}].ti variable set "
                f"{sorted(map(str, zone_kinds))} does not match the primary "
                f"TI's {sorted(map(str, primary_kinds))}."
            )
        for name, kind in primary_kinds.items():
            if zone_kinds[name] != kind:
                raise ValueError(
                    f"MPSModel: zones[{i}].ti variable {name!r} is "
                    f"{'categorical' if zone_kinds[name] else 'continuous'} "
                    f"but the primary TI's is "
                    f"{'categorical' if kind else 'continuous'} — the "
                    "categorical/continuous kind must match per variable."
                )
            # The scan indexes the PRIMARY TI's penalty_matrix with codes read
            # from the SELECTED domain, so a zone code >= C would raise a raw
            # IndexError mid-scan (and a negative one would wrap silently).
            penalty_matrix = primary_ti.variable(name).penalty_matrix
            if penalty_matrix is not None:
                label = "data" if name is None else f"variable {name!r} data"
                _check_category_codes(
                    z.ti.variable(name).data,
                    penalty_matrix.shape[0],
                    f"MPSModel: zones[{i}].ti {label} (checked against the "
                    "primary TI's penalty_matrix)",
                )
    return zones


class MPSModel:
    """MPS configuration: training image and search algorithm parameters.

    Analogous to ``CovModel``: holds the training image and the search
    hyper-parameters. Carries no seed, thread count, or backend selector —
    those are call-time concerns on :class:`~gstools.mps.DirectSampling`.

    Parameters
    ----------
    ti : :any:`TrainingImage`
        Training image (univariate or multivariate).
    scan_fraction : :class:`float`, optional
        Fraction of the TI to scan per node (capped at the valid search
        window). Must be in (0, 1]. Default: 1.0.
    threshold : :class:`float`, optional
        Distance threshold for early acceptance. 0.0 → DSBC mode. Default: 0.0.
    boundary : :class:`str`, optional
        Search-window strategy: ``"strict"`` (default) or ``"partial"``.
    scan_path : :class:`str`, optional
        TI scan order at each node: ``"sequential"`` (default) — a wrap-around
        walk from a random start — or ``"random"`` — ``scan_fraction``'s worth
        of distinct window cells in a node-local random order. Both are
        reproducible and thread-count independent. Default: ``"sequential"``.
    zones : :class:`list` of :any:`Zone` or None, optional
        Region-bound training images (Mariethoz2010 para [40]): nodes inside a
        zone's region scan that zone's TI instead of the primary one, while the
        data event still spans all simulation-grid neighbours. Every zone TI
        must share the primary TI's variable set and kinds. ``None`` → no zones.
        Default: ``None``.
    post_processing : :class:`int`, optional
        Number of cleanup passes (Mariethoz2010 §4): each re-simulates already
        simulated nodes with a fully informed neighbourhood to reduce noise.
        ``0`` → no cleanup. Default: ``0``.
    post_processing_factor : :class:`float`, optional
        Divides ``scan_fraction`` and ``n_neighbors`` for the cleanup passes
        (``> 1`` cheaper, ``< 1`` more thorough). Must be finite and > 0.
        Default: ``1.0``.
    post_processing_path : :class:`str`, array-like, or None, optional
        Cleanup visit order: ``None`` → inherit the main-path mode; ``"same"``
        → reuse the main path (filtered to eligible nodes); ``"random"`` /
        ``"sequential"``; or an explicit ``(N, dim)`` array. Default: ``None``.
    post_processing_variables : :class:`list` of :any:`Variable` or None, optional
        Per-variable search-setting overrides for the cleanup passes, built with
        ``ti.variable(name).replace(...)`` (data/kind must match the primary).
        ``None`` → inherit the primary settings. Default: ``None``.
    post_processing_mask : :class:`numpy.ndarray` of bool or None, optional
        Which simulation-grid nodes are eligible for cleanup update (shape
        checked at call time). ``None`` → all simulated nodes. Default: ``None``.
    rotation : scalar, array-like, or callable, optional
        Grid-independent non-stationary rotation spec (Tait-Bryan angles;
        ``no_of_angles(dim)`` components). Resolved per node at call time by
        :func:`gstools.mps.nonstationary.resolve_spec`. Undefined in 1-D.
        ``None`` → no rotation. Default: ``None``.
    scale : scalar, array-like, or callable, optional
        Grid-independent non-stationary scale spec (``dim`` components; a
        scalar dilates all axes). Must be strictly positive. Resolved per node
        at call time. ``None`` → no scaling. Default: ``None``.
    """

    def __init__(
        self,
        ti,
        scan_fraction=1.0,
        threshold=0.0,
        boundary="strict",
        scan_path="sequential",
        rotation=None,
        scale=None,
        zones=None,
        post_processing=0,
        post_processing_factor=1.0,
        post_processing_path=None,
        post_processing_variables=None,
        post_processing_mask=None,
    ):
        if not isinstance(ti, TrainingImage):
            raise TypeError(
                f"MPSModel: ti must be a TrainingImage, got {type(ti)!r}"
            )
        self._ti = ti
        self._scan_fraction = _validate_scan_fraction(scan_fraction)
        self._threshold = _validate_threshold(threshold)
        self._boundary = _validate_boundary(boundary)
        self._scan_path = _validate_scan_path(scan_path)
        dim = len(ti.shape)
        self._rotation = _validate_rotation(rotation, dim)
        self._scale = _validate_scale(scale, dim)
        self._zones = _validate_zones(zones, ti)
        self._post_processing = _validate_post_processing(post_processing)
        self._post_processing_factor = _validate_post_processing_factor(
            post_processing_factor
        )
        self._post_processing_path = _validate_post_processing_path(
            post_processing_path
        )
        self._post_processing_variables = _validate_post_processing_variables(
            post_processing_variables, ti, self._zones
        )
        self._post_processing_mask = _validate_post_processing_mask(
            post_processing_mask
        )
        # Early failure for an impossible post-pass effort (variation < 2).
        _validate_post_pass_effort(
            ti,
            self._post_processing,
            self._post_processing_factor,
            self._post_processing_variables,
        )

    @property
    def ti(self):
        """:any:`TrainingImage`: The training image."""
        return self._ti

    @property
    def scan_fraction(self):
        """:class:`float`: Fraction of the TI scanned per node."""
        return self._scan_fraction

    @scan_fraction.setter
    def scan_fraction(self, value):
        self._scan_fraction = _validate_scan_fraction(value)

    @property
    def threshold(self):
        """:class:`float`: Distance threshold for early acceptance."""
        return self._threshold

    @threshold.setter
    def threshold(self, value):
        self._threshold = _validate_threshold(value)

    @property
    def boundary(self):
        """:class:`str`: Search-window boundary strategy (``"strict"`` or ``"partial"``)."""
        return self._boundary

    @boundary.setter
    def boundary(self, value):
        self._boundary = _validate_boundary(value)

    @property
    def rotation(self):
        """scalar, array-like, callable, or None: non-stationary rotation spec."""
        return self._rotation

    @rotation.setter
    def rotation(self, value):
        self._rotation = _validate_rotation(value, len(self._ti.shape))

    @property
    def scale(self):
        """scalar, array-like, callable, or None: non-stationary scale spec."""
        return self._scale

    @scale.setter
    def scale(self, value):
        self._scale = _validate_scale(value, len(self._ti.shape))

    @property
    def scan_path(self):
        """:class:`str`: TI scan order per node, ``"sequential"`` or ``"random"``."""
        return self._scan_path

    @scan_path.setter
    def scan_path(self, value):
        self._scan_path = _validate_scan_path(value)

    @property
    def zones(self):
        """:class:`list` of :any:`Zone`: Zone TIs scanned inside their regions (``[]`` if none)."""
        return list(self._zones)

    @zones.setter
    def zones(self, value):
        self._zones = _validate_zones(value, self._ti)

    @property
    def post_processing(self):
        """:class:`int`: Number of cleanup (post-processing) passes (``0`` → none)."""
        return self._post_processing

    @property
    def post_processing_factor(self):
        """:class:`float`: Divisor for cleanup ``scan_fraction``/``n_neighbors`` effort."""
        return self._post_processing_factor

    @property
    def post_processing_path(self):
        """:class:`str`, array, or None: Cleanup visit order (``None`` → inherit main path)."""
        return self._post_processing_path

    @property
    def post_processing_variables(self):
        """:class:`list` of :any:`Variable`: Per-variable cleanup overrides (``[]`` if none)."""
        return list(self._post_processing_variables)

    @property
    def post_processing_mask(self):
        """:class:`numpy.ndarray` of bool or None: Which nodes are eligible for cleanup."""
        return self._post_processing_mask

    def __repr__(self):
        args = [repr(self._ti)]
        defaults = dict(
            scan_fraction=1.0,
            threshold=0.0,
            boundary="strict",
            scan_path="sequential",
        )
        for name, default in defaults.items():
            val = getattr(self, f"_{name}")
            if val != default:
                args.append(f"{name}={val!r}")
        return f"MPSModel({', '.join(args)})"
