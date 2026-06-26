"""MPS configuration object — training image and search parameters."""

import warnings

from gstools.mps.training_image import TrainingImage

__all__ = ["MPSModel"]

_VALID_BOUNDARY = ("strict", "partial")


def _validate_boundary(value):
    """Validate boundary; return normalized value or raise ValueError."""
    if value not in _VALID_BOUNDARY:
        raise ValueError(
            f"MPSModel: boundary must be one of {_VALID_BOUNDARY!r}, "
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
    """Validate threshold >= 0, warn if > 1; return float or raise ValueError."""
    if float(value) < 0:
        raise ValueError(f"MPSModel: threshold must be >= 0, got {value!r}")
    if float(value) > 1.0:
        warnings.warn(
            "MPSModel: threshold > 1.0 guarantees the first candidate is always accepted.",
            UserWarning,
            stacklevel=3,
        )
    return float(value)


def _validate_max_radius(value):
    """Validate max_radius is positive or None; return float/None or raise ValueError."""
    if value is not None and float(value) <= 0:
        raise ValueError(
            f"MPSModel: max_radius must be a positive float, got {value!r}"
        )
    return float(value) if value is not None else None


def _validate_variation_n_neighbors(n_neighbors, ti):
    """Reject variation distance with a single-neighbour data event.

    The variation distance (Mariethoz2010 Eq. 9) compares each event to its own
    local mean. With ``n_neighbors == 1`` the local mean equals the sole value,
    so every deviation is zero and every TI candidate matches — the scan
    degenerates to a random draw. Fail fast instead.
    """
    dist_type = ti.distance_type
    if isinstance(dist_type, dict):
        dist_map = dist_type
        n_map = (
            n_neighbors
            if isinstance(n_neighbors, dict)
            else {v: n_neighbors for v in dist_map}
        )
    else:
        dist_map = {None: dist_type}
        n_map = {None: n_neighbors}
    for var, dt in dist_map.items():
        if isinstance(dt, str) and dt.lower().startswith("variation"):
            if int(n_map[var]) < 2:
                where = "" if var is None else f" for variable {var!r}"
                raise ValueError(
                    f"distance='variation' requires n_neighbors >= 2{where}: "
                    "a single-point data event has no local mean deviation."
                )


def _validate_n_neighbors(value, ti):
    """Validate and normalise *n_neighbors*; return ``int`` or ``dict of int``."""
    if isinstance(value, dict):
        if not ti.multivariate:
            raise ValueError(
                "MPSModel: dict n_neighbors is only valid for "
                "multivariate TrainingImages."
            )
        missing = set(ti.variables) - set(value)
        extra = set(value) - set(ti.variables)
        if missing or extra:
            raise ValueError(
                f"MPSModel: n_neighbors dict keys must match TI "
                f"variables {ti.variables!r}. Missing: {sorted(missing)}, "
                f"extra: {sorted(extra)}."
            )
        for k, v in value.items():
            if int(v) < 1:
                raise ValueError(
                    f"MPSModel: n_neighbors[{k!r}] must be >= 1, got {v!r}"
                )
        return {k: int(v) for k, v in value.items()}
    else:
        if int(value) < 1:
            raise ValueError(
                f"MPSModel: n_neighbors must be >= 1, got {value!r}"
            )
        return int(value)


class MPSModel:
    """MPS configuration: training image and search algorithm parameters.

    Analogous to ``CovModel``: holds the training image and the search
    hyper-parameters. Carries no seed, thread count, or backend selector —
    those are call-time concerns on :class:`~gstools.mps.DirectSampling`.

    Parameters
    ----------
    ti : :any:`TrainingImage`
        Training image (univariate or multivariate).
    n_neighbors : :class:`int` or :class:`dict`, optional
        Maximum neighbours per node. A dict maps variable name to int for
        multivariate TIs; an int broadcasts to all variables. Default: 32.
    scan_fraction : :class:`float`, optional
        Fraction of the TI to scan per node (capped at the valid search
        window). Must be in (0, 1]. Default: 1.0.
    threshold : :class:`float`, optional
        Distance threshold for early acceptance. 0.0 → DSBC mode. Default: 0.0.
    cond_weight : :class:`float`, optional
        Weight multiplier for conditioning nodes in distance. Default: 1.0.
    boundary : :class:`str`, optional
        Search-window strategy: ``"strict"`` (default) or ``"partial"``.
    max_radius : :class:`float` or :any:`None`, optional
        Exclude SG neighbours beyond this Euclidean distance from the data
        event. ``None`` → no limit (default).
    """

    def __init__(
        self,
        ti,
        n_neighbors=32,
        scan_fraction=1.0,
        threshold=0.0,
        cond_weight=1.0,
        boundary="strict",
        max_radius=None,
    ):
        if not isinstance(ti, TrainingImage):
            raise TypeError(
                f"MPSModel: ti must be a TrainingImage, got {type(ti)!r}"
            )
        self._ti = ti
        self._n_neighbors = _validate_n_neighbors(n_neighbors, self._ti)
        _validate_variation_n_neighbors(self._n_neighbors, self._ti)
        self._scan_fraction = _validate_scan_fraction(scan_fraction)
        self._threshold = _validate_threshold(threshold)
        self._cond_weight = float(cond_weight)
        self._boundary = _validate_boundary(boundary)
        self._max_radius = _validate_max_radius(max_radius)

    @property
    def ti(self):
        """:any:`TrainingImage`: The training image."""
        return self._ti

    @property
    def n_neighbors(self):
        """:class:`int` or :class:`dict`: Maximum number of neighbours in the data event."""
        return self._n_neighbors

    @n_neighbors.setter
    def n_neighbors(self, value):
        validated = _validate_n_neighbors(value, self._ti)
        _validate_variation_n_neighbors(validated, self._ti)
        self._n_neighbors = validated

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
    def cond_weight(self):
        """:class:`float`: Weight multiplier for conditioning nodes."""
        return self._cond_weight

    @cond_weight.setter
    def cond_weight(self, value):
        self._cond_weight = float(value)

    @property
    def boundary(self):
        """:class:`str`: Search-window boundary strategy (``"strict"`` or ``"partial"``)."""
        return self._boundary

    @boundary.setter
    def boundary(self, value):
        self._boundary = _validate_boundary(value)

    @property
    def max_radius(self):
        """:class:`float` or :any:`None`: Maximum neighbour radius, or ``None`` for no limit."""
        return self._max_radius

    @max_radius.setter
    def max_radius(self, value):
        self._max_radius = _validate_max_radius(value)

    def __repr__(self):
        args = [repr(self._ti)]
        defaults = dict(
            n_neighbors=32,
            scan_fraction=1.0,
            threshold=0.0,
            cond_weight=1.0,
            boundary="strict",
            max_radius=None,
        )
        for name, default in defaults.items():
            val = getattr(self, f"_{name}")
            if val != default:
                args.append(f"{name}={val!r}")
        return f"MPSModel({', '.join(args)})"
