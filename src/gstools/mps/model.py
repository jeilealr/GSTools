"""MPS configuration object — training image and search parameters."""

import warnings

import numpy as np

from gstools.mps.training_image import TrainingImage
from gstools.tools.geometric import no_of_angles

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
    cond_weight : :class:`float`, optional
        Weight multiplier for conditioning nodes in distance. Default: 1.0.
    boundary : :class:`str`, optional
        Search-window strategy: ``"strict"`` (default) or ``"partial"``.
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
        cond_weight=1.0,
        boundary="strict",
        rotation=None,
        scale=None,
    ):
        if not isinstance(ti, TrainingImage):
            raise TypeError(
                f"MPSModel: ti must be a TrainingImage, got {type(ti)!r}"
            )
        self._ti = ti
        self._scan_fraction = _validate_scan_fraction(scan_fraction)
        self._threshold = _validate_threshold(threshold)
        self._cond_weight = float(cond_weight)
        self._boundary = _validate_boundary(boundary)
        dim = len(ti.shape)
        self._rotation = _validate_rotation(rotation, dim)
        self._scale = _validate_scale(scale, dim)

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

    def __repr__(self):
        args = [repr(self._ti)]
        defaults = dict(
            scan_fraction=1.0,
            threshold=0.0,
            cond_weight=1.0,
            boundary="strict",
        )
        for name, default in defaults.items():
            val = getattr(self, f"_{name}")
            if val != default:
                args.append(f"{name}={val!r}")
        return f"MPSModel({', '.join(args)})"
