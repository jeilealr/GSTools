"""Small shared validators for DS counts and float64 representation."""

import numbers

import numpy as np


def exact_count(value, *, minimum, context):
    """Accept integers, bools and integral real floats without truncation.

    Integer types never pass through float64. Integral floats and bools remain
    supported for compatibility; strings and non-integral/nonfinite reals do not.
    """
    if isinstance(value, (numbers.Integral, np.bool_)):
        result = int(value)
    elif isinstance(value, numbers.Real) and np.isfinite(value):
        result = int(value)
        if value != result:
            raise ValueError(f"{context} must be an integer, got {value!r}")
    else:
        raise ValueError(f"{context} must be an integer, got {value!r}")
    if result < minimum:
        raise ValueError(f"{context} must be >= {minimum}, got {value!r}")
    return result


def _validate_float64_integers(values, categorical, context):
    """Reject integer values whose conversion to the engine's float64 loses bits.

    Python integers make the comparison safe at signed/unsigned endpoints.
    Exactly representable large powers of two remain supported. This checks
    representation, not whether a value is a valid penalty matrix index.
    """
    array = np.asarray(values)
    if not np.issubdtype(array.dtype, np.integer) or not array.size:
        return
    if int(array.min()) >= -(2**53) and int(array.max()) <= 2**53:
        return
    for value in np.unique(array):
        integer = int(value)
        if int(float(integer)) != integer:
            remedy = (
                "recode categorical labels to small integer codes"
                if categorical
                else "center/rescale continuous values before float64 conversion"
            )
            raise ValueError(
                f"{context}: integer value {integer} is not exactly representable "
                f"in the float64 DS engine; {remedy}."
            )
