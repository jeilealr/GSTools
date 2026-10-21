"""Central gstools_core import + MPS capability check.

MPS Direct Sampling is Rust-only: it requires ``gstools_core`` (>= 1.4.0, the
release that first shipped the MPS kernels). This module hard-imports the core,
validates its version *and* that every required MPS export is present, then
re-exports the kernels. A single, explicit error is raised for both failure
modes — core missing, or an older core without the MPS exports.

It is imported lazily (only when ``gstools.mps`` is first used), so plain
``import gstools`` and the SRF/kriging paths keep working without the Rust core;
the requirement applies to MPS only.
"""

_REQUIRED_EXPORTS = (
    "mps_simulate",
    "mps_dist_block_cat",
    "mps_dist_block_cat_masked",
    "mps_dist_block_cat_rayon",
    "mps_dist_block_l1",
    "mps_dist_block_l1_masked",
    "mps_dist_block_l2",
    "mps_dist_block_l2_masked",
    "mps_dist_block_lp",
    "mps_dist_block_lp_masked",
    "mps_dist_block_variation",
    "mps_dist_block_variation_masked",
    "mps_scan_node",
    "mps_scan_node_cat",
    "mps_cleanup_pass",
    "mps_api_version",
)
_MIN_VERSION = (1, 4, 0)
# MPS native ABI revision this Python layer requires. The crate version can
# stay 1.4.0 across an ABI change, so version alone is insufficient; the Core
# exports ``mps_api_version()`` and we gate on it. Bump in lockstep with the
# Rust ``mps_api_version_py`` whenever an MPS entry-point signature changes.
_REQUIRED_MPS_API = 7
_MSG = "GSTools MPS requires gstools-core>=1.4.0 with MPS support"


def _version_tuple(version):
    """Best-effort ``"1.4.0"`` -> ``(1, 4, 0)`` (ignores pre-release suffixes)."""
    parts = []
    for chunk in str(version).split("."):
        digits = "".join(ch for ch in chunk if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts[:3])


try:
    import gstools_core as _gstools_core
except ImportError as exc:  # core not installed at all
    raise ImportError(f"{_MSG}; gstools_core is not installed.") from exc

_missing = [
    name for name in _REQUIRED_EXPORTS if not hasattr(_gstools_core, name)
]
_version = getattr(_gstools_core, "__version__", "0")
if _missing or _version_tuple(_version) < _MIN_VERSION:
    _detail = (
        f" (found gstools_core {_version}, missing MPS exports: "
        f"{', '.join(_missing)})"
        if _missing
        else f" (found gstools_core {_version})"
    )
    raise ImportError(f"{_MSG}{_detail}.")

# ABI-revision gate: a same-versioned Core built before the current MPS
# entry-point signatures must be rejected with an actionable message rather
# than failing later with an opaque argument/type error.
_found_api = int(_gstools_core.mps_api_version())
if _found_api < _REQUIRED_MPS_API:
    raise ImportError(
        f"{_MSG}: this GSTools needs MPS native ABI revision "
        f">= {_REQUIRED_MPS_API}, but the installed gstools_core {_version} "
        f"reports revision {_found_api}. Rebuild gstools-core from the matching "
        "source (maturin build --release) and reinstall it."
    )

# Re-export the validated kernels so the MPS modules import them from here.
mps_simulate = _gstools_core.mps_simulate
mps_dist_block_cat = _gstools_core.mps_dist_block_cat
mps_dist_block_cat_masked = _gstools_core.mps_dist_block_cat_masked
mps_dist_block_cat_rayon = _gstools_core.mps_dist_block_cat_rayon
mps_dist_block_l1 = _gstools_core.mps_dist_block_l1
mps_dist_block_l1_masked = _gstools_core.mps_dist_block_l1_masked
mps_dist_block_l2 = _gstools_core.mps_dist_block_l2
mps_dist_block_l2_masked = _gstools_core.mps_dist_block_l2_masked
mps_dist_block_lp = _gstools_core.mps_dist_block_lp
mps_dist_block_lp_masked = _gstools_core.mps_dist_block_lp_masked
mps_dist_block_variation = _gstools_core.mps_dist_block_variation
mps_dist_block_variation_masked = _gstools_core.mps_dist_block_variation_masked
mps_scan_node = _gstools_core.mps_scan_node
mps_scan_node_cat = _gstools_core.mps_scan_node_cat
mps_cleanup_pass = _gstools_core.mps_cleanup_pass
mps_api_version = _gstools_core.mps_api_version
