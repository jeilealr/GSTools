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
)
_MIN_VERSION = (1, 4, 0)
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
