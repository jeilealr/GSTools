"""Backend benchmarks for GSTools.

Usage:
    cd /my/path/to/GSTools
    conda install -c conda-forge asv
    asv machine
    asv run --quick
    asv run HEAD^!
    asv run
    asv publish
    asv preview
    asv compare HEAD~1 HEAD

Backend speedup should be interpreted as:
    speedup = cython_fallback_time / rust_core_time

Values greater than 1.0 mean the Rust backend is faster on the same machine
for the same benchmark and commit.
"""

from __future__ import annotations

import contextlib

import numpy as np

import gstools as gs


BACKENDS = ("cython_fallback", "rust_core")


@contextlib.contextmanager
def gstools_backend(use_core):
    """Temporarily force either gstools-core or the Cython fallback."""
    previous = (gs.config._GSTOOLS_CORE_AVAIL, gs.config.USE_GSTOOLS_CORE)
    try:
        if use_core:
            if not previous[0]:
                raise NotImplementedError("gstools_core is not available")
            gs.config._GSTOOLS_CORE_AVAIL = True
            gs.config.USE_GSTOOLS_CORE = True
        else:
            gs.config._GSTOOLS_CORE_AVAIL = False
            gs.config.USE_GSTOOLS_CORE = False
        yield
    finally:
        gs.config._GSTOOLS_CORE_AVAIL, gs.config.USE_GSTOOLS_CORE = previous


def _use_core(backend):
    if backend == "rust_core":
        return True
    if backend == "cython_fallback":
        return False
    raise ValueError(f"Unknown backend: {backend}")


class BackendBenchmarks:
    """Runtime and peak-memory benchmarks for backend-dispatched operations."""

    params = BACKENDS
    param_names = ["backend"]

    def setup_cache(self):
        """Create deterministic data once per benchmark environment."""
        srf_x = np.random.RandomState(20220425).rand(2000) * 100.0
        srf_y = np.random.RandomState(20220426).rand(2000) * 100.0

        vario_x = np.random.RandomState(20220427).rand(900) * 100.0
        vario_y = np.random.RandomState(20220428).rand(900) * 100.0
        vario_field = np.sin(vario_x / 10.0) + np.cos(vario_y / 15.0)
        vario_bins = np.linspace(0.0, 60.0, 16)

        rng = np.random.RandomState(20220429)
        cond_x = rng.rand(40) * 50.0
        cond_y = rng.rand(40) * 50.0
        cond_val = np.sin(cond_x / 8.0) + np.cos(cond_y / 9.0)
        target_pos = (rng.rand(1000) * 50.0, rng.rand(1000) * 50.0)

        return {
            "srf": (srf_x, srf_y),
            "variogram": ((vario_x, vario_y), vario_field, vario_bins),
            "krige": ((cond_x, cond_y), cond_val, target_pos),
        }

    def setup(self, data, backend):
        """Skip only the Rust parameter when gstools-core is unavailable."""
        if backend == "rust_core" and not gs.config._GSTOOLS_CORE_AVAIL:
            raise NotImplementedError("gstools_core is not available")

    def time_srf(self, data, backend):
        with gstools_backend(_use_core(backend)):
            self._run_srf(data)

    def peakmem_srf(self, data, backend):
        with gstools_backend(_use_core(backend)):
            self._run_srf(data)

    def time_variogram(self, data, backend):
        with gstools_backend(_use_core(backend)):
            self._run_variogram(data)

    def peakmem_variogram(self, data, backend):
        with gstools_backend(_use_core(backend)):
            self._run_variogram(data)

    def time_krige(self, data, backend):
        with gstools_backend(_use_core(backend)):
            self._run_krige(data)

    def peakmem_krige(self, data, backend):
        with gstools_backend(_use_core(backend)):
            self._run_krige(data)

    def _run_srf(self, data):
        x, y = data["srf"]
        model = gs.Exponential(dim=2, var=2.0, len_scale=8.0)
        srf = gs.SRF(model, mean=1.0, seed=20220425, mode_no=512)
        return srf((x, y), mesh_type="unstructured")

    def _run_variogram(self, data):
        pos, field, bins = data["variogram"]
        return gs.vario_estimate(
            pos,
            field,
            bins,
            mesh_type="unstructured",
            return_counts=True,
        )

    def _run_krige(self, data):
        cond_pos, cond_val, target_pos = data["krige"]
        model = gs.Exponential(dim=2, var=1.5, len_scale=12.0, nugget=0.05)
        krige = gs.Krige(
            model,
            cond_pos,
            cond_val,
            exact=False,
            cond_err=0.05,
        )
        return krige(
            target_pos,
            mesh_type="unstructured",
            return_var=True,
            store=False,
        )
