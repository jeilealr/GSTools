"""Workflow benchmarks for GSTools backends.

Usage:
    cd /path/to/MPS-Tools/GSTools
    python -m pip install -e ".[benchmark]"
    asv machine --yes
    asv run --quick --show-stderr --bench benchmark_backends
    asv run HEAD^! --bench benchmark_backends
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
VARIOGRAM_CASES = (
    "full_900",
    "sampled_5000_to_1500",
    "sampled_15000_to_4500",
)
KRIGE_CASES = ("small_30x500", "large_120x2000", "extra_large_360x6000")
FIELD_CASES = (
    "srf_unstructured_randmeth",
    "srf_structured_randmeth",
    "srf_structured_fourier",
    "condsrf_unstructured",
)


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


def _random_points(seed, count, scale):
    rng = np.random.RandomState(seed)
    return rng.rand(count) * scale, rng.rand(count) * scale


def _smooth_field(x, y):
    return np.sin(x / 10.0) + np.cos(y / 15.0)


def _make_variogram_data(seed, count, scale=100.0):
    x, y = _random_points(seed, count, scale)
    field = _smooth_field(x, y)
    bins = np.linspace(0.0, scale * 0.6, 16)
    return (x, y), field, bins


def _make_krige_data(seed, cond_count, target_count, scale=50.0):
    rng = np.random.RandomState(seed)
    cond_x = rng.rand(cond_count) * scale
    cond_y = rng.rand(cond_count) * scale
    cond_val = _smooth_field(cond_x, cond_y)
    target_pos = (
        rng.rand(target_count) * scale,
        rng.rand(target_count) * scale,
    )
    return (cond_x, cond_y), cond_val, target_pos


class VariogramWorkflowBenchmarks:
    """Variogram workflow benchmarks by case and backend."""

    params = [VARIOGRAM_CASES, BACKENDS]
    param_names = ["case", "backend"]

    def setup_cache(self):
        return {
            "full_900": _make_variogram_data(20220501, 900),
            "sampled_5000_to_1500": _make_variogram_data(20220502, 5000),
            "sampled_15000_to_4500": _make_variogram_data(20220503, 15000),
        }

    def setup(self, data, case, backend):
        if backend == "rust_core" and not gs.config._GSTOOLS_CORE_AVAIL:
            raise NotImplementedError("gstools_core is not available")

    def time_variogram_estimate(self, data, case, backend):
        with gstools_backend(_use_core(backend)):
            self._run_variogram(data, case)

    def peakmem_variogram_estimate(self, data, case, backend):
        with gstools_backend(_use_core(backend)):
            self._run_variogram(data, case)

    def _run_variogram(self, data, case):
        pos, field, bins = data[case]
        kwargs = {}
        if case == "sampled_5000_to_1500":
            kwargs = {"sampling_size": 1500, "sampling_seed": 20220504}
        if case == "sampled_15000_to_4500":
            kwargs = {"sampling_size": 4500, "sampling_seed": 20220505}
        return gs.vario_estimate(
            pos,
            field,
            bins,
            mesh_type="unstructured",
            return_counts=True,
            **kwargs,
        )


class KrigingWorkflowBenchmarks:
    """Global kriging workflow benchmarks by case and backend."""

    params = [KRIGE_CASES, BACKENDS]
    param_names = ["case", "backend"]

    def setup_cache(self):
        return {
            "small_30x500": _make_krige_data(20220506, 30, 500),
            "large_120x2000": _make_krige_data(20220507, 120, 2000),
            "extra_large_360x6000": _make_krige_data(20220508, 360, 6000),
        }

    def setup(self, data, case, backend):
        if backend == "rust_core" and not gs.config._GSTOOLS_CORE_AVAIL:
            raise NotImplementedError("gstools_core is not available")

    def time_global_krige(self, data, case, backend):
        with gstools_backend(_use_core(backend)):
            self._run_krige(data, case)

    def peakmem_global_krige(self, data, case, backend):
        with gstools_backend(_use_core(backend)):
            self._run_krige(data, case)

    def _run_krige(self, data, case):
        cond_pos, cond_val, target_pos = data[case]
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


class RandomFieldWorkflowBenchmarks:
    """SRF and CondSRF workflow benchmarks by case and backend."""

    params = [FIELD_CASES, BACKENDS]
    param_names = ["case", "backend"]

    def setup_cache(self):
        return {
            "unstructured_pos": _random_points(20220509, 2000, 100.0),
            "structured_pos": (
                np.linspace(0.0, 100.0, 64),
                np.linspace(0.0, 100.0, 64),
            ),
            "condsrf": _make_krige_data(20220510, 40, 1000),
        }

    def setup(self, data, case, backend):
        if backend == "rust_core" and not gs.config._GSTOOLS_CORE_AVAIL:
            raise NotImplementedError("gstools_core is not available")

    def time_field_generation(self, data, case, backend):
        with gstools_backend(_use_core(backend)):
            self._run_field(data, case)

    def peakmem_field_generation(self, data, case, backend):
        with gstools_backend(_use_core(backend)):
            self._run_field(data, case)

    def _run_field(self, data, case):
        if case == "srf_unstructured_randmeth":
            return self._run_srf_unstructured(data)
        if case == "srf_structured_randmeth":
            return self._run_srf_structured(data)
        if case == "srf_structured_fourier":
            return self._run_srf_fourier(data)
        if case == "condsrf_unstructured":
            return self._run_condsrf(data)
        raise ValueError(f"Unknown field benchmark case: {case}")

    def _run_srf_unstructured(self, data):
        model = gs.Exponential(dim=2, var=2.0, len_scale=8.0)
        srf = gs.SRF(model, mean=1.0, seed=20220508, mode_no=512)
        return srf(data["unstructured_pos"], mesh_type="unstructured")

    def _run_srf_structured(self, data):
        model = gs.Exponential(dim=2, var=2.0, len_scale=8.0)
        srf = gs.SRF(model, mean=1.0, seed=20220509, mode_no=512)
        return srf(data["structured_pos"], mesh_type="structured")

    def _run_srf_fourier(self, data):
        model = gs.Gaussian(dim=2, var=2.0, len_scale=30.0)
        srf = gs.SRF(
            model,
            generator="Fourier",
            period=[100.0, 100.0],
            mode_no=[32, 32],
            seed=20220510,
        )
        return srf(data["structured_pos"], mesh_type="structured")

    def _run_condsrf(self, data):
        cond_pos, cond_val, target_pos = data["condsrf"]
        model = gs.Exponential(dim=2, var=1.5, len_scale=12.0, nugget=0.05)
        krige = gs.Krige(
            model,
            cond_pos,
            cond_val,
            exact=False,
            cond_err=0.05,
        )
        cond_srf = gs.CondSRF(krige, seed=20220511, mode_no=512)
        return cond_srf(
            target_pos,
            mesh_type="unstructured",
            seed=20220512,
            store=False,
            krige_store=False,
        )
