#!/usr/bin/env python
"""Check that GSTools benchmark backends are ready for parallel runs.

This is a fast CI probe. It verifies that GSTools-Cython reports OpenMP support
and that the Rust backend can run a small workflow while GSTools is configured
with more than one thread.
"""

from __future__ import annotations

import argparse
import importlib
import sys

import numpy as np
from check_cython_openmp import MODULES, check_module, package_version


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--threads",
        default=2,
        type=int,
        help="Thread count to request for the Rust backend smoke workflow.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-module Cython OpenMP thread details.",
    )
    return parser.parse_args()


def check_cython_openmp(verbose=False):
    default_values = []
    for label, module_name in MODULES.items():
        try:
            label, default_threads, explicit = check_module(label, module_name)
        except ModuleNotFoundError as err:
            print(f"Cython OpenMP readiness: FAIL. Missing module: {err.name}")
            return False
        default_values.append(default_threads)
        if verbose:
            explicit_text = ", ".join(
                f"{request}->{actual}" for request, actual in explicit.items()
            )
            print(f"{label} default None -> {default_threads}")
            print(f"{label} explicit -> {explicit_text}")
        if explicit.get(2) != 2:
            print(
                "Cython OpenMP readiness: FAIL. "
                f"{label} did not accept an explicit 2-thread request."
            )
            return False

    if min(default_values) <= 1:
        print(
            "Cython OpenMP readiness: FAIL. GSTools-Cython reports one "
            "default thread."
        )
        return False

    print("Cython OpenMP readiness: PASS")
    return True


def check_rust_backend(threads):
    try:
        import gstools as gs
    except ModuleNotFoundError as err:
        print(f"Rust backend readiness: FAIL. Missing module: {err.name}")
        return False

    try:
        importlib.import_module("gstools_core")
    except ModuleNotFoundError as err:
        print(f"Rust backend readiness: FAIL. Missing module: {err.name}")
        return False

    if not gs.config._GSTOOLS_CORE_AVAIL:
        print(
            "Rust backend readiness: FAIL. GSTools did not detect gstools_core."
        )
        return False

    previous = (
        gs.config._GSTOOLS_CORE_AVAIL,
        gs.config.USE_GSTOOLS_CORE,
        gs.config.NUM_THREADS,
    )
    try:
        gs.config._GSTOOLS_CORE_AVAIL = True
        gs.config.USE_GSTOOLS_CORE = True
        gs.config.NUM_THREADS = threads

        x = np.linspace(0.0, 10.0, 12)
        y = np.linspace(0.0, 5.0, 12)
        field = np.sin(x) + np.cos(y)
        bins = np.linspace(0.0, 8.0, 6)
        gs.vario_estimate(
            (x, y),
            field,
            bins,
            mesh_type="unstructured",
            return_counts=True,
        )
    finally:
        (
            gs.config._GSTOOLS_CORE_AVAIL,
            gs.config.USE_GSTOOLS_CORE,
            gs.config.NUM_THREADS,
        ) = previous

    print(f"Rust backend readiness: PASS with NUM_THREADS={threads}")
    return True


def main():
    args = parse_args()
    print(f"python: {sys.executable}")
    print(f"gstools: {package_version('gstools')}")
    print(f"gstools_cython: {package_version('gstools_cython')}")
    print(f"gstools_core: {package_version('gstools_core')}")

    cython_ready = check_cython_openmp(verbose=args.verbose)
    rust_ready = check_rust_backend(args.threads)
    return 0 if cython_ready and rust_ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
