#!/usr/bin/env python
"""Check whether GSTools-Cython detects OpenMP parallel support.

This script verifies the active Python environment. Use it with the editable
development environment or with an ASV-created environment.

Examples:
    python benchmarks/tools/check_cython_openmp.py
    python benchmarks/tools/check_cython_openmp.py --fail-if-no-openmp
    python benchmarks/tools/check_cython_openmp.py --verbose
    .asv/env/<hash>/bin/python3 benchmarks/tools/check_cython_openmp.py
"""

from __future__ import annotations

import argparse
import importlib
import sys


MODULES = {
    "variogram": "gstools_cython.variogram",
    "field": "gstools_cython.field",
    "krige": "gstools_cython.krige",
}
EXPLICIT_THREAD_COUNTS = (1, 2, 4, 8, 16)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fail-if-no-openmp",
        action="store_true",
        help="Exit with status 1 if OpenMP thread detection reports <= 1.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-module default and explicit thread-count values.",
    )
    return parser.parse_args()


def package_version(package_name):
    try:
        package = importlib.import_module(package_name)
    except ModuleNotFoundError:
        return "not installed"
    return getattr(package, "__version__", "unknown")


def check_module(label, module_name):
    module = importlib.import_module(module_name)
    default_threads = module.set_num_threads(None)
    explicit = {
        count: module.set_num_threads(count)
        for count in EXPLICIT_THREAD_COUNTS
    }
    return label, default_threads, explicit


def main():
    args = parse_args()

    print(f"python: {sys.executable}")
    print(f"gstools: {package_version('gstools')}")
    print(f"gstools_cython: {package_version('gstools_cython')}")
    print(f"gstools_core: {package_version('gstools_core')}")
    if args.verbose:
        print(
            "OpenMP evidence: default None should be >1. "
            "Explicit values only prove the wrapper accepts the requested count."
        )

    default_values = []
    for label, module_name in MODULES.items():
        try:
            label, default_threads, explicit = check_module(label, module_name)
        except ModuleNotFoundError as err:
            print(f"OpenMP check: FAIL. Missing module: {err.name}")
            return 1
        default_values.append(default_threads)
        if args.verbose:
            explicit_text = ", ".join(
                f"{request}->{actual}" for request, actual in explicit.items()
            )
            print(f"{label} default None -> {default_threads}")
            print(f"{label} explicit -> {explicit_text}")

    if min(default_values) > 1:
        print("OpenMP check: PASS")
        return 0

    print(
        "OpenMP check: FAIL. GSTools-Cython reports one default thread. "
        "Explicit thread values may be accepted by the wrapper, but this does "
        "not prove that the compiled extension is using OpenMP."
    )
    return 1 if args.fail_if_no_openmp else 0


if __name__ == "__main__":
    raise SystemExit(main())
