#!/usr/bin/env python
"""Profile the representative GSTools benchmark workflows with cProfile.

This is a quick measurement helper. ASV remains the source of truth for saved
benchmark results, while this script helps identify the top cumulative Python
call sites before making algorithmic changes.

Usage:
    cd /path/to/MPS-Tools/GSTools
    ASV_ENV="$(ls -td .asv/env/* | head -n 1)"
    "$ASV_ENV/bin/python" benchmarks/tools/profile_benchmark_workflows.py --list
    "$ASV_ENV/bin/python" benchmarks/tools/profile_benchmark_workflows.py \
        --case variogram-sampled
    "$ASV_ENV/bin/python" benchmarks/tools/profile_benchmark_workflows.py \
        --case krige-large \
        --backend rust_core --threads threads_1 --limit 30
"""

from __future__ import annotations

import argparse
import cProfile
import pstats
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))


CASES = {
    "variogram-full": (
        "VariogramWorkflowBenchmarks",
        "time_variogram_estimate",
        ("full_900",),
    ),
    "variogram-sampled": (
        "VariogramWorkflowBenchmarks",
        "time_variogram_estimate",
        ("sampled_5000_to_1500",),
    ),
    "variogram-extra-large": (
        "VariogramWorkflowBenchmarks",
        "time_variogram_estimate",
        ("sampled_15000_to_4500",),
    ),
    "krige-small": (
        "KrigingWorkflowBenchmarks",
        "time_global_krige",
        ("small_30x500",),
    ),
    "krige-large": (
        "KrigingWorkflowBenchmarks",
        "time_global_krige",
        ("large_120x2000",),
    ),
    "krige-extra-large": (
        "KrigingWorkflowBenchmarks",
        "time_global_krige",
        ("extra_large_360x6000",),
    ),
    "srf-unstructured": (
        "RandomFieldWorkflowBenchmarks",
        "time_field_generation",
        ("srf_unstructured_randmeth",),
    ),
    "srf-structured": (
        "RandomFieldWorkflowBenchmarks",
        "time_field_generation",
        ("srf_structured_randmeth",),
    ),
    "srf-fourier": (
        "RandomFieldWorkflowBenchmarks",
        "time_field_generation",
        ("srf_structured_fourier",),
    ),
    "condsrf": (
        "RandomFieldWorkflowBenchmarks",
        "time_field_generation",
        ("condsrf_unstructured",),
    ),
}

THREAD_COUNTS = (
    "threads_1",
    "threads_2",
    "threads_4",
    "threads_8",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        default="all",
        choices=["all", *CASES],
        help="Workflow to profile. Defaults to all workflows.",
    )
    parser.add_argument(
        "--repeat",
        default=1,
        type=int,
        help="Number of times to run each selected workflow.",
    )
    parser.add_argument(
        "--limit",
        default=25,
        type=int,
        help="Number of cProfile rows to print per workflow.",
    )
    parser.add_argument(
        "--sort",
        default="cumtime",
        choices=["cumtime", "tottime", "calls"],
        help="pstats sort key.",
    )
    parser.add_argument(
        "--backend",
        default="rust_core",
        choices=["cython_fallback", "rust_core"],
        help="Backend label to force while profiling.",
    )
    parser.add_argument(
        "--threads",
        default="threads_1",
        choices=THREAD_COUNTS,
        help="GSTools thread count label.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available workflow cases and exit.",
    )
    return parser.parse_args()


def iter_selected(case):
    if case == "all":
        yield from CASES.items()
        return
    yield case, CASES[case]


def load_suite_class(class_name):
    try:
        from benchmarks import benchmark_backends
    except ModuleNotFoundError as err:
        print(
            "Could not import GSTools benchmark dependencies. Activate the "
            "GSTools benchmark environment, run this script with an ASV env "
            "Python from .asv/env/<env-id>/bin/python, or install the project "
            f"dependencies first. Original error: {err}",
            file=sys.stderr,
        )
        raise SystemExit(1) from err
    return getattr(benchmark_backends, class_name)


def run_case(
    name,
    class_name,
    method_name,
    params,
    repeat,
    limit,
    sort,
    backend,
    threads,
):
    suite_cls = load_suite_class(class_name)
    suite = suite_cls()
    data = suite.setup_cache()
    method = getattr(suite, method_name)

    profiler = cProfile.Profile()
    profiler.enable()
    for _ in range(repeat):
        method(data, *params, backend, threads)
    profiler.disable()

    print(f"\n== {name} [{backend}, {threads}] ==")
    stats = pstats.Stats(profiler, stream=sys.stdout)
    stats.strip_dirs().sort_stats(sort).print_stats(limit)


def main():
    args = parse_args()
    if args.list:
        for name in CASES:
            print(name)
        return

    for name, (suite_cls, method_name, params) in iter_selected(args.case):
        run_case(
            name,
            suite_cls,
            method_name,
            params,
            args.repeat,
            args.limit,
            args.sort,
            args.backend,
            args.threads,
        )


if __name__ == "__main__":
    main()
