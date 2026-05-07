#!/usr/bin/env python
"""Print Rust-vs-Cython speedups from local ASV result files.

The summary is optional. ASV itself remains the source of truth for benchmark
storage and visualization.

Usage:
    python benchmarks/tools/asv_speedup_summary.py
    python benchmarks/tools/asv_speedup_summary.py --results-dir .asv/results
    python benchmarks/tools/asv_speedup_summary.py --include-legacy

Speedup is calculated as:
    cython_fallback_time / rust_core_time

Values greater than 1.0 mean Rust was faster on the same machine, commit,
environment, benchmark, and non-backend parameter combination.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path


BACKENDS = ("cython_fallback", "rust_core")
LEGACY_BENCHMARKS = {
    "time_srf",
    "peakmem_srf",
    "time_variogram",
    "peakmem_variogram",
    "time_krige",
    "peakmem_krige",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        default=".asv/results",
        type=Path,
        help="Path to the ASV results directory.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Include non-time benchmarks as ratios too.",
    )
    parser.add_argument(
        "--include-legacy",
        action="store_true",
        help="Include removed BackendBenchmarks rows from older saved results.",
    )
    return parser.parse_args()


def iter_result_files(results_dir):
    for path in sorted(results_dir.glob("**/*.json")):
        if path.name in {"benchmarks.json", "machine.json"}:
            continue
        yield path


def load_json(path):
    try:
        with path.open(encoding="utf8") as handle:
            return json.load(handle)
    except json.JSONDecodeError:
        return None


def result_entry(raw_result, result_columns):
    if isinstance(raw_result, dict):
        return raw_result
    if isinstance(raw_result, list) and result_columns:
        return dict(zip(result_columns, raw_result))
    return {"result": raw_result, "params": []}


def is_number(value):
    return isinstance(value, (int, float)) and not math.isnan(value)


def flatten_values(values):
    if isinstance(values, list):
        for value in values:
            yield from flatten_values(value)
        return
    yield values


def backend_values(entry):
    result = entry.get("result")
    params = entry.get("params") or []
    if not isinstance(result, list) or not params:
        return {}

    values = {}
    combinations = itertools.product(*params)
    for combo, value in zip(combinations, flatten_values(result)):
        if not is_number(value):
            continue
        combo_values = [str(item).strip("'\"") for item in combo]
        for backend in BACKENDS:
            if backend in combo_values:
                values[backend] = float(value)
    return values


def backend_rows(entry):
    result = entry.get("result")
    params = entry.get("params") or []
    if not isinstance(result, list) or not params:
        return []

    rows = []
    combinations = itertools.product(*params)
    for combo, value in zip(combinations, flatten_values(result)):
        if not is_number(value):
            continue
        combo_values = [str(item).strip("'\"") for item in combo]
        backend = next(
            (candidate for candidate in BACKENDS if candidate in combo_values),
            None,
        )
        if backend is None:
            continue
        case_values = [item for item in combo_values if item not in BACKENDS]
        rows.append(
            {
                "backend": backend,
                "case": "/".join(case_values) if case_values else "-",
                "value": float(value),
            }
        )
    return rows


def short_benchmark_name(name):
    return name.rsplit(".", maxsplit=1)[-1]


def collect_speedups(results_dir, include_all, include_legacy):
    rows = []
    for path in iter_result_files(results_dir):
        data = load_json(path)
        if not data:
            continue
        result_columns = data.get("result_columns", [])
        commit = data.get("commit_hash", "unknown")[:8]
        env_name = data.get("env_name", path.stem)
        results = data.get("results", {})
        for benchmark, raw_result in results.items():
            benchmark_name = short_benchmark_name(benchmark)
            if not include_legacy and benchmark_name in LEGACY_BENCHMARKS:
                continue
            if not include_all and ".time_" not in benchmark:
                continue
            by_case = {}
            for row in backend_rows(result_entry(raw_result, result_columns)):
                by_case.setdefault(row["case"], {})[row["backend"]] = row[
                    "value"
                ]
            for case, values in by_case.items():
                cython = values.get("cython_fallback")
                rust = values.get("rust_core")
                if not is_number(cython) or not is_number(rust) or rust == 0:
                    continue
                rows.append(
                    {
                        "commit": commit,
                        "env": env_name,
                        "benchmark": benchmark_name,
                        "case": case,
                        "cython": cython,
                        "rust": rust,
                        "speedup": cython / rust,
                    }
                )
    return rows


def print_table(rows):
    if not rows:
        print("No matching Rust-vs-Cython ASV results found.")
        return

    headers = [
        "commit",
        "env",
        "benchmark",
        "case",
        "cython",
        "rust",
        "speedup",
    ]
    table = [
        [
            row["commit"],
            row["env"],
            row["benchmark"],
            row["case"],
            f"{row['cython']:.6g}",
            f"{row['rust']:.6g}",
            f"{row['speedup']:.3f}x",
        ]
        for row in rows
    ]
    widths = [
        max(len(str(item)) for item in column)
        for column in zip(headers, *table)
    ]

    def fmt(row):
        return "  ".join(
            str(item).ljust(width) for item, width in zip(row, widths)
        )

    print(fmt(headers))
    print(fmt(["-" * width for width in widths]))
    for row in table:
        print(fmt(row))


def main():
    args = parse_args()
    rows = collect_speedups(args.results_dir, args.all, args.include_legacy)
    print_table(rows)


if __name__ == "__main__":
    main()
