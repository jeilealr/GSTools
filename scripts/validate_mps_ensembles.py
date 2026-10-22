"""Reproducible synthetic DS diagnostic ensembles, not release equivalence proof.

Outputs a predeclared design before simulation, per-realization metrics, normal
approximation Monte Carlo intervals over realizations, representative fields,
source hashes, and plots. Diagnostic tolerances are not scientific acceptance
criteria. Run with --out NEW_DIRECTORY; existing study outputs are not replaced.
"""

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import label

import gstools as gs
from gstools.mps import DirectSampling, MPSModel, TrainingImage, Variable, Zone

CONTRACT = "ds-seedsequence-mt19937-v1"
CASES = [
    "channels",
    "penalty",
    "l1",
    "l2",
    "variation",
    "mixed",
    "zones",
    "cleanup",
]


def construct(case):
    x, y = np.indices((32, 36))
    channels = (np.abs((x - 3 * np.sin(y / 6)) % 12 - 6) < 2).astype(float)
    continuous = np.sin(x / 4) + np.cos(y / 6)
    kwargs = dict(n_neighbors=8, max_radius=4)
    if case in ("channels", "cleanup", "zones"):
        ti = TrainingImage(channels, **kwargs)
    elif case == "penalty":
        ti = TrainingImage(
            Variable(
                None,
                (x // 4 + y // 7) % 3,
                penalty_matrix=[[0, 0.2, 1], [0.7, 0, 0.4], [1, 0.8, 0]],
                **kwargs,
            )
        )
    elif case == "mixed":
        ti = TrainingImage(
            [
                Variable("facies", channels, **kwargs),
                Variable(
                    "value",
                    channels * 3 + 0.1 * continuous,
                    categorical=False,
                    distance="l2",
                    **kwargs,
                ),
            ]
        )
    else:
        ti = TrainingImage(
            continuous,
            categorical=False,
            distance="variation" if case == "variation" else case,
            **kwargs,
        )
    config = dict(
        scan_fraction=0.2,
        threshold=0.0,
        scan_path="random",
        boundary="partial",
    )
    if case == "zones":
        other = channels[:27, :29].T.copy()
        other[::5, ::7] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            zone = TrainingImage(other, **kwargs)
        mask = np.indices((24, 24))[0] >= 12
        config["zones"] = [Zone(zone, where=mask)]
    if case == "cleanup":
        mask = np.indices((24, 24))[0] >= 12
        config.update(post_processing=1, post_processing_mask=mask)
    ds = DirectSampling(MPSModel(ti, **config))
    if case == "variation":
        ds.set_condition([[0, 23], [0, 23]], [10.0, 12.0])
    else:
        value = {"facies": [1.0], "value": [3.0]} if case == "mixed" else [0.0]
        ds.set_condition([[0], [0]], value)
    return ds, ti


def metrics(field, categorical, categories=()):
    """Per-realization metrics with a FIXED category schema (finding F2).

    ``categories`` is the declared category universe for this variable (from the
    TI/zone TIs, fixed before sampling). Every category emits the same keys in
    every realization — an absent category reports ``fraction = 0`` and empty
    connectivity rather than vanishing from the row, so the summary cannot
    ``KeyError`` on, or silently drop, a category that a later realization omits.
    """
    out = dict(
        mean=float(np.mean(field)),
        variance=float(np.var(field)),
        q10=float(np.quantile(field, 0.1)),
        q90=float(np.quantile(field, 0.9)),
        semivar_x=float(0.5 * np.mean(np.diff(field, axis=0) ** 2)),
        semivar_y=float(0.5 * np.mean(np.diff(field, axis=1) ** 2)),
    )
    if categorical:
        for category in categories:
            mask = field == category
            if mask.any():
                components, count = label(mask)  # face-connected 4-neighbour
                sizes = np.bincount(components.ravel())[1:]
                a, b = set(components[0]) - {0}, set(components[-1]) - {0}
                largest = float(sizes.max() / field.size) if count else 0.0
                span = float(bool(a & b))
            else:
                largest, span = 0.0, 0.0
            out[f"fraction_{int(category)}"] = float(mask.mean())
            out[f"largest_{int(category)}"] = largest
            out[f"span_x_{int(category)}"] = span
        out["contact_x"] = float(np.mean(field[1:] != field[:-1]))
        out["contact_y"] = float(np.mean(field[:, 1:] != field[:, :-1]))
    return out


def category_universe(ti, zones, name):
    """Sorted integer category universe for a variable, declared before sampling.

    Union of the finite category codes in the primary TI variable and every
    zone TI's matching variable, so a category that is rare or absent in a given
    realization still has a fixed schema slot (finding F2).
    """
    sources = [ti.variable(name) if ti.multivariate else ti.variable()]
    for zone in zones:
        zt = zone.ti
        sources.append(zt.variable(name) if zt.multivariate else zt.variable())
    values = np.concatenate(
        [np.asarray(src.data, dtype=float).ravel() for src in sources]
    )
    finite = values[np.isfinite(values)]
    return sorted({int(round(c)) for c in np.unique(finite)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, default=100)
    args = parser.parse_args()
    # Validate the count BEFORE creating the study directory (finding F2). The
    # sample-SD Monte Carlo interval needs at least two realizations; one
    # realization is a smoke profile with intervals disabled.
    if args.count < 1:
        parser.error(f"--count must be >= 1, got {args.count}")
    intervals_enabled = args.count >= 2
    args.out.mkdir(parents=True, exist_ok=False)
    design = dict(
        cases=CASES,
        count=args.count,
        pilot_first=20,
        shape=[24, 24],
        ti_shape=[32, 36],
        node_seeds="200000+i",
        path_seeds="100000+i",
        contract=CONTRACT,
        connectivity="4-neighbor face-connected",
        hard_invariants=[
            "finite",
            "hard values exact",
            "mixed collocated tuple relation",
            "cleanup frozen half",
            "first realization serial==parallel",
        ],
        diagnostic_flags={
            "absolute_mean_difference_TI": 0.3,
            "absolute_facies_fraction_difference_TI": 0.15,
        },
        interpretation="Diagnostic flags fixed before sampling; not calibrated scientific equivalence limits. No release-ready conclusion. TI whole-grid metrics have different support; finite-grid variability not calibrated.",
        intervals="Mean +/-1.96 sample_SD/sqrt(realizations); approximate Monte Carlo intervals, no multiple-comparison equivalence claim.",
    )
    # Declare the per-case category universe BEFORE sampling (finding F2), so
    # the metric schema is fixed independently of any realization's draw.
    universes = {}
    for case in CASES:
        _ds, _ti = construct(case)
        zones = _ds.mps_model.zones
        case_universe = {}
        for v in _ti.variables:
            if v.categorical:
                key = v.name if case == "mixed" else "field"
                case_universe[key] = category_universe(_ti, zones, v.name)
        universes[case] = case_universe
    design["category_universes"] = universes
    (args.out / "design.json").write_text(json.dumps(design, indent=2))
    # Provenance written up front with a running status (finding F2); the
    # completed manifest below overwrites it on success.
    (args.out / "manifest.json").write_text(
        json.dumps({"status": "running", "count": args.count}, indent=2)
    )
    rows, summaries, flags, timing = [], {}, [], {}
    grid = [np.arange(24)] * 2
    for case in CASES:
        ds, ti = construct(case)
        case_universe = universes[case]
        start = time.monotonic()
        case_rows = []
        for i in range(args.count):
            controls = dict(
                path_seed=100000 + i,
                node_seed=200000 + i,
                num_threads=4,
                post_process=False,
                store=False,
            )
            field = ds(grid, **controls)
            fields = field if isinstance(field, dict) else {"field": field}
            assert all(np.isfinite(f).all() for f in fields.values())
            if case == "mixed":
                assert (
                    fields["facies"][0, 0] == 1 and fields["value"][0, 0] == 3
                )
                assert (
                    np.max(np.abs(fields["value"] - 3 * fields["facies"]))
                    <= 0.201
                )
            elif case == "variation":
                assert (
                    fields["field"][0, 0] == 10
                    and fields["field"][-1, -1] == 12
                )
            else:
                assert fields["field"][0, 0] == 0
            if i == 0:
                serial = ds(grid, **{**controls, "num_threads": 1})
                serial = (
                    serial if isinstance(serial, dict) else {"field": serial}
                )
                for name in fields:
                    np.testing.assert_array_equal(fields[name], serial[name])
                np.savez_compressed(
                    args.out / f"{case}-representative.npz", **fields
                )
            if case == "cleanup":
                ds.mps_model.post_processing = 0
                raw = ds(grid, **controls)
                ds.mps_model.post_processing = 1
                np.testing.assert_array_equal(fields["field"][:12], raw[:12])
            row = dict(
                case=case,
                realization=i,
                path_seed=100000 + i,
                node_seed=200000 + i,
            )
            for name, f in fields.items():
                variable = (
                    ti.variable(name) if case == "mixed" else ti.variable()
                )
                row.update(
                    {
                        name + "_" + key: value
                        for key, value in metrics(
                            f,
                            variable.categorical,
                            case_universe.get(name, ()),
                        ).items()
                    }
                )
                if case == "zones":
                    row["zone_primary_fraction"] = float(f[:12].mean())
                    row["zone_override_fraction"] = float(f[12:].mean())
                    row["zone_seam_contact"] = float(np.mean(f[11] != f[12]))
            if case == "mixed":
                row["cross_correlation"] = float(
                    np.corrcoef(
                        fields["facies"].ravel(), fields["value"].ravel()
                    )[0, 1]
                )
            if case == "cleanup":
                row["changed_fraction"] = float(
                    np.mean(raw != fields["field"])
                )
            rows.append(row)
            case_rows.append(row)
            if i == 19:
                (args.out / f"{case}-pilot.json").write_text(
                    json.dumps(
                        dict(seconds=time.monotonic() - start, realizations=20)
                    )
                )
        summary = {}
        # Iterate the union of all realizations' keys, not the first row's
        # (finding F2): with the fixed schema every row shares keys, but the
        # union is robust and also covers the case-specific extra columns.
        metric_keys = sorted(set().union(*(r.keys() for r in case_rows)))
        for key in metric_keys:
            if key in ("case", "realization", "path_seed", "node_seed"):
                continue
            values = np.array([r[key] for r in case_rows])
            mean = values.mean()
            # One realization has no sample SD; report the mean with an
            # undefined (NaN) interval rather than dividing by zero.
            error = (
                1.96 * values.std(ddof=1) / np.sqrt(len(values))
                if intervals_enabled
                else float("nan")
            )
            summary[key] = dict(
                mean=float(mean),
                lower=float(mean - error),
                upper=float(mean + error),
            )
        for variable in ti.variables:
            name = variable.name if case == "mixed" else "field"
            finite = variable.data[np.isfinite(variable.data)]
            difference = abs(
                summary[name + "_mean"]["mean"] - float(finite.mean())
            )
            if case != "variation" and difference > 0.3:
                flags.append(
                    dict(
                        case=case,
                        metric="mean_difference_TI",
                        difference=difference,
                    )
                )
            if variable.categorical:
                for category in case_universe.get(name, ()):
                    key = name + "_fraction_" + str(int(category))
                    difference = abs(
                        summary.get(key, {"mean": 0.0})["mean"]
                        - float(np.mean(finite == category))
                    )
                    if difference > 0.15:
                        flags.append(
                            dict(case=case, metric=key, difference=difference)
                        )
        summaries[case] = summary
        timing[case] = time.monotonic() - start
        print(case, args.count, round(timing[case], 2), flush=True)
    columns = sorted(set().union(*(r.keys() for r in rows)))
    with (args.out / "metrics.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    (args.out / "summary.json").write_text(
        json.dumps(
            dict(metrics=summaries, diagnostic_flags=flags, timings=timing),
            indent=2,
        )
    )
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for j, metric in enumerate(("field_mean", "field_semivar_x")):
        selected = [case for case in CASES if metric in summaries[case]]
        means = [summaries[c][metric]["mean"] for c in selected]
        error = [
            summaries[c][metric]["upper"] - summaries[c][metric]["mean"]
            for c in selected
        ]
        axes[j].errorbar(np.arange(len(selected)), means, yerr=error, fmt="o")
        axes[j].set_xticks(np.arange(len(selected)), selected, rotation=45)
        axes[j].set_title(metric + " (95% Monte Carlo intervals)")
    fig.tight_layout()
    fig.savefig(args.out / "uncertainty.png", dpi=160)
    root = Path(__file__).resolve().parents[1]
    hashes = {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (root / "src/gstools/mps").glob("*.py")
    }
    manifest = dict(
        platform=platform.platform(),
        python=sys.executable,
        numpy=np.__version__,
        gstools=gs.__file__,
        contract=CONTRACT,
        source_hashes=hashes,
        head=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        script_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        status="completed",
        outcome="diagnostic_only",
        count=args.count,
        intervals_enabled=intervals_enabled,
    )
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (args.out / "REPORT.txt").write_text(
        f"Synthetic diagnostic ensembles: {args.count} realizations x {len(CASES)} cases.\nAll deterministic invariants passed. {len(flags)} predeclared diagnostic flags: see summary.json.\nIntervals use independent realizations, not pixels. A 20-realization runtime pilot is included within each 100-realization study, not an independent confirmation sample.\nNo reference comparison or finite-grid equivalence calibration was performed. Connectivity/contact summaries are diagnostics; release scientific fidelity remains a maintainer decision.\n"
    )


if __name__ == "__main__":
    main()
