#!/usr/bin/env python
"""Write a per-job ASV configuration for GitHub Actions.

The committed ASV configs stay convenient for local benchmarking. CI uses this
helper to pin one Python/NumPy/SciPy combination per matrix job without turning
normal local ``asv run`` commands into a large dependency matrix.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def validate_asv_req_spec(package, spec):
    if spec.startswith((">", "<", "~", "!=")) or "," in spec:
        raise ValueError(
            f"{package} uses open-ended CI spec {spec!r}. ASV's conda "
            "backend expects exact matrix pins here, for example '==2.1.3'."
        )
    return spec


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--python-version", required=True)
    parser.add_argument("--numpy", required=True)
    parser.add_argument("--scipy", required=True)
    parser.add_argument("--env-dir", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--html-dir", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    with args.base_config.open(encoding="utf8") as config_file:
        config = json.load(config_file)

    config["pythons"] = [args.python_version]
    config["env_dir"] = args.env_dir
    config["results_dir"] = args.results_dir
    config["html_dir"] = args.html_dir

    req = config.setdefault("matrix", {}).setdefault("req", {})
    try:
        req["numpy"] = [validate_asv_req_spec("numpy", args.numpy)]
        req["scipy"] = [validate_asv_req_spec("scipy", args.scipy)]
    except ValueError as err:
        raise SystemExit(str(err)) from err

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf8") as config_file:
        json.dump(config, config_file, indent=2)
        config_file.write("\n")

    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
