#!/usr/bin/env python
"""Write ASV machine metadata for a non-interactive CI runner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--machine", required=True)
    parser.add_argument("--os", required=True)
    parser.add_argument("--arch", required=True)
    parser.add_argument("--cpu", required=True)
    parser.add_argument("--num-cpu", required=True)
    parser.add_argument("--ram", required=True)
    parser.add_argument(
        "--path",
        type=Path,
        default=Path.home() / ".asv-machine.json",
        help="Machine metadata file written by 'asv machine'.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.path.exists():
        with args.path.open(encoding="utf8") as machine_file:
            machines = json.load(machine_file)
    else:
        machines = {"version": 1}

    machines["version"] = 1
    machines[args.machine] = {
        "machine": args.machine,
        "os": args.os,
        "arch": args.arch,
        "cpu": args.cpu,
        "num_cpu": str(args.num_cpu),
        "ram": args.ram,
    }

    args.path.parent.mkdir(parents=True, exist_ok=True)
    with args.path.open("w", encoding="utf8") as machine_file:
        json.dump(machines, machine_file, indent=2)
        machine_file.write("\n")

    print(f"Wrote ASV machine metadata for {args.machine} to {args.path}")


if __name__ == "__main__":
    main()
