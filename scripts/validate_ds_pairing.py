"""Paired Direct Sampling (frontend + Rust Core) release-validation driver.

This script is the single, CI-agnostic entry point for verifying that the
GSTools DS frontend works against the *matching* ``gstools-core`` built from
source. It builds both wheels, installs the two **exact wheel paths** into a
throwaway environment, asserts the imported packages come from that environment
(not an editable tree or a registry build), then runs the native smoke test, the
bounded-subprocess boundary matrix and the MPS test suite from outside the
source tree. It writes a ``manifest.json`` binding both repository SHAs and both
wheel SHA-256 hashes to the results.

Why it exists (audit finding F3): the two repositories have independent CI, and
neither currently tests the frontend against the paired Core. Rather than encode
that in a specific CI system, the whole sequence lives here so a maintainer only
has to *call* it from whatever workflow they run.

Maintainer CI checklist (what still needs a decision/secret, per the audit):
  * Add one job per supported runner that does:
        python scripts/validate_ds_pairing.py --core <path-or-checkout-of-core>
    on untrusted PRs run it WITHOUT publish credentials.
  * Decide the macOS arch matrix explicitly (macos-13 = Intel, macos-14 = arm64)
    and, if a universal2 wheel is shipped, build it with
    ``--target universal2-apple-darwin`` and verify slices with ``lipo -archs``.
  * Decide whether 32-bit (Linux i686 / Windows x86) stays supported; if so, run
    this driver on a real 32-bit interpreter (finding F7).
  * Choose paired release versions and a dependency floor before enabling
    publish; keep the ABI gate until package versioning distinguishes this Core
    from the historical 1.4.0 (finding F5).

Usage (local or CI):
    python scripts/validate_ds_pairing.py --core ../GSTools-Core
    python scripts/validate_ds_pairing.py --core ../GSTools-Core --out /tmp/pair
Options:
    --no-build        reuse wheels already present in <out>/wheels
    --maturin PATH    maturin executable (default: found on PATH)
    --system-site     create the venv with --system-site-packages so heavy
                      runtime deps (numpy/scipy/pytest) are reused instead of
                      reinstalled; the two project wheels still override them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import venv
from pathlib import Path


def run(cmd, log, env=None, cwd=None):
    """Run a command, tee to a log file, raise on non-zero exit."""
    print("+ " + " ".join(str(c) for c in cmd), flush=True)
    with open(log, "ab") as handle:
        handle.write(("+ " + " ".join(str(c) for c in cmd) + "\n").encode())
        result = subprocess.run(
            [str(c) for c in cmd],
            env=env,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        handle.write(result.stdout)
    sys.stdout.buffer.write(result.stdout[-2000:])
    if result.returncode != 0:
        raise SystemExit(f"command failed ({result.returncode}): {cmd[0]} — see {log}")


def git(repo, *args):
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), *args], text=True
        ).strip()
    except Exception:
        return "unknown"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def one_wheel(directory, pattern):
    hits = sorted(Path(directory).glob(pattern))
    if len(hits) != 1:
        raise SystemExit(f"expected exactly one {pattern} in {directory}, got {hits}")
    return hits[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core", required=True, type=Path, help="gstools-core source dir")
    parser.add_argument(
        "--frontend",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="frontend (GSTools) source dir; default: this repo",
    )
    parser.add_argument("--out", type=Path, default=Path("ds-pairing-out"))
    parser.add_argument("--maturin", default=shutil.which("maturin"))
    parser.add_argument("--no-build", action="store_true")
    parser.add_argument("--system-site", action="store_true")
    args = parser.parse_args()

    core = args.core.resolve()
    frontend = args.frontend.resolve()
    out = args.out.resolve()
    wheels = out / "wheels"
    logs = out / "logs"
    if not args.no_build and out.exists():
        raise SystemExit(f"--out {out} exists; choose a fresh directory")
    wheels.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)

    manifest = {
        "frontend": {"path": str(frontend), "head": git(frontend, "rev-parse", "HEAD"),
                     "dirty": bool(git(frontend, "status", "--porcelain"))},
        "core": {"path": str(core), "head": git(core, "rev-parse", "HEAD"),
                 "dirty": bool(git(core, "status", "--porcelain"))},
        "steps": {},
    }

    # 1. Build both wheels. maturin must run with conda prefixes unset.
    if not args.no_build:
        if not args.maturin:
            raise SystemExit("maturin not found; pass --maturin PATH")
        build_env = {k: v for k, v in os.environ.items()
                     if k not in ("CONDA_PREFIX", "CONDA_DEFAULT_ENV")}
        run([args.maturin, "build", "--release", "--locked",
             "-m", core / "Cargo.toml", "-o", wheels],
            logs / "core-build.log", env=build_env)
        run([sys.executable, "-m", "build", "--wheel", "-o", wheels, frontend],
            logs / "frontend-build.log")

    core_wheel = one_wheel(wheels, "gstools_core-*.whl")
    frontend_wheel = one_wheel(wheels, "gstools-*.whl")
    manifest["core"]["wheel"] = {"name": core_wheel.name, "sha256": sha256(core_wheel)}
    manifest["frontend"]["wheel"] = {"name": frontend_wheel.name, "sha256": sha256(frontend_wheel)}

    # 2. Fresh environment; install the two EXACT wheel paths (never by name).
    env_dir = out / "venv"
    if env_dir.exists():
        shutil.rmtree(env_dir)
    venv.EnvBuilder(with_pip=True, system_site_packages=args.system_site).create(env_dir)
    vpy = env_dir / ("Scripts" if os.name == "nt" else "bin") / "python"
    run([vpy, "-m", "pip", "install", "--no-deps", "--force-reinstall",
         core_wheel, frontend_wheel], logs / "install.log")

    # 3. Assert the imports resolve to the fresh environment, not an editable
    #    source tree or a registry build.
    run([vpy, "-c",
         "import gstools, gstools_core, sys, pathlib;"
         "root=pathlib.Path(sys.prefix).resolve();"
         "g=pathlib.Path(gstools.__file__).resolve();"
         "c=pathlib.Path(gstools_core.__file__).resolve();"
         "assert root in g.parents, g;"
         "assert root in c.parents, c;"
         "print('api', gstools_core.mps_api_version(), g, c)"],
        logs / "origins.log")

    # 4. Native smoke + boundary matrix + MPS suite, run from outside the trees.
    work = out / "work"
    work.mkdir(exist_ok=True)
    run([vpy, core / "scripts" / "check_mps_wheel.py"], logs / "smoke.log", cwd=work)
    run([vpy, core / "scripts" / "check_mps_boundaries.py",
         "--output", out / "boundaries.json"], logs / "boundaries.log", cwd=work)
    run([vpy, "-m", "pytest", frontend / "tests", "-k", "mps", "-q",
         "--junitxml", out / "mps-junit.xml"], logs / "mps-tests.log", cwd=work)
    manifest["steps"] = {"build": not args.no_build, "install": True,
                         "origins": True, "smoke": True, "boundaries": True,
                         "mps_tests": True}
    manifest["status"] = "passed"
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("\nPAIRED DS VALIDATION: PASS")
    print(f"  manifest: {out / 'manifest.json'}")


if __name__ == "__main__":
    main()
