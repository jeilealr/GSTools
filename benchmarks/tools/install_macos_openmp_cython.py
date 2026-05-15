#!/usr/bin/env python
"""Install GSTools-Cython with OpenMP inside a macOS ASV environment.

This helper is intentionally macOS-specific. It is called from
``asv.macos-openmp.conf.json`` after ASV has created a conda environment that
contains ``llvm-openmp``.
"""

from __future__ import annotations

import os
import platform
import stat
import subprocess
import sys
from pathlib import Path


def run(command, env=None, check=True):
    print("+ " + " ".join(str(part) for part in command), flush=True)
    return subprocess.run(command, check=check, env=env)


def write_wrapper(path, force_cxx=False):
    text = """#!/bin/bash
set -e
prefix="${GSTOOLS_OPENMP_PREFIX:-${CONDA_PREFIX:-}}"
name="$(basename "$0")"
if [[ "${GSTOOLS_FORCE_CXX:-0}" == "1" || "$name" == *++* ]]; then
  real="${GSTOOLS_REAL_CXX:-/usr/bin/clang++}"
else
  real="${GSTOOLS_REAL_CC:-/usr/bin/clang}"
fi
is_compile=0
for arg in "$@"; do
  [[ "$arg" == "-c" ]] && is_compile=1
done
args=()
for arg in "$@"; do
  if [[ "$arg" == "-fopenmp" ]]; then
    if [[ "$is_compile" == "1" ]]; then
      args+=("-Xpreprocessor" "-fopenmp" "-I${prefix}/include")
    else
      args+=("-L${prefix}/lib" "-lomp" "-Wl,-rpath,${prefix}/lib")
    fi
  else
    args+=("$arg")
  fi
done
exec "$real" "${args[@]}"
"""
    if force_cxx:
        text = """#!/bin/bash
GSTOOLS_FORCE_CXX=1 exec "$(dirname "$0")/gstools-asv-clang-openmp" "$@"
"""
    path.write_text(text, encoding="utf8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def main():
    if len(sys.argv) != 2:
        print(
            "Usage: install_macos_openmp_cython.py <asv-env-dir>",
            file=sys.stderr,
        )
        return 2

    if platform.system() != "Darwin":
        print(
            "This helper is macOS-specific. Use the default ASV config or "
            "write an OpenMP setup for this platform.",
            file=sys.stderr,
        )
        return 2

    env_dir = Path(sys.argv[1]).resolve()
    include_dir = env_dir / "include"
    lib_dir = env_dir / "lib"
    omp_header = include_dir / "omp.h"
    omp_lib = lib_dir / "libomp.dylib"

    if not omp_header.exists() or not omp_lib.exists():
        print(
            "llvm-openmp was not found in the ASV environment. Expected "
            f"{omp_header} and {omp_lib}.",
            file=sys.stderr,
        )
        return 2

    cc_wrapper = env_dir / "bin" / "gstools-asv-clang-openmp"
    cxx_wrapper = env_dir / "bin" / "gstools-asv-clang-openmp++"
    write_wrapper(cc_wrapper)
    write_wrapper(cxx_wrapper, force_cxx=True)

    build_env = os.environ.copy()
    build_env.update(
        {
            "GSTOOLS_BUILD_PARALLEL": "1",
            "GSTOOLS_OPENMP_PREFIX": str(env_dir),
            "CC": str(cc_wrapper),
            "CXX": str(cxx_wrapper),
            "CFLAGS": f"-I{include_dir}",
            "LDFLAGS": f"-L{lib_dir}",
        }
    )

    run(
        [
            sys.executable,
            "-m",
            "pip",
            "uninstall",
            "-y",
            "gstools-cython",
            "gstools_cython",
        ],
        env=build_env,
        check=False,
    )
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-build-isolation",
            "--no-cache-dir",
            "--force-reinstall",
            "--no-binary=gstools-cython",
            "--no-deps",
            "gstools-cython",
        ],
        env=build_env,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
