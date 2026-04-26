# GSTools ASV Benchmarks

This directory contains performance benchmarks for GSTools. Unit tests in
`tests/` should remain focused on correctness; ASV benchmarks in this
directory measure runtime and peak memory only.

For a beginner-friendly explanation of ASV and every file added here, read
[`ASV_TUTORIAL.md`](ASV_TUTORIAL.md).

## Setup

Run ASV commands from the GSTools repository root, where `asv.conf.json`
is located:

```bash
cd /Users/lealroja/Documents/UFZ/MPS-Tools/GSTools
conda install -c conda-forge asv
asv machine
```

## Common Commands

```bash
asv run --quick
asv run HEAD^!
asv run
asv publish
asv preview
asv compare HEAD~1 HEAD
```

`asv run --quick` is the quick development check for ASV 0.6.x. It runs each
benchmark only once and does not save useful performance results.

`asv run HEAD^!` benchmarks only the current commit. Plain `asv run` follows
the branches configured in `asv.conf.json`.

If you need to run ASV from another directory, pass the config explicitly:

```bash
asv --config /Users/lealroja/Documents/UFZ/MPS-Tools/GSTools/asv.conf.json run --quick
```

## Backend Comparison

Benchmarks are parameterized with readable backend labels:

- `cython_fallback`
- `rust_core`

ASV tracks each backend separately. Interpret Rust speedup on the same machine
and same benchmark as:

```text
speedup = cython_fallback_time / rust_core_time
```

So:

- `speedup > 1.0` means Rust is faster
- `speedup = 1.0` means similar performance
- `speedup < 1.0` means Rust is slower

Do not compare absolute benchmark times across different machines.
