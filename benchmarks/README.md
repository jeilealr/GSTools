# GSTools Benchmark Guide

This directory contains the Airspeed Velocity ([ASV](https://github.com/airspeed-velocity/asv/)) benchmark suite for GSTools and a complementary profiling helper implemented with cProfile (part of the Python standard library).

This is a measurement-first guide: benchmark real workflows, inspect the
results, profile the slow paths, and then decide what to optimize.

Unit tests in `tests/` answer "is the code correct?". The ASV benchmarks in
`benchmarks/` answer "how fast is this workflow, how much memory does it use,
and did that change across commits?". The complementary cProfile helper
answers "inside this workflow, which Python functions are taking most of the
time right now?".

The benchmarks compare two GSTools backends, which gives more context for
deciding where optimization work should go:

- `cython_fallback`: the default Cython-backed fallback implementation from
  [gstools-cython](https://github.com/GeoStat-Framework/GSTools-Cython).
- `rust_core`: the Rust-backed implementation from
  [gstools_core](https://github.com/GeoStat-Framework/GSTools-Core).

## Index

- [Setup](#setup)
- [Benchmarking Scripts](#benchmarking-scripts)
- [ASV Configuration](#asv-configuration)
- [Benchmark Naming](#benchmark-naming)
- [Benchmark Coverage](#benchmark-coverage)
- [Benchmark Classes](#benchmark-classes)
- [VariogramWorkflowBenchmarks](#variogramworkflowbenchmarks)
- [KrigingWorkflowBenchmarks](#krigingworkflowbenchmarks)
- [RandomFieldWorkflowBenchmarks](#randomfieldworkflowbenchmarks)
- [Running The Benchmarks](#running-the-benchmarks)
- [Profiling With cProfile](#profiling-with-cprofile)
- [More ASV Commands](#more-asv-commands)
- [External Reference](#external-reference)

## Setup

The regular installation commands in the main `README.md` install GSTools for
normal use. For benchmark work, install this local checkout with the optional
benchmark dependencies.

1. Move to the GSTools repository root:

```bash
cd /path/to/GSTools
```

2. Install GSTools in editable mode with the benchmark tooling and Rust backend:

```bash
python -m pip install -e ".[benchmark,rust]"
```

3. Create a machine profile once per computer:

```bash
asv machine --yes
```

Notes:

- The machine profile records local hardware information so ASV can label
  results correctly. Do not compare absolute times across different machines.
- You can also install ASV with conda or pip, and you can install the Rust
  backend package from
  [gstools_core](https://github.com/GeoStat-Framework/GSTools-Core) directly.

## Benchmarking Scripts

The benchmarking setup currently consists of:

- `asv.conf.json`: tells ASV how to build GSTools, where benchmarks live, where
  to store results, and which Python/environment matrix to use.
- `benchmarks/benchmark_backends.py`: contains the ASV benchmark classes.
- `benchmarks/README.md`: this practical guide.
- `benchmarks/tools/asv_speedup_summary.py`: reads `.asv/results/` and prints
  Rust-vs-Cython speedup ratios.
- `benchmarks/tools/profile_benchmark_workflows.py`: runs one representative
  workflow from `benchmark_backends.py` under Python's built-in `cProfile`, so
  you can see which functions take time in the current checkout.

Do not run `benchmarks/benchmark_backends.py` directly with Python. ASV loads
that file, discovers benchmark classes and methods, and runs them inside
isolated benchmark environments. The scripts in `benchmarks/tools/` are
different: run them directly with Python. The profiling helper can run against
the current checkout at any time; the speedup-summary helper needs saved ASV
results in `.asv/results/`.

### ASV Configuration

The repo root `asv.conf.json` is tailored to this GSTools checkout:

```json
{
  "repo": ".",
  "branches": ["main"],
  "benchmark_dir": "benchmarks",
  "env_dir": ".asv/env",
  "results_dir": ".asv/results",
  "html_dir": ".asv/html",
  "environment_type": "conda",
  "pythons": ["3.12"],
  "install_command": [
    "in-dir={env_dir} python -m pip install {build_dir}[rust]"
  ]
}
```

Important details:

- `install_command` installs the checked-out GSTools revision with the `[rust]`
  extra, so `gstools_core` should be available for Rust backend measurements.
  ASV still needs its own `install_command` because it creates isolated
  environments for the commits it benchmarks.
- This is separate from any editable install in your active development
  environment, such as `python -m pip install -e ".[benchmark,rust]"`.
  The editable install is only needed when you want your active environment to
  import the current checkout directly, for example when running
  `benchmarks/tools/profile_benchmark_workflows.py` with `--backend rust_core`.
- ASV and the cProfile helper use different environments. ASV runs
  `benchmarks/benchmark_backends.py` inside `.asv/env/`; the cProfile helper
  imports the same benchmark classes but runs them in your active Python
  environment.

ASV creates these generated directories:

```text
.asv/env/      benchmark environments
.asv/results/  local benchmark result JSON files
.asv/html/     generated local benchmark website
```

Those directories are machine-specific generated artifacts. They should
normally stay out of git.

If needed, users can list more than one branch, Python version, benchmark
directory, and so on. For example:

```json
"branches": ["main", "my-feature-branch"]
```

Users can also benchmark any explicit branch, commit, tag, or range without
changing `asv.conf.json`:

```bash
asv run my-feature-branch^! --bench benchmark_backends
asv run main..my-feature-branch --bench benchmark_backends
```

ASV checks out package code at each git commit being benchmarked. Commit source
changes before benchmarking them with ASV. Otherwise ASV may benchmark the last
committed package code rather than your uncommitted source changes.


### Benchmark Naming

ASV recognizes benchmark methods by name:

- methods starting with `time_` measure runtime
- methods starting with `peakmem_` measure peak memory
- `setup_cache()` creates reusable data once per benchmark environment
- `setup()` can skip or prepare individual parameter combinations

## Benchmark Coverage

### Shared Constants

```python
BACKENDS = ("cython_fallback", "rust_core")
VARIOGRAM_CASES = (
    "full_900",
    "sampled_5000_to_1500",
    "sampled_15000_to_4500",
)
KRIGE_CASES = ("small_30x500", "large_120x2000", "extra_large_360x6000")
FIELD_CASES = (
    "srf_unstructured_randmeth",
    "srf_structured_randmeth",
    "srf_structured_fourier",
    "condsrf_unstructured",
)
```

These constants define parameter labels shown in ASV results.

`BACKENDS` compares:

- `cython_fallback`
- `rust_core`

### Shared Helpers

`gstools_backend(use_core)` temporarily forces GSTools to use either the Cython
fallback backend or the Rust `gstools_core` backend.

`_random_points(seed, count, scale)` creates deterministic 2D point clouds.

`_smooth_field(x, y)` creates deterministic synthetic values:

```python
np.sin(x / 10.0) + np.cos(y / 15.0)
```

`_make_variogram_data(...)` creates positions, field values, and bins for
variogram estimation.

`_make_krige_data(...)` creates conditioning points, conditioning values, and
target points for kriging and conditioned random fields.

The fixed random seeds are intentional. They keep benchmark inputs stable so
changes in results are more likely to come from code changes, not new random
data.

### Benchmark Classes

The ASV benchmarking is organized around workflow classes. Each workflow class
compares `cython_fallback` and `rust_core`, and each class includes both
runtime and peak-memory methods.

The suite currently measures:

- `VariogramWorkflowBenchmarks`: full pairwise work vs sampled large work
- `KrigingWorkflowBenchmarks`: small vs larger global kriging systems
- `RandomFieldWorkflowBenchmarks`: unstructured SRF, structured SRF, Fourier
  SRF, and conditioned SRF

This keeps the ASV suite focused on representative workflows rather than
separate duplicate backend checks.

#### VariogramWorkflowBenchmarks

This class measures variogram estimation cases:

```text
full_900
sampled_5000_to_1500
sampled_15000_to_4500
```

The labels mean:

- `full_900`: create 900 scattered points and use all 900 points for the
  variogram calculation.
- `sampled_5000_to_1500`: create 5,000 scattered points, then randomly select
  1,500 of those points for the variogram calculation.
- `sampled_15000_to_4500`: create 15,000 scattered points, then randomly select
  4,500 of those points for the variogram calculation.

The sampled cases still represent larger input datasets, but the variogram
calculation is done on the randomly selected subset so the pairwise work stays
practical.

#### KrigingWorkflowBenchmarks

This class measures global kriging at three scales:

```text
small_30x500
large_120x2000
extra_large_360x6000
```

The labels mean:

- `small_30x500`: 30 conditioning points, 500 target points
- `large_120x2000`: 120 conditioning points, 2,000 target points
- `extra_large_360x6000`: 360 conditioning points, 6,000 target points

#### RandomFieldWorkflowBenchmarks

This class measures SRF and CondSRF generation workflows:

```text
srf_unstructured_randmeth
srf_structured_randmeth
srf_structured_fourier
condsrf_unstructured
```

The cases are:

- `srf_unstructured_randmeth`: SRF using RandMeth on 2,000 unstructured points
- `srf_structured_randmeth`: SRF using RandMeth on a 64 by 64 structured grid
- `srf_structured_fourier`: SRF using the Fourier generator on a 64 by 64
  structured grid
- `condsrf_unstructured`: conditioned SRF with 40 conditioning points and 1,000
  target points

## Running The Benchmarks

Check that the benchmark module imports and runs:

```bash
asv run --quick --show-stderr --bench benchmark_backends
```

Save a baseline for the current commit:

```bash
asv run HEAD^! --bench benchmark_backends
```

Run the last five commits on a linear branch:

```bash
asv run HEAD~5..HEAD --bench benchmark_backends
```

Build and open the local website:

```bash
asv publish
asv preview
```

Then open the printed local URL,  for example:

```text
http://127.0.0.1:8082/#/
```
(or any other `http://127.0.0.1:<port>/#/` URL shown by the running preview).

After ASV has saved results, print explicit Rust-vs-Cython speedup ratios:

```bash
python benchmarks/tools/asv_speedup_summary.py
```

The helper reads `.asv/results/` and reports:

```text
speedup = cython_fallback_time / rust_core_time
```

Interpret the ratio as:

- `speedup > 1.0` means Rust is faster
- `speedup = 1.0` means similar performance
- `speedup < 1.0` means Rust is slower

The browser report shows ASV plots and trends. The speedup helper prints the
backend ratio explicitly in the terminal. By default, the helper skips removed
legacy duplicate rows from older saved results.

## Profiling With cProfile

`cProfile` is useful for the current checkout. It does not update the ASV
browser report. Instead, it prints a table in the terminal showing which Python
functions consumed time while one workflow ran.

The helper script is:

```text
benchmarks/tools/profile_benchmark_workflows.py
```

It imports the ASV benchmark classes from `benchmark_backends.py`, selects one
case, forces one backend, and runs that case under `cProfile`.

List available cases:

```bash
python benchmarks/tools/profile_benchmark_workflows.py --list
```

Profile selected cases:

```bash
python benchmarks/tools/profile_benchmark_workflows.py --case variogram-sampled --backend rust_core --limit 10
python benchmarks/tools/profile_benchmark_workflows.py --case variogram-extra-large --backend rust_core --limit 10
python benchmarks/tools/profile_benchmark_workflows.py --case krige-large --backend rust_core --limit 10
python benchmarks/tools/profile_benchmark_workflows.py --case krige-extra-large --backend rust_core --limit 10
python benchmarks/tools/profile_benchmark_workflows.py --case condsrf --backend rust_core --limit 10
```

Useful options:

- `--case`: choose one workflow, or use `all`
- `--backend`: choose `cython_fallback` or `rust_core`
- `--limit`: number of function rows to print from the cProfile table
- `--sort cumtime`: sort by cumulative time, usually the best first view
- `--sort tottime`: sort by time spent directly in each function
- `--repeat`: repeat a workflow inside the profiler

For example, `--limit 10` means "print the top 10 function rows after sorting".

## More ASV Commands

Save results for only the current commit:

```bash
asv run HEAD^! --bench benchmark_backends
```

Compare current commit with previous commit:

```bash
asv run HEAD~1^! --bench benchmark_backends
asv run HEAD^! --bench benchmark_backends
asv compare HEAD~1 HEAD
```

Compare local `main` with the current branch tip:

```bash
asv run main^! --bench benchmark_backends
asv run HEAD^! --bench benchmark_backends
asv compare main HEAD
```

Compare remote `main` with the current branch tip:

```bash
git fetch origin main
asv run origin/main^! --bench benchmark_backends
asv run HEAD^! --bench benchmark_backends
asv compare origin/main HEAD
```

On a linear branch, `HEAD~5..HEAD` benchmarks:

```text
HEAD~4
HEAD~3
HEAD~2
HEAD~1
HEAD
```

Run a selected list of commits:

```bash
git rev-parse HEAD HEAD~3 main e20c88f7 > /tmp/gstools-asv-commits.txt
asv run HASHFILE:/tmp/gstools-asv-commits.txt --bench benchmark_backends
```

Use full commit hashes when sharing results. Short hashes and branch names are
fine locally but can become ambiguous later.

If running ASV from outside the repo root, pass the config explicitly:

```bash
asv --config /path/to/MPS-Tools/GSTools/asv.conf.json run --quick --bench benchmark_backends
```

## External Reference

For complete ASV command syntax, see:

```text
https://asv.readthedocs.io/en/stable/commands.html
```
