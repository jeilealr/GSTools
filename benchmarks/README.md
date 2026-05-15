# GSTools Benchmark Guide

This directory contains the Airspeed Velocity ([ASV](https://github.com/airspeed-velocity/asv/)) benchmark suite for GSTools and a complementary profiling helper implemented with cProfile (part of the Python standard library).

This guide benchmarks GSTools, inspects the
results, profiles where runtime is spent, and then decides what to optimize.

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
  - [Shared Constants](#shared-constants)
  - [Shared Helpers](#shared-helpers)
  - [Benchmark Classes](#benchmark-classes)
    - [VariogramWorkflowBenchmarks](#variogramworkflowbenchmarks)
    - [KrigingWorkflowBenchmarks](#krigingworkflowbenchmarks)
    - [RandomFieldWorkflowBenchmarks](#randomfieldworkflowbenchmarks)
- [Running The Benchmarks](#running-the-benchmarks)
  - [Baseline Benchmark](#baseline-benchmark)
    - [Current Commit Baseline](#current-commit-baseline)
    - [Several Commits Baseline](#several-commits-baseline)
    - [Summary of Results](#summary-of-results)
    - [Visualization of Results](#visualization-of-results)
    - [Profiling With cProfile](#profiling-with-cprofile)
- [Optional Parallelisation with OpenMP](#optional-parallelisation-with-openmp)
  - [Shared OpenMP Rule](#shared-openmp-rule)
  - [macOS Example](#macos-example)
    - [What The macOS OpenMP Config Does](#what-the-macos-openmp-config-does)
    - [Run On macOS](#run-on-macos)
    - [Interpreting The macOS OpenMP Run](#interpreting-the-macos-openmp-run)
  - [Windows Example](#windows-example)
  - [Linux Example](#linux-example)
  - [HPC Example](#hpc-example)
  - [Profiling With cProfile for Multiple Threads](#profiling-with-cprofile-for-multiple-threads)
- [More ASV Commands](#more-asv-commands)
- [External Reference](#external-reference)

## Setup

The regular installation commands in the main `README.md` install GSTools for
normal use. This benchmark guide uses conda because ASV creates isolated
benchmark environments for the commits it measures.

The default benchmark configuration intentionally compares both backends with
one GSTools thread:

```text
gstools.config.NUM_THREADS = 1
```

That keeps the first comparison simple: Cython fallback vs Rust core without
parallelism as a confounding factor. Parallel/OpenMP scaling is treated as a
separate optional experiment because the correct Cython OpenMP build depends on
the user's operating system, compiler, and runtime environment.

To run the benchmark and the optional cProfile helper, follow these steps:

1. Move to the GSTools repository root:

```bash
cd /path/to/GSTools
```

2. Create and activate a conda environment for local benchmark work:

```bash
conda create -n gstools-benchmark -c conda-forge python=3.12 asv packaging
conda activate gstools-benchmark
```

If you already have a suitable conda environment, activate that instead.

3. If you use an existing environment, make sure ASV is installed:

```bash
conda install -c conda-forge asv
```

4. Create a machine profile once per computer:

```bash
asv machine --yes
```

The machine profile records local hardware information so ASV can label results correctly. Do not compare absolute times across different machines.

## Benchmarking Scripts

The benchmarking setup currently consists of:

- `asv.conf.json`: tells ASV how to build GSTools, where benchmarks live, where
  to store results, and which Python/environment matrix to use.
- `asv.macos-openmp.conf.json`: optional macOS-specific ASV configuration that
  builds `gstools-cython` from source with OpenMP inside ASV's own environment.
- `benchmarks/benchmark_backends.py`: contains the ASV benchmark classes.
- `benchmarks/README.md`: this practical guide.
- `benchmarks/tools/asv_speedup_summary.py`: reads `.asv/results/` and prints
  Rust-vs-Cython speedup ratios.
- `benchmarks/tools/profile_benchmark_workflows.py`: runs one representative
  workflow from `benchmark_backends.py` under Python's built-in `cProfile`, so
  you can see which functions take time in the current checkout.
- `benchmarks/tools/check_cython_openmp.py`: optional helper for checking
  whether the active Python environment's GSTools-Cython extensions detect
  OpenMP parallel support.
- `benchmarks/tools/install_macos_openmp_cython.py`: helper used only by
  `asv.macos-openmp.conf.json` to compile `gstools-cython` with `llvm-openmp`
  on macOS.


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
  "matrix": {
    "req": {
      "emcee": [""],
      "hankel": [""],
      "meshio": [""],
      "numpy": [""],
      "pyevtk": [""],
      "scipy": [""],
      "gstools-cython": [""]
    }
  },
  "install_command": [
    "in-dir={env_dir} python -m pip install gstools_core>=1.0.0",
    "in-dir={env_dir} python -m pip install --no-deps {build_dir}"
  ]
}
```

Important details:

- `environment_type: "conda"` means conda is required for the ASV workflow in
  this guide. ASV creates isolated conda environments for the commits it
  benchmarks.
- `pythons: ["3.12"]` means ASV creates Python 3.12 benchmark environments.
  Keep this pinned unless you intentionally validate a newer Python/GSTools
  backend stack.
- `matrix.req` asks ASV to install GSTools runtime dependencies before
  installing the checked-out GSTools source. It includes `gstools-cython`
  explicitly because the GSTools commit is installed with `--no-deps`.
- `{build_dir}` is ASV's temporary checkout/build directory for the exact
  GSTools commit being benchmarked.
- `install_command` installs the checked-out GSTools revision with `--no-deps`.
  It also installs `gstools_core` with pip because `gstools-core` is not
  available as a conda package in every solver/platform combination.
- ASV still needs its own `install_command` because it creates isolated
  environments for the commits it benchmarks.
- Run the cProfile helper with the Python executable from ASV's isolated
  environment, for example `.asv/env/<env-id>/bin/python`. In that mode, the
  ASV environment provides dependencies while the helper imports the current
  checkout through the repo `src/` path.

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

This section describes what is measured by the ASV suite and how the benchmark
labels map to real GSTools workflows. The goal is to cover representative
operations that are relevant for geostatistical work, not isolated
micro-functions.

The current suite measures runtime and peak memory for variogram estimation,
global kriging, spatial random field generation, and conditioned random field
generation. Each workflow is run with both backends so the results can show
both absolute performance and Rust-vs-Cython differences.

### Shared Constants

```python
BACKENDS = ("cython_fallback", "rust_core")
THREAD_COUNTS = _configured_thread_counts()
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

`THREAD_COUNTS` defaults to:

- `threads_1`: force `gstools.config.NUM_THREADS = 1`

That is the default because the first benchmark target is a clean Cython-vs-Rust
backend comparison without parallelism.

### Shared Helpers

`gstools_backend(use_core, num_threads)` temporarily forces GSTools to use
either the Cython fallback backend or the Rust `gstools_core` backend, and
sets `gstools.config.NUM_THREADS` for that benchmark run.

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

### Baseline Benchmark

The baseline benchmark is the first result set to create before doing any
optimization work. It uses the default ASV configuration, so each workflow is
measured with `threads_1` for both `cython_fallback` and `rust_core`.

#### Current Commit Baseline

- Save a baseline for the current commit:

```bash
asv run HEAD^! --bench benchmark_backends
```

#### Several Commits Baseline

As mentioned previously, ASV can also compare several commits, here we will run the last five commits:

- Run the last five commits on main branch:

```bash
asv run HEAD~5..HEAD --bench benchmark_backends
```

#### Summary of Results

After running ASV, inspect the explicit Rust-vs-Cython speedup ratios:

```bash
python benchmarks/tools/asv_speedup_summary.py
```

The helper reads `.asv/results/` and reports ratios per case and thread label:

```text
speedup = cython_fallback_time / rust_core_time
```

Interpret the ratio as:

- `speedup > 1.0` means Rust is faster
- `speedup = 1.0` means similar performance
- `speedup < 1.0` means Rust is slower

The speedup helper prints the backend ratio explicitly in the terminal. By
default, the helper skips removed legacy duplicate rows from older saved
results.

#### Visualization of Results

You can inspect the results in the ASV browser report by building and opening
the local website:

```bash
asv publish
asv preview
```

Then open the printed local URL,  for example:

```text
http://127.0.0.1:8082/#/
```
(or any other `http://127.0.0.1:<port>/#/` URL shown by the running preview).

The browser report shows ASV plots and trends. ASV plot views do not draw a line/graph when there is only one x-axis point, therefore running `asv run HEAD^! --bench benchmark_backends` will most likely not load any graphs.

For the default benchmark run, the `threads` column should show `threads_1`.
If you later run the
[optional OpenMP scaling experiment](#optional-parallelisation-with-openmp),
the same column can be used to compare several threads.


### Profiling With cProfile

`cProfile` does not update the ASV results shown in the browser report.
Instead, it prints a table in the terminal showing which Python
functions consumed time while one workflow ran.

The helper script is:

```text
benchmarks/tools/profile_benchmark_workflows.py
```

It imports the ASV benchmark classes from `benchmark_backends.py`, selects one
case, forces one backend, and runs that case under `cProfile`.

Since ASV has already created an isolated Python environment, select that
environment to execute the profiling helper:

```bash
ASV_ENV="$(ls -td .asv/env/* | head -n 1)"
ASV_PYTHON="$ASV_ENV/bin/python"
```

The helper still profiles the current checkout because
`profile_benchmark_workflows.py` adds the repository `src/` directory to
`sys.path`. The ASV environment provides the installed dependencies, including
`gstools-cython` and `gstools_core`.

List available cases:

```bash
"$ASV_PYTHON" benchmarks/tools/profile_benchmark_workflows.py --list
```

Possible profile selected cases:

```bash
"$ASV_PYTHON" benchmarks/tools/profile_benchmark_workflows.py --case variogram-sampled --backend rust_core --threads threads_1 --limit 10
"$ASV_PYTHON" benchmarks/tools/profile_benchmark_workflows.py --case variogram-extra-large --backend rust_core --threads threads_1 --limit 10
"$ASV_PYTHON" benchmarks/tools/profile_benchmark_workflows.py --case krige-large --backend rust_core --threads threads_1 --limit 10
"$ASV_PYTHON" benchmarks/tools/profile_benchmark_workflows.py --case krige-extra-large --backend rust_core --threads threads_1 --limit 10
"$ASV_PYTHON" benchmarks/tools/profile_benchmark_workflows.py --case condsrf --backend rust_core --threads threads_1 --limit 10
```

## Optional Parallelisation with OpenMP

This section collects optional workflows for testing Cython and Rust with
several thread counts. OpenMP setup is platform-dependent, so each operating
system should have its own tested instructions.

The default setup above remains the recommended baseline: one thread, normal
ASV environment, and no extra OpenMP build steps. Use this section only when
you explicitly want to measure backend scaling with multiple thread counts.

### Shared OpenMP Rule

The benchmark code can be run with several thread labels by setting for example 
`GSTOOLS_BENCHMARK_THREADS=1,2,4,8,16`. That only passes different
`gstools.config.NUM_THREADS` values to GSTools. It does not, by itself, make
the Cython backend parallel.

For Cython OpenMP scaling, the Cython extension must be compiled with OpenMP
support inside the same ASV environment that runs the benchmark. Always verify
that environment before interpreting Cython scaling results:

```bash
ASV_ENV="$(ls -td .asv-openmp/env/* | head -n 1)"
"$ASV_ENV/bin/python" benchmarks/tools/check_cython_openmp.py --fail-if-no-openmp
```

If the check fails, the benchmark may still run, but the Cython backend should
not be interpreted as an OpenMP-enabled Cython run.

### macOS Example

This is the currently tested OpenMP workflow. It is separate from the
default setup above.

The default ASV configuration, `asv.conf.json`, stays conservative: it is the
one-thread baseline and uses the normal conda-forge `gstools-cython` package.
The default `.asv/env/` environment does not provide Cython OpenMP support. That is why this section uses a second ASV configuration:

```text
asv.macos-openmp.conf.json
```

This OpenMP config creates separate generated directories:

```text
.asv-openmp/env/
.asv-openmp/results/
.asv-openmp/html/
```

That keeps the OpenMP experiment separate from the default `.asv/` baseline.

#### What The macOS OpenMP Config Does

`asv.macos-openmp.conf.json` asks conda to install the build/runtime pieces
needed for the macOS OpenMP experiment:

```text
llvm-openmp
cython
extension-helpers
setuptools
wheel
```

During ASV installation, it runs:

```bash
benchmarks/tools/install_macos_openmp_cython.py
```

That helper compiles `gstools-cython` from source inside ASV's own environment,
not inside your active conda environment. This matters because ASV benchmarks
the packages installed under `.asv-openmp/env/`.

Internally, the helper sets:

```text
GSTOOLS_BUILD_PARALLEL=1
CC=<ASV OpenMP env>/bin/gstools-asv-clang-openmp
CXX=<ASV OpenMP env>/bin/gstools-asv-clang-openmp++
```

The wrapper translates the plain `-fopenmp` flag used by the Cython build into
Apple-clang-compatible compiler and linker arguments that use conda's
`llvm-openmp`.

#### Run On macOS

In the previous section, the default config gives a quick overview for both
backends with `threads_1`. In this section, the OpenMP config runs several
thread labels: `threads_1`, `threads_2`, `threads_4`, `threads_8`, and
`threads_16`.

Start from the GSTools repository root:

```bash
cd /path/to/GSTools
```

Create a clean driver environment. This environment only runs ASV; ASV will
create the real benchmark environment under `.asv-openmp/env/`.

```bash
conda create -n gstools-benchmark -c conda-forge python=3.12 asv
conda activate gstools-benchmark
```

Create the ASV machine profile once:

```bash
asv --config asv.macos-openmp.conf.json machine --yes
```

Run a quick current-commit OpenMP check. This builds the OpenMP-enabled
`gstools-cython` package inside `.asv-openmp/env/` and runs the benchmark suite:

```bash
GSTOOLS_BENCHMARK_THREADS=1,2,4,8,16 \
asv --config asv.macos-openmp.conf.json run HEAD^! --quick --bench benchmark_backends --show-stderr
```

Verify that the ASV OpenMP environment really uses Cython OpenMP:

```bash
ASV_OPENMP_ENV="$(ls -td .asv-openmp/env/* | head -n 1)"
"$ASV_OPENMP_ENV/bin/python" benchmarks/tools/check_cython_openmp.py --verbose
"$ASV_OPENMP_ENV/bin/python" benchmarks/tools/check_cython_openmp.py --fail-if-no-openmp
```

Expected result on the tested Mac M2 setup:

```text
variogram default None -> 10
field default None -> 10
krige default None -> 10
OpenMP check: PASS
```

If that check passes, run the last-five-commits OpenMP benchmark:

```bash
GSTOOLS_BENCHMARK_THREADS=1,2,4,8,16 \
asv --config asv.macos-openmp.conf.json run HEAD~5..HEAD --bench benchmark_backends --show-stderr
```

Print Rust-vs-Cython ratios from the OpenMP result folder:

```bash
python benchmarks/tools/asv_speedup_summary.py --results-dir .asv-openmp/results
```

Build and preview the OpenMP browser report:

```bash
asv --config asv.macos-openmp.conf.json publish
asv --config asv.macos-openmp.conf.json preview
```

#### Interpreting The macOS OpenMP Run

- Use default `asv.conf.json` for the reproducible one-thread baseline.
- Use `asv.macos-openmp.conf.json` for the macOS OpenMP experiment.
- Only claim Cython OpenMP scaling if `check_cython_openmp.py` passes inside
  `.asv-openmp/env/...`.
- The active `gstools-benchmark` conda environment does not need `gstools`
  installed. It only needs ASV. The benchmarked GSTools packages live inside
  `.asv-openmp/env/...`.

This workflow is intended for macOS systems that use Apple clang with conda's
`llvm-openmp`. It should be portable across many macOS machines, including
Apple Silicon and Intel Macs, but it is not guaranteed for every macOS setup.

It is not guaranteed to run without local changes on:

- older macOS versions
- systems missing Xcode command-line tools
- systems with a nonstandard compiler setup
- HPC or managed macOS environments
- unusual conda installations

Do not assume this exact OpenMP setup applies to Linux, Windows, or HPC systems.

### Windows Example

### Linux Example

### HPC Example

### Profiling With cProfile for Multiple Threads

To profile how a workflow changes across configured thread counts, run the
same cProfile case several times with the OpenMP ASV environment:

```bash
ASV_OPENMP_ENV="$(ls -td .asv-openmp/env/* | head -n 1)"
ASV_OPENMP_PYTHON="$ASV_OPENMP_ENV/bin/python"

for threads in threads_1 threads_2 threads_4 threads_8 threads_16; do
  "$ASV_OPENMP_PYTHON" benchmarks/tools/profile_benchmark_workflows.py --case krige-extra-large --backend rust_core --threads "$threads" --limit 10
done
```

Useful options:

- `--case`: choose one workflow, or use `all`
- `--backend`: choose `cython_fallback` or `rust_core`
- `--threads`: choose `threads_1`, `threads_2`, `threads_4`, `threads_8`,
  or `threads_16`
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
