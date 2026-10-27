# MPS migration implementation log

Working file for the migration of Niklas's Python MPS features into the existing
Rust backend. **Untracked / unstaged** — not a git checkpoint. Append dated
sessions; never rewrite history here.

## Environment & checkouts (recorded at session start)

| Role | Checkout | Branch | HEAD | Tree |
|---|---|---|---|---|
| Target (Python) | `GSTools-RUST-Backend` | `feature/mps-direct-sampling` | `272140985a367588178fa81054c4e27eb8da1295` | clean |
| Native (Core) | `GSTools-Core` | `ds-rust-backend` | `dd9f7e57372d7887a7c9db1903f8961b4acd198d` | untracked: `.DS_Store CLAUDE.md docs/ graphify-out/` |
| Reference (read-only) | `GSTools-niklas` | `mps-direct-sampling` | `293e0347533cd9de48d39978427649d76eb025db` | untracked: audit md |

- Env: `/Users/lealroja/miniconda3/envs/gstools` · Python 3.14.6 · pip 26.1.2 · pytest 9.1.0 · maturin 1.14.1 · cargo/rustc 1.97.0
- Core: PyO3 0.28 `abi3-py38` (stable ABI → loads on 3.14), numpy 0.28, ndarray 0.17 (+rayon), rayon 1.10. Release profile `panic="abort"`, `lto=true`, `codegen-units=1`.
- Scratch dir path recorded in `/tmp/gstools_mps_scratch_path.txt` (temporary; not archival).
- Core baseline wheel sha256: `fb428c2920feb04ef9ee477c9f96fea0c5f2a1dd01aa0020a0b1327edd2823c4` (built from untouched Core working tree).

## Baseline (Phase 0)

- Baseline Core (untouched, locally built, release) installed into the env; `mps_simulate` present.
- Target MPS suite (`test_mps.py`, `test_mps_core_required.py`, `test_mps_nonstationary_specs.py`, `test_mps_port_contract.py`, `test_mps_rust_backend.py`): **243 passed, 11 warnings** against the untouched pair. This is the green baseline.

## Key starting-state findings

- The Core `ds-rust-backend` is **well ahead of its own `CLAUDE.md`** (which still describes only 6 kernels + univariate scan). `lib.rs` already exposes a full multivariate `mps_simulate` engine:
  - per-variable arrays: `n_neighbors`, `max_radius`, `variable_weights`, `metric_kinds`, `has_nan`, `d_max`, `p_norm`
  - `lag_matrices` (3-D, nonstationary transforms), `offsets`, `u_start`/`u_fallback` (RNG inputs), `partial_boundary`
  - DAG scheduling already present (returns `level_count`, `max_ready_width`, `used_threads`, `collapsed_lag_count`, `strict_fallback_count`)
  - **`distance_power` and `cond_weight` are still SCALAR** → Phase 2 target (make per-variable).
  - **No `mps_api_version()`** export → Phase 2 target (capability/version gate).
- Target MPS Python layout is Rust-only: `_core.py direct_sampling.py distance.py model.py neighbors.py nonstationary.py simulate.py training_image.py` (no `scan.py`/`runner.py`/`data_event.py`/`zone.py`; those exist only in the reference).
- Reference adds `zone.py` (zones → Phase 6) + the Python engine `scan.py`/`runner.py`/`data_event.py` (behavioral reference only; NOT to be restored into the target).
- `gs.mps` exposes `TrainingImage/Variable/MPSModel/DirectSampling`; **no `Zone`** yet. Top-level `gs.TrainingImage/MPSModel/DirectSampling` all present in the target.

## Phase status

| Phase | Title | Status |
|---|---|---|
| 0 | Preflight, inventory, baselines, benchmark protection | **verified** |
| 1 | Correctness & native-boundary hardening | Python items done; native-boundary + entangled items deferred |
| 2 | Public API, Variable/TI helpers, per-variable native settings | **done** (capability gate, per-variable distance_power/cond_weight, penalty_matrix, replace(), window(), named-univariate, `_explicit` tracking [added in Phase 7]) |
| 3 | Model rotation/scale & neighbor geometry | **done** (MPSModel.rotation/scale now drive the sim via resolve_spec; absolute-scale + matrix_derotate; set_nonstationary/anis removed) |
| 4 | Categorical penalty kernels & engine integration | **done** (native `T[sim,ti]` penalty distance, ABI rev 3; Phase-2 guard removed) |
| 5 | TI scan order & RNG contract | **done** (scan_path sequential/random; node-local LCG sampler, ABI rev 4; serial==parallel verified) |
| 6 | Zoned native simulation | **done** — Zone API + validation + selector + native per-node multi-TI engine (ABI rev 5); no-zone path bit-identical |
| 7 | Serial cleanup passes, overrides, paths, masks | **done** — post-processing API + validators + native serial Gauss-Seidel cleanup pass (ABI rev 6); `post_processing=0` bit-identical |
| 8 | Deterministic parallel cleanup & measured cache reuse | **done** — cleanup pass runs over a position-oriented dependency DAG + level scheduler (ABI rev 7); serial==parallel bit-identical; one neighbour cache feeds DAG + scan |
| 9 | Integration, examples/docs, benchmarks, handoff | **done** — 4 new gallery examples (post-processing/zones/penalty/scan_path) + README; post-processing benchmark case; Core CLAUDE.md refreshed; handoff summary below |

## Session notes

### 2026-10-19 — session 1
- Phase 0 completed and verified (see above). No git writes; no branch/env changes.
- **Phase 1 triage:** ran the reference-API audit suite (`test_mps_audit_regressions.py`) against the target → 5 passed / 18 failed. Most failures are because the **target is an earlier fork than the reference** (293e034): its `Variable` has no `penalty_matrix` kwarg, no `cond_weight`, no `replace()`; `distance_power` is a scalar on `TrainingImage` (not per-variable); named-univariate TIs aren't usable; `post_processing_*` validation absent. Those are Phase 2/4/7 features, not pure Phase-1 defects.
- **Phase 1 slice implemented & verified (target `src/gstools/mps/training_image.py`):**
  - `Variable.__init__` now rejects ±inf float data (NaN remains the missing-value marker). [P1-8]
  - `Variable.__init__` now rejects non-finite / negative `weight` (`0 <= w < inf`). This also covers the negative-multivariate-weight finding. [P1-5]
  - `_parse_distance` now requires a finite exponent `0 < p < inf` for both `l<p>` and `variation<p>` (rejects `linf`/`lnan`), matching the reference.
  - `_data_range` deliberately **left unchanged** — the reference keeps the identical `max-min` form, so porting = matching accepted behavior; no gratuitous dtype change introduced.
  - Regressions: `tests/test_mps_migration_phase1.py` (17 tests) — all pass. Full MPS baseline re-run: **243 passed** (no regressions).
- **Phase 1 remaining** (next tasks, each needs direct_sampling.py/model.py/distance.py/neighbors.py or native work; several entangled with later-phase features): out-of-grid conditioning warning + simulation-coordinate finiteness (`direct_sampling.py.__call__`/`_snap_to_nearest`); transactional `set_condition` + canonical `(n_cond,)` shapes; `post_processing_factor` finite validation (only once `post_processing` exists in target — Phase 7); masked variation common-support mean-shift; native mixed finite/unlimited radii + zero-weight `0*inf` handling; comprehensive native-boundary validation of `mps_simulate`/`scan_node`.
- **Phase 1 slice 2 implemented & verified (target `src/gstools/mps/direct_sampling.py`):**
  - `_conditions_to_grid`: ported the reference's **out-of-grid snap warning** (points beyond half a cell outside the grid) and **collision warning** (finite values landing on an occupied (node, variable)). [P2 warnings]
  - `set_condition`: added module-level `_check_finite_positions` and call it in both the multivariate and univariate branches → **non-finite conditioning positions now raise** (`np.isfinite(pos).all()`). [sim-coordinate finiteness, conditioning side]
  - Regressions extended in `tests/test_mps_migration_phase1.py` → **22 tests pass**. Baseline re-run: **243 passed** (warnings 11→14: existing tests with colliding/edge conditioning now emit the reference-identical warnings; behavior is correct-by-construction, ported verbatim).
- **Phase 1 remaining** (re-scoped after reading target source): `post_processing_factor` finite-validation and masked-variation common-support are **entangled with features not in the target yet** (`post_processing`, per-variable overrides) → defer to Phase 2/7. `set_condition` canonical `(n_cond,)` shape rejection of `(N,2)` and native-boundary hardening (mixed finite/unlimited radii, zero-weight `0*inf`, raw-entry validation of `mps_simulate`/`scan_node`) still pending.
- `model.py` read: target **already has** `_validate_threshold` (finite/≥0/>1-warn), `_validate_scan_fraction` (0,1], and `_validate_rotation`/`_validate_scale` finiteness-at-construction — those Phase-1 items need no change. **No `post_processing*`** on the target `MPSModel` → that P1 item is unreachable until Phase 7. Noted divergence for Phase 3: `MPSModel.rotation/scale` (resolve_spec-based) coexists with `DirectSampling.set_nonstationary`/`_resolve_nonstationary_map`, and `__call__` currently uses the latter — must reconcile.

### 2026-10-19 — session 1 (Phase 2 start: capability gate)
- **Native (Core `src/lib.rs`):** added `#[pyfunction] mps_api_version() -> u32` (returns `1`) inside the declarative `#[pymodule]` block (auto-collected by PyO3 0.28). `cargo check --no-default-features` clean; release wheel rebuilt + installed (sha recorded by filename in scratch `core-wheel-p2a`).
- **Python (`src/gstools/mps/_core.py`):** added `mps_api_version` to `_REQUIRED_EXPORTS`, a `_REQUIRED_MPS_API = 1` ABI gate (rejects a same-versioned but older Core with an actionable "rebuild" message), and re-export. Verified: gate passes at api 1.
- Verified: import gate OK; **265 passed** (243 baseline + 22 Phase-1). Clean checkpoint — nothing half-edited.
- **Rule for the per-variable change (next task):** bump `mps_api_version_py` 1→2 **in lockstep** with `_REQUIRED_MPS_API` 1→2 when `mps_simulate`'s `distance_power`/`cond_weight` go scalar→per-variable, so a stale Core is rejected cleanly.

#### Phase 2 resume plan (per-variable `distance_power` + `cond_weight`, then full `Variable`)
Self-contained native+Python unit — do it all before re-verifying:
1. **Rust `mps_engine.rs`:** `simulate_engine` params `distance_power: f64, cond_weight: f64` → `ArrayView1<f64>` (per variable, indexed in TI-row order). `compute_node_weights` (called per-variable block at ~line 561) indexes the arrays by the current variable. Guard lengths == n_variables.
2. **Rust `lib.rs`:** `mps_simulate_py` `distance_power: f64, cond_weight: f64` → `PyReadonlyArray1<f64>`; `.to_owned()` before `detach`; bump `mps_api_version` → 2.
3. **Python `simulate.py` (~line 330):** pass per-variable arrays `[v.distance_power ...]` / `[v.cond_weight ...]` instead of the scalar TI-level / model-level values.
4. **Python `training_image.py` `Variable`:** add per-variable `distance_power` (finite ≥0) and `cond_weight` (finite ≥0) with the reference's validation; decide back-compat for the current TI-level `distance_power` (keep as the default feeding each Variable) and model-level `MPSModel.cond_weight` (keep feeding each Variable's default). Do **not** expose per-variable values the engine ignores — gate 1→3 above land together.
5. **`_core.py`:** `_REQUIRED_MPS_API` → 2. Rebuild Core, reinstall, re-verify baseline + add regressions proving non-uniform per-variable `distance_power`/`cond_weight` change the output.
Then the remaining Phase-2 API: `Variable.penalty_matrix` (store+validate now; distance use is Phase 4 — guard a sim that uses it until then), `replace()`, `TrainingImage.window()`, named-univariate TI usability, `_trusted` fast path, `_validate_n_neighbors`/`_check_category_codes` helpers.

### 2026-10-19 — session 1 (Phase 2: per-variable distance_power/cond_weight native ABI, rev 2)
Landed the native half of per-variable `distance_power`/`cond_weight`, behavior-preserving:
- **`GSTools-Core/src/mps_engine.rs`:** `simulate_engine` + `simulate_node` params `distance_power`/`cond_weight` `f64` → `ArrayView1<'_, f64>`; added length==`variable_count` validation; the per-variable loop indexes `distance_power[variable]`/`cond_weight[variable]` (`compute_node_weights` unchanged — it already works per-variable-block). Updated both `#[cfg(test)]` fixtures to length-1 arrays.
- **`GSTools-Core/src/lib.rs`:** `mps_simulate_py` `distance_power`/`cond_weight` → `PyReadonlyArray1<f64>` (owned via `.to_owned()` before `detach`, passed as `.view()`); **`mps_api_version` bumped 1 → 2** with revision note.
- **`GSTools-RUST-Backend/src/gstools/mps/_core.py`:** `_REQUIRED_MPS_API` 1 → 2.
- **`GSTools-RUST-Backend/src/gstools/mps/simulate.py`:** the `_mps_simulate_gsc` call now passes `np.full(n_vars, ti.distance_power)` and `np.full(n_vars, self.cond_weight)` (uniform broadcast of the former scalars → bit-identical).
- Verification: `cargo clippy --all-targets --no-default-features -D warnings` clean; `cargo test --no-default-features` **64 passed** (run with `env -u PYO3_BUILD_EXTENSION_MODULE DYLD_FALLBACK_LIBRARY_PATH=$GST_ENV/lib` — the PyO3 test-link fix from §10.3). Release wheel rebuilt+installed, `mps_api_version()==2`, gate passes. Full Python MPS suite **265 passed** (bit-identical, no regressions).
- ABI-rev-2 wheel sha256: `450afab3073606a0b5bdaf6136fe2e60c686a82e8e6ecc29bf01052d96f29bf7` (scratch `core-wheel-p2b`).
- **Note:** per-variable values are not yet user-reachable (no public `Variable.distance_power`/`cond_weight`), so nothing is silently ignored — the native path receives uniform arrays identical to the old scalars.
- **Next concrete task (reference-exact design, confirmed by reading `GSTools-niklas` training_image.py/model.py):** this is an API *restructure*, give it a focused pass:
  1. `Variable.__init__`: add `distance_power=0.0`, `cond_weight=1.0` (concrete; validate finite ≥0 — code already read at ref lines 319-330) + read-only `distance_power`/`cond_weight` properties.
  2. `TrainingImage.__init__`: add `distance_power=_SENTINEL`, `cond_weight=_SENTINEL`; in the **bare-array** sugar forward them to the anonymous `Variable`; for a `Variable` **list** they must NOT be passed (raise). Keep `TrainingImage.distance_power` property reading the (univariate) variable for back-compat.
  3. **`MPSModel`: remove `cond_weight`** (reference moved it onto `Variable`). This breaks any `MPSModel(..., cond_weight=)` / `DirectSampling.cond_weight` usage → update those call sites AND the affected baseline tests/examples. Grep target tests for `cond_weight` first and plan the edits.
  4. `simulate.py`: build `self.var_distance_power`/`self.var_cond_weight` dicts from each `Variable` (keyed in `self.variables` order) and feed them via `np.fromiter(...)` in place of the current `np.full(...)` broadcasts. No Core rebuild needed (ABI rev 2 already supports it).
  5. Regression: non-uniform per-variable `distance_power`/`cond_weight` must change the field vs uniform.
  Then the rest of Phase 2: `penalty_matrix` (store+validate; sim-use guarded until Phase 4), `replace()`, `TrainingImage.window()`, named-univariate TI usability, `_trusted` fast path, `_validate_n_neighbors`/`_check_category_codes` helpers.
- **State after capability-gate + native-ABI work: clean, fully verified.** Core ABI rev 2 built+installed; `cargo test` 64 passed; Python MPS suite 265 passed.

### 2026-10-19 — session 1 (Phase 2: per-variable distance_power/cond_weight PUBLIC API — reference-aligned)
Completed the API restructure moving `distance_power`/`cond_weight` onto `Variable` (no Core rebuild — ABI rev 2 already supports it):
- **`training_image.py`:** `Variable` gains `distance_power=0.0`, `cond_weight=1.0` (both validated finite ≥0) + read-only properties. `TrainingImage` takes them as `_SENTINEL` sugar: forwarded to the anonymous Variable for bare-array input; **rejected** (ValueError) when a `Variable`/list is passed (they belong on the Variable). TI-level `distance_power` property now reads the univariate variable (raises for MV). `vec_distance_var` uses `v.distance_power`.
- **`model.py`:** **removed `cond_weight`** entirely (param/storage/property/setter/repr) — matches reference `MPSModel`.
- **`direct_sampling.py`:** `cond_weight` property → `_collapse_per_var("cond_weight")` (scalar if all equal, else dict); dropped `cond_weight=` from the `ds_simulate` call; updated `set_condition` docstring.
- **`simulate.py`:** `_DirectSamplingEngine` no longer takes `cond_weight`; builds `self.var_distance_power`/`self.var_cond_weight` from each Variable and feeds them via `np.fromiter(...)` to the engine. `ds_simulate` lost its `cond_weight` param.
- **Call-site migration:** `tests/test_mps.py` — 7 sites moved `cond_weight`/`distance_power` from `MPSModel`/TI-list onto the TI sugar / per-Variable (incl. rewriting the `TestCondWeightOverride` class to the new design, not weakened). Examples `12_continuous_texture.py` and `13_3d_fluvsim_multifacies.py` migrated (`cond_weight` → TI sugar).
- **New regressions:** `tests/test_mps_migration_phase2.py` (13 tests) incl. Variable validation, MPSModel rejecting `cond_weight`, TI sugar forwarding + list rejection, the `ds.cond_weight` scalar/dict view, and **non-uniform per-variable `distance_power` and `cond_weight` demonstrably changing the field**.
- Verified: full MPS suite **278 passed**; examples `py_compile` OK; `ruff format --check` clean (reformatted `training_image.py`). Integrity: no staged files, branches/HEADs unchanged, reference untouched.
- Modified now: Core `lib.rs`/`mps_engine.rs`; target `_core.py`,`direct_sampling.py`,`model.py`,`simulate.py`,`training_image.py`,`tests/test_mps.py`,`examples/13_mps/12*`,`13*`. New: log + `tests/test_mps_migration_phase1.py` + `tests/test_mps_migration_phase2.py`.

#### Phase 2 remaining (next tasks)
- `Variable.penalty_matrix` (+ `_validate_penalty_matrix`, `_check_category_codes`): store & validate now; a simulation that *uses* a penalty matrix must fail explicitly until Phase 4 wires the native categorical-penalty kernel.
- `Variable.replace(**overrides)` (ref uses `_trusted=True` to reuse the frozen buffer) and the `_trusted` fast path in `Variable.__init__`.
- `TrainingImage.window(...)` with range/missingness recompute.
- Named-univariate TI usability (`gs.mps.TrainingImage(Variable("f", ...))` end-to-end) — was one of the triage failures.
- `_validate_n_neighbors` shared helper (ref factors the variation-≥2 guard out).
- Reference `MPSModel` is also keyword-only (`*`) with `threshold=None` DSBC default — decide whether to adopt here (separate from this unit).

### 2026-10-19 — session 1 (Phase 2: penalty_matrix, replace(), window(), named-univariate) — DONE
Pure-Python (no Core rebuild; ABI rev 2 unchanged):
- **`training_image.py`:** module helpers `_validate_n_neighbors` + `_check_category_codes`; `_view_variable` + `_REPLACE_FIELDS`. `Variable` now takes `penalty_matrix=None` and `_trusted=False`; `_trusted` skips the defensive copy (for `replace()`/`window()` sharing the frozen buffer). Added `_validate_penalty_matrix` (2-D square, zero diagonal, off-diag ∈ (0,1] NaN-rejected, asymmetric OK, data category-code screening), the `penalty_matrix` property, and `Variable.replace(**overrides)` (shares buffer, rejects data/categorical/unknown fields). `n_neighbors` setter now delegates to the helper. `TrainingImage.window(slices)` + `_validate_window_slices` (unit-step/in-bounds/non-degenerate; view, recomputed d_max; uni + MV).
- **`simulate.py`:** `_DirectSamplingEngine` raises `NotImplementedError` if any Variable has a `penalty_matrix` (validated but not yet applied by the native engine — Phase 4). Named-univariate fix: `self.weights` keys by the actual single-variable name (not hardcoded `None`).
- **`direct_sampling.py`:** named-univariate fix — univariate conditions keyed by the real variable name, and the univariate return uses `result[<name>]` (both reduce to `None` for the anonymous bare-array sugar → bit-identical).
- **New regressions** (`tests/test_mps_migration_phase2.py`, now 31 total): penalty_matrix validation (non-square/diagonal/off-diag/NaN/continuous/bad-codes), frozen storage, **sim-use guard**, `replace()` (buffer sharing + field rules + penalty preservation), `window()` (view + recomputed range + MV + bad-slice rejection), and **named-univariate end-to-end run**.
- Verified: full MPS suite **297 passed**; `ruff format --check src/gstools/mps/` clean. Integrity: nothing staged, branches/HEADs unchanged, reference untouched.
- **Deferred to their phases:** `_record_explicit`/`_explicit` field tracking (only used by Phase-7 post-processing override merge); adopting reference `MPSModel` keyword-only + `threshold=None` DSBC default (API-cosmetic, decide with Phase 7); `max_radius` grid-cell semantics (`1 ≤ r < inf`) + resolve-time rotation/scale re-validation (Phase 3); native categorical-penalty kernel (Phase 4).
- **Next:** Phase 3 — reconcile the two nonstationary paths (`MPSModel.rotation/scale` vs `DirectSampling.set_nonstationary`/`_resolve_nonstationary_map`), match GSTools angle conventions (`matrix_derotate`, scalar-3D first-angle padding, reject 1-D rotation), absolute-scale vs relative-anis semantics, and the neighbour-offset envelope.

### 2026-10-19 — session 1 (Phase 3: nonstationary reconciliation) — DONE
Pure-Python (ABI rev 2 unchanged; the Rust engine just consumes the per-node `lag_matrices`, agnostic to how M is built). Resolved the "two nonstationary paths" divergence: `MPSModel.rotation/scale` now **drive** the simulation; the parallel `DirectSampling.set_nonstationary`/`anis` path is removed.
- **`neighbors.py`:** `_lag_transform_matrix(dim, rotation_map, scale_map, x_i)` rewritten to the reference form `diag(1/s(x_i)) @ matrix_derotate(dim, θ(x_i))` over canonical `grid_shape+(n_comp,)` maps (uniform `map[idx]` indexing, no ndim heuristics). Imports `matrix_derotate, no_of_angles` (was `matrix_isometrize, set_angles, set_anis`).
- **`direct_sampling.py`:** `__call__` resolves `MPSModel.rotation` (`resolve_spec(..., angles=True)`) and `scale` (`resolve_spec(..., positive=True)`) to canonical maps using **meshgridded flat grid coords** (`np.meshgrid(*self.pos, indexing="ij")` → one value per node, the callable contract). Passes `scale_map` to `ds_simulate`. Removed `_resolve_nonstationary_map`, `set_nonstationary`, `self._rotation`/`self._anis`; `__repr__` reads `MPSModel.rotation/scale`. Added `resolve_spec` import.
- **`simulate.py`:** `anis_map` → `scale_map` throughout (`_DirectSamplingEngine` param/attr, the `_lag_transform_matrix` call + the `is not None` gate, `ds_simulate` param/docstring).
- **Convention:** absolute per-axis `scale` replaces relative `anis`; the old `anis=a` maps to `scale=[1.0, a, …]` (main axis 1, transversal = ratio) — the exact absolute-scale analogue of the former `set_anis`-based isometrize. Rotation keeps the `set_angles` scalar-3D padding (`angles=True`); 1-D rotation rejected by `MPSModel._validate_rotation`. Non-callable arrays re-validated at resolve time against the grid; callables materialized + finiteness-checked.
- **Call-site migration:** `TestNonstationarity` (test_mps.py) rewritten to `MPSModel(rotation=, scale=)` (`test_anis_changes_output`→`test_scale_changes_output`, `test_combined_rotation_anis_runs`→`..._scale_runs`); `test_mps_rust_backend.py` nonstationary spy test; examples `06_channel_strebelle_nonstat.py` + `07_channel_strebelle_radial.py` (`anis` map → `scale=stack([ones, anis], -1)`).
- **New regressions** `tests/test_mps_migration_phase3.py` (7): scalar/stationary-vector/callable rotation agree end-to-end; model rotation & callable scale actually change output vs plain; identity (rotation=0, scale=1) bit-identical to plain; 1-D rotation rejected; non-positive scale rejected; 3-D scalar rotation == `[θ,0,0]` padded vector.
- Verified: full MPS suite **304 passed**; examples `py_compile` OK; `ruff format --check src/gstools/mps/` clean. Integrity: nothing staged, branches/HEADs unchanged, reference untouched. No leftover `set_nonstationary` anywhere.
- **Deferred:** `max_radius` grid-cell semantics (`1 ≤ r < inf`, Phase 3-geometry refinement) and the neighbour-offset envelope per-axis cap were **not** reworked this pass (current `_precompute_offsets` + `max_radius>0` behavior retained, baseline-green); revisit if a mixed finite/unlimited-radius case needs it. `_warn_nonuniform_axes` (reference grid-spacing warning) not ported.
- **Next:** Phase 4 — native categorical penalty-matrix kernel (remove the Phase-2 `NotImplementedError` guard once the engine applies `T[sim_cat, ti_cat]`).

### 2026-10-19 — session 1 (Phase 4: native categorical penalty kernel, ABI rev 3) — DONE
Implemented the categorical penalty distance `T[sim_code, ti_code]` end to end.
- **`GSTools-Core/src/mps.rs`:** `METRIC_CATEGORICAL_PENALTY = 3`; `penalty_cost(penalty, offset, ncat, sim, ti)` (out-of-range codes → full-mismatch 1.0, no panic); `categorical_penalty_candidate_distance` (masked + unmasked, same renormalization as plain categorical). `scan_node` takes `penalty_flat`/`penalty_offsets`/`penalty_ncat` (active-indexed offsets/ncat; whole flat buffer), with a `METRIC_CATEGORICAL_PENALTY` branch. Added 3 Rust unit tests (hand-computed 0.15 case; asymmetric-orientation `penalty_cost`; all-off-diag-1 ≡ plain mismatch).
- **`mps_engine.rs`:** `simulate_node` + `simulate_engine` thread the 3 arrays; `simulate_node` builds `active_penalty_offsets`/`active_penalty_ncat`; `simulate_engine` validates `penalty_offsets/ncat` length == variable_count. Both `#[cfg(test)]` fixtures updated (empty penalty, ncat=[0]).
- **`lib.rs`:** `mps_simulate_py` + `mps_scan_node_py` gain the 3 `PyReadonlyArray1` args (owned before `detach`); **`mps_api_version` 2 → 3**. All 7 `scan_node` test call-sites + both engine fixtures updated.
- **`_core.py`:** `_REQUIRED_MPS_API` 2 → 3.
- **`simulate.py`:** removed the Phase-2 `NotImplementedError` guard; `self.var_penalty` dict; `metric_kinds` emits `3` for a categorical variable with a penalty_matrix; packs `penalty_flat` (concatenated row-major CxC) + `penalty_offsets` + `penalty_ncat` and passes them to the engine.
- Verification: `cargo clippy --all-targets` clean; `cargo test` **67 passed**; release wheel rebuilt+installed (`api==3`, sha `9635b66d…`, scratch `core-wheel-p4`); new `tests/test_mps_migration_phase4.py` (5) — all-ones≡plain (bit-identical), penalty-changes-output, asymmetric≠transpose, masked-TI runs, multivariate-penalty runs; Phase-2 guard test converted to "penalty simulation runs". Full MPS suite **309 passed**. `ruff`/`cargo fmt` clean. Integrity: nothing staged, branches/HEADs unchanged, reference untouched.
- **Note:** the legacy non-engine helper `TrainingImage.vec_distance_var`/`_dispatch_metric` (block-distance kernels) does **not** apply penalties — only the production `scan_node` engine path does. No production path uses that helper for penalties.
- **Next:** Phase 5 — TI scan order & RNG contract (random `scan_path` sampling-without-replacement, deterministic node-local; preserve the sequential default; `max(1, min(win, int(f·ti_size)))` budget).

### 2026-10-19 — session 1 (Phase 5: scan_path & RNG contract, ABI rev 4) — DONE
Added a `scan_path="sequential"` (default) / `"random"` option; random scan visits `max_scan` DISTINCT window cells in a node-local random order.
- **`GSTools-Core/src/mps.rs`:** `random_scan_positions(win_size, scan_seed, count)` — full-period LCG over the next-pow2 modulus (Knuth MMIX multiplier, odd increment → Hull–Dobell full period); skips out-of-range draws; **O(count) memory (the scan budget, not a window permutation)**; distinct guaranteed for `count ≤ win_size`. `scan_node` gains `random_scan: bool, scan_seed: u64`; the scan loop selects random positions vs the sequential wrap-around walk (`#[allow(needless_range_loop)]` — index drives both branches). Rust unit test `random_scan_positions_are_distinct_and_in_range` (permutation coverage, partial budget, determinism, seed-sensitivity).
- **`mps_engine.rs`:** `simulate_node` derives `scan_seed = (u_start[node]·2^53) as u64` (reference's derivation) and threads `random_scan`+`scan_seed` to `scan_node`; `simulate_engine` threads `random_scan`. Node-local seed ⇒ identical serial/parallel output. Both fixtures updated.
- **`lib.rs`:** `mps_simulate_py` gains `random_scan: bool`; `mps_scan_node_py` gains `random_scan`/`scan_seed`; **`mps_api_version` 3 → 4**. All 7 `scan_node` test call-sites updated (`false, 0`).
- **`_core.py`:** `_REQUIRED_MPS_API` 3 → 4.
- **`model.py`:** `scan_path="sequential"` param + `_validate_scan_path` ({sequential,random}) + property/setter + docstring + repr default.
- **`direct_sampling.py` / `simulate.py`:** `__call__` passes `scan_path=model.scan_path`; `ds_simulate`/`_DirectSamplingEngine` take `scan_path`; engine sets `random_scan = scan_path=="random"` and passes it to the native call.
- **New regressions** `tests/test_mps_migration_phase5.py` (7): scan_path validation + default; random scan runs; **reproducible for a fixed seed**; random ≠ sequential; **random serial==parallel bit-identical at 576 nodes (>512 cutoff)**; sequential serial==parallel guard.
- Verified: clippy clean, `cargo fmt` clean, **cargo test 68 passed**; wheel rebuilt+installed (`api==4`, sha `71bc4e71…`, scratch `core-wheel-p5`); full MPS suite **316 passed** (sequential default bit-identical → no regression). Integrity: nothing staged, branches/HEADs unchanged, reference untouched.
- **Parity note:** the native random-scan RNG (LCG) is NOT the reference's `numpy default_rng(...).choice(replace=False)` stream — exact Python/Rust realization identity is *not* promised for random scan (documented); the mandatory invariant (deterministic, node-local, serial==parallel) holds and is tested. Sequential mode's stream is unchanged.
- **Next:** Phase 6 — zoned simulation (`Zone`/`build_zone_selector`, multiple TIs/domains selected inside Rust).

### 2026-10-19 — session 1 (Phase 6: Zone API + validation + selector; native engine GUARDED) — PARTIAL
Delivered the full public zonation surface + validation (reference-exact), with the native multi-TI run explicitly guarded (the engine's single-TI assumption is the largest remaining refactor).
- **New `src/gstools/mps/zone.py`:** `Zone(ti, where)` — bool-mask or callable region; rejects non-boolean arrays (no silent `!=0`). Exported as `gs.mps.Zone` (not top-level, matching the reference).
- **`nonstationary.py`:** ported `resolve_where` (callable/array → grid-shape bool mask, n_nodes or grid_shape accepted) and `build_zone_selector` (folds zones into an intp selector; `0`=primary, `i+1`=zones[i]; **rejects overlapping regions**). Added to `__all__`; refreshed the stale "zonation is a later phase" docstring.
- **`model.py`:** `_validate_zones` (zone TIs must share the primary's ndim + variable set + per-variable categorical/continuous kind; penalty-matrix variables also get `_check_category_codes` against the primary's C). `MPSModel(zones=None)` param + `zones` property/setter + docstring.
- **`direct_sampling.py` `__call__`:** when `model.zones` is set, calls `build_zone_selector(...)` (so mask shape/dtype/**overlap** errors surface against the actual grid) then raises `NotImplementedError` — zoned simulation is validated but not yet run natively.
- **`mps/__init__.py`:** export `Zone`.
- **New regressions** `tests/test_mps_migration_phase6.py` (9): Zone validation; `resolve_where` callable/array + non-bool rejection; `build_zone_selector` labels + overlap rejection; `_validate_zones` kind/dim mismatch; zones default `[]` + round-trip; **guarded run raises `NotImplementedError` after validating masks**; overlap reported at call time.
- Verified: full MPS suite **325 passed** (no Core change → no rebuild, ABI still 4); `ruff` clean; `gs.mps.Zone` importable. Integrity: nothing staged, branches/HEADs unchanged, reference untouched.

#### Phase 6 native resume plan (multi-TI engine — the big one)
Additive, backward-compatible design (no-zone path stays bit-identical):
1. **`lib.rs`/`mps_engine.rs` (ABI rev 5):** keep primary `ti`/`ti_shape`/`d_max`; ADD `zone_ti: (n_variables, Σ zone_sizes)`, `zone_offsets (n_zones+1)`, `zone_shapes (n_zones, dim)`, `zone_d_max (n_zones, n_variables)`, and `node_domain (n_path,)` (0 = primary, k = zones[k-1]). `simulate_engine` precomputes `finite_all`/`ti_has_nan` **per domain** (Vec-of-Vec / Vec<bool>); `simulate_node` selects its domain's `ti` view + `ti_shape` + `d_max` + `finite_all` + `ti_has_nan` at the top (everything else — n_neighbors, metrics, weights, penalty, transforms — stays primary). All other per-node logic (window, lags, scan, penalty, fallback) is unchanged once those are rebound. `node_domain` all-zero ⇒ identical to today.
2. **`simulate.py`:** build the selector via `build_zone_selector`, map it to `node_domain` along `self.path`; pack `zone_ti`/offsets/shapes/`zone_d_max` from each `Zone.ti` (variable order = `self.variables`); pass to the engine; remove the `NotImplementedError` guard in `direct_sampling.py`.
3. **Tests:** no-zone equivalence (bit-identical); simple disjoint zones pick the right TI (e.g. a zone whose TI is all-constant forces its region to that value); differently-shaped zone TIs; cross-boundary data events; penalty/continuous-range per domain; serial==parallel with zones.
- **Deferred within Phase 6 too:** `TrainingImage.window`-based zone TIs already work as inputs (Phase 2 shipped `window()`); the unconditional joint-vector fallback must read the SELECTED domain's `finite_all` (covered by step 1).

### 2026-10-20 — session 1 (Phase 6 NATIVE: per-node multi-TI engine, ABI rev 5) — DONE
Implemented the native multi-TI engine exactly per the additive plan above; the no-zone path is provably bit-identical.
- **`GSTools-Core/src/mps_engine.rs`:** `simulate_engine` gains `zone_ti`/`zone_offsets`/`zone_shapes`/`zone_d_max`/`node_domain`; validates their shapes + `node_domain ∈ [0, n_zones]` + per-zone offset/shape-product consistency. Builds per-domain views (`domain_tis`/`domain_shapes`/`domain_d_max`) + per-domain `finite_all`/`ti_has_nan` (domain 0 = primary, k = zones[k-1]). The per-node closure selects `domain = node_domain[node]` and passes that domain's TI view/shape/d_max/finite_all/has_nan to **`simulate_node` — whose signature is UNCHANGED**. Neighbour gathering (`build_neighbor_cache`) is grid-based and untouched, so data events cross zone boundaries. Imported `s!`. Both test fixtures take empty zone args.
- **`lib.rs`:** `mps_simulate_py` gains the 5 zone args (owned before `detach`); **`mps_api_version` 4 → 5**.
- **`_core.py`:** `_REQUIRED_MPS_API` 4 → 5.
- **`simulate.py`:** `ds_simulate`/`_DirectSamplingEngine` take `zones`/`zone_selector`; `_run_rust_engine` builds `node_domain` (`zone_selector[tuple(path.T)]`, else zeros) and packs `zone_ti` (zone TIs concatenated column-wise in `self.variables` order), `zone_offsets`, `zone_shapes`, `zone_d_max` (None→1.0); passes all to the native call.
- **`direct_sampling.py`:** `__call__` resolves `build_zone_selector(...)` (overlap/shape/dtype errors surface against the real grid) and threads `zones`/`zone_selector` to `ds_simulate` — the Phase-6 `NotImplementedError` guard is removed.
- **Regressions** `tests/test_mps_migration_phase6.py` now 12: Zone/selector/validation (as before) **plus** native: zone-TI selected per region (primary all-0 vs zone all-1 → exact 0/1 split), differently-shaped zone TI, **no-zone (`zones=[]`) bit-identical to plain**, zoned serial==parallel at 576 nodes, overlap reported at call time.
- Verified: clippy + `cargo fmt` clean; `cargo test` 68 passed; wheel rebuilt+installed (`api==5`, sha `30034c85…`, scratch `core-wheel-p6`); full MPS suite **328 passed**. Integrity: nothing staged, branches/HEADs unchanged, reference untouched.
- **Next:** Phase 7 — serial cleanup (post-processing) passes: `post_processing`/`post_processing_factor`/`post_processing_variables`/`post_processing_path`/`post_processing_mask`, Gauss-Seidel reopen-and-resimulate with hard-data preservation (requires an initial-vs-cleanup pass mode in the engine).

### 2026-10-20 — session 1 (Phase 7: post-processing API + validation; native cleanup GUARDED) — PARTIAL
Delivered the full post-processing public surface + validation (reference-exact) and closed the Phase-2 `_explicit` gap; native cleanup execution is guarded (it needs a genuinely new Gauss-Seidel reopen-and-resimulate pass mode in the engine — the most algorithmically subtle phase).
- **`training_image.py`:** added `import functools` + `_record_explicit` decorator (sets `Variable._explicit = kwargs ∩ _REPLACE_FIELDS`), applied to `Variable.__init__`; ported `_resolve_post_ti` (builds the post-pass TI by `primary.replace(**explicit overrides)`; multivariate weight all-or-none check) and `_post_pass_n_neighbors` (`round(n/p_f)` floored at 1; rejects variation < 2).
- **`model.py`:** `_VALID_POST_PATH` + validators `_validate_post_processing`, `_validate_post_processing_factor`, `_validate_post_processing_path`, `_validate_post_processing_variables` (overrides must keep primary data+kind; penalty codes checked vs zones), `_validate_post_pass_effort`, `_validate_post_processing_mask`. Added `post_processing`/`post_processing_factor`/`post_processing_path`/`post_processing_variables`/`post_processing_mask` params + properties + docstrings; effort checked at construction.
- **`direct_sampling.py` `__call__`:** raises `NotImplementedError` when `post_processing > 0` (validated but not run natively).
- **New regressions** `tests/test_mps_migration_phase7.py` (16): `_explicit` tracking; all validators (negative p, bad factor, path modes/array, non-bool mask, override data/kind match, replace-override accepted, variation<2 effort); `_resolve_post_ti` applies only explicit overrides; `_post_pass_n_neighbors` round/floor; defaults; guarded run.
- Verified: full MPS suite **344 passed** (no Core change → no rebuild, ABI still 5); `ruff` clean. Integrity: nothing staged, branches/HEADs unchanged, reference untouched.

### 2026-10-20 — session 2 (Phase 7 NATIVE: serial Gauss-Seidel cleanup pass, ABI rev 6) — DONE
Implemented the native cleanup pass exactly per the resume plan below; the `post_processing=0` path is provably bit-identical (zero cleanup calls, zero extra RNG draws).
- **`GSTools-Core/src/mps_engine.rs`:** new `cleanup_pass_engine` — one serial pass that, per path node, erases the node's non-conditioned variables (`fields[(v,flat)]=NaN`) so `simulate_node` picks them up as targets with the stale value suppressed, re-simulates against the live field, and writes back immediately (Gauss-Seidel). New `build_cleanup_neighbor_cache` gathers ALL in-bounds nearest-`n_k` neighbours per variable (fully-informed, no path-order dependency, `level=0`). Reuses `simulate_node` UNCHANGED, so the variation mean-shift paste rule follows the effective `metric_kinds` automatically (variation↔Lp override flips it). Extracted a shared `build_domains` helper (per-domain TI view/shape/d_max/finite_all/has_nan) and refactored `simulate_engine` to use it (pure extraction; used `slice_move`/`index_axis_move` to keep the `'a` lifetime). Added `Axis` import. 2 new Rust unit tests (hand-computed `[0,1,1]` Gauss-Seidel fixed point + determinism).
- **`lib.rs`:** `mps_cleanup_pass` wrapper (owns all arrays before `detach`, returns `(fields, strict_fallback_count, collapsed_lag_count)`); `mps_api_version` 5 → 6 with revision note.
- **`_core.py`:** added `mps_cleanup_pass` to `_REQUIRED_EXPORTS` + re-export; `_REQUIRED_MPS_API` 5 → 6.
- **`simulate.py`:** `_DirectSamplingEngine` takes the 5 `post_processing*` params, stores `_rng_path`/`_rng_nodes`, resolves `_post_ti`/`_post_path`/`_post_max_radius_per_var`/mask, fails the effort check early, and **extends the shared offset table to cover post radii only when `post_processing>0`** (farther candidates the main path never reaches → pp=0 table unchanged). Extracted `_lag_matrices_for(path)`/`_node_domain_for(path)`/`_pack_zones()` from `_run_rust_engine` (reused by main + cleanup). `run()` now calls `_run_post_passes()` after the main pass. `_run_post_passes` builds `resim` (≥1 non-cond var, `&= mask`), divides `n_neighbors`/`scan_fraction` by `p_f`, and loops `post_processing` times (order via `_build_path`/`"same"` filter; fresh `u_start`/`u_fallback` per pass — RNG draw order matches the reference). `_run_cleanup_pass` + `_cleanup_var_arrays` pack the **effective** post-pass per-variable config from `_post_ti` and call the native engine, writing results back into `self.sg`. `ds_simulate` gained the 5 params.
- **`direct_sampling.py`:** removed the Phase-7 `NotImplementedError` guard; threads the 5 `post_processing*` params from `MPSModel` into `ds_simulate`.
- **Regressions** `tests/test_mps_migration_phase7.py` now 22 (was 16; replaced the guarded test): **oracle equivalence** (serial cleanup == composition of single-node DSBC full-scan resimulations through the same engine, start-independent; TI smaller than sim so the pass genuinely moves the field), `post_processing=0` bit-identical (`assert_array_equal`), reproducibility, **hard data unchanged**, **mask freezes non-selected cells** (`fm[~mask]==f0[~mask]`, masked cells change), **both metric-override directions** (l1↔variation change the cleaned field).
- Verified: clippy + `cargo fmt` clean; `cargo test` **70 passed** (+2 cleanup); wheel rebuilt+installed (`api==6`, sha `d6a9ee5b…`); full MPS suite **350 passed**; `ruff format --check`/`ruff check` clean. Integrity: nothing staged, branches/HEADs unchanged (Core `ds-rust-backend` dd9f7e57, Target `feature/mps-direct-sampling` 272140985), reference `GSTools-niklas` untouched.
- **Deferred to Phase 8:** deterministic PARALLEL cleanup (the fully-informed neighbourhood is cyclic, so the level scheduler can't encode it; this pass is serial) + measured cross-pass neighbour-cache reuse.

### 2026-10-20 — session 2 (Phase 8: deterministic PARALLEL cleanup, ABI rev 7) — DONE
Made the cleanup pass parallel and thread-count invariant. The earlier "cyclic neighbourhood" worry was the wrong framing: like the reference's post-pass DAG, orienting every eligible-node neighbour edge by **visit position** (low→high) makes the visit order a topological order, so the EXISTING level scheduler applies directly — predecessors (earlier position) commit first → read NEW; successors (later position) commit later → read OLD; adjacent nodes land in different levels so within-level nodes never read each other. That reproduces serial Gauss-Seidel at any thread count.
- **`mps_engine.rs`:** `simulate_node` gained a `cleanup: bool` — in cleanup mode a variable is a target iff `conditioned==0` (its stale value suppressed, no NaN erase needed) and only hard variables are collocated; the main pass keeps the NaN rule (`cleanup=false`, bit-identical). New `build_cleanup_levels` derives the position-oriented symmetric-closure DAG from the neighbour cache (duplicate lower-neighbour sightings are harmless for the level max) and returns per-level node groups. `cleanup_pass_engine` rewritten: builds the cleanup cache ONCE (feeds both the DAG and the per-node scan — within-pass cache reuse), level-schedules exactly like `simulate_engine` (same `>=512` small-path cutoff + one Rayon pool), reads a per-level snapshot and commits after each level. No more NaN-erase / immediate-writeback. Now takes `num_threads` and returns `used_threads` too. 1 new Rust unit test (`cleanup_levels_order_mutual_neighbours_by_visit_position`); the hand-computed serial fixture still gives `[0,1,1]` (level-order == path-order for the Gauss-Seidel DAG).
- **`lib.rs`:** `mps_cleanup_pass` gained `num_threads` + a 4th return (`used_threads`); `mps_api_version` 6 → 7.
- **`_core.py`:** `_REQUIRED_MPS_API` 6 → 7.
- **`simulate.py`:** `run()` threads `n_threads` into `_run_post_passes` → `_run_cleanup_pass` → native; unpacks the 4-tuple, stores `self._cleanup_used_threads`, and fires a new module hook `_MPS_CLEANUP_STATS_HOOK({"used_threads": ...})` per pass (lets tests confirm the parallel branch ran). No behavioural change at any fixed thread count.
- **Regressions** `tests/test_mps_migration_phase8.py` (4): cleanup **serial == parallel** bit-identical on a 576-node (>512) grid with `post_processing=2` (asserts the hook saw `[1,1,4,4]` so parallel genuinely engaged), parallel reproducibility, serial==parallel under a **random** post path, and serial==parallel **with zones** (`[1,4]`). Phase-7's serial oracle + all its tests stay green (level-order serial == path-order serial).
- Verified: clippy + `cargo fmt` clean; `cargo test` **71 passed** (+1); wheel rebuilt+installed (`api==7`, sha `710a5132…`); full MPS suite **354 passed**; `ruff format --check`/`ruff check` clean on the touched files (`simulate.py`, `test_mps_migration_phase8.py`). Integrity: nothing staged, branches/HEADs unchanged (Core `ds-rust-backend` dd9f7e57, Target `feature/mps-direct-sampling` 272140985), reference `GSTools-niklas` untouched.
- **Note on "measured cache reuse":** the neighbour cache is now built once per pass and reused by BOTH the DAG build and the scan (the within-pass reuse). CROSS-pass reuse (the reference's `nb_cache`) is not available because Python drives the pass loop with one native call per pass; it would require moving the pass loop into Rust and is deferred as a pure-perf optimisation (correctness/determinism are complete). **Next:** Phase 9 — integration, examples/docs, benchmarks, handoff.

### 2026-10-20 — session 2 (Phase 9: integration, examples/docs, benchmarks, handoff) — DONE
Closed out the migration. No engine/ABI change this phase (Core stays ABI rev 7); all work is examples, docs, and a benchmark case.
- **Examples** (`examples/13_mps/`, 4 new, each runs headless under `MPLBACKEND=Agg` and carries assertions): `15_post_processing_cleanup.py` (raw vs 3 cleanup passes on the Strebelle TI; isolated-pixel count 7→1), `16_zoned_simulation.py` (primary channel TI + its transpose as a top-half `gs.mps.Zone`), `17_categorical_penalty.py` (synthetic 3-facies ordinal TI; ordinal-penalty matrix drops 0↔2 contacts 22→2), `18_scan_path.py` (sequential vs random scan, reproducibility asserted). `README.rst` gained a paragraph introducing the four algorithm-control examples.
- **Benchmark** (`benchmarks/benchmark_mps.py`): added DS case `cat_dsbc_cleanup` (categorical 60×60 TI, 30×30 SG, `post_processing=3`); `MPSModel(...)` in `setup()` now reads `spec.get("post_processing", 0)` (existing cases unchanged → default 0). Smoke test via the ASV setup_cache/setup/time flow: baseline `cat_dsbc_medium` ≈ 11.9 ms, `cat_dsbc_cleanup` ≈ 38.6 ms (≈ 3 extra passes). `tests/test_benchmark_suite.py` still 9 passed.
- **Docs/handoff:** refreshed the stale Core `CLAUDE.md` (it still described only the 6 univariate block kernels) — now documents `src/mps_engine.rs`, the full multivariate engine, penalties/zones/nonstationary/cleanup, the versioned `mps_api_version()` ABI gate (rev 7), and the real export set (pinned in `_core.py::_REQUIRED_EXPORTS`).
- Verified: 4 examples run (exit 0, assertions hold); benchmark case runs; full MPS suite + `cargo test`/clippy/fmt + ruff all green (see the handoff summary). Integrity: nothing staged, branches/HEADs unchanged, reference `GSTools-niklas` untouched.

## Migration complete — handoff summary

All phases 0–9 done. The Rust backend now owns the complete Niklas MPS feature set; the no-feature path (no zones, no penalty, no rotation/scale, `post_processing=0`, sequential scan) is bit-identical to the Phase-0 baseline.

- **Native ABI history** (`mps_api_version()` gated against `_core.py::_REQUIRED_MPS_API`): rev 1 version gate · rev 2 per-variable `distance_power`/`cond_weight` · rev 3 categorical penalty kernel · rev 4 `scan_path` random · rev 5 zoned multi-TI · rev 6 serial cleanup pass · **rev 7** deterministic parallel cleanup (+`num_threads`). Current installed wheel sha `710a5132…`.
- **Feature → phase → tests:** Variable/TI helpers + per-variable settings (P2, `test_mps_migration_phase2.py`), rotation/scale nonstationary (P3, `…phase3`), categorical penalty matrix (P4, `…phase4`), scan_path + RNG contract (P5, `…phase5`), zoned simulation (P6, `…phase6`), post-processing cleanup API + serial native (P7, `…phase7`), deterministic parallel cleanup (P8, `…phase8`). Phase-1 correctness hardening in `…phase1`.
- **Rebuild recipe:** from `GSTools-Core`, `env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV <venv>/maturin build --release`, then `pip install --no-index --no-deps --force-reinstall <wheel>`. Rust tests: `env -u PYO3_BUILD_EXTENSION_MODULE DYLD_FALLBACK_LIBRARY_PATH=$CONDA_PREFIX/lib cargo test --no-default-features`. Python tests: `PYTHONPATH=<RUST-Backend>/src MPLBACKEND=Agg pytest`.
- **Still local/unstaged** (the user owns all git writes): Core `lib.rs`/`mps.rs`/`mps_engine.rs` + `CLAUDE.md`; Target `src/gstools/mps/*`, `examples/13_mps/0*–18*`, `benchmarks/benchmark_mps.py`, `tests/test_mps*.py`, this log.
- **Remaining (optional, out of the original scope):** cross-pass neighbour-cache reuse (pure perf, needs the cleanup pass loop moved into Rust); the Phase-1 deferred native-boundary hardening items noted in that session.

## Pre-release cleanup checklist (migration scaffolding to remove)

Things added *only* to support the staged migration. They are safe to remove /
consolidate before cutting the final "GSTools DS Rust backend" release. Each
line says what to do and why. **None of these are features** — the feature code
(`mps_simulate`, `mps_cleanup_pass`, the engine, the public API, examples, the
benchmark case) stays.

1. **ABI-revision integer gate — remove, replace with proper versioning.**
   - `src/gstools/mps/_core.py`: delete `_REQUIRED_MPS_API`, the
     `_found_api = int(_gstools_core.mps_api_version())` + `if _found_api <
     _REQUIRED_MPS_API: raise ImportError(...)` block, the `"mps_api_version"`
     entry in `_REQUIRED_EXPORTS`, and the `mps_api_version = ...` re-export.
   - Core `src/lib.rs`: delete the `mps_api_version_py` `#[pyfunction]` (+ its
     revision-history doc comment).
   - **Why:** it only exists because the crate version stayed `1.4.0` across
     every ABI change during development, so the version string couldn't detect
     a stale hand-built Core. At release, bump the `gstools-core` crate version
     for the DS-backend release and have GSTools require `gstools-core>=<that>`;
     the existing `_MIN_VERSION`/`_REQUIRED_EXPORTS` *presence* check then covers
     the rest. **Keep `mps_cleanup_pass` (and the other kernels) in
     `_REQUIRED_EXPORTS`** — that is the presence check, unrelated to the gate.
   - (Optional, if softening instead of deleting: keep `mps_api_version()` as a
     plain diagnostic and drop only the hard `raise`.)

2. **`tests/test_mps_migration_phase{1..8}.py` — consolidate, do not just delete.**
   These lock in real behaviour (validators, penalty kernel, zones, the cleanup
   oracle, serial==parallel). Fold that coverage into the permanent suite
   (`test_mps.py` or per-feature modules) and drop the phase-scoped framing.
   Some assert *private* helpers (`_resolve_post_ti`, `_post_pass_n_neighbors`,
   `Variable._explicit`, the stats hook); keep the behavioural assertions, and
   drop/replace the white-box ones if those internals are refactored.

3. **Test-only introspection seams added by the migration — remove if their test
   is reworked.**
   - `src/gstools/mps/simulate.py`: `_MPS_CLEANUP_STATS_HOOK` (module global +
     the `if _MPS_CLEANUP_STATS_HOOK is not None:` call) and
     `self._cleanup_used_threads`, both added in Phase 8 only so
     `test_mps_migration_phase8.py` could confirm the parallel branch engaged.
   - The 4th return value `used_threads` of `mps_cleanup_pass`
     (`lib.rs` wrapper + `cleanup_pass_engine` + the `_used` unpack in
     `simulate.py`) exists mainly for that assertion and for symmetry with
     `mps_simulate`. Dropping it back to a 3-tuple is an ABI change — do it in
     the same pass as removing the gate (item 1) if at all.
   - **Note:** `_MPS_RUST_ENGINE_STATS_HOOK` in `simulate.py` **predates** this
     migration (used by `test_mps_rust_backend.py`); leave it unless you are
     cleaning that up separately.

4. **`MPS_MIGRATION_IMPLEMENTATION_LOG.md` (this file) — do not ship.** It is an
   untracked working record. Delete it, or move its handoff summary into an
   internal dev doc; it is not user documentation.

5. **Migration-historical inline comments — optional doc tidy.** Comments that
   narrate the migration rather than the code (`"native ABI rev N"`,
   `"Phase-2 guard"`, references to `RUST_ONLY_MIGRATION.md`) appear in
   `_core.py`, `simulate.py`, `training_image.py`, `distance.py`. Rewrite them to
   describe current behaviour, and make sure any doc they cite
   (`RUST_ONLY_MIGRATION.md`) exists at release or drop the reference.

6. **Ephemeral dev artifacts (nothing in the repo to remove).**
   `/tmp/gstools_mps_scratch_path.txt`, the scratch dir, and the recorded wheel
   sha256 values are temporary; they are not tracked and need no action.

**Separate, pre-existing simplification (NOT migration scaffolding, out of
scope):** the standalone Python exports `mps_dist_block_*`, `mps_scan_node`,
`mps_scan_node_cat` and `TrainingImage.vec_distance_var` / `_dispatch_metric`
are exercised only by unit tests (`test_mps.py`, `test_mps_rust_backend.py`,
`tests/_mps_distance_ref.py`), not by the production engine path (which does all
distance/scan work inside `mps_simulate`). Collapsing that legacy block-distance
layer is a possible future tidy, but it predates this migration and is not
required by it — flagged here only so it is not mistaken for feature code.

#### Phase 7 native resume plan (Gauss-Seidel cleanup — the subtle one)
1. **`mps_engine.rs` (ABI rev 6):** add a cleanup-pass mode. Inputs: the COMPLETE field (not NaN), the immutable hard-data mask, an update-eligibility mask (node×variable), and a cleanup visit path. A cleanup neighbour cache gathers ALL in-bounds neighbours (every one is informed) up to the effective `n_neighbors` — no initial-pass dependency filtering. Visit the path **serially**, and for each eligible node: (a) suppress the node's own current value as a self-constraint (don't include it in its own event — it already isn't), (b) build the event from the live field (earlier nodes already updated → Gauss-Seidel), (c) scan the effective post-pass TI with the effective metric/weights, (d) **write back immediately** so later nodes see it, (e) never overwrite hard-conditioned variables. Parallel cleanup is Phase 8.
2. **`simulate.py`:** per-pass effective config (`_resolve_post_ti` → effective `n_neighbors` via `_post_pass_n_neighbors`, effective `scan_fraction = f/p_f` capped, metric/weight overrides); resolve `post_processing_path` (inherit/same/random/sequential/explicit); resolve `post_processing_mask` → eligibility; loop `post_processing` times, one native cleanup call per pass; `post_processing=0`/empty overrides = true no-op (no extra RNG draws). Remove the guard.
3. **Metric-override paste rule:** variation→L1 must turn OFF the mean-shift paste adjustment and vice-versa (distance AND value-adjustment both come from the effective pass config).
4. **Tests:** serial cleanup vs a small controlled oracle; hard data unchanged; mask freezes only intended updates; both metric-override directions; `post_processing=0` ≡ no cleanup (bit-identical); seed continuation.
