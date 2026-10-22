.. _mps-contracts:

Direct Sampling controls and reproducibility
=================================================

Direct Sampling (DS) uses the GSTools-Core Rust engine. GSTools installs Core
as a required dependency; DS additionally checks the native MPS interface and
required exports before running. A package version alone does not identify the
source of a development wheel. Record both repository revisions and the wheel
hash when comparing development builds.

Construct training images and zones with ``gs.mps.TrainingImage`` and
``gs.mps.Zone``. ``gs.MPSModel`` and ``gs.DirectSampling`` are available at the
top level. No top-level ``gs.Zone`` alias is needed.

Candidate selection
-------------------

``threshold=0.0`` is the default best-candidate (DSBC) mode. ``None``, integer
zero, and negative zero are equivalent inputs. DS returns the lowest-distance
selectable candidate examined within the scan budget; it can stop at an exact
zero distance because no smaller nonnegative distance exists. This does not
require an exact match and does not imply a full-TI search. Ties retain the first
minimum encountered.

For a finite positive threshold, the first selectable candidate with distance
**strictly below** the threshold is accepted. If none qualifies, DS uses the
best visited candidate. Negative and nonfinite thresholds are rejected. Values
above one retain the warning policy for normalized distances. Hard conditioning
is preserved independently of the threshold in the raw DS field.

For a nonempty valid candidate window, the budget is
``max(1, min(window_size, floor(scan_fraction * selected_TI_size)))``. The
selected TI is the zone TI at a zoned node. Cleanup uses the effective scan
fraction after division by ``post_processing_factor``. The optimum is over the
examined selectable candidates, not all TI locations. Empty events and searches
without selectable candidates use the engine's unconditional fallback; these
are distinct from malformed native input, which raises a validation error.
With missing TI values, fallback draws from centers finite in all variables.

The controls address different orders and budgets:

* ``path`` sets the simulation-grid visit order.
* ``scan_path`` sets the TI candidate order at each node. Sequential scanning
  wraps raster order from a node-local random start; random scanning samples
  distinct positions without replacement.
* ``post_processing_path`` sets the cleanup revisit order.
* ``scan_fraction`` sets the candidate budget relative to TI size, capped by
  the valid window.
* ``threshold`` sets approximate-match early acceptance.

Coordinates and transformations
-------------------------------

Neighborhood offsets and ``max_radius`` use **grid-index units**. Physical axes
are used for coordinates, conditioning-point snapping, and coordinate-dependent
callables. Unequal axis spacing and nonuniform axes do not turn the neighbor
search into a physical-distance search. Use finite axes and account explicitly
for index-space geometry when choosing radii and transformations.

Rotation angles are in **radians**, in GSTools angle ordering. Scale factors
are positive and dimensionless, one per axis. They transform simulation-grid
lags into TI lags; they do not rotate or resample the TI data. The inverse map
is ``M = diag(1 / scale) @ matrix_derotate(dim, angles)``; row-vector lags use
``lags_ti = lags_sg @ M.T`` before integer-grid rounding. For example, in 2D,
``rotation=np.pi/2`` with ``scale=[2, 1]`` applies a quarter-turn and a factor-two
stretch of simulated structures along the first local axis relative to the TI.
Exact identity settings preserve the untransformed path.

Distances and penalties
-----------------------

Variation distance compares deviations from local event means. Both means use
the same finite comparison support. For each supported pair, the centered
residual is divided by ``2*d_max``; the weighted Lp aggregate is clipped at one.
Masked comparisons renormalize weights on their valid support. The pasted
center is shifted by the difference of the event means on that same support.
Simulated values can therefore lie outside the original TI range. Requested
variation neighborhoods need at least two points, but the realized common
support can be smaller.

Thresholds apply to this implemented normalization. Dividing a centered
residual by ``2*d_max`` rather than ``d_max`` reduces its distance and makes a
fixed positive threshold easier to satisfy before clipping. Thresholds are not
automatically transferable between those normalizations or between variation
and absolute-value metrics.

For categorical variables, a penalty matrix replaces plain mismatch costs with
``T[simulation_category, TI_category]``. Asymmetric costs are supported, the
diagonal is zero, and off-diagonal entries lie in ``(0, 1]``. These costs compare
corresponding event values. They do not prohibit adjacent facies pairs or
directly impose a pairwise transition energy on the simulated grid. Spatial
transition statistics can change indirectly and should be measured over an
ensemble.

Cleanup and zones
------------------

``MPSModel(post_processing=p)`` requests DS resimulation passes. Each pass uses
the current field, including updates earlier in that pass. In contrast,
``ds(..., post_process=False)`` disables the Field mean/normalizer/trend
pipeline; it does not disable DS cleanup. Assess hard-data invariants on the
raw DS field: a later configured Field transformation can change displayed
numeric values.

Cleanup revisits nodes with at least one unconditioned variable, intersected
with the exact-grid-shaped ``post_processing_mask`` when supplied. Conditioned
components of a partially conditioned node remain fixed. Cleanup inherits the
primary variable configuration unless ``post_processing_variables`` overrides
it. Its requested neighbor counts use ``max(1, round(n/post_processing_factor))``
(Python's tie-to-even rounding); requested variation counts must remain at
least two. The effective scan fraction is divided by the same factor, with the
candidate budget still capped by the window. ``post_processing_path=None``
uses sequential order for a sequential main path and random order otherwise;
``"same"`` reuses the eligible main order, while
``"random"``, ``"sequential"``, or an explicit path select another order.
Zero cleanup passes consume no cleanup random draws. Cleanup can alter
connectivity and distributions as well as isolated pixels; improvement of any
one measure is not guaranteed.

Zones select TI data, shape, value ranges, and missingness by domain. Search
settings come from the effective primary or cleanup configuration, not the
zone TI's unused search settings. Events can include neighbors across zone
boundaries; this permits interaction without guaranteeing a seamless boundary.
For multivariate simulation, an unadjusted collocated vector copied without
retained partial conditioning comes from one TI location. That property does
not guarantee exact finite-realization marginal frequencies, connectivity, or
multipoint distributions.

Random streams and threads
--------------------------

DS uses the opt-in ``SeedSequence``/``MT19937`` contract
``ds-seedsequence-mt19937-v1``. SeedSequences spawn child streams without
compressing them to the former 16-bit seed space. This intentionally changes
seeded DS realizations. Default seeding for other GSTools features is unchanged.
NumPy spawning provides a very high probability of independent streams, not a
proof that every finite output differs for every seed.

Constructor ``seed=None`` or default NaN creates an entropy-seeded master.
At call time, an omitted ``seed`` or NaN continues the master sequence; an
integer resets it reproducibly; ``seed=None`` creates a fresh entropy-seeded
master. Omitted, NaN, or None ``path_seed`` and ``node_seed`` derive their streams
from the master; explicit integers select repeatable role streams independent
of unrelated master changes. Equal explicit role seeds can initialize equal
streams. A fixed path stream fixes path controls; changing node order can still
change the final field even when the node stream is fixed.

Thread selection uses call-time ``num_threads``, then instance
``num_threads``, then ``gstools.config.NUM_THREADS``. The current unset/falsy
fallback is **one thread**, not all cores. Paths shorter than 512 nodes run
serially; larger dependency graphs may still offer little useful concurrency.
Within a fixed supported build and dependency environment, identical seed
controls and inputs are intended to produce identical fields at different DS
thread counts. This does not promise bitwise identity across operating systems,
compiler/dependency versions, or algorithm releases. Record NumPy, source
revisions, stream contract, controls, scan mode, and effective threads with
reproducibility results.

Scientific checks
------------------

A plausible plot or passing deterministic tests does not establish ensemble
quality. Validate the statistics needed for the application, for example
facies proportions, face-connected connectivity, spatial correlations, and
cross-variable distributions. Use realizations as the independent units for
Monte Carlo uncertainty, rather than treating spatially dependent pixels as
independent samples. Variation with shifted conditioning needs centered or
detrended targets, rather than raw TI marginals alone. Example contact counts
and cleanup effects describe their particular realizations.

Numeric input limits
---------------------

The DS engine stores values as float64. Integer-typed inputs are checked for
exact float64 representability before packing. Integers in ``[-2**53, 2**53]``
are representable; some larger values, such as powers of two, are also exact.
Other large labels or values raise an actionable error: recode categorical
labels to small integers, or center/rescale continuous data before conversion.
For example, ``2**60`` is exact but ``2**60 + 3`` is not. This check cannot
recover distinctions already lost in an input converted to floating point.

``n_neighbors`` and ``post_processing`` accept integer types without converting
them through float. Integral floats and booleans remain accepted for
compatibility; fractional values, strings, and nonfinite numbers are rejected.
Neighbor counts must be positive; cleanup pass counts may be zero.

For a rotation-only comparison with one categorical variable and several facies,
see :ref:`sphx_glr_examples_13_mps_07_herten_nonstationary.py`.
