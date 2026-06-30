r"""
Continuous variables and distance metrics
-----------------------------------------

Direct Sampling is not limited to categorical facies: with ``categorical=False``
<<<<<<< HEAD
it simulates continuous variables (permeability, porosity, elevation, among other continuous properties).
The distance metric then selects how two continuous neighborhoods are compared:

* ``"l1"``: Manhattan distance on the raw values.
* ``"l2"``: Euclidean distance on the raw values.
* ``"variation"``: compares relative variations, so similar local shapes can
  match even when the local mean shifts.

We keep ``threshold=0.0`` for all three runs. This is the DSBC setting: DS
scans the requested fraction of the training image and takes the best candidate
found. That keeps this first continuous example focused on the metric choice
instead of threshold tuning.

No conditioning data are used here. Example ``01`` already introduced hard
conditioning, so this page isolates one new concept: how continuous data events
are compared. Example ``03`` combines the workflow with conditioning again on a
real training image.
=======
it simulates continuous variables (permeability, porosity, elevation, ...). The
``distance`` argument then selects how two patterns are compared:

* ``"l1"`` / ``"l2"`` — Manhattan / Euclidean distance on the raw values
  (Mariethoz et al., 2010, Eq. 6 / Eq. 4).
* ``"variation"`` — compares only the *relative* variations within a pattern
  (Eq. 9), tolerating a locally varying mean. Useful for non-stationary data.

Here we use a smooth continuous training image and a small acceptance
``threshold`` (standard DS mode) to allow approximate matches.

.. note::

    The acceptance ``threshold`` is **not comparable across metrics**: the
    ``"variation"`` distance is normalised by ``2·d_max``, so the same value is
    stricter than it would be for ``"l1"``/``"l2"``. Re-tune it when you switch
    the distance metric.
>>>>>>> n0228a/mps-direct-sampling
"""

import matplotlib.pyplot as plt
import numpy as np

import gstools as gs

###############################################################################
<<<<<<< HEAD
# A smooth, continuous synthetic training image with connected high-value bands.
=======
# A smooth, continuous synthetic training image.
>>>>>>> n0228a/mps-direct-sampling

gx, gy = np.meshgrid(np.arange(60), np.arange(60), indexing="ij")
ti_data = np.sin(gx / 6.0) * np.cos(gy / 8.0)

###############################################################################
<<<<<<< HEAD
# Simulate the same grid three times, changing only the distance metric.

grid = [np.arange(32, dtype=float), np.arange(32, dtype=float)]
metrics = [
    ("l1", "Manhattan (L1)"),
    ("l2", "Euclidean (L2)"),
    ("variation", "Variation"),
]
fields = {}
for distance, label in metrics:
    ti = gs.TrainingImage(ti_data, categorical=False, distance=distance)
    ds = gs.DirectSampling(
        ti, n_neighbors=12, scan_fraction=0.3, threshold=0.0
    )
    fields[label] = ds(grid, seed=3)

###############################################################################
# Plot. A shared colour scale keeps the comparison visually fair.

arrays = [ti_data, *fields.values()]
vmin = min(arr.min() for arr in arrays)
vmax = max(arr.max() for arr in arrays)

fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
axes = axes.ravel()
im = axes[0].imshow(
    ti_data,
    cmap="RdBu_r",
    origin="lower",
    vmin=vmin,
    vmax=vmax,
)
axes[0].set_title("Training image")
for ax, (label, field) in zip(axes[1:], fields.items()):
    ax.imshow(field, cmap="RdBu_r", origin="lower", vmin=vmin, vmax=vmax)
    ax.set_title(label)
fig.colorbar(im, ax=axes, shrink=0.8)
=======
# Build a continuous training image with the Euclidean (``"l2"``) distance.

ti = gs.TrainingImage(ti_data, categorical=False, distance="l2", n_neighbors=12)
print(ti)

###############################################################################
# Simulate. ``threshold=0.03`` accepts the first pattern within that distance
# (standard DS), which is faster than the exhaustive best-candidate search for
# continuous variables.

ds = gs.DirectSampling(
    gs.MPSModel(ti, scan_fraction=0.3, threshold=0.03)
)
field = ds([np.arange(32, dtype=float)] * 2, seed=3)

###############################################################################
# Plot. The realization reproduces the smooth wavy structure of the TI without
# copying it. A shared colour scale makes the comparison fair.

vmin, vmax = ti_data.min(), ti_data.max()
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 5))
im = ax0.imshow(ti_data, cmap="RdBu_r", origin="lower", vmin=vmin, vmax=vmax)
ax0.set_title("Training image (continuous)")
ax1.imshow(field, cmap="RdBu_r", origin="lower", vmin=vmin, vmax=vmax)
ax1.set_title("DS realization")
fig.colorbar(im, ax=(ax0, ax1), shrink=0.7)
>>>>>>> n0228a/mps-direct-sampling
